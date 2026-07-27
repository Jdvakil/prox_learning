#!/usr/bin/env python3
"""Label-collision, nearest-neighbour ambiguity and local-identifiability audit.

Handoff steps 6-8. This is the decisive test. Ensemble disagreement can flag the historical
failures, but if two frames with the same deployable observation carry opposite oracle
labels, then *no* deterministic function of that observation classifies both correctly, and
no amount of uncertainty modelling over the same inputs changes that.

Exclusions matter more than the search here. A neighbour from the same trajectory, the same
episode under a different driving policy, or the same underlying frame stored twice is not
evidence of ambiguity -- it is the same observation counted twice. All three are removed,
the last by comparing the scientific state hash.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from causal_parked_skin import engine
from causal_parked_skin import threshold as thr
from causal_parked_skin.data import SOURCE_MODES, load_partition
from causal_parked_skin.engine import load_checkpoint, make_batch

ALL_PARTITIONS = ("reference_train", "reference_validation", "reference_calibration",
                  "offline_reference_test")
REFERENCE_POOL = "reference_train"
ONSET_MIN_FRAMES = 10
ONSET_FRACTION = 0.10
K_VALUES = (1, 4, 8, 16, 32)

# predeclared near-identity tolerances (handoff step 7)
TOLERANCE_CLOSENESS = 1e-5
TOLERANCE_QPOS = 1e-6
TOLERANCE_QVEL = 1e-6
TOLERANCE_ACTION = 1e-6

# high-ambiguity band (handoff step 8)
AMBIGUOUS_LOW = 0.2
AMBIGUOUS_HIGH = 0.8

FEATURE_SPACES = ("A_CURRENT_PROX_RAW", "B_CURRENT_PROX_EMBEDDING",
                  "C_FULL_DEPLOYABLE_INPUT", "D_FROZEN_MODEL_EMBEDDING")


def onset_cutoff(length: int) -> int:
    return max(ONSET_MIN_FRAMES, int(np.ceil(ONSET_FRACTION * length)))


def build_index(cache: Path, model, device):
    """Every frame in the paired dataset with its identity and four feature vectors."""
    import torch

    meta: list[dict] = []
    raw, prox_embedding, full_input, model_embedding = [], [], [], []
    captured: dict = {}

    def hook(_module, _inputs, output):
        captured["prox"] = output.detach()

    handle = model.to_token.register_forward_hook(hook)

    for name in ALL_PARTITIONS:
        partition = load_partition(cache, name)
        trajectory_index = np.asarray(partition["trajectory"])
        oracle_active = np.asarray(partition["oracle_active"]).astype(bool)
        hazard = np.asarray(partition["hazard_present"]).astype(bool)
        modes = np.asarray(partition["source_mode"])
        current = np.asarray(partition["current"])
        valid = np.asarray(partition["current_valid"])
        state = np.asarray(partition["state"])

        for index in range(trajectory_index.max() + 1):
            rows = np.flatnonzero(trajectory_index == index)
            length = len(rows)
            cutoff = onset_cutoff(length)
            for offset, row in enumerate(rows):
                meta.append({
                    "partition": name,
                    "trajectory_id": partition.trajectory_ids[index],
                    "episode_id": partition.episode_ids[index],
                    "distribution": SOURCE_MODES[int(modes[row])],
                    "hazard_present": bool(hazard[row]),
                    "step": offset, "trajectory_length": length,
                    "progress_fraction": offset / max(length - 1, 1),
                    "onset": offset < cutoff,
                    "oracle_active": bool(oracle_active[row]),
                })
            flat_current = current[rows].reshape(length, -1)
            flat_valid = valid[rows].reshape(length, -1).astype(np.float32)
            raw.append(np.concatenate([flat_current, flat_valid], axis=1))
            full_input.append(np.concatenate([flat_current, state[rows]], axis=1))

            for start in range(0, length, 256):
                chunk = rows[start:start + 256]
                batch = make_batch(partition, chunk, device)
                with torch.no_grad():
                    out = model(batch["history"], batch["history_valid"], batch["state"])
                prox_embedding.append(
                    captured["prox"].reshape(len(chunk), -1).float().cpu().numpy())
                model_embedding.append(out["mask_logits"].reshape(
                    len(chunk), -1).float().cpu().numpy())
    handle.remove()

    features = {
        "A_CURRENT_PROX_RAW": np.concatenate(raw, axis=0),
        "B_CURRENT_PROX_EMBEDDING": np.concatenate(prox_embedding, axis=0),
        "C_FULL_DEPLOYABLE_INPUT": np.concatenate(full_input, axis=0),
        "D_FROZEN_MODEL_EMBEDDING": np.concatenate(model_embedding, axis=0),
    }
    return meta, features


def scientific_hash(closeness, state) -> str:
    import hashlib

    digest = hashlib.sha256()
    for array in (np.ascontiguousarray(closeness, dtype=np.float32),
                  np.ascontiguousarray(state, dtype=np.float32)):
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def neighbour_search(query_features, pool_features, query_meta, pool_meta,
                     pool_rows, device, k_max=32, chunk=64):
    """Nearest neighbours with same-trajectory / same-episode / duplicate exclusion."""
    import torch

    pool = torch.from_numpy(pool_features).to(device, torch.float32)
    pool_norm = (pool ** 2).sum(dim=1)
    episodes = np.array([m["episode_id"] for m in pool_meta])
    trajectories = np.array([m["trajectory_id"] for m in pool_meta])
    labels = np.array([m["oracle_active"] for m in pool_meta])
    hashes = np.array([m["scientific_hash"] for m in pool_meta])

    results = []
    for start in range(0, len(query_features), chunk):
        block = torch.from_numpy(
            query_features[start:start + chunk]).to(device, torch.float32)
        distance = (torch.cdist(block, pool) ** 2)
        for offset in range(block.shape[0]):
            record = query_meta[start + offset]
            excluded = ((episodes == record["episode_id"])
                        | (trajectories == record["trajectory_id"])
                        | (hashes == record["scientific_hash"]))
            row = distance[offset].clone()
            row[torch.from_numpy(excluded).to(device)] = float("inf")
            order = torch.argsort(row)[:k_max].cpu().numpy()
            valid = [i for i in order if np.isfinite(float(row[i]))]
            results.append({
                "neighbour_rows": [int(pool_rows[i]) for i in valid],
                "neighbour_labels": [bool(labels[i]) for i in valid],
                "neighbour_distances": [float(np.sqrt(float(row[i]))) for i in valid],
                "excluded_count": int(excluded.sum()),
            })
        del distance
    del pool, pool_norm
    torch.cuda.empty_cache()
    return results


def summarise_neighbours(query_meta, neighbours) -> dict:
    out = {}
    for k in K_VALUES:
        entropies, opposite_fractions, nearest_opposite = [], [], []
        opposite_over_same, nearest_is_opposite = [], []
        for record, block in zip(query_meta, neighbours):
            labels = np.array(block["neighbour_labels"][:k])
            distances = np.array(block["neighbour_distances"][:k])
            if labels.size == 0:
                continue
            same = labels == record["oracle_active"]
            fraction = float((~same).mean())
            opposite_fractions.append(fraction)
            p = np.clip(fraction, 1e-12, 1 - 1e-12)
            entropies.append(float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))))
            nearest_is_opposite.append(bool(not same[0]))
            if (~same).any():
                nearest_opposite.append(float(distances[~same].min()))
                if same.any():
                    opposite_over_same.append(
                        float(distances[~same].min() / max(distances[same].min(), 1e-12)))
        out[f"k={k}"] = {
            "queries": len(opposite_fractions),
            "median_label_entropy": float(np.median(entropies)) if entropies else None,
            "mean_opposite_label_fraction": float(np.mean(opposite_fractions))
            if opposite_fractions else None,
            "median_nearest_opposite_distance": float(np.median(nearest_opposite))
            if nearest_opposite else None,
            "median_opposite_over_same_distance_ratio": float(
                np.median(opposite_over_same)) if opposite_over_same else None,
            "fraction_nearest_neighbour_opposite_label": float(
                np.mean(nearest_is_opposite)) if nearest_is_opposite else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--onset-audit", required=True, type=Path)
    ap.add_argument("--per-group", type=int, default=250)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch

    stack = json.loads(args.stack.read_text())
    engine.set_sensor_names(stack["sensor_contract"]["ordered_names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, payload = load_checkpoint(args.checkpoint, device)

    meta, features = build_index(args.cache, model, device)
    total = len(meta)
    print(f"indexed {total} frames; feature dims "
          f"{ {k: v.shape[1] for k, v in features.items()} }")

    raw = features["A_CURRENT_PROX_RAW"]
    full = features["C_FULL_DEPLOYABLE_INPUT"]
    closeness = full[:, :2560]
    state = full[:, 2560:]
    for i, record in enumerate(meta):
        record["scientific_hash"] = scientific_hash(closeness[i], state[i])
        record["current_prox_hash"] = scientific_hash(closeness[i], np.zeros(1))

    historical = json.loads(args.onset_audit.read_text())["known_false_positive_frames"]
    historical_keys = {(r["trajectory_id"], r["step"]) for r in historical}
    is_historical = np.array([(m["trajectory_id"], m["step"]) in historical_keys
                              for m in meta])
    active = np.array([m["oracle_active"] for m in meta])
    onset = np.array([m["onset"] for m in meta])
    in_pool = np.array([m["partition"] == REFERENCE_POOL for m in meta])

    # ---- step 7: exact and near-identity collisions --------------------------------
    print("searching for exact observable collisions...")
    collisions = {"exact_current_prox": [], "exact_full_input": [], "near_identity": []}
    from collections import defaultdict

    by_prox = defaultdict(list)
    by_full = defaultdict(list)
    for i, record in enumerate(meta):
        by_prox[record["current_prox_hash"]].append(i)
        by_full[record["scientific_hash"]].append(i)

    def record_pair(a, b, kind):
        return {
            "kind": kind,
            "a": {k: meta[a][k] for k in ("trajectory_id", "episode_id", "distribution",
                                          "hazard_present", "step",
                                          "progress_fraction", "oracle_active")},
            "b": {k: meta[b][k] for k in ("trajectory_id", "episode_id", "distribution",
                                          "hazard_present", "step",
                                          "progress_fraction", "oracle_active")},
            "current_field_max_abs_delta": float(
                np.abs(closeness[a] - closeness[b]).max()),
            "state_action_max_abs_delta": float(np.abs(state[a] - state[b]).max()),
            "changed_pixels": int((closeness[a] != closeness[b]).sum()),
            "changed_sensors": int((np.abs(
                (closeness[a] - closeness[b]).reshape(40, -1)).max(axis=1) > 0).sum()),
        }

    for table, key in ((by_prox, "exact_current_prox"), (by_full, "exact_full_input")):
        for members in table.values():
            if len(members) < 2:
                continue
            labels = active[members]
            if labels.all() or not labels.any():
                continue
            for a in members:
                for b in members:
                    if a < b and active[a] != active[b] \
                            and meta[a]["episode_id"] != meta[b]["episode_id"]:
                        collisions[key].append(record_pair(a, b, key))
    print(f"  exact current-proximity collisions : {len(collisions['exact_current_prox'])}")
    print(f"  exact full-input collisions        : {len(collisions['exact_full_input'])}")

    # near-identity: search each active frame's nearest zero frames in raw space
    print("searching for near-identity collisions...")
    pool_tensor = torch.from_numpy(raw).to(device, torch.float32)
    near_hits = 0
    sample = np.flatnonzero(active)
    for start in range(0, len(sample), 64):
        chunk = sample[start:start + 64]
        block = torch.from_numpy(raw[chunk]).to(device, torch.float32)
        distance = torch.cdist(block, pool_tensor)
        nearest = torch.topk(distance, 64, largest=False).indices.cpu().numpy()
        for offset, query in enumerate(chunk):
            for candidate in nearest[offset]:
                if active[candidate] or meta[query]["episode_id"] == \
                        meta[candidate]["episode_id"]:
                    continue
                if np.abs(closeness[query] - closeness[candidate]).max() > \
                        TOLERANCE_CLOSENESS:
                    continue
                delta_state = np.abs(state[query] - state[candidate])
                if delta_state[:9].max() > TOLERANCE_QPOS or \
                        delta_state[9:18].max() > TOLERANCE_QVEL or \
                        delta_state[18:26].max() > TOLERANCE_ACTION:
                    continue
                collisions["near_identity"].append(
                    record_pair(int(query), int(candidate), "near_identity"))
                near_hits += 1
        del distance
    del pool_tensor
    torch.cuda.empty_cache()
    print(f"  near-identity collisions           : {near_hits}")

    # ---- step 6: nearest-neighbour ambiguity in four feature spaces ----------------
    rng = np.random.default_rng(thr.BOOTSTRAP_SEED)
    query_groups = {
        "HISTORICAL_FALSE_POSITIVE": np.flatnonzero(is_historical),
        "ONSET_ZERO": np.flatnonzero(onset & ~active & ~is_historical),
        "LATE_ZERO": np.flatnonzero(~onset & ~active),
        "ONSET_ACTIVE": np.flatnonzero(onset & active),
        "LATE_ACTIVE": np.flatnonzero(~onset & active),
    }
    query_rows = []
    for name, rows in query_groups.items():
        picked = rows if len(rows) <= args.per_group else rng.choice(
            rows, args.per_group, replace=False)
        query_groups[name] = np.sort(picked)
        query_rows.extend(query_groups[name].tolist())
    query_rows = np.array(sorted(set(query_rows)))
    print(f"neighbour queries: {len(query_rows)}")

    pool_rows = np.flatnonzero(in_pool)
    pool_meta = [meta[i] for i in pool_rows]
    neighbour_report = {}
    for space in FEATURE_SPACES:
        matrix = features[space]
        neighbours = neighbour_search(
            matrix[query_rows], matrix[pool_rows],
            [meta[i] for i in query_rows], pool_meta, pool_rows, device)
        lookup = {int(r): n for r, n in zip(query_rows, neighbours)}
        block = {"pool_frames": len(pool_rows),
                 "pool_partition": REFERENCE_POOL,
                 "feature_dimension": int(matrix.shape[1]),
                 "overall": summarise_neighbours(
                     [meta[i] for i in query_rows], neighbours),
                 "by_group": {}}
        for name, rows in query_groups.items():
            block["by_group"][name] = summarise_neighbours(
                [meta[i] for i in rows], [lookup[int(r)] for r in rows])
        block["historical_frames"] = [{
            "trajectory_id": meta[int(r)]["trajectory_id"],
            "step": meta[int(r)]["step"],
            "hazard_present": meta[int(r)]["hazard_present"],
            "nearest_neighbour_label": lookup[int(r)]["neighbour_labels"][0],
            "nearest_distance": lookup[int(r)]["neighbour_distances"][0],
            "opposite_fraction_k8": float(np.mean(
                [lbl != meta[int(r)]["oracle_active"]
                 for lbl in lookup[int(r)]["neighbour_labels"][:8]])),
        } for r in query_groups["HISTORICAL_FALSE_POSITIVE"]]
        neighbour_report[space] = block
        overall = block["overall"]["k=8"]
        print(f"  {space:<28} k8 entropy={overall['median_label_entropy']:.4f} "
              f"opp_frac={overall['mean_opposite_label_fraction']:.4f} "
              f"nn_opposite={overall['fraction_nearest_neighbour_opposite_label']:.4f}")

    # ---- step 8: local activity probability, leave-one-trajectory-out ---------------
    print("estimating local activity probability...")
    space = "D_FROZEN_MODEL_EMBEDDING"
    matrix = features[space]
    neighbours = neighbour_search(matrix[query_rows], matrix[pool_rows],
                                  [meta[i] for i in query_rows], pool_meta,
                                  pool_rows, device)
    p_local = {}
    for row, block in zip(query_rows, neighbours):
        labels = np.array(block["neighbour_labels"][:32])
        p_local[int(row)] = {
            "p_local_active": float(labels.mean()) if labels.size else None,
            "effective_sample_count": int(labels.size),
        }
    for value in p_local.values():
        p = value["p_local_active"]
        value["label_entropy"] = (
            float(-(p * np.log2(max(p, 1e-12)) + (1 - p) * np.log2(max(1 - p, 1e-12))))
            if p is not None else None)
        value["ambiguous"] = bool(p is not None and AMBIGUOUS_LOW <= p <= AMBIGUOUS_HIGH)

    ambiguity_by_group = {}
    for name, rows in query_groups.items():
        values = [p_local[int(r)] for r in rows if int(r) in p_local]
        ambiguity_by_group[name] = {
            "queries": len(values),
            "fraction_ambiguous": float(np.mean([v["ambiguous"] for v in values]))
            if values else None,
            "median_p_local_active": float(np.median(
                [v["p_local_active"] for v in values])) if values else None,
            "median_label_entropy": float(np.median(
                [v["label_entropy"] for v in values])) if values else None,
        }

    calibration_bins = []
    edges = np.linspace(0, 1, 11)
    for low, high in itertools.pairwise(edges):
        rows = [int(r) for r in query_rows
                if low <= (p_local[int(r)]["p_local_active"] or -1) < high]
        if not rows:
            calibration_bins.append({"bin": [float(low), float(high)], "count": 0})
            continue
        calibration_bins.append({
            "bin": [float(low), float(high)], "count": len(rows),
            "predicted": float(np.mean([p_local[r]["p_local_active"] for r in rows])),
            "empirical": float(np.mean([meta[r]["oracle_active"] for r in rows])),
        })

    report = {
        "schema": "hybrid_obstacle_activity_collision_audit_v1",
        "checkpoint_config_hash": payload["config_hash"],
        "frames_indexed": total,
        "neighbour_pool": {"partition": REFERENCE_POOL, "frames": len(pool_rows)},
        "exclusions": ["same trajectory", "same episode identity",
                       "duplicate scientific state hash"],
        "feature_spaces": {k: int(v.shape[1]) for k, v in features.items()},
        "query_groups": {k: len(v) for k, v in query_groups.items()},
        "collisions": {
            "tolerances": {"closeness": TOLERANCE_CLOSENESS, "qpos": TOLERANCE_QPOS,
                           "qvel": TOLERANCE_QVEL, "action": TOLERANCE_ACTION},
            "exact_current_prox_count": len(collisions["exact_current_prox"]),
            "exact_full_input_count": len(collisions["exact_full_input"]),
            "near_identity_count": len(collisions["near_identity"]),
            "exact_full_input_pairs": collisions["exact_full_input"][:50],
            "exact_current_prox_pairs": collisions["exact_current_prox"][:50],
            "near_identity_pairs": collisions["near_identity"][:50],
            "any_exact_full_input_collision": bool(collisions["exact_full_input"]),
        },
        "neighbours": neighbour_report,
        "local_ambiguity": {
            "feature_space": space,
            "k": 32,
            "ambiguous_band": [AMBIGUOUS_LOW, AMBIGUOUS_HIGH],
            "by_group": ambiguity_by_group,
            "calibration_bins": calibration_bins,
            "historical_frames": [{
                "trajectory_id": meta[int(r)]["trajectory_id"],
                "step": meta[int(r)]["step"],
                **p_local[int(r)]} for r in query_groups["HISTORICAL_FALSE_POSITIVE"]],
        },
        "training_performed": False,
        "dataset_modified": False,
    }
    report["report_sha256"] = thr.canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print("\nlocal ambiguity (fraction in [0.2, 0.8]):")
    for name, block in ambiguity_by_group.items():
        print(f"  {name:<28} {block['fraction_ambiguous']:.4f} "
              f"(median p={block['median_p_local_active']:.4f})")
    print(f"exact full-input opposite-label collisions: "
          f"{report['collisions']['exact_full_input_count']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
