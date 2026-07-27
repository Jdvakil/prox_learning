#!/usr/bin/env python3
"""Frozen three-seed ensemble audit and epistemic-uncertainty separability test.

Handoff steps 3-5. Read-only throughout: three checkpoints are loaded, run, and never
written; no optimizer is constructed anywhere in this file.

The question is narrow. The old head fails on 17 oracle-zero frames with near-maximal
confidence. If those frames sit where the three independently-seeded parked-field models
disagree, an abstention signal exists and is worth building on. If the seeds agree
confidently and wrongly, disagreement is not measuring what we need, and stacking more
same-input models will not manufacture the missing information.

Seeds 1 and 2 are diagnostics. They are never averaged into a deployed output and never
become a deployment candidate; a test asserts the decision artifact keeps seed 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from causal_parked_skin import engine
from causal_parked_skin import threshold as thr
from causal_parked_skin.data import SOURCE_MODES, load_partition, sensor_link_ids
from causal_parked_skin.engine import load_checkpoint, make_batch
from causal_parked_skin.model import FrozenSafetyHead

DIAGNOSTIC_PARTITIONS = ("offline_reference_test", "reference_calibration",
                         "reference_validation")
ONSET_MIN_FRAMES = 10
ONSET_FRACTION = 0.10
CHANGED_PIXEL_DECISION = 0.5
ACTIVE_SENSOR_DECISION = 0.5

# predeclared separability gates (handoff step 5)
MIN_REJECTED_HISTORICAL = 16          # of 17
MIN_RETAINED_ACTIVE = 0.80
MIN_RETAINED_HARD_ACTIVE = 0.80

GROUPS = ("A_HISTORICAL_FALSE_POSITIVE", "B_ONSET_ZERO", "C_LATE_ZERO",
          "D_ONSET_ACTIVE", "E_LATE_ACTIVE", "F_HAZARD_ABSENT_ZERO",
          "G_HARD_TRUE_ACTIVE")


def onset_cutoff(length: int) -> int:
    return max(ONSET_MIN_FRAMES, int(np.ceil(ONSET_FRACTION * length)))


def array_hash(*arrays) -> str:
    import hashlib

    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype.str).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    label = np.asarray(label, dtype=bool)
    positives, negatives = int(label.sum()), int((~label).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    return float((ranks[label].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def partial_auroc(score: np.ndarray, label: np.ndarray, max_fpr: float = 0.05) -> float:
    """Standardised partial AUROC over FPR in [0, max_fpr] (McClish), 0.5 = chance."""
    label = np.asarray(label, dtype=bool)
    if label.sum() == 0 or (~label).sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    hits = label[order]
    tpr = np.cumsum(hits) / label.sum()
    fpr = np.cumsum(~hits) / (~label).sum()
    keep = fpr <= max_fpr
    if not keep.any():
        return float("nan")
    area = np.trapezoid(np.concatenate([[0.0], tpr[keep]]),
                        np.concatenate([[0.0], fpr[keep]]))
    # McClish standardisation onto [0.5, 1]
    minimum = max_fpr ** 2 / 2
    maximum = max_fpr
    return float(0.5 * (1 + (area - minimum) / (maximum - minimum)))


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    label = np.asarray(label, dtype=bool)
    positives = int(label.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    hits = label[order]
    precision = np.cumsum(hits) / np.arange(1, hits.size + 1)
    return float((precision * hits).sum() / positives)


def jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Jaccard agreement between two boolean masks."""
    intersection = (a & b).sum(axis=-1).astype(np.float64)
    union = (a | b).sum(axis=-1).astype(np.float64)
    return np.where(union > 0, intersection / np.maximum(union, 1), 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--checkpoint-root", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--onset-audit", required=True, type=Path)
    ap.add_argument("--groups-out", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch

    stack = json.loads(args.stack.read_text())
    sensor_names = stack["sensor_contract"]["ordered_names"]
    engine.set_sensor_names(sensor_names)
    link_ids, links = sensor_link_ids(sensor_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = FrozenSafetyHead.load(args.safety_dir, device=device)

    models = {}
    for seed in (0, 1, 2):
        path = args.checkpoint_root / f"CURRENT_FRAME_ONLY__seed{seed}" / "best.pt"
        model, payload = load_checkpoint(path, device)
        if payload["config"]["variant"] != "CURRENT_FRAME_ONLY" or \
                payload["config"]["seed"] != seed:
            raise SystemExit(f"checkpoint {path} is not CURRENT_FRAME_ONLY seed {seed}")
        models[seed] = model
    before = {s: [p.detach().clone() for p in m.parameters()] for s, m in models.items()}

    historical = json.loads(args.onset_audit.read_text())["known_false_positive_frames"]
    historical_keys = {(r["trajectory_id"], r["step"]) for r in historical}
    if len(historical_keys) != 17:
        raise SystemExit(f"expected 17 historical frames, got {len(historical_keys)}")

    # ---- score every diagnostic frame with all three seeds ------------------------
    records: list[dict] = []
    per_seed: dict[int, dict[str, list]] = {s: {"parked": [], "head": [], "dq": [],
                                                "changed": [], "sensor": []}
                                            for s in (0, 1, 2)}
    old_head_activity: list[np.ndarray] = []

    for name in DIAGNOSTIC_PARTITIONS:
        partition = load_partition(args.cache, name)
        trajectory_index = np.asarray(partition["trajectory"])
        oracle_active = np.asarray(partition["oracle_active"]).astype(bool)
        hazard = np.asarray(partition["hazard_present"]).astype(bool)
        modes = np.asarray(partition["source_mode"])
        oracle_dq = np.asarray(partition["oracle_dq"], dtype=np.float64)
        current = np.asarray(partition["current"])
        state = np.asarray(partition["state"])

        for index in range(trajectory_index.max() + 1):
            rows = np.flatnonzero(trajectory_index == index)
            length = len(rows)
            cutoff = onset_cutoff(length)
            for offset, row in enumerate(rows):
                records.append({
                    "partition": name, "row": int(row),
                    "trajectory_id": partition.trajectory_ids[index],
                    "episode_id": partition.episode_ids[index],
                    "distribution": SOURCE_MODES[int(modes[row])],
                    "hazard_present": bool(hazard[row]),
                    "step": offset, "trajectory_length": length,
                    "progress_fraction": offset / max(length - 1, 1),
                    "onset": offset < cutoff,
                    "oracle_active": bool(oracle_active[row]),
                    "oracle_dq_norm": float(np.linalg.norm(oracle_dq[row])),
                    "current_prox_hash": array_hash(current[row]),
                    "state_action_hash": array_hash(state[row]),
                })

            for start in range(0, length, 256):
                chunk = rows[start:start + 256]
                batch = make_batch(partition, chunk, device)
                for seed, model in models.items():
                    with torch.no_grad():
                        out = model(batch["history"], batch["history_valid"],
                                    batch["state"])
                        parked = out["parked"]
                        predicted_head = head(parked)
                        dq = head(out["current"]) - predicted_head
                        probability = out["changed_probability"]
                        changed = (probability > CHANGED_PIXEL_DECISION)
                        sensor_active = changed.reshape(
                            changed.shape[0], 40, -1).any(dim=-1)
                    per_seed[seed]["parked"].append(parked.float().cpu().numpy())
                    per_seed[seed]["head"].append(predicted_head.double().cpu().numpy())
                    per_seed[seed]["dq"].append(dq.double().cpu().numpy())
                    per_seed[seed]["changed"].append(
                        changed.reshape(changed.shape[0], -1).cpu().numpy())
                    per_seed[seed]["sensor"].append(sensor_active.cpu().numpy())
                    if seed == 0:
                        flat = probability.reshape(probability.shape[0], -1)
                        old_head_activity.append(flat.amax(dim=1).double().cpu().numpy())

    for seed in models:
        for key in per_seed[seed]:
            per_seed[seed][key] = np.concatenate(per_seed[seed][key], axis=0)
    activity = np.concatenate(old_head_activity)
    total = len(records)
    print(f"scored {total} diagnostic frames with 3 seeds")

    weights_unchanged = all(
        torch.equal(a, b) for seed in models
        for a, b in zip(before[seed], models[seed].parameters()))

    # ---- uncertainty metrics -----------------------------------------------------
    parked = np.stack([per_seed[s]["parked"] for s in (0, 1, 2)])        # (3, N, 40,8,8)
    heads = np.stack([per_seed[s]["head"] for s in (0, 1, 2)])           # (3, N, 7)
    dqs = np.stack([per_seed[s]["dq"] for s in (0, 1, 2)])               # (3, N, 7)
    changed = np.stack([per_seed[s]["changed"] for s in (0, 1, 2)])      # (3, N, 2560)
    sensors = np.stack([per_seed[s]["sensor"] for s in (0, 1, 2)])       # (3, N, 40)

    pixel_variance = parked.var(axis=0, ddof=0).reshape(total, -1)
    norms = np.linalg.norm(dqs, axis=-1)                                 # (3, N)
    mean_norm = norms.mean(axis=0)

    def pairwise_cosine(vectors):
        out = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            a, b = vectors[i], vectors[j]
            denominator = (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
                           + thr.COSINE_EPSILON)
            out.append((a * b).sum(-1) / denominator)
        return np.stack(out)

    cosines = pairwise_cosine(dqs)
    link_masks = np.stack([
        np.stack([sensors[s][:, link_ids == i].any(axis=1)
                  for i in range(len(links))], axis=1) for s in range(3)])

    metrics = {
        "mean_parked_field_variance": pixel_variance.mean(axis=1),
        "max_parked_field_variance": pixel_variance.max(axis=1),
        "predicted_head_variance": heads.var(axis=0, ddof=0).mean(axis=1),
        "differential_norm_variance": norms.var(axis=0, ddof=0),
        "norm_coefficient_of_variation": np.where(
            mean_norm > 1e-9, norms.std(axis=0, ddof=0) / np.maximum(mean_norm, 1e-9), 0.0),
        "mean_pairwise_differential_cosine": cosines.mean(axis=0),
        "min_pairwise_differential_cosine": cosines.min(axis=0),
        "changed_pixel_mask_agreement": np.stack(
            [jaccard(changed[i], changed[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        ).mean(axis=0),
        "active_sensor_set_agreement": np.stack(
            [jaccard(sensors[i], sensors[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        ).mean(axis=0),
        "active_link_set_agreement": np.stack(
            [jaccard(link_masks[i], link_masks[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        ).mean(axis=0),
    }

    # ---- frozen diagnostic groups -------------------------------------------------
    is_historical = np.array([(r["trajectory_id"], r["step"]) in historical_keys
                              for r in records])
    onset = np.array([r["onset"] for r in records])
    active = np.array([r["oracle_active"] for r in records])
    hazard = np.array([r["hazard_present"] for r in records])
    dq_norm = np.array([r["oracle_dq_norm"] for r in records])

    lowest_historical_activity = activity[is_historical].min()
    hard_true_active = active & (activity < activity[is_historical].max())
    group_masks = {
        "A_HISTORICAL_FALSE_POSITIVE": is_historical,
        "B_ONSET_ZERO": onset & ~active,
        "C_LATE_ZERO": ~onset & ~active,
        "D_ONSET_ACTIVE": onset & active,
        "E_LATE_ACTIVE": ~onset & active,
        "F_HAZARD_ABSENT_ZERO": ~hazard & (dq_norm == 0.0),
        "G_HARD_TRUE_ACTIVE": hard_true_active,
    }
    if int(is_historical.sum()) != 17:
        raise SystemExit(f"historical group has {int(is_historical.sum())} frames")

    groups_payload = {
        "schema": "hybrid_obstacle_activity_groups_v1",
        "onset_definition": (f"episode_step < max({ONSET_MIN_FRAMES}, "
                             f"ceil({ONSET_FRACTION} * trajectory_length))"),
        "hard_true_active_definition": (
            "oracle-active frames whose old-head activity score is below the score of at "
            "least one historical false-positive frame, i.e. below the maximum of the 17"),
        "historical_activity_range": [float(lowest_historical_activity),
                                      float(activity[is_historical].max())],
        "partitions": list(DIAGNOSTIC_PARTITIONS),
        "development4_frames": 0,
        "confirmatory41_frames": 0,
        "groups": {},
    }
    for name, mask in group_masks.items():
        members = [records[i] for i in np.flatnonzero(mask)]
        groups_payload["groups"][name] = {
            "count": len(members),
            "hazard_present": sum(1 for m in members if m["hazard_present"]),
            "hazard_absent": sum(1 for m in members if not m["hazard_present"]),
            "distributions": {d: sum(1 for m in members if m["distribution"] == d)
                              for d in SOURCE_MODES},
            "frames": members if name == "A_HISTORICAL_FALSE_POSITIVE" else members[:200],
            "frames_truncated": name != "A_HISTORICAL_FALSE_POSITIVE"
            and len(members) > 200,
        }
    groups_payload["groups_sha256"] = thr.canonical_hash(
        {k: v["count"] for k, v in groups_payload["groups"].items()})
    args.groups_out.parent.mkdir(parents=True, exist_ok=True)
    args.groups_out.write_text(json.dumps(groups_payload, indent=2, sort_keys=True,
                                          default=str) + "\n")
    for name, block in groups_payload["groups"].items():
        print(f"  {name:<32} n={block['count']:>6} "
              f"(haz+ {block['hazard_present']}, haz- {block['hazard_absent']})")

    # ---- separability -------------------------------------------------------------
    retained_positive = group_masks["D_ONSET_ACTIVE"] | group_masks["E_LATE_ACTIVE"]
    evaluation_mask = is_historical | retained_positive
    separability = {}
    for name, values in metrics.items():
        # agreement metrics run the other way: low agreement means high uncertainty
        higher_is_uncertain = "agreement" not in name and "cosine" not in name
        signal = values if higher_is_uncertain else -values
        label = is_historical[evaluation_mask]
        score = signal[evaluation_mask]

        quantiles = {}
        for group, mask in group_masks.items():
            subset = values[mask]
            quantiles[group] = {
                "count": int(mask.sum()),
                "median": float(np.median(subset)) if subset.size else None,
                "q05": float(np.quantile(subset, 0.05)) if subset.size else None,
                "q95": float(np.quantile(subset, 0.95)) if subset.size else None,
            }

        order = np.argsort(-signal, kind="stable")
        rank_of = np.empty(total, dtype=np.int64)
        rank_of[order] = np.arange(total)
        top_fraction = {
            f"top_{int(p * 100)}pct": float(
                (rank_of[is_historical] < p * total).mean())
            for p in (0.01, 0.05, 0.10, 0.25)}

        recall_after_rejection = {}
        for reject in (0.01, 0.02, 0.05, 0.10):
            cutoff = np.quantile(signal, 1.0 - reject)
            keep = signal < cutoff
            recall_after_rejection[f"reject_{int(reject * 100)}pct"] = {
                "active_recall": float(keep[active].mean()),
                "hard_true_active_recall": float(
                    keep[group_masks["G_HARD_TRUE_ACTIVE"]].mean())
                if group_masks["G_HARD_TRUE_ACTIVE"].any() else None,
                "historical_rejected": int((~keep)[is_historical].sum()),
            }

        # does ANY monotonic threshold meet the predeclared gates simultaneously?
        candidates = np.unique(signal[is_historical])
        gate_met = None
        for candidate in np.sort(candidates):
            reject = signal >= candidate
            rejected = int(reject[is_historical].sum())
            if rejected < MIN_REJECTED_HISTORICAL:
                continue
            retained_active = float((~reject)[active].mean())
            retained_hard = float((~reject)[group_masks["G_HARD_TRUE_ACTIVE"]].mean()) \
                if group_masks["G_HARD_TRUE_ACTIVE"].any() else 1.0
            if retained_active >= MIN_RETAINED_ACTIVE and \
                    retained_hard >= MIN_RETAINED_HARD_ACTIVE:
                gate_met = {"threshold": float(candidate),
                            "historical_rejected": rejected,
                            "active_recall": retained_active,
                            "hard_true_active_recall": retained_hard}
                break

        separability[name] = {
            "higher_value_means_more_uncertain": higher_is_uncertain,
            "auroc_historical_vs_active": auroc(score, label),
            "partial_auroc_fpr_5pct": partial_auroc(score, label, 0.05),
            "average_precision": average_precision(score, label),
            "by_group": quantiles,
            "historical_in_top": top_fraction,
            "recall_after_rejection": recall_after_rejection,
            "gate_satisfied": gate_met is not None,
            "gate_operating_point": gate_met,
        }

    any_gate = [k for k, v in separability.items() if v["gate_satisfied"]]
    report = {
        "schema": "hybrid_obstacle_activity_ensemble_audit_v1",
        "frames_scored": total,
        "seeds": [0, 1, 2],
        "seed_disposition": {
            "deployment_candidate": "seed 0",
            "seeds_1_and_2": "diagnostic only",
            "checkpoints_averaged": False,
            "ensembled_for_execution": False,
        },
        "model_weights_unchanged": weights_unchanged,
        "training_performed": False,
        "groups": {k: int(v.sum()) for k, v in group_masks.items()},
        "separability_gates": {
            "min_rejected_historical": MIN_REJECTED_HISTORICAL,
            "historical_total": 17,
            "min_retained_active": MIN_RETAINED_ACTIVE,
            "min_retained_hard_true_active": MIN_RETAINED_HARD_ACTIVE,
        },
        "metrics": separability,
        "metrics_satisfying_gate": any_gate,
        "any_metric_satisfies_gate": bool(any_gate),
        "historical_frame_metrics": {
            name: [float(values[i]) for i in np.flatnonzero(is_historical)]
            for name, values in metrics.items()},
    }
    report["report_sha256"] = thr.canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"\nmodel weights unchanged: {weights_unchanged}")
    print(f"{'metric':<38}{'AUROC':>8}{'pAUC5':>8}{'gate':>7}")
    for name, block in separability.items():
        print(f"  {name:<36}{block['auroc_historical_vs_active']:>8.4f}"
              f"{block['partial_auroc_fpr_5pct']:>8.4f}"
              f"{block['gate_satisfied']!s:>7}")
    print(f"metrics satisfying the predeclared gate: {any_gate or 'NONE'}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
