#!/usr/bin/env python3
"""Onset-versus-late separability and old-head score-tail analysis.

Handoff steps 9-10.

Step 9 asks whether onset ambiguity is a property of the *fields* rather than of the episode
index. Episode index is never used as a feature here; it only selects which frames go in
which comparison, and the comparison is between hazard-present and hazard-absent onset
fields before any privileged counterfactual is applied.

Step 10 replaces full-range AUROC as the headline. A head at AUROC 0.998 can still be
unusable if its rare errors sit above most true positives, so what matters is the upper tail
and whether errors cluster. Intervals are trajectory-clustered.
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
from causal_parked_skin.data import load_partition, sensor_link_ids
from causal_parked_skin.engine import load_checkpoint, make_batch

DIAGNOSTIC_PARTITIONS = ("offline_reference_test", "reference_calibration",
                         "reference_validation")
ONSET_MIN_FRAMES = 10
ONSET_FRACTION = 0.10
PREVIOUS_THRESHOLD = 0.99960857629776
CHANGED_PIXEL_DECISION = 0.5
TAIL_QUANTILES = (0.001, 0.005, 0.01, 0.02, 0.05)
D_MAX = 0.5


def onset_cutoff(length: int) -> int:
    return max(ONSET_MIN_FRAMES, int(np.ceil(ONSET_FRACTION * length)))


def partial_roc_tpr_at_fpr(score, label, target_fpr):
    """TPR at a fixed FPR, read off the empirical ROC."""
    label = np.asarray(label, dtype=bool)
    if label.sum() == 0 or (~label).sum() == 0:
        return None
    threshold = np.quantile(score[~label], 1.0 - target_fpr)
    return float((score[label] > threshold).mean())


def clustered_bootstrap(values, clusters, statistic, *, replicates=10_000,
                        seed=thr.BOOTSTRAP_SEED, quantiles=(0.025, 0.975)):
    """Resample whole trajectories, not frames."""
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, unique.size, size=(replicates, unique.size))
    grouped = [values[clusters == c] for c in unique]
    draws = np.empty(replicates)
    for r in range(replicates):
        pooled = np.concatenate([grouped[i] for i in index[r]])
        draws[r] = statistic(pooled)
    return {"point": float(statistic(values)),
            "ci_low": float(np.quantile(draws, quantiles[0])),
            "ci_high": float(np.quantile(draws, quantiles[1])),
            "replicates": replicates, "seed": seed, "resampled_unit": "trajectory"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--onset-audit", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch

    stack = json.loads(args.stack.read_text())
    sensor_names = stack["sensor_contract"]["ordered_names"]
    engine.set_sensor_names(sensor_names)
    link_ids, links = sensor_link_ids(sensor_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, payload = load_checkpoint(args.checkpoint, device)

    activity, trajectory_key, onset, active, hazard = [], [], [], [], []
    current_fields, valid_masks, changed_targets, parked_fields = [], [], [], []
    embeddings = []
    captured = {}
    handle = model.to_token.register_forward_hook(
        lambda _m, _i, o: captured.__setitem__("prox", o.detach()))

    for name in DIAGNOSTIC_PARTITIONS:
        partition = load_partition(args.cache, name)
        trajectory_index = np.asarray(partition["trajectory"])
        for index in range(trajectory_index.max() + 1):
            rows = np.flatnonzero(trajectory_index == index)
            length = len(rows)
            cutoff = onset_cutoff(length)
            trajectory_key.extend([partition.trajectory_ids[index]] * length)
            onset.extend([offset < cutoff for offset in range(length)])
            active.extend(np.asarray(partition["oracle_active"])[rows].tolist())
            hazard.extend(np.asarray(partition["hazard_present"])[rows].tolist())
            current_fields.append(np.asarray(partition["current"])[rows])
            valid_masks.append(np.asarray(partition["current_valid"])[rows])
            changed_targets.append(np.asarray(partition["changed"])[rows])
            parked_fields.append(np.asarray(partition["parked"])[rows])
            for start in range(0, length, 256):
                chunk = rows[start:start + 256]
                batch = make_batch(partition, chunk, device)
                with torch.no_grad():
                    out = model(batch["history"], batch["history_valid"], batch["state"])
                    probability = out["changed_probability"]
                    flat = probability.reshape(probability.shape[0], -1)
                activity.append(flat.amax(dim=1).double().cpu().numpy())
                embeddings.append(
                    captured["prox"].reshape(len(chunk), -1).float().cpu().numpy())
    handle.remove()

    activity = np.concatenate(activity)
    embeddings = np.concatenate(embeddings, axis=0)
    current = np.concatenate(current_fields, axis=0)
    valid = np.concatenate(valid_masks, axis=0)
    changed = np.concatenate(changed_targets, axis=0)
    parked = np.concatenate(parked_fields, axis=0)
    trajectory_key = np.array(trajectory_key)
    onset = np.array(onset)
    active = np.array(active)
    hazard = np.array(hazard)
    total = len(activity)
    print(f"scored {total} frames")

    historical = json.loads(args.onset_audit.read_text())["known_false_positive_frames"]

    # ---- step 9: onset vs late field comparison ------------------------------------
    def field_summary(mask) -> dict:
        if not mask.any():
            return {"count": 0}
        subset_current = current[mask]
        depth = (1.0 - subset_current) * D_MAX
        sensor_max = subset_current.reshape(mask.sum(), 40, -1).max(axis=2)
        sensor_active = sensor_max > 0.0
        link_active = np.stack([sensor_active[:, link_ids == i].any(axis=1)
                                for i in range(len(links))], axis=1)
        return {
            "count": int(mask.sum()),
            "mean_closeness": float(subset_current.mean()),
            "mean_nonzero_pixel_fraction": float((subset_current > 0).mean()),
            "mean_max_closeness": float(subset_current.reshape(
                mask.sum(), -1).max(axis=1).mean()),
            "mean_valid_fraction": float(valid[mask].mean()),
            "mean_min_valid_depth_m": float(depth.reshape(
                mask.sum(), -1).min(axis=1).mean()),
            "mean_active_sensor_count": float(sensor_active.sum(axis=1).mean()),
            "mean_active_link_count": float(link_active.sum(axis=1).mean()),
            "mean_changed_pixel_fraction": float(changed[mask].mean()),
            "mean_parked_field_delta": float(
                np.abs(subset_current - parked[mask]).mean()),
        }

    def embedding_distance(mask_a, mask_b, limit=400) -> float:
        if not mask_a.any() or not mask_b.any():
            return float("nan")
        rng = np.random.default_rng(thr.BOOTSTRAP_SEED)
        a_rows = np.flatnonzero(mask_a)
        b_rows = np.flatnonzero(mask_b)
        a = embeddings[rng.choice(a_rows, min(limit, a_rows.size), replace=False)]
        b = embeddings[rng.choice(b_rows, min(limit, b_rows.size), replace=False)]
        with torch.no_grad():
            distance = torch.cdist(torch.from_numpy(a).to(device),
                                   torch.from_numpy(b).to(device))
            return float(distance.mean())

    onset_comparison = {
        "onset_zero": field_summary(onset & ~active),
        "onset_active": field_summary(onset & active),
        "late_zero": field_summary(~onset & ~active),
        "late_active": field_summary(~onset & active),
        "onset_hazard_present_zero": field_summary(onset & ~active & hazard),
        "onset_hazard_absent_zero": field_summary(onset & ~active & ~hazard),
        # Between-class distance is only interpretable against the within-class
        # baseline: onset embeddings are more compressed overall, so a raw comparison
        # of onset-between to late-between would confuse compression with overlap.
        "embedding_distance": {
            "onset_zero_to_onset_active": embedding_distance(onset & ~active,
                                                             onset & active),
            "late_zero_to_late_active": embedding_distance(~onset & ~active,
                                                           ~onset & active),
            "onset_zero_within": embedding_distance(onset & ~active, onset & ~active),
            "onset_active_within": embedding_distance(onset & active, onset & active),
            "late_zero_within": embedding_distance(~onset & ~active, ~onset & ~active),
            "late_active_within": embedding_distance(~onset & active, ~onset & active),
            "onset_hazard_present_vs_absent_zero": embedding_distance(
                onset & ~active & hazard, onset & ~active & ~hazard),
        },
        "episode_index_used_as_feature": False,
    }

    distances = onset_comparison["embedding_distance"]
    onset_within = 0.5 * (distances["onset_zero_within"] + distances["onset_active_within"])
    late_within = 0.5 * (distances["late_zero_within"] + distances["late_active_within"])
    onset_comparison["separation_ratio"] = {
        "definition": ("between-class embedding distance divided by the mean within-class "
                       "distance; 1.0 means the classes are indistinguishable in this space"),
        "onset": float(distances["onset_zero_to_onset_active"] / onset_within),
        "late": float(distances["late_zero_to_late_active"] / late_within),
        "onset_within_baseline": float(onset_within),
        "late_within_baseline": float(late_within),
    }

    # separability of hazard-present vs hazard-absent onset fields, before the
    # privileged counterfactual: a simple max-closeness statistic
    onset_zero = onset & ~active
    stat = current.reshape(total, -1).max(axis=1)
    onset_comparison["hazard_separability_before_counterfactual"] = {
        "statistic": "max current closeness over the whole field",
        "onset_hazard_present_median": float(np.median(stat[onset_zero & hazard])),
        "onset_hazard_absent_median": float(np.median(stat[onset_zero & ~hazard])),
        "late_hazard_present_median": float(np.median(stat[~onset & ~active & hazard])),
        "late_hazard_absent_median": float(np.median(stat[~onset & ~active & ~hazard])),
    }

    # ---- step 10: score tails -------------------------------------------------------
    zero_scores = activity[~active]
    active_scores = activity[active]
    per_trajectory = {}
    for key in np.unique(trajectory_key):
        mask = (trajectory_key == key) & ~active
        if not mask.any():
            continue
        ordered = np.sort(activity[mask])[::-1]
        per_trajectory[key] = {
            "max": float(ordered[0]),
            "top2": float(ordered[1]) if ordered.size > 1 else None,
            "top5": float(ordered[4]) if ordered.size > 4 else None,
            "false_positives_at_previous_threshold": int(
                (activity[mask] >= PREVIOUS_THRESHOLD).sum()),
        }

    fired = activity >= PREVIOUS_THRESHOLD
    runs = []
    for key in np.unique(trajectory_key):
        mask = trajectory_key == key
        runs.append(thr.max_true_run(fired[mask] & ~active[mask]))
    clustered_episodes = sum(1 for r in runs if r >= 2)

    overlap = {}
    for q in TAIL_QUANTILES:
        cutoff = np.quantile(zero_scores, 1.0 - q)
        overlap[f"zero_top_{q * 100:g}pct"] = {
            "score_cutoff": float(cutoff),
            "active_fraction_below_cutoff": float((active_scores < cutoff).mean()),
            "zero_frames_above_cutoff": int((zero_scores >= cutoff).sum()),
        }

    partial_roc = {}
    for target in TAIL_QUANTILES:
        value = partial_roc_tpr_at_fpr(activity, active, target)
        partial_roc[f"tpr_at_fpr_{target * 100:g}pct"] = value

    def mean_statistic(values):
        return float(np.mean(values))

    tail = {
        "zero_score_distribution": {
            "count": int(zero_scores.size),
            "median": float(np.median(zero_scores)),
            **{f"q{p}": float(np.quantile(zero_scores, p / 100))
               for p in (90, 99, 99.5, 99.9)},
            "max": float(zero_scores.max()),
        },
        "active_score_distribution": {
            "count": int(active_scores.size),
            "median": float(np.median(active_scores)),
            **{f"q{p}": float(np.quantile(active_scores, p / 100))
               for p in (1, 5, 10, 25)},
            "min": float(active_scores.min()),
        },
        "per_trajectory_max_score": {
            "median": float(np.median([v["max"] for v in per_trajectory.values()])),
            "max": float(np.max([v["max"] for v in per_trajectory.values()])),
            "trajectories": len(per_trajectory),
        },
        "tail_overlap": overlap,
        "partial_roc": partial_roc,
        "clustered_false_positive_episodes": clustered_episodes,
        "max_consecutive_false_positive_length": int(max(runs)),
        "false_positive_run_histogram": {str(r): runs.count(r)
                                         for r in sorted(set(runs))},
        "false_positive_rate_clustered_bootstrap": clustered_bootstrap(
            fired[~active].astype(np.float64), trajectory_key[~active], mean_statistic),
        "active_recall_clustered_bootstrap": clustered_bootstrap(
            fired[active].astype(np.float64), trajectory_key[active], mean_statistic),
        "note": ("full-range AUROC is deliberately not the headline; the upper tail and "
                 "the clustering of errors are what determine usability"),
    }

    report = {
        "schema": "hybrid_obstacle_activity_onset_tail_audit_v1",
        "checkpoint_config_hash": payload["config_hash"],
        "frames": total,
        "previous_threshold": PREVIOUS_THRESHOLD,
        "onset_vs_late": onset_comparison,
        "score_tail": tail,
        "historical_frame_count": len(historical),
        "training_performed": False,
    }
    report["report_sha256"] = thr.canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print("\nonset vs late field statistics:")
    for name in ("onset_zero", "onset_active", "late_zero", "late_active"):
        block = onset_comparison[name]
        print(f"  {name:<14} n={block['count']:>6} maxclose={block['mean_max_closeness']:.4f} "
              f"sensors={block['mean_active_sensor_count']:.2f} "
              f"changed={block['mean_changed_pixel_fraction']:.6f}")
    print("embedding distances:", json.dumps(
        {k: round(v, 3) for k, v in onset_comparison["embedding_distance"].items()}))
    ratio = onset_comparison["separation_ratio"]
    print(f"separation ratio (between/within): onset={ratio['onset']:.4f} "
          f"late={ratio['late']:.4f}")
    print(f"\nscore tails: zero q99.9={tail['zero_score_distribution']['q99.9']:.6f} "
          f"max={tail['zero_score_distribution']['max']:.6f}")
    print(f"active q1={tail['active_score_distribution']['q1']:.6f} "
          f"min={tail['active_score_distribution']['min']:.6f}")
    print("partial ROC:", json.dumps({k: round(v, 4) for k, v in partial_roc.items()
                                      if v is not None}))
    print(f"clustered FP episodes: {clustered_episodes}, "
          f"max run {tail['max_consecutive_false_positive_length']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
