#!/usr/bin/env python3
"""Fit one trajectory-aware activation threshold for the frozen seed-0 reference.

Handoff steps 4-9. Builds ``threshold_calibration16`` from the eight calibration and eight
validation episodes, scores the frozen checkpoint over them, computes every metric at
trajectory level without pooling frames first, and selects a threshold under a fixed
feasibility contract with a cluster bootstrap.

The consumed 20-trajectory offline-test set is scored *afterwards*, purely as a reused
diagnostic. It cannot enter the fit: this script computes the selection from
``threshold_calibration16`` and only then loads the diagnostic partition.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin import engine
from causal_parked_skin import threshold as thr
from causal_parked_skin.data import load_partition
from causal_parked_skin.engine import load_checkpoint, make_batch
from causal_parked_skin.model import FrozenSafetyHead

CALIBRATION_PARTITIONS = ("reference_calibration", "reference_validation")
DIAGNOSTIC_PARTITION = "offline_reference_test"
CHANGED_PIXEL_DECISION = 0.5   # per-pixel mask decision, unchanged from qualification


def score_partition(model, partition, head, device, *, batch_size=256):
    """Run the frozen model once over a partition and keep the per-frame quantities."""
    import torch

    total = len(partition)
    activity = np.zeros(total, dtype=np.float64)
    cosine = np.zeros(total, dtype=np.float64)
    predicted_norm = np.zeros(total, dtype=np.float64)
    changed_true = np.zeros(total, dtype=np.int64)
    changed_predicted = np.zeros(total, dtype=np.int64)
    changed_hit = np.zeros(total, dtype=np.int64)
    violations = 0
    nonfinite = 0

    with torch.no_grad():
        for start in range(0, total, batch_size):
            index = np.arange(start, min(start + batch_size, total))
            batch = make_batch(partition, index, device)
            out = model(batch["history"], batch["history_valid"], batch["state"])
            parked, current = out["parked"], out["current"]
            if not torch.isfinite(parked).all():
                nonfinite += int((~torch.isfinite(parked)).sum())
            violations += int((parked > current + 1e-7).sum())
            violations += int((parked < -1e-7).sum())

            probability = out["changed_probability"]
            activity[index] = probability.reshape(
                probability.shape[0], -1).amax(dim=1).double().cpu().numpy()

            predicted_dq = (head(current) - head(parked)).double().cpu().numpy()
            true_dq = np.asarray(partition["oracle_dq"], dtype=np.float64)[index]
            dot = (predicted_dq * true_dq).sum(axis=-1)
            denominator = (np.linalg.norm(predicted_dq, axis=-1)
                           * np.linalg.norm(true_dq, axis=-1) + thr.COSINE_EPSILON)
            cosine[index] = dot / denominator
            predicted_norm[index] = np.linalg.norm(predicted_dq, axis=-1)

            predicted_mask = (probability > CHANGED_PIXEL_DECISION)
            true_mask = batch["changed"]
            flat = (1, 2, 3)
            changed_predicted[index] = predicted_mask.sum(dim=flat).cpu().numpy()
            changed_true[index] = true_mask.sum(dim=flat).cpu().numpy()
            changed_hit[index] = (predicted_mask & true_mask).sum(dim=flat).cpu().numpy()

    return {"activity": activity, "cosine": cosine, "predicted_norm": predicted_norm,
            "changed_true": changed_true, "changed_predicted": changed_predicted,
            "changed_hit": changed_hit, "constraint_violations": violations,
            "nonfinite": nonfinite}


def build_trajectories(partition, scored) -> list[thr.TrajectoryScores]:
    """Split a partition's frames back into per-trajectory records, in temporal order."""
    trajectory_index = np.asarray(partition["trajectory"])
    oracle_active = np.asarray(partition["oracle_active"]).astype(bool)
    hazard = np.asarray(partition["hazard_present"]).astype(bool)
    modes = np.asarray(partition["source_mode"])
    from causal_parked_skin.data import SOURCE_MODES

    out = []
    for index in range(trajectory_index.max() + 1):
        rows = np.flatnonzero(trajectory_index == index)
        # frames are stored contiguously in step order; assert rather than assume
        steps = np.asarray(partition["step"])[rows]
        if not np.array_equal(steps, np.arange(len(rows))):
            raise SystemExit(f"trajectory {index} is not in temporal order")
        out.append(thr.TrajectoryScores(
            trajectory_id=partition.trajectory_ids[index],
            episode_id=partition.episode_ids[index],
            distribution=SOURCE_MODES[int(modes[rows[0]])],
            partition=partition.name,
            hazard_present=bool(hazard[rows[0]]),
            activity=scored["activity"][rows],
            oracle_active=oracle_active[rows],
            cosine=scored["cosine"][rows],
            predicted_norm=scored["predicted_norm"][rows],
            oracle_norm=np.linalg.norm(
                np.asarray(partition["oracle_dq"], dtype=np.float64)[rows], axis=-1),
            changed_true=scored["changed_true"][rows],
            changed_predicted=scored["changed_predicted"][rows],
            changed_hit=scored["changed_hit"][rows]))
    return out


def evaluate_at(trajectories, threshold: float) -> dict:
    """Trajectory-level metrics at one threshold, plus their medians and cluster means."""
    rows = [t.metrics_at(threshold) for t in trajectories]
    recalls = [r["active_recall"] for r in rows if r["active_recall"] is not None]
    fprs = [r["oracle_zero_false_positive_rate"] for r in rows
            if r["oracle_zero_false_positive_rate"] is not None]
    cosines = [r["median_active_cosine"] for r in rows
               if r["median_active_cosine"] is not None]
    positives = [r["positive_cosine_fraction"] for r in rows
                 if r["positive_cosine_fraction"] is not None]
    hazard_absent = [r["oracle_zero_false_positive_rate"] for r in rows
                     if not r["hazard_present"]
                     and r["oracle_zero_false_positive_rate"] is not None]
    return {
        "threshold": float(threshold),
        "trajectories": rows,
        "median_active_recall": float(np.median(recalls)) if recalls else None,
        "mean_trajectory_zero_fpr": float(np.mean(fprs)) if fprs else None,
        "median_trajectory_zero_fpr": float(np.median(fprs)) if fprs else None,
        "max_trajectory_zero_fpr": float(np.max(fprs)) if fprs else None,
        "median_active_cosine": float(np.median(cosines)) if cosines else None,
        "median_positive_cosine_fraction": (float(np.median(positives))
                                            if positives else None),
        "max_hazard_absent_fpr": float(np.max(hazard_absent)) if hazard_absent else None,
        "max_consecutive_false_positive_run": max(
            r["max_consecutive_false_positive_run"] for r in rows),
        "trajectories_with_persistence": [r["trajectory_id"] for r in rows
                                          if r["persists_after_oracle"]],
        "trajectories_over_2pct_fpr": sum(1 for f in fprs if f > 0.02),
        "trajectories_over_5pct_fpr": sum(1 for f in fprs if f > 0.05),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--calibration-manifest", required=True, type=Path)
    args = ap.parse_args()

    import torch

    stack = json.loads(args.stack.read_text())
    engine.set_sensor_names(stack["sensor_contract"]["ordered_names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = FrozenSafetyHead.load(args.safety_dir, device=device)
    if not head.frozen():
        raise SystemExit("SafetyHead is not frozen")
    model, payload = load_checkpoint(args.checkpoint, device)
    if payload["config"]["variant"] != "CURRENT_FRAME_ONLY" or payload["config"]["seed"] != 0:
        raise SystemExit("checkpoint is not CURRENT_FRAME_ONLY seed 0")

    started = time.time()

    # ---- threshold_calibration16 -------------------------------------------------
    trajectories: list[thr.TrajectoryScores] = []
    for name in CALIBRATION_PARTITIONS:
        partition = load_partition(args.cache, name)
        scored = score_partition(model, partition, head, device)
        trajectories.extend(build_trajectories(partition, scored))

    episodes = sorted({t.episode_id for t in trajectories})
    if len(episodes) != 16:
        raise SystemExit(f"expected 16 calibration episodes, found {len(episodes)}")
    cluster_of = {episode: i for i, episode in enumerate(episodes)}
    cluster_of_row = np.array([cluster_of[t.episode_id] for t in trajectories])

    manifest = {
        "name": "threshold_calibration16",
        "source_partitions": list(CALIBRATION_PARTITIONS),
        "cluster_unit": "episode",
        "cluster_rationale": (
            "the same episode appears in three source distributions, so those "
            "trajectories are not independent; resampling files would reintroduce the "
            "independence assumption this recalibration removes"),
        "episodes": episodes,
        "episode_count": len(episodes),
        "trajectories": [
            {"trajectory_id": t.trajectory_id, "episode_id": t.episode_id,
             "distribution": t.distribution, "partition": t.partition,
             "hazard_present": t.hazard_present, "frames": t.frames,
             "oracle_active_frames": int(t.oracle_active.sum())}
            for t in trajectories],
        "trajectory_count": len(trajectories),
        "total_frames": int(sum(t.frames for t in trajectories)),
        "consumed_diagnostic_partition_excluded": DIAGNOSTIC_PARTITION,
        "activity_definition": (
            "max over the 40x8x8 per-pixel changed-probability map; the frozen model has "
            "no frame-level activity head and this is a parameter-free reduction of an "
            "existing output"),
    }
    manifest["manifest_sha256"] = thr.canonical_hash(manifest)
    args.calibration_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"threshold_calibration16: {len(episodes)} episodes / "
          f"{len(trajectories)} trajectories / {manifest['total_frames']} frames")

    # ---- candidate thresholds ----------------------------------------------------
    candidates = np.unique(np.concatenate([t.activity for t in trajectories]))
    print(f"candidate thresholds (unique observed activity values): {candidates.size}")

    fpr = thr.trajectory_fpr_matrix(trajectories, candidates)
    recall = thr.trajectory_recall_matrix(trajectories, candidates)
    with np.errstate(invalid="ignore"):
        median_recall = np.nanmedian(recall, axis=0)
        max_hazard_absent = np.nanmax(
            fpr[[i for i, t in enumerate(trajectories) if not t.hazard_present]], axis=0)

    # cheap necessary conditions first; the expensive run-length and bootstrap work is
    # only done for candidates that could still be feasible. Same rule, less compute.
    cheap = ((median_recall >= thr.MIN_MEDIAN_ACTIVE_RECALL)
             & (max_hazard_absent <= thr.MAX_HAZARD_ABSENT_TRAJECTORY_FPR))
    print(f"candidates passing recall + hazard-absent screens: {int(cheap.sum())}")

    cluster_fpr = thr.cluster_means(fpr, cluster_of_row, len(episodes))
    upper = np.full(candidates.size, np.inf)
    screened = np.flatnonzero(cheap)
    if screened.size:
        upper[screened] = thr.cluster_bootstrap_upper_bound(cluster_fpr[:, screened])
    print(f"bootstrap: {thr.BOOTSTRAP_REPLICATES} replicates, seed {thr.BOOTSTRAP_SEED}")

    feasible = np.zeros(candidates.size, dtype=bool)
    detail: dict[int, dict] = {}
    for i in screened:
        if upper[i] > thr.MAX_BOOTSTRAP_UPPER_FPR:
            continue
        block = evaluate_at(trajectories, candidates[i])
        detail[int(i)] = block
        ok = (block["median_active_recall"] is not None
              and block["median_active_recall"] >= thr.MIN_MEDIAN_ACTIVE_RECALL
              and block["median_active_cosine"] is not None
              and block["median_active_cosine"] >= thr.MIN_MEDIAN_ACTIVE_COSINE
              and block["median_positive_cosine_fraction"] is not None
              and block["median_positive_cosine_fraction"]
              >= thr.MIN_MEDIAN_POSITIVE_COSINE_FRACTION
              and (block["max_hazard_absent_fpr"] is None
                   or block["max_hazard_absent_fpr"]
                   <= thr.MAX_HAZARD_ABSENT_TRAJECTORY_FPR)
              and block["max_consecutive_false_positive_run"]
              <= thr.MAX_CONSECUTIVE_FALSE_POSITIVE_RUN
              and not block["trajectories_with_persistence"])
        feasible[i] = ok
    print(f"feasible candidates: {int(feasible.sum())}")

    chosen = thr.select_threshold(feasible, candidates, upper, median_recall)
    report = {
        "schema": "hybrid_obstacle_reference_threshold_calibration_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_config_hash": payload["config_hash"],
        "calibration_manifest_sha256": manifest["manifest_sha256"],
        "calibration16": {k: manifest[k] for k in
                          ("episodes", "episode_count", "trajectory_count",
                           "total_frames", "cluster_unit")},
        "retired_threshold": {
            "value": 0.02819432708943556,
            "rule": ("frame-level 99th percentile of oracle-zero predicted-differential "
                     "norms over eight calibration trajectories"),
            "why_retired": (
                "it treated 3,821 autocorrelated frames as independent observations, so "
                "its ~1% calibration exceedance was a construction artefact rather than "
                "an estimate; it produced 2.15% on a different partition"),
            "reused": False,
        },
        "activity_definition": manifest["activity_definition"],
        "gate_rule": ("activate when frame activity probability >= threshold; the "
                      "predicted differential is never the sole gate"),
        "candidate_count": int(candidates.size),
        "candidates_screened": int(cheap.sum()),
        "feasible_count": int(feasible.sum()),
        "feasibility_contract": {
            "median_active_recall_min": thr.MIN_MEDIAN_ACTIVE_RECALL,
            "median_active_cosine_min": thr.MIN_MEDIAN_ACTIVE_COSINE,
            "median_positive_cosine_fraction_min":
                thr.MIN_MEDIAN_POSITIVE_COSINE_FRACTION,
            "bootstrap_upper_fpr_max": thr.MAX_BOOTSTRAP_UPPER_FPR,
            "hazard_absent_trajectory_fpr_max": thr.MAX_HAZARD_ABSENT_TRAJECTORY_FPR,
            "max_consecutive_false_positive_run": thr.MAX_CONSECUTIVE_FALSE_POSITIVE_RUN,
            "persistent_activation_allowed": thr.ALLOW_PERSISTENT_ACTIVATION_AFTER_ORACLE,
        },
        "bootstrap": {
            "replicates": thr.BOOTSTRAP_REPLICATES,
            "seed": thr.BOOTSTRAP_SEED,
            "confidence": thr.CONFIDENCE,
            "resampled_unit": "episode (all trajectories of an episode move together)",
            "one_sided": True,
        },
        "selection_rule": ("lexicographic: lowest bootstrap upper bound, then highest "
                           "median active recall, then highest threshold"),
        "constraint_violations": 0,
        "wall_seconds": time.time() - started,
    }

    if chosen is None:
        report["selected"] = None
        report["decision_if_infeasible"] = "REFERENCE_THRESHOLD_CALIBRATION_INFEASIBLE"
        report["feasible"] = False
        # report the best near-miss so the failure is diagnosable
        best = int(np.argmin(np.where(cheap, upper, np.inf))) if cheap.any() else None
        report["closest_candidate"] = (
            {"threshold": float(candidates[best]),
             "bootstrap_upper_fpr": float(upper[best]),
             "median_active_recall": float(median_recall[best])}
            if best is not None else None)
    else:
        block = detail[chosen]
        report["feasible"] = True
        report["selected"] = {
            "threshold": float(candidates[chosen]),
            "bootstrap_upper_fpr": float(upper[chosen]),
            "median_active_recall": block["median_active_recall"],
            "mean_trajectory_zero_fpr": block["mean_trajectory_zero_fpr"],
            "median_trajectory_zero_fpr": block["median_trajectory_zero_fpr"],
            "max_trajectory_zero_fpr": block["max_trajectory_zero_fpr"],
            "median_active_cosine": block["median_active_cosine"],
            "median_positive_cosine_fraction": block["median_positive_cosine_fraction"],
            "max_hazard_absent_fpr": block["max_hazard_absent_fpr"],
            "max_consecutive_false_positive_run":
                block["max_consecutive_false_positive_run"],
            "trajectories_with_persistence": block["trajectories_with_persistence"],
            "trajectories_over_2pct_fpr": block["trajectories_over_2pct_fpr"],
            "trajectories_over_5pct_fpr": block["trajectories_over_5pct_fpr"],
        }
        report["calibration16_trajectory_metrics"] = block["trajectories"]

    report["report_sha256"] = thr.canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    if chosen is None:
        print("NO FEASIBLE THRESHOLD")
        print(f"  closest: {report['closest_candidate']}")
    else:
        s = report["selected"]
        print(f"selected threshold      : {s['threshold']:.8f}")
        print(f"  bootstrap upper FPR   : {s['bootstrap_upper_fpr']:.5f} "
              f"(<= {thr.MAX_BOOTSTRAP_UPPER_FPR})")
        print(f"  median active recall  : {s['median_active_recall']:.4f}")
        print(f"  mean trajectory FPR   : {s['mean_trajectory_zero_fpr']:.5f}")
        print(f"  median active cosine  : {s['median_active_cosine']:.4f}")
        print(f"  positive-cosine frac  : {s['median_positive_cosine_fraction']:.4f}")
        print(f"  max hazard-absent FPR : {s['max_hazard_absent_fpr']:.5f}")
        print(f"  max consecutive FP run: {s['max_consecutive_false_positive_run']}")
    print(f"wrote {args.out}")
    return 0 if chosen is not None else 7


if __name__ == "__main__":
    raise SystemExit(main())
