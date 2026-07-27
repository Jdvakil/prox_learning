#!/usr/bin/env python3
"""Robustness audit of the selected threshold, plus frozen-inference repeatability.

Handoff steps 9 and 10.

Step 9 re-scores the selected threshold on the 20-episode offline set that was already
opened during the previous task. Every metric from that set is labelled
``reused_nonconfirmatory_diagnostic``: its 2.15% result is what motivated this task, so it
can still *block* live testing, but passing it is not an independent generalization claim.
The genuinely held-out check is the live development4 run.

Step 10 repeats inference on a fixed batch. Training was nondeterministic -- the same
configuration and seed moved validation MAE by 15% between runs -- so the claim that the
*frozen* forward pass is stable has to be measured rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin import engine
from causal_parked_skin import threshold as thr
from causal_parked_skin.data import load_partition
from causal_parked_skin.engine import load_checkpoint, make_batch
from causal_parked_skin.model import FrozenSafetyHead

sys.path.insert(0, str(ROOT / "scripts"))
from hybrid_obstacle_reference_threshold_calibrate import (
    build_trajectories,
    evaluate_at,
    score_partition,
)

DIAGNOSTIC_PARTITION = "offline_reference_test"
DIAGNOSTIC_LABEL = "reused_nonconfirmatory_diagnostic"
INFERENCE_REPEATS = 24
INFERENCE_TOLERANCE = 1e-7


def inference_repeatability(model, partition, device, repeats=INFERENCE_REPEATS) -> dict:
    """Repeat the frozen forward pass on one fixed batch and measure drift."""
    import torch

    index = np.arange(256)
    batch = make_batch(partition, index, device)
    activity_runs, parked_runs, delta_runs = [], [], []
    model.eval()
    with torch.no_grad():
        for _ in range(repeats):
            out = model(batch["history"], batch["history_valid"], batch["state"])
            probability = out["changed_probability"]
            activity_runs.append(probability.reshape(
                probability.shape[0], -1).amax(dim=1).double().cpu().numpy())
            parked_runs.append(out["parked"].double().cpu().numpy())
            delta_runs.append(out["delta"].double().cpu().numpy())

    activity = np.stack(activity_runs)
    parked = np.stack(parked_runs)
    delta = np.stack(delta_runs)
    activity_delta = float(np.abs(activity - activity[0]).max())
    parked_delta = float(np.abs(parked - parked[0]).max())
    delta_delta = float(np.abs(delta - delta[0]).max())
    return {
        "repeats": repeats,
        "batch_frames": len(index),
        "activity_bit_identical": bool(activity_delta == 0.0),
        "activity_max_abs_delta": activity_delta,
        "parked_field_max_abs_delta": parked_delta,
        "predicted_delta_max_abs_delta": delta_delta,
        "tolerance": INFERENCE_TOLERANCE,
        "stable": bool(activity_delta <= INFERENCE_TOLERANCE
                       and parked_delta <= INFERENCE_TOLERANCE
                       and delta_delta <= INFERENCE_TOLERANCE),
        "training_kernels_invoked": False,
    }


def head_repeatability(model, partition, head, device, repeats=INFERENCE_REPEATS) -> dict:
    """The 7-D differential must be as stable as the field it comes from."""
    import torch

    index = np.arange(256)
    batch = make_batch(partition, index, device)
    runs = []
    with torch.no_grad():
        for _ in range(repeats):
            out = model(batch["history"], batch["history_valid"], batch["state"])
            runs.append((head(out["current"]) - head(out["parked"])
                         ).double().cpu().numpy())
    stacked = np.stack(runs)
    drift = float(np.abs(stacked - stacked[0]).max())
    return {"repeats": repeats, "head_differential_max_abs_delta": drift,
            "stable": bool(drift <= INFERENCE_TOLERANCE),
            "tolerance": INFERENCE_TOLERANCE}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch

    stack = json.loads(args.stack.read_text())
    engine.set_sensor_names(stack["sensor_contract"]["ordered_names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = FrozenSafetyHead.load(args.safety_dir, device=device)
    model, _payload = load_checkpoint(args.checkpoint, device)

    calibration = json.loads(args.calibration.read_text())
    if not calibration.get("feasible"):
        raise SystemExit("calibration reported no feasible threshold")
    selected = float(calibration["selected"]["threshold"])
    print(f"auditing threshold {selected:.8f}")

    partition = load_partition(args.cache, DIAGNOSTIC_PARTITION)
    scored = score_partition(model, partition, head, device)
    trajectories = build_trajectories(partition, scored)
    episodes = sorted({t.episode_id for t in trajectories})
    block = evaluate_at(trajectories, selected)

    # cluster bootstrap on the diagnostic set too, for a comparable interval
    cluster_of = {episode: i for i, episode in enumerate(episodes)}
    cluster_of_row = np.array([cluster_of[t.episode_id] for t in trajectories])
    fpr = thr.trajectory_fpr_matrix(trajectories, np.array([selected]))
    recall = thr.trajectory_recall_matrix(trajectories, np.array([selected]))
    upper_fpr = float(thr.cluster_bootstrap_upper_bound(
        thr.cluster_means(fpr, cluster_of_row, len(episodes)))[0])
    with np.errstate(invalid="ignore"):
        lower_recall = float(np.quantile(
            np.nanmean(thr.cluster_means(recall, cluster_of_row, len(episodes))[
                np.random.default_rng(thr.BOOTSTRAP_SEED).integers(
                    0, len(episodes), size=(thr.BOOTSTRAP_REPLICATES, len(episodes)))],
                axis=1), 1.0 - thr.CONFIDENCE))

    hazard_absent = [r for r in block["trajectories"] if not r["hazard_present"]]
    hazard_present = [r for r in block["trajectories"] if r["hazard_present"]]
    active_fraction_absent = [
        (r["false_positive_frames"] + r["retained_active_frames"]) / r["frames"]
        for r in hazard_absent]

    failures = []
    mean_fpr = block["mean_trajectory_zero_fpr"]
    if mean_fpr is not None and mean_fpr > thr.DIAGNOSTIC_MAX_MEAN_ZERO_FPR:
        failures.append(f"mean trajectory zero FPR {mean_fpr:.4f} > "
                        f"{thr.DIAGNOSTIC_MAX_MEAN_ZERO_FPR}")
    if active_fraction_absent and max(active_fraction_absent) > \
            thr.DIAGNOSTIC_MAX_HAZARD_ABSENT_ACTIVE_FRACTION:
        failures.append(f"hazard-absent trajectory active fraction "
                        f"{max(active_fraction_absent):.4f} > "
                        f"{thr.DIAGNOSTIC_MAX_HAZARD_ABSENT_ACTIVE_FRACTION}")
    if block["median_active_recall"] is not None and \
            block["median_active_recall"] < thr.DIAGNOSTIC_MIN_MEDIAN_RECALL:
        failures.append(f"median active recall {block['median_active_recall']:.4f} < "
                        f"{thr.DIAGNOSTIC_MIN_MEDIAN_RECALL}")
    if block["median_active_cosine"] is not None and \
            block["median_active_cosine"] < thr.DIAGNOSTIC_MIN_MEDIAN_COSINE:
        failures.append(f"median cosine {block['median_active_cosine']:.4f} < "
                        f"{thr.DIAGNOSTIC_MIN_MEDIAN_COSINE}")
    if block["max_consecutive_false_positive_run"] > thr.DIAGNOSTIC_MAX_PERSISTENT_RUN:
        failures.append(f"false-active run "
                        f"{block['max_consecutive_false_positive_run']} > "
                        f"{thr.DIAGNOSTIC_MAX_PERSISTENT_RUN}")

    stability = inference_repeatability(model, partition, device)
    head_stability = head_repeatability(model, partition, head, device)

    report = {
        "schema": "hybrid_obstacle_reference_threshold_audit_v1",
        "threshold": selected,
        "calibration_report_sha256": calibration["report_sha256"],
        "diagnostic_set": {
            "label": DIAGNOSTIC_LABEL,
            "partition": DIAGNOSTIC_PARTITION,
            "episodes": len(episodes),
            "trajectories": len(trajectories),
            "frames": int(sum(t.frames for t in trajectories)),
            "status": ("already opened during the previous task; its 2.15% result "
                       "motivated this recalibration, so it is a consumed diagnostic "
                       "and not an untouched test"),
            "used_for_threshold_fitting": False,
            "provides_final_readiness_gate": False,
        },
        "diagnostic_metrics": {
            "median_active_recall": block["median_active_recall"],
            "mean_trajectory_zero_fpr": block["mean_trajectory_zero_fpr"],
            "median_trajectory_zero_fpr": block["median_trajectory_zero_fpr"],
            "max_trajectory_zero_fpr": block["max_trajectory_zero_fpr"],
            "bootstrap_upper_zero_fpr": upper_fpr,
            "bootstrap_lower_active_recall": lower_recall,
            "median_active_cosine": block["median_active_cosine"],
            "median_positive_cosine_fraction": block["median_positive_cosine_fraction"],
            "max_consecutive_false_positive_run":
                block["max_consecutive_false_positive_run"],
            "trajectories_with_persistence": block["trajectories_with_persistence"],
            "trajectories_over_2pct_fpr": block["trajectories_over_2pct_fpr"],
            "trajectories_over_5pct_fpr": block["trajectories_over_5pct_fpr"],
            "hazard_present_trajectories": len(hazard_present),
            "hazard_absent_trajectories": len(hazard_absent),
            "max_hazard_absent_active_fraction": (float(max(active_fraction_absent))
                                                  if active_fraction_absent else None),
            "constraint_violations": scored["constraint_violations"],
            "nonfinite_outputs": scored["nonfinite"],
        },
        "diagnostic_trajectory_metrics": block["trajectories"],
        "blocking_checks": {
            "thresholds": {
                "mean_zero_fpr_max": thr.DIAGNOSTIC_MAX_MEAN_ZERO_FPR,
                "hazard_absent_active_fraction_max":
                    thr.DIAGNOSTIC_MAX_HAZARD_ABSENT_ACTIVE_FRACTION,
                "median_recall_min": thr.DIAGNOSTIC_MIN_MEDIAN_RECALL,
                "median_cosine_min": thr.DIAGNOSTIC_MIN_MEDIAN_COSINE,
                "persistent_run_max": thr.DIAGNOSTIC_MAX_PERSISTENT_RUN,
            },
            "failures": failures,
            "passed": not failures,
        },
        "inference_stability": stability,
        "head_stability": head_stability,
        "decision_if_transfer_failed": "REFERENCE_THRESHOLD_TRANSFER_FAILED",
        "decision_if_inference_unstable": "REFERENCE_MODEL_INFERENCE_UNSTABLE",
    }
    report["report_sha256"] = thr.canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    m = report["diagnostic_metrics"]
    print(f"  [{DIAGNOSTIC_LABEL}] {len(episodes)} episodes / "
          f"{len(trajectories)} trajectories")
    print(f"  median active recall  : {m['median_active_recall']}")
    print(f"  mean trajectory FPR   : {m['mean_trajectory_zero_fpr']:.5f} "
          f"(bootstrap upper {upper_fpr:.5f})")
    print(f"  median cosine         : {m['median_active_cosine']}")
    print(f"  max hazard-absent act : {m['max_hazard_absent_active_fraction']}")
    print(f"  max consecutive FP run: {m['max_consecutive_false_positive_run']}")
    print(f"  >2% / >5% FPR traj    : {m['trajectories_over_2pct_fpr']} / "
          f"{m['trajectories_over_5pct_fpr']}")
    print(f"  blocking checks passed: {report['blocking_checks']['passed']}")
    for failure in failures:
        print(f"    FAIL {failure}")
    print(f"  inference stable      : {stability['stable']} "
          f"(activity delta {stability['activity_max_abs_delta']:.3e}, "
          f"parked {stability['parked_field_max_abs_delta']:.3e}, "
          f"head {head_stability['head_differential_max_abs_delta']:.3e})")
    print(f"wrote {args.out}")

    if not stability["stable"] or not head_stability["stable"]:
        return 8
    return 0 if report["blocking_checks"]["passed"] else 9


if __name__ == "__main__":
    raise SystemExit(main())
