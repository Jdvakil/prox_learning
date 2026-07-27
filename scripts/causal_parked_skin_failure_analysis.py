#!/usr/bin/env python3
"""Attribute the outcome to a specific cause rather than a general disappointment.

Handoff step 15. Each hypothesis below is scored against evidence already recorded in the
final training report, so the diagnosis is reproducible and does not depend on narrative.
Hypotheses are evaluated whether or not the gates passed: knowing *why* a passing model
passes is as useful as knowing why a failing one fails, and a model that passes for the
wrong reason (predicting the majority class, say) should be visible here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin.data import SOURCE_MODES, sensor_link_ids
from causal_parked_skin.metrics import summarize_seeds
from causal_parked_skin.model import (
    BASELINE_CURRENT,
    BASELINE_FULL,
    BASELINE_QPOS,
    BASELINE_ZERO,
)

TEST = "offline_reference_test"


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def runs_for(report: dict, variant: str) -> list:
    return [v for v in report["runs"].values() if v["variant"] == variant]


def mean_metric(runs, path, partition=TEST):
    values = []
    for run in runs:
        block = run["metrics"][partition]
        for key in path:
            block = block.get(key) if isinstance(block, dict) else None
            if block is None:
                break
        if isinstance(block, (int, float)):
            values.append(float(block))
    return float(np.mean(values)) if values else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final-training", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    final = json.loads(args.final_training.read_text())
    stack = json.loads(args.stack.read_text())
    sensor_names = list(stack["sensor_contract"]["ordered_names"])
    link_ids, links = sensor_link_ids(sensor_names)

    full = runs_for(final, BASELINE_FULL)
    current_frame = runs_for(final, BASELINE_CURRENT)
    qpos_only = runs_for(final, BASELINE_QPOS)
    zero = final["baselines"][BASELINE_ZERO]["metrics"][TEST]
    zero_mae = zero["head"]["differential_mae"]

    full_mae = mean_metric(full, ["head", "differential_mae"])
    current_mae = mean_metric(current_frame, ["head", "differential_mae"])
    qpos_mae = mean_metric(qpos_only, ["head", "differential_mae"])

    # ---- per-sensor and per-link attribution -------------------------------------
    per_sensor = np.array([run["metrics"][TEST]["per_sensor_mae"] for run in full],
                          dtype=np.float64)
    sensor_mean = np.nanmean(per_sensor, axis=0)
    per_link = {}
    for index, name in enumerate(links):
        hits = link_ids == index
        per_link[name] = {"sensors": int(hits.sum()),
                          "field_mae": float(np.nanmean(sensor_mean[hits]))}
    worst_sensors = sorted(
        ({"sensor": sensor_names[i], "link": links[link_ids[i]],
          "field_mae": float(sensor_mean[i])} for i in range(len(sensor_names))),
        key=lambda r: -r["field_mae"])[:8]

    # ---- hypotheses ---------------------------------------------------------------
    active_recall = mean_metric(full, ["activation", "oracle_active_recall"])
    false_positive = mean_metric(
        full, ["activation", "oracle_zero_false_positive_rate"])
    cosine = mean_metric(full, ["head", "median_direction_cosine_active"])
    field_mae = mean_metric(full, ["pixel", "all_valid_parked_mae"])
    mask_f1 = mean_metric(full, ["mask", "f1"])
    mask_auprc = mean_metric(full, ["mask", "auprc"])
    prevalence = mean_metric(full, ["mask", "prevalence"])
    zero_field_mae = zero["pixel"]["all_valid_parked_mae"]

    improvement_over_zero = 1.0 - full_mae / zero_mae if zero_mae else None
    history_gain = 1.0 - full_mae / current_mae if current_mae else None
    proximity_gain = 1.0 - current_mae / qpos_mae if qpos_mae else None

    seed_spread = summarize_seeds(
        [run["metrics"][TEST]["head"]["differential_mae"] for run in full])
    threshold_spread = summarize_seeds(
        [run["calibration_threshold"]["threshold"] for run in final["runs"].values()
         if run["variant"] == BASELINE_FULL])

    source_mode_cosines = {}
    source_mode_mae = {}
    for name in SOURCE_MODES:
        values = [run["metrics"][TEST]["by_source_mode"][name]
                  for run in full
                  if run["metrics"][TEST]["by_source_mode"][name].get("available")]
        if not values:
            source_mode_cosines[name] = None
            source_mode_mae[name] = None
            continue
        cosines = [v["median_direction_cosine_active"] for v in values
                   if v["median_direction_cosine_active"] is not None]
        source_mode_cosines[name] = float(np.mean(cosines)) if cosines else None
        source_mode_mae[name] = float(np.mean([v["differential_mae"] for v in values]))

    available_modes = [m for m, v in source_mode_mae.items() if v is not None]
    mode_spread = (max(source_mode_mae[m] for m in available_modes)
                   / min(source_mode_mae[m] for m in available_modes)
                   if len(available_modes) > 1 else None)

    hypotheses = [
        {
            "hypothesis": "signal_not_learnable_from_causal_inputs",
            "supported": bool(improvement_over_zero is not None
                              and improvement_over_zero < 0.05),
            "evidence": {"offline_mae": full_mae, "zero_baseline_mae": zero_mae,
                         "relative_improvement": improvement_over_zero},
            "reading": ("a model that cannot separate itself from predicting 'no obstacle "
                        "ever' has not extracted the counterfactual from its inputs"),
        },
        {
            "hypothesis": "model_predicts_the_majority_zero_class",
            "supported": bool(active_recall is not None and active_recall < 0.10),
            "evidence": {"oracle_active_recall": active_recall,
                         "oracle_zero_false_positive_rate": false_positive,
                         "field_mae_vs_zero_baseline_field_mae":
                             {"model": field_mae, "zero": zero_field_mae}},
            "reading": ("near-zero recall with a near-zero false-positive rate is the "
                        "signature of a model that has learned to stay silent"),
        },
        {
            "hypothesis": "static_state_sufficient_history_unnecessary",
            "supported": bool(history_gain is not None and history_gain < 0.10),
            "evidence": {"full_causal_mae": full_mae,
                         "current_frame_only_mae": current_mae,
                         "history_relative_gain": history_gain,
                         "qpos_only_mae": qpos_mae,
                         "proximity_relative_gain_over_qpos_only": proximity_gain},
            "reading": ("if four frames do not beat one, the obstacle signature is "
                        "instantaneous rather than temporal"),
        },
        {
            "hypothesis": "learner_induced_distribution_causes_failure",
            "supported": False,
            "evidence": {"learner_induced_present_in_offline_test": False,
                         "offline_test_source_modes": available_modes},
            "reading": ("not evaluable: the learner-induced distribution exists only in "
                        "the train partition, so its generalization cannot be measured "
                        "here at all. This is a coverage limitation of the frozen "
                        "dataset, not a finding about the model."),
        },
        {
            "hypothesis": "oracle_active_events_too_sparse",
            "supported": bool(prevalence is not None and prevalence < 1e-4),
            "evidence": {"changed_pixel_prevalence": prevalence,
                         "oracle_active_frame_prevalence_offline":
                             zero.get("frames") and None,
                         "mask_auprc": mask_auprc,
                         "mask_f1": mask_f1},
            "reading": "sparsity of the positive class relative to what the mask head sees",
        },
        {
            "hypothesis": "field_accurate_but_head_space_output_wrong",
            "supported": bool(field_mae is not None and zero_field_mae
                              and field_mae <= zero_field_mae
                              and improvement_over_zero is not None
                              and improvement_over_zero < 0.25),
            "evidence": {"field_mae": field_mae, "zero_field_mae": zero_field_mae,
                         "head_relative_improvement": improvement_over_zero,
                         "direction_cosine": cosine},
            "reading": ("the frozen head amplifies small field errors; pixel accuracy "
                        "does not automatically become differential accuracy"),
        },
        {
            "hypothesis": "head_space_accurate_but_localization_weak",
            "supported": bool(improvement_over_zero is not None
                              and improvement_over_zero >= 0.25
                              and mask_f1 is not None and mask_f1 < 0.2),
            "evidence": {"head_relative_improvement": improvement_over_zero,
                         "mask_f1": mask_f1, "mask_auprc": mask_auprc,
                         "changed_pixel_prevalence": prevalence},
            "reading": ("the 7-D output can be right while the model is wrong about "
                        "which pixels moved, because the head pools over the field"),
        },
        {
            "hypothesis": "source_mode_distribution_shift",
            "supported": bool(mode_spread is not None and mode_spread > 2.0),
            "evidence": {"differential_mae_by_source_mode": source_mode_mae,
                         "direction_cosine_by_source_mode": source_mode_cosines,
                         "worst_over_best_mae_ratio": mode_spread},
            "reading": "error concentrated in one driving distribution rather than spread",
        },
        {
            "hypothesis": "calibration_instability",
            "supported": bool(
                threshold_spread["coefficient_of_variation"] is not None
                and threshold_spread["coefficient_of_variation"] > 0.5),
            "evidence": {"calibrated_threshold_across_seeds": threshold_spread,
                         "offline_false_positive_rate": false_positive,
                         "seed_spread_of_offline_mae": seed_spread},
            "reading": ("a threshold that moves sharply between seeds will not transfer "
                        "to a live run"),
        },
    ]

    report = {
        "schema": "causal_parked_skin_failure_analysis_v1",
        "primary_variant": BASELINE_FULL,
        "summary": {
            "offline_mae": {"full_causal": full_mae, "current_frame_only": current_mae,
                            "qpos_only": qpos_mae, "zero_differential": zero_mae},
            "relative_improvement_over_zero": improvement_over_zero,
            "history_relative_gain_over_current_frame": history_gain,
            "proximity_relative_gain_over_qpos_only": proximity_gain,
            "median_direction_cosine_active": cosine,
            "oracle_active_recall": active_recall,
            "oracle_zero_false_positive_rate": false_positive,
            "seed_spread": seed_spread,
        },
        "hypotheses": hypotheses,
        "supported_hypotheses": [h["hypothesis"] for h in hypotheses if h["supported"]],
        "per_link_field_mae": per_link,
        "worst_sensors_by_field_mae": worst_sensors,
        "per_sensor_field_mae": {sensor_names[i]: float(sensor_mean[i])
                                 for i in range(len(sensor_names))},
        "source_mode_coverage": {
            "offline_test_modes": available_modes,
            "modes_absent_from_offline_test": [m for m in SOURCE_MODES
                                               if m not in available_modes],
        },
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"offline MAE  full={full_mae:.6f} current={current_mae:.6f} "
          f"qpos={qpos_mae:.6f} zero={zero_mae:.6f}")
    print(f"improvement over zero : {improvement_over_zero:.3f}")
    print(f"history gain          : {history_gain:.3f}")
    print(f"proximity gain        : {proximity_gain:.3f}")
    print("supported hypotheses:")
    for name in report["supported_hypotheses"]:
        print(f"  - {name}")
    print("per-link field MAE:")
    for name, block in sorted(per_link.items()):
        print(f"  {name:<12} {block['field_mae']:.6f}  ({block['sensors']} sensors)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
