#!/usr/bin/env python3
"""Covariate-shift audit: V1 against V2 on the same on-policy states.

Handoff step 11. Both models are evaluated on identical frames from the disjoint
validation and offline-test trajectories, so every difference is the model's.

Reports feature-space distance to each model's own training support, predicted and oracle
norms, norm ratio, cosine, zero-frame activation, and error broken down by policy
condition, task phase, grasp contact, and by whether the controller had already changed
the trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT / "scripts"), str(ROOT / "submodules" / "act")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from deployable_reference import (
    FEATURE_BUILDERS,
    FEATURE_FIELDS,
    MLP_REFERENCE_ID,
    SupportEnvelopeGate,
    SupportGate,
    load_reference,
)
from hybrid_obstacle_train_on_policy_reference import (
    ORACLE_ZERO_TOLERANCE,
    cosine,
    load_expert,
    load_on_policy,
    stack,
)


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def support_distance(features: np.ndarray, standardizer) -> np.ndarray:
    """Mahalanobis-like distance in the model's own standardized feature space.

    A frame far from the training mean in units of the training standard deviation is a
    frame the model was never asked about. Using each model's *own* statistics is the
    point: it measures how far the deployment states are from what that model saw.
    """
    scaled = standardizer(features)
    return np.linalg.norm(scaled, axis=1) / np.sqrt(scaled.shape[1])


def evaluate_model(name: str, model, gate, data: dict[str, np.ndarray]) -> dict[str, Any]:
    features = FEATURE_BUILDERS[MLP_REFERENCE_ID](
        {k: data[k] for k in FEATURE_FIELDS[MLP_REFERENCE_ID]})
    predicted_dq = (data["current_head"]
                    - np.asarray(model.predict(features), dtype=np.float32)).astype(np.float32)
    # V1's gate takes the scene minimum depth as its support condition; V2's does not.
    if isinstance(gate, SupportGate):
        depths = np.asarray(data["minimum_depth"], dtype=np.float64)
        executed = np.stack([gate(predicted_dq[i], float(depths[i]))[0]
                             for i in range(len(predicted_dq))])
    else:
        executed = np.stack([gate(predicted_dq[i])[0] for i in range(len(predicted_dq))])
    oracle = data["privileged_oracle_dq"]
    oracle_norm = np.asarray(data["privileged_oracle_norm"], dtype=np.float64)
    active = oracle_norm > ORACLE_ZERO_TOLERANCE
    zero = ~active
    activated = np.linalg.norm(executed, axis=1) > 0
    cosines = cosine(executed, oracle)
    finite = np.isfinite(cosines)
    ratios = np.divide(np.linalg.norm(executed, axis=1), oracle_norm,
                       out=np.full(len(oracle_norm), np.nan), where=oracle_norm > 1e-12)
    distance = support_distance(features, model.standardizer)
    grasped = data["gripper_state"][:, 0] < np.median(data["gripper_state"][:, 0])
    # "after the controller changed the trajectory": frames in a rollout that has already
    # activated at least once
    changed = np.zeros(len(activated), dtype=bool)
    for row_index in np.unique(data["row_index"]):
        mask = data["row_index"] == row_index
        indices = np.flatnonzero(mask)
        first = next((i for i in indices if activated[i]), None)
        if first is not None:
            changed[indices[indices > first]] = True

    def error_on(mask) -> float | None:
        return (float(np.mean(np.abs(executed[mask] - oracle[mask]))) if mask.any()
                else None)

    phase = np.digitize(data["row_index"] * 0 + np.arange(len(activated)) %
                        max(int(np.bincount(data["row_index"]).max()), 1), [67, 134])

    return {
        "model": name,
        "frames": len(oracle_norm),
        "feature_distance_to_training_support": {
            "median": float(np.median(distance)), "p95": float(np.percentile(distance, 95)),
            "max": float(distance.max())},
        "predicted_norm": {"median": float(np.median(np.linalg.norm(predicted_dq, axis=1))),
                           "p99": float(np.percentile(np.linalg.norm(predicted_dq, axis=1),
                                                      99))},
        "oracle_norm": {"median": float(np.median(oracle_norm[active])) if active.any()
                        else None,
                        "p99": float(np.percentile(oracle_norm[active], 99)) if active.any()
                        else None},
        "median_norm_ratio": (float(np.nanmedian(ratios[active & activated]))
                              if (active & activated).any() else None),
        "norm_ratio_error": (float(np.nanmedian(np.abs(ratios[active & activated] - 1.0)))
                             if (active & activated).any() else None),
        "median_cosine": (float(np.median(cosines[active & activated & finite]))
                          if (active & activated & finite).any() else None),
        "positive_cosine_fraction": (
            float(np.mean(cosines[active & activated & finite] > 0))
            if (active & activated & finite).any() else None),
        "oracle_zero_activation_rate": (float(np.mean(activated[zero])) if zero.any()
                                        else None),
        # Threshold-free quiet comparison. The gated activation rate is only meaningful
        # when both models have a real activation contract; the raw predicted norm on
        # oracle-zero frames is comparable regardless.
        "oracle_zero_predicted_norm": {
            "median": float(np.median(np.linalg.norm(predicted_dq[zero], axis=1)))
            if zero.any() else None,
            "p95": float(np.percentile(np.linalg.norm(predicted_dq[zero], axis=1), 95))
            if zero.any() else None,
            "rms": float(np.sqrt(np.mean(np.linalg.norm(predicted_dq[zero], axis=1) ** 2)))
            if zero.any() else None},
        "differential_mae_active": error_on(active),
        "differential_mae_all": error_on(np.ones(len(active), dtype=bool)),
        "error_by_phase": {f"third_{k}": error_on(phase == k) for k in (0, 1, 2)},
        "error_during_grasp_contact": error_on(grasped),
        "error_after_controller_changed_trajectory": error_on(changed),
        "error_before_controller_changed_trajectory": error_on(~changed),
        "activation_rate": float(np.mean(activated)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--labelling-schedule", required=True, type=Path)
    ap.add_argument("--expert-paired-dir", required=True, type=Path)
    ap.add_argument("--v1-manifest", required=True, type=Path)
    ap.add_argument("--v2-manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    for name in ("partition", "labelling_schedule", "expert_paired_dir", "v1_manifest",
                 "v2_manifest", "out"):
        setattr(args, name, Path(getattr(args, name)).resolve())

    partition = json.loads(args.partition.read_text())
    labelling = json.loads(args.labelling_schedule.read_text())
    v1_model, v1_gate, v1_manifest = load_reference(args.v1_manifest, device="cpu")
    v2_model, v2_gate, v2_manifest = load_reference(args.v2_manifest, device="cpu")
    if not isinstance(v1_gate, SupportGate) or not isinstance(v2_gate, SupportEnvelopeGate):
        raise SystemExit("unexpected gate types; V1 must be the depth gate and V2 the "
                         "support envelope")

    results: dict[str, Any] = {}
    for set_name in ("reference_validation", "offline_reference_test"):
        ids = [r["episode_id"] for r in partition["partitions"][set_name]]
        per_distribution = {}
        for distribution, rows in (
            ("expert", load_expert(args.expert_paired_dir, ids)),
            ("act_only_on_policy",
             load_on_policy(labelling, "act_only_on_policy", set(ids))),
            ("oracle_on_policy",
             load_on_policy(labelling, "oracle_on_policy", set(ids))),
        ):
            if not rows:
                continue
            data = stack(rows)
            per_distribution[distribution] = {
                "V1": evaluate_model("V1", v1_model, v1_gate, data),
                "V2": evaluate_model("V2", v2_model, v2_gate, data),
            }
        results[set_name] = per_distribution

    # ---- the predeclared improvement requirements ------------------------- #
    def pooled_nested(model_key: str, metric: str, field: str,
                      distribution: str) -> float | None:
        values = [results[s][distribution][model_key][metric][field]
                  for s in results if distribution in results[s]
                  and results[s][distribution][model_key][metric][field] is not None]
        return float(np.mean(values)) if values else None

    def pooled(model_key: str, metric: str, distribution: str) -> float | None:
        values = [results[s][distribution][model_key][metric]
                  for s in results if distribution in results[s]
                  and results[s][distribution][model_key][metric] is not None]
        return float(np.mean(values)) if values else None

    diagnostic_v2_gate = bool(v2_manifest.get("diagnostic_only"))
    improvements = {}
    for label, metric, distribution, lower_is_better in (
        ("act_only_on_policy_differential_mae", "differential_mae_active",
         "act_only_on_policy", True),
        ("oracle_on_policy_differential_mae", "differential_mae_active",
         "oracle_on_policy", True),
        ("oracle_zero_false_activation", "oracle_zero_activation_rate",
         "act_only_on_policy", True),
        ("norm_ratio_error", "norm_ratio_error", "act_only_on_policy", True),
    ):
        v1_value = pooled("V1", metric, distribution)
        v2_value = pooled("V2", metric, distribution)
        entry = {"v1": v1_value, "v2": v2_value, "distribution": distribution,
                 "metric": metric}
        if metric == "oracle_zero_activation_rate" and diagnostic_v2_gate:
            # V2 has no feasible activation contract, so its gated activation rate is an
            # artefact of the diagnostic tau=0 and is not comparable to V1's. The
            # threshold-free quiet statistic below is what can honestly be compared.
            entry.update({
                "improved": None, "evaluable": False,
                "why_not_evaluable": ("V2 has no feasible tau, so a gated activation rate "
                                      "is an artefact of the diagnostic gate"),
                "threshold_free_v1_oracle_zero_predicted_norm_median":
                    pooled_nested("V1", "oracle_zero_predicted_norm", "median",
                                  distribution),
                "threshold_free_v2_oracle_zero_predicted_norm_median":
                    pooled_nested("V2", "oracle_zero_predicted_norm", "median",
                                  distribution)})
        else:
            entry["improved"] = bool(
                v1_value is not None and v2_value is not None
                and (v2_value < v1_value if lower_is_better else v2_value > v1_value))
            entry["evaluable"] = True
        improvements[label] = entry

    report = {
        "schema": "hybrid_obstacle_on_policy_shift_audit_v2",
        "v1_manifest_sha256": v1_manifest["manifest_sha256"],
        "v2_manifest_sha256": v2_manifest["manifest_sha256"],
        "v1_gate": {"type": "SupportGate", "tau": v1_manifest["tau"],
                    "d_act": v1_manifest.get("d_act")},
        "v2_gate": {"type": "SupportEnvelopeGate", "tau": v2_manifest["tau"],
                    "rho_max": v2_manifest["rho_max"]},
        "evaluated_on": "identical frames from the disjoint validation and offline-test "
                        "trajectories",
        "results": results,
        "required_improvements": improvements,
        "all_evaluable_required_improvements_met": all(
            v["improved"] for v in improvements.values() if v.get("evaluable")),
        "improvements_not_evaluable": [k for k, v in improvements.items()
                                       if not v.get("evaluable")],
        "v2_gate_is_diagnostic_only": diagnostic_v2_gate,
        "caveat": ("offline improvement alone does not establish that the architecture is "
                   "adequate; the live development evaluation is the test that matters"),
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    for set_name, distributions in results.items():
        print(f"\n{set_name}")
        for distribution, models in distributions.items():
            v1, v2 = models["V1"], models["V2"]
            print(f"  {distribution:<20} MAE {v1['differential_mae_active']} -> "
                  f"{v2['differential_mae_active']} | cos {v1['median_cosine']} -> "
                  f"{v2['median_cosine']} | zeroFA {v1['oracle_zero_activation_rate']} -> "
                  f"{v2['oracle_zero_activation_rate']}")
    print("\nrequired improvements:")
    for label, info in improvements.items():
        mark = ("n/a " if not info.get("evaluable")
                else "PASS" if info["improved"] else "FAIL")
        print(f"  [{mark}] {label}: {info['v1']} -> {info['v2']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
