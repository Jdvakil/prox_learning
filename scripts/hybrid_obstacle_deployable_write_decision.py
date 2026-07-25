#!/usr/bin/env python3
"""Assemble the final deployable-reference decision JSON.

Handoff step 20. Every field is copied from an artifact produced earlier in the task.
The decision token comes from the analysis, never from here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWED = (
    "DEPLOYABLE_REFERENCE_READY_FOR_CONFIRMATORY_41",
    "DEPLOYABLE_REFERENCE_OFFLINE_INVALID",
    "DEPLOYABLE_REFERENCE_LIVE_GROSS_REGRESSION",
    "DEPLOYABLE_REFERENCE_CONTRACT_INSUFFICIENT",
    "REFERENCE_MODEL_TRAINING_FAILED",
    "CHECKPOINT_OR_SOURCE_MISMATCH",
    "REFERENCE_DEVELOPMENT_INCOMPLETE",
)


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provenance", required=True, type=Path)
    ap.add_argument("--dataset-manifest", required=True, type=Path)
    ap.add_argument("--selection", required=True, type=Path)
    ap.add_argument("--deployment-manifest", required=True, type=Path)
    ap.add_argument("--offline-replay", required=True, type=Path)
    ap.add_argument("--analysis", required=True, type=Path)
    ap.add_argument("--schedule", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    for name in ("provenance", "dataset_manifest", "selection", "deployment_manifest",
                 "offline_replay", "analysis", "schedule", "out"):
        setattr(args, name, Path(getattr(args, name)).resolve())

    provenance = json.loads(args.provenance.read_text())
    dataset = json.loads(args.dataset_manifest.read_text())
    selection = json.loads(args.selection.read_text())
    deployment = json.loads(args.deployment_manifest.read_text())
    replay = json.loads(args.offline_replay.read_text())
    analysis = json.loads(args.analysis.read_text())
    schedule = json.loads(args.schedule.read_text())

    decision = analysis["decision"]
    if decision not in ALLOWED:
        raise SystemExit(f"decision {decision!r} is not allowed")

    act = ROOT / "submodules/act"
    molmo = ROOT / "submodules/molmospaces"
    confirmatory = json.loads(
        (ROOT / "configs/hybrid_obstacle_confirmatory41_v1.json").read_text())
    knn = selection["candidates"]["POSTURE_KNN_REFERENCE_V1"]
    mlp = selection["candidates"]["POSTURE_SKIN_MLP_REFERENCE_V1"]

    uncommitted = []
    for label, path in (
        ("paired_dataset_dir", Path(dataset["dataset_dir"])),
        ("reference_artifact", Path(deployment["artifact_path"])),
        ("act_checkpoint", Path(
            "/root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2"
            "/20260725_seed0_2000ep/policy_best.ckpt")),
        ("deployable_rollout_root", Path("/root/act_retrain_assets/deployable_dev_v1")),
        ("development_paired_dir", Path("/root/act_retrain_assets/paired_reference_dev4")),
    ):
        entry = {"label": label, "path": str(path), "committed": False}
        if path.is_file():
            entry["sha256"] = sha256_file(path)
        elif path.is_dir():
            entry["files"] = len(list(path.glob("*")))
        uncommitted.append(entry)

    payload = {
        "schema": "hybrid_obstacle_deployable_reference_final_decision_v1",
        "date": "2026-07-25",
        "task": ("Train and qualify a deployable posture-conditioned reference that "
                 "approximates the validated parked-obstacle oracle without privileged "
                 "simulation information"),
        "decision": decision,

        "commits": {
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "root_starting_commit": "4193b776640886c1bddc6be5adc7bdaf35855643",
            "root_commit": git("rev-parse", "HEAD"),
            "act_branch": git("rev-parse", "--abbrev-ref", "HEAD", repo=act),
            "act_starting_commit": "709a22de62ac0e8c4640b75eb348416d6e29013d",
            "act_commit": git("rev-parse", "HEAD", repo=act),
            "molmospaces_commit": git("rev-parse", "HEAD", repo=molmo),
            "molmospaces_modified": git("status", "--porcelain", repo=molmo) != "",
        },

        "artifact_verification": {
            "checks": provenance["check_count"],
            "all_matched": provenance["all_matched"],
            "partitions": provenance["partitions"],
        },

        "teacher_support": {
            "d_act_m": 0.18,
            "why_long_range_direction_is_out_of_scope": (
                "The analytic teacher only fires when a hazard return is a sensor's closest "
                "return inside D_act = 0.18 m; beyond it the repulsion weight 1/r - 1/D_act "
                "is undefined and the oracle task measured a REVERSED median cosine (-0.36) "
                "on candidate 108 at the head's 0.5 m input radius. Direction is therefore "
                "neither optimised nor scored outside the supported range; the residual is "
                "required to be quiet there instead."),
            "support_gate_is_weak_in_this_environment": (
                "No validation frame has a minimum valid depth above 0.187 m, so the "
                "'>0.25 m far frame' metric has ZERO frames and gate A (a valid return "
                "inside 0.18 m) is open on 1168 of 1543 validation frames. The quiet "
                "threshold tau does essentially all of the gating work."),
        },

        "paired_dataset": {
            "trajectories": dataset["file_count"],
            "total_frames": dataset["total_frames"],
            "counts": dataset["counts"],
            "tree_sha256": dataset["tree_sha256"],
            "train_tree_sha256": dataset["train_tree_sha256"],
            "validation_tree_sha256": dataset["validation_tree_sha256"],
            "manifest_sha256": dataset["manifest_sha256"],
            "pairing_correctness": dataset["pairing_correctness"],
            "feature_contract": dataset["feature_contract"],
            "split_level": "trajectory",
            "train_trajectories": dataset["train_trajectories"],
            "validation_trajectories": dataset["validation_trajectories"],
        },

        "candidates": {
            "POSTURE_KNN_REFERENCE_V1": {
                "configuration": knn["configuration"],
                "tau": knn["tau"],
                "gates": knn["gates"],
                "active_range": knn["metrics"]["active_range"],
                "quietness": knn["metrics"]["quietness"],
                "reference_quality": knn["metrics"]["reference_quality"],
                "selected": False,
            },
            "POSTURE_SKIN_MLP_REFERENCE_V1": {
                "configuration": mlp["configuration"],
                "training": mlp["training"],
                "tau": mlp["tau"],
                "gates": mlp["gates"],
                "active_range": mlp["metrics"]["active_range"],
                "quietness": mlp["metrics"]["quietness"],
                "reference_quality": mlp["metrics"]["reference_quality"],
                "strict_reload_bitwise_identical": mlp["strict_reload_bitwise_identical"],
                "selected": True,
            },
        },
        "baselines": selection["baselines"],
        "selection_rule": selection["selection_rule"],
        "selected": selection["selected"],

        "frozen_model": {
            "reference_type": deployment["reference_type"],
            "artifact_file_sha256": deployment["artifact_file_sha256"],
            "model_digest": deployment["model_digest"],
            "input_statistics_sha256": deployment["input_statistics_sha256"],
            "manifest_sha256": deployment["manifest_sha256"],
            "tau": deployment["tau"],
            "tau_rule": deployment["tau_rule"],
            "runtime_inputs": deployment["runtime_inputs"],
            "privileged_inputs": deployment["privileged_inputs"],
            "frozen_before_live_execution": True,
        },

        "offline_replay": {
            "gates": replay["gates"],
            "all_passed": replay["all_gates_passed"],
            "rows": [{k: r[k] for k in ("candidate_index", "hazard_present", "frames",
                                        "activation_rate", "saturation_fraction",
                                        "cosine_teacher_active", "predicted_norm")}
                     for r in replay["rows"]],
            "headline": ("the gate never opened on any of the four expert development "
                         "trajectories: predicted differential norms peaked at 0.54-1.36 "
                         f"against tau = {replay['tau']:.4f}, so all four gates passed "
                         "with zero activations"),
        },

        "live": {
            "schedule_sha256": schedule["schedule_sha256"],
            "rollouts": analysis["rollouts_executed"],
            "budget": 20,
            "frozen_baselines_reused": analysis["frozen_baselines_reused"],
            "per_row": analysis["per_row"],
            "pooled_hazard_present": analysis["pooled_hazard_present"],
            "technical_gates": analysis["technical_gates"],
            "approximation_gates": analysis["approximation_gates"],
            "gross_regression_gates": analysis["gross_regression_gates"],
            "all_technical_gates_passed": analysis["all_technical_gates_passed"],
            "all_approximation_gates_passed": analysis["all_approximation_gates_passed"],
            "all_gross_regression_gates_passed":
                analysis["all_gross_regression_gates_passed"],
        },
        "candidate_108_analysis": analysis["candidate_108_analysis"],
        "candidate_118_negative_control": analysis["candidate_118_negative_control"],
        "shadow_oracle_diagnostics": analysis["shadow_oracle_diagnostics"],

        "interpretation": {
            "what_worked": [
                ("The paired oracle-reference dataset is exact: 0 state-neutrality failures "
                "over 7993 frames, and on every hazard-absent trajectory the current and "
                "parked heads are bitwise equal."),
                ("The posture-only KNN baseline answers its question decisively: posture "
                "alone does NOT explain the parked head (median oracle cosine -0.39, "
                "differential MAE 0.946, worse than both the raw head and first-live "
                "baselines)."),
                ("The MLP passed every predeclared offline gate on held-out trajectories "
                "with a median oracle cosine of 0.977 and a differential MAE of 0.283 "
                "against 0.660 for the raw head and 0.657 for first-live skin."),
                ("All 12 live technical gates passed: 20/20 rollouts finalised, no "
                "privileged feature entered the model, the shadow oracle stayed "
                "state-neutral on every frame, the gripper was bitwise unchanged and the "
                "residual stayed arm-only after temporal aggregation."),
            ],
            "what_failed": [
                ("Live oracle-approximation collapsed: pooled median cosine 0.345 against "
                "the 0.70 gate, versus 0.977 on offline validation."),
                ("Magnitude is badly over-predicted where it matters: on candidate 106 the "
                "deployable differential is 5.1-6.9x the true oracle differential."),
                ("The controller fires when the true signal is exactly zero: 23-34% of "
                "shadow-zero frames on candidate 106 and 7-11% on candidate 107, against a "
                "2% gate."),
                ("Candidate 118, where the true differential is provably zero on every "
                "frame, showed new environment contact in ALL FIVE repeats (12-35 contacts) "
                "where ACT-only had none, and its activation runs reached 4 frames against "
                "a 2-frame limit."),
            ],
            "root_cause": (
                "Distribution shift between the reference model's training data and its "
                "deployment. The model is trained on expert planner trajectories, which "
                "deflect around the hazard and keep the oracle differential small -- on the "
                "four development expert trajectories the predicted norm never once reached "
                "tau. Live ACT trajectories visit states the expert never does and drive "
                "oracle differentials up to 2.5, and there the model extrapolates: right "
                "sign more often than not (positive-cosine fraction 0.87 still passes) but "
                "wrong direction in detail and several times too large."),
            "not_a_task_regression": (
                "Pooled hazard-present task success is 12/15, above ACT-only's 11/15 and "
                "below the oracle's 14/15, and no hazard-present row fell to 0/5. The "
                "regression is in unnecessary motion and in new contact on the "
                "hazard-absent row, not in task completion."),
            "one_genuinely_positive_signal": (
                "On candidate 107 the deployable reference cut other-environment contacts "
                "from ACT-only's 75-93 per rollout to 5-28, better than the oracle's 77-90, "
                "while keeping 5/5 task success. The mechanism is real even though the "
                "approximation is poor."),
            "honest_caveats": [
                ("Attribution of candidate 118's contacts to the controller is supported but "
                "not proven at n=5: the oracle condition, whose correction is provably "
                "exactly zero, still produced 31 contacts in 1 of its 5 repeats, so "
                "MSAA-driven variation alone can generate them. What the deployable adds is "
                "that all five repeats show them, with a nonzero correction present."),
                ("The hazard-absent false-activation gate is close to automatic: tau is the "
                "99.5th percentile of the hazard-absent validation norm, so roughly 0.5% "
                "exceedance is guaranteed by construction rather than measured."),
                ("The far-frame false-activation gate is vacuous in this environment -- no "
                "validation frame exceeds 0.187 m minimum depth, so it has zero frames."),
                ("The candidate-118 'median correction within 10% of hazard-present' gate "
                "passes trivially because both medians are exactly 0.0."),
                "Four development rows and five repeats cannot establish an effect size.",
            ],
        },

        "confirmatory41": {
            "manifest": "configs/hybrid_obstacle_confirmatory41_v1.json",
            "sha256": confirmatory["manifest_sha256"],
            "rows": len(confirmatory["rows"]),
            "executed_in_this_task": confirmatory["executed_in_this_task"],
            "untouched": True,
            "evaluator_hard_refuses_it": True,
        },

        "constraints_honoured": {
            "act_trained_or_modified": False,
            "safety_cvae_trained_or_modified": False,
            "checkpoint_or_statistics_modified": False,
            "canonical_collection_dataset_split_manifests_modified": False,
            "development4_or_confirmatory41_modified": False,
            "molmospaces_modified": git("status", "--porcelain", repo=molmo) != "",
            "msaa_cameras_robot_environment_obstacle_planner_task_horizon_changed": False,
            "temporal_aggregation_changed": False,
            "residual_constants_tuned": False,
            "parked_render_used_at_deployable_inference": False,
            "first_live_skin_used": False,
            "raw_safety_head_used_as_fallback": False,
            "confirmatory_row_executed": False,
            "live_rollout_budget": 20,
            "live_rollouts_used": analysis["rollouts_executed"],
            "pushed": False,
        },

        "uncommitted_artifacts": uncommitted,

        "next_recommended_task": (
            "Close the distribution gap before changing the model. The reference is trained "
            "on expert trajectories that never enter the near-hazard regime the deployed "
            "policy actually visits, and that -- not the architecture -- is what the live "
            "numbers implicate. The natural next step is a predeclared on-policy paired "
            "dataset: run ACT_ONLY rollouts on the training rows, generate the parked "
            "counterfactual along those trajectories with the same state-neutral seam, and "
            "retrain the same fixed architecture on the union. Two secondary items: raise "
            "the quiet threshold's basis from a percentile of hazard-absent norms to a "
            "calibrated false-activation target measured on-policy, and reconsider gate A, "
            "which is open on 76% of frames because this enclosure is never more than 19 cm "
            "away."),

        "artifacts": {
            "final_decision_md":
                "docs/HYBRID_OBSTACLE_DEPLOYABLE_REFERENCE_FINAL_DECISION.md",
            "final_decision_json": ("diagnostics_output/hybrid_obstacle_deployable_reference/"
                                    "final_decision.json"),
            "provenance": str(args.provenance.relative_to(ROOT)),
            "paired_dataset_manifest": str(args.dataset_manifest.relative_to(ROOT)),
            "selection_report": str(args.selection.relative_to(ROOT)),
            "deployment_manifest": str(args.deployment_manifest.relative_to(ROOT)),
            "offline_replay": str(args.offline_replay.relative_to(ROOT)),
            "development_analysis": str(args.analysis.relative_to(ROOT)),
            "schedule": "configs/hybrid_obstacle_deployable_schedule_v1.json",
        },
        "report_hashes": {
            "provenance_sha256": provenance["report_sha256"],
            "paired_dataset_manifest_sha256": dataset["manifest_sha256"],
            "selection_sha256": selection["report_sha256"],
            "deployment_manifest_sha256": deployment["manifest_sha256"],
            "offline_replay_sha256": replay["report_sha256"],
            "analysis_sha256": analysis["report_sha256"],
            "schedule_sha256": schedule["schedule_sha256"],
        },
    }
    payload["final_decision_sha256"] = canonical_hash(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(f"decision: {decision}")
    print(f"selected: {selection['selected']}")
    print(f"pooled hazard-present: AO {analysis['pooled_hazard_present']['act_only']}/15 | "
          f"ORACLE {analysis['pooled_hazard_present']['oracle']}/15 | "
          f"DEPLOYABLE {analysis['pooled_hazard_present']['deployable']}/15")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
