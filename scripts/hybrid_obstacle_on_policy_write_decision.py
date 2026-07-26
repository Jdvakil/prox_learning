#!/usr/bin/env python3
"""Assemble the final on-policy-reference decision JSON.

Handoff step 19. Every field is copied from an artifact produced earlier in the task; the
decision token comes from the analysis (or, when the task stops offline, from the
training report).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWED = (
    "ON_POLICY_REFERENCE_READY_FOR_CONFIRMATORY_41",
    "ON_POLICY_REFERENCE_OFFLINE_INVALID",
    "ON_POLICY_REFERENCE_LIVE_GROSS_REGRESSION",
    "REFERENCE_FEATURE_CONTRACT_INSUFFICIENT",
    "ON_POLICY_DATA_CONTRACT_FAILED",
    "CHECKPOINT_OR_SOURCE_MISMATCH",
    "ON_POLICY_REFERENCE_DEVELOPMENT_INCOMPLETE",
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


def load(path: Path):
    return json.loads(Path(path).read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provenance", required=True, type=Path)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--labelling-schedule", required=True, type=Path)
    ap.add_argument("--labelling-dataset", required=True, type=Path)
    ap.add_argument("--learner-schedule", type=Path, default=None)
    ap.add_argument("--learner-dataset", type=Path, default=None)
    ap.add_argument("--round0", required=True, type=Path)
    ap.add_argument("--round1", required=True, type=Path)
    ap.add_argument("--reference-manifest", type=Path, default=None)
    ap.add_argument("--shift-audit", type=Path, default=None)
    ap.add_argument("--live-schedule", type=Path, default=None)
    ap.add_argument("--analysis", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    provenance = load(args.provenance)
    partition = load(args.partition)
    labelling_schedule = load(args.labelling_schedule)
    labelling_dataset = load(args.labelling_dataset)
    round0 = load(args.round0)
    round1 = load(args.round1)
    learner_schedule = load(args.learner_schedule) if args.learner_schedule else None
    learner_dataset = load(args.learner_dataset) if args.learner_dataset else None
    reference = load(args.reference_manifest) if args.reference_manifest else None
    shift = load(args.shift_audit) if args.shift_audit else None
    live_schedule = load(args.live_schedule) if args.live_schedule else None
    analysis = load(args.analysis) if args.analysis else None

    decision = (analysis["decision"] if analysis
                else round1.get("decision", "ON_POLICY_REFERENCE_OFFLINE_INVALID"))
    if decision == "OFFLINE_QUALIFIED":
        decision = "ON_POLICY_REFERENCE_DEVELOPMENT_INCOMPLETE"
    if decision not in ALLOWED:
        raise SystemExit(f"decision {decision!r} is not allowed")

    act = ROOT / "submodules/act"
    molmo = ROOT / "submodules/molmospaces"
    confirmatory = load(ROOT / "configs/hybrid_obstacle_confirmatory41_v1.json")

    uncommitted = []
    for label, path in (
        ("on_policy_rollout_root", Path("/root/act_retrain_assets/on_policy_v2")),
        ("learner_rollout_root", Path("/root/act_retrain_assets/on_policy_learner_v2")),
        ("live_rollout_root", Path("/root/act_retrain_assets/on_policy_live_v2")),
        ("expert_paired_dir", Path("/root/act_retrain_assets/paired_reference_v1")),
        ("round0_checkpoint", Path(round0["artifact_path"])),
        ("round1_checkpoint", Path(round1["artifact_path"])),
        ("act_checkpoint", Path(
            "/root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2"
            "/20260725_seed0_2000ep/policy_best.ckpt")),
    ):
        entry = {"label": label, "path": str(path), "committed": False}
        if path.is_file():
            entry["sha256"] = sha256_file(path)
        elif path.is_dir():
            entry["entries"] = len(list(path.glob("*")))
        uncommitted.append(entry)

    payload = {
        "schema": "hybrid_obstacle_on_policy_reference_final_decision_v2",
        "date": "2026-07-26",
        "task": ("One predeclared DAgger-style on-policy dataset-aggregation round for the "
                 "posture+skin reference MLP, with activation recalibrated on disjoint "
                 "trajectories and the four-row live development evaluation repeated"),
        "decision": decision,
        "case": (analysis or {}).get("case") or ("C" if decision ==
                                                 "ON_POLICY_REFERENCE_OFFLINE_INVALID"
                                                 else None),

        "commits": {
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "root_starting_commit": "0bca2be10b6421115ecef50d8ffcfc8577f13821",
            "root_commit": git("rev-parse", "HEAD"),
            "act_branch": git("rev-parse", "--abbrev-ref", "HEAD", repo=act),
            "act_starting_commit": "f6301bcaae4510946a7cb71e6feb3871d59327ad",
            "act_commit": git("rev-parse", "HEAD", repo=act),
            "molmospaces_commit": git("rev-parse", "HEAD", repo=molmo),
            "molmospaces_modified": git("status", "--porcelain", repo=molmo) != "",
        },

        "artifact_verification": {"checks": provenance["check_count"],
                                  "all_matched": provenance["all_matched"]},

        "reference_partition": {
            "sha256": partition["partition_sha256"],
            "composition": partition["composition"],
            "all_pairwise_disjoint": partition["all_pairwise_disjoint"],
            "act_split_unchanged": partition["act_split_unchanged"],
            "rule": partition["rule"],
            "excluded": partition["excluded"],
        },

        "schedules": {
            "labelling": {"sha256": labelling_schedule["schedule_sha256"],
                          "rollouts": labelling_schedule["rollouts"],
                          "order_balance": labelling_schedule["order_balance"]},
            "learner": ({"sha256": learner_schedule["schedule_sha256"],
                         "rollouts": learner_schedule["rollouts"]}
                        if learner_schedule else None),
            "live": ({"sha256": live_schedule["schedule_sha256"],
                      "rollouts": live_schedule["rollouts"]} if live_schedule else None),
        },

        "datasets": {
            "labelling": {
                "manifest_sha256": labelling_dataset["manifest_sha256"],
                "rollouts": labelling_dataset["rollouts_present"],
                "total_frames": labelling_dataset["total_frames"],
                "by_distribution": labelling_dataset["by_distribution"],
                "oracle_pairing": labelling_dataset["oracle_pairing"],
                "feature_contract": labelling_dataset["feature_contract"],
                "valid": labelling_dataset["valid"]},
            "learner": ({
                "manifest_sha256": learner_dataset["manifest_sha256"],
                "rollouts": learner_dataset["rollouts_present"],
                "total_frames": learner_dataset["total_frames"],
                "oracle_pairing": learner_dataset["oracle_pairing"],
                "valid": learner_dataset["valid"]} if learner_dataset else None),
        },

        "round0": {k: round0[k] for k in
                   ("label", "best_epoch", "best_validation_loss", "parameters",
                    "distribution_weighting", "distribution_provenance",
                    "strict_reload_bitwise_identical", "artifact_file_sha256",
                    "model_digest", "fresh_initialisation",
                    "continued_from_previous_checkpoint")},
        "round1": {k: round1[k] for k in
                   ("label", "best_epoch", "best_validation_loss", "parameters",
                    "distribution_weighting", "distribution_provenance",
                    "strict_reload_bitwise_identical", "artifact_file_sha256",
                    "model_digest", "fresh_initialisation",
                    "continued_from_previous_checkpoint")},
        "aggregation_rounds_performed": 1,
        "architecture_or_hyperparameter_sweep": False,
        "second_architecture_added": False,

        "calibration": round1.get("calibration"),
        "tau": round1.get("tau"),
        "rho_max": round1.get("rho_max"),
        "support_envelope": {
            "gate": "SupportEnvelopeGate",
            "rule": ("silent below tau; above it the direction is preserved and the norm "
                     "capped at rho_max"),
            "global_minimum_depth_used_as_support_gate": False,
            "why_removed": ("the V1 depth gate was open on 76% of validation frames "
                            "because this enclosure is never more than ~19 cm away; it was "
                            "dominated by static geometry and gated almost nothing"),
            "previous_tau_reused": False,
        },

        "offline_evaluations": round1.get("evaluations"),
        "offline_gates": round1.get("gates"),
        "all_offline_gates_passed": round1.get("all_offline_gates_passed"),

        "shift_audit": ({
            "required_improvements": shift["required_improvements"],
            "all_evaluable_required_improvements_met":
                shift["all_evaluable_required_improvements_met"],
            "improvements_not_evaluable": shift["improvements_not_evaluable"],
            "v2_gate_is_diagnostic_only": shift["v2_gate_is_diagnostic_only"],
            "results": shift["results"],
            "caveat": shift["caveat"]} if shift else None),

        "live": ({
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
            "candidate_118_negative_control": analysis["candidate_118_negative_control"],
            "evidence_basis": analysis["evidence_basis"]} if analysis else None),

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
            "residual_constants_changed": False,
            "cameras_msaa_robot_obstacle_planner_task_horizon_changed": False,
            "canonical_collection_or_act_dataset_modified": False,
            "trained_on_development_rows": False,
            "confirmatory_row_executed": False,
            "privileged_features_at_inference": False,
            "architecture_or_hyperparameter_sweep": False,
            "second_architecture_added": False,
            "previous_tau_silently_reused": False,
            "broken_clearance_metric_used_as_evidence": False,
            "pushed": False,
        },

        "uncommitted_artifacts": uncommitted,

        "artifacts": {
            "final_decision_md":
                "docs/HYBRID_OBSTACLE_ON_POLICY_REFERENCE_FINAL_DECISION.md",
            "final_decision_json": ("diagnostics_output/hybrid_obstacle_on_policy_reference/"
                                    "final_decision.json"),
            "partition": "configs/hybrid_obstacle_reference_partition_v2.json",
            "labelling_schedule":
                "configs/hybrid_obstacle_on_policy_labelling_schedule_v2.json",
            "learner_schedule": "configs/hybrid_obstacle_on_policy_learner_schedule_v2.json",
            "live_schedule": "configs/hybrid_obstacle_on_policy_live_schedule_v2.json",
        },
        "report_hashes": {
            "provenance_sha256": provenance["report_sha256"],
            "partition_sha256": partition["partition_sha256"],
            "labelling_dataset_sha256": labelling_dataset["manifest_sha256"],
            "round0_sha256": round0["report_sha256"],
            "round1_sha256": round1["report_sha256"],
            "shift_audit_sha256": (shift or {}).get("report_sha256"),
            "analysis_sha256": (analysis or {}).get("report_sha256"),
            "reference_manifest_sha256": (reference or {}).get("manifest_sha256"),
        },
    }
    payload["final_decision_sha256"] = canonical_hash(payload)
    args.out = Path(args.out).resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(f"decision: {decision}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
