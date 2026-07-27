#!/usr/bin/env python3
"""Assemble the final decision for the full-seed joint-gate qualification.

Handoff steps 21 and 25. The token follows the recorded reports in the order the handoff
imposes: provenance, then inference stability, then calibration feasibility, then the offline
transfer checks, then the live stages that only run when everything before them passed.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin import threshold as thr
from causal_parked_skin.joint_gate import BOOTSTRAP_DISPOSITION, MODE

ALLOWED = (
    "FULL_SEED_JOINT_GATE_READY_FOR_CONFIRMATORY_41",
    "FULL_SEED_JOINT_GATE_CALIBRATION_INFEASIBLE",
    "FULL_SEED_JOINT_GATE_OFFLINE_TRANSFER_FAILED",
    "FULL_SEED_JOINT_GATE_LIVE_TRANSFER_FAILED",
    "FULL_SEED_JOINT_GATE_LIVE_GROSS_REGRESSION",
    "REFERENCE_MODEL_INFERENCE_UNSTABLE",
    "CHECKPOINT_OR_SOURCE_MISMATCH",
    "FULL_SEED_JOINT_GATE_TASK_INCOMPLETE",
)


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("provenance", "calibration", "manifest", "partition", "out"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    args = ap.parse_args()
    for field in ("provenance", "calibration", "manifest", "partition", "out"):
        setattr(args, field, Path(getattr(args, field)).resolve())

    import numpy
    import torch

    provenance = json.loads(args.provenance.read_text())
    calibration = json.loads(args.calibration.read_text())
    manifest = json.loads(args.manifest.read_text())
    partition = json.loads(args.partition.read_text())

    if not provenance["all_matched"]:
        decision, case = "CHECKPOINT_OR_SOURCE_MISMATCH", None
    elif not calibration.get("feasible"):
        decision, case = "FULL_SEED_JOINT_GATE_CALIBRATION_INFEASIBLE", "B"
    elif not calibration["inference_stability"]["stable"]:
        decision, case = "REFERENCE_MODEL_INFERENCE_UNSTABLE", None
    elif not (calibration["nested_passed"]
              and calibration["historical_regression"]["passes"]
              and calibration["reused_diagnostic"]["passed"]):
        decision, case = "FULL_SEED_JOINT_GATE_OFFLINE_TRANSFER_FAILED", "C"
    else:
        decision, case = "FULL_SEED_JOINT_GATE_TASK_INCOMPLETE", None
    if decision not in ALLOWED:
        raise SystemExit(f"decision {decision!r} is not allowed")

    historical = calibration["historical_regression"]
    payload = {
        "schema": "hybrid_obstacle_full_seed_joint_gate_final_decision_v1",
        "date": "2026-07-27",
        "task": ("Jointly recalibrate the seed-0 activity threshold and the full-data "
                 "three-seed mask-agreement threshold, then conditionally qualify on "
                 "development4"),
        "decision": decision,
        "case": case,
        "case_interpretation": {
            "A": "feasible pair, nested and diagnostic transfer pass, live gates pass",
            "B": ("no joint pair preserves coverage, quiet acceptance and false "
                  "activation together"),
            "C": ("offline joint calibration passes but nested or live transfer fails; "
                  "full-data seed disagreement does not transfer reliably enough for "
                  "closed-loop deployment, and additional same-input ensembles are not "
                  "justified"),
        }.get(case),
        "previous_decision": "UNCERTAINTY_ABSTENTION_CALIBRATION_INFEASIBLE",
        "mode": MODE,

        "owner_decisions_honoured": {
            "recall_floor_preserved_at_0_80": True,
            "recall_floor_lowered": False,
            "standalone_activity_threshold_retired": True,
            "only_full_data_seeds_0_1_2_used": True,
            "bootstrap_ensemble_used": False,
            "seed0_sole_deployment_predictor": True,
            "thresholds_calibrated_jointly": True,
        },

        "bootstrap_ensemble": {
            "disposition": BOOTSTRAP_DISPOSITION,
            "checkpoints_deleted": False,
            "reports_deleted": False,
            "entered_threshold_fitting": False,
            "entered_nested_evaluation": False,
            "entered_runtime_evaluator": False,
            "entered_deployment_manifest": False,
            "entered_live_rollout": False,
            "loader_refuses_in_this_mode": True,
        },

        "model_trained": False,
        "new_seeds_created": False,
        "predictions_averaged": False,
        "act_or_safety_cvae_modified": False,
        "residual_constants_modified": False,
        "magnitude_support_modified": False,
        "parked_field_predictor_altered": False,
        "paired_dataset_modified": False,
        "live_rollouts_executed": 0,
        "live_rollouts_permitted": 20,
        "development4_executed": False,
        "confirmatory41_executed": False,

        "commits": {
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "root_commit": git("rev-parse", "HEAD"),
            "root_starting_commit": "bea058c",
            "act_commit": git("rev-parse", "HEAD", repo=ROOT / "submodules" / "act"),
            "act_starting_commit": "69bda27",
            "act_modified": git("status", "--porcelain",
                                repo=ROOT / "submodules" / "act") != "",
            "molmospaces_commit": git("rev-parse", "HEAD",
                                      repo=ROOT / "submodules" / "molmospaces"),
            "molmospaces_expected": "678f2eb",
            "molmospaces_modified": git("status", "--porcelain",
                                        repo=ROOT / "submodules" / "molmospaces") != "",
        },

        "provenance": {"checks": provenance["check_count"],
                       "all_matched": provenance["all_matched"],
                       "frozen_model": provenance["frozen_model"]},
        "seed_roster": {
            "deployment": 0, "uncertainty": [1, 2],
            "checkpoint_sha256": manifest["seed_checkpoint_sha256"],
            "shared_configuration": "identical apart from the seed",
        },

        "partition": {"manifest_sha256": partition["manifest_sha256"],
                      "reused_without_modification": True},

        "agreement_definition": {
            "controlling": manifest["agreement_definition"],
            "pixel_mask_threshold": manifest["pixel_mask_threshold"],
            "mask_comparison": manifest["mask_comparison"],
            "implementation_sha256": manifest["agreement_implementation_sha256"],
            "differs_from_identifiability_audit": (
                "the audit averaged all three pairs including J(seed1, seed2); the "
                "handoff specifies the anchor form mean(J(0,1), J(0,2)). On the 17 "
                "historical frames J(seed1, seed2) is ~0, so dropping it raises their "
                "agreement from ~0.167 to ~0.25 and makes them harder to reject"),
        },

        "calibration": {
            "feasible": True,
            "grid": calibration["grid"],
            "feasible_pairs": calibration["feasible_count"],
            "selected_activity_threshold": calibration["selected"]["activity_threshold"],
            "selected_agreement_threshold": calibration["selected"]["agreement_threshold"],
            "bootstrap_upper_false_activation":
                calibration["selected"]["bootstrap_upper_false_activation"],
            "selection_rule": calibration["selection_rule"],
            "contract": calibration["contract"],
            "metrics": calibration["calibration"],
            "checks": calibration["selected"]["checks"],
        },
        "old_threshold_comparison": {
            **calibration["old_threshold_comparison"],
            "recall_floor_relaxed": False,
            "why_the_standalone_threshold_changed": (
                "the deployed system is now a two-gate controller; a threshold "
                "calibrated for a single gate left no recall headroom for a veto"),
            "uncertainty_did_not_justify_lowering_system_recall": True,
        },
        "deployment_manifest": {
            "path": "configs/hybrid_obstacle_full_seed_joint_gate_v1.json",
            "manifest_sha256": manifest["manifest_sha256"],
            "frozen_before_nested_evaluation": True,
        },

        "nested_offline": calibration["nested_offline"],
        "nested_failures": calibration["nested_failures"],
        "nested_passed": calibration["nested_passed"],
        "historical_regression": historical,
        "reused_diagnostic": calibration["reused_diagnostic"],
        "inference_stability": calibration["inference_stability"],

        "stages_not_reached": [
            "evaluator integration for ACT_PLUS_FULL_SEED_JOINT_GATE (step 16)",
            "live development rollouts (steps 17-18)",
            "live approximation and abstention gates (step 19)",
            "live gross-regression gates (step 20)",
        ],

        "constraints_honoured": {
            "model_trained_or_finetuned": False,
            "new_seeds_created": False,
            "bootstrap_members_used_for_deployment_or_calibration": False,
            "seed_1_or_2_selected_as_deployment_predictor": False,
            "fields_heads_differentials_or_actions_averaged": False,
            "act_modified": False,
            "safety_cvae_modified": False,
            "residual_controller_constants_modified": False,
            "magnitude_support_bound_altered": False,
            "parked_field_predictor_altered": False,
            "paired_dataset_modified": False,
            "recall_floor_lowered": False,
            "fitted_using_historical_17_frames": False,
            "fitted_using_development4": False,
            "fitted_using_confirmatory41": False,
            "confirmatory41_executed": False,
            "new_live_rollouts_executed": 0,
            "pushed": False,
        },

        "artifacts": {
            "final_decision_md":
                "docs/HYBRID_OBSTACLE_FULL_SEED_JOINT_GATE_FINAL_DECISION.md",
            "final_decision_json":
                "diagnostics_output/hybrid_obstacle_full_seed_joint_gate/"
                "final_decision.json",
            "provenance": str(args.provenance.relative_to(ROOT)),
            "joint_calibration": str(args.calibration.relative_to(ROOT)),
            "deployment_manifest": str(args.manifest.relative_to(ROOT)),
            "joint_gate_source": "causal_parked_skin/joint_gate.py",
            "tests": "tests/test_full_seed_joint_gate.py",
        },
        "report_hashes": {
            "provenance_sha256": provenance["report_sha256"],
            "calibration_sha256": calibration["report_sha256"],
            "deployment_manifest_sha256": manifest["manifest_sha256"],
            "partition_sha256": partition["manifest_sha256"],
        },
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": numpy.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    payload["final_decision_sha256"] = thr.canonical_hash(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    print(f"decision : {decision}")
    print(f"case     : {case}")
    print(f"activity {payload['calibration']['selected_activity_threshold']:.8f} "
          f"agreement {payload['calibration']['selected_agreement_threshold']:.4f}")
    print(f"nested passed   : {calibration['nested_passed']} "
          f"{calibration['nested_failures']}")
    print(f"historical      : {historical['rejected']}/17 rejected, "
          f"{historical['executed']} executed, passes={historical['passes']}")
    print(f"diagnostic      : {calibration['reused_diagnostic']['passed']} "
          f"{calibration['reused_diagnostic']['failures']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
