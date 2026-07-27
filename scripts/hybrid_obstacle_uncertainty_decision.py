#!/usr/bin/env python3
"""Assemble the final decision for the uncertainty-abstention qualification.

Handoff steps 20 and 24. The token follows the recorded reports in the order the handoff
imposes: provenance, then ensemble training, then calibration feasibility, then the offline
and live stages that only run when everything before them passed.
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

ALLOWED = (
    "UNCERTAINTY_ABSTENTION_READY_FOR_CONFIRMATORY_41",
    "UNCERTAINTY_ABSTENTION_CALIBRATION_INFEASIBLE",
    "UNCERTAINTY_ABSTENTION_OFFLINE_INVALID",
    "UNCERTAINTY_ABSTENTION_LIVE_TRANSFER_FAILED",
    "UNCERTAINTY_ABSTENTION_LIVE_GROSS_REGRESSION",
    "UNCERTAINTY_ENSEMBLE_TRAINING_FAILED",
    "CHECKPOINT_OR_SOURCE_MISMATCH",
    "UNCERTAINTY_ABSTENTION_TASK_INCOMPLETE",
)


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("provenance", "ensemble-training", "ensemble-manifest",
                 "calibration", "partition", "out"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    ap.add_argument("--agreement-diagnostic", type=Path, default=None)
    args = ap.parse_args()
    for field in ("provenance", "ensemble_training", "ensemble_manifest",
                  "calibration", "partition", "out"):
        setattr(args, field, Path(getattr(args, field)).resolve())

    import numpy
    import torch

    provenance = json.loads(args.provenance.read_text())
    training = json.loads(args.ensemble_training.read_text())
    ensemble = json.loads(args.ensemble_manifest.read_text())
    calibration = json.loads(args.calibration.read_text())
    partition = json.loads(args.partition.read_text())

    if not provenance["all_matched"]:
        decision, case = "CHECKPOINT_OR_SOURCE_MISMATCH", None
    elif not training["all_members_trained_and_reloaded"]:
        decision, case = "UNCERTAINTY_ENSEMBLE_TRAINING_FAILED", None
    elif not calibration.get("feasible"):
        decision, case = "UNCERTAINTY_ABSTENTION_CALIBRATION_INFEASIBLE", "C"
    elif not calibration["nested_passed"] or \
            not calibration["reused_diagnostic"]["passed"] or \
            not calibration["historical_regression"]["passes"] or \
            not calibration["inference_stability"]["stable"]:
        decision, case = "UNCERTAINTY_ABSTENTION_OFFLINE_INVALID", "B"
    else:
        decision, case = "UNCERTAINTY_ABSTENTION_TASK_INCOMPLETE", None
    if decision not in ALLOWED:
        raise SystemExit(f"decision {decision!r} is not allowed")

    sweep = calibration.get("sweep", [])
    unconstrained = next((row for row in sweep if row["threshold"] == 0.0), None)

    payload = {
        "schema": "hybrid_obstacle_uncertainty_abstention_final_decision_v1",
        "date": "2026-07-27",
        "task": ("Train one predeclared trajectory-bootstrap uncertainty ensemble, use "
                 "its changed-pixel-mask disagreement only to abstain from the frozen "
                 "seed-0 safety correction, and qualify on development4"),
        "decision": decision,
        "case": case,
        "case_interpretation": {
            "A": "bootstrap disagreement passes calibration, nested evaluation and live",
            "B": ("uncertainty identifies the historical frames offline but fails nested "
                  "or live transfer"),
            "C": ("trajectory-bootstrap disagreement becomes degenerate or cannot "
                  "preserve active recall and ordinary quiet acceptance; the original "
                  "three-seed finding was not robust enough for deployment"),
        }.get(case),
        "previous_decision": "EPISTEMIC_UNCERTAINTY_SIGNAL_PRESENT",

        "seed0_retrained_or_altered": False,
        "seed_1_or_2_selected_for_deployment": False,
        "predictions_averaged": False,
        "member_replaced_seed0": False,
        "act_or_safety_cvae_modified": False,
        "residual_constants_modified": False,
        "activity_threshold_modified": False,
        "magnitude_support_bound_modified": False,
        "paired_dataset_modified": False,
        "new_data_collected": False,
        "temporal_history_reintroduced": False,
        "live_rollouts_executed": 0,
        "live_rollouts_permitted": 20,
        "development4_executed": False,
        "confirmatory41_executed": False,

        "commits": {
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "root_commit": git("rev-parse", "HEAD"),
            "root_starting_commit": "893c83a",
            "act_branch": git("rev-parse", "--abbrev-ref", "HEAD",
                              repo=ROOT / "submodules" / "act"),
            "act_commit": git("rev-parse", "HEAD", repo=ROOT / "submodules" / "act"),
            "act_starting_commit": "91fc42a",
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

        "ensemble": {
            "ensemble_id": training["ensemble_id"],
            "members": training["member_count"],
            "bootstrap_seeds": ensemble["bootstrap_seeds"],
            "bootstrap_unit": ensemble["bootstrap_unit"],
            "bootstrap_rule": ensemble["bootstrap_rule"],
            "role": ensemble["role"],
            "manifest_sha256": ensemble["manifest_sha256"],
            "all_members_trained_and_reloaded":
                training["all_members_trained_and_reloaded"],
            "acceptance_failures": training["acceptance_failures"],
            "members_dropped": training["members_dropped"],
            "members_replaced": training["members_replaced"],
            "zero_differential_validation_mae":
                training["zero_differential_validation_mae"],
            "diversity": training["diversity"],
            "member_records": [
                {k: v for k, v in m.items()
                 if k in ("index", "bootstrap_seed", "unique_clusters",
                          "out_of_bag_count", "checkpoint", "checkpoint_sha256",
                          "parameter_count", "best_epoch", "validation")}
                for m in training["members"]],
        },

        "partition": {
            "manifest": "configs/hybrid_obstacle_prox_activity_partition_v1.json",
            "manifest_sha256": partition["manifest_sha256"],
            "reused_without_modification": True,
            "splits": {name: {k: block[k] for k in
                              ("episode_count", "hazard_present", "hazard_absent")}
                       for name, block in partition["splits"].items()},
        },

        "anchor_agreement": {
            "definition": ("mean Jaccard between the frozen seed-0 changed-pixel mask "
                           "and each of the five bootstrap-member masks; two empty masks "
                           "agree completely"),
            "pixel_mask_threshold": 0.5,
            "only_metric_controlling_abstention": True,
        },

        "calibration": {
            "feasible": False,
            "candidate_count": calibration.get("candidate_count"),
            "feasible_count": 0,
            "activity_threshold": 0.99960857629776,
            "activity_threshold_refit": False,
            "sweep": sweep,
            "unconstrained_operating_point": unconstrained,
            "binding_failures": sorted({
                name for row in sweep
                for name, ok in row["checks"].items() if not ok}),
            "decision_if_infeasible": "UNCERTAINTY_ABSTENTION_CALIBRATION_INFEASIBLE",
        },

        "stages_not_reached": [
            "combined deployment manifest freeze (step 10)",
            "nested offline evaluation (step 11)",
            "historical 17-frame regression (step 12)",
            "reused diagnostic audit (step 13)",
            "frozen inference determinism (step 14)",
            "live development rollouts (steps 16-17)",
            "live uncertainty and approximation gates (step 18)",
            "live gross-regression gates (step 19)",
        ],
        "deployment_manifest_written": False,

        "constraints_honoured": {
            "seed0_retrained_or_altered": False,
            "seed_1_or_2_selected_for_deployment": False,
            "predictions_averaged_across_models": False,
            "member_replaced_seed0_prediction": False,
            "act_modified": False,
            "safety_cvae_modified": False,
            "residual_controller_constants_modified": False,
            "activity_threshold_modified": False,
            "magnitude_support_bound_modified": False,
            "paired_dataset_modified": False,
            "new_training_or_on_policy_data_collected": False,
            "temporal_history_reintroduced": False,
            "uncertainty_used_to_amplify_or_redirect": False,
            "development4_or_confirmatory41_used_for_fitting": False,
            "confirmatory41_executed": False,
            "new_live_rollouts_executed": 0,
            "pushed": False,
        },

        "artifacts": {
            "final_decision_md":
                "docs/HYBRID_OBSTACLE_UNCERTAINTY_ABSTENTION_FINAL_DECISION.md",
            "final_decision_json":
                "diagnostics_output/hybrid_obstacle_uncertainty_abstention/"
                "final_decision.json",
            "provenance": str(args.provenance.relative_to(ROOT)),
            "ensemble_training": str(args.ensemble_training.relative_to(ROOT)),
            "ensemble_manifest": str(args.ensemble_manifest.relative_to(ROOT)),
            "calibration": str(args.calibration.relative_to(ROOT)),
            "abstention_source": "causal_parked_skin/abstention.py",
            "evaluator_integration": "submodules/act/uncertainty_abstention.py",
            "tests": "tests/test_uncertainty_abstention.py",
        },
        "report_hashes": {
            "provenance_sha256": provenance["report_sha256"],
            "ensemble_training_sha256": training["report_sha256"],
            "ensemble_manifest_sha256": ensemble["manifest_sha256"],
            "calibration_sha256": calibration["report_sha256"],
            "partition_sha256": partition["manifest_sha256"],
        },
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": numpy.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    if args.agreement_diagnostic and args.agreement_diagnostic.is_file():
        payload["agreement_diagnostic"] = json.loads(
            args.agreement_diagnostic.read_text())
    payload["final_decision_sha256"] = thr.canonical_hash(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    print(f"decision : {decision}")
    print(f"case     : {case}")
    print(f"members  : {training['member_count']} trained, "
          f"{training['members_dropped']} dropped")
    print(f"binding failures: {payload['calibration']['binding_failures']}")
    print("live rollouts executed: 0")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
