#!/usr/bin/env python3
"""Assemble the final decision for the threshold-qualification task.

Handoff steps 16, 19 and 20. The token is derived from the recorded reports rather than
argued: calibration feasibility, the reused-diagnostic blocking checks and the inference
stability result determine it, in that order.
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
    "REFERENCE_THRESHOLD_READY_FOR_CONFIRMATORY_41",
    "REFERENCE_THRESHOLD_CALIBRATION_INFEASIBLE",
    "REFERENCE_THRESHOLD_TRANSFER_FAILED",
    "REFERENCE_MODEL_LIVE_GROSS_REGRESSION",
    "REFERENCE_MODEL_INFERENCE_UNSTABLE",
    "CHECKPOINT_OR_SOURCE_MISMATCH",
    "REFERENCE_THRESHOLD_QUALIFICATION_INCOMPLETE",
)


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provenance", required=True, type=Path)
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--calibration-manifest", required=True, type=Path)
    ap.add_argument("--threshold-manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    # accept relative or absolute paths; the report records repo-relative ones
    for field in ("provenance", "calibration", "audit", "calibration_manifest",
                  "threshold_manifest", "out"):
        setattr(args, field, Path(getattr(args, field)).resolve())

    import numpy
    import torch

    provenance = json.loads(args.provenance.read_text())
    calibration = json.loads(args.calibration.read_text())
    audit = json.loads(args.audit.read_text())
    calibration16 = json.loads(args.calibration_manifest.read_text())
    manifest = json.loads(args.threshold_manifest.read_text())

    # --- decision, in the order the handoff imposes -------------------------------
    if not provenance["all_matched"]:
        decision, case = "CHECKPOINT_OR_SOURCE_MISMATCH", None
    elif not calibration.get("feasible"):
        decision, case = "REFERENCE_THRESHOLD_CALIBRATION_INFEASIBLE", "B"
    elif not audit["inference_stability"]["stable"] or \
            not audit["head_stability"]["stable"]:
        decision, case = "REFERENCE_MODEL_INFERENCE_UNSTABLE", None
    elif not audit["blocking_checks"]["passed"]:
        decision, case = "REFERENCE_THRESHOLD_TRANSFER_FAILED", "C"
    else:
        # live gates would decide between Case A and a live-regression token; this task
        # never reached them, so reaching this branch without a live run is incomplete
        decision, case = "REFERENCE_THRESHOLD_QUALIFICATION_INCOMPLETE", None
    if decision not in ALLOWED:
        raise SystemExit(f"decision {decision!r} is not allowed")

    payload = {
        "schema": "hybrid_obstacle_reference_threshold_final_decision_v1",
        "date": "2026-07-27",
        "task": ("Recalibrate and qualify the frozen CURRENT_FRAME_ONLY parked-reference "
                 "model using a trajectory-aware activation contract"),
        "decision": decision,
        "case": case,
        "case_interpretation": {
            "A": "calibration feasible, diagnostics pass, live gates pass",
            "B": "trajectory-aware calibration infeasible",
            "C": ("calibration succeeds offline but false activation remains; threshold "
                  "transfer is unreliable and the next change must affect the activity "
                  "model or training objective, not only the threshold"),
        }[case] if case else None,

        "previous_decision": {
            "token": "PARKED_REFERENCE_MODEL_OVERFIT",
            "why_the_token_is_misleading": (
                "the rubric required it, but the evidence is not conventional "
                "overfitting: offline-test MAE (0.009087) was less than half validation "
                "MAE (0.020782), the model improved on the trivial baseline by ~73%, "
                "median oracle direction cosine was ~0.999 and active recall 94-98%. The "
                "single failed gate was seed-0 oracle-zero FPR at 2.15% against a 2.00% "
                "ceiling, with calibration FPR ~1% by construction and threshold "
                "coefficient of variation ~0.72 across seeds. The blocker was "
                "activation-threshold transfer instability."),
        },

        "model_trained_or_finetuned": False,
        "seed_reselected": False,
        "act_trained_or_modified": False,
        "safety_cvae_trained_or_modified": False,
        "controller_constants_changed": False,
        "architecture_or_inputs_changed": False,
        "causal_history_reintroduced": False,
        "dataset_modified": False,
        "live_rollouts_executed": 0,
        "live_rollouts_permitted": 20,
        "confirmatory41_executed": False,
        "development4_executed": False,

        "commits": {
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "root_commit": git("rev-parse", "HEAD"),
            "root_starting_commit": "b2051ae",
            "act_commit": git("rev-parse", "HEAD", repo=ROOT / "submodules" / "act"),
            "act_expected": "91fc42a",
            "act_modified": git("status", "--porcelain",
                                repo=ROOT / "submodules" / "act") != "",
            "molmospaces_commit": git("rev-parse", "HEAD",
                                      repo=ROOT / "submodules" / "molmospaces"),
            "molmospaces_expected": "678f2eb",
            "molmospaces_modified": git("status", "--porcelain",
                                        repo=ROOT / "submodules" / "molmospaces") != "",
        },

        "provenance": {
            "checks": provenance["check_count"],
            "all_matched": provenance["all_matched"],
            "frozen_model": provenance["frozen_model"],
        },
        "seed_disposition": {
            "production_candidate": "seed 0",
            "reason": "it was the preselected model at qualification",
            "seeds_1_and_2": "reported as sensitivity diagnostics only",
            "selecting_another_seed_now_would_be": "post-hoc model selection",
            "seed_reselected": False,
            "seed_thresholds_reused": False,
        },

        "retired_threshold": calibration["retired_threshold"],
        "calibration16": {
            "manifest": str(args.calibration_manifest.relative_to(ROOT)),
            "manifest_sha256": calibration16["manifest_sha256"],
            "episodes": calibration16["episode_count"],
            "trajectories": calibration16["trajectory_count"],
            "frames": calibration16["total_frames"],
            "cluster_unit": calibration16["cluster_unit"],
            "cluster_rationale": calibration16["cluster_rationale"],
            "source_partitions": calibration16["source_partitions"],
        },
        "activity_definition": calibration["activity_definition"],
        "gate_rule": calibration["gate_rule"],
        "bootstrap": calibration["bootstrap"],
        "feasibility_contract": calibration["feasibility_contract"],
        "selection_rule": calibration["selection_rule"],
        "candidates": {
            "count": calibration["candidate_count"],
            "screened": calibration["candidates_screened"],
            "feasible": calibration["feasible_count"],
        },
        "selected_threshold": calibration["selected"],
        "calibration16_trajectory_metrics": calibration["calibration16_trajectory_metrics"],

        "consumed_diagnostic": {
            **audit["diagnostic_set"],
            "metrics": audit["diagnostic_metrics"],
            "blocking_checks": audit["blocking_checks"],
        },
        "inference_stability": audit["inference_stability"],
        "head_stability": audit["head_stability"],

        "magnitude_support": manifest["magnitude_support"],
        "threshold_manifest": {
            "path": str(args.threshold_manifest.relative_to(ROOT)),
            "manifest_sha256": manifest["manifest_sha256"],
            "authorized_for_live": manifest["authorized_for_live"],
            "authorization_blocker": manifest["authorization_blocker"],
        },
        "live_schedule": manifest["live_schedule"],
        "log_schema": manifest["log_schema"],
        "excluded_log_fields": manifest["excluded_log_fields"],
        "controller": manifest["controller"],

        "constraints_honoured": {
            "reference_model_trained_or_finetuned": False,
            "seed_selected_on_observed_test_performance": False,
            "act_trained_or_modified": False,
            "safety_cvae_trained_or_modified": False,
            "residual_constants_changed": False,
            "architecture_or_inputs_changed": False,
            "causal_history_reintroduced": False,
            "paired_dataset_modified": False,
            "msaa_cameras_environment_changed": False,
            "two_percent_target_loosened": False,
            "prior_offline_test_used_to_fit_threshold": False,
            "confirmatory41_row_executed": False,
            "max_new_policy_rollouts": 20,
            "new_policy_rollouts_executed": 0,
            "pushed": False,
        },

        "artifacts": {
            "final_decision_md":
                "docs/HYBRID_OBSTACLE_REFERENCE_THRESHOLD_FINAL_DECISION.md",
            "final_decision_json":
                "diagnostics_output/hybrid_obstacle_reference_threshold/"
                "final_decision.json",
            "provenance": str(args.provenance.relative_to(ROOT)),
            "calibration": str(args.calibration.relative_to(ROOT)),
            "audit": str(args.audit.relative_to(ROOT)),
            "calibration16_manifest": str(args.calibration_manifest.relative_to(ROOT)),
            "threshold_manifest": str(args.threshold_manifest.relative_to(ROOT)),
            "threshold_module": "causal_parked_skin/threshold.py",
            "tests": "tests/test_reference_threshold_qualification.py",
        },
        "report_hashes": {
            "provenance_sha256": provenance["report_sha256"],
            "calibration_sha256": calibration["report_sha256"],
            "audit_sha256": audit["report_sha256"],
            "calibration16_sha256": calibration16["manifest_sha256"],
            "threshold_manifest_sha256": manifest["manifest_sha256"],
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    payload["final_decision_sha256"] = thr.canonical_hash(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    print(f"decision : {decision}")
    print(f"case     : {case}")
    print(f"threshold: {calibration['selected']['threshold']:.8f}")
    print(f"live rollouts executed: {payload['live_rollouts_executed']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
