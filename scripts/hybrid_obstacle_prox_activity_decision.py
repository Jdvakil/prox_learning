#!/usr/bin/env python3
"""Assemble the final decision for the proximity-only activity-gate task.

Handoff steps 21 and 25. The token follows from the recorded reports in the order the
handoff imposes: provenance, then calibration feasibility, then offline validity, then the
live stages that only run if everything before them passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin import threshold as thr

ALLOWED = (
    "PROX_ACTIVITY_GATE_READY_FOR_CONFIRMATORY_41",
    "PROX_ACTIVITY_GATE_CALIBRATION_INFEASIBLE",
    "PROX_ACTIVITY_GATE_OFFLINE_INVALID",
    "PROX_ACTIVITY_GATE_LIVE_TRANSFER_FAILED",
    "PROX_ACTIVITY_GATE_LIVE_GROSS_REGRESSION",
    "ACTIVITY_ONSET_CAUSE_UNRESOLVED",
    "CHECKPOINT_OR_SOURCE_MISMATCH",
    "PROX_ACTIVITY_GATE_TASK_INCOMPLETE",
)


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("provenance", "path-audit", "onset-audit", "partition", "training",
                 "evaluation", "out"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    args = ap.parse_args()
    for field in ("provenance", "path_audit", "onset_audit", "partition", "training",
                  "evaluation", "out"):
        setattr(args, field, Path(getattr(args, field)).resolve())

    import numpy
    import torch

    provenance = json.loads(args.provenance.read_text())
    path_audit = json.loads(args.path_audit.read_text())
    onset = json.loads(args.onset_audit.read_text())
    partition = json.loads(args.partition.read_text())
    training = json.loads(args.training.read_text())
    evaluation = json.loads(args.evaluation.read_text())

    if not provenance["all_matched"]:
        decision, case = "CHECKPOINT_OR_SOURCE_MISMATCH", None
    elif not evaluation.get("feasible"):
        decision, case = "PROX_ACTIVITY_GATE_CALIBRATION_INFEASIBLE", "B"
    elif not evaluation["nested_gates"]["passed"] or \
            not evaluation["reused_diagnostic"]["blocking_checks"]["passed"] or \
            not evaluation["historical_onset_regression"]["all_inactive"]:
        decision, case = "PROX_ACTIVITY_GATE_OFFLINE_INVALID", "B"
    else:
        decision, case = "PROX_ACTIVITY_GATE_TASK_INCOMPLETE", None
    if decision not in ALLOWED:
        raise SystemExit(f"decision {decision!r} is not allowed")

    checkpoint = Path(training["best_checkpoint"])
    payload = {
        "schema": "hybrid_obstacle_prox_activity_gate_final_decision_v1",
        "date": "2026-07-27",
        "task": ("Diagnose the episode-onset activity prior, replace only the activity "
                 "path with a proximity-evidence gate, and conditionally repeat the "
                 "four-row live development evaluation"),
        "decision": decision,
        "case": case,
        "case_interpretation": {
            "A": "state/shared-context onset bias; proximity-only gate passes offline and live",
            "B": ("proximity-only gate cannot separate oracle activity offline; current "
                  "proximity alone does not identify removable hazard evidence reliably. "
                  "Do not add another threshold or activity MLP; the next model must use "
                  "spatial uncertainty or jointly model activity and parked-field "
                  "credibility"),
            "C": "offline gate passes but onset or zero-frame failure returns live",
        }.get(case),
        "previous_decision": "REFERENCE_THRESHOLD_TRANSFER_FAILED",

        "parked_field_retrained": False,
        "parked_field_seed_reselected": False,
        "seeds_ensembled": False,
        "act_trained_or_modified": False,
        "safety_cvae_trained_or_modified": False,
        "residual_controller_changed": False,
        "magnitude_support_changed": False,
        "temporal_history_reintroduced": False,
        "dataset_modified": False,
        "live_rollouts_executed": 0,
        "live_rollouts_permitted": 20,
        "development4_executed": False,
        "confirmatory41_executed": False,

        "commits": {
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "root_commit": git("rev-parse", "HEAD"),
            "root_starting_commit": "bbfdb5e",
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

        "provenance": {"checks": provenance["check_count"],
                       "all_matched": provenance["all_matched"],
                       "frozen_model": provenance["frozen_model"]},

        "old_activity_path": {
            "dedicated_activity_head_exists":
                path_audit["activity_head"]["dedicated_activity_head_exists"],
            "state_reaches_activity": path_audit["activity_head"]["state_reaches_activity"],
            "proximity_reaches_activity":
                path_audit["activity_head"]["proximity_reaches_activity"],
            "shares_context_conditioned_tokens_with_parked_decoder":
                path_audit["activity_head"][
                    "shares_context_conditioned_tokens_with_parked_decoder"],
            "stages": path_audit["stages"],
            "finding": path_audit["finding"],
            "training_gradient_connectivity": path_audit["training_gradient_connectivity"],
        },

        "onset_attribution": {
            "classification": onset["classification"],
            "classification_evidence": onset["classification_evidence"],
            "group_sizes": onset["group_sizes"],
            "known_false_positive_count": onset["known_false_positive_count"],
            "known_false_positive_frames": onset["known_false_positive_frames"],
            "results": onset["results"],
            "harness_control": onset["harness_control"],
            "interventions": onset["interventions"],
        },

        "gate": {
            "gate_id": training["gate_id"],
            "parameter_count": training["parameter_count"],
            "parameter_budget": training["parameter_budget"],
            "permitted_inputs": ["current_closeness", "current_valid_mask",
                                 "sensor_identity"],
            "architecture": {
                "per_sensor": "Linear(128->64) SiLU Linear(64->64) SiLU + sensor embedding",
                "cross_sensor": "TransformerEncoder layers=2 d_model=64 heads=4 ff=128 "
                                "pre-norm dropout=0",
                "pooling": "concat(mean, max) over 40 sensor tokens -> 128",
                "activity_head": "Linear(128->64) SiLU Linear(64->1)",
            },
            "label_definition": training["label_definition"],
            "onset_definition": training["onset_definition"],
            "sampling": training["sampling"],
            "loss": training["loss"],
            "optimization": training["optimization"],
            "frames": training["frames"],
            "best_epoch": training["best_epoch"],
            "best_validation_metric": training["best_validation_metric"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": (hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                                  if checkpoint.is_file() else None),
            "checkpoint_committed": False,
            "checkpoint_policy": ("gate weights remain external to git; no approved "
                                  "artifact policy exists for binary weights here"),
        },

        "partition": {
            "manifest": "configs/hybrid_obstacle_prox_activity_partition_v1.json",
            "manifest_sha256": partition["manifest_sha256"],
            "source_partition": partition["source_partition"],
            "assignment_rule": partition["assignment_rule"],
            "splits": {name: {k: block[k] for k in
                              ("episode_count", "hazard_present", "hazard_absent",
                               "trajectory_count")}
                       for name, block in partition["splits"].items()},
            "excluded": partition["excluded_partitions"],
        },

        "calibration": evaluation["calibration"],
        "threshold_sweep": evaluation.get("threshold_sweep"),
        "no_threshold_satisfies": evaluation.get("no_threshold_satisfies"),
        "separability": evaluation.get("separability"),
        "post_termination_diagnostics_note":
            evaluation.get("post_termination_diagnostics_note"),

        "stages_not_reached": [
            "nested offline evaluation (step 13)",
            "reused diagnostic audit (step 14)",
            "runtime integration (step 15-16)",
            "live development rollouts (step 17-18)",
            "live approximation gates (step 19)",
            "live gross-regression gates (step 20)",
        ] if decision == "PROX_ACTIVITY_GATE_CALIBRATION_INFEASIBLE" else [],

        "constraints_honoured": {
            "parked_field_retrained": False,
            "another_parked_field_seed_selected": False,
            "seeds_ensembled": False,
            "act_trained_or_modified": False,
            "safety_cvae_trained_or_modified": False,
            "residual_controller_changed": False,
            "parked_field_physical_constraint_changed": False,
            "magnitude_support_bound_changed": False,
            "temporal_proximity_history_reintroduced": False,
            "paired_dataset_modified": False,
            "msaa_cameras_robot_obstacle_task_changed": False,
            "state_features_used_by_gate": False,
            "hand_coded_first_n_frame_suppression": False,
            "threshold_only_change": False,
            "development4_or_confirmatory41_used_for_gate_training": False,
            "confirmatory41_executed": False,
            "new_live_rollouts_executed": 0,
            "max_new_live_rollouts": 20,
            "pushed": False,
        },

        "artifacts": {
            "final_decision_md":
                "docs/HYBRID_OBSTACLE_PROX_ACTIVITY_GATE_FINAL_DECISION.md",
            "final_decision_json":
                "diagnostics_output/hybrid_obstacle_prox_activity_gate/final_decision.json",
            "provenance": str(args.provenance.relative_to(ROOT)),
            "current_activity_path": str(args.path_audit.relative_to(ROOT)),
            "onset_attribution": str(args.onset_audit.relative_to(ROOT)),
            "partition_manifest": str(args.partition.relative_to(ROOT)),
            "gate_training": str(args.training.relative_to(ROOT)),
            "gate_evaluation": str(args.evaluation.relative_to(ROOT)),
            "gate_source": "causal_parked_skin/activity_gate.py",
            "tests": "tests/test_prox_activity_gate.py",
        },
        "report_hashes": {
            "provenance_sha256": provenance["report_sha256"],
            "path_audit_sha256": path_audit["report_sha256"],
            "onset_audit_sha256": onset["report_sha256"],
            "partition_sha256": partition["manifest_sha256"],
            "training_sha256": training["report_sha256"],
            "evaluation_sha256": evaluation["report_sha256"],
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

    print(f"decision              : {decision}")
    print(f"case                  : {case}")
    print(f"onset classification  : {onset['classification']}")
    print(f"gate parameters       : {training['parameter_count']:,}")
    print("live rollouts executed: 0")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
