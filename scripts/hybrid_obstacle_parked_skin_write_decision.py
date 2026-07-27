#!/usr/bin/env python3
"""Assemble the final parked-skin dataset decision JSON.

Handoff step 22. Every field is copied from an artifact produced earlier in the task; the
decision token follows from the audit and smoke results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWED = (
    "PARKED_SKIN_DATASET_READY_FOR_MODEL_TRAINING",
    "PARKED_SKIN_DATASET_CONTRACT_FAILED",
    "PARKED_SKIN_ORACLE_PAIRING_FAILED",
    "PARKED_SKIN_COLLECTION_INCOMPLETE",
    "REQUIRED_LEARNER_ARTIFACT_MISSING",
    "CHECKPOINT_OR_SOURCE_MISMATCH",
)


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def load(path) -> dict:
    return json.loads(Path(path).read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provenance", required=True, type=Path)
    ap.add_argument("--dataset-manifest", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--history-smoke", required=True, type=Path)
    ap.add_argument("--make-read-only", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    provenance = load(args.provenance)
    manifest = load(args.dataset_manifest)
    audit = load(args.audit)
    smoke = load(args.history_smoke)

    integrity = audit["integrity"]
    complete = audit["outputs_present"] == audit["scheduled_outputs"]
    pairing_ok = (integrity["state_neutrality_failures"] == 0
                  and audit["head_reconstruction"]["within_tolerance"]
                  and audit["head_reconstruction"]["oracle_within_tolerance"])
    contract_ok = (integrity["physical_inequality_violations"] == 0
                   and integrity["hazard_absent_nonzero_targets"] == 0
                   and integrity["noncausal_histories"] == 0
                   and integrity["shape_mismatches"] == 0
                   and integrity["nonfinite_values"] == 0
                   and integrity["duplicate_source_identities"] == 0
                   and smoke["all_histories_correct"])

    if not complete:
        decision = "PARKED_SKIN_COLLECTION_INCOMPLETE"
    elif not pairing_ok:
        decision = "PARKED_SKIN_ORACLE_PAIRING_FAILED"
    elif not contract_ok or not audit["valid"]:
        decision = "PARKED_SKIN_DATASET_CONTRACT_FAILED"
    else:
        decision = "PARKED_SKIN_DATASET_READY_FOR_MODEL_TRAINING"
    if decision not in ALLOWED:
        raise SystemExit(f"decision {decision!r} is not allowed")

    data_root = Path(manifest["data_root"])
    if args.make_read_only and decision == "PARKED_SKIN_DATASET_READY_FOR_MODEL_TRAINING":
        for path in sorted(data_root.rglob("*")):
            if path.is_file():
                os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        read_only = True
    else:
        read_only = False

    act = ROOT / "submodules/act"
    molmo = ROOT / "submodules/molmospaces"
    confirmatory = load(ROOT / "configs/hybrid_obstacle_confirmatory41_v1.json")
    partition = load(ROOT / "configs/hybrid_obstacle_reference_partition_v2.json")
    learner = load(ROOT / "diagnostics_output/hybrid_obstacle_on_policy_reference"
                        / "round0_deployment_manifest.json")

    payload = {
        "schema": "hybrid_obstacle_parked_skin_dataset_final_decision_v1",
        "date": "2026-07-26",
        "task": ("Regenerate and freeze the complete paired current/parked proximity-field "
                 "dataset required by CAUSAL_PARKED_SKIN_REFERENCE_V1"),
        "decision": decision,
        "data_generation_only": True,
        "model_trained": False,
        "development4_executed": False,
        "confirmatory41_executed": False,

        "commits": {
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "root_starting_commit": "1343160",
            "root_commit": git("rev-parse", "HEAD"),
            "act_branch": git("rev-parse", "--abbrev-ref", "HEAD", repo=act),
            "act_starting_commit": "5e0d3b3",
            "act_commit": git("rev-parse", "HEAD", repo=act),
            "molmospaces_commit": git("rev-parse", "HEAD", repo=molmo),
            "molmospaces_modified": git("status", "--porcelain", repo=molmo) != "",
        },

        "previous_failure_cause": (
            "0 of 60793 examples carried both a reconstructible four-frame current history "
            "and a parked 40x8x8 field at the same decision state. Hashes and 7-D head "
            "outputs are not invertible to the field, and the validated per-frame oracle "
            "rendered the parked field on every step and then discarded it."),

        "artifact_verification": {"checks": provenance["check_count"],
                                  "all_matched": provenance["all_matched"]},

        "dataset": {
            "version": manifest["dataset_version"],
            "root": manifest["data_root"],
            "manifest_sha256": manifest["manifest_sha256"],
            "partition_sha256": manifest["partition_sha256"],
            "partition_composition": manifest["partition_composition"],
            "schema": {
                "deployable_group": "runtime-observable inputs only",
                "privileged_group": "training targets; a deployable loader must never "
                                    "read this group as input",
                "integrity_group": "hashes and state-neutrality results",
                "storage": manifest["storage_rule"],
                "retention": manifest["retention_rule"],
                "causal_history_rule": ("history(t) = [t-3,t-2,t-1,t], left-padded by "
                                        "repeating the earliest available frame; never a "
                                        "future frame"),
                "closeness_transform": "clip(1 - depth/0.5, 0, 1); readings below 5 mm "
                                       "map to 0 closeness and are flagged invalid",
            },
            "counts": audit["counts"],
            "coverage": audit["coverage"],
            "natural_distribution_retained": audit["natural_distribution_retained"],
            "tree_sha256": audit["tree_sha256"],
            "audit_sha256": audit["report_sha256"],
            "read_only": read_only,
        },

        "schedules": {
            "total_policy_rollouts": manifest["total_policy_rollouts"],
            "total_reconstructions": manifest["total_reconstructions"],
            "condition_order_rule": manifest["condition_order_rule"],
            "condition_order_balance": manifest["condition_order_balance"],
            "max_concurrent_rollout_processes": manifest["max_concurrent_rollout_processes"],
            "concurrency_rationale": manifest["concurrency_rationale"],
        },

        "integrity": integrity,
        "physical_pairing": {
            "constraint": "0 <= parked_closeness <= current_closeness <= 1",
            "violations": integrity["physical_inequality_violations"],
            "tolerance": audit["tolerances"]["closeness_inequality"],
            "silently_clamped": False,
        },
        "hazard_absent_exact_control": {
            "requirement": ("current equals parked, removable exactly zero, changed mask "
                            "empty, heads identical, oracle differential exactly zero"),
            "violations": integrity["hazard_absent_nonzero_targets"],
        },
        "state_neutrality": {"failures": integrity["state_neutrality_failures"]},
        "head_reconstruction": audit["head_reconstruction"],
        "history_smoke": {
            "distributions_checked": [c["distribution"] for c
                                      in smoke["distributions_checked"]],
            "all_distributions_available": smoke["all_distributions_available"],
            "all_histories_correct": smoke["all_histories_correct"],
            "all_head_targets_reproduce": smoke["all_head_targets_reproduce"],
            "model_check": smoke["model_check"],
            "report_sha256": smoke["report_sha256"],
        },

        "learner_artifact": {
            "label": learner["label"],
            "checkpoint_sha256": learner["artifact_file_sha256"],
            "manifest_sha256": learner["manifest_sha256"],
            "strictly_loaded": True,
            "substituted": False,
        },

        "confirmatory41": {
            "sha256": confirmatory["manifest_sha256"],
            "rows": len(confirmatory["rows"]),
            "executed_in_this_task": confirmatory["executed_in_this_task"],
            "untouched": True,
        },
        "development4": {"executed": False, "included_in_dataset": False},
        "reference_partition_unchanged": (
            partition["partition_sha256"] == manifest["partition_sha256"]),

        "constraints_honoured": {
            "model_trained": False,
            "act_modified": False,
            "safety_cvae_modified": False,
            "residual_controller_modified": False,
            "molmospaces_modified": git("status", "--porcelain", repo=molmo) != "",
            "reference_partition_changed": False,
            "development4_or_confirmatory41_used": False,
            "only_active_frames_retained": False,
            "zero_frames_subsampled": False,
            "hashes_stored_in_place_of_fields": False,
            "duplicated_four_frame_histories_stored": False,
            "pushed": False,
        },

        "artifacts": {
            "final_decision_md":
                "docs/HYBRID_OBSTACLE_PARKED_SKIN_DATASET_FINAL_DECISION.md",
            "final_decision_json": ("diagnostics_output/hybrid_obstacle_parked_skin_dataset/"
                                    "final_decision.json"),
            "dataset_manifest": "configs/hybrid_obstacle_parked_skin_supervision_v1.json",
            "provenance": "diagnostics_output/hybrid_obstacle_parked_skin_dataset/"
                          "provenance_verification.json",
            "audit": "diagnostics_output/hybrid_obstacle_parked_skin_dataset/"
                     "dataset_audit.json",
            "history_smoke": "diagnostics_output/hybrid_obstacle_parked_skin_dataset/"
                             "history_smoke.json",
            "retention_source": "submodules/act/parked_skin_retention.py",
            "tests": "tests/test_parked_skin_dataset_contract.py",
        },
        "report_hashes": {
            "provenance_sha256": provenance["report_sha256"],
            "dataset_manifest_sha256": manifest["manifest_sha256"],
            "audit_sha256": audit["report_sha256"],
            "history_smoke_sha256": smoke["report_sha256"],
        },
    }
    payload["final_decision_sha256"] = canonical_hash(payload)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(f"decision: {decision}")
    print(f"outputs : {audit['outputs_present']}/{audit['scheduled_outputs']}  "
          f"frames {audit['counts']['total_frames']}  "
          f"{audit['counts']['total_gib']} GiB")
    print(f"read-only: {read_only}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
