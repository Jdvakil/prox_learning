#!/usr/bin/env python3
"""Finalize provenance for an interrupted, irreconcilable PACT evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

TOKEN = "PACT_EXPERIMENT_INCOMPLETE"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def validate_self_hash(document: dict[str, Any], key: str) -> str:
    payload = dict(document)
    observed = payload.pop(key)
    if canonical_hash(payload) != observed:
        raise ValueError(f"{key} mismatch")
    return observed


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def artifact_record(root: Path, path: Path) -> dict[str, str]:
    try:
        display = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        display = str(path.resolve())
    return {"path": display, "sha256": file_hash(path)}


def insert_interruption_section(
    report: str,
    *,
    incident: dict[str, Any],
    analysis: dict[str, Any],
) -> str:
    marker = "## Decision\n"
    if report.count(marker) != 1:
        raise ValueError("final report does not have exactly one Decision section")
    interruption = incident["interruption"]
    reconciliation = analysis["reconciliation"]
    section = "\n".join(
        [
            "## Dispatch integrity and interruption",
            "",
            "The predeclared launch-smoke row passed in one invocation and one "
            "attempt. Its boundary, result, driver, schedule-row, checkpoint, "
            "and recorded hashes reconcile. The smoke endpoint was not "
            "interpreted.",
            "",
            "After the full pool was released, eight additional rows accepted "
            "initial observations, but the evaluator process group disappeared "
            "before any of them wrote a scientific result. All eight logs stop "
            "after initial-observation acceptance without a traceback. No "
            "kernel OOM or GPU Xid was observed in the audit window; the exact "
            "external initiator is unknown.",
            "",
            "Under the frozen boundary rule, those eight rows are terminal "
            "post-boundary failures and cannot be replaced or rerun. The "
            "remaining 951 rows were never started after irrecoverability was "
            "known, because further rollouts could not restore a valid "
            "confirmatory decision.",
            "",
            f"The frozen analyzer sees {reconciliation['valid']} valid row and "
            f"{len(reconciliation['missing'])} rows without valid results. The "
            "incident ledger resolves that latter count into "
            f"{interruption['post_boundary_terminal_count']} terminal "
            "post-boundary rows and "
            f"{interruption['never_started_count']} never-started rows. No "
            "endpoint comparison, Fisher test, or instance bootstrap was "
            "interpreted.",
            "",
            f"Incident SHA-256: `{incident['incident_sha256']}`.",
            "",
        ]
    )
    return report.replace(marker, section + "\n" + marker)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--incident", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--provenance-out", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    incident = json.loads(args.incident.read_text())
    validate_self_hash(incident, "incident_sha256")
    analysis = json.loads(args.analysis.read_text())
    decision = json.loads(args.decision.read_text())
    execution_path = args.output_root / "execution_summary.json"
    execution = json.loads(execution_path.read_text())
    schedule_path = root / "diagnostics_output/pact_vs_act/schedule.json"
    dispatch_path = (
        root / "diagnostics_output/pact_vs_act/confirmatory_dispatch_v2.json"
    )
    schedule = json.loads(schedule_path.read_text())
    dispatch = json.loads(dispatch_path.read_text())
    validate_self_hash(schedule, "schedule_sha256")
    validate_self_hash(dispatch, "dispatch_contract_sha256")

    if (
        decision.get("decision") != TOKEN
        or analysis.get("results_available") is not False
        or analysis["reconciliation"].get("reconciled") is not False
        or analysis["reconciliation"].get("expected") != 960
        or analysis["reconciliation"].get("valid") != 1
        or len(analysis["reconciliation"].get("missing", [])) != 959
    ):
        raise ValueError("frozen incomplete analysis does not match interruption")
    if (
        execution.get("complete_count") != 1
        or execution.get("post_boundary_failure_count") != 8
        or execution.get("dispatched_rows") != 9
        or len(execution.get("missing", [])) != 951
        or execution.get("scientific_schedule_reconciled") is not False
    ):
        raise ValueError("execution ledger does not match interruption")
    if canonical_hash(execution) != incident["execution_summary_sha256"]:
        raise ValueError("execution summary differs from incident")
    analysis_script = root / "scripts/analyze_pact_confirmatory.py"
    if (
        file_hash(analysis_script)
        != dispatch["frozen_inputs"]["analysis_script_sha256"]
    ):
        raise ValueError("frozen analysis script changed after schedule release")

    training_path = (
        root / "diagnostics_output/pact_vs_act/policy_training_summary_v2.json"
    )
    training = json.loads(training_path.read_text())
    checkpoint_records = []
    for record in training["records"]:
        checkpoint = Path(record["checkpoint"])
        stats = Path(record["dataset_stats"])
        if (
            file_hash(checkpoint) != record["checkpoint_sha256"]
            or file_hash(stats) != record["dataset_stats_sha256"]
        ):
            raise ValueError("policy checkpoint or statistics hash changed")
        checkpoint_records.append(
            {
                "arm": record["arm"],
                "seed": record["seed"],
                "best_epoch": record["best_epoch"],
                "best_validation_loss": record["best_validation_loss"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": record["checkpoint_sha256"],
                "dataset_stats": str(stats),
                "dataset_stats_sha256": record["dataset_stats_sha256"],
            }
        )
    surface_path = Path(training["surface_encoder"])
    if file_hash(surface_path) != training["surface_encoder_sha256"]:
        raise ValueError("surface encoder hash changed")

    base_report = args.report.read_text()
    report = insert_interruption_section(
        base_report,
        incident=incident,
        analysis=analysis,
    )
    if report.rstrip().splitlines()[-1] != TOKEN:
        raise ValueError("final report token is not the last nonblank line")
    args.report.write_text(report)

    small_paths = {
        "candidate_manifest": root
        / "configs/pact_collision_candidate_manifest_v2.json",
        "machine_preregistration": root
        / "configs/pact_collision_environment_v2.json",
        "narrative_preregistration": root
        / "docs/PACT_VS_ACT_REMEDIATION_PREREGISTRATION.md",
        "environment_gate": root
        / "diagnostics_output/pact_vs_act/environment_gate.json",
        "environment_report": root / "docs/PACT_ENVIRONMENT_ADEQUACY.md",
        "full_conversion": root
        / "diagnostics_output/pact_vs_act/full_conversion_encoded_v2.json",
        "full_split": root
        / "diagnostics_output/pact_vs_act/full_act_split_encoded_v2.json",
        "surface_encoder_report": root
        / "diagnostics_output/pact_vs_act/surface_encoder_report_v2.json",
        "surface_token_encoding": root
        / "diagnostics_output/pact_vs_act/surface_token_encoding_v2.json",
        "policy_training_summary": training_path,
        "confirmatory_schedule": schedule_path,
        "confirmatory_dispatch": dispatch_path,
        "confirmatory_interruption": args.incident,
        "execution_summary": execution_path,
        "frozen_analysis_script": analysis_script,
        "analysis": args.analysis,
        "final_decision": args.decision,
        "final_report": args.report,
    }
    provenance: dict[str, Any] = {
        "schema_version": "pact_vs_act_provenance_v3",
        "experiment_stage": "confirmatory_dispatch_interrupted",
        "decision": TOKEN,
        "branch": "experiment/pact-vs-act-remediation-v2",
        "source_commits_at_finalization": {
            "root": git_head(root),
            "act": git_head(root / "submodules/act"),
            "molmospaces": git_head(root / "submodules/molmospaces"),
        },
        "small_artifacts": {
            name: artifact_record(root, path) for name, path in small_paths.items()
        },
        "surface_encoder": {
            "path": str(surface_path),
            "sha256": training["surface_encoder_sha256"],
            "frozen": True,
        },
        "policy_checkpoints": checkpoint_records,
        "pact_zero": {
            "separately_trained": False,
            "checkpoint_aliases_pact_by_seed": True,
            "inference_proximity_zeroed": True,
        },
        "confirmatory_execution": {
            "output_root": str(args.output_root),
            "expected_rows": 960,
            "workers": 8,
            "smoke_complete": 1,
            "post_boundary_terminal_failures": 8,
            "never_started": 951,
            "rows_rerun": 0,
            "scientific_schedule_reconciled": False,
            "endpoint_outcomes_interpreted": False,
            "incident_sha256": incident["incident_sha256"],
            "execution_summary_sha256": file_hash(execution_path),
        },
        "analysis_integrity": {
            "frozen_script_sha256": file_hash(analysis_script),
            "matches_dispatch_contract": True,
            "results_available": False,
            "statistical_comparisons_interpreted": False,
        },
        "protected_chain": {
            "modified_by_pact_work": False,
            "preexisting_worktree_changes_preserved": True,
            "used_as_pact_evidence": False,
            "confirmatory41_touched_by_pact_work": False,
        },
    }
    provenance["provenance_sha256"] = canonical_hash(provenance)
    write_json(args.provenance_out, provenance)
    print(provenance["provenance_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
