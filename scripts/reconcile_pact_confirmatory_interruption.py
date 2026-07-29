#!/usr/bin/env python3
"""Reconcile an interrupted PACT dispatch without launching or rerunning any row."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_pact_confirmatory_schedule as runner


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconcile_rows(
    schedule: dict[str, Any],
    *,
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconcile only rows that already have boundary/result/driver artifacts."""
    terminal: list[dict[str, Any]] = []
    not_started: list[dict[str, Any]] = []
    for row in schedule["rows"]:
        row_dir = output_root / row["output_relpath"]
        driver_path = row_dir / "driver_result.json"
        result_path = row_dir / "result.json"
        boundary_path = row_dir / "initial_observation_accepted.json"
        if not (driver_path.exists() or result_path.exists() or boundary_path.exists()):
            not_started.append(row)
            continue
        if result_path.exists() and not boundary_path.exists():
            raise RuntimeError(f"{result_path}: result exists without boundary marker")
        if driver_path.exists() and not (result_path.exists() or boundary_path.exists()):
            raise RuntimeError(f"{driver_path}: driver exists without scientific artifacts")
        # run_one cannot launch here because the guard above requires an existing
        # terminal driver, scientific result, or scientific-boundary marker.
        driver = runner.run_one(
            row,
            manifest_path="RECONCILIATION_ONLY_NO_SUBPROCESS",
            output_root=str(output_root),
            save_video=False,
            single_attempt=True,
        )
        terminal.append({"row": row, "driver": driver})
    return terminal, not_started


def build_documents(
    schedule: dict[str, Any],
    contract: dict[str, Any],
    *,
    output_root: Path,
    terminal: list[dict[str, Any]],
    not_started: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_status: dict[str, int] = {}
    for item in terminal:
        status = item["driver"]["status"]
        by_status[status] = by_status.get(status, 0) + 1
    expected_terminal = {"complete": 1, "post_boundary_failure": 8}
    if by_status != expected_terminal:
        raise RuntimeError(f"unexpected interrupted terminal ledger: {by_status}")
    if len(not_started) != 951:
        raise RuntimeError(f"expected 951 never-started rows, got {len(not_started)}")

    smoke = runner.validate_launch_smoke(
        schedule=schedule,
        contract=contract,
        output_root=output_root,
    )
    post_boundary = []
    for item in terminal:
        if item["driver"]["status"] != "post_boundary_failure":
            continue
        row = item["row"]
        row_dir = output_root / row["output_relpath"]
        boundary_path = row_dir / "initial_observation_accepted.json"
        log_path = row_dir / "process_attempt_000.log"
        post_boundary.append(
            {
                "schedule_index": row["schedule_index"],
                "rollout_id": row["rollout_id"],
                "schedule_row_sha256": row["schedule_row_sha256"],
                "instance_episode_id": row["instance_episode_id"],
                "arm": row["arm"],
                "checkpoint_seed": row["checkpoint_seed"],
                "boundary_marker_sha256": file_hash(boundary_path),
                "process_log_sha256": file_hash(log_path),
                "scientific_result_written": False,
                "rerun": False,
                "terminal_status": "post_boundary_failure",
            }
        )

    terminal_ids = {item["row"]["rollout_id"] for item in terminal}
    not_started_ids = [row["rollout_id"] for row in not_started]
    noncomplete = [
        row["rollout_id"]
        for row in schedule["rows"]
        if row["rollout_id"] not in {
            item["row"]["rollout_id"]
            for item in terminal
            if item["driver"]["status"] == "complete"
        }
    ]
    nonterminal = [
        row["rollout_id"]
        for row in schedule["rows"]
        if row["rollout_id"] not in terminal_ids
    ]
    execution = {
        "schema_version": "pact_schedule_execution_v2",
        "schedule_sha256": schedule["schedule_sha256"],
        "dispatch_contract_sha256": contract["dispatch_contract_sha256"],
        "launch_smoke_sha256": smoke["launch_smoke_sha256"],
        "launch_smoke_rollout_id": smoke["rollout_id"],
        "workers": 8,
        "pilot_gate_mode": False,
        "expected": len(schedule["rows"]),
        "dispatched_rows": len(terminal),
        "complete_count": by_status["complete"],
        "post_boundary_failure_count": by_status["post_boundary_failure"],
        "pre_observation_infrastructure_failures": 0,
        "missing": not_started_ids,
        "noncomplete": noncomplete,
        "nonterminal": nonterminal,
        "terminal_ledger_reconciled": False,
        "scientific_schedule_reconciled": False,
        "stopped_early": True,
        "stop_reason": (
            "Eight rows accepted initial observations but the evaluator process "
            "group disappeared before writing results. They are terminal and "
            "cannot be rerun; the remaining rows were not launched because the "
            "confirmatory decision was already forced to PACT_EXPERIMENT_INCOMPLETE."
        ),
    }
    incident: dict[str, Any] = {
        "schema_version": "pact_confirmatory_interruption_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "dispatch_contract_sha256": contract["dispatch_contract_sha256"],
        "output_root": str(output_root),
        "smoke_audit": {
            "passed": True,
            "smoke_invocations": smoke["smoke_invocations"],
            "rollout_id": smoke["rollout_id"],
            "launch_smoke_sha256": smoke["launch_smoke_sha256"],
            "scientific_result_sha256": smoke["scientific_result_sha256"],
            "driver_result_sha256": smoke["driver_result_sha256"],
            "endpoint_fields_inspected_during_audit": False,
        },
        "interruption": {
            "exact_external_initiator": "unknown",
            "kernel_oom_or_gpu_xid_observed_in_audit_window": False,
            "post_boundary_terminal_rows": post_boundary,
            "post_boundary_terminal_count": len(post_boundary),
            "rows_rerun": 0,
            "never_started_count": len(not_started),
            "remaining_rows_launched_after_irrecoverability_known": 0,
        },
        "decision_consequence": {
            "scientific_schedule_reconciled": False,
            "endpoint_analysis_permitted": False,
            "required_token": "PACT_EXPERIMENT_INCOMPLETE",
        },
        "execution_summary_sha256": canonical_hash(execution),
    }
    incident["incident_sha256"] = canonical_hash(incident)
    return execution, incident


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch-contract", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--incident-out", required=True, type=Path)
    args = parser.parse_args()

    schedule = json.loads(args.schedule.read_text())
    payload = dict(schedule)
    observed = payload.pop("schedule_sha256")
    if canonical_hash(payload) != observed:
        raise SystemExit("schedule self-hash mismatch")
    contract = runner.load_dispatch_contract(
        args.dispatch_contract,
        schedule,
        manifest_path=args.manifest,
        output_root=args.output_root,
    )
    if runner.protected_eval_processes():
        raise SystemExit("protected confirmatory evaluator is active")
    active_pact = []
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = entry.read_bytes()
        except OSError:
            continue
        if b"eval_pact_collision_row.py" in command:
            active_pact.append(entry.parent.name)
    if active_pact:
        raise SystemExit(f"PACT evaluators are still active: {active_pact}")

    terminal, not_started = reconcile_rows(
        schedule,
        output_root=args.output_root,
    )
    execution, incident = build_documents(
        schedule,
        contract,
        output_root=args.output_root,
        terminal=terminal,
        not_started=not_started,
    )
    write_json(args.output_root / "execution_summary.json", execution)
    write_json(args.incident_out, incident)
    print(incident["incident_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
