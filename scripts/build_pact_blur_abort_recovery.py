#!/usr/bin/env python3
"""Freeze an outcome-blind all-inflight recovery after supervisor abort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_pact_confirmatory_schedule import canonical_hash
from run_pact_frontend_screen_supervisor import process_identity, utc_now


def validate_hash(document: dict, key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != canonical_hash(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return str(observed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--completion-ledger", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to replace an existing recovery event")
    schedule = json.loads(args.schedule.read_text())
    schedule_sha = validate_hash(schedule, "schedule_sha256", "schedule")
    state = json.loads(args.state.read_text())
    validate_hash(state, "state_sha256", "aborted supervisor state")
    ledger = json.loads(args.completion_ledger.read_text())
    validate_hash(ledger, "completion_ledger_sha256", "completion ledger")
    if (
        state.get("status") != "aborted"
        or not str(state.get("abort_reason", "")).startswith(
            "isolated_post_observation_failure:"
        )
        or state.get("schedule_sha256") != schedule_sha
        or ledger.get("schedule_sha256") != schedule_sha
    ):
        raise ValueError("source state is not the frozen blur supervisor abort")
    completed_ids = {row["rollout_id"] for row in ledger["completions"]}
    if len(completed_ids) != 33:
        raise ValueError("recovery was designed at exactly 33 completed rows")
    schedule_by_id = {row["rollout_id"]: row for row in schedule["rows"]}
    rows = []
    for active in state["active_cohort"]:
        row = schedule_by_id.get(active["rollout_id"])
        if row is None or row["schedule_row_sha256"] != active["schedule_row_sha256"]:
            raise ValueError("active cohort row does not match the schedule")
        row_dir = args.output_root / row["output_relpath"]
        result_present = (row_dir / "result.json").exists()
        if result_present or row["rollout_id"] in completed_ids:
            raise ValueError("recovery cohort contains a scientific result")
        if process_identity(int(active["pid"])) is not None:
            raise ValueError("recovery cohort still has a live evaluator")
        log_path = Path(active["process_log"])
        if not log_path.is_file():
            raise ValueError("recovery cohort process log is absent")
        rows.append(
            {
                "schedule_index": row["schedule_index"],
                "rollout_id": row["rollout_id"],
                "schedule_row_sha256": row["schedule_row_sha256"],
                "attempt_index": active["attempt_index"],
                "pid": active["pid"],
                "process_log": str(log_path.resolve()),
                "process_alive_at_freeze": False,
                "result_present": False,
                "initial_observation_accepted": (
                    row_dir / "initial_observation_accepted.json"
                ).exists(),
            }
        )
    rows.sort(key=lambda row: row["schedule_index"])
    if len(rows) != 12:
        raise ValueError("aborted active cohort is not exactly twelve rows")
    event = {
        "schema_version": "pact_frontend_screen_group_recovery_v1",
        "schedule_sha256": schedule_sha,
        "qualifying_indiscriminate_termination": True,
        "all_inflight_rows_rerun": True,
        "result_absent_for_all": True,
        "active_cohort_size": len(rows),
        "rows": rows,
        "frozen_utc": utc_now(),
        "resume_after_supervisor_abort": True,
        "source_supervisor_state": str(args.state.resolve()),
        "source_supervisor_state_sha256": state["state_sha256"],
        "source_abort_reason": state["abort_reason"],
        "completed_rows_preserved": len(completed_ids),
        "scientific_result_files_opened": False,
        "endpoint_fields_read": False,
        "recovery_scope": "all twelve rows active when the supervisor aborted; no subset",
    }
    event["recovery_event_sha256"] = canonical_hash(event)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    print(event["recovery_event_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
