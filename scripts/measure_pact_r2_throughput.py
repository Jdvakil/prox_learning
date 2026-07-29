#!/usr/bin/env python3
"""Measure outcome-blind R2 throughput over the frozen first 20 minutes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from run_pact_confirmatory_schedule import write_json_atomic

WINDOW_MINUTES = 20


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch-contract", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    schedule = json.loads(args.schedule.read_text())
    contract = json.loads(args.dispatch_contract.read_text())
    payload = dict(contract)
    observed_contract_hash = payload.pop("dispatch_contract_sha256")
    if canonical_hash(payload) != observed_contract_hash:
        raise ValueError("dispatch contract self-hash mismatch")
    if (
        schedule["schedule_sha256"]
        != contract["scientific_schedule"]["schedule_sha256"]
        or schedule["workers"] != 8
        or schedule["rollouts"] != 960
        or Path(contract["execution"]["output_root"]).resolve()
        != args.output_root.resolve()
    ):
        raise ValueError("throughput inputs differ from frozen R2 dispatch")
    state = json.loads((args.output_root / "supervisor_state.json").read_text())
    start_value = state.get("full_dispatch_started_utc")
    if state.get("mode") != "full" or start_value is None:
        raise ValueError("full R2 dispatch has not started")
    started = parse_utc(start_value)
    window_end = started + timedelta(minutes=WINDOW_MINUTES)
    measured = datetime.now(timezone.utc)
    if measured < window_end:
        remaining = (window_end - measured).total_seconds()
        raise ValueError(
            f"first-20-minute window is incomplete ({remaining:.1f}s remain)"
        )
    ledger = json.loads(
        (args.output_root / "completion_ledger.json").read_text()
    )
    if ledger["schedule_sha256"] != schedule["schedule_sha256"]:
        raise ValueError("completion ledger schedule mismatch")
    completions = [
        {
            "rollout_id": item["rollout_id"],
            "schedule_index": int(item["schedule_index"]),
            "completed_utc": item["completed_utc"],
        }
        for item in ledger["completions"]
    ]
    in_window = [
        item
        for item in completions
        if started < parse_utc(item["completed_utc"]) <= window_end
    ]
    complete_by_window_end = [
        item for item in completions if parse_utc(item["completed_utc"]) <= window_end
    ]
    rate = len(in_window) / WINDOW_MINUTES
    remaining_rows = 960 - len(complete_by_window_end)
    eta = (
        window_end + timedelta(minutes=remaining_rows / rate)
        if rate > 0
        else None
    )
    artifact: dict[str, Any] = {
        "schema_version": "pact_r2_throughput_first_20_minutes_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "dispatch_contract_sha256": observed_contract_hash,
        "workers": 8,
        "window_minutes": WINDOW_MINUTES,
        "full_dispatch_started_utc": format_utc(started),
        "window_ended_utc": format_utc(window_end),
        "measured_utc": format_utc(measured),
        "completed_during_window": len(in_window),
        "completed_schedule_rows_by_window_end": len(complete_by_window_end),
        "throughput_rollouts_per_minute": rate,
        "remaining_rows_at_window_end": remaining_rows,
        "revised_expected_finish_utc": format_utc(eta) if eta else None,
        "completion_identities": sorted(
            in_window, key=lambda item: item["schedule_index"]
        ),
        "source_fields": [
            "schedule_index",
            "rollout_id",
            "completed_utc",
        ],
        "result_files_opened": 0,
        "endpoint_fields_read": False,
        "schedule_changed": False,
        "worker_count_changed": False,
    }
    artifact["throughput_sha256"] = canonical_hash(artifact)
    write_json_atomic(args.output, artifact)
    print(
        f"{rate:.3f} rollouts/min; revised finish "
        f"{artifact['revised_expected_finish_utc']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
