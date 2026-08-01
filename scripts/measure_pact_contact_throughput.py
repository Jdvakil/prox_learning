#!/usr/bin/env python3
"""Record outcome-blind throughput for the first 20 contact-run minutes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from run_pact_confirmatory_schedule import canonical_hash, write_json_atomic


EXPECTED_ROWS = 1200


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--measurement-minutes", type=int, default=20)
    args = parser.parse_args()
    root = args.output_root.resolve()
    schedule = json.loads(args.schedule.read_text())
    schedule_payload = dict(schedule)
    schedule_sha = schedule_payload.pop("schedule_sha256", None)
    if schedule_sha != canonical_hash(schedule_payload):
        raise SystemExit("contact schedule self-hash mismatch")
    workers = int(schedule["workers"])
    receipt = json.loads((root / "full_launcher_receipt.json").read_text())
    started = parse_utc(receipt["launched_utc"])
    cutoff = started + timedelta(minutes=args.measurement_minutes)
    now = datetime.now(timezone.utc)
    if now < cutoff:
        raise SystemExit(f"measurement window not complete until {cutoff.isoformat()}")
    ledger = json.loads((root / "completion_ledger.json").read_text())
    completions = [
        item
        for item in ledger["completions"]
        if started <= parse_utc(item["completed_utc"]) <= cutoff
    ]
    completed = len(completions)
    rate = completed / args.measurement_minutes
    remaining = max(0, EXPECTED_ROWS - len(ledger["completions"]))
    artifact = {
        "schema_version": "pact_contact_endpoint_throughput_v1",
        "measurement_started_utc": started.isoformat().replace("+00:00", "Z"),
        "measurement_cutoff_utc": cutoff.isoformat().replace("+00:00", "Z"),
        "measurement_minutes": args.measurement_minutes,
        "completed_schedule_indices": sorted(int(item["schedule_index"]) for item in completions),
        "completed_rows": completed,
        "rows_per_minute": rate,
        "minutes_per_row": 1.0 / rate if rate > 0.0 else None,
        "remaining_rows_at_measurement": remaining,
        "projected_remaining_hours": remaining / rate / 60.0 if rate > 0.0 else None,
        "workers": workers,
        "schedule_sha256": schedule_sha,
        "worker_amendment_sha256": schedule["worker_amendment_sha256"],
        "source": "completion ledger timestamps and row identities only",
        "endpoint_fields_read": False,
        "schedule_changed": False,
        "workers_changed": workers != 8,
        "eta_source": "remeasured first-20-minute completion rate",
    }
    artifact["throughput_sha256"] = canonical_hash(artifact)
    write_json_atomic(root / "throughput_first_20_minutes.json", artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
