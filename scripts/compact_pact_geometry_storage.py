#!/usr/bin/env python3
"""Continuously compact every completed geometry-evaluation rollout payload."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import compact_pact_contact_storage as storage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    payload = dict(schedule)
    observed = payload.pop("schedule_sha256", None)
    if observed != storage.canonical_hash(payload):
        raise SystemExit("schedule self-hash mismatch")
    dispatch = json.loads(args.dispatch.read_text())
    storage_rule = dispatch.get("storage", {})
    if (
        storage_rule.get("compact_every_completed_row") is not True
        or storage_rule.get("outcome_based_selection") is not False
        or storage_rule.get("retain_raw_rollout_payloads") is not False
    ):
        raise SystemExit("frozen geometry storage rule changed")
    output_root = args.output_root.resolve()
    rows = schedule["rows"]
    while True:
        compacted = 0
        last_error = None
        for row in rows:
            row_dir = output_root / row["output_relpath"]
            if not (row_dir / "result.json").is_file() or not (row_dir / "driver_result.json").is_file():
                continue
            try:
                record = storage.compact_row(row, output_root)
                if record.get("status") == "complete":
                    compacted += 1
            except Exception as error:  # noqa: BLE001 - durable compaction heartbeat
                last_error = f"{type(error).__name__}: {error}"
                break
        heartbeat = {
            "schema_version": "pact_geometry_storage_heartbeat_v1",
            "schedule_sha256": schedule["schedule_sha256"],
            "compacted_count": compacted,
            "expected_count": len(rows),
            "last_error": last_error,
            "outcome_fields_emitted": False,
        }
        heartbeat["storage_heartbeat_sha256"] = storage.canonical_hash(heartbeat)
        storage.write_json_atomic(output_root / "storage_compactor_heartbeat.json", heartbeat)
        if last_error:
            raise RuntimeError(last_error)
        if (output_root / "full_execution_summary.json").is_file() and compacted == len(rows):
            summary = {
                "schema_version": "pact_geometry_storage_summary_v1",
                "schedule_sha256": schedule["schedule_sha256"],
                "compacted_count": compacted,
                "expected_count": len(rows),
                "complete": True,
                "outcome_based_selection": False,
            }
            summary["storage_summary_sha256"] = storage.canonical_hash(summary)
            storage.write_json_atomic(output_root / "storage_compaction_summary.json", summary)
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
