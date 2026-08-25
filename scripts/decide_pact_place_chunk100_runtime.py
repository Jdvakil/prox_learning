#!/usr/bin/env python3
"""Apply the frozen smoke and runtime-cut rules for chunk-100 place eval."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("experiment start must include a UTC offset")
    return parsed.astimezone(dt.timezone.utc)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--experiment-started-utc", required=True)
    args = parser.parse_args()
    smoke = json.loads((args.output_root / "smoke_launcher_summary.json").read_text())
    if smoke["jobs_requested"] != 6 or smoke["jobs_complete"] != 6 or smoke["errors"]:
        raise SystemExit("smoke did not reconcile all six required rollouts")
    closes = sum(bool(row["gripper_close_commanded"]) for row in smoke["results"])
    durations = [float(row["wall_clock_seconds"]) for row in smoke["results"]]
    measured_minutes = statistics.mean(durations) / 60.0
    projected_t6_hours = 120 * measured_minutes / 10 / 60
    now = dt.datetime.now(dt.timezone.utc)
    elapsed_hours = (now - parse_utc(args.experiment_started_utc)).total_seconds() / 3600
    if closes == 0:
        decision = "STOP_NO_GRIPPER_CLOSE"
        full_arms: list[str] = []
    elif elapsed_hours + projected_t6_hours > 7.5:
        decision = "RUN_TRAINED_ARMS_ONLY"
        full_arms = ["ACT", "PACT"]
    else:
        decision = "RUN_ALL_THREE_ARMS"
        full_arms = ["ACT", "PACT", "PACT_PERMUTED"]
    record: dict[str, Any] = {
        "schema_version": "pact_place_chunk100_runtime_decision_v1",
        "experiment_started_utc": args.experiment_started_utc,
        "decided_utc": now.isoformat().replace("+00:00", "Z"),
        "smoke_rollouts": 6,
        "smoke_gripper_close_commanded": closes,
        "smoke_wall_clock_seconds": durations,
        "measured_minutes_per_rollout_mean": measured_minutes,
        "projected_T6_hours": projected_t6_hours,
        "elapsed_since_start_hours": elapsed_hours,
        "elapsed_plus_projected_T6_hours": elapsed_hours + projected_t6_hours,
        "runtime_cut_threshold_hours": 7.5,
        "decision": decision,
        "full_arms": full_arms,
    }
    record["decision_sha256"] = canonical_hash(record)
    write_json_atomic(args.output_root / "runtime_decision.json", record)
    print(json.dumps(record, sort_keys=True))
    return 2 if decision == "STOP_NO_GRIPPER_CLOSE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
