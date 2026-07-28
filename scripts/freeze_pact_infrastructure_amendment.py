#!/usr/bin/env python3
"""Freeze the pre-observation retry amendment and pilot launch-smoke contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_SCHEDULE_SHA256 = (
    "e0515adf10a12cca22412d349d37b56ec5400446894b450b0e84edbe139b564e"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "4fca3b0b0542d6ae65c7d44f1fd562cd376199532f91b08aaf5722109a858db6"
)
OLD_ERROR = (
    "FileNotFoundError: [Errno 2] No such file or directory: "
    "'configs/pact_collision_candidate_manifest_v2.json'"
)


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


def validate_self_hash(document: dict[str, Any], key: str) -> str:
    payload = dict(document)
    observed = payload.pop(key)
    if canonical_hash(payload) != observed:
        raise ValueError(f"{key} mismatch")
    return observed


def old_dispatch_record(root: Path, schedule: dict[str, Any]) -> dict[str, Any]:
    summary_path = root / "execution_summary.json"
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("schedule_sha256") != EXPECTED_SCHEDULE_SHA256
        or summary.get("expected") != 64
        or summary.get("complete_count") != 0
        or not summary.get("terminal_ledger_reconciled")
        or summary.get("scientific_schedule_reconciled")
    ):
        raise ValueError("old pilot dispatch does not match the recorded failure")

    expected = {row["rollout_id"] for row in schedule["rows"]}
    drivers = []
    matching_errors = 0
    for path in sorted(root.glob("rows/*/driver_result.json")):
        driver = json.loads(path.read_text())
        drivers.append(
            {
                "rollout_id": driver["rollout_id"],
                "schedule_row_sha256": driver["schedule_row_sha256"],
                "status": driver["status"],
                "driver_sha256": file_hash(path),
                "process_log_sha256": file_hash(Path(driver["process_log"])),
            }
        )
        matching_errors += OLD_ERROR in Path(driver["process_log"]).read_text(
            errors="replace"
        )
    if (
        len(drivers) != 64
        or {item["rollout_id"] for item in drivers} != expected
        or {item["status"] for item in drivers} != {"invocation_failure"}
        or matching_errors != 64
        or list(root.glob("rows/*/result.json"))
    ):
        raise ValueError("old pilot failure ledger is not the frozen 64-row ledger")
    return {
        "output_root": str(root.resolve()),
        "execution_summary_sha256": file_hash(summary_path),
        "driver_ledger_sha256": canonical_hash(drivers),
        "attempts": 64,
        "initial_observations": 0,
        "actions": 0,
        "scientific_outcomes": 0,
        "common_root_cause": "relative manifest path invalid from evaluator cwd",
        "matching_root_cause_logs": matching_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--machine-preregistration", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--old-output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    preregistration_text = args.preregistration.read_text()
    machine = json.loads(args.machine_preregistration.read_text())
    amendment = machine["infrastructure_health"]["scientific_boundary_amendment"]
    if not amendment.get("frozen_before_repaired_pilot_dispatch"):
        raise ValueError("machine-readable amendment is not frozen")
    required_phrases = (
        "A schedule row becomes outcome-bearing",
        "retryable without limit",
        "zero initial observations, zero actions, and zero scientific outcomes",
        "content-independent",
        "launch smoke",
    )
    normalized_preregistration = " ".join(preregistration_text.lower().split())
    for phrase in required_phrases:
        if " ".join(phrase.lower().split()) not in normalized_preregistration:
            raise ValueError(f"preregistration omits required phrase: {phrase}")

    schedule = json.loads(args.schedule.read_text())
    schedule_sha256 = validate_self_hash(schedule, "schedule_sha256")
    if schedule_sha256 != EXPECTED_SCHEDULE_SHA256 or len(schedule["rows"]) != 64:
        raise ValueError("unexpected scientific pilot schedule")

    training = json.loads(args.training_summary.read_text())
    records = training["records"]
    if (
        len(records) != 1
        or records[0]["arm"] != "ACT"
        or records[0]["seed"] != 1101
        or records[0]["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256
        or file_hash(Path(records[0]["checkpoint"])) != EXPECTED_CHECKPOINT_SHA256
    ):
        raise ValueError("retained pilot checkpoint identity mismatch")

    smoke = schedule["rows"][0]
    document: dict[str, Any] = {
        "schema_version": "pact_pilot_retry_dispatch_v2",
        "scientific_schedule": {
            "path": str(args.schedule),
            "schedule_sha256": schedule_sha256,
            "rows": 64,
            "workers": 8,
            "rows_changed": 0,
        },
        "protocol_amendment": {
            "preregistration_path": str(args.preregistration),
            "preregistration_sha256": file_hash(args.preregistration),
            "machine_preregistration_path": str(args.machine_preregistration),
            "machine_preregistration_sha256": file_hash(
                args.machine_preregistration
            ),
            "scientific_boundary": amendment["boundary"],
            "pre_observation_retry_limit": amendment["retry_limit"],
            "post_observation_terminal": amendment["terminal_after_boundary"],
            "thresholds_changed": False,
            "scene_changed": False,
            "checkpoint_changed": False,
        },
        "prior_failed_dispatch": old_dispatch_record(
            args.old_output_root, schedule
        ),
        "retry_justification": {
            "no_scientific_outcome_to_select_on": True,
            "fix_committed_before_any_policy_result": True,
            "path_resolution_content_independent": True,
            "outcome_based_replacement": False,
        },
        "launch_smoke": {
            "required_before_full_dispatch": True,
            "schedule_index": smoke["schedule_index"],
            "rollout_id": smoke["rollout_id"],
            "instance_episode_id": smoke["instance_episode_id"],
            "schedule_row_sha256": smoke["schedule_row_sha256"],
            "output_relpath": smoke["output_relpath"],
            "required_artifact": "launch_smoke.json",
            "required_result_status": "complete",
            "full_dispatch_must_reconcile_without_rerun": True,
        },
        "retained_pilot_checkpoint": {
            "arm": "ACT",
            "seed": 1101,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "retrained": False,
        },
        "future_confirmatory_requirement": {
            "launch_smoke_required": True,
            "planned_rollouts": 960,
        },
    }
    document["dispatch_contract_sha256"] = canonical_hash(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["dispatch_contract_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
