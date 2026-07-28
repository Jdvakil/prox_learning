#!/usr/bin/env python3
"""Execute the frozen PACT schedule with one fresh subprocess per rollout."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/act_retrain_venv/bin/python")
EVALUATOR = ROOT / "submodules" / "act" / "eval_pact_collision_row.py"


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def protected_eval_processes() -> list[int]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"eval_act_obstacle_on_policy.py" in command:
            matches.append(int(entry.name))
    return matches


def command_for(
    row: dict[str, Any],
    *,
    manifest_path: Path,
    output_dir: Path,
    save_video: bool,
) -> list[str]:
    checkpoint_path = Path(row["checkpoint_path"])
    command = [
        str(PYTHON),
        str(EVALUATOR),
        "--arm",
        row["arm"],
        "--episode-id",
        row["instance_episode_id"],
        "--manifest",
        str(manifest_path),
        "--checkpoint-dir",
        str(checkpoint_path.parent),
        "--checkpoint-sha256",
        row["checkpoint_sha256"],
        "--checkpoint-seed",
        str(row["checkpoint_seed"]),
        "--stats-sha256",
        row["dataset_stats_sha256"],
        "--schedule-row-sha256",
        row["schedule_row_sha256"],
        "--rollout-id",
        row["rollout_id"],
        "--output-dir",
        str(output_dir),
    ]
    if row["arm"] in ("PACT", "PACT_ZERO"):
        command.extend(
            [
                "--surface-encoder",
                row["surface_encoder_path"],
                "--surface-encoder-sha256",
                row["surface_encoder_sha256"],
            ]
        )
    if save_video:
        command.append("--save-video")
    return command


def _validate_scientific_result(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(path.read_text())
    expected = {
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "episode_id": row["instance_episode_id"],
        "arm": row["arm"],
        "checkpoint_sha256": row["checkpoint_sha256"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"{path}: {key}={result.get(key)!r} != {value!r}")
    if result.get("status") != "complete":
        raise RuntimeError(f"{path}: scientific result is not complete")
    return result


def run_one(
    row: dict[str, Any],
    *,
    manifest_path: str,
    output_root: str,
    save_video: bool,
) -> dict[str, Any]:
    row_dir = Path(output_root) / row["output_relpath"]
    row_dir.mkdir(parents=True, exist_ok=True)
    driver_path = row_dir / "driver_result.json"
    scientific_path = row_dir / "result.json"
    if driver_path.exists():
        driver = json.loads(driver_path.read_text())
        if (
            driver.get("rollout_id") != row["rollout_id"]
            or driver.get("schedule_row_sha256") != row["schedule_row_sha256"]
        ):
            raise RuntimeError(f"{driver_path}: terminal driver identity mismatch")
        return driver
    if scientific_path.exists():
        _validate_scientific_result(scientific_path, row)
        driver = {
            "status": "complete",
            "resume_action": "reconciled_existing_scientific_result_without_rerun",
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "returncode": 0,
        }
        write_json_atomic(driver_path, driver)
        return driver

    command = command_for(
        row,
        manifest_path=Path(manifest_path),
        output_dir=row_dir,
        save_video=save_video,
    )
    log_path = row_dir / "process.log"
    with log_path.open("wb") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT / "submodules" / "act",
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status = "complete"
    error = None
    if completed.returncode != 0:
        status = "invocation_failure"
        error = f"evaluator exited {completed.returncode}"
    elif not scientific_path.exists():
        status = "invocation_failure"
        error = "evaluator exited zero but wrote no result.json"
    else:
        try:
            _validate_scientific_result(scientific_path, row)
        except Exception as exc:  # terminal identity failure, never auto-rerun
            status = "invocation_failure"
            error = f"{type(exc).__name__}: {exc}"
    driver = {
        "status": status,
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "returncode": completed.returncode,
        "command": command,
        "process_log": str(log_path),
        "error": error,
    }
    write_json_atomic(driver_path, driver)
    return driver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    payload = dict(schedule)
    observed_hash = payload.pop("schedule_sha256")
    if canonical_hash(payload) != observed_hash:
        raise SystemExit("schedule self-hash mismatch")
    if int(schedule["workers"]) != 8:
        raise SystemExit("frozen schedule worker count is not 8")
    active = protected_eval_processes()
    if active:
        raise SystemExit(
            "protected confirmatory evaluation is still active; refusing to launch "
            f"(PIDs {active})"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                run_one,
                row,
                manifest_path=str(args.manifest),
                output_root=str(args.output_root),
                save_video=True,
            ): row
            for row in schedule["rows"]
        }
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except BaseException as exc:
                result = {
                    "status": "driver_crash",
                    "rollout_id": row["rollout_id"],
                    "schedule_row_sha256": row["schedule_row_sha256"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            print(
                f"{row['schedule_index']:03d} {row['arm']:9s} "
                f"{row['instance_episode_id'][:10]} {result['status']}",
                flush=True,
            )
    by_id = {result["rollout_id"]: result for result in results}
    missing = [
        row["rollout_id"] for row in schedule["rows"] if row["rollout_id"] not in by_id
    ]
    noncomplete = [
        row["rollout_id"]
        for row in schedule["rows"]
        if by_id.get(row["rollout_id"], {}).get("status") != "complete"
    ]
    summary = {
        "schema_version": "pact_confirmatory_execution_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "workers": 8,
        "expected": len(schedule["rows"]),
        "complete_count": sum(
            result.get("status") == "complete" for result in results
        ),
        "missing": missing,
        "noncomplete": noncomplete,
        "reconciled": not missing and not noncomplete,
    }
    write_json_atomic(args.output_root / "execution_summary.json", summary)
    if not summary["reconciled"]:
        print("PACT CONFIRMATORY SCHEDULE DID NOT RECONCILE", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
