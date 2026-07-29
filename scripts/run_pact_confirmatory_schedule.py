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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    # Evaluators run from the ACT submodule, not from the schedule runner's
    # working directory. Resolve caller-supplied paths before that cwd change.
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
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
    if result.get("initial_observation_accepted") is not True:
        raise RuntimeError(f"{path}: scientific boundary was not recorded")
    return result


def _validate_boundary(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    marker = json.loads(path.read_text())
    expected = {
        "initial_observation_accepted": True,
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "episode_id": row["instance_episode_id"],
        "checkpoint_sha256": row["checkpoint_sha256"],
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise RuntimeError(
                f"{path}: boundary {key}={marker.get(key)!r} != {value!r}"
            )
    return marker


def load_dispatch_contract(
    path: Path,
    schedule: dict[str, Any],
    *,
    manifest_path: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    contract = json.loads(path.read_text())
    payload = dict(contract)
    observed = payload.pop("dispatch_contract_sha256")
    if canonical_hash(payload) != observed:
        raise RuntimeError("dispatch contract self-hash mismatch")
    scientific = contract["scientific_schedule"]
    if (
        scientific["schedule_sha256"] != schedule["schedule_sha256"]
        or scientific["rows"] != len(schedule["rows"])
        or scientific["workers"] != schedule["workers"]
        or scientific["rows_changed"] != 0
    ):
        raise RuntimeError("dispatch contract changes the scientific schedule")
    if manifest_path is not None:
        if (
            Path(scientific["manifest_path"]).resolve() != manifest_path.resolve()
            or scientific["manifest_sha256"] != sha256_file(manifest_path)
        ):
            raise RuntimeError("dispatch manifest differs from frozen contract")
    if output_root is not None:
        if Path(contract["execution"]["output_root"]).resolve() != output_root.resolve():
            raise RuntimeError("dispatch output root differs from frozen contract")
    smoke = contract["launch_smoke"]
    matching = [
        row
        for row in schedule["rows"]
        if row["schedule_index"] == smoke["schedule_index"]
    ]
    if len(matching) != 1:
        raise RuntimeError("launch-smoke row does not resolve exactly once")
    row = matching[0]
    for key in (
        "rollout_id",
        "instance_episode_id",
        "schedule_row_sha256",
        "output_relpath",
    ):
        if smoke[key] != row[key]:
            raise RuntimeError(f"launch-smoke {key} does not match schedule")
    if not smoke.get("required_before_full_dispatch"):
        raise RuntimeError("dispatch contract does not require launch smoke")
    return contract


def run_one(
    row: dict[str, Any],
    *,
    manifest_path: str,
    output_root: str,
    save_video: bool,
    single_attempt: bool = False,
) -> dict[str, Any]:
    row_dir = Path(output_root) / row["output_relpath"]
    row_dir.mkdir(parents=True, exist_ok=True)
    driver_path = row_dir / "driver_result.json"
    scientific_path = row_dir / "result.json"
    boundary_path = row_dir / "initial_observation_accepted.json"
    attempt_ledger_path = row_dir / "attempt_ledger.json"
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
        _validate_boundary(boundary_path, row)
        driver = {
            "status": "complete",
            "resume_action": "reconciled_existing_scientific_result_without_rerun",
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "returncode": 0,
            "attempt_count": 0,
            "pre_observation_infrastructure_failures": 0,
        }
        write_json_atomic(driver_path, driver)
        return driver
    if boundary_path.exists():
        _validate_boundary(boundary_path, row)
        driver = {
            "status": "post_boundary_failure",
            "resume_action": "reconciled_boundary_marker_without_rerun",
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "returncode": None,
            "attempt_count": 0,
            "pre_observation_infrastructure_failures": 0,
            "error": "initial observation accepted but no scientific result exists",
        }
        write_json_atomic(driver_path, driver)
        return driver

    if attempt_ledger_path.exists():
        ledger = json.loads(attempt_ledger_path.read_text())
        if (
            ledger.get("rollout_id") != row["rollout_id"]
            or ledger.get("schedule_row_sha256") != row["schedule_row_sha256"]
        ):
            raise RuntimeError(f"{attempt_ledger_path}: attempt identity mismatch")
        attempts = list(ledger["attempts"])
    else:
        attempts = []

    while True:
        attempt_index = len(attempts)
        command = command_for(
            row,
            manifest_path=Path(manifest_path),
            output_dir=row_dir,
            save_video=save_video,
        )
        log_path = row_dir / f"process_attempt_{attempt_index:03d}.log"
        with log_path.open("wb") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT / "submodules" / "act",
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        status: str
        error: str | None
        if scientific_path.exists():
            _validate_scientific_result(scientific_path, row)
            _validate_boundary(boundary_path, row)
            status = "complete"
            error = None
        elif boundary_path.exists():
            _validate_boundary(boundary_path, row)
            status = "post_boundary_failure"
            error = (
                f"evaluator exited {completed.returncode} after accepting "
                "the initial observation and wrote no scientific result"
            )
        else:
            status = "pre_observation_infrastructure_failure"
            error = (
                f"evaluator exited {completed.returncode} before accepting "
                "the initial observation"
            )

        attempt = {
            "attempt_index": attempt_index,
            "status": status,
            "returncode": completed.returncode,
            "command": command,
            "process_log": str(log_path),
            "process_log_sha256": sha256_file(log_path),
            "initial_observation_accepted": boundary_path.exists(),
            "scientific_result_written": scientific_path.exists(),
            "error": error,
        }
        attempts.append(attempt)
        write_json_atomic(
            attempt_ledger_path,
            {
                "schema_version": "pact_schedule_attempt_ledger_v1",
                "rollout_id": row["rollout_id"],
                "schedule_row_sha256": row["schedule_row_sha256"],
                "attempts": attempts,
            },
        )

        if status == "pre_observation_infrastructure_failure":
            if single_attempt:
                return {
                    "status": status,
                    "rollout_id": row["rollout_id"],
                    "schedule_row_sha256": row["schedule_row_sha256"],
                    "attempt_count": len(attempts),
                    "pre_observation_infrastructure_failures": len(attempts),
                    "error": error,
                }
            print(
                f"{row['schedule_index']:03d} {row['arm']:9s} "
                f"{row['instance_episode_id'][:10]} retryable_infrastructure "
                f"attempt={attempt_index}",
                flush=True,
            )
            continue

        driver = {
            "status": status,
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "returncode": completed.returncode,
            "attempt_count": len(attempts),
            "pre_observation_infrastructure_failures": sum(
                item["status"] == "pre_observation_infrastructure_failure"
                for item in attempts
            ),
            "final_attempt_index": attempt_index,
            "final_process_log": str(log_path),
            "error": error,
        }
        write_json_atomic(driver_path, driver)
        return driver


def launch_smoke(
    *,
    schedule: dict[str, Any],
    contract: dict[str, Any],
    manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    smoke_spec = contract["launch_smoke"]
    row = next(
        row
        for row in schedule["rows"]
        if row["schedule_index"] == smoke_spec["schedule_index"]
    )
    result = run_one(
        row,
        manifest_path=str(manifest_path),
        output_root=str(output_root),
        save_video=True,
        single_attempt=True,
    )
    row_dir = output_root / row["output_relpath"]
    passed = result["status"] == "complete"
    smoke_path = output_root / smoke_spec["required_artifact"]
    previous_attempts = 0
    if smoke_path.exists():
        previous = json.loads(smoke_path.read_text())
        if (
            previous.get("rollout_id") != row["rollout_id"]
            or previous.get("dispatch_contract_sha256")
            != contract["dispatch_contract_sha256"]
        ):
            raise RuntimeError("existing launch-smoke identity mismatch")
        previous_attempts = int(previous.get("smoke_invocations", 0))
    artifact = {
        "schema_version": "pact_schedule_launch_smoke_v1",
        "dispatch_contract_sha256": contract["dispatch_contract_sha256"],
        "scientific_schedule_sha256": schedule["schedule_sha256"],
        "passed": passed,
        "smoke_invocations": previous_attempts + 1,
        "schedule_index": row["schedule_index"],
        "rollout_id": row["rollout_id"],
        "instance_episode_id": row["instance_episode_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "driver_status": result["status"],
        "attempt_count": result.get("attempt_count"),
        "pre_observation_infrastructure_failures": result.get(
            "pre_observation_infrastructure_failures"
        ),
        "scientific_result_sha256": (
            sha256_file(row_dir / "result.json") if passed else None
        ),
        "driver_result_sha256": (
            sha256_file(row_dir / "driver_result.json") if passed else None
        ),
    }
    artifact["launch_smoke_sha256"] = canonical_hash(artifact)
    write_json_atomic(smoke_path, artifact)
    return artifact


def validate_launch_smoke(
    *,
    schedule: dict[str, Any],
    contract: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    smoke_path = output_root / contract["launch_smoke"]["required_artifact"]
    if not smoke_path.exists():
        raise RuntimeError("full dispatch refused: launch smoke is missing")
    artifact = json.loads(smoke_path.read_text())
    payload = dict(artifact)
    observed = payload.pop("launch_smoke_sha256")
    if canonical_hash(payload) != observed:
        raise RuntimeError("launch-smoke self-hash mismatch")
    expected = {
        "dispatch_contract_sha256": contract["dispatch_contract_sha256"],
        "scientific_schedule_sha256": schedule["schedule_sha256"],
        "passed": True,
        "schedule_index": contract["launch_smoke"]["schedule_index"],
        "rollout_id": contract["launch_smoke"]["rollout_id"],
        "schedule_row_sha256": contract["launch_smoke"]["schedule_row_sha256"],
        "driver_status": "complete",
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise RuntimeError(f"launch smoke {key} mismatch")
    row = next(
        row
        for row in schedule["rows"]
        if row["schedule_index"] == artifact["schedule_index"]
    )
    row_dir = output_root / row["output_relpath"]
    _validate_scientific_result(row_dir / "result.json", row)
    _validate_boundary(row_dir / "initial_observation_accepted.json", row)
    if sha256_file(row_dir / "result.json") != artifact["scientific_result_sha256"]:
        raise RuntimeError("launch-smoke scientific result changed")
    if sha256_file(row_dir / "driver_result.json") != artifact["driver_result_sha256"]:
        raise RuntimeError("launch-smoke driver result changed")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch-contract", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="run only the predeclared launch-smoke row and require result.json",
    )
    parser.add_argument(
        "--pilot-gate",
        action="store_true",
        help=(
            "accept a reconciled terminal pilot ledger even if a row fails "
            "after its initial observation; pre-observation failures retry"
        ),
    )
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    payload = dict(schedule)
    observed_hash = payload.pop("schedule_sha256")
    if canonical_hash(payload) != observed_hash:
        raise SystemExit("schedule self-hash mismatch")
    if int(schedule["workers"]) != 8:
        raise SystemExit("frozen schedule worker count is not 8")
    contract = load_dispatch_contract(
        args.dispatch_contract,
        schedule,
        manifest_path=args.manifest,
        output_root=args.output_root,
    )
    active = protected_eval_processes()
    if active:
        raise SystemExit(
            "protected confirmatory evaluation is still active; refusing to launch "
            f"(PIDs {active})"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.smoke_only:
        smoke = launch_smoke(
            schedule=schedule,
            contract=contract,
            manifest_path=args.manifest,
            output_root=args.output_root,
        )
        print(
            f"launch_smoke passed={smoke['passed']} "
            f"rollout_id={smoke['rollout_id']}",
            flush=True,
        )
        return 0 if smoke["passed"] else 1
    smoke = validate_launch_smoke(
        schedule=schedule,
        contract=contract,
        output_root=args.output_root,
    )
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
            except BaseException as exc:  # noqa: BLE001 - worker death reconciliation
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
    terminal_driver_statuses = {"complete", "post_boundary_failure"}
    nonterminal = [
        row["rollout_id"]
        for row in schedule["rows"]
        if by_id.get(row["rollout_id"], {}).get("status")
        not in terminal_driver_statuses
    ]
    summary = {
        "schema_version": "pact_schedule_execution_v2",
        "schedule_sha256": schedule["schedule_sha256"],
        "dispatch_contract_sha256": contract["dispatch_contract_sha256"],
        "launch_smoke_sha256": smoke["launch_smoke_sha256"],
        "launch_smoke_rollout_id": smoke["rollout_id"],
        "workers": 8,
        "pilot_gate_mode": args.pilot_gate,
        "expected": len(schedule["rows"]),
        "complete_count": sum(
            result.get("status") == "complete" for result in results
        ),
        "post_boundary_failure_count": sum(
            result.get("status") == "post_boundary_failure"
            for result in results
        ),
        "pre_observation_infrastructure_failures": sum(
            int(result.get("pre_observation_infrastructure_failures", 0))
            for result in results
        ),
        "missing": missing,
        "noncomplete": noncomplete,
        "nonterminal": nonterminal,
        "terminal_ledger_reconciled": not missing and not nonterminal,
        "scientific_schedule_reconciled": not missing and not noncomplete,
    }
    write_json_atomic(args.output_root / "execution_summary.json", summary)
    accepted = (
        summary["terminal_ledger_reconciled"]
        if args.pilot_gate
        else summary["scientific_schedule_reconciled"]
    )
    if not accepted:
        print("PACT SCHEDULE DID NOT RECONCILE", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
