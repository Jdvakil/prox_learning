#!/usr/bin/env python3
"""Run attempt-2 expert rows in fresh subprocesses with a batch watchdog."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MOLMO = ROOT / "submodules" / "molmospaces"
for path in (SCRIPTS, MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_geometry_generalization_v2_contract import (  # noqa: E402
    CANDIDATES,
    canonical_json,
    load_manifest,
    sha256_payload,
)
from run_pact_geometry_expert_screen import (  # noqa: E402
    TERMINAL_STATUSES,
    active_protected_processes,
)


PYTHON = Path("/root/act_retrain_venv/bin/python")
ROW_SCRIPT = SCRIPTS / "run_pact_geometry_v2_expert_row.py"
POLL_SECONDS = 2.0


@dataclass
class ActiveRow:
    row: dict[str, Any]
    process: subprocess.Popen
    log_stream: Any
    log_path: Path
    launched_at: float
    attempt: int


def write_json_atomic(path: Path, value: Any) -> None:
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


def self_hashed(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result[key] = sha256_payload(result)
    return result


def phase_rows(
    manifest: dict[str, Any], phase: str, selection: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if phase == "phase0a":
        if selection is not None:
            raise ValueError("phase0a does not accept a selection artifact")
        return list(manifest["phase0a_rows"])
    if selection is None:
        raise ValueError("phase0b requires the frozen phase0b selection artifact")
    selected = selection.get("selected_candidate_ids")
    if not isinstance(selected, list) or len(selected) != 2:
        raise ValueError("phase0b selection must contain exactly two candidate IDs")
    if len(set(selected)) != 2 or not set(selected).issubset(CANDIDATES):
        raise ValueError("phase0b selection has unknown or duplicate candidate IDs")
    return [
        row
        for row in manifest["phase0b_candidate_rows"]
        if row["condition_id"] in selected
    ]


def result_path(output_root: Path, phase: str, row: dict[str, Any]) -> Path:
    return (
        output_root
        / phase
        / "expert_screen_rows"
        / row["condition_id"]
        / f"{row['role_index']:02d}_{row['episode_id'][:16]}"
        / "result.json"
    )


def boundary_path(output_root: Path, phase: str, row: dict[str, Any]) -> Path:
    return result_path(output_root, phase, row).with_name(
        "initial_observation_accepted.json"
    )


def validate_result(path: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if (
        value.get("status") not in TERMINAL_STATUSES
        or value.get("episode_id") != row["episode_id"]
        or value.get("row_sha256") != row["row_sha256"]
    ):
        raise RuntimeError(f"invalid expert result: {path}")
    return value


def _archive_preboundary_result(path: Path, attempt: int) -> None:
    if not path.exists():
        return
    archive = path.parent / "infrastructure_attempts" / f"attempt_{attempt:03d}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise RuntimeError(f"attempt archive already exists: {archive}")
    shutil.move(str(path), str(archive))


def no_active_boundary(
    active: dict[str, ActiveRow], output_root: Path, phase: str
) -> bool:
    return not any(boundary_path(output_root, phase, item.row).exists() for item in active.values())


def terminate_batch(active: dict[str, ActiveRow]) -> None:
    for item in active.values():
        if item.process.poll() is None:
            try:
                os.killpg(item.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and any(
        item.process.poll() is None for item in active.values()
    ):
        time.sleep(0.25)
    for item in active.values():
        if item.process.poll() is None:
            try:
                os.killpg(item.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        item.process.wait()
        item.log_stream.close()


def launch_row(
    *,
    row: dict[str, Any],
    phase: str,
    manifest_path: Path,
    output_root: Path,
    attempt: int,
) -> ActiveRow:
    log_path = (
        output_root
        / phase
        / "logs"
        / f"{row['condition_id']}_{int(row['instance_index']):02d}_attempt{attempt:03d}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("ab", buffering=0)
    command = [
        str(PYTHON),
        str(ROW_SCRIPT),
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(output_root),
        "--phase",
        phase,
        "--condition-id",
        str(row["condition_id"]),
        "--instance-index",
        str(row["instance_index"]),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONUNBUFFERED": "1",
            "MLSPACES_ASSETS_DIR": str(ROOT / "assets"),
            "PYTHONPATH": str(MOLMO),
            "PACT_CONTACT_AUDIT_SUMMARY_ONLY": "1",
        }
    )
    environment.pop("DISPLAY", None)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return ActiveRow(
        row=row,
        process=process,
        log_stream=log_stream,
        log_path=log_path,
        launched_at=time.monotonic(),
        attempt=attempt,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("phase0a", "phase0b"))
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--watchdog-seconds", type=float)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise SystemExit("workers must be in [1, 12]")
    if not PYTHON.exists():
        raise SystemExit(f"python not found: {PYTHON}")
    protected = active_protected_processes()
    if protected:
        raise SystemExit(f"protected confirmatory evaluation is active: {protected}")

    manifest = load_manifest(args.manifest)
    selection = json.loads(args.selection.read_text()) if args.selection else None
    rows = phase_rows(manifest, args.phase, selection)
    expected_watchdog = float(manifest["expert_watchdog"]["no_completion_seconds"])
    watchdog_seconds = (
        expected_watchdog if args.watchdog_seconds is None else args.watchdog_seconds
    )
    # A non-contract watchdog value exists only for deterministic unit tests.
    if args.watchdog_seconds is not None and os.environ.get("PACT_V2_TEST_WATCHDOG") != "1":
        raise SystemExit("watchdog duration is frozen by the manifest")

    phase_root = args.output_root / args.phase
    phase_root.mkdir(parents=True, exist_ok=True)
    attempts = {row["episode_id"]: 0 for row in rows}
    completed: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for row in rows:
        value = validate_result(result_path(args.output_root, args.phase, row), row)
        if value is None:
            pending.append(row)
        elif value["status"] == "infrastructure_failure":
            if boundary_path(args.output_root, args.phase, row).exists():
                completed[row["episode_id"]] = value
            else:
                _archive_preboundary_result(
                    result_path(args.output_root, args.phase, row), 0
                )
                pending.append(row)
        else:
            completed[row["episode_id"]] = value

    events: list[dict[str, Any]] = []
    active: dict[str, ActiveRow] = {}
    last_completion = time.monotonic()
    started_wall = time.time()
    unreconciled_reason: str | None = None

    def record_event(kind: str, **fields: Any) -> None:
        event = {"event_index": len(events), "kind": kind, "time_unix": time.time(), **fields}
        events.append(event)
        write_json_atomic(
            phase_root / "watchdog_events.json",
            self_hashed(
                {
                    "schema_version": "pact_geometry_v2_watchdog_events",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "phase": args.phase,
                    "events": events,
                },
                "watchdog_events_sha256",
            ),
        )

    try:
        while pending or active:
            while pending and len(active) < args.workers:
                row = pending.pop(0)
                episode_id = row["episode_id"]
                attempts[episode_id] += 1
                item = launch_row(
                    row=row,
                    phase=args.phase,
                    manifest_path=args.manifest.resolve(),
                    output_root=args.output_root.resolve(),
                    attempt=attempts[episode_id],
                )
                active[episode_id] = item
                record_event(
                    "launch",
                    episode_id=episode_id,
                    condition_id=row["condition_id"],
                    instance_index=row["instance_index"],
                    attempt=item.attempt,
                    pid=item.process.pid,
                )

            time.sleep(POLL_SECONDS)
            exited = [episode_id for episode_id, item in active.items() if item.process.poll() is not None]
            restart_trigger: tuple[str, str] | None = None
            for episode_id in exited:
                item = active[episode_id]
                item.log_stream.close()
                path = result_path(args.output_root, args.phase, item.row)
                result = validate_result(path, item.row)
                if result is not None and result["status"] != "infrastructure_failure":
                    completed[episode_id] = result
                    active.pop(episode_id)
                    last_completion = time.monotonic()
                    record_event(
                        "row_terminal",
                        episode_id=episode_id,
                        status=result["status"],
                        exit_code=item.process.returncode,
                        attempt=item.attempt,
                    )
                    print(
                        f"{len(completed):03d}/{len(rows):03d} {item.row['condition_id']} "
                        f"{item.row['instance_index']:02d} {result['status']} "
                        f"clean={result.get('clean_success')}",
                        flush=True,
                    )
                elif boundary_path(args.output_root, args.phase, item.row).exists():
                    if result is not None:
                        completed[episode_id] = result
                    active.pop(episode_id)
                    unreconciled_reason = (
                        f"post-boundary row failure for {episode_id}; row is not rerunnable"
                    )
                    record_event(
                        "post_boundary_failure",
                        episode_id=episode_id,
                        exit_code=item.process.returncode,
                        attempt=item.attempt,
                    )
                    break
                else:
                    restart_trigger = (episode_id, "pre_boundary_process_failure")
                    break

            if unreconciled_reason:
                break

            no_completion_for = time.monotonic() - last_completion
            if restart_trigger is None and active and no_completion_for >= watchdog_seconds:
                if no_active_boundary(active, args.output_root, args.phase):
                    restart_trigger = ("batch", "watchdog_no_completion")
                else:
                    record_event(
                        "watchdog_deferred_active_boundary",
                        no_completion_seconds=no_completion_for,
                        active_episode_ids=sorted(active),
                    )
                    last_completion = time.monotonic()

            if restart_trigger is not None:
                if not no_active_boundary(active, args.output_root, args.phase):
                    unreconciled_reason = (
                        "restart trigger occurred while an active scientific boundary existed"
                    )
                    record_event(
                        "batch_restart_blocked",
                        trigger_episode_id=restart_trigger[0],
                        reason=restart_trigger[1],
                        active_episode_ids=sorted(active),
                    )
                    break
                batch = list(active.values())
                terminate_batch(active)
                active.clear()
                for item in batch:
                    _archive_preboundary_result(
                        result_path(args.output_root, args.phase, item.row), item.attempt
                    )
                    pending.append(item.row)
                record_event(
                    "batch_restart",
                    trigger_episode_id=restart_trigger[0],
                    reason=restart_trigger[1],
                    restarted_episode_ids=sorted(item.row["episode_id"] for item in batch),
                )
                last_completion = time.monotonic()

            write_json_atomic(
                phase_root / "heartbeat.json",
                {
                    "schema_version": "pact_geometry_v2_expert_heartbeat",
                    "time_unix": time.time(),
                    "phase": args.phase,
                    "expected": len(rows),
                    "completed": len(completed),
                    "pending": len(pending),
                    "active": [
                        {
                            "episode_id": item.row["episode_id"],
                            "condition_id": item.row["condition_id"],
                            "instance_index": item.row["instance_index"],
                            "pid": item.process.pid,
                            "attempt": item.attempt,
                            "initial_observation_accepted": boundary_path(
                                args.output_root, args.phase, item.row
                            ).exists(),
                        }
                        for item in active.values()
                    ],
                    "unreconciled_reason": unreconciled_reason,
                },
            )
    finally:
        if active:
            terminate_batch(active)

    statuses = {key: value.get("status") for key, value in completed.items()}
    reconciled = (
        unreconciled_reason is None
        and len(completed) == len(rows)
        and all(status in {"complete", "sampling_failure"} for status in statuses.values())
    )
    dispatch = self_hashed(
        {
            "schema_version": "pact_geometry_v2_expert_dispatch",
            "manifest_sha256": manifest["manifest_sha256"],
            "phase": args.phase,
            "selection_sha256": selection.get("phase0b_selection_sha256") if selection else None,
            "workers": args.workers,
            "watchdog_seconds": watchdog_seconds,
            "fresh_subprocess_per_rollout": True,
            "batch_restart_only": True,
            "expected_rows": len(rows),
            "terminal_rows": len(completed),
            "status_counts": {
                status: list(statuses.values()).count(status)
                for status in sorted(set(statuses.values()))
            },
            "attempt_counts": dict(sorted(attempts.items())),
            "reconciled": reconciled,
            "unreconciled_reason": unreconciled_reason,
            "started_unix": started_wall,
            "finished_unix": time.time(),
            "elapsed_seconds": time.time() - started_wall,
        },
        "dispatch_sha256",
    )
    write_json_atomic(phase_root / "dispatch.json", dispatch)
    print(canonical_json(dispatch), flush=True)
    return 0 if reconciled else 1


if __name__ == "__main__":
    raise SystemExit(main())
