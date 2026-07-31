#!/usr/bin/env python3
"""Durable eight-worker supervisor for the frozen front-end screen."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_pact_confirmatory_schedule as base

PYTHON = Path("/root/act_retrain_venv/bin/python")
EVALUATOR = (
    ROOT / "submodules/act/eval_pact_frontend_screen_row.py"
)
WORKERS = 8
HEARTBEAT_SECONDS = 5.0
COHORT_LOSS_WINDOW_SECONDS = 5.0


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def process_identity(pid: int) -> dict[str, int] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    if len(fields) <= 21 or fields[2] == "Z":
        return None
    return {
        "pid": pid,
        "ppid": int(fields[3]),
        "process_group_id": int(fields[4]),
        "session_id": int(fields[5]),
        "start_time_ticks": int(fields[21]),
    }


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


@dataclass
class ActiveAttempt:
    row: dict[str, Any]
    process: subprocess.Popen
    log_stream: Any
    log_path: Path
    attempt_index: int
    started_utc: str
    recovery_event: Path | None

    def state(self) -> dict[str, Any]:
        identity = process_identity(self.process.pid)
        return {
            "schedule_index": self.row["schedule_index"],
            "rollout_id": self.row["rollout_id"],
            "schedule_row_sha256": self.row[
                "schedule_row_sha256"
            ],
            "output_relpath": self.row["output_relpath"],
            "pid": self.process.pid,
            "process_start_time_ticks": (
                identity["start_time_ticks"]
                if identity is not None
                else None
            ),
            "attempt_index": self.attempt_index,
            "started_utc": self.started_utc,
            "process_log": str(self.log_path),
            "recovery_event": (
                str(self.recovery_event)
                if self.recovery_event is not None
                else None
            ),
        }


class ScreenSupervisor:
    def __init__(
        self,
        *,
        schedule_path: Path,
        contract_path: Path,
        manifest_path: Path,
        output_root: Path,
        mode: str,
    ) -> None:
        self.schedule_path = schedule_path.resolve()
        self.contract_path = contract_path.resolve()
        self.manifest_path = manifest_path.resolve()
        self.output_root = output_root.resolve()
        self.mode = mode
        self.schedule = json.loads(self.schedule_path.read_text())
        payload = dict(self.schedule)
        observed = payload.pop("schedule_sha256", None)
        if canonical_hash(payload) != observed:
            raise RuntimeError("screen schedule self-hash mismatch")
        if (
            self.schedule.get("schema_version")
            != "pact_frontend_screen_schedule_v1"
            or self.schedule.get("workers") != WORKERS
            or self.schedule.get("rollouts") != 120
        ):
            raise RuntimeError("screen schedule design mismatch")
        self.contract = base.load_dispatch_contract(
            self.contract_path,
            self.schedule,
            manifest_path=self.manifest_path,
            output_root=self.output_root,
        )
        if (
            self.contract.get("schema_version")
            != "pact_frontend_screen_dispatch_v1"
        ):
            raise RuntimeError("wrong screen dispatch contract")
        for label, record in self.contract["frozen_inputs"][
            "runtime"
        ].items():
            if base.sha256_file(Path(record["path"])) != record[
                "sha256"
            ]:
                raise RuntimeError(
                    f"frozen screen runtime changed: {label}"
                )
        active_protected = protected_eval_processes()
        if active_protected:
            raise RuntimeError(
                "protected shared evaluation is active: "
                f"{active_protected}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.pid_path = self.output_root / "supervisor_pid.json"
        self.state_path = self.output_root / "supervisor_state.json"
        self.heartbeat_path = self.output_root / "heartbeat.json"
        self.completion_path = (
            self.output_root / "completion_ledger.json"
        )
        self.recovery_root = self.output_root / "recovery_events"
        self.recovery_root.mkdir(exist_ok=True)
        self.active: dict[str, ActiveAttempt] = {}
        self.pending: collections.deque[dict[str, Any]] = (
            collections.deque()
        )
        self.completions = self._load_completions()
        self.completed_ids = {
            item["rollout_id"] for item in self.completions
        }
        self.authorized_recovery: dict[str, Path] = {}
        self.abort_reason: str | None = None
        self.started_utc = utc_now()
        self.full_dispatch_started_utc = (
            self.started_utc if mode == "full" else None
        )
        self.last_heartbeat = 0.0
        self._install_signals()

    def _install_signals(self) -> None:
        def handler(signum, _frame):
            self.abort_reason = f"supervisor_received_signal_{signum}"

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def _row_dir(self, row: dict[str, Any]) -> Path:
        return self.output_root / row["output_relpath"]

    def _load_completions(self) -> list[dict[str, Any]]:
        if not self.completion_path.exists():
            return []
        document = json.loads(self.completion_path.read_text())
        if document.get("schedule_sha256") != self.schedule[
            "schedule_sha256"
        ]:
            raise RuntimeError("completion ledger schedule mismatch")
        return list(document["completions"])

    def _write_completions(self) -> None:
        document = {
            "schema_version": (
                "pact_frontend_screen_completion_ledger_v1"
            ),
            "schedule_sha256": self.schedule["schedule_sha256"],
            "completions": self.completions,
        }
        document["completion_ledger_sha256"] = canonical_hash(
            document
        )
        base.write_json_atomic(self.completion_path, document)

    def _valid_result(self, row: dict[str, Any]) -> bool:
        path = self._row_dir(row) / "result.json"
        if not path.exists():
            return False
        base._validate_scientific_result(path, row)
        return True

    def _attempts(
        self, row: dict[str, Any]
    ) -> tuple[Path, list[dict[str, Any]]]:
        path = self._row_dir(row) / "attempt_ledger.json"
        if not path.exists():
            return path, []
        document = json.loads(path.read_text())
        if (
            document.get("rollout_id") != row["rollout_id"]
            or document.get("schedule_row_sha256")
            != row["schedule_row_sha256"]
        ):
            raise RuntimeError("attempt ledger identity mismatch")
        return path, list(document["attempts"])

    def _write_attempt(
        self, row: dict[str, Any], attempt: dict[str, Any]
    ) -> None:
        path, attempts = self._attempts(row)
        if int(attempt["attempt_index"]) != len(attempts):
            raise RuntimeError("attempt index is not append-only")
        attempts.append(attempt)
        base.write_json_atomic(
            path,
            {
                "schema_version": (
                    "pact_frontend_screen_attempt_ledger_v1"
                ),
                "rollout_id": row["rollout_id"],
                "schedule_row_sha256": row[
                    "schedule_row_sha256"
                ],
                "attempts": attempts,
            },
        )

    def _command(
        self,
        row: dict[str, Any],
        *,
        attempt_index: int,
        recovery_event: Path | None,
    ) -> list[str]:
        command = base.command_for(
            row,
            manifest_path=self.manifest_path,
            output_dir=self._row_dir(row),
            save_video=True,
        )
        command[0] = str(PYTHON)
        command[1] = str(EVALUATOR)
        command.extend(["--attempt-index", str(attempt_index)])
        if recovery_event is not None:
            command.extend(
                [
                    "--inflight-recovery-event",
                    str(recovery_event),
                ]
            )
        return command

    def _launch(self, row: dict[str, Any]) -> None:
        row_dir = self._row_dir(row)
        row_dir.mkdir(parents=True, exist_ok=True)
        _path, attempts = self._attempts(row)
        attempt_index = len(attempts)
        recovery = self.authorized_recovery.pop(
            row["rollout_id"], None
        )
        command = self._command(
            row,
            attempt_index=attempt_index,
            recovery_event=recovery,
        )
        log_path = (
            row_dir / f"process_attempt_{attempt_index:03d}.log"
        )
        log_stream = log_path.open("ab")
        process = subprocess.Popen(
            command,
            cwd=ROOT / "submodules/act",
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.active[row["rollout_id"]] = ActiveAttempt(
            row=row,
            process=process,
            log_stream=log_stream,
            log_path=log_path,
            attempt_index=attempt_index,
            started_utc=utc_now(),
            recovery_event=recovery,
        )
        self._write_state()

    def _finish_attempt(
        self,
        active: ActiveAttempt,
        *,
        status: str,
        recovery_event_sha256: str | None = None,
    ) -> None:
        active.log_stream.flush()
        active.log_stream.close()
        row_dir = self._row_dir(active.row)
        self._write_attempt(
            active.row,
            {
                "attempt_index": active.attempt_index,
                "status": status,
                "returncode": active.process.poll(),
                "process_log": str(active.log_path),
                "process_log_sha256": base.sha256_file(
                    active.log_path
                ),
                "initial_observation_accepted": (
                    row_dir / "initial_observation_accepted.json"
                ).exists(),
                "scientific_result_written": (
                    row_dir / "result.json"
                ).exists(),
                "recovery_event_sha256": recovery_event_sha256,
                "finished_utc": utc_now(),
            },
        )

    def _reconcile_complete(
        self, active: ActiveAttempt
    ) -> None:
        row = active.row
        if not self._valid_result(row):
            raise RuntimeError("complete reconciliation lacks result")
        self._finish_attempt(active, status="complete")
        row_dir = self._row_dir(row)
        _attempt_path, attempts = self._attempts(row)
        driver = {
            "status": "complete",
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row[
                "schedule_row_sha256"
            ],
            "attempt_count": len(attempts),
            "final_attempt_index": active.attempt_index,
            "returncode": active.process.poll(),
        }
        base.write_json_atomic(
            row_dir / "driver_result.json", driver
        )
        completion = {
            "schedule_index": row["schedule_index"],
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row[
                "schedule_row_sha256"
            ],
            "arm": row["arm"],
            "completed_utc": utc_now(),
            "result_sha256": base.sha256_file(
                row_dir / "result.json"
            ),
            "driver_result_sha256": base.sha256_file(
                row_dir / "driver_result.json"
            ),
        }
        self.completions.append(completion)
        self.completed_ids.add(row["rollout_id"])
        self._write_completions()

    def _freeze_group_recovery(
        self, cohort: list[ActiveAttempt]
    ) -> Path:
        rows = []
        for active in cohort:
            row_dir = self._row_dir(active.row)
            rows.append(
                {
                    "schedule_index": active.row[
                        "schedule_index"
                    ],
                    "rollout_id": active.row["rollout_id"],
                    "schedule_row_sha256": active.row[
                        "schedule_row_sha256"
                    ],
                    "attempt_index": active.attempt_index,
                    "pid": active.process.pid,
                    "result_present": (
                        row_dir / "result.json"
                    ).exists(),
                    "initial_observation_accepted": (
                        row_dir
                        / "initial_observation_accepted.json"
                    ).exists(),
                }
            )
        event: dict[str, Any] = {
            "schema_version": (
                "pact_frontend_screen_group_recovery_v1"
            ),
            "schedule_sha256": self.schedule["schedule_sha256"],
            "qualifying_indiscriminate_termination": True,
            "all_inflight_rows_rerun": True,
            "result_absent_for_all": not any(
                row["result_present"] for row in rows
            ),
            "active_cohort_size": len(rows),
            "rows": rows,
            "frozen_utc": utc_now(),
        }
        event["recovery_event_sha256"] = canonical_hash(event)
        path = (
            self.recovery_root
            / f"group_{len(list(self.recovery_root.glob('group_*.json'))):03d}_"
            f"{event['recovery_event_sha256'][:12]}.json"
        )
        base.write_json_atomic(path, event)
        return path

    def _handle_exits(self) -> None:
        exited = [
            active
            for active in self.active.values()
            if active.process.poll() is not None
        ]
        if not exited:
            return
        for active in list(exited):
            if self._valid_result(active.row):
                self._reconcile_complete(active)
                self.active.pop(active.row["rollout_id"])
                exited.remove(active)
        if not exited:
            return
        has_post_observation_exit = any(
            (
                self._row_dir(active.row)
                / "initial_observation_accepted.json"
            ).exists()
            for active in exited
        )
        if has_post_observation_exit:
            cohort = list(self.active.values())
            time.sleep(COHORT_LOSS_WINDOW_SECONDS)
            all_dead = all(
                active.process.poll() is not None for active in cohort
            )
            no_results = all(
                not self._valid_result(active.row) for active in cohort
            )
            if all_dead and no_results:
                event_path = self._freeze_group_recovery(cohort)
                event = json.loads(event_path.read_text())
                for active in cohort:
                    self._finish_attempt(
                        active,
                        status=(
                            "group_termination_post_observation"
                            if (
                                self._row_dir(active.row)
                                / "initial_observation_accepted.json"
                            ).exists()
                            else "group_termination_pre_observation"
                        ),
                        recovery_event_sha256=event[
                            "recovery_event_sha256"
                        ],
                    )
                    self.active.pop(active.row["rollout_id"])
                    self.authorized_recovery[
                        active.row["rollout_id"]
                    ] = event_path
                    self.pending.append(active.row)
                return
        pre_observation = [
            active
            for active in exited
            if not (
                self._row_dir(active.row)
                / "initial_observation_accepted.json"
            ).exists()
        ]
        for active in pre_observation:
            self._finish_attempt(
                active,
                status="pre_observation_infrastructure_failure",
            )
            self.active.pop(active.row["rollout_id"])
            self.pending.appendleft(active.row)
            exited.remove(active)
        if not exited:
            return
        bad = exited[0]
        self.abort_reason = (
            "isolated_post_observation_failure:"
            f"{bad.row['rollout_id']}"
        )

    def _write_state(self, *, status: str = "running") -> None:
        document = {
            "schema_version": (
                "pact_frontend_screen_supervisor_state_v1"
            ),
            "schedule_sha256": self.schedule["schedule_sha256"],
            "dispatch_contract_sha256": self.contract[
                "dispatch_contract_sha256"
            ],
            "mode": self.mode,
            "status": status,
            "supervisor_pid": os.getpid(),
            "started_utc": self.started_utc,
            "full_dispatch_started_utc": (
                self.full_dispatch_started_utc
            ),
            "active_cohort": [
                active.state()
                for active in self.active.values()
            ],
            "pending_count": len(self.pending),
            "complete_count": len(self.completed_ids),
            "abort_reason": self.abort_reason,
            "updated_utc": utc_now(),
        }
        document["state_sha256"] = canonical_hash(document)
        base.write_json_atomic(self.state_path, document)

    def _heartbeat(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_heartbeat < HEARTBEAT_SECONDS:
            return
        document = {
            "schema_version": (
                "pact_frontend_screen_heartbeat_v1"
            ),
            "schedule_sha256": self.schedule["schedule_sha256"],
            "mode": self.mode,
            "supervisor_pid": os.getpid(),
            "active_count": len(self.active),
            "active": [
                active.state()
                for active in self.active.values()
            ],
            "pending_count": len(self.pending),
            "complete_count": len(self.completed_ids),
            "heartbeat_utc": utc_now(),
        }
        document["heartbeat_sha256"] = canonical_hash(document)
        base.write_json_atomic(self.heartbeat_path, document)
        self.last_heartbeat = now

    def _prepare(self) -> None:
        smoke_row = self.schedule["rows"][0]
        if self.mode == "smoke":
            rows = [smoke_row]
        else:
            base.validate_launch_smoke(
                schedule=self.schedule,
                contract=self.contract,
                output_root=self.output_root,
            )
            proof = self.output_root / self.contract[
                "detachment_proof"
            ]["required_artifact"]
            if not proof.exists():
                raise RuntimeError(
                    "full dispatch requires detachment proof"
                )
            rows = list(self.schedule["rows"])
        for row in rows:
            if row["rollout_id"] in self.completed_ids:
                continue
            if self._valid_result(row):
                row_dir = self._row_dir(row)
                if not (row_dir / "driver_result.json").exists():
                    base.write_json_atomic(
                        row_dir / "driver_result.json",
                        {
                            "status": "complete",
                            "rollout_id": row["rollout_id"],
                            "schedule_row_sha256": row[
                                "schedule_row_sha256"
                            ],
                            "resume_action": (
                                "reconciled_existing_result"
                            ),
                        },
                    )
                self.completions.append(
                    {
                        "schedule_index": row["schedule_index"],
                        "rollout_id": row["rollout_id"],
                        "schedule_row_sha256": row[
                            "schedule_row_sha256"
                        ],
                        "arm": row["arm"],
                        "completed_utc": utc_now(),
                        "result_sha256": base.sha256_file(
                            row_dir / "result.json"
                        ),
                        "driver_result_sha256": base.sha256_file(
                            row_dir / "driver_result.json"
                        ),
                    }
                )
                self.completed_ids.add(row["rollout_id"])
                continue
            boundary = (
                self._row_dir(row)
                / "initial_observation_accepted.json"
            )
            if boundary.exists():
                raise RuntimeError(
                    "result-free boundary exists without frozen "
                    f"group recovery: {row['rollout_id']}"
                )
            self.pending.append(row)
        self._write_completions()

    def _write_smoke(self) -> None:
        row = self.schedule["rows"][0]
        row_dir = self._row_dir(row)
        artifact = {
            "schema_version": (
                "pact_frontend_screen_launch_smoke_v1"
            ),
            "dispatch_contract_sha256": self.contract[
                "dispatch_contract_sha256"
            ],
            "scientific_schedule_sha256": self.schedule[
                "schedule_sha256"
            ],
            "passed": True,
            "smoke_invocations": 1,
            "schedule_index": row["schedule_index"],
            "rollout_id": row["rollout_id"],
            "instance_episode_id": row[
                "instance_episode_id"
            ],
            "schedule_row_sha256": row[
                "schedule_row_sha256"
            ],
            "driver_status": "complete",
            "scientific_result_sha256": base.sha256_file(
                row_dir / "result.json"
            ),
            "driver_result_sha256": base.sha256_file(
                row_dir / "driver_result.json"
            ),
        }
        artifact["launch_smoke_sha256"] = canonical_hash(artifact)
        base.write_json_atomic(
            self.output_root
            / self.contract["launch_smoke"]["required_artifact"],
            artifact,
        )

    def run(self) -> int:
        identity = process_identity(os.getpid())
        pid_record = {
            "schema_version": (
                "pact_frontend_screen_supervisor_pid_v1"
            ),
            "pid": os.getpid(),
            "identity": identity,
            "mode": self.mode,
            "schedule_sha256": self.schedule["schedule_sha256"],
            "started_utc": self.started_utc,
        }
        pid_record["pid_record_sha256"] = canonical_hash(pid_record)
        base.write_json_atomic(self.pid_path, pid_record)
        self._prepare()
        self._write_state()
        self._heartbeat(force=True)
        target = 1 if self.mode == "smoke" else 120
        while len(self.completed_ids) < target and not self.abort_reason:
            while self.pending and len(self.active) < (
                1 if self.mode == "smoke" else WORKERS
            ):
                self._launch(self.pending.popleft())
            self._handle_exits()
            self._write_state()
            self._heartbeat()
            time.sleep(0.25)
        if self.abort_reason:
            for active in self.active.values():
                if active.process.poll() is None:
                    active.process.terminate()
            self._write_state(status="aborted")
            self._heartbeat(force=True)
            return 1
        if self.mode == "smoke":
            self._write_smoke()
        summary = {
            "schema_version": (
                "pact_frontend_screen_execution_v1"
            ),
            "schedule_sha256": self.schedule["schedule_sha256"],
            "dispatch_contract_sha256": self.contract[
                "dispatch_contract_sha256"
            ],
            "mode": self.mode,
            "workers": 1 if self.mode == "smoke" else WORKERS,
            "expected": target,
            "complete_count": len(self.completed_ids),
            "scientific_schedule_reconciled": (
                len(self.completed_ids) == target
            ),
            "finished_utc": utc_now(),
        }
        base.write_json_atomic(
            self.output_root
            / f"{self.mode}_execution_summary.json",
            summary,
        )
        self._write_state(status="complete")
        self._heartbeat(force=True)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument(
        "--dispatch-contract", required=True, type=Path
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--mode", required=True, choices=("smoke", "full")
    )
    args = parser.parse_args()
    supervisor = ScreenSupervisor(
        schedule_path=args.schedule,
        contract_path=args.dispatch_contract,
        manifest_path=args.manifest,
        output_root=args.output_root,
        mode=args.mode,
    )
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
