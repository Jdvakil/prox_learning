#!/usr/bin/env python3
"""Durable eight-worker supervisor for the confirmatory PACT R2 schedule."""

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

WORKERS = 8
LOSS_WINDOW_SECONDS = 5.0
HEARTBEAT_SECONDS = 5.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def pid_alive(pid: int) -> bool:
    return process_identity(pid) is not None


def process_identity(pid: int) -> dict[str, int] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    if len(stat) <= 21 or stat[2] == "Z":
        return None
    return {
        "pid": pid,
        "ppid": int(stat[3]),
        "process_group_id": int(stat[4]),
        "session_id": int(stat[5]),
        "start_time_ticks": int(stat[21]),
    }


def recorded_process_alive(item: dict[str, Any]) -> bool:
    if item.get("pid") is None:
        return False
    identity = process_identity(int(item["pid"]))
    if identity is None:
        return False
    expected = item.get("process_start_time_ticks")
    return expected is None or identity["start_time_ticks"] == int(expected)


def write_json(path: Path, value: dict[str, Any]) -> None:
    base.write_json_atomic(path, value)


def load_attempts(path: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    document = json.loads(path.read_text())
    if (
        document.get("rollout_id") != row["rollout_id"]
        or document.get("schedule_row_sha256") != row["schedule_row_sha256"]
    ):
        raise RuntimeError(f"{path}: attempt ledger identity mismatch")
    return list(document["attempts"])


def write_attempts(
    path: Path,
    row: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> None:
    write_json(
        path,
        {
            "schema_version": "pact_r2_attempt_ledger_v1",
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "attempts": attempts,
        },
    )


@dataclass
class ActiveAttempt:
    row: dict[str, Any]
    process: subprocess.Popen
    log_stream: Any
    log_path: Path
    attempt_index: int
    started_utc: str
    recovery_event_path: Path | None

    def state_record(self) -> dict[str, Any]:
        identity = process_identity(self.process.pid)
        return {
            "schedule_index": self.row["schedule_index"],
            "rollout_id": self.row["rollout_id"],
            "schedule_row_sha256": self.row["schedule_row_sha256"],
            "output_relpath": self.row["output_relpath"],
            "pid": self.process.pid,
            "process_start_time_ticks": (
                identity["start_time_ticks"] if identity is not None else None
            ),
            "attempt_index": self.attempt_index,
            "started_utc": self.started_utc,
            "process_log": str(self.log_path),
            "recovery_event_path": (
                str(self.recovery_event_path)
                if self.recovery_event_path is not None
                else None
            ),
        }


class R2Supervisor:
    def __init__(
        self,
        *,
        schedule_path: Path,
        contract_path: Path,
        manifest_path: Path,
        output_root: Path,
        mode: str,
        evaluator: Path | None = None,
    ) -> None:
        self.schedule_path = schedule_path.resolve()
        self.contract_path = contract_path.resolve()
        self.manifest_path = manifest_path.resolve()
        self.output_root = output_root.resolve()
        self.mode = mode
        self.schedule = json.loads(self.schedule_path.read_text())
        payload = dict(self.schedule)
        observed = payload.pop("schedule_sha256")
        if canonical_hash(payload) != observed:
            raise RuntimeError("R2 schedule self-hash mismatch")
        if (
            self.schedule.get("schema_version")
            != "pact_confirmatory_r2_schedule_v1"
            or self.schedule.get("workers") != WORKERS
            or self.schedule.get("rollouts") != 960
        ):
            raise RuntimeError("R2 schedule design mismatch")
        self.contract = base.load_dispatch_contract(
            self.contract_path,
            self.schedule,
            manifest_path=self.manifest_path,
            output_root=self.output_root,
        )
        if (
            self.contract.get("schema_version") != "pact_r2_dispatch_v1"
            or self.contract["boundary_amendment"].get(
                "all_inflight_rows_rerun"
            )
            is not True
        ):
            raise RuntimeError("dispatch contract lacks the R2 cohort amendment")
        if (
            self.contract["scientific_schedule"]["file_sha256"]
            != base.sha256_file(self.schedule_path)
        ):
            raise RuntimeError("R2 schedule file differs from dispatch contract")
        for label, record in self.contract["frozen_inputs"]["runtime"].items():
            if base.sha256_file(Path(record["path"])) != record["sha256"]:
                raise RuntimeError(
                    f"frozen R2 runtime file changed after dispatch: {label}"
                )
        self.evaluator = (
            evaluator.resolve()
            if evaluator is not None
            else ROOT / "submodules/act/eval_pact_collision_row.py"
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.pid_path = self.output_root / "supervisor_pid.json"
        self.state_path = self.output_root / "supervisor_state.json"
        self.heartbeat_path = self.output_root / "heartbeat.json"
        self.completion_path = self.output_root / "completion_ledger.json"
        self.recovery_root = self.output_root / "recovery_events"
        self.recovery_root.mkdir(exist_ok=True)
        self.started_utc = utc_now()
        self.full_dispatch_started_utc = (
            self.started_utc if mode == "full" else None
        )
        if mode == "full" and self.state_path.exists():
            previous_state = json.loads(self.state_path.read_text())
            if (
                previous_state.get("schema_version")
                == "pact_r2_supervisor_state_v1"
                and previous_state.get("schedule_sha256")
                == self.schedule["schedule_sha256"]
                and previous_state.get("mode") == "full"
                and previous_state.get("status") == "running"
                and previous_state.get("full_dispatch_started_utc") is not None
            ):
                self.full_dispatch_started_utc = previous_state[
                    "full_dispatch_started_utc"
                ]
        self.active: dict[str, ActiveAttempt] = {}
        self.completed_ids: set[str] = set()
        self.completions: list[dict[str, Any]] = self._load_completions()
        self.completed_ids.update(item["rollout_id"] for item in self.completions)
        self.authorized_recovery: dict[str, Path] = {}
        self.abort_reason: str | None = None
        self.pending: collections.deque[dict[str, Any]] = collections.deque()
        self.launch_reservation: dict[str, Any] | None = None
        self.last_heartbeat = 0.0
        self._install_signal_handlers()

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            self.abort_reason = f"supervisor_received_signal_{signum}"

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def _load_completions(self) -> list[dict[str, Any]]:
        if not self.completion_path.exists():
            return []
        document = json.loads(self.completion_path.read_text())
        if document.get("schedule_sha256") != self.schedule["schedule_sha256"]:
            raise RuntimeError("completion ledger schedule mismatch")
        return list(document["completions"])

    def _write_completions(self) -> None:
        write_json(
            self.completion_path,
            {
                "schema_version": "pact_r2_completion_ledger_v1",
                "schedule_sha256": self.schedule["schedule_sha256"],
                "completions": self.completions,
            },
        )

    def _pid_guard(self) -> None:
        if self.pid_path.exists():
            old = json.loads(self.pid_path.read_text())
            old_pid = int(old["pid"])
            if old_pid != os.getpid() and pid_alive(old_pid):
                raise RuntimeError(f"R2 supervisor already active at PID {old_pid}")
        write_json(
            self.pid_path,
            {
                "schema_version": "pact_r2_supervisor_pid_v1",
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "process_group_id": os.getpgrp(),
                "session_id": os.getsid(0),
                "mode": self.mode,
                "started_utc": self.started_utc,
                "schedule_sha256": self.schedule["schedule_sha256"],
                "dispatch_contract_sha256": self.contract[
                    "dispatch_contract_sha256"
                ],
            },
        )

    def _row_dir(self, row: dict[str, Any]) -> Path:
        return self.output_root / row["output_relpath"]

    def _valid_result(self, row: dict[str, Any]) -> bool:
        path = self._row_dir(row) / "result.json"
        if not path.exists():
            return False
        base._validate_scientific_result(path, row)
        base._validate_boundary(
            self._row_dir(row) / "initial_observation_accepted.json",
            row,
        )
        return True

    def _driver_complete(self, row: dict[str, Any]) -> bool:
        path = self._row_dir(row) / "driver_result.json"
        if not path.exists():
            return False
        driver = json.loads(path.read_text())
        return (
            driver.get("rollout_id") == row["rollout_id"]
            and driver.get("schedule_row_sha256") == row["schedule_row_sha256"]
            and driver.get("status") == "complete"
        )

    def _record_completion(self, row: dict[str, Any], attempt_index: int) -> None:
        if row["rollout_id"] in self.completed_ids:
            return
        record = {
            "schedule_index": row["schedule_index"],
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "completed_utc": utc_now(),
            "attempt_index": attempt_index,
        }
        self.completions.append(record)
        self.completed_ids.add(row["rollout_id"])
        self._write_completions()

    def _reconcile_result(
        self,
        row: dict[str, Any],
        *,
        attempt_index: int,
        returncode: int | None,
    ) -> None:
        if not self._valid_result(row):
            raise RuntimeError("cannot reconcile an absent/invalid result")
        row_dir = self._row_dir(row)
        attempts = load_attempts(row_dir / "attempt_ledger.json", row)
        if not attempts or attempts[-1].get("attempt_index") != attempt_index:
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "status": "complete",
                    "returncode": returncode,
                    "initial_observation_accepted": True,
                    "scientific_result_written": True,
                    "completed_utc": utc_now(),
                }
            )
            write_attempts(row_dir / "attempt_ledger.json", row, attempts)
        driver = {
            "schema_version": "pact_r2_driver_v1",
            "status": "complete",
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "returncode": returncode,
            "attempt_count": len(attempts),
            "pre_observation_infrastructure_failures": sum(
                item["status"] == "pre_observation_infrastructure_failure"
                for item in attempts
            ),
            "group_recovery_attempts": sum(
                item["status"].startswith("group_termination_")
                for item in attempts
            ),
            "final_attempt_index": attempt_index,
            "error": None,
        }
        write_json(row_dir / "driver_result.json", driver)
        self._record_completion(row, attempt_index)

    def _load_previous_active(self) -> list[dict[str, Any]]:
        if not self.state_path.exists():
            return []
        state = json.loads(self.state_path.read_text())
        if (
            state.get("schedule_sha256") != self.schedule["schedule_sha256"]
            or state.get("status") != "running"
        ):
            return []
        previous = list(state.get("active_cohort", []))
        reservation = state.get("launch_reservation")
        if reservation is not None:
            matches = self._find_evaluator_pids(reservation["rollout_id"])
            if reservation.get("pid") is None and len(matches) == 1:
                reservation = dict(reservation)
                reservation["pid"] = matches[0]
                identity = process_identity(matches[0])
                reservation["process_start_time_ticks"] = (
                    identity["start_time_ticks"]
                    if identity is not None
                    else None
                )
            elif reservation.get("pid") is None and len(matches) > 1:
                raise RuntimeError(
                    "launch reservation resolves to multiple evaluator PIDs"
                )
            row_dir = self.output_root / reservation["output_relpath"]
            if (
                reservation.get("pid") is not None
                or (row_dir / "initial_observation_accepted.json").exists()
                or (row_dir / "result.json").exists()
            ):
                previous.append(reservation)
        unique: dict[str, dict[str, Any]] = {}
        for item in previous:
            unique[item["rollout_id"]] = item
        return list(unique.values())

    def _find_evaluator_pids(self, rollout_id: str) -> list[int]:
        matches = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                argv = [
                    value.decode(errors="replace")
                    for value in (entry / "cmdline").read_bytes().split(b"\0")
                    if value
                ]
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if (
                str(self.evaluator) in argv
                and "--rollout-id" in argv
                and rollout_id in argv
            ):
                matches.append(int(entry.name))
        return matches

    def _append_recovered_attempt(
        self,
        item: dict[str, Any],
        row: dict[str, Any],
        *,
        status: str,
        recovery_event_sha256: str | None = None,
    ) -> None:
        row_dir = self._row_dir(row)
        attempts_path = row_dir / "attempt_ledger.json"
        attempts = load_attempts(attempts_path, row)
        attempt_index = int(item["attempt_index"])
        if len(attempts) != attempt_index:
            raise RuntimeError("durable attempt index differs from attempt ledger")
        log_path = Path(item["process_log"])
        attempts.append(
            {
                "attempt_index": attempt_index,
                "status": status,
                "returncode": None,
                "process_log": str(log_path),
                "process_log_sha256": (
                    base.sha256_file(log_path) if log_path.exists() else None
                ),
                "initial_observation_accepted": (
                    row_dir / "initial_observation_accepted.json"
                ).exists(),
                "scientific_result_written": (row_dir / "result.json").exists(),
                "recovery_event_sha256": recovery_event_sha256,
                "finished_utc": utc_now(),
            }
        )
        write_attempts(attempts_path, row, attempts)

    def _reconcile_previous_supervisor_loss(self) -> None:
        previous = self._load_previous_active()
        if not previous:
            return
        rows_by_id = {
            row["rollout_id"]: row for row in self.schedule["rows"]
        }
        if any(recorded_process_alive(item) for item in previous):
            raise RuntimeError(
                "prior supervisor is gone but one or more evaluator PIDs remain active"
            )
        rows = [rows_by_id[item["rollout_id"]] for item in previous]
        results = [self._valid_result(row) for row in rows]
        if all(results):
            for item, row, valid in zip(previous, rows, results):
                if valid:
                    self._reconcile_result(
                        row,
                        attempt_index=int(item["attempt_index"]),
                        returncode=None,
                    )
            return
        if any(results):
            for item, row, valid in zip(previous, rows, results):
                if valid:
                    self._reconcile_result(
                        row,
                        attempt_index=int(item["attempt_index"]),
                        returncode=None,
                    )
                elif not (
                    self._row_dir(row) / "initial_observation_accepted.json"
                ).exists():
                    self._append_recovered_attempt(
                        item,
                        row,
                        status="pre_observation_supervisor_loss_with_cohort_results",
                    )
            return
        event_path = self._freeze_recovery_event(
            [
                {
                    "row": row,
                    "pid": (
                        int(item["pid"]) if item.get("pid") is not None else None
                    ),
                    "attempt_index": int(item["attempt_index"]),
                    "log_path": Path(item["process_log"]),
                    "returncode": None,
                }
                for item, row in zip(previous, rows)
            ],
            reason="supervisor_restart_all_recorded_evaluators_dead",
        )
        event = json.loads(event_path.read_text())
        for item, row in zip(previous, rows):
            self._append_recovered_attempt(
                item,
                row,
                status=(
                    "group_termination_post_observation"
                    if (
                        self._row_dir(row)
                        / "initial_observation_accepted.json"
                    ).exists()
                    else "group_termination_pre_observation"
                ),
                recovery_event_sha256=event["recovery_event_sha256"],
            )
            self.authorized_recovery[row["rollout_id"]] = event_path

    def _prepare_pending(self) -> None:
        rows = (
            [self.schedule["rows"][0]]
            if self.mode == "smoke"
            else list(self.schedule["rows"])
        )
        if self.mode == "full":
            base.validate_launch_smoke(
                schedule=self.schedule,
                contract=self.contract,
                output_root=self.output_root,
            )
            self._validate_detachment_proof()
        for row in rows:
            if self._valid_result(row):
                if not self._driver_complete(row):
                    self._reconcile_result(
                        row,
                        attempt_index=max(
                            0,
                            len(
                                load_attempts(
                                    self._row_dir(row) / "attempt_ledger.json",
                                    row,
                                )
                            )
                            - 1,
                        ),
                        returncode=0,
                    )
                else:
                    self._record_completion(row, 0)
                continue
            row_dir = self._row_dir(row)
            if (
                (row_dir / "initial_observation_accepted.json").exists()
                and row["rollout_id"] not in self.authorized_recovery
            ):
                raise RuntimeError(
                    "result-free boundary marker lacks all-cohort recovery authorization"
                )
            self.pending.append(row)

    def _validate_detachment_proof(self) -> dict[str, Any]:
        spec = self.contract.get("detachment_proof", {})
        if spec.get("required_before_full_dispatch") is not True:
            raise RuntimeError("R2 dispatch does not require detachment proof")
        path = self.output_root / spec["required_artifact"]
        if not path.exists():
            raise RuntimeError("full dispatch refused: detachment proof is missing")
        proof = json.loads(path.read_text())
        payload = dict(proof)
        observed = payload.pop("detachment_proof_sha256", None)
        if canonical_hash(payload) != observed:
            raise RuntimeError("detachment proof self-hash mismatch")
        smoke = self.contract["launch_smoke"]
        expected = {
            "passed": True,
            "dispatch_contract_sha256": self.contract[
                "dispatch_contract_sha256"
            ],
            "schedule_sha256": self.schedule["schedule_sha256"],
            "rollout_id": smoke["rollout_id"],
            "schedule_row_sha256": smoke["schedule_row_sha256"],
            "endpoint_fields_inspected": False,
        }
        for key, value in expected.items():
            if proof.get(key) != value:
                raise RuntimeError(f"detachment proof {key} mismatch")
        if (
            proof["heartbeat"].get("advanced_after_shell_kill") is not True
            or proof["supervisor"].get("survived_or_completed") is not True
            or proof["evaluator"].get("survived_or_completed") is not True
            or proof["smoke"].get("result_count") != 1
        ):
            raise RuntimeError("detachment proof survival checks did not pass")
        result_path = (
            self.output_root / smoke["output_relpath"] / "result.json"
        )
        if base.sha256_file(result_path) != proof["smoke"]["result_sha256"]:
            raise RuntimeError("detachment proof smoke result hash changed")
        return proof

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
        command[1] = str(self.evaluator)
        command.extend(["--attempt-index", str(attempt_index)])
        if recovery_event is not None:
            command.extend(
                ["--inflight-recovery-event", str(recovery_event.resolve())]
            )
        return command

    def _start_row(self, row: dict[str, Any]) -> None:
        row_dir = self._row_dir(row)
        row_dir.mkdir(parents=True, exist_ok=True)
        attempts = load_attempts(row_dir / "attempt_ledger.json", row)
        attempt_index = len(attempts)
        recovery_event = self.authorized_recovery.pop(row["rollout_id"], None)
        if (
            (row_dir / "initial_observation_accepted.json").exists()
            and recovery_event is None
        ):
            raise RuntimeError("selected post-observation retry refused")
        log_path = row_dir / f"process_attempt_{attempt_index:03d}.log"
        log_stream = log_path.open("wb")
        command = self._command(
            row,
            attempt_index=attempt_index,
            recovery_event=recovery_event,
        )
        self.launch_reservation = {
            "schedule_index": row["schedule_index"],
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "output_relpath": row["output_relpath"],
            "pid": None,
            "attempt_index": attempt_index,
            "started_utc": utc_now(),
            "process_log": str(log_path),
            "recovery_event_path": (
                str(recovery_event) if recovery_event is not None else None
            ),
        }
        self._write_state("running")
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT / "submodules/act",
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        except OSError:
            log_stream.flush()
            log_stream.close()
            self.launch_reservation = None
            if recovery_event is not None:
                self.authorized_recovery[row["rollout_id"]] = recovery_event
            self.pending.appendleft(row)
            self._write_state("running")
            return
        self.launch_reservation["pid"] = process.pid
        identity = process_identity(process.pid)
        self.launch_reservation["process_start_time_ticks"] = (
            identity["start_time_ticks"] if identity is not None else None
        )
        self._write_state("running")
        active = ActiveAttempt(
            row=row,
            process=process,
            log_stream=log_stream,
            log_path=log_path,
            attempt_index=attempt_index,
            started_utc=utc_now(),
            recovery_event_path=recovery_event,
        )
        self.active[row["rollout_id"]] = active
        self.launch_reservation = None
        self._write_state("running")

    def _close_attempt(self, active: ActiveAttempt) -> None:
        if not active.log_stream.closed:
            active.log_stream.flush()
            active.log_stream.close()

    def _append_attempt(
        self,
        active: ActiveAttempt,
        *,
        status: str,
        recovery_event_sha256: str | None = None,
    ) -> None:
        self._close_attempt(active)
        row_dir = self._row_dir(active.row)
        attempts = load_attempts(row_dir / "attempt_ledger.json", active.row)
        if len(attempts) != active.attempt_index:
            raise RuntimeError("attempt ledger changed while evaluator was active")
        attempts.append(
            {
                "attempt_index": active.attempt_index,
                "status": status,
                "returncode": active.process.poll(),
                "process_log": str(active.log_path),
                "process_log_sha256": base.sha256_file(active.log_path),
                "initial_observation_accepted": (
                    row_dir / "initial_observation_accepted.json"
                ).exists(),
                "scientific_result_written": (row_dir / "result.json").exists(),
                "recovery_event_sha256": recovery_event_sha256,
                "finished_utc": utc_now(),
            }
        )
        write_attempts(row_dir / "attempt_ledger.json", active.row, attempts)

    def _freeze_recovery_event(
        self,
        cohort: list[dict[str, Any]],
        *,
        reason: str,
    ) -> Path:
        event_index = len(list(self.recovery_root.glob("event_*.json")))
        rows = []
        for item in sorted(
            cohort, key=lambda value: int(value["row"]["schedule_index"])
        ):
            row = item["row"]
            row_dir = self._row_dir(row)
            log_path = Path(item["log_path"])
            rows.append(
                {
                    "schedule_index": row["schedule_index"],
                    "rollout_id": row["rollout_id"],
                    "schedule_row_sha256": row["schedule_row_sha256"],
                    "pid": item["pid"],
                    "attempt_index": item["attempt_index"],
                    "returncode": item["returncode"],
                    "boundary_marker_present": (
                        row_dir / "initial_observation_accepted.json"
                    ).exists(),
                    "result_present": False,
                    "process_log": str(log_path),
                    "process_log_sha256": base.sha256_file(log_path),
                }
            )
        event: dict[str, Any] = {
            "schema_version": "pact_r2_group_recovery_v1",
            "event_index": event_index,
            "schedule_sha256": self.schedule["schedule_sha256"],
            "dispatch_contract_sha256": self.contract[
                "dispatch_contract_sha256"
            ],
            "reason": reason,
            "qualifying_indiscriminate_termination": True,
            "all_inflight_rows_rerun": True,
            "result_absent_for_all": True,
            "active_cohort_size": len(rows),
            "rows": rows,
            "frozen_utc": utc_now(),
        }
        event["recovery_event_sha256"] = canonical_hash(event)
        path = self.recovery_root / f"event_{event_index:03d}.json"
        write_json(path, event)
        return path

    def _group_recover(self, cohort: list[ActiveAttempt], *, reason: str) -> None:
        records = []
        for active in cohort:
            self._close_attempt(active)
            records.append(
                {
                    "row": active.row,
                    "pid": active.process.pid,
                    "attempt_index": active.attempt_index,
                    "log_path": active.log_path,
                    "returncode": active.process.poll(),
                }
            )
        event_path = self._freeze_recovery_event(records, reason=reason)
        event = json.loads(event_path.read_text())
        for active in cohort:
            status = (
                "group_termination_post_observation"
                if (
                    self._row_dir(active.row)
                    / "initial_observation_accepted.json"
                ).exists()
                else "group_termination_pre_observation"
            )
            self._append_attempt(
                active,
                status=status,
                recovery_event_sha256=event["recovery_event_sha256"],
            )
            self.active.pop(active.row["rollout_id"], None)
            self.authorized_recovery[active.row["rollout_id"]] = event_path
        for active in sorted(
            cohort,
            key=lambda item: int(item.row["schedule_index"]),
            reverse=True,
        ):
            self.pending.appendleft(active.row)
        self._write_state("running")

    def _handle_ended(self) -> None:
        ended = [
            active
            for active in self.active.values()
            if active.process.poll() is not None
        ]
        if not ended:
            return
        no_result = [
            active for active in ended if not self._valid_result(active.row)
        ]
        if no_result:
            snapshot = list(self.active.values())
            deadline = time.monotonic() + LOSS_WINDOW_SECONDS
            while time.monotonic() < deadline:
                if all(item.process.poll() is not None for item in snapshot):
                    break
                time.sleep(0.1)
            all_exited = all(item.process.poll() is not None for item in snapshot)
            no_snapshot_results = all(
                not self._valid_result(item.row) for item in snapshot
            )
            if all_exited and no_snapshot_results:
                self._group_recover(
                    snapshot,
                    reason="live_all_active_evaluators_exited_within_five_seconds",
                )
                return

        for active in list(self.active.values()):
            if active.process.poll() is None:
                continue
            row = active.row
            row_dir = self._row_dir(row)
            if self._valid_result(row):
                self._append_attempt(active, status="complete")
                self._reconcile_result(
                    row,
                    attempt_index=active.attempt_index,
                    returncode=active.process.returncode,
                )
                self.active.pop(row["rollout_id"], None)
            elif not (row_dir / "initial_observation_accepted.json").exists():
                self._append_attempt(
                    active,
                    status="pre_observation_infrastructure_failure",
                )
                self.active.pop(row["rollout_id"], None)
                self.pending.appendleft(row)
            else:
                self._append_attempt(
                    active,
                    status="nonqualifying_post_observation_failure",
                )
                attempts = load_attempts(row_dir / "attempt_ledger.json", row)
                write_json(
                    row_dir / "driver_result.json",
                    {
                        "schema_version": "pact_r2_driver_v1",
                        "status": "post_boundary_failure",
                        "rollout_id": row["rollout_id"],
                        "schedule_row_sha256": row["schedule_row_sha256"],
                        "attempt_count": len(attempts),
                        "returncode": active.process.returncode,
                        "error": (
                            "non-qualifying post-observation evaluator loss; "
                            "selected retry forbidden"
                        ),
                    },
                )
                self.active.pop(row["rollout_id"], None)
                self.abort_reason = (
                    "nonqualifying_post_observation_failure_"
                    + row["rollout_id"]
                )
        self._write_state("running")

    def _write_state(self, status: str) -> None:
        write_json(
            self.state_path,
            {
                "schema_version": "pact_r2_supervisor_state_v1",
                "schedule_sha256": self.schedule["schedule_sha256"],
                "dispatch_contract_sha256": self.contract[
                    "dispatch_contract_sha256"
                ],
                "mode": self.mode,
                "status": status,
                "supervisor_pid": os.getpid(),
                "started_utc": self.started_utc,
                "full_dispatch_started_utc": self.full_dispatch_started_utc,
                "active_cohort": [
                    active.state_record()
                    for active in sorted(
                        self.active.values(),
                        key=lambda item: int(item.row["schedule_index"]),
                    )
                ],
                "launch_reservation": self.launch_reservation,
                "pending_count": len(self.pending),
                "completed_count": len(self.completed_ids),
                "abort_reason": self.abort_reason,
                "updated_utc": utc_now(),
            },
        )

    def _write_heartbeat(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_heartbeat < HEARTBEAT_SECONDS:
            return
        heartbeat: dict[str, Any] = {
            "schema_version": "pact_r2_heartbeat_v1",
            "schedule_sha256": self.schedule["schedule_sha256"],
            "mode": self.mode,
            "supervisor_pid": os.getpid(),
            "supervisor_ppid": os.getppid(),
            "process_group_id": os.getpgrp(),
            "session_id": os.getsid(0),
            "active_count": len(self.active),
            "active": [
                {
                    "schedule_index": active.row["schedule_index"],
                    "rollout_id": active.row["rollout_id"],
                    "pid": active.process.pid,
                    "attempt_index": active.attempt_index,
                }
                for active in sorted(
                    self.active.values(),
                    key=lambda item: int(item.row["schedule_index"]),
                )
            ],
            "pending_count": len(self.pending),
            "completed_count": len(self.completed_ids),
            "abort_reason": self.abort_reason,
            "updated_utc": utc_now(),
        }
        heartbeat["heartbeat_sha256"] = canonical_hash(heartbeat)
        write_json(self.heartbeat_path, heartbeat)
        self.last_heartbeat = now

    def _write_launch_smoke(self) -> None:
        row = self.schedule["rows"][0]
        row_dir = self._row_dir(row)
        driver = json.loads((row_dir / "driver_result.json").read_text())
        artifact: dict[str, Any] = {
            "schema_version": "pact_r2_launch_smoke_v1",
            "dispatch_contract_sha256": self.contract[
                "dispatch_contract_sha256"
            ],
            "scientific_schedule_sha256": self.schedule["schedule_sha256"],
            "passed": driver["status"] == "complete",
            "smoke_invocations": 1,
            "schedule_index": row["schedule_index"],
            "rollout_id": row["rollout_id"],
            "instance_episode_id": row["instance_episode_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "driver_status": driver["status"],
            "attempt_count": driver["attempt_count"],
            "scientific_result_sha256": base.sha256_file(
                row_dir / "result.json"
            ),
            "driver_result_sha256": base.sha256_file(
                row_dir / "driver_result.json"
            ),
            "endpoint_fields_inspected": False,
        }
        artifact["launch_smoke_sha256"] = canonical_hash(artifact)
        write_json(self.output_root / "launch_smoke.json", artifact)

    def _write_execution_summary(self) -> dict[str, Any]:
        drivers = {}
        for row in self.schedule["rows"]:
            path = self._row_dir(row) / "driver_result.json"
            if path.exists():
                drivers[row["rollout_id"]] = json.loads(path.read_text())
        complete = [
            row["rollout_id"]
            for row in self.schedule["rows"]
            if drivers.get(row["rollout_id"], {}).get("status") == "complete"
            and self._valid_result(row)
        ]
        noncomplete = [
            row["rollout_id"]
            for row in self.schedule["rows"]
            if row["rollout_id"] not in complete
        ]
        summary = {
            "schema_version": "pact_r2_execution_summary_v1",
            "schedule_sha256": self.schedule["schedule_sha256"],
            "dispatch_contract_sha256": self.contract[
                "dispatch_contract_sha256"
            ],
            "workers": WORKERS,
            "mode": self.mode,
            "expected": 1 if self.mode == "smoke" else 960,
            "complete_count": (
                int(self.schedule["rows"][0]["rollout_id"] in complete)
                if self.mode == "smoke"
                else len(complete)
            ),
            "noncomplete": (
                []
                if self.mode == "smoke"
                and self.schedule["rows"][0]["rollout_id"] in complete
                else (
                    [self.schedule["rows"][0]["rollout_id"]]
                    if self.mode == "smoke"
                    else noncomplete
                )
            ),
            "recovery_event_count": len(
                list(self.recovery_root.glob("event_*.json"))
            ),
            "abort_reason": self.abort_reason,
            "scientific_schedule_reconciled": (
                self.abort_reason is None
                and (
                    self.schedule["rows"][0]["rollout_id"] in complete
                    if self.mode == "smoke"
                    else len(complete) == 960
                )
            ),
            "endpoint_fields_inspected": False,
            "finished_utc": utc_now(),
        }
        write_json(self.output_root / f"{self.mode}_execution_summary.json", summary)
        if self.mode == "full":
            write_json(self.output_root / "execution_summary.json", summary)
        return summary

    def run(self) -> int:
        self._pid_guard()
        self._reconcile_previous_supervisor_loss()
        self._prepare_pending()
        self._write_state("running")
        self._write_heartbeat(force=True)
        while self.pending or self.active:
            while (
                self.pending
                and len(self.active) < WORKERS
                and self.abort_reason is None
            ):
                self._start_row(self.pending.popleft())
            self._handle_ended()
            self._write_heartbeat()
            if self.abort_reason is not None and not self.active:
                break
            time.sleep(0.2)
        if self.mode == "smoke" and self.abort_reason is None:
            self._write_launch_smoke()
        summary = self._write_execution_summary()
        final_status = (
            "complete"
            if summary["scientific_schedule_reconciled"]
            else "incomplete"
        )
        self._write_state(final_status)
        self._write_heartbeat(force=True)
        return 0 if final_status == "complete" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch-contract", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--evaluator", type=Path)
    args = parser.parse_args()
    active_protected = base.protected_eval_processes()
    if active_protected:
        raise SystemExit(
            f"protected confirmatory evaluator is active: {active_protected}"
        )
    supervisor = R2Supervisor(
        schedule_path=args.schedule,
        contract_path=args.dispatch_contract,
        manifest_path=args.manifest,
        output_root=args.output_root,
        mode=args.mode,
        evaluator=args.evaluator,
    )
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
