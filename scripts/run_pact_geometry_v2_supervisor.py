#!/usr/bin/env python3
"""Durable eight-worker attempt-2 supervisor with a frozen batch watchdog."""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import time
from pathlib import Path

import run_pact_frontend_screen_supervisor as screen
import run_pact_geometry_supervisor as v1


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "submodules/act/eval_pact_geometry_generalization_v2_row.py"
WORKERS = 8
WATCHDOG_SECONDS = 600.0
ARMS = {"ACT", "PACT", "PACT_PERMUTED"}


class GeometryV2Supervisor(v1.GeometrySupervisor):
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
        if screen.canonical_hash(payload) != observed:
            raise RuntimeError("attempt-2 geometry schedule self-hash mismatch")
        surviving = self.schedule.get("surviving_condition_ids", [])
        if (
            self.schedule.get("schema_version") != "pact_geometry_generalization_v2_schedule"
            or self.schedule.get("workers") != WORKERS
            or self.schedule.get("rollouts") != 900
            or len(self.schedule.get("rows", [])) != 900
            or {row["arm"] for row in self.schedule["rows"]} != ARMS
            or set(self.schedule.get("checkpoint_seeds", [])) != {3101, 3102, 3103}
            or surviving != ["C0", "C2", "Z_093", "HALF_Y_030"]
        ):
            raise RuntimeError("attempt-2 geometry schedule design mismatch")
        self.contract = screen.base.load_dispatch_contract(
            self.contract_path,
            self.schedule,
            manifest_path=self.manifest_path,
            output_root=self.output_root,
        )
        if self.contract.get("schema_version") != "pact_geometry_generalization_v2_dispatch":
            raise RuntimeError("wrong attempt-2 geometry dispatch contract")
        if self.contract["execution"].get("fixed_worker_count") != WORKERS:
            raise RuntimeError("geometry worker count changed")
        if float(self.contract["watchdog"]["no_completion_seconds"]) != WATCHDOG_SECONDS:
            raise RuntimeError("geometry watchdog interval changed")
        for label, record in self.contract["frozen_inputs"]["runtime"].items():
            if screen.base.sha256_file(Path(record["path"])) != record["sha256"]:
                raise RuntimeError(f"frozen attempt-2 runtime changed: {label}")
        protected = screen.protected_eval_processes()
        if protected:
            raise RuntimeError(f"protected shared evaluation is active: {protected}")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.pid_path = self.output_root / "supervisor_pid.json"
        self.state_path = self.output_root / "supervisor_state.json"
        self.heartbeat_path = self.output_root / "heartbeat.json"
        self.completion_path = self.output_root / "completion_ledger.json"
        self.recovery_root = self.output_root / "recovery_events"
        self.recovery_root.mkdir(exist_ok=True)
        self.watchdog_root = self.output_root / "watchdog_events"
        self.watchdog_root.mkdir(exist_ok=True)
        self.active = {}
        self.pending = collections.deque()
        self.completions = self._load_completions()
        self.completed_ids = {item["rollout_id"] for item in self.completions}
        self.authorized_recovery = {}
        self.abort_reason = None
        self.started_utc = screen.utc_now()
        self.full_dispatch_started_utc = self.started_utc if mode == "full" else None
        self.last_heartbeat = 0.0
        self.last_finalization = time.monotonic()
        self._install_signals()

    def _command(self, row, *, attempt_index: int, recovery_event: Path | None):
        command = screen.base.command_for(
            row,
            manifest_path=self.manifest_path,
            output_dir=self._row_dir(row),
            save_video=False,
        )
        command[0] = str(screen.PYTHON)
        command[1] = str(EVALUATOR)
        if row["arm"] == "PACT_PERMUTED":
            command.extend(
                [
                    "--surface-encoder",
                    row["surface_encoder_path"],
                    "--surface-encoder-sha256",
                    row["surface_encoder_sha256"],
                    "--token-plan-manifest",
                    row["token_plan_manifest_path"],
                    "--token-plan-row",
                    str(row["token_plan_row"]),
                ]
            )
        command.extend(["--attempt-index", str(attempt_index)])
        if recovery_event is not None:
            command.extend(["--inflight-recovery-event", str(recovery_event)])
        return command

    def _reconcile_complete(self, active) -> None:
        super()._reconcile_complete(active)
        self.last_finalization = time.monotonic()

    def _watchdog_event(self, kind: str, rows: list[dict], **extra) -> Path:
        index = len(list(self.watchdog_root.glob("event_*.json")))
        event = {
            "schema_version": "pact_geometry_generalization_v2_watchdog_event",
            "schedule_sha256": self.schedule["schedule_sha256"],
            "event_index": index,
            "kind": kind,
            "no_completion_seconds": WATCHDOG_SECONDS,
            "rows": rows,
            "endpoint_fields_read": False,
            "recorded_utc": screen.utc_now(),
            **extra,
        }
        event["watchdog_event_sha256"] = screen.canonical_hash(event)
        path = self.watchdog_root / f"event_{index:03d}_{event['watchdog_event_sha256'][:12]}.json"
        screen.base.write_json_atomic(path, event)
        return path

    def _watchdog(self) -> None:
        if not self.active or time.monotonic() - self.last_finalization < WATCHDOG_SECONDS:
            return
        cohort = sorted(self.active.values(), key=lambda item: item.row["schedule_index"])
        rows = []
        any_boundary = False
        any_result = False
        for active in cohort:
            row_dir = self._row_dir(active.row)
            boundary = (row_dir / "initial_observation_accepted.json").exists()
            result = (row_dir / "result.json").exists()
            any_boundary |= boundary
            any_result |= result
            rows.append(
                {
                    "schedule_index": active.row["schedule_index"],
                    "rollout_id": active.row["rollout_id"],
                    "schedule_row_sha256": active.row["schedule_row_sha256"],
                    "attempt_index": active.attempt_index,
                    "pid": active.process.pid,
                    "initial_observation_accepted": boundary,
                    "result_present": result,
                }
            )
        if any_boundary or any_result:
            self._watchdog_event(
                "deferred_active_scientific_boundary",
                rows,
                restart_performed=False,
                reason="at_least_one_active_row_holds_boundary_or_result",
            )
            self.last_finalization = time.monotonic()
            return
        event_path = self._watchdog_event(
            "all_inflight_pre_boundary_restart",
            rows,
            restart_performed=True,
            all_inflight_rows_restarted=True,
            individual_row_restart=False,
        )
        event = json.loads(event_path.read_text())
        for active in cohort:
            if active.process.poll() is None:
                try:
                    os.killpg(active.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and any(
            active.process.poll() is None for active in cohort
        ):
            time.sleep(0.25)
        for active in cohort:
            if active.process.poll() is None:
                try:
                    os.killpg(active.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            active.process.wait()
            self._finish_attempt(
                active,
                status="watchdog_all_inflight_pre_boundary_restart",
                recovery_event_sha256=event["watchdog_event_sha256"],
            )
            self.active.pop(active.row["rollout_id"])
            self.pending.append(active.row)
        self.last_finalization = time.monotonic()

    def run(self) -> int:
        identity = screen.process_identity(os.getpid())
        pid_record = {
            "schema_version": "pact_geometry_generalization_v2_supervisor_pid",
            "pid": os.getpid(),
            "identity": identity,
            "mode": self.mode,
            "schedule_sha256": self.schedule["schedule_sha256"],
            "started_utc": self.started_utc,
        }
        pid_record["pid_record_sha256"] = screen.canonical_hash(pid_record)
        screen.base.write_json_atomic(self.pid_path, pid_record)
        self._prepare()
        self._write_state()
        self._heartbeat(force=True)
        target = 1 if self.mode == "smoke" else 900
        while len(self.completed_ids) < target and not self.abort_reason:
            limit = 1 if self.mode == "smoke" else WORKERS
            while self.pending and len(self.active) < limit:
                self._launch(self.pending.popleft())
            self._handle_exits()
            self._watchdog()
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
            "schema_version": "pact_geometry_generalization_v2_execution",
            "schedule_sha256": self.schedule["schedule_sha256"],
            "dispatch_contract_sha256": self.contract["dispatch_contract_sha256"],
            "mode": self.mode,
            "workers": 1 if self.mode == "smoke" else WORKERS,
            "expected": target,
            "complete_count": len(self.completed_ids),
            "scientific_schedule_reconciled": len(self.completed_ids) == target,
            "finished_utc": screen.utc_now(),
        }
        screen.base.write_json_atomic(self.output_root / f"{self.mode}_execution_summary.json", summary)
        self._write_state(status="complete")
        self._heartbeat(force=True)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch-contract", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    args = parser.parse_args()
    return GeometryV2Supervisor(
        schedule_path=args.schedule,
        contract_path=args.dispatch_contract,
        manifest_path=args.manifest,
        output_root=args.output_root,
        mode=args.mode,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
