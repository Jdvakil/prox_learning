#!/usr/bin/env python3
"""Durable twelve-worker supervisor for the frozen RGB-blur sweep."""

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
EVALUATOR = ROOT / "submodules/act/eval_pact_blur_sweep_row.py"
WORKERS = 12
WATCHDOG_SECONDS = 600.0
ARMS = {"ACT", "PACT", "PACT_PERMUTED"}
EXECUTION_AMENDMENT = (
    ROOT
    / "diagnostics_output/pact_blur_sweep/execution_recovery_amendment.json"
)


class BlurSweepSupervisor(v1.GeometrySupervisor):
    def __init__(
        self,
        *,
        schedule_path: Path,
        contract_path: Path,
        manifest_path: Path,
        output_root: Path,
        mode: str,
        resume_recovery_event: Path | None = None,
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
            raise RuntimeError("RGB blur schedule self-hash mismatch")
        if (
            self.schedule.get("schema_version") != "pact_blur_sweep_schedule_v1"
            or self.schedule.get("workers") != WORKERS
            or self.schedule.get("rollouts") != 900
            or len(self.schedule.get("rows", [])) != 900
            or {row["arm"] for row in self.schedule["rows"]} != ARMS
            or set(self.schedule.get("checkpoint_seeds", [])) != {3101, 3102, 3103}
            or self.schedule.get("blur_sigmas") != [0.0, 0.5, 1.0, 2.0]
            or self.schedule.get("instances_count") != 25
        ):
            raise RuntimeError("RGB blur schedule design mismatch")
        self.contract = screen.base.load_dispatch_contract(
            self.contract_path,
            self.schedule,
            manifest_path=self.manifest_path,
            output_root=self.output_root,
        )
        if self.contract.get("schema_version") != "pact_blur_sweep_dispatch":
            raise RuntimeError("wrong RGB blur dispatch contract")
        if self.contract["execution"].get("fixed_worker_count") != WORKERS:
            raise RuntimeError("blur worker count changed")
        if float(self.contract["watchdog"]["no_completion_seconds"]) != WATCHDOG_SECONDS:
            raise RuntimeError("blur watchdog interval changed")
        amendment = None
        if EXECUTION_AMENDMENT.is_file():
            amendment = json.loads(EXECUTION_AMENDMENT.read_text())
            amendment_payload = dict(amendment)
            amendment_sha = amendment_payload.pop("amendment_sha256", None)
            if (
                screen.canonical_hash(amendment_payload) != amendment_sha
                or amendment.get("schema_version")
                != "pact_blur_sweep_execution_recovery_amendment_v1"
                or amendment.get("schedule_sha256")
                != self.schedule["schedule_sha256"]
                or amendment.get("dispatch_contract_sha256")
                != self.contract["dispatch_contract_sha256"]
            ):
                raise RuntimeError("blur execution amendment is invalid")
        for label, record in self.contract["frozen_inputs"]["runtime"].items():
            current_sha = screen.base.sha256_file(Path(record["path"]))
            if current_sha == record["sha256"]:
                continue
            override = (amendment or {}).get("runtime_overrides", {}).get(label)
            if (
                override is None
                or override.get("path") != record["path"]
                or override.get("old_sha256") != record["sha256"]
                or override.get("new_sha256") != current_sha
            ):
                raise RuntimeError(f"frozen RGB-blur runtime changed: {label}")
        if amendment is not None:
            publisher = amendment["publisher_runtime"]
            if screen.base.sha256_file(Path(publisher["path"])) != publisher[
                "new_sha256"
            ]:
                raise RuntimeError("amended no-video publisher changed")
            requested_affinity = set(amendment["thread_affinity"]["cpu_ids"])
            os.sched_setaffinity(0, requested_affinity)
            if os.sched_getaffinity(0) != requested_affinity:
                raise RuntimeError("amended supervisor CPU affinity did not apply")
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
        self.resume_recovery_event = (
            resume_recovery_event.resolve()
            if resume_recovery_event is not None
            else None
        )
        self.resume_event = self._load_resume_event()
        self.abort_reason = None
        self.started_utc = screen.utc_now()
        self.full_dispatch_started_utc = self.started_utc if mode == "full" else None
        self.last_heartbeat = 0.0
        self.last_finalization = time.monotonic()
        self._install_signals()

    def _load_resume_event(self) -> dict | None:
        if self.resume_recovery_event is None:
            return None
        event = json.loads(self.resume_recovery_event.read_text())
        payload = dict(event)
        observed = payload.pop("recovery_event_sha256", None)
        if (
            observed != screen.canonical_hash(payload)
            or event.get("schema_version")
            != "pact_frontend_screen_group_recovery_v1"
            or event.get("schedule_sha256") != self.schedule["schedule_sha256"]
            or event.get("qualifying_indiscriminate_termination") is not True
            or event.get("all_inflight_rows_rerun") is not True
            or event.get("result_absent_for_all") is not True
            or event.get("active_cohort_size") != len(event.get("rows", []))
            or event.get("resume_after_supervisor_abort") is not True
        ):
            raise RuntimeError("blur resume recovery event is invalid")
        rollout_ids = [row["rollout_id"] for row in event["rows"]]
        if len(rollout_ids) != len(set(rollout_ids)) or len(rollout_ids) != WORKERS:
            raise RuntimeError("blur resume cohort is not exactly twelve unique rows")
        schedule_by_id = {row["rollout_id"]: row for row in self.schedule["rows"]}
        for event_row in event["rows"]:
            row = schedule_by_id.get(event_row["rollout_id"])
            if (
                row is None
                or row["schedule_row_sha256"]
                != event_row["schedule_row_sha256"]
                or (self.output_root / row["output_relpath"] / "result.json").exists()
            ):
                raise RuntimeError("blur resume row identity/result state changed")
        return event

    def _record_interrupted_attempt(self, row: dict, event_row: dict) -> None:
        _path, attempts = self._attempts(row)
        previous = int(event_row["attempt_index"])
        event_sha = self.resume_event["recovery_event_sha256"]
        if len(attempts) == previous + 1:
            if attempts[-1].get("recovery_event_sha256") != event_sha:
                raise RuntimeError("existing interrupted-attempt audit differs")
            return
        if len(attempts) != previous:
            raise RuntimeError("interrupted attempt ledger is not append-only")
        log_path = Path(event_row["process_log"])
        if not log_path.is_file():
            raise RuntimeError("interrupted attempt process log is absent")
        self._write_attempt(
            row,
            {
                "attempt_index": previous,
                "status": (
                    "indiscriminate_supervisor_abort_post_observation"
                    if event_row["initial_observation_accepted"]
                    else "indiscriminate_supervisor_abort_pre_observation"
                ),
                "returncode": None,
                "process_log": str(log_path),
                "process_log_sha256": screen.base.sha256_file(log_path),
                "initial_observation_accepted": event_row[
                    "initial_observation_accepted"
                ],
                "scientific_result_written": False,
                "recovery_event_sha256": event_sha,
                "finished_utc": self.resume_event["frozen_utc"],
            },
        )

    def _validate_compacted_launch_smoke(self) -> dict:
        smoke_contract = self.contract["launch_smoke"]
        artifact_path = self.output_root / smoke_contract["required_artifact"]
        artifact = json.loads(artifact_path.read_text())
        payload = dict(artifact)
        observed = payload.pop("launch_smoke_sha256", None)
        if observed != screen.canonical_hash(payload):
            raise RuntimeError("launch-smoke self-hash mismatch")
        expected = {
            "dispatch_contract_sha256": self.contract["dispatch_contract_sha256"],
            "scientific_schedule_sha256": self.schedule["schedule_sha256"],
            "passed": True,
            "schedule_index": smoke_contract["schedule_index"],
            "rollout_id": smoke_contract["rollout_id"],
            "schedule_row_sha256": smoke_contract["schedule_row_sha256"],
            "driver_status": "complete",
        }
        for key, value in expected.items():
            if artifact.get(key) != value:
                raise RuntimeError(f"launch smoke {key} mismatch")
        row = self.schedule["rows"][artifact["schedule_index"]]
        row_dir = self._row_dir(row)
        result_path = row_dir / "result.json"
        screen.base._validate_scientific_result(result_path, row)
        screen.base._validate_boundary(
            row_dir / "initial_observation_accepted.json", row
        )
        current_result_sha = screen.base.sha256_file(result_path)
        if current_result_sha != artifact["scientific_result_sha256"]:
            result = json.loads(result_path.read_text())
            storage = result.get("storage_compaction", {})
            archive_path = row_dir / "storage_archive.json"
            archive = json.loads(archive_path.read_text())
            archive_payload = dict(archive)
            archive_sha = archive_payload.pop("storage_archive_sha256", None)
            if (
                storage.get("original_result", {}).get("sha256")
                != artifact["scientific_result_sha256"]
                or archive_sha != screen.canonical_hash(archive_payload)
                or archive.get("status") != "complete"
                or archive.get("compact_result_sha256") != current_result_sha
            ):
                raise RuntimeError("compacted launch-smoke hash chain is invalid")
        if (
            screen.base.sha256_file(row_dir / "driver_result.json")
            != artifact["driver_result_sha256"]
        ):
            raise RuntimeError("launch-smoke driver result changed")
        return artifact

    def _prepare(self) -> None:
        if self.mode == "smoke":
            if self.resume_event is not None:
                raise RuntimeError("smoke cannot use a resume recovery event")
            return super()._prepare()
        self._validate_compacted_launch_smoke()
        proof = self.output_root / self.contract["detachment_proof"][
            "required_artifact"
        ]
        if not proof.exists():
            raise RuntimeError("full dispatch requires detachment proof")
        recovery_rows = {
            row["rollout_id"]: row for row in (self.resume_event or {}).get("rows", [])
        }
        for row in self.schedule["rows"]:
            if row["rollout_id"] in self.completed_ids:
                continue
            if self._valid_result(row):
                row_dir = self._row_dir(row)
                if not (row_dir / "driver_result.json").exists():
                    screen.base.write_json_atomic(
                        row_dir / "driver_result.json",
                        {
                            "status": "complete",
                            "rollout_id": row["rollout_id"],
                            "schedule_row_sha256": row["schedule_row_sha256"],
                            "resume_action": "reconciled_existing_result",
                        },
                    )
                self.completions.append(
                    {
                        "schedule_index": row["schedule_index"],
                        "rollout_id": row["rollout_id"],
                        "schedule_row_sha256": row["schedule_row_sha256"],
                        "arm": row["arm"],
                        "completed_utc": screen.utc_now(),
                        "result_sha256": screen.base.sha256_file(
                            row_dir / "result.json"
                        ),
                        "driver_result_sha256": screen.base.sha256_file(
                            row_dir / "driver_result.json"
                        ),
                    }
                )
                self.completed_ids.add(row["rollout_id"])
                continue
            event_row = recovery_rows.get(row["rollout_id"])
            boundary = self._row_dir(row) / "initial_observation_accepted.json"
            if event_row is not None:
                if boundary.exists() != bool(event_row["initial_observation_accepted"]):
                    raise RuntimeError("recovery boundary state changed")
                self._record_interrupted_attempt(row, event_row)
                self.authorized_recovery[row["rollout_id"]] = (
                    self.resume_recovery_event
                )
                self.pending.append(row)
                continue
            if boundary.exists():
                raise RuntimeError(
                    "result-free boundary exists outside frozen recovery: "
                    f"{row['rollout_id']}"
                )
            self.pending.append(row)
        if recovery_rows and set(recovery_rows) - {
            row["rollout_id"] for row in self.schedule["rows"]
        }:
            raise RuntimeError("recovery event contains an unknown row")
        self._write_completions()

    def _command(self, row, *, attempt_index: int, recovery_event: Path | None):
        command = screen.base.command_for(
            row,
            manifest_path=self.manifest_path,
            output_dir=self._row_dir(row),
            save_video=False,
        )
        command[0] = str(screen.PYTHON)
        command[1] = str(EVALUATOR)
        command.extend(["--blur-sigma", str(row["blur_sigma"])])
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
        result_path = self._row_dir(active.row) / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if (
                result.get("blur_sigma") != active.row["blur_sigma"]
                or result.get("policy_info", {}).get("blur_sigma")
                != active.row["blur_sigma"]
            ):
                raise RuntimeError(
                    f"row {active.row['schedule_index']} silently dropped blur sigma"
                )
        super()._reconcile_complete(active)
        self.last_finalization = time.monotonic()

    def _watchdog_event(self, kind: str, rows: list[dict], **extra) -> Path:
        index = len(list(self.watchdog_root.glob("event_*.json")))
        event = {
            "schema_version": "pact_blur_sweep_watchdog_event",
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
            "schema_version": "pact_blur_sweep_supervisor_pid",
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
            "schema_version": "pact_blur_sweep_execution",
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
    parser.add_argument("--resume-recovery-event", type=Path)
    args = parser.parse_args()
    return BlurSweepSupervisor(
        schedule_path=args.schedule,
        contract_path=args.dispatch_contract,
        manifest_path=args.manifest,
        output_root=args.output_root,
        mode=args.mode,
        resume_recovery_event=args.resume_recovery_event,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
