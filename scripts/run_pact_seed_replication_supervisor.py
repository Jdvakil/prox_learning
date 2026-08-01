#!/usr/bin/env python3
"""Durable eight-worker supervisor for the frozen seed replication."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import run_pact_frontend_screen_supervisor as screen

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "submodules/act/eval_pact_seed_replication_row.py"


class SeedReplicationSupervisor(screen.ScreenSupervisor):
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
            raise RuntimeError("seed-replication schedule self-hash mismatch")
        if (
            self.schedule.get("schema_version") != "pact_seed_replication_schedule_v1"
            or self.schedule.get("workers") != screen.WORKERS
            or self.schedule.get("rollouts") != 120
            or len(self.schedule.get("rows", [])) != 120
            or {row["arm"] for row in self.schedule["rows"]} != {"ACT", "PACT", "PACT_PERMUTED"}
        ):
            raise RuntimeError("seed-replication schedule design mismatch")
        self.contract = screen.base.load_dispatch_contract(
            self.contract_path,
            self.schedule,
            manifest_path=self.manifest_path,
            output_root=self.output_root,
        )
        if self.contract.get("schema_version") != "pact_seed_replication_dispatch_v1":
            raise RuntimeError("wrong seed-replication dispatch contract")
        for label, record in self.contract["frozen_inputs"]["runtime"].items():
            if screen.base.sha256_file(Path(record["path"])) != record["sha256"]:
                raise RuntimeError(f"frozen seed-replication runtime changed: {label}")
        active_protected = screen.protected_eval_processes()
        if active_protected:
            raise RuntimeError(f"protected shared evaluation is active: {active_protected}")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.pid_path = self.output_root / "supervisor_pid.json"
        self.state_path = self.output_root / "supervisor_state.json"
        self.heartbeat_path = self.output_root / "heartbeat.json"
        self.completion_path = self.output_root / "completion_ledger.json"
        self.recovery_root = self.output_root / "recovery_events"
        self.recovery_root.mkdir(exist_ok=True)
        self.active = {}
        self.pending = collections.deque()
        self.completions = self._load_completions()
        self.completed_ids = {item["rollout_id"] for item in self.completions}
        self.authorized_recovery = {}
        self.abort_reason = None
        self.started_utc = screen.utc_now()
        self.full_dispatch_started_utc = self.started_utc if mode == "full" else None
        self.last_heartbeat = 0.0
        self._install_signals()

    def _command(self, row, *, attempt_index: int, recovery_event: Path | None):
        command = screen.base.command_for(
            row,
            manifest_path=self.manifest_path,
            output_dir=self._row_dir(row),
            save_video=True,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch-contract", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    args = parser.parse_args()
    return SeedReplicationSupervisor(
        schedule_path=args.schedule,
        contract_path=args.dispatch_contract,
        manifest_path=args.manifest,
        output_root=args.output_root,
        mode=args.mode,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
