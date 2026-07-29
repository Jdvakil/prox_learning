from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import launch_pact_r2_detached as launcher  # noqa: E402
import measure_pact_r2_throughput as throughput  # noqa: E402
import pact_r2_contract as r2_contract  # noqa: E402
import run_pact_r2_supervisor as supervisor  # noqa: E402


def _row(index: int) -> dict:
    return {
        "schedule_index": index,
        "rollout_id": f"rollout-{index}",
        "schedule_row_sha256": f"row-sha-{index}",
        "output_relpath": f"rows/{index:03d}",
    }


class _DeadProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode = -9

    def poll(self):
        return self.returncode


def _active(tmp_path: Path, row: dict, pid: int):
    row_dir = tmp_path / row["output_relpath"]
    row_dir.mkdir(parents=True)
    log_path = row_dir / "process_attempt_000.log"
    log_stream = log_path.open("wb")
    log_stream.write(b"evaluator killed\n")
    return supervisor.ActiveAttempt(
        row=row,
        process=_DeadProcess(pid),
        log_stream=log_stream,
        log_path=log_path,
        attempt_index=0,
        started_utc="2026-07-30T00:00:00Z",
        recovery_event_path=None,
    )


def test_group_recovery_requeues_every_active_row(tmp_path):
    rows = [_row(0), _row(1), _row(2)]
    instance = supervisor.R2Supervisor.__new__(supervisor.R2Supervisor)
    instance.output_root = tmp_path
    instance.recovery_root = tmp_path / "recovery_events"
    instance.recovery_root.mkdir()
    instance.schedule = {"schedule_sha256": "schedule", "rows": rows}
    instance.contract = {"dispatch_contract_sha256": "contract"}
    cohort = [
        _active(tmp_path, row, 1000 + index)
        for index, row in enumerate(rows)
    ]
    instance.active = {
        active.row["rollout_id"]: active for active in cohort
    }
    instance.authorized_recovery = {}
    instance.pending = collections.deque()
    instance._write_state = lambda status: None

    instance._group_recover(
        cohort, reason="live_all_active_evaluators_exited_within_five_seconds"
    )

    assert not instance.active
    assert [row["rollout_id"] for row in instance.pending] == [
        row["rollout_id"] for row in rows
    ]
    assert set(instance.authorized_recovery) == {
        row["rollout_id"] for row in rows
    }
    event_path = next((tmp_path / "recovery_events").glob("event_*.json"))
    event = json.loads(event_path.read_text())
    payload = dict(event)
    observed = payload.pop("recovery_event_sha256")
    assert observed == supervisor.canonical_hash(payload)
    assert event["active_cohort_size"] == 3
    assert [item["rollout_id"] for item in event["rows"]] == [
        row["rollout_id"] for row in rows
    ]
    assert event["result_absent_for_all"] is True
    for row in rows:
        ledger = json.loads(
            (
                tmp_path
                / row["output_relpath"]
                / "attempt_ledger.json"
            ).read_text()
        )
        assert len(ledger["attempts"]) == 1
        assert ledger["attempts"][0]["status"].startswith(
            "group_termination_"
        )


def test_recovery_event_only_authorizes_recorded_next_attempt(tmp_path):
    event = {
        "schema_version": "pact_r2_group_recovery_v1",
        "event_index": 0,
        "schedule_sha256": "schedule",
        "dispatch_contract_sha256": "contract",
        "reason": "test",
        "qualifying_indiscriminate_termination": True,
        "all_inflight_rows_rerun": True,
        "result_absent_for_all": True,
        "active_cohort_size": 2,
        "rows": [
            {
                "rollout_id": "row-a",
                "schedule_row_sha256": "sha-a",
                "attempt_index": 0,
                "result_present": False,
            },
            {
                "rollout_id": "row-b",
                "schedule_row_sha256": "sha-b",
                "attempt_index": 0,
                "result_present": False,
            },
        ],
    }
    event["recovery_event_sha256"] = r2_contract.sha256_payload(event)
    path = tmp_path / "event.json"
    path.write_text(json.dumps(event))
    loaded = r2_contract.validate_recovery_event(
        path,
        rollout_id="row-a",
        schedule_row_sha256="sha-a",
        attempt_index=1,
    )
    assert loaded["active_cohort_size"] == 2
    with pytest.raises(
        r2_contract.PactR2ContractError,
        match="immediately preceding",
    ):
        r2_contract.validate_recovery_event(
            path,
            rollout_id="row-a",
            schedule_row_sha256="sha-a",
            attempt_index=2,
        )


def test_launch_reservation_is_durable_before_evaluator_spawn(
    tmp_path, monkeypatch
):
    row = _row(0)
    instance = supervisor.R2Supervisor.__new__(supervisor.R2Supervisor)
    instance.output_root = tmp_path
    instance.state_path = tmp_path / "supervisor_state.json"
    instance.schedule = {"schedule_sha256": "schedule", "rows": [row]}
    instance.contract = {"dispatch_contract_sha256": "contract"}
    instance.mode = "smoke"
    instance.started_utc = "2026-07-30T00:00:00Z"
    instance.full_dispatch_started_utc = None
    instance.active = {}
    instance.pending = collections.deque()
    instance.completed_ids = set()
    instance.abort_reason = None
    instance.launch_reservation = None
    instance.authorized_recovery = {}
    instance._command = lambda *args, **kwargs: ["/fake/evaluator"]
    observed = {}

    class _LiveProcess:
        pid = 43210

        def poll(self):
            return None

    def fake_popen(*args, **kwargs):
        state = json.loads(instance.state_path.read_text())
        observed["reservation"] = state["launch_reservation"]
        return _LiveProcess()

    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_popen)
    instance._start_row(row)
    assert observed["reservation"]["rollout_id"] == row["rollout_id"]
    assert observed["reservation"]["pid"] is None
    final_state = json.loads(instance.state_path.read_text())
    assert final_state["launch_reservation"] is None
    assert final_state["active_cohort"][0]["pid"] == 43210
    instance.active[row["rollout_id"]].log_stream.close()


def test_detached_launcher_uses_sets_id_and_nohup(tmp_path):
    args = argparse.Namespace(
        schedule=tmp_path / "schedule.json",
        dispatch_contract=tmp_path / "contract.json",
        manifest=tmp_path / "manifest.json",
        mode="smoke",
    )
    command = launcher.build_command(args, tmp_path / "output")
    assert command[:2] == ["/usr/bin/setsid", "/usr/bin/nohup"]
    assert command[2] == "/root/act_retrain_venv/bin/python"
    assert command[-2:] == ["--mode", "smoke"]


def test_throughput_uses_completion_ledger_without_result_files(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    output_root.mkdir()
    schedule = {
        "schedule_sha256": "schedule",
        "workers": 8,
        "rollouts": 960,
    }
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(json.dumps(schedule))
    contract = {
        "schema_version": "pact_r2_dispatch_v1",
        "scientific_schedule": {"schedule_sha256": "schedule"},
        "execution": {"output_root": str(output_root)},
    }
    contract["dispatch_contract_sha256"] = throughput.canonical_hash(contract)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    started = datetime.now(timezone.utc) - timedelta(minutes=21)
    started_text = started.isoformat().replace("+00:00", "Z")
    (output_root / "supervisor_state.json").write_text(
        json.dumps(
            {
                "mode": "full",
                "full_dispatch_started_utc": started_text,
            }
        )
    )

    def stamp(minutes):
        return (started + timedelta(minutes=minutes)).isoformat().replace(
            "+00:00", "Z"
        )

    (output_root / "completion_ledger.json").write_text(
        json.dumps(
            {
                "schedule_sha256": "schedule",
                "completions": [
                    {
                        "schedule_index": 0,
                        "rollout_id": "smoke",
                        "completed_utc": stamp(-1),
                    },
                    {
                        "schedule_index": 1,
                        "rollout_id": "row-1",
                        "completed_utc": stamp(1),
                    },
                    {
                        "schedule_index": 2,
                        "rollout_id": "row-2",
                        "completed_utc": stamp(10),
                    },
                ],
            }
        )
    )
    artifact_path = output_root / "throughput.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_pact_r2_throughput.py",
            "--schedule",
            str(schedule_path),
            "--dispatch-contract",
            str(contract_path),
            "--output-root",
            str(output_root),
            "--output",
            str(artifact_path),
        ],
    )
    assert throughput.main() == 0
    artifact = json.loads(artifact_path.read_text())
    assert artifact["completed_during_window"] == 2
    assert artifact["throughput_rollouts_per_minute"] == pytest.approx(0.1)
    assert artifact["result_files_opened"] == 0
    assert artifact["endpoint_fields_read"] is False
