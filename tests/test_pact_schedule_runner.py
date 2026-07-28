from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_pact_confirmatory_schedule",
        ROOT / "scripts" / "run_pact_confirmatory_schedule.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = _load()


def _act_row():
    return {
        "arm": "ACT",
        "instance_episode_id": "episode",
        "checkpoint_path": "/ckpt/policy_best.ckpt",
        "checkpoint_sha256": "c",
        "checkpoint_seed": 1101,
        "dataset_stats_sha256": "d",
        "schedule_row_sha256": "r",
        "rollout_id": "i",
        "surface_encoder_path": None,
        "surface_encoder_sha256": None,
    }


def test_commands_resolve_paths_before_evaluator_cwd_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    command = runner.command_for(
        _act_row(),
        manifest_path=Path("configs/manifest.json"),
        output_dir=Path("outputs/row"),
        save_video=False,
    )
    assert command[command.index("--manifest") + 1] == str(
        tmp_path / "configs" / "manifest.json"
    )
    assert command[command.index("--output-dir") + 1] == str(
        tmp_path / "outputs" / "row"
    )


def test_commands_use_pact_checkpoint_for_zero_ablation():
    row = {
        "arm": "PACT_ZERO",
        "instance_episode_id": "episode",
        "checkpoint_path": "/ckpt/policy_best.ckpt",
        "checkpoint_sha256": "c",
        "checkpoint_seed": 3101,
        "dataset_stats_sha256": "d",
        "schedule_row_sha256": "r",
        "rollout_id": "i",
        "surface_encoder_path": "/encoder.pt",
        "surface_encoder_sha256": "e",
    }
    command = runner.command_for(
        row,
        manifest_path=Path("/manifest.json"),
        output_dir=Path("/out"),
        save_video=False,
    )
    assert command[command.index("--arm") + 1] == "PACT_ZERO"
    assert command[command.index("--checkpoint-dir") + 1] == "/ckpt"
    assert "--surface-encoder" in command


def _write_boundary_and_result(output_dir: Path, row: dict, *, result: bool):
    boundary = {
        "initial_observation_accepted": True,
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "episode_id": row["instance_episode_id"],
        "checkpoint_sha256": row["checkpoint_sha256"],
    }
    (output_dir / "initial_observation_accepted.json").write_text(
        json.dumps(boundary)
    )
    if result:
        (output_dir / "result.json").write_text(
            json.dumps(
                {
                    **boundary,
                    "status": "complete",
                    "arm": row["arm"],
                }
            )
        )


def test_pre_observation_failure_retries_then_records_scientific_result(
    tmp_path, monkeypatch
):
    row = _act_row()
    row.update(
        {
            "schedule_index": 0,
            "output_relpath": "rows/000_act",
        }
    )
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        kwargs["stdout"].write(f"attempt {calls}\n".encode())
        if calls == 1:
            return SimpleNamespace(returncode=7)
        output_dir = Path(command[command.index("--output-dir") + 1])
        _write_boundary_and_result(output_dir, row, result=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.run_one(
        row,
        manifest_path=str(tmp_path / "manifest.json"),
        output_root=str(tmp_path),
        save_video=False,
    )
    assert calls == 2
    assert result["status"] == "complete"
    assert result["attempt_count"] == 2
    assert result["pre_observation_infrastructure_failures"] == 1
    ledger = json.loads(
        (tmp_path / row["output_relpath"] / "attempt_ledger.json").read_text()
    )
    assert [attempt["status"] for attempt in ledger["attempts"]] == [
        "pre_observation_infrastructure_failure",
        "complete",
    ]


def test_post_boundary_failure_is_terminal_and_not_retried(tmp_path, monkeypatch):
    row = _act_row()
    row.update(
        {
            "schedule_index": 0,
            "output_relpath": "rows/000_act",
        }
    )
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        kwargs["stdout"].write(b"post-boundary crash\n")
        output_dir = Path(command[command.index("--output-dir") + 1])
        _write_boundary_and_result(output_dir, row, result=False)
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.run_one(
        row,
        manifest_path=str(tmp_path / "manifest.json"),
        output_root=str(tmp_path),
        save_video=False,
    )
    assert calls == 1
    assert result["status"] == "post_boundary_failure"
    assert result["attempt_count"] == 1
    assert result["pre_observation_infrastructure_failures"] == 0


def test_full_dispatch_refuses_to_start_without_launch_smoke(tmp_path):
    row = _act_row()
    row.update(
        {
            "schedule_index": 0,
            "output_relpath": "rows/000_act",
        }
    )
    schedule = {
        "schedule_sha256": "schedule",
        "workers": 8,
        "rows": [row],
    }
    contract = {
        "dispatch_contract_sha256": "contract",
        "launch_smoke": {
            "required_artifact": "launch_smoke.json",
            "schedule_index": 0,
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
        },
    }
    with pytest.raises(RuntimeError, match="launch smoke is missing"):
        runner.validate_launch_smoke(
            schedule=schedule,
            contract=contract,
            output_root=tmp_path,
        )
