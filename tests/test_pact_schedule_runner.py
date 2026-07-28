from __future__ import annotations

import importlib.util
from pathlib import Path

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
