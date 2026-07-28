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
