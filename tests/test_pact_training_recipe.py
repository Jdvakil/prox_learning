from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "train_pact_policies", ROOT / "scripts" / "train_pact_policies.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


trainer = _load()


def _command(arm):
    return trainer.training_command(
        arm=arm,
        seed=3101,
        dataset_dir=Path("/data"),
        split_manifest=Path("/split.json"),
        dataset_manifest=Path("/dataset.json"),
        output_dir=Path("/ckpt"),
        split_sha256="s",
        dataset_tree_sha256="t",
        episode_horizon=150,
        surface_encoder_sha256="e" * 64,
    )


def test_act_and_pact_recipes_differ_only_in_modality_flags():
    act = _command("ACT")
    pact = _command("PACT")
    assert "--use_proximity" not in act
    assert "--n_proximity_sensors" not in act
    assert "--use_proximity" in pact
    assert pact[pact.index("--n_proximity_sensors") + 1] == "40"
    for flag, value in (
        ("--enc_layers", "7"),
        ("--dec_layers", "7"),
        ("--batch_size", "8"),
        ("--num_epochs", "2000"),
        ("--chunk_size", "100"),
        ("--hidden_dim", "512"),
    ):
        assert act[act.index(flag) + 1] == value
        assert pact[pact.index(flag) + 1] == value
