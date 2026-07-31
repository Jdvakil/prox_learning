from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules" / "act"))

from detr.models.detr_vae import DETRVAE


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


trainer = _load(
    "train_pact_frontend_screen_policy",
    ROOT / "scripts/train_pact_frontend_screen_policy.py",
)
preparer = _load(
    "prepare_pact_embedding_dataset",
    ROOT / "scripts/prepare_pact_embedding_dataset.py",
)


class _Transformer(nn.Module):
    d_model = 512


def test_detr_projection_accepts_frozen_feature_width_without_changing_act():
    pact = DETRVAE(
        None,
        _Transformer(),
        nn.Identity(),
        state_dim=9,
        num_queries=100,
        camera_names=["wrist_camera"],
        action_dim=8,
        n_proximity_sensors=40,
        proximity_feature_dim=32,
    )
    assert pact.input_proj_proximity.in_features == 32
    assert pact.input_proj_proximity.out_features == 512
    assert tuple(pact.additional_pos_embed.weight.shape) == (42, 512)

    act = DETRVAE(
        None,
        _Transformer(),
        nn.Identity(),
        state_dim=9,
        num_queries=100,
        camera_names=["wrist_camera"],
        action_dim=8,
        n_proximity_sensors=0,
        proximity_feature_dim=32,
    )
    assert act.input_proj_proximity is None
    assert tuple(act.additional_pos_embed.weight.shape) == (2, 512)


def test_screen_training_command_is_pact_only_and_frozen_recipe():
    command = trainer.training_command(
        dataset_dir=Path("/data"),
        split_manifest=Path("/split.json"),
        dataset_manifest=Path("/dataset.json"),
        output_dir=Path("/output"),
        split_sha256="split",
        dataset_tree_sha256="tree",
        episode_horizon=195,
        encoder_sha256="e" * 64,
    )
    assert command[command.index("--seed") + 1] == "3101"
    assert command[command.index("--proximity_feature_dim") + 1] == "32"
    assert command[command.index("--n_proximity_sensors") + 1] == "40"
    assert command[command.index("--num_epochs") + 1] == "2000"
    assert command[command.index("--batch_size") + 1] == "8"
    assert command[command.index("--enc_layers") + 1] == "7"
    assert command[command.index("--dec_layers") + 1] == "7"
    assert command[command.index("--ckpt_every") + 1] == "2000"
    assert trainer.ACT_CHECKPOINT_SHA256 == (
        "a5ebbf3d5537315337e17e0f28951de068ce6960974d0f282b77fcfcca672eb1"
    )


def test_screen_encoder_quality_gate_is_frozen_before_policy_training():
    metrics = {
        "mean_euclidean_error_m": 0.032,
        "median_euclidean_error_m": 0.019,
        "within_2cm_rate": 0.51,
        "validity_precision": 0.999,
        "validity_recall": 0.999,
        "active_pixel_reconstruction_mae": 0.10,
    }
    assert trainer.verify_encoder_quality(
        {"heldout_metrics": metrics}
    )["passed"]
    metrics["validity_recall"] = 0.97
    assert not trainer.verify_encoder_quality(
        {"heldout_metrics": metrics}
    )["passed"]


def test_screen_evaluator_replaces_both_legacy_config_and_direct_policy():
    source = (
        ROOT
        / "submodules"
        / "act"
        / "eval_pact_frontend_screen_row.py"
    ).read_text()
    assert (
        "legacy.PactCollisionInferencePolicy = "
        "PactFrontendScreenInferencePolicy"
    ) in source
    assert (
        "legacy.PactCollisionPolicyConfig = PactFrontendScreenPolicyConfig"
    ) in source
    assert "legacy.load_eval_manifest = load_screen_manifest" in source
    assert "legacy.retry_seed_for = screen_retry_seed" in source


def test_rematerialized_raw_semantic_hash_is_string_stable(tmp_path):
    source = tmp_path / "source.hdf5"
    destination = tmp_path / "destination.hdf5"
    with h5py.File(source, "w") as handle:
        handle.attrs["sim"] = True
        handle.attrs["pact_surface_tokens_frozen"] = True
        handle.create_dataset(
            "action", data=np.zeros((1, 8), dtype=np.float32)
        )
        observations = handle.create_group("observations")
        observations.create_dataset(
            "qpos", data=np.zeros((1, 9), dtype=np.float32)
        )
        observations.create_dataset(
            "qvel", data=np.zeros((1, 9), dtype=np.float32)
        )
        observations.create_dataset(
            "proximity",
            data=np.zeros((1, 40, 4, 8, 8), dtype=np.float32),
        )
        observations.create_dataset(
            "proximity_extrinsic_cv",
            data=np.zeros((1, 40, 3, 4), dtype=np.float64),
        )
        observations.create_dataset(
            "proximity_intrinsic_cv",
            data=np.zeros((1, 40, 3, 3), dtype=np.float64),
        )
        observations.create_dataset(
            "proximity_sensor_names",
            data=np.asarray(
                [f"sensor_{index:02d}" for index in range(40)],
                dtype=object,
            ),
            dtype=h5py.string_dtype("utf-8"),
        )
        observations.create_dataset(
            "proximity_positions",
            data=np.zeros((1, 40, 3), dtype=np.float32),
        )
        images = observations.create_group("images")
        images.create_dataset(
            "wrist_camera",
            data=np.zeros((1, 2, 2, 3), dtype=np.uint8),
        )
        provenance = handle.create_group("pact_provenance")
        provenance.create_dataset("row", data=b'{"row": 1}')
        provenance.create_dataset(
            "collection_result", data=b'{"status": "complete"}'
        )
    details = preparer.copy_episode(source, destination)
    assert not details["legacy_surface_tokens_present"]
    source_hash = preparer.semantic_sha256(
        source, ignored_datasets=preparer.LEGACY_TOKEN_DATASETS
    )
    assert preparer.semantic_sha256(destination) == source_hash
    assert preparer.semantic_sha256(destination) == source_hash
