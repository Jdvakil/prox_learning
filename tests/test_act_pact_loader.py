from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules" / "act"))

from utils import EpisodicDataset


def _dataset(path: Path, encoder_sha: str):
    with h5py.File(path, "w") as handle:
        handle.attrs["sim"] = True
        handle.attrs["pact_surface_encoder_sha256"] = encoder_sha
        handle.create_dataset("action", data=np.zeros((2, 8), dtype=np.float32))
        obs = handle.create_group("observations")
        obs.create_dataset("qpos", data=np.zeros((2, 9), dtype=np.float32))
        obs.create_dataset("qvel", data=np.zeros((2, 9), dtype=np.float32))
        images = obs.create_group("images")
        images.create_dataset(
            "wrist_camera", data=np.zeros((2, 8, 8, 3), dtype=np.uint8)
        )
        positions = np.zeros((2, 40, 3), dtype=np.float32)
        positions[0, 9] = [0.1, 0.0, 0.1]
        obs.create_dataset("proximity_positions", data=positions)


def _stats():
    return {
        "action_mean": np.zeros(8, dtype=np.float32),
        "action_std": np.ones(8, dtype=np.float32),
        "qpos_mean": np.zeros(9, dtype=np.float32),
        "qpos_std": np.ones(9, dtype=np.float32),
    }


def test_pact_loader_returns_frozen_positions(tmp_path, monkeypatch):
    sha = "a" * 64
    _dataset(tmp_path / "episode_0.hdf5", sha)
    monkeypatch.setattr(np.random, "choice", lambda _: 0)
    dataset = EpisodicDataset(
        [0],
        str(tmp_path),
        ["wrist_camera"],
        _stats(),
        100,
        use_proximity=True,
        n_proximity_sensors=40,
        expected_proximity_encoder_sha256=sha,
    )
    sample = dataset[0]
    assert len(sample) == 5
    assert tuple(sample[-1].shape) == (40, 3)
    np.testing.assert_allclose(sample[-1][9].numpy(), [0.1, 0.0, 0.1])


def test_act_loader_remains_four_tuple(tmp_path):
    _dataset(tmp_path / "episode_0.hdf5", "a" * 64)
    dataset = EpisodicDataset(
        [0], str(tmp_path), ["wrist_camera"], _stats(), 100
    )
    assert len(dataset[0]) == 4


def test_pact_loader_rejects_wrong_encoder(tmp_path):
    _dataset(tmp_path / "episode_0.hdf5", "a" * 64)
    with pytest.raises(ValueError, match="surface encoder sha256"):
        EpisodicDataset(
            [0],
            str(tmp_path),
            ["wrist_camera"],
            _stats(),
            100,
            use_proximity=True,
            n_proximity_sensors=40,
            expected_proximity_encoder_sha256="b" * 64,
        )


def test_pact_loader_returns_frozen_32d_embeddings(tmp_path, monkeypatch):
    sha = "c" * 64
    path = tmp_path / "episode_0.hdf5"
    _dataset(path, sha)
    with h5py.File(path, "r+") as handle:
        embeddings = np.zeros((2, 40, 32), dtype=np.float32)
        embeddings[0, 9] = np.arange(32, dtype=np.float32)
        handle["observations"].create_dataset(
            "proximity_embeddings", data=embeddings
        )
    monkeypatch.setattr(np.random, "choice", lambda _: 0)
    dataset = EpisodicDataset(
        [0],
        str(tmp_path),
        ["wrist_camera"],
        _stats(),
        100,
        use_proximity=True,
        n_proximity_sensors=40,
        proximity_feature_dim=32,
        expected_proximity_encoder_sha256=sha,
    )
    sample = dataset[0]
    assert tuple(sample[-1].shape) == (40, 32)
    np.testing.assert_allclose(
        sample[-1][9].numpy(), np.arange(32, dtype=np.float32)
    )
