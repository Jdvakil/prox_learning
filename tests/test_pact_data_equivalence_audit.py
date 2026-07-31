from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_pact_act_data_and_zero_support as audit


def _episode(path: Path, *, add_embedding: bool, offset: float = 0.0) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=np.arange(24, dtype=np.float32).reshape(3, 8) + offset)
        observations = handle.create_group("observations")
        images = observations.create_group("images")
        images.create_dataset(
            "wrist_camera", data=np.arange(36, dtype=np.uint8).reshape(3, 2, 2, 3)
        )
        observations.create_dataset("qpos", data=np.ones((3, 9), dtype=np.float32))
        observations.create_dataset("qvel", data=np.zeros((3, 9), dtype=np.float32))
        observations.create_dataset("proximity_positions", data=np.zeros((3, 40, 3), dtype=np.float32))
        observations.create_dataset("proximity_valid", data=np.zeros((3, 40), dtype=bool))
        if add_embedding:
            observations.create_dataset(
                "proximity_embeddings", data=np.ones((3, 40, 32), dtype=np.float32)
            )


def test_core_equivalence_accepts_only_added_embedding(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    _episode(old / "episode_0.hdf5", add_embedding=False)
    _episode(new / "episode_0.hdf5", add_embedding=True)
    report = audit.compare_core_payloads(old, new)
    assert report["passed"] is True
    assert report["only_added_dataset_is_proximity_embeddings"] is True
    assert all(value["equal"] for value in report["core_payload_digests"].values())


def test_core_equivalence_detects_action_difference(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    _episode(old / "episode_0.hdf5", add_embedding=False)
    _episode(new / "episode_0.hdf5", add_embedding=True, offset=1.0)
    report = audit.compare_core_payloads(old, new)
    assert report["passed"] is False
    assert report["core_payload_digests"]["action"]["equal"] is False


def test_token_summary_distinguishes_zero_support(tmp_path):
    directory = tmp_path / "dataset"
    directory.mkdir()
    _episode(directory / "episode_0.hdf5", add_embedding=True)
    old = audit.summarize_tokens(
        directory,
        [0],
        "observations/proximity_positions",
        "observations/proximity_valid",
    )
    new = audit.summarize_tokens(
        directory, [0], "observations/proximity_embeddings"
    )
    assert old["exact_zero_vector_count"] == 120
    assert old["validity"]["invalid_exact_zero_count"] == 120
    assert new["exact_zero_vector_count"] == 0
