from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "train_pact_surface_encoder",
        ROOT / "scripts" / "train_pact_surface_encoder.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


trainer = _load()


def _episode(path: Path):
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("observations")
        names = [f"sensor_{index}" for index in range(40)]
        obs.create_dataset(
            "proximity_sensor_names",
            data=np.asarray(names, dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        values = np.full((2, 40, 4, 8, 8), 2.0, dtype=np.float32)
        values[1, 7, -1, 2, 3] = 0.10
        obs.create_dataset("proximity", data=values)


def test_surface_dataset_balances_negatives_and_builds_causal_sample(tmp_path):
    _episode(tmp_path / "episode_0.hdf5")
    dataset = trainer.SurfaceSampleDataset(
        tmp_path, seed=11, negative_to_positive_ratio=1.0
    )
    assert dataset.positive_count == 1
    assert len(dataset) == 2
    valid_samples = [dataset[index] for index in range(len(dataset))]
    valid = [float(sample[2]) for sample in valid_samples]
    assert sorted(valid) == [0.0, 1.0]
    for window, point, flag in valid_samples:
        assert tuple(window.shape) == (32, 8, 8)
        assert tuple(point.shape) == (3,)
