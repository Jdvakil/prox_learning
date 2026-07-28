from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "convert_pact_collision_to_act",
        ROOT / "scripts" / "convert_pact_collision_to_act.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


converter = _load()


def _source(path: Path, names: list[str], timesteps: int = 3):
    with h5py.File(path, "w") as handle:
        trajectory = handle.create_group("traj_0")
        for sensor_index, name in enumerate(names):
            trajectory.create_dataset(
                f"obs/proximity/{name}",
                data=np.full((timesteps, 4, 8, 8), sensor_index, dtype=np.float32),
            )
            trajectory.create_dataset(
                f"obs/sensor_param/{name}/extrinsic_cv",
                data=np.full((timesteps, 3, 4), sensor_index, dtype=np.float64),
            )
            trajectory.create_dataset(
                f"obs/sensor_param/{name}/intrinsic_cv",
                data=np.full((timesteps, 3, 3), sensor_index, dtype=np.float64),
            )


def test_extract_proximity_preserves_manifest_order(tmp_path):
    names = ["sensor_b", "sensor_a"]
    path = tmp_path / "source.h5"
    _source(path, names)
    with h5py.File(path) as handle:
        proximity, extrinsic, intrinsic = converter.extract_proximity(
            handle["traj_0"], names, 3
        )
    assert proximity.shape == (3, 2, 4, 8, 8)
    assert extrinsic.shape == (3, 2, 3, 4)
    assert intrinsic.shape == (3, 2, 3, 3)
    assert np.all(proximity[:, 0] == 0)
    assert np.all(proximity[:, 1] == 1)


def test_extract_proximity_rejects_missing_sensor(tmp_path):
    path = tmp_path / "source.h5"
    _source(path, ["sensor_a"])
    with h5py.File(path) as handle, pytest.raises(RuntimeError, match="sensor mismatch"):
        converter.extract_proximity(
            handle["traj_0"], ["sensor_a", "sensor_b"], 3
        )


def test_usable_demo_filter_is_exactly_the_primary_endpoint():
    clean = {
        "task_success": True,
        "collision_free_task_success": True,
        "contact_audit": {
            "contact_class_totals": {
                "grasp_target": 7,
                "hazard_bar": 0,
                "other_environment": 0,
            }
        },
    }
    assert converter._is_usable_clean_demo(clean)
    for contact_class in ("hazard_bar", "other_environment"):
        contacting = {
            **clean,
            "collision_free_task_success": False,
            "contact_audit": {
                "contact_class_totals": {
                    "grasp_target": 7,
                    "hazard_bar": int(contact_class == "hazard_bar"),
                    "other_environment": int(contact_class == "other_environment"),
                }
            },
        }
        assert not converter._is_usable_clean_demo(contacting)


def test_usable_demo_filter_rejects_endpoint_mismatch():
    with pytest.raises(RuntimeError, match="disagrees"):
        converter._is_usable_clean_demo(
            {
                "task_success": True,
                "collision_free_task_success": True,
                "contact_audit": {
                    "contact_class_totals": {
                        "hazard_bar": 1,
                        "other_environment": 0,
                    }
                },
            }
        )
