from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/calibrate_pact_blur.py"


def load_script():
    spec = importlib.util.spec_from_file_location("blur_calibration", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_uses_real_native_wrist_frame_and_is_monotone() -> None:
    module = load_script()
    document = module.calibrate(module.DEFAULT_SOURCE)
    assert document["source"]["frame_shape"] == [240, 320, 3]
    assert document["source"]["frame_dtype"] == "uint8"
    assert document["candidate_sigmas"] == [0.0, 0.25, 0.5, 1.0, 2.0]
    assert document["monotone_nonincreasing"] is True
    assert document["primitive"]["sigma_zero_same_object_and_bit_identical"] is True
    assert document["primitive"]["proximity_unchanged"] is True
    rows = document["measurements"]
    assert rows[0]["tensor_changed_from_sharp"] is False
    assert all(row["tensor_changed_from_sharp"] for row in rows[1:])
    retained = [row["retained_fraction_of_sharp"] for row in rows]
    assert all(left >= right for left, right in zip(retained, retained[1:]))
