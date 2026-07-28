from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules" / "act"))

from surface_proximity_encoder import (  # noqa: E402
    CAUSAL_FRAMES,
    SurfaceProximityEncoder,
    causal_sensor_window,
    depth_to_closeness,
    nearest_surface_target,
    parameter_count,
)


def test_nearest_target_excludes_far_plane_and_missing_values():
    depth = np.full((8, 8), 2.5, dtype=np.float32)
    depth[1, 2] = np.nan
    point, valid = nearest_surface_target(depth)
    assert not valid
    np.testing.assert_array_equal(point, np.zeros(3, dtype=np.float32))

    depth[3, 4] = 0.12
    point, valid = nearest_surface_target(depth)
    assert valid
    assert point[2] == pytest.approx(0.12)


def test_closeness_is_zero_outside_twenty_centimeters():
    depth = np.asarray([0.05, 0.20, 0.21, 2.5, np.nan], dtype=np.float32)
    closeness = depth_to_closeness(depth)
    np.testing.assert_allclose(closeness[:2], [0.75, 0.0], atol=1e-6)
    np.testing.assert_array_equal(closeness[2:], np.zeros(3, dtype=np.float32))


def test_causal_window_left_pads_without_future_frames():
    proximity = np.full((3, 2, 4, 8, 8), 0.3, dtype=np.float32)
    proximity[0, 1] = 0.10
    proximity[1, 1] = 0.15
    window = causal_sensor_window(proximity, timestep=1, sensor_index=1)
    assert window.shape == (CAUSAL_FRAMES, 8, 8)
    assert np.all(window[:28] == pytest.approx(0.5))
    assert np.all(window[28:] == pytest.approx(0.25))


def test_model_shape_and_parameter_budget():
    model = SurfaceProximityEncoder()
    xyz, logits = model(torch.zeros(2, CAUSAL_FRAMES, 8, 8))
    assert xyz.shape == (2, 3)
    assert logits.shape == (2,)
    assert 800_000 <= parameter_count(model) <= 840_000
