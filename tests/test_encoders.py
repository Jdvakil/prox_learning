"""Skin encoder package: peak closeness + surface geometry, same public API."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "submodules" / "act"))

from encoders import (  # noqa: E402
    CAUSAL_FRAMES,
    D_MAX,
    MAX_SURFACE_RANGE_M,
    PeakClosenessEncoder,
    ProxCVAEEncoder,
    SurfaceGeometryEncoder,
    causal_sensor_window,
    feat_dim_for,
    featurize_np,
    list_encoders,
    load_encoder,
    nearest_surface_target,
    to_causal_closeness,
)
from hybrid_skin_sensors import DEAD_PIXEL_M, HYBRID_SKIN_SENSOR_ORDER  # noqa: E402


def test_list_encoders_has_both_functions():
    names = list_encoders()
    assert "peak_closeness" in names
    assert "nearest_surface" in names
    assert "surface_embedding" in names


def test_resolve_aliases():
    raw = load_encoder("raw", device="cpu")
    assert isinstance(raw, PeakClosenessEncoder)
    assert raw.name == "peak_closeness"
    geom = load_encoder("xyz", device="cpu")
    assert isinstance(geom, SurfaceGeometryEncoder)
    assert geom.kind == "xyz"


def test_unknown_encoder_name():
    with pytest.raises(ValueError, match="unknown encoder"):
        load_encoder("not_a_real_encoder")


def test_peak_closeness_matches_legacy_raw_math():
    enc = load_encoder("peak_closeness", device="cpu")
    assert enc.n_act_sensors == 40
    assert enc.act_feat_dim == 1
    prox = torch.full((3, 40, 8, 8), 0.1)
    prox[:, 5, 2, 2] = 0.05
    out = enc.policy_features(prox)
    assert out.shape == (3, 40, 1)
    assert float(out[0, 5, 0]) == pytest.approx(1.0 - 0.05 / D_MAX)
    assert float(out[0, 0, 0]) == pytest.approx(1.0 - 0.1 / D_MAX)


def test_peak_closeness_unbatched_numpy():
    enc = load_encoder("peak_closeness", device="cpu")
    prox = np.full((40, 8, 8), 0.2, dtype=np.float32)
    out = enc.policy_features(prox)
    assert out.shape == (40, 1)


def test_prox_cvae_alias_is_same_class():
    assert ProxCVAEEncoder is PeakClosenessEncoder


def test_featurize_dead_pixel_and_range():
    prox = np.full((2, 40, 8, 8), 0.25, dtype=np.float32)
    prox[0, 0, 0, 0] = 0.001
    prox[1, 1, 0, 0] = 10.0
    x = featurize_np(prox)
    assert x.shape == (2, 40 * 64)
    assert x[0, 0] == 0.0
    mid = 1.0 - 0.25 / D_MAX
    assert x[0, 1] == pytest.approx(mid)
    assert x[1, 64] == 0.0
    assert DEAD_PIXEL_M == 0.005
    assert D_MAX == 0.5


def test_feat_dim_raw_no_ckpt():
    assert feat_dim_for(None, "raw") == 40


def test_cvae_trunk_without_weights_exits():
    with pytest.raises(SystemExit):
        load_encoder("cvae_trunk", device="cpu")


def test_nearest_surface_target_picks_closest_in_range_pixel():
    depth = np.full((8, 8), 10.0, dtype=np.float32)
    depth[2, 5] = 0.10
    xyz, valid = nearest_surface_target(depth)
    assert valid is True
    assert xyz[2] == pytest.approx(0.10)
    fy = 0.5 * 8.0 / math.tan(math.radians(45.0) / 2.0)
    expected_x = (5.0 + 0.5 - 4.0) * 0.10 / fy
    expected_y = (2.0 + 0.5 - 4.0) * 0.10 / fy
    assert xyz[0] == pytest.approx(expected_x)
    assert xyz[1] == pytest.approx(expected_y)


def test_nearest_surface_target_rejects_beyond_20cm():
    depth = np.full((8, 8), 0.25, dtype=np.float32)
    xyz, valid = nearest_surface_target(depth)
    assert valid is False
    assert np.array_equal(xyz, np.zeros(3, dtype=np.float32))


def test_causal_sensor_window_left_pads():
    episode = np.zeros((3, 2, 4, 8, 8), dtype=np.float32)
    episode[0, 1] = 0.10
    episode[2, 1] = 0.05
    window = causal_sensor_window(episode, timestep=0, sensor_index=1)
    assert window.shape == (CAUSAL_FRAMES, 8, 8)
    # t=0 pads with the first step; closeness of 0.10 m at 20 cm cap.
    expected = 1.0 - 0.10 / MAX_SURFACE_RANGE_M
    assert window[0, 0, 0] == pytest.approx(expected)
    assert window[-1, 0, 0] == pytest.approx(expected)


def test_to_causal_closeness_tiles_pact_snapshot():
    skin = torch.full((2, 40, 8, 8), 0.10)
    windows, squeeze_batch, squeeze_sensor = to_causal_closeness(skin, unit="metres")
    assert windows.shape == (2, 40, CAUSAL_FRAMES, 8, 8)
    assert squeeze_batch is False
    assert squeeze_sensor is False
    assert float(windows[0, 0, 0, 0, 0]) == pytest.approx(1.0 - 0.10 / MAX_SURFACE_RANGE_M)
    assert torch.equal(windows[:, :, 0], windows[:, :, -1])


def test_surface_xyz_policy_features_shape():
    enc = load_encoder("nearest_surface", device="cpu")
    prox = torch.full((1, 2, 8, 8), 0.10)
    xyz = enc.policy_features(prox)
    assert xyz.shape == (1, 2, 3)
    assert enc.n_sensors == len(HYBRID_SKIN_SENSOR_ORDER)
    assert enc.act_feat_dim == 3


def test_surface_embedding_policy_features_shape():
    enc = load_encoder("surface_embedding", device="cpu")
    prox = torch.full((1, 2, 8, 8), 0.10)
    z = enc.policy_features(prox)
    assert z.shape == (1, 2, 32)


def test_encode_episode_uses_causal_history_shape():
    enc = SurfaceGeometryEncoder(kind="xyz", device="cpu")
    episode = np.full((2, 3, 4, 8, 8), 0.12, dtype=np.float32)
    out = enc.encode_episode(episode)
    assert out.shape == (2, 3, 3)


def test_both_encoders_eat_the_same_pact_tensor():
    prox = torch.full((1, 40, 8, 8), 0.08)
    raw = load_encoder("peak_closeness", device="cpu").policy_features(prox)
    xyz = load_encoder("nearest_surface", device="cpu").policy_features(prox)
    assert raw.shape[:2] == xyz.shape[:2] == (1, 40)
