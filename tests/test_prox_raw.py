"""PACT-raw skin path: no Safety-CVAE weights required.

These tests exist because cvae_v3 was *not* worth keeping as a policy encoder:
trunk/delta lost as PACT features; raw is peak closeness and never ran the net.
They lock the math and sensor order so convert/train/eval stay aligned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ACT = Path(__file__).resolve().parents[1] / "submodules" / "act"
sys.path.insert(0, str(_ACT))

from hybrid_skin_sensors import (  # noqa: E402
    DEAD_PIXEL_M,
    D_MAX,
    HYBRID_SKIN_SENSOR_ORDER,
)
from prox_cvae import ProxCVAEEncoder, featurize_np, feat_dim_for  # noqa: E402


def test_sensor_order_is_forty_and_link5_back_first():
    names = list(HYBRID_SKIN_SENSOR_ORDER)
    assert len(names) == 40
    assert len(set(names)) == 40
    i_back = names.index("link5_back_sensor_0")
    i_front = names.index("link5_front_sensor_0")
    assert i_back < i_front


def test_featurize_dead_pixel_and_range():
    prox = np.full((2, 40, 8, 8), 0.25, dtype=np.float32)
    prox[0, 0, 0, 0] = 0.001  # dead
    prox[1, 1, 0, 0] = 10.0  # far -> closeness 0
    x = featurize_np(prox)
    assert x.shape == (2, 40 * 64)
    assert x[0, 0] == 0.0
    mid = 1.0 - 0.25 / D_MAX
    assert x[0, 1] == pytest.approx(mid)
    assert x[1, 64] == 0.0  # sensor 1 peak far
    assert DEAD_PIXEL_M == 0.005
    assert D_MAX == 0.5


def test_raw_encoder_needs_no_ckpt():
    enc = ProxCVAEEncoder(ckpt_dir=None, feature="raw", device="cpu", layout="per_sensor")
    assert enc.model is None
    assert enc.n_act_sensors == 40
    assert enc.act_feat_dim == 1
    prox = torch.full((3, 40, 8, 8), 0.1)
    prox[:, 5, 2, 2] = 0.05  # closer on sensor 5
    out = enc(prox)
    assert out.shape == (3, 40, 1)
    # peak closeness of 0.05 m = 1 - 0.05/0.5 = 0.9
    assert float(out[0, 5, 0]) == pytest.approx(1.0 - 0.05 / D_MAX)
    # ambient 0.1 m
    assert float(out[0, 0, 0]) == pytest.approx(1.0 - 0.1 / D_MAX)


def test_feat_dim_raw_no_ckpt():
    assert feat_dim_for(None, "raw") == 40


def test_trunk_without_weights_exits():
    with pytest.raises(SystemExit):
        ProxCVAEEncoder(ckpt_dir=None, feature="trunk", device="cpu")
