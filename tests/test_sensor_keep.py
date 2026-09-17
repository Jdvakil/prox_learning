"""Per-link Poisson sensor keep: p=0/1 exact, clip, fixed mask, fill=D_MAX."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_ACT = _REPO / "submodules" / "act"
_SCRIPTS = _REPO / "scripts"
sys.path.insert(0, str(_ACT))
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_REPO))

from hybrid_skin_sensors import D_MAX, HYBRID_SKIN_SENSOR_ORDER  # noqa: E402
from prox_cvae import featurize_np  # noqa: E402
from sensor_keep import (  # noqa: E402
    apply_mask,
    compose_first_frame_rgbd_sensors,
    group_indices,
    mask_record,
    mask_rng,
    poisson_keep_mask,
    summarize_keep_records,
)
import pact_eval_sensor_keep  # noqa: E402


ORDER = list(HYBRID_SKIN_SENSOR_ORDER)


def test_groups_split_link5_faces():
    groups = group_indices(ORDER)
    assert list(groups) == [
        "link1",
        "link2",
        "link3",
        "link4",
        "link5_back",
        "link5_front",
        "link6",
    ]
    assert len(groups["link1"]) == 7
    assert len(groups["link5_back"]) == 6
    assert len(groups["link5_front"]) == 4
    assert sum(len(v) for v in groups.values()) == 40


def test_p0_drops_all_p1_keeps_all():
    rng = np.random.default_rng(0)
    z = poisson_keep_mask(ORDER, 0.0, rng)
    a = poisson_keep_mask(ORDER, 1.0, rng)
    assert z.shape == (40,)
    assert not z.any()
    assert a.all()
    # p=1 is exact: even a stream of poisson draws cannot drop
    rng2 = np.random.default_rng(999)
    for _ in range(50):
        assert poisson_keep_mask(ORDER, 1.0, rng2).all()


def test_poisson_mean_near_lambda_on_large_n():
    """n=40 as one fake link is not our grouping; use many trials on link1 n=7."""
    idxs = group_indices(ORDER)["link1"]
    n_l = len(idxs)
    p = 0.5
    kept = []
    rng = np.random.default_rng(2026)
    for _ in range(4000):
        mask = poisson_keep_mask(ORDER, p, rng)
        kept.append(int(mask[idxs].sum()))
    mean = float(np.mean(kept))
    # Clip pulls mean slightly below lambda=3.5; Poisson(3.5) rarely hits n=7.
    assert 3.2 < mean < 3.55
    assert min(kept) >= 0
    assert max(kept) <= n_l


def test_clip_cannot_exceed_n():
    rng = np.random.default_rng(1)
    for _ in range(200):
        mask = poisson_keep_mask(ORDER, 0.9, rng)
        groups = group_indices(ORDER)
        for idxs in groups.values():
            assert int(mask[idxs].sum()) <= len(idxs)


def test_reproducible_and_episode_varies():
    a = poisson_keep_mask(ORDER, 0.5, mask_rng(2026, 0, fixed=False))
    b = poisson_keep_mask(ORDER, 0.5, mask_rng(2026, 0, fixed=False))
    c = poisson_keep_mask(ORDER, 0.5, mask_rng(2026, 1, fixed=False))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sensor_mask_fixed_ignores_episode_idx():
    a = poisson_keep_mask(ORDER, 0.5, mask_rng(7, 0, fixed=True))
    b = poisson_keep_mask(ORDER, 0.5, mask_rng(7, 99, fixed=True))
    assert np.array_equal(a, b)
    vary = poisson_keep_mask(ORDER, 0.5, mask_rng(7, 99, fixed=False))
    assert not np.array_equal(a, vary)


def test_apply_mask_zeros_raw_closeness():
    prox = np.full((40, 8, 8), 0.08, dtype=np.float32)
    mask = np.zeros(40, dtype=bool)
    mask[:3] = True
    out = apply_mask(prox, mask, fill=D_MAX)
    assert out.shape == (40, 8, 8)
    assert np.allclose(out[:3], 0.08)
    assert np.allclose(out[3:], D_MAX)
    feat = featurize_np(out[None])[0].reshape(40, 64)
    kept_peak = feat[:3].max(axis=1)
    drop_peak = feat[3:].max(axis=1)
    assert np.all(kept_peak == pytest.approx(1.0 - 0.08 / D_MAX))
    assert np.all(drop_peak == pytest.approx(0.0))


def test_apply_mask_refuses_fill_zero():
    prox = np.full((40, 8, 8), 0.2, dtype=np.float32)
    mask = np.ones(40, dtype=bool)
    mask[0] = False
    with pytest.raises(ValueError, match="fill must be > 0"):
        apply_mask(prox, mask, fill=0.0)


def test_mask_record_and_summary_mean_sd():
    rng = np.random.default_rng(0)
    recs = []
    for i in range(30):
        mask = poisson_keep_mask(ORDER, 0.5, rng)
        recs.append(
            mask_record(
                ORDER,
                mask,
                keep_frac=0.5,
                mask_seed=0,
                mask_fixed=False,
                episode_idx=i,
            )
        )
    summary = summarize_keep_records(recs)
    assert summary is not None
    assert summary["n_episodes"] == 30
    assert 0.3 < summary["realized_frac_mean"] < 0.6
    assert summary["realized_frac_sd"] >= 0.0
    assert "link5_front" in summary["per_link"]
    assert summary["per_link"]["link5_front"]["n"] == 4


def test_protocol_fields_and_resume_defaults():
    class A:
        sensor_keep_frac = 0.5
        sensor_mask_fixed = True

    fields = pact_eval_sensor_keep.protocol_fields(A())
    assert fields == {"sensor_keep_frac": 0.5, "sensor_mask_fixed": True}
    assert pact_eval_sensor_keep.keep_fields_mismatch({}, fields)
    assert pact_eval_sensor_keep.keep_fields_mismatch(
        {"sensor_keep_frac": 0.5, "sensor_mask_fixed": True}, fields
    ) is None
    # old H-B summaries lack keys -> treat as 100% / unfixed
    assert (
        pact_eval_sensor_keep.keep_fields_mismatch(
            {}, {"sensor_keep_frac": 1.0, "sensor_mask_fixed": False}
        )
        is None
    )


def test_compose_first_frame_grays_dropped():
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    depth = np.full((48, 64), 0.4, dtype=np.float32)
    prox = np.full((40, 8, 8), 0.1, dtype=np.float32)
    mask = np.zeros(40, dtype=bool)
    mask[0] = True
    img = compose_first_frame_rgbd_sensors(
        rgb_by_cam={"wrist_camera": rgb},
        depth_by_cam={"wrist_camera": depth},
        cameras=["wrist_camera"],
        prox=prox,
        names=ORDER,
        mask=mask,
        caption="p=0 test",
        panel_h=48,
        mosaic_h=120,
    )
    assert img.dtype == np.uint8
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.shape[0] > 48 + 120
    gray = np.array([80, 80, 80], dtype=np.uint8)
    assert (img == gray).all(axis=2).any()


def test_unwrap_obs_list_like_molmo_reset():
    inner = {"wrist_camera": np.zeros((8, 8, 3), np.uint8)}
    got = pact_eval_sensor_keep.unwrap_obs([inner])
    assert got is inner
    assert pact_eval_sensor_keep.unwrap_obs(inner) is inner


def test_mixin_begin_and_mask():
    class P(pact_eval_sensor_keep.SensorKeepMixin):
        def __init__(self):
            self.pc = type("PC", (), {"use_proximity": True})()
            self._prox_encoder = type("E", (), {"sensor_order": ORDER})()

    pol = P()
    pol.configure_sensor_keep(keep_frac=0.0, mask_seed=1, mask_fixed=False)
    info = pol.begin_sensor_keep(0)
    assert info["sensor_keep_count"] == 0
    prox = np.full((40, 8, 8), 0.12, dtype=np.float32)
    out = pol.mask_proximity(prox)
    assert np.allclose(out, D_MAX)
