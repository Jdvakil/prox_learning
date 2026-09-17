"""CLI + eval hooks for per-link Poisson sensor keep.

Flags are default-off (keep_frac=1.0) so existing hallway / v1011d / v107
evals do not change. 0% keep is an explicit control, not the default.

Do not send first-frame PNGs to W&B. --save_first_frame is not a protocol field.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sensor_keep import (
    HYBRID_SKIN_SENSOR_ORDER,
    apply_mask,
    compose_first_frame_rgbd_sensors,
    mask_record,
    mask_rng,
    poisson_keep_mask,
    summarize_keep_records,
    write_first_frame_png,
)


def add_cli_flags(parser) -> None:
    parser.add_argument(
        "--sensor_keep_frac",
        type=float,
        default=1.0,
        help="Mean keep rate per link (0=drop all, 1=keep all exact). "
        "Interior values: k~Poisson(p*n_link) clipped to [0, n], then "
        "uniform sample. Nominal p is the column header; log realized "
        "mean+/-sd. 0 is the required no-skin control.",
    )
    parser.add_argument(
        "--sensor_mask_seed",
        type=int,
        default=None,
        help="RNG seed for Poisson keep. Default --seed_base. "
        "Per-episode seed is mask_seed + 1000003*episode_idx unless "
        "--sensor_mask_fixed.",
    )
    parser.add_argument(
        "--sensor_mask_fixed",
        action="store_true",
        help="Sample one mask from --sensor_mask_seed and reuse every "
        "episode (figure). Default off: i.i.d. per episode.",
    )
    parser.add_argument(
        "--save_first_frame",
        action="store_true",
        help="Write t=0 RGB+depth+sensor mosaic PNG under "
        "output_dir/first_frames/. Enables record_depth on policy RGB "
        "cams. Not a protocol field.",
    )


def resolved_mask_seed(args) -> int:
    seed = getattr(args, "sensor_mask_seed", None)
    if seed is None:
        return int(getattr(args, "seed_base", 0) or 0)
    return int(seed)


def protocol_fields(args) -> dict:
    """Resume-identity fields. save_first_frame is intentionally omitted."""
    return {
        "sensor_keep_frac": float(getattr(args, "sensor_keep_frac", 1.0)),
        "sensor_mask_fixed": bool(getattr(args, "sensor_mask_fixed", False)),
    }


def protocol_defaults_for_resume(old: Mapping[str, Any], key: str):
    """Missing keys on old H-B summaries mean 100% keep, unfixed mask."""
    if key == "sensor_keep_frac":
        val = old.get(key)
        return 1.0 if val is None else val
    if key == "sensor_mask_fixed":
        val = old.get(key)
        return False if val is None else val
    return old.get(key)


def keep_fields_mismatch(old: Mapping[str, Any], new: Mapping[str, Any]) -> str | None:
    for key in ("sensor_keep_frac", "sensor_mask_fixed"):
        if key not in new:
            continue
        old_val = protocol_defaults_for_resume(old, key)
        new_val = new[key]
        if key == "sensor_keep_frac":
            try:
                mismatch = abs(float(old_val) - float(new_val)) > 1e-12
            except (TypeError, ValueError):
                mismatch = True
        else:
            mismatch = bool(old_val) != bool(new_val)
        if mismatch:
            return f"protocol {key}={old_val!r} != {new_val!r}"
    return None


def depth_uuid(camera_name: str) -> str:
    return f"{camera_name}_depth"


def unwrap_obs(obs):
    """Molmo reset/step returns a list of dicts. Policy uses obs[0]."""
    cur = obs
    for _ in range(4):
        if isinstance(cur, (list, tuple)):
            if not cur:
                raise SystemExit("[sensor-keep] empty observation list from reset")
            cur = cur[0]
            continue
        break
    if not isinstance(cur, Mapping):
        raise SystemExit(
            f"[sensor-keep] observation is {type(obs).__name__} after unwrap; "
            f"need a dict with camera keys. sample={type(cur).__name__}"
        )
    return cur


def assert_policy_rgbd(obs: Mapping, cameras: Sequence[str]) -> None:
    """RGB uuid is the camera name; depth is {cam}_depth. Additive sensors."""
    for cam in cameras:
        if cam not in obs:
            raise SystemExit(
                f"[sensor-keep] first-frame dump missing RGB obs[{cam!r}]. "
                "Policy reads this key; record_depth must not rename it."
            )
        rgb = np.asarray(obs[cam])
        if rgb.ndim < 2:
            raise SystemExit(
                f"[sensor-keep] obs[{cam!r}] shape {rgb.shape} is not an image"
            )
        key = depth_uuid(cam)
        if key not in obs:
            raise SystemExit(
                f"[sensor-keep] missing obs[{key!r}]. Hallway 977acd6 and "
                "submodule molmospaces both register DepthSensor as "
                f"{{cam}}_depth when record_depth=True. RGB uuid stays {cam}."
            )


class SensorKeepMixin:
    """Mixin for FrozenACTPolicy. Call configure_sensor_keep after prepare_model."""

    def configure_sensor_keep(
        self,
        *,
        keep_frac: float,
        mask_seed: int,
        mask_fixed: bool,
    ) -> None:
        self._sensor_keep_frac = float(keep_frac)
        self._sensor_mask_seed = int(mask_seed)
        self._sensor_mask_fixed = bool(mask_fixed)
        self._sensor_keep_mask = None
        self._sensor_keep_info: dict | None = None
        use_prox = bool(getattr(getattr(self, "pc", None), "use_proximity", False))
        if self._sensor_keep_frac < 1.0 and not use_prox:
            print(
                "[sensor-keep] --sensor_keep_frac "
                f"{self._sensor_keep_frac} but ckpt has no proximity (ACT). "
                "Mask is a no-op.",
                flush=True,
            )

    def begin_sensor_keep(self, episode_idx: int) -> dict:
        order = list(
            getattr(getattr(self, "_prox_encoder", None), "sensor_order", None)
            or HYBRID_SKIN_SENSOR_ORDER
        )
        frac = float(getattr(self, "_sensor_keep_frac", 1.0))
        seed = int(getattr(self, "_sensor_mask_seed", 0))
        fixed = bool(getattr(self, "_sensor_mask_fixed", False))
        rng = mask_rng(seed, episode_idx, fixed=fixed)
        mask = poisson_keep_mask(order, frac, rng)
        self._sensor_keep_mask = mask
        info = mask_record(
            order,
            mask,
            keep_frac=frac,
            mask_seed=seed,
            mask_fixed=fixed,
            episode_idx=int(episode_idx),
        )
        self._sensor_keep_info = info
        use_prox = bool(getattr(getattr(self, "pc", None), "use_proximity", False))
        if use_prox:
            print(
                f"[sensor-keep] p={frac:g} fixed={int(fixed)} ep={episode_idx} "
                f"kept={info['sensor_keep_count']}/{info['sensor_keep_n']} "
                f"per_link={info['sensor_keep_per_link']}",
                flush=True,
            )
        return info

    def mask_proximity(self, frame: np.ndarray) -> np.ndarray:
        mask = getattr(self, "_sensor_keep_mask", None)
        if mask is None:
            return frame
        use_prox = bool(getattr(getattr(self, "pc", None), "use_proximity", False))
        if not use_prox:
            return frame
        return apply_mask(frame, mask)

    def sensor_keep_log(self) -> dict:
        info = getattr(self, "_sensor_keep_info", None)
        return dict(info) if info else {}


def dump_first_frame(
    *,
    obs: Mapping,
    policy,
    cameras: Sequence[str],
    output_dir: Path,
    episode_idx: int,
    seed: int,
    prox_pool: str = "min",
) -> Path:
    """Write output_dir/first_frames/epXXX_seedY_rgbd_sensors.png."""
    from encoders.peak_closeness import stack_obs_proximity

    obs = unwrap_obs(obs)
    cams = list(cameras)
    assert_policy_rgbd(obs, cams)
    order = list(
        getattr(getattr(policy, "_prox_encoder", None), "sensor_order", None)
        or HYBRID_SKIN_SENSOR_ORDER
    )
    prox = None
    try:
        prox = stack_obs_proximity(obs, order, pool=prox_pool)
        prox = policy.mask_proximity(prox) if hasattr(policy, "mask_proximity") else prox
    except KeyError:
        prox = None
    info = getattr(policy, "_sensor_keep_info", None) or {}
    mask = getattr(policy, "_sensor_keep_mask", None)
    caption = (
        f"p={info.get('sensor_keep_frac', 1.0)} "
        f"kept={info.get('sensor_keep_count', '?')}/"
        f"{info.get('sensor_keep_n', 40)} "
        f"fixed={int(bool(info.get('sensor_mask_fixed', False)))} "
        f"ep={episode_idx} seed={seed}"
    )
    rgb_by_cam = {cam: np.asarray(obs[cam]) for cam in cams}
    depth_by_cam = {cam: np.asarray(obs[depth_uuid(cam)]) for cam in cams}
    image = compose_first_frame_rgbd_sensors(
        rgb_by_cam=rgb_by_cam,
        depth_by_cam=depth_by_cam,
        cameras=cams,
        prox=prox,
        names=order,
        mask=mask,
        caption=caption,
    )
    path = (
        Path(output_dir)
        / "first_frames"
        / f"ep{int(episode_idx):03d}_seed{int(seed)}_rgbd_sensors.png"
    )
    out = write_first_frame_png(path, image)
    print(f"[sensor-keep] first frame {out}", flush=True)
    return out


def summary_keep_block(records: Sequence[Mapping]) -> dict | None:
    return summarize_keep_records(records)
