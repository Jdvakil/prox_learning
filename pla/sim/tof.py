"""ToF sensor rendering — turns the live MuJoCo physics state into ``[N, 8, 8]``
depth readings, one per VL53L5CX sensor camera in the MJCF.

Two entry points:

* ``ToFSensorArray`` — class. Caches the renderer, the sensor-camera ID list
  (in MJCF enumeration order!), and the noise RNG. Call ``render(data)`` once
  per env step. Cheap; no per-step allocation.

* ``extend_obs_with_tof(obs, env, ...)`` — function. Convenience wrapper around
  ``ToFSensorArray`` for legacy callers and one-off scripts (sanity checks,
  replay verifiers, etc.). Internally it caches an ``ToFSensorArray`` on the
  env so repeated calls are fast.

Camera-name convention
    Sensor cameras are *any* camera whose name contains the substring
    ``"sensor"``. We do NOT sort lexicographically — MuJoCo enumeration order
    matches the order the cameras appeared in the MJCF source, which is what
    the build pipeline (``pla.sim.build_mjcf``) controls. Sorting by name
    breaks the per-sensor token order between training and eval.

Noise model (PROJECT.md §3.2):
    Gaussian per-pixel noise with sigma=5 mm, then 5% per-zone dropout to
    saturated max-range (4000 mm), then re-clip to [20, 4000] mm.

Why we do this in MJCF cameras and not a custom raycast: cameras let us reuse
MuJoCo's renderer, which is fast on EGL and gives correct hidden-surface
removal for free. Each camera is 8x8 with 45 deg FOV — exactly what the
VL53L5CX measures.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import mujoco
except ImportError:  # noqa: F401
    mujoco = None  # type: ignore[assignment]


# VL53L5CX physical operating range, per the datasheet.
ZNEAR_MM = 20.0
ZFAR_MM = 4000.0
NOISE_SIGMA_MM = 5.0
DROPOUT_P = 0.05


class ToFSensorArray:
    """Renders all sensor cameras in an MJCF model to depth, every step."""

    def __init__(
        self,
        model,
        height: int = 8,
        width: int = 8,
        sensor_substring: str = "sensor",
        noise_sigma_mm: float = NOISE_SIGMA_MM,
        dropout_p: float = DROPOUT_P,
        rng: np.random.Generator | None = None,
    ) -> None:
        if mujoco is None:
            raise RuntimeError("mujoco not installed; install mujoco>=3.0")
        self.model = model
        self.height = height
        self.width = width
        self.noise_sigma_mm = float(noise_sigma_mm)
        self.dropout_p = float(dropout_p)
        self.rng = rng or np.random.default_rng()

        # Build sensor camera list IN MJCF ORDER.
        cam_ids: list[int] = []
        cam_names: list[str] = []
        for i in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name and sensor_substring in name:
                cam_ids.append(i)
                cam_names.append(name)
        if not cam_ids:
            raise RuntimeError(
                f"no cameras containing {sensor_substring!r} found in model "
                f"(ncam={model.ncam}). Did you run pla.sim.build_mjcf?"
            )
        self.sensor_cam_ids = cam_ids
        self.sensor_cam_names = cam_names
        self.n_sensors = len(cam_ids)

        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.renderer.enable_depth_rendering()

    def render(
        self,
        data,
        add_noise: bool = True,
        clip_min_mm: float = ZNEAR_MM,
        clip_max_mm: float = ZFAR_MM,
    ) -> np.ndarray:
        """Returns ``[N_sensors, H, W]`` float32 in millimetres."""
        out = np.zeros((self.n_sensors, self.height, self.width), dtype=np.float32)
        for i, cam_id in enumerate(self.sensor_cam_ids):
            self.renderer.update_scene(data, camera=cam_id)
            depth_m = self.renderer.render().copy()  # metres
            depth_mm = depth_m * 1000.0
            depth_mm = np.clip(depth_mm, clip_min_mm, clip_max_mm)
            if add_noise:
                depth_mm = depth_mm + self.rng.standard_normal(depth_mm.shape) * self.noise_sigma_mm
                if self.dropout_p > 0:
                    mask = self.rng.random(depth_mm.shape) < self.dropout_p
                    depth_mm = np.where(mask, clip_max_mm, depth_mm)
                depth_mm = np.clip(depth_mm, clip_min_mm, clip_max_mm)
            out[i] = depth_mm.astype(np.float32)
        return out


def _get_or_make_array(env) -> ToFSensorArray:
    arr = getattr(env, "_pla_tof_array", None)
    if arr is None:
        arr = ToFSensorArray(env.model)
        env._pla_tof_array = arr  # type: ignore[attr-defined]
    return arr


def extend_obs_with_tof(
    obs: dict,
    env,
    *,
    sensor_cam_names: Iterable[str] | None = None,
    add_noise: bool = True,
) -> dict:
    """Add ``obs['tof']`` of shape [N_sensors, 8, 8] in mm.

    ``sensor_cam_names`` is accepted for backwards compatibility; if provided
    it overrides the auto-discovered list (used by tests that mount cameras
    with non-standard names).
    """
    if sensor_cam_names is not None:
        # Build a one-off array constrained to those names.
        arr = ToFSensorArray(env.model, sensor_substring="")
        # Filter to requested names, preserving the order the user gave.
        wanted = list(sensor_cam_names)
        ids = [arr.sensor_cam_ids[arr.sensor_cam_names.index(n)] for n in wanted]
        arr.sensor_cam_ids = ids
        arr.sensor_cam_names = wanted
        arr.n_sensors = len(ids)
    else:
        arr = _get_or_make_array(env)
    obs["tof"] = arr.render(env.data, add_noise=add_noise)
    return obs
