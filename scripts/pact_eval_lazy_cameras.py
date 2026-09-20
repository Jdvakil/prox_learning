"""Eval speed hooks. Neither changes what the policy or the judges see.

1. ``install``: lazy registry poses for the 40 proximity cameras.
2. ``install_export_sensor_filter``: drop the per-step segmentation annotation
   sensor (the larger cost in the 2026-09-18 profile: 82 s of a 164 s
   100-step rollout, vs 68 s for pose refresh).

--- 1. lazy poses ---

``CPUMujocoEnv.step`` refreshes every registry camera after every ctrl
substep (33 per policy step). With the hybrid skin that is 33 x 40 x 1050
~= 1.4 M ``update_pose`` calls per episode, ~1100 s of wall, and nothing
reads them: skin depth renders from the MJCF camera by name
(``record_proximity_depths``), not from the registry pose. The only reader
is ``CameraParameterSensor`` (``sensor_param_<cam>``), which no policy,
encoder, or metric consumes.

This patch skips those 40 refreshes and recomputes a proximity camera's
pose from current sim state the moment something reads it. Cameras outside
``prox_names`` (``exo_camera_1``, ``wrist_camera``) keep the original
per-substep ``update_pose`` call, so policy RGB is bit-identical.

Not a protocol field. ``--eager_cameras`` restores the original for A/B.
Do not edit molmospaces for this.
"""
from __future__ import annotations

import weakref
from typing import Iterable

_LOG = "[act-eval-place]"
_INSTALLED = False
_STATS = {"skipped": 0, "flushed": 0}


def add_cli_flags(parser) -> None:
    parser.add_argument(
        "--eager_cameras",
        action="store_true",
        help="Original behavior: refresh all 40 proximity-camera registry "
        "poses after every ctrl substep. Same results, ~10x slower on PACT "
        "arms. A/B check only. Not a protocol field.",
    )


    parser.add_argument(
        "--keep_export_sensors",
        action="store_true",
        help="Original behavior: keep ObjectImagePointsSensor in the suite. It "
        "renders one segmentation frame per (task object x camera) on every "
        "control step (3 x 42 = 126 with the skin) for a dataset annotation no "
        "policy, judge, or metric reads. A/B check only. Not a protocol field.",
    )


def stats() -> dict:
    return dict(_STATS)


# Dataset annotation only. Same filter idea as
# submodules/act/eval_place_fast_hooks.py:_without_export_sensors, minus
# EnvStateSensor (not a hot spot in the 2026-09-18 profile; left in place).
_EXPORT_SENSOR_CLASS_NAMES = frozenset({"ObjectImagePointsSensor"})


def install_export_sensor_filter() -> None:
    """Drop ``object_image_points`` from new sensor suites.

    ``ObjectImagePointsSensor`` bypasses the chunk gate and segmentation-renders
    every task object through every configured camera on every step. Nothing in
    eval reads its output. It does call ``np.random.choice``; nothing else draws
    from the global numpy RNG inside a rollout (action noise is off) and every
    episode reseeds, so dropping it does not move any result. The eager-vs-lazy
    identity check is the proof.
    """
    import molmo_spaces.env.sensors as sensors_module

    original = sensors_module.get_core_sensors
    if getattr(original, "_pact_export_filter", False):
        return

    def get_core_sensors(*args, **kwargs):
        sensors = original(*args, **kwargs)
        kept = [s for s in sensors if type(s).__name__ not in _EXPORT_SENSOR_CLASS_NAMES]
        removed = [s.uuid for s in sensors if type(s).__name__ in _EXPORT_SENSOR_CLASS_NAMES]
        if removed and not _STATS.get("export_logged"):
            _STATS["export_logged"] = 1
            print(f"{_LOG} omit export-only sensors {removed}", flush=True)
        return kept

    get_core_sensors._pact_export_filter = True
    sensors_module.get_core_sensors = get_core_sensors


def install(prox_names: Iterable[str]) -> int:
    """Patch ``CameraRegistry``. Returns the number of lazy cameras."""
    global _INSTALLED
    from molmo_spaces.env.camera_manager import CameraRegistry

    lazy = frozenset(str(n) for n in prox_names)
    if not lazy:
        print(f"{_LOG} lazy prox-camera poses OFF (no proximity cameras)", flush=True)
        return 0
    if _INSTALLED:
        raise RuntimeError(f"{_LOG} lazy prox-camera poses already installed")

    orig_getitem = CameraRegistry.__getitem__
    orig_iter = CameraRegistry.__iter__

    def _flush(self, camera) -> None:
        gen = getattr(self, "_lazy_gen", 0)
        if getattr(camera, "_lazy_seen_gen", 0) == gen:
            return
        camera._lazy_seen_gen = gen
        # Weak ref: CameraManager deliberately holds no env reference.
        ref = getattr(self, "_lazy_env", None)
        env = ref() if ref is not None else None
        if env is None:
            return
        # Full recompute from current sim state. No 1e-6 skip against a stale pose.
        if hasattr(camera, "_last_reference_pose"):
            camera._last_reference_pose = None
        camera.update_pose(env)
        _STATS["flushed"] += 1

    def update_all_cameras(self, env):
        updated: list[str] = []
        n_lazy = 0
        for camera in self.cameras.values():
            if camera.name in lazy:
                n_lazy += 1
                continue
            if camera.update_pose(env):
                updated.append(camera.name)
        self._lazy_env = weakref.ref(env)
        self._lazy_gen = getattr(self, "_lazy_gen", 0) + 1
        _STATS["skipped"] += n_lazy
        return updated

    def __getitem__(self, key):
        camera = orig_getitem(self, key)
        if key in lazy:
            _flush(self, camera)
        return camera

    def __iter__(self):
        for camera in self.cameras.values():
            if camera.name in lazy:
                _flush(self, camera)
        return orig_iter(self)

    CameraRegistry.update_all_cameras = update_all_cameras
    CameraRegistry.__getitem__ = __getitem__
    CameraRegistry.__iter__ = __iter__
    _INSTALLED = True
    print(
        f"{_LOG} lazy prox-camera poses ON (n={len(lazy)}). "
        "RGB cameras keep per-substep update. Skin EGL render unchanged.",
        flush=True,
    )
    return len(lazy)
