#!/usr/bin/env python3
"""W1 resolving-power instrument for the 40-sensor hybrid skin.

Each proximity sensor is an 8x8 depth camera with a 45 deg square FOV
(``camera_configs.py``: ``fov=45.0``, ``is_proximity_sensor=True``).  The pixel
pitch at range ``R`` is therefore::

    pitch(R) = 2 * tan(22.5 deg) / 8 * R = 0.103553 * R

An object registers on the skin only when its extent across the view axis
exceeds that pitch.  This module computes, for a candidate hazard and a frozen
trajectory, per (frame, sensor):

``range_m``
    distance from the sensor origin to the nearest hazard surface point that
    lies inside the 22.5 deg half-cone (nearest point overall when nothing is
    in the cone).
``w_perp_m``
    hazard extent projected onto the sensor's image axes -- the camera ``right``
    and ``up`` vectors, i.e. the axes the 8x8 grid is laid out along.
``subtense_px``
    ``w_perp_m / (0.103553 * range_m)``.

Two predicted pixel counts accompany the subtense so that the model can be
checked against the measured raw counterfactual rather than trusted:

``aabb_pixel_hits``
    ray/box intersections of the 64 pixel rays against the hazard's renderable
    world AABBs.  Pure geometry, no occlusion.  This is the quantity a siting
    sweep can evaluate for a *hypothetical* hazard without building a scene.
``ray_pixel_hits``
    ``mujoco.mj_ray`` against the whole scene with the proximity renderer's own
    geom-group filter, counting only pixels whose *first* hit belongs to the
    hazard.  Exact, occlusion-aware, and directly comparable to the validator's
    ``changed_values`` (which is ``4 * ray_pixel_hits`` summed over the window,
    because the four production substeps repeat one frame at frozen qpos).

Nothing here renders, steps physics, or authorizes anything.
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Sequence

import numpy as np

SENSOR_FOV_DEG = 45.0
SENSOR_HALF_FOV_DEG = SENSOR_FOV_DEG / 2.0
SENSOR_HALF_FOV_RAD = math.radians(SENSOR_HALF_FOV_DEG)
SENSOR_HALF_FOV_COS = math.cos(SENSOR_HALF_FOV_RAD)
SENSOR_TAN_HALF_FOV = math.tan(SENSOR_HALF_FOV_RAD)
PIXEL_GRID = 8
#: 2 * tan(22.5 deg) / 8 -- the pixel pitch per metre of range.
PIXEL_PITCH_COEFF = 2.0 * SENSOR_TAN_HALF_FOV / PIXEL_GRID
SUBTENSE_THRESHOLDS_PX = (1.0, 2.0, 4.0)
#: Renderable geom groups, mirroring ``Env._get_proximity_scene_option`` which
#: hides group 2 (the cosmetic skin) so sensors do not see their own housing.
RENDERED_GEOM_GROUPS = (0, 1)
MJ_GEOMGROUP = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
#: Production substep count: at a frozen qpos the buffer repeats one real frame.
PRODUCTION_SUBSTEPS = 4


# --------------------------------------------------------------------------
# pixel geometry
# --------------------------------------------------------------------------
def pixel_tan_offsets() -> np.ndarray:
    """Tangent-space centre offsets of the 8 pixel columns/rows."""
    index = np.arange(PIXEL_GRID, dtype=np.float64)
    return ((2.0 * index + 1.0) / PIXEL_GRID - 1.0) * SENSOR_TAN_HALF_FOV


def pixel_directions(xmat: np.ndarray) -> np.ndarray:
    """64 unit ray directions for one sensor, in MuJoCo image order.

    ``xmat`` is the 3x3 world-from-camera rotation from ``data.cam_xmat``.
    MuJoCo cameras look along ``-z``; image column increases along camera ``+x``
    and image row increases along camera ``-y`` (row 0 is the top).
    """
    xmat = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    right, up, forward = xmat[:, 0], xmat[:, 1], -xmat[:, 2]
    offsets = pixel_tan_offsets()
    u = offsets[None, :]           # columns -> +right
    v = -offsets[:, None]          # rows    -> -up (row 0 at the top)
    dirs = (
        forward[None, None, :]
        + u[..., None] * right[None, None, :]
        + v[..., None] * up[None, None, :]
    ).reshape(-1, 3)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


# --------------------------------------------------------------------------
# hazard geometry
# --------------------------------------------------------------------------
class HazardGeometry:
    """Posed surface geometry of one hazard, in world coordinates.

    ``points`` is a surface point cloud used for the range/subtense model.
    ``boxes`` is a list of ``(low, high)`` world AABBs used for the
    occlusion-free ray/box prediction; a clustered hazard contributes one box
    per item so the silhouette is not inflated by a single enclosing box.
    """

    __slots__ = ("name", "points", "boxes", "geom_ids")

    def __init__(
        self,
        name: str,
        points: np.ndarray,
        boxes: Sequence[tuple[np.ndarray, np.ndarray]],
        geom_ids: Sequence[int] = (),
    ) -> None:
        self.name = str(name)
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not self.points.size:
            raise ValueError(f"hazard {name!r} has no surface points")
        self.boxes = [
            (np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64))
            for low, high in boxes
        ]
        self.geom_ids = frozenset(int(value) for value in geom_ids)

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    def frontal_extent_m(self) -> np.ndarray:
        low, high = self.aabb
        return high - low


def box_surface_points(
    center: np.ndarray, half: np.ndarray, rotation: np.ndarray | None = None,
    spacing_m: float = 0.02,
) -> np.ndarray:
    """Sample the surface of a (possibly rotated) box on a ~``spacing_m`` grid."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    half = np.asarray(half, dtype=np.float64).reshape(3)
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, float).reshape(3, 3)
    axes = []
    for extent in half:
        count = max(2, int(math.ceil(2.0 * extent / max(spacing_m, 1e-4))) + 1)
        axes.append(np.linspace(-extent, extent, count))
    points: list[np.ndarray] = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        grid_a, grid_b = np.meshgrid(axes[other[0]], axes[other[1]], indexing="ij")
        for sign in (-1.0, 1.0):
            local = np.zeros((grid_a.size, 3), dtype=np.float64)
            local[:, other[0]] = grid_a.ravel()
            local[:, other[1]] = grid_b.ravel()
            local[:, axis] = sign * half[axis]
            points.append(local)
    local_points = np.unique(np.concatenate(points, axis=0), axis=0)
    return local_points @ rotation.T + center


def subsample_points(points: np.ndarray, limit: int) -> np.ndarray:
    """Deterministic stride subsample that always retains the AABB corners."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    low, high = points.min(axis=0), points.max(axis=0)
    corners = np.array(
        [
            [low[0] if (i & 1) else high[0],
             low[1] if (i & 2) else high[1],
             low[2] if (i & 4) else high[2]]
            for i in range(8)
        ]
    )
    if points.shape[0] <= limit:
        return np.concatenate([points, corners], axis=0)
    stride = int(math.ceil(points.shape[0] / limit))
    return np.concatenate([points[::stride], corners], axis=0)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
def subtense_for_frame(
    cam_pos: np.ndarray,
    cam_xmat: np.ndarray,
    hazard: HazardGeometry,
    max_range_m: float,
) -> dict[str, np.ndarray]:
    """Per-sensor range / perpendicular extent / subtense for one frame.

    ``cam_pos`` is ``(S, 3)`` and ``cam_xmat`` is ``(S, 3, 3)``.
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(-1, 3)
    cam_xmat = np.asarray(cam_xmat, dtype=np.float64).reshape(-1, 3, 3)
    n_sensors = cam_pos.shape[0]
    right = cam_xmat[:, :, 0]
    up = cam_xmat[:, :, 1]
    forward = -cam_xmat[:, :, 2]

    delta = hazard.points[None, :, :] - cam_pos[:, None, :]        # (S, P, 3)
    distance = np.linalg.norm(delta, axis=2)                        # (S, P)
    safe = np.maximum(distance, 1e-12)
    cos_angle = np.einsum("spk,sk->sp", delta, forward) / safe
    in_cone = (cos_angle > SENSOR_HALF_FOV_COS) & (distance <= max_range_m)

    range_m = np.zeros(n_sensors)
    w_perp_m = np.zeros(n_sensors)
    w_right_m = np.zeros(n_sensors)
    w_up_m = np.zeros(n_sensors)
    in_cone_points = in_cone.sum(axis=1).astype(np.int32)

    coord_right = np.einsum("spk,sk->sp", delta, right)
    coord_up = np.einsum("spk,sk->sp", delta, up)
    for sensor in range(n_sensors):
        mask = in_cone[sensor]
        if not mask.any():
            range_m[sensor] = float(distance[sensor].min())
            continue
        range_m[sensor] = float(distance[sensor][mask].min())
        a = coord_right[sensor][mask]
        b = coord_up[sensor][mask]
        w_right_m[sensor] = float(a.max() - a.min())
        w_up_m[sensor] = float(b.max() - b.min())
        w_perp_m[sensor] = max(w_right_m[sensor], w_up_m[sensor])
    pitch = PIXEL_PITCH_COEFF * np.maximum(range_m, 1e-6)
    subtense_px = np.where(in_cone_points > 0, w_perp_m / pitch, 0.0)
    return {
        "range_m": range_m,
        "w_perp_m": w_perp_m,
        "w_right_m": w_right_m,
        "w_up_m": w_up_m,
        "in_cone_points": in_cone_points,
        "subtense_px": subtense_px,
    }


def pixel_direction_series(cam_xmat: np.ndarray) -> np.ndarray:
    """Pixel ray directions for a whole replay: ``(..., S, 3, 3)`` -> ``(..., S, 64, 3)``."""
    cam_xmat = np.asarray(cam_xmat, dtype=np.float64)
    right = cam_xmat[..., :, 0]
    up = cam_xmat[..., :, 1]
    forward = -cam_xmat[..., :, 2]
    offsets = pixel_tan_offsets()
    u = np.repeat(offsets[None, :], PIXEL_GRID, axis=0).reshape(-1)     # columns
    v = -np.repeat(offsets[:, None], PIXEL_GRID, axis=1).reshape(-1)    # rows
    dirs = (
        forward[..., None, :]
        + u[:, None] * right[..., None, :]
        + v[:, None] * up[..., None, :]
    )
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)


def subtense_series(
    cam_pos: np.ndarray,
    cam_xmat: np.ndarray,
    points: np.ndarray,
    max_range_m: float,
    frame_chunk: int = 32,
) -> dict[str, np.ndarray]:
    """Vectorised ``subtense_for_frame`` over a whole replay.

    ``cam_pos`` is ``(T, S, 3)``, ``cam_xmat`` is ``(T, S, 3, 3)`` and ``points``
    is a static ``(P, 3)`` world-frame surface cloud.  Returns ``(T, S)`` arrays.
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    cam_xmat = np.asarray(cam_xmat, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n_frames, n_sensors = cam_pos.shape[0], cam_pos.shape[1]
    out = {
        "range_m": np.zeros((n_frames, n_sensors)),
        "w_perp_m": np.zeros((n_frames, n_sensors)),
        "subtense_px": np.zeros((n_frames, n_sensors)),
        "in_cone_points": np.zeros((n_frames, n_sensors), dtype=np.int32),
    }
    for start in range(0, n_frames, max(1, frame_chunk)):
        stop = min(n_frames, start + max(1, frame_chunk))
        pos = cam_pos[start:stop]                       # (t, S, 3)
        xmat = cam_xmat[start:stop]
        right, up = xmat[..., :, 0], xmat[..., :, 1]
        forward = -xmat[..., :, 2]
        delta = points[None, None, :, :] - pos[:, :, None, :]      # (t, S, P, 3)
        distance = np.linalg.norm(delta, axis=-1)
        safe = np.maximum(distance, 1e-12)
        cos_angle = np.einsum("tspk,tsk->tsp", delta, forward) / safe
        in_cone = (cos_angle > SENSOR_HALF_FOV_COS) & (distance <= max_range_m)
        any_cone = in_cone.any(axis=-1)
        coord_a = np.einsum("tspk,tsk->tsp", delta, right)
        coord_b = np.einsum("tspk,tsk->tsp", delta, up)
        big = np.inf
        near = np.where(in_cone, distance, big).min(axis=-1)
        a_hi = np.where(in_cone, coord_a, -big).max(axis=-1)
        a_lo = np.where(in_cone, coord_a, big).min(axis=-1)
        b_hi = np.where(in_cone, coord_b, -big).max(axis=-1)
        b_lo = np.where(in_cone, coord_b, big).min(axis=-1)
        width = np.where(any_cone, np.maximum(a_hi - a_lo, b_hi - b_lo), 0.0)
        near = np.where(any_cone, near, distance.min(axis=-1))
        pitch = PIXEL_PITCH_COEFF * np.maximum(near, 1e-6)
        out["range_m"][start:stop] = near
        out["w_perp_m"][start:stop] = width
        out["subtense_px"][start:stop] = np.where(any_cone, width / pitch, 0.0)
        out["in_cone_points"][start:stop] = in_cone.sum(axis=-1)
    return out


def aabb_pixel_hits_series(
    cam_pos: np.ndarray,
    cam_xmat: np.ndarray,
    boxes: Sequence[tuple[np.ndarray, np.ndarray]],
    max_range_m: float,
    frame_chunk: int = 32,
) -> np.ndarray:
    """Occlusion-free ``(T, S)`` pixel-hit counts against a set of world AABBs."""
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    cam_xmat = np.asarray(cam_xmat, dtype=np.float64)
    n_frames, n_sensors = cam_pos.shape[0], cam_pos.shape[1]
    counts = np.zeros((n_frames, n_sensors), dtype=np.int32)
    for start in range(0, n_frames, max(1, frame_chunk)):
        stop = min(n_frames, start + max(1, frame_chunk))
        dirs = pixel_direction_series(cam_xmat[start:stop])          # (t, S, 64, 3)
        origin = cam_pos[start:stop][:, :, None, :]
        safe = np.where(np.abs(dirs) < 1e-12, 1.0, dirs)
        inv = np.where(np.abs(dirs) < 1e-12, np.inf, 1.0 / safe)
        hit = np.zeros(dirs.shape[:-1], dtype=bool)
        for low, high in boxes:
            t_low = (np.asarray(low) - origin) * inv
            t_high = (np.asarray(high) - origin) * inv
            t_near = np.minimum(t_low, t_high).max(axis=-1)
            t_far = np.maximum(t_low, t_high).min(axis=-1)
            hit |= (t_far >= np.maximum(t_near, 0.0)) & (t_near <= max_range_m)
        counts[start:stop] = hit.sum(axis=-1).astype(np.int32)
    return counts


def _ray_boxes(origin: np.ndarray, dirs: np.ndarray,
               boxes: Sequence[tuple[np.ndarray, np.ndarray]],
               max_range_m: float) -> np.ndarray:
    """Slab test of ``dirs`` (N, 3) from ``origin`` against world AABBs."""
    hit = np.zeros(dirs.shape[0], dtype=bool)
    inv = np.where(np.abs(dirs) < 1e-12, np.inf, 1.0 / np.where(np.abs(dirs) < 1e-12, 1.0, dirs))
    for low, high in boxes:
        t_low = (low[None, :] - origin[None, :]) * inv
        t_high = (high[None, :] - origin[None, :]) * inv
        t_near = np.max(np.minimum(t_low, t_high), axis=1)
        t_far = np.min(np.maximum(t_low, t_high), axis=1)
        hit |= (t_far >= np.maximum(t_near, 0.0)) & (t_near <= max_range_m)
    return hit


def aabb_pixel_hits_for_frame(
    cam_pos: np.ndarray, cam_xmat: np.ndarray, hazard: HazardGeometry,
    max_range_m: float,
) -> np.ndarray:
    """Occlusion-free count of the 64 pixel rays that strike the hazard."""
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(-1, 3)
    cam_xmat = np.asarray(cam_xmat, dtype=np.float64).reshape(-1, 3, 3)
    counts = np.zeros(cam_pos.shape[0], dtype=np.int32)
    for sensor in range(cam_pos.shape[0]):
        dirs = pixel_directions(cam_xmat[sensor])
        counts[sensor] = int(
            _ray_boxes(cam_pos[sensor], dirs, hazard.boxes, max_range_m).sum()
        )
    return counts


def ray_pixel_hits_for_frame(
    model, data, cam_pos: np.ndarray, cam_xmat: np.ndarray,
    hazard_geom_ids: frozenset[int], max_range_m: float,
) -> np.ndarray:
    """Occlusion-aware pixel count via ``mj_ray`` against the whole scene."""
    import mujoco

    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(-1, 3)
    cam_xmat = np.asarray(cam_xmat, dtype=np.float64).reshape(-1, 3, 3)
    counts = np.zeros(cam_pos.shape[0], dtype=np.int32)
    geomid = np.zeros(1, dtype=np.int32)
    for sensor in range(cam_pos.shape[0]):
        origin = np.ascontiguousarray(cam_pos[sensor])
        dirs = pixel_directions(cam_xmat[sensor])
        struck = 0
        for ray in dirs:
            distance = mujoco.mj_ray(
                model, data, origin, np.ascontiguousarray(ray),
                MJ_GEOMGROUP, 1, -1, geomid,
            )
            if distance < 0.0 or distance > max_range_m:
                continue
            if int(geomid[0]) in hazard_geom_ids:
                struck += 1
        counts[sensor] = struck
    return counts


def box_corners(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """The 8 corners of one world AABB."""
    low = np.asarray(low, dtype=np.float64).reshape(3)
    high = np.asarray(high, dtype=np.float64).reshape(3)
    return np.array(
        [
            [high[0] if (i & 1) else low[0],
             high[1] if (i & 2) else low[1],
             high[2] if (i & 4) else low[2]]
            for i in range(8)
        ]
    )


def screen_candidate(
    cam_pos: np.ndarray,
    cam_xmat: np.ndarray,
    boxes: Sequence[tuple[np.ndarray, np.ndarray]],
    max_range_m: float,
    frame_chunk: int = 96,
) -> dict[str, np.ndarray]:
    """Fast screen of a *static* candidate hazard given as a set of world AABBs.

    A convex box projects to the convex hull of its projected corners, so the
    corner set alone fixes both the metric extent and the image-plane span.
    Only (frame, sensor) pairs whose cone can reach the hazard are handed to the
    ray/box pixel test, which keeps a placement sweep affordable.

    Returns ``(T, S)`` arrays:

    ``subtense_px``     the plan's ``W_perp / (0.1036 * R)``, from the corner set;
    ``image_span_px``   the same quantity clipped to the 8x8 frame, so it is a
                        literal pixel-column span and saturates at 8;
    ``range_m``         nearest corner distance;
    ``pixel_hits``      occlusion-free count of the 64 pixel rays that strike.
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    cam_xmat = np.asarray(cam_xmat, dtype=np.float64)
    corners = np.concatenate([box_corners(low, high) for low, high in boxes], axis=0)
    n_frames, n_sensors = cam_pos.shape[0], cam_pos.shape[1]
    tan_half = SENSOR_TAN_HALF_FOV
    pitch_tan = 2.0 * tan_half / PIXEL_GRID
    out = {
        "subtense_px": np.zeros((n_frames, n_sensors)),
        "image_span_px": np.zeros((n_frames, n_sensors)),
        "range_m": np.full((n_frames, n_sensors), np.inf),
        "pixel_hits": np.zeros((n_frames, n_sensors), dtype=np.int32),
    }
    for start in range(0, n_frames, max(1, frame_chunk)):
        stop = min(n_frames, start + max(1, frame_chunk))
        pos = cam_pos[start:stop]
        xmat = cam_xmat[start:stop]
        right, up = xmat[..., :, 0], xmat[..., :, 1]
        forward = -xmat[..., :, 2]
        delta = corners[None, None, :, :] - pos[:, :, None, :]         # (t, S, C, 3)
        distance = np.linalg.norm(delta, axis=-1)
        depth = np.einsum("tsck,tsk->tsc", delta, forward)
        coord_a = np.einsum("tsck,tsk->tsc", delta, right)
        coord_b = np.einsum("tsck,tsk->tsc", delta, up)
        front = depth > 1e-6
        near = np.where(front & (distance <= max_range_m), distance, np.inf).min(axis=-1)
        out["range_m"][start:stop] = near

        # Metric extent perpendicular to the view axis, over the in-front corners.
        a_hi = np.where(front, coord_a, -np.inf).max(axis=-1)
        a_lo = np.where(front, coord_a, np.inf).min(axis=-1)
        b_hi = np.where(front, coord_b, -np.inf).max(axis=-1)
        b_lo = np.where(front, coord_b, np.inf).min(axis=-1)
        any_front = front.any(axis=-1)
        w_perp = np.where(any_front, np.maximum(a_hi - a_lo, b_hi - b_lo), 0.0)
        reachable = any_front & np.isfinite(near)
        pitch = PIXEL_PITCH_COEFF * np.maximum(near, 1e-6)

        # Image-plane span, clipped to the frame.
        safe_depth = np.where(front, depth, 1.0)
        u = np.where(front, coord_a / safe_depth, np.nan)
        v = np.where(front, coord_b / safe_depth, np.nan)
        with np.errstate(invalid="ignore"):
            u_hi = np.minimum(np.nanmax(np.where(front, u, -np.inf), axis=-1), tan_half)
            u_lo = np.maximum(np.nanmin(np.where(front, u, np.inf), axis=-1), -tan_half)
            v_hi = np.minimum(np.nanmax(np.where(front, v, -np.inf), axis=-1), tan_half)
            v_lo = np.maximum(np.nanmin(np.where(front, v, np.inf), axis=-1), -tan_half)
        visible = reachable & (u_hi >= u_lo) & (v_hi >= v_lo)
        span = np.where(visible, np.maximum(u_hi - u_lo, v_hi - v_lo) / pitch_tan, 0.0)
        out["image_span_px"][start:stop] = span
        # The metric form is only meaningful where the hazard is actually in
        # frame; without that gate an object beside the sensor scores highly.
        out["subtense_px"][start:stop] = np.where(visible, w_perp / pitch, 0.0)

        # Ray/box pixel test, only where the hazard can enter the frame at all.
        frames_idx, sensors_idx = np.nonzero(visible)
        if frames_idx.size:
            sub_pos = pos[frames_idx, sensors_idx]                       # (N, 3)
            sub_xmat = xmat[frames_idx, sensors_idx]
            dirs = pixel_direction_series(sub_xmat[:, None, :, :])[:, 0] # (N, 64, 3)
            origin = sub_pos[:, None, :]
            safe = np.where(np.abs(dirs) < 1e-12, 1.0, dirs)
            inv = np.where(np.abs(dirs) < 1e-12, np.inf, 1.0 / safe)
            hit = np.zeros(dirs.shape[:-1], dtype=bool)
            for low, high in boxes:
                t_low = (np.asarray(low) - origin) * inv
                t_high = (np.asarray(high) - origin) * inv
                t_near = np.minimum(t_low, t_high).max(axis=-1)
                t_far = np.maximum(t_low, t_high).min(axis=-1)
                hit |= (t_far >= np.maximum(t_near, 0.0)) & (t_near <= max_range_m)
            counts = hit.sum(axis=-1).astype(np.int32)
            block = out["pixel_hits"][start:stop]
            block[frames_idx, sensors_idx] = counts
    out["range_m"][~np.isfinite(out["range_m"])] = np.nan
    return out


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def aggregate(
    subtense_px: np.ndarray,
    sensor_names: Sequence[str],
    frame_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Aggregate a ``(T, S)`` subtense array over an optional frame mask."""
    subtense_px = np.asarray(subtense_px, dtype=np.float64)
    if frame_mask is not None:
        subtense_px = subtense_px[np.asarray(frame_mask, dtype=bool)]
    summary: dict[str, Any] = {
        "n_frames": int(subtense_px.shape[0]),
        "n_sensors": int(subtense_px.shape[1]),
        "max_subtense_px": float(subtense_px.max()) if subtense_px.size else 0.0,
    }
    for threshold in SUBTENSE_THRESHOLDS_PX:
        key = f"{threshold:g}".replace(".", "p")
        clears = subtense_px >= threshold
        summary[f"n_sensor_frames_ge_{key}px"] = int(clears.sum())
        summary[f"n_sensors_ge_{key}px"] = int(clears.any(axis=0).sum())
        summary[f"sensors_ge_{key}px"] = [
            str(name) for name, flag in zip(sensor_names, clears.any(axis=0)) if flag
        ]
    per_sensor = []
    for index, name in enumerate(sensor_names):
        column = subtense_px[:, index] if subtense_px.size else np.zeros(0)
        if not column.size or column.max() <= 0.0:
            continue
        per_sensor.append(
            {
                "sensor_name": str(name),
                "link": str(name).split("_sensor_", 1)[0],
                "max_subtense_px": float(column.max()),
                "n_frames_ge_1px": int((column >= 1.0).sum()),
                "n_frames_ge_2px": int((column >= 2.0).sum()),
                "n_frames_ge_4px": int((column >= 4.0).sum()),
            }
        )
    per_sensor.sort(key=lambda item: -item["max_subtense_px"])
    summary["per_sensor"] = per_sensor
    return summary


def pixel_hit_summary(
    hits: np.ndarray, sensor_names: Sequence[str],
    frame_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    hits = np.asarray(hits, dtype=np.int64)
    if frame_mask is not None:
        hits = hits[np.asarray(frame_mask, dtype=bool)]
    per_sensor_total = hits.sum(axis=0) if hits.size else np.zeros(len(sensor_names), int)
    return {
        "total_pixel_hits": int(hits.sum()),
        "predicted_changed_values": int(hits.sum()) * PRODUCTION_SUBSTEPS,
        "n_responding_sensors": int((per_sensor_total > 0).sum()),
        "responding_sensors": {
            str(name): int(total)
            for name, total in zip(sensor_names, per_sensor_total)
            if total > 0
        },
        "predicted_changed_values_by_sensor": {
            str(name): int(total) * PRODUCTION_SUBSTEPS
            for name, total in zip(sensor_names, per_sensor_total)
            if total > 0
        },
    }


def resolving_floor_m(range_m: float, pixels: float = 1.0) -> float:
    """Smallest frontal width that spans ``pixels`` at ``range_m``."""
    return float(pixels * PIXEL_PITCH_COEFF * range_m)


def range_for_subtense_m(width_m: float, pixels: float = 1.0) -> float:
    """Range at which ``width_m`` subtends ``pixels``."""
    return float(width_m / (pixels * PIXEL_PITCH_COEFF))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _table(rows: Iterable[Sequence[str]]) -> str:
    rows = [list(map(str, row)) for row in rows]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in rows
    )


def main() -> int:
    """Print the resolving-power table this instrument is built around."""
    rows = [["object", "width_m", "1 px at (m)", "2 px at (m)", "4 px at (m)"]]
    for label, width in (
        ("intrusion panel", 0.480),
        ("clustered hazard floor", 0.250),
        ("soapbottle", 0.089),
        ("candle", 0.016),
    ):
        rows.append(
            [
                label,
                f"{width:.3f}",
                f"{range_for_subtense_m(width, 1.0):.2f}",
                f"{range_for_subtense_m(width, 2.0):.2f}",
                f"{range_for_subtense_m(width, 4.0):.2f}",
            ]
        )
    print(f"pixel pitch coefficient: {PIXEL_PITCH_COEFF:.6f} per metre of range")
    print(_table(rows))
    print(json.dumps({"authorizes_gate": False, "authorizes_collection": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
