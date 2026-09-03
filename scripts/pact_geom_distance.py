#!/usr/bin/env python3
"""Hardened geom-to-geom distance for the PACT place corridor.

``mujoco.mj_geomDistance`` (MuJoCo 3.5) has three observed failure modes in this
codebase, all of which fabricate contacts between separated geoms:

1. **Scalar zero, ``fromto`` written.** Found by v8b. The oriented closest-point
   segment is correct, so its span is the distance.
2. **Stale ``fromto``.** The caller's buffer is reused across calls and MuJoCo
   does not always write it, so a pair can inherit the previous pair's endpoints.
   ``measure_pact_place_v8b_realized.py:41`` has this shape.
3. **Scalar zero, ``fromto`` untouched.** Found by v8c C0. Not covered by (1).
   Reproduced on a BOX/BOX pair whose centres are 0.39 m apart, at every
   ``distmax`` tried (10.0 / 1.0 / 0.5 / 0.2) and with the arguments swapped.
   Left alone it reports a phantom contact.

This module clears the buffer before every call, uses the ``fromto`` span for (1),
and for (3) computes the distance itself with GJK over the two geoms' support
functions -- an exact convex-geometry result, not a bound. MuJoCo treats mesh
geoms as their convex hulls for collision, so a support function over the hull
vertices is the right model for them.

An AABB gap is used only as a last-resort *disproof* when GJK cannot be applied,
and is counted rather than returned. **No AABB-derived value is ever returned as
a clearance.**

Module-level counters record every fallback so a caller can report how often the
instrument had to work around MuJoCo rather than trust it.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

# Fallback tallies. Reset with reset_counters(); read with counters().
_COUNTERS = {
    "calls": 0,
    "fromto_span_fallback": 0,      # mode 1
    "gjk_fallback": 0,              # mode 3, resolved exactly
    "aabb_disproof_only": 0,        # mode 3, GJK unavailable -> skipped
    "accepted_zero": 0,             # scalar zero that survived every check
}

_GJK_MAX_ITER = 128
_GJK_TOL = 1e-12

_PRIMITIVE_TYPES = (
    int(mujoco.mjtGeom.mjGEOM_BOX),
    int(mujoco.mjtGeom.mjGEOM_SPHERE),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
)
PRIMITIVE_GEOM_TYPES = _PRIMITIVE_TYPES
CONTACT_DISTANCE_M = 1e-10


def counters() -> dict[str, int]:
    return dict(_COUNTERS)


def reset_counters() -> None:
    for key in _COUNTERS:
        _COUNTERS[key] = 0


# --------------------------------------------------------------------------
# world AABB -- lifted from run_pact_place_swept_volume_v7.world_aabb_for_geom
# --------------------------------------------------------------------------
def geom_world_aabb(model, data, gid: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(data.geom_xpos[gid], dtype=np.float64)
    mat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[gid], dtype=np.float64)
    gtype = int(model.geom_type[gid])
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        corners = np.array(
            [
                [ix * size[0], iy * size[1], iz * size[2]]
                for ix in (-1.0, 1.0)
                for iy in (-1.0, 1.0)
                for iz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        world = corners @ mat.T + pos
        return world.min(axis=0), world.max(axis=0)
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        radius = float(size[0])
        return pos - radius, pos + radius
    if gtype in (
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    ):
        radius, half_len = float(size[0]), float(size[1])
        axis = mat[:, 2]
        p1 = pos - axis * half_len
        p2 = pos + axis * half_len
        return np.minimum(p1, p2) - radius, np.maximum(p1, p2) + radius
    bound = float(model.geom_rbound[gid])
    return pos - bound, pos + bound


def aabb_gap(model, data, left_gid: int, right_gid: int) -> float:
    """Strict lower bound on true separation. Used only to disprove a contact."""
    left_lo, left_hi = geom_world_aabb(model, data, int(left_gid))
    right_lo, right_hi = geom_world_aabb(model, data, int(right_gid))
    gap = np.maximum(np.maximum(left_lo - right_hi, right_lo - left_hi), 0.0)
    return float(np.linalg.norm(gap))


# --------------------------------------------------------------------------
# GJK distance over support functions
# --------------------------------------------------------------------------
def mesh_vertices(model, gid: int) -> np.ndarray | None:
    mesh_id = int(model.geom_dataid[gid])
    if mesh_id < 0:
        return None
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    if count <= 0:
        return None
    return np.asarray(
        model.mesh_vert[start : start + count], dtype=np.float64
    ).reshape(-1, 3)


def convex_hull_vertices(verts: np.ndarray) -> np.ndarray:
    """Unique convex-hull vertices. GJK support is identical to the full cloud."""
    points = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] <= 4:
        return np.ascontiguousarray(points, dtype=np.float64)
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:
        return np.ascontiguousarray(points, dtype=np.float64)
    try:
        hull = ConvexHull(points)
    except QhullError:
        return np.ascontiguousarray(points, dtype=np.float64)
    return np.ascontiguousarray(points[np.unique(hull.vertices)], dtype=np.float64)


def _support_local(gtype: int, size: np.ndarray, verts: np.ndarray | None,
                   direction: np.ndarray) -> np.ndarray:
    """Farthest point of the geom along ``direction``, in the geom frame."""
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        return np.sign(direction) * size[:3]
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        norm = np.linalg.norm(direction)
        return direction / norm * size[0] if norm > 0 else np.zeros(3)
    if gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        scaled = direction * size[:3]
        norm = np.linalg.norm(scaled)
        return scaled * size[:3] / norm if norm > 0 else np.zeros(3)
    if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        point = np.array([0.0, 0.0, np.sign(direction[2]) * size[1]])
        norm = np.linalg.norm(direction)
        return point + (direction / norm * size[0] if norm > 0 else np.zeros(3))
    if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radial = np.array([direction[0], direction[1], 0.0])
        norm = np.linalg.norm(radial)
        point = radial / norm * size[0] if norm > 0 else np.zeros(3)
        return point + np.array([0.0, 0.0, np.sign(direction[2]) * size[1]])
    if verts is not None:
        return verts[int(np.argmax(verts @ direction))]
    raise ValueError(f"no support function for geom type {gtype}")


class GeomShape:
    """A convex geom posed in world, exposing a GJK support function."""

    def __init__(
        self,
        gtype: int,
        pos: np.ndarray,
        mat: np.ndarray,
        size: np.ndarray,
        verts: np.ndarray | None = None,
    ) -> None:
        self.gtype = int(gtype)
        self.pos = np.asarray(pos, dtype=np.float64).reshape(3)
        self.mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
        self.size = np.asarray(size, dtype=np.float64).reshape(-1)
        self.verts = (
            None
            if verts is None
            else np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        )
        self.supported = self.gtype in _PRIMITIVE_TYPES or self.verts is not None

    @classmethod
    def from_mujoco(cls, model, data, gid: int) -> "GeomShape":
        gtype = int(model.geom_type[gid])
        verts = None if gtype in _PRIMITIVE_TYPES else mesh_vertices(model, gid)
        return cls(
            gtype,
            data.geom_xpos[gid],
            data.geom_xmat[gid],
            model.geom_size[gid],
            verts,
        )

    @classmethod
    def posed_axis_aligned_box(
        cls, center: np.ndarray, half: np.ndarray
    ) -> "GeomShape":
        return cls(
            int(mujoco.mjtGeom.mjGEOM_BOX),
            np.asarray(center, dtype=np.float64).reshape(3),
            np.eye(3, dtype=np.float64),
            np.asarray(half, dtype=np.float64).reshape(3),
            None,
        )

    def support(self, direction: np.ndarray) -> np.ndarray:
        local_dir = self.mat.T @ direction
        local = _support_local(self.gtype, self.size, self.verts, local_dir)
        return self.mat @ local + self.pos

    def world_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        pos, mat, size, gtype = self.pos, self.mat, self.size, self.gtype
        if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            corners = np.array(
                [
                    [ix * size[0], iy * size[1], iz * size[2]]
                    for ix in (-1.0, 1.0)
                    for iy in (-1.0, 1.0)
                    for iz in (-1.0, 1.0)
                ],
                dtype=np.float64,
            )
            world = corners @ mat.T + pos
            return world.min(axis=0), world.max(axis=0)
        if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            radius = float(size[0])
            return pos - radius, pos + radius
        if gtype in (
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        ):
            radius, half_len = float(size[0]), float(size[1])
            axis = mat[:, 2]
            p1 = pos - axis * half_len
            p2 = pos + axis * half_len
            return np.minimum(p1, p2) - radius, np.maximum(p1, p2) + radius
        if self.verts is not None:
            world = self.verts @ mat.T + pos
            return world.min(axis=0), world.max(axis=0)
        raise ValueError(f"no world AABB for geom type {gtype}")


class _Shape(GeomShape):
    """Backward-compatible geom wrapper used by pair_distance."""

    def __init__(self, model, data, gid: int) -> None:
        other = GeomShape.from_mujoco(model, data, gid)
        super().__init__(other.gtype, other.pos, other.mat, other.size, other.verts)


def _closest_point_on_simplex(points: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """Closest point to the origin on the convex hull of 1-4 points."""
    n = len(points)
    if n == 1:
        return points[0], [0]
    best_point = None
    best_norm = np.inf
    best_set: list[int] = []
    # Enumerate every non-empty subset; for <= 4 points this is 15 cases and is
    # numerically safer than the classic Johnson recursion.
    for mask in range(1, 1 << n):
        idx = [i for i in range(n) if mask & (1 << i)]
        sub = np.stack([points[i] for i in idx])
        k = len(idx)
        if k == 1:
            candidate = sub[0]
            weights = np.array([1.0])
        else:
            base = sub[0]
            edges = sub[1:] - base                      # (k-1, 3)
            gram = edges @ edges.T
            try:
                coeffs = np.linalg.solve(gram, -edges @ base)
            except np.linalg.LinAlgError:
                continue
            weights = np.concatenate([[1.0 - coeffs.sum()], coeffs])
            if np.any(weights < -1e-12):
                continue                                 # not an interior projection
            candidate = weights @ sub
        norm = float(np.linalg.norm(candidate))
        if norm < best_norm:
            best_norm = norm
            best_point = candidate
            best_set = idx
    if best_point is None:
        return points[0], [0]
    return best_point, best_set


def gjk_distance(shape_a: GeomShape, shape_b: GeomShape) -> float:
    """Exact distance between two convex geoms. 0.0 means they intersect."""
    direction = shape_a.pos - shape_b.pos
    if np.linalg.norm(direction) < 1e-12:
        direction = np.array([1.0, 0.0, 0.0])
    simplex = [shape_a.support(direction) - shape_b.support(-direction)]
    for _ in range(_GJK_MAX_ITER):
        closest, keep = _closest_point_on_simplex(simplex)
        distance = float(np.linalg.norm(closest))
        if distance < CONTACT_DISTANCE_M:
            return 0.0                                   # origin inside: intersecting
        simplex = [simplex[i] for i in keep]
        direction = -closest                             # search toward the origin
        support = shape_a.support(direction) - shape_b.support(-direction)
        # van den Bergen's termination: with v the current closest point and w the
        # new support, the supporting plane gives a lower bound (v.v - v.w)/|v| on
        # the true distance. Here direction = -v, so v.w == -(support @ direction)
        # and the bound is (|v|^2 + support @ direction) / |v|.
        if (distance * distance + float(support @ direction)) / distance <= _GJK_TOL:
            return distance
        # A repeated support point means the hull cannot grow toward the origin.
        if any(np.allclose(support, point, atol=1e-14) for point in simplex):
            return distance
        simplex.append(support)
        if len(simplex) > 4:
            simplex = simplex[-4:]
    closest, _ = _closest_point_on_simplex(simplex)
    return float(np.linalg.norm(closest))


def exact_pair_distance(model, data, left_gid: int, right_gid: int) -> float | None:
    """GJK distance for a pair, or None if either geom has no support function."""
    shape_a = _Shape(model, data, int(left_gid))
    shape_b = _Shape(model, data, int(right_gid))
    if not (shape_a.supported and shape_b.supported):
        return None
    return gjk_distance(shape_a, shape_b)


# --------------------------------------------------------------------------
# the hardened instrument
# --------------------------------------------------------------------------
def pair_distance(model, data, left_gid: int, right_gid: int,
                  distmax: float = 10.0) -> float:
    """Distance between two geoms, hardened against all three failure modes.

    Returns ``inf`` only when MuJoCo reported a zero that GJK could not check and
    the AABB gap proves the pair is separated -- i.e. the pair is known not to be
    touching but its exact distance is unavailable. Such pairs are counted in
    ``aabb_disproof_only`` rather than silently contributing a zero.
    """
    _COUNTERS["calls"] += 1
    segment = np.zeros(6, dtype=np.float64)
    value = float(
        mujoco.mj_geomDistance(
            model, data, int(left_gid), int(right_gid), float(distmax), segment
        )
    )
    if value != 0.0:
        return value

    span = float(np.linalg.norm(segment[3:] - segment[:3]))
    if span > 1e-9:
        _COUNTERS["fromto_span_fallback"] += 1
        return span

    exact = exact_pair_distance(model, data, int(left_gid), int(right_gid))
    if exact is not None:
        if exact > 0.0:
            _COUNTERS["gjk_fallback"] += 1
        else:
            _COUNTERS["accepted_zero"] += 1
        return exact

    if aabb_gap(model, data, int(left_gid), int(right_gid)) > 0.0:
        _COUNTERS["aabb_disproof_only"] += 1
        return float("inf")
    _COUNTERS["accepted_zero"] += 1
    return 0.0


def true_distance(model, data, left: list[int], right: list[int] | int,
                  distmax: float = 10.0) -> float:
    """Minimum hardened distance between two groups of collision geoms."""
    right_gids = [int(right)] if isinstance(right, (int, np.integer)) else list(right)
    best = float("inf")
    for left_gid in left:
        for right_gid in right_gids:
            value = pair_distance(model, data, int(left_gid), int(right_gid), distmax)
            if value < best:
                best = value
    return best
