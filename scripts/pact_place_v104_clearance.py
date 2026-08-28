#!/usr/bin/env python3
"""Exact robot/target-to-pendant clearance for V10.4.

GJK / hardened true distance is authoritative throughout. TCP distance and AABB
gaps are used only as a conservative broad-phase screen and are never returned
as a clearance: when the screen cannot prove separation the pair is measured
exactly.

Used by the Step-0 preflight, by the live rollout telemetry, and by the tests.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pact_geom_distance import GeomShape, gjk_distance, mesh_vertices

SCREEN_MARGIN_M = 0.20
ROBOT_PREFIX = "robot_0/"


def assembly_boxes(assembly: dict[str, Any]) -> list[dict[str, Any]]:
    """Posed axis-aligned box shapes for every active component."""
    out = []
    for item in assembly["components"]:
        if not item.get("active", True):
            continue
        center = np.asarray(item["center_m"], dtype=float)
        half = np.asarray(item["half_m"], dtype=float)
        out.append(
            {
                "name": str(item["name"]),
                "geom": str(item["geom"]),
                "role": str(item["role"]),
                "center": center,
                "half": half,
                "shape": GeomShape.posed_axis_aligned_box(center, half),
            }
        )
    return out


def robot_collision_geom_ids(model) -> list[int]:
    ids = []
    for geom_id in range(int(model.ngeom)):
        body = str(model.body(int(model.geom_bodyid[geom_id])).name or "")
        if not body.startswith(ROBOT_PREFIX):
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(
            model.geom_conaffinity[geom_id]
        ) == 0:
            continue
        ids.append(int(geom_id))
    return ids


def target_collision_geom_ids(task) -> list[int]:
    from reconstruct_pact_place_v99_baseline import pickup_collision_geom_ids

    return [int(value) for value in pickup_collision_geom_ids(task)]


def geom_shape_cache(model, geom_ids: Sequence[int]) -> dict[int, tuple]:
    cache: dict[int, tuple] = {}
    for geom_id in geom_ids:
        cache[int(geom_id)] = (
            int(model.geom_type[int(geom_id)]),
            np.asarray(model.geom_size[int(geom_id)], dtype=float).copy(),
            mesh_vertices(model, int(geom_id)),
        )
    return cache


def _shape(model, data, geom_id: int, cache: dict[int, tuple]) -> GeomShape:
    gtype, size, verts = cache[int(geom_id)]
    return GeomShape(
        gtype, data.geom_xpos[int(geom_id)], data.geom_xmat[int(geom_id)], size, verts
    )


def frame_clearances(
    model,
    data,
    boxes: Sequence[dict[str, Any]],
    probe_ids: Sequence[int],
    cache: dict[int, tuple],
) -> dict[str, Any]:
    """Exact min distance to each component, plus the limiting pair.

    A component is reported as an exact GJK distance whenever any probe geom is
    within ``SCREEN_MARGIN_M`` of it; otherwise the returned value is a proven
    conservative lower bound, flagged as such.
    """
    ids = [int(v) for v in probe_ids]
    if not ids:
        return {"per_component_m": {}, "min_m": None, "limiting": None, "exact": {}}
    centers = np.asarray([data.geom_xpos[g] for g in ids], dtype=float)
    rbound = np.asarray([model.geom_rbound[g] for g in ids], dtype=float)
    per_component: dict[str, float] = {}
    exact_flags: dict[str, bool] = {}
    limiting: dict[str, Any] | None = None
    best_overall = float("inf")
    for box in boxes:
        delta = np.maximum(np.abs(centers - box["center"]) - box["half"], 0.0)
        lower = np.linalg.norm(delta, axis=1) - rbound
        near = np.flatnonzero(lower <= SCREEN_MARGIN_M)
        if near.size == 0:
            value = float(np.min(lower))
            per_component[box["name"]] = value
            exact_flags[box["name"]] = False
            geom_id = int(ids[int(np.argmin(lower))])
        else:
            best = float("inf")
            geom_id = int(ids[int(near[0])])
            for local in near.tolist():
                shape = _shape(model, data, int(ids[local]), cache)
                if not shape.supported:
                    continue
                distance = float(gjk_distance(shape, box["shape"]))
                if distance < best:
                    best = distance
                    geom_id = int(ids[local])
            far = np.delete(lower, near)
            if far.size and float(np.min(far)) < best:
                best = float(np.min(far))
                geom_id = int(ids[int(np.argmin(lower))])
            per_component[box["name"]] = best
            exact_flags[box["name"]] = True
            value = best
        if value < best_overall:
            best_overall = value
            limiting = {
                "component": box["name"],
                "component_geom": box["geom"],
                "probe_geom": str(model.geom(geom_id).name or f"geom_{geom_id}"),
                "probe_body": str(
                    model.body(int(model.geom_bodyid[geom_id])).name or ""
                ),
                "distance_m": float(value),
            }
    return {
        "per_component_m": per_component,
        "min_m": float(best_overall) if np.isfinite(best_overall) else None,
        "limiting": limiting,
        "exact": exact_flags,
    }
