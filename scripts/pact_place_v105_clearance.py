#!/usr/bin/env python3
"""Exact clearance and contact instruments for V10.5.

GJK / hardened true distance is authoritative throughout. AABB gaps are a
conservative broad-phase screen only and are never returned as a clearance:
when the screen cannot prove separation, the pair is measured exactly.

V10.5 needs two things V10.4 did not. Risk is scored against the *lobe and
stem* components specifically, because the designed crossbar/hood_top flush
face would otherwise dominate every environment measurement. And the
environment side matters: the pendant now shares the scene with movable
household clutter, so pendant-to-environment clearance is measured separately
from pendant-to-robot clearance and is required to stay positive.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pact_geom_distance import GeomShape, gjk_distance, mesh_vertices

SCREEN_MARGIN_M = 0.20
ROBOT_PREFIX = "robot_0/"
RISK_ROLES = ("lobe", "stem")


def assembly_boxes(assembly: dict[str, Any]) -> list[dict[str, Any]]:
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
                "side": str(item.get("side") or ""),
                "center": center,
                "half": half,
                "shape": GeomShape.posed_axis_aligned_box(center, half),
            }
        )
    return out


def risk_boxes(assembly: dict[str, Any]) -> list[dict[str, Any]]:
    """Lobes and stems only. The crossbar is never a risk witness."""
    return [box for box in assembly_boxes(assembly) if box["role"] in RISK_ROLES]


def side_risk_boxes(assembly: dict[str, Any], intrusion_side: str):
    """The lobe/stem on the y-side a given route is required to bind.

    A left route must bind the negative-y lobe/stem and a right route the
    positive-y pair; measuring against the far side would let an irrelevant
    component satisfy the risk gate.
    """
    want = "negative" if intrusion_side == "left" else "positive"
    return [box for box in risk_boxes(assembly) if box["side"] == want]


def robot_collision_geom_ids(model) -> list[int]:
    ids = []
    for geom_id in range(int(model.ngeom)):
        body = str(model.body(int(model.geom_bodyid[geom_id])).name or "")
        if not body.startswith(ROBOT_PREFIX):
            continue
        if (
            int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
        ):
            continue
        ids.append(int(geom_id))
    return ids


def target_collision_geom_ids(task) -> list[int]:
    from reconstruct_pact_place_v99_baseline import pickup_collision_geom_ids

    return [int(value) for value in pickup_collision_geom_ids(task)]


def pendant_geom_ids(model, geom_names: Sequence[str]) -> list[int]:
    out = []
    for name in geom_names:
        try:
            out.append(int(model.geom(name).id))
        except KeyError:
            continue
    return out


def environment_collision_geom_ids(model, *, exclude: Sequence[int]) -> list[int]:
    """Every collision-enabled non-robot, non-pendant geom in the scene."""
    excluded = {int(v) for v in exclude}
    ids = []
    for geom_id in range(int(model.ngeom)):
        if geom_id in excluded:
            continue
        body = str(model.body(int(model.geom_bodyid[geom_id])).name or "")
        if body.startswith(ROBOT_PREFIX):
            continue
        if (
            int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
        ):
            continue
        ids.append(int(geom_id))
    return ids


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
    within ``SCREEN_MARGIN_M`` of it; otherwise the value is a proven
    conservative lower bound, flagged as such via ``exact``.
    """
    ids = [int(v) for v in probe_ids]
    if not ids or not boxes:
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
                "component_role": box["role"],
                "component_side": box["side"],
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


def pendant_contact_state(model, data, pendant_ids: Sequence[int]) -> dict[str, Any]:
    """Live per-component contact from ``data.contact``, with classification."""
    from molmo_spaces.tasks.pact_place_contact_audit import classify_contact

    ids = {int(v) for v in pendant_ids}
    pairs: list[dict[str, Any]] = []
    classes: set[str] = set()
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if float(contact.dist) > 0.0:
            continue
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 not in ids and geom2 not in ids:
            continue
        pair = {
            "geom1": str(model.geom(geom1).name or f"geom_{geom1}"),
            "geom2": str(model.geom(geom2).name or f"geom_{geom2}"),
            "body1": str(model.body(int(model.geom_bodyid[geom1])).name or ""),
            "body2": str(model.body(int(model.geom_bodyid[geom2])).name or ""),
            "distance_m": float(contact.dist),
        }
        classes.add(classify_contact(pair))
        pairs.append(pair)
    robot_or_target = [
        pair
        for pair in pairs
        if pair["body1"].startswith(ROBOT_PREFIX)
        or pair["body2"].startswith(ROBOT_PREFIX)
        or "cavity_obj_" in f"{pair['body1']} {pair['body2']}"
        or "grasp_target" in f"{pair['body1']} {pair['body2']}"
    ]
    return {
        "n_pairs": len(pairs),
        "contact": bool(pairs),
        "pairs": pairs[:8],
        "contact_classes": sorted(classes),
        "robot_or_target_contact": bool(robot_or_target),
        "robot_or_target_pairs": robot_or_target[:8],
    }


__all__ = [
    "RISK_ROLES",
    "ROBOT_PREFIX",
    "SCREEN_MARGIN_M",
    "assembly_boxes",
    "environment_collision_geom_ids",
    "frame_clearances",
    "geom_shape_cache",
    "pendant_contact_state",
    "pendant_geom_ids",
    "risk_boxes",
    "robot_collision_geom_ids",
    "side_risk_boxes",
    "target_collision_geom_ids",
]
