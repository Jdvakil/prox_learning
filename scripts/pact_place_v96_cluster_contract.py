#!/usr/bin/env python3
"""V9.6 clustered-hazard palette and layout, sized by the skin's resolving power.

W1 measured what the 40-sensor skin can resolve: pixel pitch ``0.1036 * R``, so a
0.089 m soap bottle clears one pixel only inside 0.86 m and two pixels only
inside 0.43 m, and the inbound leg's 0.089 m bottle changed 40 of 4.85M raw
values.  V9.6 therefore replaces each single-bottle hazard with a **cluster** of
three tall vessels standing shoulder to shoulder, so each leg presents a
contiguous silhouette of at least 0.25 m.

Two properties are load bearing:

* the palette keeps **twelve prop slots** -- six cluster members and six
  RGB-only decor items -- so the scene still reads as household clutter;
* the cluster is built only from UIDs already accepted in
  ``diagnostics_output/pact_place_v9_v0b/palette_v9_1.json``, so no new asset
  enters the experiment at this stage.

Nothing in this module renders, steps physics, or authorizes anything.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from pact_place_v9_contract import (
    MIN_OBJECT_GAP_M,
    PANEL_FACE_JITTER_MAX_M,
    PANEL_HALF_X_M,
    PANEL_HALF_Y_M,
    PANEL_INNER_FACE_Y_M,
    PANEL_X_JITTER_MAX_M,
    PANEL_X_M,
    SHELF_TOP_Z,
    WORKSPACE_HIGH_XYZ,
    WORKSPACE_LOW_XYZ,
    load_palette,
    panel_corridor_metrics,
    route_blocker_metrics,
)

ENVIRONMENT_VERSION = "pact_place_corridor_v9_6_cluster"
CONTRACT_VERSION = "pact_place_v9_6_cluster_v1"
SAMPLER_CLASS = "PactPlaceCorridorV96ClusterSampler"

#: Minimum contiguous frontal width per leg -- the resolving-power floor from W1.
MIN_CLUSTER_SPAN_M = 0.25
#: Maximum surface-to-surface gap between neighbours, so the silhouette stays
#: contiguous at the pixel pitch of the ranges involved.
MAX_CLUSTER_GAP_M = 0.04
DEFAULT_CLUSTER_GAP_M = 0.025
CLUSTER_ROLES = ("inbound_cluster", "outbound_cluster")

#: Cluster members: accepted UIDs whose collision height lies in the V9 vessel
#: band (0.15-0.25 m) and whose width is large enough to matter in a silhouette.
CLUSTER_RECIPES: dict[str, dict[str, Any]] = {
    "C3_wide": {
        "inbound_cluster": ("Soap_Bottle_11", "857d3f1a93b54b25bcc14aab9203346e",
                            "e3227ecd37d44cd6be1331941d9cfa2f"),
        "outbound_cluster": ("Soap_Bottle_30", "Soap_Bottle_3",
                             "5d13903e21044558bfb2bb7b72e76b4d"),
    },
    "C3_tall": {
        "inbound_cluster": ("Soap_Bottle_11", "Soap_Bottle_1",
                            "e3227ecd37d44cd6be1331941d9cfa2f"),
        "outbound_cluster": ("Soap_Bottle_30", "Soap_Bottle_3",
                             "857d3f1a93b54b25bcc14aab9203346e"),
    },
}
#: RGB-only decor.  Every one of these sits below the resolving floor at working
#: range and is therefore invisible to the skin by physics, not by choice.
DECOR_UIDS = (
    "Mug_2",
    "Plate_10",
    "Plate_22",
    "Apple_29",
    "Potato_1",
    # A bowl was tried here first and would not settle: its collision hull is
    # 0.172 m across against a 0.128 m mesh, and it rocked at 0.44 rad/s.
    "663b5edc92a543668c1b602981e724a4",
)
#: Decor sits deep in the bench, clear of every candidate cluster depth, so a
#: siting decision is never made or unmade by where the RGB-only props landed.
DECOR_POSITIONS_XY_M = (
    (1.020, 0.100),
    (1.140, -0.050),
    (1.140, 0.300),
    (1.290, -0.120),
    (1.290, 0.120),
    (1.020, -0.300),
)
#: Standing orientation used by every V9 prop slot.
STANDING_QUAT_WXYZ = (2**-0.5, 2**-0.5, 0.0, 0.0)


def accepted_records() -> dict[str, dict[str, Any]]:
    document = load_palette()
    return {
        str(item["uid"]): item
        for item in document.get("records") or []
        if item.get("accepted")
    }


def _record(records: dict[str, dict[str, Any]], uid: str) -> dict[str, Any]:
    record = records.get(uid)
    if record is None:
        raise ValueError(f"V9.6 cluster UID is not an accepted palette record: {uid}")
    return record


def build_cluster_palette(recipe_id: str) -> dict[str, Any]:
    """Twelve prop slots: six cluster members, six RGB-only decor items."""
    if recipe_id not in CLUSTER_RECIPES:
        raise ValueError(f"unknown V9.6 cluster recipe: {recipe_id}")
    records = accepted_records()
    recipe = CLUSTER_RECIPES[recipe_id]
    palette: list[dict[str, Any]] = []
    slot_index = 0
    for role in CLUSTER_ROLES:
        for member_index, uid in enumerate(recipe[role]):
            record = _record(records, uid)
            dimensions = [float(value) for value in record["collision_dimensions_m"]]
            if not 0.15 <= dimensions[2] <= 0.25:
                raise ValueError(
                    f"V9.6 cluster member height is outside 0.15-0.25 m: {uid} {dimensions}"
                )
            palette.append(
                {
                    "slot": f"{slot_index:02d}",
                    "slot_class": "prop",
                    "role": role,
                    "cluster_member_index": member_index,
                    "uid": uid,
                    "category": str(record["category"]),
                    "dimensions_m": dimensions,
                    "annotation_dimensions_m": [float(v) for v in record["dimensions_m"]],
                    "half_m": [value / 2.0 for value in dimensions],
                    "max_dimension_m": max(dimensions),
                    "support": "shelf_standing",
                    "quat_wxyz": list(STANDING_QUAT_WXYZ),
                    "body_prefix": f"pact_clutter_{slot_index:02d}/",
                }
            )
            slot_index += 1
    for uid in DECOR_UIDS:
        record = _record(records, uid)
        dimensions = [float(value) for value in record["collision_dimensions_m"]]
        palette.append(
            {
                "slot": f"{slot_index:02d}",
                "slot_class": "prop",
                "role": "decor",
                "uid": uid,
                "category": str(record["category"]),
                "dimensions_m": dimensions,
                "annotation_dimensions_m": [float(v) for v in record["dimensions_m"]],
                "half_m": [value / 2.0 for value in dimensions],
                "max_dimension_m": max(dimensions),
                "support": "shelf_standing",
                "quat_wxyz": list(STANDING_QUAT_WXYZ),
                "body_prefix": f"pact_clutter_{slot_index:02d}/",
            }
        )
        slot_index += 1
    if len(palette) != 12:
        raise ValueError(f"V9.6 palette must hold 12 slots, got {len(palette)}")
    return {
        "schema_version": CONTRACT_VERSION,
        "recipe_id": recipe_id,
        "derived_for_environment_version": ENVIRONMENT_VERSION,
        "authorizes_gate": False,
        "authorizes_collection": False,
        "palette": palette,
    }


# --------------------------------------------------------------------------
# cluster geometry
# --------------------------------------------------------------------------
def support_width_m(half_xy: Sequence[float], theta_rad: float) -> float:
    """Width of an axis-aligned footprint measured along a line direction."""
    return 2.0 * (
        abs(float(half_xy[0]) * math.cos(theta_rad))
        + abs(float(half_xy[1]) * math.sin(theta_rad))
    )


def cluster_geometry(
    members: Sequence[dict[str, Any]],
    center_xy: Sequence[float],
    theta_deg: float,
    gap_m: float = DEFAULT_CLUSTER_GAP_M,
    bench_top_z: float = SHELF_TOP_Z,
) -> dict[str, Any]:
    """Pose ``members`` shoulder to shoulder along a line through ``center_xy``.

    Items keep the world-axis-aligned footprint the V5 injector gives them; only
    the line the cluster is strung along rotates.  Returns per-item centres and
    half extents plus the silhouette measurements the admission floor uses.
    """
    theta = math.radians(float(theta_deg))
    direction = np.array([math.cos(theta), math.sin(theta)])
    widths = [support_width_m(item["half_m"][:2], theta) for item in members]
    span = float(sum(widths) + gap_m * (len(members) - 1))
    offsets = []
    cursor = -span / 2.0
    for width in widths:
        offsets.append(cursor + width / 2.0)
        cursor += width + gap_m
    center = np.asarray(center_xy, dtype=float).reshape(2)
    objects = []
    for item, offset in zip(members, offsets):
        half = [float(value) for value in item["half_m"]]
        xy = center + direction * offset
        objects.append(
            {
                "palette_slot": str(item["slot"]),
                "uid": str(item["uid"]),
                "role": str(item["role"]),
                "category": str(item["category"]),
                "support": "bench_standing",
                "center_m": [float(xy[0]), float(xy[1]), float(bench_top_z + half[2])],
                "half_m": half,
                "quat_wxyz": [float(value) for value in item["quat_wxyz"]],
                "cluster_offset_m": float(offset),
                "size_class": (
                    "small" if max(half) * 2 <= 0.10
                    else "medium" if max(half) * 2 <= 0.18
                    else "large"
                ),
            }
        )
    lows = np.array([np.array(o["center_m"]) - np.array(o["half_m"]) for o in objects])
    highs = np.array([np.array(o["center_m"]) + np.array(o["half_m"]) for o in objects])
    union_low, union_high = lows.min(axis=0), highs.max(axis=0)
    return {
        "objects": objects,
        "center_xy_m": [float(center[0]), float(center[1])],
        "theta_deg": float(theta_deg),
        "gap_m": float(gap_m),
        "member_widths_m": [float(value) for value in widths],
        "span_along_line_m": span,
        "union_low_m": [float(value) for value in union_low],
        "union_high_m": [float(value) for value in union_high],
        "union_extent_m": [float(value) for value in (union_high - union_low)],
        "union_center_m": [float(value) for value in (union_high + union_low) / 2.0],
        "union_half_m": [float(value) for value in (union_high - union_low) / 2.0],
        "boxes": [
            (
                [float(v) for v in low],
                [float(v) for v in high],
            )
            for low, high in zip(lows, highs)
        ],
    }


def panel_envelope(side: str) -> tuple[np.ndarray, np.ndarray]:
    """Worst-case world AABB of one intrusion panel, including its jitter."""
    if side not in {"left", "right"}:
        raise ValueError(f"panel side must be left or right: {side!r}")
    sign = 1.0 if side == "left" else -1.0
    inner = PANEL_INNER_FACE_Y_M - PANEL_FACE_JITTER_MAX_M
    outer = inner + 2.0 * PANEL_HALF_Y_M
    y_low, y_high = sorted((sign * inner, sign * outer))
    x_low = PANEL_X_M - PANEL_X_JITTER_MAX_M - PANEL_HALF_X_M
    x_high = PANEL_X_M + PANEL_X_JITTER_MAX_M + PANEL_HALF_X_M
    # The panel body is 0.18 m tall, centred at z = 0.89 in every retained row.
    return (
        np.array([x_low, y_low, 0.80]),
        np.array([x_high, y_high, 0.98]),
    )


def boxes_overlap(
    low_a: Sequence[float], high_a: Sequence[float],
    low_b: Sequence[float], high_b: Sequence[float],
    clearance_m: float = 0.0,
) -> bool:
    low_a, high_a = np.asarray(low_a, float), np.asarray(high_a, float)
    low_b, high_b = np.asarray(low_b, float), np.asarray(high_b, float)
    return bool(np.all(low_a - clearance_m < high_b) and np.all(low_b - clearance_m < high_a))


def within_workspace(low: Sequence[float], high: Sequence[float]) -> bool:
    low, high = np.asarray(low, float), np.asarray(high, float)
    return bool(
        np.all(low >= np.asarray(WORKSPACE_LOW_XYZ) - 1e-6)
        and np.all(high <= np.asarray(WORKSPACE_HIGH_XYZ) + 1e-6)
    )


def build_cluster_layout(
    palette_document: dict[str, Any],
    *,
    family_id: str,
    intrusion_side: str,
    inbound: dict[str, Any],
    outbound: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a full V9.6 layout from two posed clusters plus fixed decor."""
    if intrusion_side not in {"left", "right"}:
        raise ValueError(f"intrusion_side must be left or right: {intrusion_side!r}")
    palette = list(palette_document["palette"])
    by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in CLUSTER_ROLES}
    decor = [item for item in palette if str(item["role"]) == "decor"]
    for item in palette:
        if str(item["role"]) in by_role:
            by_role[str(item["role"])].append(item)
    inbound_geometry = cluster_geometry(
        by_role["inbound_cluster"], inbound["center_xy_m"],
        inbound["theta_deg"], float(inbound.get("gap_m", DEFAULT_CLUSTER_GAP_M)),
    )
    outbound_geometry = cluster_geometry(
        by_role["outbound_cluster"], outbound["center_xy_m"],
        outbound["theta_deg"], float(outbound.get("gap_m", DEFAULT_CLUSTER_GAP_M)),
    )
    objects = list(inbound_geometry["objects"]) + list(outbound_geometry["objects"])
    for item, xy in zip(decor, DECOR_POSITIONS_XY_M):
        half = [float(value) for value in item["half_m"]]
        objects.append(
            {
                "palette_slot": str(item["slot"]),
                "uid": str(item["uid"]),
                "role": "decor",
                "category": str(item["category"]),
                "support": "bench_standing",
                "center_m": [float(xy[0]), float(xy[1]), float(SHELF_TOP_Z + half[2])],
                "half_m": half,
                "quat_wxyz": [float(value) for value in item["quat_wxyz"]],
                "size_class": "small" if max(half) * 2 <= 0.10 else "medium",
            }
        )
    layout = {
        "layout_id": f"v9_6_cluster_{intrusion_side}_{family_id}",
        "layout_contract_version": CONTRACT_VERSION,
        "layout_family_id": family_id,
        "intrusion_side": intrusion_side,
        "objects": objects,
        "inbound_cluster": {
            key: value for key, value in inbound_geometry.items() if key != "objects"
        },
        "outbound_cluster": {
            key: value for key, value in outbound_geometry.items() if key != "objects"
        },
        "inbound_cluster_slots": [str(item["slot"]) for item in by_role["inbound_cluster"]],
        "outbound_cluster_slots": [str(item["slot"]) for item in by_role["outbound_cluster"]],
        "route_blocker_slot": str(by_role["outbound_cluster"][0]["slot"]),
        "route_blocker_center_xy_m": list(outbound_geometry["union_center_m"][:2]),
        "inbound_vessel_center_xy_m": list(inbound_geometry["union_center_m"][:2]),
        "expected_bow_direction": "-y" if intrusion_side == "left" else "+y",
        "shelf_top_z_m": SHELF_TOP_Z,
        "support": "bench_standing",
        "workspace_bounds_m": [list(WORKSPACE_LOW_XYZ), list(WORKSPACE_HIGH_XYZ)],
        "legacy_panel_active": True,
        "cluster_span_floor_m": MIN_CLUSTER_SPAN_M,
        "cluster_gap_ceiling_m": MAX_CLUSTER_GAP_M,
    }
    return layout


def route_metrics_for_cluster(layout: dict[str, Any]) -> dict[str, Any]:
    """Reuse the settled route/corridor contract with the cluster as one blocker."""
    union = layout["outbound_cluster"]
    proxy = {
        **layout,
        "objects": [
            {
                "palette_slot": layout["route_blocker_slot"],
                "center_m": list(union["union_center_m"]),
                "half_m": list(union["union_half_m"]),
            }
        ],
    }
    return {
        "route": route_blocker_metrics(proxy),
        "corridor": panel_corridor_metrics(proxy),
    }


#: Lateral half-width of the loaded transport envelope, and the aperture reserve.
ARM_ENVELOPE_HALF_Y_M = 0.15
ARM_OBSTACLE_CLEARANCE_M = 0.04
APERTURE_HALF_WIDTH_M = 0.425
APERTURE_EDGE_RESERVE_M = 0.02
CORRIDOR_X_START_M = 0.44
CORRIDOR_X_END_M = 0.80
CORRIDOR_X_STEP_M = 0.005


def _free_intervals(blocked, y_low: float, y_high: float):
    """Complement of a set of blocked y-intervals inside ``[y_low, y_high]``."""
    merged = []
    for low, high in sorted(blocked):
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    free = []
    cursor = y_low
    for low, high in merged:
        if low > cursor:
            free.append((cursor, min(low, y_high)))
        cursor = max(cursor, high)
        if cursor >= y_high:
            break
    if cursor < y_high:
        free.append((cursor, y_high))
    return [(low, high) for low, high in free if high > low]


def corridor_lane(
    obstacles,
    *,
    arm_half_y_m: float = ARM_ENVELOPE_HALF_Y_M,
    clearance_m: float = ARM_OBSTACLE_CLEARANCE_M,
) -> dict[str, Any]:
    """Is there a continuous lane the loaded envelope can follow through depth?

    ``panel_corridor_metrics`` collapses the corridor to one scalar, which is
    correct only when the panel and the blocker sit at the same depth.  A
    clustered hazard is deliberately sited at a different depth, so the lane has
    to be measured slice by slice: at each ``x`` the obstacles that actually span
    that depth remove a band of ``y``, and the lane is admitted only if a free
    band at least one envelope wide persists, and stays connected, from the
    aperture to the target.

    ``obstacles`` is a sequence of ``(low_xyz, high_xyz)`` world AABBs.
    """
    y_low = -APERTURE_HALF_WIDTH_M + APERTURE_EDGE_RESERVE_M
    y_high = APERTURE_HALF_WIDTH_M - APERTURE_EDGE_RESERVE_M
    width = 2.0 * arm_half_y_m
    slices = []
    reachable: list[tuple[float, float]] | None = None
    x = CORRIDOR_X_START_M
    narrowest = float("inf")
    narrowest_x = None
    while x <= CORRIDOR_X_END_M + 1e-9:
        blocked = [
            (float(low[1]) - clearance_m, float(high[1]) + clearance_m)
            for low, high in obstacles
            if float(low[0]) - clearance_m <= x <= float(high[0]) + clearance_m
        ]
        free = [
            (low, high)
            for low, high in _free_intervals(blocked, y_low, y_high)
            if high - low >= width - 1e-9
        ]
        widest = max((high - low for low, high in free), default=0.0)
        if widest < narrowest:
            narrowest, narrowest_x = widest, round(x, 4)
        if reachable is None:
            reachable = list(free)
        else:
            carried = []
            for low, high in free:
                for prev_low, prev_high in reachable:
                    # The envelope centre may slide continuously in y, so a lane
                    # continues wherever two admissible bands overlap at all.
                    if min(high, prev_high) > max(low, prev_low):
                        carried.append((low, high))
                        break
            reachable = carried
        slices.append(
            {
                "x_m": round(x, 4),
                "free_bands": [[round(low, 4), round(high, 4)] for low, high in free],
                "widest_free_band_m": round(widest, 4),
            }
        )
        if not reachable:
            return {
                "lane_admitted": False,
                "closed_at_x_m": round(x, 4),
                "narrowest_free_band_m": round(narrowest, 4),
                "narrowest_at_x_m": narrowest_x,
                "required_band_m": round(width, 4),
                "slices": slices,
            }
        x += CORRIDOR_X_STEP_M
    return {
        "lane_admitted": True,
        "closed_at_x_m": None,
        "narrowest_free_band_m": round(narrowest, 4),
        "narrowest_at_x_m": narrowest_x,
        "required_band_m": round(width, 4),
        "final_bands": [[round(low, 4), round(high, 4)] for low, high in (reachable or [])],
        "slices": slices,
    }


def cluster_obstacles(layout: dict[str, Any], side: str):
    """Panel plus both clusters, as world AABBs, for one panel side."""
    boxes = [tuple(np.asarray(v) for v in panel_envelope(side))]
    for role in CLUSTER_ROLES:
        cluster = layout[role]
        boxes.append(
            (
                np.asarray(cluster["union_low_m"], float),
                np.asarray(cluster["union_high_m"], float),
            )
        )
    return boxes


def layout_feasibility(layout: dict[str, Any]) -> dict[str, Any]:
    """Geometric feasibility only.  This never scores sensing."""
    reasons: list[str] = []
    objects = list(layout["objects"])
    for item in objects:
        center = np.asarray(item["center_m"], float)
        half = np.asarray(item["half_m"], float)
        if not within_workspace(center - half, center + half):
            reasons.append(f"workspace_escape:{item['palette_slot']}")
    for index, left in enumerate(objects):
        lc, lh = np.asarray(left["center_m"], float), np.asarray(left["half_m"], float)
        for right in objects[index + 1 :]:
            rc, rh = np.asarray(right["center_m"], float), np.asarray(right["half_m"], float)
            same_cluster = str(left["role"]) == str(right["role"]) and str(left["role"]) in CLUSTER_ROLES
            gap = 1e-9 if same_cluster else MIN_OBJECT_GAP_M
            separated = any(
                abs(lc[axis] - rc[axis]) >= lh[axis] + rh[axis] + gap for axis in (0, 1)
            )
            if not separated:
                reasons.append(
                    f"object_overlap:{left['palette_slot']}-{right['palette_slot']}"
                )
    for role in CLUSTER_ROLES:
        cluster = layout[role]
        if float(cluster["span_along_line_m"]) < MIN_CLUSTER_SPAN_M - 1e-9:
            reasons.append(f"span_below_floor:{role}")
        if float(cluster["gap_m"]) > MAX_CLUSTER_GAP_M + 1e-9:
            reasons.append(f"gap_above_ceiling:{role}")
        low = np.asarray(cluster["union_low_m"], float)
        high = np.asarray(cluster["union_high_m"], float)
        for side in ("left", "right"):
            panel_low, panel_high = panel_envelope(side)
            if boxes_overlap(low, high, panel_low, panel_high, MIN_OBJECT_GAP_M):
                reasons.append(f"panel_envelope_overlap:{role}:{side}")
    try:
        metrics = route_metrics_for_cluster(layout)
    except ValueError as error:
        metrics = {"route": {"error": str(error)}, "corridor": {"error": str(error)}}
        reasons.append("outbound_cluster_does_not_block_the_nominal_route")
    if metrics["route"].get("direct_route_blocked") is False:
        reasons.append("outbound_cluster_does_not_block_the_nominal_route")
    # The V9.3 corridor scalar assumes the panel and the blocker share a depth.
    # A depth-separated cluster needs the slice-by-slice lane test instead.
    lanes = {}
    for side in ("left", "right"):
        lane = corridor_lane(cluster_obstacles(layout, side))
        lanes[side] = {key: value for key, value in lane.items() if key != "slices"}
        if not lane["lane_admitted"]:
            reasons.append(f"corridor_lane_closed:{side}")
    return {
        "feasible": not reasons,
        "reasons": sorted(set(reasons)),
        "route_metrics": metrics["route"],
        "corridor_metrics": metrics["corridor"],
        "corridor_lanes": lanes,
    }
