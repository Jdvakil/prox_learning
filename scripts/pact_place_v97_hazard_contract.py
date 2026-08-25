#!/usr/bin/env python3
"""V9.7: hazards contracted by measured subtense, not by an assumed span.

The V9.6 contract required every hazard to span >= 0.25 m.  That number was a
proxy, not a property of the sensor.  The sensor's requirement is angular:

    subtense_px = W_perp / (0.1036 * R)

    width needed for 2 px:  R = 0.15 -> 0.031    R = 0.30 -> 0.062
                            R = 0.40 -> 0.083    R = 0.58 -> 0.120

The corridor's width budget is 0.120 m, which clears 2 px out to R = 0.58 m, and
the ranges W1 measured are 0.11-0.14 m.  A hazard that *fits* the corridor is
resolvable at the ranges that actually occur -- V9.6 simply never scored one,
because the span contract made every candidate too wide to fit by construction.

So this module drops the span floor and admits any composition of one to four
accepted vessels.  What width buys is **sensor count**, not pixel count, and the
sensor-count floor is deliberately left unset here: E1 publishes the reachable
distribution so the floor can be chosen against evidence.

Geometry, posing, the corridor lane test and the panel envelope are imported
from `pact_place_v96_cluster_contract` unchanged.

Nothing in this module renders, steps physics, or authorizes anything.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import pact_place_v96_cluster_contract as v96
from pact_place_v9_contract import (
    NOMINAL_OUTBOUND_END_XY_M,
    NOMINAL_OUTBOUND_START_XY_M,
    SHELF_TOP_Z,
)

ENVIRONMENT_VERSION = "pact_place_corridor_v9_7_subtense"
CONTRACT_VERSION = "pact_place_v9_7_subtense_v1"
SAMPLER_CLASS = "PactPlaceCorridorV97HazardSampler"

CLUSTER_ROLES = v96.CLUSTER_ROLES
DEFAULT_GAP_M = v96.DEFAULT_CLUSTER_GAP_M
#: Widths now in scope.  The lower bound is where a hazard stops being resolvable
#: at any range the trajectories reach; the upper bound is V9.6's old floor.
WIDTH_SCOPE_M = (0.06, 0.25)

#: Accepted UIDs whose collision height sits in the V9 vessel band, ordered by
#: footprint width.  Every one is already settled in palette_v9_1.json.
VESSEL_UIDS = (
    "Soap_Bottle_3",
    "Soap_Bottle_30",
    "Soap_Bottle_11",
    "e3227ecd37d44cd6be1331941d9cfa2f",
    "857d3f1a93b54b25bcc14aab9203346e",
    "Soap_Bottle_1",
    "5d13903e21044558bfb2bb7b72e76b4d",
    "Candle_4",
)
#: Compositions are disjoint between the two legs so no UID is installed twice.
INBOUND_COMPOSITIONS: dict[str, tuple[str, ...]] = {
    "S1_bottle11": ("Soap_Bottle_11",),
    "S1_spraycan": ("e3227ecd37d44cd6be1331941d9cfa2f",),
    "S1_bottle1": ("Soap_Bottle_1",),
    "S1_candle4": ("Candle_4",),
    "P2_bottle11_bottle1": ("Soap_Bottle_11", "Soap_Bottle_1"),
    "P2_spraycan_candle4": ("e3227ecd37d44cd6be1331941d9cfa2f", "Candle_4"),
    "T3_wide": ("Soap_Bottle_11", "857d3f1a93b54b25bcc14aab9203346e",
                "e3227ecd37d44cd6be1331941d9cfa2f"),
}
OUTBOUND_COMPOSITIONS: dict[str, tuple[str, ...]] = {
    "S1_bottle30": ("Soap_Bottle_30",),
    "S1_bottle3": ("Soap_Bottle_3",),
    "S1_candle857": ("857d3f1a93b54b25bcc14aab9203346e",),
    "S1_can5d13": ("5d13903e21044558bfb2bb7b72e76b4d",),
    "P2_bottle30_bottle3": ("Soap_Bottle_30", "Soap_Bottle_3"),
    "P2_bottle30_can5d13": ("Soap_Bottle_30", "5d13903e21044558bfb2bb7b72e76b4d"),
    "T3_wide": ("Soap_Bottle_30", "Soap_Bottle_3",
                "5d13903e21044558bfb2bb7b72e76b4d"),
}
DECOR_UIDS = v96.DECOR_UIDS
DECOR_POSITIONS_XY_M = v96.DECOR_POSITIONS_XY_M
STANDING_QUAT_WXYZ = v96.STANDING_QUAT_WXYZ


def hazard_members(uids: Sequence[str], role: str, slot_offset: int = 0) -> list[dict[str, Any]]:
    """Member dicts for one leg, from the accepted palette records."""
    records = v96.accepted_records()
    members = []
    for index, uid in enumerate(uids):
        record = records.get(str(uid))
        if record is None:
            raise ValueError(f"V9.7 hazard UID is not an accepted palette record: {uid}")
        dimensions = [float(value) for value in record["collision_dimensions_m"]]
        if not 0.15 <= dimensions[2] <= 0.25:
            raise ValueError(f"V9.7 hazard height outside 0.15-0.25 m: {uid} {dimensions}")
        slot = f"{slot_offset + index:02d}"
        members.append(
            {
                "slot": slot,
                "slot_class": "prop",
                "role": role,
                "cluster_member_index": index,
                "uid": str(uid),
                "category": str(record["category"]),
                "dimensions_m": dimensions,
                "annotation_dimensions_m": [float(v) for v in record["dimensions_m"]],
                "half_m": [value / 2.0 for value in dimensions],
                "max_dimension_m": max(dimensions),
                "support": "shelf_standing",
                "quat_wxyz": list(STANDING_QUAT_WXYZ),
                "body_prefix": f"pact_clutter_{slot}/",
            }
        )
    return members


def build_hazard_palette(
    inbound_uids: Sequence[str], outbound_uids: Sequence[str]
) -> dict[str, Any]:
    """Palette of both hazard legs plus the six RGB-only decor items."""
    if set(inbound_uids) & set(outbound_uids):
        raise ValueError("V9.7 legs must not share a UID; each slot installs one body")
    inbound = hazard_members(inbound_uids, "inbound_cluster", 0)
    outbound = hazard_members(outbound_uids, "outbound_cluster", len(inbound))
    records = v96.accepted_records()
    palette = list(inbound) + list(outbound)
    slot_index = len(palette)
    for uid in DECOR_UIDS:
        record = records[str(uid)]
        dimensions = [float(value) for value in record["collision_dimensions_m"]]
        slot = f"{slot_index:02d}"
        palette.append(
            {
                "slot": slot,
                "slot_class": "prop",
                "role": "decor",
                "uid": str(uid),
                "category": str(record["category"]),
                "dimensions_m": dimensions,
                "annotation_dimensions_m": [float(v) for v in record["dimensions_m"]],
                "half_m": [value / 2.0 for value in dimensions],
                "max_dimension_m": max(dimensions),
                "support": "shelf_standing",
                "quat_wxyz": list(STANDING_QUAT_WXYZ),
                "body_prefix": f"pact_clutter_{slot}/",
            }
        )
        slot_index += 1
    return {
        "schema_version": CONTRACT_VERSION,
        "derived_for_environment_version": ENVIRONMENT_VERSION,
        "authorizes_gate": False,
        "authorizes_collection": False,
        "inbound_uids": [str(u) for u in inbound_uids],
        "outbound_uids": [str(u) for u in outbound_uids],
        "palette": palette,
    }


def hazard_geometry(
    uids: Sequence[str], role: str, center_xy, theta_deg: float, gap_m: float = DEFAULT_GAP_M
) -> dict[str, Any]:
    """Pose one leg's hazard.  A single member ignores theta by construction."""
    return v96.cluster_geometry(
        hazard_members(uids, role), center_xy, theta_deg, gap_m, SHELF_TOP_Z
    )


def build_hazard_layout(
    palette_document: dict[str, Any], *, family_id: str, intrusion_side: str,
    inbound: dict[str, Any], outbound: dict[str, Any],
) -> dict[str, Any]:
    layout = v96.build_cluster_layout(
        palette_document, family_id=family_id, intrusion_side=intrusion_side,
        inbound=inbound, outbound=outbound,
    )
    layout["layout_id"] = f"v9_7_hazard_{intrusion_side}_{family_id}"
    layout["layout_contract_version"] = CONTRACT_VERSION
    # V9.6's span/gap contract is deliberately absent: subtense decides.
    layout.pop("cluster_span_floor_m", None)
    layout.pop("cluster_gap_ceiling_m", None)
    return layout


def unary_geometry_reasons(geometry: dict[str, Any], role: str) -> list[str]:
    """Physical impossibilities only.  No span floor and no lateral band."""
    reasons: list[str] = []
    low = np.asarray(geometry["union_low_m"], float)
    high = np.asarray(geometry["union_high_m"], float)
    if not v96.within_workspace(low, high):
        reasons.append("workspace_escape")
    for side in ("left", "right"):
        panel_low, panel_high = v96.panel_envelope(side)
        if v96.boxes_overlap(low, high, panel_low, panel_high, 0.010):
            reasons.append(f"panel_envelope_overlap:{side}")
    if role == "outbound_cluster":
        start, end = NOMINAL_OUTBOUND_START_XY_M, NOMINAL_OUTBOUND_END_XY_M
        t_cross = (float(geometry["union_center_m"][0]) - start[0]) / (end[0] - start[0])
        if not 0.02 < t_cross < 0.98:
            reasons.append("not_crossed_on_the_nominal_loaded_leg")
    return sorted(set(reasons))


def hazard_feasibility(layout: dict[str, Any]) -> dict[str, Any]:
    """Joint feasibility: overlaps plus the slice-by-slice corridor lane test.

    Identical to `v96.layout_feasibility` except that the >= 0.25 m span floor
    and the <= 0.04 m gap ceiling are not applied -- those were the V9.6
    contract, and E1 exists to test whether they were the binding constraint.
    """
    reasons: list[str] = []
    objects = list(layout["objects"])
    for item in objects:
        center = np.asarray(item["center_m"], float)
        half = np.asarray(item["half_m"], float)
        if not v96.within_workspace(center - half, center + half):
            reasons.append(f"workspace_escape:{item['palette_slot']}")
    for index, left in enumerate(objects):
        lc = np.asarray(left["center_m"], float)
        lh = np.asarray(left["half_m"], float)
        for right in objects[index + 1 :]:
            rc = np.asarray(right["center_m"], float)
            rh = np.asarray(right["half_m"], float)
            same_leg = (
                str(left["role"]) == str(right["role"]) and str(left["role"]) in CLUSTER_ROLES
            )
            gap = 1e-9 if same_leg else v96.MIN_OBJECT_GAP_M
            separated = any(
                abs(lc[axis] - rc[axis]) >= lh[axis] + rh[axis] + gap for axis in (0, 1)
            )
            if not separated:
                reasons.append(
                    f"object_overlap:{left['palette_slot']}-{right['palette_slot']}"
                )
    for role in CLUSTER_ROLES:
        low = np.asarray(layout[role]["union_low_m"], float)
        high = np.asarray(layout[role]["union_high_m"], float)
        for side in ("left", "right"):
            panel_low, panel_high = v96.panel_envelope(side)
            if v96.boxes_overlap(low, high, panel_low, panel_high, v96.MIN_OBJECT_GAP_M):
                reasons.append(f"panel_envelope_overlap:{role}:{side}")
    try:
        route = v96.route_metrics_for_cluster(layout)["route"]
    except ValueError as error:
        route = {"error": str(error)}
        reasons.append("outbound_hazard_does_not_block_the_nominal_route")
    if route.get("direct_route_blocked") is False:
        reasons.append("outbound_hazard_does_not_block_the_nominal_route")
    lanes = {}
    for side in ("left", "right"):
        lane = v96.corridor_lane(v96.cluster_obstacles(layout, side))
        lanes[side] = {key: value for key, value in lane.items() if key != "slices"}
        if not lane["lane_admitted"]:
            reasons.append(f"corridor_lane_closed:{side}")
    return {
        "feasible": not reasons,
        "reasons": sorted(set(reasons)),
        "route_metrics": route,
        "corridor_lanes": lanes,
    }
