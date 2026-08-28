#!/usr/bin/env python3
"""V10.2 raised, collision-legible pendant geometry.

The V10 lattice enumerator and its ``build_lobe`` bounds describe the frozen
V10/V10.1 search space. V10.2 is not a lattice search: it registers exactly one
raised assembly whose lobes sit above ``LOBE_TOP_MAX_M`` and whose stems are
12 mm square in both collision and visible geometry. Nothing here changes
``planning_probe_assembly()`` or reinterprets a V10/V10.1 artifact.

Pure numpy. No MuJoCo.
"""

from __future__ import annotations

from typing import Any, Sequence

from pact_place_v10_compound_pendant_contract import (
    CEILING_TOP_Z_M,
    CROSSBAR_GEOM,
    CROSSBAR_HEIGHT_M,
    DEFAULT_APERTURE_WIDTH_M,
    HOOD_TOP_BOTTOM_Z_M,
    LOBE_GEOMS,
    PENDANT_BODY,
    PENDANT_DEPTH_BOUNDS_M,
    PENDANT_SUPPORT,
    STEM_GEOMS,
    STEM_TOP_Z_M,
    component_aabb,
    component_volume_m3,
    round_m,
    round_vec,
)
from pact_place_v10_geometry import (
    assembly_volume_m3,
    connected_stems_and_crossbar,
    forbidden_static_overlap,
    hood_top_attachment_ok,
    union_fixture,
)

ENVIRONMENT_VERSION_V102 = "pact_place_corridor_v10_2_raised_pendant"
PROBE_LABEL_V102 = "probe_v102_raised"

# 12 mm square, used for stem collision, stem visible geometry, and the
# crossbar x thickness. There is no visual-only sleeve: what the reviewer sees
# is the collision geometry.
STEM_SQUARE_V102_M = 0.012
STEM_HALF_V102_M = STEM_SQUARE_V102_M / 2.0

RAISED_NEGATIVE_LOBE = {
    "center_m": (0.70, -0.18, 1.14),
    "half_m": (0.01, 0.04, 0.04),
}
RAISED_POSITIVE_LOBE = {
    "center_m": (0.70, 0.22, 1.14),
    "half_m": (0.01, 0.02, 0.02),
}
RAISED_STEM_Y_M = (-0.22, 0.24)
RAISED_LOWEST_PENDANT_Z_M = 1.10
SHELF_TOP_Z_M = 0.72
RAISED_SHELF_GAP_M = 0.38
NEGATIVE_LOBE_MAX_Y_M = -0.08
POSITIVE_LOBE_MIN_Y_M = 0.08


def _inside_aperture_enclosure(
    center: Sequence[float],
    half: Sequence[float],
    *,
    aperture_width_m: float,
) -> None:
    lo, hi = component_aabb(center, half)
    if lo[0] < PENDANT_DEPTH_BOUNDS_M[0] - 1e-9 or hi[0] > PENDANT_DEPTH_BOUNDS_M[1] + 1e-9:
        raise ValueError(
            f"component x [{lo[0]:.9f}, {hi[0]:.9f}] is outside {PENDANT_DEPTH_BOUNDS_M}"
        )
    y_limit = float(aperture_width_m) / 2.0
    if lo[1] < -y_limit - 1e-9 or hi[1] > y_limit + 1e-9:
        raise ValueError(
            f"component y [{lo[1]:.9f}, {hi[1]:.9f}] exceeds aperture +/-{y_limit:.9f}"
        )


def build_raised_lobe(
    *,
    center_m: Sequence[float],
    half_m: Sequence[float],
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    center = tuple(float(item) for item in center_m)
    half = tuple(float(item) for item in half_m)
    if len(center) != 3 or len(half) != 3:
        raise ValueError("raised lobe needs three-vector center and half extents")
    if any(value <= 0.0 for value in half):
        raise ValueError(f"raised lobe half extents must be positive: {half}")
    bottom = center[2] - half[2]
    top = center[2] + half[2]
    if bottom < RAISED_LOWEST_PENDANT_Z_M - 1e-9:
        raise ValueError(
            f"raised lobe bottom {bottom:.9f} is below {RAISED_LOWEST_PENDANT_Z_M}"
        )
    if top > STEM_TOP_Z_M - 1e-9:
        raise ValueError(f"raised lobe top {top:.9f} leaves no stem height")
    if center[1] < 0.0:
        if center[1] + half[1] > NEGATIVE_LOBE_MAX_Y_M + 1e-9:
            raise ValueError("negative lobe must lie wholly below y=-0.08")
        side = "negative"
    else:
        if center[1] - half[1] < POSITIVE_LOBE_MIN_Y_M - 1e-9:
            raise ValueError("positive lobe must lie wholly above y=+0.08")
        side = "positive"
    _inside_aperture_enclosure(center, half, aperture_width_m=aperture_width_m)
    return {
        "role": "lobe",
        "side": side,
        "center_m": round_vec(center),
        "half_m": round_vec(half),
        "bottom_z_m": round_m(bottom),
        "top_z_m": round_m(top),
        "key": tuple(round_vec(center) + round_vec(half)),
        "volume_m3": component_volume_m3(half),
    }


def raised_stem_for_lobe(
    lobe: dict[str, Any],
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    """12 mm square stem from the raised lobe top up to ``STEM_TOP_Z_M``."""
    center = lobe["center_m"]
    half = lobe["half_m"]
    outward = -1.0 if lobe["side"] == "negative" else 1.0
    stem_y = float(center[1]) + outward * float(half[1])
    lobe_top = float(lobe["top_z_m"])
    if STEM_TOP_Z_M <= lobe_top + 1e-12:
        raise ValueError("stem has non-positive height")
    half_z = (STEM_TOP_Z_M - lobe_top) / 2.0
    stem_center = (float(center[0]), stem_y, lobe_top + half_z)
    stem_half = (STEM_HALF_V102_M, STEM_HALF_V102_M, half_z)
    _inside_aperture_enclosure(stem_center, stem_half, aperture_width_m=aperture_width_m)
    return {
        "role": "stem",
        "side": lobe["side"],
        "center_m": round_vec(stem_center),
        "half_m": round_vec(stem_half),
        "derived_from_lobe_top_z_m": round_m(lobe_top),
        "volume_m3": component_volume_m3(stem_half),
    }


def raised_crossbar_for_stems(
    stems: Sequence[dict[str, Any]],
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    if len(stems) < 2:
        raise ValueError("crossbar requires at least two stems")
    xs = [float(item["center_m"][0]) for item in stems]
    if max(xs) - min(xs) > 1e-9:
        raise ValueError("stems must share center x")
    ys = [float(item["center_m"][1]) for item in stems]
    y_lo = min(ys) - STEM_HALF_V102_M
    y_hi = max(ys) + STEM_HALF_V102_M
    half_z = CROSSBAR_HEIGHT_M / 2.0
    center = (xs[0], 0.5 * (y_lo + y_hi), CEILING_TOP_Z_M - half_z)
    half = (STEM_HALF_V102_M, 0.5 * (y_hi - y_lo), half_z)
    _inside_aperture_enclosure(center, half, aperture_width_m=aperture_width_m)
    top = center[2] + half[2]
    if abs(top - HOOD_TOP_BOTTOM_Z_M) > 1e-9:
        raise ValueError(f"crossbar top {top:.9f} is not flush to hood_top at 1.515")
    return {
        "role": "crossbar",
        "center_m": round_vec(center),
        "half_m": round_vec(half),
        "volume_m3": component_volume_m3(half),
    }


def _component_record(
    *,
    name: str,
    role: str,
    geom: str,
    slot: int,
    payload: dict[str, Any],
    active: bool,
) -> dict[str, Any]:
    record = {
        "name": name,
        "role": role,
        "geom": geom,
        "slot": int(slot),
        "active": bool(active),
        "center_m": list(payload["center_m"]) if active else [0.0, 0.0, 0.0],
        "half_m": list(payload["half_m"]) if active else [0.001, 0.001, 0.001],
        "volume_m3": float(payload["volume_m3"]) if active else 0.0,
    }
    if active and "side" in payload:
        record["side"] = payload["side"]
    if active and "key" in payload:
        record["key"] = list(payload["key"])
    return record


def assembly_id_for_raised(lobes: Sequence[dict[str, Any]]) -> str:
    keys = ["-".join(f"{value:.9f}" for value in lobe["key"]) for lobe in lobes]
    return "v102_raised_two_lobe_" + "_".join(keys)


def build_raised_assembly(
    lobes: Sequence[dict[str, Any]],
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    if len(lobes) != 2:
        raise ValueError("V10.2 registers exactly one two-lobe assembly")
    sides = {item["side"] for item in lobes}
    if sides != {"negative", "positive"}:
        raise ValueError("V10.2 assembly needs one lobe on each side")
    xs = [float(item["center_m"][0]) for item in lobes]
    if max(xs) - min(xs) > 1e-9:
        raise ValueError("assembly lobes must share center x")
    ordered = sorted(lobes, key=lambda item: (item["side"], item["center_m"][1]))
    stems = [
        raised_stem_for_lobe(lobe, aperture_width_m=aperture_width_m)
        for lobe in ordered
    ]
    bar = raised_crossbar_for_stems(stems, aperture_width_m=aperture_width_m)
    if not connected_stems_and_crossbar(stems, bar):
        raise ValueError("stems and crossbar are not a connected assembly")
    if not hood_top_attachment_ok(bar):
        raise ValueError("crossbar is not attached to hood_top at z=1.515")
    components: list[dict[str, Any]] = []
    for slot in range(3):
        active = slot < len(ordered)
        inactive = {
            "center_m": [0.0, 0.0, 0.0],
            "half_m": [0.001, 0.001, 0.001],
            "volume_m3": 0.0,
        }
        components.append(
            _component_record(
                name=f"lobe_{slot}",
                role="lobe",
                geom=LOBE_GEOMS[slot],
                slot=slot,
                payload=ordered[slot] if active else inactive,
                active=active,
            )
        )
        components.append(
            _component_record(
                name=f"stem_{slot}",
                role="stem",
                geom=STEM_GEOMS[slot],
                slot=slot,
                payload=stems[slot] if active else inactive,
                active=active,
            )
        )
    components.append(
        _component_record(
            name="crossbar",
            role="crossbar",
            geom=CROSSBAR_GEOM,
            slot=0,
            payload=bar,
            active=True,
        )
    )
    for item in components:
        if not item["active"]:
            continue
        if forbidden_static_overlap(item, allow_hood_top=item["role"] == "crossbar"):
            raise ValueError(f"{item['name']} overlaps forbidden static geometry")
    assembly = {
        "topology": "two_lobe",
        "assembly_id": assembly_id_for_raised(ordered),
        "support": PENDANT_SUPPORT,
        "body": PENDANT_BODY,
        "identical_on_both_panel_sides": True,
        "active_on": ["inbound_empty", "outbound_loaded"],
        "center_x_m": round_m(xs[0]),
        "components": components,
        "volume_m3": 0.0,
        "union_fixture": {},
        "stem_square_m": round_m(STEM_SQUARE_V102_M),
        "collision_and_visible_stem_identical": True,
        "kinematic_fixed_assembly": True,
        "physical_swing_dynamics": False,
    }
    assembly["volume_m3"] = assembly_volume_m3(assembly)
    assembly["union_fixture"] = union_fixture(assembly)
    return assembly


def planning_probe_v102_raised_assembly(
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    """The single registered V10.2 raised assembly."""
    negative = build_raised_lobe(
        center_m=RAISED_NEGATIVE_LOBE["center_m"],
        half_m=RAISED_NEGATIVE_LOBE["half_m"],
        aperture_width_m=aperture_width_m,
    )
    positive = build_raised_lobe(
        center_m=RAISED_POSITIVE_LOBE["center_m"],
        half_m=RAISED_POSITIVE_LOBE["half_m"],
        aperture_width_m=aperture_width_m,
    )
    assembly = build_raised_assembly(
        [negative, positive], aperture_width_m=aperture_width_m
    )
    stems = [
        item
        for item in assembly["components"]
        if item["role"] == "stem" and item["active"]
    ]
    stem_ys = tuple(sorted(float(item["center_m"][1]) for item in stems))
    if (
        abs(stem_ys[0] - RAISED_STEM_Y_M[0]) > 1e-9
        or abs(stem_ys[1] - RAISED_STEM_Y_M[1]) > 1e-9
    ):
        raise ValueError(f"raised stems are {stem_ys}, expected {RAISED_STEM_Y_M}")
    assembly["probe_label"] = PROBE_LABEL_V102
    return assembly


def raised_assembly_expectations(assembly: dict[str, Any]) -> dict[str, Any]:
    """Derive, from the assembly itself, the facts the contract must assert."""
    active = [item for item in assembly["components"] if item.get("active")]
    lows = []
    for item in active:
        lo, _hi = component_aabb(item["center_m"], item["half_m"])
        lows.append(float(lo[2]))
    lobes = {item["side"]: item for item in active if item["role"] == "lobe"}
    stems = sorted(
        (item for item in active if item["role"] == "stem"),
        key=lambda item: float(item["center_m"][1]),
    )
    crossbar = next(item for item in active if item["role"] == "crossbar")
    _lo, crossbar_hi = component_aabb(crossbar["center_m"], crossbar["half_m"])
    stem_tops = []
    for item in stems:
        _slo, shi = component_aabb(item["center_m"], item["half_m"])
        stem_tops.append(round_m(float(shi[2])))
    return {
        "lowest_pendant_z_m": round_m(min(lows)),
        "shelf_top_z_m": round_m(SHELF_TOP_Z_M),
        "shelf_to_pendant_gap_m": round_m(min(lows) - SHELF_TOP_Z_M),
        "negative_lobe_center_m": list(lobes["negative"]["center_m"]),
        "negative_lobe_half_m": list(lobes["negative"]["half_m"]),
        "positive_lobe_center_m": list(lobes["positive"]["center_m"]),
        "positive_lobe_half_m": list(lobes["positive"]["half_m"]),
        "stem_center_y_m": [round_m(float(item["center_m"][1])) for item in stems],
        "stem_top_z_m": stem_tops,
        "stem_square_m": [
            round_m(2.0 * float(item["half_m"][0])) for item in stems
        ],
        "stem_square_y_m": [
            round_m(2.0 * float(item["half_m"][1])) for item in stems
        ],
        "crossbar_top_z_m": round_m(float(crossbar_hi[2])),
        "crossbar_square_x_m": round_m(2.0 * float(crossbar["half_m"][0])),
        "kinematic_fixed_assembly": True,
        "physical_swing_dynamics": False,
    }
