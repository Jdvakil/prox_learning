#!/usr/bin/env python3
"""V10.6 pendant geometry: an asymmetric global lattice on the V10.5 shape.

V10.5's audited evidence is that the two sides are not interchangeable: the
loaded-outbound pass is the meaningful near-approach and the two intrusion
sides bind at different radii. V10.6 therefore gives the negative and positive
lobes independent radii while keeping one global assembly for every family, and
keeps the height, lobe/stem/crossbar dimensions, and compiled-static discipline
of V10.4/V10.5 unchanged.

The crossbar spans the actual asymmetric stem endpoints. It is no longer
centred on y = d, and its half-length is derived from those endpoints rather
than assumed, so connectivity is a computed fact and is asserted.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

ENVIRONMENT_VERSION_V106 = "pact_place_corridor_v10_6_v95_clutter_asymmetric_pendant"
CONTRACT_VERSION_V106 = "pact_place_v106_v95_clutter_asymmetric_pendant_v1"
SAMPLER_CLASS_V106 = "PactPlaceCorridorV106Sampler"

BASE_SCENE_NAME_V106 = "pact_place_corridor_v5.xml"
BASE_SCENE_RELATIVE_V106 = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    + BASE_SCENE_NAME_V106
)

PENDANT_BODY_V106 = "pact_clutter_mount_v106"
COMPONENT_NAMES_V106 = ("lobe_0", "lobe_1", "stem_0", "stem_1", "crossbar")
GEOM_NAMES_V106 = tuple(f"{PENDANT_BODY_V106}_{n}_g" for n in COMPONENT_NAMES_V106)
ALL_GEOMS_V106 = GEOM_NAMES_V106

# Frozen shape, inherited unchanged. Not searchable.
LOBE_HALF_M = (0.010, 0.010, 0.030)
LOBE_CENTER_Z_M = 1.01
STEM_HALF_XY_M = 0.006
STEM_SQUARE_M = 0.012
STEM_CENTER_Z_M = 1.2725
STEM_HALF_Z_M = 0.2325
CROSSBAR_CENTER_Z_M = 1.510
CROSSBAR_HALF_X_M = 0.006
CROSSBAR_HALF_Z_M = 0.005

LOBE_BOTTOM_Z_M = 0.98
LOBE_TOP_Z_M = 1.04
STEM_TOP_Z_M = 1.505
HOOD_TOP_BOTTOM_Z_M = 1.515
SHELF_TOP_Z_M = 0.72
PENDANT_RGBA = (0.55, 0.56, 0.58, 1.0)

# ---------------------------------------------------------------------------
# Registered asymmetric lattice. Small and global: one (x, r_neg, r_pos) for
# every layout family, never per-family placement.
# ---------------------------------------------------------------------------
LATTICE_X_M: tuple[float, ...] = (0.800,)
LATTICE_R_NEG_M: tuple[float, ...] = (0.325, 0.330, 0.335)
LATTICE_R_POS_M: tuple[float, ...] = (0.295, 0.300, 0.305)
POSE_OFFSETS_M: dict[str, float] = {"neg5": -0.005, "center": 0.000, "pos5": +0.005}
POSE_IDS: tuple[str, ...] = ("neg5", "center", "pos5")

RISK_BAND_M = (0.015, 0.035)
CLEARANCE_FLOOR_M = 0.015

# Preregistered fallback, fixed before any V10.6 scoring. Never edited after
# results are seen.
FALLBACK_MAX_CONTACTS = 0
FALLBACK_ABSOLUTE_MIN_CLEARANCE_M = 0.010
FALLBACK_MIN_FRACTION_GE_FLOOR = 0.90
FALLBACK_MIN_GROUP_FRACTION_GE_FLOOR = 0.80


def round_m(value: float) -> float:
    return float(round(float(value), 9))


def component_aabb(center_m: Sequence[float], half_m: Sequence[float]):
    lo = [round_m(float(c) - float(h)) for c, h in zip(center_m, half_m)]
    hi = [round_m(float(c) + float(h)) for c, h in zip(center_m, half_m)]
    return lo, hi


def component_specs(
    x_m: float, r_neg_m: float, r_pos_m: float, d_m: float
) -> tuple[dict[str, Any], ...]:
    """Five components for one (x, r_neg, r_pos, d).

    Stems sit on the outward face of their own lobe, so their y positions are
    derived from the two independent radii. The crossbar is then built to span
    exactly those two stem centres.
    """
    x, rn, rp, d = float(x_m), float(r_neg_m), float(r_pos_m), float(d_m)
    stem_neg_y = round_m(-rn - 0.010 + d)
    stem_pos_y = round_m(rp + 0.010 + d)
    bar_center_y = round_m((stem_neg_y + stem_pos_y) / 2.0)
    # Half-length reaches each stem centre plus that stem's half width, so the
    # bar physically overlaps both stems rather than merely touching them.
    bar_half_y = round_m((stem_pos_y - stem_neg_y) / 2.0 + STEM_HALF_XY_M)
    return (
        {"name": "lobe_0", "role": "lobe", "side": "negative",
         "center_m": (x, round_m(-rn + d), LOBE_CENTER_Z_M), "half_m": LOBE_HALF_M},
        {"name": "lobe_1", "role": "lobe", "side": "positive",
         "center_m": (x, round_m(rp + d), LOBE_CENTER_Z_M), "half_m": LOBE_HALF_M},
        {"name": "stem_0", "role": "stem", "side": "negative",
         "center_m": (x, stem_neg_y, STEM_CENTER_Z_M),
         "half_m": (STEM_HALF_XY_M, STEM_HALF_XY_M, STEM_HALF_Z_M)},
        {"name": "stem_1", "role": "stem", "side": "positive",
         "center_m": (x, stem_pos_y, STEM_CENTER_Z_M),
         "half_m": (STEM_HALF_XY_M, STEM_HALF_XY_M, STEM_HALF_Z_M)},
        {"name": "crossbar", "role": "crossbar",
         "center_m": (x, bar_center_y, CROSSBAR_CENTER_Z_M),
         "half_m": (CROSSBAR_HALF_X_M, bar_half_y, CROSSBAR_HALF_Z_M)},
    )


def build_assembly(
    x_m: float,
    r_neg_m: float,
    r_pos_m: float,
    d_m: float,
    *,
    pose_id: str | None = None,
) -> dict[str, Any]:
    """One asymmetric assembly, with every derived fact asserted."""
    components: list[dict[str, Any]] = []
    for item in component_specs(x_m, r_neg_m, r_pos_m, d_m):
        lo, hi = component_aabb(item["center_m"], item["half_m"])
        record = {
            "name": item["name"], "role": item["role"],
            "geom": f"{PENDANT_BODY_V106}_{item['name']}_g",
            "active": True, "static": True,
            "center_m": [round_m(v) for v in item["center_m"]],
            "half_m": [round_m(v) for v in item["half_m"]],
            "aabb_lo_m": lo, "aabb_hi_m": hi,
            "volume_m3": float(
                8.0 * item["half_m"][0] * item["half_m"][1] * item["half_m"][2]
            ),
        }
        if "side" in item:
            record["side"] = item["side"]
        components.append(record)
    by_name = {item["name"]: item for item in components}

    for slot in (0, 1):
        lobe, stem = by_name[f"lobe_{slot}"], by_name[f"stem_{slot}"]
        if abs(lobe["aabb_lo_m"][2] - LOBE_BOTTOM_Z_M) > 1e-9:
            raise ValueError(f"{lobe['name']} bottom is not {LOBE_BOTTOM_Z_M}")
        if abs(lobe["aabb_hi_m"][2] - LOBE_TOP_Z_M) > 1e-9:
            raise ValueError(f"{lobe['name']} top is not {LOBE_TOP_Z_M}")
        if abs(2.0 * stem["half_m"][0] - STEM_SQUARE_M) > 1e-9:
            raise ValueError(f"stem_{slot} x is not 12 mm square")
        if abs(2.0 * stem["half_m"][1] - STEM_SQUARE_M) > 1e-9:
            raise ValueError(f"stem_{slot} y is not 12 mm square")
        if abs(stem["aabb_lo_m"][2] - lobe["aabb_hi_m"][2]) > 1e-9:
            raise ValueError(f"stem_{slot} does not meet lobe_{slot}")
        if abs(stem["aabb_hi_m"][2] - STEM_TOP_Z_M) > 1e-9:
            raise ValueError(f"stem_{slot} top is not {STEM_TOP_Z_M}")
        outward = -1.0 if lobe["side"] == "negative" else 1.0
        face = round_m(float(lobe["center_m"][1]) + outward * float(lobe["half_m"][1]))
        if abs(float(stem["center_m"][1]) - face) > 1e-9:
            raise ValueError(f"stem_{slot} is not on the outward lobe face")

    bar = by_name["crossbar"]
    if abs(2.0 * bar["half_m"][0] - STEM_SQUARE_M) > 1e-9:
        raise ValueError("crossbar x thickness is not 12 mm")
    if abs(bar["aabb_hi_m"][2] - HOOD_TOP_BOTTOM_Z_M) > 1e-9:
        raise ValueError("crossbar top is not flush with hood_top at 1.515")
    if abs(bar["aabb_lo_m"][2] - STEM_TOP_Z_M) > 1e-9:
        raise ValueError("crossbar bottom does not meet the stem tops")

    # Connectivity, computed rather than assumed: the bar must overlap each
    # stem's full 12 mm width in y, on both asymmetric ends.
    bar_lo = float(bar["center_m"][1]) - float(bar["half_m"][1])
    bar_hi = float(bar["center_m"][1]) + float(bar["half_m"][1])
    for slot in (0, 1):
        stem = by_name[f"stem_{slot}"]
        s_lo = float(stem["center_m"][1]) - float(stem["half_m"][1])
        s_hi = float(stem["center_m"][1]) + float(stem["half_m"][1])
        if not (bar_lo <= s_lo + 1e-9 and bar_hi >= s_hi - 1e-9):
            raise ValueError(
                f"crossbar [{bar_lo}, {bar_hi}] does not span stem_{slot} "
                f"[{s_lo}, {s_hi}]"
            )
    if abs(bar_lo - (float(by_name["stem_0"]["center_m"][1]) - STEM_HALF_XY_M)) > 1e-9:
        raise ValueError("crossbar negative end is not at the negative stem edge")
    if abs(bar_hi - (float(by_name["stem_1"]["center_m"][1]) + STEM_HALF_XY_M)) > 1e-9:
        raise ValueError("crossbar positive end is not at the positive stem edge")

    for name in COMPONENT_NAMES_V106:
        if abs(float(by_name[name]["center_m"][0]) - float(x_m)) > 1e-9:
            raise ValueError(f"{name} is not at the assembly depth {x_m}")

    asymmetric = abs(float(r_neg_m) - float(r_pos_m)) > 1e-12
    return {
        "environment_version": ENVIRONMENT_VERSION_V106,
        "contract_version": CONTRACT_VERSION_V106,
        "assembly_id": (
            f"v106_x{int(round(x_m * 1000))}"
            f"_rn{int(round(r_neg_m * 1000))}"
            f"_rp{int(round(r_pos_m * 1000))}"
            f"_d{int(round(d_m * 1000)):+d}"
        ),
        "body": PENDANT_BODY_V106,
        "support": "ceiling",
        "pose_id": pose_id,
        "x_m": round_m(x_m),
        "r_neg_m": round_m(r_neg_m),
        "r_pos_m": round_m(r_pos_m),
        "d_m": round_m(d_m),
        "asymmetric": bool(asymmetric),
        "static": True, "has_joint": False, "has_freejoint": False,
        "is_mocap": False, "has_actuator": False,
        "runtime_repositioned": False, "runtime_bound_refresh": False,
        "connected": True,
        "crossbar_spans_asymmetric_stem_endpoints": True,
        "crossbar_span_y_m": [round_m(bar_lo), round_m(bar_hi)],
        "collision_and_visible_identical": True,
        "visual_only_sleeve": False,
        "components": components,
        "lobe_bottom_z_m": round_m(LOBE_BOTTOM_Z_M),
        "lobe_top_z_m": round_m(LOBE_TOP_Z_M),
        "crossbar_top_z_m": round_m(bar["aabb_hi_m"][2]),
        "shelf_top_z_m": round_m(SHELF_TOP_Z_M),
        "negative_inner_face_abs_y_m": round_m(
            abs(float(by_name["lobe_0"]["center_m"][1]) + float(LOBE_HALF_M[1]))
        ),
        "positive_inner_face_abs_y_m": round_m(
            abs(float(by_name["lobe_1"]["center_m"][1]) - float(LOBE_HALF_M[1]))
        ),
        "volume_m3": float(sum(item["volume_m3"] for item in components)),
    }


def lattice_candidates() -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (x, rn, rp)
        for x in LATTICE_X_M
        for rn in LATTICE_R_NEG_M
        for rp in LATTICE_R_POS_M
    )


def bundle_assemblies(x_m: float, r_neg_m: float, r_pos_m: float):
    return {
        pose: build_assembly(x_m, r_neg_m, r_pos_m, POSE_OFFSETS_M[pose],
                             pose_id=pose)
        for pose in POSE_IDS
    }


def _fmt(values: Sequence[float]) -> str:
    out = []
    for value in values:
        text = f"{float(value):.9f}".rstrip("0").rstrip(".")
        out.append(text if text not in ("", "-0") else "0")
    return " ".join(out)


def scene_model_name(assembly: dict[str, Any]) -> str:
    return f"pact_place_corridor_v10_6_{assembly['assembly_id']}"


def scene_xml_text(assembly: dict[str, Any] | None) -> str:
    """Additive scene: the frozen V5 shell plus one compiled-static pendant."""
    if assembly is None:
        return "\n".join([
            '<mujoco model="pact_place_corridor_v10_6_no_pendant">',
            "  <!-- V10.6 no-pendant counterfactual: the frozen V5 shell only. -->",
            f'  <include file="{BASE_SCENE_NAME_V106}"/>',
            "</mujoco>", "",
        ])
    lines = [
        f'<mujoco model="{scene_model_name(assembly)}">',
        "  <!-- The frozen V5 shell is included verbatim and never edited. V10.6",
        "       adds one connected asymmetric two-lobe pendant compiled at its",
        "       final poses and sizes. The body has no joint, freejoint, mocap",
        "       flag, or actuator, so nothing moves or resizes it at runtime and",
        "       its broad-phase bounds cannot go stale. The crossbar spans the",
        "       actual asymmetric stem endpoints. Visible geometry IS the",
        "       collision geometry. -->",
        f"  <!-- pose_id={assembly['pose_id']} x={assembly['x_m']} "
        f"r_neg={assembly['r_neg_m']} r_pos={assembly['r_pos_m']} "
        f"d={assembly['d_m']} -->",
        f'  <include file="{BASE_SCENE_NAME_V106}"/>',
        "  <worldbody>",
        f'    <body name="{PENDANT_BODY_V106}" pos="0 0 0">',
    ]
    for item in assembly["components"]:
        lines.append(
            f'      <geom name="{item["geom"]}" class="hood" type="box"\n'
            f'            pos="{_fmt(item["center_m"])}" size="{_fmt(item["half_m"])}"\n'
            f'            rgba="{_fmt(PENDANT_RGBA)}"/>'
        )
    lines += ["    </body>", "  </worldbody>", "</mujoco>", ""]
    return "\n".join(lines)


def scene_xml_sha256(assembly: dict[str, Any] | None) -> str:
    return hashlib.sha256(scene_xml_text(assembly).encode()).hexdigest()


def assembly_sha256(assembly: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(assembly, sort_keys=True, separators=(",", ":"),
                   default=str).encode()
    ).hexdigest()


__all__ = [
    "ALL_GEOMS_V106", "BASE_SCENE_NAME_V106", "BASE_SCENE_RELATIVE_V106",
    "CLEARANCE_FLOOR_M", "COMPONENT_NAMES_V106", "CONTRACT_VERSION_V106",
    "ENVIRONMENT_VERSION_V106", "FALLBACK_ABSOLUTE_MIN_CLEARANCE_M",
    "FALLBACK_MAX_CONTACTS", "FALLBACK_MIN_FRACTION_GE_FLOOR",
    "FALLBACK_MIN_GROUP_FRACTION_GE_FLOOR", "GEOM_NAMES_V106",
    "LATTICE_R_NEG_M", "LATTICE_R_POS_M", "LATTICE_X_M", "PENDANT_BODY_V106",
    "POSE_IDS", "POSE_OFFSETS_M", "RISK_BAND_M", "SAMPLER_CLASS_V106",
    "assembly_sha256", "build_assembly", "bundle_assemblies",
    "component_specs", "lattice_candidates", "round_m", "scene_model_name",
    "scene_xml_sha256", "scene_xml_text",
]
