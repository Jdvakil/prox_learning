#!/usr/bin/env python3
"""V10.4 first-shot static pendant geometry and scene serialization.

One frozen trust-anchor assembly. This is not a search lattice: there is no
height sweep, no lane, and no jitter on the production pendant.

The assembly is compiled into its own additive scene at its final poses and
sizes. The body carries no joint, freejoint, or mocap flag, and nothing writes
``model.geom_pos``, ``model.geom_size``, ``model.geom_aabb``,
``model.geom_rbound`` or BVH fields at episode runtime. That is the direct fix
for the stale broad-phase defect recorded in
``diagnostics_output/pact_place_v102_preflight/contact_parity_root_cause.json``.

Pure numpy plus text serialization. No MuJoCo.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

ENVIRONMENT_VERSION_V104 = "pact_place_corridor_v10_4_first_shot_static_pendant"
CONTRACT_VERSION_V104 = "pact_place_v104_first_shot_static_pendant_v1"
BASE_SAMPLER_CLASS = "PactPlaceCorridorV3Sampler"
BASE_CONFIG_RELATIVE = "configs/pact_place_corridor_v6c.json"
BASE_SCENE_RELATIVE = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v3.xml"
)
SCENE_XML_RELATIVE_V104 = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v10_4.xml"
)
SCENE_METADATA_RELATIVE_V104 = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v10_4_metadata.json"
)

PENDANT_BODY_V104 = "pact_clutter_mount_v104"
LOBE_GEOMS_V104 = (
    "pact_clutter_mount_v104_lobe_0_g",
    "pact_clutter_mount_v104_lobe_1_g",
)
STEM_GEOMS_V104 = (
    "pact_clutter_mount_v104_stem_0_g",
    "pact_clutter_mount_v104_stem_1_g",
)
CROSSBAR_GEOM_V104 = "pact_clutter_mount_v104_crossbar_g"
ALL_GEOMS_V104 = LOBE_GEOMS_V104 + STEM_GEOMS_V104 + (CROSSBAR_GEOM_V104,)

SHELF_TOP_Z_M = 0.72
HOOD_TOP_BOTTOM_Z_M = 1.515
STEM_TOP_Z_M = 1.505
CROSSBAR_HEIGHT_M = 0.010
STEM_SQUARE_M = 0.012
STEM_HALF_M = 0.006

# Registered production assembly. Frozen before the first live V10.4 row.
PRODUCTION_COMPONENTS: tuple[dict[str, Any], ...] = (
    {
        "name": "lobe_0",
        "role": "lobe",
        "side": "negative",
        "geom": LOBE_GEOMS_V104[0],
        "center_m": (0.78, -0.34, 1.01),
        "half_m": (0.010, 0.010, 0.030),
    },
    {
        "name": "lobe_1",
        "role": "lobe",
        "side": "positive",
        "geom": LOBE_GEOMS_V104[1],
        "center_m": (0.78, 0.34, 1.01),
        "half_m": (0.010, 0.010, 0.030),
    },
    {
        "name": "stem_0",
        "role": "stem",
        "side": "negative",
        "geom": STEM_GEOMS_V104[0],
        "center_m": (0.78, -0.35, 1.2725),
        "half_m": (0.006, 0.006, 0.2325),
    },
    {
        "name": "stem_1",
        "role": "stem",
        "side": "positive",
        "geom": STEM_GEOMS_V104[1],
        "center_m": (0.78, 0.35, 1.2725),
        "half_m": (0.006, 0.006, 0.2325),
    },
    {
        "name": "crossbar",
        "role": "crossbar",
        "geom": CROSSBAR_GEOM_V104,
        "center_m": (0.78, 0.0, 1.510),
        "half_m": (0.006, 0.356, 0.005),
    },
)

LOBE_RGBA = (0.55, 0.56, 0.58, 1.0)
STEM_RGBA = (0.55, 0.56, 0.58, 1.0)
CROSSBAR_RGBA = (0.55, 0.56, 0.58, 1.0)

# Offline robustness diagnostic only. The production pendant has no jitter.
CORNER_TRANSLATION_M = 0.005
CORNER_MIN_CLEARANCE_V95_M = 0.030
CORNER_MIN_CLEARANCE_V6C_M = 0.040

# Read-only audit results that Step 0B must independently reproduce or improve.
AUDIT_V95_HARD_FLOOR_M = 0.035
AUDIT_V95_OBSERVED_MIN_M = 0.04052
AUDIT_V6C_HARD_FLOOR_M = 0.050
AUDIT_V6C_OBSERVED_MIN_M = 0.05523
AUDIT_PROVENANCE_TOLERANCE_M = 0.001


def round_m(value: float) -> float:
    return float(round(float(value), 9))


def component_aabb(center_m: Sequence[float], half_m: Sequence[float]):
    return (
        [round_m(float(center_m[i]) - float(half_m[i])) for i in range(3)],
        [round_m(float(center_m[i]) + float(half_m[i])) for i in range(3)],
    )


def production_assembly() -> dict[str, Any]:
    """The single frozen V10.4 assembly, with its derived facts asserted."""
    components = []
    for item in PRODUCTION_COMPONENTS:
        lo, hi = component_aabb(item["center_m"], item["half_m"])
        record = {
            "name": item["name"],
            "role": item["role"],
            "geom": item["geom"],
            "active": True,
            "static": True,
            "center_m": [round_m(v) for v in item["center_m"]],
            "half_m": [round_m(v) for v in item["half_m"]],
            "aabb_lo_m": lo,
            "aabb_hi_m": hi,
            "volume_m3": float(
                8.0 * item["half_m"][0] * item["half_m"][1] * item["half_m"][2]
            ),
        }
        if "side" in item:
            record["side"] = item["side"]
        components.append(record)
    by_name = {item["name"]: item for item in components}

    lobes = [by_name["lobe_0"], by_name["lobe_1"]]
    for lobe in lobes:
        if abs(lobe["aabb_lo_m"][2] - 0.98) > 1e-9:
            raise ValueError(f"{lobe['name']} bottom is not 0.98 m")
        if abs(lobe["aabb_hi_m"][2] - 1.04) > 1e-9:
            raise ValueError(f"{lobe['name']} top is not 1.04 m")
    gap = round_m(lobes[0]["aabb_lo_m"][2] - SHELF_TOP_Z_M)
    if abs(gap - 0.26) > 1e-9:
        raise ValueError(f"shelf-to-lobe gap is {gap}, expected 0.26")

    for slot in (0, 1):
        lobe = by_name[f"lobe_{slot}"]
        stem = by_name[f"stem_{slot}"]
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
    if abs(2.0 * bar["half_m"][2] - CROSSBAR_HEIGHT_M) > 1e-9:
        raise ValueError("crossbar height is not 10 mm")
    if abs(bar["aabb_hi_m"][2] - HOOD_TOP_BOTTOM_Z_M) > 1e-9:
        raise ValueError("crossbar top is not flush with hood_top at 1.515")
    if abs(bar["aabb_lo_m"][2] - STEM_TOP_Z_M) > 1e-9:
        raise ValueError("crossbar bottom does not meet the stem tops")
    for slot in (0, 1):
        stem = by_name[f"stem_{slot}"]
        if abs(stem["center_m"][1]) > float(bar["half_m"][1]) + 1e-9:
            raise ValueError(f"crossbar does not span stem_{slot}")

    # y symmetry
    if by_name["lobe_0"]["center_m"][1] != -by_name["lobe_1"]["center_m"][1]:
        raise ValueError("lobes are not symmetric in y")
    if by_name["stem_0"]["center_m"][1] != -by_name["stem_1"]["center_m"][1]:
        raise ValueError("stems are not symmetric in y")
    if abs(float(bar["center_m"][1])) > 1e-12:
        raise ValueError("crossbar is not centred in y")
    for a, b in (("lobe_0", "lobe_1"), ("stem_0", "stem_1")):
        if by_name[a]["half_m"] != by_name[b]["half_m"]:
            raise ValueError(f"{a}/{b} half extents differ")
        if by_name[a]["center_m"][0] != by_name[b]["center_m"][0]:
            raise ValueError(f"{a}/{b} centre x differ")
        if by_name[a]["center_m"][2] != by_name[b]["center_m"][2]:
            raise ValueError(f"{a}/{b} centre z differ")

    assembly = {
        "environment_version": ENVIRONMENT_VERSION_V104,
        "contract_version": CONTRACT_VERSION_V104,
        "assembly_id": "v104_first_shot_symmetric_two_lobe",
        "body": PENDANT_BODY_V104,
        "support": "ceiling",
        "static": True,
        "has_joint": False,
        "has_freejoint": False,
        "is_mocap": False,
        "runtime_repositioned": False,
        "runtime_bound_refresh": False,
        "symmetric_in_y": True,
        "identical_on_both_panel_sides": True,
        "visual_only_sleeve": False,
        "collision_and_visible_identical": True,
        "components": components,
        "lobe_bottom_z_m": round_m(lobes[0]["aabb_lo_m"][2]),
        "lobe_top_z_m": round_m(lobes[0]["aabb_hi_m"][2]),
        "shelf_top_z_m": round_m(SHELF_TOP_Z_M),
        "shelf_to_lobe_gap_m": gap,
        "stem_square_m": round_m(STEM_SQUARE_M),
        "crossbar_square_x_m": round_m(STEM_SQUARE_M),
        "crossbar_height_m": round_m(CROSSBAR_HEIGHT_M),
        "crossbar_top_z_m": round_m(bar["aabb_hi_m"][2]),
        "volume_m3": float(sum(item["volume_m3"] for item in components)),
    }
    return assembly


def assembly_expectations(assembly: dict[str, Any]) -> dict[str, Any]:
    """Derived facts recomputed from the assembly itself."""
    by_name = {item["name"]: item for item in assembly["components"]}
    return {
        "lobe_bottom_z_m": [
            round_m(by_name[f"lobe_{i}"]["aabb_lo_m"][2]) for i in (0, 1)
        ],
        "lobe_top_z_m": [round_m(by_name[f"lobe_{i}"]["aabb_hi_m"][2]) for i in (0, 1)],
        "shelf_to_lobe_gap_m": round_m(
            by_name["lobe_0"]["aabb_lo_m"][2] - SHELF_TOP_Z_M
        ),
        "lobe_abs_center_y_m": [
            round_m(abs(by_name[f"lobe_{i}"]["center_m"][1])) for i in (0, 1)
        ],
        "stem_abs_center_y_m": [
            round_m(abs(by_name[f"stem_{i}"]["center_m"][1])) for i in (0, 1)
        ],
        "stem_square_m": [
            round_m(2.0 * by_name[f"stem_{i}"]["half_m"][0]) for i in (0, 1)
        ],
        "stem_square_y_m": [
            round_m(2.0 * by_name[f"stem_{i}"]["half_m"][1]) for i in (0, 1)
        ],
        "stem_top_z_m": [round_m(by_name[f"stem_{i}"]["aabb_hi_m"][2]) for i in (0, 1)],
        "crossbar_top_z_m": round_m(by_name["crossbar"]["aabb_hi_m"][2]),
        "crossbar_height_m": round_m(2.0 * by_name["crossbar"]["half_m"][2]),
        "crossbar_square_x_m": round_m(2.0 * by_name["crossbar"]["half_m"][0]),
        "symmetric_in_y": True,
        "static": True,
        "has_joint": False,
        "is_mocap": False,
    }


def corner_assemblies(assembly: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Eight rigid +/-5 mm translations of the whole assembly, diagnostic only."""
    out = []
    for dx in (-CORNER_TRANSLATION_M, CORNER_TRANSLATION_M):
        for dy in (-CORNER_TRANSLATION_M, CORNER_TRANSLATION_M):
            for dz in (-CORNER_TRANSLATION_M, CORNER_TRANSLATION_M):
                shifted = dict(assembly)
                shifted["corner_key"] = f"dx{dx:+.3f}_dy{dy:+.3f}_dz{dz:+.3f}"
                shifted["corner_translation_m"] = [dx, dy, dz]
                shifted["components"] = [
                    {
                        **item,
                        "center_m": [
                            round_m(item["center_m"][0] + dx),
                            round_m(item["center_m"][1] + dy),
                            round_m(item["center_m"][2] + dz),
                        ],
                    }
                    for item in assembly["components"]
                ]
                out.append(shifted)
    return tuple(out)


def _fmt(values: Sequence[float]) -> str:
    return " ".join(f"{float(item):.9g}" for item in values)


def scene_xml_text(assembly: dict[str, Any]) -> str:
    """Additive scene: the frozen V3 shell plus one compiled-static pendant."""
    rgba = {"lobe": LOBE_RGBA, "stem": STEM_RGBA, "crossbar": CROSSBAR_RGBA}
    lines = [
        '<mujoco model="pact_place_corridor_v10_4">',
        "  <!-- The frozen V3 shell is included verbatim and never edited. V10.4 adds",
        "       one symmetric two-lobe pendant compiled at its final poses and sizes.",
        "       The body has no joint, freejoint, or mocap flag, so nothing moves or",
        "       resizes it at runtime and its broad-phase bounds cannot go stale.",
        "       Visible geometry IS the collision geometry; the hood class supplies",
        "       contype=8 conaffinity=15 and a matte neutral material. -->",
        f'  <include file="pact_place_corridor_v3.xml"/>',
        "  <worldbody>",
        f'    <body name="{PENDANT_BODY_V104}" pos="0 0 0">',
    ]
    for item in assembly["components"]:
        lines.append(
            f'      <geom name="{item["geom"]}" class="hood" type="box"\n'
            f'            pos="{_fmt(item["center_m"])}" size="{_fmt(item["half_m"])}"\n'
            f'            rgba="{_fmt(rgba[item["role"]])}"/>'
        )
    lines += ["    </body>", "  </worldbody>", "</mujoco>", ""]
    return "\n".join(lines)


def scene_xml_sha256(assembly: dict[str, Any]) -> str:
    return hashlib.sha256(scene_xml_text(assembly).encode()).hexdigest()
