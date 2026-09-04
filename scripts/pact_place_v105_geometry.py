#!/usr/bin/env python3
"""V10.5 pendant geometry: the V10.4 shape, parameterised by (x, r, d).

The shape and height are reused verbatim from V10.4 — lobe bottom z=0.98,
crossbar top flush with ``hood_top`` at z=1.515. Only the depth ``x``, the
symmetric lobe magnitude ``r``, and the rigid lateral offset ``d`` are free,
and all three come from a preregistered lattice that is never widened.

Every scene here is compiled-static: geoms are written at their final poses
into a body with no joint, freejoint, mocap flag, or actuator, and nothing
writes ``geom_pos``/``geom_size``/``geom_aabb``/``geom_rbound`` at runtime.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

ENVIRONMENT_VERSION_V105 = "pact_place_corridor_v10_5_v95_clutter_static_pendant"
CONTRACT_VERSION_V105 = "pact_place_v105_v95_clutter_static_pendant_v1"
SAMPLER_CLASS_V105 = "PactPlaceCorridorV105Sampler"

BASE_SCENE_RELATIVE_V105 = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v5.xml"
)
BASE_SCENE_NAME_V105 = "pact_place_corridor_v5.xml"

PENDANT_BODY_V105 = "pact_clutter_mount_v105"
COMPONENT_NAMES_V105 = ("lobe_0", "lobe_1", "stem_0", "stem_1", "crossbar")
GEOM_NAMES_V105 = tuple(f"{PENDANT_BODY_V105}_{name}_g" for name in COMPONENT_NAMES_V105)
ALL_GEOMS_V105 = GEOM_NAMES_V105

# Fixed shape, inherited from V10.4. Not free parameters; not searchable.
LOBE_HALF_M = (0.010, 0.010, 0.030)
LOBE_CENTER_Z_M = 1.01
STEM_HALF_XY_M = 0.006
STEM_SQUARE_M = 0.012
STEM_CENTER_Z_M = 1.2725
STEM_HALF_Z_M = 0.2325
CROSSBAR_CENTER_Z_M = 1.510
CROSSBAR_HALF_X_M = 0.006
CROSSBAR_HALF_Z_M = 0.005
CROSSBAR_HALF_Y_PAD_M = 0.016

LOBE_BOTTOM_Z_M = 0.98
LOBE_TOP_Z_M = 1.04
STEM_TOP_Z_M = 1.505
HOOD_TOP_BOTTOM_Z_M = 1.515
SHELF_TOP_Z_M = 0.72

PENDANT_RGBA = (0.55, 0.56, 0.58, 1.0)

# ---------------------------------------------------------------------------
# Preregistered lattice. Never widened, densified, or re-centred on results.
# ---------------------------------------------------------------------------
LATTICE_X_M: tuple[float, ...] = (0.740, 0.760, 0.780, 0.800)
LATTICE_R_M: tuple[float, ...] = (
    0.290,
    0.295,
    0.300,
    0.305,
    0.310,
    0.315,
    0.320,
    0.325,
)
POSE_OFFSETS_M: dict[str, float] = {"neg5": -0.005, "center": 0.000, "pos5": +0.005}
POSE_IDS: tuple[str, ...] = ("neg5", "center", "pos5")

RISK_BAND_M = (0.015, 0.035)
CLEARANCE_FLOOR_M = 0.015
POSE_ORDERING_MIN_SEPARATION_M = 0.005


def round_m(value: float) -> float:
    return float(round(float(value), 9))


def component_aabb(center_m: Sequence[float], half_m: Sequence[float]):
    lo = [round_m(float(c) - float(h)) for c, h in zip(center_m, half_m)]
    hi = [round_m(float(c) + float(h)) for c, h in zip(center_m, half_m)]
    return lo, hi


def component_specs(x_m: float, r_m: float, d_m: float) -> tuple[dict[str, Any], ...]:
    """The five components for one (x, r, d), exactly as the plan tabulates."""
    x = float(x_m)
    r = float(r_m)
    d = float(d_m)
    return (
        {
            "name": "lobe_0",
            "role": "lobe",
            "side": "negative",
            "center_m": (x, round_m(-r + d), LOBE_CENTER_Z_M),
            "half_m": LOBE_HALF_M,
        },
        {
            "name": "lobe_1",
            "role": "lobe",
            "side": "positive",
            "center_m": (x, round_m(r + d), LOBE_CENTER_Z_M),
            "half_m": LOBE_HALF_M,
        },
        {
            "name": "stem_0",
            "role": "stem",
            "side": "negative",
            "center_m": (x, round_m(-r - 0.010 + d), STEM_CENTER_Z_M),
            "half_m": (STEM_HALF_XY_M, STEM_HALF_XY_M, STEM_HALF_Z_M),
        },
        {
            "name": "stem_1",
            "role": "stem",
            "side": "positive",
            "center_m": (x, round_m(r + 0.010 + d), STEM_CENTER_Z_M),
            "half_m": (STEM_HALF_XY_M, STEM_HALF_XY_M, STEM_HALF_Z_M),
        },
        {
            "name": "crossbar",
            "role": "crossbar",
            "center_m": (x, round_m(d), CROSSBAR_CENTER_Z_M),
            "half_m": (
                CROSSBAR_HALF_X_M,
                round_m(r + CROSSBAR_HALF_Y_PAD_M),
                CROSSBAR_HALF_Z_M,
            ),
        },
    )


def build_assembly(x_m: float, r_m: float, d_m: float, *, pose_id: str | None = None):
    """One assembly, with every derived fact asserted rather than assumed."""
    components: list[dict[str, Any]] = []
    for item in component_specs(x_m, r_m, d_m):
        lo, hi = component_aabb(item["center_m"], item["half_m"])
        record = {
            "name": item["name"],
            "role": item["role"],
            "geom": f"{PENDANT_BODY_V105}_{item['name']}_g",
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

    for slot in (0, 1):
        lobe = by_name[f"lobe_{slot}"]
        if abs(lobe["aabb_lo_m"][2] - LOBE_BOTTOM_Z_M) > 1e-9:
            raise ValueError(f"{lobe['name']} bottom is not {LOBE_BOTTOM_Z_M} m")
        if abs(lobe["aabb_hi_m"][2] - LOBE_TOP_Z_M) > 1e-9:
            raise ValueError(f"{lobe['name']} top is not {LOBE_TOP_Z_M} m")
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
    if abs(2.0 * bar["half_m"][2] - 2.0 * CROSSBAR_HALF_Z_M) > 1e-9:
        raise ValueError("crossbar height is wrong")
    if abs(bar["aabb_hi_m"][2] - HOOD_TOP_BOTTOM_Z_M) > 1e-9:
        raise ValueError("crossbar top is not flush with hood_top at 1.515")
    if abs(bar["aabb_lo_m"][2] - STEM_TOP_Z_M) > 1e-9:
        raise ValueError("crossbar bottom does not meet the stem tops")
    for slot in (0, 1):
        stem = by_name[f"stem_{slot}"]
        lo = float(bar["center_m"][1]) - float(bar["half_m"][1])
        hi = float(bar["center_m"][1]) + float(bar["half_m"][1])
        if not (lo - 1e-9 <= float(stem["center_m"][1]) <= hi + 1e-9):
            raise ValueError(f"crossbar does not span stem_{slot}")

    # Connectedness and the shared depth column.
    for name in COMPONENT_NAMES_V105:
        if abs(float(by_name[name]["center_m"][0]) - float(x_m)) > 1e-9:
            raise ValueError(f"{name} is not at the assembly depth {x_m}")

    # The lateral offset translates the whole assembly rigidly: with d removed,
    # the shape is symmetric about y=0.
    if abs(
        (float(by_name["lobe_0"]["center_m"][1]) - float(d_m))
        + (float(by_name["lobe_1"]["center_m"][1]) - float(d_m))
    ) > 1e-9:
        raise ValueError("lobes are not symmetric about the offset centre")

    inner_face = round_m(
        min(
            abs(float(by_name["lobe_0"]["center_m"][1]) + float(LOBE_HALF_M[1])),
            abs(float(by_name["lobe_1"]["center_m"][1]) - float(LOBE_HALF_M[1])),
        )
    )
    return {
        "environment_version": ENVIRONMENT_VERSION_V105,
        "contract_version": CONTRACT_VERSION_V105,
        "assembly_id": f"v105_x{int(round(x_m * 1000))}_r{int(round(r_m * 1000))}"
        f"_d{int(round(d_m * 1000)):+d}",
        "body": PENDANT_BODY_V105,
        "support": "ceiling",
        "pose_id": pose_id,
        "x_m": round_m(x_m),
        "r_m": round_m(r_m),
        "d_m": round_m(d_m),
        "static": True,
        "has_joint": False,
        "has_freejoint": False,
        "is_mocap": False,
        "has_actuator": False,
        "runtime_repositioned": False,
        "runtime_bound_refresh": False,
        "connected": True,
        "collision_and_visible_identical": True,
        "visual_only_sleeve": False,
        "components": components,
        "lobe_bottom_z_m": round_m(LOBE_BOTTOM_Z_M),
        "lobe_top_z_m": round_m(LOBE_TOP_Z_M),
        "crossbar_top_z_m": round_m(bar["aabb_hi_m"][2]),
        "shelf_top_z_m": round_m(SHELF_TOP_Z_M),
        "shelf_to_lobe_gap_m": round_m(LOBE_BOTTOM_Z_M - SHELF_TOP_Z_M),
        "min_inner_lobe_face_abs_y_m": inner_face,
        "volume_m3": float(sum(item["volume_m3"] for item in components)),
    }


def bundle_assemblies(x_m: float, r_m: float) -> dict[str, dict[str, Any]]:
    """The complete three-pose production bundle for one (x, r)."""
    return {
        pose_id: build_assembly(x_m, r_m, POSE_OFFSETS_M[pose_id], pose_id=pose_id)
        for pose_id in POSE_IDS
    }


def lattice_candidates() -> tuple[tuple[float, float], ...]:
    return tuple((x, r) for x in LATTICE_X_M for r in LATTICE_R_M)


def _fmt(values: Sequence[float]) -> str:
    out = []
    for value in values:
        text = f"{float(value):.9f}".rstrip("0").rstrip(".")
        out.append(text if text not in ("", "-0") else "0")
    return " ".join(out)


def scene_model_name(assembly: dict[str, Any]) -> str:
    return f"pact_place_corridor_v10_5_{assembly['assembly_id']}"


def scene_xml_text(assembly: dict[str, Any] | None) -> str:
    """Additive scene: the frozen V5 shell plus one compiled-static pendant.

    ``assembly=None`` produces the no-pendant counterfactual control scene,
    which is byte-derived from the same generator so the only difference is
    the pendant body itself.
    """
    if assembly is None:
        return "\n".join(
            [
                '<mujoco model="pact_place_corridor_v10_5_no_pendant">',
                "  <!-- V10.5 no-pendant counterfactual. The frozen V5 shell,",
                "       included verbatim, with no pendant body at all. Used only",
                "       to isolate the pendant's contribution to the proximity",
                "       tensor at byte-identical robot/target/clutter state. -->",
                f'  <include file="{BASE_SCENE_NAME_V105}"/>',
                "</mujoco>",
                "",
            ]
        )
    lines = [
        f'<mujoco model="{scene_model_name(assembly)}">',
        "  <!-- The frozen V5 shell is included verbatim and never edited. V10.5",
        "       adds one connected two-lobe pendant compiled at its final poses",
        "       and sizes. The body has no joint, freejoint, mocap flag, or",
        "       actuator, so nothing moves or resizes it at runtime and its",
        "       broad-phase bounds cannot go stale. Visible geometry IS the",
        "       collision geometry; the hood class supplies contype/conaffinity",
        "       and a matte neutral material. -->",
        f"  <!-- pose_id={assembly['pose_id']} x={assembly['x_m']} "
        f"r={assembly['r_m']} d={assembly['d_m']} -->",
        f'  <include file="{BASE_SCENE_NAME_V105}"/>',
        "  <worldbody>",
        f'    <body name="{PENDANT_BODY_V105}" pos="0 0 0">',
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
    import json

    return hashlib.sha256(
        json.dumps(assembly, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


__all__ = [
    "ALL_GEOMS_V105",
    "BASE_SCENE_NAME_V105",
    "BASE_SCENE_RELATIVE_V105",
    "CLEARANCE_FLOOR_M",
    "COMPONENT_NAMES_V105",
    "CONTRACT_VERSION_V105",
    "ENVIRONMENT_VERSION_V105",
    "GEOM_NAMES_V105",
    "LATTICE_R_M",
    "LATTICE_X_M",
    "PENDANT_BODY_V105",
    "POSE_IDS",
    "POSE_OFFSETS_M",
    "POSE_ORDERING_MIN_SEPARATION_M",
    "RISK_BAND_M",
    "SAMPLER_CLASS_V105",
    "assembly_sha256",
    "build_assembly",
    "bundle_assemblies",
    "component_specs",
    "lattice_candidates",
    "round_m",
    "scene_model_name",
    "scene_xml_sha256",
    "scene_xml_text",
]
