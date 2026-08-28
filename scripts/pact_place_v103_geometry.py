#!/usr/bin/env python3
"""V10.3 static two-lobe pendant geometry and scene serialization.

The lobe shape, x/y placement, asymmetry, stem y, 12 mm stem/crossbar square,
and the crossbar's flush attachment to ``hood_top`` are inherited unchanged from
V10.2. The only search variable is ``lowest_lobe_bottom_z_m``.

Unlike V10/V10.1/V10.2, a V10.3 assembly is *compiled* into its own scene XML at
its final poses and sizes. Nothing writes ``model.geom_pos`` or
``model.geom_size`` at episode runtime, and the pendant body carries no joint,
freejoint, or mocap flag, so it cannot move and its broad-phase bounds cannot go
stale. That defect is recorded in
``diagnostics_output/pact_place_v102_preflight/contact_parity_root_cause.json``.

Pure numpy plus text serialization. No MuJoCo.
"""

from __future__ import annotations

from typing import Any, Sequence

from pact_place_v10_compound_pendant_contract import (
    CROSSBAR_HEIGHT_M,
    DEFAULT_APERTURE_WIDTH_M,
    HOOD_TOP_BOTTOM_Z_M,
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
from pact_place_v102_geometry import (
    NEGATIVE_LOBE_MAX_Y_M,
    POSITIVE_LOBE_MIN_Y_M,
    RAISED_STEM_Y_M,
    SHELF_TOP_Z_M,
    STEM_HALF_V102_M,
    STEM_SQUARE_V102_M,
    _inside_aperture_enclosure,
    raised_crossbar_for_stems,
    raised_stem_for_lobe,
)

ENVIRONMENT_VERSION_V103 = "pact_place_corridor_v10_3_static_pendant_joint_route"
SAMPLER_CLASS_V103 = "PactPlaceCorridorV103StaticPendantSampler"

PENDANT_BODY_V103 = "pact_clutter_mount_v103"
LOBE_GEOMS_V103 = (
    "pact_clutter_mount_v103_lobe_0_g",
    "pact_clutter_mount_v103_lobe_1_g",
)
STEM_GEOMS_V103 = (
    "pact_clutter_mount_v103_stem_0_g",
    "pact_clutter_mount_v103_stem_1_g",
)
CROSSBAR_GEOM_V103 = "pact_clutter_mount_v103_crossbar_g"
ALL_GEOMS_V103 = LOBE_GEOMS_V103 + STEM_GEOMS_V103 + (CROSSBAR_GEOM_V103,)

# Registered height lattice. Searched exactly once; never extended after
# observing results. The failed V10.2 bottom (1.10 m) and the old V10/V10.1
# bottom (0.82 m) are deliberately excluded.
HEIGHT_LATTICE_M = (0.92, 0.96, 1.00, 1.04)
FORBIDDEN_HEIGHTS_M = (0.82, 0.84, 1.10)
MIN_SHELF_SEPARATION_M = 0.20

NEGATIVE_LOBE_XY_M = (0.70, -0.18)
NEGATIVE_LOBE_HALF_XY_M = (0.01, 0.04)
NEGATIVE_LOBE_HALF_Z_M = 0.04
POSITIVE_LOBE_XY_M = (0.70, 0.22)
POSITIVE_LOBE_HALF_XY_M = (0.01, 0.02)
POSITIVE_LOBE_HALF_Z_M = 0.02
POSITIVE_LOBE_BOTTOM_OFFSET_M = 0.02

# Physical x faces of the pendant slab, used by the staging buffers.
PENDANT_NEAR_FACE_X_M = 0.69
PENDANT_FAR_FACE_X_M = 0.71

LOBE_RGBA = (0.43, 0.44, 0.47, 1.0)
STEM_RGBA = (0.50, 0.51, 0.53, 1.0)
CROSSBAR_RGBA = (0.36, 0.37, 0.40, 1.0)

SCENE_XML_RELATIVE_V103 = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v10_3.xml"
)
SCENE_METADATA_RELATIVE_V103 = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v10_3_metadata.json"
)


def validate_height(lowest_lobe_bottom_z_m: float) -> float:
    value = round_m(float(lowest_lobe_bottom_z_m))
    if not any(abs(value - item) <= 1e-9 for item in HEIGHT_LATTICE_M):
        raise ValueError(
            f"{value} is not in the registered V10.3 height lattice {HEIGHT_LATTICE_M}"
        )
    if any(abs(value - item) <= 1e-9 for item in FORBIDDEN_HEIGHTS_M):
        raise ValueError(f"{value} is an excluded historical height")
    if value - SHELF_TOP_Z_M < MIN_SHELF_SEPARATION_M - 1e-9:
        raise ValueError(
            f"{value} leaves {value - SHELF_TOP_Z_M:.3f} m above the shelf, "
            f"below the registered {MIN_SHELF_SEPARATION_M} m separation"
        )
    return value


def build_v103_lobe(
    *,
    side: str,
    lowest_lobe_bottom_z_m: float,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    bottom = validate_height(lowest_lobe_bottom_z_m)
    if side == "negative":
        center_xy = NEGATIVE_LOBE_XY_M
        half_xy = NEGATIVE_LOBE_HALF_XY_M
        half_z = NEGATIVE_LOBE_HALF_Z_M
    elif side == "positive":
        center_xy = POSITIVE_LOBE_XY_M
        half_xy = POSITIVE_LOBE_HALF_XY_M
        half_z = POSITIVE_LOBE_HALF_Z_M
    else:
        raise ValueError(f"side must be negative or positive, got {side!r}")
    # Both lobe centres share the negative lobe's centre z, so the thinner
    # positive lobe's bottom sits exactly 20 mm higher.
    center_z = bottom + NEGATIVE_LOBE_HALF_Z_M
    center = (float(center_xy[0]), float(center_xy[1]), float(center_z))
    half = (float(half_xy[0]), float(half_xy[1]), float(half_z))
    if side == "negative":
        if center[1] + half[1] > NEGATIVE_LOBE_MAX_Y_M + 1e-9:
            raise ValueError("negative lobe must lie wholly below y=-0.08")
    else:
        if center[1] - half[1] < POSITIVE_LOBE_MIN_Y_M - 1e-9:
            raise ValueError("positive lobe must lie wholly above y=+0.08")
    _inside_aperture_enclosure(center, half, aperture_width_m=aperture_width_m)
    return {
        "role": "lobe",
        "side": side,
        "center_m": round_vec(center),
        "half_m": round_vec(half),
        "bottom_z_m": round_m(center_z - half_z),
        "top_z_m": round_m(center_z + half_z),
        "key": tuple(round_vec(center) + round_vec(half)),
        "volume_m3": component_volume_m3(half),
    }


def _record(
    *, name: str, geom: str, slot: int, payload: dict[str, Any]
) -> dict[str, Any]:
    record = {
        "name": name,
        "role": payload["role"],
        "geom": geom,
        "slot": int(slot),
        "active": True,
        "static": True,
        "center_m": list(payload["center_m"]),
        "half_m": list(payload["half_m"]),
        "volume_m3": float(payload["volume_m3"]),
    }
    if "side" in payload:
        record["side"] = payload["side"]
    if "key" in payload:
        record["key"] = list(payload["key"])
    return record


def build_v103_assembly(
    lowest_lobe_bottom_z_m: float,
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    bottom = validate_height(lowest_lobe_bottom_z_m)
    negative = build_v103_lobe(
        side="negative",
        lowest_lobe_bottom_z_m=bottom,
        aperture_width_m=aperture_width_m,
    )
    positive = build_v103_lobe(
        side="positive",
        lowest_lobe_bottom_z_m=bottom,
        aperture_width_m=aperture_width_m,
    )
    if abs(float(positive["bottom_z_m"]) - float(negative["bottom_z_m"])
           - POSITIVE_LOBE_BOTTOM_OFFSET_M) > 1e-9:
        raise ValueError("positive lobe bottom is not 20 mm above the negative lobe")
    ordered = [negative, positive]
    stems = [
        raised_stem_for_lobe(lobe, aperture_width_m=aperture_width_m)
        for lobe in ordered
    ]
    crossbar = raised_crossbar_for_stems(stems, aperture_width_m=aperture_width_m)
    if not connected_stems_and_crossbar(stems, crossbar):
        raise ValueError("stems and crossbar are not a connected assembly")
    if not hood_top_attachment_ok(crossbar):
        raise ValueError("crossbar is not attached to hood_top at z=1.515")
    components = [
        _record(name="lobe_0", geom=LOBE_GEOMS_V103[0], slot=0, payload=ordered[0]),
        _record(name="stem_0", geom=STEM_GEOMS_V103[0], slot=0, payload=stems[0]),
        _record(name="lobe_1", geom=LOBE_GEOMS_V103[1], slot=1, payload=ordered[1]),
        _record(name="stem_1", geom=STEM_GEOMS_V103[1], slot=1, payload=stems[1]),
        _record(name="crossbar", geom=CROSSBAR_GEOM_V103, slot=0, payload=crossbar),
    ]
    for item in components:
        if forbidden_static_overlap(item, allow_hood_top=item["role"] == "crossbar"):
            raise ValueError(f"{item['name']} overlaps forbidden static geometry")
    stem_ys = tuple(sorted(float(item["center_m"][1]) for item in stems))
    if any(abs(stem_ys[i] - RAISED_STEM_Y_M[i]) > 1e-9 for i in range(2)):
        raise ValueError(f"stem y drifted: {stem_ys} != {RAISED_STEM_Y_M}")
    assembly = {
        "environment_version": ENVIRONMENT_VERSION_V103,
        "topology": "two_lobe_static",
        "assembly_id": f"v103_static_two_lobe_bottom_{bottom:.3f}",
        "lowest_lobe_bottom_z_m": bottom,
        "body": PENDANT_BODY_V103,
        "support": "ceiling",
        "static": True,
        "has_joint": False,
        "has_freejoint": False,
        "is_mocap": False,
        "runtime_repositioned": False,
        "identical_on_both_panel_sides": True,
        "active_on": ["inbound_empty", "outbound_loaded"],
        "center_x_m": round_m(NEGATIVE_LOBE_XY_M[0]),
        "components": components,
        "stem_square_m": round_m(STEM_SQUARE_V102_M),
        "crossbar_square_x_m": round_m(STEM_SQUARE_V102_M),
        "collision_and_visible_stem_identical": True,
        "visual_only_sleeve": False,
        "shelf_top_z_m": round_m(SHELF_TOP_Z_M),
        "shelf_to_pendant_gap_m": round_m(bottom - SHELF_TOP_Z_M),
        "volume_m3": 0.0,
        "union_fixture": {},
    }
    assembly["volume_m3"] = assembly_volume_m3(assembly)
    assembly["union_fixture"] = union_fixture(assembly)
    return assembly


def enumerate_v103_assemblies() -> tuple[dict[str, Any], ...]:
    return tuple(build_v103_assembly(value) for value in HEIGHT_LATTICE_M)


def assembly_expectations(assembly: dict[str, Any]) -> dict[str, Any]:
    lows = []
    for item in assembly["components"]:
        lo, _hi = component_aabb(item["center_m"], item["half_m"])
        lows.append(float(lo[2]))
    lobes = {item["side"]: item for item in assembly["components"] if item["role"] == "lobe"}
    stems = sorted(
        (item for item in assembly["components"] if item["role"] == "stem"),
        key=lambda item: float(item["center_m"][1]),
    )
    crossbar = next(item for item in assembly["components"] if item["role"] == "crossbar")
    _lo, crossbar_hi = component_aabb(crossbar["center_m"], crossbar["half_m"])
    return {
        "lowest_pendant_z_m": round_m(min(lows)),
        "negative_lobe_bottom_z_m": round_m(float(lobes["negative"]["center_m"][2])
                                            - float(lobes["negative"]["half_m"][2])),
        "positive_lobe_bottom_z_m": round_m(float(lobes["positive"]["center_m"][2])
                                            - float(lobes["positive"]["half_m"][2])),
        "shelf_top_z_m": round_m(SHELF_TOP_Z_M),
        "shelf_to_pendant_gap_m": round_m(min(lows) - SHELF_TOP_Z_M),
        "stem_center_y_m": [round_m(float(item["center_m"][1])) for item in stems],
        "stem_square_m": [round_m(2.0 * float(item["half_m"][0])) for item in stems],
        "stem_square_y_m": [round_m(2.0 * float(item["half_m"][1])) for item in stems],
        "crossbar_top_z_m": round_m(float(crossbar_hi[2])),
        "crossbar_height_m": round_m(2.0 * float(crossbar["half_m"][2])),
        "crossbar_square_x_m": round_m(2.0 * float(crossbar["half_m"][0])),
        "hood_top_bottom_z_m": round_m(HOOD_TOP_BOTTOM_Z_M),
        "registered_crossbar_height_m": round_m(CROSSBAR_HEIGHT_M),
        "static": True,
        "has_joint": False,
        "is_mocap": False,
    }


def _fmt(values: Sequence[float]) -> str:
    return " ".join(f"{float(item):.9g}" for item in values)


def scene_xml_text(assembly: dict[str, Any]) -> str:
    """Serialize the selected static assembly into a compilable scene."""
    rgba = {"lobe": LOBE_RGBA, "stem": STEM_RGBA, "crossbar": CROSSBAR_RGBA}
    lines = [
        '<mujoco model="pact_place_corridor_v10_3">',
        "  <!-- V5 shell verbatim, plus one static two-lobe pendant compiled at its",
        "       final poses and sizes. The body carries no joint, freejoint, or mocap",
        "       flag, so nothing repositions or resizes it at runtime and its",
        "       broad-phase bounds cannot go stale. Visible geometry IS the collision",
        "       geometry; the hood class supplies contype=8 conaffinity=15. -->",
        f"  <!-- lowest_lobe_bottom_z_m = {float(assembly['lowest_lobe_bottom_z_m']):.9g} -->",
        '  <include file="pact_place_corridor_v5.xml"/>',
        "  <worldbody>",
        f'    <body name="{PENDANT_BODY_V103}" pos="0 0 0">',
    ]
    for item in assembly["components"]:
        lines.append(
            f'      <geom name="{item["geom"]}" class="hood" type="box"\n'
            f'            pos="{_fmt(item["center_m"])}" size="{_fmt(item["half_m"])}"\n'
            f'            rgba="{_fmt(rgba[item["role"]])}"/>'
        )
    lines += ["    </body>", "  </worldbody>", "</mujoco>", ""]
    return "\n".join(lines)


def write_scene_xml(path, assembly: dict[str, Any]) -> str:
    from pathlib import Path

    from pact_place_corridor_contract import sha256_file

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(scene_xml_text(assembly))
    metadata = Path(str(target).replace(".xml", "_metadata.json"))
    metadata.write_text('{ "objects": {} }\n')
    return sha256_file(target)
