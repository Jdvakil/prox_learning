#!/usr/bin/env python3
"""Manifest and layout helpers for the non-authorizing V9.3 environment build."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PALETTE_PATH = ROOT / "diagnostics_output/pact_place_v9_v0b/palette_v9_1.json"
SITING_PATH = ROOT / "diagnostics_output/pact_place_v9_v0c/siting.json"
SCENE_RELATIVE = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"
)
SAMPLER_CLASS = "PactPlaceCorridorV93Sampler"
N_ROWS = 24
MIN_CLEAN_SUCCESSES = 20
GATE_MASTER_SEED = 2026083101
REVIEW_MASTER_SEED = 2026083102
V9_REVIEW_STREAM = "pact-place-v9.3-human-review"
V9_GATE_STREAM = "pact-place-v9.3-phase0-gate"
SHELF_TOP_Z = 0.72
APERTURE_WIDTH = 0.85
TUBE_X0 = 0.58
MIN_DEPTH = 0.20
APERTURE_EDGE_RESERVE_M = 0.02
OUTBOUND_ENVELOPE_HALF_Y_M = 0.15
ROUTE_BLOCKER_SAFE_GAP_M = 0.04
MIN_FORCED_BOW_M = 0.04
NOMINAL_OUTBOUND_START_XY_M = (0.75, 0.02)
NOMINAL_OUTBOUND_END_XY_M = (0.44, 0.30)
PANEL_X_M = 0.615
PANEL_HALF_X_M = 0.055
PANEL_HALF_Y_M = 0.240
PANEL_INNER_FACE_Y_M = 0.100
PANEL_X_JITTER_MAX_M = 0.015
PANEL_FACE_JITTER_MAX_M = 0.005
PANEL_SAFE_GAP_M = 0.14

# V9.3 keeps the original per-episode panel and uses two substantial vessels in
# a real 2-D chicane. Coordinates are identical in paired left/right rows; panel
# side therefore cannot be inferred from visible clutter. Both x and y vary by
# family, before independent paired +/-20 mm row jitter is applied.
LAYOUT_FAMILIES: dict[str, dict[str, Any]] = {
    "F0_target_side_stagger": {
        "inbound_vessel_xy_m": (0.565, -0.010),
        "outbound_vessel_xy_m": (0.680, 0.000),
    },
    "F1_inner_panel_stagger": {
        "inbound_vessel_xy_m": (0.575, 0.010),
        "outbound_vessel_xy_m": (0.695, 0.000),
    },
    "F2_outer_panel_stagger": {
        "inbound_vessel_xy_m": (0.563, -0.015),
        "outbound_vessel_xy_m": (0.680, 0.000),
    },
    "F3_aperture_side_stagger": {
        "inbound_vessel_xy_m": (0.585, 0.015),
        "outbound_vessel_xy_m": (0.695, 0.000),
    },
}
DEFAULT_LAYOUT_FAMILY = next(iter(LAYOUT_FAMILIES))
WORKSPACE_LOW_XYZ = (0.50, -0.43, SHELF_TOP_Z)
WORKSPACE_HIGH_XYZ = (1.34, 0.43, 1.50)
MIN_OBJECT_GAP_M = 0.010


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def load_palette(path: Path = PALETTE_PATH) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("authorizes_gate") is not False:
        raise ValueError("V9 palette unexpectedly authorizes the gate")
    palette = [dict(item) for item in list(document.get("palette") or [])]
    if len(palette) < 8 or len(palette) > 12:
        raise ValueError("V9 palette size is outside the 2+6-10 contract")
    # The rejected V9.2 called the 15.7 mm-wide Candle_2 a vessel.  The first
    # V9.3 draft promoted a 151.7 mm can, but raw 40-camera counterfactuals
    # proved that it was not visible on the inbound decision leg.  Use the
    # independently settled Soap_Bottle_1 (180.2 mm tall, 68.1 mm base) as the
    # second real vessel; Soap_Bottle_30 remains the outbound vessel.
    by_slot = {str(item["slot"]): item for item in palette}
    records = {str(item.get("uid")): item for item in document.get("records") or []}
    inbound_record = records.get("Soap_Bottle_1")
    if not inbound_record or not inbound_record.get("accepted"):
        raise ValueError("V9.3 source palette is missing settled Soap_Bottle_1")
    dimensions = [float(value) for value in inbound_record["collision_dimensions_m"]]
    by_slot["00"]["role"] = "decor"
    by_slot["06"] = {
        "slot": "06",
        "slot_class": "prop",
        "role": "inbound_vessel",
        "uid": "Soap_Bottle_1",
        "category": str(inbound_record["category"]),
        "dimensions_m": dimensions,
        "annotation_dimensions_m": [
            float(value) for value in inbound_record["dimensions_m"]
        ],
        "half_m": [value / 2.0 for value in dimensions],
        "max_dimension_m": max(dimensions),
        "support": "shelf_standing",
        "quat_wxyz": [2**-0.5, 2**-0.5, 0.0, 0.0],
        "body_prefix": "pact_clutter_06/",
    }
    by_slot["01"]["role"] = "outbound_vessel"
    palette = [by_slot[str(item["slot"])] for item in palette]
    derived = dict(document)
    derived["palette"] = palette
    derived["derived_for_environment_version"] = "pact_place_corridor_v9_3"
    derived["role_changes_from_source"] = {
        "00": "inbound_vessel_to_decor",
        "06": "decor_can_replaced_by_settled_Soap_Bottle_1_inbound_vessel",
    }
    return derived


def _vessel_for_role(palette: list[dict[str, Any]], role: str) -> dict[str, Any]:
    matches = [item for item in palette if str(item.get("role")) == role]
    if len(matches) != 1:
        raise ValueError(f"palette must contain exactly one {role}")
    return matches[0]


def route_blocker_metrics(layout: dict[str, Any]) -> dict[str, Any]:
    """Measure whether the nominal loaded route is blocked but laterally clearable."""
    blockers = [
        item
        for item in list(layout.get("objects") or [])
        if str(item.get("palette_slot")) == str(layout.get("route_blocker_slot"))
    ]
    if len(blockers) != 1:
        raise ValueError("layout must contain exactly one route blocker")
    blocker = blockers[0]
    center = tuple(map(float, blocker["center_m"][:2]))
    half = tuple(map(float, blocker["half_m"][:2]))
    start = NOMINAL_OUTBOUND_START_XY_M
    end = NOMINAL_OUTBOUND_END_XY_M
    delta_x = end[0] - start[0]
    t_cross = (center[0] - start[0]) / delta_x
    if not 0.02 < t_cross < 0.98:
        raise ValueError("route blocker is not crossed on the nominal loaded leg")
    crossing_y = start[1] + t_cross * (end[1] - start[1])
    bow_direction = str(layout.get("expected_bow_direction") or "")
    if bow_direction not in {"+y", "-y"}:
        raise ValueError("layout must declare an expected panel-selected bow direction")
    desired_side = 1.0 if bow_direction == "+y" else -1.0
    open_face_y = center[1] + desired_side * half[1]
    straight_clearance = (
        desired_side * (crossing_y - open_face_y) - OUTBOUND_ENVELOPE_HALF_Y_M
    )
    required_bow = ROUTE_BLOCKER_SAFE_GAP_M - straight_clearance
    waypoint_y = crossing_y + desired_side * required_bow
    lateral_limit = APERTURE_WIDTH / 2.0 - OUTBOUND_ENVELOPE_HALF_Y_M - APERTURE_EDGE_RESERVE_M
    return {
        "nominal_start_xy_m": list(start),
        "nominal_end_xy_m": list(end),
        "crossing_fraction": float(t_cross),
        "crossing_y_m": float(crossing_y),
        "straight_envelope_clearance_m": float(straight_clearance),
        "required_bow_m": float(required_bow),
        "planned_waypoint_y_m": float(waypoint_y),
        "lateral_limit_m": float(lateral_limit),
        "bow_direction": bow_direction,
        "direct_route_blocked": bool(straight_clearance < 0.0),
        "detour_admitted": bool(
            required_bow >= MIN_FORCED_BOW_M and abs(waypoint_y) <= lateral_limit
        ),
    }


def panel_corridor_metrics(layout: dict[str, Any]) -> dict[str, Any]:
    """Check that the active panel and centred bottle leave one safe lane."""
    side_name = str(layout.get("intrusion_side") or "")
    if side_name not in {"left", "right"}:
        raise ValueError("layout must bind exactly one left/right intrusion panel")
    side = 1.0 if side_name == "left" else -1.0
    expected_direction = "-y" if side_name == "left" else "+y"
    blockers = [
        item
        for item in list(layout.get("objects") or [])
        if str(item.get("palette_slot")) == str(layout.get("route_blocker_slot"))
    ]
    if len(blockers) != 1:
        raise ValueError("layout must contain exactly one route blocker")
    blocker = blockers[0]
    blocker_center = tuple(map(float, blocker["center_m"][:2]))
    blocker_half = tuple(map(float, blocker["half_m"][:2]))
    blocker_offset_toward_panel = side * blocker_center[1]
    worst_panel_face = PANEL_INNER_FACE_Y_M - PANEL_FACE_JITTER_MAX_M
    panel_lane_center = (
        OUTBOUND_ENVELOPE_HALF_Y_M + PANEL_SAFE_GAP_M - worst_panel_face
    )
    blocker_lane_center = (
        OUTBOUND_ENVELOPE_HALF_Y_M
        + ROUTE_BLOCKER_SAFE_GAP_M
        + blocker_half[1]
        - blocker_offset_toward_panel
    )
    required_lane_center = max(panel_lane_center, blocker_lane_center)
    lateral_limit = (
        APERTURE_WIDTH / 2.0
        - OUTBOUND_ENVELOPE_HALF_Y_M
        - APERTURE_EDGE_RESERVE_M
    )

    panel_x_low = PANEL_X_M - PANEL_X_JITTER_MAX_M - PANEL_HALF_X_M
    panel_x_high = PANEL_X_M + PANEL_X_JITTER_MAX_M + PANEL_HALF_X_M
    blocker_x_low = blocker_center[0] - blocker_half[0]
    blocker_x_high = blocker_center[0] + blocker_half[0]
    x_overlap = min(panel_x_high, blocker_x_high) - max(panel_x_low, blocker_x_low)
    panel_blocker_surface_gap = worst_panel_face - (
        blocker_offset_toward_panel + blocker_half[1]
    )
    return {
        "intrusion_side": side_name,
        "expected_bow_direction": expected_direction,
        "panel_active": bool(layout.get("legacy_panel_active")),
        "panel_x_range_with_jitter_m": [float(panel_x_low), float(panel_x_high)],
        "panel_inner_face_worst_case_m": float(worst_panel_face),
        "blocker_center_xy_m": list(blocker_center),
        "blocker_x_range_m": [float(blocker_x_low), float(blocker_x_high)],
        "panel_blocker_x_overlap_m": float(max(0.0, x_overlap)),
        "panel_blocker_surface_gap_m": float(panel_blocker_surface_gap),
        "required_lane_center_offset_m": float(required_lane_center),
        "lateral_limit_m": float(lateral_limit),
        "corridor_margin_m": float(lateral_limit - required_lane_center),
        "detour_admitted": bool(
            required_lane_center <= lateral_limit
            and (x_overlap <= 0.0 or panel_blocker_surface_gap >= MIN_OBJECT_GAP_M)
        ),
    }


def build_layout(
    palette_document: dict[str, Any],
    *,
    family_id: str = DEFAULT_LAYOUT_FAMILY,
    intrusion_side: str = "left",
    inbound_center_xy: tuple[float, float] | None = None,
    outbound_center_xy: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Build a staggered bench layout from measured collision dimensions.

    ``inbound_center_xy`` and ``outbound_center_xy`` remain available to the
    measurement scripts, but production rows select one of the frozen layout
    families.  Slot 01 is the route blocker; slot 00 is a tall visual/sensing
    control parked behind and lateral to the target.
    """
    if family_id not in LAYOUT_FAMILIES:
        raise ValueError(f"unknown V9 layout family: {family_id}")
    palette = list(palette_document["palette"])
    if intrusion_side not in {"left", "right"}:
        raise ValueError(f"intrusion_side must be left or right, got {intrusion_side!r}")
    family = LAYOUT_FAMILIES[family_id]
    inbound_xy = tuple(
        inbound_center_xy or tuple(map(float, family["inbound_vessel_xy_m"]))
    )
    blocker_xy = tuple(
        outbound_center_xy or tuple(map(float, family["outbound_vessel_xy_m"]))
    )
    expected_bow_direction = "-y" if intrusion_side == "left" else "+y"
    positions: dict[str, tuple[float, float]] = {
        "00": (0.820, -0.350),
        "01": blocker_xy,
        "02": (0.840, 0.310),
        "03": (0.980, -0.220),
        "04": (1.090, 0.300),
        "05": (1.210, -0.280),
        "06": inbound_xy,
        "07": (1.060, 0.020),
        "08": (1.180, 0.090),
        "09": (1.170, -0.100),
        "10": (1.300, 0.310),
        "11": (1.300, -0.310),
    }
    objects: list[dict[str, Any]] = []
    for item in palette:
        slot = str(item["slot"])
        dimensions = [float(value) for value in item["dimensions_m"]]
        half = [value / 2.0 for value in dimensions]
        x, y = positions.get(slot, (0.68, 0.0))
        objects.append(
            {
                "palette_slot": slot,
                "uid": str(item["uid"]),
                "role": str(item["role"]),
                "category": str(item["category"]),
                "support": "bench_standing",
                "center_m": [float(x), float(y), float(SHELF_TOP_Z + half[2])],
                "half_m": half,
                "quat_wxyz": [float(value) for value in item["quat_wxyz"]],
                "size_class": (
                    "small"
                    if max(dimensions) <= 0.10
                    else "medium"
                    if max(dimensions) <= 0.18
                    else "large"
                ),
            }
        )
    layout = {
        "layout_id": f"v9_3_panel_{intrusion_side}_{family_id}",
        "layout_family_id": family_id,
        "intrusion_side": intrusion_side,
        "objects": objects,
        "inbound_vessel_slot": "06",
        "outbound_vessel_slot": "01",
        "route_blocker_slot": "01",
        "route_blocker_center_xy_m": list(map(float, blocker_xy)),
        "inbound_vessel_center_xy_m": list(map(float, inbound_xy)),
        "expected_bow_direction": expected_bow_direction,
        "shelf_top_z_m": SHELF_TOP_Z,
        "support": "bench_standing",
        "workspace_bounds_m": [list(WORKSPACE_LOW_XYZ), list(WORKSPACE_HIGH_XYZ)],
        "legacy_panel_active": True,
    }
    layout["nominal_route_metrics"] = route_blocker_metrics(layout)
    layout["panel_corridor_metrics"] = panel_corridor_metrics(layout)
    validate_layout(layout)
    return layout


def validate_layout(layout: dict[str, Any]) -> None:
    """Reject line layouts, overlaps, and objects outside the real bench."""
    objects = list(layout.get("objects") or [])
    if len(objects) < 8:
        raise ValueError("V9.2 layout must activate both vessels and 6-10 decor objects")
    centers = [tuple(map(float, item["center_m"])) for item in objects]
    if max(center[0] for center in centers) - min(center[0] for center in centers) < 0.40:
        raise ValueError("V9.2 objects collapsed into a transverse line")
    if len({round(center[0], 3) for center in centers}) < 6:
        raise ValueError("V9.2 layout lacks depth diversity")
    workspace_low = WORKSPACE_LOW_XYZ
    workspace_high = WORKSPACE_HIGH_XYZ
    for item, center in zip(objects, centers):
        half = tuple(map(float, item["half_m"]))
        if any(center[k] - half[k] < workspace_low[k] - 1e-6 for k in range(3)):
            raise ValueError(f"object {item['palette_slot']} escapes the low workspace bound")
        if any(center[k] + half[k] > workspace_high[k] + 1e-6 for k in range(3)):
            raise ValueError(f"object {item['palette_slot']} escapes the high workspace bound")
    for left_index, left in enumerate(objects):
        lc = tuple(map(float, left["center_m"]))
        lh = tuple(map(float, left["half_m"]))
        for right in objects[left_index + 1 :]:
            rc = tuple(map(float, right["center_m"]))
            rh = tuple(map(float, right["half_m"]))
            separated = any(abs(lc[k] - rc[k]) >= lh[k] + rh[k] + MIN_OBJECT_GAP_M for k in (0, 1))
            if not separated:
                raise ValueError(
                    f"layout objects overlap: {left['palette_slot']} and {right['palette_slot']}"
                )
    metrics = route_blocker_metrics(layout)
    if not metrics["direct_route_blocked"]:
        raise ValueError("route blocker does not obstruct the nominal loaded envelope")
    if not metrics["detour_admitted"]:
        raise ValueError("route blocker has no admitted lateral detour")
    if metrics["bow_direction"] != layout.get("expected_bow_direction"):
        raise ValueError("route blocker bow direction disagrees with its family contract")
    corridor = panel_corridor_metrics(layout)
    if layout.get("legacy_panel_active") is not True:
        raise ValueError("V9.2 requires one active legacy side panel")
    if corridor["expected_bow_direction"] != layout.get("expected_bow_direction"):
        raise ValueError("panel side and expected bow direction disagree")
    if not corridor["detour_admitted"]:
        raise ValueError("active panel and blocker close the only safe corridor")


def _seed(stream: str, index: int, master_seed: int) -> tuple[int, int]:
    digest = hashlib.sha256(f"{stream}:{int(master_seed)}:{int(index)}".encode()).digest()
    return int.from_bytes(digest[:4], "big"), int.from_bytes(digest[:8], "big")


def _rows(
    *,
    stream: str,
    master_seed: int,
    palette: list[dict[str, Any]],
    layouts: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rng = random.Random(int(master_seed))
    # Shuffle 12 paired cells, never individual sides. Each pair has identical
    # visible geometry and identical vessel jitter under left/right panels.
    pairs = [
        (family_id, repeat)
        for family_id in layouts
        for repeat in range(N_ROWS // (2 * len(layouts)))
    ]
    rng.shuffle(pairs)
    rows = []
    for pair_index, (family_id, repeat) in enumerate(pairs):
        vessel_slots = ("01", "06")
        x_jitter = {
            "01": round(rng.uniform(-0.020, 0.020), 9),
            "06": round(rng.uniform(-0.005, 0.005), 9),
        }
        y_jitter = {
            "01": round(rng.uniform(-0.005, 0.005), 9),
            "06": round(rng.uniform(-0.010, 0.010), 9),
        }
        panel_x_jitter = round(rng.uniform(-0.015, 0.015), 9)
        panel_face_jitter = round(rng.uniform(-0.005, 0.005), 9)
        for side in ("left", "right"):
            index = len(rows)
            layout = layouts[family_id][side]
            seed_u32, seed_u64 = _seed(stream, index, master_seed)
            row = {
                "role_index": index,
                "episode_id": hashlib.sha256(
                    f"{stream}:expert:{master_seed}:{pair_index}:{side}".encode()
                ).hexdigest(),
                "intrusion_side": side,
                "panel_x_jitter_m": panel_x_jitter,
                "panel_face_jitter_m": panel_face_jitter,
                "clutter_x_jitter_m": dict(x_jitter),
                "clutter_y_jitter_m": dict(y_jitter),
                "paired_side_cell": pair_index,
                "family_repeat": repeat,
                "scene_template_house_index": 1,
                "task_seed_u32": seed_u32,
                "task_seed_u64": seed_u64,
                "max_sampling_retries": 12,
                "sampler_class": SAMPLER_CLASS,
                "pact_clutter_palette": palette,
                "pact_clutter_layout": layout,
                "layout_id": layout["layout_id"],
                "layout_family_id": family_id,
                "seed_stream": stream,
            }
            row["row_sha256"] = sha256_payload(row)
            rows.append(row)
    return rows


def build_contract(
    *,
    palette_path: Path = PALETTE_PATH,
    siting_path: Path = SITING_PATH,
) -> dict[str, Any]:
    palette_document = load_palette(palette_path)
    # The original V0c result admitted zero candidates and has no chosen pair.
    # It is retained as a failed-design record, never used as a fallback.
    siting = json.loads(siting_path.read_text())
    siting_payload = dict(siting)
    observed_siting_hash = siting_payload.pop("analysis_sha256", None)
    if observed_siting_hash != sha256_payload(siting_payload):
        raise ValueError("V0c siting artifact self-hash mismatch")
    layouts = {
        family_id: {
            side: build_layout(
                palette_document,
                family_id=family_id,
                intrusion_side=side,
            )
            for side in ("left", "right")
        }
        for family_id in LAYOUT_FAMILIES
    }
    palette = list(palette_document["palette"])
    review_rows = _rows(
        stream=V9_REVIEW_STREAM,
        master_seed=REVIEW_MASTER_SEED,
        palette=palette,
        layouts=layouts,
    )
    gate_rows = _rows(
        stream=V9_GATE_STREAM,
        master_seed=GATE_MASTER_SEED,
        palette=palette,
        layouts=layouts,
    )
    document: dict[str, Any] = {
        "schema_version": "pact_place_corridor_v9_3",
        "status": "v9_3_two_vessel_2d_redesign_requires_validation",
        "role": "v9_environment_build_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "scene": {
            "xml": SCENE_RELATIVE,
            "sampler_class": SAMPLER_CLASS,
            "environment_version": "pact_place_corridor_v9_3",
            "aperture_plane_x_m": TUBE_X0,
            "aperture_width_m": APERTURE_WIDTH,
            "min_depth_m": MIN_DEPTH,
            "place_receptacle_center_xyz_m": [0.35, 0.32, 0.0],
            "place_tray_x_bounds_m": [0.25, 0.45],
            "clutter_body_prefix": "pact_clutter_",
            "clutter_movable_free_bodies": True,
            "clutter_added_to_obstacle_aabbs": True,
            "legacy_panel_active": True,
            "exactly_one_active_panel_per_episode": True,
            "legacy_panel_geometry_file_unchanged": True,
            "clutter_workspace_bounds_m": [
                list(WORKSPACE_LOW_XYZ),
                list(WORKSPACE_HIGH_XYZ),
            ],
            "layout_is_two_dimensional": True,
            "minimum_layout_x_span_m": 0.40,
            "decor_is_rgb_occlusion_only": True,
            "visible_clutter_is_panel_side_invariant": True,
        },
        "palette_path": str(palette_path.resolve().relative_to(ROOT)),
        "palette_sha256": sha256_payload(palette_document),
        "palette": palette,
        "layout_families": layouts,
        "layout_family_ids": list(layouts),
        "siting_path": str(siting_path.resolve().relative_to(ROOT)),
        "failed_v0c_siting_analysis_sha256": observed_siting_hash,
        "failed_v0c_chosen_pair": siting.get("chosen_pair"),
        "v0c3_causal_proximity": {
            "executed": False,
            "passed": False,
            "blocks_v1b": True,
            "requires_real_40_sensor_observation": True,
            "requires_blocker_present_vs_parked_counterfactual": True,
            "sensor_cone_or_aabb_proxy_is_sufficient": False,
        },
        "review": {
            "role": "human_design_review_not_a_gate",
            "stream": V9_REVIEW_STREAM,
            "master_seed": REVIEW_MASTER_SEED,
            "stop_at_clean_successes": 3,
            "stop_at_failures": 3,
            "hard_cap_attempts": 24,
            "render_every_attempt": True,
            "clean_rate_is_not_an_estimate": True,
            "do_not_reuse_for_gate": True,
        },
        "review_rows": review_rows,
        "phase0_gate": {
            "requires_explicit_user_approval": True,
            "stream": V9_GATE_STREAM,
            "master_seed": GATE_MASTER_SEED,
            "n": N_ROWS,
            "minimum_clean_successes": MIN_CLEAN_SUCCESSES,
            "executed": False,
            "authorizes_gate": False,
            "authorizes_collection": False,
            "clean_success_definition": (
                "task success, zero hazard_bar, other_environment, and clutter "
                "contacts, zero clutter-stability events, and no place_receptacle "
                "contact outside placement including preplace"
            ),
            "on_fail": "stop_and_report_without_collection_or_training",
        },
        "expert": {
            "class": "PactPlaceCorridorPolicy",
            "detection_gate": {
                "half_fov_deg": 22.5,
                "range_m": 1.0,
                "range_derate": 0.85,
                "returns_hazard_identity": True,
                "role": "diagnostic_proxy_not_expert_authority",
                "triggered_in_selected_redesign_smokes": False,
                "observed_families": [],
                "silent_families_require_raw_counterfactual": list(LAYOUT_FAMILIES),
            },
            "planning_geometry": "privileged_expert_only_not_a_student_input",
            "inbound_maneuver": "panel_bow_plus_center_blocker_tightening",
            "outbound_maneuver": "panel_bow_plus_center_blocker_tightening",
            "route_blocker_safe_gap_m": ROUTE_BLOCKER_SAFE_GAP_M,
            "panel_safe_gap_m": PANEL_SAFE_GAP_M,
            "legacy_panel_active": True,
            "release_clearance_m": 0.005,
            "outbound_approach_max_step_m": 0.04,
            "multi_maneuver_interactions_reported": True,
            "task_horizon": 900,
        },
        "source": {
            "instrument": "scripts/pact_geom_distance.py",
            "no_aabb_clearance_admission": True,
            "replay_only_through_v1b": True,
        },
        "expert_screen_rows": gate_rows,
    }
    document["config_sha256"] = sha256_payload(document)
    validate_contract(document)
    return document


def validate_contract(document: dict[str, Any]) -> None:
    payload = dict(document)
    observed = payload.pop("config_sha256")
    if observed != sha256_payload(payload):
        raise ValueError("V9 config self-hash mismatch")
    if document.get("authorizes_gate") is not False:
        raise ValueError("V9 config authorizes the gate")
    if document.get("authorizes_collection") is not False:
        raise ValueError("V9 config authorizes collection")
    if document["scene"]["sampler_class"] != SAMPLER_CLASS:
        raise ValueError("V9 config uses the wrong sampler")
    if not str(document["scene"]["xml"]).endswith("pact_place_corridor_v5.xml"):
        raise ValueError("V9 must use the V5 scene shell")
    for rows_key, stream in (
        ("review_rows", V9_REVIEW_STREAM),
        ("expert_screen_rows", V9_GATE_STREAM),
    ):
        rows = document[rows_key]
        if len(rows) != N_ROWS:
            raise ValueError(f"{rows_key} must contain 24 rows")
        if sum(str(row["intrusion_side"]) == "left" for row in rows) != N_ROWS // 2:
            raise ValueError(f"{rows_key} is not side-balanced")
        if any(row.get("seed_stream") != stream for row in rows):
            raise ValueError(f"{rows_key} has a wrong seed stream")
        family_counts = {
            family_id: sum(row.get("layout_family_id") == family_id for row in rows)
            for family_id in LAYOUT_FAMILIES
        }
        if set(family_counts.values()) != {N_ROWS // len(LAYOUT_FAMILIES)}:
            raise ValueError(f"{rows_key} is not layout-family balanced: {family_counts}")
        for family_id in LAYOUT_FAMILIES:
            family_rows = [row for row in rows if row.get("layout_family_id") == family_id]
            if sum(row["intrusion_side"] == "left" for row in family_rows) != len(family_rows) // 2:
                raise ValueError(f"{rows_key}:{family_id} is not side-label balanced")
        for row in rows:
            layout = row.get("pact_clutter_layout") or {}
            if layout.get("intrusion_side") != row.get("intrusion_side"):
                raise ValueError(
                    f"row/layout panel-side mismatch at {rows_key}:{row['role_index']}"
                )
            if layout.get("legacy_panel_active") is not True:
                raise ValueError(
                    f"row has no active panel at {rows_key}:{row['role_index']}"
                )
            row_payload = dict(row)
            observed_row_hash = row_payload.pop("row_sha256")
            if observed_row_hash != sha256_payload(row_payload):
                raise ValueError(f"row self-hash mismatch at {rows_key}:{row['role_index']}")
    review_ids = {row["episode_id"] for row in document["review_rows"]}
    gate_ids = {row["episode_id"] for row in document["expert_screen_rows"]}
    if review_ids & gate_ids:
        raise ValueError("review and gate episode IDs overlap")
    palette = list(document["palette"])
    if (
        len([item for item in palette if item.get("role") in {"inbound_vessel", "outbound_vessel"}])
        != 2
    ):
        raise ValueError("V9 contract does not contain two vessels")
    layouts = document["layout_families"]
    if set(layouts) != set(LAYOUT_FAMILIES):
        raise ValueError("V9.2 contract does not contain the frozen layout families")
    for family_id, variants in layouts.items():
        if set(variants) != {"left", "right"}:
            raise ValueError(f"V9.2 family {family_id} lacks both panel-side variants")
        left_objects = variants["left"]["objects"]
        right_objects = variants["right"]["objects"]
        if left_objects != right_objects:
            raise ValueError(f"V9.2 family {family_id} leaks panel side through visible clutter")
        for side, layout in variants.items():
            validate_layout(layout)
            if layout.get("intrusion_side") != side:
                raise ValueError(f"V9.2 family {family_id} has a mismatched panel side")
            layout_slots = {str(item["palette_slot"]) for item in layout["objects"]}
            if not {"00", "01"}.issubset(layout_slots):
                raise ValueError("V9.2 layout omits a vessel")


def write_contract(path: Path, document: dict[str, Any]) -> None:
    validate_contract(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    output = ROOT / "configs/pact_place_corridor_v9.json"
    write_contract(output, build_contract())
    print(output)
