#!/usr/bin/env python3
"""Deterministic V10.2 raised-pendant contract.

Authorizes nothing. Every artifact built from this module carries
``authorizes_gate``, ``authorizes_collection``, ``authorizes_training`` and
``authorizes_evaluation`` false. V10.1 remains permanently recorded; V10.2 is a
new environment/route/review version with new seed streams and output paths.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

from pact_place_corridor_contract import sha256_file
from pact_place_v10_compound_pendant_contract import (
    ALL_GEOMS,
    ENDPOINT_ONLY_PRIMITIVE,
    MIN_DETOUR_M,
    PLACE_V10_SCENE_SHA256,
    SCENE_XML_RELATIVE,
)
from pact_place_v101_empirical_qualification_contract import (
    PROTECTED_ARTIFACTS,
    verify_protected_artifacts,
)
from pact_place_v102_geometry import (
    ENVIRONMENT_VERSION_V102,
    PROBE_LABEL_V102,
    RAISED_LOWEST_PENDANT_Z_M,
    RAISED_NEGATIVE_LOBE,
    RAISED_POSITIVE_LOBE,
    RAISED_SHELF_GAP_M,
    RAISED_STEM_Y_M,
    SHELF_TOP_Z_M,
    STEM_SQUARE_V102_M,
    planning_probe_v102_raised_assembly,
    raised_assembly_expectations,
)
from pact_place_v102_route import (
    EMPIRICAL_LIVE_CONTACT_V2,
    EMPTY_ARM_APPROACH_SPEED_M_S,
    PENDANT_PASS_SPEED_M_S,
    PREGRASP_APPROACH_SPEED_M_S,
    classify_route_piece,
    speed_cap_violation,
    speed_schedule,
    speed_schedule_sha256,
)
from pact_place_v9_contract import sha256_payload
from pact_place_v95_contract import build_v95_layout, load_v95_palette
from run_pact_place_v96_cluster_causal_proximity import ADMISSION_FLOOR, CORRIDOR_LINKS
from run_pact_place_v9_v0c3_causal_proximity import (
    ABS_DELTA_FLOOR_M,
    MAX_PAIRED_CHANGED_VALUE_RATIO,
)

ROOT = Path(__file__).resolve().parents[1]

CONTRACT_VERSION = "pact_place_v102_raised_pendant_v1"
ENVIRONMENT_VERSION = ENVIRONMENT_VERSION_V102
SAMPLER_CLASS = "PactPlaceCorridorV102RaisedPendantSampler"

SCREEN_STREAM = "pact-place-v10.2-raised-pendant-smoke"
SCREEN_MASTER_SEED = 2026092000
REVIEW_STREAM = "pact-place-v10.2-raised-pendant-human-review"
REVIEW_MASTER_SEED = 2026092002
GATE_STREAM = "pact-place-v10.2-raised-pendant-phase0"
GATE_MASTER_SEED = 2026092001

N_SCREEN_REPEATS = 1
N_REVIEW_REPEATS = 2
N_GATE_REPEATS = 4
N_SCREEN_ROWS = 6
N_REVIEW_ROWS = 12
N_GATE_ROWS = 24

PHYSICS_CLEAN_FAMILIES = (
    "F0_target_side_stagger",
    "F1_inner_panel_stagger",
    "F2_outer_panel_stagger",
)
REGRESSION_ONLY_FAMILIES = ("F3_aperture_side_stagger",)

LEFT_LANE_Y_M = -0.30
RIGHT_LANE_Y_M = 0.30
SLAB_PADDING_M = 0.08
MAX_SEGMENT_TRANSLATION_M = 0.005
MAX_SEGMENT_ROTATION_DEG = 2.0
FROZEN_ENDPOINT_ATOL_M = 1e-9
FROZEN_ENDPOINT_ATOL_RAD = 1e-9

MIN_PENDANT_CLEARANCE_M = 0.015
MIN_SCREEN_CLEAN_SUCCESSES = 6
MIN_REVIEW_CLEAN_SUCCESSES = 10
MIN_CLEAN_PER_REVIEW_CELL = 1
MIN_GATE_CLEAN_SUCCESSES = 20
MIN_CLEAN_PER_GATE_CELL = 3
MAX_SIDE_IMBALANCE = MAX_PAIRED_CHANGED_VALUE_RATIO

POLICY_TIMESTEP_MS = 66.0
REVIEW_FPS = 1000.0 / POLICY_TIMESTEP_MS
REVIEW_FRAME_STRIDE = 1
REVIEW_TINT_LABEL = "REVIEW TINT"

IMPLEMENTATION_PATHS = (
    "scripts/pact_place_v102_raised_pendant_contract.py",
    "scripts/pact_place_v102_geometry.py",
    "scripts/pact_place_v102_route.py",
    "scripts/run_pact_place_v102_preflight.py",
    "scripts/run_pact_place_v102_screen.py",
    "scripts/run_pact_place_v102_review.py",
    "scripts/run_pact_place_v102_review_video.py",
    "scripts/run_pact_place_v102_causal.py",
    "scripts/run_pact_place_v102_phase0.py",
    "scripts/pact_place_v10_compound_pendant_contract.py",
    "scripts/pact_place_v10_geometry.py",
    "scripts/pact_place_v10_route.py",
    "scripts/pact_place_v10_scene.py",
    "scripts/run_pact_place_expert_screen.py",
    "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    SCENE_XML_RELATIVE,
)


def empty_authorization() -> dict[str, bool]:
    return {
        "authorizes_gate": False,
        "authorizes_collection": False,
        "authorizes_training": False,
        "authorizes_evaluation": False,
        "authorizes_eval": False,
        "phase0_passed": False,
        "eligible_for_separate_collection_authorization": False,
        "environment_qualified": False,
    }


def frozen_raised_lobes() -> dict[str, Any]:
    return {
        "negative_lobe": {
            "center_m": list(RAISED_NEGATIVE_LOBE["center_m"]),
            "half_m": list(RAISED_NEGATIVE_LOBE["half_m"]),
        },
        "positive_lobe": {
            "center_m": list(RAISED_POSITIVE_LOBE["center_m"]),
            "half_m": list(RAISED_POSITIVE_LOBE["half_m"]),
        },
    }


def registered_assembly_expectations() -> dict[str, Any]:
    """The consequences the plan requires the contract to assert."""
    return {
        "lowest_pendant_z_m": float(RAISED_LOWEST_PENDANT_Z_M),
        "shelf_top_z_m": float(SHELF_TOP_Z_M),
        "shelf_to_pendant_gap_m": float(RAISED_SHELF_GAP_M),
        "negative_lobe_center_m": list(RAISED_NEGATIVE_LOBE["center_m"]),
        "negative_lobe_half_m": list(RAISED_NEGATIVE_LOBE["half_m"]),
        "positive_lobe_center_m": list(RAISED_POSITIVE_LOBE["center_m"]),
        "positive_lobe_half_m": list(RAISED_POSITIVE_LOBE["half_m"]),
        "stem_center_y_m": [float(value) for value in RAISED_STEM_Y_M],
        "stem_top_z_m": [1.505, 1.505],
        "stem_square_m": [float(STEM_SQUARE_V102_M), float(STEM_SQUARE_V102_M)],
        "stem_square_y_m": [float(STEM_SQUARE_V102_M), float(STEM_SQUARE_V102_M)],
        "crossbar_top_z_m": 1.515,
        "crossbar_square_x_m": float(STEM_SQUARE_V102_M),
        "kinematic_fixed_assembly": True,
        "physical_swing_dynamics": False,
    }


def frozen_assembly() -> dict[str, Any]:
    """The single registered raised assembly, checked against the contract."""
    assembly = planning_probe_v102_raised_assembly()
    if assembly.get("probe_label") != PROBE_LABEL_V102:
        raise ValueError("V10.2 probe label drifted")
    observed = raised_assembly_expectations(assembly)
    expected = registered_assembly_expectations()
    for key, value in expected.items():
        if isinstance(value, list):
            if [float(item) for item in observed[key]] != [float(item) for item in value]:
                raise ValueError(f"V10.2 assembly {key} drifted: {observed[key]} != {value}")
        elif isinstance(value, bool):
            if bool(observed[key]) is not value:
                raise ValueError(f"V10.2 assembly {key} drifted")
        elif abs(float(observed[key]) - float(value)) > 1e-9:
            raise ValueError(f"V10.2 assembly {key} drifted: {observed[key]} != {value}")
    stem_squares = set()
    for item in assembly["components"]:
        if item["role"] not in {"stem", "crossbar"} or not item.get("active"):
            continue
        stem_squares.add(round(2.0 * float(item["half_m"][0]), 9))
    if stem_squares != {round(float(STEM_SQUARE_V102_M), 9)}:
        raise ValueError(f"V10.2 stem/crossbar x thickness drifted: {stem_squares}")
    return assembly


def frozen_route_for_side(intrusion_side: str) -> dict[str, Any]:
    side = str(intrusion_side)
    if side == "left":
        lane = LEFT_LANE_Y_M
    elif side == "right":
        lane = RIGHT_LANE_Y_M
    else:
        raise ValueError(f"intrusion_side must be left or right, got {intrusion_side!r}")
    return {
        "rewrite_primitive": ENDPOINT_ONLY_PRIMITIVE,
        "qualification_mode": EMPIRICAL_LIVE_CONTACT_V2,
        "environment_version": ENVIRONMENT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "inbound_lane_y_m": float(lane),
        "outbound_lane_y_m": float(lane),
        "inbound_padding_m": float(SLAB_PADDING_M),
        "outbound_padding_m": float(SLAB_PADDING_M),
        "slab_padding_m": float(SLAB_PADDING_M),
        "left_lane_y_m": float(LEFT_LANE_Y_M),
        "right_lane_y_m": float(RIGHT_LANE_Y_M),
        "min_detour_m": float(MIN_DETOUR_M),
        "max_segment_translation_m": float(MAX_SEGMENT_TRANSLATION_M),
        "max_segment_rotation_deg": float(MAX_SEGMENT_ROTATION_DEG),
        "frozen_endpoint_atol_m": float(FROZEN_ENDPOINT_ATOL_M),
        "frozen_endpoint_atol_rad": float(FROZEN_ENDPOINT_ATOL_RAD),
        "min_pendant_clearance_m": float(MIN_PENDANT_CLEARANCE_M),
        "speed_schedule": speed_schedule(),
        "speed_schedule_sha256": speed_schedule_sha256(),
    }


def implementation_hashes() -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def implementation_sha256() -> str:
    return sha256_payload(implementation_hashes())


def _seed(stream: str, index: int, master_seed: int) -> tuple[int, int]:
    digest = hashlib.sha256(
        f"{stream}:{int(master_seed)}:{int(index)}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big"), int.from_bytes(digest[:8], "big")


def _layouts(palette_document: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        family_id: {
            side: build_v95_layout(
                palette_document, family_id=family_id, intrusion_side=side
            )
            for side in ("left", "right")
        }
        for family_id in PHYSICS_CLEAN_FAMILIES
    }


def _rows(
    *,
    stream: str,
    master_seed: int,
    n_repeats: int,
    palette_document: dict[str, Any],
    assembly: dict[str, Any],
    implementation_digest: str,
) -> list[dict[str, Any]]:
    """V10.1 jitter distribution, new stream and master seed."""
    rng = random.Random(int(master_seed))
    layouts = _layouts(palette_document)
    pairs = [
        (family_id, repeat)
        for family_id in PHYSICS_CLEAN_FAMILIES
        for repeat in range(n_repeats)
    ]
    rng.shuffle(pairs)
    palette = list(palette_document["palette"])
    assembly_digest = sha256_payload(assembly)
    rows: list[dict[str, Any]] = []
    for pair_index, (family_id, repeat) in enumerate(pairs):
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
                "target_x_jitter_m": 0.0,
                "target_y_jitter_m": 0.0,
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
                "family": family_id,
                "seed_stream": stream,
                "pact_v10_pendant_assembly": assembly,
                "pact_v102_assembly_sha256": assembly_digest,
                "pact_v10_route": frozen_route_for_side(side),
                "pact_v102_contract_version": CONTRACT_VERSION,
                "implementation_sha256": implementation_digest,
            }
            row["row_sha256"] = sha256_payload(row)
            rows.append(row)
    return rows


def build_contract() -> dict[str, Any]:
    palette_document = load_v95_palette()
    assembly = frozen_assembly()
    digest = implementation_sha256()
    screen_rows = _rows(
        stream=SCREEN_STREAM,
        master_seed=SCREEN_MASTER_SEED,
        n_repeats=N_SCREEN_REPEATS,
        palette_document=palette_document,
        assembly=assembly,
        implementation_digest=digest,
    )
    review_rows = _rows(
        stream=REVIEW_STREAM,
        master_seed=REVIEW_MASTER_SEED,
        n_repeats=N_REVIEW_REPEATS,
        palette_document=palette_document,
        assembly=assembly,
        implementation_digest=digest,
    )
    gate_rows = _rows(
        stream=GATE_STREAM,
        master_seed=GATE_MASTER_SEED,
        n_repeats=N_GATE_REPEATS,
        palette_document=palette_document,
        assembly=assembly,
        implementation_digest=digest,
    )
    for label, rows, expected in (
        ("screen", screen_rows, N_SCREEN_ROWS),
        ("review", review_rows, N_REVIEW_ROWS),
        ("gate", gate_rows, N_GATE_ROWS),
    ):
        if len(rows) != expected:
            raise RuntimeError(f"expected {expected} {label} rows, got {len(rows)}")
    seed_sets = {
        "screen": {int(row["task_seed_u32"]) for row in screen_rows},
        "review": {int(row["task_seed_u32"]) for row in review_rows},
        "gate": {int(row["task_seed_u32"]) for row in gate_rows},
    }
    for left in ("screen", "review", "gate"):
        for right in ("screen", "review", "gate"):
            if left >= right:
                continue
            if seed_sets[left] & seed_sets[right]:
                raise RuntimeError(f"{left} and {right} task-seed streams intersect")
    payload = {
        "schema_version": CONTRACT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "scene_xml": SCENE_XML_RELATIVE,
        "scene_sha256": PLACE_V10_SCENE_SHA256,
        "compiles_v10_scene": True,
        "parks_all_legacy_mounts": True,
        "screen_stream": SCREEN_STREAM,
        "review_stream": REVIEW_STREAM,
        "gate_stream": GATE_STREAM,
        "screen_master_seed": SCREEN_MASTER_SEED,
        "review_master_seed": REVIEW_MASTER_SEED,
        "gate_master_seed": GATE_MASTER_SEED,
        "n_screen_rows": N_SCREEN_ROWS,
        "n_review_rows": N_REVIEW_ROWS,
        "n_gate_rows": N_GATE_ROWS,
        "families": list(PHYSICS_CLEAN_FAMILIES),
        "regression_only_families": list(REGRESSION_ONLY_FAMILIES),
        "raised_lobes": frozen_raised_lobes(),
        "assembly_id": assembly.get("assembly_id"),
        "assembly": assembly,
        "assembly_expectations": registered_assembly_expectations(),
        "assembly_self_sha256": sha256_payload(assembly),
        "pendant_geoms": list(ALL_GEOMS),
        "route_constants": {
            "rewrite_primitive": ENDPOINT_ONLY_PRIMITIVE,
            "qualification_mode": EMPIRICAL_LIVE_CONTACT_V2,
            "slab_padding_m": SLAB_PADDING_M,
            "left_lane_y_m": LEFT_LANE_Y_M,
            "right_lane_y_m": RIGHT_LANE_Y_M,
            "min_detour_m": MIN_DETOUR_M,
            "max_segment_translation_m": MAX_SEGMENT_TRANSLATION_M,
            "max_segment_rotation_deg": MAX_SEGMENT_ROTATION_DEG,
            "frozen_endpoint_atol_m": FROZEN_ENDPOINT_ATOL_M,
            "frozen_endpoint_atol_rad": FROZEN_ENDPOINT_ATOL_RAD,
        },
        "speed_schedule": speed_schedule(),
        "speed_schedule_sha256": speed_schedule_sha256(),
        "min_pendant_clearance_m": MIN_PENDANT_CLEARANCE_M,
        "review_fps": REVIEW_FPS,
        "review_frame_stride": REVIEW_FRAME_STRIDE,
        "policy_timestep_ms": POLICY_TIMESTEP_MS,
        "admission_floor": ADMISSION_FLOOR,
        "abs_delta_floor_m": ABS_DELTA_FLOOR_M,
        "max_paired_changed_value_ratio": MAX_SIDE_IMBALANCE,
        "corridor_links": list(CORRIDOR_LINKS),
        "implementation_sha256": digest,
        "implementation_files": implementation_hashes(),
        "protected_artifacts": {
            relative: expected for relative, expected, _kind in PROTECTED_ARTIFACTS
        },
        "screen_rows": screen_rows,
        "review_rows": review_rows,
        "gate_rows": gate_rows,
        **empty_authorization(),
    }
    payload["contract_sha256"] = sha256_payload(payload)
    return payload


def cell_key(family_id: str, intrusion_side: str) -> tuple[str, str]:
    return str(family_id), str(intrusion_side)


def distribution_counts(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = cell_key(str(row["layout_family_id"]), str(row["intrusion_side"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def paired_side_clutter_identical(rows: Sequence[dict[str, Any]]) -> bool:
    by_pair: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(int(row["paired_side_cell"]), []).append(row)
    for pair_rows in by_pair.values():
        if len(pair_rows) != 2:
            return False
        left, right = pair_rows
        if {left["intrusion_side"], right["intrusion_side"]} != {"left", "right"}:
            return False
        for key in (
            "clutter_x_jitter_m",
            "clutter_y_jitter_m",
            "panel_x_jitter_m",
            "panel_face_jitter_m",
        ):
            if left[key] != right[key]:
                return False
        left_xy = {
            str(item["palette_slot"]): list(item["center_m"][:2])
            for item in left["pact_clutter_layout"]["objects"]
            if str(item.get("role")) in {"inbound_vessel", "outbound_vessel"}
        }
        right_xy = {
            str(item["palette_slot"]): list(item["center_m"][:2])
            for item in right["pact_clutter_layout"]["objects"]
            if str(item.get("role")) in {"inbound_vessel", "outbound_vessel"}
        }
        if left_xy != right_xy:
            return False
        if (
            left["pact_clutter_layout"]["inbound_vessel_center_xy_m"]
            != right["pact_clutter_layout"]["inbound_vessel_center_xy_m"]
        ):
            return False
    return True


# --------------------------------------------------------------------------
# Row-level V10.2 admission
# --------------------------------------------------------------------------
REQUIRED_ROUTE_KEYS = (
    "rewrite_primitive",
    "qualification_mode",
    "lane_y_m",
    "padding_m",
    "min_abs_detour_m",
    "detour_meets_minimum",
    "fallback_taken",
    "clipped",
    "wrong_way",
    "frozen_endpoint_preserved",
    "offline_strict_environment_preclearance_used",
    "strict_environment_preclearance_intentionally_not_used",
    "speed_schedule_sha256",
    "piece_speeds",
    "waypoints_attempted",
    "waypoints_solved",
    "complete_sequential_ik",
)


def _route_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    block = result.get("pendant_v10") or {}
    return [dict(block.get("inbound") or {}), dict(block.get("outbound") or {})]


def route_telemetry_complete(result: dict[str, Any]) -> bool:
    records = _route_records(result)
    if len(records) != 2 or any(not record for record in records):
        return False
    for record in records:
        if any(key not in record for key in REQUIRED_ROUTE_KEYS):
            return False
        if record.get("rewrite_primitive") != ENDPOINT_ONLY_PRIMITIVE:
            return False
        if record.get("qualification_mode") != EMPIRICAL_LIVE_CONTACT_V2:
            return False
        if record.get("speed_schedule_sha256") != speed_schedule_sha256():
            return False
        if record.get("offline_strict_environment_preclearance_used") is not False:
            return False
        if record.get("strict_environment_preclearance_intentionally_not_used") is not True:
            return False
    return True


def route_defects(result: dict[str, Any]) -> list[str]:
    defects: list[str] = []
    if not route_telemetry_complete(result):
        return ["missing_telemetry"]
    for prefix, record in zip(("inbound", "outbound"), _route_records(result)):
        if record.get("fallback_taken"):
            defects.append(f"{prefix}_fallback")
        if record.get("clipped"):
            defects.append(f"{prefix}_clipped")
        if record.get("wrong_way"):
            defects.append(f"{prefix}_wrong_way")
        if record.get("frozen_endpoint_preserved") is not True:
            defects.append(f"{prefix}_endpoint_mutation")
        if record.get("detour_meets_minimum") is False:
            defects.append(f"{prefix}_insufficient_detour")
        if record.get("complete_sequential_ik") is not True:
            defects.append(f"{prefix}_incomplete_sequential_ik")
        for piece in list(record.get("piece_speeds") or []):
            violation = speed_cap_violation(
                str(piece.get("name")), float(piece.get("requested_speed_m_s"))
            )
            if violation:
                defects.append(f"{prefix}_{violation}")
    return sorted(set(defects))


def frame_defects(result: dict[str, Any]) -> list[str]:
    """Per-policy-frame stem-contact, clearance, and telemetry requirements."""
    frames = result.get("pendant_frame_telemetry") or {}
    if not frames:
        return ["missing_frame_telemetry"]
    defects: list[str] = []
    if int(frames.get("n_frames") or 0) <= 0:
        defects.append("missing_frame_telemetry")
        return defects
    if int(frames.get("n_frames_measured") or 0) != int(frames.get("n_frames") or 0):
        defects.append("incomplete_frame_clearance")
    if int(frames.get("live_pendant_contact_frames") or 0) > 0:
        defects.append("live_pendant_contact")
    minimum = frames.get("min_clearance_m")
    if minimum is None:
        defects.append("missing_frame_clearance")
    elif float(minimum) < MIN_PENDANT_CLEARANCE_M - 1e-12:
        defects.append("frame_clearance_below_floor")
    for name, value in (frames.get("per_component_min_clearance_m") or {}).items():
        if value is None:
            defects.append(f"missing_component_clearance:{name}")
        elif float(value) < MIN_PENDANT_CLEARANCE_M - 1e-12:
            defects.append(f"component_clearance_below_floor:{name}")
    for piece in list(frames.get("segment_speeds") or []):
        try:
            violation = speed_cap_violation(
                str(piece.get("name")), float(piece.get("commanded_speed_m_s"))
            )
        except ValueError:
            continue
        if violation:
            defects.append(f"realized_{violation}")
    return sorted(set(defects))


def contact_defects(result: dict[str, Any]) -> list[str]:
    audit = result.get("contact_audit") or {}
    totals = audit.get("contact_class_totals") or {}
    defects: list[str] = []
    if not totals:
        return ["missing_contact_audit"]
    for key in ("mounted_fixture", "hazard_bar", "other_environment", "clutter"):
        if int(totals.get(key, 0)) > 0:
            defects.append(f"{key}_contact")
    if result.get("clutter_stability_events"):
        defects.append("clutter_stability_event")
    return sorted(set(defects))


def row_defects(result: dict[str, Any]) -> list[str]:
    if result.get("status") == "sampling_failure":
        return ["sampling_failure"]
    if result.get("status") == "infrastructure_failure":
        return ["infrastructure_failure"]
    if result.get("status") != "complete":
        return ["nonterminal"]
    defects = list(route_defects(result))
    defects.extend(contact_defects(result))
    defects.extend(frame_defects(result))
    if not result.get("task_success"):
        defects.append("task_failure")
    if not result.get("clean_success"):
        defects.append("not_strict_clean_success")
    return sorted(set(defects))


def is_v102_clean_success(result: dict[str, Any]) -> bool:
    return not row_defects(result)


def _cell_table() -> dict[tuple[str, str], int]:
    return {
        (family, side): 0
        for family in PHYSICS_CLEAN_FAMILIES
        for side in ("left", "right")
    }


def _stage_eligibility(
    rows: Sequence[dict[str, Any]],
    results: Sequence[dict[str, Any]],
    *,
    expected_rows: int,
    min_clean: int,
    min_clean_per_cell: int,
    passed_key: str,
) -> dict[str, Any]:
    by_episode = {str(item["episode_id"]): item for item in results}
    failures: list[dict[str, Any]] = []
    reconciled = True
    if len(rows) != expected_rows or len(results) != expected_rows:
        reconciled = False
        failures.append(
            {
                "code": "row_count",
                "detail": f"rows={len(rows)} results={len(results)} expected={expected_rows}",
            }
        )
    infrastructure = 0
    clean_by_cell = _cell_table()
    n_clean = 0
    for row in rows:
        result = by_episode.get(str(row["episode_id"]))
        if result is None:
            reconciled = False
            failures.append(
                {
                    "code": "missing_result",
                    "episode_id": row["episode_id"],
                    "role_index": row["role_index"],
                }
            )
            continue
        if result.get("row_sha256") != row.get("row_sha256"):
            reconciled = False
            failures.append({"code": "row_sha_mismatch", "role_index": row["role_index"]})
        if result.get("status") == "infrastructure_failure":
            infrastructure += 1
        for defect in row_defects(result):
            failures.append(
                {
                    "code": defect,
                    "role_index": row["role_index"],
                    "episode_id": row["episode_id"],
                    "family": row.get("layout_family_id"),
                    "intrusion_side": row.get("intrusion_side"),
                }
            )
        if is_v102_clean_success(result):
            n_clean += 1
            clean_by_cell[cell_key(str(row["layout_family_id"]), str(row["intrusion_side"]))] += 1
    cell_failures = [
        {
            "code": "cell_clean_shortfall",
            "family": family,
            "intrusion_side": side,
            "clean": count,
            "required": min_clean_per_cell,
        }
        for (family, side), count in sorted(clean_by_cell.items())
        if count < min_clean_per_cell
    ]
    failures.extend(cell_failures)
    passed = bool(
        reconciled
        and infrastructure == 0
        and n_clean >= min_clean
        and not cell_failures
    )
    return {
        passed_key: passed,
        "reconciled": reconciled,
        "n_rows": len(results),
        "infrastructure_failures": infrastructure,
        "clean_successes": n_clean,
        "min_clean_successes": min_clean,
        "min_clean_per_cell": min_clean_per_cell,
        "clean_by_cell": {
            f"{family}:{side}": count for (family, side), count in sorted(clean_by_cell.items())
        },
        "failures": failures,
        **empty_authorization(),
    }


def screen_eligibility(
    rows: Sequence[dict[str, Any]], results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    return _stage_eligibility(
        rows,
        results,
        expected_rows=N_SCREEN_ROWS,
        min_clean=MIN_SCREEN_CLEAN_SUCCESSES,
        min_clean_per_cell=1,
        passed_key="screen_passed",
    )


def review_eligibility(
    rows: Sequence[dict[str, Any]], results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    report = _stage_eligibility(
        rows,
        results,
        expected_rows=N_REVIEW_ROWS,
        min_clean=MIN_REVIEW_CLEAN_SUCCESSES,
        min_clean_per_cell=MIN_CLEAN_PER_REVIEW_CELL,
        passed_key="eligible_for_human_review",
    )
    return report


def gate_eligibility(
    rows: Sequence[dict[str, Any]], results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    return _stage_eligibility(
        rows,
        results,
        expected_rows=N_GATE_ROWS,
        min_clean=MIN_GATE_CLEAN_SUCCESSES,
        min_clean_per_cell=MIN_CLEAN_PER_GATE_CELL,
        passed_key="phase0_passed",
    )


def lowest_clean_row_per_cell(
    rows: Sequence[dict[str, Any]], results: Sequence[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    by_episode = {str(item["episode_id"]): item for item in results}
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: int(item["role_index"])):
        result = by_episode.get(str(row["episode_id"]))
        if result is None or not is_v102_clean_success(result):
            continue
        key = cell_key(str(row["layout_family_id"]), str(row["intrusion_side"]))
        if key not in selected:
            selected[key] = {"row": row, "result": result}
    return selected


def assert_phase0_approval(
    approval: dict[str, Any] | None,
    *,
    contract_sha256: str,
    preflight_sha256: str,
    screen_manifest_sha256: str,
    review_manifest_sha256: str,
    causal_artifact_sha256: str,
) -> None:
    if not approval:
        raise PermissionError("Phase 0 requires an explicit owner human_approval.json")
    if approval.get("decision") != "approve_phase0":
        raise PermissionError(f"Phase 0 refused: decision={approval.get('decision')!r}")
    expected = {
        "contract_sha256": contract_sha256,
        "preflight_sha256": preflight_sha256,
        "screen_manifest_sha256": screen_manifest_sha256,
        "review_manifest_sha256": review_manifest_sha256,
        "causal_artifact_sha256": causal_artifact_sha256,
    }
    for key, digest in expected.items():
        if approval.get(key) != digest:
            raise PermissionError(
                f"Phase 0 approval is not bound to {key}: {approval.get(key)!r} != {digest!r}"
            )


__all__ = [name for name in dir() if not name.startswith("_")]
