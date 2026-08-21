#!/usr/bin/env python3
"""Frozen Phase-0 contract for the forked PACT pick-and-place corridor."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
DEFAULT_MASTER_SEED = 2026082701
N_EXPERT_ROWS = 24
MIN_CLEAN_SUCCESSES = 20
PASS_TOKEN = "PACT_PLACE_CORRIDOR_PHASE0_PASS"
FAIL_TOKEN = "PACT_PLACE_CORRIDOR_PHASE0_FAIL"
PLACE_V2_CENTER_XYZ_M = [0.35, 0.32, 0.0]
PLACE_V2_TRAY_X_BOUNDS_M = [0.25, 0.45]
ATTEMPT5_SWEEP_SHA256 = (
    "b657da019b8638ba8b94e1bfa64a1d31ddfa7c27d7a7c6f4b6f22824602e211b"
)
CLUTTER_SLOT_NAMES = ("l0", "l1", "r0", "r1")
CLUTTER_JITTER_M = 0.02
V7_CLUTTER_POOL_SLOT_NAMES = tuple(f"{index:02d}" for index in range(16))
V7_CLUTTER_ACTIVE_SLOT_NAMES = tuple(f"{index:02d}" for index in range(13))
V7_DESIGN_REVIEW_MASTER_SEED = 2026082801
N_DESIGN_REVIEW_ROWS = 12
DESIGN_REVIEW_STOP_CLEAN = 3
PLACE_V3_SCENE_SHA256 = (
    "cf104be203c8e13b3dc02a0561853bfb7c61804f6bf1ec156d86174455b31c6f"
)
PLACE_V4_SCENE_SHA256 = (
    "f9eef910c28d113ca48dcc8045207463da08e6517f3fdfe3141c7ebbdbfb281b"
)
PLACE_V5_SCENE_SHA256 = (
    "5ac1ebd3e04f0bf509f6b8e11f0d086ac8c43bd550349762aba6c4129aebd61c"
)
V8_FAMILY_REVIEW_MASTER_SEED = 2026082901
V8_GATE_MASTER_SEED = 2026083001
V8_FAMILIES = (
    "F1_near_forearm_left",
    "F2_near_forearm_right",
    "F3_front_stagger",
    "F4_rear_stagger",
    "F5_overhead_elbow",
    "F6_target_occluding",
)
V8_SELECTED_LAYOUTS_RELATIVE = (
    "diagnostics_output/pact_place_clutter_sweep_v8/selected_layouts.json"
)
ATTEMPT6_SWEEP_RELATIVE = (
    "diagnostics_output/pact_place_clutter_sweep/analysis.json"
)
ATTEMPT6B_SWEEP_RELATIVE = (
    "diagnostics_output/pact_place_clutter_sweep_v6b/analysis.json"
)
ATTEMPT6C_SWEEP_RELATIVE = (
    "diagnostics_output/pact_place_clutter_sweep_v6c/analysis.json"
)


def resolve_master_seed(master_seed: int | None = None) -> int:
    if master_seed is not None:
        return int(master_seed)
    env = os.environ.get("PACT_PLACE_MASTER_SEED")
    if env:
        return int(env)
    return DEFAULT_MASTER_SEED


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_payload(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def row_seed(index: int, master_seed: int | None = None) -> tuple[int, int]:
    seed = resolve_master_seed(master_seed)
    digest = hashlib.sha256(f"pact-place-v1:{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big"), int.from_bytes(digest[:8], "big")


def retry_seed(row: dict[str, Any], retry_index: int) -> tuple[int, int]:
    digest = hashlib.sha256(
        f"{row['row_sha256']}:pre-boundary-retry:{retry_index}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big"), int.from_bytes(digest[:8], "big")


def _source_hashes() -> dict[str, str]:
    paths = [
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml",
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v1.xml",
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v1_metadata.json",
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v2.xml",
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v2_metadata.json",
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v3.xml",
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v3_metadata.json",
        "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
        "submodules/molmospaces/molmo_spaces/tasks/pact_contact_audit.py",
        "submodules/molmospaces/molmo_spaces/tasks/pact_place_contact_audit.py",
        "submodules/molmospaces/molmo_spaces/tasks/pick_and_place_task.py",
        "submodules/molmospaces/molmo_spaces/policy/solvers/object_manipulation/pick_and_place_planner_policy.py",
        "submodules/molmospaces/molmo_spaces/policy/solvers/object_manipulation/base_object_manipulation_planner_policy.py",
        "submodules/molmospaces/molmo_spaces/configs/task_configs.py",
        "scripts/pact_place_corridor_contract.py",
        "scripts/run_pact_place_expert_screen.py",
        "scripts/run_pact_place_reachability_sweep.py",
        "scripts/run_pact_place_clutter_sweep.py",
        "scripts/run_pact_place_clutter_sweep_v6b.py",
        "scripts/run_pact_place_clutter_sweep_v6c.py",
    ]
    return {relative: sha256_file(ROOT / relative) for relative in paths}


def _protected_artifact_hashes() -> dict[str, str]:
    candidates = [
        "docs/PACT_CONTACT_ENDPOINT_DECISION.md",
        "diagnostics_output/pact_contact_endpoint/analysis.json",
        "diagnostics_output/pact_contact_endpoint/final_decision.json",
        "docs/PACT_GEOMETRY_GENERALIZATION.md",
        "diagnostics_output/pact_geometry_generalization_v3/analysis.json",
        "diagnostics_output/pact_geometry_generalization_v3/final_decision.json",
        "docs/PACT_BLUR_SWEEP.md",
        "diagnostics_output/pact_blur_sweep/analysis.json",
        "diagnostics_output/pact_blur_sweep/final_decision.json",
        "docs/PACT_BLIND_RGB.md",
        "diagnostics_output/pact_blind_rgb/analysis.json",
        "diagnostics_output/pact_blind_rgb/final_decision.json",
    ]
    return {
        relative: sha256_file(ROOT / relative)
        for relative in candidates
        if (ROOT / relative).is_file()
    }


def clutter_jitters_for_seed(
    seed_u64: int,
    slot_names: tuple[str, ...] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    rng = random.Random(int(seed_u64))
    names = tuple(slot_names) if slot_names is not None else CLUTTER_SLOT_NAMES
    x_jitter: dict[str, float] = {}
    y_jitter: dict[str, float] = {}
    for slot in names:
        x_jitter[slot] = round(rng.uniform(-CLUTTER_JITTER_M, CLUTTER_JITTER_M), 9)
        y_jitter[slot] = round(rng.uniform(-CLUTTER_JITTER_M, CLUTTER_JITTER_M), 9)
    return x_jitter, y_jitter


def _attempt6_sweep_block() -> dict[str, Any] | None:
    path = ROOT / ATTEMPT6_SWEEP_RELATIVE
    if not path.is_file():
        return None
    analysis = json.loads(path.read_text())
    chosen = analysis.get("chosen") or {}
    return {
        "path": ATTEMPT6_SWEEP_RELATIVE,
        "sweep_sha256": analysis.get("sweep_sha256"),
        "chosen_slots_xy_m": chosen.get("slots_xy_m"),
        "chosen_height_m": chosen.get("height_m"),
        "chosen_half_xy_m": chosen.get("half_xy_m"),
        "n_candidate_sets": analysis.get("n_candidate_sets"),
        "n_footprint_ok": analysis.get("n_footprint_ok"),
        "n_ik_ok": analysis.get("n_ik_ok"),
        "n_eligible": analysis.get("n_eligible"),
    }


def _attempt6b_sweep_block() -> dict[str, Any] | None:
    path = ROOT / ATTEMPT6B_SWEEP_RELATIVE
    if not path.is_file():
        return None
    analysis = json.loads(path.read_text())
    chosen = analysis.get("chosen") or {}
    return {
        "path": ATTEMPT6B_SWEEP_RELATIVE,
        "sweep_sha256": analysis.get("sweep_sha256"),
        "chosen_slots_xy_m": chosen.get("slots_xy_m"),
        "chosen_height_m": chosen.get("height_m"),
        "chosen_half_xy_m": chosen.get("half_xy_m"),
        "n_candidate_sets": analysis.get("n_candidate_sets"),
        "n_footprint_ok": analysis.get("n_footprint_ok"),
        "n_ik_ok": analysis.get("n_ik_ok"),
        "n_eligible": analysis.get("n_eligible"),
        "step0_axis": analysis.get("step0_axis"),
    }


def _attempt6c_sweep_block() -> dict[str, Any] | None:
    path = ROOT / ATTEMPT6C_SWEEP_RELATIVE
    if not path.is_file():
        return None
    analysis = json.loads(path.read_text())
    chosen = analysis.get("chosen") or {}
    return {
        "path": ATTEMPT6C_SWEEP_RELATIVE,
        "sweep_sha256": analysis.get("sweep_sha256"),
        "chosen_slots_xy_m": chosen.get("slots_xy_m"),
        "chosen_height_m": chosen.get("height_m"),
        "chosen_top_z_m": chosen.get("top_z_m"),
        "chosen_half_x_m": chosen.get("half_x_m"),
        "chosen_half_y_m": chosen.get("half_y_m"),
        "min_inner_face_abs_y_m": chosen.get("min_inner_face_abs_y_m"),
        "max_outer_face_x_m": chosen.get("max_outer_face_x_m"),
        "n_candidate_sets": analysis.get("n_candidate_sets"),
        "n_footprint_ok": analysis.get("n_footprint_ok"),
        "n_enclosure_ok": analysis.get("n_enclosure_ok"),
        "n_ik_ok": analysis.get("n_ik_ok"),
        "n_eligible": analysis.get("n_eligible"),
        "inner_face_held_at_abs_y_m": analysis.get("inner_face_held_at_abs_y_m"),
        "boxes_not_shrunk": analysis.get("boxes_not_shrunk"),
    }


def episode_id_for(index: int, side: str, master_seed: int | None = None) -> str:
    # Frozen v1/v2 contracts hashed only (index, side) and share 12 IDs across
    # master seeds. New contracts include master_seed. Do not rewrite those files;
    # join historical rows on (config_sha256, role_index), never episode_id.
    seed = resolve_master_seed(master_seed)
    return hashlib.sha256(
        f"pact-place-v1:expert:{seed}:{index}:{side}".encode()
    ).hexdigest()


def build_contract(master_seed: int | None = None) -> dict[str, Any]:
    seed = resolve_master_seed(master_seed)
    rng = random.Random(seed)
    sides = ["left", "right"] * (N_EXPERT_ROWS // 2)
    rng.shuffle(sides)
    rows = []
    for index, side in enumerate(sides):
        seed_u32, seed_u64 = row_seed(index, seed)
        clutter_x, clutter_y = clutter_jitters_for_seed(seed_u64)
        row = {
            "role_index": index,
            "episode_id": episode_id_for(index, side, seed),
            "intrusion_side": side,
            "panel_x_jitter_m": round(rng.uniform(-0.015, 0.015), 9),
            "panel_face_jitter_m": round(rng.uniform(-0.005, 0.005), 9),
            "clutter_x_jitter_m": clutter_x,
            "clutter_y_jitter_m": clutter_y,
            "scene_template_house_index": 1,
            "task_seed_u32": seed_u32,
            "task_seed_u64": seed_u64,
            "max_sampling_retries": 4,
        }
        row["row_sha256"] = sha256_payload(row)
        rows.append(row)
    document: dict[str, Any] = {
        "schema_version": "pact_place_corridor_v6",
        "status": "phase0_preregistered",
        "created_utc": "2026-08-19T00:00:00Z",
        "route": "collision_route_pick_and_place",
        "master_seed": seed,
        "scene": {
            "xml": "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v3.xml",
            "sampler_class": "PactPlaceCorridorV3Sampler",
            "base_forward_m": 0.14,
            "aperture_plane_x_m": 0.58,
            "place_receptacle_center_xyz_m": list(PLACE_V2_CENTER_XYZ_M),
            "place_tray_x_bounds_m": list(PLACE_V2_TRAY_X_BOUNDS_M),
            "place_receptacle_footprint": "shrunk_0.10x0.10",
            "place_receptacle_contact_exempt": False,
            "place_receptacle_exempt_during_placement_including_preplace": True,
            "legacy_contact_classes_unchanged": [
                "grasp_target",
                "hazard_bar",
                "other_environment",
            ],
            "clutter_body_prefix": "pact_clutter_",
            "clutter_slot_names": list(CLUTTER_SLOT_NAMES),
            "clutter_jitter_m": CLUTTER_JITTER_M,
            "clutter_immovable_mocap": True,
            "clutter_drawn_from_task_seed_not_intrusion_side": True,
            "clutter_half_x_m": 0.025,
            "clutter_half_y_m": 0.05,
            "clutter_height_m": 0.10,
            "clutter_top_z_m": 0.82,
            "clutter_inner_face_abs_y_m": 0.29,
            "clutter_nominal_rear_outer_x_m": 0.775,
            "clutter_lateral_margin_at_closest_approach_m": 0.028,
        },
        "success_criterion": {
            "implementation": "PickAndPlaceTask.get_info",
            "supported_weight_fraction": 0.5,
            "robot_no_longer_touching_target": True,
            "max_receptacle_position_displacement_m": 0.1,
            "max_receptacle_tilt_radians": 0.7853981633974483,
            "lift_one_centimetre_criterion_on_success_path": False,
        },
        "phase0_gate": {
            "n": N_EXPERT_ROWS,
            "minimum_clean_successes": MIN_CLEAN_SUCCESSES,
            "clean_success_definition": (
                "task_success and zero hazard_bar entries and zero "
                "other_environment entries and zero clutter entries and zero "
                "place_receptacle contact outside placement; preplace is treated "
                "as placement"
            ),
            "report_separately": [
                "grasp_phase_success",
                "place_phase_success_given_grasp",
                "inbound_hazard_contact",
                "outbound_hazard_contact",
                "place_receptacle_outside_placement",
                "clutter_contact",
            ],
            "pass_token": PASS_TOKEN,
            "fail_token": FAIL_TOKEN,
            "on_fail": "stop_without_collection_or_training",
            "no_fifth_attempt": False,
            "fifth_attempt": "receptacle_relocation_and_phase_aware_exemption",
            "sixth_attempt": "fixed_shelf_clutter",
            "sixth_b_attempt": "resite_clutter_laterally_after_cup_contact",
            "sixth_c_attempt": "grow_clutter_presence_hold_inner_face",
            "v5_was_phase0_pass_not_a_failure": True,
            "prior_screens_not_comparable_on_clean_success": True,
            "v6_clean_success_stricter_than_v5": True,
            "a0_clearance_probe_zero_clutter": True,
            "phase0_not_run": False,
            "do_not_freeze_until_zero_clutter_probe": False,
            "reachability_sweep": {
                "path": "diagnostics_output/pact_place_reachability_sweep/analysis.json",
                "sweep_sha256": ATTEMPT5_SWEEP_SHA256,
                "chosen_center_xy_m": [0.35, 0.32],
                "footprint": "shrunk_0.10x0.10",
                "clearance_beyond_traversal_y_m": 0.113,
                "n_candidates": 26,
                "n_reachable": 26,
                "n_clears_8cm": 3,
                "n_eligible": 3,
            },
            "attempt6_prediction": {
                "recorded_before_first_episode": True,
                "prior_screens_clean": [18, 18, 18, 15, 22],
                "v5_failures_clutter_does_not_touch": [3, 10],
                "predicted_clean_successes": [19, 22],
                "predicted_clean_rate": [0.792, 0.917],
                "bar": 20,
                "on_20_or_more": "gate_passes_collect",
                "on_19": "clutter_costs_about_one_row_gate_marginal",
                "on_18_or_fewer": "clutter_interferes_with_grasp_resite_via_a0",
                "note": "Drafted for v6; v6 stopped at A0. v6b uses attempt6b_prediction.",
            },
            "attempt6b_prediction": {
                "recorded_before_first_episode": True,
                "prior_screens_clean": [18, 18, 18, 15, 22],
                "v6_was_a0_stop_not_a_screen": True,
                "v6b_probe_clean": [7, 8],
                "v6b_probe_clutter_episodes": 0,
                "v5_failures_clutter_does_not_touch": [3, 10],
                "predicted_clean_successes": [19, 22],
                "predicted_clean_rate": [0.792, 0.917],
                "bar": 20,
                "on_20_or_more": "gate_passes_collect",
                "on_19": "clutter_costs_about_one_row_gate_marginal",
                "on_18_or_fewer": "clutter_interferes_with_grasp_resite_via_a0",
            },
            "clearance_probe_v6b": {
                "path": "diagnostics_output/pact_place_corridor_v6b_clearance_probe",
                "role": "diagnostic_not_a_gate",
                "n": 8,
                "clutter_contact_episodes": 0,
                "clean": 7,
                "task_success": 7,
                "diagnostic_config_sha256": (
                    "461a04609fa3fb5d540bb9c274f857226b2afe539cf4379f9667625569a91a48"
                ),
            },
            "attempt6c_prediction": {
                "recorded_before_first_episode": True,
                "prior_screens_clean": [18, 18, 18, 15, 22, 20],
                "v6_was_a0_stop_not_a_screen": True,
                "v6b_was_phase0_pass": True,
                "v6b_clean": 20,
                "v6b_clutter_episodes": 0,
                "v6b_failures_not_clutter": [2, 9, 14, 22],
                "v6c_probe_clean": [6, 8],
                "v6c_probe_clutter_episodes": 0,
                "predicted_clean_successes": [19, 22],
                "predicted_clean_rate": [0.792, 0.917],
                "bar": 20,
                "on_20_or_more": "gate_passes_collect",
                "on_19": "honest_and_marginal",
                "on_18_or_fewer": "enlargement_bit_return_to_a0c_not_tuning",
                "note": (
                    "v6b scored 20/24 with four retrieval failures and zero "
                    "clutter. Larger boxes at the same 28 mm inner-face margin "
                    "should not change that."
                ),
            },
            "clearance_probe_v6c": {
                "path": "diagnostics_output/pact_place_corridor_v6c_clearance_probe",
                "role": "diagnostic_not_a_gate",
                "n": 8,
                "clutter_contact_episodes": 0,
                "clean": 6,
                "task_success": 6,
                "top_z_m": 0.82,
                "fallback_to_top_080": False,
                "diagnostic_config_sha256": (
                    "1b18c40cf77e7e7bf817519967edb8f384d7361004f0002aa3f41edb67237c4f"
                ),
            },
        },
        "expert": {
            "class": "PactPlaceCorridorPolicy",
            "inbound_safe_gap_m": 0.10,
            "outbound_safe_gap_m": 0.14,
            "outbound_carried_envelope_half_y_m": 0.15,
            "outbound_carry_raise_m": 0.0,
            "outbound_pass_speed_m_s": 0.045,
            "release_clearance_m": 0.005,
            "outbound_approach_max_step_m": 0.04,
            "initial_observation_rejects_robot_environment_contact": True,
            "empty_gripper_disarmed_on_placement_descent": True,
            "empty_gripper_persist_steps": 3,
            "gripper_empty_threshold_m": 0.002,
            "empty_gripper_repair_is_not_a_threshold_change": True,
            "grasp_servo_duration_s": 2.0,
            "action_noise": False,
            "expert_rollout_sensor_polling": ["qpos", "tcp_pose"],
            "observation_sensors_rationale": (
                "normal sensor/camera geometry is built during sampling, then the "
                "privileged expert's polling suite is reduced to the qpos and tcp_pose "
                "sensors required by the upstream planner/config freezer; physics, "
                "contacts, robot, scene, and sensor bodies remain unchanged"
            ),
            "task_horizon": 900,
        },
        "expert_screen_rows": rows,
        "source_sha256": _source_hashes(),
        "protected_artifact_sha256_before": _protected_artifact_hashes(),
    }
    sweep = _attempt6_sweep_block()
    if sweep is not None:
        document["phase0_gate"]["clutter_sweep"] = sweep
    sweep_b = _attempt6b_sweep_block()
    if sweep_b is not None:
        document["phase0_gate"]["clutter_sweep_v6b"] = sweep_b
    sweep_c = _attempt6c_sweep_block()
    if sweep_c is not None:
        document["phase0_gate"]["clutter_sweep_v6c"] = sweep_c
    document["config_sha256"] = sha256_payload(document)
    validate_contract(document)
    return document


def validate_contract(document: dict[str, Any]) -> None:
    payload = dict(document)
    observed = payload.pop("config_sha256")
    if observed != sha256_payload(payload):
        raise ValueError("config self-hash mismatch")
    rows = document["expert_screen_rows"]
    if len(rows) != N_EXPERT_ROWS:
        raise ValueError(f"expected {N_EXPERT_ROWS} expert rows")
    if sum(row["intrusion_side"] == "left" for row in rows) != N_EXPERT_ROWS // 2:
        raise ValueError("expert screen is not side-balanced")
    if len({row["episode_id"] for row in rows}) != N_EXPERT_ROWS:
        raise ValueError("episode IDs are not unique")
    for row in rows:
        payload = dict(row)
        row_hash = payload.pop("row_sha256")
        if row_hash != sha256_payload(payload):
            raise ValueError(f"row self-hash mismatch at {row['role_index']}")
        if not -0.015 <= row["panel_x_jitter_m"] <= 0.015:
            raise ValueError("panel x jitter outside frozen corridor support")
        if not -0.005 <= row["panel_face_jitter_m"] <= 0.005:
            raise ValueError("panel face jitter outside frozen corridor support")
        if "clutter_x_jitter_m" in row or "clutter_y_jitter_m" in row:
            slot_names = tuple(
                document["scene"].get("clutter_slot_names") or CLUTTER_SLOT_NAMES
            )
            expected_x, expected_y = clutter_jitters_for_seed(
                int(row["task_seed_u64"]),
                slot_names=slot_names,
            )
            if row.get("clutter_x_jitter_m") != expected_x:
                raise ValueError(
                    f"clutter x jitter is not a function of task_seed_u64 at {row['role_index']}"
                )
            if row.get("clutter_y_jitter_m") != expected_y:
                raise ValueError(
                    f"clutter y jitter is not a function of task_seed_u64 at {row['role_index']}"
                )
            for slot in slot_names:
                if not -CLUTTER_JITTER_M <= expected_x[slot] <= CLUTTER_JITTER_M:
                    raise ValueError("clutter x jitter outside frozen support")
                if not -CLUTTER_JITTER_M <= expected_y[slot] <= CLUTTER_JITTER_M:
                    raise ValueError("clutter y jitter outside frozen support")
    scene = document["scene"]
    if not scene["place_tray_x_bounds_m"][1] < scene["aperture_plane_x_m"]:
        raise ValueError("place tray is not wholly outside the aperture")
    prefix = str(scene.get("clutter_body_prefix") or "")
    if prefix:
        if prefix != "pact_clutter_":
            raise ValueError("clutter bodies must use the pact_clutter_ prefix")
        for forbidden in ("cavity_obj_", "pact_intrusion_", "place_receptacle"):
            if forbidden in prefix:
                raise ValueError("clutter prefix collides with an exempt namespace")
        for slot in scene.get("clutter_slot_names") or ():
            body = f"{prefix}{slot}"
            for forbidden in ("cavity_obj_", "pact_intrusion_", "place_receptacle"):
                if forbidden in body:
                    raise ValueError(f"illegal clutter body name {body!r}")


def review_row_seed(index: int, master_seed: int) -> tuple[int, int]:
    digest = hashlib.sha256(
        f"pact-place-v7-review:{int(master_seed)}:{int(index)}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big"), int.from_bytes(digest[:8], "big")


def review_episode_id_for(index: int, side: str, master_seed: int) -> str:
    return hashlib.sha256(
        f"pact-place-v7-review:expert:{int(master_seed)}:{int(index)}:{side}".encode()
    ).hexdigest()


def _v7_clutter_nominal() -> dict[str, dict[str, Any]]:
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV4Sampler

    lattice: dict[str, dict[str, Any]] = {}
    for slot, spec in PactPlaceCorridorV4Sampler.CLUTTER_SLOT_NOMINAL.items():
        lattice[slot] = {
            "center_m": [float(v) for v in spec["center_m"]],
            "half_m": [float(v) for v in spec["half_m"]],
            "support": spec["support"],
            "size_name": spec["size_name"],
        }
    return lattice


def build_design_review_contract(
    master_seed: int | None = None,
) -> dict[str, Any]:
    seed = int(master_seed if master_seed is not None else V7_DESIGN_REVIEW_MASTER_SEED)
    rng = random.Random(seed)
    sides = ["left", "right"] * (N_DESIGN_REVIEW_ROWS // 2)
    rng.shuffle(sides)
    rows = []
    for index, side in enumerate(sides):
        seed_u32, seed_u64 = review_row_seed(index, seed)
        clutter_x, clutter_y = clutter_jitters_for_seed(
            seed_u64, slot_names=V7_CLUTTER_POOL_SLOT_NAMES
        )
        row = {
            "role_index": index,
            "episode_id": review_episode_id_for(index, side, seed),
            "intrusion_side": side,
            "panel_x_jitter_m": round(rng.uniform(-0.015, 0.015), 9),
            "panel_face_jitter_m": round(rng.uniform(-0.005, 0.005), 9),
            "clutter_x_jitter_m": clutter_x,
            "clutter_y_jitter_m": clutter_y,
            "scene_template_house_index": 1,
            "task_seed_u32": seed_u32,
            "task_seed_u64": seed_u64,
            "max_sampling_retries": 4,
        }
        row["row_sha256"] = sha256_payload(row)
        rows.append(row)
    document: dict[str, Any] = {
        "schema_version": "pact_place_corridor_v7_design_review",
        "status": "human_design_review_not_a_gate",
        "role": "human_design_review_not_a_gate",
        "authorizes_collection": False,
        "authorizes_probe": False,
        "authorizes_gate": False,
        "not_a_clean_rate_estimate": True,
        "created_utc": "2026-08-20T00:00:00Z",
        "route": "collision_route_pick_and_place",
        "master_seed": seed,
        "design_review_seeds": {
            "master_seed": seed,
            "stream": "pact-place-v7-review",
            "n_pregenerated": N_DESIGN_REVIEW_ROWS,
            "stop_at_clean_successes": DESIGN_REVIEW_STOP_CLEAN,
            "hard_cap_episodes": N_DESIGN_REVIEW_ROWS,
            "do_not_reuse_for_probe_or_gate": True,
        },
        "scene": {
            "xml": "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v4.xml",
            "sampler_class": "PactPlaceCorridorV4Sampler",
            "xml_sha256": PLACE_V4_SCENE_SHA256,
            "base_forward_m": 0.14,
            "aperture_plane_x_m": 0.58,
            "place_receptacle_center_xyz_m": list(PLACE_V2_CENTER_XYZ_M),
            "place_tray_x_bounds_m": list(PLACE_V2_TRAY_X_BOUNDS_M),
            "place_receptacle_footprint": "shrunk_0.10x0.10",
            "place_receptacle_contact_exempt": False,
            "place_receptacle_exempt_during_placement_including_preplace": True,
            "legacy_contact_classes_unchanged": [
                "grasp_target",
                "hazard_bar",
                "other_environment",
            ],
            "clutter_body_prefix": "pact_clutter_",
            "clutter_slot_names": list(V7_CLUTTER_POOL_SLOT_NAMES),
            "clutter_active_slot_names": list(V7_CLUTTER_ACTIVE_SLOT_NAMES),
            "clutter_jitter_m": CLUTTER_JITTER_M,
            "clutter_immovable_mocap": True,
            "clutter_drawn_from_task_seed_not_intrusion_side": True,
            "clutter_pool_park_xyz_m": [0.0, 0.0, -2.0],
            "clutter_nominal": _v7_clutter_nominal(),
        },
        "success_criterion": {
            "implementation": "PickAndPlaceTask.get_info",
            "supported_weight_fraction": 0.5,
            "robot_no_longer_touching_target": True,
            "max_receptacle_position_displacement_m": 0.1,
            "max_receptacle_tilt_radians": 0.7853981633974483,
            "lift_one_centimetre_criterion_on_success_path": False,
        },
        "expert": {
            "class": "PactPlaceCorridorPolicy",
            "inbound_safe_gap_m": 0.10,
            "outbound_safe_gap_m": 0.14,
            "outbound_carried_envelope_half_y_m": 0.15,
            "outbound_carry_raise_m": 0.0,
            "outbound_pass_speed_m_s": 0.045,
            "release_clearance_m": 0.005,
            "outbound_approach_max_step_m": 0.04,
            "initial_observation_rejects_robot_environment_contact": True,
            "empty_gripper_disarmed_on_placement_descent": True,
            "empty_gripper_persist_steps": 3,
            "gripper_empty_threshold_m": 0.002,
            "empty_gripper_repair_is_not_a_threshold_change": True,
            "grasp_servo_duration_s": 2.0,
            "action_noise": False,
            "task_horizon": 900,
        },
        "expert_screen_rows": rows,
        "protected_artifact_sha256_before": _protected_artifact_hashes(),
    }
    document["config_sha256"] = sha256_payload(document)
    validate_design_review_contract(document)
    return document


def validate_design_review_contract(document: dict[str, Any]) -> None:
    payload = dict(document)
    observed = payload.pop("config_sha256")
    if observed != sha256_payload(payload):
        raise ValueError("design-review config self-hash mismatch")
    if document.get("role") != "human_design_review_not_a_gate":
        raise ValueError("design-review role must not authorize a gate")
    if document.get("authorizes_collection") is not False:
        raise ValueError("design-review must not authorize collection")
    if document.get("authorizes_probe") is not False:
        raise ValueError("design-review must not authorize the probe")
    if document.get("authorizes_gate") is not False:
        raise ValueError("design-review must not authorize the gate")
    rows = document["expert_screen_rows"]
    if len(rows) != N_DESIGN_REVIEW_ROWS:
        raise ValueError(f"expected {N_DESIGN_REVIEW_ROWS} design-review rows")
    if sum(row["intrusion_side"] == "left" for row in rows) != N_DESIGN_REVIEW_ROWS // 2:
        raise ValueError("design-review screen is not side-balanced")
    scene = document["scene"]
    if not str(scene["xml"]).endswith("pact_place_corridor_v4.xml"):
        raise ValueError("design-review scene must be pact_place_corridor_v4.xml")
    if scene.get("sampler_class") != "PactPlaceCorridorV4Sampler":
        raise ValueError("design-review sampler must be PactPlaceCorridorV4Sampler")
    slot_names = tuple(scene.get("clutter_slot_names") or ())
    if slot_names != V7_CLUTTER_POOL_SLOT_NAMES:
        raise ValueError("design-review clutter pool must be slots 00-15")
    for row in rows:
        payload = dict(row)
        row_hash = payload.pop("row_sha256")
        if row_hash != sha256_payload(payload):
            raise ValueError(f"row self-hash mismatch at {row['role_index']}")
        expected_x, expected_y = clutter_jitters_for_seed(
            int(row["task_seed_u64"]), slot_names=slot_names
        )
        if row.get("clutter_x_jitter_m") != expected_x:
            raise ValueError(
                f"clutter x jitter is not a function of task_seed_u64 at {row['role_index']}"
            )
        if row.get("clutter_y_jitter_m") != expected_y:
            raise ValueError(
                f"clutter y jitter is not a function of task_seed_u64 at {row['role_index']}"
            )


def load_design_review_contract(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    validate_design_review_contract(document)
    return document


def load_contract(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    validate_contract(document)
    return document


def _v8_seed(stream: str, index: int, master_seed: int) -> tuple[int, int]:
    digest = hashlib.sha256(
        f"pact-place-v8:{stream}:{int(master_seed)}:{int(index)}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big"), int.from_bytes(digest[:8], "big")


def _v8_rows(
    layouts: list[dict[str, Any]],
    palette: list[dict[str, Any]],
    *,
    stream: str,
    master_seed: int,
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for family in V8_FAMILIES:
        family_layouts = [item for item in layouts if item["family"] == family]
        left = sorted(
            (item for item in family_layouts if item["intrusion_side"] == "left"),
            key=lambda item: item["layout_id"],
        )
        right = sorted(
            (item for item in family_layouts if item["intrusion_side"] == "right"),
            key=lambda item: item["layout_id"],
        )
        if len(left) != 2 or len(right) != 2:
            raise ValueError(f"v8 layout quota is not 2/2 for {family}")
        ordered.extend((left[0], right[0], left[1], right[1]))
    rows: list[dict[str, Any]] = []
    family_attempts: dict[str, int] = {family: 0 for family in V8_FAMILIES}
    for index, layout in enumerate(ordered):
        family = str(layout["family"])
        family_attempts[family] += 1
        seed_u32, seed_u64 = _v8_seed(stream, index, master_seed)
        rng = random.Random(seed_u64)
        side = str(layout["intrusion_side"])
        episode_id = hashlib.sha256(
            f"pact-place-v8:{stream}:expert:{master_seed}:{index}:{family}:{side}".encode()
        ).hexdigest()
        row = {
            "role_index": index,
            "family": family,
            "family_attempt": family_attempts[family],
            "layout_id": layout["layout_id"],
            "episode_id": episode_id,
            "intrusion_side": side,
            "panel_x_jitter_m": round(rng.uniform(-0.015, 0.015), 9),
            "panel_face_jitter_m": round(rng.uniform(-0.005, 0.005), 9),
            "scene_template_house_index": 1,
            "task_seed_u32": seed_u32,
            "task_seed_u64": seed_u64,
            "max_sampling_retries": 4,
            "pact_clutter_palette": palette,
            "pact_clutter_layout": layout,
        }
        row["row_sha256"] = sha256_payload(row)
        rows.append(row)
    return rows


def _v8_source_hashes() -> dict[str, str]:
    relatives = (
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml",
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v1.xml",
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v2.xml",
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v3.xml",
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v4.xml",
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml",
        "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
        "submodules/molmospaces/molmo_spaces/tasks/pact_contact_audit.py",
        "submodules/molmospaces/molmo_spaces/tasks/pact_place_contact_audit.py",
        "scripts/pact_place_corridor_contract.py",
        "scripts/run_pact_place_expert_screen.py",
        "scripts/run_pact_place_v8_baseline.py",
        "scripts/run_pact_place_clutter_sweep_v8.py",
        "scripts/run_pact_place_v8_scoring_check.py",
        "scripts/run_pact_place_v8_family_review.py",
    )
    return {relative: sha256_file(ROOT / relative) for relative in relatives}


def build_v8_contract(
    selected_layouts_path: Path | None = None,
) -> dict[str, Any]:
    path = selected_layouts_path or (ROOT / V8_SELECTED_LAYOUTS_RELATIVE)
    selected = json.loads(path.read_text())
    selected_payload = dict(selected)
    selected_digest = selected_payload.pop("selected_layouts_sha256")
    if selected_digest != sha256_payload(selected_payload):
        raise ValueError("v8 selected-layout document self-hash mismatch")
    palette = list(selected["palette"])
    layouts = list(selected["layouts"])
    family_rows = _v8_rows(
        layouts,
        palette,
        stream="family-review",
        master_seed=V8_FAMILY_REVIEW_MASTER_SEED,
    )
    gate_rows = _v8_rows(
        layouts,
        palette,
        stream="phase0-gate",
        master_seed=V8_GATE_MASTER_SEED,
    )
    document: dict[str, Any] = {
        "schema_version": "pact_place_corridor_v8",
        "status": "family_review_preregistered_gate_not_authorized",
        "role": "human_design_review_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "created_utc": "2026-08-20T00:00:00Z",
        "route": "collision_route_pick_and_place",
        "scene": {
            "xml": "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml",
            "sampler_class": "PactPlaceCorridorV5Sampler",
            "xml_sha256": PLACE_V5_SCENE_SHA256,
            "environment_version": "pact_place_corridor_v5",
            "aperture_plane_x_m": 0.58,
            "place_receptacle_center_xyz_m": list(PLACE_V2_CENTER_XYZ_M),
            "place_tray_x_bounds_m": list(PLACE_V2_TRAY_X_BOUNDS_M),
            "clutter_body_prefix": "pact_clutter_",
            "clutter_movable_free_bodies": True,
            "clutter_added_to_obstacle_aabbs": False,
            "complete_qpos_required": True,
            "unused_palette_objects_parked": True,
        },
        "palette": palette,
        "selected_layouts_path": str(path.resolve().relative_to(ROOT)),
        "selected_layouts_sha256": selected_digest,
        "selected_layouts": layouts,
        "family_review": {
            "role": "human_design_review_not_a_gate",
            "authorizes_gate": False,
            "authorizes_collection": False,
            "master_seed": V8_FAMILY_REVIEW_MASTER_SEED,
            "stream": "pact-place-v8-family-review",
            "one_clean_success_per_family": True,
            "max_attempts_per_family": 4,
            "hard_cap_attempts": 24,
            "not_a_clean_rate_estimate": True,
            "do_not_reuse_for_gate": True,
        },
        "family_review_rows": family_rows,
        "phase0_gate": {
            "requires_explicit_user_approval": True,
            "master_seed": V8_GATE_MASTER_SEED,
            "stream": "pact-place-v8-phase0-gate",
            "n": 24,
            "minimum_clean_successes": MIN_CLEAN_SUCCESSES,
            "side_balance": {"left": 12, "right": 12},
            "family_quota": 4,
            "attempt8_prediction": {
                "recorded_before_first_gate_episode": True,
                "prior_screens_clean": [18, 18, 18, 15, 22, 20, 23],
                "predicted_clean_successes": [19, 23],
                "bar": MIN_CLEAN_SUCCESSES,
                "on_20_or_more": "pass_and_proceed_to_24_episode_pilot",
                "on_19": "honest_and_marginal_report_without_tuning",
                "on_18_or_fewer": "return_to_B2_with_larger_C",
                "note": "v6c scored 23 against a 19-22 prediction; top-of-band is not evidence by itself",
            },
        },
        "expert_screen_rows": gate_rows,
        "expert": {
            "class": "PactPlaceCorridorPolicy",
            "frozen": True,
            "release_clearance_m": 0.005,
            "empty_gripper_persist_steps": 3,
            "initial_observation_rejects_robot_environment_contact": True,
            "task_horizon": 900,
        },
        "source_sha256": _v8_source_hashes(),
        "protected_artifact_sha256_before": _protected_artifact_hashes(),
    }
    document["config_sha256"] = sha256_payload(document)
    validate_v8_contract(document)
    return document


def validate_v8_contract(document: dict[str, Any]) -> None:
    payload = dict(document)
    observed = payload.pop("config_sha256")
    if observed != sha256_payload(payload):
        raise ValueError("v8 config self-hash mismatch")
    scene = document["scene"]
    if scene["sampler_class"] != "PactPlaceCorridorV5Sampler":
        raise ValueError("v8 must use PactPlaceCorridorV5Sampler")
    if not str(scene["xml"]).endswith("pact_place_corridor_v5.xml"):
        raise ValueError("v8 must use pact_place_corridor_v5.xml")
    if sha256_file(ROOT / scene["xml"]) != PLACE_V5_SCENE_SHA256:
        raise ValueError("v8 scene hash mismatch")
    if scene.get("clutter_added_to_obstacle_aabbs") is not False:
        raise ValueError("v8 clutter must not alter the expert speed law")
    palette = document["palette"]
    if not 12 <= len(palette) <= 20:
        raise ValueError("v8 palette must contain 12-20 frozen uids")
    if len({item["uid"] for item in palette}) != len(palette):
        raise ValueError("v8 palette uids are not unique")
    layouts = document["selected_layouts"]
    if len(layouts) != 24:
        raise ValueError("v8 must freeze 24 layouts")
    for family in V8_FAMILIES:
        family_layouts = [item for item in layouts if item["family"] == family]
        if len(family_layouts) != 4:
            raise ValueError(f"v8 family quota mismatch for {family}")
        if sum(item["intrusion_side"] == "left" for item in family_layouts) != 2:
            raise ValueError(f"v8 family side balance mismatch for {family}")
    all_episode_ids: set[str] = set()
    for key in ("family_review_rows", "expert_screen_rows"):
        rows = document[key]
        if len(rows) != 24:
            raise ValueError(f"v8 {key} must contain 24 rows")
        if sum(row["intrusion_side"] == "left" for row in rows) != 12:
            raise ValueError(f"v8 {key} is not side-balanced")
        for row in rows:
            row_payload = dict(row)
            row_hash = row_payload.pop("row_sha256")
            if row_hash != sha256_payload(row_payload):
                raise ValueError(f"v8 row hash mismatch at {key}/{row['role_index']}")
            if row["episode_id"] in all_episode_ids:
                raise ValueError("v8 family-review and gate episode streams overlap")
            all_episode_ids.add(row["episode_id"])
            for obj in row["pact_clutter_layout"]["objects"]:
                body = f"pact_clutter_{obj['palette_slot']}"
                if not body.startswith("pact_clutter_"):
                    raise ValueError("v8 clutter body lacks required prefix")
                for forbidden in ("cavity_obj_", "pact_intrusion_", "place_receptacle"):
                    if forbidden in body:
                        raise ValueError(f"illegal v8 clutter body name {body!r}")
    prediction = document["phase0_gate"]["attempt8_prediction"]
    if prediction["predicted_clean_successes"] != [19, 23]:
        raise ValueError("attempt8 prediction must remain frozen at 19-23")


def load_v8_contract(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    validate_v8_contract(document)
    return document
