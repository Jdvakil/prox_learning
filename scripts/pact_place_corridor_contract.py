#!/usr/bin/env python3
"""Frozen Phase-0 contract for the forked PACT pick-and-place corridor."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
MASTER_SEED = 2026081801
N_EXPERT_ROWS = 24
MIN_CLEAN_SUCCESSES = 20
PASS_TOKEN = "PACT_PLACE_CORRIDOR_PHASE0_PASS"
FAIL_TOKEN = "PACT_PLACE_CORRIDOR_PHASE0_FAIL"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_payload(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def row_seed(index: int) -> tuple[int, int]:
    digest = hashlib.sha256(f"pact-place-v1:{MASTER_SEED}:{index}".encode()).digest()
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
        "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
        "submodules/molmospaces/molmo_spaces/tasks/pact_contact_audit.py",
        "submodules/molmospaces/molmo_spaces/tasks/pact_place_contact_audit.py",
        "submodules/molmospaces/molmo_spaces/tasks/pick_and_place_task.py",
        "submodules/molmospaces/molmo_spaces/configs/task_configs.py",
        "scripts/pact_place_corridor_contract.py",
        "scripts/run_pact_place_expert_screen.py",
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


def build_contract() -> dict[str, Any]:
    rng = random.Random(MASTER_SEED)
    sides = ["left", "right"] * (N_EXPERT_ROWS // 2)
    rng.shuffle(sides)
    rows = []
    for index, side in enumerate(sides):
        seed_u32, seed_u64 = row_seed(index)
        row = {
            "role_index": index,
            "episode_id": hashlib.sha256(
                f"pact-place-v1:expert:{index}:{side}".encode()
            ).hexdigest(),
            "intrusion_side": side,
            "panel_x_jitter_m": round(rng.uniform(-0.015, 0.015), 9),
            "panel_face_jitter_m": round(rng.uniform(-0.005, 0.005), 9),
            "scene_template_house_index": 1,
            "task_seed_u32": seed_u32,
            "task_seed_u64": seed_u64,
            "max_sampling_retries": 4,
        }
        row["row_sha256"] = sha256_payload(row)
        rows.append(row)
    document: dict[str, Any] = {
        "schema_version": "pact_place_corridor_v1",
        "status": "phase0_preregistered",
        "created_utc": "2026-08-18T00:00:00Z",
        "route": "collision_route_pick_and_place",
        "master_seed": MASTER_SEED,
        "scene": {
            "xml": "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v1.xml",
            "base_forward_m": 0.14,
            "aperture_plane_x_m": 0.58,
            "place_receptacle_center_xyz_m": [0.35, 0.0, 0.0],
            "place_tray_x_bounds_m": [0.25, 0.45],
            "place_receptacle_contact_exempt": True,
            "legacy_contact_classes_unchanged": [
                "grasp_target",
                "hazard_bar",
                "other_environment",
            ],
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
                "other_environment entries; place_receptacle contact exempt"
            ),
            "report_separately": [
                "grasp_phase_success",
                "place_phase_success_given_grasp",
                "inbound_hazard_contact",
                "outbound_hazard_contact",
            ],
            "pass_token": PASS_TOKEN,
            "fail_token": FAIL_TOKEN,
            "on_fail": "stop_without_collection_or_training",
        },
        "expert": {
            "class": "PactPlaceCorridorPolicy",
            "inbound_safe_gap_m": 0.10,
            "outbound_safe_gap_m": 0.14,
            "outbound_carried_envelope_half_y_m": 0.15,
            "outbound_carry_raise_m": 0.0,
            "outbound_pass_speed_m_s": 0.045,
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
    scene = document["scene"]
    if not scene["place_tray_x_bounds_m"][1] < scene["aperture_plane_x_m"]:
        raise ValueError("place tray is not wholly outside the aperture")


def load_contract(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    validate_contract(document)
    return document
