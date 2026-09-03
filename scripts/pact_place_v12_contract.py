#!/usr/bin/env python3
"""v12 place-corridor contract. Self-contained on main; no V10.x imports.

Env identity: park Soap_Bottle_30, keep Soap_Bottle_11 toward the robot,
hover-then-drop. Clutter palette/layouts are frozen in
configs/pact_place_v12_clutter.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLUTTER_PATH = ROOT / "configs" / "pact_place_v12_clutter.json"
SCENE_RELATIVE = "custom_scenes/pact_place_corridor_v12.xml"

CONTRACT_VERSION_V12 = "pact_place_v12_v1"
CONTRACT_VERSION_V1010 = CONTRACT_VERSION_V12
PLAN_RELATIVE = ""
ENVIRONMENT_VERSION = "pact_place_corridor_v12"
SAMPLER_CLASS = "PactPlaceCorridorV1010FourObjectSampler"
TASK_HORIZON = 1050
PROXIMITY_SENSOR_PERIOD_MS = 16.6667
N_PROXIMITY_SENSORS = 40
PROXIMITY_FRAME_SHAPE = (4, 8, 8)
REQUIRED_ACTION_KEYS = (
    "commanded_action", "ee_pose", "ee_twist", "joint_pos", "joint_pos_rel",
)
REQUIRED_AGENT_KEYS = ("qpos", "qvel")
WRIST_VIDEO_SUFFIX = "_wrist_camera.mp4"

V95_LAYOUT_FAMILY_IDS: tuple[str, ...] = (
    "F0_target_side_stagger",
    "F1_inner_panel_stagger",
    "F2_outer_panel_stagger",
    "F3_aperture_side_stagger",
)
INTRUSION_SIDES: tuple[str, ...] = ("left", "right")
POSE_IDS: tuple[str, ...] = ("neg5", "center", "pos5")
POSE_OFFSETS_M: dict[str, float] = {
    "neg5": -0.005, "center": 0.000, "pos5": +0.005,
}

ACTIVE_CLUTTER_SLOTS = ("01", "03", "04", "06")
INACTIVE_CLUTTER_SLOTS = ("00", "02", "05", "07")
ACTIVE_CLUTTER_COUNT = 4
ACTIVE_CLUTTER_UIDS = {
    "01": "Soap_Bottle_30", "03": "Plate_10",
    "04": "Plate_22", "06": "Soap_Bottle_11",
}
OBJECT_LABELS = {
    "01": "bottle_soap_30", "06": "bottle_soap_11",
    "03": "plate_10", "04": "plate_22",
}

COLLECTION_MASTER_SEED = 2026101001
SPLIT_MASTER_SEED = 2026101002
EVAL_MASTER_SEED = 2026101003
COLLECTION_STREAM = "pact_place_v12"
HISTORICAL_MASTER_SEEDS = (2026082101, 2026082901, 2026082902, 2026083001)

QUOTA_PER_CELL = 6
TARGET_SUCCESSES = 144
MAX_SCIENTIFIC_ATTEMPTS = 900
MAX_WALL_CLOCK_HOURS = 16.0
MAX_SAMPLING_RETRIES = 12

COLLECTION_ROOT = "output/collection_v12"
DATASET_ROOT = "output/pact_place_corridor_v12"
SELECTED_ASSEMBLY = {"x_m": 0.800, "r_neg_m": 0.330, "r_pos_m": 0.300}

# Per-family vessel jitter inherited from the V9.5 layout families.
V95_VESSEL_JITTER = (
    ({"01": -0.015, "06": -0.004}, {"01": -0.004, "06": 0.009}),
    ({"01": -0.005, "06": 0.003}, {"01": 0.003, "06": -0.006}),
    ({"01": 0.006, "06": -0.002}, {"01": -0.002, "06": 0.0045}),
    ({"01": 0.015, "06": 0.004}, {"01": 0.004, "06": -0.009}),
)


def _clutter() -> dict[str, Any]:
    return json.loads(CLUTTER_PATH.read_text())


def _scene_sha256() -> str:
    path = ROOT / SCENE_RELATIVE
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def SCENE_BY_POSE() -> dict[str, dict[str, str]]:
    digest = _scene_sha256()
    entry = {"relative": SCENE_RELATIVE, "sha256": digest}
    return {pose: dict(entry) for pose in POSE_IDS}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def canonical_payload_sha256(document: dict[str, Any]) -> str:
    return sha256_payload(
        {k: v for k, v in document.items() if k != "payload_sha256"}
    )


class ImmutableArtifactError(RuntimeError):
    """Refused to replace an artifact that already exists."""


def write_immutable_create_only(path: Path, document: dict[str, Any]) -> dict[str, str]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in document.items() if k != "payload_sha256"}
    digest = sha256_payload(payload)
    payload["payload_sha256"] = digest
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise ImmutableArtifactError(
                f"refusing to replace an existing artifact: {target}"
            ) from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"payload_sha256": digest, "raw_file_sha256": sha256_file(target)}


def empty_authorization() -> dict[str, bool]:
    return {
        "eligible_for_human_review": False,
        "human_approval_present": False,
        "authorizes_phase0": False,
        "authorizes_gate": False,
        "authorizes_collection": False,
        "authorizes_conversion": False,
        "authorizes_training": False,
        "authorizes_evaluation": False,
        "phase0_passed": False,
    }


def cells() -> list[tuple[str, str, str]]:
    return [
        (family, side, pose)
        for family in V95_LAYOUT_FAMILY_IDS
        for side in INTRUSION_SIDES
        for pose in POSE_IDS
    ]


def cell_key(family: str, side: str, pose: str) -> str:
    return f"{family}|{side}|{pose}"


def cell_seed(
    family: str,
    side: str,
    pose: str,
    attempt_index: int,
    *,
    stream: str = COLLECTION_STREAM,
    master: int = COLLECTION_MASTER_SEED,
) -> dict[str, int]:
    digest = hashlib.sha256(
        f"{stream}:{master}:{family}:{side}:{pose}:{int(attempt_index)}".encode()
    ).digest()
    value = int.from_bytes(digest[:8], "big")
    return {"seed_u32": value % (2**32), "seed_u64": value}


def attempt_id(family: str, side: str, pose: str, attempt_index: int) -> str:
    return hashlib.sha256(
        f"{COLLECTION_STREAM}:{COLLECTION_MASTER_SEED}:"
        f"{family}:{side}:{pose}:{int(attempt_index)}:attempt".encode()
    ).hexdigest()


def v12_row_payload(family_id: str, intrusion_side: str) -> dict[str, Any]:
    clutter = _clutter()
    key = f"{family_id}|{intrusion_side}"
    layout = clutter["layouts"].get(key)
    if layout is None:
        raise ValueError(f"missing frozen layout {key}")
    jitter = V95_VESSEL_JITTER[V95_LAYOUT_FAMILY_IDS.index(family_id)]
    return {
        "family": family_id,
        "layout_family_id": family_id,
        "layout_id": layout["layout_id"],
        "family_attempt": 0,
        "scene_template_house_index": 1,
        "max_sampling_retries": MAX_SAMPLING_RETRIES,
        "clutter_x_jitter_m": dict(jitter[0]),
        "clutter_y_jitter_m": dict(jitter[1]),
        "panel_face_jitter_m": 0.0,
        "panel_x_jitter_m": 0.0,
        "target_x_jitter_m": 0.0,
        "target_y_jitter_m": 0.0,
        "pact_clutter_palette": list(clutter["palette"]),
        "pact_clutter_layout": dict(layout),
    }


def build_row(family: str, side: str, pose: str, attempt_index: int) -> dict[str, Any]:
    seed = cell_seed(family, side, pose, attempt_index)
    payload = v12_row_payload(family, side)
    scene = SCENE_BY_POSE()[pose]
    row: dict[str, Any] = {
        "role_index": 0,
        "attempt_id": attempt_id(family, side, pose, attempt_index),
        "episode_id": attempt_id(family, side, pose, attempt_index),
        "cell": cell_key(family, side, pose),
        "family_id": family,
        "family": family,
        "layout_family_id": family,
        "intrusion_side": side,
        "pose_id": pose,
        "pose_offset_m": POSE_OFFSETS_M[pose],
        "attempt_index": int(attempt_index),
        "seed_stream": COLLECTION_STREAM,
        "task_seed_u32": int(seed["seed_u32"]),
        "task_seed_u64": int(seed["seed_u64"]),
        "max_sampling_retries": MAX_SAMPLING_RETRIES,
        "environment_version": ENVIRONMENT_VERSION,
        "contract_version": CONTRACT_VERSION_V12,
        "sampler_class": SAMPLER_CLASS,
        "task_sampler_class": SAMPLER_CLASS,
        "scene_template_house_index": 1,
        "pact_v106_x_m": float(SELECTED_ASSEMBLY["x_m"]),
        "pact_v106_r_neg_m": float(SELECTED_ASSEMBLY["r_neg_m"]),
        "pact_v106_r_pos_m": float(SELECTED_ASSEMBLY["r_pos_m"]),
        "pact_v106_scene_sha256": scene["sha256"],
        "pact_v1010_scene_relative": scene["relative"],
        "pact_v1010_active_clutter_slots": list(ACTIVE_CLUTTER_SLOTS),
        "pact_v1010_inactive_clutter_slots": list(INACTIVE_CLUTTER_SLOTS),
        "pact_v1010_active_clutter_count": ACTIVE_CLUTTER_COUNT,
        "pact_v1010_active_clutter_uids": dict(ACTIVE_CLUTTER_UIDS),
        **{
            k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
            for k, v in payload.items()
        },
    }
    active = [
        obj
        for obj in row["pact_clutter_layout"]["objects"]
        if str(obj["palette_slot"]) in ACTIVE_CLUTTER_SLOTS
    ]
    if len(active) != ACTIVE_CLUTTER_COUNT:
        raise ValueError(f"row exposes {len(active)} active slots")
    row["pact_v1010_identity_sha256"] = hashlib.sha256(
        json.dumps(
            [
                {
                    "palette_slot": str(obj["palette_slot"]),
                    "uid": str(obj["uid"]),
                    "role": str(obj.get("role", "")),
                }
                for obj in sorted(active, key=lambda item: str(item["palette_slot"]))
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    row["row_sha256"] = sha256_payload(row)
    return row


def row_defects(result: dict[str, Any]) -> list[str]:
    defects: list[str] = []
    if result.get("status") != "complete":
        defects.append(f"status={result.get('status')}")
    if not result.get("task_success"):
        defects.append("task_not_successful")
    if not result.get("grasp_phase_success"):
        defects.append("grasp_phase_failed")
    if not result.get("place_phase_success"):
        defects.append("place_phase_failed")
    if not result.get("cup_lifted_one_cm"):
        defects.append("cup_not_lifted")
    audit = result.get("contact_audit") or {}
    for name, count in (audit.get("contact_class_totals") or {}).items():
        if name != "grasp_target" and int(count) > 0:
            defects.append(f"contact:{name}={count}")
    if result.get("clutter_stability_events"):
        defects.append(
            f"clutter_stability_events={len(result['clutter_stability_events'])}"
        )
    telemetry = result.get("pact_v106_frame_telemetry") or {}
    if not telemetry:
        defects.append("missing_frame_telemetry")
    else:
        if telemetry.get("pendant_robot_or_target_contact_frames"):
            defects.append("pendant_contact")
        if telemetry.get("min_clearance_m") is None:
            defects.append("missing_clearance_telemetry")
    return defects


def build_contract() -> dict[str, Any]:
    scenes = SCENE_BY_POSE()
    document = {
        **empty_authorization(),
        "schema_version": CONTRACT_VERSION_V12,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "scene_relative": SCENE_RELATIVE,
        "four_objects": {
            "active_slots": list(ACTIVE_CLUTTER_SLOTS),
            "inactive_slots": list(INACTIVE_CLUTTER_SLOTS),
            "active_count": ACTIVE_CLUTTER_COUNT,
            "active_uids": dict(ACTIVE_CLUTTER_UIDS),
            "labels": dict(OBJECT_LABELS),
        },
        "collection": {
            "quota_per_cell": QUOTA_PER_CELL,
            "cells": len(cells()),
            "target_successes": TARGET_SUCCESSES,
            "max_scientific_attempts": MAX_SCIENTIFIC_ATTEMPTS,
        },
        "scene_by_pose": scenes,
        "selected_assembly": SELECTED_ASSEMBLY,
        "clutter_config": str(CLUTTER_PATH.relative_to(ROOT)),
    }
    document["config_sha256"] = canonical_payload_sha256(document)
    return document


__all__ = [
    "ACTIVE_CLUTTER_COUNT",
    "ACTIVE_CLUTTER_SLOTS",
    "ACTIVE_CLUTTER_UIDS",
    "COLLECTION_MASTER_SEED",
    "COLLECTION_ROOT",
    "CONTRACT_VERSION_V12",
    "DATASET_ROOT",
    "ENVIRONMENT_VERSION",
    "INTRUSION_SIDES",
    "MAX_SAMPLING_RETRIES",
    "N_PROXIMITY_SENSORS",
    "POSE_IDS",
    "POSE_OFFSETS_M",
    "PROXIMITY_FRAME_SHAPE",
    "PROXIMITY_SENSOR_PERIOD_MS",
    "REQUIRED_ACTION_KEYS",
    "REQUIRED_AGENT_KEYS",
    "SAMPLER_CLASS",
    "SCENE_BY_POSE",
    "SELECTED_ASSEMBLY",
    "TARGET_SUCCESSES",
    "TASK_HORIZON",
    "V95_LAYOUT_FAMILY_IDS",
    "WRIST_VIDEO_SUFFIX",
    "attempt_id",
    "build_contract",
    "build_row",
    "canonical_payload_sha256",
    "cell_key",
    "cell_seed",
    "cells",
    "empty_authorization",
    "row_defects",
    "sha256_file",
    "sha256_payload",
    "write_immutable_create_only",
]


if __name__ == "__main__":
    print(json.dumps(build_contract(), indent=2, sort_keys=True))
