#!/usr/bin/env python3
"""Frozen row contract for held-out PACT geometry generalization."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from pact_frontend_screen_contract import validate_recovery_event as _validate_recovery_event


SCHEMA_VERSION = "pact_geometry_generalization_manifest_v1"
ENVIRONMENT_VERSION = "pact_collision_corridor_geometry_generalization_v1"
MASTER_SEED = 2026080601
SCENE_TEMPLATE_ID = "pact_collision_corridor_narrow_cup_v1"
SCENE_TEMPLATE_HOUSE_INDEX = 1
MAX_SAMPLING_RETRIES = 4
EXPERT_INSTANCES = 12
POLICY_INSTANCES = 25
TASK_STREAM_ID = 1
GEOMETRY_STREAM_ID = 2
SIDE_STREAM_ID = 3
RETRY_STREAM_ID = 4

TRAINING_SUPPORT = {
    "panel_x_m": [0.600, 0.630],
    "panel_inner_face_y_m": [0.095, 0.105],
    "panel_z_m": [0.89, 0.89],
    "aperture_width_m": [0.85, 0.85],
    "base_forward_m": [0.14, 0.14],
    "panel_half_extents_m": [[0.055, 0.240, 0.090], [0.055, 0.240, 0.090]],
}

CONDITIONS = {
    "C0": {
        "label": "in_distribution_control",
        "sampler_class": "PactCollisionCorridorControlSampler",
        "panel_x_m": 0.615,
        "panel_x_jitter_m": [-0.015, 0.015],
        "panel_inner_face_y_m": 0.100,
        "panel_face_jitter_m": [-0.005, 0.005],
        "panel_z_m": 0.89,
        "aperture_width_m": 0.85,
        "moved_axes": [],
    },
    "C1": {
        "label": "deeper_and_higher",
        "sampler_class": "PactCollisionCorridorDeeperHigherSampler",
        "panel_x_m": 0.68,
        "panel_x_jitter_m": [0.0, 0.0],
        "panel_inner_face_y_m": 0.100,
        "panel_face_jitter_m": [-0.005, 0.005],
        "panel_z_m": 0.96,
        "aperture_width_m": 0.85,
        "moved_axes": ["panel_x_m", "panel_z_m"],
    },
    "C2": {
        "label": "tighter",
        "sampler_class": "PactCollisionCorridorTighterSampler",
        "panel_x_m": 0.615,
        "panel_x_jitter_m": [-0.015, 0.015],
        "panel_inner_face_y_m": 0.070,
        "panel_face_jitter_m": [0.0, 0.0],
        "panel_z_m": 0.89,
        "aperture_width_m": 0.70,
        "moved_axes": ["panel_inner_face_y_m", "aperture_width_m"],
    },
    "C3": {
        "label": "shallower_and_wider",
        "sampler_class": "PactCollisionCorridorShallowerWiderSampler",
        "panel_x_m": 0.55,
        "panel_x_jitter_m": [0.0, 0.0],
        "panel_inner_face_y_m": 0.100,
        "panel_face_jitter_m": [-0.005, 0.005],
        "panel_z_m": 0.89,
        "aperture_width_m": 1.00,
        "moved_axes": ["panel_x_m", "aperture_width_m"],
    },
}


class PactGeometryContractError(ValueError):
    """The geometry-generalization contract was violated."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(index: int, stream_id: int, retry_index: int = 0) -> dict[str, int]:
    words = np.random.SeedSequence(
        [MASTER_SEED, int(index), int(stream_id), int(retry_index)]
    ).generate_state(2, dtype=np.uint32)
    low, high = int(words[0]), int(words[1])
    return {"seed_u32": low, "seed_u64": low | (high << 32)}


def retry_seed(row: dict[str, Any], retry_index: int) -> dict[str, int]:
    return derive_seed(int(row["candidate_index"]), RETRY_STREAM_ID, retry_index)


def _side_order(count: int, *, role_offset: int) -> list[str]:
    left = count // 2
    sides = np.asarray(["left"] * left + ["right"] * (count - left), dtype=object)
    seed = np.random.SeedSequence([MASTER_SEED, SIDE_STREAM_ID, role_offset]).generate_state(1)[0]
    np.random.default_rng(int(seed)).shuffle(sides)
    return [str(value) for value in sides]


def _episode_id(role: str, condition_id: str, instance_index: int) -> str:
    return hashlib.sha256(
        "\x1f".join(
            (SCHEMA_VERSION, str(MASTER_SEED), role, condition_id, str(instance_index))
        ).encode()
    ).hexdigest()


def _rows(role: str, count: int, *, task_offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_sides = _side_order(count, role_offset=task_offset)
    for condition_index, (condition_id, condition) in enumerate(CONDITIONS.items()):
        # Odd-sized policy cells alternate the extra side, yielding exact 50/50
        # balance over all four candidates while staying within one row per cell.
        sides = (
            base_sides
            if count % 2 == 0 or condition_index % 2 == 0
            else ["left" if side == "right" else "right" for side in base_sides]
        )
        for instance_index, side in enumerate(sides):
            latent_index = task_offset + instance_index
            task_seed = derive_seed(latent_index, TASK_STREAM_ID)
            geometry_seed = derive_seed(
                task_offset + condition_index * count + instance_index,
                GEOMETRY_STREAM_ID,
            )
            rng = np.random.default_rng(geometry_seed["seed_u64"])
            x_lo, x_hi = condition["panel_x_jitter_m"]
            y_lo, y_hi = condition["panel_face_jitter_m"]
            x_jitter = float(rng.uniform(x_lo, x_hi)) if x_hi > x_lo else float(x_lo)
            face_jitter = float(rng.uniform(y_lo, y_hi)) if y_hi > y_lo else float(y_lo)
            candidate_index = len(rows) + (0 if role == "expert_screen" else 1000)
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "environment_version": ENVIRONMENT_VERSION,
                "master_seed": MASTER_SEED,
                "candidate_index": candidate_index,
                "role": role,
                "role_index": instance_index,
                "condition_id": condition_id,
                "condition_index": condition_index,
                "condition_label": condition["label"],
                "task_sampler_class": condition["sampler_class"],
                "instance_index": instance_index,
                "instance_cluster_id": f"{role}:{instance_index:02d}",
                "episode_id": _episode_id(role, condition_id, instance_index),
                "scene_template_id": SCENE_TEMPLATE_ID,
                "scene_template_house_index": SCENE_TEMPLATE_HOUSE_INDEX,
                "intrusion_side": side,
                "panel_x_jitter_m": x_jitter,
                "panel_face_jitter_m": face_jitter,
                "realized_geometry": {
                    "panel_x_m": condition["panel_x_m"] + x_jitter,
                    "panel_inner_face_y_m": condition["panel_inner_face_y_m"] + face_jitter,
                    "panel_z_m": condition["panel_z_m"],
                    "aperture_width_m": condition["aperture_width_m"],
                    "base_forward_m": 0.14,
                    "panel_half_extents_m": [0.055, 0.240, 0.090],
                },
                "task_seed_u32": task_seed["seed_u32"],
                "task_seed_u64": task_seed["seed_u64"],
                "max_sampling_retries": MAX_SAMPLING_RETRIES,
            }
            row["row_sha256"] = sha256_payload(row)
            rows.append(row)
    return rows


def build_manifest(*, source_hashes: dict[str, str], sensor_names: list[str]) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "master_seed": MASTER_SEED,
        "scene_template_id": SCENE_TEMPLATE_ID,
        "scene_template_house_index": SCENE_TEMPLATE_HOUSE_INDEX,
        "training_support": TRAINING_SUPPORT,
        "conditions": CONDITIONS,
        "expert_screen_gate": {
            "episodes_per_condition": EXPERT_INSTANCES,
            "pass_requirement": "at least 10 of 12 task successes with zero hazard_bar and zero other_environment contacts",
            "minimum_clean_successes": 10,
            "failed_condition_action": "drop_without_retuning",
            "minimum_shifted_conditions_to_continue": 2,
        },
        "planned_policy_design": {
            "instances_per_condition": POLICY_INSTANCES,
            "arms": ["ACT", "PACT", "PACT_PERMUTED"],
            "checkpoint_seeds": [3101, 3102, 3103],
            "workers": 8,
        },
        "sensor_names": list(sensor_names),
        "sensor_order_sha256": sha256_payload(sensor_names),
        "source_hashes": dict(sorted(source_hashes.items())),
        "expert_screen_rows": _rows("expert_screen", EXPERT_INSTANCES, task_offset=0),
        "rows": _rows("policy_eval", POLICY_INSTANCES, task_offset=100),
    }
    document["manifest_sha256"] = sha256_payload(document)
    validate_manifest(document)
    return document


def _outside(value: float, support: list[float]) -> bool:
    return value < float(support[0]) or value > float(support[1])


def validate_manifest(document: dict[str, Any]) -> None:
    payload = dict(document)
    observed = payload.pop("manifest_sha256", None)
    if observed != sha256_payload(payload):
        raise PactGeometryContractError("manifest self-hash mismatch")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise PactGeometryContractError("schema version changed")
    if document.get("conditions") != CONDITIONS or document.get("training_support") != TRAINING_SUPPORT:
        raise PactGeometryContractError("condition geometry changed")
    if len(document.get("sensor_names", [])) != 40:
        raise PactGeometryContractError("exactly 40 proximity sensors are required")
    expert_rows = document.get("expert_screen_rows", [])
    policy_rows = document.get("rows", [])
    if len(expert_rows) != 4 * EXPERT_INSTANCES or len(policy_rows) != 4 * POLICY_INSTANCES:
        raise PactGeometryContractError("row counts changed")
    all_rows = expert_rows + policy_rows
    if len({row["episode_id"] for row in all_rows}) != len(all_rows):
        raise PactGeometryContractError("episode IDs are not unique")
    for row in all_rows:
        payload = dict(row)
        row_hash = payload.pop("row_sha256", None)
        if row_hash != sha256_payload(payload):
            raise PactGeometryContractError("row self-hash mismatch")
        condition = CONDITIONS[row["condition_id"]]
        if row["task_sampler_class"] != condition["sampler_class"]:
            raise PactGeometryContractError("sampler class differs from condition")
        geometry = row["realized_geometry"]
        for axis in condition["moved_axes"]:
            if not _outside(float(geometry[axis]), TRAINING_SUPPORT[axis]):
                raise PactGeometryContractError(
                    f"{row['condition_id']} {axis} lies inside training support"
                )
        if row["condition_id"] != "C0" and len(condition["moved_axes"]) < 2:
            raise PactGeometryContractError("shifted conditions must move at least two axes")
    for role, rows, count in (
        ("expert_screen", expert_rows, EXPERT_INSTANCES),
        ("policy_eval", policy_rows, POLICY_INSTANCES),
    ):
        for condition_id in CONDITIONS:
            cell = [row for row in rows if row["condition_id"] == condition_id]
            if len(cell) != count:
                raise PactGeometryContractError(f"{role} {condition_id} row count changed")
            sides = Counter(row["intrusion_side"] for row in cell)
            if abs(sides["left"] - sides["right"]) > 1:
                raise PactGeometryContractError(f"{role} {condition_id} is not side-balanced")
        total_sides = Counter(row["intrusion_side"] for row in rows)
        if total_sides["left"] != total_sides["right"]:
            raise PactGeometryContractError(f"{role} is not exactly side-balanced")


def load_manifest(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    validate_manifest(document)
    return document


def validate_recovery_event(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    try:
        return _validate_recovery_event(path, **kwargs)
    except ValueError as error:
        raise PactGeometryContractError(str(error)) from error
