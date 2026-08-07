#!/usr/bin/env python3
"""Frozen policy-instance contract for geometry generalization attempt 2."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from pact_frontend_screen_contract import validate_recovery_event as _validate_recovery_event


SCHEMA_VERSION = "pact_geometry_generalization_v2_manifest"
ENVIRONMENT_VERSION = "pact_collision_corridor_geometry_generalization_v2"
MASTER_SEED = 2026080605
SCENE_TEMPLATE_ID = "pact_collision_corridor_narrow_cup_v1"
SCENE_TEMPLATE_HOUSE_INDEX = 1
MAX_SAMPLING_RETRIES = 4
POLICY_INSTANCES = 25
TASK_STREAM = 301
GEOMETRY_STREAM = 302
SIDE_STREAM = 303
RETRY_STREAM = 304

TRAINING_SUPPORT = {
    "panel_x_m": [0.600, 0.630],
    "panel_inner_face_y_m": [0.095, 0.105],
    "panel_z_m": [0.89, 0.89],
    "aperture_width_m": [0.85, 0.85],
    "base_forward_m": [0.14, 0.14],
    "panel_half_y_m": [0.240, 0.240],
}

CONDITIONS = {
    "C0": {
        "label": "in_distribution_control_fresh_instances",
        "sampler_class": "PactCollisionCorridorControlSampler",
        "panel_x_m": 0.615,
        "panel_x_jitter_m": [-0.015, 0.015],
        "panel_inner_face_y_m": 0.100,
        "panel_face_jitter_m": [-0.005, 0.005],
        "panel_z_m": 0.89,
        "aperture_width_m": 0.85,
        "panel_half_extents_m": [0.055, 0.240, 0.090],
        "moved_axes": [],
        "phase0_provenance": "attempt1_C0_11_of_12_clean",
    },
    "C2": {
        "label": "tighter_carried_attempt1",
        "sampler_class": "PactCollisionCorridorTighterSampler",
        "panel_x_m": 0.615,
        "panel_x_jitter_m": [-0.015, 0.015],
        "panel_inner_face_y_m": 0.070,
        "panel_face_jitter_m": [0.0, 0.0],
        "panel_z_m": 0.89,
        "aperture_width_m": 0.70,
        "panel_half_extents_m": [0.055, 0.240, 0.090],
        "moved_axes": ["panel_inner_face_y_m", "aperture_width_m"],
        "phase0_provenance": "attempt1_C2_12_of_12_clean_carried_without_rerun",
    },
    "Z_093": {
        "label": "single_axis_panel_z_0_93",
        "sampler_class": "PactCollisionCorridorPanelZ093Sampler",
        "panel_x_m": 0.615,
        "panel_x_jitter_m": [-0.015, 0.015],
        "panel_inner_face_y_m": 0.100,
        "panel_face_jitter_m": [-0.005, 0.005],
        "panel_z_m": 0.93,
        "aperture_width_m": 0.85,
        "panel_half_extents_m": [0.055, 0.240, 0.090],
        "moved_axes": ["panel_z_m"],
        "phase0_provenance": "attempt2_phase0b_11_of_12_clean",
    },
    "HALF_Y_030": {
        "label": "single_axis_panel_half_y_0_30",
        "sampler_class": "PactCollisionCorridorPanelHalfY030Sampler",
        "panel_x_m": 0.615,
        "panel_x_jitter_m": [-0.015, 0.015],
        "panel_inner_face_y_m": 0.100,
        "panel_face_jitter_m": [-0.005, 0.005],
        "panel_z_m": 0.89,
        "aperture_width_m": 0.85,
        "panel_half_extents_m": [0.055, 0.300, 0.090],
        "moved_axes": ["panel_half_y_m"],
        "phase0_provenance": "attempt2_phase0b_11_of_12_clean",
    },
}


class GeometryV2MainContractError(ValueError):
    pass


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


def derive_seed(index: int, stream: int, retry: int = 0) -> dict[str, int]:
    low, high = (
        int(value)
        for value in np.random.SeedSequence(
            [MASTER_SEED, int(index), int(stream), int(retry)]
        ).generate_state(2, dtype=np.uint32)
    )
    return {"seed_u32": low, "seed_u64": low | (high << 32)}


def retry_seed(row: dict[str, Any], retry_index: int) -> dict[str, int]:
    return derive_seed(int(row["candidate_index"]), RETRY_STREAM, retry_index)


def _side_order() -> list[str]:
    values = np.asarray(["left"] * 12 + ["right"] * 13, dtype=object)
    seed = np.random.SeedSequence([MASTER_SEED, SIDE_STREAM]).generate_state(1)[0]
    np.random.default_rng(int(seed)).shuffle(values)
    return [str(item) for item in values]


def _episode_id(condition_id: str, instance_index: int) -> str:
    return hashlib.sha256(
        "\x1f".join(
            (SCHEMA_VERSION, str(MASTER_SEED), "policy_eval", condition_id, str(instance_index))
        ).encode()
    ).hexdigest()


def build_rows() -> list[dict[str, Any]]:
    rows = []
    base_sides = _side_order()
    for condition_index, (condition_id, condition) in enumerate(CONDITIONS.items()):
        sides = (
            base_sides
            if condition_index % 2 == 0
            else ["left" if side == "right" else "right" for side in base_sides]
        )
        for instance_index, side in enumerate(sides):
            task_seed = derive_seed(instance_index, TASK_STREAM)
            geometry_seed = derive_seed(instance_index, GEOMETRY_STREAM)
            rng = np.random.default_rng(geometry_seed["seed_u64"])
            x_lo, x_hi = condition["panel_x_jitter_m"]
            y_lo, y_hi = condition["panel_face_jitter_m"]
            x_jitter = float(rng.uniform(x_lo, x_hi)) if x_hi > x_lo else float(x_lo)
            face_jitter = float(rng.uniform(y_lo, y_hi)) if y_hi > y_lo else float(y_lo)
            panel_half = list(condition["panel_half_extents_m"])
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "environment_version": ENVIRONMENT_VERSION,
                "master_seed": MASTER_SEED,
                "candidate_index": condition_index * POLICY_INSTANCES + instance_index,
                "role": "policy_eval",
                "role_index": instance_index,
                "condition_id": condition_id,
                "condition_index": condition_index,
                "condition_label": condition["label"],
                "task_sampler_class": condition["sampler_class"],
                "instance_index": instance_index,
                "instance_cluster_id": f"policy_eval:{instance_index:02d}",
                "episode_id": _episode_id(condition_id, instance_index),
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
                    "panel_half_extents_m": panel_half,
                    "panel_half_y_m": panel_half[1],
                },
                "task_seed_u32": task_seed["seed_u32"],
                "task_seed_u64": task_seed["seed_u64"],
                "max_sampling_retries": MAX_SAMPLING_RETRIES,
            }
            row["row_sha256"] = sha256_payload(row)
            rows.append(row)
    return rows


def build_manifest(
    *,
    phase0_manifest: dict[str, Any],
    expert_screen: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    surviving = expert_screen.get("surviving_condition_ids")
    if surviving != list(CONDITIONS) or expert_screen.get("continue_to_policy_evaluation") is not True:
        raise GeometryV2MainContractError("Phase 0 did not authorize the exact four-condition design")
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "master_seed": MASTER_SEED,
        "scene_template_id": SCENE_TEMPLATE_ID,
        "scene_template_house_index": SCENE_TEMPLATE_HOUSE_INDEX,
        "training_support": TRAINING_SUPPORT,
        "conditions": CONDITIONS,
        "phase0_bindings": {
            "phase0_manifest_sha256": phase0_manifest["manifest_sha256"],
            "expert_screen_sha256": expert_screen["expert_screen_sha256"],
            "envelope_map_sha256": expert_screen["phase0a_envelope_map_sha256"],
            "phase0b_selection_sha256": expert_screen["phase0b_selection_sha256"],
            "carried_C0_clean": [11, 12],
            "carried_C2_clean": [12, 12],
            "Z_093_clean": [11, 12],
            "HALF_Y_030_clean": [11, 12],
        },
        "planned_policy_design": {
            "instances_per_condition": POLICY_INSTANCES,
            "arms": ["ACT", "PACT", "PACT_PERMUTED"],
            "checkpoint_seeds": [3101, 3102, 3103],
            "workers": 8,
            "rollouts": 900,
        },
        "sensor_names": list(phase0_manifest["sensor_names"]),
        "sensor_order_sha256": phase0_manifest["sensor_order_sha256"],
        "frozen_policy_artifacts": phase0_manifest["frozen_policy_artifacts"],
        "source_hashes": dict(sorted(source_hashes.items())),
        "rows": build_rows(),
    }
    document["manifest_sha256"] = sha256_payload(document)
    validate_manifest(document)
    return document


def _outside(value: float, support: list[float]) -> bool:
    return value < support[0] or value > support[1]


def validate_manifest(document: dict[str, Any]) -> None:
    payload = dict(document)
    observed = payload.pop("manifest_sha256", None)
    if observed != sha256_payload(payload):
        raise GeometryV2MainContractError("manifest self-hash mismatch")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise GeometryV2MainContractError("schema changed")
    if document.get("conditions") != CONDITIONS or document.get("training_support") != TRAINING_SUPPORT:
        raise GeometryV2MainContractError("condition contract changed")
    rows = document.get("rows", [])
    if len(rows) != 100 or len(document.get("sensor_names", [])) != 40:
        raise GeometryV2MainContractError("row or sensor count changed")
    if len({row["episode_id"] for row in rows}) != 100:
        raise GeometryV2MainContractError("duplicate episode IDs")
    for condition_id, condition in CONDITIONS.items():
        cell = [row for row in rows if row["condition_id"] == condition_id]
        if len(cell) != 25:
            raise GeometryV2MainContractError("condition cell count changed")
        sides = Counter(row["intrusion_side"] for row in cell)
        if sorted(sides.values()) != [12, 13]:
            raise GeometryV2MainContractError("condition side balance changed")
        for row in cell:
            row_payload = dict(row)
            row_hash = row_payload.pop("row_sha256", None)
            if row_hash != sha256_payload(row_payload):
                raise GeometryV2MainContractError("row self-hash mismatch")
            geometry = row["realized_geometry"]
            for axis in condition["moved_axes"]:
                if not _outside(float(geometry[axis]), TRAINING_SUPPORT[axis]):
                    raise GeometryV2MainContractError(f"{condition_id} {axis} is inside training support")
            unchanged = set(TRAINING_SUPPORT) - set(condition["moved_axes"])
            for axis in unchanged:
                low, high = TRAINING_SUPPORT[axis]
                if not low <= float(geometry[axis]) <= high:
                    raise GeometryV2MainContractError(f"{condition_id} unexpectedly shifted {axis}")
    if Counter(row["intrusion_side"] for row in rows) != {"left": 50, "right": 50}:
        raise GeometryV2MainContractError("overall side balance changed")
    for instance_index in range(25):
        task_seeds = {
            (row["task_seed_u32"], row["task_seed_u64"])
            for row in rows
            if row["instance_index"] == instance_index
        }
        if len(task_seeds) != 1:
            raise GeometryV2MainContractError("conditions do not share fresh task instances")


def load_manifest(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    validate_manifest(document)
    return document


def validate_recovery_event(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    try:
        return _validate_recovery_event(path, **kwargs)
    except ValueError as error:
        raise GeometryV2MainContractError(str(error)) from error
