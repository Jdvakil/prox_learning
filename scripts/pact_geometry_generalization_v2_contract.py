#!/usr/bin/env python3
"""Attempt-2 Phase-0 contract for the held-out geometry envelope map."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "pact_geometry_generalization_v2_phase0_manifest"
MASTER_SEED = 2026080604
SCENE_TEMPLATE_ID = "pact_collision_corridor_narrow_cup_v1"
SCENE_TEMPLATE_HOUSE_INDEX = 1
MAX_SAMPLING_RETRIES = 4
PHASE0A_EPISODES = 8
PHASE0B_EPISODES = 12
TASK_STREAM = 201
GEOMETRY_STREAM = 202
SIDE_STREAM = 203
RETRY_STREAM = 204

TRAINING_SUPPORT = {
    "panel_x_m": [0.600, 0.630],
    "panel_inner_face_y_m": [0.095, 0.105],
    "panel_z_m": [0.89, 0.89],
    "aperture_width_m": [0.85, 0.85],
    "base_forward_m": [0.14, 0.14],
    "panel_half_y_m": [0.240, 0.240],
}

CANDIDATES = {
    "X_058": {
        "axis": "panel_x_m",
        "value": 0.58,
        "sampler_class": "PactCollisionCorridorPanelX058Sampler",
        "geometry": {"panel_x_m": 0.58},
    },
    "X_065": {
        "axis": "panel_x_m",
        "value": 0.65,
        "sampler_class": "PactCollisionCorridorPanelX065Sampler",
        "geometry": {"panel_x_m": 0.65},
    },
    "Z_085": {
        "axis": "panel_z_m",
        "value": 0.85,
        "sampler_class": "PactCollisionCorridorPanelZ085Sampler",
        "geometry": {"panel_z_m": 0.85},
    },
    "Z_093": {
        "axis": "panel_z_m",
        "value": 0.93,
        "sampler_class": "PactCollisionCorridorPanelZ093Sampler",
        "geometry": {"panel_z_m": 0.93},
    },
    "HALF_Y_018": {
        "axis": "panel_half_y_m",
        "value": 0.180,
        "sampler_class": "PactCollisionCorridorPanelHalfY018Sampler",
        "geometry": {"panel_half_y_m": 0.180},
    },
    "HALF_Y_030": {
        "axis": "panel_half_y_m",
        "value": 0.300,
        "sampler_class": "PactCollisionCorridorPanelHalfY030Sampler",
        "geometry": {"panel_half_y_m": 0.300},
    },
    "AP_W_095": {
        "axis": "aperture_width_m",
        "value": 0.95,
        "sampler_class": "PactCollisionCorridorAperture095Sampler",
        "geometry": {"aperture_width_m": 0.95},
    },
}

SELECTION_PRIORITY = {
    "axis_order": [
        "panel_z_m",
        "panel_half_y_m",
        "panel_x_m",
        "aperture_width_m",
    ],
    "candidate_order_within_axis": {
        "panel_z_m": ["Z_093", "Z_085"],
        "panel_half_y_m": ["HALF_Y_030", "HALF_Y_018"],
        "panel_x_m": ["X_065", "X_058"],
        "aperture_width_m": ["AP_W_095"],
    },
    "slots": 2,
    "distinct_axes_required": True,
    "selection_uses": "pass/fail only; never clean-success rank above threshold",
}


class GeometryV2ContractError(ValueError):
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


def _sides(count: int, role_stream: int) -> list[str]:
    if count % 2:
        raise GeometryV2ContractError("expert screen count must be even")
    values = np.asarray(["left"] * (count // 2) + ["right"] * (count // 2), dtype=object)
    seed = np.random.SeedSequence([MASTER_SEED, SIDE_STREAM, role_stream]).generate_state(1)[0]
    np.random.default_rng(int(seed)).shuffle(values)
    return [str(item) for item in values]


def _episode_id(phase: str, candidate_id: str, instance_index: int) -> str:
    return hashlib.sha256(
        "\x1f".join(
            (SCHEMA_VERSION, str(MASTER_SEED), phase, candidate_id, str(instance_index))
        ).encode()
    ).hexdigest()


def _realized_geometry(candidate: dict[str, Any]) -> dict[str, Any]:
    geometry = {
        "panel_x_m": 0.615,
        "panel_inner_face_y_m": 0.100,
        "panel_z_m": 0.89,
        "aperture_width_m": 0.85,
        "base_forward_m": 0.14,
        "panel_half_extents_m": [0.055, 0.240, 0.090],
        "panel_half_y_m": 0.240,
    }
    geometry.update(candidate["geometry"])
    if candidate["axis"] == "panel_half_y_m":
        geometry["panel_half_extents_m"] = [0.055, candidate["value"], 0.090]
    return geometry


def build_rows(phase: str, count: int, *, candidate_offset: int, task_offset: int) -> list[dict[str, Any]]:
    rows = []
    sides = _sides(count, task_offset)
    for candidate_position, (candidate_id, candidate) in enumerate(CANDIDATES.items()):
        for instance_index, side in enumerate(sides):
            task_seed = derive_seed(task_offset + instance_index, TASK_STREAM)
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "environment_version": "pact_collision_corridor_geometry_generalization_v2",
                "master_seed": MASTER_SEED,
                "candidate_index": candidate_offset + candidate_position * count + instance_index,
                "role": phase,
                "role_index": instance_index,
                "condition_id": candidate_id,
                "condition_label": f"single_axis_{candidate['axis']}_{candidate['value']}",
                "task_sampler_class": candidate["sampler_class"],
                "instance_index": instance_index,
                "instance_cluster_id": f"{phase}:{instance_index:02d}",
                "episode_id": _episode_id(phase, candidate_id, instance_index),
                "scene_template_id": SCENE_TEMPLATE_ID,
                "scene_template_house_index": SCENE_TEMPLATE_HOUSE_INDEX,
                "intrusion_side": side,
                "panel_x_jitter_m": 0.0,
                "panel_face_jitter_m": 0.0,
                "realized_geometry": _realized_geometry(candidate),
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
        "environment_version": "pact_collision_corridor_geometry_generalization_v2",
        "master_seed": MASTER_SEED,
        "scene_template_id": SCENE_TEMPLATE_ID,
        "scene_template_house_index": SCENE_TEMPLATE_HOUSE_INDEX,
        "training_support": TRAINING_SUPPORT,
        "candidates": CANDIDATES,
        "selection_priority_frozen_before_phase0a": SELECTION_PRIORITY,
        "phase0a_gate": {
            "episodes_per_candidate": PHASE0A_EPISODES,
            "minimum_clean_successes": 7,
            "clean_success_definition": "task success and zero hazard_bar and zero other_environment contact entries",
            "failed_candidate_action": "exclude_without_retuning",
        },
        "phase0b_gate": {
            "episodes_per_selected_condition": PHASE0B_EPISODES,
            "minimum_clean_successes": 10,
            "carried_conditions": {
                "C0": {"clean_successes": 11, "n": 12},
                "C2": {"clean_successes": 12, "n": 12},
            },
            "proceed_requirement": "C0 plus at least two shifted conditions",
        },
        "expert_watchdog": {
            "no_completion_seconds": 600,
            "restart_only_if_no_active_initial_observation_accepted": True,
            "restart_all_active_rows_never_subset": True,
        },
        "sensor_names": list(sensor_names),
        "sensor_order_sha256": sha256_payload(sensor_names),
        "source_hashes": dict(sorted(source_hashes.items())),
        "phase0a_rows": build_rows("phase0a_envelope", PHASE0A_EPISODES, candidate_offset=0, task_offset=300),
        "phase0b_candidate_rows": build_rows("phase0b_composed", PHASE0B_EPISODES, candidate_offset=1000, task_offset=400),
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
        raise GeometryV2ContractError("manifest self-hash mismatch")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise GeometryV2ContractError("schema changed")
    if document.get("candidates") != CANDIDATES or document.get("training_support") != TRAINING_SUPPORT:
        raise GeometryV2ContractError("candidate contract changed")
    if document.get("selection_priority_frozen_before_phase0a") != SELECTION_PRIORITY:
        raise GeometryV2ContractError("selection priority changed")
    if len(document.get("sensor_names", [])) != 40:
        raise GeometryV2ContractError("40 sensors required")
    row_groups = (
        (document.get("phase0a_rows", []), PHASE0A_EPISODES),
        (document.get("phase0b_candidate_rows", []), PHASE0B_EPISODES),
    )
    seen = set()
    for rows, expected_per_candidate in row_groups:
        if len(rows) != len(CANDIDATES) * expected_per_candidate:
            raise GeometryV2ContractError("row count changed")
        for candidate_id, candidate in CANDIDATES.items():
            cell = [row for row in rows if row["condition_id"] == candidate_id]
            if len(cell) != expected_per_candidate:
                raise GeometryV2ContractError("candidate cell size changed")
            if Counter(row["intrusion_side"] for row in cell) != {
                "left": expected_per_candidate // 2,
                "right": expected_per_candidate // 2,
            }:
                raise GeometryV2ContractError("candidate side balance changed")
            for row in cell:
                axis = candidate["axis"]
                if not _outside(float(row["realized_geometry"][axis]), TRAINING_SUPPORT[axis]):
                    raise GeometryV2ContractError(f"{candidate_id} is inside training support")
        for row in rows:
            row_payload = dict(row)
            row_hash = row_payload.pop("row_sha256", None)
            if row_hash != sha256_payload(row_payload):
                raise GeometryV2ContractError("row self-hash mismatch")
            if row["episode_id"] in seen:
                raise GeometryV2ContractError("duplicate episode ID")
            seen.add(row["episode_id"])


def select_candidates(pass_fail: dict[str, bool]) -> list[str]:
    if set(pass_fail) != set(CANDIDATES):
        raise GeometryV2ContractError("selection requires every candidate result")
    selected = []
    for axis in SELECTION_PRIORITY["axis_order"]:
        for candidate_id in SELECTION_PRIORITY["candidate_order_within_axis"][axis]:
            if pass_fail[candidate_id]:
                selected.append(candidate_id)
                break
        if len(selected) == SELECTION_PRIORITY["slots"]:
            break
    return selected


def load_manifest(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    validate_manifest(document)
    return document
