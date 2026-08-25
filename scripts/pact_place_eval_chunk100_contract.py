"""Frozen contract for the held-out chunk-100 place rollout evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "pact_place_eval_chunk100_manifest_v1"
ENVIRONMENT_VERSION = "pact_place_corridor_v2"
MASTER_SEED = 2026082101
SCENE_TEMPLATE_ID = "pact_place_corridor_v2"
SCENE_TEMPLATE_HOUSE_INDEX = 1
INSTANCE_COUNT = 40
MAX_SAMPLING_RETRIES = 4
ROLE = "place_chunk100_eval"
TASK_STREAM_ID = 1
GEOMETRY_STREAM_ID = 2
RETRY_STREAM_ID = 3
SIDE_STREAM_ID = 4
PANEL_X_JITTER_RANGE_M = (-0.015, 0.015)
PANEL_FACE_JITTER_RANGE_M = (-0.005, 0.005)


class PactPlaceEvalContractError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def derive_seed(candidate_index: int, stream_id: int, retry_index: int = 0) -> dict[str, int]:
    entropy = [MASTER_SEED, int(candidate_index), int(stream_id), int(retry_index)]
    low, high = (
        int(value) for value in np.random.SeedSequence(entropy).generate_state(2, dtype=np.uint32)
    )
    return {"seed_u32": low, "seed_u64": low | (high << 32)}


def retry_seed(row: dict[str, Any], retry_index: int) -> dict[str, int]:
    entropy = [
        int(row["master_seed"]),
        int(row["candidate_index"]),
        RETRY_STREAM_ID,
        int(retry_index),
    ]
    low, high = (
        int(value) for value in np.random.SeedSequence(entropy).generate_state(2, dtype=np.uint32)
    )
    return {"seed_u32": low, "seed_u64": low | (high << 32)}


def episode_id_for(candidate_index: int) -> str:
    preimage = "\x1f".join(
        (SCHEMA_VERSION, str(MASTER_SEED), str(int(candidate_index)), SCENE_TEMPLATE_ID)
    )
    return hashlib.sha256(preimage.encode()).hexdigest()


def build_manifest(sensor_names: list[str], training_task_seeds: set[int]) -> dict[str, Any]:
    first_sides = np.asarray(["left"] * 10 + ["right"] * 10, dtype=object)
    second_sides = np.asarray(["left"] * 10 + ["right"] * 10, dtype=object)
    np.random.default_rng(derive_seed(0, SIDE_STREAM_ID)["seed_u64"]).shuffle(first_sides)
    np.random.default_rng(derive_seed(20, SIDE_STREAM_ID)["seed_u64"]).shuffle(second_sides)
    sides = np.concatenate((first_sides, second_sides))
    rows = []
    for candidate_index, side in enumerate(sides.tolist()):
        geometry_seed = derive_seed(candidate_index, GEOMETRY_STREAM_ID)
        geometry_rng = np.random.default_rng(geometry_seed["seed_u64"])
        task_seed = derive_seed(candidate_index, TASK_STREAM_ID)
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "environment_version": ENVIRONMENT_VERSION,
            "master_seed": MASTER_SEED,
            "candidate_index": candidate_index,
            "role": ROLE,
            "role_index": candidate_index,
            "episode_id": episode_id_for(candidate_index),
            "scene_template_id": SCENE_TEMPLATE_ID,
            "scene_template_house_index": SCENE_TEMPLATE_HOUSE_INDEX,
            "task_sampler_class": "PactPlaceCorridorV2Sampler",
            "hazard_present": True,
            "intrusion_side": str(side),
            "panel_x_jitter_m": float(geometry_rng.uniform(*PANEL_X_JITTER_RANGE_M)),
            "panel_face_jitter_m": float(geometry_rng.uniform(*PANEL_FACE_JITTER_RANGE_M)),
            "task_seed_u32": task_seed["seed_u32"],
            "task_seed_u64": task_seed["seed_u64"],
            "max_sampling_retries": MAX_SAMPLING_RETRIES,
        }
        row["row_sha256"] = sha256_payload(row)
        rows.append(row)
    eval_seeds = {int(row["task_seed_u64"]) for row in rows}
    overlap = sorted(eval_seeds & training_task_seeds)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "master_seed": MASTER_SEED,
        "scene_template_id": SCENE_TEMPLATE_ID,
        "scene_template_house_index": SCENE_TEMPLATE_HOUSE_INDEX,
        "sensor_names": list(sensor_names),
        "sensor_order_sha256": sha256_payload(sensor_names),
        "role_counts": {ROLE: INSTANCE_COUNT},
        "total_candidates": INSTANCE_COUNT,
        "held_out_seed_audit": {
            "training_seed_count": len(training_task_seeds),
            "evaluation_seed_count": len(eval_seeds),
            "intersection_count": len(overlap),
            "intersection_task_seed_u64": overlap,
        },
        "rows": rows,
    }
    document["manifest_sha256"] = sha256_payload(document)
    validate_manifest(document, training_task_seeds=training_task_seeds)
    return document


def validate_manifest(
    document: dict[str, Any], *, training_task_seeds: set[int] | None = None
) -> None:
    payload = dict(document)
    observed = payload.pop("manifest_sha256", None)
    if observed != sha256_payload(payload):
        raise PactPlaceEvalContractError("manifest self-hash mismatch")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "master_seed": MASTER_SEED,
        "scene_template_id": SCENE_TEMPLATE_ID,
        "scene_template_house_index": SCENE_TEMPLATE_HOUSE_INDEX,
        "role_counts": {ROLE: INSTANCE_COUNT},
        "total_candidates": INSTANCE_COUNT,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise PactPlaceEvalContractError(f"{key} differs from contract")
    if len(document.get("sensor_names", [])) != 40:
        raise PactPlaceEvalContractError("exactly 40 sensors are required")
    rows = document.get("rows", [])
    if len(rows) != INSTANCE_COUNT:
        raise PactPlaceEvalContractError("exactly 40 rows are required")
    if Counter(row["intrusion_side"] for row in rows) != {"left": 20, "right": 20}:
        raise PactPlaceEvalContractError("intrusion sides are not balanced")
    if [int(row["role_index"]) for row in rows] != list(range(INSTANCE_COUNT)):
        raise PactPlaceEvalContractError("role indices are not contiguous")
    for row in rows:
        row_payload = dict(row)
        row_hash = row_payload.pop("row_sha256", None)
        if row_hash != sha256_payload(row_payload):
            raise PactPlaceEvalContractError("row self-hash mismatch")
        if row["task_sampler_class"] != "PactPlaceCorridorV2Sampler":
            raise PactPlaceEvalContractError("place sampler registration changed")
        if row["hazard_present"] is not True:
            raise PactPlaceEvalContractError("hazard_present must be true")
        if any("clutter" in key for key in row):
            raise PactPlaceEvalContractError("v2 rows must not contain clutter fields")
    audit = document.get("held_out_seed_audit", {})
    if audit.get("intersection_count") != 0 or audit.get("intersection_task_seed_u64") != []:
        raise PactPlaceEvalContractError("evaluation seeds overlap training seeds")
    if training_task_seeds is not None:
        eval_seeds = {int(row["task_seed_u64"]) for row in rows}
        if eval_seeds & training_task_seeds:
            raise PactPlaceEvalContractError("evaluation seeds overlap supplied training seeds")


def load_manifest(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    validate_manifest(document)
    return document
