"""Frozen row contract for the PACT collision-corridor experiment.

This module is deliberately independent of the hybrid-obstacle safety stack.
It has no MuJoCo or Torch dependency and can be regenerated and tested offline.
Every scientific row is fixed before simulation. Worker identity, execution
order, retry outcome, and file layout are excluded from row identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "pact_collision_manifest_v1"
ENVIRONMENT_VERSION = "pact_collision_corridor_v1"
MASTER_SEED = 20260728
SCENE_TEMPLATE_ID = "pact_collision_corridor_narrow_cup_v1"
SCENE_TEMPLATE_HOUSE_INDEX = 1
MAX_SAMPLING_RETRIES = 4

# Development rows may inform a versioned scene revision. The remaining roles
# are frozen candidates for the named environment version and must not be
# outcome-selected, replaced, or reassigned.
ROLE_COUNTS: dict[str, int] = {
    "development": 8,
    "pilot_train": 24,
    "pilot_eval": 24,
    "full_train": 160,
    "full_validation": 40,
    "confirmatory_eval": 80,
}

ROLE_STREAM_IDS = {
    "development": 100,
    "pilot_train": 101,
    "pilot_eval": 102,
    "full_train": 103,
    "full_validation": 104,
    "confirmatory_eval": 105,
}

TASK_STREAM_ID = 1
GEOMETRY_STREAM_ID = 2
RETRY_STREAM_ID = 3

PANEL_X_JITTER_RANGE_M = (-0.015, 0.015)
PANEL_FACE_JITTER_RANGE_M = (-0.005, 0.005)


class PactContractError(ValueError):
    """The candidate manifest violates its frozen contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(
    candidate_index: int,
    stream_id: int,
    retry_index: int = 0,
    *,
    master_seed: int = MASTER_SEED,
) -> dict[str, int]:
    """Derive a stable seed without spawn-order or process-order dependence."""
    entropy = [int(master_seed), int(candidate_index), int(stream_id), int(retry_index)]
    if any(value < 0 for value in entropy):
        raise PactContractError(f"seed entropy must be non-negative: {entropy}")
    words = np.random.SeedSequence(entropy).generate_state(2, dtype=np.uint32)
    low, high = int(words[0]), int(words[1])
    return {"seed_u32": low, "seed_u64": low | (high << 32)}


def retry_seed(row: dict[str, Any], retry_index: int) -> dict[str, int]:
    return derive_seed(
        int(row["candidate_index"]),
        RETRY_STREAM_ID,
        int(retry_index),
        master_seed=int(row["master_seed"]),
    )


def episode_id_for(candidate_index: int) -> str:
    preimage = "\x1f".join(
        (
            SCHEMA_VERSION,
            str(MASTER_SEED),
            str(int(candidate_index)),
            SCENE_TEMPLATE_ID,
        )
    )
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def _balanced_sides(role: str, count: int) -> list[str]:
    if count % 2:
        raise PactContractError(f"role {role} must have even count, got {count}")
    sides = np.asarray(["left"] * (count // 2) + ["right"] * (count // 2), dtype=object)
    stream = ROLE_STREAM_IDS[role]
    seed = np.random.SeedSequence([MASTER_SEED, stream]).generate_state(1)[0]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(sides)
    return [str(value) for value in sides.tolist()]


def build_manifest(*, source_hashes: dict[str, str], sensor_names: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    candidate_index = 0
    for role, count in ROLE_COUNTS.items():
        sides = _balanced_sides(role, count)
        for role_index, side in enumerate(sides):
            geometry_seed = derive_seed(candidate_index, GEOMETRY_STREAM_ID)
            geometry_rng = np.random.default_rng(geometry_seed["seed_u64"])
            task_seed = derive_seed(candidate_index, TASK_STREAM_ID)
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "environment_version": ENVIRONMENT_VERSION,
                "master_seed": MASTER_SEED,
                "candidate_index": candidate_index,
                "role": role,
                "role_index": role_index,
                "episode_id": episode_id_for(candidate_index),
                "scene_template_id": SCENE_TEMPLATE_ID,
                "scene_template_house_index": SCENE_TEMPLATE_HOUSE_INDEX,
                "intrusion_side": side,
                "panel_x_jitter_m": float(
                    geometry_rng.uniform(*PANEL_X_JITTER_RANGE_M)
                ),
                "panel_face_jitter_m": float(
                    geometry_rng.uniform(*PANEL_FACE_JITTER_RANGE_M)
                ),
                "task_seed_u32": task_seed["seed_u32"],
                "task_seed_u64": task_seed["seed_u64"],
                "max_sampling_retries": MAX_SAMPLING_RETRIES,
            }
            row["row_sha256"] = sha256_payload(row)
            rows.append(row)
            candidate_index += 1

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "master_seed": MASTER_SEED,
        "scene_template_id": SCENE_TEMPLATE_ID,
        "scene_template_house_index": SCENE_TEMPLATE_HOUSE_INDEX,
        "role_counts": dict(ROLE_COUNTS),
        "total_candidates": sum(ROLE_COUNTS.values()),
        "sensor_names": list(sensor_names),
        "sensor_order_sha256": sha256_payload(sensor_names),
        "source_hashes": dict(sorted(source_hashes.items())),
        "geometry_jitter_m": {
            "panel_x": list(PANEL_X_JITTER_RANGE_M),
            "panel_inner_face": list(PANEL_FACE_JITTER_RANGE_M),
        },
        "rows": rows,
    }
    document["manifest_sha256"] = sha256_payload(document)
    validate_manifest(document)
    return document


def validate_manifest(document: dict[str, Any]) -> None:
    payload = dict(document)
    observed_manifest_hash = payload.pop("manifest_sha256", None)
    expected_manifest_hash = sha256_payload(payload)
    if observed_manifest_hash != expected_manifest_hash:
        raise PactContractError(
            f"manifest hash mismatch: {observed_manifest_hash} != {expected_manifest_hash}"
        )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise PactContractError("wrong schema version")
    if document.get("role_counts") != ROLE_COUNTS:
        raise PactContractError("role counts differ from frozen contract")
    rows = document.get("rows", [])
    if len(rows) != sum(ROLE_COUNTS.values()):
        raise PactContractError("wrong row count")
    if len(document.get("sensor_names", [])) != 40:
        raise PactContractError("the contract requires exactly 40 proximity sensors")

    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for role, count in ROLE_COUNTS.items():
        role_rows = [row for row in rows if row.get("role") == role]
        if len(role_rows) != count:
            raise PactContractError(f"role {role} has {len(role_rows)} rows, expected {count}")
        side_counts = {
            side: sum(row.get("intrusion_side") == side for row in role_rows)
            for side in ("left", "right")
        }
        if side_counts != {"left": count // 2, "right": count // 2}:
            raise PactContractError(f"role {role} is not side-balanced: {side_counts}")

    for row in rows:
        row_payload = dict(row)
        observed_row_hash = row_payload.pop("row_sha256", None)
        if observed_row_hash != sha256_payload(row_payload):
            raise PactContractError(f"row {row.get('candidate_index')} hash mismatch")
        candidate_index = int(row["candidate_index"])
        if candidate_index in seen_indices:
            raise PactContractError(f"duplicate candidate index {candidate_index}")
        if row["episode_id"] in seen_ids:
            raise PactContractError(f"duplicate episode ID {row['episode_id']}")
        if row["episode_id"] != episode_id_for(candidate_index):
            raise PactContractError(f"row {candidate_index} episode ID mismatch")
        seen_indices.add(candidate_index)
        seen_ids.add(row["episode_id"])

    expected_indices = set(range(len(rows)))
    if seen_indices != expected_indices:
        raise PactContractError("candidate indices are not contiguous")


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path) as stream:
        document = json.load(stream)
    validate_manifest(document)
    return document


def rows_for_role(document: dict[str, Any], role: str) -> list[dict[str, Any]]:
    if role not in ROLE_COUNTS:
        raise PactContractError(f"unknown role {role!r}")
    return sorted(
        (row for row in document["rows"] if row["role"] == role),
        key=lambda row: int(row["role_index"]),
    )
