"""Fresh-instance and recovery contract for the contact endpoint run."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from pact_frontend_screen_contract import validate_recovery_event as _validate_recovery_event


SCHEMA_VERSION = "pact_contact_endpoint_manifest_v1"
ENVIRONMENT_VERSION = "pact_collision_corridor_v1"
MASTER_SEED = 2026080102
SCENE_TEMPLATE_ID = "pact_collision_corridor_narrow_cup_v1"
SCENE_TEMPLATE_HOUSE_INDEX = 1
INSTANCE_COUNT = 100
MAX_SAMPLING_RETRIES = 4
ROLE = "contact_endpoint_eval"
ROLE_STREAM_ID = 109
TASK_STREAM_ID = 1
GEOMETRY_STREAM_ID = 2
RETRY_STREAM_ID = 3
PANEL_X_JITTER_RANGE_M = (-0.015, 0.015)
PANEL_FACE_JITTER_RANGE_M = (-0.005, 0.005)


class PactContactEndpointContractError(ValueError):
    """The contact-endpoint manifest contract was violated."""


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


def derive_seed(
    candidate_index: int,
    stream_id: int,
    retry_index: int = 0,
    *,
    master_seed: int = MASTER_SEED,
) -> dict[str, int]:
    entropy = [int(master_seed), int(candidate_index), int(stream_id), int(retry_index)]
    if any(value < 0 for value in entropy):
        raise PactContactEndpointContractError("seed entropy must be non-negative")
    low, high = (
        int(value)
        for value in np.random.SeedSequence(entropy).generate_state(2, dtype=np.uint32)
    )
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
        (SCHEMA_VERSION, str(MASTER_SEED), str(int(candidate_index)), SCENE_TEMPLATE_ID)
    )
    return hashlib.sha256(preimage.encode()).hexdigest()


def _balanced_sides() -> list[str]:
    sides = np.asarray(
        ["left"] * (INSTANCE_COUNT // 2) + ["right"] * (INSTANCE_COUNT // 2),
        dtype=object,
    )
    seed = np.random.SeedSequence([MASTER_SEED, ROLE_STREAM_ID]).generate_state(1)[0]
    np.random.default_rng(int(seed)).shuffle(sides)
    return [str(value) for value in sides]


def build_manifest(
    *,
    source_hashes: dict[str, str],
    sensor_names: list[str],
    excluded_episode_ids: set[str],
    excluded_manifests: dict[str, str],
) -> dict[str, Any]:
    rows = []
    for candidate_index, side in enumerate(_balanced_sides()):
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
    episode_ids = {row["episode_id"] for row in rows}
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "master_seed": MASTER_SEED,
        "scene_template_id": SCENE_TEMPLATE_ID,
        "scene_template_house_index": SCENE_TEMPLATE_HOUSE_INDEX,
        "role_counts": {ROLE: INSTANCE_COUNT},
        "total_candidates": INSTANCE_COUNT,
        "sensor_names": list(sensor_names),
        "sensor_order_sha256": sha256_payload(sensor_names),
        "source_hashes": dict(sorted(source_hashes.items())),
        "prior_evaluation_quarantine": {
            "excluded_manifest_sha256s": dict(sorted(excluded_manifests.items())),
            "excluded_episode_id_count": len(excluded_episode_ids),
            "overlap_episode_ids": sorted(episode_ids & excluded_episode_ids),
            "prior_endpoint_loaded": False,
        },
        "geometry_jitter_m": {
            "panel_x": list(PANEL_X_JITTER_RANGE_M),
            "panel_inner_face": list(PANEL_FACE_JITTER_RANGE_M),
        },
        "rows": rows,
    }
    document["manifest_sha256"] = sha256_payload(document)
    validate_manifest(document, excluded_episode_ids=excluded_episode_ids)
    return document


def validate_manifest(
    document: dict[str, Any], *, excluded_episode_ids: set[str] | None = None
) -> None:
    payload = dict(document)
    observed = payload.pop("manifest_sha256", None)
    if observed != sha256_payload(payload):
        raise PactContactEndpointContractError("manifest self-hash mismatch")
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
            raise PactContactEndpointContractError(f"{key} differs from contract")
    if len(document.get("sensor_names", [])) != 40:
        raise PactContactEndpointContractError("exactly 40 sensors are required")
    rows = document.get("rows", [])
    if len(rows) != INSTANCE_COUNT:
        raise PactContactEndpointContractError("exactly 100 rows are required")
    indices = list(range(INSTANCE_COUNT))
    if [int(row["candidate_index"]) for row in rows] != indices or [
        int(row["role_index"]) for row in rows
    ] != indices:
        raise PactContactEndpointContractError("manifest indices are not contiguous")
    episode_ids = {row["episode_id"] for row in rows}
    if len(episode_ids) != INSTANCE_COUNT:
        raise PactContactEndpointContractError("episode IDs are not unique")
    side_counts = Counter(row["intrusion_side"] for row in rows)
    if side_counts != {"left": 50, "right": 50}:
        raise PactContactEndpointContractError(f"sides are not balanced: {side_counts}")
    for row in rows:
        row_payload = dict(row)
        row_hash = row_payload.pop("row_sha256", None)
        if row_hash != sha256_payload(row_payload):
            raise PactContactEndpointContractError("row self-hash mismatch")
        if row["episode_id"] != episode_id_for(int(row["candidate_index"])):
            raise PactContactEndpointContractError("episode ID mismatch")
        if row["role"] != ROLE:
            raise PactContactEndpointContractError("role mismatch")
    if document["prior_evaluation_quarantine"]["overlap_episode_ids"]:
        raise PactContactEndpointContractError("fresh manifest overlaps prior evaluation")
    if excluded_episode_ids is not None and episode_ids & excluded_episode_ids:
        raise PactContactEndpointContractError("manifest overlaps supplied prior IDs")


def load_manifest(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    validate_manifest(document)
    return document


def rows_for_role(document: dict[str, Any], role: str) -> list[dict[str, Any]]:
    if role != ROLE:
        raise PactContactEndpointContractError(f"unknown role {role!r}")
    return sorted(document["rows"], key=lambda row: int(row["role_index"]))


def validate_recovery_event(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    try:
        return _validate_recovery_event(path, **kwargs)
    except ValueError as error:
        raise PactContactEndpointContractError(str(error)) from error
