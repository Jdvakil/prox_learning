"""Frozen row contract for the fresh PACT confirmatory R2 population."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "pact_confirmatory_r2_manifest_v1"
ENVIRONMENT_VERSION = "pact_collision_corridor_v1"
MASTER_SEED = 2026073001
SCENE_TEMPLATE_ID = "pact_collision_corridor_narrow_cup_v1"
SCENE_TEMPLATE_HOUSE_INDEX = 1
INSTANCE_COUNT = 160
MAX_SAMPLING_RETRIES = 4
ROLE = "confirmatory_eval"
ROLE_STREAM_ID = 105
TASK_STREAM_ID = 1
GEOMETRY_STREAM_ID = 2
RETRY_STREAM_ID = 3
PANEL_X_JITTER_RANGE_M = (-0.015, 0.015)
PANEL_FACE_JITTER_RANGE_M = (-0.005, 0.005)


class PactR2ContractError(ValueError):
    """The R2 candidate manifest violates its frozen contract."""


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
    entropy = [
        int(master_seed),
        int(candidate_index),
        int(stream_id),
        int(retry_index),
    ]
    if any(value < 0 for value in entropy):
        raise PactR2ContractError(f"seed entropy must be non-negative: {entropy}")
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
    r1_episode_ids: set[str],
    r1_manifest_sha256: str,
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
        "r1_quarantine": {
            "r1_manifest_sha256": r1_manifest_sha256,
            "r1_episode_id_count": len(r1_episode_ids),
            "overlap_episode_ids": sorted(
                {row["episode_id"] for row in rows} & r1_episode_ids
            ),
            "r1_endpoint_loaded": False,
        },
        "geometry_jitter_m": {
            "panel_x": list(PANEL_X_JITTER_RANGE_M),
            "panel_inner_face": list(PANEL_FACE_JITTER_RANGE_M),
        },
        "rows": rows,
    }
    document["manifest_sha256"] = sha256_payload(document)
    validate_manifest(document, r1_episode_ids=r1_episode_ids)
    return document


def validate_manifest(
    document: dict[str, Any],
    *,
    r1_episode_ids: set[str] | None = None,
) -> None:
    payload = dict(document)
    observed = payload.pop("manifest_sha256", None)
    if observed != sha256_payload(payload):
        raise PactR2ContractError("manifest self-hash mismatch")
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
            raise PactR2ContractError(f"{key} differs from R2 contract")
    if len(document.get("sensor_names", [])) != 40:
        raise PactR2ContractError("R2 requires exactly 40 proximity sensors")
    rows = document.get("rows", [])
    if len(rows) != INSTANCE_COUNT:
        raise PactR2ContractError("R2 requires exactly 160 rows")
    if [int(row["role_index"]) for row in rows] != list(range(INSTANCE_COUNT)):
        raise PactR2ContractError("role indices are not contiguous")
    if [int(row["candidate_index"]) for row in rows] != list(range(INSTANCE_COUNT)):
        raise PactR2ContractError("candidate indices are not contiguous")
    if len({row["episode_id"] for row in rows}) != INSTANCE_COUNT:
        raise PactR2ContractError("episode IDs are not unique")
    side_counts = {
        side: sum(row["intrusion_side"] == side for row in rows)
        for side in ("left", "right")
    }
    if side_counts != {"left": 80, "right": 80}:
        raise PactR2ContractError(f"R2 sides are not balanced: {side_counts}")
    for row in rows:
        row_payload = dict(row)
        row_hash = row_payload.pop("row_sha256", None)
        if row_hash != sha256_payload(row_payload):
            raise PactR2ContractError("row self-hash mismatch")
        if row["episode_id"] != episode_id_for(int(row["candidate_index"])):
            raise PactR2ContractError("episode ID mismatch")
        if row["role"] != ROLE:
            raise PactR2ContractError("row role mismatch")
    overlap = set(document["r1_quarantine"]["overlap_episode_ids"])
    if overlap:
        raise PactR2ContractError("R2 episode IDs overlap R1")
    if r1_episode_ids is not None:
        actual_overlap = {row["episode_id"] for row in rows} & r1_episode_ids
        if actual_overlap:
            raise PactR2ContractError("R2 episode IDs overlap supplied R1 IDs")


def load_manifest(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    validate_manifest(document)
    return document


def validate_recovery_event(
    path: str | Path,
    *,
    rollout_id: str,
    schedule_row_sha256: str,
    attempt_index: int,
) -> dict[str, Any]:
    event = json.loads(Path(path).read_text())
    payload = dict(event)
    observed = payload.pop("recovery_event_sha256", None)
    if observed != sha256_payload(payload):
        raise PactR2ContractError(
            "R2 in-flight recovery event self-hash mismatch"
        )
    if (
        event.get("schema_version") != "pact_r2_group_recovery_v1"
        or event.get("qualifying_indiscriminate_termination") is not True
        or event.get("all_inflight_rows_rerun") is not True
        or event.get("result_absent_for_all") is not True
    ):
        raise PactR2ContractError(
            "R2 recovery event does not authorize group recovery"
        )
    matches = [
        row
        for row in event.get("rows", [])
        if row.get("rollout_id") == rollout_id
        and row.get("schedule_row_sha256") == schedule_row_sha256
    ]
    if len(matches) != 1 or matches[0].get("result_present") is not False:
        raise PactR2ContractError(
            "R2 recovery event does not contain this result-free row"
        )
    if int(matches[0].get("attempt_index", -1)) + 1 != int(attempt_index):
        raise PactR2ContractError(
            "R2 recovery event is not for the immediately preceding attempt"
        )
    if len(event["rows"]) != int(event["active_cohort_size"]):
        raise PactR2ContractError("R2 recovery event cohort size mismatch")
    return event


def rows_for_role(document: dict[str, Any], role: str) -> list[dict[str, Any]]:
    if role != ROLE:
        raise PactR2ContractError(f"unknown R2 role {role!r}")
    return sorted(document["rows"], key=lambda row: int(row["role_index"]))
