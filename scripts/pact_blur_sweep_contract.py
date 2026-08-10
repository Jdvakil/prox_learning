#!/usr/bin/env python3
"""Frozen instance contract for the inference-time RGB blur sweep."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pact_frontend_screen_contract import (
    validate_recovery_event as _validate_recovery_event,
)
from pact_geometry_generalization_v3_contract import retry_seed as _v3_retry_seed


SCHEMA_VERSION = "pact_blur_sweep_manifest_v1"
ENVIRONMENT_VERSION = "pact_collision_corridor_blur_sweep_v1"
INSTANCE_COUNT = 25
BLUR_SIGMAS = [0.0, 0.5, 1.0, 2.0]


class PactBlurSweepContractError(ValueError):
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


def retry_seed(row: dict[str, Any], retry_index: int) -> dict[str, int]:
    return _v3_retry_seed(row, retry_index)


def validate_manifest(document: dict[str, Any]) -> None:
    payload = dict(document)
    observed = payload.pop("manifest_sha256", None)
    if observed != sha256_payload(payload):
        raise PactBlurSweepContractError("manifest self-hash mismatch")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "blur_sigmas": BLUR_SIGMAS,
        "instance_count": INSTANCE_COUNT,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise PactBlurSweepContractError(f"{key} changed")
    if len(document.get("sensor_names", [])) != 40:
        raise PactBlurSweepContractError("exactly 40 sensors are required")
    rows = document.get("rows", [])
    if len(rows) != INSTANCE_COUNT:
        raise PactBlurSweepContractError("exactly 25 shared instances are required")
    if [int(row["blur_role_index"]) for row in rows] != list(range(INSTANCE_COUNT)):
        raise PactBlurSweepContractError("blur instance indices are not contiguous")
    if len({row["episode_id"] for row in rows}) != INSTANCE_COUNT:
        raise PactBlurSweepContractError("blur episode IDs are not unique")
    sides = Counter(row["intrusion_side"] for row in rows)
    if sides not in ({"left": 12, "right": 13}, {"left": 13, "right": 12}):
        raise PactBlurSweepContractError(f"blur sides are not near-balanced: {sides}")
    for row in rows:
        row_payload = dict(row)
        row_hash = row_payload.pop("row_sha256", None)
        if row_hash != sha256_payload(row_payload):
            raise PactBlurSweepContractError("instance row self-hash mismatch")
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("environment_version") != ENVIRONMENT_VERSION
            or row.get("source_condition_id") != "C0"
            or row.get("task_sampler_class")
            != "PactCollisionCorridorControlSampler"
        ):
            raise PactBlurSweepContractError("instance is not a frozen v3 C0 row")


def load_manifest(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    validate_manifest(document)
    return document


def validate_recovery_event(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    try:
        return _validate_recovery_event(path, **kwargs)
    except ValueError as error:
        raise PactBlurSweepContractError(str(error)) from error
