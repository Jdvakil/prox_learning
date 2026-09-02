#!/usr/bin/env python3
"""Frozen 100-success V10.11c collection contract."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pact_place_v1011c_contract as environment  # noqa: E402
from pact_place_v105_contract import (  # noqa: E402
    canonical_payload_sha256,
    empty_authorization,
    recompute_payload_sha256,
    sha256_file,
    sha256_payload,
    write_immutable_create_only,
)

CONTRACT_VERSION = "pact_place_v1011c_collection_100_v1"
ENVIRONMENT_VERSION = environment.ENVIRONMENT_VERSION
SAMPLER_CLASS = environment.SAMPLER_CLASS
PLAN_RELATIVE = "docs/PACT_PLACE_V1011C_COLLECTION_100_PLAN.md"

COLLECTION_STREAM = "pact_place_v1011c_collection_100"
COLLECTION_MASTER_SEED = 2026101131
SMOKE_STREAM = "pact_place_v1011c_collection_100_smoke"
SMOKE_MASTER_SEED = 2026101130
HISTORICAL_MASTER_SEEDS = tuple(environment.HISTORICAL_MASTER_SEEDS) + (
    environment.PREFLIGHT_MASTER_SEED,
    environment.REVIEW_MASTER_SEED,
    2026101001,
    2026101002,
    2026101003,
    2026108001,
)

BASE_QUOTA_PER_CELL = 4
BONUS_CELLS = (
    ("F0_target_side_stagger", "left", "neg5"),
    ("F1_inner_panel_stagger", "right", "center"),
    ("F2_outer_panel_stagger", "left", "center"),
    ("F3_aperture_side_stagger", "right", "pos5"),
)
TARGET_SUCCESSES = 100
MAX_SCIENTIFIC_ATTEMPTS = 900
MAX_WALL_CLOCK_HOURS = 16.0
MAX_SAMPLING_RETRIES = environment.MAX_SAMPLING_RETRIES
MIN_FREE_GIB = 2.0
DEFAULT_WORKERS = 8

COLLECTION_ROOT = "diagnostics_output/pact_place_v1011c_collection_100"
DATASET_ROOT = "assets/datagen/pact_place_corridor_v10_11c_100"
CAMERA_SYSTEM = "FrankaSkinHybridCameraSystem"
TABLE_CAMERA = "exo_camera_1"
TABLE_CAMERA_CALIBRATION_KEYS = (
    "extrinsic_cv",
    "cam2world_gl",
    "intrinsic_cv",
)
SCENE_BY_POSE = environment.parent.SCENE_BY_POSE
ACTIVE_CLUTTER_SLOTS = environment.ACTIVE_CLUTTER_SLOTS
OBJECT_LABELS = {
    "01": "tall_route_cylinder",
    "03": "plate_10",
    "04": "plate_22",
    "06": "soap_bottle_11",
    "08": "tall_near_target_cylinder",
    "09": "tall_near_target_box",
}

UPSTREAM_ARTIFACTS = {
    "environment_contract": "diagnostics_output/pact_place_v1011c_contract/contract.json",
    "paired_visibility": "diagnostics_output/pact_place_v1011c_visibility/visibility.json",
    "preflight": "diagnostics_output/pact_place_v1011c_preflight/preflight.json",
    "review_manifest": "diagnostics_output/pact_place_v1011c_review/review_manifest.json",
}
IMPLEMENTATION_PATHS = (
    "scripts/pact_place_v1011c_collection_contract.py",
    "scripts/run_pact_place_v1011c_collect.py",
    "scripts/run_pact_place_v1010_tablecam_validation.py",
    "scripts/run_pact_place_v108_collect.py",
    "scripts/run_pact_place_expert_screen.py",
    "scripts/pact_place_v1011c_contract.py",
    "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    "submodules/molmospaces/molmo_spaces/configs/camera_configs.py",
    "submodules/molmospaces/molmo_spaces/utils/save_utils.py",
)

cells = environment.cells
cell_key = environment.cell_key


def quotas() -> dict[str, int]:
    result = {cell_key(*cell): BASE_QUOTA_PER_CELL for cell in cells()}
    for cell in BONUS_CELLS:
        key = cell_key(*cell)
        if key not in result:
            raise ValueError(f"unknown bonus cell {key}")
        result[key] += 1
    if sum(result.values()) != TARGET_SUCCESSES:
        raise ValueError(f"quota sum {sum(result.values())} != {TARGET_SUCCESSES}")
    return result


def quota_totals() -> dict[str, Any]:
    totals = quotas()
    by_family: dict[str, int] = {}
    by_side: dict[str, int] = {}
    by_pose: dict[str, int] = {}
    for key, count in totals.items():
        family, side, pose = key.split("|")
        by_family[family] = by_family.get(family, 0) + count
        by_side[side] = by_side.get(side, 0) + count
        by_pose[pose] = by_pose.get(pose, 0) + count
    expected_family = {family: 25 for family, _, _ in cells()[::6]}
    if by_family != expected_family:
        raise ValueError(f"family totals are not 25 each: {by_family}")
    if by_side != {"left": 50, "right": 50}:
        raise ValueError(f"side totals are not balanced: {by_side}")
    if by_pose != {"neg5": 33, "center": 34, "pos5": 33}:
        raise ValueError(f"pose totals are wrong: {by_pose}")
    return {
        "by_cell": totals,
        "by_family": by_family,
        "by_side": by_side,
        "by_pose": by_pose,
        "total": sum(totals.values()),
    }


def cell_seed(
    family: str, side: str, pose: str, attempt_index: int
) -> dict[str, int]:
    digest = hashlib.sha256(
        f"{COLLECTION_STREAM}:{COLLECTION_MASTER_SEED}:"
        f"{family}:{side}:{pose}:{int(attempt_index)}".encode()
    ).digest()
    value = int.from_bytes(digest[:8], "big")
    return {"seed_u32": value % (2**32), "seed_u64": value}


def attempt_id(family: str, side: str, pose: str, attempt_index: int) -> str:
    return hashlib.sha256(
        f"{COLLECTION_STREAM}:{COLLECTION_MASTER_SEED}:"
        f"{family}:{side}:{pose}:{int(attempt_index)}:attempt".encode()
    ).hexdigest()


def build_row(
    family: str, side: str, pose: str, attempt_index: int
) -> dict[str, Any]:
    row = environment.build_row(
        family,
        side,
        pose,
        attempt_index,
        stream=COLLECTION_STREAM,
        master_seed=COLLECTION_MASTER_SEED,
        role_index=0,
    )
    seed = cell_seed(family, side, pose, attempt_index)
    if int(row["task_seed_u32"]) != int(seed["seed_u32"]):
        raise AssertionError("environment row and collection seed disagree")
    identifier = attempt_id(family, side, pose, attempt_index)
    row.update(
        {
            "attempt_id": identifier,
            "episode_id": identifier,
            "cell": cell_key(family, side, pose),
            "contract_version": CONTRACT_VERSION,
            "seed_stream": COLLECTION_STREAM,
            "task_seed_u32": int(seed["seed_u32"]),
            "task_seed_u64": int(seed["seed_u64"]),
            "max_sampling_retries": MAX_SAMPLING_RETRIES,
            "pact_v1011c_collection_target": TARGET_SUCCESSES,
            "pact_v1011c_table_camera_required": True,
        }
    )
    row.pop("row_sha256", None)
    row["row_sha256"] = sha256_payload(row)
    return row


def streams_are_disjoint() -> dict[str, Any]:
    new = (SMOKE_MASTER_SEED, COLLECTION_MASTER_SEED)
    overlap = sorted(set(new) & set(HISTORICAL_MASTER_SEEDS))
    return {
        "smoke_master_seed": SMOKE_MASTER_SEED,
        "collection_master_seed": COLLECTION_MASTER_SEED,
        "historical_master_seeds": list(HISTORICAL_MASTER_SEEDS),
        "overlap": overlap,
        "disjoint": not overlap and len(set(new)) == len(new),
    }


def upstream_bindings() -> dict[str, Any]:
    output = {}
    for name, relative in UPSTREAM_ARTIFACTS.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        if name == "environment_contract":
            gate_passed = bool(
                payload.get("schema_version") == environment.CONTRACT_VERSION
                and payload.get("environment_version") == ENVIRONMENT_VERSION
                and payload.get("sampler_class") == SAMPLER_CLASS
            )
        elif name in {"paired_visibility", "preflight"}:
            gate_passed = payload.get("passed") is True
        elif name == "review_manifest":
            gate_passed = payload.get("eligible_for_owner_review") is True
        else:  # pragma: no cover - UPSTREAM_ARTIFACTS is frozen above
            raise AssertionError(f"unregistered upstream artifact {name}")
        output[name] = {
            "path": relative,
            "raw_sha256": sha256_file(path),
            "payload_sha256": recompute_payload_sha256(path),
            "passed": gate_passed,
        }
    if not all(item["passed"] for item in output.values()):
        raise ValueError(f"upstream V10.11c gate is not passed: {output}")
    return output


def implementation_bindings() -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        bindings[relative] = sha256_file(path)
    return {
        "files": bindings,
        "digest": sha256_payload(sorted(bindings.items())),
    }


def build_contract() -> dict[str, Any]:
    document = {
        **empty_authorization(),
        "schema_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "plan": PLAN_RELATIVE,
        "owner_override": {
            "authorized": True,
            "requested_target_successes": TARGET_SUCCESSES,
            "does_not_claim_phase0_pass": True,
        },
        "upstream": upstream_bindings(),
        "implementation": implementation_bindings(),
        "streams": streams_are_disjoint(),
        "collection": {
            "target_successes": TARGET_SUCCESSES,
            "quota_totals": quota_totals(),
            "max_scientific_attempts": MAX_SCIENTIFIC_ATTEMPTS,
            "max_wall_clock_hours": MAX_WALL_CLOCK_HOURS,
            "one_in_flight_per_cell": True,
            "strict_clean_only": True,
        },
        "observations": {
            "camera_system": CAMERA_SYSTEM,
            "wrist_rgb": True,
            "table_camera_rgb": TABLE_CAMERA,
            "table_camera_calibration_keys": list(TABLE_CAMERA_CALIBRATION_KEYS),
            "raw_proximity_sensors": 40,
            "contact_audit_storage": "summary_only",
        },
        "active_clutter_slots": list(ACTIVE_CLUTTER_SLOTS),
        "object_labels": dict(OBJECT_LABELS),
    }
    document["payload_sha256"] = canonical_payload_sha256(document)
    return document


def write_contract(path: Path) -> None:
    write_immutable_create_only(path, build_contract())


__all__ = [name for name in globals() if name.isupper()] + [
    "attempt_id",
    "build_contract",
    "build_row",
    "cell_key",
    "cell_seed",
    "cells",
    "quota_totals",
    "quotas",
    "streams_are_disjoint",
    "implementation_bindings",
    "upstream_bindings",
    "write_contract",
]
