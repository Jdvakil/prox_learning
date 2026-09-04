#!/usr/bin/env python3
"""Continue the stopped V10.11d n100 collect until 200 accepted.

Keeps the n100 seed stream, row IDs, and dataset tree. Doubles the per-cell
quota so the 81 already accepted still count toward 200.
"""

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

import pact_place_v1011b_contract as v1011b  # noqa: E402
import pact_place_v1011d_contract as environment  # noqa: E402
from pact_place_v105_contract import (  # noqa: E402
    canonical_payload_sha256,
    empty_authorization,
    recompute_payload_sha256,
    sha256_file,
    sha256_payload,
    write_immutable_create_only,
)

CONTRACT_VERSION = "pact_place_v1011d_collection_200_from_n100_continue_v1"
ENVIRONMENT_VERSION = environment.ENVIRONMENT_VERSION
SAMPLER_CLASS = environment.SAMPLER_CLASS
POLICY_CLASS = "PactPlaceCorridorPolicy"
PLAN_RELATIVE = "docs/PACT_PLACE_V1011D_RANDOMIZED_CLUTTER_VALIDATION_PLAN.md"

COLLECTION_STREAM = "pact_place_v1011d_collection_100"
COLLECTION_MASTER_SEED = 2026101145
# Mac 6-ep inspect used the official V10.11d review expert.
SMOKE_STREAM = environment.REVIEW_STREAM
SMOKE_MASTER_SEED = environment.REVIEW_MASTER_SEED
HISTORICAL_MASTER_SEEDS = tuple(environment.HISTORICAL_MASTER_SEEDS) + (
    environment.PREFLIGHT_MASTER_SEED,
    environment.REVIEW_MASTER_SEED,
    2026101143,
    2026101144,
)

INSPECT_CELLS = (
    ("F0_target_side_stagger", "left", "neg5"),
    ("F1_inner_panel_stagger", "right", "neg5"),
    ("F1_inner_panel_stagger", "right", "center"),
    ("F0_target_side_stagger", "right", "center"),
    ("F2_outer_panel_stagger", "left", "center"),
    ("F3_aperture_side_stagger", "right", "center"),
)
BASE_QUOTA_PER_CELL = 8
BONUS_CELLS = (
    ("F0_target_side_stagger", "left", "neg5"),
    ("F1_inner_panel_stagger", "right", "center"),
    ("F2_outer_panel_stagger", "left", "center"),
    ("F3_aperture_side_stagger", "right", "pos5"),
)
BONUS_PER_CELL = 2
TARGET_SUCCESSES = 200
MAX_SCIENTIFIC_ATTEMPTS = 1800
MAX_WALL_CLOCK_HOURS = 24.0
MAX_SAMPLING_RETRIES = environment.MAX_SAMPLING_RETRIES
MIN_FREE_GIB = 40.0
DEFAULT_WORKERS = 1 if sys.platform == "darwin" else 4
DEFAULT_GPUS = 2

COLLECTION_ROOT = "diagnostics_output/pact_place_v1011d_n200"
DATASET_ROOT = "output/pact_place_corridor_v10_11d_n100"
CAMERA_SYSTEM = "FrankaSkinHybridCameraSystem"
TABLE_CAMERA = "exo_camera_1"
TABLE_CAMERA_CALIBRATION_KEYS = (
    "extrinsic_cv",
    "cam2world_gl",
    "intrinsic_cv",
)
SCENE_BY_POSE = v1011b.SCENE_BY_POSE
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
    "environment_contract": "diagnostics_output/pact_place_v1011d_contract/contract.json",
    "preflight": "diagnostics_output/pact_place_v1011d_preflight/preflight.json",
    "review_manifest": "diagnostics_output/pact_place_v1011d_review/review_manifest.json",
}
IMPLEMENTATION_PATHS = (
    "scripts/pact_place_v1011d_n200_collection_contract.py",
    "scripts/run_pact_place_v1011d_n200_collect.py",
    "scripts/run_pact_place_v1010_tablecam_validation.py",
    "scripts/run_pact_place_v108_collect.py",
    "scripts/pact_place_v1011d_contract.py",
    "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
)

cells = environment.cells
cell_key = environment.cell_key


def quotas() -> dict[str, int]:
    result = {cell_key(*cell): BASE_QUOTA_PER_CELL for cell in cells()}
    for cell in BONUS_CELLS:
        key = cell_key(*cell)
        if key not in result:
            raise ValueError(f"unknown bonus cell {key}")
        result[key] += BONUS_PER_CELL
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
    return {
        "by_cell": totals,
        "by_family": by_family,
        "by_side": by_side,
        "by_pose": by_pose,
        "total": sum(totals.values()),
        "inspect_cells": [cell_key(*cell) for cell in INSPECT_CELLS],
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
            "pact_v1011d_collection_target": TARGET_SUCCESSES,
            "pact_v1011d_table_camera_required": True,
        }
    )
    row.pop("row_sha256", None)
    row["row_sha256"] = sha256_payload(row)
    return row


def streams_are_disjoint() -> dict[str, Any]:
    review_replay = (
        SMOKE_STREAM == environment.REVIEW_STREAM
        and SMOKE_MASTER_SEED == environment.REVIEW_MASTER_SEED
    )
    overlap = sorted(set((COLLECTION_MASTER_SEED,)) & set(HISTORICAL_MASTER_SEEDS))
    if not review_replay:
        overlap = sorted(
            set((SMOKE_MASTER_SEED, COLLECTION_MASTER_SEED))
            & set(HISTORICAL_MASTER_SEEDS)
        )
    return {
        "smoke_master_seed": SMOKE_MASTER_SEED,
        "collection_master_seed": COLLECTION_MASTER_SEED,
        "historical_master_seeds": list(HISTORICAL_MASTER_SEEDS),
        "overlap": overlap,
        "replays_official_v1011d_review_expert": review_replay,
        "disjoint": not overlap and COLLECTION_MASTER_SEED != SMOKE_MASTER_SEED,
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
        elif name == "preflight":
            gate_passed = payload.get("passed") is True
        elif name == "review_manifest":
            gate_passed = payload.get("eligible_for_owner_review") is True
        else:
            raise AssertionError(f"unregistered upstream artifact {name}")
        output[name] = {
            "path": relative,
            "raw_sha256": sha256_file(path),
            "payload_sha256": recompute_payload_sha256(path),
            "passed": gate_passed,
        }
    if not all(item["passed"] for item in output.values()):
        raise ValueError(f"upstream V10.11d gate is not passed: {output}")
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
        "policy_class": POLICY_CLASS,
        "plan": PLAN_RELATIVE,
        "owner_override": {
            "authorized": True,
            "requested_target_successes": TARGET_SUCCESSES,
            "continues_stopped_n100": True,
            "does_not_claim_phase0_pass": True,
            "mac_inspect_before_batman": True,
            "uses_mac_inspect_collect_files": True,
            "replays_official_v1011d_review_expert": True,
            "retain_failed_rows": True,
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
            "retain_failed_rows": True,
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
