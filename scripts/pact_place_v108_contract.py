#!/usr/bin/env python3
"""V10.8 exploratory demonstration collection: contract, quotas, seed streams.

**This is an explicit owner override for scientific curiosity. It is NOT a
Phase-0 pass.** V10.7's Phase-0 gate failed at 8/24 and is permanently closed;
nothing here changes, reinterprets, or reruns it. V10.8 uses new contracts,
streams, manifests and output directories, and reuses the V10.7 environment
exactly: same static pendant geometry and three poses, same expert policy,
routes, speeds, sensors, and clean-success definition.

Everything a later reader would need to check the collection was fair is frozen
here before the first attempt: the per-cell quotas, the deterministic
cell-specific seed streams, the round-robin schedule, and the budget.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v105_contract import (  # noqa: E402
    ImmutableArtifactError,
    canonical_payload_sha256,
    empty_authorization,
    recompute_payload_sha256,
    sha256_file,
    sha256_payload,
    v95_row_payload,
    write_immutable_create_only,
    write_immutable_text_create_only,
)
from pact_place_v106_contract import INTRUSION_SIDES, V95_LAYOUT_FAMILY_IDS  # noqa: E402
from pact_place_v106_geometry import (  # noqa: E402
    ENVIRONMENT_VERSION_V106,
    POSE_IDS,
    POSE_OFFSETS_M,
    SAMPLER_CLASS_V106,
)

CONTRACT_VERSION_V108 = "pact_place_v108_exploratory_collection_v1"
ENVIRONMENT_VERSION = ENVIRONMENT_VERSION_V106
SAMPLER_CLASS = SAMPLER_CLASS_V106
PLAN_RELATIVE = "docs/PACT_PLACE_V108_EXPLORATORY_COLLECTION_PLAN.md"

IS_PHASE0_PASS = False
IS_EXPLORATORY_OWNER_OVERRIDE = True
V107_PHASE0_RESULT = "failed_8_of_24_permanently_closed"

COLLECTION_ROOT = "diagnostics_output/pact_place_v108_collection"
DATASET_ROOT = "assets/datagen/pact_place_corridor_v10_8"

# Upstream, read-only. V10.7 artifacts are inputs and are never modified.
V107_CERT_JSON = "diagnostics_output/pact_place_v107_certification/certification.json"
V107_SELECTION_JSON = "diagnostics_output/pact_place_v107_selection/selection.json"
V107_PHASE0_JSON = "diagnostics_output/pact_place_v107_phase0/gate.json"
V107_POOL_JSON = "diagnostics_output/pact_place_v107_pool/pool.json"

# ---------------------------------------------------------------------------
# Quotas. Frozen before any attempt; never relaxed or redistributed.
# ---------------------------------------------------------------------------
BASE_QUOTA_PER_CELL = 6
BONUS_CELLS: tuple[tuple[str, str, str], ...] = (
    ("F0_target_side_stagger", "left", "neg5"),
    ("F0_target_side_stagger", "right", "center"),
    ("F1_inner_panel_stagger", "left", "center"),
    ("F1_inner_panel_stagger", "right", "pos5"),
    ("F2_outer_panel_stagger", "left", "pos5"),
    ("F2_outer_panel_stagger", "right", "neg5"),
    ("F3_aperture_side_stagger", "left", "center"),
    ("F3_aperture_side_stagger", "right", "pos5"),
)
TARGET_SUCCESSES = 152

# Budget. Hard stop; never extended.
MAX_SCIENTIFIC_ATTEMPTS = 900
MAX_WALL_CLOCK_HOURS = 16.0

# Streams, distinct from every V10.5/V10.6/V10.7 stream.
COLLECTION_STREAM = "pact-place-v10.8-exploratory-collection"
COLLECTION_MASTER_SEED = 2026108001

MAX_SAMPLING_RETRIES = 12
TASK_HORIZON = 1050
PROXIMITY_SENSOR_PERIOD_MS = 16.6667
N_PROXIMITY_SENSORS = 40
PROXIMITY_FRAME_SHAPE = (4, 8, 8)
DEFAULT_WORKERS = 12
REPORT_AFTER_ATTEMPTS = 24

# Trainable schema. An accepted episode must satisfy every one of these.
REQUIRED_ACTION_KEYS = (
    "commanded_action", "ee_pose", "ee_twist", "joint_pos", "joint_pos_rel",
)
REQUIRED_AGENT_KEYS = ("qpos", "qvel")
WRIST_VIDEO_SUFFIX = "_wrist_camera.mp4"

IMPLEMENTATION_PATHS: tuple[str, ...] = (
    PLAN_RELATIVE,
    "scripts/pact_place_v108_contract.py",
    "scripts/run_pact_place_v108_collect.py",
    "scripts/verify_pact_place_v108_dataset.py",
    "scripts/pact_place_v107_contract.py",
    "scripts/pact_place_v106_geometry.py",
    "scripts/pact_place_v106_contract.py",
    "scripts/pact_place_v95_contract.py",
    "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    "submodules/molmospaces/molmo_spaces/tasks/pact_place_contact_audit.py",
    "tests/test_pact_place_v108.py",
)


def cells() -> list[tuple[str, str, str]]:
    """The 24 family x side x pose cells, in frozen deterministic order."""
    return [
        (family, side, pose)
        for family in V95_LAYOUT_FAMILY_IDS
        for side in INTRUSION_SIDES
        for pose in POSE_IDS
    ]


def cell_key(family: str, side: str, pose: str) -> str:
    return f"{family}|{side}|{pose}"


def quotas() -> dict[str, int]:
    """Base 6 per cell plus the eight registered bonus cells = 152."""
    out = {cell_key(*c): BASE_QUOTA_PER_CELL for c in cells()}
    for bonus in BONUS_CELLS:
        key = cell_key(*bonus)
        if key not in out:
            raise ValueError(f"bonus cell is not a registered cell: {key}")
        out[key] += 1
    total = sum(out.values())
    if total != TARGET_SUCCESSES:
        raise ValueError(f"quotas total {total}, expected {TARGET_SUCCESSES}")
    return out


def quota_totals() -> dict[str, Any]:
    """Derived family/side/pose totals, asserted against the registered values."""
    q = quotas()
    by_family: dict[str, int] = {f: 0 for f in V95_LAYOUT_FAMILY_IDS}
    by_side: dict[str, int] = {s: 0 for s in INTRUSION_SIDES}
    by_pose: dict[str, int] = {p: 0 for p in POSE_IDS}
    for key, value in q.items():
        family, side, pose = key.split("|")
        by_family[family] += value
        by_side[side] += value
        by_pose[pose] += value
    expected_family = {f: 38 for f in V95_LAYOUT_FAMILY_IDS}
    expected_side = {s: 76 for s in INTRUSION_SIDES}
    expected_pose = {"neg5": 50, "center": 51, "pos5": 51}
    if by_family != expected_family:
        raise ValueError(f"family totals {by_family} != {expected_family}")
    if by_side != expected_side:
        raise ValueError(f"side totals {by_side} != {expected_side}")
    if by_pose != expected_pose:
        raise ValueError(f"pose totals {by_pose} != {expected_pose}")
    return {
        "by_cell": q, "by_family": by_family, "by_side": by_side,
        "by_pose": by_pose, "total": sum(q.values()),
    }


# ---------------------------------------------------------------------------
# Deterministic cell-specific seed streams, frozen before any attempt
# ---------------------------------------------------------------------------
def cell_seed(family: str, side: str, pose: str, attempt_index: int) -> dict[str, int]:
    """The attempt_index-th seed of one cell's own stream.

    Each cell has its own stream, so a cell that needs many attempts never
    consumes another cell's seeds and the schedule stays reproducible whatever
    order cells are visited in.
    """
    digest = hashlib.sha256(
        f"{COLLECTION_STREAM}:{COLLECTION_MASTER_SEED}:"
        f"{family}:{side}:{pose}:{int(attempt_index)}".encode()
    ).digest()
    seed_u64 = int.from_bytes(digest[:8], "big")
    return {"seed_u32": seed_u64 % (2**32), "seed_u64": seed_u64}


def attempt_id(family: str, side: str, pose: str, attempt_index: int) -> str:
    return hashlib.sha256(
        f"{COLLECTION_STREAM}:{COLLECTION_MASTER_SEED}:"
        f"{family}:{side}:{pose}:{int(attempt_index)}:attempt".encode()
    ).hexdigest()


def round_robin_schedule(
    accepted: dict[str, int], attempted: dict[str, int], budget: int
) -> list[dict[str, Any]]:
    """Cells with unmet quotas, in deterministic round-robin order.

    Quotas are never relaxed or redistributed: a cell drops out of the rotation
    only when its own quota is met, and no other cell inherits its shortfall.
    """
    q = quotas()
    order = [cell_key(*c) for c in cells()]
    plan: list[dict[str, Any]] = []
    projected_accepted = dict(accepted)
    projected_attempted = dict(attempted)
    while len(plan) < budget:
        progressed = False
        for key in order:
            if len(plan) >= budget:
                break
            if projected_accepted.get(key, 0) >= q[key]:
                continue
            family, side, pose = key.split("|")
            index = projected_attempted.get(key, 0)
            plan.append({
                "cell": key, "family_id": family, "intrusion_side": side,
                "pose_id": pose, "attempt_index": index,
                "attempt_id": attempt_id(family, side, pose, index),
                "seed": cell_seed(family, side, pose, index),
            })
            projected_attempted[key] = index + 1
            # One in flight per cell per round: the next round re-reads the
            # real accepted counts, so a cell that succeeds stops immediately.
            progressed = True
        if not progressed:
            break
    return plan


def next_attempts(
    accepted: dict[str, int], attempted: dict[str, int], count: int
) -> list[dict[str, Any]]:
    return round_robin_schedule(accepted, attempted, count)


def remaining_quota(accepted: dict[str, int]) -> dict[str, int]:
    q = quotas()
    return {k: max(0, q[k] - accepted.get(k, 0)) for k in q}


def quotas_met(accepted: dict[str, int]) -> bool:
    return all(v == 0 for v in remaining_quota(accepted).values())


# ---------------------------------------------------------------------------
# Manifest rows
# ---------------------------------------------------------------------------
def build_attempt_row(
    attempt: dict[str, Any], *, selected: dict[str, float],
    scene_by_pose: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """A V9.5 row plus the frozen V10.7 pendant binding and this attempt's seed."""
    family, side, pose = (
        attempt["family_id"], attempt["intrusion_side"], attempt["pose_id"])
    payload = v95_row_payload(family, side)
    row: dict[str, Any] = {
        "role_index": 0,
        "attempt_id": attempt["attempt_id"],
        "episode_id": attempt["attempt_id"],
        "cell": attempt["cell"],
        "family_id": family, "family": family, "layout_family_id": family,
        "intrusion_side": side,
        "pose_id": pose,
        "pose_offset_m": POSE_OFFSETS_M[pose],
        "attempt_index": int(attempt["attempt_index"]),
        "seed_stream": COLLECTION_STREAM,
        "task_seed_u32": int(attempt["seed"]["seed_u32"]),
        "task_seed_u64": int(attempt["seed"]["seed_u64"]),
        "max_sampling_retries": MAX_SAMPLING_RETRIES,
        "environment_version": ENVIRONMENT_VERSION,
        "contract_version": CONTRACT_VERSION_V108,
        "sampler_class": SAMPLER_CLASS,
        "pact_v106_x_m": float(selected["x_m"]),
        "pact_v106_r_neg_m": float(selected["r_neg_m"]),
        "pact_v106_r_pos_m": float(selected["r_pos_m"]),
        "pact_v106_scene_sha256": scene_by_pose[pose]["sha256"],
        "pact_v108_scene_relative": scene_by_pose[pose]["relative"],
        **{k: (dict(v) if isinstance(v, dict)
               else list(v) if isinstance(v, list) else v)
           for k, v in payload.items()},
    }
    row["row_sha256"] = sha256_payload(row)
    return row


# ---------------------------------------------------------------------------
# Clean-success definition — unchanged from V10.7
# ---------------------------------------------------------------------------
def row_defects(result: dict[str, Any]) -> list[str]:
    from pact_place_v107_contract import row_defects as v107_row_defects

    return v107_row_defects(result)


def is_clean(result: dict[str, Any]) -> bool:
    return not row_defects(result)


def file_hashes(paths: Sequence[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for relative in paths:
        target = ROOT / relative
        out[relative] = (
            {"raw_file_sha256": sha256_file(target), "present": True}
            if target.is_file()
            else {"raw_file_sha256": "absent", "present": False}
        )
    return out


def implementation_digest() -> str:
    return sha256_payload(
        [[p, file_hashes([p])[p]["raw_file_sha256"]] for p in IMPLEMENTATION_PATHS]
    )


def build_contract() -> dict[str, Any]:
    totals = quota_totals()
    return {
        "schema_version": "pact_place_v108_contract_v1",
        "contract_version": CONTRACT_VERSION_V108,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "role": "exploratory demonstration collection under owner override",
        "is_phase0_pass": IS_PHASE0_PASS,
        "is_exploratory_owner_override": IS_EXPLORATORY_OWNER_OVERRIDE,
        "v107_phase0_result": V107_PHASE0_RESULT,
        "v107_preserved_unmodified": True,
        "reuses_v107_pool_or_phase0_rows": False,
        "uses_full_datagen_pipeline": True,
        "uses_expert_screen_harness": False,
        "target_successes": TARGET_SUCCESSES,
        "quotas": totals,
        "base_quota_per_cell": BASE_QUOTA_PER_CELL,
        "bonus_cells": [cell_key(*c) for c in BONUS_CELLS],
        "quota_relaxation_permitted": False,
        "quota_redistribution_permitted": False,
        "schedule": "deterministic round-robin over cells with unmet quotas",
        "budget": {
            "max_scientific_attempts": MAX_SCIENTIFIC_ATTEMPTS,
            "max_wall_clock_hours": MAX_WALL_CLOCK_HOURS,
            "extension_permitted": False,
        },
        "streams": {
            "collection_stream": COLLECTION_STREAM,
            "collection_master_seed": COLLECTION_MASTER_SEED,
            "per_cell_streams": True,
        },
        "trainable_schema": {
            "action_keys": list(REQUIRED_ACTION_KEYS),
            "agent_keys": list(REQUIRED_AGENT_KEYS),
            "n_proximity_sensors": N_PROXIMITY_SENSORS,
            "proximity_frame_shape": list(PROXIMITY_FRAME_SHAPE),
            "proximity_dtype": "float32",
            "proximity_must_be_finite": True,
            "proximity_must_be_nonconstant": True,
            "wrist_rgb": (
                "published by the datagen pipeline as "
                f"episode_*{WRIST_VIDEO_SUFFIX}; validated by decoding and "
                "matching its frame count to T"
            ),
        },
        "runtime": {
            "task_horizon": TASK_HORIZON,
            "proximity_sensor_period_ms": PROXIMITY_SENSOR_PERIOD_MS,
            "max_sampling_retries": MAX_SAMPLING_RETRIES,
            "default_workers": DEFAULT_WORKERS,
        },
        "upstream_inputs": file_hashes(
            (V107_CERT_JSON, V107_SELECTION_JSON, V107_PHASE0_JSON, V107_POOL_JSON)),
        "implementation_files": file_hashes(IMPLEMENTATION_PATHS),
        "implementation_digest": implementation_digest(),
        **empty_authorization(),
        "authorizes_conversion": False,
        "authorizes_training": False,
        "authorizes_evaluation": False,
    }


__all__ = [
    "BASE_QUOTA_PER_CELL", "BONUS_CELLS", "COLLECTION_MASTER_SEED",
    "COLLECTION_ROOT", "COLLECTION_STREAM", "CONTRACT_VERSION_V108",
    "DATASET_ROOT", "DEFAULT_WORKERS", "ENVIRONMENT_VERSION",
    "IMPLEMENTATION_PATHS", "ImmutableArtifactError",
    "INTRUSION_SIDES", "IS_EXPLORATORY_OWNER_OVERRIDE", "IS_PHASE0_PASS",
    "MAX_SAMPLING_RETRIES", "MAX_SCIENTIFIC_ATTEMPTS", "MAX_WALL_CLOCK_HOURS",
    "N_PROXIMITY_SENSORS", "PLAN_RELATIVE", "POSE_IDS",
    "PROXIMITY_FRAME_SHAPE", "PROXIMITY_SENSOR_PERIOD_MS",
    "REPORT_AFTER_ATTEMPTS", "REQUIRED_ACTION_KEYS", "REQUIRED_AGENT_KEYS",
    "ROOT", "SAMPLER_CLASS", "TARGET_SUCCESSES", "TASK_HORIZON",
    "V107_CERT_JSON", "V107_PHASE0_JSON", "V107_POOL_JSON",
    "V107_SELECTION_JSON", "V95_LAYOUT_FAMILY_IDS", "WRIST_VIDEO_SUFFIX",
    "attempt_id", "build_attempt_row", "build_contract", "cell_key",
    "cell_seed", "cells", "empty_authorization", "file_hashes",
    "implementation_digest", "is_clean", "next_attempts", "quota_totals",
    "quotas", "quotas_met", "remaining_quota", "round_robin_schedule",
    "row_defects", "sha256_file", "sha256_payload", "v95_row_payload",
    "write_immutable_create_only", "write_immutable_text_create_only",
]
