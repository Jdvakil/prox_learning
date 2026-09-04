#!/usr/bin/env python3
"""Frozen V10.11b contract: V10.11 with only three taller primitives."""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pact_place_v1011_contract as v1011  # noqa: E402
from pact_place_v105_contract import (  # noqa: E402
    canonical_payload_sha256,
    empty_authorization,
    sha256_payload,
    write_immutable_create_only,
)
from pact_place_v109_eval_contract import SCENE_BY_POSE, SELECTED_ASSEMBLY  # noqa: E402
from pact_place_v9_contract import panel_corridor_metrics, route_blocker_metrics  # noqa: E402

CONTRACT_VERSION = "pact_place_v1011b_tall_primitive_validation_v1"
ENVIRONMENT_VERSION = "pact_place_corridor_v10_11b_tall_primitives"
SAMPLER_CLASS = "PactPlaceCorridorV1011BTallPrimitiveSampler"
DISPLAY_VERSION = "V10.11b"
PLAN_RELATIVE = "docs/PACT_PLACE_V1011B_TALL_PRIMITIVE_VALIDATION_PLAN.md"

ACTIVE_CLUTTER_SLOTS = v1011.ACTIVE_CLUTTER_SLOTS
INACTIVE_CLUTTER_SLOTS = v1011.INACTIVE_CLUTTER_SLOTS
PRIMITIVE_SLOTS = v1011.PRIMITIVE_SLOTS
MESH_SLOTS = v1011.MESH_SLOTS
NEAR_TARGET_SLOTS = v1011.NEAR_TARGET_SLOTS
ACTIVE_CLUTTER_COUNT = v1011.ACTIVE_CLUTTER_COUNT

# The only geometry change relative to V10.11. XY dimensions and every mesh
# object remain byte-for-byte equal to the parent contract.
PRIMITIVE_HEIGHTS_M = {"01": 0.245, "08": 0.180, "09": 0.180}
PARENT_PRIMITIVE_HEIGHTS_M = {"01": 0.220, "08": 0.100, "09": 0.100}

PREFLIGHT_MASTER_SEED = 2026101111
REVIEW_MASTER_SEED = 2026101112
PREFLIGHT_STREAM = "pact_place_v1011b_preflight"
REVIEW_STREAM = "pact_place_v1011b_review"
HISTORICAL_MASTER_SEEDS = tuple(v1011.HISTORICAL_MASTER_SEEDS) + (
    v1011.PREFLIGHT_MASTER_SEED,
    v1011.REVIEW_MASTER_SEED,
)

PREFLIGHT_REPLICATES = v1011.PREFLIGHT_REPLICATES
MAX_SAMPLING_RETRIES = v1011.MAX_SAMPLING_RETRIES
REVIEW_SUCCESSES = v1011.REVIEW_SUCCESSES
REVIEW_FAILURES = v1011.REVIEW_FAILURES
REVIEW_MAX_ATTEMPTS_PER_CELL = v1011.REVIEW_MAX_ATTEMPTS_PER_CELL
TASK_HORIZON = v1011.TASK_HORIZON
CONTRACT_ROOT = "diagnostics_output/pact_place_v1011b_contract"
VISIBILITY_ROOT = "diagnostics_output/pact_place_v1011b_visibility"
PREFLIGHT_ROOT = "diagnostics_output/pact_place_v1011b_preflight"
REVIEW_ROOT = "diagnostics_output/pact_place_v1011b_review"
VISIBILITY_RUNNER_RELATIVE = "scripts/run_pact_place_v1011b_visibility.py"
PIPELINE_RUNNER_RELATIVE = "scripts/run_pact_place_v1011b_pipeline.py"

cells = v1011.cells
cell_key = v1011.cell_key
identity_payload = v1011.identity_payload


def _seed(
    stream: str,
    master: int,
    family: str,
    side: str,
    pose: str,
    attempt_index: int,
) -> tuple[int, int]:
    digest = hashlib.sha256(
        f"{stream}:{master}:{family}:{side}:{pose}:{int(attempt_index)}".encode()
    ).digest()
    value = int.from_bytes(digest[:8], "big")
    return value % (2**32), value


def _make_tall(payload: dict[str, Any]) -> dict[str, Any]:
    """Patch only primitive Z dimensions while preserving every XY value."""
    parent_payload = copy.deepcopy(payload)
    payload = copy.deepcopy(payload)
    palette = {str(item["slot"]): item for item in payload["pact_clutter_palette"]}
    layout = payload["pact_clutter_layout"]
    objects = {str(item["palette_slot"]): item for item in layout["objects"]}

    for slot, height in PRIMITIVE_HEIGHTS_M.items():
        item = palette[slot]
        old_dimensions = [float(value) for value in item["dimensions_m"]]
        dimensions = list(old_dimensions)
        dimensions[2] = float(height)
        item["dimensions_m"] = dimensions
        item["annotation_dimensions_m"] = list(dimensions)
        item["half_m"] = [value / 2.0 for value in dimensions]
        item["max_dimension_m"] = max(dimensions)
        primitive = item["primitive"]
        if primitive["shape"] == "cylinder":
            primitive["height_m"] = float(height)
        elif primitive["shape"] == "box":
            size = [float(value) for value in primitive["size_m"]]
            size[2] = float(height)
            primitive["size_m"] = size
        else:  # pragma: no cover - parent contract fixes these three shapes
            raise ValueError(f"unsupported V10.11b primitive shape in slot {slot}")

        layout_item = objects[slot]
        old_xy = [float(value) for value in layout_item["half_m"][:2]]
        layout_item["half_m"] = list(item["half_m"])
        layout_item["center_m"][2] = 0.72 + float(item["half_m"][2])
        layout_item["primitive"] = copy.deepcopy(primitive)
        if [float(value) for value in layout_item["half_m"][:2]] != old_xy:
            raise AssertionError(f"V10.11b changed slot {slot} XY footprint")

    # Assert the parent XY footprints and every mesh palette entry are intact.
    parent_palette = {
        str(item["slot"]): item
        for item in parent_payload["pact_clutter_palette"]
    }
    for slot in PRIMITIVE_SLOTS:
        if palette[slot]["dimensions_m"][:2] != parent_palette[slot]["dimensions_m"][:2]:
            raise AssertionError(f"V10.11b changed slot {slot} XY dimensions")
    for slot in MESH_SLOTS:
        if palette[slot] != parent_palette[slot]:
            raise AssertionError(f"V10.11b changed mesh slot {slot}")

    layout["route_blocker_center_xy_m"] = list(objects["01"]["center_m"][:2])
    layout["nominal_route_metrics"] = route_blocker_metrics(layout)
    layout["panel_corridor_metrics"] = panel_corridor_metrics(layout)
    if not layout["nominal_route_metrics"]["detour_admitted"]:
        raise ValueError("taller primitive vessel closes the nominal detour")
    if not layout["panel_corridor_metrics"]["detour_admitted"]:
        raise ValueError("taller primitive vessel closes the panel corridor")
    layout["layout_contract_version"] = CONTRACT_VERSION
    return payload


def _mixed_payload(family: str, side: str) -> dict[str, Any]:
    return _make_tall(v1011._mixed_payload(family, side))


def build_row(
    family: str,
    side: str,
    pose: str,
    attempt_index: int,
    *,
    stream: str = REVIEW_STREAM,
    master_seed: int = REVIEW_MASTER_SEED,
    role_index: int = 0,
) -> dict[str, Any]:
    payload = _mixed_payload(family, side)
    seed_u32, seed_u64 = _seed(
        stream, master_seed, family, side, pose, attempt_index
    )
    identity = sha256_payload(identity_payload(payload["pact_clutter_palette"]))
    episode_id = hashlib.sha256(
        f"{stream}:{master_seed}:{family}:{side}:{pose}:{attempt_index}:episode".encode()
    ).hexdigest()
    row: dict[str, Any] = {
        **payload,
        "role_index": int(role_index),
        "attempt_index": int(attempt_index),
        "episode_id": episode_id,
        "attempt_id": episode_id,
        "cell": cell_key(family, side, pose),
        "family_id": family,
        "family": family,
        "layout_family_id": family,
        "intrusion_side": side,
        "pose_id": pose,
        "pose_offset_m": float(v1011.POSE_OFFSETS_M[pose]),
        "seed_stream": stream,
        "task_seed_u32": int(seed_u32),
        "task_seed_u64": int(seed_u64),
        "max_sampling_retries": MAX_SAMPLING_RETRIES,
        "environment_version": ENVIRONMENT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "task_sampler_class": SAMPLER_CLASS,
        "scene_template_house_index": 1,
        "pact_v106_x_m": float(SELECTED_ASSEMBLY["x_m"]),
        "pact_v106_r_neg_m": float(SELECTED_ASSEMBLY["r_neg_m"]),
        "pact_v106_r_pos_m": float(SELECTED_ASSEMBLY["r_pos_m"]),
        "pact_v106_scene_sha256": SCENE_BY_POSE[pose]["sha256"],
        # Compatibility names are retained because the parent sampler owns the
        # established V10.11 layout machinery.
        "pact_v1011_scene_relative": SCENE_BY_POSE[pose]["relative"],
        "pact_v1011_active_clutter_slots": list(ACTIVE_CLUTTER_SLOTS),
        "pact_v1011_inactive_clutter_slots": list(INACTIVE_CLUTTER_SLOTS),
        "pact_v1011_primitive_slots": list(PRIMITIVE_SLOTS),
        "pact_v1011_mesh_slots": list(MESH_SLOTS),
        "pact_v1011_active_clutter_count": ACTIVE_CLUTTER_COUNT,
        "pact_v1011_identity_sha256": identity,
        "pact_v1011b_parent_environment_version": v1011.ENVIRONMENT_VERSION,
        "pact_v1011b_tall_primitive_heights_m": dict(PRIMITIVE_HEIGHTS_M),
        "pact_v1011b_footprints_unchanged": True,
    }
    row["row_sha256"] = sha256_payload(row)
    return row


def preflight_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    role = 0
    for replicate in range(PREFLIGHT_REPLICATES):
        for family, side, pose in cells():
            row = build_row(
                family,
                side,
                pose,
                replicate,
                stream=PREFLIGHT_STREAM,
                master_seed=PREFLIGHT_MASTER_SEED,
                role_index=role,
            )
            row["replicate"] = replicate
            row.pop("row_sha256")
            row["row_sha256"] = sha256_payload(row)
            rows.append(row)
            role += 1
    return rows


def streams_are_disjoint() -> dict[str, Any]:
    new = (PREFLIGHT_MASTER_SEED, REVIEW_MASTER_SEED)
    overlap = sorted(set(new) & set(HISTORICAL_MASTER_SEEDS))
    return {
        "new_master_seeds": list(new),
        "historical_master_seeds": list(HISTORICAL_MASTER_SEEDS),
        "overlap": overlap,
        "disjoint": not overlap and len(set(new)) == len(new),
    }


def build_contract() -> dict[str, Any]:
    parent_contract = v1011.build_contract()
    row = build_row(*cells()[0], 0)
    document = {
        **empty_authorization(),
        "schema_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "plan": PLAN_RELATIVE,
        "scope": "height_only_successor_through_owner_review",
        "parent": {
            "contract_version": v1011.CONTRACT_VERSION,
            "contract_payload_sha256": parent_contract["payload_sha256"],
            "environment_version": v1011.ENVIRONMENT_VERSION,
        },
        "height_only_amendment": {
            "from_m": dict(PARENT_PRIMITIVE_HEIGHTS_M),
            "to_m": dict(PRIMITIVE_HEIGHTS_M),
            "xy_footprints_unchanged": True,
            "mesh_objects_unchanged": True,
            "cup_and_target_dimensions_unchanged": True,
        },
        "composition": {
            "active_slots": list(ACTIVE_CLUTTER_SLOTS),
            "inactive_slots": list(INACTIVE_CLUTTER_SLOTS),
            "primitive_slots": list(PRIMITIVE_SLOTS),
            "mesh_slots": list(MESH_SLOTS),
            "near_target_slots": list(NEAR_TARGET_SLOTS),
            "identity_sha256": row["pact_v1011_identity_sha256"],
        },
        "streams": streams_are_disjoint(),
        "preflight": {
            "cells": len(cells()),
            "replicates": PREFLIGHT_REPLICATES,
            "rows": len(preflight_rows()),
            "requires_all": True,
        },
        "visibility": {
            "paired_parent_and_tall_scene_at_identical_robot_qpos": True,
            "raw_production_proximity_path": True,
            "does_not_hardcode_visibility": True,
        },
        "review": {
            "clean_successes": REVIEW_SUCCESSES,
            "failures": REVIEW_FAILURES,
            "selection": "first three of each in frozen cell/attempt order",
        },
    }
    document["payload_sha256"] = canonical_payload_sha256(document)
    return document


def write_contract(path: Path) -> None:
    write_immutable_create_only(path, build_contract())


__all__ = [name for name in globals() if name.isupper()] + [
    "build_contract",
    "build_row",
    "cell_key",
    "cells",
    "identity_payload",
    "preflight_rows",
    "streams_are_disjoint",
    "write_contract",
]
