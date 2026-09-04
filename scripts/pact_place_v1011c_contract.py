#!/usr/bin/env python3
"""Frozen V10.11c contract: all V10.11b primitive heights times 1.33."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pact_place_v1011b_contract as parent  # noqa: E402
from pact_place_v105_contract import (  # noqa: E402
    canonical_payload_sha256,
    empty_authorization,
    sha256_payload,
    write_immutable_create_only,
)
from pact_place_v9_contract import panel_corridor_metrics, route_blocker_metrics  # noqa: E402

CONTRACT_VERSION = "pact_place_v1011c_33pct_taller_primitive_validation_v1"
ENVIRONMENT_VERSION = "pact_place_corridor_v10_11c_33pct_taller_primitives"
SAMPLER_CLASS = "PactPlaceCorridorV1011C33PctTallerPrimitiveSampler"
DISPLAY_VERSION = "V10.11c"
PLAN_RELATIVE = "docs/PACT_PLACE_V1011C_33PCT_TALLER_VALIDATION_PLAN.md"

HEIGHT_MULTIPLIER_FROM_PARENT = 1.33
PARENT_PRIMITIVE_HEIGHTS_M = dict(parent.PRIMITIVE_HEIGHTS_M)
PRIMITIVE_HEIGHTS_M = {
    slot: float(height * HEIGHT_MULTIPLIER_FROM_PARENT)
    for slot, height in PARENT_PRIMITIVE_HEIGHTS_M.items()
}

ACTIVE_CLUTTER_SLOTS = parent.ACTIVE_CLUTTER_SLOTS
INACTIVE_CLUTTER_SLOTS = parent.INACTIVE_CLUTTER_SLOTS
PRIMITIVE_SLOTS = parent.PRIMITIVE_SLOTS
MESH_SLOTS = parent.MESH_SLOTS
NEAR_TARGET_SLOTS = parent.NEAR_TARGET_SLOTS
ACTIVE_CLUTTER_COUNT = parent.ACTIVE_CLUTTER_COUNT

PREFLIGHT_MASTER_SEED = 2026101121
REVIEW_MASTER_SEED = 2026101122
PREFLIGHT_STREAM = "pact_place_v1011c_preflight"
REVIEW_STREAM = "pact_place_v1011c_review"
HISTORICAL_MASTER_SEEDS = tuple(parent.HISTORICAL_MASTER_SEEDS) + (
    parent.PREFLIGHT_MASTER_SEED,
    parent.REVIEW_MASTER_SEED,
)
PREFLIGHT_REPLICATES = parent.PREFLIGHT_REPLICATES
MAX_SAMPLING_RETRIES = parent.MAX_SAMPLING_RETRIES
REVIEW_SUCCESSES = parent.REVIEW_SUCCESSES
REVIEW_FAILURES = parent.REVIEW_FAILURES
REVIEW_MAX_ATTEMPTS_PER_CELL = parent.REVIEW_MAX_ATTEMPTS_PER_CELL
TASK_HORIZON = parent.TASK_HORIZON

CONTRACT_ROOT = "diagnostics_output/pact_place_v1011c_contract"
VISIBILITY_ROOT = "diagnostics_output/pact_place_v1011c_visibility"
PREFLIGHT_ROOT = "diagnostics_output/pact_place_v1011c_preflight"
REVIEW_ROOT = "diagnostics_output/pact_place_v1011c_review"
VISIBILITY_RUNNER_RELATIVE = "scripts/run_pact_place_v1011c_visibility.py"
PIPELINE_RUNNER_RELATIVE = "scripts/run_pact_place_v1011c_pipeline.py"

cells = parent.cells
cell_key = parent.cell_key
identity_payload = parent.identity_payload


def _make_taller(payload: dict[str, Any]) -> dict[str, Any]:
    source = copy.deepcopy(payload)
    payload = copy.deepcopy(payload)
    palette = {str(item["slot"]): item for item in payload["pact_clutter_palette"]}
    source_palette = {
        str(item["slot"]): item for item in source["pact_clutter_palette"]
    }
    layout = payload["pact_clutter_layout"]
    objects = {str(item["palette_slot"]): item for item in layout["objects"]}
    source_objects = {
        str(item["palette_slot"]): item
        for item in source["pact_clutter_layout"]["objects"]
    }

    for slot, height in PRIMITIVE_HEIGHTS_M.items():
        item = palette[slot]
        dimensions = [float(value) for value in item["dimensions_m"]]
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
        else:  # pragma: no cover
            raise ValueError(f"unsupported V10.11c primitive shape in slot {slot}")

        obj = objects[slot]
        obj["half_m"] = list(item["half_m"])
        obj["center_m"][2] = 0.72 + float(item["half_m"][2])
        obj["primitive"] = copy.deepcopy(primitive)
        if item["dimensions_m"][:2] != source_palette[slot]["dimensions_m"][:2]:
            raise AssertionError(f"V10.11c changed slot {slot} XY dimensions")
        if obj["center_m"][:2] != source_objects[slot]["center_m"][:2]:
            raise AssertionError(f"V10.11c changed slot {slot} XY center")
        if obj["half_m"][:2] != source_objects[slot]["half_m"][:2]:
            raise AssertionError(f"V10.11c changed slot {slot} XY footprint")

    for slot in MESH_SLOTS:
        if palette[slot] != source_palette[slot] or objects[slot] != source_objects[slot]:
            raise AssertionError(f"V10.11c changed mesh slot {slot}")

    layout["route_blocker_center_xy_m"] = list(objects["01"]["center_m"][:2])
    layout["nominal_route_metrics"] = route_blocker_metrics(layout)
    layout["panel_corridor_metrics"] = panel_corridor_metrics(layout)
    if not layout["nominal_route_metrics"]["detour_admitted"]:
        raise ValueError("V10.11c vessel closes the nominal detour")
    if not layout["panel_corridor_metrics"]["detour_admitted"]:
        raise ValueError("V10.11c vessel closes the panel corridor")
    layout["layout_contract_version"] = CONTRACT_VERSION
    return payload


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
    row = parent.build_row(
        family,
        side,
        pose,
        attempt_index,
        stream=stream,
        master_seed=master_seed,
        role_index=role_index,
    )
    row = _make_taller(row)
    identity = sha256_payload(identity_payload(row["pact_clutter_palette"]))
    row.update(
        {
            "environment_version": ENVIRONMENT_VERSION,
            "contract_version": CONTRACT_VERSION,
            "sampler_class": SAMPLER_CLASS,
            "task_sampler_class": SAMPLER_CLASS,
            "pact_v1011_identity_sha256": identity,
            "pact_v1011c_parent_environment_version": parent.ENVIRONMENT_VERSION,
            "pact_v1011c_primitive_heights_m": dict(PRIMITIVE_HEIGHTS_M),
            "pact_v1011c_height_multiplier_from_v1011b": (
                HEIGHT_MULTIPLIER_FROM_PARENT
            ),
            "pact_v1011c_footprints_unchanged": True,
        }
    )
    row.pop("pact_v1011b_tall_primitive_heights_m", None)
    row.pop("pact_v1011b_footprints_unchanged", None)
    row.pop("row_sha256", None)
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
    parent_document = parent.build_contract()
    row = build_row(*cells()[0], 0)
    document = {
        **empty_authorization(),
        "schema_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "plan": PLAN_RELATIVE,
        "scope": "height_only_successor_through_owner_review",
        "parent": {
            "contract_version": parent.CONTRACT_VERSION,
            "contract_payload_sha256": parent_document["payload_sha256"],
            "environment_version": parent.ENVIRONMENT_VERSION,
        },
        "height_only_amendment": {
            "multiplier": HEIGHT_MULTIPLIER_FROM_PARENT,
            "from_m": dict(PARENT_PRIMITIVE_HEIGHTS_M),
            "to_m": dict(PRIMITIVE_HEIGHTS_M),
            "xy_footprints_unchanged": True,
            "mesh_objects_unchanged": True,
            "cup_and_target_dimensions_unchanged": True,
        },
        "composition": {
            "active_slots": list(ACTIVE_CLUTTER_SLOTS),
            "primitive_slots": list(PRIMITIVE_SLOTS),
            "mesh_slots": list(MESH_SLOTS),
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
            "paired_v1011b_and_v1011c_at_identical_robot_qpos": True,
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
