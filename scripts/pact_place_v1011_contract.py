#!/usr/bin/env python3
"""Frozen V10.11 mixed mesh/primitive clutter validation contract."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v105_contract import (  # noqa: E402
    canonical_payload_sha256,
    empty_authorization,
    sha256_payload,
    v95_row_payload,
    write_immutable_create_only,
)
from pact_place_v106_contract import INTRUSION_SIDES, V95_LAYOUT_FAMILY_IDS  # noqa: E402
from pact_place_v106_geometry import POSE_IDS, POSE_OFFSETS_M  # noqa: E402
from pact_place_v109_eval_contract import SCENE_BY_POSE, SELECTED_ASSEMBLY  # noqa: E402
from pact_place_v9_contract import panel_corridor_metrics, route_blocker_metrics  # noqa: E402

CONTRACT_VERSION = "pact_place_v1011_mixed_clutter_validation_v1"
ENVIRONMENT_VERSION = "pact_place_corridor_v10_11_mixed_clutter"
SAMPLER_CLASS = "PactPlaceCorridorV1011MixedClutterSampler"
PLAN_RELATIVE = "docs/PACT_PLACE_V1011_MIXED_CLUTTER_VALIDATION_PLAN.md"

ACTIVE_CLUTTER_SLOTS = ("01", "03", "04", "06", "08", "09")
INACTIVE_CLUTTER_SLOTS = ("00", "02", "05", "07")
PRIMITIVE_SLOTS = ("01", "08", "09")
MESH_SLOTS = ("03", "04", "06")
NEAR_TARGET_SLOTS = ("08", "09")
ACTIVE_CLUTTER_COUNT = 6

PRIMITIVES: dict[str, dict[str, Any]] = {
    "01": {
        "uid": "pact_primitive_cylinder_01",
        "role": "outbound_vessel",
        "category": "vase",
        "dimensions_m": [0.090, 0.090, 0.220],
        "primitive": {
            "shape": "cylinder",
            "radius_m": 0.045,
            "height_m": 0.220,
            "density_kg_m3": 1000.0,
            "rgba": [0.78, 0.56, 0.28, 1.0],
        },
    },
    "08": {
        "uid": "pact_primitive_cylinder_08",
        "role": "decor",
        "category": "primitive_cylinder",
        "dimensions_m": [0.070, 0.070, 0.100],
        "primitive": {
            "shape": "cylinder",
            "radius_m": 0.035,
            "height_m": 0.100,
            "density_kg_m3": 1000.0,
            "rgba": [0.26, 0.57, 0.82, 1.0],
        },
    },
    "09": {
        "uid": "pact_primitive_box_09",
        "role": "decor",
        "category": "primitive_box",
        "dimensions_m": [0.070, 0.070, 0.100],
        "primitive": {
            "shape": "box",
            "size_m": [0.070, 0.070, 0.100],
            "density_kg_m3": 1000.0,
            "rgba": [0.62, 0.36, 0.72, 1.0],
        },
    },
}

PREFLIGHT_MASTER_SEED = 2026101101
REVIEW_MASTER_SEED = 2026101102
PREFLIGHT_STREAM = "pact_place_v1011_preflight"
REVIEW_STREAM = "pact_place_v1011_review"
HISTORICAL_MASTER_SEEDS = (
    2026082101,
    2026082901,
    2026082902,
    2026083001,
    2026101001,
    2026101002,
    2026101003,
)

PREFLIGHT_REPLICATES = 4
MAX_SAMPLING_RETRIES = 12
REVIEW_SUCCESSES = 3
REVIEW_FAILURES = 3
REVIEW_MAX_ATTEMPTS_PER_CELL = 20
TASK_HORIZON = 1050
PREFLIGHT_ROOT = "diagnostics_output/pact_place_v1011_preflight"
REVIEW_ROOT = "diagnostics_output/pact_place_v1011_review"


def cells() -> list[tuple[str, str, str]]:
    return [
        (family, side, pose)
        for family in V95_LAYOUT_FAMILY_IDS
        for side in INTRUSION_SIDES
        for pose in POSE_IDS
    ]


def cell_key(family: str, side: str, pose: str) -> str:
    return f"{family}|{side}|{pose}"


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


def _primitive_palette_item(slot: str) -> dict[str, Any]:
    source = copy.deepcopy(PRIMITIVES[slot])
    dimensions = [float(value) for value in source["dimensions_m"]]
    return {
        "slot": slot,
        "slot_class": "prop",
        "role": source["role"],
        "uid": source["uid"],
        "category": source["category"],
        "dimensions_m": dimensions,
        "annotation_dimensions_m": list(dimensions),
        "half_m": [value / 2.0 for value in dimensions],
        "max_dimension_m": max(dimensions),
        "support": "shelf_standing",
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "body_prefix": f"pact_clutter_{slot}/",
        "primitive": source["primitive"],
    }


def _mixed_payload(family: str, side: str) -> dict[str, Any]:
    payload = copy.deepcopy(v95_row_payload(family, side))
    palette = {
        str(item["slot"]): copy.deepcopy(item)
        for item in payload["pact_clutter_palette"]
    }
    palette["01"] = _primitive_palette_item("01")
    palette["08"] = _primitive_palette_item("08")
    palette["09"] = _primitive_palette_item("09")
    payload["pact_clutter_palette"] = [palette[key] for key in sorted(palette)]

    layout = payload["pact_clutter_layout"]
    by_slot = {
        str(item["palette_slot"]): copy.deepcopy(item)
        for item in layout["objects"]
    }
    old_center = list(by_slot["01"]["center_m"])
    for slot in ("01", "08", "09"):
        source = palette[slot]
        if slot == "01":
            center = [old_center[0], old_center[1], 0.72 + source["half_m"][2]]
        else:
            # Placeholders are replaced by the sampler after its single target
            # draw. They remain valid metadata boxes if inspected pre-draw.
            center = [0.90 if slot == "08" else 1.08,
                      -0.12 if slot == "08" else 0.12,
                      0.72 + source["half_m"][2]]
        by_slot[slot] = {
            "palette_slot": slot,
            "uid": source["uid"],
            "role": source["role"],
            "category": source["category"],
            "support": "bench_standing",
            "center_m": center,
            "half_m": list(source["half_m"]),
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "size_class": "large" if slot == "01" else "medium",
            "primitive": copy.deepcopy(source["primitive"]),
            "target_relative_placeholder": slot in NEAR_TARGET_SLOTS,
        }
    layout["objects"] = [by_slot[key] for key in sorted(by_slot)]
    layout["route_blocker_center_xy_m"] = list(by_slot["01"]["center_m"][:2])
    layout["nominal_route_metrics"] = route_blocker_metrics(layout)
    layout["panel_corridor_metrics"] = panel_corridor_metrics(layout)
    if not layout["nominal_route_metrics"]["detour_admitted"]:
        raise ValueError("primitive outbound vessel does not admit the route")
    if not layout["panel_corridor_metrics"]["detour_admitted"]:
        raise ValueError("primitive outbound vessel closes the panel corridor")
    layout["layout_contract_version"] = CONTRACT_VERSION
    payload["pact_clutter_layout"] = layout
    return payload


def identity_payload(palette: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "slot": str(item["slot"]),
            "uid": str(item["uid"]),
            "role": str(item.get("role", "")),
            "primitive": item.get("primitive"),
        }
        for item in sorted(palette, key=lambda value: str(value["slot"]))
    ]


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
        "pose_offset_m": float(POSE_OFFSETS_M[pose]),
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
        "pact_v1011_scene_relative": SCENE_BY_POSE[pose]["relative"],
        "pact_v1011_active_clutter_slots": list(ACTIVE_CLUTTER_SLOTS),
        "pact_v1011_inactive_clutter_slots": list(INACTIVE_CLUTTER_SLOTS),
        "pact_v1011_primitive_slots": list(PRIMITIVE_SLOTS),
        "pact_v1011_mesh_slots": list(MESH_SLOTS),
        "pact_v1011_active_clutter_count": ACTIVE_CLUTTER_COUNT,
        "pact_v1011_identity_sha256": identity,
    }
    row["row_sha256"] = sha256_payload(row)
    return row


def preflight_rows() -> list[dict[str, Any]]:
    rows = []
    role = 0
    for replicate in range(PREFLIGHT_REPLICATES):
        for family, side, pose in cells():
            attempt = replicate
            row = build_row(
                family,
                side,
                pose,
                attempt,
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
    row = build_row(*cells()[0], 0)
    document = {
        **empty_authorization(),
        "schema_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "plan": PLAN_RELATIVE,
        "scope": "environment_validation_only_through_owner_review",
        "composition": {
            "active_slots": list(ACTIVE_CLUTTER_SLOTS),
            "inactive_slots": list(INACTIVE_CLUTTER_SLOTS),
            "primitive_slots": list(PRIMITIVE_SLOTS),
            "mesh_slots": list(MESH_SLOTS),
            "near_target_slots": list(NEAR_TARGET_SLOTS),
            "primitive_specs": PRIMITIVES,
            "identity_sha256": row["pact_v1011_identity_sha256"],
        },
        "streams": streams_are_disjoint(),
        "preflight": {
            "cells": len(cells()),
            "replicates": PREFLIGHT_REPLICATES,
            "rows": len(preflight_rows()),
            "requires_all": True,
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

