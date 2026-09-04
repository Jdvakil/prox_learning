#!/usr/bin/env python3
"""Frozen V10.11d contract: V10.11c clutter with every clutter position randomized.

V10.11c keeps the frozen V9.5 layout, in which the two plates sit at exactly
(0.980, -0.220) and (1.090, +0.300) in all eight family/side combinations and
never move, while the two vessels get only the inherited millimetre-scale
jitter. V10.11d changes nothing about *what* the clutter is -- same palette,
same primitive shapes, same heights, same identity hash -- and randomizes
*where* each item stands.

The proposal boxes below bound the draw; they do not define admissibility. Every
candidate is rejected unless it stays inside the bench shell, clears the target
and every already-placed body, and -- for the route-bearing slot 01 -- still
satisfies both registered route predicates. Slot 01's admissible y window was
measured at roughly 60 mm wide with a side-dependent sign, so the predicates are
re-evaluated per candidate instead of being folded into a hardcoded box.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pact_place_v1011c_contract as parent  # noqa: E402
from pact_place_v105_contract import (  # noqa: E402
    canonical_payload_sha256,
    empty_authorization,
    sha256_payload,
    write_immutable_create_only,
)

CONTRACT_VERSION = "pact_place_v1011d_randomized_clutter_validation_v1"
ENVIRONMENT_VERSION = "pact_place_corridor_v10_11d_randomized_clutter"
SAMPLER_CLASS = "PactPlaceCorridorV1011DRandomizedLayoutSampler"
DISPLAY_VERSION = "V10.11d"
PLAN_RELATIVE = "docs/PACT_PLACE_V1011D_RANDOMIZED_CLUTTER_VALIDATION_PLAN.md"

# Identical to V10.11c in every respect that describes the clutter itself.
PRIMITIVE_HEIGHTS_M = dict(parent.PRIMITIVE_HEIGHTS_M)
ACTIVE_CLUTTER_SLOTS = parent.ACTIVE_CLUTTER_SLOTS
INACTIVE_CLUTTER_SLOTS = parent.INACTIVE_CLUTTER_SLOTS
PRIMITIVE_SLOTS = parent.PRIMITIVE_SLOTS
MESH_SLOTS = parent.MESH_SLOTS
NEAR_TARGET_SLOTS = parent.NEAR_TARGET_SLOTS
ACTIVE_CLUTTER_COUNT = parent.ACTIVE_CLUTTER_COUNT

# The four slots V10.11c left effectively static. Slots 08/09 already draw from
# the inherited target-relative annulus and are unchanged here.
RANDOMIZED_SLOTS = ("01", "06", "03", "04")
SLOT_RANDOMIZATION_BOXES_M = {
    # Slot 01's floor is 0.650 rather than the wider span the route predicates
    # alone allow: the two vessels need 98 mm of x separation and slot 06 cannot
    # go below 0.545 without leaving the bench shell, so a lower floor here
    # starves slot 06. Measured over 800 synthetic layouts, these boxes place
    # all four slots in 100% of draws; a 0.600 floor dropped that to 68.5%.
    "01": {"x": (0.650, 0.740), "y": (-0.055, 0.055)},
    "06": {"x": (0.545, 0.600), "y": (-0.050, 0.050)},
    "03": {"x": (0.920, 1.240), "y": (-0.320, 0.320)},
    "04": {"x": (0.920, 1.240), "y": (-0.320, 0.320)},
}
BASE_MAX_CANDIDATES = 96
# What V10.11c actually did, recorded so the diff is legible in the artifact.
V1011C_STATIC_CENTERS_XY_M = {
    "03": (0.980, -0.220),
    "04": (1.090, 0.300),
}
V1011C_VESSEL_JITTER_M = {
    "01": {"x": 0.020, "y": 0.005},
    "06": {"x": 0.005, "y": 0.010},
}

PREFLIGHT_MASTER_SEED = 2026101141
REVIEW_MASTER_SEED = 2026101142
PREFLIGHT_STREAM = "pact_place_v1011d_preflight"
REVIEW_STREAM = "pact_place_v1011d_review"
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

CONTRACT_ROOT = "diagnostics_output/pact_place_v1011d_contract"
PREFLIGHT_ROOT = "diagnostics_output/pact_place_v1011d_preflight"
REVIEW_ROOT = "diagnostics_output/pact_place_v1011d_review"

cells = parent.cells
cell_key = parent.cell_key
identity_payload = parent.identity_payload


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
    """A V10.11c row re-badged for V10.11d.

    The row's ``pact_clutter_layout`` still carries the nominal V9.5 centres.
    The sampler overwrites them per episode, exactly as V10.11 already does for
    the slot 08/09 placeholders, so the row stays a valid metadata box if it is
    inspected before the draw.
    """
    row = parent.build_row(
        family, side, pose, attempt_index,
        stream=stream, master_seed=master_seed, role_index=role_index,
    )
    identity = sha256_payload(identity_payload(row["pact_clutter_palette"]))
    if identity != row["pact_v1011_identity_sha256"]:  # pragma: no cover
        raise AssertionError("V10.11d altered the V10.11c clutter identity")
    row.update(
        {
            "environment_version": ENVIRONMENT_VERSION,
            "contract_version": CONTRACT_VERSION,
            "sampler_class": SAMPLER_CLASS,
            "task_sampler_class": SAMPLER_CLASS,
            "pact_v1011d_parent_environment_version": parent.ENVIRONMENT_VERSION,
            "pact_v1011d_randomized_slots": list(RANDOMIZED_SLOTS),
            "pact_v1011d_slot_randomization_boxes_m": {
                slot: {"x": list(box["x"]), "y": list(box["y"])}
                for slot, box in SLOT_RANDOMIZATION_BOXES_M.items()
            },
            "pact_v1011d_base_max_candidates": BASE_MAX_CANDIDATES,
            "pact_v1011d_clutter_identity_matches_v1011c": True,
        }
    )
    row.pop("row_sha256", None)
    row["row_sha256"] = sha256_payload(row)
    return row


def preflight_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    role = 0
    for replicate in range(PREFLIGHT_REPLICATES):
        for family, side, pose in cells():
            row = build_row(
                family, side, pose, replicate,
                stream=PREFLIGHT_STREAM, master_seed=PREFLIGHT_MASTER_SEED,
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
        "scope": "position_only_successor_through_owner_review",
        "parent": {
            "contract_version": parent.CONTRACT_VERSION,
            "contract_payload_sha256": parent_document["payload_sha256"],
            "environment_version": parent.ENVIRONMENT_VERSION,
        },
        "position_only_amendment": {
            "randomized_slots": list(RANDOMIZED_SLOTS),
            "proposal_boxes_m": {
                slot: {"x": list(box["x"]), "y": list(box["y"])}
                for slot, box in SLOT_RANDOMIZATION_BOXES_M.items()
            },
            "max_candidates_per_slot": BASE_MAX_CANDIDATES,
            "near_target_slots_unchanged": list(NEAR_TARGET_SLOTS),
            "clutter_identity_unchanged": True,
            "primitive_heights_unchanged_m": dict(PRIMITIVE_HEIGHTS_M),
            "mesh_objects_unchanged": True,
            "cup_and_target_dimensions_unchanged": True,
            "v1011c_static_centers_xy_m": {
                slot: list(value)
                for slot, value in V1011C_STATIC_CENTERS_XY_M.items()
            },
            "v1011c_vessel_jitter_m": V1011C_VESSEL_JITTER_M,
            "target_clearance": (
                "delegated to the runtime settle and initial-contact check. A "
                "measured V10.11c row has the cup AABB overlapping slot 01's in "
                "all three axes with zero forbidden contact, so any "
                "conservative planar separation rule would reject V10.11c's own "
                "working layout."
            ),
            "measured_placement_success_rate": 1.0,
            "admissibility": (
                "per-candidate: bench shell containment, target clearance, "
                "mutual separation, and for slot 01 both route_blocker_metrics "
                "and panel_corridor_metrics must admit the detour"
            ),
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
