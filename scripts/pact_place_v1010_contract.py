#!/usr/bin/env python3
"""V10.10 four-object collection / training / evaluation: frozen contract.

Derived from V10.7; V10.7 itself is untouched and remains a failed, permanently
closed qualification. Human review and Phase 0 are skipped at the owner's
explicit request, so nothing here is a gate pass and every authorization field
stays false.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v105_contract import (  # noqa: E402
    ImmutableArtifactError, canonical_payload_sha256, empty_authorization,
    file_hashes, implementation_digest, recompute_payload_sha256, sha256_file,
    sha256_payload, v95_row_payload, wilson_interval,
    write_immutable_create_only, write_immutable_text_create_only,
)
from pact_place_v106_contract import INTRUSION_SIDES, V95_LAYOUT_FAMILY_IDS  # noqa: E402
from pact_place_v106_geometry import POSE_IDS, POSE_OFFSETS_M  # noqa: E402
from pact_place_v109_eval_contract import SCENE_BY_POSE, SELECTED_ASSEMBLY  # noqa: E402

CONTRACT_VERSION_V1010 = "pact_place_v1010_four_object_v1"
PLAN_RELATIVE = "docs/PACT_PLACE_V1010_FOUR_OBJECT_TRAIN_EVAL_PLAN.md"
ENVIRONMENT_VERSION = "pact_place_corridor_v10_10_four_object"
SAMPLER_CLASS = "PactPlaceCorridorV1010FourObjectSampler"

IS_PHASE0_PASS = False
HUMAN_REVIEW_SKIPPED_BY_OWNER = True
PHASE0_SKIPPED_BY_OWNER = True
V107_PHASE0_RESULT = "failed_8_of_24_permanently_closed"

# --- the four active objects -------------------------------------------------
ACTIVE_CLUTTER_SLOTS = ("01", "03", "04", "06")
INACTIVE_CLUTTER_SLOTS = ("00", "02", "05", "07")
ACTIVE_CLUTTER_COUNT = 4
ACTIVE_CLUTTER_UIDS = {
    "01": "Soap_Bottle_30", "03": "Plate_10",
    "04": "Plate_22", "06": "Soap_Bottle_11",
}
OBJECT_LABELS = {
    "01": "bottle_soap_30", "06": "bottle_soap_11",
    "03": "plate_10", "04": "plate_22",
}

# --- frozen streams ----------------------------------------------------------
COLLECTION_MASTER_SEED = 2026101001
SPLIT_MASTER_SEED = 2026101002
EVAL_MASTER_SEED = 2026101003
COLLECTION_STREAM = "pact_place_v1010_collection"
EVAL_STREAM = "pact_place_v1010_eval"
# Every master seed used anywhere in V10.7-V10.9, for the disjointness assertion.
HISTORICAL_MASTER_SEEDS = (2026082101, 2026082901, 2026082902, 2026083001)

# --- quotas ------------------------------------------------------------------
QUOTA_PER_CELL = 6
TARGET_SUCCESSES = 144
MAX_SCIENTIFIC_ATTEMPTS = 900
MAX_WALL_CLOCK_HOURS = 16.0
MAX_IN_FLIGHT_PER_CELL = 1
MAX_SAMPLING_RETRIES = 12

# --- split / training --------------------------------------------------------
TRAIN_PER_CELL = 5
VALIDATION_PER_CELL = 1
TRAIN_COUNT = 120
VALIDATION_COUNT = 24
BASE_EPISODE_HORIZON = 635
HORIZON_MARGIN = 8
ENCODER_SHA256 = "6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206"

# --- evaluation --------------------------------------------------------------
EVAL_INSTANCES = 40
EVAL_DOUBLED_CELLS = 16
EVAL_SMOKE_INSTANCES = 4
EVAL_TASK_HORIZON = 900
EVAL_NUM_QUERIES = 100
USE_V109R_EVENT_DECODER = False   # explicitly the original chunk-100 decoder

# --- outputs -----------------------------------------------------------------
COLLECTION_ROOT = "diagnostics_output/pact_place_v1010_collection"
DATASET_ROOT = "assets/datagen/pact_place_corridor_v10_10"
CONVERTED_DATASET_ROOT = "assets/act_style_data/pact_place_v1010_144"
WORK_ROOT = "diagnostics_output/pact_place_v1010_train_eval"
EVAL_ROOT = "diagnostics_output/pact_place_v1010_eval"
PREFLIGHT_ROOT = "diagnostics_output/pact_place_v1010_preflight"
TRAINING_ROOT = "/root/pact_place_v1010_144_pact_vs_act_chunk100_seed3101"


def cells() -> list[tuple[str, str, str]]:
    return [(f, s, p) for f in V95_LAYOUT_FAMILY_IDS
            for s in INTRUSION_SIDES for p in POSE_IDS]


def cell_key(family: str, side: str, pose: str) -> str:
    return f"{family}|{side}|{pose}"


def quotas() -> dict[str, int]:
    out = {cell_key(*c): QUOTA_PER_CELL for c in cells()}
    if sum(out.values()) != TARGET_SUCCESSES:
        raise ValueError(f"quotas total {sum(out.values())}")
    return out


def cell_seed(family: str, side: str, pose: str, attempt_index: int,
              *, stream: str = COLLECTION_STREAM,
              master: int = COLLECTION_MASTER_SEED) -> dict[str, int]:
    digest = hashlib.sha256(
        f"{stream}:{master}:{family}:{side}:{pose}:{int(attempt_index)}".encode()).digest()
    value = int.from_bytes(digest[:8], "big")
    return {"seed_u32": value % (2**32), "seed_u64": value}


def attempt_id(family: str, side: str, pose: str, attempt_index: int) -> str:
    return hashlib.sha256(
        f"{COLLECTION_STREAM}:{COLLECTION_MASTER_SEED}:"
        f"{family}:{side}:{pose}:{int(attempt_index)}:attempt".encode()).hexdigest()


def streams_are_disjoint() -> dict[str, Any]:
    new = (COLLECTION_MASTER_SEED, SPLIT_MASTER_SEED, EVAL_MASTER_SEED)
    overlap = sorted(set(new) & set(HISTORICAL_MASTER_SEEDS))
    return {
        "new_master_seeds": list(new),
        "historical_master_seeds": list(HISTORICAL_MASTER_SEEDS),
        "overlap": overlap,
        "disjoint": not overlap and len(set(new)) == len(new),
    }


def build_row(family: str, side: str, pose: str, attempt_index: int) -> dict[str, Any]:
    """A V9.5 row plus the V10.7 pendant binding and the V10.10 four-object binding."""
    seed = cell_seed(family, side, pose, attempt_index)
    payload = v95_row_payload(family, side)
    row: dict[str, Any] = {
        "role_index": 0,
        "attempt_id": attempt_id(family, side, pose, attempt_index),
        "episode_id": attempt_id(family, side, pose, attempt_index),
        "cell": cell_key(family, side, pose),
        "family_id": family, "family": family, "layout_family_id": family,
        "intrusion_side": side, "pose_id": pose,
        "pose_offset_m": POSE_OFFSETS_M[pose],
        "attempt_index": int(attempt_index),
        "seed_stream": COLLECTION_STREAM,
        "task_seed_u32": int(seed["seed_u32"]),
        "task_seed_u64": int(seed["seed_u64"]),
        "max_sampling_retries": MAX_SAMPLING_RETRIES,
        "environment_version": ENVIRONMENT_VERSION,
        "contract_version": CONTRACT_VERSION_V1010,
        "sampler_class": SAMPLER_CLASS,
        "task_sampler_class": SAMPLER_CLASS,
        "scene_template_house_index": 1,
        "pact_v106_x_m": float(SELECTED_ASSEMBLY["x_m"]),
        "pact_v106_r_neg_m": float(SELECTED_ASSEMBLY["r_neg_m"]),
        "pact_v106_r_pos_m": float(SELECTED_ASSEMBLY["r_pos_m"]),
        "pact_v106_scene_sha256": SCENE_BY_POSE[pose]["sha256"],
        "pact_v1010_scene_relative": SCENE_BY_POSE[pose]["relative"],
        "pact_v1010_active_clutter_slots": list(ACTIVE_CLUTTER_SLOTS),
        "pact_v1010_inactive_clutter_slots": list(INACTIVE_CLUTTER_SLOTS),
        "pact_v1010_active_clutter_count": ACTIVE_CLUTTER_COUNT,
        "pact_v1010_active_clutter_uids": dict(ACTIVE_CLUTTER_UIDS),
        **{k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
           for k, v in payload.items()},
    }
    active = [o for o in row["pact_clutter_layout"]["objects"]
              if str(o["palette_slot"]) in ACTIVE_CLUTTER_SLOTS]
    if len(active) != ACTIVE_CLUTTER_COUNT:
        raise ValueError(f"row exposes {len(active)} active slots")
    # The row binds *which* four objects are active. It cannot bind their
    # positions: V9.3 applies per-episode clutter jitter, so the positional
    # hash is only knowable after sampling and is recorded in telemetry.
    row["pact_v1010_identity_sha256"] = hashlib.sha256(json.dumps(
        [{"palette_slot": str(o["palette_slot"]), "uid": str(o["uid"]),
          "role": str(o.get("role", ""))}
         for o in sorted(active, key=lambda x: str(x["palette_slot"]))],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    row["row_sha256"] = sha256_payload(row)
    return row


def build_contract() -> dict[str, Any]:
    document = {
        **empty_authorization(),
        "schema_version": CONTRACT_VERSION_V1010,
        "plan": PLAN_RELATIVE,
        "is_phase0_pass": IS_PHASE0_PASS,
        "human_review_skipped_by_owner": HUMAN_REVIEW_SKIPPED_BY_OWNER,
        "phase0_skipped_by_owner": PHASE0_SKIPPED_BY_OWNER,
        "v107_phase0_result": V107_PHASE0_RESULT,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "four_objects": {
            "active_slots": list(ACTIVE_CLUTTER_SLOTS),
            "inactive_slots": list(INACTIVE_CLUTTER_SLOTS),
            "active_count": ACTIVE_CLUTTER_COUNT,
            "active_uids": dict(ACTIVE_CLUTTER_UIDS),
            "labels": dict(OBJECT_LABELS),
            "excluded_from_the_count":
                "target cup, static pendant, intrusion panel, tray, enclosure",
            "palette_stays_compiled": True,
        },
        "streams": streams_are_disjoint() | {
            "collection": COLLECTION_MASTER_SEED,
            "split": SPLIT_MASTER_SEED, "evaluation": EVAL_MASTER_SEED,
        },
        "collection": {
            "quota_per_cell": QUOTA_PER_CELL, "cells": len(cells()),
            "target_successes": TARGET_SUCCESSES,
            "max_scientific_attempts": MAX_SCIENTIFIC_ATTEMPTS,
            "max_wall_clock_hours": MAX_WALL_CLOCK_HOURS,
            "max_in_flight_per_cell": MAX_IN_FLIGHT_PER_CELL,
        },
        "split": {"train_per_cell": TRAIN_PER_CELL,
                  "validation_per_cell": VALIDATION_PER_CELL,
                  "train": TRAIN_COUNT, "validation": VALIDATION_COUNT},
        "training": {"root": TRAINING_ROOT, "seed": 3101,
                     "base_episode_horizon": BASE_EPISODE_HORIZON,
                     "horizon_margin": HORIZON_MARGIN,
                     "encoder_sha256": ENCODER_SHA256},
        "evaluation": {"instances": EVAL_INSTANCES,
                       "doubled_cells": EVAL_DOUBLED_CELLS,
                       "smoke_instances": EVAL_SMOKE_INSTANCES,
                       "task_horizon": EVAL_TASK_HORIZON,
                       "num_queries": EVAL_NUM_QUERIES,
                       "use_v109r_event_decoder": USE_V109R_EVENT_DECODER,
                       "decoder": "original chunk-100 temporal ensemble"},
        "scene_by_pose": SCENE_BY_POSE,
        "selected_assembly": SELECTED_ASSEMBLY,
    }
    document["config_sha256"] = canonical_payload_sha256(document)
    return document


__all__ = [
    "ACTIVE_CLUTTER_COUNT", "ACTIVE_CLUTTER_SLOTS", "ACTIVE_CLUTTER_UIDS",
    "COLLECTION_MASTER_SEED", "COLLECTION_ROOT", "CONTRACT_VERSION_V1010",
    "CONVERTED_DATASET_ROOT", "DATASET_ROOT", "ENCODER_SHA256",
    "ENVIRONMENT_VERSION", "EVAL_MASTER_SEED", "EVAL_ROOT",
    "INACTIVE_CLUTTER_SLOTS", "MAX_IN_FLIGHT_PER_CELL",
    "MAX_SCIENTIFIC_ATTEMPTS", "MAX_WALL_CLOCK_HOURS", "OBJECT_LABELS",
    "PREFLIGHT_ROOT", "QUOTA_PER_CELL", "SAMPLER_CLASS", "SCENE_BY_POSE",
    "SPLIT_MASTER_SEED", "TARGET_SUCCESSES", "TRAINING_ROOT", "TRAIN_COUNT",
    "VALIDATION_COUNT", "WORK_ROOT", "attempt_id", "build_contract",
    "build_row", "canonical_payload_sha256", "cell_key", "cell_seed", "cells",
    "empty_authorization", "quotas", "sha256_file", "sha256_payload",
    "streams_are_disjoint", "wilson_interval", "write_immutable_create_only",
]

if __name__ == "__main__":
    print(json.dumps(build_contract(), indent=2, sort_keys=True))
