#!/usr/bin/env python3
"""V10.9 exploratory conversion / training / evaluation: shared contract.

**Explicit owner authorization** to convert, train, and evaluate the 141 accepted
V10.8 demonstrations, despite V10.7's failed Phase-0 gate and V10.8's early stop.
Neither historical result is altered or reinterpreted here:

* V10.7 Phase-0 remains ``failed_8_of_24_permanently_closed``.
* V10.8 remains an exploratory owner-override collection stopped early by owner
  instruction at 141 of 152 target successes, with quotas unmet.

Everything a later reader needs to check this work was fair is frozen here before
the first conversion: the source population identity, the canonical row order, the
sensor order, the encoder identity, the split algorithm, and the training/eval
parameters.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v105_contract import (  # noqa: E402
    ImmutableArtifactError,
    canonical_payload_sha256,
    empty_authorization,
    file_hashes,
    implementation_digest,
    recompute_payload_sha256,
    sha256_file,
    sha256_payload,
    wilson_interval,
    write_immutable_create_only,
    write_immutable_text_create_only,
)
from pact_place_v108_contract import (  # noqa: E402
    attempt_id as v108_attempt_id,
)
from pact_place_v108_contract import (  # noqa: E402
    cell_key,
    cell_seed,
    cells,
    quotas,
)

__all__ = [
    "ACT_TRAIN_COMMIT_V5", "CANONICAL_SENSOR_NAMES", "COLLECTION_ROOT",
    "CONTRACT_VERSION_V109", "CONVERTED_DATASET_ROOT", "ENCODER_PATH",
    "ENCODER_SHA256", "EVAL_MASTER_SEED", "EVAL_ROOT", "IS_PHASE0_PASS",
    "ImmutableArtifactError", "LEDGER_SHA256", "N_ACCEPTED", "N_ATTEMPTS",
    "PACT_ONLY_FLAGS", "SENSOR_ORDER_SHA256", "SPLIT_MASTER_SEED",
    "TRAIN_COUNT", "TRAINING_ROOT", "V107_PHASE0_RESULT", "VALIDATION_COUNT",
    "WORK_ROOT", "canonical_payload_sha256", "canonical_row_order",
    "cell_key", "cell_seed", "cells", "empty_authorization", "file_hashes",
    "freeze_split", "implementation_digest", "quotas", "rank_key",
    "recompute_payload_sha256", "sha256_file", "sha256_payload",
    "training_command", "wilson_interval", "write_immutable_create_only",
    "write_immutable_text_create_only",
]

CONTRACT_VERSION_V109 = "pact_place_v109_train_eval_v1"
PLAN_RELATIVE = "docs/PACT_PLACE_V109_TRAIN_EVAL_PLAN.md"

# --- historical results this task must not touch ---------------------------
IS_PHASE0_PASS = False
IS_EXPLORATORY_OWNER_OVERRIDE = True
V107_PHASE0_RESULT = "failed_8_of_24_permanently_closed"
V108_STOP_REASON = "owner_instructed_early_stop"

# --- the frozen source population ------------------------------------------
COLLECTION_ROOT = "diagnostics_output/pact_place_v108_collection"
SOURCE_DATASET_ROOT = "assets/datagen/pact_place_corridor_v10_8"
LEDGER_SHA256 = "ca4adea083d4fd0f25eb2e0dfd39b910c36f877ad1d76309beabc563a63038f6"
N_ATTEMPTS = 353
N_ACCEPTED = 141
N_REJECTED = 212
T_MIN = 356
T_MAX = 627
T_SUM = 71_511
N_SENSORS = 40
N_SENSOR_WINDOWS = T_SUM * N_SENSORS  # 2,860,440
ACCEPTED_MIN_CLEARANCE_M = 0.008272895299859126
ACCEPTED_PENDANT_CONTACT_ROWS = 0

# The single rejected attempt that did touch the pendant. Recorded because the
# claim "no pendant involvement anywhere" was false.
PENDANT_CONTACT_ATTEMPT_ID = (
    "1a756c9304311cdc07091641e59af8da16b6098550aa2fc9ce9d1c0c99cb6ae8"
)

# --- outputs ---------------------------------------------------------------
WORK_ROOT = "diagnostics_output/pact_place_v109_train_eval"
EVAL_ROOT = "diagnostics_output/pact_place_v109_eval"
CONVERTED_DATASET_ROOT = "assets/act_style_data/pact_place_v108_141"
TRAINING_ROOT = "/root/pact_place_v108_141_pact_vs_act_chunk100_seed3101"

# --- frozen sensor order (front before back; NOT alphabetical) --------------
CANONICAL_SENSOR_NAMES: tuple[str, ...] = (
    "link1_sensor_0", "link1_sensor_1", "link1_sensor_2", "link1_sensor_3",
    "link1_sensor_4", "link1_sensor_5", "link1_sensor_6",
    "link2_sensor_0", "link2_sensor_1", "link2_sensor_2", "link2_sensor_3",
    "link2_sensor_4", "link2_sensor_5", "link2_sensor_6",
    "link3_sensor_0", "link3_sensor_1", "link3_sensor_2", "link3_sensor_3",
    "link3_sensor_4",
    "link4_sensor_0", "link4_sensor_1", "link4_sensor_2", "link4_sensor_3",
    "link4_sensor_4",
    "link5_front_sensor_0", "link5_front_sensor_1", "link5_front_sensor_2",
    "link5_front_sensor_3",
    "link5_back_sensor_0", "link5_back_sensor_1", "link5_back_sensor_2",
    "link5_back_sensor_3", "link5_back_sensor_4", "link5_back_sensor_5",
    "link6_sensor_0", "link6_sensor_1", "link6_sensor_2", "link6_sensor_3",
    "link6_sensor_4", "link6_sensor_5",
)
SENSOR_ORDER_SHA256 = (
    "2198e29b796ce63f43d8b0db50a92da7d4429895f8571f7d87b655bc265c8fe1"
)

# --- frozen proximity encoder ----------------------------------------------
ENCODER_PATH = "/root/pact_frontend_screen_artifacts/encoder_v1/embedding_encoder_frozen.pt"
ENCODER_SHA256 = (
    "6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206"
)
ENCODER_SCHEMA = "pact_surface_embedding_encoder_v1"
ENCODER_CLASS = "SurfaceEmbeddingEncoder"
PROXIMITY_FEATURE_DIM = 32

# --- split -----------------------------------------------------------------
SPLIT_MASTER_SEED = 2026082901
TRAIN_COUNT = 113
VALIDATION_COUNT = 28

# --- evaluation ------------------------------------------------------------
EVAL_MASTER_SEED = 2026082902
EVAL_INSTANCES = 40
EVAL_DOUBLED_CELLS = 16
EVAL_SMOKE_INSTANCES = 4
EVAL_TASK_HORIZON = 900
EVAL_NUM_QUERIES = 100

# --- training --------------------------------------------------------------
ACT_TRAIN_COMMIT_V5 = "01751759c49d7237f2b14ff8a16fd3c10ae4c089"
EPISODE_HORIZON = 635  # T_MAX 627 < 635
TRAIN_PARAMS: dict[str, Any] = {
    "task_name": "obstacle_baseline",
    "policy_class": "ACT",
    "batch_size": 8,
    "seed": 3101,
    "num_epochs": 2000,
    "lr": "1e-5",
    "kl_weight": 10,
    "chunk_size": 100,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "enc_layers": 7,
    "dec_layers": 7,
    "camera_names": "wrist_camera",
    "episode_horizon": EPISODE_HORIZON,
    "state_dim": 9,
    "action_dim": 8,
    "num_workers": 4,
    "ckpt_every": 200,
}
PACT_ONLY_FLAGS: dict[str, Any] = {
    "--use_proximity": True,
    "--n_proximity_sensors": "40",
    "--prox_tokens_per_sensor": "1",
    "--proximity_feature_dim": "32",
    "--proximity_encoder_sha256": ENCODER_SHA256,
}

IMPLEMENTATION_FILES: tuple[str, ...] = (
    "scripts/pact_place_v109_contract.py",
    "scripts/verify_pact_place_v109_source.py",
    "scripts/convert_pact_place_v109_to_act.py",
    "scripts/build_pact_place_v109_split.py",
    "scripts/run_pact_place_v109_train.py",
)


# ---------------------------------------------------------------------------
# Canonical row order
# ---------------------------------------------------------------------------
def canonical_cell_index() -> dict[str, int]:
    """Registered V10.8 cell order -> position. Never alphabetical."""
    return {cell_key(*c): i for i, c in enumerate(cells())}


def canonical_row_order(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Registered cell order, then attempt_index, then attempt_id.

    Parallel completion order and ledger arrival order are both
    nondeterministic; neither may decide which episode gets which index.
    """
    index = canonical_cell_index()
    unknown = sorted({r["cell"] for r in rows} - set(index))
    if unknown:
        raise ValueError(f"rows carry unregistered cells: {unknown}")
    return sorted(
        rows,
        key=lambda r: (index[r["cell"]], int(r["attempt_index"]), r["attempt_id"]),
    )


# ---------------------------------------------------------------------------
# Frozen split algorithm (see plan section 4)
# ---------------------------------------------------------------------------
def rank_key(*parts: Any) -> str:
    """Deterministic hash rank over the split seed and the given parts."""
    joined = ":".join(str(p) for p in parts)
    return hashlib.sha256(f"{SPLIT_MASTER_SEED}:{joined}".encode()).hexdigest()


SOLE_ROW_CELL = "F3_aperture_side_stagger|right|neg5"


def freeze_split(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Cell-stratified 113/28 split, decided only by identity hashes.

    Nothing about a trajectory -- loss, length, clearance, contact, or any
    learned outcome -- is consulted. Only ``attempt_id`` and the cell key.
    """
    if len(rows) != N_ACCEPTED:
        raise ValueError(f"expected {N_ACCEPTED} rows, got {len(rows)}")
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault(row["cell"], []).append(row)
    for cell, members in by_cell.items():
        members.sort(key=lambda r: rank_key(r["attempt_id"]))

    validation: list[str] = []
    reserved_from: list[str] = []
    # Step 1: the sole-row cell can never be split; it goes to training.
    if len(by_cell.get(SOLE_ROW_CELL, [])) != 1:
        raise ValueError(
            f"{SOLE_ROW_CELL} was expected to hold exactly one row, "
            f"got {len(by_cell.get(SOLE_ROW_CELL, []))}"
        )
    # Step 2: one hash-ranked validation row from every other nonempty cell.
    for cell in sorted(by_cell):
        if cell == SOLE_ROW_CELL:
            continue
        validation.append(by_cell[cell][0]["attempt_id"])
        reserved_from.append(cell)
    if len(validation) != 23:
        raise ValueError(f"step 2 reserved {len(validation)} rows, expected 23")

    # Step 3: remaining slots by largest remainder, capped to keep >=1 train row.
    remaining = VALIDATION_COUNT - len(validation)
    total_pool = sum(len(v) for c, v in by_cell.items() if c != SOLE_ROW_CELL)
    quota_share = {
        cell: (len(members) * VALIDATION_COUNT) / total_pool
        for cell, members in by_cell.items()
        if cell != SOLE_ROW_CELL
    }
    remainders = sorted(
        quota_share,
        key=lambda cell: (
            -(quota_share[cell] - 1.0),          # largest remainder above the 1 already given
            rank_key("cell", cell),               # deterministic tie-break
        ),
    )
    extra: dict[str, int] = {cell: 0 for cell in quota_share}
    cursor = 0
    while remaining > 0:
        progressed = False
        for cell in remainders:
            if remaining == 0:
                break
            taken = 1 + extra[cell]
            # Cap: the cell must keep at least one training row. Rows 0..taken
            # would be validation, i.e. taken+1 of them, so require
            # taken + 1 <= len - 1.
            if taken >= len(by_cell[cell]) - 1:
                continue
            extra[cell] += 1
            validation.append(by_cell[cell][taken]["attempt_id"])
            remaining -= 1
            progressed = True
        cursor += 1
        if not progressed:
            raise ValueError("cannot allocate remaining validation slots under the cap")

    validation_set = set(validation)
    if len(validation_set) != VALIDATION_COUNT:
        raise ValueError(f"validation holds {len(validation_set)} unique rows")
    assignments = {
        row["attempt_id"]: ("validation" if row["attempt_id"] in validation_set else "train")
        for row in rows
    }
    n_train = sum(1 for v in assignments.values() if v == "train")
    if n_train != TRAIN_COUNT:
        raise ValueError(f"train holds {n_train} rows, expected {TRAIN_COUNT}")
    train_cells = {r["cell"] for r in rows if assignments[r["attempt_id"]] == "train"}
    val_cells = {r["cell"] for r in rows if assignments[r["attempt_id"]] == "validation"}
    if len(train_cells) != 24:
        raise ValueError(f"training covers {len(train_cells)} cells, expected 24")
    if len(val_cells) != 23:
        raise ValueError(f"validation covers {len(val_cells)} cells, expected 23")
    return {
        "assignments": assignments,
        "extra_validation_by_cell": {c: n for c, n in extra.items() if n},
        "train_cells": sorted(train_cells),
        "validation_cells": sorted(val_cells),
        "sole_row_cell": SOLE_ROW_CELL,
    }


# ---------------------------------------------------------------------------
# Training commands
# ---------------------------------------------------------------------------
def training_command(
    *,
    arm: str,
    ckpt_dir: str,
    dataset_dir: str,
    split_manifest: str,
    dataset_manifest: str,
    expect_split_sha256: str,
    expect_dataset_tree_sha256: str,
) -> list[str]:
    """The frozen command for one arm. PACT appends only the five extra flags."""
    if arm not in ("act", "pact"):
        raise ValueError(f"unknown arm {arm!r}")
    p = TRAIN_PARAMS
    command = [
        "python", "imitate_episodes.py",
        "--task_name", p["task_name"],
        "--ckpt_dir", ckpt_dir,
        "--exact_ckpt_dir",
        "--policy_class", p["policy_class"],
        "--batch_size", str(p["batch_size"]),
        "--seed", str(p["seed"]),
        "--num_epochs", str(p["num_epochs"]),
        "--lr", p["lr"],
        "--kl_weight", str(p["kl_weight"]),
        "--chunk_size", str(p["chunk_size"]),
        "--hidden_dim", str(p["hidden_dim"]),
        "--dim_feedforward", str(p["dim_feedforward"]),
        "--enc_layers", str(p["enc_layers"]),
        "--dec_layers", str(p["dec_layers"]),
        "--camera_names", p["camera_names"],
        "--dataset_dir", dataset_dir,
        "--split_manifest", split_manifest,
        "--dataset_manifest", dataset_manifest,
        "--expect_split_sha256", expect_split_sha256,
        "--expect_dataset_tree_sha256", expect_dataset_tree_sha256,
        "--episode_horizon", str(p["episode_horizon"]),
        "--state_dim", str(p["state_dim"]),
        "--action_dim", str(p["action_dim"]),
        "--num_workers", str(p["num_workers"]),
        "--ckpt_every", str(p["ckpt_every"]),
        "--no_wandb",
    ]
    if arm == "pact":
        command += [
            "--use_proximity",
            "--n_proximity_sensors", PACT_ONLY_FLAGS["--n_proximity_sensors"],
            "--prox_tokens_per_sensor", PACT_ONLY_FLAGS["--prox_tokens_per_sensor"],
            "--proximity_feature_dim", PACT_ONLY_FLAGS["--proximity_feature_dim"],
            "--proximity_encoder_sha256", PACT_ONLY_FLAGS["--proximity_encoder_sha256"],
        ]
    return command


def parse_flags(command: Sequence[str]) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    i = 0
    while i < len(command):
        token = command[i]
        if token.startswith("--"):
            if i + 1 < len(command) and not command[i + 1].startswith("--"):
                flags[token] = command[i + 1]
                i += 2
            else:
                flags[token] = True
                i += 1
        else:
            i += 1
    return flags


def command_diff(act: Sequence[str], pact: Sequence[str]) -> dict[str, Any]:
    """Parsed flag/value diff. Fails closed on anything beyond the allowance."""
    fa, fp = parse_flags(act), parse_flags(pact)
    only_pact = {k: v for k, v in fp.items() if k not in fa}
    only_act = {k: v for k, v in fa.items() if k not in fp}
    differing = {k: [fa[k], fp[k]] for k in fa if k in fp and fa[k] != fp[k]}
    violations: list[str] = []
    if only_act:
        violations.append(f"flags present only in ACT: {sorted(only_act)}")
    if set(only_pact) != set(PACT_ONLY_FLAGS):
        violations.append(
            f"PACT-only flags {sorted(only_pact)} != allowed {sorted(PACT_ONLY_FLAGS)}"
        )
    else:
        for key, value in PACT_ONLY_FLAGS.items():
            if only_pact[key] != value:
                violations.append(f"{key} is {only_pact[key]!r}, expected {value!r}")
    if set(differing) - {"--ckpt_dir"}:
        violations.append(
            f"values differ beyond --ckpt_dir: {sorted(set(differing) - {'--ckpt_dir'})}"
        )
    return {
        "only_in_pact": only_pact,
        "only_in_act": only_act,
        "differing_values": differing,
        "violations": violations,
        "identical_except_allowance": not violations,
    }


def build_contract() -> dict[str, Any]:
    return {
        **empty_authorization(),
        "schema_version": CONTRACT_VERSION_V109,
        "plan": PLAN_RELATIVE,
        "is_phase0_pass": IS_PHASE0_PASS,
        "is_exploratory_owner_override": IS_EXPLORATORY_OWNER_OVERRIDE,
        "v107_phase0_result": V107_PHASE0_RESULT,
        "v108_stop_reason": V108_STOP_REASON,
        "source": {
            "collection_root": COLLECTION_ROOT,
            "dataset_root": SOURCE_DATASET_ROOT,
            "ledger_sha256": LEDGER_SHA256,
            "n_attempts": N_ATTEMPTS,
            "n_accepted": N_ACCEPTED,
            "n_rejected": N_REJECTED,
            "t_min": T_MIN, "t_max": T_MAX, "t_sum": T_SUM,
            "n_sensors": N_SENSORS, "n_sensor_windows": N_SENSOR_WINDOWS,
            "accepted_min_clearance_m": ACCEPTED_MIN_CLEARANCE_M,
            "accepted_pendant_contact_rows": ACCEPTED_PENDANT_CONTACT_ROWS,
            "pendant_contact_attempt_id": PENDANT_CONTACT_ATTEMPT_ID,
        },
        "sensor_order": {
            "names": list(CANONICAL_SENSOR_NAMES),
            "sha256": SENSOR_ORDER_SHA256,
            "is_alphabetical": list(CANONICAL_SENSOR_NAMES) == sorted(CANONICAL_SENSOR_NAMES),
        },
        "encoder": {
            "path": ENCODER_PATH, "sha256": ENCODER_SHA256,
            "schema": ENCODER_SCHEMA, "class": ENCODER_CLASS,
            "feature_dim": PROXIMITY_FEATURE_DIM,
        },
        "split": {
            "master_seed": SPLIT_MASTER_SEED,
            "train": TRAIN_COUNT, "validation": VALIDATION_COUNT,
            "sole_row_cell": SOLE_ROW_CELL,
        },
        "training": {
            "root": TRAINING_ROOT,
            "params": TRAIN_PARAMS,
            "pact_only_flags": PACT_ONLY_FLAGS,
            "act_train_commit_v5": ACT_TRAIN_COMMIT_V5,
        },
        "evaluation": {
            "master_seed": EVAL_MASTER_SEED,
            "instances": EVAL_INSTANCES,
            "doubled_cells": EVAL_DOUBLED_CELLS,
            "smoke_instances": EVAL_SMOKE_INSTANCES,
            "task_horizon": EVAL_TASK_HORIZON,
            "num_queries": EVAL_NUM_QUERIES,
            "run_pact_permuted": False,
        },
    }


if __name__ == "__main__":
    doc = build_contract()
    doc["config_sha256"] = canonical_payload_sha256(doc)
    print(json.dumps(doc, indent=2, sort_keys=True))
