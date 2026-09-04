#!/usr/bin/env python3
"""V10.9 step 8: the frozen paired 40-instance held-out evaluation contract.

Forty held-out physical instances, run by both arms on the identical instances:
80 learned-policy rollouts. The balance, the doubled cells, and every seed are
fixed here before any rollout, and the whole manifest carries a self-hash.

Balance (all asserted, not assumed):

* each of the 24 family x side x pose cells appears at least once
* 16 cells receive a second instance
* 10 instances per family, 20 left / 20 right
* pose totals 14 center, 13 neg5, 13 pos5

Those four constraints are simultaneously satisfiable and jointly pin the
doubled cells to exactly 4 per family, 8 per side, and 6/5/5 by pose. The
particular set of 16 is chosen by ranking every feasible set by SHA-256 under
the evaluation master seed and taking the first -- a constrained selection that
no later reader can accuse of being picked for its outcome.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v106_contract import INTRUSION_SIDES, V95_LAYOUT_FAMILY_IDS  # noqa: E402
from pact_place_v106_geometry import POSE_IDS, POSE_OFFSETS_M  # noqa: E402
from pact_place_v108_contract import (  # noqa: E402
    MAX_SAMPLING_RETRIES,
    cell_key,
    cells,
    v95_row_payload,
)

SCHEMA_VERSION = "pact_place_v109_eval_manifest_v1"
CONTRACT_VERSION = "pact_place_v109_eval_v1"
ENVIRONMENT_VERSION = "pact_place_corridor_v10_6_v95_clutter_asymmetric_pendant"
SAMPLER_CLASS = "PactPlaceCorridorV106Sampler"

EVAL_MASTER_SEED = 2026082902
INSTANCE_COUNT = 40
DOUBLED_CELLS = 16
SMOKE_INSTANCE_COUNT = 4
EVAL_TASK_HORIZON = 900
EVAL_NUM_QUERIES = 100
ROLE = "place_v109_eval"
SMOKE_ROLE = "place_v109_eval_smoke"

TASK_STREAM_ID = 1
RETRY_STREAM_ID = 3
SMOKE_STREAM_ID = 7

EXPECTED_PER_FAMILY = 10
EXPECTED_PER_SIDE = 20
EXPECTED_PER_POSE = {"center": 14, "neg5": 13, "pos5": 13}
DOUBLES_PER_FAMILY = 4
DOUBLES_PER_SIDE = 8
DOUBLES_PER_POSE = {"center": 6, "neg5": 5, "pos5": 5}

# The three certified V10.7 static-pendant scenes and their certification hashes.
SCENE_BY_POSE: dict[str, dict[str, str]] = {
    "neg5": {
        "relative": "submodules/molmospaces/molmo_spaces/data_generation/"
                    "custom_scenes/pact_place_corridor_v10_7_neg5.xml",
        "sha256": "df50679c749c6ad771d00023e73a08e0bfaf59d5391df9b42cf05de4ed7893a7",
    },
    "center": {
        "relative": "submodules/molmospaces/molmo_spaces/data_generation/"
                    "custom_scenes/pact_place_corridor_v10_7_center.xml",
        "sha256": "b5a41d0d8934240b078f1cdbf3a6991b2e94a46558ddf1c9eae0119c8b8e138a",
    },
    "pos5": {
        "relative": "submodules/molmospaces/molmo_spaces/data_generation/"
                    "custom_scenes/pact_place_corridor_v10_7_pos5.xml",
        "sha256": "762a5a4662a8fc0d31a3a0ee1135b347d6dd2c882daf4e65c2f706ab2d6fe565",
    },
}
# The certified V10.7 selection, unchanged.
SELECTED_ASSEMBLY = {"x_m": 0.800, "r_neg_m": 0.330, "r_pos_m": 0.300}


class PactPlaceV109EvalContractError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def derive_seed(candidate_index: int, stream_id: int, retry_index: int = 0) -> dict[str, int]:
    entropy = [EVAL_MASTER_SEED, int(candidate_index), int(stream_id), int(retry_index)]
    low, high = (
        int(v) for v in np.random.SeedSequence(entropy).generate_state(2, dtype=np.uint32)
    )
    return {"seed_u32": low, "seed_u64": low | (high << 32)}


def retry_seed(row: dict[str, Any], retry_index: int) -> dict[str, int]:
    entropy = [int(row["master_seed"]), int(row["candidate_index"]),
               RETRY_STREAM_ID, int(retry_index)]
    low, high = (
        int(v) for v in np.random.SeedSequence(entropy).generate_state(2, dtype=np.uint32)
    )
    return {"seed_u32": low, "seed_u64": low | (high << 32)}


def episode_id_for(candidate_index: int, role: str) -> str:
    preimage = "\x1f".join(
        (SCHEMA_VERSION, str(EVAL_MASTER_SEED), role, str(int(candidate_index))))
    return hashlib.sha256(preimage.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Deterministic constrained selection of the 16 doubled cells
# ---------------------------------------------------------------------------
def _satisfies(subset: Iterable[tuple[str, str, str]]) -> bool:
    families: dict[str, int] = {f: 0 for f in V95_LAYOUT_FAMILY_IDS}
    sides: dict[str, int] = {s: 0 for s in INTRUSION_SIDES}
    poses: dict[str, int] = {p: 0 for p in POSE_IDS}
    for family, side, pose in subset:
        families[family] += 1
        sides[side] += 1
        poses[pose] += 1
    return (
        all(v == DOUBLES_PER_FAMILY for v in families.values())
        and all(v == DOUBLES_PER_SIDE for v in sides.values())
        and poses == DOUBLES_PER_POSE
    )


def doubled_cells() -> list[str]:
    """The 16 cells receiving a second instance, hash-ranked among feasible sets."""
    universe = cells()
    feasible = [
        subset for subset in itertools.combinations(universe, DOUBLED_CELLS)
        if _satisfies(subset)
    ]
    if not feasible:
        raise PactPlaceV109EvalContractError(
            "no set of 16 cells satisfies the family/side/pose balance")
    ranked = sorted(
        ([cell_key(*c) for c in subset] for subset in feasible),
        key=lambda keys: hashlib.sha256(
            f"{EVAL_MASTER_SEED}:doubled:{canonical_json(sorted(keys))}".encode()
        ).hexdigest(),
    )
    chosen = sorted(ranked[0])
    if len(chosen) != DOUBLED_CELLS or len(set(chosen)) != DOUBLED_CELLS:
        raise PactPlaceV109EvalContractError("doubled-cell selection is malformed")
    return chosen


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------
def instance_plan() -> list[tuple[str, str, str, int]]:
    """(family, side, pose, repeat_index) for all 40 instances, canonically ordered."""
    doubled = set(doubled_cells())
    plan: list[tuple[str, str, str, int]] = [(f, s, p, 0) for f, s, p in cells()]
    plan += [(f, s, p, 1) for f, s, p in cells() if cell_key(f, s, p) in doubled]
    if len(plan) != INSTANCE_COUNT:
        raise PactPlaceV109EvalContractError(f"plan holds {len(plan)} instances")
    return plan


def build_row(candidate_index: int, family: str, side: str, pose: str,
              repeat_index: int, *, role: str, stream_id: int) -> dict[str, Any]:
    seed = derive_seed(candidate_index, stream_id)
    payload = v95_row_payload(family, side)
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "master_seed": EVAL_MASTER_SEED,
        "role": role,
        "role_index": candidate_index,
        "candidate_index": candidate_index,
        "repeat_index": int(repeat_index),
        "episode_id": episode_id_for(candidate_index, role),
        "cell": cell_key(family, side, pose),
        "family_id": family, "family": family, "layout_family_id": family,
        "intrusion_side": side,
        "pose_id": pose,
        "pose_offset_m": POSE_OFFSETS_M[pose],
        "hazard_present": True,
        "max_sampling_retries": MAX_SAMPLING_RETRIES,
        "scene_template_house_index": 1,
        "scene_template_id": "pact_place_corridor_v10_7",
        "task_sampler_class": SAMPLER_CLASS,
        "task_seed_u32": int(seed["seed_u32"]),
        "task_seed_u64": int(seed["seed_u64"]),
        "pact_v106_x_m": float(SELECTED_ASSEMBLY["x_m"]),
        "pact_v106_r_neg_m": float(SELECTED_ASSEMBLY["r_neg_m"]),
        "pact_v106_r_pos_m": float(SELECTED_ASSEMBLY["r_pos_m"]),
        "pact_v106_scene_sha256": SCENE_BY_POSE[pose]["sha256"],
        "pact_v109_scene_relative": SCENE_BY_POSE[pose]["relative"],
        **{k: (dict(v) if isinstance(v, dict)
               else list(v) if isinstance(v, list) else v)
           for k, v in payload.items()},
    }
    row["row_sha256"] = sha256_payload(row)
    return row


def build_manifest(sensor_names: list[str], sensor_order_sha256: str,
                   excluded_seeds: dict[str, list[int]]) -> dict[str, Any]:
    rows = [
        build_row(index, family, side, pose, repeat,
                  role=ROLE, stream_id=TASK_STREAM_ID)
        for index, (family, side, pose, repeat) in enumerate(instance_plan())
    ]
    smoke_plan = [cells()[i] for i in (0, 7, 14, 21)]
    smoke_rows = [
        build_row(index, family, side, pose, 0,
                  role=SMOKE_ROLE, stream_id=SMOKE_STREAM_ID)
        for index, (family, side, pose) in enumerate(smoke_plan)
    ]

    def tally(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for entry in entries:
            out[entry[key]] = out.get(entry[key], 0) + 1
        return dict(sorted(out.items()))

    by_family = tally(rows, "family_id")
    by_side = tally(rows, "intrusion_side")
    by_pose = tally(rows, "pose_id")
    by_cell = tally(rows, "cell")
    problems: list[str] = []
    if len(rows) != INSTANCE_COUNT:
        problems.append(f"{len(rows)} instances")
    if any(v != EXPECTED_PER_FAMILY for v in by_family.values()):
        problems.append(f"family totals {by_family}")
    if any(v != EXPECTED_PER_SIDE for v in by_side.values()):
        problems.append(f"side totals {by_side}")
    if by_pose != EXPECTED_PER_POSE:
        problems.append(f"pose totals {by_pose} != {EXPECTED_PER_POSE}")
    if len(by_cell) != 24:
        problems.append(f"{len(by_cell)} cells represented, expected 24")
    if sum(1 for v in by_cell.values() if v == 2) != DOUBLED_CELLS:
        problems.append(f"{sum(1 for v in by_cell.values() if v == 2)} doubled cells")
    if any(v not in (1, 2) for v in by_cell.values()):
        problems.append("a cell holds other than 1 or 2 instances")

    eval_seeds = [int(r["task_seed_u32"]) for r in rows]
    smoke_seeds = [int(r["task_seed_u32"]) for r in smoke_rows]
    if len(set(eval_seeds)) != len(eval_seeds):
        problems.append("duplicate evaluation task seeds")
    if set(eval_seeds) & set(smoke_seeds):
        problems.append("evaluation and smoke seeds overlap")
    if len({r["episode_id"] for r in rows + smoke_rows}) != len(rows) + len(smoke_rows):
        problems.append("duplicate episode ids")
    collisions: dict[str, list[int]] = {}
    for name, seeds in excluded_seeds.items():
        overlap = sorted(set(eval_seeds + smoke_seeds) & set(int(s) for s in seeds))
        if overlap:
            collisions[name] = overlap
            problems.append(f"seed collision with {name}: {overlap[:4]}")

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "master_seed": EVAL_MASTER_SEED,
        "role": ROLE,
        "total_candidates": len(rows),
        "task_horizon": EVAL_TASK_HORIZON,
        "num_queries": EVAL_NUM_QUERIES,
        "end_on_success": False,
        "action_noise_enabled": False,
        "sampler_class": SAMPLER_CLASS,
        "scene_by_pose": SCENE_BY_POSE,
        "selected_assembly": SELECTED_ASSEMBLY,
        "sensor_names": list(sensor_names),
        "sensor_order_sha256": sensor_order_sha256,
        "doubled_cells": doubled_cells(),
        "balance": {
            "by_family": by_family, "by_side": by_side,
            "by_pose": by_pose, "by_cell": by_cell,
            "expected_per_family": EXPECTED_PER_FAMILY,
            "expected_per_side": EXPECTED_PER_SIDE,
            "expected_per_pose": EXPECTED_PER_POSE,
        },
        "held_out_seed_audit": {
            "excluded_sources": {k: len(v) for k, v in excluded_seeds.items()},
            "collisions": collisions,
            "disjoint": not collisions,
        },
        "paired": True,
        "paired_note":
            "ACT and PACT run the identical rows: same task seed, pendant pose, "
            "clutter layout, and sampler. No completed scientific row is replaced, "
            "cherry-picked, or reseeded.",
        "run_pact_permuted": False,
        "smoke": {
            "role": SMOKE_ROLE,
            "instances": len(smoke_rows),
            "rollouts": 2 * len(smoke_rows),
            "gates_full_evaluation_on_performance": False,
            "note": "infrastructure only: checkpoint loading, scene identity, "
                    "telemetry, memory, ETA. Poor performance does not stop the "
                    "full evaluation.",
            "rows": smoke_rows,
        },
        "problems": problems,
        "valid": not problems,
        "rows": rows,
    }
    # The immutable writer appends its own ``payload_sha256`` to whatever it is
    # given, so the self-hash must be computed over the document without either
    # self-referential key or it can never validate after a round trip.
    document["manifest_sha256"] = sha256_payload(
        {k: v for k, v in document.items()
         if k not in ("manifest_sha256", "payload_sha256")})
    return document


def load_manifest(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    if document.get("schema_version") != SCHEMA_VERSION:
        raise PactPlaceV109EvalContractError(
            f"manifest schema {document.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    stored = document.get("manifest_sha256")
    recomputed = sha256_payload(
        {k: v for k, v in document.items()
         if k not in ("manifest_sha256", "payload_sha256")})
    if stored != recomputed:
        raise PactPlaceV109EvalContractError(
            f"manifest self-hash mismatch: {recomputed} != {stored}")
    if not document.get("valid"):
        raise PactPlaceV109EvalContractError(
            f"manifest is not valid: {document.get('problems')}")
    for row in document["rows"] + document["smoke"]["rows"]:
        if row["task_sampler_class"] != SAMPLER_CLASS:
            raise PactPlaceV109EvalContractError(
                f"row {row['candidate_index']} names sampler {row['task_sampler_class']!r}")
        if row["pact_v106_scene_sha256"] != SCENE_BY_POSE[row["pose_id"]]["sha256"]:
            raise PactPlaceV109EvalContractError(
                f"row {row['candidate_index']} scene hash is not the certified one")
        if sha256_payload({k: v for k, v in row.items() if k != "row_sha256"}) \
                != row["row_sha256"]:
            raise PactPlaceV109EvalContractError(
                f"row {row['candidate_index']} self-hash mismatch")
    return document


__all__ = [
    "CONTRACT_VERSION", "DOUBLED_CELLS", "ENVIRONMENT_VERSION",
    "EVAL_MASTER_SEED", "EVAL_NUM_QUERIES", "EVAL_TASK_HORIZON",
    "INSTANCE_COUNT", "PactPlaceV109EvalContractError", "ROLE", "SAMPLER_CLASS",
    "SCENE_BY_POSE", "SCHEMA_VERSION", "SELECTED_ASSEMBLY", "SMOKE_ROLE",
    "build_manifest", "build_row", "derive_seed", "doubled_cells",
    "episode_id_for", "instance_plan", "load_manifest", "retry_seed",
    "sha256_payload",
]
