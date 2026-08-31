#!/usr/bin/env python3
"""V10.10 paired 40-instance evaluation contract.

Preserves the V10.9 balance exactly: all 24 cells present, 16 doubled, 10 per
family, 20 per side, poses 14/13/13. The only differences are the sampler, the
four-object binding, and a new master seed asserted disjoint from every earlier
stream.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v106_contract import INTRUSION_SIDES, V95_LAYOUT_FAMILY_IDS  # noqa: E402
from pact_place_v106_geometry import POSE_IDS  # noqa: E402
from pact_place_v1010_contract import (  # noqa: E402
    ACTIVE_CLUTTER_SLOTS, ACTIVE_CLUTTER_UIDS, EVAL_MASTER_SEED,
    EVAL_NUM_QUERIES, EVAL_TASK_HORIZON, INACTIVE_CLUTTER_SLOTS,
    SAMPLER_CLASS, SCENE_BY_POSE, build_row, cell_key, cells,
)

SCHEMA_VERSION = "pact_place_v1010_eval_manifest_v1"
ENVIRONMENT_VERSION = "pact_place_corridor_v10_10_four_object"
ROLE = "place_v1010_eval"
SMOKE_ROLE = "place_v1010_eval_smoke"
INSTANCE_COUNT = 40
DOUBLED_CELLS = 16
TASK_STREAM_ID = 11
RETRY_STREAM_ID = 13
SMOKE_STREAM_ID = 17
EXPECTED_PER_FAMILY = 10
EXPECTED_PER_SIDE = 20
EXPECTED_PER_POSE = {"center": 14, "neg5": 13, "pos5": 13}
DOUBLES_PER_FAMILY = 4
DOUBLES_PER_SIDE = 8
DOUBLES_PER_POSE = {"center": 6, "neg5": 5, "pos5": 5}


class PactPlaceV1010EvalContractError(ValueError):
    pass


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def derive_seed(candidate_index: int, stream_id: int, retry_index: int = 0):
    entropy = [EVAL_MASTER_SEED, int(candidate_index), int(stream_id), int(retry_index)]
    low, high = (int(v) for v in
                 np.random.SeedSequence(entropy).generate_state(2, dtype=np.uint32))
    return {"seed_u32": low, "seed_u64": low | (high << 32)}


def retry_seed(row: dict[str, Any], retry_index: int):
    entropy = [int(row["master_seed"]), int(row["candidate_index"]),
               RETRY_STREAM_ID, int(retry_index)]
    low, high = (int(v) for v in
                 np.random.SeedSequence(entropy).generate_state(2, dtype=np.uint32))
    return {"seed_u32": low, "seed_u64": low | (high << 32)}


def episode_id_for(candidate_index: int, role: str) -> str:
    return hashlib.sha256("\x1f".join(
        (SCHEMA_VERSION, str(EVAL_MASTER_SEED), role, str(int(candidate_index)))
    ).encode()).hexdigest()


def _satisfies(subset) -> bool:
    f, s, p = {}, {}, {}
    for family, side, pose in subset:
        f[family] = f.get(family, 0) + 1
        s[side] = s.get(side, 0) + 1
        p[pose] = p.get(pose, 0) + 1
    return (all(v == DOUBLES_PER_FAMILY for v in f.values())
            and all(v == DOUBLES_PER_SIDE for v in s.values())
            and p == DOUBLES_PER_POSE)


def doubled_cells() -> list[str]:
    feasible = [x for x in itertools.combinations(cells(), DOUBLED_CELLS) if _satisfies(x)]
    if not feasible:
        raise PactPlaceV1010EvalContractError("no balanced set of 16 cells")
    ranked = sorted(([cell_key(*c) for c in subset] for subset in feasible),
                    key=lambda keys: hashlib.sha256(
                        f"{EVAL_MASTER_SEED}:doubled:"
                        f"{json.dumps(sorted(keys), separators=(',', ':'))}".encode()
                    ).hexdigest())
    return sorted(ranked[0])


def instance_plan() -> list[tuple[str, str, str, int]]:
    doubled = set(doubled_cells())
    plan = [(f, s, p, 0) for f, s, p in cells()]
    plan += [(f, s, p, 1) for f, s, p in cells() if cell_key(f, s, p) in doubled]
    if len(plan) != INSTANCE_COUNT:
        raise PactPlaceV1010EvalContractError(f"{len(plan)} instances")
    return plan


def build_eval_row(candidate_index: int, family: str, side: str, pose: str,
                   repeat: int, *, role: str, stream_id: int) -> dict[str, Any]:
    seed = derive_seed(candidate_index, stream_id)
    row = build_row(family, side, pose, 0)
    for key in ("attempt_id", "episode_id", "attempt_index", "seed_stream",
                "task_seed_u32", "task_seed_u64", "row_sha256"):
        row.pop(key, None)
    row.update({
        "schema_version": SCHEMA_VERSION, "master_seed": EVAL_MASTER_SEED,
        "role": role, "role_index": candidate_index,
        "candidate_index": candidate_index, "repeat_index": int(repeat),
        "episode_id": episode_id_for(candidate_index, role),
        "hazard_present": True,
        "task_seed_u32": int(seed["seed_u32"]), "task_seed_u64": int(seed["seed_u64"]),
        "task_sampler_class": SAMPLER_CLASS,
        "scene_template_id": "pact_place_corridor_v10_7",
    })
    row["row_sha256"] = sha256_payload(row)
    return row


def build_manifest(sensor_names, sensor_order_sha256, excluded_seeds) -> dict[str, Any]:
    rows = [build_eval_row(i, f, s, p, r, role=ROLE, stream_id=TASK_STREAM_ID)
            for i, (f, s, p, r) in enumerate(instance_plan())]
    smoke = [build_eval_row(i, f, s, p, 0, role=SMOKE_ROLE, stream_id=SMOKE_STREAM_ID)
             for i, (f, s, p) in enumerate([cells()[j] for j in (0, 7, 14, 21)])]

    def tally(entries, key):
        out: dict[str, int] = {}
        for e in entries:
            out[e[key]] = out.get(e[key], 0) + 1
        return dict(sorted(out.items()))

    by_family, by_side = tally(rows, "family_id"), tally(rows, "intrusion_side")
    by_pose, by_cell = tally(rows, "pose_id"), tally(rows, "cell")
    problems: list[str] = []
    if any(v != EXPECTED_PER_FAMILY for v in by_family.values()):
        problems.append(f"family totals {by_family}")
    if any(v != EXPECTED_PER_SIDE for v in by_side.values()):
        problems.append(f"side totals {by_side}")
    if by_pose != EXPECTED_PER_POSE:
        problems.append(f"pose totals {by_pose}")
    if len(by_cell) != 24:
        problems.append(f"{len(by_cell)} cells")
    if sum(1 for v in by_cell.values() if v == 2) != DOUBLED_CELLS:
        problems.append("doubled-cell count is wrong")
    eval_seeds = [int(r["task_seed_u32"]) for r in rows]
    smoke_seeds = [int(r["task_seed_u32"]) for r in smoke]
    if len(set(eval_seeds)) != len(eval_seeds):
        problems.append("duplicate evaluation seeds")
    if set(eval_seeds) & set(smoke_seeds):
        problems.append("evaluation and smoke seeds overlap")
    collisions = {}
    for name, seeds in excluded_seeds.items():
        overlap = sorted(set(eval_seeds + smoke_seeds) & set(int(s) for s in seeds))
        if overlap:
            collisions[name] = overlap
            problems.append(f"seed collision with {name}")
    document = {
        "schema_version": SCHEMA_VERSION, "environment_version": ENVIRONMENT_VERSION,
        "master_seed": EVAL_MASTER_SEED, "role": ROLE,
        "total_candidates": len(rows), "task_horizon": EVAL_TASK_HORIZON,
        "num_queries": EVAL_NUM_QUERIES, "end_on_success": False,
        "action_noise_enabled": False, "sampler_class": SAMPLER_CLASS,
        "use_v109r_event_decoder": False,
        "decoder": "original chunk-100 temporal ensemble and gripper decoder",
        "scene_by_pose": SCENE_BY_POSE,
        "active_clutter_slots": list(ACTIVE_CLUTTER_SLOTS),
        "inactive_clutter_slots": list(INACTIVE_CLUTTER_SLOTS),
        "active_clutter_uids": dict(ACTIVE_CLUTTER_UIDS),
        "sensor_names": list(sensor_names),
        "sensor_order_sha256": sensor_order_sha256,
        "doubled_cells": doubled_cells(),
        "balance": {"by_family": by_family, "by_side": by_side,
                    "by_pose": by_pose, "by_cell": by_cell},
        "held_out_seed_audit": {
            "excluded_sources": {k: len(v) for k, v in excluded_seeds.items()},
            "collisions": collisions, "disjoint": not collisions},
        "paired": True, "run_pact_permuted": False,
        "smoke": {"role": SMOKE_ROLE, "instances": len(smoke),
                  "rollouts": 2 * len(smoke),
                  "gates_full_evaluation_on_performance": False, "rows": smoke},
        "problems": problems, "valid": not problems, "rows": rows,
    }
    document["manifest_sha256"] = sha256_payload(
        {k: v for k, v in document.items()
         if k not in ("manifest_sha256", "payload_sha256")})
    return document


def load_manifest(path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    if document.get("schema_version") != SCHEMA_VERSION:
        raise PactPlaceV1010EvalContractError("wrong schema")
    recomputed = sha256_payload(
        {k: v for k, v in document.items()
         if k not in ("manifest_sha256", "payload_sha256")})
    if recomputed != document.get("manifest_sha256"):
        raise PactPlaceV1010EvalContractError("manifest self-hash mismatch")
    if not document.get("valid"):
        raise PactPlaceV1010EvalContractError(f"invalid: {document.get('problems')}")
    for row in document["rows"] + document["smoke"]["rows"]:
        if row["task_sampler_class"] != SAMPLER_CLASS:
            raise PactPlaceV1010EvalContractError(
                f"row {row['candidate_index']} names {row['task_sampler_class']!r}")
        if row["pact_v106_scene_sha256"] != SCENE_BY_POSE[row["pose_id"]]["sha256"]:
            raise PactPlaceV1010EvalContractError("scene hash is not the certified one")
        if list(row["pact_v1010_active_clutter_slots"]) != list(ACTIVE_CLUTTER_SLOTS):
            raise PactPlaceV1010EvalContractError("row does not bind the four slots")
    return document
