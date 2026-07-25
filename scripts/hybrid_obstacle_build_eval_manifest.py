#!/usr/bin/env python3
"""Build the held-out expert-feasible evaluation pool and the frozen smoke manifest.

Handoff step 6. The evaluation pool is exactly the successful collection rows that
the canonical 100-episode selection excluded -- 35 hazard-present and 10
hazard-absent -- so no evaluation row was ever seen by ACT training, validation,
or the normalization statistics.

This is an **expert-feasible held-out pool**, not an unrestricted sample of every
environment attempt: the 15 rows whose expert planner failed during collection are
deliberately not in it, because there is no accepted initial state to replay for
them. That limitation is recorded in the manifest itself.

Each row carries everything needed to replay the accepted attempt exactly:
candidate index, episode ID, manifest-row hash, hazard label, accepted retry
index, source H5 hash, an initial-state hash, robot initial qpos, target/object
identity and pose, obstacle theta, the sensor-order hash and the predeclared
ranks.

The smoke manifest is the three lowest-ranked hazard-present reserve rows plus the
lowest-ranked hazard-absent reserve row, chosen by predeclared stratum rank before
any rollout was run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL_LABEL = "held_out_expert_feasible_reserve_v1"


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def initial_state_hash(observations: dict, row: dict) -> str:
    """Hash exactly the fields that define the accepted initial task state."""
    return canonical_hash({
        "robot_initial_qpos": observations.get("robot_initial_qpos"),
        "robot_initial_qvel": observations.get("robot_initial_qvel"),
        "object_initial_pose": observations.get("object_initial_pose"),
        "mocap_pos": observations.get("mocap_pos"),
        "obstacle_theta": observations.get("obstacle_theta"),
        "target_uid": observations.get("target_uid"),
        "selected_object": observations.get("selected_object"),
        "hazard_present": bool(row["hazard_present"]),
        "candidate_index": int(row["candidate_index"]),
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--collection-manifest", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path, help="source collection run dir")
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--out-pool", required=True, type=Path)
    ap.add_argument("--out-smoke", required=True, type=Path)
    args = ap.parse_args()

    canon = json.loads(args.canonical.read_text())
    split = json.loads(args.split.read_text())
    coll = json.loads(args.collection_manifest.read_text())
    stack = json.loads(args.stack.read_text())
    sensor_order_hash = stack["sensor_contract"]["sensor_order_hash"]

    coll_rows = {r["episode_id"]: r for r in coll["rows"]}
    excluded = canon["excluded_successful"]

    # ---- no-overlap proof -------------------------------------------------
    train_ids = {e["episode_id"] for e in split["episodes"] if e["split"] == "train"}
    val_ids = {e["episode_id"] for e in split["episodes"] if e["split"] == "validation"}
    train_hashes = {e["source_h5_sha256"] for e in split["episodes"] if e["split"] == "train"}
    val_hashes = {e["source_h5_sha256"] for e in split["episodes"] if e["split"] == "validation"}
    selected_ids = {e["episode_id"] for e in canon["selected"]}

    pool_ids = {e["episode_id"] for e in excluded}
    pool_hashes = {e["source_h5_sha256"] for e in excluded}
    overlap = {
        "pool_vs_train_ids": sorted(pool_ids & train_ids),
        "pool_vs_validation_ids": sorted(pool_ids & val_ids),
        "pool_vs_canonical_selected_ids": sorted(pool_ids & selected_ids),
        "pool_vs_train_source_hashes": sorted(pool_hashes & train_hashes),
        "pool_vs_validation_source_hashes": sorted(pool_hashes & val_hashes),
    }
    if any(overlap.values()):
        raise SystemExit(f"evaluation pool overlaps the training or validation data: {overlap}")

    # ---- assemble the pool with full replay state -------------------------
    rows = []
    for e in excluded:
        eid = e["episode_id"]
        outcome = json.loads((args.run / "rows" / eid / "outcome.json").read_text())
        if outcome["status"] != "success":
            raise SystemExit(f"{eid} is in the excluded-successful list but its outcome is "
                             f"{outcome['status']!r}")
        obs = outcome.get("observations", {}) or {}
        crow = coll_rows[eid]
        rows.append({
            "candidate_index": int(e["candidate_index"]),
            "episode_id": eid,
            "manifest_row_sha256": e["row_sha256"],
            "hazard_present": bool(e["hazard_present"]),
            "hazard_stratum": e["hazard_stratum"],
            "accepted_retry_index": int(outcome.get("retry_count", 0)),
            "retry_history": outcome.get("retry_history", []),
            "source_h5_sha256": e["source_h5_sha256"],
            "source_relpath": e["source_relpath"],
            "initial_state_sha256": initial_state_hash(obs, e),
            "task_state_sha256": e["task_state_sha256"],
            "core_trajectory_sha256": e["core_trajectory_sha256"],
            "episode_spec_sha256": e["episode_spec_sha256"],
            "robot_initial_qpos": obs.get("robot_initial_qpos"),
            "robot_initial_qvel_len": len(obs.get("robot_initial_qvel") or []),
            "object_initial_pose": obs.get("object_initial_pose"),
            "mocap_pos": obs.get("mocap_pos"),
            "target_uid": obs.get("target_uid"),
            "selected_object": obs.get("selected_object"),
            "selected_grasp": obs.get("selected_grasp"),
            "obstacle_theta": obs.get("obstacle_theta"),
            "observed_hazard_present": obs.get("observed_hazard_present"),
            "scene_template_id": crow["scene_template_id"],
            "scene_template_house_index": crow["scene_template_house_index"],
            "sensor_order_sha256": sensor_order_hash,
            "predeclared_stratum_rank": int(e["predeclared_stratum_rank"]),
            "predeclared_canonical_rank": e["predeclared_canonical_rank"],
            "reserve_rank_within_stratum": None,   # filled below
            "selection_position": int(e["selection_position"]),
            "manifest_split_label": e["manifest_split_label"],
            "exclusion_reason": e["exclusion_reason"],
            "master_seed": int(crow["master_seed"]),
            "max_retries": int(crow["max_retries"]),
            "seed_map": crow["seed_map"],
        })

    rows.sort(key=lambda r: (not r["hazard_present"], r["predeclared_stratum_rank"]))
    for stratum in ("hazard_present", "hazard_absent"):
        subset = [r for r in rows if r["hazard_stratum"] == stratum]
        subset.sort(key=lambda r: r["predeclared_stratum_rank"])
        for i, r in enumerate(subset):
            r["reserve_rank_within_stratum"] = i

    n_present = sum(1 for r in rows if r["hazard_present"])
    n_absent = len(rows) - n_present
    if (n_present, n_absent) != (35, 10):
        raise SystemExit(f"expected 35/10 reserve composition, got {n_present}/{n_absent}")

    pool = {
        "schema": "hybrid_obstacle_eval_pool_v1",
        "label": POOL_LABEL,
        "description": (
            "Successful collection rows excluded from the canonical 100-episode dataset. "
            "Held out from ACT training, validation and normalization statistics."
        ),
        "scope_caveat": (
            "Expert-feasible pool. The 15 collection rows whose expert planner failed have "
            "no accepted initial state to replay and are deliberately absent, so this pool "
            "is not an unrestricted sample of every environment attempt."
        ),
        "manifest_version": canon["manifest_version"],
        "master_seed": canon["master_seed"],
        "source_collection_tree_sha256": canon["source_collection_tree_sha256"],
        "source_run_dir": canon["source_run_dir"],
        "canonical_manifest_sha256": canon["manifest_sha256"],
        "split_manifest_sha256": split["split_manifest_sha256"],
        "collection_manifest_sha256": coll["manifest_sha256"],
        "sensor_order_sha256": sensor_order_hash,
        "composition": {"total": len(rows), "hazard_present": n_present, "hazard_absent": n_absent},
        "no_overlap_proof": {
            "checked": sorted(overlap),
            "overlaps_found": overlap,
            "train_episodes": len(train_ids),
            "validation_episodes": len(val_ids),
            "canonical_selected": len(selected_ids),
            "pool_episodes": len(pool_ids),
            "clean": True,
        },
        "rows": rows,
    }
    pool["pool_manifest_sha256"] = canonical_hash(pool)

    # ---- frozen smoke manifest: 3 lowest-ranked present + 1 lowest absent --
    present = sorted((r for r in rows if r["hazard_present"]),
                     key=lambda r: r["predeclared_stratum_rank"])[:3]
    absent = sorted((r for r in rows if not r["hazard_present"]),
                    key=lambda r: r["predeclared_stratum_rank"])[:1]
    smoke_rows = present + absent
    if len(smoke_rows) != 4 or sum(r["hazard_present"] for r in smoke_rows) != 3:
        raise SystemExit("smoke manifest must be exactly 3 hazard-present + 1 hazard-absent")

    smoke = {
        "schema": "hybrid_obstacle_eval_smoke4_v1",
        "label": "smoke4_bounded_paired",
        "parent_pool_label": POOL_LABEL,
        "parent_pool_sha256": pool["pool_manifest_sha256"],
        "selection_rule": (
            "the three lowest-ranked hazard-present reserve rows and the lowest-ranked "
            "hazard-absent reserve row, by predeclared stratum rank; frozen before any rollout"
        ),
        "composition": {"total": 4, "hazard_present": 3, "hazard_absent": 1},
        "preflight_row_episode_id": present[0]["episode_id"],
        "rows": smoke_rows,
    }
    smoke["smoke_manifest_sha256"] = canonical_hash(smoke)

    args.out_pool.parent.mkdir(parents=True, exist_ok=True)
    args.out_pool.write_text(json.dumps(pool, indent=2, sort_keys=True) + "\n")
    args.out_smoke.parent.mkdir(parents=True, exist_ok=True)
    args.out_smoke.write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n")

    print(f"pool                : {n_present} hazard-present + {n_absent} hazard-absent = {len(rows)}")
    print(f"pool sha256         : {pool['pool_manifest_sha256']}")
    print("no-overlap proof    : clean (ids and source hashes vs train, validation, canonical)")
    print(f"smoke4 sha256       : {smoke['smoke_manifest_sha256']}")
    for r in smoke_rows:
        print(f"  cand {r['candidate_index']:3d}  {'present' if r['hazard_present'] else 'absent '}  "
              f"stratum_rank {r['predeclared_stratum_rank']:3d}  retry {r['accepted_retry_index']}  "
              f"{r['episode_id'][:16]}…")
    print(f"preflight row       : {smoke['preflight_row_episode_id'][:16]}…")
    print(f"wrote {args.out_pool}")
    print(f"wrote {args.out_smoke}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
