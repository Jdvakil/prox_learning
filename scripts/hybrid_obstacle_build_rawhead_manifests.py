#!/usr/bin/env python3
"""Freeze the development4 and confirmatory41 manifests.

Handoff step 3. The four rows already used by the paired smoke become the
development set; every other successful reserve row becomes the untouched
confirmatory set.

  hybrid_obstacle_controller_development4_v1   3 hazard-present + 1 hazard-absent
  hybrid_obstacle_confirmatory41_v1           32 hazard-present + 9 hazard-absent

The 45-row reserve pool is the successful collection rows the canonical 100-episode
dataset excluded, so neither manifest can touch ACT training, validation, or the
normalization statistics. That is re-proved here rather than assumed.

No confirmatory row is executed by this task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV_LABEL = "hybrid_obstacle_controller_development4_v1"
CONF_LABEL = "hybrid_obstacle_confirmatory41_v1"

REPLAY_FIELDS = (
    "candidate_index", "episode_id", "manifest_row_sha256", "hazard_present",
    "hazard_stratum", "accepted_retry_index", "source_h5_sha256", "source_relpath",
    "initial_state_sha256", "task_state_sha256", "robot_initial_qpos",
    "object_initial_pose", "mocap_pos", "target_uid", "selected_object",
    "selected_grasp", "obstacle_theta", "observed_hazard_present",
    "scene_template_id", "scene_template_house_index", "sensor_order_sha256",
    "predeclared_stratum_rank", "reserve_rank_within_stratum", "master_seed",
    "max_retries", "seed_map",
)


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--smoke4", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--stats-manifest", required=True, type=Path)
    ap.add_argument("--out-dev", required=True, type=Path)
    ap.add_argument("--out-conf", required=True, type=Path)
    args = ap.parse_args()

    pool = json.loads(args.pool.read_text())
    smoke = json.loads(args.smoke4.read_text())
    split = json.loads(args.split.read_text())
    stats = json.loads(args.stats_manifest.read_text())

    pool_by_id = {r["episode_id"]: r for r in pool["rows"]}
    dev_ids = [r["episode_id"] for r in smoke["rows"]]
    if len(dev_ids) != 4 or len(set(dev_ids)) != 4:
        raise SystemExit(f"smoke4 must hold 4 unique rows, got {len(set(dev_ids))}")
    missing = [e for e in dev_ids if e not in pool_by_id]
    if missing:
        raise SystemExit(f"development rows absent from the reserve pool: {missing}")

    def project(row: dict) -> dict:
        return {k: row.get(k) for k in REPLAY_FIELDS}

    dev_rows = sorted((project(pool_by_id[e]) for e in dev_ids),
                      key=lambda r: (not r["hazard_present"], r["predeclared_stratum_rank"]))
    conf_rows = sorted((project(r) for r in pool["rows"] if r["episode_id"] not in set(dev_ids)),
                       key=lambda r: (not r["hazard_present"], r["predeclared_stratum_rank"]))

    dp = sum(1 for r in dev_rows if r["hazard_present"])
    cp = sum(1 for r in conf_rows if r["hazard_present"])
    if (dp, len(dev_rows) - dp) != (3, 1):
        raise SystemExit(f"development4 must be 3 present + 1 absent, got {dp}/{len(dev_rows)-dp}")
    if (cp, len(conf_rows) - cp) != (32, 9):
        raise SystemExit(f"confirmatory41 must be 32 present + 9 absent, got {cp}/{len(conf_rows)-cp}")

    # ---- no-overlap proof ----------------------------------------------------
    train_ids = {e["episode_id"] for e in split["episodes"] if e["split"] == "train"}
    val_ids = {e["episode_id"] for e in split["episodes"] if e["split"] == "validation"}
    train_h = {e["source_h5_sha256"] for e in split["episodes"] if e["split"] == "train"}
    val_h = {e["source_h5_sha256"] for e in split["episodes"] if e["split"] == "validation"}
    # normalization statistics contributors are exactly the training act indices
    stat_idx = set(stats["train_episode_indices"])
    train_act_idx = {e["act_episode_index"] for e in split["episodes"] if e["split"] == "train"}
    if stat_idx != train_act_idx:
        raise SystemExit("statistics contributors disagree with the training split")

    def overlap(rows: list[dict], name: str) -> dict:
        ids = {r["episode_id"] for r in rows}
        hs = {r["source_h5_sha256"] for r in rows}
        return {
            f"{name}_vs_train_ids": sorted(ids & train_ids),
            f"{name}_vs_validation_ids": sorted(ids & val_ids),
            f"{name}_vs_train_source_hashes": sorted(hs & train_h),
            f"{name}_vs_validation_source_hashes": sorted(hs & val_h),
        }

    ov = {**overlap(dev_rows, "development4"), **overlap(conf_rows, "confirmatory41")}
    ov["development4_vs_confirmatory41_ids"] = sorted(
        {r["episode_id"] for r in dev_rows} & {r["episode_id"] for r in conf_rows})
    if any(ov.values()):
        raise SystemExit(f"overlap detected: { {k: v for k, v in ov.items() if v} }")

    common = {
        "parent_pool_label": pool["label"],
        "parent_pool_sha256": pool["pool_manifest_sha256"],
        "collection_manifest_sha256": pool["collection_manifest_sha256"],
        "canonical_manifest_sha256": pool["canonical_manifest_sha256"],
        "split_manifest_sha256": pool["split_manifest_sha256"],
        "source_collection_tree_sha256": pool["source_collection_tree_sha256"],
        "sensor_order_sha256": pool["sensor_order_sha256"],
        "source_run_dir": pool["source_run_dir"],
        "no_overlap_proof": {
            "checked": sorted(ov),
            "overlaps_found": ov,
            "clean": True,
            "normalization_contributors_are_the_80_training_rows": True,
        },
    }

    dev = {
        "schema": "hybrid_obstacle_controller_development4_v1",
        "label": DEV_LABEL,
        "role": "DEVELOPMENT_ONLY",
        "role_note": ("These four rows were already executed by the paired smoke, so they are "
                      "development evidence. They must never enter the confirmatory analysis."),
        "provenance": f"identical row set to {args.smoke4.name} (smoke_manifest_sha256 "
                      f"{smoke['smoke_manifest_sha256']})",
        "smoke4_manifest_sha256": smoke["smoke_manifest_sha256"],
        "composition": {"total": 4, "hazard_present": 3, "hazard_absent": 1},
        "preflight_row_episode_id": smoke["preflight_row_episode_id"],
        "rows": dev_rows, **common,
    }
    dev["manifest_sha256"] = canonical_hash(dev)

    conf = {
        "schema": "hybrid_obstacle_confirmatory41_v1",
        "label": CONF_LABEL,
        "role": "CONFIRMATORY_UNTOUCHED",
        "role_note": ("Never executed by this task. Every successful reserve row except the four "
                      "development rows. The confirmatory unit is the manifest row; analysis must "
                      "cluster over rows, not over repeats."),
        "composition": {"total": 41, "hazard_present": 32, "hazard_absent": 9},
        "executed_in_this_task": False,
        "rows": conf_rows, **common,
    }
    conf["manifest_sha256"] = canonical_hash(conf)

    args.out_dev.parent.mkdir(parents=True, exist_ok=True)
    args.out_dev.write_text(json.dumps(dev, indent=2, sort_keys=True) + "\n")
    args.out_conf.write_text(json.dumps(conf, indent=2, sort_keys=True) + "\n")

    print(f"development4  : {dp} present + {len(dev_rows)-dp} absent = {len(dev_rows)}")
    print(f"                sha256 {dev['manifest_sha256']}")
    for r in dev_rows:
        print(f"                cand {r['candidate_index']:3d} "
              f"{'present' if r['hazard_present'] else 'absent '} "
              f"rank {r['predeclared_stratum_rank']:3d} retry {r['accepted_retry_index']} "
              f"{r['episode_id'][:14]}…")
    print(f"confirmatory41: {cp} present + {len(conf_rows)-cp} absent = {len(conf_rows)}")
    print(f"                sha256 {conf['manifest_sha256']}")
    print(f"                executed_in_this_task = {conf['executed_in_this_task']}")
    print("no-overlap    : clean (ids and source hashes vs train, validation, and each other)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
