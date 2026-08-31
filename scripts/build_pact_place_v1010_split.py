#!/usr/bin/env python3
"""V10.10: freeze the deterministic 120/24 split, five train + one validation per cell.

Every cell contributes the same shape, so the split cannot itself introduce a
family, side or pose imbalance. Which episode is held out is decided only by
SHA-256 of (split seed, attempt_id) -- never by any trajectory property.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "submodules" / "act"))

from pact_place_v1010_contract import (  # noqa: E402
    CONTRACT_VERSION_V1010, QUOTA_PER_CELL, SPLIT_MASTER_SEED, TARGET_SUCCESSES,
    TRAIN_COUNT, TRAIN_PER_CELL, VALIDATION_COUNT, VALIDATION_PER_CELL,
    WORK_ROOT, cells, cell_key, write_immutable_text_create_only,
)

SPLIT_SCHEMA = "hybrid_obstacle_canonical_split_v2"
SPLIT_RULE = (
    "V10.10 per-cell split, master seed 2026101002. Each of the 24 cells holds "
    "exactly six accepted episodes; the one whose SHA-256 of (split seed, "
    "attempt_id) ranks first goes to validation and the other five to training. "
    "No trajectory property is consulted, so every family, side and pose is "
    "represented identically in both halves."
)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rank(attempt_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_MASTER_SEED}:{attempt_id}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path,
                        default=ROOT / WORK_ROOT / "source_manifest.json")
    parser.add_argument("--conversion-manifest", type=Path,
                        default=ROOT / WORK_ROOT / "conversion_manifest_encoded.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / WORK_ROOT / "split_manifest.json")
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text())
    conversion = json.loads(args.conversion_manifest.read_text())
    if not source.get("verified"):
        raise SystemExit("source manifest is not verified")
    rows = source["rows"]
    if len(rows) != TARGET_SUCCESSES:
        raise SystemExit(f"{len(rows)} rows, expected {TARGET_SUCCESSES}")
    by_id = {e["episode_id"]: e for e in conversion["episodes"]}
    if {r["attempt_id"] for r in rows} != set(by_id):
        raise SystemExit("conversion manifest does not cover exactly the source rows")

    by_cell: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_cell[row["cell"]].append(row)
    if len(by_cell) != 24:
        raise SystemExit(f"{len(by_cell)} cells present, expected 24")
    validation_ids: set[str] = set()
    for cell, members in by_cell.items():
        if len(members) != QUOTA_PER_CELL:
            raise SystemExit(f"{cell} holds {len(members)} rows, expected {QUOTA_PER_CELL}")
        members.sort(key=lambda r: rank(r["attempt_id"]))
        validation_ids.update(m["attempt_id"] for m in members[:VALIDATION_PER_CELL])

    episodes, ranks = [], {"train": 0, "validation": 0}
    for row in sorted(rows, key=lambda r: int(r["act_episode_index"])):
        label = "validation" if row["attempt_id"] in validation_ids else "train"
        episodes.append({
            "act_episode_index": int(row["act_episode_index"]),
            "episode_id": row["attempt_id"],
            "candidate_index": int(row["attempt_index"]),
            "hazard_present": True, "split": label,
            "split_rank": ranks[label],
            "source_h5_sha256": row["trajectory_h5_sha256"],
            "cell": row["cell"], "family_id": row["family_id"],
            "intrusion_side": row["intrusion_side"], "pose_id": row["pose_id"],
        })
        ranks[label] += 1

    train = [e for e in episodes if e["split"] == "train"]
    validation = [e for e in episodes if e["split"] == "validation"]
    if (len(train), len(validation)) != (TRAIN_COUNT, VALIDATION_COUNT):
        raise SystemExit(f"split is {len(train)}/{len(validation)}")

    def tally(entries, key):
        return dict(sorted(collections.Counter(e[key] for e in entries).items()))

    for name, entries, per_cell in (("train", train, TRAIN_PER_CELL),
                                    ("validation", validation, VALIDATION_PER_CELL)):
        counts = tally(entries, "cell")
        if len(counts) != 24 or set(counts.values()) != {per_cell}:
            raise SystemExit(f"{name} is not exactly {per_cell} per cell: {counts}")

    document: dict[str, Any] = {
        "schema": SPLIT_SCHEMA,
        "experiment": "pact_place_v1010_144_four_object",
        "contract_version": CONTRACT_VERSION_V1010,
        "split_master_seed": SPLIT_MASTER_SEED,
        "canonical_manifest_sha256": conversion["source_manifest_payload_sha256"],
        "source_collection_tree_sha256": conversion["converted_tree_file_sha256"],
        "source_collection_tree_hash_kind": "file",
        "conversion_manifest_payload_sha256": conversion["payload_sha256"],
        "split_rule": SPLIT_RULE,
        "counts": {
            "train": {"total": len(train), "hazard_present": len(train), "hazard_absent": 0},
            "validation": {"total": len(validation),
                           "hazard_present": len(validation), "hazard_absent": 0}},
        "stratification": {
            "cells_total": 24, "train_per_cell": TRAIN_PER_CELL,
            "validation_per_cell": VALIDATION_PER_CELL,
            "balanced_by_construction": True,
            "train_by_family": tally(train, "family_id"),
            "validation_by_family": tally(validation, "family_id"),
            "train_by_side": tally(train, "intrusion_side"),
            "validation_by_side": tally(validation, "intrusion_side"),
            "train_by_pose": tally(train, "pose_id"),
            "validation_by_pose": tally(validation, "pose_id")},
        "episodes": episodes,
    }
    document["split_manifest_sha256"] = canonical_hash(
        {k: v for k, v in document.items() if k != "split_manifest_sha256"})
    raw = write_immutable_text_create_only(
        args.out, json.dumps(document, indent=2, sort_keys=True) + "\n")

    from fixed_split_data import load_split_manifest  # noqa: PLC0415

    reloaded = load_split_manifest(str(args.out))
    if (len(reloaded["train"]), len(reloaded["val"])) != (TRAIN_COUNT, VALIDATION_COUNT):
        raise SystemExit("round trip through the training loader changed the split")
    print(json.dumps({
        "train": len(train), "validation": len(validation),
        "train_by_family": document["stratification"]["train_by_family"],
        "validation_by_family": document["stratification"]["validation_by_family"],
        "split_manifest_sha256": document["split_manifest_sha256"],
        "raw_file_sha256": raw}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
