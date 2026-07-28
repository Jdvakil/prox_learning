#!/usr/bin/env python3
"""Build the fixed ACT train/validation split from a PACT conversion manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "hybrid_obstacle_canonical_split_v2"


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build(conversion: dict, mode: str) -> dict:
    episodes = sorted(
        conversion["episodes"], key=lambda episode: int(episode["act_episode_index"])
    )
    if mode == "pilot":
        if conversion.get("roles") != ["pilot_train"]:
            raise ValueError("pilot mode requires exactly the pilot_train conversion")
        train_count = int(len(episodes) * 0.8)
        if train_count < 1 or train_count == len(episodes):
            raise ValueError("pilot conversion is too small for a nonempty fixed split")
        labels = ["train"] * train_count + ["validation"] * (
            len(episodes) - train_count
        )
        split_rule = (
            "collision-free successful pilot_train rows in frozen role_index order; "
            "first floor(0.8*N) train, remainder validation"
        )
    else:
        if set(conversion.get("roles", [])) != {"full_train", "full_validation"}:
            raise ValueError(
                "full mode requires combined full_train and full_validation roles"
            )
        labels = [
            "train" if episode["role"] == "full_train" else "validation"
            for episode in episodes
        ]
        split_rule = (
            "manifest role full_train -> train and full_validation -> validation; "
            "within-role order is frozen role_index"
        )
    output_episodes = []
    rank = {"train": 0, "validation": 0}
    for episode, label in zip(episodes, labels):
        output_episodes.append(
            {
                "act_episode_index": int(episode["act_episode_index"]),
                "episode_id": episode["episode_id"],
                "candidate_index": int(episode["candidate_index"]),
                "hazard_present": True,
                "split": label,
                "split_rank": rank[label],
                "source_h5_sha256": episode["source_h5_sha256"],
            }
        )
        rank[label] += 1
    counts = {
        label: {
            "total": sum(entry["split"] == label for entry in output_episodes),
            "hazard_present": sum(entry["split"] == label for entry in output_episodes),
            "hazard_absent": 0,
        }
        for label in ("train", "validation")
    }
    dataset_tree_sha256 = conversion.get("converted_tree_semantic_sha256")
    dataset_tree_kind = "semantic"
    if not dataset_tree_sha256:
        # Token encoding changes the HDF5 files after the raw conversion. The
        # updated manifest deliberately drops the now-stale semantic tree hash
        # but carries a current byte-for-byte file tree hash.
        dataset_tree_sha256 = conversion["converted_tree_file_sha256"]
        dataset_tree_kind = "file"
    document = {
        "schema": SCHEMA,
        "experiment": f"pact_collision_{mode}_v1",
        "canonical_manifest_sha256": conversion["source_manifest_sha256"],
        "source_collection_tree_sha256": dataset_tree_sha256,
        "source_collection_tree_hash_kind": dataset_tree_kind,
        "split_rule": split_rule,
        "counts": counts,
        "episodes": output_episodes,
    }
    document["split_manifest_sha256"] = canonical_hash(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-manifest", required=True, type=Path)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fresh = build(json.loads(args.conversion_manifest.read_text()), args.mode)
    if args.check:
        if not args.output.exists() or json.loads(args.output.read_text()) != fresh:
            print("split does not match deterministic regeneration")
            return 1
        print(f"split OK {fresh['split_manifest_sha256']}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n")
    print(f"split_manifest_sha256={fresh['split_manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
