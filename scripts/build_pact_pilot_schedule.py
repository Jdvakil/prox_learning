#!/usr/bin/env python3
"""Freeze the 24-row vision-only ACT environment-gate schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_collision_contract import load_manifest, rows_for_role  # noqa: E402


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build(manifest: dict, training: dict) -> dict:
    records = training["records"]
    if len(records) != 1 or records[0]["arm"] != "ACT" or records[0]["seed"] != 1101:
        raise ValueError("pilot schedule requires exactly ACT seed 1101")
    checkpoint = records[0]
    rows = []
    for index, instance in enumerate(rows_for_role(manifest, "pilot_eval")):
        identity = {
            "schedule_schema": "pact_pilot_act_schedule_v1",
            "instance_episode_id": instance["episode_id"],
            "arm": "ACT",
            "checkpoint_seed": 1101,
        }
        rollout_id = canonical_hash(identity)
        row = {
            "schedule_index": index,
            "instance_role_index": instance["role_index"],
            "instance_episode_id": instance["episode_id"],
            "instance_row_sha256": instance["row_sha256"],
            "intrusion_side": instance["intrusion_side"],
            "arm_order_index": 0,
            "arm_order": ["ACT"],
            "arm": "ACT",
            "checkpoint_seed": 1101,
            "checkpoint_path": checkpoint["checkpoint"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "dataset_stats_path": checkpoint["dataset_stats"],
            "dataset_stats_sha256": checkpoint["dataset_stats_sha256"],
            "surface_encoder_path": None,
            "surface_encoder_sha256": None,
            "rollout_id": rollout_id,
            "output_relpath": f"rows/{index:03d}_{rollout_id[:16]}_act",
        }
        row["schedule_row_sha256"] = canonical_hash(row)
        rows.append(row)
    document = {
        "schema_version": "pact_pilot_act_schedule_v1",
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "training_summary_sha256": canonical_hash(training),
        "instances": 24,
        "arms": ["ACT"],
        "repeats_per_instance_per_arm": 1,
        "rollouts": 24,
        "workers": 8,
        "fresh_subprocess_per_rollout": True,
        "no_outcome_based_reruns": True,
        "rows": rows,
    }
    document["schedule_sha256"] = canonical_hash(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    document = build(
        load_manifest(args.manifest),
        json.loads(args.training_summary.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"schedule_sha256={document['schedule_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
