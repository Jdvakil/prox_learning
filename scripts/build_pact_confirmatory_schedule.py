#!/usr/bin/env python3
"""Freeze the 80-instance x 3-arm PACT confirmatory schedule."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_collision_contract import load_manifest, rows_for_role, sha256_file  # noqa: E402

ARMS = ("ACT", "PACT", "PACT_ZERO")
SCHEDULE_SEED = 2026072802
WORKERS = 8


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _arm_orders(count: int) -> list[tuple[str, ...]]:
    permutations = list(itertools.permutations(ARMS))
    rng = np.random.default_rng(SCHEDULE_SEED)
    orders = []
    full_blocks, remainder = divmod(count, len(permutations))
    for _ in range(full_blocks):
        block = list(permutations)
        rng.shuffle(block)
        orders.extend(block)
    if remainder:
        candidates = list(permutations)
        rng.shuffle(candidates)
        selected: list[tuple[str, ...]] = []
        for candidate in candidates:
            tentative = selected + [candidate]
            if all(
                max(Counter(order[position] for order in tentative).values())
                - min(
                    [
                        Counter(order[position] for order in tentative).get(arm, 0)
                        for arm in ARMS
                    ]
                )
                <= 1
                for position in range(3)
            ):
                selected.append(candidate)
                if len(selected) == remainder:
                    break
        if len(selected) != remainder:
            raise RuntimeError("could not balance residual arm orders")
        orders.extend(selected)
    return orders


def _training_records(training: dict) -> dict[tuple[str, int], dict]:
    records = {}
    for record in training["records"]:
        records[(record["arm"], int(record["seed"]))] = record
    expected = {
        (arm, seed)
        for arm in ("ACT", "PACT")
        for seed in (3101, 3102)
    }
    if set(records) != expected:
        raise ValueError(f"training records {set(records)} != {expected}")
    return records


def build(manifest: dict, training: dict, surface_encoder: Path) -> dict:
    instances = rows_for_role(manifest, "confirmatory_eval")
    if len(instances) != 80:
        raise ValueError(f"expected 80 held-out instances, got {len(instances)}")
    records = _training_records(training)
    surface_sha256 = sha256_file(surface_encoder)
    if surface_sha256 != training["surface_encoder_sha256"]:
        raise ValueError("surface encoder differs from training summary")
    orders = _arm_orders(len(instances))
    rows = []
    schedule_index = 0
    for instance, order in zip(instances, orders):
        # Matched initialization block: all three arms on an instance use the
        # same predeclared policy seed; PACT_ZERO aliases that PACT checkpoint.
        checkpoint_seed = 3101 if int(instance["role_index"]) % 2 == 0 else 3102
        for order_index, arm in enumerate(order):
            trained_arm = "ACT" if arm == "ACT" else "PACT"
            record = records[(trained_arm, checkpoint_seed)]
            identity = {
                "schedule_schema": "pact_confirmatory_schedule_v1",
                "instance_episode_id": instance["episode_id"],
                "arm": arm,
                "checkpoint_seed": checkpoint_seed,
            }
            rollout_id = canonical_hash(identity)
            row = {
                "schedule_index": schedule_index,
                "instance_role_index": int(instance["role_index"]),
                "instance_episode_id": instance["episode_id"],
                "instance_row_sha256": instance["row_sha256"],
                "intrusion_side": instance["intrusion_side"],
                "arm_order_index": order_index,
                "arm_order": list(order),
                "arm": arm,
                "checkpoint_seed": checkpoint_seed,
                "checkpoint_path": record["checkpoint"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "dataset_stats_path": record["dataset_stats"],
                "dataset_stats_sha256": record["dataset_stats_sha256"],
                "surface_encoder_path": (
                    str(surface_encoder) if arm in ("PACT", "PACT_ZERO") else None
                ),
                "surface_encoder_sha256": (
                    surface_sha256 if arm in ("PACT", "PACT_ZERO") else None
                ),
                "rollout_id": rollout_id,
                "output_relpath": (
                    f"rows/{schedule_index:03d}_{rollout_id[:16]}_{arm.lower()}"
                ),
            }
            row["schedule_row_sha256"] = canonical_hash(row)
            rows.append(row)
            schedule_index += 1
    document = {
        "schema_version": "pact_confirmatory_schedule_v1",
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "training_summary_sha256": canonical_hash(training),
        "surface_encoder_sha256": surface_sha256,
        "instances": 80,
        "arms": list(ARMS),
        "repeats_per_instance_per_arm": 1,
        "rollouts": len(rows),
        "workers": WORKERS,
        "fresh_subprocess_per_rollout": True,
        "no_outcome_based_reruns": True,
        "schedule_seed": SCHEDULE_SEED,
        "bootstrap_seed": 2026072801,
        "bootstrap_replicates": 20000,
        "primary_endpoint": (
            "task_success and hazard_bar contacts == 0 and "
            "other_environment contacts == 0"
        ),
        "rows": rows,
    }
    document["schedule_sha256"] = canonical_hash(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--surface-encoder", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fresh = build(
        load_manifest(args.manifest),
        json.loads(args.training_summary.read_text()),
        args.surface_encoder,
    )
    if args.check:
        if not args.output.exists() or json.loads(args.output.read_text()) != fresh:
            print("schedule differs from deterministic regeneration")
            return 1
        print(f"schedule OK {fresh['schedule_sha256']}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n")
    print(f"schedule_sha256={fresh['schedule_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
