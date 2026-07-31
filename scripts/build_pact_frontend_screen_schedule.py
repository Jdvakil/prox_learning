#!/usr/bin/env python3
"""Freeze the balanced 40-instance, 120-row front-end screen schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_frontend_screen_contract import (
    load_manifest,
    rows_for_role,
    sha256_file,
)

ARMS = ("ACT", "PACT", "PACT_ZERO")
CHECKPOINT_SEED = 3101
SCHEDULE_SEED = 2026073102
BOOTSTRAP_SEED = 2026073103
BOOTSTRAP_REPLICATES = 20000
WORKERS = 8
ACT_CHECKPOINT = Path(
    "/root/pact_remediation_artifacts_v2/full/policies_v2/"
    "act_seed3101/policy_best.ckpt"
)
ACT_CHECKPOINT_SHA256 = (
    "a5ebbf3d5537315337e17e0f28951de068ce6960974d0f282b77fcfcca672eb1"
)
ACT_STATS = ACT_CHECKPOINT.parent / "dataset_stats.pkl"
ACT_STATS_SHA256 = (
    "1fff47c6d6e75fce68d953bfef5029ffbad5794d08854ea9d0f7dafadc7be6ec"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def validate_self_hash(
    document: dict[str, Any], key: str, label: str
) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != canonical_hash(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return observed


def arm_orders(count: int) -> list[tuple[str, ...]]:
    rng = np.random.default_rng(SCHEDULE_SEED)
    base = list(ARMS)
    rng.shuffle(base)
    rotations = [
        tuple(base[offset:] + base[:offset]) for offset in range(3)
    ]
    orders: list[tuple[str, ...]] = []
    for block_start in range(0, count, 3):
        block = list(rotations)
        rng.shuffle(block)
        orders.extend(block[: min(3, count - block_start)])
    return orders


def build(
    *,
    manifest: dict[str, Any],
    training: dict[str, Any],
    encoder_report: dict[str, Any],
    preregistration: dict[str, Any],
    dataset_amendment: dict[str, Any],
) -> dict[str, Any]:
    prereg_hash = validate_self_hash(
        preregistration,
        "preregistration_sha256",
        "preregistration",
    )
    amendment_hash = validate_self_hash(
        dataset_amendment, "amendment_sha256", "dataset amendment"
    )
    if preregistration["design"] != {
        "arms": list(ARMS),
        "checkpoint_seed": CHECKPOINT_SEED,
        "fresh_subprocess_per_rollout": True,
        "instances": 40,
        "repeats_per_instance_per_arm": 1,
        "rollouts": 120,
        "schedule_seed": SCHEDULE_SEED,
        "screen_not_confirmatory": True,
        "workers": WORKERS,
    }:
        raise ValueError("preregistration design changed")
    if training.get("arm") != "PACT" or training.get(
        "seed"
    ) != CHECKPOINT_SEED:
        raise ValueError("screen requires exactly one seed-3101 PACT")
    if training["encoder_sha256"] != encoder_report[
        "checkpoint_sha256"
    ]:
        raise ValueError("training and encoder report disagree")
    if training["encoder_quality_gate"]["passed"] is not True:
        raise ValueError("front-end quality gate did not pass")
    if sha256_file(ACT_CHECKPOINT) != ACT_CHECKPOINT_SHA256:
        raise ValueError("reused ACT checkpoint changed")
    if sha256_file(ACT_STATS) != ACT_STATS_SHA256:
        raise ValueError("reused ACT statistics changed")
    pact_checkpoint = Path(training["checkpoint"])
    pact_stats = Path(training["dataset_stats"])
    encoder_path = Path(training["encoder"])
    for path, expected in (
        (pact_checkpoint, training["checkpoint_sha256"]),
        (pact_stats, training["dataset_stats_sha256"]),
        (encoder_path, training["encoder_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen model input changed: {path}")
    instances = rows_for_role(manifest, "frontend_screen_eval")
    if len(instances) != 40:
        raise ValueError("screen requires exactly 40 fresh instances")
    rows = []
    for instance, order in zip(
        instances, arm_orders(len(instances))
    ):
        for order_index, arm in enumerate(order):
            if arm == "ACT":
                checkpoint = ACT_CHECKPOINT
                checkpoint_sha = ACT_CHECKPOINT_SHA256
                stats = ACT_STATS
                stats_sha = ACT_STATS_SHA256
                surface = None
                surface_sha = None
            else:
                checkpoint = pact_checkpoint
                checkpoint_sha = training["checkpoint_sha256"]
                stats = pact_stats
                stats_sha = training["dataset_stats_sha256"]
                surface = encoder_path
                surface_sha = training["encoder_sha256"]
            identity = {
                "schedule_schema": (
                    "pact_frontend_screen_schedule_v1"
                ),
                "instance_episode_id": instance["episode_id"],
                "arm": arm,
                "checkpoint_seed": CHECKPOINT_SEED,
            }
            rollout_id = canonical_hash(identity)
            schedule_index = len(rows)
            row = {
                "schedule_index": schedule_index,
                "instance_role_index": int(instance["role_index"]),
                "instance_episode_id": instance["episode_id"],
                "instance_row_sha256": instance["row_sha256"],
                "intrusion_side": instance["intrusion_side"],
                "condition_order_index": order_index,
                "condition_order": list(order),
                "arm": arm,
                "checkpoint_seed": CHECKPOINT_SEED,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "dataset_stats_path": str(stats),
                "dataset_stats_sha256": stats_sha,
                "surface_encoder_path": (
                    str(surface) if surface is not None else None
                ),
                "surface_encoder_sha256": surface_sha,
                "proximity_feature_dim": 0 if arm == "ACT" else 32,
                "rollout_id": rollout_id,
                "output_relpath": (
                    f"rows/{schedule_index:03d}_"
                    f"{rollout_id[:16]}_{arm.lower()}"
                ),
            }
            row["schedule_row_sha256"] = canonical_hash(row)
            rows.append(row)
    expected_cells = {arm: 40 for arm in ARMS}
    if Counter(row["arm"] for row in rows) != expected_cells:
        raise ValueError("screen arm cells are not balanced")
    position_balance = {
        str(position): dict(
            Counter(
                order[position] for order in arm_orders(len(instances))
            )
        )
        for position in range(3)
    }
    document: dict[str, Any] = {
        "schema_version": "pact_frontend_screen_schedule_v1",
        "screen_not_confirmatory": True,
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "training_summary_sha256": canonical_hash(training),
        "encoder_report_sha256": canonical_hash(encoder_report),
        "preregistration_sha256": prereg_hash,
        "dataset_hash_amendment_sha256": amendment_hash,
        "instances": 40,
        "arms": list(ARMS),
        "checkpoint_seeds": [CHECKPOINT_SEED],
        "repeats_per_instance_per_arm": 1,
        "rollouts": 120,
        "workers": WORKERS,
        "fresh_subprocess_per_rollout": True,
        "no_outcome_based_row_replacement": True,
        "indiscriminate_all_inflight_recovery": True,
        "condition_position_balance": position_balance,
        "detectable_effect_statement": (
            "This 40-pair screen is intended to resolve a large "
            "PACT-minus-PACT_ZERO gap of roughly 0.15; smaller effects may "
            "not be detected and ACT is a non-decision-bearing reference."
        ),
        "schedule_seed": SCHEDULE_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_endpoint": (
            "task_success and hazard_bar contacts == 0 and "
            "other_environment contacts == 0"
        ),
        "primary_contrast": "PACT_minus_PACT_ZERO",
        "secondary_contrast": "PACT_minus_ACT",
        "rows": rows,
    }
    document["schedule_sha256"] = canonical_hash(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--training-summary", required=True, type=Path
    )
    parser.add_argument("--encoder-report", required=True, type=Path)
    parser.add_argument(
        "--preregistration", required=True, type=Path
    )
    parser.add_argument(
        "--dataset-hash-amendment", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = build(
        manifest=load_manifest(args.manifest),
        training=json.loads(args.training_summary.read_text()),
        encoder_report=json.loads(args.encoder_report.read_text()),
        preregistration=json.loads(args.preregistration.read_text()),
        dataset_amendment=json.loads(
            args.dataset_hash_amendment.read_text()
        ),
    )
    if args.check:
        if (
            not args.output.exists()
            or json.loads(args.output.read_text()) != document
        ):
            print("screen schedule differs from regeneration")
            return 1
        print(f"screen schedule OK {document['schedule_sha256']}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    print(f"schedule_sha256={document['schedule_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
