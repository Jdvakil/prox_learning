#!/usr/bin/env python3
"""Freeze the balanced 40-instance, three-arm seed-3102 schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("ACT", "PACT", "PACT_PERMUTED")
CHECKPOINT_SEED = 3102
INSTANCES = 40
WORKERS = 8
SCHEDULE_SEED = 2026080101
BOOTSTRAP_SEED = 2026080102
BOOTSTRAP_REPLICATES = 20000
ACT_CHECKPOINT = Path(
    "/root/pact_remediation_artifacts_v2/full/policies_v2/act_seed3102/policy_best.ckpt"
)
ACT_CHECKPOINT_SHA256 = "e98d98bad87e2762cef37eb953d9ab55fcb65ed6355d2d8e9a881f38ef48c8d4"
ACT_STATS = ACT_CHECKPOINT.parent / "dataset_stats.pkl"
ACT_STATS_SHA256 = "1fff47c6d6e75fce68d953bfef5029ffbad5794d08854ea9d0f7dafadc7be6ec"
MANIFEST_SHA256 = "e047641ec007ec86b91c577ce45d5932dbcc48e8f9667a4e2e5ddffcaa4ff65c"
ENCODER_SHA256 = "6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206"
TOKEN_PLAN_SHA256 = "a6cc273b8a1facd4745e137f9aee39dd7f61b5569e0465278010383ea22f8643"
DATASET_TREE_SHA256 = "7a95581dff2907da1720f17425b67244fd20cc934a88a83cb9b66e2ee1d6ce97"
SPLIT_SHA256 = "7d25e88445cb4608238f71ddb0ea850ac78041f9d1a5dfdf252f16a27717a486"
TRAINING_RECIPE = {
    "backbone": "resnet18",
    "encoder_layers": 7,
    "decoder_layers": 7,
    "heads": 8,
    "hidden_dim": 512,
    "chunk": 100,
    "learning_rate": 1e-5,
    "batch": 8,
    "epochs": 2000,
    "kl_beta": 10,
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != canonical_hash(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return str(observed)


def arm_orders(count: int) -> list[tuple[str, ...]]:
    """Balanced Latin rotations, with the first smoke exercising permutation."""
    rng = np.random.default_rng(SCHEDULE_SEED)
    base = ["PACT_PERMUTED", "ACT", "PACT"]
    rotations = [tuple(base[offset:] + base[:offset]) for offset in range(3)]
    orders: list[tuple[str, ...]] = [rotations[0]]
    remaining = count - 1
    while remaining:
        block = list(rotations)
        rng.shuffle(block)
        take = min(3, remaining)
        orders.extend(block[:take])
        remaining -= take
    return orders


def _model_records(training: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if (
        training.get("schema_version") != "pact_seed_replication_policy_training_v1"
        or training.get("arm") != "PACT"
        or training.get("seed") != CHECKPOINT_SEED
        or training.get("encoder_sha256") != ENCODER_SHA256
        or training.get("policy_feature_dim") != 32
        or training.get("dataset_tree_sha256") != DATASET_TREE_SHA256
        or training.get("split_manifest_sha256") != SPLIT_SHA256
        or training.get("dataset_stats_sha256") != ACT_STATS_SHA256
        or training.get("recipe") != TRAINING_RECIPE
        or training.get("only_recipe_difference_from_seed_3101") != "seed=3102"
        or training.get("encoder_quality_gate", {}).get("passed") is not True
    ):
        raise ValueError("training summary is not PACT seed 3102")
    records = {
        "ACT": {
            "checkpoint_path": str(ACT_CHECKPOINT),
            "checkpoint_sha256": ACT_CHECKPOINT_SHA256,
            "dataset_stats_path": str(ACT_STATS),
            "dataset_stats_sha256": ACT_STATS_SHA256,
            "surface_encoder_path": None,
            "surface_encoder_sha256": None,
            "proximity_feature_dim": 0,
        }
    }
    pact = {
        "checkpoint_path": training["checkpoint"],
        "checkpoint_sha256": training["checkpoint_sha256"],
        "dataset_stats_path": training["dataset_stats"],
        "dataset_stats_sha256": training["dataset_stats_sha256"],
        "surface_encoder_path": training["encoder"],
        "surface_encoder_sha256": training["encoder_sha256"],
        "proximity_feature_dim": 32,
    }
    records["PACT"] = dict(pact)
    records["PACT_PERMUTED"] = dict(pact)
    for record in records.values():
        for path_key, hash_key in (
            ("checkpoint_path", "checkpoint_sha256"),
            ("dataset_stats_path", "dataset_stats_sha256"),
            ("surface_encoder_path", "surface_encoder_sha256"),
        ):
            raw_path = record[path_key]
            if raw_path is not None and file_hash(Path(raw_path)) != record[hash_key]:
                raise ValueError(f"frozen model changed: {raw_path}")
    return records


def _seed3101_references(
    *,
    instances: list[dict[str, Any]],
    screen_schedule: dict[str, Any],
    screen_output_root: Path,
    valid_schedule: dict[str, Any],
    valid_output_root: Path,
) -> list[dict[str, Any]]:
    sources: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}
    for row in screen_schedule["rows"]:
        if row["arm"] in ("ACT", "PACT"):
            sources[(row["instance_episode_id"], row["arm"])] = (
                row,
                screen_output_root,
            )
    for row in valid_schedule["rows"]:
        sources[(row["instance_episode_id"], "PACT_PERMUTED")] = (
            row,
            valid_output_root,
        )
    references: list[dict[str, Any]] = []
    for instance in instances:
        episode_id = instance["episode_id"]
        for arm in ARMS:
            row, root = sources[(episode_id, arm)]
            row_dir = root / row["output_relpath"]
            result = row_dir / "result.json"
            driver = row_dir / "driver_result.json"
            if not result.exists() or not driver.exists():
                raise ValueError(f"seed-3101 reference missing: {episode_id} {arm}")
            references.append(
                {
                    "checkpoint_seed": 3101,
                    "instance_episode_id": episode_id,
                    "arm": arm,
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "result_path": str(result.resolve()),
                    "result_sha256": file_hash(result),
                    "driver_path": str(driver.resolve()),
                    "driver_sha256": file_hash(driver),
                }
            )
    if len(references) != 120:
        raise ValueError("seed-3101 reference matrix is incomplete")
    return references


def build(
    *,
    manifest: dict[str, Any],
    training: dict[str, Any],
    token_plan: dict[str, Any],
    token_plan_path: Path,
    preregistration: dict[str, Any],
    screen_schedule: dict[str, Any],
    screen_output_root: Path,
    valid_schedule: dict[str, Any],
    valid_output_root: Path,
) -> dict[str, Any]:
    manifest_sha = validate_self_hash(manifest, "manifest_sha256", "manifest")
    prereg_sha = validate_self_hash(preregistration, "preregistration_sha256", "preregistration")
    token_sha = validate_self_hash(token_plan, "token_plan_sha256", "token plan")
    if manifest_sha != MANIFEST_SHA256:
        raise ValueError("the exact 40 screen instances changed")
    if preregistration["design"] != {
        "arms": list(ARMS),
        "checkpoint_seed": CHECKPOINT_SEED,
        "instances": INSTANCES,
        "repeats_per_instance_per_arm": 1,
        "rollouts": 120,
        "schedule_seed": SCHEDULE_SEED,
        "workers": WORKERS,
    }:
        raise ValueError("preregistered design changed")
    if (
        token_plan.get("schema_version") != "pact_permuted_token_plan_v2"
        or token_sha != TOKEN_PLAN_SHA256
        or token_plan.get("seed") != 2026073105
        or token_plan.get("rows") != INSTANCES
        or token_plan.get("max_control_steps") != 900
        or token_plan.get("token_shape") != [40, 32]
    ):
        raise ValueError("permutation scheme or 900-step token horizon changed")
    token_tensor = Path(token_plan["files"]["tokens"]["path"])
    if file_hash(token_tensor) != token_plan["files"]["tokens"]["sha256"]:
        raise ValueError("permutation tensor changed")
    instances = list(manifest["rows"])
    if len(instances) != INSTANCES:
        raise ValueError("seed replication requires exactly 40 instances")
    models = _model_records(training)
    rows: list[dict[str, Any]] = []
    orders = arm_orders(INSTANCES)
    for instance, order in zip(instances, orders):
        for position, arm in enumerate(order):
            identity = {
                "schedule_schema": "pact_seed_replication_schedule_v1",
                "instance_episode_id": instance["episode_id"],
                "arm": arm,
                "checkpoint_seed": CHECKPOINT_SEED,
            }
            rollout_id = canonical_hash(identity)
            index = len(rows)
            row = {
                "schedule_index": index,
                "instance_role_index": int(instance["role_index"]),
                "instance_episode_id": instance["episode_id"],
                "instance_row_sha256": instance["row_sha256"],
                "intrusion_side": instance["intrusion_side"],
                "condition_order_index": position,
                "condition_order": list(order),
                "arm": arm,
                "checkpoint_seed": CHECKPOINT_SEED,
                **models[arm],
                "token_plan_manifest_path": (
                    str(token_plan_path.resolve()) if arm == "PACT_PERMUTED" else None
                ),
                "token_plan_sha256": token_sha if arm == "PACT_PERMUTED" else None,
                "token_plan_row": (int(instance["role_index"]) if arm == "PACT_PERMUTED" else None),
                "max_control_steps": 900,
                "rollout_id": rollout_id,
                "output_relpath": (f"rows/{index:03d}_{rollout_id[:16]}_{arm.lower()}"),
            }
            row["schedule_row_sha256"] = canonical_hash(row)
            rows.append(row)
    if Counter(row["arm"] for row in rows) != {arm: INSTANCES for arm in ARMS}:
        raise ValueError("arms are not balanced")
    position_balance = {
        str(position): dict(Counter(order[position] for order in orders)) for position in range(3)
    }
    references = _seed3101_references(
        instances=instances,
        screen_schedule=screen_schedule,
        screen_output_root=screen_output_root,
        valid_schedule=valid_schedule,
        valid_output_root=valid_output_root,
    )
    document: dict[str, Any] = {
        "schema_version": "pact_seed_replication_schedule_v1",
        "manifest_sha256": manifest_sha,
        "training_summary_sha256": canonical_hash(training),
        "preregistration_sha256": prereg_sha,
        "token_plan_sha256": token_sha,
        "token_plan_file_sha256": file_hash(token_plan_path),
        "instances": INSTANCES,
        "arms": list(ARMS),
        "checkpoint_seeds": [CHECKPOINT_SEED],
        "repeats_per_instance_per_arm": 1,
        "rollouts": 120,
        "workers": WORKERS,
        "fresh_subprocess_per_rollout": True,
        "no_outcome_based_row_replacement": True,
        "indiscriminate_all_inflight_recovery": True,
        "condition_position_balance": position_balance,
        "schedule_seed": SCHEDULE_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_endpoint": (
            "task_success and hazard_bar contacts == 0 and other_environment contacts == 0"
        ),
        "primary_contrast": "PACT_minus_PACT_PERMUTED_seed3102",
        "decomposition_contrasts": [
            "PACT_minus_PACT_PERMUTED",
            "PACT_PERMUTED_minus_ACT",
            "PACT_minus_ACT",
        ],
        "decision_rule": preregistration["decision_rule"],
        "seed_3101_references": references,
        "rows": rows,
    }
    document["schedule_sha256"] = canonical_hash(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--token-plan", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--screen-schedule", required=True, type=Path)
    parser.add_argument("--screen-output-root", required=True, type=Path)
    parser.add_argument("--valid-schedule", required=True, type=Path)
    parser.add_argument("--valid-output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    document = build(
        manifest=json.loads(args.manifest.read_text()),
        training=json.loads(args.training_summary.read_text()),
        token_plan=json.loads(args.token_plan.read_text()),
        token_plan_path=args.token_plan.resolve(),
        preregistration=json.loads(args.preregistration.read_text()),
        screen_schedule=json.loads(args.screen_schedule.read_text()),
        screen_output_root=args.screen_output_root.resolve(),
        valid_schedule=json.loads(args.valid_schedule.read_text()),
        valid_output_root=args.valid_output_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["schedule_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
