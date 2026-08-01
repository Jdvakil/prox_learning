#!/usr/bin/env python3
"""Freeze the 100-instance, three-seed, four-arm contact schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ARMS = ("ACT", "PACT", "PACT_ZERO", "PACT_PERMUTED")
SEEDS = (3101, 3102, 3103)
INSTANCES = 100
ORIGINAL_WORKERS = 8
MAX_CONTROL_STEPS = 900
SCHEDULE_SEED = 2026080104
BOOTSTRAP_SEED = 2026080105
BOOTSTRAP_REPLICATES = 20000


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


def condition_orders(count: int = INSTANCES) -> list[tuple[tuple[int, str], ...]]:
    """Deterministic Latin rotations; smoke is seed-3101 PACT_PERMUTED."""
    arm_base = ["PACT_PERMUTED", "ACT", "PACT_ZERO", "PACT"]
    seed_base = list(SEEDS)
    orders = []
    for instance_index in range(count):
        seed_offset = instance_index % len(seed_base)
        seed_order = seed_base[seed_offset:] + seed_base[:seed_offset]
        conditions = []
        for seed_position, policy_seed in enumerate(seed_order):
            arm_offset = (instance_index + SEEDS.index(policy_seed)) % len(arm_base)
            arm_order = arm_base[arm_offset:] + arm_base[:arm_offset]
            conditions.extend((policy_seed, arm) for arm in arm_order)
        orders.append(tuple(conditions))
    return orders


def _validate_model_registry(registry: dict[str, Any]) -> None:
    if registry.get("schema_version") != "pact_contact_policy_registry_v1":
        raise ValueError("policy registry schema changed")
    if registry.get("checkpoint_seeds") != list(SEEDS):
        raise ValueError("policy registry seeds changed")
    for policy_seed in SEEDS:
        records = registry["seeds"][str(policy_seed)]
        if set(records) != set(ARMS):
            raise ValueError(f"policy registry arms changed for {policy_seed}")
        if (
            records["PACT_ZERO"]["checkpoint_sha256"] != records["PACT"]["checkpoint_sha256"]
            or records["PACT_PERMUTED"]["checkpoint_sha256"] != records["PACT"]["checkpoint_sha256"]
        ):
            raise ValueError("PACT ablations must alias PACT weights")
        for arm, record in records.items():
            for path_key, sha_key in (
                ("checkpoint_path", "checkpoint_sha256"),
                ("dataset_stats_path", "dataset_stats_sha256"),
                ("surface_encoder_path", "surface_encoder_sha256"),
            ):
                raw_path = record.get(path_key)
                expected = record.get(sha_key)
                if raw_path is not None and file_hash(Path(raw_path)) != expected:
                    raise ValueError(
                        f"frozen model artifact changed: {policy_seed} {arm} {raw_path}"
                    )
        if records["ACT"]["proximity_feature_dim"] != 0:
            raise ValueError("ACT must have no proximity tokens")
        if any(
            records[arm]["proximity_feature_dim"] != 32
            for arm in ("PACT", "PACT_ZERO", "PACT_PERMUTED")
        ):
            raise ValueError("PACT arms must use 32-D tokens")


def build(
    *,
    manifest: dict[str, Any],
    registry: dict[str, Any],
    token_plan: dict[str, Any],
    token_plan_path: Path,
    occlusion: dict[str, Any],
    power: dict[str, Any],
    preregistration: dict[str, Any],
    worker_amendment: dict[str, Any],
    worker_amendment_path: Path,
) -> dict[str, Any]:
    manifest_sha = validate_self_hash(manifest, "manifest_sha256", "manifest")
    registry_sha = validate_self_hash(registry, "policy_registry_sha256", "policy registry")
    token_sha = validate_self_hash(token_plan, "token_plan_sha256", "token plan")
    occlusion_sha = validate_self_hash(occlusion, "occlusion_subset_sha256", "occlusion partition")
    power_sha = validate_self_hash(power, "power_sha256", "power calculation")
    prereg_sha = validate_self_hash(preregistration, "preregistration_sha256", "preregistration")
    worker_amendment_sha = validate_self_hash(
        worker_amendment, "worker_amendment_sha256", "worker amendment"
    )
    amended_workers = int(worker_amendment["amendment"]["new_count"])
    if (
        worker_amendment.get("schema_version") != "pact_contact_worker_amendment_v1"
        or worker_amendment["amendment"].get("old_count") != ORIGINAL_WORKERS
        or worker_amendment["amendment"].get("only_worker_count_changed") is not True
        or worker_amendment["amendment"].get("rows_changed") != 0
        or worker_amendment["zero_results_proof"].get("result_file_count") != 0
        or worker_amendment["outcome_blinding"].get("no_outcome_had_been_observed") is not True
    ):
        raise ValueError("worker amendment contract changed")
    _validate_model_registry(registry)
    if (
        token_plan.get("schema_version") != "pact_contact_permuted_token_plan_v1"
        or token_plan.get("rows") != INSTANCES
        or token_plan.get("max_control_steps") != MAX_CONTROL_STEPS
        or token_plan.get("token_shape") != [40, 32]
    ):
        raise ValueError("contact permutation plan changed")
    token_tensor = Path(token_plan["files"]["tokens"]["path"])
    if file_hash(token_tensor) != token_plan["files"]["tokens"]["sha256"]:
        raise ValueError("contact permutation tensor changed")
    if occlusion["partition"]["action"] != "drop_subset_analysis_degenerate":
        raise ValueError("occlusion viability decision changed")
    if power["power"]["chosen_fresh_instances"] != INSTANCES:
        raise ValueError("power-based instance count changed")
    expected_design = {
        "arms": list(ARMS),
        "checkpoint_seeds": list(SEEDS),
        "instances": INSTANCES,
        "repeats_per_instance_per_arm_seed": 1,
        "rollouts": INSTANCES * len(SEEDS) * len(ARMS),
        "schedule_seed": SCHEDULE_SEED,
        "workers": ORIGINAL_WORKERS,
    }
    if preregistration["design"] != expected_design:
        raise ValueError("preregistered design changed")
    instances = list(manifest["rows"])
    if len(instances) != INSTANCES:
        raise ValueError("manifest must contain 100 fresh instances")
    rows = []
    orders = condition_orders()
    for instance, order in zip(instances, orders):
        for position, (policy_seed, arm) in enumerate(order):
            model = registry["seeds"][str(policy_seed)][arm]
            identity = {
                "schedule_schema": "pact_contact_endpoint_schedule_v1",
                "manifest_sha256": manifest_sha,
                "instance_episode_id": instance["episode_id"],
                "checkpoint_seed": policy_seed,
                "arm": arm,
                "token_plan_sha256": token_sha if arm == "PACT_PERMUTED" else None,
            }
            rollout_id = canonical_hash(identity)
            schedule_index = len(rows)
            row: dict[str, Any] = {
                "schedule_index": schedule_index,
                "instance_role_index": int(instance["role_index"]),
                "instance_episode_id": instance["episode_id"],
                "instance_row_sha256": instance["row_sha256"],
                "intrusion_side": instance["intrusion_side"],
                "condition_order_index": position,
                "condition_order": [f"{seed}:{name}" for seed, name in order],
                "arm": arm,
                "checkpoint_seed": policy_seed,
                **model,
                "token_plan_manifest_path": (
                    str(token_plan_path.resolve()) if arm == "PACT_PERMUTED" else None
                ),
                "token_plan_sha256": token_sha if arm == "PACT_PERMUTED" else None,
                "token_plan_row": (int(instance["role_index"]) if arm == "PACT_PERMUTED" else None),
                "max_control_steps": MAX_CONTROL_STEPS,
                "rollout_id": rollout_id,
                "output_relpath": (
                    f"rows/{schedule_index:04d}_{rollout_id[:16]}_{arm.lower()}_s{policy_seed}"
                ),
            }
            row["schedule_row_sha256"] = canonical_hash(row)
            rows.append(row)
    expected_count = Counter((row["checkpoint_seed"], row["arm"]) for row in rows)
    if expected_count != Counter({(seed, arm): INSTANCES for seed in SEEDS for arm in ARMS}):
        raise ValueError("seed-arm cells are not balanced")
    position_balance = {
        str(position): dict(
            sorted(
                Counter(f"{order[position][0]}:{order[position][1]}" for order in orders).items()
            )
        )
        for position in range(len(SEEDS) * len(ARMS))
    }
    document: dict[str, Any] = {
        "schema_version": "pact_contact_endpoint_schedule_v1",
        "manifest_sha256": manifest_sha,
        "policy_registry_sha256": registry_sha,
        "token_plan_sha256": token_sha,
        "token_plan_file_sha256": file_hash(token_plan_path),
        "occlusion_subset_sha256": occlusion_sha,
        "power_sha256": power_sha,
        "preregistration_sha256": prereg_sha,
        "worker_amendment_sha256": worker_amendment_sha,
        "worker_amendment_file_sha256": file_hash(worker_amendment_path),
        "instances": instances,
        "instance_count": INSTANCES,
        "arms": list(ARMS),
        "checkpoint_seeds": list(SEEDS),
        "repeats_per_instance_per_arm_seed": 1,
        "rollouts": len(rows),
        "workers": amended_workers,
        "fresh_subprocess_per_rollout": True,
        "no_outcome_based_row_replacement": True,
        "indiscriminate_all_inflight_recovery": True,
        "condition_position_balance": position_balance,
        "schedule_seed": SCHEDULE_SEED,
        "max_control_steps": MAX_CONTROL_STEPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "population": "full_distribution_only",
        "occlusion_subset_analysis": "dropped_pre_rollout_degenerate_285_of_285",
        "co_primary_endpoints": [
            "task_success and hazard_bar entries == 0 and other_environment entries == 0",
            "hazard_bar frames_with_contact per rollout",
        ],
        "decision_bearing_contrast": "PACT_minus_PACT_PERMUTED_hazard_bar_contact_frames",
        "pact_zero_label": "OOD sensor-failure probe; not modality evidence",
        "decision_rule": preregistration["decision_rule"],
        "rows": rows,
    }
    document["schedule_sha256"] = canonical_hash(document)
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--policy-registry", required=True, type=Path)
    parser.add_argument("--token-plan", required=True, type=Path)
    parser.add_argument("--occlusion-subset", required=True, type=Path)
    parser.add_argument("--power", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--worker-amendment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = build(
        manifest=json.loads(args.manifest.read_text()),
        registry=json.loads(args.policy_registry.read_text()),
        token_plan=json.loads(args.token_plan.read_text()),
        token_plan_path=args.token_plan,
        occlusion=json.loads(args.occlusion_subset.read_text()),
        power=json.loads(args.power.read_text()),
        preregistration=json.loads(args.preregistration.read_text()),
        worker_amendment=json.loads(args.worker_amendment.read_text()),
        worker_amendment_path=args.worker_amendment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["schedule_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
