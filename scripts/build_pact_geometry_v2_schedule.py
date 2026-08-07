#!/usr/bin/env python3
"""Freeze the 900-row attempt-2 zero-shot policy schedule."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pact_geometry_generalization_v2_main_contract import (
    load_manifest,
    sha256_file,
    sha256_payload,
)


ARMS = ("ACT", "PACT", "PACT_PERMUTED")
SEEDS = (3101, 3102, 3103)
WORKERS = 8
MAX_CONTROL_STEPS = 900
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 2026080606
SCHEDULE_SEED = 2026080607


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != sha256_payload(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return str(observed)


def validate_models(registry: dict[str, Any]) -> None:
    validate_self_hash(registry, "policy_registry_sha256", "policy registry")
    cache: dict[str, str] = {}
    for seed in SEEDS:
        records = registry["seeds"][str(seed)]
        if records["PACT_PERMUTED"]["checkpoint_sha256"] != records["PACT"]["checkpoint_sha256"]:
            raise ValueError("PACT_PERMUTED must alias PACT weights")
        for arm in ARMS:
            record = records[arm]
            for path_key, hash_key in (
                ("checkpoint_path", "checkpoint_sha256"),
                ("dataset_stats_path", "dataset_stats_sha256"),
                ("surface_encoder_path", "surface_encoder_sha256"),
            ):
                raw_path = record.get(path_key)
                expected = record.get(hash_key)
                if raw_path is None:
                    continue
                if raw_path not in cache:
                    cache[raw_path] = sha256_file(raw_path)
                if cache[raw_path] != expected:
                    raise ValueError(f"frozen artifact changed: {seed} {arm} {raw_path}")


def condition_order(instance_index: int, condition_index: int) -> list[tuple[int, str]]:
    seed_order = list(SEEDS[instance_index % 3 :] + SEEDS[: instance_index % 3])
    order = []
    for seed in seed_order:
        offset = (instance_index + condition_index + SEEDS.index(seed)) % len(ARMS)
        arms = list(ARMS[offset:] + ARMS[:offset])
        order.extend((seed, arm) for arm in arms)
    return order


def build(
    *,
    manifest: dict[str, Any],
    expert_screen: dict[str, Any],
    registry: dict[str, Any],
    token_plan: dict[str, Any],
    token_plan_path: Path,
    analysis_script: Path,
) -> dict[str, Any]:
    validate_models(registry)
    token_sha = validate_self_hash(token_plan, "token_plan_sha256", "token plan")
    screen_sha = validate_self_hash(expert_screen, "expert_screen_sha256", "expert screen")
    if expert_screen.get("expert_screen_sha256") != manifest["phase0_bindings"]["expert_screen_sha256"]:
        raise ValueError("expert screen belongs to another main manifest")
    if expert_screen.get("continue_to_policy_evaluation") is not True:
        raise ValueError("expert screen did not authorize policy evaluation")
    surviving = list(expert_screen["surviving_condition_ids"])
    if surviving != list(manifest["conditions"]):
        raise ValueError("surviving conditions differ from frozen main manifest")
    rows = []
    for instance in manifest["rows"]:
        condition_index = int(instance["condition_index"])
        instance_index = int(instance["instance_index"])
        order = condition_order(instance_index, condition_index)
        for position, (seed, arm) in enumerate(order):
            model = registry["seeds"][str(seed)][arm]
            identity = {
                "schedule_schema": "pact_geometry_generalization_v2_schedule",
                "manifest_sha256": manifest["manifest_sha256"],
                "instance_episode_id": instance["episode_id"],
                "checkpoint_seed": seed,
                "arm": arm,
                "token_plan_sha256": token_sha if arm == "PACT_PERMUTED" else None,
            }
            rollout_id = sha256_payload(identity)
            row: dict[str, Any] = {
                "schedule_index": len(rows),
                "condition_id": instance["condition_id"],
                "condition_label": instance["condition_label"],
                "instance_index": instance_index,
                "instance_cluster_id": instance["instance_cluster_id"],
                "instance_episode_id": instance["episode_id"],
                "instance_row_sha256": instance["row_sha256"],
                "intrusion_side": instance["intrusion_side"],
                "realized_geometry": instance["realized_geometry"],
                "condition_order_index": position,
                "condition_order": [f"{item_seed}:{item_arm}" for item_seed, item_arm in order],
                "arm": arm,
                "checkpoint_seed": seed,
                **model,
                "token_plan_manifest_path": (
                    str(token_plan_path.resolve()) if arm == "PACT_PERMUTED" else None
                ),
                "token_plan_sha256": token_sha if arm == "PACT_PERMUTED" else None,
                "token_plan_row": condition_index * 25 + instance_index if arm == "PACT_PERMUTED" else None,
                "max_control_steps": MAX_CONTROL_STEPS,
                "rollout_id": rollout_id,
                "output_relpath": (
                    f"rows/{len(rows):04d}_{rollout_id[:16]}_{instance['condition_id'].lower()}_"
                    f"{arm.lower()}_s{seed}"
                ),
            }
            row["schedule_row_sha256"] = sha256_payload(row)
            rows.append(row)
    expected = Counter(
        (condition, seed, arm)
        for condition in surviving
        for seed in SEEDS
        for arm in ARMS
        for _ in range(25)
    )
    observed = Counter((row["condition_id"], row["checkpoint_seed"], row["arm"]) for row in rows)
    if observed != expected:
        raise ValueError("condition/seed/arm cells are not balanced")
    smoke_candidates = [
        row
        for row in rows
        if row["condition_id"] == "C0"
        and row["instance_index"] == 0
        and row["checkpoint_seed"] == 3101
        and row["arm"] == "PACT_PERMUTED"
    ]
    if len(smoke_candidates) != 1:
        raise ValueError("launch smoke does not resolve exactly once")
    smoke = smoke_candidates[0]
    rows.remove(smoke)
    rows.insert(0, smoke)
    for index, row in enumerate(rows):
        row["schedule_index"] = index
        row["output_relpath"] = (
            f"rows/{index:04d}_{row['rollout_id'][:16]}_{row['condition_id'].lower()}_"
            f"{row['arm'].lower()}_s{row['checkpoint_seed']}"
        )
        payload = dict(row)
        payload.pop("schedule_row_sha256")
        row["schedule_row_sha256"] = sha256_payload(payload)
    document: dict[str, Any] = {
        "schema_version": "pact_geometry_generalization_v2_schedule",
        "manifest_sha256": manifest["manifest_sha256"],
        "expert_screen_sha256": screen_sha,
        "policy_registry_sha256": registry["policy_registry_sha256"],
        "token_plan_sha256": token_sha,
        "token_plan_file_sha256": sha256_file(token_plan_path),
        "analysis_script_path": str(analysis_script.resolve()),
        "analysis_script_sha256": sha256_file(analysis_script),
        "surviving_condition_ids": surviving,
        "shifted_condition_ids": [item for item in surviving if item != "C0"],
        "instances": manifest["rows"],
        "instances_per_condition": 25,
        "arms": list(ARMS),
        "checkpoint_seeds": list(SEEDS),
        "rollouts": len(rows),
        "workers": WORKERS,
        "max_control_steps": MAX_CONTROL_STEPS,
        "fresh_subprocess_per_rollout": True,
        "no_outcome_based_row_replacement": True,
        "indiscriminate_all_inflight_recovery": True,
        "schedule_seed": SCHEDULE_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "original_reference": {
            "PACT_minus_PACT_PERMUTED_any_hazard_contact_rate": -0.093,
            "PACT_minus_PACT_PERMUTED_hazard_contact_frames": -1980.0,
        },
        "c0_reproduction_rule": {
            "requirements": [
                "PACT-minus-PERMUTED point estimate is negative for any hazard contact",
                "PACT-minus-PERMUTED point estimate is negative for hazard contact frames",
                "the original -0.093 and -1980 references fall within their corresponding C0 95% instance-cluster bootstrap intervals",
            ],
            "ci_exclusion_of_zero_not_required_at_25_instances": True,
        },
        "decision_rule": {
            "GEOMETRY_GENERALIZES": "C0 reproduces; both contact modality gaps are negative in every surviving shifted condition; and both pooled-shifted 95% CIs exclude zero below zero",
            "GEOMETRY_PARTIAL": "C0 reproduces and both contact modality gaps are negative in at least one but not every shifted condition",
            "GEOMETRY_DOES_NOT_GENERALIZE": "C0 reproduces but no shifted condition has both negative gaps, either pooled-shifted contact CI includes zero, or either pooled-shifted contact gap reverses",
            "GEOMETRY_TEST_INCONCLUSIVE": "C0 does not reproduce, fewer than two shifted conditions survive, or the schedule does not reconcile",
            "precedence": [
                "GEOMETRY_TEST_INCONCLUSIVE",
                "GEOMETRY_GENERALIZES",
                "GEOMETRY_PARTIAL",
                "GEOMETRY_DOES_NOT_GENERALIZE",
            ],
        },
        "rows": rows,
    }
    document["schedule_sha256"] = sha256_payload(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expert-screen", required=True, type=Path)
    parser.add_argument("--policy-registry", required=True, type=Path)
    parser.add_argument("--token-plan", required=True, type=Path)
    parser.add_argument("--analysis-script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    document = build(
        manifest=load_manifest(args.manifest),
        expert_screen=json.loads(args.expert_screen.read_text()),
        registry=json.loads(args.policy_registry.read_text()),
        token_plan=json.loads(args.token_plan.read_text()),
        token_plan_path=args.token_plan,
        analysis_script=args.analysis_script,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["schedule_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
