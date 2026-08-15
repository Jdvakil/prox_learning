#!/usr/bin/env python3
"""Freeze the balanced 450-row sighted/blind RGB schedule."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pact_blind_rgb_contract import load_manifest, sha256_file, sha256_payload


ARMS = ("ACT", "PACT", "PACT_PERMUTED")
SEEDS = (3101, 3102, 3103)
CONDITIONS = ("sighted", "blind")
WORKERS = 12
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 2026081502
SCHEDULE_SEED = 2026081503


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != sha256_payload(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return str(observed)


def execution_order(instance_index: int) -> list[tuple[str, int, str]]:
    conditions = CONDITIONS[instance_index % 2 :] + CONDITIONS[: instance_index % 2]
    output = []
    for condition_position, condition in enumerate(conditions):
        offset = (instance_index + condition_position) % len(SEEDS)
        seeds = SEEDS[offset:] + SEEDS[:offset]
        for seed_position, seed in enumerate(seeds):
            arm_offset = (instance_index + condition_position + seed_position) % len(ARMS)
            arms = ARMS[arm_offset:] + ARMS[:arm_offset]
            output.extend((condition, seed, arm) for arm in arms)
    return output


def build(
    *,
    manifest: dict[str, Any],
    registry: dict[str, Any],
    token_plan: dict[str, Any],
    token_plan_path: Path,
    blur_schedule: dict[str, Any],
    analysis_script: Path,
) -> dict[str, Any]:
    registry_sha = validate_self_hash(registry, "policy_registry_sha256", "policy registry")
    token_sha = validate_self_hash(token_plan, "token_plan_sha256", "token plan")
    validate_self_hash(blur_schedule, "schedule_sha256", "blur schedule")
    token_rows = {}
    for row in blur_schedule["rows"]:
        if row["arm"] == "PACT_PERMUTED":
            token_rows.setdefault(int(row["instance_index"]), int(row["token_plan_row"]))
            if token_rows[int(row["instance_index"])] != int(row["token_plan_row"]):
                raise ValueError("blur token-plan row changed within an instance")
    rows = []
    for instance in manifest["rows"]:
        index = int(instance["blind_role_index"])
        for position, (condition, seed, arm) in enumerate(execution_order(index)):
            model = registry["seeds"][str(seed)][arm]
            blind_rgb = condition == "blind"
            identity = {
                "schedule_schema": "pact_blind_rgb_schedule_v1",
                "manifest_sha256": manifest["manifest_sha256"],
                "instance_episode_id": instance["episode_id"],
                "vision_condition": condition,
                "blind_rgb": blind_rgb,
                "checkpoint_seed": seed,
                "arm": arm,
                "token_plan_sha256": token_sha if arm == "PACT_PERMUTED" else None,
            }
            rollout_id = sha256_payload(identity)
            row: dict[str, Any] = {
                "schedule_index": len(rows),
                "instance_index": index,
                "instance_cluster_id": instance["instance_cluster_id"],
                "instance_episode_id": instance["episode_id"],
                "instance_row_sha256": instance["row_sha256"],
                "intrusion_side": instance["intrusion_side"],
                "realized_geometry": instance["realized_geometry"],
                "order_index": position,
                "vision_condition": condition,
                "blind_rgb": blind_rgb,
                "blur_sigma": 0.0,
                "arm": arm,
                "checkpoint_seed": seed,
                **model,
                "token_plan_manifest_path": (
                    str(token_plan_path.resolve()) if arm == "PACT_PERMUTED" else None
                ),
                "token_plan_sha256": token_sha if arm == "PACT_PERMUTED" else None,
                "token_plan_row": token_rows[index] if arm == "PACT_PERMUTED" else None,
                "max_control_steps": 900,
                "rollout_id": rollout_id,
                "output_relpath": (
                    f"rows/{len(rows):04d}_{rollout_id[:16]}_{condition}_"
                    f"{arm.lower()}_s{seed}"
                ),
            }
            row["schedule_row_sha256"] = sha256_payload(row)
            rows.append(row)
    expected = Counter(
        (condition, seed, arm)
        for condition in CONDITIONS
        for seed in SEEDS
        for arm in ARMS
        for _ in range(25)
    )
    observed = Counter(
        (row["vision_condition"], row["checkpoint_seed"], row["arm"])
        for row in rows
    )
    if observed != expected:
        raise ValueError("condition/seed/arm cells are not balanced")
    smoke = next(
        row
        for row in rows
        if row["instance_index"] == 0
        and row["vision_condition"] == "blind"
        and row["checkpoint_seed"] == 3101
        and row["arm"] == "PACT_PERMUTED"
    )
    rows.remove(smoke)
    rows.insert(0, smoke)
    for index, row in enumerate(rows):
        row["schedule_index"] = index
        row["output_relpath"] = (
            f"rows/{index:04d}_{row['rollout_id'][:16]}_{row['vision_condition']}_"
            f"{row['arm'].lower()}_s{row['checkpoint_seed']}"
        )
        payload = dict(row)
        payload.pop("schedule_row_sha256")
        row["schedule_row_sha256"] = sha256_payload(payload)
    decision_rule = {
        "PROXIMITY_STANDALONE_CONTACT_BENEFIT": (
            "in blind rows, the pooled 95% instance-cluster bootstrap CI upper bound "
            "for PACT-minus-ACT hazard-bar contact frames is below zero and the CI "
            "upper bound for PACT-minus-PACT_PERMUTED is below zero"
        ),
        "PROXIMITY_STANDALONE_TASK_BENEFIT": (
            "the standalone-contact condition holds, the pooled PACT-minus-ACT "
            "collision-free-success CI lower bound exceeds zero, and blind PACT "
            "manipulation success is at least 5%"
        ),
        "NO_STANDALONE_BENEFIT": (
            "the schedule reconciles, the collapse rule does not apply, and either "
            "decision-bearing blind contact CI includes zero or reverses"
        ),
        "BLIND_UNINFORMATIVE_COLLAPSE": (
            "all blind arms are below 10% collision-free success and 5% manipulation "
            "success, and the standalone contact benefit is not established"
        ),
        "BLIND_EXPERIMENT_INCOMPLETE": "the 450-row schedule does not reconcile",
        "token_priority": [
            "BLIND_EXPERIMENT_INCOMPLETE",
            "PROXIMITY_STANDALONE_TASK_BENEFIT",
            "PROXIMITY_STANDALONE_CONTACT_BENEFIT",
            "BLIND_UNINFORMATIVE_COLLAPSE",
            "NO_STANDALONE_BENEFIT",
        ],
        "priority_clarification": (
            "collapse is uninformative only when no contact benefit is measurable; "
            "this resolves overlap with the predeclared expected safety-only result"
        ),
    }
    document: dict[str, Any] = {
        "schema_version": "pact_blind_rgb_schedule_v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "policy_registry_sha256": registry_sha,
        "token_plan_sha256": token_sha,
        "token_plan_file_sha256": sha256_file(token_plan_path),
        "analysis_script_path": str(analysis_script.resolve()),
        "analysis_script_sha256": sha256_file(analysis_script),
        "vision_conditions": list(CONDITIONS),
        "arms": list(ARMS),
        "checkpoint_seeds": list(SEEDS),
        "instances": manifest["rows"],
        "instances_shared_across_conditions_arms_seeds": True,
        "instances_count": 25,
        "rollouts": 450,
        "workers": WORKERS,
        "fresh_subprocess_per_rollout": True,
        "no_outcome_based_row_replacement": True,
        "indiscriminate_all_inflight_recovery": True,
        "schedule_seed": SCHEDULE_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "collapse_floor_collision_free_success": 0.10,
        "collapse_floor_manipulation_success": 0.05,
        "decision_rule": decision_rule,
        "predeclared_expected_outcome": manifest["predeclared_expected_outcome"],
        "analysis_contract": {
            "cluster": "instance; both conditions, all arms, and all seeds move together",
            "bootstrap_replicates_minimum": 20000,
            "seeds_unpooled_before_pooling": True,
            "co_primary": [
                "collision_free_task_success",
                "hazard_bar_contact_frames_per_rollout",
            ],
            "secondary": [
                "any_hazard_contact_rate",
                "contact_entries",
                "maximum_hazard_penetration_depth_m",
                "manipulation_success",
            ],
            "headline_contrast": "PACT_BLIND_minus_ACT_BLIND",
            "modality_instrument": "PACT_BLIND_minus_PACT_PERMUTED_BLIND",
            "paired_degradation": "blind_minus_sighted_per_arm",
        },
        "rows": rows,
    }
    document["schedule_sha256"] = sha256_payload(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--policy-registry", required=True, type=Path)
    parser.add_argument("--token-plan", required=True, type=Path)
    parser.add_argument("--blur-schedule", required=True, type=Path)
    parser.add_argument("--analysis-script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace blind-RGB schedule: {args.output}")
    document = build(
        manifest=load_manifest(args.manifest),
        registry=json.loads(args.policy_registry.read_text()),
        token_plan=json.loads(args.token_plan.read_text()),
        token_plan_path=args.token_plan,
        blur_schedule=json.loads(args.blur_schedule.read_text()),
        analysis_script=args.analysis_script,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["schedule_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
