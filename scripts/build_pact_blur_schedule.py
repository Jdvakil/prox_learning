#!/usr/bin/env python3
"""Freeze the calibrated 900-row inference-time RGB blur schedule."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pact_blur_sweep_contract import (
    BLUR_SIGMAS,
    load_manifest,
    sha256_file,
    sha256_payload,
)


ARMS = ("ACT", "PACT", "PACT_PERMUTED")
SEEDS = (3101, 3102, 3103)
WORKERS = 12
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 2026081002
SCHEDULE_SEED = 2026081003
COLLAPSE_FLOOR = 0.10


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != sha256_payload(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return str(observed)


def execution_order(instance_index: int) -> list[tuple[float, int, str]]:
    sigmas = tuple(BLUR_SIGMAS)
    sigma_order = sigmas[instance_index % len(sigmas) :] + sigmas[: instance_index % len(sigmas)]
    output = []
    for sigma_position, sigma in enumerate(sigma_order):
        seeds = SEEDS[(instance_index + sigma_position) % 3 :] + SEEDS[: (instance_index + sigma_position) % 3]
        for seed_position, seed in enumerate(seeds):
            offset = (instance_index + sigma_position + seed_position) % len(ARMS)
            arms = ARMS[offset:] + ARMS[:offset]
            output.extend((sigma, seed, arm) for arm in arms)
    return output


def build(
    *,
    manifest: dict[str, Any],
    registry: dict[str, Any],
    token_plan: dict[str, Any],
    token_plan_path: Path,
    v3_schedule: dict[str, Any],
    analysis_script: Path,
) -> dict[str, Any]:
    registry_sha = validate_self_hash(registry, "policy_registry_sha256", "policy registry")
    token_sha = validate_self_hash(token_plan, "token_plan_sha256", "token plan")
    validate_self_hash(v3_schedule, "schedule_sha256", "v3 schedule")
    source_token_rows = {
        int(index): int(value)
        for index, value in enumerate(v3_schedule["token_plan_row_map_by_instance_index"])
    }
    rows = []
    for instance in manifest["rows"]:
        order = execution_order(int(instance["blur_role_index"]))
        for position, (sigma, seed, arm) in enumerate(order):
            model = registry["seeds"][str(seed)][arm]
            identity = {
                "schedule_schema": "pact_blur_sweep_schedule_v1",
                "manifest_sha256": manifest["manifest_sha256"],
                "instance_episode_id": instance["episode_id"],
                "blur_sigma": sigma,
                "checkpoint_seed": seed,
                "arm": arm,
                "token_plan_sha256": token_sha if arm == "PACT_PERMUTED" else None,
            }
            rollout_id = sha256_payload(identity)
            row: dict[str, Any] = {
                "schedule_index": len(rows),
                "instance_index": instance["blur_role_index"],
                "instance_cluster_id": instance["instance_cluster_id"],
                "instance_episode_id": instance["episode_id"],
                "instance_row_sha256": instance["row_sha256"],
                "source_v3_instance_index": instance["source_v3_instance_index"],
                "source_v3_row_sha256": instance["source_v3_row_sha256"],
                "intrusion_side": instance["intrusion_side"],
                "realized_geometry": instance["realized_geometry"],
                "order_index": position,
                "blur_sigma": sigma,
                "arm": arm,
                "checkpoint_seed": seed,
                **model,
                "token_plan_manifest_path": (
                    str(token_plan_path.resolve()) if arm == "PACT_PERMUTED" else None
                ),
                "token_plan_sha256": token_sha if arm == "PACT_PERMUTED" else None,
                "token_plan_row": (
                    source_token_rows[int(instance["source_v3_instance_index"])]
                    if arm == "PACT_PERMUTED"
                    else None
                ),
                "max_control_steps": 900,
                "rollout_id": rollout_id,
                "output_relpath": (
                    f"rows/{len(rows):04d}_{rollout_id[:16]}_sigma_{str(sigma).replace('.', 'p')}_"
                    f"{arm.lower()}_s{seed}"
                ),
            }
            row["schedule_row_sha256"] = sha256_payload(row)
            rows.append(row)
    expected = Counter(
        (sigma, seed, arm)
        for sigma in BLUR_SIGMAS
        for seed in SEEDS
        for arm in ARMS
        for _ in range(25)
    )
    observed = Counter(
        (row["blur_sigma"], row["checkpoint_seed"], row["arm"])
        for row in rows
    )
    if observed != expected:
        raise ValueError("sigma/seed/arm cells are not balanced")
    smoke = next(
        row
        for row in rows
        if row["instance_index"] == 0
        and row["blur_sigma"] == 0.0
        and row["checkpoint_seed"] == 3101
        and row["arm"] == "PACT_PERMUTED"
    )
    rows.remove(smoke)
    rows.insert(0, smoke)
    for index, row in enumerate(rows):
        row["schedule_index"] = index
        row["output_relpath"] = (
            f"rows/{index:04d}_{row['rollout_id'][:16]}_sigma_"
            f"{str(row['blur_sigma']).replace('.', 'p')}_{row['arm'].lower()}_s{row['checkpoint_seed']}"
        )
        payload = dict(row)
        payload.pop("schedule_row_sha256")
        row["schedule_row_sha256"] = sha256_payload(payload)
    document: dict[str, Any] = {
        "schema_version": "pact_blur_sweep_schedule_v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "calibration_sha256": manifest["calibration_binding"]["calibration_sha256"],
        "policy_registry_sha256": registry_sha,
        "token_plan_sha256": token_sha,
        "token_plan_file_sha256": sha256_file(token_plan_path),
        "analysis_script_path": str(analysis_script.resolve()),
        "analysis_script_sha256": sha256_file(analysis_script),
        "blur_sigmas": BLUR_SIGMAS,
        "arms": list(ARMS),
        "checkpoint_seeds": list(SEEDS),
        "instances": manifest["rows"],
        "instances_shared_across_sigmas_arms_seeds": True,
        "instances_count": 25,
        "rollouts": 900,
        "workers": WORKERS,
        "fresh_subprocess_per_rollout": True,
        "no_outcome_based_row_replacement": True,
        "indiscriminate_all_inflight_recovery": True,
        "schedule_seed": SCHEDULE_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "collapse_floor_collision_free_success": COLLAPSE_FLOOR,
        "decision_rule": {
            "BLUR_ROBUSTNESS_ESTABLISHED": (
                "PACT-minus-ACT collision-free-success point estimate is positive at "
                "every sigma>0, strictly increases at each successive positive sigma, "
                "and its 95% instance-cluster bootstrap CI lower bound exceeds zero at "
                "the largest sigma where pooled ACT and PACT are each >=0.10"
            ),
            "BLUR_PARTIAL": (
                "not established, but at least one sigma>0 has PACT-minus-ACT "
                "collision-free-success CI lower bound above zero"
            ),
            "NO_BLUR_ROBUSTNESS": (
                "not collapsed and no sigma>0 has PACT-minus-ACT collision-free-success "
                "CI lower bound above zero"
            ),
            "BLUR_UNINFORMATIVE_COLLAPSE": (
                "every arm has pooled collision-free success <0.10 at every sigma>0"
            ),
            "BLUR_EXPERIMENT_INCOMPLETE": "the 900-row schedule does not reconcile",
            "token_priority": [
                "BLUR_EXPERIMENT_INCOMPLETE",
                "BLUR_UNINFORMATIVE_COLLAPSE",
                "BLUR_ROBUSTNESS_ESTABLISHED",
                "BLUR_PARTIAL",
                "NO_BLUR_ROBUSTNESS",
            ],
        },
        "analysis_contract": {
            "cluster": "instance; all sigmas, arms, and seeds move together",
            "bootstrap_replicates_minimum": 20000,
            "seeds_unpooled_before_pooling": True,
            "within_instance_sigma_slope": True,
            "co_primary": [
                "collision_free_task_success",
                "hazard_bar_contact_frames_per_rollout",
            ],
            "secondary": [
                "any_hazard_contact_rate",
                "contact_entries",
                "maximum_hazard_penetration_depth_m",
                "task_success",
            ],
            "contrast_order": [
                "PACT_minus_ACT",
                "PACT_minus_PACT_PERMUTED",
                "PACT_PERMUTED_minus_ACT",
            ],
        },
        "rows": rows,
    }
    document["schedule_sha256"] = sha256_payload(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--policy-registry", required=True, type=Path)
    parser.add_argument("--token-plan", required=True, type=Path)
    parser.add_argument("--v3-schedule", required=True, type=Path)
    parser.add_argument("--analysis-script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace blur schedule: {args.output}")
    manifest = load_manifest(args.manifest)
    registry = json.loads(args.policy_registry.read_text())
    token_plan = json.loads(args.token_plan.read_text())
    v3_schedule = json.loads(args.v3_schedule.read_text())
    document = build(
        manifest=manifest,
        registry=registry,
        token_plan=token_plan,
        token_plan_path=args.token_plan,
        v3_schedule=v3_schedule,
        analysis_script=args.analysis_script,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["schedule_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
