#!/usr/bin/env python3
"""Freeze the remediation-v2 160-instance, two-seed PACT schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_collision_contract import load_manifest, rows_for_role, sha256_file

ARMS = ("ACT", "PACT", "PACT_ZERO")
CHECKPOINT_SEEDS = (3101, 3102)
CONDITIONS = tuple((arm, seed) for seed in CHECKPOINT_SEEDS for arm in ARMS)
SCHEDULE_SEED = 2026072904
WORKERS = 8


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _condition_orders(count: int) -> list[tuple[tuple[str, int], ...]]:
    """Seeded cyclic design; every condition is balanced at every position."""
    rng = np.random.default_rng(SCHEDULE_SEED)
    base = list(CONDITIONS)
    rng.shuffle(base)
    rotations = [tuple(base[offset:] + base[:offset]) for offset in range(len(base))]
    orders: list[tuple[tuple[str, int], ...]] = []
    for block_start in range(0, count, len(rotations)):
        block = list(rotations)
        rng.shuffle(block)
        orders.extend(block[: min(len(block), count - block_start)])
    return orders


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def approximate_two_sample_power(
    baseline: float,
    alternative: float,
    *,
    per_arm_instances: int,
) -> float:
    """Conservative unpaired normal approximation; repeats receive no credit."""
    if not 0.0 <= baseline < alternative <= 1.0:
        raise ValueError("requires 0 <= baseline < alternative <= 1")
    pooled = (baseline + alternative) / 2.0
    null_se = math.sqrt(2.0 * pooled * (1.0 - pooled) / per_arm_instances)
    alt_se = math.sqrt(
        (
            baseline * (1.0 - baseline)
            + alternative * (1.0 - alternative)
        )
        / per_arm_instances
    )
    delta = alternative - baseline
    critical = 1.959963984540054 * null_se
    return _normal_cdf((delta - critical) / alt_se) + _normal_cdf(
        (-delta - critical) / alt_se
    )


def detectable_increase(baseline: float, *, per_arm_instances: int) -> float | None:
    """Smallest 0.1 pp increase reaching 80% approximate power."""
    for alternative in np.arange(baseline + 0.001, 1.0, 0.001):
        if approximate_two_sample_power(
            baseline, float(alternative), per_arm_instances=per_arm_instances
        ) >= 0.80:
            return float(alternative - baseline)
    return None


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


def build(
    manifest: dict,
    training: dict,
    surface_encoder: Path,
    environment_gate: dict,
) -> dict:
    instances = rows_for_role(manifest, "confirmatory_eval")
    if len(instances) != 160:
        raise ValueError(f"expected 160 held-out instances, got {len(instances)}")
    if environment_gate.get("decision") != "PACT_ENVIRONMENT_ADEQUATE":
        raise ValueError("confirmatory schedule requires an adequate v2 environment gate")
    records = _training_records(training)
    surface_sha256 = sha256_file(surface_encoder)
    if surface_sha256 != training["surface_encoder_sha256"]:
        raise ValueError("surface encoder differs from training summary")
    orders = _condition_orders(len(instances))
    rows = []
    schedule_index = 0
    for instance, order in zip(instances, orders):
        for order_index, (arm, checkpoint_seed) in enumerate(order):
            trained_arm = "ACT" if arm == "ACT" else "PACT"
            record = records[(trained_arm, checkpoint_seed)]
            identity = {
                "schedule_schema": "pact_confirmatory_schedule_v2",
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
                "condition_order_index": order_index,
                "condition_order": [
                    {"arm": condition_arm, "checkpoint_seed": condition_seed}
                    for condition_arm, condition_seed in order
                ],
                "arm": arm,
                "checkpoint_seed": checkpoint_seed,
                "repeat_index": CHECKPOINT_SEEDS.index(checkpoint_seed),
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
    pilot_act = environment_gate["act"]
    pilot_baseline = float(
        pilot_act["collision_free_task_success"] / pilot_act["scientific_outcomes"]
    )
    mde = detectable_increase(pilot_baseline, per_arm_instances=len(instances))
    document = {
        "schema_version": "pact_confirmatory_schedule_v2",
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "training_summary_sha256": canonical_hash(training),
        "environment_gate_sha256": canonical_hash(environment_gate),
        "surface_encoder_sha256": surface_sha256,
        "instances": len(instances),
        "arms": list(ARMS),
        "checkpoint_seeds": list(CHECKPOINT_SEEDS),
        "repeats_per_instance_per_arm": len(CHECKPOINT_SEEDS),
        "rollouts": len(rows),
        "workers": WORKERS,
        "fresh_subprocess_per_rollout": True,
        "no_outcome_based_reruns": True,
        "detectable_effect_statement": (
            f"Using the frozen pilot ACT collision-free rate {pilot_baseline:.3f}, "
            f"160 independent instances per arm have approximately 80% power at "
            f"two-sided alpha=0.05 for a {mde:.3f} absolute increase under a "
            "conservative unpaired normal approximation. The second checkpoint "
            "seed per instance and pairing receive no power credit."
        ),
        "detectable_absolute_increase": mde,
        "pilot_act_collision_free_rate": pilot_baseline,
        "schedule_seed": SCHEDULE_SEED,
        "bootstrap_seed": 2026072902,
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
    parser.add_argument("--environment-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fresh = build(
        load_manifest(args.manifest),
        json.loads(args.training_summary.read_text()),
        args.surface_encoder,
        json.loads(args.environment_gate.read_text()),
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
