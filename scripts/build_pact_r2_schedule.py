#!/usr/bin/env python3
"""Freeze the fresh 160-instance, 960-row PACT confirmatory R2 schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_pact_confirmatory_schedule import detectable_increase
from pact_r2_contract import load_manifest, rows_for_role, sha256_file

ARMS = ("ACT", "PACT", "PACT_ZERO")
CHECKPOINT_SEEDS = (3101, 3102)
CONDITIONS = tuple((arm, seed) for seed in CHECKPOINT_SEEDS for arm in ARMS)
SCHEDULE_SEED = 2026073002
WORKERS = 8


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def condition_orders(count: int) -> list[tuple[tuple[str, int], ...]]:
    rng = np.random.default_rng(SCHEDULE_SEED)
    base = list(CONDITIONS)
    rng.shuffle(base)
    rotations = [tuple(base[offset:] + base[:offset]) for offset in range(6)]
    orders: list[tuple[tuple[str, int], ...]] = []
    for block_start in range(0, count, 6):
        block = list(rotations)
        rng.shuffle(block)
        orders.extend(block[: min(6, count - block_start)])
    return orders


def training_records(training: dict) -> dict[tuple[str, int], dict]:
    records = {
        (record["arm"], int(record["seed"])): record
        for record in training["records"]
    }
    expected = {
        (arm, seed) for arm in ("ACT", "PACT") for seed in CHECKPOINT_SEEDS
    }
    if set(records) != expected:
        raise ValueError("R2 training records differ from frozen four checkpoints")
    return records


def build(
    manifest: dict,
    training: dict,
    surface_encoder: Path,
    environment_gate: dict,
    preregistration: dict,
) -> dict:
    instances = rows_for_role(manifest, "confirmatory_eval")
    if len(instances) != 160:
        raise ValueError("R2 requires exactly 160 fresh instances")
    if environment_gate.get("decision") != "PACT_ENVIRONMENT_ADEQUATE":
        raise ValueError("R2 requires the carried adequate environment gate")
    prereg_payload = dict(preregistration)
    prereg_hash = prereg_payload.pop("preregistration_sha256")
    if canonical_hash(prereg_payload) != prereg_hash:
        raise ValueError("R2 preregistration self-hash mismatch")
    records = training_records(training)
    surface_sha = sha256_file(surface_encoder)
    if surface_sha != training["surface_encoder_sha256"]:
        raise ValueError("R2 surface encoder differs from frozen training input")

    rows = []
    for instance, order in zip(instances, condition_orders(len(instances))):
        for order_index, (arm, checkpoint_seed) in enumerate(order):
            trained_arm = "ACT" if arm == "ACT" else "PACT"
            record = records[(trained_arm, checkpoint_seed)]
            identity = {
                "schedule_schema": "pact_confirmatory_r2_schedule_v1",
                "instance_episode_id": instance["episode_id"],
                "arm": arm,
                "checkpoint_seed": checkpoint_seed,
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
                    surface_sha if arm in ("PACT", "PACT_ZERO") else None
                ),
                "rollout_id": rollout_id,
                "output_relpath": (
                    f"rows/{schedule_index:03d}_{rollout_id[:16]}_{arm.lower()}"
                ),
            }
            row["schedule_row_sha256"] = canonical_hash(row)
            rows.append(row)
    pilot = environment_gate["act"]
    baseline = pilot["collision_free_task_success"] / pilot["scientific_outcomes"]
    mde = detectable_increase(baseline, per_arm_instances=160)
    document = {
        "schema_version": "pact_confirmatory_r2_schedule_v1",
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "training_summary_sha256": canonical_hash(training),
        "environment_gate_sha256": canonical_hash(environment_gate),
        "r2_preregistration_sha256": prereg_hash,
        "r1_schedule_quarantined": True,
        "r1_endpoint_loaded": False,
        "surface_encoder_sha256": surface_sha,
        "instances": 160,
        "arms": list(ARMS),
        "checkpoint_seeds": list(CHECKPOINT_SEEDS),
        "repeats_per_instance_per_arm": 2,
        "rollouts": 960,
        "workers": WORKERS,
        "fresh_subprocess_per_rollout": True,
        "no_outcome_based_row_replacement": True,
        "indiscriminate_all_inflight_recovery": True,
        "detectable_effect_statement": (
            f"Using the frozen pilot ACT collision-free rate {baseline:.3f}, "
            f"160 independent instances per arm have approximately 80% power at "
            f"two-sided alpha=0.05 for a {mde:.3f} absolute increase under a "
            "conservative unpaired normal approximation. The second checkpoint "
            "seed per instance and pairing receive no power credit."
        ),
        "detectable_absolute_increase": mde,
        "pilot_act_collision_free_rate": baseline,
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
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = build(
        load_manifest(args.manifest),
        json.loads(args.training_summary.read_text()),
        args.surface_encoder,
        json.loads(args.environment_gate.read_text()),
        json.loads(args.preregistration.read_text()),
    )
    if args.check:
        if not args.output.exists() or json.loads(args.output.read_text()) != document:
            print("R2 schedule differs from deterministic regeneration")
            return 1
        print(f"R2 schedule OK {document['schedule_sha256']}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"schedule_sha256={document['schedule_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
