from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pact_collision_contract as contract


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_pact_confirmatory_schedule",
        ROOT / "scripts" / "build_pact_confirmatory_schedule.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


schedule = _load()


def test_arm_seed_orders_are_balanced_and_deterministic():
    first = schedule._condition_orders(160)
    second = schedule._condition_orders(160)
    assert first == second
    assert all(set(order) == set(schedule.CONDITIONS) for order in first)
    for position in range(6):
        counts = Counter(order[position] for order in first)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_detectable_effect_is_frozen_from_pilot_baseline():
    effect = schedule.detectable_increase(0.5, per_arm_instances=160)
    assert effect is not None
    assert 0.14 < effect < 0.18
    assert schedule.approximate_two_sample_power(
        0.5, 0.5 + effect, per_arm_instances=160
    ) >= 0.80


def test_v2_schedule_has_two_seed_repeats_for_every_instance(tmp_path):
    surface = tmp_path / "surface.ckpt"
    surface.write_bytes(b"frozen surface")
    surface_sha = schedule.sha256_file(surface)
    training = {
        "surface_encoder_sha256": surface_sha,
        "records": [
            {
                "arm": arm,
                "seed": seed,
                "checkpoint": str(tmp_path / f"{arm}_{seed}.ckpt"),
                "checkpoint_sha256": f"{arm}-{seed}",
                "dataset_stats": str(tmp_path / "stats.pkl"),
                "dataset_stats_sha256": "stats",
            }
            for arm in ("ACT", "PACT")
            for seed in schedule.CHECKPOINT_SEEDS
        ],
    }
    manifest = contract.build_manifest(
        source_hashes={"unit": "a" * 64},
        sensor_names=[f"sensor_{index}" for index in range(40)],
    )
    gate = {
        "decision": "PACT_ENVIRONMENT_ADEQUATE",
        "act": {
            "collision_free_task_success": 32,
            "scientific_outcomes": 64,
        },
    }
    document = schedule.build(manifest, training, surface, gate)
    assert document["instances"] == 160
    assert document["repeats_per_instance_per_arm"] == 2
    assert document["rollouts"] == 960
    first_instance = document["rows"][:6]
    assert {
        (row["arm"], row["checkpoint_seed"]) for row in first_instance
    } == set(schedule.CONDITIONS)
