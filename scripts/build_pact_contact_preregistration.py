#!/usr/bin/env python3
"""Freeze the contact-endpoint design and render its narrative preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARMS = ["ACT", "PACT", "PACT_ZERO", "PACT_PERMUTED"]
SEEDS = [3101, 3102, 3103]
TOKENS = [
    "CONTACT_REDUCTION_ESTABLISHED",
    "CONTACT_REDUCTION_WITH_TASK_BENEFIT",
    "CONTACT_REDUCTION_SUBSET_ONLY",
    "NO_CONTACT_REDUCTION",
    "CONTACT_INCREASE",
    "CONTACT_EXPERIMENT_INCOMPLETE",
]


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


def build(occlusion_path: Path, power_path: Path, analyzer_path: Path) -> dict[str, Any]:
    occlusion = json.loads(occlusion_path.read_text())
    power = json.loads(power_path.read_text())
    occlusion_sha = validate_self_hash(
        occlusion, "occlusion_subset_sha256", "occlusion partition"
    )
    power_sha = validate_self_hash(power, "power_sha256", "power calculation")
    if occlusion["partition"]["action"] != "drop_subset_analysis_degenerate":
        raise ValueError("occlusion viability result changed")
    if power["power"]["chosen_fresh_instances"] != 100:
        raise ValueError("power-based instance count changed")
    decision_rule = {
        "decision_bearing_contrast": "PACT_minus_PACT_PERMUTED",
        "decision_bearing_endpoint": "hazard_bar_contact_frames_per_rollout",
        "lower_contact_is_better": True,
        "tokens": {
            "CONTACT_REDUCTION_WITH_TASK_BENEFIT": (
                "pooled PACT-minus-PERMUTED contact-frame CI is strictly below zero; "
                "PACT-minus-PERMUTED contact-frame difference is negative in every seed; "
                "pooled PACT-minus-ACT collision-free task-success difference is positive; "
                "and PACT-minus-ACT collision-free task-success difference is positive in every seed"
            ),
            "CONTACT_REDUCTION_ESTABLISHED": (
                "pooled PACT-minus-PERMUTED contact-frame CI is strictly below zero and "
                "PACT-minus-PERMUTED contact-frame difference is negative in every seed, "
                "without satisfying the task-benefit clause"
            ),
            "CONTACT_REDUCTION_SUBSET_ONLY": (
                "reserved for reduction confined to a viable preregistered occlusion subset; "
                "unavailable because the geometry-only partition was degenerate and dropped before rollouts"
            ),
            "CONTACT_INCREASE": (
                "pooled PACT-minus-PERMUTED contact-frame CI is strictly above zero"
            ),
            "NO_CONTACT_REDUCTION": (
                "the pooled contact-frame CI includes zero, or a pooled reduction excludes zero "
                "but its sign is not negative in every seed"
            ),
            "CONTACT_EXPERIMENT_INCOMPLETE": "the fixed 1,200-row schedule does not reconcile",
        },
        "evaluation_order": [
            "incomplete",
            "contact_increase",
            "contact_reduction_with_seed_consistency",
            "task_benefit_enhancement",
            "no_contact_reduction_fallback",
        ],
        "pact_zero_decision_bearing": False,
        "pact_minus_act_decision_bearing_only_for_enhanced_token": True,
        "exhaustive": True,
    }
    document: dict[str, Any] = {
        "schema_version": "pact_contact_endpoint_preregistration_v1",
        "frozen_before_first_rollout": True,
        "rollout_outcomes_seen_before_freeze": False,
        "environment": "pact_collision_corridor_v1 unchanged",
        "encoder": "frozen validated 32-D surface-geometry encoder unchanged",
        "training": "existing seeds 3101/3102 plus independently initialized matched ACT/PACT seed 3103; identical 2000-epoch recipe",
        "design": {
            "arms": ARMS,
            "checkpoint_seeds": SEEDS,
            "instances": 100,
            "repeats_per_instance_per_arm_seed": 1,
            "rollouts": 1200,
            "schedule_seed": 2026080104,
            "workers": 8,
        },
        "populations": {
            "primary": "full fresh held-out instance distribution",
            "occlusion_rule": "active panel outside wrist frustum or occluded for at least 50% of pre-grasp steps",
            "occlusion_partition_result": "285 of 285 eligible recorded episodes classified vision-disadvantaged",
            "occlusion_partition_action": "drop_subset_analysis_degenerate",
            "subset_token_available": False,
            "threshold_tuned": False,
            "occlusion_subset_sha256": occlusion_sha,
            "occlusion_subset_file_sha256": file_hash(occlusion_path),
        },
        "co_primary_endpoints": {
            "collision_free_task_success": (
                "task_success AND zero hazard_bar contact entries AND zero other_environment contact entries"
            ),
            "hazard_bar_contact_frames_per_rollout": (
                "number of physics audit frames containing one or more hazard_bar contacts"
            ),
        },
        "diagnostic": (
            "hazard-bar contact-frame paired contrast restricted to seed-instance pairs in which both compared arms achieved manipulation success"
        ),
        "secondary_endpoints": [
            "hazard_bar_contact_frames",
            "hazard_bar_contact_entries",
            "hazard_bar_any_contact",
            "other_environment_contact_frames",
            "other_environment_contact_entries",
            "hazard_bar_maximum_penetration_depth_m",
            "other_environment_maximum_penetration_depth_m",
            "non_target_maximum_penetration_depth_m",
            "manipulation_success",
            "ordinary_task_success",
        ],
        "contrasts": [
            ["PACT", "PACT_PERMUTED", "modality_information", True],
            ["PACT_PERMUTED", "ACT", "architecture_training_seed", False],
            ["PACT", "ACT", "combined_policy_difference", False],
            ["PACT", "PACT_ZERO", "sensor_failure_robustness_ood", False],
            ["PACT_ZERO", "ACT", "failed_sensor_cost_ood", False],
        ],
        "analysis": {
            "method": "paired mean differences with deterministic whole-instance cluster bootstrap",
            "bootstrap_replicates": 20000,
            "bootstrap_seed": 2026080105,
            "cluster": "whole instance; all arms and all three policy seeds move together",
            "confidence_level": 0.95,
            "contact_medians_reported": True,
            "seed_specific_results_reported_before_pooled": True,
            "frozen_analysis_script": str(analyzer_path.resolve()),
            "frozen_analysis_script_sha256": file_hash(analyzer_path),
        },
        "power": {
            "power_sha256": power_sha,
            "power_file_sha256": file_hash(power_path),
            "alpha_two_sided": power["power"]["alpha_two_sided"],
            "target_power": power["power"]["target_power"],
            "chosen_fresh_instances": 100,
            "contact_frame_mde": power["power"]["contact_frame_mde_at_chosen_n"],
            "historical_absolute_contact_frame_difference": power["power"][
                "historical_absolute_contact_frame_difference"
            ],
            "instances_for_historical_contact_effect": power["power"][
                "instances_for_historical_contact_frame_difference"
            ],
            "instances_for_historical_binary_effect": power["power"][
                "instances_for_historical_any_contact_probability_difference"
            ],
            "prior_outcomes_used_for_design_only": True,
        },
        "storage": {
            "summary_only_contact_frame_payload": True,
            "preserve_full_rows": [0, 1199],
            "other_rows": "retain endpoint-complete result and SHA-256 inventory, then irreversibly delete trajectory/video payload bytes",
            "selection_outcome_blind": True,
            "required_by_available_storage": True,
        },
        "execution": {
            "fresh_subprocess_per_rollout": True,
            "unique_output_directory": True,
            "no_outcome_based_row_replacement": True,
            "fixed_workers": 8,
            "launch_smoke_and_shell_kill_detachment_proof_required": True,
            "smoke_requires_900_step_token_plan_and_endpoint_instrumentation": True,
            "first_20_minute_outcome_blind_throughput_required": True,
            "all_inflight_rows_recovered_as_one_cohort_after_external_termination": True,
        },
        "decision_rule": decision_rule,
        "allowed_tokens": TOKENS,
    }
    document["preregistration_sha256"] = canonical_hash(document)
    return document


def render(document: dict[str, Any]) -> str:
    power = document["power"]
    return f"""# PACT contact-endpoint preregistration

This document freezes the contact-endpoint experiment before its first rollout. The scene, frozen surface encoder, learned 32-D representation, existing seeds, and distribution-matched permutation ablation remain unchanged. Seed 3103 adds one independently initialized matched ACT/PACT pair.

The run has 100 fresh instances, three policy seeds, four arms, one repeat, and 1,200 total rollouts at a fixed eight workers. The arms are ACT, PACT, PACT_ZERO, and PACT_PERMUTED. PACT_ZERO is explicitly an out-of-distribution sensor-failure probe; it is never modality evidence. PACT_PERMUTED is the decision-bearing modality-information instrument.

## Frozen populations

The geometry-only 50% wrist-camera criterion classified 285/285 eligible recorded episodes as vision-disadvantaged. This is a degenerate partition, so subset analysis is dropped without changing the threshold. Only the full fresh-instance distribution is analyzed. Partition SHA-256: `{document['populations']['occlusion_subset_sha256']}`.

## Endpoints and power

The co-primary operational endpoint is task success with zero hazard-bar and zero other-environment contact entries. The co-primary magnitude endpoint is hazard-bar contact frames per rollout. Hazard frames conditioned on both compared arms achieving manipulation success are diagnostic. All declared secondary contact, penetration, and manipulation metrics remain reported.

The paired-normal design calculation used prior outcomes only for sizing. At n=100, the 80%-power, two-sided 5% MDE is {power['contact_frame_mde']:.3f} hazard frames. The historical absolute effect was {power['historical_absolute_contact_frame_difference']:.3f}; the approximation required {power['instances_for_historical_contact_effect']} instances for the count endpoint versus {power['instances_for_historical_binary_effect']} for binary any-contact. Thus the count endpoint saves only about nine instances, not a material reduction.

## Frozen analysis and decisions

Every paired difference uses 20,000 deterministic bootstrap replicates. Whole instances are clusters: all arms and all seeds move together. Seeds are shown separately before pooling, and medians accompany heavy-tailed contact-frame means.

`CONTACT_REDUCTION_ESTABLISHED` requires the pooled PACT-minus-PERMUTED hazard-frame 95% CI to be strictly below zero and a negative difference in every seed. `CONTACT_REDUCTION_WITH_TASK_BENEFIT` additionally requires positive PACT-minus-ACT collision-free task success pooled and in every seed. A CI strictly above zero yields `CONTACT_INCREASE`. A CI including zero, or a pooled reduction with inconsistent seed signs, yields `NO_CONTACT_REDUCTION`. The subset-only token is unavailable because the partition was dropped. An unreconciled schedule yields `CONTACT_EXPERIMENT_INCOMPLETE`.

Frozen analyzer SHA-256: `{document['analysis']['frozen_analysis_script_sha256']}`. Preregistration SHA-256: `{document['preregistration_sha256']}`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occlusion", required=True, type=Path)
    parser.add_argument("--power", required=True, type=Path)
    parser.add_argument("--analyzer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    document = build(args.occlusion, args.power, args.analyzer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    args.report.write_text(render(document))
    print(document["preregistration_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
