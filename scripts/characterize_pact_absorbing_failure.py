#!/usr/bin/env python3
"""Post-hoc characterization of target disengagement during PACT contact tails."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("ACT", "PACT", "PACT_ZERO", "PACT_PERMUTED")
NON_OOD_ARMS = ("ACT", "PACT", "PACT_PERMUTED")
SEEDS = (3101, 3102, 3103)
EXPECTED_ROLLOUTS = 1200
EXPECTED_INSTANCES = 100
HIGH_CONTACT_THRESHOLD = 500
TARGET_THRESHOLDS = (1, 10, 50, 100)
OPERATIONAL_TARGET_THRESHOLD = 50
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2026080601
CLIP4_EPISODE_ID = (
    "e99dc657bfa703eac0d75566c733613ca0ffede3a4bbc394a35c350c753a4391"
)


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


def rate(count: int, total: int) -> float | None:
    return float(count) / float(total) if total else None


def percentile(values: Iterable[float], percent: float) -> float | None:
    data = [float(value) for value in values]
    return float(np.percentile(data, percent)) if data else None


def numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "mean": statistics.fmean(data) if data else None,
        "median": statistics.median(data) if data else None,
        "percentile_25": percentile(data, 25),
        "percentile_75": percentile(data, 75),
        "minimum": min(data) if data else None,
        "maximum": max(data) if data else None,
    }


def result_record(
    row: dict[str, Any], output_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = output_root / row["output_relpath"] / "result.json"
    result = json.loads(path.read_text())
    expected = {
        "status": "complete",
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "episode_id": row["instance_episode_id"],
        "arm": row["arm"],
        "checkpoint_seed": int(row["checkpoint_seed"]),
    }
    observed = {key: result.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"result identity mismatch at schedule row {row['schedule_index']}"
        )
    audit = result["contact_audit"]
    trajectory_path = result.get("trajectory_path")
    trajectory_exists = bool(
        trajectory_path is not None and Path(trajectory_path).is_file()
    )
    record = {
        "schedule_index": int(row["schedule_index"]),
        "episode_id": result["episode_id"],
        "instance_role_index": int(row["instance_role_index"]),
        "arm": result["arm"],
        "checkpoint_seed": int(result["checkpoint_seed"]),
        "intrusion_side": result["intrusion_side"],
        "task_success": bool(result["task_success"]),
        "grasp_target_frames": int(
            audit["frames_with_contact"]["grasp_target"]
        ),
        "hazard_frames": int(audit["frames_with_contact"]["hazard_bar"]),
        "first_grasp_target_contact_step": audit["first_contact_step"][
            "grasp_target"
        ],
        "first_hazard_contact_step": audit["first_contact_step"]["hazard_bar"],
        "trajectory_path_recorded": trajectory_path is not None,
        "trajectory_file_exists": trajectory_exists,
        "contact_frame_payload_retained": bool(
            audit["contact_frame_payload_retained"]
        ),
    }
    source = {
        "schedule_index": record["schedule_index"],
        "path": str(path.resolve()),
        "sha256": file_hash(path),
    }
    return record, source


def contact_pattern(record: dict[str, Any], target_threshold: int) -> str:
    target = record["grasp_target_frames"] >= target_threshold
    hazard = record["hazard_frames"] > 0
    if target and not hazard:
        return "target_engaged_never_hazard"
    if not target and hazard:
        return "hazard_engaged_target_not_engaged"
    if not target and not hazard:
        return "neither_engaged"
    target_step = record["first_grasp_target_contact_step"]
    hazard_step = record["first_hazard_contact_step"]
    if target_step is None or hazard_step is None:
        raise ValueError("engaged contact class lacks first-contact step")
    if hazard_step == target_step:
        raise ValueError("simultaneous target/hazard first contact is not predeclared")
    return (
        "hazard_first_then_target"
        if hazard_step < target_step
        else "target_first_then_hazard"
    )


PATTERN_ORDER = (
    "target_engaged_never_hazard",
    "hazard_first_then_target",
    "target_first_then_hazard",
    "hazard_engaged_target_not_engaged",
    "neither_engaged",
)


def pattern_summary(
    records: list[dict[str, Any]], target_threshold: int
) -> dict[str, Any]:
    patterns = Counter(contact_pattern(record, target_threshold) for record in records)
    successes = Counter(
        contact_pattern(record, target_threshold)
        for record in records
        if record["task_success"]
    )
    return {
        name: {
            "rollouts": patterns[name],
            "task_successes": successes[name],
            "task_success_fraction": rate(successes[name], patterns[name]),
        }
        for name in PATTERN_ORDER
    }


def low_target_tail_summary(
    records: list[dict[str, Any]], target_threshold: int
) -> dict[str, Any]:
    high = [
        record
        for record in records
        if record["hazard_frames"] > HIGH_CONTACT_THRESHOLD
    ]
    low = [
        record
        for record in high
        if record["grasp_target_frames"] < target_threshold
    ]
    return {
        "high_contact_rollouts": len(high),
        "low_target_engagement_rollouts": len(low),
        "low_target_engagement_fraction_given_high_contact": rate(
            len(low), len(high)
        ),
        "task_successes_in_low_target_group": sum(
            record["task_success"] for record in low
        ),
    }


def bootstrap_difference(
    instances: list[str],
    draws: np.ndarray,
    records: list[dict[str, Any]],
    left_filter: Callable[[dict[str, Any]], bool],
    right_filter: Callable[[dict[str, Any]], bool],
    numerator_filter: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    index = {episode_id: offset for offset, episode_id in enumerate(instances)}
    left_den = np.zeros(len(instances), dtype=np.int32)
    left_num = np.zeros(len(instances), dtype=np.int32)
    right_den = np.zeros(len(instances), dtype=np.int32)
    right_num = np.zeros(len(instances), dtype=np.int32)
    for record in records:
        offset = index[record["episode_id"]]
        if left_filter(record):
            left_den[offset] += 1
            left_num[offset] += int(numerator_filter(record))
        if right_filter(record):
            right_den[offset] += 1
            right_num[offset] += int(numerator_filter(record))

    left_rate = float(left_num.sum() / left_den.sum())
    right_rate = float(right_num.sum() / right_den.sum())
    left_den_boot = left_den[draws].sum(axis=1)
    right_den_boot = right_den[draws].sum(axis=1)
    valid = (left_den_boot > 0) & (right_den_boot > 0)
    differences = (
        left_num[draws].sum(axis=1)[valid] / left_den_boot[valid]
        - right_num[draws].sum(axis=1)[valid] / right_den_boot[valid]
    )
    return {
        "left_numerator": int(left_num.sum()),
        "left_denominator": int(left_den.sum()),
        "left_fraction": left_rate,
        "right_numerator": int(right_num.sum()),
        "right_denominator": int(right_den.sum()),
        "right_fraction": right_rate,
        "difference": left_rate - right_rate,
        "instance_cluster_bootstrap_ci_95": [
            float(np.percentile(differences, 2.5)),
            float(np.percentile(differences, 97.5)),
        ],
        "bootstrap_valid_replicates": int(valid.sum()),
    }


def high_contact_contrast(
    instances: list[str],
    draws: np.ndarray,
    records: list[dict[str, Any]],
    comparator: str,
    seed: int | None,
    target_threshold: int,
) -> dict[str, Any]:
    def selected_arm(arm: str) -> Callable[[dict[str, Any]], bool]:
        return lambda record: (
            record["arm"] == arm
            and (seed is None or record["checkpoint_seed"] == seed)
            and record["hazard_frames"] > HIGH_CONTACT_THRESHOLD
        )

    return bootstrap_difference(
        instances,
        draws,
        records,
        selected_arm("PACT"),
        selected_arm(comparator),
        lambda record: record["grasp_target_frames"] < target_threshold,
    )


def both_high_contrast(
    instances: list[str],
    draws: np.ndarray,
    records: list[dict[str, Any]],
    comparator: str,
    seed: int | None,
    target_threshold: int,
) -> dict[str, Any]:
    lookup = {
        (record["episode_id"], record["checkpoint_seed"], record["arm"]): record
        for record in records
    }
    both_high: set[tuple[str, int]] = set()
    for episode_id in instances:
        for checkpoint_seed in SEEDS:
            if seed is not None and checkpoint_seed != seed:
                continue
            pact = lookup[(episode_id, checkpoint_seed, "PACT")]
            other = lookup[(episode_id, checkpoint_seed, comparator)]
            if (
                pact["hazard_frames"] > HIGH_CONTACT_THRESHOLD
                and other["hazard_frames"] > HIGH_CONTACT_THRESHOLD
            ):
                both_high.add((episode_id, checkpoint_seed))

    return bootstrap_difference(
        instances,
        draws,
        records,
        lambda record: (
            record["arm"] == "PACT"
            and (record["episode_id"], record["checkpoint_seed"]) in both_high
        ),
        lambda record: (
            record["arm"] == comparator
            and (record["episode_id"], record["checkpoint_seed"]) in both_high
        ),
        lambda record: record["grasp_target_frames"] < target_threshold,
    )


def ordering_contrast(
    instances: list[str],
    draws: np.ndarray,
    records: list[dict[str, Any]],
    arms: tuple[str, ...],
    target_threshold: int,
) -> dict[str, Any]:
    def group(name: str) -> Callable[[dict[str, Any]], bool]:
        return lambda record: (
            record["arm"] in arms
            and contact_pattern(record, target_threshold) == name
        )

    return bootstrap_difference(
        instances,
        draws,
        records,
        group("hazard_first_then_target"),
        group("target_first_then_hazard"),
        lambda record: bool(record["task_success"]),
    )


def target_percentile(records: list[dict[str, Any]], value: int) -> dict[str, Any]:
    values = [record["grasp_target_frames"] for record in records]
    less = sum(item < value for item in values)
    equal = sum(item == value for item in values)
    return {
        "value": value,
        "population_n": len(values),
        "strictly_less": less,
        "equal": equal,
        "ascending_rank_min": less + 1,
        "ascending_rank_max": less + equal,
        "midrank_empirical_percentile": (less + 0.5 * equal) / len(values),
    }


def build(
    *,
    schedule_path: Path,
    final_decision_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    schedule = json.loads(schedule_path.read_text())
    final_decision = json.loads(final_decision_path.read_text())
    schedule_sha = validate_self_hash(schedule, "schedule_sha256", "schedule")
    decision_sha = validate_self_hash(
        final_decision, "final_decision_sha256", "final decision"
    )
    if (
        schedule.get("rollouts") != EXPECTED_ROLLOUTS
        or schedule.get("instance_count") != EXPECTED_INSTANCES
        or tuple(schedule.get("arms", [])) != ARMS
        or tuple(schedule.get("checkpoint_seeds", [])) != SEEDS
        or len(schedule.get("rows", [])) != EXPECTED_ROLLOUTS
    ):
        raise ValueError("frozen schedule design mismatch")
    if final_decision.get("decision") != "CONTACT_REDUCTION_WITH_TASK_BENEFIT":
        raise ValueError("awarded token changed")

    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for row in schedule["rows"]:
        record, source = result_record(row, output_root)
        records.append(record)
        sources.append(source)
    if len(records) != EXPECTED_ROLLOUTS:
        raise ValueError("result reconciliation failed")
    unique = {
        (record["episode_id"], record["arm"], record["checkpoint_seed"])
        for record in records
    }
    if len(unique) != EXPECTED_ROLLOUTS:
        raise ValueError("result cells are not unique")
    non_ood = [record for record in records if record["arm"] in NON_OOD_ARMS]
    if len(non_ood) != 900:
        raise ValueError("non-OOD rollout count is not 900")

    instances = [
        row["instance_episode_id"]
        for row in sorted(
            (
                row
                for row in schedule["rows"]
                if row["arm"] == "ACT" and row["checkpoint_seed"] == SEEDS[0]
            ),
            key=lambda row: row["instance_role_index"],
        )
    ]
    if len(instances) != EXPECTED_INSTANCES or len(set(instances)) != len(instances):
        raise ValueError("instance cluster inventory mismatch")
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0,
        len(instances),
        size=(BOOTSTRAP_REPLICATES, len(instances)),
        dtype=np.int16,
    )

    sensitivity: dict[str, Any] = {}
    for threshold in TARGET_THRESHOLDS:
        sensitivity[str(threshold)] = {
            "pooled_all_arms": pattern_summary(records, threshold),
            "pooled_non_ood": pattern_summary(non_ood, threshold),
            "by_arm": {
                arm: pattern_summary(
                    [record for record in records if record["arm"] == arm],
                    threshold,
                )
                for arm in ARMS
            },
            "high_contact_low_target_by_arm": {
                arm: low_target_tail_summary(
                    [record for record in non_ood if record["arm"] == arm],
                    threshold,
                )
                for arm in NON_OOD_ARMS
            },
        }

    by_seed = {
        str(seed): {
            arm: low_target_tail_summary(
                [
                    record
                    for record in non_ood
                    if record["arm"] == arm
                    and record["checkpoint_seed"] == seed
                ],
                OPERATIONAL_TARGET_THRESHOLD,
            )
            for arm in NON_OOD_ARMS
        }
        for seed in SEEDS
    }
    contrasts_by_seed = {
        str(seed): {
            f"PACT_minus_{comparator}": high_contact_contrast(
                instances,
                draws,
                non_ood,
                comparator,
                seed,
                OPERATIONAL_TARGET_THRESHOLD,
            )
            for comparator in ("PACT_PERMUTED", "ACT")
        }
        for seed in SEEDS
    }
    pooled_contrasts = {
        f"PACT_minus_{comparator}": high_contact_contrast(
            instances,
            draws,
            non_ood,
            comparator,
            None,
            OPERATIONAL_TARGET_THRESHOLD,
        )
        for comparator in ("PACT_PERMUTED", "ACT")
    }
    both_high = {
        f"PACT_minus_{comparator}": {
            "by_seed": {
                str(seed): both_high_contrast(
                    instances,
                    draws,
                    non_ood,
                    comparator,
                    seed,
                    OPERATIONAL_TARGET_THRESHOLD,
                )
                for seed in SEEDS
            },
            "pooled": both_high_contrast(
                instances,
                draws,
                non_ood,
                comparator,
                None,
                OPERATIONAL_TARGET_THRESHOLD,
            ),
        }
        for comparator in ("PACT_PERMUTED", "ACT")
    }

    strict_absorbing = [
        record
        for record in non_ood
        if contact_pattern(record, 1) == "hazard_engaged_target_not_engaged"
    ]
    absorbing_instances = {record["episode_id"] for record in strict_absorbing}
    absorbing_cells = {
        (record["episode_id"], record["checkpoint_seed"])
        for record in strict_absorbing
    }
    successful_same_instances = [
        record
        for record in non_ood
        if record["episode_id"] in absorbing_instances and record["task_success"]
    ]
    successful_same_cells = [
        record
        for record in successful_same_instances
        if (record["episode_id"], record["checkpoint_seed"]) in absorbing_cells
    ]

    ordering = {
        "target_threshold_frames": 1,
        "pooled_non_ood": ordering_contrast(
            instances, draws, non_ood, NON_OOD_ARMS, 1
        ),
        "by_arm": {
            arm: ordering_contrast(instances, draws, non_ood, (arm,), 1)
            for arm in NON_OOD_ARMS
        },
        "threshold_sensitivity_pooled_non_ood": {
            str(threshold): ordering_contrast(
                instances, draws, non_ood, NON_OOD_ARMS, threshold
            )
            for threshold in TARGET_THRESHOLDS
        },
        "interpretation": (
            "Hazard-first and target-first success rates are observational post-contact "
            "groups. Whole instances are resampled together, but ordering is not randomized."
        ),
    }

    clip4 = next(
        record
        for record in records
        if record["episode_id"] == CLIP4_EPISODE_ID
        and record["checkpoint_seed"] == 3103
        and record["arm"] == "PACT"
    )
    expected_clip4 = {
        "grasp_target_frames": 12,
        "hazard_frames": 17609,
        "first_grasp_target_contact_step": 151,
        "first_hazard_contact_step": 128,
    }
    if {key: clip4[key] for key in expected_clip4} != expected_clip4:
        raise ValueError("clip 4 frozen record changed")
    clip4_instance = sorted(
        [record for record in records if record["episode_id"] == CLIP4_EPISODE_ID],
        key=lambda record: (record["checkpoint_seed"], ARMS.index(record["arm"])),
    )
    pact_records = [record for record in records if record["arm"] == "PACT"]
    pact_high = [
        record
        for record in pact_records
        if record["hazard_frames"] > HIGH_CONTACT_THRESHOLD
    ]

    trajectory_records = [record for record in records if record["trajectory_file_exists"]]
    document: dict[str, Any] = {
        "schema_version": "pact_absorbing_failure_characterization_v1",
        "status": "post_hoc_exploratory_descriptive",
        "decision_bearing": False,
        "confirmatory_p_values_computed": False,
        "awarded_token": final_decision["decision"],
        "awarded_token_changed": False,
        "definitions": {
            "hazard_engaged": "hazard_bar frames_with_contact > 0",
            "strict_target_engaged": "grasp_target frames_with_contact >= 1",
            "operational_sustained_target_engaged": (
                "grasp_target frames_with_contact >= 50; fixed from the supplied scan, "
                "equivalent to about 100 ms cumulative audited contact at 2 ms/frame, "
                "not necessarily contiguous"
            ),
            "high_contact": "hazard_bar frames_with_contact > 500",
            "threshold_not_tuned": True,
        },
        "sources": {
            "schedule": {
                "path": str(schedule_path.resolve()),
                "file_sha256": file_hash(schedule_path),
                "schedule_sha256": schedule_sha,
            },
            "final_decision": {
                "path": str(final_decision_path.resolve()),
                "file_sha256": file_hash(final_decision_path),
                "final_decision_sha256": decision_sha,
            },
            "output_root": str(output_root.resolve()),
            "result_files_reparsed": len(records),
            "result_file_inventory_sha256": canonical_hash(sources),
            "rollout_records_sha256": canonical_hash(records),
        },
        "contact_pattern_sensitivity": sensitivity,
        "absorbing_state_primary": {
            "definition": "hazard engaged and zero grasp_target contact frames",
            "rollouts": len(strict_absorbing),
            "task_successes": sum(record["task_success"] for record in strict_absorbing),
            "task_success_fraction": rate(
                sum(record["task_success"] for record in strict_absorbing),
                len(strict_absorbing),
            ),
            "unique_instances": len(absorbing_instances),
            "unique_instance_seed_cells": len(absorbing_cells),
        },
        "high_contact_low_target_engagement": {
            "target_threshold_frames": OPERATIONAL_TARGET_THRESHOLD,
            "seeds_unpooled_first": by_seed,
            "pooled_by_arm": sensitivity[str(OPERATIONAL_TARGET_THRESHOLD)][
                "high_contact_low_target_by_arm"
            ],
            "contrasts_seeds_unpooled_first": contrasts_by_seed,
            "pooled_contrasts": pooled_contrasts,
            "matched_both_arms_high_subset": both_high,
            "bootstrap": {
                "method": "paired whole-instance cluster percentile bootstrap",
                "instances": len(instances),
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "all_arms_and_seeds_within_sampled_instance_move_together": True,
                "confirmatory": False,
            },
        },
        "where_it_diverges": {
            "strict_absorbing_first_hazard_contact_step": numeric_summary(
                record["first_hazard_contact_step"] for record in strict_absorbing
            ),
            "successful_rollouts_on_same_instances": {
                "rollouts": len(successful_same_instances),
                "unique_instances": len(
                    {record["episode_id"] for record in successful_same_instances}
                ),
                "with_hazard_contact": sum(
                    record["first_hazard_contact_step"] is not None
                    for record in successful_same_instances
                ),
                "without_hazard_contact": sum(
                    record["first_hazard_contact_step"] is None
                    for record in successful_same_instances
                ),
                "first_hazard_contact_step_when_available": numeric_summary(
                    record["first_hazard_contact_step"]
                    for record in successful_same_instances
                    if record["first_hazard_contact_step"] is not None
                ),
            },
            "successful_rollouts_on_same_instance_seed_cells": {
                "rollouts": len(successful_same_cells),
                "with_hazard_contact": sum(
                    record["first_hazard_contact_step"] is not None
                    for record in successful_same_cells
                ),
                "first_hazard_contact_step_when_available": numeric_summary(
                    record["first_hazard_contact_step"]
                    for record in successful_same_cells
                    if record["first_hazard_contact_step"] is not None
                ),
            },
            "distance_travelled_before_first_hazard": {
                "available": False,
                "reason": (
                    "Only 2/1200 trajectory files survive and neither is an absorbing-state "
                    "rollout. One is a successful control in a relevant instance-seed cell, "
                    "but its matched absorbing ACT trajectory is absent, so no paired distance "
                    "comparison is possible."
                ),
            },
            "target_neighbourhood_reached": {
                "available": False,
                "reason": (
                    "No target-distance, end-effector-distance, or neighbourhood-reach "
                    "summary was retained in result.json."
                ),
            },
            "trajectory_payload_inventory": {
                "surviving_trajectory_files": len(trajectory_records),
                "surviving_records": [
                    {
                        "schedule_index": record["schedule_index"],
                        "arm": record["arm"],
                        "episode_id": record["episode_id"],
                        "checkpoint_seed": record["checkpoint_seed"],
                    }
                    for record in trajectory_records
                ],
            },
        },
        "ordering_negative_result": ordering,
        "clip4_placement": {
            "record": clip4,
            "exact_values_verified": expected_clip4,
            "pact_target_engagement_all_rollouts_percentile": target_percentile(
                pact_records, clip4["grasp_target_frames"]
            ),
            "pact_target_engagement_high_contact_percentile": target_percentile(
                pact_high, clip4["grasp_target_frames"]
            ),
            "same_instance_records": clip4_instance,
            "same_instance_high_contact_counts": {
                "all_arms": sum(
                    record["hazard_frames"] > HIGH_CONTACT_THRESHOLD
                    for record in clip4_instance
                ),
                "all_arms_denominator": len(clip4_instance),
                "non_ood": sum(
                    record["arm"] in NON_OOD_ARMS
                    and record["hazard_frames"] > HIGH_CONTACT_THRESHOLD
                    for record in clip4_instance
                ),
                "non_ood_denominator": sum(
                    record["arm"] in NON_OOD_ARMS for record in clip4_instance
                ),
            },
            "same_instance_operational_low_target_high_contact_cells": [
                {
                    "arm": record["arm"],
                    "checkpoint_seed": record["checkpoint_seed"],
                    "grasp_target_frames": record["grasp_target_frames"],
                    "hazard_frames": record["hazard_frames"],
                    "task_success": record["task_success"],
                }
                for record in clip4_instance
                if record["hazard_frames"] > HIGH_CONTACT_THRESHOLD
                and record["grasp_target_frames"] < OPERATIONAL_TARGET_THRESHOLD
            ],
        },
        "mechanism_assessment": {
            "selected_statement": "data_cannot_fully_separate_on_task_effect_from_selection",
            "plain_statement": (
                "PACT's low-target fraction is much smaller than PACT_PERMUTED's, and the "
                "difference persists in the thin matched subset where both arms are already "
                "high-contact. That is evidence consistent with proximity keeping the policy "
                "on task during contact, not just preventing contact entry. But high-contact "
                "membership is itself changed by the modality, the grouping is post-treatment, "
                "and trajectory/target-distance instrumentation is absent; these data cannot "
                "fully separate a within-contact on-task mechanism from selection."
            ),
            "discovery_update_warranted": True,
        },
        "verification": {
            "all_arm_pattern_counts_sum": len(records),
            "non_ood_pattern_counts_sum": len(non_ood),
            "strict_absorbing_expected_0_of_67": (
                len(strict_absorbing) == 67
                and not any(record["task_success"] for record in strict_absorbing)
            ),
            "clip4_exact": True,
        },
    }
    document["absorbing_failure_characterization_sha256"] = canonical_hash(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--final-decision", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    document = build(
        schedule_path=args.schedule.resolve(),
        final_decision_path=args.final_decision.resolve(),
        output_root=args.output_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["absorbing_failure_characterization_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
