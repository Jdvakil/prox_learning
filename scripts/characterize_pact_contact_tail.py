#!/usr/bin/env python3
"""Post-hoc descriptive characterization of the frozen PACT contact tail."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ARMS = ("ACT", "PACT", "PACT_ZERO", "PACT_PERMUTED")
SEEDS = (3101, 3102, 3103)
EXPECTED_ROLLOUTS = 1200
EXPECTED_INSTANCES = 100
HORIZON = 900
# Fixed from PACT_TAIL_CHARACTERIZATION_PLAN.md before inspecting row-level tails.
HIGH_CONTACT_THRESHOLD_FRAMES = 500
PERCENTILES = (50, 75, 90, 95, 99, 100)
CONCENTRATION_FRACTIONS = (0.01, 0.05, 0.10)


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


def percentile(values: Iterable[float], percent: float) -> float | None:
    """Linear-interpolated percentile, matching NumPy's default method."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(percent) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    return statistics.fmean(data) if data else None


def median(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    return statistics.median(data) if data else None


def rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def concentration(values: list[int]) -> dict[str, Any]:
    ordered = sorted((int(value) for value in values), reverse=True)
    total_frames = sum(ordered)
    output: dict[str, Any] = {"total_hazard_frames": total_frames}
    for fraction in CONCENTRATION_FRACTIONS:
        count = max(1, math.ceil(len(ordered) * fraction))
        frames = sum(ordered[:count])
        output[f"top_{int(fraction * 100)}_percent"] = {
            "rollout_count": count,
            "hazard_frames": frames,
            "share_of_arm_hazard_frames": rate(frames, total_frames),
        }
    return output


def set_overlap(left: set[Any], right: set[Any]) -> dict[str, Any]:
    intersection = left & right
    union = left | right
    return {
        "left_count": len(left),
        "right_count": len(right),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard": rate(len(intersection), len(union)),
        "fraction_of_left_also_right": rate(len(intersection), len(left)),
        "fraction_of_right_also_left": rate(len(intersection), len(right)),
    }


def result_record(row: dict[str, Any], output_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    path = output_root / row["output_relpath"] / "result.json"
    result = json.loads(path.read_text())
    if (
        result.get("status") != "complete"
        or result.get("rollout_id") != row["rollout_id"]
        or result.get("schedule_row_sha256") != row["schedule_row_sha256"]
        or result.get("episode_id") != row["instance_episode_id"]
        or result.get("arm") != row["arm"]
        or int(result.get("checkpoint_seed", -1)) != int(row["checkpoint_seed"])
    ):
        raise ValueError(f"result identity mismatch for schedule row {row['schedule_index']}")
    audit = result["contact_audit"]
    hazard_frames = int(audit["frames_with_contact"]["hazard_bar"])
    hazard_pair_samples = int(audit["contact_class_totals"]["hazard_bar"])
    first_contact = audit["first_contact_step"]["hazard_bar"]
    record = {
        "schedule_index": int(row["schedule_index"]),
        "episode_id": row["instance_episode_id"],
        "instance_role_index": int(row["instance_role_index"]),
        "arm": row["arm"],
        "checkpoint_seed": int(row["checkpoint_seed"]),
        "hazard_frames": hazard_frames,
        "hazard_pair_samples": hazard_pair_samples,
        "frames_to_pair_samples_ratio": (
            float(hazard_frames) / float(hazard_pair_samples)
            if hazard_pair_samples > 0
            else None
        ),
        "first_hazard_contact_step": int(first_contact) if first_contact is not None else None,
        "hazard_maximum_penetration_depth_m": float(
            audit["maximum_penetration_depth_m"]["hazard_bar"]
        ),
        "task_success": bool(result["task_success"]),
        "collision_free_task_success": bool(result["collision_free_task_success"]),
        "failure_taxonomy": str(result["failure_taxonomy"]),
        "contact_frame_payload_retained": bool(audit["contact_frame_payload_retained"]),
        "contact_frame_payload_count": len(audit.get("contact_frames", [])),
    }
    source = {
        "schedule_index": record["schedule_index"],
        "path": str(path.resolve()),
        "result_file_sha256": file_hash(path),
    }
    return record, source


def arm_characterization(records: list[dict[str, Any]]) -> dict[str, Any]:
    frames = [record["hazard_frames"] for record in records]
    nonzero = [value for value in frames if value > 0]
    high = [
        record
        for record in records
        if record["hazard_frames"] > HIGH_CONTACT_THRESHOLD_FRAMES
    ]
    first_steps = [
        record["first_hazard_contact_step"]
        for record in high
        if record["first_hazard_contact_step"] is not None
    ]
    ratios = [
        record["frames_to_pair_samples_ratio"]
        for record in high
        if record["frames_to_pair_samples_ratio"] is not None
    ]
    penetrations = [record["hazard_maximum_penetration_depth_m"] for record in high]
    high_frames = [record["hazard_frames"] for record in high]
    high_pair_samples = [record["hazard_pair_samples"] for record in high]
    return {
        "rollouts": len(records),
        "distribution": {
            "zero_hazard_frames": {
                "count": len(frames) - len(nonzero),
                "fraction": rate(len(frames) - len(nonzero), len(frames)),
            },
            "nonzero_hazard_frames": {
                "count": len(nonzero),
                "fraction": rate(len(nonzero), len(frames)),
                "percentiles_linear": {
                    str(percent): percentile(nonzero, percent) for percent in PERCENTILES
                },
            },
            "all_rollouts_mean_hazard_frames": mean(frames),
            "all_rollouts_median_hazard_frames": median(frames),
        },
        "concentration": concentration(frames),
        "high_contact_regime": {
            "definition": f"hazard_frames > {HIGH_CONTACT_THRESHOLD_FRAMES}",
            "entry_count": len(high),
            "entry_fraction": rate(len(high), len(records)),
            "share_of_all_hazard_frames": rate(sum(high_frames), sum(frames)),
            "hazard_frames_given_entry": {
                "mean": mean(high_frames),
                "median": median(high_frames),
                "percentile_75": percentile(high_frames, 75),
                "percentile_90": percentile(high_frames, 90),
                "maximum": max(high_frames) if high_frames else None,
            },
            "hazard_pair_samples_given_entry": {
                "mean": mean(high_pair_samples),
                "median": median(high_pair_samples),
            },
            "frames_to_pair_samples_ratio": {
                "mean": mean(ratios),
                "median": median(ratios),
                "percentile_90": percentile(ratios, 90),
                "interpretation_limit": (
                    "Pair totals count simultaneous contact-pair samples, not contact-entry "
                    "transitions; this ratio cannot distinguish a contiguous run from repeated brushing."
                ),
            },
            "first_hazard_contact_step": {
                "available_count": len(first_steps),
                "mean": mean(first_steps),
                "median": median(first_steps),
                "percentile_25": percentile(first_steps, 25),
                "percentile_75": percentile(first_steps, 75),
                "minimum": min(first_steps) if first_steps else None,
                "maximum": max(first_steps) if first_steps else None,
                "median_fraction_of_900_step_horizon": (
                    median(first_steps) / HORIZON if first_steps else None
                ),
            },
            "maximum_penetration_depth_m": {
                "mean": mean(penetrations),
                "median": median(penetrations),
                "maximum": max(penetrations) if penetrations else None,
            },
            "task_success": {
                "count": sum(record["task_success"] for record in high),
                "fraction": rate(sum(record["task_success"] for record in high), len(high)),
            },
            "collision_free_task_success": {
                "count": sum(record["collision_free_task_success"] for record in high),
                "fraction": rate(
                    sum(record["collision_free_task_success"] for record in high), len(high)
                ),
            },
            "failure_taxonomy": dict(
                sorted(Counter(record["failure_taxonomy"] for record in high).items())
            ),
        },
    }


def overlap_characterization(records: list[dict[str, Any]]) -> dict[str, Any]:
    high = [
        record
        for record in records
        if record["hazard_frames"] > HIGH_CONTACT_THRESHOLD_FRAMES
    ]
    matched_sets = {
        arm: {
            (record["episode_id"], record["checkpoint_seed"])
            for record in high
            if record["arm"] == arm
        }
        for arm in ARMS
    }
    instance_sets = {
        arm: {record["episode_id"] for record in high if record["arm"] == arm}
        for arm in ARMS
    }
    pairs = (
        ("PACT", "ACT"),
        ("PACT", "PACT_PERMUTED"),
        ("ACT", "PACT_PERMUTED"),
        ("PACT", "PACT_ZERO"),
    )
    non_ood_arms = ("ACT", "PACT", "PACT_PERMUTED")
    multiplicity = Counter(
        sum(episode_id in instance_sets[arm] for arm in non_ood_arms)
        for episode_id in {record["episode_id"] for record in records}
    )
    return {
        "definition": f"hazard_frames > {HIGH_CONTACT_THRESHOLD_FRAMES}",
        "matched_instance_seed_overlap": {
            f"{left}_vs_{right}": set_overlap(matched_sets[left], matched_sets[right])
            for left, right in pairs
        },
        "instance_overlap_any_seed": {
            f"{left}_vs_{right}": set_overlap(instance_sets[left], instance_sets[right])
            for left, right in pairs
        },
        "non_ood_instance_tail_arm_multiplicity": {
            str(count): multiplicity.get(count, 0) for count in range(4)
        },
        "instances_high_in_all_non_ood_arms": len(
            set.intersection(*(instance_sets[arm] for arm in non_ood_arms))
        ),
        "instances_high_in_any_non_ood_arm": len(
            set.union(*(instance_sets[arm] for arm in non_ood_arms))
        ),
    }


def paired_instance_characterization(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals: defaultdict[tuple[str, str], int] = defaultdict(int)
    role_indices: dict[str, int] = {}
    for record in records:
        totals[(record["episode_id"], record["arm"])] += record["hazard_frames"]
        role_indices[record["episode_id"]] = record["instance_role_index"]
    comparisons: dict[str, Any] = {}
    for comparator in ("ACT", "PACT_PERMUTED"):
        rows = []
        for episode_id in sorted(role_indices, key=lambda item: role_indices[item]):
            pact = totals[(episode_id, "PACT")]
            other = totals[(episode_id, comparator)]
            rows.append(
                {
                    "instance_role_index": role_indices[episode_id],
                    "episode_id": episode_id,
                    "pact_hazard_frames_three_seed_sum": pact,
                    f"{comparator.lower()}_hazard_frames_three_seed_sum": other,
                    "pact_minus_comparator_hazard_frames": pact - other,
                }
            )
        differences = [row["pact_minus_comparator_hazard_frames"] for row in rows]
        comparisons[f"PACT_vs_{comparator}"] = {
            "better_instances": sum(value < 0 for value in differences),
            "worse_instances": sum(value > 0 for value in differences),
            "tied_instances": sum(value == 0 for value in differences),
            "mean_pact_minus_comparator_hazard_frames": mean(differences),
            "median_pact_minus_comparator_hazard_frames": median(differences),
            "instances": rows,
        }
    return comparisons


def mechanism_characterization(
    arms: dict[str, dict[str, Any]], overlap: dict[str, Any]
) -> dict[str, Any]:
    pact = arms["PACT"]["high_contact_regime"]
    act = arms["ACT"]["high_contact_regime"]
    permuted = arms["PACT_PERMUTED"]["high_contact_regime"]

    def contrast(comparator: dict[str, Any]) -> dict[str, Any]:
        pact_entry = float(pact["entry_fraction"])
        comparator_entry = float(comparator["entry_fraction"])
        pact_frames = float(pact["hazard_frames_given_entry"]["mean"])
        comparator_frames = float(comparator["hazard_frames_given_entry"]["mean"])
        return {
            "entry_fraction_difference": pact_entry - comparator_entry,
            "relative_entry_reduction": 1.0 - pact_entry / comparator_entry,
            "conditional_mean_hazard_frames_difference": pact_frames - comparator_frames,
            "relative_conditional_mean_hazard_frames_reduction": 1.0
            - pact_frames / comparator_frames,
            "median_first_contact_step_difference": float(
                pact["first_hazard_contact_step"]["median"]
            )
            - float(comparator["first_hazard_contact_step"]["median"]),
        }

    valid_overlap = overlap["matched_instance_seed_overlap"][
        "PACT_vs_PACT_PERMUTED"
    ]
    return {
        "plain_statement": (
            "The retained totals support proximity primarily preventing entry into the "
            "high-contact regime. PACT also has fewer conditional contact frames than "
            "PACT_PERMUTED, but the absence of per-step runs means shortened entrapment "
            "or faster escape cannot be established. Tail susceptibility is strongly "
            "instance-linked, while proximity changes whether susceptible cases trigger."
        ),
        "PACT_vs_ACT": contrast(act),
        "PACT_vs_PACT_PERMUTED": contrast(permuted),
        "pact_high_rollouts_also_high_under_matched_permuted_condition": {
            "count": valid_overlap["intersection_count"],
            "of_pact_high_rollouts": valid_overlap["left_count"],
            "fraction": valid_overlap["fraction_of_left_also_right"],
        },
        "shortened_entrapment_established": False,
        "reason_shortening_not_established": (
            "No per-step contact classes or contiguous-run lengths were retained, and pair "
            "totals are not transition-entry counts."
        ),
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
        schedule.get("schema_version") != "pact_contact_endpoint_schedule_v1"
        or schedule.get("rollouts") != EXPECTED_ROLLOUTS
        or schedule.get("instance_count") != EXPECTED_INSTANCES
        or tuple(schedule.get("arms", [])) != ARMS
        or tuple(schedule.get("checkpoint_seeds", [])) != SEEDS
        or len(schedule.get("rows", [])) != EXPECTED_ROLLOUTS
    ):
        raise ValueError("frozen schedule design mismatch")
    if final_decision.get("decision") != "CONTACT_REDUCTION_WITH_TASK_BENEFIT":
        raise ValueError("awarded decision token changed")
    records: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for row in schedule["rows"]:
        record, source = result_record(row, output_root)
        records.append(record)
        sources.append(source)
    key_counts = Counter(
        (record["episode_id"], record["arm"], record["checkpoint_seed"])
        for record in records
    )
    if len(key_counts) != EXPECTED_ROLLOUTS or set(key_counts.values()) != {1}:
        raise ValueError("rollouts are not unique by instance, arm, and seed")
    if any(record["contact_frame_payload_retained"] for record in records) or any(
        record["contact_frame_payload_count"] for record in records
    ):
        raise ValueError("unexpected per-step contact payload was retained")
    grouped = {
        arm: [record for record in records if record["arm"] == arm] for arm in ARMS
    }
    if any(len(grouped[arm]) != 300 for arm in ARMS):
        raise ValueError("expected 300 rollouts per arm")
    arm_results = {arm: arm_characterization(grouped[arm]) for arm in ARMS}
    overlap = overlap_characterization(records)
    document: dict[str, Any] = {
        "schema_version": "pact_contact_tail_characterization_v1",
        "status": "post_hoc_exploratory_descriptive",
        "decision_bearing": False,
        "awarded_token_changed": False,
        "awarded_token": final_decision["decision"],
        "threshold": {
            "metric": "hazard_bar frames_with_contact",
            "operator": ">",
            "value": HIGH_CONTACT_THRESHOLD_FRAMES,
            "units": "physics contact frames",
            "source": "docs/PACT_TAIL_CHARACTERIZATION_PLAN.md",
            "selected_before_row_level_tail_analysis": True,
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
            "result_files": len(sources),
            "result_file_inventory_sha256": canonical_hash(sources),
            "rollout_metric_records_sha256": canonical_hash(records),
        },
        "instrumentation_limits": {
            "per_step_contact_payload_retained_rollouts": 0,
            "longest_contiguous_contact_run_available": False,
            "contact_class_totals_semantics": (
                "contact-pair samples accumulated at each audited physics frame; not transition entries"
            ),
            "frames_to_pair_samples_ratio_can_establish_entrapment": False,
        },
        "arms": arm_results,
        "tail_overlap": overlap,
        "paired_instance_view": paired_instance_characterization(records),
        "mechanism_characterization": mechanism_characterization(arm_results, overlap),
        "statistical_scope": {
            "confirmatory_p_values_computed": False,
            "post_hoc_confidence_intervals_computed": False,
            "description": (
                "All thresholds and groupings in this artifact are exploratory descriptions "
                "and do not modify the frozen confirmatory analysis."
            ),
        },
    }
    document["tail_characterization_sha256"] = canonical_hash(document)
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
    print(document["tail_characterization_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
