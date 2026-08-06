from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/absorbing_failure_characterization.json"
)
REPORT = ROOT / "docs/PACT_ABSORBING_FAILURE_CHARACTERIZATION.md"
RAW_ROOT = Path("/root/pact_contact_endpoint_artifacts/evaluation_v1/rows")
SPEC = importlib.util.spec_from_file_location(
    "pact_absorbing_failure_characterizer",
    ROOT / "scripts/characterize_pact_absorbing_failure.py",
)
assert SPEC is not None and SPEC.loader is not None
characterization = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(characterization)


def load_artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_artifact_is_self_hashed_exploratory_and_reconciled() -> None:
    document = load_artifact()
    payload = dict(document)
    observed = payload.pop("absorbing_failure_characterization_sha256")
    assert observed == characterization.canonical_hash(payload)
    assert document["status"] == "post_hoc_exploratory_descriptive"
    assert document["decision_bearing"] is False
    assert document["confirmatory_p_values_computed"] is False
    assert document["awarded_token"] == "CONTACT_REDUCTION_WITH_TASK_BENEFIT"
    assert document["verification"]["all_arm_pattern_counts_sum"] == 1200
    assert document["verification"]["non_ood_pattern_counts_sum"] == 900
    for threshold in ("1", "10", "50", "100"):
        all_patterns = document["contact_pattern_sensitivity"][threshold][
            "pooled_all_arms"
        ]
        non_ood = document["contact_pattern_sensitivity"][threshold][
            "pooled_non_ood"
        ]
        assert sum(item["rollouts"] for item in all_patterns.values()) == 1200
        assert sum(item["rollouts"] for item in non_ood.values()) == 900


def test_raw_files_independently_reproduce_strict_pattern_table() -> None:
    counts: Counter[str] = Counter()
    successes: Counter[str] = Counter()
    rows = 0
    for path in RAW_ROOT.glob("*/result.json"):
        result = json.loads(path.read_text())
        if result["arm"] == "PACT_ZERO":
            continue
        rows += 1
        audit = result["contact_audit"]
        grasp_frames = int(audit["frames_with_contact"]["grasp_target"])
        hazard_frames = int(audit["frames_with_contact"]["hazard_bar"])
        if grasp_frames > 0 and hazard_frames == 0:
            pattern = "target_engaged_never_hazard"
        elif grasp_frames == 0 and hazard_frames > 0:
            pattern = "hazard_engaged_target_not_engaged"
        elif grasp_frames == 0 and hazard_frames == 0:
            pattern = "neither_engaged"
        elif (
            audit["first_contact_step"]["hazard_bar"]
            < audit["first_contact_step"]["grasp_target"]
        ):
            pattern = "hazard_first_then_target"
        else:
            pattern = "target_first_then_hazard"
        counts[pattern] += 1
        successes[pattern] += int(result["task_success"])
    assert rows == 900
    assert counts == {
        "target_engaged_never_hazard": 701,
        "hazard_first_then_target": 59,
        "target_first_then_hazard": 53,
        "hazard_engaged_target_not_engaged": 67,
        "neither_engaged": 20,
    }
    assert successes["hazard_engaged_target_not_engaged"] == 0
    assert successes["hazard_first_then_target"] == 17
    assert successes["target_first_then_hazard"] == 17


def test_operational_tail_rates_and_sensitivity_are_fixed() -> None:
    document = load_artifact()
    sensitivity = document["contact_pattern_sensitivity"]
    expected = {
        "1": {"ACT": 34, "PACT": 3, "PACT_PERMUTED": 30},
        "10": {"ACT": 34, "PACT": 4, "PACT_PERMUTED": 33},
        "50": {"ACT": 34, "PACT": 5, "PACT_PERMUTED": 33},
        "100": {"ACT": 34, "PACT": 5, "PACT_PERMUTED": 33},
    }
    denominators = {"ACT": 59, "PACT": 33, "PACT_PERMUTED": 58}
    independently_reparsed = {
        arm: {"numerator": 0, "denominator": 0}
        for arm in ("ACT", "PACT", "PACT_PERMUTED")
    }
    for path in RAW_ROOT.glob("*/result.json"):
        result = json.loads(path.read_text())
        arm = result["arm"]
        if arm not in independently_reparsed:
            continue
        frames = result["contact_audit"]["frames_with_contact"]
        if int(frames["hazard_bar"]) > 500:
            independently_reparsed[arm]["denominator"] += 1
            independently_reparsed[arm]["numerator"] += int(
                int(frames["grasp_target"]) < 50
            )
    assert independently_reparsed == {
        "ACT": {"numerator": 34, "denominator": 59},
        "PACT": {"numerator": 5, "denominator": 33},
        "PACT_PERMUTED": {"numerator": 33, "denominator": 58},
    }
    for threshold, by_arm in expected.items():
        observed = sensitivity[threshold]["high_contact_low_target_by_arm"]
        for arm, numerator in by_arm.items():
            assert observed[arm]["low_target_engagement_rollouts"] == numerator
            assert observed[arm]["high_contact_rollouts"] == denominators[arm]
            assert observed[arm]["task_successes_in_low_target_group"] == 0


def test_seed_first_bootstrap_and_matched_both_high_subset() -> None:
    document = load_artifact()
    tail = document["high_contact_low_target_engagement"]
    assert tail["bootstrap"] == {
        "all_arms_and_seeds_within_sampled_instance_move_together": True,
        "confirmatory": False,
        "instances": 100,
        "method": "paired whole-instance cluster percentile bootstrap",
        "replicates": 20000,
        "seed": 2026080601,
    }
    assert list(tail["contrasts_seeds_unpooled_first"]) == ["3101", "3102", "3103"]
    for seed in ("3101", "3102", "3103"):
        assert (
            tail["contrasts_seeds_unpooled_first"][seed][
                "PACT_minus_PACT_PERMUTED"
            ]["difference"]
            < 0
        )
    pooled = tail["pooled_contrasts"]["PACT_minus_PACT_PERMUTED"]
    assert pooled["left_numerator"] == 5
    assert pooled["left_denominator"] == 33
    assert pooled["right_numerator"] == 33
    assert pooled["right_denominator"] == 58
    assert pooled["instance_cluster_bootstrap_ci_95"][1] < 0
    both_high = tail["matched_both_arms_high_subset"][
        "PACT_minus_PACT_PERMUTED"
    ]["pooled"]
    assert (both_high["left_numerator"], both_high["left_denominator"]) == (5, 31)
    assert (both_high["right_numerator"], both_high["right_denominator"]) == (
        13,
        31,
    )
    assert both_high["instance_cluster_bootstrap_ci_95"][1] < 0


def test_ordering_clip4_and_instrumentation_limits() -> None:
    document = load_artifact()
    ordering = document["ordering_negative_result"]
    assert (
        ordering["pooled_non_ood"]["left_numerator"],
        ordering["pooled_non_ood"]["left_denominator"],
    ) == (17, 59)
    assert (
        ordering["pooled_non_ood"]["right_numerator"],
        ordering["pooled_non_ood"]["right_denominator"],
    ) == (17, 53)
    assert ordering["pooled_non_ood"]["instance_cluster_bootstrap_ci_95"][0] < 0
    assert ordering["pooled_non_ood"]["instance_cluster_bootstrap_ci_95"][1] > 0

    clip4 = document["clip4_placement"]
    assert clip4["exact_values_verified"] == {
        "first_grasp_target_contact_step": 151,
        "first_hazard_contact_step": 128,
        "grasp_target_frames": 12,
        "hazard_frames": 17609,
    }
    assert clip4["same_instance_high_contact_counts"] == {
        "all_arms": 10,
        "all_arms_denominator": 12,
        "non_ood": 8,
        "non_ood_denominator": 9,
    }
    where = document["where_it_diverges"]
    assert where["trajectory_payload_inventory"]["surviving_trajectory_files"] == 2
    assert where["distance_travelled_before_first_hazard"]["available"] is False
    assert where["target_neighbourhood_reached"]["available"] is False


def test_frozen_decision_files_unchanged_and_report_references_artifact() -> None:
    expected_hashes = {
        ROOT / "docs/PACT_CONTACT_ENDPOINT_DECISION.md": (
            "6ea97ea789e1c5d57572c7e231b30c867881f0e21b81ad11f950cef9bf860f1b"
        ),
        ROOT / "diagnostics_output/pact_contact_endpoint/analysis.json": (
            "4bbf25c97472bbfd8f13c3352b6b1c8afae9284cd00351e0502149362ccde1f7"
        ),
        ROOT / "diagnostics_output/pact_contact_endpoint/final_decision.json": (
            "52ce515bc09771d9ecdc8a39e31111f66e7e632f5949e82ed2497908ac2458bf"
        ),
    }
    for path, expected in expected_hashes.items():
        assert characterization.file_hash(path) == expected
    artifact = load_artifact()
    report = REPORT.read_text()
    assert artifact["absorbing_failure_characterization_sha256"] in report
    assert "cannot fully separate" in report
    assert "no confirmatory p-values were computed" in report
