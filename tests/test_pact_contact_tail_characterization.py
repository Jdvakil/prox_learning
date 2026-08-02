from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import characterize_pact_contact_tail as tail


ARTIFACT = ROOT / "diagnostics_output/pact_contact_endpoint/tail_characterization.json"
REPORT = ROOT / "docs/PACT_TAIL_CHARACTERIZATION.md"


def test_linear_percentile_contract() -> None:
    values = [0, 10, 20, 30]
    assert tail.percentile(values, 0) == 0
    assert tail.percentile(values, 50) == 15
    assert tail.percentile(values, 75) == pytest.approx(22.5)
    assert tail.percentile(values, 100) == 30
    assert tail.percentile([], 50) is None


def test_concentration_uses_fixed_rollout_fractions() -> None:
    values = list(range(1, 101))
    observed = tail.concentration(values)
    assert observed["top_1_percent"]["rollout_count"] == 1
    assert observed["top_5_percent"]["rollout_count"] == 5
    assert observed["top_10_percent"]["rollout_count"] == 10
    assert observed["top_1_percent"]["hazard_frames"] == 100


def test_overlap_reports_directional_fractions() -> None:
    observed = tail.set_overlap({1, 2}, {2, 3, 4})
    assert observed == {
        "left_count": 2,
        "right_count": 3,
        "intersection_count": 1,
        "union_count": 4,
        "jaccard": 0.25,
        "fraction_of_left_also_right": 0.5,
        "fraction_of_right_also_left": pytest.approx(1 / 3),
    }


def test_frozen_tail_artifact_is_self_hashed_and_non_decision_bearing() -> None:
    document = json.loads(ARTIFACT.read_text())
    payload = dict(document)
    observed = payload.pop("tail_characterization_sha256")
    assert observed == tail.canonical_hash(payload)
    assert document["status"] == "post_hoc_exploratory_descriptive"
    assert document["decision_bearing"] is False
    assert document["awarded_token_changed"] is False
    assert document["awarded_token"] == "CONTACT_REDUCTION_WITH_TASK_BENEFIT"
    assert document["threshold"]["value"] == 500
    assert document["threshold"]["operator"] == ">"
    assert document["sources"]["result_files"] == 1200
    assert document["statistical_scope"]["confirmatory_p_values_computed"] is False
    assert document["statistical_scope"]["post_hoc_confidence_intervals_computed"] is False


def test_frozen_characterization_captures_entry_prevention_and_limits() -> None:
    document = json.loads(ARTIFACT.read_text())
    arms = document["arms"]
    assert arms["PACT"]["high_contact_regime"]["entry_count"] == 33
    assert arms["ACT"]["high_contact_regime"]["entry_count"] == 59
    assert arms["PACT_PERMUTED"]["high_contact_regime"]["entry_count"] == 58
    assert document["instrumentation_limits"][
        "longest_contiguous_contact_run_available"
    ] is False
    assert document["mechanism_characterization"]["shortened_entrapment_established"] is False
    valid_overlap = document["tail_overlap"]["matched_instance_seed_overlap"][
        "PACT_vs_PACT_PERMUTED"
    ]
    assert valid_overlap["intersection_count"] == 31
    assert valid_overlap["fraction_of_left_also_right"] == pytest.approx(31 / 33)


def test_report_is_explicitly_post_hoc_and_binds_machine_artifact() -> None:
    document = json.loads(ARTIFACT.read_text())
    report = REPORT.read_text()
    assert "Post-hoc exploratory characterization" in report
    assert "cannot establish faster escape" in report
    assert document["tail_characterization_sha256"] in report
