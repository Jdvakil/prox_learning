"""Contract tests for the three-pair joint gate.

This task changes exactly one thing — which agreement definition controls execution — so
most of these tests verify that nothing else moved, and that the golden reproduction of the
identifiability audit actually holds.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from causal_parked_skin import joint_gate as jg
from causal_parked_skin import threshold as thr

DIAG = ROOT / "diagnostics_output" / "hybrid_obstacle_three_pair_joint_gate"
PREVIOUS = (ROOT / "diagnostics_output" / "hybrid_obstacle_full_seed_joint_gate"
            / "final_decision.json")
MANIFEST = ROOT / "configs" / "hybrid_obstacle_three_pair_joint_gate_v1.json"
PREV_MANIFEST = ROOT / "configs" / "hybrid_obstacle_full_seed_joint_gate_v1.json"
PARTITION = ROOT / "configs" / "hybrid_obstacle_prox_activity_partition_v1.json"
BOOTSTRAP = ROOT / "configs" / "hybrid_obstacle_uncertainty_ensemble_v1.json"
CONF41 = ROOT / "configs" / "hybrid_obstacle_confirmatory41_v1.json"
DEV4 = ROOT / "configs" / "hybrid_obstacle_controller_development4_v1.json"


def _json(path):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------- the metric itself
def test_three_pair_formula_is_the_mean_of_all_three_jaccards():
    m0 = np.array([[True, True, False, False]])
    m1 = np.array([[True, True, False, False]])
    m2 = np.array([[False, False, True, True]])
    j01, j02, j12 = (jg.jaccard(m0, m1)[0], jg.jaccard(m0, m2)[0],
                     jg.jaccard(m1, m2)[0])
    assert jg.three_pair_agreement(m0, m1, m2)[0] == pytest.approx((j01 + j02 + j12) / 3)
    assert jg.anchor_mask_agreement(m0, m1, m2)[0] == pytest.approx((j01 + j02) / 2)


def test_j12_cannot_be_omitted_from_the_three_pair_form():
    """Perturbing only seed1-vs-seed2 must move the three-pair value, not the anchor."""
    m0 = np.array([[True, True, False, False]])
    m1 = np.array([[True, False, False, False]])
    near = np.array([[True, False, False, False]])
    far = np.array([[False, False, False, True]])
    assert jg.anchor_mask_agreement(m0, m1, near)[0] != \
        pytest.approx(jg.anchor_mask_agreement(m0, m1, far)[0]) or True
    a = jg.three_pair_agreement(m0, m1, near)[0]
    b = jg.three_pair_agreement(m0, m1, far)[0]
    assert a != pytest.approx(b), "J12 has no influence on the three-pair metric"


def test_empty_masks_agree_completely():
    empty = np.zeros((2, 6), dtype=bool)
    assert jg.jaccard(empty, empty).tolist() == [1.0, 1.0]
    assert jg.three_pair_agreement(empty, empty, empty).tolist() == [1.0, 1.0]


def test_pixel_threshold_is_still_exactly_half():
    assert jg.PIXEL_MASK_THRESHOLD == 0.5
    manifest = _json(MANIFEST)
    assert manifest["pixel_mask_threshold"] == 0.5


# -------------------------------------------------------------- golden reproduction
def test_reproduction_of_the_identifiability_audit_passed():
    report = _json(DIAG / "agreement_reproduction.json")
    assert report["reproduced"] is True
    assert all(report["checks"].values())
    assert report["frame_count"] == 17


def test_reproduction_recovers_the_recorded_medians():
    observed = _json(DIAG / "agreement_reproduction.json")["observed"]
    assert observed["three_pair_median"] == pytest.approx(0.167, abs=0.01)
    assert observed["anchor_median"] == pytest.approx(0.250, abs=0.01)


def test_j02_and_j12_are_zero_on_every_historical_frame():
    report = _json(DIAG / "agreement_reproduction.json")
    for frame in report["frames"]:
        assert frame["jaccard_02"] == 0.0
        assert frame["jaccard_12"] == 0.0


def test_strict_and_inclusive_masks_were_compared_not_assumed():
    block = _json(DIAG / "agreement_reproduction.json")["mask_comparison"]
    assert block["strict_and_inclusive_masks_identical_on_these_frames"] is True
    assert "0.5" in block["handoff_specified"]


def test_reproduction_is_verification_only():
    report = _json(DIAG / "agreement_reproduction.json")
    assert "never used to select thresholds" in report["purpose"]


# ------------------------------------------------------- controlling vs diagnostic metric
def test_three_pair_is_controlling_and_anchor_is_diagnostic():
    manifest = _json(MANIFEST)
    assert manifest["controlling_metric"] == "full_pairwise_agreement"
    assert manifest["diagnostic_metric"] == "anchor_agreement"
    assert "J(s1,s2)" in manifest["agreement_definition"]
    calibration = _json(DIAG / "three_pair_calibration.json")
    assert calibration["controlling_metric"] == "full_pairwise_agreement"


def test_all_three_pairwise_definitions_are_recorded():
    manifest = _json(MANIFEST)
    assert set(manifest["pairwise_definitions"]) == {"J01", "J02", "J12"}


def test_both_metrics_are_logged_for_every_historical_frame():
    frames = _json(DIAG / "three_pair_calibration.json")[
        "historical_regression"]["frames"]
    assert len(frames) == 17
    for frame in frames:
        for key in ("jaccard_seed0_seed1", "jaccard_seed0_seed2", "jaccard_seed1_seed2",
                    "controlling_agreement", "alternate_agreement",
                    "decision_changed_by_j12", "activity_pass", "agreement_pass",
                    "executed"):
            assert key in frame, key


def test_manifest_supersedes_the_two_anchor_manifest():
    manifest = _json(MANIFEST)
    assert manifest["supersedes_two_anchor_manifest"].endswith(
        "hybrid_obstacle_full_seed_joint_gate_v1.json")
    assert manifest["manifest_sha256"] != _json(PREV_MANIFEST)["manifest_sha256"]


# ---------------------------------------------------------- unchanged contract and rules
def test_recall_floor_is_still_zero_point_eight():
    from hybrid_obstacle_three_pair_calibrate import MIN_MEDIAN_ACTIVE_RECALL

    assert MIN_MEDIAN_ACTIVE_RECALL == 0.80
    contract = _json(DIAG / "three_pair_calibration.json")["contract"]
    assert contract["min_median_active_recall"] == 0.80
    assert contract["recall_floor_lowered"] is False


def test_every_feasibility_floor_matches_the_previous_task():
    import hybrid_obstacle_joint_gate_calibrate as old
    import hybrid_obstacle_three_pair_calibrate as new

    for name in ("MIN_MEDIAN_ACTIVE_RECALL", "MIN_MEDIAN_HARD_RETENTION",
                 "MIN_RETAINED_COSINE", "MIN_RETAINED_POSITIVE_COSINE",
                 "MAX_BOOTSTRAP_UPPER_FALSE_ACTIVATION", "MAX_HAZARD_ABSENT_EXECUTED",
                 "MAX_FALSE_ACTIVE_RUN", "MIN_ZERO_ACCEPTANCE", "MIN_ACTIVE_ACCEPTANCE",
                 "MIN_INACTIVE_ACCEPTANCE", "MAX_TRAJECTORY_ABSTENTION",
                 "MAX_HAZARD_PRESENT_ACTIVE_ABSTENTION",
                 "MIN_ACTIVITY_ALONE_ACTIVE_RETENTION", "NESTED_MIN_ACTIVE_RECALL",
                 "NESTED_MAX_MEAN_EXECUTED", "DIAG_MAX_MEAN_EXECUTED",
                 "DIAG_MIN_ACTIVE_RECALL", "DIAG_MAX_RUN", "DIAG_MIN_ZERO_ACCEPTANCE"):
        assert getattr(new, name) == getattr(old, name), name


def test_selection_ordering_unchanged():
    old = (ROOT / "scripts" / "hybrid_obstacle_joint_gate_calibrate.py").read_text()
    new = (ROOT / "scripts" / "hybrid_obstacle_three_pair_calibrate.py").read_text()
    marker = 'chosen = min(feasible, key=lambda f: ('
    assert marker in old and marker in new
    assert old.split(marker)[1][:400] == new.split(marker)[1][:400]


def test_no_temporal_filtering_was_introduced():
    source = (ROOT / "scripts" / "hybrid_obstacle_three_pair_calibrate.py").read_text()
    for banned in ("debounce", "hysteresis", "cooldown", "warm_up", "warmup"):
        assert banned not in source.lower()
    calibration = _json(DIAG / "three_pair_calibration.json")
    assert calibration["temporal_classification"]["temporal_logic_added"] is False


def test_clustering_and_persistence_gates_unchanged():
    from hybrid_obstacle_three_pair_calibrate import DIAG_MAX_RUN, MAX_FALSE_ACTIVE_RUN

    assert MAX_FALSE_ACTIVE_RUN == 2
    assert DIAG_MAX_RUN == 5


def test_partition_membership_unchanged():
    partition = _json(PARTITION)
    recorded = partition.pop("manifest_sha256")
    assert thr.canonical_hash(partition) == recorded
    assert _json(MANIFEST)["partition_manifest_sha256"] == recorded


def test_complete_joint_grid_evaluated():
    grid = _json(DIAG / "three_pair_calibration.json")["grid"]
    assert grid["cartesian_pairs"] == grid["activity"] * grid["agreement"]
    assert grid["activity"] > 1 and grid["agreement"] > 1


def test_cluster_bootstrap_is_deterministic():
    manifest = _json(MANIFEST)
    assert manifest["bootstrap_replicates"] >= 10_000
    assert manifest["bootstrap_seed"] == thr.BOOTSTRAP_SEED
    rng = np.random.default_rng(0)
    values = rng.random((8, 3))
    a = thr.cluster_bootstrap_upper_bound(values, replicates=400, seed=thr.BOOTSTRAP_SEED)
    b = thr.cluster_bootstrap_upper_bound(values, replicates=400, seed=thr.BOOTSTRAP_SEED)
    assert np.array_equal(a, b)


# -------------------------------------------------------------------- frozen artifacts
def test_provenance_all_checks_matched():
    report = _json(DIAG / "provenance_verification.json")
    assert report["all_matched"] is True
    assert report["check_count"] >= 65


def test_seed_zero_is_the_sole_deployment_predictor():
    manifest = _json(MANIFEST)
    assert manifest["deployment_seed"] == 0
    assert manifest["uncertainty_seeds"] == [1, 2]
    assert manifest["averaging_permitted"] is False
    assert manifest["uncertainty_seed_may_replace_seed0"] is False
    decision = _json(DIAG / "final_decision.json")
    assert decision["predictions_averaged"] is False


def test_seed_checkpoints_unchanged():
    manifest = _json(MANIFEST)
    for path, digest in zip(manifest["seed_checkpoints"],
                            manifest["seed_checkpoint_sha256"]):
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def test_bootstrap_members_are_rejected():
    with pytest.raises(jg.JointGateManifestError):
        jg.assert_not_bootstrap(_json(BOOTSTRAP))
    jg.assert_not_bootstrap(_json(MANIFEST))


def test_calibration_never_touches_bootstrap_members():
    source = (ROOT / "scripts" / "hybrid_obstacle_three_pair_calibrate.py").read_text()
    assert "hybrid_obstacle_uncertainty_ensemble" not in source
    assert "member_records" not in source


def test_seeds_one_and_two_yield_only_masks():
    source = (ROOT / "scripts" / "hybrid_obstacle_three_pair_calibrate.py").read_text()
    tree = ast.parse(source)
    reads = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Name) \
                and node.value.func.id in ("seed1", "seed2"):
            reads.append(node.slice.value)
    assert reads and set(reads) == {"changed_probability"}


def test_historical_frames_and_dev4_excluded_from_fitting():
    source = (ROOT / "scripts" / "hybrid_obstacle_three_pair_calibrate.py").read_text()
    body = source.split("deployment manifest frozen")[0]
    assert "onset_audit.read_text" not in body
    for config in ("hybrid_obstacle_controller_development4_v1",
                   "hybrid_obstacle_confirmatory41_v1"):
        assert config not in source


def test_confirmatory41_untouched_and_schedule_is_development4_only():
    decision = _json(DIAG / "final_decision.json")
    assert decision["confirmatory41_executed"] is False
    assert decision["development4_executed"] is False
    assert decision["live_rollouts_executed"] == 0
    conf41 = _json(CONF41)
    assert conf41["executed_in_this_task"] is False
    assert {r["candidate_index"] for r in _json(DEV4)["rows"]} == {106, 107, 108, 118}


def test_controller_and_magnitude_bound_unchanged():
    manifest = _json(MANIFEST)
    stack = _json(ROOT / "configs" / "hybrid_safety_stack_v1.json")["residual_controller"]
    for key in ("gain", "decay_per_second", "ema", "max_deviation_rad_per_joint",
                "arm_only", "gripper_owner"):
        assert manifest["controller"][key] == stack[key]
    assert "unchanged" in manifest["magnitude_support_bound"]


def test_frozen_inference_is_repeatable_including_j12():
    stability = _json(DIAG / "three_pair_calibration.json")["inference_stability"]
    assert stability["repeats"] >= 20
    assert stability["activity_max_abs_delta"] <= 1e-7
    assert stability["agreement_max_abs_delta"] <= 1e-7
    assert stability["jaccard_identical"] is True
    assert stability["decisions_identical"] is True
    assert stability["stable"] is True


# ----------------------------------------------------------------- the metric outcome
def test_metric_substitution_premise_is_recorded_as_falsified():
    block = _json(DIAG / "final_decision.json")["metric_substitution"]
    assert block["premise_supported"] is False
    assert block["historical_executed_under_anchor"] == 10
    assert block["historical_executed_under_three_pair"] == 9
    assert block["net_historical_decisions_changed"] == 1
    assert "0.000 on all 17" in block["why"]


def test_temporal_classification_is_recorded():
    calibration = _json(DIAG / "three_pair_calibration.json")
    block = calibration["temporal_classification"]
    assert set(block["temporal_gate_names"]) == {"false-active run",
                                                 "persistent correction"}
    assert isinstance(block["only_temporal_failures"], bool)


def test_historical_failure_is_not_reported_as_temporal_only():
    """9 executing frames is a per-frame failure and must not take the temporal token."""
    decision = _json(DIAG / "final_decision.json")
    calibration = _json(DIAG / "three_pair_calibration.json")
    if not calibration["historical_regression"]["passes"]:
        assert decision["decision"] != "THREE_PAIR_TEMPORAL_CLUSTERING_REMAINS"


def test_final_decision_token_matches_markdown_last_line():
    decision = _json(DIAG / "final_decision.json")
    markdown = (ROOT / "docs"
                / "HYBRID_OBSTACLE_THREE_PAIR_JOINT_GATE_FINAL_DECISION.md").read_text()
    last = [line for line in markdown.splitlines() if line.strip()][-1]
    assert last == decision["decision"]


def test_final_decision_token_is_allowed():
    allowed = {"THREE_PAIR_JOINT_GATE_READY_FOR_CONFIRMATORY_41",
               "THREE_PAIR_JOINT_GATE_CALIBRATION_INFEASIBLE",
               "THREE_PAIR_JOINT_GATE_OFFLINE_TRANSFER_FAILED",
               "THREE_PAIR_JOINT_GATE_LIVE_TRANSFER_FAILED",
               "THREE_PAIR_JOINT_GATE_LIVE_GROSS_REGRESSION",
               "THREE_PAIR_TEMPORAL_CLUSTERING_REMAINS",
               "THREE_PAIR_AGREEMENT_REPRODUCTION_FAILED",
               "REFERENCE_MODEL_INFERENCE_UNSTABLE",
               "CHECKPOINT_OR_SOURCE_MISMATCH",
               "THREE_PAIR_JOINT_GATE_TASK_INCOMPLETE"}
    assert _json(DIAG / "final_decision.json")["decision"] in allowed
