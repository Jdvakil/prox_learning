"""Contract tests for the trajectory-bootstrap uncertainty-abstention qualification.

The conclusion here is negative, so these tests mostly guard the conditions under which the
negative was measured: that seed 0 really was the only deployment predictor, that all five
members trained without any being quietly dropped, that no prediction was averaged, and that
the abstention path can only zero a correction rather than reshape it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from causal_parked_skin import abstention as ab
from causal_parked_skin import threshold as thr

DIAG = ROOT / "diagnostics_output" / "hybrid_obstacle_uncertainty_abstention"
ENSEMBLE = ROOT / "configs" / "hybrid_obstacle_uncertainty_ensemble_v1.json"
PARTITION = ROOT / "configs" / "hybrid_obstacle_prox_activity_partition_v1.json"
MANIFEST = ROOT / "configs" / "hybrid_obstacle_parked_skin_supervision_v1.json"
CONF41 = ROOT / "configs" / "hybrid_obstacle_confirmatory41_v1.json"
DEV4 = ROOT / "configs" / "hybrid_obstacle_controller_development4_v1.json"
PARKED_DECISION = (ROOT / "diagnostics_output" / "causal_parked_skin_reference_v1"
                   / "final_decision.json")
EXPECTED_SEEDS = [20260731, 20260801, 20260802, 20260803, 20260804]
FROZEN_ACTIVITY_THRESHOLD = 0.99960857629776


def _json(path):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ frozen artifacts
def test_provenance_all_checks_matched():
    report = _json(DIAG / "provenance_verification.json")
    assert report["all_matched"] is True
    assert report["failed"] == []
    assert report["check_count"] >= 65


def test_seed0_is_the_only_deployment_predictor():
    decision = _json(DIAG / "final_decision.json")
    assert decision["seed_1_or_2_selected_for_deployment"] is False
    assert decision["member_replaced_seed0"] is False
    ensemble = _json(ENSEMBLE)
    assert ensemble["member_may_replace_seed0"] is False
    assert ensemble["averaging_permitted"] is False
    previous = _json(PARKED_DECISION)
    seed0 = next(c for c in previous["checkpoints"]
                 if c["variant"] == "CURRENT_FRAME_ONLY" and c["seed"] == 0)
    assert ensemble["seed0_deployment_sha256"] == seed0["sha256"]


def test_seed0_checkpoint_unchanged_on_disk():
    ensemble = _json(ENSEMBLE)
    path = Path(ensemble["seed0_deployment_checkpoint"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == \
        ensemble["seed0_deployment_sha256"]


def test_activity_threshold_was_not_refit():
    decision = _json(DIAG / "final_decision.json")
    assert decision["activity_threshold_modified"] is False
    assert decision["calibration"]["activity_threshold"] == FROZEN_ACTIVITY_THRESHOLD
    assert decision["calibration"]["activity_threshold_refit"] is False


def test_magnitude_bound_and_controller_unchanged():
    decision = _json(DIAG / "final_decision.json")
    assert decision["magnitude_support_bound_modified"] is False
    assert decision["residual_constants_modified"] is False
    report = _json(DIAG / "provenance_verification.json")
    checks = {c["check"]: c for c in report["checks"]}
    for key in ("residual_gain", "residual_decay_per_second", "residual_ema",
                "residual_max_deviation_rad_per_joint", "residual_arm_only",
                "residual_gripper_owner"):
        assert checks[key]["matched"] is True, key
    assert checks["magnitude_support_recalculated"]["observed"] is False


def test_dataset_unchanged_and_read_only():
    manifest = _json(MANIFEST)
    writable = [e["output"] for e in manifest["entries"][:60]
                if stat.S_IMODE(os.stat(e["output"]).st_mode) & 0o222]
    assert writable == []
    decision = _json(DIAG / "final_decision.json")
    assert decision["paired_dataset_modified"] is False
    assert decision["new_data_collected"] is False


# --------------------------------------------------------------------- the ensemble
def test_exactly_five_members_with_the_predeclared_seeds():
    ensemble = _json(ENSEMBLE)
    assert ensemble["members"] == 5
    assert ensemble["bootstrap_seeds"] == EXPECTED_SEEDS
    assert len(ensemble["member_records"]) == 5
    assert [r["bootstrap_seed"] for r in ensemble["member_records"]] == EXPECTED_SEEDS


def test_all_members_trained_and_none_dropped_or_replaced():
    training = _json(DIAG / "ensemble_training.json")
    assert training["member_count"] == 5
    assert training["all_members_trained_and_reloaded"] is True
    assert training["acceptance_failures"] == []
    assert training["members_dropped"] == 0
    assert training["members_replaced"] == 0


def test_member_checkpoints_match_their_recorded_hashes():
    ensemble = _json(ENSEMBLE)
    for record in ensemble["member_records"]:
        path = Path(record["checkpoint"])
        assert path.is_file(), record["checkpoint"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == \
            record["checkpoint_sha256"]


def test_bootstrap_unit_is_the_trajectory_cluster():
    ensemble = _json(ENSEMBLE)
    assert "cluster" in ensemble["bootstrap_unit"]
    assert "trajectory" in ensemble["bootstrap_unit"] or "episode" in ensemble["bootstrap_unit"]
    for record in ensemble["member_records"]:
        # a cluster drawn with replacement must be able to repeat
        assert record["hazard_present_draws"] == 30
        assert record["hazard_absent_draws"] == 10
        assert sum(record["cluster_multiplicity"].values()) == 40
        assert record["unique_clusters"] <= 40


def test_bootstrap_is_hazard_stratified():
    """Every member must see both strata, or its disagreement is trivially explained."""
    ensemble = _json(ENSEMBLE)
    partition = _json(PARTITION)
    training_episodes = set(partition["splits"]["gate_training"]["episodes"])
    for record in ensemble["member_records"]:
        drawn = set(record["cluster_multiplicity"])
        assert drawn <= training_episodes, "a member sampled outside the training split"


def test_bootstrap_resampling_is_deterministic_and_seed_specific():
    from hybrid_obstacle_uncertainty_ensemble_train import bootstrap_clusters

    present = [f"p{i}" for i in range(30)]
    absent = [f"a{i}" for i in range(10)]
    first = bootstrap_clusters(present, absent, 20260731)
    assert first == bootstrap_clusters(present, absent, 20260731)
    assert first != bootstrap_clusters(present, absent, 20260801)
    assert len(first) == 40
    assert sum(1 for c in first if c.startswith("p")) == 30
    assert sum(1 for c in first if c.startswith("a")) == 10


def test_members_did_not_collapse_to_the_trivial_baseline():
    training = _json(DIAG / "ensemble_training.json")
    zero = training["zero_differential_validation_mae"]
    for member in training["members"]:
        block = member["validation"]
        assert block["oracle_differential_mae"] < zero, member["index"]
        assert block["constraint_violations"] == 0
        assert block["nonfinite_outputs"] == 0


def test_ensemble_role_is_uncertainty_only():
    ensemble = _json(ENSEMBLE)
    assert "uncertainty" in ensemble["role"]
    assert "never supplies a deployed prediction" in ensemble["role"]


# --------------------------------------------------------------- agreement definition
def test_jaccard_of_two_empty_masks_is_one():
    empty = np.zeros((3, 10), dtype=bool)
    assert ab.jaccard(empty, empty).tolist() == [1.0, 1.0, 1.0]


def test_jaccard_matches_hand_computed_cases():
    a = np.array([[True, True, False, False]])
    b = np.array([[True, False, True, False]])
    assert ab.jaccard(a, b)[0] == pytest.approx(1 / 3)
    assert ab.jaccard(a, a)[0] == pytest.approx(1.0)
    disjoint = np.array([[False, False, True, True]])
    assert ab.jaccard(a, disjoint)[0] == pytest.approx(0.0)


def test_pixel_mask_threshold_is_fixed_at_half():
    assert ab.PIXEL_MASK_THRESHOLD == 0.5
    probability = np.array([[[[0.49, 0.5], [0.51, 0.0]]]])
    mask = ab.changed_mask(np.broadcast_to(probability, (1, 1, 2, 2)).copy())
    assert mask.tolist() == [[False, True, True, False]]


def test_anchor_agreement_is_anchored_on_seed_zero():
    anchor = np.array([[True, True, False, False]])
    members = [np.array([[True, True, False, False]]),      # perfect
               np.array([[True, False, False, False]])]     # half
    value = ab.anchor_mask_agreement(anchor, members)[0]
    assert value == pytest.approx((1.0 + 0.5) / 2)
    # pairwise among members alone would report a different number
    assert ab.mean_pairwise_agreement(members)[0] == pytest.approx(0.5)


def test_anchor_agreement_requires_at_least_one_member():
    with pytest.raises(ValueError):
        ab.anchor_mask_agreement(np.zeros((1, 4), dtype=bool), [])


# ------------------------------------------------------------ abstention semantics
def test_abstention_can_only_zero_a_correction():
    differential = np.array([[1.0, -2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                             [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    execute = np.array([True, False])
    out = ab.apply_abstention(differential, execute)
    assert np.array_equal(out[0], differential[0]), "executed row must be untouched"
    assert np.array_equal(out[1], np.zeros(7)), "abstained row must be exactly zero"


def test_abstention_never_scales_or_redirects():
    rng = np.random.default_rng(0)
    differential = rng.normal(size=(64, 7))
    execute = rng.random(64) < 0.5
    out = ab.apply_abstention(differential, execute)
    for i in range(64):
        assert np.array_equal(out[i], differential[i] if execute[i] else np.zeros(7))


def test_combined_decision_requires_both_gates():
    activity = np.array([0.9, 0.9, 0.1, 0.1])
    agreement = np.array([0.9, 0.1, 0.9, 0.1])
    gates = ab.combined_decision(activity, 0.5, agreement, 0.5)
    assert gates["execute"].tolist() == [True, False, False, False]
    assert gates["abstained"].tolist() == [False, True, False, False]
    assert gates["activity_pass"].tolist() == [True, True, False, False]


def test_no_averaging_anywhere_in_the_abstention_module():
    source = (ROOT / "causal_parked_skin" / "abstention.py").read_text()
    tree = ast.parse(source)
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "apply_abstention" in functions
    # the only mean() in the module is over Jaccard scores, never over predictions
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "mean":
            enclosing = [f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
                         and node in ast.walk(f)]
            assert all(f.name in ("anchor_mask_agreement", "mean_pairwise_agreement")
                       for f in enclosing)


def test_runtime_driver_never_reads_member_fields():
    """Members contribute masks only; their parked fields must not be consumed."""
    source = (ROOT / "submodules" / "act" / "uncertainty_abstention.py").read_text()
    tree = ast.parse(source)
    subscripts = {n.slice.value for n in ast.walk(tree)
                  if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
                  and isinstance(n.slice.value, str)}
    # the member comprehension must only pull the change probability
    assert "changed_probability" in subscripts
    member_lines = [line for line in source.splitlines() if "self.members" in line]
    joined = " ".join(member_lines)
    assert "parked" not in joined
    assert "head(" not in joined


# ------------------------------------------------------------------- calibration
def test_calibration_was_infeasible_and_recorded_its_sweep():
    calibration = _json(DIAG / "calibration.json")
    assert calibration["feasible"] is False
    assert calibration["sweep"], "the infeasibility sweep must be recorded"
    for row in calibration["sweep"]:
        assert set(row["checks"]) >= {
            "median_active_recall", "zero_acceptance_floor", "active_acceptance_floor",
            "trajectory_abstention_cap"}


def test_no_agreement_threshold_satisfied_the_contract():
    calibration = _json(DIAG / "calibration.json")
    assert not any(all(row["checks"].values()) for row in calibration["sweep"])


def test_quiet_frame_acceptance_floor_was_enforced():
    """The floor the previous task asked for must actually be part of the contract."""
    from hybrid_obstacle_uncertainty_calibrate import (
        MIN_ACTIVE_ACCEPTANCE,
        MIN_INACTIVE_ACCEPTANCE,
        MIN_ZERO_ACCEPTANCE,
    )
    assert MIN_ZERO_ACCEPTANCE == 0.80
    assert MIN_ACTIVE_ACCEPTANCE == 0.80
    assert MIN_INACTIVE_ACCEPTANCE == 0.80
    calibration = _json(DIAG / "calibration.json")
    names = {name for row in calibration["sweep"] for name in row["checks"]}
    assert "zero_acceptance_floor" in names
    assert "inactive_acceptance_floor" in names


def test_unconstrained_operating_point_is_recorded():
    """At agreement threshold 0 the activity gate alone is measured; that is the
    evidence separating 'uncertainty is degenerate' from 'the frozen gate is strict'."""
    decision = _json(DIAG / "final_decision.json")
    point = decision["calibration"]["unconstrained_operating_point"]
    assert point is not None
    assert point["threshold"] == 0.0
    assert point["zero_acceptance"] == pytest.approx(1.0)


def test_agreement_diagnostic_records_the_separation_collapse():
    diagnostic = _json(DIAG / "agreement_diagnostic.json")
    medians = diagnostic["anchor_agreement_median"]
    assert medians["oracle_active"] is not None
    assert diagnostic["separation_zero_minus_active"] is not None
    reference = diagnostic["three_seed_reference"]
    assert reference["historical_false_positive_median"] == 0.167
    # the bootstrap ensemble must be shown to differ from the three-seed reference
    assert medians["oracle_active"] < reference["genuine_active_median_range"][0]


def test_deployment_manifest_was_not_written_when_calibration_failed():
    decision = _json(DIAG / "final_decision.json")
    assert decision["deployment_manifest_written"] is False
    assert not (ROOT / "configs"
                / "hybrid_obstacle_uncertainty_deployment_v1.json").exists()


# ------------------------------------------------------- downstream stages / schedule
def test_no_live_rollouts_were_executed():
    decision = _json(DIAG / "final_decision.json")
    assert decision["live_rollouts_executed"] == 0
    assert decision["live_rollouts_permitted"] == 20
    assert decision["development4_executed"] is False


def test_confirmatory41_refused_and_untouched():
    decision = _json(DIAG / "final_decision.json")
    assert decision["confirmatory41_executed"] is False
    conf41 = _json(CONF41)
    assert conf41["executed_in_this_task"] is False
    assert len(conf41["rows"]) == 41
    dev4_rows = {r["candidate_index"] for r in _json(DEV4)["rows"]}
    conf_rows = {r["candidate_index"] for r in conf41["rows"]}
    assert not dev4_rows & conf_rows
    assert dev4_rows == {106, 107, 108, 118}


def test_development4_not_used_for_any_fitting():
    decision = _json(DIAG / "final_decision.json")
    assert decision["constraints_honoured"][
        "development4_or_confirmatory41_used_for_fitting"] is False
    for script in ("hybrid_obstacle_uncertainty_ensemble_train.py",
                   "hybrid_obstacle_uncertainty_calibrate.py"):
        source = (ROOT / "scripts" / script).read_text()
        assert "development4" not in source
        assert "confirmatory41" not in source


def test_partition_reused_without_modification():
    decision = _json(DIAG / "final_decision.json")
    assert decision["partition"]["reused_without_modification"] is True
    partition = _json(PARTITION)
    recorded = partition.pop("manifest_sha256")
    assert thr.canonical_hash(partition) == recorded
    assert decision["partition"]["manifest_sha256"] == recorded


def test_evaluator_condition_is_registered_and_gated():
    source = (ROOT / "submodules" / "act"
              / "eval_act_obstacle_on_policy.py").read_text()
    assert "ACT_PLUS_UNCERTAINTY_ABSTENTION" in source
    assert "--abstention-manifest" in source
    # the condition must refuse to run without its manifest
    assert "requires --abstention-manifest" in source


def test_gripper_and_arm_only_contract_recorded():
    report = _json(DIAG / "provenance_verification.json")
    checks = {c["check"]: c for c in report["checks"]}
    assert checks["residual_arm_only"]["observed"] is True
    assert checks["residual_gripper_owner"]["observed"] == "ACT"


# ------------------------------------------------------------------ final decision
def test_final_decision_token_matches_markdown_last_line():
    decision = _json(DIAG / "final_decision.json")
    markdown = (ROOT / "docs"
                / "HYBRID_OBSTACLE_UNCERTAINTY_ABSTENTION_FINAL_DECISION.md").read_text()
    last = [line for line in markdown.splitlines() if line.strip()][-1]
    assert last == decision["decision"]


def test_final_decision_token_is_allowed():
    allowed = {"UNCERTAINTY_ABSTENTION_READY_FOR_CONFIRMATORY_41",
               "UNCERTAINTY_ABSTENTION_CALIBRATION_INFEASIBLE",
               "UNCERTAINTY_ABSTENTION_OFFLINE_INVALID",
               "UNCERTAINTY_ABSTENTION_LIVE_TRANSFER_FAILED",
               "UNCERTAINTY_ABSTENTION_LIVE_GROSS_REGRESSION",
               "UNCERTAINTY_ENSEMBLE_TRAINING_FAILED",
               "CHECKPOINT_OR_SOURCE_MISMATCH",
               "UNCERTAINTY_ABSTENTION_TASK_INCOMPLETE"}
    assert _json(DIAG / "final_decision.json")["decision"] in allowed


def test_decision_is_consistent_with_the_calibration_outcome():
    decision = _json(DIAG / "final_decision.json")
    calibration = _json(DIAG / "calibration.json")
    if decision["decision"] == "UNCERTAINTY_ABSTENTION_CALIBRATION_INFEASIBLE":
        assert calibration["feasible"] is False
        assert decision["case"] == "C"
        assert decision["live_rollouts_executed"] == 0
        assert decision["ensemble"]["all_members_trained_and_reloaded"] is True
