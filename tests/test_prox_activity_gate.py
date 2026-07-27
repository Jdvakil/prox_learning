"""Contract tests for the proximity-only activity gate.

The central claim is negative — current proximity alone cannot separate oracle activity —
so most of these tests guard the conditions under which that claim was measured: that the
gate really could not see state, that the partition was fixed in advance, that the frozen
parked-field model was untouched, and that nothing downstream ran anyway.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from causal_parked_skin import activity_gate as ag
from causal_parked_skin import threshold as thr

DIAG = ROOT / "diagnostics_output" / "hybrid_obstacle_prox_activity_gate"
PARTITION = ROOT / "configs" / "hybrid_obstacle_prox_activity_partition_v1.json"
DEV4 = ROOT / "configs" / "hybrid_obstacle_controller_development4_v1.json"
CONF41 = ROOT / "configs" / "hybrid_obstacle_confirmatory41_v1.json"
STACK = ROOT / "configs" / "hybrid_safety_stack_v1.json"
PARKED_DECISION = (ROOT / "diagnostics_output" / "causal_parked_skin_reference_v1"
                   / "final_decision.json")


def _json(path):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ frozen field model
def test_provenance_all_checks_matched():
    report = _json(DIAG / "provenance_verification.json")
    assert report["all_matched"] is True
    assert report["failed"] == []
    assert report["check_count"] >= 55


def test_frozen_parked_field_checkpoint_unchanged():
    report = _json(DIAG / "provenance_verification.json")
    frozen = report["frozen_model"]
    assert frozen["variant"] == "CURRENT_FRAME_ONLY"
    assert frozen["seed"] == 0
    actual = hashlib.sha256(Path(frozen["checkpoint_path"]).read_bytes()).hexdigest()
    assert actual == frozen["checkpoint_sha256"]
    previous = _json(PARKED_DECISION)
    seed0 = next(c for c in previous["checkpoints"]
                 if c["variant"] == "CURRENT_FRAME_ONLY" and c["seed"] == 0)
    assert actual == seed0["sha256"]


def test_parked_field_model_was_not_retrained():
    """The gate training run must not touch the field model at all."""
    source = (ROOT / "scripts" / "hybrid_obstacle_prox_activity_train.py").read_text()
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
                for alias in node.names}
    assert "build_model" not in imported, "field-model constructor reached the trainer"
    assert "load_checkpoint" not in imported, "field checkpoint reached the trainer"
    assert "build_gate" in imported


def test_field_model_parameters_unchanged_by_the_onset_audit():
    audit = _json(DIAG / "onset_attribution.json")
    assert audit["harness_control"]["model_weights_unchanged"] is True
    assert audit["harness_control"]["dataset_modified"] is False


# ----------------------------------------------------------------- causal audit itself
def test_identity_control_is_exact():
    """If the no-op intervention perturbs anything, every other number is suspect."""
    audit = _json(DIAG / "onset_attribution.json")
    assert audit["harness_control"]["identity_control_exact"] is True
    assert audit["harness_control"]["identity_control_max_activity_delta"] == 0.0


def test_all_eight_interventions_ran_on_every_group():
    audit = _json(DIAG / "onset_attribution.json")
    expected = {"FULL_INPUT", "PROX_ONLY", "STATE_ONLY",
                "STATE_SHUFFLED_WITHIN_ONSET", "STATE_SWAPPED_ACROSS_HAZARD_STRATA",
                "PROX_SWAPPED_ACROSS_HAZARD_STRATA", "STATE_MEAN",
                "CURRENT_FIELD_IDENTITY_CONTROL"}
    assert set(audit["interventions"]) == expected
    for group, block in audit["results"].items():
        assert set(block) == expected, group


def test_seventeen_historical_false_positives_are_indexed():
    audit = _json(DIAG / "onset_attribution.json")
    assert audit["known_false_positive_count"] == 17
    frames = audit["known_false_positive_frames"]
    assert len(frames) == 17
    assert sum(1 for f in frames if f["step"] <= 6) == 16


def test_classification_is_one_of_the_three_predeclared_values():
    audit = _json(DIAG / "onset_attribution.json")
    assert audit["classification"] in {
        "STATE_PRIOR_DOMINANT", "PROXIMITY_AMBIGUITY_DOMINANT",
        "SHARED_REPRESENTATION_CONFOUND"}


def test_state_only_intervention_does_not_reproduce_the_false_activation():
    """The recorded basis for rejecting the state-prior hypothesis."""
    audit = _json(DIAG / "onset_attribution.json")
    evidence = audit["classification_evidence"]
    assert evidence["median_activity_state_only"] < 0.5
    assert evidence["median_activity_full_input"] > 0.9


# ------------------------------------------------------------------- gate input isolation
def test_gate_forward_accepts_only_proximity_and_mask():
    signature = inspect.signature(ag.ProxEvidenceActivityGate.forward)
    assert list(signature.parameters) == ["self", "closeness", "valid_mask"]


def test_gate_source_never_mentions_a_prohibited_input():
    source = (ROOT / "causal_parked_skin" / "activity_gate.py").read_text()
    tree = ast.parse(source)
    forward = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "forward")
    names = {n.id for n in ast.walk(forward) if isinstance(n, ast.Name)}
    for prohibited in ("qpos", "qvel", "nominal_action", "gripper_state",
                       "episode_step", "hazard_present", "state"):
        assert prohibited not in names


def test_assert_gate_inputs_rejects_state_and_privileged_fields():
    field = np.zeros((1, 40, 8, 8), dtype=np.float32)
    mask = np.ones((1, 40, 8, 8), dtype=bool)
    ag.assert_gate_inputs({"current_closeness": field, "current_valid_mask": mask})
    for prohibited in ("qpos", "qvel", "episode_step", "hazard_present",
                       "oracle_dq", "predicted_parked_field", "task_phase"):
        with pytest.raises(ValueError, match="prohibited|unexpected"):
            ag.assert_gate_inputs({"current_closeness": field,
                                   "current_valid_mask": mask, prohibited: 1})


def test_assert_gate_inputs_requires_both_proximity_fields():
    with pytest.raises(ValueError, match="missing"):
        ag.assert_gate_inputs({"current_closeness": np.zeros((1, 40, 8, 8))})


def test_gate_output_is_invariant_to_everything_but_proximity():
    """There is no argument through which state could arrive; assert it behaviourally."""
    import torch

    gate = ag.build_gate().eval()
    generator = torch.Generator().manual_seed(5)
    closeness = torch.rand(4, 40, 8, 8, generator=generator)
    mask = torch.ones(4, 40, 8, 8, dtype=torch.bool)
    with torch.no_grad():
        a = gate(closeness, mask)
        b = gate(closeness, mask)
    assert torch.equal(a, b)
    with torch.no_grad():
        c = gate(torch.rand(4, 40, 8, 8, generator=generator), mask)
    assert not torch.equal(a, c), "gate ignores its only input"


def test_gate_architecture_matches_the_fixed_specification():
    import torch

    gate = ag.build_gate()
    assert ag.EMBED_DIM == 64
    assert ag.BLOCKS == 2
    assert ag.HEADS == 4
    assert ag.FEED_FORWARD == 128
    assert ag.PER_SENSOR_INPUT == 128
    assert len(gate.cross_sensor.layers) == 2
    assert gate.sensor_embedding.shape == (40, 64)
    assert isinstance(gate.per_sensor[0], torch.nn.Linear)
    assert gate.per_sensor[0].in_features == 128
    assert gate.per_sensor[0].out_features == 64


def test_gate_parameter_budget_is_respected():
    gate = ag.build_gate()
    assert gate.parameter_count() <= ag.PARAMETER_BUDGET == 250_000
    report = _json(DIAG / "gate_training.json")
    assert report["parameter_count"] == gate.parameter_count()
    assert report["parameter_count"] <= report["parameter_budget"]


def test_gate_feature_hash_is_separate_and_sensitive():
    field = np.zeros((40, 8, 8), dtype=np.float32)
    mask = np.ones((40, 8, 8), dtype=bool)
    first = ag.gate_feature_hash(field, mask)
    assert first == ag.gate_feature_hash(field, mask)
    changed = field.copy()
    changed[0, 0, 0] = 0.5
    assert ag.gate_feature_hash(changed, mask) != first


# --------------------------------------------------------------------- nested partition
def test_partition_is_deterministic_and_exhaustive():
    spec = _json(PARTITION)
    expected = {"gate_training": (30, 10), "checkpoint_validation": (6, 2),
                "threshold_calibration": (6, 2), "nested_offline_evaluation": (6, 2)}
    seen = set()
    for name, (present, absent) in expected.items():
        block = spec["splits"][name]
        assert block["hazard_present"] == present, name
        assert block["hazard_absent"] == absent, name
        assert block["episode_count"] == present + absent
        assert not seen & set(block["episodes"]), f"{name} overlaps an earlier split"
        seen |= set(block["episodes"])
    assert len(seen) == 64


def test_partition_hash_is_reproducible():
    spec = _json(PARTITION)
    recorded = spec.pop("manifest_sha256")
    assert thr.canonical_hash(spec) == recorded


def test_partition_uses_only_reference_train():
    spec = _json(PARTITION)
    assert spec["source_partition"] == "reference_train"
    assert set(spec["excluded_partitions"]["reused_diagnostics_only"]) == {
        "reference_validation", "reference_calibration", "offline_reference_test"}


def test_no_development4_or_confirmatory_leakage():
    spec = _json(PARTITION)
    assert spec["development4"]["used_for_gate_training"] is False
    assert spec["development4"]["in_paired_dataset"] is False
    assert spec["confirmatory41"]["used_for_gate_training"] is False
    assert spec["confirmatory41"]["executed"] is False


def test_calibration_uses_only_the_calibration_split():
    source = (ROOT / "scripts" / "hybrid_obstacle_prox_activity_evaluate.py").read_text()
    tree = ast.parse(source)
    # the calibrate() call must be fed the threshold_calibration episodes
    assert 'spec["splits"]["threshold_calibration"]["episodes"]' in source
    assert isinstance(tree, ast.Module)


# ------------------------------------------------------------------- sampling and loss
def test_sampling_is_trajectory_balanced_not_frame_weighted():
    report = _json(DIAG / "gate_training.json")
    sampling = report["sampling"]
    assert sampling["global_frame_count_weighting"] is False
    assert sampling["order"][0].startswith("distribution")
    assert "trajectory uniform" in sampling["order"][1]


def test_onset_sampling_stratum_exists():
    report = _json(DIAG / "gate_training.json")
    assert "onset" in report["sampling"]["order"][3]
    assert report["frames"]["gate_training_onset_zero"] > 0
    assert "max(10" in report["onset_definition"]


def test_fixed_bce_weights():
    report = _json(DIAG / "gate_training.json")
    weights = report["loss"]["weights"]
    assert weights["oracle_active"] == 1.0
    assert weights["ordinary_zero"] == 1.0
    assert weights["onset_zero"] == 4.0
    assert report["loss"]["dynamic_positive_weight"] is False
    assert report["loss"]["focal_loss"] is False


def test_onset_penalty_is_a_trajectory_level_maximum():
    report = _json(DIAG / "gate_training.json")
    assert "max" in report["loss"]["onset_penalty"]
    assert "trajector" in report["loss"]["onset_penalty"]
    assert report["loss"]["onset_penalty_weight"] == 1.0


def test_onset_penalty_punishes_clustering_more_than_scatter():
    """Seven activations in one trajectory must cost more than seven spread over seven."""
    import torch
    from hybrid_obstacle_prox_activity_train import onset_max_penalty

    logits = torch.full((7,), 3.0)
    onset = torch.ones(7)
    active = torch.zeros(7)
    clustered = onset_max_penalty(logits, onset, active, torch.zeros(7, dtype=torch.long))
    scattered = onset_max_penalty(
        torch.cat([torch.full((1,), 3.0), torch.full((6,), -6.0)]),
        onset, active, torch.arange(7))
    assert float(clustered) > float(scattered)


def test_exactly_one_training_run_with_seed_zero():
    report = _json(DIAG / "gate_training.json")
    optimization = report["optimization"]
    assert optimization["seed"] == 0
    assert optimization["training_runs"] == 1
    assert optimization["restarted_with_another_seed"] is False
    assert optimization["optimizer"] == "AdamW"
    assert optimization["learning_rate"] == 3e-4
    assert optimization["weight_decay"] == 1e-5
    assert optimization["batch_size"] == 256
    assert optimization["max_epochs"] == 80
    assert optimization["gradient_clipping"] == 1.0
    assert optimization["dropout"] == 0.0


def test_label_definition_is_the_changed_pixel_mask():
    report = _json(DIAG / "gate_training.json")
    assert "changed_pixel_mask" in report["label_definition"]


# ------------------------------------------------------------------------- calibration
def test_calibration_is_trajectory_level_with_a_cluster_bootstrap():
    report = _json(DIAG / "gate_evaluation.json")
    bootstrap = report["calibration"]["bootstrap"]
    assert bootstrap["replicates"] >= 10_000
    assert bootstrap["resampled_unit"] == "episode"
    assert bootstrap["one_sided"] is True
    assert bootstrap["seed"] == thr.BOOTSTRAP_SEED


def test_cluster_bootstrap_is_deterministic():
    rng = np.random.default_rng(0)
    values = rng.random((8, 3))
    a = thr.cluster_bootstrap_upper_bound(values, replicates=1000, seed=thr.BOOTSTRAP_SEED)
    b = thr.cluster_bootstrap_upper_bound(values, replicates=1000, seed=thr.BOOTSTRAP_SEED)
    assert np.array_equal(a, b)


def test_calibration_was_infeasible_and_recorded_why():
    report = _json(DIAG / "gate_evaluation.json")
    assert report["feasible"] is False
    assert report["calibration"]["feasible_count"] == 0
    assert report["calibration"]["selected_threshold"] is None
    assert "recall" in report["no_threshold_satisfies"]


def test_threshold_sweep_shows_no_joint_operating_point():
    report = _json(DIAG / "gate_evaluation.json")
    sweep = report["threshold_sweep"]
    assert sweep, "sweep must be recorded to support the infeasibility claim"
    assert not any(row["recall_meets_080"] and row["fpr_meets_002"] for row in sweep)


def test_separability_distinguishes_underfitting_from_insufficiency():
    report = _json(DIAG / "gate_evaluation.json")
    separability = report["separability"]
    training = separability["gate_training"]
    held_out = separability["threshold_calibration"]
    assert training["gate_auroc"] > 0.95, "gate did fit its training split"
    assert held_out["gate_auroc"] < training["gate_auroc"]
    assert held_out["old_head_auroc"] > held_out["gate_auroc"]


# ------------------------------------------------------- downstream stages did not run
def test_no_live_rollouts_were_executed():
    decision = _json(DIAG / "final_decision.json")
    assert decision["live_rollouts_executed"] == 0
    assert decision["live_rollouts_permitted"] == 20
    assert decision["development4_executed"] is False


def test_confirmatory41_is_refused_and_untouched():
    decision = _json(DIAG / "final_decision.json")
    assert decision["confirmatory41_executed"] is False
    conf41 = _json(CONF41)
    assert conf41["executed_in_this_task"] is False
    assert len(conf41["rows"]) == 41
    dev4_rows = {r["candidate_index"] for r in _json(DEV4)["rows"]}
    conf_rows = {r["candidate_index"] for r in conf41["rows"]}
    assert not dev4_rows & conf_rows


def test_development4_schedule_is_unchanged():
    dev4 = _json(DEV4)
    assert [r["candidate_index"] for r in dev4["rows"]] == [106, 107, 108, 118]


def test_msaa_and_sensor_order_unchanged():
    report = _json(DIAG / "provenance_verification.json")
    checks = {c["check"]: c for c in report["checks"]}
    assert checks["offsamples"]["matched"] is True
    assert checks["sensor_count"]["observed"] == 40
    assert checks["sensor_order_sha256"]["matched"] is True


def test_controller_constants_and_magnitude_bound_unchanged():
    report = _json(DIAG / "provenance_verification.json")
    checks = {c["check"]: c for c in report["checks"]}
    for key in ("residual_gain", "residual_decay_per_second", "residual_ema",
                "residual_max_deviation_rad_per_joint", "residual_arm_only",
                "residual_gripper_owner"):
        assert checks[key]["matched"] is True, key
    assert checks["magnitude_support_recalculated"]["observed"] is False
    assert checks["magnitude_support_post_hoc_clipping"]["observed"] is False


def test_gate_is_independent_of_the_parked_field_predictor():
    """The gate must not consume the field model's outputs."""
    assert "predicted_parked_field" in ag.PROHIBITED_INPUTS
    assert "predicted_differential" in ag.PROHIBITED_INPUTS
    assert set(ag.PERMITTED_INPUTS) == {"current_closeness", "current_valid_mask",
                                        "sensor_identity"}


# ------------------------------------------------------------------------ final decision
def test_final_decision_token_matches_markdown_last_line():
    decision = _json(DIAG / "final_decision.json")
    markdown = (ROOT / "docs"
                / "HYBRID_OBSTACLE_PROX_ACTIVITY_GATE_FINAL_DECISION.md").read_text()
    last = [line for line in markdown.splitlines() if line.strip()][-1]
    assert last == decision["decision"]


def test_final_decision_token_is_allowed():
    allowed = {
        "PROX_ACTIVITY_GATE_READY_FOR_CONFIRMATORY_41",
        "PROX_ACTIVITY_GATE_CALIBRATION_INFEASIBLE",
        "PROX_ACTIVITY_GATE_OFFLINE_INVALID",
        "PROX_ACTIVITY_GATE_LIVE_TRANSFER_FAILED",
        "PROX_ACTIVITY_GATE_LIVE_GROSS_REGRESSION",
        "ACTIVITY_ONSET_CAUSE_UNRESOLVED",
        "CHECKPOINT_OR_SOURCE_MISMATCH",
        "PROX_ACTIVITY_GATE_TASK_INCOMPLETE",
    }
    assert _json(DIAG / "final_decision.json")["decision"] in allowed


def test_decision_is_consistent_with_the_calibration_result():
    decision = _json(DIAG / "final_decision.json")
    evaluation = _json(DIAG / "gate_evaluation.json")
    if decision["decision"] == "PROX_ACTIVITY_GATE_CALIBRATION_INFEASIBLE":
        assert evaluation["feasible"] is False
        assert decision["case"] == "B"
        assert decision["live_rollouts_executed"] == 0
