"""Contract tests for the trajectory-aware threshold recalibration.

The failure this task exists to prevent is a threshold that looks calibrated because the
statistics used to fit it assumed independence they do not have. Most of these tests are
therefore about the *fitting procedure* rather than the number it produced.
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

from causal_parked_skin import threshold as thr

DIAG = ROOT / "diagnostics_output" / "hybrid_obstacle_reference_threshold"
CAL16 = ROOT / "configs" / "hybrid_obstacle_threshold_calibration16_v1.json"
THRESHOLD_MANIFEST = ROOT / "configs" / "hybrid_obstacle_reference_threshold_v1.json"
DEV4 = ROOT / "configs" / "hybrid_obstacle_controller_development4_v1.json"
CONF41 = ROOT / "configs" / "hybrid_obstacle_confirmatory41_v1.json"
STACK = ROOT / "configs" / "hybrid_safety_stack_v1.json"
PARKED_DECISION = (ROOT / "diagnostics_output" / "causal_parked_skin_reference_v1"
                   / "final_decision.json")


def _json(path):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ frozen model / seed
def test_provenance_all_checks_matched():
    report = _json(DIAG / "provenance_verification.json")
    assert report["all_matched"] is True
    assert report["failed"] == []
    assert report["check_count"] >= 40


def test_frozen_model_is_current_frame_only_seed_zero():
    report = _json(DIAG / "provenance_verification.json")
    frozen = report["frozen_model"]
    assert frozen["variant"] == "CURRENT_FRAME_ONLY"
    assert frozen["seed"] == 0
    assert frozen["config"]["variant"] == "CURRENT_FRAME_ONLY"
    assert frozen["config"]["seed"] == 0


def test_checkpoint_hash_matches_previous_decision():
    report = _json(DIAG / "provenance_verification.json")
    previous = _json(PARKED_DECISION)
    seed0 = next(c for c in previous["checkpoints"]
                 if c["variant"] == "CURRENT_FRAME_ONLY" and c["seed"] == 0)
    assert report["frozen_model"]["checkpoint_sha256"] == seed0["sha256"]
    actual = hashlib.sha256(
        Path(seed0["local_path"]).read_bytes()).hexdigest()
    assert actual == seed0["sha256"]


def test_seed_one_and_two_are_not_selected():
    """Seeds 1 and 2 had lower offline FPR; choosing one now would be post hoc."""
    manifest = _json(THRESHOLD_MANIFEST)
    assert manifest["model"]["seed"] == 0
    calibration = _json(DIAG / "threshold_calibration.json")
    previous = _json(PARKED_DECISION)
    other = [c["local_path"] for c in previous["checkpoints"]
             if c["variant"] == "CURRENT_FRAME_ONLY" and c["seed"] != 0]
    text = json.dumps(calibration)
    for path in other:
        assert path not in text, "a non-zero seed checkpoint entered the calibration"


def test_four_frame_history_is_rejected():
    manifest = _json(THRESHOLD_MANIFEST)
    assert manifest["model"]["history_frames"] == 1
    from hybrid_obstacle_reference_threshold_manifest import (
        ThresholdManifestError,
        load_threshold_manifest,
    )
    payload = _json(THRESHOLD_MANIFEST)
    payload["model"]["history_frames"] = 4
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = thr.canonical_hash(payload)
    scratch = DIAG / "_tmp_history_manifest.json"
    scratch.write_text(json.dumps(payload))
    try:
        with pytest.raises(ThresholdManifestError, match="history"):
            load_threshold_manifest(scratch, require_live_authorization=False)
    finally:
        scratch.unlink()


def test_full_causal_variant_is_rejected_by_the_loader():
    from hybrid_obstacle_reference_threshold_manifest import (
        ThresholdManifestError,
        load_threshold_manifest,
    )
    payload = _json(THRESHOLD_MANIFEST)
    payload["model"]["variant"] = "FULL_CAUSAL"
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = thr.canonical_hash(payload)
    scratch = DIAG / "_tmp_variant_manifest.json"
    scratch.write_text(json.dumps(payload))
    try:
        with pytest.raises(ThresholdManifestError, match="CURRENT_FRAME_ONLY"):
            load_threshold_manifest(scratch, require_live_authorization=False)
    finally:
        scratch.unlink()


# ---------------------------------------------------------------------- calibration16
def test_calibration16_membership_is_exactly_sixteen_episodes():
    manifest = _json(CAL16)
    assert manifest["episode_count"] == 16
    assert len(set(manifest["episodes"])) == 16
    assert set(manifest["source_partitions"]) == {"reference_calibration",
                                                  "reference_validation"}
    assert manifest["cluster_unit"] == "episode"


def test_calibration16_hash_is_reproducible():
    manifest = _json(CAL16)
    recorded = manifest.pop("manifest_sha256")
    assert thr.canonical_hash(manifest) == recorded


def test_consumed_offline_test_is_excluded_from_the_fit():
    manifest = _json(CAL16)
    assert manifest["consumed_diagnostic_partition_excluded"] == "offline_reference_test"
    for entry in manifest["trajectories"]:
        assert entry["partition"] != "offline_reference_test"


def test_calibration_script_fits_only_on_calibration_partitions():
    """Structural: the fit must not be able to read the consumed set."""
    source = (ROOT / "scripts"
              / "hybrid_obstacle_reference_threshold_calibrate.py").read_text()
    tree = ast.parse(source)
    fit_partitions = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CALIBRATION_PARTITIONS"
                for t in node.targets):
            fit_partitions = [e.value for e in node.value.elts]
    assert fit_partitions == ["reference_calibration", "reference_validation"]
    assert "offline_reference_test" not in fit_partitions


def test_audit_labels_the_consumed_set_correctly():
    audit = _json(DIAG / "threshold_audit.json")
    block = audit["diagnostic_set"]
    assert block["label"] == "reused_nonconfirmatory_diagnostic"
    assert block["used_for_threshold_fitting"] is False
    assert block["provides_final_readiness_gate"] is False


# ------------------------------------------------------------- trajectory-level metrics
def _trajectory(activity, active, cosine=None, hazard=True):
    n = len(activity)
    return thr.TrajectoryScores(
        trajectory_id="t", episode_id="e", distribution="EXPERT_RECONSTRUCTED",
        partition="p", hazard_present=hazard,
        activity=np.asarray(activity, dtype=float),
        oracle_active=np.asarray(active, dtype=bool),
        cosine=np.asarray(cosine if cosine is not None else np.ones(n), dtype=float),
        predicted_norm=np.ones(n), oracle_norm=np.ones(n),
        changed_true=np.ones(n, dtype=int), changed_predicted=np.ones(n, dtype=int),
        changed_hit=np.ones(n, dtype=int))


def test_trajectory_metrics_are_computed_without_pooling():
    trajectory = _trajectory([0.9, 0.1, 0.95, 0.2], [True, False, True, False])
    metrics = trajectory.metrics_at(0.5)
    assert metrics["active_recall"] == 1.0
    assert metrics["oracle_zero_false_positive_rate"] == 0.0
    assert metrics["oracle_active_frames"] == 2
    assert metrics["oracle_zero_frames"] == 2


def test_false_positive_rate_uses_zero_frames_only():
    trajectory = _trajectory([0.9, 0.9, 0.9, 0.1], [True, False, False, False])
    metrics = trajectory.metrics_at(0.5)
    assert metrics["oracle_zero_false_positive_rate"] == pytest.approx(2 / 3)
    assert metrics["active_recall"] == 1.0


def test_max_consecutive_run_is_measured_not_counted():
    assert thr.max_true_run([False, True, True, False, True]) == 2
    assert thr.max_true_run([True] * 7) == 7
    assert thr.max_true_run([False, False]) == 0
    assert thr.max_true_run([]) == 0


def test_persistence_after_oracle_detects_trailing_activation():
    assert thr.persists_after_oracle([False, True, False, True],
                                     [False, True, False, False]) is True
    assert thr.persists_after_oracle([False, True, False, False],
                                     [False, True, False, False]) is False
    # a hazard-absent trajectory has no oracle activity at all: any firing persists
    assert thr.persists_after_oracle([False, True], [False, False]) is True


def test_activity_probability_is_the_pixel_maximum():
    field = np.zeros((2, 40, 8, 8))
    field[0, 3, 4, 5] = 0.87
    field[1, 0, 0, 0] = 0.42
    activity = thr.activity_probability(field)
    assert activity.tolist() == [pytest.approx(0.87), pytest.approx(0.42)]


# ------------------------------------------------------------------- cluster bootstrap
def test_bootstrap_resamples_clusters_not_frames():
    """A cluster bootstrap on identical clusters must produce zero spread."""
    values = np.tile(np.array([[0.5]]), (8, 1))
    bound = thr.cluster_bootstrap_upper_bound(values, replicates=500, seed=1)
    assert bound[0] == pytest.approx(0.5)


def test_bootstrap_is_deterministic_for_a_seed():
    rng = np.random.default_rng(0)
    values = rng.random((16, 5))
    a = thr.cluster_bootstrap_upper_bound(values, replicates=1000, seed=thr.BOOTSTRAP_SEED)
    b = thr.cluster_bootstrap_upper_bound(values, replicates=1000, seed=thr.BOOTSTRAP_SEED)
    assert np.array_equal(a, b)


def test_bootstrap_changes_with_a_different_seed():
    rng = np.random.default_rng(0)
    values = rng.random((16, 5))
    a = thr.cluster_bootstrap_upper_bound(values, replicates=1000, seed=1)
    b = thr.cluster_bootstrap_upper_bound(values, replicates=1000, seed=2)
    assert not np.array_equal(a, b)


def test_upper_bound_is_one_sided_and_above_the_mean():
    rng = np.random.default_rng(3)
    values = rng.random((16, 4)) * 0.05
    bound = thr.cluster_bootstrap_upper_bound(values, replicates=4000, seed=7)
    assert (bound >= values.mean(axis=0) - 1e-9).all(), "upper bound below the mean"
    lower = np.quantile(values.mean(axis=0), 0.0)
    assert (bound >= lower).all()


def test_cluster_bootstrap_is_wider_than_a_frame_bootstrap_would_be():
    """The whole point: clustering must not understate uncertainty.

    Eight clusters that disagree strongly give a wide bound; pretending each cluster's
    rows were independent observations would give a much tighter one.
    """
    clusters = np.array([[0.0], [0.0], [0.0], [0.0], [0.08], [0.08], [0.08], [0.08]])
    clustered = thr.cluster_bootstrap_upper_bound(clusters, replicates=5000, seed=11)[0]
    naive = np.quantile(
        np.random.default_rng(11).choice(clusters.ravel(), size=(5000, 1000)).mean(axis=1),
        0.95)
    assert clustered > naive


def test_bootstrap_replicate_count_and_seed_are_committed():
    assert thr.BOOTSTRAP_REPLICATES >= 10_000
    assert isinstance(thr.BOOTSTRAP_SEED, int)
    report = _json(DIAG / "threshold_calibration.json")
    assert report["bootstrap"]["replicates"] == thr.BOOTSTRAP_REPLICATES
    assert report["bootstrap"]["seed"] == thr.BOOTSTRAP_SEED
    assert report["bootstrap"]["one_sided"] is True
    assert report["bootstrap"]["resampled_unit"].startswith("episode")


# ------------------------------------------------------------------- feasibility rule
def test_feasibility_contract_matches_the_predeclared_values():
    report = _json(DIAG / "threshold_calibration.json")
    contract = report["feasibility_contract"]
    assert contract["median_active_recall_min"] == 0.80
    assert contract["median_active_cosine_min"] == 0.75
    assert contract["median_positive_cosine_fraction_min"] == 0.85
    assert contract["bootstrap_upper_fpr_max"] == 0.02
    assert contract["hazard_absent_trajectory_fpr_max"] == 0.05
    assert contract["max_consecutive_false_positive_run"] == 2
    assert contract["persistent_activation_allowed"] is False


def test_two_percent_target_was_not_loosened():
    assert thr.MAX_BOOTSTRAP_UPPER_FPR == 0.02
    manifest = _json(THRESHOLD_MANIFEST)
    assert manifest["calibration"]["feasibility_contract"][
        "bootstrap_upper_fpr_max"] == 0.02


def test_selection_is_lexicographic():
    thresholds = np.array([0.1, 0.2, 0.3])
    upper = np.array([0.01, 0.01, 0.005])
    recall = np.array([0.9, 0.95, 0.85])
    feasible = np.array([True, True, True])
    # lowest bound wins outright
    assert thr.select_threshold(feasible, thresholds, upper, recall) == 2
    # tie on bound -> highest recall
    assert thr.select_threshold(feasible, thresholds,
                                np.array([0.01, 0.01, 0.01]), recall) == 1
    # tie on both -> highest threshold
    assert thr.select_threshold(feasible, thresholds, np.array([0.01] * 3),
                                np.array([0.9] * 3)) == 2


def test_select_threshold_returns_none_when_nothing_is_feasible():
    assert thr.select_threshold(np.array([False, False]), np.array([0.1, 0.2]),
                                np.array([1.0, 1.0]), np.array([0.1, 0.1])) is None


def test_retired_threshold_is_not_reused():
    report = _json(DIAG / "threshold_calibration.json")
    assert report["retired_threshold"]["reused"] is False
    assert report["selected"]["threshold"] != report["retired_threshold"]["value"]


# ------------------------------------------------------------------- manifest / magnitude
def test_threshold_manifest_strict_loading_rejects_a_tampered_hash():
    from hybrid_obstacle_reference_threshold_manifest import (
        ThresholdManifestError,
        load_threshold_manifest,
    )
    payload = _json(THRESHOLD_MANIFEST)
    payload["activation"]["selected_threshold"] = 0.5
    scratch = DIAG / "_tmp_tampered.json"
    scratch.write_text(json.dumps(payload))
    try:
        with pytest.raises(ThresholdManifestError, match="self-hash"):
            load_threshold_manifest(scratch, require_live_authorization=False)
    finally:
        scratch.unlink()


def test_unauthorized_manifest_is_refused_for_live_use():
    from hybrid_obstacle_reference_threshold_manifest import (
        ThresholdManifestError,
        load_threshold_manifest,
    )
    manifest = _json(THRESHOLD_MANIFEST)
    assert manifest["authorized_for_live"] is False
    with pytest.raises(ThresholdManifestError, match="not authorized"):
        load_threshold_manifest(THRESHOLD_MANIFEST)


def test_magnitude_support_bound_is_unchanged():
    manifest = _json(THRESHOLD_MANIFEST)
    block = manifest["magnitude_support"]
    assert block["recalculated_in_this_task"] is False
    assert block["post_hoc_clipping"] is False
    assert "0 <= predicted_parked <= current_closeness <= 1" in block["guarantee"]


def test_threshold_changes_only_the_activation_decision():
    """The gate is applied after the field and differential are produced."""
    manifest = _json(THRESHOLD_MANIFEST)
    assert manifest["activation"]["gate_variable"] == "frame activity probability"
    assert manifest["activation"]["gates_on_differential_norm_alone"] is False
    # the frozen construction is quoted verbatim and carries no threshold term
    assert "threshold" not in manifest["magnitude_support"]["construction"]


def test_controller_constants_are_unchanged():
    manifest = _json(THRESHOLD_MANIFEST)
    stack = _json(STACK)["residual_controller"]
    for key in ("gain", "decay_per_second", "ema", "max_deviation_rad_per_joint",
                "arm_only", "gripper_owner"):
        assert manifest["controller"][key] == stack[key]
    assert manifest["controller"]["changed_in_this_task"] is False
    assert manifest["controller"]["arm_only"] is True
    assert manifest["controller"]["gripper_owner"] == "ACT"


def test_residual_applies_after_temporal_aggregation():
    manifest = _json(THRESHOLD_MANIFEST)
    assert "after aggregation" in manifest["act"]["temporal_aggregation"]


def test_no_privileged_field_is_a_deployable_input():
    contract = _json(ROOT / "diagnostics_output" / "causal_parked_skin_reference_v1"
                     / "input_contract_audit.json")
    inputs = set(contract["contract"]["model_input_fields"])
    for prohibited in ("parked_closeness", "oracle_dq", "changed_pixel_mask",
                       "parked_head", "oracle_active", "current_head"):
        assert prohibited not in inputs


# --------------------------------------------------------------------- inference / schema
def test_frozen_inference_is_repeatable():
    audit = _json(DIAG / "threshold_audit.json")
    stability = audit["inference_stability"]
    assert stability["repeats"] >= 20
    assert stability["stable"] is True
    assert stability["activity_max_abs_delta"] <= 1e-7
    assert stability["parked_field_max_abs_delta"] <= 1e-7
    assert audit["head_stability"]["head_differential_max_abs_delta"] <= 1e-7
    assert stability["training_kernels_invoked"] is False


def test_live_log_schema_is_complete():
    from hybrid_obstacle_reference_threshold_manifest import (
        EXCLUDED_LOG_FIELDS,
        LIVE_LOG_SCHEMA,
    )
    required = {"qpos", "qvel", "nominal_act_action", "aggregated_act_action",
                "activity_probability", "selected_threshold", "activation_decision",
                "ungated_differential", "magnitude_capped_differential",
                "privileged_oracle_differential", "deployable_oracle_cosine",
                "deployable_oracle_norm_ratio", "false_active_sequence_length",
                "filtered_correction", "accumulated_correction", "executed_action",
                "gripper_command", "contact_classes", "penetration", "task_success"}
    assert required <= set(LIVE_LOG_SCHEMA)
    assert "minimum_clearance_m" in EXCLUDED_LOG_FIELDS
    assert "minimum_clearance_m" not in LIVE_LOG_SCHEMA


def test_manifest_records_the_schema_and_the_exclusion():
    manifest = _json(THRESHOLD_MANIFEST)
    assert "minimum_clearance_m" in manifest["excluded_log_fields"]
    assert "mj_geomDistance" in manifest["excluded_field_reason"]


# ------------------------------------------------------------------- schedule / refusal
def test_live_schedule_is_development4_only():
    manifest = _json(THRESHOLD_MANIFEST)
    schedule = manifest["live_schedule"]
    assert schedule["rows"] == [106, 107, 108, 118]
    assert schedule["repeats_per_row"] == 5
    assert schedule["total_rollouts"] == 20
    dev4 = _json(DEV4)
    assert [r["candidate_index"] for r in dev4["rows"]] == schedule["rows"]


def test_confirmatory41_is_refused_and_untouched():
    manifest = _json(THRESHOLD_MANIFEST)
    assert manifest["live_schedule"]["confirmatory41_permitted"] is False
    conf41 = _json(CONF41)
    assert conf41["executed_in_this_task"] is False
    assert len(conf41["rows"]) == 41
    # no confirmatory row may appear in the development schedule
    conf_rows = {r["candidate_index"] for r in conf41["rows"]}
    assert not conf_rows & set(manifest["live_schedule"]["rows"])


def test_no_rollouts_were_executed_when_blocked():
    manifest = _json(THRESHOLD_MANIFEST)
    audit = _json(DIAG / "threshold_audit.json")
    if not audit["blocking_checks"]["passed"]:
        assert manifest["live_schedule"]["executed"] == 0
        assert manifest["authorized_for_live"] is False


def test_msaa_and_sensor_order_unchanged():
    manifest = _json(THRESHOLD_MANIFEST)
    stack = _json(STACK)
    assert manifest["offsamples"] == 4
    assert manifest["sensor_count"] == 40
    assert manifest["sensor_order_sha256"] == stack["sensor_contract"][
        "sensor_order_hash"]


# ------------------------------------------------------------------------ final decision
def test_final_decision_token_matches_markdown_last_line():
    decision = _json(DIAG / "final_decision.json")
    markdown = (ROOT / "docs"
                / "HYBRID_OBSTACLE_REFERENCE_THRESHOLD_FINAL_DECISION.md").read_text()
    last = [line for line in markdown.splitlines() if line.strip()][-1]
    assert last == decision["decision"]


def test_final_decision_token_is_allowed():
    allowed = {
        "REFERENCE_THRESHOLD_READY_FOR_CONFIRMATORY_41",
        "REFERENCE_THRESHOLD_CALIBRATION_INFEASIBLE",
        "REFERENCE_THRESHOLD_TRANSFER_FAILED",
        "REFERENCE_MODEL_LIVE_GROSS_REGRESSION",
        "REFERENCE_MODEL_INFERENCE_UNSTABLE",
        "CHECKPOINT_OR_SOURCE_MISMATCH",
        "REFERENCE_THRESHOLD_QUALIFICATION_INCOMPLETE",
    }
    assert _json(DIAG / "final_decision.json")["decision"] in allowed


def test_decision_is_consistent_with_the_blocking_checks():
    decision = _json(DIAG / "final_decision.json")
    audit = _json(DIAG / "threshold_audit.json")
    calibration = _json(DIAG / "threshold_calibration.json")
    if decision["decision"] == "REFERENCE_THRESHOLD_TRANSFER_FAILED":
        assert calibration["feasible"] is True, "transfer failure presumes a fitted threshold"
        assert audit["blocking_checks"]["passed"] is False
        assert decision["live_rollouts_executed"] == 0
        assert decision["confirmatory41_executed"] is False
