"""Contract tests for the full-data three-seed joint gate.

The conclusion is a transfer failure, so these tests guard the conditions the failure was
measured under: the exact seed roster, the exact agreement implementation, that the bootstrap
ensemble is refused, that the recall floor never moved, and that the historical frames and
development4 never touched the fit.
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

from causal_parked_skin import joint_gate as jg
from causal_parked_skin import threshold as thr

DIAG = ROOT / "diagnostics_output" / "hybrid_obstacle_full_seed_joint_gate"
MANIFEST = ROOT / "configs" / "hybrid_obstacle_full_seed_joint_gate_v1.json"
PARTITION = ROOT / "configs" / "hybrid_obstacle_prox_activity_partition_v1.json"
BOOTSTRAP = ROOT / "configs" / "hybrid_obstacle_uncertainty_ensemble_v1.json"
DATASET = ROOT / "configs" / "hybrid_obstacle_parked_skin_supervision_v1.json"
CONF41 = ROOT / "configs" / "hybrid_obstacle_confirmatory41_v1.json"
DEV4 = ROOT / "configs" / "hybrid_obstacle_controller_development4_v1.json"
PARKED_DECISION = (ROOT / "diagnostics_output" / "causal_parked_skin_reference_v1"
                   / "final_decision.json")
OLD_ACTIVITY_THRESHOLD = 0.99960857629776


def _json(path):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------- frozen artifacts
def test_provenance_all_checks_matched():
    report = _json(DIAG / "provenance_verification.json")
    assert report["all_matched"] is True
    assert report["failed"] == []
    assert report["check_count"] >= 65


def test_exactly_full_data_seeds_zero_one_two():
    manifest = _json(MANIFEST)
    assert manifest["deployment_seed"] == 0
    assert manifest["uncertainty_seeds"] == [1, 2]
    assert len(manifest["seed_checkpoints"]) == 3
    previous = _json(PARKED_DECISION)
    recorded = {c["seed"]: c["sha256"] for c in previous["checkpoints"]
                if c["variant"] == "CURRENT_FRAME_ONLY"}
    for seed, digest in enumerate(manifest["seed_checkpoint_sha256"]):
        assert digest == recorded[seed], f"seed {seed} is not the full-data model"


def test_seed_checkpoints_unchanged_on_disk():
    manifest = _json(MANIFEST)
    for path, digest in zip(manifest["seed_checkpoints"],
                            manifest["seed_checkpoint_sha256"]):
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def test_all_three_seeds_share_one_configuration_apart_from_the_seed():
    import torch

    previous = _json(PARKED_DECISION)
    shapes = set()
    for record in previous["checkpoints"]:
        if record["variant"] != "CURRENT_FRAME_ONLY":
            continue
        payload = torch.load(record["local_path"], map_location="cpu",
                             weights_only=False)
        config = {k: v for k, v in payload["config"].items() if k != "seed"}
        shapes.add(thr.canonical_hash(config))
    assert len(shapes) == 1, "the three seeds do not share a training configuration"


def test_seed_roster_validation_rejects_anything_else():
    jg.validate_seed_roster([0, 1, 2])
    for bad in ([0, 1], [1, 0, 2], [0, 1, 2, 3], [0, 2, 1]):
        with pytest.raises(jg.JointGateManifestError):
            jg.validate_seed_roster(bad)


def test_seed0_is_the_sole_deployment_predictor():
    decision = _json(DIAG / "final_decision.json")
    assert decision["owner_decisions_honoured"]["seed0_sole_deployment_predictor"] is True
    assert decision["predictions_averaged"] is False
    manifest = _json(MANIFEST)
    assert manifest["averaging_permitted"] is False
    assert manifest["uncertainty_seed_may_replace_seed0"] is False


# ------------------------------------------------------------- bootstrap ensemble refusal
def test_bootstrap_ensemble_is_refused_in_this_mode():
    bootstrap = _json(BOOTSTRAP)
    with pytest.raises(jg.JointGateManifestError, match="cluster_omission_variance"):
        jg.assert_not_bootstrap(bootstrap)


def test_a_bootstrap_member_record_is_refused():
    with pytest.raises(jg.JointGateManifestError, match="bootstrap member"):
        jg.assert_not_bootstrap({"member_records": [{"bootstrap_seed": 20260731}]})


def test_a_full_seed_manifest_is_accepted():
    jg.assert_not_bootstrap(_json(MANIFEST))


def test_bootstrap_artifacts_retained_but_excluded():
    decision = _json(DIAG / "final_decision.json")
    block = decision["bootstrap_ensemble"]
    assert block["disposition"] == (
        "invalid_for_deployment_uncertainty_due_to_cluster_omission_variance")
    assert block["checkpoints_deleted"] is False
    assert block["reports_deleted"] is False
    for key in ("entered_threshold_fitting", "entered_nested_evaluation",
                "entered_runtime_evaluator", "entered_deployment_manifest",
                "entered_live_rollout"):
        assert block[key] is False, key
    assert BOOTSTRAP.exists(), "the bootstrap manifest must not be deleted"


def test_calibration_script_never_loads_bootstrap_members():
    """The ensemble manifest and member records must be unreachable from the fit.

    Note `thr.BOOTSTRAP_SEED` is the *cluster* bootstrap used for confidence intervals
    and is unrelated to the trajectory-bootstrap ensemble, so a bare substring check on
    "bootstrap" would flag correct code.
    """
    source = (ROOT / "scripts" / "hybrid_obstacle_joint_gate_calibrate.py").read_text()
    assert "hybrid_obstacle_uncertainty_ensemble" not in source
    assert "member_records" not in source
    assert "PARKED_SKIN_TRAJECTORY_BOOTSTRAP_ENSEMBLE" not in source
    tree = ast.parse(source)
    literals = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not any("uncertainty_ensemble" in v for v in literals)


# -------------------------------------------------------------- agreement implementation
def test_mask_threshold_is_fixed_and_strict():
    assert jg.PIXEL_MASK_THRESHOLD == 0.5
    probability = np.array([0.49, 0.50, 0.51, 1.0]).reshape(1, 1, 2, 2)
    assert jg.changed_mask(probability).tolist() == [[False, False, True, True]]


def test_mask_comparison_matches_the_identifiability_audit():
    """The audit used `> 0.5`; a later module used `>=`. This must follow the audit."""
    source = (ROOT / "causal_parked_skin" / "joint_gate.py").read_text()
    assert "> PIXEL_MASK_THRESHOLD" in source
    assert ">= PIXEL_MASK_THRESHOLD" not in source
    audit = (ROOT / "scripts" / "hybrid_obstacle_activity_ensemble_audit.py").read_text()
    assert "> CHANGED_PIXEL_DECISION" in audit


def test_jaccard_of_two_empty_masks_is_one():
    empty = np.zeros((2, 8), dtype=bool)
    assert jg.jaccard(empty, empty).tolist() == [1.0, 1.0]


def test_jaccard_hand_computed():
    a = np.array([[True, True, False, False]])
    b = np.array([[True, False, True, False]])
    assert jg.jaccard(a, b)[0] == pytest.approx(1 / 3)
    assert jg.jaccard(a, a)[0] == pytest.approx(1.0)


def test_anchor_agreement_is_the_mean_of_the_two_anchor_pairs():
    m0 = np.array([[True, True, False, False]])
    m1 = np.array([[True, True, False, False]])       # J = 1.0
    m2 = np.array([[False, False, True, True]])       # J = 0.0
    assert jg.anchor_mask_agreement(m0, m1, m2)[0] == pytest.approx(0.5)
    # the audit's three-pair form includes J(m1, m2) = 0 and is therefore lower
    assert jg.three_pair_agreement(m0, m1, m2)[0] == pytest.approx(1 / 3)


def test_anchor_and_three_pair_forms_are_both_recorded():
    decision = _json(DIAG / "final_decision.json")
    block = decision["agreement_definition"]
    assert "mean(J(seed0,seed1), J(seed0,seed2))" in block["controlling"]
    assert "differs_from_identifiability_audit" in block
    assert "J(seed1, seed2)" in block["differs_from_identifiability_audit"]


def test_agreement_implementation_hash_is_pinned():
    manifest = _json(MANIFEST)
    source = (ROOT / "causal_parked_skin" / "joint_gate.py").read_bytes()
    assert manifest["agreement_implementation_sha256"] == \
        hashlib.sha256(source).hexdigest()


# --------------------------------------------------------------------- the joint rule
def test_joint_rule_requires_both_gates():
    activity = np.array([0.9, 0.9, 0.1, 0.1])
    agreement = np.array([0.9, 0.1, 0.9, 0.1])
    gates = jg.joint_decision(activity, 0.5, agreement, 0.5)
    assert gates["execute"].tolist() == [True, False, False, False]
    assert gates["abstained_by_uncertainty"].tolist() == [False, True, False, False]


def test_uncertainty_cannot_trigger_when_activity_fails():
    gates = jg.joint_decision(np.array([0.1]), 0.5, np.array([1.0]), 0.5)
    assert gates["execute"].tolist() == [False]
    assert gates["abstained_by_uncertainty"].tolist() == [False]


def test_gate_output_is_either_the_seed0_vector_or_exactly_zero():
    rng = np.random.default_rng(0)
    differential = rng.normal(size=(50, 7))
    execute = rng.random(50) < 0.5
    out = jg.apply_gate(differential, execute)
    for i in range(50):
        assert np.array_equal(out[i], differential[i] if execute[i] else np.zeros(7))


def test_no_averaging_in_the_joint_gate_module():
    tree = ast.parse((ROOT / "causal_parked_skin" / "joint_gate.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "mean":
            enclosing = [f.name for f in ast.walk(tree)
                         if isinstance(f, ast.FunctionDef) and node in ast.walk(f)]
            assert all(name in ("anchor_mask_agreement", "three_pair_agreement")
                       for name in enclosing), enclosing


def test_calibration_reads_only_masks_from_seeds_one_and_two():
    """Seeds 1 and 2 may only yield changed_probability, never a field or head output."""
    source = (ROOT / "scripts" / "hybrid_obstacle_joint_gate_calibrate.py").read_text()
    tree = ast.parse(source)
    reads = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
                and call.func.id in ("seed1", "seed2"):
            assert isinstance(node.slice, ast.Constant)
            reads.append(node.slice.value)
    assert reads, "no seed1/seed2 call found"
    assert set(reads) == {"changed_probability"}, reads


# ------------------------------------------------------------------------ calibration
def test_recall_floor_was_not_lowered():
    from hybrid_obstacle_joint_gate_calibrate import MIN_MEDIAN_ACTIVE_RECALL

    assert MIN_MEDIAN_ACTIVE_RECALL == 0.80
    decision = _json(DIAG / "final_decision.json")
    assert decision["owner_decisions_honoured"]["recall_floor_preserved_at_0_80"] is True
    assert decision["owner_decisions_honoured"]["recall_floor_lowered"] is False
    assert decision["calibration"]["contract"]["recall_floor_lowered"] is False
    assert decision["calibration"]["contract"]["min_median_active_recall"] == 0.80


def test_joint_grid_is_cartesian_and_includes_boundaries():
    calibration = _json(DIAG / "joint_calibration.json")
    grid = calibration["grid"]
    assert grid["cartesian_pairs"] == grid["activity"] * grid["agreement"]
    assert grid["activity"] > 1 and grid["agreement"] > 1


def test_activity_anti_degeneracy_floor_is_enforced():
    from hybrid_obstacle_joint_gate_calibrate import (
        MIN_ACTIVITY_ALONE_ACTIVE_RETENTION,
    )
    assert MIN_ACTIVITY_ALONE_ACTIVE_RETENTION == 0.85
    calibration = _json(DIAG / "joint_calibration.json")
    assert calibration["selected"]["checks"]["activity_alone_retention"] is True


def test_quiet_acceptance_floors_are_enforced():
    from hybrid_obstacle_joint_gate_calibrate import (
        MIN_ACTIVE_ACCEPTANCE,
        MIN_INACTIVE_ACCEPTANCE,
        MIN_ZERO_ACCEPTANCE,
    )
    for value in (MIN_ZERO_ACCEPTANCE, MIN_ACTIVE_ACCEPTANCE, MIN_INACTIVE_ACCEPTANCE):
        assert value == 0.80
    checks = _json(DIAG / "joint_calibration.json")["selected"]["checks"]
    for key in ("zero_acceptance", "active_acceptance", "inactive_acceptance"):
        assert checks[key] is True, key


def test_cluster_bootstrap_is_deterministic_and_recorded():
    manifest = _json(MANIFEST)
    assert manifest["bootstrap_replicates"] >= 10_000
    assert manifest["bootstrap_seed"] == thr.BOOTSTRAP_SEED
    rng = np.random.default_rng(0)
    values = rng.random((8, 4))
    a = thr.cluster_bootstrap_upper_bound(values, replicates=500, seed=thr.BOOTSTRAP_SEED)
    b = thr.cluster_bootstrap_upper_bound(values, replicates=500, seed=thr.BOOTSTRAP_SEED)
    assert np.array_equal(a, b)


def test_selected_pair_satisfied_every_calibration_check():
    calibration = _json(DIAG / "joint_calibration.json")
    assert calibration["feasible"] is True
    assert all(calibration["selected"]["checks"].values())


def test_old_threshold_was_retired_and_the_comparison_recorded():
    decision = _json(DIAG / "final_decision.json")
    block = decision["old_threshold_comparison"]
    assert block["old_activity_threshold"] == OLD_ACTIVITY_THRESHOLD
    assert block["new_activity_threshold"] != OLD_ACTIVITY_THRESHOLD
    assert block["recall_floor_relaxed"] is False
    assert block["uncertainty_did_not_justify_lowering_system_recall"] is True
    assert decision["owner_decisions_honoured"][
        "standalone_activity_threshold_retired"] is True


# --------------------------------------------------------------- isolation of the fit
def test_historical_frames_were_not_used_for_fitting():
    calibration = _json(DIAG / "joint_calibration.json")
    assert "never used for fitting" in calibration["historical_regression"]["note"]
    source = (ROOT / "scripts" / "hybrid_obstacle_joint_gate_calibrate.py").read_text()
    # the onset audit is read only after the deployment manifest is frozen
    body = source.split("deployment manifest frozen")[0]
    assert "onset_audit.read_text" not in body


def test_development4_and_confirmatory41_never_reached_the_fit():
    """Naming them in a recorded assertion field is fine; loading their manifests is not."""
    for script in ("hybrid_obstacle_joint_gate_calibrate.py",
                   "hybrid_obstacle_joint_gate_decision.py"):
        source = (ROOT / "scripts" / script).read_text()
        for config in ("hybrid_obstacle_controller_development4_v1",
                       "hybrid_obstacle_confirmatory41_v1"):
            assert config not in source, f"{script} references {config}"
        tree = ast.parse(source)
        literals = {n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert not any(v.endswith(".json") and
                       ("development4" in v or "confirmatory" in v) for v in literals)
    decision = _json(DIAG / "final_decision.json")
    for key in ("fitted_using_historical_17_frames", "fitted_using_development4",
                "fitted_using_confirmatory41"):
        assert decision["constraints_honoured"][key] is False


def test_partition_reused_without_redefinition():
    decision = _json(DIAG / "final_decision.json")
    assert decision["partition"]["reused_without_modification"] is True
    partition = _json(PARTITION)
    recorded = partition.pop("manifest_sha256")
    assert thr.canonical_hash(partition) == recorded
    assert decision["partition"]["manifest_sha256"] == recorded


def test_deployment_manifest_frozen_before_nested_evaluation():
    decision = _json(DIAG / "final_decision.json")
    assert decision["deployment_manifest"]["frozen_before_nested_evaluation"] is True
    manifest = _json(MANIFEST)
    recorded = manifest.pop("manifest_sha256")
    assert thr.canonical_hash(manifest) == recorded


def test_dataset_unchanged_and_read_only():
    manifest = _json(DATASET)
    writable = [e["output"] for e in manifest["entries"][:60]
                if stat.S_IMODE(os.stat(e["output"]).st_mode) & 0o222]
    assert writable == []


# ------------------------------------------------------- transfer results and stability
def test_frozen_inference_is_repeatable_across_all_three_seeds():
    stability = _json(DIAG / "joint_calibration.json")["inference_stability"]
    assert stability["repeats"] >= 20
    assert stability["activity_max_abs_delta"] <= 1e-7
    assert stability["agreement_max_abs_delta"] <= 1e-7
    assert stability["jaccard_identical"] is True
    assert stability["decisions_identical"] is True
    assert stability["training_kernels_invoked"] is False
    assert stability["stable"] is True


def test_historical_regression_result_is_recorded_frame_by_frame():
    historical = _json(DIAG / "joint_calibration.json")["historical_regression"]
    assert historical["count"] == 17
    for frame in historical["frames"]:
        for key in ("seed0_activity", "activity_threshold", "seed0_mask_pixels",
                    "seed1_mask_pixels", "seed2_mask_pixels", "jaccard_seed0_seed1",
                    "jaccard_seed0_seed2", "anchor_mask_agreement",
                    "agreement_threshold", "activity_pass", "agreement_pass",
                    "executed"):
            assert key in frame, key


def test_decision_matches_the_offline_transfer_outcome():
    decision = _json(DIAG / "final_decision.json")
    calibration = _json(DIAG / "joint_calibration.json")
    if decision["decision"] == "FULL_SEED_JOINT_GATE_OFFLINE_TRANSFER_FAILED":
        assert calibration["feasible"] is True
        assert not (calibration["nested_passed"]
                    and calibration["historical_regression"]["passes"]
                    and calibration["reused_diagnostic"]["passed"])
        assert decision["case"] == "C"
        assert decision["live_rollouts_executed"] == 0


def test_no_live_rollouts_and_confirmatory41_untouched():
    decision = _json(DIAG / "final_decision.json")
    assert decision["live_rollouts_executed"] == 0
    assert decision["live_rollouts_permitted"] == 20
    assert decision["development4_executed"] is False
    assert decision["confirmatory41_executed"] is False
    conf41 = _json(CONF41)
    assert conf41["executed_in_this_task"] is False
    assert len(conf41["rows"]) == 41
    assert {r["candidate_index"] for r in _json(DEV4)["rows"]} == {106, 107, 108, 118}


def test_controller_contract_unchanged():
    manifest = _json(MANIFEST)
    stack = _json(ROOT / "configs" / "hybrid_safety_stack_v1.json")["residual_controller"]
    for key in ("gain", "decay_per_second", "ema", "max_deviation_rad_per_joint",
                "arm_only", "gripper_owner"):
        assert manifest["controller"][key] == stack[key]
    assert manifest["controller"]["arm_only"] is True
    assert manifest["controller"]["gripper_owner"] == "ACT"


# ------------------------------------------------------------------------ final decision
def test_final_decision_token_matches_markdown_last_line():
    decision = _json(DIAG / "final_decision.json")
    markdown = (ROOT / "docs"
                / "HYBRID_OBSTACLE_FULL_SEED_JOINT_GATE_FINAL_DECISION.md").read_text()
    last = [line for line in markdown.splitlines() if line.strip()][-1]
    assert last == decision["decision"]


def test_final_decision_token_is_allowed():
    allowed = {"FULL_SEED_JOINT_GATE_READY_FOR_CONFIRMATORY_41",
               "FULL_SEED_JOINT_GATE_CALIBRATION_INFEASIBLE",
               "FULL_SEED_JOINT_GATE_OFFLINE_TRANSFER_FAILED",
               "FULL_SEED_JOINT_GATE_LIVE_TRANSFER_FAILED",
               "FULL_SEED_JOINT_GATE_LIVE_GROSS_REGRESSION",
               "REFERENCE_MODEL_INFERENCE_UNSTABLE",
               "CHECKPOINT_OR_SOURCE_MISMATCH",
               "FULL_SEED_JOINT_GATE_TASK_INCOMPLETE"}
    assert _json(DIAG / "final_decision.json")["decision"] in allowed
