"""Contract tests for the activity-identifiability audit.

The audit's conclusion is positive — uncertainty separates the failures — so these tests
mostly guard against the ways that conclusion could be spurious: a training path sneaking
in, seeds becoming a deployment candidate, neighbours drawn from the query's own
trajectory, or a metric that "passes" only by abstaining everywhere.
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

from causal_parked_skin import threshold as thr

DIAG = ROOT / "diagnostics_output" / "hybrid_obstacle_activity_identifiability"
PARKED_DECISION = (ROOT / "diagnostics_output" / "causal_parked_skin_reference_v1"
                   / "final_decision.json")
GATE_DECISION = (ROOT / "diagnostics_output" / "hybrid_obstacle_prox_activity_gate"
                 / "final_decision.json")
MANIFEST = ROOT / "configs" / "hybrid_obstacle_parked_skin_supervision_v1.json"
CONF41 = ROOT / "configs" / "hybrid_obstacle_confirmatory41_v1.json"
DEV4 = ROOT / "configs" / "hybrid_obstacle_controller_development4_v1.json"

AUDIT_SCRIPTS = (
    "hybrid_obstacle_activity_ensemble_audit.py",
    "hybrid_obstacle_activity_collision_audit.py",
    "hybrid_obstacle_activity_onset_tail_audit.py",
    "hybrid_obstacle_activity_identifiability_decision.py",
)


def _json(path):
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------- frozen artifacts
def test_provenance_all_checks_matched():
    report = _json(DIAG / "provenance_verification.json")
    assert report["all_matched"] is True
    assert report["failed"] == []
    assert report["check_count"] >= 60


def test_all_three_seed_checkpoints_match_recorded_hashes():
    previous = _json(PARKED_DECISION)
    seeds = {c["seed"]: c for c in previous["checkpoints"]
             if c["variant"] == "CURRENT_FRAME_ONLY"}
    assert set(seeds) == {0, 1, 2}
    for seed, record in seeds.items():
        path = Path(record["local_path"])
        assert path.is_file(), f"seed {seed} checkpoint missing"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_seed_zero_remains_the_deployment_candidate():
    decision = _json(DIAG / "final_decision.json")
    disposition = decision["seed_disposition"]
    assert disposition["deployment_candidate"] == "seed 0"
    assert disposition["seeds_1_and_2"] == "diagnostic only"
    assert decision["seeds_1_2_selected_for_deployment"] is False


def test_no_checkpoint_averaging_or_execution_ensembling():
    decision = _json(DIAG / "final_decision.json")
    assert decision["seed_disposition"]["checkpoints_averaged"] is False
    assert decision["seed_disposition"]["ensembled_for_execution"] is False
    assert decision["models_ensembled_for_execution"] is False


def test_model_weights_unchanged_by_the_audit():
    ensemble = _json(DIAG / "ensemble_audit.json")
    assert ensemble["model_weights_unchanged"] is True
    assert ensemble["training_performed"] is False


def test_dataset_remains_read_only():
    manifest = _json(MANIFEST)
    writable = [e["output"] for e in manifest["entries"][:60]
                if stat.S_IMODE(os.stat(e["output"]).st_mode) & 0o222]
    assert writable == []


def test_dataset_tree_hash_unchanged():
    manifest = _json(MANIFEST)
    dataset = _json(ROOT / "diagnostics_output" / "hybrid_obstacle_parked_skin_dataset"
                    / "final_decision.json")
    files = [{"distribution": e["distribution"], "episode_id": e["episode_id"],
              "file_sha256": hashlib.sha256(Path(e["output"]).read_bytes()).hexdigest()}
             for e in manifest["entries"]]
    tree = thr.canonical_hash(sorted(files, key=lambda f: (f["distribution"],
                                                           f["episode_id"])))
    assert tree == dataset["dataset"]["tree_sha256"]


# -------------------------------------------------------------------- no training path
@pytest.mark.parametrize("script", AUDIT_SCRIPTS)
def test_audit_scripts_contain_no_training_path(script):
    """No optimizer, no backward, no gradient step anywhere in the audit."""
    source = (ROOT / "scripts" / script).read_text()
    tree = ast.parse(source)
    attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("backward", "step", "zero_grad", "AdamW", "Adam", "SGD"):
        assert forbidden not in attributes, f"{script} reaches {forbidden}"
    assert "optim" not in source or "torch.optim" not in source


@pytest.mark.parametrize("script", AUDIT_SCRIPTS)
def test_audit_scripts_never_read_development4_or_confirmatory41(script):
    """Mentioning the names in a recorded assertion field is fine; loading them is not.

    A plain substring check fails on lines like `"confirmatory41_executed": False`, which
    are exactly the assertions we want the reports to carry.
    """
    source = (ROOT / "scripts" / script).read_text()
    for config in ("hybrid_obstacle_controller_development4_v1",
                   "hybrid_obstacle_confirmatory41_v1"):
        assert config not in source, f"{script} references the {config} manifest"
    tree = ast.parse(source)
    literals = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not any(v.endswith(".json") and "development4" in v for v in literals)
    assert not any(v.endswith(".json") and "confirmatory" in v for v in literals)


def test_confirmatory41_untouched():
    conf41 = _json(CONF41)
    assert conf41["executed_in_this_task"] is False
    assert len(conf41["rows"]) == 41
    decision = _json(DIAG / "final_decision.json")
    assert decision["confirmatory41_executed"] is False
    assert decision["development4_executed"] is False
    dev4_rows = {r["candidate_index"] for r in _json(DEV4)["rows"]}
    conf_rows = {r["candidate_index"] for r in conf41["rows"]}
    assert not dev4_rows & conf_rows


def test_no_live_evaluation_or_new_trajectories():
    decision = _json(DIAG / "final_decision.json")
    assert decision["live_rollouts_run"] is False
    assert decision["constraints_honoured"]["new_simulator_trajectories_generated"] is False
    assert decision["constraints_honoured"]["live_policy_evaluation_run"] is False
    assert decision["thresholds_changed"] is False


# ------------------------------------------------------------------- diagnostic groups
def test_diagnostic_groups_have_the_required_membership():
    groups = _json(DIAG / "diagnostic_groups.json")["groups"]
    expected = {"A_HISTORICAL_FALSE_POSITIVE", "B_ONSET_ZERO", "C_LATE_ZERO",
                "D_ONSET_ACTIVE", "E_LATE_ACTIVE", "F_HAZARD_ABSENT_ZERO",
                "G_HARD_TRUE_ACTIVE"}
    assert set(groups) == expected
    assert groups["A_HISTORICAL_FALSE_POSITIVE"]["count"] == 17
    for name, block in groups.items():
        assert block["count"] > 0, name


def test_historical_group_is_exactly_the_seventeen_known_frames():
    groups = _json(DIAG / "diagnostic_groups.json")
    frames = groups["groups"]["A_HISTORICAL_FALSE_POSITIVE"]["frames"]
    assert len(frames) == 17
    onset_audit = _json(ROOT / "diagnostics_output" / "hybrid_obstacle_prox_activity_gate"
                        / "onset_attribution.json")
    expected = {(r["trajectory_id"], r["step"])
                for r in onset_audit["known_false_positive_frames"]}
    assert {(f["trajectory_id"], f["step"]) for f in frames} == expected


def test_hazard_absent_zero_group_has_exact_zero_differential():
    groups = _json(DIAG / "diagnostic_groups.json")["groups"]
    for frame in groups["F_HAZARD_ABSENT_ZERO"]["frames"]:
        assert frame["hazard_present"] is False
        assert frame["oracle_dq_norm"] == 0.0


def test_hard_true_active_frames_are_genuinely_active():
    groups = _json(DIAG / "diagnostic_groups.json")["groups"]
    for frame in groups["G_HARD_TRUE_ACTIVE"]["frames"]:
        assert frame["oracle_active"] is True


def test_no_development4_or_confirmatory_frames_in_groups():
    groups = _json(DIAG / "diagnostic_groups.json")
    assert groups["development4_frames"] == 0
    assert groups["confirmatory41_frames"] == 0


# ----------------------------------------------------------------- neighbour exclusions
def test_neighbour_search_excludes_same_trajectory_and_episode():
    report = _json(DIAG / "collision_audit.json")
    assert set(report["exclusions"]) == {"same trajectory", "same episode identity",
                                         "duplicate scientific state hash"}
    source = (ROOT / "scripts" / "hybrid_obstacle_activity_collision_audit.py").read_text()
    assert 'episodes == record["episode_id"]' in source
    assert 'trajectories == record["trajectory_id"]' in source
    assert 'hashes == record["scientific_hash"]' in source


def test_all_four_feature_spaces_were_searched():
    report = _json(DIAG / "collision_audit.json")
    assert set(report["feature_spaces"]) == {
        "A_CURRENT_PROX_RAW", "B_CURRENT_PROX_EMBEDDING",
        "C_FULL_DEPLOYABLE_INPUT", "D_FROZEN_MODEL_EMBEDDING"}
    assert set(report["neighbours"]) == set(report["feature_spaces"])


def test_neighbour_report_covers_every_required_k():
    report = _json(DIAG / "collision_audit.json")
    for space in report["neighbours"].values():
        assert set(space["overall"]) == {"k=1", "k=4", "k=8", "k=16", "k=32"}


def test_nearest_neighbour_search_is_deterministic():
    """Same query, same pool, same result — the search uses no randomness."""
    import torch
    from hybrid_obstacle_activity_collision_audit import neighbour_search

    rng = np.random.default_rng(0)
    pool = rng.random((64, 12)).astype(np.float32)
    query = rng.random((5, 12)).astype(np.float32)
    pool_meta = [{"episode_id": f"e{i}", "trajectory_id": f"t{i}",
                  "oracle_active": bool(i % 2), "scientific_hash": f"h{i}"}
                 for i in range(64)]
    query_meta = [{"episode_id": "q", "trajectory_id": "q", "oracle_active": False,
                   "scientific_hash": "qh"} for _ in range(5)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = np.arange(64)
    a = neighbour_search(query, pool, query_meta, pool_meta, rows, device)
    b = neighbour_search(query, pool, query_meta, pool_meta, rows, device)
    assert [x["neighbour_rows"] for x in a] == [x["neighbour_rows"] for x in b]


def test_neighbour_search_removes_excluded_candidates():
    import torch
    from hybrid_obstacle_activity_collision_audit import neighbour_search

    pool = np.eye(8, dtype=np.float32)
    query = np.eye(8, dtype=np.float32)[:1]
    pool_meta = [{"episode_id": "shared" if i < 4 else f"e{i}",
                  "trajectory_id": f"t{i}", "oracle_active": False,
                  "scientific_hash": f"h{i}"} for i in range(8)]
    query_meta = [{"episode_id": "shared", "trajectory_id": "tq",
                   "oracle_active": False, "scientific_hash": "hq"}]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = neighbour_search(query, pool, query_meta, pool_meta, np.arange(8), device)
    assert all(row >= 4 for row in result[0]["neighbour_rows"])
    assert result[0]["excluded_count"] == 4


# ------------------------------------------------------------------------- collisions
def test_collision_tolerances_are_the_predeclared_values():
    report = _json(DIAG / "collision_audit.json")
    tolerances = report["collisions"]["tolerances"]
    assert tolerances["closeness"] == 1e-5
    assert tolerances["qpos"] == 1e-6
    assert tolerances["qvel"] == 1e-6
    assert tolerances["action"] == 1e-6


def test_collision_counts_are_recorded_and_consistent():
    report = _json(DIAG / "collision_audit.json")["collisions"]
    for key in ("exact_current_prox_count", "exact_full_input_count",
                "near_identity_count"):
        assert isinstance(report[key], int)
    assert report["any_exact_full_input_collision"] == bool(
        report["exact_full_input_count"])


def test_reported_collision_pairs_actually_have_opposite_labels():
    report = _json(DIAG / "collision_audit.json")["collisions"]
    for key in ("exact_full_input_pairs", "exact_current_prox_pairs",
                "near_identity_pairs"):
        for pair in report[key]:
            assert pair["a"]["oracle_active"] != pair["b"]["oracle_active"]
            assert pair["a"]["episode_id"] != pair["b"]["episode_id"]


# --------------------------------------------------------------- uncertainty machinery
def test_partial_auroc_is_bounded_and_chance_calibrated():
    from hybrid_obstacle_activity_ensemble_audit import partial_auroc

    rng = np.random.default_rng(0)
    label = rng.random(2000) < 0.1
    perfect = label.astype(float)
    # not exactly 1.0: the ROC is discrete, so the last retained point sits just below
    # the 5% FPR bound and the trapezoid loses that sliver
    assert partial_auroc(perfect, label, 0.05) == pytest.approx(1.0, abs=5e-3)
    chance = rng.random(2000)
    assert 0.3 < partial_auroc(chance, label, 0.05) < 0.7


def test_auroc_matches_a_hand_computed_case():
    from hybrid_obstacle_activity_ensemble_audit import auroc

    score = np.array([0.1, 0.4, 0.35, 0.8])
    label = np.array([False, False, True, True])
    assert auroc(score, label) == pytest.approx(0.75)


def test_uncertainty_metrics_are_reproducible_from_the_report():
    """Every metric must be recorded for all 17 failures, not just summarised."""
    ensemble = _json(DIAG / "ensemble_audit.json")
    for name, values in ensemble["historical_frame_metrics"].items():
        assert len(values) == 17, name
        assert all(np.isfinite(v) for v in values), name
        assert name in ensemble["metrics"]


def test_degenerate_metric_passes_are_flagged_not_used():
    """A metric that abstains on quiet frames too must not be selected."""
    decision = _json(DIAG / "final_decision.json")
    block = decision["ensemble_uncertainty"]
    assert block["selected_metric"] not in block["metrics_passing_letter_but_degenerate"]
    if block["metrics_passing_letter_but_degenerate"]:
        assert block["selected_metric"] is not None


def test_selected_metric_separates_failures_from_quiet_frames():
    decision = _json(DIAG / "final_decision.json")
    block = decision["ensemble_uncertainty"]
    detail = block["selected_metric_detail"]
    groups = detail["by_group"]
    historical = groups["A_HISTORICAL_FALSE_POSITIVE"]["median"]
    higher_is_uncertain = detail["higher_value_means_more_uncertain"]
    for quiet in ("B_ONSET_ZERO", "C_LATE_ZERO", "F_HAZARD_ABSENT_ZERO"):
        value = groups[quiet]["median"]
        assert (value < historical) if higher_is_uncertain else (value > historical)


def test_separability_gate_thresholds_are_the_predeclared_values():
    ensemble = _json(DIAG / "ensemble_audit.json")
    gates = ensemble["separability_gates"]
    assert gates["min_rejected_historical"] == 16
    assert gates["historical_total"] == 17
    assert gates["min_retained_active"] == 0.80
    assert gates["min_retained_hard_true_active"] == 0.80


# ------------------------------------------------------------------ bootstrap / tails
def test_clustered_bootstrap_resamples_trajectories():
    from hybrid_obstacle_activity_onset_tail_audit import clustered_bootstrap

    values = np.concatenate([np.zeros(50), np.ones(50)])
    clusters = np.array(["a"] * 50 + ["b"] * 50)
    result = clustered_bootstrap(values, clusters, lambda v: float(np.mean(v)),
                                 replicates=500)
    assert result["resampled_unit"] == "trajectory"
    assert result["point"] == pytest.approx(0.5)
    # two fully-separated clusters must produce a wide interval, not a tight one
    assert result["ci_high"] - result["ci_low"] > 0.4


def test_clustered_bootstrap_is_deterministic():
    from hybrid_obstacle_activity_onset_tail_audit import clustered_bootstrap

    rng = np.random.default_rng(0)
    values = rng.random(200)
    clusters = np.repeat(np.arange(20), 10).astype(str)
    a = clustered_bootstrap(values, clusters, lambda v: float(np.mean(v)), replicates=300)
    b = clustered_bootstrap(values, clusters, lambda v: float(np.mean(v)), replicates=300)
    assert a == b


def test_partial_roc_reports_every_required_fpr():
    report = _json(DIAG / "onset_tail_audit.json")["score_tail"]
    assert set(report["partial_roc"]) == {
        "tpr_at_fpr_0.1pct", "tpr_at_fpr_0.5pct", "tpr_at_fpr_1pct",
        "tpr_at_fpr_2pct", "tpr_at_fpr_5pct"}
    assert set(report["tail_overlap"]) == {
        "zero_top_0.1pct", "zero_top_0.5pct", "zero_top_1pct",
        "zero_top_2pct", "zero_top_5pct"}


def test_full_range_auroc_is_not_the_headline():
    report = _json(DIAG / "onset_tail_audit.json")["score_tail"]
    assert "auroc" not in json.dumps(report).lower() or "note" in report
    assert "not the headline" in report["note"]


def test_onset_separation_uses_a_within_class_baseline():
    """Between-class distance alone confuses embedding compression with overlap."""
    report = _json(DIAG / "onset_tail_audit.json")["onset_vs_late"]
    ratio = report["separation_ratio"]
    assert "onset_within_baseline" in ratio
    assert "late_within_baseline" in ratio
    assert ratio["onset"] > 0 and ratio["late"] > 0
    assert report["episode_index_used_as_feature"] is False


# ------------------------------------------------------------------------ the decision
def test_final_decision_token_matches_markdown_last_line():
    decision = _json(DIAG / "final_decision.json")
    markdown = (ROOT / "docs"
                / "HYBRID_OBSTACLE_ACTIVITY_IDENTIFIABILITY_FINAL_DECISION.md").read_text()
    last = [line for line in markdown.splitlines() if line.strip()][-1]
    assert last == decision["decision"]


def test_final_decision_token_is_allowed():
    allowed = {"EPISTEMIC_UNCERTAINTY_SIGNAL_PRESENT",
               "CURRENT_OBSERVATION_NOT_IDENTIFIABLE",
               "UNCERTAINTY_SIGNAL_INCONCLUSIVE",
               "CHECKPOINT_OR_SOURCE_MISMATCH",
               "ACTIVITY_IDENTIFIABILITY_AUDIT_INCOMPLETE"}
    assert _json(DIAG / "final_decision.json")["decision"] in allowed


def test_every_decision_trigger_is_recorded_with_evidence():
    rules = _json(DIAG / "final_decision.json")["decision_rules"]
    expected = {"exact_or_near_identical_inputs_with_opposite_labels",
                "historical_failures_show_low_ensemble_disagreement",
                "high_neighbour_label_entropy_around_the_failures",
                "no_uncertainty_metric_rejects_failures_while_retaining_recall",
                "onset_active_and_zero_states_overlap_materially"}
    assert set(rules["not_identifiable_triggers"]) == expected
    for block in rules["not_identifiable_triggers"].values():
        assert isinstance(block["fired"], bool)
        assert block["evidence"]


def test_decision_is_consistent_with_its_triggers():
    decision = _json(DIAG / "final_decision.json")
    rules = decision["decision_rules"]
    if decision["decision"] == "EPISTEMIC_UNCERTAINTY_SIGNAL_PRESENT":
        assert rules["triggers_fired"] == []
        assert rules["all_signal_present_criteria_met"] is True
        assert decision["collisions"]["exact_full_input_count"] == 0
    if decision["decision"] == "CURRENT_OBSERVATION_NOT_IDENTIFIABLE":
        assert rules["triggers_fired"]


def test_handoff_commit_discrepancy_is_disclosed():
    decision = _json(DIAG / "final_decision.json")
    block = decision["handoff_commit_discrepancy"]
    assert block["handoff_stated_root_commit"] == "db326d7"
    assert block["resolves_to_an_object"] is False
    assert block["actual_previous_commit"] == "db326d1"


def test_uncertainty_claim_is_not_merely_that_seeds_differ():
    """The handoff forbids that claim; the report must show targeted separation."""
    decision = _json(DIAG / "final_decision.json")
    assert decision["constraints_honoured"][
        "uncertainty_claimed_merely_because_seeds_differ"] is False
    detail = decision["ensemble_uncertainty"]["selected_metric_detail"]
    assert detail["auroc_historical_vs_active"] > 0.9
    assert detail["gate_operating_point"]["historical_rejected"] >= 16
    assert detail["gate_operating_point"]["active_recall"] >= 0.80
