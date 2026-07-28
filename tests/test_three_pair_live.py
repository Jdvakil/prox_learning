"""Contract tests for the three-pair live development run."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from causal_parked_skin import joint_gate as jg

DIAG = ROOT / "diagnostics_output" / "hybrid_obstacle_three_pair_live"
MANIFEST = ROOT / "configs" / "hybrid_obstacle_three_pair_joint_gate_v1.json"
CONF41 = ROOT / "configs" / "hybrid_obstacle_confirmatory41_v1.json"
DEV4 = ROOT / "configs" / "hybrid_obstacle_controller_development4_v1.json"


def _json(p):
    return json.loads(Path(p).read_text())


def test_provenance_matched():
    r = _json(DIAG / "provenance_verification.json")
    assert r["all_matched"] is True and r["check_count"] >= 65


def test_all_twenty_rollouts_finalized_on_the_frozen_schedule():
    a = _json(DIAG / "live_analysis.json")
    assert a["rollout_count"] == 20
    counts = {c: b["rollouts"] for c, b in a["per_candidate"].items()}
    assert counts == {"106": 5, "107": 5, "108": 5, "118": 5}
    assert {r["candidate_index"] for r in a["rollouts"]} == {106, 107, 108, 118}


def test_thresholds_were_not_modified_for_the_live_run():
    m = _json(MANIFEST)
    d = _json(DIAG / "final_decision.json")
    assert d["activity_threshold"] == m["activity_threshold"]
    assert d["agreement_threshold"] == m["agreement_threshold"]
    assert d["thresholds_modified"] is False
    assert d["model_trained"] is False


def test_condition_is_the_three_pair_joint_gate():
    a = _json(DIAG / "live_analysis.json")
    assert a["condition"] == "ACT_PLUS_THREE_PAIR_JOINT_GATE"
    assert a["deployment_manifest_sha256"] == _json(MANIFEST)["manifest_sha256"]


def test_gripper_bitwise_nominal_and_actions_finite():
    a = _json(DIAG / "live_analysis.json")
    for r in a["rollouts"]:
        assert r["gripper_bitwise_preserved"] is True
        assert r["gripper_matches_nominal"] is True
        assert r["nonfinite_actions"] == 0
        assert r["executed_dq_finite"] is True


def test_shadow_diagnostics_recorded_every_rollout():
    a = _json(DIAG / "live_analysis.json")
    for r in a["rollouts"]:
        assert "activity_only_shadow_frames" in r
        assert r["oracle_active_frames"] + r["oracle_zero_frames"] == r["frames"]
        assert r["state_neutral"] is True


def test_privileged_oracle_stayed_state_neutral():
    a = _json(DIAG / "live_analysis.json")
    assert all(r["state_neutral"] for r in a["rollouts"])


def test_false_positive_bursts_were_negligible():
    fp = _json(DIAG / "live_analysis.json")["false_positive_summary"]
    assert fp["total_false_positive_frames"] <= 2
    assert fp["max_burst_length"] <= 2
    assert fp["bursts_persisting_after_end"] == 0
    assert fp["max_arm_deviation_in_any_burst_rad"] < 0.01 * fp["max_dev_limit_rad"]


def test_no_hazard_bar_contact_anywhere():
    h = _json(DIAG / "live_analysis.json")["harm_assessment"]
    assert h["new_hazard_bar_contact"]["rollouts_with_hazard_bar_contact"] == 0


def test_candidate_118_matches_the_act_only_baseline():
    """The negative control gained no environment contact, unlike the earlier V1."""
    h = _json(DIAG / "live_analysis.json")["harm_assessment"]
    assert h["candidate_118_other_environment"]["per_repeat"] == [0, 0, 0, 0, 0]
    assert h["candidate_118_other_environment"]["all_five_repeats"] is False


def test_candidate_118_controller_never_intervened():
    """0/5 task success on 118 cannot be ours if the action was bitwise nominal."""
    a = _json(DIAG / "live_analysis.json")
    rows = [r for r in a["rollouts"] if r["candidate_index"] == 118]
    assert len(rows) == 5
    for r in rows:
        assert r["executed_frames"] == 0
        assert r["maximum_nominal_deviation_norm"] == 0.0


def test_uncertainty_veto_contribution_is_recorded_even_though_zero():
    v = _json(DIAG / "live_analysis.json")["uncertainty_veto_contribution"]
    assert v["executed_frames"] == v["activity_only_shadow_frames"]
    assert v["veto_frames"] == 0
    assert "never executed" in v["note"]


def test_all_development_criteria_passed():
    d = _json(DIAG / "final_decision.json")
    assert d["all_criteria_passed"] is True
    assert all(d["development_criteria"].values())


def test_measurement_resolution_limitation_is_disclosed():
    m = _json(DIAG / "live_analysis.json")["measurement_resolution"]
    assert "contact class totals" in m["episode_resolved"]
    assert "per-frame contact classes are not in the logged rollout schema" \
        in m["limitation"]


def test_confirmatory41_untouched():
    d = _json(DIAG / "final_decision.json")
    assert d["confirmatory41_executed"] is False
    c = _json(CONF41)
    assert c["executed_in_this_task"] is False and len(c["rows"]) == 41
    assert {r["candidate_index"] for r in _json(DEV4)["rows"]} == {106, 107, 108, 118}


def test_rollout_budget_respected():
    d = _json(DIAG / "final_decision.json")
    assert d["constraints_honoured"]["live_rollouts_executed"] == 20
    assert d["constraints_honoured"]["live_rollouts_permitted"] == 20


def test_driver_reads_only_masks_from_seeds_one_and_two():
    src = (ROOT / "submodules" / "act" / "three_pair_joint_gate_driver.py").read_text()
    tree = ast.parse(src)
    reads = [n.slice.value for n in ast.walk(tree)
             if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Subscript)
             and isinstance(n.slice, ast.Constant)]
    assert "changed_probability" in reads
    lines = [ln for ln in src.splitlines() if "self.models[1]" in ln
             or "self.models[2]" in ln]
    assert lines and all("parked" not in ln for ln in lines)


def test_driver_refuses_a_bootstrap_manifest():
    with pytest.raises(jg.JointGateManifestError):
        jg.assert_not_bootstrap(
            _json(ROOT / "configs" / "hybrid_obstacle_uncertainty_ensemble_v1.json"))


def test_gate_output_is_seed0_or_zero():
    d = np.array([[1.0, -2.0, 3.0, 0, 0, 0, 0], [0.5] * 7])
    out = jg.apply_gate(d, np.array([True, False]))
    assert np.array_equal(out[0], d[0]) and np.array_equal(out[1], np.zeros(7))


def test_final_decision_token_matches_markdown_last_line():
    d = _json(DIAG / "final_decision.json")
    md = (ROOT / "docs"
          / "HYBRID_OBSTACLE_THREE_PAIR_LIVE_FINAL_DECISION.md").read_text()
    assert [ln for ln in md.splitlines() if ln.strip()][-1] == d["decision"]


def test_final_decision_token_is_allowed():
    assert _json(DIAG / "final_decision.json")["decision"] in {
        "THREE_PAIR_LIVE_DEVELOPMENT_PASSED",
        "THREE_PAIR_FALSE_BURST_HARM_CONFIRMED",
        "THREE_PAIR_LIVE_DEVELOPMENT_AMBIGUOUS",
        "THREE_PAIR_LIVE_DEVELOPMENT_INCOMPLETE"}
