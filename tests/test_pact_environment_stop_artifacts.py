from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "diagnostics_output" / "pact_vs_act"
TOKEN = "PACT_EXPERIMENT_INCOMPLETE"


def _json(path: Path):
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_both_required_reports_end_with_exact_environment_token():
    for path in (
        ROOT / "docs" / "PACT_ENVIRONMENT_ADEQUACY.md",
        ROOT / "docs" / "PACT_VS_ACT_FINAL_DECISION.md",
    ):
        assert path.read_text().rstrip().splitlines()[-1] == TOKEN


def test_gate_records_adequate_surface_signal_and_infrastructure_stop():
    gate = _json(DIAG / "environment_gate.json")
    assert gate["decision"] == TOKEN
    assert gate["surface_observability"]["robust_classification"] == "adequate"
    assert gate["surface_observability"]["all_leave_one_episode_out_points_pass"]
    assert gate["expert"]["attempts"] == 64
    assert gate["expert"]["usable_clean_demonstrations"] == 58
    assert gate["expert"]["no_scientific_outcome"] == 2
    assert gate["expert"]["status_counts"] == {
        "success": 59,
        "task_failure": 3,
        "sampling_failure": 2,
    }
    assert gate["act"]["attempts"] == 64
    assert gate["act"]["scientific_outcomes"] == 0
    assert gate["act"]["status_counts"] == {"infrastructure_failure": 64}
    assert gate["act"]["minimum_scientific_rows_met"] is False
    assert gate["gate_b"]["robust_classification"] == "inconclusive"
    assert gate["gate_c"]["robust_classification"] == "inconclusive"


def test_required_machine_artifacts_describe_a_valid_nonconfirmatory_stop():
    schedule = _json(DIAG / "schedule.json")
    analysis = _json(DIAG / "analysis.json")
    decision = _json(DIAG / "final_decision.json")
    assert schedule["status"] == (
        "not_instantiated_due_to_phase1_infrastructure_gate"
    )
    assert schedule["rows"] == []
    assert schedule["rollouts_executed"] == 0
    assert schedule["confirmatory_outcomes_seen"] is False
    assert schedule["pilot_execution"]["terminal_ledger_reconciled"] is True
    assert schedule["pilot_execution"]["scientific_schedule_reconciled"] is False
    assert schedule["pilot_execution"]["status_counts"] == {
        "invocation_failure": 64
    }
    assert analysis["valid_preregistered_stop"] is True
    assert analysis["arm_comparison"]["status"] == (
        "not_run_due_to_phase1_infrastructure_failure"
    )
    assert analysis["pilot_evaluation"]["rows_rerun"] == 0
    assert analysis["pilot_evaluation"]["scientific_outcomes"] == 0
    assert decision["decision"] == TOKEN
    assert decision["arm_comparison_available"] is False
    assert decision["full_policy_training_performed"] is False
    assert decision["confirmatory_evaluation_performed"] is False
    assert decision["final_arm_checkpoint_sha256s"] == []


def test_provenance_records_pilot_checkpoint_and_terminal_driver_ledger():
    provenance = _json(DIAG / "provenance.json")
    assert provenance["experiment_stage"] == (
        "stopped_at_phase1_infrastructure_gate"
    )
    assert provenance["decision"] == TOKEN
    assert provenance["final_arm_checkpoints"] == []
    assert provenance["surface_encoder"]["status"] == (
        "not_trained_due_to_phase1_gate"
    )
    checkpoint = provenance["pilot_checkpoint"]
    assert checkpoint["sha256"] == checkpoint["observed_sha256"]
    checkpoint_path = Path(checkpoint["path"])
    if checkpoint_path.exists():
        assert _sha256(checkpoint_path) == checkpoint["sha256"]
    execution = provenance["pilot_execution"]
    assert execution["driver_ledger"]["files"] == 64
    assert execution["scientific_result_files"] == 0
    assert execution["rows_rerun"] == 0


def test_small_artifact_hashes_and_protected_chain_claims_match():
    provenance = _json(DIAG / "provenance.json")
    for record in provenance["small_artifacts"].values():
        assert _sha256(ROOT / record["path"]) == record["sha256"]
    assert provenance["protected_chain"] == {
        "modified_by_pact_work": False,
        "preexisting_worktree_changes_preserved": True,
        "used_as_pact_evidence": False,
        "confirmatory41_touched_by_pact_work": False,
    }
