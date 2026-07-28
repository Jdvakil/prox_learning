from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "diagnostics_output" / "pact_vs_act"
TOKEN = "PACT_ENVIRONMENT_INADEQUATE"


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


def test_gate_records_the_single_failed_prerequisite_and_early_stop():
    gate = _json(DIAG / "environment_gate.json")
    assert gate["decision"] == TOKEN
    assert gate["stop_before_policy_training"] is True
    assert gate["expert"]["ordinary_task_success"] == 20
    assert gate["expert"]["collision_free_task_success"] == 19
    assert gate["expert"]["episodes_with_intrusion_sighting"] == 20
    assert gate["expert"]["status_counts"] == {
        "success": 20,
        "sampling_failure": 1,
        "infrastructure_failure": 3,
    }
    failed = [name for name, passed in gate["checks"].items() if not passed]
    assert failed == [
        "expert_collision_free_task_success_at_least_20_of_24"
    ]
    assert gate["act"] == {
        "n": 0,
        "status": "not_run_due_to_failed_expert_prerequisite",
    }


def test_required_machine_artifacts_describe_a_valid_nonconfirmatory_stop():
    schedule = _json(DIAG / "schedule.json")
    analysis = _json(DIAG / "analysis.json")
    decision = _json(DIAG / "final_decision.json")
    assert schedule["status"] == (
        "not_instantiated_due_to_phase1_environment_gate"
    )
    assert schedule["rows"] == []
    assert schedule["rollouts_executed"] == 0
    assert schedule["confirmatory_outcomes_seen"] is False
    assert analysis["valid_early_stop"] is True
    assert analysis["arm_comparison"]["status"] == (
        "not_run_due_to_failed_environment_gate"
    )
    assert analysis["policy_checkpoint_sha256s"] == []
    assert decision["decision"] == TOKEN
    assert decision["policy_training_performed"] is False
    assert decision["confirmatory_evaluation_performed"] is False
    assert decision["checkpoint_sha256s"] == []


def test_provenance_hashes_all_terminal_pilot_artifacts():
    provenance = _json(DIAG / "provenance.json")
    pilot = provenance["pilot_collection"]
    assert provenance["experiment_stage"] == (
        "stopped_at_phase1_environment_gate"
    )
    assert provenance["policy_checkpoint_status"] == (
        "not_trained_due_to_phase1_environment_gate"
    )
    assert provenance["policy_checkpoints"] == []
    assert provenance["surface_encoder"]["status"] == (
        "not_trained_due_to_phase1_environment_gate"
    )
    assert len(pilot["rows"]) == 24
    assert pilot["status_counts"] == {
        "success": 20,
        "task_failure": 0,
        "sampling_failure": 1,
        "infrastructure_failure": 3,
    }
    assert pilot["artifact_file_count"] == sum(
        len(row["artifacts"]) for row in pilot["rows"]
    )
    for row in pilot["rows"]:
        for artifact in row["artifacts"]:
            path = ROOT / artifact["path"]
            assert path.stat().st_size == artifact["size_bytes"]
            assert _sha256(path) == artifact["sha256"]


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
