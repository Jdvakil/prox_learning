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


def _canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_self_hash(path: Path, key: str):
    document = _json(path)
    payload = dict(document)
    observed = payload.pop(key)
    assert _canonical_hash(payload) == observed
    return document


def test_required_reports_end_with_their_exact_tokens():
    assert (
        (ROOT / "docs" / "PACT_ENVIRONMENT_ADEQUACY.md")
        .read_text()
        .rstrip()
        .splitlines()[-1]
        == "PACT_ENVIRONMENT_ADEQUATE"
    )
    assert (
        (ROOT / "docs" / "PACT_VS_ACT_FINAL_DECISION.md")
        .read_text()
        .rstrip()
        .splitlines()[-1]
        == TOKEN
    )


def test_gate_is_adequate_and_b_and_c_use_the_recorded_counts():
    gate = _json(DIAG / "environment_gate.json")
    assert gate["decision"] == "PACT_ENVIRONMENT_ADEQUATE"
    assert gate["surface_observability"]["robust_classification"] == "adequate"
    assert gate["expert"]["usable_clean_demonstrations"] == 58
    assert gate["act"]["scientific_outcomes"] == 64
    assert gate["act"]["collision_free_task_success"] == 23
    assert gate["act"]["episodes_with_hazard_bar_contact"] == 23
    assert gate["act"]["episodes_with_other_environment_contact"] == 0
    assert gate["gate_b"]["robust_classification"] == "adequate"
    assert gate["gate_c"]["robust_classification"] == "adequate"


def test_confirmatory_schedule_and_incomplete_analysis_are_frozen():
    schedule = _validate_self_hash(DIAG / "schedule.json", "schedule_sha256")
    analysis = _json(DIAG / "analysis.json")
    decision = _json(DIAG / "final_decision.json")
    assert schedule["instances"] == 160
    assert schedule["rollouts"] == 960
    assert schedule["workers"] == 8
    assert len(schedule["rows"]) == 960
    assert analysis["results_available"] is False
    assert analysis["reconciliation"]["expected"] == 960
    assert analysis["reconciliation"]["valid"] == 1
    assert len(analysis["reconciliation"]["missing"]) == 959
    assert analysis["reconciliation"]["reconciled"] is False
    assert "pooled" not in analysis
    assert "fisher_exact" not in analysis
    assert "paired_instance_bootstrap" not in analysis
    assert decision == {
        "schema_version": "pact_final_decision_v2",
        "schedule_sha256": schedule["schedule_sha256"],
        "decision": TOKEN,
        "reason": "The frozen schedule did not reconcile; outcomes are not analyzed.",
    }


def test_interruption_ledger_records_no_reruns_and_no_endpoint_interpretation():
    incident = _validate_self_hash(
        DIAG / "confirmatory_interruption_v1.json",
        "incident_sha256",
    )
    assert incident["smoke_audit"]["passed"] is True
    assert incident["smoke_audit"]["smoke_invocations"] == 1
    assert incident["smoke_audit"]["endpoint_fields_inspected_during_audit"] is False
    interruption = incident["interruption"]
    assert interruption["post_boundary_terminal_count"] == 8
    assert interruption["never_started_count"] == 951
    assert interruption["rows_rerun"] == 0
    assert interruption["remaining_rows_launched_after_irrecoverability_known"] == 0
    assert all(
        row["terminal_status"] == "post_boundary_failure"
        and row["rerun"] is False
        and row["scientific_result_written"] is False
        for row in interruption["post_boundary_terminal_rows"]
    )
    assert incident["decision_consequence"] == {
        "scientific_schedule_reconciled": False,
        "endpoint_analysis_permitted": False,
        "required_token": TOKEN,
    }


def test_provenance_hashes_small_artifacts_checkpoints_and_encoder():
    provenance = _validate_self_hash(
        DIAG / "provenance.json",
        "provenance_sha256",
    )
    assert provenance["experiment_stage"] == "confirmatory_dispatch_interrupted"
    assert provenance["decision"] == TOKEN
    assert provenance["confirmatory_execution"] == {
        **provenance["confirmatory_execution"],
        "expected_rows": 960,
        "workers": 8,
        "smoke_complete": 1,
        "post_boundary_terminal_failures": 8,
        "never_started": 951,
        "rows_rerun": 0,
        "scientific_schedule_reconciled": False,
        "endpoint_outcomes_interpreted": False,
    }
    for record in provenance["small_artifacts"].values():
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        assert _sha256(path) == record["sha256"]
    assert len(provenance["policy_checkpoints"]) == 4
    for checkpoint in provenance["policy_checkpoints"]:
        assert _sha256(Path(checkpoint["checkpoint"])) == checkpoint[
            "checkpoint_sha256"
        ]
        assert _sha256(Path(checkpoint["dataset_stats"])) == checkpoint[
            "dataset_stats_sha256"
        ]
    assert _sha256(Path(provenance["surface_encoder"]["path"])) == provenance[
        "surface_encoder"
    ]["sha256"]
    assert provenance["analysis_integrity"]["matches_dispatch_contract"] is True
    assert provenance["analysis_integrity"]["statistical_comparisons_interpreted"] is False
    assert provenance["protected_chain"] == {
        "modified_by_pact_work": False,
        "preexisting_worktree_changes_preserved": True,
        "used_as_pact_evidence": False,
        "confirmatory41_touched_by_pact_work": False,
    }
