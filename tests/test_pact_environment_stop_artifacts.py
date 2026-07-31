from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "diagnostics_output" / "pact_vs_act"
TOKEN = "PACT_NO_CONFIRMED_BENEFIT"


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


def test_confirmatory_r2_schedule_and_analysis_are_frozen():
    schedule = _validate_self_hash(DIAG / "schedule.json", "schedule_sha256")
    analysis = _json(DIAG / "analysis.json")
    decision = _json(DIAG / "final_decision.json")
    assert schedule["schema_version"] == "pact_confirmatory_r2_schedule_v1"
    assert (
        schedule["schedule_sha256"]
        == "35e1377c9029f4934ff816b2d04c15f9134f232c7baa7136545565ea6b0057ad"
    )
    assert schedule["instances"] == 160
    assert schedule["rollouts"] == 960
    assert schedule["workers"] == 8
    assert len(schedule["rows"]) == 960
    assert analysis["results_available"] is True
    assert analysis["reconciliation"]["expected"] == 960
    assert analysis["reconciliation"]["valid"] == 960
    assert analysis["reconciliation"]["missing"] == []
    assert analysis["reconciliation"]["driver_noncomplete"] == []
    assert analysis["reconciliation"]["invalid"] == []
    assert analysis["reconciliation"]["reconciled"] is True
    assert analysis["pooled"]["ACT"]["collision_free_task_success"] == 170
    assert analysis["pooled"]["PACT"]["collision_free_task_success"] == 159
    assert (
        analysis["pooled"]["PACT_ZERO"]["collision_free_task_success"] == 160
    )
    assert decision["schema_version"] == "pact_final_decision_v2"
    assert decision["schedule_sha256"] == schedule["schedule_sha256"]
    assert decision["decision"] == TOKEN
    assert decision["PACT_minus_ACT"]["difference"] == -0.034375
    assert decision["PACT_minus_PACT_ZERO"]["difference"] == -0.003125


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
        "required_token": "PACT_EXPERIMENT_INCOMPLETE",
    }


def test_provenance_hashes_small_artifacts_checkpoints_and_encoder():
    provenance = _validate_self_hash(
        DIAG / "provenance.json",
        "provenance_sha256",
    )
    assert provenance["schema_version"] == "pact_vs_act_provenance_v4"
    assert provenance["experiment_stage"] == "confirmatory_r2_complete"
    assert provenance["decision"] == TOKEN
    execution = provenance["confirmatory_execution"]
    assert execution["expected_rows"] == 960
    assert execution["complete_rows"] == 960
    assert execution["scientific_schedule_reconciled"] is True
    assert execution["abort_reason"] is None
    assert execution["recovery_event_count"] == 0
    assert execution["row_audit"]["single_attempt_complete_rows"] == 960
    assert execution["row_audit"]["compacted_rows_verified"] == 958
    assert execution["row_audit"]["intact_schedule_indices"] == [0, 959]
    for record in provenance["artifacts"].values():
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        assert path.stat().st_size == record["size_bytes"]
        assert _sha256(path) == record["sha256"]
    assert len(provenance["policy_checkpoints"]) == 4
    for checkpoint in provenance["policy_checkpoints"]:
        assert (
            _sha256(Path(checkpoint["checkpoint"]))
            == checkpoint["checkpoint_sha256"]
        )
        assert _sha256(Path(checkpoint["dataset_stats"])) == checkpoint[
            "dataset_stats_sha256"
        ]
    assert _sha256(Path(provenance["surface_encoder"]["path"])) == provenance[
        "surface_encoder"
    ]["sha256"]
    assert all(provenance["analysis_integrity"]["byte_identical"].values())
    assert provenance["analysis_integrity"]["results_available"] is True
    assert provenance["analysis_integrity"]["reconciliation"]["reconciled"]
    assert provenance["storage"]["content_independent"] is True
    assert provenance["storage"]["final_schedule_row_unpacked"] is True
    assert (
        provenance["storage"]["full_original_payloads_byte_exact_recoverable"]
        is True
    )
    assert provenance["protected_chain"] == {
        "modified_by_pact_work": False,
        "preexisting_worktree_changes_preserved": True,
        "used_as_pact_evidence": False,
        "confirmatory41_touched_by_pact_work": False,
    }


def test_readme_links_final_report():
    assert "docs/PACT_VS_ACT_FINAL_DECISION.md" in (
        ROOT / "README.md"
    ).read_text()
