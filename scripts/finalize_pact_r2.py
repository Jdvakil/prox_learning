#!/usr/bin/env python3
"""Audit and atomically promote the completed PACT confirmatory R2 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/root/pact_remediation_artifacts_v2/confirmatory_r2_35e1377c")
DIAG = ROOT / "diagnostics_output" / "pact_vs_act"
R2_DIAG = ROOT / "diagnostics_output" / "pact_vs_act_r2"
SCHEDULE_SOURCE = R2_DIAG / "schedule.json"
ANALYSIS_SOURCE = OUTPUT / "frozen_analysis.json"
DECISION_SOURCE = OUTPUT / "frozen_final_decision.json"
REPORT_SOURCE = OUTPUT / "PACT_VS_ACT_FINAL_DECISION.md"
TOKEN = "PACT_NO_CONFIRMED_BENEFIT"
SCHEDULE_SHA256 = (
    "35e1377c9029f4934ff816b2d04c15f9134f232c7baa7136545565ea6b0057ad"
)
ANALYZER_SHA256 = (
    "fd3c7f2e91a1737e248fc3ebe803018dcb4f9455d2b4e413d56946a4aebe25be"
)
EXPECTED_SOURCE_HASHES = {
    SCHEDULE_SOURCE: (
        "a2da2a138121dd86f1848f184690e7579012ab6d63fb7fb7a434ec663dd62ad5"
    ),
    ANALYSIS_SOURCE: (
        "d615ea7e63c9627e54a667a0fea700194ecf53b38f88b5385418e6616b9584e8"
    ),
    DECISION_SOURCE: (
        "78cbc831488a70235b23b8c26a83ebb67d22aa931e03921c3a3af320f8934214"
    ),
    REPORT_SOURCE: (
        "84ba1848d63739c5444c4893285c1ed22acb38a1ec02797e90c1099ac368cd6e"
    ),
    OUTPUT / "execution_summary.json": (
        "bde4db6b99dd29e501bf01978bbe484d78c2da38c68ee3439a0f26bdbb0cfcd4"
    ),
    OUTPUT / "storage_compaction_summary.json": (
        "b3a4020978c7be94806a00e7e370ae8f97d16c5fc382b90df4c4ef8a784c8e28"
    ),
}
PROMOTIONS = {
    SCHEDULE_SOURCE: DIAG / "schedule.json",
    ANALYSIS_SOURCE: DIAG / "analysis.json",
    DECISION_SOURCE: DIAG / "final_decision.json",
    REPORT_SOURCE: ROOT / "docs" / "PACT_VS_ACT_FINAL_DECISION.md",
}


class FinalizationError(RuntimeError):
    """The frozen R2 artifacts failed their final integrity audit."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_self_hashed(
    path: Path, *, hash_key: str, schema_version: str | None = None
) -> dict[str, Any]:
    document = load_json(path)
    payload = dict(document)
    observed = payload.pop(hash_key, None)
    if canonical_hash(payload) != observed:
        raise FinalizationError(f"{path}: {hash_key} mismatch")
    if (
        schema_version is not None
        and document.get("schema_version") != schema_version
    ):
        raise FinalizationError(f"{path}: schema mismatch")
    return document


def artifact_record(path: Path, *, display_path: str | None = None) -> dict:
    return {
        "path": display_path or str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git_value(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_fixed_sources() -> None:
    for path, expected in EXPECTED_SOURCE_HASHES.items():
        if sha256_file(path) != expected:
            raise FinalizationError(f"frozen artifact changed: {path}")
    if sha256_file(ROOT / "scripts/analyze_pact_confirmatory.py") != (
        ANALYZER_SHA256
    ):
        raise FinalizationError("frozen analyzer changed")
    if REPORT_SOURCE.read_text().rstrip().splitlines()[-1] != TOKEN:
        raise FinalizationError("frozen report token mismatch")


def verify_scientific_contract() -> tuple[dict, dict, dict, dict]:
    schedule = load_self_hashed(
        SCHEDULE_SOURCE,
        hash_key="schedule_sha256",
        schema_version="pact_confirmatory_r2_schedule_v1",
    )
    if (
        schedule["schedule_sha256"] != SCHEDULE_SHA256
        or len(schedule["rows"]) != 960
        or schedule["workers"] != 8
    ):
        raise FinalizationError("scientific schedule contract changed")
    if Counter(row["arm"] for row in schedule["rows"]) != {
        "ACT": 320,
        "PACT": 320,
        "PACT_ZERO": 320,
    }:
        raise FinalizationError("schedule arm balance changed")
    if any(
        count != 160
        for count in Counter(
            (row["arm"], row["checkpoint_seed"])
            for row in schedule["rows"]
        ).values()
    ):
        raise FinalizationError("schedule seed balance changed")

    execution = load_json(OUTPUT / "execution_summary.json")
    if execution != {
        **execution,
        "schedule_sha256": SCHEDULE_SHA256,
        "expected": 960,
        "complete_count": 960,
        "noncomplete": [],
        "workers": 8,
        "recovery_event_count": 0,
        "abort_reason": None,
        "scientific_schedule_reconciled": True,
        "endpoint_fields_inspected": False,
    }:
        raise FinalizationError("execution summary did not reconcile")

    analysis = load_json(ANALYSIS_SOURCE)
    reconciliation = analysis["reconciliation"]
    if (
        analysis["schedule_sha256"] != SCHEDULE_SHA256
        or analysis["results_available"] is not True
        or reconciliation != {
            "driver_noncomplete": [],
            "expected": 960,
            "invalid": [],
            "missing": [],
            "reconciled": True,
            "valid": 960,
        }
    ):
        raise FinalizationError("frozen analysis did not reconcile")
    decision = load_json(DECISION_SOURCE)
    if (
        decision["schedule_sha256"] != SCHEDULE_SHA256
        or decision["decision"] != TOKEN
    ):
        raise FinalizationError("frozen decision mismatch")
    return schedule, execution, analysis, decision


def verify_row_and_storage_integrity(schedule: dict) -> dict[str, Any]:
    ledger = load_json(OUTPUT / "completion_ledger.json")
    completions = ledger["completions"]
    if (
        ledger["schedule_sha256"] != SCHEDULE_SHA256
        or len(completions) != 960
        or len({item["rollout_id"] for item in completions}) != 960
    ):
        raise FinalizationError("completion ledger did not reconcile")

    storage_count = 0
    archive_bytes = 0
    videos_verified = 0
    for row in schedule["rows"]:
        index = int(row["schedule_index"])
        row_dir = OUTPUT / row["output_relpath"]
        driver = load_json(row_dir / "driver_result.json")
        attempts = load_json(row_dir / "attempt_ledger.json")
        boundary = load_json(row_dir / "initial_observation_accepted.json")
        if (
            driver["status"] != "complete"
            or driver["attempt_count"] != 1
            or driver["group_recovery_attempts"] != 0
            or driver["pre_observation_infrastructure_failures"] != 0
            or len(attempts["attempts"]) != 1
            or attempts["attempts"][0]["status"] != "complete"
            or boundary["initial_observation_accepted"] is not True
        ):
            raise FinalizationError(f"row {index}: attempt integrity failure")
        if (
            driver["rollout_id"] != row["rollout_id"]
            or boundary["rollout_id"] != row["rollout_id"]
        ):
            raise FinalizationError(f"row {index}: identity mismatch")

        storage_path = row_dir / "storage_archive.json"
        trajectory_path = row_dir / "trajectory.h5"
        result_path = row_dir / "result.json"
        if index in {0, 959}:
            result = load_json(result_path)
            if (
                storage_path.exists()
                or not trajectory_path.exists()
                or "storage_compaction" in result
            ):
                raise FinalizationError(
                    f"row {index}: declared intact row was transformed"
                )
            continue

        storage = load_self_hashed(
            storage_path,
            hash_key="storage_archive_sha256",
            schema_version="pact_r2_storage_archive_v1",
        )
        storage_count += 1
        if trajectory_path.exists():
            raise FinalizationError(f"row {index}: unpacked trajectory remains")
        if (
            storage["schedule_index"] != index
            or storage["rollout_id"] != row["rollout_id"]
            or storage["outcome_based_selection"] is not False
            or storage["original_payloads_recoverable"] is not True
        ):
            raise FinalizationError(f"row {index}: storage identity mismatch")
        compact_record = storage["compact_result"]
        if (
            result_path.stat().st_size != compact_record["size_bytes"]
            or sha256_file(result_path) != compact_record["sha256"]
        ):
            raise FinalizationError(f"row {index}: compact result changed")
        for key in ("result_archive", "trajectory_archive"):
            record = storage[key]
            archive = Path(record["archive_path"])
            if (
                archive.stat().st_size != record["archive_size_bytes"]
                or sha256_file(archive) != record["archive_sha256"]
                or record["decompression_verified"] is not True
            ):
                raise FinalizationError(
                    f"row {index}: {key} archive changed"
                )
            archive_bytes += record["archive_size_bytes"]
        for record in storage["videos"]:
            video = Path(record["path"])
            if (
                video.stat().st_size != record["size_bytes"]
                or sha256_file(video) != record["sha256"]
            ):
                raise FinalizationError(f"row {index}: video changed")
            videos_verified += 1

    storage_summary = load_self_hashed(
        OUTPUT / "storage_compaction_summary.json",
        hash_key="storage_compaction_summary_sha256",
        schema_version="pact_r2_storage_compaction_summary_v1",
    )
    if (
        storage_count != 958
        or storage_summary["compacted_count"] != 958
        or storage_summary["expected_compacted_count"] != 958
        or storage_summary["excluded_intact_schedule_indices"] != [0, 959]
        or storage_summary["reconciled_execution_observed"] is not True
        or storage_summary["outcome_based_selection"] is not False
    ):
        raise FinalizationError("storage summary did not reconcile")
    return {
        "single_attempt_complete_rows": 960,
        "recovery_events": 0,
        "compacted_rows_verified": storage_count,
        "archive_bytes_hashed": archive_bytes,
        "videos_verified": videos_verified,
        "intact_schedule_indices": [0, 959],
    }


def verify_reanalysis() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="pact_r2_final_reanalysis_", dir="/dev/shm"
    ) as temporary:
        temp = Path(temporary)
        paths = {
            "analysis": temp / "analysis.json",
            "decision": temp / "decision.json",
            "report": temp / "report.md",
        }
        subprocess.run(
            [
                "/root/act_retrain_venv/bin/python",
                str(ROOT / "scripts/analyze_pact_confirmatory.py"),
                "--schedule",
                str(SCHEDULE_SOURCE),
                "--output-root",
                str(OUTPUT),
                "--analysis-out",
                str(paths["analysis"]),
                "--decision-out",
                str(paths["decision"]),
                "--report-out",
                str(paths["report"]),
                "--environment-gate",
                str(DIAG / "environment_gate.json"),
                "--surface-report",
                str(DIAG / "surface_encoder_report_v2.json"),
                "--training-summary",
                str(DIAG / "policy_training_summary_v2.json"),
            ],
            check=True,
        )
        comparisons = {
            "analysis": paths["analysis"].read_bytes()
            == ANALYSIS_SOURCE.read_bytes(),
            "decision": paths["decision"].read_bytes()
            == DECISION_SOURCE.read_bytes(),
            "report": paths["report"].read_bytes()
            == REPORT_SOURCE.read_bytes(),
        }
    if not all(comparisons.values()):
        raise FinalizationError("independent frozen reanalysis differs")
    return {
        "byte_identical": comparisons,
        "analyzer_sha256": ANALYZER_SHA256,
    }


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(Path(temporary)) != sha256_file(source):
            raise FinalizationError(f"copy verification failed: {destination}")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def policy_records() -> list[dict[str, Any]]:
    training = load_json(DIAG / "policy_training_summary_v2.json")
    records = []
    for source in training["records"]:
        checkpoint = Path(source["checkpoint"])
        stats = Path(source["dataset_stats"])
        if (
            sha256_file(checkpoint) != source["checkpoint_sha256"]
            or sha256_file(stats) != source["dataset_stats_sha256"]
        ):
            raise FinalizationError("trained policy hash changed")
        records.append(
            {
                "arm": source["arm"],
                "seed": source["seed"],
                "best_epoch": source["best_epoch"],
                "best_validation_loss": source["best_validation_loss"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": source["checkpoint_sha256"],
                "dataset_stats": str(stats),
                "dataset_stats_sha256": source["dataset_stats_sha256"],
            }
        )
    return records


def build_provenance(
    *,
    execution: dict,
    analysis: dict,
    row_audit: dict,
    reanalysis: dict,
) -> dict[str, Any]:
    surface = load_json(DIAG / "surface_encoder_report_v2.json")
    surface_path = Path(surface["checkpoint_path"])
    if sha256_file(surface_path) != surface["checkpoint_sha256"]:
        raise FinalizationError("surface encoder hash changed")
    state = load_json(OUTPUT / "supervisor_state.json")
    artifacts = {
        "analysis": DIAG / "analysis.json",
        "confirmatory_schedule": DIAG / "schedule.json",
        "dispatch_contract": R2_DIAG / "dispatch_contract.json",
        "environment_gate": DIAG / "environment_gate.json",
        "environment_report": ROOT / "docs/PACT_ENVIRONMENT_ADEQUACY.md",
        "execution_summary": OUTPUT / "execution_summary.json",
        "final_decision": DIAG / "final_decision.json",
        "final_report": ROOT / "docs/PACT_VS_ACT_FINAL_DECISION.md",
        "finalization_script": ROOT / "scripts/finalize_pact_r2.py",
        "frozen_analysis_script": ROOT / "scripts/analyze_pact_confirmatory.py",
        "launch_smoke": OUTPUT / "launch_smoke.json",
        "policy_training_summary": DIAG / "policy_training_summary_v2.json",
        "r2_manifest": ROOT / "configs/pact_confirmatory_r2_manifest_v1.json",
        "r2_preregistration": ROOT / "configs/pact_r2_preregistration_v1.json",
        "storage_amendment": ROOT / "configs/pact_r2_storage_amendment_v1.json",
        "storage_amendment_report": ROOT / "docs/PACT_R2_STORAGE_AMENDMENT.md",
        "storage_compaction_summary": OUTPUT / "storage_compaction_summary.json",
        "surface_encoder_report": DIAG / "surface_encoder_report_v2.json",
    }
    document: dict[str, Any] = {
        "schema_version": "pact_vs_act_provenance_v4",
        "experiment_stage": "confirmatory_r2_complete",
        "decision": TOKEN,
        "branch": git_value(ROOT, "branch", "--show-current"),
        "source_commits_before_final_artifact_commit": {
            "root": git_value(ROOT, "rev-parse", "HEAD"),
            "act": git_value(ROOT / "submodules/act", "rev-parse", "HEAD"),
            "molmospaces": git_value(
                ROOT / "submodules/molmospaces", "rev-parse", "HEAD"
            ),
        },
        "scientific_contract": {
            "schedule_sha256": SCHEDULE_SHA256,
            "rollouts": 960,
            "instances": 160,
            "arms": ["ACT", "PACT", "PACT_ZERO"],
            "repeats_per_instance_per_arm": 2,
            "workers": 8,
            "primary_endpoint": analysis["primary_endpoint"],
            "target_contact_exempt": True,
            "rows_replaced_based_on_outcome": 0,
        },
        "confirmatory_execution": {
            "output_root": str(OUTPUT),
            "started_utc": state["full_dispatch_started_utc"],
            "finished_utc": execution["finished_utc"],
            "expected_rows": execution["expected"],
            "complete_rows": execution["complete_count"],
            "scientific_schedule_reconciled": True,
            "abort_reason": None,
            "recovery_event_count": 0,
            "row_audit": row_audit,
        },
        "analysis_integrity": {
            **reanalysis,
            "results_available": True,
            "reconciliation": analysis["reconciliation"],
        },
        "storage": {
            "amendment_authorized_before_compaction": True,
            "content_independent": True,
            "outcome_values_emitted_during_compaction": False,
            "full_original_payloads_byte_exact_recoverable": True,
            "final_schedule_row_unpacked": True,
            "storage_summary": load_json(
                OUTPUT / "storage_compaction_summary.json"
            ),
        },
        "pact_zero": {
            "separately_trained": False,
            "checkpoint_aliases_pact_by_seed": True,
            "inference_proximity_zeroed": True,
        },
        "surface_encoder": {
            "frozen": surface["frozen"],
            "path": str(surface_path),
            "sha256": surface["checkpoint_sha256"],
            "parameter_count": surface["parameter_count"],
            "heldout_metrics": surface["heldout_metrics"],
        },
        "policy_checkpoints": policy_records(),
        "r1_preservation": {
            "decision": "PACT_EXPERIMENT_INCOMPLETE",
            "output_root": (
                "/root/pact_remediation_artifacts_v2/confirmatory_b6d9b3f7"
            ),
            "used_as_r2_endpoint_evidence": False,
        },
        "protected_chain": {
            "modified_by_pact_work": False,
            "preexisting_worktree_changes_preserved": True,
            "used_as_pact_evidence": False,
            "confirmatory41_touched_by_pact_work": False,
        },
        "artifacts": {
            name: artifact_record(
                path,
                display_path=(
                    str(path.relative_to(ROOT))
                    if path.is_relative_to(ROOT)
                    else str(path)
                ),
            )
            for name, path in artifacts.items()
        },
    }
    document["provenance_sha256"] = canonical_hash(document)
    return document


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    verify_fixed_sources()
    schedule, execution, analysis, _decision = verify_scientific_contract()
    row_audit = verify_row_and_storage_integrity(schedule)
    reanalysis = verify_reanalysis()
    for source, destination in PROMOTIONS.items():
        atomic_copy(source, destination)
    for source, destination in PROMOTIONS.items():
        if sha256_file(source) != sha256_file(destination):
            raise FinalizationError(f"promoted artifact differs: {destination}")
    if (
        ROOT.joinpath("README.md").read_text().find(
            "docs/PACT_VS_ACT_FINAL_DECISION.md"
        )
        < 0
    ):
        raise FinalizationError("README lacks final-report link")
    provenance = build_provenance(
        execution=execution,
        analysis=analysis,
        row_audit=row_audit,
        reanalysis=reanalysis,
    )
    write_json_atomic(DIAG / "provenance.json", provenance)
    print(
        json.dumps(
            {
                "decision": TOKEN,
                "provenance_sha256": provenance["provenance_sha256"],
                "row_audit": row_audit,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
