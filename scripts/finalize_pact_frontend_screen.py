#!/usr/bin/env python3
"""Reconcile frozen screen artifacts and write provenance after analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

SCREEN_TOKENS = {
    "FRONTEND_SCREEN_SIGNAL_PRESENT",
    "FRONTEND_SCREEN_WEAK_SIGNAL",
    "FRONTEND_SCREEN_NO_SIGNAL",
    "FRONTEND_SCREEN_INCONCLUSIVE",
}
CONFIRMATORY_TOKENS = {
    "PACT_BENEFIT_ESTABLISHED",
    "PACT_NO_CONFIRMED_BENEFIT",
    "PACT_WORSE_THAN_ACT",
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--preregistration", required=True, type=Path
    )
    parser.add_argument(
        "--dataset-hash-amendment", required=True, type=Path
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--dataset-conversion", required=True, type=Path
    )
    parser.add_argument("--encoder-report", required=True, type=Path)
    parser.add_argument("--token-report", required=True, type=Path)
    parser.add_argument(
        "--training-summary", required=True, type=Path
    )
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument(
        "--dispatch-contract", required=True, type=Path
    )
    parser.add_argument(
        "--storage-amendment", required=True, type=Path
    )
    parser.add_argument(
        "--storage-report", required=True, type=Path
    )
    parser.add_argument(
        "--storage-summary", required=True, type=Path
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--final-decision", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--provenance-out", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    analysis = json.loads(args.analysis.read_text())
    decision = json.loads(args.final_decision.read_text())
    dispatch = json.loads(args.dispatch_contract.read_text())
    storage_amendment = json.loads(
        args.storage_amendment.read_text()
    )
    storage_summary = json.loads(args.storage_summary.read_text())
    token = decision.get("decision")
    if token not in SCREEN_TOKENS or token in CONFIRMATORY_TOKENS:
        raise ValueError("final decision is not an allowed screen token")
    if schedule["schedule_sha256"] != analysis[
        "schedule_sha256"
    ] or schedule["schedule_sha256"] != decision[
        "schedule_sha256"
    ]:
        raise ValueError("analysis/decision schedule identity mismatch")
    if analysis["analysis_sha256"] != decision["analysis_sha256"]:
        raise ValueError("decision is not bound to analysis")
    if decision["final_decision_sha256"] != canonical_hash(
        {
            key: value
            for key, value in decision.items()
            if key != "final_decision_sha256"
        }
    ):
        raise ValueError("final decision self-hash mismatch")
    for document, key, label in (
        (
            storage_amendment,
            "storage_amendment_sha256",
            "storage amendment",
        ),
        (
            storage_summary,
            "storage_compaction_summary_sha256",
            "storage summary",
        ),
    ):
        payload = {
            item_key: value
            for item_key, value in document.items()
            if item_key != key
        }
        if document.get(key) != canonical_hash(payload):
            raise ValueError(f"{label} self-hash mismatch")
    if (
        storage_amendment["schedule_sha256"]
        != schedule["schedule_sha256"]
        or storage_summary["schedule_sha256"]
        != schedule["schedule_sha256"]
        or storage_amendment["dispatch_contract_sha256"]
        != dispatch["dispatch_contract_sha256"]
    ):
        raise ValueError("storage artifacts are not bound to dispatch")
    exclusions = set(
        storage_amendment["excluded_intact_schedule_indices"]
    )
    if exclusions != {0, 119}:
        raise ValueError("unexpected raw storage exclusions")
    storage_archives = []
    for row in schedule["rows"]:
        index = int(row["schedule_index"])
        row_dir = args.output_root / row["output_relpath"]
        archive_path = row_dir / "storage_archive.json"
        if index in exclusions:
            if archive_path.exists() or not (
                row_dir / "trajectory.h5"
            ).exists():
                raise ValueError(
                    f"raw excluded row {index} is not intact"
                )
            continue
        archive = json.loads(archive_path.read_text())
        payload = {
            key: value
            for key, value in archive.items()
            if key != "storage_archive_sha256"
        }
        if archive.get(
            "storage_archive_sha256"
        ) != canonical_hash(payload):
            raise ValueError(
                f"row {index}: storage archive self-hash mismatch"
            )
        if (
            archive["schedule_index"] != index
            or archive["rollout_id"] != row["rollout_id"]
            or archive["outcome_based_selection"] is not False
            or archive["original_payloads_recoverable"] is not True
        ):
            raise ValueError(
                f"row {index}: storage archive identity mismatch"
            )
        compact_record = archive["compact_result"]
        result_path = row_dir / "result.json"
        if (
            result_path.stat().st_size
            != compact_record["size_bytes"]
            or file_hash(result_path) != compact_record["sha256"]
        ):
            raise ValueError(
                f"row {index}: compact result changed"
            )
        for archive_key in (
            "result_archive",
            "trajectory_archive",
        ):
            record = archive[archive_key]
            path = Path(record["archive_path"])
            if (
                not path.exists()
                or path.stat().st_size != record["archive_size_bytes"]
                or file_hash(path) != record["archive_sha256"]
            ):
                raise ValueError(
                    f"row {index}: {archive_key} changed"
                )
        storage_archives.append(archive_path)
    if (
        len(storage_archives) != 118
        or storage_summary["compacted_count"] != 118
        or storage_summary["expected_compacted_count"] != 118
        or storage_summary["excluded_intact_schedule_indices"]
        != [0, 119]
        or storage_summary["reconciled_execution_observed"] is not True
        or storage_summary["outcome_based_selection"] is not False
        or storage_summary[
            "endpoint_values_emitted_during_compaction"
        ]
        is not False
    ):
        raise ValueError("storage compaction did not reconcile")
    report_lines = [
        line.strip()
        for line in args.report.read_text().splitlines()
        if line.strip()
    ]
    if not report_lines or report_lines[-1] != token:
        raise ValueError("report last nonblank line is not screen token")
    result_paths = list(
        args.output_root.glob("rows/*/result.json")
    )
    driver_paths = list(
        args.output_root.glob("rows/*/driver_result.json")
    )
    expected = 120 if token != "FRONTEND_SCREEN_INCONCLUSIVE" else len(
        result_paths
    )
    if token != "FRONTEND_SCREEN_INCONCLUSIVE" and (
        len(result_paths) != 120 or len(driver_paths) != 120
    ):
        raise ValueError("conclusive screen lacks 120 result/driver pairs")
    paths = {
        "preregistration": args.preregistration,
        "dataset_hash_amendment": args.dataset_hash_amendment,
        "manifest": args.manifest,
        "dataset_conversion": args.dataset_conversion,
        "encoder_report": args.encoder_report,
        "token_report": args.token_report,
        "training_summary": args.training_summary,
        "schedule": args.schedule,
        "dispatch_contract": args.dispatch_contract,
        "storage_amendment": args.storage_amendment,
        "storage_report": args.storage_report,
        "storage_summary": args.storage_summary,
        "analysis": args.analysis,
        "final_decision": args.final_decision,
        "report": args.report,
        "detachment_proof": (
            args.output_root / "detachment_proof.json"
        ),
        "launch_smoke": args.output_root / "launch_smoke.json",
        "throughput": (
            args.output_root
            / "throughput_first_20_minutes.json"
        ),
        "completion_ledger": (
            args.output_root / "completion_ledger.json"
        ),
        "full_execution_summary": (
            args.output_root / "full_execution_summary.json"
        ),
    }
    missing = [
        label for label, path in paths.items() if not path.exists()
    ]
    if missing:
        raise ValueError(f"missing final artifacts: {missing}")
    document: dict[str, Any] = {
        "schema_version": (
            "pact_frontend_screen_provenance_v1"
        ),
        "screen_not_confirmatory": True,
        "decision": token,
        "root_commit_before_finalization": git_commit(
            args.root.resolve()
        ),
        "act_submodule_commit": git_commit(
            args.root.resolve() / "submodules/act"
        ),
        "schedule_sha256": schedule["schedule_sha256"],
        "dispatch_contract_sha256": dispatch[
            "dispatch_contract_sha256"
        ],
        "analysis_sha256": analysis["analysis_sha256"],
        "final_decision_sha256": decision[
            "final_decision_sha256"
        ],
        "expected_rollouts": int(schedule["rollouts"]),
        "scientific_results_present": len(result_paths),
        "driver_results_present": len(driver_paths),
        "reconciled_result_target": expected,
        "frozen_artifacts": {
            label: {
                "path": str(path.resolve()),
                "sha256": file_hash(path),
            }
            for label, path in paths.items()
        },
        "model_artifacts": dispatch["frozen_inputs"][
            "checkpoints"
        ]
        + dispatch["frozen_inputs"]["surface_encoders"],
        "storage": {
            "compacted_rows_verified": len(storage_archives),
            "excluded_intact_schedule_indices": sorted(exclusions),
            "original_payloads_recoverable": True,
            "outcome_based_selection": False,
            "endpoint_values_emitted_during_compaction": False,
            "compatibility_link": str(
                (
                    args.output_root / "execution_summary.json"
                ).resolve()
            ),
        },
        "weights_committed": False,
        "rollout_h5_committed": False,
        "videos_committed": False,
        "pushed": False,
    }
    document["provenance_sha256"] = canonical_hash(document)
    args.provenance_out.parent.mkdir(
        parents=True, exist_ok=True
    )
    args.provenance_out.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    print(document["provenance_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
