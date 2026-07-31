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
