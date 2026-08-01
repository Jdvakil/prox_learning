#!/usr/bin/env python3
"""Freeze the outcome-blind contact worker-count amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OLD_WORKERS = 8
PEAK_8_MIB = 14170
MEMORY_CEILING_MIB = 19000
WORKER_CAP = 12
MINIMUM_AMENDED_WORKERS = 10
GPU_ABORT_THRESHOLD_MIB = 20000
ANALYZER_SHA256 = "e2d9a5061e3a26599fa03a9d4f147ceda8386eb8ad87481f5674d7022f681589"
MEMORY_SOURCE = {
    "record_path": (
        "/root/.codex/sessions/2026/07/28/"
        "rollout-2026-07-28T04-05-13-019fa6e5-ec5b-7cf0-80bd-6033dd450620.jsonl"
    ),
    "record_timestamp_utc": "2026-07-28T04:07:03.521Z",
    "observed_nvidia_smi_summary": "14170 MiB, 23028 MiB, 69 %",
    "worker_count": OLD_WORKERS,
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def result_paths(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(path.resolve()) for path in root.rglob("result.json") if path.is_file())


def build(*, artifact_root: Path, checked_utc: str) -> dict[str, Any]:
    results = result_paths(artifact_root)
    if results:
        raise ValueError(
            "worker amendment is void after evaluation results exist: "
            f"{len(results)} result.json files"
        )
    per_worker = PEAK_8_MIB / OLD_WORKERS
    uncapped = math.floor(MEMORY_CEILING_MIB / per_worker)
    capped = min(WORKER_CAP, uncapped)
    selected = OLD_WORKERS if capped < MINIMUM_AMENDED_WORKERS else capped
    document: dict[str, Any] = {
        "schema_version": "pact_contact_worker_amendment_v1",
        "created_utc": checked_utc,
        "amendment": {
            "field": "design.workers",
            "old_count": OLD_WORKERS,
            "new_count": selected,
            "only_worker_count_changed": True,
            "rows_changed": 0,
        },
        "memory_calculation": {
            "peak_8_mib": PEAK_8_MIB,
            "peak_source": MEMORY_SOURCE,
            "per_worker_mib": per_worker,
            "ceiling_mib": MEMORY_CEILING_MIB,
            "formula": "floor(19000 / (peak_8 / 8)), capped at 12",
            "uncapped_worker_count": uncapped,
            "capped_worker_count": capped,
            "selected_worker_count": selected,
            "projected_selected_peak_mib": per_worker * selected,
            "projected_ceiling_headroom_mib": MEMORY_CEILING_MIB - per_worker * selected,
            "runtime_abort_threshold_mib": GPU_ABORT_THRESHOLD_MIB,
        },
        "zero_results_proof": {
            "artifact_root": str(artifact_root.resolve()),
            "pattern": "**/result.json",
            "result_file_count": 0,
            "matching_paths": [],
            "checked_utc": checked_utc,
        },
        "outcome_blinding": {
            "no_outcome_had_been_observed": True,
            "endpoint_fields_read": False,
            "outcome_values_read": False,
        },
        "reason": (
            "Pre-dispatch throughput amendment selected from the largest recorded "
            "eight-worker GPU-memory peak, with the prescribed 19000 MiB ceiling."
        ),
        "unchanged_contract": {
            "instances": 100,
            "arms": ["ACT", "PACT", "PACT_ZERO", "PACT_PERMUTED"],
            "checkpoint_seeds": [3101, 3102, 3103],
            "rollouts": 1200,
            "max_control_steps": 900,
            "analyzer_sha256": ANALYZER_SHA256,
            "endpoint_analysis_decision_thresholds_changed": False,
        },
    }
    document["worker_amendment_sha256"] = canonical_hash(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    document = build(artifact_root=args.artifact_root, checked_utc=utc_now())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["worker_amendment_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
