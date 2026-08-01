#!/usr/bin/env python3
"""Freeze lossless, outcome-blind storage control for seed replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERIC_COMPACTOR = ROOT / "scripts/compact_pact_r2_storage.py"
WRAPPER = ROOT / "scripts/compact_pact_seed_replication_storage.py"
EXCLUDED = [0, 119]


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != canonical_hash(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return str(observed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--analyzer", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    dispatch = json.loads(args.dispatch.read_text())
    manifest = json.loads(args.manifest.read_text())
    schedule_sha = validate_self_hash(schedule, "schedule_sha256", "schedule")
    dispatch_sha = validate_self_hash(dispatch, "dispatch_contract_sha256", "dispatch")
    validate_self_hash(manifest, "manifest_sha256", "manifest")
    output_root = args.output_root.resolve()
    if (
        schedule.get("schema_version") != "pact_seed_replication_schedule_v1"
        or schedule.get("rollouts") != 120
        or dispatch.get("schema_version") != "pact_seed_replication_dispatch_v1"
        or dispatch["scientific_schedule"]["schedule_sha256"] != schedule_sha
        or dispatch["execution"]["output_root"] != str(output_root)
        or dispatch["storage"]["eligible_rows_losslessly_compacted"] != list(range(1, 119))
    ):
        raise ValueError("dispatch does not freeze seed-replication storage")
    if output_root.exists() and any(item.is_file() for item in output_root.rglob("*")):
        raise ValueError("storage must be frozen before any output exists")
    document: dict[str, Any] = {
        "schema_version": "pact_r2_storage_amendment_v1",
        "seed_replication_storage_schema": ("pact_seed_replication_storage_amendment_v1"),
        "schedule_sha256": schedule_sha,
        "schedule_file_sha256": file_hash(args.schedule),
        "dispatch_contract_sha256": dispatch_sha,
        "dispatch_contract_file_sha256": file_hash(args.dispatch),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": file_hash(args.manifest),
        "analyzer_sha256": file_hash(args.analyzer),
        "compactor_sha256": file_hash(GENERIC_COMPACTOR),
        "seed_replication_compactor_wrapper_sha256": file_hash(WRAPPER),
        "output_root": str(output_root),
        "excluded_intact_schedule_indices": EXCLUDED,
        "authorization": {
            "source": "predeclared seed-replication storage contract",
            "raw_smoke_requirement": "keep_schedule_row_0_fully_unpacked",
            "raw_final_requirement": "keep_schedule_row_119_fully_unpacked",
        },
        "compaction": {
            "archive_codec": "zstd",
            "archive_level": 1,
            "archive_threads": 2,
            "full_original_result_retained_byte_exact": True,
            "original_trajectory_retained_byte_exact": True,
            "verified_before_original_payload_removal": True,
            "compact_result_is_frozen_analyzer_compatible": True,
            "videos_retained_unmodified": True,
        },
        "frozen_scientific_contract": {
            "analysis_changed": False,
            "checkpoint_changed": False,
            "endpoint_changed": False,
            "environment_changed": False,
            "permutation_changed": False,
            "row_order_changed": False,
            "rows_replaced_or_rerun": False,
            "worker_count_changed": False,
        },
        "provenance": {
            "reason": "predeclared filesystem capacity control",
            "rows_complete_at_freeze": 0,
            "endpoint_outcomes_inspected_before_amendment": False,
            "outcome_based_selection": False,
            "outcome_values_emitted_by_compactor": False,
        },
    }
    document["storage_amendment_sha256"] = canonical_hash(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["storage_amendment_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
