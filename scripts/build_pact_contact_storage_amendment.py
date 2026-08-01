#!/usr/bin/env python3
"""Bind the outcome-blind contact storage rule to the frozen dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = [0, 1199]


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


def validate(document: dict[str, Any], key: str, label: str) -> str:
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
    schedule_sha = validate(schedule, "schedule_sha256", "schedule")
    dispatch_sha = validate(dispatch, "dispatch_contract_sha256", "dispatch")
    validate(manifest, "manifest_sha256", "manifest")
    output_root = args.output_root.resolve()
    if (
        schedule.get("schema_version") != "pact_contact_endpoint_schedule_v1"
        or schedule.get("rollouts") != 1200
        or dispatch.get("schema_version") != "pact_contact_endpoint_dispatch_v1"
        or dispatch["scientific_schedule"]["schedule_sha256"] != schedule_sha
        or dispatch["execution"]["output_root"] != str(output_root)
        or dispatch["storage"]["eligible_rows_compacted"] != list(range(1, 1199))
    ):
        raise ValueError("dispatch does not freeze contact storage")
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise ValueError("storage must be frozen before any rollout output exists")
    compactor = ROOT / "scripts/compact_pact_contact_storage.py"
    document: dict[str, Any] = {
        "schema_version": "pact_contact_storage_amendment_v1",
        "schedule_sha256": schedule_sha,
        "schedule_file_sha256": file_hash(args.schedule),
        "dispatch_contract_sha256": dispatch_sha,
        "dispatch_contract_file_sha256": file_hash(args.dispatch),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": file_hash(args.manifest),
        "analyzer_sha256": file_hash(args.analyzer),
        "compactor_sha256": file_hash(compactor),
        "output_root": str(output_root),
        "excluded_intact_schedule_indices": EXCLUDED,
        "authorization": {
            "source": "frozen preregistration and disk-capacity requirement",
            "keep_fully_intact": EXCLUDED,
            "other_rows": "endpoint-complete summary plus original payload SHA-256/size inventory",
        },
        "compaction": {
            "contact_frame_detail_already_suppressed_at_collection": True,
            "trajectory_and_video_payload_bytes_deleted": True,
            "deleted_payload_bytes_recoverable": False,
            "endpoint_fields_retained": True,
            "original_payload_hash_and_size_retained": True,
            "compact_result_frozen_analyzer_compatible": True,
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
            "rows_complete_at_freeze": 0,
            "endpoint_outcomes_inspected_before_amendment": False,
            "outcome_based_selection": False,
            "endpoint_values_emitted_by_compactor": False,
        },
    }
    document["storage_amendment_sha256"] = canonical_hash(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["storage_amendment_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
