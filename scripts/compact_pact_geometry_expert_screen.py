#!/usr/bin/env python3
"""Compact all expert-screen contact payloads without outcome selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pact_geometry_generalization_contract import load_manifest, sha256_file, sha256_payload


SUMMARY_KEYS = (
    "collision_free",
    "contact_class_totals",
    "contact_taxonomy_version",
    "first_contact_step",
    "frames_with_contact",
    "maximum_penetration_depth_m",
    "non_target_contact_entries",
    "sample_count",
    "sampling_level",
)


def result_path(root: Path, row: dict[str, Any]) -> Path:
    return (
        root
        / "expert_screen_rows"
        / row["condition_id"]
        / f"{row['role_index']:02d}_{row['episode_id'][:16]}"
        / "result.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    records = []
    for row in manifest["expert_screen_rows"]:
        path = result_path(args.output_root, row)
        original = json.loads(path.read_text())
        if original.get("episode_id") != row["episode_id"] or original.get("row_sha256") != row["row_sha256"]:
            raise ValueError(f"row identity mismatch: {path}")
        if original.get("expert_storage_compaction") is not None:
            records.append(original["expert_storage_compaction"])
            continue
        inventory = {
            "path": str(path),
            "original_size_bytes": path.stat().st_size,
            "original_sha256": sha256_file(path),
        }
        audit = original.get("contact_audit")
        if audit is not None:
            missing = [key for key in SUMMARY_KEYS if key not in audit]
            if missing:
                raise ValueError(f"contact summary fields missing from {path}: {missing}")
            original["contact_audit"] = {key: audit[key] for key in SUMMARY_KEYS}
            original["contact_audit"]["contact_frame_payload_retained"] = False
            original["contact_audit"]["contact_frames"] = []
        original["expert_storage_compaction"] = {
            **inventory,
            "content_transform": "drop_contact_pair_samples_keep_all_frozen_gate_and_contact_summary_fields",
            "outcome_based_selection": False,
            "raw_contact_pair_payload_recoverable": False,
        }
        original.pop("result_sha256", None)
        original["result_sha256"] = sha256_payload(original)
        path.write_text(json.dumps(original, indent=2, sort_keys=True) + "\n")
        records.append(original["expert_storage_compaction"])
    summary = {
        "schema_version": "pact_geometry_expert_storage_compaction_v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "row_count": len(records),
        "all_rows_compacted": len(records) == len(manifest["expert_screen_rows"]),
        "outcome_based_selection": False,
        "records": records,
    }
    summary["storage_compaction_sha256"] = sha256_payload(summary)
    destination = args.output_root / "expert_screen_storage.json"
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary["storage_compaction_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
