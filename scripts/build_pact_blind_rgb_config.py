#!/usr/bin/env python3
"""Freeze the blind-RGB design onto the blur sweep's 25 C0 instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pact_blind_rgb_contract import (
    ENVIRONMENT_VERSION,
    SCHEMA_VERSION,
    VISION_CONDITIONS,
    sha256_file,
    sha256_payload,
    validate_manifest,
)
from pact_blur_sweep_contract import load_manifest as load_blur_manifest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/pact_blur_sweep_v1.json"
OUTPUT = ROOT / "configs/pact_blind_rgb_v1.json"
PROTECTED = (
    ROOT / "docs/PACT_BLUR_SWEEP.md",
    ROOT / "diagnostics_output/pact_blur_sweep/analysis.json",
    ROOT / "diagnostics_output/pact_blur_sweep/final_decision.json",
    ROOT / "docs/PACT_CONTACT_ENDPOINT_DECISION.md",
    ROOT / "diagnostics_output/pact_contact_endpoint/analysis.json",
    ROOT / "diagnostics_output/pact_contact_endpoint/final_decision.json",
    ROOT / "docs/PACT_GEOMETRY_GENERALIZATION_V3.md",
    ROOT / "diagnostics_output/pact_geometry_generalization_v3/analysis.json",
    ROOT / "diagnostics_output/pact_geometry_generalization_v3/final_decision.json",
)


def build(source: dict[str, Any], source_path: Path) -> dict[str, Any]:
    rows = []
    for index, original in enumerate(source["rows"]):
        row = dict(original)
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "environment_version": ENVIRONMENT_VERSION,
                "blind_role_index": index,
                "role": "blind_rgb_policy_eval",
                "role_index": index,
                "instance_cluster_id": f"blind_rgb_policy_eval:{index:02d}",
                "source_blur_row_sha256": original["row_sha256"],
            }
        )
        row.pop("blur_role_index", None)
        row.pop("row_sha256", None)
        row["row_sha256"] = sha256_payload(row)
        rows.append(row)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "intervention": {
            "field": "wrist RGB only",
            "blind_fill_rgb_0_to_1": [0.485, 0.456, 0.406],
            "rationale": "ImageNet mean becomes exactly zero after ACT normalization",
            "inference_only": True,
            "blur_and_blind_mutually_exclusive": True,
        },
        "vision_conditions": list(VISION_CONDITIONS),
        "instance_count": 25,
        "sensor_names": source["sensor_names"],
        "sensor_order_sha256": source["sensor_order_sha256"],
        "source_blur_manifest": {
            "path": str(source_path.resolve()),
            "file_sha256": sha256_file(source_path),
            "manifest_sha256": source["manifest_sha256"],
            "instances_reused": 25,
            "policy_outcomes_read_for_selection": False,
        },
        "planned_design": {
            "arms": ["ACT", "PACT", "PACT_PERMUTED"],
            "checkpoint_seeds": [3101, 3102, 3103],
            "vision_conditions": list(VISION_CONDITIONS),
            "instances": 25,
            "instances_shared_across_every_condition_arm_seed": True,
            "rollouts": 450,
            "workers": 12,
            "no_retraining": True,
        },
        "predeclared_expected_outcome": {
            "task_success": "collapses for every blind arm because proximity does not localize the cup",
            "hazard_contact": "PACT_BLIND remains lower than ACT_BLIND",
            "permitted_claim_if_observed": "proximity alone keeps the arm safe but cannot do the task",
            "task_success_win_claim_precluded_by_expectation": True,
        },
        "frozen_artifacts": source["frozen_artifacts"],
        "no_upstream_merge": source["no_upstream_merge"],
        "protected_scientific_artifacts": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in PROTECTED
        },
        "rows": rows,
    }
    document["manifest_sha256"] = sha256_payload(document)
    validate_manifest(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace blind-RGB config: {args.output}")
    document = build(load_blur_manifest(args.source), args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
