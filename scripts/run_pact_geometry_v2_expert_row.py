#!/usr/bin/env python3
"""Run one immutable attempt-2 expert-screen row in a fresh process."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import run_pact_geometry_expert_screen as legacy
from pact_geometry_generalization_v2_contract import load_manifest, retry_seed


def sampler_class(name: str) -> type:
    from molmo_spaces.tasks.enclosure_reach import (
        PactCollisionCorridorAperture095Sampler,
        PactCollisionCorridorPanelHalfY018Sampler,
        PactCollisionCorridorPanelHalfY030Sampler,
        PactCollisionCorridorPanelX058Sampler,
        PactCollisionCorridorPanelX065Sampler,
        PactCollisionCorridorPanelZ085Sampler,
        PactCollisionCorridorPanelZ093Sampler,
    )

    allowed = {
        cls.__name__: cls
        for cls in (
            PactCollisionCorridorPanelX058Sampler,
            PactCollisionCorridorPanelX065Sampler,
            PactCollisionCorridorPanelZ085Sampler,
            PactCollisionCorridorPanelZ093Sampler,
            PactCollisionCorridorPanelHalfY018Sampler,
            PactCollisionCorridorPanelHalfY030Sampler,
            PactCollisionCorridorAperture095Sampler,
        )
    }
    if name not in allowed:
        raise ValueError(f"unregistered v2 sampler class {name!r}")
    return allowed[name]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("phase0a", "phase0b"))
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--instance-index", required=True, type=int)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    rows = (
        manifest["phase0a_rows"]
        if args.phase == "phase0a"
        else manifest["phase0b_candidate_rows"]
    )
    matches = [
        row
        for row in rows
        if row["condition_id"] == args.condition_id
        and int(row["instance_index"]) == args.instance_index
    ]
    if len(matches) != 1:
        raise SystemExit(f"row resolved {len(matches)} times")
    # Phase-specific roots prevent the 8-row envelope map and 12-row composed
    # screen from ever resolving to the same legacy result path.  Summary-only
    # contact auditing preserves every endpoint used by this screen without
    # retaining the very large per-contact sample stream.
    phase_output_root = args.output_root / args.phase
    os.environ["PACT_CONTACT_AUDIT_SUMMARY_ONLY"] = "1"
    legacy.retry_seed = retry_seed
    legacy.sampler_class = sampler_class
    result = legacy.run_row(
        matches[0],
        manifest_sha256=manifest["manifest_sha256"],
        output_root=str(phase_output_root),
    )
    print(
        f"{result.get('condition_id')} {result.get('instance_index')} "
        f"{result.get('status')} clean={result.get('clean_success')}"
    )
    return 0 if result.get("status") in legacy.TERMINAL_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())
