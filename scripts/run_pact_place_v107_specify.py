#!/usr/bin/env python3
"""V10.7 Step 1: write the immutable specification before anything executes.

Binds every runner, audit script, contract, geometry file, test, sealed input
JSON/NPZ, and base scene by raw SHA-256. Every later stage re-verifies this
specification and fails closed on drift.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v107_contract import (  # noqa: E402
    CAUSAL_LINK_TOKENS,
    CAUSAL_MAX_SIDE_RATIO,
    CAUSAL_MIN_CHANGED_SENSORS,
    CAUSAL_MIN_CHANGED_VALUES,
    CAUSAL_MIN_ONSET_FRAMES,
    CAUSAL_MIN_ONSET_SECONDS,
    CLEARANCE_FLOOR_M,
    CONTRACT_VERSION_V107,
    ENVIRONMENT_VERSION,
    IMPLEMENTATION_PATHS,
    N_GROUPS,
    N_POOL_ROWS,
    N_REVIEW_VIDEOS,
    POOL_MASTER_SEED,
    POOL_MIN_CLEAN,
    POOL_MIN_CLEAN_PER_POSE,
    POOL_MIN_CLEAN_PER_SIDE,
    POOL_MIN_CLEAN_PER_SIDE_POSE,
    POOL_STREAM,
    POSE_IDS,
    PROXIMITY_TENSOR_SHAPE,
    RISK_BAND_M,
    SAMPLER_CLASS,
    SEALED_INPUTS,
    SPEC_ROOT,
    THRESHOLD_NEAR_M,
    empty_authorization,
    file_hashes,
    implementation_digest,
    write_immutable_create_only,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / SPEC_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")

    sealed = file_hashes(SEALED_INPUTS)
    implementation = file_hashes(IMPLEMENTATION_PATHS)
    missing = sorted(
        path for section in (sealed, implementation)
        for path, entry in section.items() if not entry["present"]
    )
    if missing:
        raise SystemExit(f"cannot specify: missing bound files {missing}")

    document = {
        "schema_version": "pact_place_v107_specification_v1",
        "contract_version": CONTRACT_VERSION_V107,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "written_before_execution": True,
        "role": "immutable specification bound before any V10.7 stage runs",
        "treats_v106_geometry_results_as_historical_inputs": True,
        "modifies_any_existing_artifact": False,
        "sealed_inputs": sealed,
        "implementation_files": implementation,
        "implementation_digest": implementation_digest(IMPLEMENTATION_PATHS),
        "hash_discipline": {
            "score_npz_written_before_its_manifest_json": True,
            "manifest_json_binds_npz_raw_sha256": True,
            "every_stage_reverifies_and_fails_closed_on_drift": True,
            "raw_and_canonical_payload_hashes_are_distinct": True,
        },
        "relevance_test": {
            "name": "natural_clearance_band_plus_six_group_causal",
            "requires_universal_clearance_m": CLEARANCE_FLOOR_M,
            "requires_all_group_minima_in_band_m": list(RISK_BAND_M),
            "n_groups": N_GROUPS,
            "requires_six_group_causal": True,
            "cardinal_tcp_contact_perturbation_is_a_gate": False,
            "retirement_rationale": (
                "The all-six cardinal-TCP contact test measured the reach of a "
                "straight-line TCP displacement, not physical reachability, and "
                "is retired as a gate. It may run as a non-gating diagnostic."
            ),
        },
        "certification": {
            "instruments": [
                "analytic GJK", "hardened signed mj_geomDistance",
                "live data.contact", "place contact audit",
            ],
            "covers": "all six group minima and every threshold-near witness",
            "threshold_near_m": THRESHOLD_NEAR_M,
            "any_instrument_disagreement_fails_closed": True,
        },
        "causal": {
            "per_group": True,
            "n_groups": N_GROUPS,
            "tensor_shape": list(PROXIMITY_TENSOR_SHAPE),
            "min_changed_values": CAUSAL_MIN_CHANGED_VALUES,
            "min_changed_sensors": CAUSAL_MIN_CHANGED_SENSORS,
            "required_link_tokens": list(CAUSAL_LINK_TOKENS),
            "min_onset_frames": CAUSAL_MIN_ONSET_FRAMES,
            "min_onset_seconds": CAUSAL_MIN_ONSET_SECONDS,
            "max_side_ratio": CAUSAL_MAX_SIDE_RATIO,
            "numeric_artifact_supports_independent_reaggregation": True,
        },
        "contact_diagnostic": {
            "gating": False,
            "carried_target_moves_rigidly_with_the_gripper": True,
            "allowlist": "gripper-pad to carried-target grasp contacts only",
            "tracks_worsening_baseline_penetrations": True,
            "requires_instrument_agreement": True,
        },
        "pool": {
            "stream": POOL_STREAM, "master_seed": POOL_MASTER_SEED,
            "n_rows": N_POOL_ROWS, "frozen_before_row_0": True,
            "min_clean": POOL_MIN_CLEAN,
            "min_clean_per_side": POOL_MIN_CLEAN_PER_SIDE,
            "min_clean_per_pose": POOL_MIN_CLEAN_PER_POSE,
            "min_clean_per_side_pose": POOL_MIN_CLEAN_PER_SIDE_POSE,
            "checked_before_publishing_any_packet": True,
        },
        "review": {
            "n_videos": N_REVIEW_VIDEOS,
            "composition": "three natural strict-clean successes, three natural failures",
            "agent_creates_human_approval": False,
            "agent_runs_phase0": False,
        },
        "pose_ids": list(POSE_IDS),
        **empty_authorization(),
    }
    hashes = write_immutable_create_only(
        output_root / "specification.json", document
    )
    print(json.dumps({
        "specification_written": True,
        "n_sealed_inputs": len(sealed),
        "n_implementation_files": len(implementation),
        "implementation_digest": document["implementation_digest"],
        **hashes,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
