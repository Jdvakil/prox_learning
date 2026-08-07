#!/usr/bin/env python3
"""Build or verify the attempt-2 policy-evaluation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules/molmospaces"
for path in (ROOT / "scripts", MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_geometry_generalization_v2_contract import load_manifest as load_phase0  # noqa: E402
from pact_geometry_generalization_v2_main_contract import (  # noqa: E402
    build_manifest,
    sha256_file,
    validate_manifest,
)


OUTPUT = ROOT / "configs/pact_geometry_generalization_v2.json"
PHASE0 = ROOT / "configs/pact_geometry_generalization_v2_phase0.json"
EXPERT = ROOT / "diagnostics_output/pact_geometry_generalization_v2/expert_screen.json"
SOURCE_FILES = {
    "manifest_builder": Path(__file__),
    "main_contract": ROOT / "scripts/pact_geometry_generalization_v2_main_contract.py",
    "schedule_builder": ROOT / "scripts/build_pact_geometry_v2_schedule.py",
    "dispatch_builder": ROOT / "scripts/freeze_pact_geometry_v2_dispatch.py",
    "supervisor": ROOT / "scripts/run_pact_geometry_v2_supervisor.py",
    "detached_launcher": ROOT / "scripts/launch_pact_geometry_v2_detached.py",
    "full_launcher": ROOT / "scripts/launch_pact_geometry_v2_full.py",
    "detachment_proof": ROOT / "scripts/prove_pact_geometry_v2_detachment.py",
    "analysis": ROOT / "scripts/analyze_pact_geometry_generalization.py",
    "evaluator": ROOT / "submodules/act/eval_pact_geometry_generalization_v2_row.py",
    "contact_evaluator": ROOT / "submodules/act/eval_pact_contact_endpoint_row.py",
    "samplers": ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    "scene_xml": ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml",
    "contact_taxonomy": ROOT / "submodules/molmospaces/molmo_spaces/tasks/pact_contact_audit.py",
    "storage_compactor": ROOT / "scripts/compact_pact_geometry_storage.py",
    "throughput_monitor": ROOT / "scripts/measure_pact_geometry_throughput.py",
    "attempt2_respec": Path(
        "/root/prox_learning_hybrid_safety/docs/PACT_GEOMETRY_GENERALIZATION_RESPEC.md"
    ),
}


def build() -> dict:
    phase0 = load_phase0(PHASE0)
    expert = json.loads(EXPERT.read_text())
    expert_payload = dict(expert)
    observed = expert_payload.pop("expert_screen_sha256", None)
    from pact_geometry_generalization_v2_main_contract import sha256_payload

    if observed != sha256_payload(expert_payload):
        raise ValueError("attempt-2 expert screen self-hash mismatch")
    if expert.get("continue_to_policy_evaluation") is not True:
        raise ValueError("attempt-2 expert screen did not authorize policy evaluation")
    document = build_manifest(
        phase0_manifest=phase0,
        expert_screen=expert,
        source_hashes={name: sha256_file(path) for name, path in SOURCE_FILES.items()},
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = build()
    if args.check:
        if not OUTPUT.exists() or json.loads(OUTPUT.read_text()) != document:
            print("attempt-2 policy manifest differs from deterministic regeneration")
            return 1
        validate_manifest(document)
        print(document["manifest_sha256"])
        return 0
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
