#!/usr/bin/env python3
"""Build or verify the fresh PACT remediation-v2 candidate manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "submodules" / "molmospaces"))

from molmo_spaces.configs.camera_configs import (
    FrankaSkinHybridWristOnlyCameraSystem,
)
from pact_collision_contract import build_manifest, sha256_file

OUTPUT = ROOT / "configs" / "pact_collision_candidate_manifest_v2.json"
SOURCE_FILES = {
    "manifest_builder": Path(__file__),
    "manifest_contract": ROOT / "scripts" / "pact_collision_contract.py",
    "environment_preregistration": ROOT / "configs" / "pact_collision_environment_v2.json",
    "scene_xml": (
        ROOT
        / "submodules"
        / "molmospaces"
        / "molmo_spaces"
        / "data_generation"
        / "custom_scenes"
        / "pact_collision_corridor.xml"
    ),
    "scene_metadata": (
        ROOT
        / "submodules"
        / "molmospaces"
        / "molmo_spaces"
        / "data_generation"
        / "custom_scenes"
        / "pact_collision_corridor_metadata.json"
    ),
    "camera_config": (
        ROOT
        / "submodules"
        / "molmospaces"
        / "molmo_spaces"
        / "configs"
        / "camera_configs.py"
    ),
    "sampler_and_expert": (
        ROOT
        / "submodules"
        / "molmospaces"
        / "molmo_spaces"
        / "tasks"
        / "enclosure_reach.py"
    ),
    "contact_taxonomy": (
        ROOT
        / "submodules"
        / "molmospaces"
        / "molmo_spaces"
        / "tasks"
        / "pact_contact_audit.py"
    ),
    "contact_sampling_hook": (
        ROOT
        / "submodules"
        / "molmospaces"
        / "molmo_spaces"
        / "tasks"
        / "task.py"
    ),
    "rollout_runtime": (
        ROOT
        / "submodules"
        / "molmospaces"
        / "molmo_spaces"
        / "data_generation"
        / "pipeline.py"
    ),
    "sensor_assembly": (
        ROOT
        / "submodules"
        / "molmospaces"
        / "molmo_spaces"
        / "env"
        / "sensors.py"
    ),
    "camera_parameter_sensor": (
        ROOT
        / "submodules"
        / "molmospaces"
        / "molmo_spaces"
        / "env"
        / "sensors_cameras.py"
    ),
    "datagen_config": (
        ROOT
        / "submodules"
        / "molmospaces"
        / "molmo_spaces"
        / "data_generation"
        / "config"
        / "object_manipulation_datagen_configs.py"
    ),
    "collection_runtime": ROOT / "scripts" / "run_pact_collision_collection.py",
    "demo_filter_and_converter": ROOT / "scripts" / "convert_pact_collision_to_act.py",
    "development_audit": ROOT / "scripts" / "analyze_pact_expert_development.py",
    "gate_statistics": ROOT / "scripts" / "pact_gate_statistics.py",
    "gate_analysis": ROOT / "scripts" / "analyze_pact_environment_gate.py",
    "pilot_schedule_builder": ROOT / "scripts" / "build_pact_pilot_schedule.py",
}


def build() -> dict:
    camera = FrankaSkinHybridWristOnlyCameraSystem()
    names = [spec.name for spec in camera.cameras]
    if names[0] != "wrist_camera":
        raise SystemExit(f"first camera must be wrist_camera, got {names[0]!r}")
    proximity_names = names[1:]
    return build_manifest(
        source_hashes={name: sha256_file(path) for name, path in SOURCE_FILES.items()},
        sensor_names=proximity_names,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = build()
    if args.check:
        if not OUTPUT.exists():
            print(f"missing: {OUTPUT}")
            return 1
        if json.loads(OUTPUT.read_text()) != document:
            print("PACT remediation manifest differs from deterministic regeneration")
            return 1
        print(f"manifest OK {document['manifest_sha256']}")
        return 0
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT}")
    print(f"manifest_sha256={document['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
