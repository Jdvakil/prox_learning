#!/usr/bin/env python3
"""Build or verify the held-out geometry-generalization manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for path in (ROOT / "scripts", MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from molmo_spaces.configs.camera_configs import FrankaSkinHybridWristOnlyCameraSystem  # noqa: E402
from pact_geometry_generalization_contract import build_manifest, sha256_file  # noqa: E402


OUTPUT = ROOT / "configs" / "pact_geometry_generalization_v1.json"
SAMPLER_PATH = MOLMO / "molmo_spaces/tasks/enclosure_reach.py"
SCENE_XML = MOLMO / "molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml"
SOURCE_FILES = {
    "manifest_builder": Path(__file__),
    "manifest_contract": ROOT / "scripts/pact_geometry_generalization_contract.py",
    "expert_screen_runner": ROOT / "scripts/run_pact_geometry_expert_screen.py",
    "schedule_builder": ROOT / "scripts/build_pact_geometry_schedule.py",
    "frozen_analysis": ROOT / "scripts/analyze_pact_geometry_generalization.py",
    "geometry_evaluator": ROOT / "submodules/act/eval_pact_geometry_generalization_row.py",
    "legacy_evaluator": ROOT / "submodules/act/eval_pact_collision_row.py",
    "sampler_and_expert": SAMPLER_PATH,
    "scene_xml": SCENE_XML,
    "scene_metadata": MOLMO / "molmo_spaces/data_generation/custom_scenes/pact_collision_corridor_metadata.json",
    "datagen_config": MOLMO / "molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py",
    "contact_taxonomy": MOLMO / "molmo_spaces/tasks/pact_contact_audit.py",
    "camera_config": MOLMO / "molmo_spaces/configs/camera_configs.py",
}
FROZEN_RESULT_FILES = {
    "contact_endpoint_decision": ROOT / "docs/PACT_CONTACT_ENDPOINT_DECISION.md",
    "contact_endpoint_analysis": ROOT / "diagnostics_output/pact_contact_endpoint/analysis.json",
    "contact_endpoint_final_decision": ROOT / "diagnostics_output/pact_contact_endpoint/final_decision.json",
    "policy_registry": ROOT / "diagnostics_output/pact_contact_endpoint/policy_training.json",
    "permuted_token_plan": ROOT / "diagnostics_output/pact_contact_endpoint/token_plan.json",
}


def class_source(text: str, *, end_marker: str) -> bytes:
    start = text.index("class PactCollisionCorridorSampler(")
    end = text.index(end_marker, start)
    return text[start:end].rstrip().encode() + b"\n"


def base_sampler_integrity() -> dict[str, str | bool]:
    current = SAMPLER_PATH.read_text()
    committed = subprocess.run(
        ["git", "show", "HEAD:molmo_spaces/tasks/enclosure_reach.py"],
        cwd=MOLMO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    current_block = class_source(
        current,
        end_marker="\n\nclass PactCollisionCorridorControlSampler",
    )
    committed_block = class_source(
        committed,
        end_marker="\nclass PactCollisionCorridorPolicy",
    )
    current_hash = hashlib.sha256(current_block).hexdigest()
    committed_hash = hashlib.sha256(committed_block).hexdigest()
    return {
        "current_base_class_source_sha256": current_hash,
        "committed_base_class_source_sha256": committed_hash,
        "byte_identical": current_block == committed_block,
    }


def committed_scene_integrity() -> dict[str, str | bool]:
    committed = subprocess.run(
        ["git", "show", "HEAD:molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml"],
        cwd=MOLMO,
        check=True,
        capture_output=True,
    ).stdout
    current = SCENE_XML.read_bytes()
    return {
        "current_sha256": hashlib.sha256(current).hexdigest(),
        "committed_sha256": hashlib.sha256(committed).hexdigest(),
        "byte_identical": current == committed,
    }


def build() -> dict:
    base = base_sampler_integrity()
    scene = committed_scene_integrity()
    if base["byte_identical"] is not True:
        raise ValueError("base PactCollisionCorridorSampler changed")
    if scene["byte_identical"] is not True:
        raise ValueError("pact_collision_corridor.xml changed")
    camera = FrankaSkinHybridWristOnlyCameraSystem()
    names = [spec.name for spec in camera.cameras]
    if names[0] != "wrist_camera" or len(names[1:]) != 40:
        raise ValueError("camera/sensor contract changed")
    document = build_manifest(
        source_hashes={name: sha256_file(path) for name, path in SOURCE_FILES.items()},
        sensor_names=names[1:],
    )
    payload = dict(document)
    payload.pop("manifest_sha256")
    payload["frozen_result_hashes"] = {
        name: sha256_file(path) for name, path in FROZEN_RESULT_FILES.items()
    }
    payload["base_sampler_integrity"] = base
    payload["scene_xml_integrity"] = scene
    from pact_geometry_generalization_contract import sha256_payload, validate_manifest

    payload["manifest_sha256"] = sha256_payload(payload)
    validate_manifest(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = build()
    if args.check:
        if not OUTPUT.exists() or json.loads(OUTPUT.read_text()) != document:
            print("geometry manifest differs from deterministic regeneration")
            return 1
        print(document["manifest_sha256"])
        return 0
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
