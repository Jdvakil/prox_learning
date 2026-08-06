#!/usr/bin/env python3
"""Build or verify the frozen attempt-2 Phase-0 geometry manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for path in (ROOT / "scripts", MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from molmo_spaces.configs.camera_configs import (  # noqa: E402
    FrankaSkinHybridWristOnlyCameraSystem,
)
from pact_geometry_generalization_v2_contract import (  # noqa: E402
    build_manifest,
    sha256_file,
    sha256_payload,
    validate_manifest,
)


OUTPUT = ROOT / "configs" / "pact_geometry_generalization_v2_phase0.json"
SAMPLER_PATH = MOLMO / "molmo_spaces/tasks/enclosure_reach.py"
SCENE_XML = MOLMO / "molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml"
RESPEC = Path("/root/prox_learning_hybrid_safety/docs/PACT_GEOMETRY_GENERALIZATION_RESPEC.md")
EXPECTED_BASE_SHA256 = "ccd5f752f5f727d76931409798a7bda7bc2401b53a842cb3ea16d02e2d1869cc"
EXPECTED_XML_SHA256 = "f8c04b07b9416593eb60ad4797ccbae91f7d3524effd30438ef552e5a2d75540"
EXPECTED_V1_MANIFEST_SHA256 = "33e48ab83dfe398fbeb78f64565312c48a5a8b09cb1a873a2a2521e06fcbe7b2"
EXPECTED_V1_SCREEN_SHA256 = "3bf7d5c8f86814b9c10308c10cf1576488e992d0d20564359cb911d312d78a2c"
EXPECTED_POLICY_REGISTRY_FILE_SHA256 = "b644b3c1307ab296d4424a502820c1c6b776c2818d7cd5ab463a8b931d86d5f7"
EXPECTED_POLICY_REGISTRY_SHA256 = "d2e643d111d3d0abb7cb96d5643454e3c223fa9f68e5896ff29557702931e58e"
EXPECTED_ENCODER_SHA256 = "6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206"
EXPECTED_TOKEN_PLAN_FILE_SHA256 = "f4843f696d2c9cb21531668bbcc6c7507da8226d4165cd0c023ca61caf8e8784"
EXPECTED_FROZEN_FILES = {
    "contact_endpoint_decision": "6ea97ea789e1c5d57572c7e231b30c867881f0e21b81ad11f950cef9bf860f1b",
    "contact_endpoint_analysis": "4bbf25c97472bbfd8f13c3352b6b1c8afae9284cd00351e0502149362ccde1f7",
    "contact_endpoint_final_decision": "52ce515bc09771d9ecdc8a39e31111f66e7e632f5949e82ed2497908ac2458bf",
}
SOURCE_FILES = {
    "manifest_builder": Path(__file__),
    "manifest_contract": ROOT / "scripts/pact_geometry_generalization_v2_contract.py",
    "expert_row_runner": ROOT / "scripts/run_pact_geometry_v2_expert_row.py",
    "expert_supervisor": ROOT / "scripts/run_pact_geometry_v2_expert_supervisor.py",
    "expert_screen_analysis": ROOT / "scripts/analyze_pact_geometry_v2_expert_screens.py",
    "sampler_and_expert": SAMPLER_PATH,
    "scene_xml": SCENE_XML,
    "scene_metadata": MOLMO / "molmo_spaces/data_generation/custom_scenes/pact_collision_corridor_metadata.json",
    "datagen_config": MOLMO / "molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py",
    "contact_taxonomy": MOLMO / "molmo_spaces/tasks/pact_contact_audit.py",
    "camera_config": MOLMO / "molmo_spaces/configs/camera_configs.py",
    "frozen_policy_registry": ROOT / "diagnostics_output/pact_contact_endpoint/policy_training.json",
    "frozen_permuted_token_plan": ROOT / "diagnostics_output/pact_contact_endpoint/token_plan.json",
    "attempt2_respec": RESPEC,
}
FROZEN_FILES = {
    "contact_endpoint_decision": ROOT / "docs/PACT_CONTACT_ENDPOINT_DECISION.md",
    "contact_endpoint_analysis": ROOT / "diagnostics_output/pact_contact_endpoint/analysis.json",
    "contact_endpoint_final_decision": ROOT / "diagnostics_output/pact_contact_endpoint/final_decision.json",
}


def class_source(text: str) -> bytes:
    start = text.index("class PactCollisionCorridorSampler(")
    end = text.index("\n\nclass PactCollisionCorridorControlSampler", start)
    return text[start:end].rstrip().encode() + b"\n"


def integrity_records() -> tuple[dict, dict, dict, dict]:
    base_hash = hashlib.sha256(class_source(SAMPLER_PATH.read_text())).hexdigest()
    if base_hash != EXPECTED_BASE_SHA256:
        raise ValueError("base PactCollisionCorridorSampler changed from attempt 1")
    xml_hash = sha256_file(SCENE_XML)
    if xml_hash != EXPECTED_XML_SHA256:
        raise ValueError("pact_collision_corridor.xml changed from attempt 1")

    v1_manifest_path = ROOT / "configs/pact_geometry_generalization_v1.json"
    v1_screen_path = ROOT / "diagnostics_output/pact_geometry_generalization/expert_screen.json"
    v1_manifest = json.loads(v1_manifest_path.read_text())
    v1_screen = json.loads(v1_screen_path.read_text())
    if v1_manifest.get("manifest_sha256") != EXPECTED_V1_MANIFEST_SHA256:
        raise ValueError("attempt-1 manifest self-hash changed")
    if v1_screen.get("expert_screen_sha256") != EXPECTED_V1_SCREEN_SHA256:
        raise ValueError("attempt-1 expert-screen self-hash changed")

    frozen_observed = {name: sha256_file(path) for name, path in FROZEN_FILES.items()}
    if frozen_observed != EXPECTED_FROZEN_FILES:
        raise ValueError("frozen contact-endpoint artifacts changed")
    return (
        {
            "current_base_class_source_sha256": base_hash,
            "attempt1_base_class_source_sha256": EXPECTED_BASE_SHA256,
            "byte_identical_to_attempt1": True,
        },
        {
            "current_sha256": xml_hash,
            "attempt1_sha256": EXPECTED_XML_SHA256,
            "byte_identical_to_attempt1": True,
        },
        {
            "manifest_sha256": EXPECTED_V1_MANIFEST_SHA256,
            "expert_screen_sha256": EXPECTED_V1_SCREEN_SHA256,
            "manifest_file_sha256": sha256_file(v1_manifest_path),
            "expert_screen_file_sha256": sha256_file(v1_screen_path),
            "decision": "GEOMETRY_TEST_INCONCLUSIVE",
            "preserved_as_distinct_attempt": True,
        },
        frozen_observed,
    )


def policy_artifact_integrity() -> dict:
    registry_path = ROOT / "diagnostics_output/pact_contact_endpoint/policy_training.json"
    if sha256_file(registry_path) != EXPECTED_POLICY_REGISTRY_FILE_SHA256:
        raise ValueError("frozen contact-endpoint policy registry file changed")
    registry = json.loads(registry_path.read_text())
    if registry.get("policy_registry_sha256") != EXPECTED_POLICY_REGISTRY_SHA256:
        raise ValueError("frozen contact-endpoint policy registry identity changed")
    token_plan_path = ROOT / "diagnostics_output/pact_contact_endpoint/token_plan.json"
    if sha256_file(token_plan_path) != EXPECTED_TOKEN_PLAN_FILE_SHA256:
        raise ValueError("frozen PACT_PERMUTED token plan changed")
    records = []
    for seed in (3101, 3102, 3103):
        for arm in ("ACT", "PACT"):
            item = registry["seeds"][str(seed)][arm]
            path = Path(item["checkpoint_path"])
            observed = sha256_file(path)
            if observed != item["checkpoint_sha256"]:
                raise ValueError(f"checkpoint hash mismatch for {arm} seed {seed}")
            records.append(
                {
                    "arm": arm,
                    "checkpoint_seed": seed,
                    "path": str(path),
                    "checkpoint_sha256": observed,
                }
            )
    encoder_path = Path(registry["seeds"]["3101"]["PACT"]["surface_encoder_path"])
    encoder_hash = sha256_file(encoder_path)
    if encoder_hash != EXPECTED_ENCODER_SHA256:
        raise ValueError("frozen 32-D encoder hash mismatch")
    return {
        "policy_registry_file_sha256": EXPECTED_POLICY_REGISTRY_FILE_SHA256,
        "policy_registry_sha256": EXPECTED_POLICY_REGISTRY_SHA256,
        "token_plan_file_sha256": EXPECTED_TOKEN_PLAN_FILE_SHA256,
        "token_plan_sha256": json.loads(token_plan_path.read_text())["token_plan_sha256"],
        "checkpoints": records,
        "checkpoint_count": len(records),
        "encoder_path": str(encoder_path),
        "encoder_sha256": encoder_hash,
        "verified_from_bytes": True,
    }


def build() -> dict:
    base, scene, attempt1, frozen = integrity_records()
    policy_artifacts = policy_artifact_integrity()
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
    payload.update(
        {
            "base_sampler_integrity": base,
            "scene_xml_integrity": scene,
            "attempt1_integrity": attempt1,
            "frozen_contact_endpoint_hashes": frozen,
            "frozen_policy_artifacts": policy_artifacts,
            "execution_authorization": {
                "phase0a": "authorized_by_respec",
                "phase0b": "only_after_frozen_phase0a_selection",
                "policy_evaluation": "only_after_phase0b_gate",
            },
        }
    )
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
            print("attempt-2 Phase-0 manifest differs from deterministic regeneration")
            return 1
        print(document["manifest_sha256"])
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
