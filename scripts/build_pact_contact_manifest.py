#!/usr/bin/env python3
"""Build or verify the fresh contact-endpoint evaluation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "submodules/molmospaces"))

from molmo_spaces.configs.camera_configs import FrankaSkinHybridWristOnlyCameraSystem
from pact_contact_endpoint_contract import build_manifest, sha256_file


OUTPUT = ROOT / "configs/pact_contact_endpoint_manifest_v1.json"
PRIOR_MANIFESTS = {
    "collision_candidate_v1": ROOT / "configs/pact_collision_candidate_manifest_v1.json",
    "collision_candidate_v2": ROOT / "configs/pact_collision_candidate_manifest_v2.json",
    "confirmatory_r2": ROOT / "configs/pact_confirmatory_r2_manifest_v1.json",
    "frontend_screen": ROOT / "configs/pact_frontend_screen_manifest_v1.json",
}
SOURCE_FILES = {
    "manifest_builder": Path(__file__),
    "manifest_contract": ROOT / "scripts/pact_contact_endpoint_contract.py",
    "preregistration": ROOT / "configs/pact_contact_endpoint_preregistration_v1.json",
    "preregistration_narrative": ROOT / "docs/PACT_CONTACT_ENDPOINT_PREREGISTRATION.md",
    "scene_xml": ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml",
    "scene_metadata": ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_collision_corridor_metadata.json",
    "camera_config": ROOT / "submodules/molmospaces/molmo_spaces/configs/camera_configs.py",
    "sampler": ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    "contact_taxonomy": ROOT / "submodules/molmospaces/molmo_spaces/tasks/pact_contact_audit.py",
    "evaluation_runtime": ROOT / "submodules/act/eval_pact_contact_endpoint_row.py",
}


def build() -> dict:
    prior = {label: json.loads(path.read_text()) for label, path in PRIOR_MANIFESTS.items()}
    prior_ids = {
        row["episode_id"]
        for document in prior.values()
        for row in document["rows"]
    }
    camera = FrankaSkinHybridWristOnlyCameraSystem()
    names = [spec.name for spec in camera.cameras]
    if names[0] != "wrist_camera" or len(names[1:]) != 40:
        raise ValueError("contact manifest camera/sensor order changed")
    return build_manifest(
        source_hashes={label: sha256_file(path) for label, path in SOURCE_FILES.items()},
        sensor_names=names[1:],
        excluded_episode_ids=prior_ids,
        excluded_manifests={
            label: document["manifest_sha256"] for label, document in prior.items()
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = build()
    if args.check:
        if not OUTPUT.exists() or json.loads(OUTPUT.read_text()) != document:
            print("contact manifest differs from deterministic regeneration")
            return 1
        print(f"contact manifest OK {document['manifest_sha256']}")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
