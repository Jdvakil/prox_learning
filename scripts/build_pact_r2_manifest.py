#!/usr/bin/env python3
"""Build or verify the fresh, R1-disjoint PACT confirmatory R2 manifest."""

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
from pact_r2_contract import build_manifest, sha256_file

OUTPUT = ROOT / "configs" / "pact_confirmatory_r2_manifest_v1.json"
R1_MANIFEST = ROOT / "configs" / "pact_collision_candidate_manifest_v2.json"
SOURCE_FILES = {
    "r2_manifest_builder": Path(__file__),
    "r2_manifest_contract": ROOT / "scripts" / "pact_r2_contract.py",
    "r2_preregistration": ROOT / "configs" / "pact_r2_preregistration_v1.json",
    "r2_preregistration_narrative": ROOT / "docs" / "PACT_R2_PREREGISTRATION.md",
    "scene_xml": (
        ROOT
        / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes"
        / "pact_collision_corridor.xml"
    ),
    "scene_metadata": (
        ROOT
        / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes"
        / "pact_collision_corridor_metadata.json"
    ),
    "camera_config": (
        ROOT / "submodules/molmospaces/molmo_spaces/configs/camera_configs.py"
    ),
    "sampler": (
        ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py"
    ),
    "contact_taxonomy": (
        ROOT
        / "submodules/molmospaces/molmo_spaces/tasks/pact_contact_audit.py"
    ),
    "evaluation_runtime": ROOT / "submodules/act/eval_pact_collision_row.py",
}


def build() -> dict:
    r1 = json.loads(R1_MANIFEST.read_text())
    r1_episode_ids = {row["episode_id"] for row in r1["rows"]}
    camera = FrankaSkinHybridWristOnlyCameraSystem()
    names = [spec.name for spec in camera.cameras]
    if names[0] != "wrist_camera":
        raise ValueError("first R2 camera must be wrist_camera")
    return build_manifest(
        source_hashes={
            name: sha256_file(path) for name, path in SOURCE_FILES.items()
        },
        sensor_names=names[1:],
        r1_episode_ids=r1_episode_ids,
        r1_manifest_sha256=r1["manifest_sha256"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = build()
    if args.check:
        if not OUTPUT.exists() or json.loads(OUTPUT.read_text()) != document:
            print("R2 manifest differs from deterministic regeneration")
            return 1
        print(f"R2 manifest OK {document['manifest_sha256']}")
        return 0
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"manifest_sha256={document['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
