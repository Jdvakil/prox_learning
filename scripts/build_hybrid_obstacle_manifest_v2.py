#!/usr/bin/env python3
"""Build and freeze the hybrid_obstacle_independent_v2 candidate manifest.

Writes:
    configs/hybrid_obstacle_independent_v2.yaml       the machine-readable contract
    configs/hybrid_obstacle_candidate_manifest_v2.json  the frozen 160 candidate rows
    configs/hybrid_obstacle_manifest_v2_smoke8.json     the bounded smoke subset

Regeneration is deterministic: running this again on the same inputs reproduces
the identical manifest hash. It is safe to re-run, and a test asserts it.

The master seed is fixed at 20260725 and must never be changed after any
simulation result has been observed.

Usage:
    python scripts/build_hybrid_obstacle_manifest_v2.py [--check]

``--check`` regenerates in memory and verifies the committed files match,
without writing anything.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules" / "molmospaces"))

from molmo_spaces.data_generation.episode_manifest import (  # noqa: E402
    CANONICAL_HAZARD_ABSENT,
    CANONICAL_HAZARD_PRESENT,
    DESIGN_OBSTACLE_P,
    HAZARD_ABSENT_COUNT,
    HAZARD_PRESENT_COUNT,
    MANIFEST_VERSION,
    MASTER_SEED,
    MAX_RETRIES_PER_ROW,
    SCENE_TEMPLATE_HOUSE_INDEX,
    SCENE_TEMPLATE_ID,
    STREAM_HAZARD_SCHEDULE,
    STREAM_NAMES,
    TOTAL_CANDIDATES,
    TRAIN_HAZARD_ABSENT,
    TRAIN_HAZARD_PRESENT,
    VAL_HAZARD_ABSENT,
    VAL_HAZARD_PRESENT,
    build_manifest,
    build_smoke_subset,
    sha256_file,
    sha256_payload,
    validate_manifest,
    validate_smoke_subset,
)

MANIFEST_PATH = ROOT / "configs" / "hybrid_obstacle_candidate_manifest_v2.json"
CONTRACT_PATH = ROOT / "configs" / "hybrid_obstacle_independent_v2.yaml"
SMOKE_PATH = ROOT / "configs" / "hybrid_obstacle_manifest_v2_smoke8.json"
SAFETY_STACK_PATH = ROOT / "configs" / "hybrid_safety_stack_v1.json"
ROBOT_MODEL_PATH = ROOT / "assets" / "robots" / "franka_skin" / "model_hybrid.xml"
SCENE_PATH = (
    ROOT / "submodules" / "molmospaces" / "molmo_spaces" / "data_generation" /
    "custom_scenes" / "fumehood.xml"
)

#: MolmoSpaces commit the manifest is pinned to. This is the BASE commit the
#: manifest-runner branch was created from (repair/datagen-worker-completeness).
#: It is deliberately not the runner commit: the manifest must be frozen before
#: any simulation, and a manifest whose hash moved every time the runner was
#: amended would not be a freeze. The exact frozen source is proven separately by
#: the tree hashes recorded in the runtime specification.
MOLMOSPACES_BASE_COMMIT = "fa8e61f40eb97e27bf3b69480c8bc65b0450f362"

#: Environment/config parameters pinned for the manifest config. Every value here
#: is fixed BEFORE simulation. Two of them (the z-offset scalars) are the ones the
#: legacy configs drew from an unseeded global RNG at module import time.
PINNED_ENV_CONFIG = {
    "config_class": "FrankaSkinHybridObstacleManifestV2Config",
    "task_sampler_class": "ObstacleFumehoodPickSampler",
    "policy_config": "ObstacleAwarePickPlannerPolicyConfig",
    "robot_config": "FrankaSkinHybridRobotConfig",
    "robot_base_size": [0.4, 0.4, 0.35],
    "robot_xml_path": "model_hybrid.xml",
    "camera_config": "FrankaSkinHybridCameraSystem",
    "scene_xml": "fumehood.xml",
    "house_inds": [SCENE_TEMPLATE_HOUSE_INDEX],
    "samples_per_house": 1,
    "added_pickup_objects": None,
    "num_added_pickups": 0,
    "check_robot_placement_visibility": False,
    "max_total_attempts_multiplier": 10,
    "robot_object_z_offset_random_min": -0.5,
    "robot_object_z_offset_random_max": 0.5,
    "robot_placement_rotation_range_rad": 0.52,
    "randomize_textures": True,
    "randomize_lighting": False,
    "randomize_dynamics": False,
    "filter_for_successful_trajectories": False,
    "obstacle_p_design_only": DESIGN_OBSTACLE_P,
    "obstacle_bar_face_y": [0.14, 0.24],
    "obstacle_bar_x_frac": [0.20, 0.55],
    "obstacle_obj_gap": [0.12, 0.20],
    "light_scale_range": [0.75, 1.10],
}

#: Pinned runtime. Determinism is claimed for this runtime only.
RUNTIME_CONTRACT = {
    "mujoco": "3.5.0",
    "warp-lang": "1.11.1",
    "python_min": "3.11",
    "numpy_bit_generator": "PCG64 via numpy.random.SeedSequence",
    "seed_derivation": (
        "SeedSequence([master_seed, candidate_index, stream_id, retry_index])"
    ),
    "identity_hash": "SHA-256",
    "forbidden": [
        "python builtin hash() for persistent IDs or seeds",
        "worker ID as random entropy",
        "wraparound house index as random entropy",
    ],
}


def git_commit(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def build() -> tuple[dict, dict, dict]:
    safety_stack = json.loads(SAFETY_STACK_PATH.read_text())
    sensor_contract = safety_stack["sensor_contract"]
    pinned = safety_stack["pinned_hashes"]

    safety_cvae_contract = {
        "stack_schema_version": safety_stack["schema_version"],
        "sensor_order_sha256": sensor_contract["sensor_order_hash"],
        "sensor_input_shape": sensor_contract["input_shape"],
        "safety_model_sha256": pinned["safety_model_sha256"],
        "safety_meta_sha256": pinned["safety_meta_sha256"],
        "safety_sweep_sha256": pinned["safety_sweep_sha256"],
        "note": (
            "Reference only. This task does not train, modify or re-checkpoint the "
            "Safety-CVAE; the hashes pin what a future consumer must match."
        ),
    }

    robot_model_sha256 = sha256_file(ROBOT_MODEL_PATH)
    if robot_model_sha256 != pinned["model_hybrid_xml_sha256"]:
        raise SystemExit(
            f"model_hybrid.xml hash {robot_model_sha256} does not match the pinned "
            f"{pinned['model_hybrid_xml_sha256']}; the robot model changed"
        )

    env_config_sha256 = sha256_payload(PINNED_ENV_CONFIG)
    runtime_contract_sha256 = sha256_payload(RUNTIME_CONTRACT)

    document = build_manifest(
        sensor_order_sha256=sensor_contract["sensor_order_hash"],
        robot_model_sha256=robot_model_sha256,
        env_config_sha256=env_config_sha256,
        safety_cvae_contract=safety_cvae_contract,
        molmospaces_source_commit=MOLMOSPACES_BASE_COMMIT,
        runtime_contract_sha256=runtime_contract_sha256,
        scene_sha256=sha256_file(SCENE_PATH),
    )
    validate_manifest(document)

    subset = build_smoke_subset(document, per_stratum=4)
    validate_smoke_subset(subset, document)

    contract = {
        "schema": "hybrid_obstacle_independent_v2_contract",
        "manifest_version": MANIFEST_VERSION,
        "master_seed": MASTER_SEED,
        "master_seed_policy": (
            "Fixed before any simulation result was observed. Never changed afterwards."
        ),
        "total_candidates": TOTAL_CANDIDATES,
        "hazard_present_count": HAZARD_PRESENT_COUNT,
        "hazard_absent_count": HAZARD_ABSENT_COUNT,
        "design_obstacle_p": DESIGN_OBSTACLE_P,
        "hazard_assignment": {
            "method": (
                "Exactly 120 True and 40 False values are created and deterministically "
                "permuted from the fixed master seed, then committed before simulation."
            ),
            "runtime_bernoulli": "bypassed for this config; preserved exactly for every legacy config",
            "schedule_stream_id": STREAM_HAZARD_SCHEDULE,
        },
        "scene_template_id": SCENE_TEMPLATE_ID,
        "scene_template_house_index": SCENE_TEMPLATE_HOUSE_INDEX,
        "scene_sha256": document["scene_sha256"],
        "robot_model_sha256": robot_model_sha256,
        "sensor_order_sha256": sensor_contract["sensor_order_hash"],
        "env_config_sha256": env_config_sha256,
        "runtime_contract_sha256": runtime_contract_sha256,
        "molmospaces_source_commit": MOLMOSPACES_BASE_COMMIT,
        "root_commit_at_build": git_commit(ROOT),
        "pinned_env_config": PINNED_ENV_CONFIG,
        "runtime_contract": RUNTIME_CONTRACT,
        "safety_cvae_contract": safety_cvae_contract,
        "stream_ids": dict(STREAM_NAMES),
        "max_retries_per_row": MAX_RETRIES_PER_ROW,
        "identity": {
            "episode_id": (
                "SHA256(manifest_version || master_seed || candidate_index || scene_template_id)"
            ),
            "row_identity": "episode_id + manifest_row_sha256",
            "forbidden_identity_inputs": [
                "worker-local episode counter",
                "batch index alone",
                "house directory",
                "wraparound house alias",
                "file ordering",
            ],
        },
        "canonical_selection": {
            "hazard_present": CANONICAL_HAZARD_PRESENT,
            "hazard_absent": CANONICAL_HAZARD_ABSENT,
            "rule": (
                "first N successful rows of each stratum by predeclared stratum rank; "
                "never inspects rollout quality"
            ),
        },
        "split": {
            "train": {
                "hazard_present": TRAIN_HAZARD_PRESENT,
                "hazard_absent": TRAIN_HAZARD_ABSENT,
            },
            "val": {"hazard_present": VAL_HAZARD_PRESENT, "hazard_absent": VAL_HAZARD_ABSENT},
        },
        "shortfall_policy": (
            "If a future 160-row collection yields too few successes: do not duplicate rows, "
            "do not lower the quota, do not alter this manifest. A separately versioned "
            "deterministic extension manifest requires its own explicitly approved task."
        ),
        "manifest_sha256": document["manifest_sha256"],
        "smoke8_sha256": subset["subset_sha256"],
    }
    return document, subset, contract


def to_yaml(payload, indent: int = 0) -> str:
    """Minimal YAML emitter (no PyYAML dependency in this environment)."""
    pad = "  " * indent
    lines = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{pad}{key}:")
                lines.append(to_yaml(value, indent + 1))
            else:
                lines.append(f"{pad}{key}: {_scalar(value)}")
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}-")
                lines.append(to_yaml(item, indent + 1))
            else:
                lines.append(f"{pad}- {_scalar(item)}")
    else:
        lines.append(f"{pad}{_scalar(payload)}")
    return "\n".join(line for line in lines if line)


def _scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if any(ch in text for ch in ':#\n"') or text.strip() != text or text == "":
        return json.dumps(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed files match a fresh regeneration",
    )
    args = parser.parse_args()

    document, subset, contract = build()

    if args.check:
        problems = []
        for path, fresh, key in (
            (MANIFEST_PATH, document, "manifest_sha256"),
            (SMOKE_PATH, subset, "subset_sha256"),
        ):
            if not path.exists():
                problems.append(f"{path.name} is missing")
                continue
            committed = json.loads(path.read_text())
            if committed.get(key) != fresh[key]:
                problems.append(
                    f"{path.name}: committed {committed.get(key)} != regenerated {fresh[key]}"
                )
        if problems:
            print("MANIFEST CHECK FAILED")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print(f"manifest OK       {document['manifest_sha256']}")
        print(f"smoke8 OK         {subset['subset_sha256']}")
        return 0

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    SMOKE_PATH.write_text(json.dumps(subset, indent=2, sort_keys=True) + "\n")
    CONTRACT_PATH.write_text(
        "# hybrid_obstacle_independent_v2 -- frozen episode-manifest contract.\n"
        "# Generated by scripts/build_hybrid_obstacle_manifest_v2.py. Do not hand-edit.\n"
        + to_yaml(contract)
        + "\n"
    )

    present = sum(1 for row in document["rows"] if row["hazard_present"])
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"wrote {SMOKE_PATH.relative_to(ROOT)}")
    print(f"wrote {CONTRACT_PATH.relative_to(ROOT)}")
    print(f"  manifest_version   {MANIFEST_VERSION}")
    print(f"  master_seed        {MASTER_SEED}")
    print(f"  rows               {len(document['rows'])} ({present} hazard / {len(document['rows']) - present} clear)")
    print(f"  manifest_sha256    {document['manifest_sha256']}")
    print(f"  smoke8_sha256      {subset['subset_sha256']}")
    print(f"  smoke8 candidates  {subset['candidate_indices']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
