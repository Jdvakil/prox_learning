#!/usr/bin/env python3
"""Produce deterministic machine-readable audits for the hybrid safety stack.

This script only loads existing configuration, model, and validation artifacts. It does
not train, mutate checkpoints, construct rollout environments, or execute policies.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MOLMO_ROOT = REPO_ROOT / "submodules/molmospaces"
for path in (REPO_ROOT / "scripts", MOLMO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (  # noqa: E402
    FrankaSkinHybridObstacleConfig,
)
from molmo_spaces.env.sensors import get_core_sensors  # noqa: E402
from molmo_spaces.env.sensors_cameras import ProximityDepthBufferSensor  # noqa: E402
from train_safety_cvae import SafetyHead, featurize  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "diagnostics_output/hybrid_safety_stack"
CONTRACT_PATH = REPO_ROOT / "configs/hybrid_safety_stack_v1.json"
MODEL_PATH = REPO_ROOT / "assets/robots/franka_skin/model_hybrid.xml"
CAMERA_CONFIG_PATH = (
    MOLMO_ROOT / "molmo_spaces/configs/camera_configs.py"
)
SAFETY_DIR = REPO_ROOT / "assets/safety/cvae_v3"
SWEEP_PATH = REPO_ROOT / "assets/safety/sweep_v3.h5"
EXPECTED_DATASET = (
    REPO_ROOT
    / "assets/datagen/hybrid_obstacle_v1/"
    "FrankaSkinHybridObstacleConfig/20260612_183855"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.check_output(
        ["git", "-C", str(cwd), *args], text=True
    ).strip()


def file_record(relative_path: str) -> dict[str, Any]:
    path = REPO_ROOT / relative_path
    return {
        "path": relative_path,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def artifact_inventory() -> dict[str, Any]:
    required_files = [
        "assets/robots/franka_skin/model_hybrid.xml",
        "assets/safety/sweep_v3.h5",
        "scripts/train_safety_cvae.py",
        "scripts/safety_sweep.py",
        "scripts/safety_flinch_demo.py",
        "scripts/safety_react_demo.py",
        "scripts/safety_moving_demo.py",
        "scripts/safety_orbit_demo.py",
        "scripts/safety_sphere_demo.py",
        "scripts/verify_hybrid_skin_sensors.py",
        "scripts/build_hybrid_on_franka_skin.py",
        "scripts/convert_obstacle_to_act.py",
        "submodules/act/eval_act_obstacle.py",
        "submodules/act/imitate_episodes.py",
    ]
    safety_files = [
        file_record(str(path.relative_to(REPO_ROOT)))
        for path in sorted(SAFETY_DIR.iterdir())
        if path.is_file()
    ]
    checkpoint_candidates = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in REPO_ROOT.rglob("policy_best.ckpt")
        if ".git" not in path.parts
    )
    stats_candidates = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in REPO_ROOT.rglob("dataset_stats.pkl")
        if ".git" not in path.parts
    )
    return {
        "schema_version": "hybrid_safety_artifact_inventory_v1",
        "required_files": [file_record(path) for path in required_files],
        "canonical_safety_directory": {
            "path": str(SAFETY_DIR.relative_to(REPO_ROOT)),
            "exists": SAFETY_DIR.is_dir(),
            "files": safety_files,
        },
        "canonical_act_obstacle_checkpoint": {
            "expected_name": "policy_best.ckpt",
            "candidates": checkpoint_candidates,
            "status": "present" if checkpoint_candidates else "missing",
        },
        "canonical_act_dataset_stats": {
            "expected_name": "dataset_stats.pkl",
            "candidates": stats_candidates,
            "status": "present" if stats_candidates else "missing",
        },
        "hybrid_obstacle_v1_source_dataset": {
            "expected_path": str(EXPECTED_DATASET.relative_to(REPO_ROOT)),
            "exists": EXPECTED_DATASET.is_dir(),
            "status": "present" if EXPECTED_DATASET.is_dir() else "missing",
        },
        "all_required_code_and_safety_files_exist": all(
            (REPO_ROOT / path).is_file() for path in required_files
        )
        and SAFETY_DIR.is_dir(),
    }


def xml_sensor_records() -> list[dict[str, Any]]:
    root = ET.parse(MODEL_PATH).getroot()
    records: list[dict[str, Any]] = []

    def visit(element: ET.Element, parent_body: str | None = None) -> None:
        body = element.attrib.get("name", parent_body) if element.tag == "body" else parent_body
        for child in element:
            if child.tag == "camera" and "_sensor_" in child.attrib.get("name", ""):
                resolution = [int(value) for value in child.attrib["resolution"].split()]
                records.append(
                    {
                        "name": child.attrib["name"],
                        "link_assignment": body,
                        "resolution": resolution,
                    }
                )
            visit(child, body)

    visit(root)
    return records


def sensor_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    canonical = contract["sensor_contract"]["ordered_names"]
    meta = json.loads((SAFETY_DIR / "meta.json").read_text())
    with h5py.File(SWEEP_PATH, "r") as h5:
        sweep_names = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in h5["sensors"][:]
        ]

    config = FrankaSkinHybridObstacleConfig()
    config_proximity = [
        camera.name
        for camera in config.camera_config.cameras
        if getattr(camera, "is_proximity_sensor", False)
    ]
    emitted = [
        sensor
        for sensor in get_core_sensors(config)
        if isinstance(sensor, ProximityDepthBufferSensor)
    ]
    emitted_names = [sensor.uuid for sensor in emitted]
    emitted_shapes = [list(sensor.observation_space.shape) for sensor in emitted]
    xml_records = xml_sensor_records()
    xml_by_name = {record["name"]: record for record in xml_records}
    ordered_records = [
        {
            "index": index,
            "name": name,
            "link_assignment": xml_by_name[name]["link_assignment"],
            "patch_shape": xml_by_name[name]["resolution"],
        }
        for index, name in enumerate(canonical)
    ]
    order_hash = json_hash(canonical)
    checks = {
        "canonical_count_is_40": len(canonical) == 40,
        "canonical_names_are_unique": len(set(canonical)) == 40,
        "canonical_order_hash_matches_contract": (
            order_hash == contract["sensor_contract"]["sensor_order_hash"]
        ),
        "training_meta_order_matches": meta["sensors"] == canonical,
        "training_sweep_order_matches": sweep_names == canonical,
        "camera_config_sensor_set_matches": set(config_proximity) == set(canonical),
        "emitted_observation_key_set_matches": set(emitted_names) == set(canonical),
        "emitted_shapes_are_4x8x8": all(shape == [4, 8, 8] for shape in emitted_shapes),
        "model_sensor_set_matches": set(xml_by_name) == set(canonical),
        "model_resolutions_are_8x8": all(
            record["resolution"] == [8, 8] for record in xml_records
        ),
        "hybrid_robot_uses_model_hybrid_xml": (
            str(config.robot_config.robot_xml_path) == "model_hybrid.xml"
        ),
        "obstacle_config_uses_hybrid_robot": (
            type(config.robot_config).__name__ == "FrankaSkinHybridRobotConfig"
        ),
        "obstacle_config_uses_hybrid_camera_system": (
            type(config.camera_config).__name__ == "FrankaSkinHybridCameraSystem"
        ),
    }
    return {
        "schema_version": "hybrid_safety_sensor_order_manifest_v1",
        "ordered_sensors": ordered_records,
        "sensor_order_hash": order_hash,
        "model_hash": sha256_file(MODEL_PATH),
        "camera_config_hash": sha256_file(CAMERA_CONFIG_PATH),
        "ordering_rule": (
            "Lexicographically sorted sensor names, exactly matching sweep_v3.h5 "
            "and cvae_v3/meta.json; raw XML and config declaration order are not "
            "used as inference order."
        ),
        "camera_config_declaration_order": config_proximity,
        "model_xml_declaration_order": [record["name"] for record in xml_records],
        "emitted_observation_key_order": emitted_names,
        "emitted_observation_shapes": emitted_shapes,
        "timing": {
            "policy_dt_ms": float(config.policy_dt_ms),
            "control_dt_ms": float(config.ctrl_dt_ms),
            "simulation_dt_ms": float(config.sim_dt_ms),
            "proximity_sensor_period_ms": float(config.proximity_sensor_period_ms),
            "proximity_substeps_per_policy_observation": emitted_shapes[0][0],
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def safety_cvae_audit(contract: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(2)
    head = SafetyHead.load(SAFETY_DIR)
    meta = json.loads((SAFETY_DIR / "meta.json").read_text())
    history = json.loads((SAFETY_DIR / "history.json").read_text())
    config = json.loads((SAFETY_DIR / "config.json").read_text())

    far = np.full((40, 8, 8), 10.0, dtype=np.float32)
    close = far.copy()
    close[18] = 0.05
    far_first = head(far)
    far_second = head(far)
    close_output = head(close)
    boundary_depths = np.full((1, 1, 8, 8), 1.0, dtype=np.float32)
    boundary_depths[0, 0, 0, :6] = [-1.0, 0.0, 0.004999, 0.005, 0.5, 0.500001]
    boundary_features = featurize(boundary_depths).reshape(1, 1, 8, 8)

    with h5py.File(SWEEP_PATH, "r") as h5:
        proximity = h5["prox"][:]
        labels = h5["label_dq"][:].astype(np.float32)
        minimum_depth = h5["min_depth"][:].astype(np.float32)
        sweep_names = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in h5["sensors"][:]
        ]
        sweep_shapes = {
            "prox": list(h5["prox"].shape),
            "label_dq": list(h5["label_dq"].shape),
            "min_depth": list(h5["min_depth"].shape),
            "sensors": list(h5["sensors"].shape),
        }
    minimum_per_sample = np.min(
        np.where(np.isfinite(minimum_depth), minimum_depth, np.inf), axis=1
    )
    close_mask = minimum_per_sample < 0.12
    far_mask = minimum_per_sample > 0.25
    recomputed_scale = float(
        np.sqrt(np.mean(np.linalg.norm(labels[close_mask], axis=1) ** 2))
    )

    np.random.seed(int(config["seed"]))
    indices = np.random.permutation(len(proximity))
    validation_size = max(256, len(proximity) // 10)
    validation_indices = indices[:validation_size]
    validation_features = torch.from_numpy(
        featurize(proximity[validation_indices])
    )
    validation_labels = torch.from_numpy(
        labels[validation_indices] / recomputed_scale
    )
    with torch.no_grad():
        prediction = head.model.act(validation_features)
    recomputed_mse = float(torch.mean((prediction - validation_labels) ** 2))
    validation_close = torch.from_numpy(close_mask[validation_indices])
    validation_far = torch.from_numpy(far_mask[validation_indices])
    recomputed_cosine = float(
        torch.nn.functional.cosine_similarity(
            prediction[validation_close],
            validation_labels[validation_close],
            dim=-1,
        ).mean()
    )
    recomputed_far_quiet = float(
        prediction[validation_far].norm(dim=-1).mean()
    )

    checks = {
        "loaded_only_via_safety_head_load": True,
        "input_shape_is_40x8x8": meta["n_in"] == 40 * 8 * 8,
        "output_shape_is_7": far_first.shape == (7,) and meta["n_out"] == 7,
        "z_zero_inference_is_exactly_deterministic": np.array_equal(
            far_first, far_second
        ),
        "metadata_sensor_order_matches_contract": (
            meta["sensors"] == contract["sensor_contract"]["ordered_names"]
        ),
        "sweep_sensor_order_matches_contract": (
            sweep_names == contract["sensor_contract"]["ordered_names"]
        ),
        "label_scale_recomputes_exactly": (
            abs(recomputed_scale - float(meta["label_scale"])) < 1e-7
        ),
        "best_validation_mse_recomputes": (
            abs(recomputed_mse - float(meta["best_val_mse"])) < 1e-7
        ),
        "close_cosine_recomputes": (
            abs(recomputed_cosine - float(meta["close_cos"])) < 1e-6
        ),
        "far_quiet_recomputes": (
            abs(recomputed_far_quiet - float(meta["far_quiet"])) < 1e-7
        ),
        "far_input_is_near_zero_in_normalized_units": (
            float(np.linalg.norm(far_first) / head.scale) < 0.05
        ),
        "close_obstacle_changes_retreat": (
            float(np.linalg.norm(close_output - far_first)) > 1.0
        ),
        "dead_and_below_range_pixels_map_to_zero": np.array_equal(
            boundary_features[0, 0, 0, :3], np.zeros(3, dtype=np.float32)
        ),
        "depth_at_activation_max_maps_to_zero": (
            float(boundary_features[0, 0, 0, 4]) == 0.0
        ),
        "depth_above_activation_max_maps_to_zero": (
            float(boundary_features[0, 0, 0, 5]) == 0.0
        ),
    }
    return {
        "schema_version": "canonical_safety_cvae_audit_v1",
        "load_api": 'SafetyHead.load("assets/safety/cvae_v3")',
        "checkpoint_hashes": {
            "model_pt": sha256_file(SAFETY_DIR / "model.pt"),
            "meta_json": sha256_file(SAFETY_DIR / "meta.json"),
            "config_json": sha256_file(SAFETY_DIR / "config.json"),
            "history_json": sha256_file(SAFETY_DIR / "history.json"),
        },
        "sweep_hash": sha256_file(SWEEP_PATH),
        "training_shapes": sweep_shapes,
        "training_counts": {
            "samples": int(len(proximity)),
            "close_samples": int(close_mask.sum()),
            "far_samples": int(far_mask.sum()),
            "validation_samples": int(validation_size),
            "validation_close_samples": int(close_mask[validation_indices].sum()),
            "validation_far_samples": int(far_mask[validation_indices].sum()),
        },
        "preprocessing": {
            "formula": "closeness = clip(1 - depth / 0.5, 0, 1)",
            "dead_pixel_rule": "depth < 0.005 m becomes zero",
            "boundary_feature_values": [
                float(value) for value in boundary_features[0, 0, 0, :6]
            ],
        },
        "outputs": {
            "far_physical": [float(value) for value in far_first],
            "far_physical_norm": float(np.linalg.norm(far_first)),
            "far_normalized_norm": float(np.linalg.norm(far_first) / head.scale),
            "close_sensor_index": 18,
            "close_depth_m": 0.05,
            "close_physical": [float(value) for value in close_output],
            "close_physical_norm": float(np.linalg.norm(close_output)),
            "close_minus_far_norm": float(np.linalg.norm(close_output - far_first)),
        },
        "validation_metrics": {
            "label_scale_metadata": float(meta["label_scale"]),
            "label_scale_recomputed": recomputed_scale,
            "best_val_mse_metadata": float(meta["best_val_mse"]),
            "best_val_mse_recomputed": recomputed_mse,
            "close_cos_metadata": float(meta["close_cos"]),
            "close_cos_recomputed": recomputed_cosine,
            "far_quiet_metadata": float(meta["far_quiet"]),
            "far_quiet_recomputed": recomputed_far_quiet,
            "history_final_val_mse": float(history["val_mse"][-1]),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def verifier_summary() -> dict[str, Any]:
    path = OUTPUT_DIR / "hybrid_sensor_verify/sensor_verify.csv"
    if not path.is_file():
        return {"status": "not_run", "path": str(path.relative_to(REPO_ROOT))}
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    failures = [row["sensor"] for row in rows if row["VERDICT"] != "PASS"]
    return {
        "status": "completed",
        "path": str(path.relative_to(REPO_ROOT)),
        "sensor_count": len(rows),
        "passed_count": len(rows) - len(failures),
        "failed_count": len(failures),
        "failed_sensors": failures,
        "result": f"PASS {len(rows) - len(failures)}/{len(rows)}",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT_PATH.read_text())
    inventory = artifact_inventory()
    sensors = sensor_manifest(contract)
    safety = safety_cvae_audit(contract)
    stack = {
        "schema_version": "hybrid_safety_stack_audit_v1",
        "repository": {
            "root_branch": git("branch", "--show-current"),
            "root_commit": git("rev-parse", "HEAD"),
            "act_branch": git(
                "branch", "--show-current", cwd=REPO_ROOT / "submodules/act"
            ),
            "act_commit": git("rev-parse", "HEAD", cwd=REPO_ROOT / "submodules/act"),
            "molmospaces_branch": git(
                "branch", "--show-current", cwd=MOLMO_ROOT
            ),
            "molmospaces_commit": git("rev-parse", "HEAD", cwd=MOLMO_ROOT),
        },
        "artifact_inventory_passed": inventory[
            "all_required_code_and_safety_files_exist"
        ],
        "missing_runtime_artifacts": [
            name
            for name, missing in {
                "canonical_act_obstacle_checkpoint": not inventory[
                    "canonical_act_obstacle_checkpoint"
                ]["candidates"],
                "canonical_act_dataset_stats": not inventory[
                    "canonical_act_dataset_stats"
                ]["candidates"],
                "hybrid_obstacle_v1_source_dataset": not inventory[
                    "hybrid_obstacle_v1_source_dataset"
                ]["exists"],
            }.items()
            if missing
        ],
        "sensor_contract_passed": sensors["passed"],
        "safety_cvae_passed": safety["passed"],
        "hybrid_sensor_geometry_verifier": verifier_summary(),
        "integration_status": {
            "safety_react_demo": "recorded_or_synthetic_nominal_trajectory_not_live_act",
            "eval_act_obstacle": "live_vanilla_act_without_safety_residual",
            "live_adapter_at_audit_start": "missing",
            "live_adapter_implemented": (
                REPO_ROOT / "submodules/act/eval_act_obstacle_safety.py"
            ).is_file(),
            "paired_launcher_implemented": (
                REPO_ROOT / "submodules/act/run_paired_hybrid_safety_eval.py"
            ).is_file(),
        },
    }
    outputs = {
        "artifact_inventory.json": inventory,
        "sensor_order_manifest.json": sensors,
        "safety_cvae_audit.json": safety,
        "stack_audit.json": stack,
    }
    for name, payload in outputs.items():
        (OUTPUT_DIR / name).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {OUTPUT_DIR / name}")
    if not sensors["passed"] or not safety["passed"]:
        raise SystemExit("Hybrid contract audit failed")


if __name__ == "__main__":
    main()
