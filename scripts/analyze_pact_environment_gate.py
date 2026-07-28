#!/usr/bin/env python3
"""Adjudicate the frozen collision-route environment gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "submodules" / "act"
SCRIPTS = ROOT / "scripts"
for path in (ACT, SCRIPTS):
    sys.path.insert(0, str(path))

from pact_collision_contract import load_manifest, rows_for_role  # noqa: E402
from scripts.convert_obstacle_to_act import _decode_action  # noqa: E402
from surface_proximity_encoder import native_camera_intrinsic  # noqa: E402


def _trajectory(handle: h5py.File):
    keys = [key for key in handle if key.startswith("traj_")]
    if len(keys) != 1:
        raise ValueError(f"expected one trajectory group, got {keys}")
    return handle[keys[0]]


def _first_close_step(group) -> int:
    for step, value in enumerate(group["actions/joint_pos"]):
        action, valid = _decode_action(value)
        if valid and float(action[7]) >= 127.5:
            return step
    return len(group["actions/joint_pos"])


def intrusion_surface_activity(
    trajectory_path: Path,
    *,
    sensor_names: list[str],
    panel_center: np.ndarray,
    panel_half: np.ndarray,
    first_target_contact_step: int | None,
    tolerance_m: float = 0.01,
) -> dict[str, Any]:
    intrinsic = native_camera_intrinsic()
    u, v = np.meshgrid(np.arange(8, dtype=np.float64), np.arange(8, dtype=np.float64))
    x_factor = (u + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
    y_factor = (v + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
    active20 = active12 = 0
    with h5py.File(trajectory_path, "r") as handle:
        group = _trajectory(handle)
        end = _first_close_step(group)
        if first_target_contact_step is not None:
            end = min(end, int(first_target_contact_step))
        timesteps = min(
            end,
            min(len(group[f"obs/proximity/{name}"]) for name in sensor_names),
        )
        for timestep in range(timesteps):
            seen20 = seen12 = False
            for name in sensor_names:
                depths = np.asarray(
                    group[f"obs/proximity/{name}"][timestep], dtype=np.float64
                )
                extrinsic = np.asarray(
                    group[f"obs/sensor_param/{name}/extrinsic_cv"][timestep],
                    dtype=np.float64,
                )
                rotation = extrinsic[:, :3]
                translation = extrinsic[:, 3]
                for depth in depths:
                    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= 0.20)
                    if not np.any(valid):
                        continue
                    local = np.stack(
                        (x_factor * depth, y_factor * depth, depth), axis=-1
                    )
                    world = (local.reshape(-1, 3) - translation) @ rotation
                    inside = np.all(
                        np.abs(world - panel_center)
                        <= panel_half + float(tolerance_m),
                        axis=1,
                    ).reshape(8, 8)
                    hit20 = inside & valid
                    if np.any(hit20):
                        seen20 = True
                        if np.any(hit20 & (depth <= 0.12)):
                            seen12 = True
                    if seen12:
                        break
                if seen12:
                    break
            active20 += int(seen20)
            active12 += int(seen12)
    return {
        "pregrasp_control_steps": timesteps,
        "steps_intrusion_inside_20cm": active20,
        "steps_intrusion_inside_12cm": active12,
        "episode_has_intrusion_sighting": bool(active20),
    }


def _expert_results(collection: Path, manifest: dict):
    results = []
    for row in rows_for_role(manifest, "pilot_train"):
        path = collection / "rows" / row["episode_id"] / "result.json"
        if not path.exists():
            raise RuntimeError(f"unreconciled expert row: {path}")
        result = json.loads(path.read_text())
        if result.get("row_sha256") != row["row_sha256"]:
            raise RuntimeError(f"expert row identity mismatch: {path}")
        status = result.get("status")
        if status not in (
            "success",
            "task_failure",
            "sampling_failure",
            "infrastructure_failure",
        ):
            raise RuntimeError(f"expert row has unknown terminal status: {path}")
        if status in ("sampling_failure", "infrastructure_failure"):
            # These are predeclared terminal row outcomes, not missing data and
            # not eligible for replacement. Count them as failed expert rows
            # with no observable pre-grasp trajectory.
            result["task_success"] = False
            result["collision_free_task_success"] = False
            result["surface_activity"] = {
                "pregrasp_control_steps": 0,
                "steps_intrusion_inside_20cm": 0,
                "steps_intrusion_inside_12cm": 0,
                "episode_has_intrusion_sighting": False,
            }
            results.append(result)
            continue
        audit = result["contact_audit"]
        center = np.asarray(result["scene_params"]["protr_center"], dtype=float)
        half = np.asarray(result["scene_params"]["protr_half"], dtype=float)
        target_step = audit["first_contact_step"].get("grasp_target")
        activity = intrusion_surface_activity(
            Path(result["trajectory_path"]),
            sensor_names=manifest["sensor_names"],
            panel_center=center,
            panel_half=half,
            first_target_contact_step=target_step,
        )
        result["surface_activity"] = activity
        results.append(result)
    return results


def _pilot_act_results(schedule_path: Path, output_root: Path):
    schedule = json.loads(schedule_path.read_text())
    results = []
    for row in schedule["rows"]:
        result_path = output_root / row["output_relpath"] / "result.json"
        driver_path = output_root / row["output_relpath"] / "driver_result.json"
        if not result_path.exists() or not driver_path.exists():
            raise RuntimeError(f"pilot ACT row is unreconciled: {row['rollout_id']}")
        driver = json.loads(driver_path.read_text())
        result = json.loads(result_path.read_text())
        if driver.get("status") != "complete":
            raise RuntimeError(f"pilot ACT driver failed: {row['rollout_id']}")
        if (
            result.get("rollout_id") != row["rollout_id"]
            or result.get("schedule_row_sha256") != row["schedule_row_sha256"]
        ):
            raise RuntimeError(f"pilot ACT identity mismatch: {row['rollout_id']}")
        results.append(result)
    return results, schedule


def analyze(
    *,
    manifest: dict,
    expert_results: list[dict[str, Any]],
    act_results: list[dict[str, Any]],
    pilot_schedule: dict,
) -> dict[str, Any]:
    expert_task = sum(bool(result["task_success"]) for result in expert_results)
    expert_primary = sum(
        bool(result["collision_free_task_success"]) for result in expert_results
    )
    total_pregrasp = sum(
        result["surface_activity"]["pregrasp_control_steps"]
        for result in expert_results
    )
    inside20 = sum(
        result["surface_activity"]["steps_intrusion_inside_20cm"]
        for result in expert_results
    )
    inside12 = sum(
        result["surface_activity"]["steps_intrusion_inside_12cm"]
        for result in expert_results
    )
    active_episodes = sum(
        result["surface_activity"]["episode_has_intrusion_sighting"]
        for result in expert_results
    )
    act_task = sum(bool(result["task_success"]) for result in act_results)
    act_primary = sum(
        bool(result["collision_free_task_success"]) for result in act_results
    )
    act_any_non_target = sum(
        (
            int(
                result["contact_audit"]["contact_class_totals"].get(
                    "hazard_bar", 0
                )
            )
            + int(
                result["contact_audit"]["contact_class_totals"].get(
                    "other_environment", 0
                )
            )
        )
        > 0
        for result in act_results
    )
    act_hazard = sum(
        int(
            result["contact_audit"]["contact_class_totals"].get("hazard_bar", 0)
        )
        > 0
        for result in act_results
    )
    checks = {
        "expert_task_success_at_least_20_of_24": expert_task >= 20,
        "expert_collision_free_task_success_at_least_20_of_24": expert_primary >= 20,
        "surface_active_episodes_at_least_20_of_24": active_episodes >= 20,
        "surface_pregrasp_inside_20cm_at_least_30_percent": (
            total_pregrasp > 0 and inside20 / total_pregrasp >= 0.30
        ),
        "surface_pregrasp_inside_12cm_at_least_5_percent": (
            total_pregrasp > 0 and inside12 / total_pregrasp >= 0.05
        ),
        "act_primary_between_8_and_16_of_24": 8 <= act_primary <= 16,
        "act_task_success_at_least_12_of_24": act_task >= 12,
        "act_any_non_target_contact_at_least_6_of_24": act_any_non_target >= 6,
        "act_hazard_contact_at_least_6_of_24": act_hazard >= 6,
    }
    adequate = all(checks.values())
    return {
        "schema_version": "pact_environment_gate_v1",
        "route": "collision",
        "gate_a_applicable": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "pilot_schedule_sha256": pilot_schedule["schedule_sha256"],
        "expert": {
            "n": len(expert_results),
            "ordinary_task_success": expert_task,
            "collision_free_task_success": expert_primary,
            "pregrasp_control_steps": total_pregrasp,
            "steps_intrusion_inside_20cm": inside20,
            "fraction_pregrasp_inside_20cm": (
                inside20 / total_pregrasp if total_pregrasp else None
            ),
            "steps_intrusion_inside_12cm": inside12,
            "fraction_pregrasp_inside_12cm": (
                inside12 / total_pregrasp if total_pregrasp else None
            ),
            "episodes_with_intrusion_sighting": active_episodes,
        },
        "act": {
            "n": len(act_results),
            "ordinary_task_success": act_task,
            "collision_free_task_success": act_primary,
            "episodes_with_any_non_target_contact": act_any_non_target,
            "episodes_with_hazard_bar_contact": act_hazard,
            "failure_taxonomy": dict(
                __import__("collections").Counter(
                    result["failure_taxonomy"] for result in act_results
                )
            ),
        },
        "checks": checks,
        "all_applicable_gates_pass": adequate,
        "decision": (
            "PACT_ENVIRONMENT_ADEQUATE"
            if adequate
            else "PACT_ENVIRONMENT_INADEQUATE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expert-collection", required=True, type=Path)
    parser.add_argument("--pilot-schedule", required=True, type=Path)
    parser.add_argument("--pilot-output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    experts = _expert_results(args.expert_collection, manifest)
    acts, schedule = _pilot_act_results(
        args.pilot_schedule, args.pilot_output_root
    )
    if len(experts) != 24 or len(acts) != 24:
        raise SystemExit("Phase 1 requires exactly 24 reconciled rows in each set")
    result = analyze(
        manifest=manifest,
        expert_results=experts,
        act_results=acts,
        pilot_schedule=schedule,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["decision"])
    return 0 if result["all_applicable_gates_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
