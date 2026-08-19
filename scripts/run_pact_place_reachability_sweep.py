#!/usr/bin/env python3
"""A0: static IK reachability sweep for a relocated place tray. No rollouts."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for search_path in (ROOT / "scripts", MOLMO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import sha256_payload  # noqa: E402
from run_pact_place_expert_screen import _make_config  # noqa: E402

TRAVERSAL_Y_ABS_M = 0.107
CLEARANCE_MARGIN_M = 0.08
RELEASE_CLEARANCE_M = 0.005
CANDIDATE_X = (0.30, 0.35, 0.40)
CANDIDATE_Y = (0.20, 0.24, 0.28, 0.32)
CURRENT_CENTER = (0.35, 0.0)
V4_ROW0 = ROOT / (
    "diagnostics_output/pact_place_corridor_v4/expert_screen_rows/"
    "00_132234fef92e64f4"
)

# y half-extents of every receptacle geom. The limiting footprint is the max.
FOOTPRINTS = {
    "current_0.10x0.16": {
        "floor_half_xy_m": [0.10, 0.16],
        "lip_half_xy_m": [0.07, 0.015],
        "lip_pos_y_m": 0.145,
        "pedestal_base_half_xy_m": [0.10, 0.15],
        "pedestal_stem_half_xy_m": [0.07, 0.11],
    },
    "shrunk_0.10x0.10": {
        "floor_half_xy_m": [0.10, 0.10],
        "lip_half_xy_m": [0.07, 0.015],
        "lip_pos_y_m": 0.085,
        "pedestal_base_half_xy_m": [0.10, 0.10],
        "pedestal_stem_half_xy_m": [0.07, 0.07],
    },
}


def _pose(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def footprint_half_y(spec: dict[str, Any]) -> float:
    return max(
        float(spec["floor_half_xy_m"][1]),
        float(spec["lip_pos_y_m"]) + float(spec["lip_half_xy_m"][1]),
        float(spec["pedestal_base_half_xy_m"][1]),
        float(spec["pedestal_stem_half_xy_m"][1]),
    )


def clearance_m(centre_y: float, half_y: float) -> float:
    inner_edge = abs(float(centre_y)) - float(half_y)
    return inner_edge - TRAVERSAL_Y_ABS_M


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources, setup_policy
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime
    from molmo_spaces.utils.grasp_sample import compute_grasp_pose
    from molmo_spaces.utils.mj_model_and_data_utils import body_aabb

    assert_supported_runtime(strict=True)
    result = json.loads((V4_ROW0 / "result.json").read_text())
    trajectory = json.loads((V4_ROW0 / "trajectory.json").read_text())
    carry_step = next(
        step
        for step in trajectory["steps"]
        if step.get("policy_phase") == "outbound_approach"
        and step.get("tcp_position_m") is not None
    )
    row = json.loads((ROOT / "configs/pact_place_corridor_v4.json").read_text())[
        "expert_screen_rows"
    ][0]
    scratch = Path(tempfile.mkdtemp(prefix="pact_place_a0_"))
    task = policy = sampler = None
    try:
        config = _make_config(scratch / "dummy.json")
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("sample_task returned None")
        policy = setup_policy(config, task, None, None)
        task.reset()
        robot_view = task.env.current_robot.robot_view
        reset_qpos = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in robot_view.get_qpos_dict().items()
        }
        qpos = np.asarray(carry_step["qpos"], dtype=np.float64)
        if qpos.shape != (task.env.current_data.qpos.size,):
            raise RuntimeError("carry qpos length does not match the model")
        task.env.current_data.qpos[:] = qpos
        mujoco.mj_forward(task.env.current_model, task.env.current_data)
        carry_qpos = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in robot_view.get_qpos_dict().items()
        }
        robot_view.set_qpos_dict(reset_qpos)
        mujoco.mj_forward(task.env.current_model, task.env.current_data)

        pickup = task.env.object_managers[task.env.current_batch_index].get_object_by_name(
            task.config.task_config.pickup_obj_name
        )
        receptacle = task.env.object_managers[task.env.current_batch_index].get_object_by_name(
            task.config.task_config.place_receptacle_name
        )
        pc = policy.policy_config
        grasp_pose_world = compute_grasp_pose(
            policy,
            pickup,
            robot_view,
            check_collision=False,
            n_collision_checks=pc.grasp_collision_max_grasps,
            collision_batch_size=pc.grasp_collision_batch_size,
            check_ik=False,
            n_ik_checks=pc.grasp_feasibility_max_grasps,
            ik_batch_size=pc.grasp_feasibility_batch_size,
        )
        aabb_c, aabb_s = body_aabb(
            task.env.current_data.model, task.env.current_data, receptacle.object_id
        )
        receptacle_top_z = float(aabb_c[2] + aabb_s[2] / 2)
        pickup_c, pickup_s = body_aabb(
            task.env.current_data.model, task.env.current_data, pickup.object_id
        )
        pickup_bottom_z = float(pickup_c[2] - pickup_s[2] / 2)
        clearance_offset = max(float(grasp_pose_world[2, 3] - pickup_bottom_z), 0.0)
        grasp_minus_object = grasp_pose_world[:3, 3] - np.asarray(pickup.position, dtype=float)
        place_z_offset = float(policy.policy_config.place_z_offset)

        def placement_poses(center_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
            preplace = grasp_pose_world.copy()
            preplace[0, 3] = float(center_xy[0])
            preplace[1, 3] = float(center_xy[1])
            preplace[2, 3] = (
                receptacle_top_z + clearance_offset + place_z_offset
            )
            preplace[:3, 3] += grasp_minus_object
            place = preplace.copy()
            place[2, 3] = receptacle_top_z + clearance_offset + RELEASE_CLEARANCE_M
            return preplace, place

        def ik_ok(pose: np.ndarray, seed: dict[str, np.ndarray]) -> bool:
            robot_view.set_qpos_dict(seed)
            mujoco.mj_forward(task.env.current_model, task.env.current_data)
            return bool(policy.check_feasible_ik(pose))

        candidates = []
        for footprint_name, spec in FOOTPRINTS.items():
            half_y = footprint_half_y(spec)
            centres = [CURRENT_CENTER] + [
                (float(x), float(y)) for x in CANDIDATE_X for y in CANDIDATE_Y
            ]
            for center in centres:
                preplace, place = placement_poses(center)
                gap = clearance_m(center[1], half_y)
                record = {
                    "footprint": footprint_name,
                    "center_xy_m": [float(center[0]), float(center[1])],
                    "footprint_half_y_m": half_y,
                    "inner_edge_abs_y_m": abs(center[1]) - half_y,
                    "clearance_beyond_traversal_y_m": gap,
                    "clears_8cm": bool(gap + 1e-12 >= CLEARANCE_MARGIN_M),
                    "preplace_position_m": [float(v) for v in preplace[:3, 3]],
                    "place_position_m": [float(v) for v in place[:3, 3]],
                    "ik_preplace_from_reset": ik_ok(preplace, reset_qpos),
                    "ik_place_from_reset": ik_ok(place, reset_qpos),
                    "ik_preplace_from_carry": ik_ok(preplace, carry_qpos),
                    "ik_place_from_carry": ik_ok(place, carry_qpos),
                }
                record["reachable"] = bool(
                    record["ik_preplace_from_reset"]
                    and record["ik_place_from_reset"]
                    and record["ik_preplace_from_carry"]
                    and record["ik_place_from_carry"]
                )
                record["eligible"] = bool(record["reachable"] and record["clears_8cm"])
                candidates.append(record)

        eligible = [item for item in candidates if item["eligible"]]
        chosen = None
        if eligible:
            eligible.sort(
                key=lambda item: (
                    abs(item["center_xy_m"][1]),
                    abs(item["center_xy_m"][0] - CURRENT_CENTER[0]),
                    item["footprint_half_y_m"],
                )
            )
            chosen = eligible[0]

        document = {
            "schema_version": "pact_place_reachability_sweep_v1",
            "no_rollouts": True,
            "source_row": {
                "config_sha256": result["config_sha256"],
                "role_index": 0,
                "episode_id": result["episode_id"],
                "carry_step": carry_step["step"],
                "carry_phase": carry_step["policy_phase"],
            },
            "grasp_world_position_m": [float(v) for v in grasp_pose_world[:3, 3]],
            "pickup_position_m": [float(v) for v in pickup.position],
            "grasp_minus_object_m": [float(v) for v in grasp_minus_object],
            "receptacle_top_z_m": receptacle_top_z,
            "pickup_clearance_offset_m": clearance_offset,
            "place_z_offset_m": place_z_offset,
            "traversal_y_abs_m": TRAVERSAL_Y_ABS_M,
            "required_clearance_m": CLEARANCE_MARGIN_M,
            "footprints": FOOTPRINTS,
            "candidates": candidates,
            "n_candidates": len(candidates),
            "n_reachable": sum(item["reachable"] for item in candidates),
            "n_clears_8cm": sum(item["clears_8cm"] for item in candidates),
            "n_eligible": len(eligible),
            "chosen": chosen,
        }
        if chosen is None:
            document["decision"] = "no_candidate_reachable_and_clear_stop"
            document["fallback"] = "lower_the_pedestal_needs_its_own_sweep"
        else:
            document["decision"] = "relocate_to_chosen_centre"
        document["sweep_sha256"] = sha256_payload(document)
        out_dir = ROOT / "diagnostics_output" / "pact_place_reachability_sweep"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "analysis.json"
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        print(path)
        print(json.dumps({
            "n_candidates": document["n_candidates"],
            "n_reachable": document["n_reachable"],
            "n_clears_8cm": document["n_clears_8cm"],
            "n_eligible": document["n_eligible"],
            "decision": document["decision"],
            "chosen": chosen,
        }, indent=2, sort_keys=True))
        return 0 if chosen is not None else 2
    finally:
        cleanup_episode_resources(
            task=task,
            policy=policy,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=sampler is not None,
        )


if __name__ == "__main__":
    raise SystemExit(main())
