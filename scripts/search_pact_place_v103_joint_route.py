#!/usr/bin/env python3
"""V10.3 Step-0B cached offline joint-route search.

Compiles nothing at episode runtime, never calls ``env.step``, never renders an
observation, and never generates an episode. It may call model compilation,
``mj_forward``, FK, IK, and exact distance.

Height is scored analytically against every candidate node and edge, so the four
registered geometries share one pass of IK and one pass of edge interpolation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_geom_distance import GeomShape, gjk_distance, mesh_vertices  # noqa: E402
from pact_place_v10_exact import verify_v99_inputs  # noqa: E402
from pact_place_v10_runtime import establish_v10_runtime_env, write_immutable  # noqa: E402
from pact_place_v103_contract import (  # noqa: E402
    CONTRACT_VERSION,
    SEARCH_ROOT_RELATIVE,
    empty_authorization,
    implementation_sha256,
    search_lattice,
    verify_protected_artifacts,
)
from pact_place_v103_geometry import enumerate_v103_assemblies  # noqa: E402
from pact_place_v103_joint_route import (  # noqa: E402
    ARM_GROUP,
    CORNER_MIN_CLEARANCE_M,
    DISALLOWED_CONTACT_CLASSES,
    EDGE_MIN_CLEARANCE_M,
    FR3_JOINT_VELOCITY_LIMITS_RAD_S,
    MAX_JOINT_STEP_RAD,
    MAX_ORIENTATION_RESIDUAL_DEG,
    MAX_POSITION_RESIDUAL_M,
    NODE_MIN_CLEARANCE_M,
    N_ARM_JOINTS,
    build_control_poses,
    control_pose_key,
    dedup_joint_solutions,
    enumerate_templates,
    halton_joint_seeds,
    qpos_sequence_sha256,
    route_geometry_report,
    segment_commanded_speed,
    segment_duration_s,
    segment_speed_class,
    segment_x_monotonic,
    select_path,
    template_corners,
)
from pact_place_v99_exact import snapshot_jobs_from_reconstruction  # noqa: E402
from pact_place_v99_geometry import window_masks  # noqa: E402
from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v9_contract import sha256_payload  # noqa: E402
from reconstruct_pact_place_v99_baseline import cleanup_task  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / SEARCH_ROOT_RELATIVE
HEIGHT_KEYS = ("0.920", "0.960", "1.000", "1.040")
SEED_CLASS_CODES = {
    "actual_directional_start": 0,
    "actual_directional_end": 1,
    "joint_center": 2,
    "halton": 3,
    "preceding_layer": 4,
    "following_layer": 5,
}
EDGE_FAILURE_CODES = {
    None: 0,
    "joint_limit": 1,
    "self_collision": 2,
    "environment_contact": 3,
    "pendant_clearance": 4,
}
SCREEN_MARGIN_M = 0.15
SCHEMA_VERSION = "pact_place_v103_ik_search_v1"


def _rotation_residual_deg(actual: np.ndarray, target: np.ndarray) -> float:
    delta = np.asarray(actual, dtype=float).T @ np.asarray(target, dtype=float)
    trace = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


def _pose_matrix(position: Sequence[float], rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(rotation, dtype=float)
    pose[:3, 3] = np.asarray(position, dtype=float)
    return pose


def _quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    xyzw = R.from_matrix(np.asarray(matrix, dtype=float)).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)


class CellContext:
    """One live frozen cell, used for IK, FK, collision, and exact distance."""

    def __init__(self, job: dict[str, Any], cell: dict[str, Any], direction: str):
        import mujoco
        from pact_place_v10_environment import prepare_v10_parked_task
        from run_pact_place_v7_replay_videos import apply_recorded_qpos

        self.mujoco = mujoco
        self.direction = str(direction)
        self.cell = cell
        self.role_index = int(cell["role_index"])
        self.family = str(cell["family"])
        self.intrusion_side = str(cell["intrusion_side"])
        self.cell_key = f"{self.family}:{self.intrusion_side}"
        self.steps = json.loads(
            (Path(job["row_dir"]) / "trajectory.json").read_text()
        )["steps"]
        self.task, self.sampler, self.scratch = prepare_v10_parked_task(
            job["manifest_row"],
            seed_u32=(job.get("selected_seed") or {}).get("seed_u32"),
        )
        self.env = self.task.env
        self.model = self.env.current_model
        self.data = self.env.current_data
        self.robot_view = self.env.current_robot.robot_view
        self.kinematics = self.env.current_robot.kinematics
        self.gripper_group = self.robot_view.get_gripper_movegroup_ids()[0]
        self.arm = self.robot_view.get_move_group(ARM_GROUP)
        limits = np.asarray(self.arm.joint_pos_limits, dtype=float)
        # The live move group exposes (n_joints, 2); accept (2, n_joints) too.
        if limits.shape == (N_ARM_JOINTS, 2):
            self.joint_low = limits[:, 0].copy()
            self.joint_high = limits[:, 1].copy()
        elif limits.shape == (2, N_ARM_JOINTS):
            self.joint_low = limits[0].copy()
            self.joint_high = limits[1].copy()
        else:
            raise RuntimeError(f"unexpected joint_pos_limits shape {limits.shape}")
        vel = getattr(self.arm, "vel_limits", None)
        if vel is None:
            self.velocity_limits = np.asarray(
                FR3_JOINT_VELOCITY_LIMITS_RAD_S, dtype=float
            )
            self.velocity_limit_source = "fr3_datasheet_constants"
        else:
            self.velocity_limits = np.asarray(vel, dtype=float).reshape(N_ARM_JOINTS)
            self.velocity_limit_source = "live_move_group"
        apply_recorded_qpos(self.env, self.steps[0]["qpos"])
        mujoco.mj_forward(self.model, self.data)
        self.base_pose = np.asarray(self.robot_view.base.pose, dtype=float).copy()
        self.base_qpos = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in self.robot_view.get_qpos_dict().items()
        }
        self._collect_geoms()
        self._collect_target()
        self.halton = halton_joint_seeds(self.joint_low, self.joint_high)
        self.joint_center = 0.5 * (self.joint_low + self.joint_high)
        self.ik_cache: dict[tuple, list[np.ndarray]] = {}
        self.counters = {
            "ik_calls": 0,
            "ik_success": 0,
            "ik_cache_hits": 0,
            "forward_calls": 0,
            "gjk_calls": 0,
        }

    # -- setup -----------------------------------------------------------
    def _collect_geoms(self) -> None:
        model = self.model
        self.robot_geom_ids: list[int] = []
        for geom_id in range(int(model.ngeom)):
            body = str(model.body(int(model.geom_bodyid[geom_id])).name or "")
            if not body.startswith("robot_0/"):
                continue
            if int(model.geom_contype[geom_id]) == 0 and int(
                model.geom_conaffinity[geom_id]
            ) == 0:
                continue
            self.robot_geom_ids.append(int(geom_id))
        self.geom_cache: dict[int, tuple[int, np.ndarray, np.ndarray | None]] = {}
        for geom_id in self.robot_geom_ids:
            gtype = int(model.geom_type[geom_id])
            self.geom_cache[geom_id] = (
                gtype,
                np.asarray(model.geom_size[geom_id], dtype=float).copy(),
                mesh_vertices(model, geom_id),
            )

    def _collect_target(self) -> None:
        from reconstruct_pact_place_v99_baseline import pickup_collision_geom_ids

        self.target_geom_ids = [int(v) for v in pickup_collision_geom_ids(self.task)]
        for geom_id in self.target_geom_ids:
            self.geom_cache[geom_id] = (
                int(self.model.geom_type[geom_id]),
                np.asarray(self.model.geom_size[geom_id], dtype=float).copy(),
                mesh_vertices(self.model, geom_id),
            )
        manager = self.env.object_managers[self.env.current_batch_index]
        pickup = manager.get_object_by_name(
            self.task.config.task_config.pickup_obj_name
        )
        self.target_body = str(getattr(pickup, "name", "") or "")
        body_id = int(self.model.body(self.target_body).id)
        joint_adr = int(self.model.body_jntadr[body_id])
        self.target_qpos_adr = (
            int(self.model.jnt_qposadr[joint_adr]) if joint_adr >= 0 else -1
        )
        self.carry_transform: np.ndarray | None = None

    # -- state -----------------------------------------------------------
    def gripper_pose(self) -> np.ndarray:
        return np.asarray(
            self.robot_view.get_gripper(self.gripper_group).leaf_frame_to_world,
            dtype=float,
        ).copy()

    def apply_recorded_step(self, index: int) -> None:
        from run_pact_place_v7_replay_videos import apply_recorded_qpos

        apply_recorded_qpos(self.env, self.steps[int(index)]["qpos"])
        self.mujoco.mj_forward(self.model, self.data)
        self.counters["forward_calls"] += 1

    def arm_qpos(self) -> np.ndarray:
        return np.asarray(
            self.robot_view.get_qpos_dict()[ARM_GROUP], dtype=float
        ).reshape(N_ARM_JOINTS).copy()

    def freeze_carry(self) -> None:
        """Record the carried target's rigid offset from the gripper."""
        if self.target_qpos_adr < 0:
            self.carry_transform = None
            return
        body_id = int(self.model.body(self.target_body).id)
        target = np.eye(4)
        target[:3, :3] = np.asarray(self.data.xmat[body_id], dtype=float).reshape(3, 3)
        target[:3, 3] = np.asarray(self.data.xpos[body_id], dtype=float)
        self.carry_transform = np.linalg.inv(self.gripper_pose()) @ target

    def set_arm(self, joints: Sequence[float], *, carry: bool) -> None:
        qpos = {key: value.copy() for key, value in self.base_qpos.items()}
        qpos[ARM_GROUP] = np.asarray(joints, dtype=float).reshape(N_ARM_JOINTS)
        self.robot_view.set_qpos_dict(qpos)
        self.mujoco.mj_forward(self.model, self.data)
        self.counters["forward_calls"] += 1
        if carry and self.carry_transform is not None and self.target_qpos_adr >= 0:
            pose = self.gripper_pose() @ self.carry_transform
            adr = self.target_qpos_adr
            self.data.qpos[adr : adr + 3] = pose[:3, 3]
            self.data.qpos[adr + 3 : adr + 7] = _quat_from_matrix(pose[:3, :3])
            self.mujoco.mj_forward(self.model, self.data)
            self.counters["forward_calls"] += 1

    # -- predicates -------------------------------------------------------
    def joint_limit_margin(self, joints: Sequence[float]) -> float:
        value = np.asarray(joints, dtype=float).reshape(N_ARM_JOINTS)
        finite = np.isfinite(self.joint_low) & np.isfinite(self.joint_high)
        if not np.any(finite):
            return float("inf")
        margins = np.minimum(value - self.joint_low, self.joint_high - value)
        return float(np.min(margins[finite]))

    def self_collision(self) -> bool:
        model, data = self.model, self.data
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            if float(contact.dist) > 0.0:
                continue
            body1 = str(model.body(int(model.geom_bodyid[int(contact.geom1)])).name or "")
            body2 = str(model.body(int(model.geom_bodyid[int(contact.geom2)])).name or "")
            if body1.startswith("robot_0/") and body2.startswith("robot_0/"):
                return True
        return False

    def environment_contact_classes(self) -> set[str]:
        from molmo_spaces.tasks.pact_place_contact_audit import (
            classify_contact,
            place_environment_contact_pairs,
        )

        return {
            classify_contact(pair)
            for pair in place_environment_contact_pairs(self.env)
        }

    def environment_clear(self) -> tuple[bool, list[str]]:
        classes = self.environment_contact_classes()
        offending = sorted(classes & DISALLOWED_CONTACT_CLASSES)
        return (not offending), offending

    def _shape(self, geom_id: int) -> GeomShape:
        gtype, size, verts = self.geom_cache[geom_id]
        return GeomShape(
            gtype,
            self.data.geom_xpos[geom_id],
            self.data.geom_xmat[geom_id],
            size,
            verts,
        )

    def pendant_clearances(
        self, boxes_by_height: dict[str, list[dict[str, Any]]], *, include_target: bool
    ) -> dict[str, dict[str, Any]]:
        """Exact min distance from robot (+target) to each pendant component."""
        ids = list(self.robot_geom_ids)
        if include_target:
            ids = ids + list(self.target_geom_ids)
        centers = np.asarray([self.data.geom_xpos[g] for g in ids], dtype=float)
        rbound = np.asarray([self.model.geom_rbound[g] for g in ids], dtype=float)
        out: dict[str, dict[str, Any]] = {}
        for height_key, boxes in boxes_by_height.items():
            per_component: dict[str, float] = {}
            exact_flags: dict[str, bool] = {}
            for box in boxes:
                center = box["center"]
                half = box["half"]
                delta = np.maximum(np.abs(centers - center) - half, 0.0)
                lower = np.linalg.norm(delta, axis=1) - rbound
                near = np.flatnonzero(lower <= SCREEN_MARGIN_M)
                if near.size == 0:
                    per_component[box["name"]] = float(np.min(lower))
                    exact_flags[box["name"]] = False
                    continue
                best = float("inf")
                for local in near.tolist():
                    shape = self._shape(int(ids[local]))
                    if not shape.supported:
                        best = min(best, float(lower[local]))
                        continue
                    self.counters["gjk_calls"] += 1
                    best = min(best, float(gjk_distance(shape, box["shape"])))
                far = np.delete(lower, near)
                if far.size:
                    best = min(best, float(np.min(far)))
                per_component[box["name"]] = best
                exact_flags[box["name"]] = True
            finite = [v for v in per_component.values() if np.isfinite(v)]
            out[height_key] = {
                "per_component_m": per_component,
                "min_m": float(min(finite)) if finite else None,
                "exact": exact_flags,
            }
        return out

    def nearest_geom_witness(
        self, boxes: Sequence[dict[str, Any]], *, include_target: bool
    ) -> dict[str, Any]:
        """Name the closest robot/target geom to each pendant component."""
        ids = list(self.robot_geom_ids)
        if include_target:
            ids = ids + list(self.target_geom_ids)
        out: dict[str, Any] = {}
        for box in boxes:
            best = (float("inf"), None)
            for geom_id in ids:
                shape = self._shape(int(geom_id))
                if not shape.supported:
                    continue
                self.counters["gjk_calls"] += 1
                distance = float(gjk_distance(shape, box["shape"]))
                if distance < best[0]:
                    name = self.model.geom(int(geom_id)).name or ""
                    body = str(
                        self.model.body(int(self.model.geom_bodyid[int(geom_id)])).name
                        or ""
                    )
                    best = (distance, name or body)
            out[box["name"]] = {
                "distance_m": None if not np.isfinite(best[0]) else float(best[0]),
                "nearest_geom": best[1],
            }
        return out

    # -- IK ---------------------------------------------------------------
    def solve_ik(self, pose: np.ndarray, seed_arm: Sequence[float]):
        seed = {key: value.copy() for key, value in self.base_qpos.items()}
        seed[ARM_GROUP] = np.asarray(seed_arm, dtype=float).reshape(N_ARM_JOINTS)
        self.counters["ik_calls"] += 1
        solution = self.kinematics.ik(
            self.gripper_group,
            np.asarray(pose, dtype=float),
            self.robot_view.move_group_ids(),
            seed,
            base_pose=self.base_pose,
        )
        if solution is None:
            return None
        self.counters["ik_success"] += 1
        return np.asarray(solution[ARM_GROUP], dtype=float).reshape(N_ARM_JOINTS).copy()

    def close(self) -> None:
        cleanup_task(self.task, self.sampler, self.scratch)


def pendant_boxes(assemblies: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for assembly in assemblies:
        key = f"{float(assembly['lowest_lobe_bottom_z_m']):.3f}"
        boxes = []
        for item in assembly["components"]:
            center = np.asarray(item["center_m"], dtype=float)
            half = np.asarray(item["half_m"], dtype=float)
            boxes.append(
                {
                    "name": str(item["name"]),
                    "center": center,
                    "half": half,
                    "shape": GeomShape.posed_axis_aligned_box(center, half),
                }
            )
        out[key] = boxes
    return out


def stock_reference(context: CellContext) -> dict[str, Any]:
    """Retained TCP path and pinned endpoint states for one direction."""
    cell = context.cell
    phases = [str(step.get("policy_phase") or "") for step in context.steps]
    windows = window_masks(phases)
    tcp = np.asarray(cell["tcp_m"], dtype=float)
    mat = np.asarray(cell["tcp_mat"], dtype=float).reshape(len(tcp), 3, 3)
    if context.direction == "inbound":
        mask = np.asarray(cell["inbound_mask"], dtype=bool)
        end_index = int(windows["pregrasp_index"])
        start_index = 0
        indices = np.concatenate(
            [np.flatnonzero(mask), np.asarray([end_index], dtype=int)]
        )
    else:
        lift = [i for i, phase in enumerate(phases) if phase == "lift"]
        if not lift:
            raise RuntimeError("no lift phase in the retained trajectory")
        start_index = int(lift[-1])
        mask = np.asarray(cell["outbound_mask"], dtype=bool)
        outbound = np.flatnonzero(mask)
        end_index = int(outbound[-1])
        indices = np.concatenate([np.asarray([start_index], dtype=int), outbound])
    indices = np.unique(indices)
    order = np.argsort(indices)
    indices = indices[order]
    return {
        "positions_m": tcp[indices].copy(),
        "rotations": mat[indices].copy(),
        "source_step_indices": indices.tolist(),
        "start_step_index": int(start_index),
        "end_step_index": int(end_index),
        "start_position_m": tcp[start_index].copy(),
        "start_rotation": mat[start_index].copy(),
        "end_position_m": tcp[end_index].copy(),
        "end_rotation": mat[end_index].copy(),
        "target_held": bool(context.direction == "outbound"),
    }


def endpoint_state(context: CellContext, stock: dict[str, Any], which: str) -> dict[str, Any]:
    index = int(stock["start_step_index"] if which == "start" else stock["end_step_index"])
    context.apply_recorded_step(index)
    if which == "start" and context.direction == "outbound":
        context.freeze_carry()
    joints = context.arm_qpos()
    pose = context.gripper_pose()
    expected = _pose_matrix(
        stock["start_position_m"] if which == "start" else stock["end_position_m"],
        stock["start_rotation"] if which == "start" else stock["end_rotation"],
    )
    return {
        "which": which,
        "step_index": index,
        "arm_qpos": joints,
        "qpos_sha256": qpos_sequence_sha256([joints]),
        "tcp_position_residual_m": float(
            np.linalg.norm(pose[:3, 3] - expected[:3, 3])
        ),
        "tcp_orientation_residual_deg": _rotation_residual_deg(
            pose[:3, :3], expected[:3, :3]
        ),
    }


def evaluate_node(
    context: CellContext,
    joints: np.ndarray,
    target_pose: np.ndarray,
    boxes: dict[str, list[dict[str, Any]]],
    *,
    carry: bool,
) -> dict[str, Any]:
    context.set_arm(joints, carry=carry)
    pose = context.gripper_pose()
    position_residual = float(np.linalg.norm(pose[:3, 3] - target_pose[:3, 3]))
    orientation_residual = _rotation_residual_deg(pose[:3, :3], target_pose[:3, :3])
    margin = context.joint_limit_margin(joints)
    self_hit = context.self_collision()
    env_ok, offending = context.environment_clear()
    clearances = context.pendant_clearances(boxes, include_target=carry)
    base_ok = bool(
        position_residual <= MAX_POSITION_RESIDUAL_M
        and orientation_residual <= MAX_ORIENTATION_RESIDUAL_DEG
        and margin >= 0.0
        and not self_hit
        and env_ok
    )
    per_height = {}
    for key, report in clearances.items():
        minimum = report["min_m"]
        per_height[key] = {
            "min_clearance_m": minimum,
            "per_component_m": report["per_component_m"],
            "clear": bool(minimum is not None and minimum >= NODE_MIN_CLEARANCE_M),
        }
    return {
        "arm_qpos": np.asarray(joints, dtype=float).copy(),
        "tcp_position_m": pose[:3, 3].copy(),
        "position_residual_m": position_residual,
        "orientation_residual_deg": orientation_residual,
        "joint_limit_margin_rad": margin,
        "self_collision": bool(self_hit),
        "environment_clear": bool(env_ok),
        "environment_contact_classes": offending,
        "base_valid": base_ok,
        "per_height": per_height,
    }


def _rejection(node: dict[str, Any]) -> str:
    if node["position_residual_m"] > MAX_POSITION_RESIDUAL_M:
        return "position_residual"
    if node["orientation_residual_deg"] > MAX_ORIENTATION_RESIDUAL_DEG:
        return "orientation_residual"
    if node["joint_limit_margin_rad"] < 0.0:
        return "joint_limit"
    if node["self_collision"]:
        return "self_collision"
    if not node["environment_clear"]:
        return "environment_contact"
    return "pendant_clearance"


def evaluate_edge(
    context: CellContext,
    start: np.ndarray,
    end: np.ndarray,
    boxes: dict[str, list[dict[str, Any]]],
    *,
    carry: bool,
    height_keys: Sequence[str],
) -> dict[str, Any]:
    """Joint-space interpolation with exact distance at every sample.

    Every return carries the same keys: a failing edge is still a fully
    described record, because the search artifact reports rejected edges as
    well as accepted ones.
    """
    a = np.asarray(start, dtype=float).reshape(N_ARM_JOINTS)
    b = np.asarray(end, dtype=float).reshape(N_ARM_JOINTS)
    travel = np.abs(b - a)
    steps = (
        int(np.ceil(float(np.max(travel)) / MAX_JOINT_STEP_RAD))
        if float(travel.max()) > 0.0
        else 1
    )
    steps = max(1, steps)
    alive = {key: True for key in height_keys}
    min_clearance = {key: float("inf") for key in height_keys}
    min_margin = float("inf")
    positions: list[np.ndarray] = []
    failure: dict[str, Any] | None = None

    def record() -> dict[str, Any]:
        path = (
            np.asarray(positions, dtype=float)
            if positions
            else np.zeros((0, 3), dtype=float)
        )
        arc = (
            float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
            if len(path) > 1
            else 0.0
        )
        passed = {
            key: bool(alive[key] and failure is None) for key in height_keys
        }
        return {
            "passed_by_height": passed,
            "min_clearance_by_height_m": {
                key: (
                    None
                    if not np.isfinite(min_clearance[key])
                    else float(min_clearance[key])
                )
                for key in height_keys
            },
            "min_joint_limit_margin_rad": (
                0.0 if not np.isfinite(min_margin) else float(min_margin)
            ),
            "joint_travel_rad": float(np.sum(np.abs(b - a))),
            "tcp_arc_length_m": arc,
            "tcp_path_m": path,
            "x_monotonic": bool(segment_x_monotonic(path)) if len(path) > 1 else True,
            "n_samples": int(len(positions)),
            "failure": failure,
        }

    for index in range(steps + 1):
        t = index / steps
        joints = a + t * (b - a)
        context.set_arm(joints, carry=carry)
        pose = context.gripper_pose()
        positions.append(pose[:3, 3].copy())
        min_margin = min(min_margin, context.joint_limit_margin(joints))
        if min_margin < 0.0:
            failure = {"code": "joint_limit", "sample": index}
            return record()
        if context.self_collision():
            failure = {"code": "self_collision", "sample": index}
            return record()
        env_ok, offending = context.environment_clear()
        if not env_ok:
            failure = {
                "code": "environment_contact",
                "sample": index,
                "classes": offending,
            }
            return record()
        live = [key for key in height_keys if alive[key]]
        if not live:
            failure = {"code": "pendant_clearance", "sample": index}
            return record()
        clearances = context.pendant_clearances(
            {key: boxes[key] for key in live}, include_target=carry
        )
        for key in live:
            value = clearances[key]["min_m"]
            if value is None:
                alive[key] = False
                min_clearance[key] = float("-inf")
                continue
            min_clearance[key] = min(min_clearance[key], float(value))
            if float(value) < EDGE_MIN_CLEARANCE_M:
                alive[key] = False
    return record()


# ---------------------------------------------------------------------------
# Per-case search
# ---------------------------------------------------------------------------
def _seed_key(seed: np.ndarray) -> tuple:
    return tuple(np.round(np.asarray(seed, dtype=float), 6).tolist())


def _cached_ik(
    context: CellContext, pose_key: tuple, pose: np.ndarray, seed: np.ndarray
):
    """IK cache keyed by (control pose, seed). Height never re-runs IK."""
    key = (pose_key, _seed_key(seed))
    if key in context.ik_cache:
        context.counters["ik_cache_hits"] += 1
        return context.ik_cache[key]
    solution = context.solve_ik(pose, seed)
    context.ik_cache[key] = solution
    return solution


def _seed_class(label: str) -> int:
    if str(label).startswith("halton"):
        return SEED_CLASS_CODES["halton"]
    return SEED_CLASS_CODES.get(str(label), 6)


def build_layer(
    context: CellContext,
    pose: dict[str, Any],
    boxes: dict[str, list[dict[str, Any]]],
    seeds: Sequence[tuple[str, np.ndarray]],
    *,
    carry: bool,
    rejections: dict[str, int],
    recorder: list[list[float]] | None = None,
    template_index: int = -1,
    layer_index: int = -1,
) -> list[dict[str, Any]]:
    key = control_pose_key(pose)
    matrix = _pose_matrix(pose["position_m"], pose["rotation"])
    solutions: list[np.ndarray] = []
    provenance: list[str] = []
    for seed_index, (label, seed) in enumerate(seeds):
        solution = _cached_ik(context, key, matrix, seed)
        if solution is None:
            rejections["ik_no_solution"] = rejections.get("ik_no_solution", 0) + 1
            if recorder is not None:
                recorder.append(
                    [
                        float(template_index),
                        float(layer_index),
                        float(seed_index),
                        float(_seed_class(label)),
                        0.0,
                    ]
                    + [float("nan")] * 7
                    + [float("nan")] * 4
                    + [float("nan")] * 4
                    + [0.0]
                )
            continue
        solutions.append(solution)
        provenance.append(label)
    if not solutions:
        return []
    kept = dedup_joint_solutions(solutions)
    nodes: list[dict[str, Any]] = []
    for index in kept:
        node = evaluate_node(context, solutions[index], matrix, boxes, carry=carry)
        node["control_pose"] = str(pose["name"])
        node["seed_provenance"] = provenance[index]
        node["node_key"] = qpos_sequence_sha256([solutions[index]])[:16]
        if recorder is not None:
            clearances = [
                node["per_height"].get(hk, {}).get("min_clearance_m")
                for hk in HEIGHT_KEYS
            ]
            recorder.append(
                [
                    float(template_index),
                    float(layer_index),
                    -1.0,
                    float(_seed_class(provenance[index])),
                    1.0,
                ]
                + [float(v) for v in solutions[index]]
                + [
                    float(node["position_residual_m"]),
                    float(node["orientation_residual_deg"]),
                    float(node["joint_limit_margin_rad"]),
                    float(bool(node["self_collision"])),
                ]
                + [float("nan") if v is None else float(v) for v in clearances]
                + [float(bool(node["base_valid"]))]
            )
        if not node["base_valid"]:
            code = _rejection(node)
            rejections[code] = rejections.get(code, 0) + 1
            continue
        nodes.append(node)
    return nodes


def fixed_seeds(context: CellContext, start_q, end_q) -> list[tuple[str, np.ndarray]]:
    """The registered seed set: three fixed states plus 24 Halton seeds."""
    seeds = [
        ("actual_directional_start", np.asarray(start_q, dtype=float)),
        ("actual_directional_end", np.asarray(end_q, dtype=float)),
        ("joint_center", context.joint_center.copy()),
    ]
    seeds += [
        (f"halton_{index:02d}", context.halton[index].copy())
        for index in range(context.halton.shape[0])
    ]
    return seeds


def evaluate_template(
    context: CellContext,
    stock: dict[str, Any],
    template: dict[str, Any],
    boxes: dict[str, list[dict[str, Any]]],
    height_keys: Sequence[str],
    endpoints: dict[str, Any],
    edge_cache: dict[tuple, dict[str, Any]],
    *,
    corner_thresholds: bool = False,
    node_recorder: list[list[float]] | None = None,
    edge_recorder: list[list[float]] | None = None,
    template_index: int = -1,
) -> dict[str, Any]:
    carry = bool(stock["target_held"])
    poses = build_control_poses(stock, template, direction=context.direction)
    start_q = endpoints["start"]["arm_qpos"]
    end_q = endpoints["end"]["arm_qpos"]
    rejections: dict[str, int] = {}
    base = fixed_seeds(context, start_q, end_q)

    layers: list[list[dict[str, Any]]] = [[endpoints["start_node"]]]
    for pose in poses:
        seeds = list(base)
        for node in layers[-1]:
            seeds.append(("preceding_layer", node["arm_qpos"]))
        layers.append(
            build_layer(
                context,
                pose,
                boxes,
                seeds,
                carry=carry,
                rejections=rejections,
                recorder=node_recorder,
                template_index=template_index,
                layer_index=len(layers),
            )
        )
        if not layers[-1]:
            return {
                "template_key": template["template_key"],
                "feasible_by_height": {key: False for key in height_keys},
                "stop": "empty_layer",
                "empty_layer": pose["name"],
                "rejections": rejections,
            }
    layers.append([endpoints["end_node"]])
    # Reverse pass: each intermediate layer is also seeded from the layer after it.
    for index in range(len(poses), 0, -1):
        seeds = list(base)
        for node in layers[index + 1]:
            seeds.append(("following_layer", node["arm_qpos"]))
        extra = build_layer(
            context,
            poses[index - 1],
            boxes,
            seeds,
            carry=carry,
            rejections=rejections,
            recorder=node_recorder,
            template_index=template_index,
            layer_index=index,
        )
        merged = layers[index] + extra
        kept = dedup_joint_solutions([item["arm_qpos"] for item in merged])
        layers[index] = [merged[i] for i in kept]

    edges: dict[tuple[int, int, int], dict[str, Any]] = {}
    for layer_index in range(len(layers) - 1):
        for src, node_a in enumerate(layers[layer_index]):
            for dst, node_b in enumerate(layers[layer_index + 1]):
                key = (node_a["node_key"], node_b["node_key"])
                if key not in edge_cache:
                    edge_cache[key] = evaluate_edge(
                        context,
                        node_a["arm_qpos"],
                        node_b["arm_qpos"],
                        boxes,
                        carry=carry,
                        height_keys=height_keys,
                    )
                edge = edge_cache[key]
                edges[(layer_index, src, dst)] = edge
                if edge_recorder is not None:
                    failure_code = EDGE_FAILURE_CODES.get(
                        (edge.get("failure") or {}).get("code"), 5
                    )
                    clearances = [
                        edge.get("min_clearance_by_height_m", {}).get(hk)
                        for hk in HEIGHT_KEYS
                    ]
                    edge_recorder.append(
                        [
                            float(template_index),
                            float(layer_index),
                            float(src),
                            float(dst),
                            float(edge["n_samples"]),
                            float(edge["joint_travel_rad"]),
                            float(edge["tcp_arc_length_m"]),
                            float(bool(edge["x_monotonic"])),
                            float(edge["min_joint_limit_margin_rad"]),
                        ]
                        + [float("nan") if v is None else float(v) for v in clearances]
                        + [float(failure_code)]
                    )
                failure = edge.get("failure")
                if failure and not any(edge["passed_by_height"].values()):
                    code = f"edge_{failure['code']}"
                    rejections[code] = rejections.get(code, 0) + 1

    threshold = CORNER_MIN_CLEARANCE_M if corner_thresholds else EDGE_MIN_CLEARANCE_M
    results: dict[str, Any] = {}
    for height_key in height_keys:
        typed_layers = []
        for layer in layers:
            keep = []
            for node in layer:
                minimum = node["per_height"].get(height_key, {}).get("min_clearance_m")
                if minimum is None or float(minimum) < threshold:
                    continue
                keep.append(node)
            typed_layers.append(keep)
        selection = None
        if all(typed_layers):
            position = {
                layer_index: {id(node): index for index, node in enumerate(layer)}
                for layer_index, layer in enumerate(typed_layers)
            }
            compact_edges = {}
            for (layer_index, src, dst), edge in edges.items():
                node_a = layers[layer_index][src]
                node_b = layers[layer_index + 1][dst]
                new_src = position[layer_index].get(id(node_a))
                new_dst = position[layer_index + 1].get(id(node_b))
                if new_src is None or new_dst is None:
                    continue
                minimum = edge["min_clearance_by_height_m"].get(height_key)
                passed = bool(
                    minimum is not None
                    and float(minimum) >= threshold
                    and edge.get("failure") is None
                    and edge["x_monotonic"]
                )
                compact_edges[(layer_index, new_src, new_dst)] = {
                    "passed": passed,
                    "min_clearance_m": float(minimum) if minimum is not None else -1.0,
                    "joint_travel_rad": float(edge["joint_travel_rad"]),
                    "min_joint_limit_margin_rad": float(
                        edge["min_joint_limit_margin_rad"]
                    ),
                }
            selection = select_path(typed_layers, compact_edges)
        results[height_key] = {
            "feasible": selection is not None,
            "selection": selection,
            "layers": typed_layers,
            "n_nodes_per_layer": [len(layer) for layer in typed_layers],
        }
    return {
        "template_key": template["template_key"],
        "template": {
            key: value for key, value in template.items() if key != "pass_rotation_axis"
        },
        "feasible_by_height": {
            key: bool(results[key]["feasible"]) for key in height_keys
        },
        "per_height": results,
        "rejections": rejections,
        "n_nodes_per_layer": [len(layer) for layer in layers],
        "control_poses": [
            {
                "name": pose["name"],
                "position_m": np.asarray(pose["position_m"], dtype=float).tolist(),
            }
            for pose in poses
        ],
    }


def finalize_route(
    context: CellContext,
    stock: dict[str, Any],
    template: dict[str, Any],
    evaluation: dict[str, Any],
    height_key: str,
    edge_cache: dict[tuple, dict[str, Any]],
) -> dict[str, Any]:
    """Concatenate the selected path and score the realized FK route."""
    result = evaluation["per_height"][height_key]
    selection = result["selection"]
    layers = result["layers"]
    nodes = [layers[i][j] for i, j in enumerate(selection["node_indices"])]
    middle = [pose["name"] for pose in evaluation["control_poses"]]
    if context.direction == "inbound":
        names = ["actual_initial"] + middle + ["actual_pregrasp_endpoint"]
    else:
        names = ["actual_loaded_lift"] + middle + ["actual_outbound_endpoint"]
    waypoints = [np.asarray(node["arm_qpos"], dtype=float) for node in nodes]
    segments = []
    path_chunks = []
    total_duration = 0.0
    for index in range(len(nodes) - 1):
        edge = edge_cache[(nodes[index]["node_key"], nodes[index + 1]["node_key"])]
        label = f"{names[index]}->{names[index + 1]}"
        speed_class = segment_speed_class(label, direction=context.direction)
        commanded = segment_commanded_speed(speed_class)
        duration = segment_duration_s(
            tcp_arc_length_m=float(edge["tcp_arc_length_m"]),
            joint_displacements_rad=(waypoints[index + 1] - waypoints[index]),
            commanded_speed_m_s=commanded,
            velocity_limits_rad_s=context.velocity_limits,
        )
        total_duration += float(duration["duration_s"])
        segments.append(
            {
                "name": label,
                "speed_class": speed_class,
                "tcp_arc_length_m": float(edge["tcp_arc_length_m"]),
                "joint_travel_rad": float(edge["joint_travel_rad"]),
                "min_clearance_m": edge["min_clearance_by_height_m"].get(height_key),
                "n_interpolation_samples": int(edge["n_samples"]),
                "x_monotonic": bool(edge["x_monotonic"]),
                **duration,
            }
        )
        chunk = np.asarray(edge["tcp_path_m"], dtype=float)
        path_chunks.append(chunk if index == 0 else chunk[1:])
    path = np.vstack(path_chunks)
    geometry = route_geometry_report(
        path,
        np.asarray(stock["positions_m"], dtype=float),
        lane_y_m=float(template["lane_y_m"]),
        direction=context.direction,
        control_x_sequence=[float(node["tcp_position_m"][0]) for node in nodes],
    )
    return {
        "cell_key": context.cell_key,
        "direction": context.direction,
        "height_key": height_key,
        "template_key": template["template_key"],
        "route_key": selection["route_key"],
        "control_pose_names": names,
        "waypoints_rad": [item.tolist() for item in waypoints],
        "qpos_sha256": qpos_sequence_sha256(waypoints),
        "n_waypoints": len(waypoints),
        "min_clearance_m": float(selection["min_clearance_m"]),
        "min_joint_limit_margin_rad": float(selection["min_joint_limit_margin_rad"]),
        "total_joint_travel_rad": float(selection["total_joint_travel_rad"]),
        "total_orientation_deviation_deg": float(
            selection["total_orientation_deviation_deg"]
        ),
        "total_duration_s": float(total_duration),
        "segments": segments,
        "geometry": geometry,
        "target_held": bool(stock["target_held"]),
        "start_step_index": int(stock["start_step_index"]),
        "end_step_index": int(stock["end_step_index"]),
        "admitted": bool(
            geometry["detour_meets_minimum"]
            and geometry["correct_lane_side"]
            and geometry["aperture_contained"]
            and all(item["x_monotonic"] for item in segments)
        ),
    }


def _rank_key(route: dict[str, Any]) -> tuple:
    return (
        -float(route["min_clearance_m"]),
        -float(route["min_joint_limit_margin_rad"]),
        float(route["total_joint_travel_rad"]),
        float(route["total_orientation_deviation_deg"]),
        str(route["route_key"]),
    )


def _endpoint_summary(endpoints: dict[str, Any], height_keys) -> dict[str, Any]:
    return {
        which: {
            "step_index": endpoints[which]["step_index"],
            "qpos_sha256": endpoints[which]["qpos_sha256"],
            "tcp_position_residual_m": endpoints[which]["tcp_position_residual_m"],
            "tcp_orientation_residual_deg": endpoints[which][
                "tcp_orientation_residual_deg"
            ],
            "base_valid": bool(endpoints[f"{which}_node"]["base_valid"]),
            "per_height_min_clearance_m": {
                key: endpoints[f"{which}_node"]["per_height"][key]["min_clearance_m"]
                for key in height_keys
            },
        }
        for which in ("start", "end")
    }


def _pin_threads() -> None:
    """One BLAS/OMP thread per process.

    The IK solve is a small damped least squares; multi-threaded BLAS is pure
    overhead, and twelve processes each spawning one thread per core on a
    128-core host spend all their time in futex contention rather than working.
    """
    import os

    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"


def run_case(payload: dict[str, Any]) -> dict[str, Any]:
    _pin_threads()
    establish_v10_runtime_env()
    started = time.time()
    assemblies = enumerate_v103_assemblies()
    boxes = pendant_boxes(assemblies)
    height_keys = list(boxes)
    context = CellContext(payload["job"], payload["cell"], payload["direction"])
    try:
        stock = stock_reference(context)
        endpoints = {
            "start": endpoint_state(context, stock, "start"),
            "end": endpoint_state(context, stock, "end"),
        }
        carry = bool(stock["target_held"])
        for which in ("start", "end"):
            context.apply_recorded_step(endpoints[which]["step_index"])
            if which == "start" and carry:
                context.freeze_carry()
            node = evaluate_node(
                context,
                endpoints[which]["arm_qpos"],
                _pose_matrix(
                    stock[f"{'start' if which == 'start' else 'end'}_position_m"],
                    stock[f"{'start' if which == 'start' else 'end'}_rotation"],
                ),
                boxes,
                carry=carry,
            )
            node["control_pose"] = f"actual_{which}"
            node["seed_provenance"] = "pinned_actual_qpos"
            node["node_key"] = qpos_sequence_sha256([endpoints[which]["arm_qpos"]])[:16]
            endpoints[f"{which}_node"] = node
        # A pinned endpoint that already violates the node clearance floor
        # cannot be rescued by any route template: every edge leaving it has
        # that sample as its own first sample. Record the witness and drop the
        # height rather than spending template work that provably cannot pass.
        height_admission: dict[str, Any] = {}
        for height_key in height_keys:
            entry: dict[str, Any] = {"height_key": height_key}
            admitted = True
            for which in ("start", "end"):
                node = endpoints[f"{which}_node"]
                minimum = node["per_height"][height_key]["min_clearance_m"]
                clear = bool(
                    minimum is not None and float(minimum) >= NODE_MIN_CLEARANCE_M
                )
                entry[which] = {
                    "min_clearance_m": minimum,
                    "per_component_m": node["per_height"][height_key][
                        "per_component_m"
                    ],
                    "meets_node_floor": clear,
                    "base_valid": bool(node["base_valid"]),
                }
                if not clear:
                    context.apply_recorded_step(endpoints[which]["step_index"])
                    if which == "start" and carry:
                        context.freeze_carry()
                    context.set_arm(endpoints[which]["arm_qpos"], carry=carry)
                    entry[which]["witness"] = context.nearest_geom_witness(
                        boxes[height_key], include_target=carry
                    )
                admitted = admitted and clear
            entry["admitted"] = admitted
            height_admission[height_key] = entry
        alive_heights = [
            key for key in height_keys if height_admission[key]["admitted"]
        ]
        if not alive_heights:
            return {
                "cell_key": context.cell_key,
                "role_index": context.role_index,
                "family": context.family,
                "intrusion_side": context.intrusion_side,
                "direction": context.direction,
                "n_templates": 0,
                "n_templates_with_complete_layers": 0,
                "n_feasible_by_height": {key: 0 for key in height_keys},
                "best_by_height": {key: None for key in height_keys},
                "all_feasible_route_keys": {key: [] for key in height_keys},
                "rejections": {"pinned_endpoint_clearance": len(height_keys)},
                "height_admission": height_admission,
                "excluded_heights": list(height_keys),
                "stop": "pinned_endpoint_clearance_below_node_floor",
                "endpoints": _endpoint_summary(endpoints, height_keys),
                "velocity_limit_source": context.velocity_limit_source,
                "counters": dict(context.counters),
                "elapsed_s": float(time.time() - started),
                "node_rows": np.zeros((0, 21), dtype=np.float32),
                "edge_rows": np.zeros((0, 14), dtype=np.float32),
            }
        templates = enumerate_templates(context.intrusion_side)
        limit = int(payload.get("template_limit") or 0)
        if limit:
            templates = templates[:limit]
        edge_cache: dict[tuple, dict[str, Any]] = {}
        feasible: dict[str, list[dict[str, Any]]] = {key: [] for key in height_keys}
        rejections: dict[str, int] = {}
        n_templates_with_layers = 0
        node_rows: list[list[float]] = []
        edge_rows: list[list[float]] = []
        for template_index, template in enumerate(templates):
            evaluation = evaluate_template(
                context,
                stock,
                template,
                {key: boxes[key] for key in alive_heights},
                alive_heights,
                endpoints,
                edge_cache,
                node_recorder=node_rows,
                edge_recorder=edge_rows,
                template_index=template_index,
            )
            for code, count in evaluation.get("rejections", {}).items():
                rejections[code] = rejections.get(code, 0) + count
            if evaluation.get("stop") == "empty_layer":
                continue
            n_templates_with_layers += 1
            for height_key in alive_heights:
                if not evaluation["feasible_by_height"][height_key]:
                    continue
                route = finalize_route(
                    context, stock, template, evaluation, height_key, edge_cache
                )
                if not route["admitted"]:
                    rejections["route_geometry"] = rejections.get("route_geometry", 0) + 1
                    continue
                feasible[height_key].append(route)
        for height_key in height_keys:
            feasible[height_key].sort(key=_rank_key)
        return {
            "cell_key": context.cell_key,
            "role_index": context.role_index,
            "family": context.family,
            "intrusion_side": context.intrusion_side,
            "direction": context.direction,
            "n_templates": len(templates),
            "n_templates_with_complete_layers": n_templates_with_layers,
            "n_feasible_by_height": {
                key: len(feasible[key]) for key in height_keys
            },
            "best_by_height": {
                key: (feasible[key][0] if feasible[key] else None)
                for key in height_keys
            },
            "all_feasible_route_keys": {
                key: [item["template_key"] for item in feasible[key]]
                for key in height_keys
            },
            "rejections": rejections,
            "height_admission": height_admission,
            "excluded_heights": [
                key for key in height_keys if key not in alive_heights
            ],
            "stop": None,
            "endpoints": {
                which: {
                    "step_index": endpoints[which]["step_index"],
                    "qpos_sha256": endpoints[which]["qpos_sha256"],
                    "tcp_position_residual_m": endpoints[which]["tcp_position_residual_m"],
                    "tcp_orientation_residual_deg": endpoints[which][
                        "tcp_orientation_residual_deg"
                    ],
                    "base_valid": bool(endpoints[f"{which}_node"]["base_valid"]),
                    "per_height_min_clearance_m": {
                        key: endpoints[f"{which}_node"]["per_height"][key][
                            "min_clearance_m"
                        ]
                        for key in height_keys
                    },
                }
                for which in ("start", "end")
            },
            "velocity_limit_source": context.velocity_limit_source,
            "counters": dict(context.counters),
            "elapsed_s": float(time.time() - started),
            "node_rows": np.asarray(node_rows, dtype=np.float32)
            if node_rows
            else np.zeros((0, 21), dtype=np.float32),
            "edge_rows": np.asarray(edge_rows, dtype=np.float32)
            if edge_rows
            else np.zeros((0, 14), dtype=np.float32),
        }
    finally:
        context.close()


def _cases(template_limit: int = 0) -> list[dict[str, Any]]:
    reconstruction, _snapshot, cells = verify_v99_inputs()
    cells = sorted(cells, key=lambda item: int(item["role_index"]))
    jobs = {
        int(json.loads((Path(job["row_dir"]) / "result.json").read_text())["role_index"]): job
        for job in snapshot_jobs_from_reconstruction(reconstruction)
    }
    payloads = []
    for cell in cells:
        # Only these fields are used downstream. The full snapshot carries mesh
        # vertex catalogs and per-frame pose arrays; pickling those to twelve
        # workers starves them while the parent serializes gigabytes.
        slim = {
            "role_index": int(cell["role_index"]),
            "family": str(cell["family"]),
            "intrusion_side": str(cell["intrusion_side"]),
            "tcp_m": np.asarray(cell["tcp_m"], dtype=float),
            "tcp_mat": np.asarray(cell["tcp_mat"], dtype=float),
            "inbound_mask": np.asarray(cell["inbound_mask"], dtype=bool),
            "outbound_mask": np.asarray(cell["outbound_mask"], dtype=bool),
        }
        for direction in ("inbound", "outbound"):
            payloads.append(
                {
                    "cell": slim,
                    "job": jobs[int(cell["role_index"])],
                    "direction": direction,
                    "template_limit": template_limit,
                }
            )
    return payloads


def select_geometry(cases: Sequence[dict[str, Any]], height_keys: Sequence[str]):
    """Fixed geometry ranking over the heights that route all twelve cases."""
    from pact_place_v103_geometry import HEIGHT_LATTICE_M

    survivors = []
    for height_key in height_keys:
        routes = [case["best_by_height"].get(height_key) for case in cases]
        if any(route is None for route in routes):
            continue
        survivors.append(
            {
                "height_key": height_key,
                "lowest_lobe_bottom_z_m": float(height_key),
                "worst_clearance_m": float(
                    min(float(route["min_clearance_m"]) for route in routes)
                ),
                "worst_joint_limit_margin_rad": float(
                    min(float(route["min_joint_limit_margin_rad"]) for route in routes)
                ),
                "total_joint_travel_rad": float(
                    sum(float(route["total_joint_travel_rad"]) for route in routes)
                ),
                "routes": routes,
            }
        )
    if not survivors:
        return None, survivors
    survivors.sort(
        key=lambda item: (
            -item["worst_clearance_m"],
            -item["lowest_lobe_bottom_z_m"],
            -item["worst_joint_limit_margin_rad"],
            item["total_joint_travel_rad"],
            item["height_key"],
        )
    )
    return survivors[0], survivors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--template-limit", type=int, default=0)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    _pin_threads()
    establish_v10_runtime_env()
    protected = verify_protected_artifacts()
    payloads = _cases(args.template_limit)
    if args.pilot:
        payloads = payloads[:2]
    results: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, min(args.workers, len(payloads))),
        mp_context=context,
        max_tasks_per_child=1,
    ) as executor:
        futures = {executor.submit(run_case, payload): payload for payload in payloads}
        for future in concurrent.futures.as_completed(futures):
            payload = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"{result['cell_key']} {result['direction']}: "
                f"layers_ok={result['n_templates_with_complete_layers']}/"
                f"{result['n_templates']} feasible={result['n_feasible_by_height']} "
                f"ik={result['counters']['ik_calls']} "
                f"({result['elapsed_s']:.0f}s)",
                flush=True,
            )
    results.sort(key=lambda item: (item["cell_key"], item["direction"]))
    node_tables, edge_tables, case_labels = [], [], []
    for index, item in enumerate(results):
        nodes = np.asarray(item.pop("node_rows"), dtype=np.float32)
        edges = np.asarray(item.pop("edge_rows"), dtype=np.float32)
        case_labels.append(f"{item['cell_key']}|{item['direction']}")
        if nodes.size:
            node_tables.append(
                np.hstack([np.full((len(nodes), 1), index, dtype=np.float32), nodes])
            )
        if edges.size:
            edge_tables.append(
                np.hstack([np.full((len(edges), 1), index, dtype=np.float32), edges])
            )
        item["n_node_records"] = int(len(nodes))
        item["n_edge_records"] = int(len(edges))
    height_keys = list(HEIGHT_KEYS)
    selected, survivors = select_geometry(results, height_keys)
    stop_reason = None
    if selected is None:
        stop_reason = (
            "no_static_geometry_with_twelve_joint_routes"
            if any(
                any(case["n_feasible_by_height"].values()) for case in results
            )
            else "no_route_with_nominal_clearance"
        )
    document = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "implementation_sha256": implementation_sha256(),
        "protected_artifacts": protected,
        "search_lattice": search_lattice(),
        "pilot": bool(args.pilot),
        "template_limit": int(args.template_limit),
        "n_cases": len(results),
        "cases": results,
        "height_survivors": [
            {key: value for key, value in item.items() if key != "routes"}
            for item in survivors
        ],
        "selected_geometry": (
            None
            if selected is None
            else {key: value for key, value in selected.items() if key != "routes"}
        ),
        "selected_routes": None if selected is None else selected["routes"],
        "stop_reason": stop_reason,
        "search_passed": bool(selected is not None),
        **empty_authorization(),
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    node_array = (
        np.vstack(node_tables) if node_tables else np.zeros((0, 22), dtype=np.float32)
    )
    edge_array = (
        np.vstack(edge_tables) if edge_tables else np.zeros((0, 15), dtype=np.float32)
    )
    np.savez_compressed(
        output_root / "nodes.npz",
        rows=node_array,
        columns=np.asarray(
            [
                "case_index", "template_index", "layer_index", "seed_index",
                "seed_class", "ik_solved", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                "position_residual_m", "orientation_residual_deg",
                "joint_limit_margin_rad", "self_collision",
                "clearance_0.920_m", "clearance_0.960_m", "clearance_1.000_m",
                "clearance_1.040_m", "base_valid",
            ],
            dtype="U40",
        ),
        case_labels=np.asarray(case_labels, dtype="U64"),
        seed_class_codes=np.asarray(
            [f"{k}={v}" for k, v in SEED_CLASS_CODES.items()], dtype="U48"
        ),
    )
    np.savez_compressed(
        output_root / "edges.npz",
        rows=edge_array,
        columns=np.asarray(
            [
                "case_index", "template_index", "layer_index", "src", "dst",
                "n_samples", "joint_travel_rad", "tcp_arc_length_m", "x_monotonic",
                "min_joint_limit_margin_rad",
                "min_clearance_0.920_m", "min_clearance_0.960_m",
                "min_clearance_1.000_m", "min_clearance_1.040_m", "failure_code",
            ],
            dtype="U40",
        ),
        case_labels=np.asarray(case_labels, dtype="U64"),
        failure_codes=np.asarray(
            [f"{k}={v}" for k, v in EDGE_FAILURE_CODES.items()], dtype="U40"
        ),
    )
    routes = document["selected_routes"] or []
    np.savez_compressed(
        output_root / "selected_routes.npz",
        n_routes=np.asarray([len(routes)], dtype=np.int32),
        cell_keys=np.asarray([r["cell_key"] for r in routes], dtype="U64"),
        directions=np.asarray([r["direction"] for r in routes], dtype="U16"),
        template_keys=np.asarray([r["template_key"] for r in routes], dtype="U64"),
        qpos_sha256=np.asarray([r["qpos_sha256"] for r in routes], dtype="U64"),
        waypoints=(
            np.asarray([r["waypoints_rad"] for r in routes], dtype=np.float64)
            if routes
            else np.zeros((0, 0, 7), dtype=np.float64)
        ),
        min_clearance_m=np.asarray(
            [r["min_clearance_m"] for r in routes], dtype=np.float64
        ),
        stop_reason=np.asarray([stop_reason or ""], dtype="U64"),
    )
    for name in ("nodes.npz", "edges.npz", "selected_routes.npz"):
        document.setdefault("artifact_files", {})[name] = sha256_file(
            output_root / name
        )
    digest = write_immutable(output_root / "search.json", document)
    print(
        json.dumps(
            {
                "search_passed": document["search_passed"],
                "stop_reason": stop_reason,
                "selected_geometry": document["selected_geometry"],
                "artifact_sha256": digest,
            },
            indent=2,
        )
    )
    return 0 if document["search_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
