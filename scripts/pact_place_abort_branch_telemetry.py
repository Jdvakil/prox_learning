"""Read-only instrumentation for TCPMoveSequence.check_failure branches.

Does not change control flow, thresholds, or any value the controller consumes.
The predicates here copy MoveSequence/TCPMoveSequence.check_failure exactly.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

BRANCHES = (
    "empty_gripper",
    "pos_err",
    "rot_err",
    "sequence_complete",
    "ik_cascade",
    "unclassified",
)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def classify_check_failure_branch(
    *,
    failed: bool,
    action_index: int,
    n_primitives: int,
    sequential_ik_failures: int,
    max_sequential_ik_failures: int,
    is_holding_object: bool,
    inter_finger_dist_m: float | None,
    gripper_empty_trip_m: float | None,
    pos_err_m: float | None,
    rot_err_rad: float | None,
    tcp_pos_err_threshold_m: float | None,
    tcp_rot_err_threshold_rad: float | None,
) -> str | None:
    """Name the branch that would make check_failure return True, in code order."""
    if not failed:
        return None
    if sequential_ik_failures >= max_sequential_ik_failures:
        return "ik_cascade"
    if n_primitives <= 0 or action_index >= n_primitives:
        return "sequence_complete"
    empty = (
        is_holding_object
        and inter_finger_dist_m is not None
        and gripper_empty_trip_m is not None
        and inter_finger_dist_m < gripper_empty_trip_m
    )
    if empty:
        return "empty_gripper"
    if (
        pos_err_m is not None
        and tcp_pos_err_threshold_m is not None
        and pos_err_m > tcp_pos_err_threshold_m
    ):
        return "pos_err"
    if (
        rot_err_rad is not None
        and tcp_rot_err_threshold_rad is not None
        and rot_err_rad > tcp_rot_err_threshold_rad
    ):
        return "rot_err"
    return "unclassified"


def observe_abort_branch(policy, failed: bool) -> dict[str, Any]:
    """Read the same quantities TCPMoveSequence.check_failure tests. No writes."""
    from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
        MoveSequence,
        TCPMoveSequence,
    )
    from scipy.spatial.transform import Rotation as R

    primitives = getattr(policy, "action_primitives", None) or []
    action_index = int(getattr(policy, "action_idx", 0))
    sequential_ik = int(getattr(policy, "sequential_ik_failures", 0))
    max_ik = int(policy.policy_config.max_sequential_ik_failures)
    snapshot: dict[str, Any] = {
        "check_failure_returned": bool(failed),
        "action_index": action_index,
        "n_primitives": len(primitives),
        "sequential_ik_failures": sequential_ik,
        "max_sequential_ik_failures": max_ik,
        "policy_phase": None,
        "action_primitive": None,
        "move_segment_index": None,
        "is_holding_object": False,
        "inter_finger_dist_m": None,
        "inter_finger_dist_range0_m": None,
        "gripper_empty_threshold_m": None,
        "gripper_empty_trip_m": None,
        "empty_gripper_predicate": False,
        "pos_err_m": None,
        "rot_err_rad": None,
        "rot_err_deg": None,
        "tcp_pos_err_threshold_m": None,
        "tcp_rot_err_threshold_rad": None,
        "pos_err_predicate": False,
        "rot_err_predicate": False,
        "object_to_tcp_offset_m": None,
        "object_to_tcp_distance_m": None,
        "gripper_joint_pos_m": None,
        "gripper_joint_posadr": None,
        "empty_gripper_streak": 0,
        "branch": None,
    }
    try:
        snapshot["policy_phase"] = str(policy.get_phase())
    except Exception:
        snapshot["policy_phase"] = None

    primitive = None
    if 0 <= action_index < len(primitives):
        primitive = primitives[action_index]
        snapshot["action_primitive"] = type(primitive).__name__
        snapshot["empty_gripper_streak"] = int(
            getattr(primitive, "_empty_gripper_streak", 0) or 0
        )

    gripper = None
    try:
        gripper_id = policy.robot_view.get_gripper_movegroup_ids()[0]
        gripper = policy.robot_view.get_gripper(gripper_id)
        snapshot["inter_finger_dist_m"] = _float(gripper.inter_finger_dist)
        snapshot["inter_finger_dist_range0_m"] = _float(gripper.inter_finger_dist_range[0])
        try:
            snapshot["gripper_joint_pos_m"] = [
                float(value) for value in np.asarray(gripper.joint_pos).tolist()
            ]
            snapshot["gripper_joint_posadr"] = [
                int(index) for index in gripper._joint_posadr
            ]
        except Exception:
            snapshot["gripper_joint_pos_m"] = None
            snapshot["gripper_joint_posadr"] = None
    except Exception:
        gripper = None

    if isinstance(primitive, MoveSequence):
        snapshot["is_holding_object"] = bool(primitive.is_holding_object)
        snapshot["gripper_empty_threshold_m"] = _float(primitive.gripper_empty_threshold)
        if snapshot["inter_finger_dist_range0_m"] is not None:
            snapshot["gripper_empty_trip_m"] = (
                snapshot["inter_finger_dist_range0_m"]
                + float(primitive.gripper_empty_threshold)
            )
        snapshot["empty_gripper_predicate"] = bool(
            primitive.is_holding_object
            and snapshot["inter_finger_dist_m"] is not None
            and snapshot["gripper_empty_trip_m"] is not None
            and snapshot["inter_finger_dist_m"] < snapshot["gripper_empty_trip_m"]
        )
        if isinstance(primitive, TCPMoveSequence):
            snapshot["tcp_pos_err_threshold_m"] = _float(primitive.tcp_pos_err_threshold)
            snapshot["tcp_rot_err_threshold_rad"] = _float(
                primitive.tcp_rot_err_threshold
            )
            if primitive.move_seg_idx is not None:
                snapshot["move_segment_index"] = int(primitive.move_seg_idx)
                try:
                    target_pose = primitive.get_current_target_pose()
                    actual_pose = gripper.leaf_frame_to_world
                    trf = np.linalg.inv(actual_pose) @ target_pose
                    pos_err = float(np.linalg.norm(trf[:3, 3]))
                    rot_err = float(R.from_matrix(trf[:3, :3]).magnitude())
                    snapshot["pos_err_m"] = pos_err
                    snapshot["rot_err_rad"] = rot_err
                    snapshot["rot_err_deg"] = float(np.degrees(rot_err))
                    snapshot["pos_err_predicate"] = bool(
                        pos_err > primitive.tcp_pos_err_threshold
                    )
                    snapshot["rot_err_predicate"] = bool(
                        rot_err > primitive.tcp_rot_err_threshold
                    )
                except Exception:
                    pass

    try:
        manager = policy.task.env.object_managers[policy.task.env.current_batch_index]
        pickup = manager.get_object_by_name(policy.task.config.task_config.pickup_obj_name)
        tcp = gripper.leaf_frame_to_world[:3, 3]
        offset = np.asarray(pickup.position, dtype=float) - np.asarray(tcp, dtype=float)
        snapshot["object_to_tcp_offset_m"] = [float(v) for v in offset]
        snapshot["object_to_tcp_distance_m"] = float(np.linalg.norm(offset))
    except Exception:
        pass

    snapshot["branch"] = classify_check_failure_branch(
        failed=failed,
        action_index=action_index,
        n_primitives=len(primitives),
        sequential_ik_failures=sequential_ik,
        max_sequential_ik_failures=max_ik,
        is_holding_object=bool(snapshot["is_holding_object"]),
        inter_finger_dist_m=snapshot["inter_finger_dist_m"],
        gripper_empty_trip_m=snapshot["gripper_empty_trip_m"],
        pos_err_m=snapshot["pos_err_m"],
        rot_err_rad=snapshot["rot_err_rad"],
        tcp_pos_err_threshold_m=snapshot["tcp_pos_err_threshold_m"],
        tcp_rot_err_threshold_rad=snapshot["tcp_rot_err_threshold_rad"],
    )
    return snapshot


def record_on_policy(policy, failed: bool) -> dict[str, Any]:
    snapshot = observe_abort_branch(policy, failed)
    steps = getattr(policy, "_pact_abort_branch_steps", None)
    if steps is None:
        policy._pact_abort_branch_steps = []
        steps = policy._pact_abort_branch_steps
    steps.append(snapshot)
    if failed and getattr(policy, "_pact_abort_branch_terminal", None) is None:
        policy._pact_abort_branch_terminal = snapshot
    return snapshot


def terminal_tracking_fields(policy) -> dict[str, Any]:
    snapshot = getattr(policy, "_pact_abort_branch_terminal", None)
    if snapshot is None:
        snapshot = observe_abort_branch(policy, failed=False)
    return {
        "check_failure_branch": snapshot.get("branch"),
        "check_failure_returned": snapshot.get("check_failure_returned"),
        "inter_finger_dist_m": snapshot.get("inter_finger_dist_m"),
        "inter_finger_dist_range0_m": snapshot.get("inter_finger_dist_range0_m"),
        "gripper_empty_threshold_m": snapshot.get("gripper_empty_threshold_m"),
        "gripper_empty_trip_m": snapshot.get("gripper_empty_trip_m"),
        "is_holding_object": snapshot.get("is_holding_object"),
        "rotation_error_rad": snapshot.get("rot_err_rad"),
        "rotation_error_deg": snapshot.get("rot_err_deg"),
        "gripper_joint_pos_m": snapshot.get("gripper_joint_pos_m"),
        "gripper_joint_posadr": snapshot.get("gripper_joint_posadr"),
        "empty_gripper_streak": snapshot.get("empty_gripper_streak"),
        "check_pos_err_m": snapshot.get("pos_err_m"),
        "empty_gripper_predicate": snapshot.get("empty_gripper_predicate"),
        "pos_err_predicate": snapshot.get("pos_err_predicate"),
        "rot_err_predicate": snapshot.get("rot_err_predicate"),
        "object_to_tcp_offset_m": snapshot.get("object_to_tcp_offset_m"),
        "object_to_tcp_distance_m": snapshot.get("object_to_tcp_distance_m"),
    }


def write_sidecar(policy) -> Path | None:
    raw = os.environ.get("PACT_PLACE_ABORT_SIDECAR")
    if not raw:
        return None
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    terminal = getattr(policy, "_pact_abort_branch_terminal", None)
    payload = {
        "schema_version": "pact_place_abort_branch_telemetry_v1",
        "role": "diagnostic_not_a_gate",
        "authorizes_collection": False,
        "terminal": terminal,
        "n_steps": len(getattr(policy, "_pact_abort_branch_steps", []) or []),
        "steps": list(getattr(policy, "_pact_abort_branch_steps", []) or []),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def install() -> None:
    """Wrap planner methods on the imported producing tree. Return values unchanged."""
    from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
        BaseObjectManipulationPlannerPolicy,
    )
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

    pact_check = PactPlaceCorridorPolicy._check_for_failures
    base_check = BaseObjectManipulationPlannerPolicy._check_for_failures
    if pact_check is base_check:
        def wrapped_check(self) -> bool:
            failed = base_check(self)
            record_on_policy(self, failed)
            return failed

        wrapped_check._pact_abort_instrumented = True  # type: ignore[attr-defined]
        BaseObjectManipulationPlannerPolicy._check_for_failures = wrapped_check

    if (
        PactPlaceCorridorPolicy._handle_failure
        is BaseObjectManipulationPlannerPolicy._handle_failure
    ):
        _install_handle_failure(BaseObjectManipulationPlannerPolicy)
    _install_get_info(PactPlaceCorridorPolicy)


def _install_handle_failure(cls) -> None:
    if getattr(cls._handle_failure, "_pact_abort_instrumented", False):
        return
    original = cls._handle_failure

    def wrapped(self):
        if getattr(self, "_pact_abort_branch_terminal", None) is None:
            snapshot = record_on_policy(self, True)
            if (
                int(self.sequential_ik_failures)
                >= int(self.policy_config.max_sequential_ik_failures)
            ):
                snapshot["branch"] = "ik_cascade"
                self._pact_abort_branch_terminal = snapshot
        return original(self)

    wrapped._pact_abort_instrumented = True  # type: ignore[attr-defined]
    cls._handle_failure = wrapped


def _install_get_info(cls) -> None:
    if getattr(cls.get_info, "_pact_abort_instrumented", False):
        return
    original = cls.get_info

    def wrapped(self):
        info = original(self)
        tracking = dict(info.get("terminal_tracking") or {})
        tracking.update(terminal_tracking_fields(self))
        info["terminal_tracking"] = tracking
        write_sidecar(self)
        return info

    wrapped._pact_abort_instrumented = True  # type: ignore[attr-defined]
    cls.get_info = wrapped


def cup_collision_geom_ids(model, env, pickup_name: str) -> set[int]:
    from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms

    manager = env.object_managers[env.current_batch_index]
    pickup = manager.get_object_by_name(pickup_name)
    geoms = descendant_geoms(model, int(pickup.object_id), visual_only=False)
    ids = set()
    for geom_id in geoms:
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        if contype or conaffinity:
            ids.add(int(geom_id))
    return ids


def wall_thickness_along_radial_ray(
    model,
    data,
    *,
    body_id: int,
    local_point: np.ndarray,
    cup_geom_ids: set[int],
) -> dict[str, Any]:
    """Thickness of Cup_10 at a grasp-local point, along the local XZ radial."""
    local_point = np.asarray(local_point, dtype=np.float64)
    xpos = np.asarray(data.xpos[body_id], dtype=np.float64)
    xmat = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    axis_local = np.array([0.0, 1.0, 0.0])
    radial_local = np.array([local_point[0], 0.0, local_point[2]], dtype=np.float64)
    radial_norm = float(np.linalg.norm(radial_local))
    if radial_norm < 1e-9:
        return {
            "grasp_object_local_position_m": [float(v) for v in local_point],
            "radial_distance_from_axis_m": 0.0,
            "wall_thickness_m": None,
            "reason": "grasp_on_axis",
        }
    radial_dir_local = radial_local / radial_norm
    axis_point_local = np.array([0.0, local_point[1], 0.0])
    axis_point_world = xpos + xmat @ axis_point_local
    radial_dir_world = xmat @ radial_dir_local
    import mujoco

    start = axis_point_world + radial_dir_world * 0.12
    direction = -radial_dir_world
    hits: list[dict[str, Any]] = []
    origin = start.copy()
    leftover = 0.24
    for _ in range(12):
        geomid = np.array([-1], dtype=np.int32)
        dist = float(
            mujoco.mj_ray(model, data, origin, direction, None, 1, -1, geomid)
        )
        if dist < 0.0 or dist > leftover:
            break
        origin = origin + direction * dist
        leftover -= dist
        gid = int(geomid[0])
        if gid in cup_geom_ids:
            radius = float(np.linalg.norm(origin - axis_point_world))
            hits.append(
                {
                    "geom_id": gid,
                    "geom_name": str(model.geom(gid).name),
                    "radius_from_axis_m": radius,
                    "hit_world_m": [float(v) for v in origin],
                }
            )
        origin = origin + direction * 2e-4
        leftover -= 2e-4
        if leftover <= 0.0:
            break
    radii = [hit["radius_from_axis_m"] for hit in hits]
    outer = None if not radii else max(radii)
    inner = None if not radii else min(radii)
    thickness = None
    if outer is not None and inner is not None and len(radii) >= 2:
        # A ray through a hollow cup hits outer/inner on each side. Consecutive
        # inner-to-inner hits across the cavity are not wall thickness.
        thickness = outer - inner
    return {
        "grasp_object_local_position_m": [float(v) for v in local_point],
        "radial_distance_from_axis_m": radial_norm,
        "n_cup_hits": len(hits),
        "hits": hits[:4],
        "wall_thickness_m": thickness,
        "outer_radius_m": outer,
        "inner_radius_m": inner,
    }
