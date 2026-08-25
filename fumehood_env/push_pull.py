"""Push and pull tasks for the cluttered fume-hood.

Push: closed-gripper sweep that shoves the object along the bench toward a
target drawn per episode (deeper into the hood or laterally). Pull: grasp the
object and drag it along the bench toward the mouth without lifting. Both keep
the clutter, hood-size and deep-reach machinery of the pick sampler.

Design choices that keep this on well-trodden upstream paths:

* End-effector orientation comes from ``compute_grasp_pose`` (the annotated
  grasp files), and the push/pull waypoints are translations of that anchor —
  no hand-rolled rotation matrices, and every waypoint is IK-checked the same
  way the pick planner does it.
* Success is displacement-based and read from the same task-config fields the
  pick task already logs (``pickup_obj_start_pose`` / ``pickup_obj_goal_pose``),
  so nothing new needs to be persisted.
"""
from __future__ import annotations

import logging

import numpy as np

from molmo_spaces.configs.policy_configs import ObjectManipulationPlannerPolicyConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    BaseObjectManipulationPlannerPolicy,
    GripperAction,
    TCPMoveSegment,
    TCPMoveSequence,
)
from molmo_spaces.tasks.enclosure_reach import TUBE_X0
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.utils.grasp_sample import compute_grasp_pose

from fumehood_env.cluttered_fumehood import ClutteredFumehoodPickSampler

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tasks: displacement-based success                                           #
# --------------------------------------------------------------------------- #
class PushFumehoodTask(PickTask):
    """Success: the object progressed >= SUCC_PROGRESS along start->goal (XY)
    and stayed on the bench (no big z excursion, no topple off)."""

    SUCC_PROGRESS = 0.08   # metres of displacement along the commanded direction
    Z_GUARD = 0.12         # object must stay within this of its start height

    def get_task_description(self) -> str:
        name = self.config.task_config.referral_expressions.get("pickup_obj_name", "object")
        return f"Push the {name} along the bench"

    def _progress(self, data) -> tuple[float, float]:
        tc = self.config.task_config
        obj = MlSpacesObject(data=data, object_name=tc.pickup_obj_name)
        start = np.asarray(tc.pickup_obj_start_pose[:3], dtype=float)
        goal = np.asarray(tc.pickup_obj_goal_pose[:3], dtype=float)
        direction = goal[:2] - start[:2]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return 0.0, 0.0
        progress = float(np.dot(obj.position[:2] - start[:2], direction / norm))
        dz = float(abs(obj.position[2] - start[2]))
        return progress, dz

    def judge_success(self) -> bool:
        return self.get_info()[0]["success"]

    def get_info(self):
        metrics = []
        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]
            progress, dz = self._progress(data)
            tc = self.config.task_config
            obj = MlSpacesObject(data=data, object_name=tc.pickup_obj_name)
            goal = np.asarray(tc.pickup_obj_goal_pose[:3], dtype=float)
            metrics.append({
                "position_error": float(np.linalg.norm(obj.position[:2] - goal[:2])),
                "rotation_error": 0.0,
                "progress": progress,
                "success": bool(progress >= self.SUCC_PROGRESS and dz <= self.Z_GUARD),
                "episode_step": self.episode_step_count,
            })
        return metrics

    def get_reward(self) -> np.ndarray:
        rewards = np.zeros(self._env.n_batch)
        for i in range(self._env.n_batch):
            progress, dz = self._progress(self._env.mj_datas[i])
            rewards[i] = np.clip(progress if dz <= self.Z_GUARD else 0.0, 0.0, 1000.0)
        return rewards


class PullFumehoodTask(PushFumehoodTask):
    """Same displacement criterion; the sampler draws the goal toward the mouth,
    so 'pull' is push-task machinery with an outward goal and a dragging expert."""

    def get_task_description(self) -> str:
        name = self.config.task_config.referral_expressions.get("pickup_obj_name", "object")
        return f"Pull the {name} toward the front of the hood"


# --------------------------------------------------------------------------- #
# Planner policies                                                            #
# --------------------------------------------------------------------------- #
def _grasp_anchor(policy) -> np.ndarray:
    """Feasible EE pose at the object from the annotated grasp files — used as
    the orientation anchor for every push/pull waypoint."""
    task_config = policy.config.task_config
    robot_view = policy.task.env.current_robot.robot_view
    om = policy.task.env.object_managers[policy.task.env.current_batch_index]
    pickup_obj = om.get_object_by_name(task_config.pickup_obj_name)
    return compute_grasp_pose(
        policy,
        pickup_obj,
        robot_view,
        check_collision=policy.policy_config.filter_colliding_grasps,
        n_collision_checks=policy.policy_config.grasp_collision_max_grasps,
        collision_batch_size=policy.policy_config.grasp_collision_batch_size,
        check_ik=policy.policy_config.filter_feasible_grasps,
        n_ik_checks=policy.policy_config.grasp_feasibility_max_grasps,
        ik_batch_size=policy.policy_config.grasp_feasibility_batch_size,
        pos_cost_weight=policy.policy_config.grasp_pos_cost_weight,
        rot_cost_weight=policy.policy_config.grasp_rot_cost_weight,
        vertical_cost_weight=policy.policy_config.grasp_vertical_cost_weight,
        com_dist_cost_weight=policy.policy_config.grasp_com_dist_cost_weight,
    )


def _translated(pose: np.ndarray, dxyz) -> np.ndarray:
    out = pose.copy()
    out[:3, 3] += np.asarray(dxyz, dtype=float)
    return out


def _require_ik(policy, name: str, pose: np.ndarray) -> np.ndarray:
    if not policy.check_feasible_ik(pose):
        raise ValueError(f"IK failed for {name} pose")
    return pose


class PushPlannerPolicy(BaseObjectManipulationPlannerPolicy):
    """Close the gripper, come in behind the object, sweep through it toward
    the goal, retreat upward."""

    def _compute_target_poses(self) -> dict[str, np.ndarray]:
        pc = self.policy_config
        tc = self.config.task_config
        anchor = _grasp_anchor(self)
        start = np.asarray(tc.pickup_obj_start_pose[:3], dtype=float)
        goal = np.asarray(tc.pickup_obj_goal_pose[:3], dtype=float)
        d = goal[:2] - start[:2]
        dist = float(np.clip(np.linalg.norm(d), 0.06, 0.12))
        u0 = d / max(float(np.linalg.norm(d)), 1e-6)

        # The drawn direction often fails IK (pushing deeper into a small hood
        # runs out of reach), so fall back through easier directions: lateral,
        # then outward toward the mouth. Whichever passes rewrites the task
        # goal so success is judged on the direction actually executed.
        candidates = [u0, np.array([0.0, 1.0]), np.array([0.0, -1.0]),
                      np.array([-1.0, 0.0])]
        last_err = None
        for u in candidates:
            back = -u * pc.push_standoff
            fwd = u * (dist + pc.push_overshoot)
            poses = {
                "approach": _translated(anchor, [back[0], back[1], pc.approach_z]),
                "contact": _translated(anchor, [back[0], back[1], 0.0]),
                "push_end": _translated(anchor, [fwd[0], fwd[1], 0.0]),
            }
            poses["retreat"] = _translated(poses["push_end"], [0.0, 0.0, pc.retreat_z])
            try:
                for name, pose in poses.items():
                    _require_ik(self, name, pose)
            except ValueError as e:
                last_err = e
                continue
            new_goal = list(tc.pickup_obj_goal_pose)
            new_goal[0] = float(start[0] + u[0] * dist)
            new_goal[1] = float(start[1] + u[1] * dist)
            tc.pickup_obj_goal_pose = new_goal
            # some base-policy paths peek at poses["grasp"]; alias the contact
            poses["grasp"] = poses["contact"]
            return poses
        raise last_err

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        pc = self.policy_config
        robot_view = self.task.env.current_robot.robot_view
        poses = self._compute_target_poses()
        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        start_ee = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world
        common = dict(
            gripper_empty_threshold=pc.gripper_empty_threshold,
            tcp_pos_err_threshold=pc.tcp_pos_err_threshold,
            tcp_rot_err_threshold=pc.tcp_rot_err_threshold,
        )
        return [
            GripperAction(robot_view, False, pc.gripper_close_duration),
            TCPMoveSequence(
                robot_view, self._tcp_to_jp_fn, pc.move_settle_time, **common,
                move_segments=[
                    TCPMoveSegment(name="approach", start_pose=start_ee,
                                   end_pose=poses["approach"], speed=pc.speed_fast),
                    TCPMoveSegment(name="contact", start_pose=poses["approach"],
                                   end_pose=poses["contact"], speed=pc.speed_slow),
                ],
            ),
            TCPMoveSequence(
                robot_view, self._tcp_to_jp_fn, pc.move_settle_time, **common,
                move_segments=[
                    TCPMoveSegment(name="push", start_pose=poses["contact"],
                                   end_pose=poses["push_end"], speed=pc.speed_slow),
                ],
            ),
            TCPMoveSequence(
                robot_view, self._tcp_to_jp_fn, pc.move_settle_time, **common,
                move_segments=[
                    TCPMoveSegment(name="retreat", start_pose=poses["push_end"],
                                   end_pose=poses["retreat"], speed=pc.speed_fast),
                ],
            ),
        ]


class PullPlannerPolicy(BaseObjectManipulationPlannerPolicy):
    """Grasp, drag along the bench toward the mouth (small lift to break
    friction, no transport height), release, retreat."""

    def _compute_target_poses(self) -> dict[str, np.ndarray]:
        pc = self.policy_config
        tc = self.config.task_config
        anchor = _grasp_anchor(self)
        start = np.asarray(tc.pickup_obj_start_pose[:3], dtype=float)
        goal = np.asarray(tc.pickup_obj_goal_pose[:3], dtype=float)
        d = goal[:2] - start[:2]

        poses = {
            "pregrasp": _translated(anchor, [0.0, 0.0, pc.pregrasp_z_offset]),
            "grasp": anchor,
            "drag_end": _translated(anchor, [d[0], d[1], pc.drag_lift]),
        }
        poses["retreat"] = _translated(poses["drag_end"], [0.0, 0.0, pc.retreat_z])
        for name, pose in poses.items():
            _require_ik(self, name, pose)
        return poses

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        pc = self.policy_config
        robot_view = self.task.env.current_robot.robot_view
        poses = self._compute_target_poses()
        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        start_ee = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world
        common = dict(
            gripper_empty_threshold=pc.gripper_empty_threshold,
            tcp_pos_err_threshold=pc.tcp_pos_err_threshold,
            tcp_rot_err_threshold=pc.tcp_rot_err_threshold,
        )
        return [
            GripperAction(robot_view, True, 0.0),
            TCPMoveSequence(
                robot_view, self._tcp_to_jp_fn, pc.move_settle_time, **common,
                move_segments=[
                    TCPMoveSegment(name="pregrasp", start_pose=start_ee,
                                   end_pose=poses["pregrasp"], speed=pc.speed_fast),
                    TCPMoveSegment(name="grasp", start_pose=poses["pregrasp"],
                                   end_pose=poses["grasp"], speed=pc.speed_slow),
                ],
            ),
            GripperAction(robot_view, False, pc.gripper_close_duration),
            TCPMoveSequence(
                robot_view, self._tcp_to_jp_fn, pc.move_settle_time,
                is_holding_object=True, **common,
                move_segments=[
                    TCPMoveSegment(name="drag", start_pose=poses["grasp"],
                                   end_pose=poses["drag_end"], speed=pc.speed_slow),
                ],
            ),
            GripperAction(robot_view, True, pc.gripper_open_duration),
            TCPMoveSequence(
                robot_view, self._tcp_to_jp_fn, pc.move_settle_time, **common,
                move_segments=[
                    TCPMoveSegment(name="retreat", start_pose=poses["drag_end"],
                                   end_pose=poses["retreat"], speed=pc.speed_fast),
                ],
            ),
        ]


class PushPlannerPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    policy_cls: type = PushPlannerPolicy
    push_standoff: float = 0.08    # EE starts this far behind the object
    push_overshoot: float = 0.01   # EE ends this far past the goal
    approach_z: float = 0.04       # come in above contact height, drop to it
    retreat_z: float = 0.10        # small hoods leave little headroom


class PullPlannerPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    policy_cls: type = PullPlannerPolicy
    drag_lift: float = 0.02        # slight lift while dragging to break friction
    retreat_z: float = 0.15


# --------------------------------------------------------------------------- #
# Samplers: draw a per-episode displacement goal, return the matching task     #
# --------------------------------------------------------------------------- #
class ClutteredFumehoodPushSampler(ClutteredFumehoodPickSampler):
    """Push the object deeper into the hood or laterally along the bench."""

    TASK_CLS = PushFumehoodTask
    PUSH_SPAN = (0.06, 0.12)
    MARGIN = 0.10   # keep the goal this far off the hood walls

    def _draw_theta(self):
        th = super()._draw_theta()
        mode = np.random.choice(["deeper", "lateral"])
        if mode == "deeper":
            u = np.array([1.0, 0.0])
        else:
            u = np.array([0.0, float(np.random.choice([-1.0, 1.0]))])
        th["push_dir"] = [float(u[0]), float(u[1])]
        th["push_dist"] = float(np.random.uniform(*self.PUSH_SPAN))
        return th

    def _displacement_goal(self, start):
        th = getattr(self, "_theta", None) or {}
        u = np.asarray(th.get("push_dir", [1.0, 0.0]), dtype=float)
        dist = float(th.get("push_dist", 0.10))
        goal = np.asarray(start, dtype=float).copy()
        goal[0] += u[0] * dist
        goal[1] += u[1] * dist
        half_w, depth, _ = th.get("hood_dims", (0.30, 0.60, 0.60))
        goal[0] = float(np.clip(goal[0], TUBE_X0 + 0.06, TUBE_X0 + depth - self.MARGIN))
        goal[1] = float(np.clip(goal[1], -(half_w - self.MARGIN), half_w - self.MARGIN))
        return goal

    def _sample_task(self, env):
        super()._sample_task(env)   # runs placement, robot, referral machinery
        tc = self.config.task_config
        start = np.asarray(tc.pickup_obj_start_pose, dtype=float)
        goal = start.copy()
        goal[:3] = np.concatenate([self._displacement_goal(start[:3])[:2], [start[2]]])
        tc.pickup_obj_goal_pose = [float(v) for v in goal]
        th = getattr(self, "_theta", None)
        if isinstance(th, dict):
            th["displacement_goal"] = [float(v) for v in goal[:3]]
        return self.TASK_CLS(env, self.config)


class ClutteredFumehoodPullSampler(ClutteredFumehoodPushSampler):
    """Pull the object toward the mouth of the hood."""

    TASK_CLS = PullFumehoodTask
    PULL_SPAN = (0.08, 0.16)

    def _draw_theta(self):
        th = super()._draw_theta()
        th["push_dir"] = [-1.0, float(np.random.uniform(-0.2, 0.2))]
        th["push_dist"] = float(np.random.uniform(*self.PULL_SPAN))
        return th

    def _displacement_goal(self, start):
        goal = super()._displacement_goal(start)
        # never pull past the mouth lip
        goal[0] = float(max(goal[0], TUBE_X0 + 0.04))
        return goal
