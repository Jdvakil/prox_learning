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
from molmo_spaces.utils.linalg_utils import transform_to_twist, twist_to_transform
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

    SUCC_FRACTION = 0.6    # fraction of the commanded displacement that counts
    MIN_PROGRESS = 0.04    # ...but never call a sub-4cm nudge a success
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
                "success": bool(progress >= max(self.MIN_PROGRESS,
                                                self.SUCC_FRACTION * float(
                                                    np.linalg.norm(goal[:2] - np.asarray(
                                                        tc.pickup_obj_start_pose[:2], dtype=float))))
                                and dz <= self.Z_GUARD),
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

    # Searched in order; the first fully feasible combination wins, so the
    # leading entries are the ones that make the nicest demonstration.
    MIN_PUSH = 0.05                      # never command a shorter push than this
    _STANDOFFS = (0.055, 0.040, 0.028)   # how far behind the object the EE starts
    _LIFTS = (0.030, 0.0)                # approach height above contact
    _DIST_SCALES = (1.0, 0.7)            # fraction of the drawn push distance

    def _direction_candidates(self, u0: np.ndarray) -> list[np.ndarray]:
        return [u0, np.array([0.0, 1.0]), np.array([0.0, -1.0]), np.array([-1.0, 0.0])]

    PATH_SAMPLES = 4   # interpolation points checked between consecutive waypoints

    @staticmethod
    def _interpolate(start: np.ndarray, end: np.ndarray, n: int) -> list[np.ndarray]:
        """The same twist interpolation TCPMoveSequence tracks at runtime
        (get_current_target_pose): start @ twist(lin*t, ang*t)."""
        lin_vel, ang_vel = transform_to_twist(np.linalg.inv(start) @ end)
        return [start @ twist_to_transform(lin_vel * t, ang_vel * t)
                for t in np.linspace(0.0, 1.0, n + 2)[1:-1]]

    def _feasible_mask(self, poses: np.ndarray) -> np.ndarray:
        """Batched reachability over a stack of (4,4) poses, chunked to the
        solver's batch size (check_feasible_ik asserts on anything larger)."""
        cap = self.policy_config.grasp_feasibility_batch_size
        out = []
        for i in range(0, len(poses), cap):
            out.append(np.atleast_1d(self.check_feasible_ik(poses[i:i + cap])))
        return np.concatenate(out) if out else np.zeros(0, dtype=bool)

    def _compute_target_poses(self) -> dict[str, np.ndarray]:
        pc = self.policy_config
        tc = self.config.task_config
        anchor = _grasp_anchor(self)
        start = np.asarray(tc.pickup_obj_start_pose[:3], dtype=float)
        goal = np.asarray(tc.pickup_obj_goal_pose[:3], dtype=float)
        d = goal[:2] - start[:2]
        drawn = float(np.linalg.norm(d))
        u0 = d / max(drawn, 1e-6)

        # A fixed standoff does not survive here. check_feasible_ik is pure
        # reachability seeded at the current arm configuration, and the pick
        # planner only ever offsets the validated grasp pose *vertically*
        # (pregrasp = grasp + pregrasp_z_offset). Translating that same pose
        # horizontally, deep inside a hood, leaves the reachable set almost
        # every time — so search direction x standoff x lift x distance and
        # take the first combination whose whole waypoint set is reachable.
        gripper_mg_id = self.task.env.current_robot.robot_view.get_gripper_movegroup_ids()[0]
        start_ee = self.task.env.current_robot.robot_view.get_move_group(
            gripper_mg_id).leaf_frame_to_world

        combos, stacked, per_combo = [], [], 0
        for u in self._direction_candidates(u0):
            for standoff in self._STANDOFFS:
                for lift in self._LIFTS:
                    for scale in self._DIST_SCALES:
                        dist = max(drawn * scale, self.MIN_PUSH)
                        back = -u * standoff
                        fwd = u * (dist + pc.push_overshoot)
                        approach = _translated(anchor, [back[0], back[1], lift])
                        contact = _translated(anchor, [back[0], back[1], 0.0])
                        push_end = _translated(anchor, [fwd[0], fwd[1], 0.0])
                        # Back off the way we came rather than straight up: a
                        # vertical retreat keeps the wrist at full radius, and
                        # the arm is already near its envelope at this depth.
                        retreat = _translated(push_end, [-u[0] * 0.05, -u[1] * 0.05,
                                                         pc.retreat_z])
                        combos.append((u, dist, approach, contact, push_end, retreat))
                        # Endpoint reachability is not enough: TCPMoveSequence
                        # re-solves IK against an interpolated target every step
                        # and aborts the segment once the tracking error passes
                        # tcp_pos_err_threshold. Preflight showed exactly that -
                        # 27 approaches started, 13 reached grasp, 3 reached the
                        # push - so the whole path each segment sweeps has to be
                        # reachable, not just the poses at its ends.
                        waypoints = [start_ee, approach, contact, push_end, retreat]
                        path = [approach, contact, push_end, retreat]
                        for a, b in zip(waypoints[:-1], waypoints[1:]):
                            path += self._interpolate(a, b, self.PATH_SAMPLES)
                        per_combo = len(path)
                        stacked += path

        ok = self._feasible_mask(np.stack(stacked)).reshape(len(combos), per_combo).all(axis=1)
        # Among feasible combinations, take the one whose furthest waypoint sits
        # closest to the robot base: the arm is near its envelope in here, and
        # the least-stretched option is the one most likely to track cleanly.
        base_xy = self.task.env.current_robot.robot_view.base.pose[:2, 3]
        reach = np.array([max(float(np.linalg.norm(p[:2, 3] - base_xy)) for p in c[2:])
                          for c in combos])
        order = np.lexsort((reach, ~ok))
        if not ok.any():
            raise ValueError(
                f"no reachable push waypoints among {len(combos)} candidates "
                f"(object at {np.round(start, 3)})"
            )

        u, dist, approach, contact, push_end, retreat = combos[int(order[0])]
        # Judge success on the direction and distance actually commanded.
        new_goal = list(tc.pickup_obj_goal_pose)
        new_goal[0] = float(start[0] + u[0] * dist)
        new_goal[1] = float(start[1] + u[1] * dist)
        tc.pickup_obj_goal_pose = new_goal
        log.info("[Push] dir=%s dist=%.3f standoff=%.3f path-feasible=%d/%d",
                 np.round(u, 2), dist,
                 float(np.linalg.norm(contact[:3, 3] - anchor[:3, 3])),
                 int(ok.sum()), len(combos))

        # The segment that meets the object is named "grasp": GraspPoseSensor
        # reads target_poses["grasp"] unguarded, and target_poses is keyed by
        # move-segment name, so any policy without one crashes the sensor.
        return {"approach": approach, "grasp": contact,
                "push_end": push_end, "retreat": retreat}

    def get_all_phases(self):
        phases = super().get_all_phases()
        for name in ("approach", "push", "push_end", "drag", "drag_end"):
            if name not in phases:
                phases[name] = max(phases.values()) + 1
        return phases

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
                    TCPMoveSegment(name="grasp", start_pose=poses["approach"],
                                   end_pose=poses["grasp"], speed=pc.speed_slow),
                ],
            ),
            TCPMoveSequence(
                robot_view, self._tcp_to_jp_fn, pc.move_settle_time, **common,
                move_segments=[
                    TCPMoveSegment(name="push", start_pose=poses["grasp"],
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
        gripper_mg_id = self.task.env.current_robot.robot_view.get_gripper_movegroup_ids()[0]
        start_ee = self.task.env.current_robot.robot_view.get_move_group(
            gripper_mg_id).leaf_frame_to_world

        for scale in (1.0, 0.75, 0.5):
            poses = {
                "pregrasp": _translated(anchor, [0.0, 0.0, pc.pregrasp_z_offset]),
                "grasp": anchor,
                "drag_end": _translated(anchor, [d[0] * scale, d[1] * scale, pc.drag_lift]),
            }
            poses["retreat"] = _translated(poses["drag_end"], [0.0, 0.0, pc.retreat_z])
            chain = [start_ee, poses["pregrasp"], poses["grasp"],
                     poses["drag_end"], poses["retreat"]]
            path = list(chain[1:])
            for a, b in zip(chain[:-1], chain[1:]):
                path += PushPlannerPolicy._interpolate(a, b, 4)
            if PushPlannerPolicy._feasible_mask(self, np.stack(path)).all():
                new_goal = list(tc.pickup_obj_goal_pose)
                new_goal[0] = float(start[0] + d[0] * scale)
                new_goal[1] = float(start[1] + d[1] * scale)
                tc.pickup_obj_goal_pose = new_goal
                return poses
        raise ValueError("IK failed for drag waypoints at every distance")

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
    retreat_z: float = 0.04        # small hoods leave little headroom


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
