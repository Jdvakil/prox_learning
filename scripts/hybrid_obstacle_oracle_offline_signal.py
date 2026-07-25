#!/usr/bin/env python3
"""Offline oracle-differential audit over the four development source trajectories.

Handoff step 7. No ACT, no residual controller in the loop, no policy: the four recorded
expert action sequences are replayed open-loop through the live environment, and at every
real timestep the per-frame parked counterfactual is rendered exactly as the live oracle
will render it.

Why replay rather than pose-setting
-----------------------------------
``env_states/articulations/panda`` stores the robot's qpos/qvel, but the trajectory H5
stores **no per-step pose for the pickup object** (``env_states/actors`` is empty, and
``obj_start``/``obj_end`` are the task's fixed start and goal). Setting only the robot
would therefore leave the target frozen at its rest pose for the whole episode, and the
handoff requires the exact robot, target *and* obstacle state. Deterministically
replaying the recorded joint commands from the verified initial state reconstructs all
three. Fidelity is not assumed: every reconstructed frame's re-rendered skin is compared
against the proximity recorded in the H5, and the agreement is reported.

Per frame this records the current head, the parked head, their differential, the frozen
controller's predicted response, sensor attribution, hazard distance, and the analytic
hazard-only potential-field teacher. Nothing is tuned from any of it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT / "scripts"), str(ROOT / "submodules" / "act")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import mujoco
from hybrid_safety_residual import (
    DEFAULT_ACTIVATION_DEPTH_M,
    DEFAULT_DEAD_PIXEL_BELOW_M,
    DEFAULT_DECAY,
    DEFAULT_EMA,
    DEFAULT_GAIN,
    DEFAULT_MAX_DEVIATION,
    ResidualSafetyController,
    extract_latest_proximity,
)
from parked_obstacle_reference import PerFrameParkedObstacleReference

#: safety_sweep.py:58 / :59 -- the analytic teacher's activation radius and the tolerance
#: within which a back-projected return counts as landing on a known scene surface.
D_ACT = 0.18
HIT_TOL = 0.025
#: meta.json d_max_input -- the depth beyond which the Safety-CVAE's own input is clipped.
#: Used only for the supplementary geometric direction audit, never for the committed
#: teacher, whose activation radius stays at D_ACT.
AUDIT_RADIUS = 0.5
FOVY = 45.0
KNOWN_SELF_RETURN = ("link5_front_sensor_1", "link5_front_sensor_2")


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


# --------------------------------------------------------------------------- #
# analytic potential-field teacher (safety_sweep.py:322-351), hazard-only form
# --------------------------------------------------------------------------- #
class HazardTeacher:
    """The committed analytic repulsion, restricted to returns landing on the hazard.

    ``safety_sweep`` sums, over every sensor whose closest valid return lies within
    ``D_ACT`` of a *known scene surface*, ``J_p(sensor)^T @ unit(sensor_pos - hit) *
    (1/r - 1/D_ACT)``. Restricting the surface set to the single hazard box yields the
    hazard's own contribution, which needs no global box inventory and so is exactly
    computable in the live manifest scene. That is the quantity the oracle differential
    is supposed to recover.
    """

    def __init__(self, model, sensor_order: list[str], hazard_box, arm_dofadr,
                 activation_radius: float = D_ACT) -> None:
        self.activation_radius = float(activation_radius)
        self.model = model
        self.sensor_order = sensor_order
        self.hazard_box = hazard_box            # (center(3), half(3)) or None
        self.arm_dofadr = arm_dofadr
        self.cam_ids = {name: model.camera(f"robot_0/{name}").id for name in sensor_order}
        self.focal = 4.0 / np.tan(np.deg2rad(FOVY / 2))
        self.principal = 3.5
        self.uu, self.vv = np.meshgrid(np.arange(8), np.arange(8))
        self._jacp = np.zeros((3, model.nv))

    @staticmethod
    def _box_surface_distance(point, center, half) -> float:
        q = np.abs(point - center) - half
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(max(q[0], q[1], q[2]), 0.0)
        return abs(outside + inside)

    def __call__(self, data, skin: np.ndarray) -> dict[str, Any]:
        if self.hazard_box is None:
            return {"tau": np.zeros(7), "contributing_sensors": [], "minimum_hazard_return_m": None}
        center, half = self.hazard_box
        radius_limit = self.activation_radius
        tau = np.zeros(self.model.nv)
        contributing, closest = [], np.inf
        for index, name in enumerate(self.sensor_order):
            patch = skin[index]
            valid = patch >= DEFAULT_DEAD_PIXEL_BELOW_M
            if not valid.any():
                continue
            flat = int(np.argmin(np.where(valid, patch, np.inf)))
            depth = float(patch.flat[flat])
            if depth >= radius_limit:
                continue
            cam_id = self.cam_ids[name]
            rotation = data.cam_xmat[cam_id].reshape(3, 3)
            position = data.cam_xpos[cam_id]
            u, v = self.uu.flat[flat], self.vv.flat[flat]
            # MuJoCo camera frame is GL: +x right, +y up, -z forward.
            in_camera = np.array([(u - self.principal) * depth / self.focal,
                                  -(v - self.principal) * depth / self.focal,
                                  -depth])
            world = rotation @ in_camera + position
            if self._box_surface_distance(world, center, half) > HIT_TOL:
                continue                      # not a hazard return
            radius = float(np.linalg.norm(position - world))
            if radius < 1e-4 or radius >= radius_limit:
                continue
            closest = min(closest, radius)
            direction = (position - world) / radius
            weight = 1.0 / radius - 1.0 / radius_limit
            mujoco.mj_jac(self.model, data, self._jacp, None, position,
                          int(self.model.cam_bodyid[cam_id]))
            tau += self._jacp.T @ (direction * weight)
            contributing.append({"sensor": name, "range_m": radius, "weight": weight})
        return {
            "tau": tau[self.arm_dofadr],
            "contributing_sensors": contributing,
            "minimum_hazard_return_m": None if not np.isfinite(closest) else float(closest),
        }


# --------------------------------------------------------------------------- #
class ExpertReplayProbe:
    """Open-loop replay of the recorded joint commands, with a counterfactual per frame."""

    def __init__(self, actions, sensor_order, head, hazard_box, out_frames,
                 recorded_proximity) -> None:
        self.actions = actions
        self.recorded_proximity = recorded_proximity
        self.sensor_order = sensor_order
        self.head = head
        self.hazard_box = hazard_box
        self.frames = out_frames
        self.oracle: PerFrameParkedObstacleReference | None = None
        self.teacher: HazardTeacher | None = None
        self.controller = ResidualSafetyController(
            label_scale=head.scale, dt=0.066, gain=DEFAULT_GAIN, decay=DEFAULT_DECAY,
            ema=DEFAULT_EMA, max_deviation=DEFAULT_MAX_DEVIATION)
        self.step = 0
        self.task = None
        self.mj_forward_shift: dict[str, Any] | None = None

    def bind(self, task) -> None:
        self.task = task
        self.oracle = PerFrameParkedObstacleReference(task, self.sensor_order)
        model = task.env.current_model
        arm_dofadr = [model.joint(f"robot_0/fr3_joint{i}").dofadr[0] for i in range(1, 8)]
        self.teacher = HazardTeacher(model, self.sensor_order, self.hazard_box, arm_dofadr)
        # Supplementary audit at the head's own input radius: on rows where the hazard
        # never becomes a sensor's closest return inside D_ACT the committed teacher
        # cannot fire at all, yet the counterfactual still isolates a hazard direction.
        self.audit_teacher = HazardTeacher(model, self.sensor_order, self.hazard_box,
                                           arm_dofadr, activation_radius=AUDIT_RADIUS)

    def observe(self, observation) -> dict[str, Any]:
        assert self.oracle is not None and self.teacher is not None
        if self.step == 5:
            # Measured mid-episode: immediately after task.reset() the scene is already
            # forward-consistent, so step 0 would understate it.
            self.mj_forward_shift = self.oracle.measure_mj_forward_pose_shift()
        data = self.task.env.current_data
        obs = observation[0] if isinstance(observation, (list, tuple)) else observation
        current = extract_latest_proximity(obs, self.sensor_order)
        rerender = self.oracle.render_current_skin()
        parked, neutrality = self.oracle.parked_skin()

        current_head = np.asarray(self.head(current), dtype=np.float64)
        rerender_head = np.asarray(self.head(rerender), dtype=np.float64)
        parked_head = np.asarray(self.head(parked), dtype=np.float64)
        # Two candidate differentials:
        #   observation-paired : head(observation skin) - head(parked at decision state)
        #   pose-consistent    : head(re-render at decision state) - head(parked, same state)
        # They differ only by the sub-step pose lag between the observation's last
        # proximity render and the decision state. Both are logged so the size of that
        # lag decides which one the live oracle uses.
        differential = current_head - parked_head
        pose_consistent = rerender_head - parked_head
        residual = self.controller.step(rerender_head.astype(np.float32),
                                        parked_head.astype(np.float32))

        teacher_current = self.teacher(data, rerender)
        teacher_parked = self.teacher(data, parked)
        audit_current = self.audit_teacher(data, rerender)

        valid = current >= DEFAULT_DEAD_PIXEL_BELOW_M
        minima = np.where(valid.any(axis=(1, 2)),
                          np.where(valid, current, np.inf).min(axis=(1, 2)), np.inf)
        active = [self.sensor_order[i] for i in range(40)
                  if minima[i] < DEFAULT_ACTIVATION_DEPTH_M]
        # sensors whose patch actually changes when the hazard is parked
        # rerender-vs-parked: comparing the observation against the parked render would
        # count every sensor that merely moved during the sub-step lag.
        changed = [self.sensor_order[i] for i in range(40)
                   if not np.array_equal(rerender[i], parked[i])]

        record = {
            "step": self.step,
            "current_head": current_head.tolist(),
            "parked_head": parked_head.tolist(),
            "current_head_norm": float(np.linalg.norm(current_head)),
            "parked_head_norm": float(np.linalg.norm(parked_head)),
            "differential": differential.tolist(),
            "differential_norm": float(np.linalg.norm(differential)),
            "differential_max_abs": float(np.max(np.abs(differential))),
            "rerender_head": rerender_head.tolist(),
            "rerender_head_norm": float(np.linalg.norm(rerender_head)),
            "pose_consistent_differential": pose_consistent.tolist(),
            "pose_consistent_differential_norm": float(np.linalg.norm(pose_consistent)),
            "substep_lag_head_delta_norm": float(np.linalg.norm(rerender_head - current_head)),
            "substep_lag_contamination_ratio": (
                float(np.linalg.norm(rerender_head - current_head)
                      / np.linalg.norm(differential))
                if np.linalg.norm(differential) > 1e-12 else None),
            "predicted_filtered_norm": float(np.linalg.norm(residual.filtered_safety_dq)),
            "predicted_correction": residual.correction.tolist(),
            "predicted_correction_norm": float(np.linalg.norm(residual.correction)),
            "predicted_saturated": bool(
                np.max(np.abs(residual.correction)) >= DEFAULT_MAX_DEVIATION - 1e-9),
            "active_sensors": active,
            "active_sensor_count": len(active),
            "hazard_changed_sensors": changed,
            "hazard_changed_sensor_count": len(changed),
            "known_self_return_active": sorted(set(active) & set(KNOWN_SELF_RETURN)),
            "known_self_return_changed": sorted(set(changed) & set(KNOWN_SELF_RETURN)),
            "minimum_active_depth_m": float(minima.min()) if np.isfinite(minima).any() else None,
            "teacher_hazard_only": teacher_current["tau"].tolist(),
            "teacher_hazard_only_norm": float(np.linalg.norm(teacher_current["tau"])),
            "teacher_parked_hazard_only_norm": float(np.linalg.norm(teacher_parked["tau"])),
            "teacher_contributing_sensors": [c["sensor"] for c in
                                             teacher_current["contributing_sensors"]],
            "minimum_hazard_return_m": teacher_current["minimum_hazard_return_m"],
            "teacher_active": bool(teacher_current["contributing_sensors"]),
            "audit_teacher_norm": float(np.linalg.norm(audit_current["tau"])),
            "audit_teacher_active": bool(audit_current["contributing_sensors"]),
            "audit_minimum_hazard_return_m": audit_current["minimum_hazard_return_m"],
            "skins_bit_identical": bool(np.array_equal(rerender, parked)),
            "rerender_matches_observation": bool(np.array_equal(rerender, current)),
            "rerender_max_abs_delta": float(np.max(np.abs(
                rerender.astype(np.float64) - current.astype(np.float64)))),
            "state_neutral": neutrality["neutral"],
        }
        # Fidelity of the reconstruction: the replayed frame's own skin against the skin
        # the collection recorded at this timestep. Equality proves the replay put the
        # robot, target and obstacle back where they were.
        if self.step < len(self.recorded_proximity):
            reference = np.asarray(self.recorded_proximity[self.step], dtype=np.float32)
            delta = np.abs(current.astype(np.float64) - reference.astype(np.float64))
            record["matches_recorded_proximity"] = bool(np.array_equal(current, reference))
            record["recorded_proximity_max_abs_delta"] = float(delta.max())
            # Max-abs over 2560 pixels is dominated by single pixels straddling a depth
            # discontinuity, where an arbitrarily small pose change flips a return between
            # a near surface and the far background. The pixel-agreement fraction is the
            # meaningful fidelity statistic.
            record["recorded_proximity_pixel_agreement"] = float(np.mean(delta <= 1e-3))
            record["recorded_proximity_median_abs_delta"] = float(np.median(delta))
        else:
            record["matches_recorded_proximity"] = None
            record["recorded_proximity_max_abs_delta"] = None
            record["recorded_proximity_pixel_agreement"] = None
            record["recorded_proximity_median_abs_delta"] = None
        # The pose-consistent pair is the primary differential; see the evaluator's
        # PoseConsistentProximity for why the observation-paired form is contaminated.
        teacher_norm = np.linalg.norm(teacher_current["tau"])
        diff_norm = np.linalg.norm(pose_consistent)
        record["cosine_head_vs_teacher"] = (
            float(np.dot(pose_consistent, teacher_current["tau"]) / (teacher_norm * diff_norm))
            if teacher_norm > 1e-12 and diff_norm > 1e-12 else None)
        audit_norm = np.linalg.norm(audit_current["tau"])
        record["cosine_head_vs_audit_teacher"] = (
            float(np.dot(pose_consistent, audit_current["tau"]) / (audit_norm * diff_norm))
            if audit_norm > 1e-12 and diff_norm > 1e-12 else None)
        self.frames.append(record)
        return record

    def action(self):
        index = min(self.step, len(self.actions) - 1)
        arm, gripper = self.actions[index]
        return {"arm": np.asarray(arm, dtype=np.float32),
                "gripper": np.asarray(gripper, dtype=np.float32)}


# --------------------------------------------------------------------------- #
def load_recorded(h5_path: Path, sensor_order: list[str]):
    import h5py
    with h5py.File(h5_path, "r") as handle:
        key = next(k for k in handle if k.startswith("traj"))
        group = handle[key]
        # The final recorded step carries an empty action ({}), because the episode
        # terminated before another command was issued. Replay stops at the last real one.
        actions = []
        for row in group["actions/joint_pos"][:]:
            payload = json.loads(bytes(row).split(b"\x00")[0].decode())
            if "arm" not in payload:
                break
            actions.append((payload["arm"], payload["gripper"]))
        streams = []
        for name in sensor_order:
            values = np.asarray(group[f"obs/proximity/{name}"][()], dtype=np.float32)
            streams.append(values[:, -1] if values.ndim == 4 else values)
        proximity = np.stack(streams, axis=1)
        scene = json.loads(group["obs_scene"][()])
        success = bool(group["success"][:][-1])
    return actions, proximity, scene, success


def replay_row(row, collection_manifest, sensor_order, head, max_steps=None):
    """One open-loop expert replay with a counterfactual render at every timestep."""
    from eval_act_obstacle_manifest_safety import initial_state_hash
    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
        FrankaSkinHybridObstacleManifestV2Config,
    )
    from molmo_spaces.data_generation.episode_manifest import install_row_seed_contract
    from molmo_spaces.data_generation.manifest_runner import (
        extract_row_observations,
        reset_episode_scoped_sampler_state,
    )

    coll_row = next(r for r in collection_manifest["rows"]
                    if r["episode_id"] == row["episode_id"])
    run = Path(row["_run_dir"])
    actions, recorded, scene, recorded_success = load_recorded(
        run / "rows" / row["episode_id"] / "trajectory.h5", sensor_order)
    horizon = len(actions) if max_steps is None else min(max_steps, len(actions))

    params = scene["scene_params"]
    hazard_box = None
    if params.get("protrusion_present"):
        hazard_box = (np.asarray(params["protr_center"], dtype=float),
                      np.asarray(params["protr_half"], dtype=float))

    config = FrankaSkinHybridObstacleManifestV2Config()
    config.task_horizon = horizon
    sampler = config.task_sampler_config.task_sampler_class(config)
    reset_episode_scoped_sampler_state(sampler)
    install_row_seed_contract(coll_row, int(row["accepted_retry_index"]), task_sampler=sampler)
    sampler.set_manifest_row(coll_row, int(row["accepted_retry_index"]))
    task = sampler.sample_task(house_index=coll_row["scene_template_house_index"])
    if task is None:
        raise SystemExit(f"task sampling returned None for {row['episode_id']}")

    observations = extract_row_observations(task, coll_row)
    replay_hash = initial_state_hash(observations, coll_row)
    if replay_hash != row["initial_state_sha256"]:
        raise SystemExit(f"REPLAY INITIAL-STATE MISMATCH for cand {row['candidate_index']}")

    frames: list[dict[str, Any]] = []
    probe = ExpertReplayProbe(actions, sensor_order, head, hazard_box, frames, recorded)
    observation, _ = task.reset()
    probe.bind(task)
    while probe.step < horizon and not task.is_done():
        probe.observe(observation)
        observation, _r, _t, _tr, _i = task.step([probe.action()])
        probe.step += 1

    return {
        "frames": frames,
        "recorded_timesteps": len(recorded),
        "replay_initial_state_sha256": replay_hash,
        "recorded_success": recorded_success,
        "hazard_box": None if hazard_box is None else
                      [hazard_box[0].tolist(), hazard_box[1].tolist()],
        "scene_params": {k: v for k, v in params.items() if k != "obstacle_aabbs"},
        "oracle_provenance": probe.oracle.provenance() if probe.oracle else None,
        "mj_forward_pose_shift": probe.mj_forward_shift,
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--development-manifest", required=True, type=Path)
    ap.add_argument("--collection-manifest", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke only")
    ap.add_argument("--only-candidate", type=int, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch
    from train_safety_cvae import SafetyHead

    dev = json.loads(args.development_manifest.read_text())
    coll = json.loads(args.collection_manifest.read_text())
    stack = json.loads(args.stack.read_text())
    sensor_order = list(stack["sensor_contract"]["ordered_names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = SafetyHead.load(str(args.safety_dir), device=device)

    run_dir = Path(dev["source_run_dir"])
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    rows_out = []
    for row in sorted(dev["rows"], key=lambda r: r["candidate_index"]):
        if args.only_candidate is not None and row["candidate_index"] != args.only_candidate:
            continue
        row = {**row, "_run_dir": str(run_dir)}
        print(f"\n=== cand {row['candidate_index']} "
              f"({'hazard present' if row['hazard_present'] else 'hazard absent'}) ===",
              flush=True)
        result = replay_row(row, coll, sensor_order, head, args.max_steps)
        frames = result["frames"]
        fidelity = [f["matches_recorded_proximity"] for f in frames
                    if f["matches_recorded_proximity"] is not None]
        fidelity_delta = [f["recorded_proximity_max_abs_delta"] for f in frames
                          if f["recorded_proximity_max_abs_delta"] is not None]

        active = [f for f in frames if f["teacher_active"]]
        cosines = [f["cosine_head_vs_teacher"] for f in active
                   if f["cosine_head_vs_teacher"] is not None]
        audit_active = [f for f in frames if f["audit_teacher_active"]]
        audit_cosines = [f["cosine_head_vs_audit_teacher"] for f in audit_active
                         if f["cosine_head_vs_audit_teacher"] is not None]
        # Any frame whose differential is nonzero must have at least one sensor whose
        # patch the hazard actually changes: that is what rules out static geometry.
        attributable = [f for f in frames if f["differential_norm"] > 1e-9
                        or f["pose_consistent_differential_norm"] > 1e-9]
        unattributable = [f["step"] for f in frames
                          if f["pose_consistent_differential_norm"] > 1e-9
                          and f["hazard_changed_sensor_count"] == 0]
        changed_counts = [f["hazard_changed_sensor_count"] for f in frames]
        nonzero = [f for f in frames if f["differential_max_abs"] > 1e-7]
        known_share = (sum(1 for f in nonzero if f["known_self_return_changed"]) / len(nonzero)
                       if nonzero else 0.0)

        rows_out.append({
            "candidate_index": row["candidate_index"],
            "episode_id": row["episode_id"],
            "hazard_present": bool(row["hazard_present"]),
            "replayed_timesteps": len(frames),
            "recorded_timesteps": result["recorded_timesteps"],
            "replay_initial_state_sha256": result["replay_initial_state_sha256"],
            "hazard_box": result["hazard_box"],
            "scene_params": result["scene_params"],
            "oracle_provenance": result["oracle_provenance"],
            "mj_forward_pose_shift": result["mj_forward_pose_shift"],
            "state_neutral_every_frame": all(f["state_neutral"] for f in frames),
            "rerender_matches_observation_every_frame":
                all(f["rerender_matches_observation"] for f in frames),
            "rerender_max_abs_delta": max((f["rerender_max_abs_delta"] for f in frames),
                                          default=0.0),
            "reconstruction_fidelity": {
                "frames_compared": len(fidelity),
                "frames_bit_identical_to_recording": int(sum(fidelity)),
                "fraction_bit_identical": float(np.mean(fidelity)) if fidelity else None,
                "max_abs_delta_vs_recording": max(fidelity_delta, default=None),
                "mean_pixel_agreement_vs_recording": float(np.mean(
                    [f["recorded_proximity_pixel_agreement"] for f in frames
                     if f["recorded_proximity_pixel_agreement"] is not None])) if fidelity else None,
                "median_abs_delta_vs_recording": float(np.median(
                    [f["recorded_proximity_median_abs_delta"] for f in frames
                     if f["recorded_proximity_median_abs_delta"] is not None])) if fidelity else None,
                "first_divergent_step": next((i for i, m in enumerate(fidelity) if not m), None),
            },
            "current_head_norm": {
                "min": min(f["current_head_norm"] for f in frames),
                "max": max(f["current_head_norm"] for f in frames),
                "mean": float(np.mean([f["current_head_norm"] for f in frames]))},
            "parked_head_norm": {
                "min": min(f["parked_head_norm"] for f in frames),
                "max": max(f["parked_head_norm"] for f in frames),
                "mean": float(np.mean([f["parked_head_norm"] for f in frames]))},
            "differential_norm": {
                "min": min(f["differential_norm"] for f in frames),
                "max": max(f["differential_norm"] for f in frames),
                "mean": float(np.mean([f["differential_norm"] for f in frames])),
                "median": float(np.median([f["differential_norm"] for f in frames]))},
            "differential_max_abs": max(f["differential_max_abs"] for f in frames),
            "pose_consistent_differential_norm": {
                "max": max(f["pose_consistent_differential_norm"] for f in frames),
                "mean": float(np.mean([f["pose_consistent_differential_norm"] for f in frames])),
                "median": float(np.median([f["pose_consistent_differential_norm"]
                                           for f in frames]))},
            "substep_lag_head_delta_norm": {
                "max": max(f["substep_lag_head_delta_norm"] for f in frames),
                "mean": float(np.mean([f["substep_lag_head_delta_norm"] for f in frames])),
                "median": float(np.median([f["substep_lag_head_delta_norm"] for f in frames]))},
            "substep_lag_contamination_ratio_median": float(np.median(
                [f["substep_lag_contamination_ratio"] for f in frames
                 if f["substep_lag_contamination_ratio"] is not None])),
            "differential_per_joint_abs_max":
                np.abs(np.asarray([f["differential"] for f in frames])).max(axis=0).tolist(),
            "frames_with_nonzero_differential": len(nonzero),
            "skins_bit_identical_every_frame": all(f["skins_bit_identical"] for f in frames),
            "hazard_changed_sensor_count": {
                "min": min(changed_counts), "max": max(changed_counts),
                "mean": float(np.mean(changed_counts))},
            "teacher_active_frames": len(active),
            "teacher_hazard_only_norm": {
                "max": max((f["teacher_hazard_only_norm"] for f in frames), default=0.0),
                "mean_on_active": float(np.mean([f["teacher_hazard_only_norm"]
                                                 for f in active])) if active else 0.0},
            "minimum_hazard_return_m": min(
                (f["minimum_hazard_return_m"] for f in frames
                 if f["minimum_hazard_return_m"] is not None), default=None),
            "cosine_head_vs_teacher": {
                "n": len(cosines),
                "median": float(np.median(cosines)) if cosines else None,
                "mean": float(np.mean(cosines)) if cosines else None,
                "fraction_positive": (float(np.mean([c > 0 for c in cosines]))
                                      if cosines else None),
                "values": cosines},
            "geometric_direction_audit": {
                "activation_radius_m": AUDIT_RADIUS,
                "why": ("the committed teacher only fires when a hazard return is a sensor's "
                        "closest return inside D_ACT=0.18 m; at the head's own input radius "
                        "the hazard direction is still well defined, so direction can be "
                        "audited on rows the committed teacher never activates on"),
                "active_frames": len(audit_active),
                "n": len(audit_cosines),
                "median": float(np.median(audit_cosines)) if audit_cosines else None,
                "mean": float(np.mean(audit_cosines)) if audit_cosines else None,
                "fraction_positive": (float(np.mean([c > 0 for c in audit_cosines]))
                                      if audit_cosines else None),
                "values": audit_cosines},
            "static_geometry_attribution": {
                "rule": ("every frame with a nonzero pose-consistent differential must have "
                         "at least one sensor patch the hazard changes"),
                "frames_with_nonzero_differential": len(attributable),
                "frames_nonzero_without_a_hazard_changed_sensor": len(unattributable),
                "offending_steps": unattributable[:20],
                "passed": not unattributable},
            "known_self_return_share_of_active_differential": known_share,
            "predicted_correction_norm": {
                "max": max(f["predicted_correction_norm"] for f in frames),
                "mean": float(np.mean([f["predicted_correction_norm"] for f in frames])),
                "final": frames[-1]["predicted_correction_norm"]},
            "predicted_saturation_fraction":
                float(np.mean([f["predicted_saturated"] for f in frames])),
            "frames": frames,
        })
        latest = rows_out[-1]
        print(f"  T={latest['replayed_timesteps']}  "
              f"diff|max={latest['differential_norm']['max']:.4f}  "
              f"teacher-active={latest['teacher_active_frames']}  "
              f"cos-med={latest['cosine_head_vs_teacher']['median']}  "
              f"neutral={latest['state_neutral_every_frame']}", flush=True)

    report = {
        "schema": "hybrid_obstacle_oracle_offline_signal_v1",
        "reference_id": "ORACLE_PARKED_REFERENCE_V1",
        "privileged": True,
        "deployable": False,
        "method": ("open-loop replay of the recorded expert joint commands from the verified "
                   "initial state, with a per-frame parked counterfactual render; no ACT, no "
                   "residual controller in the loop, no policy"),
        "frozen_constants": {"gain": DEFAULT_GAIN, "decay": DEFAULT_DECAY, "ema": DEFAULT_EMA,
                             "max_dev": DEFAULT_MAX_DEVIATION, "dt": 0.066,
                             "label_scale": float(head.scale)},
        "teacher": {
            "form": "analytic potential-field repulsion restricted to hazard-box returns",
            "source": "scripts/safety_sweep.py:322-351",
            "d_act_m": D_ACT, "hit_tolerance_m": HIT_TOL,
            "why_hazard_only": ("The committed teacher needs a global inventory of scene "
                                "surfaces to reject self-hits. The manifest scene poses its "
                                "enclosure per episode, so that inventory is not directly "
                                "available; restricting the surface set to the hazard box "
                                "removes the need for it and yields exactly the hazard's own "
                                "contribution, which is the quantity the oracle differential "
                                "should recover."),
        },
        "development_manifest_sha256": dev["manifest_sha256"],
        "max_steps": args.max_steps,
        "tuning_performed": False,
        "rows": rows_out,
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
