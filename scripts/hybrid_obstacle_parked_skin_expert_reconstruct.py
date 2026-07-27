#!/usr/bin/env python3
"""Reconstruct the EXPERT_RECONSTRUCTED distribution with paired current/parked fields.

Handoff step 8. For one canonical trajectory: restore each recorded decision state by
deterministic open-loop replay of the recorded expert joint commands from the verified
initial state, take the current field at that state, generate the parked field through the
validated state-neutral oracle, restore the live state exactly, and retain every real
timestep.

ACT is never run -- the nominal-action field is filled from the recorded expert command,
which is what actually drove this trajectory. The source H5 is opened read-only and never
altered.

Replay rather than pose-setting because the trajectory H5 stores no per-step pose for the
pickup object, so setting only the robot would leave the target frozen at its rest pose.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT / "scripts"), str(ROOT / "submodules" / "act")):
    if extra not in sys.path:
        sys.path.insert(0, extra)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--episode-id", required=True)
    ap.add_argument("--collection-manifest", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import h5py
    import torch
    from eval_act_obstacle_manifest_safety import initial_state_hash
    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
        FrankaSkinHybridObstacleManifestV2Config,
    )
    from molmo_spaces.data_generation.episode_manifest import install_row_seed_contract
    from molmo_spaces.data_generation.manifest_runner import (
        extract_row_observations,
        reset_episode_scoped_sampler_state,
    )
    from parked_obstacle_reference import PerFrameParkedObstacleReference
    from parked_skin_retention import TrajectoryRetention, array_hash
    from train_safety_cvae import SafetyHead

    manifest = json.loads(args.manifest.read_text())
    entry = next(e for e in manifest["entries"]
                 if e["episode_id"] == args.episode_id
                 and e["distribution"] == "EXPERT_RECONSTRUCTED")
    collection = json.loads(args.collection_manifest.read_text())
    coll_row = next(r for r in collection["rows"] if r["episode_id"] == args.episode_id)
    if coll_row["row_sha256"] != entry["manifest_row_id"]:
        raise SystemExit("row hash disagrees with the frozen dataset manifest")

    stack = json.loads(args.stack.read_text())
    sensor_order = list(stack["sensor_contract"]["ordered_names"])
    head = SafetyHead.load(str(args.safety_dir),
                           device="cuda" if torch.cuda.is_available() else "cpu")

    source = args.run_dir / "rows" / args.episode_id / "trajectory.h5"
    with h5py.File(source, "r") as handle:                      # read-only, never altered
        key = next(k for k in handle if k.startswith("traj"))
        actions = []
        for row in handle[f"{key}/actions/joint_pos"][:]:
            payload = json.loads(bytes(row).split(b"\x00")[0].decode())
            if "arm" not in payload:
                break                    # the terminal step records an empty action
            actions.append((payload["arm"], payload["gripper"]))

    config = FrankaSkinHybridObstacleManifestV2Config()
    config.task_horizon = len(actions)
    sampler = config.task_sampler_config.task_sampler_class(config)
    reset_episode_scoped_sampler_state(sampler)
    retry = int(coll_row.get("accepted_retry_index", 0))
    install_row_seed_contract(coll_row, retry, task_sampler=sampler)
    sampler.set_manifest_row(coll_row, retry)
    task = sampler.sample_task(house_index=coll_row["scene_template_house_index"])
    if task is None:
        raise SystemExit("task sampling returned None")

    observations = extract_row_observations(task, coll_row)
    replay_hash = initial_state_hash(observations, coll_row)
    offsamples = int(task.env.current_model.vis.quality.offsamples)
    if offsamples != 4:
        raise SystemExit(f"offsamples is {offsamples}, expected 4")

    identity = {k: entry[k] for k in
                ("dataset_version", "distribution", "partition", "trajectory_id",
                 "manifest_row_id", "episode_id", "candidate_index", "hazard_present",
                 "policy_condition", "source_h5_sha256")}
    retention = TrajectoryRetention(
        identity=identity,
        provenance={"replayed_initial_state_sha256": replay_hash,
                    "offsamples": offsamples, "accepted_retry_index": retry,
                    "reconstruction": "open-loop replay of the recorded expert commands",
                    "act_run": False})

    policy_dt = float(config.policy_dt_ms) / 1000.0
    oracle: PerFrameParkedObstacleReference | None = None
    observation, _info = task.reset()
    step = 0
    while step < len(actions) and not task.is_done():
        obs = observation[0] if isinstance(observation, (list, tuple)) else observation
        if oracle is None:
            oracle = PerFrameParkedObstacleReference(task, sensor_order)

        current_skin = oracle.render_current_skin()
        parked_skin, neutrality = oracle.parked_skin()
        current_head = np.asarray(head(current_skin), dtype=np.float32)
        parked_head = np.asarray(head(parked_skin), dtype=np.float32)

        arm = np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32)
        grip = np.asarray((obs["qpos"].get("gripper") or [0.0, 0.0])[:2], dtype=np.float32)
        velocity = obs.get("qvel")
        vel_arm = np.asarray(velocity["arm"][:7], dtype=np.float32) \
            if isinstance(velocity, dict) else np.zeros(7, dtype=np.float32)
        vel_grip = np.asarray((velocity.get("gripper") or [0.0, 0.0])[:2],
                              dtype=np.float32) if isinstance(velocity, dict) \
            else np.zeros(2, dtype=np.float32)
        arm_command, gripper_command = actions[step]
        nominal = np.concatenate([np.asarray(arm_command, dtype=np.float32),
                                  np.asarray(gripper_command, dtype=np.float32)[:1]])

        retention.add(
            episode_step=step, control_timestamp=step * policy_dt,
            current_depth=current_skin, parked_depth=parked_skin,
            qpos=np.concatenate([arm, grip]), qvel=np.concatenate([vel_arm, vel_grip]),
            nominal_action=nominal, gripper_state=grip,
            gripper_command=nominal[7:8],
            current_head=current_head, parked_head=parked_head,
            teacher_dq=None, teacher_valid=False,
            scientific_state_sha256=array_hash(
                np.concatenate([arm, grip, vel_arm, vel_grip])),
            state_neutral=bool(neutrality["neutral"]))

        observation, _r, _t, _tr, _i = task.step([{
            "arm": np.asarray(arm_command, dtype=np.float32),
            "gripper": np.asarray(gripper_command, dtype=np.float32)}])
        step += 1

    record = retention.publish(args.out.resolve())
    print(json.dumps({k: record[k] for k in
                      ("episode_id", "frames", "oracle_active_frames",
                       "physical_inequality_violations", "bytes")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
