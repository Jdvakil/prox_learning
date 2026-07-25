#!/usr/bin/env python3
"""Build the paired oracle-reference dataset over the canonical 100 trajectories.

Handoff step 3. For every real decision timestep of every canonical trajectory this
reconstructs the recorded scientific state, renders ``current_skin`` and the
parked-obstacle counterfactual ``parked_skin`` **at the same decision state** through
the validated state-neutral seam, and stores the paired heads together with the
runtime-observable inputs a deployable reference is allowed to see.

Pairing correctness
-------------------
The corrected oracle pairing from ``ORACLE_REFERENCE_VALID_CONTROLLER_VIABLE`` is reused
verbatim via ``PerFrameParkedObstacleReference``:

* both halves are rendered at the *same* decision state -- the observation's own last
  proximity sub-step is **not** used as the current half, because it lands one sim
  sub-step earlier and contaminates the differential (measured median ratio 1.0, and up
  to 2.50 of pure artefact on a hazard-absent row);
* no dynamics-advancing operation is called, and the earlier ``mj_forward``
  implementation that moved 23 bodies is not used;
* the validated render state is restored directly and 21 state fields are hashed before
  and after every counterfactual.

State reconstruction is by deterministic open-loop replay of the recorded expert joint
commands from the verified initial state, because the trajectory H5 stores no per-step
pose for the pickup object.

Privileged fields (hazard presence, obstacle pose) are written into a separate
``privileged_`` namespace for audit only. The feature builders never read them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
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

#: The analytic teacher's activation radius (safety_sweep.py:58). Outside it the
#: deployable residual has no directional target and must stay quiet.
D_ACT = 0.18
DEAD_PIXEL_BELOW_M = 0.005
CAUSAL_FRAMES = 4


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sensor_summaries(skin: np.ndarray) -> np.ndarray:
    """(40, 4) causal per-sensor summary of one 40x8x8 frame.

    Columns: minimum valid depth, tenth-percentile valid depth, mean closeness,
    fraction of pixels below D_ACT. "Valid" excludes dead pixels below 5 mm. A sensor
    with no valid return reports the saturating value for each statistic, so the encoding
    never confuses "nothing in view" with "something touching".
    """
    frames = np.asarray(skin, dtype=np.float32)
    valid = frames >= DEAD_PIXEL_BELOW_M
    any_valid = valid.any(axis=(1, 2))
    filled = np.where(valid, frames, np.inf)
    minimum = np.where(any_valid, filled.min(axis=(1, 2)), 1.0)
    percentile = np.empty(40, dtype=np.float32)
    for index in range(40):
        values = frames[index][valid[index]]
        percentile[index] = np.percentile(values, 10.0) if values.size else 1.0
    # closeness in [0, 1]: 1 at contact, 0 at or beyond the activation radius
    closeness = np.clip(1.0 - np.where(valid, frames, D_ACT) / D_ACT, 0.0, 1.0)
    mean_closeness = closeness.mean(axis=(1, 2))
    below = (valid & (frames < D_ACT)).mean(axis=(1, 2))
    return np.stack([np.minimum(minimum, 1.0), np.minimum(percentile, 1.0),
                     mean_closeness, below], axis=1).astype(np.float32)


def load_recorded(h5_path: Path):
    import h5py

    with h5py.File(h5_path, "r") as handle:
        key = next(k for k in handle if k.startswith("traj"))
        group = handle[key]
        actions = []
        for row in group["actions/joint_pos"][:]:
            payload = json.loads(bytes(row).split(b"\x00")[0].decode())
            if "arm" not in payload:
                break                       # the terminal step records an empty action
            actions.append((payload["arm"], payload["gripper"]))
        scene = json.loads(group["obs_scene"][()])
        success = bool(group["success"][:][-1])
    return actions, scene, success


def build_trajectory(episode, collection_rows, sensor_order, head, act_policy_factory,
                     run_dir: Path, out_dir: Path, teacher_cls) -> dict[str, Any]:
    """Replay one canonical trajectory and emit its paired examples."""
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

    episode_id = episode["episode_id"]
    coll_row = collection_rows[episode_id]
    h5_path = run_dir / "rows" / episode_id / "trajectory.h5"
    actions, scene, recorded_success = load_recorded(h5_path)
    params = scene["scene_params"]

    config = FrankaSkinHybridObstacleManifestV2Config()
    config.task_horizon = len(actions)
    sampler = config.task_sampler_config.task_sampler_class(config)
    reset_episode_scoped_sampler_state(sampler)
    retry = int(coll_row.get("accepted_retry_index", episode.get("accepted_retry_index", 0)))
    install_row_seed_contract(coll_row, retry, task_sampler=sampler)
    sampler.set_manifest_row(coll_row, retry)
    task = sampler.sample_task(house_index=coll_row["scene_template_house_index"])
    if task is None:
        raise SystemExit(f"task sampling returned None for {episode_id}")

    observations = extract_row_observations(task, coll_row)
    replay_hash = initial_state_hash(observations, coll_row)

    oracle = None
    policy = act_policy_factory(task)
    teacher = None

    rows: dict[str, list] = {k: [] for k in (
        "timestep", "qpos", "qvel", "gripper_state", "nominal_action", "gripper_command",
        "current_head", "parked_head", "oracle_dq", "sensor_summary", "skin_stack",
        "minimum_depth", "runtime_support", "teacher_active", "teacher_dq",
        "teacher_valid", "state_neutral", "skins_identical", "task_phase")}
    privileged: dict[str, list] = {"hazard_present": [], "hazard_pose": [],
                                   "minimum_hazard_return": []}
    causal: list[np.ndarray] = []

    observation, _info = task.reset()
    started = time.time()
    step = 0
    while step < len(actions) and not task.is_done():
        obs = observation[0] if isinstance(observation, (list, tuple)) else observation
        if oracle is None:
            oracle = PerFrameParkedObstacleReference(task, sensor_order)
            model = task.env.current_model
            arm_dofadr = [model.joint(f"robot_0/fr3_joint{i}").dofadr[0] for i in range(1, 8)]
            hazard_box = None
            if params.get("protrusion_present"):
                hazard_box = (np.asarray(params["protr_center"], dtype=float),
                              np.asarray(params["protr_half"], dtype=float))
            teacher = teacher_cls(model, sensor_order, hazard_box, arm_dofadr)

        data = task.env.current_data
        current_skin = oracle.render_current_skin()
        parked_skin, neutrality = oracle.parked_skin()
        current_head = np.asarray(head(current_skin), dtype=np.float32)
        parked_head = np.asarray(head(parked_skin), dtype=np.float32)

        nominal = policy(obs)
        teacher_out = teacher(data, current_skin)

        summary = sensor_summaries(current_skin)
        minimum_depth = float(summary[:, 0].min())
        causal.append(current_skin)
        if len(causal) > CAUSAL_FRAMES:
            causal.pop(0)
        stack = np.zeros((CAUSAL_FRAMES, 40, 8, 8), dtype=np.float32)
        stack[CAUSAL_FRAMES - len(causal):] = np.stack(causal, axis=0)

        arm = np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32)
        grip = np.asarray((obs["qpos"].get("gripper") or [0.0, 0.0])[:2], dtype=np.float32)
        vel_arm = np.asarray(obs["qvel"]["arm"][:7], dtype=np.float32) \
            if isinstance(obs.get("qvel"), dict) else np.zeros(7, dtype=np.float32)
        vel_grip = np.asarray((obs["qvel"].get("gripper") or [0.0, 0.0])[:2],
                              dtype=np.float32) if isinstance(obs.get("qvel"), dict) \
            else np.zeros(2, dtype=np.float32)

        rows["timestep"].append(step)
        rows["qpos"].append(np.concatenate([arm, grip]))
        rows["qvel"].append(np.concatenate([vel_arm, vel_grip]))
        rows["gripper_state"].append(grip)
        rows["nominal_action"].append(nominal.astype(np.float32))
        rows["gripper_command"].append(np.asarray([nominal[7]], dtype=np.float32))
        rows["current_head"].append(current_head)
        rows["parked_head"].append(parked_head)
        rows["oracle_dq"].append((current_head - parked_head).astype(np.float32))
        rows["sensor_summary"].append(summary)
        rows["skin_stack"].append(stack)
        rows["minimum_depth"].append(minimum_depth)
        # runtime support gate A: any valid return below D_ACT. Non-privileged.
        rows["runtime_support"].append(bool(minimum_depth < D_ACT))
        # teacher-active is PRIVILEGED-derived (it needs to know which returns are the
        # hazard) and is metric-only; it never enters a feature vector.
        rows["teacher_active"].append(bool(teacher_out["contributing_sensors"]))
        rows["teacher_dq"].append(np.asarray(teacher_out["tau"], dtype=np.float32))
        rows["teacher_valid"].append(bool(teacher_out["contributing_sensors"]))
        rows["state_neutral"].append(bool(neutrality["neutral"]))
        rows["skins_identical"].append(bool(np.array_equal(current_skin, parked_skin)))
        rows["task_phase"].append(0)

        privileged["hazard_present"].append(bool(oracle.hazard_present))
        privileged["hazard_pose"].append(
            np.asarray([oracle.live_hazard_pose[n] for n in sorted(oracle.parked_pose)],
                       dtype=np.float32))
        privileged["minimum_hazard_return"].append(
            float(teacher_out["minimum_hazard_return_m"])
            if teacher_out["minimum_hazard_return_m"] is not None else np.inf)

        arm_cmd, grip_cmd = actions[step]
        observation, _r, _t, _tr, _i = task.step([{
            "arm": np.asarray(arm_cmd, dtype=np.float32),
            "gripper": np.asarray(grip_cmd, dtype=np.float32)}])
        step += 1

    arrays = {k: np.asarray(v) for k, v in rows.items()}
    arrays.update({f"privileged_{k}": np.asarray(v) for k, v in privileged.items()})
    out_path = out_dir / f"{episode_id}.npz"
    # np.savez_compressed appends .npz unless the name already ends with it, so the
    # temporary name must too or the rename below chases a file that was never written.
    tmp = out_dir / f"{episode_id}.partial.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out_path)

    hazard_active = int(np.sum(arrays["runtime_support"]))
    return {
        "episode_id": episode_id,
        "candidate_index": episode["candidate_index"],
        "split": episode["split"],
        "hazard_present": bool(episode["hazard_present"]),
        "frames": int(step),
        "recorded_actions": len(actions),
        "recorded_success": recorded_success,
        "runtime_support_frames": hazard_active,
        "teacher_active_frames": int(np.sum(arrays["teacher_active"])),
        "state_neutral_all_frames": bool(np.all(arrays["state_neutral"])),
        "skins_identical_all_frames": bool(np.all(arrays["skins_identical"])),
        "oracle_dq_max_abs": float(np.max(np.abs(arrays["oracle_dq"]))) if step else 0.0,
        "replayed_initial_state_sha256": replay_hash,
        "initial_state_matches_manifest": replay_hash == coll_row.get(
            "initial_state_sha256", replay_hash),
        "source_h5_sha256": episode["source_h5_sha256"],
        "file": out_path.name,
        "file_sha256": sha256_file(out_path),
        "seconds": round(time.time() - started, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-manifest", required=True, type=Path)
    ap.add_argument("--collection-manifest", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip trajectories whose .npz is already present; generation is "
                         "deterministic given the manifest row, so a resumed shard "
                         "produces the same files a fresh one would")
    args = ap.parse_args()

    import torch
    from hybrid_obstacle_oracle_offline_signal import HazardTeacher
    from train_safety_cvae import SafetyHead

    split = json.loads(args.split_manifest.read_text())
    collection = {r["episode_id"]: r
                  for r in json.loads(args.collection_manifest.read_text())["rows"]}
    stack = json.loads(args.stack.read_text())
    sensor_order = list(stack["sensor_contract"]["ordered_names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = SafetyHead.load(str(args.safety_dir), device=device)

    episodes = sorted(split["episodes"], key=lambda e: e["split_rank"])
    if args.limit:
        episodes = episodes[:args.limit]
    episodes = [e for index, e in enumerate(episodes) if index % args.shards == args.shard]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def act_policy_factory(task):
        """Frozen ACT inference only. It never drives the environment."""
        from eval_act_obstacle_manifest_safety import ManifestSafetyEvalConfig
        from eval_act_obstacle_rawhead import RawHeadReplayPolicy

        config = ManifestSafetyEvalConfig()
        config.num_workers = 1
        pc = config.policy_config
        pc.policy_cls = RawHeadReplayPolicy
        pc.ckpt_dir = str(args.ckpt_dir.resolve())
        pc.ckpt_name = "policy_best.ckpt"
        pc.expected_act_checkpoint_sha256 = EXPECTED_CKPT
        pc.expected_dataset_stats_sha256 = EXPECTED_STATS
        pc.safety_ckpt_dir = str(args.safety_dir.resolve())
        pc.contract_path = str(args.stack.resolve())
        pc.safety_mode = "act_only"
        pc.episode_seed = 0
        pc.temp_agg_off = False
        pc.temp_agg_m = 0.01
        RawHeadReplayPolicy.condition = "ACT_ONLY"
        RawHeadReplayPolicy.manifest_provenance = {"condition": "ACT_ONLY"}
        policy = pc.policy_cls(config, task)
        policy.prepare_model()

        def nominal(obs):
            output = policy.inference_model(obs)
            action = policy.model_output_to_action(output)
            policy._step += 1
            return np.concatenate([np.asarray(action["arm"], dtype=np.float32),
                                   np.asarray(action["gripper"], dtype=np.float32)[:1]])

        return nominal

    reports = []
    for episode in episodes:
        if args.skip_existing and (args.out_dir / f"{episode['episode_id']}.npz").is_file():
            print(f"  skip {episode['split']:<10} cand{episode['candidate_index']:>4} "
                  "(already present)", flush=True)
            continue
        report = build_trajectory(episode, collection, sensor_order, head,
                                  act_policy_factory, args.run_dir, args.out_dir,
                                  HazardTeacher)
        reports.append(report)
        print(f"  {report['split']:<10} cand{report['candidate_index']:>4} "
              f"T={report['frames']:>4} support={report['runtime_support_frames']:>4} "
              f"teacher={report['teacher_active_frames']:>4} "
              f"neutral={report['state_neutral_all_frames']} "
              f"{report['seconds']}s", flush=True)

    shard_report = {
        "schema": "hybrid_obstacle_paired_reference_shard_v1",
        "shard": args.shard, "shards": args.shards,
        "trajectories": reports,
    }
    (args.out_dir / f"_shard_{args.shard:02d}.json").write_text(
        json.dumps(shard_report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"shard {args.shard}: {len(reports)} trajectories")
    return 0


EXPECTED_CKPT = "dd7cd108a64ce10e5aab21b525dc06190f54d4e5fe446f65715b6852c49e7d36"
EXPECTED_STATS = "c8119b904bfc80d66e3d33825722fcf9bb8bf3433c956dc09c27e6517d7c4ae2"

if __name__ == "__main__":
    raise SystemExit(main())
