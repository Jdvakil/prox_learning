#!/usr/bin/env python3
"""Offline teacher-forced evaluation of the canonical ACT baseline checkpoint.

Handoff step 14. Reloads ``policy_best.ckpt`` and scores it on the fixed 20
validation trajectories with ground-truth actions supplied (teacher forcing).

These are offline imitation metrics only. Nothing here rolls out a policy in the
simulator, measures task success, or says anything about collision avoidance,
safety, or superiority over any other policy.

Evaluation protocol, fixed before the numbers were seen:

* every real timestep of every validation trajectory is used as a query start,
  so the metric does not depend on a sampled window;
* at start t the model predicts a chunk of ``chunk_size`` actions; only the
  ``min(chunk_size, T - t)`` real steps are scored, padding is never scored;
* normalized MAE is computed in the training statistics' normalized space,
  denormalized MAE in raw action units;
* the CVAE runs in inference mode (prior sample, no action encoder), which is
  how the policy is used at deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path("/root/prox_learning_hybrid_safety")
ACT_DIR = ROOT / "submodules" / "act"
sys.path.insert(0, str(ACT_DIR))


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_policy(policy_config, ckpt_path):
    # DETR re-parses sys.argv; present an argv it accepts.
    saved = sys.argv
    sys.argv = [saved[0], "--ckpt_dir", "/tmp/unused", "--policy_class", "ACT",
                "--task_name", "obstacle_baseline", "--seed", "0", "--num_epochs", "1"]
    try:
        from policy import ACTPolicy
        policy = ACTPolicy(policy_config)
    finally:
        sys.argv = saved
    state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    policy.load_state_dict(state, strict=True)   # raises if a key is missing or extra
    policy.cuda().eval()
    return policy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-starts", type=int, default=32,
                    help="query starts evaluated per forward pass")
    args = ap.parse_args()

    run_manifest = json.loads((args.run_dir / "run_manifest.json").read_text())
    with open(args.run_dir / "dataset_stats.pkl", "rb") as f:
        stats = pickle.load(f)

    # the statistics used at eval must be the ones this run trained with
    stats_meta = json.loads((args.run_dir / "dataset_stats_manifest.json").read_text())
    recomputed = canonical_hash({k: stats_meta[k] for k in
                                 ("qpos_mean", "qpos_std", "action_mean", "action_std")})
    if recomputed != stats_meta["statistics_sha256"]:
        raise SystemExit("dataset statistics manifest self-hash mismatch")
    for key, arr in (("qpos_mean", stats["qpos_mean"]), ("action_mean", stats["action_mean"])):
        if not np.allclose(np.asarray(stats_meta[key]), np.asarray(arr), atol=1e-12):
            raise SystemExit(f"{key} in dataset_stats.pkl differs from the statistics manifest")

    pc = run_manifest["policy_config"]
    chunk = int(pc["num_queries"])
    action_dim = int(pc["action_dim"])
    cameras = list(pc["camera_names"])

    ckpt = args.run_dir / "policy_best.ckpt"
    policy = build_policy(pc, ckpt)

    a_mean = torch.tensor(np.asarray(stats["action_mean"]), dtype=torch.float32, device="cuda")
    a_std = torch.tensor(np.asarray(stats["action_std"]), dtype=torch.float32, device="cuda")
    q_mean = torch.tensor(np.asarray(stats["qpos_mean"]), dtype=torch.float32, device="cuda")
    q_std = torch.tensor(np.asarray(stats["qpos_std"]), dtype=torch.float32, device="cuda")

    val_eps = run_manifest["val_episodes"]
    per_episode = []
    tot_norm_abs = torch.zeros(action_dim, dtype=torch.float64, device="cuda")
    tot_denorm_abs = torch.zeros(action_dim, dtype=torch.float64, device="cuda")
    tot_count = 0
    haz = {True: {"abs": 0.0, "n": 0}, False: {"abs": 0.0, "n": 0}}

    torch.manual_seed(0)
    np.random.seed(0)

    for ep in val_eps:
        idx = ep["act_episode_index"]
        path = args.dataset_dir / f"episode_{idx}.hdf5"
        with h5py.File(path, "r") as f:
            actions = np.asarray(f["/action"][()], dtype=np.float32)
            qpos_all = np.asarray(f["/observations/qpos"][()], dtype=np.float32)
            imgs = {c: np.asarray(f[f"/observations/images/{c}"][()]) for c in cameras}
        T = actions.shape[0]

        ep_norm_abs = torch.zeros(action_dim, dtype=torch.float64, device="cuda")
        ep_denorm_abs = torch.zeros(action_dim, dtype=torch.float64, device="cuda")
        ep_count = 0

        for s0 in range(0, T, args.batch_starts):
            starts = list(range(s0, min(s0 + args.batch_starts, T)))
            b = len(starts)
            image = np.stack([np.stack([imgs[c][t] for c in cameras], 0) for t in starts], 0)
            image_t = torch.from_numpy(image).cuda().float().permute(0, 1, 4, 2, 3) / 255.0
            qpos_t = torch.from_numpy(qpos_all[starts]).cuda()
            qpos_n = (qpos_t - q_mean) / q_std

            with torch.inference_mode():
                a_hat = policy(qpos_n, image_t)          # (b, chunk, action_dim), normalized

            # ground-truth chunk, normalized, with a validity mask over real steps
            gt = torch.zeros(b, chunk, action_dim, device="cuda")
            mask = torch.zeros(b, chunk, dtype=torch.bool, device="cuda")
            for i, t in enumerate(starts):
                n = min(chunk, T - t)
                gt[i, :n] = torch.from_numpy(actions[t:t + n]).cuda()
                mask[i, :n] = True
            gt_n = (gt - a_mean) / a_std

            err_n = (a_hat - gt_n).abs() * mask.unsqueeze(-1)
            err_d = ((a_hat * a_std + a_mean) - gt).abs() * mask.unsqueeze(-1)
            ep_norm_abs += err_n.sum(dim=(0, 1)).double()
            ep_denorm_abs += err_d.sum(dim=(0, 1)).double()
            ep_count += int(mask.sum().item())

        per_episode.append({
            "act_episode_index": idx,
            "source_episode_id": ep["source_episode_id"],
            "hazard_present": bool(ep["hazard_present"]),
            "real_timesteps": int(T),
            "scored_action_elements": ep_count,
            "normalized_action_mae": float((ep_norm_abs.sum() / (ep_count * action_dim)).item()),
            "denormalized_action_mae": float((ep_denorm_abs.sum() / (ep_count * action_dim)).item()),
        })
        tot_norm_abs += ep_norm_abs
        tot_denorm_abs += ep_denorm_abs
        tot_count += ep_count
        k = bool(ep["hazard_present"])
        haz[k]["abs"] += float(ep_denorm_abs.sum().item())
        haz[k]["n"] += ep_count * action_dim

    per_dim_norm = (tot_norm_abs / tot_count).tolist()
    per_dim_denorm = (tot_denorm_abs / tot_count).tolist()
    overall_norm = float(tot_norm_abs.sum().item() / (tot_count * action_dim))
    overall_denorm = float(tot_denorm_abs.sum().item() / (tot_count * action_dim))

    arm_denorm = float(sum(per_dim_denorm[:7]) / 7)
    grip_denorm = float(per_dim_denorm[7])
    arm_norm = float(sum(per_dim_norm[:7]) / 7)
    grip_norm = float(per_dim_norm[7])

    # Determinism of the reload on a fixed validation batch: build a *second*
    # policy from the same checkpoint file and confirm it produces the identical
    # output for the identical input under the same seed.
    def fixed_batch_output(p):
        ep = val_eps[0]
        with h5py.File(args.dataset_dir / f"episode_{ep['act_episode_index']}.hdf5", "r") as f:
            q = np.asarray(f["/observations/qpos"][:8], dtype=np.float32)
            im = np.stack([np.stack([np.asarray(f[f"/observations/images/{c}"][t])
                                     for c in cameras], 0) for t in range(8)], 0)
        qt = (torch.from_numpy(q).cuda() - q_mean) / q_std
        it = torch.from_numpy(im).cuda().float().permute(0, 1, 4, 2, 3) / 255.0
        torch.manual_seed(0)
        with torch.inference_mode():
            return p(qt, it).clone()

    o1 = fixed_batch_output(policy)
    o2 = fixed_batch_output(build_policy(pc, ckpt))
    deterministic = bool(torch.equal(o1, o2))
    max_reload_diff = float((o1 - o2).abs().max().item())

    report = {
        "schema": "hybrid_obstacle_act_baseline_offline_validation",
        "disclaimer": ("Offline teacher-forced imitation metrics only. No simulator rollout, "
                       "no task-success measurement, and no claim about collision avoidance, "
                       "safety, or superiority over any other policy."),
        "run_dir": str(args.run_dir),
        "dataset_dir": str(args.dataset_dir),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "policy_last_sha256": sha256_file(args.run_dir / "policy_last.ckpt"),
        "dataset_stats_pkl_sha256": sha256_file(args.run_dir / "dataset_stats.pkl"),
        "statistics_sha256": stats_meta["statistics_sha256"],
        "statistics_match_run_manifest": (
            stats_meta["statistics_sha256"] == run_manifest["statistics"]["statistics_sha256"]),
        "split_manifest_sha256": run_manifest["split_manifest_sha256"],
        "state_dim": int(pc["state_dim"]),
        "protocol": {
            "teacher_forced": True,
            "query_starts": "every real timestep of every validation trajectory",
            "chunk_size": chunk,
            "padding_scored": False,
            "cvae_mode": "inference (prior sample, no action encoder)",
        },
        "validation_trajectories": len(val_eps),
        "validation_real_timesteps": sum(e["real_timesteps"] for e in per_episode),
        "scored_action_elements": tot_count * action_dim,
        "normalized_action_mae": overall_norm,
        "denormalized_action_mae": overall_denorm,
        "per_action_dimension_mae": {
            "normalized": per_dim_norm,
            "denormalized": per_dim_denorm,
            "dimension_names": [f"arm_joint_{i}" for i in range(7)] + ["gripper_command"],
        },
        "arm_joint_mae": {"normalized": arm_norm, "denormalized": arm_denorm},
        "gripper_command_mae": {"normalized": grip_norm, "denormalized": grip_denorm},
        "hazard_present_denormalized_mae": haz[True]["abs"] / haz[True]["n"],
        "hazard_absent_denormalized_mae": haz[False]["abs"] / haz[False]["n"],
        "hazard_present_trajectories": sum(1 for e in per_episode if e["hazard_present"]),
        "hazard_absent_trajectories": sum(1 for e in per_episode if not e["hazard_present"]),
        "reload_deterministic_on_fixed_batch": deterministic,
        "reload_max_abs_diff": max_reload_diff,
        "per_episode": per_episode,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"validation trajectories      {report['validation_trajectories']}")
    print(f"validation real timesteps    {report['validation_real_timesteps']}")
    print(f"normalized action MAE        {overall_norm:.6f}")
    print(f"denormalized action MAE      {overall_denorm:.6f}")
    print(f"arm-joint MAE (denorm)       {arm_denorm:.6f}")
    print(f"gripper MAE (denorm)         {grip_denorm:.6f}")
    print(f"hazard-present MAE (denorm)  {report['hazard_present_denormalized_mae']:.6f}")
    print(f"hazard-absent MAE (denorm)   {report['hazard_absent_denormalized_mae']:.6f}")
    print(f"reload deterministic         {deterministic} (max diff {max_reload_diff:.3e})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
