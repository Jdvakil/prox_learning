"""Sensor importance analysis.

For a trained PLA checkpoint, mask each sensor individually and measure the
performance drop on the near-contact task. Output: per-sensor success rate
delta, suitable for plotting as a heatmap on the FR3 body
(see ``pla.viz.heatmap``).

Method
    1. Run a baseline eval with no masking; record success per episode.
    2. For each sensor index ``i`` in ``range(n_sensors)``, run an eval with
       ``model.sensor_mask[i] = True`` and the same per-episode seeds; record
       per-episode success.
    3. ``importance[i] = baseline_sr - masked_sr``. Higher = more important.

Pairing
    The same seed sequence is reused for the baseline and every masked run,
    so importance values are directly comparable per-episode (no rerolling
    of scenes / object placements).

Run::

    python -m pla.eval.sensor_importance \
        --checkpoint runs/pla_concat_v1/best.pt \
        --task near_contact \
        --n-episodes 50 \
        --out reports/tables/sensor_importance.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pla.eval.run_eval import _make_env, load_model_from_checkpoint
from pla.eval.tasks import REGISTRY


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-sensor importance via masking")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", default="near_contact")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dummy-vlm", action="store_true")
    p.add_argument("--max-steps-per-episode", type=int, default=600)
    return p.parse_args()


def _eval_with_mask(
    model,
    cfg,
    stats,
    spec,
    n_episodes: int,
    seed_base: int,
    masked_indices: list[int] | None,
    device: torch.device,
    max_steps: int,
) -> list[int]:
    from pla.sim.tof import ToFSensorArray

    if masked_indices is not None:
        mask = torch.zeros(cfg["n_sensors"], dtype=torch.bool, device=device)
        for i in masked_indices:
            mask[i] = True
        model.sensor_mask = mask
    else:
        model.sensor_mask = torch.zeros(cfg["n_sensors"], dtype=torch.bool, device=device)

    out = []
    for i in range(n_episodes):
        seed = seed_base + i
        env = _make_env(spec, seed=seed)
        tof = ToFSensorArray(env.model)
        obs = env.reset()
        obs["tof"] = tof.render(env.data, add_noise=False)
        info: dict = {}
        for _ in range(max_steps):
            obs_norm = {
                "tof": (np.asarray(obs["tof"]) - np.asarray(stats["tof_mean"]))
                       / (np.asarray(stats["tof_std"]) + 1e-6),
                "rgb": np.asarray(obs["rgb"]).astype(np.float32) / 255.0,
                "qpos": (np.asarray(obs["qpos"]) - np.asarray(stats["qpos_mean"]))
                        / (np.asarray(stats["qpos_std"]) + 1e-6),
                "language": spec.language,
            }
            actions_chunk = model.get_action(obs_norm, stats, device=device)
            done = False
            for a in actions_chunk:
                obs, _r, done, info = env.step(a)
                obs["tof"] = tof.render(env.data, add_noise=False)
                if done:
                    break
            if done:
                break
        out.append(int(info.get("success", False)))
        if hasattr(env, "close"):
            env.close()
    return out


def compute_sensor_importance(
    *,
    ckpt_path: Path,
    task: str,
    n_episodes: int,
    seed_base: int,
    dummy_vlm: bool = False,
    max_steps_per_episode: int = 600,
) -> dict:
    if task not in REGISTRY:
        raise ValueError(f"unknown task {task!r}")
    spec = REGISTRY[task]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model_from_checkpoint(ckpt_path, dummy_vlm=dummy_vlm, device=device)
    with open(cfg["stats_path"]) as f:
        stats = json.load(f)
    n_sensors = cfg["n_sensors"]

    base_succ = _eval_with_mask(
        model, cfg, stats, spec, n_episodes, seed_base,
        masked_indices=None, device=device, max_steps=max_steps_per_episode,
    )
    base_sr = float(np.mean(base_succ))

    importance: list[dict] = []
    for i in range(n_sensors):
        succ_i = _eval_with_mask(
            model, cfg, stats, spec, n_episodes, seed_base,
            masked_indices=[i], device=device, max_steps=max_steps_per_episode,
        )
        sr_i = float(np.mean(succ_i))
        importance.append({
            "sensor_index": i,
            "masked_success_rate": sr_i,
            "delta_vs_baseline": base_sr - sr_i,
            "successes": succ_i,
        })
        print(f"sensor {i:2d}: masked_sr={sr_i:.3f} delta={base_sr - sr_i:+.3f}")

    return {
        "task": task,
        "checkpoint": str(ckpt_path),
        "n_episodes": n_episodes,
        "baseline_success_rate": base_sr,
        "baseline_successes": base_succ,
        "per_sensor": importance,
    }


def main() -> None:
    args = parse_args()
    out = compute_sensor_importance(
        ckpt_path=args.checkpoint,
        task=args.task,
        n_episodes=args.n_episodes,
        seed_base=args.seed_base,
        dummy_vlm=args.dummy_vlm,
        max_steps_per_episode=args.max_steps_per_episode,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
