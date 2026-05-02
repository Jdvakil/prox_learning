"""Evaluation harness.

Runs N episodes for one model checkpoint against a task in
``pla.eval.tasks.REGISTRY``, writes a JSON result file, and prints (or
appends to) a results table with bootstrap 95% CIs and paired-bootstrap
p-values vs the VLM-only baseline.

The benchmark backend is the MolmoSpaces ``FrankaPickandPlace`` simulator (and
its near-contact / pnp_color / pnp_next_to variants — see
``pla.eval.tasks.REGISTRY``). We import it lazily; if MolmoSpaces is not
importable the script can still print existing JSON results and aggregate
tables (handy when you have results from another machine).

Pairing
    The same per-episode (seed, scene_id, language) tuple is used across all
    methods so the paired bootstrap is valid. We drive that from
    ``--seed-base`` plus ``range(n_episodes)``.

Output JSON shape::

    {
      "method": "PLA",
      "task": "near_contact",
      "n_episodes": 100,
      "successes": [0, 1, 1, ...],
      "language": [...],
      "scene_ids": [...],
      "seeds":     [...],
      "checkpoint": "runs/pla_concat_v1/best.pt",
      "config":    {...},
    }
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from pla.eval.bootstrap import bootstrap_ci, paired_bootstrap_p
from pla.eval.tasks import REGISTRY
from pla.models import DummyVLBackbone, FrozenMolmo2
from pla.train.train import build_model_from_cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", type=str, default=None,
                   help="task name from pla.eval.tasks.REGISTRY")
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--method-name", type=str, default=None)
    p.add_argument("--dummy-vlm", action="store_true")
    p.add_argument("--max-steps-per-episode", type=int, default=600)
    p.add_argument("--print-table", type=Path, default=None,
                   help="if given, print a markdown table over all JSON files in this dir")
    return p.parse_args()


def load_model_from_checkpoint(
    ckpt_path: Path, *, dummy_vlm: bool, device: torch.device,
):
    state = torch.load(ckpt_path, map_location=device)
    cfg = state["config"]
    if dummy_vlm or cfg.get("dummy_vlm", False):
        vl = DummyVLBackbone(d_model=cfg["d_model"])
    else:
        vl = FrozenMolmo2(
            model_name=cfg.get("vlm_model_name", "allenai/Molmo-4B-D-0924"),
            d_model=cfg["d_model"],
        )
    model = build_model_from_cfg(cfg, vl).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, cfg


def _make_env(task_spec, seed: int):
    """Return an env exposing reset()/step()/close() and ``model``/``data``."""
    try:
        mod = importlib.import_module(task_spec.env_module)
        cls = getattr(mod, task_spec.env_class)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"Could not import benchmark env "
            f"{task_spec.env_module}.{task_spec.env_class}. "
            f"Install MolmoSpaces submodule. Original error: {e}"
        )
    return cls(seed=seed, **task_spec.env_kwargs)


def evaluate_checkpoint(
    *,
    ckpt_path: Path,
    task: str,
    n_episodes: int,
    seed_base: int,
    out_path: Path,
    method_name: str | None = None,
    dummy_vlm: bool = False,
    max_steps_per_episode: int = 600,
) -> dict:
    if task not in REGISTRY:
        raise ValueError(f"unknown task {task!r}; choose one of {list(REGISTRY)}")
    spec = REGISTRY[task]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model_from_checkpoint(ckpt_path, dummy_vlm=dummy_vlm, device=device)
    method = method_name or cfg.get("run_name", ckpt_path.stem)

    stats_path = cfg["stats_path"]
    with open(stats_path) as f:
        stats = json.load(f)

    from pla.sim.tof import ToFSensorArray

    successes: list[int] = []
    languages: list[str] = []
    scene_ids: list[str] = []
    seeds: list[int] = []

    for i in range(n_episodes):
        seed = seed_base + i
        env = _make_env(spec, seed=seed)
        tof = ToFSensorArray(env.model)
        obs = env.reset()
        obs["tof"] = tof.render(env.data, add_noise=False)
        scene_ids.append(getattr(env, "scene_id", str(i)))
        languages.append(spec.language)
        info: dict = {}

        for _ in range(max_steps_per_episode):
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
        successes.append(int(info.get("success", False)))
        seeds.append(seed)
        if hasattr(env, "close"):
            env.close()

    out = {
        "method": method,
        "task": task,
        "n_episodes": n_episodes,
        "successes": successes,
        "language": languages,
        "scene_ids": scene_ids,
        "seeds": seeds,
        "checkpoint": str(ckpt_path),
        "config": cfg,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}: {sum(successes)}/{n_episodes} = "
          f"{100*np.mean(successes):.1f}%")
    return out


def print_results_table(results_dir: Path, baseline: str = "VLM-only ACT") -> None:
    """Print a markdown-ish table of all results JSON files in a directory."""
    files = sorted(Path(results_dir).glob("*.json"))
    if not files:
        print(f"no JSON results found in {results_dir}")
        return
    by_task: dict[str, dict[str, dict]] = {}
    for f in files:
        d = json.loads(f.read_text())
        by_task.setdefault(d["task"], {})[d["method"]] = d

    for task, methods in by_task.items():
        print(f"\n## task: {task}")
        baseline_d = methods.get(baseline)
        header = f"{'Method':<25} {'SR':>8} {'95% CI':>18}"
        if baseline_d is not None:
            header += f"  p vs {baseline}"
        print(header)
        print("-" * len(header))
        for name, d in sorted(methods.items()):
            s = np.asarray(d["successes"])
            mean, lo, hi = bootstrap_ci(s)
            line = f"{name:<25} {100*mean:>7.1f}%  [{100*lo:>5.1f}%, {100*hi:>5.1f}%]"
            if baseline_d is not None and name != baseline:
                p = paired_bootstrap_p(s, np.asarray(baseline_d["successes"]))
                line += f"  p={p:.4f}"
            print(line)


def main() -> None:
    args = parse_args()
    if args.print_table is not None:
        print_results_table(args.print_table)
        return
    if args.task is None or args.out is None:
        raise SystemExit("--task and --out are required unless --print-table is set")
    evaluate_checkpoint(
        ckpt_path=args.checkpoint,
        task=args.task,
        n_episodes=args.n_episodes,
        seed_base=args.seed_base,
        out_path=args.out,
        method_name=args.method_name,
        dummy_vlm=args.dummy_vlm,
        max_steps_per_episode=args.max_steps_per_episode,
    )


if __name__ == "__main__":
    main()
