"""Trajectory collection harness.

Drives MolmoBot-Engine TAMP planning in MolmoSpaces procthor-objaverse scenes
to collect 1000+ expert trajectories with whole-body ToF, RGB, and qpos.

Critical requirement (PROJECT.md §3.3): at least 30% of trajectories must
include ToF readings below 200 mm. Standard PnP may not hit this — use the
near-contact task with a fixed obstacle 5-8 cm from the expert path.

HDF5 schema per episode::

    episode_N/
      observations/
        tof:  [T, N_sensors, 8, 8]  float32  mm, clipped [20, 4000]
        rgb:  [T, 3, 224, 224]       uint8
        qpos: [T, 7]                 float32
      actions:  [T, 7]              float32  joint delta
      metadata.attrs: {task, scene_id, success, seed, policy_phase, n_sensors}

Two collection paths are provided:

    1. ``collect_episode(env, policy, tof_array, episode_idx, out_dir)`` — the
       core single-episode loop. It mutates ``obs`` in place to add ``tof``
       via ``ToFSensorArray.render`` and serializes to HDF5. Reuse this from
       inside MolmoSpaces' ``data_generation.main`` worker.

    2. ``main()`` — CLI entry point. Reads a YAML task config, instantiates
       the MolmoSpaces env + a TAMP policy, and runs ``collect_episode`` in
       a tight loop until ``--n-traj`` are written.

The MolmoSpaces wiring is intentionally optional: when the submodule is not
importable we still let users run ``--dry-run`` to validate the schema and
output paths.

Run::

    python -m pla.data.collect --config configs/data/near_contact.yaml --n-traj 1000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect expert trajectories")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="defaults to data/raw/<task_name>")
    p.add_argument("--n-traj", type=int, default=1000)
    p.add_argument("--n-envs", type=int, default=10, help="distinct procthor scenes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="skip MolmoSpaces; write a synthetic schema-valid file")
    return p.parse_args()


def _write_episode_h5(
    h5_path: Path,
    obs_seq: dict[str, np.ndarray],
    actions: np.ndarray,
    *,
    success: bool,
    n_sensors: int,
    policy_phase: np.ndarray | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        g = f.create_group("episode_0")
        og = g.create_group("observations")
        og.create_dataset("tof", data=obs_seq["tof"].astype(np.float32))
        og.create_dataset("rgb", data=obs_seq["rgb"].astype(np.uint8))
        og.create_dataset("qpos", data=obs_seq["qpos"].astype(np.float32))
        g.create_dataset("actions", data=actions.astype(np.float32))
        if policy_phase is not None:
            g.create_dataset("policy_phase", data=np.asarray(policy_phase))
        g.attrs["success"] = bool(success)
        g.attrs["n_sensors"] = int(n_sensors)
        for k, v in (extra_attrs or {}).items():
            g.attrs[k] = v


def collect_episode(
    env,
    policy,
    tof_array,
    episode_idx: int,
    out_dir: Path,
    *,
    extra_attrs: dict[str, Any] | None = None,
) -> bool:
    """Run one episode, write one HDF5 file, return ``success``.

    The ``env`` must expose ``reset()``, ``step(action)``, and ``data`` (the
    live mjData). The ``policy`` must expose ``get_action(obs)``.
    """
    obs = env.reset()
    obs["tof"] = tof_array.render(env.data, add_noise=True)

    tof_buf, rgb_buf, qpos_buf, act_buf, phase_buf = [], [], [], [], []
    done = False
    info: dict[str, Any] = {}

    while not done:
        action = policy.get_action(obs)
        next_obs, _reward, done, info = env.step(action)
        next_obs["tof"] = tof_array.render(env.data, add_noise=True)

        tof_buf.append(obs["tof"].copy())
        rgb_buf.append(obs["rgb"].copy())
        qpos_buf.append(obs["qpos"].copy())
        act_buf.append(np.asarray(action, dtype=np.float32))
        phase_buf.append(int(info.get("policy_phase", 0)))
        obs = next_obs

    success = bool(info.get("success", False))
    h5_path = out_dir / f"episode_{episode_idx:06d}.h5"
    _write_episode_h5(
        h5_path,
        obs_seq={
            "tof": np.stack(tof_buf),
            "rgb": np.stack(rgb_buf),
            "qpos": np.stack(qpos_buf),
        },
        actions=np.stack(act_buf),
        success=success,
        n_sensors=tof_array.n_sensors,
        policy_phase=np.asarray(phase_buf, dtype=np.int32),
        extra_attrs=extra_attrs,
    )
    return success


def _write_synthetic_episode(out_dir: Path, idx: int, n_sensors: int = 32) -> None:
    """Schema-valid synthetic episode used for ``--dry-run``."""
    T = 250
    rng = np.random.default_rng(idx)
    tof = rng.uniform(20.0, 4000.0, size=(T, n_sensors, 8, 8)).astype(np.float32)
    # Inject some near-contact frames so the proximity-informative check passes.
    tof[100:120, :4] = rng.uniform(40.0, 180.0, size=(20, 4, 8, 8))
    rgb = (rng.uniform(0, 255, size=(T, 3, 224, 224))).astype(np.uint8)
    qpos = rng.standard_normal((T, 7)).astype(np.float32)
    actions = rng.standard_normal((T, 7)).astype(np.float32) * 0.01
    _write_episode_h5(
        out_dir / f"episode_{idx:06d}.h5",
        obs_seq={"tof": tof, "rgb": rgb, "qpos": qpos},
        actions=actions,
        success=bool(rng.random() > 0.1),
        n_sensors=n_sensors,
        policy_phase=np.zeros(T, dtype=np.int32),
        extra_attrs={"task": "synthetic_dryrun"},
    )


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    task_name = cfg.get("task_name") or args.config.stem
    out_dir = args.out_dir or Path("data/raw") / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path("reports/logs") / f"collect_{task_name}_{int(time.time())}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for i in range(args.n_traj):
            _write_synthetic_episode(out_dir, i, n_sensors=cfg.get("n_sensors", 32))
        log_path.write_text(json.dumps({
            "n_traj": args.n_traj, "task": task_name, "dry_run": True,
        }, indent=2))
        return

    try:
        from molmo_spaces.data_generation.main import run as run_molmospaces  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "MolmoSpaces is not importable. Install the submodule (`git submodule "
            "update --init --recursive`) or pass --dry-run to write a schema-valid "
            "synthetic dataset for pipeline tests."
        ) from e

    # MolmoSpaces takes its own config object; pass through.
    run_molmospaces(
        task_config=cfg,
        out_dir=out_dir,
        n_traj=args.n_traj,
        n_envs=args.n_envs,
        seed=args.seed,
        episode_writer=collect_episode,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()
