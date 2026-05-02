"""Per-channel mean/std for ToF, qpos, and actions.

Run once on the *training* split, save to ``stats.json``, apply at DataLoader
load time, and **unnormalize** at inference before the action chunk hits the
controller.

Why per-channel (not global)? Each ToF sensor has a slightly different
expected depth distribution because of where it is on the body. A single
global mean would over-shrink the wrist sensors (which see hand and object
all the time) and under-shrink the upper-arm sensors (which mostly stare into
free space). Per-channel = each sensor's MLP input occupies the same range.

CRITICAL: stats.json must be computed on training files only. If you compute
on the full dataset, val and test leak into the training stats and your
reported numbers are too good. Set ``--val-frac`` consistently across this
script and the DataLoader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("stats.json"))
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="fraction of files held out for val (NOT included in stats)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def compute_stats(data_dir: Path, val_frac: float = 0.1, seed: int = 0) -> dict:
    files = sorted(p for p in Path(data_dir).rglob("*.h5"))
    if not files:
        raise RuntimeError(f"no h5 files under {data_dir}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    n_val = max(1, int(len(files) * val_frac))
    train_idx = perm[n_val:]
    train_files = [files[i] for i in train_idx]

    tof_chunks: list[np.ndarray] = []
    qpos_chunks: list[np.ndarray] = []
    act_chunks: list[np.ndarray] = []
    n_sensors_seen: int | None = None

    for f in train_files:
        with h5py.File(f, "r") as h:
            for ep in h.keys():
                obs = h[f"{ep}/observations"]
                tof_chunks.append(obs["tof"][:])
                qpos_chunks.append(obs["qpos"][:])
                act_chunks.append(h[f"{ep}/actions"][:])
                if n_sensors_seen is None:
                    n_sensors_seen = int(obs["tof"].shape[1])

    tof = np.concatenate(tof_chunks, axis=0)        # [T, N, 8, 8]
    qpos = np.concatenate(qpos_chunks, axis=0)      # [T, 7]
    acts = np.concatenate(act_chunks, axis=0)       # [T, 7]

    stats = {
        "tof_mean": tof.mean(axis=0).tolist(),
        "tof_std": tof.std(axis=0).tolist(),
        "qpos_mean": qpos.mean(axis=0).tolist(),
        "qpos_std": qpos.std(axis=0).tolist(),
        "act_mean": acts.mean(axis=0).tolist(),
        "act_std": acts.std(axis=0).tolist(),
        "n_sensors": int(n_sensors_seen or 0),
        "n_episodes": len(train_files),
        "n_total_files": len(files),
        "val_frac": val_frac,
        "seed": seed,
        "train_files": [str(p) for p in train_files],
        "tof_min_mm": float(tof.min()),
        "tof_max_mm": float(tof.max()),
        "act_min": float(acts.min()),
        "act_max": float(acts.max()),
    }
    return stats


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x - mean) / (std + eps)


def unnormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


def main() -> None:
    args = parse_args()
    stats = compute_stats(args.data_dir, val_frac=args.val_frac, seed=args.seed)
    args.out.write_text(json.dumps(stats, indent=2))
    print(f"wrote {args.out} from {stats['n_episodes']} train files")
    print(f"  tof:  range [{stats['tof_min_mm']:.0f}, "
          f"{stats['tof_max_mm']:.0f}] mm, n_sensors={stats['n_sensors']}")
    print(f"  acts: range [{stats['act_min']:.3f}, {stats['act_max']:.3f}]")


if __name__ == "__main__":
    main()
