"""Load native pact_place skin tensors from ``rows/*/trajectory.h5``."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def row_dirs(src: Path) -> list[Path]:
    rows = src / "rows" if (src / "rows").is_dir() else src
    dirs = [p for p in rows.iterdir() if p.is_dir() and (p / "trajectory.h5").is_file()]
    dirs.sort(key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(f"no rows/*/trajectory.h5 under {src}")
    return dirs


def stack_native_proximity(grp, sensor_order: list[str]) -> np.ndarray:
    """``obs/proximity/<sensor>`` -> ``(T, S, 4, 8, 8)`` or ``(T, S, 8, 8)`` float32 metres."""
    prox = grp["obs/proximity"]
    chans = []
    for name in sensor_order:
        if name not in prox:
            raise KeyError(f"missing sensor {name!r} in {list(prox.keys())[:8]}")
        chans.append(np.asarray(prox[name], dtype=np.float32))
    return np.stack(chans, axis=1)


def load_episode_proximity(row: Path, sensor_order: list[str]) -> np.ndarray:
    import h5py

    with h5py.File(row / "trajectory.h5", "r") as handle:
        return stack_native_proximity(handle["traj_0"], sensor_order)
