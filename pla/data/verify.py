"""Dataset sanity check.

Run this **before** every training start. It walks every HDF5 shard in the
data dir, validates the schema (delegating to ``pla.data.schema.validate``),
counts NaNs, and computes the proximity-informative fraction.

Targets (from PROJECT.md §3.3 / TIMELINE.md Day 2):

    * 0 NaN
    * schema_ok / total == 1.0
    * proximity-informative trajectories >= 30 % (any reading < 200 mm)

If any target is missed, the script exits non-zero so a CI pipeline / shell
script can short-circuit before training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from pla.data.schema import proximity_informative_fraction, validate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--threshold-mm", type=float, default=200.0)
    p.add_argument("--min-frac-prox-informative", type=float, default=0.30)
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero on any failure (recommended in CI).")
    return p.parse_args()


def verify_dataset(data_dir: Path, threshold_mm: float = 200.0) -> dict:
    files = sorted(p for p in Path(data_dir).rglob("*.h5"))
    if not files:
        return {"total": 0, "files": []}
    stats = {
        "total": 0,
        "success": 0,
        "nan_free": 0,
        "schema_ok": 0,
        "prox_informative": 0,
        "min_depth_mm": float("inf"),
        "max_depth_mm": -float("inf"),
        "T_lengths": [],
        "files": [str(p) for p in files],
        "errors": [],
    }
    n_close_steps = 0
    n_total_steps = 0

    for f in files:
        ok, errs = validate(f)
        if errs:
            stats["errors"].append({"file": str(f), "errors": errs})
        with h5py.File(f, "r") as h:
            for key in h.keys():
                try:
                    tof = h[f"{key}/observations/tof"][:]
                    qpos = h[f"{key}/observations/qpos"][:]
                    actions = h[f"{key}/actions"][:]
                    stats["total"] += 1
                    stats["T_lengths"].append(int(tof.shape[0]))
                    if not np.isnan(tof).any() and not np.isnan(qpos).any() and not np.isnan(actions).any():
                        stats["nan_free"] += 1
                    if ok:
                        stats["schema_ok"] += 1
                    informative = bool(np.any(tof < threshold_mm))
                    if informative:
                        stats["prox_informative"] += 1
                    n_total_steps += int(tof.shape[0])
                    n_close_steps += int(np.any(tof.reshape(tof.shape[0], -1) < threshold_mm, axis=1).sum())
                    stats["min_depth_mm"] = min(stats["min_depth_mm"], float(tof.min()))
                    stats["max_depth_mm"] = max(stats["max_depth_mm"], float(tof.max()))
                    if h[key].attrs.get("success", False):
                        stats["success"] += 1
                except Exception as e:  # noqa: BLE001
                    stats["errors"].append({"file": str(f), "ep": key, "errors": [str(e)]})

    stats["n_total_steps"] = n_total_steps
    stats["n_close_steps"] = n_close_steps
    stats["frac_steps_close"] = (n_close_steps / max(n_total_steps, 1))
    stats["frac_traj_close"] = (stats["prox_informative"] / max(stats["total"], 1))
    return stats


def print_report(stats: dict, threshold_mm: float, min_frac: float) -> bool:
    n = stats["total"]
    print("=" * 60)
    print(f"Episodes processed:       {n}")
    if n == 0:
        return False
    print(f"Schema OK:                {stats['schema_ok']}/{n}")
    print(f"NaN-free:                 {stats['nan_free']}/{n}")
    print(f"Successful:               {stats['success']}/{n}")
    print(f"Proximity-informative:    {stats['prox_informative']}/{n}"
          f" ({100*stats['frac_traj_close']:.1f}% of trajectories)")
    print(f"Frac steps with reading <{threshold_mm:.0f}mm: "
          f"{100*stats['frac_steps_close']:.1f}%")
    if stats["T_lengths"]:
        print(f"Mean episode length:      {np.mean(stats['T_lengths']):.0f} steps")
        print(f"Min/max ep length:        {min(stats['T_lengths'])} / {max(stats['T_lengths'])}")
    print(f"Depth range:              [{stats['min_depth_mm']:.0f}, {stats['max_depth_mm']:.0f}] mm")
    if stats["errors"]:
        print(f"\nERRORS ({len(stats['errors'])}):")
        for e in stats["errors"][:20]:
            print(f"  {e}")
    print("=" * 60)
    print(f"\nTarget: prox_informative trajectories >= {100*min_frac:.0f}%")
    print(f"Got:    {100*stats['frac_traj_close']:.1f}%")
    return (
        stats["frac_traj_close"] >= min_frac
        and stats["nan_free"] == n
        and stats["schema_ok"] == n
        and not stats["errors"]
    )


def main() -> None:
    args = parse_args()
    stats = verify_dataset(args.data_dir, threshold_mm=args.threshold_mm)
    ok = print_report(stats, args.threshold_mm, args.min_frac_prox_informative)
    if not ok and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
