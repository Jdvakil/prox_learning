"""Clutter-vs-skin-activation plot from collected fume-hood episodes.

Answers "is the clutter detected by the sensors?" quantitatively: for every
trajectory, counts skin sensors whose minimum depth dips below a threshold and
plots that against the number of clutter items the sampler actually placed
(``n_clutter_placed``, logged in ``scene_params``).

Usage (after any collection run from this package):

    python fumehood_env/analysis/plot_clutter_activation.py \
        --data '<output_dir>/house_*/trajectories_batch_*.h5' \
        --out clutter_activation.png
"""
from __future__ import annotations

import argparse
import glob
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="glob for trajectories_batch_*.h5")
p.add_argument("--out", default="clutter_activation.png")
p.add_argument("--near", type=float, default=0.30,
               help="a sensor counts as active if its min depth < this (m)")
args = p.parse_args()

rows = []
for path in sorted(glob.glob(args.data)):
    with h5py.File(path, "r") as f:
        for traj in f:
            g = f[traj]
            if "obs/proximity" not in g or "obs_scene" not in g:
                continue
            raw = g["obs_scene"][()]
            scene = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
            sp = scene.get("scene_params", {})
            n_clutter = sp.get("n_clutter_placed")
            if n_clutter is None:
                continue
            prox = g["obs/proximity"]
            active = 0
            for sensor in prox:
                depth = np.asarray(prox[sensor])          # (T, sub, 8, 8)
                valid = depth[depth > 0.005]              # drop dead pixels
                if valid.size and float(valid.min()) < args.near:
                    active += 1
            rows.append((int(n_clutter), active, len(list(prox))))

if not rows:
    raise SystemExit("no trajectories with proximity + scene_params found")

n_clutter = np.array([r[0] for r in rows])
n_active = np.array([r[1] for r in rows])
n_sensors = rows[0][2]

fig, ax = plt.subplots(figsize=(7, 4.5))
levels = sorted(set(n_clutter))
ax.boxplot([n_active[n_clutter == c] for c in levels], positions=levels, widths=0.6)
jitter = (np.random.rand(len(n_clutter)) - 0.5) * 0.25
ax.plot(n_clutter + jitter, n_active, ".", alpha=0.45, ms=5)
r = float(np.corrcoef(n_clutter, n_active)[0, 1]) if len(levels) > 1 else float("nan")
ax.set_xlabel("clutter items placed (scene_params.n_clutter_placed)")
ax.set_ylabel(f"skin sensors active (min depth < {args.near:.2f} m)")
ax.set_title(f"Clutter is seen by the skin — {len(rows)} episodes, "
             f"{n_sensors} sensors, r = {r:.2f}")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.out, dpi=130)
print(f"wrote {args.out}   episodes={len(rows)}  pearson_r={r:.3f}")
