#!/usr/bin/env python
"""Parameter coverage histograms pooled over all scenes.

Reads scene_params from obs_scene JSON blobs across all trajectories in the
three franka_skin smoke run dirs (fumehood, panel, cubby) and plots coverage
histograms for: clearance(cm), residual_margin(cm), depth(m), light_scale(log),
target_frac.
"""
import os
import glob
import json

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260610_plots_v3"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "param_coverage.png")

RUN_DIRS = {
    "fumehood": "/home/jaydv/code/prox_learning/assets/datagen/fumehood_smoke/FrankaSkinFumehoodSmokeConfig/20260610_090517",
    "panel":    "/home/jaydv/code/prox_learning/assets/datagen/panel_slalom_smoke/FrankaSkinPanelSlalomSmokeConfig/20260610_092607",
    "cubby":    "/home/jaydv/code/prox_learning/assets/datagen/cubby_smoke/FrankaSkinCubbySmokeConfig/20260610_095739",
}


def decode_scene(traj):
    """Decode obs_scene JSON blob from a trajectory group."""
    blob = traj["obs_scene"][()]
    if getattr(blob, "shape", ()) != ():
        blob = blob[0]
    s = (blob.tobytes() if hasattr(blob, "tobytes") else blob)
    if isinstance(s, bytes):
        s = s.decode("utf-8", "ignore")
    s = s.rstrip("\x00")
    return json.loads(s)


# Accumulators
clearance_cm = []
residual_cm = []
depth_m = []
light_scale = []
target_frac = []

n_traj = 0
n_err = 0

for scene, run_dir in RUN_DIRS.items():
    h5_files = sorted(glob.glob(os.path.join(run_dir, "house_*", "trajectories_batch_*.h5")))
    for h5_path in h5_files:
        try:
            f = h5py.File(h5_path, "r")
        except Exception:
            continue
        with f:
            for key in f.keys():
                if not key.startswith("traj_"):
                    continue
                try:
                    traj = f[key]
                    scn = decode_scene(traj)
                    sp = scn.get("scene_params", {})
                    n_traj += 1

                    c = sp.get("clearance", None)
                    if c is not None and np.isfinite(c):
                        clearance_cm.append(float(c) * 100.0)

                    rm = sp.get("residual_margin", None)
                    if rm is not None and np.isfinite(rm):
                        residual_cm.append(float(rm) * 100.0)

                    d = sp.get("depth", None)
                    if d is not None and np.isfinite(d):
                        depth_m.append(float(d))

                    ls = sp.get("light_scale", None)
                    if ls is not None and np.isfinite(ls) and ls > 0:
                        light_scale.append(float(ls))

                    tf = sp.get("target_frac", None)
                    if tf is not None and np.isfinite(tf):
                        target_frac.append(float(tf))
                except Exception:
                    n_err += 1
                    continue

print(f"trajectories parsed: {n_traj}, errors skipped: {n_err}")
print(f"clearance n={len(clearance_cm)}, residual n={len(residual_cm)}, "
      f"depth n={len(depth_m)}, light n={len(light_scale)}, target_frac n={len(target_frac)}")

clearance_cm = np.asarray(clearance_cm, dtype=float)
residual_cm = np.asarray(residual_cm, dtype=float)
depth_m = np.asarray(depth_m, dtype=float)
light_scale = np.asarray(light_scale, dtype=float)
target_frac = np.asarray(target_frac, dtype=float)


def annot(ax, data, unit, log=False):
    """Add min/median/max annotation and reference lines."""
    if data.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    vmin, vmed, vmax = np.min(data), np.median(data), np.max(data)
    txt = (f"min  = {vmin:.3g} {unit}\n"
           f"med  = {vmed:.3g} {unit}\n"
           f"max  = {vmax:.3g} {unit}\n"
           f"n    = {data.size}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.axvline(vmed, color="crimson", ls="--", lw=1.5, label=f"median = {vmed:.3g}")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)


fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Scene parameter coverage (pooled: fumehood + panel + cubby)\n"
             f"{n_traj} trajectories sampled", fontsize=15, fontweight="bold")

BAR = "#4C72B0"

# 1. clearance (cm)
ax = axes[0, 0]
if clearance_cm.size:
    ax.hist(clearance_cm, bins=30, color=BAR, edgecolor="white")
annot(ax, clearance_cm, "cm")
ax.set_title("Clearance", fontweight="bold")
ax.set_xlabel("clearance (cm)")
ax.set_ylabel("count (trajectories)")
ax.grid(axis="y", alpha=0.3)

# 2. residual_margin (cm, NaN dropped)
ax = axes[0, 1]
if residual_cm.size:
    ax.hist(residual_cm, bins=30, color=BAR, edgecolor="white")
annot(ax, residual_cm, "cm")
ax.set_title("Residual margin (NaN dropped = free cells)", fontweight="bold")
ax.set_xlabel("residual margin (cm)")
ax.set_ylabel("count (trajectories)")
ax.grid(axis="y", alpha=0.3)

# 3. depth (m)
ax = axes[0, 2]
if depth_m.size:
    ax.hist(depth_m, bins=30, color=BAR, edgecolor="white")
annot(ax, depth_m, "m")
ax.set_title("Cell depth", fontweight="bold")
ax.set_xlabel("depth (m)")
ax.set_ylabel("count (trajectories)")
ax.grid(axis="y", alpha=0.3)

# 4. light_scale (log x)
ax = axes[1, 0]
if light_scale.size:
    lo, hi = light_scale.min(), light_scale.max()
    if lo <= 0:
        lo = light_scale[light_scale > 0].min()
    bins = np.logspace(np.log10(lo), np.log10(hi), 30) if hi > lo else 30
    ax.hist(light_scale, bins=bins, color=BAR, edgecolor="white")
    ax.set_xscale("log")
annot(ax, light_scale, "x")
ax.set_title("Light scale (log x)", fontweight="bold")
ax.set_xlabel("light_scale (relative, log scale)")
ax.set_ylabel("count (trajectories)")
ax.grid(axis="y", alpha=0.3)

# 5. target_frac
ax = axes[1, 1]
if target_frac.size:
    ax.hist(target_frac, bins=30, color=BAR, edgecolor="white")
annot(ax, target_frac, "")
ax.set_title("Target fraction", fontweight="bold")
ax.set_xlabel("target_frac (fraction of depth)")
ax.set_ylabel("count (trajectories)")
ax.grid(axis="y", alpha=0.3)

# 6. unused panel -> summary text
ax = axes[1, 2]
ax.axis("off")
summary = (
    "Parameter sampling spans\n"
    "-------------------------\n"
)
def rng(name, data, unit):
    if data.size == 0:
        return f"{name}: no data\n"
    return f"{name}: [{np.min(data):.3g}, {np.max(data):.3g}] {unit}  (n={data.size})\n"
summary += rng("clearance", clearance_cm, "cm")
summary += rng("residual", residual_cm, "cm")
summary += rng("depth", depth_m, "m")
summary += rng("light", light_scale, "x")
summary += rng("targ_frac", target_frac, "")
ax.text(0.02, 0.98, summary, transform=ax.transAxes, ha="left", va="top",
        fontsize=11, family="monospace",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.6"))

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_PNG, dpi=150)
plt.close(fig)

size = os.path.getsize(OUT_PNG)
print(f"WROTE {OUT_PNG} ({size} bytes)")
print("OK" if size > 5120 else "TOO_SMALL")
