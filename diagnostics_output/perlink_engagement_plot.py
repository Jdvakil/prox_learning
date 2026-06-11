#!/usr/bin/env python
"""Per-link skin engagement plot for franka_skin proximity episodes.

Panel A: For each scene and each link group (link2/link3/link5/link6),
         fraction of timesteps where ANY sensor on that link reads <8cm.
Panel B: Overall fraction of steps with any zone<8cm per scene (probe-4),
         with a 0.30 PASS line.
"""
import os
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260610_plots_v3"
os.makedirs(OUT_DIR, exist_ok=True)

RUNS = {
    "fumehood": "/home/jaydv/code/prox_learning/assets/datagen/fumehood_smoke/FrankaSkinFumehoodSmokeConfig/20260610_090517",
    "panel":    "/home/jaydv/code/prox_learning/assets/datagen/panel_slalom_smoke/FrankaSkinPanelSlalomSmokeConfig/20260610_092607",
    "cubby":    "/home/jaydv/code/prox_learning/assets/datagen/cubby_smoke/FrankaSkinCubbySmokeConfig/20260610_095739",
}

LINK_GROUPS = ["link2", "link3", "link5", "link6"]
NO_RETURN = 2.0   # values > 2.0 m mean no return -> mask
ENGAGE_M = 0.08   # 8 cm engagement threshold
PASS_LINE = 0.30

def sensor_mins(prox_arr):
    """prox_arr shape (T,4,8,8) -> (T,) min over the (8,8) grid, ignoring no-return.
    Returns array of per-timestep min distance (np.inf where all masked)."""
    # mean over the 4 sub-frames -> (T,8,8)
    grid = prox_arr.mean(axis=1)
    masked = np.where(grid > NO_RETURN, np.inf, grid)
    flat = masked.reshape(masked.shape[0], -1)
    return np.nanmin(np.where(np.isinf(flat), np.inf, flat), axis=1)

# Accumulators: per scene, per link -> [engaged_steps, total_steps]
link_counts = {sc: {lg: [0, 0] for lg in LINK_GROUPS} for sc in RUNS}
# per scene -> [any-zone-engaged steps, total steps]
overall_counts = {sc: [0, 0] for sc in RUNS}

for scene, run_dir in RUNS.items():
    files = sorted(glob.glob(os.path.join(run_dir, "house_*", "trajectories_batch_*.h5")))
    n_traj = 0
    for fp in files:
        try:
            f = h5py.File(fp, "r")
        except Exception:
            continue
        with f:
            for tname in f.keys():
                try:
                    t = f[tname]
                    prox = t["obs/proximity"]
                    sensor_names = list(prox.keys())
                    if not sensor_names:
                        continue
                    T = prox[sensor_names[0]].shape[0]

                    # per-sensor per-timestep min distance
                    per_sensor_min = {}
                    for sn in sensor_names:
                        per_sensor_min[sn] = sensor_mins(prox[sn][()])

                    # per-link engagement (any sensor on link < 8cm)
                    for lg in LINK_GROUPS:
                        lg_sensors = [sn for sn in sensor_names if sn.startswith(lg + "_")]
                        if not lg_sensors:
                            continue
                        stack = np.stack([per_sensor_min[sn] for sn in lg_sensors], axis=0)  # (n_sens, T)
                        any_eng = (stack < ENGAGE_M).any(axis=0)  # (T,)
                        link_counts[scene][lg][0] += int(any_eng.sum())
                        link_counts[scene][lg][1] += int(T)

                    # overall: any sensor (any link) < 8cm at that timestep
                    all_stack = np.stack([per_sensor_min[sn] for sn in sensor_names], axis=0)  # (29,T)
                    any_zone = (all_stack < ENGAGE_M).any(axis=0)
                    overall_counts[scene][0] += int(any_zone.sum())
                    overall_counts[scene][1] += int(T)
                    n_traj += 1
                except Exception:
                    continue
    print(f"[{scene}] processed {n_traj} trajectories from {len(files)} files")

# ---- compute fractions ----
scenes = list(RUNS.keys())
scene_colors = {"fumehood": "#1f77b4", "panel": "#ff7f0e", "cubby": "#2ca02c"}

link_frac = {sc: [] for sc in scenes}
for sc in scenes:
    for lg in LINK_GROUPS:
        eng, tot = link_counts[sc][lg]
        link_frac[sc].append(eng / tot if tot > 0 else 0.0)

overall_frac = {}
for sc in scenes:
    eng, tot = overall_counts[sc]
    overall_frac[sc] = eng / tot if tot > 0 else 0.0

# ---- plot ----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.0, 1.0]})

# Panel A: grouped bars, x = link group, bars = scene
x = np.arange(len(LINK_GROUPS))
n_sc = len(scenes)
width = 0.8 / n_sc
for i, sc in enumerate(scenes):
    offs = (i - (n_sc - 1) / 2.0) * width
    bars = axL.bar(x + offs, link_frac[sc], width, label=sc, color=scene_colors[sc], edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, link_frac[sc]):
        axL.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

axL.set_xticks(x)
axL.set_xticklabels([lg.replace("link", "link ") for lg in LINK_GROUPS])
axL.set_xlabel("Robot link group (skin sensor cluster)")
axL.set_ylabel("Fraction of timesteps with ANY sensor < 8 cm")
axL.set_title("Per-link skin engagement\n(how often each link's SPAD cluster sees an obstacle inside 8 cm)")
axL.set_ylim(0, max(0.05, max(max(v) for v in link_frac.values())) * 1.20 + 0.02)
axL.legend(title="Scene", framealpha=0.9)
axL.grid(axis="y", linestyle=":", alpha=0.5)

# Panel B: overall any-zone<8cm per scene with PASS line
xb = np.arange(len(scenes))
vals = [overall_frac[sc] for sc in scenes]
bars = axR.bar(xb, vals, 0.6, color=[scene_colors[sc] for sc in scenes], edgecolor="black", linewidth=0.5)
for b, v in zip(bars, vals):
    axR.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axR.axhline(PASS_LINE, color="red", linestyle="--", linewidth=1.8, label=f"PASS = {PASS_LINE:.2f}")
axR.set_xticks(xb)
axR.set_xticklabels(scenes)
axR.set_xlabel("Scene")
axR.set_ylabel("Fraction of steps with any zone < 8 cm")
axR.set_title("Overall skin engagement (probe-4)\nany of 29 sensors < 8 cm")
axR.set_ylim(0, max(PASS_LINE, max(vals)) * 1.25 + 0.05)
axR.legend(framealpha=0.9)
axR.grid(axis="y", linestyle=":", alpha=0.5)

fig.suptitle("Franka skin engagement: per-link clusters vs. overall (3 scenes pooled per scene)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_png = os.path.join(OUT_DIR, "perlink_engagement.png")
fig.savefig(out_png, dpi=150)
plt.close(fig)

sz = os.path.getsize(out_png)
print(f"SAVED {out_png} ({sz} bytes)")
print("overall_frac:", {k: round(v, 4) for k, v in overall_frac.items()})
print("link_frac:", {k: [round(x, 4) for x in v] for k, v in link_frac.items()})
