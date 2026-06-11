#!/usr/bin/env python
"""Behavior + outcome composition across all three scenes.

Panel A (per scene): stacked bar of cell x behavior_class counts.
Panel B: success rate per scene, decomposed into credited aborts vs real grasps.
"""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import h5py

OUTDIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260610_plots_v3"
os.makedirs(OUTDIR, exist_ok=True)

RUNS = {
    "fumehood": "/home/jaydv/code/prox_learning/assets/datagen/fumehood_smoke/FrankaSkinFumehoodSmokeConfig/20260610_090517",
    "panel":    "/home/jaydv/code/prox_learning/assets/datagen/panel_slalom_smoke/FrankaSkinPanelSlalomSmokeConfig/20260610_092607",
    "cubby":    "/home/jaydv/code/prox_learning/assets/datagen/cubby_smoke/FrankaSkinCubbySmokeConfig/20260610_095739",
}

CELLS = ["free", "hidden", "visible", "abort"]
BEHAVIORS = ["free", "deflect", "abort"]


def decode_blob(ds):
    blob = ds[()] if ds.shape == () else ds[0]
    s = blob.tobytes() if hasattr(blob, "tobytes") else blob
    if isinstance(s, bytes):
        s = s.decode("utf-8", "ignore")
    return s.rstrip("\x00")


def load_scene(run_dir):
    """Return list of dicts: {cell, behavior, success} for each episode."""
    recs = []
    files = sorted(glob.glob(os.path.join(run_dir, "house_*", "trajectories_batch_*.h5")))
    for f in files:
        try:
            h = h5py.File(f, "r")
        except Exception as e:
            print("  skip file", f, e)
            continue
        with h:
            for tk in h.keys():
                try:
                    t = h[tk]
                    sc = json.loads(decode_blob(t["obs_scene"]))
                    cell = sc.get("scene_params", {}).get("cell")
                    beh = sc.get("behavior_class")
                    succ = bool(np.asarray(t["success"])[-1])
                    recs.append({"cell": cell, "behavior": beh, "success": succ})
                except Exception as e:
                    print("  skip traj", f, tk, e)
                    continue
    return recs


# ---- gather ----
data = {scene: load_scene(rd) for scene, rd in RUNS.items()}
scenes = list(RUNS.keys())
for s in scenes:
    print(f"{s}: {len(data[s])} episodes")

# ---- figure ----
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6.2))

# Panel A: per scene, stacked bar over cell categories, segments = behavior_class.
# Group bars by scene; within scene one bar per cell; stack by behavior.
beh_colors = {"free": "#2c7fb8", "deflect": "#f4a300", "abort": "#d7301f"}

n_scenes = len(scenes)
n_cells = len(CELLS)
group_w = 0.8
bar_w = group_w / n_cells
x_scene = np.arange(n_scenes)

for ci, cell in enumerate(CELLS):
    offset = (ci - (n_cells - 1) / 2) * bar_w
    bottoms = np.zeros(n_scenes)
    for beh in BEHAVIORS:
        heights = []
        for s in scenes:
            cnt = sum(1 for r in data[s] if r["cell"] == cell and r["behavior"] == beh)
            heights.append(cnt)
        heights = np.array(heights, dtype=float)
        axA.bar(x_scene + offset, heights, bar_w * 0.92, bottom=bottoms,
                color=beh_colors[beh], edgecolor="white", linewidth=0.4)
        bottoms += heights
    # annotate total above each cell bar + cell label
    for si in range(n_scenes):
        tot = int(bottoms[si])
        if tot > 0:
            axA.text(x_scene[si] + offset, bottoms[si] + 0.15, str(tot),
                     ha="center", va="bottom", fontsize=8, fontweight="bold")
        axA.text(x_scene[si] + offset, -0.6, cell, ha="center", va="top",
                 fontsize=7.5, rotation=90, color="#444")

axA.set_xticks(x_scene)
axA.set_xticklabels([s + f"\n(n={len(data[s])})" for s in scenes], fontsize=10)
axA.set_ylabel("Episode count", fontsize=11)
axA.set_title("Scene cell-type x policy behavior_class composition\n"
              "(bars grouped by scene; sub-bars = cell type; color = behavior)",
              fontsize=11)
ymaxA = max((sum(1 for r in data[s] if r["cell"] == c) for s in scenes for c in CELLS), default=1)
axA.set_ylim(-2.0, ymaxA * 1.18)
axA.grid(axis="y", alpha=0.3, linestyle=":")
legA = [Patch(facecolor=beh_colors[b], edgecolor="white", label=f"behavior: {b}") for b in BEHAVIORS]
axA.legend(handles=legA, loc="upper right", fontsize=9, framealpha=0.9, title="policy behavior_class")

# Panel B: success rate per scene, decomposed credited-abort vs real-grasp.
# credited abort = behavior=="abort" AND success
# real grasp     = behavior in (free, deflect) AND success
n_total = [len(data[s]) for s in scenes]
n_succ = [sum(1 for r in data[s] if r["success"]) for s in scenes]
n_abort_succ = [sum(1 for r in data[s] if r["success"] and r["behavior"] == "abort") for s in scenes]
n_grasp_succ = [sum(1 for r in data[s] if r["success"] and r["behavior"] in ("free", "deflect")) for s in scenes]

succ_rate = [ (ns / nt * 100.0) if nt else 0.0 for ns, nt in zip(n_succ, n_total) ]

xb = np.arange(n_scenes)
bw = 0.55
# stacked counts: grasp successes (bottom) + abort successes (top)
b1 = axB.bar(xb, n_grasp_succ, bw, color="#1a9850", edgecolor="white",
             label="real grasp success (free/deflect)")
b2 = axB.bar(xb, n_abort_succ, bw, bottom=n_grasp_succ, color="#8073ac", edgecolor="white",
             label="credited abort success")

# annotate segment counts
for si in range(n_scenes):
    if n_grasp_succ[si] > 0:
        axB.text(xb[si], n_grasp_succ[si] / 2, str(n_grasp_succ[si]),
                 ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    if n_abort_succ[si] > 0:
        axB.text(xb[si], n_grasp_succ[si] + n_abort_succ[si] / 2, str(n_abort_succ[si]),
                 ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    # total successes + success rate above bar
    top = n_grasp_succ[si] + n_abort_succ[si]
    axB.text(xb[si], top + 0.25,
             f"{n_succ[si]}/{n_total[si]}\n{succ_rate[si]:.0f}% succ",
             ha="center", va="bottom", fontsize=9, fontweight="bold", color="#222")

axB.set_xticks(xb)
axB.set_xticklabels(scenes, fontsize=10)
axB.set_ylabel("Successful episodes (count)", fontsize=11)
axB.set_title("Outcome composition: success decomposed into\ncredited aborts vs real grasps",
              fontsize=11)
axB.grid(axis="y", alpha=0.3, linestyle=":")
ymax_b = max((g + a for g, a in zip(n_grasp_succ, n_abort_succ)), default=1)
axB.set_ylim(0, ymax_b * 1.30 + 1)
axB.legend(loc="upper left", fontsize=9, framealpha=0.9)

fig.suptitle("Franka-skin proximity dataset: behavior + outcome composition (all three scenes)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])

out = os.path.join(OUTDIR, "behavior_composition.png")
fig.savefig(out, dpi=150)
plt.close(fig)

sz = os.path.getsize(out)
print("WROTE", out, sz, "bytes")
print("PANEL_B grasp_succ", n_grasp_succ, "abort_succ", n_abort_succ, "rate", [round(r,1) for r in succ_rate])
