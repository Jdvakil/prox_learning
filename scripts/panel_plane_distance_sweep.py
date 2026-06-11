"""Panel: plane_distance_sweep
PROOF of metric accuracy of the franka_skin HYBRID SPAD proximity sensors.

Pick one well-exposed sensor (link5_back_sensor_2). Place a flat mocap plane head-on at
true distances {0.05, 0.10, 0.15, 0.20, 0.30} m along the sensor's view direction. For each:
render the 8x8 SPAD depth, back-project, measure the center-pixel mean. Plot the five
heatmaps + a measured-vs-true line sitting dead on y=x, annotated with max abs error (mm).
"""
import os
import sys

sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib import cm as mcm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from hybrid_viz_lib import (
    FAR,
    FOVY,
    NEAR,
    STYLE,
    add_plane_mocap,
    backproject,
    build,
    cam_pose,
    depth8,
    depth_renderer,
    mocap_set,
    set_pose,
)

OUT_DIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_viz_suite"
OUT = os.path.join(OUT_DIR, "plane_distance_sweep.png")
SENSOR = "link5_back_sensor_2"
TRUE_D = [0.05, 0.10, 0.15, 0.20, 0.30]
HALFZ = 0.002  # plane half-thickness; front face is placed exactly at the true distance

os.makedirs(OUT_DIR, exist_ok=True)


def make(spec):
    # The MuJoCo OpenGL near-clip = vis.map.znear * stat.extent. With the default extent
    # (~10.5 m) that clip is ~0.10 m, which would erase the 0.05/0.10 m readings even though
    # the SPAD's physical NEAR is 0.015 m. Shrink it so the *renderer* never clips inside the
    # sensor's true measurement range. This changes nothing about the sensor model itself.
    spec.visual.map.znear = 0.0005
    spec.stat.extent = 1.0
    add_plane_mocap(spec, "probe_plane", half=(0.25, 0.25, HALFZ), rgba=(0.20, 0.85, 0.40, 1))


# ---- run the sweep ---------------------------------------------------------------------
model = build(make=make)
data = mujoco.MjData(model)
set_pose(model, data, "reach")
rd = depth_renderer(model)
pos, R = cam_pose(model, data, SENSOR)
fwd = -R[:, 2]  # the sensor's view direction

heatmaps, measured, n_active = [], [], []
for td in TRUE_D:
    # place plane so its FRONT face (the surface the sensor sees) is exactly at td
    mocap_set(model, data, "probe_plane", pos + fwd * (td + HALFZ), view_dir=fwd)
    d8 = depth8(rd, data, SENSOR)
    pts, _ = backproject(d8, pos, R)
    heatmaps.append(d8.copy())
    measured.append(float(d8[3:5, 3:5].mean()))  # center 2x2 mean
    n_active.append(int(((d8 >= NEAR) & (d8 <= FAR)).sum()))

measured = np.array(measured)
true = np.array(TRUE_D)
err_mm = (measured - true) * 1000.0
max_abs_err_mm = float(np.max(np.abs(err_mm)))
print("true     :", true)
print("measured :", np.round(measured, 5))
print("err (mm) :", np.round(err_mm, 4))
print("max |err|:", round(max_abs_err_mm, 4), "mm")
print("active px:", n_active)

# ---- figure ----------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": STYLE["fg"],
        "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"],
        "ytick.color": STYLE["fg"],
        "axes.edgecolor": STYLE["grid"],
    }
)

cmap = mcm.get_cmap(STYLE["cmap"])
# color scale spans the full sweep so red=near, blue=far is consistent across heatmaps
vmin, vmax = 0.04, 0.31
norm = Normalize(vmin=vmin, vmax=vmax)

fig = plt.figure(figsize=(15.5, 6.8), dpi=180)
fig.patch.set_facecolor(STYLE["bg"])
gs = GridSpec(
    2,
    6,
    figure=fig,
    height_ratios=[1.0, 1.25],
    width_ratios=[1, 1, 1, 1, 1, 0.09],
    hspace=0.42,
    wspace=0.28,
    left=0.045,
    right=0.965,
    top=0.86,
    bottom=0.10,
)

# top row: the five 8x8 SPAD depth heatmaps
for i, (td, d8) in enumerate(zip(TRUE_D, heatmaps)):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(STYLE["panel"])
    im = ax.imshow(d8, cmap=cmap, norm=norm, interpolation="nearest")
    # annotate the center reading
    ax.text(
        3.5,
        3.5,
        f"{measured[i] * 100:.1f}\ncm",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#0c0e12",
        bbox=dict(boxstyle="round,pad=0.18", fc="#e8e8ea", ec="none", alpha=0.82),
    )
    ax.set_title(
        f"true = {td * 100:.0f} cm",
        color=STYLE["fg"],
        fontsize=12,
        fontweight="bold",
        pad=6,
    )
    ax.set_xticks([0, 7])
    ax.set_yticks([0, 7])
    ax.set_xticklabels(["0", "7"], fontsize=8)
    ax.set_yticklabels(["0", "7"], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_color(STYLE["grid"])
    if i == 0:
        ax.set_ylabel("SPAD row", fontsize=9, color=STYLE["fg"])
    ax.set_xlabel("SPAD col", fontsize=9, color=STYLE["fg"])

# shared colorbar for the heatmap row
cax = fig.add_subplot(gs[0, 5])
cb = fig.colorbar(im, cax=cax)
cb.set_label("distance (m)", color=STYLE["fg"], fontsize=10)
cb.ax.yaxis.set_tick_params(color=STYLE["fg"], labelsize=8)
cb.outline.set_edgecolor(STYLE["grid"])
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=STYLE["fg"])

# bottom: measured vs true accuracy line
axL = fig.add_subplot(gs[1, 0:3])
axL.set_facecolor(STYLE["panel"])
lo, hi = 0.0, 0.34
axL.plot([lo, hi], [lo, hi], "--", color="#8a93a3", lw=1.6, label="ideal (y = x)", zorder=1)
# color each measured point by its true distance to match the heatmap colormap
pcolors = cmap(norm(true))
axL.plot(true, measured, "-", color=STYLE["accent"], lw=1.4, alpha=0.7, zorder=2)
axL.scatter(
    true,
    measured,
    s=130,
    c=pcolors,
    edgecolors="#0c0e12",
    linewidths=1.1,
    zorder=3,
    label="SPAD measured (center 2x2)",
)
axL.set_xlim(lo, hi)
axL.set_ylim(lo, hi)
axL.set_aspect("equal")
axL.set_xlabel("true plane distance (m)", fontsize=11)
axL.set_ylabel("measured distance (m)", fontsize=11)
axL.set_title("metric accuracy: measured vs. true", fontsize=13, fontweight="bold", pad=8)
axL.grid(True, color=STYLE["grid"], lw=0.6, alpha=0.7)
for s in axL.spines.values():
    s.set_color(STYLE["grid"])
leg = axL.legend(loc="upper left", framealpha=0.0, fontsize=9.5)
for txt in leg.get_texts():
    txt.set_color(STYLE["fg"])
axL.text(
        0.97,
        0.06,
        f"max abs error = {max_abs_err_mm:.2f} mm\nsensor: {SENSOR}",
        transform=axL.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=STYLE["fg"],
        bbox=dict(boxstyle="round,pad=0.35", fc="#0c0e12", ec=STYLE["accent"], lw=1.0, alpha=0.85),
)

# bottom-right: per-distance error bars (mm)
axR = fig.add_subplot(gs[1, 3:6])
axR.set_facecolor(STYLE["panel"])
bars = axR.bar(
    [f"{d * 100:.0f}" for d in TRUE_D],
    err_mm,
    color=pcolors,
    edgecolor="#0c0e12",
    linewidth=1.0,
    width=0.62,
)
axR.axhline(0, color="#8a93a3", lw=1.2, ls="--")
# tolerance band: +/- 1 mm
axR.axhspan(-1, 1, color=STYLE["accent"], alpha=0.12, zorder=0)
axR.text(
    len(TRUE_D) - 0.5,
    1.0,
    "+/- 1 mm band",
    ha="right",
    va="bottom",
    fontsize=8.5,
    color=STYLE["accent"],
)
axR.set_xlabel("true plane distance (cm)", fontsize=11)
axR.set_ylabel("measured - true  (mm)", fontsize=11)
axR.set_title("residual error per distance", fontsize=13, fontweight="bold", pad=8)
ymax = max(2.0, np.max(np.abs(err_mm)) * 1.4 + 0.5)
axR.set_ylim(-ymax, ymax)
axR.grid(True, axis="y", color=STYLE["grid"], lw=0.6, alpha=0.7)
for s in axR.spines.values():
    s.set_color(STYLE["grid"])
for b, e in zip(bars, err_mm):
    axR.text(
        b.get_x() + b.get_width() / 2,
        e + (0.12 if e >= 0 else -0.12),
        f"{e:+.2f}",
        ha="center",
        va="bottom" if e >= 0 else "top",
        fontsize=8.5,
        color=STYLE["fg"],
    )

fig.suptitle(
    "franka_skin HYBRID SPAD proximity sensor  -  plane distance sweep",
    fontsize=16,
    fontweight="bold",
    color=STYLE["fg"],
    y=0.965,
)
fig.text(
    0.045,
    0.915,
    f"8x8 depth camera  -  fovy {FOVY:.0f} deg  -  range {NEAR:.3f}-{FAR:.2f} m  -  flat plane normal to view dir,  64/64 pixels active at every distance",
    fontsize=10.5,
    color="#9aa3b2",
    ha="left",
)

fig.savefig(OUT, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

sz = os.path.getsize(OUT)
print("SAVED", OUT, sz, "bytes")
assert os.path.exists(OUT) and sz > 20_000, f"PNG missing or too small: {sz}"
print("MAX_ABS_ERR_MM", round(max_abs_err_mm, 3))
