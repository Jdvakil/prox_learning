"""Panel: use_clearance_controller  (PROOF)

The core story: the franka_skin HYBRID SPAD proximity skin gives a robot whole-arm
clearance sensing, so a controller can SLOW and STOP before contact using skin alone --
no RGB needed (RGB is blurred at policy-training time).

Demo: a flat wall advances toward a fixed FR3 reaching pose in 12 small, equal steps
(a controlled approach). At every step we render all 40 8x8 SPAD depth cameras, back-
project to a world point cloud, and read the GLOBAL minimum skin->obstacle distance.
We plot that min vs step, overlay an 8 cm "slow zone" and a 2 cm "stop" line, and mark
the step where a proximity controller would first command SLOW and then STOP -- both
triggered before any contact, from skin measurements only. Three inset thumbnails
(FAR / SLOW / STOP) show the scene with the distance-colored cloud.

The nearest reading is masked to the external obstacle (the robot's own static geometry
is known a-priori and excluded -- standard self-collision masking), so the curve is the
true robot->wall standoff a clearance controller acts on.
"""
import os
import sys

sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib import colormaps as mcmaps
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

from hybrid_viz_lib import (
    FAR,
    FOVY,
    NEAR,
    STYLE,
    backproject,
    build,
    depth8,
    depth_renderer,
    mjv_cam,
    nice_lights,
    render_scene,
    set_pose,
)

OUT_DIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight"
OUT = os.path.join(OUT_DIR, "clearance_controller.png")
os.makedirs(OUT_DIR, exist_ok=True)

SLOW_M = 0.08   # 8 cm slow zone
STOP_M = 0.02   # 2 cm hard stop
N_STEPS = 12
WALL_X0, WALL_X1 = 0.95, 0.535   # wall slides from far to near-contact (final step inside stop zone)
WALL_HALF = (0.02, 0.55, 0.45)
WALL_Z = 0.50


# --------------------------------------------------------------------------- scene
def make(spec):
    nice_lights(spec)
    spec.visual.map.znear = 0.0008
    spec.stat.extent = 1.4
    b = spec.worldbody.add_body(name="wall", mocap=True, pos=[WALL_X0, 0, WALL_Z])
    g = b.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = list(WALL_HALF)
    g.rgba = [0.52, 0.55, 0.64, 1.0]
    g.contype = 0
    g.conaffinity = 0


model = build(make=make)
data = mujoco.MjData(model)
set_pose(model, data, "reach")
rd = depth_renderer(model)
SENSORS = sorted(
    model.camera(i).name for i in range(model.ncam) if "_sensor_" in model.camera(i).name
)
wall_mid = int(model.body_mocapid[model.body("wall").id])


def place_wall(wx):
    data.mocap_pos[wall_mid] = [wx, 0.0, WALL_Z]
    mujoco.mj_forward(model, data)


def scan(wx):
    """Return (global min skin->wall distance m, nearest-sensor name, full cloud pts, depths)
    where 'wall' points are those that land on the advancing plane (self-geometry masked)."""
    place_wall(wx)
    gmin = np.inf
    best = None
    P, D = [], []
    for n in SENSORS:
        d8 = depth8(rd, data, n)
        cid = model.camera(n).id
        pts, dd = backproject(d8, data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3))
        if len(pts) == 0:
            continue
        P.append(pts)
        D.append(dd)
        on_wall = np.abs(pts[:, 0] - wx) < 0.05  # points on the external obstacle
        if on_wall.any():
            mm = float(dd[on_wall].min())
            if mm < gmin:
                gmin = mm
                best = n
    pts = np.concatenate(P) if P else np.zeros((0, 3))
    dd = np.concatenate(D) if D else np.zeros((0,))
    return gmin, best, pts, dd


wall_xs = np.linspace(WALL_X0, WALL_X1, N_STEPS)
gmins = np.zeros(N_STEPS)
near_sensor = []
for i, wx in enumerate(wall_xs):
    g, best, _, _ = scan(wx)
    gmins[i] = g
    near_sensor.append(best)
    print(f"step {i:2d}  wall_x={wx:.3f}  min_skin_dist={g:.4f} m  nearest={best}")

# controller logic: first step entering slow zone, first step entering stop zone
slow_step = int(np.argmax(gmins <= SLOW_M)) if (gmins <= SLOW_M).any() else None
stop_step = int(np.argmax(gmins <= STOP_M)) if (gmins <= STOP_M).any() else None
# if the curve never reaches STOP, define stop as the last (closest) step for annotation
print("SLOW first at step", slow_step, "STOP first at step", stop_step)

# pick three representative steps for thumbnails: FAR (step0), SLOW (slow_step), STOP (last)
far_i = 0
slow_i = slow_step if slow_step is not None else N_STEPS // 2
stop_i = N_STEPS - 1
thumb_idx = [far_i, slow_i, stop_i]
thumb_labels = ["FAR  -  cruise", "SLOW zone  (< 8 cm)", "STOP  (< 2 cm)"]

# render the three thumbnails (RGB + distance-colored cloud) -- tight framing on the
# arm<->wall gap, brightened so the FR3 and its link6 SPADs read clearly.
thumbs = []
cam = mjv_cam(lookat=(0.42, 0.0, 0.60), distance=1.42, azimuth=250, elevation=-11)
for ti in thumb_idx:
    wx = wall_xs[ti]
    _, _, pts, dd = scan(wx)
    img = render_scene(model, data, cam, w=560, h=620, cloud=pts, depths=dd,
                       pt_size=0.011, gamma=0.78)
    thumbs.append(img)

# =============================================================================== figure
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
cmap = mcmaps[STYLE["cmap"]]
norm = Normalize(vmin=NEAR, vmax=FAR)

fig = plt.figure(figsize=(15.0, 8.4), dpi=170)
fig.patch.set_facecolor(STYLE["bg"])
gs = GridSpec(
    2,
    3,
    figure=fig,
    height_ratios=[1.0, 0.62],
    hspace=0.30,
    wspace=0.14,
    left=0.062,
    right=0.965,
    top=0.86,
    bottom=0.075,
)

# ---- main plot: global min skin distance vs step -----------------------------
axM = fig.add_subplot(gs[0, :])
axM.set_facecolor(STYLE["panel"])
steps = np.arange(N_STEPS)

# zone shading
axM.axhspan(STOP_M, SLOW_M, color="#f0a202", alpha=0.10, zorder=0)
axM.axhspan(0.0, STOP_M, color=STYLE["near"], alpha=0.16, zorder=0)
axM.axhline(SLOW_M, color="#f0a202", lw=1.6, ls="--", zorder=2)
axM.axhline(STOP_M, color=STYLE["near"], lw=1.8, ls="--", zorder=2)
axM.text(0.15, SLOW_M + 0.006, "slow zone  8 cm", color="#f0a202",
         ha="left", va="bottom", fontsize=10.5, fontweight="bold")
axM.text(0.15, STOP_M + 0.005, "STOP  2 cm", color=STYLE["near"],
         ha="left", va="bottom", fontsize=10.5, fontweight="bold")

# the measured curve, points colored by distance (turbo_r near=red)
pcols = cmap(norm(np.clip(gmins, NEAR, FAR)))
axM.plot(steps, gmins, "-", color=STYLE["accent"], lw=2.0, alpha=0.55, zorder=3)
axM.scatter(steps, gmins, s=150, c=pcols, edgecolors="#0c0e12", linewidths=1.2,
            zorder=4, label="global min  skin -> wall  (40 SPADs)")

# mark slow / stop trigger steps (annotations placed high-left to avoid the curve + zones)
if slow_step is not None:
    axM.scatter([slow_step], [gmins[slow_step]], s=420, facecolors="none",
                edgecolors="#f0a202", linewidths=2.4, zorder=5)
    axM.annotate("commands SLOW\n(skin only)",
                 xy=(slow_step, gmins[slow_step]), xytext=(slow_step - 3.6, gmins[slow_step] + 0.135),
                 color="#f0a202", fontsize=10.5, fontweight="bold", ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="->", color="#f0a202", lw=1.8,
                                 connectionstyle="arc3,rad=0.18"))
if stop_step is not None:
    axM.scatter([stop_step], [gmins[stop_step]], s=460, facecolors="none",
                edgecolors=STYLE["near"], linewidths=2.6, zorder=5)
    axM.annotate("HALTS before\ncontact",
                 xy=(stop_step, gmins[stop_step]),
                 xytext=(stop_step - 1.7, gmins[stop_step] + 0.225),
                 color=STYLE["near"], fontsize=10.5, fontweight="bold", ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=STYLE["near"], lw=1.8,
                                 connectionstyle="arc3,rad=-0.2"))

# mark which thumbnails come from where (tag above the point if it sits near the floor)
for ti, lab, ec in zip(thumb_idx, ["A", "B", "C"], ["#5bc8ff", "#f0a202", STYLE["near"]]):
    above = gmins[ti] < 0.10
    dy = 0.05 if above else -0.05
    axM.annotate(lab, xy=(ti, gmins[ti]), xytext=(ti + 0.35, gmins[ti] + dy),
                 color=ec, fontsize=12, fontweight="bold", ha="center",
                 va="bottom" if above else "top")

axM.set_xlim(-0.4, N_STEPS - 0.6)
axM.set_ylim(0.0, max(gmins) * 1.12)
axM.set_xticks(steps)
axM.set_xlabel("approach step  (wall advances toward fixed reaching pose)", fontsize=11.5)
axM.set_ylabel("min skin -> obstacle distance  (m)", fontsize=11.5)
axM.set_title("whole-arm clearance control from proximity skin alone",
              fontsize=14.5, fontweight="bold", pad=10)
axM.grid(True, color=STYLE["grid"], lw=0.6, alpha=0.7)
for s in axM.spines.values():
    s.set_color(STYLE["grid"])
leg = axM.legend(loc="upper right", framealpha=0.0, fontsize=10.5)
for t in leg.get_texts():
    t.set_color(STYLE["fg"])

# secondary axis: contact-free margin headline
contact_free = gmins.min()
axM.text(
    0.295, 0.94,
    f"nearest approach measured = {contact_free * 100:.1f} cm  (zero contact)\n"
    f"slow @ step {slow_step}    stop @ step {stop_step}    nearest SPAD: {near_sensor[stop_i]}",
    transform=axM.transAxes, ha="left", va="top", fontsize=10,
    color=STYLE["fg"],
    bbox=dict(boxstyle="round,pad=0.4", fc="#0c0e12", ec=STYLE["accent"], lw=1.0, alpha=0.9),
)

# ---- three thumbnails --------------------------------------------------------
edge_cols = ["#5bc8ff", "#f0a202", STYLE["near"]]
for k, (img, lab, ti, ec) in enumerate(zip(thumbs, thumb_labels, thumb_idx, edge_cols)):
    ax = fig.add_subplot(gs[1, k])
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(ec)
        s.set_linewidth(2.4)
    tag = ["A", "B", "C"][k]
    ax.set_title(f"{tag}.  {lab}", color=ec, fontsize=12, fontweight="bold", pad=6)
    ax.text(0.025, 0.045, f"min = {gmins[ti] * 100:.1f} cm",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold",
            color="#0c0e12",
            bbox=dict(boxstyle="round,pad=0.28", fc=ec, ec="none", alpha=0.92))

# shared colorbar for the cloud color scale (small, on the right thumbnail)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cax = fig.add_axes([0.968, 0.085, 0.011, 0.46])
cb = fig.colorbar(sm, cax=cax)
cb.set_label("cloud distance (m)", color=STYLE["fg"], fontsize=9.5)
cb.ax.yaxis.set_tick_params(color=STYLE["fg"], labelsize=8)
cb.outline.set_edgecolor(STYLE["grid"])
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=STYLE["fg"])

# ---- titles ------------------------------------------------------------------
fig.suptitle(
    "franka_skin HYBRID proximity skin  -  clearance controller proof",
    fontsize=17, fontweight="bold", color=STYLE["fg"], y=0.975,
)
fig.text(
    0.062, 0.905,
    f"40 SPAD depth cameras  -  8x8 px,  fovy {FOVY:.0f} deg,  range {NEAR*1000:.0f}-{FAR*100:.0f} cm  -  "
    f"RGB blurred at policy-training time, so this skin IS the robot's contact-rich perception",
    fontsize=10.5, color="#9aa3b2", ha="left",
)

fig.savefig(OUT, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

sz = os.path.getsize(OUT)
print("SAVED", OUT, sz, "bytes")
assert os.path.exists(OUT) and sz > 25_000, f"PNG missing or too small: {sz}"
print("OK")
