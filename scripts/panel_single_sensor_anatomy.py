"""single_sensor_anatomy: educational 3-panel figure explaining ONE SPAD proximity sensor.
(1) 3D render of arm with this sensor's FOV cone + its 64 back-projected points
(2) raw 8x8 depth heatmap with per-cell distance text
(3) the back-projected 64-point patch lying on the probe plane in 3D
"""
import os, sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, sensors, depth8, backproject, cam_pose,
    add_plane_mocap, mocap_set, depth_renderer, fit_plane, FOVY, NEAR, FAR, STYLE)
import mujoco, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_viz_suite"
os.makedirs(OUT, exist_ok=True)
KEY = "single_sensor_anatomy"
SENSOR = "link4_sensor_0"
PLANE_D = 0.15          # plane distance along sensor forward dir
CONE_LEN = 0.30         # draw FOV cone lines out to 0.3 m

# ---- scene: arm + probe plane placed in front of the chosen sensor -------------
model = build(make=lambda s: add_plane_mocap(s, half=(0.10, 0.10, 0.004),
                                             rgba=(0.30, 0.78, 0.55, 0.92)),
              offw=1400, offh=1400)
data = mujoco.MjData(model)
set_pose(model, data, "reach")
rd = depth_renderer(model)

pos, R = cam_pose(model, data, SENSOR)
fwd = -R[:, 2]                      # camera looks down -z of its frame
right = R[:, 0]
up = R[:, 1]
mocap_set(model, data, "probe_plane", pos + fwd * PLANE_D, view_dir=fwd)

# ---- read the sensor --------------------------------------------------------
d8 = depth8(rd, data, SENSOR)                      # (8,8) meters
pts, depths = backproject(d8, pos, R)              # 64 world points + depths
c, n, rms = fit_plane(pts)
n_ret = int(((d8 >= NEAR) & (d8 <= FAR)).sum())
dmean = float(depths.mean())
dmin, dmax = float(depths.min()), float(depths.max())
print(f"{SENSOR}: returns={n_ret}/64  d=[{dmin:.4f},{dmax:.4f}] mean={dmean:.4f}  fit_rms={rms*1000:.4f}mm")

# color scheme keyed to distance (turbo_r: near=red, far=blue)
cmap = matplotlib.colormaps[STYLE["cmap"]]
vmin, vmax = NEAR, FAR
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# ---- FOV cone corner rays (4 corners of the 8x8 sensor at CONE_LEN) ----------
half = np.tan(np.deg2rad(FOVY / 2.0)) * CONE_LEN
corners_cam = np.array([
    [-half, -half, -CONE_LEN],
    [ half, -half, -CONE_LEN],
    [ half,  half, -CONE_LEN],
    [-half,  half, -CONE_LEN],
])
corners_world = (R @ corners_cam.T).T + pos   # (4,3)

# =============================================================================
# PANEL 1: 3D RGB render of the arm with FOV cone + highlighted points
# =============================================================================
H, W = 1100, 1100
rcol = mujoco.Renderer(model, H, W)
cam = mujoco.MjvCamera()
cam.lookat = pos + fwd * 0.085          # frame tight on the sensor + its FOV cone
cam.distance = 0.52
cam.azimuth = 28
cam.elevation = -10
rcol.update_scene(data, cam)
scn = rcol.scene

def add_line(p0, p1, rgba, width=0.0016):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    p0 = np.asarray(p0, np.float64); p1 = np.asarray(p1, np.float64)
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.eye(3).ravel(), np.array(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, p0, p1)
    scn.ngeom += 1

def add_sphere(p, rgba, rad=0.006):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, 0, 0]),
                        np.asarray(p, np.float64), np.eye(3).ravel(),
                        np.array(rgba, np.float32))
    scn.ngeom += 1

# FOV cone: sensor origin -> 4 corners, plus the far rectangle outline
cone_rgba = (0.30, 0.79, 0.94, 0.85)   # cyan accent
for cw in corners_world:
    add_line(pos, cw, cone_rgba, width=0.0018)
for i in range(4):
    add_line(corners_world[i], corners_world[(i + 1) % 4], cone_rgba, width=0.0016)
# central axis ray
add_line(pos, pos + fwd * CONE_LEN, (1.0, 1.0, 1.0, 0.55), width=0.0011)
# sensor origin marker
add_sphere(pos, (1.0, 0.95, 0.30, 1.0), rad=0.009)

# the 64 back-projected points, colored by distance
for p, dd in zip(pts, depths):
    rgba = cmap(norm(dd))
    add_sphere(p, (rgba[0], rgba[1], rgba[2], 1.0), rad=0.0072)

img = rcol.render()

# =============================================================================
# FIGURE LAYOUT
# =============================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
    "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
    "axes.edgecolor": STYLE["grid"],
})
fig = plt.figure(figsize=(18, 6.6), dpi=170)
fig.patch.set_facecolor(STYLE["bg"])
gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 0.92, 1.0],
                      left=0.012, right=0.985, top=0.80, bottom=0.075, wspace=0.16)

# ---- Panel 1: 3D render -----------------------------------------------------
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(img)
ax0.set_facecolor(STYLE["panel"])
ax0.set_xticks([]); ax0.set_yticks([])
for sp in ax0.spines.values():
    sp.set_edgecolor(STYLE["accent"]); sp.set_linewidth(1.2)
ax0.set_title("1 - sensor in situ on the arm\nFOV cone + 64 returns",
              color=STYLE["fg"], fontsize=13, pad=8, loc="left")
# annotation legend inside panel
ax0.text(0.022, 0.965, "yellow = SPAD origin", transform=ax0.transAxes,
         color="#ffe54d", fontsize=10.5, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.3", fc="#0c0e12", ec="#3a3f49", alpha=0.85))
ax0.text(0.022, 0.905, "cyan = 45 deg FOV cone (to 0.30 m)", transform=ax0.transAxes,
         color="#4dc9f0", fontsize=10.5, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.3", fc="#0c0e12", ec="#3a3f49", alpha=0.85))
ax0.text(0.022, 0.845, "dots = back-projected depth (turbo_r)", transform=ax0.transAxes,
         color=STYLE["fg"], fontsize=10.5, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.3", fc="#0c0e12", ec="#3a3f49", alpha=0.85))

# ---- Panel 2: raw 8x8 depth heatmap with per-cell text ----------------------
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_facecolor(STYLE["panel"])
disp = np.where((d8 >= NEAR) & (d8 <= FAR), d8, np.nan)
im = ax1.imshow(disp, cmap=STYLE["cmap"], vmin=vmin, vmax=vmax,
                interpolation="nearest", origin="upper")
for i in range(8):
    for j in range(8):
        val = d8[i, j]
        if NEAR <= val <= FAR:
            # text color contrast vs cell
            rgba = cmap(norm(val))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = "#0a0c10" if lum > 0.55 else "#f2f2f4"
            ax1.text(j, i, f"{val*100:.1f}", ha="center", va="center",
                     fontsize=8.0, color=tc)
        else:
            ax1.text(j, i, "-", ha="center", va="center", fontsize=9, color="#555")
ax1.set_xticks(range(8)); ax1.set_yticks(range(8))
ax1.set_xticklabels(range(8), fontsize=8); ax1.set_yticklabels(range(8), fontsize=8)
ax1.set_xlabel("column (u)", fontsize=10.5); ax1.set_ylabel("row (v)", fontsize=10.5)
ax1.set_xticks(np.arange(-0.5, 8, 1), minor=True)
ax1.set_yticks(np.arange(-0.5, 8, 1), minor=True)
ax1.grid(which="minor", color=STYLE["bg"], linewidth=1.4)
ax1.tick_params(which="minor", length=0)
for sp in ax1.spines.values():
    sp.set_edgecolor(STYLE["grid"])
ax1.set_title("2 - raw 8x8 depth frame, per-cell distance (cm)\n"
              "flat plane -> all 64 cells read 14.6 cm (ground truth)",
              color=STYLE["fg"], fontsize=12, pad=8, loc="left")

# ---- Panel 3: back-projected 64-point patch in 3D ---------------------------
ax2 = fig.add_subplot(gs[0, 2], projection="3d")
ax2.set_facecolor(STYLE["panel"])
ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=depths, cmap=STYLE["cmap"],
            vmin=vmin, vmax=vmax, s=46, depthshade=False, edgecolors="#0c0e12",
            linewidths=0.35)
# sensor origin + central ray to the patch centroid
ax2.scatter([pos[0]], [pos[1]], [pos[2]], color="#ffe54d", s=70, marker="^",
            depthshade=False, edgecolors="#0a0c10", linewidths=0.5, label="SPAD origin")
ax2.plot([pos[0], c[0]], [pos[1], c[1]], [pos[2], c[2]],
         color="#ffffff", lw=1.0, alpha=0.5, ls="--")
# draw fitted plane as a faint quad through the patch
e1 = np.cross(n, [0, 0, 1.0]); e1 /= np.linalg.norm(e1) + 1e-9
e2 = np.cross(n, e1); e2 /= np.linalg.norm(e2) + 1e-9
span = 0.055
quad = np.array([c + e1*span + e2*span, c + e1*span - e2*span,
                 c - e1*span - e2*span, c - e1*span + e2*span])
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ax2.add_collection3d(Poly3DCollection([quad], facecolor="#30c98c", alpha=0.10,
                                      edgecolor="#30c98c", linewidths=0.8))
ax2.set_xlabel("x (m)", fontsize=10, labelpad=2)
ax2.set_ylabel("y (m)", fontsize=10, labelpad=2)
ax2.set_zlabel("z (m)", fontsize=10, labelpad=2)
ax2.tick_params(labelsize=7.5, colors=STYLE["fg"])
ax2.xaxis.pane.set_facecolor(STYLE["panel"]); ax2.xaxis.pane.set_alpha(0.6)
ax2.yaxis.pane.set_facecolor(STYLE["panel"]); ax2.yaxis.pane.set_alpha(0.6)
ax2.zaxis.pane.set_facecolor(STYLE["panel"]); ax2.zaxis.pane.set_alpha(0.6)
ax2.xaxis.pane.set_edgecolor(STYLE["grid"])
ax2.yaxis.pane.set_edgecolor(STYLE["grid"])
ax2.zaxis.pane.set_edgecolor(STYLE["grid"])
for axisline in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
    axisline._axinfo["grid"]["color"] = STYLE["grid"]
    axisline._axinfo["grid"]["linewidth"] = 0.5
ax2.view_init(elev=20, azim=-58)
ax2.set_title(f"3 - back-projected point cloud\n64 pts on plane, fit RMS = {rms*1000:.2f} mm",
              color=STYLE["fg"], fontsize=13, pad=2, loc="left")
ax2.legend(loc="upper left", fontsize=8.5, facecolor=STYLE["panel"],
           edgecolor=STYLE["grid"], labelcolor=STYLE["fg"], framealpha=0.85)
try:
    ax2.set_box_aspect((1, 1, 1))
except Exception:
    pass

# ---- shared colorbar --------------------------------------------------------
sm = cm.ScalarMappable(cmap=STYLE["cmap"], norm=plt.Normalize(vmin, vmax))
cax = fig.add_axes([0.39, 0.045, 0.012, 0.74])
cb = fig.colorbar(sm, cax=cax)
cb.set_label("distance (m)", color=STYLE["fg"], fontsize=10.5)
cb.ax.yaxis.set_tick_params(color=STYLE["fg"], labelsize=8.5)
cb.outline.set_edgecolor(STYLE["grid"])
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=STYLE["fg"])

# ---- suptitle / global annotation ------------------------------------------
fig.text(0.012, 0.955,
         "Anatomy of ONE SPAD proximity sensor  -  franka_skin HYBRID skin",
         color=STYLE["fg"], fontsize=18, fontweight="bold", ha="left", va="top")
fig.text(0.012, 0.885,
         f"sensor '{SENSOR}'  |  8x8 SPAD depth camera, fovy = {FOVY:.0f} deg  |  "
         f"range {NEAR*100:.1f}-{FAR*100:.0f} cm  |  probe plane at {PLANE_D*100:.0f} cm  |  "
         f"{n_ret}/64 returns, mean depth {dmean*100:.1f} cm",
         color="#b9bcc4", fontsize=12.0, ha="left", va="top")
fig.text(0.985, 0.955,
         "1 of 40 sensors\nRGB blurred at train time -> these ARE perception",
         color=STYLE["accent"], fontsize=10.5, ha="right", va="top", style="italic")

out = os.path.join(OUT, f"{KEY}.png")
fig.savefig(out, dpi=170, facecolor=STYLE["bg"], bbox_inches="tight")
plt.close(fig)
sz = os.path.getsize(out)
print(f"SAVED {out}  {sz/1024:.1f} KB  exists={os.path.exists(out)}")
print(f"RESULT n_ret={n_ret} mean={dmean:.4f} rms_mm={rms*1000:.4f}")
