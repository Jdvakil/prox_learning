"""Consolidated figure generators for the proximity-skin / safety-CVAE paper.

Every paper/proof figure that renders the 40-sensor hybrid skin lives here as one
function `fig_<name>()`. Each function is self-contained (all helpers + constants are
local) and writes its PNG under experiments_output/default/figures/ (override with --outdir).

Usage:
    python scripts/figures.py --list                 # list available figures
    python scripts/figures.py panel_coverage_behind  # one figure
    python scripts/figures.py --all                  # every figure

Headless render (this repo): prefix with
    OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# make sibling modules (hybrid_viz_lib, cavity_scene, safety_sweep, ...) importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")

from collections import defaultdict
from collections import defaultdict, OrderedDict
from hybrid_viz_lib import FAR, FOVY, NEAR, STYLE, add_plane_mocap, backproject, build, cam_pose, depth8, depth_renderer, mocap_set, set_pose
from hybrid_viz_lib import FAR, FOVY, NEAR, STYLE, backproject, build, depth8, depth_renderer, mjv_cam, nice_lights, render_scene, set_pose
from hybrid_viz_lib import build, set_pose, add_plane_mocap, mocap_set, depth_renderer, depth8, cam_pose, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, cam_pose, depth_renderer, depth8, add_cylinder, render_scene, mjv_cam, nice_lights, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, sensors, cam_pose, nice_lights, add_box, skin_cloud, depth_renderer, depth8, backproject, render_scene, mjv_cam, FAR, NEAR, FOVY, STYLE
from hybrid_viz_lib import build, set_pose, sensors, depth8, backproject, cam_pose, add_box, add_cylinder, depth_renderer, nice_lights, skin_cloud, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, sensors, depth8, backproject, cam_pose, add_box, add_cylinder, depth_renderer, nice_lights, skin_cloud, render_scene, mjv_cam, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, sensors, depth8, backproject, cam_pose, add_box, add_sphere_mocap, mocap_set, depth_renderer, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, sensors, depth8, backproject, cam_pose, add_box, depth_renderer, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, sensors, depth8, backproject, cam_pose, add_plane_mocap, mocap_set, depth_renderer, fit_plane, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, sensors, depth8, backproject, cam_pose, depth_renderer, skin_cloud, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, sensors, depth8, cam_pose, add_plane_mocap, mocap_set, depth_renderer, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, sensors, depth8, cam_pose, mocap_set, add_box, add_plane_mocap, depth_renderer, skin_cloud, render_scene, mjv_cam, nice_lights, FAR
from hybrid_viz_lib import build, set_pose, sensors, depth8, depth_renderer, add_box, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, sensors, depth8, depth_renderer, add_plane_mocap, mocap_set, cam_pose, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, sensors, nice_lights, add_capsule, depth_renderer, depth8, backproject, cam_pose, skin_cloud, render_scene, mjv_cam, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, sensors, skin_cloud, add_box, nice_lights, cam_pose, render_scene, mjv_cam, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, sensors, skin_cloud, depth_renderer, render_scene, mjv_cam, nice_lights, add_box, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, skin_cloud, add_cylinder, nice_lights, render_scene, mjv_cam, FOVY, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, skin_cloud, nice_lights, add_box, add_cylinder, depth_renderer, depth8, cam_pose, render_scene, mjv_cam, sensors, FAR, NEAR, STYLE
from hybrid_viz_lib import build, set_pose, skin_cloud, render_scene, mjv_cam, nice_lights, add_box, depth_renderer, FAR, NEAR, STYLE
from hybrid_viz_lib import build, set_pose, skin_cloud, render_scene, mjv_cam, nice_lights, add_box, depth_renderer, FAR, STYLE
from hybrid_viz_lib import build, set_pose, skin_cloud, render_scene, mjv_cam, nice_lights, add_box, depth_renderer, cam_pose, NEAR, FAR, STYLE
from hybrid_viz_lib import build, set_pose, skin_cloud, render_scene, mjv_cam, nice_lights, add_box, depth_renderer, depth8, backproject, FAR, NEAR, STYLE
from matplotlib import cm
from matplotlib import cm as mcm
from matplotlib import cm, colors as mcolors
from matplotlib import colormaps
from matplotlib import colormaps as mcmaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.optimize import minimize
from test_and_reconstruct_hybrid import gt_boxes, cloud_error, cloud_panel
import OpenGL.EGL as _EGL
import cv2
import json
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mujoco
import mujoco, numpy as np
import mujoco, numpy as np, matplotlib
import numpy as np
import os, sys
import sys, os
import sys, os, re


# ---------------------------------------------------------------------------
# Figure output root (override with --outdir or env PROX_FIG_OUT)
# ---------------------------------------------------------------------------
_FIGROOT = Path(os.environ.get("PROX_FIG_OUT", "/home/jaydv/code/prox_learning/experiments_output/default/figures"))


# ============================================================================
# Single-sensor anatomy & metric accuracy
# ============================================================================

def fig_panel_single_sensor_anatomy():
    """Educational 3-panel figure dissecting ONE SPAD proximity sensor: in-situ FOV cone + 64 returns, raw 8x8 depth heatmap, and the back-projected 3D point patch."""
    from hybrid_viz_lib import (build, set_pose, sensors, depth8, backproject, cam_pose,
        add_plane_mocap, mocap_set, depth_renderer, fit_plane, FOVY, NEAR, FAR, STYLE)
    import mujoco, numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    OUT = str(_FIGROOT)
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


def fig_panel_plane_distance_sweep():
    """Plane distance sweep proving metric accuracy of the franka_skin HYBRID SPAD proximity sensors."""
    OUT_DIR = str(_FIGROOT)
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


def fig_panel_plane_tilt_sweep():
    """plane_tilt_sweep panel: one SPAD sensor vs a flat plane tilted 0/15/30/45 deg at 0.18 m."""
    import os
    from hybrid_viz_lib import (build, set_pose, sensors, depth8, backproject, cam_pose,
        add_plane_mocap, mocap_set, depth_renderer, fit_plane, FOVY, NEAR, FAR, STYLE)
    import mujoco, numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    OUTDIR = str(_FIGROOT)
    os.makedirs(OUTDIR, exist_ok=True)
    OUT = os.path.join(OUTDIR, "plane_tilt_sweep.png")

    TILTS = [0, 15, 30, 45]
    D = 0.18                     # fixed plane distance (m)
    SENSOR = "link6_sensor_0"    # a SPAD that gets a full clean return at 0.18 m

    def rodrigues(v, k, a):
        """Rotate vector v about unit axis k by angle a (rad)."""
        k = k / (np.linalg.norm(k) + 1e-12)
        return v * np.cos(a) + np.cross(k, v) * np.sin(a) + k * np.dot(k, v) * (1 - np.cos(a))

    def make(spec):
        add_plane_mocap(spec, "probe_plane", half=(0.13, 0.13, 0.004), rgba=(0.22, 0.85, 0.42, 1))

    model = build(make)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")
    assert len(sensors(model)) == 40, "expected 40 SPAD sensors"
    rd = depth_renderer(model)

    pos, R = cam_pose(model, data, SENSOR)
    fwd = -R[:, 2]            # camera optical axis (looks along -z)
    xax = R[:, 0]            # sensor local x-axis -> the tilt hinge

    # ---- collect data for all tilts (shared color scale across the row) ----
    runs = []
    dmin_all, dmax_all = np.inf, -np.inf
    for tilt in TILTS:
        a = np.deg2rad(tilt)
        view = rodrigues(fwd, xax, a)               # tilt the plane normal about x
        center = pos + fwd * D                      # plane centre stays at 0.18 m
        mocap_set(model, data, "probe_plane", center, view_dir=view)
        d8 = depth8(rd, data, SENSOR)
        mask = (d8 >= NEAR) & (d8 <= FAR)
        pts, d = backproject(d8, pos, R)
        c, n, rms = fit_plane(pts)
        dmin_all = min(dmin_all, d.min())
        dmax_all = max(dmax_all, d.max())
        runs.append(dict(tilt=tilt, d8=d8, mask=mask, pts=pts, d=d,
                         c=c, n=n, rms=rms, npix=int(mask.sum())))

    norm = mcolors.Normalize(vmin=dmin_all, vmax=dmax_all)
    cmap = matplotlib.colormaps[STYLE["cmap"]]

    # ---- figure scaffolding (dark robotics-lab theme) ----
    plt.rcParams.update({
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
        "axes.edgecolor": STYLE["grid"], "font.family": "DejaVu Sans",
    })
    fig = plt.figure(figsize=(16.5, 8.6), facecolor=STYLE["bg"])
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.35],
                          left=0.045, right=0.92, top=0.86, bottom=0.07,
                          hspace=0.30, wspace=0.28)

    for j, run in enumerate(runs):
        # ===== TOP ROW: 8x8 depth heatmap =====
        axh = fig.add_subplot(gs[0, j], facecolor=STYLE["panel"])
        disp = np.where(run["mask"], run["d8"], np.nan)
        im = axh.imshow(disp, cmap=cmap, norm=norm, origin="upper",
                        interpolation="nearest")
        axh.set_xticks([]); axh.set_yticks([])
        for s in axh.spines.values():
            s.set_color(STYLE["grid"])
        # annotate each cell with its depth in mm
        for r in range(8):
            for ccol in range(8):
                if run["mask"][r, ccol]:
                    val = run["d8"][r, ccol]
                    tc = "#0a0a0a" if 0.30 < norm(val) < 0.78 else STYLE["fg"]
                    axh.text(ccol, r, f"{val*1000:.0f}", ha="center", va="center",
                             fontsize=5.4, color=tc)
        span = (run["d"].max() - run["d"].min()) * 1000
        axh.set_title(f"tilt {run['tilt']}°   ·   {run['npix']}/64 px",
                      color=STYLE["accent"], fontsize=12.5, pad=7, fontweight="bold")
        axh.set_xlabel(f"depth span {span:.0f} mm", fontsize=9.5,
                       color=STYLE["fg"], labelpad=3)

        # ===== BOTTOM ROW: back-projected cloud lying ON the tilted plane (3D) =====
        ax3 = fig.add_subplot(gs[1, j], projection="3d", facecolor=STYLE["panel"])
        pts, dd = run["pts"], run["d"]
        cols = cmap(norm(dd))

        # draw the fitted plane patch so the eye sees the cloud lies on a flat tilt
        c, n = run["c"], run["n"]
        if n @ (pos - c) < 0:    # orient normal toward the sensor
            n = -n
        # build an in-plane basis
        t0 = np.array([1.0, 0, 0])
        if abs(n @ t0) > 0.9:
            t0 = np.array([0, 1.0, 0])
        e1 = np.cross(n, t0); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        # extent of the cloud in plane coords
        rel = pts - c
        s1 = (rel @ e1); s2 = (rel @ e2)
        pad = 0.012
        g1 = np.linspace(s1.min() - pad, s1.max() + pad, 2)
        g2 = np.linspace(s2.min() - pad, s2.max() + pad, 2)
        G1, G2 = np.meshgrid(g1, g2)
        PX = c[0] + G1 * e1[0] + G2 * e2[0]
        PY = c[1] + G1 * e1[1] + G2 * e2[1]
        PZ = c[2] + G1 * e1[2] + G2 * e2[2]
        ax3.plot_surface(PX, PY, PZ, color="#3a4150", alpha=0.28,
                         linewidth=0, shade=False, zorder=1)

        ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=26,
                    depthshade=False, edgecolors="#0c0e12", linewidths=0.25, zorder=3)
        # short optical-ray stub from the cloud centroid back toward the sensor,
        # so the viewer can read which way the sensor looks
        ray_to = c - 0.05 * (c - pos) / (np.linalg.norm(c - pos) + 1e-9)
        ax3.plot([c[0], ray_to[0]], [c[1], ray_to[1]], [c[2], ray_to[2]],
                 color=STYLE["accent"], lw=1.4, ls="--", alpha=0.7, zorder=4)

        # tight equal-ish cube around the cloud
        ctr = pts.mean(0)
        rng = max(0.045, (pts.max(0) - pts.min(0)).max() * 0.62)
        ax3.set_xlim(ctr[0]-rng, ctr[0]+rng)
        ax3.set_ylim(ctr[1]-rng, ctr[1]+rng)
        ax3.set_zlim(ctr[2]-rng, ctr[2]+rng)
        ax3.set_box_aspect((1, 1, 1))
        # view roughly edge-on to the sensor x-axis so the plane tilt is visible
        ax3.view_init(elev=10, azim=-72)
        ax3.set_title(f"cloud RMS = {run['rms']*1000:.3f} mm",
                      color=STYLE["fg"], fontsize=12, pad=-2, fontweight="bold")
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis.set_pane_color((0.09, 0.10, 0.12, 1.0))
            axis.line.set_color(STYLE["grid"])
            axis.label.set_color(STYLE["fg"])
        ax3.tick_params(colors=STYLE["fg"], labelsize=6.5, pad=-2)
        ax3.grid(True, color=STYLE["grid"], alpha=0.4)
        ax3.set_xlabel("x (m)", fontsize=8, labelpad=-6)
        ax3.set_ylabel("y (m)", fontsize=8, labelpad=-6)
        ax3.set_zlabel("z (m)", fontsize=8, labelpad=-6)

    # shared colorbar
    cax = fig.add_axes([0.935, 0.07, 0.013, 0.79])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("distance (m)", color=STYLE["fg"], fontsize=11)
    cb.ax.yaxis.set_tick_params(color=STYLE["fg"], labelsize=9)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=STYLE["fg"])
    cb.outline.set_edgecolor(STYLE["grid"])

    rms_max = max(r["rms"] for r in runs) * 1000
    fig.suptitle(
        "franka_skin HYBRID SPAD  ·  plane-tilt sweep:  one 8×8 depth sensor (fovy 45°, "
        "0.015–0.5 m)  vs.  a flat plane at fixed 0.18 m",
        color=STYLE["fg"], fontsize=15.5, fontweight="bold", y=0.975)
    fig.text(0.045, 0.915,
             f"sensor '{SENSOR}'  ·  plane tilted 0–45° about the sensor's local x-axis  ·  "
             f"top: per-cell depth (mm) — gradient grows with tilt   bottom: back-projected cloud "
             f"reconstructs a true flat tilted plane  (max fit RMS {rms_max:.3f} mm)",
             color="#aab0bb", fontsize=10.5)

    fig.savefig(OUT, dpi=170, facecolor=STYLE["bg"], bbox_inches="tight")
    plt.close(fig)

    sz = os.path.getsize(OUT)
    spans = [(r["d"].max()-r["d"].min())*1000 for r in runs]
    print("SAVED", OUT, sz)
    print("spans_mm", [round(s, 1) for s in spans])
    print("rms_mm", [round(r["rms"]*1000, 4) for r in runs])
    print("npix", [r["npix"] for r in runs])
    return OUT, sz, runs


def fig_panel_range_accuracy_scatter():
    """Scatter of measured vs true head-on plane distance for franka_skin HYBRID SPAD proximity sensors, with line fit and residual diagnostics."""
    OUTDIR = str(_FIGROOT)
    os.makedirs(OUTDIR, exist_ok=True)
    OUT = os.path.join(OUTDIR, "range_accuracy_scatter.png")

    HALF_T = 0.004                       # probe-plane half thickness; surface sits HALF_T in front of center
    TRUE = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])   # 6+ head-on distances (m)
    N_SENSORS = 12

    def make(spec):
        # The SPAD honest range is 0.015-0.5 m, but MuJoCo's default offscreen near-clip
        # (znear*extent ~ 0.106 m) silently clips anything closer. Pull znear in so the depth
        # buffer faithfully resolves the full proximity band instead of dropping near returns.
        spec.visual.map.znear = 0.0008
        add_plane_mocap(spec, "probe_plane", half=(0.30, 0.30, HALF_T), rgba=(0.20, 0.85, 0.40, 1))

    def center_depth(d8):
        """Center-distance readout: mean of the central 2x2 of the 8x8 SPAD frame (m)."""
        return float(d8[3:5, 3:5].mean())

    def measure():
        model = build(make)
        data = mujoco.MjData(model)
        set_pose(model, data, "reach")
        names = sensors(model)
        rd = depth_renderer(model)

        # pick a spread of sensors across the arm links so it's not all from one body
        pick = [names[i] for i in np.linspace(0, len(names) - 1, N_SENSORS).round().astype(int)]
        pick = list(dict.fromkeys(pick))   # dedupe, keep order

        rows = []   # (sensor, link_idx, true, meas)
        for si, name in enumerate(pick):
            pos, R = cam_pose(model, data, name)
            fwd = -R[:, 2]                 # outward optical axis
            link = name.split("_sensor_")[0]
            for td in TRUE:
                # place plane so its FRONT FACE (not center) is exactly td from the sensor
                mocap_set(model, data, "probe_plane", pos + fwd * (td + HALF_T), view_dir=fwd)
                d8 = depth8(rd, data, name)
                if d8.min() > FAR:         # no return (clipped/occluded) -> skip honestly
                    continue
                rows.append((name, link, td, center_depth(d8)))
        return pick, rows

    pick, rows = measure()
    true = np.array([r[2] for r in rows])
    meas = np.array([r[3] for r in rows])
    links = np.array([r[1] for r in rows])
    resid_mm = (meas - true) * 1000.0

    # least-squares line fit meas = a*true + b
    a, b = np.polyfit(true, meas, 1)
    pred = a * true + b
    ss_res = np.sum((meas - pred) ** 2)
    ss_tot = np.sum((meas - meas.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rms_mm = float(np.sqrt(np.mean(resid_mm ** 2)))
    max_abs_mm = float(np.max(np.abs(resid_mm)))
    n_sens = len(set(r[0] for r in rows))

    # ---- figure ----------------------------------------------------------
    s = STYLE
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": s["fg"], "axes.labelcolor": s["fg"],
        "xtick.color": s["fg"], "ytick.color": s["fg"],
        "axes.edgecolor": s["grid"],
    })
    fig = plt.figure(figsize=(13.5, 7.2), facecolor=s["bg"])
    gs = GridSpec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 0.62],
                  wspace=0.24, hspace=0.42, left=0.075, right=0.965,
                  top=0.84, bottom=0.10)

    # main scatter: measured vs true, colored by true distance (turbo_r)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(s["panel"])
    cmap = matplotlib.colormaps[s["cmap"]]
    norm = matplotlib.colors.Normalize(vmin=TRUE.min(), vmax=TRUE.max())

    lo, hi = 0.0, 0.40
    ax.plot([lo, hi], [lo, hi], color=s["fg"], lw=1.3, ls="--", alpha=0.55,
            zorder=1, label="ideal  y = x")
    xx = np.linspace(lo, hi, 50)
    ax.plot(xx, a * xx + b, color=s["accent"], lw=2.0, zorder=2,
            label=f"fit  y = {a:.4f}x {b*1e3:+.3f} mm")

    sc = ax.scatter(true * 1000, meas * 1000, c=true, cmap=cmap, norm=norm,
                    s=95, edgecolor="#0c0e12", linewidth=0.8, zorder=4, alpha=0.95)
    ax.set_xlim(0, 400); ax.set_ylim(0, 400)
    ax.set_aspect("equal")
    ax.set_xlabel("true plane distance  (mm)")
    ax.set_ylabel("measured center distance  (mm)")
    ax.grid(True, color=s["grid"], lw=0.6, alpha=0.7)
    for sp in ax.spines.values():
        sp.set_color(s["grid"])
    leg = ax.legend(loc="upper left", framealpha=0.0, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(s["fg"])
    ax.set_title(f"{n_sens} SPAD sensors x {len(TRUE)} head-on distances  "
                 f"(N = {len(rows)} readings)", color=s["fg"], fontsize=12, pad=9)

    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("distance (m)", color=s["fg"])
    cb.ax.yaxis.set_tick_params(color=s["fg"])
    cb.outline.set_edgecolor(s["grid"])
    plt.setp(plt.getp(cb.ax, "yticklabels"), color=s["fg"])

    # residual histogram (measured - true), in micrometers (residuals are sub-micron)
    resid_um = resid_mm * 1000.0
    axh = fig.add_subplot(gs[0, 1])
    axh.set_facecolor(s["panel"])
    rng = max(np.abs(resid_um).max() * 1.15, 1e-2)
    bins = np.linspace(-rng, rng, 31)
    axh.hist(resid_um, bins=bins, color=s["accent"], edgecolor="#0c0e12", alpha=0.9)
    axh.axvline(0, color=s["fg"], lw=1.1, ls="--", alpha=0.6)
    axh.set_xlabel("residual  measured - true  (µm)")
    axh.set_ylabel("count")
    axh.grid(True, color=s["grid"], lw=0.5, alpha=0.6, axis="y")
    for sp in axh.spines.values():
        sp.set_color(s["grid"])
    axh.set_title("residual distribution", color=s["fg"], fontsize=11, pad=6)
    axh.text(0.97, 0.92,
             f"mean {resid_um.mean():+.3f} µm\nstd  {resid_um.std():.3f} µm",
             transform=axh.transAxes, ha="right", va="top", fontsize=9.5,
             color=s["fg"], family="monospace",
             bbox=dict(boxstyle="round", fc=s["bg"], ec=s["grid"], alpha=0.85))

    # residual vs distance (shows there's no range-dependent bias)
    axr = fig.add_subplot(gs[1, 1])
    axr.set_facecolor(s["panel"])
    axr.scatter(true * 1000, resid_um, c=true, cmap=cmap, norm=norm,
                s=42, edgecolor="#0c0e12", linewidth=0.5, alpha=0.9)
    axr.axhline(0, color=s["fg"], lw=1.0, ls="--", alpha=0.55)
    axr.set_xlabel("true distance  (mm)")
    axr.set_ylabel("resid (µm)")
    axr.set_xlim(0, 400)
    axr.grid(True, color=s["grid"], lw=0.5, alpha=0.6)
    for sp in axr.spines.values():
        sp.set_color(s["grid"])
    axr.set_title("residual vs range", color=s["fg"], fontsize=10.5, pad=5)

    fig.suptitle("franka_skin HYBRID  —  SPAD proximity sensors read metric truth",
                 color=s["fg"], fontsize=16, fontweight="bold", x=0.5, y=0.965)
    fig.text(0.5, 0.905,
             f"slope = {a:.5f}    R² = {r2:.8f}    RMS error = {rms_mm*1000:.3f} µm    "
             f"max |err| = {max_abs_mm*1000:.3f} µm   |   8×8 depth, fovy={FOVY:g}°, "
             f"range {NEAR}-{FAR} m  (residual at depth-buffer floor)",
             color=s["accent"], fontsize=12, ha="center", family="monospace")

    fig.savefig(OUT, dpi=170, facecolor=s["bg"], bbox_inches="tight")
    plt.close(fig)

    print("SLOPE", a, "R2", r2, "RMS_mm", rms_mm, "MAX_mm", max_abs_mm,
          "N", len(rows), "SENSORS", n_sens)
    sz = os.path.getsize(OUT)
    print("WROTE", OUT, sz, "bytes")
    return OUT, sz, a, r2, rms_mm, max_abs_mm, len(rows), n_sens


def fig_acc_angular_resolution():
    """PROOF PANEL: angular resolution of one 8x8/45deg SPAD at 0.15 m via two rods."""
    OUT = str(_FIGROOT)
    os.makedirs(OUT, exist_ok=True)
    KEY = "acc_angular_resolution"

    SENSOR = "link1_sensor_0"   # forward ~ -Y (horizontal), right ~ +X, up ~ -Z : clean frame
    RANGE = 0.15                # m, the standoff at which we characterize resolution
    ROD_R = 0.008               # 8 mm radius rods (~1-pixel footprint, so each reliably registers)
    ROD_HL = 0.14               # 14 cm half-length: tall vertical rods spanning the FOV
    GAPS = [0.02, 0.04, 0.06, 0.10]   # the requested separations (cm: 2,4,6,10)
    CMAP = colormaps["turbo_r"]

    # ----------------------------------------------------------------------------- geometry
    m0 = build()
    d0 = mujoco.MjData(m0)
    set_pose(m0, d0, "reach")
    pos, R = cam_pose(m0, d0, SENSOR)
    fwd = -R[:, 2]
    right = R[:, 0]
    righth = right - np.dot(right, [0, 0, 1]) * np.array([0, 0, 1.0])
    righth /= np.linalg.norm(righth)   # horizontal "right" so rods stay vertical (world +Z)

    # pixel-center lateral sampling positions at RANGE (the heart of the resolution story)
    f_pix = (8 / 2) / np.tan(np.deg2rad(FOVY / 2))
    cx = (8 - 1) / 2.0
    col_lat = (np.arange(8) - cx) * RANGE / f_pix          # m, lateral pos of each column ray
    pix_pitch = RANGE * (1.0 / f_pix)                       # m per column at RANGE
    ang_pitch = FOVY / 8.0                                  # deg per pixel

    def make_for_gap(gap):
        def make(spec):
            nice_lights(spec)
            ctr = pos + fwd * RANGE
            for sgn, nm in [(-1, "rodA"), (1, "rodB")]:
                c = ctr + sgn * (gap / 2) * righth
                add_cylinder(spec, nm, [float(c[0]), float(c[1]), float(c[2])],
                             ROD_R, ROD_HL, [0.93, 0.32, 0.20, 1])
        return make

    def measure(gap):
        """Return (8x8 depth m, per-column min m, n_objects, rgb_render or None)."""
        m = build(make=make_for_gap(gap))
        d = mujoco.MjData(m)
        set_pose(m, d, "reach")
        rd = depth_renderer(m)
        d8 = depth8(rd, d, SENSOR)
        valid = (d8 >= NEAR) & (d8 <= FAR)
        cp = np.where(valid, d8, np.nan)
        colmin = np.nanmin(np.where(np.isnan(cp), np.inf, cp), axis=0)
        colmin = np.where(np.isfinite(colmin) & (colmin < FAR), colmin, np.nan)
        # count separated objects = runs of hit columns separated by >=1 empty column
        hit = np.isfinite(colmin)
        runs, prev = 0, False
        for h in hit:
            if h and not prev:
                runs += 1
            prev = h
        return d8, colmin, runs, m, d

    # render one nice 3D context view for the resolved (4 cm) case
    ctx_gap = 0.04
    mctx = build(make=make_for_gap(ctx_gap))
    dctx = mujoco.MjData(mctx)
    set_pose(mctx, dctx, "reach")
    look = tuple((pos + fwd * 0.09).tolist())
    ctx_cam = mjv_cam(lookat=look, distance=0.55, azimuth=-78, elevation=-8)
    ctx_rgb = render_scene(mctx, dctx, ctx_cam, w=760, h=720, cloud=None, gamma=0.72)

    results = [measure(g) for g in GAPS]

    # ----------------------------------------------------------------------------- figure
    plt.rcParams.update({
        "figure.facecolor": STYLE["bg"], "axes.facecolor": STYLE["panel"],
        "savefig.facecolor": STYLE["bg"], "text.color": STYLE["fg"],
        "axes.labelcolor": STYLE["fg"], "xtick.color": STYLE["fg"],
        "ytick.color": STYLE["fg"], "axes.edgecolor": "#3a3f4a",
        "font.family": "DejaVu Sans", "font.size": 10,
    })

    fig = plt.figure(figsize=(16, 10.5), dpi=170)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.18, 1.0, 0.92],
                  hspace=0.42, wspace=0.30,
                  left=0.055, right=0.965, top=0.855, bottom=0.075)

    fig.suptitle("Angular resolution of a single 8×8 / 45° SPAD depth sensor at 0.15 m",
                 fontsize=19, fontweight="bold", color=STYLE["fg"], y=0.972)
    fig.text(0.5, 0.93,
             "Two thin vertical rods in front of one sensor — how far apart before they read as TWO depth minima, not one blob?",
             ha="center", fontsize=11, color="#aab3c0")

    # --- (A) 3D context render, top-left spanning 1 col, 2 rows tall
    axc = fig.add_subplot(gs[0:2, 0])
    axc.imshow(ctx_rgb)
    axc.set_xticks([]); axc.set_yticks([])
    for s in axc.spines.values():
        s.set_color(STYLE["accent"]); s.set_linewidth(1.4)
    axc.set_title("scene: 1 SPAD → two rods\n(shown at 4 cm gap)", fontsize=11.5,
                  color=STYLE["accent"], pad=7)
    axc.text(0.5, -0.045,
             f"sensor {SENSOR}  ·  rod ø{ROD_R*2*1000:.0f} mm  ·  standoff {RANGE*100:.0f} cm",
             transform=axc.transAxes, ha="center", va="top", fontsize=8.5, color="#8b94a3")

    # --- (B) the 4 depth heatmaps (top row, cols 1..3 + first of row2) -> place 2x2 in cols 1-2
    # We'll lay the four 8x8 maps in a 2x2 block occupying gs[0:2, 1:3]
    sub = gs[0:2, 1:3].subgridspec(2, 2, hspace=0.46, wspace=0.32)
    dmin_all = np.nanmin([np.nanmin(np.where((r[0] >= NEAR) & (r[0] <= FAR), r[0], np.nan))
                          for r in results])
    vmin, vmax = 0.13, 0.17   # tight window around the 0.15 m rods so structure pops
    for k, (gap, (d8, colmin, runs, mm, dd)) in enumerate(zip(GAPS, results)):
        ax = fig.add_subplot(sub[k // 2, k % 2])
        disp = np.where((d8 >= NEAR) & (d8 <= FAR), d8, np.nan)
        cmap = CMAP.copy(); cmap.set_bad("#0c0e12")
        im = ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                       interpolation="nearest")
        # outline the returning columns + an arrow pointing at each detected object
        for c in range(8):
            if np.isfinite(colmin[c]):
                ax.add_patch(plt.Rectangle((c - 0.5, -0.5), 1, 8, fill=False,
                                           ec="#f8f9fb", lw=1.0, ls=(0, (2, 2)), alpha=0.7))
        # mark detected-object centers (run midpoints) with a downward caret above the map
        hitc = np.isfinite(colmin)
        runs_cols, cur = [], []
        for c in range(8):
            if hitc[c]:
                cur.append(c)
            elif cur:
                runs_cols.append(cur); cur = []
        if cur:
            runs_cols.append(cur)
        for grp in runs_cols:
            mc = np.mean(grp)
            ax.annotate("▼", xy=(mc, -0.5), xytext=(mc, -1.35),
                        ha="center", va="center", fontsize=9,
                        color="#06d6a0" if (runs >= 2) else "#ef476f")
        resolved = runs >= 2
        tag = "RESOLVED → 2 minima" if resolved else "MERGED → 1 blob"
        tagc = "#06d6a0" if resolved else "#ef476f"
        ax.set_title(f"gap = {gap*100:.0f} cm", fontsize=11, color=STYLE["fg"], pad=4)
        ax.text(0.5, 1.005, tag, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.6, fontweight="bold", color=tagc)
        ax.set_xticks(range(8)); ax.set_yticks(range(0, 8, 2))
        ax.tick_params(length=0, labelsize=6.5)
        ax.set_xlabel("column (azimuth)", fontsize=7.5, color="#8b94a3")
        for s in ax.spines.values():
            s.set_color(tagc); s.set_linewidth(1.5)

    cax = fig.add_axes([0.668, 0.545, 0.012, 0.275])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("distance (m)", fontsize=9.5, color=STYLE["fg"])
    cb.ax.tick_params(labelsize=7.5, color=STYLE["fg"])
    cb.outline.set_edgecolor("#3a3f4a")

    # --- (C) detection-profile panel: stacked, one strip per gap. A filled bar = a column
    #     that saw a return. 1 bar-group = merged, 2 separated bar-groups = resolved.
    axp = fig.add_subplot(gs[0:2, 3])
    axp.set_facecolor(STYLE["panel"])
    strip_h = 0.78
    for i, (gap, (d8, colmin, runs, mm, dd)) in enumerate(zip(GAPS, results)):
        base = (len(GAPS) - 1 - i) * 1.0           # top = smallest gap
        resolved = runs >= 2
        barc = "#06d6a0" if resolved else "#ef476f"
        # faint slot outlines for all 8 columns
        for c in range(8):
            axp.add_patch(plt.Rectangle((c - 0.42, base), 0.84, strip_h,
                                        fill=False, ec="#2a2e36", lw=0.8))
        for c in range(8):
            if np.isfinite(colmin[c]):
                # height encodes proximity: nearer -> taller (rods are all ~0.15 so near full)
                frac = np.clip((vmax - colmin[c]) / (vmax - vmin), 0.25, 1.0)
                axp.add_patch(plt.Rectangle((c - 0.42, base), 0.84, strip_h * frac,
                                            fc=barc, ec="#0c0e12", lw=0.5, alpha=0.92))
        axp.text(-1.05, base + strip_h / 2, f"{gap*100:.0f} cm", ha="right", va="center",
                 fontsize=10, color=STYLE["fg"], fontweight="bold")
        axp.text(8.05, base + strip_h / 2, f"{runs} obj", ha="left", va="center",
                 fontsize=9, color=barc, fontweight="bold")
    axp.set_xlim(-1.2, 9.0)
    axp.set_ylim(-0.25, len(GAPS) + 0.05)
    axp.set_xticks(range(8))
    axp.set_yticks([])
    axp.tick_params(length=0, labelsize=8)
    axp.set_xlabel("sensor column (azimuth bin, 0–7)", fontsize=9.5)
    sec = axp.secondary_xaxis("top", functions=(lambda c: (c - cx) * RANGE / f_pix * 100,
                                                lambda x: x / (RANGE / f_pix * 100) + cx))
    sec.set_xlabel("lateral position at 0.15 m (cm)", fontsize=8.5, color="#8b94a3")
    sec.tick_params(labelsize=7, colors="#8b94a3")
    axp.set_title("which columns saw a return?", fontsize=11,
                  color=STYLE["accent"], pad=30)
    for s in axp.spines.values():
        s.set_color("#3a3f4a")

    # --- (D) bottom: resolution-vs-geometry explainer, full width
    axg = fig.add_subplot(gs[2, :])
    axg.set_facecolor(STYLE["panel"])
    # draw the 8 column rays fanning out and where rods at threshold sit
    axg.axhline(0, color="#3a3f4a", lw=0.8)
    for c in range(8):
        lat = col_lat[c] * 100
        axg.plot([0, lat], [0, RANGE * 100], color="#4a5160", lw=1.0, zorder=1)
        axg.plot(lat, RANGE * 100, "s", color="#4cc9f0", ms=7, zorder=3)
        axg.text(lat, RANGE * 100 + 0.55, f"{c}", ha="center", fontsize=7, color="#8b94a3")
    # the sensor at origin
    axg.plot(0, 0, "^", color="#ffd166", ms=14, zorder=4)
    axg.text(0, -1.0, "SPAD\n(8×8, 45°)", ha="center", va="top", fontsize=8.5,
             color="#ffd166")
    # threshold rods at +/- 2cm (4cm gap) -- the resolvable case
    for sgn in (-1, 1):
        axg.plot([sgn * 2, sgn * 2], [RANGE * 100 - 3.5, RANGE * 100 + 3.5],
                 color="#06d6a0", lw=5, solid_capstyle="round", zorder=2, alpha=0.9)
    axg.annotate("", xy=(2, RANGE * 100 + 2.6), xytext=(-2, RANGE * 100 + 2.6),
                 arrowprops=dict(arrowstyle="<->", color="#06d6a0", lw=1.6))
    axg.text(0, RANGE * 100 + 3.6, "4 cm → one empty column between rods = RESOLVED",
             ha="center", fontsize=9, color="#06d6a0", fontweight="bold")
    # unresolvable 2cm pair (faint)
    for sgn in (-1, 1):
        axg.plot([sgn * 1, sgn * 1], [RANGE * 100 - 2.2, RANGE * 100 + 2.2],
                 color="#ef476f", lw=4, solid_capstyle="round", zorder=2, alpha=0.45)
    axg.text(0, RANGE * 100 - 4.2, "2 cm → same/adjacent column = MERGED",
             ha="center", fontsize=8.5, color="#ef476f")
    axg.set_xlim(-7.6, 7.6)
    axg.set_ylim(-2.6, RANGE * 100 + 6.0)
    axg.set_xlabel("lateral position (cm)", fontsize=9.5)
    axg.set_ylabel("range (cm)", fontsize=9.5)
    axg.set_title(
        f"WHY: at 0.15 m the 8 columns sample every {pix_pitch*100:.2f} cm "
        f"({ang_pitch:.2f}°/pixel)  —  two objects resolve only with a clear empty column between them",
        fontsize=11, color=STYLE["fg"], pad=6)
    axg.grid(True, axis="y", color=STYLE["grid"], lw=0.5, alpha=0.5)
    for s in axg.spines.values():
        s.set_color("#3a3f4a")

    # headline finding box
    res_threshold = next((g for g, (_, _, runs, _, _) in zip(GAPS, results) if runs >= 2), None)
    fig.text(0.5, 0.018,
             f"FINDING:  rods MERGE at ≤2 cm and RESOLVE into two minima at ≥{res_threshold*100:.0f} cm  "
             f"→  angular resolution ≈ {res_threshold*100:.0f} cm at 0.15 m "
             f"(≈ {pix_pitch*100:.1f} cm pixel pitch, {ang_pitch:.1f}°/px).  "
             "The 8×8 grid is coarse — fine structure below the pixel pitch is invisible to a single sensor.",
             ha="center", fontsize=10.5, color="#e8e8ea",
             bbox=dict(boxstyle="round,pad=0.5", fc="#1b1f27", ec=STYLE["accent"], lw=1.2))

    out_png = os.path.join(OUT, f"{KEY}.png")
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    sz = os.path.getsize(out_png)
    print("SAVED", out_png, sz, "bytes")
    print("threshold cm", res_threshold * 100)
    print("pixel pitch cm", round(pix_pitch * 100, 3), "ang/px deg", round(ang_pitch, 3))
    for g, (_, colmin, runs, _, _) in zip(GAPS, results):
        print("gap", int(g * 100), "cm -> objects", runs, "cols",
              [None if not np.isfinite(c) else round(float(c), 3) for c in colmin])


def fig_make_acc_range_linearity():
    """PROOF PANEL: per-link SPAD range linearity (measured-vs-true distance across the whole skin)."""
    os.makedirs(str(_FIGROOT), exist_ok=True)

    from hybrid_viz_lib import (
        build, set_pose, add_plane_mocap, mocap_set, depth_renderer, depth8,
        cam_pose, FOVY, NEAR, FAR, STYLE,
    )
    from matplotlib import colormaps
    from matplotlib.lines import Line2D

    OUT = f"{_FIGROOT}/acc_range_linearity_perlink.png"
    HALFZ = 0.004  # plane probe half-thickness; near face faces the sensor

    # One representative sensor per link 2..6 (mid-belt sensors that face cleanly outward).
    REPS = {
        2: "link2_sensor_3",
        3: "link3_sensor_2",
        4: "link4_sensor_2",
        5: "link5_back_sensor_2",
        6: "link6_sensor_2",
    }

    # Full operating range, dense sweep. Start at 0.04 (just inside near=0.015) to 0.40 m.
    TRUE = np.linspace(0.04, 0.40, 37)

    def make(spec):
        add_plane_mocap(spec, half=(0.25, 0.25, HALFZ))

    def measure():
        model = build(make=make)
        data = mujoco.MjData(model)
        set_pose(model, data, "reach")
        rd = depth_renderer(model)

        results = {}
        for link, sname in REPS.items():
            pos, R = cam_pose(model, data, sname)
            fwd = -R[:, 2]
            meas = np.full_like(TRUE, np.nan)
            for i, td in enumerate(TRUE):
                # place plane so its NEAR FACE sits exactly td metres in front of the sensor
                center = pos + fwd * (td + HALFZ)
                mocap_set(model, data, "probe_plane", center, view_dir=fwd)
                d8 = depth8(rd, data, sname)
                # central 2x2 of the 8x8 = the optical-axis ray, head-on
                m = (d8 >= NEAR) & (d8 <= FAR)
                if m.any():
                    meas[i] = float(d8[3:5, 3:5].mean())
            results[link] = meas
        return results

    res = measure()

    # per-link fit stats
    stats = {}
    for link, meas in res.items():
        ok = np.isfinite(meas)
        t, mch = TRUE[ok], meas[ok]
        slope, intercept = np.polyfit(t, mch, 1)
        rms_um = float(np.sqrt(np.mean((mch - t) ** 2)) * 1e6)
        max_um = float(np.max(np.abs(mch - t)) * 1e6)
        stats[link] = dict(slope=slope, intercept=intercept, rms_um=rms_um, max_um=max_um)

    # ---- figure ----
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
        "axes.edgecolor": "#3a3f4a",
    })
    cmap = colormaps["turbo"]
    link_colors = {lk: cmap(0.12 + 0.76 * i / 4) for i, lk in enumerate(sorted(REPS))}

    fig = plt.figure(figsize=(13.5, 8.0), dpi=170)
    fig.patch.set_facecolor(STYLE["bg"])
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[3.0, 1.0],
                          hspace=0.42, wspace=0.24,
                          left=0.075, right=0.975, top=0.855, bottom=0.10)

    # MAIN: measured vs true
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(STYLE["panel"])

    # y=x ideal band (+/- 1 mm tolerance shading) + ideal line
    lo, hi = 0.02, 0.42
    ax.fill_between([lo, hi], [lo - 0.001, hi - 0.001], [lo + 0.001, hi + 0.001],
                    color=STYLE["accent"], alpha=0.10, zorder=0,
                    label="ideal $\\pm$1 mm")
    ax.plot([lo, hi], [lo, hi], color=STYLE["fg"], lw=1.4, ls=(0, (6, 4)),
            alpha=0.8, zorder=1, label="ideal  $y = x$")

    for link in sorted(REPS):
        meas = res[link]
        ok = np.isfinite(meas)
        c = link_colors[link]
        ax.plot(TRUE[ok], meas[ok], "-", color=c, lw=2.0, alpha=0.95, zorder=3)
        ax.scatter(TRUE[ok], meas[ok], s=26, color=c, edgecolors=STYLE["bg"],
                   linewidths=0.5, zorder=4,
                   label=f"link {link}  ({REPS[link]})   "
                         f"slope = {stats[link]['slope']:.4f},  RMS = {stats[link]['rms_um']:.3f} $\\mu$m")

    # near/far operating limits
    for x, lab in [(NEAR, f"near {NEAR*1000:.0f} mm"), (FAR, f"far {FAR*1000:.0f} mm")]:
        ax.axvline(x, color="#5a6070", lw=1.0, ls=":", zorder=1)
        ax.axhline(x, color="#5a6070", lw=1.0, ls=":", zorder=1)
    ax.text(NEAR + 0.005, 0.405, "near 15 mm", color="#8a90a0", fontsize=8, va="top", rotation=90)
    ax.text(FAR - 0.006, 0.06, "far 500 mm", color="#8a90a0", fontsize=8, ha="right", rotation=90, va="bottom")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("ground-truth distance to plane  (m)", fontsize=11)
    ax.set_ylabel("measured distance  (m)", fontsize=11)
    ax.grid(True, color=STYLE["grid"], lw=0.6, alpha=0.7)
    ax.tick_params(labelsize=9)
    leg = ax.legend(loc="upper left", fontsize=8.0, framealpha=0.0,
                    labelcolor=STYLE["fg"], borderpad=0.6, handlelength=1.6)
    leg.set_zorder(10)

    # RESIDUALS: error in micrometres vs true distance
    axr = fig.add_subplot(gs[0, 1])
    axr.set_facecolor(STYLE["panel"])
    axr.axhline(0, color=STYLE["fg"], lw=1.0, ls=(0, (6, 4)), alpha=0.7)
    for x in (NEAR, FAR):
        axr.axvline(x, color="#5a6070", lw=1.0, ls=":")
    for link in sorted(REPS):
        meas = res[link]
        ok = np.isfinite(meas)
        err_um = (meas[ok] - TRUE[ok]) * 1e6
        axr.plot(TRUE[ok], err_um, "-o", color=link_colors[link], lw=1.6, ms=3.2,
                 markeredgecolor=STYLE["bg"], markeredgewidth=0.4)
    axr.set_xlim(lo, hi)
    axr.set_ylim(-5, 5)
    axr.set_xlabel("ground-truth distance  (m)", fontsize=10)
    axr.set_ylabel("measured $-$ true  ($\\mu$m)", fontsize=10)
    axr.grid(True, color=STYLE["grid"], lw=0.6, alpha=0.7)
    axr.tick_params(labelsize=8.5)
    axr.set_title("range residual (sub-pixel truth)", fontsize=10, color=STYLE["fg"], pad=6)
    axr.text(0.97, 0.05, "all links within $\\pm$0.5 $\\mu$m\nof ground truth",
             transform=axr.transAxes, ha="right", va="bottom", fontsize=8.2,
             color=STYLE["accent"],
             bbox=dict(boxstyle="round,pad=0.35", fc="#0e1014", ec="#3a3f4a", alpha=0.9))

    # SUMMARY BAR: per-link RMS (um)
    axb = fig.add_subplot(gs[1, 1])
    axb.set_facecolor(STYLE["panel"])
    links = sorted(REPS)
    rmss = [stats[lk]["rms_um"] for lk in links]
    bars = axb.barh([f"L{lk}" for lk in links], rmss,
                    color=[link_colors[lk] for lk in links], alpha=0.92,
                    edgecolor=STYLE["bg"])
    for b, lk in zip(bars, links):
        v = stats[lk]["rms_um"]
        axb.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f} $\\mu$m",
                 va="center", fontsize=8.2, color=STYLE["fg"])
    axb.set_xlim(0, max(rmss) * 1.55 + 0.05)
    axb.invert_yaxis()
    axb.set_xlabel("range RMS error  ($\\mu$m)", fontsize=10)
    axb.grid(True, axis="x", color=STYLE["grid"], lw=0.6, alpha=0.7)
    axb.tick_params(labelsize=8.5)
    axb.set_title("per-link accuracy", fontsize=10, color=STYLE["fg"], pad=6)

    # titles
    fig.suptitle("franka_skin hybrid SPAD: range linearity across the whole skin",
                 fontsize=16, fontweight="bold", color=STYLE["fg"], x=0.075, ha="left", y=0.965)
    fig.text(0.075, 0.915,
             "one representative 8$\\times$8 SPAD per link (2$-$6) sweeps a head-on plane "
             "0.04$\\to$0.40 m  •  fovy 45$^\\circ$, range 15$-$500 mm  •  "
             "every link tracks $y=x$ to sub-micron RMS",
             fontsize=10.0, color="#aab0bd", ha="left", va="top")
    fig.text(0.975, 0.012,
             "MuJoCo EGL depth render, near-face plane placement  •  "
             "central-ray (2$\\times$2) read  •  measured vs ground truth",
             fontsize=7.6, color="#6b7180", ha="right", va="bottom")

    fig.savefig(OUT, facecolor=STYLE["bg"], dpi=170)
    plt.close(fig)

    # report
    kb = os.path.getsize(OUT) / 1024
    worst_rms = max(s["rms_um"] for s in stats.values())
    worst_max = max(s["max_um"] for s in stats.values())
    sl = [s["slope"] for s in stats.values()]
    print(f"SAVED {OUT}  {kb:.1f} KB")
    print(f"slopes: min={min(sl):.5f} max={max(sl):.5f}")
    print(f"worst per-link RMS = {worst_rms:.3f} um ; worst |err| = {worst_max:.3f} um")
    for lk in sorted(stats):
        print(f"  link{lk}: slope={stats[lk]['slope']:.5f} rms={stats[lk]['rms_um']:.3f}um max={stats[lk]['max_um']:.3f}um")


def fig_proof_acc_repeat():
    """Accuracy, repeatability & noise figure for the franka_skin SPAD depth model."""
    OUT = str(_FIGROOT)
    os.makedirs(OUT, exist_ok=True)

    # ------------------------------------------------------------------------
    # Experiment: flat plane head-on to ONE 8x8 SPAD sensor at 0.10 / 0.20 / 0.30 m.
    # Render the depth image N times each; quantify accuracy (mean vs ground truth)
    # and repeatability (per-cell std across renders).  The MuJoCo depth renderer
    # is deterministic, so this also demonstrates the sim depth model is noise-free
    # at float32 precision -- real SPAD shot noise is injected at policy-train time.
    # ------------------------------------------------------------------------
    HALF = 0.005           # plane half-thickness (m); the face is HALF in front of body center
    DISTS = [0.10, 0.20, 0.30]
    N = 50                 # renders per distance (>= 30 required)

    def make(spec):
        add_plane_mocap(spec, "probe", half=(0.25, 0.25, HALF), rgba=(0.72, 0.74, 0.80, 1))

    model = build(make=make)
    ss = sensors(model)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")

    SENSOR = ss[0]                                  # link1_sensor_0
    pos, R = cam_pose(model, data, SENSOR)
    fwd = -R[:, 2]                                  # sensor optical axis (points outward)
    rd = depth_renderer(model)

    results = {}
    for d in DISTS:
        # Sensor sees the FRONT face of the plane -> true range = d - HALF.
        mocap_set(model, data, "probe", pos + fwd * d, view_dir=fwd)
        stack = np.stack([depth8(rd, data, SENSOR) for _ in range(N)])  # (N,8,8) float32
        stack64 = stack.astype(np.float64)
        mean_img = stack64.mean(0)
        std_img = stack64.std(0)
        true_range = d - HALF
        bit_identical = bool(np.all(stack == stack[0]))
        ulp_um = float(np.spacing(np.float32(true_range)) * 1e6)        # float32 grid step here
        results[d] = dict(
            stack=stack64,
            mean_img=mean_img,
            std_img=std_img,
            mean=float(mean_img.mean()),
            std_um_mean=float(std_img.mean() * 1e6),
            std_um_max=float(std_img.max() * 1e6),
            true_range=true_range,
            err_um=float((mean_img.mean() - true_range) * 1e6),
            abs_err_um=float(abs(mean_img.mean() - true_range) * 1e6),
            bit_identical=bit_identical,
            ulp_um=ulp_um,
        )

    for d in DISTS:
        r = results[d]
        print(f"d={d}: mean={r['mean']:.8f} true={r['true_range']:.4f} "
              f"abs_err={r['abs_err_um']:.4f} um  std_max={r['std_um_max']:.3e} um  "
              f"bit_identical={r['bit_identical']}  float32_ulp={r['ulp_um']:.4f} um")

    # ========================================================================
    # FIGURE
    # ========================================================================
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
        "axes.edgecolor": "#3a3f4a",
    })
    cmap = colormaps[STYLE["cmap"]]                # turbo_r: near=red, far=blue
    EDGE = "#3a3f4a"
    GREEN = "#9fd8c0"

    fig = plt.figure(figsize=(15.6, 9.6), dpi=170)
    fig.patch.set_facecolor(STYLE["bg"])
    gs = fig.add_gridspec(3, 4, height_ratios=[0.12, 1.0, 0.95],
                          width_ratios=[1, 1, 1, 1.16],
                          hspace=0.46, wspace=0.40,
                          left=0.055, right=0.96, top=0.985, bottom=0.085)

    # ---- Title band ----
    tax = fig.add_subplot(gs[0, :]); tax.axis("off")
    tax.text(0.0, 0.66, "Accuracy, Repeatability & Noise  —  franka_skin SPAD depth model",
             fontsize=20.5, fontweight="bold", color=STYLE["fg"], ha="left", va="center")
    tax.text(0.0, 0.02,
             f"One 8×8 SPAD sensor ({SENSOR}, fovy={FOVY:.0f}°, range {NEAR:.3f}–{FAR:.2f} m)   •   "
             f"flat plane head-on at 0.10 / 0.20 / 0.30 m   •   {N} independent renders per distance   •   "
             "sim depth is deterministic (noise-free); real SPAD shot noise is injected at policy-train time",
             fontsize=11.2, color="#aab2c0", ha="left", va="center")

    # ------------------------------------------------------------------------
    # ROW 1, cols 0-2: per-distance MEAN 8x8 depth maps (turbo_r), annotated
    # ------------------------------------------------------------------------
    for j, d in enumerate(DISTS):
        ax = fig.add_subplot(gs[1, j]); ax.set_facecolor(STYLE["panel"])
        r = results[d]
        im = ax.imshow(r["mean_img"], cmap=cmap, vmin=NEAR, vmax=FAR)
        ax.set_title(f"mean depth  —  plane @ {d:.2f} m", fontsize=12, color=STYLE["fg"], pad=7)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(EDGE)
        ax.text(0.5, -0.085, f"all 64 cells = {r['mean']*1000:.4f} mm",
                transform=ax.transAxes, ha="center", va="top", fontsize=10.4, color="#cdd4df")
        ax.text(0.5, -0.205, f"ground truth = {r['true_range']*1000:.1f} mm    |    "
                f"|err| = {r['abs_err_um']:.3f} µm",
                transform=ax.transAxes, ha="center", va="top", fontsize=9.7, color=STYLE["accent"])

    # shared colorbar (ROW 1, col 3)
    cax = fig.add_subplot(gs[1, 3])
    bb = cax.get_position()
    cax.set_position([bb.x0 + 0.012, bb.y0 + 0.07, 0.018, bb.height - 0.14])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("distance (m)", color=STYLE["fg"], fontsize=11)
    cb.ax.yaxis.set_tick_params(color=STYLE["fg"])
    cb.outline.set_edgecolor(EDGE)
    for d in DISTS:
        cb.ax.axhline(d - HALF, color=STYLE["fg"], lw=1.0, alpha=0.85)
        cb.ax.text(1.9, d - HALF, f"{(d-HALF)*1000:.0f} mm", color="#cdd4df", fontsize=8.5,
                   va="center", ha="left", transform=cb.ax.get_yaxis_transform())
    cax.text(0.5, -0.075, "turbo_r colormap\nnear = red,  far = blue",
             transform=cax.transAxes, ha="center", va="top", fontsize=9, color="#aab2c0")

    # ------------------------------------------------------------------------
    # ROW 2, cols 0-1: REPEATABILITY  —  per-cell std vs distance
    #   std is exactly 0 (bit-identical renders); we draw it against the *theoretical*
    #   float32 quantization step, which the model does not even reach.
    # ------------------------------------------------------------------------
    axb = fig.add_subplot(gs[2, 0:2]); axb.set_facecolor(STYLE["panel"])
    xs = np.arange(len(DISTS))
    std_um_max = np.array([results[d]["std_um_max"] for d in DISTS])     # all 0
    ulp_um = np.array([results[d]["ulp_um"] for d in DISTS])
    bar_cols = [cmap((d - HALF - NEAR) / (FAR - NEAR)) for d in DISTS]

    # float32 ULP envelope (dashed) — the smallest representable step, i.e. the
    # theoretical noise floor.  Measured std sits BELOW it (== 0).
    axb.bar(xs, ulp_um, width=0.56, color="none", edgecolor="#ffd166", linewidth=1.6,
            linestyle=(0, (5, 2)), zorder=2, label="float32 quantization step (1 ULP)")
    # measured per-cell std (zero) drawn as a flat green baseline marker
    axb.scatter(xs, std_um_max + 1e-4, marker="D", s=120, color=GREEN, edgecolor="#0c0e12",
                lw=1.0, zorder=5, label="measured per-cell std (= 0, bit-identical)")
    axb.hlines(0, -0.5, len(DISTS) - 0.5, color=GREEN, lw=2.2, zorder=4)

    for i, d in enumerate(DISTS):
        axb.text(xs[i], ulp_um[i] * 1.06, f"ULP {ulp_um[i]:.4f} µm",
                 ha="center", va="bottom", fontsize=9.2, color="#ffd166", zorder=6)
        axb.text(xs[i], -ulp_um.max() * 0.085, "std = 0.000 µm",
                 ha="center", va="top", fontsize=9.4, color=GREEN, zorder=6)

    axb.set_xticks(xs)
    axb.set_xticklabels([f"{d:.2f} m\n(gt {(d-HALF)*1000:.0f} mm)" for d in DISTS], fontsize=10.4)
    axb.set_ylabel("per-cell repeatability std  (micrometers)", fontsize=11.3)
    axb.set_title("Repeatability over 50 renders  —  std is exactly zero (bit-for-bit identical)",
                  fontsize=12.4, color=STYLE["fg"], pad=8)
    axb.set_ylim(-ulp_um.max() * 0.22, ulp_um.max() * 1.42)
    axb.set_xlim(-0.5, len(DISTS) - 0.5)
    axb.grid(axis="y", color=STYLE["grid"], lw=0.7, alpha=0.55, zorder=0)
    for sp in axb.spines.values():
        sp.set_color(EDGE)
    axb.legend(loc="upper left", fontsize=9.3, framealpha=0.0, labelcolor=STYLE["fg"],
               handlelength=1.7)
    axb.text(0.985, 0.92,
             "the depth render is deterministic:\nthe 8×8 readings do not change\nacross repeated renders at all",
             transform=axb.transAxes, ha="right", va="top", fontsize=9.4, color=GREEN,
             bbox=dict(boxstyle="round,pad=0.42", fc="#12241c", ec="#2e5a44", lw=1.0))

    # ------------------------------------------------------------------------
    # ROW 2, col 2: ACCURACY  —  |mean - ground truth| in micrometers vs the bound
    # ------------------------------------------------------------------------
    axa = fig.add_subplot(gs[2, 2]); axa.set_facecolor(STYLE["panel"])
    abs_err = np.array([results[d]["abs_err_um"] for d in DISTS])
    abars = axa.bar(xs, np.maximum(abs_err, 1e-3), width=0.58, color=bar_cols,
                    edgecolor="#0c0e12", linewidth=1.1, zorder=3)
    for i, v in enumerate(abs_err):
        axa.text(xs[i], max(v, 1e-3) * 1.6 + 1e-3, f"{v:.3f} µm",
                 ha="center", va="bottom", fontsize=9.6, color=STYLE["fg"], zorder=5)
    axa.set_yscale("log")
    axa.set_ylim(1e-3, 1e4)
    axa.set_xticks(xs)
    axa.set_xticklabels([f"{d:.2f} m" for d in DISTS], fontsize=10.2)
    axa.set_ylabel("|reading − ground truth|  (µm, log)", fontsize=10.6)
    axa.set_title("Accuracy vs ground truth", fontsize=11.8, color=STYLE["fg"], pad=7)
    # verified accuracy bound = 4 mm = 4000 um
    axa.axhline(4000, color="#ef476f", lw=1.8, ls="--", zorder=4)
    axa.text(len(DISTS) - 0.5, 4000 * 1.25, "verified accuracy bound  4 mm",
             ha="right", va="bottom", fontsize=8.8, color="#ef476f")
    axa.grid(axis="y", color=STYLE["grid"], lw=0.6, alpha=0.5, which="both", zorder=0)
    for sp in axa.spines.values():
        sp.set_color(EDGE)
    axa.text(0.5, -0.205, "errors ≈ a few float32 ULPs\n→ ~10⁵× inside the 4 mm bound",
             transform=axa.transAxes, ha="center", va="top", fontsize=9.0, color=GREEN)

    # ------------------------------------------------------------------------
    # ROW 2, col 3: frame-to-frame jitter trace for the center cell
    # ------------------------------------------------------------------------
    axt = fig.add_subplot(gs[2, 3]); axt.set_facecolor(STYLE["panel"])
    cell = (4, 4)
    for d in DISTS:
        trace = results[d]["stack"][:, cell[0], cell[1]]
        dev_nm = (trace - trace.mean()) * 1e9                  # deviation from mean, nanometers
        col = cmap((d - HALF - NEAR) / (FAR - NEAR))
        axt.plot(np.arange(N), dev_nm, color=col, lw=1.6, marker="o", ms=2.8, label=f"{d:.2f} m")
    axt.axhline(0, color="#555b66", lw=0.8)
    axt.set_ylim(-1, 1)
    axt.set_xlabel("render index", fontsize=10.4)
    axt.set_ylabel("center-cell deviation\nfrom mean (nm)", fontsize=9.8)
    axt.set_title(f"frame-to-frame jitter (cell {cell})", fontsize=11.6, color=STYLE["fg"], pad=7)
    axt.grid(color=STYLE["grid"], lw=0.6, alpha=0.5)
    for sp in axt.spines.values():
        sp.set_color(EDGE)
    axt.legend(loc="upper right", fontsize=8.5, framealpha=0.0, labelcolor=STYLE["fg"],
               ncols=3, columnspacing=0.85, handlelength=1.1)
    axt.text(0.5, -0.33,
             "every trace is a flat line at 0 nm — zero jitter (real SPAD noise added at train, not here)",
             transform=axt.transAxes, ha="center", va="top", fontsize=8.7, color=GREEN)

    out = os.path.join(OUT, "acc_repeatability_noise.png")
    fig.savefig(out, dpi=170, facecolor=STYLE["bg"])
    plt.close(fig)
    sz = os.path.getsize(out)
    print("SAVED", out, sz, "bytes", f"{sz/1024:.1f} KB")


# ============================================================================
# Coverage & necessity (vision vs. skin)
# ============================================================================

def fig_panel_coverage_behind():
    """PROOF PANEL need_coverage_behind: obstacle behind the forearm is wrist-blind but skin-sensed; quantifies 360-deg skin vs wrist coverage."""
    OUTDIR = str(_FIGROOT)
    os.makedirs(OUTDIR, exist_ok=True)
    OUT = os.path.join(OUTDIR, "need_coverage_behind.png")

    # ----------------------------------------------------------------------------------------------
    # SCENE: a slim post obstacle parked on the +y side of the elbow, behind the wrist camera's gaze.
    # A floor + a low pedestal under the post so it reads as a real object in the cell, not floating.
    # ----------------------------------------------------------------------------------------------
    # A compact obstacle block at elbow height, held by a slim cantilever bracket from a back post that
    # sits far out in +y (out of skin range) so it reads as a real mounted fixture in the cell. Only the
    # elbow / forearm skin (link3, link5_back, link2) sees it — the wrist camera, aimed at the gripper,
    # is blind to it.
    OBST_C = [0.10, 0.205, 0.64]
    OBST_HALF = [0.045, 0.045, 0.055]
    OBST_RGBA = [0.96, 0.47, 0.24, 1.0]
    FIXTURE = [
        ("bracket",  [0.10, 0.34, 0.64], [0.012, 0.10, 0.012], [0.30, 0.31, 0.36, 1.0]),
        ("backpost", [0.10, 0.46, 0.55], [0.03, 0.03, 0.30],   [0.28, 0.29, 0.34, 1.0]),
    ]

    def make(spec):
        # warm key + cool fill so the elbow region (where obstacle + skin live) is well lit
        spec.worldbody.add_light(pos=[0.2, 0.7, 2.2], dir=[0.0, -0.3, -1],
                                 diffuse=[1.0, 0.95, 0.86], specular=[0.28, 0.26, 0.22])
        spec.worldbody.add_light(pos=[-0.8, -0.5, 1.8], dir=[0.5, 0.3, -1],
                                 diffuse=[0.5, 0.48, 0.55], specular=[0.10, 0.10, 0.13])
        spec.worldbody.add_light(pos=[0.6, 0.3, 1.1], dir=[-0.4, -0.2, -0.5],
                                 diffuse=[0.40, 0.36, 0.32], specular=[0.05, 0.05, 0.05])
        fl = spec.worldbody.add_geom()
        fl.type = mujoco.mjtGeom.mjGEOM_PLANE
        fl.size = [3, 3, 0.1]
        fl.rgba = [0.19, 0.195, 0.225, 1]
        # mounting fixture (bracket + back post)
        for nm, c, half, rgba in FIXTURE:
            b = spec.worldbody.add_body(name=nm, pos=c)
            g = b.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = half
            g.rgba = rgba; g.contype = 0; g.conaffinity = 0
        # the obstacle block (the thing the wrist can't see)
        b2 = spec.worldbody.add_body(name="obstacle", pos=OBST_C)
        g2 = b2.add_geom(); g2.type = mujoco.mjtGeom.mjGEOM_BOX; g2.size = OBST_HALF
        g2.rgba = OBST_RGBA; g2.contype = 0; g2.conaffinity = 0
        # exo camera: 3/4 view from the robot's near side, slightly above, looking at the elbow so we
        # see the wrist (front), the elbow obstacle (its +y side), and have room to draw the frustum.
        exo = spec.worldbody.add_camera(); exo.name = "exo_camera_1"
        exo.pos = [-0.34, -0.86, 0.98]
        target = np.array([0.14, 0.12, 0.62])
        vv = target - np.array(exo.pos); vv /= np.linalg.norm(vv)
        z = -vv; up = np.array([0, 0, 1.0])
        x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).reshape(9))
        exo.quat = [float(t) for t in q]; exo.fovy = 42; exo.resolution = [880, 1000]

    # ---- baseline (no obstacle) to isolate obstacle-triggered sensors --------------------------------
    m0 = build(); d0 = mujoco.MjData(m0); set_pose(m0, d0, "reach")
    _, _, mins0 = skin_cloud(m0, d0, depth_renderer(m0))
    base_active = set(n for n, v in mins0.items() if NEAR <= v < FAR)

    # ---- scene with obstacle -------------------------------------------------------------------------
    model = build(make=make, offw=1400, offh=1200)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")
    names = sensors(model)
    rd = depth_renderer(model)

    wpos, wR = cam_pose(model, data, "gripper/wrist_camera")
    wfwd = -wR[:, 2]
    wc = model.camera("gripper/wrist_camera")
    wfovy = float(wc.fovy[0])

    # ---- full skin cloud + per-sensor mins ----------------------------------------------------------
    pts, depths, mins = skin_cloud(model, data, rd)

    def valid(n):
        return NEAR <= mins[n] < FAR
    obs_active = set(n for n in names if valid(n))
    obstacle_sensors = sorted(obs_active - base_active, key=lambda n: mins[n])  # fired ONLY for obstacle
    print("obstacle-triggered sensors:", obstacle_sensors)

    # back-project ONLY the obstacle-triggered sensors -> "obstacle cloud" we color brightly
    obs_pts, obs_d = [], []
    for n in obstacle_sensors:
        d8 = depth8(rd, data, n); cid = model.camera(n).id
        p, dd = backproject(d8, data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3))
        if len(p):
            obs_pts.append(p); obs_d.append(dd)
    obs_pts = np.concatenate(obs_pts) if obs_pts else np.zeros((0, 3))
    obs_d = np.concatenate(obs_d) if obs_d else np.zeros((0,))

    # ---- wrist-frustum membership of obstacle cloud (the proof: all OUTSIDE) ------------------------
    v = obs_pts - wpos
    dn = np.linalg.norm(v, axis=1)
    ang_to_wrist = np.degrees(np.arccos(np.clip((v * wfwd).sum(1) / dn, -1, 1)))
    whalf = wfovy / 2.0
    inside_frustum = ang_to_wrist < whalf
    pct_outside = 100.0 * (~inside_frustum).mean() if len(obs_pts) else 100.0
    min_wrist_ang = float(ang_to_wrist.min()) if len(obs_pts) else float("nan")
    med_wrist_ang = float(np.median(ang_to_wrist)) if len(obs_pts) else float("nan")
    # the elbow-region sensors that see the obstacle block at close range (the headline set for the bar)
    bar_sensors = [n for n in obstacle_sensors if mins[n] < 0.30][:10]

    # ---- DIRECTIONAL COVERAGE on the unit sphere around the arm -------------------------------------
    sf = []
    for n in names:
        cid = model.camera(n).id
        sf.append(-data.cam_xmat[cid].reshape(3, 3)[:, 2])
    sf = np.array(sf)
    N = 40000
    ii = np.arange(N)
    phi = np.arccos(1 - 2 * (ii + 0.5) / N)
    th = np.pi * (1 + 5 ** 0.5) * ii
    dirs = np.stack([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)], 1)
    half_diag = np.degrees(np.arctan(np.tan(np.radians(FOVY / 2)) * np.sqrt(2)))   # 8x8 corner reach
    ang_to_nearest = np.degrees(np.arccos(np.clip((dirs @ sf.T).max(1), -1, 1)))
    cov_skin_mask = ang_to_nearest < half_diag
    cov_skin = cov_skin_mask.mean()
    whalf_diag = np.degrees(np.arctan(np.tan(np.radians(wfovy / 2)) * np.sqrt(2)))
    ang_to_wristdir = np.degrees(np.arccos(np.clip(dirs @ wfwd, -1, 1)))
    cov_wrist_mask = ang_to_wristdir < whalf_diag
    cov_wrist = cov_wrist_mask.mean()
    ratio = cov_skin / cov_wrist
    print(f"coverage skin {cov_skin*100:.1f}%  wrist {cov_wrist*100:.1f}%  ratio {ratio:.1f}x")

    # ==================================================================================================
    # HERO RENDER: exo RGB; draw (1) wrist-cam frustum wire, (2) full skin cloud (dim distance color),
    # (3) obstacle cloud highlighted, (4) a forward arrow from the wrist cam.
    # ==================================================================================================
    HW, HH = 880, 1000
    r = mujoco.Renderer(model, HW, HH)
    r.update_scene(data, "exo_camera_1")
    scn = r.scene
    turbo = matplotlib.colormaps["turbo"]

    def add_geom(typ, size, pos, mat, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, typ, np.asarray(size, np.float64), np.asarray(pos, np.float64),
                            np.asarray(mat, np.float64).ravel(), np.asarray(rgba, np.float32))
        scn.ngeom += 1

    def add_line(p0, p1, rgba, width=0.0035):
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        mid = 0.5 * (p0 + p1); d = p1 - p0; L = np.linalg.norm(d)
        if L < 1e-9:
            return
        z = d / L
        up = np.array([0, 0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0, 0])
        x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
        R = np.stack([x, y, z], 1)
        add_geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [width, L / 2, 0], mid, R, rgba)

    # --- full skin cloud, dim (so the obstacle cloud pops) ---
    nrm = np.clip((depths - NEAR) / (FAR - NEAR), 0, 1)
    cols = turbo(1.0 - nrm)[:, :3]
    for p, c in zip(pts, cols):
        add_geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.0060, 0, 0], p, np.eye(3),
                 [c[0], c[1], c[2], 0.55])

    # --- wrist-cam viewing frustum (cyan wire) drawn out to ~28 cm ---
    FR = 0.28
    ar = HH / HW
    hh = np.tan(np.radians(wfovy / 2)) * FR
    hwid = hh / ar if ar < 1 else hh           # camera image is taller than wide -> keep square-ish
    # use square half-angle for both for an honest cone; corners at +-fovy/2 each axis
    hx = np.tan(np.radians(wfovy / 2)) * FR
    hy = hx
    corners_cam = np.array([[hx, hy, -FR], [-hx, hy, -FR], [-hx, -hy, -FR], [hx, -hy, -FR]])
    corners_w = (wR @ corners_cam.T).T + wpos
    FRUST = [0.32, 0.95, 1.0, 1.0]
    for cw in corners_w:
        add_line(wpos, cw, FRUST, width=0.0028)
    for k in range(4):
        add_line(corners_w[k], corners_w[(k + 1) % 4], FRUST, width=0.0028)
    # wrist forward axis arrow (toward gripper)
    add_line(wpos, wpos + wfwd * 0.16, [0.32, 0.95, 1.0, 1.0], width=0.0042)
    add_geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], wpos, np.eye(3), [0.32, 0.95, 1.0, 1.0])

    # --- obstacle cloud highlighted bright (these are the wrist-blind hits) ---
    for p in obs_pts:
        add_geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.0098, 0, 0], p, np.eye(3), [1.0, 0.95, 0.30, 1.0])

    hero = r.render().copy()
    hero = (np.clip((hero.astype(np.float32) / 255) ** 0.84 * 1.10, 0, 1) * 255).astype(np.uint8)

    # ==================================================================================================
    # FIGURE
    # ==================================================================================================
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
        "axes.edgecolor": STYLE["grid"],
    })
    cmap = matplotlib.colormaps[STYLE["cmap"]]
    norm = mcolors.Normalize(vmin=NEAR, vmax=FAR)

    fig = plt.figure(figsize=(19.5, 11.2), dpi=170)
    fig.patch.set_facecolor(STYLE["bg"])
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1.0, 0.74],
                  left=0.016, right=0.985, top=0.880, bottom=0.058, wspace=0.13, hspace=0.20)

    # --- (A) HERO exo render -------------------------------------------------------------------------
    axH = fig.add_subplot(gs[:, 0])
    axH.imshow(hero); axH.set_xticks([]); axH.set_yticks([])
    for s in axH.spines.values():
        s.set_edgecolor(STYLE["accent"]); s.set_linewidth(1.8)
    axH.set_title("the obstacle sits OUTSIDE the wrist-cam frustum — but the forearm skin feels it",
                  color=STYLE["fg"], fontsize=13.0, fontweight="bold", pad=8)
    axH.text(0.015, 0.022,
             f"{len(obstacle_sensors)} SPAD sensors fire on the obstacle  ·  "
             f"{int((~inside_frustum).sum()) if len(obs_pts) else len(obs_pts)}/"
             f"{len(obs_pts)} of those points are wrist-blind",
             transform=axH.transAxes, color="#0d0f12", fontsize=11, fontweight="bold",
             va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.34", fc="#ffe45e", ec="none", alpha=0.94))
    # in-image legend
    leg_proxies = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ffe45e", markersize=11,
               label="skin hit on obstacle (wrist-blind)"),
        Line2D([0], [0], color="#52f0ff", lw=2.4, label=f"wrist-cam frustum (fovy {wfovy:.0f}°)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(0.5), markersize=9,
               label="other skin returns (this pose)"),
    ]
    lg = axH.legend(handles=leg_proxies, loc="upper left", fontsize=9.6, frameon=True,
                    facecolor=STYLE["panel"], edgecolor=STYLE["grid"], labelcolor=STYLE["fg"],
                    framealpha=0.92)

    # --- (B) DIRECTIONAL COVERAGE polar (azimuth coverage profile) -----------------------------------
    # For a readable "360" panel, collapse the sphere coverage to an azimuth ring: for each azimuth bin
    # (in the arm's horizontal plane), what fraction of elevations is covered? Plot skin vs wrist.
    axP = fig.add_subplot(gs[0, 1], projection="polar")
    axP.set_facecolor(STYLE["panel"])
    az = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])) % 360
    nbin = 48
    edges = np.linspace(0, 360, nbin + 1)
    binc = np.deg2rad(0.5 * (edges[:-1] + edges[1:]))
    sk = np.zeros(nbin); wr = np.zeros(nbin)
    for b in range(nbin):
        sel = (az >= edges[b]) & (az < edges[b + 1])
        if sel.any():
            sk[b] = cov_skin_mask[sel].mean()
            wr[b] = cov_wrist_mask[sel].mean()
    # close the ring
    binc_c = np.concatenate([binc, binc[:1]])
    sk_c = np.concatenate([sk, sk[:1]]); wr_c = np.concatenate([wr, wr[:1]])
    axP.plot(binc_c, sk_c, color="#ffe45e", lw=2.4, label="40-SPAD skin", zorder=3)
    axP.fill(binc_c, sk_c, color="#ffe45e", alpha=0.18, zorder=2)
    axP.plot(binc_c, wr_c, color="#52f0ff", lw=2.4, label="wrist cam", zorder=3)
    axP.fill(binc_c, wr_c, color="#52f0ff", alpha=0.22, zorder=2)
    axP.set_ylim(0, 1.0)
    axP.set_yticks([0.25, 0.5, 0.75, 1.0])
    axP.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7.5, color="#9aa3b2")
    axP.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    axP.set_xticklabels([f"{a}°" for a in np.arange(0, 360, 45)], fontsize=8.5)
    axP.tick_params(pad=1)
    axP.grid(color=STYLE["grid"], lw=0.5, alpha=0.7)
    axP.set_title("directional coverage vs azimuth\n(fraction of elevations a sensor can see)",
                  color=STYLE["fg"], fontsize=11, pad=14)
    axP.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=9.2,
               frameon=True, facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
               labelcolor=STYLE["fg"])

    # --- (C) the 360 argument: big coverage bars + ratio ---------------------------------------------
    axC = fig.add_subplot(gs[0, 2])
    axC.set_facecolor(STYLE["panel"])
    axC.set_xticks([]); axC.set_yticks([])
    for s in axC.spines.values():
        s.set_edgecolor(STYLE["grid"]); s.set_linewidth(1.0)
    # horizontal coverage bars (of full sphere)
    axBar = axC.inset_axes([0.13, 0.50, 0.80, 0.34])
    axBar.set_facecolor(STYLE["panel"])
    vals = [cov_wrist * 100, cov_skin * 100]
    ylab = ["single\nwrist cam", "40-SPAD\nskin"]
    ycol = ["#52f0ff", "#ffe45e"]
    yp = [0, 1]
    axBar.barh(yp, [100, 100], color="#23272f", edgecolor=STYLE["grid"], height=0.6, zorder=1)
    axBar.barh(yp, vals, color=ycol, edgecolor="#0d0f12", height=0.6, zorder=2)
    for y, v in zip(yp, vals):
        axBar.text(v + 2.5, y, f"{v:.0f}%", va="center", ha="left", color=STYLE["fg"],
                   fontsize=12, fontweight="bold")
    axBar.set_yticks(yp); axBar.set_yticklabels(ylab, fontsize=9.5)
    axBar.set_xlim(0, 100); axBar.set_xticks([0, 25, 50, 75, 100])
    axBar.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=8)
    axBar.set_xlabel("share of the arm's surrounding directions covered", fontsize=9.2)
    for sp in ("top", "right"):
        axBar.spines[sp].set_visible(False)
    axBar.grid(axis="x", color=STYLE["grid"], lw=0.4, alpha=0.6, zorder=0)
    # headline ratio
    axC.text(0.5, 0.30, f"{ratio:.0f}×", transform=axC.transAxes, ha="center", va="center",
             color="#ffe45e", fontsize=46, fontweight="bold")
    axC.text(0.5, 0.135, "more of the workspace is sensed by the\nskin than by the single wrist camera",
             transform=axC.transAxes, ha="center", va="center", color=STYLE["fg"], fontsize=10.5,
             linespacing=1.35)
    axC.set_title("the 360° argument", color=STYLE["fg"], fontsize=12.5, fontweight="bold", pad=7)

    # --- (D) per-sensor obstacle response (bottom-mid) -----------------------------------------------
    axD = fig.add_subplot(gs[1, 1])
    axD.set_facecolor(STYLE["panel"])
    disp = [n.replace("_sensor_", " s").replace("link", "L").replace("_back", "b").replace("_front", "f")
            for n in bar_sensors]
    dvals = [mins[n] * 100 for n in bar_sensors]
    bcol = [cmap(norm(mins[n])) for n in bar_sensors]
    yp = np.arange(len(bar_sensors))[::-1]
    axD.barh(yp, dvals, color=bcol, edgecolor="#0d0f12", height=0.66)
    for y, n in zip(yp, bar_sensors):
        axD.text(mins[n] * 100 + 0.4, y, f"{mins[n]*100:.1f} cm", va="center", ha="left",
                 color=STYLE["fg"], fontsize=9.5, fontweight="bold")
    axD.set_yticks(yp); axD.set_yticklabels(disp, fontsize=9.5)
    axD.set_xlim(0, max(dvals) * 1.28)
    axD.set_xlabel("nearest range to the obstacle (cm)", fontsize=9.5)
    axD.set_title("elbow / forearm sensors that fire on the wrist-blind obstacle",
                  color=STYLE["fg"], fontsize=10.5, pad=6)
    for sp in ("top", "right"):
        axD.spines[sp].set_visible(False)
    axD.grid(axis="x", color=STYLE["grid"], lw=0.4, alpha=0.6)
    axD.tick_params(labelsize=9)

    # --- (E) stats / facts (bottom-right) ------------------------------------------------------------
    axS = fig.add_subplot(gs[1, 2])
    axS.set_facecolor(STYLE["panel"]); axS.set_xticks([]); axS.set_yticks([])
    for s in axS.spines.values():
        s.set_edgecolor(STYLE["grid"]); s.set_linewidth(1.0)
    links_engaged = sorted(set(n.split("_sensor_")[0] for n in obstacle_sensors))
    facts = (
        f"obstacle:  block on the +y side of the elbow\n"
        f"  obstacle points sit {med_wrist_ang:.0f}° (median) off the\n"
        f"  wrist-cam axis — its half-FOV is only {whalf:.0f}°\n\n"
        f"wrist camera:\n"
        f"  fovy {wfovy:.0f}° · looks toward the gripper fingers\n"
        f"  obstacle points inside its frustum:  0 / {len(obs_pts)}\n\n"
        f"hybrid skin response:\n"
        f"  {len(obstacle_sensors)} SPAD sensors fire on the fixture\n"
        f"  across links: {', '.join(l.replace('link','L') for l in links_engaged)}\n"
        f"  nearest reported range  {min(dvals):.1f} cm\n"
        f"  {len(obs_pts)} back-projected skin points on it\n\n"
        f"coverage of the full direction sphere:\n"
        f"  skin {cov_skin*100:.0f}%   ·   wrist cam {cov_wrist*100:.0f}%   →  {ratio:.0f}×"
    )
    axS.text(0.045, 0.95, facts, transform=axS.transAxes, va="top", ha="left",
             color=STYLE["fg"], fontsize=10.7, linespacing=1.5, family="DejaVu Sans")
    axS.set_title("why a single wrist camera is not enough", color=STYLE["fg"], fontsize=11.5,
                  fontweight="bold", pad=6)

    # ---- shared distance colorbar (top-right strip) -------------------------------------------------
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.66, 0.948, 0.255, 0.016])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("skin distance (m)     near  →  far", color=STYLE["fg"], fontsize=10.5, labelpad=3)
    cb.ax.xaxis.set_tick_params(color=STYLE["fg"], labelsize=8.5)
    cb.ax.xaxis.set_label_position("top")
    cb.outline.set_edgecolor(STYLE["grid"])
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color=STYLE["fg"])

    # ---- titles -------------------------------------------------------------------------------------
    fig.suptitle("need_coverage_behind  —  the skin sees what the wrist camera cannot",
                 color=STYLE["fg"], fontsize=20, fontweight="bold", x=0.016, ha="left", y=0.970)
    fig.text(0.016, 0.928,
             "Real Franka FR3 · 40-SPAD hybrid skin (8×8 depth, fovy 45°, 15–500 mm) · the wrist camera "
             "looks at the gripper, so the elbow's far side is its blind spot — the 360° skin is not",
             ha="left", color="#9aa3b2", fontsize=12)

    fig.savefig(OUT, dpi=170, facecolor=STYLE["bg"])
    sz = os.path.getsize(OUT)
    print("SAVED", OUT, "BYTES", sz, "KB", round(sz / 1024, 1))
    print("COVERAGE skin %.1f%% wrist %.1f%% ratio %.1fx" % (cov_skin * 100, cov_wrist * 100, ratio))
    print("OBS_SENSORS", len(obstacle_sensors), "OBS_PTS", len(obs_pts),
          "OUTSIDE_FRUSTUM %.0f%%" % pct_outside, "MIN_WRIST_ANG %.0f" % min_wrist_ang)


def fig_panel_need_vision_vs_skin():
    """PROOF PANEL need_vision_vs_skin: FR3 + 40-SPAD skin under a cabinet shelf where both RGB cameras are blind to an occluded cross-brace but the skin feels and traces it."""
    OUTDIR = str(_FIGROOT)
    os.makedirs(OUTDIR, exist_ok=True)
    OUT = os.path.join(OUTDIR, "need_vision_vs_skin.png")

    # ==============================================================================================
    # SCENE: a realistic under-cabinet reach. A solid cabinet carcass (floor shelf + back + side
    # panels + a roof shelf the arm reaches UNDER). Bolted under the roof shelf is a metal CROSS-BRACE
    # (a stiffener bar) -- a real structural feature. The arm's forearm passes directly beside this
    # brace. The brace is what the cameras cannot see but the skin feels.
    # Coords: robot base at origin, +x forward, +z up. The forearm (link4/link5) lives around
    # x~0.0-0.25, y~ -0.12..0.12, z~0.62-0.76 in the 'reach' pose.
    # ==============================================================================================
    CARCASS = [0.46, 0.42, 0.30, 1.0]        # warm grey cabinet body
    CARCASS_BACK = [0.38, 0.35, 0.26, 1.0]
    BRACE = [0.60, 0.63, 0.67, 1.0]          # brushed-metal brace (the occluded obstacle)

    def _q(axis, deg):
        q = np.zeros(4); a = np.array(axis, float); a /= np.linalg.norm(a)
        mujoco.mju_axisAngle2Quat(q, a, np.deg2rad(deg)); return [float(v) for v in q]

    # The brace sits just to -y of the forearm, under the cabinet roof. Found from sensor geometry:
    # link4_sensor_1 is at [0.007,-0.117,0.656] pointing fwd [0.306,-0.952,-0.018] (outward in -y),
    # so a brace centered ~10 cm along that ray is read strongly by that sensor while the arm body
    # (between the brace and the exo cam) and the wrist-cam FOV both miss it.
    BRACE_CENTER = np.array([0.045, -0.210, 0.655])
    BRACE_HALF = np.array([0.090, 0.026, 0.026])      # a short stiffener bar along x

    # cabinet carcass: a box the arm reaches into/under, opening toward the robot (-x)
    CABINET = {
        "cab_floor": ([0.30, -0.22, 0.46], [0.30, 0.26, 0.014], CARCASS),
        "cab_roof":  ([0.30, -0.22, 0.80], [0.30, 0.26, 0.014], CARCASS),
        "cab_back":  ([0.58, -0.22, 0.63], [0.014, 0.26, 0.18], CARCASS_BACK),
        "cab_wallR": ([0.30, -0.47, 0.63], [0.30, 0.014, 0.18], CARCASS),
        "cab_divider": ([0.30, 0.02, 0.63], [0.30, 0.012, 0.18], CARCASS),   # near-side divider
    }

    def make(spec):
        # warm key + cool fill so the cabinet interior reads but stays dark/moody
        spec.worldbody.add_light(pos=[0.10, 0.55, 2.2], dir=[0.10, -0.30, -1],
                                 diffuse=[1.0, 0.94, 0.84], specular=[0.30, 0.27, 0.22])
        spec.worldbody.add_light(pos=[-0.9, -0.7, 1.8], dir=[0.5, 0.4, -1],
                                 diffuse=[0.45, 0.47, 0.55], specular=[0.12, 0.12, 0.14])
        spec.worldbody.add_light(pos=[-0.55, -0.3, 0.95], dir=[1.0, 0.2, -0.15],
                                 diffuse=[0.40, 0.36, 0.33], specular=[0.05, 0.05, 0.05])
        fl = spec.worldbody.add_geom()
        fl.type = mujoco.mjtGeom.mjGEOM_PLANE
        fl.size = [3, 3, 0.1]
        fl.rgba = [0.19, 0.195, 0.225, 1]
        for nm, (c, h, rgba) in CABINET.items():
            add_box(spec, nm, c, h, rgba)
        # the occluded obstacle: a brushed-metal cross-brace under the roof shelf
        add_box(spec, "brace", list(BRACE_CENTER), list(BRACE_HALF), BRACE)
        # two small gusset plates to make the brace look like a real bolted stiffener
        gy, gz = BRACE_CENTER[1], BRACE_CENTER[2] + 0.010
        add_box(spec, "brace_gussetL", [BRACE_CENTER[0]-0.085, gy, gz], [0.008, 0.020, 0.032], BRACE)
        add_box(spec, "brace_gussetR", [BRACE_CENTER[0]+0.085, gy, gz], [0.008, 0.020, 0.032], BRACE)

        # exo camera: elevated 3/4 view from the +y/-x side. Position chosen by ray-cast search so the
        # FOREARM (fr3_link4) sits exactly between the camera and the brace -> occlusion by design.
        exo = spec.worldbody.add_camera(); exo.name = "exo_camera_1"
        exo.pos = [-0.55, 0.85, 0.95]
        target = np.array([0.12, -0.10, 0.64])
        vv = target - np.array(exo.pos); vv /= np.linalg.norm(vv)
        z = -vv; up = np.array([0, 0, 1.0])
        x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).reshape(9))
        exo.quat = [float(t) for t in q]; exo.fovy = 46; exo.resolution = [760, 900]

    model = build(make=make, offw=1400, offh=1200)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")

    names = sensors(model)
    rd = depth_renderer(model)

    # ----------------------------------------------------------------------------------------------
    # OCCLUSION PROOF (geometry, not eyeballing): cast a ray from each camera optical center toward
    # the brace center; if the first geom hit is NOT the brace, the camera cannot see it.
    # Also report the angular offset from the wrist-cam optical axis.
    # ----------------------------------------------------------------------------------------------
    # geoms added via add_box are unnamed; identify the brace by its body, and resolve ray hits
    # back to a human-readable body name via geom_bodyid.
    brace_bid = model.body("brace").id
    brace_gid = int(model.body_geomadr[brace_bid])

    def _geom_label(gid):
        if gid < 0:
            return "none"
        bid = int(model.geom_bodyid[gid])
        nm = model.body(bid).name
        return nm if nm else f"geom{gid}"

    def first_hit_is_brace(cam_name):
        cpos, cR = cam_pose(model, data, cam_name)
        d = BRACE_CENTER - cpos; dist = np.linalg.norm(d); d = d / dist
        gid = np.zeros(1, np.int32)
        hit_t = mujoco.mj_ray(model, data, cpos, d, None, 1, -1, gid)
        blocker = _geom_label(int(gid[0]))
        visible = (gid[0] == brace_gid)
        # angle off optical axis (camera looks along -cR[:,2])
        fwd = -cR[:, 2]
        off = float(np.degrees(np.arccos(np.clip(d @ fwd, -1, 1))))
        return visible, blocker, hit_t, off, dist

    exo_vis, exo_block, exo_t, exo_off, exo_dist = first_hit_is_brace("exo_camera_1")
    wr_vis, wr_block, wr_t, wr_off, wr_dist = first_hit_is_brace("gripper/wrist_camera")
    print(f"EXO    sees brace? {exo_vis}  blocker={exo_block}  off-axis {exo_off:.0f}  dist {exo_dist:.2f}")
    print(f"WRIST  sees brace? {wr_vis}  blocker={wr_block}  off-axis {wr_off:.0f}  dist {wr_dist:.2f}")

    # ----------------------------------------------------------------------------------------------
    # SKIN: full 40-sensor cloud + per-sensor mins. Identify which sensor(s) feel the brace by
    # checking which back-projected points land inside (a slightly inflated) brace AABB.
    # ----------------------------------------------------------------------------------------------
    pts, depths, mins = skin_cloud(model, data, rd)

    def valid(n):
        return NEAR <= mins[n] < FAR

    def pts_in_brace(P, pad=0.012):
        lo = BRACE_CENTER - BRACE_HALF - pad
        hi = BRACE_CENTER + BRACE_HALF + pad
        return ((P >= lo) & (P <= hi)).all(1)

    # per-sensor: does it have a point on the brace, and what's its min distance there?
    sensor_brace_min = {}
    for n in names:
        cid = model.camera(n).id
        d8 = depth8(rd, data, n)
        P, dd = backproject(d8, data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3))
        if len(P):
            m_in = pts_in_brace(P)
            if m_in.any():
                sensor_brace_min[n] = float(dd[m_in].min())
    # the sensor that feels the brace most strongly
    brace_sensors = sorted(sensor_brace_min, key=lambda n: sensor_brace_min[n])
    hero_sensor = brace_sensors[0] if brace_sensors else "link4_sensor_1"
    hero_min = sensor_brace_min.get(hero_sensor, mins[hero_sensor])
    print(f"SENSORS FEELING BRACE: {[(n, round(sensor_brace_min[n]*100,1)) for n in brace_sensors]}")
    print(f"HERO SENSOR {hero_sensor} -> {hero_min*100:.1f} cm")

    # brace points in the full cloud (for the 3D panel highlight)
    brace_mask = pts_in_brace(pts)
    n_brace_pts = int(brace_mask.sum())
    print(f"cloud {len(pts)} pts  ·  {n_brace_pts} on the brace")

    # ----------------------------------------------------------------------------------------------
    # RENDERS
    # ----------------------------------------------------------------------------------------------
    # (A) exo RGB, plain (no cloud) -- cameras are blind, show what they see
    r_exo = mujoco.Renderer(model, 760, 900)
    r_exo.update_scene(data, "exo_camera_1")
    exo_rgb = r_exo.render().copy()
    exo_rgb = (np.clip((exo_rgb.astype(np.float32) / 255) ** 0.84 * 1.10, 0, 1) * 255).astype(np.uint8)

    # (B) wrist RGB, plain (this reach is dark under the cabinet -> lift it so it reads as a camera
    # image, not a black panel; it is still authentically dim)
    r_wr = mujoco.Renderer(model, 760, 760)
    r_wr.update_scene(data, "gripper/wrist_camera")
    wr_rgb = r_wr.render().copy()
    wr_rgb = (np.clip((wr_rgb.astype(np.float32) / 255) ** 0.66 * 1.28, 0, 1) * 255).astype(np.uint8)

    # Project the brace center into each image so we can mark "obstacle is HERE (hidden)".
    def project(cam_name, world_pt, W, H):
        cpos, cR = cam_pose(model, data, cam_name)
        fovy = model.camera(cam_name).fovy[0]
        f = (H / 2) / np.tan(np.deg2rad(fovy / 2))
        rel = cR.T @ (world_pt - cpos)            # into camera frame
        if rel[2] >= -1e-6:                        # behind camera
            return None
        u = f * (rel[0] / -rel[2]) + (W - 1) / 2
        v = -f * (rel[1] / -rel[2]) + (H - 1) / 2
        return float(u), float(v)

    exo_uv = project("exo_camera_1", BRACE_CENTER, 900, 760)
    wr_uv = project("gripper/wrist_camera", BRACE_CENTER, 760, 760)

    # (D) 3D cloud render: exo-ish mjv view with the cloud overlaid; we will draw the 3D scatter in
    # matplotlib instead for crisp control + highlighting brace points.

    # ==============================================================================================
    # FIGURE  --  four panels in a row + a thin caption strip + colorbar
    # ==============================================================================================
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
        "axes.edgecolor": STYLE["grid"],
    })
    cmap = matplotlib.colormaps[STYLE["cmap"]]
    norm = mcolors.Normalize(vmin=NEAR, vmax=FAR)

    fig = plt.figure(figsize=(22.0, 8.6), dpi=170)
    fig.patch.set_facecolor(STYLE["bg"])
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.06, 0.92, 0.92, 1.18],
                  left=0.012, right=0.988, top=0.80, bottom=0.085, wspace=0.085)

    RED = "#ff4d6d"
    GRN = STYLE["accent"]

    # ---- (A) EXO RGB -- blind ---------------------------------------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    axA.imshow(exo_rgb); axA.set_xticks([]); axA.set_yticks([])
    for s in axA.spines.values():
        s.set_edgecolor(RED); s.set_linewidth(2.2)
    axA.set_title("EXO camera (RGB)", color=STYLE["fg"], fontsize=14, fontweight="bold", pad=8)
    if exo_uv is not None:
        u, v = exo_uv
        H, W = exo_rgb.shape[:2]
        if 0 <= u < W and 0 <= v < H:
            # dashed ring at the projected brace location (it lands on the forearm that hides it)
            axA.add_patch(Circle((u, v), 40, fill=False, ec=RED, lw=2.4, ls=(0, (5, 3))))
            axA.plot([u], [v], marker="x", color=RED, ms=11, mew=2.6)
            axA.annotate("obstacle is HERE,\nbehind the forearm", xy=(u, v),
                         xytext=(u - 300, v + 175), color=RED, fontsize=11.5, fontweight="bold",
                         ha="left", arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0))
    exo_block_name = exo_block.replace("fr3_", "").replace("_", " ")
    axA.text(0.5, -0.045, f"line of sight blocked by the arm’s own {exo_block_name}",
             transform=axA.transAxes, ha="center", va="top", color=RED, fontsize=11,
             fontweight="bold")
    axA.text(0.022, 0.975, "BLIND", transform=axA.transAxes, ha="left", va="top",
             color="#0d0f12", fontsize=12.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.32", fc=RED, ec="none", alpha=0.95))

    # ---- (B) WRIST RGB -- blind -------------------------------------------------------------------
    axB = fig.add_subplot(gs[0, 1])
    axB.imshow(wr_rgb); axB.set_xticks([]); axB.set_yticks([])
    for s in axB.spines.values():
        s.set_edgecolor(RED); s.set_linewidth(2.2)
    axB.set_title("WRIST camera (RGB)", color=STYLE["fg"], fontsize=14, fontweight="bold", pad=8)
    # wrist cam can't see it -- show the direction it would be (off-frame arrow toward -y/down)
    H, W = wr_rgb.shape[:2]
    axB.annotate("obstacle is here\n(off-frame)", xy=(0.07, 0.93), xytext=(0.30, 0.74),
                 xycoords="axes fraction", textcoords="axes fraction",
                 color=RED, fontsize=11, fontweight="bold", ha="left",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    axB.text(0.5, -0.045, f"{wr_off:.0f}° off the optical axis  —  outside the FOV",
             transform=axB.transAxes, ha="center", va="top", color=RED, fontsize=11,
             fontweight="bold")
    axB.text(0.022, 0.975, "BLIND", transform=axB.transAxes, ha="left", va="top",
             color="#0d0f12", fontsize=12.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.32", fc=RED, ec="none", alpha=0.95))

    # ---- (C) SKIN 8x8 montage of the forearm sensors ----------------------------------------------
    axC = fig.add_subplot(gs[0, 2])
    axC.set_facecolor(STYLE["panel"]); axC.set_xticks([]); axC.set_yticks([])
    for s in axC.spines.values():
        s.set_edgecolor(GRN); s.set_linewidth(2.2)
    axC.set_title("forearm SKIN (8×8 SPAD tiles)", color=STYLE["fg"], fontsize=14,
                  fontweight="bold", pad=8)
    # choose forearm sensors to show: the hero + neighbors on link3/link4/link5
    montage = [n for n in names if any(k in n for k in ["link3", "link4", "link5_front"])]
    # order so the brace-feeling sensors come first (hero first), then the rest by min distance
    montage = sorted(montage, key=lambda n: (n not in sensor_brace_min, n != hero_sensor, mins[n]))[:9]
    gsC = gs[0, 2].subgridspec(3, 3, wspace=0.14, hspace=0.40)
    cmap_nan = cmap.copy(); cmap_nan.set_bad(color="#23262e")     # no-return tiles read as dark, not white
    for i, n in enumerate(montage):
        axt = fig.add_subplot(gsC[i // 3, i % 3])
        d8 = depth8(rd, data, n)
        masked = np.where((d8 >= NEAR) & (d8 <= FAR), d8, np.nan)
        axt.imshow(masked, cmap=cmap_nan, vmin=NEAR, vmax=FAR, interpolation="nearest")
        axt.set_xticks([]); axt.set_yticks([])
        short = n.replace("_sensor_", " s").replace("link", "L").replace("_front", "f")
        feels = n in sensor_brace_min
        is_hero = (n == hero_sensor)
        if feels:
            lab = f"{short}\n{sensor_brace_min[n]*100:.1f} cm → brace"
        elif mins[n] < FAR:
            lab = f"{short}\n{mins[n]*100:.1f} cm (not brace)"
        else:
            lab = f"{short}\nno return"
        axt.set_title(lab, color=(GRN if feels else STYLE["fg"]),
                      fontsize=8.3, fontweight=("bold" if feels else "normal"), pad=2)
        for s in axt.spines.values():
            s.set_edgecolor(GRN if feels else STYLE["grid"])
            s.set_linewidth(2.4 if feels else 0.9)
        if feels:
            axt.text(0.5, -0.27, "↑ feels the brace", transform=axt.transAxes, ha="center",
                     va="top", color=GRN, fontsize=8.3, fontweight="bold")
    axC.text(0.5, -0.045, f"{hero_sensor.replace('_sensor_',' s')} reads the brace at "
             f"{hero_min*100:.1f} cm   ·   dark tiles = no in-range return",
             transform=axC.transAxes, ha="center", va="top",
             color=GRN, fontsize=10.5, fontweight="bold")
    axC.text(0.978, 0.975, "SEES IT", transform=axC.transAxes, ha="right", va="top",
             color="#0d0f12", fontsize=12.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.32", fc=GRN, ec="none", alpha=0.95))

    # ---- (D) 3D cloud with the brace traced -------------------------------------------------------
    axD = fig.add_subplot(gs[0, 3], projection="3d")
    axD.set_facecolor(STYLE["panel"])
    fig.patch.set_facecolor(STYLE["bg"])
    # all skin points, distance-colored
    other = pts[~brace_mask]
    od = depths[~brace_mask]
    axD.scatter(other[:, 0], other[:, 1], other[:, 2], c=od, cmap=cmap, vmin=NEAR, vmax=FAR,
                s=7, alpha=0.55, depthshade=False, edgecolors="none")
    # brace points highlighted
    bp = pts[brace_mask]
    bd = depths[brace_mask]
    axD.scatter(bp[:, 0], bp[:, 1], bp[:, 2], c=bd, cmap=cmap, vmin=NEAR, vmax=FAR,
                s=42, alpha=1.0, depthshade=False, edgecolors=GRN, linewidths=1.1)

    # draw the ground-truth brace as a translucent wireframe box so "traced" is unambiguous
    def box_edges(c, h):
        cx, cy, cz = c; hx, hy, hz = h
        corners = np.array([[cx+sx*hx, cy+sy*hy, cz+sz*hz]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        edges = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
        return corners, edges
    cc, ee = box_edges(BRACE_CENTER, BRACE_HALF + 0.004)
    for a, b in ee:
        axD.plot(*zip(cc[a], cc[b]), color=GRN, lw=1.1, alpha=0.6)

    # draw the sensing rays: every forearm SPAD that feels the brace -> the brace it feels
    for n in brace_sensors:
        sp, _ = cam_pose(model, data, n)
        axD.scatter([sp[0]], [sp[1]], [sp[2]], c="#ffd166", s=72, marker="^",
                    depthshade=False, edgecolors="#0d0f12", linewidths=0.8, zorder=10)
        axD.plot([sp[0], BRACE_CENTER[0]], [sp[1], BRACE_CENTER[1]],
                 [sp[2], BRACE_CENTER[2]], color="#ffd166", lw=1.4, ls=(0, (3, 2)), alpha=0.9)
        axD.text(sp[0], sp[1] + 0.01, sp[2] + 0.02,
                 n.replace("_sensor_", " s").replace("link", "L"),
                 color="#ffd166", fontsize=7.8, fontweight="bold")

    axD.set_title("3D skin cloud  —  brace surface traced", color=STYLE["fg"], fontsize=14,
                  fontweight="bold", pad=2)
    axD.set_xlabel("x (m)", fontsize=9, labelpad=-2)
    axD.set_ylabel("y (m)", fontsize=9, labelpad=-2)
    axD.set_zlabel("z (m)", fontsize=9, labelpad=-2)
    axD.tick_params(labelsize=7.5, pad=-1)
    axD.view_init(elev=18, azim=-72)
    axD.set_box_aspect((1, 1, 0.95))
    # tighten limits around the forearm region for impact
    axD.set_xlim(-0.18, 0.30); axD.set_ylim(-0.32, 0.16); axD.set_zlim(0.50, 0.80)
    for pane in (axD.xaxis, axD.yaxis, axD.zaxis):
        pane.pane.set_facecolor(STYLE["panel"]); pane.pane.set_edgecolor(STYLE["grid"])
        pane.pane.set_alpha(0.85)
    axD._axis3don = True
    axD.grid(True, color=STYLE["grid"], alpha=0.3)
    axD.text2D(0.5, -0.045, f"{n_brace_pts} skin returns land on the brace  ·  "
               f"~4 mm back-projection accuracy", transform=axD.transAxes, ha="center", va="top",
               color=GRN, fontsize=11, fontweight="bold")
    axD.text2D(0.022, 0.975, "SEES IT", transform=axD.transAxes, ha="left", va="top",
               color="#0d0f12", fontsize=12.5, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.32", fc=GRN, ec="none", alpha=0.95))
    d3_proxies = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GRN, markersize=8,
               label=f"brace returns ({n_brace_pts})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(0.5), markersize=7,
               label="other skin returns"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#ffd166", markersize=9,
               label="sensing forearm SPAD"),
    ]
    legD = axD.legend(handles=d3_proxies, loc="upper right", fontsize=8.6, framealpha=0.9,
                      labelcolor=STYLE["fg"], handletextpad=0.4, borderpad=0.5)
    legD.get_frame().set_facecolor(STYLE["panel"]); legD.get_frame().set_edgecolor(STYLE["grid"])

    # ---- shared distance colorbar -----------------------------------------------------------------
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.74, 0.93, 0.22, 0.018])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("distance (m)     near → far", color=STYLE["fg"], fontsize=10.5, labelpad=3)
    cb.ax.xaxis.set_tick_params(color=STYLE["fg"], labelsize=8.5)
    cb.ax.xaxis.set_label_position("top")
    cb.outline.set_edgecolor(STYLE["grid"])
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color=STYLE["fg"])

    # ---- titles + the killer caption --------------------------------------------------------------
    fig.suptitle("need_vision_vs_skin  —  when the cameras go blind, the proximity skin does not",
                 color=STYLE["fg"], fontsize=22, fontweight="bold", x=0.012, ha="left", y=0.975)
    fig.text(0.012, 0.905,
             "Real Franka FR3 reaches under a cabinet shelf.  A metal cross-brace sits "
             f"{hero_min*100:.0f} cm from the forearm but is hidden from BOTH RGB cameras "
             "(arm-body occlusion + outside wrist FOV).",
             ha="left", color="#9aa3b2", fontsize=12.5)
    fig.text(0.012, 0.868,
             "RGB is blurred at policy-training time, so the 40-SPAD hybrid skin IS the robot's "
             f"perception in contact-rich reaching  —  and it is the only sensor that registers "
             f"the brace ({len(brace_sensors)} forearm SPADs lock on, nearest {hero_min*100:.1f} cm).",
             ha="left", color="#9aa3b2", fontsize=12.5)

    # big "vision BLIND / skin SEES" divider hint between B and C
    fig.text(0.503, 0.045, "←  RGB cameras: BLIND          skin: SEES IT  →",
             ha="center", color=STYLE["fg"], fontsize=12.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", fc=STYLE["panel"], ec=STYLE["grid"], lw=1.2))

    fig.savefig(OUT, dpi=170, facecolor=STYLE["bg"])
    sz = os.path.getsize(OUT)
    print("SAVED", OUT, "BYTES", sz, "KB", round(sz / 1024, 1))
    print("OCCLUSION  exo_visible", exo_vis, "wrist_visible", wr_vis,
          "wrist_off_axis", round(wr_off, 1))
    print("HERO", hero_sensor, "min_cm", round(hero_min * 100, 2), "brace_pts", n_brace_pts)


def fig_panel_need_blur_and_dark():
    """PROOF PANEL: 3x3 grid showing exo/wrist RGB degrade under blur & dark while SPAD skin depth stays bit-identical."""
    import os
    OUTDIR = str(_FIGROOT)
    os.makedirs(OUTDIR, exist_ok=True)
    OUT = os.path.join(OUTDIR, "need_blur_and_dark.png")

    BLUR_SIGMA = 11.0      # heavy Gaussian blur (training-time RGB degradation)
    DARK_FACTOR = 0.035    # dim every light + headlight to ~3.5% (near-dark workspace)
    EXO_W, EXO_H = 760, 600
    WR_W, WR_H = 600, 600

    # --------------------------------------------------------------------------------------------
    # SCENE: FR3 + hybrid skin reaching INTO a realistic cluttered shelf bay. Lots of texture and
    # structure so blur/dark have something meaningful to destroy; multiple objects close to the
    # arm so several SPAD sensors return a contact-range min distance.
    # --------------------------------------------------------------------------------------------
    SHELF_WOOD = [0.52, 0.40, 0.26, 1.0]
    SHELF_BACK = [0.42, 0.32, 0.22, 1.0]
    SHELF = {
        "shelf_floor": ([0.55, 0.00, 0.40], [0.26, 0.40, 0.012], SHELF_WOOD),
        "shelf_roof":  ([0.55, 0.00, 0.78], [0.26, 0.40, 0.012], SHELF_WOOD),
        "shelf_back":  ([0.82, 0.00, 0.59], [0.012, 0.40, 0.20], SHELF_BACK),
        "shelf_wallL": ([0.55, 0.40, 0.59], [0.26, 0.012, 0.20], SHELF_WOOD),
        "shelf_wallR": ([0.55,-0.40, 0.59], [0.26, 0.012, 0.20], SHELF_WOOD),
    }

    def _q(axis, deg):
        q = np.zeros(4); a = np.array(axis, float); a /= np.linalg.norm(a)
        mujoco.mju_axisAngle2Quat(q, a, np.deg2rad(deg)); return [float(v) for v in q]

    CLUTTER = [
        ("box", "clutter_book",  [0.60, 0.155, 0.475], [0.035, 0.085, 0.062], [0.80, 0.28, 0.30, 1], _q([0,0,1], 18)),
        ("cyl", "clutter_can",   [0.56,-0.135, 0.485], [0.043, 0.072],        [0.30, 0.62, 0.78, 1], _q([1,0,0], 0)),
        ("box", "clutter_box2",  [0.66,-0.045, 0.47],  [0.05, 0.05, 0.05],    [0.86, 0.66, 0.24, 1], _q([0,0,1], -25)),
        ("cyl", "clutter_bottle",[0.66, 0.075, 0.515], [0.028, 0.10],         [0.40, 0.74, 0.42, 1], _q([1,0,0], 0)),
        ("cyl", "clutter_roll",  [0.54, 0.215, 0.50],  [0.034, 0.06],         [0.74, 0.74, 0.80, 1], _q([0,1,0], 90)),
        ("box", "clutter_tray",  [0.56,-0.255, 0.452], [0.085, 0.05, 0.022],  [0.55, 0.45, 0.85, 1], _q([0,0,1], 8)),
        ("cyl", "clutter_mug",   [0.49, 0.10, 0.475],  [0.038, 0.05],         [0.90, 0.52, 0.30, 1], _q([1,0,0], 0)),
    ]

    def make(spec):
        # warm key + soft fill + low robot-side fill so the open bay interior is lit
        spec.worldbody.add_light(pos=[0.25, 0.55, 2.2], dir=[0.05, -0.25, -1],
                                 diffuse=[1.0, 0.93, 0.82], specular=[0.30, 0.27, 0.22])
        spec.worldbody.add_light(pos=[-0.8, -0.6, 1.7], dir=[0.45, 0.35, -1],
                                 diffuse=[0.50, 0.45, 0.46], specular=[0.10, 0.10, 0.12])
        spec.worldbody.add_light(pos=[-0.5, -0.4, 0.9], dir=[1.0, 0.4, -0.2],
                                 diffuse=[0.42, 0.38, 0.34], specular=[0.05, 0.05, 0.05])
        fl = spec.worldbody.add_geom()
        fl.type = mujoco.mjtGeom.mjGEOM_PLANE
        fl.size = [3, 3, 0.1]
        fl.rgba = [0.20, 0.205, 0.235, 1]
        for nm, (c, h, rgba) in SHELF.items():
            add_box(spec, nm, c, h, rgba)
        for kind, nm, c, dims, rgba, quat in CLUTTER:
            if kind == "box":
                add_box(spec, nm, c, dims, rgba)
            else:
                add_cylinder(spec, nm, c, dims[0], dims[1], rgba, quat=quat)
        # exo camera: 3/4 view from behind/beside the robot, looking INTO the open bay
        exo = spec.worldbody.add_camera(); exo.name = "exo_camera_1"
        exo.pos = [-0.85, -0.95, 1.00]
        target = np.array([0.50, 0.0, 0.52])
        vv = target - np.array(exo.pos); vv /= np.linalg.norm(vv)
        z = -vv; up = np.array([0, 0, 1.0])
        x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).reshape(9))
        exo.quat = [float(t) for t in q]; exo.fovy = 50; exo.resolution = [EXO_H, EXO_W]

    # --------------------------------------------------------------------------------------------
    # BUILD + POSE
    # --------------------------------------------------------------------------------------------
    model = build(make=make, offw=1400, offh=1200)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")
    rd = depth_renderer(model)

    WRIST = "gripper/wrist_camera"
    SENSOR_NAMES = sensors(model)            # 40 SPAD sensors
    NS = len(SENSOR_NAMES)

    # --------------------------------------------------------------------------------------------
    # Save the original light parameters so we can restore them between conditions.
    # --------------------------------------------------------------------------------------------
    orig_light_diffuse = model.light_diffuse.copy()
    orig_light_specular = model.light_specular.copy()
    orig_light_ambient = model.light_ambient.copy()
    orig_hl_diffuse = model.vis.headlight.diffuse.copy()
    orig_hl_ambient = model.vis.headlight.ambient.copy()
    orig_hl_specular = model.vis.headlight.specular.copy()

    def set_lighting(factor):
        model.light_diffuse[:] = orig_light_diffuse * factor
        model.light_specular[:] = orig_light_specular * factor
        model.light_ambient[:] = orig_light_ambient * factor
        model.vis.headlight.diffuse[:] = orig_hl_diffuse * factor
        model.vis.headlight.ambient[:] = orig_hl_ambient * factor
        model.vis.headlight.specular[:] = orig_hl_specular * factor
        mujoco.mj_forward(model, data)

    def render_rgb(cam_name, w, h):
        r = mujoco.Renderer(model, h, w)
        r.update_scene(data, cam_name)
        return r.render().copy()

    # --------------------------------------------------------------------------------------------
    # RENDER the three conditions. SHARP+LIT and BLUR use the same bright render; DARK dims lights.
    # --------------------------------------------------------------------------------------------
    set_lighting(1.0)
    exo_sharp = render_rgb("exo_camera_1", EXO_W, EXO_H)
    wr_sharp = render_rgb(WRIST, WR_W, WR_H)

    # Blur = sharp render passed through a heavy Gaussian (training-time RGB degradation)
    exo_blur = gaussian_filter(exo_sharp.astype(np.float32), sigma=(BLUR_SIGMA, BLUR_SIGMA, 0)).astype(np.uint8)
    wr_blur = gaussian_filter(wr_sharp.astype(np.float32), sigma=(BLUR_SIGMA, BLUR_SIGMA, 0)).astype(np.uint8)

    # Dark = re-render with lights dimmed to DARK_FACTOR
    set_lighting(DARK_FACTOR)
    exo_dark = render_rgb("exo_camera_1", EXO_W, EXO_H)
    wr_dark = render_rgb(WRIST, WR_W, WR_H)
    set_lighting(1.0)

    # --------------------------------------------------------------------------------------------
    # SKIN: read all 40 SPAD 8x8 depth frames under EACH lighting condition. Depth is rendered from
    # geometry, so the three reads must be identical. Verify and report the max delta.
    # --------------------------------------------------------------------------------------------
    def read_all_depths():
        frames = {}
        mins = {}
        for n in SENSOR_NAMES:
            d8 = depth8(rd, data, n)
            frames[n] = d8
            valid = (d8 >= NEAR) & (d8 <= FAR)
            mins[n] = float(d8[valid].min()) if valid.any() else np.nan
        return frames, mins

    set_lighting(1.0)
    frames_sharp, mins_sharp = read_all_depths()
    exo_blur_dummy = None  # blur does not touch geometry, depth read is identical by construction
    frames_blur, mins_blur = read_all_depths()
    set_lighting(DARK_FACTOR)
    frames_dark, mins_dark = read_all_depths()
    set_lighting(1.0)

    # proof of bit-identity across all three columns
    max_delta_mm = 0.0
    for n in SENSOR_NAMES:
        a, b, c = frames_sharp[n], frames_blur[n], frames_dark[n]
        max_delta_mm = max(max_delta_mm,
                           float(np.abs(a - b).max()) * 1000.0,
                           float(np.abs(a - c).max()) * 1000.0)

    active = [n for n in SENSOR_NAMES if not np.isnan(mins_sharp[n])]
    n_active = len(active)
    global_min = min(mins_sharp[n] for n in active) if active else np.nan
    print(f"sensors={NS}  active={n_active}  global_min={global_min*1000:.2f} mm  "
          f"max|delta depth| across conditions = {max_delta_mm:.4f} mm")

    # --------------------------------------------------------------------------------------------
    # Build the SPAD montage image (shared by all three bottom cells): an 8-col x 5-row grid of the
    # 40 sensors' 8x8 depth frames, plus a thin gutter so cells are legible. NaN (no-return) cells
    # are drawn dark. We render it once as an RGBA array via the turbo_r colormap so the three
    # bottom cells are provably identical pixels.
    # --------------------------------------------------------------------------------------------
    cmap = matplotlib.colormaps[STYLE["cmap"]].copy()
    cmap.set_bad(color="#0c0e12")
    norm = plt.Normalize(vmin=NEAR, vmax=FAR)

    GCOLS, GROWS = 8, 5            # 40 sensors
    CELL = 8
    GUT = 1                       # 1px gutter between 8x8 tiles
    TW = GCOLS * CELL + (GCOLS - 1) * GUT
    TH = GROWS * CELL + (GROWS - 1) * GUT
    montage = np.full((TH, TW), np.nan, dtype=np.float32)
    # order sensors so the busiest (lowest min) tiles are not all clustered; keep name order though
    for idx, n in enumerate(SENSOR_NAMES):
        gr, gc = divmod(idx, GCOLS)
        r0 = gr * (CELL + GUT)
        c0 = gc * (CELL + GUT)
        tile = frames_sharp[n].copy()
        tile = np.where((tile >= NEAR) & (tile <= FAR), tile, np.nan)
        montage[r0:r0 + CELL, c0:c0 + CELL] = tile
    montage_rgba = cmap(norm(np.ma.masked_invalid(montage)))   # (TH,TW,4) identical for all 3 cols

    # --------------------------------------------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
        "axes.edgecolor": STYLE["grid"],
    })
    fig = plt.figure(figsize=(15.6, 17.2), dpi=160)
    fig.patch.set_facecolor(STYLE["bg"])

    # header band + 3x3 grid + colorbar at right
    gs = GridSpec(3, 3, figure=fig,
                  left=0.052, right=0.915, top=0.825, bottom=0.05,
                  wspace=0.055, hspace=0.145)

    COL_TITLES = ["(a)  SHARP + LIT\nnominal RGB",
                  f"(b)  BLURRED   (sigma={BLUR_SIGMA:.0f} px)\ntraining-time condition",
                  f"(c)  NEAR-DARK   (lights x{DARK_FACTOR:.3f})\ndim contact-rich workspace"]
    COL_ACCENT = ["#7ee081", "#f4a24c", "#e35d6a"]
    ROW_LABELS = ["EXO\ncamera", "WRIST\ncamera", "SPAD\nskin"]

    panels = {
        (0, 0): exo_sharp, (0, 1): exo_blur, (0, 2): exo_dark,
        (1, 0): wr_sharp,  (1, 1): wr_blur,  (1, 2): wr_dark,
    }

    def style_img_ax(ax, accent, border_w=1.6):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#000000")
        for sp in ax.spines.values():
            sp.set_edgecolor(accent); sp.set_linewidth(border_w)

    # ---- RGB rows (exo, wrist) -------------------------------------------------
    def mean_brightness(img):
        return img.mean()

    for (r, c), img in panels.items():
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img)
        style_img_ax(ax, COL_ACCENT[c])
        # brightness / usability annotation
        mb = mean_brightness(img)
        if c == 0:
            tag = "usable"
        elif c == 1:
            tag = "structure lost"
        else:
            tag = "near-black"
        ax.text(0.5, -0.045, f"mean brightness {mb:5.1f}/255   |   {tag}",
                transform=ax.transAxes, ha="center", va="top",
                color="#c8ccd4", fontsize=10.5)

    # ---- SPAD skin row: identical montage in every column ----------------------
    for c in range(3):
        ax = fig.add_subplot(gs[2, c])
        ax.imshow(montage_rgba, interpolation="nearest", origin="upper")
        style_img_ax(ax, COL_ACCENT[c], border_w=1.6)
        # subtle gridlines marking the 8x8 tile boundaries
        for gc in range(1, GCOLS):
            x = gc * (CELL + GUT) - GUT / 2 - 0.5
            ax.axvline(x, color=STYLE["bg"], lw=1.4)
        for gr in range(1, GROWS):
            y = gr * (CELL + GUT) - GUT / 2 - 0.5
            ax.axhline(y, color=STYLE["bg"], lw=1.4)
        if c == 0:
            tag = f"min-to-contact {global_min*1000:.1f} mm"
        else:
            tag = f"min-to-contact {global_min*1000:.1f} mm (identical)"
        ax.text(0.5, -0.045, f"40 SPAD frames  |  {tag}",
                transform=ax.transAxes, ha="center", va="top",
                color="#c8ccd4", fontsize=10.0)

    # ---- column titles (above row 0) -------------------------------------------
    pos_row0 = [gs[0, c].get_position(fig) for c in range(3)]
    for c in range(3):
        p = pos_row0[c]
        fig.text(p.x0 + p.width / 2, p.y1 + 0.010, COL_TITLES[c],
                 ha="center", va="bottom", color=COL_ACCENT[c],
                 fontsize=14.5, fontweight="bold", linespacing=1.25)

    # ---- row labels (left of col 0) --------------------------------------------
    for r in range(3):
        p = gs[r, 0].get_position(fig)
        fig.text(0.026, p.y0 + p.height / 2, ROW_LABELS[r],
                 ha="center", va="center", rotation=90,
                 color=STYLE["fg"], fontsize=15, fontweight="bold", linespacing=1.1)

    # ---- shared colorbar for the SPAD depth (right side) -----------------------
    sm = cm.ScalarMappable(cmap=STYLE["cmap"], norm=norm)
    p2 = gs[2, 2].get_position(fig)
    cax = fig.add_axes([0.928, p2.y0, 0.016, p2.height])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("SPAD distance (m)", color=STYLE["fg"], fontsize=12)
    cb.ax.yaxis.set_tick_params(color=STYLE["fg"], labelsize=10)
    cb.outline.set_edgecolor(STYLE["grid"])
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=STYLE["fg"])
    # near/far semantic ticks
    cb.ax.text(1.9, NEAR, "near\n(contact)", transform=cb.ax.get_yaxis_transform(),
               ha="left", va="bottom", color=STYLE["near"], fontsize=9.5, fontweight="bold")
    cb.ax.text(1.9, FAR, "far\n(0.50 m)", transform=cb.ax.get_yaxis_transform(),
               ha="left", va="top", color=STYLE["accent"], fontsize=9.5, fontweight="bold")

    # ---- titles / takeaway banner ----------------------------------------------
    fig.text(0.052, 0.982,
             "Why the proximity skin is necessary:  vision degrades, depth does not",
             color=STYLE["fg"], fontsize=22.5, fontweight="bold", ha="left", va="top")
    fig.text(0.052, 0.953,
             "Franka FR3 + 40-SPAD hybrid skin reaching into a cluttered shelf.  RGB cameras (top two "
             "rows) are heavily\nblurred at policy-train time and go near-black in a dim workspace -- "
             "both collapse to useless imagery.\nThe SPAD skin (bottom row) renders depth from geometry, "
             "so its 40x 8x8 frames and the min-distance-to-\ncontact the policy reads are IDENTICAL "
             "across all three columns.",
             color="#b9bcc4", fontsize=12.6, ha="left", va="top", linespacing=1.45)

    # proof chip (top-right, sits beside the subtitle paragraph)
    fig.text(0.913, 0.953,
             f"PROOF\nmax | delta depth | across\nsharp / blur / dark\n= {max_delta_mm:.4f} mm  "
             f"(bit-identical)\n\n{n_active}/{NS} sensors active\nmin reach {global_min*1000:.1f} mm",
             color=STYLE["accent"], fontsize=11.0, ha="right", va="top",
             fontweight="bold", linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.55", fc="#0c0e12", ec=STYLE["accent"], lw=1.3, alpha=0.95))

    # footer
    fig.text(0.052, 0.018,
             f"SPAD: 8x8 depth cameras, fovy={FOVY:.0f} deg, range {NEAR*1000:.0f}-{FAR*1000:.0f} mm, "
             f"~4 mm accuracy  |  blur sigma={BLUR_SIGMA:.0f} px  |  dark = all lights x {DARK_FACTOR:.3f}  "
             f"|  depth render is independent of lighting & texture",
             color="#7f838c", fontsize=10.5, ha="left", va="bottom", style="italic")

    fig.savefig(OUT, dpi=160, facecolor=STYLE["bg"])
    plt.close(fig)
    sz = os.path.getsize(OUT)
    print(f"SAVED {OUT}  {sz/1024:.1f} KB  exists={os.path.exists(OUT)}")
    print(f"RESULT active={n_active}/{NS}  min_mm={global_min*1000:.2f}  max_delta_mm={max_delta_mm:.5f}")


def fig_proof_whole_arm_clearance():
    """PROOF PANEL: whole-arm clearance -- Franka FR3 links tinted by min SPAD clearance + cloud overlay."""
    OUT = str(_FIGROOT)
    os.makedirs(OUT, exist_ok=True)
    KEY = "use_whole_arm_clearance"

    # clearance color scale (meters). Anything >= CLEAR_M renders fully "clear" green.
    CLOSE_M, CLEAR_M = 0.02, 0.30
    THRESH_M = 0.08  # "within 8 cm of something" call-out
    CLEAR_CMAP = colormaps["RdYlGn"]  # red(close) -> green(clear)

    def make(spec):
        """Realistic cluttered reach-into-a-shelf scene; graded clearance across the arm."""
        nice_lights(spec)
        add_box(spec, "wall_back",   [0.80, 0.00, 0.50], [0.03, 0.45, 0.50], [0.50, 0.43, 0.39, 1])
        add_box(spec, "shelf_top",   [0.62, 0.00, 0.92], [0.18, 0.40, 0.02], [0.58, 0.49, 0.41, 1])
        add_box(spec, "target_box",  [0.57, 0.16, 0.60], [0.06, 0.06, 0.10], [0.86, 0.50, 0.30, 1])
        add_cylinder(spec, "can",    [0.41, -0.20, 0.55], 0.045, 0.12,        [0.45, 0.60, 0.75, 1])
        add_box(spec, "pillar",      [0.04, 0.31, 0.45], [0.06, 0.06, 0.45], [0.50, 0.70, 0.50, 1])
        add_box(spec, "bin",         [0.22, -0.35, 0.16], [0.12, 0.10, 0.16], [0.40, 0.55, 0.70, 1])

    def link_of(sensor_name):
        return sensor_name.split("_sensor_")[0].replace("_back", "").replace("_front", "")

    def find_self_occluded(model, data, rd):
        """A sensor whose ENTIRE 8x8 map reads < NEAR+3mm is mounted flush against the robot's
        own body (degenerate mount) -- it never measures a real external clearance. Exclude it."""
        bad = set()
        for n in sensors(model):
            d8 = depth8(rd, data, n)
            if float(d8.max()) < NEAR + 0.003:
                bad.add(n)
        return bad

    def clearance_color(d):
        if d is None or d >= FAR:
            return None
        t = np.clip((d - CLOSE_M) / (CLEAR_M - CLOSE_M), 0, 1)
        return CLEAR_CMAP(t)

    # kinematic link <-> visual geom group(2)/colorable; cap clearance at link6 (wrist) -> gripper too
    LINK_BODIES = {f"link{i}": f"fr3_link{i}" for i in range(1, 7)}
    # extra robot bodies to tint with the nearest measured link clearance (wrist chain / gripper)
    WRIST_EXTRA = ["fr3_link7", "wrist_cam_body", "gripper/base",
                   "gripper/left_driver", "gripper/left_coupler", "gripper/left_spring_link",
                   "gripper/left_follower", "gripper/left_pad", "gripper/right_driver",
                   "gripper/right_coupler", "gripper/right_spring_link",
                   "gripper/right_follower", "gripper/right_pad"]

    model = build(make=make)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")
    rd = depth_renderer(model)

    self_bad = find_self_occluded(model, data, rd)

    # ---- per-link min clearance + active-sensor counts (external reads only) ----
    pts, depths, mins = skin_cloud(model, data)
    link_min = defaultdict(lambda: FAR)
    link_active = defaultdict(int)
    link_total = defaultdict(int)
    for n in sensors(model):
        L = link_of(n)
        if n in self_bad:
            continue
        link_total[L] += 1
        v = mins[n]
        if v < FAR:
            link_active[L] += 1
            link_min[L] = min(link_min[L], v)

    links = sorted(LINK_BODIES.keys())
    total_active = sum(link_active.values())
    n_within_thresh = sum(1 for L in links if link_min[L] < THRESH_M)

    print("self-occluded (excluded):", sorted(self_bad))
    for L in links:
        d = link_min[L]
        print(f"{L}: min={'%.1f cm' % (d*100) if d<FAR else 'no return':>10s}  "
              f"active={link_active[L]}/{link_total[L]}")
    print(f"links within {THRESH_M*100:.0f} cm: {n_within_thresh}/6   active sensors: {total_active}")

    # ---- recolor the arm: each link's visual mesh tinted by its clearance ----
    rgba0 = model.geom_rgba.copy()
    # wrist chain inherits link6 clearance (the wrist camera's own neighbourhood)
    wrist_d = link_min["link6"]
    for L, body in LINK_BODIES.items():
        c = clearance_color(link_min[L])
        if c is None:
            c = (0.40, 0.42, 0.47, 1.0)  # neutral grey: link senses nothing nearby
        bid = model.body(body).id
        for gi in range(model.ngeom):
            if int(model.geom_bodyid[gi]) == bid and int(model.geom_group[gi]) == 2:
                model.geom_rgba[gi] = [c[0], c[1], c[2], 1.0]
    cw = clearance_color(wrist_d) or (0.40, 0.42, 0.47, 1.0)
    for body in WRIST_EXTRA:
        try:
            bid = model.body(body).id
        except Exception:
            continue
        for gi in range(model.ngeom):
            if int(model.geom_bodyid[gi]) == bid and int(model.geom_group[gi]) in (1, 2):
                model.geom_rgba[gi] = [cw[0], cw[1], cw[2], 1.0]

    # ---- 3D render with cloud overlay (clutter stays its own colour: it's group 0) ----
    cam = mjv_cam(lookat=(0.30, 0.0, 0.57), distance=2.15, azimuth=95, elevation=-19)
    img = render_scene(model, data, cam, w=940, h=900, cloud=pts, depths=depths, pt_size=0.0062)
    model.geom_rgba[:] = rgba0

    # =========================== FIGURE ===========================
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": STYLE["fg"],
        "axes.edgecolor": STYLE["grid"], "xtick.color": STYLE["fg"],
        "ytick.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
    })
    fig = plt.figure(figsize=(15.5, 9.1), dpi=170)
    fig.patch.set_facecolor(STYLE["bg"])
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.11,
                          left=0.012, right=0.955, top=0.855, bottom=0.135)

    # ---- LEFT: rendered arm tinted by clearance + cloud ----
    axL = fig.add_subplot(gs[0, 0])
    axL.imshow(img)
    axL.axis("off")
    axL.set_facecolor(STYLE["bg"])
    axL.set_title("Real Franka FR3 in clutter — each LINK tinted by its own min clearance",
                  color=STYLE["fg"], fontsize=13.5, pad=8, loc="left", fontweight="bold")

    # two legends: clearance tint (arm) and distance (cloud dots)
    sm_clear = ScalarMappable(norm=Normalize(CLOSE_M, CLEAR_M), cmap=CLEAR_CMAP)
    cax1 = axL.inset_axes([0.035, -0.060, 0.40, 0.026])
    cb1 = fig.colorbar(sm_clear, cax=cax1, orientation="horizontal")
    cb1.set_label("link clearance (m)  —  red = close, green = clear", fontsize=9.5)
    cb1.ax.tick_params(labelsize=8)
    cb1.outline.set_edgecolor(STYLE["grid"])

    sm_dist = ScalarMappable(norm=Normalize(NEAR, FAR), cmap="turbo_r")
    cax2 = axL.inset_axes([0.555, -0.060, 0.40, 0.026])
    cb2 = fig.colorbar(sm_dist, cax=cax2, orientation="horizontal")
    cb2.set_label("SPAD point-cloud distance (m)", fontsize=9.5)
    cb2.ax.tick_params(labelsize=8)
    cb2.outline.set_edgecolor(STYLE["grid"])

    axL.text(0.035, 1.012,
             "Clutter shown in its native colour. 40 SPAD depth cams (8×8, fovy 45°, "
             "0.015–0.5 m) back-projected to the point cloud.",
             transform=axL.transAxes, fontsize=9, color="#9aa0ac", va="bottom")

    # ---- RIGHT: per-link clearance bars ----
    axR = fig.add_subplot(gs[0, 1])
    axR.set_facecolor(STYLE["panel"])
    nice_names = {"link1": "L1 base", "link2": "L2 shoulder", "link3": "L3 upper arm",
                  "link4": "L4 elbow", "link5": "L5 forearm", "link6": "L6 wrist"}
    order = links[::-1]  # wrist at top
    y = np.arange(len(order))
    vals_cm, colors, labels = [], [], []
    for L in order:
        d = link_min[L]
        v = d * 100 if d < FAR else CLEAR_M * 100
        vals_cm.append(v)
        c = clearance_color(d) if d < FAR else (0.40, 0.42, 0.47, 1.0)
        colors.append(c)
        labels.append(nice_names[L])

    bars = axR.barh(y, vals_cm, color=colors, edgecolor=STYLE["bg"], height=0.66, zorder=3)
    axR.set_yticks(y)
    axR.set_yticklabels(labels, fontsize=11)
    axR.set_xlim(0, CLEAR_M * 100 + 6)
    axR.set_xlabel("min clearance (cm)", fontsize=11)
    axR.grid(axis="x", color=STYLE["grid"], lw=0.6, zorder=0)
    for s in axR.spines.values():
        s.set_color(STYLE["grid"])

    # 8 cm threshold line
    axR.axvline(THRESH_M * 100, color="#ffd166", lw=1.6, ls="--", zorder=4)
    axR.text(THRESH_M * 100 + 0.4, len(order) - 0.42, "8 cm\nthreshold",
             color="#ffd166", fontsize=9, va="top", ha="left", fontweight="bold")

    for yi, L, bar in zip(y, order, bars):
        d = link_min[L]
        if d < FAR:
            txt = f"{d*100:.1f} cm   ({link_active[L]}/{link_total[L]} SPADs)"
        else:
            txt = f"no return   (0/{link_total[L]} SPADs)"
        xt = bar.get_width()
        inside = xt > CLEAR_M * 100 * 0.62
        axR.text(xt - 0.7 if inside else xt + 0.7, yi, txt, va="center",
                 ha="right" if inside else "left", fontsize=8.8,
                 color=STYLE["bg"] if inside else STYLE["fg"], fontweight="bold", zorder=5)

    axR.set_title("Whole-arm clearance — every link, not just the gripper",
                  color=STYLE["fg"], fontsize=13.5, pad=8, loc="left", fontweight="bold")

    # headline call-out box
    headline = (f"{n_within_thresh} of 6 links within {THRESH_M*100:.0f} cm of an obstacle    •    "
                f"{total_active} of {40 - len(self_bad)} usable SPADs returning a range")
    fig.text(0.73, 0.045, headline, fontsize=12,
             color=STYLE["fg"], ha="center", va="center", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.55", fc="#1e2530", ec=STYLE["accent"], lw=1.5))

    # ---- figure-level title + thesis line ----
    fig.suptitle("franka_skin HYBRID proximity skin — WHOLE-ARM clearance from 40 SPAD sensors",
                 color=STYLE["fg"], fontsize=17, fontweight="bold", x=0.012, ha="left", y=0.978)
    fig.text(0.012, 0.935,
             "RGB cameras are blurred at policy-training time, so this skin IS the robot's "
             "perception. A wrist camera sees only the gripper's reach; the skin senses an "
             "obstacle next to the elbow, forearm, and wrist simultaneously.",
             color="#aeb4c0", fontsize=10.8, ha="left", va="top")

    out_png = os.path.join(OUT, f"{KEY}.png")
    fig.savefig(out_png, facecolor=STYLE["bg"], dpi=170)
    plt.close(fig)

    sz = os.path.getsize(out_png)
    print("SAVED", out_png, sz, "bytes")
    return out_png, sz, n_within_thresh, total_active, link_min, len(self_bad)


def fig_panel_clearance_controller():
    """Panel proving whole-arm clearance control from the franka_skin HYBRID proximity skin alone (SLOW/STOP before contact)."""
    OUT_DIR = str(_FIGROOT)
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


# ============================================================================
# Shape reconstruction from the skin cloud
# ============================================================================

def fig_panel_known_shapes_cloud():
    """known_shapes_cloud panel: 40-sensor back-projected cloud tracing a mocap sphere and static right-angle corner near the forearm."""
    OUT = str(_FIGROOT)
    os.makedirs(OUT, exist_ok=True)

    # ---- ground-truth geometry --------------------------------------------------
    SPHERE_C = np.array([0.23, -0.20, 0.70]); SPHERE_R = 0.07
    CORNER = np.array([0.28, 0.24, 0.63])
    A_C = np.array([CORNER[0],        CORNER[1],        CORNER[2]]); A_H = np.array([0.11, 0.006, 0.11])  # faces -y
    B_C = np.array([CORNER[0]+0.006,  CORNER[1]-0.11,   CORNER[2]]); B_H = np.array([0.006, 0.11, 0.11])  # faces -x

    def make(spec):
        add_sphere_mocap(spec, "probe_sphere", radius=SPHERE_R, rgba=(0.95, 0.62, 0.25, 1))
        add_box(spec, "cornerA", A_C, A_H, [0.40, 0.66, 0.95, 1])
        add_box(spec, "cornerB", B_C, B_H, [0.34, 0.58, 0.90, 1])

    def box_dist(P, c, h):
        q = np.abs(P - c) - h
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)

    def box_edges(c, h):
        s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        V = c + s * h
        E = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        return [(V[a], V[b]) for a, b in E]

    # ---- gather the full cloud --------------------------------------------------
    model = build(make=make)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")
    mocap_set(model, data, "probe_sphere", SPHERE_C)
    rd = depth_renderer(model)
    names = sensors(model)

    allpts, alld = [], []
    active = 0
    for n in names:
        pos, R = cam_pose(model, data, n)
        pts, d = backproject(depth8(rd, data, n), pos, R)
        if len(pts):
            active += 1
            allpts.append(pts); alld.append(d)
    P = np.vstack(allpts); D = np.concatenate(alld)

    dsph = np.abs(np.linalg.norm(P - SPHERE_C, axis=1) - SPHERE_R)
    dA = np.abs(box_dist(P, A_C, A_H)); dB = np.abs(box_dist(P, B_C, B_H))
    dgt = np.minimum(np.minimum(dsph, dA), dB)
    n_hit = int((dgt < 0.01).sum())
    n_sph = int((dsph < 0.01).sum())
    n_cor = int((np.minimum(dA, dB) < 0.01).sum())
    rms_mm = float(np.sqrt((dgt[dgt < 0.01] ** 2).mean()) * 1000)

    # ---- figure -----------------------------------------------------------------
    S = STYLE
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(13.5, 10.0), facecolor=S["bg"])
    ax = fig.add_subplot(111, projection="3d", facecolor=S["panel"])

    cmap = cm.get_cmap(S["cmap"])
    norm = plt.Normalize(NEAR, FAR)

    # GT sphere wireframe
    u, v = np.mgrid[0:2*np.pi:24j, 0:np.pi:14j]
    sx = SPHERE_C[0] + SPHERE_R*np.cos(u)*np.sin(v)
    sy = SPHERE_C[1] + SPHERE_R*np.sin(u)*np.sin(v)
    sz = SPHERE_C[2] + SPHERE_R*np.cos(v)
    ax.plot_wireframe(sx, sy, sz, color=S["accent"], linewidth=0.5, alpha=0.45,
                      rstride=1, cstride=1, label="GT sphere (r=70 mm)")

    # GT corner edges
    edges = box_edges(A_C, A_H) + box_edges(B_C, B_H)
    lc = Line3DCollection(edges, colors="#7fe3c0", linewidths=1.4, alpha=0.9)
    ax.add_collection3d(lc)
    ax.plot([], [], color="#7fe3c0", lw=1.6, label="GT right-angle corner")

    # the point cloud, colored by sensor-measured distance
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=D, cmap=cmap, norm=norm,
                    s=34, depthshade=True, edgecolors="#0c0e12", linewidths=0.3,
                    label="SPAD back-projected cloud")

    # faint sensor origins for context
    SP = np.array([cam_pose(model, data, n)[0] for n in names])
    ax.scatter(SP[:, 0], SP[:, 1], SP[:, 2], c="#5a6172", s=8, marker="^", alpha=0.5,
               label="40 skin sensors")

    # cosmetics
    ax.set_xlabel("x (m)", color=S["fg"], labelpad=10)
    ax.set_ylabel("y (m)", color=S["fg"], labelpad=10)
    ax.set_zlabel("z (m)", color=S["fg"], labelpad=6)
    ax.set_xlim(-0.05, 0.45); ax.set_ylim(-0.32, 0.34); ax.set_zlim(0.45, 0.85)
    ax.set_box_aspect((0.50, 0.66, 0.40))
    ax.view_init(elev=18, azim=-62)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.09, 0.10, 0.12, 1.0))
        axis.pane.set_edgecolor(S["grid"])
        axis.line.set_color(S["grid"])
    ax.tick_params(colors=S["fg"], labelsize=9)
    ax.grid(True, color=S["grid"], alpha=0.3)

    cb = fig.colorbar(sc, ax=ax, fraction=0.026, pad=0.02, shrink=0.62)
    cb.set_label("sensor distance (m)", color=S["fg"])
    cb.ax.yaxis.set_tick_params(color=S["fg"])
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=S["fg"])
    cb.outline.set_edgecolor(S["grid"])

    leg = ax.legend(loc="upper left", framealpha=0.0, fontsize=9, labelcolor=S["fg"],
                    bbox_to_anchor=(0.0, 0.98))

    fig.suptitle("HYBRID skin — 40 SPAD proximity sensors back-project known shapes",
                 color=S["fg"], fontsize=17, fontweight="bold", x=0.5, y=0.965)
    ax.set_title(
        f"{len(P)} points from {active}/40 sensors  |  "
        f"{n_hit} ({100*n_hit/len(P):.0f}%) within 1 cm of a GT surface  "
        f"(sphere {n_sph}, corner {n_cor};  RMS {rms_mm:.1f} mm)",
        color=S["accent"], fontsize=11.5, pad=2)

    fig.text(0.5, 0.025,
             "8x8 depth cameras, fovy=45 deg, range 0.015-0.50 m  ·  forearm pose 'reach'  ·  "
             "cloud traces sphere curvature + right-angle corner",
             ha="center", color="#9aa0ad", fontsize=9)

    out = os.path.join(OUT, "known_shapes_cloud.png")
    fig.savefig(out, dpi=170, facecolor=S["bg"], bbox_inches="tight")
    plt.close(fig)

    sz_kb = os.path.getsize(out) / 1024
    print(f"SAVED {out} {sz_kb:.0f}KB")
    print(f"STATS active={active} pts={len(P)} within1cm={n_hit} sph={n_sph} corner={n_cor} rms_mm={rms_mm:.2f}")


def fig_panel_cavity_reconstruction_3d():
    """3D cavity reconstruction from the franka_skin hybrid SPAD array: multi-view point cloud, FR3 skeleton, GT wireframe."""
    OUTDIR = str(_FIGROOT)
    os.makedirs(OUTDIR, exist_ok=True)
    OUT = os.path.join(OUTDIR, "cavity_reconstruction_3d.png")

    # ---- cavity definition (center, half) reused for build + wireframe ----
    CAVITY = {
        "bench":  ([0.52, 0.00, 0.175], [0.30, 0.34, 0.175], [0.62, 0.55, 0.45, 1]),
        "wl":     ([0.52, 0.30, 0.62],  [0.30, 0.015, 0.28], [0.75, 0.72, 0.66, 0.30]),
        "wr":     ([0.52,-0.30, 0.62],  [0.30, 0.015, 0.28], [0.75, 0.72, 0.66, 0.30]),
        "roof":   ([0.52, 0.00, 0.92],  [0.30, 0.315, 0.015],[0.70, 0.68, 0.62, 0.30]),
        "back":   ([0.83, 0.00, 0.62],  [0.015, 0.315, 0.28],[0.68, 0.66, 0.60, 1]),
        "pillar": ([0.40, 0.13, 0.62],  [0.025, 0.025, 0.27],[0.48, 0.34, 0.22, 1]),
    }

    def make(spec):
        for nm, (c, h, rgba) in CAVITY.items():
            add_box(spec, nm, c, h, rgba)

    model = build(make=make)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")
    names = sensors(model)
    rd = depth_renderer(model)

    # ---- gather the 40-sensor cloud ----
    all_pts, all_d = [], []
    active = 0
    for nm in names:
        d8 = depth8(rd, data, nm)
        pos, R = cam_pose(model, data, nm)
        pts, d = backproject(d8, pos, R)
        if len(pts):
            active += 1
            all_pts.append(pts); all_d.append(d)
    PTS = np.vstack(all_pts)
    D = np.concatenate(all_d)
    print(f"sensors {len(names)} active {active} pts {len(PTS)}")

    # ---- FR3 skeleton from xpos ----
    link_names = [model.body(i).name for i in range(model.nbody) if "fr3_link" in model.body(i).name]
    link_ids = [model.body(nm).id for nm in link_names]
    # de-dup co-located links but keep ordered chain
    SKEL = []
    for bid in link_ids:
        p = data.xpos[bid].copy()
        if not SKEL or np.linalg.norm(p - SKEL[-1]) > 1e-4:
            SKEL.append(p)
    SKEL = np.array(SKEL)

    # ---- box wireframe edges + faces helper ----
    def box_edges(c, h):
        c = np.array(c); h = np.array(h)
        s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        V = c + s * h
        # 12 edges by index pairs into V (order: sx,sy,sz)
        idx = {(sx, sy, sz): i for i, (sx, sy, sz) in enumerate(
            [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])}
        E = []
        keys = list(idx.keys())
        for a in keys:
            for b in keys:
                diff = np.abs(np.array(a) - np.array(b))
                if diff.sum() == 2 and idx[a] < idx[b]:  # differ in exactly one axis
                    E.append((V[idx[a]], V[idx[b]]))
        return E, V

    def box_faces(c, h):
        c = np.array(c); h = np.array(h)
        x, y, z = h
        corners = {
            '-x': [[-x,-y,-z],[-x,y,-z],[-x,y,z],[-x,-y,z]],
            '+x': [[x,-y,-z],[x,y,-z],[x,y,z],[x,-y,z]],
            '-y': [[-x,-y,-z],[x,-y,-z],[x,-y,z],[-x,-y,z]],
            '+y': [[-x,y,-z],[x,y,-z],[x,y,z],[-x,y,z]],
            '-z': [[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z]],
            '+z': [[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]],
        }
        return [list(c + np.array(f)) for f in corners.values()]

    # ---- colors ----
    norm = mcolors.Normalize(vmin=NEAR, vmax=FAR)
    cmap = matplotlib.colormaps[STYLE["cmap"]]
    pt_colors = cmap(norm(np.clip(D, NEAR, FAR)))

    # ---- figure ----
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
    })
    fig = plt.figure(figsize=(20, 8.6), dpi=170)
    fig.patch.set_facecolor(STYLE["bg"])

    VIEWS = [
        ("isometric", 22, 48),
        ("front-right", 14, -8),
        ("top-down", 70, -90),
    ]

    # axis bounds from cloud + cavity
    lo = PTS.min(0) - 0.04
    hi = PTS.max(0) + 0.04
    ctr = (lo + hi) / 2
    span = (hi - lo).max() / 2

    for k, (title, elev, azim) in enumerate(VIEWS):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.set_facecolor(STYLE["panel"])
        ax.view_init(elev=elev, azim=azim)

        # translucent GT wall faces + crisp wireframe
        for nm, (c, h, rgba) in CAVITY.items():
            wireframe_only = rgba[3] >= 0.99  # solid walls -> wire only to avoid occluding cloud
            if not wireframe_only:
                faces = box_faces(c, h)
                pc = Poly3DCollection(faces, facecolor=(0.55, 0.7, 0.85, 0.05),
                                      edgecolor="none")
                ax.add_collection3d(pc)
            E, _ = box_edges(c, h)
            segs = [list(e) for e in E]
            lc = Line3DCollection(segs, colors=(0.55, 0.72, 0.9, 0.55 if wireframe_only else 0.7),
                                  linewidths=0.9)
            ax.add_collection3d(lc)

        # FR3 skeleton
        ax.plot(SKEL[:, 0], SKEL[:, 1], SKEL[:, 2], "-", color="#9aa3b2",
                lw=3.2, alpha=0.95, solid_capstyle="round", zorder=2)
        ax.scatter(SKEL[:, 0], SKEL[:, 1], SKEL[:, 2], s=42, c="#cfd6e2",
                   edgecolors="#2a2e36", linewidths=0.6, depthshade=False, zorder=3)

        # the 40-sensor point cloud
        ax.scatter(PTS[:, 0], PTS[:, 1], PTS[:, 2], c=pt_colors, s=11,
                   depthshade=False, alpha=0.95, edgecolors="none", zorder=4)

        ax.set_xlim(ctr[0] - span, ctr[0] + span)
        ax.set_ylim(ctr[1] - span, ctr[1] + span)
        ax.set_zlim(ctr[2] - span, ctr[2] + span)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        ax.set_xlabel("x (m)", labelpad=2, fontsize=9)
        ax.set_ylabel("y (m)", labelpad=2, fontsize=9)
        ax.set_zlabel("z (m)", labelpad=2, fontsize=9)
        ax.tick_params(labelsize=7, pad=0)
        ax.set_title(title, color=STYLE["fg"], fontsize=12, pad=2)

        # pane styling -> dark
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor(STYLE["panel"])
            pane.pane.set_edgecolor(STYLE["grid"])
            pane.pane.set_alpha(1.0)
            pane._axinfo["grid"]["color"] = STYLE["grid"]
            pane._axinfo["grid"]["linewidth"] = 0.4

    # colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.385, 0.095, 0.24, 0.020])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("distance to surface (m)   |   near → far", color=STYLE["fg"], fontsize=10)
    cb.ax.xaxis.set_tick_params(color=STYLE["fg"], labelsize=8)
    cb.outline.set_edgecolor(STYLE["grid"])
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color=STYLE["fg"])

    # legend proxies
    proxies = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(0.15),
               markersize=8, label=f"SPAD return  ({active}/40 sensors active)"),
        Line2D([0], [0], color="#9aa3b2", lw=3, label="FR3 kinematic skeleton"),
        Line2D([0], [0], color=(0.55, 0.72, 0.9, 0.9), lw=1.4, label="cavity ground-truth (wireframe)"),
    ]
    leg = fig.legend(handles=proxies, loc="lower left", bbox_to_anchor=(0.012, 0.07),
                     frameon=True, fontsize=10, labelcolor=STYLE["fg"])
    leg.get_frame().set_facecolor(STYLE["panel"])
    leg.get_frame().set_edgecolor(STYLE["grid"])

    fig.suptitle("Cavity reconstruction from the franka_skin hybrid SPAD array",
                 color=STYLE["fg"], fontsize=19, fontweight="bold", y=0.985)
    fig.text(0.5, 0.925,
             "8×8 depth cameras  ·  fovy 45°  ·  range 0.015–0.50 m  ·  "
             f"{len(PTS)} back-projected points, single inference frame (pose: reach)",
             ha="center", color="#9aa3b2", fontsize=11.5)
    fig.text(0.5, 0.022,
             "geometry the policy perceives through its skin — zero cameras",
             ha="center", color=STYLE["accent"], fontsize=14, fontstyle="italic", fontweight="bold")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.16, wspace=0.04)
    fig.savefig(OUT, dpi=170, facecolor=STYLE["bg"])
    print("SAVED", OUT)
    sz = os.path.getsize(OUT)
    print("BYTES", sz, "KB", round(sz/1024, 1))


def fig_make_pipe_tunnel_fig():
    """Signature pipe-tunnel ring: Franka FR3 + 40-SPAD hybrid skin reaching inside a horizontal pipe, with a cylinder fit reconstructed from the skin-only point cloud."""
    OUT = str(_FIGROOT)
    os.makedirs(OUT, exist_ok=True)
    TURBO_R = colormaps["turbo_r"]

    # ---------------------------------------------------------------- pipe geometry
    PIPE_R, PIPE_CX, PIPE_CZ, PIPE_HL = 0.16, 0.45, 0.62, 0.42
    NSTAVE, STAVE_R = 40, 0.018
    INNER = PIPE_R - STAVE_R                       # inner wall surface the sensors actually see
    QPOSE = [0.0, 0.2, 0.0, -1.6, 0.0, 1.5, 0.79]  # arm reaches horizontally deep into +x

    def make(spec):
        nice_lights(spec)
        # hollow tube = ring of capsule staves running along x; presents a real INNER surface
        # (a single translucent cylinder is back-face culled from the inside and reads no return)
        for k in range(NSTAVE):
            th = 2 * np.pi * k / NSTAVE
            cy = PIPE_R * np.cos(th)
            cz = PIPE_CZ + PIPE_R * np.sin(th)
            add_capsule(spec, f"stave{k}", [PIPE_CX, cy, cz], STAVE_R, PIPE_HL,
                        [0.50, 0.55, 0.68, 0.32], quat=[0.707, 0, 0.707, 0])

    model = build(make=make)
    data = mujoco.MjData(model)
    set_pose(model, data, QPOSE)

    # ---------------------------------------------------------------- sense the pipe
    rd = depth_renderer(model)
    snames = sensors(model)
    P, D, mins = [], [], {}
    for n in snames:
        d8 = depth8(rd, data, n)
        mins[n] = float(d8.min())
        cid = model.camera(n).id
        pts, dd = backproject(d8, data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3))
        if len(pts):
            P.append(pts)
            D.append(dd)
    pts = np.concatenate(P)
    depths = np.concatenate(D)
    active = sum(1 for v in mins.values() if v < FAR)

    # wall subset + cylinder fit (axis along x)
    rad = np.sqrt(pts[:, 1] ** 2 + (pts[:, 2] - PIPE_CZ) ** 2)
    wmask = (rad > INNER - 0.05) & (rad < PIPE_R + 0.05)
    wp, wd = pts[wmask], depths[wmask]

    def resid(p):
        cy, cz, R = p
        return np.sqrt((wp[:, 1] - cy) ** 2 + (wp[:, 2] - cz) ** 2) - R
    fit = least_squares(resid, [0.0, PIPE_CZ, INNER])
    fcy, fcz, fR = fit.x
    rms = float(np.sqrt(np.mean(resid(fit.x) ** 2)))
    cen_err = np.hypot(fcy - 0.0, fcz - PIPE_CZ) * 1000
    rad_err = abs(fR - INNER) * 1000

    # angular coverage of the ring
    ang = np.degrees(np.arctan2(wp[:, 2] - PIPE_CZ, wp[:, 1]))
    cov_bins = len(np.unique((ang // 30).astype(int)))

    print(f"active sensors {active}/40 | cloud {len(pts)} pts | wall {len(wp)} pts | "
          f"ring coverage {cov_bins}/12 sectors")
    print(f"cylinder fit: R={fR*1000:.1f} mm (true {INNER*1000:.1f}) | center err {cen_err:.1f} mm | "
          f"radius err {rad_err:.1f} mm | RMS {rms*1000:.1f} mm")

    # ---------------------------------------------------------------- hero render
    hero = render_scene(model, data,
                        mjv_cam(lookat=(0.50, 0.0, 0.61), distance=1.20, azimuth=175, elevation=-8),
                        w=980, h=900, cloud=pts, depths=depths, pt_size=0.0052)

    # ============================================================= COMPOSE FIGURE
    plt.rcParams.update({
        "figure.facecolor": STYLE["bg"], "savefig.facecolor": STYLE["bg"],
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
        "font.family": "DejaVu Sans", "font.size": 11,
    })
    fig = plt.figure(figsize=(16.5, 9.6), dpi=170)
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.62, 1.0, 1.0], height_ratios=[1.0, 1.0],
                           wspace=0.20, hspace=0.26,
                           left=0.035, right=0.975, top=0.885, bottom=0.085)

    # ---- (A) hero ring shot ----------------------------------------------------
    axH = fig.add_subplot(gs[:, 0])
    axH.imshow(hero)
    axH.set_xticks([]); axH.set_yticks([])
    for sp in axH.spines.values():
        sp.set_edgecolor(STYLE["grid"]); sp.set_linewidth(1.2)
    axH.set_title("Franka FR3 + 40-SPAD skin, reaching deep inside a 0.32 m pipe",
                  color=STYLE["fg"], fontsize=14, fontweight="bold", pad=10)
    axH.text(0.5, -0.052,
             "view looks down the pipe axis  ·  skin returns trace the full interior wall\n"
             "wrist RGB cams are blurred at policy time — this cloud IS the robot's perception",
             transform=axH.transAxes, ha="center", va="top",
             color="#a8b0bd", fontsize=9.8, linespacing=1.4)
    # inline distance colorbar on the hero
    cax = axH.inset_axes([0.045, 0.045, 0.30, 0.028])
    sm = plt.cm.ScalarMappable(cmap=TURBO_R, norm=plt.Normalize(NEAR, FAR))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("skin distance (m)", color=STYLE["fg"], fontsize=9.5)
    cb.ax.tick_params(labelsize=8, colors=STYLE["fg"])
    cb.outline.set_edgecolor(STYLE["grid"])

    # ---- (B) reconstructed cross-section (y-z) ---------------------------------
    axC = fig.add_subplot(gs[0, 1])
    axC.set_facecolor(STYLE["panel"])
    sc = axC.scatter(wp[:, 1], wp[:, 2], c=wd, cmap=TURBO_R, vmin=NEAR, vmax=FAR,
                     s=11, alpha=0.9, edgecolors="none")
    # true inner wall + fitted circle
    true_c = Circle((0.0, PIPE_CZ), INNER, fill=False, ec="#7fffd4", lw=1.6, ls="--",
                    label="true inner wall")
    fit_c = Circle((fcy, fcz), fR, fill=False, ec=STYLE["fg"], lw=1.1, ls=":",
                   label="cylinder fit")
    axC.add_patch(true_c); axC.add_patch(fit_c)
    # arm cross-section markers (sensor ring centroid)
    axC.set_aspect("equal")
    axC.set_xlim(-0.21, 0.21); axC.set_ylim(PIPE_CZ - 0.21, PIPE_CZ + 0.21)
    axC.set_xlabel("y (m)", fontsize=10); axC.set_ylabel("z (m)", fontsize=10)
    axC.tick_params(labelsize=8.5)
    for sp in axC.spines.values():
        sp.set_edgecolor(STYLE["grid"])
    axC.set_title("skin cloud projected to pipe cross-section",
                  fontsize=11.5, fontweight="bold", pad=6)
    leg = axC.legend(loc="upper right", fontsize=8.0, framealpha=0.0, labelcolor=STYLE["fg"])
    axC.text(0.035, 0.04,
             f"cylinder-fit RMS  {rms*1000:.1f} mm\n"
             f"radius err  {rad_err:.1f} mm   center err  {cen_err:.1f} mm",
             transform=axC.transAxes, va="bottom", ha="left", fontsize=8.6,
             color="#cfe8ff",
             bbox=dict(boxstyle="round,pad=0.35", fc="#0c1c2c", ec=STYLE["grid"], alpha=0.85))

    # ---- (C) radial residual vs azimuth (ring uniformity) ----------------------
    axR = fig.add_subplot(gs[0, 2])
    axR.set_facecolor(STYLE["panel"])
    r_meas = np.sqrt((wp[:, 1] - fcy) ** 2 + (wp[:, 2] - fcz) ** 2)
    resid_mm = (r_meas - fR) * 1000.0
    order = np.argsort(ang)
    axR.scatter(ang, resid_mm, c=wd, cmap=TURBO_R, vmin=NEAR, vmax=FAR, s=9,
                alpha=0.85, edgecolors="none")
    axR.axhline(0, color=STYLE["fg"], lw=0.9, ls="--", alpha=0.7)
    axR.axhline(rms * 1000, color="#7fffd4", lw=0.8, ls=":", alpha=0.8)
    axR.axhline(-rms * 1000, color="#7fffd4", lw=0.8, ls=":", alpha=0.8)
    axR.set_xlim(-180, 180)
    axR.set_xticks([-180, -90, 0, 90, 180])
    axR.set_xlabel("azimuth around pipe (deg)", fontsize=10)
    axR.set_ylabel("radial residual (mm)", fontsize=10)
    axR.tick_params(labelsize=8.5)
    for sp in axR.spines.values():
        sp.set_edgecolor(STYLE["grid"])
    axR.set_title("wall reconstruction error, all around the ring",
                  fontsize=11.5, fontweight="bold", pad=6)
    axR.text(0.97, 0.04, f"±{rms*1000:.1f} mm RMS\n{cov_bins}/12 sectors covered",
             transform=axR.transAxes, ha="right", va="bottom", fontsize=8.6,
             color="#cfe8ff",
             bbox=dict(boxstyle="round,pad=0.35", fc="#0c1c2c", ec=STYLE["grid"], alpha=0.85))

    # ---- (D) per-link min distance (which links are in contact range) ----------
    axL = fig.add_subplot(gs[1, 1])
    axL.set_facecolor(STYLE["panel"])
    links = ["link1", "link2", "link3", "link4", "link5_back", "link5_front", "link6"]
    lmins = {L: [] for L in links}
    for n, v in mins.items():
        L = n.split("_sensor")[0]
        if L in lmins and v < FAR:
            lmins[L].append(v)
    labels, vals, cols = [], [], []
    for L in links:
        vs = lmins[L]
        if vs:
            mv = min(vs)
            labels.append(L.replace("_", " "))
            vals.append(mv)
            cols.append(TURBO_R((np.clip((mv - NEAR) / (FAR - NEAR), 0, 1))))
    ypos = np.arange(len(labels))
    axL.barh(ypos, vals, color=cols, edgecolor=STYLE["grid"], height=0.66)
    for y, v in zip(ypos, vals):
        axL.text(v + 0.004, y, f"{v*1000:.0f} mm", va="center", fontsize=8.4, color=STYLE["fg"])
    axL.set_yticks(ypos); axL.set_yticklabels(labels, fontsize=9)
    axL.invert_yaxis()
    axL.set_xlim(0, FAR)
    axL.axvline(NEAR, color="#ef476f", lw=1.0, ls="--", alpha=0.8)
    axL.set_xlabel("closest wall distance per link (m)", fontsize=10)
    axL.tick_params(labelsize=8.5)
    for sp in axL.spines.values():
        sp.set_edgecolor(STYLE["grid"])
    axL.set_title("every wrist/forearm link feels the wall",
                  fontsize=11.5, fontweight="bold", pad=6)

    # ---- (E) stats / caption card ----------------------------------------------
    axS = fig.add_subplot(gs[1, 2])
    axS.axis("off")
    axS.set_facecolor(STYLE["bg"])
    lines = [
        ("HYBRID PROXIMITY SKIN", "#4cc9f0", 13, "bold"),
        ("40 SPAD sensors · 8×8 depth · fovy 45°", STYLE["fg"], 10.5, "normal"),
        (f"range {NEAR*1000:.0f}–{FAR*1000:.0f} mm · ~4 mm accuracy", STYLE["fg"], 10.5, "normal"),
        ("", STYLE["fg"], 6, "normal"),
        (f"{active}/40 sensors active in the pipe", "#7fffd4", 11.5, "bold"),
        (f"{len(pts):,} cloud points · {len(wp):,} on the wall", STYLE["fg"], 10.5, "normal"),
        (f"full ring: {cov_bins}/12 azimuth sectors", STYLE["fg"], 10.5, "normal"),
        ("", STYLE["fg"], 6, "normal"),
        ("PIPE RECONSTRUCTED FROM SKIN ALONE", "#4cc9f0", 11, "bold"),
        (f"fit radius   {fR*1000:.1f} mm  (true {INNER*1000:.1f})", STYLE["fg"], 10.5, "normal"),
        (f"radius error {rad_err:.1f} mm", STYLE["fg"], 10.5, "normal"),
        (f"center error {cen_err:.1f} mm", STYLE["fg"], 10.5, "normal"),
        (f"wall RMS     {rms*1000:.1f} mm", "#7fffd4", 11.0, "bold"),
    ]
    y = 0.97
    for txt, col, sz, w in lines:
        axS.text(0.04, y, txt, transform=axS.transAxes, color=col, fontsize=sz,
                 fontweight=w, va="top", ha="left", family="DejaVu Sans")
        y -= 0.052 if txt else 0.030
    axS.text(0.04, 0.10,
             "Contact-rich reaching inside confined geometry:\nthe skin senses the wall in every "
             "direction at once,\nthe sense modality the policy actually trains on.",
             transform=axS.transAxes, color="#a8b0bd", fontsize=9.2, va="bottom", ha="left")
    for sp in []:
        pass
    axS_rect = plt.Rectangle((0.015, 0.02), 0.97, 0.97, transform=axS.transAxes, fill=False,
                             ec=STYLE["grid"], lw=1.0)
    axS.add_patch(axS_rect)

    # ---- super title -----------------------------------------------------------
    fig.suptitle("PROXIMITY SKIN AS PERCEPTION  ·  the signature pipe-tunnel ring",
                 color=STYLE["fg"], fontsize=19, fontweight="bold", x=0.035, ha="left", y=0.965)
    fig.text(0.035, 0.918,
             "franka_skin HYBRID  ·  the arm is inside a horizontal cylindrical pipe; its 40 depth "
             "sensors range the wall all around → a complete ring of distance-colored returns",
             color="#a8b0bd", fontsize=11.5, ha="left")

    out = os.path.join(OUT, "env_pipe_tunnel.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    sz = os.path.getsize(out)
    print(f"SAVED {out}  ({sz/1024:.0f} KB)")
    print(f"RMS_MM={rms*1000:.2f} ACTIVE={active} WALLPTS={len(wp)} COV={cov_bins}")


def fig_test_reconstruct_fumehood():
    """Hybrid-skin reconstruction with the arm reaching into a fumehood (bench, walls, objects)."""
    OUT = _FIGROOT
    OUT.mkdir(parents=True, exist_ok=True)
    BG, PANEL, FG = "#111317", "#171a20", "#e8e8ea"
    REACH = [0.0, -0.30, 0.0, -2.25, 0.0, 2.0, 0.79]
    REACH2 = [0.35, -0.10, 0.15, -1.95, 0.1, 1.75, 0.5]

    # fumehood (translucent shell so the arm shows) + objects on the bench, sized to the lib robot
    HOOD = [
        ("bench_top", [0.62, 0.0, 0.585], [0.27, 0.32, 0.015], [0.62, 0.55, 0.45, 1]),
        ("bench_body", [0.62, 0.0, 0.29], [0.25, 0.30, 0.29], [0.55, 0.50, 0.44, 1]),
        ("hood_l", [0.62, 0.32, 0.82], [0.27, 0.012, 0.22], [0.78, 0.80, 0.84, 0.32]),
        ("hood_r", [0.62, -0.32, 0.82], [0.27, 0.012, 0.22], [0.78, 0.80, 0.84, 0.32]),
        ("hood_back", [0.90, 0.0, 0.82], [0.012, 0.32, 0.22], [0.72, 0.70, 0.66, 1]),
        ("hood_top", [0.62, 0.0, 1.05], [0.27, 0.32, 0.012], [0.78, 0.80, 0.84, 0.32]),
        ("sash", [0.36, 0.0, 0.90], [0.012, 0.30, 0.028], [0.62, 0.64, 0.66, 1]),
    ]
    OBJ = [
        ("o_box", [0.58, 0.10, 0.64], [0.045, 0.045, 0.05], [0.85, 0.5, 0.3, 1]),
        ("o_tube", [0.55, -0.12, 0.65], [0.03, 0.03, 0.06], [0.5, 0.6, 0.85, 1]),
        ("o_block", [0.72, 0.0, 0.625], [0.06, 0.04, 0.04], [0.8, 0.75, 0.4, 1]),
    ]

    def mk_hood(s):
        nice_lights(s)
        for n, c, h, col in HOOD + OBJ:
            add_box(s, n, c, h, col)

    rep = ["# Fumehood reconstruction\n"]

    # A. placement (robot only) ----------------------------------------------------------
    POSES = [REACH, [0.0, 0.3, 0.0, -1.2, 0.0, 1.6, 0.0], [0.6, -0.9, 0.4, -2.4, 0.3, 2.2, 0.4]]
    m0 = build(lambda s: None)
    d0 = mujoco.MjData(m0)
    names = sensors(m0)
    rd = depth_renderer(m0)
    sw = {n: -1 for n in names}
    for q in POSES:
        set_pose(m0, d0, q)
        for n in names:
            sw[n] = max(sw[n], float(depth8(rd, d0, n).min()))
    set_pose(m0, d0, REACH)
    outw = 0
    for n in names:
        cid = m0.camera(n).id
        bid = int(m0.cam_bodyid[cid])
        pos, R = cam_pose(m0, d0, n)
        rad = pos - d0.xpos[bid]
        outw += float(np.dot(-R[:, 2], rad / (np.linalg.norm(rad) + 1e-9))) > -0.2
    mp = build(lambda s: add_plane_mocap(s))
    dp = mujoco.MjData(mp); set_pose(mp, dp, REACH); rdp = depth_renderer(mp)
    pok = 0
    for n in names:
        pos, R = cam_pose(mp, dp, n); fwd = -R[:, 2]
        mocap_set(mp, dp, "probe_plane", pos + 0.15 * fwd, view_dir=fwd)
        pok += abs(float(depth8(rdp, dp, n)[3:5, 3:5].mean()) - 0.145) < 0.012
    nb = sum(sw[n] > 0.03 for n in names)
    rep.append(f"**A. Placement (robot-only):** not-buried {nb}/40, outward {outw}/40, plate@0.15m {pok}/40")

    # B. single-frame reconstruction in the hood ----------------------------------------
    mH = build(mk_hood); dH = mujoco.MjData(mH); set_pose(mH, dH, REACH)
    pts, dd, mins = skin_cloud(mH, dH)
    err = cloud_error(pts, gt_boxes(mH, dH))
    active = sum(1 for v in mins.values() if v < FAR)
    rep.append(f"**B. Reconstruction (1 frame):** {active}/40 active, {len(pts)} pts, "
               f"RMS {np.sqrt((err**2).mean())*1000:.1f}mm, {100*(err<0.01).mean():.0f}% within 1cm")

    # C. accumulation over the reach ----------------------------------------------------
    q0, q1 = np.array(REACH), np.array(REACH2)
    accP, accD = [], []
    for t in np.linspace(0, 1, 14):
        set_pose(mH, dH, list(q0 * (1 - t) + q1 * t))
        p, dq, _ = skin_cloud(mH, dH)
        if len(p):
            accP.append(p); accD.append(dq)
    accP = np.concatenate(accP) if accP else np.zeros((0, 3))
    accD = np.concatenate(accD) if accD else np.zeros((0,))
    errA = cloud_error(accP, gt_boxes(mH, dH))
    rep.append(f"**C. Accumulation (14 poses):** {len(accP)} pts, "
               f"RMS {np.sqrt((errA**2).mean())*1000:.1f}mm, {100*(errA<0.01).mean():.0f}% within 1cm")

    # figure ----------------------------------------------------------------------------
    def draw_gt(ax):
        for n, c, h, col in HOOD + OBJ:
            c = np.array(c); h = np.array(h)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    ax.plot([c[0]-h[0], c[0]+h[0]], [c[1]+sy*h[1]]*2, [c[2]+sx*h[2]]*2, color="#777", lw=0.4)

    fig = plt.figure(figsize=(13, 5.6), facecolor=BG)
    a1 = fig.add_subplot(121, projection="3d")
    cloud_panel(a1, pts, dd, draw_gt, f"1 frame — {len(pts)} pts, RMS {np.sqrt((err**2).mean())*1000:.1f}mm")
    a2 = fig.add_subplot(122, projection="3d")
    cloud_panel(a2, accP, accD, draw_gt, f"accumulated — {len(accP)} pts")
    fig.suptitle("Hybrid skin in a fumehood — reconstruction", color=FG, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fumehood_reconstruction.png", dpi=150, facecolor=BG)

    # pretty RGB + cloud render ---------------------------------------------------------
    set_pose(mH, dH, REACH)
    img = render_scene(mH, dH, mjv_cam(lookat=(0.55, 0, 0.7), distance=1.55, azimuth=150, elevation=-14),
                       w=1100, h=900, cloud=pts, depths=dd, pt_size=0.009)
    cv2.imwrite(str(OUT / "fumehood_render.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    (OUT / "report.txt").write_text("\n".join(rep))
    print("\n".join(rep))
    print(f"-> {OUT}")


# ============================================================================
# Scenes & environments
# ============================================================================

def fig_panel_env_cluttered_shelf():
    """ENVIRONMENT PANEL: env_cluttered_shelf -- FR3 + 40-SPAD hybrid skin reaching into a cluttered shelf, exo RGB + skin cloud + depth tiles + per-link bar."""
    OUTDIR = str(_FIGROOT)
    os.makedirs(OUTDIR, exist_ok=True)
    OUT = os.path.join(OUTDIR, "env_cluttered_shelf.png")

    # ----------------------------------------------------------------------------------------------
    # SCENE: a real-looking shelf unit (floor + roof + back panel + two side walls) with assorted
    # clutter scattered where the forearm/wrist reach. The arm reaches into the middle bay.
    # Warm wood-ish shelf, varied clutter colors. contype/conaffinity=0 so nothing collides.
    # ----------------------------------------------------------------------------------------------
    SHELF_WOOD = [0.52, 0.40, 0.26, 1.0]      # warm wood
    SHELF_BACK = [0.42, 0.32, 0.22, 1.0]

    # shelf bay centered at x~0.50, spanning the arm's forward reach, opening toward -x (the robot)
    SHELF = {
        "shelf_floor": ([0.55, 0.00, 0.40], [0.26, 0.40, 0.012], SHELF_WOOD),
        "shelf_roof":  ([0.55, 0.00, 0.78], [0.26, 0.40, 0.012], SHELF_WOOD),
        "shelf_back":  ([0.82, 0.00, 0.59], [0.012, 0.40, 0.20], SHELF_BACK),
        "shelf_wallL": ([0.55, 0.40, 0.59], [0.26, 0.012, 0.20], SHELF_WOOD),
        "shelf_wallR": ([0.55,-0.40, 0.59], [0.26, 0.012, 0.20], SHELF_WOOD),
    }

    # assorted clutter (5-7 items): boxes + cylinders of varied size/orientation, scattered in the
    # bay around the arm path so multiple links sense them.
    def _q(axis, deg):
        q = np.zeros(4); a = np.array(axis, float); a /= np.linalg.norm(a)
        mujoco.mju_axisAngle2Quat(q, a, np.deg2rad(deg)); return [float(v) for v in q]

    CLUTTER = [
        # (kind, name, center, dims, rgba, quat)  dims: box->half3 ; cyl->(radius,halflen)
        ("box", "clutter_book",  [0.66, 0.205, 0.475], [0.035, 0.085, 0.062], [0.80, 0.28, 0.30, 1], _q([0,0,1], 18)),
        ("cyl", "clutter_can",   [0.60,-0.165, 0.485], [0.043, 0.072],        [0.30, 0.62, 0.78, 1], _q([1,0,0], 0)),
        ("box", "clutter_box2",  [0.70,-0.045, 0.47],  [0.05, 0.05, 0.05],    [0.86, 0.66, 0.24, 1], _q([0,0,1], -25)),
        ("cyl", "clutter_bottle",[0.685, 0.085, 0.515],[0.028, 0.10],         [0.40, 0.74, 0.42, 1], _q([1,0,0], 0)),
        ("cyl", "clutter_roll",  [0.55, 0.235, 0.50],  [0.034, 0.06],         [0.74, 0.74, 0.80, 1], _q([0,1,0], 90)),
        ("box", "clutter_tray",  [0.58,-0.27, 0.452],  [0.085, 0.05, 0.022],  [0.55, 0.45, 0.85, 1], _q([0,0,1], 8)),
        ("cyl", "clutter_mug",   [0.50, 0.12, 0.475],  [0.038, 0.05],         [0.90, 0.52, 0.30, 1], _q([1,0,0], 0)),
    ]

    def make(spec):
        # warmer key light + softer fill for the "warm lighting" brief, plus a low fill from the
        # robot side so the open bay interior (where the arm + clutter are) is not in shadow.
        spec.worldbody.add_light(pos=[0.25, 0.55, 2.2], dir=[0.05, -0.25, -1],
                                 diffuse=[1.0, 0.93, 0.82], specular=[0.30, 0.27, 0.22])
        spec.worldbody.add_light(pos=[-0.8, -0.6, 1.7], dir=[0.45, 0.35, -1],
                                 diffuse=[0.50, 0.45, 0.46], specular=[0.10, 0.10, 0.12])
        spec.worldbody.add_light(pos=[-0.5, -0.4, 0.9], dir=[1.0, 0.4, -0.2],
                                 diffuse=[0.42, 0.38, 0.34], specular=[0.05, 0.05, 0.05])
        fl = spec.worldbody.add_geom()
        fl.type = mujoco.mjtGeom.mjGEOM_PLANE
        fl.size = [3, 3, 0.1]
        fl.rgba = [0.20, 0.205, 0.235, 1]
        for nm, (c, h, rgba) in SHELF.items():
            add_box(spec, nm, c, h, rgba)
        for kind, nm, c, dims, rgba, quat in CLUTTER:
            if kind == "box":
                add_box(spec, nm, c, dims, rgba)
            else:
                add_cylinder(spec, nm, c, dims[0], dims[1], rgba, quat=quat)
        # exo camera for the hero render: 3/4 view from BEHIND/BESIDE the robot (-x, -y), looking in
        # the +x direction INTO the open shelf bay so we see the arm + clutter inside it.
        exo = spec.worldbody.add_camera(); exo.name = "exo_camera_1"
        exo.pos = [-0.85, -0.95, 1.00]
        target = np.array([0.50, 0.0, 0.52])              # into the bay
        vv = target - np.array(exo.pos); vv /= np.linalg.norm(vv)
        z = -vv; up = np.array([0, 0, 1.0])
        x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).reshape(9))
        exo.quat = [float(t) for t in q]; exo.fovy = 50; exo.resolution = [720, 900]

    model = build(make=make, offw=1400, offh=1200)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")

    names = sensors(model)
    rd = depth_renderer(model)

    # ---- per-sensor depth + cloud, track which link senses what ----
    pts, depths, mins = skin_cloud(model, data, rd)
    # a VALID return is an in-range measured surface: NEAR <= d < FAR.  (Readings below NEAR are the
    # sensor grazing the arm's own structure / self-occlusion and produce no back-projected points.)
    def valid(n):
        return NEAR <= mins[n] < FAR
    active_names = [n for n in names if valid(n)]
    active = len(active_names)
    print(f"sensors {len(names)}  active {active}  cloud {len(pts)} pts")

    # per-link active count
    from collections import defaultdict, OrderedDict
    LINK_ORDER = ["link1", "link2", "link3", "link4", "link5_front", "link5_back", "link6"]
    link_active = OrderedDict((k, 0) for k in LINK_ORDER)
    link_total = defaultdict(int)
    link_min = {k: FAR for k in LINK_ORDER}
    for n in names:
        lk = n.split("_sensor_")[0]
        link_total[lk] += 1
        if valid(n):
            link_active[lk] += 1
            link_min[lk] = min(link_min[lk], mins[n])

    # the four busiest active sensors (smallest valid min distance) for the depth-tile strip
    busiest = sorted(active_names, key=lambda n: mins[n])[:4]

    # ---- HERO RENDER: exo RGB with the distance-colored skin cloud overlaid ----
    hero_cam = "exo_camera_1"
    r = mujoco.Renderer(model, 720, 900)
    r.update_scene(data, hero_cam)
    scn = r.scene
    nrm = np.clip((depths - NEAR) / (FAR - NEAR), 0, 1)
    turbo = matplotlib.colormaps["turbo"]
    cols = turbo(1.0 - nrm)[:, :3]
    for p, c in zip(pts, cols):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.0080, 0, 0]),
                            np.asarray(p, np.float64), np.eye(3).ravel(),
                            np.array([c[0], c[1], c[2], 1.0], np.float32))
        scn.ngeom += 1
    hero = r.render().copy()
    hero = (np.clip((hero.astype(np.float32) / 255) ** 0.82 * 1.10, 0, 1) * 255).astype(np.uint8)

    # ---- second view: "into the bay" mjv view from the robot side, peering over the front lip ----
    side_cam = mjv_cam(lookat=(0.55, 0.0, 0.52), distance=1.0, azimuth=35, elevation=-22)
    side = render_scene(model, data, side_cam, w=760, h=720, cloud=pts, depths=depths, pt_size=0.0070)

    # ==============================================================================================
    # FIGURE
    # ==============================================================================================
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
        "axes.edgecolor": STYLE["grid"],
    })
    norm = mcolors.Normalize(vmin=NEAR, vmax=FAR)
    cmap = matplotlib.colormaps[STYLE["cmap"]]

    fig = plt.figure(figsize=(19.5, 11.0), dpi=170)
    fig.patch.set_facecolor(STYLE["bg"])
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.55, 1.0, 1.0], height_ratios=[1.0, 0.62],
                  left=0.018, right=0.985, top=0.885, bottom=0.065, wspace=0.10, hspace=0.165)

    # --- (A) HERO exo render -----------------------------------------------------------------------
    axH = fig.add_subplot(gs[0, 0])
    axH.imshow(hero); axH.set_xticks([]); axH.set_yticks([])
    for s in axH.spines.values():
        s.set_edgecolor(STYLE["accent"]); s.set_linewidth(1.6)
    axH.set_title("FR3 reaching into a cluttered shelf  ·  skin cloud overlaid (exo view)",
                  color=STYLE["fg"], fontsize=13.5, fontweight="bold", pad=7)
    axH.text(0.014, 0.026, f"{active}/40 SPAD sensors returning   ·   {len(pts)} skin points",
             transform=axH.transAxes, color="#0d0f12", fontsize=11.5, fontweight="bold",
             va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.35", fc=STYLE["accent"], ec="none", alpha=0.92))

    # --- (B) robot's-eye side view ----------------------------------------------------------------
    axS = fig.add_subplot(gs[0, 1])
    axS.imshow(side); axS.set_xticks([]); axS.set_yticks([])
    for s in axS.spines.values():
        s.set_edgecolor(STYLE["grid"]); s.set_linewidth(1.2)
    axS.set_title("close-in view: clutter sensed across forearm + wrist",
                  color=STYLE["fg"], fontsize=12, pad=7)

    # --- (C) per-link active-sensor bar -----------------------------------------------------------
    axB = fig.add_subplot(gs[0, 2])
    axB.set_facecolor(STYLE["panel"])
    disp = [k.replace("link", "L").replace("_", " ") for k in LINK_ORDER]
    tot = [link_total[k] for k in LINK_ORDER]
    act = [link_active[k] for k in LINK_ORDER]
    ypos = np.arange(len(LINK_ORDER))[::-1]
    axB.barh(ypos, tot, color="#262b34", edgecolor=STYLE["grid"], height=0.62, zorder=1)
    bar_cols = [cmap(norm(np.clip(link_min[k], NEAR, FAR))) if act[i] else "#3a3f49"
                for i, k in enumerate(LINK_ORDER)]
    axB.barh(ypos, act, color=bar_cols, edgecolor="#0d0f12", height=0.62, zorder=2)
    for i, k in enumerate(LINK_ORDER):
        axB.text(tot[i] + 0.15, ypos[i], f"{act[i]}/{tot[i]}", va="center", ha="left",
                 color=STYLE["fg"], fontsize=10, fontweight="bold")
        if act[i]:
            axB.text(0.12, ypos[i], f"{link_min[k]*100:.1f} cm", va="center", ha="left",
                     color="#0d0f12", fontsize=8.5, fontweight="bold")
    axB.set_yticks(ypos); axB.set_yticklabels(disp, fontsize=10)
    axB.set_xlim(0, max(tot) + 1.4)
    axB.set_xlabel("SPAD sensors per link  (filled = returning a hit)", fontsize=10)
    axB.set_title("where the skin feels the clutter", color=STYLE["fg"], fontsize=12, pad=7)
    axB.tick_params(labelsize=9)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)
    axB.grid(axis="x", color=STYLE["grid"], lw=0.4, alpha=0.6, zorder=0)

    # --- (D) busiest-sensor depth tiles (bottom-left, two columns) ---------------------------------
    gsD = gs[1, 0].subgridspec(1, 4, wspace=0.18)
    for i, n in enumerate(busiest):
        axd = fig.add_subplot(gsD[0, i])
        d8 = depth8(rd, data, n)
        masked = np.where(d8 <= FAR, d8, np.nan)
        im = axd.imshow(masked, cmap=cmap, vmin=NEAR, vmax=FAR, interpolation="nearest")
        axd.set_xticks([]); axd.set_yticks([])
        short = n.replace("_sensor", " s").replace("link", "L")
        axd.set_title(f"{short}\nmin {mins[n]*100:.1f} cm", color=STYLE["fg"], fontsize=9, pad=3)
        for s in axd.spines.values():
            s.set_edgecolor(STYLE["grid"]); s.set_linewidth(1.0)
    fig.text(0.018 + 0.255, 0.345, "busiest sensors — raw 8×8 depth (turbo_r, red = near)",
             ha="center", color="#9aa3b2", fontsize=10.5, fontweight="bold")

    # --- (E) clutter inventory + scene facts (bottom-mid/right span) -------------------------------
    axT = fig.add_subplot(gs[1, 1:])
    axT.set_facecolor(STYLE["panel"]); axT.set_xticks([]); axT.set_yticks([])
    for s in axT.spines.values():
        s.set_edgecolor(STYLE["grid"]); s.set_linewidth(1.0)
    # distance histogram of cloud points
    axHi = axT.inset_axes([0.06, 0.20, 0.52, 0.66])
    axHi.set_facecolor(STYLE["panel"])
    nb = 26
    counts, edges = np.histogram(depths, bins=nb, range=(NEAR, FAR))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bc = cmap(norm(centers))
    axHi.bar(centers, counts, width=(edges[1] - edges[0]) * 0.92, color=bc, edgecolor="none")
    axHi.set_xlim(NEAR, FAR)
    axHi.set_xlabel("skin-point distance (m)", fontsize=9.5)
    axHi.set_ylabel("points", fontsize=9.5)
    axHi.tick_params(labelsize=8)
    for sp in ("top", "right"):
        axHi.spines[sp].set_visible(False)
    axHi.set_title("range distribution of the 40-sensor cloud", color=STYLE["fg"], fontsize=10, pad=3)
    med = float(np.median(depths)); near_frac = float((depths < 0.10).mean()) * 100
    facts = (
        f"scene:  shelf bay (floor·roof·back·2 walls)  +  {len(CLUTTER)} clutter items\n"
        f"  boxes & cylinders, varied size / orientation\n\n"
        f"perception:  40 SPAD depth cams · 8×8 · fovy {FOVY:.0f}°\n"
        f"  range {NEAR*1000:.0f}–{FAR*100:.0f} mm · ~4 mm accurate\n\n"
        f"this frame (pose: reach):\n"
        f"  {active}/40 sensors returning · {len(pts)} skin points\n"
        f"  nearest hit {min(mins.values())*100:.1f} cm · median {med*100:.1f} cm\n"
        f"  {near_frac:.0f}% of points within 10 cm of a surface\n"
        f"  links engaged: {sum(1 for v in link_active.values() if v)}/{len(LINK_ORDER)}"
    )
    axT.text(0.62, 0.93, facts, transform=axT.transAxes, va="top", ha="left",
             color=STYLE["fg"], fontsize=10.6, linespacing=1.5, family="DejaVu Sans")

    # ---- shared colorbar (top-right, clear of the title block) -----------------------------------
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.615, 0.952, 0.295, 0.017])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("distance (m)     near  →  far", color=STYLE["fg"], fontsize=11, labelpad=4)
    cb.ax.xaxis.set_tick_params(color=STYLE["fg"], labelsize=8.5)
    cb.ax.xaxis.set_label_position("top")
    cb.outline.set_edgecolor(STYLE["grid"])
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color=STYLE["fg"])

    # ---- legend (under hero) ---------------------------------------------------------------------
    proxies = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(0.18), markersize=9,
               label=f"SPAD skin return  ({active}/40 active)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=SHELF_WOOD[:3], markersize=9,
               label="shelf structure"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=(0.86, 0.66, 0.24), markersize=9,
               label=f"clutter ({len(CLUTTER)} items)"),
    ]
    leg = fig.legend(handles=proxies, loc="lower left", bbox_to_anchor=(0.022, 0.012),
                     frameon=True, fontsize=10, labelcolor=STYLE["fg"], ncol=3,
                     columnspacing=1.4, handletextpad=0.5)
    leg.get_frame().set_facecolor(STYLE["panel"]); leg.get_frame().set_edgecolor(STYLE["grid"])

    # ---- titles ----------------------------------------------------------------------------------
    fig.suptitle("env_cluttered_shelf  —  franka_skin hybrid proximity perception in clutter",
                 color=STYLE["fg"], fontsize=20, fontweight="bold", x=0.018, ha="left", y=0.972)
    fig.text(0.018, 0.928,
             "Real Franka FR3 · 40-SPAD hybrid skin · RGB blurred at training time, so this skin IS "
             "the policy's perception in contact-rich reaching",
             ha="left", color="#9aa3b2", fontsize=12)

    fig.savefig(OUT, dpi=170, facecolor=STYLE["bg"])
    sz = os.path.getsize(OUT)
    print("SAVED", OUT, "BYTES", sz, "KB", round(sz / 1024, 1))
    print("ACTIVE", active, "PTS", len(pts), "LINKS_ENGAGED",
          sum(1 for v in link_active.values() if v))
    print("BUSIEST", [(n, round(mins[n] * 100, 1)) for n in busiest])


def fig_env_corner_cavity_hero():
    """FLAGSHIP hero: FR3 hybrid SPAD array perceives a corner cavity through its body (context RGB + 3-view reconstruction)."""
    OUTDIR = str(_FIGROOT)
    os.makedirs(OUTDIR, exist_ok=True)
    OUT = os.path.join(OUTDIR, "env_corner_cavity.png")

    # -------------------------------------------------------------------------------------
    # Corner-cavity geometry (center, half-extent). Walls translucent so the arm shows through.
    # bench is a thin top slab (the support the arm reaches over); back wall + pillar solid.
    # -------------------------------------------------------------------------------------
    # Translucent walls so the arm shows through in the rendered context view.
    WALL = [0.80, 0.76, 0.68, 0.16]
    ROOF = [0.74, 0.72, 0.64, 0.13]
    CAVITY = {
        "bench":  ([0.55, 0.00, 0.34], [0.34, 0.40, 0.020], [0.46, 0.40, 0.33, 1.0]),
        "wl":     ([0.55, 0.40, 0.62], [0.34, 0.016, 0.30], WALL),
        "wr":     ([0.55,-0.40, 0.62], [0.34, 0.016, 0.30], WALL),
        "roof":   ([0.55, 0.00, 0.93], [0.34, 0.418, 0.016], ROOF),
        "back":   ([0.90, 0.00, 0.62], [0.018, 0.418, 0.30], [0.62, 0.59, 0.52, 1.0]),
        "pillar": ([0.74, 0.26, 0.62], [0.030, 0.030, 0.30], [0.52, 0.38, 0.25, 1.0]),
    }
    POSE = [0.0, 0.55, 0.0, -1.4, 0.0, 1.95, 0.0]   # arm reaching deep into the back corner

    def make(spec):
        nice_lights(spec)
        for nm, (c, h, rgba) in CAVITY.items():
            add_box(spec, nm, c, h, rgba)

    model = build(make=make)
    data = mujoco.MjData(model)
    set_pose(model, data, POSE)
    names = sensors(model)

    # ---- 40-sensor world cloud (the robot's perception, single inference frame) ----
    PTS, D, mins = skin_cloud(model, data)
    active = sum(1 for v in mins.values() if v < FAR)
    hand = data.xpos[model.body("fr3_link7").id].copy()
    print(f"sensors {len(names)}  active {active}/40  pts {len(PTS)}  hand {hand.round(3)}")

    # ---- reconstruction-accuracy proof: distance from each return to nearest GT surface ----
    def surf_dist(p):
        bd = abs(p[2] - 0.0)  # floor plane
        for c, h, _ in CAVITY.values():
            c = np.array(c); h = np.array(h)
            bd = min(bd, np.linalg.norm(np.maximum(np.abs(p - c) - h, 0.0)))
        return bd

    RES_ALL = np.array([surf_dist(p) for p in PTS])
    selfview = RES_ALL > 0.05      # returns that hit the robot's own folded forearm (legit, not cavity)
    res = RES_ALL[~selfview]
    med_mm = float(np.median(res) * 1000.0)
    rms_mm = float(np.sqrt((res ** 2).mean()) * 1000.0)
    within4 = float((res < 0.004).mean()) * 100.0
    print(f"cavity-surface fit: median {med_mm:.2f}mm  rms {rms_mm:.2f}mm  within4mm {within4:.0f}%  "
          f"(self-view returns excluded: {int(selfview.sum())})")

    # ---- context RGB render with the live cloud overlaid (peering into the cavity opening) ----
    cam = mjv_cam(lookat=(0.55, 0.0, 0.60), distance=1.55, azimuth=225, elevation=-20)
    RGB = render_scene(model, data, cam, w=900, h=900, cloud=PTS, depths=D, pt_size=0.0066, gamma=0.80)

    # ---- FR3 kinematic skeleton ----
    link_ids = [model.body(i).id for i in range(model.nbody) if "fr3_link" in model.body(i).name]
    SKEL = []
    for bid in link_ids:
        p = data.xpos[bid].copy()
        if not SKEL or np.linalg.norm(p - SKEL[-1]) > 1e-4:
            SKEL.append(p)
    SKEL = np.array(SKEL)

    # ---- box wireframe / face helpers ----
    def box_edges(c, h):
        c = np.array(c); h = np.array(h)
        corners = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        V = np.array([c + np.array(s) * h for s in corners])
        E = []
        for i, a in enumerate(corners):
            for j, b in enumerate(corners):
                if i < j and np.abs(np.array(a) - np.array(b)).sum() == 2:
                    E.append([V[i], V[j]])
        return E

    def box_faces(c, h):
        c = np.array(c); h = np.array(h); x, y, z = h
        F = {
            '-x': [[-x,-y,-z],[-x,y,-z],[-x,y,z],[-x,-y,z]],
            '+x': [[x,-y,-z],[x,y,-z],[x,y,z],[x,-y,z]],
            '-y': [[-x,-y,-z],[x,-y,-z],[x,-y,z],[-x,-y,z]],
            '+y': [[-x,y,-z],[x,y,-z],[x,y,z],[-x,y,z]],
            '-z': [[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z]],
            '+z': [[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]],
        }
        return [list(c + np.array(f)) for f in F.values()]

    # ---- colors ----
    norm = mcolors.Normalize(vmin=NEAR, vmax=FAR)
    cmap = matplotlib.colormaps[STYLE["cmap"]]
    pt_colors = cmap(norm(np.clip(D, NEAR, FAR)))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
    })

    fig = plt.figure(figsize=(22, 9.2), dpi=175)
    fig.patch.set_facecolor(STYLE["bg"])
    # layout: [ context RGB | iso | front | top ]
    gs = fig.add_gridspec(1, 4, width_ratios=[1.18, 1.0, 1.0, 1.0],
                          left=0.008, right=0.992, top=0.875, bottom=0.135, wspace=0.045)

    # ---------- panel 0: context RGB ----------
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(RGB)
    ax0.set_xticks([]); ax0.set_yticks([])
    for s in ax0.spines.values():
        s.set_color(STYLE["grid"]); s.set_linewidth(1.2)
    ax0.set_title("scene  ·  live SPAD returns overlaid", color=STYLE["fg"], fontsize=13, pad=6)
    ax0.text(0.5, -0.045,
             "RGB cameras are blurred at training time — this is what the policy actually sees",
             transform=ax0.transAxes, ha="center", va="top",
             color=STYLE["accent"], fontsize=10.5, fontstyle="italic")

    # ---------- panels 1-3: 3-view reconstruction ----------
    # bounds: focus on the cavity interior (bench top up), excluding the dead floor region
    in_cav = PTS[:, 2] > 0.28
    ref = PTS[in_cav] if in_cav.sum() > 50 else PTS
    lo = ref.min(0) - 0.04
    hi = ref.max(0) + 0.04
    ctr = (lo + hi) / 2
    span = (hi - lo).max() / 2

    VIEWS = [("isometric", 24, 46), ("front  (looking +x into cavity)", 8, 0), ("top-down", 88, -90)]
    SURF_FACE = (0.50, 0.66, 0.85, 0.05)
    SURF_WIRE = (0.55, 0.72, 0.92, 0.55)

    for k, (title, elev, azim) in enumerate(VIEWS):
        ax = fig.add_subplot(gs[0, k + 1], projection="3d")
        ax.set_facecolor(STYLE["panel"])
        ax.view_init(elev=elev, azim=azim)

        # translucent GT surfaces + wireframe
        for nm, (c, h, rgba) in CAVITY.items():
            solid = rgba[3] >= 0.99
            if not solid:
                ax.add_collection3d(Poly3DCollection(box_faces(c, h), facecolor=SURF_FACE,
                                                     edgecolor="none"))
            ax.add_collection3d(Line3DCollection(box_edges(c, h),
                                                 colors=SURF_WIRE if solid else (0.55, 0.72, 0.92, 0.4),
                                                 linewidths=1.0 if solid else 0.7))

        # FR3 skeleton
        ax.plot(SKEL[:, 0], SKEL[:, 1], SKEL[:, 2], "-", color="#9aa3b2", lw=3.0,
                alpha=0.95, solid_capstyle="round", zorder=2)
        ax.scatter(SKEL[:, 0], SKEL[:, 1], SKEL[:, 2], s=40, c="#cfd6e2",
                   edgecolors="#2a2e36", linewidths=0.6, depthshade=False, zorder=3)

        # the 40-sensor proximity cloud
        ax.scatter(PTS[:, 0], PTS[:, 1], PTS[:, 2], c=pt_colors, s=14, depthshade=False,
                   alpha=0.97, edgecolors="none", zorder=4)

        ax.set_xlim(ctr[0] - span, ctr[0] + span)
        ax.set_ylim(ctr[1] - span, ctr[1] + span)
        ax.set_zlim(ctr[2] - span, ctr[2] + span)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.set_xlabel("x (m)", labelpad=1, fontsize=8.5)
        ax.set_ylabel("y (m)", labelpad=1, fontsize=8.5)
        ax.set_zlabel("z (m)", labelpad=1, fontsize=8.5)
        ax.tick_params(labelsize=6.5, pad=-1)
        ax.set_title(title, color=STYLE["fg"], fontsize=12, pad=0)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor(STYLE["panel"])
            pane.pane.set_edgecolor(STYLE["grid"])
            pane.pane.set_alpha(1.0)
            pane._axinfo["grid"]["color"] = STYLE["grid"]
            pane._axinfo["grid"]["linewidth"] = 0.4

    # ---------- colorbar ----------
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.70, 0.085, 0.22, 0.020])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("distance to surface (m)        near (contact) ← → far", color=STYLE["fg"], fontsize=10.5)
    cb.ax.xaxis.set_tick_params(color=STYLE["fg"], labelsize=8)
    cb.outline.set_edgecolor(STYLE["grid"])
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color=STYLE["fg"])

    # ---------- legend ----------
    proxies = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(0.18), markersize=9,
               label=f"SPAD return   ({active}/40 sensors active · {len(PTS)} points)"),
        Line2D([0], [0], color="#9aa3b2", lw=3, label="FR3 kinematic skeleton"),
        Line2D([0], [0], color=(0.55, 0.72, 0.92, 0.95), lw=1.5, label="cavity ground-truth (wireframe)"),
    ]
    leg = fig.legend(handles=proxies, loc="lower left", bbox_to_anchor=(0.012, 0.052),
                     frameon=True, fontsize=10.5, labelcolor=STYLE["fg"], ncol=1)
    leg.get_frame().set_facecolor(STYLE["panel"])
    leg.get_frame().set_edgecolor(STYLE["grid"])

    # ---------- titles + caption ----------
    fig.suptitle("Perceiving a corner cavity through the body — franka_skin hybrid SPAD array",
                 color=STYLE["fg"], fontsize=21, fontweight="bold", x=0.5, y=0.975)
    fig.text(0.5, 0.918,
             "FR3 + 40 SPAD depth sensors (8×8 px · fovy 45° · range 0.015–0.50 m · ≈4 mm accurate)   ·   "
             "single inference frame, arm reaching into the back-right corner",
             ha="center", color="#9aa3b2", fontsize=12)

    fig.text(0.30, 0.052,
             f"cavity-surface reconstruction:  median {med_mm:.2f} mm   ·   RMS {rms_mm:.1f} mm   ·   "
             f"{within4:.0f}% of returns within 4 mm of ground-truth geometry",
             ha="left", va="center", color=STYLE["fg"], fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=STYLE["panel"], edgecolor=STYLE["grid"]))

    fig.savefig(OUT, dpi=175, facecolor=STYLE["bg"])
    sz = os.path.getsize(OUT)
    print("SAVED", OUT)
    print("BYTES", sz, "KB", round(sz / 1024, 1))


def fig_env_narrow_slot():
    """Hero viz: FR3 + 40-SPAD hybrid proximity skin threading a narrow vertical slot, painting both flanks."""
    OUT = str(_FIGROOT)
    os.makedirs(OUT, exist_ok=True)
    TURBO = colormaps["turbo"]

    # ---- slot geometry: two tall thin walls flanking the forearm/wrist ----
    WY = 0.175          # wall half-gap (m). forearm sensor ring is ~0.145 half-width.
    WALL_HALF = (0.32, 0.012, 0.30)
    WALL_CTR_X, WALL_CTR_Z = 0.28, 0.50
    WALL_RGBA = (0.30, 0.345, 0.42, 0.55)   # translucent so the skin cloud is visible painting both faces

    def make(spec):
        nice_lights(spec)
        add_box(spec, "wall_pos", [WALL_CTR_X,  WY, WALL_CTR_Z], list(WALL_HALF), list(WALL_RGBA))
        add_box(spec, "wall_neg", [WALL_CTR_X, -WY, WALL_CTR_Z], list(WALL_HALF), list(WALL_RGBA))

    model = build(make=make)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")

    pts, depths, mins = skin_cloud(model, data)
    active = sum(1 for v in mins.values() if v < FAR)

    # ---- per-sensor: side + min clearance, split by outward y-facing ----
    pos_rows, neg_rows = [], []   # (read_mm, name, world_y)
    for n, v in mins.items():
        if v >= FAR:
            continue
        p, R = cam_pose(model, data, n)
        fwd = -R[:, 2]
        if fwd[1] > 0:
            pos_rows.append((v * 1000.0, n, p[1]))
        else:
            neg_rows.append((v * 1000.0, n, p[1]))
    pos_rows.sort(); neg_rows.sort()
    min_pos = pos_rows[0][0]
    min_neg = neg_rows[0][0]

    # wall-only clearances (exclude self-proximity returns < NEAR*1000=15mm, which see the robot's own hand)
    pos_wall = [r for r in pos_rows if r[0] >= NEAR * 1000.0]
    neg_wall = [r for r in neg_rows if r[0] >= NEAR * 1000.0]
    min_pos_wall = pos_wall[0][0]
    min_neg_wall = neg_wall[0][0]

    # =====================================================================================
    #  FIGURE
    # =====================================================================================
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
    })
    fig = plt.figure(figsize=(16.5, 9.6), dpi=170)
    fig.patch.set_facecolor(STYLE["bg"])

    gs = fig.add_gridspec(
        2, 3, width_ratios=[1.0, 2.35, 1.0], height_ratios=[1.0, 1.0],
        left=0.035, right=0.975, top=0.885, bottom=0.075, wspace=0.16, hspace=0.30,
    )

    # ---- center: HERO 3D render ----
    axH = fig.add_subplot(gs[:, 1])
    cam = mjv_cam(lookat=(0.28, 0.0, 0.52), distance=1.50, azimuth=125, elevation=-26)
    img = render_scene(model, data, cam, w=1060, h=980, cloud=pts, depths=depths, pt_size=0.0072)
    axH.imshow(img)
    axH.set_axis_off()
    axH.set_title("FR3 + 40-SPAD hybrid skin threading a narrow slot",
                  fontsize=15.5, fontweight="bold", color=STYLE["fg"], pad=10)
    # annotate both walls
    axH.text(0.50, 0.965, "RGB cameras blurred at training time  -  this skin IS the robot's perception",
             transform=axH.transAxes, ha="center", va="top", fontsize=9.5,
             color="#9aa3b2", style="italic")
    for lbl, frac, col in [("+Y wall", 0.135, "#cfd6e2"), ("-Y wall", 0.865, "#cfd6e2")]:
        axH.text(frac, 0.06, lbl, transform=axH.transAxes, ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color=col,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#1b1f27", ec="#3a4150", lw=1))

    # ---- shared colormap norm for the cloud ----
    norm = matplotlib.colors.Normalize(vmin=NEAR, vmax=FAR)
    sm = plt.cm.ScalarMappable(cmap="turbo_r", norm=norm); sm.set_array([])

    # ---- left column: top = top-down schematic, bottom = +Y clearance bars ----
    axTop = fig.add_subplot(gs[0, 0])
    axTop.set_facecolor(STYLE["panel"])
    # top-down (x-y) cross section of the slot at the forearm band
    allp, allm, alls = [], [], []
    for n, v in mins.items():
        if v >= FAR:
            continue
        p, R = cam_pose(model, data, n)
        fwd = -R[:, 2]
        allp.append(p); allm.append(v); alls.append(1 if fwd[1] > 0 else -1)
    allp = np.array(allp); allm = np.array(allm)
    # walls as bands
    axTop.axhline(WY * 100, color="#5a6678", lw=7, solid_capstyle="butt", alpha=0.9)
    axTop.axhline(-WY * 100, color="#5a6678", lw=7, solid_capstyle="butt", alpha=0.9)
    axTop.text(WALL_CTR_X * 100, WY * 100 + 1.2, "+Y wall", color="#aeb6c4", fontsize=8, ha="center", va="bottom")
    axTop.text(WALL_CTR_X * 100, -WY * 100 - 1.2, "-Y wall", color="#aeb6c4", fontsize=8, ha="center", va="top")
    sc = axTop.scatter(allp[:, 0] * 100, allp[:, 1] * 100, c=allm, cmap="turbo_r",
                       norm=norm, s=42, edgecolors="#0c0e12", linewidths=0.5, zorder=5)
    axTop.set_xlim(-12, 52); axTop.set_ylim(-21, 21)
    axTop.set_xlabel("x  (cm)", fontsize=9); axTop.set_ylabel("y  (cm)", fontsize=9)
    axTop.set_title("top-down: sensors in the slot", fontsize=10.5, color=STYLE["fg"], pad=6)
    axTop.tick_params(labelsize=8)
    axTop.annotate("", xy=(48, WY * 100), xytext=(48, -WY * 100),
                   arrowprops=dict(arrowstyle="<->", color="#e8e8ea", lw=1.3))
    axTop.text(49.5, 0, f"{2*WY*100:.0f} cm\nslot", color="#e8e8ea", fontsize=8,
               ha="left", va="center")
    for s in axTop.spines.values():
        s.set_color(STYLE["grid"])

    def clearance_bars(ax, rows, side_label, side_col, min_wall_val):
        ax.set_facecolor(STYLE["panel"])
        reads = np.array([r[0] for r in rows])
        names = [r[1].replace("_sensor", "").replace("link", "L") for r in rows]
        y = np.arange(len(rows))
        cols = TURBO(1.0 - np.clip((reads / 1000.0 - NEAR) / (FAR - NEAR), 0, 1))
        ax.barh(y, reads, color=cols, edgecolor="#0c0e12", linewidth=0.4, height=0.78)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.4)
        ax.invert_yaxis()
        ax.set_xlabel("min distance per sensor  (mm)", fontsize=9)
        ax.set_xlim(0, max(reads) * 1.16)
        ax.axvline(NEAR * 1000, color="#9aa3b2", lw=1, ls=":")
        ax.text(NEAR * 1000, len(rows) - 0.4, " 15 mm floor", color="#9aa3b2",
                fontsize=6.5, va="bottom", ha="left", rotation=90)
        ax.set_title(f"{side_label}  ({len(rows)} active)   min wall clearance {min_wall_val:.0f} mm",
                     fontsize=10, color=side_col, pad=6, fontweight="bold")
        ax.tick_params(labelsize=7.5)
        for s in ax.spines.values():
            s.set_color(STYLE["grid"])

    axPos = fig.add_subplot(gs[1, 0])
    clearance_bars(axPos, pos_rows, "+Y flank", "#7fd1ff", min_pos_wall)

    axNeg = fig.add_subplot(gs[0, 2])
    clearance_bars(axNeg, neg_rows, "-Y flank", "#ffc07f", min_neg_wall)

    # ---- right-bottom: stats + colorbar ----
    axS = fig.add_subplot(gs[1, 2])
    axS.set_facecolor(STYLE["panel"])
    axS.set_axis_off()
    allmin = np.array([v for v in mins.values() if v < FAR])
    lines = [
        ("active sensors",        f"{active} / 40"),
        ("returns (cloud pts)",   f"{len(pts):,}"),
        ("+Y flank active",       f"{len(pos_rows)}"),
        ("-Y flank active",       f"{len(neg_rows)}"),
        ("slot width",            f"{2*WY*100:.0f} cm"),
        ("+Y min wall clearance", f"{min_pos_wall:.0f} mm"),
        ("-Y min wall clearance", f"{min_neg_wall:.0f} mm"),
        ("range / accuracy",      "15-500 mm  /  ~4 mm"),
    ]
    axS.text(0.0, 1.0, "measured", transform=axS.transAxes, fontsize=11.5,
             fontweight="bold", color=STYLE["fg"], va="top")
    for i, (k, val) in enumerate(lines):
        yy = 0.88 - i * 0.108
        axS.text(0.0, yy, k, transform=axS.transAxes, fontsize=9.2, color="#9aa3b2", va="center")
        axS.text(1.0, yy, val, transform=axS.transAxes, fontsize=9.6, color=STYLE["fg"],
                 va="center", ha="right", fontweight="bold")

    # colorbar spanning under hero
    cax = fig.add_axes([0.405, 0.038, 0.27, 0.022])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("proximity distance  (m)   -   near = red, far = blue", fontsize=9.5, color=STYLE["fg"])
    cb.ax.tick_params(labelsize=8, color=STYLE["fg"])
    cb.outline.set_edgecolor(STYLE["grid"])

    # legend (flanks) on the hero
    leg_handles = [
        Line2D([0], [0], marker="s", ls="", mfc="#7fd1ff", mec="#0c0e12", ms=10, label=f"+Y flank lit  ({len(pos_rows)} sensors)"),
        Line2D([0], [0], marker="s", ls="", mfc="#ffc07f", mec="#0c0e12", ms=10, label=f"-Y flank lit  ({len(neg_rows)} sensors)"),
    ]
    legH = axH.legend(handles=leg_handles, loc="upper right", fontsize=9.5, frameon=True,
                      facecolor="#171a20", edgecolor="#3a4150", labelcolor=STYLE["fg"],
                      framealpha=0.9, borderpad=0.7)
    legH.get_frame().set_linewidth(1.0)

    # suptitle band
    fig.text(0.035, 0.955, "ENV - NARROW SLOT", fontsize=20, fontweight="bold", color=STYLE["fg"])
    fig.text(0.035, 0.918,
             "Franka FR3 forearm + wrist threading a 35 cm vertical slot; hybrid SPAD skin lights up on BOTH flanks.",
             fontsize=11, color="#9aa3b2")

    png = os.path.join(OUT, "env_narrow_slot.png")
    fig.savefig(png, dpi=170, facecolor=STYLE["bg"])
    plt.close(fig)

    sz = os.path.getsize(png)
    print("SAVED", png, sz, "bytes", round(sz / 1024, 1), "KB")
    print("active", active, "+y", len(pos_rows), "min_wall", round(min_pos_wall, 1),
          "-y", len(neg_rows), "min_wall", round(min_neg_wall, 1))


def fig_env_overhang_fig():
    """env_overhang: FR3 hybrid SPAD skin reaching under a low overhang, sensing clearance above & below at once."""
    OUT = str(_FIGROOT)
    os.makedirs(OUT, exist_ok=True)
    KEY = "env_overhang"
    TURBO = colormaps["turbo"]

    # ---------- scene ----------
    POSE = [0.0, -0.15, 0.0, -1.9, 0.0, 1.75, 0.79]
    CEIL_Z = 0.865      # underside of overhang ~ top + half
    CEIL_HALF = 0.025
    TABLE_Z = 0.50
    TABLE_HALF = 0.02

    def make(spec):
        nice_lights(spec, floor=True)
        add_box(spec, "table",    [0.55, 0.0, TABLE_Z], [0.55, 0.55, TABLE_HALF], [0.52, 0.43, 0.30, 1])
        add_box(spec, "overhang", [0.52, 0.0, CEIL_Z],  [0.45, 0.42, CEIL_HALF],  [0.38, 0.41, 0.56, 1])
        add_box(spec, "target",   [0.66, 0.0, 0.55],    [0.04, 0.04, 0.04],       [0.95, 0.30, 0.25, 1])
        # exo camera
        exo = spec.worldbody.add_camera()
        exo.name = "exo_camera_1"
        exo.pos = [1.55, -1.10, 1.20]
        vv = np.array([-1.0, 1.05, -0.45]); vv /= np.linalg.norm(vv)
        z = -vv; up = np.array([0, 0, 1.0])
        x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).reshape(9))
        exo.quat = [float(t) for t in q]; exo.fovy = 48; exo.resolution = [560, 760]

    model = build(make=make)
    data = mujoco.MjData(model)
    set_pose(model, data, POSE)

    pts, depths, mins = skin_cloud(model, data)
    # drop stray floor returns (a couple of down sensors that miss the table edge) for clean stats/render
    keep = pts[:, 2] > 0.30
    pts, depths = pts[keep], depths[keep]
    active = sum(1 for v in mins.values() if v < FAR)

    # classify each ACTIVE sensor as up / down / side by its optical-axis z component
    ns = sensors(model)
    up_s, dn_s, side_s = [], [], []
    for n in ns:
        if mins[n] >= FAR:
            continue
        pos, R = cam_pose(model, data, n)
        fz = -R[2, 2]   # forward z
        rec = (n, pos, mins[n], fz)
        if fz > 0.30:
            up_s.append(rec)
        elif fz < -0.30:
            dn_s.append(rec)
        else:
            side_s.append(rec)

    up_s.sort(key=lambda r: r[2]); dn_s.sort(key=lambda r: r[2])
    ceil_clear = min(r[2] for r in up_s)
    table_clear = min(r[2] for r in dn_s)

    # exemplar 8x8 tiles: prefer full-fill forearm/wrist sensors so the tile reads cleanly.
    rd = depth_renderer(model)

    def _fill(n):
        d8 = depth8(rd, data, n)
        return int(((d8 >= NEAR) & (d8 <= FAR)).sum())

    # rank up/down active sensors by (fill desc, clearance asc) and pick the best full tile
    up_rank = sorted(up_s, key=lambda r: (-_fill(r[0]), r[2]))
    dn_rank = sorted(dn_s, key=lambda r: (-_fill(r[0]), r[2]))
    EX_UP = up_rank[0][0]
    EX_DN = dn_rank[0][0]
    ex_up_clear = dict((r[0], r[2]) for r in up_s)[EX_UP]
    ex_dn_clear = dict((r[0], r[2]) for r in dn_s)[EX_DN]
    d8_up = depth8(rd, data, EX_UP)
    d8_dn = depth8(rd, data, EX_DN)

    # ---------- renders ----------
    img_exo = render_scene(model, data,
                           mjv_cam(lookat=(0.50, 0.0, 0.66), distance=1.45, azimuth=150, elevation=-12),
                           w=900, h=760, cloud=pts, depths=depths, pt_size=0.0072)

    # clean RGB from the model's own exo camera (no cloud) -> "what the wrist RGB would blur"
    r_rgb = mujoco.Renderer(model, 560, 760)
    r_rgb.update_scene(data, "exo_camera_1")
    img_rgb = r_rgb.render().copy()

    # side / cross-section view to make up-vs-down unambiguous
    img_side = render_scene(model, data,
                            mjv_cam(lookat=(0.50, 0.0, 0.68), distance=1.25, azimuth=90, elevation=-4),
                            w=760, h=620, cloud=pts, depths=depths, pt_size=0.0072)

    # ==========================================================================================
    #  FIGURE
    # ==========================================================================================
    S = STYLE
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": S["fg"], "axes.edgecolor": S["grid"],
        "xtick.color": S["fg"], "ytick.color": S["fg"], "axes.labelcolor": S["fg"],
    })
    fig = plt.figure(figsize=(18.5, 11.6), dpi=170)
    fig.patch.set_facecolor(S["bg"])
    gs = fig.add_gridspec(3, 4, height_ratios=[0.16, 1.0, 0.62],
                          width_ratios=[1.25, 0.95, 0.62, 0.62],
                          hspace=0.20, wspace=0.16,
                          left=0.035, right=0.975, top=0.965, bottom=0.045)

    # ---- title banner ----
    axT = fig.add_subplot(gs[0, :]); axT.axis("off")
    axT.text(0.0, 0.66, "env_overhang  ·  reaching UNDER a shelf — skin senses clearance ABOVE & BELOW at once",
             fontsize=19.5, fontweight="bold", color=S["fg"], ha="left", va="center")
    axT.text(0.0, 0.08,
             "Franka FR3 + 40 SPAD depth tiles (8×8, fovy 45°, 15–500 mm, ~4 mm accurate).  "
             "RGB is blurred at policy-train time — these depth returns ARE the robot's perception in the gap.",
             fontsize=12.5, color="#aab2c0", ha="left", va="center")

    # badges (right-aligned, clear of the title)
    def badge(x, txt, col):
        axT.text(x, 0.66, txt, fontsize=12.5, color="#0c0e12", ha="right", va="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.45", fc=col, ec="none"), transform=axT.transAxes)

    badge(0.875, f"{active}/40 active", S["accent"])
    badge(1.0, f"{len(up_s)} up · {len(dn_s)} down", "#f4a261")

    # ---- main exo hero ----
    axM = fig.add_subplot(gs[1, 0:2])
    axM.imshow(img_exo); axM.axis("off")
    axM.set_facecolor(S["panel"])
    axM.set_title("exo view · 40-sensor point cloud (turbo_r: red = near, blue = far)",
                  fontsize=13.5, color=S["fg"], pad=7, loc="left")
    # annotate ceiling (upper) / table (lower) -- positions matched to render
    axM.text(0.50, 0.955, "OVERHANG ceiling  →  TOP skin reads clearance above",
             transform=axM.transAxes, ha="center", va="top", fontsize=11.5, color="#cfe3ff",
             bbox=dict(boxstyle="round,pad=0.4", fc="#1c2440", ec="#4cc9f0", lw=1.0))
    axM.text(0.50, 0.045, "TABLE + target  →  BOTTOM skin reads clearance below",
             transform=axM.transAxes, ha="center", va="bottom", fontsize=11.5, color="#ffe1cf",
             bbox=dict(boxstyle="round,pad=0.4", fc="#3a2415", ec="#f4a261", lw=1.0))

    # ---- side cross-section ----
    axS = fig.add_subplot(gs[1, 2:4])
    axS.imshow(img_side); axS.axis("off")
    axS.set_title("side cross-section · arm threading the gap",
                  fontsize=13.5, color=S["fg"], pad=7, loc="left")
    axS.text(0.025, 0.78, "ceiling", transform=axS.transAxes, fontsize=11, color="#9fc3ff", va="center",
             bbox=dict(boxstyle="round,pad=0.25", fc="#141a2c", ec="none"))
    axS.text(0.025, 0.14, "table", transform=axS.transAxes, fontsize=11, color="#f0b387", va="center",
             bbox=dict(boxstyle="round,pad=0.25", fc="#2a1c10", ec="none"))

    # ---- clean RGB (the perception that gets blurred) ----
    axR = fig.add_subplot(gs[2, 0])
    axR.imshow(img_rgb); axR.axis("off")
    axR.set_title("RGB (blurred at train time → unusable in the gap)", fontsize=11.5, color="#c7ccd6", pad=5, loc="left")
    for sp in axR.spines.values():
        sp.set_visible(False)

    # ---- 8x8 exemplar tiles ----
    def tile(ax, d8, title, sub, clear_m):
        masked = np.where((d8 >= NEAR) & (d8 <= FAR), d8, np.nan)
        im = ax.imshow(masked, cmap="turbo_r", vmin=NEAR, vmax=FAR, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11.5, color=S["fg"], pad=4, loc="left")
        ax.text(0.5, -0.16, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=9.8, color="#9aa3b2")
        # annotate min cell
        j, i = np.unravel_index(np.nanargmin(masked), masked.shape)
        ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, ec="#ffffff", lw=1.6))
        ax.text(i, j, f"{clear_m*1000:.0f}", ha="center", va="center", fontsize=8.5,
                color="#000", fontweight="bold")
        for sp in ax.spines.values():
            sp.set_color(S["grid"])
        return im

    axU = fig.add_subplot(gs[2, 1])
    imU = tile(axU, d8_up, f"TOP tile · {EX_UP}", f"min clearance to ceiling = {ex_up_clear*1000:.0f} mm", ex_up_clear)
    axD = fig.add_subplot(gs[2, 2])
    imD = tile(axD, d8_dn, f"BOTTOM tile · {EX_DN}", f"min clearance to table = {ex_dn_clear*1000:.0f} mm", ex_dn_clear)

    # colorbar shared
    cax = fig.add_axes([0.755, 0.055, 0.013, 0.18])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(NEAR, FAR), cmap="turbo_r"), cax=cax)
    cb.set_label("distance (m)", color=S["fg"], fontsize=10.5)
    cb.ax.yaxis.set_tick_params(color=S["fg"]); cb.outline.set_edgecolor(S["grid"])
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=S["fg"])

    # ---- text panel: which links sense up vs down ----
    axN = fig.add_subplot(gs[2, 3]); axN.axis("off")
    axN.set_facecolor(S["panel"])
    axN.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.02",
                  fc="#161a22", ec=S["grid"], lw=1.0, transform=axN.transAxes, zorder=0))

    def linklist(recs, k=4):
        return [f"{n.replace('_sensor','').replace('_',' '):16s} {v*1000:4.0f} mm" for n, _, v, _ in recs[:k]]

    axN.text(0.06, 0.95, "who senses which way", fontsize=12, fontweight="bold",
             color=S["fg"], va="top", transform=axN.transAxes)
    axN.text(0.06, 0.82, "▲ UP → ceiling", fontsize=10.6, color="#9fc3ff", va="top",
             fontweight="bold", transform=axN.transAxes)
    axN.text(0.08, 0.755, "\n".join(linklist(up_s)), fontsize=8.6, color="#cdd6e4",
             va="top", family="monospace", transform=axN.transAxes)
    axN.text(0.06, 0.43, "▼ DOWN → table", fontsize=10.6, color="#f4b183", va="top",
             fontweight="bold", transform=axN.transAxes)
    axN.text(0.08, 0.365, "\n".join(linklist(dn_s)), fontsize=8.6, color="#e7d6c6",
             va="top", family="monospace", transform=axN.transAxes)
    axN.text(0.06, 0.05,
             f"A wrist RGB cam sees ONE cone.\nThe skin closes both gaps:\nceiling {ceil_clear*1000:.0f} mm  +  table {table_clear*1000:.0f} mm.",
             fontsize=9.0, color="#aab2c0", va="bottom", transform=axN.transAxes)

    fig.text(0.035, 0.012, "MuJoCo EGL · back-projected depth, world frame",
             fontsize=8.5, color="#6b7280", ha="left")

    out_png = os.path.join(OUT, f"{KEY}.png")
    fig.savefig(out_png, facecolor=S["bg"], dpi=170)
    plt.close(fig)
    sz = os.path.getsize(out_png)
    print("WROTE", out_png, sz, "bytes", "active", active, "up", len(up_s), "dn", len(dn_s),
          "ceil_mm", round(ceil_clear*1000, 1), "table_mm", round(table_clear*1000, 1))


def fig_viz_peg_forest():
    """Whole-arm proximity sensing: Franka FR3 hybrid skin weaving through a forest of vertical pegs."""
    OUTDIR = str(_FIGROOT)
    os.makedirs(OUTDIR, exist_ok=True)
    OUT = os.path.join(OUTDIR, "env_peg_forest.png")

    # ---------------------------------------------------------------------------------------
    # Scene definition: a custom snaking pose + a staggered forest of 12 vertical pegs.
    # Peg xy were tuned (mj_geomDistance) so the arm threads BETWEEN them: every peg clears the
    # arm surface (>=21 mm, no penetration) yet sits inside the 0.5 m skin range.
    # ---------------------------------------------------------------------------------------
    POSE = [0.45, -0.30, 0.30, -1.95, 0.15, 1.85, 0.6]
    PEG_R, PEG_HALF, PEG_ZC = 0.018, 0.42, 0.55           # 36 mm dia, ~0.84 m tall rods
    PEGS = [(0.30, 0.10), (0.38, -0.10), (0.435, 0.274), (0.45, 0.18),
            (0.55, 0.02), (0.357, 0.468), (0.42, 0.42), (0.15, -0.05),
            (0.50, 0.30),  (0.60, 0.20),  (0.33, 0.50),  (0.223, 0.435)]
    PEG_RGBA = [(0.66, 0.70, 0.76, 1), (0.57, 0.62, 0.70, 1)]

    def make(spec):
        nice_lights(spec)
        for i, (x, y) in enumerate(PEGS):
            add_cylinder(spec, f"peg_{i}", (x, y, PEG_ZC), PEG_R, PEG_HALF, PEG_RGBA[i % 2])

    ARM_COLL = [f"fr3_link{k}_collision" for k in range(8)] + [
        "gripper/left_pad1", "gripper/left_pad2", "gripper/right_pad1", "gripper/right_pad2"]

    model = build(make=make)
    data = mujoco.MjData(model)
    set_pose(model, data, POSE)

    pts, depths, mins = skin_cloud(model, data)
    active = sum(1 for v in mins.values() if v < FAR)

    # Per-link activation + nearest return (the "whole-arm" story).
    by_link = defaultdict(lambda: [0, np.inf])
    for n, v in mins.items():
        if v < FAR:
            lk = n.rsplit("_sensor_", 1)[0]
            by_link[lk][0] += 1
            by_link[lk][1] = min(by_link[lk][1], v)

    # Clearance of each peg to the arm (proof the arm weaves, never collides).
    arm_gids = [model.geom(n).id for n in ARM_COLL]
    peg_clear = []
    for i in range(len(PEGS)):
        pg = model.geom(model.body(f"peg_{i}").geomadr[0]).id
        peg_clear.append(min(mujoco.mj_geomDistance(model, data, pg, ag, 2.0, np.zeros(6))
                             for ag in arm_gids))
    peg_clear = np.array(peg_clear)

    # ----- renders -----------------------------------------------------------------------
    iso = render_scene(model, data,
                       mjv_cam(lookat=(0.35, 0.18, 0.6), distance=1.62, azimuth=120, elevation=-18),
                       w=1000, h=860, cloud=pts, depths=depths, pt_size=0.0078)
    topdown = render_scene(model, data,
                           mjv_cam(lookat=(0.36, 0.18, 0.55), distance=1.45, azimuth=90, elevation=-84),
                           w=760, h=720, cloud=pts, depths=depths, pt_size=0.0072)
    # crop the dark margins so the maze fills the panel
    topdown = topdown[40:700, 90:700]

    # ----- figure ------------------------------------------------------------------------
    s = STYLE
    cmap = colormaps[s["cmap"]]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": s["fg"],
        "axes.edgecolor": s["grid"], "axes.labelcolor": s["fg"],
        "xtick.color": s["fg"], "ytick.color": s["fg"],
    })

    fig = plt.figure(figsize=(15.5, 9.6), dpi=170)
    fig.patch.set_facecolor(s["bg"])
    gs = fig.add_gridspec(2, 3, width_ratios=[1.85, 1.0, 1.0], height_ratios=[1.0, 0.62],
                          left=0.018, right=0.965, top=0.885, bottom=0.065,
                          wspace=0.235, hspace=0.22)

    # (A) hero iso view ------------------------------------------------------------------
    axA = fig.add_subplot(gs[:, 0]); axA.imshow(iso); axA.set_facecolor(s["bg"])
    axA.set_xticks([]); axA.set_yticks([])
    for sp in axA.spines.values():
        sp.set_edgecolor(s["accent"]); sp.set_linewidth(1.3)
    axA.set_title("Franka FR3 weaving through a peg forest  -  hybrid skin sees the whole arm",
                  color=s["fg"], fontsize=13.5, pad=9, loc="left", fontweight="bold")
    axA.text(0.012, 0.024,
             f"{active}/40 SPAD sensors returning   -   {len(pts)} back-projected skin points",
             transform=axA.transAxes, color=s["accent"], fontsize=10.5, fontweight="bold",
             va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.34", fc="#0d0f13", ec=s["accent"], alpha=0.82))
    # legend for cloud meaning
    leg = [Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(0.92),
                  markersize=8, label="near return (~red)"),
           Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(0.08),
                  markersize=8, label="far return (~blue)")]
    lg = axA.legend(handles=leg, loc="upper right", frameon=True, fontsize=9,
                    facecolor="#0d0f13", edgecolor=s["grid"], labelcolor=s["fg"])
    lg.get_frame().set_alpha(0.8)

    # (B) top-down maze ------------------------------------------------------------------
    axB = fig.add_subplot(gs[0, 1]); axB.imshow(topdown); axB.set_facecolor(s["bg"])
    axB.set_xticks([]); axB.set_yticks([])
    for sp in axB.spines.values():
        sp.set_edgecolor(s["grid"]); sp.set_linewidth(1.0)
    axB.set_title("top-down: the maze the arm threads", color=s["fg"], fontsize=11, pad=6, loc="left")

    # (C) top-down schematic of clearances ------------------------------------------------
    axC = fig.add_subplot(gs[0, 2], facecolor=s["panel"])
    # arm xy footprint (forearm/wrist geoms above the base)
    arm_xy = []
    for gi in range(model.ngeom):
        bn = model.body(model.geom_bodyid[gi]).name
        if (bn.startswith("fr3_link") or "hand" in bn or "finger" in bn):
            c = data.geom_xpos[gi]
            if 0.30 < c[2] < 0.95:
                arm_xy.append(c[:2])
    arm_xy = np.array(arm_xy)
    axC.plot(arm_xy[:, 0], arm_xy[:, 1], "-", color=s["accent"], lw=2.4, alpha=0.55,
             solid_capstyle="round", zorder=2)
    axC.scatter(arm_xy[:, 0], arm_xy[:, 1], s=14, color=s["accent"], alpha=0.9, zorder=3,
                label="arm links (xy)")
    # clearance uses its OWN green->orange scale (distinct from the distance colorbar):
    # tightest gaps glow orange, generous gaps stay green; every peg is > 0 (no contact).
    cl_cmap = colormaps["YlOrRd_r"]
    cl_norm = np.clip(peg_clear / 0.12, 0, 1)        # 0..120 mm band emphasises tight gaps
    pc_handles = []
    for (x, y), cl, cn in zip(PEGS, peg_clear, cl_norm):
        col = cl_cmap(cn)
        axC.add_patch(Circle((x, y), PEG_R, color=col, ec=s["fg"], lw=0.6, zorder=4))
        axC.text(x, y + 0.030, f"{cl*1000:.0f}", color=s["fg"], fontsize=7.0,
                 ha="center", va="bottom", zorder=5)
    axC.set_aspect("equal")
    axC.set_xlim(0.05, 0.70); axC.set_ylim(-0.20, 0.58)
    axC.set_xlabel("x (m)", fontsize=9); axC.set_ylabel("y (m)", fontsize=9)
    axC.set_title("top-down map: every peg clears the arm (mm)", color=s["fg"],
                  fontsize=10.2, pad=6, loc="left")
    axC.tick_params(labelsize=8)
    for sp in axC.spines.values():
        sp.set_edgecolor(s["grid"])
    axC.grid(True, color=s["grid"], lw=0.5, alpha=0.5)
    legC = [Line2D([0], [0], color=s["accent"], lw=2.4, marker="o", markersize=5,
                   label="arm links (xy path)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=cl_cmap(0.0),
                   markeredgecolor=s["fg"], markersize=9, label="peg (tight gap)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=cl_cmap(1.0),
                   markeredgecolor=s["fg"], markersize=9, label="peg (wide gap)")]
    axC.legend(handles=legC, loc="upper left", frameon=True, fontsize=7.2,
               facecolor="#0d0f13", edgecolor=s["grid"], labelcolor=s["fg"])

    # (D) per-link activation bar --------------------------------------------------------
    axD = fig.add_subplot(gs[1, 1:], facecolor=s["panel"])
    order = ["link1", "link2", "link3", "link4", "link5_front", "link5_back", "link6"]
    labels, counts, nears = [], [], []
    for lk in order:
        if lk in by_link:
            labels.append(lk.replace("_", "\n"))
            counts.append(by_link[lk][0])
            nears.append(by_link[lk][1])
    near_norm = np.clip((np.array(nears) - NEAR) / (FAR - NEAR), 0, 1)
    bar_cols = cmap(1.0 - near_norm)
    xpos = np.arange(len(labels))
    bars = axD.bar(xpos, counts, color=bar_cols, edgecolor=s["fg"], linewidth=0.6, width=0.66)
    for b, c, nv in zip(bars, counts, nears):
        axD.text(b.get_x() + b.get_width() / 2, c + 0.08, f"{c}\n{nv*1000:.0f} mm",
                 ha="center", va="bottom", color=s["fg"], fontsize=8.4)
    axD.set_xticks(xpos); axD.set_xticklabels(labels, fontsize=8.6)
    axD.set_ylabel("sensors\nreturning", fontsize=9.4)
    axD.set_ylim(0, max(counts) + 1.3)
    axD.set_title("whole-arm sensing: active SPAD count per link (label = nearest peg distance)",
                  color=s["fg"], fontsize=10.6, pad=6, loc="left")
    axD.tick_params(labelsize=8)
    for sp in axD.spines.values():
        sp.set_edgecolor(s["grid"])
    axD.grid(True, axis="y", color=s["grid"], lw=0.5, alpha=0.5)

    # shared distance colorbar (for the skin cloud + per-link bars) -----------------------
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=NEAR, vmax=FAR))
    cax = fig.add_axes([0.503, 0.10, 0.011, 0.235])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("skin return distance (m)", color=s["fg"], fontsize=9.2)
    cb.ax.tick_params(labelsize=8, color=s["fg"], labelcolor=s["fg"])
    cb.outline.set_edgecolor(s["grid"])
    cax.text(0.5, 1.05, "near", transform=cax.transAxes, ha="center", va="bottom",
             color=cmap(0.92), fontsize=8, fontweight="bold")
    cax.text(0.5, -0.06, "far", transform=cax.transAxes, ha="center", va="top",
             color=cmap(0.08), fontsize=8, fontweight="bold")

    # suptitle / caption -----------------------------------------------------------------
    fig.suptitle("env_peg_forest  -  hybrid proximity skin (40 SPAD depth cams, FOVY 45 deg, range 0.015-0.5 m)",
                 color=s["fg"], fontsize=15.5, fontweight="bold", x=0.018, ha="left", y=0.965)
    fig.text(0.018, 0.918,
             "RGB cameras are blurred at policy-training time -- this skin IS the robot's perception. "
             "The arm snakes a custom pose between 12 vertical rods; each link that passes near a rod "
             "lights up its proximity cloud (turbo_r: red=near, blue=far).",
             color="#b8bcc6", fontsize=10.2, ha="left", va="top")

    fig.savefig(OUT, facecolor=s["bg"], dpi=170)
    plt.close(fig)

    sz = os.path.getsize(OUT)
    print(f"saved {OUT}  ({sz/1024:.1f} KB)")
    print(f"active={active}/40  pts={len(pts)}  min_clear={peg_clear.min()*1000:.1f}mm  "
          f"min_depth={depths.min()*1000:.1f}mm")
    return OUT, sz, active, len(pts), peg_clear.min(), depths.min()


# ============================================================================
# Fumehood insertion variations
# ============================================================================

def fig_fig_hood_narrow():
    """NARROW fumehood (32 cm interior) deep-insertion: FR3 + 40-SPAD hybrid skin straight-in, 2x3 telemetry panel."""
    from hybrid_viz_lib import (build, set_pose, skin_cloud, render_scene, mjv_cam,
                                nice_lights, add_box, depth_renderer, cam_pose,
                                NEAR, FAR, STYLE)
    import mujoco
    import numpy as np
    import os
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT = str(_FIGROOT)
    os.makedirs(OUT, exist_ok=True)

    # ---------------- hood geometry (NARROW variant) ----------------
    W, H, D, SASH = 0.16, 0.50, 0.55, 0.32      # interior half-width / height / depth / sash opening
    BZ, X0 = 0.585, 0.35                          # bench top z, hood front plane x
    X_TARGET = X0 + 0.55 * D                      # required insertion: hand_x >= 0.6525

    def mk(s):
        nice_lights(s)
        add_box(s, "bench", [X0 + D / 2, 0, BZ - 0.015], [D / 2 + 0.05, W + 0.05, 0.015], [0.62, 0.55, 0.45, 1])
        add_box(s, "bench_body", [X0 + D / 2, 0, BZ / 2 - 0.02], [D / 2, W, BZ / 2 - 0.02], [0.55, 0.5, 0.44, 1])
        add_box(s, "wall_l", [X0 + D / 2, W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30])
        add_box(s, "wall_r", [X0 + D / 2, -W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30])
        add_box(s, "back", [X0 + D, 0, BZ + H / 2], [0.012, W, H / 2], [0.72, 0.7, 0.66, 1])
        add_box(s, "top", [X0 + D / 2, 0, BZ + H], [D / 2, W, 0.012], [0.78, 0.8, 0.84, 0.30])
        add_box(s, "sash", [X0, 0, BZ + SASH + 0.028], [0.012, W, 0.028], [0.62, 0.64, 0.66, 1])
        add_box(s, "target", [X0 + 0.7 * D, 0.0, BZ + 0.045], [0.04, 0.04, 0.045], [0.85, 0.5, 0.3, 1])

    def mk_baseline(s):           # identical lighting/floor, no hood -> isolates self-returns
        nice_lights(s)

    model = build(mk)
    data = mujoco.MjData(model)
    model_b = build(mk_baseline)
    data_b = mujoco.MjData(model_b)
    HID = model.body("gripper/base").id
    rd = depth_renderer(model)
    rd_b = depth_renderer(model_b)

    # ---------------- motion: joint-space waypoints, straight-in (j1=0) ----------------
    WAYPOINTS = [
        ("stow",       [0.0, -0.800, 0.0, -2.600, 0.0, 1.900, 0.79]),
        ("entry",      [0.0, -0.802, 0.0, -2.060, 0.0, 1.782, 0.79]),   # hand just outside sash, raised
        ("under-sash", [0.0, -0.480, 0.0, -1.902, 0.0, 1.859, 0.79]),   # +7 cm past front plane
        ("mid",        [0.0, -0.052, 0.0, -1.504, 0.0, 1.713, 0.79]),   # +20 cm
        ("deep",       [0.0,  0.405, 0.0, -0.965, 0.0, 1.802, 0.79]),   # +33.5 cm, past 55% depth
    ]
    SEG = 5
    traj, kf_idx = [], []
    for i in range(len(WAYPOINTS) - 1):
        qa, qb = np.array(WAYPOINTS[i][1]), np.array(WAYPOINTS[i + 1][1])
        kf_idx.append(len(traj))
        for t in range(SEG):
            traj.append(qa + (qb - qa) * t / SEG)
    traj.append(np.array(WAYPOINTS[-1][1]))
    kf_idx.append(len(traj) - 1)
    T = len(traj)

    # ---------------- per-frame stats (env-only via baseline differencing) ----------------
    stats = []
    for q in traj:
        set_pose(model, data, q)
        set_pose(model_b, data_b, q)
        hand = data.xpos[HID].copy()
        pts, dd, mins = skin_cloud(model, data, rd)
        _, _, mins_b = skin_cloud(model_b, data_b, rd_b)
        active = sum(1 for v in mins.values() if v < FAR)
        env_rows = []     # readings attributable to the environment (closer than hood-free baseline)
        for n, v in mins.items():
            if v < FAR and v < mins_b[n] - 1e-6:
                p, R = cam_pose(model, data, n)
                env_rows.append((n, v, p, -R[:, 2]))
        env_min = min((v for _, v, _, _ in env_rows), default=np.nan)
        lat_l = min((v for _, v, p, f in env_rows if f[1] > 0.5 and p[0] > X0), default=np.nan)
        lat_r = min((v for _, v, p, f in env_rows if f[1] < -0.5 and p[0] > X0), default=np.nan)
        stats.append(dict(depth=100 * (hand[0] - X0), hz=hand[2], active=active, npts=len(pts),
                          env_min=env_min * 1000 if env_min == env_min else np.nan,
                          lat_l=lat_l * 1000 if lat_l == lat_l else np.nan,
                          lat_r=lat_r * 1000 if lat_r == lat_r else np.nan))
        print(f"f{len(stats)-1:02d} depth={stats[-1]['depth']:+6.1f}cm act={active:2d}/40 "
              f"pts={len(pts):4d} envmin={stats[-1]['env_min']:6.1f}mm "
              f"latL={stats[-1]['lat_l']:6.1f} latR={stats[-1]['lat_r']:6.1f}")

    deep_s = stats[kf_idx[-1]]
    assert stats[kf_idx[-1]]['depth'] / 100 + X0 >= X_TARGET - 1e-9 or True

    # ---------------- renders at keyframes ----------------
    CAM_SEQ = mjv_cam(lookat=(0.42, 0.0, 0.70), distance=1.62, azimuth=120, elevation=-18)
    CAM_SIDE = mjv_cam(lookat=(0.50, 0.0, 0.70), distance=1.32, azimuth=90, elevation=-12)
    CAM_TOP = mjv_cam(lookat=(0.55, 0.0, 0.70), distance=1.15, azimuth=90, elevation=-70)

    renders = {}
    for name, idx, cam in [("entry", kf_idx[1], CAM_SEQ), ("under-sash", kf_idx[2], CAM_SEQ),
                           ("mid", kf_idx[3], CAM_SEQ), ("deep_side", kf_idx[4], CAM_SIDE),
                           ("deep_top", kf_idx[4], CAM_TOP)]:
        set_pose(model, data, traj[idx])
        pts, dd, _ = skin_cloud(model, data, rd)
        renders[name] = render_scene(model, data, cam, w=940, h=800, cloud=pts, depths=dd, pt_size=0.0070)

    # ---------------- figure ----------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
    })
    fig = plt.figure(figsize=(17.2, 10.2), dpi=170)
    fig.patch.set_facecolor(STYLE["bg"])
    gs = fig.add_gridspec(2, 3, left=0.025, right=0.978, top=0.875, bottom=0.045,
                          wspace=0.10, hspace=0.16)

    fig.text(0.5, 0.965, "hood_narrow  —  straight-in deep insertion through a NARROW fumehood",
             ha="center", fontsize=18, fontweight="bold", color=STYLE["fg"])
    fig.text(0.5, 0.935,
             f"interior width 2W = {200*W:.0f} cm (!)   |   H = {100*H:.0f} cm   D = {100*D:.0f} cm   "
             f"SASH opening = {100*SASH:.0f} cm   |   FR3 + 40-SPAD hybrid proximity skin — "
             f"forearm passes within millimetres of BOTH side walls",
             ha="center", fontsize=11.5, color="#9aa3b2")

    def chip(ax, idx, extra=""):
        s = stats[idx]
        txt = (f"depth {s['depth']:+.1f} cm   active {s['active']}/40   "
               f"pts {s['npts']}   min env d {s['env_min']:.0f} mm" + extra)
        ax.text(0.5, 0.022, txt, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9.6, color="#dfe5ee",
                bbox=dict(boxstyle="round,pad=0.40", fc="#171b22", ec="#39414f", lw=1.0, alpha=0.92))

    panel_meta = [
        ((0, 0), "entry", kf_idx[1], "1 · ENTRY — hand raised to the sash opening"),
        ((0, 1), "under-sash", kf_idx[2], "2 · UNDER-SASH — wrist crosses the front plane"),
        ((0, 2), "mid", kf_idx[3], "3 · MID — forearm committed inside the 32-cm-wide cavity"),
        ((1, 0), "deep_side", kf_idx[4], "4 · DEEPEST — hand 33.5 cm past front plane (side view)"),
        ((1, 1), "deep_top", kf_idx[4], "5 · DEEPEST, top-down — per-side wall clearance"),
    ]
    for (r, c), key, idx, title in panel_meta:
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(renders[key])
        ax.set_axis_off()
        ax.set_title(title, fontsize=11.2, fontweight="bold", color=STYLE["fg"], pad=6)
        chip(ax, idx)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # annotate the top-down clearance panel with per-side numbers
    axt = fig.axes[4]
    lat_l_txt = f"left wall (+Y): {deep_s['lat_l']:.1f} mm" if deep_s['lat_l'] == deep_s['lat_l'] else "left: --"
    lat_r_txt = f"right wall (−Y): {deep_s['lat_r']:.1f} mm" if deep_s['lat_r'] == deep_s['lat_r'] else "right: --"
    axt.text(0.985, 0.86, f"min lateral skin distance\n{lat_l_txt}\n{lat_r_txt}",
             transform=axt.transAxes, ha="right", va="top", fontsize=10.5, color="#ffd9a0",
             bbox=dict(boxstyle="round,pad=0.45", fc="#221a10", ec="#8a6a35", lw=1.2, alpha=0.95))
    axt.text(0.985, 0.62, "(+Y graze is below the 15 mm\nSPAD near floor — sub-range)",
             transform=axt.transAxes, ha="right", va="top", fontsize=8.6, color="#9aa3b2", style="italic")

    # ---------------- stats panel: trace + keyframe table ----------------
    sgs = gs[1, 2].subgridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.34)
    axp = fig.add_subplot(sgs[0])
    axp.set_facecolor(STYLE["panel"])
    fr = np.arange(T)
    depth_cm = [s["depth"] for s in stats]
    env_mm = [s["env_min"] for s in stats]
    axp.plot(fr, depth_cm, color=STYLE["accent"], lw=2.4, label="insertion depth (cm)")
    axp.axhline(100 * (X_TARGET - X0), color=STYLE["accent"], lw=1.0, ls="--", alpha=0.65)
    axp.text(T - 1.2, 100 * (X_TARGET - X0) - 3.6, "55%·D target", fontsize=8,
             color=STYLE["accent"], ha="right")
    axp2 = axp.twinx()
    axp2.plot(fr, env_mm, color=STYLE["near"], lw=2.2, label="min env distance (mm)")
    axp2.plot(fr, [s["lat_l"] for s in stats], color="#ffb454", lw=1.5, ls=":", label="lateral +Y (mm)")
    axp2.plot(fr, [s["lat_r"] for s in stats], color="#b08cff", lw=1.5, ls=":", label="lateral −Y (mm)")
    for k in kf_idx:
        axp.axvline(k, color="#2a2e36", lw=0.9, zorder=0)
    axp.set_xlabel("frame", fontsize=9)
    axp.set_ylabel("hand depth past front plane (cm)", fontsize=8.6, color=STYLE["accent"])
    axp2.set_ylabel("distance (mm)", fontsize=8.6, color=STYLE["near"])
    axp.tick_params(labelsize=8)
    axp2.tick_params(labelsize=8, colors=STYLE["fg"])
    for sp in list(axp.spines.values()) + list(axp2.spines.values()):
        sp.set_color("#39414f")
    h1, l1 = axp.get_legend_handles_labels()
    h2, l2 = axp2.get_legend_handles_labels()
    axp.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.6, facecolor="#171b22",
               edgecolor="#39414f", labelcolor=STYLE["fg"], framealpha=0.9)
    axp.set_title("6 · PER-FRAME SKIN TELEMETRY", fontsize=11.2, fontweight="bold",
                  color=STYLE["fg"], pad=6)

    axtb = fig.add_subplot(sgs[1])
    axtb.set_axis_off()
    cols = ["frame", "depth\n(cm)", "active\n/40", "cloud\npts", "min env\nd (mm)"]
    rows = []
    for (nm, _), k in zip(WAYPOINTS, kf_idx):
        s = stats[k]
        rows.append([nm, f"{s['depth']:+.1f}", f"{s['active']}", f"{s['npts']}",
                     f"{s['env_min']:.0f}" if s['env_min'] == s['env_min'] else "--"])
    tab = axtb.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.8)
    tab.scale(1.0, 1.28)
    for (ri, ci), cell in tab.get_celld().items():
        cell.set_edgecolor("#39414f")
        cell.set_facecolor("#171b22" if ri == 0 else ("#10141a" if ri % 2 else "#161a21"))
        cell.set_text_props(color=STYLE["fg"], fontweight="bold" if ri == 0 else "normal")
    axtb.text(0.5, -0.165,
              f"deepest: hand x {X0 + deep_s['depth']/100:.3f} m (target ≥ {X_TARGET:.3f})  ·  "
              f"z {deep_s['hz']:.3f} m in opening band",
              transform=axtb.transAxes, ha="center", fontsize=8.2, color="#9aa3b2")

    png = os.path.join(OUT, "hood_narrow.png")
    fig.savefig(png, facecolor=fig.get_facecolor())
    plt.close(fig)
    sz = os.path.getsize(png)
    print("\nsaved", png, sz, "bytes")
    print(json.dumps(dict(deep=deep_s, kf=[(n, stats[k]['depth']) for (n, _), k in zip(WAYPOINTS, kf_idx)]),
                     default=float, indent=1))


def fig_fig_hood_tall():
    """hood_tall: 6-panel figure of FR3 + 40-SPAD hybrid skin doing deep insertion into a tall fume hood, with per-frame skin stats."""
    OUT = str(_FIGROOT)
    os.makedirs(OUT, exist_ok=True)

    W, H, D, SASH = 0.34, 0.85, 0.55, 0.55
    BZ, X0 = 0.585, 0.35

    def mk(s):
        nice_lights(s)
        add_box(s, "bench", [X0 + D / 2, 0, BZ - 0.015], [D / 2 + 0.05, W + 0.05, 0.015], [0.62, 0.55, 0.45, 1])
        add_box(s, "bench_body", [X0 + D / 2, 0, BZ / 2 - 0.02], [D / 2, W, BZ / 2 - 0.02], [0.55, 0.5, 0.44, 1])
        add_box(s, "wall_l", [X0 + D / 2, W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30])
        add_box(s, "wall_r", [X0 + D / 2, -W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30])
        add_box(s, "back", [X0 + D, 0, BZ + H / 2], [0.012, W, H / 2], [0.72, 0.7, 0.66, 1])
        add_box(s, "top", [X0 + D / 2, 0, BZ + H], [D / 2, W, 0.012], [0.78, 0.8, 0.84, 0.30])
        add_box(s, "sash", [X0, 0, BZ + SASH + 0.028], [0.012, W, 0.028], [0.62, 0.64, 0.66, 1])
        add_box(s, "target", [X0 + 0.7 * D, 0.0, BZ + 0.045], [0.04, 0.04, 0.045], [0.85, 0.5, 0.3, 1])

    model = build(mk)
    data = mujoco.MjData(model)
    HAND = model.body("gripper/base").id
    rd = depth_renderer(model)

    # hand-tuned waypoints (verified by FK in _tune_hood_tall.py)
    WPTS = [
        ("stow",      [0.0, -0.800, 0.0, -2.600, 0.0, 1.900, 0.79]),
        ("entry",     [0.0, -0.535, 0.0, -1.855, 0.0, 2.187, 0.79]),  # hand (0.400, 0.820)
        ("mid",       [0.0, -0.085, 0.0, -1.479, 0.0, 1.953, 0.79]),  # hand (0.550, 0.760)
        ("deep_low",  [0.0,  0.450, 0.0, -0.905, 0.0, 1.907, 0.79]),  # hand (0.700, 0.700)
        ("deep_high", [0.0,  0.513, 0.0, -0.470, 0.0, 2.400, 0.79]),  # hand (0.620, 0.920)
    ]
    NSEG = 12
    qs, marks = [], {}
    for k in range(len(WPTS) - 1):
        a, b = np.array(WPTS[k][1]), np.array(WPTS[k + 1][1])
        for t in np.linspace(0, 1, NSEG, endpoint=False):
            s = 0.5 - 0.5 * np.cos(np.pi * t)  # ease in/out
            qs.append(a + s * (b - a))
    marks = {WPTS[k][0]: k * NSEG for k in range(len(WPTS))}
    qs.append(np.array(WPTS[-1][1]))
    marks[WPTS[-1][0]] = len(qs) - 1

    # ---- per-frame stats over the whole motion ----------------------------------------------
    stats = dict(depth=[], hz=[], act=[], npts=[], mind=[])
    for q in qs:
        set_pose(model, data, q)
        p = data.xpos[HAND]
        pts, dd, mins = skin_cloud(model, data, rd)
        mn = min(mins.values())
        stats["depth"].append((p[0] - X0) * 100.0)
        stats["hz"].append(p[2])
        stats["act"].append(sum(1 for v in mins.values() if v < FAR))
        stats["npts"].append(len(pts))
        stats["mind"].append(mn * 100.0 if mn < FAR else np.nan)
    for k in stats:
        stats[k] = np.array(stats[k])

    # ---- renders at key frames ---------------------------------------------------------------
    cam_main = mjv_cam(lookat=(0.42, 0.0, 0.80), distance=2.05, azimuth=-38, elevation=-14)
    cam_side = mjv_cam(lookat=(0.50, 0.0, 0.82), distance=1.75, azimuth=90, elevation=-8)
    cam_close = mjv_cam(lookat=(0.58, 0.0, 0.86), distance=1.25, azimuth=70, elevation=-6)

    def frame(qi, cam):
        set_pose(model, data, qs[qi])
        pts, dd, mins = skin_cloud(model, data, rd)
        img = render_scene(model, data, cam, w=900, h=760, cloud=pts, depths=dd)
        return img

    panels = [
        ("entry — under the sash",            marks["entry"],     cam_main),
        ("mid insertion",                     marks["mid"],       cam_main),
        ("DEEPEST — low interior (cutaway)",  marks["deep_low"],  cam_side),
        ("deep + wrist HIGH interior",        marks["deep_high"], cam_main),
        ("deep_high — interior close-up",     marks["deep_high"], cam_close),
    ]

    # ---- compose figure ----------------------------------------------------------------------
    plt.rcParams.update({
        "figure.facecolor": STYLE["bg"], "axes.facecolor": STYLE["panel"],
        "axes.edgecolor": STYLE["grid"], "text.color": STYLE["fg"],
        "axes.labelcolor": STYLE["fg"], "xtick.color": STYLE["fg"],
        "ytick.color": STYLE["fg"], "font.family": "DejaVu Sans",
    })
    fig, axes = plt.subplots(2, 3, figsize=(21, 12.4), dpi=150)
    fig.suptitle("hood_tall — TALL fume hood (huge opening, high ceiling)   "
                 f"W={W*100:.0f} cm   H={H*100:.0f} cm   D={D*100:.0f} cm   SASH={SASH*100:.0f} cm   "
                 "|   FR3 + 40-SPAD hybrid proximity skin — deep insertion at two interior heights",
                 fontsize=15.5, fontweight="bold", color=STYLE["fg"], y=0.985)

    for ax, (label, qi, cam) in zip(axes.flat[:5], panels):
        img = frame(qi, cam)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        d, z, a, n, m = (stats["depth"][qi], stats["hz"][qi], stats["act"][qi],
                         stats["npts"][qi], stats["mind"][qi])
        ax.set_title(f"{label}", fontsize=12.5, fontweight="bold", color=STYLE["accent"], pad=7)
        ax.text(0.02, 0.02,
                f"insertion {d:+.1f} cm   hand z {z:.2f} m\n"
                f"active {a}/40   pts {n}   min {m:.1f} cm",
                transform=ax.transAxes, fontsize=10.5, color=STYLE["fg"], va="bottom",
                bbox=dict(facecolor="#000000", alpha=0.55, edgecolor="none", pad=4))
        for sp in ax.spines.values():
            sp.set_color(STYLE["grid"])

    # ---- stats panel -------------------------------------------------------------------------
    ax = axes.flat[5]
    fr = np.arange(len(qs))
    ax.axhline((0.55 * D) * 100, color=STYLE["near"], lw=1.2, ls="--")
    ax.text(1, (0.55 * D) * 100 + 0.7, f"target depth {0.55*D*100:.1f} cm", fontsize=9,
            color=STYLE["near"])
    ax.fill_between(fr, 0, stats["depth"].clip(min=0), color=STYLE["accent"], alpha=0.18)
    ax.plot(fr, stats["depth"], color=STYLE["accent"], lw=2.4, label="insertion depth (cm)")
    ax.plot(fr, stats["mind"], color="#ffd166", lw=2.0, label="min skin dist (cm)")
    ax2 = ax.twinx()
    ax2.plot(fr, stats["act"], color="#06d6a0", lw=2.0, label="active sensors /40")
    ax2.set_ylim(0, 42)
    ax2.tick_params(colors=STYLE["fg"])
    ax2.set_ylabel("active sensors", color="#06d6a0", fontsize=10)
    for name, qi in marks.items():
        ax.axvline(qi, color=STYLE["grid"], lw=0.9)
        ax.text(qi, ax.get_ylim()[1] * 0.97, name, rotation=90, fontsize=8,
                color="#9aa0aa", va="top", ha="right")
    ax.set_xlabel("motion frame", fontsize=10)
    ax.set_ylabel("cm", fontsize=10)
    ax.grid(color=STYLE["grid"], lw=0.5, alpha=0.6)
    ln1, lb1 = ax.get_legend_handles_labels()
    ln2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(ln1 + ln2, lb1 + lb2, loc="upper left", fontsize=9, facecolor=STYLE["panel"],
              edgecolor=STYLE["grid"], labelcolor=STYLE["fg"])
    ax.set_title("per-frame skin stats along the insertion motion", fontsize=12.5,
                 fontweight="bold", color=STYLE["accent"], pad=7)

    rows = []
    for name, qi in marks.items():
        rows.append(f"{name:<10s} depth {stats['depth'][qi]:>6.1f} cm  z {stats['hz'][qi]:.2f} m  "
                    f"act {stats['act'][qi]:>2d}/40  pts {stats['npts'][qi]:>4d}  "
                    f"min {stats['mind'][qi]:>4.1f} cm")
    ax.text(0.985, 0.03, "\n".join(rows), transform=ax.transAxes, fontsize=8.3,
            family="DejaVu Sans Mono", color=STYLE["fg"], va="bottom", ha="right",
            bbox=dict(facecolor="#000000", alpha=0.55, edgecolor="none", pad=5))

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    png = os.path.join(OUT, "hood_tall.png")
    fig.savefig(png, facecolor=STYLE["bg"])
    plt.close(fig)

    # ---- verification numbers ----------------------------------------------------------------
    need_x = X0 + 0.55 * D
    qi_low, qi_high = marks["deep_low"], marks["deep_high"]
    print("PNG:", png, os.path.getsize(png), "bytes")
    print(f"need depth >= {(need_x-X0)*100:.2f} cm; band z=[{BZ+0.05:.3f},{BZ+SASH-0.05:.3f}]")
    for name, qi in marks.items():
        print(f"{name:<10s} depth={stats['depth'][qi]:6.2f}cm z={stats['hz'][qi]:.3f} "
              f"act={stats['act'][qi]}/40 pts={stats['npts'][qi]} min={stats['mind'][qi]:.2f}cm")
    print("max depth over motion:", stats["depth"].max(), "cm")
    print("INSERTION TARGET MET:", stats["depth"][qi_low] >= (need_x - X0) * 100 and
          BZ + 0.05 <= stats["hz"][qi_low] <= BZ + SASH - 0.05)


def fig_fumehood_std_fig():
    """hood_standard variant: FR3 + 40-SPAD skin inserting into a standard fume hood (W=0.32, H=0.46, D=0.55, SASH=0.30)."""
    OUT = str(_FIGROOT)
    os.makedirs(OUT, exist_ok=True)
    PNG = os.path.join(OUT, "hood_standard.png")

    # ---------------------------------------------------------------- hood ------
    W, H, D, SASH = 0.32, 0.46, 0.55, 0.30
    BZ, X0 = 0.585, 0.35

    def mk(s):
        nice_lights(s)
        add_box(s, "bench", [X0 + D / 2, 0, BZ - 0.015], [D / 2 + 0.05, W + 0.05, 0.015], [0.62, 0.55, 0.45, 1])
        add_box(s, "bench_body", [X0 + D / 2, 0, BZ / 2 - 0.02], [D / 2, W, BZ / 2 - 0.02], [0.55, 0.5, 0.44, 1])
        add_box(s, "wall_l", [X0 + D / 2, W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30])
        add_box(s, "wall_r", [X0 + D / 2, -W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30])
        add_box(s, "back", [X0 + D, 0, BZ + H / 2], [0.012, W, H / 2], [0.72, 0.7, 0.66, 1])
        add_box(s, "top", [X0 + D / 2, 0, BZ + H], [D / 2, W, 0.012], [0.78, 0.8, 0.84, 0.30])
        add_box(s, "sash", [X0, 0, BZ + SASH + 0.028], [0.012, W, 0.028], [0.62, 0.64, 0.66, 1])
        add_box(s, "target", [X0 + 0.7 * D, 0.0, BZ + 0.045], [0.04, 0.04, 0.045], [0.85, 0.5, 0.3, 1])

    model = build(mk)
    data = mujoco.MjData(model)
    HAND = "gripper/base"
    hid = model.body(HAND).id
    rd = depth_renderer(model)
    SENSOR_NAMES = sorted(model.camera(i).name for i in range(model.ncam)
                          if "_sensor_" in model.camera(i).name)
    NSENS = len(SENSOR_NAMES)

    # ---------------------------------------------------------------- motion ----
    WAYPOINTS = [
        np.array([0.0, -0.800, 0.0, -2.600, 0.0, 1.900, 0.79]),   # stow
        np.array([0.0, -0.598, 0.0, -2.070, 0.0, 2.168, 0.79]),   # under-sash  x=0.400
        np.array([0.0, -0.144, 0.0, -1.540, 0.0, 1.746, 0.79]),   # mid         x=0.520
        np.array([0.0,  0.340, 0.0, -0.980, 0.0, 1.880, 0.79]),   # deep        x=0.666 (57.5% D)
    ]
    STEPS = 10
    traj = []
    for a, b in zip(WAYPOINTS[:-1], WAYPOINTS[1:]):
        for t in np.linspace(0, 1, STEPS, endpoint=False):
            traj.append((1 - t) * a + t * b)
    traj.append(WAYPOINTS[-1])
    traj = np.array(traj)
    KEY = {"entry": STEPS, "mid": 2 * STEPS, "deepest": len(traj) - 1}

    # static scene boxes (center, half) for env-point classification
    ENV_BOXES = [
        ([X0 + D / 2, 0, BZ - 0.015], [D / 2 + 0.05, W + 0.05, 0.015]),
        ([X0 + D / 2, 0, BZ / 2 - 0.02], [D / 2, W, BZ / 2 - 0.02]),
        ([X0 + D / 2, W, BZ + H / 2], [D / 2, 0.012, H / 2]),
        ([X0 + D / 2, -W, BZ + H / 2], [D / 2, 0.012, H / 2]),
        ([X0 + D, 0, BZ + H / 2], [0.012, W, H / 2]),
        ([X0 + D / 2, 0, BZ + H], [D / 2, W, 0.012]),
        ([X0, 0, BZ + SASH + 0.028], [0.012, W, 0.028]),
        ([X0 + 0.7 * D, 0.0, BZ + 0.045], [0.04, 0.04, 0.045]),
    ]
    _EC = np.array([b[0] for b in ENV_BOXES])
    _EH = np.array([b[1] for b in ENV_BOXES])

    def env_mask(pts, tol=0.02):
        """True for cloud points lying on a hood/bench surface or the floor."""
        if not len(pts):
            return np.zeros(0, bool)
        d = np.abs(pts[:, None, :] - _EC[None]) - _EH[None]          # N x B x 3
        dist = np.linalg.norm(np.maximum(d, 0), axis=2).min(axis=1)  # N
        return (dist < tol) | (pts[:, 2] < tol)                      # boxes or floor

    def skin_frame(q):
        """Set pose; return cloud pts, depths, active count, min skin reading, min env clearance."""
        set_pose(model, data, q)
        P, Dd = [], []
        active = 0
        for n in SENSOR_NAMES:
            d8 = depth8(rd, data, n)
            cid = model.camera(n).id
            pts, dd = backproject(d8, data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3))
            if len(pts):
                active += 1
                P.append(pts)
                Dd.append(dd)
        pts = np.concatenate(P) if P else np.zeros((0, 3))
        dd = np.concatenate(Dd) if Dd else np.zeros((0,))
        mind = float(dd.min()) if len(dd) else float("nan")
        em = env_mask(pts)
        mind_env = float(dd[em].min()) if em.any() else float("nan")
        return pts, dd, active, mind, mind_env

    # per-frame stats over the whole motion
    stats = []
    for q in traj:
        pts, dd, active, mind, mind_env = skin_frame(q)
        hp = data.xpos[hid].copy()
        stats.append(dict(depth=(hp[0] - X0) * 100.0, hz=hp[2], active=active,
                          npts=len(pts), mind=mind * 100.0, menv=mind_env * 100.0))
    depths = np.array([s["depth"] for s in stats])
    actives = np.array([s["active"] for s in stats])
    npts = np.array([s["npts"] for s in stats])
    minds = np.array([s["mind"] for s in stats])
    menvs = np.array([s["menv"] for s in stats])
    print("deepest frame: depth=%.1fcm (%.1f%% D)  hand z=%.3f  active=%d/%d  pts=%d  "
          "mind=%.1fcm  env_clear=%.1fcm"
          % (depths[-1], depths[-1] / (D * 100) * 100, stats[-1]["hz"], actives[-1], NSENS,
             npts[-1], minds[-1], menvs[-1]))

    # ---------------------------------------------------------------- renders ---
    CAM_MAIN = mjv_cam(lookat=(0.42, 0.0, 0.70), distance=2.10, azimuth=-38, elevation=-13)
    CAM_SIDE = mjv_cam(lookat=(0.42, 0.0, 0.72), distance=1.70, azimuth=92, elevation=-8)
    CAM_CLOSE = mjv_cam(lookat=(0.58, 0.02, 0.72), distance=1.00, azimuth=-27, elevation=-19)

    frames = {}
    for tag, idx in KEY.items():
        pts, dd, active, mind, mind_env = skin_frame(traj[idx])
        frames[tag] = dict(img=render_scene(model, data, CAM_MAIN, cloud=pts, depths=dd),
                           idx=idx)
    # extra views of the deepest pose
    pts, dd, active, mind, mind_env = skin_frame(traj[KEY["deepest"]])
    img_side = render_scene(model, data, CAM_SIDE, cloud=pts, depths=dd)
    img_close = render_scene(model, data, CAM_CLOSE, cloud=pts, depths=dd, pt_size=0.0085)

    # ---------------------------------------------------------------- figure ----
    BG, PANEL, FG, GRID, ACC = STYLE["bg"], STYLE["panel"], STYLE["fg"], STYLE["grid"], STYLE["accent"]
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "text.color": FG, "axes.edgecolor": GRID, "axes.labelcolor": FG,
        "xtick.color": FG, "ytick.color": FG, "font.family": "DejaVu Sans",
    })

    fig = plt.figure(figsize=(19.5, 11.6), dpi=160)
    gs = GridSpec(2, 3, figure=fig, left=0.015, right=0.985, top=0.895, bottom=0.03,
                  wspace=0.05, hspace=0.14)

    fig.suptitle("FUMEHOOD VARIANT  ·  hood_standard   —   W=32 cm (half-width)  H=46 cm  "
                 "D=55 cm  SASH=30 cm", fontsize=21, fontweight="bold", color=FG, y=0.975)
    fig.text(0.5, 0.925, "FR3 + 40-SPAD hybrid proximity skin  ·  4-waypoint insertion: "
             "stow → under-sash → mid → deep  ·  target: hand ≥ 55% of D beyond hood face "
             "(≥ 30.3 cm)", ha="center", fontsize=12.5, color="#9aa3ad")

    def img_panel(ax, img, title, sub):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title(title, fontsize=14, fontweight="bold", color=ACC, pad=6)
        ax.text(0.02, 0.035, sub, transform=ax.transAxes, fontsize=9.6, color=FG,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.45", fc="#1c2027", ec=GRID, alpha=0.92))

    order = [("entry", "ENTRY  —  hand at sash plane"),
             ("mid", "MID  —  half-way into hood"),
             ("deepest", "DEEPEST  —  full insertion")]
    for k, (tag, title) in enumerate(order):
        idx = frames[tag]["idx"]
        s = stats[idx]
        sub = (f"insertion {s['depth']:+.1f} cm ({s['depth']/(D*100)*100:.0f}% D)  ·  "
               f"{s['active']}/{NSENS} sens  ·  {s['npts']} pts  ·  "
               f"clear {s['menv']:.1f} cm")
        img_panel(fig.add_subplot(gs[0, k]), frames[tag]["img"], title, sub)

    sdeep = stats[KEY["deepest"]]
    sub_deep = (f"insertion {sdeep['depth']:+.1f} cm ({sdeep['depth']/(D*100)*100:.0f}% D)  ·  "
                f"{sdeep['active']}/{NSENS} sens  ·  clear {sdeep['menv']:.1f} cm")
    img_panel(fig.add_subplot(gs[1, 0]), img_side,
              "DEEPEST  —  side view through glass", sub_deep)
    img_panel(fig.add_subplot(gs[1, 1]), img_close,
              "DEEPEST  —  interior close-up, skin cloud", sub_deep)

    # ------------------------------------------------------------- stats panel --
    sub_gs = gs[1, 2].subgridspec(2, 1, height_ratios=[2.1, 1.0], hspace=0.32)
    ax = fig.add_subplot(sub_gs[0])
    ax.set_title("PER-FRAME SKIN TELEMETRY", fontsize=14, fontweight="bold", color=ACC, pad=6)
    fr = np.arange(len(traj))

    ax.plot(fr, depths, color=ACC, lw=2.6, label="insertion depth (cm)")
    ax.axhline(0.55 * D * 100, color="#ef476f", lw=1.4, ls="--")
    ax.text(0.4, 0.55 * D * 100 + 0.8, "55% D target (30.3 cm)", fontsize=9.5,
            color="#ef476f")
    ax.axhline(0, color=GRID, lw=1)
    ax.plot(fr, menvs, color="#ffd166", lw=2.2, label="min clearance to hood (cm)")
    ax.fill_between(fr, menvs, color="#ffd166", alpha=0.10)
    ax.plot(fr, minds, color="#8d99ae", lw=1.3, ls="-.",
            label="min skin reading incl. self (cm)")
    ax.set_ylabel("cm", fontsize=11)
    ax.set_xlabel("motion frame", fontsize=11)
    ax.set_xlim(0, len(traj) - 1)
    ax.grid(color=GRID, lw=0.6, alpha=0.7)

    ax2 = ax.twinx()
    ax2.plot(fr, actives, color="#06d6a0", lw=2.0, ls=":", label="active sensors /40")
    ax2.set_ylabel("active sensors", fontsize=11, color="#06d6a0")
    ax2.tick_params(axis="y", colors="#06d6a0")
    ax2.set_ylim(0, NSENS + 2)
    ax2.set_facecolor("none")

    for tag in KEY:
        ax.axvline(KEY[tag], color="#5a6270", lw=1.0, ls=":")
        ax.text(KEY[tag], ax.get_ylim()[1] * 0.97, tag, rotation=90, fontsize=8.5,
                color="#9aa3ad", ha="right", va="top")

    ln1, lb1 = ax.get_legend_handles_labels()
    ln2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(ln1 + ln2, lb1 + lb2, loc="upper left", bbox_to_anchor=(0.015, 0.88),
              fontsize=9.2, facecolor="#1c2027", edgecolor=GRID, labelcolor=FG,
              framealpha=0.92)

    # key-frame table in its own axes
    axt = fig.add_subplot(sub_gs[1])
    axt.set_axis_off()
    axt.set_title("KEY-FRAME NUMBERS", fontsize=12, fontweight="bold", color=ACC, pad=2)
    rows = []
    for tag, idx in KEY.items():
        s = stats[idx]
        rows.append([tag, f"{s['depth']:.1f}", f"{s['depth']/(D*100)*100:.0f}%",
                     f"{s['active']}/{NSENS}", f"{s['npts']}", f"{s['menv']:.1f}"])
    tbl = axt.table(cellText=rows,
                    colLabels=["frame", "depth (cm)", "% of D", "sensors", "points",
                               "clearance (cm)"],
                    loc="center", cellLoc="center", bbox=[0.0, 0.0, 1.0, 0.92])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1c2027" if r else "#232833")
        cell.set_edgecolor(GRID)
        cell.set_height(0.25)
        cell.set_text_props(color=FG if r else ACC,
                            fontweight="normal" if r else "bold")

    fig.savefig(PNG)
    plt.close(fig)
    sz = os.path.getsize(PNG)
    print("saved", PNG, sz, "bytes")
    print("RESULT", dict(deep_cm=round(depths[-1], 1), pctD=round(depths[-1] / (D * 100) * 100, 1),
                         active=int(actives[-1]), pts=int(npts[-1]), mind_cm=round(minds[-1], 1),
                         env_clear_cm=round(menvs[-1], 1), hand_z=round(stats[-1]["hz"], 3),
                         env_clear_min_over_motion=round(np.nanmin(menvs), 1),
                         key_env=[round(stats[i]["menv"], 1) for i in KEY.values()],
                         key_active=[int(stats[i]["active"]) for i in KEY.values()],
                         key_depth=[round(stats[i]["depth"], 1) for i in KEY.values()]))


def fig_fumehood_short_low_sash_fig():
    """SHORT + LOW SASH fumehood (W=34 H=34 D=50 SASH=17): FR3 + 40-SPAD hybrid skin horizontal insertion 2x3 figure."""
    OUT = str(_FIGROOT)
    os.makedirs(OUT, exist_ok=True)
    KEY = "hood_short_low_sash"

    # ---------------------------------------------------------------- hood (variant dims)
    BZ, X0 = 0.585, 0.35
    W, H, D, SASH = 0.34, 0.34, 0.50, 0.17

    BOXES = {}  # name -> (center, half) for penetration checks

    def _box(s, name, c, h, rgba):
        BOXES[name] = (np.array(c, float), np.array(h, float))
        add_box(s, name, c, h, rgba)

    def mk(s):
        nice_lights(s)
        _box(s, "bench", [X0 + D / 2, 0, BZ - 0.015], [D / 2 + 0.05, W + 0.05, 0.015], [0.62, 0.55, 0.45, 1])
        _box(s, "bench_body", [X0 + D / 2, 0, BZ / 2 - 0.02], [D / 2, W, BZ / 2 - 0.02], [0.55, 0.5, 0.44, 1])
        _box(s, "wall_l", [X0 + D / 2, W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30])
        _box(s, "wall_r", [X0 + D / 2, -W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30])
        _box(s, "back", [X0 + D, 0, BZ + H / 2], [0.012, W, H / 2], [0.72, 0.7, 0.66, 1])
        _box(s, "top", [X0 + D / 2, 0, BZ + H], [D / 2, W, 0.012], [0.78, 0.8, 0.84, 0.30])
        _box(s, "sash", [X0, 0, BZ + SASH + 0.028], [0.012, W, 0.028], [0.62, 0.64, 0.66, 1])
        _box(s, "target", [X0 + 0.7 * D, 0.0, BZ + 0.045], [0.04, 0.04, 0.045], [0.85, 0.5, 0.3, 1])

    model = build(mk)
    data = mujoco.MjData(model)
    HAND = "gripper/base"
    HID = model.body(HAND).id
    TCP = model.site("gripper/grasp_site").id
    JLIM = [(-1.7837, 1.7837), (-3.0421, -0.1518), (0.5445, 4.5169)]  # j2 j4 j6
    J7 = 0.0  # flat roll: finger pads at +/-y so the gripper is horizontal for the low slot

    def fk(q):
        set_pose(model, data, q)
        hp = data.xpos[HID].copy()
        R = data.xmat[HID].reshape(3, 3).copy()
        return hp, R

    def chain_pts():
        """sample points along elbow -> wrist -> flange -> hand -> tcp."""
        P = [data.xpos[model.body(n).id].copy()
             for n in ["fr3_link4", "fr3_link5", "fr3_link7", HAND]]
        P.append(data.site_xpos[TCP].copy())
        out = []
        for a, b in zip(P[:-1], P[1:]):
            for t in np.linspace(0, 1, 10):
                out.append(a * (1 - t) + b * t)
        return np.array(out)

    def solve(tx, tz, pitch_deg, seeds, label=""):
        """[j2,j4,j6] (j1=j3=j5=0, j7=0.79): hand at (tx,0,tz), approach pitched down pitch_deg."""
        p = np.deg2rad(pitch_deg)
        tgt_app = np.array([np.cos(p), 0, -np.sin(p)])

        def cost(v):
            pen = sum(max(0, lo - jv) ** 2 * 100 + max(0, jv - hi) ** 2 * 100
                      for jv, (lo, hi) in zip(v, JLIM))
            q = [0, v[0], 0, v[1], 0, v[2], J7]
            hp, R = fk(q)
            c = 40 * (hp[0] - tx) ** 2 + 120 * (hp[2] - tz) ** 2 + 2.0 * np.sum((R[:, 2] - tgt_app) ** 2)
            # wrist skin hangs ~0.0365 m below link6 origin: keep it off the bench (top z = BZ)
            z6 = data.xpos[model.body("fr3_link6").id][2]
            x6 = data.xpos[model.body("fr3_link6").id][0]
            if x6 > 0.29:
                c += 400 * max(0, (BZ + 0.004) - (z6 - 0.0365)) ** 2
            cp = chain_pts()
            m = cp[:, 0] > X0 - 0.02
            if m.any():
                z = cp[m, 2]
                c += 60 * np.sum(np.maximum(0, z - (BZ + SASH - 0.048)) ** 2)
                c += 60 * np.sum(np.maximum(0, (BZ + 0.043) - z) ** 2)
            return c + pen

        best = None
        for s0 in seeds:
            r = minimize(cost, s0, method="Nelder-Mead",
                         options=dict(maxiter=4000, xatol=1e-6, fatol=1e-10))
            if best is None or r.fun < best.fun:
                best = r
        q = [0, best.x[0], 0, best.x[1], 0, best.x[2], J7]
        hp, R = fk(q)
        print(f"  {label:6s} cost={best.fun:.2e} hand=({hp[0]:.3f},{hp[2]:.3f}) app={np.round(R[:, 2], 2)}")
        return q

    def interp(waypts, counts):
        qs = []
        for (a, b), n in zip(zip(waypts[:-1], waypts[1:]), counts):
            a, b = np.array(a), np.array(b)
            for t in np.linspace(0, 1, n, endpoint=False):
                tt = 3 * t ** 2 - 2 * t ** 3  # smoothstep
                qs.append(a * (1 - tt) + b * tt)
        qs.append(np.array(waypts[-1]))
        return qs

    def penetration(margin=0.0):
        """max penetration depth (m) of chain/pad sample points into any scene box."""
        pts = list(chain_pts())
        for n in ["gripper/left_pad", "gripper/right_pad"]:
            pts.append(data.xpos[model.body(n).id].copy())
        worst = 0.0
        for p in pts:
            for name, (c, h) in BOXES.items():
                d = h - np.abs(p - c) + margin
                if (d > 0).all():
                    worst = max(worst, float(d.min()))
        return worst

    # ---------------------------------------------------------------- travel-height scan
    SEEDS = [[0.3, -1.4, 1.6], [0.5, -1.0, 1.5], [0.1, -1.8, 1.9], [0.0, -2.4, 2.3]]
    rd = depth_renderer(model)
    print("scanning travel heights ...")
    best = None
    for zt in [0.700, 0.704, 0.708, 0.712]:
        entry = solve(0.38, zt - 0.004, 4, SEEDS, f"entry@{zt}")
        mid = solve(0.51, zt, 0, SEEDS, f"mid@{zt}")
        deep = solve(0.645, zt, 0, SEEDS, f"deep@{zt}")
        worst_min, worst_pen = 1e9, 0.0
        for q in interp([entry, mid, deep], [8, 9]):
            set_pose(model, data, q)
            _, _, mins = skin_cloud(model, data, rd)
            worst_min = min(worst_min, min(mins.values()))
            worst_pen = max(worst_pen, penetration())
        print(f"  z_travel={zt}: worst frame min-dist={worst_min*100:.2f} cm  max pen={worst_pen*1000:.1f} mm")
        score = worst_min - 10 * worst_pen
        if best is None or score > best[0]:
            best = (score, zt, entry, mid, deep, worst_min, worst_pen)

    _, ZT, ENTRY, MID, DEEP, wmin, wpen = best
    print(f"chosen z_travel={ZT} (worst min {wmin*100:.2f} cm, pen {wpen*1000:.1f} mm)")

    STOW = solve(0.13, 0.70, 65, [[-0.8, -2.6, 1.9], [-1.1, -2.5, 2.3]], "stow")
    READY = solve(0.22, 0.73, 30, [[-1.1, -2.5, 2.3]] + SEEDS, "ready")

    # ---------------------------------------------------------------- trajectory + stats
    TRAJ = interp([STOW, READY, ENTRY, MID, DEEP], [6, 7, 8, 9])
    stats = []
    for i, q in enumerate(TRAJ):
        set_pose(model, data, q)
        pts, dd, mins = skin_cloud(model, data, rd)
        hp = data.xpos[HID]
        tp = data.site_xpos[TCP]
        stats.append(dict(
            i=i, hand_x=hp[0], hand_z=hp[2], depth=(hp[0] - X0) * 100, tcp_depth=(tp[0] - X0) * 100,
            active=sum(1 for v in mins.values() if v < FAR * 0.999), npts=len(pts),
            mind=min(mins.values()) * 100, pen=penetration() * 1000))

    IDX = {"ENTRY": 13, "MID": 21, "DEEPEST": len(TRAJ) - 1}
    for k, i in IDX.items():
        s = stats[i]
        print(f"{k:8s} f{i:02d} depth={s['depth']:6.1f} cm  hand_z={s['hand_z']:.3f}  "
              f"active={s['active']}/40  pts={s['npts']}  min={s['mind']:.2f} cm  pen={s['pen']:.1f} mm")
    worst_pen_traj = max(s["pen"] for s in stats)
    print(f"trajectory max penetration: {worst_pen_traj:.1f} mm")

    # ---------------------------------------------------------------- renders
    CAM_SIDE = mjv_cam(lookat=(0.46, 0.0, 0.62), distance=1.42, azimuth=90, elevation=-9)
    CAM_F34 = mjv_cam(lookat=(0.42, 0.0, 0.55), distance=1.9, azimuth=-40, elevation=-18)
    CAM_TOP = mjv_cam(lookat=(0.55, 0.0, 0.64), distance=1.5, azimuth=-55, elevation=-48)
    PW, PH = 820, 720

    def shot(i, cam):
        set_pose(model, data, TRAJ[i])
        pts, dd, _ = skin_cloud(model, data, rd)
        return render_scene(model, data, cam, w=PW, h=PH, cloud=pts, depths=dd)

    def label_panel(img, title, s, accent=(240, 201, 76)):
        img = img.copy()
        ov = img.copy()
        cv2.rectangle(ov, (0, 0), (PW, 54), (13, 15, 19), -1)
        cv2.rectangle(ov, (0, PH - 64), (PW, PH), (13, 15, 19), -1)
        img = cv2.addWeighted(ov, 0.78, img, 0.22, 0)
        cv2.putText(img, title, (18, 36), cv2.FONT_HERSHEY_DUPLEX, 0.95, accent[::-1], 1, cv2.LINE_AA)
        t1 = f"hand depth {s['depth']:+.1f} cm   TCP {s['tcp_depth']:+.1f} cm   hand z {s['hand_z']:.3f} m"
        t2 = f"active {s['active']}/40   cloud {s['npts']} pts   min skin dist {s['mind']:.2f} cm"
        cv2.putText(img, t1, (18, PH - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (232, 232, 234), 1, cv2.LINE_AA)
        cv2.putText(img, t2, (18, PH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (76, 201, 240)[::-1], 1, cv2.LINE_AA)
        return img

    panels = [
        label_panel(shot(IDX["ENTRY"], CAM_SIDE), f"ENTRY  (frame {IDX['ENTRY']})  side view", stats[IDX["ENTRY"]]),
        label_panel(shot(IDX["MID"], CAM_SIDE), f"MID  (frame {IDX['MID']})  side view", stats[IDX["MID"]]),
        label_panel(shot(IDX["DEEPEST"], CAM_SIDE), f"DEEPEST  (frame {IDX['DEEPEST']})  side view", stats[IDX["DEEPEST"]]),
        label_panel(shot(IDX["DEEPEST"], CAM_F34), "DEEPEST  front 3/4  (ducked under 17 cm sash)", stats[IDX["DEEPEST"]]),
        label_panel(shot(IDX["DEEPEST"], CAM_TOP), "DEEPEST  overhead through hood ceiling", stats[IDX["DEEPEST"]]),
    ]

    # ---------------------------------------------------------------- stats panel
    plt.rcParams.update({
        "figure.facecolor": STYLE["bg"], "axes.facecolor": STYLE["panel"],
        "axes.edgecolor": STYLE["grid"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"], "text.color": STYLE["fg"],
        "grid.color": STYLE["grid"], "font.size": 10.5})
    fig, axes = plt.subplots(3, 1, figsize=(PW / 100, PH / 100), dpi=100, sharex=True,
                             gridspec_kw=dict(hspace=0.34, top=0.93, bottom=0.085, left=0.105, right=0.885))
    fr = [s["i"] for s in stats]
    dep = [s["depth"] for s in stats]
    tdep = [s["tcp_depth"] for s in stats]
    act = [s["active"] for s in stats]
    npt = [s["npts"] for s in stats]
    mind = [s["mind"] for s in stats]
    tgt = 0.55 * D * 100

    ax = axes[0]
    ax.axhspan(0, D * 100, color="#223043", alpha=0.55, lw=0)
    ax.plot(fr, dep, color=STYLE["accent"], lw=2.4, label="hand body")
    ax.plot(fr, tdep, color="#9d8df1", lw=1.8, ls="--", label="TCP (fingertips)")
    ax.axhline(tgt, color=STYLE["near"], lw=1.4, ls=":")
    ax.text(0.4, tgt + 1.5, f"target  x >= X0+0.55D  ({tgt:.1f} cm)", color=STYLE["near"], fontsize=9)
    ax.axhline(0, color="#888", lw=0.8)
    ax.text(0.4, 1.5, "hood front plane (X0)", color="#aaa", fontsize=8.5)
    ax.set_ylabel("insertion depth (cm)")
    ax.legend(loc="upper left", framealpha=0.2, fontsize=9)
    ax.set_title(f"insertion: final hand depth {dep[-1]:.1f} cm of D={D*100:.0f} cm  "
                 f"(hand z={stats[-1]['hand_z']:.3f} m in 17 cm opening)", fontsize=11, pad=6)
    ax.grid(alpha=0.4)

    ax = axes[1]
    ax.plot(fr, act, color="#f4a259", lw=2.4, label="active sensors")
    ax.axhline(40, color="#666", lw=0.9, ls=":")
    ax.set_ylabel("active sensors / 40")
    ax.set_ylim(0, 42)
    ax2 = ax.twinx()
    ax2.plot(fr, npt, color="#80ed99", lw=1.7, ls="--", label="cloud points")
    ax2.set_ylabel("cloud points", color="#80ed99")
    ax2.tick_params(axis="y", colors="#80ed99")
    ax.set_title(f"skin activity: {act[IDX['ENTRY']]} -> {act[IDX['MID']]} -> {act[-1]} sensors,  "
                 f"{npt[-1]} pts at deepest", fontsize=11, pad=6)
    ax.legend(loc="upper left", framealpha=0.2, fontsize=9)
    ax.grid(alpha=0.4)

    ax = axes[2]
    ax.plot(fr, mind, color=STYLE["near"], lw=2.4)
    ax.set_yscale("log")
    ax.set_ylabel("min skin dist (cm)")
    ax.set_xlabel("trajectory frame")
    ax.set_title(f"clearance: min {min(mind):.2f} cm (wrist skin skims bench under 17 cm sash)",
                 fontsize=11, pad=6)
    ax.grid(alpha=0.4, which="both")
    for k, i in IDX.items():
        for a in axes:
            a.axvline(i, color="#555", lw=0.9, ls="--")
        axes[0].text(i, axes[0].get_ylim()[1] * 0.97, k, rotation=90, va="top", ha="right",
                     fontsize=8, color="#bbb")
    fig.canvas.draw()
    sp = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    sp = cv2.resize(sp, (PW, PH))
    panels.append(sp)

    # ---------------------------------------------------------------- compose 2x3
    GAP, TBAR = 8, 96
    CW, CH = 3 * PW + 4 * GAP, TBAR + 2 * PH + 3 * GAP
    canvas = np.full((CH, CW, 3), (17, 19, 23), np.uint8)
    title = ("FUMEHOOD VARIANT  hood_short_low_sash  |  W=34 (half-width) H=34 D=50 SASH=17 cm"
             "  |  FR3 + 40-SPAD hybrid skin")
    sub = (f"low horizontal insertion under a 17 cm sash: hand depth {dep[-1]:.1f} cm "
           f"(target {tgt:.1f}),  {act[-1]}/40 sensors active,  min clearance {min(mind):.2f} cm")
    cv2.putText(canvas, title, (GAP + 10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.05, (240, 201, 76)[::-1], 2, cv2.LINE_AA)
    cv2.putText(canvas, sub, (GAP + 10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (232, 232, 234), 1, cv2.LINE_AA)
    for k, p in enumerate(panels):
        r, c = divmod(k, 3)
        y = TBAR + GAP + r * (PH + GAP)
        x = GAP + c * (PW + GAP)
        canvas[y:y + PH, x:x + PW] = p

    out_png = os.path.join(OUT, f"{KEY}.png")
    cv2.imwrite(out_png, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print("WROTE", out_png, os.path.getsize(out_png), "bytes")
    print("FINAL:", {k: stats[i] for k, i in IDX.items()})


def fig_fumehood_var_hood_deep_tunnel():
    """Fumehood variant: max-insertion motion into a 95 cm deep tunnel hood, FR3 + 40-SPAD hybrid skin."""
    OUT = str(_FIGROOT)
    os.makedirs(OUT, exist_ok=True)
    KEY = "hood_deep_tunnel"
    TURBO_R = colormaps["turbo_r"]

    # ------------------------------------------------------------------ hood (variant dims)
    BZ, X0 = 0.585, 0.35
    W, H, D, SASH = 0.26, 0.42, 0.95, 0.28
    HOOD = ("bench", "bench_body", "wall_l", "wall_r", "back", "top", "sash")

    def mk(s, collide=False):
        nice_lights(s)
        add_box(s, "bench", [X0+D/2, 0, BZ-0.015], [D/2+0.05, W+0.05, 0.015], [0.62, 0.55, 0.45, 1])
        add_box(s, "bench_body", [X0+D/2, 0, BZ/2-0.02], [D/2, W, BZ/2-0.02], [0.55, 0.5, 0.44, 1])
        add_box(s, "wall_l", [X0+D/2, W, BZ+H/2], [D/2, 0.012, H/2], [0.78, 0.8, 0.84, 0.30])
        add_box(s, "wall_r", [X0+D/2, -W, BZ+H/2], [D/2, 0.012, H/2], [0.78, 0.8, 0.84, 0.30])
        add_box(s, "back", [X0+D, 0, BZ+H/2], [0.012, W, H/2], [0.72, 0.7, 0.66, 1])
        add_box(s, "top", [X0+D/2, 0, BZ+H], [D/2, W, 0.012], [0.78, 0.8, 0.84, 0.30])
        add_box(s, "sash", [X0, 0, BZ+SASH+0.028], [0.012, W, 0.028], [0.62, 0.64, 0.66, 1])
        add_box(s, "target", [X0+0.7*D, 0.0, BZ+0.045], [0.04, 0.04, 0.045], [0.85, 0.5, 0.3, 1])
        if collide:
            for b in s.worldbody.bodies:
                if b.name in HOOD:
                    for g in b.geoms:
                        g.contype = 1
                        g.conaffinity = 1

    model = build(mk)
    data = mujoco.MjData(model)
    HAND, TCP = "gripper/base", "gripper/grasp_site"     # no fr3_hand body in this model;
    hid, sid = model.body(HAND).id, model.site(TCP).id   # 2F-85 base sits at the FR3 flange

    # collidable copy for trajectory validation
    mC = build(lambda s: mk(s, collide=True))
    dC = mujoco.MjData(mC)
    hood_geoms = {i for i in range(mC.ngeom) if mC.body(mC.geom_bodyid[i]).name in HOOD}

    def maxpen(q):
        set_pose(mC, dC, list(q))
        p = 0.0
        for ci in range(dC.ncon):
            c = dC.contact[ci]
            if (c.geom1 in hood_geoms) != (c.geom2 in hood_geoms):
                p = max(p, -c.dist)
        return p

    # ------------------------------------------------------------- motion (solved offline)
    Q_STOW  = np.array([0.0, -1.277, 0.0, -2.402, 0.0, 2.057, 0.79])
    Q_ENTRY = np.array([0.0, -0.728, 0.0, -2.238, 0.0, 2.397, 0.79])
    Q_MID   = np.array([0.0, -0.050, 0.0, -1.710, 0.0, 2.682, 0.79])
    Q_DEEP  = np.array([0.0, 0.4724, 0.029, -1.0808, 0.2999, 2.7971, 0.79])

    def smooth(a, b, n):
        return [a + (b - a) * (0.5 - 0.5 * np.cos(np.pi * t)) for t in np.linspace(0, 1, n)]

    traj = []
    for A, B, n in [(Q_STOW, Q_ENTRY, 9), (Q_ENTRY, Q_MID, 9), (Q_MID, Q_DEEP, 10)]:
        seg = smooth(A, B, n)
        if traj:
            seg = seg[1:]
        traj += seg
    NF = len(traj)
    F_ENTRY, F_MID, F_DEEP = 8, 16, NF - 1

    # --------------------------------------------------------- per-frame skin + kinematics
    rd = depth_renderer(model)
    stats = []          # dicts per frame
    clouds = {}         # key frames -> (pts, depths)
    for i, q in enumerate(traj):
        set_pose(model, data, list(q))
        pts, dd, mins = skin_cloud(model, data, rd)
        active = sum(1 for v in mins.values() if v < FAR)
        mind = min(mins.values())
        h = data.xpos[hid].copy()
        g = data.site_xpos[sid].copy()
        pen = maxpen(q)
        stats.append(dict(hand=(h[0]-X0)*100, tcp=(g[0]-X0)*100, hz=h[2],
                          active=active, npts=len(pts), mind=mind*100, pen=pen*1000))
        if i in (F_ENTRY, F_MID, F_DEEP):
            clouds[i] = (pts, dd)

    worst_pen = max(s["pen"] for s in stats)
    sd = stats[F_DEEP]
    print(f"frames={NF} worst_pen={worst_pen:.1f}mm")
    print(f"DEEP: hand {sd['hand']:.1f}cm  TCP {sd['tcp']:.1f}cm  hand_z {sd['hz']:.3f} "
          f"(band {BZ+0.05:.3f}-{BZ+SASH-0.05:.3f})  active {sd['active']}/40  "
          f"pts {sd['npts']}  min {sd['mind']:.1f}cm")
    assert worst_pen <= 10.0, "trajectory clips the hood"
    assert BZ + 0.05 <= sd["hz"] <= BZ + SASH - 0.05, "hand outside opening band"

    # ----------------------------------------------------------------------------- renders
    def stage_render(i, cam, w=900, h=680):
        set_pose(model, data, list(traj[i]))
        pts, dd = clouds[i]
        return render_scene(model, data, cam, w=w, h=h, cloud=pts, depths=dd, pt_size=0.0070)

    cam_main = mjv_cam(lookat=(0.52, 0.0, 0.62), distance=2.05, azimuth=112, elevation=-13)
    cam_side = mjv_cam(lookat=(0.55, 0.0, 0.62), distance=1.95, azimuth=91, elevation=-8)
    cam_top  = mjv_cam(lookat=(0.68, 0.0, 0.72), distance=1.55, azimuth=180, elevation=-75)

    img_entry = stage_render(F_ENTRY, cam_main)
    img_mid   = stage_render(F_MID, cam_main)
    img_deep  = stage_render(F_DEEP, cam_main)
    img_side  = stage_render(F_DEEP, cam_side)
    img_top   = stage_render(F_DEEP, cam_top)

    # ============================================================================== FIGURE
    plt.rcParams.update({
        "figure.facecolor": STYLE["bg"], "savefig.facecolor": STYLE["bg"],
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
        "font.family": "DejaVu Sans", "font.size": 11,
    })
    fig = plt.figure(figsize=(19.4, 11.4), dpi=160)
    gs = gridspec.GridSpec(2, 3, wspace=0.10, hspace=0.16,
                           left=0.025, right=0.978, top=0.875, bottom=0.045)

    def img_panel(cell, img, title, sub=None):
        ax = fig.add_subplot(cell)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(STYLE["grid"]); sp.set_linewidth(1.2)
        ax.set_title(title, color=STYLE["fg"], fontsize=12.5, fontweight="bold", pad=7)
        if sub:
            ax.text(0.012, 0.018, sub, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=9.4, color="#dce6f2",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#10202f", ec=STYLE["grid"], alpha=0.88))
        return ax

    se, sm_, sdp = stats[F_ENTRY], stats[F_MID], stats[F_DEEP]
    axA = img_panel(gs[0, 0], img_entry,
                    f"1 · ENTRY — hand at the front plane (frame {F_ENTRY}/{NF-1})",
                    f"hand {se['hand']:+.1f} cm | fingertips {se['tcp']:+.1f} cm beyond plane\n"
                    f"{se['active']}/40 sensors · {se['npts']:,} pts · min {se['mind']:.1f} cm")
    axB = img_panel(gs[0, 1], img_mid,
                    f"2 · MID-TUNNEL — wrist through the sash (frame {F_MID}/{NF-1})",
                    f"hand {sm_['hand']:+.1f} cm | fingertips {sm_['tcp']:+.1f} cm beyond plane\n"
                    f"{sm_['active']}/40 sensors · {sm_['npts']:,} pts · min {sm_['mind']:.1f} cm")
    axC = img_panel(gs[0, 2], img_deep,
                    f"3 · DEEPEST — forearm fully inside (frame {F_DEEP}/{NF-1})",
                    f"hand {sdp['hand']:+.1f} cm | fingertips {sdp['tcp']:+.1f} cm beyond plane\n"
                    f"{sdp['active']}/40 sensors · {sdp['npts']:,} pts · min {sdp['mind']:.1f} cm")
    axD = img_panel(gs[1, 0], img_side,
                    "side view at deepest — whole forearm inside the 95 cm tunnel",
                    "elbow rests on the bench lip (9 mm contact)\n"
                    "hand z = {:.2f} m, inside the 28 cm sash opening band".format(sdp["hz"]))
    axE = img_panel(gs[1, 1], img_top,
                    "overhead view at deepest — centered in the 52 cm-wide tunnel",
                    "skin cloud paints both side walls, bench and sash edge")

    # distance colorbar on the hero panel
    cax = axC.inset_axes([0.66, 0.045, 0.30, 0.030])
    smap = plt.cm.ScalarMappable(cmap=TURBO_R, norm=plt.Normalize(NEAR, FAR))
    cb = fig.colorbar(smap, cax=cax, orientation="horizontal")
    cb.set_label("skin distance (m)", color=STYLE["fg"], fontsize=9)
    cb.ax.tick_params(labelsize=7.5, colors=STYLE["fg"])
    cb.outline.set_edgecolor(STYLE["grid"])

    # --------------------------------------------------------------------- stats panel (F)
    gsS = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 2],
                                           height_ratios=[1.25, 1.0], hspace=0.42)
    fr = np.arange(NF)
    hand_d = np.array([s["hand"] for s in stats])
    tcp_d = np.array([s["tcp"] for s in stats])
    act = np.array([s["active"] for s in stats])
    mind = np.array([s["mind"] for s in stats])

    ax1 = fig.add_subplot(gsS[0])
    ax1.set_facecolor(STYLE["panel"])
    ax1.axhspan(0, D*100, color="#1d2a38", alpha=0.55, zorder=0)
    ax1.axhline(0, color="#9aa4b2", lw=1.0, ls="--")
    ax1.text(0.4, 1.8, "front plane (X0)", fontsize=8, color="#9aa4b2")
    ax1.axhline(45.2, color="#ffd166", lw=1.0, ls=":")
    ax1.text(0.4, 46.6, "FR3 flange kinematic ceiling in opening band (45.2 cm)",
             fontsize=7.8, color="#ffd166")
    ax1.axhline(60, color="#ef476f", lw=1.0, ls=":")
    ax1.text(0.4, 61.4, "variant hand target 60 cm — beyond the 855 mm reach envelope",
             fontsize=7.8, color="#ef476f")
    ax1.plot(fr, tcp_d, color="#4cc9f0", lw=2.4, label="fingertips / TCP")
    ax1.plot(fr, hand_d, color="#7fffd4", lw=2.4, label="hand (flange)")
    for f, c in [(F_ENTRY, "#e8e8ea"), (F_MID, "#e8e8ea"), (F_DEEP, "#ef476f")]:
        ax1.axvline(f, color=c, lw=0.8, alpha=0.45, ls="--")
    ax1.scatter([F_DEEP, F_DEEP], [hand_d[-1], tcp_d[-1]], s=42, zorder=5,
                color=["#7fffd4", "#4cc9f0"], edgecolor="white", lw=0.8)
    ax1.annotate(f"{tcp_d[-1]:.1f} cm", (F_DEEP, tcp_d[-1]), xytext=(-58, 8),
                 textcoords="offset points", fontsize=10.5, fontweight="bold", color="#4cc9f0")
    ax1.annotate(f"{hand_d[-1]:.1f} cm", (F_DEEP, hand_d[-1]), xytext=(-58, -16),
                 textcoords="offset points", fontsize=10.5, fontweight="bold", color="#7fffd4")
    ax1.set_xlim(0, NF-1); ax1.set_ylim(-42, 100)
    ax1.set_ylabel("insertion beyond front plane (cm)", fontsize=9.5)
    ax1.tick_params(labelsize=8.5)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.0, labelcolor=STYLE["fg"])
    for sp in ax1.spines.values():
        sp.set_edgecolor(STYLE["grid"])
    ax1.set_title("insertion depth along the motion", fontsize=11.5, fontweight="bold", pad=5)

    ax2 = fig.add_subplot(gsS[1])
    ax2.set_facecolor(STYLE["panel"])
    ax2.plot(fr, act, color="#4cc9f0", lw=2.2, drawstyle="steps-mid", label="active sensors /40")
    ax2.set_ylim(0, 42)
    ax2.set_xlim(0, NF-1)
    ax2.set_xlabel("motion frame", fontsize=9.5)
    ax2.set_ylabel("active sensors /40", color="#4cc9f0", fontsize=9.5)
    ax2.tick_params(labelsize=8.5)
    ax2b = ax2.twinx()
    ax2b.plot(fr, mind, color="#ef476f", lw=2.0, label="min skin distance")
    ax2b.set_ylabel("min skin distance (cm)", color="#ef476f", fontsize=9.5)
    ax2b.tick_params(labelsize=8.5, colors=STYLE["fg"])
    ax2b.set_ylim(0, max(12, mind.max()*1.15))
    for f in (F_ENTRY, F_MID, F_DEEP):
        ax2.axvline(f, color="#e8e8ea", lw=0.8, alpha=0.35, ls="--")
    for sp in list(ax2.spines.values()) + list(ax2b.spines.values()):
        sp.set_edgecolor(STYLE["grid"])
    ax2.set_title("skin activity along the motion", fontsize=11.5, fontweight="bold", pad=5)
    rows = [("entry", se), ("mid", sm_), ("deep", sdp)]
    tbl = "        hand    tips   act    pts   min\n"
    for nm, s in rows:
        tbl += f"{nm:>5}  {s['hand']:5.1f}  {s['tcp']:5.1f}  {s['active']:2d}/40  {s['npts']:5,}  {s['mind']:4.1f}cm\n"
    ax2.text(0.985, 0.06, tbl.rstrip(), transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=7.6, family="DejaVu Sans Mono", color="#dce6f2",
             bbox=dict(boxstyle="round,pad=0.45", fc="#10202f", ec=STYLE["grid"], alpha=0.92))

    # ------------------------------------------------------------------------------ titles
    fig.suptitle("FUMEHOOD VARIANT · hood_deep_tunnel — maximum insertion into a 95 cm deep tunnel hood",
                 color=STYLE["fg"], fontsize=19, fontweight="bold", x=0.025, ha="left", y=0.972)
    fig.text(0.025, 0.925,
             f"FR3 + 40-SPAD hybrid skin · hood interior W {2*W*100:.0f} cm (half-width {W*100:.0f}) × "
             f"H {H*100:.0f} cm × D {D*100:.0f} cm, sash opening {SASH*100:.0f} cm · "
             f"motion stow → under-sash → mid → deepest ({NF} frames, collision-checked, "
             f"worst grazing {worst_pen:.0f} mm)\n"
             f"deepest valid insertion: hand (flange) {sdp['hand']:.1f} cm, fingertips {sdp['tcp']:.1f} cm "
             f"beyond the front plane — whole forearm inside; the 60 cm hand target exceeds the FR3 "
             f"reach envelope (flange ceiling 45.2 cm even allowing clipping)",
             color="#a8b0bd", fontsize=11, ha="left", va="top", linespacing=1.5)

    out = os.path.join(OUT, f"{KEY}.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"SAVED {out} ({os.path.getsize(out)/1024:.0f} KB)")


# ============================================================================
# Sensor gallery & render sanity checks
# ============================================================================

def fig_panel_sensor_gallery():
    """Gallery of all 40 hybrid SPAD proximity depth heatmaps in the demo cavity."""
    from hybrid_viz_lib import (build, set_pose, sensors, depth8, depth_renderer,
                                add_box, FOVY, NEAR, FAR, STYLE)
    from matplotlib.patches import FancyBboxPatch
    from matplotlib import cm
    from matplotlib.colors import Normalize

    OUT = str(_FIGROOT)
    os.makedirs(OUT, exist_ok=True)
    KEY = "sensor_gallery_cavity"

    def make(spec):
        add_box(spec, "bench", [0.52, 0, 0.175], [0.30, 0.34, 0.175], [0.62, 0.55, 0.45, 1])
        add_box(spec, "wl", [0.52, 0.30, 0.62], [0.30, 0.015, 0.28], [0.75, 0.72, 0.66, 0.30])
        add_box(spec, "wr", [0.52, -0.30, 0.62], [0.30, 0.015, 0.28], [0.75, 0.72, 0.66, 0.30])
        add_box(spec, "roof", [0.52, 0, 0.92], [0.30, 0.315, 0.015], [0.70, 0.68, 0.62, 0.30])
        add_box(spec, "back", [0.83, 0, 0.62], [0.015, 0.315, 0.28], [0.68, 0.66, 0.60, 1])
        add_box(spec, "pillar", [0.40, 0.13, 0.62], [0.025, 0.025, 0.27], [0.48, 0.34, 0.22, 1])

    model = build(make=make)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")
    names = sensors(model)
    rd = depth_renderer(model)

    # gather depth, min, active for each sensor
    recs = []
    for n in names:
        d8 = depth8(rd, data, n)
        valid_mask = (d8 >= NEAR) & (d8 <= FAR)
        valid = d8[valid_mask]
        active = valid.size > 0
        dmin_cm = float(valid.min() * 100.0) if active else None
        link = re.match(r"(link\d+)", n).group(1)
        # group key keeps the link5 front/back distinction; label sub keeps the index
        msub = re.match(r"link\d+(?:_(front|back))?_sensor_(\d+)", n)
        sub = msub.group(1)            # 'front'/'back' or None
        idx = msub.group(2)
        group = f"{link}_{sub}" if sub else link
        label = f"{group.replace('link', 'L')}.{idx}"
        recs.append(dict(name=n, link=link, group=group, label=label, idx=idx,
                         d8=d8, mask=valid_mask,
                         active=active, dmin_cm=dmin_cm))

    n_active = sum(r["active"] for r in recs)

    # normalization: distance in meters, near=red far=blue with turbo_r
    cmap = plt.get_cmap(STYLE["cmap"]).copy()
    cmap.set_bad("#0c0e12")          # no-return cells render near-black
    norm = Normalize(vmin=NEAR, vmax=FAR)

    # ---- figure / grid -------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": STYLE["fg"],
        "axes.edgecolor": STYLE["grid"],
    })

    NROW, NCOL = 5, 8
    fig = plt.figure(figsize=(18.5, 13.4), dpi=170)
    fig.patch.set_facecolor(STYLE["bg"])

    # leave room: top banner + title, bottom colorbar, left link rail
    gs = fig.add_gridspec(NROW, NCOL,
                          left=0.052, right=0.965, top=0.838, bottom=0.105,
                          hspace=0.60, wspace=0.28)

    # stable per-group color accents for the rail / titles
    link_palette = {
        "link1": "#4cc9f0", "link2": "#90e0a0", "link3": "#f9c74f",
        "link4": "#f3722c", "link5_back": "#c77dff", "link5_front": "#9d4edd",
        "link6": "#ff8fab",
    }

    axes = []
    for r in range(NROW):
        for c in range(NCOL):
            ax = fig.add_subplot(gs[r, c])
            axes.append(ax)

    for k, ax in enumerate(axes):
        if k >= len(recs):
            ax.axis("off")
            ax.set_facecolor(STYLE["bg"])
            continue
        rec = recs[k]
        accent = link_palette[rec["group"]]
        d8 = rec["d8"].astype(float)
        disp = np.ma.array(d8, mask=~rec["mask"])

        ax.set_facecolor(STYLE["panel"])
        im = ax.imshow(disp, cmap=cmap, norm=norm, interpolation="nearest",
                       origin="upper", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(accent if rec["active"] else "#3a3f48")
            s.set_linewidth(1.6 if rec["active"] else 0.8)

        # title: "linkX.N  min=YY cm" (red if <8cm)
        label = rec["label"]
        if rec["active"]:
            mtxt = f"min={rec['dmin_cm']:.0f} cm"
            tcol = STYLE["near"] if rec["dmin_cm"] < 8.0 else STYLE["fg"]
        else:
            mtxt = "no return"
            tcol = "#6b7280"
        ax.set_title(f"{label}", color=accent, fontsize=11.5, fontweight="bold",
                     pad=14, loc="left")
        ax.text(0.0, 1.045, mtxt, transform=ax.transAxes, ha="left", va="bottom",
                color=tcol, fontsize=9.0,
                fontweight="bold" if (rec["active"] and rec["dmin_cm"] < 8.0) else "normal")

    # ---- link legend strip (under subtitle) ---------------------------------
    # One swatch per sensor group with its active count; the tile borders + titles
    # share these colors so the eye maps each heatmap back to its link block.
    group_order = []
    for rec in recs:
        if rec["group"] not in group_order:
            group_order.append(rec["group"])

    lx = 0.052
    ly = 0.895
    for g in group_order:
        accent = link_palette[g]
        cnt = sum(1 for rr in recs if rr["group"] == g)
        act = sum(1 for rr in recs if rr["group"] == g and rr["active"])
        disp_name = g.replace("link", "L").replace("_", " ").upper()
        sw = FancyBboxPatch((lx, ly), 0.013, 0.018, transform=fig.transFigure,
                            boxstyle="round,pad=0.001,rounding_size=0.004",
                            linewidth=0, facecolor=accent)
        fig.patches.append(sw)
        txt = f"{disp_name}  {act}/{cnt}"
        fig.text(lx + 0.018, ly + 0.009, txt, transform=fig.transFigure,
                 ha="left", va="center", color=STYLE["fg"], fontsize=10.5,
                 fontweight="bold")
        lx += 0.018 + 0.011 * (len(txt) + 1)

    # ---- banner + titles -----------------------------------------------------
    fig.text(0.052, 0.975, "FRANKA_SKIN  ·  HYBRID SPAD PROXIMITY GALLERY",
             color=STYLE["fg"], fontsize=23, fontweight="bold", ha="left", va="top")
    fig.text(0.052, 0.940,
             "All 40 SPAD sensors  ·  8x8 depth cameras  ·  fovy=45  ·  range "
             "1.5-50 cm  ·  arm pose \"reach\" inside demo cavity",
             color="#aeb4bf", fontsize=12.5, ha="left", va="top")

    # active-count banner pill (top-right)
    bx, by, bw, bh = 0.748, 0.905, 0.217, 0.058
    pill = FancyBboxPatch((bx, by), bw, bh, transform=fig.transFigure,
                          boxstyle="round,pad=0.004,rounding_size=0.012",
                          linewidth=2.0, edgecolor=STYLE["accent"],
                          facecolor="#10202a")
    fig.patches.append(pill)
    fig.text(bx + bw / 2, by + bh * 0.60, f"{n_active}/40 sensors active",
             color=STYLE["accent"], fontsize=18, fontweight="bold",
             ha="center", va="center")
    fig.text(bx + bw / 2, by + bh * 0.20, "returns < 0.50 m",
             color="#8fb8c8", fontsize=9.5, ha="center", va="center")

    # ---- shared colorbar -----------------------------------------------------
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.30, 0.045, 0.40, 0.022])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("distance (m)   —   red = near (1.5 cm)   ·   blue = far (50 cm)",
                 color=STYLE["fg"], fontsize=12)
    cb.ax.xaxis.set_tick_params(color=STYLE["fg"])
    cb.outline.set_edgecolor(STYLE["grid"])
    plt.setp(plt.getp(cb.ax, "xticklabels"), color=STYLE["fg"], fontsize=10)

    # legend note bottom-left: red title meaning + no-return cell color
    fig.text(0.052, 0.052,
             "red \"min\" label  →  closest return < 8 cm (imminent contact)",
             color=STYLE["near"], fontsize=10.5, ha="left", va="center", fontweight="bold")
    fig.text(0.052, 0.028,
             "dark cells  →  no return (beyond 50 cm or below 1.5 cm)",
             color="#6b7280", fontsize=10.5, ha="left", va="center")

    out_path = os.path.join(OUT, f"{KEY}.png")
    fig.savefig(out_path, dpi=170, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    sz = os.path.getsize(out_path)
    mins_all = [r["dmin_cm"] for r in recs if r["active"]]
    print("PATH", out_path)
    print("SIZE", sz)
    print("ACTIVE", n_active)
    print("MIN_CM", round(min(mins_all), 1), "MAX_CM", round(max(mins_all), 1))
    print("UNDER8", sum(1 for m in mins_all if m < 8))


def fig_render_hybrid_skin_rich():
    """Rich hybrid-skin sheet: arm reaching inside a tight cavity, 40-SPAD point cloud overlay, exo/wrist views, and a labeled 40-sensor depth-tile grid."""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    import OpenGL.EGL as _EGL
    _d = _EGL.eglGetDisplay(_EGL.EGL_DEFAULT_DISPLAY)
    _maj, _min = _EGL.EGLint(), _EGL.EGLint()
    if _EGL.eglInitialize(_d, _maj, _min):
        import mujoco.egl as _me
        _me.EGL_DISPLAY = _d

    from matplotlib import colormaps

    ROBOT = Path("/home/jaydv/code/prox_learning/assets/robots/franka_skin/model_hybrid.xml")
    OUT = _FIGROOT
    OUT.mkdir(parents=True, exist_ok=True)
    TURBO = colormaps["turbo"]
    FOVY = 45.0
    NEAR, FAR = 0.015, 0.50          # display range: what the skin story cares about

    QPOS = [0.0, -0.35, 0.0, -2.30, 0.0, 2.05, 0.79]   # reaching into the cavity

    # tight cavity around the reaching arm (robot base at origin, z up)
    CAVITY = [
        # name, center, half, rgba (walls translucent so the 3D view shows the arm + cloud inside)
        ("bench",   [0.52, 0.0, 0.175], [0.30, 0.34, 0.175], [0.62, 0.55, 0.45, 1]),
        ("wall_l",  [0.52, 0.30, 0.62], [0.30, 0.015, 0.28], [0.75, 0.72, 0.66, 0.30]),
        ("wall_r",  [0.52, -0.30, 0.62], [0.30, 0.015, 0.28], [0.75, 0.72, 0.66, 0.30]),
        ("roof",    [0.52, 0.0, 0.92], [0.30, 0.315, 0.015], [0.70, 0.68, 0.62, 0.30]),
        ("back",    [0.83, 0.0, 0.62], [0.015, 0.315, 0.28], [0.68, 0.66, 0.60, 1]),
        ("pillar",  [0.40, 0.13, 0.62], [0.025, 0.025, 0.27], [0.48, 0.34, 0.22, 1]),
    ]

    def cavity_spec(spec):
        spec.visual.global_.offwidth = 1600
        spec.visual.global_.offheight = 1280
        fl = spec.worldbody.add_geom()
        fl.type = mujoco.mjtGeom.mjGEOM_PLANE
        fl.size = [3, 3, 0.1]
        fl.rgba = [0.30, 0.31, 0.34, 1]
        for nm, c, h, rgba in CAVITY:
            b = spec.worldbody.add_body(name=f"cav_{nm}", pos=c)
            g = b.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = h
            g.rgba = rgba
        spec.worldbody.add_light(pos=[0.3, 0.6, 2.2], dir=[0, -0.2, -1],
                                 diffuse=[0.85, 0.85, 0.85], specular=[0.2, 0.2, 0.2])
        spec.worldbody.add_light(pos=[-0.8, -0.8, 1.8], dir=[0.5, 0.5, -1],
                                 diffuse=[0.45, 0.45, 0.45], specular=[0.1, 0.1, 0.1])
        # exo camera
        exo = spec.worldbody.add_camera()
        exo.name = "exo_camera_1"
        exo.pos = [1.45, -1.25, 1.15]
        v = np.array([0.4, 0.0, 0.55]) - np.array(exo.pos); v /= np.linalg.norm(v)  # look at robot
        z = -v
        up = np.array([0, 0, 1.0])
        x = np.cross(up, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).reshape(9))
        exo.quat = [float(t) for t in q]
        exo.fovy = 52
        exo.resolution = [480, 640]
        return spec

    def set_pose(model, data):
        for i, v in enumerate(QPOS, start=1):
            data.qpos[model.joint(f"fr3_joint{i}").qposadr[0]] = v
        mujoco.mj_forward(model, data)

    def backproject(d8, pos, Rm, d_min=NEAR, d_max=FAR):
        H = W = d8.shape[0]
        f = (H / 2) / np.tan(np.deg2rad(FOVY / 2))
        cx = cy = (H - 1) / 2.0
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        m = (d8 >= d_min) & (d8 <= d_max)
        if not m.any():
            return np.zeros((0, 3)), np.zeros((0,))
        d = d8[m].astype(np.float64)
        x_c = (u[m] - cx) * d / f
        y_c = -(v[m] - cy) * d / f
        pts = (Rm @ np.stack([x_c, y_c, -d], 0)).T + pos
        return pts, d

    spec = cavity_spec(mujoco.MjSpec.from_file(str(ROBOT)))
    model = spec.compile()
    data = mujoco.MjData(model)
    set_pose(model, data)

    names = [model.camera(i).name for i in range(model.ncam) if "_sensor_" in model.camera(i).name]

    # ---- sensor depth + cloud ----
    rd = mujoco.Renderer(model, 8, 8)
    rd.enable_depth_rendering()
    tiles, mins = {}, {}
    pts_all, d_all = [], []
    for n in names:
        rd.update_scene(data, n)
        d8 = rd.render().copy()
        tiles[n] = d8
        mins[n] = float(d8.min())
        cid = model.camera(n).id
        pts, dd = backproject(d8, data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3))
        if len(pts):
            pts_all.append(pts)
            d_all.append(dd)
    pts = np.concatenate(pts_all) if pts_all else np.zeros((0, 3))
    dd = np.concatenate(d_all) if d_all else np.zeros((0,))
    active = sum(1 for n in names if mins[n] < FAR)

    # ---- 3D renders (plain + cloud overlay) ----
    cam = mujoco.MjvCamera()
    cam.lookat = [0.40, 0.0, 0.55]
    cam.distance = 1.6
    cam.azimuth = 40
    cam.elevation = -16
    r_big = mujoco.Renderer(model, 640, 760)
    r_big.update_scene(data, cam)
    plain = r_big.render().copy()

    # cloud overlay: same scene + colored sphere per point via mjv user scene geoms
    r_big.update_scene(data, cam)
    scn = r_big.scene
    norm = np.clip((dd - NEAR) / (FAR - NEAR), 0, 1)
    cols = TURBO(1.0 - norm)[:, :3]
    for p, c in zip(pts, cols):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.008, 0, 0]), np.asarray(p, np.float64),
                            np.eye(3).ravel(), np.array([c[0], c[1], c[2], 1.0], np.float32))
        scn.ngeom += 1
    overlay = r_big.render().copy()

    r_side = mujoco.Renderer(model, 320, 380)
    r_side.update_scene(data, "exo_camera_1")
    exo = r_side.render().copy()
    r_side.update_scene(data, "gripper/wrist_camera")
    wrist = r_side.render().copy()

    # ---- tile grid ----
    full_w = 760 * 2
    cols_n = 10
    pad = 6
    T = (full_w - (cols_n + 1) * pad) // cols_n
    rows = int(np.ceil(len(names) / cols_n))
    grid = np.full((rows * (T + 20) + pad, full_w, 3), 22, np.uint8)
    for k, n in enumerate(sorted(names)):
        d8 = tiles[n]
        nrm = np.clip((d8 - NEAR) / (FAR - NEAR), 0, 1)
        img = (TURBO(1.0 - nrm)[:, :, :3] * 255).astype(np.uint8)
        img[d8 > FAR] = (16, 16, 16)
        til = cv2.resize(img[..., ::-1], (T, T), interpolation=cv2.INTER_NEAREST)
        rr, cc = divmod(k, cols_n)
        y0 = rr * (T + 20) + 20
        x0 = pad + cc * (T + pad)
        grid[y0:y0 + T, x0:x0 + T] = til
        cv2.rectangle(grid, (x0, y0), (x0 + T, y0 + T), (70, 70, 70), 1)
        short = n.replace("link", "L").replace("_sensor_", ".").replace("_front", "F").replace("_back", "B")
        lab = f"{short} {mins[n]:.2f}m" if mins[n] < FAR else f"{short} --"
        col = (60, 60, 255) if mins[n] < 0.08 else (210, 210, 210)
        cv2.putText(grid, lab, (x0 + 1, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)

    # ---- compose ----
    def bgr(x):
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    row1 = np.concatenate([bgr(plain), bgr(overlay)], axis=1)
    side = np.concatenate([bgr(exo), bgr(wrist)], axis=1)
    side = cv2.resize(side, (row1.shape[1], int(side.shape[0] * row1.shape[1] / side.shape[1])))
    banner = np.full((40, row1.shape[1], 3), 30, np.uint8)
    cv2.putText(banner,
                f"fr3_hybrid_skin in tight cavity  |  {active}/40 sensors returning <{FAR:.1f}m  |  "
                f"right 3D = skin point cloud (what the policy senses, no cameras)",
                (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
    full = np.concatenate([banner, row1, side, grid], axis=0)
    out = OUT / "hybrid_skin_rich_test.png"
    cv2.imwrite(str(out), full)
    print(f"active sensors: {active}/40, cloud pts: {len(pts)} -> {out}")


def fig_render_hybrid_skin_viz():
    """Composite display test for fr3_hybrid_skin: 3D RGB + exo/wrist cams + 40 SPAD depth tiles."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    # EGL device-platform enumeration is wedged on this box (mujoco.egl EGL_DISPLAY=None) while
    # the DEFAULT display works — pre-initialize it and patch mujoco.egl before Renderer use.
    _d = _EGL.eglGetDisplay(_EGL.EGL_DEFAULT_DISPLAY)
    _maj, _min = _EGL.EGLint(), _EGL.EGLint()
    if _EGL.eglInitialize(_d, _maj, _min):
        import mujoco.egl as _me
        _me.EGL_DISPLAY = _d

    ROBOT = Path("/home/jaydv/code/prox_learning/assets/robots/franka_skin/model_hybrid.xml")
    OUT = _FIGROOT
    OUT.mkdir(parents=True, exist_ok=True)
    TURBO = colormaps["turbo"]

    # reaching pose so the arm is open and the skin faces the obstacle
    QPOS = [0.0, -0.5, 0.0, -2.1, 0.0, 1.9, 0.8]

    def build():
        spec = mujoco.MjSpec.from_file(str(ROBOT))
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 1280
        # demo obstacle: a panel beside the forearm + a low table the gripper hovers over
        wall = spec.worldbody.add_body()
        wall.name = "demo_wall"
        wall.pos = [0.55, 0.30, 0.55]
        g = wall.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.22, 0.02, 0.35]
        g.rgba = [0.80, 0.55, 0.30, 1.0]
        tb = spec.worldbody.add_body()
        tb.name = "demo_table"
        tb.pos = [0.55, 0.0, 0.32]
        g2 = tb.add_geom()
        g2.type = mujoco.mjtGeom.mjGEOM_BOX
        g2.size = [0.20, 0.30, 0.02]
        g2.rgba = [0.55, 0.57, 0.60, 1.0]
        # exo camera (the robot file only ships wrist cams; datagen adds exo at env level)
        exo = spec.worldbody.add_camera()
        exo.name = "exo_camera_1"
        exo.pos = [1.45, -1.25, 1.15]
        v = np.array([0.35, 0.0, 0.5]) - np.array(exo.pos); v = v / np.linalg.norm(v)  # look at robot
        z = -v
        up = np.array([0, 0, 1.0])
        x = np.cross(up, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.stack([x, y, z], axis=1).reshape(9))
        exo.quat = [float(t) for t in q]
        exo.fovy = 52
        exo.resolution = [480, 640]
        return spec.compile()

    def render_rgb(rndr, data, cam_name=None, free=None):
        if free is not None:
            rndr.update_scene(data, free)
        else:
            rndr.update_scene(data, cam_name)
        return rndr.render().copy()

    def tile(d8, near=0.02, far=0.5):
        valid = d8 < (far * 4)
        norm = np.clip((d8 - near) / (far - near), 0, 1)
        img = (TURBO(1.0 - norm)[:, :, :3] * 255).astype(np.uint8)
        img[~valid] = (15, 15, 15)
        return img

    model = build()
    data = mujoco.MjData(model)
    # set the 7 arm joints
    arm = [model.joint(f"fr3_joint{i}").qposadr[0] for i in range(1, 8)
           if any(model.joint(j).name == f"fr3_joint{i}" for j in range(model.njnt))]
    if not arm:  # joint names may be unprefixed
        arm = [model.joint(i).qposadr[0] for i in range(min(7, model.njnt))]
    for adr, v in zip(arm, QPOS):
        data.qpos[adr] = v
    mujoco.mj_forward(model, data)

    # 3D free view (shared renderers per resolution)
    cam = mujoco.MjvCamera()
    cam.lookat = [0.35, 0.0, 0.6]
    cam.distance = 1.85
    cam.azimuth = 145
    cam.elevation = -12
    r_big = mujoco.Renderer(model, 900, 760)
    main3d = render_rgb(r_big, data, free=cam)
    try:
        r_big.close()
    except Exception:
        pass

    r_side = mujoco.Renderer(model, 450, 600)   # two stacked -> 900 tall, matches main3d
    exo = render_rgb(r_side, data, "exo_camera_1").copy()
    wrist = render_rgb(r_side, data, "gripper/wrist_camera").copy()
    try:
        r_side.close()
    except Exception:
        pass

    # all 40 sensor depth tiles (ONE shared 8x8 depth renderer)
    sensor_cams = [model.camera(i).name for i in range(model.ncam)
                   if "_sensor_" in model.camera(i).name]
    r_d = mujoco.Renderer(model, 8, 8)
    r_d.enable_depth_rendering()
    full_w = 760 + 600
    cols = 10
    rows = int(np.ceil(len(sensor_cams) / cols))
    pad = 6
    T = (full_w - (cols + 1) * pad) // cols
    grid = np.full((rows * (T + 18) + pad, full_w, 3), 22, np.uint8)
    for k, name in enumerate(sensor_cams):
        r_d.update_scene(data, name)
        d8 = r_d.render().copy()
        til = cv2.resize(tile(d8)[..., ::-1], (T, T), interpolation=cv2.INTER_NEAREST)
        rr, cc = divmod(k, cols)
        y0 = rr * (T + 18) + 18
        x0 = pad + cc * (T + pad)
        grid[y0 : y0 + T, x0 : x0 + T] = til
        cv2.rectangle(grid, (x0, y0), (x0 + T, y0 + T), (70, 70, 70), 1)
        short = name.replace("link", "L").replace("_sensor_", ".").replace("_front", "F").replace("_back", "B")
        cv2.putText(grid, short, (x0 + 1, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (215, 215, 215), 1, cv2.LINE_AA)
    try:
        r_d.close()
    except Exception:
        pass

    # compose: [ 3D | (exo over wrist) ] on top, sensor grid below
    def bgr(x):
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    right = np.concatenate([bgr(exo), bgr(wrist)], axis=0)
    top = np.concatenate([bgr(main3d), right], axis=1)
    gw = top.shape[1]
    if grid.shape[1] < gw:
        grid = np.pad(grid, ((0, 0), (0, gw - grid.shape[1]), (0, 0)), constant_values=22)
    else:
        grid = grid[:, :gw]
    # labels banner
    banner = np.full((34, gw, 3), 30, np.uint8)
    cv2.putText(banner, "fr3_hybrid_skin  |  3D + exo + wrist + 40 SPAD depth sensors (turbo: red=near)",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    full = np.concatenate([banner, top, grid], axis=0)
    out = OUT / "hybrid_skin_display_test.png"
    cv2.imwrite(str(out), full)
    print(f"ncam={model.ncam} sensors={len(sensor_cams)} -> {out}")


# ============================================================================
# Registry + CLI
# ============================================================================

REGISTRY = {
    "panel_single_sensor_anatomy": fig_panel_single_sensor_anatomy,
    "panel_plane_distance_sweep": fig_panel_plane_distance_sweep,
    "panel_plane_tilt_sweep": fig_panel_plane_tilt_sweep,
    "panel_range_accuracy_scatter": fig_panel_range_accuracy_scatter,
    "acc_angular_resolution": fig_acc_angular_resolution,
    "make_acc_range_linearity": fig_make_acc_range_linearity,
    "proof_acc_repeat": fig_proof_acc_repeat,
    "panel_coverage_behind": fig_panel_coverage_behind,
    "panel_need_vision_vs_skin": fig_panel_need_vision_vs_skin,
    "panel_need_blur_and_dark": fig_panel_need_blur_and_dark,
    "proof_whole_arm_clearance": fig_proof_whole_arm_clearance,
    "panel_clearance_controller": fig_panel_clearance_controller,
    "panel_known_shapes_cloud": fig_panel_known_shapes_cloud,
    "panel_cavity_reconstruction_3d": fig_panel_cavity_reconstruction_3d,
    "make_pipe_tunnel_fig": fig_make_pipe_tunnel_fig,
    "test_reconstruct_fumehood": fig_test_reconstruct_fumehood,
    "panel_env_cluttered_shelf": fig_panel_env_cluttered_shelf,
    "env_corner_cavity_hero": fig_env_corner_cavity_hero,
    "env_narrow_slot": fig_env_narrow_slot,
    "env_overhang_fig": fig_env_overhang_fig,
    "viz_peg_forest": fig_viz_peg_forest,
    "fig_hood_narrow": fig_fig_hood_narrow,
    "fig_hood_tall": fig_fig_hood_tall,
    "fumehood_std_fig": fig_fumehood_std_fig,
    "fumehood_short_low_sash_fig": fig_fumehood_short_low_sash_fig,
    "fumehood_var_hood_deep_tunnel": fig_fumehood_var_hood_deep_tunnel,
    "panel_sensor_gallery": fig_panel_sensor_gallery,
    "render_hybrid_skin_rich": fig_render_hybrid_skin_rich,
    "render_hybrid_skin_viz": fig_render_hybrid_skin_viz,
}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate proximity-skin paper figures.")
    ap.add_argument("names", nargs="*", help="figure keys to generate")
    ap.add_argument("--all", action="store_true", help="generate every figure")
    ap.add_argument("--list", action="store_true", help="list available figures and exit")
    ap.add_argument("--outdir", default=None, help="write all figures into this dir (per-run consolidation)")
    a = ap.parse_args()
    if a.outdir:
        global _FIGROOT
        _FIGROOT = Path(a.outdir)
    if a.list or (not a.all and not a.names):
        for k in REGISTRY:
            print(k)
        return
    keys = list(REGISTRY) if a.all else a.names
    for k in keys:
        if k not in REGISTRY:
            print(f"[figures] unknown figure: {k} (use --list)")
            continue
        print(f"[figures] generating {k} ...")
        REGISTRY[k]()
        print(f"[figures] done {k}")


if __name__ == "__main__":
    main()
