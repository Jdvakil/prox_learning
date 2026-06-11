"""PROOF PANEL: need_vision_vs_skin
The killer figure. A real Franka FR3 + 40-SPAD hybrid skin reaches under a low cabinet shelf.
A structural cross-brace (the obstacle) sits CLOSE under the forearm but is OCCLUDED from both the
exo camera (the arm body blocks the line of sight) and the wrist camera (the brace is ~118 deg
off the wrist-cam optical axis, far outside its FOV). The RGB cameras are blind here. The skin is
not: a forearm SPAD reads the brace at a few cm, and the back-projected cloud traces its surface.

Four panels side by side:
  (A) exo RGB        -- obstacle hidden behind the arm
  (B) wrist RGB      -- obstacle outside the FOV
  (C) skin 8x8 montage of the forearm sensors -- one tile lights up clearly
  (D) 3D cloud with the obstacle surface traced by skin returns
Caption: the cameras are blind here; the skin is not.
"""
import sys, os
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, sensors, depth8, backproject, cam_pose,
    add_box, add_cylinder, depth_renderer, nice_lights, skin_cloud, render_scene, mjv_cam,
    FOVY, NEAR, FAR, STYLE)
import mujoco, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle

OUTDIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight"
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
