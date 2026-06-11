"""PROOF PANEL: need_coverage_behind
A small obstacle is placed on the FAR side of the forearm/elbow, in a direction the wrist camera
(which looks toward the gripper) physically cannot see. We draw the wrist-cam viewing frustum, show
the obstacle sitting well outside it, and show the forearm SPAD skin sensors lighting up on it.
We quantify the 360-deg argument: what fraction of the arm's surrounding directions is covered by
the 40-sensor skin vs the single wrist camera.

  exo hero render with wrist frustum + skin cloud   |   directional-coverage polar (skin vs wrist)
  per-sensor obstacle response (wrist-blind region)  |   stats / the 360 argument
"""
import sys, os
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, sensors, depth8, backproject, cam_pose,
    depth_renderer, skin_cloud, FOVY, NEAR, FAR, STYLE)
import mujoco, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

OUTDIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight"
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
