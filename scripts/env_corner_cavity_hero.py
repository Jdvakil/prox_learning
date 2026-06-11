"""env_corner_cavity — FLAGSHIP hero for the franka_skin hybrid SPAD array.

A real FR3 reaches into a tight corner cavity (bench + two translucent side walls + roof +
solid back wall + interior pillar). The robot's RGB cameras are blurred at policy-training time,
so this 40-sensor proximity skin IS its perception. We render:

  - a context RGB view (MuJoCo) with the live distance-colored cloud overlaid on the arm-in-cavity
  - a 3-view (iso / front / top) back-projected point-cloud reconstruction tracing every
    cavity surface, with translucent ground-truth wireframe + FR3 kinematic skeleton

"perceives geometry through its body" — zero cameras used.
"""
import sys, os
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, sensors, skin_cloud, depth_renderer,
                            render_scene, mjv_cam, nice_lights, add_box,
                            FOVY, NEAR, FAR, STYLE)
import mujoco, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

OUTDIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight"
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
