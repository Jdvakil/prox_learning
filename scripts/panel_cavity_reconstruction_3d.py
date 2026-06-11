import sys, os
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, sensors, depth8, backproject, cam_pose,
    add_box, depth_renderer, FOVY, NEAR, FAR, STYLE)
import mujoco, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

OUTDIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_viz_suite"
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
from matplotlib.lines import Line2D
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
