"""known_shapes_cloud panel: full 40-sensor back-projected point cloud tracing a
mocap SPHERE and a static right-angle CORNER placed within ~0.25 m of the forearm.
Cloud colored by distance (turbo_r), GT sphere wireframe + corner edges overlaid.
"""
import os, sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, sensors, depth8, backproject, cam_pose,
    add_box, add_sphere_mocap, mocap_set, depth_renderer, FOVY, NEAR, FAR, STYLE)
import mujoco, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_viz_suite"
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
