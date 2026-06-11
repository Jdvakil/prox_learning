"""env_overhang: Franka FR3 hybrid SPAD skin reaching UNDER a low overhang/shelf ceiling
toward a target on the table. Top-facing SPADs read the ceiling clearance, bottom-facing SPADs
read the table -- the skin senses ABOVE and BELOW simultaneously, something a wrist RGB cam cannot.
"""
import sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
import os
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from hybrid_viz_lib import (build, set_pose, sensors, cam_pose, nice_lights, add_box,
                            skin_cloud, depth_renderer, depth8, backproject,
                            render_scene, mjv_cam, FAR, NEAR, FOVY, STYLE)

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight"
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

fig.text(0.035, 0.012, "MuJoCo EGL · back-projected depth, world frame · diagnostics_output/20260611_hybrid_overnight",
         fontsize=8.5, color="#6b7280", ha="left")

out_png = os.path.join(OUT, f"{KEY}.png")
fig.savefig(out_png, facecolor=S["bg"], dpi=170)
plt.close(fig)
sz = os.path.getsize(out_png)
print("WROTE", out_png, sz, "bytes", "active", active, "up", len(up_s), "dn", len(dn_s),
      "ceil_mm", round(ceil_clear*1000, 1), "table_mm", round(table_clear*1000, 1))
EOF_MARKER_NONE = None
