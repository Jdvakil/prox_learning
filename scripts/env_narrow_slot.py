"""env_narrow_slot — hero viz of the franka_skin HYBRID proximity skin threading a narrow
vertical slot. A real FR3 reaches between two parallel walls; the 40 SPAD depth cams paint
BOTH flanks and report per-side clearance. Top-lab figure aesthetic."""
import sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (
    build, set_pose, sensors, skin_cloud, add_box, nice_lights, cam_pose,
    render_scene, mjv_cam, FOVY, NEAR, FAR, STYLE,
)
import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import os

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight"
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
