"""PROOF PANEL: acc_angular_resolution
Two thin vertical rods in front of ONE 8x8 SPAD sensor at 0.15 m range. Vary the gap
between them and ask: at what separation do the two rods stop merging into one blob and
become two distinct depth minima? That separation IS the angular resolution of an
8x8 / 45deg SPAD at 0.15 m. Honest about the coarse 8x8 sampling.
"""
import sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, cam_pose, depth_renderer, depth8,
                            add_cylinder, render_scene, mjv_cam, nice_lights,
                            FOVY, NEAR, FAR, STYLE)
import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.gridspec import GridSpec
import os

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight"
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
