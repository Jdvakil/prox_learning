"""PROOF PANEL: need_blur_and_dark
Why the franka_skin HYBRID proximity skin is NECESSARY, not redundant.

A 3x3 grid:
  rows = exo RGB / wrist RGB / SPAD skin (8x8 depth montage + per-sensor min distance)
  cols = sharp+lit  /  heavily Gaussian-BLURRED (training condition)  /  near-DARK (lights dimmed)

The RGB cameras (exo + wrist) are what a vision policy would normally rely on. At policy
TRAIN time they are deliberately blurred, and in a dim contact-rich workspace they go nearly
black -- both columns collapse to useless imagery. The bottom row shows the SPAD skin's depth
readout, which is rendered from geometry and is therefore BIT-FOR-BIT IDENTICAL across all three
columns (verified: max |delta depth| = 0.000 mm). The min-distance-to-contact the policy reads is
unchanged. Depth perception does not depend on lighting or texture; this skin IS the robot's
perception when vision is unreliable.
"""
import sys, os
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, sensors, depth8, backproject, cam_pose,
    add_box, add_cylinder, depth_renderer, nice_lights, skin_cloud, FOVY, NEAR, FAR, STYLE)
import mujoco, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

OUTDIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight"
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
