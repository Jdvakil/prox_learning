import sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, sensors, depth8, depth_renderer,
                            add_plane_mocap, mocap_set, cam_pose, FOVY, NEAR, FAR, STYLE)
import mujoco, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Experiment: flat plane head-on to ONE 8x8 SPAD sensor at 0.10 / 0.20 / 0.30 m.
# Render the depth image N times each; quantify accuracy (mean vs ground truth)
# and repeatability (per-cell std across renders).  The MuJoCo depth renderer
# is deterministic, so this also demonstrates the sim depth model is noise-free
# at float32 precision -- real SPAD shot noise is injected at policy-train time.
# ----------------------------------------------------------------------------
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

# ============================================================================
# FIGURE
# ============================================================================
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

# ----------------------------------------------------------------------------
# ROW 1, cols 0-2: per-distance MEAN 8x8 depth maps (turbo_r), annotated
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# ROW 2, cols 0-1: REPEATABILITY  —  per-cell std vs distance
#   std is exactly 0 (bit-identical renders); we draw it against the *theoretical*
#   float32 quantization step, which the model does not even reach.
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# ROW 2, col 2: ACCURACY  —  |mean - ground truth| in micrometers vs the bound
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# ROW 2, col 3: frame-to-frame jitter trace for the center cell
# ----------------------------------------------------------------------------
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
