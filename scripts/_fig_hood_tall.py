import sys, os
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, skin_cloud, render_scene, mjv_cam,
                            nice_lights, add_box, depth_renderer, FAR, STYLE)
import mujoco, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_fumehood_variations"
os.makedirs(OUT, exist_ok=True)

W, H, D, SASH = 0.34, 0.85, 0.55, 0.55
BZ, X0 = 0.585, 0.35

def mk(s):
    nice_lights(s)
    add_box(s, "bench", [X0 + D / 2, 0, BZ - 0.015], [D / 2 + 0.05, W + 0.05, 0.015], [0.62, 0.55, 0.45, 1])
    add_box(s, "bench_body", [X0 + D / 2, 0, BZ / 2 - 0.02], [D / 2, W, BZ / 2 - 0.02], [0.55, 0.5, 0.44, 1])
    add_box(s, "wall_l", [X0 + D / 2, W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30])
    add_box(s, "wall_r", [X0 + D / 2, -W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30])
    add_box(s, "back", [X0 + D, 0, BZ + H / 2], [0.012, W, H / 2], [0.72, 0.7, 0.66, 1])
    add_box(s, "top", [X0 + D / 2, 0, BZ + H], [D / 2, W, 0.012], [0.78, 0.8, 0.84, 0.30])
    add_box(s, "sash", [X0, 0, BZ + SASH + 0.028], [0.012, W, 0.028], [0.62, 0.64, 0.66, 1])
    add_box(s, "target", [X0 + 0.7 * D, 0.0, BZ + 0.045], [0.04, 0.04, 0.045], [0.85, 0.5, 0.3, 1])

model = build(mk)
data = mujoco.MjData(model)
HAND = model.body("gripper/base").id
rd = depth_renderer(model)

# hand-tuned waypoints (verified by FK in _tune_hood_tall.py)
WPTS = [
    ("stow",      [0.0, -0.800, 0.0, -2.600, 0.0, 1.900, 0.79]),
    ("entry",     [0.0, -0.535, 0.0, -1.855, 0.0, 2.187, 0.79]),  # hand (0.400, 0.820)
    ("mid",       [0.0, -0.085, 0.0, -1.479, 0.0, 1.953, 0.79]),  # hand (0.550, 0.760)
    ("deep_low",  [0.0,  0.450, 0.0, -0.905, 0.0, 1.907, 0.79]),  # hand (0.700, 0.700)
    ("deep_high", [0.0,  0.513, 0.0, -0.470, 0.0, 2.400, 0.79]),  # hand (0.620, 0.920)
]
NSEG = 12
qs, marks = [], {}
for k in range(len(WPTS) - 1):
    a, b = np.array(WPTS[k][1]), np.array(WPTS[k + 1][1])
    for t in np.linspace(0, 1, NSEG, endpoint=False):
        s = 0.5 - 0.5 * np.cos(np.pi * t)  # ease in/out
        qs.append(a + s * (b - a))
marks = {WPTS[k][0]: k * NSEG for k in range(len(WPTS))}
qs.append(np.array(WPTS[-1][1]))
marks[WPTS[-1][0]] = len(qs) - 1

# ---- per-frame stats over the whole motion ----------------------------------------------
stats = dict(depth=[], hz=[], act=[], npts=[], mind=[])
for q in qs:
    set_pose(model, data, q)
    p = data.xpos[HAND]
    pts, dd, mins = skin_cloud(model, data, rd)
    mn = min(mins.values())
    stats["depth"].append((p[0] - X0) * 100.0)
    stats["hz"].append(p[2])
    stats["act"].append(sum(1 for v in mins.values() if v < FAR))
    stats["npts"].append(len(pts))
    stats["mind"].append(mn * 100.0 if mn < FAR else np.nan)
for k in stats:
    stats[k] = np.array(stats[k])

# ---- renders at key frames ---------------------------------------------------------------
cam_main = mjv_cam(lookat=(0.42, 0.0, 0.80), distance=2.05, azimuth=-38, elevation=-14)
cam_side = mjv_cam(lookat=(0.50, 0.0, 0.82), distance=1.75, azimuth=90, elevation=-8)
cam_close = mjv_cam(lookat=(0.58, 0.0, 0.86), distance=1.25, azimuth=70, elevation=-6)

def frame(qi, cam):
    set_pose(model, data, qs[qi])
    pts, dd, mins = skin_cloud(model, data, rd)
    img = render_scene(model, data, cam, w=900, h=760, cloud=pts, depths=dd)
    return img

panels = [
    ("entry — under the sash",            marks["entry"],     cam_main),
    ("mid insertion",                     marks["mid"],       cam_main),
    ("DEEPEST — low interior (cutaway)",  marks["deep_low"],  cam_side),
    ("deep + wrist HIGH interior",        marks["deep_high"], cam_main),
    ("deep_high — interior close-up",     marks["deep_high"], cam_close),
]

# ---- compose figure ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": STYLE["bg"], "axes.facecolor": STYLE["panel"],
    "axes.edgecolor": STYLE["grid"], "text.color": STYLE["fg"],
    "axes.labelcolor": STYLE["fg"], "xtick.color": STYLE["fg"],
    "ytick.color": STYLE["fg"], "font.family": "DejaVu Sans",
})
fig, axes = plt.subplots(2, 3, figsize=(21, 12.4), dpi=150)
fig.suptitle("hood_tall — TALL fume hood (huge opening, high ceiling)   "
             f"W={W*100:.0f} cm   H={H*100:.0f} cm   D={D*100:.0f} cm   SASH={SASH*100:.0f} cm   "
             "|   FR3 + 40-SPAD hybrid proximity skin — deep insertion at two interior heights",
             fontsize=15.5, fontweight="bold", color=STYLE["fg"], y=0.985)

for ax, (label, qi, cam) in zip(axes.flat[:5], panels):
    img = frame(qi, cam)
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    d, z, a, n, m = (stats["depth"][qi], stats["hz"][qi], stats["act"][qi],
                     stats["npts"][qi], stats["mind"][qi])
    ax.set_title(f"{label}", fontsize=12.5, fontweight="bold", color=STYLE["accent"], pad=7)
    ax.text(0.02, 0.02,
            f"insertion {d:+.1f} cm   hand z {z:.2f} m\n"
            f"active {a}/40   pts {n}   min {m:.1f} cm",
            transform=ax.transAxes, fontsize=10.5, color=STYLE["fg"], va="bottom",
            bbox=dict(facecolor="#000000", alpha=0.55, edgecolor="none", pad=4))
    for sp in ax.spines.values():
        sp.set_color(STYLE["grid"])

# ---- stats panel -------------------------------------------------------------------------
ax = axes.flat[5]
fr = np.arange(len(qs))
ax.axhline((0.55 * D) * 100, color=STYLE["near"], lw=1.2, ls="--")
ax.text(1, (0.55 * D) * 100 + 0.7, f"target depth {0.55*D*100:.1f} cm", fontsize=9,
        color=STYLE["near"])
ax.fill_between(fr, 0, stats["depth"].clip(min=0), color=STYLE["accent"], alpha=0.18)
ax.plot(fr, stats["depth"], color=STYLE["accent"], lw=2.4, label="insertion depth (cm)")
ax.plot(fr, stats["mind"], color="#ffd166", lw=2.0, label="min skin dist (cm)")
ax2 = ax.twinx()
ax2.plot(fr, stats["act"], color="#06d6a0", lw=2.0, label="active sensors /40")
ax2.set_ylim(0, 42)
ax2.tick_params(colors=STYLE["fg"])
ax2.set_ylabel("active sensors", color="#06d6a0", fontsize=10)
for name, qi in marks.items():
    ax.axvline(qi, color=STYLE["grid"], lw=0.9)
    ax.text(qi, ax.get_ylim()[1] * 0.97, name, rotation=90, fontsize=8,
            color="#9aa0aa", va="top", ha="right")
ax.set_xlabel("motion frame", fontsize=10)
ax.set_ylabel("cm", fontsize=10)
ax.grid(color=STYLE["grid"], lw=0.5, alpha=0.6)
ln1, lb1 = ax.get_legend_handles_labels()
ln2, lb2 = ax2.get_legend_handles_labels()
ax.legend(ln1 + ln2, lb1 + lb2, loc="upper left", fontsize=9, facecolor=STYLE["panel"],
          edgecolor=STYLE["grid"], labelcolor=STYLE["fg"])
ax.set_title("per-frame skin stats along the insertion motion", fontsize=12.5,
             fontweight="bold", color=STYLE["accent"], pad=7)

rows = []
for name, qi in marks.items():
    rows.append(f"{name:<10s} depth {stats['depth'][qi]:>6.1f} cm  z {stats['hz'][qi]:.2f} m  "
                f"act {stats['act'][qi]:>2d}/40  pts {stats['npts'][qi]:>4d}  "
                f"min {stats['mind'][qi]:>4.1f} cm")
ax.text(0.985, 0.03, "\n".join(rows), transform=ax.transAxes, fontsize=8.3,
        family="DejaVu Sans Mono", color=STYLE["fg"], va="bottom", ha="right",
        bbox=dict(facecolor="#000000", alpha=0.55, edgecolor="none", pad=5))

fig.tight_layout(rect=[0, 0, 1, 0.965])
png = os.path.join(OUT, "hood_tall.png")
fig.savefig(png, facecolor=STYLE["bg"])
plt.close(fig)

# ---- verification numbers ----------------------------------------------------------------
need_x = X0 + 0.55 * D
qi_low, qi_high = marks["deep_low"], marks["deep_high"]
print("PNG:", png, os.path.getsize(png), "bytes")
print(f"need depth >= {(need_x-X0)*100:.2f} cm; band z=[{BZ+0.05:.3f},{BZ+SASH-0.05:.3f}]")
for name, qi in marks.items():
    print(f"{name:<10s} depth={stats['depth'][qi]:6.2f}cm z={stats['hz'][qi]:.3f} "
          f"act={stats['act'][qi]}/40 pts={stats['npts'][qi]} min={stats['mind'][qi]:.2f}cm")
print("max depth over motion:", stats["depth"].max(), "cm")
print("INSERTION TARGET MET:", stats["depth"][qi_low] >= (need_x - X0) * 100 and
      BZ + 0.05 <= stats["hz"][qi_low] <= BZ + SASH - 0.05)
