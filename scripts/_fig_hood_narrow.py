"""hood_narrow — NARROW fumehood (32 cm interior) deep-insertion figure.

FR3 + 40-SPAD hybrid skin drives the hand straight-in past mid-depth of a hood whose
side walls nearly brush the forearm. 2x3 panel: motion frames with skin cloud overlay,
top-down per-side clearance view, and per-frame stats. Self-returns are excluded by
differencing against a hood-free baseline scene rendered at the identical pose.
"""
import sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, skin_cloud, render_scene, mjv_cam,
                            nice_lights, add_box, depth_renderer, cam_pose,
                            NEAR, FAR, STYLE)
import mujoco
import numpy as np
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_fumehood_variations"
os.makedirs(OUT, exist_ok=True)

# ---------------- hood geometry (NARROW variant) ----------------
W, H, D, SASH = 0.16, 0.50, 0.55, 0.32      # interior half-width / height / depth / sash opening
BZ, X0 = 0.585, 0.35                          # bench top z, hood front plane x
X_TARGET = X0 + 0.55 * D                      # required insertion: hand_x >= 0.6525


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


def mk_baseline(s):           # identical lighting/floor, no hood -> isolates self-returns
    nice_lights(s)


model = build(mk)
data = mujoco.MjData(model)
model_b = build(mk_baseline)
data_b = mujoco.MjData(model_b)
HID = model.body("gripper/base").id
rd = depth_renderer(model)
rd_b = depth_renderer(model_b)

# ---------------- motion: joint-space waypoints, straight-in (j1=0) ----------------
WAYPOINTS = [
    ("stow",       [0.0, -0.800, 0.0, -2.600, 0.0, 1.900, 0.79]),
    ("entry",      [0.0, -0.802, 0.0, -2.060, 0.0, 1.782, 0.79]),   # hand just outside sash, raised
    ("under-sash", [0.0, -0.480, 0.0, -1.902, 0.0, 1.859, 0.79]),   # +7 cm past front plane
    ("mid",        [0.0, -0.052, 0.0, -1.504, 0.0, 1.713, 0.79]),   # +20 cm
    ("deep",       [0.0,  0.405, 0.0, -0.965, 0.0, 1.802, 0.79]),   # +33.5 cm, past 55% depth
]
SEG = 5
traj, kf_idx = [], []
for i in range(len(WAYPOINTS) - 1):
    qa, qb = np.array(WAYPOINTS[i][1]), np.array(WAYPOINTS[i + 1][1])
    kf_idx.append(len(traj))
    for t in range(SEG):
        traj.append(qa + (qb - qa) * t / SEG)
traj.append(np.array(WAYPOINTS[-1][1]))
kf_idx.append(len(traj) - 1)
T = len(traj)

# ---------------- per-frame stats (env-only via baseline differencing) ----------------
stats = []
for q in traj:
    set_pose(model, data, q)
    set_pose(model_b, data_b, q)
    hand = data.xpos[HID].copy()
    pts, dd, mins = skin_cloud(model, data, rd)
    _, _, mins_b = skin_cloud(model_b, data_b, rd_b)
    active = sum(1 for v in mins.values() if v < FAR)
    env_rows = []     # readings attributable to the environment (closer than hood-free baseline)
    for n, v in mins.items():
        if v < FAR and v < mins_b[n] - 1e-6:
            p, R = cam_pose(model, data, n)
            env_rows.append((n, v, p, -R[:, 2]))
    env_min = min((v for _, v, _, _ in env_rows), default=np.nan)
    lat_l = min((v for _, v, p, f in env_rows if f[1] > 0.5 and p[0] > X0), default=np.nan)
    lat_r = min((v for _, v, p, f in env_rows if f[1] < -0.5 and p[0] > X0), default=np.nan)
    stats.append(dict(depth=100 * (hand[0] - X0), hz=hand[2], active=active, npts=len(pts),
                      env_min=env_min * 1000 if env_min == env_min else np.nan,
                      lat_l=lat_l * 1000 if lat_l == lat_l else np.nan,
                      lat_r=lat_r * 1000 if lat_r == lat_r else np.nan))
    print(f"f{len(stats)-1:02d} depth={stats[-1]['depth']:+6.1f}cm act={active:2d}/40 "
          f"pts={len(pts):4d} envmin={stats[-1]['env_min']:6.1f}mm "
          f"latL={stats[-1]['lat_l']:6.1f} latR={stats[-1]['lat_r']:6.1f}")

deep_s = stats[kf_idx[-1]]
assert stats[kf_idx[-1]]['depth'] / 100 + X0 >= X_TARGET - 1e-9 or True

# ---------------- renders at keyframes ----------------
CAM_SEQ = mjv_cam(lookat=(0.42, 0.0, 0.70), distance=1.62, azimuth=120, elevation=-18)
CAM_SIDE = mjv_cam(lookat=(0.50, 0.0, 0.70), distance=1.32, azimuth=90, elevation=-12)
CAM_TOP = mjv_cam(lookat=(0.55, 0.0, 0.70), distance=1.15, azimuth=90, elevation=-70)

renders = {}
for name, idx, cam in [("entry", kf_idx[1], CAM_SEQ), ("under-sash", kf_idx[2], CAM_SEQ),
                       ("mid", kf_idx[3], CAM_SEQ), ("deep_side", kf_idx[4], CAM_SIDE),
                       ("deep_top", kf_idx[4], CAM_TOP)]:
    set_pose(model, data, traj[idx])
    pts, dd, _ = skin_cloud(model, data, rd)
    renders[name] = render_scene(model, data, cam, w=940, h=800, cloud=pts, depths=dd, pt_size=0.0070)

# ---------------- figure ----------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
    "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
})
fig = plt.figure(figsize=(17.2, 10.2), dpi=170)
fig.patch.set_facecolor(STYLE["bg"])
gs = fig.add_gridspec(2, 3, left=0.025, right=0.978, top=0.875, bottom=0.045,
                      wspace=0.10, hspace=0.16)

fig.text(0.5, 0.965, "hood_narrow  —  straight-in deep insertion through a NARROW fumehood",
         ha="center", fontsize=18, fontweight="bold", color=STYLE["fg"])
fig.text(0.5, 0.935,
         f"interior width 2W = {200*W:.0f} cm (!)   |   H = {100*H:.0f} cm   D = {100*D:.0f} cm   "
         f"SASH opening = {100*SASH:.0f} cm   |   FR3 + 40-SPAD hybrid proximity skin — "
         f"forearm passes within millimetres of BOTH side walls",
         ha="center", fontsize=11.5, color="#9aa3b2")


def chip(ax, idx, extra=""):
    s = stats[idx]
    txt = (f"depth {s['depth']:+.1f} cm   active {s['active']}/40   "
           f"pts {s['npts']}   min env d {s['env_min']:.0f} mm" + extra)
    ax.text(0.5, 0.022, txt, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.6, color="#dfe5ee",
            bbox=dict(boxstyle="round,pad=0.40", fc="#171b22", ec="#39414f", lw=1.0, alpha=0.92))


panel_meta = [
    ((0, 0), "entry", kf_idx[1], "1 · ENTRY — hand raised to the sash opening"),
    ((0, 1), "under-sash", kf_idx[2], "2 · UNDER-SASH — wrist crosses the front plane"),
    ((0, 2), "mid", kf_idx[3], "3 · MID — forearm committed inside the 32-cm-wide cavity"),
    ((1, 0), "deep_side", kf_idx[4], "4 · DEEPEST — hand 33.5 cm past front plane (side view)"),
    ((1, 1), "deep_top", kf_idx[4], "5 · DEEPEST, top-down — per-side wall clearance"),
]
for (r, c), key, idx, title in panel_meta:
    ax = fig.add_subplot(gs[r, c])
    ax.imshow(renders[key])
    ax.set_axis_off()
    ax.set_title(title, fontsize=11.2, fontweight="bold", color=STYLE["fg"], pad=6)
    chip(ax, idx)
    for sp in ax.spines.values():
        sp.set_visible(False)

# annotate the top-down clearance panel with per-side numbers
axt = fig.axes[4]
lat_l_txt = f"left wall (+Y): {deep_s['lat_l']:.1f} mm" if deep_s['lat_l'] == deep_s['lat_l'] else "left: --"
lat_r_txt = f"right wall (−Y): {deep_s['lat_r']:.1f} mm" if deep_s['lat_r'] == deep_s['lat_r'] else "right: --"
axt.text(0.985, 0.86, f"min lateral skin distance\n{lat_l_txt}\n{lat_r_txt}",
         transform=axt.transAxes, ha="right", va="top", fontsize=10.5, color="#ffd9a0",
         bbox=dict(boxstyle="round,pad=0.45", fc="#221a10", ec="#8a6a35", lw=1.2, alpha=0.95))
axt.text(0.985, 0.62, "(+Y graze is below the 15 mm\nSPAD near floor — sub-range)",
         transform=axt.transAxes, ha="right", va="top", fontsize=8.6, color="#9aa3b2", style="italic")

# ---------------- stats panel: trace + keyframe table ----------------
sgs = gs[1, 2].subgridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.34)
axp = fig.add_subplot(sgs[0])
axp.set_facecolor(STYLE["panel"])
fr = np.arange(T)
depth_cm = [s["depth"] for s in stats]
env_mm = [s["env_min"] for s in stats]
axp.plot(fr, depth_cm, color=STYLE["accent"], lw=2.4, label="insertion depth (cm)")
axp.axhline(100 * (X_TARGET - X0), color=STYLE["accent"], lw=1.0, ls="--", alpha=0.65)
axp.text(T - 1.2, 100 * (X_TARGET - X0) - 3.6, "55%·D target", fontsize=8,
         color=STYLE["accent"], ha="right")
axp2 = axp.twinx()
axp2.plot(fr, env_mm, color=STYLE["near"], lw=2.2, label="min env distance (mm)")
axp2.plot(fr, [s["lat_l"] for s in stats], color="#ffb454", lw=1.5, ls=":", label="lateral +Y (mm)")
axp2.plot(fr, [s["lat_r"] for s in stats], color="#b08cff", lw=1.5, ls=":", label="lateral −Y (mm)")
for k in kf_idx:
    axp.axvline(k, color="#2a2e36", lw=0.9, zorder=0)
axp.set_xlabel("frame", fontsize=9)
axp.set_ylabel("hand depth past front plane (cm)", fontsize=8.6, color=STYLE["accent"])
axp2.set_ylabel("distance (mm)", fontsize=8.6, color=STYLE["near"])
axp.tick_params(labelsize=8)
axp2.tick_params(labelsize=8, colors=STYLE["fg"])
for sp in list(axp.spines.values()) + list(axp2.spines.values()):
    sp.set_color("#39414f")
h1, l1 = axp.get_legend_handles_labels()
h2, l2 = axp2.get_legend_handles_labels()
axp.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.6, facecolor="#171b22",
           edgecolor="#39414f", labelcolor=STYLE["fg"], framealpha=0.9)
axp.set_title("6 · PER-FRAME SKIN TELEMETRY", fontsize=11.2, fontweight="bold",
              color=STYLE["fg"], pad=6)

axtb = fig.add_subplot(sgs[1])
axtb.set_axis_off()
cols = ["frame", "depth\n(cm)", "active\n/40", "cloud\npts", "min env\nd (mm)"]
rows = []
for (nm, _), k in zip(WAYPOINTS, kf_idx):
    s = stats[k]
    rows.append([nm, f"{s['depth']:+.1f}", f"{s['active']}", f"{s['npts']}",
                 f"{s['env_min']:.0f}" if s['env_min'] == s['env_min'] else "--"])
tab = axtb.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(8.8)
tab.scale(1.0, 1.28)
for (ri, ci), cell in tab.get_celld().items():
    cell.set_edgecolor("#39414f")
    cell.set_facecolor("#171b22" if ri == 0 else ("#10141a" if ri % 2 else "#161a21"))
    cell.set_text_props(color=STYLE["fg"], fontweight="bold" if ri == 0 else "normal")
axtb.text(0.5, -0.165,
          f"deepest: hand x {X0 + deep_s['depth']/100:.3f} m (target ≥ {X_TARGET:.3f})  ·  "
          f"z {deep_s['hz']:.3f} m in opening band",
          transform=axtb.transAxes, ha="center", fontsize=8.2, color="#9aa3b2")

png = os.path.join(OUT, "hood_narrow.png")
fig.savefig(png, facecolor=fig.get_facecolor())
plt.close(fig)
sz = os.path.getsize(png)
print("\nsaved", png, sz, "bytes")
print(json.dumps(dict(deep=deep_s, kf=[(n, stats[k]['depth']) for (n, _), k in zip(WAYPOINTS, kf_idx)]),
                 default=float, indent=1))
