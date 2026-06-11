"""hood_standard variant figure: FR3 + 40-SPAD skin inserting into a standard fume hood.

W=0.32 (interior half-width), H=0.46, D=0.55, SASH=0.30. Motion target: hand >= 55% of D.
"""
import sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, skin_cloud, render_scene, mjv_cam,
                            nice_lights, add_box, depth_renderer, depth8, backproject,
                            FAR, NEAR, STYLE)
import mujoco
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_fumehood_variations"
os.makedirs(OUT, exist_ok=True)
PNG = os.path.join(OUT, "hood_standard.png")

# ---------------------------------------------------------------- hood ------
W, H, D, SASH = 0.32, 0.46, 0.55, 0.30
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
HAND = "gripper/base"
hid = model.body(HAND).id
rd = depth_renderer(model)
SENSOR_NAMES = sorted(model.camera(i).name for i in range(model.ncam)
                      if "_sensor_" in model.camera(i).name)
NSENS = len(SENSOR_NAMES)

# ---------------------------------------------------------------- motion ----
WAYPOINTS = [
    np.array([0.0, -0.800, 0.0, -2.600, 0.0, 1.900, 0.79]),   # stow
    np.array([0.0, -0.598, 0.0, -2.070, 0.0, 2.168, 0.79]),   # under-sash  x=0.400
    np.array([0.0, -0.144, 0.0, -1.540, 0.0, 1.746, 0.79]),   # mid         x=0.520
    np.array([0.0,  0.340, 0.0, -0.980, 0.0, 1.880, 0.79]),   # deep        x=0.666 (57.5% D)
]
STEPS = 10
traj = []
for a, b in zip(WAYPOINTS[:-1], WAYPOINTS[1:]):
    for t in np.linspace(0, 1, STEPS, endpoint=False):
        traj.append((1 - t) * a + t * b)
traj.append(WAYPOINTS[-1])
traj = np.array(traj)
KEY = {"entry": STEPS, "mid": 2 * STEPS, "deepest": len(traj) - 1}


# static scene boxes (center, half) for env-point classification
ENV_BOXES = [
    ([X0 + D / 2, 0, BZ - 0.015], [D / 2 + 0.05, W + 0.05, 0.015]),
    ([X0 + D / 2, 0, BZ / 2 - 0.02], [D / 2, W, BZ / 2 - 0.02]),
    ([X0 + D / 2, W, BZ + H / 2], [D / 2, 0.012, H / 2]),
    ([X0 + D / 2, -W, BZ + H / 2], [D / 2, 0.012, H / 2]),
    ([X0 + D, 0, BZ + H / 2], [0.012, W, H / 2]),
    ([X0 + D / 2, 0, BZ + H], [D / 2, W, 0.012]),
    ([X0, 0, BZ + SASH + 0.028], [0.012, W, 0.028]),
    ([X0 + 0.7 * D, 0.0, BZ + 0.045], [0.04, 0.04, 0.045]),
]
_EC = np.array([b[0] for b in ENV_BOXES])
_EH = np.array([b[1] for b in ENV_BOXES])


def env_mask(pts, tol=0.02):
    """True for cloud points lying on a hood/bench surface or the floor."""
    if not len(pts):
        return np.zeros(0, bool)
    d = np.abs(pts[:, None, :] - _EC[None]) - _EH[None]          # N x B x 3
    dist = np.linalg.norm(np.maximum(d, 0), axis=2).min(axis=1)  # N
    return (dist < tol) | (pts[:, 2] < tol)                      # boxes or floor


def skin_frame(q):
    """Set pose; return cloud pts, depths, active count, min skin reading, min env clearance."""
    set_pose(model, data, q)
    P, Dd = [], []
    active = 0
    for n in SENSOR_NAMES:
        d8 = depth8(rd, data, n)
        cid = model.camera(n).id
        pts, dd = backproject(d8, data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3))
        if len(pts):
            active += 1
            P.append(pts)
            Dd.append(dd)
    pts = np.concatenate(P) if P else np.zeros((0, 3))
    dd = np.concatenate(Dd) if Dd else np.zeros((0,))
    mind = float(dd.min()) if len(dd) else float("nan")
    em = env_mask(pts)
    mind_env = float(dd[em].min()) if em.any() else float("nan")
    return pts, dd, active, mind, mind_env


# per-frame stats over the whole motion
stats = []
for q in traj:
    pts, dd, active, mind, mind_env = skin_frame(q)
    hp = data.xpos[hid].copy()
    stats.append(dict(depth=(hp[0] - X0) * 100.0, hz=hp[2], active=active,
                      npts=len(pts), mind=mind * 100.0, menv=mind_env * 100.0))
depths = np.array([s["depth"] for s in stats])
actives = np.array([s["active"] for s in stats])
npts = np.array([s["npts"] for s in stats])
minds = np.array([s["mind"] for s in stats])
menvs = np.array([s["menv"] for s in stats])
print("deepest frame: depth=%.1fcm (%.1f%% D)  hand z=%.3f  active=%d/%d  pts=%d  "
      "mind=%.1fcm  env_clear=%.1fcm"
      % (depths[-1], depths[-1] / (D * 100) * 100, stats[-1]["hz"], actives[-1], NSENS,
         npts[-1], minds[-1], menvs[-1]))

# ---------------------------------------------------------------- renders ---
CAM_MAIN = mjv_cam(lookat=(0.42, 0.0, 0.70), distance=2.10, azimuth=-38, elevation=-13)
CAM_SIDE = mjv_cam(lookat=(0.42, 0.0, 0.72), distance=1.70, azimuth=92, elevation=-8)
CAM_CLOSE = mjv_cam(lookat=(0.58, 0.02, 0.72), distance=1.00, azimuth=-27, elevation=-19)

frames = {}
for tag, idx in KEY.items():
    pts, dd, active, mind, mind_env = skin_frame(traj[idx])
    frames[tag] = dict(img=render_scene(model, data, CAM_MAIN, cloud=pts, depths=dd),
                       idx=idx)
# extra views of the deepest pose
pts, dd, active, mind, mind_env = skin_frame(traj[KEY["deepest"]])
img_side = render_scene(model, data, CAM_SIDE, cloud=pts, depths=dd)
img_close = render_scene(model, data, CAM_CLOSE, cloud=pts, depths=dd, pt_size=0.0085)

# ---------------------------------------------------------------- figure ----
BG, PANEL, FG, GRID, ACC = STYLE["bg"], STYLE["panel"], STYLE["fg"], STYLE["grid"], STYLE["accent"]
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": FG, "axes.edgecolor": GRID, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG, "font.family": "DejaVu Sans",
})

fig = plt.figure(figsize=(19.5, 11.6), dpi=160)
gs = GridSpec(2, 3, figure=fig, left=0.015, right=0.985, top=0.895, bottom=0.03,
              wspace=0.05, hspace=0.14)

fig.suptitle("FUMEHOOD VARIANT  ·  hood_standard   —   W=32 cm (half-width)  H=46 cm  "
             "D=55 cm  SASH=30 cm", fontsize=21, fontweight="bold", color=FG, y=0.975)
fig.text(0.5, 0.925, "FR3 + 40-SPAD hybrid proximity skin  ·  4-waypoint insertion: "
         "stow → under-sash → mid → deep  ·  target: hand ≥ 55% of D beyond hood face "
         "(≥ 30.3 cm)", ha="center", fontsize=12.5, color="#9aa3ad")

def img_panel(ax, img, title, sub):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title(title, fontsize=14, fontweight="bold", color=ACC, pad=6)
    ax.text(0.02, 0.035, sub, transform=ax.transAxes, fontsize=9.6, color=FG,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", fc="#1c2027", ec=GRID, alpha=0.92))

order = [("entry", "ENTRY  —  hand at sash plane"),
         ("mid", "MID  —  half-way into hood"),
         ("deepest", "DEEPEST  —  full insertion")]
for k, (tag, title) in enumerate(order):
    idx = frames[tag]["idx"]
    s = stats[idx]
    sub = (f"insertion {s['depth']:+.1f} cm ({s['depth']/(D*100)*100:.0f}% D)  ·  "
           f"{s['active']}/{NSENS} sens  ·  {s['npts']} pts  ·  "
           f"clear {s['menv']:.1f} cm")
    img_panel(fig.add_subplot(gs[0, k]), frames[tag]["img"], title, sub)

sdeep = stats[KEY["deepest"]]
sub_deep = (f"insertion {sdeep['depth']:+.1f} cm ({sdeep['depth']/(D*100)*100:.0f}% D)  ·  "
            f"{sdeep['active']}/{NSENS} sens  ·  clear {sdeep['menv']:.1f} cm")
img_panel(fig.add_subplot(gs[1, 0]), img_side,
          "DEEPEST  —  side view through glass", sub_deep)
img_panel(fig.add_subplot(gs[1, 1]), img_close,
          "DEEPEST  —  interior close-up, skin cloud", sub_deep)

# ------------------------------------------------------------- stats panel --
sub_gs = gs[1, 2].subgridspec(2, 1, height_ratios=[2.1, 1.0], hspace=0.32)
ax = fig.add_subplot(sub_gs[0])
ax.set_title("PER-FRAME SKIN TELEMETRY", fontsize=14, fontweight="bold", color=ACC, pad=6)
fr = np.arange(len(traj))

ax.plot(fr, depths, color=ACC, lw=2.6, label="insertion depth (cm)")
ax.axhline(0.55 * D * 100, color="#ef476f", lw=1.4, ls="--")
ax.text(0.4, 0.55 * D * 100 + 0.8, "55% D target (30.3 cm)", fontsize=9.5,
        color="#ef476f")
ax.axhline(0, color=GRID, lw=1)
ax.plot(fr, menvs, color="#ffd166", lw=2.2, label="min clearance to hood (cm)")
ax.fill_between(fr, menvs, color="#ffd166", alpha=0.10)
ax.plot(fr, minds, color="#8d99ae", lw=1.3, ls="-.",
        label="min skin reading incl. self (cm)")
ax.set_ylabel("cm", fontsize=11)
ax.set_xlabel("motion frame", fontsize=11)
ax.set_xlim(0, len(traj) - 1)
ax.grid(color=GRID, lw=0.6, alpha=0.7)

ax2 = ax.twinx()
ax2.plot(fr, actives, color="#06d6a0", lw=2.0, ls=":", label="active sensors /40")
ax2.set_ylabel("active sensors", fontsize=11, color="#06d6a0")
ax2.tick_params(axis="y", colors="#06d6a0")
ax2.set_ylim(0, NSENS + 2)
ax2.set_facecolor("none")

for tag in KEY:
    ax.axvline(KEY[tag], color="#5a6270", lw=1.0, ls=":")
    ax.text(KEY[tag], ax.get_ylim()[1] * 0.97, tag, rotation=90, fontsize=8.5,
            color="#9aa3ad", ha="right", va="top")

ln1, lb1 = ax.get_legend_handles_labels()
ln2, lb2 = ax2.get_legend_handles_labels()
ax.legend(ln1 + ln2, lb1 + lb2, loc="upper left", bbox_to_anchor=(0.015, 0.88),
          fontsize=9.2, facecolor="#1c2027", edgecolor=GRID, labelcolor=FG,
          framealpha=0.92)

# key-frame table in its own axes
axt = fig.add_subplot(sub_gs[1])
axt.set_axis_off()
axt.set_title("KEY-FRAME NUMBERS", fontsize=12, fontweight="bold", color=ACC, pad=2)
rows = []
for tag, idx in KEY.items():
    s = stats[idx]
    rows.append([tag, f"{s['depth']:.1f}", f"{s['depth']/(D*100)*100:.0f}%",
                 f"{s['active']}/{NSENS}", f"{s['npts']}", f"{s['menv']:.1f}"])
tbl = axt.table(cellText=rows,
                colLabels=["frame", "depth (cm)", "% of D", "sensors", "points",
                           "clearance (cm)"],
                loc="center", cellLoc="center", bbox=[0.0, 0.0, 1.0, 0.92])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#1c2027" if r else "#232833")
    cell.set_edgecolor(GRID)
    cell.set_height(0.25)
    cell.set_text_props(color=FG if r else ACC,
                        fontweight="normal" if r else "bold")

fig.savefig(PNG)
plt.close(fig)
sz = os.path.getsize(PNG)
print("saved", PNG, sz, "bytes")
print("RESULT", dict(deep_cm=round(depths[-1], 1), pctD=round(depths[-1] / (D * 100) * 100, 1),
                     active=int(actives[-1]), pts=int(npts[-1]), mind_cm=round(minds[-1], 1),
                     env_clear_cm=round(menvs[-1], 1), hand_z=round(stats[-1]["hz"], 3),
                     env_clear_min_over_motion=round(np.nanmin(menvs), 1),
                     key_env=[round(stats[i]["menv"], 1) for i in KEY.values()],
                     key_active=[int(stats[i]["active"]) for i in KEY.values()],
                     key_depth=[round(stats[i]["depth"], 1) for i in KEY.values()]))
