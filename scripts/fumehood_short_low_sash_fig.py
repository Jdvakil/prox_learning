"""hood_short_low_sash: SHORT + LOW SASH fumehood variant for the FR3 + 40-SPAD hybrid skin.

W=0.34 H=0.34 D=0.50 SASH=0.17 -> the arm must flatten out and duck under a 17 cm opening.
Builds the hood, solves a 5-waypoint horizontal insertion (stow -> ready -> entry -> mid ->
deep), sweeps the trajectory, logs per-frame skin stats, and composes a 2x3 dark-theme figure.
"""
import sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, skin_cloud, render_scene, mjv_cam,
                            nice_lights, add_box, depth_renderer, FAR, NEAR, STYLE)
import mujoco
import numpy as np
import cv2
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_fumehood_variations"
os.makedirs(OUT, exist_ok=True)
KEY = "hood_short_low_sash"

# ---------------------------------------------------------------- hood (variant dims)
BZ, X0 = 0.585, 0.35
W, H, D, SASH = 0.34, 0.34, 0.50, 0.17

BOXES = {}  # name -> (center, half) for penetration checks


def _box(s, name, c, h, rgba):
    BOXES[name] = (np.array(c, float), np.array(h, float))
    add_box(s, name, c, h, rgba)


def mk(s):
    nice_lights(s)
    _box(s, "bench", [X0 + D / 2, 0, BZ - 0.015], [D / 2 + 0.05, W + 0.05, 0.015], [0.62, 0.55, 0.45, 1])
    _box(s, "bench_body", [X0 + D / 2, 0, BZ / 2 - 0.02], [D / 2, W, BZ / 2 - 0.02], [0.55, 0.5, 0.44, 1])
    _box(s, "wall_l", [X0 + D / 2, W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30])
    _box(s, "wall_r", [X0 + D / 2, -W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30])
    _box(s, "back", [X0 + D, 0, BZ + H / 2], [0.012, W, H / 2], [0.72, 0.7, 0.66, 1])
    _box(s, "top", [X0 + D / 2, 0, BZ + H], [D / 2, W, 0.012], [0.78, 0.8, 0.84, 0.30])
    _box(s, "sash", [X0, 0, BZ + SASH + 0.028], [0.012, W, 0.028], [0.62, 0.64, 0.66, 1])
    _box(s, "target", [X0 + 0.7 * D, 0.0, BZ + 0.045], [0.04, 0.04, 0.045], [0.85, 0.5, 0.3, 1])


model = build(mk)
data = mujoco.MjData(model)
HAND = "gripper/base"
HID = model.body(HAND).id
TCP = model.site("gripper/grasp_site").id
JLIM = [(-1.7837, 1.7837), (-3.0421, -0.1518), (0.5445, 4.5169)]  # j2 j4 j6
J7 = 0.0  # flat roll: finger pads at +/-y so the gripper is horizontal for the low slot


def fk(q):
    set_pose(model, data, q)
    hp = data.xpos[HID].copy()
    R = data.xmat[HID].reshape(3, 3).copy()
    return hp, R


def chain_pts():
    """sample points along elbow -> wrist -> flange -> hand -> tcp."""
    P = [data.xpos[model.body(n).id].copy()
         for n in ["fr3_link4", "fr3_link5", "fr3_link7", HAND]]
    P.append(data.site_xpos[TCP].copy())
    out = []
    for a, b in zip(P[:-1], P[1:]):
        for t in np.linspace(0, 1, 10):
            out.append(a * (1 - t) + b * t)
    return np.array(out)


def solve(tx, tz, pitch_deg, seeds, label=""):
    """[j2,j4,j6] (j1=j3=j5=0, j7=0.79): hand at (tx,0,tz), approach pitched down pitch_deg."""
    p = np.deg2rad(pitch_deg)
    tgt_app = np.array([np.cos(p), 0, -np.sin(p)])

    def cost(v):
        pen = sum(max(0, lo - jv) ** 2 * 100 + max(0, jv - hi) ** 2 * 100
                  for jv, (lo, hi) in zip(v, JLIM))
        q = [0, v[0], 0, v[1], 0, v[2], J7]
        hp, R = fk(q)
        c = 40 * (hp[0] - tx) ** 2 + 120 * (hp[2] - tz) ** 2 + 2.0 * np.sum((R[:, 2] - tgt_app) ** 2)
        # wrist skin hangs ~0.0365 m below link6 origin: keep it off the bench (top z = BZ)
        z6 = data.xpos[model.body("fr3_link6").id][2]
        x6 = data.xpos[model.body("fr3_link6").id][0]
        if x6 > 0.29:
            c += 400 * max(0, (BZ + 0.004) - (z6 - 0.0365)) ** 2
        cp = chain_pts()
        m = cp[:, 0] > X0 - 0.02
        if m.any():
            z = cp[m, 2]
            c += 60 * np.sum(np.maximum(0, z - (BZ + SASH - 0.048)) ** 2)
            c += 60 * np.sum(np.maximum(0, (BZ + 0.043) - z) ** 2)
        return c + pen

    best = None
    for s0 in seeds:
        r = minimize(cost, s0, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-6, fatol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    q = [0, best.x[0], 0, best.x[1], 0, best.x[2], J7]
    hp, R = fk(q)
    print(f"  {label:6s} cost={best.fun:.2e} hand=({hp[0]:.3f},{hp[2]:.3f}) app={np.round(R[:, 2], 2)}")
    return q


def interp(waypts, counts):
    qs = []
    for (a, b), n in zip(zip(waypts[:-1], waypts[1:]), counts):
        a, b = np.array(a), np.array(b)
        for t in np.linspace(0, 1, n, endpoint=False):
            tt = 3 * t ** 2 - 2 * t ** 3  # smoothstep
            qs.append(a * (1 - tt) + b * tt)
    qs.append(np.array(waypts[-1]))
    return qs


def penetration(margin=0.0):
    """max penetration depth (m) of chain/pad sample points into any scene box."""
    pts = list(chain_pts())
    for n in ["gripper/left_pad", "gripper/right_pad"]:
        pts.append(data.xpos[model.body(n).id].copy())
    worst = 0.0
    for p in pts:
        for name, (c, h) in BOXES.items():
            d = h - np.abs(p - c) + margin
            if (d > 0).all():
                worst = max(worst, float(d.min()))
    return worst


# ---------------------------------------------------------------- travel-height scan
SEEDS = [[0.3, -1.4, 1.6], [0.5, -1.0, 1.5], [0.1, -1.8, 1.9], [0.0, -2.4, 2.3]]
rd = depth_renderer(model)
print("scanning travel heights ...")
best = None
for zt in [0.700, 0.704, 0.708, 0.712]:
    entry = solve(0.38, zt - 0.004, 4, SEEDS, f"entry@{zt}")
    mid = solve(0.51, zt, 0, SEEDS, f"mid@{zt}")
    deep = solve(0.645, zt, 0, SEEDS, f"deep@{zt}")
    worst_min, worst_pen = 1e9, 0.0
    for q in interp([entry, mid, deep], [8, 9]):
        set_pose(model, data, q)
        _, _, mins = skin_cloud(model, data, rd)
        worst_min = min(worst_min, min(mins.values()))
        worst_pen = max(worst_pen, penetration())
    print(f"  z_travel={zt}: worst frame min-dist={worst_min*100:.2f} cm  max pen={worst_pen*1000:.1f} mm")
    score = worst_min - 10 * worst_pen
    if best is None or score > best[0]:
        best = (score, zt, entry, mid, deep, worst_min, worst_pen)

_, ZT, ENTRY, MID, DEEP, wmin, wpen = best
print(f"chosen z_travel={ZT} (worst min {wmin*100:.2f} cm, pen {wpen*1000:.1f} mm)")

STOW = solve(0.13, 0.70, 65, [[-0.8, -2.6, 1.9], [-1.1, -2.5, 2.3]], "stow")
READY = solve(0.22, 0.73, 30, [[-1.1, -2.5, 2.3]] + SEEDS, "ready")

# ---------------------------------------------------------------- trajectory + stats
TRAJ = interp([STOW, READY, ENTRY, MID, DEEP], [6, 7, 8, 9])
stats = []
for i, q in enumerate(TRAJ):
    set_pose(model, data, q)
    pts, dd, mins = skin_cloud(model, data, rd)
    hp = data.xpos[HID]
    tp = data.site_xpos[TCP]
    stats.append(dict(
        i=i, hand_x=hp[0], hand_z=hp[2], depth=(hp[0] - X0) * 100, tcp_depth=(tp[0] - X0) * 100,
        active=sum(1 for v in mins.values() if v < FAR * 0.999), npts=len(pts),
        mind=min(mins.values()) * 100, pen=penetration() * 1000))

IDX = {"ENTRY": 13, "MID": 21, "DEEPEST": len(TRAJ) - 1}
for k, i in IDX.items():
    s = stats[i]
    print(f"{k:8s} f{i:02d} depth={s['depth']:6.1f} cm  hand_z={s['hand_z']:.3f}  "
          f"active={s['active']}/40  pts={s['npts']}  min={s['mind']:.2f} cm  pen={s['pen']:.1f} mm")
worst_pen_traj = max(s["pen"] for s in stats)
print(f"trajectory max penetration: {worst_pen_traj:.1f} mm")

# ---------------------------------------------------------------- renders
CAM_SIDE = mjv_cam(lookat=(0.46, 0.0, 0.62), distance=1.42, azimuth=90, elevation=-9)
CAM_F34 = mjv_cam(lookat=(0.42, 0.0, 0.55), distance=1.9, azimuth=-40, elevation=-18)
CAM_TOP = mjv_cam(lookat=(0.55, 0.0, 0.64), distance=1.5, azimuth=-55, elevation=-48)
PW, PH = 820, 720


def shot(i, cam):
    set_pose(model, data, TRAJ[i])
    pts, dd, _ = skin_cloud(model, data, rd)
    return render_scene(model, data, cam, w=PW, h=PH, cloud=pts, depths=dd)


def label_panel(img, title, s, accent=(240, 201, 76)):
    img = img.copy()
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (PW, 54), (13, 15, 19), -1)
    cv2.rectangle(ov, (0, PH - 64), (PW, PH), (13, 15, 19), -1)
    img = cv2.addWeighted(ov, 0.78, img, 0.22, 0)
    cv2.putText(img, title, (18, 36), cv2.FONT_HERSHEY_DUPLEX, 0.95, accent[::-1], 1, cv2.LINE_AA)
    t1 = f"hand depth {s['depth']:+.1f} cm   TCP {s['tcp_depth']:+.1f} cm   hand z {s['hand_z']:.3f} m"
    t2 = f"active {s['active']}/40   cloud {s['npts']} pts   min skin dist {s['mind']:.2f} cm"
    cv2.putText(img, t1, (18, PH - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (232, 232, 234), 1, cv2.LINE_AA)
    cv2.putText(img, t2, (18, PH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (76, 201, 240)[::-1], 1, cv2.LINE_AA)
    return img


panels = [
    label_panel(shot(IDX["ENTRY"], CAM_SIDE), f"ENTRY  (frame {IDX['ENTRY']})  side view", stats[IDX["ENTRY"]]),
    label_panel(shot(IDX["MID"], CAM_SIDE), f"MID  (frame {IDX['MID']})  side view", stats[IDX["MID"]]),
    label_panel(shot(IDX["DEEPEST"], CAM_SIDE), f"DEEPEST  (frame {IDX['DEEPEST']})  side view", stats[IDX["DEEPEST"]]),
    label_panel(shot(IDX["DEEPEST"], CAM_F34), "DEEPEST  front 3/4  (ducked under 17 cm sash)", stats[IDX["DEEPEST"]]),
    label_panel(shot(IDX["DEEPEST"], CAM_TOP), "DEEPEST  overhead through hood ceiling", stats[IDX["DEEPEST"]]),
]

# ---------------------------------------------------------------- stats panel
plt.rcParams.update({
    "figure.facecolor": STYLE["bg"], "axes.facecolor": STYLE["panel"],
    "axes.edgecolor": STYLE["grid"], "axes.labelcolor": STYLE["fg"],
    "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"], "text.color": STYLE["fg"],
    "grid.color": STYLE["grid"], "font.size": 10.5})
fig, axes = plt.subplots(3, 1, figsize=(PW / 100, PH / 100), dpi=100, sharex=True,
                         gridspec_kw=dict(hspace=0.34, top=0.93, bottom=0.085, left=0.105, right=0.885))
fr = [s["i"] for s in stats]
dep = [s["depth"] for s in stats]
tdep = [s["tcp_depth"] for s in stats]
act = [s["active"] for s in stats]
npt = [s["npts"] for s in stats]
mind = [s["mind"] for s in stats]
tgt = 0.55 * D * 100

ax = axes[0]
ax.axhspan(0, D * 100, color="#223043", alpha=0.55, lw=0)
ax.plot(fr, dep, color=STYLE["accent"], lw=2.4, label="hand body")
ax.plot(fr, tdep, color="#9d8df1", lw=1.8, ls="--", label="TCP (fingertips)")
ax.axhline(tgt, color=STYLE["near"], lw=1.4, ls=":")
ax.text(0.4, tgt + 1.5, f"target  x >= X0+0.55D  ({tgt:.1f} cm)", color=STYLE["near"], fontsize=9)
ax.axhline(0, color="#888", lw=0.8)
ax.text(0.4, 1.5, "hood front plane (X0)", color="#aaa", fontsize=8.5)
ax.set_ylabel("insertion depth (cm)")
ax.legend(loc="upper left", framealpha=0.2, fontsize=9)
ax.set_title(f"insertion: final hand depth {dep[-1]:.1f} cm of D={D*100:.0f} cm  "
             f"(hand z={stats[-1]['hand_z']:.3f} m in 17 cm opening)", fontsize=11, pad=6)
ax.grid(alpha=0.4)

ax = axes[1]
ax.plot(fr, act, color="#f4a259", lw=2.4, label="active sensors")
ax.axhline(40, color="#666", lw=0.9, ls=":")
ax.set_ylabel("active sensors / 40")
ax.set_ylim(0, 42)
ax2 = ax.twinx()
ax2.plot(fr, npt, color="#80ed99", lw=1.7, ls="--", label="cloud points")
ax2.set_ylabel("cloud points", color="#80ed99")
ax2.tick_params(axis="y", colors="#80ed99")
ax.set_title(f"skin activity: {act[IDX['ENTRY']]} -> {act[IDX['MID']]} -> {act[-1]} sensors,  "
             f"{npt[-1]} pts at deepest", fontsize=11, pad=6)
ax.legend(loc="upper left", framealpha=0.2, fontsize=9)
ax.grid(alpha=0.4)

ax = axes[2]
ax.plot(fr, mind, color=STYLE["near"], lw=2.4)
ax.set_yscale("log")
ax.set_ylabel("min skin dist (cm)")
ax.set_xlabel("trajectory frame")
ax.set_title(f"clearance: min {min(mind):.2f} cm (wrist skin skims bench under 17 cm sash)",
             fontsize=11, pad=6)
ax.grid(alpha=0.4, which="both")
for k, i in IDX.items():
    for a in axes:
        a.axvline(i, color="#555", lw=0.9, ls="--")
    axes[0].text(i, axes[0].get_ylim()[1] * 0.97, k, rotation=90, va="top", ha="right",
                 fontsize=8, color="#bbb")
fig.canvas.draw()
sp = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
plt.close(fig)
sp = cv2.resize(sp, (PW, PH))
panels.append(sp)

# ---------------------------------------------------------------- compose 2x3
GAP, TBAR = 8, 96
CW, CH = 3 * PW + 4 * GAP, TBAR + 2 * PH + 3 * GAP
canvas = np.full((CH, CW, 3), (17, 19, 23), np.uint8)
title = ("FUMEHOOD VARIANT  hood_short_low_sash  |  W=34 (half-width) H=34 D=50 SASH=17 cm"
         "  |  FR3 + 40-SPAD hybrid skin")
sub = (f"low horizontal insertion under a 17 cm sash: hand depth {dep[-1]:.1f} cm "
       f"(target {tgt:.1f}),  {act[-1]}/40 sensors active,  min clearance {min(mind):.2f} cm")
cv2.putText(canvas, title, (GAP + 10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.05, (240, 201, 76)[::-1], 2, cv2.LINE_AA)
cv2.putText(canvas, sub, (GAP + 10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (232, 232, 234), 1, cv2.LINE_AA)
for k, p in enumerate(panels):
    r, c = divmod(k, 3)
    y = TBAR + GAP + r * (PH + GAP)
    x = GAP + c * (PW + GAP)
    canvas[y:y + PH, x:x + PW] = p

out_png = os.path.join(OUT, f"{KEY}.png")
cv2.imwrite(out_png, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
print("WROTE", out_png, os.path.getsize(out_png), "bytes")
print("FINAL:", {k: stats[i] for k, i in IDX.items()})
