"""hood_deep_tunnel — FUMEHOOD VARIATION figure for the FR3 + 40-SPAD hybrid proximity skin.

DEEP tunnel hood (interior 52 cm wide x 42 cm tall x 95 cm deep, 28 cm sash opening).
Max-insertion motion: stow -> under-sash entry -> mid -> deepest. The deepest pose is the
collision-valid kinematic maximum for this geometry: the elbow rests on the bench lip
(9 mm contact, like a person leaning into a hood) and the whole forearm + wrist + hand end
up inside the tunnel. All waypoints were solved against a collidable copy of the hood
(<=10 mm allowed grazing); every interpolated frame is verified.

Physics note (measured, not assumed): the variant's hand target x >= X0+0.60 = 0.95 m is
outside the FR3 reach envelope — the flange would need ~1.0 m from the shoulder vs ~0.86 m
max; even ignoring all collisions the flange ceiling inside the opening band is x = 0.802 m
(45.2 cm). The deepest physically-valid insertion achieved: hand 38.0 cm, fingertips/TCP
52.6 cm beyond the front plane.
"""
import sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, skin_cloud, render_scene, mjv_cam,
                            nice_lights, add_box, depth_renderer, FAR, NEAR, STYLE)
import mujoco, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colormaps
import os

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_fumehood_variations"
os.makedirs(OUT, exist_ok=True)
KEY = "hood_deep_tunnel"
TURBO_R = colormaps["turbo_r"]

# ------------------------------------------------------------------ hood (variant dims)
BZ, X0 = 0.585, 0.35
W, H, D, SASH = 0.26, 0.42, 0.95, 0.28
HOOD = ("bench", "bench_body", "wall_l", "wall_r", "back", "top", "sash")

def mk(s, collide=False):
    nice_lights(s)
    add_box(s, "bench", [X0+D/2, 0, BZ-0.015], [D/2+0.05, W+0.05, 0.015], [0.62, 0.55, 0.45, 1])
    add_box(s, "bench_body", [X0+D/2, 0, BZ/2-0.02], [D/2, W, BZ/2-0.02], [0.55, 0.5, 0.44, 1])
    add_box(s, "wall_l", [X0+D/2, W, BZ+H/2], [D/2, 0.012, H/2], [0.78, 0.8, 0.84, 0.30])
    add_box(s, "wall_r", [X0+D/2, -W, BZ+H/2], [D/2, 0.012, H/2], [0.78, 0.8, 0.84, 0.30])
    add_box(s, "back", [X0+D, 0, BZ+H/2], [0.012, W, H/2], [0.72, 0.7, 0.66, 1])
    add_box(s, "top", [X0+D/2, 0, BZ+H], [D/2, W, 0.012], [0.78, 0.8, 0.84, 0.30])
    add_box(s, "sash", [X0, 0, BZ+SASH+0.028], [0.012, W, 0.028], [0.62, 0.64, 0.66, 1])
    add_box(s, "target", [X0+0.7*D, 0.0, BZ+0.045], [0.04, 0.04, 0.045], [0.85, 0.5, 0.3, 1])
    if collide:
        for b in s.worldbody.bodies:
            if b.name in HOOD:
                for g in b.geoms:
                    g.contype = 1
                    g.conaffinity = 1

model = build(mk)
data = mujoco.MjData(model)
HAND, TCP = "gripper/base", "gripper/grasp_site"     # no fr3_hand body in this model;
hid, sid = model.body(HAND).id, model.site(TCP).id   # 2F-85 base sits at the FR3 flange

# collidable copy for trajectory validation
mC = build(lambda s: mk(s, collide=True))
dC = mujoco.MjData(mC)
hood_geoms = {i for i in range(mC.ngeom) if mC.body(mC.geom_bodyid[i]).name in HOOD}

def maxpen(q):
    set_pose(mC, dC, list(q))
    p = 0.0
    for ci in range(dC.ncon):
        c = dC.contact[ci]
        if (c.geom1 in hood_geoms) != (c.geom2 in hood_geoms):
            p = max(p, -c.dist)
    return p

# ------------------------------------------------------------- motion (solved offline)
Q_STOW  = np.array([0.0, -1.277, 0.0, -2.402, 0.0, 2.057, 0.79])
Q_ENTRY = np.array([0.0, -0.728, 0.0, -2.238, 0.0, 2.397, 0.79])
Q_MID   = np.array([0.0, -0.050, 0.0, -1.710, 0.0, 2.682, 0.79])
Q_DEEP  = np.array([0.0, 0.4724, 0.029, -1.0808, 0.2999, 2.7971, 0.79])

def smooth(a, b, n):
    return [a + (b - a) * (0.5 - 0.5 * np.cos(np.pi * t)) for t in np.linspace(0, 1, n)]

traj = []
for A, B, n in [(Q_STOW, Q_ENTRY, 9), (Q_ENTRY, Q_MID, 9), (Q_MID, Q_DEEP, 10)]:
    seg = smooth(A, B, n)
    if traj:
        seg = seg[1:]
    traj += seg
NF = len(traj)
F_ENTRY, F_MID, F_DEEP = 8, 16, NF - 1

# --------------------------------------------------------- per-frame skin + kinematics
rd = depth_renderer(model)
stats = []          # dicts per frame
clouds = {}         # key frames -> (pts, depths)
for i, q in enumerate(traj):
    set_pose(model, data, list(q))
    pts, dd, mins = skin_cloud(model, data, rd)
    active = sum(1 for v in mins.values() if v < FAR)
    mind = min(mins.values())
    h = data.xpos[hid].copy()
    g = data.site_xpos[sid].copy()
    pen = maxpen(q)
    stats.append(dict(hand=(h[0]-X0)*100, tcp=(g[0]-X0)*100, hz=h[2],
                      active=active, npts=len(pts), mind=mind*100, pen=pen*1000))
    if i in (F_ENTRY, F_MID, F_DEEP):
        clouds[i] = (pts, dd)

worst_pen = max(s["pen"] for s in stats)
sd = stats[F_DEEP]
print(f"frames={NF} worst_pen={worst_pen:.1f}mm")
print(f"DEEP: hand {sd['hand']:.1f}cm  TCP {sd['tcp']:.1f}cm  hand_z {sd['hz']:.3f} "
      f"(band {BZ+0.05:.3f}-{BZ+SASH-0.05:.3f})  active {sd['active']}/40  "
      f"pts {sd['npts']}  min {sd['mind']:.1f}cm")
assert worst_pen <= 10.0, "trajectory clips the hood"
assert BZ + 0.05 <= sd["hz"] <= BZ + SASH - 0.05, "hand outside opening band"

# ----------------------------------------------------------------------------- renders
def stage_render(i, cam, w=900, h=680):
    set_pose(model, data, list(traj[i]))
    pts, dd = clouds[i]
    return render_scene(model, data, cam, w=w, h=h, cloud=pts, depths=dd, pt_size=0.0070)

cam_main = mjv_cam(lookat=(0.52, 0.0, 0.62), distance=2.05, azimuth=112, elevation=-13)
cam_side = mjv_cam(lookat=(0.55, 0.0, 0.62), distance=1.95, azimuth=91, elevation=-8)
cam_top  = mjv_cam(lookat=(0.68, 0.0, 0.72), distance=1.55, azimuth=180, elevation=-75)

img_entry = stage_render(F_ENTRY, cam_main)
img_mid   = stage_render(F_MID, cam_main)
img_deep  = stage_render(F_DEEP, cam_main)
img_side  = stage_render(F_DEEP, cam_side)
img_top   = stage_render(F_DEEP, cam_top)

# ============================================================================== FIGURE
plt.rcParams.update({
    "figure.facecolor": STYLE["bg"], "savefig.facecolor": STYLE["bg"],
    "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
    "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
    "font.family": "DejaVu Sans", "font.size": 11,
})
fig = plt.figure(figsize=(19.4, 11.4), dpi=160)
gs = gridspec.GridSpec(2, 3, wspace=0.10, hspace=0.16,
                       left=0.025, right=0.978, top=0.875, bottom=0.045)

def img_panel(cell, img, title, sub=None):
    ax = fig.add_subplot(cell)
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(STYLE["grid"]); sp.set_linewidth(1.2)
    ax.set_title(title, color=STYLE["fg"], fontsize=12.5, fontweight="bold", pad=7)
    if sub:
        ax.text(0.012, 0.018, sub, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9.4, color="#dce6f2",
                bbox=dict(boxstyle="round,pad=0.4", fc="#10202f", ec=STYLE["grid"], alpha=0.88))
    return ax

se, sm_, sdp = stats[F_ENTRY], stats[F_MID], stats[F_DEEP]
axA = img_panel(gs[0, 0], img_entry,
                f"1 · ENTRY — hand at the front plane (frame {F_ENTRY}/{NF-1})",
                f"hand {se['hand']:+.1f} cm | fingertips {se['tcp']:+.1f} cm beyond plane\n"
                f"{se['active']}/40 sensors · {se['npts']:,} pts · min {se['mind']:.1f} cm")
axB = img_panel(gs[0, 1], img_mid,
                f"2 · MID-TUNNEL — wrist through the sash (frame {F_MID}/{NF-1})",
                f"hand {sm_['hand']:+.1f} cm | fingertips {sm_['tcp']:+.1f} cm beyond plane\n"
                f"{sm_['active']}/40 sensors · {sm_['npts']:,} pts · min {sm_['mind']:.1f} cm")
axC = img_panel(gs[0, 2], img_deep,
                f"3 · DEEPEST — forearm fully inside (frame {F_DEEP}/{NF-1})",
                f"hand {sdp['hand']:+.1f} cm | fingertips {sdp['tcp']:+.1f} cm beyond plane\n"
                f"{sdp['active']}/40 sensors · {sdp['npts']:,} pts · min {sdp['mind']:.1f} cm")
axD = img_panel(gs[1, 0], img_side,
                "side view at deepest — whole forearm inside the 95 cm tunnel",
                "elbow rests on the bench lip (9 mm contact)\n"
                "hand z = {:.2f} m, inside the 28 cm sash opening band".format(sdp["hz"]))
axE = img_panel(gs[1, 1], img_top,
                "overhead view at deepest — centered in the 52 cm-wide tunnel",
                "skin cloud paints both side walls, bench and sash edge")

# distance colorbar on the hero panel
cax = axC.inset_axes([0.66, 0.045, 0.30, 0.030])
smap = plt.cm.ScalarMappable(cmap=TURBO_R, norm=plt.Normalize(NEAR, FAR))
cb = fig.colorbar(smap, cax=cax, orientation="horizontal")
cb.set_label("skin distance (m)", color=STYLE["fg"], fontsize=9)
cb.ax.tick_params(labelsize=7.5, colors=STYLE["fg"])
cb.outline.set_edgecolor(STYLE["grid"])

# --------------------------------------------------------------------- stats panel (F)
gsS = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 2],
                                       height_ratios=[1.25, 1.0], hspace=0.42)
fr = np.arange(NF)
hand_d = np.array([s["hand"] for s in stats])
tcp_d = np.array([s["tcp"] for s in stats])
act = np.array([s["active"] for s in stats])
mind = np.array([s["mind"] for s in stats])

ax1 = fig.add_subplot(gsS[0])
ax1.set_facecolor(STYLE["panel"])
ax1.axhspan(0, D*100, color="#1d2a38", alpha=0.55, zorder=0)
ax1.axhline(0, color="#9aa4b2", lw=1.0, ls="--")
ax1.text(0.4, 1.8, "front plane (X0)", fontsize=8, color="#9aa4b2")
ax1.axhline(45.2, color="#ffd166", lw=1.0, ls=":")
ax1.text(0.4, 46.6, "FR3 flange kinematic ceiling in opening band (45.2 cm)",
         fontsize=7.8, color="#ffd166")
ax1.axhline(60, color="#ef476f", lw=1.0, ls=":")
ax1.text(0.4, 61.4, "variant hand target 60 cm — beyond the 855 mm reach envelope",
         fontsize=7.8, color="#ef476f")
ax1.plot(fr, tcp_d, color="#4cc9f0", lw=2.4, label="fingertips / TCP")
ax1.plot(fr, hand_d, color="#7fffd4", lw=2.4, label="hand (flange)")
for f, c in [(F_ENTRY, "#e8e8ea"), (F_MID, "#e8e8ea"), (F_DEEP, "#ef476f")]:
    ax1.axvline(f, color=c, lw=0.8, alpha=0.45, ls="--")
ax1.scatter([F_DEEP, F_DEEP], [hand_d[-1], tcp_d[-1]], s=42, zorder=5,
            color=["#7fffd4", "#4cc9f0"], edgecolor="white", lw=0.8)
ax1.annotate(f"{tcp_d[-1]:.1f} cm", (F_DEEP, tcp_d[-1]), xytext=(-58, 8),
             textcoords="offset points", fontsize=10.5, fontweight="bold", color="#4cc9f0")
ax1.annotate(f"{hand_d[-1]:.1f} cm", (F_DEEP, hand_d[-1]), xytext=(-58, -16),
             textcoords="offset points", fontsize=10.5, fontweight="bold", color="#7fffd4")
ax1.set_xlim(0, NF-1); ax1.set_ylim(-42, 100)
ax1.set_ylabel("insertion beyond front plane (cm)", fontsize=9.5)
ax1.tick_params(labelsize=8.5)
ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.0, labelcolor=STYLE["fg"])
for sp in ax1.spines.values():
    sp.set_edgecolor(STYLE["grid"])
ax1.set_title("insertion depth along the motion", fontsize=11.5, fontweight="bold", pad=5)

ax2 = fig.add_subplot(gsS[1])
ax2.set_facecolor(STYLE["panel"])
ax2.plot(fr, act, color="#4cc9f0", lw=2.2, drawstyle="steps-mid", label="active sensors /40")
ax2.set_ylim(0, 42)
ax2.set_xlim(0, NF-1)
ax2.set_xlabel("motion frame", fontsize=9.5)
ax2.set_ylabel("active sensors /40", color="#4cc9f0", fontsize=9.5)
ax2.tick_params(labelsize=8.5)
ax2b = ax2.twinx()
ax2b.plot(fr, mind, color="#ef476f", lw=2.0, label="min skin distance")
ax2b.set_ylabel("min skin distance (cm)", color="#ef476f", fontsize=9.5)
ax2b.tick_params(labelsize=8.5, colors=STYLE["fg"])
ax2b.set_ylim(0, max(12, mind.max()*1.15))
for f in (F_ENTRY, F_MID, F_DEEP):
    ax2.axvline(f, color="#e8e8ea", lw=0.8, alpha=0.35, ls="--")
for sp in list(ax2.spines.values()) + list(ax2b.spines.values()):
    sp.set_edgecolor(STYLE["grid"])
ax2.set_title("skin activity along the motion", fontsize=11.5, fontweight="bold", pad=5)
rows = [("entry", se), ("mid", sm_), ("deep", sdp)]
tbl = "        hand    tips   act    pts   min\n"
for nm, s in rows:
    tbl += f"{nm:>5}  {s['hand']:5.1f}  {s['tcp']:5.1f}  {s['active']:2d}/40  {s['npts']:5,}  {s['mind']:4.1f}cm\n"
ax2.text(0.985, 0.06, tbl.rstrip(), transform=ax2.transAxes, ha="right", va="bottom",
         fontsize=7.6, family="DejaVu Sans Mono", color="#dce6f2",
         bbox=dict(boxstyle="round,pad=0.45", fc="#10202f", ec=STYLE["grid"], alpha=0.92))

# ------------------------------------------------------------------------------ titles
fig.suptitle("FUMEHOOD VARIANT · hood_deep_tunnel — maximum insertion into a 95 cm deep tunnel hood",
             color=STYLE["fg"], fontsize=19, fontweight="bold", x=0.025, ha="left", y=0.972)
fig.text(0.025, 0.925,
         f"FR3 + 40-SPAD hybrid skin · hood interior W {2*W*100:.0f} cm (half-width {W*100:.0f}) × "
         f"H {H*100:.0f} cm × D {D*100:.0f} cm, sash opening {SASH*100:.0f} cm · "
         f"motion stow → under-sash → mid → deepest ({NF} frames, collision-checked, "
         f"worst grazing {worst_pen:.0f} mm)\n"
         f"deepest valid insertion: hand (flange) {sdp['hand']:.1f} cm, fingertips {sdp['tcp']:.1f} cm "
         f"beyond the front plane — whole forearm inside; the 60 cm hand target exceeds the FR3 "
         f"reach envelope (flange ceiling 45.2 cm even allowing clipping)",
         color="#a8b0bd", fontsize=11, ha="left", va="top", linespacing=1.5)

out = os.path.join(OUT, f"{KEY}.png")
fig.savefig(out, dpi=160)
plt.close(fig)
print(f"SAVED {out} ({os.path.getsize(out)/1024:.0f} KB)")
