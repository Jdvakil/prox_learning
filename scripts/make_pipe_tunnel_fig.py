"""env_pipe_tunnel — signature 'ring' shot for the franka_skin HYBRID proximity skin.

A real Franka FR3 reaches deep inside a horizontal cylindrical pipe. Its 40 SPAD depth
sensors (8x8, fovy=45, range 0.015-0.5 m) range the pipe wall all the way around, producing
a complete RING of distance-colored cloud points tracing the cylinder interior. We then fit a
cylinder to that skin-only point cloud to show the wall is reconstructed to a few mm.
"""
import sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (
    build, set_pose, sensors, nice_lights, add_capsule, depth_renderer, depth8,
    backproject, cam_pose, skin_cloud, render_scene, mjv_cam, FOVY, NEAR, FAR, STYLE)
import mujoco, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy.optimize import least_squares
import os

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight"
os.makedirs(OUT, exist_ok=True)
TURBO_R = colormaps["turbo_r"]

# ---------------------------------------------------------------- pipe geometry
PIPE_R, PIPE_CX, PIPE_CZ, PIPE_HL = 0.16, 0.45, 0.62, 0.42
NSTAVE, STAVE_R = 40, 0.018
INNER = PIPE_R - STAVE_R                       # inner wall surface the sensors actually see
QPOSE = [0.0, 0.2, 0.0, -1.6, 0.0, 1.5, 0.79]  # arm reaches horizontally deep into +x

def make(spec):
    nice_lights(spec)
    # hollow tube = ring of capsule staves running along x; presents a real INNER surface
    # (a single translucent cylinder is back-face culled from the inside and reads no return)
    for k in range(NSTAVE):
        th = 2 * np.pi * k / NSTAVE
        cy = PIPE_R * np.cos(th)
        cz = PIPE_CZ + PIPE_R * np.sin(th)
        add_capsule(spec, f"stave{k}", [PIPE_CX, cy, cz], STAVE_R, PIPE_HL,
                    [0.50, 0.55, 0.68, 0.32], quat=[0.707, 0, 0.707, 0])

model = build(make=make)
data = mujoco.MjData(model)
set_pose(model, data, QPOSE)

# ---------------------------------------------------------------- sense the pipe
rd = depth_renderer(model)
snames = sensors(model)
P, D, mins = [], [], {}
for n in snames:
    d8 = depth8(rd, data, n)
    mins[n] = float(d8.min())
    cid = model.camera(n).id
    pts, dd = backproject(d8, data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3))
    if len(pts):
        P.append(pts)
        D.append(dd)
pts = np.concatenate(P)
depths = np.concatenate(D)
active = sum(1 for v in mins.values() if v < FAR)

# wall subset + cylinder fit (axis along x)
rad = np.sqrt(pts[:, 1] ** 2 + (pts[:, 2] - PIPE_CZ) ** 2)
wmask = (rad > INNER - 0.05) & (rad < PIPE_R + 0.05)
wp, wd = pts[wmask], depths[wmask]

def resid(p):
    cy, cz, R = p
    return np.sqrt((wp[:, 1] - cy) ** 2 + (wp[:, 2] - cz) ** 2) - R
fit = least_squares(resid, [0.0, PIPE_CZ, INNER])
fcy, fcz, fR = fit.x
rms = float(np.sqrt(np.mean(resid(fit.x) ** 2)))
cen_err = np.hypot(fcy - 0.0, fcz - PIPE_CZ) * 1000
rad_err = abs(fR - INNER) * 1000

# angular coverage of the ring
ang = np.degrees(np.arctan2(wp[:, 2] - PIPE_CZ, wp[:, 1]))
cov_bins = len(np.unique((ang // 30).astype(int)))

print(f"active sensors {active}/40 | cloud {len(pts)} pts | wall {len(wp)} pts | "
      f"ring coverage {cov_bins}/12 sectors")
print(f"cylinder fit: R={fR*1000:.1f} mm (true {INNER*1000:.1f}) | center err {cen_err:.1f} mm | "
      f"radius err {rad_err:.1f} mm | RMS {rms*1000:.1f} mm")

# ---------------------------------------------------------------- hero render
hero = render_scene(model, data,
                    mjv_cam(lookat=(0.50, 0.0, 0.61), distance=1.20, azimuth=175, elevation=-8),
                    w=980, h=900, cloud=pts, depths=depths, pt_size=0.0052)

# ============================================================= COMPOSE FIGURE
plt.rcParams.update({
    "figure.facecolor": STYLE["bg"], "savefig.facecolor": STYLE["bg"],
    "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
    "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
    "font.family": "DejaVu Sans", "font.size": 11,
})
fig = plt.figure(figsize=(16.5, 9.6), dpi=170)
gs = gridspec.GridSpec(2, 3, width_ratios=[1.62, 1.0, 1.0], height_ratios=[1.0, 1.0],
                       wspace=0.20, hspace=0.26,
                       left=0.035, right=0.975, top=0.885, bottom=0.085)

# ---- (A) hero ring shot ----------------------------------------------------
axH = fig.add_subplot(gs[:, 0])
axH.imshow(hero)
axH.set_xticks([]); axH.set_yticks([])
for sp in axH.spines.values():
    sp.set_edgecolor(STYLE["grid"]); sp.set_linewidth(1.2)
axH.set_title("Franka FR3 + 40-SPAD skin, reaching deep inside a 0.32 m pipe",
              color=STYLE["fg"], fontsize=14, fontweight="bold", pad=10)
axH.text(0.5, -0.052,
         "view looks down the pipe axis  ·  skin returns trace the full interior wall\n"
         "wrist RGB cams are blurred at policy time — this cloud IS the robot's perception",
         transform=axH.transAxes, ha="center", va="top",
         color="#a8b0bd", fontsize=9.8, linespacing=1.4)
# inline distance colorbar on the hero
cax = axH.inset_axes([0.045, 0.045, 0.30, 0.028])
sm = plt.cm.ScalarMappable(cmap=TURBO_R, norm=plt.Normalize(NEAR, FAR))
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("skin distance (m)", color=STYLE["fg"], fontsize=9.5)
cb.ax.tick_params(labelsize=8, colors=STYLE["fg"])
cb.outline.set_edgecolor(STYLE["grid"])

# ---- (B) reconstructed cross-section (y-z) ---------------------------------
axC = fig.add_subplot(gs[0, 1])
axC.set_facecolor(STYLE["panel"])
sc = axC.scatter(wp[:, 1], wp[:, 2], c=wd, cmap=TURBO_R, vmin=NEAR, vmax=FAR,
                 s=11, alpha=0.9, edgecolors="none")
# true inner wall + fitted circle
true_c = Circle((0.0, PIPE_CZ), INNER, fill=False, ec="#7fffd4", lw=1.6, ls="--",
                label="true inner wall")
fit_c = Circle((fcy, fcz), fR, fill=False, ec=STYLE["fg"], lw=1.1, ls=":",
               label="cylinder fit")
axC.add_patch(true_c); axC.add_patch(fit_c)
# arm cross-section markers (sensor ring centroid)
axC.set_aspect("equal")
axC.set_xlim(-0.21, 0.21); axC.set_ylim(PIPE_CZ - 0.21, PIPE_CZ + 0.21)
axC.set_xlabel("y (m)", fontsize=10); axC.set_ylabel("z (m)", fontsize=10)
axC.tick_params(labelsize=8.5)
for sp in axC.spines.values():
    sp.set_edgecolor(STYLE["grid"])
axC.set_title("skin cloud projected to pipe cross-section",
              fontsize=11.5, fontweight="bold", pad=6)
leg = axC.legend(loc="upper right", fontsize=8.0, framealpha=0.0, labelcolor=STYLE["fg"])
axC.text(0.035, 0.04,
         f"cylinder-fit RMS  {rms*1000:.1f} mm\n"
         f"radius err  {rad_err:.1f} mm   center err  {cen_err:.1f} mm",
         transform=axC.transAxes, va="bottom", ha="left", fontsize=8.6,
         color="#cfe8ff",
         bbox=dict(boxstyle="round,pad=0.35", fc="#0c1c2c", ec=STYLE["grid"], alpha=0.85))

# ---- (C) radial residual vs azimuth (ring uniformity) ----------------------
axR = fig.add_subplot(gs[0, 2])
axR.set_facecolor(STYLE["panel"])
r_meas = np.sqrt((wp[:, 1] - fcy) ** 2 + (wp[:, 2] - fcz) ** 2)
resid_mm = (r_meas - fR) * 1000.0
order = np.argsort(ang)
axR.scatter(ang, resid_mm, c=wd, cmap=TURBO_R, vmin=NEAR, vmax=FAR, s=9,
            alpha=0.85, edgecolors="none")
axR.axhline(0, color=STYLE["fg"], lw=0.9, ls="--", alpha=0.7)
axR.axhline(rms * 1000, color="#7fffd4", lw=0.8, ls=":", alpha=0.8)
axR.axhline(-rms * 1000, color="#7fffd4", lw=0.8, ls=":", alpha=0.8)
axR.set_xlim(-180, 180)
axR.set_xticks([-180, -90, 0, 90, 180])
axR.set_xlabel("azimuth around pipe (deg)", fontsize=10)
axR.set_ylabel("radial residual (mm)", fontsize=10)
axR.tick_params(labelsize=8.5)
for sp in axR.spines.values():
    sp.set_edgecolor(STYLE["grid"])
axR.set_title("wall reconstruction error, all around the ring",
              fontsize=11.5, fontweight="bold", pad=6)
axR.text(0.97, 0.04, f"±{rms*1000:.1f} mm RMS\n{cov_bins}/12 sectors covered",
         transform=axR.transAxes, ha="right", va="bottom", fontsize=8.6,
         color="#cfe8ff",
         bbox=dict(boxstyle="round,pad=0.35", fc="#0c1c2c", ec=STYLE["grid"], alpha=0.85))

# ---- (D) per-link min distance (which links are in contact range) ----------
axL = fig.add_subplot(gs[1, 1])
axL.set_facecolor(STYLE["panel"])
links = ["link1", "link2", "link3", "link4", "link5_back", "link5_front", "link6"]
lmins = {L: [] for L in links}
for n, v in mins.items():
    L = n.split("_sensor")[0]
    if L in lmins and v < FAR:
        lmins[L].append(v)
labels, vals, cols = [], [], []
for L in links:
    vs = lmins[L]
    if vs:
        mv = min(vs)
        labels.append(L.replace("_", " "))
        vals.append(mv)
        cols.append(TURBO_R((np.clip((mv - NEAR) / (FAR - NEAR), 0, 1))))
ypos = np.arange(len(labels))
axL.barh(ypos, vals, color=cols, edgecolor=STYLE["grid"], height=0.66)
for y, v in zip(ypos, vals):
    axL.text(v + 0.004, y, f"{v*1000:.0f} mm", va="center", fontsize=8.4, color=STYLE["fg"])
axL.set_yticks(ypos); axL.set_yticklabels(labels, fontsize=9)
axL.invert_yaxis()
axL.set_xlim(0, FAR)
axL.axvline(NEAR, color="#ef476f", lw=1.0, ls="--", alpha=0.8)
axL.set_xlabel("closest wall distance per link (m)", fontsize=10)
axL.tick_params(labelsize=8.5)
for sp in axL.spines.values():
    sp.set_edgecolor(STYLE["grid"])
axL.set_title("every wrist/forearm link feels the wall",
              fontsize=11.5, fontweight="bold", pad=6)

# ---- (E) stats / caption card ----------------------------------------------
axS = fig.add_subplot(gs[1, 2])
axS.axis("off")
axS.set_facecolor(STYLE["bg"])
lines = [
    ("HYBRID PROXIMITY SKIN", "#4cc9f0", 13, "bold"),
    ("40 SPAD sensors · 8×8 depth · fovy 45°", STYLE["fg"], 10.5, "normal"),
    (f"range {NEAR*1000:.0f}–{FAR*1000:.0f} mm · ~4 mm accuracy", STYLE["fg"], 10.5, "normal"),
    ("", STYLE["fg"], 6, "normal"),
    (f"{active}/40 sensors active in the pipe", "#7fffd4", 11.5, "bold"),
    (f"{len(pts):,} cloud points · {len(wp):,} on the wall", STYLE["fg"], 10.5, "normal"),
    (f"full ring: {cov_bins}/12 azimuth sectors", STYLE["fg"], 10.5, "normal"),
    ("", STYLE["fg"], 6, "normal"),
    ("PIPE RECONSTRUCTED FROM SKIN ALONE", "#4cc9f0", 11, "bold"),
    (f"fit radius   {fR*1000:.1f} mm  (true {INNER*1000:.1f})", STYLE["fg"], 10.5, "normal"),
    (f"radius error {rad_err:.1f} mm", STYLE["fg"], 10.5, "normal"),
    (f"center error {cen_err:.1f} mm", STYLE["fg"], 10.5, "normal"),
    (f"wall RMS     {rms*1000:.1f} mm", "#7fffd4", 11.0, "bold"),
]
y = 0.97
for txt, col, sz, w in lines:
    axS.text(0.04, y, txt, transform=axS.transAxes, color=col, fontsize=sz,
             fontweight=w, va="top", ha="left", family="DejaVu Sans")
    y -= 0.052 if txt else 0.030
axS.text(0.04, 0.10,
         "Contact-rich reaching inside confined geometry:\nthe skin senses the wall in every "
         "direction at once,\nthe sense modality the policy actually trains on.",
         transform=axS.transAxes, color="#a8b0bd", fontsize=9.2, va="bottom", ha="left")
for sp in []:
    pass
axS_rect = plt.Rectangle((0.015, 0.02), 0.97, 0.97, transform=axS.transAxes, fill=False,
                         ec=STYLE["grid"], lw=1.0)
axS.add_patch(axS_rect)

# ---- super title -----------------------------------------------------------
fig.suptitle("PROXIMITY SKIN AS PERCEPTION  ·  the signature pipe-tunnel ring",
             color=STYLE["fg"], fontsize=19, fontweight="bold", x=0.035, ha="left", y=0.965)
fig.text(0.035, 0.918,
         "franka_skin HYBRID  ·  the arm is inside a horizontal cylindrical pipe; its 40 depth "
         "sensors range the wall all around → a complete ring of distance-colored returns",
         color="#a8b0bd", fontsize=11.5, ha="left")

out = os.path.join(OUT, "env_pipe_tunnel.png")
fig.savefig(out, dpi=170)
plt.close(fig)
sz = os.path.getsize(out)
print(f"SAVED {out}  ({sz/1024:.0f} KB)")
print(f"RMS_MM={rms*1000:.2f} ACTIVE={active} WALLPTS={len(wp)} COV={cov_bins}")
