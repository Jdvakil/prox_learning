"""ENVIRONMENT PANEL: env_cluttered_shelf
Franka FR3 + 40-SPAD hybrid skin reaching INTO a cluttered shelf. The cloud overlay shows the
skin sensing clutter on multiple links. Exo RGB + skin cloud, depth tiles for the busiest
sensors, and a per-link active-sensor bar -- one stunning composited figure.
"""
import sys, os
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, sensors, depth8, backproject, cam_pose,
    add_box, add_cylinder, depth_renderer, nice_lights, skin_cloud, render_scene, mjv_cam,
    FOVY, NEAR, FAR, STYLE)
import mujoco, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

OUTDIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight"
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "env_cluttered_shelf.png")

# ----------------------------------------------------------------------------------------------
# SCENE: a real-looking shelf unit (floor + roof + back panel + two side walls) with assorted
# clutter scattered where the forearm/wrist reach. The arm reaches into the middle bay.
# Warm wood-ish shelf, varied clutter colors. contype/conaffinity=0 so nothing collides.
# ----------------------------------------------------------------------------------------------
SHELF_WOOD = [0.52, 0.40, 0.26, 1.0]      # warm wood
SHELF_BACK = [0.42, 0.32, 0.22, 1.0]

# shelf bay centered at x~0.50, spanning the arm's forward reach, opening toward -x (the robot)
SHELF = {
    "shelf_floor": ([0.55, 0.00, 0.40], [0.26, 0.40, 0.012], SHELF_WOOD),
    "shelf_roof":  ([0.55, 0.00, 0.78], [0.26, 0.40, 0.012], SHELF_WOOD),
    "shelf_back":  ([0.82, 0.00, 0.59], [0.012, 0.40, 0.20], SHELF_BACK),
    "shelf_wallL": ([0.55, 0.40, 0.59], [0.26, 0.012, 0.20], SHELF_WOOD),
    "shelf_wallR": ([0.55,-0.40, 0.59], [0.26, 0.012, 0.20], SHELF_WOOD),
}

# assorted clutter (5-7 items): boxes + cylinders of varied size/orientation, scattered in the
# bay around the arm path so multiple links sense them.
def _q(axis, deg):
    q = np.zeros(4); a = np.array(axis, float); a /= np.linalg.norm(a)
    mujoco.mju_axisAngle2Quat(q, a, np.deg2rad(deg)); return [float(v) for v in q]

CLUTTER = [
    # (kind, name, center, dims, rgba, quat)  dims: box->half3 ; cyl->(radius,halflen)
    ("box", "clutter_book",  [0.66, 0.205, 0.475], [0.035, 0.085, 0.062], [0.80, 0.28, 0.30, 1], _q([0,0,1], 18)),
    ("cyl", "clutter_can",   [0.60,-0.165, 0.485], [0.043, 0.072],        [0.30, 0.62, 0.78, 1], _q([1,0,0], 0)),
    ("box", "clutter_box2",  [0.70,-0.045, 0.47],  [0.05, 0.05, 0.05],    [0.86, 0.66, 0.24, 1], _q([0,0,1], -25)),
    ("cyl", "clutter_bottle",[0.685, 0.085, 0.515],[0.028, 0.10],         [0.40, 0.74, 0.42, 1], _q([1,0,0], 0)),
    ("cyl", "clutter_roll",  [0.55, 0.235, 0.50],  [0.034, 0.06],         [0.74, 0.74, 0.80, 1], _q([0,1,0], 90)),
    ("box", "clutter_tray",  [0.58,-0.27, 0.452],  [0.085, 0.05, 0.022],  [0.55, 0.45, 0.85, 1], _q([0,0,1], 8)),
    ("cyl", "clutter_mug",   [0.50, 0.12, 0.475],  [0.038, 0.05],         [0.90, 0.52, 0.30, 1], _q([1,0,0], 0)),
]


def make(spec):
    # warmer key light + softer fill for the "warm lighting" brief, plus a low fill from the
    # robot side so the open bay interior (where the arm + clutter are) is not in shadow.
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
    # exo camera for the hero render: 3/4 view from BEHIND/BESIDE the robot (-x, -y), looking in
    # the +x direction INTO the open shelf bay so we see the arm + clutter inside it.
    exo = spec.worldbody.add_camera(); exo.name = "exo_camera_1"
    exo.pos = [-0.85, -0.95, 1.00]
    target = np.array([0.50, 0.0, 0.52])              # into the bay
    vv = target - np.array(exo.pos); vv /= np.linalg.norm(vv)
    z = -vv; up = np.array([0, 0, 1.0])
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).reshape(9))
    exo.quat = [float(t) for t in q]; exo.fovy = 50; exo.resolution = [720, 900]


model = build(make=make, offw=1400, offh=1200)
data = mujoco.MjData(model)
set_pose(model, data, "reach")

names = sensors(model)
rd = depth_renderer(model)

# ---- per-sensor depth + cloud, track which link senses what ----
pts, depths, mins = skin_cloud(model, data, rd)
# a VALID return is an in-range measured surface: NEAR <= d < FAR.  (Readings below NEAR are the
# sensor grazing the arm's own structure / self-occlusion and produce no back-projected points.)
def valid(n):
    return NEAR <= mins[n] < FAR
active_names = [n for n in names if valid(n)]
active = len(active_names)
print(f"sensors {len(names)}  active {active}  cloud {len(pts)} pts")

# per-link active count
from collections import defaultdict, OrderedDict
LINK_ORDER = ["link1", "link2", "link3", "link4", "link5_front", "link5_back", "link6"]
link_active = OrderedDict((k, 0) for k in LINK_ORDER)
link_total = defaultdict(int)
link_min = {k: FAR for k in LINK_ORDER}
for n in names:
    lk = n.split("_sensor_")[0]
    link_total[lk] += 1
    if valid(n):
        link_active[lk] += 1
        link_min[lk] = min(link_min[lk], mins[n])

# the four busiest active sensors (smallest valid min distance) for the depth-tile strip
busiest = sorted(active_names, key=lambda n: mins[n])[:4]

# ---- HERO RENDER: exo RGB with the distance-colored skin cloud overlaid ----
hero_cam = "exo_camera_1"
r = mujoco.Renderer(model, 720, 900)
r.update_scene(data, hero_cam)
scn = r.scene
nrm = np.clip((depths - NEAR) / (FAR - NEAR), 0, 1)
turbo = matplotlib.colormaps["turbo"]
cols = turbo(1.0 - nrm)[:, :3]
for p, c in zip(pts, cols):
    if scn.ngeom >= scn.maxgeom:
        break
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.0080, 0, 0]),
                        np.asarray(p, np.float64), np.eye(3).ravel(),
                        np.array([c[0], c[1], c[2], 1.0], np.float32))
    scn.ngeom += 1
hero = r.render().copy()
hero = (np.clip((hero.astype(np.float32) / 255) ** 0.82 * 1.10, 0, 1) * 255).astype(np.uint8)

# ---- second view: "into the bay" mjv view from the robot side, peering over the front lip ----
side_cam = mjv_cam(lookat=(0.55, 0.0, 0.52), distance=1.0, azimuth=35, elevation=-22)
side = render_scene(model, data, side_cam, w=760, h=720, cloud=pts, depths=depths, pt_size=0.0070)

# ==============================================================================================
# FIGURE
# ==============================================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
    "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
    "axes.edgecolor": STYLE["grid"],
})
norm = mcolors.Normalize(vmin=NEAR, vmax=FAR)
cmap = matplotlib.colormaps[STYLE["cmap"]]

fig = plt.figure(figsize=(19.5, 11.0), dpi=170)
fig.patch.set_facecolor(STYLE["bg"])
gs = GridSpec(2, 3, figure=fig, width_ratios=[1.55, 1.0, 1.0], height_ratios=[1.0, 0.62],
              left=0.018, right=0.985, top=0.885, bottom=0.065, wspace=0.10, hspace=0.165)

# --- (A) HERO exo render -----------------------------------------------------------------------
axH = fig.add_subplot(gs[0, 0])
axH.imshow(hero); axH.set_xticks([]); axH.set_yticks([])
for s in axH.spines.values():
    s.set_edgecolor(STYLE["accent"]); s.set_linewidth(1.6)
axH.set_title("FR3 reaching into a cluttered shelf  ·  skin cloud overlaid (exo view)",
              color=STYLE["fg"], fontsize=13.5, fontweight="bold", pad=7)
axH.text(0.014, 0.026, f"{active}/40 SPAD sensors returning   ·   {len(pts)} skin points",
         transform=axH.transAxes, color="#0d0f12", fontsize=11.5, fontweight="bold",
         va="bottom", ha="left",
         bbox=dict(boxstyle="round,pad=0.35", fc=STYLE["accent"], ec="none", alpha=0.92))

# --- (B) robot's-eye side view ----------------------------------------------------------------
axS = fig.add_subplot(gs[0, 1])
axS.imshow(side); axS.set_xticks([]); axS.set_yticks([])
for s in axS.spines.values():
    s.set_edgecolor(STYLE["grid"]); s.set_linewidth(1.2)
axS.set_title("close-in view: clutter sensed across forearm + wrist",
              color=STYLE["fg"], fontsize=12, pad=7)

# --- (C) per-link active-sensor bar -----------------------------------------------------------
axB = fig.add_subplot(gs[0, 2])
axB.set_facecolor(STYLE["panel"])
disp = [k.replace("link", "L").replace("_", " ") for k in LINK_ORDER]
tot = [link_total[k] for k in LINK_ORDER]
act = [link_active[k] for k in LINK_ORDER]
ypos = np.arange(len(LINK_ORDER))[::-1]
axB.barh(ypos, tot, color="#262b34", edgecolor=STYLE["grid"], height=0.62, zorder=1)
bar_cols = [cmap(norm(np.clip(link_min[k], NEAR, FAR))) if act[i] else "#3a3f49"
            for i, k in enumerate(LINK_ORDER)]
axB.barh(ypos, act, color=bar_cols, edgecolor="#0d0f12", height=0.62, zorder=2)
for i, k in enumerate(LINK_ORDER):
    axB.text(tot[i] + 0.15, ypos[i], f"{act[i]}/{tot[i]}", va="center", ha="left",
             color=STYLE["fg"], fontsize=10, fontweight="bold")
    if act[i]:
        axB.text(0.12, ypos[i], f"{link_min[k]*100:.1f} cm", va="center", ha="left",
                 color="#0d0f12", fontsize=8.5, fontweight="bold")
axB.set_yticks(ypos); axB.set_yticklabels(disp, fontsize=10)
axB.set_xlim(0, max(tot) + 1.4)
axB.set_xlabel("SPAD sensors per link  (filled = returning a hit)", fontsize=10)
axB.set_title("where the skin feels the clutter", color=STYLE["fg"], fontsize=12, pad=7)
axB.tick_params(labelsize=9)
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)
axB.grid(axis="x", color=STYLE["grid"], lw=0.4, alpha=0.6, zorder=0)

# --- (D) busiest-sensor depth tiles (bottom-left, two columns) ---------------------------------
gsD = gs[1, 0].subgridspec(1, 4, wspace=0.18)
for i, n in enumerate(busiest):
    axd = fig.add_subplot(gsD[0, i])
    d8 = depth8(rd, data, n)
    masked = np.where(d8 <= FAR, d8, np.nan)
    im = axd.imshow(masked, cmap=cmap, vmin=NEAR, vmax=FAR, interpolation="nearest")
    axd.set_xticks([]); axd.set_yticks([])
    short = n.replace("_sensor", " s").replace("link", "L")
    axd.set_title(f"{short}\nmin {mins[n]*100:.1f} cm", color=STYLE["fg"], fontsize=9, pad=3)
    for s in axd.spines.values():
        s.set_edgecolor(STYLE["grid"]); s.set_linewidth(1.0)
fig.text(0.018 + 0.255, 0.345, "busiest sensors — raw 8×8 depth (turbo_r, red = near)",
         ha="center", color="#9aa3b2", fontsize=10.5, fontweight="bold")

# --- (E) clutter inventory + scene facts (bottom-mid/right span) -------------------------------
axT = fig.add_subplot(gs[1, 1:])
axT.set_facecolor(STYLE["panel"]); axT.set_xticks([]); axT.set_yticks([])
for s in axT.spines.values():
    s.set_edgecolor(STYLE["grid"]); s.set_linewidth(1.0)
# distance histogram of cloud points
axHi = axT.inset_axes([0.06, 0.20, 0.52, 0.66])
axHi.set_facecolor(STYLE["panel"])
nb = 26
counts, edges = np.histogram(depths, bins=nb, range=(NEAR, FAR))
centers = 0.5 * (edges[:-1] + edges[1:])
bc = cmap(norm(centers))
axHi.bar(centers, counts, width=(edges[1] - edges[0]) * 0.92, color=bc, edgecolor="none")
axHi.set_xlim(NEAR, FAR)
axHi.set_xlabel("skin-point distance (m)", fontsize=9.5)
axHi.set_ylabel("points", fontsize=9.5)
axHi.tick_params(labelsize=8)
for sp in ("top", "right"):
    axHi.spines[sp].set_visible(False)
axHi.set_title("range distribution of the 40-sensor cloud", color=STYLE["fg"], fontsize=10, pad=3)
med = float(np.median(depths)); near_frac = float((depths < 0.10).mean()) * 100
facts = (
    f"scene:  shelf bay (floor·roof·back·2 walls)  +  {len(CLUTTER)} clutter items\n"
    f"  boxes & cylinders, varied size / orientation\n\n"
    f"perception:  40 SPAD depth cams · 8×8 · fovy {FOVY:.0f}°\n"
    f"  range {NEAR*1000:.0f}–{FAR*100:.0f} mm · ~4 mm accurate\n\n"
    f"this frame (pose: reach):\n"
    f"  {active}/40 sensors returning · {len(pts)} skin points\n"
    f"  nearest hit {min(mins.values())*100:.1f} cm · median {med*100:.1f} cm\n"
    f"  {near_frac:.0f}% of points within 10 cm of a surface\n"
    f"  links engaged: {sum(1 for v in link_active.values() if v)}/{len(LINK_ORDER)}"
)
axT.text(0.62, 0.93, facts, transform=axT.transAxes, va="top", ha="left",
         color=STYLE["fg"], fontsize=10.6, linespacing=1.5, family="DejaVu Sans")

# ---- shared colorbar (top-right, clear of the title block) -----------------------------------
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cax = fig.add_axes([0.615, 0.952, 0.295, 0.017])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("distance (m)     near  →  far", color=STYLE["fg"], fontsize=11, labelpad=4)
cb.ax.xaxis.set_tick_params(color=STYLE["fg"], labelsize=8.5)
cb.ax.xaxis.set_label_position("top")
cb.outline.set_edgecolor(STYLE["grid"])
plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color=STYLE["fg"])

# ---- legend (under hero) ---------------------------------------------------------------------
proxies = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(0.18), markersize=9,
           label=f"SPAD skin return  ({active}/40 active)"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=SHELF_WOOD[:3], markersize=9,
           label="shelf structure"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=(0.86, 0.66, 0.24), markersize=9,
           label=f"clutter ({len(CLUTTER)} items)"),
]
leg = fig.legend(handles=proxies, loc="lower left", bbox_to_anchor=(0.022, 0.012),
                 frameon=True, fontsize=10, labelcolor=STYLE["fg"], ncol=3,
                 columnspacing=1.4, handletextpad=0.5)
leg.get_frame().set_facecolor(STYLE["panel"]); leg.get_frame().set_edgecolor(STYLE["grid"])

# ---- titles ----------------------------------------------------------------------------------
fig.suptitle("env_cluttered_shelf  —  franka_skin hybrid proximity perception in clutter",
             color=STYLE["fg"], fontsize=20, fontweight="bold", x=0.018, ha="left", y=0.972)
fig.text(0.018, 0.928,
         "Real Franka FR3 · 40-SPAD hybrid skin · RGB blurred at training time, so this skin IS "
         "the policy's perception in contact-rich reaching",
         ha="left", color="#9aa3b2", fontsize=12)

fig.savefig(OUT, dpi=170, facecolor=STYLE["bg"])
sz = os.path.getsize(OUT)
print("SAVED", OUT, "BYTES", sz, "KB", round(sz / 1024, 1))
print("ACTIVE", active, "PTS", len(pts), "LINKS_ENGAGED",
      sum(1 for v in link_active.values() if v))
print("BUSIEST", [(n, round(mins[n] * 100, 1)) for n in busiest])
