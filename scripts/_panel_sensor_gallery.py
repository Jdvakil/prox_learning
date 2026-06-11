import sys, os, re
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, sensors, depth8, depth_renderer,
                            add_box, FOVY, NEAR, FAR, STYLE)
import mujoco, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import cm
from matplotlib.colors import Normalize

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_viz_suite"
os.makedirs(OUT, exist_ok=True)
KEY = "sensor_gallery_cavity"


def make(spec):
    add_box(spec, "bench", [0.52, 0, 0.175], [0.30, 0.34, 0.175], [0.62, 0.55, 0.45, 1])
    add_box(spec, "wl", [0.52, 0.30, 0.62], [0.30, 0.015, 0.28], [0.75, 0.72, 0.66, 0.30])
    add_box(spec, "wr", [0.52, -0.30, 0.62], [0.30, 0.015, 0.28], [0.75, 0.72, 0.66, 0.30])
    add_box(spec, "roof", [0.52, 0, 0.92], [0.30, 0.315, 0.015], [0.70, 0.68, 0.62, 0.30])
    add_box(spec, "back", [0.83, 0, 0.62], [0.015, 0.315, 0.28], [0.68, 0.66, 0.60, 1])
    add_box(spec, "pillar", [0.40, 0.13, 0.62], [0.025, 0.025, 0.27], [0.48, 0.34, 0.22, 1])


model = build(make=make)
data = mujoco.MjData(model)
set_pose(model, data, "reach")
names = sensors(model)
rd = depth_renderer(model)

# gather depth, min, active for each sensor
recs = []
for n in names:
    d8 = depth8(rd, data, n)
    valid_mask = (d8 >= NEAR) & (d8 <= FAR)
    valid = d8[valid_mask]
    active = valid.size > 0
    dmin_cm = float(valid.min() * 100.0) if active else None
    link = re.match(r"(link\d+)", n).group(1)
    # group key keeps the link5 front/back distinction; label sub keeps the index
    msub = re.match(r"link\d+(?:_(front|back))?_sensor_(\d+)", n)
    sub = msub.group(1)            # 'front'/'back' or None
    idx = msub.group(2)
    group = f"{link}_{sub}" if sub else link
    label = f"{group.replace('link', 'L')}.{idx}"
    recs.append(dict(name=n, link=link, group=group, label=label, idx=idx,
                     d8=d8, mask=valid_mask,
                     active=active, dmin_cm=dmin_cm))

n_active = sum(r["active"] for r in recs)

# normalization: distance in meters, near=red far=blue with turbo_r
cmap = plt.get_cmap(STYLE["cmap"]).copy()
cmap.set_bad("#0c0e12")          # no-return cells render near-black
norm = Normalize(vmin=NEAR, vmax=FAR)

# ---- figure / grid -------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": STYLE["fg"],
    "axes.edgecolor": STYLE["grid"],
})

NROW, NCOL = 5, 8
fig = plt.figure(figsize=(18.5, 13.4), dpi=170)
fig.patch.set_facecolor(STYLE["bg"])

# leave room: top banner + title, bottom colorbar, left link rail
gs = fig.add_gridspec(NROW, NCOL,
                      left=0.052, right=0.965, top=0.838, bottom=0.105,
                      hspace=0.60, wspace=0.28)

# stable per-group color accents for the rail / titles
link_palette = {
    "link1": "#4cc9f0", "link2": "#90e0a0", "link3": "#f9c74f",
    "link4": "#f3722c", "link5_back": "#c77dff", "link5_front": "#9d4edd",
    "link6": "#ff8fab",
}

axes = []
for r in range(NROW):
    for c in range(NCOL):
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

for k, ax in enumerate(axes):
    if k >= len(recs):
        ax.axis("off")
        ax.set_facecolor(STYLE["bg"])
        continue
    rec = recs[k]
    accent = link_palette[rec["group"]]
    d8 = rec["d8"].astype(float)
    disp = np.ma.array(d8, mask=~rec["mask"])

    ax.set_facecolor(STYLE["panel"])
    im = ax.imshow(disp, cmap=cmap, norm=norm, interpolation="nearest",
                   origin="upper", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(accent if rec["active"] else "#3a3f48")
        s.set_linewidth(1.6 if rec["active"] else 0.8)

    # title: "linkX.N  min=YY cm" (red if <8cm)
    label = rec["label"]
    if rec["active"]:
        mtxt = f"min={rec['dmin_cm']:.0f} cm"
        tcol = STYLE["near"] if rec["dmin_cm"] < 8.0 else STYLE["fg"]
    else:
        mtxt = "no return"
        tcol = "#6b7280"
    ax.set_title(f"{label}", color=accent, fontsize=11.5, fontweight="bold",
                 pad=14, loc="left")
    ax.text(0.0, 1.045, mtxt, transform=ax.transAxes, ha="left", va="bottom",
            color=tcol, fontsize=9.0,
            fontweight="bold" if (rec["active"] and rec["dmin_cm"] < 8.0) else "normal")

# ---- link legend strip (under subtitle) ---------------------------------
# One swatch per sensor group with its active count; the tile borders + titles
# share these colors so the eye maps each heatmap back to its link block.
group_order = []
for rec in recs:
    if rec["group"] not in group_order:
        group_order.append(rec["group"])

lx = 0.052
ly = 0.895
for g in group_order:
    accent = link_palette[g]
    cnt = sum(1 for rr in recs if rr["group"] == g)
    act = sum(1 for rr in recs if rr["group"] == g and rr["active"])
    disp_name = g.replace("link", "L").replace("_", " ").upper()
    sw = FancyBboxPatch((lx, ly), 0.013, 0.018, transform=fig.transFigure,
                        boxstyle="round,pad=0.001,rounding_size=0.004",
                        linewidth=0, facecolor=accent)
    fig.patches.append(sw)
    txt = f"{disp_name}  {act}/{cnt}"
    fig.text(lx + 0.018, ly + 0.009, txt, transform=fig.transFigure,
             ha="left", va="center", color=STYLE["fg"], fontsize=10.5,
             fontweight="bold")
    lx += 0.018 + 0.011 * (len(txt) + 1)

# ---- banner + titles -----------------------------------------------------
fig.text(0.052, 0.975, "FRANKA_SKIN  ·  HYBRID SPAD PROXIMITY GALLERY",
         color=STYLE["fg"], fontsize=23, fontweight="bold", ha="left", va="top")
fig.text(0.052, 0.940,
         "All 40 SPAD sensors  ·  8x8 depth cameras  ·  fovy=45  ·  range "
         "1.5-50 cm  ·  arm pose \"reach\" inside demo cavity",
         color="#aeb4bf", fontsize=12.5, ha="left", va="top")

# active-count banner pill (top-right)
bx, by, bw, bh = 0.748, 0.905, 0.217, 0.058
pill = FancyBboxPatch((bx, by), bw, bh, transform=fig.transFigure,
                      boxstyle="round,pad=0.004,rounding_size=0.012",
                      linewidth=2.0, edgecolor=STYLE["accent"],
                      facecolor="#10202a")
fig.patches.append(pill)
fig.text(bx + bw / 2, by + bh * 0.60, f"{n_active}/40 sensors active",
         color=STYLE["accent"], fontsize=18, fontweight="bold",
         ha="center", va="center")
fig.text(bx + bw / 2, by + bh * 0.20, "returns < 0.50 m",
         color="#8fb8c8", fontsize=9.5, ha="center", va="center")

# ---- shared colorbar -----------------------------------------------------
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cax = fig.add_axes([0.30, 0.045, 0.40, 0.022])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("distance (m)   —   red = near (1.5 cm)   ·   blue = far (50 cm)",
             color=STYLE["fg"], fontsize=12)
cb.ax.xaxis.set_tick_params(color=STYLE["fg"])
cb.outline.set_edgecolor(STYLE["grid"])
plt.setp(plt.getp(cb.ax, "xticklabels"), color=STYLE["fg"], fontsize=10)

# legend note bottom-left: red title meaning + no-return cell color
fig.text(0.052, 0.052,
         "red \"min\" label  →  closest return < 8 cm (imminent contact)",
         color=STYLE["near"], fontsize=10.5, ha="left", va="center", fontweight="bold")
fig.text(0.052, 0.028,
         "dark cells  →  no return (beyond 50 cm or below 1.5 cm)",
         color="#6b7280", fontsize=10.5, ha="left", va="center")

out_path = os.path.join(OUT, f"{KEY}.png")
fig.savefig(out_path, dpi=170, facecolor=fig.get_facecolor(),
            bbox_inches="tight", pad_inches=0.18)
plt.close(fig)

sz = os.path.getsize(out_path)
mins_all = [r["dmin_cm"] for r in recs if r["active"]]
print("PATH", out_path)
print("SIZE", sz)
print("ACTIVE", n_active)
print("MIN_CM", round(min(mins_all), 1), "MAX_CM", round(max(mins_all), 1))
print("UNDER8", sum(1 for m in mins_all if m < 8))
