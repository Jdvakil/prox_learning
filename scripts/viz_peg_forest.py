"""env_peg_forest -- Whole-arm proximity sensing in a forest of vertical pegs.

A real Franka FR3 (40-SPAD hybrid proximity skin) snakes a custom reaching pose between
a maze of thin vertical rods. Every link that passes near a peg lights up its skin cloud,
demonstrating that the hybrid skin gives the policy whole-arm obstacle awareness -- not just
fingertip contact -- when the RGB cameras are blurred at training time.

Output: diagnostics_output/20260611_hybrid_overnight/env_peg_forest.png
"""
import sys, os
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (
    build, set_pose, skin_cloud, add_cylinder, nice_lights, render_scene, mjv_cam,
    FOVY, NEAR, FAR, STYLE,
)
import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from collections import defaultdict

OUTDIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight"
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "env_peg_forest.png")

# ---------------------------------------------------------------------------------------
# Scene definition: a custom snaking pose + a staggered forest of 12 vertical pegs.
# Peg xy were tuned (mj_geomDistance) so the arm threads BETWEEN them: every peg clears the
# arm surface (>=21 mm, no penetration) yet sits inside the 0.5 m skin range.
# ---------------------------------------------------------------------------------------
POSE = [0.45, -0.30, 0.30, -1.95, 0.15, 1.85, 0.6]
PEG_R, PEG_HALF, PEG_ZC = 0.018, 0.42, 0.55           # 36 mm dia, ~0.84 m tall rods
PEGS = [(0.30, 0.10), (0.38, -0.10), (0.435, 0.274), (0.45, 0.18),
        (0.55, 0.02), (0.357, 0.468), (0.42, 0.42), (0.15, -0.05),
        (0.50, 0.30),  (0.60, 0.20),  (0.33, 0.50),  (0.223, 0.435)]
PEG_RGBA = [(0.66, 0.70, 0.76, 1), (0.57, 0.62, 0.70, 1)]


def make(spec):
    nice_lights(spec)
    for i, (x, y) in enumerate(PEGS):
        add_cylinder(spec, f"peg_{i}", (x, y, PEG_ZC), PEG_R, PEG_HALF, PEG_RGBA[i % 2])


ARM_COLL = [f"fr3_link{k}_collision" for k in range(8)] + [
    "gripper/left_pad1", "gripper/left_pad2", "gripper/right_pad1", "gripper/right_pad2"]


def main():
    model = build(make=make)
    data = mujoco.MjData(model)
    set_pose(model, data, POSE)

    pts, depths, mins = skin_cloud(model, data)
    active = sum(1 for v in mins.values() if v < FAR)

    # Per-link activation + nearest return (the "whole-arm" story).
    by_link = defaultdict(lambda: [0, np.inf])
    for n, v in mins.items():
        if v < FAR:
            lk = n.rsplit("_sensor_", 1)[0]
            by_link[lk][0] += 1
            by_link[lk][1] = min(by_link[lk][1], v)

    # Clearance of each peg to the arm (proof the arm weaves, never collides).
    arm_gids = [model.geom(n).id for n in ARM_COLL]
    peg_clear = []
    for i in range(len(PEGS)):
        pg = model.geom(model.body(f"peg_{i}").geomadr[0]).id
        peg_clear.append(min(mujoco.mj_geomDistance(model, data, pg, ag, 2.0, np.zeros(6))
                             for ag in arm_gids))
    peg_clear = np.array(peg_clear)

    # ----- renders -----------------------------------------------------------------------
    iso = render_scene(model, data,
                       mjv_cam(lookat=(0.35, 0.18, 0.6), distance=1.62, azimuth=120, elevation=-18),
                       w=1000, h=860, cloud=pts, depths=depths, pt_size=0.0078)
    topdown = render_scene(model, data,
                           mjv_cam(lookat=(0.36, 0.18, 0.55), distance=1.45, azimuth=90, elevation=-84),
                           w=760, h=720, cloud=pts, depths=depths, pt_size=0.0072)
    # crop the dark margins so the maze fills the panel
    topdown = topdown[40:700, 90:700]

    # ----- figure ------------------------------------------------------------------------
    s = STYLE
    cmap = colormaps[s["cmap"]]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": s["fg"],
        "axes.edgecolor": s["grid"], "axes.labelcolor": s["fg"],
        "xtick.color": s["fg"], "ytick.color": s["fg"],
    })

    fig = plt.figure(figsize=(15.5, 9.6), dpi=170)
    fig.patch.set_facecolor(s["bg"])
    gs = fig.add_gridspec(2, 3, width_ratios=[1.85, 1.0, 1.0], height_ratios=[1.0, 0.62],
                          left=0.018, right=0.965, top=0.885, bottom=0.065,
                          wspace=0.235, hspace=0.22)

    # (A) hero iso view ------------------------------------------------------------------
    axA = fig.add_subplot(gs[:, 0]); axA.imshow(iso); axA.set_facecolor(s["bg"])
    axA.set_xticks([]); axA.set_yticks([])
    for sp in axA.spines.values():
        sp.set_edgecolor(s["accent"]); sp.set_linewidth(1.3)
    axA.set_title("Franka FR3 weaving through a peg forest  -  hybrid skin sees the whole arm",
                  color=s["fg"], fontsize=13.5, pad=9, loc="left", fontweight="bold")
    axA.text(0.012, 0.024,
             f"{active}/40 SPAD sensors returning   -   {len(pts)} back-projected skin points",
             transform=axA.transAxes, color=s["accent"], fontsize=10.5, fontweight="bold",
             va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.34", fc="#0d0f13", ec=s["accent"], alpha=0.82))
    # legend for cloud meaning
    leg = [Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(0.92),
                  markersize=8, label="near return (~red)"),
           Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(0.08),
                  markersize=8, label="far return (~blue)")]
    lg = axA.legend(handles=leg, loc="upper right", frameon=True, fontsize=9,
                    facecolor="#0d0f13", edgecolor=s["grid"], labelcolor=s["fg"])
    lg.get_frame().set_alpha(0.8)

    # (B) top-down maze ------------------------------------------------------------------
    axB = fig.add_subplot(gs[0, 1]); axB.imshow(topdown); axB.set_facecolor(s["bg"])
    axB.set_xticks([]); axB.set_yticks([])
    for sp in axB.spines.values():
        sp.set_edgecolor(s["grid"]); sp.set_linewidth(1.0)
    axB.set_title("top-down: the maze the arm threads", color=s["fg"], fontsize=11, pad=6, loc="left")

    # (C) top-down schematic of clearances ------------------------------------------------
    axC = fig.add_subplot(gs[0, 2], facecolor=s["panel"])
    # arm xy footprint (forearm/wrist geoms above the base)
    arm_xy = []
    for gi in range(model.ngeom):
        bn = model.body(model.geom_bodyid[gi]).name
        if (bn.startswith("fr3_link") or "hand" in bn or "finger" in bn):
            c = data.geom_xpos[gi]
            if 0.30 < c[2] < 0.95:
                arm_xy.append(c[:2])
    arm_xy = np.array(arm_xy)
    axC.plot(arm_xy[:, 0], arm_xy[:, 1], "-", color=s["accent"], lw=2.4, alpha=0.55,
             solid_capstyle="round", zorder=2)
    axC.scatter(arm_xy[:, 0], arm_xy[:, 1], s=14, color=s["accent"], alpha=0.9, zorder=3,
                label="arm links (xy)")
    # clearance uses its OWN green->orange scale (distinct from the distance colorbar):
    # tightest gaps glow orange, generous gaps stay green; every peg is > 0 (no contact).
    cl_cmap = colormaps["YlOrRd_r"]
    cl_norm = np.clip(peg_clear / 0.12, 0, 1)        # 0..120 mm band emphasises tight gaps
    pc_handles = []
    for (x, y), cl, cn in zip(PEGS, peg_clear, cl_norm):
        col = cl_cmap(cn)
        axC.add_patch(Circle((x, y), PEG_R, color=col, ec=s["fg"], lw=0.6, zorder=4))
        axC.text(x, y + 0.030, f"{cl*1000:.0f}", color=s["fg"], fontsize=7.0,
                 ha="center", va="bottom", zorder=5)
    axC.set_aspect("equal")
    axC.set_xlim(0.05, 0.70); axC.set_ylim(-0.20, 0.58)
    axC.set_xlabel("x (m)", fontsize=9); axC.set_ylabel("y (m)", fontsize=9)
    axC.set_title("top-down map: every peg clears the arm (mm)", color=s["fg"],
                  fontsize=10.2, pad=6, loc="left")
    axC.tick_params(labelsize=8)
    for sp in axC.spines.values():
        sp.set_edgecolor(s["grid"])
    axC.grid(True, color=s["grid"], lw=0.5, alpha=0.5)
    legC = [Line2D([0], [0], color=s["accent"], lw=2.4, marker="o", markersize=5,
                   label="arm links (xy path)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=cl_cmap(0.0),
                   markeredgecolor=s["fg"], markersize=9, label="peg (tight gap)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=cl_cmap(1.0),
                   markeredgecolor=s["fg"], markersize=9, label="peg (wide gap)")]
    axC.legend(handles=legC, loc="upper left", frameon=True, fontsize=7.2,
               facecolor="#0d0f13", edgecolor=s["grid"], labelcolor=s["fg"])

    # (D) per-link activation bar --------------------------------------------------------
    axD = fig.add_subplot(gs[1, 1:], facecolor=s["panel"])
    order = ["link1", "link2", "link3", "link4", "link5_front", "link5_back", "link6"]
    labels, counts, nears = [], [], []
    for lk in order:
        if lk in by_link:
            labels.append(lk.replace("_", "\n"))
            counts.append(by_link[lk][0])
            nears.append(by_link[lk][1])
    near_norm = np.clip((np.array(nears) - NEAR) / (FAR - NEAR), 0, 1)
    bar_cols = cmap(1.0 - near_norm)
    xpos = np.arange(len(labels))
    bars = axD.bar(xpos, counts, color=bar_cols, edgecolor=s["fg"], linewidth=0.6, width=0.66)
    for b, c, nv in zip(bars, counts, nears):
        axD.text(b.get_x() + b.get_width() / 2, c + 0.08, f"{c}\n{nv*1000:.0f} mm",
                 ha="center", va="bottom", color=s["fg"], fontsize=8.4)
    axD.set_xticks(xpos); axD.set_xticklabels(labels, fontsize=8.6)
    axD.set_ylabel("sensors\nreturning", fontsize=9.4)
    axD.set_ylim(0, max(counts) + 1.3)
    axD.set_title("whole-arm sensing: active SPAD count per link (label = nearest peg distance)",
                  color=s["fg"], fontsize=10.6, pad=6, loc="left")
    axD.tick_params(labelsize=8)
    for sp in axD.spines.values():
        sp.set_edgecolor(s["grid"])
    axD.grid(True, axis="y", color=s["grid"], lw=0.5, alpha=0.5)

    # shared distance colorbar (for the skin cloud + per-link bars) -----------------------
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=NEAR, vmax=FAR))
    cax = fig.add_axes([0.503, 0.10, 0.011, 0.235])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("skin return distance (m)", color=s["fg"], fontsize=9.2)
    cb.ax.tick_params(labelsize=8, color=s["fg"], labelcolor=s["fg"])
    cb.outline.set_edgecolor(s["grid"])
    cax.text(0.5, 1.05, "near", transform=cax.transAxes, ha="center", va="bottom",
             color=cmap(0.92), fontsize=8, fontweight="bold")
    cax.text(0.5, -0.06, "far", transform=cax.transAxes, ha="center", va="top",
             color=cmap(0.08), fontsize=8, fontweight="bold")

    # suptitle / caption -----------------------------------------------------------------
    fig.suptitle("env_peg_forest  -  hybrid proximity skin (40 SPAD depth cams, FOVY 45 deg, range 0.015-0.5 m)",
                 color=s["fg"], fontsize=15.5, fontweight="bold", x=0.018, ha="left", y=0.965)
    fig.text(0.018, 0.918,
             "RGB cameras are blurred at policy-training time -- this skin IS the robot's perception. "
             "The arm snakes a custom pose between 12 vertical rods; each link that passes near a rod "
             "lights up its proximity cloud (turbo_r: red=near, blue=far).",
             color="#b8bcc6", fontsize=10.2, ha="left", va="top")

    fig.savefig(OUT, facecolor=s["bg"], dpi=170)
    plt.close(fig)

    sz = os.path.getsize(OUT)
    print(f"saved {OUT}  ({sz/1024:.1f} KB)")
    print(f"active={active}/40  pts={len(pts)}  min_clear={peg_clear.min()*1000:.1f}mm  "
          f"min_depth={depths.min()*1000:.1f}mm")
    return OUT, sz, active, len(pts), peg_clear.min(), depths.min()


if __name__ == "__main__":
    main()
