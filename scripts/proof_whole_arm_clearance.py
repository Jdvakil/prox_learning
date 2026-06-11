"""PROOF PANEL: use_whole_arm_clearance.

In a cluttered scene the hybrid SPAD skin reports a minimum clearance for EVERY link
(elbow, forearm, wrist) -- not just the gripper a single wrist camera can see. We render
the real Franka FR3 with each kinematic link COLORED by its min clearance (red=close,
green=clear), overlay the back-projected 40-sensor point cloud, and quantify how many links
sit within 8 cm of something.
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import mujoco

from hybrid_viz_lib import (
    build, set_pose, skin_cloud, nice_lights, add_box, add_cylinder,
    depth_renderer, depth8, cam_pose, render_scene, mjv_cam, sensors,
    FAR, NEAR, STYLE,
)

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight"
os.makedirs(OUT, exist_ok=True)
KEY = "use_whole_arm_clearance"

# clearance color scale (meters). Anything >= CLEAR_M renders fully "clear" green.
CLOSE_M, CLEAR_M = 0.02, 0.30
THRESH_M = 0.08  # "within 8 cm of something" call-out
CLEAR_CMAP = colormaps["RdYlGn"]  # red(close) -> green(clear)


def make(spec):
    """Realistic cluttered reach-into-a-shelf scene; graded clearance across the arm."""
    nice_lights(spec)
    add_box(spec, "wall_back",   [0.80, 0.00, 0.50], [0.03, 0.45, 0.50], [0.50, 0.43, 0.39, 1])
    add_box(spec, "shelf_top",   [0.62, 0.00, 0.92], [0.18, 0.40, 0.02], [0.58, 0.49, 0.41, 1])
    add_box(spec, "target_box",  [0.57, 0.16, 0.60], [0.06, 0.06, 0.10], [0.86, 0.50, 0.30, 1])
    add_cylinder(spec, "can",    [0.41, -0.20, 0.55], 0.045, 0.12,        [0.45, 0.60, 0.75, 1])
    add_box(spec, "pillar",      [0.04, 0.31, 0.45], [0.06, 0.06, 0.45], [0.50, 0.70, 0.50, 1])
    add_box(spec, "bin",         [0.22, -0.35, 0.16], [0.12, 0.10, 0.16], [0.40, 0.55, 0.70, 1])


def link_of(sensor_name):
    return sensor_name.split("_sensor_")[0].replace("_back", "").replace("_front", "")


def find_self_occluded(model, data, rd):
    """A sensor whose ENTIRE 8x8 map reads < NEAR+3mm is mounted flush against the robot's
    own body (degenerate mount) -- it never measures a real external clearance. Exclude it."""
    bad = set()
    for n in sensors(model):
        d8 = depth8(rd, data, n)
        if float(d8.max()) < NEAR + 0.003:
            bad.add(n)
    return bad


def clearance_color(d):
    if d is None or d >= FAR:
        return None
    t = np.clip((d - CLOSE_M) / (CLEAR_M - CLOSE_M), 0, 1)
    return CLEAR_CMAP(t)


# kinematic link <-> visual geom group(2)/colorable; cap clearance at link6 (wrist) -> gripper too
LINK_BODIES = {f"link{i}": f"fr3_link{i}" for i in range(1, 7)}
# extra robot bodies to tint with the nearest measured link clearance (wrist chain / gripper)
WRIST_EXTRA = ["fr3_link7", "wrist_cam_body", "gripper/base",
               "gripper/left_driver", "gripper/left_coupler", "gripper/left_spring_link",
               "gripper/left_follower", "gripper/left_pad", "gripper/right_driver",
               "gripper/right_coupler", "gripper/right_spring_link",
               "gripper/right_follower", "gripper/right_pad"]


def main():
    model = build(make=make)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")
    rd = depth_renderer(model)

    self_bad = find_self_occluded(model, data, rd)

    # ---- per-link min clearance + active-sensor counts (external reads only) ----
    pts, depths, mins = skin_cloud(model, data)
    link_min = defaultdict(lambda: FAR)
    link_active = defaultdict(int)
    link_total = defaultdict(int)
    for n in sensors(model):
        L = link_of(n)
        if n in self_bad:
            continue
        link_total[L] += 1
        v = mins[n]
        if v < FAR:
            link_active[L] += 1
            link_min[L] = min(link_min[L], v)

    links = sorted(LINK_BODIES.keys())
    total_active = sum(link_active.values())
    n_within_thresh = sum(1 for L in links if link_min[L] < THRESH_M)

    print("self-occluded (excluded):", sorted(self_bad))
    for L in links:
        d = link_min[L]
        print(f"{L}: min={'%.1f cm' % (d*100) if d<FAR else 'no return':>10s}  "
              f"active={link_active[L]}/{link_total[L]}")
    print(f"links within {THRESH_M*100:.0f} cm: {n_within_thresh}/6   active sensors: {total_active}")

    # ---- recolor the arm: each link's visual mesh tinted by its clearance ----
    rgba0 = model.geom_rgba.copy()
    # wrist chain inherits link6 clearance (the wrist camera's own neighbourhood)
    wrist_d = link_min["link6"]
    for L, body in LINK_BODIES.items():
        c = clearance_color(link_min[L])
        if c is None:
            c = (0.40, 0.42, 0.47, 1.0)  # neutral grey: link senses nothing nearby
        bid = model.body(body).id
        for gi in range(model.ngeom):
            if int(model.geom_bodyid[gi]) == bid and int(model.geom_group[gi]) == 2:
                model.geom_rgba[gi] = [c[0], c[1], c[2], 1.0]
    cw = clearance_color(wrist_d) or (0.40, 0.42, 0.47, 1.0)
    for body in WRIST_EXTRA:
        try:
            bid = model.body(body).id
        except Exception:
            continue
        for gi in range(model.ngeom):
            if int(model.geom_bodyid[gi]) == bid and int(model.geom_group[gi]) in (1, 2):
                model.geom_rgba[gi] = [cw[0], cw[1], cw[2], 1.0]

    # ---- 3D render with cloud overlay (clutter stays its own colour: it's group 0) ----
    cam = mjv_cam(lookat=(0.30, 0.0, 0.57), distance=2.15, azimuth=95, elevation=-19)
    img = render_scene(model, data, cam, w=940, h=900, cloud=pts, depths=depths, pt_size=0.0062)
    model.geom_rgba[:] = rgba0

    # =========================== FIGURE ===========================
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": STYLE["fg"],
        "axes.edgecolor": STYLE["grid"], "xtick.color": STYLE["fg"],
        "ytick.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
    })
    fig = plt.figure(figsize=(15.5, 9.1), dpi=170)
    fig.patch.set_facecolor(STYLE["bg"])
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.11,
                          left=0.012, right=0.955, top=0.855, bottom=0.135)

    # ---- LEFT: rendered arm tinted by clearance + cloud ----
    axL = fig.add_subplot(gs[0, 0])
    axL.imshow(img)
    axL.axis("off")
    axL.set_facecolor(STYLE["bg"])
    axL.set_title("Real Franka FR3 in clutter — each LINK tinted by its own min clearance",
                  color=STYLE["fg"], fontsize=13.5, pad=8, loc="left", fontweight="bold")

    # two legends: clearance tint (arm) and distance (cloud dots)
    sm_clear = ScalarMappable(norm=Normalize(CLOSE_M, CLEAR_M), cmap=CLEAR_CMAP)
    cax1 = axL.inset_axes([0.035, -0.060, 0.40, 0.026])
    cb1 = fig.colorbar(sm_clear, cax=cax1, orientation="horizontal")
    cb1.set_label("link clearance (m)  —  red = close, green = clear", fontsize=9.5)
    cb1.ax.tick_params(labelsize=8)
    cb1.outline.set_edgecolor(STYLE["grid"])

    sm_dist = ScalarMappable(norm=Normalize(NEAR, FAR), cmap="turbo_r")
    cax2 = axL.inset_axes([0.555, -0.060, 0.40, 0.026])
    cb2 = fig.colorbar(sm_dist, cax=cax2, orientation="horizontal")
    cb2.set_label("SPAD point-cloud distance (m)", fontsize=9.5)
    cb2.ax.tick_params(labelsize=8)
    cb2.outline.set_edgecolor(STYLE["grid"])

    axL.text(0.035, 1.012,
             "Clutter shown in its native colour. 40 SPAD depth cams (8×8, fovy 45°, "
             "0.015–0.5 m) back-projected to the point cloud.",
             transform=axL.transAxes, fontsize=9, color="#9aa0ac", va="bottom")

    # ---- RIGHT: per-link clearance bars ----
    axR = fig.add_subplot(gs[0, 1])
    axR.set_facecolor(STYLE["panel"])
    nice_names = {"link1": "L1 base", "link2": "L2 shoulder", "link3": "L3 upper arm",
                  "link4": "L4 elbow", "link5": "L5 forearm", "link6": "L6 wrist"}
    order = links[::-1]  # wrist at top
    y = np.arange(len(order))
    vals_cm, colors, labels = [], [], []
    for L in order:
        d = link_min[L]
        v = d * 100 if d < FAR else CLEAR_M * 100
        vals_cm.append(v)
        c = clearance_color(d) if d < FAR else (0.40, 0.42, 0.47, 1.0)
        colors.append(c)
        labels.append(nice_names[L])

    bars = axR.barh(y, vals_cm, color=colors, edgecolor=STYLE["bg"], height=0.66, zorder=3)
    axR.set_yticks(y)
    axR.set_yticklabels(labels, fontsize=11)
    axR.set_xlim(0, CLEAR_M * 100 + 6)
    axR.set_xlabel("min clearance (cm)", fontsize=11)
    axR.grid(axis="x", color=STYLE["grid"], lw=0.6, zorder=0)
    for s in axR.spines.values():
        s.set_color(STYLE["grid"])

    # 8 cm threshold line
    axR.axvline(THRESH_M * 100, color="#ffd166", lw=1.6, ls="--", zorder=4)
    axR.text(THRESH_M * 100 + 0.4, len(order) - 0.42, "8 cm\nthreshold",
             color="#ffd166", fontsize=9, va="top", ha="left", fontweight="bold")

    for yi, L, bar in zip(y, order, bars):
        d = link_min[L]
        if d < FAR:
            txt = f"{d*100:.1f} cm   ({link_active[L]}/{link_total[L]} SPADs)"
        else:
            txt = f"no return   (0/{link_total[L]} SPADs)"
        xt = bar.get_width()
        inside = xt > CLEAR_M * 100 * 0.62
        axR.text(xt - 0.7 if inside else xt + 0.7, yi, txt, va="center",
                 ha="right" if inside else "left", fontsize=8.8,
                 color=STYLE["bg"] if inside else STYLE["fg"], fontweight="bold", zorder=5)

    axR.set_title("Whole-arm clearance — every link, not just the gripper",
                  color=STYLE["fg"], fontsize=13.5, pad=8, loc="left", fontweight="bold")

    # headline call-out box
    headline = (f"{n_within_thresh} of 6 links within {THRESH_M*100:.0f} cm of an obstacle    •    "
                f"{total_active} of {40 - len(self_bad)} usable SPADs returning a range")
    fig.text(0.73, 0.045, headline, fontsize=12,
             color=STYLE["fg"], ha="center", va="center", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.55", fc="#1e2530", ec=STYLE["accent"], lw=1.5))

    # ---- figure-level title + thesis line ----
    fig.suptitle("franka_skin HYBRID proximity skin — WHOLE-ARM clearance from 40 SPAD sensors",
                 color=STYLE["fg"], fontsize=17, fontweight="bold", x=0.012, ha="left", y=0.978)
    fig.text(0.012, 0.935,
             "RGB cameras are blurred at policy-training time, so this skin IS the robot's "
             "perception. A wrist camera sees only the gripper's reach; the skin senses an "
             "obstacle next to the elbow, forearm, and wrist simultaneously.",
             color="#aeb4c0", fontsize=10.8, ha="left", va="top")

    out_png = os.path.join(OUT, f"{KEY}.png")
    fig.savefig(out_png, facecolor=STYLE["bg"], dpi=170)
    plt.close(fig)

    sz = os.path.getsize(out_png)
    print("SAVED", out_png, sz, "bytes")
    return out_png, sz, n_within_thresh, total_active, link_min, len(self_bad)


if __name__ == "__main__":
    main()
