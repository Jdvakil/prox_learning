"""Same sensor test + reconstruction as test_and_reconstruct_hybrid.py, but the arm reaches
INTO A FUMEHOOD (bench + side walls + back + top + raised sash rail) with objects on the bench.

A. PLACEMENT  — robot-only (scene-independent): not-buried / outward / plate@0.15m.
B. RECONSTRUCT — single frame: 40-sensor cloud vs all real surfaces (hood + objects).
C. ACCUMULATE  — sweep the reach; accumulate the cloud into a dense hood+object reconstruction.

Out: diagnostics_output/<ts>_hybrid_fumehood_reconstruct/ (report.txt, figure, RGB render).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_viz_lib import (build, set_pose, sensors, depth8, cam_pose, mocap_set,  # noqa: E402
                            add_box, add_plane_mocap, depth_renderer, skin_cloud,
                            render_scene, mjv_cam, nice_lights, FAR)
from test_and_reconstruct_hybrid import gt_boxes, cloud_error, cloud_panel  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_fumehood_reconstruct")
OUT.mkdir(parents=True, exist_ok=True)
BG, PANEL, FG = "#111317", "#171a20", "#e8e8ea"
REACH = [0.0, -0.30, 0.0, -2.25, 0.0, 2.0, 0.79]
REACH2 = [0.35, -0.10, 0.15, -1.95, 0.1, 1.75, 0.5]

# fumehood (translucent shell so the arm shows) + objects on the bench, sized to the lib robot
HOOD = [
    ("bench_top", [0.62, 0.0, 0.585], [0.27, 0.32, 0.015], [0.62, 0.55, 0.45, 1]),
    ("bench_body", [0.62, 0.0, 0.29], [0.25, 0.30, 0.29], [0.55, 0.50, 0.44, 1]),
    ("hood_l", [0.62, 0.32, 0.82], [0.27, 0.012, 0.22], [0.78, 0.80, 0.84, 0.32]),
    ("hood_r", [0.62, -0.32, 0.82], [0.27, 0.012, 0.22], [0.78, 0.80, 0.84, 0.32]),
    ("hood_back", [0.90, 0.0, 0.82], [0.012, 0.32, 0.22], [0.72, 0.70, 0.66, 1]),
    ("hood_top", [0.62, 0.0, 1.05], [0.27, 0.32, 0.012], [0.78, 0.80, 0.84, 0.32]),
    ("sash", [0.36, 0.0, 0.90], [0.012, 0.30, 0.028], [0.62, 0.64, 0.66, 1]),
]
OBJ = [
    ("o_box", [0.58, 0.10, 0.64], [0.045, 0.045, 0.05], [0.85, 0.5, 0.3, 1]),
    ("o_tube", [0.55, -0.12, 0.65], [0.03, 0.03, 0.06], [0.5, 0.6, 0.85, 1]),
    ("o_block", [0.72, 0.0, 0.625], [0.06, 0.04, 0.04], [0.8, 0.75, 0.4, 1]),
]


def mk_hood(s):
    nice_lights(s)
    for n, c, h, col in HOOD + OBJ:
        add_box(s, n, c, h, col)


def main():
    rep = ["# Fumehood reconstruction\n"]

    # A. placement (robot only) ----------------------------------------------------------
    POSES = [REACH, [0.0, 0.3, 0.0, -1.2, 0.0, 1.6, 0.0], [0.6, -0.9, 0.4, -2.4, 0.3, 2.2, 0.4]]
    m0 = build(lambda s: None)
    d0 = mujoco.MjData(m0)
    names = sensors(m0)
    rd = depth_renderer(m0)
    sw = {n: -1 for n in names}
    for q in POSES:
        set_pose(m0, d0, q)
        for n in names:
            sw[n] = max(sw[n], float(depth8(rd, d0, n).min()))
    set_pose(m0, d0, REACH)
    outw = 0
    for n in names:
        cid = m0.camera(n).id
        bid = int(m0.cam_bodyid[cid])
        pos, R = cam_pose(m0, d0, n)
        rad = pos - d0.xpos[bid]
        outw += float(np.dot(-R[:, 2], rad / (np.linalg.norm(rad) + 1e-9))) > -0.2
    mp = build(lambda s: add_plane_mocap(s))
    dp = mujoco.MjData(mp); set_pose(mp, dp, REACH); rdp = depth_renderer(mp)
    pok = 0
    for n in names:
        pos, R = cam_pose(mp, dp, n); fwd = -R[:, 2]
        mocap_set(mp, dp, "probe_plane", pos + 0.15 * fwd, view_dir=fwd)
        pok += abs(float(depth8(rdp, dp, n)[3:5, 3:5].mean()) - 0.145) < 0.012
    nb = sum(sw[n] > 0.03 for n in names)
    rep.append(f"**A. Placement (robot-only):** not-buried {nb}/40, outward {outw}/40, plate@0.15m {pok}/40")

    # B. single-frame reconstruction in the hood ----------------------------------------
    mH = build(mk_hood); dH = mujoco.MjData(mH); set_pose(mH, dH, REACH)
    pts, dd, mins = skin_cloud(mH, dH)
    err = cloud_error(pts, gt_boxes(mH, dH))
    active = sum(1 for v in mins.values() if v < FAR)
    rep.append(f"**B. Reconstruction (1 frame):** {active}/40 active, {len(pts)} pts, "
               f"RMS {np.sqrt((err**2).mean())*1000:.1f}mm, {100*(err<0.01).mean():.0f}% within 1cm")

    # C. accumulation over the reach ----------------------------------------------------
    q0, q1 = np.array(REACH), np.array(REACH2)
    accP, accD = [], []
    for t in np.linspace(0, 1, 14):
        set_pose(mH, dH, list(q0 * (1 - t) + q1 * t))
        p, dq, _ = skin_cloud(mH, dH)
        if len(p):
            accP.append(p); accD.append(dq)
    accP = np.concatenate(accP) if accP else np.zeros((0, 3))
    accD = np.concatenate(accD) if accD else np.zeros((0,))
    errA = cloud_error(accP, gt_boxes(mH, dH))
    rep.append(f"**C. Accumulation (14 poses):** {len(accP)} pts, "
               f"RMS {np.sqrt((errA**2).mean())*1000:.1f}mm, {100*(errA<0.01).mean():.0f}% within 1cm")

    # figure ----------------------------------------------------------------------------
    def draw_gt(ax):
        for n, c, h, col in HOOD + OBJ:
            c = np.array(c); h = np.array(h)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    ax.plot([c[0]-h[0], c[0]+h[0]], [c[1]+sy*h[1]]*2, [c[2]+sx*h[2]]*2, color="#777", lw=0.4)

    fig = plt.figure(figsize=(13, 5.6), facecolor=BG)
    a1 = fig.add_subplot(121, projection="3d")
    cloud_panel(a1, pts, dd, draw_gt, f"1 frame — {len(pts)} pts, RMS {np.sqrt((err**2).mean())*1000:.1f}mm")
    a2 = fig.add_subplot(122, projection="3d")
    cloud_panel(a2, accP, accD, draw_gt, f"accumulated — {len(accP)} pts")
    fig.suptitle("Hybrid skin in a fumehood — reconstruction", color=FG, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fumehood_reconstruction.png", dpi=150, facecolor=BG)

    # pretty RGB + cloud render ---------------------------------------------------------
    set_pose(mH, dH, REACH)
    img = render_scene(mH, dH, mjv_cam(lookat=(0.55, 0, 0.7), distance=1.55, azimuth=150, elevation=-14),
                       w=1100, h=900, cloud=pts, depths=dd, pt_size=0.009)
    cv2.imwrite(str(OUT / "fumehood_render.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    (OUT / "report.txt").write_text("\n".join(rep))
    print("\n".join(rep))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
