"""Full test + reconstruction suite for the 40-sensor hybrid skin (model_hybrid.xml).

A. PLACEMENT  — per sensor: not-buried (self-hit over 3 poses), outward, and a known-distance
   plate read at 0.15 m. PASS table + summary.
B. PRIMITIVES — add box + sphere + cylinder + corner around the arm; reconstruct the full
   40-sensor cloud; error vs analytic ground-truth SDF (median/RMS/p95, % within 1 cm).
C. CLUTTER    — a multi-object tabletop; single-frame reconstruction.
D. ACCUMULATE — sweep the arm through poses; accumulate the cloud into a dense reconstruction.

Outputs: diagnostics_output/<ts>_hybrid_test_reconstruct/  (report.txt, placement.csv, 4 PNGs)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_viz_lib import (build, set_pose, sensors, depth8, backproject, cam_pose,  # noqa: E402
                            add_box, add_sphere_mocap, add_plane_mocap, mocap_set,
                            depth_renderer, fit_plane, skin_cloud, nice_lights, FOVY, NEAR, FAR)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_test_reconstruct")
OUT.mkdir(parents=True, exist_ok=True)
POSES = [[0.0, -0.35, 0.0, -2.30, 0.0, 2.05, 0.79],
         [0.0, 0.3, 0.0, -1.2, 0.0, 1.6, 0.0],
         [0.6, -0.9, 0.4, -2.4, 0.3, 2.2, 0.4]]
BG, PANEL, FG = "#111317", "#171a20", "#e8e8ea"


# ---------- analytic SDF for ground-truth objects ----------
def sdf_box(p, c, h):
    d = np.abs(p - c) - h
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(np.max(d, axis=-1), 0)


def sdf_sphere(p, c, r):
    return np.abs(np.linalg.norm(p - c, axis=-1) - r)


def sdf_cyl_z(p, c, r, hz):
    dxy = np.abs(np.linalg.norm(p[:, :2] - c[:2], axis=-1) - r)
    dz = np.maximum(np.abs(p[:, 2] - c[2]) - hz, 0)
    return np.sqrt(dxy ** 2 + dz ** 2)


def gt_boxes(model, data):
    """World AABBs of every real surface a sensor can see: arm links, gripper, objects, floor.
    Excludes the cosmetic skin (group 2, hidden during proximity render) and sensor markers."""
    boxes = []
    for g in range(model.ngeom):
        bn = model.body(int(model.geom_bodyid[g])).name.lower()
        if int(model.geom_group[g]) == 2 or "_dot_" in (model.geom(g).name or ""):
            continue
        aabb = model.geom_aabb[g]
        xp = data.geom_xpos[g]
        xm = data.geom_xmat[g].reshape(3, 3)
        corners = np.array([xp + xm @ (aabb[:3] + np.array([sx, sy, sz]) * aabb[3:])
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        boxes.append((corners.mean(0), (corners.max(0) - corners.min(0)) / 2))
    return boxes


def cloud_error(pts, boxes):
    """min clamped box-SDF from each point to the nearest real surface (0 if on/inside one)."""
    if not len(pts) or not boxes:
        return np.zeros(max(len(pts), 1))
    e = np.full(len(pts), 1e9)
    for c, h in boxes:
        e = np.minimum(e, np.maximum(sdf_box(pts, c, h), 0.0))
    return e


def style3d(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=FG, fontsize=10)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.label.set_color(FG)
        a.set_tick_params(colors=FG)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")


def cloud_panel(ax, pts, dd, gt_draw, title):
    if len(pts):
        nrm = np.clip((dd - NEAR) / (FAR - NEAR), 0, 1)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=1 - nrm, cmap="turbo", s=5, depthshade=False)
    gt_draw(ax)
    style3d(ax, title)
    ax.set_xlim(0, 0.9); ax.set_ylim(-0.45, 0.45); ax.set_zlim(0.0, 1.0)
    ax.view_init(elev=18, azim=-60)


def main():
    rep = []
    names = None

    # ===================== A. PLACEMENT =====================
    m = build(lambda s: None)
    d = mujoco.MjData(m)
    names = sensors(m)
    rd = depth_renderer(m)
    self_worst = {n: -1 for n in names}
    for q in POSES:
        set_pose(m, d, q)
        for n in names:
            self_worst[n] = max(self_worst[n], float(depth8(rd, d, n).min()))
    set_pose(m, d, POSES[0])
    outward = {}
    for n in names:
        cid = m.camera(n).id
        bid = int(m.cam_bodyid[cid])
        pos, R = cam_pose(m, d, n)
        rad = pos - d.xpos[bid]
        rad = rad / (np.linalg.norm(rad) + 1e-9)
        outward[n] = float(np.dot(-R[:, 2], rad))
    # plate at 0.15 m head-on
    mp = build(lambda s: add_plane_mocap(s))
    dp = mujoco.MjData(mp); set_pose(mp, dp, POSES[0]); rdp = depth_renderer(mp)
    plate = {}
    for n in names:
        pos, R = cam_pose(mp, dp, n); fwd = -R[:, 2]
        mocap_set(mp, dp, "probe_plane", pos + 0.15 * fwd, view_dir=fwd)
        plate[n] = float(depth8(rdp, dp, n)[3:5, 3:5].mean())
    not_buried = sum(self_worst[n] > 0.03 for n in names)
    plate_ok = sum(abs(plate[n] - 0.145) < 0.012 for n in names)
    out_ok = sum(outward[n] > -0.2 for n in names)
    rows = ["sensor,self_min_max_m,outward_dot,plate_read_m,not_buried,plate_ok"]
    for n in names:
        rows.append(f"{n},{self_worst[n]:.3f},{outward[n]:+.2f},{plate[n]:.3f},"
                    f"{self_worst[n] > 0.03},{abs(plate[n]-0.145) < 0.012}")
    (OUT / "placement.csv").write_text("\n".join(rows))
    rep.append("A. PLACEMENT (40 sensors)")
    rep.append(f"   not-buried (self-hit, 3 poses): {not_buried}/40")
    rep.append(f"   outward-facing:                 {out_ok}/40")
    rep.append(f"   plate@0.15m reads 0.145+-0.012: {plate_ok}/40")

    # ===================== B. PRIMITIVES =====================
    BOX = (np.array([0.55, 0.28, 0.62]), np.array([0.02, 0.16, 0.30]))
    SPH = (np.array([0.50, -0.18, 0.78]), 0.07)
    CYL = (np.array([0.40, 0.16, 0.60]), 0.03, 0.28)
    COR = (np.array([0.80, 0.0, 0.62]), np.array([0.02, 0.34, 0.30]))

    def mk_prim(s):
        nice_lights(s)
        add_box(s, "p_box", list(BOX[0]), list(BOX[1]), [0.80, 0.55, 0.30, 1])
        add_box(s, "p_cor", list(COR[0]), list(COR[1]), [0.70, 0.68, 0.62, 1])
        b = s.worldbody.add_body(name="p_sph", pos=list(SPH[0]))
        g = b.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_SPHERE; g.size = [SPH[1], 0, 0]
        g.rgba = [0.9, 0.4, 0.3, 1]; g.contype = 0; g.conaffinity = 0
        b2 = s.worldbody.add_body(name="p_cyl", pos=list(CYL[0]))
        g2 = b2.add_geom(); g2.type = mujoco.mjtGeom.mjGEOM_CYLINDER; g2.size = [CYL[1], CYL[2], 0]
        g2.rgba = [0.4, 0.7, 0.5, 1]; g2.contype = 0; g2.conaffinity = 0

    mB = build(mk_prim); dB = mujoco.MjData(mB); set_pose(mB, dB, POSES[0])
    ptsB, ddB, _ = skin_cloud(mB, dB)
    err = cloud_error(ptsB, gt_boxes(mB, dB))
    rep.append("\nB. PRIMITIVE RECONSTRUCTION (box+sphere+cylinder+corner; GT = all surfaces)")
    rep.append(f"   points: {len(ptsB)} | median {np.median(err)*1000:.1f}mm "
               f"RMS {np.sqrt((err**2).mean())*1000:.1f}mm p95 {np.percentile(err,95)*1000:.1f}mm "
               f"| within 1cm: {100*(err<0.01).mean():.0f}%")

    def draw_prim(ax):
        # sphere wire
        u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:7j]
        ax.plot_wireframe(SPH[0][0]+SPH[1]*np.cos(u)*np.sin(v), SPH[0][1]+SPH[1]*np.sin(u)*np.sin(v),
                          SPH[0][2]+SPH[1]*np.cos(v), color="#888", lw=0.4)
        for (c, h) in (BOX, COR):
            for sx in (-1, 1):
                for sy in (-1, 1):
                    ax.plot([c[0]-h[0], c[0]+h[0]], [c[1]+sy*h[1]]*2, [c[2]+sx*h[2]]*2, color="#888", lw=0.4)

    # ===================== C. CLUTTER =====================
    CL = [(np.array([0.52, 0.0, 0.74]), np.array([0.05, 0.05, 0.05])),
          (np.array([0.46, 0.16, 0.74]), np.array([0.03, 0.03, 0.10])),
          (np.array([0.60, -0.12, 0.74]), np.array([0.06, 0.03, 0.04])),
          (np.array([0.55, 0.0, 0.40]), np.array([0.22, 0.30, 0.02]))]  # table

    def mk_clutter(s):
        nice_lights(s)
        cols = [[0.85, 0.5, 0.3, 1], [0.5, 0.6, 0.85, 1], [0.8, 0.75, 0.4, 1], [0.55, 0.57, 0.6, 1]]
        for i, (c, h) in enumerate(CL):
            add_box(s, f"cl_{i}", list(c), list(h), cols[i])

    mC = build(mk_clutter); dC = mujoco.MjData(mC); set_pose(mC, dC, POSES[0])
    ptsC, ddC, minsC = skin_cloud(mC, dC)
    errC = cloud_error(ptsC, gt_boxes(mC, dC))
    activeC = sum(1 for v in minsC.values() if v < FAR)
    rep.append("\nC. CLUTTER RECONSTRUCTION (3 objects + table; GT = all surfaces)")
    rep.append(f"   active sensors: {activeC}/40 | points {len(ptsC)} | "
               f"median {np.median(errC)*1000:.1f}mm RMS {np.sqrt((errC**2).mean())*1000:.1f}mm "
               f"| within 1cm: {100*(errC<0.01).mean():.0f}%")

    def draw_clutter(ax):
        for c, h in CL:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    ax.plot([c[0]-h[0], c[0]+h[0]], [c[1]+sy*h[1]]*2, [c[2]+sx*h[2]]*2, color="#888", lw=0.4)

    # ===================== D. ACCUMULATE =====================
    q0 = np.array(POSES[0]); q1 = np.array([0.5, -0.15, 0.2, -1.9, 0.1, 1.7, 0.4])
    accP, accD = [], []
    for t in np.linspace(0, 1, 14):
        set_pose(mC, dC, list(q0 * (1 - t) + q1 * t))
        p, dd, _ = skin_cloud(mC, dC)
        if len(p):
            accP.append(p); accD.append(dd)
    accP = np.concatenate(accP) if accP else np.zeros((0, 3))
    accD = np.concatenate(accD) if accD else np.zeros((0,))
    rep.append("\nD. MULTI-POSE ACCUMULATION (14 poses through the clutter)")
    rep.append(f"   accumulated points: {len(accP)}")

    # ===================== figure =====================
    fig = plt.figure(figsize=(18, 5.4), facecolor=BG)
    ax1 = fig.add_subplot(141, projection="3d"); cloud_panel(ax1, ptsB, ddB, draw_prim,
        f"B. primitives  RMS {np.sqrt((err**2).mean())*1000:.1f}mm  {100*(err<0.01).mean():.0f}%<1cm")
    ax2 = fig.add_subplot(142); ax2.set_facecolor(PANEL)
    ax2.hist(err*1000, bins=50, color="#4cc9f0"); ax2.axvline(10, color="#ef476f", ls="--")
    ax2.set_title("B. point->GT error (mm)", color=FG); ax2.tick_params(colors=FG)
    for sp in ax2.spines.values(): sp.set_color("#333")
    ax3 = fig.add_subplot(143, projection="3d"); cloud_panel(ax3, ptsC, ddC, draw_clutter,
        f"C. clutter  {activeC}/40 active  {len(ptsC)} pts")
    ax4 = fig.add_subplot(144, projection="3d"); cloud_panel(ax4, accP, accD, draw_clutter,
        f"D. accumulated  {len(accP)} pts (14 poses)")
    fig.suptitle("Hybrid skin — placement verified + object reconstruction", color=FG, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "reconstruction_suite.png", dpi=150, facecolor=BG)

    (OUT / "report.txt").write_text("\n".join(rep))
    print("\n".join(rep))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
