"""One hazard bar SWEEPS a half-circle over the arm and slides along it; the arm avoids it
with the learned layer, and different sensors light up in turn.

The arm holds an extended posture. A single bar arcs across the OUTWARD face of the arm
(a half-circle in the plane perpendicular to the arm axis, on the side away from the base)
while sliding from the elbow (link4) to the wrist (link6). Because the arc only covers the
outward hemisphere and the arm leans AWAY from the bar, the bar stays outside the arm — it
never encircles or passes through it (the earlier full-circle version did, because a ring
centred on the arm has no escape direction). As the bar visits each link/sensor patch the
skin senses it and the proximity CVAE (skin-only) pushes that part of the arm away.

    center(t)  = lerp(elbow, wrist, frac(t))                  # slide along the arm
    alpha(t)   = (arc/2) * sin(2*pi*passes*t)                 # oscillate across the half-circle
    bar(t)     = center + R*(cos alpha * r_hat + sin alpha * t_hat)   # r_hat = outward, in-plane
    executed_q = q0 + correction
    correction += (gain * dq - decay * correction) * dt
    dq          = head(skin_with_bar) - head(skin_bar_parked)

Outputs an .mcap (Foxglove) and an H.264 .mp4 (re-encoded via ffmpeg for smooth playback).

Usage:
    OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      /opt/conda/envs/mlspaces/bin/python scripts/safety_orbit_demo.py \
      --ckpt assets/safety/cvae_v2 --out assets/safety/orbit_demo.mcap
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import safety_sweep as sw  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402

import foxglove  # noqa: E402
from foxglove import channels as C  # noqa: E402
from foxglove.schemas import (  # noqa: E402
    ArrowPrimitive, Color, CompressedImage, FrameTransform, FrameTransforms, Pose,
    Quaternion, SceneEntity, SceneUpdate, Vector3,
)
from foxglove_viz import (  # noqa: E402
    backproject, extract_body_meshes, pack_cloud, robot_mesh_scene_update,
)
from safety_flinch_demo import FPS, DT, NS, apply_aperture, heatmap_mosaic, render_all, ts_at  # noqa: E402
from safety_react_demo import bars_scene_update, extended_posture  # noqa: E402
from safety_moving_demo import heat_bgr  # noqa: E402
from train_safety_cvae import SafetyHead  # noqa: E402

LINKS = ["link1", "link2", "link3", "link4", "link5", "link6"]
D_FAR = 0.40


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("/home/jaydv/code/prox_learning/assets/safety/cvae_v3"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/jaydv/code/prox_learning/experiments_output/default/demos/orbit.mcap"))
    ap.add_argument("--runs", nargs="+", type=Path,
                    default=[Path("/home/jaydv/code/prox_learning/assets/datagen/"
                                  "hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855")])
    ap.add_argument("--radius", type=float, default=0.20,
                    help="bar distance from the arm axis (m); kept on the OUTWARD side so the bar "
                         "stays outside the arm (never penetrates) while still inside sensing range")
    ap.add_argument("--arc", type=float, default=180.0,
                    help="angular span of the sweep around the arm (deg); <360 so the bar arcs over "
                         "one outward face instead of encircling the arm")
    ap.add_argument("--passes", type=float, default=3.0, help="number of arc sweeps across the face")
    ap.add_argument("--secs", type=float, default=16.0)
    ap.add_argument("--gain", type=float, default=3.5)
    ap.add_argument("--decay", type=float, default=2.2)
    ap.add_argument("--ema", type=float, default=0.75)
    ap.add_argument("--max-dev", type=float, default=0.30)
    ap.add_argument("--bar", default="bar_m", choices=list(sw.BARS))
    ap.add_argument("--obstacle", choices=["bar", "sphere"], default="bar")
    ap.add_argument("--clean", action="store_true",
                    help="minimal overlay: only min skin depth, obstacle distance, deviation")
    ap.add_argument("--sphere-radius", type=float, default=0.045,
                    help="sphere radius when --obstacle sphere (distinct from the orbit --radius)")
    args = ap.parse_args()

    head = SafetyHead.load(args.ckpt)
    model = sw.build_model()
    hood_names = set(sw.STATIC_BOXES) | {"sash", "jamb_l", "jamb_r"}
    for gid in range(model.ngeom):
        if model.body(model.geom_bodyid[gid]).name in hood_names:
            model.geom_group[gid] = 3
    if args.obstacle == "sphere":
        sw.bars_to_spheres(model, args.sphere_radius)
    data = mujoco.MjData(model)

    sensors = sorted(model.camera(i).name.removeprefix(NS) for i in range(model.ncam)
                     if "_sensor_" in model.camera(i).name)
    cam_ids = {s: model.camera(f"{NS}{s}").id for s in sensors}
    sensors_of = {ln: [i for i, s in enumerate(sensors) if s.startswith(ln)] for ln in LINKS}
    arm_qadr = [model.joint(f"{NS}fr3_joint{i}").qposadr[0] for i in range(1, 8)]
    arm_dofadr = [model.joint(f"{NS}fr3_joint{i}").dofadr[0] for i in range(1, 8)]
    jnames = [model.joint(i).name for i in range(model.njnt)]
    finger_qadr = [model.joint(f"{NS}gripper/{n}").qposadr[0]
                   for n in ("left_driver_joint", "right_driver_joint")
                   if f"{NS}gripper/{n}" in jnames]
    mid = {n: int(model.body_mocapid[model.body(n).id])
           for n in ("sash", "jamb_l", "jamb_r", *sw.BARS, f"{NS}base")}
    base_mid = mid[f"{NS}base"]
    hand_bid = model.body(f"{NS}fr3_link7").id
    l4 = model.body(f"{NS}fr3_link4").id
    l5 = model.body(f"{NS}fr3_link5").id
    l6 = model.body(f"{NS}fr3_link6").id
    link_bids = [model.body(f"{NS}fr3_link{i}").id for i in range(2, 8)]

    geoms_of_link = {ln: [] for ln in LINKS}
    for gid in range(model.ngeom):
        if model.geom_group[gid] != 2:
            continue
        bn = model.body(model.geom_bodyid[gid]).name.replace(NS, "").replace("fr3_", "")
        if bn in LINKS:
            geoms_of_link[bn].append(gid)
    orig_rgba = model.geom_rgba.copy()

    rd = mujoco.Renderer(model, 8, 8)
    rd.enable_depth_rendering()
    rd.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[2] = 0
    opt.geomgroup[3] = 1

    ap_w, ap_h = 0.675, 0.535
    q0, grip, base = extended_posture(args.runs, model, data, arm_qadr, finger_qadr,
                                      base_mid, mid, ap_w, ap_h)
    data.mocap_pos[base_mid] = base[:3]
    data.mocap_quat[base_mid] = base[3:7]
    apply_aperture(data, mid, ap_w, ap_h)
    for n in sw.BARS:
        data.mocap_pos[mid[n]] = sw.PARK

    def pose(qarm):
        for adr, v in zip(arm_qadr, qarm):
            data.qpos[adr] = float(v)
        for adr in finger_qadr:
            data.qpos[adr] = float(grip)

    pose(q0)
    mujoco.mj_forward(model, data)
    # arm skeleton at rest (fixed reference): the bar sweeps a half-circle over the OUTWARD
    # face of the arm, sliding from the elbow (link4) to the wrist (link6). Because the arc
    # only covers the outward hemisphere and the arm leans AWAY from the bar, the bar stays
    # outside the arm — it never encircles or passes through it — while different link/sensor
    # patches light up in turn as it slides along.
    p4 = data.xpos[l4].copy()
    p5 = data.xpos[l5].copy()
    p6 = data.xpos[l6].copy()
    span = p6 - p4
    axis = span / (np.linalg.norm(span) + 1e-9)
    base_pos = data.xpos[model.body(f"{NS}base").id].copy()
    arc = np.deg2rad(args.arc)
    T = int(args.secs * FPS)
    print(f"sweep: elbow {p4.round(2)} -> wrist {p6.round(2)}, radius {args.radius} m, "
          f"arc {args.arc} deg, {args.passes} passes, {T} frames")

    rgb = mujoco.Renderer(model, 540, 960)
    vcam = mujoco.MjvCamera()
    vcam.type = mujoco.mjtCamera.mjCAMERA_FREE
    vcam.lookat[:] = p5
    vcam.distance, vcam.azimuth, vcam.elevation = 1.6, 120.0, -8.0
    vopt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(vopt)
    vopt.geomgroup[3] = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ctx = foxglove.Context()
    writer = foxglove.open_mcap(str(args.out), allow_overwrite=True, context=ctx)
    ch_tf = C.FrameTransformsChannel("/tf", context=ctx)
    ch_mesh = C.SceneUpdateChannel("/robot", context=ctx)
    ch_scene = C.SceneUpdateChannel("/scene_gt", context=ctx)
    ch_pc = C.PointCloudChannel("/proximity", context=ctx)
    ch_img = C.CompressedImageChannel("/sensors/heatmap8", context=ctx)
    ch_arrow = C.SceneUpdateChannel("/safety_arrow", context=ctx)
    ch_json = foxglove.Channel("/safety", message_encoding="json", context=ctx)
    mp4_path, tmp_path = args.out.with_suffix(".mp4"), args.out.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (960, 540))

    body_meshes = extract_body_meshes(model)
    pub_bodies = [model.body(i).name for i in range(model.nbody) if model.body(i).name in body_meshes]
    bar_name = args.bar
    correction = np.zeros(7)
    dq = np.zeros(7)
    jacp = np.zeros((3, model.nv))
    heat_ema = {ln: 0.0 for ln in LINKS}
    min_clear = np.inf

    for t in range(T):
        tau = t / max(T - 1, 1)
        frac = 0.5 * (1 - np.cos(np.pi * tau))                 # 0->1 once: slide elbow -> wrist
        center = p4 + frac * span
        alpha = 0.5 * arc * np.sin(2 * np.pi * args.passes * tau)   # oscillate across the half-circle
        # outward, in the plane perpendicular to the arm axis (points away from the base)
        out = center - base_pos
        out = out - axis * float(np.dot(out, axis))
        r_hat = out / (np.linalg.norm(out) + 1e-9)
        t_hat = np.cross(axis, r_hat)
        t_hat = t_hat / (np.linalg.norm(t_hat) + 1e-9)
        bar_dir = np.cos(alpha) * r_hat + np.sin(alpha) * t_hat    # outward hemisphere only
        bar_pos = center + args.radius * bar_dir
        pose(q0 + correction)
        data.mocap_pos[mid[bar_name]] = bar_pos
        mujoco.mj_forward(model, data)
        # clearance of the bar to the nearest arm link body (sanity: stays > 0)
        min_clear = min(min_clear, min(float(np.linalg.norm(bar_pos - data.xpos[b])) for b in (l4, l5, l6)))

        depths = render_all(model, data, rd, opt, sensors)
        data.mocap_pos[mid[bar_name]] = sw.PARK
        mujoco.mj_forward(model, data)
        depths_rest = render_all(model, data, rd, opt, sensors)
        data.mocap_pos[mid[bar_name]] = bar_pos
        mujoco.mj_forward(model, data)

        dq_raw = (head(depths) - head(depths_rest)) / max(head.scale, 1e-6)
        dq = args.ema * dq + (1 - args.ema) * dq_raw
        correction = np.clip(correction + (args.gain * dq - args.decay * correction) * DT,
                             -args.max_dev, args.max_dev)
        dev = float(np.linalg.norm(correction))

        def link_min(dl, idxs):
            vals = [float(dl[i][dl[i] >= 0.005].min()) for i in idxs if (dl[i] >= 0.005).any()]
            return min(vals) if vals else np.inf
        heat = {}
        for ln in LINKS:
            c_w = np.clip(1 - link_min(depths, sensors_of[ln]) / D_FAR, 0, 1)
            c_r = np.clip(1 - link_min(depths_rest, sensors_of[ln]) / D_FAR, 0, 1)
            heat_ema[ln] = 0.6 * heat_ema[ln] + 0.4 * float(np.clip((c_w - c_r) * 1.4, 0, 1))
            heat[ln] = heat_ema[ln]
            col = heat_bgr(heat[ln])
            for gid in geoms_of_link[ln]:
                model.geom_rgba[gid, :3] = col

        ts, ns = ts_at(t)
        tfs = []
        for bn in pub_bodies:
            bid = model.body(bn).id
            p, qt = data.xpos[bid], data.xquat[bid]
            tfs.append(FrameTransform(
                timestamp=ts, parent_frame_id="world", child_frame_id=bn,
                translation=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                rotation=Quaternion(x=float(qt[1]), y=float(qt[2]), z=float(qt[3]), w=float(qt[0]))))
        ch_tf.log(FrameTransforms(transforms=tfs), log_time=ns)
        if t == 0:
            ch_mesh.log(robot_mesh_scene_update(body_meshes, ts), log_time=ns)
        ch_scene.log(bars_scene_update(data, mid, [(bar_name, np.asarray(sw.BARS[bar_name]), bar_pos)], ts),
                     log_time=ns)

        pts_all, d_all = [], []
        for si in range(len(sensors)):
            cid = cam_ids[sensors[si]]
            c2w = np.eye(4)
            c2w[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
            c2w[:3, 3] = data.cam_xpos[cid]
            pp, ddi = backproject(depths[si], c2w, 0.015, 1.5)
            if len(pp):
                pts_all.append(pp)
                d_all.append(ddi)
        pts = np.concatenate(pts_all, 0) if pts_all else np.zeros((0, 3))
        ddc = np.concatenate(d_all, 0) if d_all else np.zeros((0,))
        ch_pc.log(pack_cloud(pts, ddc, ts, 0.02, 0.60), log_time=ns)
        ch_img.log(CompressedImage(timestamp=ts, frame_id="world", format="jpeg",
                                   data=heatmap_mosaic(depths)), log_time=ns)

        mujoco.mj_jac(model, data, jacp, None, data.xpos[hand_bid], hand_bid)
        vvec = jacp[:, arm_dofadr] @ dq
        vn = float(np.linalg.norm(vvec))
        arrows = []
        if vn > 0.02:
            dn = vvec / vn
            yaw = np.arctan2(dn[1], dn[0])
            pitch = -np.arcsin(np.clip(dn[2], -1, 1))
            cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
            cp, spn = np.cos(pitch / 2), np.sin(pitch / 2)
            p = data.xpos[hand_bid]
            arrows.append(ArrowPrimitive(
                pose=Pose(position=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                          orientation=Quaternion(x=float(-sy * spn), y=float(cy * spn),
                                                 z=float(sy * cp), w=float(cy * cp))),
                shaft_length=min(0.12 + 0.25 * vn, 0.45), shaft_diameter=0.018,
                head_length=0.05, head_diameter=0.04, color=Color(r=1.0, g=0.15, b=0.1, a=0.9)))
        ch_arrow.log(SceneUpdate(entities=[SceneEntity(timestamp=ts, frame_id="world", id="push",
                                                       frame_locked=False, arrows=arrows)]), log_time=ns)

        md = float(depths[depths >= 0.005].min()) if (depths >= 0.005).any() else float("inf")
        ch_json.log({"dev_norm": dev, "dq_norm": float(np.linalg.norm(dq)), "min_depth": md,
                     "alpha_deg": float(np.degrees(alpha)), "along_frac": float(frac),
                     "heat": {ln: heat[ln] for ln in LINKS}}, log_time=ns)

        rgb.update_scene(data, camera=vcam, scene_option=vopt)
        frame = cv2.cvtColor(rgb.render(), cv2.COLOR_RGB2BGR)
        if args.clean:
            gap = sw.min_obstacle_gap(model, data, mid, [bar_name], link_bids)
            sw.clean_hud(frame, md, gap, dev)
        else:
            cv2.putText(frame, "obstacle sweeping the arm", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
            cv2.putText(frame, f"learned avoidance: {dev:4.2f} rad", (20, 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (40, 40, 240) if dev > 0.05 else (200, 200, 200), 2, cv2.LINE_AA)
        vw.write(frame)
        model.geom_rgba[:] = orig_rgba

        if (t + 1) % 100 == 0:
            print(f"frame {t+1}/{T}  alpha {np.degrees(alpha):+4.0f}  along {frac:.2f}  "
                  f"dev {dev:.3f}  min_clear {min_clear:.3f}")

    writer.close()
    vw.release()
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-movflags", "+faststart", str(mp4_path)], check=True)
        tmp_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[warn] ffmpeg failed ({e}); keeping mp4v")
        tmp_path.replace(mp4_path)
    print(f"wrote {args.out} and {mp4_path}  ({T/FPS:.1f} s)  min bar-to-link clearance {min_clear:.3f} m")


if __name__ == "__main__":
    main()
