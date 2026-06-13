"""Moving obstacles + whole-arm reactive avoidance, with the per-link skin signal shown.

The arm holds an extended posture (its nominal). Two hazard bars PATROL up and down the
whole kinematic chain — wrist -> forearm -> elbow -> shoulder and back. As each bar nears
a link, that link's skin reads it; the proximity CVAE (skin-only) produces a whole-arm
repulsion that is added as a residual correction, so the arm pushes the threatened part
away and relaxes back to nominal when clear. Every link's geometry is tinted by its own
obstacle-induced proximity, so you can watch the signal travel across ALL the links.

    executed_q = q0 + correction
    correction += (gain * dq - decay * correction) * dt
    dq          = head(skin_with_bars) - head(skin_bars_parked)     # the bars' marginal push
    link heat   = closeness(link, with bars) - closeness(link, parked)   # per-link obstacle signal

Outputs an .mcap (Foxglove) and an annotated .mp4. The MP4 tints the arm per link and
draws a per-link signal strip (link1..link6) so the all-links coverage is explicit.

Usage:
    OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      /opt/conda/envs/mlspaces/bin/python scripts/safety_moving_demo.py \
      --ckpt assets/safety/cvae_v2 --out assets/safety/moving_demo.mcap
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
from train_safety_cvae import SafetyHead  # noqa: E402

LINKS = ["link1", "link2", "link3", "link4", "link5", "link6"]   # links carrying skin
D_FAR = 0.40   # closeness normalization for the per-link heat (m)


def build_encounters(link_seq, link_wp, link_far, T):
    """For each link in sequence: travel (at a FAR radius, outside the arm) to that link's
    home, APPROACH to its near waypoint, DWELL, RETREAT to home. The bar is only near the
    arm during an approach/dwell, never plowing across it between links. Returns (T,3)."""
    n = len(link_seq)
    budget = T // n
    n_travel = max(2, int(0.28 * budget))
    n_app = max(2, int(0.26 * budget))
    n_dwell = max(2, int(0.20 * budget))
    n_ret = max(2, budget - n_travel - n_app - n_dwell)
    segs, prev_home = [], link_far[link_seq[0]]
    for ln in link_seq:
        home, wp = link_far[ln], link_wp[ln]
        segs.append(np.linspace(prev_home, home, n_travel, endpoint=False))   # far transit
        segs.append(np.linspace(home, wp, n_app, endpoint=False))             # approach
        segs.append(np.tile(wp, (n_dwell, 1)))                                # dwell at the link
        segs.append(np.linspace(wp, home, n_ret, endpoint=False))            # retreat to far
        prev_home = home
    path = np.concatenate(segs, 0)
    if len(path) < T:
        path = np.concatenate([path, np.tile(path[-1], (T - len(path), 1))], 0)
    return path[:T]


def heat_bgr(h):
    """0 -> cool steel-blue, 1 -> hot red (BGR for cv2 / mujoco rgb is RGB; see callers)."""
    h = float(np.clip(h, 0, 1))
    cold = np.array([0.30, 0.45, 0.85])   # RGB
    hot = np.array([0.95, 0.10, 0.05])
    return (1 - h) * cold + h * hot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("/home/jaydv/code/prox_learning/assets/safety/cvae_v2"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/jaydv/code/prox_learning/assets/safety/moving_demo.mcap"))
    ap.add_argument("--runs", nargs="+", type=Path,
                    default=[Path("/home/jaydv/code/prox_learning/assets/datagen/"
                                  "hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855")])
    ap.add_argument("--gain", type=float, default=3.0)
    ap.add_argument("--decay", type=float, default=2.2)
    ap.add_argument("--ema", type=float, default=0.7)
    ap.add_argument("--max-dev", type=float, default=0.30)
    ap.add_argument("--secs", type=float, default=18.0, help="demo duration")
    ap.add_argument("--standoff", type=float, default=0.14, help="bar FACE stand-off at a link (m)")
    args = ap.parse_args()

    head = SafetyHead.load(args.ckpt)
    model = sw.build_model()
    # hood -> render group 3 (this instance only): hidden in RGB, kept in the depth render
    hood_names = set(sw.STATIC_BOXES) | {"sash", "jamb_l", "jamb_r"}
    for gid in range(model.ngeom):
        if model.body(model.geom_bodyid[gid]).name in hood_names:
            model.geom_group[gid] = 3
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

    # group-2 (visual) geoms by link body, for per-link tinting; keep originals
    geoms_of_link = {ln: [] for ln in LINKS}
    for gid in range(model.ngeom):
        if model.geom_group[gid] != 2:
            continue
        bn = model.body(model.geom_bodyid[gid]).name.replace(NS, "")
        if bn.replace("fr3_", "") in LINKS:
            geoms_of_link[bn.replace("fr3_", "")].append(gid)
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

    # per link: a NEAR waypoint (clearest sensor, FACE `standoff` off skin) and a FAR home
    # (same axis, 0.42 m out) the bar retreats to before moving on -> no plow-through.
    dep0 = render_all(model, data, rd, opt, sensors)
    link_wp, link_far = {}, {}
    for ln in LINKS:
        idxs = sensors_of[ln]
        rmin = [float(dep0[i][dep0[i] >= 0.005].min()) if (dep0[i] >= 0.005).any() else np.inf
                for i in idxs]
        i = idxs[int(np.argmax(rmin))]
        cid = cam_ids[sensors[i]]
        pos = data.cam_xpos[cid].copy()
        fwd = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
        link_wp[ln] = pos + fwd * (args.standoff + 0.025)
        link_far[ln] = pos + fwd * 0.42
    T = int(args.secs * FPS)
    # one bar makes a single slow pass wrist->shoulder, approach->dwell->retreat per link
    pathA = build_encounters(["link6", "link5", "link4", "link3", "link2"],
                             link_wp, link_far, T)
    bars = [("bar_m", np.asarray(sw.BARS["bar_m"]), pathA)]
    print(f"{T} frames ({args.secs:.0f}s), {len(bars)} bar; standoff {args.standoff} m, far home 0.42 m")

    rgb = mujoco.Renderer(model, 540, 960)
    vcam = mujoco.MjvCamera()
    vcam.type = mujoco.mjtCamera.mjCAMERA_FREE
    vcam.lookat[:] = [0.45, 0.10, 1.05]
    vcam.distance, vcam.azimuth, vcam.elevation = 2.1, 115.0, -10.0
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
    mp4_path = args.out.with_suffix(".mp4")
    tmp_path = args.out.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (960, 540))

    body_meshes = extract_body_meshes(model)
    pub_bodies = [model.body(i).name for i in range(model.nbody) if model.body(i).name in body_meshes]

    correction = np.zeros(7)
    dq = np.zeros(7)
    jacp = np.zeros((3, model.nv))
    heat_ema = {ln: 0.0 for ln in LINKS}     # temporal smoothing so the link colors don't flicker

    for t in range(T):
        pose(q0 + correction)
        for name, _half, path in bars:
            data.mocap_pos[mid[name]] = path[t]
        mujoco.mj_forward(model, data)

        depths = render_all(model, data, rd, opt, sensors)
        for name, _half, _p in bars:
            data.mocap_pos[mid[name]] = sw.PARK
        mujoco.mj_forward(model, data)
        depths_rest = render_all(model, data, rd, opt, sensors)
        for name, _half, path in bars:
            data.mocap_pos[mid[name]] = path[t]
        mujoco.mj_forward(model, data)

        dq_raw = (head(depths) - head(depths_rest)) / max(head.scale, 1e-6)
        dq = args.ema * dq + (1 - args.ema) * dq_raw
        correction = np.clip(correction + (args.gain * dq - args.decay * correction) * DT,
                             -args.max_dev, args.max_dev)
        dev = float(np.linalg.norm(correction))

        # per-link obstacle-induced closeness -> heat, and tint that link's geoms
        def link_min(dlist, idxs):
            vals = [float(dlist[i][dlist[i] >= 0.005].min()) for i in idxs if (dlist[i] >= 0.005).any()]
            return min(vals) if vals else np.inf
        heat = {}
        for ln in LINKS:
            c_w = np.clip(1 - link_min(depths, sensors_of[ln]) / D_FAR, 0, 1)
            c_r = np.clip(1 - link_min(depths_rest, sensors_of[ln]) / D_FAR, 0, 1)
            raw = float(np.clip((c_w - c_r) * 1.4, 0, 1))
            heat_ema[ln] = 0.6 * heat_ema[ln] + 0.4 * raw     # smooth -> no color strobing
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
        placed = [(name, half, data.mocap_pos[mid[name]].copy()) for name, half, _p in bars]
        ch_scene.log(bars_scene_update(data, mid, placed, ts), log_time=ns)

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
        v = jacp[:, arm_dofadr] @ dq
        vn = float(np.linalg.norm(v))
        arrows = []
        if vn > 0.02:
            dirn = v / vn
            yaw = np.arctan2(dirn[1], dirn[0])
            pitch = -np.arcsin(np.clip(dirn[2], -1, 1))
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
                     "heat": {ln: heat[ln] for ln in LINKS}}, log_time=ns)

        rgb.update_scene(data, camera=vcam, scene_option=vopt)
        frame = cv2.cvtColor(rgb.render(), cv2.COLOR_RGB2BGR)
        hot = max(LINKS, key=lambda ln: heat[ln])
        noun = "obstacle" if len(bars) == 1 else "obstacles"
        cv2.putText(frame, f"moving {noun}: {len(bars)}    joint deviation: {dev:4.2f} rad",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, f"skin active: {hot}" if heat[hot] > 0.05 else "skin clear",
                    (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (40, 40, 240) if heat[hot] > 0.05 else (200, 200, 200), 2, cv2.LINE_AA)
        # per-link signal strip (bottom): one square per link, tinted by its heat
        x0, y0, sq = 20, 500, 34
        for k, ln in enumerate(LINKS):
            c = (heat_bgr(heat[ln])[::-1] * 255).astype(int)   # RGB->BGR
            cv2.rectangle(frame, (x0 + k * (sq + 28), y0), (x0 + k * (sq + 28) + sq, y0 + sq),
                          (int(c[0]), int(c[1]), int(c[2])), -1)
            cv2.putText(frame, ln, (x0 + k * (sq + 28) - 6, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        vw.write(frame)
        # restore originals so next frame re-tints from a clean base
        model.geom_rgba[:] = orig_rgba

        if (t + 1) % 100 == 0:
            print(f"frame {t+1}/{T}  dev {dev:.3f}  hot {hot} {heat[hot]:.2f}  min_depth {md:.3f}")

    writer.close()
    vw.release()
    # re-encode to H.264 yuv420p so it plays smoothly everywhere (mp4v stutters in browsers)
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-movflags", "+faststart", str(mp4_path)], check=True)
        tmp_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[warn] ffmpeg re-encode failed ({e}); keeping mp4v")
        tmp_path.replace(mp4_path)
    print(f"wrote {args.out} and {mp4_path}  ({T/FPS:.1f} s)")


if __name__ == "__main__":
    main()
