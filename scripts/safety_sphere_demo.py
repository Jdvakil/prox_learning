"""Blue-sphere demo: the SAME safety head (trained only on orange box bars) flinches away
from a BLUE SPHERE — proof that the skin senses DEPTH, so the head is colour- and
shape-agnostic.

A blue sphere marches straight down a chosen forearm-sensor's view axis toward the arm,
then a second one on the other side. Every frame: render the 40 SPAD depths, ask the head
for a retreat RELATIVE TO REST (baseline-subtracted, so the arm is still until the sphere
adds a new close return), integrate it kinematically with a soft spring back to nominal.
The arm flinches as the sphere closes in and relaxes when it retreats — exactly as it does
for the bars, because at no point does the head see colour or know the obstacle's shape; it
only ever sees an 8x8 depth blob.

Built on the flinch demo's machinery (same posture pick, same approach schedule, same
spring-return reaction), with the box bar swapped for the blue sphere added to
safety_sweep.build_model().

Outputs a Foxglove .mcap (same channels as the flinch demo) and an H.264 .mp4.

Usage:
    OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      /opt/conda/envs/mlspaces/bin/python scripts/safety_sphere_demo.py \
      --ckpt assets/safety/cvae_v3 --out assets/safety/sphere_demo.mcap
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import safety_sweep as sw  # noqa: E402  (also installs the EGL workaround)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402

import foxglove  # noqa: E402
from foxglove import channels as C  # noqa: E402
from foxglove.schemas import (  # noqa: E402
    ArrowPrimitive, Color, CompressedImage, CubePrimitive, FrameTransform,
    FrameTransforms, Pose, Quaternion, SceneEntity, SceneUpdate, SpherePrimitive, Vector3,
)
from foxglove_viz import (  # noqa: E402
    backproject, extract_body_meshes, pack_cloud, robot_mesh_scene_update,
)
from safety_flinch_demo import (  # noqa: E402
    FPS, DT, NS, annotate, apply_aperture, bar_axis_schedule, clear_reaching_posture,
    heatmap_mosaic, pick_targets, render_all, ts_at,
)
from train_safety_cvae import SafetyHead  # noqa: E402


def scene_with_sphere(data, mid, sphere_r, ts) -> SceneUpdate:
    """Hood/bench/sash/jambs (gray cubes) + the moving blue sphere."""
    cubes = []

    def cube(c, h, col):
        cubes.append(CubePrimitive(
            pose=Pose(position=Vector3(x=float(c[0]), y=float(c[1]), z=float(c[2])),
                      orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
            size=Vector3(x=2 * float(h[0]), y=2 * float(h[1]), z=2 * float(h[2])), color=col))

    gray = Color(r=0.75, g=0.78, b=0.82, a=0.25)
    for name, (c, h) in sw.STATIC_BOXES.items():
        if name.startswith("room") or name == "floor":
            continue
        cube(c, h, gray)
    cube(data.mocap_pos[mid["sash"]], sw.SASH_HALF, gray)
    cube(data.mocap_pos[mid["jamb_l"]], sw.JAMB_HALF, gray)
    cube(data.mocap_pos[mid["jamb_r"]], sw.JAMB_HALF, gray)
    sc = data.mocap_pos[mid["sphere"]]
    spheres = []
    if sc[2] > -1:
        spheres.append(SpherePrimitive(
            pose=Pose(position=Vector3(x=float(sc[0]), y=float(sc[1]), z=float(sc[2])),
                      orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
            size=Vector3(x=2 * sphere_r, y=2 * sphere_r, z=2 * sphere_r),
            color=Color(r=0.10, g=0.35, b=0.95, a=0.95)))
    return SceneUpdate(entities=[SceneEntity(timestamp=ts, frame_id="world", id="scene",
                                             frame_locked=False, cubes=cubes, spheres=spheres)])


def caption(img_bgr, md, dev):
    """Flinch readout + a banner making the visual-agnostic point explicit."""
    annotate(img_bgr, md, dev)
    cv2.putText(img_bgr, "BLUE SPHERE  (head trained only on orange boxes)", (20, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 60), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, "skin senses DEPTH -> colour & shape agnostic", (20, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 60), 2, cv2.LINE_AA)
    return img_bgr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path,
                    default=[Path("/home/jaydv/code/prox_learning/assets/datagen/"
                                  "hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855")])
    ap.add_argument("--ckpt", type=Path,
                    default=Path("/home/jaydv/code/prox_learning/assets/safety/cvae_v3"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/jaydv/code/prox_learning/assets/safety/sphere_demo.mcap"))
    ap.add_argument("--radius", type=float, default=sw.SPHERE_R, help="sphere radius (m)")
    ap.add_argument("--gain", type=float, default=4.5, help="flinch rate (rad/s per unit head output)")
    ap.add_argument("--spring", type=float, default=1.5, help="return-to-nominal rate (1/s)")
    ap.add_argument("--max-dev", type=float, default=0.35, help="per-joint deviation clamp (rad)")
    args = ap.parse_args()

    head = SafetyHead.load(args.ckpt)
    model = sw.build_model()
    # set the sphere geom radius (build_model uses the module default)
    sgid = [g for g in range(model.ngeom)
            if model.body(model.geom_bodyid[g]).name == "sphere"][0]
    model.geom_size[sgid, 0] = args.radius
    data = mujoco.MjData(model)
    sensors = sorted(model.camera(i).name.removeprefix(NS) for i in range(model.ncam)
                     if "_sensor_" in model.camera(i).name)
    cam_ids = {s: model.camera(f"{NS}{s}").id for s in sensors}
    arm_qadr = [model.joint(f"{NS}fr3_joint{i}").qposadr[0] for i in range(1, 8)]
    arm_dofadr = [model.joint(f"{NS}fr3_joint{i}").dofadr[0] for i in range(1, 8)]
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    finger_qadr = [model.joint(f"{NS}gripper/{n}").qposadr[0]
                   for n in ("left_driver_joint", "right_driver_joint")
                   if f"{NS}gripper/{n}" in joint_names]
    mid = {n: int(model.body_mocapid[model.body(n).id])
           for n in ("sash", "jamb_l", "jamb_r", "sphere", *sw.BARS, f"{NS}base")}
    base_mid = mid[f"{NS}base"]

    rd = mujoco.Renderer(model, 8, 8)
    rd.enable_depth_rendering()
    rd.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[2] = 0

    rgb = mujoco.Renderer(model, 540, 960)
    vcam = mujoco.MjvCamera()
    vcam.type = mujoco.mjtCamera.mjCAMERA_FREE
    vcam.lookat[:] = [0.40, 0.0, 1.0]
    vcam.distance, vcam.azimuth, vcam.elevation = 2.0, 100.0, -10.0
    vopt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(vopt)

    ap_w, ap_h = 0.675, 0.535
    q0, grip, base = clear_reaching_posture(
        args.runs, model, data, rd, opt, sensors, arm_qadr, finger_qadr,
        base_mid, mid, ap_w, ap_h)
    data.mocap_pos[base_mid] = base[:3]
    data.mocap_quat[base_mid] = base[3:7]
    apply_aperture(data, mid, ap_w, ap_h)
    for n in (*sw.BARS, "sphere"):
        data.mocap_pos[mid[n]] = sw.PARK
    for adr, val in zip(arm_qadr, q0):
        data.qpos[adr] = float(val)
    for adr in finger_qadr:
        data.qpos[adr] = grip
    mujoco.mj_forward(model, data)

    # rest baseline: head output with the sphere parked (subtracted every frame)
    rest_depths = render_all(model, data, rd, opt, sensors)
    dq_rest = head(rest_depths)
    print(f"rest |head| (physical) = {np.linalg.norm(dq_rest):.3f}  -> baseline-subtracted")

    targets = pick_targets(model, data, rd, opt, sensors, cam_ids, k=2)
    half = np.array([args.radius, args.radius, args.radius])
    path = bar_axis_schedule(targets, half)
    T = len(path)
    print(f"{T} frames ({T / FPS:.1f} s), blue sphere r={args.radius}, {len(targets)} approach(es)")

    hand_bid = model.body(f"{NS}fr3_link7").id

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
    pub_bodies = [model.body(i).name for i in range(model.nbody)
                  if model.body(i).name in body_meshes]
    q = q0.copy()
    jacp = np.zeros((3, model.nv))
    dev_peak = 0.0
    md_min_seen = np.inf

    for t in range(T):
        if np.isfinite(path[t]).all():
            data.mocap_pos[mid["sphere"]] = path[t]
        else:
            data.mocap_pos[mid["sphere"]] = sw.PARK
        for adr, val in zip(arm_qadr, q):
            data.qpos[adr] = float(val)
        mujoco.mj_forward(model, data)

        depths = render_all(model, data, rd, opt, sensors)
        pts_all, d_all = [], []
        for si, s in enumerate(sensors):
            cid = cam_ids[s]
            c2w = np.eye(4)
            c2w[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
            c2w[:3, 3] = data.cam_xpos[cid]
            pts, dd = backproject(depths[si], c2w, 0.015, 1.5)
            if len(pts):
                pts_all.append(pts)
                d_all.append(dd)

        dq = (head(depths) - dq_rest) / max(head.scale, 1e-6)   # baseline-subtracted, unit-RMS
        q = q + (args.gain * dq - args.spring * (q - q0)) * DT
        q = np.clip(q, q0 - args.max_dev, q0 + args.max_dev)
        dev = float(np.linalg.norm(q - q0))
        dev_peak = max(dev_peak, dev)

        ts, ns = ts_at(t)
        tfs = []
        for bname in pub_bodies:
            bid = model.body(bname).id
            p, qt = data.xpos[bid], data.xquat[bid]
            tfs.append(FrameTransform(
                timestamp=ts, parent_frame_id="world", child_frame_id=bname,
                translation=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                rotation=Quaternion(x=float(qt[1]), y=float(qt[2]),
                                    z=float(qt[3]), w=float(qt[0]))))
        ch_tf.log(FrameTransforms(transforms=tfs), log_time=ns)
        if t == 0:
            ch_mesh.log(robot_mesh_scene_update(body_meshes, ts), log_time=ns)
        ch_scene.log(scene_with_sphere(data, mid, args.radius, ts), log_time=ns)
        pts = np.concatenate(pts_all, 0) if pts_all else np.zeros((0, 3))
        dd = np.concatenate(d_all, 0) if d_all else np.zeros((0,))
        ch_pc.log(pack_cloud(pts, dd, ts, 0.02, 0.60), log_time=ns)
        ch_img.log(CompressedImage(timestamp=ts, frame_id="world", format="jpeg",
                                   data=heatmap_mosaic(depths)), log_time=ns)

        # EE-space push arrow at the wrist
        mujoco.mj_jac(model, data, jacp, None, data.xpos[hand_bid], hand_bid)
        v = jacp[:, arm_dofadr] @ dq
        vn = float(np.linalg.norm(v))
        arrows = []
        if vn > 0.02:
            d = v / vn
            yaw = np.arctan2(d[1], d[0])
            pitch = -np.arcsin(np.clip(d[2], -1, 1))
            cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
            cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
            ln = min(0.12 + 0.25 * vn, 0.45)
            p = data.xpos[hand_bid]
            arrows.append(ArrowPrimitive(
                pose=Pose(position=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                          orientation=Quaternion(x=float(-sy * sp), y=float(cy * sp),
                                                 z=float(sy * cp), w=float(cy * cp))),
                shaft_length=ln, shaft_diameter=0.018,
                head_length=0.05, head_diameter=0.04,
                color=Color(r=0.1, g=0.5, b=1.0, a=0.9)))
        ch_arrow.log(SceneUpdate(entities=[SceneEntity(
            timestamp=ts, frame_id="world", id="push", frame_locked=False,
            arrows=arrows)]), log_time=ns)

        md = float(depths[depths >= 0.005].min()) if (depths >= 0.005).any() else float("inf")
        md_min_seen = min(md_min_seen, md)
        sphere_gap = float(np.linalg.norm(data.mocap_pos[mid["sphere"]] - data.xpos[hand_bid]))
        ch_json.log({"dq": [float(x) for x in dq], "dq_norm": float(np.linalg.norm(dq)),
                     "min_depth": md, "sphere_gap": sphere_gap, "dev_norm": dev}, log_time=ns)

        rgb.update_scene(data, camera=vcam, scene_option=vopt)
        vw.write(caption(cv2.cvtColor(rgb.render(), cv2.COLOR_RGB2BGR), md, dev))

        if (t + 1) % 100 == 0:
            print(f"frame {t + 1}/{T}  min_depth {md:.3f}  |dq| {np.linalg.norm(dq):.2f}  dev {dev:.3f}")

    writer.close()
    vw.release()
    # re-encode to H.264 for smooth playback
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4_path)], check=True)
        tmp_path.unlink()
    except Exception as e:  # noqa: BLE001
        tmp_path.replace(mp4_path)
        print(f"[ffmpeg re-encode skipped: {e}]")
    print(f"wrote {args.out}  ({T / FPS:.1f} s)  peak dev {dev_peak:.3f} rad  "
          f"min skin depth seen {md_min_seen:.3f} m")
    print(f"wrote {mp4_path}")


if __name__ == "__main__":
    main()
