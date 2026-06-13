"""Flinch demo: the trained safety head reacting live to an approaching hazard bar.

Loads the safety CVAE (scripts/train_safety_cvae.py checkpoint), poses the hybrid-skin
FR3 at a CLEAR forward-reaching posture (a real collected frame whose skin reads far at
rest, so nothing is touching the skin until the bar arrives), then marches a hazard bar
straight down a chosen exposed forearm sensor's view axis — first one sensor, then a
second on the other side. Every frame: render the 40 SPAD depths, ask the head for a
retreat delta RELATIVE TO REST (baseline-subtracted, so the arm is still until the bar
adds a new close return), integrate it kinematically with a soft spring back to nominal.
The arm visibly flinches away as the bar closes in and relaxes home when it retreats.

Why baseline subtraction: the head fires on ANY close surface (hood walls, the arm's own
links). Subtracting its rest output makes the demo react only to the CHANGE the bar
causes — dev = 0 when the bar is parked, regardless of static clutter.

Written to an .mcap for Foxglove:
    /tf /robot           robot meshes (same as foxglove_viz)
    /scene_gt            hood/bench/sash/jambs (gray) + the moving bar (red)
    /proximity           back-projected skin returns, turbo-colored by range
    /sensors/heatmap8    mosaic of the raw 8x8 sensor frames (targeted tile flashes red)
    /safety              json: per-joint dq, |dq|, min skin depth, bar gap, dev
    /safety_arrow        EE-space push direction (J @ dq) drawn at the wrist

Usage:
    OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      /opt/conda/envs/mlspaces/bin/python scripts/safety_flinch_demo.py \
      --runs assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855 \
      --ckpt assets/safety/cvae_v1 --out assets/safety/flinch_demo.mcap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import safety_sweep as sw  # noqa: E402  (also installs the EGL workaround)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from matplotlib import colormaps  # noqa: E402

import foxglove  # noqa: E402
from foxglove import channels as C  # noqa: E402
from foxglove.schemas import (  # noqa: E402
    ArrowPrimitive, Color, CompressedImage, CubePrimitive, FrameTransform,
    FrameTransforms, Pose, Quaternion, SceneEntity, SceneUpdate, Timestamp, Vector3,
)
from foxglove_viz import (  # noqa: E402
    backproject, extract_body_meshes, pack_cloud, robot_mesh_scene_update,
)
from train_safety_cvae import SafetyHead  # noqa: E402

NS = sw.NS
FPS = 30
DT = 1.0 / FPS
TURBO = colormaps["turbo"]
# forearm / wrist sensors that face outward — good targets for an approaching bar
FOREARM_PREFIXES = ("link5_front", "link5_back", "link6", "link4", "link2")


def ts_at(frame: int) -> tuple[Timestamp, int]:
    ns = int(round(frame * DT * 1e9))
    return Timestamp(sec=ns // 1_000_000_000, nsec=ns % 1_000_000_000), ns


def render_all(model, data, rd, opt, sensors) -> np.ndarray:
    depths = np.zeros((len(sensors), 8, 8), np.float32)
    for si, s in enumerate(sensors):
        rd.update_scene(data, camera=f"{NS}{s}", scene_option=opt)
        depths[si] = rd.render()
    return depths


def apply_aperture(data, mid, ap_w, ap_h):
    data.mocap_pos[mid["sash"]] = [sw.TUBE_X0, 0.0, sw.BENCH_Z + ap_h + 0.025]
    data.mocap_pos[mid["jamb_l"]] = [sw.TUBE_X0, ap_w / 2 + 0.18, sw.BENCH_Z + 0.20]
    data.mocap_pos[mid["jamb_r"]] = [sw.TUBE_X0, -ap_w / 2 - 0.18, sw.BENCH_Z + 0.20]


def clear_reaching_posture(runs, model, data, rd, opt, sensors, arm_qadr, finger_qadr,
                           base_mid, mid, ap_w, ap_h, n_cand=300, min_hand_x=0.38):
    """Pick the real collected frame that reaches forward (hand x > min_hand_x) AND whose
    skin is clearest at rest (max over candidates of the all-sensor minimum depth)."""
    q_all, grip_all, base_all, _ = sw.load_postures([Path(p) for p in runs])
    hand_bid = model.body(f"{NS}fr3_link7").id
    rng = np.random.default_rng(0)
    rows = rng.choice(len(q_all), size=min(n_cand, len(q_all)), replace=False)
    for n in sw.BARS:
        data.mocap_pos[mid[n]] = sw.PARK
    apply_aperture(data, mid, ap_w, ap_h)
    best, best_clear = None, -np.inf
    for r in rows:
        data.mocap_pos[base_mid] = base_all[r][:3]
        data.mocap_quat[base_mid] = base_all[r][3:7]
        for adr, val in zip(arm_qadr, q_all[r]):
            data.qpos[adr] = float(val)
        for adr in finger_qadr:
            data.qpos[adr] = float(grip_all[r])
        mujoco.mj_forward(model, data)
        if float(data.xpos[hand_bid][0]) < min_hand_x:
            continue
        depths = render_all(model, data, rd, opt, sensors)
        m = depths >= 0.005
        clear = float(depths[m].min()) if m.any() else np.inf
        if clear > best_clear:
            best_clear, best = clear, r
    print(f"posture: row {best}, rest skin clearance = {best_clear:.3f} m")
    return q_all[best].copy(), float(grip_all[best]), base_all[best].copy()


def pick_targets(model, data, rd, opt, sensors, cam_ids, k=2):
    """Among forearm sensors, the k most exposed (largest rest depth), spread across the
    +y / -y sides if possible. Returns list of (sensor, cam_pos, forward_axis, z_height)."""
    depths = render_all(model, data, rd, opt, sensors)
    rest_min = np.array([float(depths[i][depths[i] >= 0.005].min())
                         if (depths[i] >= 0.005).any() else np.inf
                         for i in range(len(sensors))])
    cand = [(i, sensors[i]) for i in range(len(sensors))
            if sensors[i].startswith(FOREARM_PREFIXES)]
    cand.sort(key=lambda t: -rest_min[t[0]])
    chosen, sides = [], set()
    for i, s in cand:
        cid = cam_ids[s]
        pos = data.cam_xpos[cid].copy()
        side = np.sign(pos[1]) or 1.0
        if len(chosen) and side in sides and len(cand) > k:
            continue                          # prefer a second target on the other side
        fwd = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
        chosen.append((s, pos, fwd, float(pos[2])))
        sides.add(side)
        if len(chosen) >= k:
            break
    for s, pos, fwd, z in chosen:
        print(f"  target sensor {s:22s} rest={rest_min[sensors.index(s)]:.2f}m  "
              f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
    return chosen


def bar_axis_schedule(targets, half, d_far=0.40, d_near=0.12):
    """For each target: march the bar from d_far -> d_near along the sensor view axis
    (12 cm/s), dwell 1.2 s, retreat (25 cm/s), pause parked. (T,3) centers; nan = parked.
    The bar stays a vertical rod; only its centre rides the ray. d_near is the bar FACE
    stand-off — kept ~0.10 m so the rod covers PART of the 8x8 frame (the head's strong,
    in-distribution band) instead of flooding it at point-blank range (OOD -> weak dq)."""
    legs = []
    for (_s, pos, fwd, _z) in targets:
        face = half[1]                       # bring the bar FACE (not centre) to d_near
        in_n = max(4, int((d_far - d_near) / (0.12 * DT)))
        out_n = max(4, int((d_far - d_near) / (0.25 * DT)))
        ds_in = np.linspace(d_far, d_near + face, in_n)
        ds_out = np.linspace(d_near + face, d_far, out_n)
        for ds in (ds_in,):
            legs.append(pos[None, :] + np.outer(ds, fwd))
        legs.append(np.tile(pos + fwd * (d_near + face), (int(1.2 * FPS), 1)))   # dwell
        legs.append(pos[None, :] + np.outer(ds_out, fwd))
        legs.append(np.full((int(0.8 * FPS), 3), np.nan))                         # parked
    return np.concatenate(legs, 0)


def scene_boxes_update(data, mid, bar_name, bar_half, ts) -> SceneUpdate:
    cubes = []

    def cube(c, h, col):
        cubes.append(CubePrimitive(
            pose=Pose(position=Vector3(x=float(c[0]), y=float(c[1]), z=float(c[2])),
                      orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
            size=Vector3(x=2 * float(h[0]), y=2 * float(h[1]), z=2 * float(h[2])),
            color=col))

    gray = Color(r=0.75, g=0.78, b=0.82, a=0.25)
    for name, (c, h) in sw.STATIC_BOXES.items():
        if name.startswith("room") or name == "floor":
            continue
        cube(c, h, gray)
    cube(data.mocap_pos[mid["sash"]], sw.SASH_HALF, gray)
    cube(data.mocap_pos[mid["jamb_l"]], sw.JAMB_HALF, gray)
    cube(data.mocap_pos[mid["jamb_r"]], sw.JAMB_HALF, gray)
    bc = data.mocap_pos[mid[bar_name]]
    if bc[2] > -1:
        cube(bc, bar_half, Color(r=1.0, g=0.30, b=0.05, a=0.95))
    return SceneUpdate(entities=[SceneEntity(timestamp=ts, frame_id="world", id="scene",
                                             frame_locked=False, cubes=cubes)])


def heatmap_mosaic(depths: np.ndarray, d_far: float = 0.5) -> bytes:
    """(S, 8, 8) -> JPEG mosaic, 8 columns, turbo, near=red."""
    cell = 96
    cols = 8
    rows = int(np.ceil(len(depths) / cols))
    canvas = np.zeros((rows * cell, cols * cell, 3), np.uint8)
    for i, d8 in enumerate(depths):
        norm = np.clip(d8.astype(np.float32) / d_far, 0, 1)
        rgb = (TURBO(1.0 - norm)[..., :3] * 255).astype(np.uint8)
        big = cv2.resize(rgb, (cell, cell), interpolation=cv2.INTER_NEAREST)
        r, c = divmod(i, cols)
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = big
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def annotate(img_bgr: np.ndarray, md: float, dev: float) -> np.ndarray:
    """On-frame readout so the MP4 stands alone: min skin depth, joint retreat, and a
    red PROXIMITY banner once any sensor reads inside the 0.18 m activation distance."""
    h, w = img_bgr.shape[:2]

    def txt(s, y, col, sc=0.8):
        cv2.putText(img_bgr, s, (20, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)

    mdcm = md * 100 if np.isfinite(md) else 999.0
    txt(f"min skin depth: {mdcm:5.1f} cm", 42, (40, 40, 40))
    txt(f"joint retreat:  {dev:4.2f} rad", 76, (40, 40, 40))
    if md < 0.18:
        cv2.rectangle(img_bgr, (0, 0), (w, 6), (0, 0, 255), -1)
        txt("PROXIMITY  ->  RETREATING", 112, (0, 0, 220))
    return img_bgr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path,
                    default=[Path("/home/jaydv/code/prox_learning/assets/datagen/"
                                  "hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855")],
                    help="datagen run dir(s) to source the posture from")
    ap.add_argument("--ckpt", type=Path,
                    default=Path("/home/jaydv/code/prox_learning/assets/safety/cvae_v2"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/jaydv/code/prox_learning/assets/safety/flinch_demo.mcap"))
    ap.add_argument("--gain", type=float, default=4.5,
                    help="flinch rate (rad/s per unit head output)")
    ap.add_argument("--spring", type=float, default=1.5,
                    help="return-to-nominal rate (1/s)")
    ap.add_argument("--max-dev", type=float, default=0.35,
                    help="per-joint deviation clamp from the nominal posture (rad)")
    ap.add_argument("--bar", default="bar_m", choices=list(sw.BARS))
    args = ap.parse_args()

    head = SafetyHead.load(args.ckpt)
    model = sw.build_model()
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
           for n in ("sash", "jamb_l", "jamb_r", *sw.BARS, f"{NS}base")}
    base_mid = mid[f"{NS}base"]

    rd = mujoco.Renderer(model, 8, 8)
    rd.enable_depth_rendering()
    rd.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[2] = 0

    # 3rd-person RGB renderer -> an MP4 the flinch can be watched as a plain video
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
    for n in sw.BARS:
        data.mocap_pos[mid[n]] = sw.PARK
    for adr, val in zip(arm_qadr, q0):
        data.qpos[adr] = float(val)
    for adr in finger_qadr:
        data.qpos[adr] = grip
    mujoco.mj_forward(model, data)

    # rest baseline: head output with the bar parked (subtracted every frame)
    rest_depths = render_all(model, data, rd, opt, sensors)
    dq_rest = head(rest_depths)
    print(f"rest |head| (physical) = {np.linalg.norm(dq_rest):.3f}  -> baseline-subtracted")

    targets = pick_targets(model, data, rd, opt, sensors, cam_ids, k=2)
    bar_half = np.asarray(sw.BARS[args.bar])
    path = bar_axis_schedule(targets, bar_half)
    T = len(path)
    print(f"{T} frames ({T / FPS:.1f} s), bar '{args.bar}', {len(targets)} approach(es)")

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

    vid_path = args.out.with_suffix(".mp4")
    vw = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (960, 540))

    body_meshes = extract_body_meshes(model)
    pub_bodies = [model.body(i).name for i in range(model.nbody)
                  if model.body(i).name in body_meshes]
    q = q0.copy()
    jacp = np.zeros((3, model.nv))
    dev_peak = 0.0
    md_min_seen = np.inf

    for t in range(T):
        if np.isfinite(path[t]).all():
            data.mocap_pos[mid[args.bar]] = path[t]
        else:
            data.mocap_pos[mid[args.bar]] = sw.PARK
        for adr, val in zip(arm_qadr, q):
            data.qpos[adr] = float(val)
        mujoco.mj_forward(model, data)

        depths = render_all(model, data, rd, opt, sensors)
        pts_all, d_all = [], []
        for si, s in enumerate(sensors):
            cid = cam_ids[s]
            c2w = np.eye(4)
            # cam_xmat is GL; flip to the CV convention foxglove_viz.backproject expects
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
        ch_scene.log(scene_boxes_update(data, mid, args.bar, bar_half, ts), log_time=ns)
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
            # quaternion for yaw-then-pitch (x-axis arrow): q = qz(yaw) * qy(pitch)
            qw = cy * cp
            qx = -sy * sp
            qy_ = cy * sp
            qz = sy * cp
            ln = min(0.12 + 0.25 * vn, 0.45)
            p = data.xpos[hand_bid]
            arrows.append(ArrowPrimitive(
                pose=Pose(position=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                          orientation=Quaternion(x=float(qx), y=float(qy_),
                                                 z=float(qz), w=float(qw))),
                shaft_length=ln, shaft_diameter=0.018,
                head_length=0.05, head_diameter=0.04,
                color=Color(r=1.0, g=0.15, b=0.1, a=0.9)))
        ch_arrow.log(SceneUpdate(entities=[SceneEntity(
            timestamp=ts, frame_id="world", id="push", frame_locked=False,
            arrows=arrows)]), log_time=ns)

        md = float(depths[depths >= 0.005].min()) if (depths >= 0.005).any() else float("inf")
        md_min_seen = min(md_min_seen, md)
        bar_gap = float(np.linalg.norm(data.mocap_pos[mid[args.bar]] - data.xpos[hand_bid]))
        ch_json.log({"dq": [float(x) for x in dq], "dq_norm": float(np.linalg.norm(dq)),
                     "min_depth": md, "bar_gap": bar_gap, "dev_norm": dev}, log_time=ns)

        # annotated 3rd-person RGB frame for the MP4 (arm now at this frame's pose)
        rgb.update_scene(data, camera=vcam, scene_option=vopt)
        vw.write(annotate(cv2.cvtColor(rgb.render(), cv2.COLOR_RGB2BGR), md, dev))

        if (t + 1) % 100 == 0:
            print(f"frame {t + 1}/{T}  min_depth {md:.3f}  |dq| {np.linalg.norm(dq):.2f}  dev {dev:.3f}")

    writer.close()
    vw.release()
    print(f"wrote {args.out}  ({T / FPS:.1f} s)  peak dev {dev_peak:.3f} rad  "
          f"min skin depth seen {md_min_seen:.3f} m")
    print(f"wrote {vid_path}")


if __name__ == "__main__":
    main()
