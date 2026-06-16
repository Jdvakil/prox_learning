"""Reactive collision-avoidance LAYER on a nominal trajectory (deviate-and-rejoin).

The robot executes a planned trajectory (a real recorded reach-grasp-lift, obstacle-
naive). Several hazard bars stand in the workspace along the path. The proximity safety
head (CVAE, skin-only input) produces a joint-space repulsion that is added as a SMOOTH
residual correction on top of the nominal joints. The trajectory clock NEVER stops: the
arm bulges around each obstacle as it sweeps past and rejoins the path immediately after
— one continuous, smooth motion that completes the trajectory while never letting the
skin touch a bar.

    executed_q(t) = q_nom(s) + correction          s advances every frame (deviate-and-rejoin)
    dq_raw        = head(skin_with_bars) - head(skin_bars_parked)   # per-frame baseline -> the bars' push
    dq            = ema * dq + (1-ema) * dq_raw                      # low-pass the head -> smooth
    correction   += (gain * dq - decay * correction) * dt           # grows near a bar, decays when clear
    correction    = clip(correction, +/- max_dev)

Outputs an .mcap (Foxglove) and an annotated .mp4:
    /tf /robot        executed arm
    /nominal          the planned TCP path (green) + marker at the current nominal target
    /scene_gt         hood + the hazard bars (red)
    /proximity        back-projected skin returns
    /sensors/heatmap8 raw 8x8 frames
    /safety_arrow     EE-space push at the wrist
    /safety           json: phase s, status, dq, correction norm, min_depth

Usage:
    OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      /opt/conda/envs/mlspaces/bin/python scripts/safety_react_demo.py \
      --ckpt assets/safety/cvae_v2 --out assets/safety/react_demo.mcap
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import safety_sweep as sw  # noqa: E402  (installs EGL workaround)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402

import foxglove  # noqa: E402
from foxglove import channels as C  # noqa: E402
from foxglove.schemas import (  # noqa: E402
    ArrowPrimitive, Color, CompressedImage, CubePrimitive, FrameTransform,
    FrameTransforms, LinePrimitive, LinePrimitiveLineType, Point3, Pose, Quaternion,
    SceneEntity, SceneUpdate, SpherePrimitive, Vector3,
)
from foxglove_viz import (  # noqa: E402
    backproject, extract_body_meshes, pack_cloud, robot_mesh_scene_update,
)
from safety_flinch_demo import (  # noqa: E402
    FPS, DT, NS, FOREARM_PREFIXES, annotate, apply_aperture, clear_reaching_posture,
    heatmap_mosaic, render_all, ts_at,
)
from train_safety_cvae import SafetyHead  # noqa: E402


def pick_episode(runs):
    """Successful episode with the LARGEST end-effector travel (most sweep, so distinct
    phases put the arm in distinct places) -> exact arm qpos & grip & base."""
    import h5py
    files = sorted(p for d in runs for p in glob.glob(f"{d}/house_*/trajectories_batch_*.h5"))
    best, best_span, best_id = None, -1.0, ""
    for f in files:
        with h5py.File(f, "r") as h:
            for k in sorted(kk for kk in h if kk.startswith("traj")):
                t = h[k]
                if not bool(t["success"][:][-1]):
                    continue
                panda = t["env_states/articulations/panda"][:]
                tcp = t["obs/extra/tcp_pose"][:][:, :3]
                span = float(np.linalg.norm(tcp.max(0) - tcp.min(0)))   # bounding-box diagonal of TCP path
                if 50 <= panda.shape[0] <= 140 and span > best_span:
                    best_span, best_id = span, f"{Path(f).parent.name}/{k}"
                    best = (panda[:, 0:7].astype(np.float64), panda[:, 7].astype(np.float64),
                            t["obs/extra/robot_base_pose"][:][0].astype(np.float64))
    if best is None:
        raise SystemExit("no successful episode found")
    print(f"nominal trajectory: {best_id}  (TCP travel span {best_span:.2f} m)")
    return best


def extended_posture(runs, model, data, arm_qadr, finger_qadr, base_mid, mid, ap_w, ap_h, n_cand=400):
    """Forward posture with the LARGEST wrist reach radius from the base axis, so a base-
    joint sweep moves the wrist over a wide arc (well-separated obstacle encounters)."""
    q_all, grip_all, base_all, _ = sw.load_postures([Path(p) for p in runs])
    hand_bid = model.body(f"{NS}fr3_link7").id
    wrist_bid = model.body(f"{NS}fr3_link6").id
    base_bid = model.body(f"{NS}base").id
    rng = np.random.default_rng(0)
    rows = rng.choice(len(q_all), size=min(n_cand, len(q_all)), replace=False)
    for n in sw.BARS:
        data.mocap_pos[mid[n]] = sw.PARK
    apply_aperture(data, mid, ap_w, ap_h)
    best, best_r = None, -1.0
    for r in rows:
        data.mocap_pos[base_mid] = base_all[r][:3]
        data.mocap_quat[base_mid] = base_all[r][3:7]
        for adr, v in zip(arm_qadr, q_all[r]):
            data.qpos[adr] = float(v)
        for adr in finger_qadr:
            data.qpos[adr] = float(grip_all[r])
        mujoco.mj_forward(model, data)
        if data.xpos[hand_bid][0] < 0.40:
            continue
        rad = float(np.linalg.norm(data.xpos[wrist_bid][:2] - data.xpos[base_bid][:2]))
        if rad > best_r:
            best_r, best = rad, r
    print(f"posture: row {best}, wrist reach radius {best_r:.2f} m (extended for a wide sweep)")
    return q_all[best].copy(), float(grip_all[best]), base_all[best].copy()


def resample(arr, n):
    T = arr.shape[0]
    src, dst = np.linspace(0, 1, T), np.linspace(0, 1, n)
    if arr.ndim == 1:
        return np.interp(dst, src, arr)
    return np.stack([np.interp(dst, src, arr[:, j]) for j in range(arr.shape[1])], 1)


def smooth(arr, w=9):
    """Centered moving-average along axis 0 (edge-replicated) -> a C1-ish nominal."""
    if arr.shape[0] < w:
        return arr.copy()
    pad = w // 2
    ker = np.ones(w) / w
    ap = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(ap[:, j], ker, mode="valid") for j in range(arr.shape[1])], 1)


def bars_scene_update(data, mid, placed, ts) -> SceneUpdate:
    """Hood (gray) + every placed hazard bar (red)."""
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
    for name, half, wp in placed:
        cube(wp, half, Color(r=1.0, g=0.30, b=0.05, a=0.95))
    return SceneUpdate(entities=[SceneEntity(timestamp=ts, frame_id="world", id="scene",
                                             frame_locked=False, cubes=cubes)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path,
                    default=[Path("/home/jaydv/code/prox_learning/assets/datagen/"
                                  "hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855")])
    ap.add_argument("--ckpt", type=Path,
                    default=Path("/home/jaydv/code/prox_learning/assets/safety/cvae_v3"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/jaydv/code/prox_learning/experiments_output/default/demos/react.mcap"))
    ap.add_argument("--mode", choices=["sweep", "pick"], default="sweep",
                    help="sweep = smooth wide lateral sweep (clean multi-obstacle weave); "
                         "pick = replay a recorded pick trajectory")
    ap.add_argument("--sweep-amp", type=float, default=0.55, help="base-joint sweep amplitude (rad)")
    ap.add_argument("--gain", type=float, default=4.0, help="repulsion gain (rad/s per unit head)")
    ap.add_argument("--decay", type=float, default=2.2, help="correction decay rate (1/s) when clear")
    ap.add_argument("--ema", type=float, default=0.75, help="head-output low-pass (0=none, ->1 smoother)")
    ap.add_argument("--max-dev", type=float, default=0.35, help="per-joint correction clamp (rad)")
    ap.add_argument("--traj-secs", type=float, default=10.0, help="nominal trajectory duration")
    ap.add_argument("--standoff", type=float, default=0.10, help="bar FACE distance from the skin (m)")
    ap.add_argument("--obstacle", choices=["bar", "sphere"], default="bar")
    ap.add_argument("--clean", action="store_true",
                    help="minimal overlay: only min skin depth, obstacle distance, deviation")
    ap.add_argument("--radius", type=float, default=0.045, help="sphere radius when --obstacle sphere")
    args = ap.parse_args()

    head = SafetyHead.load(args.ckpt)
    model = sw.build_model()
    # tag hood/walls/sash/jambs to render group 3 (this model instance only) so the RGB
    # video can hide them while the 8x8 depth render keeps them (group 3 enabled there) —
    # the head still sees the full scene exactly as in training.
    hood_names = set(sw.STATIC_BOXES) | {"sash", "jamb_l", "jamb_r"}
    for gid in range(model.ngeom):
        if model.body(model.geom_bodyid[gid]).name in hood_names:
            model.geom_group[gid] = 3
    if args.obstacle == "sphere":
        sw.bars_to_spheres(model, args.radius)
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
    hand_bid = model.body(f"{NS}fr3_link7").id
    link_bids = [model.body(f"{NS}fr3_link{i}").id for i in range(2, 8)]
    bar_mocaps = list(sw.BARS)   # bar_s, bar_m, bar_l (3 available)
    ap_w, ap_h = 0.675, 0.535

    rd = mujoco.Renderer(model, 8, 8)
    rd.enable_depth_rendering()
    rd.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[2] = 0          # depth/sensor render: hide cosmetic skin...
    opt.geomgroup[3] = 1          # ...but KEEP the hood (head sees the full scene)

    Tnom = int(args.traj_secs * FPS)
    ds_nom = 1.0 / (Tnom - 1)

    def pose(qarm, qgrip):
        for adr, v in zip(arm_qadr, qarm):
            data.qpos[adr] = float(v)
        for adr in finger_qadr:
            data.qpos[adr] = float(qgrip)

    if args.mode == "sweep":
        # smooth wide LATERAL sweep about the base joint from a clear forward-reaching
        # posture -> the arm traverses open space, so the obstacles are met one at a time.
        q0, grip0, base = extended_posture(
            args.runs, model, data, arm_qadr, finger_qadr, base_mid, mid, ap_w, ap_h)
        # trapezoidal velocity: cosine ease in/out + CONSTANT velocity in the middle, so
        # bars placed in the core are passed at uniform speed -> brief, evenly-spaced dodges
        ramp = max(2, int(0.16 * Tnom))
        vel = np.ones(Tnom)
        vel[:ramp] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp)))
        vel[-ramp:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, ramp)))
        prof = np.cumsum(vel)
        prof /= prof[-1]                                   # 0 -> 1, uniform through the middle
        q_nom = np.tile(q0, (Tnom, 1))
        q_nom[:, 0] = q0[0] + args.sweep_amp * (2 * prof - 1)
        grip_nom = np.full(Tnom, grip0)
    else:
        q_raw, grip_raw, base = pick_episode(args.runs)
        q_nom = smooth(resample(q_raw, Tnom), 9)
        grip_nom = resample(grip_raw, Tnom)

    data.mocap_pos[base_mid] = base[:3]
    data.mocap_quat[base_mid] = base[3:7]
    apply_aperture(data, mid, ap_w, ap_h)
    for n in sw.BARS:
        data.mocap_pos[mid[n]] = sw.PARK

    def qnom_at(s):
        f = np.clip(s, 0, 1) * (Tnom - 1)
        i = int(np.floor(f))
        j = min(i + 1, Tnom - 1)
        a = f - i
        return q_nom[i] * (1 - a) + q_nom[j] * a, float(grip_nom[i] * (1 - a) + grip_nom[j] * a)

    # nominal TCP path (green ghost line)
    nom_tcp = []
    for s in np.linspace(0, 1, Tnom):
        qa, qg = qnom_at(s)
        pose(qa, qg)
        mujoco.mj_forward(model, data)
        nom_tcp.append(data.xpos[hand_bid].copy())
    nom_tcp = np.array(nom_tcp)

    if args.mode == "sweep":
        # place bars in the constant-velocity core so each is passed quickly -> clean gaps
        chosen_phases = list(np.linspace(0.30, 0.70, len(bar_mocaps)))
    else:
        # farthest-point sampling of the wrist path -> separated encounters on a pick
        wrist_bid = model.body(f"{NS}fr3_link6").id
        probe = np.linspace(0.18, 0.86, 10)
        hpos = []
        for p in probe:
            qa, qg = qnom_at(p)
            pose(qa, qg)
            mujoco.mj_forward(model, data)
            hpos.append(data.xpos[wrist_bid].copy())
        hpos = np.array(hpos)
        picks = [3]
        while len(picks) < len(bar_mocaps):
            far = np.min([np.linalg.norm(hpos - hpos[j], axis=1) for j in picks], axis=0)
            picks.append(int(np.argmax(far)))
        chosen_phases = sorted(float(probe[j]) for j in picks)
    print(f"obstacle phases: {[round(p, 2) for p in chosen_phases]}")

    # stand a hazard bar at each chosen phase: in front of the most-exposed wrist/forearm
    # sensor at that nominal pose, FACE `standoff` m off the skin (so the arm must bulge).
    placed = []   # (mocap_name, half(3,), world_pos(3,))
    for k, p in enumerate(chosen_phases):
        qa, qg = qnom_at(p)
        pose(qa, qg)
        mujoco.mj_forward(model, data)
        dep = render_all(model, data, rd, opt, sensors)
        rest_min = np.array([float(dep[i][dep[i] >= 0.005].min()) if (dep[i] >= 0.005).any() else np.inf
                             for i in range(len(sensors))])
        # anchor to the WRIST (link6) — it travels with the hand, so a bar on the clearest
        # wrist sensor's axis lands at a distinct place each (farthest-point) phase. Fall
        # back to the forearm only if no wrist sensor is clear.
        pref = [i for i in range(len(sensors)) if sensors[i].startswith("link6")]
        pref = [i for i in pref if np.isfinite(rest_min[i]) and rest_min[i] > 0.4] or \
               [i for i in range(len(sensors)) if sensors[i].startswith(("link6", "link5"))]
        i = max(pref, key=lambda j: rest_min[j])
        cid = cam_ids[sensors[i]]
        fwd = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
        name = bar_mocaps[k]
        half = np.asarray(sw.BARS[name])
        wp = data.cam_xpos[cid].copy() + fwd * (args.standoff + half[1])
        placed.append((name, half, wp))
        print(f"bar {name} at phase {p:.2f}: sensor {sensors[i]} (rest {rest_min[i]:.2f} m) "
              f"-> world ({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f})")
    for name, _half, wp in placed:
        data.mocap_pos[mid[name]] = wp

    # 3rd-person RGB
    rgb = mujoco.Renderer(model, 540, 960)
    vcam = mujoco.MjvCamera()
    vcam.type = mujoco.mjtCamera.mjCAMERA_FREE
    vcam.lookat[:] = [0.66, 0.20, 1.05]
    vcam.distance, vcam.azimuth, vcam.elevation = 2.0, 110.0, -12.0
    vopt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(vopt)
    vopt.geomgroup[3] = 0          # RGB video: hide the hood/walls for a clean shot

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ctx = foxglove.Context()
    writer = foxglove.open_mcap(str(args.out), allow_overwrite=True, context=ctx)
    ch_tf = C.FrameTransformsChannel("/tf", context=ctx)
    ch_mesh = C.SceneUpdateChannel("/robot", context=ctx)
    ch_scene = C.SceneUpdateChannel("/scene_gt", context=ctx)
    ch_nom = C.SceneUpdateChannel("/nominal", context=ctx)
    ch_pc = C.PointCloudChannel("/proximity", context=ctx)
    ch_img = C.CompressedImageChannel("/sensors/heatmap8", context=ctx)
    ch_arrow = C.SceneUpdateChannel("/safety_arrow", context=ctx)
    ch_json = foxglove.Channel("/safety", message_encoding="json", context=ctx)
    vw = cv2.VideoWriter(str(args.out.with_suffix(".mp4")),
                         cv2.VideoWriter_fourcc(*"mp4v"), FPS, (960, 540))

    body_meshes = extract_body_meshes(model)
    pub_bodies = [model.body(i).name for i in range(model.nbody) if model.body(i).name in body_meshes]
    nom_line = LinePrimitive(
        type=LinePrimitiveLineType.LineStrip, thickness=2.0, scale_invariant=True,
        pose=Pose(position=Vector3(x=0.0, y=0.0, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
        points=[Point3(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in nom_tcp],
        color=Color(r=0.1, g=0.85, b=0.2, a=0.8))

    s = 0.0
    correction = np.zeros(7)
    dq = np.zeros(7)
    jacp = np.zeros((3, model.nv))
    maxframes = Tnom + int(2.0 * FPS)
    min_depth_per_bar = [np.inf] * len(placed)

    for t in range(maxframes):
        qa, qg = qnom_at(s)
        q = qa + correction
        pose(q, qg)
        for name, _half, wp in placed:
            data.mocap_pos[mid[name]] = wp
        mujoco.mj_forward(model, data)

        depths = render_all(model, data, rd, opt, sensors)
        # per-frame baseline: same pose, ALL bars parked -> isolate the bars' marginal push
        for name in bar_mocaps:
            data.mocap_pos[mid[name]] = sw.PARK
        mujoco.mj_forward(model, data)
        depths_rest = render_all(model, data, rd, opt, sensors)
        for name, _half, wp in placed:
            data.mocap_pos[mid[name]] = wp
        mujoco.mj_forward(model, data)

        dq_raw = (head(depths) - head(depths_rest)) / max(head.scale, 1e-6)
        dq = args.ema * dq + (1 - args.ema) * dq_raw
        correction = correction + (args.gain * dq - args.decay * correction) * DT
        correction = np.clip(correction, -args.max_dev, args.max_dev)
        threat = float(np.linalg.norm(dq))
        dev = float(np.linalg.norm(correction))

        s = min(1.0, s + ds_nom)                      # deviate-and-rejoin: clock never stops
        status = "AVOIDING" if dev > 0.05 else ("DONE" if s >= 1.0 else "FOLLOWING")

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
        ch_scene.log(bars_scene_update(data, mid, placed, ts), log_time=ns)

        nom_now, _ = qnom_at(s)
        pose(nom_now, qg)
        mujoco.mj_forward(model, data)
        nm = data.xpos[hand_bid].copy()
        pose(q, qg)
        mujoco.mj_forward(model, data)
        ch_nom.log(SceneUpdate(entities=[SceneEntity(
            timestamp=ts, frame_id="world", id="nominal", frame_locked=False, lines=[nom_line],
            spheres=[SpherePrimitive(
                pose=Pose(position=Vector3(x=float(nm[0]), y=float(nm[1]), z=float(nm[2])),
                          orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
                size=Vector3(x=0.04, y=0.04, z=0.04), color=Color(r=0.1, g=0.85, b=0.2, a=0.9))])]),
            log_time=ns)

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
        for bi, (_n, _h, wp) in enumerate(placed):
            min_depth_per_bar[bi] = min(min_depth_per_bar[bi],
                                        float(np.linalg.norm(wp - data.xpos[hand_bid])))
        ch_json.log({"phase": float(s), "status": status, "dq_norm": threat,
                     "min_depth": md, "dev_norm": dev, "n_obstacles": len(placed),
                     "dq": [float(x) for x in dq]}, log_time=ns)

        rgb.update_scene(data, camera=vcam, scene_option=vopt)
        frame = cv2.cvtColor(rgb.render(), cv2.COLOR_RGB2BGR)
        avoiding = status == "AVOIDING"
        if args.clean:
            gap = sw.min_obstacle_gap(model, data, mid, [n for n, _, _ in placed], link_bids)
            sw.clean_hud(frame, md, gap, dev)
        else:
            def txt(strg, y, col, sc=0.8):
                cv2.putText(frame, strg, (20, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)
            txt(f"trajectory: {s*100:3.0f}%   obstacles: {len(placed)}", 42, (40, 40, 40))
            txt(f"min skin depth: {md*100:4.1f} cm", 76, (40, 40, 40))
            txt(f"joint deviation: {dev:4.2f} rad", 110, (40, 40, 40))
            if avoiding:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 6), (0, 0, 255), -1)
                txt("AVOIDING OBSTACLE", 146, (0, 0, 220))
            else:
                txt("FOLLOWING TRAJECTORY" if s < 1.0 else "TRAJECTORY COMPLETE", 146, (150, 90, 0))
        vw.write(frame)

        if (t + 1) % 100 == 0:
            print(f"frame {t+1}  s={s:.2f}  {status:9s}  min_depth {md:.3f}  |dq| {threat:.2f}  dev {dev:.3f}")
        if s >= 1.0 and dev < 0.03:
            break

    writer.close()
    vw.release()
    print(f"wrote {args.out} and {args.out.with_suffix('.mp4')}  ({t+1} frames, {(t+1)/FPS:.1f} s)")
    print(f"closest hand-to-bar distances: {[round(x,3) for x in min_depth_per_bar]} m")


if __name__ == "__main__":
    main()
