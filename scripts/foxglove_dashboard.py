"""Engineer-grade Foxglove dashboard export for franka_skin proximity episodes.

Builds on foxglove_export.py (3D robot + point cloud + cameras) and adds everything a
robotics engineer wants on one screen (pair with foxglove_dashboard_layout.json):

3D panel
  /tf, /robot          kinematic tree + robot mesh (FK-replayed from the saved joints)
  /proximity           29-SPAD 8x8 returns back-projected to a world cloud (turbo: red=near)
  /sensor_fov          every SPAD's 45-deg FOV cone, frame-locked to its link (per-link color)
  /scene_gt            ground-truth obstacle AABBs from scene_params (protrusion in red),
                       embed-transformed for in-house episodes; target start/goal spheres
  /tcp                 end-effector pose axis

Image panels
  /camera/exo, /camera/wrist     recorded RGB feeds
  /skin/link2|link3|link5|link6  per-sensor 8x8 depth tiles (turbo), labeled with sensor id
                                 + live min distance — the "what does each sensor see" feed

Plot panels (flat JSON topics -> Foxglove Plot message paths)
  /joints   q1..q7 (measured), c1..c7 (commanded), e1..e7 (tracking error, deg),
            v1..v7 (velocity), err_cm (TCP tracking error norm), grip_mm
  /skin     link2/link3/link5/link6 per-link min distance (m), min (global), zones_lt8 count
  /motion   tcp_speed (cm/s), phase id — watch speed drop exactly when /skin.min drops
  /task     Log stream: task description, phase transitions, success/fail at the end

Usage:
  python scripts/foxglove_dashboard.py --h5 PATH.h5 --traj 3 --out ep3_dashboard.mcap
  python scripts/foxglove_dashboard.py --h5 PATH.h5 --traj all --out-dir mcaps/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import foxglove_export as fx  # noqa: E402  (reuse model/mesh/cloud machinery)

import foxglove  # noqa: E402
from foxglove import channels as C  # noqa: E402
from foxglove.schemas import (  # noqa: E402
    Color,
    CompressedImage,
    CubePrimitive,
    FrameTransform,
    FrameTransforms,
    Log,
    LogLevel,
    Point3,
    Pose,
    PoseInFrame,
    Quaternion,
    SceneEntity,
    SceneUpdate,
    SpherePrimitive,
    Timestamp,
    TriangleListPrimitive,
    Vector3,
)

NS = "robot_0/"
# links covered by the skin — superset of the 29-sensor (2/3/5/6) and 40-sensor hybrid
# (1/2/3/4/5_front/5_back/6) layouts; per-sensor code derives the link from the name.
LINKS = ("link1", "link2", "link3", "link4", "link5", "link6")
LINK_COLOR = {"link1": (255, 209, 102), "link2": (230, 57, 70), "link3": (42, 157, 143),
              "link4": (155, 93, 229), "link5": (98, 110, 250), "link6": (244, 162, 97)}


def _link_of(sensor_name: str) -> str:
    """'link5_front_sensor_2' -> 'link5'; 'link2_sensor_0' -> 'link2'."""
    return sensor_name.split("_sensor_")[0].split("_")[0]
FOV_DEG = 45.0          # SPAD full FOV (matches the sim cameras)
CONE_LEN = 0.30         # rendered cone length (m)


# --------------------------------------------------------------------------- #
# h5 loading (current franka_skin format)
# --------------------------------------------------------------------------- #
def _dec(ds, i) -> dict:
    b = ds[i].tobytes()
    n = b.find(b"\x00")
    return json.loads(b[: n if n != -1 else None].decode("utf-8", "ignore"))


def load_episode(h5_path: str, idx: int) -> dict:
    with h5py.File(h5_path, "r") as f:
        t = f[f"traj_{idx}"]
        T = t["env_states/articulations/panda"].shape[0]
        prox = {k: t[f"obs/proximity/{k}"][:] for k in t["obs/proximity"]}
        tcp = t["obs/extra/tcp_pose"][:].astype(np.float64)
        base = t["obs/extra/robot_base_pose"][:].astype(np.float64)
        phase = t["obs/extra/policy_phase"][:]
        success = bool(t["success"][-1])
        q_rows, grip_rows, dq_rows, cmd_rows = [], [], [], []
        for i in range(T):
            qp = _dec(t["obs/agent/qpos"], i)
            q_rows.append(qp["arm"])
            grip_rows.append((qp.get("gripper") or [0.0])[0])
            qv = _dec(t["obs/agent/qvel"], i)
            arm_v = qv.get("arm")
            dq_rows.append(arm_v if isinstance(arm_v, list) and len(arm_v) == 7 else [0.0] * 7)
            ac = _dec(t["actions/joint_pos"], i)
            arm_c = ac.get("arm")
            if not (isinstance(arm_c, list) and len(arm_c) == 7):
                # gripper-only action step: command holds (carry the last arm command)
                arm_c = cmd_rows[-1] if cmd_rows else qp["arm"]
            cmd_rows.append(arm_c)
        q = np.array(q_rows)
        grip = np.array(grip_rows)
        dq = np.array(dq_rows)
        cmd = np.array(cmd_rows)
        raw = t["obs_scene"]
        raw = raw[()] if raw.shape == () else raw[0]
        s = (raw.tobytes() if isinstance(raw, np.ndarray) else raw).decode("utf-8", "ignore").rstrip("\x00")
        try:
            scene = json.loads(s)
        except Exception:
            scene = {}
    return dict(T=T, prox=prox, tcp=tcp, base=base, phase=np.asarray(phase), success=success,
                q=q, dq=dq, cmd=cmd, grip=grip, scene=scene,
                dt_s=float(scene.get("policy_dt_ms", 66.0)) / 1000.0)


# --------------------------------------------------------------------------- #
# Per-sensor depth mosaics
# --------------------------------------------------------------------------- #
TILE = 96
HDR = 16


def link_mosaic(prox: dict, link: str, t: int, near: float, far: float) -> np.ndarray:
    names = sorted(k for k in prox if k.startswith(link))
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    canvas = np.full((rows * (TILE + HDR), cols * TILE, 3), 24, np.uint8)
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        d8 = prox[name][t].mean(axis=0)            # (8,8) mean over substeps
        dmin = float(d8.min())
        norm = np.clip((d8 - near) / (far - near), 0, 1)
        img = (fx._TURBO(1.0 - norm)[:, :, :3] * 255).astype(np.uint8)   # near -> red
        img[d8 > 2.0] = (12, 12, 12)               # no-return sentinel -> black
        img = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
        y0 = r * (TILE + HDR)
        x0 = c * TILE
        canvas[y0 + HDR : y0 + HDR + TILE, x0 : x0 + TILE] = img[..., ::-1]  # RGB->BGR
        lab = f"S{name.split('_')[-1]} " + (f"{dmin:.2f}m" if dmin < 2.0 else "--")
        col = (60, 60, 255) if dmin < 0.08 else (220, 220, 220)
        cv2.putText(canvas, lab, (x0 + 4, y0 + HDR - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #
# Static 3D extras
# --------------------------------------------------------------------------- #
def fov_cones(model: mujoco.MjModel, sensor_cams: dict[str, int], ts: Timestamp) -> SceneUpdate:
    """One translucent cone per SPAD, in its parent BODY frame (frame_locked follows /tf)."""
    half = np.deg2rad(FOV_DEG / 2)
    rad = float(np.tan(half) * CONE_LEN)
    segs = 12
    ang = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    entities = []
    for h5name, cid in sensor_cams.items():
        bid = int(model.cam_bodyid[cid])
        bname = model.body(bid).name
        p0 = model.cam_pos[cid].astype(np.float64)          # body-local
        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, model.cam_quat[cid].astype(np.float64))
        Rm = Rm.reshape(3, 3)
        axis = -Rm[:, 2]                                    # camera looks along -z
        u, v = Rm[:, 0], Rm[:, 1]
        rim = [p0 + axis * CONE_LEN + rad * (np.cos(a) * u + np.sin(a) * v) for a in ang]
        pts = [Point3(x=float(p0[0]), y=float(p0[1]), z=float(p0[2]))] + \
              [Point3(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in rim]
        idx = []
        for i in range(segs):
            idx += [0, 1 + i, 1 + (i + 1) % segs]
        link = next(l for l in LINKS if l in h5name)
        r, g, b = LINK_COLOR[link]
        tri = TriangleListPrimitive(points=pts, indices=idx,
                                    color=Color(r=r / 255, g=g / 255, b=b / 255, a=0.14))
        entities.append(SceneEntity(timestamp=ts, frame_id=bname, id=f"fov_{h5name}",
                                    frame_locked=True, triangles=[tri]))
    return SceneUpdate(entities=entities)


def _embed_T(scene: dict) -> np.ndarray:
    e = (scene.get("scene_params") or {}).get("embed")
    T = np.eye(4)
    if e:
        bx, by, yaw = e
        c, s = np.cos(yaw), np.sin(yaw)
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        T[:3, 3] = [bx, by, 0]
    return T


def scene_gt(scene: dict, ts: Timestamp) -> SceneUpdate | None:
    sp = scene.get("scene_params") or {}
    boxes = sp.get("obstacle_aabbs")
    if not boxes:
        return None
    T = _embed_T(scene)
    yaw = float((sp.get("embed") or [0, 0, 0])[2])
    qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])  # w,x,y,z
    cubes = []
    n = len(boxes)
    protr = bool(sp.get("protrusion_present")) and "protr_center" in sp
    for i, (c, h) in enumerate(boxes):
        w = (T @ np.array([c[0], c[1], c[2], 1.0]))[:3]
        is_protr = protr and i == n - 1
        col = Color(r=0.90, g=0.22, b=0.21, a=0.55) if is_protr else \
              Color(r=0.75, g=0.78, b=0.82, a=0.22)
        cubes.append(CubePrimitive(
            pose=Pose(position=Vector3(x=float(w[0]), y=float(w[1]), z=float(w[2])),
                      orientation=Quaternion(x=float(qz[1]), y=float(qz[2]),
                                             z=float(qz[3]), w=float(qz[0]))),
            size=Vector3(x=2 * float(h[0]), y=2 * float(h[1]), z=2 * float(h[2])),
            color=col))
    return SceneUpdate(entities=[SceneEntity(timestamp=ts, frame_id="world", id="scene_gt",
                                             frame_locked=False, cubes=cubes)])


def target_markers(h5_path: str, idx: int, ts: Timestamp) -> SceneUpdate:
    with h5py.File(h5_path, "r") as f:
        t = f[f"traj_{idx}"]
        start = t["obs/extra/obj_start"][0]
        end = t["obs/extra/obj_end"][0]
    sph = []
    for p, col in ((start, Color(r=0.18, g=0.80, b=0.44, a=0.9)),
                   (end, Color(r=0.18, g=0.55, b=0.95, a=0.5))):
        sph.append(SpherePrimitive(
            pose=Pose(position=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                      orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
            size=Vector3(x=0.05, y=0.05, z=0.05), color=col))
    return SceneUpdate(entities=[SceneEntity(timestamp=ts, frame_id="world", id="targets",
                                             frame_locked=False, spheres=sph)])


# --------------------------------------------------------------------------- #
# Export one episode
# --------------------------------------------------------------------------- #
def export(h5_path: str, idx: int, out_path: str, *, model, mesh_update, sensor_cams,
           near=0.02, far=0.60, d_max=1.5, stride=1):
    ep = load_episode(h5_path, idx)
    vids = fx.load_camera_videos(h5_path, idx)
    data = mujoco.MjData(model)
    arm_qadr = [model.joint(j).qposadr[0] for j in fx.SENSOR_LINK_JOINTS]
    finger_qadr = [model.joint(f"{NS}gripper/{n}").qposadr[0]
                   for n in ("left_driver_joint", "right_driver_joint")
                   if f"{NS}gripper/{n}" in [model.joint(i).name for i in range(model.njnt)]]
    pub_bodies = sorted({model.body(i).name for i in range(model.nbody)
                         if model.body(i).name and model.body(i).name != "world"})
    # base placement from the episode itself (works for standalone AND in-house episodes)
    bp = ep["base"][0]
    base_bid = model.body(f"{NS}base").id
    base_mid = int(model.body_mocapid[base_bid])

    ctx = foxglove.Context()
    writer = foxglove.open_mcap(out_path, allow_overwrite=True, context=ctx)
    tf_ch = C.FrameTransformsChannel("/tf", context=ctx)
    pc_ch = C.PointCloudChannel("/proximity", context=ctx)
    tcp_ch = C.PoseInFrameChannel("/tcp", context=ctx)
    log_ch = C.LogChannel("/task", context=ctx)
    mesh_ch = C.SceneUpdateChannel("/robot", context=ctx)
    fov_ch = C.SceneUpdateChannel("/sensor_fov", context=ctx)
    gt_ch = C.SceneUpdateChannel("/scene_gt", context=ctx)
    tgt_ch = C.SceneUpdateChannel("/targets", context=ctx)
    img_ch = {n: C.CompressedImageChannel(f"/camera/{n}", context=ctx) for n in vids}
    skin_img_ch = {l: C.CompressedImageChannel(f"/skin/{l}", context=ctx) for l in LINKS}
    joints_ch = foxglove.Channel("/joints", message_encoding="json", context=ctx)
    skin_ch = foxglove.Channel("/skin", message_encoding="json", context=ctx)
    motion_ch = foxglove.Channel("/motion", message_encoding="json", context=ctx)

    dt = ep["dt_s"]

    def ts_at(t):
        ns = int(round(t * dt * 1e9))
        return Timestamp(sec=ns // 1_000_000_000, nsec=ns % 1_000_000_000), ns

    ts0, _ = ts_at(0)
    sp = ep["scene"].get("scene_params") or {}
    desc = (f"cell={sp.get('cell','?')} clearance={100*sp.get('clearance',0):.1f}cm "
            f"depth={sp.get('depth',0):.2f}m light={sp.get('light_scale',0):.2f} "
            f"behavior={ep['scene'].get('behavior_class','?')}")
    log_ch.log(Log(timestamp=ts0, level=LogLevel.Info,
                   message=f"EPISODE {idx}: {desc}", name="task"), log_time=0)
    gt = scene_gt(ep["scene"], ts0)
    if gt:
        gt_ch.log(gt, log_time=0)
    tgt_ch.log(target_markers(h5_path, idx, ts0), log_time=0)

    inv_phase = {}
    pm = ep["scene"].get("policy_phases") or {}
    inv_phase = {v: k for k, v in pm.items()}
    last_phase = None
    tcp_prev = None

    for t in range(0, ep["T"], stride):
        ts, log_ns = ts_at(t)
        # FK replay: base from the episode, arm joints from qpos
        data.mocap_pos[base_mid] = bp[:3]
        data.mocap_quat[base_mid] = bp[3:7]
        for adr, val in zip(arm_qadr, ep["q"][t]):
            data.qpos[adr] = float(val)
        for adr in finger_qadr:
            data.qpos[adr] = float(ep["grip"][t])
        mujoco.mj_forward(model, data)

        tfs = []
        for bname in pub_bodies:
            bid = model.body(bname).id
            p, qt = data.xpos[bid], data.xquat[bid]
            tfs.append(FrameTransform(
                timestamp=ts, parent_frame_id="world", child_frame_id=bname,
                translation=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                rotation=Quaternion(x=float(qt[1]), y=float(qt[2]), z=float(qt[3]), w=float(qt[0]))))
        tf_ch.log(FrameTransforms(transforms=tfs), log_time=log_ns)
        if t == 0:
            mesh_ch.log(mesh_update, log_time=log_ns)
            fov_ch.log(fov_cones(model, sensor_cams, ts), log_time=log_ns)

        # proximity cloud + per-link minima
        all_pts, all_d = [], []
        link_min = {l: 9.99 for l in LINKS}
        zones_lt8 = 0
        for h5name, cid in sensor_cams.items():
            d8 = ep["prox"][h5name][t].mean(axis=0)
            dmin = float(d8.min())
            link = next(l for l in LINKS if l in h5name)
            link_min[link] = min(link_min[link], dmin)
            if dmin < 0.08:
                zones_lt8 += 1
            c2w = np.eye(4)
            c2w[:3, :3] = data.cam_xmat[cid].reshape(3, 3)
            c2w[:3, 3] = data.cam_xpos[cid]
            pts, d = fx.backproject_sensor(d8, c2w, FOV_DEG, near, d_max)
            if len(pts):
                all_pts.append(pts)
                all_d.append(d)
        if all_pts:
            pts = np.concatenate(all_pts)
            d = np.concatenate(all_d)
            pc_ch.log(fx.pack_cloud(pts, fx.turbo_rgba(d, near, far), ts), log_time=log_ns)
        else:
            pc_ch.log(fx.pack_cloud(np.zeros((0, 3)), np.zeros((0, 4), np.uint8), ts), log_time=log_ns)

        # skin mosaics
        for l in LINKS:
            ok, buf = cv2.imencode(".jpg", link_mosaic(ep["prox"], l, t, near, far),
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                skin_img_ch[l].log(CompressedImage(timestamp=ts, frame_id=f"skin_{l}",
                                                   data=buf.tobytes(), format="jpeg"), log_time=log_ns)

        # cameras
        for n, arr in vids.items():
            if t < len(arr):
                ok, buf = cv2.imencode(".jpg", arr[t], [cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok:
                    img_ch[n].log(CompressedImage(timestamp=ts, frame_id=f"camera_{n}",
                                                  data=buf.tobytes(), format="jpeg"), log_time=log_ns)

        # tcp + speed
        tcp = ep["tcp"][t]
        tcp_ch.log(PoseInFrame(timestamp=ts, frame_id="world", pose=Pose(
            position=Vector3(x=float(tcp[0]), y=float(tcp[1]), z=float(tcp[2])),
            orientation=Quaternion(x=float(tcp[4]), y=float(tcp[5]), z=float(tcp[6]), w=float(tcp[3])))),
            log_time=log_ns)
        speed = 0.0 if tcp_prev is None else float(np.linalg.norm(tcp[:3] - tcp_prev) / dt)
        tcp_prev = tcp[:3].copy()

        # flat JSON plot topics
        q, cmd, dq = ep["q"][t], ep["cmd"][t], ep["dq"][t]
        err = np.degrees(cmd - q)
        jmsg = {f"q{i+1}": float(q[i]) for i in range(7)}
        jmsg |= {f"c{i+1}": float(cmd[i]) for i in range(7)}
        jmsg |= {f"e{i+1}": float(err[i]) for i in range(7)}
        jmsg |= {f"v{i+1}": float(dq[i]) for i in range(min(7, len(dq)))}
        jmsg["grip_mm"] = float(ep["grip"][t] * 1000)
        joints_ch.log(jmsg, log_time=log_ns)
        skin_ch.log({**{l: round(link_min[l], 4) for l in LINKS},
                     "min": round(min(link_min.values()), 4),
                     "zones_lt8": zones_lt8}, log_time=log_ns)
        motion_ch.log({"tcp_speed_cms": round(speed * 100, 2),
                       "phase": int(ep["phase"][t])}, log_time=log_ns)

        ph = int(ep["phase"][t])
        if ph != last_phase:
            log_ch.log(Log(timestamp=ts, level=LogLevel.Info,
                           message=f"phase -> {inv_phase.get(ph, ph)}", name="phase"), log_time=log_ns)
            last_phase = ph

    ts_end, ns_end = ts_at(ep["T"] - 1)
    log_ch.log(Log(timestamp=ts_end, level=LogLevel.Info if ep["success"] else LogLevel.Warning,
                   message=f"EPISODE END: {'SUCCESS' if ep['success'] else 'FAIL'}",
                   name="task"), log_time=ns_end)
    writer.close()
    return ep["T"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--traj", default="0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mount-z", type=float, default=0.35,
                    help="robot mount height (0.35 = enclosure-era platform)")
    ap.add_argument("--near", type=float, default=0.02)
    ap.add_argument("--far", type=float, default=0.60)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    fx.BASE_Z = args.mount_z
    model = fx.build_robot_model()
    mesh_update = fx.robot_mesh_scene_update(fx.extract_body_meshes(model),
                                             Timestamp(sec=0, nsec=0))
    with h5py.File(args.h5, "r") as f:
        any_traj = next(k for k in f if k.startswith("traj_"))
        sensor_names = list(f[any_traj]["obs/proximity"].keys())
        idxs = sorted(int(k.split("_")[1]) for k in f if k.startswith("traj_"))
    sensor_cams = {s: model.camera(f"{NS}{s}").id for s in sensor_names}

    kw = dict(model=model, mesh_update=mesh_update, sensor_cams=sensor_cams,
              near=args.near, far=args.far, stride=args.stride)
    if args.traj == "all":
        out_dir = Path(args.out_dir or "mcaps")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in idxs:
            T = export(args.h5, i, str(out_dir / f"ep{i}_dashboard.mcap"), **kw)
            print(f"traj {i}: {T} frames -> {out_dir}/ep{i}_dashboard.mcap")
    else:
        i = int(args.traj)
        out = args.out or f"ep{i}_dashboard.mcap"
        T = export(args.h5, i, out, **kw)
        print(f"traj {i}: {T} frames -> {out}")


if __name__ == "__main__":
    main()
