"""One-stop Foxglove export for franka_skin / hybrid-skin datagen episodes.

Takes an .h5 file (or a datagen run directory containing house_*/trajectories*.h5) and
writes ONE .mcap with every episode concatenated back-to-back on a single timeline — no
episode number to pick. Open the .mcap in Foxglove and import scripts/foxglove_layout.json.

Topics:
  /tf, /robot         robot kinematic tree + mesh, FK-replayed from the saved joints
  /scene_gt           ground-truth obstacle boxes from scene_params (when present)
  /targets            object start (green) / goal (blue) markers
  /proximity          all skin sensors' 8x8 returns back-projected to a world point cloud
                      using the saved per-step cam2world (turbo: red=near, blue=far)
  /tcp                end-effector pose
  /camera/exo         exo RGB video           (episode_*_exo_camera_1_*.mp4)
  /camera/wrist       wrist RGB video         (episode_*_wrist_camera_*.mp4)
  /sensors/heatmap8   8x8 sensor heatmap mosaic    (episode_*_sensors_depth8_heatmap_*.mp4)
  /sensors/rgb256     256x256 sensor RGB mosaic    (episode_*_sensors_rgb256_*.mp4)
  /joints             q1..q7 (rad), v1..v7 (rad/s), grip
  /task               episode start (index, task text), phase transitions, SUCCESS/FAIL

Usage:
  python scripts/foxglove_viz.py --h5 PATH.h5                      # one h5, all episodes
  python scripts/foxglove_viz.py --h5 RUN_DIR --out run.mcap      # every house_*/...h5
  python scripts/foxglove_viz.py --h5 PATH.h5 --mount-z 0.35      # robot pedestal height
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

import cv2
import h5py
import mujoco
import numpy as np
from matplotlib import colormaps

import foxglove
from foxglove import channels as C
from foxglove.schemas import (
    Color,
    CompressedImage,
    CubePrimitive,
    FrameTransform,
    FrameTransforms,
    Log,
    LogLevel,
    PackedElementField,
    PackedElementFieldNumericType as NT,
    Point3,
    PointCloud,
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

ROBOT_DIR = Path("/home/jaydv/code/prox_learning/assets/robots/franka_skin")
NS = "robot_0/"
FOVY = 45.0           # SPAD full FOV (matches the sim sensor cameras)
EP_GAP_S = 0.5        # silence between concatenated episodes

# h5-video-suffix -> mcap topic
VIDEO_TOPICS = {
    "exo_camera_1": "/camera/exo",
    "wrist_camera": "/camera/wrist",
    "sensors_depth8_heatmap": "/sensors/heatmap8",
    "sensors_rgb256": "/sensors/rgb256",
}

_TURBO = colormaps["turbo"]


# --------------------------------------------------------------------------- #
# Robot model + mesh (FK replay drives /tf; the mesh is frame_locked to it)
# --------------------------------------------------------------------------- #
def build_robot_model(robot_xml: Path, mount_z: float) -> mujoco.MjModel:
    """Robot attached to an empty world on a mocap base, lifted by the pedestal height."""
    xml = (
        '<mujoco model="prox_viz"><compiler angle="radian"/>'
        '<option gravity="0 0 -9.8" integrator="implicitfast"/>'
        '<worldbody><light pos="0 0 5" directional="true"/></worldbody></mujoco>'
    )
    fd = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
    fd.write(xml)
    fd.close()
    spec = mujoco.MjSpec.from_file(fd.name)
    robot = mujoco.MjSpec.from_file(str(robot_xml))
    base = spec.worldbody.add_body(name=f"{NS}base", pos=[0, 0, 0], quat=[1, 0, 0, 0], mocap=True)
    frame = base.add_frame(pos=[0, 0, mount_z])
    frame.attach_body(robot.worldbody.first_body(), NS, "")
    return spec.compile()


def extract_body_meshes(model: mujoco.MjModel) -> dict[str, dict]:
    """Visual mesh verts/faces per body, in the body frame (collision group 3 skipped)."""
    out: dict[str, dict] = {}
    for gi in range(model.ngeom):
        if int(model.geom_type[gi]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_group[gi]) == 3:
            continue
        mesh_id = int(model.geom_dataid[gi])
        if mesh_id < 0:
            continue
        vadr, vnum = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        fadr, fnum = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        verts = model.mesh_vert[vadr : vadr + vnum].reshape(-1, 3).astype(np.float64)
        faces = model.mesh_face[fadr : fadr + fnum].reshape(-1, 3).astype(np.int64)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[gi].astype(np.float64))
        verts = verts @ R.reshape(3, 3).T + model.geom_pos[gi].astype(np.float64)
        bname = model.body(int(model.geom_bodyid[gi])).name
        d = out.setdefault(bname, {"verts": [], "faces": [], "nv": 0})
        d["faces"].append(faces + d["nv"])
        d["verts"].append(verts)
        d["nv"] += vnum
    for d in out.values():
        d["verts"] = np.concatenate(d["verts"], axis=0)
        d["faces"] = np.concatenate(d["faces"], axis=0)
    return out


def robot_mesh_scene_update(body_meshes: dict[str, dict], ts: Timestamp) -> SceneUpdate:
    entities = []
    for bname, d in body_meshes.items():
        rgb = (90, 120, 200) if ("skin" in bname or "sensor" in bname) else (170, 170, 175)
        tri = TriangleListPrimitive(
            points=[Point3(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in d["verts"]],
            indices=[int(i) for i in d["faces"].ravel()],
            color=Color(r=rgb[0] / 255, g=rgb[1] / 255, b=rgb[2] / 255, a=1.0),
        )
        entities.append(SceneEntity(timestamp=ts, frame_id=bname, id=bname,
                                    frame_locked=True, triangles=[tri]))
    return SceneUpdate(entities=entities)


# --------------------------------------------------------------------------- #
# h5 loading
# --------------------------------------------------------------------------- #
def _dec(ds, i) -> dict:
    b = ds[i].tobytes()
    n = b.find(b"\x00")
    return json.loads(b[: n if n != -1 else None].decode("utf-8", "ignore"))


def load_episode(f: h5py.File, key: str) -> dict:
    t = f[key]
    T = t["obs/extra/tcp_pose"].shape[0]
    prox = {k: t[f"obs/proximity/{k}"][:] for k in t["obs/proximity"]}
    cam2w = {}
    if "obs/sensor_param" in t:
        for k in prox:
            if f"obs/sensor_param/{k}/cam2world_gl" in t:
                cam2w[k] = t[f"obs/sensor_param/{k}/cam2world_gl"][:].astype(np.float64)
    tcp = t["obs/extra/tcp_pose"][:].astype(np.float64)
    base = t["obs/extra/robot_base_pose"][:].astype(np.float64)
    phase = (t["obs/extra/policy_phase"][:] if "obs/extra/policy_phase" in t
             else np.zeros(T, int))
    success = bool(t["success"][-1]) if "success" in t else False
    q_rows, grip_rows, dq_rows = [], [], []
    for i in range(T):
        qp = _dec(t["obs/agent/qpos"], i)
        q_rows.append(qp["arm"])
        grip_rows.append((qp.get("gripper") or [0.0])[0])
        qv = _dec(t["obs/agent/qvel"], i)
        arm_v = qv.get("arm")
        dq_rows.append(arm_v if isinstance(arm_v, list) and len(arm_v) == 7 else [0.0] * 7)
    scene = {}
    if "obs_scene" in t:
        raw = t["obs_scene"]
        raw = raw[()] if raw.shape == () else raw[0]
        s = (raw.tobytes() if isinstance(raw, np.ndarray) else raw).decode("utf-8", "ignore")
        try:
            scene = json.loads(s.rstrip("\x00"))
        except Exception:
            scene = {}
    targets = {}
    for k in ("obj_start", "obj_end"):
        if f"obs/extra/{k}" in t:
            targets[k] = t[f"obs/extra/{k}"][0].astype(np.float64)
    return dict(T=T, prox=prox, cam2w=cam2w, tcp=tcp, base=base,
                phase=np.asarray(phase), success=success,
                q=np.array(q_rows), dq=np.array(dq_rows), grip=np.array(grip_rows),
                scene=scene, targets=targets,
                dt_s=float(scene.get("policy_dt_ms", 66.0)) / 1000.0)


def load_videos(h5_dir: Path, vid_id: int) -> dict[str, np.ndarray]:
    """Decode this episode's mp4s. Returns topic -> (N,H,W,3) BGR frames."""
    out = {}
    for stem, topic in VIDEO_TOPICS.items():
        cands = sorted(h5_dir.glob(f"episode_{vid_id:08d}_{stem}_batch_*.mp4"))
        if not cands:
            continue
        cap = cv2.VideoCapture(str(cands[0]))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if frames:
            out[topic] = np.stack(frames)
    return out


def episode_video_ids(h5_dir: Path) -> list[int]:
    ids = set()
    for p in h5_dir.glob("episode_*_exo_camera_1_batch_*.mp4"):
        m = re.match(r"episode_(\d+)_", p.name)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


# --------------------------------------------------------------------------- #
# Point cloud
# --------------------------------------------------------------------------- #
_PC_FIELDS = [
    PackedElementField(name="x", offset=0, type=NT.Float32),
    PackedElementField(name="y", offset=4, type=NT.Float32),
    PackedElementField(name="z", offset=8, type=NT.Float32),
    PackedElementField(name="red", offset=12, type=NT.Uint8),
    PackedElementField(name="green", offset=13, type=NT.Uint8),
    PackedElementField(name="blue", offset=14, type=NT.Uint8),
    PackedElementField(name="alpha", offset=15, type=NT.Uint8),
]


def backproject(d8: np.ndarray, c2w: np.ndarray, d_min: float, d_max: float):
    H = W = d8.shape[0]
    f = (H / 2) / np.tan(np.deg2rad(FOVY / 2))
    cx = cy = (H - 1) / 2.0
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    m = (d8 >= d_min) & (d8 <= d_max)
    if not m.any():
        return np.zeros((0, 3)), np.zeros((0,))
    d = d8[m].astype(np.float64)
    x_c = (u[m] - cx) * d / f
    y_c = -(v[m] - cy) * d / f
    p_cam = np.stack([x_c, y_c, -d, np.ones_like(d)], axis=1)
    return (c2w @ p_cam.T).T[:, :3], d


def pack_cloud(pts: np.ndarray, depths: np.ndarray, ts: Timestamp,
               near: float, far: float) -> PointCloud:
    n = len(pts)
    buf = np.zeros((n, 16), np.uint8)
    if n:
        buf[:, 0:12] = np.ascontiguousarray(pts, dtype="<f4").view(np.uint8).reshape(n, 12)
        norm = np.clip((depths - near) / (far - near), 0, 1)
        buf[:, 12:15] = (_TURBO(1.0 - norm)[:, :3] * 255).astype(np.uint8)  # near -> red
        buf[:, 15] = 255
    return PointCloud(
        timestamp=ts, frame_id="world",
        pose=Pose(position=Vector3(x=0.0, y=0.0, z=0.0),
                  orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
        point_stride=16, fields=_PC_FIELDS, data=buf.tobytes())


# --------------------------------------------------------------------------- #
# Ground-truth scene + target markers
# --------------------------------------------------------------------------- #
def scene_gt(scene: dict, ts: Timestamp) -> SceneUpdate | None:
    sp = scene.get("scene_params") or {}
    boxes = sp.get("obstacle_aabbs")
    if not boxes:
        return None
    bx, by, yaw = (sp.get("embed") or [0.0, 0.0, 0.0])
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.array([[c, -s, 0, bx], [s, c, 0, by], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])  # (w,x,y,z)
    n = len(boxes)
    protr = bool(sp.get("protrusion_present")) and "protr_center" in sp
    cubes = []
    for i, (ctr, half) in enumerate(boxes):
        w = (T @ np.array([ctr[0], ctr[1], ctr[2], 1.0]))[:3]
        col = (Color(r=0.90, g=0.22, b=0.21, a=0.55) if protr and i == n - 1
               else Color(r=0.75, g=0.78, b=0.82, a=0.22))
        cubes.append(CubePrimitive(
            pose=Pose(position=Vector3(x=float(w[0]), y=float(w[1]), z=float(w[2])),
                      orientation=Quaternion(x=float(qz[1]), y=float(qz[2]),
                                             z=float(qz[3]), w=float(qz[0]))),
            size=Vector3(x=2 * float(half[0]), y=2 * float(half[1]), z=2 * float(half[2])),
            color=col))
    return SceneUpdate(entities=[SceneEntity(timestamp=ts, frame_id="world", id="scene_gt",
                                             frame_locked=False, cubes=cubes)])


def target_markers(targets: dict, ts: Timestamp) -> SceneUpdate | None:
    if not targets:
        return None
    sph = []
    for key, col in (("obj_start", Color(r=0.18, g=0.80, b=0.44, a=0.9)),
                     ("obj_end", Color(r=0.18, g=0.55, b=0.95, a=0.5))):
        if key in targets:
            p = targets[key]
            sph.append(SpherePrimitive(
                pose=Pose(position=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                          orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
                size=Vector3(x=0.05, y=0.05, z=0.05), color=col))
    return SceneUpdate(entities=[SceneEntity(timestamp=ts, frame_id="world", id="targets",
                                             frame_locked=False, spheres=sph)])


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
class Channels:
    def __init__(self, ctx):
        self.tf = C.FrameTransformsChannel("/tf", context=ctx)
        self.mesh = C.SceneUpdateChannel("/robot", context=ctx)
        self.gt = C.SceneUpdateChannel("/scene_gt", context=ctx)
        self.tgt = C.SceneUpdateChannel("/targets", context=ctx)
        self.pc = C.PointCloudChannel("/proximity", context=ctx)
        self.tcp = C.PoseInFrameChannel("/tcp", context=ctx)
        self.log = C.LogChannel("/task", context=ctx)
        self.img = {tp: C.CompressedImageChannel(tp, context=ctx)
                    for tp in VIDEO_TOPICS.values()}
        self.joints = foxglove.Channel("/joints", message_encoding="json", context=ctx)


def export_episode(f: h5py.File, key: str, h5_dir: Path, vid_id: int, ep_label: str,
                   *, ch: Channels, model, data, mesh_update, offset_ns: int,
                   near: float, far: float, d_max: float, stride: int) -> int:
    """Writes one episode starting at offset_ns; returns its duration in ns."""
    ep = load_episode(f, key)
    vids = load_videos(h5_dir, vid_id)
    dt = ep["dt_s"]
    T = ep["T"]
    arm_qadr = [model.joint(f"{NS}fr3_joint{i}").qposadr[0] for i in range(1, 8)]
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    finger_qadr = [model.joint(f"{NS}gripper/{n}").qposadr[0]
                   for n in ("left_driver_joint", "right_driver_joint")
                   if f"{NS}gripper/{n}" in joint_names]
    pub_bodies = sorted({model.body(i).name for i in range(model.nbody)
                         if model.body(i).name and model.body(i).name != "world"})
    bp = ep["base"][0]
    base_mid = int(model.body_mocapid[model.body(f"{NS}base").id])
    cam_id = {s: model.camera(f"{NS}{s}").id for s in ep["prox"]}

    def ts_at(t):
        ns = offset_ns + int(round(t * dt * 1e9))
        return Timestamp(sec=ns // 1_000_000_000, nsec=ns % 1_000_000_000), ns

    ts0, ns0 = ts_at(0)
    desc = ep["scene"].get("task_description", "(no description)")
    ch.log.log(Log(timestamp=ts0, level=LogLevel.Info,
                   message=f"=== {ep_label}: {desc}", name="task"), log_time=ns0)
    gt = scene_gt(ep["scene"], ts0)
    if gt:
        ch.gt.log(gt, log_time=ns0)
    tgt = target_markers(ep["targets"], ts0)
    if tgt:
        ch.tgt.log(tgt, log_time=ns0)

    inv_phase = {v: k for k, v in (ep["scene"].get("policy_phases") or {}).items()}
    last_phase = None

    for t in range(0, T, stride):
        ts, ns = ts_at(t)
        # FK replay: base from the episode, arm + fingers from qpos
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
                rotation=Quaternion(x=float(qt[1]), y=float(qt[2]),
                                    z=float(qt[3]), w=float(qt[0]))))
        ch.tf.log(FrameTransforms(transforms=tfs), log_time=ns)
        if t == 0:
            ch.mesh.log(mesh_update, log_time=ns)

        # point cloud: saved cam2world when available (exact), FK pose otherwise
        all_pts, all_d = [], []
        for name, arr in ep["prox"].items():
            d8 = arr[t].mean(axis=0)
            if name in ep["cam2w"]:
                c2w = ep["cam2w"][name][t]
            else:
                c2w = np.eye(4)
                cid = cam_id[name]
                c2w[:3, :3] = data.cam_xmat[cid].reshape(3, 3)
                c2w[:3, 3] = data.cam_xpos[cid]
            pts, d = backproject(d8, c2w, near, d_max)
            if len(pts):
                all_pts.append(pts)
                all_d.append(d)
        pts = np.concatenate(all_pts) if all_pts else np.zeros((0, 3))
        dd = np.concatenate(all_d) if all_d else np.zeros((0,))
        ch.pc.log(pack_cloud(pts, dd, ts, near, far), log_time=ns)

        # videos -> image topics (proportional index if a video is off-length)
        for topic, arr in vids.items():
            vt = t if len(arr) == T else int(round(t * (len(arr) - 1) / max(T - 1, 1)))
            ok, buf = cv2.imencode(".jpg", arr[vt], [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                ch.img[topic].log(CompressedImage(timestamp=ts, frame_id=topic.strip("/"),
                                                  data=buf.tobytes(), format="jpeg"),
                                  log_time=ns)

        tcp = ep["tcp"][t]
        ch.tcp.log(PoseInFrame(timestamp=ts, frame_id="world", pose=Pose(
            position=Vector3(x=float(tcp[0]), y=float(tcp[1]), z=float(tcp[2])),
            orientation=Quaternion(x=float(tcp[4]), y=float(tcp[5]),
                                   z=float(tcp[6]), w=float(tcp[3])))), log_time=ns)

        q, dq = ep["q"][t], ep["dq"][t]
        jmsg = {f"q{i+1}": float(q[i]) for i in range(7)}
        jmsg |= {f"v{i+1}": float(dq[i]) for i in range(7)}
        jmsg["grip"] = float(ep["grip"][t])
        ch.joints.log(jmsg, log_time=ns)

        ph = int(ep["phase"][t])
        if ph != last_phase:
            ch.log.log(Log(timestamp=ts, level=LogLevel.Info,
                           message=f"phase -> {inv_phase.get(ph, ph)}", name="phase"),
                       log_time=ns)
            last_phase = ph

    ts_end, ns_end = ts_at(T - 1)
    ch.log.log(Log(timestamp=ts_end,
                   level=LogLevel.Info if ep["success"] else LogLevel.Warning,
                   message=f"=== {ep_label} END: {'SUCCESS' if ep['success'] else 'FAIL'}",
                   name="task"), log_time=ns_end)
    return int(round(T * dt * 1e9))


def find_h5s(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    h5s = sorted(path.glob("**/trajectories*.h5"))
    if not h5s:
        h5s = sorted(path.glob("**/*.h5"))
    return h5s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help=".h5 file or datagen run directory")
    ap.add_argument("--out", default=None, help="output .mcap (default: <h5 dir>/viz.mcap)")
    ap.add_argument("--mount-z", type=float, default=0.35,
                    help="robot pedestal height (fumehood-era platform = 0.35)")
    ap.add_argument("--near", type=float, default=0.02, help="turbo color near clip (m)")
    ap.add_argument("--far", type=float, default=0.60, help="turbo color far clip (m)")
    ap.add_argument("--d-max", type=float, default=1.5, help="drop returns beyond this (m)")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    src = Path(args.h5)
    h5s = find_h5s(src)
    if not h5s:
        raise SystemExit(f"no .h5 under {src}")
    out = Path(args.out) if args.out else (src.parent if src.is_file() else src) / "viz.mcap"

    # pick the robot model that matches the dataset's skin (40-sensor hybrid h5s have
    # link5_front / link1_* sensors; 29-sensor h5s do not)
    with h5py.File(h5s[0], "r") as f:
        any_traj = next(k for k in f if k.startswith("traj_"))
        sensor_names = list(f[any_traj]["obs/proximity"].keys())
    hybrid = any("link5_front" in s or s.startswith("link1_") for s in sensor_names)
    robot_xml = ROBOT_DIR / ("model_hybrid.xml" if hybrid else "model.xml")
    model = build_robot_model(robot_xml, args.mount_z)
    data = mujoco.MjData(model)
    mesh_update = robot_mesh_scene_update(extract_body_meshes(model), Timestamp(sec=0, nsec=0))

    ctx = foxglove.Context()
    writer = foxglove.open_mcap(str(out), allow_overwrite=True, context=ctx)
    ch = Channels(ctx)

    offset_ns = 0
    n_eps = 0
    for h5_path in h5s:
        h5_dir = h5_path.parent
        with h5py.File(h5_path, "r") as f:
            idxs = sorted(int(k.split("_")[1]) for k in f if k.startswith("traj_"))
            vid_ids = episode_video_ids(h5_dir)
            # traj index == episode id when the files line up; otherwise map by order
            if all(i in vid_ids for i in idxs):
                vid_of = {i: i for i in idxs}
            else:
                vid_of = dict(zip(idxs, vid_ids))
            for i in idxs:
                label = f"EPISODE {n_eps} ({h5_dir.name}/traj_{i})"
                dur = export_episode(
                    f, f"traj_{i}", h5_dir, vid_of.get(i, i), label,
                    ch=ch, model=model, data=data, mesh_update=mesh_update,
                    offset_ns=offset_ns, near=args.near, far=args.far,
                    d_max=args.d_max, stride=args.stride)
                offset_ns += dur + int(EP_GAP_S * 1e9)
                n_eps += 1
                print(f"  {label} -> {dur/1e9:.1f}s")
    writer.close()
    print(f"{n_eps} episode(s), {offset_ns/1e9:.1f}s timeline -> {out}")


if __name__ == "__main__":
    main()
