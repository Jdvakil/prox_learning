"""Export a franka_skin proximity-data trajectory (.h5) to a Foxglove .mcap file.

What you get when you open the .mcap in Foxglove (https://app.foxglove.dev, "Open local file"):

  /tf            full robot kinematic tree (every link + the 29 SPAD sensor frames),
                 replayed from the saved joint trajectory.
  /robot         the robot mesh (links + proximity "skin"), extracted straight from the
                 compiled MuJoCo model and anchored to the live frames.
  /proximity     the 29 proximity sensors' 8x8 depth returns, back-projected to a single
                 world-frame point cloud, colored by distance (turbo: red=near, blue=far).
                 This is the headline view: you literally watch surfaces light up around the
                 arm as it reaches into a cavity.
  /proximity_by_link   the same points, colored by which link's sensor saw them.
  /camera/wrist, /camera/exo   the recorded RGB camera videos, frame-synced to the rollout.
  /tcp           the end-effector pose.
  /task          a Log stream: the task description (t=0) and every policy-phase transition.

The world frame is robot-base-centric (base at the origin). The point cloud is reconstructed
from the saved depth using each sensor's forward-kinematic pose, so it is geometrically exact
relative to the arm regardless of which house/scene the data came from.

Usage:
  python scripts/foxglove_export.py --h5 PATH.h5 --traj 0 --out traj0.mcap
  python scripts/foxglove_export.py --h5 PATH.h5 --traj all --out-dir mcaps/
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import h5py
import mujoco
import numpy as np

import foxglove
from foxglove import channels as C
from foxglove.schemas import (
    CameraCalibration,
    Color,
    CompressedImage,
    FrameTransform,
    FrameTransforms,
    Log,
    LogLevel,
    ModelPrimitive,
    PackedElementField,
    PackedElementFieldNumericType as NT,
    PointCloud,
    Pose,
    PoseInFrame,
    Quaternion,
    SceneEntity,
    SceneUpdate,
    Timestamp,
    TriangleListPrimitive,
    Vector3,
    Point3,
)

from matplotlib import colormaps

ROBOT_XML = "/home/jaydv/code/prox_learning/assets/robots/franka_skin/model.xml"
NS = "robot_0/"
BASE_Z = 0.58  # robot base mount height (matches scripts/datagen/verify_synthetic_scenes.py)

# Per-link colors for the robot mesh + the by-link point cloud.
LINK_RGB = {
    "link2": (230, 57, 70),
    "link3": (42, 157, 143),
    "link5": (38, 70, 83),
    "link6": (244, 162, 97),
}
_DEFAULT_LINK_RGB = (170, 170, 175)
_SKIN_RGB = (90, 120, 200)

_TURBO = colormaps["turbo"]


# --------------------------------------------------------------------------- #
# Model + mesh extraction
# --------------------------------------------------------------------------- #
def build_robot_model() -> mujoco.MjModel:
    """franka_skin attached to an empty world, base at the origin (lifted BASE_Z)."""
    xml = (
        '<mujoco model="prox_viz"><compiler angle="radian"/>'
        '<option gravity="0 0 -9.8" integrator="implicitfast"/>'
        '<worldbody><light pos="0 0 5" directional="true"/></worldbody></mujoco>'
    )
    fd = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
    fd.write(xml)
    fd.close()
    spec = mujoco.MjSpec.from_file(fd.name)
    robot = mujoco.MjSpec.from_file(ROBOT_XML)
    base = spec.worldbody.add_body(name=f"{NS}base", pos=[0, 0, 0], quat=[1, 0, 0, 0], mocap=True)
    frame = base.add_frame(pos=[0, 0, BASE_Z])
    frame.attach_body(robot.worldbody.first_body(), NS, "")
    return spec.compile()


def _quat_from_mat9(mat9: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat9, dtype=np.float64).ravel())
    return q  # (w, x, y, z)


def _body_color(body_name: str) -> tuple[int, int, int]:
    if "skin" in body_name or "sensor" in body_name:
        return _SKIN_RGB
    for key, rgb in LINK_RGB.items():
        if key in body_name:
            return rgb
    return _DEFAULT_LINK_RGB


def extract_body_meshes(model: mujoco.MjModel) -> dict[str, dict]:
    """For each body that owns visual mesh geoms, return verts/faces in the BODY frame.

    Reads geometry straight from the compiled model (no external asset paths needed).
    Collision geoms (group 3) are skipped so we render only the visual robot.
    """
    out: dict[str, dict] = {}
    for gi in range(model.ngeom):
        if int(model.geom_type[gi]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_group[gi]) == 3:  # collision
            continue
        mesh_id = int(model.geom_dataid[gi])
        if mesh_id < 0:
            continue
        vadr = int(model.mesh_vertadr[mesh_id])
        vnum = int(model.mesh_vertnum[mesh_id])
        fadr = int(model.mesh_faceadr[mesh_id])
        fnum = int(model.mesh_facenum[mesh_id])
        verts = model.mesh_vert[vadr : vadr + vnum].reshape(-1, 3).astype(np.float64)
        faces = model.mesh_face[fadr : fadr + fnum].reshape(-1, 3).astype(np.int64)
        # geom local pose relative to its body
        gpos = model.geom_pos[gi].astype(np.float64)
        gquat = model.geom_quat[gi].astype(np.float64)  # (w,x,y,z)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, gquat)
        R = R.reshape(3, 3)
        verts_body = verts @ R.T + gpos
        bid = int(model.geom_bodyid[gi])
        bname = model.body(bid).name
        d = out.setdefault(bname, {"verts": [], "faces": [], "nv": 0})
        d["faces"].append(faces + d["nv"])
        d["verts"].append(verts_body)
        d["nv"] += vnum
    for bname, d in out.items():
        d["verts"] = np.concatenate(d["verts"], axis=0)
        d["faces"] = np.concatenate(d["faces"], axis=0)
    return out


def robot_mesh_scene_update(body_meshes: dict[str, dict], ts: Timestamp) -> SceneUpdate:
    """One SceneEntity per body, frame_locked to that body's frame (follows /tf)."""
    entities = []
    for bname, d in body_meshes.items():
        verts = d["verts"]
        faces = d["faces"]
        pts = [Point3(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in verts]
        indices = [int(i) for i in faces.ravel()]
        r, g, b = _body_color(bname)
        tri = TriangleListPrimitive(
            points=pts,
            indices=indices,
            color=Color(r=r / 255, g=g / 255, b=b / 255, a=1.0),
        )
        entities.append(
            SceneEntity(
                timestamp=ts,
                frame_id=bname,
                id=bname,
                frame_locked=True,
                triangles=[tri],
            )
        )
    return SceneUpdate(entities=entities)


# --------------------------------------------------------------------------- #
# h5 loading
# --------------------------------------------------------------------------- #
SENSOR_LINK_JOINTS = [f"{NS}fr3_joint{i}" for i in range(1, 8)]


def _decode_json_blob(ds, t) -> dict:
    b = ds[t].tobytes()
    nul = b.find(b"\x00")
    if nul != -1:
        b = b[:nul]
    return json.loads(b.decode("utf-8"))


def load_traj(h5_path: str, idx: int) -> dict:
    with h5py.File(h5_path, "r") as f:
        key = f"traj_{idx}"
        t = f[key]
        panda = t["env_states/articulations/panda"][:].astype(np.float64)  # (T,31)
        T = panda.shape[0]
        prox = {k: t[f"obs/proximity/{k}"][:] for k in t["obs/proximity"].keys()}  # (T,4,8,8)
        tcp = t["obs/extra/tcp_pose"][:].astype(np.float64)  # (T,7) [x,y,z, qw,qx,qy,qz]
        phase = t["obs/extra/policy_phase"][:] if "obs/extra/policy_phase" in t else np.zeros(T, int)
        success = t["success"][:] if "success" in t else np.zeros(T, bool)
        # task / phase metadata
        scene = {}
        if "obs_scene" in t:
            raw = t["obs_scene"]
            raw = raw[()] if raw.shape == () else raw[0]
            if isinstance(raw, (bytes, np.bytes_)):
                s = raw.decode("utf-8", "ignore").rstrip("\x00")
            elif isinstance(raw, np.ndarray):
                s = raw.tobytes().decode("utf-8", "ignore").rstrip("\x00")
            else:
                s = str(raw)
            try:
                scene = json.loads(s)
            except Exception:
                scene = {}
    dt_ms = float(scene.get("policy_dt_ms", 66.0))
    phase_map = scene.get("policy_phases", {})
    inv_phase = {v: k for k, v in phase_map.items()} if phase_map else {}
    return dict(
        panda=panda, T=T, prox=prox, tcp=tcp, phase=np.asarray(phase),
        success=np.asarray(success), scene=scene, dt_ms=dt_ms, inv_phase=inv_phase,
    )


def load_camera_videos(h5_path: str, idx: int) -> dict[str, np.ndarray]:
    """Decode the RGB mp4s next to the h5 for this episode. Returns name -> (T,H,W,3) BGR
    (BGR so cv2.imencode produces a correctly-colored JPEG for Foxglove)."""
    d = Path(h5_path).parent
    vids = {}
    for cam in ["wrist_camera", "exo_camera_1"]:
        # episode_00000000_wrist_camera_batch_1_of_1.mp4
        cands = list(d.glob(f"episode_{idx:08d}_{cam}_batch_*.mp4"))
        if not cands:
            continue
        cap = cv2.VideoCapture(str(cands[0]))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)  # keep BGR
        cap.release()
        if frames:
            short = "wrist" if "wrist" in cam else "exo"
            vids[short] = np.stack(frames)
    return vids


# --------------------------------------------------------------------------- #
# Proximity back-projection (matches verify_synthetic_scenes.reconstruct_world_points)
# --------------------------------------------------------------------------- #
def backproject_sensor(depth_8x8: np.ndarray, cam2world: np.ndarray,
                       fovy: float, d_min: float, d_max: float):
    H = W = depth_8x8.shape[0]
    f = (H / 2) / np.tan(np.deg2rad(fovy / 2))
    cx = cy = (H - 1) / 2.0
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    mask = (depth_8x8 >= d_min) & (depth_8x8 <= d_max)
    if not mask.any():
        return np.zeros((0, 3)), np.zeros((0,))
    d = depth_8x8[mask].astype(np.float64)
    uu = u[mask].astype(np.float64)
    vv = v[mask].astype(np.float64)
    x_c = (uu - cx) * d / f
    y_c = -(vv - cy) * d / f
    z_c = -d
    p_cam = np.stack([x_c, y_c, z_c, np.ones_like(d)], axis=1)
    p_world = (cam2world @ p_cam.T).T[:, :3]
    return p_world, d


def turbo_rgba(depths: np.ndarray, near: float, far: float) -> np.ndarray:
    norm = np.clip((depths - near) / (far - near), 0, 1)
    cols = (_TURBO(1.0 - norm)[:, :3] * 255).astype(np.uint8)  # near -> red
    out = np.empty((len(depths), 4), np.uint8)
    out[:, :3] = cols
    out[:, 3] = 255
    return out


_PC_FIELDS = [
    PackedElementField(name="x", offset=0, type=NT.Float32),
    PackedElementField(name="y", offset=4, type=NT.Float32),
    PackedElementField(name="z", offset=8, type=NT.Float32),
    PackedElementField(name="red", offset=12, type=NT.Uint8),
    PackedElementField(name="green", offset=13, type=NT.Uint8),
    PackedElementField(name="blue", offset=14, type=NT.Uint8),
    PackedElementField(name="alpha", offset=15, type=NT.Uint8),
]


def pack_cloud(pts: np.ndarray, rgba: np.ndarray, ts: Timestamp, frame="world") -> PointCloud:
    n = len(pts)
    buf = np.zeros((n, 16), np.uint8)
    xyz = np.ascontiguousarray(pts, dtype="<f4")
    buf[:, 0:12] = xyz.view(np.uint8).reshape(n, 12)
    buf[:, 12:16] = rgba
    return PointCloud(
        timestamp=ts, frame_id=frame, pose=Pose(
            position=Vector3(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        point_stride=16, fields=_PC_FIELDS, data=buf.tobytes(),
    )


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_traj(h5_path: str, idx: int, out_path: str, *, model, mesh_update,
                cam_ids, sensor_h5_to_cam, link_of_sensor, with_mesh=True,
                near=0.05, far=1.0, d_max=2.0, stride=1, max_frames=None):
    data = mujoco.MjData(model)
    tr = load_traj(h5_path, idx)
    vids = load_camera_videos(h5_path, idx)
    T = tr["T"]
    dt_s = tr["dt_ms"] / 1000.0
    arm_qadr = [model.joint(j).qposadr[0] for j in SENSOR_LINK_JOINTS]

    # publish a complete frame tree (every named robot body) on /tf
    pub_bodies = sorted({model.body(i).name for i in range(model.nbody)
                         if model.body(i).name and model.body(i).name != "world"})

    # isolate channels per output file so repeated topics don't collide across trajs
    ctx = foxglove.Context()
    writer = foxglove.open_mcap(out_path, allow_overwrite=True, context=ctx)
    tf_ch = C.FrameTransformsChannel("/tf", context=ctx)
    pc_ch = C.PointCloudChannel("/proximity", context=ctx)
    pcl_ch = C.PointCloudChannel("/proximity_by_link", context=ctx)
    tcp_ch = C.PoseInFrameChannel("/tcp", context=ctx)
    log_ch = C.LogChannel("/task", context=ctx)
    mesh_ch = C.SceneUpdateChannel("/robot", context=ctx) if with_mesh else None
    img_ch = {name: C.CompressedImageChannel(f"/camera/{name}", context=ctx) for name in vids}

    def ts_at(t):
        ns = int(round(t * dt_s * 1e9))
        return Timestamp(sec=ns // 1_000_000_000, nsec=ns % 1_000_000_000)

    # task description at t=0
    desc = tr["scene"].get("task_description", "(no description)")
    log_ch.log(Log(timestamp=ts_at(0), level=LogLevel.Info, message=f"TASK: {desc}", name="task"),
               log_time=0)

    last_phase = None
    frames = list(range(0, T, stride))
    if max_frames:
        frames = frames[:max_frames]
    first_t = frames[0] if frames else 0

    for t in frames:
        ts = ts_at(t)
        log_ns = int(round(t * dt_s * 1e9))

        # set arm joints + forward kinematics
        for adr, val in zip(arm_qadr, tr["panda"][t, :7]):
            data.qpos[adr] = float(val)
        mujoco.mj_forward(model, data)

        # /tf : world -> every body
        tfs = []
        for bname in pub_bodies:
            bid = model.body(bname).id
            p = data.xpos[bid]
            q = data.xquat[bid]  # (w,x,y,z)
            tfs.append(FrameTransform(
                timestamp=ts, parent_frame_id="world", child_frame_id=bname,
                translation=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                rotation=Quaternion(x=float(q[1]), y=float(q[2]), z=float(q[3]), w=float(q[0])),
            ))
        tf_ch.log(FrameTransforms(transforms=tfs), log_time=log_ns)

        # robot mesh (publish once, frame_locked entities then follow /tf)
        if with_mesh and t == first_t:
            mesh_ch.log(mesh_update, log_time=log_ns)

        # /proximity : back-project all sensors into one world cloud
        all_pts, all_depth, all_link = [], [], []
        for h5name, camname in sensor_h5_to_cam.items():
            cam_id = cam_ids[h5name]
            depth = tr["prox"][h5name][t].mean(axis=0)  # mean over 4 substeps -> (8,8)
            cam2world = np.eye(4)
            cam2world[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3)
            cam2world[:3, 3] = data.cam_xpos[cam_id]
            pts, d = backproject_sensor(depth, cam2world, 45.0, near, d_max)
            if len(pts):
                all_pts.append(pts)
                all_depth.append(d)
                all_link.append(np.full(len(pts), link_of_sensor[h5name]))
        if all_pts:
            pts = np.concatenate(all_pts)
            depth = np.concatenate(all_depth)
            link = np.concatenate(all_link)
            pc_ch.log(pack_cloud(pts, turbo_rgba(depth, near, far), ts), log_time=log_ns)
            # by-link coloring
            lk_rgba = np.zeros((len(pts), 4), np.uint8)
            lk_rgba[:, 3] = 255
            for lid, rgb in LINK_PALETTE.items():
                m = link == lid
                lk_rgba[m, :3] = rgb
            pcl_ch.log(pack_cloud(pts, lk_rgba, ts), log_time=log_ns)
        else:
            # still emit an empty cloud so the topic exists at this time
            pc_ch.log(pack_cloud(np.zeros((0, 3)), np.zeros((0, 4), np.uint8), ts), log_time=log_ns)

        # /tcp
        tcp = tr["tcp"][t]
        tcp_ch.log(PoseInFrame(
            timestamp=ts, frame_id=f"{NS}base",
            pose=Pose(
                position=Vector3(x=float(tcp[0]), y=float(tcp[1]), z=float(tcp[2])),
                orientation=Quaternion(x=float(tcp[4]), y=float(tcp[5]), z=float(tcp[6]), w=float(tcp[3])),
            ),
        ), log_time=log_ns)

        # cameras (JPEG-compressed to keep file sizes sane)
        for name, arr in vids.items():
            if t < len(arr):
                ok, buf = cv2.imencode(".jpg", arr[t], [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    img_ch[name].log(CompressedImage(
                        timestamp=ts, frame_id=f"camera_{name}",
                        data=buf.tobytes(), format="jpeg",
                    ), log_time=log_ns)

        # phase transitions
        ph = int(tr["phase"][t])
        if ph != last_phase:
            phname = tr["inv_phase"].get(ph, str(ph))
            log_ch.log(Log(timestamp=ts, level=LogLevel.Info,
                           message=f"phase -> {phname}", name="phase"), log_time=log_ns)
            last_phase = ph

    writer.close()
    return T, len(list(frames))


# link id -> color for by-link cloud
LINK_PALETTE = {2: (230, 57, 70), 3: (42, 157, 143), 5: (38, 70, 83), 6: (244, 162, 97)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--traj", default="0", help="trajectory index, or 'all'")
    ap.add_argument("--out", default=None, help="output .mcap (single traj)")
    ap.add_argument("--out-dir", default=None, help="output dir (for --traj all)")
    ap.add_argument("--no-mesh", action="store_true")
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--far", type=float, default=1.0, help="turbo color far clip (m)")
    ap.add_argument("--d-max", type=float, default=2.0, help="drop returns farther than this (m)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    model = build_robot_model()
    body_meshes = extract_body_meshes(model)
    mesh_update = robot_mesh_scene_update(body_meshes, Timestamp(sec=0, nsec=0))

    # map h5 sensor name -> model camera id + link id
    cam_ids, sensor_h5_to_cam, link_of_sensor = {}, {}, {}
    with h5py.File(args.h5, "r") as f:
        any_traj = next(k for k in f.keys() if k.startswith("traj_"))
        sensor_names = list(f[any_traj]["obs/proximity"].keys())
    for s in sensor_names:
        camname = f"{NS}{s}"
        cam_ids[s] = model.camera(camname).id
        sensor_h5_to_cam[s] = camname
        link_of_sensor[s] = int(s.split("_")[0].replace("link", ""))

    kw = dict(model=model, mesh_update=mesh_update, cam_ids=cam_ids,
              sensor_h5_to_cam=sensor_h5_to_cam, link_of_sensor=link_of_sensor,
              with_mesh=not args.no_mesh, near=args.near, far=args.far,
              d_max=args.d_max, stride=args.stride, max_frames=args.max_frames)

    if args.traj == "all":
        with h5py.File(args.h5, "r") as f:
            idxs = sorted(int(k.split("_")[1]) for k in f.keys() if k.startswith("traj_"))
        out_dir = Path(args.out_dir or "mcaps")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in idxs:
            out = out_dir / f"traj_{i}.mcap"
            T, n = export_traj(args.h5, i, str(out), **kw)
            print(f"traj {i}: {n} frames (of {T}) -> {out}")
    else:
        i = int(args.traj)
        out = args.out or f"traj_{i}.mcap"
        T, n = export_traj(args.h5, i, out, **kw)
        print(f"traj {i}: {n} frames (of {T}) -> {out}")


if __name__ == "__main__":
    main()
