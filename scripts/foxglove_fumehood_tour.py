"""ONE playable Foxglove trajectory touring ALL SIX fumehood variants with the FR3 + 40-SPAD
hybrid skin. The robot inserts into each hood in sequence (verified waypoints from the variation
suite); the hood geometry swaps at each chapter boundary.

Topics (open the mcap at app.foxglove.dev; layout: scripts/foxglove_fumehood_tour_layout.json):
  /tf, /robot        robot kinematic tree + mesh
  /scene             the CURRENT hood (boxes; translucent walls, sash, bench, target)
  /proximity         live 40-sensor skin point cloud (turbo, red=near)
  /camera/exo        RGB exo view (fixed, whole robot in frame)
  /skin_mosaic       all 40 sensor tiles as one image
  /skin (json)       per-link min distance, global min, active count
  /motion (json)     insertion depth (cm), variant id
  /task (Log)        chapter announcements with hood dims + verified stats

Usage:  python scripts/foxglove_fumehood_tour.py  [--out PATH.mcap]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_viz_lib import (build, set_pose, skin_cloud, depth_renderer, depth8,  # noqa: E402
                            nice_lights, add_box, NEAR, FAR)
import foxglove_export as fx  # noqa: E402  (mesh extraction, cloud packing)

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402

import foxglove  # noqa: E402
from foxglove import channels as C  # noqa: E402
from foxglove.schemas import (CompressedImage, Color, CubePrimitive, FrameTransform,  # noqa: E402
                              FrameTransforms, Log, LogLevel, Pose, Quaternion,
                              SceneEntity, SceneUpdate, Timestamp, Vector3)

TURBO = colormaps["turbo"]
BZ, X0 = 0.585, 0.35
DT = 0.12          # s per frame on the playback timeline
NSEG = 8           # interp steps between waypoints

# ---------------------------------------------------------------------------------------------
# the six variants: dims (W half-width, H, D, SASH) + verified insertion waypoints
# ---------------------------------------------------------------------------------------------
VARIANTS = [
    ("standard", (0.32, 0.46, 0.55, 0.30), "64x46x55cm sash 30 — 31.6cm in, 28/40 active", [
        [0.0, -0.800, 0.0, -2.600, 0.0, 1.900, 0.79],
        [0.0, -0.598, 0.0, -2.070, 0.0, 2.168, 0.79],
        [0.0, -0.144, 0.0, -1.540, 0.0, 1.746, 0.79],
        [0.0, 0.340, 0.0, -0.980, 0.0, 1.880, 0.79]]),
    ("tall", (0.34, 0.85, 0.55, 0.55), "68x85x55cm sash 55 — deep + high interior sweep", [
        [0.0, -0.800, 0.0, -2.600, 0.0, 1.900, 0.79],
        [0.0, -0.535, 0.0, -1.855, 0.0, 2.187, 0.79],
        [0.0, -0.085, 0.0, -1.479, 0.0, 1.953, 0.79],
        [0.0, 0.450, 0.0, -0.905, 0.0, 1.907, 0.79],
        [0.0, 0.513, 0.0, -0.470, 0.0, 2.400, 0.79]]),
    ("short_low_sash", (0.34, 0.34, 0.50, 0.17), "68x34x50cm sash 17 — ducks flat, 0.3cm clearance", [
        [0.0, -1.3107, 0.0, -2.3498, 0.0, 1.4755, 0.0],
        [0.0, -1.1988, 0.0, -2.4978, 0.0, 2.3461, 0.0],
        [0.0, -0.6486, 0.0, -2.4553, 0.0, 3.3124, 0.0],
        [0.0, -0.2254, 0.0, -2.1006, 0.0, 3.4214, 0.0],
        [0.0, 0.1888, 0.0, -1.6044, 0.0, 3.3337, 0.0]]),
    ("narrow", (0.16, 0.50, 0.55, 0.32), "32cm-wide interior — 0.4cm lateral clearance", [
        [0.0, -0.800, 0.0, -2.600, 0.0, 1.900, 0.79],
        [0.0, -0.802, 0.0, -2.060, 0.0, 1.782, 0.79],
        [0.0, -0.480, 0.0, -1.902, 0.0, 1.859, 0.79],
        [0.0, -0.052, 0.0, -1.504, 0.0, 1.713, 0.79],
        [0.0, 0.405, 0.0, -0.965, 0.0, 1.802, 0.79]]),
    ("wide_big", (0.55, 0.65, 0.70, 0.45), "110x65x70cm — deep + lateral sweep", [
        [0.0, -0.800, 0.0, -2.600, 0.0, 1.900, 0.79],
        [0.0, -0.144, 0.0, -1.540, 0.0, 1.746, 0.79],
        [0.0, 0.340, 0.0, -0.980, 0.0, 1.880, 0.79],
        [0.45, 0.340, 0.0, -0.980, 0.0, 1.880, 0.79],
        [-0.45, 0.340, 0.0, -0.980, 0.0, 1.880, 0.79],
        [0.0, 0.340, 0.0, -0.980, 0.0, 1.880, 0.79]]),
    ("deep_tunnel", (0.26, 0.42, 0.95, 0.28), "95cm deep — 52.6cm TCP, whole forearm inside", [
        [0.0, -1.277, 0.0, -2.402, 0.0, 2.057, 0.79],
        [0.0, -0.728, 0.0, -2.238, 0.0, 2.397, 0.79],
        [0.0, -0.050, 0.0, -1.710, 0.0, 2.682, 0.79],
        [0.0, 0.4724, 0.029, -1.0808, 0.2999, 2.7971, 0.79]]),
]


def hood_boxes(W, H, D, SASH):
    return [
        ("bench", [X0 + D / 2, 0, BZ - 0.015], [D / 2 + 0.05, W + 0.05, 0.015], [0.62, 0.55, 0.45, 1]),
        ("bench_body", [X0 + D / 2, 0, BZ / 2 - 0.02], [D / 2, W, BZ / 2 - 0.02], [0.55, 0.5, 0.44, 1]),
        ("wall_l", [X0 + D / 2, W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30]),
        ("wall_r", [X0 + D / 2, -W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30]),
        ("back", [X0 + D, 0, BZ + H / 2], [0.012, W, H / 2], [0.72, 0.7, 0.66, 1]),
        ("top", [X0 + D / 2, 0, BZ + H], [D / 2, W, 0.012], [0.78, 0.8, 0.84, 0.30]),
        ("sash", [X0, 0, BZ + SASH + 0.028], [0.012, W, 0.028], [0.62, 0.64, 0.66, 1]),
    ]


def scene_update(boxes, ts) -> SceneUpdate:
    cubes = []
    for nm, c, h, col in boxes:
        cubes.append(CubePrimitive(
            pose=Pose(position=Vector3(x=c[0], y=c[1], z=c[2]),
                      orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
            size=Vector3(x=2 * h[0], y=2 * h[1], z=2 * h[2]),
            color=Color(r=col[0], g=col[1], b=col[2], a=col[3])))
    return SceneUpdate(entities=[SceneEntity(timestamp=ts, frame_id="world", id="hood",
                                             frame_locked=False, cubes=cubes)])


def interp(wps, nseg):
    out = []
    for a, b in zip(wps[:-1], wps[1:]):
        a, b = np.array(a), np.array(b)
        for t in np.linspace(0, 1, nseg, endpoint=False):
            s = 0.5 - 0.5 * np.cos(np.pi * t)
            out.append(a + s * (b - a))
    out.append(np.array(wps[-1]))
    return out


def mosaic(model, data, rd, names):
    T, cols, pad = 56, 10, 4
    rows = int(np.ceil(len(names) / cols))
    canvas = np.full((rows * (T + 16), cols * (T + pad), 3), 20, np.uint8)
    for k, n in enumerate(sorted(names)):
        d8 = depth8(rd, data, n)
        nrm = np.clip((d8 - NEAR) / (FAR - NEAR), 0, 1)
        img = (TURBO(1.0 - nrm)[:, :, :3] * 255).astype(np.uint8)
        img[d8 > FAR] = (14, 14, 14)
        til = cv2.resize(img[..., ::-1], (T, T), interpolation=cv2.INTER_NEAREST)
        r, c = divmod(k, cols)
        y0, x0 = r * (T + 16) + 16, c * (T + pad)
        canvas[y0:y0 + T, x0:x0 + T] = til
        short = n.replace("link", "L").replace("_sensor_", ".").replace("_front", "F").replace("_back", "B")
        cv2.putText(canvas, short, (x0 + 1, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diagnostics_output/20260611_fumehood_tour/fumehood_tour.mcap")
    args = ap.parse_args()
    outp = Path("/home/jaydv/code/prox_learning") / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)

    # one model containing ALL hoods, each variant's boxes parked far away as mocap bodies we
    # don't move — simpler: ONE model per variant is impossible mid-mcap, so build a single
    # model with mocap hood slabs we re-pose per chapter (mirrors the datagen mocap approach).
    # The in-hood target is a REAL objaverse mesh riding a mocap mount (re-posable per chapter);
    # its triangles flow into /robot's mesh extraction so Foxglove shows the actual object.
    def attach_obja_target(spec):
        from molmo_spaces.utils.synset_utils import get_valid_pickupable_obja_uids
        from molmo_spaces.utils.object_metadata import ObjectMeta
        from molmo_spaces.utils.lazy_loading_utils import install_uid
        from scipy.spatial.transform import Rotation as R
        for uid in get_valid_pickupable_obja_uids():
            anno = ObjectMeta.annotation(uid) or {}
            cat = str(anno.get("category", "")).lower()
            bb = anno.get("boundingBox", {})
            dims = sorted(float(bb.get(k, 0)) for k in "xyz")
            if cat != "egg" and 0.03 <= dims[0] and dims[2] <= 0.07:
                break
        pxml = install_uid(uid)
        ps = mujoco.MjSpec.from_file(str(pxml))
        pb = ps.worldbody.bodies[0]
        while pb.first_joint() is not None:        # rigid ride on the mocap mount
            ps.delete(pb.first_joint())
        mount = spec.worldbody.add_body(name="obj_mount", mocap=True, pos=[0, 0, -9])
        q = R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)
        mount.add_frame(pos=[0, 0, 0], quat=list(q)).attach_body(pb, "tourobj/", "")
        z_half = float(bb.get("z", 0.06)) / 2
        print(f"target object: {uid} ({cat}) z_half={z_half:.3f}")
        return z_half

    def mk(spec):
        nice_lights(spec)
        # mocap slabs: bench, bench_body, wall_l, wall_r, back, top, sash, target
        mk.obj_zhalf = attach_obja_target(spec)
        for nm, half in (("bench", [0.55, 0.65, 0.015]), ("bench_body", [0.5, 0.6, 0.27]),
                         ("wall_l", [0.5, 0.012, 0.45]), ("wall_r", [0.5, 0.012, 0.45]),
                         ("back", [0.012, 0.6, 0.45]), ("top", [0.5, 0.6, 0.012]),
                         ("sash", [0.012, 0.6, 0.028])):
            b = spec.worldbody.add_body(name=f"m_{nm}", mocap=True, pos=[0, 0, -9])
            g = b.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = half
            g.rgba = [0.7, 0.7, 0.72, 1.0]
            g.contype = 0
            g.conaffinity = 0
        exo = spec.worldbody.add_camera()
        exo.name = "exo_camera_1"
        exo.pos = [1.45, -1.25, 1.15]
        v = np.array([0.4, 0.0, 0.55]) - np.array(exo.pos)
        v /= np.linalg.norm(v)
        z = -v
        up = np.array([0, 0, 1.0])
        x = np.cross(up, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).reshape(9))
        exo.quat = [float(t) for t in q]
        exo.fovy = 52
        exo.resolution = [480, 640]

    model = build(mk, offw=1280, offh=1024)
    data = mujoco.MjData(model)
    names = sorted(model.camera(i).name for i in range(model.ncam) if "_sensor_" in model.camera(i).name)
    rd = depth_renderer(model)
    rgb = mujoco.Renderer(model, 480, 640)

    def pose_hood(W, H, D, SASH):
        sizes = dict(bench=[D / 2 + 0.05, W + 0.05, 0.015], bench_body=[D / 2, W, BZ / 2 - 0.02],
                     wall_l=[D / 2, 0.012, H / 2], wall_r=[D / 2, 0.012, H / 2],
                     back=[0.012, W, H / 2], top=[D / 2, W, 0.012],
                     sash=[0.012, W, 0.028], target=[0.04, 0.04, 0.045])
        # mocap slab sizes are fixed at compile; we only POSE them. Approximate size by centering
        # the fixed slab at the variant's surface plane (sizes above only document intent).
        pos = dict(bench=[X0 + D / 2, 0, BZ - 0.015], bench_body=[X0 + D / 2, 0, BZ / 2 - 0.02 - 0.27 + (BZ / 2 - 0.02)],
                   wall_l=[X0 + D / 2, W + 0.012, BZ + 0.45], wall_r=[X0 + D / 2, -W - 0.012, BZ + 0.45],
                   back=[X0 + D + 0.012, 0, BZ + 0.45], top=[X0 + D / 2, 0, BZ + H + 0.012],
                   sash=[X0, 0, BZ + SASH + 0.028])
        for nm, p in pos.items():
            mid = int(model.body_mocapid[model.body(f"m_{nm}").id])
            data.mocap_pos[mid] = p
        omid = int(model.body_mocapid[model.body("obj_mount").id])
        data.mocap_pos[omid] = [X0 + 0.7 * D, 0, BZ + mk.obj_zhalf]
        mujoco.mj_forward(model, data)

    # mesh + channels
    fx.BASE_Z = 0.0
    mesh_model = model
    body_meshes = fx.extract_body_meshes(mesh_model)
    mesh_update = fx.robot_mesh_scene_update(body_meshes, Timestamp(sec=0, nsec=0))
    pub_bodies = sorted({model.body(i).name for i in range(model.nbody)
                         if model.body(i).name and model.body(i).name != "world"})

    ctx = foxglove.Context()
    writer = foxglove.open_mcap(str(outp), allow_overwrite=True, context=ctx)
    tf_ch = C.FrameTransformsChannel("/tf", context=ctx)
    mesh_ch = C.SceneUpdateChannel("/robot", context=ctx)
    scene_ch = C.SceneUpdateChannel("/scene", context=ctx)
    pc_ch = C.PointCloudChannel("/proximity", context=ctx)
    img_ch = C.CompressedImageChannel("/camera/exo", context=ctx)
    mos_ch = C.CompressedImageChannel("/skin_mosaic", context=ctx)
    log_ch = C.LogChannel("/task", context=ctx)
    skin_ch = foxglove.Channel("/skin", message_encoding="json", context=ctx)
    mot_ch = foxglove.Channel("/motion", message_encoding="json", context=ctx)

    frame_i = 0
    for vi, (vname, dims, blurb, wps) in enumerate(VARIANTS):
        W, H, D, SASH = dims
        pose_hood(W, H, D, SASH)
        traj = interp(wps, NSEG)
        ns0 = int(frame_i * DT * 1e9)
        ts0 = Timestamp(sec=ns0 // 10**9, nsec=ns0 % 10**9)
        log_ch.log(Log(timestamp=ts0, level=LogLevel.Info, name="chapter",
                       message=f"[{vi+1}/6] {vname.upper()} — {blurb}"), log_time=ns0)
        scene_ch.log(scene_update(hood_boxes(W, H, D, SASH), ts0), log_time=ns0)
        for q in traj:
            ns = int(frame_i * DT * 1e9)
            ts = Timestamp(sec=ns // 10**9, nsec=ns % 10**9)
            set_pose(model, data, list(q))
            # tf + mesh
            tfs = []
            for bname in pub_bodies:
                bid = model.body(bname).id
                p, qt = data.xpos[bid], data.xquat[bid]
                tfs.append(FrameTransform(timestamp=ts, parent_frame_id="world", child_frame_id=bname,
                                          translation=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                                          rotation=Quaternion(x=float(qt[1]), y=float(qt[2]),
                                                              z=float(qt[3]), w=float(qt[0]))))
            tf_ch.log(FrameTransforms(transforms=tfs), log_time=ns)
            if frame_i == 0:
                mesh_ch.log(mesh_update, log_time=ns)
            # skin cloud + stats
            pts, dd, mins = skin_cloud(model, data, rd)
            if len(pts):
                pc_ch.log(fx.pack_cloud(pts, fx.turbo_rgba(dd, NEAR, FAR), ts), log_time=ns)
            else:
                pc_ch.log(fx.pack_cloud(np.zeros((0, 3)), np.zeros((0, 4), np.uint8), ts), log_time=ns)
            link_min = {}
            for n, v in mins.items():
                lk = n.split("_sensor_")[0].split("_")[0]
                link_min[lk] = min(link_min.get(lk, 9.9), v)
            hand = data.xpos[model.body("gripper/base").id]
            skin_ch.log({**{k: round(v, 4) for k, v in link_min.items()},
                         "min": round(min(mins.values()), 4),
                         "active": sum(1 for v in mins.values() if v < FAR)}, log_time=ns)
            mot_ch.log({"depth_cm": round((float(hand[0]) - X0) * 100, 1),
                        "variant": vi + 1}, log_time=ns)
            # exo rgb + mosaic
            rgb.update_scene(data, "exo_camera_1")
            img = rgb.render()
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                img_ch.log(CompressedImage(timestamp=ts, frame_id="exo", data=buf.tobytes(),
                                           format="jpeg"), log_time=ns)
            mo = mosaic(model, data, rd, names)
            ok, buf = cv2.imencode(".jpg", mo, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                mos_ch.log(CompressedImage(timestamp=ts, frame_id="mosaic", data=buf.tobytes(),
                                           format="jpeg"), log_time=ns)
            frame_i += 1
    writer.close()
    print(f"{frame_i} frames, 6 chapters -> {outp}")


if __name__ == "__main__":
    main()
