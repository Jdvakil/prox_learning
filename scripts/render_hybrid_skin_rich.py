"""Rich display test for the hybrid skin: the arm reaches INSIDE a tight cavity (like the
task scenes), so most of the 40 SPADs return structure — plus the INFERENCE view: the skin's
back-projected point cloud painting the surrounding geometry (what the policy can "see"
without cameras).

Sheet layout:
  [ 3D scene | 3D + skin point cloud overlay ]
  [   exo    |     wrist                     ]
  [ 40 sensor tiles, labeled, active count in banner ]
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import OpenGL.EGL as _EGL  # noqa: E402
_d = _EGL.eglGetDisplay(_EGL.EGL_DEFAULT_DISPLAY)
_maj, _min = _EGL.EGLint(), _EGL.EGLint()
if _EGL.eglInitialize(_d, _maj, _min):
    import mujoco.egl as _me
    _me.EGL_DISPLAY = _d

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402

ROBOT = Path("/home/jaydv/code/prox_learning/assets/robots/franka_skin/model_hybrid.xml")
OUT = Path("/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_skin_rich")
OUT.mkdir(parents=True, exist_ok=True)
TURBO = colormaps["turbo"]
FOVY = 45.0
NEAR, FAR = 0.015, 0.50          # display range: what the skin story cares about

QPOS = [0.0, -0.35, 0.0, -2.30, 0.0, 2.05, 0.79]   # reaching into the cavity

# tight cavity around the reaching arm (robot base at origin, z up)
CAVITY = [
    # name, center, half, rgba (walls translucent so the 3D view shows the arm + cloud inside)
    ("bench",   [0.52, 0.0, 0.175], [0.30, 0.34, 0.175], [0.62, 0.55, 0.45, 1]),
    ("wall_l",  [0.52, 0.30, 0.62], [0.30, 0.015, 0.28], [0.75, 0.72, 0.66, 0.30]),
    ("wall_r",  [0.52, -0.30, 0.62], [0.30, 0.015, 0.28], [0.75, 0.72, 0.66, 0.30]),
    ("roof",    [0.52, 0.0, 0.92], [0.30, 0.315, 0.015], [0.70, 0.68, 0.62, 0.30]),
    ("back",    [0.83, 0.0, 0.62], [0.015, 0.315, 0.28], [0.68, 0.66, 0.60, 1]),
    ("pillar",  [0.40, 0.13, 0.62], [0.025, 0.025, 0.27], [0.48, 0.34, 0.22, 1]),
]


def cavity_spec(spec):
    spec.visual.global_.offwidth = 1600
    spec.visual.global_.offheight = 1280
    fl = spec.worldbody.add_geom()
    fl.type = mujoco.mjtGeom.mjGEOM_PLANE
    fl.size = [3, 3, 0.1]
    fl.rgba = [0.30, 0.31, 0.34, 1]
    for nm, c, h, rgba in CAVITY:
        b = spec.worldbody.add_body(name=f"cav_{nm}", pos=c)
        g = b.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = h
        g.rgba = rgba
    spec.worldbody.add_light(pos=[0.3, 0.6, 2.2], dir=[0, -0.2, -1],
                             diffuse=[0.85, 0.85, 0.85], specular=[0.2, 0.2, 0.2])
    spec.worldbody.add_light(pos=[-0.8, -0.8, 1.8], dir=[0.5, 0.5, -1],
                             diffuse=[0.45, 0.45, 0.45], specular=[0.1, 0.1, 0.1])
    # exo camera
    exo = spec.worldbody.add_camera()
    exo.name = "exo_camera_1"
    exo.pos = [1.45, -1.25, 1.15]
    v = np.array([0.4, 0.0, 0.55]) - np.array(exo.pos); v /= np.linalg.norm(v)  # look at robot
    z = -v
    up = np.array([0, 0, 1.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).reshape(9))
    exo.quat = [float(t) for t in q]
    exo.fovy = 52
    exo.resolution = [480, 640]
    return spec


def set_pose(model, data):
    for i, v in enumerate(QPOS, start=1):
        data.qpos[model.joint(f"fr3_joint{i}").qposadr[0]] = v
    mujoco.mj_forward(model, data)


def backproject(d8, pos, Rm, d_min=NEAR, d_max=FAR):
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
    pts = (Rm @ np.stack([x_c, y_c, -d], 0)).T + pos
    return pts, d


def main():
    spec = cavity_spec(mujoco.MjSpec.from_file(str(ROBOT)))
    model = spec.compile()
    data = mujoco.MjData(model)
    set_pose(model, data)

    names = [model.camera(i).name for i in range(model.ncam) if "_sensor_" in model.camera(i).name]

    # ---- sensor depth + cloud ----
    rd = mujoco.Renderer(model, 8, 8)
    rd.enable_depth_rendering()
    tiles, mins = {}, {}
    pts_all, d_all = [], []
    for n in names:
        rd.update_scene(data, n)
        d8 = rd.render().copy()
        tiles[n] = d8
        mins[n] = float(d8.min())
        cid = model.camera(n).id
        pts, dd = backproject(d8, data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3))
        if len(pts):
            pts_all.append(pts)
            d_all.append(dd)
    pts = np.concatenate(pts_all) if pts_all else np.zeros((0, 3))
    dd = np.concatenate(d_all) if d_all else np.zeros((0,))
    active = sum(1 for n in names if mins[n] < FAR)

    # ---- 3D renders (plain + cloud overlay) ----
    cam = mujoco.MjvCamera()
    cam.lookat = [0.40, 0.0, 0.55]
    cam.distance = 1.6
    cam.azimuth = 40
    cam.elevation = -16
    r_big = mujoco.Renderer(model, 640, 760)
    r_big.update_scene(data, cam)
    plain = r_big.render().copy()

    # cloud overlay: same scene + colored sphere per point via mjv user scene geoms
    r_big.update_scene(data, cam)
    scn = r_big.scene
    norm = np.clip((dd - NEAR) / (FAR - NEAR), 0, 1)
    cols = TURBO(1.0 - norm)[:, :3]
    for p, c in zip(pts, cols):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.008, 0, 0]), np.asarray(p, np.float64),
                            np.eye(3).ravel(), np.array([c[0], c[1], c[2], 1.0], np.float32))
        scn.ngeom += 1
    overlay = r_big.render().copy()

    r_side = mujoco.Renderer(model, 320, 380)
    r_side.update_scene(data, "exo_camera_1")
    exo = r_side.render().copy()
    r_side.update_scene(data, "gripper/wrist_camera")
    wrist = r_side.render().copy()

    # ---- tile grid ----
    full_w = 760 * 2
    cols_n = 10
    pad = 6
    T = (full_w - (cols_n + 1) * pad) // cols_n
    rows = int(np.ceil(len(names) / cols_n))
    grid = np.full((rows * (T + 20) + pad, full_w, 3), 22, np.uint8)
    for k, n in enumerate(sorted(names)):
        d8 = tiles[n]
        nrm = np.clip((d8 - NEAR) / (FAR - NEAR), 0, 1)
        img = (TURBO(1.0 - nrm)[:, :, :3] * 255).astype(np.uint8)
        img[d8 > FAR] = (16, 16, 16)
        til = cv2.resize(img[..., ::-1], (T, T), interpolation=cv2.INTER_NEAREST)
        rr, cc = divmod(k, cols_n)
        y0 = rr * (T + 20) + 20
        x0 = pad + cc * (T + pad)
        grid[y0:y0 + T, x0:x0 + T] = til
        cv2.rectangle(grid, (x0, y0), (x0 + T, y0 + T), (70, 70, 70), 1)
        short = n.replace("link", "L").replace("_sensor_", ".").replace("_front", "F").replace("_back", "B")
        lab = f"{short} {mins[n]:.2f}m" if mins[n] < FAR else f"{short} --"
        col = (60, 60, 255) if mins[n] < 0.08 else (210, 210, 210)
        cv2.putText(grid, lab, (x0 + 1, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)

    # ---- compose ----
    def bgr(x):
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    row1 = np.concatenate([bgr(plain), bgr(overlay)], axis=1)
    side = np.concatenate([bgr(exo), bgr(wrist)], axis=1)
    side = cv2.resize(side, (row1.shape[1], int(side.shape[0] * row1.shape[1] / side.shape[1])))
    banner = np.full((40, row1.shape[1], 3), 30, np.uint8)
    cv2.putText(banner,
                f"fr3_hybrid_skin in tight cavity  |  {active}/40 sensors returning <{FAR:.1f}m  |  "
                f"right 3D = skin point cloud (what the policy senses, no cameras)",
                (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
    full = np.concatenate([banner, row1, side, grid], axis=0)
    out = OUT / "hybrid_skin_rich_test.png"
    cv2.imwrite(str(out), full)
    print(f"active sensors: {active}/40, cloud pts: {len(pts)} -> {out}")


if __name__ == "__main__":
    main()
