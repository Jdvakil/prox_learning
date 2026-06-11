"""One-image display test for the converted fr3_hybrid_skin MuJoCo model:
  - big 3D RGB view of the arm + hybrid skin + magenta sensor markers
  - the exo camera view and the wrist camera view
  - a grid of all 40 SPAD sensors' 8x8 depth returns (turbo: red=near, blue=far)
A box obstacle is placed near the forearm so the proximity sensors actually return something.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
# EGL device-platform enumeration is wedged on this box (mujoco.egl EGL_DISPLAY=None) while
# the DEFAULT display works — pre-initialize it and patch mujoco.egl before Renderer use.
import OpenGL.EGL as _EGL  # noqa: E402
_d = _EGL.eglGetDisplay(_EGL.EGL_DEFAULT_DISPLAY)
_maj, _min = _EGL.EGLint(), _EGL.EGLint()
if _EGL.eglInitialize(_d, _maj, _min):
    import mujoco.egl as _me
    _me.EGL_DISPLAY = _d
import cv2
import mujoco
import numpy as np
from matplotlib import colormaps

ROBOT = Path("/home/jaydv/code/prox_learning/assets/robots/franka_skin/model_hybrid.xml")
OUT = Path("/home/jaydv/code/prox_learning/diagnostics_output/20260610_hybrid_skin_viz")
OUT.mkdir(parents=True, exist_ok=True)
TURBO = colormaps["turbo"]

# reaching pose so the arm is open and the skin faces the obstacle
QPOS = [0.0, -0.5, 0.0, -2.1, 0.0, 1.9, 0.8]


def build():
    spec = mujoco.MjSpec.from_file(str(ROBOT))
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 1280
    # demo obstacle: a panel beside the forearm + a low table the gripper hovers over
    wall = spec.worldbody.add_body()
    wall.name = "demo_wall"
    wall.pos = [0.55, 0.30, 0.55]
    g = wall.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [0.22, 0.02, 0.35]
    g.rgba = [0.80, 0.55, 0.30, 1.0]
    tb = spec.worldbody.add_body()
    tb.name = "demo_table"
    tb.pos = [0.55, 0.0, 0.32]
    g2 = tb.add_geom()
    g2.type = mujoco.mjtGeom.mjGEOM_BOX
    g2.size = [0.20, 0.30, 0.02]
    g2.rgba = [0.55, 0.57, 0.60, 1.0]
    # exo camera (the robot file only ships wrist cams; datagen adds exo at env level)
    exo = spec.worldbody.add_camera()
    exo.name = "exo_camera_1"
    exo.pos = [1.45, -1.25, 1.15]
    v = np.array([0.35, 0.0, 0.5]) - np.array(exo.pos); v = v / np.linalg.norm(v)  # look at robot
    z = -v
    up = np.array([0, 0, 1.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], axis=1).reshape(9))
    exo.quat = [float(t) for t in q]
    exo.fovy = 52
    exo.resolution = [480, 640]
    return spec.compile()


def render_rgb(rndr, data, cam_name=None, free=None):
    if free is not None:
        rndr.update_scene(data, free)
    else:
        rndr.update_scene(data, cam_name)
    return rndr.render().copy()


def tile(d8, near=0.02, far=0.5):
    valid = d8 < (far * 4)
    norm = np.clip((d8 - near) / (far - near), 0, 1)
    img = (TURBO(1.0 - norm)[:, :, :3] * 255).astype(np.uint8)
    img[~valid] = (15, 15, 15)
    return img


def main():
    model = build()
    data = mujoco.MjData(model)
    # set the 7 arm joints
    arm = [model.joint(f"fr3_joint{i}").qposadr[0] for i in range(1, 8)
           if any(model.joint(j).name == f"fr3_joint{i}" for j in range(model.njnt))]
    if not arm:  # joint names may be unprefixed
        arm = [model.joint(i).qposadr[0] for i in range(min(7, model.njnt))]
    for adr, v in zip(arm, QPOS):
        data.qpos[adr] = v
    mujoco.mj_forward(model, data)

    # 3D free view (shared renderers per resolution)
    cam = mujoco.MjvCamera()
    cam.lookat = [0.35, 0.0, 0.6]
    cam.distance = 1.85
    cam.azimuth = 145
    cam.elevation = -12
    r_big = mujoco.Renderer(model, 900, 760)
    main3d = render_rgb(r_big, data, free=cam)
    try:
        r_big.close()
    except Exception:
        pass

    r_side = mujoco.Renderer(model, 450, 600)   # two stacked -> 900 tall, matches main3d
    exo = render_rgb(r_side, data, "exo_camera_1").copy()
    wrist = render_rgb(r_side, data, "gripper/wrist_camera").copy()
    try:
        r_side.close()
    except Exception:
        pass

    # all 40 sensor depth tiles (ONE shared 8x8 depth renderer)
    sensor_cams = [model.camera(i).name for i in range(model.ncam)
                   if "_sensor_" in model.camera(i).name]
    r_d = mujoco.Renderer(model, 8, 8)
    r_d.enable_depth_rendering()
    full_w = 760 + 600
    cols = 10
    rows = int(np.ceil(len(sensor_cams) / cols))
    pad = 6
    T = (full_w - (cols + 1) * pad) // cols
    grid = np.full((rows * (T + 18) + pad, full_w, 3), 22, np.uint8)
    for k, name in enumerate(sensor_cams):
        r_d.update_scene(data, name)
        d8 = r_d.render().copy()
        til = cv2.resize(tile(d8)[..., ::-1], (T, T), interpolation=cv2.INTER_NEAREST)
        rr, cc = divmod(k, cols)
        y0 = rr * (T + 18) + 18
        x0 = pad + cc * (T + pad)
        grid[y0 : y0 + T, x0 : x0 + T] = til
        cv2.rectangle(grid, (x0, y0), (x0 + T, y0 + T), (70, 70, 70), 1)
        short = name.replace("link", "L").replace("_sensor_", ".").replace("_front", "F").replace("_back", "B")
        cv2.putText(grid, short, (x0 + 1, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (215, 215, 215), 1, cv2.LINE_AA)
    try:
        r_d.close()
    except Exception:
        pass

    # compose: [ 3D | (exo over wrist) ] on top, sensor grid below
    def bgr(x):
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    right = np.concatenate([bgr(exo), bgr(wrist)], axis=0)
    top = np.concatenate([bgr(main3d), right], axis=1)
    gw = top.shape[1]
    if grid.shape[1] < gw:
        grid = np.pad(grid, ((0, 0), (0, gw - grid.shape[1]), (0, 0)), constant_values=22)
    else:
        grid = grid[:, :gw]
    # labels banner
    banner = np.full((34, gw, 3), 30, np.uint8)
    cv2.putText(banner, "fr3_hybrid_skin  |  3D + exo + wrist + 40 SPAD depth sensors (turbo: red=near)",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    full = np.concatenate([banner, top, grid], axis=0)
    out = OUT / "hybrid_skin_display_test.png"
    cv2.imwrite(str(out), full)
    print(f"ncam={model.ncam} sensors={len(sensor_cams)} -> {out}")


if __name__ == "__main__":
    main()
