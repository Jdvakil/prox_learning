"""Shared helpers for hybrid-skin visualizations. Import this; it sets up EGL and gives you
model building, depth rendering, and back-projection so every viz panel is consistent.

    from hybrid_viz_lib import (ROBOT, FOVY, SENSORS, build, set_pose, depth8,
                                backproject, cam_pose, add_box, add_plane_mocap, STYLE)
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

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROBOT = Path("/home/jaydv/code/prox_learning/assets/robots/franka_skin/model_hybrid.xml")
FOVY = 45.0
NEAR, FAR = 0.015, 0.50
RES = 8

# a few natural reaching poses
POSES = {
    "reach": [0.0, -0.35, 0.0, -2.30, 0.0, 2.05, 0.79],
    "open": [0.0, -0.5, 0.0, -2.1, 0.0, 1.9, 0.8],
    "up": [0.0, 0.3, 0.0, -1.2, 0.0, 1.6, 0.0],
}

STYLE = dict(
    bg="#111317", panel="#171a20", fg="#e8e8ea", grid="#2a2e36",
    accent="#4cc9f0", near="#ef476f", far="#1d3557", cmap="turbo_r",
)


def _q_lookalong(direction):
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    z = -d
    up = np.array([0, 0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0, 0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).reshape(9))
    return [float(v) for v in q]


def add_box(spec, name, center, half, rgba, static=True):
    b = spec.worldbody.add_body(name=name, pos=center, mocap=not static)
    g = b.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = half
    g.rgba = rgba
    g.contype = 0
    g.conaffinity = 0
    return name


def add_plane_mocap(spec, name="probe_plane", half=(0.12, 0.12, 0.004), rgba=(0.2, 0.85, 0.4, 1)):
    b = spec.worldbody.add_body(name=name, mocap=True, pos=[0, 0, -9])
    g = b.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = list(half)
    g.rgba = list(rgba)
    g.contype = 0
    g.conaffinity = 0
    return name


def add_sphere_mocap(spec, name="probe_sphere", radius=0.06, rgba=(0.95, 0.6, 0.2, 1)):
    b = spec.worldbody.add_body(name=name, mocap=True, pos=[0, 0, -9])
    g = b.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [radius, 0, 0]
    g.rgba = list(rgba)
    g.contype = 0
    g.conaffinity = 0
    return name


def build(make=None, offw=1280, offh=1024):
    """make(spec) lets a caller add boxes / planes / lights. Returns compiled model."""
    spec = mujoco.MjSpec.from_file(str(ROBOT))
    spec.visual.global_.offwidth = offw
    spec.visual.global_.offheight = offh
    if make is not None:
        make(spec)
    return spec.compile()


def sensors(model):
    return sorted(model.camera(i).name for i in range(model.ncam) if "_sensor_" in model.camera(i).name)


SENSORS = None  # filled lazily after first build


def set_pose(model, data, qpos):
    q = POSES[qpos] if isinstance(qpos, str) else qpos
    for i, v in enumerate(q, start=1):
        data.qpos[model.joint(f"fr3_joint{i}").qposadr[0]] = v
    mujoco.mj_forward(model, data)


def cam_pose(model, data, name):
    cid = model.camera(name).id
    return data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3).copy()


# proximity renders hide geom group 2 (cosmetic skin + robot visual meshes) — the DATAGEN
# convention (env.record_proximity_depths). Without this, link5-front sensors stare at their
# own dermis/arm at a constant 3.5-5.8cm and never vary.
_PROX_OPT = mujoco.MjvOption()
_PROX_OPT.geomgroup[2] = 0


def depth8(renderer, data, name):
    renderer.update_scene(data, name, scene_option=_PROX_OPT)
    return renderer.render().copy()


def mocap_set(model, data, name, pos, view_dir=None):
    mid = int(model.body_mocapid[model.body(name).id])
    data.mocap_pos[mid] = np.asarray(pos, float)
    if view_dir is not None:
        # orient so the thin (z) axis faces back along view_dir (plane faces the sensor)
        data.mocap_quat[mid] = _q_lookalong(view_dir)
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


def depth_renderer(model):
    r = mujoco.Renderer(model, RES, RES)
    r.enable_depth_rendering()
    return r


def fit_plane(pts):
    """return (centroid, normal, rms) of best-fit plane through points."""
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c)
    n = vt[-1]
    rms = float(np.sqrt(((pts - c) @ n) ** 2).mean()) if len(pts) else float("nan")
    return c, n, rms


# --------------------------------------------------------------------------------------------
# Scene-building + rendering helpers for environment/proof viz (added 2026-06-11)
# --------------------------------------------------------------------------------------------
from matplotlib import colormaps as _cmaps  # noqa: E402
_TURBO = _cmaps["turbo"]


def add_cylinder(spec, name, center, radius, halflen, rgba, quat=(1, 0, 0, 0)):
    b = spec.worldbody.add_body(name=name, pos=center, quat=list(quat))
    g = b.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size = [radius, halflen, 0]
    g.rgba = list(rgba)
    g.contype = 0
    g.conaffinity = 0
    return name


def add_capsule(spec, name, center, radius, halflen, rgba, quat=(1, 0, 0, 0)):
    b = spec.worldbody.add_body(name=name, pos=center, quat=list(quat))
    g = b.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    g.size = [radius, halflen, 0]
    g.rgba = list(rgba)
    g.contype = 0
    g.conaffinity = 0
    return name


def nice_lights(spec, floor=True):
    spec.worldbody.add_light(pos=[0.4, 0.6, 2.3], dir=[-0.1, -0.2, -1],
                             diffuse=[0.9, 0.9, 0.9], specular=[0.25, 0.25, 0.25])
    spec.worldbody.add_light(pos=[-0.9, -0.7, 1.9], dir=[0.5, 0.4, -1],
                             diffuse=[0.4, 0.42, 0.5], specular=[0.1, 0.1, 0.1])
    if floor:
        fl = spec.worldbody.add_geom()
        fl.type = mujoco.mjtGeom.mjGEOM_PLANE
        fl.size = [3, 3, 0.1]
        fl.rgba = [0.27, 0.28, 0.31, 1]


def mjv_cam(lookat=(0.45, 0.0, 0.55), distance=1.7, azimuth=140, elevation=-15):
    c = mujoco.MjvCamera()
    c.lookat = list(lookat)
    c.distance = distance
    c.azimuth = azimuth
    c.elevation = elevation
    return c


def skin_cloud(model, data, rd=None):
    """Render all 40 sensors, back-project. Returns (pts Nx3, depths N, mins {name:min_m})."""
    own = rd is None
    if own:
        rd = depth_renderer(model)
    names = sorted(model.camera(i).name for i in range(model.ncam) if "_sensor_" in model.camera(i).name)
    P, D, mins = [], [], {}
    for n in names:
        d8 = depth8(rd, data, n)
        mins[n] = float(d8.min())
        cid = model.camera(n).id
        pts, dd = backproject(d8, data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3))
        if len(pts):
            P.append(pts)
            D.append(dd)
    pts = np.concatenate(P) if P else np.zeros((0, 3))
    dd = np.concatenate(D) if D else np.zeros((0,))
    return pts, dd, mins


def render_scene(model, data, cam, w=820, h=720, cloud=None, depths=None,
                 pt_size=0.0075, gamma=0.85):
    """RGB render of the scene; if cloud given, overlay distance-colored dots (turbo_r near=red)."""
    r = mujoco.Renderer(model, h, w)
    r.update_scene(data, cam)
    if cloud is not None and len(cloud):
        scn = r.scene
        if depths is not None:
            nrm = np.clip((depths - NEAR) / (FAR - NEAR), 0, 1)
            cols = _TURBO(1.0 - nrm)[:, :3]
        else:
            cols = np.tile([1.0, 0.3, 0.2], (len(cloud), 1))
        for p, c in zip(cloud, cols):
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([pt_size, 0, 0]), np.asarray(p, np.float64),
                                np.eye(3).ravel(), np.array([c[0], c[1], c[2], 1.0], np.float32))
            scn.ngeom += 1
    img = r.render().copy()
    if gamma != 1.0:
        img = (np.clip((img.astype(np.float32) / 255) ** gamma * 1.08, 0, 1) * 255).astype(np.uint8)
    return img
