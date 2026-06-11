"""Reusable 'reach-into-a-cavity' furniture geometry for proximity-necessity data.

Builds a cabinet/drawer cavity (a box open on one face) sized so that a Franka arm
reaching in gets its 29 SPAD proximity sensors encapsulated by nearby walls. Exposes:

  cavity_body_xml(...)            -> an MJCF <body> string for the cavity (inject into any scene)
  build_standalone_cavity_scene  -> a compiled MjModel (floor + cavity + franka_skin) for testing
  encapsulation_report(model,..) -> proximity-depth stats proving the sensors are exercised

The cavity opens toward -x (the robot, mounted at the origin, reaches in along +x). Name the
cavity body with a low-surface prefix (e.g. "chestofdrawers_cavity") so the existing
PickAndPlaceTaskSampler source_surface_types filter treats objects resting in it as valid pickups.
"""
from __future__ import annotations
import tempfile
import numpy as np
import mujoco

ROBOT_XML = "/home/jaydv/code/prox_learning/assets/robots/franka_skin/model.xml"
NS = "robot_0/"
BASE_Z = 0.58


def _box(name, pos, size, rgba):
    return (f'<geom name="{name}" type="box" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
            f'size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}" rgba="{rgba}"/>')


def cavity_body_xml(center, interior, thickness=0.02, name="chestofdrawers_cavity",
                    open_top=False, rgba="0.55 0.38 0.22 1") -> str:
    """MJCF <body> for a 5- or 6-walled cavity open on the -x face (toward the robot).

    center: (x,y,z) of the cavity interior center, world meters.
    interior: (ix,iy,iz) interior HALF-extents.
    open_top: if True omit the top wall (an open drawer you reach down/into).
    """
    ix, iy, iz = interior
    t = thickness
    cx, cy, cz = center
    geoms = [
        _box("cav_bottom", (0, 0, -(iz + t)), (ix + 2 * t, iy + 2 * t, t), rgba),
        _box("cav_back", ((ix + t), 0, 0), (t, iy + 2 * t, iz), rgba),
        _box("cav_left", (0, (iy + t), 0), (ix + 2 * t, t, iz), rgba),
        _box("cav_right", (0, -(iy + t), 0), (ix + 2 * t, t, iz), rgba),
    ]
    if not open_top:
        geoms.append(_box("cav_top", (0, 0, (iz + t)), (ix + 2 * t, iy + 2 * t, t), rgba))
    return f'<body name="{name}" pos="{cx:.4f} {cy:.4f} {cz:.4f}">{"".join(geoms)}</body>'


def cavity_around_arm(reach_pose, pad=0.11, front_x=0.20, thickness=0.02,
                      name="chestofdrawers_cavity", open_top=False):
    """Compute a cavity that wraps the arm's sensors for a given reach pose.

    Returns (body_xml, info) where info has center/interior/support_z (object rest height)."""
    m0 = _robot_only()
    d0 = mujoco.MjData(m0)
    _set_arm(m0, d0, reach_pose)
    sensors = [m0.camera(i).name for i in range(m0.ncam) if "_sensor_" in m0.camera(i).name]
    spos = np.array([d0.cam_xpos[m0.camera(s).id].copy() for s in sensors])
    hand = d0.xpos[m0.body(f"{NS}fr3_link7").id].copy()
    back_x = hand[0] + pad
    ix = 0.5 * (back_x - front_x)
    cx = 0.5 * (front_x + back_x)
    cz = 0.5 * (spos[:, 2].min() + spos[:, 2].max())
    iz = 0.5 * (spos[:, 2].max() - spos[:, 2].min()) + pad
    iy = max(0.18, 0.5 * (spos[:, 1].max() - spos[:, 1].min())) + pad
    info = dict(center=(cx, 0.0, cz), interior=(ix, iy, iz),
                support_z=cz - iz + thickness, hand=hand.tolist())
    return cavity_body_xml((cx, 0.0, cz), (ix, iy, iz), thickness, name, open_top), info


# ----------------------- standalone testing helpers ----------------------- #
def _write_temp(xml):
    fd = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
    fd.write(xml)
    fd.close()
    return fd.name


def _attach_robot(spec):
    rs = mujoco.MjSpec.from_file(ROBOT_XML)
    base = spec.worldbody.add_body(name=f"{NS}base", pos=[0, 0, 0], quat=[1, 0, 0, 0], mocap=True)
    base.add_frame(pos=[0, 0, BASE_Z]).attach_body(rs.worldbody.first_body(), NS, "")


def _robot_only():
    spec = mujoco.MjSpec.from_file(_write_temp(
        '<mujoco><compiler angle="radian"/><option gravity="0 0 -9.8" integrator="implicitfast"/>'
        '<worldbody><light pos="0 0 5" directional="true"/></worldbody></mujoco>'))
    _attach_robot(spec)
    return spec.compile()


def _set_arm(model, data, q):
    for i, v in enumerate(q):
        data.qpos[model.joint(f"{NS}fr3_joint{i+1}").qposadr[0]] = float(v)
    mujoco.mj_forward(model, data)


def build_standalone_cavity_scene(reach_pose, open_top=False, with_target=True):
    body_xml, info = cavity_around_arm(reach_pose, open_top=open_top)
    ix, iy, iz = info["interior"]
    tgt = ""
    if with_target:
        tgt_local = (ix - 0.06, 0, -iz + 0.03)
        tgt = (f'<geom name="target" type="box" pos="{tgt_local[0]:.3f} 0 {tgt_local[2]:.3f}" '
               f'size="0.025 0.025 0.03" rgba="0.85 0.15 0.15 1"/>')
        body_xml = body_xml[:-7] + tgt + "</body>"  # splice target into the cavity body
    xml = (f'<mujoco model="cavity"><compiler angle="radian"/>'
           f'<option gravity="0 0 -9.8" integrator="implicitfast"/>'
           f'<visual><global offwidth="1600" offheight="1200"/><map znear="0.005" zfar="20"/>'
           f'<headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5"/></visual>'
           f'<asset><texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.3 0.35" '
           f'rgb2="0.22 0.22 0.27" width="300" height="300"/>'
           f'<material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/></asset>'
           f'<worldbody><light pos="1 1 3" directional="true"/>'
           f'<geom name="floor" type="plane" size="4 4 0.05" pos="0 0 0" material="grid"/>'
           f'{body_xml}</worldbody></mujoco>')
    spec = mujoco.MjSpec.from_file(_write_temp(xml))
    _attach_robot(spec)
    model = spec.compile()
    return model, info


def render_proximity_depths(model, data):
    """8x8 native depth for every proximity sensor (skin meshes hidden)."""
    r = mujoco.Renderer(model, height=8, width=8)
    r.enable_depth_rendering()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[2] = 0
    out = {}
    for i in range(model.ncam):
        nm = model.camera(i).name
        if "_sensor_" in nm:
            r.update_scene(data, camera=nm, scene_option=opt)
            out[nm] = r.render().astype(np.float32).copy()
    r.close()
    return out


def encapsulation_report(model, data, near=0.03, far=3.0):
    depths = render_proximity_depths(model, data)
    alld = []
    n_active = n_close = 0
    for nm, dep in depths.items():
        valid = (dep > near) & (dep < far)
        if valid.any():
            n_active += 1
            alld.append(dep[valid])
            if (dep[valid] < 0.30).any():
                n_close += 1
    D = np.concatenate(alld) if alld else np.zeros(0)
    return dict(
        n_returns=int(len(D)),
        median=float(np.median(D)) if len(D) else None,
        frac_within_30cm=float((D <= 0.30).mean()) if len(D) else 0.0,
        n_sensors_active=n_active,
        n_sensors_close=n_close,
    )


if __name__ == "__main__":
    REACH = [0.0, 0.55, 0.0, -1.4, 0.0, 1.95, 0.0]
    model, info = build_standalone_cavity_scene(REACH)
    data = mujoco.MjData(model)
    _set_arm(model, data, REACH)
    print("cavity info:", {k: (np.round(v, 3).tolist() if isinstance(v, (list, tuple, np.ndarray)) else v)
                           for k, v in info.items()})
    print("encapsulation:", encapsulation_report(model, data))
