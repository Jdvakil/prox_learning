"""Build the HYBRID-skin FR3 on top of the PROVEN franka_skin model (the exact robot used in
all prior datagen — real FR3 visual meshes, materials, robotiq gripper, wrist/exo cameras).

Only the skin changes:
  - delete the old 29-sensor skin bodies (link2/3/5/6_skin and everything inside them)
  - add the 7 gentact hybrid dermis meshes (mm STLs, scale 0.001) on links 1-6 at the URDF
    skin-joint origins
  - add the 40 hybrid sensors: red marker site + fovy=45 8x8 depth camera per sensor, placed
    at the URDF sensor-joint origin, viewing along the joint <axis> (the outward normal)

Output: assets/robots/franka_skin/model_hybrid.xml  (lives next to model.xml so every
relative mesh/texture path keeps working).
"""
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path("/home/jaydv/code/prox_learning")
FS = ROOT / "assets/robots/franka_skin"
URDF = ROOT / "assets/urdf/fr3_hybrid_skin.urdf"
DERMIS_SRC = ROOT / "assets/robots/fr3_hybrid_skin/meshes/skin"
OUT = FS / "model_hybrid.xml"

OLD_SKIN_BODIES = ["link2_skin", "link3_skin", "link5_skin", "link6_skin"]
SKIN_RGBA = [0.27, 0.46, 0.85, 1.0]      # the blue of the reference photo
DOT_RGBA = [0.90, 0.12, 0.12, 1.0]       # red sensor dots


def rpy_quat(rpy: list[float]) -> list[float]:
    q = R.from_euler("xyz", rpy).as_quat(scalar_first=True)
    return [float(v) for v in q]


def quat_view_along(direction) -> list[float]:
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    z = -d
    up = np.array([0, 0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0, 0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], axis=1).reshape(9))
    return [float(v) for v in q]


def parse_urdf():
    """skins: child -> (parent, xyz, rpy, mesh, scale); sensors: list of dicts."""
    tree = ET.parse(URDF)
    root = tree.getroot()
    mesh_of_link = {}
    for link in root.findall("link"):
        m = link.find("visual/geometry/mesh")
        if m is not None and "skin/hybrid/" in (m.get("filename") or ""):
            mesh_of_link[link.get("name")] = (
                m.get("filename").split("/")[-1],
                [float(v) for v in (m.get("scale") or "1 1 1").split()],
            )
    skins, sensors = {}, []
    for j in root.findall("joint"):
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()]
        if child.endswith("_skin"):
            mesh, scale = mesh_of_link.get(child, (None, None))
            skins[child] = dict(parent=parent, xyz=xyz, rpy=rpy, mesh=mesh, scale=scale)
        elif "_sensor_" in child:
            ax = j.find("axis")
            axis = [float(v) for v in (ax.get("xyz") if ax is not None else "0 0 1").split()]
            sensors.append(dict(name=child, parent=parent, xyz=xyz, rpy=rpy, axis=axis))
    return skins, sensors


def main():
    # stage dermis meshes next to the model's meshdir (assets/)
    for p in DERMIS_SRC.glob("*.stl"):
        shutil.copy(p, FS / "assets" / p.name)

    spec = mujoco.MjSpec.from_file(str(FS / "model.xml"))

    # 1) delete the old skin bodies (sensors + cameras inside go with them)
    for name in OLD_SKIN_BODIES:
        b = spec.body(name)
        if b is not None:
            spec.delete(b)

    skins, sensors = parse_urdf()

    # 2) dermis mesh assets
    added_assets = set()
    for child, s in skins.items():
        if s["mesh"] and s["mesh"] not in added_assets:
            spec.add_mesh(name=s["mesh"].replace(".stl", ""), file=s["mesh"], scale=s["scale"])
            added_assets.add(s["mesh"])

    # 3) new skin bodies on the arm links
    skin_bodies = {}
    for child, s in skins.items():
        parent = spec.body(s["parent"])
        if parent is None:
            print(f"WARN: no parent {s['parent']} for {child}")
            continue
        nb = parent.add_body(name=child, pos=s["xyz"], quat=rpy_quat(s["rpy"]))
        if s["mesh"]:
            g = nb.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = s["mesh"].replace(".stl", "")
            g.rgba = SKIN_RGBA
            g.contype = 0
            g.conaffinity = 0
            g.group = 2          # cosmetic skin — env hides group 2 during proximity render
            g.mass = 1e-6
        skin_bodies[child] = nb

    # 4a) add sensor SITES now; aim the cameras in a second pass via RAYCAST so no camera ever
    # stares into the robot. (URDF <axis> and mesh-normal heuristics each aimed several sensors
    # INTO the arm — verify_hybrid_skin_sensors.py / the near-field znear fix exposed them. The
    # robot geometry itself is the ground truth: cast rays out of each sensor and pick the
    # direction that is provably free of self-collision, biased toward the dermis surface normal.)
    import trimesh
    normals_of = {}
    for child, s in skins.items():
        if s["mesh"]:
            tm = trimesh.load(str(FS / "assets" / s["mesh"]), force="mesh")
            tm.apply_scale(s["scale"][0])
            normals_of[child] = tm
    for s in sensors:
        host = skin_bodies.get(s["parent"]) or spec.body(s["parent"])
        if host is None:
            print(f"WARN: no host for sensor {s['name']}")
            continue
        site = host.add_site(name=s["name"] + "_site", pos=s["xyz"])
        site.size = [0.0065, 0.0065, 0.0065]
        site.rgba = DOT_RGBA

    spec.visual.map.znear = 0.0002   # see note below
    probe = spec.compile()           # pass 1: kinematics only (no sensor cameras yet)

    CAM_OFFSET = 0.009   # push the camera off the skin so it clears the dermis shell

    # fibonacci sphere of candidate directions for the repair search
    K = 400
    gi = (1 + 5 ** 0.5) / 2
    ii = np.arange(K)
    phi = np.arccos(1 - 2 * (ii + 0.5) / K)
    th = 2 * np.pi * ii / gi
    CAND = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)

    pdata = mujoco.MjData(probe)
    for i, v in enumerate([0.0, -0.3, 0.0, -1.9, 0.0, 1.7, 0.4], start=1):
        pdata.qpos[probe.joint(f"fr3_joint{i}").qposadr[0]] = v   # mild splay -> low self-occlusion
    mujoco.mj_forward(probe, pdata)
    geomid = np.zeros(1, dtype=np.int32)

    def free_dist(wp, v):
        dd = mujoco.mj_ray(probe, pdata, (wp + 1e-3 * v).astype(np.float64),
                           v.astype(np.float64), None, 1, -1, geomid)
        return 9.9 if dd < 0 else dd

    aim_world = {}
    repaired = 0
    for s in sensors:
        sid = probe.site(s["name"] + "_site").id
        wp = pdata.site_xpos[sid]
        # radial out of the parent ARM LINK axis — outward + surface-perpendicular for the
        # cylindrical links; reliably free for 32/40 (mesh normals are inconsistently wound)
        fr3 = "fr3_link" + s["parent"].split("_")[0].replace("link", "")
        lo = pdata.xpos[probe.body(fr3).id]
        radial = wp - lo
        radial = radial / (np.linalg.norm(radial) + 1e-12)
        if free_dist(wp, radial) > 0.08:
            aim_world[s["name"]] = radial
            continue
        # buried radially (e.g. link5 front sensors) -> search candidates for the freest
        # direction, biased toward radial so it stays a sensible outward aim
        best, best_score = radial, -1e9
        for c in CAND:
            if float(np.dot(c, radial)) < -0.1:      # never aim back inward
                continue
            fd = free_dist(wp, c)
            if fd < 0.08:
                continue
            score = min(fd, 0.3) + 0.15 * float(np.dot(c, radial))
            if score > best_score:
                best_score, best = score, c
        aim_world[s["name"]] = best
        repaired += 1

    # 4b) pass 2: cameras with the chosen aim (converted to host-body frame) + outward offset
    n = 0
    for s in sensors:
        host = skin_bodies.get(s["parent"]) or spec.body(s["parent"])
        sid = probe.site(s["name"] + "_site").id
        Rb = pdata.site_xmat[sid].reshape(3, 3)
        aim_local = Rb.T @ aim_world[s["name"]]              # world -> host-body frame
        aim_local /= np.linalg.norm(aim_local) + 1e-12
        # link5_front sensors are wedged in the wrist concavity (2-4 mm to self-structure) — push
        # their cameras further out so they clear the surrounding geometry instead of staring at it
        off = 0.022 if "link5_front" in s["name"] else CAM_OFFSET
        cam = host.add_camera(name=s["name"],
                              pos=list(np.asarray(s["xyz"], float) + off * aim_local),
                              quat=quat_view_along(aim_local))
        cam.fovy = 45.0
        cam.resolution = [8, 8]
        n += 1
    print(f"sensor aim: {n - repaired} radial, {repaired} raycast-repaired")

    # CRITICAL: the depth-render near clip is vis.map.znear * stat.extent. With a house in the
    # scene extent grows to ~10-20 m, so the default znear (0.01) clips every proximity return
    # closer than ~0.1-0.2 m — i.e. the entire contact-imminent band the skin exists to sense.
    # Pin znear tiny so znear*extent stays < ~5 mm even for big scenes. (Found via the viz suite;
    # molmospaces datagen uses this same MjOpenGLRenderer depth path.)
    spec.visual.map.znear = 0.0002

    model = spec.compile()
    OUT.write_text(spec.to_xml())
    ncam_sens = sum(1 for i in range(model.ncam) if "_sensor_" in model.camera(i).name)
    print(f"compiled: nbody={model.nbody} ncam={model.ncam} hybrid sensors={ncam_sens} "
          f"(added {n})")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
