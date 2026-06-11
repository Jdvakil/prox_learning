"""Convert fr3_hybrid_skin.urdf -> MuJoCo MJCF and add exo + wrist + 40 SPAD sensor cameras.

The URDF references COLLADA (.dae) visual meshes (MuJoCo can't load those) via package:// paths.
We rewrite visual mesh refs to MuJoCo-loadable local files:
  - FR3 arm links  -> the detailed franka_skin link{i}.obj (standard FR3, link-frame authored)
  - skin dermis    -> the downloaded hybrid_dermis STLs (mm meshes, scale 0.001)
  - franka hand    -> the downloaded hand.stl
  - fingers        -> dropped (no STL); thin box geoms added back post-compile
All <collision> elements are dropped (visual-only viz model).

Each of the 40 skin sensors is a fixed joint (parent = link*_skin, child = link*_sensor_N) whose
<axis> is the OUTWARD surface normal. We add one fovy=45, 8x8 camera per sensor on the parent
skin body, oriented so the camera view axis (-z) points along that outward normal.

Out: assets/robots/fr3_hybrid_skin/model.xml  (+ a quick render PNG).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path("/home/jaydv/code/prox_learning")
URDF = ROOT / "assets/urdf/fr3_hybrid_skin.urdf"
ROBOT_DIR = ROOT / "assets/robots/fr3_hybrid_skin"
MD = ROBOT_DIR / "meshes"
FS_OBJ = ROOT / "assets/robots/franka_skin/assets"   # pretty FR3 arm .obj
OUT_XML = ROBOT_DIR / "model.xml"


VIS = MD / "visual"   # single meshdir of MuJoCo-loadable STLs (basenames)


def stage_meshes() -> None:
    """Copy every visual mesh we use into one meshdir (basenames), guaranteeing alignment:
    arm = gentact collision STLs (authored in the URDF link frame), skin = hybrid dermis,
    hand = franka hand STL."""
    import shutil
    VIS.mkdir(parents=True, exist_ok=True)
    # arm = the DETAILED franka_skin FR3 visual .obj (same canonical FR3, authored in link frame)
    for i in range(8):
        src = FS_OBJ / f"link{i}.obj"
        if src.exists():
            shutil.copy(src, VIS / f"link{i}.obj")
        else:
            shutil.copy(MD / "arm" / f"link{i}.stl", VIS / f"link{i}.stl")  # fallback
    for p in (MD / "skin").glob("*.stl"):
        shutil.copy(p, VIS / p.name)
    shutil.copy(MD / "hand" / "hand.stl", VIS / "hand.stl")


def resolve_visual_mesh(fname: str) -> tuple[str | None, str | None]:
    """(basename, mesh_name) for a visual mesh URDF filename, or (None,None) to drop."""
    base = fname.split("/")[-1]
    if "skin/hybrid/" in fname:                       # link*_hybrid_dermis.stl
        return base, base.replace(".stl", "")
    if "robot_arms/fr3/visual/" in fname:             # link{i}.dae -> pretty link{i}.obj
        i = base.replace(".dae", "")
        if (FS_OBJ / f"{i}.obj").exists():
            return f"{i}.obj", f"arm_{i}"
        return f"{i}.stl", f"arm_{i}"
    if "franka_hand_white/visual/hand.dae" in fname:
        return "hand.stl", "hand"
    if "finger.dae" in fname:
        return None, None                              # no STL -> drop, add box later
    return None, None


def rewrite_urdf(tmp_path: Path):
    tree = ET.parse(URDF)
    robot = tree.getroot()

    # inject <mujoco> compiler block (URDF mode)
    mj = ET.Element("mujoco")
    comp = ET.SubElement(mj, "compiler")
    comp.set("meshdir", str(VIS))
    comp.set("balanceinertia", "true")
    comp.set("discardvisual", "false")
    comp.set("fusestatic", "false")
    robot.insert(0, mj)

    # drop all collisions; rewrite visual mesh refs
    for link in robot.findall("link"):
        for col in list(link.findall("collision")):
            link.remove(col)
        for vis in list(link.findall("visual")):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            path, name = resolve_visual_mesh(mesh.get("filename", ""))
            if path is None:
                link.remove(vis)
                continue
            mesh.set("filename", path)
            mesh.set("name", name)            # stable asset name
    tree.write(tmp_path)


def parse_sensors() -> list[dict]:
    """[{name, parent, xyz, axis}] for every *_sensor_*_joint in the URDF."""
    tree = ET.parse(URDF)
    out = []
    for j in tree.getroot().findall("joint"):
        jn = j.get("name", "")
        if "_sensor_" not in jn or not jn.endswith("_joint"):
            continue
        origin = j.find("origin")
        axis = j.find("axis")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        xyz = [float(v) for v in origin.get("xyz").split()]
        a = [float(v) for v in (axis.get("xyz").split() if axis is not None else "0 0 1".split())]
        out.append(dict(name=child, parent=parent, xyz=xyz, axis=a))
    return out


def quat_view_along(direction: np.ndarray) -> list[float]:
    """quat (w,x,y,z) for a camera whose -z axis points along `direction` (world/body)."""
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    z = -d                                             # camera local +z is opposite view dir
    up = np.array([0, 0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0, 0])
    x = np.cross(up, z); x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(9))
    return [float(v) for v in q]


def main():
    stage_meshes()
    tmp = ROBOT_DIR / "_rewritten.urdf"
    rewrite_urdf(tmp)
    spec = mujoco.MjSpec.from_file(str(tmp))

    bodies = {b.name: b for b in spec.bodies}

    # 1) sensor cameras on each parent skin body
    sensors = parse_sensors()
    n_added = 0
    for s in sensors:
        parent = bodies.get(s["parent"])
        if parent is None:
            continue
        cam = parent.add_camera()
        cam.name = s["name"]
        cam.pos = s["xyz"]
        cam.quat = quat_view_along(np.array(s["axis"]))
        cam.fovy = 45.0
        cam.resolution = [8, 8]
        # small magenta marker site so the sensor is visible in the 3D view
        site = parent.add_site()
        site.name = s["name"] + "_site"
        site.pos = s["xyz"]
        site.size = [0.004, 0.004, 0.004]
        site.rgba = [1.0, 0.1, 0.8, 1.0]
        n_added += 1

    # 2) wrist camera on the hand
    hand = bodies.get("fr3_hand") or bodies.get("fr3_link8") or bodies.get("fr3_link7")
    if hand is not None:
        wc = hand.add_camera()
        wc.name = "wrist_camera"
        wc.pos = [0.05, 0.0, 0.03]
        wc.quat = quat_view_along(np.array([0.0, 0.0, 1.0]))   # look along +z (toward fingers)
        wc.fovy = 58.0
        wc.resolution = [240, 320]

    # 3) exo camera in the world, framing the robot
    exo = spec.worldbody.add_camera()
    exo.name = "exo_camera_1"
    exo.pos = [1.4, -1.2, 1.3]
    exo.quat = quat_view_along(np.array([-1.4, 1.2, -0.9]))
    exo.fovy = 50.0
    exo.resolution = [480, 640]

    # 4) thin box fingers for context (no finger STL)
    for fn in ("fr3_leftfinger", "fr3_rightfinger"):
        fb = bodies.get(fn)
        if fb is not None:
            g = fb.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [0.01, 0.018, 0.027]
            g.pos = [0, 0, 0.04]
            g.rgba = [0.25, 0.25, 0.27, 1.0]

    # world light + floor for the render
    spec.worldbody.add_light(pos=[0.6, 0.4, 2.4], dir=[-0.2, -0.1, -1],
                             diffuse=[0.9, 0.9, 0.9], specular=[0.2, 0.2, 0.2])
    fl = spec.worldbody.add_geom()
    fl.type = mujoco.mjtGeom.mjGEOM_PLANE
    fl.size = [3, 3, 0.1]
    fl.rgba = [0.32, 0.33, 0.36, 1.0]

    # color the skin dermis distinct (translucent teal) and the arm light grey
    for g in spec.geoms:
        mn = (g.meshname or "")
        if "dermis" in mn:
            g.rgba = [0.10, 0.70, 0.78, 0.38]   # translucent so the white FR3 reads through
        elif mn.startswith("link") or mn == "hand":
            g.rgba = [0.95, 0.95, 0.96, 1.0]   # FR3 white

    model = spec.compile()
    OUT_XML.write_text(spec.to_xml())
    tmp.unlink(missing_ok=True)
    print(f"compiled: nbody={model.nbody} ngeom={model.ngeom} ncam={model.ncam} "
          f"(sensors added={n_added})")
    print(f"saved -> {OUT_XML}")
    return model


if __name__ == "__main__":
    main()
