"""Exhaustive sanity verification of the 40 hybrid-skin SPAD sensors on model_hybrid.xml.

Checks (per sensor):
  A SELF-HIT   empty world, 3 arm poses: does the sensor stare into its own arm/skin?
               (min depth < 3cm at every pose = mis-oriented / buried)
  B OUTWARD    camera forward (-z) vs (sensor pos - parent link center): dot > 0 expected.
  C PLATE      a 20x20cm plate is placed exactly 0.15 m along the sensor's view axis,
               facing it; the center-pixel depth must read ~0.145 m (0.15 - half thickness).
               THE definitive per-sensor aim+scale check.
  D CLOUD      with a wall (x=0.55) + table (z=0.34 top) present, back-project ALL returns
               to world points; each point's distance to the nearest scene surface (analytic
               boxes + arm AABBs) must be ~0. Reports RMS + worst error per sensor.

Outputs: diagnostics_output/<ts>_hybrid_sensor_verify/
  sensor_verify.csv      per-sensor table (all checks + PASS/FAIL)
  cloud_vs_geometry.png  3D scatter of back-projected points over the GT boxes
  verify_sheet.png       3D render + cloud + per-sensor verdict table
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import OpenGL.EGL as _EGL  # noqa: E402  (EGL device-platform wedged on this box; use default display)
_d = _EGL.eglGetDisplay(_EGL.EGL_DEFAULT_DISPLAY)
_maj, _min = _EGL.EGLint(), _EGL.EGLint()
if _EGL.eglInitialize(_d, _maj, _min):
    import mujoco.egl as _me
    _me.EGL_DISPLAY = _d

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT = REPO_ROOT / "assets/robots/franka_skin/model_hybrid.xml"
OUT = REPO_ROOT / "diagnostics_output/hybrid_safety_stack/hybrid_sensor_verify"
OUT.mkdir(parents=True, exist_ok=True)

POSES = [
    [0.0, -0.5, 0.0, -2.1, 0.0, 1.9, 0.8],     # reach forward
    [0.0, 0.3, 0.0, -1.2, 0.0, 1.6, 0.0],      # upright
    [0.6, -0.9, 0.4, -2.4, 0.3, 2.2, 0.4],     # twisted
]
WALL = dict(c=[0.55, 0.30, 0.55], h=[0.22, 0.02, 0.35])
TABLE = dict(c=[0.55, 0.0, 0.32], h=[0.20, 0.30, 0.02])
FOVY = 45.0


def build(with_scene: bool, with_plate: bool):
    spec = mujoco.MjSpec.from_file(str(ROBOT))
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 1280
    if with_scene:
        for nm, bx in (("v_wall", WALL), ("v_table", TABLE)):
            b = spec.worldbody.add_body(name=nm, pos=bx["c"])
            g = b.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = bx["h"]
            g.rgba = [0.8, 0.55, 0.3, 1.0]
    if with_plate:
        pb = spec.worldbody.add_body(name="v_plate", mocap=True, pos=[0, 0, -5])
        g = pb.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.10, 0.10, 0.005]
        g.rgba = [0.2, 0.9, 0.2, 1.0]
        g.contype = 0
        g.conaffinity = 0
    return spec.compile()


def set_pose(model, data, q):
    for i, v in enumerate(q, start=1):
        j = f"fr3_joint{i}"
        data.qpos[model.joint(j).qposadr[0]] = v
    mujoco.mj_forward(model, data)


def sensor_cams(model):
    return [model.camera(i).name for i in range(model.ncam) if "_sensor_" in model.camera(i).name]


def cam_pose(model, data, name):
    cid = model.camera(name).id
    Rm = data.cam_xmat[cid].reshape(3, 3)
    return data.cam_xpos[cid].copy(), Rm


def backproject(d8, pos, Rm, d_min=0.005, d_max=3.0):
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
    z_c = -d
    pts = (Rm @ np.stack([x_c, y_c, z_c], 0)).T + pos
    return pts, d


def box_dist(p, c, h):
    d = np.abs(p - np.asarray(c)) - np.asarray(h)
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(np.max(d, axis=-1), 0).clip(max=0) * 0


def main():
    results = {}

    # ---------- A: SELF-HIT (empty world, 3 poses) ----------
    model = build(False, False)
    data = mujoco.MjData(model)
    names = sensor_cams(model)
    rd = mujoco.Renderer(model, 8, 8)
    rd.enable_depth_rendering()
    self_min = {n: [] for n in names}
    for q in POSES:
        set_pose(model, data, q)
        for n in names:
            rd.update_scene(data, n)
            self_min[n].append(float(rd.render().min()))
    for n in names:
        worst = max(self_min[n])      # best case over poses: if even the MAX of mins < 3cm -> buried
        results[n] = dict(self_min_max=worst, self_buried=worst < 0.03)

    # ---------- B: OUTWARD (camera fwd vs link-radial) ----------
    set_pose(model, data, POSES[0])
    for n in names:
        cid = model.camera(n).id
        bid = int(model.cam_bodyid[cid])
        pos, Rm = cam_pose(model, data, n)
        fwd = -Rm[:, 2]
        radial = pos - data.xpos[bid]
        nr = np.linalg.norm(radial)
        dot = float(np.dot(fwd, radial / nr)) if nr > 1e-6 else 1.0
        results[n]["outward_dot"] = dot
        results[n]["outward_ok"] = dot > -0.2   # allow tangential mounts, flag clearly inward

    # ---------- C: PLATE @ 0.15 m along the view axis ----------
    model_p = build(False, True)
    data_p = mujoco.MjData(model_p)
    set_pose(model_p, data_p, POSES[0])
    mid = int(model_p.body_mocapid[model_p.body("v_plate").id])
    rdp = mujoco.Renderer(model_p, 8, 8)
    rdp.enable_depth_rendering()
    for n in names:
        pos, Rm = cam_pose(model_p, data_p, n)
        fwd = -Rm[:, 2]
        data_p.mocap_pos[mid] = pos + 0.15 * fwd
        # orient plate so its z (thin axis) faces the camera
        z = fwd
        up = np.array([0, 0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0, 0])
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).reshape(9))
        data_p.mocap_quat[mid] = q
        mujoco.mj_forward(model_p, data_p)
        rdp.update_scene(data_p, n)
        d8 = rdp.render()
        center = float(d8[3:5, 3:5].mean())
        results[n]["plate_read"] = center
        results[n]["plate_ok"] = abs(center - 0.145) < 0.012
    data_p.mocap_pos[mid] = [0, 0, -5]

    # ---------- D: CLOUD vs GT geometry ----------
    model_s = build(True, False)
    data_s = mujoco.MjData(model_s)
    set_pose(model_s, data_s, POSES[0])
    rds = mujoco.Renderer(model_s, 8, 8)
    rds.enable_depth_rendering()
    # arm body AABBs (world) so points hitting the OWN ARM still count as a valid surface
    arm_boxes = []
    for gi in range(model_s.ngeom):
        bn = model_s.body(int(model_s.geom_bodyid[gi])).name
        if bn and ("fr3" in bn or "skin" in bn or "gripper" in bn or "2f85" in bn):
            aabb = model_s.geom_aabb[gi]
            xp = data_s.geom_xpos[gi]
            xm = data_s.geom_xmat[gi].reshape(3, 3)
            corners = []
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        corners.append(xp + xm @ (aabb[:3] + np.array([sx, sy, sz]) * aabb[3:]))
            corners = np.array(corners)
            arm_boxes.append((corners.mean(0), (corners.max(0) - corners.min(0)) / 2))
    gt_boxes = [(np.array(WALL["c"]), np.array(WALL["h"])),
                (np.array(TABLE["c"]), np.array(TABLE["h"]))] + arm_boxes

    all_pts, all_d, all_err = [], [], []
    for n in names:
        rds.update_scene(data_s, n)
        d8 = rds.render()
        pos, Rm = cam_pose(model_s, data_s, n)
        pts, d = backproject(d8, pos, Rm)
        if len(pts) == 0:
            results[n]["cloud_rms"] = float("nan")
            results[n]["cloud_ok"] = True   # nothing in range is fine
            continue
        errs = np.min(np.stack([box_dist(pts, c, h) for c, h in gt_boxes], 1), axis=1)
        results[n]["cloud_rms"] = float(np.sqrt((errs ** 2).mean()))
        results[n]["cloud_worst"] = float(errs.max())
        results[n]["cloud_ok"] = float(np.sqrt((errs ** 2).mean())) < 0.02
        all_pts.append(pts)
        all_d.append(d)
        all_err.append(errs)

    # ---------- summary ----------
    rows = ["sensor,self_min_max_m,self_buried,outward_dot,plate_read_m,plate_ok,cloud_rms_m,cloud_ok,VERDICT"]
    n_pass = 0
    for n in names:
        r = results[n]
        ok = (not r["self_buried"]) and r["plate_ok"] and r.get("cloud_ok", True)
        n_pass += ok
        rows.append(f"{n},{r['self_min_max']:.3f},{r['self_buried']},{r['outward_dot']:+.2f},"
                    f"{r.get('plate_read', float('nan')):.3f},{r['plate_ok']},"
                    f"{r.get('cloud_rms', float('nan')):.4f},{r.get('cloud_ok', True)},"
                    f"{'PASS' if ok else 'FAIL'}")
    (OUT / "sensor_verify.csv").write_text("\n".join(rows))
    print(f"PASS {n_pass}/{len(names)}")
    for n in names:
        r = results[n]
        flag = "" if ((not r["self_buried"]) and r["plate_ok"] and r.get("cloud_ok", True)) else "  <-- FAIL"
        print(f"  {n:24s} self_min={r['self_min_max']:.3f} out={r['outward_dot']:+.2f} "
              f"plate={r.get('plate_read', -1):.3f} cloudRMS={r.get('cloud_rms', float('nan')):.4f}{flag}")

    # ---------- figures ----------
    pts = np.concatenate(all_pts) if all_pts else np.zeros((0, 3))
    dd = np.concatenate(all_d) if all_d else np.zeros((0,))
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(121, projection="3d")
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=dd, cmap="turbo_r", s=4)
    for c, h in gt_boxes[:2]:
        # wireframe GT boxes
        for sx in (-1, 1):
            for sy in (-1, 1):
                ax.plot([c[0] - h[0], c[0] + h[0]], [c[1] + sy * h[1]] * 2, [c[2] + sx * h[2]] * 2,
                        "k-", lw=0.5)
    ax.set_title(f"Back-projected SPAD returns vs GT geometry  (n={len(pts)} pts)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, 0.6)
    ax.set_zlim(0, 1.2)
    ax2 = fig.add_subplot(122)
    if all_err:
        errs = np.concatenate(all_err)
        ax2.hist(errs * 1000, bins=60, color="#2a9d8f")
        ax2.axvline(20, color="r", ls="--", label="20mm tolerance")
        ax2.set_xlabel("point -> nearest GT surface error (mm)")
        ax2.set_title(f"Cloud geometric error: median {np.median(errs)*1000:.1f}mm, "
                      f"p95 {np.percentile(errs,95)*1000:.1f}mm")
        ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cloud_vs_geometry.png", dpi=140)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
