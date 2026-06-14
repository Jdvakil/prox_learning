"""Synthetic near-contact sweep for the proximity safety head.

Builds the fumehood scene (analytic boxes, same coordinates as fumehood.xml) with the
40-sensor hybrid-skin FR3 attached, replays REAL arm postures sampled from collected
datagen trajectories, and per sample stands 1-2 hazard bars on the bench right next to
a randomly chosen skin sensor's view axis. Renders all 40 SPAD depths (8x8, geom group
2 hidden — identical to datagen's record_proximity_depths) and computes an analytic
retreat label:

    dq = sum over sensors with an environment return closer than D_ACT of
         J_p(sensor)^T * unit(sensor_pos - hit_point) * (1/d - 1/D_ACT)

i.e. a whole-arm potential-field repulsion in joint space, exact from sim state.
Self-hits (returns on the robot's own arm) are excluded from the label by checking the
back-projected point against the analytic scene boxes, but stay in the INPUT — the
trained head must cope with them at runtime.

Output h5 (default assets/safety/sweep_v1.h5):
    prox        (N, 40, 8, 8) float16   raw planar-z depths, sensor order = /sensors
    label_dq    (N, 7)        float32   analytic repulsion (unscaled)
    qpos        (N, 7)        float32   arm joints
    grip        (N,)          float32   gripper driver value
    base_pose   (N, 7)        float32   x,y,z, qw,qx,qy,qz (h5 convention, z=floor)
    min_depth   (N, 40)       float32   per-sensor min environment-validated depth (inf if none)
    bar_center  (N, 3, 3) / bar_half (N, 3, 3) / bar_active (N, 3) bool
    sensors     (40,) bytes   sensor names (h5 obs/proximity keys, no namespace)
    attrs: d_act, d_max_input, mount_z, source runs

Usage:
    OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      /opt/conda/envs/mlspaces/bin/python scripts/safety_sweep.py \
      --runs /home/jaydv/code/prox_learning/assets/datagen/hybrid_pnp5_mass/FrankaSkinHybridPnP5MassConfig/20260612_023111 \
      --n 15000 --out assets/safety/sweep_v1.h5
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import OpenGL.EGL as _EGL  # noqa: E402  (EGL device-platform wedged on this box; use default display)
_d = _EGL.eglGetDisplay(_EGL.EGL_DEFAULT_DISPLAY)
_maj, _min = _EGL.EGLint(), _EGL.EGLint()
if _EGL.eglInitialize(_d, _maj, _min):
    import mujoco.egl as _me
    _me.EGL_DISPLAY = _d

import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

NS = "robot_0/"
ROBOT_XML = Path("/home/jaydv/code/prox_learning/assets/robots/franka_skin/model_hybrid.xml")
MOUNT_Z = 0.35
D_ACT = 0.18          # repulsion activation distance (m)
HIT_TOL = 0.025       # back-projected point must lie this close to a scene box surface
BENCH_Z = 0.72        # bench top (bar feet)
TUBE_X0 = 0.58        # hood mouth plane
FOVY = 45.0

# fumehood.xml geometry, replicated as analytic boxes: (center, half)
STATIC_BOXES = {
    "room_wall_back":  ([-2.0, 0.0, 1.4], [0.05, 2.05, 1.4]),
    "room_wall_front": ([2.4, 0.0, 1.4], [0.05, 2.05, 1.4]),
    "room_wall_left":  ([0.2, 2.0, 1.4], [2.25, 0.05, 1.4]),
    "room_wall_right": ([0.2, -2.0, 1.4], [2.25, 0.05, 1.4]),
    "room_ceiling":    ([0.2, 0.0, 2.82], [2.25, 2.05, 0.04]),
    "bench_top":       ([0.95, 0.0, 0.70], [0.45, 0.60, 0.02]),
    "bench_body":      ([0.95, 0.0, 0.35], [0.42, 0.57, 0.33]),
    "hood_side_l":     ([0.95, 0.45, 1.12], [0.40, 0.012, 0.40]),
    "hood_side_r":     ([0.95, -0.45, 1.12], [0.40, 0.012, 0.40]),
    "hood_back":       ([1.36, 0.0, 1.12], [0.012, 0.46, 0.40]),
    "hood_top":        ([0.95, 0.0, 1.53], [0.42, 0.46, 0.015]),
    "hood_frame_l":    ([0.58, 0.45, 1.12], [0.02, 0.02, 0.40]),
    "hood_frame_r":    ([0.58, -0.45, 1.12], [0.02, 0.02, 0.40]),
    "floor":           ([0.0, 0.0, -0.05], [5.0, 5.0, 0.05]),
}
# mocap bodies posed per sample (geom halves fixed; z half includes bar heights)
SASH_HALF = [0.015, 0.44, 0.025]
JAMB_HALF = [0.012, 0.18, 0.20]
BARS = {"bar_s": [0.0175, 0.0175, 0.10], "bar_m": [0.025, 0.025, 0.11],
        "bar_l": [0.035, 0.035, 0.12]}
SPHERE_R = 0.05      # blue spherical obstacle (sphere demo); proves depth sensing is shape/colour agnostic
PARK = [0.0, 0.0, -3.0]
BENCH_REGION = dict(x=(0.62, 1.32), y=(-0.40, 0.40))


def build_model() -> mujoco.MjModel:
    xml = (
        '<mujoco model="safety_sweep"><compiler angle="radian"/>'
        '<option gravity="0 0 -9.8" integrator="implicitfast"/>'
        '<visual><global offwidth="1280" offheight="720"/><map znear="0.001"/>'
        '<headlight ambient="0.5 0.5 0.5" diffuse="0.65 0.65 0.65"/></visual>'
        '<worldbody><light pos="0.4 0.5 2.6" dir="0 0 -1" directional="true"/></worldbody>'
        "</mujoco>"
    )
    fd = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
    fd.write(xml)
    fd.close()
    spec = mujoco.MjSpec.from_file(fd.name)
    for name, (c, h) in STATIC_BOXES.items():
        b = spec.worldbody.add_body(name=name, pos=c)
        g = b.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = h
        g.rgba = [0.85, 0.86, 0.88, 1.0]
        g.contype = 0
        g.conaffinity = 0
    for name, h in (("sash", SASH_HALF), ("jamb_l", JAMB_HALF), ("jamb_r", JAMB_HALF),
                    *BARS.items()):
        b = spec.worldbody.add_body(name=name, pos=PARK, mocap=True)
        g = b.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = h
        g.rgba = [1.0, 0.45, 0.05, 1.0] if name.startswith("bar") else [0.7, 0.72, 0.75, 1.0]
        g.contype = 0
        g.conaffinity = 0
    # blue spherical obstacle, parked by default (inert for the sweep + other demos). The skin
    # senses DEPTH only, so the head — trained on box bars — reacts to this sphere all the same;
    # the sphere demo drives it to show the head is shape- and colour-agnostic.
    sb = spec.worldbody.add_body(name="sphere", pos=PARK, mocap=True)
    sg = sb.add_geom()
    sg.type = mujoco.mjtGeom.mjGEOM_SPHERE
    sg.size = [SPHERE_R, SPHERE_R, SPHERE_R]
    sg.rgba = [0.10, 0.35, 0.95, 1.0]
    sg.contype = 0
    sg.conaffinity = 0
    base = spec.worldbody.add_body(name=f"{NS}base", pos=[0, 0, 0], quat=[1, 0, 0, 0],
                                   mocap=True)
    frame = base.add_frame(pos=[0, 0, MOUNT_Z])
    robot = mujoco.MjSpec.from_file(str(ROBOT_XML))
    frame.attach_body(robot.worldbody.first_body(), NS, "")
    return spec.compile()


def bars_to_spheres(model, radius: float = 0.045, rgba=(0.10, 0.35, 0.95, 1.0)) -> None:
    """Convert the box hazard-bar geoms into blue spheres in place. Depth sensing is
    shape/colour-agnostic, so any bar-driven demo becomes a sphere demo with IDENTICAL
    motion — only the rendered obstacle changes (proves the head ignores appearance)."""
    for gid in range(model.ngeom):
        if model.body(model.geom_bodyid[gid]).name in BARS:
            model.geom_type[gid] = mujoco.mjtGeom.mjGEOM_SPHERE
            model.geom_size[gid] = [radius, radius, radius]
            model.geom_rgba[gid] = list(rgba)


def min_obstacle_gap(model, data, mid, names, link_bids) -> float:
    """Min distance from any active (un-parked) obstacle mocap to the nearest arm link body."""
    best = float("inf")
    for n in names:
        c = data.mocap_pos[mid[n]]
        if c[2] <= -1.0:        # parked
            continue
        for b in link_bids:
            best = min(best, float(np.linalg.norm(c - data.xpos[b])))
    return best


def clean_hud(frame, md, gap, dev):
    """Minimal evaluation overlay: only min skin depth, distance to obstacle, deviation."""
    import cv2

    def line(s, y):
        cv2.putText(frame, s, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (245, 245, 245), 2, cv2.LINE_AA)

    mdcm = md * 100 if np.isfinite(md) else 999.0
    gcm = gap * 100 if np.isfinite(gap) else 999.0
    line(f"min skin depth    {mdcm:5.1f} cm", 44)
    line(f"sphere distance   {gcm:5.1f} cm", 80)
    line(f"deviation         {dev:4.2f} rad", 116)
    return frame


def load_postures(run_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """All (arm qpos, grip, base_pose) rows from every trajectory in the given runs."""
    qs, grips, bases, sensors = [], [], [], None
    files = sorted(p for d in run_dirs for p in d.glob("house_*/trajectories*.h5"))
    if not files:
        raise SystemExit(f"no trajectories*.h5 under {run_dirs}")
    for f in files:
        with h5py.File(f, "r") as h:
            for key in sorted(k for k in h if k.startswith("traj")):
                t = h[key]
                if sensors is None:
                    sensors = sorted(t["obs/proximity"].keys())
                base = t["obs/extra/robot_base_pose"][:].astype(np.float64)
                qds = t["obs/agent/qpos"]
                T = base.shape[0]
                for i in range(T):
                    raw = qds[i]
                    b = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
                    n = b.find(b"\x00")
                    qp = json.loads(b[: n if n != -1 else None].decode("utf-8", "ignore"))
                    qs.append(qp["arm"])
                    grips.append((qp.get("gripper") or [0.0])[0])
                    bases.append(base[i])
    return (np.asarray(qs, np.float64), np.asarray(grips, np.float64),
            np.asarray(bases, np.float64), sensors)


def box_surface_dist(p: np.ndarray, boxes: list[tuple[np.ndarray, np.ndarray]]) -> float:
    best = np.inf
    for c, h in boxes:
        q = np.abs(p - c) - h
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(max(q[0], q[1], q[2]), 0.0)
        best = min(best, abs(outside + inside))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True,
                    help="datagen run dir(s) containing house_*/trajectories*.h5")
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--out", type=Path,
                    default=Path("/home/jaydv/code/prox_learning/assets/safety/sweep_v1.h5"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    q_all, grip_all, base_all, sensors = load_postures(args.runs)
    print(f"postures: {len(q_all)} rows, {len(sensors)} sensors")

    model = build_model()
    data = mujoco.MjData(model)
    cam_ids = {s: model.camera(f"{NS}{s}").id for s in sensors}
    arm_qadr = [model.joint(f"{NS}fr3_joint{i}").qposadr[0] for i in range(1, 8)]
    arm_dofadr = [model.joint(f"{NS}fr3_joint{i}").dofadr[0] for i in range(1, 8)]
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    finger_qadr = [model.joint(f"{NS}gripper/{n}").qposadr[0]
                   for n in ("left_driver_joint", "right_driver_joint")
                   if f"{NS}gripper/{n}" in joint_names]
    mid = {n: int(model.body_mocapid[model.body(n).id])
           for n in ("sash", "jamb_l", "jamb_r", *BARS, f"{NS}base")}

    rd = mujoco.Renderer(model, 8, 8)
    rd.enable_depth_rendering()
    rd.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[2] = 0   # hide cosmetic skin, exactly like datagen

    f = 4.0 / np.tan(np.deg2rad(FOVY / 2))     # 8x8 intrinsics
    cx = cy = 3.5
    uu, vv = np.meshgrid(np.arange(8), np.arange(8))
    jacp = np.zeros((3, model.nv))
    bar_names = list(BARS)

    N = args.n
    X = np.zeros((N, len(sensors), 8, 8), np.float16)
    Y = np.zeros((N, 7), np.float32)
    Q = np.zeros((N, 7), np.float32)
    G = np.zeros((N,), np.float32)
    B = np.zeros((N, 7), np.float32)
    MD = np.full((N, len(sensors)), np.inf, np.float32)
    BC = np.zeros((N, 3, 3), np.float32)
    BH = np.zeros((N, 3, 3), np.float32)
    BA = np.zeros((N, 3), bool)
    n_close = 0

    for i in range(N):
        row = rng.integers(0, len(q_all))
        bp = base_all[row]
        data.mocap_pos[mid[f"{NS}base"]] = bp[:3]
        data.mocap_quat[mid[f"{NS}base"]] = bp[3:7]
        for adr, val in zip(arm_qadr, q_all[row]):
            data.qpos[adr] = float(val)
        for adr in finger_qadr:
            data.qpos[adr] = float(grip_all[row])

        # aperture: same distribution the Big/Obstacle samplers draw
        ap_w = rng.uniform(0.50, 0.85)
        ap_h = rng.uniform(0.45, 0.62)
        data.mocap_pos[mid["sash"]] = [TUBE_X0, 0.0, BENCH_Z + ap_h + 0.025]
        data.mocap_pos[mid["jamb_l"]] = [TUBE_X0, ap_w / 2 + 0.18, BENCH_Z + 0.20]
        data.mocap_pos[mid["jamb_r"]] = [TUBE_X0, -ap_w / 2 - 0.18, BENCH_Z + 0.20]
        for n in bar_names:
            data.mocap_pos[mid[n]] = PARK
        mujoco.mj_forward(model, data)

        # engineered bars: stand on the bench along a random sensor's view ray, so the
        # 3-15 cm band is actually populated instead of waiting for luck
        active = []
        n_bars = int(rng.integers(1, 3)) if rng.random() < 0.8 else 0
        order = rng.permutation(len(bar_names))[:n_bars]
        for k in order:
            name = bar_names[k]
            for _ in range(8):
                s = sensors[int(rng.integers(0, len(sensors)))]
                cid = cam_ids[s]
                pos = data.cam_xpos[cid]
                fwd = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
                d = rng.uniform(0.05, 0.25)
                xy = (pos + fwd * d)[:2] + rng.normal(0, 0.015, 2)
                if (BENCH_REGION["x"][0] <= xy[0] <= BENCH_REGION["x"][1]
                        and BENCH_REGION["y"][0] <= xy[1] <= BENCH_REGION["y"][1]):
                    half = BARS[name]
                    data.mocap_pos[mid[name]] = [xy[0], xy[1], BENCH_Z + half[2]]
                    active.append(name)
                    break
        mujoco.mj_forward(model, data)

        # the analytic scene of THIS sample (for self-hit rejection + labels)
        boxes = [(np.asarray(c, float), np.asarray(h, float))
                 for c, h in STATIC_BOXES.values()]
        boxes.append((data.mocap_pos[mid["sash"]].copy(), np.asarray(SASH_HALF)))
        boxes.append((data.mocap_pos[mid["jamb_l"]].copy(), np.asarray(JAMB_HALF)))
        boxes.append((data.mocap_pos[mid["jamb_r"]].copy(), np.asarray(JAMB_HALF)))
        for k, n in enumerate(bar_names):
            BC[i, k] = data.mocap_pos[mid[n]]
            BH[i, k] = BARS[n]
            BA[i, k] = n in active
            if n in active:
                boxes.append((data.mocap_pos[mid[n]].copy(), np.asarray(BARS[n])))

        tau = np.zeros(model.nv)
        for si, s in enumerate(sensors):
            cid = cam_ids[s]
            rd.update_scene(data, camera=f"{NS}{s}", scene_option=opt)
            d8 = rd.render()
            X[i, si] = d8.astype(np.float16)
            m = d8 >= 0.005
            if not m.any():
                continue
            j = np.argmin(np.where(m, d8, np.inf))
            u, v = uu.flat[j], vv.flat[j]
            dz = float(d8.flat[j])
            if dz >= D_ACT:
                continue
            Rm = data.cam_xmat[cid].reshape(3, 3)
            pos = data.cam_xpos[cid]
            # MuJoCo cam frame is GL: +x right, +y up, -z forward
            p_cam = np.array([(u - cx) * dz / f, -(v - cy) * dz / f, -dz])
            p_o = Rm @ p_cam + pos
            if box_surface_dist(p_o, boxes) > HIT_TOL:
                continue   # self-hit (own arm): excluded from the label, kept in the input
            r = float(np.linalg.norm(pos - p_o))
            if r < 1e-4 or r >= D_ACT:
                continue
            MD[i, si] = r
            direction = (pos - p_o) / r
            w = 1.0 / r - 1.0 / D_ACT
            mujoco.mj_jac(model, data, jacp, None, pos, int(model.cam_bodyid[cid]))
            tau += jacp.T @ (direction * w)

        Y[i] = tau[arm_dofadr].astype(np.float32)
        Q[i] = q_all[row]
        G[i] = grip_all[row]
        B[i] = bp
        if np.isfinite(MD[i]).any():
            n_close += 1
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{N}  close-encounter rate so far: {n_close / (i + 1):.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h:
        h.create_dataset("prox", data=X, compression="gzip", compression_opts=4)
        h.create_dataset("label_dq", data=Y)
        h.create_dataset("qpos", data=Q)
        h.create_dataset("grip", data=G)
        h.create_dataset("base_pose", data=B)
        h.create_dataset("min_depth", data=MD)
        h.create_dataset("bar_center", data=BC)
        h.create_dataset("bar_half", data=BH)
        h.create_dataset("bar_active", data=BA)
        h.create_dataset("sensors", data=np.array([s.encode() for s in sensors]))
        h.attrs["d_act"] = D_ACT
        h.attrs["mount_z"] = MOUNT_Z
        h.attrs["runs"] = json.dumps([str(p) for p in args.runs])
    nz = float(np.mean(np.linalg.norm(Y, axis=1) > 1e-6))
    print(f"wrote {args.out}  N={N}  samples-with-label: {nz:.2f}  "
          f"close-encounter rate: {n_close / N:.2f}")


if __name__ == "__main__":
    main()
