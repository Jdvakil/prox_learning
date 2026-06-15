"""Headless machinery for the standalone Safety-CVAE *ablation* benchmark.

This is the shared library behind ``safety_ablation_eval.py``. It turns the single-rollout
demo idea (``safety_flinch_demo`` / ``safety_sphere_demo``: march an obstacle at the arm and
let the skin-only head flinch) into a batched, controlled, **closed-loop** evaluator that can
attribute avoidance to the learned proximity signal.

Why this exists: the headline "89%" for the safety head is the *offline direction-cosine* on
the synthetic sweep validation set, not a closed-loop avoidance rate. Here we generate fixed
**collision-course** encounters (with the arm frozen the obstacle WOULD hit a link), replay
each under several ablation conditions that share the same scenario seed, and record whether
the arm actually stayed contact-free plus continuous quality metrics. The control conditions
(``zero`` no-skin floor, ``shuffle`` spatially-wrong, sensor-group drops, ``oracle`` analytic
ceiling) are what make the avoid-rate a *result* rather than a bare number.

Design choices, kept faithful to the demos:
  * Residual reflex, leaky integrator (README S6 / ``safety_orbit_demo``):
        correction += (gain * dq - decay * correction) * DT ; clip(+/- max_dev)
  * Per-frame **baseline subtraction**: dq = (head(skin) - head(skin_obstacle_parked)) / scale,
    so static clutter (hood walls, the arm's own links) is cancelled and only the obstacle's
    marginal push drives motion (exactly as the react/orbit demos do).
  * True surface clearance via ``mujoco.mj_geomDistance`` between the obstacle geom and the
    Franka collision capsules (same approach as ``viz_peg_forest``); the SAME function defines
    the collision-course filter and the avoid metric, so the comparison is internally
    consistent regardless of margin.

Headless on purpose: depends only on ``safety_sweep`` (foxglove-free) + mujoco + numpy + the
torch head; no Foxglove / OpenCV. Render-then-ablate where possible; renders cannot be shared
*across* conditions because each condition's closed loop diverges in q(t).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

import safety_sweep as sw  # constants, build_model, load_postures, analytic_retreat (foxglove-free)

NS = sw.NS
FPS = 30
DT = 1.0 / FPS

# default fumehood aperture (the demos' fixed clear opening)
AP_W, AP_H = 0.675, 0.535

# obstacle approach geometry
D_FAR = 0.40              # obstacle start distance along the sensor view ray (m)
PEN_TARGET = 0.02         # collision-course: drive the obstacle this far PAST the link surface (m)
QUIET_CLEAR = 0.30        # quiet set: obstacle's closest approach stays this far OUTSIDE the arm (m)
D_ACT = sw.D_ACT          # 0.18 m repulsion band (head ~silent beyond it)

# sensor-group membership for the "which sensors matter" ablations
WRIST_PREFIXES = ("link6", "link7", "gripper")
FOREARM_PREFIXES = ("link4", "link5")
LINK_TARGETS = ("link2", "link4", "link5", "link6")   # links an obstacle is aimed at

# residual-controller defaults (the react/orbit demo values)
GAIN, DECAY, MAX_DEV, EMA = 3.5, 2.2, 0.30, 0.0


# --------------------------------------------------------------------------- small helpers
def render_all(model, data, rd, opt, sensors) -> np.ndarray:
    """(S, 8, 8) planar-z depths for every skin sensor (same as the demos' render_all)."""
    depths = np.zeros((len(sensors), 8, 8), np.float32)
    for si, s in enumerate(sensors):
        rd.update_scene(data, camera=f"{NS}{s}", scene_option=opt)
        depths[si] = rd.render()
    return depths


def apply_aperture(data, mid, ap_w, ap_h):
    """Pose the mocap sash + jambs to the given clear opening (mirrors the demos)."""
    data.mocap_pos[mid["sash"]] = [sw.TUBE_X0, 0.0, sw.BENCH_Z + ap_h + 0.025]
    data.mocap_pos[mid["jamb_l"]] = [sw.TUBE_X0, ap_w / 2 + 0.18, sw.BENCH_Z + 0.20]
    data.mocap_pos[mid["jamb_r"]] = [sw.TUBE_X0, -ap_w / 2 - 0.18, sw.BENCH_Z + 0.20]


def park_obstacles(data, mid):
    for n in (*sw.BARS, "sphere"):
        data.mocap_pos[mid[n]] = sw.PARK


def set_arm(data, arm_qadr, q):
    for adr, val in zip(arm_qadr, q):
        data.qpos[adr] = float(val)


def set_grip(data, finger_qadr, g):
    for adr in finger_qadr:
        data.qpos[adr] = float(g)


def arm_collision_geoms(model):
    """Geom ids of the Franka collision capsules + gripper pads (clearance targets).

    Prefer the named ``fr3_link{k}_collision`` / ``gripper/*_pad*`` geoms (as used by
    viz_peg_forest); fall back to every non-cosmetic geom on an ``fr3_link*`` body.
    """
    named = [f"{NS}fr3_link{k}_collision" for k in range(8)] + [
        f"{NS}gripper/{p}" for p in ("left_pad1", "left_pad2", "right_pad1", "right_pad2")]
    gids = []
    for n in named:
        try:
            gids.append(model.geom(n).id)
        except KeyError:
            pass
    if gids:
        return gids
    for gid in range(model.ngeom):                       # fallback: any arm geom but the skin (group 2)
        bn = model.body(model.geom_bodyid[gid]).name
        if "fr3_link" in bn and model.geom_group[gid] != 2:
            gids.append(gid)
    return gids


def obstacle_geom_id(model, body_name):
    return int(model.geom(model.body(body_name).geomadr[0]).id)


_HAS_GEOMDIST = hasattr(mujoco, "mj_geomDistance")


def link_clearance(model, data, obs_gid, arm_gids, fromto=None, distmax=2.0) -> float:
    """Min surface distance (m) from the obstacle geom to any arm collision geom.

    Negative => penetration. Uses ``mj_geomDistance`` (mujoco >= 3.1, already used in this
    repo). Falls back to centre-to-centre distance minus bounding radii if unavailable.
    """
    if _HAS_GEOMDIST:
        if fromto is None:
            fromto = np.zeros(6)
        return float(min(mujoco.mj_geomDistance(model, data, obs_gid, ag, distmax, fromto)
                         for ag in arm_gids))
    # crude fallback (no exact surface query available)
    oc = data.geom_xpos[obs_gid]
    o_r = float(np.max(model.geom_size[obs_gid]))
    best = np.inf
    for ag in arm_gids:
        a_r = float(np.max(model.geom_size[ag]))
        best = min(best, float(np.linalg.norm(oc - data.geom_xpos[ag])) - o_r - a_r)
    return best


def chord_bow(xy: np.ndarray) -> float:
    """Max perpendicular deviation (m) of a 2D path from the straight chord between its
    endpoints -- how much the executed TCP path bowed out (cf. analyze_obstacle_dataset)."""
    if xy.shape[0] < 3:
        return 0.0
    a, b = xy[0], xy[-1]
    ab = b - a
    L = float(np.linalg.norm(ab))
    if L < 1e-6:
        return float(np.linalg.norm(xy - a, axis=1).max())
    n = np.array([-ab[1], ab[0]]) / L
    return float(np.abs((xy - a) @ n).max())


def frame_boxes(data, mid, ap_w, ap_h):
    """Analytic scene boxes (static fumehood + posed sash/jambs) for oracle self-hit
    rejection -- identical set to safety_sweep.main; the obstacle box is appended per frame."""
    boxes = [(np.asarray(c, float), np.asarray(h, float)) for c, h in sw.STATIC_BOXES.values()]
    boxes.append((np.asarray(data.mocap_pos[mid["sash"]], float), np.asarray(sw.SASH_HALF)))
    boxes.append((np.asarray(data.mocap_pos[mid["jamb_l"]], float), np.asarray(sw.JAMB_HALF)))
    boxes.append((np.asarray(data.mocap_pos[mid["jamb_r"]], float), np.asarray(sw.JAMB_HALF)))
    return boxes


def obstacle_spec(obstacle: str, radius: float | None = None):
    """(mocap body name, half-extents(3,)) for the requested obstacle shape."""
    if obstacle == "bar":
        return "bar_m", np.asarray(sw.BARS["bar_m"], float)
    if obstacle == "sphere":
        r = float(radius if radius is not None else sw.SPHERE_R)
        return "sphere", np.asarray([r, r, r], float)
    raise ValueError(f"unknown obstacle {obstacle!r} (use 'bar' or 'sphere')")


# --------------------------------------------------------------------------- context
@dataclass
class Ctx:
    model: object
    data: object
    rd: object
    opt: object
    sensors: list
    cam_ids: dict
    arm_qadr: list
    finger_qadr: list
    arm_dofadr: list
    mid: dict
    arm_gids: list


def build_ctx() -> Ctx:
    """Compile the safety-sweep fumehood + skinned FR3 and cache everything the loop needs.

    The depth renderer holds an EGL context, so build the Ctx in the same process that runs
    the rollouts.
    """
    model = sw.build_model()
    data = mujoco.MjData(model)
    sensors = sorted(model.camera(i).name.removeprefix(NS) for i in range(model.ncam)
                     if "_sensor_" in model.camera(i).name)
    cam_ids = {s: model.camera(f"{NS}{s}").id for s in sensors}
    arm_qadr = [model.joint(f"{NS}fr3_joint{i}").qposadr[0] for i in range(1, 8)]
    arm_dofadr = [model.joint(f"{NS}fr3_joint{i}").dofadr[0] for i in range(1, 8)]
    jnames = [model.joint(i).name for i in range(model.njnt)]
    finger_qadr = [model.joint(f"{NS}gripper/{n}").qposadr[0]
                   for n in ("left_driver_joint", "right_driver_joint")
                   if f"{NS}gripper/{n}" in jnames]
    mid = {n: int(model.body_mocapid[model.body(n).id])
           for n in ("sash", "jamb_l", "jamb_r", "sphere", *sw.BARS, f"{NS}base")}
    arm_gids = arm_collision_geoms(model)
    if not arm_gids:
        raise RuntimeError("no arm collision geoms found - cannot measure clearance")
    rd = mujoco.Renderer(model, 8, 8)
    rd.enable_depth_rendering()
    rd.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[2] = 0     # hide cosmetic skin from the depth render, exactly like datagen
    return Ctx(model, data, rd, opt, sensors, cam_ids, arm_qadr, finger_qadr,
               arm_dofadr, mid, arm_gids)


# --------------------------------------------------------------------------- scenarios
@dataclass
class Scenario:
    seed: int
    q0: np.ndarray
    grip: float
    base: np.ndarray
    ap_w: float
    ap_h: float
    obstacle: str          # "bar" | "sphere"
    obstacle_name: str     # mocap body driven (bar_m | sphere)
    obstacle_half: np.ndarray
    target_sensor: str
    path: np.ndarray       # (T, 3) world centres; rows of NaN = parked
    quiet: bool = False
    meta: dict = field(default_factory=dict)


def _approach_schedule(pos, fwd, d_end):
    """Obstacle centre = pos + fwd * d. March in d_far->d_end (0.12 m/s), dwell 1.2 s,
    retreat d_end->d_far (0.25 m/s), short parked tail. Same cadence as bar_axis_schedule."""
    span = max(D_FAR - d_end, 1e-3)
    in_n = max(4, int(span / (0.12 * DT)))
    out_n = max(4, int(span / (0.25 * DT)))
    ds_in = np.linspace(D_FAR, d_end, in_n)
    ds_out = np.linspace(d_end, D_FAR, out_n)
    legs = [pos[None, :] + np.outer(ds_in, fwd),
            np.tile(pos + fwd * d_end, (int(1.2 * FPS), 1)),
            pos[None, :] + np.outer(ds_out, fwd),
            np.full((int(0.5 * FPS), 3), np.nan)]
    return np.concatenate(legs, 0)


def _sample_one(ctx: Ctx, rng, q_all, grip_all, base_all, obstacle, obs_name, obs_half,
                obs_gid, ap_w, ap_h, quiet, tries=60) -> Scenario | None:
    """One collision-course (or quiet) encounter, or None if no good aim was found."""
    model, data, rd, opt = ctx.model, ctx.data, ctx.rd, ctx.opt
    sensors, cam_ids, arm_gids = ctx.sensors, ctx.cam_ids, ctx.arm_gids
    hand_bid = model.body(f"{NS}fr3_link7").id
    fromto = np.zeros(6)
    target_clr = QUIET_CLEAR if quiet else -PEN_TARGET
    ds_probe = np.linspace(0.0, D_FAR, 41)
    best = None        # (abs error to target, scenario kwargs) - kept as a fallback

    for _ in range(tries):
        row = int(rng.integers(0, len(q_all)))
        data.mocap_pos[ctx.mid[f"{NS}base"]] = base_all[row][:3]
        data.mocap_quat[ctx.mid[f"{NS}base"]] = base_all[row][3:7]
        apply_aperture(data, ctx.mid, ap_w, ap_h)
        park_obstacles(data, ctx.mid)
        set_arm(data, ctx.arm_qadr, q_all[row])
        set_grip(data, ctx.finger_qadr, grip_all[row])
        mujoco.mj_forward(model, data)
        if float(data.xpos[hand_bid][0]) < 0.38:          # must reach forward (skin exposed)
            continue

        depths = render_all(model, data, rd, opt, sensors)
        rest_min = np.array([float(depths[i][depths[i] >= 0.005].min())
                             if (depths[i] >= 0.005).any() else np.inf
                             for i in range(len(sensors))])

        # aim at the most-exposed sensor of a randomly chosen target link
        link = LINK_TARGETS[int(rng.integers(0, len(LINK_TARGETS)))]
        cand = [i for i in range(len(sensors))
                if sensors[i].startswith(link) and np.isfinite(rest_min[i])]
        if not cand:
            continue
        si = max(cand, key=lambda j: rest_min[j])
        cid = cam_ids[sensors[si]]
        pos = data.cam_xpos[cid].copy()
        fwd = -data.cam_xmat[cid].reshape(3, 3)[:, 2]     # outward view ray; smaller d = closer to arm

        # sweep the obstacle in along the ray and read true clearance at each depth
        clrs = np.empty(len(ds_probe))
        for k, d in enumerate(ds_probe):
            data.mocap_pos[ctx.mid[obs_name]] = pos + fwd * d
            mujoco.mj_forward(model, data)
            clrs[k] = link_clearance(model, data, obs_gid, arm_gids, fromto)
        park_obstacles(data, ctx.mid)
        mujoco.mj_forward(model, data)

        if not quiet and clrs.min() > target_clr:
            continue   # ray never penetrates -> not a real test; try another aim

        k = int(np.argmin(np.abs(clrs - target_clr)))
        d_end = float(ds_probe[k])
        err = float(abs(clrs[k] - target_clr))
        kw = dict(seed=0,   # real per-scenario seed assigned by make_scenarios
                  q0=q_all[row].copy(), grip=float(grip_all[row]), base=base_all[row].copy(),
                  ap_w=ap_w, ap_h=ap_h, obstacle=obstacle, obstacle_name=obs_name,
                  obstacle_half=obs_half.copy(), target_sensor=sensors[si],
                  path=_approach_schedule(pos, fwd, d_end), quiet=quiet,
                  meta=dict(link=link, d_end=d_end, end_clear=float(clrs[k]), row=row))
        if best is None or err < best[0]:
            best = (err, kw)
        if err < (0.05 if quiet else 0.02):
            break

    return Scenario(**best[1]) if best is not None else None


def make_scenarios(ctx: Ctx, runs, n, seed, obstacle="bar", radius=None,
                   ap_w=AP_W, ap_h=AP_H, quiet=False) -> list[Scenario]:
    """``n`` deterministic encounters (seeded by ``seed`` + index). Collision-course unless
    ``quiet`` (a far-approach control set for measuring false retreat)."""
    q_all, grip_all, base_all, _ = sw.load_postures([Path(p) for p in runs])
    obs_name, obs_half = obstacle_spec(obstacle, radius)
    obs_gid = obstacle_geom_id(ctx.model, obs_name)
    out = []
    for idx in range(n):
        rng = np.random.default_rng(seed * 100_003 + (1_000_000 if quiet else 0) + idx)
        sc = _sample_one(ctx, rng, q_all, grip_all, base_all, obstacle, obs_name, obs_half,
                         obs_gid, ap_w, ap_h, quiet)
        if sc is None:
            print(f"[scenario {idx}] no valid aim found - skipped")
            continue
        sc.seed = seed * 100_003 + idx
        out.append(sc)
    kind = "quiet" if quiet else "collision-course"
    print(f"generated {len(out)}/{n} {kind} {obstacle} scenarios")
    return out


# --------------------------------------------------------------------------- conditions
def _startswith_mask(sensors, prefixes) -> np.ndarray:
    return np.array([s.startswith(prefixes) for s in sensors], bool)


def make_spec(condition: str, sensors, rng) -> dict:
    """Parse a condition name into an input transform spec.

    full | zero | oracle | shuffle | drop_wrist | wrist_only | forearm_only | noise<sigma_m>
    """
    spec = {"kind": "full", "perm": None, "keep": None, "noise": 0.0}
    if condition in ("full", "zero", "oracle"):
        spec["kind"] = condition
    elif condition == "shuffle":
        spec["perm"] = rng.permutation(len(sensors))
    elif condition == "drop_wrist":
        spec["keep"] = ~_startswith_mask(sensors, WRIST_PREFIXES)
    elif condition == "wrist_only":
        spec["keep"] = _startswith_mask(sensors, WRIST_PREFIXES)
    elif condition == "forearm_only":
        spec["keep"] = _startswith_mask(sensors, FOREARM_PREFIXES)
    elif condition.startswith("noise"):
        spec["noise"] = float(condition[len("noise"):])
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return spec


def transform_depths(depths, spec, rng) -> np.ndarray:
    """Apply a condition's input transform to raw (S, 8, 8) depths.

    Permute sensors (shuffle), drop a sensor group (set to 0 -> closeness 0 after featurize),
    and/or add Gaussian depth noise. Identity for 'full'.
    """
    out = depths
    if spec["perm"] is not None:
        out = out[spec["perm"]]
    if spec["keep"] is not None:
        out = out.copy()
        out[~spec["keep"]] = 0.0
    if spec["noise"] > 0:
        out = np.clip(out + rng.normal(0.0, spec["noise"], out.shape).astype(out.dtype), 0.0, None)
    return out


def _cond_rng(seed: int, condition: str) -> np.random.Generator:
    """Deterministic per-(scenario, condition) RNG (no reliance on salted hash())."""
    h = hashlib.blake2b(f"{seed}|{condition}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little"))


# --------------------------------------------------------------------------- rollout
def rollout(ctx: Ctx, scenario: Scenario, condition: str, head, scale: float,
            gain=GAIN, decay=DECAY, max_dev=MAX_DEV, ema=EMA, margin=0.0) -> dict:
    """Run one closed-loop encounter under one ablation condition. Returns metrics.

    head : callable raw (S,8,8) depths -> (7,) retreat (already x label_scale), or None for
           the zero/oracle conditions.
    """
    model, data, rd, opt = ctx.model, ctx.data, ctx.rd, ctx.opt
    sensors, cam_ids, arm_gids = ctx.sensors, ctx.cam_ids, ctx.arm_gids
    rng = _cond_rng(scenario.seed, condition)
    spec = make_spec(condition, sensors, rng)
    can_oracle = scenario.obstacle == "bar"          # analytic self-hit rejection assumes box obstacle
    obs_name = scenario.obstacle_name
    obs_gid = obstacle_geom_id(model, obs_name)
    hand_bid = model.body(f"{NS}fr3_link7").id
    jacp = np.zeros((3, model.nv))
    fromto = np.zeros(6)

    # static scene
    data.mocap_pos[ctx.mid[f"{NS}base"]] = scenario.base[:3]
    data.mocap_quat[ctx.mid[f"{NS}base"]] = scenario.base[3:7]
    apply_aperture(data, ctx.mid, scenario.ap_w, scenario.ap_h)
    park_obstacles(data, ctx.mid)
    set_arm(data, ctx.arm_qadr, scenario.q0)
    set_grip(data, ctx.finger_qadr, scenario.grip)
    mujoco.mj_forward(model, data)
    base_boxes = frame_boxes(data, ctx.mid, scenario.ap_w, scenario.ap_h)

    path = scenario.path
    T = len(path)
    correction = np.zeros(7)
    dq = np.zeros(7)
    min_clear = np.inf
    contact = False
    contact_step = -1
    peak_dev = 0.0
    tcp_xy = []
    cos_list = []

    for t in range(T):
        active = bool(np.isfinite(path[t]).all())
        data.mocap_pos[ctx.mid[obs_name]] = path[t] if active else sw.PARK
        set_arm(data, ctx.arm_qadr, scenario.q0 + correction)
        mujoco.mj_forward(model, data)
        depths = render_all(model, data, rd, opt, sensors)

        if active:
            clr = link_clearance(model, data, obs_gid, arm_gids, fromto)
            min_clear = min(min_clear, clr)
            if clr < margin and not contact:
                contact, contact_step = True, t

        # per-frame baseline: same arm pose, obstacle parked (isolates the obstacle's push)
        data.mocap_pos[ctx.mid[obs_name]] = sw.PARK
        mujoco.mj_forward(model, data)
        rest_depths = render_all(model, data, rd, opt, sensors)
        if active:
            data.mocap_pos[ctx.mid[obs_name]] = path[t]   # restore (cam pose is arm-only, no re-forward needed)

        # analytic oracle, baseline-subtracted exactly like the learned head: the obstacle's
        # marginal analytic push = retreat(with obstacle) - retreat(obstacle parked). Used both
        # for the oracle condition and as the per-frame retreat-cosine reference.
        oracle_dq = None
        if can_oracle:
            boxes_w = base_boxes + ([(np.asarray(path[t], float), scenario.obstacle_half)]
                                    if active else [])
            o_w, _ = sw.analytic_retreat(model, data, sensors, cam_ids, depths,
                                         boxes_w, ctx.arm_dofadr, jacp)
            o_r, _ = sw.analytic_retreat(model, data, sensors, cam_ids, rest_depths,
                                         base_boxes, ctx.arm_dofadr, jacp)
            oracle_dq = (o_w - o_r).astype(np.float32)

        if spec["kind"] == "zero":
            dq_raw = np.zeros(7, np.float32)
        elif spec["kind"] == "oracle":
            dq_raw = (oracle_dq / max(scale, 1e-6)).astype(np.float32) \
                if oracle_dq is not None else np.zeros(7, np.float32)
        else:
            d_t = transform_depths(depths, spec, rng)
            r_t = transform_depths(rest_depths, spec, rng)
            dq_raw = ((head(d_t) - head(r_t)) / max(scale, 1e-6)).astype(np.float32)

        if (oracle_dq is not None and np.linalg.norm(oracle_dq) > 1e-6
                and np.linalg.norm(dq_raw) > 1e-9):
            cos_list.append(float(np.dot(dq_raw, oracle_dq)
                                  / (np.linalg.norm(dq_raw) * np.linalg.norm(oracle_dq))))

        dq = ema * dq + (1.0 - ema) * dq_raw
        correction = np.clip(correction + (gain * dq - decay * correction) * DT,
                             -max_dev, max_dev)
        peak_dev = max(peak_dev, float(np.linalg.norm(correction)))
        tcp_xy.append(data.xpos[hand_bid][:2].copy())

    return dict(
        min_clear=float(min_clear),
        avoid=bool(min_clear >= margin),
        contact=bool(contact),
        contact_step=int(contact_step),
        peak_dev=float(peak_dev),
        bow=float(chord_bow(np.asarray(tcp_xy))),
        mean_cos=float(np.mean(cos_list)) if cos_list else float("nan"),
        n_frames=int(T),
    )
