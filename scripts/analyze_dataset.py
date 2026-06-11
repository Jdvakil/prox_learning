"""Full dataset analysis for a MolmoSpaces franka_skin pick-and-place datagen run.

Sweeps every trajectory in a datagen output directory and produces dataset-level
plots + tables covering:

  * kinematics       — arm/gripper joint positions & velocities (envelopes, ranges)
  * trajectories     — TCP world-frame paths, lift arcs, approach distance profiles
  * commanded action — per-joint command increments, gripper command timing
  * action noise     — injected high-frequency jitter (command vs realized 2nd-diff),
                       per-joint tracking residual, estimated injected-noise std
  * proximity / skin — per-sensor & per-link activation, nearest-surface depth dist.,
                       sensor x phase activation map, proximity-active fraction
  * task / collision — episode length, success, env-collision probability, per-phase
                       collision probability, contacts-over-time

Reads arm/gripper qpos/qvel directly from env_states/articulations/panda (exact,
no JSON decode); decodes only the commanded actions from the uint8-JSON rows.

Usage:
    python scripts/analyze_dataset.py \
        --root assets/prox_learning_data/FrankaSkinProxNecessityPilotConfig/20260605_014315 \
        [--near_m 0.15] [--out diagnostics_output/dataset_analysis]
    python scripts/analyze_dataset.py --glob '<dir>/**/house_*/trajectories_batch_*.h5'
"""
from __future__ import annotations

import argparse
import csv
import glob as globlib
import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- constants
PHASE_NAMES = [
    "unknown", "gripper-open", "pregrasp", "grasp", "gripper-close",
    "lift", "preplace", "place", "retreat", "go_home",
]
PHASE_COLORS = plt.cm.tab10(np.linspace(0, 1, len(PHASE_NAMES)))
ARM_JOINTS = [f"j{i+1}" for i in range(7)]
PROX_MIN_VALID = 0.05   # SPAD valid range lower bound (m)
PROX_MAX_VALID = 4.0    # SPAD valid range upper bound (m); >this == no return
LINKS = ["link2", "link3", "link5", "link6"]


# ----------------------------------------------------------------------------- helpers
def decode_json_row(row):
    b = bytes(np.asarray(row, dtype=np.uint8)).split(b"\x00", 1)[0]
    if not b:
        return None
    try:
        return json.loads(b.decode("utf-8"))
    except Exception:
        return None


def decode_field(group, key, subkey):
    """Stack a sub-list (arm/gripper/...) from a (T,N) JSON-bytes dataset -> (T,d)."""
    ds = group[key][:]
    rows = [decode_json_row(r) or {} for r in ds]
    out = [np.asarray(r.get(subkey, []), dtype=np.float64) for r in rows]
    width = max((len(x) for x in out), default=0)
    arr = np.full((len(out), width), np.nan)
    for i, x in enumerate(out):
        arr[i, : len(x)] = x
    return arr


def task_info_series(traj, field):
    ds = traj["obs/extra/task_info"][:]
    vals = []
    for r in ds:
        d = decode_json_row(r) or {}
        v = d.get(field)
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(np.nan)
    return np.asarray(vals)


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tcp_to_world(tcp_pose, base_pose):
    T = tcp_pose.shape[0]
    out = np.zeros((T, 3))
    for i in range(T):
        R = quat_to_R(base_pose[i, 3:7])
        out[i] = base_pose[i, :3] + R @ tcp_pose[i, :3]
    return out


def moving_avg(x, w=7):
    """Centered moving average along axis 0 (edge-replicated), for denoising commands."""
    if x.shape[0] < w:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(w) / w
    return np.stack([np.convolve(xp[:, j], ker, mode="valid") for j in range(x.shape[1])], axis=1)


def sensor_sort_key(k):
    return (int(k.split("_")[0][4:]), int(k.split("_")[-1]))


# ----------------------------------------------------------------------------- per-traj extraction
def extract_traj(t, near_m):
    """Reduce one trajectory to compact arrays/scalars (no raw proximity retained)."""
    T = int(t["rewards"].shape[0])

    panda = t["env_states/articulations/panda"][:]           # (T,31) float32
    qpos_arm = panda[:, 0:7]
    grip_q = panda[:, 7:9]
    qvel_arm = panda[:, 9:16]

    act_arm = decode_field(t["actions"], "joint_pos", "arm")      # (T,7) commanded targets
    act_grip = decode_field(t["actions"], "joint_pos", "gripper")  # (T,1)

    tcp = t["obs/extra/tcp_pose"][:]
    base_pose = t["obs/extra/robot_base_pose"][:]
    tcp_w = tcp_to_world(tcp, base_pose)
    obj_start = t["obs/extra/obj_start"][:][0, :3]
    obj_end = t["obs/extra/obj_end"][:][0, :3]
    tcp_to_obj = np.linalg.norm(tcp_w - obj_start[None, :], axis=1)

    phase = t["obs/extra/policy_phase"][:].astype(int)
    success = bool(t["success"][:][-1])

    # proximity: per-sensor nearest valid depth (T, 29) — stream, drop raw
    prox_keys = sorted(t["obs/proximity"].keys(), key=sensor_sort_key)
    n_sensors = len(prox_keys)
    nearest = np.full((T, n_sensors), np.nan)
    mean_depth = np.zeros((T, n_sensors))
    for j, k in enumerate(prox_keys):
        px = t["obs/proximity"][k][:].mean(axis=1).reshape(T, -1)   # mean over 4 sub-samples -> (T,64)
        valid = np.where((px > PROX_MIN_VALID) & (px <= PROX_MAX_VALID), px, np.nan)
        with np.errstate(all="ignore"):
            nearest[:, j] = np.nanmin(valid, axis=1)
        mean_depth[:, j] = np.clip(px, 0, PROX_MAX_VALID).mean(axis=1)
    active = np.isfinite(nearest) & (nearest < near_m)              # (T,29) bool
    closest = np.where(np.isfinite(nearest).any(1), np.nanmin(np.where(np.isfinite(nearest), nearest, np.inf), 1), np.nan)
    prox_active_step = (active.any(axis=1))                          # any sensor active this step

    # collision
    scene = json.loads(np.asarray(t["obs_scene"]).item())
    coll = scene.get("collision_metrics", {})
    psc = np.asarray(coll.get("per_step_contacts", [0] * T), dtype=float)
    if psc.shape[0] != T:
        psc = np.resize(psc, T)
    env_collision = psc > 0
    robot_contact = task_info_series(t, "robot_contact")

    # action noise signature (arm)
    cmd_incr = np.diff(act_arm, axis=0)                # (T-1,7) command step
    cmd_jit = np.diff(act_arm, n=2, axis=0)            # (T-2,7) command 2nd diff (jitter)
    real_jit = np.diff(qpos_arm, n=2, axis=0)          # realized 2nd diff
    resid = qpos_arm[1:] - act_arm[:-1]                # tracking residual (realized vs prev cmd)
    injected = act_arm - moving_avg(act_arm, 7)        # high-freq part of command ~ injected noise

    return dict(
        T=T, success=success,
        task_description=scene.get("task_description", "?"),
        object_name=scene.get("object_name", "?"),
        receptacle=scene.get("place_receptacle_name", "?"),
        qpos_arm=qpos_arm, qvel_arm=qvel_arm, grip_q=grip_q,
        act_arm=act_arm, act_grip=act_grip,
        tcp_w=tcp_w, obj_start=obj_start, obj_end=obj_end, tcp_to_obj=tcp_to_obj,
        phase=phase,
        nearest=nearest, mean_depth=mean_depth, active=active, closest=closest,
        prox_active_step=prox_active_step, prox_keys=prox_keys,
        psc=psc, env_collision=env_collision, robot_contact=robot_contact,
        cmd_incr=cmd_incr, cmd_jit=cmd_jit, real_jit=real_jit, resid=resid, injected=injected,
    )


# ----------------------------------------------------------------------------- main
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None, help="datagen output dir (recursively globbed for h5)")
    p.add_argument("--glob", default=None)
    p.add_argument("--near_m", type=float, default=0.15,
                   help="a skin sensor is 'active' when it reads a surface closer than this (m)")
    p.add_argument("--out", default="diagnostics_output/dataset_analysis")
    args = p.parse_args()

    if args.glob:
        paths = sorted(globlib.glob(args.glob, recursive=True))
    elif args.root:
        paths = sorted(globlib.glob(str(Path(args.root) / "**/house_*/trajectories_batch_*.h5"), recursive=True))
    else:
        print("provide --root or --glob"); return 1
    if not paths:
        print("no h5 files matched"); return 1

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"analyzing {len(paths)} h5 files -> {out}")

    trajs = []           # list of extracted dicts
    rows = []            # per-traj scalar table
    for pth in paths:
        house = Path(pth).parent.name
        try:
            f = h5py.File(pth, "r")
        except Exception as e:
            print(f"[skip] {pth}: {e}"); continue
        for tk in f.keys():
            try:
                d = extract_traj(f[tk], args.near_m)
            except Exception as e:
                print(f"[skip] {pth}:{tk}: {e}"); continue
            d["house"] = house; d["traj"] = tk; d["file"] = pth
            trajs.append(d)
            rows.append(dict(
                house=house, traj=tk, T=d["T"], success=int(d["success"]),
                object=d["object_name"], receptacle=d["receptacle"],
                tcp_reach_min_m=float(d["tcp_to_obj"].min()),
                tcp_z_range_m=float(d["tcp_w"][:, 2].max() - d["tcp_w"][:, 2].min()),
                qvel_arm_max=float(np.nanmax(np.abs(d["qvel_arm"]))),
                prox_active_frac=float(d["prox_active_step"].mean()),
                n_active_sensors_mean=float(d["active"].sum(1).mean()),
                nearest_surface_m=float(np.nanmin(d["closest"])) if np.isfinite(d["closest"]).any() else np.nan,
                env_collision_prob=float(d["env_collision"].mean()),
                n_collision_steps=int(d["env_collision"].sum()),
                any_contact_prob=float(np.nanmean(d["robot_contact"] > 0)) if np.isfinite(d["robot_contact"]).any() else np.nan,
                cmd_jitter_std=float(np.nanstd(d["cmd_jit"])),
                injected_noise_std=float(np.nanstd(d["injected"])),
                tracking_resid_std=float(np.nanstd(d["resid"])),
            ))
        f.close()

    n = len(trajs)
    if n == 0:
        print("no trajectories analyzed"); return 1
    print(f"extracted {n} trajectories")

    plots = []

    def save(fig, name):
        fig.savefig(out / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        plots.append(name)

    # ============================================================ KINEMATICS
    # 01 arm qpos distribution (per joint, all steps all trajs) + per-joint limits
    all_qpos = np.concatenate([d["qpos_arm"] for d in trajs], axis=0)
    all_qvel = np.concatenate([d["qvel_arm"] for d in trajs], axis=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    a1.violinplot([all_qpos[:, j] for j in range(7)], showmeans=True, showextrema=True)
    a1.set(title=f"Arm joint POSITION distribution (all {n} trajs, {all_qpos.shape[0]} steps)",
           xlabel="joint", ylabel="rad", xticks=range(1, 8), xticklabels=ARM_JOINTS)
    a1.grid(alpha=0.3)
    a2.violinplot([all_qvel[:, j] for j in range(7)], showmeans=True, showextrema=True)
    a2.set(title="Arm joint VELOCITY distribution", xlabel="joint", ylabel="rad/s",
           xticks=range(1, 8), xticklabels=ARM_JOINTS)
    a2.grid(alpha=0.3)
    save(fig, "01_joint_pos_vel_distributions.png")

    # 02 per-joint qpos timeseries overlaid (faint, all trajs) — coverage over the motion
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    axes = axes.ravel()
    for j in range(7):
        ax = axes[j]
        for d in trajs:
            ax.plot(np.linspace(0, 1, d["T"]), d["qpos_arm"][:, j], lw=0.6, alpha=0.25, color="C0")
        ax.set_title(f"arm {ARM_JOINTS[j]} qpos", fontsize=9)
        ax.set_xlabel("normalized time"); ax.set_ylabel("rad"); ax.grid(alpha=0.3)
    axes[7].axis("off")
    fig.suptitle("Per-joint position trajectories (all trajectories, time-normalized)", y=1.00)
    save(fig, "02_qpos_overlay.png")

    # 03 per-joint qvel timeseries overlaid
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    axes = axes.ravel()
    for j in range(7):
        ax = axes[j]
        for d in trajs:
            ax.plot(np.linspace(0, 1, d["T"]), d["qvel_arm"][:, j], lw=0.6, alpha=0.25, color="C3")
        ax.set_title(f"arm {ARM_JOINTS[j]} qvel", fontsize=9)
        ax.set_xlabel("normalized time"); ax.set_ylabel("rad/s"); ax.grid(alpha=0.3)
    axes[7].axis("off")
    fig.suptitle("Per-joint velocity trajectories (all trajectories, time-normalized)", y=1.00)
    save(fig, "03_qvel_overlay.png")

    # ============================================================ TRAJECTORIES
    # 04 TCP world paths — top-down (x-y) and side (x-z), all trajs, + objects
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
    for d in trajs:
        a1.plot(d["tcp_w"][:, 0], d["tcp_w"][:, 1], lw=0.8, alpha=0.5)
        a1.scatter(*d["obj_start"][:2], c="red", s=25, marker="*", zorder=3)
        a1.scatter(*d["obj_end"][:2], c="green", s=25, marker="X", zorder=3)
        a2.plot(d["tcp_w"][:, 0], d["tcp_w"][:, 2], lw=0.8, alpha=0.5)
        a2.scatter(d["obj_start"][0], d["obj_start"][2], c="red", s=25, marker="*", zorder=3)
    a1.set(title="TCP paths — top-down (x-y, world)", xlabel="x (m)", ylabel="y (m)"); a1.axis("equal"); a1.grid(alpha=0.3)
    a2.set(title="TCP paths — side (x-z, world)  ★=pick  ✕=place", xlabel="x (m)", ylabel="z (m)"); a2.grid(alpha=0.3)
    save(fig, "04_tcp_paths.png")

    # 05 TCP height (z) profiles + tcp-to-object distance profiles, time-normalized
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    for d in trajs:
        tn = np.linspace(0, 1, d["T"])
        a1.plot(tn, d["tcp_w"][:, 2], lw=0.8, alpha=0.4, color="C0")
        a2.plot(tn, d["tcp_to_obj"], lw=0.8, alpha=0.4, color="C4")
    a1.set(title="TCP height z(t) — the lift arc", xlabel="normalized time", ylabel="z (m)"); a1.grid(alpha=0.3)
    a2.set(title="‖TCP − object_start‖ — approach / grasp / place", xlabel="normalized time", ylabel="m"); a2.grid(alpha=0.3)
    save(fig, "05_tcp_height_and_reach.png")

    # ============================================================ ACTION & NOISE
    # 06 per-joint command increment distribution (how big each action step is)
    all_incr = np.concatenate([d["cmd_incr"] for d in trajs], axis=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot([all_incr[:, j] for j in range(7)], showfliers=False, whis=(5, 95))
    ax.set(title="Commanded action increment per joint (Δ target / step)", xlabel="joint",
           ylabel="rad", xticks=range(1, 8), xticklabels=ARM_JOINTS); ax.grid(alpha=0.3)
    save(fig, "06_command_increment.png")

    # 07 ACTION NOISE: per-joint command jitter (2nd diff) vs realized jitter + injected-noise std + tracking residual
    cmd_jit_std = np.array([np.nanstd(np.concatenate([d["cmd_jit"][:, j] for d in trajs])) for j in range(7)])
    real_jit_std = np.array([np.nanstd(np.concatenate([d["real_jit"][:, j] for d in trajs])) for j in range(7)])
    inj_std = np.array([np.nanstd(np.concatenate([d["injected"][:, j] for d in trajs])) for j in range(7)])
    resid_std = np.array([np.nanstd(np.concatenate([d["resid"][:, j] for d in trajs])) for j in range(7)])
    x = np.arange(7); bw = 0.2
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - 1.5 * bw, cmd_jit_std, bw, label="command jitter  std(Δ²cmd)")
    ax.bar(x - 0.5 * bw, real_jit_std, bw, label="realized jitter  std(Δ²qpos)")
    ax.bar(x + 0.5 * bw, inj_std, bw, label="injected-noise std  (cmd − smooth)")
    ax.bar(x + 1.5 * bw, resid_std, bw, label="tracking residual std")
    ax.set(title="Action-noise signature per joint (Gaussian TCP noise ≤1 cm mapped to joints)",
           xlabel="joint", ylabel="rad", xticks=x, xticklabels=ARM_JOINTS)
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, "07_action_noise_signature.png")

    # ============================================================ PROXIMITY / SKIN
    # build canonical sensor order from first traj
    prox_keys = trajs[0]["prox_keys"]
    n_sensors = len(prox_keys)
    # 08 per-sensor activation fraction (mean over trajs of frac steps where active)
    act_frac = np.array([
        np.mean([d["active"][:, j].mean() for d in trajs]) for j in range(n_sensors)
    ])
    link_of = [k.split("_")[0] for k in prox_keys]
    bar_colors = {"link2": "C0", "link3": "C1", "link5": "C2", "link6": "C3"}
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(n_sensors), act_frac, color=[bar_colors[l] for l in link_of])
    ax.axhline(act_frac.mean(), color="k", ls="--", lw=1, label=f"mean {act_frac.mean():.2f}")
    ax.set(title=f"Per-sensor activation fraction (surface < {args.near_m} m), mean over {n} trajs",
           xlabel="sensor", ylabel="fraction of steps active",
           xticks=range(n_sensors), xticklabels=prox_keys)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=bar_colors[l]) for l in LINKS]
    ax.legend(handles + [plt.Line2D([0], [0], color="k", ls="--")], LINKS + [f"mean {act_frac.mean():.2f}"])
    save(fig, "08_sensor_activation_fraction.png")

    # 09 per-LINK activation fraction (aggregate)
    link_act = {}
    for l in LINKS:
        idx = [j for j, k in enumerate(prox_keys) if k.startswith(l)]
        link_act[l] = np.mean([d["active"][:, idx].mean() for d in trajs])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(list(link_act.keys()), list(link_act.values()), color=[bar_colors[l] for l in LINKS])
    ax.set(title="Skin activation fraction by link", ylabel=f"frac steps a sensor reads < {args.near_m} m")
    for i, (l, v) in enumerate(link_act.items()):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    ax.grid(alpha=0.3)
    save(fig, "09_link_activation.png")

    # 10 nearest-surface depth distribution (all valid sensor readings, all trajs)
    all_near = np.concatenate([d["nearest"][np.isfinite(d["nearest"])] for d in trajs])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(all_near, bins=80, range=(0, PROX_MAX_VALID), color="teal", alpha=0.85)
    ax.axvline(args.near_m, color="red", ls="--", label=f"active threshold {args.near_m} m")
    ax.axvline(PROX_MIN_VALID, color="k", ls=":", label=f"near limit {PROX_MIN_VALID} m")
    ax.set(title="Distribution of per-sensor NEAREST valid surface depth (all sensors, all steps)",
           xlabel="nearest depth (m)", ylabel="count"); ax.legend(); ax.grid(alpha=0.3)
    save(fig, "10_nearest_depth_hist.png")

    # 11 sensor x phase activation map (when does each sensor fire?)
    sp = np.zeros((n_sensors, len(PHASE_NAMES)))
    spc = np.zeros(len(PHASE_NAMES))
    for d in trajs:
        for pid in range(len(PHASE_NAMES)):
            m = d["phase"] == pid
            if m.any():
                sp[:, pid] += d["active"][m].sum(0)
                spc[pid] += m.sum()
    with np.errstate(all="ignore"):
        sp_frac = np.where(spc[None, :] > 0, sp / spc[None, :], 0)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(sp_frac, aspect="auto", cmap="magma", vmin=0, vmax=min(1.0, sp_frac.max() * 1.05))
    ax.set(title="Sensor activation fraction by manipulation phase",
           xticks=range(len(PHASE_NAMES)), yticks=range(n_sensors), yticklabels=prox_keys)
    ax.set_xticklabels(PHASE_NAMES, rotation=40, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=6)
    fig.colorbar(im, ax=ax, label="frac of phase-steps sensor is active")
    save(fig, "11_sensor_phase_activation.png")

    # 12 proximity-active fraction per trajectory (distribution + per-traj)
    pf = np.array([d["prox_active_step"].mean() for d in trajs])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 2]})
    a1.hist(pf, bins=15, color="C2", alpha=0.85)
    a1.axvline(pf.mean(), color="k", ls="--", label=f"mean {pf.mean():.2f}")
    a1.axvline(0.8, color="red", ls=":", label="0.8 target")
    a1.set(title="Proximity-active fraction per trajectory", xlabel="frac of steps ≥1 sensor active", ylabel="# trajs")
    a1.legend(); a1.grid(alpha=0.3)
    order = np.argsort(pf)
    a2.bar(range(n), pf[order], color="C2")
    a2.axhline(0.8, color="red", ls=":")
    a2.set(title="Proximity-active fraction (sorted, per traj)", xlabel="trajectory (sorted)", ylabel="frac")
    a2.grid(alpha=0.3)
    save(fig, "12_prox_active_fraction.png")

    # ============================================================ TASK / COLLISION
    # 13 episode length + success
    Ts = np.array([d["T"] for d in trajs])
    succ = np.array([d["success"] for d in trajs])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.5))
    a1.hist(Ts, bins=15, color="C0", alpha=0.85)
    a1.axvline(Ts.mean(), color="k", ls="--", label=f"mean {Ts.mean():.0f}")
    a1.set(title="Episode length distribution", xlabel="policy steps", ylabel="# trajs"); a1.legend(); a1.grid(alpha=0.3)
    a2.bar(["success", "fail"], [succ.sum(), (~succ).sum()], color=["C2", "C3"])
    a2.set(title=f"Outcome ({succ.mean():.0%} success, n={n})", ylabel="# trajs")
    save(fig, "13_episode_length_success.png")

    # 14 env-collision probability per traj + aggregate per-phase collision prob
    ecp = np.array([d["env_collision"].mean() for d in trajs])
    phase_coll_num = np.zeros(len(PHASE_NAMES)); phase_coll_den = np.zeros(len(PHASE_NAMES))
    for d in trajs:
        for pid in range(len(PHASE_NAMES)):
            m = d["phase"] == pid
            if m.any():
                phase_coll_num[pid] += d["env_collision"][m].sum()
                phase_coll_den[pid] += m.sum()
    with np.errstate(all="ignore"):
        phase_coll = np.where(phase_coll_den > 0, phase_coll_num / phase_coll_den, np.nan)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2, 2]})
    a1.bar(range(n), np.sort(ecp)[::-1], color="crimson")
    a1.axhline(ecp.mean(), color="k", ls="--", label=f"mean {ecp.mean():.1%}")
    a1.set(title="Env-collision probability per trajectory (sorted)", xlabel="trajectory", ylabel="P(collision step)")
    a1.legend(); a1.grid(alpha=0.3)
    valid_ph = [i for i in range(len(PHASE_NAMES)) if phase_coll_den[i] > 0]
    a2.bar([PHASE_NAMES[i] for i in valid_ph], [phase_coll[i] for i in valid_ph],
           color=[PHASE_COLORS[i] for i in valid_ph])
    a2.set(title="Env-collision probability by phase (dataset aggregate)", ylabel="P(collision step)")
    plt.setp(a2.get_xticklabels(), rotation=35, ha="right", fontsize=8); a2.grid(alpha=0.3)
    save(fig, "14_collision_probability.png")

    # 15 contacts over normalized time (mean +/- band) + proximity-active over time
    grid = np.linspace(0, 1, 50)
    coll_curves = np.zeros((n, 50)); pa_curves = np.zeros((n, 50))
    for i, d in enumerate(trajs):
        tn = np.linspace(0, 1, d["T"])
        coll_curves[i] = np.interp(grid, tn, d["env_collision"].astype(float))
        pa_curves[i] = np.interp(grid, tn, d["prox_active_step"].astype(float))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(grid, coll_curves.mean(0), color="crimson", lw=2, label="P(env collision)")
    ax.fill_between(grid, coll_curves.mean(0) - coll_curves.std(0), coll_curves.mean(0) + coll_curves.std(0),
                    color="crimson", alpha=0.15)
    ax.plot(grid, pa_curves.mean(0), color="C2", lw=2, label="P(≥1 skin sensor active)")
    ax.fill_between(grid, pa_curves.mean(0) - pa_curves.std(0), pa_curves.mean(0) + pa_curves.std(0),
                    color="C2", alpha=0.15)
    ax.set(title="Collision & proximity-activation probability over normalized time (mean ± std)",
           xlabel="normalized time", ylabel="probability"); ax.legend(); ax.grid(alpha=0.3)
    save(fig, "15_collision_prox_over_time.png")

    # ============================================================ TABLES / REPORT
    cols = ["house", "traj", "T", "success", "object", "receptacle", "tcp_reach_min_m",
            "tcp_z_range_m", "qvel_arm_max", "prox_active_frac", "n_active_sensors_mean",
            "nearest_surface_m", "env_collision_prob", "n_collision_steps", "any_contact_prob",
            "cmd_jitter_std", "injected_noise_std", "tracking_resid_std"]
    with open(out / "per_trajectory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})

    def m(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return float(np.nanmean(v)), float(np.nanstd(v)), float(np.nanmin(v)), float(np.nanmax(v))

    summary = {
        "root": args.root or args.glob, "near_m": args.near_m,
        "n_files": len(paths), "n_trajectories": n,
        "episode_length": {"mean": float(Ts.mean()), "min": int(Ts.min()), "max": int(Ts.max())},
        "success_rate": float(succ.mean()),
        "kinematics": {
            "qpos_arm_range_rad": [float(all_qpos.min()), float(all_qpos.max())],
            "qvel_arm_abs_max_rad_s": float(np.abs(all_qvel).max()),
            "per_joint_qpos_min": all_qpos.min(0).round(3).tolist(),
            "per_joint_qpos_max": all_qpos.max(0).round(3).tolist(),
            "per_joint_qvel_abs_max": np.abs(all_qvel).max(0).round(3).tolist(),
        },
        "action_noise": {
            "per_joint_command_jitter_std": cmd_jit_std.round(4).tolist(),
            "per_joint_realized_jitter_std": real_jit_std.round(4).tolist(),
            "per_joint_injected_noise_std": inj_std.round(4).tolist(),
            "per_joint_tracking_residual_std": resid_std.round(4).tolist(),
        },
        "proximity": {
            "prox_active_frac_mean": float(pf.mean()),
            "prox_active_frac_range": [float(pf.min()), float(pf.max())],
            "frac_trajs_meeting_0.8": float((pf >= 0.8).mean()),
            "mean_active_sensors_per_step": float(np.mean([d["active"].sum(1).mean() for d in trajs])),
            "link_activation": {k: float(v) for k, v in link_act.items()},
            "most_active_sensors": [prox_keys[i] for i in np.argsort(act_frac)[::-1][:5]],
            "least_active_sensors": [prox_keys[i] for i in np.argsort(act_frac)[:5]],
            "global_nearest_surface_m": float(np.nanmin([np.nanmin(d["closest"]) for d in trajs])),
        },
        "collision": {
            "env_collision_prob_mean": float(ecp.mean()),
            "env_collision_prob_max": float(ecp.max()),
            "frac_trajs_with_any_collision": float((ecp > 0).mean()),
            "per_phase_env_collision_prob": {PHASE_NAMES[i]: (float(phase_coll[i]) if phase_coll_den[i] > 0 else None)
                                             for i in range(len(PHASE_NAMES))},
        },
        "aggregates": {k: dict(zip(["mean", "std", "min", "max"], m(k)))
                       for k in ["tcp_reach_min_m", "tcp_z_range_m", "prox_active_frac",
                                 "env_collision_prob", "injected_noise_std", "tracking_resid_std"]},
        "plots": plots,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # markdown report
    pj = lambda arr: ", ".join(f"{ARM_JOINTS[i]}={float(v):.3g}" for i, v in enumerate(arr))
    rep = [
        f"# Dataset analysis — {n} trajectories ({len(paths)} houses)", "",
        f"**Source:** `{args.root or args.glob}`  ",
        f"**near_m (skin-active threshold):** {args.near_m} m", "",
        "## Headline",
        f"- **{n} trajectories**, episode length {Ts.min()}–{Ts.max()} (mean {Ts.mean():.0f}), "
        f"**{succ.mean():.0%} success**.",
        f"- **Proximity-active** (≥1 skin sensor < {args.near_m} m) on **{pf.mean():.0%}** of steps on average "
        f"(range {pf.min():.0%}–{pf.max():.0%}); {int((pf>=0.8).sum())}/{n} trajs ≥80%. "
        f"Mean **{summary['proximity']['mean_active_sensors_per_step']:.1f}** of {n_sensors} sensors active per step.",
        f"- Most-active sensors: {', '.join(summary['proximity']['most_active_sensors'])}. "
        f"Link activation: " + ", ".join(f"{k} {v:.0%}" for k, v in link_act.items()) + ".",
        f"- **Env-collision probability** {ecp.mean():.1%} mean (max {ecp.max():.1%}); "
        f"{int((ecp>0).sum())}/{n} trajs touch the scene at least once "
        f"(excludes the welded held object).",
        f"- **Action noise** (≤1 cm Gaussian TCP noise → joints): injected-noise std "
        f"{inj_std.mean():.4f} rad mean across joints, tracking residual {resid_std.mean():.4f} rad; "
        f"largest on joint {ARM_JOINTS[int(np.argmax(inj_std))]}. Command jitter "
        f"({cmd_jit_std.mean():.4f}) > realized jitter ({real_jit_std.mean():.4f}) — the plant smooths it. "
        f"100% success ⇒ noise is well within recoverable range.", "",
        "## Kinematics",
        f"- arm qpos range {all_qpos.min():.2f}…{all_qpos.max():.2f} rad; |qvel| max {np.abs(all_qvel).max():.2f} rad/s.",
        f"- per-joint |qvel| max: {pj(np.abs(all_qvel).max(0).round(2))}.", "",
        "## Action noise (per joint, rad)",
        f"- injected-noise std: {pj(inj_std.round(4))}",
        f"- tracking residual std: {pj(resid_std.round(4))}", "",
        "## Proximity / skin",
        f"- per-link activation: " + ", ".join(f"{k} {v:.0%}" for k, v in link_act.items()),
        f"- global nearest surface ever seen: {summary['proximity']['global_nearest_surface_m']:.3f} m", "",
        "## Collision by phase",
        *[f"- {PHASE_NAMES[i]}: {phase_coll[i]:.1%}" for i in range(len(PHASE_NAMES)) if phase_coll_den[i] > 0], "",
        "## Figures", *[f"- `{p}`" for p in plots],
        "", "## Files", "- `per_trajectory.csv` — one row per trajectory (18 metrics)",
        "- `summary.json` — dataset aggregates",
    ]
    (out / "report.md").write_text("\n".join(rep))

    print(f"[done] {len(plots)} plots + per_trajectory.csv + summary.json + report.md -> {out}")
    print(f"  success {succ.mean():.0%} | prox-active {pf.mean():.0%} | env-collision {ecp.mean():.1%} "
          f"| injected-noise {inj_std.mean():.4f} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
