"""Validation analysis for the hybrid_obstacle_v1 datagen run.

This dataset is the single-env red-cup pick with hazard bars in the fumehood
(ObstacleFumehoodPickSampler + ObstacleAwarePickPlannerPolicy). Its purpose is
to pair *close-range proximity returns* with *avoidance motion* (the planner
bows around the bar) so that (a) ACT can imitate the pick and (b) the reactive
proximity head has close-encounter data to learn from.

Unlike the generic analyze_dataset.py, this script checks the OBSTACLE-SPECIFIC
requirements:

  1. structure        — episode count, per-run / per-house breakdown, the two
                        timestamped runs, duplicate / missing houses.
  2. single object    — every episode is the same red cup (same target_uid).
  3. bar presence     — fraction of episodes with a hazard bar (~0.75 target).
  4. avoidance signal — behavior_class split (deflect vs free), deflect rate
                        among bar episodes, and the lateral TCP bow that proves
                        the arm actually veered.
  5. proximity signal — does the skin read close (< ~0.10 m) during the inbound
                        approach, and do BAR episodes read systematically
                        closer / with more active sensors than NO-BAR episodes?
                        (statistical proof the bar shows up on the skin).
  6. safety / quality — task success rate, and that the arm is NOT smashing the
                        bar: contacts during the inbound approach phases should
                        be ~0 (grasp/lift contacts are the gripper on the cup
                        and are benign).
  7. integrity        — NaNs, depth value sanity, trajectory completeness.

Usage:
    python scripts/analyze_obstacle_dataset.py \
        --root assets/datagen/hybrid_obstacle_v1 \
        [--out diagnostics_output/obstacle_analysis] [--close_m 0.10]
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

PHASE_NAMES = [
    "unknown", "gripper-open", "pregrasp", "grasp", "gripper-close",
    "lift", "preplace", "place", "retreat", "go_home",
]
# inbound approach: the arm is threading toward the cup, body passing the bar.
APPROACH_PHASES = {2, 3}        # pregrasp, grasp
PREGRASP = 2
GRIP_CLOSE = 4
# proximity valid range (m): below = self/dead pixel noise floor, above = no return
PROX_MIN_VALID = 0.02
PROX_MAX_VALID = 4.0


# --------------------------------------------------------------------------- helpers
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


def chord_bow(xy):
    """Max perpendicular deviation (m) of a 2D path from the straight chord
    connecting its first and last point — i.e. how much the path bowed out."""
    if xy.shape[0] < 3:
        return 0.0
    a, b = xy[0], xy[-1]
    ab = b - a
    L = np.linalg.norm(ab)
    if L < 1e-6:
        return float(np.linalg.norm(xy - a, axis=1).max())
    n = np.array([-ab[1], ab[0]]) / L        # unit normal to the chord
    return float(np.abs((xy - a) @ n).max())


def sensor_sort_key(k):
    parts = k.split("_")
    return (k.split("_sensor_")[0], int(parts[-1]))


# --------------------------------------------------------------------------- per-traj
def extract(t, close_m):
    T = int(t["rewards"].shape[0])
    scene = json.loads(np.asarray(t["obs_scene"]).item())
    sp = scene.get("scene_params", {})

    phase = t["obs/extra/policy_phase"][:].astype(int)
    success = bool(t["success"][:][-1])

    base_pose = t["obs/extra/robot_base_pose"][:]
    tcp = t["obs/extra/tcp_pose"][:]
    tcp_w = tcp_to_world(tcp, base_pose)

    # proximity: nearest valid depth per (step, sensor)
    prox_keys = sorted(t["obs/proximity"].keys(), key=sensor_sort_key)
    n_sensors = len(prox_keys)
    near = np.full((T, n_sensors), np.nan)
    n_dead = 0
    n_pix = 0
    for j, k in enumerate(prox_keys):
        px = t["obs/proximity"][k][:].mean(axis=1).reshape(T, -1)   # mean over 4 subsamples
        n_dead += int((px < PROX_MIN_VALID).sum())
        n_pix += px.size
        v = np.where((px > PROX_MIN_VALID) & (px <= PROX_MAX_VALID), px, np.nan)
        with np.errstate(all="ignore"):
            near[:, j] = np.nanmin(v, axis=1)
    finite = np.isfinite(near)
    with np.errstate(all="ignore"):
        glob_near = np.where(finite.any(1), np.nanmin(np.where(finite, near, np.inf), 1), np.nan)
    n_close = (near < close_m).sum(1)                               # sensors closer than close_m per step

    # collision per step, split by phase bucket
    cm = scene.get("collision_metrics", {})
    psc = np.asarray(cm.get("per_step_contacts", [0] * T), dtype=float)
    if psc.shape[0] != T:
        psc = np.resize(psc, T)
    first_close = np.where(phase == GRIP_CLOSE)[0]
    grip_start = int(first_close[0]) if first_close.size else T      # inbound = before gripper closes
    inbound = np.arange(T) < grip_start
    approach_steps = np.isin(phase, list(APPROACH_PHASES)) & inbound
    approach_contacts = float(psc[approach_steps].sum())
    held_contacts = float(psc[~inbound].sum())                       # gripper-close/lift onward = on the cup

    # windows
    pre_mask = (phase == PREGRASP) & inbound
    app_mask = approach_steps
    def wmin(mask):
        v = glob_near[mask]
        v = v[np.isfinite(v)]
        return float(v.min()) if v.size else np.nan
    def wnclose(mask):
        return int(n_close[mask].max()) if mask.any() else 0
    def wfrac(mask):
        return float((n_close[mask] > 0).mean()) if mask.any() else 0.0

    bow = chord_bow(tcp_w[app_mask, :2]) if app_mask.sum() >= 3 else 0.0

    has_bar = bool(sp.get("protrusion_present", False))
    bclass = scene.get("behavior_class", "?")
    return dict(
        T=T, success=success,
        object_name=scene.get("object_name", "?"),
        target_uid=sp.get("target_uid", scene.get("text", "?")),
        has_bar=has_bar, behavior_class=bclass,
        protr_name=sp.get("protr_name"), protr_wall=sp.get("protr_wall"),
        residual_margin=sp.get("residual_margin"), bar_face_y=sp.get("bar_face_y"),
        obj_gap=sp.get("obj_gap"), clearance=sp.get("clearance"),
        ap_w=sp.get("ap_w"), ap_h=sp.get("ap_h"), light_scale=sp.get("light_scale"),
        n_sensors=n_sensors,
        min_near_overall=float(np.nanmin(glob_near)) if np.isfinite(glob_near).any() else np.nan,
        min_near_approach=wmin(app_mask),
        min_near_pregrasp=wmin(pre_mask),
        max_nclose_approach=wnclose(app_mask),
        frac_approach_active=wfrac(app_mask),
        bow_approach_m=bow,
        approach_contacts=approach_contacts, held_contacts=held_contacts,
        any_approach_collision=approach_contacts > 0,
        dead_pixel_frac=n_dead / max(n_pix, 1),
        n_finite_any=int(finite.any(1).sum()),
    )


# --------------------------------------------------------------------------- main
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="assets/datagen/hybrid_obstacle_v1")
    p.add_argument("--out", default="diagnostics_output/obstacle_analysis")
    p.add_argument("--close_m", type=float, default=0.10,
                   help="a skin sensor is 'close' when it reads a surface nearer than this (m)")
    p.add_argument("--bar_target", type=float, default=0.75, help="expected bar-present rate (OBSTACLE_P)")
    args = p.parse_args()

    paths = sorted(globlib.glob(str(Path(args.root) / "**/house_*/trajectories_batch_*.h5"), recursive=True))
    if not paths:
        print("no h5 matched under", args.root); return 1
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"analyzing {len(paths)} h5 files -> {out}")

    rows = []
    structure = {}    # run -> house -> n_traj
    for pth in paths:
        parts = Path(pth).parts
        house = Path(pth).parent.name
        run = parts[-3]
        try:
            f = h5py.File(pth, "r")
        except Exception as e:
            print(f"[skip] {pth}: {e}"); continue
        structure.setdefault(run, {})[house] = len(f.keys())
        for tk in f.keys():
            try:
                d = extract(f[tk], args.close_m)
            except Exception as e:
                print(f"[skip] {pth}:{tk}: {e}"); continue
            d.update(run=run, house=house, traj=tk, file=pth)
            rows.append(d)
        f.close()

    n = len(rows)
    if n == 0:
        print("no trajectories"); return 1
    print(f"extracted {n} trajectories")

    def col(k):
        return np.array([r[k] for r in rows])

    has_bar = col("has_bar").astype(bool)
    bclass = col("behavior_class")
    success = col("success").astype(bool)
    uids = col("target_uid")
    objs = col("object_name")

    # ---- groups
    nobar = ~has_bar
    deflect = bclass == "deflect"
    bar_free = has_bar & ~deflect
    groups = {"no-bar": nobar, "bar-free": bar_free, "bar-deflect": has_bar & deflect}

    # ---- plots
    plots = []
    def save(fig, name):
        fig.savefig(out / name, dpi=130, bbox_inches="tight"); plt.close(fig); plots.append(name)

    # behavior class counts
    classes, counts = np.unique(bclass, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([str(c) for c in classes], counts, color="C0")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")
    ax.set(title=f"behavior_class ({n} eps; bar={has_bar.mean():.0%})", ylabel="# episodes")
    save(fig, "01_behavior_class.png")

    # min approach depth by group
    fig, ax = plt.subplots(figsize=(10, 5))
    for gname, gmask in groups.items():
        v = col("min_near_approach")[gmask]
        v = v[np.isfinite(v)]
        if v.size:
            ax.hist(v, bins=30, range=(0, 0.5), alpha=0.5, label=f"{gname} (n={v.size}, med={np.median(v):.3f})")
    ax.axvline(args.close_m, color="k", ls="--", label=f"close={args.close_m} m")
    ax.set(title="Min skin depth during inbound approach (pregrasp+grasp)", xlabel="nearest depth (m)", ylabel="# eps")
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, "02_min_approach_depth_by_group.png")

    # TCP top-down paths colored by group (proves the bow)
    fig, ax = plt.subplots(figsize=(8, 8))
    gc = {"no-bar": "0.6", "bar-free": "C0", "bar-deflect": "C3"}
    for gname, gmask in groups.items():
        first = True
        for r in (rr for rr, m in zip(rows, gmask) if m):
            with h5py.File(r["file"], "r") as f:
                t = f[r["traj"]]
                tw = tcp_to_world(t["obs/extra/tcp_pose"][:], t["obs/extra/robot_base_pose"][:])
            ax.plot(tw[:, 0], tw[:, 1], lw=0.7, alpha=0.4, color=gc[gname],
                    label=gname if first else None)
            first = False
    ax.set(title="TCP top-down paths by group (bar-deflect should bow)", xlabel="x (m)", ylabel="y (m)")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend()
    save(fig, "03_tcp_paths_by_group.png")

    # lateral bow by group
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [col("bow_approach_m")[gmask] for gmask in groups.values()]
    ax.boxplot([d[np.isfinite(d)] for d in data], labels=list(groups.keys()), showfliers=False)
    ax.set(title="Lateral TCP bow during approach (perp. deviation from chord)", ylabel="bow (m)")
    ax.grid(alpha=0.3)
    save(fig, "04_lateral_bow_by_group.png")

    # ---- stats helpers
    def grp_stat(key, mask):
        v = col(key)[mask].astype(float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                    p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                    min=float(v.min()), max=float(v.max()))

    bar_rate = float(has_bar.mean())
    deflect_rate_among_bar = float((deflect & has_bar).sum() / max(has_bar.sum(), 1))

    summary = {
        "root": args.root, "n_files": len(paths), "n_episodes": n, "close_m": args.close_m,
        "structure": structure,
        "single_object": {
            "unique_target_uids": sorted(set(map(str, uids.tolist()))),
            "unique_object_names": sorted(set(map(str, objs.tolist()))),
            "is_single_object": len(set(map(str, uids.tolist()))) == 1,
        },
        "bar_presence": {
            "bar_rate": bar_rate, "target": args.bar_target,
            "n_bar": int(has_bar.sum()), "n_nobar": int(nobar.sum()),
        },
        "behavior_class": {str(c): int(n_) for c, n_ in zip(classes, counts)},
        "avoidance": {
            "deflect_rate_among_bar": deflect_rate_among_bar,
            "n_deflect": int(deflect.sum()),
            "bow_by_group_m": {g: grp_stat("bow_approach_m", m) for g, m in groups.items()},
        },
        "proximity_signal": {
            "min_approach_depth_by_group_m": {g: grp_stat("min_near_approach", m) for g, m in groups.items()},
            "min_pregrasp_depth_by_group_m": {g: grp_stat("min_near_pregrasp", m) for g, m in groups.items()},
            "max_nclose_sensors_by_group": {g: grp_stat("max_nclose_approach", m) for g, m in groups.items()},
            "frac_approach_active_by_group": {g: grp_stat("frac_approach_active", m) for g, m in groups.items()},
            "frac_bar_eps_with_close_return": float((col("max_nclose_approach")[has_bar] > 0).mean()) if has_bar.any() else 0.0,
            "frac_nobar_eps_with_close_return": float((col("max_nclose_approach")[nobar] > 0).mean()) if nobar.any() else 0.0,
        },
        "success": {
            "overall": float(success.mean()),
            "bar": float(success[has_bar].mean()) if has_bar.any() else None,
            "nobar": float(success[nobar].mean()) if nobar.any() else None,
            "deflect": float(success[deflect].mean()) if deflect.any() else None,
            "n_fail": int((~success).sum()),
        },
        "safety": {
            "frac_eps_with_approach_collision": float(col("any_approach_collision").mean()),
            "approach_contacts_by_group": {g: grp_stat("approach_contacts", m) for g, m in groups.items()},
            "note": "approach_contacts = scene contacts during pregrasp/grasp BEFORE gripper closes (arm-vs-env). "
                    "held_contacts (gripper-close/lift) are the fingers on the cup and are benign.",
        },
        "integrity": {
            "dead_pixel_frac_mean": float(col("dead_pixel_frac").mean()),
            "episode_len": {"min": int(col("T").min()), "max": int(col("T").max()), "mean": float(col("T").mean())},
            "n_sensors": int(col("n_sensors")[0]),
            "n_sensors_consistent": bool((col("n_sensors") == col("n_sensors")[0]).all()),
            "any_empty_proximity": bool((col("n_finite_any") == 0).any()),
        },
        "plots": plots,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- per-episode csv
    cols = ["run", "house", "traj", "T", "success", "has_bar", "behavior_class",
            "protr_name", "protr_wall", "residual_margin", "bar_face_y", "obj_gap",
            "min_near_overall", "min_near_approach", "min_near_pregrasp",
            "max_nclose_approach", "frac_approach_active", "bow_approach_m",
            "approach_contacts", "held_contacts", "object_name", "target_uid"]
    with open(out / "per_episode.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    # ---- console report
    def fmt(s):
        return "n/a" if s is None else f"med={s['median']:.3f} mean={s['mean']:.3f} [{s['p10']:.3f},{s['p90']:.3f}] n={s['n']}"
    print("\n" + "=" * 70)
    print(f"OBSTACLE DATASET VALIDATION — {n} episodes, {len(paths)} files")
    print("=" * 70)
    print("STRUCTURE (run -> house: n_traj):")
    for run, hh in structure.items():
        tot = sum(hh.values())
        print(f"  {run}: {tot} eps  " + ", ".join(f"{h}:{c}" for h, c in sorted(hh.items())))
    print(f"\nSINGLE OBJECT: {'YES' if summary['single_object']['is_single_object'] else 'NO -> ' + str(summary['single_object']['unique_target_uids'])}")
    print(f"  object_name(s): {summary['single_object']['unique_object_names']}")
    print(f"\nBAR PRESENCE: {bar_rate:.1%} ({int(has_bar.sum())}/{n})  target~{args.bar_target:.0%}")
    print(f"BEHAVIOR CLASS: {summary['behavior_class']}")
    print(f"  deflect rate among bar eps: {deflect_rate_among_bar:.1%} ({int(deflect.sum())} deflect)")
    print("\nPROXIMITY (min approach depth, m):")
    for g, m in groups.items():
        print(f"  {g:12s}: {fmt(grp_stat('min_near_approach', m))}")
    print(f"  frac BAR eps with a close return (<{args.close_m}m) in approach: {summary['proximity_signal']['frac_bar_eps_with_close_return']:.1%}")
    print(f"  frac NO-BAR eps with a close return:                        {summary['proximity_signal']['frac_nobar_eps_with_close_return']:.1%}")
    print("\nLATERAL BOW during approach (m):")
    for g, m in groups.items():
        print(f"  {g:12s}: {fmt(grp_stat('bow_approach_m', m))}")
    print(f"\nSUCCESS: overall {success.mean():.1%}  bar {summary['success']['bar']}  nobar {summary['success']['nobar']}  ({int((~success).sum())} fail)")
    print(f"SAFETY: eps with approach (arm-vs-env) collision: {summary['safety']['frac_eps_with_approach_collision']:.1%}")
    print(f"  approach_contacts by group: " + ", ".join(f"{g}:{(grp_stat('approach_contacts', m) or {}).get('mean', 0):.2f}" for g, m in groups.items()))
    print(f"\nINTEGRITY: {summary['integrity']['n_sensors']} sensors (consistent={summary['integrity']['n_sensors_consistent']}), "
          f"ep_len {summary['integrity']['episode_len']['min']}-{summary['integrity']['episode_len']['max']}, "
          f"dead-pixel frac {summary['integrity']['dead_pixel_frac_mean']:.3f}, "
          f"empty-prox eps: {summary['integrity']['any_empty_proximity']}")
    print(f"\n[done] -> {out}/summary.json, per_episode.csv, {len(plots)} plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
