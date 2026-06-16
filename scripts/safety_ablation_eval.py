"""Closed-loop ABLATION benchmark for the standalone Safety-CVAE.

Generates a fixed set of *collision-course* obstacle encounters (with the arm frozen the
obstacle would hit a link), then replays each encounter under several ablation conditions
that share the same scenario seed, and records -- per (scenario, condition) -- whether the
arm stayed contact-free plus continuous quality metrics. A small *quiet* set (obstacle stays
far) measures false retreat when nothing is actionable.

The point is attribution, not a bare number: the controls (``zero`` no-skin floor, ``shuffle``
spatially-wrong, sensor-group drops, ``oracle`` analytic ceiling) are what turn the avoid-rate
into evidence that the *learned skin signal* causes the avoidance. See safety_eval_lib for the
rollout / scenario / clearance details.

Outputs (under --out):
    per_scenario[.shard<i>].csv   one row per (scenario, condition)
    summary.json                  per-condition avoid-rate + metric means for THIS run/shard

Aggregate + significance is done separately by safety_ablation_stats.py (globs per_scenario*.csv).

Usage (smoke, then full; user runs on the mlspaces box):
    OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      python scripts/safety_ablation_eval.py \
      --runs assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855 \
      --ckpt assets/safety/cvae_v3 --n 10 --conditions full,zero,shuffle,oracle \
      --out eval_output/safety_ablation_smoke

Parallel: launch K processes with --shards K --shard 0..K-1 into the SAME --out; each writes
per_scenario.shard<i>.csv and the stats/plot scripts glob them all.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import safety_eval_lib as L   # imports safety_sweep -> installs the MUJOCO_GL=egl workaround
from train_safety_cvae import SafetyHead

CSV_COLUMNS = [
    "seed", "idx", "quiet", "obstacle", "condition", "target_sensor", "link",
    "avoid", "contact", "min_clear_cm", "peak_dev", "bow_mm", "mean_cos",
    "contact_step", "n_frames", "d_end", "end_clear",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True,
                    help="datagen run dir(s) with house_*/trajectories*.h5 (posture source)")
    ap.add_argument("--ckpt", type=Path, required=True, help="SafetyHead checkpoint dir")
    ap.add_argument("--n", type=int, default=150, help="# collision-course scenarios")
    ap.add_argument("--conditions", default="full,zero,shuffle,oracle,wrist_only,drop_wrist,forearm_only",
                    help="comma-separated ablation conditions (see safety_eval_lib.make_spec)")
    ap.add_argument("--obstacle", choices=["bar", "sphere"], default="bar")
    ap.add_argument("--radius", type=float, default=None, help="sphere radius (m) when --obstacle sphere")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet-frac", type=float, default=0.2,
                    help="extra quiet (far-approach) scenarios as a fraction of --n; 0 disables")
    ap.add_argument("--quiet-conditions", default="full",
                    help="conditions to run on the quiet set (false-retreat check)")
    ap.add_argument("--gain", type=float, default=L.GAIN)
    ap.add_argument("--decay", type=float, default=L.DECAY)
    ap.add_argument("--max-dev", type=float, default=L.MAX_DEV)
    ap.add_argument("--ema", type=float, default=L.EMA)
    ap.add_argument("--pen", type=float, default=L.PEN_TARGET,
                    help="collision-course penetration depth at rest (m); shallow stays in-distribution")
    ap.add_argument("--approach-speed", type=float, default=L.APPROACH_SPEED,
                    help="obstacle inbound speed (m/s); slower gives the head more reaction lead time")
    ap.add_argument("--baseline", choices=["raw", "subtract"], default="raw",
                    help="raw = react to current scene (clean-rest gated, ~2x faster); "
                         "subtract = demo-style head(obstacle) - head(parked)")
    ap.add_argument("--dq-clip", type=float, default=L.DQ_CLIP,
                    help="clip ||dq|| each frame to saturate the reflex (the head over-fires); 0 disables")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="avoid = min surface clearance stays >= this (m); 0 = bare contact")
    ap.add_argument("--out", type=Path, default=Path("eval_output/safety_ablation_v1"))
    ap.add_argument("--shards", type=int, default=1, help="total parallel shards")
    ap.add_argument("--shard", type=int, default=0, help="this shard index in [0, shards)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    quiet_conditions = [c.strip() for c in args.quiet_conditions.split(",") if c.strip()]
    if args.obstacle != "bar":
        dropped = [c for c in ("oracle",) if c in conditions]
        if dropped:
            print(f"[note] dropping {dropped} for obstacle={args.obstacle} "
                  f"(analytic oracle assumes a box obstacle)")
            conditions = [c for c in conditions if c != "oracle"]
            quiet_conditions = [c for c in quiet_conditions if c != "oracle"]

    head = SafetyHead.load(args.ckpt)
    scale = float(head.scale)
    print(f"loaded head {args.ckpt}  label_scale={scale:.4f}")

    ctx = L.build_ctx()
    print(f"model: {len(ctx.sensors)} skin sensors, {len(ctx.arm_gids)} arm collision geoms")

    # build the work list: collision-course (+ optional quiet), each tagged with its conditions
    scenarios = L.make_scenarios(ctx, args.runs, args.n, args.seed, scale, head,
                                 obstacle=args.obstacle, radius=args.radius,
                                 pen=args.pen, approach_speed=args.approach_speed)
    work = [(sc, conditions) for sc in scenarios]
    n_quiet = int(round(args.n * args.quiet_frac))
    if n_quiet > 0 and quiet_conditions:
        quiet = L.make_scenarios(ctx, args.runs, n_quiet, args.seed, scale, head,
                                 obstacle=args.obstacle, radius=args.radius, quiet=True,
                                 approach_speed=args.approach_speed)
        work += [(sc, quiet_conditions) for sc in quiet]

    # shard by global scenario index so parallel runs cover disjoint scenarios
    work = [w for i, w in enumerate(work) if i % args.shards == args.shard]
    print(f"shard {args.shard}/{args.shards}: {len(work)} scenarios "
          f"x conditions -> {sum(len(c) for _, c in work)} rollouts")

    args.out.mkdir(parents=True, exist_ok=True)
    fname = "per_scenario.csv" if args.shards == 1 else f"per_scenario.shard{args.shard}.csv"
    rows = []
    t0 = time.time()
    for i, (sc, conds) in enumerate(work):
        for cond in conds:
            m = L.rollout(ctx, sc, cond, head, scale, gain=args.gain, decay=args.decay,
                          max_dev=args.max_dev, ema=args.ema, margin=args.margin,
                          baseline=args.baseline, dq_clip=args.dq_clip)
            rows.append({
                "seed": sc.seed, "idx": i, "quiet": int(sc.quiet), "obstacle": sc.obstacle,
                "condition": cond, "target_sensor": sc.target_sensor,
                "link": sc.meta.get("link", ""),
                "avoid": int(m["avoid"]), "contact": int(m["contact"]),
                "min_clear_cm": round(100.0 * m["min_clear"], 3),
                "peak_dev": round(m["peak_dev"], 4), "bow_mm": round(1000.0 * m["bow"], 3),
                "mean_cos": ("" if np.isnan(m["mean_cos"]) else round(m["mean_cos"], 4)),
                "contact_step": m["contact_step"], "n_frames": m["n_frames"],
                "d_end": round(sc.meta.get("d_end", float("nan")), 4),
                "end_clear": round(sc.meta.get("end_clear", float("nan")), 4),
            })
        if (i + 1) % 5 == 0 or i + 1 == len(work):
            dt = time.time() - t0
            print(f"  [{i + 1}/{len(work)}] {dt:5.1f}s  ({dt / max(i + 1, 1):.1f}s/scenario)")

    with open(args.out / fname, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # per-shard convenience summary (full aggregation/significance: safety_ablation_stats.py)
    summary = {"args": {k: str(v) for k, v in vars(args).items()},
               "n_rows": len(rows), "conditions": {}}
    for cond in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == cond and not r["quiet"]]
        if not sub:
            continue
        succ = sum(r["avoid"] for r in sub)
        summary["conditions"][cond] = {
            "n": len(sub), "avoid": succ, "avoid_rate": succ / len(sub),
            "min_clear_cm_mean": float(np.mean([r["min_clear_cm"] for r in sub])),
            "bow_mm_mean": float(np.mean([r["bow_mm"] for r in sub])),
            "peak_dev_mean": float(np.mean([r["peak_dev"] for r in sub])),
        }
    (args.out / (f"summary.shard{args.shard}.json" if args.shards != 1 else "summary.json")
     ).write_text(json.dumps(summary, indent=2))

    print(f"\nwrote {args.out / fname}  ({len(rows)} rows, {time.time() - t0:.0f}s)")
    print("avoid-rate (collision-course):")
    for cond, s in summary["conditions"].items():
        print(f"  {cond:14s} {s['avoid']:3d}/{s['n']:<3d} = {s['avoid_rate']*100:5.1f}%   "
              f"min-clear {s['min_clear_cm_mean']:5.1f} cm  bow {s['bow_mm_mean']:5.1f} mm")


if __name__ == "__main__":
    main()
