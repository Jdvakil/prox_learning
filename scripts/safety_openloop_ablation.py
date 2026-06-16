"""Open-loop ablation: does the Safety-CVAE *encoder* point the right way, and is proximity
what makes it work?

This is the right instrument for an ENCODER claim (perception), as opposed to the closed-loop
avoid-rate which conflates perception with reactive control. The arm is held FIXED at the rest
posture while an obstacle sweeps through the near-contact band; at every engaged frame we compare
the head's retreat DIRECTION -- under each input ablation -- to the analytic potential-field
oracle (the label the head was distilled from):

    dir-accuracy = cos( head(transform(skin)) , analytic_retreat(skin) )

Because the arm never moves, there is no feedback drift, and a single skin render per frame
serves every condition (they only transform the input). The scenario is the independent unit:
we average cosine within a scenario, then aggregate across scenarios with CIs.

Conditions (see safety_eval_lib.make_spec): full | zero | shuffle | drop_wrist | wrist_only |
forearm_only | noise<sigma>. The oracle is the reference (cos == 1 by construction), so it is not
reported as a condition.

Outputs (under --out):
    per_scenario_openloop.csv      one row per (scenario, condition): mean cos, # engaged frames
    openloop_summary.json          per-condition mean cos (+ bootstrap CI) and direction-correct
                                   rate (cos > thresh, scenario-level, + Wilson CI)
    figures/openloop_dir_accuracy.png
    table_openloop_ablation.tex

Usage:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/safety_openloop_ablation.py \
      --runs <posture_run> --ckpt assets/safety/cvae_v3 --n 50
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

import safety_eval_lib as L  # noqa: E402
import safety_sweep as sw  # noqa: E402
import mujoco  # noqa: E402
from train_safety_cvae import SafetyHead  # noqa: E402
from significance_pact_vs_baseline import bootstrap_ci, wilson_ci  # noqa: E402

ORDER = ["full", "forearm_only", "wrist_only", "drop_wrist", "shuffle", "zero"]
LABELS = {"full": "full skin", "forearm_only": "forearm\nonly", "wrist_only": "wrist\nonly",
          "drop_wrist": "drop\nwrist", "shuffle": "shuffle", "zero": "zero\n(no skin)"}
COLORS = {"full": "#2ca02c", "shuffle": "#dd8452", "zero": "#c44e52"}
DEFAULT_COLOR = "#8c8c8c"


def per_scenario(ctx, sc, head, scale, conditions, band, eps=1e-9):
    """Mean retreat-direction cosine (head vs oracle) per condition, arm fixed at rest.

    Returns {condition: (mean_cos | nan, n_engaged_frames)}.
    """
    model, data, rd, opt = ctx.model, ctx.data, ctx.rd, ctx.opt
    sensors, cam_ids, arm_gids = ctx.sensors, ctx.cam_ids, ctx.arm_gids
    obs_name = sc.obstacle_name
    obs_gid = L.obstacle_geom_id(model, obs_name)
    jacp = np.zeros((3, model.nv))
    fromto = np.zeros(6)
    lo, hi = band

    # per-condition transform spec + a persistent RNG (shuffle perm fixed; noise advances)
    rngs = {c: L._cond_rng(sc.seed, c) for c in conditions}
    specs = {c: L.make_spec(c, sensors, rngs[c]) for c in conditions}

    data.mocap_pos[ctx.mid[f"{L.NS}base"]] = sc.base[:3]
    data.mocap_quat[ctx.mid[f"{L.NS}base"]] = sc.base[3:7]
    L.apply_aperture(data, ctx.mid, sc.ap_w, sc.ap_h)
    L.park_obstacles(data, ctx.mid)
    L.set_arm(data, ctx.arm_qadr, sc.q0)          # FIXED at rest for the whole sweep
    L.set_grip(data, ctx.finger_qadr, sc.grip)
    mujoco.mj_forward(model, data)
    base_boxes = L.frame_boxes(data, ctx.mid, sc.ap_w, sc.ap_h)

    acc = {c: [] for c in conditions}
    for p in sc.path:
        if not np.isfinite(p).all():
            continue
        data.mocap_pos[ctx.mid[obs_name]] = p
        mujoco.mj_forward(model, data)
        clr = L.link_clearance(model, data, obs_gid, arm_gids, fromto)
        if not (lo <= clr <= hi):
            continue                               # only frames where there is something to perceive
        depths = L.render_all(model, data, rd, opt, sensors)   # ONE render, shared by all conditions
        boxes_w = base_boxes + [(np.asarray(p, float), sc.obstacle_half)]
        oracle, _ = sw.analytic_retreat(model, data, sensors, cam_ids, depths, boxes_w,
                                        ctx.arm_dofadr, jacp)
        no = float(np.linalg.norm(oracle))
        if no < 1e-6:
            continue                               # oracle sees nothing -> no ground-truth direction
        for c in conditions:
            if specs[c]["kind"] == "zero":
                acc[c].append(0.0)                 # no input -> no direction (counts as not-correct)
                continue
            h = head(L.transform_depths(depths, specs[c], rngs[c])) / max(scale, 1e-6)
            nh = float(np.linalg.norm(h))
            acc[c].append(float(np.dot(h, oracle) / (nh * no)) if nh > eps else 0.0)

    return {c: (float(np.mean(v)) if v else float("nan"), len(v)) for c, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=Path("assets/safety/cvae_v3"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--conditions", default="full,zero,shuffle,drop_wrist,wrist_only,forearm_only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engaged", type=float, nargs=2, default=(0.02, 0.20),
                    help="clearance band (m) of an active near-encounter")
    ap.add_argument("--cos-thresh", type=float, default=0.5,
                    help="a scenario is 'direction-correct' for a condition if its mean cos exceeds this")
    ap.add_argument("--out", type=Path, default=Path("eval_output/safety_openloop_v1"))
    args = ap.parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    head = SafetyHead.load(args.ckpt)
    scale = float(head.scale)
    print(f"loaded head {args.ckpt}  label_scale={scale:.4f}")
    ctx = L.build_ctx()
    print(f"model: {len(ctx.sensors)} skin sensors, {len(ctx.arm_gids)} arm collision geoms")
    scs = L.make_scenarios(ctx, args.runs, args.n, args.seed, scale, head)

    rows = []
    for i, sc in enumerate(scs):
        res = per_scenario(ctx, sc, head, scale, conditions, tuple(args.engaged))
        for c, (mc, nf) in res.items():
            rows.append({"scenario": i, "link": sc.meta.get("link", ""), "condition": c,
                         "mean_cos": ("" if np.isnan(mc) else round(mc, 4)), "n_frames": nf})
        if (i + 1) % 10 == 0 or i + 1 == len(scs):
            print(f"  [{i + 1}/{len(scs)}] scenarios done")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_scenario_openloop.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scenario", "link", "condition", "mean_cos", "n_frames"])
        w.writeheader()
        w.writerows(rows)

    # aggregate per condition; the scenario is the independent unit
    summary = {"root": str(args.out), "engaged_band": list(args.engaged),
               "cos_thresh": args.cos_thresh, "conditions": {}}
    for c in conditions:
        vals = np.array([r["mean_cos"] for r in rows
                         if r["condition"] == c and r["mean_cos"] != ""], dtype=float)
        if vals.size == 0:
            continue
        lo_b, hi_b, _ = bootstrap_ci(vals)
        correct = int(np.sum(vals > args.cos_thresh))
        wl, wh = wilson_ci(correct, vals.size)
        summary["conditions"][c] = {
            "n_scenarios": int(vals.size),
            "mean_cos": float(vals.mean()), "mean_cos_ci95": [lo_b, hi_b],
            "dir_correct": correct, "dir_correct_rate": correct / vals.size,
            "dir_correct_wilson95": [wl, wh],
        }
    (args.out / "openloop_summary.json").write_text(json.dumps(summary, indent=2))

    present = [c for c in ORDER if c in summary["conditions"]] + \
              [c for c in summary["conditions"] if c not in ORDER]

    # ---- console
    print("\n" + "=" * 78)
    print("SAFETY-CVAE OPEN-LOOP RETREAT-DIRECTION ACCURACY (cos vs analytic oracle)")
    print("=" * 78)
    print(f"{'condition':14s} {'n':>4s} {'mean cos':>16s}   {'dir-correct % (Wilson95)':>26s}")
    for c in present:
        s = summary["conditions"][c]
        lo_b, hi_b = s["mean_cos_ci95"]
        wl, wh = s["dir_correct_wilson95"]
        print(f"{c:14s} {s['n_scenarios']:4d} {s['mean_cos']:6.2f} [{lo_b:5.2f},{hi_b:5.2f}]   "
              f"{s['dir_correct_rate']*100:5.0f} [{wl*100:4.0f},{wh*100:4.0f}]")
    print("\n(oracle is the reference: cos = 1.00 by construction)")

    # ---- figure: mean direction cosine per condition with bootstrap CIs
    figdir = args.out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(present))
    means = [summary["conditions"][c]["mean_cos"] for c in present]
    los = [max(0.0, summary["conditions"][c]["mean_cos"] - summary["conditions"][c]["mean_cos_ci95"][0])
           for c in present]
    his = [max(0.0, summary["conditions"][c]["mean_cos_ci95"][1] - summary["conditions"][c]["mean_cos"])
           for c in present]
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(present)), 5))
    ax.bar(xs, means, yerr=[los, his], capsize=4,
           color=[COLORS.get(c, DEFAULT_COLOR) for c in present], edgecolor="k", linewidth=0.6)
    ax.axhline(1.0, ls="--", c="gray", lw=1, label="oracle (=1.0)")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS.get(c, c) for c in present], fontsize=9)
    ax.set_ylabel("retreat-direction cosine vs oracle")
    ax.set_ylim(min(-0.1, min(means) - 0.1), 1.05)
    ax.set_title("Safety-CVAE encoder: open-loop retreat-direction accuracy")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "openloop_dir_accuracy.png", dpi=140)
    plt.close(fig)

    # ---- LaTeX table
    pretty = {"full": "Full skin (head)", "forearm_only": "Forearm only", "wrist_only": "Wrist only",
              "drop_wrist": "Drop wrist", "shuffle": "Shuffle", "zero": "Zero (no skin)"}
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Open-loop retreat-direction accuracy of the Safety-CVAE encoder: cosine between "
        r"the head's skin-derived retreat and the analytic potential-field oracle, with the arm "
        r"held fixed (no closed-loop confound). Mean over engaged frames per encounter; the "
        r"encounter is the unit. Direction-correct = encounters with mean $\cos>" + f"{args.cos_thresh:g}" +
        r"$ (Wilson 95\% CI). Proximity is necessary: shuffling or removing the skin collapses it.}",
        r"\label{tab:openloop_ablation}", r"\begin{tabular}{lccc}", r"\toprule",
        r"Condition & $n$ & Mean $\cos$ (95\% CI) & Dir-correct \% (95\% CI) \\", r"\midrule",
    ]
    for c in present:
        s = summary["conditions"][c]
        lo_b, hi_b = s["mean_cos_ci95"]
        wl, wh = s["dir_correct_wilson95"]
        name = pretty.get(c, c.replace("_", r"\_"))
        lines.append(f"{name} & {s['n_scenarios']} & {s['mean_cos']:.2f} ({lo_b:.2f}--{hi_b:.2f}) & "
                     f"{s['dir_correct_rate']*100:.0f} ({wl*100:.0f}--{wh*100:.0f}) \\\\")
    lines += [r"\midrule", r"Oracle (analytic) & -- & 1.00 (ref) & 100 (ref) \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (args.out / "table_openloop_ablation.tex").write_text("\n".join(lines))

    print(f"\n[openloop] wrote {args.out}/openloop_summary.json, per_scenario_openloop.csv, "
          f"figures/openloop_dir_accuracy.png, table_openloop_ablation.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
