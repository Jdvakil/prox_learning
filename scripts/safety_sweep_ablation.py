"""On-distribution ablation of the Safety-CVAE encoder.

Evaluates the head + input ablations on the held-out *sweep* validation set -- the exact
distribution the head was distilled on -- measuring retreat-DIRECTION accuracy against the
analytic potential-field label:

    dir-accuracy = cos( head(transform(skin)) , label_dq )      on close samples

For `full` this reproduces the reported close-direction cosine (~0.93 for cvae_v3); the ablations
show proximity, and its spatial layout, are necessary. No MuJoCo / rendering -- it reads the
stored depths from sweep_*.h5, so it runs in seconds. The held-out split matches
train_safety_cvae (np.random.seed -> permutation -> first max(256, N//10)).

Conditions: full | zero | shuffle | drop_wrist | wrist_only | forearm_only | noise<sigma_m>.

Outputs (under --out):
    sweep_ablation_summary.json   per-condition mean cos (+ bootstrap CI) and direction-correct
                                  rate (cos > thresh, + Wilson CI), n = close val samples
    figures/sweep_ablation.png
    table_sweep_ablation.tex

Usage:
    python scripts/safety_sweep_ablation.py --data assets/safety/sweep_v3.h5 --ckpt assets/safety/cvae_v3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

from train_safety_cvae import SafetyHead, featurize  # noqa: E402
from significance_pact_vs_baseline import bootstrap_ci, wilson_ci  # noqa: E402

WRIST_PREFIXES = ("link6", "link7", "gripper")
FOREARM_PREFIXES = ("link4", "link5")
ORDER = ["full", "forearm_only", "wrist_only", "drop_wrist", "shuffle", "zero"]
LABELS = {"full": "full skin", "forearm_only": "forearm\nonly", "wrist_only": "wrist\nonly",
          "drop_wrist": "drop\nwrist", "shuffle": "shuffle", "zero": "zero\n(no skin)"}
COLORS = {"full": "#2ca02c", "shuffle": "#dd8452", "zero": "#c44e52"}
DEFAULT = "#8c8c8c"


def keep_mask(sensors, condition):
    if condition == "drop_wrist":
        return ~np.array([s.startswith(WRIST_PREFIXES) for s in sensors])
    if condition == "wrist_only":
        return np.array([s.startswith(WRIST_PREFIXES) for s in sensors])
    if condition == "forearm_only":
        return np.array([s.startswith(FOREARM_PREFIXES) for s in sensors])
    return None


def features_for(prox, condition, sensors, rng):
    """(M, S, 8, 8) depths -> (M, S*64) closeness features under the given input ablation."""
    if condition == "zero":
        return np.zeros((len(prox), prox.shape[1] * 64), np.float32)
    p = prox
    if condition == "shuffle":
        p = p[:, rng.permutation(p.shape[1])]
    elif condition.startswith("noise"):
        sig = float(condition[len("noise"):])
        p = np.clip(p + rng.normal(0.0, sig, p.shape).astype(p.dtype), 0.0, None)
    else:
        km = keep_mask(sensors, condition)
        if km is not None:
            p = p.copy()
            p[:, ~km] = 0.0
    return featurize(p)


@torch.no_grad()
def predict(head, X):
    t = torch.from_numpy(X.astype(np.float32)).to(head.device)
    return (head.model.act(t).cpu().numpy() * head.scale)


def row_cos(a, b, eps=1e-9):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > eps) & (nb > eps)
    out = np.zeros(len(a))                       # zero-norm prediction => no direction => 0
    out[ok] = np.sum(a[ok] * b[ok], 1) / (na[ok] * nb[ok])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("assets/safety/sweep_v3.h5"))
    ap.add_argument("--ckpt", type=Path, default=Path("assets/safety/cvae_v3"))
    ap.add_argument("--conditions", default="full,zero,shuffle,drop_wrist,wrist_only,forearm_only")
    ap.add_argument("--close", type=float, default=0.12, help="close-sample cutoff on min depth (m)")
    ap.add_argument("--cos-thresh", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0, help="must match the training split seed")
    ap.add_argument("--all-samples", action="store_true",
                    help="evaluate on all close samples, not just the held-out split")
    ap.add_argument("--out", type=Path, default=Path("eval_output/safety_sweep_ablation_v1"))
    args = ap.parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    head = SafetyHead.load(args.ckpt)
    with h5py.File(args.data, "r") as h:
        prox = h["prox"][:].astype(np.float32)        # (N, S, 8, 8)
        label = h["label_dq"][:].astype(np.float32)   # (N, 7)
        md = h["min_depth"][:].astype(np.float32)     # (N, S)
        sensors = [s.decode() for s in h["sensors"][:]]
    n = len(prox)
    md_min = np.where(np.isfinite(md), md, np.inf).min(axis=1)
    close = md_min < args.close

    if args.all_samples:
        sel = close
    else:                                              # reproduce train_safety_cvae's val split
        np.random.seed(args.seed)
        idx = np.random.permutation(n)
        n_val = max(256, n // 10)
        val = np.zeros(n, bool)
        val[idx[:n_val]] = True
        sel = val & close
    P, Y = prox[sel], label[sel]
    print(f"N={n}  close={int(close.sum())}  evaluating on {len(P)} "
          f"{'all-close' if args.all_samples else 'held-out close'} samples")

    summary = {"data": str(args.data), "ckpt": str(args.ckpt), "n_samples": int(len(P)),
               "close_m": args.close, "cos_thresh": args.cos_thresh, "conditions": {}}
    for c in conditions:
        rng = np.random.default_rng(abs(hash(c)) % (2**32))
        pred = predict(head, features_for(P, c, sensors, rng))
        cos = row_cos(pred, Y)
        lo_b, hi_b, _ = bootstrap_ci(cos)
        correct = int(np.sum(cos > args.cos_thresh))
        wl, wh = wilson_ci(correct, len(cos))
        summary["conditions"][c] = {
            "mean_cos": float(cos.mean()), "mean_cos_ci95": [lo_b, hi_b],
            "dir_correct": correct, "n": len(cos), "dir_correct_rate": correct / len(cos),
            "dir_correct_wilson95": [wl, wh],
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sweep_ablation_summary.json").write_text(json.dumps(summary, indent=2))
    present = [c for c in ORDER if c in summary["conditions"]] + \
              [c for c in summary["conditions"] if c not in ORDER]

    print("\n" + "=" * 74)
    print(f"SAFETY-CVAE ON-DISTRIBUTION DIRECTION ACCURACY (cos vs analytic label, n={len(P)})")
    print("=" * 74)
    print(f"{'condition':14s} {'mean cos':>16s}   {'dir-correct % (Wilson95)':>26s}")
    for c in present:
        s = summary["conditions"][c]
        lo_b, hi_b = s["mean_cos_ci95"]
        wl, wh = s["dir_correct_wilson95"]
        print(f"{c:14s} {s['mean_cos']:6.2f} [{lo_b:5.2f},{hi_b:5.2f}]   "
              f"{s['dir_correct_rate']*100:5.0f} [{wl*100:4.0f},{wh*100:4.0f}]")

    # figure
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
           color=[COLORS.get(c, DEFAULT) for c in present], edgecolor="k", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS.get(c, c) for c in present], fontsize=9)
    ax.set_ylabel("retreat-direction cosine vs analytic label")
    ax.set_ylim(min(-0.1, min(means) - 0.1), 1.02)
    ax.set_title("Safety-CVAE on-distribution direction accuracy (held-out close set)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "sweep_ablation.png", dpi=140)
    plt.close(fig)

    pretty = {"full": "Full skin (head)", "forearm_only": "Forearm only", "wrist_only": "Wrist only",
              "drop_wrist": "Drop wrist", "shuffle": "Shuffle", "zero": "Zero (no skin)"}
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{On-distribution retreat-direction accuracy of the Safety-CVAE encoder on the "
        r"held-out sweep validation set: cosine between the skin-derived retreat and the analytic "
        r"label. Full skin recovers the analytic direction; shuffling or removing the skin "
        r"collapses it (proximity and its spatial layout are necessary). Direction-correct = "
        r"samples with $\cos>" + f"{args.cos_thresh:g}" + r"$ (Wilson 95\% CI).}",
        r"\label{tab:sweep_ablation}", r"\begin{tabular}{lcc}", r"\toprule",
        r"Condition & Mean $\cos$ (95\% CI) & Dir-correct \% (95\% CI) \\", r"\midrule",
    ]
    for c in present:
        s = summary["conditions"][c]
        lo_b, hi_b = s["mean_cos_ci95"]
        wl, wh = s["dir_correct_wilson95"]
        lines.append(f"{pretty.get(c, c)} & {s['mean_cos']:.2f} ({lo_b:.2f}--{hi_b:.2f}) & "
                     f"{s['dir_correct_rate']*100:.0f} ({wl*100:.0f}--{wh*100:.0f}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (args.out / "table_sweep_ablation.tex").write_text("\n".join(lines))
    print(f"\n[sweep-ablation] wrote {args.out}/sweep_ablation_summary.json, "
          f"figures/sweep_ablation.png, table_sweep_ablation.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
