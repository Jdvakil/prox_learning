"""Figures + LaTeX table for the Safety-CVAE ablation benchmark.

Reads ``per_scenario*.csv`` under --root (same files as safety_ablation_stats.py) and writes:
    figures/avoid_rate.png        avoid-rate per condition with Wilson 95% CIs
    figures/min_clear.png         min surface clearance per condition (box)
    figures/shape_transfer.png    avoid-rate by obstacle for `full` (only if >1 obstacle)
    figures/noise_curve.png       avoid-rate vs depth-noise sigma (only if noise* conditions)
    table_safety_ablation.tex     booktabs table (condition x avoid%/clear/bow/cos)

Usage:
    python scripts/plot_safety_ablation.py --root eval_output/safety_ablation_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from safety_ablation_stats import cond_summary, load_rows  # noqa: E402
from significance_pact_vs_baseline import wilson_ci  # noqa: E402

ORDER = ["oracle", "full", "forearm_only", "wrist_only", "drop_wrist", "shuffle", "zero"]
LABELS = {
    "oracle": "oracle\n(analytic)", "full": "full skin", "forearm_only": "forearm\nonly",
    "wrist_only": "wrist\nonly", "drop_wrist": "drop\nwrist", "shuffle": "shuffle",
    "zero": "zero\n(no skin)",
}
COLORS = {"oracle": "#4c72b0", "full": "#2ca02c", "shuffle": "#dd8452", "zero": "#c44e52"}
DEFAULT_COLOR = "#8c8c8c"


def ordered_conditions(rows, include_noise=False):
    present = {r["condition"] for r in rows}
    base = [c for c in ORDER if c in present]
    extra = sorted(c for c in present if c not in ORDER and (include_noise or not c.startswith("noise")))
    return base + extra


def label_of(c):
    if c.startswith("noise"):
        return f"noise\n{float(c[5:]) * 1000:.0f}mm"
    return LABELS.get(c, c)


def fig_avoid_rate(rows, conds, out):
    summ = {c: cond_summary(rows, c) for c in conds}
    rates = [summ[c]["rate"] * 100 for c in conds]
    # clip to >= 0: Wilson bounds can fall a float-epsilon past the point estimate at 0/100%
    los = np.clip([(summ[c]["rate"] - summ[c]["wilson95"][0]) * 100 for c in conds], 0, None)
    his = np.clip([(summ[c]["wilson95"][1] - summ[c]["rate"]) * 100 for c in conds], 0, None)
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(conds)), 5))
    x = np.arange(len(conds))
    ax.bar(x, rates, yerr=[los, his], capsize=4,
           color=[COLORS.get(c, DEFAULT_COLOR) for c in conds], edgecolor="k", linewidth=0.6)
    for xi, c in zip(x, conds):
        ax.text(xi, rates[conds.index(c)] + 1.5, f"{summ[c]['avoid']}/{summ[c]['n']}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([label_of(c) for c in conds], fontsize=9)
    ax.set_ylabel("collision-free rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Safety-CVAE avoidance by ablation condition (Wilson 95% CI)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def fig_min_clear(rows, conds, out):
    data, labels = [], []
    for c in conds:
        v = [r["min_clear_cm"] for r in rows if r["condition"] == c and not r["quiet"]]
        if v:
            data.append(v); labels.append(label_of(c))
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(data)), 5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(0.0, color="k", ls="--", lw=1, label="contact (0 cm)")
    ax.set_ylabel("min surface clearance (cm)")
    ax.set_title("Closest obstacle approach per condition (higher = safer)")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def fig_shape_transfer(rows, out):
    obstacles = sorted({r["obstacle"] for r in rows})
    if len(obstacles) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, ob in enumerate(obstacles):
        sub = [r for r in rows if r["obstacle"] == ob and r["condition"] == "full" and not r["quiet"]]
        if not sub:
            continue
        succ, n = sum(r["avoid"] for r in sub), len(sub)
        wl, wh = wilson_ci(succ, n)
        yerr = [[max(0.0, 100 * (succ / n - wl))], [max(0.0, 100 * (wh - succ / n))]]
        ax.bar(i, 100 * succ / n, yerr=yerr, capsize=4, color=DEFAULT_COLOR, edgecolor="k")
        ax.text(i, 100 * succ / n + 1.5, f"{succ}/{n}", ha="center", fontsize=9)
    ax.set_xticks(range(len(obstacles))); ax.set_xticklabels(obstacles)
    ax.set_ylabel("collision-free rate (%)"); ax.set_ylim(0, 105)
    ax.set_title("Shape transfer: full skin head (trained on boxes)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def fig_noise_curve(rows, out):
    pts = []
    if any(r["condition"] == "full" for r in rows):
        s = cond_summary(rows, "full")
        pts.append((0.0, s["rate"], s["wilson95"], s["n"]))
    for c in sorted({r["condition"] for r in rows if r["condition"].startswith("noise")}):
        s = cond_summary(rows, c)
        pts.append((float(c[5:]) * 1000, s["rate"], s["wilson95"], s["n"]))
    if len(pts) < 2:
        return None
    pts.sort()
    xs = [p[0] for p in pts]
    ys = [p[1] * 100 for p in pts]
    lo = np.clip([(p[1] - p[2][0]) * 100 for p in pts], 0, None)
    hi = np.clip([(p[2][1] - p[1]) * 100 for p in pts], 0, None)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=4, color="#4c72b0")
    ax.set_xlabel("depth noise sigma (mm)"); ax.set_ylabel("collision-free rate (%)")
    ax.set_ylim(0, 105); ax.set_title("Avoidance vs skin depth noise (full head)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def write_table(rows, conds, out):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Safety-CVAE closed-loop avoidance under input/sensor ablations "
        r"(collision-course encounters). Avoid-rate is the fraction contact-free; "
        r"Wilson 95\% CIs. Cosine is the closed-loop retreat direction vs the analytic oracle.}",
        r"\label{tab:safety_ablation}", r"\begin{tabular}{lccccc}", r"\toprule",
        r"Condition & $n$ & Avoid \% (95\% CI) & Min clear (cm) & Bow (mm) & $\cos$ \\",
        r"\midrule",
    ]
    pretty = {"oracle": "Oracle (analytic)", "full": "Full skin (head)",
              "forearm_only": "Forearm only", "wrist_only": "Wrist only",
              "drop_wrist": "Drop wrist", "shuffle": "Shuffle", "zero": "Zero (no skin)"}
    for c in conds:
        s = cond_summary(rows, c)
        wl, wh = s["wilson95"]
        name = pretty.get(c, c.replace("_", r"\_"))
        cos = "--" if np.isnan(s["mean_cos"]) else f"{s['mean_cos']:.2f}"
        lines.append(
            f"{name} & {s['n']} & {s['rate']*100:.0f} ({wl*100:.0f}--{wh*100:.0f}) & "
            f"{s['min_clear_cm_mean']:.1f} & {s['bow_mm_mean']:.1f} & {cos} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out.write_text("\n".join(lines))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("eval_output/safety_ablation_v1"))
    ap.add_argument("--obstacle", default=None,
                    help="only plot this obstacle (bar|sphere); default = all rows in --root")
    args = ap.parse_args()
    rows = load_rows(args.root, args.obstacle)
    figdir = args.root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    conds = ordered_conditions(rows, include_noise=False)
    written = [
        fig_avoid_rate(rows, conds, figdir / "avoid_rate.png"),
        fig_min_clear(rows, conds, figdir / "min_clear.png"),
        fig_shape_transfer(rows, figdir / "shape_transfer.png"),
        fig_noise_curve(rows, figdir / "noise_curve.png"),
        write_table(rows, conds, args.root / "table_safety_ablation.tex"),
    ]
    for w in written:
        if w:
            print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
