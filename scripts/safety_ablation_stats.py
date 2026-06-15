"""Aggregate + significance for the Safety-CVAE ablation benchmark.

Reads every ``per_scenario*.csv`` under --root (shards merge automatically) and reports, per
condition on the collision-course set:
  * avoid-rate with a Wilson 95% CI,
  * continuous means (min clearance, lateral bow, retreat-cosine vs the analytic oracle),
and, for each control vs the learned ``full`` head (paired -- conditions share scenario seeds):
  * McNemar exact test on the avoid outcome + Newcombe 95% CI for the rate difference,
  * Wilcoxon signed-rank on min-clearance and bow.
The quiet set is summarised separately as a false-retreat (peak-deviation) check.

Reuses the CI helpers from significance_pact_vs_baseline.py so the statistics match the rest
of the paper's reporting.

Usage:
    python scripts/safety_ablation_stats.py --root eval_output/safety_ablation_v1
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from significance_pact_vs_baseline import newcombe_diff_ci, wilson_ci  # noqa: E402

FULL = "full"


def load_rows(root: Path, obstacle: str | None = None) -> list[dict]:
    paths = sorted(glob.glob(str(root / "per_scenario*.csv")))
    if not paths:
        raise SystemExit(f"no per_scenario*.csv under {root}")
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                if obstacle and r["obstacle"] != obstacle:
                    continue
                r["quiet"] = int(r["quiet"])
                r["avoid"] = int(r["avoid"])
                r["min_clear_cm"] = float(r["min_clear_cm"])
                r["bow_mm"] = float(r["bow_mm"])
                r["peak_dev"] = float(r["peak_dev"])
                r["mean_cos"] = float(r["mean_cos"]) if r["mean_cos"] not in ("", "nan") else np.nan
                rows.append(r)
    print(f"loaded {len(rows)} rows from {len(paths)} file(s)"
          + (f" (obstacle={obstacle})" if obstacle else ""))
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    """Exact (binomial) McNemar two-sided p for discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def cond_summary(rows: list[dict], cond: str) -> dict:
    sub = [r for r in rows if r["condition"] == cond and not r["quiet"]]
    n = len(sub)
    succ = sum(r["avoid"] for r in sub)
    cos = [r["mean_cos"] for r in sub if not np.isnan(r["mean_cos"])]
    lo, hi = wilson_ci(succ, n) if n else (float("nan"), float("nan"))
    return {
        "n": n, "avoid": succ, "rate": (succ / n if n else float("nan")),
        "wilson95": [lo, hi],
        "min_clear_cm_mean": float(np.mean([r["min_clear_cm"] for r in sub])) if n else float("nan"),
        "bow_mm_mean": float(np.mean([r["bow_mm"] for r in sub])) if n else float("nan"),
        "peak_dev_mean": float(np.mean([r["peak_dev"] for r in sub])) if n else float("nan"),
        "mean_cos": float(np.mean(cos)) if cos else float("nan"),
    }


def paired_vs_full(rows: list[dict], cond: str) -> dict | None:
    """Pair `cond` against `full` on shared scenario seeds (collision-course only)."""
    fmap = {r["seed"]: r for r in rows if r["condition"] == FULL and not r["quiet"]}
    cmap = {r["seed"]: r for r in rows if r["condition"] == cond and not r["quiet"]}
    keys = sorted(set(fmap) & set(cmap))
    if not keys:
        return None
    fa = np.array([fmap[k]["avoid"] for k in keys])
    ca = np.array([cmap[k]["avoid"] for k in keys])
    b = int(np.sum((fa == 1) & (ca == 0)))      # full avoids, cond fails
    c = int(np.sum((fa == 0) & (ca == 1)))      # full fails, cond avoids
    sf, sc = int(fa.sum()), int(ca.sum())
    n = len(keys)

    def wilcoxon_p(key):
        df = np.array([fmap[k][key] - cmap[k][key] for k in keys])
        if np.allclose(df, 0):
            return float("nan")
        try:
            return float(wilcoxon(df).pvalue)
        except ValueError:
            return float("nan")

    return {
        "n_pairs": n, "full_avoid": sf, "cond_avoid": sc,
        "delta_rate_pp": 100.0 * (sf - sc) / n,
        "discordant_b_full_only": b, "discordant_c_cond_only": c,
        "mcnemar_p": mcnemar_exact(b, c),
        "newcombe_95_for_full_minus_cond": list(newcombe_diff_ci(sc, n, sf, n)),
        "wilcoxon_min_clear_p": wilcoxon_p("min_clear_cm"),
        "wilcoxon_bow_p": wilcoxon_p("bow_mm"),
    }


def quiet_summary(rows: list[dict], cond: str) -> dict | None:
    sub = [r for r in rows if r["condition"] == cond and r["quiet"]]
    if not sub:
        return None
    pd = np.array([r["peak_dev"] for r in sub])
    return {"n": len(sub), "peak_dev_mean": float(pd.mean()),
            "peak_dev_p90": float(np.percentile(pd, 90))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("eval_output/safety_ablation_v1"))
    ap.add_argument("--obstacle", default=None,
                    help="only analyze this obstacle (bar|sphere); default = all rows in --root")
    ap.add_argument("--out", type=Path, default=None, help="significance.json path (default <root>/significance.json)")
    args = ap.parse_args()
    out = args.out or (args.root / "significance.json")

    rows = load_rows(args.root, args.obstacle)
    conds = sorted({r["condition"] for r in rows})
    # report order: ceiling, model, structured ablations, floor (only those present)
    order = ["oracle", FULL, "forearm_only", "wrist_only", "drop_wrist", "shuffle", "zero"]
    conds = [c for c in order if c in conds] + [c for c in conds if c not in order]

    summary = {"root": str(args.root), "conditions": {}, "paired_vs_full": {}, "quiet": {}}
    for c in conds:
        summary["conditions"][c] = cond_summary(rows, c)
        if c != FULL:
            pv = paired_vs_full(rows, c)
            if pv:
                summary["paired_vs_full"][c] = pv
        q = quiet_summary(rows, c)
        if q:
            summary["quiet"][c] = q

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    # ---- console
    print("\n" + "=" * 92)
    print("SAFETY-CVAE ABLATION (collision-course set)")
    print("=" * 92)
    print(f"{'condition':14s} {'avoid':>13s}  {'Wilson95':>15s}  {'minclr':>7s} "
          f"{'bow':>7s} {'cos':>6s}  {'p vs full':>10s}")
    for c in conds:
        s = summary["conditions"][c]
        wl, wh = s["wilson95"]
        ci = f"[{wl*100:4.0f},{wh*100:4.0f}]" if not np.isnan(wl) else "       n/a"
        pv = summary["paired_vs_full"].get(c)
        pstr = "      -" if c == FULL else (f"{pv['mcnemar_p']:.4f}" if pv else "    n/a")
        cos = "   -" if np.isnan(s["mean_cos"]) else f"{s['mean_cos']:.2f}"
        print(f"{c:14s} {s['avoid']:4d}/{s['n']:<4d}={s['rate']*100:4.0f}%  {ci:>15s}  "
              f"{s['min_clear_cm_mean']:6.1f}c {s['bow_mm_mean']:6.1f}m {cos:>6s}  {pstr:>10s}")
    if summary["quiet"]:
        print("\nquiet set (false retreat - lower is better):")
        for c, q in summary["quiet"].items():
            print(f"  {c:14s} n={q['n']:<4d} peak_dev mean {q['peak_dev_mean']:.3f} "
                  f"p90 {q['peak_dev_p90']:.3f} rad")
    print(f"\n[stats] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
