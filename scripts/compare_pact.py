"""Compare P+ACT (PACT) vs vanilla-ACT eval outcomes with honest small-N statistics.

The original PACT was dropped largely because its win (8/10 vs 4/10) was not significant
at n=10 (Fisher p=0.085) and the per-run success swung +-10pp. This helper makes the
comparison explicit: success rate with a 95% Wilson interval, collision rate, the
rate difference vs the first (baseline) arm, and a Fisher exact test. Read the eval
stdout lines:
    [act-eval] success S/T (..%)
    [act-eval] collision summary: contact_free_rate=R ...   (collision eps = round((1-R)*T))
and pass the counts in. Collisions are optional.

Each arm is `name=succ/total` or `name=succ/total,coll/total` (the first arm is the
baseline everything is compared against):

    python scripts/compare_pact.py \
        vanilla=20/50,30/50 pact_trunk=29/50,12/50 pact_delta=26/50,16/50
"""
from __future__ import annotations

import argparse
import math

try:
    from scipy.stats import fisher_exact  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (robust at small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _hypergeom_pmf(k: int, K: int, n: int, N: int) -> float:
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def fisher_one_sided_greater(b_succ: int, b_n: int, a_succ: int, a_n: int) -> float:
    """One-sided Fisher exact P(arm B has >= its observed successes | margins), i.e.
    evidence that B's success rate exceeds A's. Pure-python hypergeometric tail."""
    a_fail, b_fail = a_n - a_succ, b_n - b_succ
    N = a_n + b_n
    col_succ = a_succ + b_succ          # total successes (column margin)
    row_b = b_n                          # arm-B row margin
    lo = max(0, col_succ - a_n)
    hi = min(col_succ, row_b)
    # P(B successes >= observed b_succ) under the hypergeometric with these margins.
    return sum(_hypergeom_pmf(k, row_b, col_succ, N) for k in range(b_succ, hi + 1))


def fisher_two_sided(a_succ: int, a_n: int, b_succ: int, b_n: int) -> float:
    if _HAVE_SCIPY:
        _, p = fisher_exact([[a_succ, a_n - a_succ], [b_succ, b_n - b_succ]])
        return float(p)
    # Fallback: two-sided as 2x the smaller one-sided tail, clipped to 1.
    g = fisher_one_sided_greater(b_succ, b_n, a_succ, a_n)
    l = fisher_one_sided_greater(a_succ, a_n, b_succ, b_n)
    return min(1.0, 2 * min(g, l))


def parse_arm(spec: str) -> dict:
    name, rest = spec.split("=", 1)
    parts = rest.split(",")
    s, t = parts[0].split("/")
    arm = {"name": name, "succ": int(s), "n": int(t), "coll": None, "coll_n": None}
    if len(parts) > 1:
        c, cn = parts[1].split("/")
        arm["coll"], arm["coll_n"] = int(c), int(cn)
    return arm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arms", nargs="+", help="name=succ/total[,coll/total]; first arm = baseline")
    args = p.parse_args()
    arms = [parse_arm(a) for a in args.arms]
    base = arms[0]

    print(f"\n{'arm':<14} {'success':>12} {'95% CI':>16} {'collision':>14}")
    print("-" * 60)
    for a in arms:
        lo, hi = wilson_ci(a["succ"], a["n"])
        rate = a["succ"] / a["n"] if a["n"] else 0.0
        coll = ""
        if a["coll"] is not None:
            coll = f"{a['coll']}/{a['coll_n']} ({100*a['coll']/a['coll_n']:.0f}%)"
        print(f"{a['name']:<14} {a['succ']:>3}/{a['n']:<3} ({100*rate:4.1f}%)"
              f"  [{100*lo:4.1f},{100*hi:5.1f}]%  {coll:>14}")

    if not _HAVE_SCIPY:
        print("\n[note] scipy not found — Fisher exact uses a pure-python fallback.")
    print(f"\nvs baseline '{base['name']}' (Fisher exact):")
    for a in arms[1:]:
        d = (a["succ"] / a["n"]) - (base["succ"] / base["n"])
        p_two = fisher_two_sided(base["succ"], base["n"], a["succ"], a["n"])
        p_one = fisher_one_sided_greater(a["succ"], a["n"], base["succ"], base["n"])
        verdict = "SIGNIFICANT" if p_two < 0.05 else ("trend" if p_two < 0.15 else "n.s.")
        print(f"  {a['name']:<14} Δsuccess={100*d:+5.1f}pp  "
              f"p_two={p_two:.3f}  p_one(>)={p_one:.3f}  [{verdict}]")
    print("\nReminder: 50 rollouts is a QUICK PROOF. If a win shows a trend/significance,"
          "\nre-run with multiple seeds for confidence intervals before claiming it.\n")


if __name__ == "__main__":
    main()
