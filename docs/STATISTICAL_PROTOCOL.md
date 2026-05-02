# Statistical protocol

The exact procedure used to produce every number in the paper. This is
the document a methods-section reviewer reads.

Companion docs:
- [ARCHITECTURE.md](ARCHITECTURE.md) for shapes
- `pla/eval/README.md` for the runner CLI
- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) D13 / D14 for rationale

---

## 1. Outcome variable

Per episode, the env returns `info['success']` ∈ {0, 1}. We never use
shaped success scores or partial credit. A success means: the target
object reached the target placement region within the maximum step
budget *and* the trajectory was free of unrecoverable collisions.

## 2. Sample size

| metric                       | episodes  | comment                              |
|------------------------------|-----------|--------------------------------------|
| Day-7 first-look eval         | 50        | quick check; not paper-quality        |
| Final per-(method × task)     | **100**   | the headline number                  |
| Sensor importance per sensor  | 50        | × 32 sensors = 1600 episodes total    |

Per-episode runtime ~30-60 s on FrankaPickandPlace; 100 episodes per
condition is ~50 min wall time per cell.

## 3. Pairing

Every method is run on the **same** seed schedule:
`[seed_base, seed_base+1, ..., seed_base+n_episodes-1]`. The
`seed_base` is constant across methods (default 42). The same seed in
MolmoSpaces produces:

  - same procthor scene
  - same object placements
  - same language instruction
  - same camera randomization

So the per-episode outcome arrays for two methods are aligned: episode
*i* of PLA and episode *i* of VLM-only ACT are the *same* episode in
every respect except the policy.

## 4. Bootstrap CI

For a per-method success vector `s ∈ {0, 1}^n`:

```
B = 10,000 resamples
for b in 1..B:
    s_b = sample n with replacement from s
    bar_b = mean(s_b)
sort {bar_b}
CI_low  = quantile(bar_b, alpha/2)        with alpha = 0.05
CI_high = quantile(bar_b, 1 - alpha/2)
return (mean(s), CI_low, CI_high)
```

Implementation: `pla.eval.bootstrap.bootstrap_ci(s, n_resamples=10000,
alpha=0.05, seed=0)`.

## 5. Paired bootstrap p-value (vs VLM-only ACT)

For aligned vectors `a, b ∈ {0, 1}^n` (same scene per index):

```
diff = a - b                      # in {-1, 0, +1}
obs  = mean(diff)                  # observed paired effect
H0:  E[diff] = 0
under H0, recenter: centered = diff - obs

B = 10,000 resamples
for b in 1..B:
    cb = sample n with replacement from centered
    bar_b = mean(cb)
p = mean( |bar_b| >= |obs| )      # two-sided
```

Implementation: `pla.eval.bootstrap.paired_bootstrap_p(a, b,
n_resamples=10000, seed=0)`.

**Why two-sided?** The paper claims PLA > VLM-only on near-contact, but
that is the *prediction*, not the *test*. We do not pre-direct the
hypothesis to claim a stronger p-value.

## 6. Significance threshold

α = 0.05. The paper's headline claim requires:

  * delta = `mean(SR_PLA) - mean(SR_baseline) > 0`
  * paired bootstrap p < 0.05

If p ≥ 0.05 we report the number without claiming significance. We do not
search for a significant subset / re-slice the data.

## 7. Multiple-comparison correction

The headline comparison is **one** test (PLA vs VLM-only on `near_contact`).
No correction needed for the headline.

The ablation comparisons (4 ablations × 1 task) and the per-task
comparisons (PLA vs baseline × 4 tasks) are **secondary**. We will report
Bonferroni-corrected p-values (× 4) on these and note when an effect
survives correction. Sensor importance per sensor is not corrected —
the heatmap is descriptive, not inferential.

## 8. Reproducibility

| seed                    | controls                              |
|-------------------------|---------------------------------------|
| `seed_base` (eval)      | scene, object placements, language    |
| `split_seed` (data)     | train/val split (held constant 0)     |
| `seed=0` (bootstrap)    | resample draws (held constant 0)      |
| `torch.manual_seed`     | model init (set per-config)           |
| `np.random.default_rng` | sensor noise / dropout                |

We commit `stats.json` and the per-config YAML alongside each `best.pt`
checkpoint so the exact training-time data distribution and split are
recoverable.

## 9. What we report

### Per-(method × task) cell

```
SR  = 100 * mean(s)
CI  = 100 * (CI_low, CI_high)
p   = paired_bootstrap_p(s, s_baseline) if method != baseline else NA
```

Reported as a row in the results table:

```
| Method        | SR    | 95% CI         | p vs VLM-only |
| PLA           | 67.0% | [57.4%, 75.6%] | 0.0021        |
| VLM-only ACT  | 51.0% | [41.4%, 60.7%] |               |
```

### Sensor importance

```
delta_i = mean(s_baseline) - mean(s_masked_i)     # per sensor i
```

Reported as a heatmap on the FR3 body (one cell per sensor) with a
colorbar in pp (percentage points).

### Failure breakdown

For each (method × task) cell we also report counts per
`FailureType` ∈ {success, approach_collision, grasp_miss, place_failure,
language_failure}. This is descriptive; no significance tests on the
breakdown.

## 10. Pre-registration items

We pre-commit to these decisions before running the final eval (Day 14):

1. Headline test is paired bootstrap on `near_contact`, α=0.05, two-sided.
2. n=100 episodes per cell.
3. seed_base=42 (changed only if a randomization bug surfaces in
   MolmoSpaces).
4. Bonferroni correction on secondary comparisons (× 4).
5. We will report the headline number whether or not it is significant.

If any of these change between Day 14 and submission, we note it in the
paper's Methods §5.x with the reason.
