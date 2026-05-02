# `pla/eval/` — evaluation, statistics, sensor importance

## Purpose

Run a trained model against the four MolmoSpaces tasks, compute success
rates with bootstrap 95% CIs, and produce paired-bootstrap p-values for
every comparison vs the VLM-only baseline. This is what produces the
results table in the paper.

## Files

| file                   | role                                                |
|------------------------|-----------------------------------------------------|
| `tasks.py`             | `REGISTRY` of the 4 tasks + their MolmoSpaces specs |
| `run_eval.py`          | `evaluate_checkpoint`, `print_results_table`        |
| `bootstrap.py`         | `bootstrap_ci`, `paired_bootstrap_p`                |
| `sensor_importance.py` | per-sensor masking sweep                            |
| `failure_analysis.py`  | failure categorization for §4.4 of the paper        |

## Tasks

| name           | obstacle? | language? | role                                |
|----------------|-----------|-----------|-------------------------------------|
| `pnp`          | no        | no        | baseline competence floor           |
| `near_contact` | YES       | no        | **PRIMARY** — proximity should win  |
| `pnp_color`    | no        | YES       | language stress test                |
| `pnp_next_to`  | no        | YES       | hardest spatial-language case        |

The headline number of the paper is the **delta** between PLA and VLM-only
ACT on `near_contact`, with paired-bootstrap p < 0.05.

## Run

```bash
# 1. Per-method, per-task evaluation (100 episodes for the final table).
python -m pla.eval.run_eval \
    --checkpoint runs/pla_concat_v1/best.pt \
    --task near_contact \
    --n-episodes 100 \
    --out reports/eval/pla_near_contact.json \
    --method-name PLA

python -m pla.eval.run_eval \
    --checkpoint runs/vlm_only_baseline_v1/best.pt \
    --task near_contact \
    --n-episodes 100 \
    --out reports/eval/vlm_only_near_contact.json \
    --method-name "VLM-only ACT"

# 2. Aggregate table (reads every .json in the dir):
python -m pla.eval.run_eval --print-table reports/eval/

# 3. Sensor importance — per-sensor masking sweep:
python -m pla.eval.sensor_importance \
    --checkpoint runs/pla_concat_v1/best.pt \
    --task near_contact \
    --n-episodes 50 \
    --out reports/tables/sensor_importance.json
```

## Why these design choices

* **Paired bootstrap with shared seeds.** Each method is run on the same
  `seed_base + i` for `i in range(n_episodes)`. Same scene, same object
  positions, same language. The paired bootstrap is then a valid estimator
  of `P(SR_PLA > SR_baseline | scene)`. Without pairing the variance is
  much larger and we'd need 3-4x more episodes for the same p-value.
* **Bootstrap CIs (10000 resamples).** We avoid normality assumptions for
  small N; CIs on a 100-episode binomial benefit from this. Seeded for
  reproducibility (`seed=0` in `bootstrap_ci`).
* **JSON results.** Per (method, task) pair we write a self-contained JSON
  with successes, seeds, scene IDs, language, and the original training
  config. The aggregator (`print_results_table`) reads any directory of
  these files. This means we can mix-and-match results from different
  machines or different days.
* **Sensor importance is post-hoc.** No retraining required: we just zero
  out one sensor's input grid at a time and re-eval. The same paired-seed
  trick gives us paired deltas.

## Statistical claims

> PLA outperforms VLM-only ACT on near-contact: SR `<x>%` (95% CI
> `[<lo>, <hi>]`) vs `<y>%` (`[<lo>, <hi>]`), `delta = <d> pp`,
> paired bootstrap p < `<p>`. (PROJECT.md §4.1.)

If the paired bootstrap p-value is > 0.05 for the headline comparison,
**stop**. Re-check: (a) was the VLM-only run trained on the same data with
the same hyperparameters? (b) did `enc_grad_norm` stay > 1e-6 during PLA
training? (c) did the data verifier report >= 30% proximity-informative
trajectories? See "Critical things that go wrong" in the technical summary.

## Sanity-check checklist (Day 7 first eval, Day 14 final eval)

- [ ] Each method's eval JSON has `len(successes) == n_episodes`.
- [ ] Bootstrap CI lower bound for VLM-only is ≥ 0 and upper ≤ 1.
- [ ] PLA - VLM-only delta on `near_contact` is positive on Day 7
      (50-episode quick eval). If not, do not proceed to Day 8 ablations
      until this is fixed.
- [ ] Paired-bootstrap p-value < 0.05 on the final 100-episode eval.
- [ ] `pnp` (open workspace) shows a small or zero PLA delta — this is
      *expected*; if PLA dominates `pnp`, the proximity stream may be
      leaking task-irrelevant signal (e.g. encoding object identity from
      proximity reflectance).
