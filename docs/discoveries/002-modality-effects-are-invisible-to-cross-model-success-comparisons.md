# 002 — A real modality effect can be invisible to cross-model success comparison

*Discovered 2026-08-01, from the PACT seed-replication run.*

## The lesson, stated generally

When you add a sensor modality to an imitation policy and compare aggregate task success against
a baseline policy, you are measuring the modality effect **plus the full variance of two
independent training runs**. If the modality effect is smaller than that training variance — and
it usually is — the comparison returns noise, in whichever direction the seeds happen to fall.

The modality effect is recoverable, but only with an instrument that **holds the weights fixed**:
compare a policy against *itself* with the modality ablated in distribution. That contrast is
paired within a single set of weights, so training variance cancels exactly.

**Cross-model success comparison is the wrong instrument for modality attribution. Within-model
ablation is the right one.**

## The measurement

Two independently trained seeds, three arms, same 40 held-out instances, endpoint collision-free
task success.

| Arm | Seed 3101 | Seed 3102 | swing |
|---|---:|---:|---:|
| ACT | 47.5% | 60.0% | 12.5 pp |
| PACT | 72.5% | 52.5% | 20.0 pp |
| PACT_PERMUTED | 60.0% | 37.5% | 22.5 pp |

Every arm moves 12–22 points on seed choice alone.

| Contrast | Seed 3101 | Seed 3102 | Pooled (instance-clustered CI) |
|---|---:|---:|---:|
| **PACT − PACT_PERMUTED** (within-model) | +12.5 pp | +15.0 pp | **+13.8 pp [+3.8, +23.8]** |
| PACT_PERMUTED − ACT (cross-model) | +12.5 pp | −22.5 pp | −5.0 pp [−20.0, +10.0] |
| PACT − ACT (cross-model) | +25.0 pp | −7.5 pp | +8.8 pp [−5.0, +22.5] |

The within-model contrast replicated in **sign and magnitude** and its pooled interval excludes
zero. Both cross-model contrasts flipped sign between seeds; one swung 35 points.

## Why the within-model contrast survives

`PACT` and `PACT_PERMUTED` are the *same weights* evaluated on the *same instances* — the only
difference is whether the proximity tokens carry scene-correlated information or are permuted
from other timesteps. Initialization, optimization trajectory, and every other training accident
are held identical, so they subtract out.

`PACT` and `ACT` are two different networks from two different training runs. Their difference
carries the modality effect *and* everything that differed in how the two runs happened to land.

## The scale of the problem

From the two-seed spread, the standard deviation of PACT − ACT is roughly **23 pp**. Detecting
the observed ~8.8 pp mean against that noise requires on the order of **50+ seeds per arm** — over
100 hours of training before a single evaluation rollout.

So this is not a problem you can fix by adding seeds. At any feasible scale, a modality effect of
this size is simply not visible in a cross-model success comparison.

## The interpretive line that must not be crossed

`PACT > PACT_PERMUTED` means **the trained policy uses the modality**. It does **not** mean the
modality makes a policy better than one that never had it.

A policy trained with a sensor learns to depend on it, so removing it hurts. That is a
*dependency* measurement, not a *benefit* measurement. The two dissociate, and here they did:
the policy demonstrably used proximity (+13.8 pp, CI excludes zero) while not beating the
vision-only baseline (+8.8 pp, CI includes zero). Both statements are true at once.

Report them as different claims. Conflating them is how a dependency result gets sold as a
benefit result.

## Worked example of the instrument failing in both directions

The same experiment produced two spectacular wrong answers before producing the right one:

1. **+70 pp, far too positive.** The ablation zeroed a 32-D learned embedding. Zero never occurs
   in that representation — 0 of 1,247,040 training tokens — so the ablated arm received an
   out-of-distribution input and collapsed to 2.5%, *below* the baseline that never had the
   modality. See [001](001-zeroing-is-not-a-valid-ablation-for-embedding-front-ends.md).
2. **−7.5 pp, far too negative.** With the ablation fixed, the cross-model contrast on a second
   seed flipped sign entirely, after reading +25.0 pp on the first.

Only the within-model contrast on a distribution-matched ablation gave a stable, replicable
number. Both failure modes were invisible from the headline figure alone; each was caught only by
a structural check — a sanity floor in the first case, an independent seed in the second.

## Practical protocol

- Make the **within-model, distribution-matched ablation** the primary instrument for any
  "does the policy use modality X" question.
- Treat cross-model success comparison as answering a **different** question — "is this
  architecture better" — and budget seeds accordingly, or don't ask it.
- Always report the **three-way decomposition**: modality (within-model), architecture/training
  (ablated-vs-baseline), and the combination. Conflating the first two is how artifacts survive.
- Run **at least two seeds** before believing any cross-model number, and report them unpooled
  and side by side before pooling.
- Sanity floor: if an ablated arm falls below a baseline that never had the modality, suspect the
  ablation before believing the effect.

## Relation to the published claim

This is a quantified instance of the PACT paper's own thesis — that aggregate task success is too
coarse to characterize what an added modality contributes. The paper reached that conclusion from
a statistical tie (74% vs 72%) and pivoted to fixed-policy internal analysis. This run supplies
the missing measurement: the effect was there all along at ~14 pp, stable across seeds, and the
aggregate comparison could not see it because training variance is roughly 23 pp.
