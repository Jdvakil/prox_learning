# 003 — Proximity reduces obstacle contact, and the benefit lives entirely in the tail

*Established 2026-08-02, from the PACT contact-endpoint run. Decision token:
`CONTACT_REDUCTION_WITH_TASK_BENEFIT`.*

## The finding

Adding whole-body proximity sensing to an imitation policy **reduces environment contact**, and
the effect replicates. It does **not** produce a statistically confirmed improvement in task
success.

Pooled across 3 policy seeds × 100 held-out instances = 300 rollouts per arm:

| Arm | Collision-free success | Any hazard contact | Hazard frames (mean) |
|---|---:|---:|---:|
| ACT (vision only) | 53.0% [45.3, 60.7] | 22.3% | 3,023 |
| **PACT** (vision + proximity) | 57.0% [48.7, 65.0] | **14.0%** | **1,678** |
| PACT_PERMUTED (valid ablation) | 53.0% [45.0, 61.0] | 23.3% | 3,658 |
| PACT_ZERO (OOD probe) | 6.7% [4.0, 9.3] | 35.7% | 3,913 |

Four independent contact endpoints agree, all with intervals excluding zero:

| Contrast | Endpoint | Difference | 95% CI |
|---|---|---:|---|
| PACT − ACT | hazard frames | −1,345 | [−2,521, −279] |
| PACT − PERMUTED | hazard frames | −1,980 | [−3,153, −965] |
| PACT − PERMUTED | any hazard contact | −9.3 pp | [−14.3, −5.0] |
| PACT − PERMUTED | max penetration | −0.34 mm | [−0.55, −0.16] |

Per-seed hazard frames (PACT − PERMUTED): **−2,556 / −1,667 / −1,716** — same sign, comparable
magnitude, across three independent trainings.

Task success was **not** confirmed: PACT − ACT collision-free success +4.0 pp, CI [−2.3, +10.3],
positive in every seed (+8.0 / +1.0 / +3.0) but never separable from zero.

## The mechanism — this is the part that matters

**The median hazard-frame count is 0.0 for every arm.** Most episodes involve no obstacle contact
at all. And conditioned on both compared arms completing the manipulation, the contact difference
is **−0.8, CI [−19.8, +22.0]** — indistinguishable from nothing.

Post-hoc characterization with a threshold fixed at **more than 500 hazard frames** confirms the
tail shape and refines the mechanism:

| Arm | Enters >500-frame regime | Mean frames given entry | Median first-contact step |
|---|---:|---:|---:|
| ACT | 19.7% | 15,363 | 59 |
| **PACT** | **11.0%** | 15,236 | **291** |
| PACT_PERMUTED | 19.3% | 18,894 | 63 |

PACT therefore primarily makes the high-contact case rarer: entry falls by 44.1% relative to ACT
and 43.1% relative to the distribution-matched control. Severity given entry is essentially
unchanged versus ACT and 19.4% lower versus PACT_PERMUTED. PACT also delays median first contact by
more than 200 control steps.

The tail is scene-susceptible but policy-triggered. In the fixed-weights comparison, 31 of PACT's
33 high-contact instance-seed rollouts are also high under PACT_PERMUTED, and all 20 instances
that ever go high for PACT lie inside PACT_PERMUTED's larger 29-instance tail set. Proximity
prevents triggers within a shared set of difficult scenes rather than moving failures to a wholly
different set.

What the run **does not** establish is literal contiguous entrapment or faster escape. Per-step
contact payloads were suppressed, and `contact_class_totals` counts contact-pair samples rather
than transitions into contact. Longest contiguous runs cannot be reconstructed. The defensible
term is therefore **high-contact regime**, not proven entanglement. See
[the full post-hoc characterization](../PACT_TAIL_CHARACTERIZATION.md).

### A second exploratory mechanism: retaining target engagement

A later post-hoc raw-file sweep found that combining hazard contact with no target contact is an
absorbing failure state: **0/67** non-OOD rollouts that touched the hazard but never touched the
target succeeded. Using the scan's more tolerant definition—more than 500 hazard frames and fewer
than 50 target frames—PACT enters this state in **5/33 (15.2%)** tail rollouts, versus **34/59
(57.6%)** for ACT and **33/58 (56.9%)** for PACT_PERMUTED.

This difference is not only caused by PACT having a smaller tail. Among the 31 matched
instance-seed cells where both PACT and the same-weights PACT_PERMUTED control are high-contact,
the low-target rates are **5/31 versus 13/31**, an exploratory whole-instance bootstrap difference
of **−25.8 pp [−45.0, −8.3]**. The same direction appears at 10, 50, and 100 target frames.

This is consistent with proximity helping the policy remain target-directed when contact occurs,
but it is not a cleanly identified causal mechanism. High-contact membership is changed by the
modality, making the grouping post-treatment, and 1,198/1,200 trajectories were deleted. The data
cannot show whether the policy stalls en route, reaches the target neighbourhood and is diverted,
or recovers differently after impact. See
[the absorbing-failure characterization](../PACT_ABSORBING_FAILURE_CHARACTERIZATION.md).

The honest claim is therefore narrower than "proximity improves obstacle avoidance":

> **Proximity sensing reduces the frequency and severity of catastrophic contact events. It does
> not measurably change behaviour in episodes that go well.**

This also explains why task success barely moves. Ordinary manipulation succeeds in only 6/59
ACT high-contact rollouts, 8/33 PACT high-contact rollouts, and 7/58 PACT_PERMUTED high-contact
rollouts. Most tail cases are task failures, although episode totals alone cannot establish
whether the contact caused each failure.

## Why the effect was detectable here and not earlier

Three prior runs measured **collision-free task success**, a binary composite:

```
task success  =  manipulation  ×  avoidance
                 (fails ~40%,      (what proximity
                  proximity-        actually affects)
                  irrelevant)
```

The manipulation term contributes large variance that proximity cannot influence, diluting the
signal. Measuring contact directly removes it. Same policies, same rollouts, roughly double the
signal-to-noise:

| Endpoint | PACT − ACT | ratio to its own SE |
|---|---:|---:|
| Collision-free success | +4.0 pp | ~1.25 |
| Hazard contact frames | −1,345 | ~2.35 |

**General lesson: measure the quantity the modality acts on, not a downstream composite that
bundles it with unrelated variance.** A binary composite endpoint can hide a real effect
indefinitely.

## A correction to the earlier seed diagnosis

An earlier run reported PACT − ACT at +25.0 pp on one seed and −7.5 pp on the next, and this was
attributed to training variance of roughly 23 pp. That over-attributed it. At 40 instances, the
per-seed intervals were **[+7.5, +42.5]** and **[−25.0, +12.5]** — they *overlap*. The two seeds
were never statistically inconsistent; most of the apparent chaos was sampling noise from a small
sample.

At 100 instances per seed the estimates read +8.0 / +1.0 / +3.0, an SD of ~3.6 pp against ~5 pp of
residual sampling error — so true training variance may be modest.

**Two distinct terms, two distinct remedies:**

```
observed spread  =  training variance  +  sampling noise
                    ↓ more seeds          ↓ more instances
```

Note also that 1,200 rollouts is not an effective sample of 1,200. The bootstrap clusters on
instances, and all arms and seeds move together within a cluster, so the pooled interval is
governed by **100 instances**. Instances are the lever; seeds are the insurance.

## Why this result is credible

- **Preregistered**: endpoints, analysis script (`e2d9a506…`), and decision rule frozen before the
  first rollout.
- **Valid instrument**: PACT_PERMUTED — real embeddings shuffled across timesteps, destroying
  information while preserving the input distribution. See [001](001-zeroing-is-not-a-valid-ablation-for-embedding-front-ends.md).
- **Replicated**: three independently initialised seeds, same sign and magnitude.
- **Multiple endpoints**: frames, entries, any-contact rate, and penetration depth all agree.
- **Predicted in advance**: PACT_ZERO's collapse to 6.7% was forecast before the run as expected
  OOD behaviour, and is excluded from modality evidence rather than used as support.
- **Negative outcomes were reportable**: `CONTACT_INCREASE` was an allowed token.

## What the awarded token overstates

`CONTACT_REDUCTION_WITH_TASK_BENEFIT` fired correctly under the frozen rule, but that rule
required only that PACT − ACT task success be *positive and consistently signed* — it did not
require the interval to exclude zero. The task-benefit component is therefore **directional, not
established**. The token name promises more than the data delivers; the rule was drafted too
loosely.

Report as: *contact reduction confirmed; task success directionally positive, not confirmed.*

## What remains open

- Escape dynamics remain unidentified. A future audit should retain transition count, longest
  contiguous run, run count, and first/last contact step per class; these summaries are compact
  enough to keep for every rollout.
- The task-success gap (+4.0 pp) would need roughly **500 instances** for 80% power — about five
  times the current design — and only if the true effect is as large as the point estimate.
- The environment carries roughly **one bit** of decision-relevant information: a fixed 0.85 m
  aperture, a single active panel, and ~1 cm of jitter. A blind policy gets that bit right 50% of
  the time by guessing, which structurally caps what proximity can contribute.
- Manipulation fails in ~40% of rollouts, supplying variance unrelated to the modality.

Raising the effect (independent continuous panel placement) and lowering the noise (grasp
reliability) both move the achievable result more than additional sampling does.
