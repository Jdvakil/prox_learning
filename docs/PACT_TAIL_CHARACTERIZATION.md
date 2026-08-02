# PACT contact-tail characterization

> **Post-hoc exploratory characterization.** Nothing in this document changes the frozen
> endpoints, preregistered analysis, or awarded token. No post-hoc grouping below carries a
> confirmatory p-value or confidence interval.

## Result in one sentence

The retained totals support proximity **primarily preventing entry into a high-contact regime**:
PACT crosses the predeclared `hazard frames > 500` threshold in 11.0% of rollouts, versus 19.7%
for ACT and 19.3% for PACT_PERMUTED. The tail is concentrated in scene-susceptible instances, but
proximity changes whether those instances trigger. The data cannot establish faster escape or a
contiguous entrapment mechanism because per-step contact runs were not retained.

The confirmatory decision remains `CONTACT_REDUCTION_WITH_TASK_BENEFIT`. Its task-benefit component
is directional rather than statistically established, as documented in the frozen report.

## Scope and definitions

This analysis uses all 1,200 completed rollouts: 100 held-out instances × 3 policy seeds × 4 arms.
The high-contact threshold is **strictly greater than 500 hazard-contact physics frames**, fixed in
`PACT_TAIL_CHARACTERIZATION_PLAN.md` before row-level tail analysis. Results are paired by the
frozen instance ID, arm, and seed.

`PACT_PERMUTED` is the distribution-matched, same-weights modality instrument. `PACT_ZERO` remains
an out-of-distribution sensor-failure diagnostic and is not used for modality attribution.

## 1. Distribution shape

Most rollouts have zero contact, while a nonzero rollout usually has thousands of contact frames.

| Arm | Zero frames | Nonzero n | Nonzero p50 | p75 | p90 | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACT | 233/300 (77.7%) | 67 | 11,319 | 27,594 | 28,098 | 28,461 | 28,833 | 29,022 |
| PACT | 258/300 (86.0%) | 42 | 14,748 | 18,746 | 20,744 | 21,503 | 27,517 | 27,852 |
| PACT_ZERO (OOD) | 193/300 (64.3%) | 107 | 5,746 | 20,368 | 27,076 | 28,636 | 29,303 | 29,364 |
| PACT_PERMUTED | 230/300 (76.7%) | 70 | 18,628 | 27,151 | 27,699 | 27,867 | 28,267 | 29,098 |

The median across all 300 rollouts remains zero for every arm. The table makes the mixture clear:
there is little middle ground between no contact and a very large contact total.

## 2. Concentration

| Arm | Top 1% share | Top 5% share | Top 10% share | Frames above 500 / all frames |
|---|---:|---:|---:|---:|
| ACT | 9.5% | 46.6% | 83.3% | 99.93% |
| PACT | 15.2% | **61.7%** | **98.6%** | 99.86% |
| PACT_ZERO (OOD) | 7.5% | 35.7% | 64.4% | 99.89% |
| PACT_PERMUTED | 7.7% | 38.0% | 71.7% | 99.86% |

PACT's top 5% of rollouts carries 61.7% of all its hazard frames, and its top 10% carries 98.6%.
The `>500` group accounts for more than 99.8% of total hazard frames in every arm. The confirmed
mean reduction is therefore a tail result, not a shift in routine zero-contact behavior.

## 3. Entry versus accumulated severity

| Arm | Entered >500 | Mean frames given entry | Median given entry | Median first-contact step | Tail task success |
|---|---:|---:|---:|---:|---:|
| ACT | 59/300 (19.7%) | 15,363 | 15,423 | 59 / 900 | 6/59 (10.2%) |
| PACT | **33/300 (11.0%)** | 15,236 | 16,549 | **291 / 900** | 8/33 (24.2%) |
| PACT_ZERO (OOD) | 98/300 (32.7%) | 11,965 | 7,354 | 68 / 900 | 3/98 (3.1%) |
| PACT_PERMUTED | 58/300 (19.3%) | 18,894 | 20,487 | 63 / 900 | 7/58 (12.1%) |

Descriptively:

- Versus ACT, PACT reduces entry by 8.7 percentage points (a 44.1% relative reduction), while
  conditional mean frames differ by only −127 (0.8%).
- Versus the valid same-weights PACT_PERMUTED control, PACT reduces entry by 8.3 points (43.1%
  relative) and conditional mean frames by 3,658 (19.4%).
- PACT's median first hazard contact occurs 228–232 control steps later than the two non-OOD
  comparators.

The strong result is therefore **entry prevention**, especially prevention of early entry. Lower
conditional accumulation versus PACT_PERMUTED is also present, but it cannot be uniquely labeled
"faster escape": later onset alone leaves less episode time in which contact can accumulate, and
the retained data contains no contiguous-run lengths.

## 4. Scene susceptibility versus policy triggering

High-contact overlap is shown both for exactly matched instance-seed rollouts and for instances
that trigger under any of their three seeds.

| Comparison | Matched high overlap | Matched Jaccard | Instance overlap | Instance Jaccard |
|---|---:|---:|---:|---:|
| PACT vs ACT | 22 | 31.4% | 16 | 45.7% |
| PACT vs PACT_PERMUTED | **31** | **51.7%** | **20** | **69.0%** |
| ACT vs PACT_PERMUTED | 43 | 58.1% | 24 | 66.7% |

The most revealing fixed-weights comparison is asymmetric:

- 31 of PACT's 33 high-contact rollouts (93.9%) are also high under the matched PACT_PERMUTED
  condition.
- All 20 instances that are high for PACT under any seed also belong to PACT_PERMUTED's larger
  29-instance tail set.
- Across ACT, PACT, and PACT_PERMUTED, 64 instances never enter the tail, 8 enter for one arm, 12
  for two arms, and 16 for all three.

This supports **scene-linked susceptibility plus policy-dependent triggering**. Proximity does not
create a wholly different set of problem scenes; it prevents many triggers inside the same
susceptible set.

## 5. Is the tail entrapment?

It cannot be established directly from this run.

The audit code can retain per-step contact classes, but collection set summary-only mode for every
rollout. Consequently, all 1,200 records have `contact_frame_payload_retained=false` and an empty
`contact_frames` array. Longest contiguous contact duration is unavailable.

The recorded `contact_class_totals` are also not transition-entry counts. They increment once for
each simultaneous robot–hazard contact pair at every audited physics frame. Therefore
`frames_with_contact / contact_class_totals` mostly reflects contact-pair multiplicity; it cannot
distinguish one sustained run from repeated brushing. Treating it as an escape-duration statistic
would be incorrect.

What is observable is consistent with a trap-like high-contact regime: tail rollouts accumulate
roughly 12,000–19,000 frames on average, account for virtually all contact, and mostly fail the
task. But "contiguous entrapment" remains a hypothesis, not a demonstrated mechanism.

## 6. What else is true of tail episodes

Every `>500` rollout is labeled `hazard_bar_contact`, and none can satisfy collision-free task
success by definition. Ordinary manipulation succeeds in only 10.2% of ACT tail rollouts, 24.2%
of PACT tails, and 12.1% of PACT_PERMUTED tails. Thus most high-contact episodes are already task
failures, although contact totals alone cannot determine whether contact caused the failure.

| Arm | Median penetration | Mean penetration | Maximum penetration | First-contact middle 50% |
|---|---:|---:|---:|---:|
| ACT | 0.813 mm | 1.997 mm | 10.850 mm | steps 47–148 |
| PACT | **0.551 mm** | **0.887 mm** | 8.955 mm | steps 142–344 |
| PACT_ZERO (OOD) | 0.519 mm | 1.148 mm | 7.581 mm | steps 24–217 |
| PACT_PERMUTED | 0.893 mm | 2.255 mm | 10.460 mm | steps 54–147 |

PACT tail contacts begin later and penetrate less deeply than ACT or PACT_PERMUTED tail contacts.
These are post-hoc descriptive correlates, not additional confirmatory endpoints.

## 7. Paired per-instance view

Hazard frames were summed over the three seeds for each of the 100 frozen instances. Negative
PACT-minus-comparator values favor PACT.

| Contrast | PACT better | PACT worse | Tied | Mean difference | Median difference |
|---|---:|---:|---:|---:|---:|
| PACT vs ACT | 24 | 13 | 63 | −4,036 | 0 |
| PACT vs PACT_PERMUTED | **26** | **4** | 70 | **−5,939** | 0 |

The median instance difference is zero because most instances never enter the tail. The mean is
driven by a minority of discordant instances, but the valid same-weights contrast is not merely
one outlier: PACT improves 26 instances and worsens 4.

## Conclusion and future instrumentation

The supported description is:

> **Proximity primarily prevents entry into a scene-linked high-contact regime. It also delays
> first contact and reduces conditional accumulation versus the distribution-matched control,
> but the existing episode totals cannot establish that it shortens entrapment once contact
> begins.**

A future run aimed at escape dynamics should retain a compact run-length summary per contact
class—number of transitions into contact, longest contiguous run, total run count, and first/last
contact step. Those four integers per class would distinguish avoidance, sustained entrapment, and
repeated brushing without retaining the large per-physics-step payload.

Machine-readable artifact:
`diagnostics_output/pact_contact_endpoint/tail_characterization.json`
(self-hash `f0b14355e79aa6a0999029af7789654a1bff0882a5207cc0204087108fd15b54`).
