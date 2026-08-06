# PACT hazard-engaged, target-disengaged failure characterization

> **Post-hoc exploratory characterization.** This report does not change any frozen endpoint,
> preregistered analysis, or the awarded `CONTACT_REDUCTION_WITH_TASK_BENEFIT` token. The
> groupings and intervals below are descriptive; no confirmatory p-values were computed.

## Result in one sentence

PACT is much less likely than ACT or the same-weights `PACT_PERMUTED` control to combine a
high-contact episode with negligible target engagement. The difference remains in the thin
matched subset where both policies are high-contact, which is consistent with proximity helping
the policy stay on task during contact. However, high-contact membership is itself changed by the
modality and trajectory/target-distance data were deleted, so **these data cannot fully separate
a within-contact on-task mechanism from selection into the contact tail**.

This is a second exploratory mechanism alongside the previously documented prevention of tail
entry. It is not a new confirmatory endpoint.

## Definitions and a threshold discrepancy

All numbers were recomputed from the 1,200 frozen `result.json` files: 100 instances × 3 seeds ×
4 arms. `PACT_ZERO` is excluded from modality conclusions, leaving 900 non-OOD rollouts.

- **Hazard engaged:** `hazard_bar frames_with_contact > 0`.
- **Strict target engaged:** `grasp_target frames_with_contact >= 1`.
- **Sustained target engaged:** `grasp_target frames_with_contact >= 50`.
- **High contact:** `hazard_bar frames_with_contact > 500`.

The supplied 701/59/53/67/20 table exactly corresponds to the strict one-frame definition, not
the 50-frame definition. At 50 frames the five counts are 700/53/53/73/21. This report preserves
the 1-frame table as the literal “never touched the target” absorbing state and uses the
pre-specified 50-frame cut for the load-bearing high-contact analysis. Fifty audited physics
frames are approximately 100 ms of cumulative contact at 2 ms/frame; they need not be
contiguous. The cut was inherited from the supplied scan and was not adjusted after seeing these
results. Sensitivity at 10 and 100 frames is reported below. The machine artifact includes the
corresponding PACT_ZERO counts and reconciles to 1,200 rollouts at every threshold; the non-OOD
tables reconcile to 900.

## 1. The strict absorbing state is confirmed

### Pooled non-OOD pattern table

| Contact pattern | Rollouts | Task successes | Success rate |
|---|---:|---:|---:|
| Target engaged, never hazard | 701 | 489 | 69.8% |
| Hazard first, then target | 59 | 17 | 28.8% |
| Target first, then hazard | 53 | 17 | 32.1% |
| **Hazard engaged, target never touched** | **67** | **0** | **0.0%** |
| Neither engaged | 20 | 0 | 0.0% |

The counts sum to 900. The absorbing state spans 44 instance-seed cells across 22 unique
instances. It is exactly **0/67**, as hypothesized.

### Exact counts by arm

Each cell is `rollouts / task successes`.

| Pattern | ACT | PACT | PACT_PERMUTED | Pooled |
|---|---:|---:|---:|---:|
| Target engaged, never hazard | 229 / 159 | 251 / 171 | 221 / 159 | 701 / 489 |
| Hazard first, then target | 17 / 5 | 15 / 5 | 27 / 7 | 59 / 17 |
| Target first, then hazard | 16 / 5 | 24 / 7 | 13 / 5 | 53 / 17 |
| Hazard engaged, target never touched | 34 / 0 | 3 / 0 | 30 / 0 | 67 / 0 |
| Neither engaged | 4 / 0 | 7 / 0 | 9 / 0 | 20 / 0 |

The strict state is already much rarer under PACT: 3 rollouts versus 34 ACT and 30
PACT_PERMUTED. The next analysis uses the more tolerant “fewer than 50 target frames” definition
so that clip 4's 12-frame brush is classified as failure to sustain engagement.

## 2. High contact with negligible target engagement

### Threshold sensitivity

Counts are low-target rollouts divided by all `>500`-hazard-frame rollouts. Every low-target
group below has zero task successes.

| Target-engagement threshold | ACT | PACT | PACT_PERMUTED |
|---:|---:|---:|---:|
| 1 frame | 34/59 (57.6%) | **3/33 (9.1%)** | 30/58 (51.7%) |
| 10 frames | 34/59 (57.6%) | **4/33 (12.1%)** | 33/58 (56.9%) |
| **50 frames** | **34/59 (57.6%)** | **5/33 (15.2%)** | **33/58 (56.9%)** |
| 100 frames | 34/59 (57.6%) | **5/33 (15.2%)** | 33/58 (56.9%) |

The arm ordering and magnitude are insensitive to 10/50/100 frames. PACT's point estimate moves
only because two rare tails have 3 and 12 target frames; ACT is unchanged and PACT_PERMUTED
stabilizes by 10 frames.

### Seeds unpooled first

Intervals are deterministic 20,000-replicate paired whole-instance cluster-bootstrap intervals.
They are exploratory uncertainty summaries, not confirmatory inference. Counts appear alongside
every rate because some denominators are very small.

| Seed | ACT | PACT | PACT_PERMUTED | PACT − PERMUTED | PACT − ACT |
|---:|---:|---:|---:|---:|---:|
| 3101 | 13/21 (61.9%) | 1/4 (25.0%) | 11/15 (73.3%) | −48.3 pp [−88.9, +20.0] | −36.9 pp [−76.5, +30.0] |
| 3102 | 9/18 (50.0%) | 3/18 (16.7%) | 13/26 (50.0%) | −33.3 pp [−56.1, −10.5] | −33.3 pp [−62.0, −3.0] |
| 3103 | 12/20 (60.0%) | 1/11 (9.1%) | 9/17 (52.9%) | −43.9 pp [−70.0, −18.9] | −50.9 pp [−80.0, −19.0] |
| **Pooled** | **34/59 (57.6%)** | **5/33 (15.2%)** | **33/58 (56.9%)** | **−41.7 pp [−57.4, −25.0]** | **−42.5 pp [−61.2, −23.1]** |

Seed 3101 has only four PACT high-contact rollouts and is correspondingly uninformative; its
point difference has the same sign but its interval spans both directions. Seeds 3102 and 3103
carry most of the PACT tail and reproduce the difference.

### Does different tail membership explain everything?

Because proximity reduces entry into the high-contact tail, conditioning on high contact is a
post-treatment selection. To partially probe this, the same calculation was restricted to
matched instance-seed cells where **both** PACT and its comparator exceed 500 hazard frames.

| Fixed matched subset | PACT | Comparator | Difference |
|---|---:|---:|---:|
| Both PACT and PACT_PERMUTED high | 5/31 (16.1%) | 13/31 (41.9%) | **−25.8 pp [−45.0, −8.3]** |
| Both PACT and ACT high | 4/22 (18.2%) | 11/22 (50.0%) | −31.8 pp [−60.0, 0.0] |

The valid same-weights contrast persists when both policies are already in the tail: nine matched
cells are low-target only under PACT_PERMUTED, versus one only under PACT. This is evidence that
the finding is not purely a consequence of PACT having a smaller tail. But the subset is still
selected using policy outcomes and contains only 31 cells, so it does not cleanly identify a
causal “stay on task after impact” mechanism.

## 3. Where does the trajectory diverge?

Strict absorbing-state rollouts contact the hazard early: median first hazard contact is step
**55** (middle 50% 46.5–59.5; n=67). Across those 22 difficult instances, 63 successful
rollouts occur on 15 instances; only 11/63 touch the hazard at all, and those 11 have median first
contact step **154**.
Restricting further to the same instance-seed cells leaves only 4 successful hazard-contact
controls, with median step **146**.

This is consistent with early en-route diversion, but the planned geometric tests cannot be
performed:

- Only 2/1,200 trajectory H5s survive, both PACT_PERMUTED boundary rows, and neither is an
  absorbing-state rollout. One is a successful control in a relevant instance-seed cell, but its
  matched absorbing ACT trajectory is absent, so no paired distance comparison is possible.
- No distance-travelled-before-contact summary was retained.
- No target-distance, end-effector-distance, or target-neighbourhood flag was retained.
- Per-step contact payloads were also compacted.

First-contact timing therefore cannot distinguish “stalled before reaching the target” from
“reached its neighbourhood without contact and was pulled away.” Future instrumentation needs
compact end-effector-to-target distance at first hazard contact and minimum target distance over
the episode, in addition to contact run lengths.

## 4. Contact ordering remains a negative result

At the strict one-frame target threshold:

| Arm | Hazard-first success | Target-first success | Difference, instance-bootstrap 95% interval |
|---|---:|---:|---:|
| ACT | 5/17 (29.4%) | 5/16 (31.2%) | −1.8 pp [−38.7, +41.0] |
| PACT | 5/15 (33.3%) | 7/24 (29.2%) | +4.2 pp [−35.1, +43.2] |
| PACT_PERMUTED | 7/27 (25.9%) | 5/13 (38.5%) | −12.5 pp [−46.7, +21.4] |
| **Pooled** | **17/59 (28.8%)** | **17/53 (32.1%)** | **−3.3 pp [−28.4, +25.1]** |

Whole instances—not individual rollouts—were resampled, keeping every arm and seed together.
No arm shows a stable ordering advantage, and all intervals are wide. At 50 and 100 target frames
the pooled groups are exactly 17/53 versus 17/53, a zero-point difference. The defensible result
is not that ordering has been proven irrelevant, but that the proposed hazard-first mechanism is
**not supported by these data**, pooled or per arm.

## 5. Where clip 4 sits

The frozen clip-4 row reproduces exactly:

- PACT, seed 3103, episode `e99dc657bfa7…`, intrusion side left;
- 12 grasp-target frames and 17,609 hazard frames;
- first target contact at step 151 and first hazard contact at step 128;
- task failure.

Its 12 target frames rank **12th-lowest of 300 PACT rollouts** (midrank bottom 3.8%) and
**5th-lowest of 33 PACT high-contact rollouts** (midrank bottom 13.6%). It is one of the five PACT
tails below the 50-frame engagement threshold.

The scene is broadly difficult rather than uniquely bad for this checkpoint: 10/12 arm-seed
cells are high-contact, including 8/9 non-OOD cells. ACT is high-contact for all three seeds and
still succeeds for seeds 3102 and 3103. On seed 3103, PACT_PERMUTED also barely touches the target
(3 frames), logs 18,385 hazard frames, and fails. PACT seed 3103 is therefore a rare exception to
PACT's aggregate on-task pattern, embedded in a scene/seed cell where scrambled proximity fails
the same way.

## Supported interpretation

Of the three candidate statements, the supported one is:

> **The current data cannot fully separate “proximity keeps the policy on task during contact”
> from a selection effect caused by proximity changing which rollouts contact at all.**

The evidence leans toward a real second mechanism: the PACT-versus-PERMUTED difference is large,
replicated in direction, threshold-robust, and remains in the matched both-high subset. What is
missing is the trajectory instrumentation needed to say whether PACT stays aimed at the target,
recovers after impact, or merely enters a different portion of the selected tail. The correct
claim is “consistent with improved target engagement during contact,” not a confirmed causal
mechanism.

Machine-readable artifact:
`diagnostics_output/pact_contact_endpoint/absorbing_failure_characterization.json`
(self-hash `108cbf6706d4dc687eb959aa5a5cb204848d3e85ef1fdf43044ec562307a62b7`).
