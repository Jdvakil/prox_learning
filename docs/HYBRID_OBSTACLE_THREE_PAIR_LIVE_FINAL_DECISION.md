# Three-pair joint gate — live development evaluation

**Date:** 2026-07-28
**Condition executed:** `ACT_PLUS_THREE_PAIR_JOINT_GATE`
**Schedule:** development4, 4 candidates × 5 repeats = 20 rollouts
**Owner override:** deliberate; the previous five tasks all stopped at the same offline
gate, and none of them could answer whether the predicted bursts actually cause harm.
**Decision:** `THREE_PAIR_LIVE_DEVELOPMENT_PASSED`

## Primary question

> Do false-positive onset bursts cause closed-loop harm?

**No — and the sharper finding is that the bursts essentially did not occur.**

Five consecutive offline tasks were blocked by 17 historical false positives clustered at
trajectory onset, 16 of them in frames 0–6, which projected to multi-frame bursts of
unwanted correction at the start of every episode. Across 20 live rollouts and roughly
3,900 control frames, the contract produced **one** false-positive frame. It was a burst of
length 1. Its peak arm deviation was **0.000144 rad** — 0.04 % of the 0.35 rad per-joint
cap, about 0.008°, below what the actuator resolves. Ten frames later the residual had
decayed with no persistence. There was no hazard-bar contact in any rollout.

The offline analysis was not wrong about the frames; it was wrong about the consequence.
It measured what the gate *would fire on* over a frame distribution, and treated firing as
harm. In closed loop the correction is a bounded, decaying residual on a 7-DoF arm, and a
single isolated frame of it is not an event the environment can register.

## Provenance

67 checks, 0 failed, `all_matched: true`. Verified before the first rollout: three seed
checkpoints by SHA-256, the frozen SafetyHead, the ACT policy (`policy_best.ckpt`, epoch
1738, `dd7cd108a64ce10e…`), `dataset_stats.pkl` (`c8119b904bfc80d6…`), the sensor order
hash, the pixel-mask threshold, and both thresholds against the frozen manifest.

| Item | Value |
|---|---|
| Manifest | `configs/hybrid_obstacle_three_pair_joint_gate_v1.json` |
| Activity threshold | 0.9915476441383362 |
| Agreement threshold | 0.16666666666666666 |
| Controlling metric | `full_pairwise_agreement` (three-pair) |
| Deployment predictor | seed 0, sole source of any correction |
| Seeds 1 and 2 | read for changed-pixel masks only |

Nothing was trained, recalibrated, or modified. No threshold, controller constant, camera,
renderer, task, or environment was touched. `confirmatory41` was not executed.

## Rollout completion

20 of 20 finalized, 0 failures, 0 non-finite actions, gripper bitwise nominal everywhere.

| cand | successes | oracle-active frames | executed | veto | FP bursts | hazard-bar |
|---|---|---|---|---|---|---|
| 106 | 5/5 | 57–81 | = oracle | 0 | 0 | 0 |
| 107 | 5/5 | 23–27 | 22–23 | 0 | 0 | 0 |
| 108 | 3/5 | 0 | 0–1 | 0 | 1 (len 1) | 0 |
| 118 | 0/5 | 0 | 0 | 0 | 0 | 0 |

Mean oracle-active recall: 1.000 on 106, 0.913 on 107. Actuator saturation 0.000 in every
rollout. Median cosine between the executed correction and the oracle correction ≈ 0.99999
on 106.

## Development criteria — 10 / 10 pass

| Criterion | Result |
|---|---|
| all 20 rollouts finalized | PASS |
| candidate 106 ≥ 4/5 | PASS (5/5) |
| candidate 107 ≥ 4/5 | PASS (5/5) |
| pooled hazard-present ≥ 10/15 | PASS |
| no repeated controller-attributable contact | PASS |
| candidate 118 no systematic new contact | PASS (0/5) |
| no meaningful persistent displacement | PASS |
| gripper bitwise nominal | PASS |
| no invalid actions | PASS |
| confirmatory41 untouched | PASS |

## Three findings that qualify the pass

### 1. The uncertainty veto contributed nothing

Executed frames: 471. `ACTIVITY_ONLY_SHADOW` frames: 471. Veto frames: **0**.

The veto never fired once in 20 rollouts. Live, this controller was
behaviourally identical to a bare seed-0 activity gate. The entire calibration effort of
tasks 15, 16 and 17 — bootstrap ensembles, anchor agreement, three-pair agreement — bought
exactly zero interventions on this schedule. The pass is a pass for the *activity gate*;
the agreement term is, so far, unexercised weight.

### 2. The empty-mask convention inflated agreement on the one frame that mattered

On the single false positive: J01 = 0.000, J02 = 0.000, J12 = **1.000**, three-pair =
0.3333, comfortably above the 0.1667 threshold — so the veto passed it.

J12 = 1.0 because seeds 1 and 2 both produced **empty** masks, and the Jaccard convention
scores empty-vs-empty as perfect agreement. Two silent models were counted as two models
agreeing. That is exactly backwards on the frames this gate exists to catch: seed 0 firing
alone against two silent peers is the *strongest* available disagreement signal, and the
metric converts it into the second-highest agreement score obtainable. This is a design
flaw in the metric, not a tuning problem, and it is why the veto had no chance to fire on
the only frame where it was needed.

### 3. The task failures are not the controller's

Seven rollouts failed the task: 108 r0, 108 r2, and all five of 118.

- **118 (0/5):** the controller executed **0 frames** in all five repeats, and
  `executed_action` was **bitwise identical** to `nominal_action` on every frame of every
  repeat. The safety stack provably did nothing. These are ACT-level failures.
- **108 (2/5 failed):** 4 of 5 repeats are bitwise identical to nominal. r2 failed with
  zero intervention. r0 is the rollout with the single false positive, max arm delta
  3.018e-04 rad — 0.017° — which cannot plausibly flip a task outcome, and r2 demonstrates
  the same failure occurring with no intervention at all.

Candidate 118 is the negative control, and its other-environment contact vector is
`[0, 0, 0, 0, 0]` — identical to ACT-only. The earlier deployable reference produced
`[13, 12, 35, 20, 21]` on this same control and **failed** the gate. This controller does
not introduce contact where there is no hazard.

## Measurement limitation

Contact classes are resolved at episode granularity — per-frame contact classes are not in
the logged rollout schema. Hazard-bar contact counts are therefore episode totals, and a
contact cannot be timestamped against a specific burst frame. With one false-positive frame
and zero hazard-bar contacts anywhere, this does not affect the conclusion, but a run with
a meaningful burst rate would need per-frame contact logging to attribute harm properly.

## What this does and does not license

It licenses the development schedule only. It is 20 rollouts on 4 candidates. It says the
false-positive onset bursts that blocked five offline tasks do not materialize in closed
loop and do not cause harm on this schedule.

It does not establish that the agreement veto is useful — the evidence says it is inert —
and the empty-mask convention means it would not have caught the one frame it was built
for. Before this contract is worth more calibration effort, the empty-empty case needs a
different convention. No temporal-gating work and no `confirmatory41` execution follows
from this report.

## Artifacts

- `diagnostics_output/hybrid_obstacle_three_pair_live/provenance_verification.json`
- `diagnostics_output/hybrid_obstacle_three_pair_live/live_analysis.json`
- `diagnostics_output/hybrid_obstacle_three_pair_live/final_decision.json`
- `submodules/act/three_pair_joint_gate_driver.py`
- `scripts/hybrid_obstacle_three_pair_live_analysis.py`
- `scripts/hybrid_obstacle_three_pair_live_decision.py`
- `tests/test_three_pair_live.py`

THREE_PAIR_LIVE_DEVELOPMENT_PASSED
