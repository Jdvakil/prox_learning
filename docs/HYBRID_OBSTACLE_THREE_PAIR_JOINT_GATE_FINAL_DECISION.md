# Hybrid obstacle — three-pair joint gate, final decision

## Executive summary

The validated three-pair metric reproduces the identifiability audit exactly, and swapping it
in **does not** rescue the historical regression. The handoff's premise — that restoring
`J(seed1, seed2)` would reverse the historical decision — is falsified by direct measurement:
executions on the 17 historical frames went from **10 to 9**. One frame changed.

The reason is visible in the reproduction table and is arithmetic rather than subtle.
`J(seed1, seed2) = 0.000` on **every one** of the 17 frames, exactly as `J(seed0, seed2)`
does. Restoring a term that is identically zero across the whole group multiplies every
frame's agreement by the same 2/3 factor; it carries no information that distinguishes them
from one another. The calibration then moved the threshold down by a comparable factor —
0.225 → **0.166667**, a ratio of 0.74 against the 2/3 rescale — so the frames' position
relative to the threshold is very nearly preserved.

| | Two-anchor (previous) | Three-pair (this task) |
|---|---:|---:|
| Historical median agreement | 0.250 | 0.1667 |
| Historical max agreement | 0.375 | 0.2500 |
| Calibrated agreement threshold | 0.225 | 0.166667 |
| Historical frames executing | **10 / 17** | **9 / 17** |

Everything else is healthy and, in places, excellent. Calibration is feasible (1,352 pairs
from a 5,084 × 300 grid) with the 0.80 recall floor untouched, a bootstrap upper bound on
false activation of **0.00000**, calibration recall **1.000**, nested recall 0.997, nested
hard-active retention 0.979, and uncertainty acceptance of 0.977 / 0.999 / 0.980 on ordinary
zero / active / activity-inactive frames. Frozen inference is bit-identical across all three
seeds, including J12.

The residual nested and diagnostic failures are **purely temporal** — false-active run 5
(>2) with post-support persistence on nested, run 7 (>5) on the diagnostic set — and the
script classifies them as such. But the historical regression fails on **per-frame** grounds:
9 of 17 known false positives still execute against a requirement of zero. That is not a
clustering artifact and cannot be deferred to a temporal study, so the temporal token would
misrepresent the state.

Decision: **`THREE_PAIR_JOINT_GATE_OFFLINE_TRANSFER_FAILED`**, Case D. No live rollouts were
run (0 of 20), `development4` was not executed, and `confirmatory41` remains untouched.

## Exact cause of the prior metric mismatch

The previous task's diagnosis was correct about *what* differed and wrong about *what would
follow*. Both statements are worth keeping straight:

- **Correct:** the identifiability audit used `mean(J01, J02, J12)`; the two-anchor task used
  `mean(J01, J02)`. On the 17 frames that gives 0.167 versus 0.250, and the audit's reported
  median was indeed 0.167. Confirmed here to four decimals.
- **Wrong:** that the omission "reversed the historical decision". It did not. Because J12 is
  uniformly zero on those frames, the two metrics are related by a constant factor of exactly
  2/3 across the entire group — a monotone rescaling. A monotone rescaling cannot reorder
  frames, and the calibration re-derives the threshold on the same rescaled axis, so the set
  of frames above threshold is nearly invariant.

The one frame that did change sat at anchor 0.2353 / three-pair 0.1569, straddling the two
thresholds. Everything else kept its verdict.

## Starting and final commits

Resolved from the previous decision artifacts rather than the abbreviated forms.

| Repo | Expected | Observed | Modified |
|---|---|---|---|
| root | `bea058c` recorded / `a47fc72` actual HEAD | `a47fc72` → branch `qualify/hybrid-obstacle-three-pair-joint-gate-v1` | code added only |
| ACT | `69bda27` | `69bda27` | no — the evaluator condition was never reached |
| MolmoSpaces | `678f2eb` | `678f2eb` | no |

The previous decision JSON records `root_commit: bea058c` because it was written immediately
before the commit that carries it (`a47fc72`); both are reported rather than one silently
preferred. Remote `origin` → `Jdvakil/prox_learning`; two pre-existing stashes untouched;
nothing pushed. Runtime: Python 3.11.15, torch 2.7.1+cu126, numpy 2.4.6, CUDA 12.6, NVIDIA
A10.

## Immutable provenance

**67 checks, 0 failed**, including both ACT artifacts hashed on disk (`policy_best.ckpt`
`dd7cd108a64ce10e…`, `dataset_stats.pkl` `c8119b904bfc80d6…`, best epoch 1738). All three
full-data seeds match their recorded hashes and share one configuration hash apart from the
seed. Paired dataset tree `1fb68b3c…` unchanged, all 364 files mode 444. SafetyHead
`1fb2fc2b…`. Residual constants, magnitude-support bound, sensor order and `offsamples = 4`
unchanged. development4 rows `[106, 107, 108, 118]`; confirmatory41 41 rows,
`executed_in_this_task: false`. All 17 historical identities recoverable.

Bootstrap members remain unloadable in this mode; `assert_not_bootstrap` raises on the
ensemble manifest and on any member record carrying a `bootstrap_seed`.

## Reproduction of the identifiability-audit agreement

Run before calibration, on the exact 17 frames, used for verification only.

| Check | Result |
|---|---|
| 17 frames scored | ✓ |
| `J(seed0, seed2) = 0` on all 17 | ✓ |
| `J(seed1, seed2) ≈ 0` on all 17 | ✓ (exactly 0.000) |
| Three-pair median ≈ 0.167 | ✓ **0.1667** |
| Two-anchor median ≈ 0.250 | ✓ **0.2500** |
| Three-pair ≤ anchor everywhere | ✓ |

Full frame table (J02 and J12 are 0.000 on every row):

| J01 | three-pair | anchor | count |
|---:|---:|---:|---:|
| 0.750 | 0.2500 | 0.3750 | 5 |
| 0.647 | 0.2157 | 0.3235 | 1 |
| 0.611 | 0.2037 | 0.3056 | 1 |
| 0.500 | 0.1667 | 0.2500 | 2 |
| 0.471 | 0.1569 | 0.2353 | 1 |
| 0.421 | 0.1404 | 0.2105 | 3 |
| 0.400 | 0.1333 | 0.2000 | 2 |
| 0.385 | 0.1282 | 0.1923 | 1 |
| 0.000 | 0.0000 | 0.0000 | 1 |

**On the mask-comparison conflict.** The handoff specifies `changed_probability >= 0.5`; the
identifiability audit used `> 0.5`, and the previous task pinned the strict form. Rather than
choose by argument, both were computed and compared bitwise on these frames: the masks are
**identical**, so the distinction is empty here and the strict form was retained for
continuity with the audit whose values step 4 requires reproducing.

## Two-anchor versus three-pair definitions

Both are implemented and logged for every frame; only one controls execution.

- **Controlling** — `full_pairwise_agreement = mean(J01, J02, J12)`
- **Diagnostic** — `anchor_agreement = mean(J01, J02)`
- Binary mask at 0.5; two empty masks agree completely.

A test asserts that perturbing only the seed1–seed2 relationship moves the three-pair value,
so J12 cannot be silently dropped again.

## Joint grid and feasible pairs

Complete Cartesian product with boundaries: **5,084 activity × 300 agreement = 1,525,200
pairs**. Factorised axis screens (activity-alone retention; the three uncertainty acceptance
floors) leave 4,630 activity values and **13 of 300** agreement values; 3,427 pairs then
received full per-trajectory evaluation and **1,352 are feasible**. Every floor, the
selection ordering, the partitions and the bootstrap seed are byte-identical to the previous
task — a test compares the constants and the selection expression directly.

### Selected pair

| | Value |
|---|---:|
| Activity threshold | **0.99154764** (unchanged from the previous run) |
| Three-pair agreement threshold | **0.166667** |
| Bootstrap 95% upper bound, false activation | **0.00000** |
| Median active recall | 1.0000 |
| Activity-alone retention | 1.0000 |
| Quiet-frame acceptance | 0.9769 |

## Nested offline evaluation

Opened only after the deployment manifest (`f37337582e6f01b6…`) was frozen.

| Metric | Value | Gate |
|---|---:|---|
| Median active recall | 0.9972 | ≥0.75 ✓ |
| HARD_TRUE_ACTIVE retention | 0.9788 | ≥0.80 ✓ |
| Mean executed zero activation | 0.00503 | ≤0.02 ✓ |
| Ordinary-zero acceptance | 0.9769 | ≥0.80 ✓ |
| Active acceptance | 0.9987 | ≥0.80 ✓ |
| Activity-inactive acceptance | 0.9798 | ≥0.80 ✓ |
| Max trajectory abstention | 0.4600 | ≤0.50 ✓ |
| **Max false-active run** | **5** | **≤2 ✗** |
| **Persistent correction after support** | **yes** | **none ✗** |

Every coverage, rate and anti-degeneracy gate passes. Both failures are temporal, and the
report records that classification explicitly.

## Historical regression

| | Result | Requirement |
|---|---:|---|
| Frames not executing | 8 / 17 | ≥16 |
| Rejected specifically by disagreement | 8 | ≥16 |
| **Frames executing** | **9** | **0** |

All 17 pass the activity gate (activity ≥ 0.9998), so disagreement is the only thing that can
stop them — unchanged from the previous task. Under the three-pair metric at threshold
0.166667, the nine frames at agreement ≥ 0.1667 execute and the eight below it are rejected.

**Frames changed by restoring J12: one, net.** Ten executed under the two-anchor contract, nine
under three-pair. Within a fixed threshold of 0.166667 the two metrics disagree on 7 frames,
which is what the per-frame `decision_changed_by_j12` flag records — but that comparison holds
the threshold fixed, and the calibration does not: it re-derives the threshold on the rescaled
axis, which is why the net effect collapses to one frame.

## Reused diagnostic audit

Labelled `reused_nonconfirmatory_diagnostic`.

| Metric | Value | Gate |
|---|---:|---|
| Median active recall | 0.9706 | ≥0.70 ✓ |
| Mean executed zero activation | 0.00251 | ≤0.03 ✓ |
| Ordinary-zero acceptance | 0.9793 | ≥0.75 ✓ |
| **Max false-active run** | **7** | **≤5 ✗** |

The run of 7 is the original failing episode's onset burst, executing end to end for the third
consecutive task.

## Frozen inference repeatability

24 repeats on a fixed 256-frame batch across all three seeds: activity max delta **0.0**,
three-pair agreement max delta **0.0**, J01/J02/**J12** identical, execution decisions
identical. No training kernels invoked. The gate is exactly reproducible; nothing here is
numerical noise.

## Conditional live development

Not run. Live execution is gated on calibration, nested evaluation, historical regression and
the diagnostic audit all passing; three of the four failed. **0 of 20 permitted rollouts**, no
`ACT_PLUS_THREE_PAIR_JOINT_GATE` evaluator integration, and no live outcome, contact, task or
saturation result is claimed.

## Clustering and persistence results

Worth separating from the headline, because it is the one place the picture is unambiguous.
Across nested and diagnostic, **every non-temporal gate passed**: coverage, aggregate false
activation, all three acceptance floors, abstention caps, cosine and hard-active retention.
The only failures anywhere in the offline transfer stage, other than the historical regression,
are:

- nested max false-active run 5 against a ceiling of 2
- nested persistent correction after oracle support ends
- diagnostic max false-active run 7 against a ceiling of 5

No temporal debounce, hysteresis, warm-up, cooldown or residual reset was added; a test
asserts none of those terms appears in the calibration source.

## Case classification

**Case D.** Another offline transfer failure occurred: the historical regression fails on
per-frame decisions with 9 of 17 known false positives executing.

This is deliberately *not* Case C. The nested and diagnostic failures in isolation would
qualify for `THREE_PAIR_TEMPORAL_CLUSTERING_REMAINS`, and the script computes that
classification and records it. But awarding the temporal token would authorize a temporal
gating study while nine known false positives still execute on per-frame grounds — the exact
failure the chain exists to remove. A test asserts the temporal token is not taken while the
historical regression fails.

## Confirmatory41 and development4

Both untouched. confirmatory41: 41 rows, `executed_in_this_task: false`, manifest sha256
`7b4500e9b4b2868e2612d7e444c34762d72c5e6e7b4b7c38bcf31f027b51b69e`. development4 rows
`[106, 107, 108, 118]`, not executed and not used for fitting; tests confirm neither manifest
is referenced by the calibration source, and the historical frames are read only after the
deployment manifest is frozen.

## Constraints honoured

No model trained or fine-tuned. No additional seeds. Bootstrap members unused and unloadable.
Nothing averaged. Seed-0 parked-field prediction unchanged; seeds 1 and 2 yield only
`changed_probability`, asserted by AST. ACT and Safety-CVAE unmodified. Residual constants and
magnitude-support bound unchanged. Paired dataset and partitions unmodified. The 0.80 recall
floor was not lowered. Pixel threshold still 0.5. Clustering and persistence gates unchanged.
No temporal logic introduced. The historical 17 frames, development4 and confirmatory41 were
not used for fitting. 0 of 20 live rollouts. Nothing pushed.

## Changed files

Added: `scripts/hybrid_obstacle_three_pair_{reproduction,calibrate,decision}.py`;
`configs/hybrid_obstacle_three_pair_joint_gate_v1.json`;
`tests/test_three_pair_joint_gate.py`;
`diagnostics_output/hybrid_obstacle_three_pair_joint_gate/*.json`; this document.
`causal_parked_skin/joint_gate.py` already carried both agreement functions and was not
modified. Nothing under `submodules/` was touched.

## Reproduction commands

```bash
SCR=<scratchpad>
PY=/root/act_retrain_venv/bin/python
D=diagnostics_output/hybrid_obstacle_three_pair_joint_gate
ACTDIR=/root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep

# 1. provenance
$PY scripts/hybrid_obstacle_reference_threshold_provenance.py \
    --act-checkpoint-dir $ACTDIR --out $D/provenance_verification.json

# 2. golden reproduction of the identifiability-audit agreement values
$PY scripts/hybrid_obstacle_three_pair_reproduction.py \
    --cache $SCR/parked_skin_cache --checkpoint-root $SCR/final \
    --stack configs/hybrid_safety_stack_v1.json --safety-dir assets/safety/cvae_v3 \
    --onset-audit diagnostics_output/hybrid_obstacle_prox_activity_gate/onset_attribution.json \
    --out $D/agreement_reproduction.json

# 3. joint calibration -- the single changed argument is --agreement-metric
$PY scripts/hybrid_obstacle_three_pair_calibrate.py \
    --cache $SCR/parked_skin_cache \
    --partition configs/hybrid_obstacle_prox_activity_partition_v1.json \
    --checkpoint-root $SCR/final --stack configs/hybrid_safety_stack_v1.json \
    --safety-dir assets/safety/cvae_v3 --agreement-metric three_pair \
    --groups diagnostics_output/hybrid_obstacle_activity_identifiability/diagnostic_groups.json \
    --onset-audit diagnostics_output/hybrid_obstacle_prox_activity_gate/onset_attribution.json \
    --deployment-manifest-out configs/hybrid_obstacle_three_pair_joint_gate_v1.json \
    --out $D/three_pair_calibration.json

# 4. decision
$PY scripts/hybrid_obstacle_three_pair_decision.py \
    --provenance $D/provenance_verification.json \
    --reproduction $D/agreement_reproduction.json \
    --calibration $D/three_pair_calibration.json \
    --manifest configs/hybrid_obstacle_three_pair_joint_gate_v1.json \
    --partition configs/hybrid_obstacle_prox_activity_partition_v1.json \
    --previous-decision diagnostics_output/hybrid_obstacle_full_seed_joint_gate/final_decision.json \
    --out $D/final_decision.json

# the two-anchor run is reproducible from the same script for comparison
#   --agreement-metric anchor

# verification
$PY -m pytest tests/test_three_pair_joint_gate.py -q
$PY -m ruff check causal_parked_skin/ scripts/hybrid_obstacle_three_pair_*.py
```

## Next recommended task

The metric question is now closed, and closing it narrowed the problem considerably.

1. **Stop pursuing agreement-metric variants.** Both definitions have been calibrated under
   the identical contract, and they differ by a monotone rescaling on the frames that matter.
   No reweighting of the same three pairwise Jaccards will separate those nine frames, because
   J02 and J12 are identically zero across all seventeen — the only discriminative term is J01,
   and it spans 0.385–0.750 on frames that must all be rejected while genuine activity must be
   retained.
2. **The nine executing frames and the temporal clustering are probably one problem, not
   two.** The diagnostic run of 7 is the same onset burst that supplies most of the executing
   historical frames. A temporal-gating study is now well motivated — it is the only offline
   failure left once the historical frames are set aside — but it should be scoped to test
   whether suppressing the burst also removes the per-frame executions, rather than assuming
   they are independent.
3. **If a temporal study is authorized, predeclare that it may not relax the per-frame
   contract.** The risk is that a debounce window hides the nine frames without the underlying
   disagreement signal ever distinguishing them, which would buy a passing gate without a
   safety improvement.

THREE_PAIR_JOINT_GATE_OFFLINE_TRANSFER_FAILED
