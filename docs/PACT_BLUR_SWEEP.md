# PACT inference-time RGB blur sweep

## Decision

**NO_BLUR_ROBUSTNESS** — no positive-sigma PACT-minus-ACT collision-free-success contrast had a CI lower bound above zero.

This experiment measures robustness to inference-time vision degradation. Blur is out of distribution for every arm; it does not establish that proximity substitutes for vision under blur-aware training.

## Frozen design

- 25 instances shared across all 4 sigmas, three arms, and three checkpoint seeds.
- Sigmas: [0.0, 0.5, 1.0, 2.0].
- 900 fresh-subprocess rollouts; 20,000 deterministic instance-cluster bootstrap replicates.
- Collapse floor: 10% collision-free task success.

## Collision-free task success

| sigma | ACT | PACT | PACT_PERMUTED | PACT − ACT (95% CI) | PACT − PERMUTED (95% CI) |
|---:|---:|---:|---:|---:|---:|
| 0 | 36/75 (48.0%) | 44/75 (58.7%) | 36/75 (48.0%) | 10.7 pp [-5.3 pp, 26.7 pp] | 10.7 pp [4.0 pp, 17.3 pp] |
| 0.5 | 33/75 (44.0%) | 42/75 (56.0%) | 37/75 (49.3%) | 12.0 pp [-2.7 pp, 26.7 pp] | 6.7 pp [-1.3 pp, 14.7 pp] |
| 1 | 30/75 (40.0%) | 34/75 (45.3%) | 31/75 (41.3%) | 5.3 pp [-9.3 pp, 20.0 pp] | 4.0 pp [-4.0 pp, 12.0 pp] |
| 2 | 13/75 (17.3%) | 14/75 (18.7%) | 15/75 (20.0%) | 1.3 pp [-10.7 pp, 13.3 pp] | -1.3 pp [-9.3 pp, 6.7 pp] |

The PACT-minus-ACT point estimate was positive at every sigma, but it did not
grow monotonically and every positive-sigma interval included zero. The
within-instance PACT-minus-ACT slope interaction was -5.3 percentage points
per sigma (95% CI [-12.6, 1.4]), so the data do not support a widening
collision-free-success advantage as RGB degrades.

Seed-unpooled collision-free successes show substantial heterogeneity,
especially at sigma 2:

| sigma | seed | ACT | PACT | PACT_PERMUTED |
|---:|---:|---:|---:|---:|
| 0 | 3101 | 11/25 (44%) | 18/25 (72%) | 15/25 (60%) |
| 0 | 3102 | 12/25 (48%) | 11/25 (44%) | 10/25 (40%) |
| 0 | 3103 | 13/25 (52%) | 15/25 (60%) | 11/25 (44%) |
| 0.5 | 3101 | 10/25 (40%) | 18/25 (72%) | 15/25 (60%) |
| 0.5 | 3102 | 11/25 (44%) | 11/25 (44%) | 10/25 (40%) |
| 0.5 | 3103 | 12/25 (48%) | 13/25 (52%) | 12/25 (48%) |
| 1 | 3101 | 12/25 (48%) | 15/25 (60%) | 11/25 (44%) |
| 1 | 3102 | 6/25 (24%) | 9/25 (36%) | 8/25 (32%) |
| 1 | 3103 | 12/25 (48%) | 10/25 (40%) | 12/25 (48%) |
| 2 | 3101 | 5/25 (20%) | 4/25 (16%) | 3/25 (12%) |
| 2 | 3102 | 4/25 (16%) | 0/25 (0%) | 0/25 (0%) |
| 2 | 3103 | 4/25 (16%) | 10/25 (40%) | 12/25 (48%) |

## Hazard contact

PACT retained a clear contact-severity benefit even though the preregistered
robustness token was not earned. Lower values and negative contrasts are
better:

| sigma | ACT mean frames | PACT mean frames | PACT_PERMUTED mean frames | PACT - ACT (95% CI) | PACT - PERMUTED (95% CI) |
|---:|---:|---:|---:|---:|---:|
| 0 | 3,333 | 679 | 1,946 | -2,654 [-5,552, -364] | -1,267 [-2,935, 60] |
| 0.5 | 3,319 | 658 | 2,575 | -2,661 [-5,192, -484] | -1,918 [-4,104, -228] |
| 1 | 4,245 | 895 | 3,163 | -3,351 [-6,427, -592] | -2,268 [-4,818, -279] |
| 2 | 4,507 | 2,748 | 5,705 | -1,759 [-3,811, -22] | -2,957 [-5,474, -899] |

PACT also had fewer any-hazard-contact rollouts at every sigma: 4/75 versus
18/75 for ACT at sigma 0, 4/75 versus 17/75 at 0.5, 6/75 versus 21/75 at 1,
and 12/75 versus 25/75 at 2. The corresponding PACT-minus-ACT intervals all
excluded zero. Against PACT_PERMUTED, PACT had fewer any-contact rollouts at
all four sigmas, and each interval excluded zero.

This is evidence that usable proximity continued to reduce collision entry and
severity under RGB blur. It is not evidence for the preregistered headline
hypothesis that proximity preserves collision-free task completion as vision
degrades. At sigma 2 all arms approached the declared 10% collapse floor and
seed behavior diverged sharply, so claims about task robustness at that level
would be especially fragile.

## Ordinary task success

Ordinary task successes were ACT/PACT/PACT_PERMUTED: 37/75 (49.3%), 45/75
(60.0%), and 36/75 (48.0%) at sigma 0; 35/75 (46.7%), 44/75 (58.7%), and
37/75 (49.3%) at 0.5; 31/75 (41.3%), 35/75 (46.7%), and 31/75 (41.3%) at 1;
and 14/75 (18.7%), 16/75 (21.3%), and 17/75 (22.7%) at 2.

## Contact and slope endpoints

Hazard contact frames, any-contact rates, entries, penetration, task success, seed-unpooled results, all three contrasts, within-instance arm slopes, and arm-by-sigma interactions are recorded in `diagnostics_output/pact_blur_sweep/analysis.json`.

## Scientific boundary

This study qualifies but does not replace the confirmed contact-endpoint or geometry-generalization decisions. No checkpoint, encoder, demonstration, scene, threshold, or contact taxonomy was changed.

NO_BLUR_ROBUSTNESS
