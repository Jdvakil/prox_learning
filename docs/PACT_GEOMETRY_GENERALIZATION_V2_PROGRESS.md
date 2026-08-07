# PACT held-out geometry generalization — attempt 2 progress

Attempt 2 passed both preregistered expert-solvability stages and its frozen 900-row zero-shot
policy evaluation is running. This is a distinct experiment from attempt 1; attempt 1 and its
`GEOMETRY_TEST_INCONCLUSIVE` decision remain unchanged.

## Phase 0a: expert envelope map

The clean-success gate was at least 7/8 task successes with zero `hazard_bar` and zero
`other_environment` contact. Every registered row was retained, including sampling failures.

| Candidate | Task success | Clean success | Hazard episodes | Sampling failures | Gate |
|---|---:|---:|---:|---:|:---:|
| `PANEL_X=0.58` | 6/8 | 6/8 | 0/8 | 0 | fail |
| `PANEL_X=0.65` | 7/8 | 6/8 | 1/8 | 1 | fail |
| `PANEL_Z=0.85` | 8/8 | 8/8 | 0/8 | 0 | pass |
| `PANEL_Z=0.93` | 8/8 | 8/8 | 0/8 | 0 | pass |
| `PANEL_HALF[1]=0.18` | 6/8 | 5/8 | 1/8 | 2 | fail |
| `PANEL_HALF[1]=0.30` | 7/8 | 7/8 | 0/8 | 1 | pass |
| `ap_w=0.95` | 5/8 | 5/8 | 0/8 | 1 | fail |

The selection rule was frozen before Phase 0a and used threshold pass/fail only. It selected
`PANEL_Z=0.93` and `PANEL_HALF[1]=0.30`, from distinct axes. Envelope-map self-hash:
`b8cc4c589dba3a08c7107a5ca7958014c64b26a669f3e5c9a5bcdefa72b1f86a`.

## Phase 0b: confirmation screen

The two selected conditions were screened on 12 fresh expert episodes at a fixed gate of at least
10/12 clean successes. Both passed:

| Condition | Task success | Clean success | Hazard episodes | Gate |
|---|---:|---:|---:|:---:|
| `PANEL_Z=0.93` | 11/12 | 11/12 | 0/12 | pass |
| `PANEL_HALF[1]=0.30` | 11/12 | 11/12 | 0/12 | pass |

Carried conditions were not rerun: C0 was 11/12 and C2 was 12/12 in attempt 1. The four frozen
policy conditions are therefore C0, C2, `Z_093`, and `HALF_Y_030`. Expert-screen self-hash:
`947faeb60c4b06df1046b90d6d3c99bbddf18aa44d2154b1f884afc0e19aedd4`.

## Frozen policy experiment

- Main manifest: `5a31600a8b4d11798d4062d960d3906207bdb6f6a8030a61475c92e5ecdcd37e`
- Schedule: `725908201f3213198bc2ead1e0ea23a0b553ae2157c7aa74e8d1f74aed8cb8e3`
- Dispatch: `aed8c8fa3bb370c28120b66f89726f46f49222730d55bd09b69ea54590e7d503`
- Design: 4 conditions × 25 instances × 3 seeds × 3 arms = 900 rollouts
- Arms: ACT, PACT, PACT_PERMUTED
- Fixed workers: 8
- Analysis: unchanged attempt-1 analysis, 20,000 instance-cluster bootstrap replicates
- Detachment proof: passed, `f2780a014945fdc5564855c95ca3b3a94cbbd54e07586142646cbfa195113405`

The outcome-blind first-20-minute measurement recorded 8 policy rows at 0.4 rows/min and projected
37.125 execution hours for the remaining 891 rows. At the measurement boundary the scientific
schedule was unchanged and no endpoint fields had been read. Allow roughly one additional hour
after execution for reconciliation, frozen analysis, the final report, and commits.

## Current status

As of 2026-08-07 01:00 UTC, the detached supervisor was healthy with 9/900 total rows complete
(including the registered smoke), eight active rows, 883 pending, `abort_reason=null`, and all nine
completed rows compacted. No policy outcome has been inspected or analyzed. The final geometry
decision is intentionally not assigned until all 900 rows reconcile and the frozen analysis runs.

EVALUATION_IN_PROGRESS
