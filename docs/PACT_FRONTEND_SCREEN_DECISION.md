# PACT wider-front-end screen decision

This was a 40-instance, 120-rollout development screen. It cannot establish PACT benefit over ACT.

Decision: `FRONTEND_SCREEN_SIGNAL_PRESENT`

## Frozen 32-D front-end

- Checkpoint SHA-256: `6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206`
- Parameters: 837,700
- Held-out surface error, mean / median: 3.20 / 1.69 cm
- Within 2 cm: 52.9%
- Validity precision / recall: 99.4% / 99.9%

## PACT training

- Seed: 3101
- Best epoch: 1796
- Best validation loss: 0.085737
- Checkpoint SHA-256: `9db867f5b2cf059f5fad56f2eebd2e0e27024bb511ee0d526ea50692c4cf1457`

## Frozen screen analysis

| Arm | Collision-free task success | Ordinary task success |
|---|---:|---:|
| ACT | 19/40 (47.5%, 95% Wilson 32.9–62.5%) | 19/40 (47.5%) |
| PACT | 29/40 (72.5%, 95% Wilson 57.2–83.9%) | 29/40 (72.5%) |
| PACT_ZERO | 1/40 (2.5%, 95% Wilson 0.4–12.9%) | 2/40 (5.0%) |

Primary PACT − PACT_ZERO: +70.0 pp, paired bootstrap 95% CI [+55.0, +82.5] pp.
Discordant pairs: PACT-only success 28, PACT_ZERO-only success 0; exact McNemar p=7.451e-09.

## Post-hoc instrument-validity amendment

The frozen rule honestly awarded `FRONTEND_SCREEN_SIGNAL_PRESENT`, and that
historical token is retained below. Its decision-bearing instrument was later
shown to be invalid: zero is absent from all 1,247,040 new 32-D training tokens,
whose norms occupy a tight shell around 6.31. The all-zero inference token is
therefore a severe out-of-distribution input, not “PACT without proximity.” The
+70 pp PACT − PACT_ZERO result must not be reported as modality evidence.

A preregistered distribution-matched replacement kept the identical PACT
checkpoint and 40 instances while supplying real training-token frames with
live-scene alignment destroyed:

| Arm | Collision-free task success | Ordinary task success |
|---|---:|---:|
| PACT | 29/40 (72.5%, 95% Wilson 57.2–83.9%) | 29/40 (72.5%) |
| PACT_PERMUTED | 24/40 (60.0%, 95% Wilson 44.6–73.7%) | 24/40 (60.0%) |

PACT − PACT_PERMUTED was +12.5 pp, paired bootstrap 95% CI
[-2.5, +27.5] pp; discordant pairs were 8 versus 3 and exact McNemar
p=0.2266. The frozen result is `VALID_ABLATION_WEAK_SIGNAL`: suggestive, but it
does not clear Step 1 and does not authorize a confirmatory run.

The independent ACT conversion audit found all 255 episode lengths and all
image, qpos, qvel, and action payloads exactly equal between the reused ACT and
new PACT conversions; splits, normalization statistics, and recipes also match.
ACT retraining is not required. In the old 3-D representation, 95.0% of tokens
were exactly zero, confirming that old and new zero ablations are not comparable.

The ACT comparison below remains a development-screen candidate signal only.
It cannot replace the invalid frozen primary endpoint or the failed
distribution-matched go/no-go rule.

## Secondary diagnostics

PACT − ACT was +25.0 pp, paired whole-instance bootstrap 95% CI
[+7.5, +42.5] pp. The unpaired Fisher exact comparison was p=0.0392.
This is a development-screen sanity reference, not confirmatory evidence.

| Arm | Episodes with target contact | Episodes with hazard-bar contact | Episodes with other-environment contact |
|---|---:|---:|---:|
| ACT | 35 | 6 | 0 |
| PACT | 37 | 2 | 0 |
| PACT_ZERO | 27 | 21 | 0 |

Contact pair-entry totals were:

| Arm | `grasp_target` | `hazard_bar` | `other_environment` |
|---|---:|---:|---:|
| ACT | 7,573,898 | 163,549 | 0 |
| PACT | 10,622,992 | 135,165 | 0 |
| PACT_ZERO | 637,970 | 215,196 | 0 |

The frozen failure taxonomy was:

| Arm | Collision-free task success | Hazard-bar contact | Target contact without task success | Task failure after gripper close |
|---|---:|---:|---:|---:|
| ACT | 19 | 6 | 15 | 0 |
| PACT | 29 | 2 | 8 | 1 |
| PACT_ZERO | 1 | 21 | 13 | 5 |

## Execution and storage integrity

All 120 scheduled rows reconciled with 120 scientific results and 120 driver
records. The detached smoke proof passed, the full run used the frozen
eight-worker count, and the first-20-minute outcome-blind throughput measurement
was 0.4 rows/minute.

The outcome-blind storage amendment compacted and verified all 118 eligible
rows. The original result and trajectory payloads remain byte-exactly
recoverable. Schedule rows 0 and 119 remain fully unpacked, and no endpoint value
was emitted during compaction.

This token is a screen decision only; it is retained as historical output of
the frozen rule but rests on an invalid instrument. No confirmatory PACT
decision token may be inferred from it.

FRONTEND_SCREEN_SIGNAL_PRESENT
