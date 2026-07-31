# PACT distribution-matched ablation decision

This is a post-screen validity check, not a confirmatory experiment.
The all-zero 32-D ablation is excluded because it is out of distribution.

Decision: `VALID_ABLATION_WEAK_SIGNAL`

| Arm | Collision-free task success | Ordinary task success |
|---|---:|---:|
| PACT | 29/40 (72.5%, 95% Wilson 57.2–83.9%) | 29/40 (72.5%) |
| PACT_PERMUTED | 24/40 (60.0%, 95% Wilson 44.6–73.7%) | 24/40 (60.0%) |

Primary PACT − PACT_PERMUTED: +12.5 pp, paired bootstrap 95% CI [-2.5, +27.5] pp.
Discordant pairs: PACT-only success 8, PACT_PERMUTED-only success 3; exact McNemar p=0.2266.

## Execution integrity

The first smoke exposed a 512-frame/900-step horizon mismatch before any
scientific result was written. A pre-outcome amendment froze a 900-frame token
plan, new schedule, new dispatch, and new empty output root while leaving the
arm, seed, checkpoint, instances, endpoint, analysis, and thresholds unchanged.
The replacement smoke passed on attempt 0.

All 40 rows reconciled with 40 scientific results, 40 driver records, and no
abort. The fixed worker count was eight and the frozen first-20-minute
throughput was 0.4 rows/min. All 38 preselected eligible rows were compacted
with byte-exact recoverability; rows 0 and 39 remain fully unpacked.

The ACT data-equivalence audit passed, so ACT retraining is not required. It
also found exact zero in 95.0% of old 3-D training tokens versus 0/1,247,040 new
32-D embeddings, confirming the original screen's zero ablation was invalid.

No confirmatory PACT decision token is awarded by this check.

VALID_ABLATION_WEAK_SIGNAL
