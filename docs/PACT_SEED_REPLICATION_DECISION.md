# PACT independent-seed replication decision

This is the pre-confirmatory seed replication, not a powered confirmatory experiment.
PACT_ZERO is excluded because the all-zero 32-D token is out of distribution.

Decision: `SEED_REPLICATION_FAILED`

## Seed 3101

| Arm | Collision-free task success | Ordinary task success |
|---|---:|---:|
| ACT | 19/40 (47.5%, 95% Wilson 32.9–62.5%) | 19/40 (47.5%) |
| PACT | 29/40 (72.5%, 95% Wilson 57.2–83.9%) | 29/40 (72.5%) |
| PACT_PERMUTED | 24/40 (60.0%, 95% Wilson 44.6–73.7%) | 24/40 (60.0%) |

| Contrast | Difference | Paired CI | Discordance | p |
|---|---:|---:|---:|---:|
| PACT − PACT_PERMUTED | +12.5 pp | [-2.5, +27.5] pp | 8 / 3 | 0.2266 |
| PACT_PERMUTED − ACT | +12.5 pp | [-7.5, +32.5] pp | 11 / 6 | 0.3323 |
| PACT − ACT | +25.0 pp | [+7.5, +42.5] pp | 12 / 2 | 0.01294 |

## Seed 3102

| Arm | Collision-free task success | Ordinary task success |
|---|---:|---:|
| ACT | 24/40 (60.0%, 95% Wilson 44.6–73.7%) | 24/40 (60.0%) |
| PACT | 21/40 (52.5%, 95% Wilson 37.5–67.1%) | 22/40 (55.0%) |
| PACT_PERMUTED | 15/40 (37.5%, 95% Wilson 24.2–53.0%) | 15/40 (37.5%) |

| Contrast | Difference | Paired CI | Discordance | p |
|---|---:|---:|---:|---:|
| PACT − PACT_PERMUTED | +15.0 pp | [+2.5, +27.5] pp | 7 / 1 | 0.07031 |
| PACT_PERMUTED − ACT | -22.5 pp | [-40.0, -5.0] pp | 3 / 12 | 0.03516 |
| PACT − ACT | -7.5 pp | [-25.0, +12.5] pp | 6 / 9 | 0.6072 |

## Pooled across both seeds

Whole instances are the bootstrap clusters; both seed outcomes for a sampled instance move together.

| Contrast | Difference | Instance-clustered CI |
|---|---:|---:|
| PACT − PACT_PERMUTED | +13.8 pp | [+3.8, +23.8] pp |
| PACT_PERMUTED − ACT | -5.0 pp | [-20.0, +10.0] pp |
| PACT − ACT | +8.8 pp | [-5.0, +22.5] pp |

No `PACT_BENEFIT_ESTABLISHED`, `PACT_NO_CONFIRMED_BENEFIT`, or `PACT_WORSE_THAN_ACT` token can be awarded by this replication step.

SEED_REPLICATION_FAILED
