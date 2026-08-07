# PACT held-out geometry generalization

This zero-shot evaluation uses the frozen PACT weights under live and preregistered permuted proximity; no policy, encoder, threshold, scene outcome, or endpoint was retuned.

## Decision

**GEOMETRY_GENERALIZES** — all shifted conditions favor PACT and both pooled shifted contact CIs exclude zero.

## Expert solvability gate

Only conditions passing at least 10/12 privileged-expert clean successes entered the policy schedule. A clean success is task success with zero hazard-bar and zero other-environment contacts.

## In-distribution control

C0 reproduction: **pass**.

| Modality contrast | C0 estimate | 95% instance-cluster CI | Original reference |
|---|---:|---:|---:|
| Any hazard contact | -10.0 pp | [-19.2 pp, -1.7 pp] | −9.3 pp |
| Hazard contact frames | -1,293.3 | [-2,755.7, -139.8] | −1,980 |

## Absolute arm performance

| Condition | Arm | n | Any hazard contact | Mean hazard frames | Collision-free task success | Task success |
|---|---|---:|---:|---:|---:|---:|
| C0 | PACT | 120 | 10/120 (8.3%) | 610.4 | 70/120 (58.3%) | 75/120 (62.5%) |
| C0 | PACT_PERMUTED | 120 | 22/120 (18.3%) | 1903.7 | 63/120 (52.5%) | 66/120 (55.0%) |
| C2 | PACT | 120 | 23/120 (19.2%) | 2047.4 | 50/120 (41.7%) | 55/120 (45.8%) |
| C2 | PACT_PERMUTED | 120 | 37/120 (30.8%) | 3793.7 | 41/120 (34.2%) | 46/120 (38.3%) |
| Z_093 | PACT | 120 | 20/120 (16.7%) | 2049.4 | 55/120 (45.8%) | 67/120 (55.8%) |
| Z_093 | PACT_PERMUTED | 120 | 34/120 (28.3%) | 3795.2 | 56/120 (46.7%) | 66/120 (55.0%) |

## Shifted conditions

| Condition | PACT − PERM any contact | PACT − PERM hazard frames | Both favor PACT |
|---|---:|---:|:---:|
| C2 | -11.7 pp | -1,746.3 | yes |
| Z_093 | -11.7 pp | -1,745.8 | yes |

Pooled shifted modality contrast:

- Any hazard contact: -11.7 pp, 95% CI [-18.3 pp, -5.4 pp].
- Hazard contact frames: -1,746.1, 95% CI [-2,794.3, -811.8].

Full absolute arm performance, seed-unpooled contrasts, collision-free task success, manipulation success, entries, penetration, failure taxonomy, counts, Wilson intervals, and all 20,000-replicate bootstrap intervals are in `diagnostics_output/pact_geometry_generalization_v3/analysis.json`.

This study qualifies the existing contact-endpoint result and cannot award or replace a PACT confirmatory token.

GEOMETRY_GENERALIZES
