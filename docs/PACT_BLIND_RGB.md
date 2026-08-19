# PACT with the RGB camera blinded

## Design

The frozen ACT, PACT, and PACT_PERMUTED checkpoints were evaluated sighted and with wrist RGB replaced by the ImageNet mean. The same 25 instances and three checkpoint seeds were shared across all six cells (450 fresh subprocess rollouts, 12 fixed workers). No retraining, scene change, encoder change, or threshold change occurred.

Before rollout, the expected result was declared as task success collapsing for every blind arm while PACT retained lower panel contact. The permitted safety-only interpretation was: *proximity alone keeps the arm safe but cannot do the task*.

## Reconciliation

- Valid rows: 450/450.
- Recorded blind/blur flags matched the schedule: True.

## Absolute performance

### Sighted

| Arm | Collision-free task success | Manipulation success | Any hazard | Mean hazard frames |
|---|---:|---:|---:|---:|
| ACT | 36/75 (48.0%) | 37/75 (49.3%) | 16/75 (21.3%) | 2967.9 |
| PACT | 43/75 (57.3%) | 44/75 (58.7%) | 4/75 (5.3%) | 693.6 |
| PACT_PERMUTED | 37/75 (49.3%) | 37/75 (49.3%) | 14/75 (18.7%) | 1932.8 |
### Blind

| Arm | Collision-free task success | Manipulation success | Any hazard | Mean hazard frames |
|---|---:|---:|---:|---:|
| ACT | 0/75 (0.0%) | 1/75 (1.3%) | 48/75 (64.0%) | 12289.4 |
| PACT | 1/75 (1.3%) | 1/75 (1.3%) | 37/75 (49.3%) | 8113.7 |
| PACT_PERMUTED | 0/75 (0.0%) | 0/75 (0.0%) | 46/75 (61.3%) | 9934.5 |

## Blind contrasts

- PACT_minus_ACT: hazard frames -4175.7, 95% instance-cluster bootstrap CI [-7356.3, -1263.0]; collision-free success 1.3%, CI [0.0%, 4.0%].
- PACT_minus_PACT_PERMUTED: hazard frames -1820.8, 95% instance-cluster bootstrap CI [-2999.0, -784.8]; collision-free success 1.3%, CI [0.0%, 4.0%].

Seed-unpooled contrasts, any-contact rates, entries, penetration, failure taxonomy, and paired blind-minus-sighted degradation are recorded in `diagnostics_output/pact_blind_rgb/analysis.json`.

## Decision

both decision-bearing blind hazard-frame contrasts exclude zero in PACT's favor.

PROXIMITY_STANDALONE_CONTACT_BENEFIT
