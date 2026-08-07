# PACT held-out geometry generalization — attempt 3 preregistration

Attempt 3 is a power-only reallocation of the abandoned attempt-2 design. It was frozen without
opening or analyzing any of the nine v2 scientific results. The v2 artifacts remain preserved and
are marked `ABANDONED_PRE_INTERPRETATION`.

## Power rationale

The bootstrap clusters on instances. V2 therefore had an effective sample size of 25 per condition,
not 225 arm/seed rollouts. Scaling the established C0 modality gap of −9.3 percentage points (95%
CI [−14.3, −5.0] at 100 instances) to 25 instances doubles the approximate standard error and
produces an approximate interval of [−18.6, 0.0] percentage points. This put zero on the boundary
of the decision-bearing control and created a substantial chance of an inconclusive result even if
the established in-distribution effect remained unchanged.

At 40 instances, the corresponding approximate CI half-width is 7.35 percentage points. This is
still a modest experiment, but it materially improves the load-bearing control while reducing total
runtime.

## Frozen design

| Field | Value |
|---|---|
| Conditions | C0, C2, `Z_093` |
| Arms | PACT, PACT_PERMUTED |
| Checkpoint seeds | 3101, 3102, 3103 |
| Fresh instances | 40 per condition |
| Rollouts | 720 |
| Workers | 12 |
| Bootstrap | 20,000 deterministic instance-cluster replicates |

ACT is omitted because it is not decision-bearing: the primary modality contrast is PACT minus
PACT_PERMUTED, and PACT_PERMUTED uses the same trained weights with registered in-distribution
proximity tokens permuted. Absolute performance is reported for both retained arms.

`HALF_Y_030` is omitted while `Z_093` is retained because panel height changes which sensors can
observe the obstacle and is therefore the more informative held-out axis. C2 and `Z_093` provide
the required two shifted conditions. This selection was made without v2 policy outcomes.

## Reused expert gate

Phase 0 is not rerun. C0 passed at 11/12 clean expert successes, C2 at 12/12, and `Z_093` at
11/12. The dropped `HALF_Y_030` condition also passed at 11/12 and remains recorded in the v2
expert-screen artifact. No geometry was retuned.

## Frozen endpoints and analysis

Primary endpoints are hazard-bar contact frames per rollout and any-hazard-contact rate.
Collision-free task success is co-primary. Contact entries, maximum penetration, other-environment
contact, and manipulation success are secondary. Results are reported per seed before pooling.
PACT minus PACT_PERMUTED is bootstrapped by resampling whole instance clusters, moving all included
conditions and seeds together, for at least 20,000 deterministic replicates.

C0 must reproduce the established modality gap before either shifted condition is interpreted. A
reproduction requires negative PACT-minus-PERMUTED point estimates for both primary contact
endpoints and inclusion of the original −9.3 percentage-point and −1,980-frame reference values in
the corresponding C0 95% instance-cluster intervals. C0 need not itself exclude zero.

The decision tokens and their precedence remain unchanged:

- `GEOMETRY_GENERALIZES`: C0 reproduces; both modality gaps are negative in every shifted
  condition; both pooled-shifted contact CIs exclude zero below zero.
- `GEOMETRY_PARTIAL`: C0 reproduces and both gaps are negative in some but not all shifted
  conditions.
- `GEOMETRY_DOES_NOT_GENERALIZE`: C0 reproduces, but the shifted gap includes zero or reverses.
- `GEOMETRY_TEST_INCONCLUSIVE`: C0 fails to reproduce, fewer than two shifted conditions survive,
  or the schedule does not reconcile.

This experiment cannot award or replace a `PACT_*` confirmatory token. It qualifies the frozen
contact-endpoint result only. Checkpoints, encoder, contact taxonomy, scene classes, thresholds, and
endpoints remain unchanged. Every rollout uses a fresh subprocess, fixed arm order balancing, no
outcome-based row replacement, the scientific-result boundary rule, a launch smoke, a detached
setsid/nohup supervisor, and the 600-second stall watchdog.

PREREGISTERED_BEFORE_V3_POLICY_OUTCOMES
