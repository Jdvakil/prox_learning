# PACT independent-seed replication preregistration

This step asks whether the 25-point seed-3101 PACT–ACT advantage is structural
or initialization luck before a 960-rollout confirmatory run is authorized. It
is not itself confirmatory and cannot award a PACT-benefit decision token.

The experiment reuses the exact 40 front-end-screen instances. It evaluates
ACT-3102, a newly trained PACT-3102, and `PACT_PERMUTED` as an inference-time
alias of that same PACT-3102 checkpoint. ACT-3102 is not retrained because the
audited non-proximity training payloads and normalization are identical. The
frozen 32-D encoder is not retrained or tuned.

The permutation remains the exact distribution-matched seed-2026073105 plan,
including its 900-step tensor horizon and its per-instance token-plan row. The
first smoke row is deliberately `PACT_PERMUTED`, so checkpoint loading, the
32-D front-end path, token routing, and the corrected horizon are all exercised
before the eight-worker pool is released.

## Frozen endpoint and analysis

Collision-free task success is ordinary task success with zero `hazard_bar` and
zero `other_environment` contacts. `grasp_target` contact is exempt. Results are
reported for seed 3101 and seed 3102 side by side and unpooled first, then
pooled. The three contrasts are:

- PACT − PACT_PERMUTED: proximity information.
- PACT_PERMUTED − ACT: architecture, training, or seed.
- PACT − ACT: the two combined.

Each seed uses paired whole-instance bootstrap intervals and exact McNemar on
discordant pairs. The pooled bootstrap resamples the 40 instance identities,
moving both seeds for an instance together. Arm rates use Wilson 95% intervals;
ordinary success, all contact classes, and failure taxonomy are secondary.

## Frozen decision rule

- `SEED_REPLICATION_CONFIRMED`: seed-3102 PACT − ACT is at least +10 points and
  PACT − PACT_PERMUTED is positive on both seeds.
- `SEED_REPLICATION_PARTIAL`: the seed-3102 combined advantage is at least +10
  points, but the modality contrast is non-positive on either seed.
- `SEED_REPLICATION_FAILED`: seed-3102 PACT − ACT is below +10 points.
- `SEED_REPLICATION_INCOMPLETE`: the 240-cell two-seed matrix does not reconcile.

Only `CONFIRMED` authorizes a separately preregistered confirmatory experiment.
No third seed may be trained to break a tie. `PACT_ZERO` is excluded because the
all-zero 32-D token is outside the observed training-token support.

The machine-readable preregistration is
`configs/pact_seed_replication_preregistration_v1.json`. It freezes the analysis
script hash, training recipe, exact model and permutation inputs, eight-worker
execution, smoke/detachment test, 20-minute outcome-blind throughput measure,
and lossless content-independent storage compaction before outcomes exist.
