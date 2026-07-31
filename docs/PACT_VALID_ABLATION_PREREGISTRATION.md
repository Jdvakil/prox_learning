# PACT distribution-matched ablation preregistration

## Why this follow-up is required

The completed 32-D front-end screen honestly awarded
`FRONTEND_SCREEN_SIGNAL_PRESENT`, but its PACT − PACT_ZERO instrument is invalid.
The all-zero token is far outside the empirical training-token support and caused
global task derailment. The +70 percentage-point difference must not be treated
as evidence that the policy used proximity.

This follow-up replaces that instrument without retraining. It keeps the same
PACT seed-3101 checkpoint, same 40 instances, same scene, and same endpoint.

## Frozen ablation

The sole new arm is `PACT_PERMUTED`. At each control step it receives a complete
40-sensor × 32-D proximity frame drawn from a different randomly selected
training timestep. The live scene and supplied proximity frame are unrelated,
destroying information while keeping every input on empirical support.

The token plan is fixed with seed 2026073105 before evaluation. It contains 512
frames per rollout, samples without replacement from the frozen 199-episode
training split, and forbids consecutive source frames from the same episode.
No all-zero vector is injected. The mean-token sanity arm is not included and
may not be added conditionally after seeing this result.

## Design and endpoint

- 40 existing screen instances × one new arm × one repeat = 40 rollouts.
- PACT is the immutable completed screen result on each identical instance.
- PACT_PERMUTED uses the identical PACT checkpoint and normalization statistics.
- Primary endpoint remains collision-free task success: task success with zero
  `hazard_bar` and zero `other_environment` contacts.
- Primary contrast is paired PACT − PACT_PERMUTED.
- PACT_ZERO remains an OOD-robustness diagnostic only.

The full schedule, token-plan hashes, checkpoint hashes, analysis hash, runtime
hashes, output root, and eight-worker count are frozen before the smoke rollout.
Execution retains the proven detached smoke, fresh subprocesses, immutable
terminal boundary, indiscriminate group-recovery rule, outcome-blind monitoring,
and lossless storage compaction. Raw rows 0 and 39 remain intact.

## Frozen analysis and decision

The paired analysis uses 20,000 deterministic whole-instance bootstrap
replicates (seed 2026073106), exact two-sided McNemar on discordant pairs, and
per-arm Wilson 95% intervals. Task success, contact classes, and failure taxonomy
are secondary.

- `VALID_ABLATION_SIGNAL_PRESENT`: difference ≥ +10 pp and paired CI lower > 0.
- `VALID_ABLATION_WEAK_SIGNAL`: difference ≥ +5 pp but the signal rule is false.
- `VALID_ABLATION_NO_SIGNAL`: difference < +5 pp.
- `VALID_ABLATION_INCONCLUSIVE`: the 40-pair schedule does not reconcile.

Only `VALID_ABLATION_SIGNAL_PRESENT` clears Step 1. It still cannot establish
PACT benefit and cannot authorize confirmation until the ACT dataset-equivalence
audit also passes under a fresh confirmatory preregistration.

No token from `{PACT_BENEFIT_ESTABLISHED, PACT_NO_CONFIRMED_BENEFIT,
PACT_WORSE_THAN_ACT}` can be emitted by this follow-up.
