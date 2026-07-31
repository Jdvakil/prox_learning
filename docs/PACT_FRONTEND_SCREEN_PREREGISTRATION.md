# PACT wider-front-end screen preregistration

## Scope and interpretation

This is a development screen of the frozen proximity representation, not a
confirmatory PACT-versus-ACT experiment. It may issue only one of the four
`FRONTEND_SCREEN_*` tokens below. It cannot issue or imply
`PACT_BENEFIT_ESTABLISHED`, `PACT_NO_CONFIRMED_BENEFIT`, or
`PACT_WORSE_THAN_ACT`.

The sole decision-bearing contrast is PACT minus PACT_ZERO. These arms use
the same trained policy checkpoint on the same 40 instances; PACT_ZERO zeros
all 40 proximity tokens at inference. ACT seed 3101 is retained as a
secondary sanity reference and is not retrained.

## Frozen front-end choice

The declared primary variant is a weight-shared, frozen 32-dimensional
embedding per sensor. Its causal input is unchanged: 32 native 8-by-8 frames
formed from eight policy observations and four sensor subframes per
observation. A conv stem and four-layer, four-head transformer feed a
32-dimensional bottleneck. Auxiliary heads retain the prior sensor-local
nearest-surface XYZ and validity outputs. A reconstruction head predicts the
latest 8-by-8 closeness map so the 32-dimensional policy input is explicitly
trained to preserve more surface geometry than the former three-number
front-end.

The frozen losses are validity BCE with weight 1, surface XYZ Smooth L1 with
weight 5, and latest-closeness reconstruction Smooth L1 with weight 2.
Active pixels receive an additional reconstruction weight of 4. Encoder seed
4201, 100 epochs, batch 256, learning rate 1e-4, and best held-out composite
loss selection are fixed.

Before policy training, the primary encoder must load as a frozen
`pact_surface_embedding_encoder_v1` checkpoint and pass all of:

- finite held-out mean/median XYZ error, within-2-cm rate, validity
  precision/recall, and active-pixel reconstruction MAE;
- mean XYZ error no greater than 5 cm;
- validity precision at least 98%;
- validity recall at least 98%;
- active-pixel closeness reconstruction MAE no greater than 0.15.

The prior 3-D encoder's held-out reference is reported, not used as an
outcome-tuned threshold: 3.26/1.88 cm mean/median error, 51.3% within 2 cm,
99.9% precision, and 99.9% recall. If the declared primary variant fails its
training-clean gate, PACT policy training stops. A raw-patch fallback would
require a separate, outcome-blind frozen amendment before its training; it
will not be selected after viewing screen outcomes.

## Dataset and policy training

The 255-episode dataset is re-materialized from the preserved raw ACT-style
files by copying only action, qpos/qvel, wrist RGB, the ordered 40-sensor raw
stream, sensor names, intrinsics, and per-timestep world-to-sensor
extrinsics. Each output episode must reproduce the semantic SHA-256 recorded
before the prior 3-D tokens were added. The frozen encoder then writes a
40-by-32 proximity embedding at each policy timestep while retaining XYZ and
validity only as diagnostics.

Only PACT is trained, at seed 3101. The fixed recipe remains ResNet-18,
seven-layer encoder/decoder, eight heads, hidden width 512, chunk 100,
learning rate 1e-5, batch 8, 2000 epochs, and beta 10. Checkpoint frequency
is 2000 epochs to conserve storage and does not alter optimization. ACT reuses
the exact seed-3101 checkpoint with SHA-256
`a5ebbf3d5537315337e17e0f28951de068ce6960974d0f282b77fcfcca672eb1`.
PACT_ZERO is not trained separately.

## Screen population and schedule

The screen contains 40 new held-out instances, balanced 20 left and 20 right,
and deterministically disjoint by episode ID from both R1 and R2 evaluation
manifests. Each instance receives ACT, PACT, and PACT_ZERO once, for 120
rollouts. Arm order is balanced with a frozen seed. The worker count is fixed
at eight. Each rollout uses a fresh subprocess and unique output directory.
No row with a valid scientific `result.json` may be replaced or rerun.

Schedule construction occurs after the frozen encoder and PACT checkpoint
exist so their SHA-256s can be embedded. The complete schedule, analysis
script, model inputs, evaluator, launcher, supervisor, and empty output root
are hash-bound before the first rollout. No endpoint field may be inspected
before that freeze.

The first schedule row is a mandatory launch smoke. It must produce one valid
scientific result before the pool is released. During that smoke, the
launching shell is killed and the detached supervisor, heartbeat, and
evaluator must survive or complete. The eight-worker supervisor uses
`setsid`/`nohup`, a durable PID, parent log, state, completion ledger, and
heartbeat. If indiscriminate external termination kills an in-flight cohort
before results exist, every result-free member of that cohort is recovered
together under a frozen recovery event; selected individual post-observation
reruns are forbidden. The first 20 minutes of full dispatch are used only to
measure throughput and revise ETA.

## Frozen endpoint and analysis

The primary endpoint is collision-free task success:

`task_success AND hazard_bar contacts == 0 AND other_environment contacts == 0`

Contact with `grasp_target` is exempt. The scene, sampler, contact taxonomy,
and already-passed environment adequacy gate are unchanged and are not
re-measured.

The primary contrast is PACT minus PACT_ZERO, paired by the 40 instances. It
uses 20,000 deterministic whole-instance bootstrap replicates and an exact
two-sided McNemar test, with both directional discordant-pair counts
reported. Per-arm collision-free success receives Wilson 95% intervals.
PACT minus ACT is secondary. Ordinary task success, contact totals and
contact-episode counts by class, failure taxonomy, and two-sided Fisher exact
comparisons are reported but are not screen-decision-bearing.

With only 40 pairs, this screen is intended to detect a large gap of roughly
15 percentage points. A smaller true effect may be missed.

## Frozen decision rule

| Token | Condition |
|---|---|
| `FRONTEND_SCREEN_SIGNAL_PRESENT` | PACT minus PACT_ZERO is at least +10 percentage points and the paired bootstrap 95% CI lower bound is above zero |
| `FRONTEND_SCREEN_WEAK_SIGNAL` | PACT minus PACT_ZERO is at least +5 percentage points but the signal-present rule is false |
| `FRONTEND_SCREEN_NO_SIGNAL` | PACT minus PACT_ZERO is below +5 percentage points |
| `FRONTEND_SCREEN_INCONCLUSIVE` | The fixed 120-row schedule does not reconcile |

Signal present authorizes only a new, separately preregistered confirmatory
run. Weak signal authorizes only a fresh 60-instance, two-arm extension.
No signal stops this bottleneck screen; the encoder is not tuned and
re-screened on these instances.
