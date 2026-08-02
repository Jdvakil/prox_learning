# Step 1: replicate the PACT advantage on an independent seed

## Why this before a confirmatory run

The wider front-end produced the first pro-PACT signal in four attempts, but everything rests on
**one seed per arm** — which is exactly what defeated the previous confirmatory run. Before
spending ~40 hours at the measured 0.4 rows/min, spend ~7 hours finding out whether the effect
is structural or initialization luck.

The screen's decomposition makes the concern concrete rather than theoretical:

**ACT 47.5% → PACT_PERMUTED 60.0% → PACT 72.5%**

| Contrast | Gap | What it measures |
|---|---:|---|
| PACT − PACT_PERMUTED | +12.5 pp, CI [−2.5, +27.5], p=0.23 | proximity **information** |
| PACT_PERMUTED − ACT | +12.5 pp | architecture / training / **seed**, not modality |
| PACT − ACT | +25.0 pp, CI [+7.5, +42.5], p=0.039 | the two combined |

An arm receiving *information-free* proximity tokens still beat ACT by 12.5 pp. Something
non-modal is contributing, and with one seed there is no way to tell whether it is architecture
or initialization.

## The question this answers

**Does the PACT advantage replicate on an independent seed?**

- Replicates → the effect is structural; a confirmatory run is justified.
- Does not replicate → the +25 pp was seed luck; stop, and 40 hours plus a retraction are saved.

## Design

Three arms, seed **3102**, evaluated on the **same 40 instances** used in the screen so the
comparison is paired against the existing seed-3101 results.

| Arm | Checkpoint | Action needed |
|---|---|---|
| ACT-3102 | `e98d98bad87e2762cef37eb953d9ab55fcb65ed6355d2d8e9a881f38ef48c8d4` | **reuse — no retraining** |
| PACT-3102 | to be trained | train with the 32-D front-end |
| PACT_PERMUTED-3102 | same as PACT-3102 | inference-time ablation only |

**120 rollouts.** Estimated ~2 h training + ~5 h evaluation at 0.4 rows/min.

ACT-3102 needs no retraining: the data-equivalence audit proved the non-proximity payloads
(`action`, `observations/images/wrist_camera`, `qpos`, `qvel`) hash identically across the old
and new conversions, with normalization files byte-identical and the same 199/56 split
assignments. Reuse it directly and record that justification.

## Frozen and unchanged

Do not re-measure, re-tune, or re-derive any of this:

- Corridor scene, sampling distributions, sensor layout, contact taxonomy
- Primary endpoint: `task_success AND hazard_bar == 0 AND other_environment == 0`;
  `grasp_target` contact exempt
- The frozen 32-D encoder `6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206`
  (837,700 params) — **do not retrain or re-tune the front-end**
- The 40 screen instances and their identities
- Training recipe, identical to seed 3101 in every respect except `--seed 3102`:
  batch 8, lr 1e-5, 2000 epochs, chunk 100, hidden 512, dim_feedforward 3200, enc/dec 7,
  `--camera_names wrist_camera`, state_dim 9, action_dim 8, episode_horizon 195,
  `--use_proximity --n_proximity_sensors 40 --proximity_feature_dim 32 --prox_tokens_per_sensor 1`
- The permuted-token ablation exactly as implemented for seed 3101 — same permutation scheme,
  same seed for the permutation itself
- Wilson intervals, exact McNemar on discordant pairs, paired whole-instance bootstrap

## Ablation

`PACT_PERMUTED` only. **Do not run PACT_ZERO as a primary arm** — zeroing is a proven-invalid
instrument in this representation: 0 of 1,247,040 32-D training tokens are zero, against 95.0%
of the old 3-D tokens. If you want it at all, label it an OOD-robustness probe and keep it out
of every primary and secondary comparison.

## Execution requirements

Carried forward unchanged from the interruption and screen runs, all of which earned their keep:

- one launch-smoke row first; assert a scientific `result.json` before releasing the pool.
  The last smoke caught a 512-frame/900-step horizon mismatch before any scientific result was
  written — verify the horizon and token plan explicitly this time
- fresh subprocess per rollout, unique output directory, fixed 8 workers
- `setsid`/`nohup` supervisor with durable parent PID, log, and heartbeat; verify by killing the
  launching shell during the smoke
- boundary rule: a row is terminal once it produces a scientific result; rows lost to
  indiscriminate external process termination are infrastructure failures and are re-run, and
  when that happens **every** in-flight row is re-run, never a selected subset
- outcome-blind first-20-minute throughput measurement, recorded
- all 120 rows must reconcile with 120 scientific results and 120 driver records

## Analysis, frozen before the first rollout

Primary: **PACT-3102 − PACT_PERMUTED-3102**, paired by instance, with paired bootstrap CI and
exact McNemar. Report discordant pair counts explicitly — at n=40 they are the whole signal.

Report the full decomposition for seed 3102, and pooled across both seeds:

| Contrast | Interpretation |
|---|---|
| PACT − PACT_PERMUTED | modality information |
| PACT_PERMUTED − ACT | architecture / training / seed |
| PACT − ACT | combined |

Also report per-arm collision-free success with Wilson intervals, ordinary task success, contact
totals by class, and the failure taxonomy. Report the seed-3101 and seed-3102 estimates
**side by side and unpooled first**, then pooled — the whole point is to see whether they agree.

## Predeclared outcome rule

Freeze this before running.

| Outcome | Condition |
|---|---|
| `SEED_REPLICATION_CONFIRMED` | PACT − ACT positive on seed 3102 with point estimate ≥ +10 pp, **and** PACT − PACT_PERMUTED positive on both seeds |
| `SEED_REPLICATION_PARTIAL` | PACT − ACT positive ≥ +10 pp on seed 3102, but PACT − PACT_PERMUTED ≤ 0 on either seed |
| `SEED_REPLICATION_FAILED` | PACT − ACT < +10 pp on seed 3102 |
| `SEED_REPLICATION_INCOMPLETE` | schedule did not reconcile |

`CONFIRMED` authorizes the full confirmatory run under a fresh preregistration.

`PARTIAL` means the advantage is real but **not modality-driven** — the extra tokens are acting
as a regularizer rather than a sensor. That is a genuine finding and worth reporting, but it does
not authorize a PACT-benefit confirmatory run; it redirects the question.

`FAILED` means the seed-3101 result was initialization luck. Report it and stop. Do not train a
third seed hoping for a better draw.

**No confirmatory PACT token may be awarded by this step.** `PACT_BENEFIT_ESTABLISHED`,
`PACT_NO_CONFIRMED_BENEFIT`, and `PACT_WORSE_THAN_ACT` all remain reserved for a fully powered
run.

## If this replicates — the confirmatory design to preregister next

160 fresh held-out instances × 3 arms × 2 seeds = 960 rollouts, ~40 hours at 0.4 rows/min.
Arms: **ACT, PACT, PACT_PERMUTED** — PACT_ZERO is retired as a primary arm. Carry the three-way
decomposition as a predeclared secondary, since conflating the modality and architecture
contrasts is how the +70 pp artifact arose.

## Constraints

- Do not retrain ACT; reuse `act_seed3102`.
- Do not retrain or re-tune the 32-D encoder.
- Do not change the permutation scheme between seeds.
- Do not move the outcome rule after seeing any result.
- Do not train additional seeds to break a tie.
- Work in `/root/prox_learning_pact_remediation`. Do not touch `/root/prox_learning_hybrid_safety`,
  its submodule checkouts, or `confirmatory41`.
- Check `pgrep -fc eval_act_obstacle_on_policy.py` before heavy GPU use; a paused 410-rollout
  experiment shares the A10 and cannot run concurrently at 8 workers.
- Do not commit rollout H5s, videos, or checkpoints. Do not push.

## Artifacts

- `docs/PACT_SEED_REPLICATION_DECISION.md` — last nonblank line is the exact outcome token
- `diagnostics_output/pact_seed_replication/{schedule,dispatch,analysis,provenance,final_decision}.json`
- PACT-3102 checkpoint SHA-256, best epoch, validation loss recorded
- tests under `tests/`
