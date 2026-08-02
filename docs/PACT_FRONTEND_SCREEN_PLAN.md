# Plan: widen the proximity front-end, screen it with 120 rollouts

## Why

The confirmatory run returned `PACT_NO_CONFIRMED_BENEFIT`, and the decisive number was not
PACT vs ACT. It was:

**PACT − PACT_ZERO = −0.3%, 95% CI [−1.6%, +0.9%]**

Zeroing all 40 proximity inputs at inference changed nothing. The policy was handed the
modality and ignored it. The interval is tight, so this is a real null, not low power.

The projection was not missing — `input_proj_proximity` exists with fan-in-normalized magnitude
7.48 against vision's 13.09. But its shape is `(512, 3)`: each sensor contributes **three
numbers**. The frozen encoder compresses an 8×8 depth patch to a single 3-D surface point at
3.26 cm mean error, with only 51.3% within 2 cm. Forty sensors × 3 floats = 120 numbers, from
2,560 raw depth values.

**Hypothesis:** the front-end is an information bottleneck, and the policy ignores proximity
because the tokens are too impoverished to beat a wrist camera that already sees the scene.

**Falsifiable prediction:** widen the front-end and **PACT − PACT_ZERO moves off zero.**

That gap is the right instrument. PACT and PACT_ZERO share identical weights and are evaluated
on identical instances, so the comparison is exactly paired and carries none of the seed noise
that contaminates PACT − ACT.

## What this experiment is — and is not

This is a **screen**, not a confirmatory test. 120 rollouts cannot establish that PACT beats
ACT and must never be reported as doing so. Its only job is to decide whether a full
confirmatory run is worth ~30 hours of GPU.

It can resolve a paired PACT − PACT_ZERO gap of roughly **≥15 percentage points**. Smaller true
effects will not be detected, and a null screen means "not worth the spend on this evidence,"
never "proximity does not help."

## What stays frozen and unchanged

Do not re-measure, re-tune, or re-litigate any of this:

- The corridor scene `pact_collision_corridor_v1`, sampling distributions, sensor layout
- Contact taxonomy `{grasp_target, hazard_bar, other_environment}`; target contact exempt
- Primary endpoint: `task_success AND hazard_bar == 0 AND other_environment == 0`
- Phase 1 environment adequacy — already `PACT_ENVIRONMENT_ADEQUATE` (Gate B 23/64 = 35.9%,
  Wilson [25.3%, 48.2%], one-outcome stable; Gate C 23/64 hazard contact; surface signal 54.0%
  inside 20 cm)
- The ACT training recipe: ResNet-18, 7-layer encoder/decoder, 8 heads, hidden 512, chunk 100,
  lr 1e-5, batch 8, 2000 epochs, β=10
- Wilson intervals, Fisher exact, whole-instance deterministic bootstrap

**Reuse the existing ACT checkpoint** `act_seed3101`
(`a5ebbf3d5537315337e17e0f28951de068ce6960974d0f282b77fcfcca672eb1`). ACT has no proximity
branch, so the front-end change cannot affect it. Do not retrain ACT.

## Step 1 — build the wider front-end

Change **only** the proximity representation. Keep the freezing discipline: the encoder is
trained offline, validated, frozen, and never updated during policy training.

Primary variant: the encoder emits a **per-sensor embedding of dimension 32** instead of a 3-D
point. Retain the surface point and validity as auxiliary heads so front-end quality stays
measurable and comparable.

Report the same held-out metrics as the current encoder so the comparison is meaningful:
mean/median Euclidean error, within-2-cm rate, validity precision and recall. Current baseline
to beat: 3.26 cm / 1.88 cm, 51.3%, 99.9% / 99.9%.

Record the parameter count and checkpoint SHA-256. Freeze before any policy training.

If the embedding variant cannot be made to train cleanly, the fallback is to project the raw
8×8 depth patch per sensor directly. That removes the encoder as a confound entirely and is a
sharper test of the bottleneck hypothesis, at the cost of the validated-front-end property.
Declare which variant is used before training.

## Step 2 — dataset and PACT training

Re-convert the ACT-style dataset so it carries whatever the new front-end consumes (raw
40-sensor stream, ordered sensor names, intrinsics, per-timestep world-to-sensor extrinsics).
Verify the conversion retains the proximity channel — a previous dataset silently dropped it.

Train **PACT only**, one seed, identical recipe. PACT_ZERO is not trained: it is the same
checkpoint with all 40 proximity tokens zeroed at inference.

Record best epoch, validation loss, and checkpoint SHA-256. Validation loss is a training
diagnostic, not evidence — the previous run's 0.0834 vs 0.0848 gap meant nothing.

## Step 3 — the 120-rollout screen

**40 instances × 3 arms × 1 seed = 120 rollouts.** Arms: ACT (reused seed 3101), PACT,
PACT_ZERO. Instances drawn fresh from the held-out pool; none may be a confirmatory instance
from the interrupted or completed runs.

Freeze before the first rollout: the schedule and its SHA-256, the analysis script and its
SHA-256, the go/no-go rule below, all checkpoint hashes, worker count, and a new empty output
root.

Execution requirements, carried over from the interruption incident:

- one fresh subprocess per rollout, unique output directory
- **one launch-smoke row first** — assert it writes a scientific `result.json` before releasing
  the pool
- launch under `setsid`/`nohup` so the pool survives loss of the controlling terminal; durable
  parent PID, log, and heartbeat on disk; **verify by killing the launching shell during the
  smoke row**
- boundary rule: a row is terminal once it produces a scientific result; rows lost to
  indiscriminate external process termination are infrastructure failures and are re-run, and
  when that happens **every** in-flight row is re-run, never a selected subset
- measure throughput over the first 20 minutes and record the observed rate

**Note on allocation.** The screen's endpoint needs only PACT and PACT_ZERO. Dropping the ACT
arm would buy 60 instances instead of 40 — a 50% power gain on the actual question. ACT is
retained here as a sanity reference against the 53.1% baseline. If the screen comes back
borderline, rerun as 60 × 2 rather than adding seeds.

## Step 4 — analysis and go/no-go

Primary: **PACT − PACT_ZERO**, paired by instance, with a paired bootstrap CI and McNemar on
discordant pairs. Report discordant pair counts explicitly — with n=40 they are the whole
signal.

Secondary, reported but not decision-bearing: PACT − ACT, per-arm collision-free success with
Wilson intervals, ordinary task success, contact totals by class, failure taxonomy.

Predeclare and freeze this rule before running:

| Outcome | Condition |
|---|---|
| `FRONTEND_SCREEN_SIGNAL_PRESENT` | PACT − PACT_ZERO point estimate ≥ +10 pp **and** paired CI lower bound > 0 |
| `FRONTEND_SCREEN_WEAK_SIGNAL` | point estimate ≥ +5 pp but CI includes 0 |
| `FRONTEND_SCREEN_NO_SIGNAL` | point estimate < +5 pp |
| `FRONTEND_SCREEN_INCONCLUSIVE` | schedule did not reconcile |

`SIGNAL_PRESENT` authorizes a full confirmatory run under a fresh preregistration.
`WEAK_SIGNAL` authorizes extending the screen to 60 × 2 arms, not a confirmatory run.
`NO_SIGNAL` means the bottleneck hypothesis is not supported by this evidence — report it and
stop; do not tune the encoder and re-screen against the same instances.

**A screen may never award `PACT_BENEFIT_ESTABLISHED`, `PACT_NO_CONFIRMED_BENEFIT`, or
`PACT_WORSE_THAN_ACT`.** Those remain reserved for a fully powered confirmatory run.

## Constraints

- Change only the proximity front-end. No scene, endpoint, taxonomy, recipe, or analysis change.
- Do not retrain ACT; reuse `act_seed3101`.
- Do not re-measure Phase 1 environment adequacy.
- Do not move the go/no-go rule after seeing any outcome.
- Do not replace or re-run any row that produced a scientific result.
- Work in `/root/prox_learning_pact_remediation`. Do not touch
  `/root/prox_learning_hybrid_safety`, its submodule checkouts, or `confirmatory41`.
- Check `pgrep -fc eval_act_obstacle_on_policy.py` before heavy GPU use; a paused 410-rollout
  experiment shares the A10 and cannot run concurrently at 8 workers.
- Do not commit rollout H5s, videos, or checkpoints. Do not push.

## Artifacts

- `docs/PACT_FRONTEND_SCREEN_DECISION.md` — last nonblank line is the exact screen token
- `diagnostics_output/pact_frontend_screen/{encoder,schedule,analysis,provenance,final_decision}.json`
- frozen encoder checkpoint with SHA-256 and held-out metrics recorded
- tests under `tests/`

## What a negative screen would mean

If the gap stays at zero with a materially richer front-end, the bottleneck hypothesis is
wrong and the interesting finding shifts: a BC-trained policy does not exploit proximity even
when the signal is good. That points at the learning signal rather than the sensor — the expert
avoids using privileged geometry and never reacts to a sensor reading, so imitating it may
never require reading one. That would be the next hypothesis to test, and it is a more
fundamental one.
