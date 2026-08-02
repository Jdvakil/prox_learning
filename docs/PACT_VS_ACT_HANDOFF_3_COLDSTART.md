# Handoff: re-run the PACT pilot evaluation and adjudicate Gates B and C

Self-contained. Assumes no prior context. Verify every claim below against the artifacts
rather than trusting this document.

## Working directory — read first

```
/root/prox_learning_pact_remediation
```

This is a **git worktree** of `/root/prox_learning`, currently on branch
`experiment/pact-vs-act-remediation-v2` (commit `2ac0746`). It has its **own**
`submodules/molmospaces` checkout at `388bdd2`.

Do **not** work in `/root/prox_learning_hybrid_safety`, and do not check out branches there.
That is a separate clone of the same GitHub repository running a different experiment whose
pinned stack requires `submodules/molmospaces` at `678f2eb`. A previous PACT attempt switched
that shared checkout and caused 320 of 410 rollouts in the other experiment to be refused by
its integrity guard. Two clones, one remote — stay in yours.

Environment:

```bash
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR='/root/prox_learning_pact_remediation/assets'
export PYTHONPATH='/root/prox_learning_pact_remediation/submodules/molmospaces'
```

Interpreter: `/root/act_retrain_venv/bin/python`. Before heavy GPU use, check
`pgrep -fc eval_act_obstacle_on_policy.py` — a paused 410-rollout experiment shares the A10.

## The objective

Establish whether **PACT beats ACT**, on three arms:

1. `ACT` — wrist RGB + 9-D proprioception only
2. `PACT` — same, plus 40 proximity skin streams
3. `PACT_ZERO` — trained PACT weights, proximity zeroed at inference

**Primary endpoint — collision-free task success:**

```
task_success AND hazard_bar == 0 AND other_environment == 0
```

Contact with the grasp target is expected and **exempt**. Contact with the intruding panel
(`hazard_bar`) or any other scene geometry fails the rollout. Ordinary task success is a
separate secondary endpoint. `minimum_clearance_m` is a known-broken metric — keep it excluded.

## Background you need, and must not re-derive

The experiment takes the **collision route**: these sensors measure nearby *surfaces*, not
object position. An earlier fridge-scene attempt proved the object route is dead here — across
196 episodes × 40 sensors × ~280 steps the target was visible in only 99 sensor-frames, none
inside 20 cm, none at grasp. Do not attempt object-position encoding. Gate A (object signal)
is therefore **not applicable**.

A prior three-arm attempt on collisions scored ACT 0 static contacts vs PACT 835 — PACT was
much *worse*. That outcome must remain reportable; this is not an experiment to win.

## Current state — verify each of these

**Phase 1 environment adequacy: PASSED. Do not re-measure or re-litigate.**

| Measure | Result | Threshold |
|---|---:|---:|
| Active-panel signal inside 20 cm | 54.0% (95% bootstrap 46.0–61.9) | ≥30% |
| Active-panel signal inside 12 cm | 18.1% (95% bootstrap 13.5–22.8) | ≥5% |
| Active scientific expert episodes | 62/62 | ≥5/6 |
| Usable clean demonstrations | 58/64 | ≥48 |
| Expert no-outcome rate | 2/64 (3.1%) | <5% |

Robust to leave-one-episode-out. Scene `pact_collision_corridor_v1`, manifest
`cba7ff8879565e72…`, master seed `2026072901`, expert clearance 0.10 m.

**Pilot ACT baseline: trained, reuse it. Do not retrain.**

```
/root/pact_remediation_artifacts_v2/pilot/act_seed1101/policy_best.ckpt
  sha256 4fca3b0b0542d6ae65c7d44f1fd562cd376199532f91b08aaf5722109a858db6
  seed 1101, best epoch 1849, validation loss 0.093538
/root/pact_remediation_artifacts_v2/pilot/act_seed1101/dataset_stats.pkl
  sha256 fd1150e45470cbcd2912f83717e9576367439200378f8999e47c5c75f2b8bd0a
```

Note artifacts live **outside the repo**, under `/root/pact_remediation_artifacts_v2/`.

**Pilot evaluation: failed on a build error. This is what you are here to fix.**

All 64 rows of the pilot ACT evaluation are `invocation_failure`. Recorded root cause: *"relative
manifest path was invalid from evaluator cwd"* — every subprocess died before loading the
manifest, so there are **zero** scientific outcomes. Ledger at
`/root/pact_remediation_artifacts_v2/pilot_eval_e0515adf/` (64 files).

The path bug is **already repaired** — verify at
`scripts/run_pact_confirmatory_schedule.py:67-68` (manifest and output paths resolved before
the working-directory change), with regression tests in `tests/test_pact_confirmatory_schedule.py`
and `tests/test_pact_collection_runner.py`. Confirm these exist and pass before proceeding.

Consequently **Gates B and C have never been measured** — `n: 0`, `inconclusive` in
`diagnostics_output/pact_vs_act/analysis.json`.

## Step 1 — amend the terminal-outcome rule, and freeze it before use

The previous run treated those 64 invocation failures as terminal and refused to re-run them,
awarding `PACT_EXPERIMENT_INCOMPLETE`. That is a mis-scoped rule, not a finding. Write this
into the preregistration and hash it **before** launching anything:

> A row becomes outcome-bearing at the moment an initial observation is accepted. Any failure
> strictly before that point — invocation failure, import error, unresolvable path, missing
> file, startup OOM, CUDA init failure — is an **infrastructure failure**: retryable without
> limit, recorded with cause and retry count, reported separately from scientific outcomes,
> and unable to fail a science gate or award a decision token.
>
> Once an initial observation is accepted, the row is terminal: no later exception, contact,
> task failure, or success may cause replacement or rerun.

The existing protocol already permitted retries for failures "before an initial observation is
accepted and before any action." An invocation failure is strictly earlier than that.

## Step 2 — justify the re-run in the report

State explicitly, do not assume:

- zero observations, zero actions, zero scientific outcomes were produced;
- the fix predates knowledge of any result;
- path resolution is **content-independent** and cannot bias which rows succeed.

Re-running rows that produced no outcome is not outcome-based replacement — there was no
outcome to select on. Outcome-based replacement remains forbidden.

## Step 3 — smoke test before any schedule dispatch

Launch **one** row, wait, and assert it wrote a scientific `result.json`. Abort the dispatch if
it did not. Record the smoke row's identity in the schedule artifact.

A single 4-minute check would have caught this bug instead of consuming all 64 rows. Make this
standing practice for every schedule, including the planned confirmatory one.

## Step 4 — re-run the pilot evaluation and adjudicate Gates B and C

Run the 64-row pilot ACT evaluation with the repaired runner and the retained checkpoint.
Frozen pilot schedule SHA-256 `e0515adf10a12cca22412d349d37b56ec5400446894b450b0e84edbe139b564e`;
schedule artifact `diagnostics_output/pact_vs_act/schedule.json` (`3421b8a0418bb2a8…`).

Then adjudicate:

- **Gate B — vision alone insufficient but solvable.** ACT collision-free success in point band
  **[33.3%, 66.7%]**, 95% Wilson interval contained in **[20%, 80%]**; ordinary success above
  its floor. Rationale: at ~0% both arms fail identically and nothing is measurable (the fridge
  failure); near ceiling there is no headroom for proximity to contribute.
- **Gate C — the baseline actually collides.** Non-target contact, and specifically intrusion
  contact, in a meaningful fraction of rows. Without this the collision half of the endpoint
  cannot move.

Report both with Wilson intervals and the one-outcome-perturbation stability check already
implemented in the analysis code.

Expectation from evidence in hand: Gate C likely passes — a causal control that disabled the
expert's avoidance bow produced 192 `hazard_bar` entries across 164 frames. **Gate B is the
genuine unknown and is the measurement that decides whether this experiment can produce a
result at all.**

If Gate B fails high (ACT near ceiling), the scene needs difficulty, not signal. If it fails
low (ACT near zero), the task is not learnable as posed — stop and report
`PACT_ENVIRONMENT_INADEQUATE`.

## Step 5 — only if Gates B and C pass

Full train/validation collection; frozen nearest-surface encoder (report held-out mean/median
Euclidean error, within-2-cm rate, validity precision and recall); ACT and PACT trained on an
identical recipe differing only in proximity tokens; then the three-arm evaluation with Wilson
intervals, Fisher's exact, and instance-clustered bootstrap.

Train ≥2 seeds per arm if budget allows — one seed per arm cannot separate a modality effect
from initialization noise. State the detectable effect size before running; n=50 cannot resolve
a 2 pp difference.

## Allowed final decisions

- `PACT_BENEFIT_ESTABLISHED`
- `PACT_NO_CONFIRMED_BENEFIT`
- `PACT_WORSE_THAN_ACT`
- `PACT_ENVIRONMENT_INADEQUATE` — reserved for Gate B, Gate C, or surface-observability
  failure; never for demonstrator cleanliness or harness faults
- `PACT_EXPERIMENT_INCOMPLETE` — reserved for a genuinely unreconciled schedule, **not** for an
  infrastructure fault a retry can clear

## Constraints

- Do not re-measure Phase 1 environment adequacy; do not retrain the pilot ACT checkpoint.
- Do not move any scientific threshold. The only permitted rule change is Step 1, frozen first.
- Do not replace or re-run any row that produced a scientific outcome.
- Stay in `/root/prox_learning_pact_remediation`.
- Do not commit rollout H5s, videos, or checkpoints (a prior pilot wrote 380 MB — keep it out
  of git). Do not push.

## Artifacts to produce

- `docs/PACT_ENVIRONMENT_ADEQUACY.md` — updated with the re-run and gate adjudication
- `docs/PACT_VS_ACT_FINAL_DECISION.md` — last nonblank line must be the exact decision token
- `diagnostics_output/pact_vs_act/{environment_gate,schedule,analysis,provenance,final_decision}.json`
- tests under `tests/`

## What to preserve

The previous runs stopped rather than improvised, which is why all of this is recoverable
instead of quietly wrong. Keep that discipline. The failure was one mis-scoped rule, not the
rigour around it.
