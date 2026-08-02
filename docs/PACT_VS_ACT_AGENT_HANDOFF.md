# Handoff: establish whether PACT beats ACT, in an environment adequate to test it

## Task

Build an environment in which whole-body proximity sensing is **necessary**, collect data in it,
train three policies, and run a single pre-registered evaluation comparing them:

1. `ACT` — vision + proprioception only
2. `PACT` — vision + proprioception + proximity
3. `PACT_ZERO` — the trained PACT weights, proximity input zeroed at inference

The objective is to establish whether **PACT > ACT** and **PACT > PACT_ZERO**.

The environment is a first-class deliverable, not a preliminary. Two prior attempts failed on the
environment, not on the policy, and the single largest risk to this task is spending a training
budget inside a scene that cannot express the effect.

### Primary endpoint

**Collision-free task success**, defined exactly as:

- the task succeeds, **and**
- no contact with any object that is not the manipulation target.

Contact with the grasp target is expected and never counts against a policy. Contact with walls,
receptacles, obstacles, or other scene geometry does. Use the contact taxonomy already used
elsewhere in this repo: `grasp_target`, `hazard_bar` / `static_environment`, `other_environment`.

Report ordinary task success separately as a secondary endpoint.

## Starting state

```
root         /root/prox_learning_hybrid_safety   86bfc9f
ACT          submodules/act                      0abeb60
MolmoSpaces  submodules/molmospaces              678f2eb
```

Branch off these. Read `prox_learning/README.md` first and document there — no new README files;
new markdown reports are fine.

### Datagen recipe

```bash
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR='/root/prox_learning_hybrid_safety/assets'
export PYTHONPATH='/root/prox_learning_hybrid_safety/submodules/molmospaces'
```

Use `/root/act_retrain_venv/bin/python`. 8–12 datagen workers.

### Shared-GPU constraint — read before launching anything

A confirmatory experiment (410 rollouts, `ACT_ONLY` vs a safety residual) is **running right now**
on the single A10, pinned at 8 workers, ~14 GB of 22.6 GB, finishing roughly 5 hours from this
handoff. Do not saturate the GPU until it completes. Check with
`pgrep -fc eval_act_obstacle_on_policy.py` — wait for zero. Offline work (asset authoring, dataset
conversion, analysis scripts) is fine meanwhile.

Do not touch `configs/hybrid_obstacle_confirmatory41_v1.json`, anything under
`diagnostics_output/hybrid_obstacle_confirmatory41/`, or the frozen safety-residual stack.

## Prior findings — do not spend budget re-deriving these

These are established and recorded in `EVAL.md`. They are why the two previous attempts failed.

**1. The fridge environment gives proximity no object signal.** Across 196 episodes × 40 sensors ×
~280 steps, the pickup object was visible to *any* proximity sensor in only **99 sensor-frames**,
by 3 sensors. Closest sighting **33.8 cm**; **zero** frames inside 20 cm; zero inside grasp range.
At grasp the object sits between the fingers, outside every sensor's FOV. The paper's PACT
front-end — a frozen encoder regressing the object's 3-D position per sensor — is therefore
**untrainable** in that scene, and would be useless for grasp timing even if trained.

**2. "Maximize proximity activation" maximized wall-sensing, not object localization.** These are
different quantities. Deep encapsulation made the sensors see the fridge, not the mug.

**3. Both policies were broken in that scene.** ACT 0/6 and PACT 0/6, failing identically: the
gripper closes ~step 27 on air and stays closed. A comparison between two policies that both score
zero is uninformative regardless of sample size.

**4. On the collision reframe, PACT was much worse than ACT**, not better:

| Policy | Success | Static-environment contacts |
|---|---:|---:|
| ACT | 0/6 | **0** |
| PACT | 0/6 | **835** |
| PACT-zero | 0/6 | **955** |

**5. That PACT was not the paper's PACT.** It used a hand-crafted 3-D token per sensor
(nearest-pixel x/y + normalized depth), not a learned frozen encoder.

**6. A lesson from the safety chain worth importing:** in the 20-rollout development set, the
baseline produced *zero* hazard contacts, so no safety benefit was measurable even in principle.
An environment where the baseline never fails in the way your endpoint measures cannot show an
improvement. Verify baseline headroom before committing to an evaluation.

### What exists and what does not

- **Gone:** `submodules/act/ckpts/fridge_{act,pact}/` and `assets/datagen/fridge_two_level_v2_exoblind/`.
  Nothing to reuse; both are absent from disk.
- **Present:** the PACT code path. `submodules/act/detr/main.py` has `--use_proximity` and
  `--n_proximity_sensors` (default 0 makes the model bit-identical to vanilla ACT);
  `submodules/act/policy.py` threads `proximity_positions`.
- **Present:** the only datagen environment, `assets/datagen/hybrid_obstacle_independent_v2`, and
  its canonical collection.
- **Critical gap:** the converted ACT dataset `assets/act_style_data/hybrid_obstacle_canonical_v2`
  contains only `action`, `observations/{images,qpos,qvel}` — **no proximity channel**. PACT cannot
  be trained from it as-is. The raw collection does carry the 40×8×8 skin; re-conversion must
  preserve it.
- Sensor count here is **40**, not the paper's 29.

## Phase 1 — environment adequacy gate (do this before any training)

Design or modify a scene, collect a **small** pilot (~20–30 expert episodes is enough), and measure
the three properties below. All three must hold simultaneously. Training before this gate passes is
the failure mode of both prior attempts.

**Gate A — proximity carries object signal (only if you pursue the object-localization route).**
Fraction of pre-grasp steps in which at least one sensor sees the target within 20 cm. Predeclare a
floor; something like ≥30% of pre-grasp steps with ≥1 sensor inside 20 cm, and non-trivial sightings
inside 12 cm. Compare against the recorded failure: 99 sensor-frames total, 0 inside 20 cm.

**Gate B — vision alone is insufficient, but the task is solvable.** A vision-only ACT baseline must
land in a measurable band — neither ~0% (task broken, as before) nor near-ceiling (no headroom).
Target roughly 30–70% collision-free success. This is the necessity condition: if RGB already
solves it, proximity has nothing to add.

**Gate C — collisions actually occur in the baseline.** Non-target geometry must be reachable and
the vision-only baseline must contact it at a measurable rate. If the baseline is already
collision-free, the primary endpoint cannot move.

Two viable routes to Gate A, per the recorded conclusion:

- **Object route** — put sensors where the object will be (fingertip/palm), enlarge the target, or
  place the object against a sensed surface, so proximity localizes the *object*. Enables the
  paper's encoder and a pick-success win.
- **Collision route** — accept that proximity senses *surfaces*, and build a scene with genuine
  collision risk in cluttered approach corridors that vision resolves poorly (occlusion, low
  texture, narrow gaps). Proximity then wins on the collision half of the endpoint.

The collision route is better matched to the primary endpoint and to what these sensors actually
measure, and it does not require object visibility — so **Gate A applies only if you take the
object route**. Choose deliberately and record the choice with its rationale before collecting.

Write `docs/PACT_ENVIRONMENT_ADEQUACY.md` with the measured numbers and a decision token:

- `PACT_ENVIRONMENT_ADEQUATE` — all applicable gates pass; proceed
- `PACT_ENVIRONMENT_INADEQUATE` — stop and report; do not train

## Phase 2 — data and proximity front-end

Collect the full dataset only after the gate passes. Convert to ACT-style **preserving the 40-sensor
proximity stream** and per-sensor world→sensor extrinsics.

Choose the front-end to match the route:

- Object route: reproduce the paper's encoder — weight-shared transformer, ~0.82 M params, conv stem,
  CLS token, sinusoidal positional encodings, 4 layers × 4 heads, regressing the target's 3-D
  position in each sensor's local frame, `p_s = R_s · x_world + t_s`. Keep only samples where the
  object is in FOV and not yet grasped. Report held-out mean Euclidean error (paper: 2.0 cm).
  **Freeze it** before policy training.
- Collision route: the object-position target is wrong here. Use a surface-geometry target (e.g.
  nearest-surface distance/direction per sensor), state it explicitly, and freeze it the same way.

Freezing matters: it makes the proximity branch a fixed validated front-end so the comparison probes
one policy whose representation cannot drift.

## Phase 3 — train the three arms

Identical recipe for all arms, differing only in the proximity tokens: ResNet-18 backbone, 7-layer
encoder/decoder, 8 heads, hidden 512, chunk 100, lr 1e-5, batch 8, 2000 epochs, β=10.

- `ACT` — `--n_proximity_sensors 0` (bit-identical to vanilla)
- `PACT` — `--use_proximity --n_proximity_sensors 40`
- `PACT_ZERO` — no separate training; it is PACT at inference with proximity zeroed

Train ≥2 seeds per arm if budget allows. A single seed per arm cannot separate a modality effect
from initialization noise — the reason the paper's fixed-policy analysis exists. If only one seed
is affordable, say so explicitly in the report and treat the comparison as underpowered.

Record checkpoint SHA-256s and val losses (paper: ACT 0.078, PACT 0.072).

## Phase 4 — evaluation and statistics

Predeclare and freeze the whole schedule before the first rollout: a fixed manifest of held-out task
instances × 3 arms × repeats, with balanced arm ordering. **n=50 instances is the paper's protocol
and is the minimum**; the effect being chased there was 2 pp (74% vs 72%), which n=50 cannot resolve
— so power the run for the effect you expect on the collision endpoint, and state the detectable
effect size up front.

Requirements:

- fresh subprocess per rollout so the task sampler re-draws independently
- unique output directory per rollout; complete provenance retained
- no row replaced or re-run based on its outcome
- fixed worker count for the whole run

Analysis, frozen before results exist:

- pooled collision-free success per arm with **Wilson 95% intervals**
- **Fisher's exact** for PACT vs ACT and PACT vs PACT_ZERO
- if repeats-per-instance > 1, cluster on the **instance**, not the repeat, and bootstrap whole
  instances (≥10,000 deterministic replicates)
- report task success, contact counts by class, and failure taxonomy separately
- report per-phase sensor activity if the fixed-policy analyses from the paper are in scope

## Allowed final decisions

- `PACT_BENEFIT_ESTABLISHED` — PACT beats both ACT and PACT_ZERO, CI lower bound above zero
- `PACT_NO_CONFIRMED_BENEFIT` — valid experiment, intervals include zero
- `PACT_WORSE_THAN_ACT` — PACT significantly worse (the prior result; do not bury it)
- `PACT_ENVIRONMENT_INADEQUATE` — stopped at the Phase 1 gate
- `PACT_EXPERIMENT_INCOMPLETE` — schedule did not reconcile

A null is a legitimate and publishable outcome. Design the environment so proximity *can* matter —
that is sound experimental design and is what the recorded conclusion recommends — but do not tune
the endpoint, the scene, or the analysis after seeing results. Freeze the analysis before the
rollouts and report what it returns.

## Hard constraints

- Do not modify, rerun, or read as your own any part of the hybrid-obstacle safety-residual chain,
  and do not touch `confirmatory41`.
- Do not change the analysis, endpoint, or scene after seeing outcomes.
- Do not replace or re-run evaluation rows based on their results.
- Do not average predictions or actions across arms.
- Do not saturate the GPU while the confirmatory run is active.
- Do not push commits.

## Artifacts

- `docs/PACT_ENVIRONMENT_ADEQUACY.md` — Phase 1 gate, measured numbers, token
- `docs/PACT_VS_ACT_FINAL_DECISION.md` — final report; **last nonblank line must be the exact token**
- `diagnostics_output/pact_vs_act/{provenance,schedule,analysis,final_decision}.json`
- frozen encoder + three checkpoints with SHA-256s recorded (do not commit weights)
- tests under `tests/`
- a README link to the final document

Commit reports, schedules, manifests, and analysis. Do not commit rollout H5s, videos, checkpoints,
or large logs.
