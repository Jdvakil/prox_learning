# Handoff 2: recover the PACT experiment from `PACT_ENVIRONMENT_INADEQUATE`

## Read this first

The previous run stopped correctly. Do not treat this plan as permission to reopen that
decision, and do not re-adjudicate the existing 24 pilot rows against a changed threshold.
That would be exactly the post-hoc goalpost move the preregistration existed to prevent.

Everything below is about **re-specifying the gate and re-running it on new data**.

## Where things stand

Decision: `PACT_ENVIRONMENT_INADEQUATE`, `valid_early_stop: true`. No policies trained, no
encoder, no dataset, `policy_checkpoint_sha256s: []`, no three-arm comparison.

### What passed, and is the real asset

The failure mode that killed the two prior attempts is **solved**. Surface observability:

| | Fridge (prior failure) | `pact_collision_corridor_v1` |
|---|---:|---:|
| Pre-grasp frames with signal inside 20 cm | **0** | **1284/2495 (51.5%)** |
| Inside 12 cm | 0 | 492/2495 (19.7%) |
| Episodes with a sighting | ~0 | 20/24 |

The fridge produced 99 sensor-frames total across 196 episodes, none inside 20 cm. This
corridor gives proximity a dense, real signal. **Keep this environment.** Do not redesign the
scene.

Two design properties are worth preserving explicitly because they are easy to lose in a
rewrite:

- The panel enters from either side, balanced per manifest role, and the target is sampled
  independently — so target pixels do not disclose the avoidance direction and a single
  memorized detour cannot work.
- An intrusion sighting requires the ray's reconstructed world point to fall inside the panel
  AABB with 1 cm tolerance, so fumehood walls and self-returns cannot be miscounted as signal.

### What failed

**Expert collision-free success 19/24 against a frozen minimum of 20/24.** Missed by one row.
Ordinary success passed at 20/24. The offending row completed the task and logged 58
`hazard_bar` entries across 58 frames, with 0 `other_environment`.

Ledger: 20 success, 1 sampling failure, **3 infrastructure failures**.

### Why the stop was on the wrong quantity

The gate conflates three distinct things and fails the whole experiment on the weakest:

1. **Environment adequacy** — can this scene express a proximity effect at all?
2. **Demonstrator quality** — are the demos clean enough to train BC on?
3. **Infrastructure health** — does the harness produce terminal outcomes reliably?

Only (2) failed, and only by one episode. (2) genuinely matters — training BC on colliding
demos teaches both arms to collide and makes the endpoint meaningless — but the correct
response to a marginally dirty demonstrator is to **fix or filter the demonstrator**, not to
discard the environment.

Three reasons the stop looks over-tight:

- At n=24 one failure is 4%; a ≥20/24 threshold has zero tolerance. That is a brittle
  instrument, not a property of the scene.
- 3 of the 4 non-successes were IK/construction failures. Those measure the harness, not the
  environment, and should never have been able to fail a science gate.
- The environment's own development screen got **8/8 ordinary and collision-free with zero
  non-target contact**, and 54.3% / 23.3% signal. Expert cleanliness is therefore somewhere
  around 79–100% depending on the draw — marginal and fixable, not broken.

**And the decisive gates were never run.** Gates B and C determine whether the experiment can
produce a result at all. There is strong evidence C will pass: the causal control that
disabled the avoidance bow on development row 5, same seed, produced **192 `hazard_bar`
entries across 164 frames**. A policy that does not avoid collides heavily. That is exactly
the headroom needed, and it is what the safety chain's development set lacked.

## Phase 0 — isolation and hygiene (do before anything else)

**Work in your own clone or `git worktree`.** The previous run switched branches on the shared
root repo *and* `submodules/molmospaces`, which changed
`molmo_spaces/configs/camera_configs.py` and broke the pinned-stack hash of a confirmatory
experiment running concurrently on the same tree. That run refused 320 of 410 rollouts. Do not
share a working tree with another experiment again.

Specifically, do not touch: `configs/hybrid_obstacle_confirmatory41_v1.json`,
`diagnostics_output/hybrid_obstacle_confirmatory41/`, the frozen safety-residual stack, or the
checkout state of `submodules/molmospaces` and `submodules/act` outside your own worktree.

**Artifact hygiene.** The pilot wrote **380 MB across 104 files** into `assets/datagen/`,
including `.mp4` renders and `trajectory.h5`. Confirm these are gitignored or otherwise kept
out of commits. Commit reports, manifests, schedules, and analysis only.

Check GPU contention with `pgrep -fc eval_act_obstacle_on_policy.py` before launching.

## Phase 1 — fix the expert

Root cause to address: one pilot row completed the task while clipping the panel for 58
frames. The nominal surface-clearance margin is 0.08 m.

Do at least one of, and state which:

- raise the clearance margin above 0.08 m;
- add a contact check to the expert with **reject-and-resample** on any demonstration that
  registers `hazard_bar` or `other_environment` contact — filtering demonstrations is standard
  practice and is not p-hacking, provided the filter is defined before collection and applied
  blind to downstream outcomes;
- widen the aperture, provided the opposite-side bow stays physically open and the
  side-balance property is preserved.

Acceptance: on a fresh seed set, the expert should produce clean demonstrations at a rate high
enough that filtering still leaves the demo count you need for training. Report the rate with
a confidence interval, not a bare fraction.

## Phase 2 — fix the infrastructure failure rate

4 of 24 rows (~17%) produced no scientific outcome — 3 infrastructure, 1 sampling. Diagnose
the IK/construction failures. At that rate the planned 80-instance × 3-arm schedule (240
rollouts) would lose roughly 40 rows.

Target under 5% before scaling. Report infrastructure failures **separately** from scientific
outcomes in every ledger from now on; they must never again be able to fail a science gate.

## Phase 3 — re-specify the gate, then freeze it

Rewrite the preregistration so the three concerns are independent:

**Environment adequacy** (the only thing that can award `PACT_ENVIRONMENT_INADEQUATE`):
- surface observability — keep the existing thresholds; they passed decisively
- **Gate B** — vision-only ACT collision-free success in a band that is neither floor nor
  ceiling; keep the 8–16/24 intent but express it as a proportion with an interval
- **Gate C** — the baseline actually collides; keep "intrusion contact in ≥6/24" in spirit,
  expressed as a proportion

**Demonstrator quality** — enforced by the Phase 1 filter, with the gate expressed as a floor
on the *number of usable clean demonstrations*, not on the fraction of attempts that were
clean. A demonstrator that is 85% clean and produces 200 usable demos is fine.

**Infrastructure health** — reported, monitored, never a science gate.

Express thresholds as proportions with intervals rather than brittle `x/24` counts, and size
the pilot so a single row cannot flip the decision.

Freeze this before collecting. No threshold may move after outcomes are visible.

## Phase 4 — fresh pilot

Run the re-specified gate on a **new, independently seeded** pilot. Not a re-scoring of the
existing 24 rows.

This time **run Gates B and C** — they are the decisive ones and have never been measured.
That means training the pilot wrist-RGB+proprio ACT baseline on the filtered clean expert rows
and evaluating it, as the original protocol described.

Expected outcome based on the evidence in hand: surface observability passes again; Gate C
passes (the disabled-bow control produced 192 contacts); Gate B is the genuine unknown — the
baseline must be solvable but not saturated.

If Gate B fails high (ACT already near-ceiling), the scene needs more difficulty, not more
signal. If it fails low (ACT near zero), that is the fridge failure again and the task is not
learnable as posed — stop and report.

## Phase 5 — only if the gate passes

Proceed with the original protocol: full collection, frozen nearest-surface encoder (report
held-out mean/median Euclidean error, within-2-cm rate, validity precision and recall), then
ACT / PACT training on the identical recipe differing only in proximity tokens, then the
three-arm evaluation with Wilson intervals, Fisher's exact, and instance-clustered bootstrap.

Train ≥2 seeds per arm if budget allows; one seed per arm cannot separate a modality effect
from initialization noise. Power the run for the effect you expect on the collision endpoint —
n=50 cannot resolve the paper's 2 pp success gap, and you should state the detectable effect
size before running.

## Allowed final decisions

- `PACT_BENEFIT_ESTABLISHED`
- `PACT_NO_CONFIRMED_BENEFIT`
- `PACT_WORSE_THAN_ACT` — the prior recorded result on collisions was ACT 0 static contacts vs
  PACT 835; this outcome must stay reportable
- `PACT_ENVIRONMENT_INADEQUATE` — reserved for a failure of *environment* adequacy, i.e. Gate
  B or C or surface observability, not demonstrator cleanliness or harness flakiness
- `PACT_EXPERIMENT_INCOMPLETE`

## Constraints

- Do not reopen or re-score the previous pilot; new data only.
- Do not move any threshold after seeing outcomes.
- Do not redesign the corridor scene; fix the expert and the gate.
- Do not replace or re-run evaluation rows based on their results.
- Do not share a working tree or submodule checkout with another experiment.
- Do not commit rollout H5s, videos, or checkpoints.
- Do not push.

## Artifacts

- `docs/PACT_ENVIRONMENT_ADEQUACY.md` — updated with the re-specified gate and new pilot
- `docs/PACT_VS_ACT_FINAL_DECISION.md` — final report; last nonblank line must be the token
- `diagnostics_output/pact_vs_act/{environment_gate,schedule,analysis,provenance,final_decision}.json`
- tests under `tests/`
- README link to the final document

## What to carry forward from the last run

Keep the preregistration discipline. It froze thresholds, honored the stop rule, and refused
to run the expensive stages on a failed gate — that is exactly right and is why this is
recoverable. The fix is a better-specified gate on new data, not a looser reading of the old
one.
