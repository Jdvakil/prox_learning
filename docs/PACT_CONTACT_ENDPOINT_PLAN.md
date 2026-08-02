# Plan: sharpen the instrument — contact-count endpoint on a preregistered occlusion subset

## Why the current endpoint cannot answer the question

The endpoint is `task_success AND hazard_bar == 0 AND other_environment == 0` — one bit per
rollout. Two problems, both demonstrated by data already in hand.

**1. It discards magnitude.** Contacting episodes ranged from **45 to 30,402 contact entries**. A
policy that grazes the panel once and one that plows through it for thousands of frames score
identically. A ~700× range is collapsed to a single bit, which is why 40 rollouts buy so little.

**2. It conflates two failure modes, and proximity can only affect one.** From the seed-3101
screen taxonomy:

| | collision-free success | hazard-bar contact | target contact, no success |
|---|---:|---:|---:|
| ACT | 19 | **6** | **15** |
| PACT | 29 | **2** | **8** |

PACT's +10-episode advantage decomposes as **+4 from fewer collisions** (which proximity could
explain) and **+7 from fewer manipulation failures** (which it cannot — proximity has no
mechanism to improve grasping), minus 1 post-close failure.

So most of the celebrated +25 pp came from the part of the task proximity does not touch. That is
training variance riding inside the endpoint, and it is exactly consistent with seed 3102 flipping
to −7.5 pp.

**Separating these terms isolates the signal from the noise instead of averaging them together.**

## What this experiment changes

Four changes. Nothing else moves.

1. **Contact magnitude is added as a co-primary** alongside collision-free task success. The
   binary gate is *kept*, not replaced — the collision-override semantics are the intended safety
   definition and grasping success belongs inside it.
2. **Contact conditioned on manipulation success is added as a diagnostic**, so the
   proximity-sensitive term can be read separately from the proximity-irrelevant one.
3. **A preregistered occlusion subset** is defined and reported alongside the full distribution.
4. **Four arms**: ACT, PACT, PACT_ZERO, PACT_PERMUTED — with PACT_ZERO reported as a
   sensor-failure probe and PACT_PERMUTED as the modality instrument.

The scene, policies, encoder, ablation scheme, and training recipe are unchanged.

## Endpoints

Two co-primaries. They answer different questions and both are wanted.

**Co-primary 1 — collision-free task success.** `task_success AND hazard_bar == 0 AND
other_environment == 0`, unchanged. This is retained deliberately, not for continuity: a
collision *overriding* an otherwise successful manipulation is the intended safety semantics, and
grasping success belongs inside the endpoint for that reason. It stays the headline operational
number.

**Co-primary 2 — hazard-bar contact frames per rollout.** The magnitude the binary gate discards.
Frames rather than pair entries, because entry counts scale with how many geometry pairs happen
to touch and are heavy-tailed; frames measure how long the arm was actually in contact and are the
more stable quantity.

Report both as: per-arm value with an instance-clustered bootstrap CI on the paired difference.

**Diagnostic, not a replacement — contact conditioned on manipulation success.** Report contact
frames among rollouts whose manipulation succeeded. This does not replace co-primary 1; it exists
to separate the proximity-sensitive term from the proximity-irrelevant one, because the seed-3101
taxonomy showed 7 of PACT's 10-episode advantage coming from grasping rather than collisions. If
the two views disagree, that disagreement is itself the finding.

**Secondary endpoints**, all predeclared:

- hazard-bar contact frames, unconditional (all rollouts)
- hazard-bar contact **entries** (heavy-tailed count, for comparability with prior runs)
- proportion of rollouts with any hazard-bar contact (the old binary gate alone)
- `other_environment` contact frames and entries
- maximum penetration depth
- manipulation success rate, reported **separately** — the term proximity should *not* affect. It
  doubles as a noise check: a large arm difference here signals training variance rather than
  modality effect
- ordinary task success

## The occlusion subset

Predeclare, before any rollout, a criterion for "situations where vision is structurally
disadvantaged," defined **only** from scene geometry and camera extrinsics — never from any
policy outcome.

Proposed criterion: **the active panel is outside the wrist camera frustum, or occluded within
it, for at least 50% of pre-grasp control steps.** Compute it per instance from the recorded
extrinsics and geometry, exactly as the existing intrusion-sighting test does.

Freeze the resulting instance partition and its SHA-256 before running. Report:

- the full instance distribution (the primary population)
- the occluded subset
- the non-occluded complement

The scientific claim this licenses is stronger than a constructed-necessity result: *"in the X% of
naturally occurring situations where vision cannot see the hazard, proximity reduces contact by
Y."* State X up front — if the subset is very small, say so rather than leaning on it.

### Viability pre-check — do this first, before anything else

**This corridor may not support an occlusion split at all.** It was deliberately built so the
lateral intrusion sits outside the wrist camera's useful view while the target stays visible. If
that holds universally, the "occluded" subset is ~100% of instances, the partition is degenerate,
and this half of the design buys nothing.

So compute the partition **before** committing any GPU time, purely from geometry and extrinsics
on the *existing* recorded episodes. Then:

| Partition result | Action |
|---|---|
| Both sides ≥25% of instances | proceed as planned; the subset is a real population |
| Either side <25% | **drop the subset analysis**, run the contact-endpoint change alone, and record why |
| Degenerate (>95% one side) | the environment cannot express this contrast — say so explicitly and do not manufacture a split by loosening the criterion |

Do not tune the 50% threshold to force a usable partition. If the natural criterion gives a
degenerate split, that is a fact about the environment, and the honest response is to report the
contact-endpoint result on the full distribution only.

## Design

| Setting | Value |
|---|---|
| Instances | **fresh held-out**, none used in the screen, replication, or confirmatory runs |
| Arms | **ACT, PACT, PACT_ZERO, PACT_PERMUTED** — four arms |
| Seeds | **≥3 per arm** |
| Workers | 8, fixed |

### The four arms and what each one answers

| Arm | Question it answers | Decision-bearing? |
|---|---|---|
| ACT | vision-only baseline | reference |
| PACT | full system | yes, with ACT |
| **PACT_ZERO** | what happens when the sensor returns nothing — a **sensor-failure / dropout robustness** probe | **no** — not valid for modality attribution |
| PACT_PERMUTED | does the policy use proximity *information* | **yes** — the modality instrument |

`PACT_ZERO` is restored as a reported arm because ACT / PACT / PACT_ZERO is the comparison the
paper reports, and because "what happens if the skin fails at runtime" is a genuine deployment
question worth answering.

But it **must not** be used to attribute modality benefit. Zeroing a 32-D learned embedding is out
of distribution — 0 of 1,247,040 training tokens are zero, against 95.0% of the old 3-D tokens —
so the ablated arm receives an input the network has never seen. That is what produced the +70 pp
artifact. Label it in every table as an OOD/sensor-failure probe, and keep it out of the modality
contrast.

Expect PACT_ZERO to score far below ACT again. That is the expected behaviour of an OOD input, not
evidence that proximity is load-bearing.

Seeds matter more than instances here. Measured training variance is ~23 pp SD on cross-model
contrasts, so a third seed buys more than another 40 instances. Size the instance count from the
power calculation below, then spend what remains on seeds.

**Power.** Do this before choosing n, and record it. Estimate the SD of per-rollout contact frames
from the existing runs, then state the minimum detectable paired difference at 80% power for the
chosen n. Report that number in the preregistration. A count endpoint should need materially fewer
rollouts than the binary did — quantify by how much rather than assuming it.

## Analysis, frozen before the first rollout

- Instance-clustered bootstrap, ≥20,000 replicates, deterministic seed; **whole instances are the
  clusters** and all arms/seeds for a sampled instance move together
- Contact frames are counts — use the bootstrap on the paired difference of means, and report
  medians alongside, since the distribution is heavy-tailed
- Report the **full contrast set** for every endpoint:

  | Contrast | Interpretation | Decision-bearing |
  |---|---|---|
  | PACT − PACT_PERMUTED | modality **information** | yes |
  | PACT_PERMUTED − ACT | architecture / training / seed | no |
  | PACT − ACT | the two combined | reported, not decision-bearing |
  | PACT − PACT_ZERO | sensor-failure robustness (OOD) | no — never modality evidence |
  | PACT_ZERO − ACT | cost of a failed sensor vs never having one | no |
- Report each seed **unpooled and side by side first**, then pooled. Seed disagreement is a
  finding, not something to average away
- Report the full distribution and the occlusion subset separately

## Predeclared decision rule

| Outcome | Condition |
|---|---|
| `CONTACT_REDUCTION_ESTABLISHED` | PACT − PACT_PERMUTED shows fewer contact frames, CI excludes zero, **and** the sign is consistent across all seeds |
| `CONTACT_REDUCTION_WITH_TASK_BENEFIT` | the above **and** PACT − ACT positive on collision-free task success with consistent sign across seeds |
| `CONTACT_REDUCTION_SUBSET_ONLY` | the above holds on the occlusion subset but not the full distribution |
| `NO_CONTACT_REDUCTION` | CI includes zero on both populations |
| `CONTACT_INCREASE` | PACT shows *more* contact with CI excluding zero — must stay reportable |
| `CONTACT_EXPERIMENT_INCOMPLETE` | schedule did not reconcile |

The modality contrast (PACT − PACT_PERMUTED) is decision-bearing. PACT − ACT is reported for both
co-primaries but is **not** decision-bearing at these seed counts — it carries ~23 pp of training
variance and has already flipped sign once. `CONTACT_REDUCTION_WITH_TASK_BENEFIT` is the one token
that requires it, and it requires consistency across every seed precisely because a single-seed
PACT − ACT number has already misled once.

Contrasts involving PACT_ZERO are never decision-bearing under any token.

Note this experiment cannot award `PACT_BENEFIT_ESTABLISHED`. It answers "does proximity reduce
contact," which is a different and more tractable question than "does PACT beat ACT on task
success."

## Execution requirements

Unchanged from the runs that worked:

- launch-smoke row first; assert a scientific `result.json`, and verify the horizon and token plan
  explicitly — a prior smoke caught a 512-frame/900-step mismatch before any result was written
- `setsid`/`nohup` supervisor with durable PID, log, heartbeat; prove detachment by killing the
  launching shell during the smoke
- fresh subprocess per rollout, unique output directory, fixed worker count
- boundary rule: terminal once a scientific result exists; rows lost to indiscriminate external
  process termination are infrastructure failures and **all** in-flight rows are re-run, never a
  subset
- outcome-blind first-20-minute throughput measurement, recorded
- full reconciliation of every scheduled row before analysis runs

## Constraints

- **Fresh instances and a fresh preregistration.** Re-scoring the existing 240 rollouts under the
  new endpoint is exploratory only — useful for the power estimate and the subset criterion, never
  as the result.
- Define the occlusion subset from geometry and extrinsics only, never from outcomes.
- Do not retrain the encoder, ACT, or existing PACT checkpoints beyond adding seeds.
- Do not change the scene, contact taxonomy, or ablation scheme.
- Do not move the decision rule after seeing any outcome.
- Work in `/root/prox_learning_pact_remediation`; do not touch `/root/prox_learning_hybrid_safety`,
  its submodule checkouts, or `confirmatory41`.
- Check `pgrep -fc eval_act_obstacle_on_policy.py` before heavy GPU use; a paused 410-rollout
  experiment shares the A10.
- Do not commit rollout H5s, videos, or checkpoints. Do not push.

## Artifacts

- `docs/PACT_CONTACT_ENDPOINT_DECISION.md` — last nonblank line is the exact decision token
- `diagnostics_output/pact_contact_endpoint/{occlusion_subset,schedule,dispatch,analysis,provenance,final_decision}.json`
- frozen occlusion partition with SHA-256, computed before any rollout
- power calculation recorded before instance count is chosen
- tests under `tests/`

## Expected cost and most likely outcome — read before starting

4 arms × 3 seeds × n instances. At n=40 that is 480 rollouts, ~20 hours at the measured
0.4 rows/min, plus training the additional seeds (~2 h each; ACT 3101/3102 and PACT 3101/3102
already exist, so budget ~4 h for a third seed of each).

**The most likely outcome, stated up front so it is not read as failure:** the modality contrast
confirms again (it has now replicated at +12.5 and +15.0 pp), and PACT − ACT remains null. With
~23 pp of seed SD and three seeds, PACT − ACT can only resolve differences around ~25 pp, and the
pooled estimate is +8.8 pp. `CONTACT_REDUCTION_WITH_TASK_BENEFIT` is therefore unlikely to fire.

That is not a reason to skip the run — a sharper endpoint may show contact reduction clearly even
where task success cannot — but it is a reason to decide *now* what a third consecutive
task-level null means, rather than after spending the GPU time.

## What a null would mean

If contact does not drop with a count endpoint, conditioned on manipulation success, on the
population where vision is blind — then proximity is not reducing collisions in this task, and
that is a clean result. It would also mean the ~14 pp within-model effect measured earlier acts
somewhere other than contact avoidance, which is worth understanding before any further
environment work.
