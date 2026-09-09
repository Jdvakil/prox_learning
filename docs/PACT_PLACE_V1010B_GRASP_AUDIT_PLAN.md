# V10.10b grasp remediation: assignment for the auditing agent

## 1. Assignment and deliverables

Audit the completed V10.10 wrist-only ACT/PACT experiment and write an executable,
bounded plan for **V10.10b** that addresses unreliable cup-wall grasp acquisition.
Explain which failure mechanism the evidence supports, what remains uncertain,
and why the proposed intervention should address it.

This assignment is an **audit followed by a fix plan**. Use retained artifacts,
code inspection, numerical checks and offline kinematics. Put any necessary new
simulator rollouts, instrumentation, collection or training into the proposed
execution stages; do not launch them as part of writing the plan. An unresolved
causal question should produce a small conditional experiment, not a stalled
audit or an unsupported claim of certainty.

Produce:

1. `diagnostics_output/pact_place_v1010b_grasp_audit/AUDIT.md`: findings, evidence,
   counterexamples, limitations and ranked hypotheses.
2. `diagnostics_output/pact_place_v1010b_grasp_audit/evidence.json`: source hashes,
   case identities, measurement definitions, reconciled counts and supporting
   numerical comparisons. Store any new audit scripts beside it.
3. `docs/PACT_PLACE_V1010B_GRASP_FIX_PLAN.md`: one recommended first intervention,
   at most one conditional fallback, exact stages, implementation locations,
   tests, experiment matrices, compute estimates, decision gates and rollback.

Propose future implementation under
`diagnostics_output/pact_place_v1010b_grasp_v1/` with new adapters and versioned
contracts. Preserve the completed experiment and its original reports.

## 2. Start from the actual completed baseline

Work in `/root/prox_learning_pact_remediation`. Define these path aliases in the
audit for readability:

- `W = diagnostics_output/pact_place_v1010_wrist288_s3_v1`
- `G = W/geometric_failure_20260908`
- `S = W/amendments/seed3103_full50_20260908`

Read these sources first:

- [Completed geometric diagnosis](../diagnostics_output/pact_place_v1010_wrist288_s3_v1/geometric_failure_20260908/GEOMETRIC_DIAGNOSIS_300.md),
  `G/analysis_300.json`, `G/final_verification.json` and both `_300` figures.
- [Full three-seed results](../diagnostics_output/pact_place_v1010_wrist288_s3_v1/amendments/seed3103_full50_20260908/THREE_SEED_FULL150.md),
  `S/three_seed_full150.json`, `S/parent_closure.json`, and the separate
  seed-3104/3105 extension reports and closures under `W/amendments/`.
- [Original experiment plan](PACT_PLACE_V1010_WRIST288_THREE_SEED_PLAN.md) and
  [actual dataset amendment](PACT_PLACE_V1010_WRIST280_AMENDMENT.md).
- `W/amendments/wrist280/effective_config.json`,
  `W/amendments/wrist280/selected_ledger.jsonl`, `W/conversion_ledger.jsonl`,
  `W/dataset_stats.pkl`, and the six checkpoint directories under `W/checkpoints/`.
- Full-stage schedules, ledgers, completion records and initial-state pair audits
  under `W/evaluation/`. Resolve actual worker directories from records; expanded
  stages retain the original `final_<seed>_h100` worker identities.

The effective dataset is **280 demonstrations: 240 training / 40 validation**.
Every model reached **60,000 optimizer updates**, with chunk 100, averaging
history 100, wrist RGB, shared train-only normalization and the frozen proximity
encoder. There are **50 paired instances per seed**, 150 instances and 300 final
rollouts in total. Do not substitute the earlier 120-demonstration experiment,
the original reduced 17/17/16 evaluation, V5, or V10.11c as this baseline.

| Seed | Model | Task success | Collision-free task success | Touch-without-hold pickup failures | Frame avoidance: hazard or clutter |
|---|---|---:|---:|---:|---:|
| 3103 | ACT | 19/50 | 16/50 | 12/50 | 90.63% |
| 3103 | PACT | 20/50 | 16/50 | 16/50 | 96.02% |
| 3104 | ACT | 22/50 | 15/50 | 12/50 | 87.94% |
| 3104 | PACT | 19/50 | 10/50 | 18/50 | 96.48% |
| 3105 | ACT | 23/50 | 15/50 | 11/50 | 96.43% |
| 3105 | PACT | 35/50 | 25/50 | 8/50 | 98.34% |

Pooled task success is ACT **64/150**, PACT **74/150**, a **+6.67 percentage-point**
PACT advantage. The original target, PACT at least 76/150 and at least 15 more
successes than ACT, was missed. Preserve that target as a performance objective,
not a promised result or evidence that historical qualification passed.

## 3. Evidence to reproduce and challenge

Treat the current diagnosis as a supported working explanation, not a conclusion
the audit must defend. Reconcile its measurements with raw records and actively
look for cases that contradict it.

- PACT-3105 gains 16 successes over PACT-3104; 15 more cases reach a 1 cm lift.
  Pickup acquisition accounts for most of the observed difference.
- Of the 18 PACT-3104 touch-without-hold pickup failures, 16 never put the TCP
  below the rim reference in the defined early grasp window.
- In that failure group, median actual hand Z rises about **14.1 mm** over the
  first two control steps after closure, while commanded TCP Z decreases about
  **2.7 mm**. Distinguish requested depth, achieved depth and later commanded lift.
- PACT-3104 commands the hand within 2 cm horizontally of the cup origin at
  closure in **20/50** cases, versus **5/50** for PACT-3105. This is a descriptive
  tendency, not a universal valid-grasp boundary.
- Sustained bilateral contact on the same cup-wall primitive occurs in **18/19**
  successful PACT-3104 cases and **32/35** successful PACT-3105 cases. Check whether
  a usable wall pinch, rather than enclosing the whole cup, explains success.
- In seed 3104, **10 of 12 ACT-only successes** are PACT touch-without-hold
  failures. PACT's achieved minimum grasp height is a median **18.5 mm higher**
  than paired ACT's across the ACT-only group.
- The stricter blocked-descent signature appears in PACT **11/50, 14/50 and
  8/50** for seeds 3103, 3104 and 3105. Seed 3103 also has six failures after a
  lift; it does not have exactly the same mixture of errors as seed 3104.

Inspect at least these examples and their paired ACT controls, then support any
general statement with all eligible cases:

- Seed 3104: `98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f`.
- Seed 3103: `621d8bd12acc6e7256e2a671b85f49c0168e469e53ac02193dc8e0ba09be2d11`.
- The seed-3105 illustration and its selection rule are recorded in
  `G/analysis_300.json`; it is a nearby same-cell example, not an identical scene.

Reconstruct the failed pickup sequence: approach → first contact → closure →
achieved finger engagement → lift → transport. Record which event diverges first
between a successful and failed paired trajectory. Do not infer that the policy
commands an early lift merely because the measured hand moves upward.

## 4. Resolve measurement and geometry limitations first

Inspect the implementation in:

- `G/extract.py`, `G/augment.py`, `G/contact_modes.py`,
  `G/command_geometry.py`, `G/summarize.py`;
- `scripts/pact_wrist288_eval_worker.py`, `scripts/pact_wrist288_metrics.py`,
  `scripts/pact_wrist288_analysis.py`;
- `submodules/molmospaces/molmo_spaces/env/sensors.py` and
  `submodules/molmospaces/molmo_spaces/tasks/pact_place_contact_audit.py`;
- `submodules/molmospaces/molmo_spaces/robots/robot_views/franka_droid_view.py`.

Required checks:

1. Verify robot-base-to-world transforms, quaternion convention, the robot's
   35 cm mount offset, and observation/action timing. Command k follows
   observation k. The command-kinematics check had maximum error **0.978 mm near
   closure**; do not interpret smaller discrepancies as established effects.
2. Use the **five primitive Cup_10 collision boxes** matched to saved model arrays,
   including their local offsets. The initial upper edge is 71.98 mm above the
   target origin. Do not use an alternative mesh collider or confuse object
   origin, rim, TCP and pad positions.
3. The current rim reference follows cup translation with its initial orientation.
   Dynamic cup rotation, complete passive-finger states and contact forces are
   missing. Assess sensitivity to this approximation and mark measurements that
   cannot support a precise clearance or force claim.
4. Joint values are not jaw widths in metres. Audit closing-axis orientation,
   cup-wall normal, offset, projected opening clearance, approach angle and
   achieved depth together. A full-rim span above the nominal 87 mm gripper gap
   is compatible with a successful wall pinch; it is not proof of impossibility.
5. Separate robot–cup, robot–clutter and cup–clutter contacts. The official
   `clutter` class can include cup–clutter contacts. Distinguish direct obstruction
   from contacts caused later by pushing the cup.
6. `held` is a contact-only heuristic, not proof of stable grasp. Confirm actual
   lift and sustained engagement independently. Do not confuse the exclusive
   pickup-failure category with all touched-but-never-held episodes: for example,
   PACT-3103 has 43 touched minus 25 held = 18 such episodes, but only 16 in the
   exclusive pickup-failure category because later milestones take precedence.
7. Wrist image-point visibility shows that some target pixels were visible; it
   does not prove adequate pose information. No RGB videos were retained for the
   final H5-only rollouts. Do not claim a visual review that the data cannot support.

Preserve the existing diagnosis. Put corrections and counterexamples in the new
audit with explicit definitions, denominators and source references.
If reusing a diagnostic script would regenerate files inside `G`, copy/adapt it
to write under the new audit directory and pin the numbered source snapshot.

## 5. Audit competing causes

For each hypothesis, supply supporting cases, counterexamples, an alternative
explanation, and the smallest observation or intervention that distinguishes it.

| Hypothesis | Audit work | Consequence for the proposed fix |
|---|---|---|
| Target-relative grasp pose is unsuitable | Compare offset, closing-axis alignment, achieved depth and contact onset against successful demos and paired ACT; stratify by actual cup pose and layout | Consider a consistent feasible wall-pinch strategy or better grasp-pose supervision |
| Contact prevents commanded descent | Compare command FK to measured TCP before/after contact and closure; distinguish cup, bottle and other robot contacts | Test achieved-engagement feedback or a bounded corrective approach; do not prescribe more downward force blindly |
| Temporal averaging mixes incompatible approaches | Inspect query indexing, contributor ages, weights, padding/validity and gripper normalization; determine whether individual chunks disagree on approach or phase | Consider a minimal inference correction only if the aggregation mechanism is implicated |
| Demonstrations contain incompatible grasp modes or weak recovery coverage | Audit all 240 training and 40 validation identities, grasp modes, phase-local labels and failed-approach coverage; identify any split or phase weighting issue | Propose targeted supervision, sampling or a small justified data amendment rather than an automatic recollection |
| Wrist perception or PACT conditioning changes grasp localization | Inspect available observations, encoder inputs and model path; compare ACT/PACT near contact | Require evidence before changing features or sensors; zeroing proximity is a distribution-shift diagnostic, not proof of sensor causality |
| Runtime binding, decoder or controller discrepancy | Trace the actual adapter chain, checkpoint/stats hashes, normalized actions, 127.5 threshold and applied commands | Fix a demonstrated implementation defect before retraining |

Trace inference through `Wrist288InferencePolicy` and its actual parents in
`submodules/act/eval_pact_place_v109_row.py`, `eval_pact_place_row.py` and
`eval_pact_collision_row.py`. Audit training and conversion through
`scripts/pact_wrist280_amendment.py`, `scripts/pact_wrist288_data.py`,
`scripts/pact_wrist288_train.py` and the actual ACT loader/model code.
Confirm the corrected observation[t] → commanded_action[t+1] labels; do not
reintroduce the old shift based on filenames or assumptions.

Read `W/development_gate.json`: on the existing 24-pair development screen,
history 100 achieved ACT 8/24 and PACT 10/24; history 10 achieved **0/24 for both**.
Shortening history is therefore not an untested quick fix. The retained final
action trace does not contain every pre-aggregation chunk contribution; do not
claim to have demonstrated averaging between incompatible grasps if those inputs
cannot be recovered. Specify the missing recording in the proposed causal stage.

The all-case average loss alone cannot establish grasp quality. Compare
phase-local errors and feasible grasp modes, including successes and failures
from every seed. Do not select a different checkpoint using these test outcomes.

## 6. Specify a small causal experiment if retained evidence is insufficient

Training checkpoint seed and final scene block are confounded in the completed
experiment. Same-scene ACT/PACT comparisons demonstrate solvability, but do not
separate PACT-3104 versus PACT-3105 checkpoint effects from scene effects.

A useful bounded proposal is:

- Select **24 existing scenes, one per nominal cell**, by a frozen hash rule
  independent of outcomes, from the union of the completed manifests.
- Evaluate each of the three frozen PACT checkpoints on each selected scene:
  **72 checkpoint–scene combinations**. There are 24 corresponding original
  checkpoint/scene outcomes; at most those 24 can be reused if all bindings and
  measurement requirements match. This leaves **48 cross-checkpoint rollouts**;
  replay checks or extra instrumentation require separately budgeted runs.
- Keep `training_seed`, `scene/task_seed` and `scene_id` separate throughout the
  new manifest, worker and reports. Loading another checkpoint must not change
  the physical instance. Write a new cross-checkpoint initial-state audit that
  permits the intentional model difference while retaining exact physical and
  configuration matching and the original RGB tolerance. Do not weaken the
  historical pairing contract or overwrite worker directories.
- Treat these exposed scenes as diagnostic/regression data. They cannot become
  an untouched V10.10b final test set. Retain failed replays and report any
  nondeterminism or instrumentation-induced behavior change.

If this matrix is unnecessary or too expensive, explain which smaller comparison
answers the same question. Report complete rollout arithmetic, reuse eligibility,
wall-clock estimate and a stopping rule; do not turn this into a large sweep.

For a few explicitly selected instrumented cases, propose synchronized actual
cup and pad poses, full finger joint states, contact points/normals/forces, arm
targets, applied controls and achieved joint states. Store each predicted action
chunk once with query times and aggregation weights so contributors can be
reconstructed offline. Include short wrist clips if needed for a specific
perception question. Bound storage and verify that instrumentation does not alter
the policy inputs or invalidate comparison to the baseline.

## 7. Requirements for the V10.10b fix plan

Recommend **one first intervention** tied to the best-supported mechanism. Explain
why it is preferable to the other candidates and when the one fallback would be
used. Do not bundle geometry changes, decoder changes, new data and retraining
into an uninterpretable first experiment.

The plan must specify:

- **Exact implementation:** files/classes to add or change, interface and action
  semantics, proposed parameter values and how they were chosen, plus a baseline
  path that can still be run unchanged.
- **Available information:** list every input used by a proposed correction.
  Ground-truth cup pose, simulator pad contacts, task `held`/success flags and
  future trajectory information may be diagnostic evidence or offline labels;
  they must not silently become ACT/PACT inference inputs. A runtime gate needs
  an observable, justified signal and a bounded timeout/recovery behavior.
- **Fairness:** shared preprocessing, decoder or controller changes must be
  applied equally to ACT and PACT in the comparison. Isolate a PACT-specific
  change with its own ablation. Do not compare PACT plus an oracle/helper against
  unchanged ACT and attribute the difference to proximity sensing.
- **Environment preservation:** default to a learning/inference remediation on
  the same V10.10 task. Keep target/clutter assets, placement distribution,
  collision geometry, friction, actuator limits, success predicate, horizon,
  camera and sensor contract unchanged. Any justified environment or hardware
  change must be a separately identified variant, with a new baseline comparison.
- **Data and training:** first test whether the existing checkpoints can support
  the proposed change. Retrain only when the hypothesis requires it. If training
  is needed, retain all three seeds, a frozen split, train-only statistics and an
  explicit update budget. Declare any targeted-data amendment; prevent diagnostic
  or final-test scenes from entering training or validation.
- **Evaluation stages:** a small smoke, a discriminating mechanism test, fresh
  development selection, then one frozen final comparison. State checkpoint ×
  scene × arm × intervention counts explicitly. Test all seeds, preserve 3105's
  strengths, and check the later lift/placement failures in 3103.
- **Acceptance gates:** choose numerical thresholds before executing each stage,
  with sample sizes that can resolve them. Require reduced pickup failure and
  improved end-to-end success, while checking collision-free task success,
  contact duration and per-seed regressions. Do not declare a fix from a few
  repaired showcase cases, a pooled score hiding a bad seed, or increased idle time.
- **Matched controls:** use the same fresh scenes to measure baseline versus
  candidate improvements. Budget baseline rollouts explicitly. Historical rates
  on different scenes are context, not a paired improvement measurement. Retain
  50 pairs per seed as the planned final ACT/PACT evaluation unless the plan
  explicitly justifies another design and its limits.
- **Tests:** target frame transforms and alignment, input/decoder invariance,
  aggregation validity, contact-overlap accounting, grasp-signal false positives,
  timeout/recovery behavior, initial-state pairing and false completion. Test the
  failure mechanism; do not add tests that merely duplicate implementation text.
- **Execution and rollback:** list commands, dependencies, artifacts, resume
  conditions, stop conditions and how to restore the unchanged baseline.

Keep wrist RGB, chunk 100 and the existing sensor contract as the baseline.
If the selected fix requires departing from an earlier experiment restriction
such as no gripper retuning or no targeted recovery data, identify that departure
explicitly in the new plan and justify its necessity. Historical restrictions
must neither be silently ignored nor mistaken for evidence that a proposed fix
has already been approved or executed.

## 8. Metrics, costs and completion criteria

Use distinct names for:

- **Task success:** the unchanged final task predicate.
- **Collision-free task success:** the unchanged success-plus-contact predicate.
- **Exclusive touch-without-hold pickup failures:** the existing stage hierarchy;
  also report raw touched-but-never-held separately when discussing grasp sensing.
- **Frame collision avoidance for hazard/clutter:**
  `100 * (1 - count(hazard_contact OR clutter_contact) / recorded_physics_frames)`.
  Deduplicate overlaps. The current denominator is 1,485,050 physics samples per
  50-rollout arm/seed group, including recorded boundary samples; compute actual
  denominators from telemetry in new runs. This is different from the percentage
  of whole episodes without a collision. Report other contact classes separately.
- **Mechanism outcomes:** requested/achieved grasp depth, contact chronology,
  continuous bilateral/same-wall engagement, cup lift, empty-hand transport and
  failures after lift. State every window, threshold and missing-data rule.

Preserve the user's preference to omit confidence intervals and McNemar tests;
do not present small descriptive gates as statistical proof. Report raw counts
and denominators, per-seed differences and paired outcomes.

Budget the audit and proposed experiments before expensive stages. Start rollout
estimates from measured recent durations and **12 workers**, then recheck actual
capacity. Preserve the existing resource guards, observed process-exit receipts,
initial-state audits and scoped cleanup. Keep the separate monitoring task
disabled; proposed execution should use local supervision and low-frequency
parent updates. Set a new bounded schedule for V10.10b rather than copying the
completed experiment's expired deadline. Avoid large recollection, all-model
retraining or broad parameter searches as the default first action.

The audit is complete when it provides a reproducible evidence table, explicitly
distinguishes observed failure from unproven training cause, and delivers a fix
plan that an implementation agent can execute without inventing its intervention,
metrics, data split, experiment counts, budget or decision gates. End the audit
with the recommended first action and the evidence that would change that choice.
