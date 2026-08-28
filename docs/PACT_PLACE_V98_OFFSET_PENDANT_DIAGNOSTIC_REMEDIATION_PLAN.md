# PACT Place V9.8 offset pendant — diagnostic remediation plan

Status: authorized for retained-artifact diagnosis, test repair, and documentation
correction only.

Written 2026-08-25 after the two offset candidates completed their paired
eight-row screens. This plan does **not** authorize new expert episodes, another
geometry sweep, route changes, collection, training, or policy evaluation.

## Objective

Preserve the valid V9.8 Step 3 stop while correcting the unsupported causal
conclusion around it.

The established result is narrow and final for the registered candidates:

- the V9.5 regression guard remained 6/8, with all eight rows matching;
- the wide offset pendant (`center_y=0.100`, `half_y=0.056`) preserved zero of
  the six baseline-clean rows;
- the conservative offset pendant (`center_y=0.100`, `half_y=0.045`) also
  preserved zero of those six rows;
- bow direction was correct and no admitted waypoint clipped;
- therefore the fixed selection rule selected neither candidate and correctly
  stopped the 24-row gate, S2b, and S4.

What is **not** established is that the derived lateral window is physically
valid, that a wrist-height pendant is unavoidable in the 0.85 m aperture, or
that the wrist simply failed to follow an otherwise complete TCP bow. The
post-run lag instrument measures a different point from the one used to derive
the window, and raw contact onset occurs outside the phase protected by the
isolated ceiling-fixture waypoint on multiple rows.

The remediation must produce an auditable contact-onset and swept-geometry
account from retained qpos, fix the currently failing verification test, and
amend the status documents to state only what the artifacts support.

## Frozen boundaries

The coding agent must not:

- run `run_pact_place_expert_screen.py`,
  `run_pact_place_v98_v95_pendant_paired.py`, or any other episode runner;
- call `env.step`, execute the expert, or advance physics in a diagnostic;
- change the pendant geometry, aperture, panel, routing, bow logic, envelope
  constants, contract bounds, selection rule, sensing floor, or acceptance bar;
- overwrite any existing V9.8 result, trajectory, predictor, guard, or lag
  artifact;
- run the 24-row gate, S2b, S4, collection, training, or evaluation;
- describe an algebra-only waypoint prediction as swept-volume clearance.

Allowed work is limited to:

- loading retained qpos and calling `mj_forward`;
- inspecting live MuJoCo bodies, geoms, contacts, poses, and distances at those
  retained states;
- writing new diagnostic artifacts under a new output directory;
- repairing tests and diagnostic code;
- correcting `docs/PACT_PLACE_V98_PENDANT_PLAN.md`, `EVAL.md`, and `README.md`.

## Inputs that must remain immutable

- `diagnostics_output/pact_place_v95_raw_smoke/summary.json`
- `diagnostics_output/pact_place_v95_smoke_repro_guard_v5/guard.json`
- `diagnostics_output/pact_place_v98_bow_clearance_predict.json`
- `diagnostics_output/pact_place_v98_paired_offset_wide/paired.json`
- `diagnostics_output/pact_place_v98_paired_offset_wide/expert_screen_rows/`
- `diagnostics_output/pact_place_v98_paired_offset_wide/wrist_lag.json`
- `diagnostics_output/pact_place_v98_paired_offset_cons/paired.json`
- `diagnostics_output/pact_place_v98_paired_offset_cons/expert_screen_rows/`
- the centred source runs:
  `pact_place_v98_paired_halfy016/`, `pact_place_v98_paired_halfy014/`, and
  `pact_place_v98_paired_halfy012/`

The new diagnostic must record the path and SHA-256 of every JSON input it
uses. Existing files are evidence, not scratch space.

## Known discrepancies to resolve

### 1. Contact onset is not terminal phase

The current paired summary reports `terminal_policy_phase`, which is not the
contact onset. In the wide run, the retained trajectory joined to
`contact_audit.first_contact_step.mounted_fixture` shows:

| rows | first-contact phase | interpretation to test |
|---|---|---|
| left baseline-clean roles 600/602/604 | `pregrasp` | contact after the ceiling bow, on the final unbowed grasp transition |
| right baseline-clean roles 601/603/605 | `inbound_cross_vessel_pass` or `inbound_vessel_pass` | contact before the ceiling-fixture approach reaches its lateral waypoint |

Role 607 shows the same early right-side pattern but was already baseline-dirty;
role 606 was a sampling failure in the wide run. The conservative run must be
joined and classified independently rather than assumed identical.

The production route currently applies the fixture bow to `inbound_prefix` and
then appends `inbound_grasp` afterward. This is a route-composition boundary,
not evidence by itself that the aperture is too narrow.

### 2. The design lag has no reproduced physical definition

The predictor uses 0.208 m for negative bows and 0.108 m for positive bows.
Their recorded provenance says that at TCP `y=-0.268`, the wrist was at
`y=-0.061`. The promoted lag instrument instead measures the
`robot_0/fr3_link6` **body origin**, producing approximately 0.04–0.05 m and
explicitly calling the design values a “different wrist point.”

That is not a re-measurement of the design quantity. A valid quantity needs all
of the following:

- exact MuJoCo body, geom, site, contact point, or support point name;
- exact world-space coordinate being measured;
- sign convention for negative and positive bows;
- exact frame eligibility rule, including x/z overlap with the pendant;
- aggregation rule across frames and rows;
- a script and artifact that reproduce it.

Until that definition reproduces the source values, the face window
`[0.044, 0.156]` is an algebraic result from unverified physical inputs.

### 3. The existing lag replay accepts invalid reconstruction

The offset-wide `wrist_lag.json` includes role 607 with a recorded-versus-live
TCP residual of about 0.106 m. The script prepares one task from the first
source manifest row and reuses it for every row. It also samples only the three
ceiling-fixture phases, even though contact onset occurs in vessel/cross-vessel
or pregrasp phases.

No aggregate may include a row with a failed reconstruction residual.

### 4. The specified targeted test suite is not green

The current targeted run is 16 passed / 1 failed. The failure imports the
nonexistent `PlaceContactAuditor`; the production class is
`PactPlaceContactAudit`. This must be repaired with a behavioral assertion,
not another brittle source-string assertion.

## Step 0 — Preserve and restate the decision boundary

Before changing code, create no new result directory and run no simulation.
Confirm the two paired summaries and guard from the immutable inputs above.

The working conclusion for all subsequent outputs is:

> Both pre-registered offset candidates failed the paired preservation rule, so
> V9.8 stops before the 24-row gate. The failure mechanism and the physical
> validity of the lag-derived window remain unresolved.

This wording must not be strengthened during the remediation unless the new
retained-state diagnostic meets every acceptance condition below.

## Step 1 — Repair the false-green unit verification

In `tests/test_pact_place_v98_pendant.py`:

1. Import the real `PactPlaceContactAudit` class.
2. Replace source inspection with a behavioral test showing that a
   `mounted_fixture` entry contributes to `non_target_contact_entries` and
   makes `collision_free` false.
3. Keep the existing contract, dispatch, side-dependent envelope, waypoint,
   no-clipping, and wall-fixture isolation tests.

Run the three pre-registered targeted files immediately after the repair. Do
not describe verification as green until the command exits zero.

## Step 2 — Build a row-specific retained-qpos contact diagnostic

Add `scripts/diagnose_pact_place_v98_offset_contacts.py` and write its output to
the new directory:

`diagnostics_output/pact_place_v98_offset_contact_diagnosis/`

The diagnostic must operate on both `wide` and `cons`. It must not overwrite
`wrist_lag.json`.

### Row reconstruction

For each row directory:

1. Load `result.json` and `trajectory.json`.
2. Match `episode_id` to that row's own manifest record in
   `pact_place_v95_raw_smoke/summary.json`.
3. Patch that row with the exact fixture recorded in the result, the V9.8
   sampler, contract version, and lateral-bow flag used by the paired run.
4. Pass the explicit `pact_place_corridor_v5.xml` scene path.
5. Seed and construct a fresh task **for that row**. Never reuse a task made
   from another row's manifest.
6. Restore each recorded qpos and call `mj_forward`; do not step physics.

For every restored frame, compare the live TCP position with the trajectory's
recorded TCP. Record per-frame residual and per-row maximum. A row is valid
only if the maximum residual is at most 1 mm. A larger residual is a hard
diagnostic failure: exclude it from aggregates, explain it, and do not infer
geometry from it.

### Per-frame evidence

For every valid retained frame, record at least:

- role, family, side, candidate, episode id, step, and policy phase;
- TCP xyz;
- whether TCP x is before, inside, or after the pendant x interval;
- pendant center, half extents, and six AABB faces;
- all live `mounted_fixture` contact pairs, including exact geom/body/root
  names on both sides, contact position, signed distance, and penetration;
- the world pose of each contacting robot geom and its parent body;
- `fr3_link6` body origin for comparison, clearly labeled as a body origin;
- whether the phase is part of the ceiling-fixture approach/pass/exit set.

Use the live contact classifier from `pact_place_contact_audit.py`. Do not
infer a contacting link from sensor activity or a terminal symptom.

The production audit samples internal 2 ms physics states while the retained
trajectory stores control-step qpos. Join the authoritative
`first_contact_step.mounted_fixture` to the retained step and inspect at least
the neighboring control steps. If the exact live pair is absent at retained
resolution, record that limitation; do not invent a geom identity.

### Per-row onset record

For every complete row, emit:

- authoritative first contact step from the original audit;
- retained policy phase at that step;
- terminal policy phase separately;
- first reconstructed contacting pair and step, if available;
- difference between authoritative and reconstructed onset steps;
- contact classification from the frozen categories below;
- total and maximum penetration from the original result;
- reconstruction validity and residual.

Frozen contact-onset categories:

- `early_approach_coverage`: contact starts before
  `inbound_ceiling_fixture_approach`;
- `protected_ceiling_bow_contact`: contact starts during ceiling-fixture
  approach/pass/exit;
- `post_bow_pregrasp_coverage`: contact starts after ceiling-fixture exit or in
  pregrasp/grasp;
- `unreconstructed`: row or pair cannot be reconstructed within tolerance.

Do not classify from `terminal_policy_phase`.

## Step 3 — Audit and either reproduce or invalidate the lag provenance

Use the same row-specific reconstruction on the centred source runs named in
the provenance and on both offset candidates.

Measure named quantities separately; never call all of them “wrist lag”:

1. TCP-to-`fr3_link6` body-origin lateral displacement.
2. TCP-to-contacting-robot-geom body-origin displacement.
3. Collision-facing extent or nearest/contact point of the exact robot geom
   relative to the TCP while that geom and pendant overlap in x/z.
4. Exact signed robot-geom-to-pendant-geom distance where the MuJoCo API
   provides it.

For a negative route, the collision-facing lateral extent is the robot's
positive-y side; for a positive route it is the negative-y side. Record the
sign convention explicitly.

The 0.208/0.108 constants count as reproduced only if one named,
collision-relevant definition reproduces the source-row ranges within 2 mm and
the reconstruction residual is within 1 mm on every contributing row. The
definition and contributing frames must appear in the JSON artifact.

If no such definition exists:

- mark both constants `unverified_provenance`;
- mark the `[0.044, 0.156]` face window `physical_input_invalid`;
- describe the existing predictor as an algebra/dispatch test only;
- do not substitute the 0.04–0.05 m body-origin number into the old formula;
- do not derive new envelope constants or candidates in this plan.

If a valid definition does exist, recompute its value on the offset rows only
as a diagnostic. A moved value still does not authorize a new candidate or
episode.

## Step 4 — Produce the causal classification

Write
`diagnostics_output/pact_place_v98_offset_contact_diagnosis/diagnosis.json`
with a schema version, input hashes, `authorizes_new_episodes: false`,
`authorizes_gate: false`, and `authorizes_collection: false`.

The top-level decision must be one of:

- `route_composition_coverage_failure`: baseline-clean failures start outside
  the protected ceiling-bow phases;
- `verified_envelope_failure`: the lag definition is reproduced and failures
  start within the protected bow with collision-relevant clearance violated;
- `mixed_route_and_envelope_failure`: both mechanisms are directly observed;
- `mechanism_unresolved`: reconstruction or provenance requirements fail.

`verified_envelope_failure` is allowed only if **all** lag-provenance and
reconstruction acceptance conditions pass. The existence of hundreds of
contact samples, by itself, does not select that conclusion.

Include compact joins for both candidates and both panel sides. Counts must
distinguish all complete rows from the six baseline-clean rows.

## Step 5 — Correct the documentation without erasing history

Append a dated superseding audit entry to
`docs/PACT_PLACE_V98_PENDANT_PLAN.md`. Preserve the preregistration, candidate
results, and old artifacts as history.

Update `EVAL.md` and `README.md` so that all three documents agree:

- the paired selection stop is valid and final for the named candidates;
- no 24-row offset gate, S2b, S4, collection, or training ran;
- contact onset and terminal phase are different fields;
- the observed onset categories are reported row-for-row;
- the physical status of 0.208/0.108 and the face window matches Step 3;
- the preflight proved live `_bow_segment` algebra, dispatch, and no clipping,
  not complete swept-arm clearance;
- no global claim is made that a wrist-height hazard is unavoidable in a
  0.85 m aperture unless `verified_envelope_failure` is validly selected.

Replace the README heading “the wrist does not follow the TCP bow” with a
neutral result heading such as:

> V9.8 ceiling pendant gate — stopped at the paired offset selection rule

Do not silently edit the original claim. State that the retained-state audit
supersedes its causal interpretation.

## Step 6 — Verification

Use the project interpreter and environment:

```bash
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR=/root/prox_learning/assets
export PYTHONPATH=/root/prox_learning_pact_remediation/submodules/molmospaces
cd /root/prox_learning_pact_remediation
```

Required checks:

```bash
/root/act_retrain_venv/bin/python -m pytest -q \
  tests/test_pact_place_v98_pendant.py \
  tests/test_pact_place_v95_low_wall.py \
  tests/test_pact_place_v94_mounted_preview.py

/root/act_retrain_venv/bin/python \
  scripts/diagnose_pact_place_v98_offset_contacts.py

git diff --check
```

Add focused unit tests for:

- row-specific reconstruction rather than first-row template reuse;
- rejection of TCP residuals above 1 mm;
- onset classification using first-contact step rather than terminal phase;
- exclusion of invalid rows from lag aggregates;
- behavioral inclusion of `mounted_fixture` in strict non-target contact;
- artifact authorization fields remaining false.

The known unrelated `tests/test_pact_place_corridor.py` XML-hash failures are
outside this plan. Do not fix or use them to excuse a failure in the targeted
suite.

## Acceptance criteria

The remediation is complete only when all of the following hold:

1. The three targeted test files exit zero.
2. Both offset candidates have a row-level diagnosis joined to the immutable
   paired results.
3. Every aggregate uses only row-specific reconstructions with maximum TCP
   residual at or below 1 mm.
4. Every baseline-clean failure has an authoritative onset step and phase; any
   missing geom identity is explicitly labeled rather than guessed.
5. The exact physical definition behind 0.208/0.108 is either reproduced from
   code and retained states within tolerance or formally marked unverified.
6. The diagnostic distinguishes body origin, geom origin, collision-facing
   extent, nearest point, and contact point.
7. The predictor is no longer presented as proof of complete swept-volume
   clearance.
8. README, EVAL, and the V9.8 plan state the same narrow conclusion.
9. Existing artifacts remain byte-unchanged.
10. No new episode, physics step, gate, collection, training, or evaluation was
    run.

## Final disposition and handoff

Regardless of the diagnostic category, V9.8 remains stopped under the existing
selection rule. This plan does not reopen candidate design.

The coding agent's final report must provide:

- targeted test result;
- diagnostic artifact path and hash;
- per-candidate onset-category counts for baseline-clean rows;
- exact contacting geoms where reconstructable;
- lag-provenance verdict;
- selected top-level causal category;
- confirmation that all authorization fields are false and no episodes ran;
- files changed.

Any proposal to repair route coverage, change an envelope, design another
fixture, or resume V9.8 requires a separate plan and explicit authorization.
The higher-value 3-seed × 100-episode place power run remains a separate open
project item and is not authorized by this diagnostic remediation.
