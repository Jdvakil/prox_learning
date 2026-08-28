# PACT place V10.4: first-shot static pendant through human review and Phase 0

**Status:** **Steps 0, 1 and 2 passed; stopped at Step 3.** The environment
qualified offline and in six live production rows (6/6 strict clean, zero
pendant contact), and the panel causal check passed on both sides. The
owner-review packet could not be completed: the first registered diagnostic
negative control cannot reach the registered 5–30 mm penetration band on the
frozen grid, and the plan requires stopping rather than extending it. Phase 0
was not run and `human_approval.json` is absent. The measured close-out is in
[Measured status](#measured-status) at the end of this file.

**Scope:** build one new environment, perform its offline preflight, generate a
six-video owner-review packet, and—only after explicit owner approval—run one
24-row Phase-0 expert gate. This plan never authorizes demonstration collection,
conversion, training, learned-policy evaluation, or a retry of the official
gate.

## Objective and governing disposition

The goal is to qualify a final static-pendant environment with as little live
iteration as possible. Do not continue the V9.5/V10.1 route family: the frozen
V9.5 varied-seed audit predicts only 12.25 clean rows out of 24 against a 20-row
gate. Instead, use the already qualified V6c environment and expert as the
base. V6c passed its strict Phase-0 gate at 23/24 with zero clutter contact:

- `configs/pact_place_corridor_v6c.json`
- `diagnostics_output/pact_place_corridor_v6c/expert_screen.json`
- `docs/PACT_PLACE_CORRIDOR_GATE_V6C.md`

Add one compiled-static, symmetric two-lobe pendant outside the V6c trajectory
envelope. The pendant is a real collision hazard: contact with any lobe, stem,
or crossbar is a failure. It does not have to force a five-centimetre detour and
the expert must not acquire a new pendant-specific route.

The owner review packet contains exactly six videos:

1. three clean production examples from a fixed six-row qualification stream;
2. three clearly labeled failures, using naturally occurring production
   failures first and deterministic diagnostic negative controls only to fill
   any shortfall.

Negative controls validate visibility and scoring. They are not production
episodes, are not a clean-rate estimate, and can never authorize Phase 0 by
themselves.

## Non-negotiable boundaries

1. Preserve V6c and every V9.5–V10.3 artifact byte-for-byte. Never overwrite a
   historical scene, config, diagnostic, manifest, NPZ, or video.
2. The production pendant is part of the compiled MJCF. Its body has no joint,
   freejoint, or mocap flag. Do not write `model.geom_pos`, `model.geom_size`,
   `model.geom_aabb`, `model.geom_rbound`, or BVH fields at episode runtime.
3. Visible and collision geometry are identical. No visual-only sleeve, hidden
   collider, or skinny collision stem is allowed.
4. A robot or carried-target contact with a `pact_clutter_mount_v104*` geom must
   be classified as `mounted_fixture` and must make the row unclean.
5. Keep the V6c route geometry and IK behavior. No pendant route search, lane
   rewrite, detour quota, sequential-IK search, geometry search, or post-hoc
   route fallback is authorized.
6. Exactly one production geometry and one production speed schedule are
   frozen before the first live V10.4 row. Do not use review or Phase-0 outcomes
   to tune either one.
7. Review and Phase-0 seed streams are disjoint. A review row is never reused,
   substituted, or counted in Phase 0.
8. A missing row, sampling failure, planning failure, missing telemetry, hash
   mismatch, or infrastructure failure fails closed according to the rules
   below. Never replace a failed official row with another seed.
9. The coding agent must stop after generating the six-video review packet.
   It must not infer approval, create `human_approval.json`, or run Phase 0.
10. Even a Phase-0 pass authorizes nothing downstream of this document.

## Frozen V10.4 production design

### Base environment and expert

Copy `pact_place_corridor_v3.xml` to a new additive scene; do not edit the V3
scene. Retain the V6c sampler, fixed clutter, panel behavior, target/tray
geometry, jitter distributions, contact semantics, release check, and expert
route:

```text
environment_version: pact_place_corridor_v10_4_first_shot_static_pendant
contract_version:    pact_place_v104_first_shot_static_pendant_v1
base_sampler:        PactPlaceCorridorV3Sampler
base_config:         configs/pact_place_corridor_v6c.json
new_scene:           submodules/molmospaces/molmo_spaces/data_generation/
                     custom_scenes/pact_place_corridor_v10_4.xml
```

A thin pass-through V10.4 sampler may be added only if needed to assert the
environment marker and expose metadata. It must not change V6c sampling or
layout behavior. A same-manifest V6c/V10.4 reconstruction must match initial
robot, target, panel, clutter, tray, shell, and camera state; the new pendant
and the one speed cap are the only allowed differences.

### Production pendant geometry

Register exactly this assembly. This is a trust anchor, not a search lattice.

| Component | Centre `(x,y,z)` m | Half extents `(x,y,z)` m |
|---|---|---|
| negative lobe | `(0.78, -0.34, 1.01)` | `(0.010, 0.010, 0.030)` |
| positive lobe | `(0.78, +0.34, 1.01)` | `(0.010, 0.010, 0.030)` |
| negative stem | `(0.78, -0.35, 1.2725)` | `(0.006, 0.006, 0.2325)` |
| positive stem | `(0.78, +0.35, 1.2725)` | `(0.006, 0.006, 0.2325)` |
| crossbar | `(0.78, 0.0, 1.510)` | `(0.006, 0.356, 0.005)` |

Consequences that the contract must assert:

- both lobe bottoms are `z = 0.98 m` and tops are `z = 1.04 m`;
- the shelf-to-lobe gap is `0.26 m` (`SHELF_TOP_Z = 0.72 m`);
- both stems are 12 mm square and join the outward lobe faces to `z = 1.505 m`;
- the crossbar is 12 mm thick in x, 10 mm high, and its top face is flush with
  `hood_top` at `z = 1.515 m`;
- the complete assembly is symmetric in y and identical for left/right panel
  episodes;
- all five components are static, collision-enabled, and visibly rendered;
- use a matte neutral material; do not recolor the actual wrist image or add a
  fake review tint.

Use body/geom names under the existing strict taxonomy, for example:

```text
pact_clutter_mount_v104
pact_clutter_mount_v104_lobe_0_g
pact_clutter_mount_v104_lobe_1_g
pact_clutter_mount_v104_stem_0_g
pact_clutter_mount_v104_stem_1_g
pact_clutter_mount_v104_crossbar_g
```

### Frozen speed amendment

Preserve every V6c route pose, segment boundary, orientation, IK seed, and
non-initial segment speed. On the first TCP move sequence before gripper close,
cap only its first free-space segment at `0.12 m/s`:

```text
initial_free_space_speed_cap_m_s: 0.12
all_later_segment_speeds: inherited byte-for-byte from V6c
task_horizon: 1050 control steps
```

Gate the change on the exact V10.4 environment marker. Assert that exactly one
segment is changed and that no V6c or historical environment reaches the
branch. Do not identify the segment only by a broad phase string; bind it by
primitive order and verify its start/end pose against the original V6c plan.

Record per control frame:

- primitive index, segment index and segment name;
- commanded speed and realized TCP speed;
- policy/traversal phase;
- exact minimum pendant clearance and limiting component/body;
- per-component contact state;
- target-held state and ordinary contact audit.

## Streams and fixed row counts

```text
production review stream: pact-place-v10.4-first-shot-review-production
production master seed:   2026104002
production rows:          6 (three left, three right)

Phase-0 stream:           pact-place-v10.4-first-shot-phase0
Phase-0 master seed:      2026104001
Phase-0 rows:             24 (twelve left, twelve right)
```

Derive each `task_seed_u32`, `task_seed_u64`, clutter jitter, panel jitter, and
row hash deterministically from `(stream, master_seed, role_index)`. Emit both
manifests before executing their first row and prove that their episode IDs and
task seeds are disjoint. Side order must be frozen and balanced, not selected
after outcomes.

## Step 0 — implementation and offline preflight

Step 0 may compile models and call FK, exact distance, `mj_forward`, raw sensor
rendering, and deterministic qpos replay. It must not call `env.step` or create
an episode.

### 0A. Provenance and immutability

Create a V10.4 contract with byte-level SHA-256 verification. Recompute hashes
from file bytes; do not trust an embedded JSON hash as the verification result.
At minimum protect:

- the V6c config, scene, expert-screen summary, 24 row results and 24
  trajectories;
- V9.5 reconstruction inputs used by V9.9/V10;
- V10.1 review artifacts;
- V10.2 preflight/contact-parity artifacts;
- V10.3 search and endpoint certificate.

Hash every implementation input that can affect a result, including the new
scene generator, generated XML, contract, sampler/policy dispatch, distance and
contact logic, runners, renderer, and tests. Immutable writers must use atomic
create-if-absent semantics and refuse to replace an existing artifact.

### 0B. Reproduce the independent geometry trust anchor

Write a small, minutes-long exact diagnostic and record all per-row witnesses.
It must independently reproduce or improve these read-only audit results:

| Population | Required result |
|---|---:|
| six retained V9.5 clean cells, inbound and loaded outbound | all paths `>= 0.035 m` |
| same six-cell minimum observed by the audit | approximately `0.04052 m` or higher |
| 24 frozen V6c Phase-0 trajectories | 24/24 `>= 0.050 m` |
| same V6c minimum observed by the audit | approximately `0.05523 m` or higher |

Score every retained control frame, all collision-enabled robot geoms, and the
carried target where applicable. GJK/true distance is authoritative; TCP or
AABB distance is not. A result more than 1 mm below either observed minimum is
a provenance discrepancy and stops the plan even if it remains above the hard
floor.

Also score the eight rigid `+/-5 mm` assembly-translation corners. Every corner
must retain at least 30 mm clearance on the six V9.5 cells and at least 40 mm
on the 24 V6c trajectories. These corners are an offline robustness diagnostic;
the production pendant itself has no jitter.

### 0C. Static-environment and initial-state predicates

Across all frozen V6c/V9.5 reference states, require:

- no pendant overlap with panel, fixed clutter, tray, shell, hood, initial
  target, or robot;
- only the designed crossbar/`hood_top` flush face is allowed to touch;
- no assembly self-overlap except the intended lobe-stem and stem-crossbar
  attachment faces;
- compiled `geom_aabb`, `geom_rbound`, and body BVH bounds enclose the real
  final geometry without runtime repair;
- the production scene has no joint, freejoint, or mocap ID on the pendant
  body.

### 0D. Contact parity and failure semantics

Use separately compiled diagnostic-only scenes or poses; never mutate the
production scene artifact.

1. A known-clear robot/pendant pose must be clear under exact distance and live
   `data.contact`.
2. Deliberately penetrate each lobe and at least one 12 mm stem by 5–30 mm.
3. Hardened signed `mj_geomDistance`, analytic GJK, live `data.contact`, and
   `PlaceContactAudit` must agree on contact/non-contact and sign.
4. Robot–pendant and carried-target–pendant cases must classify as
   `mounted_fixture` and make strict cleanliness false.
5. Restore and hash the scene state before and after every diagnostic probe.

This specifically guards against the stale broad-phase defect discovered in
V10.2.

### 0E. Route and speed preservation

For all 24 frozen V6c manifests, construct both policies without stepping
physics. Require:

- identical waypoint positions/orientations and primitive ordering;
- identical IK outputs and route/fallback diagnostics;
- exactly the one registered initial speed difference;
- every other commanded speed unchanged;
- no V10/V10.1/V10.2/V10.3 route dispatch reached;
- predicted completion below 80% of the 1050-step horizon.

### 0F. Step-0 stop rule

Write:

```text
diagnostics_output/pact_place_v104_preflight/preflight.json
diagnostics_output/pact_place_v104_preflight/clearance_witnesses.npz
diagnostics_output/pact_place_v104_preflight/contact_parity.json
```

If any Step-0 predicate fails, stop with all authorization fields false. Do not
change geometry, route, speed, threshold, or manifest within this plan.

## Step 1 — six fixed production qualification rows

Only a passing immutable Step 0 authorizes these six episodes. Run exactly the
prewritten six-row production manifest. Use the normal live rollout path with
`env.step`, the real contact audit, and full telemetry. Do not resample or
replace a failed row.

A **strict clean success** requires all of the following:

- upstream task, grasp and placement success;
- zero `mounted_fixture`, `clutter`, `hazard_bar`, and `other_environment`
  contacts;
- zero place-receptacle contact outside the established placement exemption;
- exact per-frame pendant clearance at least 20 mm;
- no initial contact, free-body drift/stability event, empty-gripper defect,
  route fallback, IK cascade, missing telemetry, or hash mismatch;
- compiled production scene and frozen speed schedule used throughout.

The production pack is eligible for owner review only when:

- all 6/6 rows reconcile and complete with no sampling/infrastructure failure;
- at least 5/6 are strict clean successes;
- at least 2/3 are strict clean on each panel side;
- zero row touches any pendant component;
- all observed minimum pendant clearances are at least 20 mm;
- the speed amendment is reached and the telemetry proves the initial cap;
- the causal panel check in Step 2 passes.

Write every result and trajectory under a new versioned directory plus:

```text
diagnostics_output/pact_place_v104_review_production/production_manifest.json
```

The six-row outcome is a qualification check, not a clean-rate estimate.

## Step 2 — fresh two-row panel causal preservation check

Choose the lowest-role-index strict-clean left row and lowest-role-index
strict-clean right row from Step 1. At identical retained qpos and without
stepping physics, render the production 40-sensor raw proximity tensor in two
worlds:

1. selected production scene with the active panel present;
2. the same state with only that panel parked.

The pendant, clutter, target, robot, camera, and every other scene value remain
identical. Render a repeated baseline to measure noise. Hash and retain all raw
tensors. Use the actual decision window; do not substitute sampled cones, AABB
distance, or an encoder embedding.

Each side must satisfy:

- tensor contract `[40, substeps, 8, 8]` with the production substep count;
- at least 3 distinct changed sensors;
- at least one responding link5/link6 sensor;
- at least 448 changed values above
  `max(ABS_DELTA_FLOOR_M, 10 * repeat_noise)`;
- at least 7,209 changed values as the panel-preservation floor (25% of the
  historical minimum panel effect, 28,836);
- left/right changed-value ratio at most 4x;
- zero unexplained repeated-render noise or state-hash drift.

This is deliberately panel-only admission. The static pendant is not required
to produce its own causal proximity effect.

Write:

```text
diagnostics_output/pact_place_v104_causal/causal.json
diagnostics_output/pact_place_v104_causal/raw/*.npz
```

Failure blocks owner-review eligibility and Phase 0. Do not lower either floor.

## Step 3 — exactly six owner-review videos, then stop

Build the review list deterministically after Steps 1–2 pass.

### Three production-success clips

Select:

1. the strict-clean left row with the smallest exact pendant clearance;
2. the strict-clean right row with the smallest exact pendant clearance;
3. the smallest-clearance remaining strict-clean row, tie-breaking by role
   index.

### Three failure clips

Take completed natural production failures first, sorted by role index. Fill
the remaining slots from deterministic diagnostic negative controls in this
order:

1. left-lobe contact;
2. right-lobe contact;
3. stem contact.

For a diagnostic control, use retained production qpos and a separately
compiled, diagnostic-only scene. Evaluate the frozen inward-shift grid
`0.000, 0.001, ..., 0.160 m` and choose the smallest shift that creates 5–30 mm
exact penetration and live `mounted_fixture` contact. Record every tested
shift. This deterministic control-only grid is not a production-geometry
search. Do not modify or overwrite the production XML. If a control cannot
achieve the registered penetration band, stop rather than extending the grid.

Burn a persistent red banner into every control:

```text
DIAGNOSTIC NEGATIVE CONTROL — NOT PRODUCTION GEOMETRY — NOT AN EPISODE
```

Every video must be true-time at the 66 ms control period with frame stride 1.
Use three panes:

1. untinted wrist RGB;
2. third-person task view;
3. pendant side view showing both lobes, stems, crossbar, and the limiting arm
   or target body.

Overlay production/control status, side, phase, segment, commanded/realized
speed, exact clearance, limiting pair, cumulative contact counts, task outcome,
and whether the target is held. Do not tint the wrist pane; any visibility aid
belongs only in a separately labeled review pane.

Write exactly:

```text
diagnostics_output/pact_place_v104_review/review_manifest.json
diagnostics_output/pact_place_v104_review/REVIEW.md
diagnostics_output/pact_place_v104_review/videos/*.mp4  # exactly six
```

The manifest must distinguish `production_clean`, `production_failure`, and
`diagnostic_negative_control`, retain all source hashes, and state
`clean_rate_is_not_an_estimate: true`.

Then stop and report the six clickable video paths to the owner. Leave:

```text
eligible_for_human_review: true
human_approval_present: false
authorizes_phase0: false
authorizes_gate: false
authorizes_collection: false
authorizes_training: false
authorizes_evaluation: false
```

Do not create `human_approval.json`.

## Owner review and approval contract

The owner should review all six clips and confirm:

1. the production pendant is visibly above and separated from the table
   clutter;
2. lobes, stems, and crossbar remain static;
3. production successes contain no pass-through or touch;
4. negative-control contacts are visibly and numerically detected;
5. the route remains smooth and does not jump IK branches;
6. the initial free-space motion is acceptably slower;
7. overlays agree with visible motion and outcome.

Only an owner-supplied file with `decision: approve_phase0` may unlock Step 4.
It must bind the byte hashes of the contract, implementation aggregate, scene,
Step-0 preflight, production manifest, causal artifact, review manifest, and all
six videos. Absence, rejection, ambiguity, stale hashes, or any code/scene
change leaves Phase 0 unauthorized.

## Step 4 — one untouched 24-row Phase-0 gate

This section is implemented and tested before review but is not executed until
the valid owner approval exists.

Freeze the complete 24-row manifest, config, predicted range, scene,
implementation, thresholds, and approval binding before row 0. Expected clean
range is 21–24/24; the pass threshold remains 20/24.

Run exactly 24 rows from the registered Phase-0 stream, twelve per side. No
review seed, outcome-based substitution, retry, or extra row is allowed.

Phase 0 passes only with:

- 24/24 rows reconciled;
- zero infrastructure failures;
- at least 20/24 strict clean successes;
- at least 9/12 strict clean successes on each side;
- zero pendant contact in every row, including otherwise unsuccessful rows;
- no per-frame exact pendant clearance below 15 mm;
- no missing contact/clearance/speed telemetry or provenance mismatch.

Sampling or planning failures count as failed rows and are not replaced. Use
the same strict contact, grasp, placement, stability, and fallback semantics as
Step 1, with the gate's 15 mm clearance floor.

Write immutably:

```text
diagnostics_output/pact_place_v104_phase0/gate.json
diagnostics_output/pact_place_v104_phase0/rows/*
```

On failure, close V10.4 with the exact cause. Do not tune, select a fallback
candidate, change a threshold, or run another official gate under this plan.
On pass, set only `phase0_passed: true`; all downstream authorization fields
remain false.

## Suggested additive implementation surfaces

Names may differ, but keep V10.4 isolated from historical branches:

```text
scripts/pact_place_v104_contract.py
scripts/pact_place_v104_geometry.py
scripts/pact_place_v104_runtime.py
scripts/run_pact_place_v104_preflight.py
scripts/run_pact_place_v104_review_production.py
scripts/run_pact_place_v104_causal.py
scripts/run_pact_place_v104_review_video.py
scripts/run_pact_place_v104_phase0.py
tests/test_pact_place_v104_first_shot.py
```

Reuse hardened exact-distance, contact-audit, raw proximity, cleanup, and
renderer helpers where their contracts match. Do not copy the V10 runtime geom
posing defect or import V10.3 route-search behavior.

## Required behavioral tests

At minimum test:

1. exact geometry coordinates, symmetry, attachments, and scene serialization;
2. compiled-static body has no joint/freejoint/mocap and needs no bound refresh;
3. visible and collision stem/crossbar dimensions are identical;
4. environment/initial-target nonintersection and only allowed hood attachment;
5. positive and negative contact parity for lobe, stem, robot, and target;
6. all pendant contacts classify `mounted_fixture` and make cleanliness false;
7. V6c route poses/IK are unchanged on the same manifest;
8. exactly one initial segment is capped at 0.12 m/s only under V10.4;
9. per-frame telemetry and qpos restoration on success, failure, and exception;
10. review/Phase-0 seed disjointness and fixed side balance;
11. six-row eligibility (`>=5/6`, `>=2/3` per side) and fail-closed status rules;
12. causal raw-tensor floors, responding-link rule, noise and side-ratio checks;
13. deterministic three-success/three-failure clip selection and control labels;
14. approval rejects missing, stale, mismatched, or agent-created records;
15. Phase-0 counting (`>=20/24`, `>=9/12` each side), no replacement, and no
    downstream authorization;
16. immutable writers refuse an existing target and recompute byte hashes.

Run focused V10.4 tests plus the V6c expert/contract, contact-audit,
mounted-fixture, V10.2 contact-parity, and V10.3 static-scene regressions before
Step 1 and again after documentation. Record exact commands and counts. Run
`git diff --check`. The three known stale-scene-hash failures in
`tests/test_pact_place_corridor.py` may be reported only if independently
confirmed unchanged; do not hide new failures behind them.

## Required handoff after Step 3

Return to the owner with:

- completed task list and exact test results;
- preflight, production, causal, and review artifact paths and SHA-256 values;
- the six production-row outcomes and minimum clearances;
- panel causal counts by side and responding sensors/links;
- exactly six clickable MP4 paths, labeled as three successes and three
  failures/controls;
- confirmation that the production scene is compiled-static;
- confirmation that no Phase-0 row was run and `human_approval.json` is absent;
- every authorization field, which must remain false.

Do not continue automatically after reporting.


## Measured status

Everything below is measured output. The plan text above is unchanged.

**Steps 0, 1, and 2 passed. Step 3 stopped** on the plan's own rule: the first
registered diagnostic negative control cannot reach the registered 5–30 mm
penetration band on the frozen inward-shift grid, and the plan says to stop
rather than extend the grid. No owner-review packet was produced, no
`human_approval.json` exists, and Phase 0 was not run.

### Artifacts and hashes

| Item | Value |
| --- | --- |
| Contract | `pact_place_v104_first_shot_static_pendant_v1`, SHA `455379b852c994c6e4645b5650e8c690ebbc542509b40700316c8888db977707` |
| Implementation aggregate | `bf4af91fb5308a09f29e45652d10c0ee3b8227bdb0d2397ccc0bd5a4b55edd0e` |
| Production scene | `submodules/.../pact_place_corridor_v10_4.xml`, SHA `01d8adf34808a9f419cb3a9d07668ec1069d3a5acfa8cb01885c622ea09876f7` |
| Step-0 preflight | `diagnostics_output/pact_place_v104_preflight/preflight.json`, SHA `fe64e285332a3c530cab30599d2b862823a3c1e2db661a6961a0a1461f2c41d5` |
| Clearance witnesses | `diagnostics_output/pact_place_v104_preflight/clearance_witnesses.npz` |
| Contact parity | `diagnostics_output/pact_place_v104_preflight/contact_parity.json` |
| Step-1 production | `diagnostics_output/pact_place_v104_review_production/production_manifest.json`, SHA `fdcf757b4bff512c71c6e3ac241c151742523c89ec7531b46132af715e92b3af` |
| Step-2 causal | `diagnostics_output/pact_place_v104_causal/causal.json`, SHA `a30c863d61537edb58d24cc91b13291fa5d9efc47c7521b08b2304943f2f2ffc` |
| Step-3 stop | `diagnostics_output/pact_place_v104_review/control_shortfall_stop.json`, SHA `9a4abfec9992edaff3bf957dc90900c2a6dd50e430e8cb7f7cf873fe4d11103a` |
| Review manifest / videos | **not written** — the packet could not be completed |
| `human_approval.json` | **absent, not created** |
| Phase-0 gate | **not run** |

`eligible_for_human_review: false`. `human_approval_present: false`.
`authorizes_phase0: false`. `authorizes_gate: false`.
`authorizes_collection: false`. `authorizes_training: false`.
`authorizes_evaluation: false`. `phase0_passed: false`.

### Tests

`tests/test_pact_place_v104_first_shot.py`: **32 tests, all passing**, covering
exact geometry and symmetry, attachment and serialization, the compiled-static
body with no joint/freejoint/mocap and no bound repair, identical visible and
collision dimensions, only-the-designed hood face touching, `mounted_fixture`
classification, the speed amendment bound by primitive order with refusal paths,
a live V6c-versus-V10.4 plan comparison, stream disjointness and side balance,
byte-level provenance including a tamper check, the create-only immutable
writer, row admission and both eligibility rules, gate counting with no
replacement, and the approval contract rejecting missing, agent-created, stale,
and incomplete records.

Regression sweep: **334 passed**. The three failures in
`tests/test_pact_place_corridor.py` are the known stale v3/v5 scene-hash
assertions; the V3 scene bytes are independently confirmed unchanged by the
byte-level provenance check (61/61 protected artifacts matched), so they are
reported rather than hidden. `git diff --check` is clean in both trees.

One test caught a real defect before any episode ran: `empty_authorization()`
was being spread *after* the stage outcome key, silently resetting a passing
`phase0_passed` to false. Fixed, with the ordering constraint recorded in the
source.

### Step 0 — offline preflight

All six items passed.

| Item | Result |
| --- | --- |
| 0A provenance | **61/61** protected artifacts matched, recomputed from file bytes |
| 0B V9.5 trust anchor | min **0.04052 m** vs floor 0.035 and audit 0.04052 — delta **−0.00000 m** |
| 0B V6c trust anchor | min **0.05523 m** vs floor 0.050 and audit 0.05523 — delta **−0.000005 m**, **24/24** rows above floor |
| 0B corners | V9.5 worst **0.03337 m** (floor 0.030); V6c worst **0.04800 m** (floor 0.040) |
| 0C static and initial state | no disallowed overlap; the only touch is the designed `crossbar`/`hood_top` flush face, present on all 24 rows; compiled bounds enclose the final geometry; body has no joint, freejoint, or mocap |
| 0D contact parity | **30/30** fixture cases agree across hardened signed `mj_geomDistance`, analytic GJK, live `data.contact`, and the place contact audit; every penetration classifies `mounted_fixture`; all probe states restored |
| 0E route and speed preservation | **24/24** V6c manifests: poses identical, exactly one speed change (`0.20 → 0.12 m/s` at primitive 1 / segment 0 `pregrasp`), V6c itself never amended, max predicted **583** steps against the 840-step 80% limit |

Both trust anchors reproduce the read-only audit to five decimals. That is the
sharpest available evidence that the pendant sits where the audit said it does.

### Step 1 — six production rows

**6/6 strict clean successes**, 3 left and 3 right, against a `≥5/6` and
`≥2/3`-per-side bar.

| row | side | clean | min pendant clearance | pendant contact frames | control steps |
|---:|---|---|---:|---:|---:|
| 0 | left | yes | 0.06158 m | 0 | 542 |
| 1 | right | yes | 0.08070 m | 0 | 426 |
| 2 | left | yes | 0.08069 m | 0 | 607 |
| 3 | right | yes | 0.07955 m | 0 | 455 |
| 4 | left | yes | 0.07070 m | 0 | 628 |
| 5 | right | yes | 0.08494 m | 0 | 438 |

Zero pendant contact anywhere. The smallest observed clearance, 61.6 mm, is
about three times the 20 mm review floor. Every row reached and recorded the
speed amendment (`0.2 → 0.12 m/s`). Six rows are a qualification check, not a
clean-rate estimate: `clean_rate_is_not_an_estimate: true`.

### Step 2 — panel causal preservation

Both sides pass, replaying retained qpos without `env.step`.

| side | changed values | changed sensors | responding corridor links |
|---|---:|---:|---|
| left | 23,684 | 6 | `link5_back` |
| right | 12,712 | 6 | `link5_front`, `link6` |

Both are above the 448 floor and the 7,209 panel-preservation floor (25% of the
28,836 historical minimum). Left/right changed-value ratio **1.86**, within the
4× limit. Admission is deliberately panel-only; the static pendant is not
required to produce its own causal effect.

### Step 3 — stopped on the control shortfall

All six production rows are strict-clean, so there were **zero natural
production failures** and all three failure slots had to be filled by
diagnostic negative controls. The first in the registered order,
`left_lobe_contact`, cannot be made to touch:

- source: the deterministic side-matched choice, the smallest-clearance
  strict-clean left row (role 0), unshifted clearance **0.06158 m**;
- the full frozen grid `0.000 … 0.160 m` was evaluated, **161 shifts recorded**;
- maximum penetration reached at the grid end: **2.579 mm**, against a
  registered band of **5–30 mm**.

The clearance is mostly vertical and longitudinal rather than lateral, so a
purely inward translation closes it only asymptotically. The other two controls
*are* reachable on side-matched rows — `right_lobe_contact` reaches 9.57 mm and
`stem_contact` 36.64 mm at the grid end — but the plan fills controls in a fixed
order and the first one fails.

Per the plan, the grid was **not** extended, no substitute control or source row
was chosen to make it succeed, the production XML was never modified, and no
production geometry changed. `control_shortfall_stop.json` records every tested
shift.

This shortfall is a property of the *control recipe*, not of the environment.
It happens precisely because the pendant is far outside the arm's envelope —
the same fact that makes Steps 0–2 pass so comfortably.

### Deviations and judgement calls, stated explicitly

1. **Two Step-0 instrument failures preceded the passing run**, both mine, both
   preserved as `preflight_attempt_0*.json` rather than deleted. The first drove
   30 mm of penetration into a stem along its 12 mm axis, which is the
   degenerate deep box–box regime recorded in the V10.2 contact-parity root
   cause; the fixture now approaches along each component's long axis and treats
   exact touching as a boundary. The second counted the designed
   `crossbar`/`hood_top` flush face as a static overlap; the predicate now
   encodes the allowance the plan states. **No design parameter changed.**
3. **A third passing preflight was superseded**, preserved as
   `preflight_attempt_03_passed_superseded_by_complete_implementation.json`. It
   passed, but the implementation aggregate changed when the remaining runners
   and tests were written, and the production runner's own guard refused to
   generate episodes against a stale Step-0 hash. The preflight was re-run
   against the complete implementation set before any episode.
4. **Control source row.** The plan registers the control order and the shift
   grid but not the source episode. The runner uses the side-matched
   smallest-clearance strict-clean row, which is both deterministic and the most
   favourable available choice for reaching the band; no other left row would do
   better, so the shortfall is robust to that choice.
5. **Grid search cost.** Each grid point is evaluated exactly, but only over the
   frames that a rigid shift of `s` could make the worst frame — a sound bound,
   since such a shift changes any frame's clearance by at most `s`.

### Stop

V10.4 stops at Step 3. No geometry, route, speed, threshold, manifest, or grid
value was changed in response to any result. No Phase-0 row was run,
`human_approval.json` is absent and was not created, and collection, conversion,
training, and learned-policy evaluation remain unauthorized and were not run.
