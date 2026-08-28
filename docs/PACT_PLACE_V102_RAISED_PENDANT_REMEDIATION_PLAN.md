# PACT place V10.2: raised, collision-legible pendant and slower approach

**Status:** **stopped at Step 0.** The non-episode admission failed items 5, 6
and 7. No V10.2 episode was generated: no six-row screen, no 12-row review, no
causal replay, no Phase 0. Nothing in this document authorizes Phase 0,
collection, training, or evaluation, and none of those were run. The measured
close-out is in [Measured status](#measured-status) at the end of this file.

V10.1 remains permanently recorded at
`diagnostics_output/pact_place_v101_empirical_review/`. Do not overwrite its
contract, review rows, trajectories, videos, causal stop, or hashes. V10.2 is a
new environment/route/review version with new seed streams and output paths.

## Why V10.2 exists

Human inspection of the V10.1 gallery identified three design problems that
must be treated as real until measured otherwise:

1. The arm appears to pass through a vertical pendant stem in some views.
2. The pendant lobes are too low and visually merge with table clutter.
3. The initial empty-arm approach is too fast.

The source audit explains, but does not dismiss, those observations:

- V10.1 lobe bottoms are only `z=0.82/0.84 m`; the shelf top is `z=0.72 m`.
  The lowest lobe is therefore only 100 mm above the shelf.
- Each stem is a collision-enabled 6 mm square (`contype=8`,
  `conaffinity=15`). That is thin enough to alias or become occluded in the
  review render, and the empirical route intentionally skipped the old
  preclearance predicate. Zero recorded mounted-fixture contacts is not enough
  to establish visible or swept clearance.
- The inherited clear-space transport speed is `0.20 m/s`. V10.1 review videos
  also render every second 66 ms policy frame at 10 fps, so they play at about
  `0.132 / 0.100 = 1.32x` real time.

V10.2 raises the pendant, thickens its stems in both collision and visual
geometry, verifies collision/contact parity deliberately, assigns speed by
route piece instead of copying the initial fast speed to the whole lane, and
renders review videos at real time.

## Immutable inputs and authorization boundary

- Preserve every V9.5, V9.8, V9.9, V10, and V10.1 artifact byte-for-byte.
- Preserve the V10.1 conclusion: its review was ineligible and its causal stage
  lacked clean-cell inputs; it did **not** prove the pendant causally silent.
- Continue using only F0/F1/F2 and their V9.5/V10.1 paired-side-identical
  clutter layouts. F3 remains regression-only and is not admitted to V10.2.
- No lattice, morphology search, alternate lane search, three-lobe search, or
  threshold relaxation is authorized.
- All V10.2 diagnostics start with `authorizes_gate: false`,
  `authorizes_collection: false`, `authorizes_training: false`, and
  `authorizes_evaluation: false`.
- Any geometry, speed, route, cleanliness, proximity, or manifest change after
  the first V10.2 episode invalidates the whole V10.2 pack and requires a new
  version. Do not patch and continue under the same hashes.

## Registered V10.2 design

### Raised two-lobe assembly

Create a new `planning_probe_v102_raised_assembly()`; do not change
`planning_probe_assembly()` or reinterpret any V10/V10.1 artifact.

| component | center (m) | half extents (m) |
|---|---|---|
| negative lobe | `[0.70, -0.18, 1.14]` | `[0.01, 0.04, 0.04]` |
| positive lobe | `[0.70, +0.22, 1.14]` | `[0.01, 0.02, 0.02]` |

Consequences that the contract must assert:

- lowest pendant point: `1.10 m` exactly;
- shelf-to-pendant vertical gap: `1.10 - 0.72 = 0.38 m`;
- lobe x/y placement and asymmetry are unchanged from V10.1;
- stem centre y remains at the outward lobe faces (`-0.22`, `+0.24 m`);
- both stems end at `z=1.505 m` and the crossbar remains flush to the
  `hood_top` bottom at `z=1.515 m`;
- the assembly remains fixed/kinematic. This plan does not introduce physical
  swinging dynamics.

Use a 12 mm square for both stem collision and visible geometry
(`half_x=half_y=0.006 m`). Derive the stem z centres/half-heights from each
raised lobe top. Use the same 12 mm x thickness for the crossbar while keeping
its top at `1.515 m`. Do not add a larger visual-only sleeve: what the reviewer
sees must be the collision geometry.

Use a distinct environment/contract marker,
`pact_place_corridor_v10_2_raised_pendant`, while continuing to compile the V10
scene and park all legacy mounts. Manifest rows carry the full derived assembly
and its self-hash.

### Fixed endpoint-only route

Keep one route; do not search after observing results:

- rewrite primitive: `endpoint_only`;
- qualification mode: `empirical_live_contact_v2`;
- slab padding: `0.08 m` inbound and outbound;
- lane y: left `-0.30 m`, right `+0.30 m`, inbound and outbound;
- frozen grasp-side endpoint unchanged to `1e-9 m` and `1e-9 rad`;
- minimum detour against the stock TCP traversal: `0.05 m`;
- translation densification: at most `0.005 m`;
- rotation densification: at most `2 degrees`.

The route remains empirical: do not re-enable the known-bad scalar
robot-versus-all-environment preclearance. Pendant-only clearance and live
MuJoCo contact auditing are required below; ordinary environment contact is
decided by the existing live contact audit.

### Speed schedule

Do not globally change the inherited planner speeds or historical policies.
Apply this schedule only when the exact V10.2 contract marker and hash are
present:

| route piece | commanded TCP speed |
|---|---:|
| empty-arm clear-space approach to pendant entry | `0.15 m/s` |
| pendant entry/pass/exit, inbound and loaded outbound | `0.045 m/s` |
| final pregrasp/grasp approach | inherited `speed_slow = 0.08 m/s` |
| non-pendant post-pick transport/place | unchanged historical speed |

Repair `_v10_apply_lane` so it does not assign `segments[0].speed` to every
rebuilt piece. Classify the named lane pieces and assign the table above.
Record requested speed, realized TCP displacement per policy step, and segment
name in trajectory telemetry. Reject a V10.2 row if a pendant-pass segment is
commanded above `0.045 m/s` or the initial approach above `0.15 m/s`.

## Step 0: non-episode admission

Run this before generating a review episode. Stop V10.2 if any item fails.

1. Verify all protected artifact and scene hashes. Write a new V10.2
   implementation hash; never update an old artifact to the new hash.
2. Validate the raised assembly against the aperture, hood, active panel,
   clutter, initial target, and initial robot on all six frozen cells.
3. Re-run exact stock-route necessity on every F0/F1/F2 side cell in both
   directions. Require all `12/12` cell/direction cases to be obstructed by at
   least one active lobe/stem/crossbar. Raising the pendant is not allowed to
   turn it into decorative geometry.
4. Validate the fixed endpoint-only route geometrically on all twelve cases:
   correct side, no clipping/wrong-way travel, endpoints unchanged, and detour
   at least 50 mm.
5. Solve the fixed nominal route without the flawed all-environment scalar
   gate. Require a complete sequential-IK solution at every waypoint and save
   `waypoints_attempted`, `waypoints_solved`, and `complete_sequential_ik`.
   An environment abort after one waypoint must never count as an IK pass.
6. Measure robot/target-to-pendant distance against every active lobe, stem,
   and crossbar along the nominal retained-qpos route. Require at least 15 mm
   at every recorded waypoint. Report each component separately; do not reduce
   the evidence to an anonymous scalar.
7. Add a deliberate contact-parity fixture for both stems: temporarily pose
   each collision stem across a known retained robot collision geom, call
   `mj_forward`, require a live contact classified as `mounted_fixture`, then
   restore and re-hash the original scene state. Verify every active component
   has `contype=8`, `conaffinity=15` and a collision-compatible robot pair.

Write `diagnostics_output/pact_place_v102_preflight/preflight.json` with named
per-cell, per-direction, per-component witnesses. It must keep every downstream
authorization false.

## Step 1: runtime and review-video verification

Add behavioral tests before episodes:

- V10.2 dispatch cannot activate without its exact contract marker/hash.
- V10/V10.1 dispatch and geometry remain behaviorally unchanged.
- collision and visible stem sizes are identical and equal to 12 mm square;
- parked controls disable every lobe/stem/crossbar collision geom;
- deliberate stem overlap is observed by both `data.contact` and the contact
  classifier;
- endpoint-only geometry, full-waypoint IK accounting, route-piece speeds, and
  qpos restoration pass on success/failure/exception paths;
- the fixed raised assembly passes the exact Step-0 expectations computed from
  source, not hard-coded result booleans.

Repair the review renderer for V10.2 only:

- use every policy frame (`frame_stride=1`);
- set output FPS to `1000 / 66 = 15.151515...`, matching the 66 ms policy
  timestep;
- burn in simulation time, wall-video time, segment name, commanded speed,
  realized TCP speed, current component clearance, and contact state;
- tint stems/crossbar in the review rerender for legibility only, with an
  explicit `REVIEW TINT` label; do not change the rollout or wrist/student
  observation;
- include a pendant-focused side view in the three-pane layout so the arm/stem
  depth relationship is visible;
- never interpolate or drop a terminal/contact frame.

Add a renderer timing test proving that 151 policy frames produce approximately
10 seconds of video, rather than the V10.1 1.32x playback.

## Step 2: six-row engineering screen

Use a new stream `pact-place-v10.2-raised-pendant-smoke`, master seed
`2026092000`. Generate exactly one row for each F0/F1/F2 family and side. Reuse
the V10.1 jitter distribution but not its task seeds.

Admission to the full review requires all `6/6` rows to satisfy:

- task success and the existing strict clean-success definition;
- zero pendant, panel, clutter, other-environment, or forbidden receptacle
  contact;
- zero clutter-stability event and zero sampling/infrastructure failure;
- no IK cascade, fallback, clipping, wrong-way route, endpoint mutation, or
  missing telemetry;
- complete sequential IK and correct route-piece speed caps;
- no live 2 ms physics contact with any lobe/stem/crossbar;
- at least 15 mm recorded robot/target-to-pendant clearance at every policy
  frame.

Render all six videos at true real time. If any row fails, stop and return the
named phase/geom/clearance/speed witness. Do not adjust height, stem thickness,
lane, padding, speed, clutter, or seeds within V10.2.

## Step 3: fresh 12-row review and causal check

Only after the six-row screen passes, use stream
`pact-place-v10.2-raised-pendant-human-review`, master seed `2026092002`.
Generate two paired repeats for each family and side: 12 rows total, six left
and six right. Render every row, including failures.

Automatic eligibility remains:

- all 12 reconcile, zero infrastructure failures;
- at least `10/12` strict clean successes;
- at least `1/2` clean in every family/side cell;
- every clean row passes the new stem-contact, 15 mm clearance, route-speed,
  endpoint, and telemetry requirements.

Then replay the lowest-role-index clean row in each cell using present,
present-repeat, and whole-assembly-parked worlds without `env.step`. Preserve
the V10.1 raw causal floors separately for inbound and outbound: at least 3
changed sensors, 448 changed values, a link5/link6 responder, and at most 4x
paired-side imbalance.

Stop for owner review. Emit no approval file and infer no approval. The owner
packet must link every video plus the review manifest, raw causal artifact,
named contact/clearance witnesses, and exact hashes.

Human review must explicitly confirm:

- the lowest lobe is visibly separated from table clutter;
- no arm/hand/target appears to cross either stem or the crossbar;
- the initial approach speed is acceptable and video playback is real time;
- route motion is smooth and takes the intended side;
- overlays agree with visible motion and contacts.

## Step 4: Phase 0 only after explicit approval

After an owner-supplied `approve_phase0` bound to the V10.2 contract, preflight,
six-row screen, review, and causal hashes, freeze a new stream
`pact-place-v10.2-raised-pendant-phase0`, master seed `2026092001`.

Run exactly 24 untouched rows: eight per F0/F1/F2 family and four per
family/side cell. Pass only with all rows reconciled, zero infrastructure
failures, at least `20/24` strict clean successes, and at least `3/4` clean in
every family/side cell. Every clean row must also meet the stem-contact,
clearance, speed, endpoint, and telemetry requirements above.

No terminal row may be replaced or reseeded. An interrupted nonterminal row may
resume only with its exact row identity. On fail, write a permanent stop. On
pass, write `eligible_for_separate_collection_authorization: true` while
leaving collection, training, and evaluation unauthorized.

## Required close-out

Run the V10.2 tests plus V10.1, V10 route-v2/runtime, V9.9, V9.8, V9.5,
mounted-fixture, and contact-audit regressions. Require `git diff --check`.
Update this plan, `README.md`, and `EVAL.md` with measured results and artifact
hashes, clearly distinguishing real observations from stop-rule consequences.

The coding agent must stop at the first failed stage. It must not use a failure
to choose another height, thicken or thin the stems again, change speed, search
another route, run Phase 0 early, or authorize downstream work.


## Measured status

The plan file above existed before any measurement. Everything in this section
is measured output. V10.2 stopped at Step 0; Steps 1–4 produced code and tests
but no episodes.

### Artifacts and hashes

| Item | Value |
| --- | --- |
| Contract | `pact_place_v102_raised_pendant_v1` |
| Contract SHA | `16f4c263d3b0310788b27e51303f0aa3feed0241e2c09ba254a69de25eb29a8b` |
| Implementation SHA | `c061bc50c4bd9a13c40250fdd081f0c0286a84e7e1ac619a0ba306b7d2f708e6` |
| Assembly id | `v102_raised_two_lobe_0.700000000--0.180000000-1.140000000-0.010000000-0.040000000-0.040000000_0.700000000-0.220000000-1.140000000-0.010000000-0.020000000-0.020000000` |
| Assembly self-SHA (also on every manifest row) | `0751a8d4850994e59f0486bd46f411018608ccb37105a8c0c03e03cbecccdb27` |
| Speed-schedule SHA | `b9c17c5022780d8820bfff57db17f2e6715aa0d10bc2a025c25c21ce0a1e7d32` |
| Preflight | `diagnostics_output/pact_place_v102_preflight/preflight.json`, artifact SHA `6c5079916775e8a2093defb1547a3fa85ef9b32dcc4fddcf785ffa6c3276976d` |
| Contact-parity root cause | `diagnostics_output/pact_place_v102_preflight/contact_parity_root_cause.json`, artifact SHA `e4e544a999534b322d177d8b296aa4c7580d9b7627bf89bf0d42630fdd0774df` |
| Six-row screen | **not run** |
| 12-row review and gallery | **not run** |
| Causal replay | **not run** |
| Owner `approve_phase0` | **absent, not inferred** |
| 24-row Phase 0 | **not run** |

`preflight_passed: false`. `stop: true`. `authorizes_gate: false`.
`authorizes_collection: false`. `authorizes_training: false`.
`authorizes_evaluation: false`.
`eligible_for_separate_collection_authorization: false`.

### Step-0 item results

| Item | Result |
| --- | --- |
| 1 protected artifacts and scene hashes | **pass** — 17 protected V9.5/V9.9/V10/V10.1 artifacts matched byte-for-byte; V10 scene SHA `360b1407…eaddf7` unchanged. No old artifact was rewritten with the new V10.2 hash. |
| 2 raised assembly vs aperture, hood, active panel, clutter, initial target, initial robot | **pass** — panel-clear, clutter-clear, static-clear and initial-state-clear on all 6 frozen cells, every active component. Derived facts match the registered table exactly: lowest pendant point `1.10 m`, shelf-to-pendant gap `0.38 m`, stem centre y `−0.22 / +0.24 m`, stem tops `1.505 m`, crossbar top `1.515 m`, stem and crossbar x thickness `0.012 m`. |
| 3 exact stock-route necessity | **pass** — **12/12** cell×direction cases obstructed. Left cells are blocked by `lobe_0` (and `stem_0`); right cells by `lobe_1` (and `stem_1` outbound). Raising the pendant did not make it decorative. |
| 4 fixed endpoint-only route geometry | **pass** — **12/12** admitted: correct side, no clipping, no wrong-way travel, endpoints preserved, densification within limits. Inbound detour 0.103–0.120 m, outbound 0.059–0.071 m, all ≥ the registered 0.05 m. |
| 5 complete sequential IK on the fixed nominal route | **FAIL** — 5/12 cases complete. All five are inbound; every outbound case is incomplete, and F1-left inbound is 163/165. |
| 6 per-component robot/target-to-pendant clearance ≥ 15 mm | **FAIL** — **0/12** cases meet the floor. Every case is *negative*: the arm interpenetrates the raised pendant. |
| 7 deliberate stem contact-parity fixture | **FAIL** — with the stem posed to genuinely penetrate a retained robot collision geom, `data.contact` reports nothing. Root cause below. |

### Item 5 — sequential IK

> **Erratum (recorded under V10.3, immutable payload unchanged).** An earlier
> version of this prose said item 5 completed **4/12** cases. The correct count
> read from the immutable
> `diagnostics_output/pact_place_v102_preflight/preflight.json` is **5/12**:
> five inbound cases complete (F0-left, F0-right, F1-right, F2-left, F2-right),
> F1-left inbound is 163/165, and all six outbound cases are incomplete. The
> preflight artifact and its SHA are unchanged; only the prose was wrong.

`waypoints_solved / waypoints_attempted`, no environment gate, no early abort:

| family | side | inbound | outbound |
| --- | --- | ---: | ---: |
| F0 | left | 164/164 ✓ | 342/436 |
| F0 | right | 162/162 ✓ | 259/293 |
| F1 | left | 163/165 | 338/445 |
| F1 | right | 167/167 ✓ | 235/293 |
| F2 | left | 169/169 ✓ | 336/443 |
| F2 | right | 167/167 ✓ | 235/296 |

### Item 6 — the decisive result

Minimum clearance along the nominal retained-qpos route, per component, floor
`0.015 m`. Negative values are penetration depth.

| family | side | direction | worst component | min clearance (m) | second worst |
| --- | --- | --- | --- | ---: | --- |
| F0 | left | inbound | `lobe_0` | **−0.0676** | `stem_0` −0.0461 |
| F0 | left | outbound | `lobe_0` | **−0.0761** | `stem_0` −0.0488 |
| F0 | right | inbound | `lobe_1` | **−0.0651** | `stem_1` −0.0518 |
| F0 | right | outbound | `lobe_1` | **−0.0561** | `stem_1` −0.0515 |
| F1 | left | inbound | `lobe_0` | **−0.0769** | `stem_0` −0.0463 |
| F1 | left | outbound | `lobe_0` | **−0.0763** | `stem_0` −0.0359 |
| F1 | right | inbound | `lobe_1` | **−0.0665** | `stem_1` −0.0512 |
| F1 | right | outbound | `lobe_1` | **−0.0564** | `stem_1` −0.0442 |
| F2 | left | inbound | `lobe_0` | **−0.0748** | `stem_0` −0.0455 |
| F2 | left | outbound | `lobe_0` | **−0.0763** | `stem_0` −0.0262 |
| F2 | right | inbound | `lobe_1` | **−0.0666** | `stem_1` −0.0519 |
| F2 | right | outbound | `lobe_1` | **−0.0565** | `stem_1` −0.0391 |

The crossbar is clear everywhere (0.266–0.279 m), and the opposite-side lobe and
stem are clear on every case. What fails is always the same-side lobe and its
stem, in both directions, on all three families.

Read plainly: raising the lobe bottoms from `0.82/0.84 m` to `1.10 m` moved the
pendant out of the table-clutter band and into the arm's own elbow/forearm
envelope. The registered route is a **TCP** lane rewrite with the grasp-side
endpoint frozen; it has no authority over where link5/link6 sit while the TCP
tracks that lane. The V10 lattice cap `LOBE_TOP_MAX_M = 1.10` is exactly the
boundary this design crosses. This is a measured property of the registered
V10.2 height, not a stop-rule consequence.

The observation that motivated V10.2 — "the arm appears to pass through the
pendant stem" — is therefore **confirmed as literal geometry for the raised
design**, at up to 77 mm of interpenetration.

### Item 7 — contact-parity root cause, verified separately

The Step-0 fixture poses each collision stem across
`robot_0/fr3_link5_collision` and requires a live `mounted_fixture` contact. On
all six cells it observed hardened-GJK penetration of −53.8 mm and **zero**
`data.contact` entries, while the same scene simultaneously carried 83 other
contacts.

`scripts/diagnose_pact_place_v102_contact_parity.py` isolates the cause and
records it in `contact_parity_root_cause.json`:

- 19 deliberately penetrating stem poses (GJK −69 mm to −17 mm, three axes),
  **0** of them seen by `data.contact`;
- the pair is collision-compatible (`stem contype 8 / conaffinity 15`,
  `link5 contype 1 / conaffinity 1`), so the contype filter is not the cause;
- `pact_place_v10_scene.pose_assembly_geoms` writes `model.geom_pos` and
  `model.geom_size` at runtime but leaves `model.geom_aabb`
  (`[0,0,0, 0.001,0.001,0.001]`), `model.geom_rbound` (`0.001732`) and the
  pendant body's `model.bvh_aabb` rows (`[0,0,0, 0.001,0.001,0.001]`) at their
  compile-time 1 mm placeholder values, while the posed stem is
  `0.006 × 0.006 × 0.1625 m`;
- control: refreshing only those broadphase bounds, at the same pose, makes the
  contact appear immediately — `pact_clutter_mount_v10_stem_0_g` vs
  `robot_0/fr3_link5_collision` at `dist = −0.016630 m`, matching the GJK
  distance to six decimals.

MuJoCo's broadphase therefore never proposes a pendant/robot pair for the
runtime-posed V10 pendant.

**Consequence for earlier packs.** Recorded "zero `mounted_fixture` contact" for
the runtime-posed V10 pendant — including the V10.1 review rows — is not
evidence of clearance. It is what this posing path produces regardless of
geometry. This is exactly the concern the plan raised: *zero recorded
mounted-fixture contacts is not enough to establish visible or swept clearance.*
No V10/V10.1 artifact has been altered, and their recorded values remain what
they were.

Repairing `pose_assembly_geoms` would change V10 and V10.1 runtime behaviour,
which this plan forbids, and is a geometry/cleanliness change that would in any
case require a new version. It was **not** done. It is flagged for the owner.

### What was built but not run

Step 1 (runtime and renderer verification) is complete and passing:
`tests/test_pact_place_v102_raised_pendant.py`, **41 tests, all passing**,
covering V10.2 dispatch refusal without the exact marker and speed-schedule
hash, unchanged V10/V10.1 dispatch and geometry, identical 12 mm collision and
visible stem geometry with no visual-only sleeve, parked control disabling every
lobe/stem/crossbar collision geom, deliberate stem overlap observed by both
`data.contact` and the contact classifier in an articulated fixture,
endpoint-only geometry, full-waypoint IK accounting on success/failure/exception
paths with qpos restoration, route-piece speed assignment and caps, row
admission, and renderer timing.

The repaired V10.2 review renderer (`scripts/run_pact_place_v102_review_video.py`)
uses `frame_stride = 1` and `1000/66 = 15.1515… fps`, so 151 policy frames are
9.966 s of video — real time, against V10.1's measured `0.132 / 0.100 = 1.32×`.
It burns in simulation time, wall-video time, segment name, commanded speed,
realized TCP speed, per-component clearance and contact state, tints
stems/crossbar with an explicit `REVIEW TINT` label while leaving the wrist
(student) pane untinted, and adds a pendant-focused side view as the third pane.
No V10.2 video was rendered, because no V10.2 episode was generated.

`_v10_apply_lane` no longer copies `segments[0].speed` onto every rebuilt piece.
Under the exact V10.2 marker and speed-schedule hash it assigns `0.15 m/s` to
`inbound_pendant_approach`, `0.045 m/s` to every `_pass`/`_exit` piece,
`0.08 m/s` to `inbound_pendant_rejoin`, and leaves outbound non-pendant
transport at its inherited speed; a cap violation raises. Without those markers
the historical single-speed behaviour is unchanged, which the tests assert.

### Owner-requested diagnostic gallery — outside the gate

After this stop the owner asked to see the twelve clips. They were generated by
`scripts/run_pact_place_v102_diagnostic_gallery.py` into
`diagnostics_output/pact_place_v102_diagnostic_gallery/` (manifest SHA
`9d0f7b4b6c8adc51261cf58bbed26a12ec68b5b62bfd1c446ac122c94918c0a1`), with
`role: owner_requested_diagnostic_gallery_not_the_registered_review`,
`is_registered_review: false`, `eligible_for_human_review: false` and every
authorization false. The registered review path
`diagnostics_output/pact_place_v102_review/` was **not** written, the Step-0
stop artifact is untouched, and the V10.2 implementation hash is unchanged
because the gallery runner is not one of its files. This gallery is not
eligible for human review, is not an input to causal proximity or Phase 0, and
must never be cited as a review.

Measured on those twelve live rollouts: 12/12 reconciled and complete, **0/12**
clean, 11 `terminal_ik_cascade` and one `clutter_collision_contact`. Minimum
per-frame pendant clearance is negative on every row (−0.0008 m to −0.0742 m
against the 15 mm floor), with **zero** live pendant-contact frames and **zero**
`mounted_fixture` contact entries across all twelve. That is items 6 and 7
reproduced together on live episodes: the arm passes through the raised pendant,
and the contact pipeline cannot see it. Route-piece speeds appear as registered
— `inbound_pendant_approach` 0.15 m/s, `inbound_pendant_pass` 0.045 m/s, against
an inherited 0.20 m/s.

The gallery also found a defect in the registered renderer: its pendant pane is
aimed from `y = -1.15 m`, outside the hood, where `hood_side_r` occludes it. The
gallery overrides that camera in-process and records the override in its
manifest under `pendant_side_camera_override`. A successor version should adopt
the corrected pose rather than the registered one.

### Stop

V10.2 stops here. No height, stem thickness, lane, padding, speed, clutter, or
seed was changed in response to these results, no alternative route was
searched, Phase 0 was not run, and no downstream work is authorized.
