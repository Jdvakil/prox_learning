# PACT place V10.3: static-pendant joint-route qualification through Phase 0

**Status:** **stopped at Step 0B under an owner-approved early stop.** No static
geometry in the registered height lattice admits a complete joint route on all
six cells in both directions. No episode was generated and no runtime was built.
The measured close-out, including the early-stop amendment and the deviations,
is in [Measured status](#measured-status) at the end of this file.

Preregistered implementation and qualification plan for a new coding
agent. V10.3 starts unauthorized. This document permits only the named offline
search, tests, smoke/review generation, causal replay, explicit owner review,
and Phase 0 gate in the order below. It never authorizes collection, training,
or downstream evaluation.

## Objective

Produce a physically valid PACT place environment in which:

1. the two-lobe ceiling pendant is completely static and visually separated
   from table clutter;
2. every pendant component is collision-enabled, and any robot or carried-target
   contact with it is a hard row failure;
3. the robot follows a continuous joint-space route from the actual sampled
   state, without teleporting between IK branches;
4. the arm and carried target retain exact clearance from the pendant and all
   other environment geometry throughout the route;
5. the resulting environment passes the smoke pack, human review, causal
   proximity gate, and the existing 24-row Phase 0 gate.

The task is to find a valid route and height under frozen rules. It is not to
make the gate pass by weakening contact, clearance, cleanliness, detour,
tracking, causal, or success definitions.

## Historical boundary and required errata

- Preserve every V9.5, V9.8, V9.9, V10, V10.1, and V10.2 artifact byte-for-byte.
- V10.2 remains stopped at Step 0. Do not overwrite
  `diagnostics_output/pact_place_v102_preflight/`.
- Record an erratum, without changing the immutable preflight payload: the
  V10.2 item-5 case table contains **5/12** complete routes, not 4/12. Five
  inbound cases complete, F1-left inbound is 163/165, and all six outbound
  cases are incomplete. Correct the prose in
  `docs/PACT_PLACE_V102_RAISED_PENDANT_REMEDIATION_PLAN.md`, `README.md`, and
  `EVAL.md`.
- Preserve the V10.2 contact-parity conclusion. Runtime resizing of the V10
  placeholder geoms left stale MuJoCo broad-phase bounds. Consequently,
  historical zero `mounted_fixture` contacts are not clearance evidence.
- Preserve the slower V10.2 speed schedule and real-time video settings as the
  starting values for V10.3. They were tested but never exercised in an
  episode.

All V10.3 artifacts must begin with:

```text
authorizes_gate: false
authorizes_collection: false
authorizes_training: false
authorizes_evaluation: false
phase0_passed: false
```

## Non-negotiable physical contract

### Static means fixed and collidable

- The pendant has no joint, freejoint, mocap motion, compliance, or swinging
  dynamics.
- Compile the selected lobe, stem, and crossbar positions and sizes into a new
  `pact_place_corridor_v10_3.xml` before creating `MjModel`.
- Do not call `pose_assembly_geoms`, write `model.geom_pos`, write
  `model.geom_size`, or refresh broad-phase arrays at episode runtime.
- Every active pendant geom uses the registered mounted-fixture collision masks.
- Robot-pendant and carried-target-pendant contact is always a hard failure.
  Do not allow, filter, suppress, or relabel it.
- Exact distance is the admission predicate. Live `data.contact` is an
  additional parity and runtime-failure signal, not a substitute for distance.

### Fixed shape outside the height sweep

Retain the V10.2 two-lobe shape and x/y placement:

| component | x/y center (m) | half extents except z (m) |
| --- | --- | --- |
| negative lobe | `[0.70, -0.18]` | `[0.01, 0.04]` |
| positive lobe | `[0.70, +0.22]` | `[0.01, 0.02]` |

- Negative-lobe half-z remains `0.04 m`; positive-lobe half-z remains `0.02 m`.
- Both stems and their visible/collision geometry remain a 12 mm square.
- The crossbar x thickness remains 12 mm, its top remains flush with
  `hood_top` at `z=1.515 m`, and stems connect each lobe to it.
- No visual-only sleeve is allowed. Visible and collision geometry must match.
- Lobe x/y, widths, depths, asymmetry, stem y, aperture, hood, panels, clutter
  layouts, and F0/F1/F2 cell definitions are not search variables.

## Registered V10.3 search space

Search this lattice exactly once. Do not add values after observing results.

### Geometry height lattice

`lowest_lobe_bottom_z_m` is one of:

```text
0.92, 0.96, 1.00, 1.04
```

The negative lobe center-z is `lowest_lobe_bottom_z_m + 0.04`; both lobe centers
share that z. Thus the positive lobe bottom remains 20 mm higher. The shelf top
is `0.72 m`, so every candidate has at least 200 mm visible vertical separation
from the shelf. Do not include the old `0.82 m` or failed V10.2 `1.10 m` bottom.

### Joint-route template lattice

For each F0/F1/F2 side cell and direction, enumerate:

| parameter | registered values |
| --- | --- |
| lane magnitude | `0.28, 0.30, 0.32 m` |
| slab staging buffer | `0.10, 0.12 m` from the physical x faces |
| pass TCP z offset from the stock interpolation | `-0.06, -0.04, -0.02, 0.00 m` |
| left pass orientation | identity, local `Rx(-5°)`, `Rx(-10°)`, `Ry(+5°)`, `Ry(+10°)` |
| right pass orientation | identity, local `Rx(+5°)`, `Rx(+10°)`, `Ry(-5°)`, `Ry(-10°)` |

Lane sign is negative for left and positive for right. Orientation offsets are
relative to the retained stock orientation at the physical slab center. Use
SLERP for diagnostic Cartesian construction; the executed path is the selected
joint trajectory described below.

The pendant physical x faces are `0.69/0.71 m`. For a buffer `b`, define:

```text
near_staging_x = 0.69 - b
far_staging_x  = 0.71 + b
```

### Compact control-pose topology

Do not rewrite every realized V9.5 TCP frame. Build only these named control
poses, with exact retained endpoint poses and qpos:

Inbound:

```text
actual_initial
near_stock_staging
near_lane_staging
far_lane_staging
far_pregrasp_staging
actual_pregrasp_endpoint
```

Outbound:

```text
actual_loaded_lift
far_stock_staging
far_lane_staging
near_lane_staging
near_exit_staging
actual_outbound_endpoint
```

This topology deliberately performs lateral blending outside the physical
pendant slab. Outbound may first retreat in +x from the loaded lift to the
registered far staging pose; that named retreat is allowed and must not be
misclassified as an unregistered wrong-way route. No other x reversal is
allowed.

The route must cross the physical slab on the correct signed lane and have at
least 50 mm TCP detour from the retained stock path at matched x. Densifying a
large endpoint jump is not an admission criterion.

## Deterministic multi-branch joint planner

Implement an additive V10.3 planner. Do not alter historical V9.9/V10/V10.1/
V10.2 route functions.

### Directional endpoints and seeds

- Inbound starts from the retained/sampled initial qpos.
- Outbound starts from the retained/sampled loaded-lift qpos, including the
  carried target pose. Never seed outbound from the episode's first qpos.
- Pin the first and final graph nodes to their actual qpos. An alternative IK
  solution at the same TCP pose is not an endpoint unless a collision-free
  joint-space edge connects the actual qpos to it.
- Store source step indices, full qpos hashes, endpoint TCP residuals, and
  whether the target is included.

### IK candidate generation

For every intermediate control pose, generate candidates from this deterministic
seed set:

1. actual directional start qpos;
2. actual directional end qpos;
3. joint-center qpos;
4. all valid candidates in the preceding graph layer;
5. all valid candidates in the following layer during the reverse pass;
6. 24 fixed Halton joint-space seeds, scaled to the inner 80% of each finite
   joint range, with a contract-fixed Halton scramble seed.

Use the existing MuJoCo IK as the first implementation. Do not change its
global defaults to manufacture a pass. Deduplicate solutions whose seven arm
joints are within `0.02 rad` L-infinity distance.

Each candidate node must satisfy:

- position residual <= `1 mm`;
- orientation residual <= `1 degree`;
- all finite joint limits;
- no robot self-collision;
- no robot/target contact with ordinary environment geometry;
- nominal robot/target-to-every-pendant-component distance >= `0.020 m`.

Outbound tests include robot plus the carried target. Inbound tests include the
robot. Missing or non-finite distance fails closed.

### Graph edges and execution identity

Connect adjacent layers only. Interpolate every candidate edge in joint space
with maximum seven-arm-joint increment `0.01 rad`; call `mj_forward` at every
sample and evaluate:

- robot self-collision;
- strict ordinary-environment nonintersection;
- robot/target clearance to each lobe, stem, and crossbar;
- actual FK/TCP path, signed slab detour, x ordering, and aperture containment;
- joint-limit margin and maximum TCP translation/rotation between samples.

An edge passes only if every interpolation sample has pendant clearance
`>=0.020 m`. Also score eight deterministic `+/-5 mm` perturbation corners of
the registered lane/staging/pass-z parameters. Every corner must remain
`>=0.015 m`, collision-free, and connected.

Select a path through the graph by this fixed ordering:

1. maximize minimum robust pendant clearance;
2. maximize minimum joint-limit margin;
3. minimize total seven-arm-joint travel;
4. minimize total pass orientation deviation;
5. lexicographic route key.

The joint trajectory selected offline is the trajectory executed. Add a
V10.3-specific `JointMoveSequence`/`JointMoveSegment` dispatch; do not convert
the selected path back into TCP segments and rerun single-seed IK at runtime.
Hash every qpos waypoint. Runtime must fail closed if the hash, start qpos,
geometry hash, cell key, direction, or target-held state does not match.

For each joint segment, set duration to the maximum of:

- FK/TCP arc length divided by the inherited V10.2 piece speed; and
- every joint displacement divided by 50% of that joint's velocity limit.

Keep the V10.2 speed caps:

```text
empty clear-space approach: 0.15 m/s
pendant pass/exit:           0.045 m/s
pregrasp approach:           0.08 m/s
post-pick non-pendant:       inherited historical speed
```

## Step 0A — tests before search

Add behavioral tests that prove:

- the V10.2 immutable item-5 table counts 5 complete cases;
- historical route functions and artifacts are unchanged;
- every V10.3 pendant geom is compiled at its final pose/size and the body is
  fixed;
- visible and collision dimensions match exactly;
- clear, exactly touching, and penetrating robot/stem fixtures agree between
  hardened distance and MuJoCo `data.contact`;
- a parked/disabled V10.3 pendant is impossible in an admitted row;
- outbound uses the loaded-lift qpos and inbound uses the initial qpos;
- endpoint qpos cannot be replaced by an unconnected IK branch;
- lateral staging occurs outside the physical slab;
- role-600's frozen V10.2 pose remains a known negative control;
- role-601's reverse/multiseed branch is recovered as a positive IK control;
- node deduplication, graph ranking, edge interpolation, target inclusion,
  exact clearance, robust corners, and fail-closed missing distance are real
  behavioral tests;
- runtime executes the selected joint qpos sequence without invoking TCP IK;
- speed-derived duration caps and real-time review timing are enforced.

Run the V10.3 tests plus all V10.2/V10.1/V10/V9.9/V9.8/V9.5,
mounted-fixture, contact-audit, and failure-cause regressions before search.
Stop on any new failure. Record unrelated pre-existing failures separately; do
not silently exclude a newly failing test.

## Step 0B — cached offline search

This phase may call model compilation, `mj_forward`, FK, IK, and exact distance.
It must not call `env.step`, render observations, or generate episodes.

1. Verify all protected hashes and all six reconstruction NPZ hashes.
2. Cache IK by `(cell, direction, control_pose_key, seed_key)`; geometry height
   must not cause duplicate IK work.
3. Evaluate every registered route template. Do not stop at the first survivor.
4. Score every valid qpos/edge against all four compiled height candidates.
5. A geometry survives only if at least one complete route exists for all six
   cells and both directions.
6. Route templates may differ by cell and direction, but the selected mapping
   is frozen before any episode.

Choose the single geometry by:

1. maximize its worst-case robust clearance across the twelve selected paths;
2. maximize lowest-lobe bottom z;
3. maximize worst joint-limit margin;
4. minimize total joint travel across the twelve paths;
5. lexicographic geometry key.

Write immutable artifacts under:

```text
diagnostics_output/pact_place_v103_ik_search/search.json
diagnostics_output/pact_place_v103_ik_search/nodes.npz
diagnostics_output/pact_place_v103_ik_search/edges.npz
diagnostics_output/pact_place_v103_ik_search/selected_routes.npz
```

Include complete attempted/passed/rejected counts, per-predicate rejection
counts, full IK failure indices, pose residuals, seed provenance, joint margins,
minimum-clearance witnesses, actual contacting geoms, and artifact SHA-256.

Stop reasons:

- `no_static_geometry_with_twelve_joint_routes`
- `no_route_with_nominal_clearance`
- `no_route_with_robust_clearance`
- `contact_parity_failed`
- `search_input_hash_mismatch`

If the search is empty, V10.3 stops permanently. Do not add a height, lane,
orientation, staging, seed, or clearance value and rerun under V10.3.

## Step 0C — selected-scene preflight

If and only if Step 0B selects one geometry and twelve routes:

1. serialize the selected static geometry into
   `pact_place_corridor_v10_3.xml`;
2. hash the XML, contract, route mapping, qpos catalog, implementation, and
   protected inputs;
3. compile a fresh model and rerun clear/touch/penetrating contact parity;
4. rerun all twelve joint paths from their actual directional qpos;
5. require nominal clearance `>=20 mm` and all robust corners `>=15 mm`;
6. require exact endpoint equality, 50 mm detour, strict environment
   nonintersection, and no self-collision on every interpolated sample;
7. verify the pendant cannot move under repeated `mj_forward` calls.

Write:

```text
diagnostics_output/pact_place_v103_preflight/preflight.json
```

No episode is authorized unless every check is 12/12 and contact parity passes.

## Step 1 — six-row smoke

Use a new immutable manifest:

```text
stream: pact-place-v10.3-static-pendant-joint-route-smoke
master seed: 2026103000
rows: 6
```

Run exactly one F0/F1/F2 left/right row. No row substitution and no geometry,
route, IK seed, or threshold tuning after the first row starts.

Require **6/6 strict clean successes** and, on every row:

- no pendant contact or clearance below 15 mm;
- no ordinary-environment, self, hazard-bar, or clutter contact;
- no clutter stability event;
- no planning, IK, graph, route-hash, start-state, tracking, grasp, or place
  failure;
- no fallback to TCP IK or historical endpoint-only routing;
- all commanded/realized speed caps pass;
- task success, grasp success, place success, and terminal clean state.

Any failure stops V10.3 before review. Do not repair and continue under the same
manifest.

## Step 2 — twelve-row review and causal qualification

If smoke passes, use:

```text
stream: pact-place-v10.3-static-pendant-joint-route-human-review
master seed: 2026103002
rows: 12 (two paired repeats per F0/F1/F2 side cell)
```

Eligibility requires:

- at least 10/12 strict clean successes;
- at least 1/2 strict clean successes in every cell;
- zero pendant contacts on all 12 rows, including failed task rows;
- no missing trajectory, telemetry, qpos hash, clearance, video, or contact
  evidence.

Render stride-1 videos at `1000/66 fps` with simulation time, video time,
commanded and realized speed, joint-route segment, minimum per-component
clearance, and contact class. Keep the untinted wrist pane, explicit `REVIEW
TINT` label, and pendant side view introduced for V10.2.

Run the frozen-qpos causal comparison only on the lowest-index strict-clean row
from each cell. Use the existing V10.2/V10.1 raw proximity floors unchanged:

- the imported `ADMISSION_FLOOR` sensor/value thresholds;
- a responding corridor link;
- inbound and outbound windows separately;
- left/right changed-value ratio <= the imported historical maximum.

The causal replay must use the compiled static V10.3 scene and must not call
`env.step`. A missing clean cell or silent side/window fails closed and blocks
human review.

Write immutable review and causal artifacts under:

```text
diagnostics_output/pact_place_v103_review/
diagnostics_output/pact_place_v103_causal/
```

Then stop for owner review. The owner must confirm:

1. pendant lobes are visibly separated from table clutter;
2. stems and lobes remain static;
3. no robot link or carried target passes through or touches any pendant part;
4. motion follows the intended side smoothly without branch jumps;
5. the initial approach speed is acceptable and the video is real-time;
6. overlays agree with visible motion and clearance.

Approval must be explicit in `human_approval.json` with
`decision: approve_phase0` and must bind the preflight, review-manifest, causal,
selected-scene, selected-route, and implementation hashes. Absence, ambiguity,
or a hash mismatch is rejection.

## Step 3 — 24-row Phase 0

Only a valid owner approval authorizes this step.

```text
stream: pact-place-v10.3-static-pendant-joint-route-phase0
master seed: 2026103001
rows: 24 (four repeats per F0/F1/F2 side cell)
```

Freeze the approved XML, geometry, route template mapping, joint planner,
seed set, collision logic, speed schedule, renderer-independent policy code,
manifest, and thresholds. Per-row joint planning may use the actual sampled
start/end states, but only the approved template and deterministic seed set.

Phase 0 passes only with:

- at least 20/24 strict clean successes;
- at least 3/4 strict clean successes in every cell;
- zero pendant contacts across all rows;
- no row whose minimum exact pendant clearance is below 15 mm;
- no fallback, route-hash mismatch, unconnected branch, or missing telemetry;
- the existing strict clutter/contact/grasp/place cleanliness contract.

Sampling or planning failure counts as a failed row. Do not replace it with a
new seed. A row that touches the static pendant is a failure even if the task
otherwise succeeds.

Write:

```text
diagnostics_output/pact_place_v103_phase0/gate.json
```

Even a passing Phase 0 sets only `phase0_passed: true`. Keep
`authorizes_collection`, `authorizes_training`, and `authorizes_evaluation`
false. A separate owner decision and plan are required for anything downstream.

## Required implementation surfaces

Prefer additive V10.3 modules and runners, for example:

```text
scripts/pact_place_v103_contract.py
scripts/pact_place_v103_geometry.py
scripts/pact_place_v103_joint_route.py
scripts/search_pact_place_v103_joint_route.py
scripts/run_pact_place_v103_preflight.py
scripts/run_pact_place_v103_screen.py
scripts/run_pact_place_v103_review.py
scripts/run_pact_place_v103_causal.py
scripts/run_pact_place_v103_phase0.py
tests/test_pact_place_v103_static_pendant.py
```

Add a uniquely named sampler/environment marker such as
`pact_place_corridor_v10_3_static_pendant_joint_route`. Historical rows must
never dispatch into V10.3 joint routing.

## Documentation and final report

Update this plan, `README.md`, and `EVAL.md` after each completed stop boundary.
Do not describe a broad-phase overlap, IK success, or node success as a valid
route. Report:

- exact tests and regressions;
- every artifact path and SHA-256;
- protected hashes;
- full search lattice and counts;
- selected geometry and route mapping, if any;
- per-cell/direction IK, edge, clearance, and rejection witnesses;
- smoke/review/causal/Phase-0 counts actually reached;
- all authorization fields;
- confirmation that the pendant was static and any contact was counted as a
  failure;
- confirmation that no collection, training, or evaluation occurred.

Run `git diff --check` at every boundary. Stop at the first failed gate and
return the evidence; do not continue merely because later runners exist.


## Measured status

Everything below is measured output. The plan text above is unchanged.

**Stopped at Step 0B under an owner-approved early stop.** Step 0C, the six-row
smoke, the twelve-row review, the causal replay, and the 24-row Phase 0 were
**not run**. No episode was generated, `env.step` was never called, no V10.3
episode runtime was built, and collection, training, and evaluation never became
authorized.

### Artifacts and hashes

| Item | Value |
| --- | --- |
| Contract | `pact_place_v103_static_pendant_joint_route_v1` |
| Search stop record | `diagnostics_output/pact_place_v103_ik_search/search.json`, artifact SHA `f06feaa3c09d5f95a006f66d00e45c8684962393967a1acc3ad40d21dc23df98` |
| Endpoint certificate | `diagnostics_output/pact_place_v103_ik_search/endpoint_certificate.json`, artifact SHA `3ced3a35b71ac7a1cc9f94ab23549b0764dceef07508ab5302791e592c062fda` |
| Preserved worker stdout | `diagnostics_output/pact_place_v103_ik_search/search_worker_stdout.log` |
| `nodes.npz` / `edges.npz` / `selected_routes.npz` | **not written** — see the early-stop amendment |
| Selected geometry | none |
| Selected routes | none |
| Owner `approve_phase0` | absent, not sought |

`search_passed: false`. `search_exhaustive: false`.
`stop_reason: no_static_geometry_with_twelve_joint_routes`.
`global_conclusion_conclusive: true`.
`conclusive_witness: pinned_endpoint_clearance_below_node_floor`.
`remaining_cases_cannot_change_selection: true`.
`authorizes_gate/collection/training/evaluation: false`. `phase0_passed: false`.

### Step 0A — tests before search

`tests/test_pact_place_v103_static_pendant.py`: **34 tests, all passing**. They
recompute expectations from source, from live MuJoCo state, or from the
immutable V10.2 payload; none asserts a stored result boolean. Covered: the
V10.2 item-5 errata read from the immutable artifact, unchanged historical route
dispatch and V10.1/V10.2 geometry, the registered height lattice and its
exclusions, V10.2 shape inheritance with only z moving, the 20 mm positive-lobe
offset, stem/crossbar attachment, forbidden static overlap, a compiled static
scene whose pendant body carries no joint/freejoint/mocap and whose geoms cannot
move under repeated `mj_forward`, visible-equals-collision sizing with
`contype=8 conaffinity=15`, mounted-fixture classification, a clear/touching/
penetrating contact-parity fixture where hardened distance and `data.contact`
agree, the 120-template lattice per side with lateral staging outside the
physical slab, the eight ±5 mm corners, control-pose topology, stock-pose
interpolation clamping, fixed Halton seeds, L-infinity dedup, graph ranking,
speed classes and duration caps, and order-sensitive qpos hashing.

Regression sweep before search: **302 passed** across V10.3, V10.2, V10.1, V10
(compound / route-v2 / route-runtime / siting-v2), V9.9, V9.8 (pendant and
offset contact diagnosis), V9.5, V9.4 mounted preview, V9 redesign, place
corridor, failure cause, collection, and contact endpoint. Three **pre-existing**
failures in `tests/test_pact_place_corridor.py`
(`test_v3_xml_only_adds_named_clutter_bodies`,
`test_v3_xml_hash_is_unchanged_and_v4_pool_is_sixteen_bodies`,
`test_v8_contract_freezes_real_movable_clutter_without_authorizing_gate`) are
stale v3/v5 scene-hash assertions against unmodified committed scene XML. They
predate V10.3 and are recorded here rather than excluded. `git diff --check` is
clean in the superproject and the submodule.

### Step 0B — cached offline search, stopped early

Registered lattice, searched once and never extended: heights
`0.92 / 0.96 / 1.00 / 1.04 m`; lanes `0.28 / 0.30 / 0.32 m`; staging buffers
`0.10 / 0.12 m`; pass-z offsets `-0.06 / -0.04 / -0.02 / 0.00 m`; five pass
orientations per side; 24 fixed scrambled-Halton joint seeds plus the three
registered fixed seeds and both adjacent layers; node and edge floors
`0.020 m`; corner floor `0.015 m`; interpolation step `0.01 rad`.

**Nine of twelve cell/direction cases completed, all with zero feasible routes
at every height:**

| case | templates evaluated | templates with complete layers | feasible routes |
| --- | ---: | ---: | ---: |
| F0/F1/F2 left, inbound (×3) | 0 | 0 | 0 at all four heights |
| F0/F1/F2 left+right, outbound (×6) | 120 each | **0/120** each | 0 at all four heights |
| F0/F1/F2 right, inbound (×3) | **not evaluated** | — | — |

The three left-inbound cases returned in ~55 s each because every height was
excluded at the pinned endpoint before any template work. The six outbound cases
each evaluated all 120 registered templates in 22–26 min and found no template
whose control-pose layers were all non-empty. The three right-inbound cases were
still running after 4 h 42 m when the owner approved the early stop.

### The conclusive witness

`endpoint_certificate.json` records, for all three left-inbound cells × all four
registered heights × two retained frames (the pinned pregrasp endpoint and the
last retained inbound frame), 24 measurements. **Every one is a penetration and
every one is below the floor.** Both instruments agree: analytic exact GJK gives
`0.00000 m` (intersecting) and hardened signed `mj_geomDistance` gives a negative
distance. The node-clearance margin is `-0.02000 m` in all 24.

The binding component is always `lobe_0`, the negative lobe. The contacting
robot part changes with height:

| height (m) | F0 left | F1 left | F2 left | contacting robot geom |
| ---: | ---: | ---: | ---: | --- |
| 0.92 | -0.00829 | -0.00043 | -0.00220 | `robot_0/gripper/base` |
| 0.96 | -0.01431 | -0.00575 | -0.00735 | `robot_0/gripper/base` |
| 1.00 | -0.02282 | -0.01478 | -0.01633 | `robot_0/fr3_link7_collision` |
| 1.04 | -0.02445 | -0.01996 | -0.02073 | `robot_0/fr3_link6_collision` |

Signed penetration depth at the pinned pregrasp endpoint. The last retained
inbound frame is the same or worse throughout. The probe's scene state hash is
identical before and after every measurement, so the offline probe restored the
model exactly.

Read plainly: the arm occupies the negative lobe's frozen x/y column
continuously from the gripper base up through link6. A low lobe is struck by the
hand; a high lobe is struck by the wrist. There is no z window in the registered
lattice that threads between them, and lobe x/y, widths, depths, asymmetry, and
stem y are explicitly not search variables. The V10.2 finding therefore survives
being moved 6–18 cm downward: it was never specific to the 1.10 m height.

Because the endpoint is pinned to the retained qpos, and is the first
interpolation sample of every edge leaving it, no route template can rescue those
cells. Three cells with zero admissible heights means no height can route all
twelve cases, whatever the three unevaluated right-inbound cases would have
returned.

### Early-stop amendment

The owner approved stopping Step 0B before it evaluated every registered
template, on the ground that the conclusion was already global. This document
does **not** claim an exhaustive search, and neither does the artifact:
`search_exhaustive: false` and `every_registered_template_evaluated: false`.

Consequences recorded rather than papered over:

- **`cases_completed: 9`.** The three right-inbound cases carry
  `status: not_evaluated` with
  `reason: owner_approved_early_stop_before_worker_returned`. Their per-case
  counts are unknown and are not estimated.
- **`nodes.npz`, `edges.npz`, `selected_routes.npz` were not written.** Worker
  node and edge tables were returned only on case completion, so the nine
  completed cases' tables did not survive the stop; no geometry was selected, so
  there are no selected routes. The artifact names each missing file and why.
  The only surviving per-case record is the workers' verbatim stdout, preserved
  as `search_worker_stdout.log` and hashed in the stop record.
- The endpoint certificate was computed **fresh** after the stop, from the
  retained qpos and the registered geometry, so the conclusive witness does not
  depend on anything lost with the workers.

### Deviations from the plan, stated explicitly

1. **No V10.3 episode runtime was built.** The plan lists a sampler, a
   `JointMoveSequence` dispatch, and screen / review / causal / phase-0 runners.
   The search stopped before Step 0C, so none of those surfaces became
   reachable, and none was written. `run_pact_place_v103_preflight.py`,
   `run_pact_place_v103_screen.py`, `run_pact_place_v103_review.py`,
   `run_pact_place_v103_causal.py`, and `run_pact_place_v103_phase0.py` are
   recorded as `absent` in the implementation hash table.
2. **Step-0A items that depend on that runtime were not implemented**: runtime
   executes the selected joint qpos sequence without invoking TCP IK; a
   parked/disabled V10.3 pendant is impossible in an admitted row; real-time
   review timing at the runtime level; and the role-600 negative-control /
   role-601 positive-control IK pair as *runtime* controls. Role 600's frozen
   V10.2 pose does appear as a measured negative control, in the endpoint
   certificate. The 34 implemented tests are the subset that is meaningful
   without episode runtime.
3. **Registered x-reversals.** The topology's own legs run against the direction
   of travel twice: the outbound retreat in `+x` from the loaded lift (which the
   plan names as allowed) and the inbound approach back from
   `far_pregrasp_staging` to the pregrasp endpoint, which is forced because
   `far_staging_x` (0.81 / 0.83 m) lies beyond every cell's pregrasp x
   (0.736–0.746 m) for both registered buffers. The planner treats a reversal
   *within* one registered segment as a wrong-way route and the registered
   topology's own legs as registered. This reading is recorded here because the
   plan's "no other x reversal is allowed" does not name the inbound case.
4. **BLAS thread pinning.** The search pins `OPENBLAS/OMP/MKL` to one thread per
   process. Without it, twelve workers on a 128-core host spent all their time
   in futex contention. This changes no predicate or threshold.

### Stop

V10.3 stops here. No height, lane, staging buffer, pass-z offset, orientation,
seed, or clearance value was added or changed in response to these results. No
episode was generated, no runtime was built, and no downstream work is
authorized.
