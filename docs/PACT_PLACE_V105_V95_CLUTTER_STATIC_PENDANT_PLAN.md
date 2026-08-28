# PACT place V10.5: V9.5 real clutter with a behaviorally relevant static pendant

**Scope:** build one new environment, certify it offline, publish one six-video
owner-review packet, and—only after explicit owner approval—run one 24-row
Phase-0 gate. Demonstration collection, conversion, training, and learned-policy
evaluation are outside this plan and remain unauthorized.

## Objective and owner decisions

V10.4 proved that a compiled-static pendant can coexist with a reliable place
expert, but it used V6c's four rectangular clutter boxes and left at least
61.6 mm of pendant clearance. That environment is too sparse and the pendant
is not behaviorally consequential enough for the intended PACT-versus-ACT
comparison.

V10.5 must:

1. restore the settled V9.5 household-object palette and layout, including its
   movable free-body clutter and ordinary contact/stability failure semantics;
2. retain one connected, visible-equals-collision, compiled-static pendant;
3. move the pendant inward enough to create a measured near-contact risk while
   preserving a 15 mm nominal clearance floor on historically clean expert
   trajectories;
4. allow a small preregistered set of pendant poses at episode reset, with the
   selected pose remaining completely rigid and static for the entire episode;
5. use a new Phase-0 pass threshold of **16/24**, with balance floors defined
   below; and
6. stop after the review packet for the owner's decision. The coding agent must
   not infer approval or run Phase 0 without an owner-authored approval record.

The owner accepts the lower expert yield. It is acceptable for a later,
separately authorized collection to require more attempts to obtain its target
number of strict-clean episodes. That later collection is not part of this
document.

## Exact lineage: what “restore V9.5 clutter” means

Use the **fixture-free settled V9.5 clutter lineage**, not the V9.5 low-wall
fixture:

- scene base: `pact_place_corridor_v5.xml`;
- sampler behavior: `PactPlaceCorridorV93Sampler`;
- palette: `load_v95_palette()` from `scripts/pact_place_v95_contract.py`;
- layouts: `build_v95_layout()` for all F0/F1/F2/F3 families and both sides;
- active side panel, target/tray, vessel jitter, real-object injection, cameras,
  expert vessel bows, contact audit, and clutter-stability semantics inherited
  unchanged.

Do **not** use `PactPlaceCorridorV95LowWallSampler`, activate a V9.4/V9.5 wall
fixture, activate an old ceiling mount, or import any V9.8–V10.3 route branch.
The measured 51% seed-robustness result and its approximately 12.25/24
expectation came from the fixture-free V9.3 sampler with the settled V9.5
palette/layout; it did not include the V9.5 low wall.

The production manifest must carry the complete palette and layout records.
Every selected household object remains a collision-enabled movable free body.
Robot/target contact with clutter, clutter instability/drift, or contact caused
by an object being knocked over remains a strict-clean failure. Do not turn
objects into visual-only props, immovable boxes, or parked decoration.

Before any new episode, reconstruct at least one immutable row from every
V9.5 family/side cell through both the historical path and the new no-pendant
V10.5 path. Require exact equality of the initial robot and target qpos, panel
pose, all active clutter object identities/poses/quaternions, tray/shell,
camera configuration, jitter, and physics options. The only permitted
differences are the V10.5 environment marker, the selected static pendant
scene, telemetry, and the registered initial speed cap.

## Non-negotiable boundaries

1. Preserve every V9.5–V10.4 source artifact, scene, config, video, and result
   byte-for-byte. V10.4 remains reviewed but not Phase-0 approved. Use new files
   and new output directories only.
2. Freeze the V10.5 contract, candidate lattice, selection rule, reset-pose
   distribution, review manifest, Phase-0 manifest generator, thresholds, and
   implementation hashes before the first V10.5 episode.
3. The pendant is static. No pendant body may have a joint, `freejoint`, mocap
   flag, actuator, or runtime pose/size write. Never write its `geom_pos`,
   `geom_size`, `geom_aabb`, `geom_rbound`, or BVH fields at runtime.
4. Implement each production reset pose as a distinct precompiled MJCF scene.
   The manifest chooses the scene before MuJoCo model construction. Loading a
   different registered scene for a different episode is allowed; changing the
   pendant after model compilation is not.
5. Visible and collision geometry are identical. Pendant collision must be
   enabled and robot or carried-target contact with any V10.5 component must
   be classified as `mounted_fixture` and make the row unclean.
6. Keep the fixture-free V9.5/V9.3 expert route and IK behavior. Do not add a
   pendant-specific lane, route search, fallback, waypoint mutation, or
   post-hoc detour. Retain only the V10.4 initial free-space speed cap of
   0.12 m/s, gated on the exact V10.5 marker and schedule hash; leave every
   other inherited segment speed unchanged. Explicitly dispatch the new marker
   through the V9.3 panel/vessel expert branch; do not let it fall through to a
   V3/V5 policy path. The pendant must not enter a planner obstacle list or a
   privileged surface-distance speed law.
7. Review and Phase-0 seed/episode streams are disjoint. No review row may be
   substituted into or counted in Phase 0.
8. Missing telemetry, sampling/IK/infrastructure failure, hash mismatch, or an
   incomplete official row fails closed. Do not replace an official Phase-0
   row.
9. No threshold, scene, route, palette, pose, seed, or selection rule may be
   changed in response to a live review or Phase-0 outcome.
10. Even a Phase-0 pass sets only `phase0_passed: true`; it authorizes no
    collection, conversion, training, evaluation, or videos beyond this plan.

## Static pendant family

### Shape and height

Reuse the V10.4 connected two-lobe shape and height. Do not reopen the failed
height search:

| component | centre before lateral parameters (m) | half extents (m) |
|---|---|---|
| negative lobe | `(x, -r + d, 1.01)` | `(0.010, 0.010, 0.030)` |
| positive lobe | `(x, +r + d, 1.01)` | `(0.010, 0.010, 0.030)` |
| negative stem | `(x, -r - 0.010 + d, 1.2725)` | `(0.006, 0.006, 0.2325)` |
| positive stem | `(x, +r + 0.010 + d, 1.2725)` | `(0.006, 0.006, 0.2325)` |
| crossbar | `(x, d, 1.510)` | `(0.006, r + 0.016, 0.005)` |

Here `x` is the assembly depth, `r` is the symmetric lobe centre magnitude,
and `d` translates the complete connected assembly laterally. The crossbar top
remains flush with `hood_top` at z=1.515 m. That designed crossbar/hood face
may be allowlisted only for environment self-contact checks; it never exempts
robot or carried-target contact.

The lobe bottom remains z=0.98 m, above the table clutter band. Do not lower the
pendant toward the household objects or raise it into the V10.2/V10.3 wrist
envelope.

### Preregistered offline lattice

Search this small lattice once, using retained states only:

```text
x: 0.740, 0.760, 0.780, 0.800 m
r: 0.290, 0.295, 0.300, 0.305, 0.310, 0.315, 0.320, 0.325 m
d production set for every (x,r): -0.005, 0.000, +0.005 m
```

A candidate is the complete three-scene bundle for one `(x,r)`, not one
individually favorable pose. Label the production pose IDs `neg5`, `center`,
and `pos5`. The inner lobe faces are necessarily closer to the corridor centre
than V10.4's 0.330 m inner faces; a candidate that does not meet the risk gates
below cannot survive merely because it is geometrically inward.

Do not extend or densify this lattice after observing results. Do not search
z, lobe dimensions, stem thickness, crossbar thickness, route parameters,
clutter positions, or thresholds.

### Static scene construction

For every lattice probe, generate or load a complete static scene derived from
`pact_place_corridor_v5.xml`. For the selected bundle, publish exactly three
new production XML files plus one no-pendant counterfactual scene. All four
must be SHA-bound before live execution.

Each production XML contains one V10.5 pendant body at its final component
poses and sizes, with no runtime adjustment. Compile each scene independently
and assert:

- model geom positions/sizes match the registered assembly to 1e-9 m;
- `geom_aabb`, `geom_rbound`, and body BVH bounds enclose the true geometry;
- the body has no joint, freejoint, mocap ID, or actuator;
- visible and collision geoms are the same five geoms;
- collision masks match the registered mounted-fixture class; and
- compile/reload yields byte-stable metadata and identical exact distances.

## Step 0 — contract, trust anchors, and tests

Create a distinct marker and an acyclic two-stage contract, for example:

```text
environment_version: pact_place_corridor_v10_5_v95_clutter_static_pendant
contract_version:    pact_place_v105_v95_clutter_static_pendant_v1
```

Before search, write a **specification contract** binding the plan, immutable
inputs, lattice, predicates, ranking, and explicit implementation-file list.
After Steps 1–3, write one **execution contract** that binds that specification,
the reconstruction/siting/causal outputs, selected scenes, complete review and
Phase-0 runner code, manifests/stream derivation, and all thresholds. Freeze
the execution contract before the first live row.

Do not create a circular aggregate hash. Every artifact binds only already-
existing inputs. For every JSON, store separately (a) raw-file SHA-256 and
(b) canonical payload SHA-256 computed with its own hash field omitted. Never
call a pre-insertion self-hash a raw-file or final-payload hash. Hash an
explicit ordered file list; unrelated repository changes must not silently
alter the contract.

Across the two stages, the contracts must bind, by path and SHA-256:

- this plan and every V10.5 implementation/test file;
- the V5 source scene, selected palette, V9.5 layout helper, and sampler/policy
  implementation;
- `diagnostics_output/pact_place_v95_seed_fragility/fragility.json` and the
  retained source rows actually used for siting;
- relevant V9.5 correction/source artifacts;
- the three selected production scene XMLs and no-pendant control scene;
- siting, reconstruction, and causal artifacts;
- review and Phase-0 manifest generators, thresholds, streams, and master
  seeds; and
- the exact initial speed schedule and contact-classification implementation.

Do not turn the historical 7/8-versus-6/8 validated-seed discrepancy into a
new truth claim. Read strict-clean status from each retained source row used by
the new audit, record its file hash, and preserve the unresolved historical
summary discrepancy.

Add behavioral tests before the offline search. At minimum cover:

1. exact settled V9.5 palette/layout restoration and absence of the low wall;
2. three scene/pose IDs and balanced manifest assignment;
3. rejection of a scene/pose/hash mismatch;
4. no runtime pendant model writes or movable pendant body;
5. compiled-bound and exact/live contact parity;
6. robot and carried-target pendant contact classified as
   `mounted_fixture` and strict-unclean;
7. ordinary clutter contact and stability events remain strict-unclean;
8. V10.5-only 0.12 m/s initial cap and unchanged later speeds;
9. exact V9.3 expert waypoint/IK dispatch under the V10.5 marker, apart from
   the one speed cap, with no pendant planner obstacle or route mutation;
10. no pose ID, scene ID, geometry metadata, or privileged clearance leakage
    into either student's observation;
11. candidate-bundle admission and deterministic ranking;
12. qpos/model/data restoration on success, failure, and exception;
13. review selection and six-video balance rules;
14. Phase-0 counting at 16/24 and all balance floors below;
15. review/Phase-0 stream and episode-ID disjointness; and
16. authorization fields default false and cannot be overwritten by dictionary
    merge order.

Run focused tests plus V9.5, V10.2 contact-parity, V10.3 static-scene, V10.4,
mounted-fixture, contact-audit, and failure-cause regressions before any new
episode. Historical known failures must be identified and provenance-checked;
do not silently exclude or call them new regressions.

## Step 1 — independent V9.5 reconstruction

Build a new reconstruction artifact without importing stored outcome booleans.
For every retained source row used below:

- verify source result/config/trajectory/qpos hashes before load;
- reconstruct its own palette, layout, jitter, panel, robot, target, and
  clutter state;
- require TCP residual <=1 mm and exact arm/target/clutter identity;
- store float64 qpos and active geom transforms/AABBs;
- label strict-clean from the row's own contact/task telemetry; and
- verify save/reload hashes.

Use all recoverable strict-clean trajectories in the 24-seed × eight-cell V9.5
fragility corpus, not just seed 955339 or the six historically convenient
cells. If a source row lacks sufficient retained state, mark it unavailable
with a hash-bound reason; never synthesize or replace it. Require at least two
strict-clean retained trajectories per family/side cell before siting. If that
coverage is absent, stop before geometry selection.

No `env.step` and no new episode are allowed in Steps 1–3.

## Step 2 — exact environment and trajectory filtering

Score every `(x,r,d)` scene against every reconstructed applicable cell without
early termination. Use tight convex-hull AABBs only as broad phase and hardened
exact GJK/`mj_geomDistance` as the decision instruments.

A three-pose `(x,r)` bundle survives only if all of the following hold:

1. no pendant component intersects any collision-enabled non-robot environment
   geom, household object, initial target, tray, panel, or shell at any retained
   frame in any reconstructed cell, except the designed crossbar/hood_top flush
   face; pendant-to-clutter clearance must remain positive throughout;
2. initial robot and target clearance from every pendant component is >=15 mm;
3. every historically strict-clean retained trajectory remains free of robot
   and carried-target pendant contact at every recorded frame;
4. minimum exact pendant clearance on every such trajectory is >=15 mm;
5. grasp, close, lift, and release windows independently remain >=15 mm;
6. the closest-risk witness binds to a lobe or stem and a robot hand/link
   collision geom—not the crossbar/hood flush face or an AABB proxy; left-route
   witnesses must bind the negative-y lobe/stem and right-route witnesses the
   positive-y lobe/stem;
7. for each of the six `pose_id × side` groups, at least one retained clean
   trajectory has a lobe/stem minimum in the inclusive **15–35 mm** risk band;
8. each side has at least one inbound and one loaded-outbound risk-band witness
   across the three poses; and
9. no historically clean row is made unclean by ordinary environment or
   clutter reconstruction differences, including a clutter object contacting
   the pendant or acquiring a new stability event.

Record per-frame/per-component witnesses, not only aggregate minima. Store the
binding frame, phase, direction, robot geom/body, component, signed distance,
TCP pose, full qpos hash, source-row hash, and exact scene hash in an NPZ plus a
human-readable JSON summary.

## Step 3 — prove contact risk and causal sensing

### Small-deviation contact certificate

For each `pose_id × side`, take its closest clean retained witness and perturb
the robot path toward the binding pendant component along the measured
separation direction. Test preregistered magnitudes 5, 10, 15, 20, 25, and
30 mm in order. Solve the displaced pose with sequential IK seeded from the
retained qpos and validate every interpolated joint state.

Require a genuine robot-pendant contact by no more than 30 mm displacement for
all six groups. The first new collision must be the pendant, not clutter,
panel, tray, target, or shell. Signed `mj_geomDistance`, analytic GJK, live
`data.contact`, and the place contact audit must agree on the contacting pair.
These are retained-state diagnostics, not episodes, and the original qpos and
scene state must be restored after every probe.

This certificate is the operative proof that the pendant is close enough to
matter. Do not certify risk by moving the pendant, altering collision sizes, or
using the old 83–175 mm diagnostic translations.

On matched qpos, verify that the reset poses produce the expected directional
ordering rather than three nominally different but behaviorally identical
scenes. Left-route lobe/stem clearance must be lower in `pos5` than `center`
and lower in `center` than `neg5`; the right-route ordering is the reverse.
Require at least 5 mm separation between the closest and farthest pose for each
side. Record exceptions per frame, but fail the bundle if the aggregate
closest-risk witnesses do not have this ordering.

### Raw proximity causality

At the clean risk witnesses, compare each selected compiled scene with the
compiled no-pendant control at byte-identical robot, target, panel, and clutter
state. Render the real production `[40, 4, 8, 8]` proximity tensor; geometry
proxies alone cannot pass.

Require, for every pose and side:

- deterministic control repeat (`max_abs_delta == 0`);
- at least 448 changed tensor values in the registered approach/pass window;
- at least three distinct sensors responding;
- at least one link5 or link6 sensor among them;
- a response beginning at least five executed control frames and at least
  0.10 s before closest approach; and
- worst left/right changed-value ratio <=4x after aggregating paired witnesses.

Also record the V9.5 ground-clutter counterfactual honestly: household props are
RGB-visible collision hazards, but this plan makes no claim that the low-
resolution proximity skin resolves each individual prop. The pendant and
active panel supply the registered proximity evidence.

### Deterministic bundle selection

Reject any `(x,r)` bundle missing any predicate above. Rank survivors
lexicographically, with no top-N truncation:

1. maximize the number of side/direction witnesses in the 15–35 mm band;
2. minimize absolute deviation of the median lobe/stem clearance from 25 mm;
3. minimize left/right median-clearance imbalance;
4. maximize raw causal changed values at the earliest valid onset;
5. prefer the larger `r`, then larger `x`, as the conservative deterministic
   tie-break.

Select exactly one `(x,r)` and therefore exactly three production scenes. Write
the full scored lattice, all rejection reasons, selected witnesses, scene
hashes, and authorization fields to a new immutable siting artifact. If there
is no complete three-pose survivor, stop. Do not extend the lattice, relax a
floor, select fewer poses, or fall back to V10.4.

## Step 4 — freeze manifests and runtime telemetry

Only after Steps 0–3 pass, freeze the live contract and manifests.

Use distinct deterministic streams, for example:

```text
review stream: pact-place-v10.5-v95-clutter-review
review seed:   2026105002
phase0 stream: pact-place-v10.5-v95-clutter-phase0
phase0 seed:   2026105001
```

The Phase-0 manifest contains exactly the Cartesian product:

```text
4 V9.5 layout families × 2 intrusion sides × 3 pendant pose IDs = 24 rows
```

Thus each side appears 12 times, each pendant pose eight times, and each
`side × pose` cell four times. Derive a distinct task seed and episode ID for
every row. Store the entire static scene path/hash and pendant assembly/hash in
every row; the sampler must refuse a mismatch before task creation.

`pose_id`, scene ID, exact assembly coordinates, privileged clearances, and
manifest geometry fields are telemetry/expert data only. They must not be
added to the ACT or PACT student observation. Both policies retain the same
RGB/state inputs; only PACT retains the already registered proximity input.

Record per control frame:

- phase/segment name, commanded and realized TCP speed;
- pose ID and scene hash;
- exact per-component robot/target pendant clearance;
- earliest/closest sensor-response timing;
- pendant and ordinary contact pairs/classes;
- active household-object identities, poses, and stability deltas;
- IK/fallback/clip/wrong-way state; and
- task, grasp, lift, release, and strict-clean status.

Never use offline ordinary-environment clearance as a replacement for live
contact auditing. Restore qpos/model/data on every diagnostic exit.

## Step 5 — bounded production review pool and six videos

Freeze a 48-row review candidate manifest before executing row 0: two disjoint
replicates of the same 24-cell `family × side × pose` product, using only the
review stream. Execute rows in manifest order and retain every result. The
runner may stop early only after a valid six-video packet can be selected under
the deterministic rule below; otherwise execute all 48 and fail closed if no
packet exists.

All review clips must be complete **production episodes** in the selected
static scene. Do not use a separately shifted diagnostic assembly, induced
contact, composited prop, parked-object replay, or retained-qpos control as one
of the six.

Select exactly:

- three strict-clean successes, one from each pendant pose;
- three natural strict-unclean failures, one from each pendant pose;
- three left and three right rows across all six clips; and
- at least two V9.5 layout families in each outcome class.

Among subsets satisfying those constraints, prefer natural pendant-contact
failures, then clutter-contact/stability failures, then task/IK failures; break
all remaining ties lexicographically by manifest role index. Never relabel a
failure or omit an earlier row to improve appearance outside this frozen rule.
If no valid subset exists after 48 rows, stop before owner review and Phase 0.

Render each complete trajectory at real-time speed with synchronized panes:

1. untinted wrist RGB policy view;
2. wide third-person view that visibly includes the restored household-object
   clutter, robot, target, panel opening, and pendant;
3. pendant-side/clearance view; and
4. proximity heatmap or compact sensor overlay.

Overlay outcome, family, side, pose ID, phase, commanded/realized speed,
minimum pendant clearance, binding component/body, clutter stability, and
contact classes. A clean near-pass must be visually legible; do not crop the
pendant or hide the restored props.

Publish `REVIEW.md`, the immutable review manifest, six video hashes, all source
row hashes, and a complete approval template. State plainly that the expected
expert yield is lower than V10.4 and that the household clutter is not claimed
to be proximity-resolvable.

Then stop. The coding agent must not create or edit the owner approval record
and must not run Phase 0.

## Owner approval boundary

Phase 0 may run only after the owner personally reviews all six videos and
writes `diagnostics_output/pact_place_v105_review/human_approval.json` as a new
V10.5-specific approval record with:

```json
{
  "decision": "approve_phase0",
  "created_by": "owner",
  "environment_version": "pact_place_corridor_v10_5_v95_clutter_static_pendant",
  "reviewed_videos": ["six exact relative paths"],
  "bindings": {
    "contract_sha256": "...",
    "siting_payload_sha256": "...",
    "causal_payload_sha256": "...",
    "review_manifest_sha256": "...",
    "video_sha256": {
      "relative/path/to/video_1.mp4": "...",
      "relative/path/to/video_2.mp4": "...",
      "relative/path/to/video_3.mp4": "...",
      "relative/path/to/video_4.mp4": "...",
      "relative/path/to/video_5.mp4": "...",
      "relative/path/to/video_6.mp4": "..."
    }
  }
}
```

The runner must reject agent-created, missing, incomplete, stale, extra-video,
wrong-scene, or wrong-environment approval. Any code, scene, palette, route,
speed, threshold, manifest, or implementation-hash change after review
invalidates the packet and requires a new version—not silent regeneration.

## Step 6 — one untouched 24-row Phase-0 gate

After valid approval, execute the frozen 24-row manifest exactly once, in its
registered order, with no replacement, retry, repair, or early success stop.
Every row must reach a terminal recorded outcome. An infrastructure or missing-
telemetry row fails the gate; it is not replaced.

Phase 0 passes only if all of these hold:

- at least **16/24** strict-clean task successes;
- at least **7/12** strict-clean successes on each intrusion side;
- at least **4/8** strict-clean successes for each pendant pose ID;
- at least **2/4** strict-clean successes in every `side × pose` cell;
- every counted success has zero robot/target pendant contact, zero forbidden
  ordinary contact, zero clutter contact/stability event, and a valid
  grasp/lift/release;
- every counted success has complete speed, clearance, sensor, contact, and
  stability telemetry;
- at least one completed row per side and per pose records a lobe/stem closest
  approach <=35 mm, confirming that live execution did not bypass the certified
  risk region; and
- all contract, scene, manifest, implementation, source, and approval hashes
  still verify after the run.

Do not require a production pendant collision—the goal is avoidance—but report
all pendant-contact failures separately from clutter, IK, sampling, task, and
infrastructure failures. Report clean counts overall, by family, by side, by
pose, and by `side × pose`, plus exact Wilson intervals and realized episode
length/speed distributions.

On pass, write `phase0_passed: true` and keep every downstream authorization
false. On failure, write the exact limiting predicates and close V10.5. Do not
change the 16/24 bar, select a favorable seed, drop a pose, rerun failed rows,
or launch a second official gate.

## Required artifacts

Use new immutable directories, for example:

```text
diagnostics_output/pact_place_v105_reconstruction/
diagnostics_output/pact_place_v105_siting/
diagnostics_output/pact_place_v105_causal/
diagnostics_output/pact_place_v105_review/
diagnostics_output/pact_place_v105_phase0/
```

Each JSON must include a schema version, payload SHA-256, raw-file hashes,
input/output bindings, execution flags, stop reason, and explicit authorization
booleans. Large per-frame witnesses belong in hash-bound NPZ files; document
their dtypes, shapes, row ordering, units, and loader validation. Never claim
search exhaustiveness if a candidate/cell/frame was skipped.

Update `README.md` and `EVAL.md` only with measured outcomes. Preserve the
historical V9.5, V10.3, and V10.4 conclusions and make the lineage change
explicit: V10.5 restores V9.5 real-object clutter but not the V9.5 low wall.

## Final handoff format

At each stop, report:

1. completed and unexecuted steps;
2. exact tests and regressions;
3. protected-artifact verification;
4. reconstruction coverage;
5. full lattice counts and rejection table;
6. selected `(x,r)` and all three exact static assemblies/scenes;
7. per-side/per-pose clearance, contact-risk, and raw-causal witnesses;
8. review-pool result table and six clickable video paths/hashes;
9. whether owner approval exists;
10. if authorized and run, the complete Phase-0 table; and
11. confirmation that collection, training, and evaluation were not run.

The intended first handoff is the six-video review packet with Phase 0 still
locked. The owner will return a decision after watching the videos.

---

# Execution record

**Status: stopped at Step 2. The registered lattice selected no bundle.**

Executed 2026-08-28. This section records measured outcomes only.

## Step 0 — contract, trust anchors, tests: complete

Marker `pact_place_corridor_v10_5_v95_clutter_static_pendant`, contract
`pact_place_v105_v95_clutter_static_pendant_v1`.

New implementation, all additive:

| file | role |
|---|---|
| `scripts/pact_place_v105_geometry.py` | assembly from `(x, r, d)`, scene generation, lattice |
| `scripts/pact_place_v105_contract.py` | specification contract, manifests, gate accounting |
| `scripts/pact_place_v105_clearance.py` | exact clearance, risk boxes, contact state |
| `scripts/pact_place_v105_runtime.py` | the one registered speed cap, doubly gated |
| `scripts/pact_place_v105_siting_core.py` | snapshot-once scoring core |
| `scripts/run_pact_place_v105_reconstruct.py` | Step 1 |
| `scripts/run_pact_place_v105_siting.py` | Step 2 |
| `scripts/run_pact_place_v105_step3.py` | Step 3 |
| `tests/test_pact_place_v105.py` | 67 behavioural tests |

`PactPlaceCorridorV105Sampler` derives from `PactPlaceCorridorV93Sampler` and is
a marker-and-binding pass-through: it changes no palette, layout, jitter, panel,
target/tray, camera, or contact semantics, and it refuses a scene-hash mismatch
before task creation. The marker was added to the two V9 expert allow-lists
(`_v9_enabled` and the `inbound_hazard_role` set) so it dispatches through the
V9.3 panel/vessel branch rather than falling through to a V3/V5 path, and was
deliberately **not** added to `_mounted_fixture_roles` or the V10 lane family,
so the pendant never enters a planner obstacle list.

**Tests: 67 passing.** All sixteen registered areas are covered.

Two test defects were found and fixed in the tests themselves, not the code: a
substring scan for runtime model writes flagged a *read* of `model.geom_size`,
and an observation-leakage check matched the word "observation" inside a
comment. Both were replaced with AST-based checks, and the model-write check now
carries a guard test proving it still catches a genuine write.

## Step 1 — independent V9.5 reconstruction: passed

Strict-clean status was re-derived from each retained row's own contact and task
telemetry; the stored boolean is recorded for comparison only. **All 192 rows
agree** with their stored value, so nothing was disputed or excluded.

| cell | clean | | cell | clean |
|---|---:|---|---|---:|
| F0 left | 17 | | F0 right | 17 |
| F1 left | 9 | | F1 right | 11 |
| F2 left | 12 | | F2 right | 10 |
| F3 left | 14 | | F3 right | 8 |

**98/192 = 51.0% clean**, reproducing the recorded `mean_clean_rate` of
0.5104 exactly. Every cell clears the two-per-cell floor; the minimum is 8.

Sixteen rows (two per cell) were replayed through the live V9.3 sampler:
**16/16 reconstructed**, TCP residual 0.067–0.163 mm against a 1 mm limit,
21 active clutter free bodies and 972-element qpos in every case.

Two preserved instrument failures, neither a design change:
`..._attempt_01_instrument_failure` — the V10.5 sampler class was defined before
its V9.3 base class, so importing `enclosure_reach` raised `NameError`;
`..._attempt_02_instrument_failure` — the TCP accessor used `env.robot_view`
instead of `env.current_robot.robot_view`.

## Step 2 — lattice scoring: complete, and it selected nothing

Scoring 32 candidates × 3 poses against every reconstructed strict-clean
trajectory. The core snapshots each source row once and scores all 96 candidates
against that snapshot, with an AABB screen as broad phase and hardened exact GJK
as the only decision instrument. Two soundness-preserving optimisations made
this tractable: environment geoms that can never approach any candidate are
screened out once (496 → 32), and because both the pendant and most of the
environment are static, static pairs are measured once rather than per frame
(23 s → 0.40 s per candidate, identical result).

**Result: 0 survivors of 32 candidate bundles. 96/96 scenes scored against
98/98 retained strict-clean trajectories, no early termination, no failed
rows.** Artifact `diagnostics_output/pact_place_v105_siting/siting.json`,
payload `4c9f5646…`, raw file `56f5d6ba…`,
`stop_reason: no_complete_three_pose_bundle_survived_the_lattice`.

### Why every candidate was rejected

| predicate | bundles failing |
|---|---:|
| 4 — min clearance ≥ 15 mm on **every** clean trajectory | **32/32** |
| 8 — inbound *and* loaded-outbound risk-band witness per side | 30/32 |
| 3 — no robot/carried-target pendant contact on a clean row | 27/32 |
| 7 — a 15–35 mm witness for every `pose_id × side` group | 8/32 |

Predicates 1, 2, 5, 6 and 9 rejected nothing: the pendant never intersected a
household object, panel, tray or shell; initial clearance was always adequate;
the grasp/lift/release windows never bound; every closest-risk witness bound a
lobe or stem on the route's own side; and no clean row was made unclean by
reconstruction differences.

**The binding constraint is predicate 4, and it is universal.** Every one of the
96 scenes puts at least one of the 98 historically clean V9.5 trajectories below
the 15 mm floor. The trade is visible in the ranking: the bundle with the most
risk-band witnesses, `x = 0.780, r = 0.325`, has **153** witnesses in the
15–35 mm band but its worst clean trajectory falls to **3.4 mm**. The bundle
with the highest floor, `x = 0.800, r = 0.320`, still reaches only **9.4 mm**.
Moving the assembly outboard raises the floor and starves the risk band; moving
it inboard fills the band and causes contact. No lattice point does both.

The lattice boundary is also binding: the best candidates sit at the largest
`x` and largest `r` the plan registers. The plan forbids extending the lattice,
so it was not extended.

### A predicate defect found and corrected before reporting

The first aggregation counted a direction as a risk-band witness only when the
row's *overall* lobe/stem minimum was already in band. Because the loaded
outbound leg almost always passes closer than the inbound leg, that gate
suppressed every inbound witness and made the lattice look as though inbound
approach never came near the pendant at all. It does: with the gate removed,
`left:inbound` witnesses appear in 22/32 bundles and `right:inbound` in 5/32.

The defect changed no selection — 0 survivors either way — but it would have
made the reported reason wrong. The uncorrected run is preserved at
`diagnostics_output/pact_place_v105_siting_attempt_01_direction_predicate_defect/`
along with its 16-row pilot. The corrected result was re-derived from the same
stored per-row scores, so no trajectory was re-measured and no score changed.

## Steps 3–6 — not executed

Step 3 refuses to run without a selected bundle and was verified to do so. No
contact certificate, causal comparison, production scene, manifest, review
packet, video, or Phase-0 row was produced. `diagnostics_output/`
`pact_place_v105_causal/`, `..._review/` and `..._phase0/` do not exist, no
V10.5 scene XML was published into the scenes directory, and
`human_approval.json` was neither created nor required.

Per the plan: the lattice was not extended, no floor was relaxed, fewer poses
were not selected, and there was no fall back to V10.4.

## What a successor plan would have to change

The result says the V10.4 pendant *shape and height* cannot be made
behaviourally relevant against the full V9.5 trajectory corpus while preserving
a 15 mm floor on every historically clean run. Three levers remain untouched by
this plan, in rough order of promise:

1. **Per-cell or per-family placement** rather than one global `(x, r)`. The
   floor is violated by a minority of the 98 trajectories; a placement chosen
   per layout family may clear all of its own cell's rows.
2. **A lower floor with an explicit expected-yield budget.** The owner already
   accepted a lower expert yield; a 10 mm floor would admit several bundles.
   That is a threshold change and must be preregistered, not adopted here.
3. **A different pendant height or shape**, which reopens a search this
   programme has repeatedly found expensive.

Extending the `x`/`r` lattice outward is the obvious fourth lever and is the
weakest: the trend is monotonic, and further out the risk band empties.

## Consequence for the V10.4 review-v2 packet

Registering `PactPlaceCorridorV105Sampler` required editing two files that the
V10.4 Step-0 preflight binds:
`submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py`
(`939a4d4c…` → `cea7576f…`) and `scripts/run_pact_place_expert_screen.py`
(`685c8774…` → `32f7e989…`). There is no way to add a new sampler to the family
without touching the module that defines it.

The V10.4 provenance bridge therefore now fails, **which is the bridge working
correctly**. Its data checks all still pass — v1 inputs, scene, metadata, the
six production rows and the causal artifact are byte-identical, and the six
published MP4s are untouched. Only the scoped *implementation* binding moved.

By V10.4's own rule — "any implementation-hash change after review invalidates
the packet and requires a new version, not silent regeneration" — the V10.4
review-v2 packet is no longer approvable as published, and it was **not**
regenerated here. That is an owner decision, not an agent one.

## Test results

67 V10.5 tests pass. The full `tests/` sweep is **1317 passed, 26 failed,
1 skipped**. Eighteen of the 26 are the pre-existing failures unrelated to this
work (three stale V3/V5 corridor hash assertions, plus fifteen in checkpoint,
slideshow, oracle-contract and threshold areas). The remaining **eight** are the
V10.4 review-v2 provenance-bridge and approval-schema tests, failing for the
reason above; they are a true report of a real binding change, not a defect.

`git diff --check` is clean in both repositories.

## Not done

No production episode, `env.step`, Phase-0 row, collection, conversion,
training, or learned-policy evaluation occurred at any point. Every artifact
carries `authorizes_*: false` and `phase0_passed: false`.
