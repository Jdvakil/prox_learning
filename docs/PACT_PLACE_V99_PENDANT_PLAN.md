# V9.9 Fixed Pendant Qualification Plan

Status: permanently closed. Scoped conclusion: **no survivor in the
registered fixed rectangular-box lattice.**

Independent reconstruction of the frozen eight V9.5 rows succeeded (max TCP
residual 0.87 mm). Reconstruction NPZ paths and SHA-256 values are in
`reconstruction.json`. The final offline close-out stored geometry transforms
and AABBs as float64, verified reconstruction / source-row / V5 scene /
implementation hashes before load, bound snapshot metadata to those hashes,
and scored the unchanged lattice with all four exact predicates on all six
clean cells (no early termination). Live `mj_forward` contact-parity and
near-threshold checks passed.

The conservative AABB filter certified **0** candidates. That certificate
is only a broad-phase screen. Tight convex AABBs produced **12880**
dual-transit hits (the earlier 30004 figure used inflated mesh rbounds).
The exact retained-qpos `true_distance` / GJK pass scored all 12880
(no physics stepping, no episodes) and found **0 survivors**. Overlapping
reject counts: inbound_no_contact=4209, outbound_no_contact=84,
grasp_below_nominal=12880, initial_contact=0. Per-cell witnesses (frame,
geom, contact distance, grasp clearance margin) are in
`exact_witnesses.npz`. Sequential-IK routing, paired screens, and collection
were not run. The lattice and thresholds were not relaxed.
`authorizes_paired_screen: false` and `authorizes_collection: false`. V9.8
remains unchanged. Collection, training, and eval are not authorized.

This close-out does not claim that every dual-hit occupies the grasp/lift
window, and it does not claim that no ceiling pendant of any kind can work.
It claims only that the registered fixed rectangular-box lattice has no
exact survivor under the frozen six-cell layout and 25 mm / dual-transit
gates.

Success supports only:

> Under the pinned six-cell V9.5 engineering layout, one fixed, identical
> ceiling pendant can force collision-free lateral detours of at least 5 cm
> in both traversal directions while remaining bilaterally visible to the
> proximity skin.

It does not establish multi-seed robustness, authorize collection, or
establish PACT superiority. Every artifact carries `authorizes_collection:
false`. Human review can be approved only by the user.

## Summary and frozen boundary

`pact_place_corridor_v9_9_pendant` tests whether one fixed, ceiling-flush,
side-independent pendant can force a substantial lateral route while
preserving the six authoritative clean F0/F1/F2 side/family cells.

Freeze:

- The exact rows in `diagnostics_output/pact_place_v95_raw_smoke`:
  `PactPlaceCorridorV93Sampler`, V9.5 palette/layout, aperture, panel, target,
  clutter, grasp pose, and seed 955339.
- The six clean F0/F1/F2 left/right cells as the preservation population.
  F3 left/right remain regression-guard rows but are not required to become
  clean.
- Identical pendant geometry on both panel sides.
- Active pendant semantics on both empty inbound and loaded outbound
  traversal.
- The existing sensing floors: at least 3 changed sensors and 448 changed
  values per cell, a link5-front/link5-back/link6 responder, and no more than
  4× left/right imbalance.
- A minimum 0.05 m executed TCP-y detour on both traversals through the
  pendant slab.
- No collection, training, ACT/PACT evaluation, aperture widening, panel
  movement, target movement, or grasp redesign.

Do not import V9.8 face-window, lag, or ceiling-envelope validation.

Stop at the first failed stage. Do not relax the lattice or thresholds.

## Offline reconstruction, geometry search, and routing

- Independently reconstruct all eight baseline rows from their own manifests
  and retained qpos. Require at most 1 mm TCP residual and hash the baseline
  summary, each reconstruction NPZ path and SHA-256, row artifacts, actual
  V5 scene XML, contract, and implementation inputs.
- Store per clean cell the canonical pregrasp/grasp TCP poses and the seven
  arm joints at the final grasp control step. Baseline and fresh guard values
  must agree before V9.9 work continues.
- Search this fixed top-flush box lattice:
  - center x: 0.58–0.90 m in 0.02 m steps;
  - half x: 0.03, 0.05, 0.07, or 0.09 m;
  - center y: −0.12–0.12 m in 0.02 m steps;
  - half y: 0.04–0.18 m in 0.02 m steps;
  - bottom z: 1.10–1.25 m in 0.025 m steps;
  - top z exactly 1.515 m;
  - reject any box outside x [0.58, 1.36], live aperture y bounds, or the
    enclosure.
- Reject a candidate unless every clean cell demonstrates:
  - the stock retained qpos intersects the inserted pendant during both
    inbound transit and loaded outbound transit, proving that both detours
    are necessary. AABB overlap is only a broad-phase screen; exact
    intersection uses the hardened `true_distance` / GJK instrument on
    retained qpos (no `env.step`);
  - the fixed pregrasp, grasp, gripper-close/settle, and initial lift window
    has at least 25 mm *exact* pendant clearance, not an AABB gap;
  - outbound clearance includes the carried target’s collision geoms;
  - no initial-state collision or panel, target, clutter, aperture, or grasp
    modification is needed.

V9.9 full-route lane primitive, after existing panel/vessel route
composition:

- Plan inbound and outbound independently, always using left-panel → −y and
  right-panel → +y.
- Search absolute lane y on a 10 mm grid inside ±(aperture_width/2 − 0.02).
- Search a common slab padding of 80, 100, 120, or 140 mm; order entry and
  exit according to travel direction.
- Preserve the stock route’s interpolated orientation at entry and exit;
  introduce no extra orientation change.
- Rejoin the unchanged pregrasp endpoint before the stock grasp segment, and
  begin outbound rerouting only after the unchanged lift endpoint.
- Densify every segment to at most 5 mm translation or 2° rotation. Solve IK
  sequentially, seeding each sample from the preceding solution.
- Use every collision-enabled robot geom for clearance; add carried-target
  geoms on outbound.
- Require at least 25 mm nominal pendant clearance and no intersection with
  any other strict non-target environment geom.
- Evaluate all eight corner combinations of ±5 mm lane-y, entry-x, and
  exit-x perturbations; require at least 20 mm pendant clearance in every
  combination.
- Within the physical pendant x slab, compare against the densified stock
  traversal at the same monotonic x coordinate and require
  `abs(planned_y − stock_y) ≥ 0.05 m` at every sample.

Use a present, present-repeat, and pendant-parked render on planned qpos for
signal screening. Rank by worst-cell changed-value fraction so routes with
more samples do not win merely by being longer.

Select at most two distinct candidates:

- signal: highest worst-cell changed-value fraction, then changed sensors,
  robust clearance, and smaller volume.
- clearance: highest minimum robust clearance, then changed-value fraction
  and changed sensors.
- If both rankings select one geometry, use the next distinct
  clearance-ranked geometry.
- Stop if no candidate qualifies; do not relax the lattice or thresholds.

Do not route V9.8 through the new planner.

## Runtime gates and selection

1. Regression guard: fresh eight-row baseline must remain 6/8 clean with all
   eight outcomes matching the authoritative raw smoke.
2. Paired eight screens: run signal and clearance on the same eight rows.
   - All six baseline-clean rows must remain task-successful and clean.
   - Both inbound and outbound detours must execute in the correct direction
     and reach at least 0.05 m.
   - No fallback, clipping, fixture contact, other strict contact, or
     clutter-stability event is allowed.
   - Pregrasp and grasp TCP endpoints must remain unchanged.
   - At the final grasp boundary, every arm joint must be within 1 mrad of
     that cell’s canonical baseline posture.
3. Selection: choose signal if it passes; otherwise choose clearance. Stop
   if neither passes; do not add a third candidate.
4. Live-qpos close-out: reconstruct the selected paired trajectories.
   Require at least 20 mm actual robot/target-to-pendant clearance and
   confirm executed detour, terminal posture, fixture pose, and contact
   classification before authorizing the larger gate.
5. Pinned 24-row gate: four exact deterministic repeats of each of the six
   cells. Require at least 20/24 clean successes and at least one clean
   result from every cell. Require zero infrastructure failures, correct
   telemetry on every complete row, no fixture movement, and no strict
   contact on clean rows. This is a six-cell engineering gate, not a
   multi-seed robustness estimate.
6. V9.9 S2b: select the lowest-index clean repeat from each cell and render
   retained live qpos in present, present-repeat, and pendant-parked worlds.
   Use inbound and outbound V9.9 pendant approach/pass/exit windows. Apply
   the frozen per-value threshold and all existing sensor-count,
   changed-value, responder-link, and side-imbalance floors independently
   to every cell.
7. Human review: create six videos from the 24-row gate—up to three diverse
   failures, then diverse successes until six videos exist, with at least
   three successes. Review requires visible bilateral rerouting, stationary
   mounting, unchanged grasp behavior, no robot/cup-to-pendant contact, and
   no clutter disturbance. Only the user can approve this stage.

## Interfaces, artifacts, and verification

- Environment: `pact_place_corridor_v9_9_pendant`
- Sampler: `PactPlaceCorridorV99PendantSampler`
- Telemetry: detour, nominal/robust/live clearance, IK status, terminal-posture
  comparison, and fallback/clipping flags.

Immutable, self-hashed outputs live under `diagnostics_output/pact_place_v99_*`.
Every artifact carries `authorizes_collection: false` and explicit stage
fields such as `authorizes_paired_screen`, `authorizes_24_row_gate`,
`authorizes_s2b`, and `authorizes_human_review`.

Close-out artifacts:

- `diagnostics_output/pact_place_v99_baseline_reconstruction/reconstruction.json`
  (`ae2964c41ebd85ce61ac4d703d809a4198759a4c116728daa459d55d796eff1c`)
- `diagnostics_output/pact_place_v99_siting/snapshots/snapshots.json`
  (`0d6e61baeab68e645d6e04ce54a2406bc588f05e540b7441463f7c1e06af8465`),
  schema `pact_place_v9_9_exact_snapshots_v2`, dtype float64
- `diagnostics_output/pact_place_v99_siting/siting.json`
  (`71389801e8ba0663af68629234e5d767478fb7bbf452a1b333ac7963064a774f`),
  schema `pact_place_v9_9_siting_v3`, `v99_closed: true`
- `diagnostics_output/pact_place_v99_siting/exact_witnesses.npz`
  (`c8f924700360d2e9169e79a7850f2f5bca375e468dfb4787319060bbcf9a817e`)
