# V10 Connected Compound Pendant Qualification Plan

Status: **closed at routing under two separately registered primitives.**
Two-lobe exact geometry (siting v2) stands: 150,288 full-environment
survivors, 1,779 unique union AABBs. Route-v1 remains a valid scoped
historical result: **no route survivor under the registered V9.9
contiguous-group-freeze route primitive.** Route-v2 is a separately
registered endpoint-only amendment; it does not overwrite or relabel
route-v1. Endpoint-only freeze restored geometry (17,826 of 21,348
union×cell×direction evaluations; 1,032 of 1,779 unions on all six cells
and both directions) but found **zero** route-feasible unions and **zero**
route-feasible morphologies after exhaustive inbound sequential IK and
strict-environment nonintersection. Every inbound geometry-feasible
lane/padding identity that reached IK passed IK and failed environment.
Outbound, robust corners, and pendant clearance were not reached.
`stop_reason: no_route_v2_ik_clearance_survivor`. Three-lobe search was
not run and is not authorized. Signal screening was not run. Paired
screens, episodes, collection, training, and evaluation remain
unauthorized. V9.9 remains permanently closed. Its scoped conclusion is
unchanged: **no survivor in the registered fixed rectangular-box lattice.**
V10 does not reopen that lattice, edit V9.9 artifacts, or import V9.8 lag,
face-window, or ceiling-envelope validation.

The 8,554,036-row v1 catalog remains a superseded robot/target prefilter,
not a full exact-survivor set. Independently reproduced panel-clear count:
150,288 rows (1,779 union AABBs). Planning-probe v1 is retained only as
`probe_v1_invalid_panel_overlap`. Planning-probe v2 is the trust anchor and
passes robot/target exact plus panel/clutter/environment checks on all six
cells.

V10 replaces V9.9’s single rectangular prism with one static,
side-independent rigid assembly of two rectangular paddles, two ceiling
stems, and a crossbar. It changes only the registered shape family. It
preserves V9.9’s frozen six-cell V9.5 layout, exact safety thresholds,
bilateral stock-route necessity, route robustness, sensing floors, and
staged episode authorization. Success ends at environment qualification and
human review.

Success supports only:

> Under the pinned six-cell V9.5 engineering layout, one fixed, identical
> connected compound pendant can force collision-free lateral detours of at
> least 5 cm in both traversal directions while remaining bilaterally
> visible to the proximity skin.

It does not establish multi-seed robustness, authorize collection, or
establish PACT superiority. Every artifact carries `authorizes_collection:
false`, `authorizes_training: false`, and `authorizes_eval: false`. Human
approval may set `environment_qualified: true` only.

## Frozen boundary

Freeze, unchanged from V9.9:

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
  assembly slab.
- Exact 25 mm robot-and-target clearance on the pregrasp / grasp / close /
  settle / lift window, and 20 mm at every ±5 mm perturbation corner.
- No collection, training, ACT/PACT evaluation, aperture widening, panel
  movement, target movement, or grasp redesign.

Do not import V9.8 face-window, lag, or ceiling-envelope validation.
Do not modify V9.9 reconstruction, snapshots, siting, or witnesses.

Stop at the first failed stage. Do not relax the lattice or thresholds.
If a two-lobe assembly survives exact geometry but later fails routing or
sensing, stop V10; do not escalate to three lobes.

## Interfaces and geometry contract

- Environment: `pact_place_corridor_v10_compound_pendant`
- Sampler: `PactPlaceCorridorV10CompoundPendantSampler`
- Contract: `pact_place_v10_compound_pendant_v1`
- Scene fork: `pact_place_corridor_v10.xml`, which includes V5 unchanged and
  adds one new `pact_clutter_mount_v10` mocap body with seven predeclared
  box geoms (three lobes, three stems, one crossbar).

Manifest field `pact_v10_pendant_assembly` contains topology (`two_lobe` or
`three_lobe`), a deterministic assembly ID, active component records
(name, role, world center, half-extents), `identical_on_both_panel_sides:
true`, and `active_on: [inbound_empty, outbound_loaded]`.

Active components are visible and collision-enabled. Unused third-lobe
slots are hidden, collision-disabled, and excluded from hazards and
sensing. The sampler parks the legacy V3 ceiling fixture and all unused
mounts. The V10 parked control moves the entire new assembly out of the
scene.

Permit only the designed crossbar-to-`hood_top` attachment at z=1.515.
Reject any other assembly/environment overlap.

Registered lobe lattice:

- Common lobe/stem center x: 0.60–0.90 m in 0.02 m steps.
- Lobe half x: 0.01, 0.02, or 0.03 m.
- Lobe center y: signed 0.12–0.30 m in 0.02 m steps.
- Lobe half y: 0.02, 0.04, 0.06, or 0.08 m.
- Lobe center z: 0.86–0.98 m in 0.02 m steps.
- Lobe half z: 0.02, 0.04, 0.06, or 0.08 m.
- Require bottom z ≥ 0.82 m, top z ≤ 1.10 m, negative lobes wholly below
  y=−0.08 m, positive lobes wholly above y=+0.08 m, and every component
  inside the aperture/enclosure (x [0.58, 1.36], live aperture y).

Each 6 mm square stem is centered at the lobe’s outward y face and runs
from the lobe top to z=1.505 m. The 10 mm-high crossbar is ceiling-flush
and spans the active stems. Lobes share center x so the axis-aligned
crossbar forms one connected assembly.

Planning-probe-v1 (invalid; retain only as `probe_v1_invalid_panel_overlap`):

- Negative lobe: center [0.60, −0.26, 0.90], half [0.02, 0.02, 0.04].
- Positive lobe: center [0.60, 0.20, 0.90], half [0.02, 0.08, 0.04].
- Stems at y=−0.28 and +0.28 m, with the derived ceiling crossbar.
- The retained-qpos probe found lobe contact on all twelve cell/traversal
  requirements and no component below the 25 mm grasp-window threshold, but
  both lobes and their stems intersect the opposite-side intrusion panels.
  Do not treat it as a successful environment-clear fixture.

Planning-probe-v2 (trust anchor; enumerator regression fixture; not auto-selected):

- Negative lobe: center [0.70, −0.18, 0.86], half [0.01, 0.04, 0.04].
- Positive lobe: center [0.70, 0.22, 0.86], half [0.01, 0.02, 0.02].
- Stems and crossbar derived from the unchanged V10 contract
  (stems at y=−0.22 and +0.24 m).
- Must pass robot/target exact predicates plus panel/clutter/environment
  checks, including live-scene exact evaluation, on all six cells.

## Staged implementation and stop rules

1. Freeze inputs and provenance.
   Verify the V9.9 reconstruction, six float64 snapshots, V9.5 source rows,
   V5 scene, and their recorded hashes before loading. Leave V9.9
   implementation and artifacts byte-for-byte unchanged. V10 runners
   establish their own asset environment and sort every result canonically
   before hashing. No reconstruction episodes and no `env.step`.

2. Exact compound-geometry search.
   Tight AABBs are only for lossless broad-phase rejection or clearance
   certification, except that AABB overlap of two axis-aligned boxes (posed
   intrusion panels vs pendant components) is the exact panel-intersection
   predicate. AABB overlap is never an exact clearance for meshes.
   Score every retained candidate on every predicate and all six clean cells
   without early termination.

   Require exact **lobe** contact on stock inbound and loaded outbound
   traversal for all six cells. Stem or crossbar contact cannot satisfy the
   twelve necessity bits.

   Require every active component to have no initial robot/target contact,
   at least 25 mm exact robot-and-target clearance throughout pregrasp,
   grasp, close/settle, and initial lift, and no forbidden panel, clutter,
   aperture, enclosure, or target overlap (except the designed
   crossbar-to-`hood_top` attachment). Panel AABB overlap is the exact
   panel-intersection predicate because panels and pendant components are
   axis-aligned boxes. Tight AABBs may only reject or certify clearance;
   GJK/`true_distance` is required wherever AABBs do not prove separation.

   Form two-lobe assemblies from one negative and one positive branch
   sharing center x. Stream all exact set-cover pairs and test their
   derived stems and crossbar. Only if there are zero fully connected
   exact two-lobe survivors, repeat with three distinct lobes from the same
   lattice, sharing center x and containing at least one lobe on each side.
   Do not relax any threshold. If a two-lobe assembly survives exact
   geometry but later fails routing or sensing, stop V10; do not escalate
   to three lobes.

   Write component-level contact and clearance witnesses to deterministic
   NPZ/JSON artifacts. Stop with `no_exact_compound_survivor` if both
   registered families are empty.

3. Routing, sensing, and candidate selection.
   Generalize the V9.9 route primitive from one box to the union of active
   assembly components. Preserve left panel → −y and right panel → +y, the
   10 mm lane grid, 80/100/120/140 mm slab padding, ≤5 mm translation and
   ≤2° rotation densification, sequential IK seeded from the preceding
   sample, unchanged pregrasp/grasp and lift endpoints, ≥5 cm detour
   against stock traversal, and ≥25 mm nominal / ≥20 mm clearance at all
   eight ±5 mm perturbation corners.

   Check every robot collision geom, the carried target on outbound, every
   pendant component, and all strict environment geoms. Pendant components
   use the 25 mm / 20 mm robust floors; other environment geoms use
   no-intersection only. Union clustering may cache sequential-IK qpos;
   each morphology keeps its own component geometry for clearance, sensing,
   volume, and ranking. If too many morphologies remain for complete signal
   screening, stop for `too_many_morphologies_for_signal_screen`.
   Signal-screen every route survivor using present, present-repeat, and
   whole-assembly-parked renders on planned qpos. Preserve the per-cell
   sensing floors.

   Rank at most two distinct candidates: signal winner then clearance
   winner, breaking remaining ties with smaller total assembly volume. If
   both ranks choose the same assembly and no distinct runner-up exists,
   screen only one. Stop before episodes if no route-and-signal survivor
   exists.

4. Runtime qualification.
   First rerun the V9.5 eight-row guard and require the authoritative
   row-for-row 6/8 result. Run paired eight-row screens for the selected
   candidate(s) with the frozen six-cell cleanliness, detour, contact,
   clutter-stability, TCP endpoint, and 1 mrad grasp-posture rules. F3
   rows remain regression-only but must have no infrastructure failure,
   pendant contact, fallback, clipping, or telemetry failure. Require zero
   pendant contact before authorizing the larger gate.

   Run the unchanged pinned 24-row engineering gate, then S2b on the
   lowest-index clean repeat per cell, then six review videos (up to three
   diverse failures, then diverse successes, at least three successes).
   Human review checks visible bilateral rerouting, a stationary connected
   mount, unchanged grasp behavior, and no pendant or clutter contact.

   Human approval marks only `environment_qualified: true`. Every V10
   artifact retains `authorizes_collection`, `authorizes_training`, and
   `authorizes_eval: false`.

## Artifacts

Immutable, self-hashed outputs live under `diagnostics_output/pact_place_v10_*`.
Every artifact carries `authorizes_collection: false` and explicit stage
fields. At every failed stage, write a self-hashed causal close-out with
the stop reason, counts, witnesses, and all downstream authorizations
false.

V10 reuses, but does not rewrite, V9.9 reconstruction and snapshots:

- `diagnostics_output/pact_place_v99_baseline_reconstruction/reconstruction.json`
- `diagnostics_output/pact_place_v99_siting/snapshots/snapshots.json`

Superseded V10 siting v1 (`diagnostics_output/pact_place_v10_siting/siting.json`,
payload SHA `923c9380319b343e43e55f018080995db8a4b59e5a5f3cbd7f5d1a3be79d0eb6`)
is a robot/target prefilter only. It scored robot/target necessity, grasp
clearance, initial robot clearance, stems, crossbars, and hardcoded
enclosure boxes. It did **not** check posed intrusion panels or clutter.
Do not treat its **8,554,036** rows as full V10 exact survivors. Catalog
`diagnostics_output/pact_place_v10_siting/exact_survivors.npz` SHA
`63369af3552bbb806a61fea97d281011374ee25bb375004876704b920b6f3443`.
The enumerator-era planning probe (negative `[0.60, -0.26, 0.90]` /
`[0.02, 0.02, 0.04]`, positive `[0.60, 0.20, 0.90]` / `[0.02, 0.08, 0.04]`)
intersects the opposite-side panels and is retained only as
`probe_v1_invalid_panel_overlap`.

Corrected V10 siting v2 (`diagnostics_output/pact_place_v10_siting_v2/siting.json`,
schema `pact_place_v10_siting_v2`, payload SHA
`2e0b2a56bd4c22ecc920927dc149adf9c1bbc0d1d3ccbd3ee433ea450b187c1c`):

| Predicate | Count |
|---|---|
| Robot/target prefilter (v1 catalog SHA bound) | 8,554,036 |
| Panel-clear (exact AABB; independently reproduced) | 150,288 |
| Panel-clear unique union AABBs | 1,779 |
| Full environment exact survivors (panels, clutter, sash/jamb/aperture, hood/enclosure, bench, other strict scene geoms, initial target; only crossbar–`hood_top` attachment allowed) | 150,288 |
| Corrected unique union AABBs | 1,779 |
| Rejected by panel | 8,403,748 |
| Rejected by non-panel environment after panel | 0 |
| Rejected by initial target after environment | 0 |

Compact catalog `exact_survivors_v2.npz` SHA
`b84e19bf269c39cd052551639c22d4cbb5b4348eaf6188663fae0659af824d6e`
stores numeric lobe keys only (no `assembly_ids` string array). Topology and
assembly IDs are derived lazily. Planning-probe v2 (negative
`[0.70, -0.18, 0.86]` / `[0.01, 0.04, 0.04]`, positive `[0.70, 0.22, 0.86]` /
`[0.01, 0.02, 0.02]`) is the trust anchor: robot/target exact, panel-clear,
and live-scene environment-clear on all six cells; cached and direct
evaluations agree. `three_lobe.searched: false`.
`stop_reason: exact_survivors_route_not_run`. `routing_run: false`.
`physics_stepped: false`. `episodes_run: false`. All authorizations false.
Verified scene hashes: V5
`5ac1ebd3e04f0bf509f6b8e11f0d086ac8c43bd550349762aba6c4129aebd61c`;
V10
`360b1407a01d1447d8b440ade3115866399a1db09efc76321016aa3c04eaddf7`.
V9.9 reconstruction SHA
`ae2964c41ebd85ce61ac4d703d809a4198759a4c116728daa459d55d796eff1c` and
snapshot SHA
`0d6e61baeab68e645d6e04ce54a2406bc588f05e540b7441463f7c1e06af8465`
are unchanged.

Routing close-out (`diagnostics_output/pact_place_v10_route/route.json`,
schema `pact_place_v10_route_v1`, payload SHA
`c0f1b35084d6950a88531c45e6805b06437add31c82ca5fd68bb5da4f5de3ff7`).
Inputs were re-verified before search: 150,288 rows, 1,779 unions, siting-v2
payload `2e0b2a56bd4c22ecc920927dc149adf9c1bbc0d1d3ccbd3ee433ea450b187c1c`,
catalog
`b84e19bf269c39cd052551639c22d4cbb5b4348eaf6188663fae0659af824d6e`. All six
F0/F1/F2 cells, inbound and loaded outbound, registered lane-y grid and
80/100/120/140 mm paddings, open-side direction, ≥5 cm detour, ≤5 mm / ≤2°
densification. Union clustering cached lane/IK identity; no representative
union box was used for component clearance.

| Route predicate | Count |
|---|---|
| Unions attempted | 1,779 |
| Union×cell×direction evaluations | 21,348 |
| Rejected by lane construction | 0 |
| Rejected by detour / clipping / wrong-way | 21,348 |
| Rejected by nominal IK | 0 |
| Rejected by strict-environment contact | 0 |
| Rejected by nominal pendant clearance | 0 |
| Rejected by robust IK or any robust corner | 0 |
| Route-surviving unions | 0 |
| Route-surviving morphologies | 0 |

Cause: every unique union AABB has x roughly in [0.68, 0.77] m. The
registered freeze-final (inbound) / freeze-start (outbound) rule skips
rewriting any in-slab group that contains a frozen pregrasp or grasp
endpoint. On all six cells those frozen TCP x values lie inside every
union's padded slab for all four paddings, so the stock path is never
rewritten and the ≥5 cm detour fails. Sequential IK and live clearance were
therefore not evaluated. `all_eight_corners_evaluated_for_admitted_routes`
is vacuously true (zero admitted routes). `routing_run: true`.
`physics_stepped: false`. `episodes_run: false`. `signal_screen_run: false`.
`three_lobe_searched: false`. `v10_closed: true`. `stop_reason:
no_two_lobe_route_survivor`. Compact mask
`route_morphology_mask.npz` SHA
`6c2609d11dccb69537970fbd7decc2e1b4efc8c2b83275fe3ef65275728a8274`.
`REGISTERED_COMPLETE_SIGNAL_SCREEN_LIMIT` remains `None`; no post-hoc
shortlist was created. The complete-signal-screen gate was not reached
because the route-surviving population is empty. All authorizations false.

Route-v1 is not an error. It answers only the scoped question above. A
separately registered amendment then asked whether freezing only the
required endpoint, rather than suppressing its entire in-slab group,
restores feasible routing (`diagnostics_output/pact_place_v10_route_v2/`,
schema `pact_place_v10_route_v2_endpoint_only`). V9.9, V10 siting v1/v2,
route-v1, and all geometry/aperture/lane/padding/detour/clearance
thresholds were left unchanged. The new primitive
(`apply_constant_lane_endpoint_only`) rewrites every eligible in-slab
sample except exactly index 0 when `freeze_start=True` and exactly the
final sample when `freeze_final=True`. Stock orientation, 10 mm lane grid,
open-side direction, 80/100/120/140 mm paddings, ≥5 cm physical-slab
detour (including frozen endpoints), and ≤5 mm / ≤2° densification are
unchanged. Missing stock-x, discontinuities, clipping, and wrong-way still
fail closed.

Phase A independently reproduced the registered audit without hardcoding
the counts as results:

| Phase-A geometry | Count |
|---|---|
| Union×cell×direction evaluations | 21,348 |
| Evaluations with ≥1 endpoint-only geometry route | 17,826 |
| Unions with geometry on all six cells and both directions | 1,032 of 1,779 |
| Route identities generated | 1,923,772 |

Payload SHA
`48e643e6d6b768b3a2dba491c3199c859c8ba1287a75379e504a09b5fdefc74a`
(`geometry.json`). Mask SHA
`19978e65a7a239a543058919d127a4183f55c85143f79b89ed2e973290b9b509`.
Nominal IK, strict environment, robust corners, and pendant clearance were
recorded as `not_evaluated` with `evaluated_count: 0` /
`all_eight_corners_evaluated: not_applicable`.

Phase B ran only after that reproduction. Search enumerated every
geometry-feasible lane/padding identity per geometry-surviving union, cell,
and direction; did not stop at the first union-level IK/environment path
before morphology clearance; cached qpos sequences by complete route
identity; and failed a morphology only after its registered routes were
exhausted. Different morphologies and cells/directions were allowed to
select different identities. Because no inbound identity passed strict
environment, outbound IK was not reached (`not_evaluated`).

| Phase-B predicate | Attempted | Passed | Failed | Not evaluated |
|---|---:|---:|---:|---:|
| Nominal sequential IK | 666,448 | 666,448 | 0 | 742,272 |
| Strict-environment nonintersection | 666,448 | 0 | 666,448 | 742,272 |
| Robust eight-corner routes | 0 | 0 | 0 | 1,408,720 |
| Morphology-specific pendant clearance | 0 | 0 | 0 | 0 |

Corners evaluated: 0. `all_eight_corners_evaluated: not_applicable`.
Alternative-route recoveries: 0. Exhausted morphology×cell×direction
events: 1,293,696. Cache: 666,448 entries, 666,448 misses, 0 hits, 0 qpos
reuses (each identity was evaluated once; entries retain the IK/environment
result, not booleans-only without an identity key). Route-surviving unions:
0. Route-surviving morphologies: 0. Limiting cause: every inbound
geometry-feasible identity passed sequential IK and failed strict
environment at `CONTACT_DISTANCE_M`. Robust corners and 25/20 mm pendant
clearance were never reached. `routing_run: true`. `physics_stepped:
false`. `episodes_run: false`. `signal_screen_run: false`.
`three_lobe_searched: false`. `v10_closed: true`. `stop_reason:
no_route_v2_ik_clearance_survivor`. Payload SHA
`e311ba01c77c14b3a930be8dd9d4d40e9de483710521f8662d2e3a55357f71e1`
(`route.json`). Mask SHA
`a20d2e6c0ced5e0807103b3f0f7050a798a5c89c447945fe9898460f2e286bb1`.
`REGISTERED_COMPLETE_SIGNAL_SCREEN_LIMIT` remains `None`; no signal-screen
limit or shortlist was created. All authorizations false.

Route-v2 remains a historical result of its flawed scalar environment
predicate and is **not** cited as physical infeasibility. The follow-on
empirical qualification is
[`docs/PACT_PLACE_V101_EMPIRICAL_QUALIFICATION_PLAN.md`](docs/PACT_PLACE_V101_EMPIRICAL_QUALIFICATION_PLAN.md).
V10 siting and route artifacts were not overwritten.
