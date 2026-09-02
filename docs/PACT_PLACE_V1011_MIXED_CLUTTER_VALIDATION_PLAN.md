# V10.11 mixed mesh/primitive clutter validation

Status: **stopped at owner review, as registered**. The 96-layout preflight
passed and the six-video packet contains three strict-clean successes and three
natural failures. Collection, conversion, training, evaluation, and Phase 0
remain out of scope and unauthorized.

## Frozen environment

V10.11 extends the V10.10/V10.7 place corridor with six active table-clutter
bodies and four parked palette entries:

| slot | active object | representation | role | dimensions |
|---|---|---|---|---|
| 01 | cylinder | runtime MuJoCo primitive | outbound vessel | radius 0.045 m, height 0.220 m |
| 03 | `Plate_10` | mesh | decor | unchanged asset dimensions |
| 04 | `Plate_22` | mesh | decor | unchanged asset dimensions |
| 06 | `Soap_Bottle_11` | mesh | inbound vessel | unchanged asset dimensions |
| 08 | cylinder | runtime MuJoCo primitive | near-target decor | radius 0.035 m, height 0.100 m |
| 09 | box | runtime MuJoCo primitive | near-target decor | 0.070 x 0.070 x 0.100 m |

The two near-target primitives are sampled uniformly by area in a target-relative
annular sector, with a 20 mm footprint margin and at most 64 deterministic
candidates. The target is sampled once. Runtime primitives are free bodies
named under `pact_clutter_`, so the normal contact audit classifies them as
clutter. No cup, mesh-asset, pendant, route, sensor, or scene-XML dimensions
were changed.

The implementation corrects three assumptions in the original design without
changing its intended geometry: vessel categories use the runtime-approved
tokens, and target footprint rejection uses the compiled collision AABB rather
than metadata whose planar axes become invalid after the inherited target
rotation. A first implementation also used the box half-width as its planar
radius; the registered 20 mm margin requires the rotation-invariant
half-diagonal. That packet was superseded before owner review and both gates
were rerun after the correction. No object dimension changed.

## Stage 0: implementation and regression tests

The V10.11 suite covers every palette clause, the six-active/four-parked split,
primitive dimensions and namespace, target sampling exactly once, deterministic
uniform-area placement, non-overlap, route metrics, identity hashing, stream
disjointness, and contact classification. The final focused historical sweep
passed 102 tests across V10.11, V10.10 infrastructure, V10.7, V9.8, V9.5, and
the contact/failure audit.

A broader historical-state sweep also exposed three expected assertions that
are no longer true: V10.7's approval and Phase-0 artifacts now exist, and the
shared sampler implementation intentionally changed for V10.11. Those are
sealed-history/provenance assertions, not V10.11 behavioral regressions; no
V10.7 artifact was altered to silence them.

## Stage 1: registered 96-layout preflight

Artifact:
`diagnostics_output/pact_place_v1011_preflight/preflight.json`

| check | result |
|---|---:|
| registered rows | 96/96 |
| containment and initial-contact gate | pass |
| settling gate | pass |
| compiled primitive type, size, free-joint and namespace | pass |
| vessel route/panel corridor metrics | pass |
| complete expert route / IK construction | pass |

All rows found a valid deterministic pre-boundary draw by retry 7; retries did
not create or step an episode. Maximum accepted settled XY drift was 4.73 mm,
maximum linear speed 2.47 mm/s, and maximum angular speed 0.0185 rad/s. Measured
primitive-height ranges were 0.220001--0.220031 m for slot 01,
0.100000--0.100002 m for slot 08, and 0.100000 m for slot 09. The target AABB
was measured rather than asserted from an untraceable dimension. The minimum
registered target/object radial surface gaps were 20.45 mm for the cylinder
and 21.88 mm for the square box, both above the 20 mm floor.

Payload SHA-256:
`1fbeafb185643bb530b5b9d725dd0df43526951701932f170f10df65bb24861c`.
Raw SHA-256:
`7e7143c906f32916784816507ac0fc83965b0f84fa4c78e672cc293ed80569d1`.

## Stage 2: owner-review packet

Artifact:
`diagnostics_output/pact_place_v1011_review/review_manifest.json`.
The generator completed 12/12 rows with no sampling or infrastructure failure,
then selected the first three strict-clean and first three complete non-clean
rows in frozen role order. Every retained layout reconstructed with the same
layout hash, and all six full-trajectory, two-view MP4s decode at the true
control rate.

| role | side / pose | outcome | evidence |
|---:|---|---|---|
| 2 | left / pos5 | strict-clean task success | no clutter or stability event |
| 5 | right / pos5 | strict-clean task success | no clutter or stability event |
| 6 | left / neg5 | strict-clean task success | no clutter or stability event |
| 0 | left / neg5 | non-clean task success | 58 clutter contact entries |
| 1 | left / center | non-clean task success | `Soap_Bottle_11` stability event |
| 3 | right / neg5 | task failure | primitive-cylinder stability event |

Manifest payload SHA-256:
`a742c7b4222f3370393c3da6887d3b027c1258a2a342f0396cf34407e9c2440b`.
Raw SHA-256:
`743fe2f80425d76de2958a121c9b87d327efe115cbe2f7dc6529c273fd3a7b53`.

## Decision boundary

The packet is eligible for owner review. Owner acceptance validates only the
environment and ends this scope. It does not authorize collection, conversion,
training, evaluation, Phase 0, or any downstream claim. The separate sensor
visibility limitation is documented in `docs/PACT_PLACE_SKIN_VISIBILITY.md`.
