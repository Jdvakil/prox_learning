# PACT Place V10.11b — taller primitive-clutter validation

V10.11b is a height-only successor to the frozen V10.11 mixed-clutter review.
It exists because the owner liked the three primitive shapes but wanted them to
occupy more of the real robot-skin camera field of view.

## Frozen change

Only the Z dimensions of the three primitive clutter objects change:

| slot | shape | V10.11 height | V10.11b height |
|---|---|---:|---:|
| 01 | route-bearing cylinder | 0.220 m | 0.245 m |
| 08 | near-target cylinder | 0.100 m | 0.180 m |
| 09 | near-target box | 0.100 m | 0.180 m |

The XY footprints, target cup and its dimensions, three mesh clutter objects,
target-relative annulus, routes, speeds, pendant scenes, cameras, proximity
sensor implementation, contact taxonomy, task horizon, and success definition
are unchanged. All primitives remain ordinary movable clutter free bodies; no
visibility flag or sensor response is hard-coded.

## Required validation sequence

1. Unit-contract checks prove the three Z-only edits, unchanged footprints and
   mesh objects, a fresh seed stream, and registration on the V10.6 lane.
2. A paired offline visibility audit reconstructs the six frozen V10.11 review
   trajectories. It renders V10.11 and V10.11b through the real raw 40-camera
   proximity path at identical retained robot qpos and identical sampled XY
   layouts. The taller primitive bases are aligned to the parent bases. Both
   left and right rows must produce a nearer depth return above measured render
   noise. This evidence is diagnostic only and authorizes nothing downstream.
3. The complete 4-family × 2-side × 3-pose × 4-replicate preflight must pass
   96/96. It verifies the exact compiled primitive dimensions, workspace and
   initial-contact constraints, settle stability, route predicates, and full
   expert IK.
4. Only after steps 1–3 pass, generate a new packet containing the first three
   strict-clean successes and first three natural failures in frozen order.
   Render all six retained trajectories for owner review.

Stop on any failed gate. Do not overwrite V10.11 artifacts, change another
dimension, regenerate a different geometry post hoc, or authorize collection,
conversion, training, evaluation, or a Phase-0 gate from this review task.
