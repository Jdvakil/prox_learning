# PACT Place V10.11c — 33% taller primitive validation

V10.11c is a height-only successor to V10.11b. Every one of the three
primitive clutter heights is multiplied by exactly 1.33:

| slot | shape | V10.11b | V10.11c |
|---|---|---:|---:|
| 01 | route-bearing cylinder | 0.24500 m | 0.32585 m |
| 08 | near-target cylinder | 0.18000 m | 0.23940 m |
| 09 | near-target box | 0.18000 m | 0.23940 m |

No XY footprint, XY placement rule, cup or target dimension, mesh object,
route, speed, scene, sensor, success criterion, contact taxonomy, or task
horizon changes. The shapes remain ordinary movable clutter and their sensor
visibility is measured through the unmodified 40-camera raw proximity path.

The frozen sequence is: targeted tests; paired V10.11b/V10.11c raw-depth
comparison at identical retained robot qpos and sampled XY layouts; complete
96-layout settle/contact/full-IK preflight; then a fresh owner packet containing
the first three strict-clean successes and first three natural failures in the
registered order. Stop on any failed gate. This task authorizes no collection,
training, evaluation, or Phase-0 work and must not overwrite V10.11 or V10.11b.
