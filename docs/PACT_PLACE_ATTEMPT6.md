# PACT pick-and-place Phase 0, attempt 6

This is **not** a sixth official screen. v5 already passed Phase 0 at 22/24.
Attempt 6 was a chosen cluttered observation distribution: four fixed immovable
`pact_clutter_*` mocap boxes on the shelf beside the cup. The A0 rollout probe
required **zero** expert `pact_clutter` contact. No candidate on the declared
grid met that bar, so Phase 0 was **not frozen and not run**.

The decision token is `PACT_PLACE_CORRIDOR_PHASE0_NOT_RUN`.
The ledger is [`PACT_PLACE_CORRIDOR_GATE_V6.md`](PACT_PLACE_CORRIDOR_GATE_V6.md).

v5 remains `PACT_PLACE_CORRIDOR_PHASE0_PASS`. Its artifacts are unchanged.

## The five screens, plus this stop

| Screen | Master seed | Config SHA-256 | Clean | Decision |
|---|---:|---|---:|---|
| v1 | `2026081801` | `46dff849…559e40` | 18/24 | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |
| v2 | `2026081901` | `a0f30725…201c31` | 18/24 | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |
| v3 | `2026082001` | `acd7ced5…6694f` | 18/24 | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |
| v4 | `2026082101` | `fe45435d…9cdc65` | 15/24 | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |
| v5 | `2026082201` | `bd47f1c97d2815657211085590657f5211ca847b776f6039c9617f990da9c1f1` | **22/24** | `PACT_PLACE_CORRIDOR_PHASE0_PASS` |
| v6 clutter | not frozen | no `configs/pact_place_corridor_v6.json` | n/a | `PACT_PLACE_CORRIDOR_PHASE0_NOT_RUN` |

Join rows on `(config_sha256, role_index)`, never `episode_id`.
v6 clean-success would have been stricter than v5 (adds zero clutter). Those
counts are not comparable, and v6 has no official count.

## Named changes that were built, then blocked

Thresholds were not changed. `gripper_empty_threshold` stayed **0.002 m**.
`MIN_CLEAN_SUCCESSES` stayed **20**. `RELEASE_CLEARANCE_M` stayed **0.005**.
Grasps were not filtered. The tray and target were not moved. The corridor,
aperture, and panel were not widened. Boxes were not shrunk after the probes.

1. **Fork `pact_place_corridor_v3.xml` of v2.** Add only `pact_clutter_{l0,l1,r0,r1}`
   mocap boxes. v1 and v2 XML hashes unchanged.
2. **`PactPlaceCorridorV3Sampler`.** Nominal slots plus ±0.02 m jitter from
   `task_seed_u64`, not from `intrusion_side`. Clutter is not appended to
   `obstacle_aabbs`.
3. **Place-specific `clutter` class** in `pact_place_contact_audit.py`, scored
   before the legacy fallthrough, including cup-versus-clutter pairs. Shared
   `pact_contact_audit.py` is unmodified.
4. **A0 static sweep**, 30 candidate sets, hashed
   `e34038b9e4a32e5b84729f62d5dc1a851b40c3ad2aa11b6d79bccc461c3526ae`.
   12/30 footprint-ok and IK-ok. Closest eligible: set 13, height 0.06 m,
   `|y| = 0.22`. Farthest eligible: set 14, same height, `|y| = 0.28`.
   `|y| = 0.15` fails the 4 cm target-envelope gap.

## A0 probes

Both probes used `--row-limit 8`, role `diagnostic_not_a_gate`, and the live
attempt-6 row seeds. They are not gates. `summarize()` still divides by N=24,
so the process exits 1 with `reconciled: false`; all eight requested rows
completed.

| Probe | A0 set | `|y|` | Config SHA-256 | Clean | Task | Clutter episodes | Phase |
|---|---:|---:|---|---:|---:|---:|---|
| closest | 13 | 0.22 | `0f4d8058…b9778d8` | 1/8 | 7/8 | **6/8** | outbound only |
| farthest | 14 | 0.28 | `eea68143…7a8430be` | 4/8 | 7/8 | **3/8** | outbound only |

Inbound clutter, placement clutter, hazard, other-environment, and tray
contact outside placement were **zero** on both probes. Grasp is not the
blocker: 7/8 rows grasp and place. Row 7 is an `outbound_approach` /
`ik_cascade` miss, the same retrieval-reach class as v5 row 3.

The expert bows **away from the panel**, so a left intrusion sweeps toward the
right-side boxes and vice versa. Symmetric shelf clutter therefore sits in the
outbound carry corridor. Moving from `|y| = 0.22` to `|y| = 0.28` cut clutter
episodes from 6 to 3; it did not reach zero. There is no farther set on the
declared grid (`|y| ∈ {0.15, 0.22, 0.28}`). Height 0.10 m does not change the
horizontal sweep.

## Why this stops here

The plan required: if no candidate is both IK-feasible and rollout-clear,
**stop and report rather than shrinking the clutter**. That condition held
after the farthest eligible set. Freezing v6 and running 24 official rows
would have been a known-dirty expert, not a feasibility gate.

The 19–22 prediction was drafted in the live builder and was never locked
into a frozen config, because freeze is blocked until a zero-clutter probe.

Collection is not authorized. Do not reuse the v5 screen-harness collection
path. v5's 152 JSON `qpos` episodes remain the no-clutter arm and are still
not ACT HDF5.

## What is still true

- `pact_collision_corridor.xml` `f8c04b07…`
- `pact_place_corridor_v1.xml` `d853e27c…`
- `pact_place_corridor_v2.xml` `920860de…`
- planner hashes `9ee36978…` and `a7ee3570…`
- shared `pact_contact_audit.py` `f07aace3…`
- v5 contract `bd47f1c9…`, 22/24 PASS
