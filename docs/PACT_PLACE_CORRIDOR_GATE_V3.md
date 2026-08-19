# PACT pick-and-place corridor: Phase 0 attempt-3 screen (gate)

This is the **decision-bearing** fresh-seed screen after the two named repairs in
[`docs/PACT_PLACE_ATTEMPT3.md`](PACT_PLACE_ATTEMPT3.md). The original failed gate remains in
[`docs/PACT_PLACE_CORRIDOR_GATE.md`](PACT_PLACE_CORRIDOR_GATE.md). The expert-fix re-screen
remains in [`docs/PACT_PLACE_CORRIDOR_GATE_V2.md`](PACT_PLACE_CORRIDOR_GATE_V2.md). Neither
prior ledger is rewritten.

## Decision

Phase 0 failed a third time. Per the attempt-3 plan (diagnose, repair, run once; no fourth
iteration), work stops here: no demonstration collection, encoder update, policy training,
or learned-policy evaluation was run.

The gate required at least **20 clean successes in 24 fixed episodes**. A clean success means
upstream `PickAndPlaceTask` success with zero `hazard_bar` and zero `other_environment` contact
entries. Contact with `grasp_target` and the new `place_receptacle` class is expected and exempt.

## Measured Phase-0 results

| Measure | Result |
|---|---:|
| Reconciled rows | 24/24 |
| Clean pick-and-place successes | 18/24 (75.0%) |
| Ordinary task successes | 18/24 (75.0%) |
| Grasp phase successes (cup retrieved outside aperture) | 19/24 (79.2%) |
| Place phase successes | 18/24 (75.0%) |
| Place successes given grasp | 18/19 (94.7%) |
| Episodes with inbound hazard contact | 0/24 |
| Episodes with outbound hazard contact | 0/24 |
| Episodes with other-environment contact | 0/24 |
| Sampling failures | 0 |
| Infrastructure failures | 0 |
| Bow-fallback episodes | 0/24 |

## Failure localization

The six task failures split into **5 retrieval failures** and **1 placement/release failure
after successful retrieval**. All six had zero hazard-bar and zero other-environment contact.
The limiting factor was again manipulation reliability, not corridor collision.

| Failure class | Count | Row indices |
|---|---:|---|
| Cup not retrieved outside aperture | 5 | 2, 9, 12, 20, 21 |
| Retrieved, but placement/release incomplete | 1 | 6 |
| Retrieved, placed, but inbound hazard contact | 0 | — |

Mechanisms, from `result.json` / `endpoint_scalars` / `terminal_tracking` (new seed; these
are not the v2 row numbers):

| Row | Phase | What happened |
|---:|---|---|
| 12 | `pregrasp` | Never lifted. Gripper stayed open (`gripper_width_min_m` 84.0 mm). TCP tracking error 10.5 cm. |
| 20 | `lift` | Grasp-stability. Cup ended below its start (z 0.753 vs 0.794); gripper emptied. Same class as v2 row 11. |
| 2 | `outbound_approach` | Lifted 1 cm, then lost the cup (`gripper_width_min_m` 0). Tracking error 9.5 mm, under threshold. |
| 21 | `outbound_pass` | Lifted 1 cm, then lost the cup during the corridor pass (`gripper_width_min_m` 0). Cup abandoned at x 0.72. Same class as v2 row 3. |
| 9 | `outbound_approach` | Still holding (gripper 19.0 mm). TCP tracking error **12.58 cm**; actual z 0.9866 vs target 0.9062 (8.0 cm vertical) and a 9.4 cm y shortfall. Same class as v2 rows 4 and 17. |
| 6 | `placement_descent` | Grasp succeeded. Cup on the tray (`supported_by_receptacle` true) but `robot_contact` still true; place position error 20.9 mm. Same class as the original v1 A-mode handoff. |

`outbound_approach` was subdivided (row 9's planned waypoints are ~4 cm Cartesian pieces).
Shorter pieces did not keep the tracked pose in the carry plane on that instance. The plan's
second-ranked repair (IK continuity across the bow) was not applied; a fourth iteration is
not authorized.

## What the named repairs did on this seed

1. **Initial-observation robot–environment reject (Route A).** Inbound hazard contact was
   **0/24**. The rejector never fired: 0 of 8 pre-boundary retries were
   `initial_robot_environment_contact`. Those retries were IK / `HouseInvalidForTask`. One
   accepted observation (row 12) recorded exempt tray-lip contact. This seed did not redraw
   the v2 rows-6/12 packed overlap; the check is live, and hazard stayed at zero.
2. **Subdivide `outbound_approach` (Route B).** Did not convert the diagnosed tracking class.
   Row 9 is that class on the new seed.
3. **Record.** All 24 complete rows have `endpoint_scalars` with
   `endpoint_values_emitted_during_compaction: true` and a `trajectory.json` sidecar. RGB
   video was not rendered (Phase 0 still strips cameras).

`RELEASE_CLEARANCE_M = 0.005` was kept. Planning-time bow IK was not reintroduced
(`bow_fallback_taken` 0/24). Grasp candidates were not filtered.

## Integrity checks

- Frozen attempt-3 config SHA-256: `acd7ced5c7e5a0ea8f6a0070f98d507ea1ee0e983fe416f67273c786e596694f`.
- Master seed: `2026082001`.
- Reconstruct this ledger from the attempt-3 sources whose `source_sha256` matches the v3
  config, on top of parent `aad1cd23e394d619bf8924e99836b4c56fd37898` and MolmoSpaces
  `1cbb1800db66c871f41f2afc3a360affd1b40f1d`. At that tree, `enclosure_reach.py` is
  `3155ac3ea8a78b6c7a5e572b7f2bad830eee3a56833d53d80966cc3d8f685f3f` and
  `scripts/run_pact_place_expert_screen.py` is
  `b209e6a43583575200548d50f8fe6f33d9bf5e3fa6be97bf7e92a79331f9429a`.
- `pact_collision_corridor.xml` retained its pinned SHA-256 `f8c04b07b9416593eb60ad4797ccbae91f7d3524effd30438ef552e5a2d75540`.
- Upstream `pick_and_place_planner_policy.py` and `base_object_manipulation_planner_policy.py`
  were not modified; both match `9ee36978…` and `a7ee3570…`.
- Frozen contact-endpoint, geometry-v3, blur-sweep, and blind-RGB artifacts were hash-verified
  after the screen (12/12).
- The failed v1 and v2 artifacts were not overwritten.

## Episode IDs do not collide with v1 or v2

This contract includes the master seed in the `episode_id` preimage. All 24 IDs are disjoint
from both earlier screens. Join any pair of screens on `(config_sha256, role_index)` — never
on `episode_id`. This gate's `config_sha256` is
`acd7ced5c7e5a0ea8f6a0070f98d507ea1ee0e983fe416f67273c786e596694f`.

## Artifacts

- `configs/pact_place_corridor_v3.json`
- `diagnostics_output/pact_place_corridor_v3/expert_screen.json`
- `diagnostics_output/pact_place_corridor_v3/expert_screen_rows/*/result.json`
- `diagnostics_output/pact_place_corridor_v3/expert_screen_rows/*/trajectory.json`
- `diagnostics_output/pact_place_corridor_v3/expert_screen_rows/*/initial_observation_accepted.json`
- `diagnostics_output/pact_place_corridor_v3/stop_record.json`

PACT_PLACE_CORRIDOR_PHASE0_FAIL
