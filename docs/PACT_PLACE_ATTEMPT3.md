# PACT pick-and-place Phase 0, attempt 3

This is a third screen after two failures. It was defensible only because each change
repaired a named defect. All three decisions are
`PACT_PLACE_CORRIDOR_PHASE0_FAIL`. There is no fourth iteration.

The v3 gate ledger is [`PACT_PLACE_CORRIDOR_GATE_V3.md`](PACT_PLACE_CORRIDOR_GATE_V3.md).
The v2 rows-6/12 diagnosis is [`PACT_PLACE_HAZARD_ROWS_6_12.md`](PACT_PLACE_HAZARD_ROWS_6_12.md).

## The three screens

| Screen | Master seed | Config SHA-256 | Clean | Task | Place given grasp | Inbound hazard | Decision |
|---|---:|---|---:|---:|---|---:|---|
| v1 original | `2026081801` | `46dff849dd16eb3b6c0baf169053829bc66203a39866f14a4667e8eaef559e40` | 18/24 | 18/24 | 18/22 | 0/24 | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |
| v2 expert-fix re-screen | `2026081901` | `a0f30725e325a73b5584895a07fa18000fe3645cb63ebd1b4e5a6746bc201c31` | 18/24 | 20/24 | 20/20 | 2/24 | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |
| v3 attempt 3 | `2026082001` | `acd7ced5c7e5a0ea8f6a0070f98d507ea1ee0e983fe416f67273c786e596694f` | **18/24** | 18/24 | 18/19 | **0/24** | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |

Join rows on `(config_sha256, role_index)`, never `episode_id`. v1/v2 share 12 colliding IDs;
v3 includes the master seed in the preimage and collides with neither.

Two independent routes needed two conversions each to reach 20/24 clean. Neither route
cleared the bar on this seed.

## Named repairs (frozen before the first v3 episode)

1. **v2 rows 6 and 12 — initial-state panel overlap.** Contact-frame re-run of those v2
   instances showed `fr3_link7` intersecting `pact_intrusion_right` at the accepted
   initial observation (max penetration 5.54 mm and 20.46 mm on the first frame,
   gone by t = 0.53 s / 0.72 s, never returning). **Outcome 1**, not inbound scraping.
   **Repair:** reject `hazard_bar` / `other_environment` contact at the scientific
   boundary and resample pre-boundary. This is Route A.
2. **v2 rows 4 and 17 — one-shot outbound approach.** TCP tracking aborted on
   `outbound_approach` with 8.8 cm of vertical deviation. **Repair:** subdivide
   `outbound_approach` so each Cartesian piece is at most
   `OUTBOUND_APPROACH_MAX_STEP_M = 0.04`. Grasp candidates unchanged. v2 rows 3 and 11
   were left as grasp-stability failures. This is Route B. IK continuity across the bow
   was the second-ranked repair and was not applied.
3. **Record.** Every complete row emits `endpoint_scalars` before the result is written,
   plus a `trajectory.json` sidecar. Payload deletion without that block is refused.
   RGB video is not rendered during the screen (Phase 0 still strips cameras).

Kept: `RELEASE_CLEARANCE_M = 0.005`, Fix 2 reverted, `MIN_CLEAN_SUCCESSES = 20`,
`N_EXPERT_ROWS = 24`. Not done: raising `tcp_pos_err_threshold`, widening
`max_sequential_ik_failures`, filtering grasps, or reintroducing planning-time bow IK.

## What they did

| Repair | On v3 (`MASTER_SEED = 2026082001`) |
|---|---|
| Initial-contact reject | Inbound hazard **0/24**. The rejector never fired (0 retries of that class). This seed did not redraw the packed reset overlap. |
| Subdivide `outbound_approach` | Did **not** convert the tracking class. v3 row 9 aborted on `outbound_approach` still holding the cup, tracking error 12.58 cm, actual z 0.9866 vs target 0.9062. |
| Endpoint scalars / trajectory | 24/24 rows wrote the block (`endpoint_values_emitted_during_compaction: true`) and a sidecar. |

v3 clean failures are rows **2, 6, 9, 12, 20, 21**. Five are retrieval; row 6 is the
one missed place-given-grasp (`robot_contact` still true on the tray). Detail is in
the v3 ledger.

## Screen

| Item | Value |
|---|---|
| Master seed | `2026082001` |
| Config | `configs/pact_place_corridor_v3.json` |
| Config SHA-256 | `acd7ced5c7e5a0ea8f6a0070f98d507ea1ee0e983fe416f67273c786e596694f` |
| Output | `diagnostics_output/pact_place_corridor_v3/` |
| `expert_screen_sha256` | `05b59f4001a37be773016e338d99216e1180c4919f8da6f62db69dc24f19f7a0` |
| `stop_record_sha256` | `e96af8dc9144e85dfef56ef01d70e467cc6e0c68f4f518fb331533ee47247a75` |

One run only. Stop.
