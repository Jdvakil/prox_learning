# PACT pick-and-place Phase 0, attempt 4

This is a fourth screen after three failures at exactly 18/24. It was defensible
only because the empty-gripper defect had been measured, and a prediction was
recorded before the first episode. All four decisions are
`PACT_PLACE_CORRIDOR_PHASE0_FAIL`. There is no fifth iteration.

The v4 gate ledger is [`PACT_PLACE_CORRIDOR_GATE_V4.md`](PACT_PLACE_CORRIDOR_GATE_V4.md).
The abort-branch diagnostic is [`PACT_PLACE_ABORT_BRANCH.md`](PACT_PLACE_ABORT_BRANCH.md).

## The four screens

| Screen | Master seed | Config SHA-256 | Clean | Task | Place given grasp | Inbound hazard | Decision |
|---|---:|---|---:|---:|---|---:|---|
| v1 original | `2026081801` | `46dff849dd16eb3b6c0baf169053829bc66203a39866f14a4667e8eaef559e40` | 18/24 | 18/24 | 18/22 | 0/24 | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |
| v2 expert-fix re-screen | `2026081901` | `a0f30725e325a73b5584895a07fa18000fe3645cb63ebd1b4e5a6746bc201c31` | 18/24 | 20/24 | 20/20 | 2/24 | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |
| v3 attempt 3 | `2026082001` | `acd7ced5c7e5a0ea8f6a0070f98d507ea1ee0e983fe416f67273c786e596694f` | 18/24 | 18/24 | 18/19 | 0/24 | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |
| v4 attempt 4 | `2026082101` | `fe45435d55cda8daee71972451ce8b460e641b71558aa8a69b3a4686319cdc65` | **15/24** | 15/24 | 15/15 | 0/24 | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |

Join rows on `(config_sha256, role_index)`, never `episode_id`.

The prediction recorded in the v4 config before any episode was **19–22 clean**.
15/24 is ≤18: the empty-gripper mechanism is not what sets the ceiling.

## Named repairs (frozen before the first v4 episode)

`gripper_empty_threshold` stayed **0.002 m**. This is not a threshold change.

1. **Fix A — disarm empty-gripper during `placement_descent`.** Category error: that
   segment's job is to put the cup down; the next primitive is the scripted release.
   All five historical `placement_descent` empty-gripper aborts already had
   `supported_by_receptacle: true`.
2. **Fix B — persist N=3 during transport.** The artifact was a one-step
   8.5 mm → 0.00 mm sample that never appeared twice in a row in the 11 diagnostic
   traces. Require three consecutive empty samples. N was not tuned after seeing
   the outcome.
3. **Subclass, do not edit upstream.** `PactPlaceTCPMoveSequence` is constructed
   only from `PactPlaceCorridorPolicy._sequence`. Upstream planner hashes remain
   `9ee36978…` and `a7ee3570…`.

## Step 0 (bounded; did not block A/B)

`inter_finger_dist` is a live qpos sum, not a failed lookup. On the v3 abort
traces, `qpos[7:12]` (spring-linkage, ~0.755 rad) did not move at the glitch
step, and pad–cup contact persisted. The 8.5 mm holding width is **not** stored
as two millimetre-scale slide joints in the recorded full-model `qpos`.

v4 terminal telemetry logs `gripper_joint_posadr = [7, 10]`. While holding, that
pair reads ~0.69–0.76 (driver-joint radians) while `inter_finger_dist_m` is
8–16 mm; the two logged numbers are not each other's sum. A one-line physics
fix was not identified. A/B were applied as planned.

Control rows from the abort diagnostic: successes 5/15/17 had 0 empty-while-holding
events (sub-trip counts 1, 1, 2 landed on `gripper-close` / `gripper-open` /
`retreat`, when `is_holding_object` was false).

## What they did on this seed

| Repair | On v4 (`MASTER_SEED = 2026082101`) |
|---|---|
| Disarm on `placement_descent` | Place-given-grasp **15/15**. Zero placement-release misses. Zero `empty_gripper` terminals. |
| Persist N=3 on transport | No row died on a one-step empty sample. `empty_gripper_streak` was 0 at every failure. |
| Empty-gripper as ceiling | **Not confirmed.** Clean rate moved from 18/24 to **15/24**. |

v4 clean failures are rows **5, 7, 8, 9, 10, 15, 18, 20, 22**. All nine are
retrieval. Detail is in the v4 ledger.

## Screen

| Item | Value |
|---|---|
| Master seed | `2026082101` |
| Config | `configs/pact_place_corridor_v4.json` |
| Config SHA-256 | `fe45435d55cda8daee71972451ce8b460e641b71558aa8a69b3a4686319cdc65` |
| Output | `diagnostics_output/pact_place_corridor_v4/` |
| `expert_screen_sha256` | `1f19a02c945c6a96370b1cfddbd2850102c5d27983460e4dcb8c75000a244f41` |
| `stop_record_sha256` | `484f59c64f41ff316bf1ef534ad91e0ca58bcead25103044a025e77f2fd5b4da` |
| Prediction (pre-registered) | 19–22 clean; bar 20 |
| Outcome | 15/24 clean |

One run only. No fifth attempt. No collection.

PACT_PLACE_CORRIDOR_PHASE0_FAIL
