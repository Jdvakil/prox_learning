# PACT pick-and-place corridor: Phase 0 attempt-4 screen (gate)

This is the **decision-bearing** fresh-seed screen after the measured empty-gripper
repair in [`docs/PACT_PLACE_ATTEMPT4.md`](PACT_PLACE_ATTEMPT4.md). Three prior screens
failed at exactly 18/24. Those FAIL records are not rewritten:
[`docs/PACT_PLACE_CORRIDOR_GATE.md`](PACT_PLACE_CORRIDOR_GATE.md),
[`docs/PACT_PLACE_CORRIDOR_GATE_V2.md`](PACT_PLACE_CORRIDOR_GATE_V2.md),
[`docs/PACT_PLACE_CORRIDOR_GATE_V3.md`](PACT_PLACE_CORRIDOR_GATE_V3.md).

## Decision

Phase 0 failed a fourth time. The attempt-4 prediction, recorded in
`configs/pact_place_corridor_v4.json` before the first episode, was **19–22 clean
of 24**. The outcome is **15/24**. That is at or below 18, so the empty-gripper
mechanism does not set the Phase-0 ceiling. There is no fifth attempt. No
demonstration collection, encoder update, policy training, or learned-policy
evaluation was run.

The gate required at least **20 clean successes in 24 fixed episodes**. A clean success means
upstream `PickAndPlaceTask` success with zero `hazard_bar` and zero `other_environment` contact
entries. Contact with `grasp_target` and the new `place_receptacle` class is expected and exempt.

## All four screens

| Screen | Seed | Clean | Task | What changed since the previous |
|---|---|---:|---:|---|
| v1 original | `2026081801` | 18/24 | 18/24 | Original privileged expert gate |
| v2 expert-fix | `2026081901` | 18/24 | 20/24 | `RELEASE_CLEARANCE_M = 0.005` |
| v3 attempt 3 | `2026082001` | 18/24 | 18/24 | Initial-contact reject; subdivide `outbound_approach` |
| v4 attempt 4 | `2026082101` | **15/24** | 15/24 | Disarm empty-gripper on `placement_descent`; persist N=3 on transport (threshold unchanged) |

Join rows on `(config_sha256, role_index)`, never `episode_id`.

## Measured Phase-0 results

| Measure | Result |
|---|---:|
| Reconciled rows | 24/24 |
| Clean pick-and-place successes | 15/24 (62.5%) |
| Ordinary task successes | 15/24 (62.5%) |
| Grasp phase successes (cup retrieved outside aperture) | 15/24 (62.5%) |
| Place phase successes | 15/24 (62.5%) |
| Place successes given grasp | 15/15 (100%) |
| Episodes with inbound hazard contact | 0/24 |
| Episodes with outbound hazard contact | 0/24 |
| Episodes with other-environment contact | 0/24 |
| Sampling failures | 0 |
| Infrastructure failures | 0 |
| Bow-fallback episodes | 0/24 |

## Failure localization

All nine task failures are **retrieval** failures. Place-given-grasp is 15/15: every
episode that got the cup out also placed it. Zero rows terminated on
`empty_gripper`. Hazard-bar and other-environment contact were 0/24.

| Failure class | Count | Row indices |
|---|---:|---|
| `pregrasp` TCP tracking (`pos_err` ≈ 10.1–10.4 cm) | 4 | 5, 7, 15, 22 |
| `outbound_approach` TCP tracking (`pos_err` ≈ 10.1–10.2 cm) | 2 | 8, 18 |
| `outbound_approach` IK cascade (8 sequential IK failures) | 3 | 9, 10, 20 |
| Empty-gripper abort | 0 | — |
| Retrieved, but placement/release incomplete | 0 | — |

Mechanisms, from `result.json` / `endpoint_scalars` / `terminal_tracking` (new seed):

| Row | Phase | Branch | What happened |
|---:|---|---|---|
| 5, 7, 15, 22 | `pregrasp` | `pos_err` | Gripper never closed (~86.6 mm). Actual TCP z ran ~8–9 cm above the target; total error just over the 10 cm threshold. Inbound `TCPMoveSequence`, not the place subclass. |
| 8, 18 | `outbound_approach` | `pos_err` | Cup lifted. Holding width 16.2 mm / 8.2 mm. Actual z ~9 cm above the carry-plane target (0.981/0.994 vs 0.892/0.906). Same tracking class as v3 row 9. |
| 9, 10, 20 | `outbound_approach` | `ik_cascade` | Still holding (~8.4 mm). Position error 2.4–2.7 cm, under threshold. Eight sequential IK failures. |

`gripper_empty_threshold` stayed 0.002 m. Persist N stayed 3. Tracking and IK
thresholds were not changed.

## Against the recorded prediction

The frozen v4 contract recorded, before any episode: predicted clean **19–22 / 24**
(rate 0.82–0.86), bar 20; on ≤18, “mechanism story is wrong — stop.”

15/24 is below that floor. The empty-gripper repair did what it claimed on this
seed — it was not the terminal branch on any row, and place-given-grasp is
perfect — and the gate still failed because inbound/outbound tracking and IK
set a lower ceiling. That is the result. No further repair is indicated.

## Integrity checks

- Frozen attempt-4 config SHA-256: `fe45435d55cda8daee71972451ce8b460e641b71558aa8a69b3a4686319cdc65`.
- Master seed: `2026082101`.
- Reconstruct this ledger from the attempt-4 sources whose `source_sha256` matches the v4
  config, on top of parent `63813822e4847caaef3d5b136b4a5ebb3cb5afb4` and MolmoSpaces
  `1cbb1800db66c871f41f2afc3a360affd1b40f1d`. At that tree, `enclosure_reach.py` is
  `06dfc6dc2c2a196188849b174061d5b7e0c0fa30f6f4a176f8f07a62a3859644` and
  `scripts/run_pact_place_expert_screen.py` is
  `b209e6a43583575200548d50f8fe6f33d9bf5e3fa6be97bf7e92a79331f9429a`.
- `pact_collision_corridor.xml` retained its pinned SHA-256 `f8c04b07b9416593eb60ad4797ccbae91f7d3524effd30438ef552e5a2d75540`.
- Upstream `pick_and_place_planner_policy.py` and `base_object_manipulation_planner_policy.py`
  were not modified; both match `9ee36978…` and `a7ee3570…`.
- Frozen contact-endpoint, geometry-v3, blur-sweep, and blind-RGB artifacts were hash-verified
  after the screen (12/12).
- The failed v1, v2, and v3 artifacts were not overwritten.

## Episode IDs do not collide with v1–v3

This contract includes the master seed in the `episode_id` preimage. All 24 IDs are disjoint
from the earlier screens. Join any pair of screens on `(config_sha256, role_index)` — never
on `episode_id`. This gate's `config_sha256` is
`fe45435d55cda8daee71972451ce8b460e641b71558aa8a69b3a4686319cdc65`.

## Artifacts

- `configs/pact_place_corridor_v4.json`
- `diagnostics_output/pact_place_corridor_v4/expert_screen.json`
- `diagnostics_output/pact_place_corridor_v4/expert_screen_rows/*/result.json`
- `diagnostics_output/pact_place_corridor_v4/expert_screen_rows/*/trajectory.json`
- `diagnostics_output/pact_place_corridor_v4/expert_screen_rows/*/initial_observation_accepted.json`
- `diagnostics_output/pact_place_corridor_v4/stop_record.json`

PACT_PLACE_CORRIDOR_PHASE0_FAIL
