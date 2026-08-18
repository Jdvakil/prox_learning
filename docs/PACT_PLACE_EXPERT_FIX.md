# PACT pick-and-place expert fix: two-screen report

## Decision

**Stop.** The fresh-seed gate failed at **18/24 clean successes** against a required 20/24
(`PACT_PLACE_CORRIDOR_PHASE0_FAIL`). The expert-fix budget is one fix and two screens; there is
no third iteration. No demonstration collection, encoder work, or training is authorized.

The original failed Phase-0 record is unchanged:
[`docs/PACT_PLACE_CORRIDOR_GATE.md`](PACT_PLACE_CORRIDOR_GATE.md). The re-screen gate ledger is
[`docs/PACT_PLACE_CORRIDOR_GATE_V2.md`](PACT_PLACE_CORRIDOR_GATE_V2.md).

This document reports **both** screens, labeled. The diagnostic is not a gate and is never to be
cited as one.

## What was frozen before either screen

Both mechanical fixes were committed to the running tree, constants were locked, and the fresh-seed
contract was rebuilt **before** any re-screen episode ran.

| Item | Value |
|---|---|
| Release clearance | `PactPlaceCorridorPolicy.RELEASE_CLEARANCE_M = 0.005` (subclass override only; kept) |
| Bow shrink step (screen-time only) | `BOW_SHRINK_STEP_M = 0.02` at submodule `2828751ee6`; reverted after the screens as an instance filter |
| Fresh-seed master seed | `2026081901` |
| Fresh-seed config SHA-256 | `a0f30725e325a73b5584895a07fa18000fe3645cb63ebd1b4e5a6746bc201c31` |
| Original-seed config SHA-256 (diagnostic only) | `46dff849dd16eb3b6c0baf169053829bc66203a39866f14a4667e8eaef559e40` |
| `MIN_CLEAN_SUCCESSES` | 20 (unchanged) |
| Grasp candidate set | unchanged (not filtered or re-ranked) |
| Scene / sampler / success criterion | unchanged |

`pick_and_place_planner_policy.py` and `base_object_manipulation_planner_policy.py` were not
edited. They are now pinned in `source_sha256` at

- `9ee369789397add3ae74492e4821993a981940be54cc0579e3d282328a8aa36a`
- `a7ee35704d60b82246fc48db466aaa081a6a83717d983cc062df3988e53893d5`

## The two screens

1. **Gate screen — 24 fresh episodes at `MASTER_SEED = 2026081901`.** This is the decision.
2. **Diagnostic re-run — the original 24 episodes from `MASTER_SEED = 2026081801`.** Not the
   gate. Its only job is to say whether the known failures converted and whether the original
   18 successes held.

### 1. Gate screen (decision)

| Measure | Original gate | Fresh-seed re-screen |
|---|---:|---:|
| Clean successes | 18/24 | **18/24** |
| Task successes | 18/24 | 20/24 |
| Grasp-phase successes | 22/24 | 20/24 |
| Place given grasp | 18/22 | **20/20** |
| Inbound hazard episodes | 0/24 | **2/24** |
| Outbound hazard episodes | 0/24 | 0/24 |
| Other-environment episodes | 0/24 | 0/24 |
| Bow fallback taken | n/a | 0/24 |

The original A-mode failure (cup on the tray, robot still touching it, `gripper_width_min = 0`)
did not recur: every retrieved cup was released (`place_success_given_grasp = 20/20`).

The clean count stayed 18/24 because of a **new failure mode**: inbound hazard-bar contact on
two otherwise successful episodes (rows 6 and 12; 82 and 171 inbound hazard frames; 0 outbound).
The original gate had 0/24 inbound hazard contact, including on its six failures.

The four task failures (rows 3, 4, 11, 17) are retrieval failures at the level of
`grasp_phase_success`, but they are **three mechanisms**, not one:

- **Row 11** — grasp failure at `lift`. Never reached 1 cm (`cup_lifted_one_cm: false`). Cup
  ended at z 0.753, **below** its 0.794 start (`pickup_final_position_m` z 0.75267 vs
  `pickup_start_z_m` 0.79375).
- **Row 3** — cup loss during `outbound_pass`. `gripper_width_min_m = 0`. Cup abandoned at
  x ≈ 0.67 inside the corridor (`pickup_final_position_m` `[0.673, -0.157, 0.836]`).
- **Rows 4 and 17** — TCP tracking-error abort while still holding the cup
  (`gripper_width_min_m` 12.4 mm / 8.4 mm; cup at z ≈ 0.92). Terminal
  `position_error_m` 0.10177 / 0.10154 against `tcp_pos_err_threshold = 0.1`. Row 4 actual
  TCP z 0.9806 vs target 0.8921; row 17 actual z 0.992 vs target 0.904.

### 2. Diagnostic re-run (original seeds; not a gate)

This run is marked `role: diagnostic_not_a_gate` with `authorizes_collection: false` and
`next_action: none_diagnostic_only`. It does not emit `PACT_PLACE_CORRIDOR_PHASE0_PASS` or
`PACT_PLACE_CORRIDOR_PHASE0_FAIL`. Clean count on this diagnostic is 21/24; that number does
not authorize collection.

| Measure | Original gate | Same-seed diagnostic |
|---|---:|---:|
| Clean successes | 18/24 | 21/24 |
| Task successes | 18/24 | 21/24 |
| Inbound / outbound / other hazard | 0 / 0 / 0 | 0 / 0 / 0 |
| Bow fallback taken | n/a | 0/24 |

## Did the six known rows convert?

Original failures: **8, 12** (outbound-bow IK during execution) and **13, 16, 21, 22**
(descent-to-contact, empty-gripper abort before `release`).

| Row | Original class | Same first seed? | Diagnostic outcome |
|---|---|---|---|
| 13 | A (release) | yes (no resample) | converted: clean success, `robot_contact: false`, `gripper_width_min ≈ 8 mm` |
| 16 | A (release) | yes | converted, same pattern |
| 21 | A (release) | yes | converted, same pattern |
| 22 | A (release) | yes | converted, same pattern |
| 8 | B (bow IK) | **no** — first seed now raises at planning (`min_bow=0.1526 m`); two pre-boundary retries | clean success on a **different** episode |
| 12 | B (bow IK) | **no** — first seed now raises at planning (`min_bow=0.2246 m`); one retry | **not converted**: retry failed at `pregrasp` (10 cm tracking error, gripper never closed) |

The original 18 successes did **not** all hold. Rows **1** and **18** flipped to failure after
the new planning-time bow IK check rejected their original seeds and the resampled episodes
lost the cup (`gripper_width_min = 0` during lift / outbound).

Net diagnostic: 18 − 2 + 5 = 21. Row 12 remains a failure. Rows 1 and 18 are new losses.

## Which diagnosis held

**A held.** Adding 5 mm of release clearance in the subclass, without touching the shared
planner, converted all four same-seed placement/release failures. On the fresh gate, place given
grasp is 20/20. The cup is opened above the tray and settles; the scripted `release`/`retreat`
now run.

**B did not behave as described.** At screen time, `_bow_segment` IK-validated `pose_before`
and `pose_after`, and it would shrink `required_bow` in 2 cm steps down to the minimum that
still clears `safe_gap`. In this geometry that minimum **is** the planned bow, so the shrink
loop never has an interior. Fallback was taken **0/24** times on both screens. When the
planned bow fails `check_feasible_ik` at reset (home qpos), the policy raises, which the
screen treats as a pre-boundary resample. That is not “bowed less than planned.” It is a
different episode.

Worse: some bows that **executed successfully** on the original gate now fail the
planning-time check (row 1’s original seed is the example). Execution IK is seeded from the
lift configuration; planning IK is seeded from the reset qpos. The check therefore both
fails to repair the original row-12 path and rejects previously viable first seeds.

That planning-time IK filter was reverted after the screens. The frozen ledgers still
describe the code they ran against. Future use of this expert plans the outbound bow
without rejecting the instance at reset.

## New failure mode on the gate

Inbound hazard contact on successful pick-and-place episodes (rows 6 and 12 of the fresh seed).
The original gate, and the same-seed diagnostic, had zero inbound and zero outbound hazard
contact. The stop rule for a new failure mode is: report it, do not absorb it into a third fix.

Characterize these two rows by what is observable. Hazard contact is first recorded at
`first_contact_step.hazard_bar == 0` in both, with maximum penetration **5.5 mm and 20.5 mm**.
All frames fall in the `inbound` bucket. `contact_frame_payload_retained: false`, so the
per-frame records were discarded and neither the end of the contact nor the participating
bodies can be recovered. Do not describe these as grazing contacts during the inbound bow —
that is not established.

`INBOUND_ENVELOPE_HALF_Y` and `INBOUND_SAFE_GAP` are dead constants on
`PactPlaceCorridorPolicy`. `_bow_segment` has exactly one call site, with `prefix="outbound"`.
The inbound path is `PactCollisionCorridorPolicy._compute_trajectory()` reused wholesale.
Neither expert fix can reach the inbound path.

## Do not raise `tcp_pos_err_threshold`

Rows 4 and 17 exceeded the 0.1 m tracking threshold by **1.77 mm and 1.54 mm**. Raising
`tcp_pos_err_threshold` converts both and would pass the gate at 20/24. It must not be done.
Their error is **8.8 cm of vertical deviation** — actual z 0.9806 vs target 0.8921 on row 4 —
not a lateral reach shortfall. The threshold is the detector, not the cause. Loosening it
would let a badly out-of-plane arm continue through the corridor carrying the cup, which is a
plausible way to manufacture contact on the endpoint the whole project measures.

No geometric parameter separates these failures. In the fresh gate, planned outbound bow gives
9/12 clean above 0.20 m and 9/12 clean below. In the original screen the largest lateral
waypoint reach (row 11, 0.1948) was a success, exceeding both original bow-IK failures. Two
screens, two searches, no separator found.

## Record cleanup after the screens

The gates are frozen. After they were recorded, three record defects were fixed without
re-running episodes:

1. The diagnostic summary no longer carries a gate token or a `proceed_*` next action.
2. Future `episode_id` values include the master seed. Frozen v1/v2 files still collide;
   join them on `(config_sha256, role_index)`.
3. Fix 2 (planning-time bow IK / shrink loop) was reverted. It discarded geometry whose
   outbound bow is IK-infeasible from the reset pose — an instance filter, not a repair —
   and it made the diagnostic incomparable on 5/24 rows. Telemetry
   (`planned_bow_m`, `accepted_bow_m`, `bow_fallback_taken`) and Fix 1 (5 mm release
   clearance) remain. The revert makes a future screen **harder**: a hard instance fails
   during the rollout instead of being silently replaced.

`cb19130709d6961ac3fcf14ae18ee4d18004ea8a3273f2174d0083f53afdadbb` for `enclosure_reach.py`
remains reachable at submodule `2828751ee6a1fb5ffcaa30d47fda45859f835510`.

## Stop condition that fired

All of the following are true; any one of them was sufficient:

- Gate failed (<20/24 fresh clean successes).
- The six known rows did not all convert; B did not behave as described.
- A new failure mode appeared (inbound hazard contact).
- Hard budget: one fix, two screens. No third iteration.

## What a pass would not have bought

Even a green gate would not have answered the reason this corridor exists. The expert still
produced **zero outbound hazard contact** on both screens — expected of a planner, and no
evidence either way about ACT. The real risk sits downstream at the adequacy gate, which this
corridor has already failed twice: vision-only ACT must be solvable-but-not-saturated *and*
must actually contact the panel. A longer motion makes “solvable” strictly harder, with no
evidence yet that it makes “contacts the panel” more likely.

## Integrity (re-checked after both screens)

- `pact_collision_corridor.xml` SHA-256 `f8c04b07b9416593eb60ad4797ccbae91f7d3524effd30438ef552e5a2d75540`.
- All 12 `protected_artifact_sha256_before` entries unchanged.
- `PactCollisionCorridorSampler`, `PactCollisionCorridorPolicy`, `PactCollisionCorridorPolicyConfig`
  still byte-identical to the submodule commit.
- Upstream shared planner files unmodified at the hashes above.
- `lift_one_centimetre_criterion_on_success_path: false` still.
- Failed v1 artifacts and `PACT_PLACE_CORRIDOR_PHASE0_FAIL` left in place.

## Artifacts

Gate (decision):

- `configs/pact_place_corridor_v2.json`
- `diagnostics_output/pact_place_corridor_v2/expert_screen.json`
- `diagnostics_output/pact_place_corridor_v2/expert_screen_rows/*/result.json`
- `diagnostics_output/pact_place_corridor_v2/stop_record.json`

Diagnostic (not a gate):

- `configs/pact_place_corridor_v1.json` (original seeds; code is the fixed expert)
- `diagnostics_output/pact_place_corridor_v2_diagnostic_original_seeds/expert_screen.json`
- `diagnostics_output/pact_place_corridor_v2_diagnostic_original_seeds/role.json`
- `diagnostics_output/pact_place_corridor_v2_diagnostic_original_seeds/expert_screen_rows/*/result.json`

Original failed gate (untouched):

- `docs/PACT_PLACE_CORRIDOR_GATE.md`
- `diagnostics_output/pact_place_corridor/`

PACT_PLACE_CORRIDOR_PHASE0_FAIL
