# PACT pick-and-place corridor: Phase 0 expert-fix re-screen (gate)

This is the **decision-bearing** fresh-seed re-screen after the expert fix. The original failed
gate remains in [`docs/PACT_PLACE_CORRIDOR_GATE.md`](PACT_PLACE_CORRIDOR_GATE.md). The labeled
comparison of both screens, conversion of the six known rows, and the stop-condition that fired
are in [`docs/PACT_PLACE_EXPERT_FIX.md`](PACT_PLACE_EXPERT_FIX.md).

## Decision

Phase 0 failed again. Per the preregistration and the expert-fix budget (one fix, two screens, no
third iteration), work stops here: no demonstration collection, encoder update, policy training,
or learned-policy evaluation was run.

The gate required at least **20 clean successes in 24 fixed episodes**. A clean success means
upstream `PickAndPlaceTask` success with zero `hazard_bar` and zero `other_environment` contact
entries. Contact with `grasp_target` and the new `place_receptacle` class is expected and exempt.

## Measured Phase-0 results

| Measure | Result |
|---|---:|
| Reconciled rows | 24/24 |
| Clean pick-and-place successes | 18/24 (75.0%) |
| Ordinary task successes | 20/24 (83.3%) |
| Grasp phase successes (cup retrieved outside aperture) | 20/24 (83.3%) |
| Place phase successes | 20/24 (83.3%) |
| Place successes given grasp | 20/20 (100.0%) |
| Episodes with inbound hazard contact | 2/24 |
| Episodes with outbound hazard contact | 0/24 |
| Episodes with other-environment contact | 0/24 |
| Sampling failures | 0 |
| Infrastructure failures | 0 |
| Bow-fallback episodes | 0/24 |

## Failure localization

Ordinary task success rose from 18/24 to 20/24, and every successful grasp was also a successful
place (20/20). The clean count stayed **18/24** because two otherwise-successful episodes contacted
the hazard bar on the inbound pick. That contact class was 0/24 on the original failed gate.

| Failure class | Count | Row indices |
|---|---:|---|
| Cup not retrieved outside aperture | 4 | 3, 4, 11, 17 |
| Retrieved, placed, but inbound hazard contact | 2 | 6, 12 |
| Retrieved, but placement/release incomplete | 0 | — |

## Integrity checks

- Frozen re-screen config SHA-256: `a0f30725e325a73b5584895a07fa18000fe3645cb63ebd1b4e5a6746bc201c31`.
- Master seed: `2026081901`.
- `pact_collision_corridor.xml` retained its pinned SHA-256 `f8c04b07b9416593eb60ad4797ccbae91f7d3524effd30438ef552e5a2d75540`.
- The original `PactCollisionCorridorSampler`, `PactCollisionCorridorPolicy`, and `PactCollisionCorridorPolicyConfig` class bodies match the MolmoSpaces submodule commit exactly.
- Upstream `pick_and_place_planner_policy.py` and `base_object_manipulation_planner_policy.py` were not modified; both match the hashes pinned in the v2 contract.
- Frozen contact-endpoint, geometry-v3, blur-sweep, and blind-RGB artifacts were hash-verified before and after the screen (12/12).
- The failed v1 artifacts were not overwritten.

## Artifacts

- `configs/pact_place_corridor_v2.json`
- `diagnostics_output/pact_place_corridor_v2/expert_screen.json`
- `diagnostics_output/pact_place_corridor_v2/expert_screen_rows/*/result.json`
- `diagnostics_output/pact_place_corridor_v2/stop_record.json`

PACT_PLACE_CORRIDOR_PHASE0_FAIL
