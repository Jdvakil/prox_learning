# PACT pick-and-place corridor: Phase 0 expert gate

## Decision

Phase 0 failed. Per the preregistration, work stops here: no demonstration collection, encoder update, policy training, or learned-policy evaluation was run.

The gate required at least **20 clean successes in 24 fixed episodes**. A clean success means upstream `PickAndPlaceTask` success with zero `hazard_bar` and zero `other_environment` contact entries. Contact with `grasp_target` and the new `place_receptacle` class is expected and exempt.

## Measured Phase-0 results

| Measure | Result |
|---|---:|
| Reconciled rows | 24/24 |
| Clean pick-and-place successes | 18/24 (75.0%) |
| Ordinary task successes | 18/24 (75.0%) |
| Grasp phase successes (cup retrieved outside aperture) | 22/24 (91.7%) |
| Place phase successes | 18/24 (75.0%) |
| Place successes given grasp | 18/22 (81.8%) |
| Episodes with inbound hazard contact | 0/24 |
| Episodes with outbound hazard contact | 0/24 |
| Episodes with other-environment contact | 0/24 |
| Sampling failures | 0 |
| Infrastructure failures | 0 |

## Failure localization

The six task failures split into **2 retrieval failures** and **4 placement/release failures after successful retrieval**. The retrieval failures lifted the cup but terminated during `outbound_approach` before carrying it outside the aperture. The four later failures reached `placement_descent`; the cup was supported by the tray, but the robot was still touching it, so the upstream release condition correctly remained false.

| Failure class | Count | Row indices |
|---|---:|---|
| Cup not retrieved outside aperture | 2 | 8, 12 |
| Retrieved, but placement/release incomplete | 4 | 13, 16, 21, 22 |

All six failed rows also had zero hazard and zero other-environment contacts (0/6 with either contact class). The limiting factor was manipulation reliability—especially the final handoff—not corridor collision avoidance.

## What changed

The scene is a strict fork, `pact_place_corridor_v1.xml`. It preserves the panel, aperture, target sampling, robot offset, side balance, sensor layout, and existing contact classes, while adding a low pedestal and shallow tray wholly outside the aperture plane (tray x range 0.25–0.45 m; aperture x 0.58 m). The place expert composes the upstream pick-and-place planner with direction-aware panel bows on both inbound and outbound segments; the outbound carried-object envelope and clearance are deliberately larger.

## Success criterion

The endpoint is the upstream support-and-release criterion: the target is supported by the receptacle at at least 50% of its weight, the robot has released it, the receptacle moved no more than 0.1 m, and its tilt is no more than 45°. The old one-centimetre lift condition is recorded only as a grasp-progress diagnostic and is not on the task-success path.

## Integrity checks

- Frozen Phase-0 config SHA-256: `46dff849dd16eb3b6c0baf169053829bc66203a39866f14a4667e8eaef559e40`.
- `pact_collision_corridor.xml` retained its pinned SHA-256 `f8c04b07b9416593eb60ad4797ccbae91f7d3524effd30438ef552e5a2d75540`.
- The original `PactCollisionCorridorSampler`, `PactCollisionCorridorPolicy`, and `PactCollisionCorridorPolicyConfig` class bodies match the MolmoSpaces submodule commit exactly.
- Frozen contact-endpoint, geometry-v3, blur-sweep, and blind-RGB artifacts were hash-verified before and after the screen.
- No checkpoint, encoder, threshold, or existing scene was changed.

## Artifacts

- `configs/pact_place_corridor_v1.json`
- `diagnostics_output/pact_place_corridor/expert_screen.json`
- `diagnostics_output/pact_place_corridor/expert_screen_rows/*/result.json`
- `diagnostics_output/pact_place_corridor/stop_record.json`

PACT_PLACE_CORRIDOR_PHASE0_FAIL
