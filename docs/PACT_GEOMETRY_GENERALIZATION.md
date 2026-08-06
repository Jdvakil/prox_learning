# PACT held-out geometry generalization

## Decision

**GEOMETRY_TEST_INCONCLUSIVE.** The preregistered privileged-expert gate left only one solvable shifted condition, below the required minimum of two. No ACT, PACT, or PACT_PERMUTED policy rollout was launched, so these data make no claim about geometry generalization.

## Frozen design

The intended zero-shot evaluation used the existing frozen checkpoints on one fresh in-distribution control and three two-axis geometry shifts. Conditions were fixed before the expert screen; a failed condition was dropped without retuning. Policy evaluation required C0 plus at least two shifted conditions to pass at 10/12 clean expert successes.

A clean expert success means task success with zero `hazard_bar` and zero `other_environment` contact entries. Target contact remains allowed.

| Condition | Geometry | Task success | Clean success | Hazard-contact episodes | Other-environment episodes | Gate |
|---|---|---:|---:|---:|---:|---|
| C0 | in-distribution control | 11/12 | 11/12 | 0/12 | 0/12 | pass |
| C1 | PANEL_X 0.68, PANEL_Z 0.96 | 9/12 | 4/12 | 8/12 | 0/12 | drop |
| C2 | PANEL_INNER_FACE_Y 0.070, aperture 0.70 | 12/12 | 12/12 | 0/12 | 0/12 | pass |
| C3 | PANEL_X 0.55, aperture 1.00 | 7/12 | 5/12 | 2/12 | 0/12 | drop |

C0 passed 11/12 and C2 passed 12/12. C1 passed only 4/12, with hazard contact in 8/12 episodes. C3 passed 5/12; it had hazard contact in 2/12 episodes and one pre-boundary sampling failure after exhausting its fixed retries. Thus the surviving set was C0 and C2, only one of which was shifted.

## Hard stop and scientific interpretation

The 900-row policy schedule was not frozen or executed. `schedule.json` and `dispatch.json` are explicit Phase-0 stop records with zero policy rows. There are no policy outcomes to analyze, no C0 modality-gap reproduction result, and no shifted PACT-minus-PERMUTED estimate.

This is not evidence that PACT fails to generalize. It is evidence that two of the three proposed shifts were not cleanly solvable by the privileged expert under the fixed scene and planner, making the proposed test inadequate. Per the preregistration, those shifts were not adjusted after their outcomes were seen.

## Integrity and execution audit

- Manifest: `33e48ab83dfe398fbeb78f64565312c48a5a8b09cb1a873a2a2521e06fcbe7b2`; expert screen: `3bf7d5c8f86814b9c10308c10cf1576488e992d0d20564359cb911d312d78a2c`.
- All 48 expert rows reconciled; worker count stayed fixed at 8.
- After 33 terminal rows, the multiprocessing launcher stalled while recycling workers. No worker remained and no pending row had accepted an initial observation. The launcher was terminated and the 15 untouched rows were resumed under identical frozen code; no terminal or boundary-crossed row was rerun.
- Every shifted condition moves at least two axes outside the declared training support; this is asserted by the manifest contract and tests.
- The original `PactCollisionCorridorSampler` body and `pact_collision_corridor.xml` matched their committed byte references when the manifest was frozen. Only additive subclasses were used.
- All six ACT/PACT checkpoints, dataset statistics, and the frozen 32-D encoder were hash-verified even though no policy was run.
- The contact-endpoint decision, analysis, final decision, endpoint, awarded token, and confirmatory41 were not modified.

## Artifacts

- `configs/pact_geometry_generalization_v1.json` — candidate conditions, training support, and all fixed expert/policy instances.
- `diagnostics_output/pact_geometry_generalization/expert_screen.json` — reconciled Phase-0 gate.
- `diagnostics_output/pact_geometry_generalization/{schedule,dispatch,analysis,final_decision,provenance}.json` — explicit stopped-experiment record.
- `tests/test_pact_geometry_generalization.py` — geometry, balance, integrity, and decision-boundary tests.

GEOMETRY_TEST_INCONCLUSIVE
