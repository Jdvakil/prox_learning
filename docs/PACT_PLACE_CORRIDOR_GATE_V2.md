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
- Reconstruct this ledger from parent commit `9fb040624de03fae250305b1426d0e0767a0611d` and MolmoSpaces submodule `2828751ee6a1fb5ffcaa30d47fda45859f835510`. At that pair, `scripts/pact_place_corridor_contract.py` is `7ada72073207d03f539f3cfa969c8bdf9f949d5e3fbf6f9fdf4069b1f798d1e0` and `enclosure_reach.py` is `cb19130709d6961ac3fcf14ae18ee4d18004ea8a3273f2174d0083f53afdadbb`, matching `source_sha256` in the v2 config. Later cleanup reverted the planning-time bow IK filter; that revert does not change this ledger.
- `pact_collision_corridor.xml` retained its pinned SHA-256 `f8c04b07b9416593eb60ad4797ccbae91f7d3524effd30438ef552e5a2d75540`.
- The original `PactCollisionCorridorSampler`, `PactCollisionCorridorPolicy`, and `PactCollisionCorridorPolicyConfig` class bodies match the MolmoSpaces submodule commit exactly.
- Upstream `pick_and_place_planner_policy.py` and `base_object_manipulation_planner_policy.py` were not modified; both match the hashes pinned in the v2 contract.
- Frozen contact-endpoint, geometry-v3, blur-sweep, and blind-RGB artifacts were hash-verified before and after the screen (12/12).
- The failed v1 artifacts were not overwritten.

## Episode IDs collide with the v1 screen

`episode_id` in this contract is `sha256("pact-place-v1:expert:{index}:{side}")`. The master seed is not in the preimage. Twelve of 24 IDs are byte-identical to the v1 screen, while 0/24 task seeds and 0/24 panel jitters match.

Join v1 and v2 rows on `(config_sha256, role_index)` — never on `episode_id`. This gate's `config_sha256` is `a0f30725e325a73b5584895a07fa18000fe3645cb63ebd1b4e5a6746bc201c31`. The v1 gate's is `46dff849dd16eb3b6c0baf169053829bc66203a39866f14a4667e8eaef559e40`. Reconstruct the v1 code from parent `84594895c4dcdff7c2d582ce7bc5c15e4562378b` and submodule `b00dc3523b0930afdae4e95b1aac0ba7211714f3`.

Shared `episode_id` values (same `role_index` and `intrusion_side`):

| role_index | side | episode_id |
|---:|---|---|
| 0 | left | `84054858899b293a466a5c57405b4050a574f845b84230f92a7cf68bde4e3655` |
| 1 | right | `b852218578ffc5274f0a1d86e8bad1586f3e63468e0e5d0efd937e63368478a7` |
| 3 | left | `2eb73ef8687c68eff34da05455c385633631465e11d5da2c46ddc7e01edc4a5b` |
| 4 | left | `19b940316438d65d6ff790b61ef3d42a9b118bdb084c4d6467d5b38adf19c9e4` |
| 5 | left | `0a0e39be0777044ed2ead1b2c483b8255fe1c2d19329d1d0a212fb5032e3474b` |
| 7 | right | `216c6e9c7282e004bef6960051c3f8a72f94fdff75222018acd0be8b51fc1fa8` |
| 9 | right | `e8b045f28b824f74d68845681c5e0aed2a95677db661b6ad5bddc4e71d54cd26` |
| 12 | right | `0f9c2bb77b0b41f13680f03132c13d04d95f2cbb43372c708f96994e9fefc34b` |
| 14 | right | `564a2e539d2a135cc9aae6fa5ecb499d339d4699ba9dc5a611b68619120cff79` |
| 15 | left | `c2916ec449045440abac4380cf27c2fd073296ad8c5e4dabfb5f9b58cb0aee53` |
| 16 | left | `966da9edf6b1a53a1abcc15dc2313e01bca12db4c5c8f6304dd6567314331b27` |
| 22 | right | `3cc34a7c467684d2686ab132722ee3244c19d66870b981171c665710e94a520d` |

Later contracts include the master seed in the preimage. These frozen files are not rewritten.

## Artifacts

- `configs/pact_place_corridor_v2.json`
- `diagnostics_output/pact_place_corridor_v2/expert_screen.json`
- `diagnostics_output/pact_place_corridor_v2/expert_screen_rows/*/result.json`
- `diagnostics_output/pact_place_corridor_v2/stop_record.json`

PACT_PLACE_CORRIDOR_PHASE0_FAIL
