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
- Reconstruct this ledger from parent commit `84594895c4dcdff7c2d582ce7bc5c15e4562378b` and MolmoSpaces submodule `b00dc3523b0930afdae4e95b1aac0ba7211714f3`. At that pair, `scripts/pact_place_corridor_contract.py` is `fd7953f86973d858e35075bc1a31e0684acba05d76422e82fabcb1e5e431a4f7` and `enclosure_reach.py` is `f1f967f0362b9e94ec42156d071c88c5cdb771f93673720cfb729a72c158b574`, matching `source_sha256` in the v1 config.
- `pact_collision_corridor.xml` retained its pinned SHA-256 `f8c04b07b9416593eb60ad4797ccbae91f7d3524effd30438ef552e5a2d75540`.
- The original `PactCollisionCorridorSampler`, `PactCollisionCorridorPolicy`, and `PactCollisionCorridorPolicyConfig` class bodies match the MolmoSpaces submodule commit exactly.
- Frozen contact-endpoint, geometry-v3, blur-sweep, and blind-RGB artifacts were hash-verified before and after the screen.
- No checkpoint, encoder, threshold, or existing scene was changed.

## Episode IDs collide with the v2 screen

`episode_id` in this contract is `sha256("pact-place-v1:expert:{index}:{side}")`. The master seed is not in the preimage, so the ID is a function of `(index, side)` only. Twelve of 24 IDs are byte-identical to the v2 screen, while 0/24 task seeds and 0/24 panel jitters match. The instances are different; the labels are not.

Join v1 and v2 rows on `(config_sha256, role_index)` — never on `episode_id`. This gate's `config_sha256` is `46dff849dd16eb3b6c0baf169053829bc66203a39866f14a4667e8eaef559e40`. The v2 gate's is `a0f30725e325a73b5584895a07fa18000fe3645cb63ebd1b4e5a6746bc201c31`.

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

- `configs/pact_place_corridor_v1.json`
- `diagnostics_output/pact_place_corridor/expert_screen.json`
- `diagnostics_output/pact_place_corridor/expert_screen_rows/*/result.json`
- `diagnostics_output/pact_place_corridor/stop_record.json`

PACT_PLACE_CORRIDOR_PHASE0_FAIL
