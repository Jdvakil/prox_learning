# PACT pick-and-place corridor: Phase 0 attempt-5 screen (gate)

This is the **decision-bearing** fresh-seed screen after relocating the
outside tray and making `place_receptacle` contact visible outside placement.
Four prior screens failed (18, 18, 18, 15 of 24). Those FAIL records are not
rewritten:
[`docs/PACT_PLACE_CORRIDOR_GATE.md`](PACT_PLACE_CORRIDOR_GATE.md),
[`docs/PACT_PLACE_CORRIDOR_GATE_V2.md`](PACT_PLACE_CORRIDOR_GATE_V2.md),
[`docs/PACT_PLACE_CORRIDOR_GATE_V3.md`](PACT_PLACE_CORRIDOR_GATE_V3.md),
[`docs/PACT_PLACE_CORRIDOR_GATE_V4.md`](PACT_PLACE_CORRIDOR_GATE_V4.md).

This is attempt 5. The clean-success rule is **stricter** than v1–v4:
`place_receptacle` contact is exempt only during placement, and `preplace` is
treated as placement. The four prior screens are not comparable on this endpoint.

## Decision

Phase 0 passed. The attempt-5 prediction, recorded in
`configs/pact_place_corridor_v5.json` before the first official episode, was
**20–23 clean of 24**. The outcome is **22/24**. Demonstration collection is
authorized under a separately frozen collection contract. Encoder work, policy
training, and learned-policy evaluation are not authorized by this gate.

The gate required at least **20 clean successes in 24 fixed episodes**. A clean
success means upstream `PickAndPlaceTask` success, zero `hazard_bar` entries,
zero `other_environment` entries, and zero `place_receptacle` contact outside
placement (`preplace` counted as placement).

## All five screens

| Screen | Seed | Clean | Task | What changed since the previous |
|---|---|---:|---:|---|
| v1 original | `2026081801` | 18/24 | 18/24 | Original privileged expert gate |
| v2 expert-fix | `2026081901` | 18/24 | 20/24 | `RELEASE_CLEARANCE_M = 0.005` |
| v3 attempt 3 | `2026082001` | 18/24 | 18/24 | Initial-contact reject; subdivide `outbound_approach` |
| v4 attempt 4 | `2026082101` | 15/24 | 15/24 | Disarm empty-gripper on `placement_descent`; persist N=3 on transport |
| v5 attempt 5 | `2026082201` | **22/24** | 22/24 | Relocate/shrink tray; phase-aware receptacle exemption |

Join rows on `(config_sha256, role_index)`, never `episode_id`.

## Measured Phase-0 results

| Measure | Result |
|---|---:|
| Reconciled rows | 24/24 |
| Clean pick-and-place successes | 22/24 (91.7%) |
| Ordinary task successes | 22/24 (91.7%) |
| Grasp phase successes (cup retrieved outside aperture) | 22/24 (91.7%) |
| Place phase successes | 22/24 (91.7%) |
| Place successes given grasp | 22/22 (100%) |
| Episodes with inbound hazard contact | 0/24 |
| Episodes with outbound hazard contact | 0/24 |
| Episodes with other-environment contact | 0/24 |
| Episodes with `place_receptacle` contact outside placement | 0/24 |
| Sampling failures | 0 |
| Infrastructure failures | 0 |
| Bow-fallback episodes | 0/24 |

## Failure localization

Both task failures are **retrieval** failures. Place-given-grasp is 22/22.
Zero rows had tray contact outside placement, including the two failures.

| Failure class | Count | Row indices |
|---|---:|---|
| `outbound_approach` IK cascade (8 sequential IK failures) | 1 | 3 |
| `lift` empty-gripper (persist N=3) | 1 | 10 |
| Tray contact outside placement | 0 | — |

| Row | Phase | Branch | What happened |
|---:|---|---|---|
| 3 | `outbound_approach` | `ik_cascade` | Holding (8.5 mm). Position error 2.5 cm, under the 10 cm tracking threshold. Eight sequential IK failures. No tray contact. Same reach-limit class as v4 rows 9 and 10. |
| 10 | `lift` | `empty_gripper` | Persist N=3 fired (`empty_gripper_streak = 3`, `inter_finger_dist_m = 0`). No tray contact. Threshold stayed 0.002 m. |

## Against the recorded prediction

The frozen v5 contract recorded, before any official episode: predicted clean
**20–23 / 24** if the seven v4 tray-involved failures convert and the two
reach-limited ones persist. 22/24 is inside that range. The two remaining
failures are retrieval without tray contact.

## What changed (frozen before the first official episode)

Relocating the tray does **not** make the corridor, panel, aperture, target
sampling, or the policy's avoidance problem easier. It removes a self-inflicted
collision with furniture that sat in the arm's inbound/outbound sweep and was
invisible to v1–v4 clean-success.

1. **A0 — static IK sweep, no rollouts.** 26 candidates (current centre plus
   x ∈ {0.30, 0.35, 0.40} × y ∈ {0.20, 0.24, 0.28, 0.32}, two footprints).
   All 26 were kinematically reachable from reset and from a carry qpos.
   Three cleared |y| = 0.107 by ≥ 8 cm, all `shrunk_0.10x0.10` at y = 0.32.
   Chosen: centre **(0.35, 0.32) m**, footprint half-y 0.10 m, clearance
   **11.3 cm**. Sweep SHA-256
   `b657da019b8638ba8b94e1bfa64a1d31ddfa7c27d7a7c6f4b6f22824602e211b`.
2. **A1 — fork XML and probe.** `pact_place_corridor_v2.xml` moves only
   `place_pedestal`, `place_receptacle`, and `place_receptacle_lips` (and
   shrinks those geoms). `pact_place_corridor_v1.xml` is unchanged. An
   8-episode diagnostic on the first eight v5 seeds showed **zero**
   `place_receptacle` contact outside placement (7/8 clean; row 3 failed on
   IK with no tray contact).
3. **A2 — phase-aware exemption.** Clean-success now fails on tray contact in
   inbound / outbound / other. `_traversal_phase` maps `preplace` to
   `placement` so the existing phase buckets can enforce that. Collision-free
   in the audit still ignores the tray, as before; the gate rule does not.
4. **A3 — one screen.** Master seed `2026082201`. `MIN_CLEAN_SUCCESSES = 20`.
   `RELEASE_CLEARANCE_M = 0.005`, empty-gripper disarm + persist-3, initial
   observation contact reject, and Fix 2 reverted were kept.

`PactPlaceCorridorSampler` and the v1 scene were not edited.
`PactPlaceCorridorV2Sampler` is a subclass.

## Operational note on the 24-row run

Two gate supervisors were started against the same output directory (a
duplicate launch). They were killed after OpenBLAS thread exhaustion produced
transient `sampling_failure` prints. Incomplete row directories were removed.
A single supervisor resumed, skipped 19 already-complete terminals whose
`config_sha256` matched the frozen contract, and finished the remaining five
rows. The reconciled ledger is 24/24 `complete` with matching config hash.
This is not a second scientific screen.

## Phase B — the contact-endpoint result is already an occluded-hazard result

No new visibility contrast was built. On the existing 285-episode corridor
collection (`diagnostics_output/pact_contact_endpoint/occlusion_subset.json`):

```
panel_visible_steps            = 0     in 285/285 episodes
panel_geometry_occluded_steps  = 0     in 285/285
vision_disadvantaged_fraction  = 1.0   in 285/285
```

The panel is never inside the wrist-camera frustum during the pregrasp window.
The policy consumes only `wrist_camera`
(`submodules/act/eval_pact_frontend_screen_row.py`). The sampler's
`cam_visible` flag is `True` in 24/24 v5 rows because `_cam_visible_label`
ORs the wrist camera with a synthetic exo camera the policy never sees, so it
cannot be the contrast variable.

The contact-endpoint subset analysis was dropped as degenerate
(`docs/PACT_CONTACT_ENDPOINT_DECISION.md` is not rewritten). That dropped
partition is the occluded-hazard framing of the result already in hand:
proximity substituting for vision that structurally cannot see the panel, in
285/285 episodes. A manipulated visible-hazard contrast needs its own
preregistration after this Phase 0 close.

## Integrity checks

- Frozen attempt-5 config SHA-256: `bd47f1c97d2815657211085590657f5211ca847b776f6039c9617f990da9c1f1`.
- Master seed: `2026082201`.
- `expert_screen_sha256`: `7c4cc9ad4740c1e6bbd4be4ee0f31581854c7365634b07d0daa979224363d92f`.
- Parent `63813822e4847caaef3d5b136b4a5ebb3cb5afb4`; MolmoSpaces submodule
  `1cbb1800db66c871f41f2afc3a360affd1b40f1d` with additive place-corridor
  subclasses only.
- At freeze, `enclosure_reach.py` is
  `5bf3d2d443ec146178898ffbe5ecc0a1d82f8c5356684502fab677671de938ff` and
  `scripts/run_pact_place_expert_screen.py` is
  `29a8796ae8ecdf98ea8f644b7ba75ec109f1e79df7357496e4415e286404fb85`.
- `pact_collision_corridor.xml` retained
  `f8c04b07b9416593eb60ad4797ccbae91f7d3524effd30438ef552e5a2d75540`.
- `pact_place_corridor_v1.xml` is unchanged
  (`d853e27ca453a246a73a7fa590e3a05f5c93db0893805cd41067e73441aba942`).
- Upstream `pick_and_place_planner_policy.py` and
  `base_object_manipulation_planner_policy.py` match `9ee36978…` and
  `a7ee3570…`.
- Frozen contact-endpoint, geometry-v3, blur-sweep, and blind-RGB artifacts
  were hash-verified after the screen (12/12).
- The failed v1–v4 artifacts were not overwritten.

## Episode IDs do not collide with v1–v4

This contract includes the master seed in the `episode_id` preimage. All 24 IDs
are disjoint from the earlier screens. Join any pair of screens on
`(config_sha256, role_index)` — never on `episode_id`.

## Artifacts

- `configs/pact_place_corridor_v5.json`
- `diagnostics_output/pact_place_reachability_sweep/analysis.json`
- `diagnostics_output/pact_place_corridor_v5_clearance_probe/`
- `diagnostics_output/pact_place_corridor_v5/expert_screen.json`
- `diagnostics_output/pact_place_corridor_v5/expert_screen_rows/*/result.json`
- `diagnostics_output/pact_place_corridor_v5/expert_screen_rows/*/trajectory.json`
- `diagnostics_output/pact_place_corridor_v5/expert_screen_rows/*/initial_observation_accepted.json`
- `configs/pact_place_corridor_v5_collection.json` (310 candidates, target 255 clean)
- Collection stopped by user at 152 kept / 174 attempted / 22 discarded.
  JSON dump (not ACT HDF5): https://huggingface.co/datasets/Lundii/pact_place_corridor_v5

PACT_PLACE_CORRIDOR_PHASE0_PASS
