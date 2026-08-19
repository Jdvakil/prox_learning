# PACT pick-and-place Phase 0, attempt 5

This is a fifth screen after four failures (18, 18, 18, 15 of 24). It was
defensible because seven of nine attempt-4 failures had the arm on the tray at
the terminal step, and that contact was exempt from clean-success. The
prediction was recorded before the first official episode. The decision is
`PACT_PLACE_CORRIDOR_PHASE0_PASS`.

The v5 gate ledger is [`PACT_PLACE_CORRIDOR_GATE_V5.md`](PACT_PLACE_CORRIDOR_GATE_V5.md).
The four prior FAIL records are unchanged.

## The five screens

| Screen | Master seed | Config SHA-256 | Clean | Task | Place given grasp | Inbound hazard | Decision |
|---|---:|---|---:|---:|---|---:|---|
| v1 original | `2026081801` | `46dff849dd16eb3b6c0baf169053829bc66203a39866f14a4667e8eaef559e40` | 18/24 | 18/24 | 18/22 | 0/24 | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |
| v2 expert-fix re-screen | `2026081901` | `a0f30725e325a73b5584895a07fa18000fe3645cb63ebd1b4e5a6746bc201c31` | 18/24 | 20/24 | 20/20 | 2/24 | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |
| v3 attempt 3 | `2026082001` | `acd7ced5c7e5a0ea8f6a0070f98d507ea1ee0e983fe416f67273c786e596694f` | 18/24 | 18/24 | 18/19 | 0/24 | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |
| v4 attempt 4 | `2026082101` | `fe45435d55cda8daee71972451ce8b460e641b71558aa8a69b3a4686319cdc65` | 15/24 | 15/24 | 15/15 | 0/24 | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |
| v5 attempt 5 | `2026082201` | `bd47f1c97d2815657211085590657f5211ca847b776f6039c9617f990da9c1f1` | **22/24** | 22/24 | 22/22 | 0/24 | `PACT_PLACE_CORRIDOR_PHASE0_PASS` |

Join rows on `(config_sha256, role_index)`, never `episode_id`.

v5 clean-success is not comparable to v1–v4: tray contact outside placement now
fails the row. `preplace` is treated as placement for that exemption.

The prediction recorded in the v5 config before any official episode was
**20–23 clean**. 22/24 is inside that range.

## Named changes (frozen before the first official v5 episode)

Thresholds were not changed. `gripper_empty_threshold` stayed **0.002 m**.
`MIN_CLEAN_SUCCESSES` stayed **20**. `RELEASE_CLEARANCE_M` stayed **0.005**.
Planning-time bow IK was not reintroduced. Grasps were not filtered.

1. **Relocate and shrink the tray.** A0 (static IK, no rollouts) picked the
   closest reachable centre whose footprint clears the |y| = 0.107 traversal
   band by 8 cm: **(0.35, 0.32) m**, `shrunk_0.10x0.10`, 11.3 cm clearance.
   All 26 sweep candidates were reachable; clearance, not workspace, was the
   constraint. Fork `pact_place_corridor_v2.xml`; leave v1 byte-identical.
2. **Phase-aware tray exemption.** Exempt `place_receptacle` only during
   placement, with `preplace` mapped into that bucket. Contact during
   pregrasp / inbound / transport is a failure.
3. **Keep the attempt-4 empty-gripper subclass.** Disarm on
   `placement_descent`; persist N=3 on transport.
4. **New sampler subclass only.** `PactPlaceCorridorV2Sampler`.
   `PactCollisionCorridor*` class bodies and upstream planner files unchanged.

## What they did on this seed

| Change | On v5 (`MASTER_SEED = 2026082201`) |
|---|---|
| Tray out of the sweep | 0/24 episodes with tray contact outside placement. A1 probe was also 0/8. |
| Stricter clean-success | Still 22/24. The extra rule did not create the two remaining failures. |
| Seven v4 tray-involved failures | Converted in the sense that this seed's failures are not tray collisions. |
| Empty-gripper persist | Fired once (row 10, `lift`, streak 3). Not a one-step glitch. |
| Ceiling | **22/24**, above the bar of 20. |

v5 clean failures are rows **3** and **10**. Both are retrieval. Detail is in
the v5 ledger.

## Screen

| Item | Value |
|---|---|
| Master seed | `2026082201` |
| Config | `configs/pact_place_corridor_v5.json` |
| Config SHA-256 | `bd47f1c97d2815657211085590657f5211ca847b776f6039c9617f990da9c1f1` |
| Output | `diagnostics_output/pact_place_corridor_v5/` |
| `expert_screen_sha256` | `7c4cc9ad4740c1e6bbd4be4ee0f31581854c7365634b07d0daa979224363d92f` |
| Prediction (pre-registered) | 20–23 clean; bar 20 |
| Outcome | 22/24 clean |

One run. Collection is authorized; encoder/training/eval are not.
Collection was stopped at 152 kept of 174 attempted. The kept JSON episodes
are at https://huggingface.co/datasets/Lundii/pact_place_corridor_v5 (not ACT
HDF5; no wrist RGB / proximity).

## Phase B (free)

The 285-episode contact-endpoint occlusion subset is 100% vision-disadvantaged
because the panel is never in the wrist camera. Report that as an
occluded-hazard result; do not treat `cam_visible` as a contrast. See the v5
ledger.

PACT_PLACE_CORRIDOR_PHASE0_PASS
