# PACT pick-and-place Phase 0, attempt 6c

v6b passed at 20/24 with zero clutter contact, but the 6 cm cubes read as
specks. The boxes already overlapped the carried cup vertically; the missing
ingredient was presence, not proximity. This attempt grows the boxes in the
axes that have room, holds the 28 mm inner-face margin, and pulls the rear
ring out of the back wall. The decision is
`PACT_PLACE_CORRIDOR_PHASE0_PASS`.

The v6c gate ledger is
[`PACT_PLACE_CORRIDOR_GATE_V6C.md`](PACT_PLACE_CORRIDOR_GATE_V6C.md).
v6b remains PASS. v6 remains NOT_RUN. v5 remains PASS.

## The screens

| Screen | Master seed | Config SHA-256 | Clean | Decision |
|---|---:|---|---:|---|
| v1 | `2026081801` | `46dff849…559e40` | 18/24 | `PHASE0_FAIL` |
| v2 | `2026081901` | `a0f30725…201c31` | 18/24 | `PHASE0_FAIL` |
| v3 | `2026082001` | `acd7ced5…6694f` | 18/24 | `PHASE0_FAIL` |
| v4 | `2026082101` | `fe45435d…9cdc65` | 15/24 | `PHASE0_FAIL` |
| v5 | `2026082201` | `bd47f1c97d2815657211085590657f5211ca847b776f6039c9617f990da9c1f1` | **22/24** | `PHASE0_PASS` |
| v6 | not frozen | no `configs/pact_place_corridor_v6.json` | n/a | `PHASE0_NOT_RUN` |
| v6b | `2026082501` | `ebf1be0359b6ff810772a3d4bbb0adf8913710fae4eec14c88a8f403a70e3671` | **20/24** | `PHASE0_PASS` |
| v6c | `2026082701` | `1dbef6cdd38c6c273d389cbe75717229f78a416831964f9907b00ce4ec58f04e` | **23/24** | `PHASE0_PASS` |

Join rows on `(config_sha256, role_index)`, never `episode_id`.
v6c clean-success is the same stricter rule as v6b (adds zero clutter). Those
counts are not comparable to v5.

The prediction recorded in the v6c config before any official episode was
**19–22 clean**. 23/24 is above that band and meets the bar of 20.

## Named changes (frozen before the first official v6c episode)

Thresholds were not changed. `gripper_empty_threshold` stayed **0.002 m**.
`MIN_CLEAN_SUCCESSES` stayed **20**. `RELEASE_CLEARANCE_M` stayed **0.005**.
Grasps were not filtered. The tray, target, corridor, aperture, and panel
were not moved. The inner face was not moved below `|y| = 0.29`. Boxes were
not shrunk to force a pass.

1. **Geometry, corrected.** The contacting body is the carried cup, not the
   TCP. Cup bottom sits 21.6 mm below the v6b box top, so height is a
   visibility lever. True lateral margin at closest approach is **28 mm**
   (cup edge 0.262 vs inner face 0.29).
2. **A0c grid.** Inner face held at 0.29. y half-extent 0.05 → centre
   `|y| = 0.34`. x half-extent 0.025, centres 0.70 / 0.75 → rear outer face
   0.775, inside the shallowest back wall at 0.780. Sweep top-z
   `{0.80, 0.82}`; both IK-ok from the expert trajectory. Chosen: 0.82.
   Boxes go from 6×6×6 cm to **5×10×10 cm**.
3. **8-episode probe** at 0.82: 0 clutter, 6/8 clean. Did not fall back.
4. **New sampler rest pose only.** Jitter still from `task_seed_u64`, not
   `intrusion_side`.

## What they did on this seed

| Change | On v6c (`MASTER_SEED = 2026082701`) |
|---|---|
| Larger boxes, same 28 mm inner face | 0/24 official episodes with `pact_clutter` contact. Probe was also 0/8. |
| Stricter clean-success | 23/24. The extra rule did not create the remaining failure. |
| Empty-gripper persist | Did not fire. |
| IK cascade | Fired once (row 7, `outbound_approach`). Same class as v5 row 3. |
| Ceiling | **23/24**, above the 19–22 prediction, at/above the bar of 20. |

## Screen

| Item | Value |
|---|---|
| Master seed | `2026082701` |
| Config | `configs/pact_place_corridor_v6c.json` |
| Config SHA-256 | `1dbef6cdd38c6c273d389cbe75717229f78a416831964f9907b00ce4ec58f04e` |
| Output | `diagnostics_output/pact_place_corridor_v6c/` |
| `expert_screen_sha256` | `fef807acfb13ce4ce400d0c0edf323da07a27283382a973486b5604f8f69fc26` |
| Prediction (pre-registered) | 19–22 clean; bar 20 |
| Outcome | 23/24 clean; 0/24 clutter |

One run. Replay clips: `scripts/run_pact_place_v6c_replay_videos.py`. Do not
edit the v5 renderer or the v6b renderer. Collection, if done later, must use
the collision datagen path, not `run_row` from the screen harness, and must
leave `proximity_sensor_period_ms` at its default.

PACT_PLACE_CORRIDOR_PHASE0_PASS
