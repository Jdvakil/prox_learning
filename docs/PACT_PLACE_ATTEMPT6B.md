# PACT pick-and-place Phase 0, attempt 6b

v6 stopped at A0 because the search grid sat inside the outbound cup sweep.
That stop is unchanged. This is a new A0 on a wider lateral range after
Step 0 named the contacting body. The decision is
`PACT_PLACE_CORRIDOR_PHASE0_PASS`.

The v6b gate ledger is
[`PACT_PLACE_CORRIDOR_GATE_V6B.md`](PACT_PLACE_CORRIDOR_GATE_V6B.md).
v5 remains PASS. v6 remains NOT_RUN.

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

Join rows on `(config_sha256, role_index)`, never `episode_id`.
v6b clean-success is stricter than v5 (adds zero clutter). Those counts are
not comparable.

The prediction recorded in the v6b config before any official episode was
**19–22 clean**. 20/24 is inside that range and meets the bar of 20.

## Named changes (frozen before the first official v6b episode)

Thresholds were not changed. `gripper_empty_threshold` stayed **0.002 m**.
`MIN_CLEAN_SUCCESSES` stayed **20**. `RELEASE_CLEARANCE_M` stayed **0.005**.
Grasps were not filtered. The tray and target were not moved. Boxes were not
shrunk.

1. **Step 0 named the body.** Set-13 rows 0 and 4 with contact frames retained:
   100% carried cup vs opposite-side `pact_clutter_r0` / `_l0`. No robot link.
   Axis: lateral. See [`PACT_PLACE_V6B_CONTACT_BODIES.md`](PACT_PLACE_V6B_CONTACT_BODIES.md).
2. **A0b grid** `|y| ∈ {0.32, 0.36, 0.40}`. 30/30 eligible. Chosen: `|y| = 0.32`,
   height 0.06 m, x 0.72/0.78. Inner faces at 0.29 vs measured outbound TCP
   `|y| ≈ 0.194`.
3. **8-episode probe** at that set: 0 clutter, 7/8 clean.
4. **New sampler rest pose only.** `PactPlaceCorridorV3Sampler` nominal slots
   at `|y| = 0.32`. Jitter still from `task_seed_u64`, not `intrusion_side`.

## What they did on this seed

| Change | On v6b (`MASTER_SEED = 2026082501`) |
|---|---|
| Clutter beside the cup | 0/24 official episodes with `pact_clutter` contact. Probe was also 0/8. |
| Stricter clean-success | 20/24. The extra rule did not create the four remaining failures. |
| Empty-gripper persist | Fired twice (rows 2 and 14, `lift`). Same class as v5 row 10. |
| IK cascade | Fired twice (rows 9 and 22, `outbound_approach`). Same class as v5 row 3. |
| Ceiling | **20/24**, at the bar of 20, inside 19–22. |

## Screen

| Item | Value |
|---|---|
| Master seed | `2026082501` |
| Config | `configs/pact_place_corridor_v6b.json` |
| Config SHA-256 | `ebf1be0359b6ff810772a3d4bbb0adf8913710fae4eec14c88a8f403a70e3671` |
| Output | `diagnostics_output/pact_place_corridor_v6b/` |
| `expert_screen_sha256` | `d37f760c80a68256d76da9047d3b8706e1beec09061588d5bcf231b74c9a508a` |
| Prediction (pre-registered) | 19–22 clean; bar 20 |
| Outcome | 20/24 clean; 0/24 clutter |

One run. Replay clips: `scripts/run_pact_place_v6b_replay_videos.py`. Do not
edit the v5 renderer. Collection, if done later, must not import `run_row`
from the screen harness.

PACT_PLACE_CORRIDOR_PHASE0_PASS
