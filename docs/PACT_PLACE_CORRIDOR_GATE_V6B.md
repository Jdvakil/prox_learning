# PACT pick-and-place corridor: Phase 0 attempt-6b screen (gate)

This is the **decision-bearing** cluttered-shelf screen after v6 stopped at A0.
v6 is unchanged: [`docs/PACT_PLACE_CORRIDOR_GATE_V6.md`](PACT_PLACE_CORRIDOR_GATE_V6.md)
(`PACT_PLACE_CORRIDOR_PHASE0_NOT_RUN`). v5 remains
[`docs/PACT_PLACE_CORRIDOR_GATE_V5.md`](PACT_PLACE_CORRIDOR_GATE_V5.md)
(`PACT_PLACE_CORRIDOR_PHASE0_PASS`, 22/24). The four earlier FAIL records
are unchanged.

The attempt narrative is [`docs/PACT_PLACE_ATTEMPT6B.md`](PACT_PLACE_ATTEMPT6B.md).
Step 0 body identity is
[`docs/PACT_PLACE_V6B_CONTACT_BODIES.md`](PACT_PLACE_V6B_CONTACT_BODIES.md).

## Decision

Phase 0 passed. The attempt-6b prediction, recorded in
`configs/pact_place_corridor_v6b.json` before the first official episode, was
**19–22 clean of 24**. The outcome is **20/24**, at the bar. Clutter contact
on the official screen is **0/24**.

A later collection would need a separately frozen contract and the collision
datagen pipeline, not the screen harness. Encoder work, policy training, and
learned-policy evaluation are not authorized by this gate. This attempt's
deliverable after the screen is the 24 qpos-replay clips.

The gate required at least **20 clean successes in 24 fixed episodes**. A clean
success means upstream `PickAndPlaceTask` success, zero `hazard_bar` entries,
zero `other_environment` entries, zero `clutter` entries, and zero
`place_receptacle` contact outside placement (`preplace` counted as placement).
That rule is **stricter** than v5. Do not compare 20/24 to v5's 22/24 as if
they were the same endpoint.

## Screens

| Screen | Seed | Clean | Task | What changed |
|---|---|---:|---:|---|
| v1–v4 | 2026081801–2101 | 18, 18, 18, 15 | see prior ledgers | `PHASE0_FAIL` |
| v5 | `2026082201` | **22/24** | 22/24 | Relocated tray; phase-aware receptacle exemption |
| v6 | not frozen | n/a | n/a | A0 grid too tight; `PHASE0_NOT_RUN` |
| v6b | `2026082501` | **20/24** | 20/24 | Clutter at \|y\| = 0.32 after cup-vs-box diagnosis |

Join rows on `(config_sha256, role_index)`, never `episode_id`.

## Measured Phase-0 results

| Measure | Result |
|---|---:|
| Reconciled rows | 24/24 |
| Clean pick-and-place successes | 20/24 (83.3%) |
| Ordinary task successes | 20/24 (83.3%) |
| Grasp phase successes | 20/24 (83.3%) |
| Place phase successes | 20/24 (83.3%) |
| Place successes given grasp | 20/20 (100%) |
| Episodes with inbound hazard contact | 0/24 |
| Episodes with outbound hazard contact | 0/24 |
| Episodes with other-environment contact | 0/24 |
| Episodes with `pact_clutter` contact | **0/24** |
| Episodes with `place_receptacle` contact outside placement | 0/24 |
| Sampling failures | 0 |
| Infrastructure failures | 0 |
| Bow-fallback episodes | 0/24 |

Config SHA-256
`ebf1be0359b6ff810772a3d4bbb0adf8913710fae4eec14c88a8f403a70e3671`.
`expert_screen_sha256`
`d37f760c80a68256d76da9047d3b8706e1beec09061588d5bcf231b74c9a508a`.

## Failure localization

All four failures are **retrieval**. Place-given-grasp is 20/20. Zero rows had
clutter, hazard, other-environment, or tray-outside-placement contact,
including the failures.

| Failure class | Count | Row indices |
|---|---:|---|
| `outbound_approach` IK cascade (8 sequential IK failures) | 2 | 9, 22 |
| `lift` empty-gripper (`gripper_width_min_m = 0`) | 2 | 2, 14 |
| `pact_clutter` contact | 0 | — |

| Row | Side | Phase | Branch | What happened |
|---:|---|---|---|---|
| 2 | right | `lift` | `empty_gripper` | Cup max z 0.8018 vs 0.7937 start; ended z 0.7527. Same class as v5 row 10. |
| 9 | left | `outbound_approach` | `ik_cascade` | Lifted and carried; 8 sequential IK failures; still holding. Same class as v5 row 3. |
| 14 | right | `lift` | `empty_gripper` | Cup max z 0.7995 vs 0.7937 start; ended z 0.7527. |
| 22 | right | `outbound_approach` | `ik_cascade` | Lifted and carried; 8 sequential IK failures; still holding. |

## Against the recorded prediction

The frozen v6b contract recorded, before any official episode: predicted clean
**19–22 / 24**. 20/24 is inside that range and meets the bar of 20. Clutter
did not add an outbound-contact failure on this seed. The four misses are the
two retrieval classes v5 already had, each appearing twice.

## What changed before freeze

1. **Step 0.** Re-ran set-13 rows 0 and 4 with contact frames retained. 100%
   of clutter pairs were carried-cup vs the opposite-side inner box. No robot
   link appeared. Axis: lateral.
2. **A0b.** Extended grid `|y| ∈ {0.32, 0.36, 0.40}`. 30/30 footprint-ok,
   enclosure-ok, and expert-trajectory IK-ok. Chosen: set 12, height 0.06 m,
   `|y| = 0.32`. Sweep SHA-256
   `388c23677516a65431b51e28d4a22becaa0b2a491c52f730f37b73e00b8e66f3`.
3. **8-episode probe.** Zero clutter, 7/8 clean. Row 7 is `ik_cascade` on
   `outbound_approach`, not clutter.
4. **One screen.** New master seed `2026082501`. Thresholds unchanged.

## Artifacts

- `configs/pact_place_corridor_v6b.json`
- `diagnostics_output/pact_place_corridor_v6b_contact_bodies/`
- `diagnostics_output/pact_place_clutter_sweep_v6b/analysis.json`
- `diagnostics_output/pact_place_corridor_v6b_clearance_probe/`
- `diagnostics_output/pact_place_corridor_v6b/expert_screen.json`
- `diagnostics_output/pact_place_corridor_v6b/expert_screen_rows/*/result.json`
- `diagnostics_output/pact_place_corridor_v6b/expert_screen_rows/*/trajectory.json`
- `diagnostics_output/pact_place_corridor_v6b_videos/` (clips locally; not committed)

PACT_PLACE_CORRIDOR_PHASE0_PASS
