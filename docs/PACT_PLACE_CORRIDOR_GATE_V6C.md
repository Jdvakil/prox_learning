# PACT pick-and-place corridor: Phase 0 attempt-6c screen (gate)

This is the **decision-bearing** screen after growing clutter presence while
holding the 28 mm expert-clean inner face. v6b is unchanged:
[`docs/PACT_PLACE_CORRIDOR_GATE_V6B.md`](PACT_PLACE_CORRIDOR_GATE_V6B.md)
(`PACT_PLACE_CORRIDOR_PHASE0_PASS`, 20/24, 24 clips). v6 remains
[`docs/PACT_PLACE_CORRIDOR_GATE_V6.md`](PACT_PLACE_CORRIDOR_GATE_V6.md)
(`PACT_PLACE_CORRIDOR_PHASE0_NOT_RUN`). v5 remains
[`docs/PACT_PLACE_CORRIDOR_GATE_V5.md`](PACT_PLACE_CORRIDOR_GATE_V5.md)
(`PACT_PLACE_CORRIDOR_PHASE0_PASS`, 22/24, 152 recovered episodes). The four
earlier FAIL records are unchanged.

The attempt narrative is [`docs/PACT_PLACE_ATTEMPT6C.md`](PACT_PLACE_ATTEMPT6C.md).

## Decision

Phase 0 passed. The attempt-6c prediction, recorded in
`configs/pact_place_corridor_v6c.json` before the first official episode, was
**19–22 clean of 24**. The outcome is **23/24**, above that band on the high
side. Clutter contact on the official screen is **0/24**. The enlargement did
not bite.

A later collection would need a separately frozen contract and the collision
datagen pipeline, not the screen harness, and must leave
`proximity_sensor_period_ms` at its default. Encoder work, policy training, and
learned-policy evaluation are not authorized by this gate. This attempt's
deliverable after the screen is the 24 qpos-replay clips.

The gate required at least **20 clean successes in 24 fixed episodes**. A clean
success means upstream `PickAndPlaceTask` success, zero `hazard_bar` entries,
zero `other_environment` entries, zero `clutter` entries, and zero
`place_receptacle` contact outside placement (`preplace` counted as placement).
That rule is the same as v6b and **stricter** than v5. Do not compare 23/24 to
v5's 22/24 or to v6b's 20/24 as if they were the same endpoint.

## Screens

| Screen | Seed | Clean | Task | What changed |
|---|---|---:|---:|---|
| v1–v4 | 2026081801–2101 | 18, 18, 18, 15 | see prior ledgers | `PHASE0_FAIL` |
| v5 | `2026082201` | **22/24** | 22/24 | Relocated tray; phase-aware receptacle exemption |
| v6 | not frozen | n/a | n/a | A0 grid too tight; `PHASE0_NOT_RUN` |
| v6b | `2026082501` | **20/24** | 20/24 | Clutter at \|y\| = 0.32 after cup-vs-box diagnosis |
| v6c | `2026082701` | **23/24** | 23/24 | Larger boxes, inner face held at 0.29, rear ring out of the wall |

Join rows on `(config_sha256, role_index)`, never `episode_id`.

## The number that matters

The gap that matters is **28 mm** — the clearance between the carried cup and
the nearest box at closest approach. Not the 26 cm it looks like on screen,
which is the distance to where the cup *started*. The expert clears it every
time because it is a precise planner; a learned policy is not, and that gap is
where the ACT-versus-PACT difference is expected to show.

Nominal geometry: inner face `|y| = 0.29`, y half-extent 0.05, x half-extent
0.025, centres `(0.70, ±0.34)` and `(0.75, ±0.34)`, top z = 0.82. Boxes are
**5 × 10 × 10 cm**. Nominal rear outer face is 0.775, inside the shallowest
sampled back wall at 0.780. Jitter is still ±20 mm from `task_seed_u64`, not
`intrusion_side`. With that jitter, one of 24 official episodes had a rear
outer face 5.9 mm past *that episode's* back wall; the inner-face jitter floor
is 0.27, the same as v6b. The inner face was not moved below 0.29.

## Measured Phase-0 results

| Measure | Result |
|---|---:|
| Reconciled rows | 24/24 |
| Clean pick-and-place successes | 23/24 (95.8%) |
| Ordinary task successes | 23/24 (95.8%) |
| Grasp phase successes | 23/24 (95.8%) |
| Place phase successes | 23/24 (95.8%) |
| Place successes given grasp | 23/23 (100%) |
| Episodes with inbound hazard contact | 0/24 |
| Episodes with outbound hazard contact | 0/24 |
| Episodes with other-environment contact | 0/24 |
| Episodes with `pact_clutter` contact | **0/24** |
| Episodes with `place_receptacle` contact outside placement | 0/24 |
| Sampling failures | 0 |
| Infrastructure failures | 0 |
| Bow-fallback episodes | 0/24 |

Config SHA-256
`1dbef6cdd38c6c273d389cbe75717229f78a416831964f9907b00ce4ec58f04e`.
`expert_screen_sha256`
`fef807acfb13ce4ce400d0c0edf323da07a27283382a973486b5604f8f69fc26`.

## Failure localization

The one failure is **retrieval**. Place-given-grasp is 23/23. Zero rows had
clutter, hazard, other-environment, or tray-outside-placement contact,
including the failure.

| Failure class | Count | Row indices |
|---|---:|---|
| `outbound_approach` IK cascade (8 sequential IK failures) | 1 | 7 |
| `lift` empty-gripper | 0 | — |
| `pact_clutter` contact | 0 | — |

| Row | Side | Phase | Branch | What happened |
|---:|---|---|---|---|
| 7 | right | `outbound_approach` | `ik_cascade` | Lifted and carried; 8 sequential IK failures; still holding. Same class as v5 row 3 and v6b rows 9/22. |

## Against the recorded prediction

The frozen v6c contract recorded, before any official episode: predicted clean
**19–22 / 24**. ≥20 passes; 19 is honest and marginal; ≤18 means the
enlargement bit and it returns to A0c, not to tuning. 23/24 is above that
band. Clutter did not add an outbound-contact failure. The leftover retrieval
class appeared once instead of twice.

## What changed before freeze

1. **A0c.** Hold inner face `|y| = 0.29`. Grow y half-extent to 0.05 and
   shorten x half-extent to 0.025. Move x centres to 0.70 / 0.75 so the rear
   outer face is 0.775. Sweep top-z ∈ {0.80, 0.82}. Both candidates
   footprint-ok, enclosure-ok, and expert-trajectory IK-ok. Chosen: taller
   top 0.82. Sweep SHA-256
   `714e0ce8c69b71141207cad5dfe02808a7aa753603a22ec6296ca89bacfd14da`.
2. **8-episode probe** at top 0.82: zero clutter, 6/8 clean. Rows 1 and 6 are
   `ik_cascade` on `outbound_approach`, not clutter. No fallback to 0.80.
3. **One screen.** New master seed `2026082701`. Thresholds unchanged.
   Inner face not moved. Boxes not shrunk. Corridor, aperture, panel, tray,
   and target unchanged.

## Artifacts

- `configs/pact_place_corridor_v6c.json`
- `diagnostics_output/pact_place_clutter_sweep_v6c/analysis.json`
- `diagnostics_output/pact_place_corridor_v6c_clearance_probe/`
- `diagnostics_output/pact_place_corridor_v6c/expert_screen.json`
- `diagnostics_output/pact_place_corridor_v6c/expert_screen_rows/*/result.json`
- `diagnostics_output/pact_place_corridor_v6c/expert_screen_rows/*/trajectory.json`
- `diagnostics_output/pact_place_corridor_v6c_videos/` (clips locally; not committed)

PACT_PLACE_CORRIDOR_PHASE0_PASS
