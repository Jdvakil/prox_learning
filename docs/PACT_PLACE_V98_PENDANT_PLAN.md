# PACT Place V9.8 — ceiling pendant environment gate

Status: pre-registered, not authorized for collection.

V9.8 tests the second form of useful clutter: a symmetric ceiling-mounted fixture
that forces a different vertical route while remaining identical under the two
panel sides. Ground clutter remains RGB-only decor that counts as failure on
contact; this plan makes no claim that the skin sees it.

## Frozen geometry contract

The fixture is a kinematic box on `pact_clutter_mount_ceiling` and is centered at
`y = 0` so its geometry cannot reveal panel side through RGB.

| bound | value |
|---|---:|
| nominal center | `(0.72, 0.0, 1.3325) m` |
| nominal half extents | `(0.10, 0.15, 0.1825) m` |
| ceiling top | `center[2] + half[2] == 1.515 m` exactly |
| center y | `0.0 m` exactly |
| bottom z | `[1.10, 1.20] m` |
| half-width y | `[0.12, 0.18] m` |
| depth | `0.58 <= center[0] - half[0]` and `center[0] + half[0] <= 1.36 m` |
| lateral lane cost | `0.000 m`, asserted |

The admission floor is frozen before measurement and is re-exported unchanged
from `run_pact_place_v96_cluster_causal_proximity.ADMISSION_FLOOR`: at least 3
distinct changed sensors and 448 changed values per role and side, no more than
4× paired-side ratio, and a link5/link6 responder.

## Gate sequence and stop rules

1. S1 sweeps bottom height and half-width against the six physics-clean F0/F1/F2
   left/right frozen trajectories. Selection maximizes worst-variant sensors,
   subject to at least 100 route-intrusion frames in every variant and at most
   4× side imbalance.
2. S2a is an occlusion-aware frozen-trajectory smoke check only and writes
   `authorizes_gate: false`.
3. S3 runs 24 expert rows and requires at least 20 clean successes.
4. S2b repeats the causal admission on six clean episodes drawn from S3.
5. S4 requires human review of three successes and three failures before any
   downstream work.

Stop if S1 has no candidate, S2a diverges materially from the panel calibration,
S3 is below 20/24, or S2b misses the floor on any clean variant. The floor is
never lowered after seeing a result.

The resolving-power caveat remains mandatory: the skin resolves a contiguous
silhouette of roughly 0.25 m and nothing smaller. The S1-selected pendant is 0.36 m wide
(`half_y = 0.18 m`) because of that floor. Ground clutter remains RGB-only decor and counts as
failure on contact, with no claim that the skin sees it.

## Outcomes

S1 completed on the six physics-clean W1 variants. It evaluated 77 candidates
and selected `bottom=1.10 m`, `half_y=0.18 m`: worst-variant 14 sensors clearing
2 px and 181 route-intrusion frames, with the fixed imbalance/intrusion rule
satisfied.

The six-variant S2a frozen-qpos smoke completed with the unchanged
`[40, 4, 8, 8]` renderer, a 0.0 repeat baseline, and both `panel` and
`ceiling_pendant` role floors met. Artifact:
`diagnostics_output/pact_place_v98_pendant_causal_smoke3/validation.json`.
This remains a smoke result and carries `authorizes_gate: false`.

S3 is the stop. All 24 rows completed without infrastructure failure, but all
24 had clutter contact and the clean count was 0/24, below the frozen 20/24
bar. Artifact:
`diagnostics_output/pact_place_v98_expert_gate/expert_screen.json`.
The review wrapper records that three clean successes were unavailable, so S4
cannot substitute for the failed feasibility gate. S2b was not run because the
stop rule forbids raw admission on a failed expert screen.

## S3 re-attribution: the 0/24 measured a mis-wired expert, not the pendant

**The first S3 run is void as a measurement of the pendant.** Re-derived from the raw row
artifacts under `diagnostics_output/pact_place_v98_expert_gate/expert_screen_rows/`:

| observation | value |
|---|---|
| terminal state, all 24 rows | `action_index=1`, `pos_err`, phase `pregrasp` |
| episode steps before failure | 83-110 of a 900-step horizon |
| body contacted | `pact_clutter_06/Soap_Bottle_11` - the bench **inbound vessel** |
| hazard-bar contact | 0 |
| grasp-phase successes | 0 |

Nothing reached the pendant, whose lowest surface is `z = 1.10 m`. The arm failed at bench height
on its first inbound move.

**Root cause: the expert policy was never wired for the V9.8 environment version.**
`PactPlaceCorridorPolicy` gates its V9 routing on hard-coded environment-version allow-lists, and
`pact_place_corridor_v9_8_pendant` was absent from four of them:

| site | effect of the omission |
|---|---|
| `_v9_enabled()` (`enclosure_reach.py:3222`) | **the entire V9 routing block never executed** |
| `inbound_hazard_role` (`:3995`) | inbound bow planned against the *outbound* vessel |
| `V93_OUTSIDE_STAGING_X_M` (`:4203`) | wrong outside-staging depth |
| outbound inbound-vessel bow (`:4278`) | inbound vessel not avoided on the outbound leg |

V9.8 reuses `load_v95_palette` and `build_v95_layout` verbatim, so it must inherit every V9.3/V9.5
*layout* behaviour. All four lists now include `pact_place_corridor_v9_8_pendant`.

The omission survived because **no V9-lineage layout had ever faced the expert screen**: the last
Phase-0 screens on record are v5 (22/24), v6b (20/24) and v6c (23/24). V9.8 was the first, and the
first to exercise this path.

### One list V9.8 is deliberately kept out of

The mounted-fixture **lateral** bow (`:4064`, `:4300`). `_bow_segment` detours only in `y`, and the
pendant is centred on `y = 0`, which maximises the computed detour: `required_bow` came out at
0.29-0.38 m, swinging the arm into the bench clutter. Vertical clearance is the correct response to a
ceiling obstacle. The 3-D `_surf_dist` speed law still sees the pendant. This exclusion is a design
choice, recorded here so it is not mistaken for a further omission, and the re-run tests it: pendant
contact in the re-run would mean vertical avoidance is genuine outstanding work.

Final verdict for the first S3 run: `authorizes_gate: false`, `authorizes_collection: false`, and
**the run does not support any conclusion about pendant feasibility.**

The resolving-power caveat remains binding: the skin resolves a contiguous
silhouette of roughly 0.25 m and nothing smaller. The S1-selected pendant is 0.36 m wide
(`half_y = 0.18 m`) because of that floor. Ground clutter remains RGB-only decor that counts as
failure on contact, with no claim that the skin sees it.
