# PACT Place V9.8 — ceiling pendant environment gate

Status: stopped. After the centred-width sweep, v2 offset the pendant to
`y = 0.100` and used side-dependent ceiling envelopes (measured lag + 4 mm).
Step 0 passed both named candidates against live `_bow_segment`. The V9.5
guard stayed **6/8 row-for-row**. Neither offset candidate preserved every
fixture-free clean row: wide and conservative are both 0/8 on the paired
join, unclipped, correct bow side, clutter 0, `mounted_fixture` contact on
every complete row. The 24-row gate was not run. The aperture is not
widened and the panel is not moved. Collection and training remain
unauthorized.

The 2026-08-25 retained-state audit below supersedes the earlier causal
reading that the wrist failed to follow the TCP bow, or that a wrist-height
hazard is unavoidable in the 0.85 m aperture. The paired selection stop
itself is unchanged.

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
   left/right frozen trajectories. The first rule maximized worst-variant
   sensors subject to at least 100 route-intrusion frames and at most 4× side
   imbalance, and admitted `half_y = 0.18` with zero detour slack. S1 v2 keeps
   those floors and also requires fixture-bow detour slack ≥ 20 mm.
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
silhouette of roughly 0.25 m and nothing smaller. The first S1-selected pendant
is 0.36 m wide (`half_y = 0.18 m`) because of that floor; S1 v2 then dropped
that width as zero-slack. Ground clutter remains RGB-only decor and counts as
failure on contact, with no claim that the skin sees it.

## Outcomes

S1 completed on the six physics-clean W1 variants. The first rule evaluated 77
candidates and selected `bottom=1.10 m`, `half_y=0.18 m`: worst-variant 14
sensors clearing 2 px and 181 route-intrusion frames. That width lands the
fixture-bow waypoint exactly on `lateral_limit` (0.305 m) with zero slack.
Artifact: `diagnostics_output/pact_place_v98_pendant_siting/siting.json`.

S1 v2 re-scored the same 77 candidates with detour slack ≥ 20 mm. Forty
candidates remain eligible; `half_y = 0.18` is ineligible. Selected
`bottom1.10_half_y0.16`: 13 sensors, 181 intrusion frames, slack ≈ 0.020 m.
Artifact: `diagnostics_output/pact_place_v98_pendant_siting_v2/siting.json`
(original siting left on disk).

The six-variant S2a frozen-qpos smoke completed with the unchanged
`[40, 4, 8, 8]` renderer, a 0.0 repeat baseline, and both `panel` and
`ceiling_pendant` role floors met. Artifact:
`diagnostics_output/pact_place_v98_pendant_causal_smoke3/validation.json`.
This remains a smoke result and carries `authorizes_gate: false`.

The first S3 run reported 0/24 clean with clutter contact. That run is **void**
as a measurement of the pendant (see below). Artifact retained:
`diagnostics_output/pact_place_v98_expert_gate/expert_screen.json`.

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

The first S3 run left V9.8 off the mounted-fixture lateral bow lists as a design
choice. That choice was reversed in the repair below so the bow's effect could
be measured rather than inferred.

Final verdict for the first S3 run: `authorizes_gate: false`, `authorizes_collection: false`, and
**the run does not support any conclusion about pendant feasibility.**

## Repair: wire the bow, pin the validated seed, re-screen

Two changes, then a new screen. The void-run artifact was not overwritten.

1. **Seed.** `scripts/run_pact_place_v98_pendant_preview.py` now uses
   `DEFAULT_SEED = 955339` with a constant `candidate_seed`, so every row carries
   `task_seed_u32 = task_seed_u64 = 955339`. Panel and target jitters stay at 0.0.
   The 24 rows are therefore **6 distinct instances repeated 4× deterministically**.
   `MIN_CLEAN_SUCCESSES = 20` is unchanged and means **5 of 6 distinct cells**.
2. **Lateral bow.** `pact_place_corridor_v9_8_pendant` was added to the mounted-fixture
   bow path with `fixture_roles = ("ceiling_fixture",)`. Per-row
   `planned_bow_m`, `accepted_bow_m`, and `bow_fallback_taken` are recorded under
   `pendant_bow` / `inbound_ceiling_fixture` / `outbound_ceiling_fixture`.

### Regression guard (required in the same run)

Replay of the 8 stored V9.5 smoke rows against
`diagnostics_output/pact_place_v95_raw_smoke/summary.json['results']`:
**6/8 clean, 8/8 row-for-row match** (F0/F1/F2 both sides clean, F3 both sides
dirty). Artifact: `diagnostics_output/pact_place_v95_smoke_repro_guard/guard.json`.
There is no code regression in the harness or the fixture-free V9.5 layout.

### Paired comparison: 8 V9.5 smoke rows with the pendant added

Same seed, layout, and jitters as the fixture-free 6/8; the only added variable
is the S1 pendant. Artifact:
`diagnostics_output/pact_place_v98_v95_pendant_paired/paired.json`.

| | fixture-free V9.5 | same rows + pendant |
|---|---:|---:|
| clean | 6/8 | **0/8** |
| grasp-phase success | 8/8 (F3 dirty after grasp) | 0/8 |
| inbound pendant bow | — | 0.136–0.160 m, no fallback |
| terminal phase | full task / F3 dirty | `inbound_ceiling_fixture_exit` or `pregrasp` |

The six rows that were clean without the pendant all died on clutter contact
during the inbound pendant bow. That measurement mixed the detour with the
fixture. `_bow_segment` is TCP-only and lateral; the pendant bottom is
`z = 1.10` against a TCP height of about `0.885` m, so the TCP had vertical
clearance and the sideways waypoint did not fit the aperture.

### Paired comparison with the lateral bow off

Same eight V9.5 rows, same pendant, `pact_v98_pendant_lateral_bow` defaulting
false. Artifact:
`diagnostics_output/pact_place_v98_v95_pendant_paired_nobow/paired.json`.
Same-run fixture-free replay:
`diagnostics_output/pact_place_v95_smoke_repro_guard_v3/guard.json` (**6/8**,
8/8 row-for-row, `mounted_fixture = 0` on every row).

| arm | clean | inbound ceiling-fixture bow | contact |
|---|---:|---:|---|
| fixture-free V9.5 | **6/8** | — | F3 clutter only |
| pendant + lateral bow | **0/8** | 0.136–0.160 m | clutter, no mount class |
| pendant, bow off | **0/8** | **0.0 m on all 8** | `mounted_fixture` 20–257, clutter 0 |

Every no-bow row has `collision_free: false`,
`failure_cause: mounted_fixture_collision_contact`, and dies at
`action_index=1` (`pos_err`) in `pregrasp` (one F3-right in
`inbound_vessel_exit`). The class totals already attribute the contact; a
per-body `contact_frames` dump was not required.

**Stop rule applied (bow-off at `half_y = 0.18`):** the pendant obstructs on a
straight TCP path. That measurement still stands. Geometry was not re-tuned
then; the next section tests whether a narrower pendant plus the TCP bow can
pass the same eight rows.

Mounted-fixture contacts are a new class
(`pact_place_robot_environment_v3`, prefix `pact_clutter_mount_`) and count
in `collision_free` / `clean_success`. V9.8's lateral bow is now a row flag
defaulting off; V9.4/V9.5 wall fixtures still bow unconditionally.

### Pinned-seed S3 re-screen (bow on; superseded as the cost attribution)

Artifact: `diagnostics_output/pact_place_v98_expert_gate_v2/expert_screen.json`.
Bow attribution: `diagnostics_output/pact_place_v98_expert_gate_v2/bow_attribution.json`.

| | value |
|---|---|
| decision | `PACT_PLACE_CORRIDOR_PHASE0_FAIL` |
| clean successes | **0/24** (0 of 6 distinct cells) |
| task / grasp success | 0/24 |
| clutter contact | 24/24 |
| hazard-bar contact | 0 |
| inbound pendant bow | 0.143–0.160 m, accepted = planned, no fallback |
| 4× repeats | identical within each cell |

The frozen 20/24 bar, under the pinned seed, is "5 of 6 distinct cells". The
measured result is 0 of 6. S2b was not run. The review wrapper records that
three clean successes were unavailable
(`diagnostics_output/pact_place_v98_pendant_review_v2/review.json`).

This bow-on 24-row gate and the bow-on paired 0/8 at `half_y = 0.18` measured
the zero-slack detour, not a cleared fixture. They remain on disk.

## Width sweep: bow on, detour slack ≥ 20 mm

Same eight V9.5 rows, bow on, bottom z = 1.10 m. Same-run fixture-free
replay: `diagnostics_output/pact_place_v95_smoke_repro_guard_v4/guard.json`
(**6/8**, 8/8 row-for-row). Waypoint algebra:
`waypoint_abs_y = 0.125 + half_y`; slack is 20 / 40 / 60 mm at 0.16 / 0.14 /
0.12. No admitted row clipped (`accepted == planned`).

| `half_y` | slack | clean | inbound bow | `mounted_fixture` | clutter | artifact |
|---|---:|---:|---|---|---:|---|
| 0.18 (bow on, prior) | 0 mm | **0/8** | 0.136–0.160 m | 0 (clutter during detour) | >0 | `pact_place_v98_v95_pendant_paired/` |
| 0.18 (bow off, prior) | 0 mm | **0/8** | 0.0 m | 20–257 | 0 | `pact_place_v98_v95_pendant_paired_nobow/` |
| 0.16 | 20 mm | **0/8** | 0.116–0.140 m, unclipped | 58–157 | 0 | `pact_place_v98_paired_halfy016/` |
| 0.14 | 40 mm | **0/8** | 0.096–0.120 m, unclipped | 60–166 | 0 | `pact_place_v98_paired_halfy014/` |
| 0.12 | 60 mm | **0/8** (7 complete) | 0.077–0.100 m, unclipped | 98–207 | 0 | `pact_place_v98_paired_halfy012/` |

Every complete narrowed row has `failure_cause: mounted_fixture_collision_contact`.
Left-side rows die in `inbound_ceiling_fixture_exit`; right-side rows die in
`pregrasp`. At `half_y = 0.12`, F3-right (role 607) is `sampling_failure`
(settled clutter overlaps the target / IK on retries). That row is already
dirty in the fixture-free baseline (`baseline_clean_success: false`); the
seven complete rows all bowed and all hit the pendant. The paired harness
now scores `inbound_ceiling_fixture_bow_took_effect` on complete rows only,
so that sampling failure is not a void bow measurement.

**Stop rule applied:** all three admitted widths are below 6/8 **with
`mounted_fixture` contact and unclipped waypoints**. The wrist does not
follow the TCP bow; a lateral detour cannot rescue a wrist-height obstacle
in this corridor. Do not widen the aperture or move the panel. The 24-row
gate (`expert_gate_v4`), S2b, and S4 were not run. `authorizes_gate: false`,
`authorizes_collection: false`.

The resolving-power caveat remains binding: the skin resolves a contiguous
silhouette of roughly 0.25 m and nothing smaller. The first S1 width
(`half_y = 0.18 m`) was required by that floor and is zero-slack; the
narrower widths that have slack still fail on the pendant itself. Ground
clutter remains RGB-only decor that counts as failure on contact, with no
claim that the skin sees it.

## Offset v2: lag-aware window, still 0/8 on the paired join

The centred-width stop left a residual: the TCP bow executed, but the wrist
trails the TCP toward the centreline. Contract v2
(`pact_place_v9_8_pendant_v2`) records that measurement as the cause and
allows a +y offset. Ceiling-fixture envelopes are side-dependent
(`CEILING_FIXTURE_ENVELOPE_HALF_Y_NEG = 0.212`, `_POS = 0.112`: design lag
+ 4 mm) and are used only for V9.8 `ceiling_fixture`. Wall fixtures stay at
`MOUNTED_FIXTURE_ENVELOPE_HALF_Y = 0.10`.

Step 0 called live `_bow_segment` (not a hand-copied formula). Both named
candidates sit in the derived face window `[0.044, 0.156]`, unclipped, with
predicted wrist lag clearing the 25 mm gap.
Artifact: `diagnostics_output/pact_place_v98_bow_clearance_predict.json`.

Same-run V9.5 replay after the envelope change:
`diagnostics_output/pact_place_v95_smoke_repro_guard_v5/guard.json`
(**6/8**, 8/8 row-for-row). Wall-fixture routing did not move.

Acceptance on the paired eight is a **per-row join**, not a 6/8 count: every
fixture-free clean row must stay clean with zero `mounted_fixture` contact.
Selection: take the widest candidate that preserves those rows; if neither
does, stop.

| candidate | center_y | half_y | span | preserved baseline-clean rows | clip | wrong-way bow | artifact |
|---|---:|---:|---|---|---|---|---|
| wide | 0.100 | 0.056 | [0.044, 0.156] | **no** (0/8 overall; 6/6 baseline-clean dirty) | none | none | `pact_place_v98_paired_offset_wide/` |
| cons | 0.100 | 0.045 | [0.055, 0.145] | **no** (0/8 overall; 6/6 baseline-clean dirty) | none | none | `pact_place_v98_paired_offset_cons/` |

Left-panel rows bowed −y (waypoint −0.193 wide / −0.182 cons) and died in
`pregrasp`. Right-panel rows bowed +y (waypoint +0.293 / +0.282) and died
in `inbound_ceiling_fixture_exit`. Clutter 0. `mounted_fixture` 553–1239
on every complete row. F3 sampling failures (wide: role 606 left; cons:
role 607 right) were already baseline-dirty.

**Stop rule applied:** neither candidate preserves the baseline-clean
rows. The 24-row gate (`expert_gate_v5`), S2b, and S4 were not run. Do not
widen the aperture or move the panel.

Wrist-lag close-out on the offset-wide qpos
(`diagnostics_output/pact_place_v98_paired_offset_wide/wrist_lag.json`):
`fr3_link6` body origin trails TCP by 0.037–0.038 m on −y bows and
0.040–0.066 m on +y. The same body on the centred `half_y=0.16` source
rows is ~0.05 m, so this metric did not move with the offset. It also does
not reproduce the design 0.208 / 0.108 (quoted as wrist y = −0.061 at TCP
y = −0.268); at that TCP y, link6 is at y ≈ −0.218. The window was not
rebuilt from the body-origin number.

## V9.5 seed robustness (independent of the pendant stop)

Every V9 result still rests on a layout validated at seed 955339. A fixture-free
24-seed sweep of the 8-row V9.5 smoke pattern measured how narrow that seed is.
Artifact: `diagnostics_output/pact_place_v95_seed_fragility/fragility.json`
(`narrow: true`).

| | value |
|---|---|
| n | 24 seeds × 8 rows |
| mean clean rate | **4.08/8 (51%)** |
| validated seed 955339 | 7/8 |
| seeds reaching 7/8 | **4 of 24** |
| canonical varied-seed 24-row expectation | **~12/24 against a bar of 20** |
| range | 0/8 to 7/8 |

955339 is a top-4-of-24 outlier which V9.5, V9.6, and V9.7 all rest on. A
canonical varied-seed screen would expect about 12 clean rows against the
frozen bar of 20, so the seed is narrow. Unreconciled: this sweep scores
955339 at **7/8** where both the original smoke and the regression guards
measured **6/8**; the sweep keeps no per-row detail to settle it. No re-run.

## 2026-08-25 retained-state audit (supersedes the causal close-out)

This entry does not reopen candidate design. The paired selection stop remains
final for the two named offset candidates. Artifact:
`diagnostics_output/pact_place_v98_offset_contact_diagnosis/diagnosis.json`
(`schema_version`: `pact_place_v9_8_offset_contact_diagnosis_v1`).

Working conclusion:

> Both pre-registered offset candidates failed the paired preservation rule, so
> V9.8 stops before the 24-row gate. The failure mechanism and the physical
> validity of the lag-derived window remain unresolved.

The diagnostic restored each complete row's recorded qpos and called
`mj_forward` only. It did not step physics, run the expert, or overwrite the
immutable paired, guard, predictor, or lag files.

### Onset is not terminal phase

Authoritative onset is `contact_audit.first_contact_step.mounted_fixture`
joined to the retained control-step `policy_phase`. Baseline-clean counts
(6 rows per candidate):

| candidate | `early_approach_coverage` | `protected_ceiling_bow_contact` | `post_bow_pregrasp_coverage` | `unreconstructed` |
|---|---:|---:|---:|---:|
| wide | 3 (right 601/603/605) | 0 | 3 (left 600/602/604) | 0 |
| cons | 3 (right 601/603/605) | 0 | 3 (left 600/602/604) | 0 |

Left baseline-clean rows first contact in `pregrasp` after
`inbound_ceiling_fixture_exit`. Right baseline-clean rows first contact in
`inbound_cross_vessel_pass` or `inbound_vessel_pass`, before
`inbound_ceiling_fixture_approach`. Terminal phase on the right is
`inbound_ceiling_fixture_exit` and is not the onset.

Reconstruction TCP residual was ≤ 1 mm on all 12 baseline-clean complete
rows. Live `mounted_fixture` pairs at the authoritative onset control step:

- conservative, both sides, and wide left: present. Left contact is
  `robot_0/fr3_link5_collision` vs `pact_clutter_mount_ceiling_g`. Right
  contact is `robot_0/fr3_link6_collision` vs the same pendant geom.
- wide right 601/603/605: no live pair at the onset step or its ±2 control
  neighbors (production audit samples 2 ms physics). The first later
  reconstructed pair is `robot_0/fr3_link6_collision`, 4–5 control steps
  later. That later geom is not treated as the 2 ms onset identity.

No baseline-clean offset failure starts inside the protected ceiling-fixture
approach/pass/exit set.

### Lag constants and face window

Named quantities on the centred provenance runs, using only rows with TCP
residual ≤ 1 mm (21 of 24; excluded: halfy012 role 607 no trajectory,
halfy014 role 605 and halfy016 role 607 residual > 1 mm):

- TCP-to-`fr3_link6` **body origin** lateral: about 0.042–0.059 m on −y and
  0.033–0.040 m on +y in ceiling-fixture phases. This is not 0.208 / 0.108.
- Collision-facing robot-geom AABB extent vs TCP, x/z overlap with contact:
  about 0.319–0.339 m (−y) and 0.145–0.149 m (+y). Not the design ranges.
- Contact-point lateral vs TCP, and `mj_geomDistance` signed geom gap, also
  fail the source-row ranges (0.198–0.208 / 0.107–0.108) within 2 mm.

Status: `unverified_provenance`. Face window `[0.044, 0.156]`:
`physical_input_invalid`. The body-origin 0.04–0.05 m number was not
substituted into the old formula. No new envelopes or candidates were
derived.

### Predictor and causal category

`diagnostics_output/pact_place_v98_bow_clearance_predict.json` remains an
algebra / dispatch / no-clip check of live `_bow_segment`. It is not
swept-arm clearance.

Top-level category: `route_composition_coverage_failure`.
`authorizes_new_episodes: false`, `authorizes_gate: false`,
`authorizes_collection: false`. The 24-row offset gate, S2b, S4, collection,
and training were not run.
