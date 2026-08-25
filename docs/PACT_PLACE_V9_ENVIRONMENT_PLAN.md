# PACT place V9.2: active side panel with feasible staggered clutter

**Scope:** environment, expert, sensor-evidence gate, human review, and Phase 0. Collection,
training, and evaluation remain in
[`PACT_PLACE_V9_TRAIN_EVAL_PLAN.md`](PACT_PLACE_V9_TRAIN_EVAL_PLAN.md) and are not authorized by
this document.

## Current status

V9.2 restores exactly one physical left/right intrusion panel per episode while retaining the
two-dimensional clutter redesign. The visible clutter is identical across panel sides, the panel
selects the open lane, and a centred movable bottle tightens that lane at one of four depth
stations. Static checks and one fixed-seed expert smoke in every family/side variant pass. Raw
causal PACT proximity, fresh human review, and Phase 0 remain pending.

```
geometry/layout validation       PASS (8 family/side variants)
fixed-seed expert smoke          PASS (8/8 clean; diagnostic only)
active physical panel            PASS (8/8)
panel or clutter contact         PASS (0/8 episodes)
raw causal PACT proximity        PENDING
fresh V1b human review           NOT RUN
Phase-0 24-row gate              NOT RUN / NOT AUTHORIZED
collection or training           NOT AUTHORIZED
```

Machine-readable evidence:

- `diagnostics_output/pact_place_v9_panel_smoke/summary.json`
- `diagnostics_output/pact_place_v9_panel_redesign/validation.json`

## Why the earlier V9 designs were invalid

The first design put all clutter at `x = 0.715 m`, creating a transverse object wall. Its V0c
measurement admitted no vessel pair, but downstream review used a hard-coded fallback and produced
`0/24` clean successes. Those artifacts remain immutable invalid-design evidence:

- `diagnostics_output/pact_place_v9_v0c/siting.json`
- `diagnostics_output/pact_place_v9_v1b/review_manifest.json`

V9.1 fixed the line layout by parking the original left/right panel and using one movable bottle as
the route blocker. It proved that scattered clutter can be feasible, but disabling the panel was
outside the intended mission. V9.2 supersedes V9.1 for future work. The old V9.1 smoke and gallery
remain historical only.

The lesson is narrower than "panel plus clutter is impossible." The infeasible geometry used a
panel and opposed vessels at nearly the same short station, closing both lanes. V9.2 instead treats
the panel and bottle as a same-corridor compound obstacle with exactly one admitted lane.

## What PACT-32D contributes

PACT is ACT plus whole-arm proximity: 40 link-mounted sensors, each rendering an 8x8 depth patch,
with a frozen encoder producing a 32-D embedding per sensor. Plain ACT gets the same wrist RGB and
robot state but no proximity projection. The valid modality control is `PACT_PERMUTED`, not an
all-zero field, because zero proximity is out of distribution.

The environment must therefore satisfy both:

1. Panel/blocker geometry requires a collision-free action change.
2. The hidden panel or blocker measurably changes the real proximity tensor during the decision
   window.

The expert may use privileged geometry. Layout coordinates and object AABBs are not student
observations. A PACT advantage must come from recorded proximity and survive the preregistered
permuted-token control.

## V9.2 environment design

### One active panel and one centred movable blocker

The original panel geometry is unchanged:

- one of `pact_intrusion_left` or `pact_intrusion_right` is active;
- panel centre `x = 0.615 m`, half extents `(0.055, 0.240, 0.090) m`;
- panel x jitter is at most `0.015 m` and inner-face jitter at most `0.005 m`;
- the inactive panel remains parked below the scene.

`Soap_Bottle_30` is the only movable route blocker. Its centre is always at `y = 0`, independent of
panel side. This matters scientifically: ACT cannot infer the hidden panel side from a mirrored
RGB-visible bottle position. A left panel requires a `-y` route and a right panel requires `+y`.

The loaded envelope uses half-width `0.15 m`, panel gap `0.14 m`, bottle gap `0.04 m`, and aperture
edge reserve `0.02 m`. Under worst-case panel jitter, the compound obstacle requires centreline
offset `0.2347 m` inside a `0.255 m` limit, leaving `0.0203 m` additional corridor margin. The
panel-to-bottle surface gap is at least `0.0503 m`, so their overlapping x ranges do not create an
initial collision.

### Frozen two-dimensional scatter

| slot | object | x (m) | y (m) | function |
|---|---|---:|---:|---|
| 00 | `Candle_2` | 0.820 | -0.350 | perimeter vessel/control |
| 01 | `Soap_Bottle_30` | family-specific | 0.000 | movable route blocker |
| 02 | `Mug_2` | 0.840 | +0.310 | decor |
| 03 | `Plate_10` | 0.980 | -0.220 | decor |
| 04 | `Plate_22` | 1.090 | +0.300 | decor |
| 05 | can | 1.210 | -0.280 | decor |
| 06 | can | 1.250 | +0.200 | decor |
| 07 | `Candle_1` | 1.060 | +0.020 | decor |

The contract rejects fewer than six distinct x coordinates, x span below `0.40 m`, workspace
escape, object overlap, an inactive/mismatched panel, visible clutter differences between panel
sides, a direct route that is not blocked, or a compound corridor with no admitted detour.

### Route families

| family | blocker x (m) | left panel | right panel | fixed-seed clean smoke |
|---|---:|---|---|---|
| F0 target-side stagger | 0.655 | `-y` | `+y` | both sides |
| F1 inner-panel stagger | 0.630 | `-y` | `+y` | both sides |
| F2 outer-panel stagger | 0.605 | `-y` | `+y` | both sides |
| F3 aperture-side stagger | 0.580 | `-y` | `+y` | both sides |

Visible object coordinates are byte-identical between the left/right variants of each family.
Blocker depth changes where the broad panel route tightens around the bottle.

## Expert behavior

`PactPlaceCorridorPolicy` composes two outbound maneuvers:

1. bow away from the active panel, keeping the original `0.14 m` panel gap;
2. while retaining that lane, tighten locally around the centred bottle to keep `0.04 m` surface
   gap.

The selected lane comes from panel geometry, not the sign of the bottle's y coordinate. This avoids
the old centreline ambiguity where a centred obstacle could make the planner choose the closed
side. Compound-segment diagnostics preserve the strongest accepted bow instead of being erased by
a later non-crossing segment.

In the retained eight smokes:

- every episode completed the pick and place cleanly;
- every scene reported one active panel and `protrusion_present=true`;
- left-panel accepted bow was approximately `0.3032 m` and right-panel bow `0.0768 m` from each
  route's original crossing point;
- every bottle introduced a further `0.0447 m` local tightening;
- panel contacts, clutter contacts, and stability events were all zero.

These are matched-seed feasibility probes, not a success-rate estimate or evidence that PACT beats
ACT.

## Validation sequence

### V0b - palette: complete

The retained palette uses stable, non-rolling decor and the measured tall bottle. Artifact:
`diagnostics_output/pact_place_v9_v0b/palette_v9_1.json`. Its filename is historical; the same
frozen asset measurements are reused by V9.2.

### V0c.1 - static panel/layout admission: complete

For every family and panel side, validate two-dimensional spread, object bounds/overlap, exact
panel-side binding, panel-active state, visible-clutter side invariance, direct-route obstruction,
panel/bottle separation, and the worst-case admitted lane.

Implemented by `pact_place_v9_contract.validate_layout` and
`scripts/validate_pact_place_v9_redesign.py`.

### V0c.2 - live expert feasibility: complete as a smoke

The retained matched-seed run covers all eight family/side variants. Admission requires task and
clean success, non-zero accepted panel and bottle bows, collision-free audit, no clutter stability
event, and a physically active panel. All eight pass.

This is not V1b and not a clean-rate estimate. Development failures remain in the smoke tree and
are summarized rather than deleted.

### V0c.3 - executed, but invalidated by the rejected V1b review

Use the real datagen observation path and replay identical qpos with the committed compound hazard
present and in controlled counterfactual states. At minimum, compare:

1. panel and bottle present;
2. panel parked, bottle unchanged;
3. bottle parked, panel unchanged.

Record all 40 raw `(substep, 8, 8)` streams, hashes, phase alignment, causal deltas per sensor/link,
first activation step, and distance to the next route change. Wrist visibility is report-only.
Sampled cones and AABB distance are not substitutes for the actual proximity tensor.

Every family/side variant must show non-degenerate hazard-caused proximity in the relevant decision
window. Any silent variant stops the pipeline and triggers a new geometry revision.

The frozen validation is
`diagnostics_output/pact_place_v9_v0c3_causal_proximity/validation.json`. All eight family/side
variants pass with a zero repeat-render noise floor. The active-panel counterfactual changes
28,836--33,896 raw depth values per variant and the centred-bottle counterfactual changes 40--668.
All raw three-world tensors and their hashes are retained alongside the validation. This artifact
is no longer sufficient for admission: it measured only the outbound decision window, while V1b
row 9 first struck and displaced the bottle during inbound pregrasp. It also admitted a
side-confounded blocker signal (left detected in 5/6 review rows, right in 0/5). A redesigned
environment requires fresh inbound and outbound causal windows with side-balanced excitation.

### V1b - complete and rejected; gate frozen

Only after V0c.3 passes:

1. Use a seed stream disjoint from Phase 0.
2. Balance all families over real left/right panel sides.
3. Render wrist RGB, third person, route-review view, active panel/bottle telemetry, hardened
   clearances, causal proximity, and free-body drift.
4. Stop after 3 clean successes and 3 failures, capped at 24 attempts.
5. Emit `authorizes_gate: false`, `authorizes_collection: false`, and
   `clean_rate_is_not_an_estimate: true`.
6. Stop for explicit review approval.

The disjoint review stopped at attempt 10 with 8 clean successes and 3 genuine failures. Its
self-hashed manifest is
`diagnostics_output/pact_place_v9_v1b_redesign/review_manifest.json`. The three-pane clips retain
the original matte panel material; only the movable bottle is orange. The burned-in sensor-cone
flag is report-only, while causal admission comes exclusively from the linked raw V0c.3 artifact.
The review was rejected after audit. The blocker was detected in 5/6 left-panel rows and 0/5
right-panel rows; outbound vessel bow was a precomputed constant equal to the bottle half-width;
the nominal inbound vessel never caused a bow; all clutter jitter was zero; and the four families
were only a one-dimensional centreline x sweep. Row 9 struck the bottle inbound, displaced it
21.8 mm and rotated it 0.209 rad before the empty-gripper termination. Its filename described the
terminal symptom instead of the causal clutter collision. V2 is not authorized. V9.2 must be
replaced, revalidated, and reviewed on a new disjoint stream.

The old reviews and V9.1 gallery are invalid for V9.2 and must not be appended or overwritten.

### V2 - Phase-0 gate: frozen after rejected V1b

After explicit approval, freeze a new config hash and predicted clean band before the first row.
Run 24 untouched rows, six per family and side-balanced within family. The structural floor remains
at least `20/24` clean successes. On failure, stop and report; do not tune, substitute, or retry the
gate.

## Verification commands

```bash
python3 -m py_compile \
  scripts/pact_place_v9_contract.py \
  scripts/run_pact_place_v9_panel_smoke.py \
  scripts/run_pact_place_v9_v1b_review.py \
  scripts/validate_pact_place_v9_redesign.py

python3 -m unittest tests.test_pact_place_v9_redesign
python3 scripts/pact_place_v9_contract.py
python3 scripts/run_pact_place_v9_panel_smoke.py --workers 4
python3 scripts/validate_pact_place_v9_redesign.py
```

These validate geometry and development smoke evidence only. They do not execute V0c.3, V1b,
Phase 0, collection, training, or evaluation.

## Non-negotiable constraints

- Exactly one original side panel remains physically active in every V9.2 episode.
- Visible clutter coordinates cannot encode panel side.
- Keep one centred movable route blocker; perimeter/decor objects cannot close the admitted lane.
- A null or silent causal proximity result stops downstream execution.
- Do not claim PACT relevance from expert geometry, AABBs, cones, or object height alone.
- Do not claim a PACT advantage without actual 32-D inputs plus `PACT_PERMUTED` evaluation.
- Do not reuse V1b episodes in Phase 0 or treat development smokes as a clean-rate estimate.
- Any post-V1b environment change invalidates V0c.3 and requires a fresh V1b.
- Do not start collection, conversion, training, or evaluation from this document.
- Preserve all prior v5-v9 artifacts; new work writes to versioned paths.
