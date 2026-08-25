# v8c: an overhead protrusion in the cup-free band, plus honest gates

*Every constant and measurement below was read from source or from the v7 swept-volume analysis.
Nothing is estimated.*

## Context

v8b **failed admission** and stopped correctly before B5b
(`diagnostics_output/pact_place_clutter_sweep_v8b/analysis.json`). Its structural gates passed —
8 objects/layout, 2 overhead/layout, category cap, an 18-uid stability-filtered palette, and a
mount-slot scoring check confirming a mocap mount contact scores unclean. **All five behavioural
gates failed**, and not marginally:

| gate | required | measured |
|---|---|---|
| `cup_is_closest_body` | ≤ 0.25 | **0.833 (5/6)** |
| `frames_link_clearance_lt_10cm` | ≥ 1852 | **34** |
| `lt_5cm` in ≥ half of episodes | ≥ 3/6 | **1/6** |
| `mean_distinct_links_exposed` | ≥ 3 | **0.5** |
| non-zero `visibility_at_min` | ≥ 1/3 | **0/6** |

Pass 3 rescored all 1,152 candidates with exact `mj_geomDistance` and could fill **1 of 12**
family/side quota cells.

### Why — three measured facts

**1. One gate was unsatisfiable and that is my error.** From
`diagnostics_output/pact_place_swept_volume_v7/analysis.json`, link occupancy of the enclosure
interior:

```
link1 0.0%   link2 0.0%   link3 0.0%   link4 0.0%
link5 62.8%  link6 49.2%  link7 29.2%  hand 24.7%  cup 25.3%
```

**Links 1–4 never enter the enclosure at all.** Only two links can ever be exposed, so
`mean_distinct_links_exposed >= 3` was impossible by construction.

**2. The cup and fingers are kinematically always closest.** Minimum clearance to v6c clutter:

```
cup 0.000 < fingers 0.045 < link5 0.057 < hand 0.081 < link6 0.088
   < link7 0.099 < link4 0.104 < link3 0.161 < link2 0.268 < link1 0.272
```

They are the most distal bodies, so they enter any free space first. Pass 3's own diagnosis:
*"real geometry puts the carried cup closer than an arm link for nearly all non-colliding candidates."*

**3. There is almost nowhere to put clutter.** Only **18.2%** of the enclosure is outside the swept
volume, and with a 30 mm keep-out what remains is peripheral — which is exactly where v7's optimiser
fled (wall tiles at `wrist_fov_visibility: 0.0`).

### The opening this run exploits

Swept AABBs, clipped to the enclosure:

```
cup    z 0.692 - 1.011
link6  z 0.866 - 1.390
link5  z 0.819 - 1.415
```

**z ∈ [1.05, 1.40] is traversed by link5 and link6 but never by the cup.** An obstacle placed there is
**link-primary by construction** — the first design since v6c that can satisfy `cup_is_closest_body`
rather than work around it.

### Why an obstacle, not more clutter

Clutter cannot occupy that band because the expert does not know about it. The hazard panel can,
because it **is** in `obstacle_aabbs` (`enclosure_reach.py:1385`) while clutter explicitly never is
(`:2042`, `pact_clutter_added_to_obstacle_aabbs: False`). The expert avoids exactly one obstacle, via
a hand-coded maneuver — not a general planner.

**And the vertical case is half-built already.** The inbound leg (corridor pick helper) has a top-wall
branch: `if wall == "left": dy = -shift / elif "right": dy = shift / else: dz = -shift`
(`:603-608`). The outbound leg `_bow_segment` (`:2507`) is lateral-only — it derives `obstacle_side`
from `center[1]` and writes `before[1]`/`after[1]`.

## Decisions

| | |
|---|---|
| Overhead bar | **yes**, in the cup-free band |
| Composition | **additional** — side panel unchanged + overhead bar, every episode |
| Scoring | **as `hazard_bar`** — contact breaks clean success, one endpoint |
| Expert | frozen except the bespoke overhead maneuver; **no general planner** |
| Clutter | retained, for **RGB disruption only** |
| Gates | recalibrated to what the geometry allows |

## Resources already in place

Four spare mocap bodies are parked at z = −2 in `pact_place_corridor_v2.xml`: `protr_s` (half
0.0175), `protr_m` (0.025), `protr_l` (0.035×0.035×0.12), and the unused `pact_intrusion_{other side}`
(0.055×0.240×0.090). The panel uses `pact_intrusion_{side}`; `protr_s/m/l` are parked by
`:206` and never placed in the PACT task.

Panel constants for reference (`:1310-1313`): `PANEL_X = 0.615`, `PANEL_Z = 0.89`,
`PANEL_HALF = [0.055, 0.240, 0.090]`, `PANEL_INNER_FACE_Y = 0.100`.

## C0 — site the overhead bar by measurement, not by choice

No rollouts. Replay the 24 v6c trajectories as `run_pact_place_swept_volume_v7.py` does, and for a
grid of candidate overhead bars compute, with **exact `mj_geomDistance`** (plus the `fromto` mesh
fallback v8b had to add — raw `mj_geomDistance` returns a spurious scalar zero for separated
mesh/primitive pairs while still populating distinct endpoints):

- min clearance to **cup, fingers, hand, link7, link6, link5** separately
- `frames_link_clearance_lt_10cm` and `lt_5cm`
- whether a **link** is the closest body (the gate that has failed since v6c)
- `visibility_at_min_link_clearance` via the `_cam_visible_label` raycast (`:~250`)

Search space, bounded by the measurements above:

```
z_bottom  >= 1.05     (cup z_max 1.011, so >= 39 mm cup clearance)
z_top     <= 1.40     (enclosure ceiling 1.42)
x         0.60 - 0.82  (inside the aperture, ahead of the shallowest back wall 0.782)
y         span the corridor; the bar hangs from the ceiling
half      start from protr_l (0.035) and the panel (0.055) cross-sections
```

**Choose the candidate that maximises `frames_link_clearance_lt_10cm` subject to zero predicted
contact with cup, fingers or hand.** Record every candidate with its scores, admitted or not.

## C1 — place the bar and make the expert clear it

**Scene:** none needed. Reuse `protr_l` or the unused `pact_intrusion_{other}`; both are already
parked. Do **not** author a new scene.

**Sampler:** add the overhead bar to `th` alongside the panel, and **append it to
`obstacle_aabbs`** exactly as the panel is at `:1385`. That single line is what lets it sit in the
path at all. The panel's own entries stay untouched.

**Expert, inbound:** already supported — set the overhead bar's wall to `"top"` so
`dz = -shift` (`:608`) fires. Verify it does; do not rewrite it.

**Expert, outbound:** `_bow_segment` (`:2507-2600`) needs a **vertical branch mirroring its lateral
one**: derive `obstacle_side` from `center[2]` rather than `center[1]`, compute the inner face and
`straight_clearance` in z, and write `before[2]`/`after[2]` instead of `[1]`. Bound the duck by the
enclosure ceiling the way the lateral case bounds by `aperture_width/2 - envelope_half_y -
APERTURE_EDGE_RESERVE` (`:2440`, 0.02). Keep `_record_bow` reporting so the fallback is visible.

**Scoring:** the overhead bar is scored as `hazard_bar`. Since it reuses a `pact_intrusion_*` or
`protr_*` body, `classify_contact` already routes it there — **verify by construction**, do not
assume. Place a bar deliberately in the swept volume, run one episode, confirm the row is unclean and
the contact is attributed to `hazard_bar`. Then remove it.

## C2 — honest gates

Replace the v8b set. Every threshold below is justified by a measurement, and the two that were
impossible are gone.

| gate | v8b | v8c | why |
|---|---|---|---|
| `cup_is_closest_body` | ≤ 0.25 | **≤ 0.25, kept** | now achievable — the bar sits where the cup never goes |
| `mean_distinct_links_exposed` | ≥ 3 | **≥ 1.5 of a maximum of 2** | links 1–4 have 0 voxels inside; only link5/link6 exist |
| `frames_link_clearance_lt_10cm` | ≥ 1852 | **set from C0's best candidate, not invented** | 1852 was carried over from a corridor-scale figure |
| `lt_5cm` in ≥ half | ≥ 3/6 | **≥ 1/6, reported not gated** | keep as a diagnostic until C0 shows what is reachable |
| non-zero `visibility_at_min` | ≥ 1/3 | **≥ 1/3, kept** | unchanged; vision must stay useful but imperfect |
| objects 8–12, ≥2 overhead, category ≤2 | pass | **kept unchanged** | these passed and govern the RGB half |

**Do not set a numeric threshold C0 has not shown to be reachable.** That is what sank v8b.

## C3 — realized measurement

Run **6 episodes**, one per family, and recompute every gate on realized `mj_geomDistance` — never on
the Pass-1 AABB approximation, which admitted 877 "link-primary" candidates that the exact instrument
showed were nothing of the kind. Report the full gate table.

**Stop and report if the cup is still closest in more than 1 of 6.** That would mean the cup-free band
does not work either, and it is the third time the same wall has been hit — a finding, not something
to tune.

## C4 — family review, then stop

Only if C3 passes. Per family, run until 1 clean success, cap 4 attempts, render **every** attempt
with family and outcome in the filename, overlay per-link minimum distance and phase. Separate seed
stream from the gate's. Output `role: human_design_review_not_a_gate`, `authorizes_gate: false`.

Then **stop for approval.** The Phase-0 gate is a separate authorization — and this run adds a second
obstacle to every episode, so its clean rate is genuinely less predictable than v6c's 23/24.

## Verification

- C0 records every candidate, admitted and rejected, with per-body clearances and the reason.
- The chosen bar's bottom is ≥ 1.05 (cup `z_max` 1.011) and its top ≤ 1.40; zero predicted contact
  with cup, fingers or hand.
- The overhead bar appears in `obstacle_aabbs`; the panel's entries are unchanged.
- Inbound `dz` branch confirmed firing; outbound vertical bow confirmed by `_record_bow` output.
- Scoring check done **by construction**: a deliberate in-path overhead bar yields an unclean row
  attributed to `hazard_bar`.
- All gates evaluated on realized `mj_geomDistance` with the `fromto` mesh fallback.
- No new scene file; `pact_place_corridor_v2.xml` byte-identical; v6c/v7/v8b artifacts untouched.
- Panel geometry and behaviour unchanged, so prior contact numbers stay comparable.

## Constraints

- **No general obstacle planner.** Extend the existing bespoke maneuver only.
- Do not modify the side panel's placement, geometry, or its `obstacle_aabbs` entry.
- Do not add clutter to `obstacle_aabbs` — clutter stays RGB-only and outside the swept volume.
- Do not set a gate threshold that C0 has not demonstrated reachable.
- Do not reuse the Pass-1 AABB instrument for any admission decision.
- Stop after C3 if the cup is closest in more than 1 of 6; stop after C4 for approval.
- Work in `/root/prox_learning_pact_remediation`; interpreter `/root/act_retrain_venv/bin/python3`;
  `MUJOCO_GL=egl`, `MLSPACES_ASSETS_DIR=/root/prox_learning/assets`, `PYTHONPATH` → repo
  `submodules/molmospaces`. Pin `OPENBLAS_NUM_THREADS=1` — the cgroup `pids.max = 3840` otherwise
  kills numpy imports.
