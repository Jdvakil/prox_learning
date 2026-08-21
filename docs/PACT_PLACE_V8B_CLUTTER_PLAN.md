# v8b: enough clutter, actually off the floor, measured on realized physics

*Revision of `PACT_PLACE_V8_CLUTTER_PLAN.md` after the B5 human checkpoint. The v8 machinery is
sound and mostly stays; the layout **content** and the admission **instrument** are what failed.*

## Context

v8 ran B1–B5 and stopped correctly at the human checkpoint without authorizing the gate. The
infrastructure it built is good and is **kept**. What the review exposed is that the selected layouts
do not implement the design, and the instrument that admitted them could not have detected that.

### What the artifacts actually show

Verified in `diagnostics_output/pact_place_clutter_sweep_v8/analysis.json` and
`.../pact_place_corridor_v8_family_review/family_review.json`:

| Finding | Evidence |
|---|---|
| **Only 2 objects per episode** | `pact_clutter_active_body_names: ["pact_clutter_02/Candle_1", "pact_clutter_08/Candle_3"]`, all 24 layouts have exactly 2 objects (1 `proximity_event` + 1 `workspace_occluder`) |
| **75% of clutter is one object** | across 48 placements: candle ×36, vase ×4, apple ×3, tomato ×2, bowl ×2, potato ×1 |
| **Zero vertical dispersion** | **all 48 objects have bottom z = 0.7200** — the shelf floor — while the enclosure runs to z ≈ 1.42 |
| **Support labels are fiction** | 4 objects tagged `overhead`, 34 `wall_adjacent`, yet every one rests on the floor |
| **The core defect is not fixed** | realized `cup_is_closest_body` **0.833 (5/6)** vs v6c **0.875 (21/24)** — indistinguishable |
| **Admission instrument disagreed with reality** | B2 predicted `cup_is_closest_body` **0/24**; realized measurement **5/6** |
| **Hazard visibility identical to v6c** | `visibility_at_min_link_clearance` fraction **0.0833** for both; values are `[0,1,0,…]` — 22 of 24 are zero |
| **Rolling objects break settling** | F3 lost 3 attempts: `clutter drifted during settle: Vase_Decorative_1 0.369517 m`, `Vase_Open_1 0.462099 m` |

For scale: v6c had **4** boxes at 5 × 10 × 10 cm. v8 shipped **2** candles at roughly 5 × 5 × 8 cm.
v8 is *less* cluttered than the thing it was meant to improve on.

### Why the instrument missed it

B2 admitted candidates by replaying **metadata boxes on B0 (v6c) qpos tracks**; the realized number
uses **actual collision geoms on v5 qpos**. Two independent errors compound:

1. **World-axis-aligned AABBs of long, diagonally-oriented links wildly overstate link extent** — a
   link at 45° has an AABB mostly full of air. Links therefore appear closer than they are, which is
   precisely the bias needed to produce `cup_is_closest_body = 0/24`.
2. **Scoring v8 candidates against v6c trajectories is circular.** Once the scene changes, the arm's
   path changes. The v8 plan baked this in by design, and it is the root of the mismatch.

This is the **third** instrument failure in this programme, after the +70 pp artifact and
`skin_engagement`. Same shape each time: a proxy that improves while the thing it names does not.
The fix here is not a better proxy — it is **measuring on realized physics before committing**.

### `visibility_spans_range` is a misleading flag

The sweep reports `visibility_spans_range: True` because the values contain both a 0 and a 1. But
22 of 24 are zero and the fraction equals v6c's exactly. Decision 2 was that visibility must **not**
be driven to 0. That verification item failed and a boolean hid it. **v8b replaces every boolean
verification item with a distribution and a threshold.**

## What stands — do not rebuild

**B3 (scene + sampler) is kept in full.** Confirmed by inspection:

- `pact_place_corridor_v5.xml` is 248 bytes and declares **no** clutter bodies. The palette is
  installed through the **MjSpec path at runtime**, the same route as the target — so changing the
  palette needs **no new scene**.
- `PactPlaceCorridorV5Sampler._layout()` iterates `layout["objects"]` and is **fully generic over
  object count**. Raising 2 → 8–12 is a config and sweep change, not a code change.
- Existing guards to keep: palette size must be 12–20 (`_palette`), a layout may not activate the
  same palette slot twice, and each object's uid must match its frozen palette slot.
- Settle machinery, drift detection and shell containment all work — they are what *caught* the
  rolling vases.

**B4 (scoring check) is kept, with one addition.** It is genuinely rigorous: a real link5 contact at
**−17 mm penetration** scored `clean_success: false`, and the topple test shows a real quaternion
flip. That closes the `cavity_obj_` exemption trap properly, and the free-body case does not need
re-running. **The mount-slot (mocap) case does** — see "Mount slots" in B2b — because a mocap body
takes a different path through the contact audit than a free body.

**Also keep:** the 6 families, the family/side quotas (verified exact across all 12 combinations),
the separate review/gate seed streams (`review_gate_episode_overlap: []`), and
`pact_clutter_added_to_obstacle_aabbs: false` — the expert stays frozen.

**Frozen, never modify:** v5 (22/24), v6b (20/24), v6c (23/24) screens and clips; place scenes
v1–v4; `PactPlaceCorridorV3Sampler` and `V4Sampler` behaviour; `pact_contact_audit.py`;
`pact_collision_corridor.xml` at `f8c04b07…`.

## B1b — rebuild the palette for stability and variety

Keep the same asset path: `get_valid_pickupable_obja_uids()` (`molmo_spaces.utils.synset_utils`) and
`ObjectMeta.annotation(uid)` (`molmo_spaces.utils.object_metadata`) — 1,629 uids, 655 categories.

**Palette size 18–20** (within the sampler's existing 12–20 bound), with a **hard category cap of 3
slots per category**. v8's palette was nominally diverse; selection then collapsed it to one object,
so diversity must be enforced at *layout* level too (see B2b), not just in the palette.

**Drop rolling shapes.** Vases, apples, tomatoes and potatoes roll on a flat shelf; that cost F3
three attempts. Prefer flat-bottomed, wide-based categories — `bowl`, `plate`, `box`, `book`,
`soapbottle`, `cellphone`, `alarmclock`, `candle` (flat-based), `lantern`, `spraybottle`.

**Add a stability pre-filter, run once per palette uid before freezing:** settle the object alone on
the shelf for `CLUTTER_SETTLE_STEPS`, and reject any uid whose centre drifts **> 5 mm** or whose
orientation changes **> 5°**. Record the drift for every candidate uid, accepted and rejected. This
is cheap and it removes an entire class of B5 attrition.

Freeze the palette by uid in the config. Never re-sample at episode start.

**The naming trap still applies.** Every clutter body must be `pact_clutter_*` and contain none of
`cavity_obj_`, `pact_intrusion_`, `place_receptacle` — `classify_contact` tests `"cavity_obj_"`
**first** and returns the exempt `grasp_target`.

## B2b — layouts with real density, real height, and honest distances

### Object count and composition — the core fix

Each layout activates **8–12 objects**, composed as:

| Role | Count | Purpose |
|---|---|---|
| `proximity_event` | **2–3** | the near-miss geometry that defines the family |
| `workspace_occluder` | **2–3** | degrade the view of the target during the risk phase |
| `scene_filler` | **4–6** | visual density; RGB ambiguity; no proximity or occlusion role |

**Per-layout diversity constraint: no category may appear more than twice.** This is what stops the
collapse to 36 candles.

### Support becomes a geometric constraint, not a label

Every object's support type is **verified against its posed geometry** and the layout is rejected if
the label and the geometry disagree:

```
shelf_standing : bottom z within 2 mm of 0.72
wall_adjacent  : bottom z within 2 mm of 0.72 AND nearest face within 30 mm of a side wall
overhead       : bottom z >= 0.95, and physically supported (ceiling-hung or wall-cantilevered)
```

**Every layout must contain at least 2 objects with bottom z ≥ 0.95.** F5 additionally requires its
`proximity_event` object to be overhead.

### Mount slots — the authorized sampler extension

A free body at z = 0.95 falls, so overhead objects cannot be free bodies. An earlier draft of this
plan said "weld them" while also forbidding sampler changes; that was contradictory, and the
contradiction is resolved here rather than worked around.

**Static (jointless) bodies are not the answer either** — their pose is fixed at compile time and
cannot be re-placed per episode. The correct mechanism is **mocap**, which is exactly what v3/v4
clutter used: kinematically posed, immovable, still fully collidable, and re-placeable every episode
through the existing `_mocap_set` path.

Whether an object is overhead is a *layout* property that varies per episode, but the free-vs-mocap
decision is made once at install time in `add_auxiliary_objects`. Resolve this by **partitioning the
frozen palette**:

| Slot class | Count | Installed as | Posed by | Used for |
|---|---|---|---|---|
| **mount slots** | 5–6 | **mocap body**, no free joint | `_mocap_set` | `overhead` objects only |
| **prop slots** | 12–14 | free body (unchanged) | `_set_free_pose` + 1 cm drop | shelf-standing and wall-adjacent |

The partition is frozen in the config alongside the palette. Layouts draw overhead objects **only**
from mount slots and floor/wall objects **only** from prop slots. Total palette stays within the
sampler's existing 12–20 bound, and the "may not activate a palette slot twice" guard is unchanged.

**This is an authorized, minimal extension to `PactPlaceCorridorV5Sampler`** — a branch on slot
class in two places (joint creation at install, pose application per episode), plus skipping the
settle/drift check for mount slots, which do not settle. Nothing else in B3 changes, and
`PactPlaceCorridorV3Sampler` / `V4Sampler` must stay byte-identical in behaviour.

Two requirements that follow:

- Mount bodies keep the `pact_clutter_*` namespace, so link-vs-mount contact still scores as clutter
  contact. **Re-run the B4 scoring check for a mount body specifically** — B4 validated a free body,
  and a mocap body travels a different code path in the contact audit. This is the one part of B4
  that must be repeated.
- A mocap object is implicitly bolted in place, so it must *look* bolted. Require every mount object
  to sit **within 20 mm of a side wall or the ceiling**, or it will read as floating in the clips.

The resulting asymmetry — floor props topple, overhead fixtures do not — is deliberate and physically
sensible: loose objects on a shelf are knockable, a wall or ceiling fixture is not. Record it in the
report so it is not mistaken for an inconsistency.

### Distances measured properly

Replace world-AABB proxies with **true geom-to-geom distance**: `mujoco.mj_geomDistance` where
available, otherwise oriented-box / mesh distance at the object's actual pose. Compute against the
**object's real collision geoms**, not its metadata bounding box, and against the **robot's collision
geoms per link**, not a link AABB.

Record the instrument name in the analysis so it can never again be ambiguous which one produced a
number.

### The iterate-once loop — the fix for circularity

Scoring v8b candidates on v6c tracks is what produced 0/24 vs 5/6. Break it:

1. **Pass 1 — propose.** Generate ≥ 400 candidate layouts. Admit on the hard rejections below,
   scored against the B0 tracks as a *first approximation only*. Select 24 by quota-constrained
   farthest-point (keep v8's `selection_rule`, which worked — `min_pairwise_selected_layout_distance`
   was 0.49).
2. **Pass 2 — measure on realized physics.** Run **6 real episodes**, one per family, and recompute
   every metric on **realized v5 qpos with the real collision geoms**.
3. **Pass 3 — re-select.** Re-score all candidates against the realized tracks and re-select the 24.
   Report Pass 1 vs Pass 3 side by side, including how many of the original 24 survived. **If Pass 1
   and Pass 3 disagree by more than 20% on `cup_is_closest_body`, report the discrepancy and its
   cause** — a repeat of the v8 mismatch is a finding about the instrument, not a nuisance.

### Hard rejections (unchanged from v8 plus two)

```
intersects swept_volume_by_link + C          (C = 0.030 m)
would contact any robot link, the hand, or the carried cup
initial contact at reset, or overlaps another object
outside the enclosure, or outside the SHALLOWEST sampled back wall (0.782) —
    assert against each episode's own depth, never the median
blocks the target or the tray, or makes the expert trajectory infeasible
target visibility floor violated
NEW: support label disagrees with posed geometry
NEW: layout violates the category cap or the >=2 overhead requirement
```

Never let the sweep manufacture a collision to make proximity look informative.

**Target visibility floor** (unchanged): the target must be clearly visible for ≥ N consecutive
frames during the approach before any occlusion. Set N in the config. An episode where the target is
never visible is *unsolvable*, and is rejected rather than counted as a hard case.

## Admission gates — numeric, on realized measurements

The 24 layouts may only be frozen when **all** of these hold on the Pass 2/3 realized numbers.
Every one replaces a v8 boolean that passed while the design failed.

| Gate | Threshold | v6c | v8 realized |
|---|---|---|---|
| `cup_is_closest_body` fraction | **≤ 0.25** | 0.875 | 0.833 |
| episodes with non-zero `visibility_at_min_link_clearance` | **≥ 1/3** | 0.083 | 0.083 |
| `mean_distinct_links_exposed` | **≥ 3.0** | 0.75 | 1.5 |
| `frames_link_clearance_lt_10cm` | **≥ 2× v6c** (≥ 1852) | 926 | 852 |
| `frames_link_clearance_lt_5cm` | **> 0 in ≥ half of episodes** | 0 | 92 total |
| objects per layout | **8–12** | 4 (boxes) | 2 |
| objects with bottom z ≥ 0.95 per layout | **≥ 2** | 0 | 0 |
| max objects of one category per layout | **≤ 2** | — | 2 of 2 identical |

If a gate cannot be met, **report which one and why, and stop.** Do not relax a threshold to pass.
The most likely genuine conflict is `cup_is_closest_body` against the frozen expert: the carried cup
is the most distal body, so it may be *structurally* hard to keep it off the closest-body spot. If
that is what the data shows, say so plainly — it is a real finding about the task, and it would mean
the endpoint needs rethinking rather than the clutter.

## B5b — re-run the family review

Unchanged in form. Per family, run until **1 clean success**, cap **4 attempts**, keep and render
**every** attempt with family and outcome in the filename, overlay per-link minimum clutter distance
and phase. Separate `family_review_seeds` stream, no overlap with the gate.

Two additions:

- **Sampling failures must render a real clip or be reported as having no clip.** v8's three F3
  failure clips are 58 KB against ~5 MB for real episodes — no rollout occurred, so they are title
  cards. Either render the settle attempt itself, or state plainly that N attempts produced no
  footage. Do not describe them as videos of the attempt.
- **Report the realized gate table** (above) alongside the clips, so the checkpoint is a decision on
  numbers rather than an impression from six videos.

Output `diagnostics_output/pact_place_corridor_v8b_family_review/`, with
`role: human_design_review_not_a_gate`, `authorizes_gate: false`, `authorizes_collection: false`.
Then **stop and report.**

## B6 / B7 / B8 — unchanged from the v8 plan

Gate (24 episodes, `MIN_CLEAN_SUCCESSES = 20`, zero clutter contact of any kind, prediction frozen
before the first episode), then the 24-episode pilot collection through the **datagen path** with
`proximity_sensor_period_ms` at default and a gate asserting on **produced files**, then the endpoint
(same-environment ACT vs PACT, 3 seeds, primary metric = arm-link contact rate).

**Revise the gate prediction.** v8's band assumed 2 objects; 8–12 movable objects with overhead
mounts is materially harder. **Predict 17–22**, and state that the lower bound reflects density, not
a change in the expert. `MIN_CLEAN_SUCCESSES` stays 20 — if density costs the gate, that is a real
trade to surface, not to tune around.

See `PACT_PLACE_V8_CLUTTER_PLAN.md` for the full text of these phases, including the collection
traps (`save_utils.py:374` bare `json.dumps` on `obs_scene`; `cam_visible` as `numpy.bool_` under
NumPy 2; cgroup `pids.max = 3840` killing 12 workers).

## Verification

- Palette stability pre-filter ran; drift recorded for every uid, accepted and rejected; no accepted
  uid drifts > 5 mm or rotates > 5°.
- Every selected layout has 8–12 objects, ≥ 2 with bottom z ≥ 0.95, and no category more than twice.
- Support labels verified against posed geometry for all objects; report the check, not the label.
- Distance instrument is true geom-to-geom, named in the analysis, and **identical** for the v6c
  baseline and v8b.
- Pass 1 vs Pass 3 reported side by side, with the count of layouts that survived re-selection.
- **All eight admission gates pass on realized numbers**, reported as a table with v6c and v8 columns
  for comparison.
- No boolean stands in for a distribution anywhere in the report.
- B4's free-body scoring check still valid; **B4 re-run for a mount (mocap) body** and it also scores
  unclean. B3 scene unmodified; V5 sampler changed **only** by the authorized mount-slot branch, with
  V3/V4 behaviour byte-identical.
- Family/side quotas exact: 6 families × 4, each 2 left / 2 right.
- `review_gate_episode_overlap` empty; `pact_clutter_added_to_obstacle_aabbs: false`.
- Every B5b attempt has either a real clip or an explicit statement that it produced no footage.
- v5, v6b, v6c artifacts and verdicts untouched; place scenes v1–v4 byte-identical.
- Work stopped after B5b for approval.

## Constraints

- **Do not modify the expert.** No changes to routing, `obstacle_aabbs`, `_seg_margin`, `ENV_LO`/
  `ENV_HI`, or the speed law.
- Do not change `gripper_empty_threshold`, `tcp_pos_err_threshold`, `tcp_rot_err_threshold`,
  `max_sequential_ik_failures`, or `MIN_CLEAN_SUCCESSES`.
- Do not widen the corridor, aperture, panel clearance, tray, or target; do not filter or re-rank
  grasp candidates.
- Do not minimise wrist visibility as an objective. Measure it; require it to span a range.
- Do not relax an admission gate to make a layout set pass.
- Do not rebuild B3. The **only** authorized V5 sampler change is the mount-slot branch described in
  B2b (joint creation at install, pose application per episode, settle skipped for mounts). Any other
  sampler or scene change needs approval first — stopping to ask, as happened over the overhead
  contradiction, is the correct behaviour.
- Re-run B4 for the mount case only; the free-body result stands. Do not edit the v5/v6b/v6c renderers.
- **Stop after B5b, and again after B7.**
- Work in `/root/prox_learning_pact_remediation`. Interpreter `/root/act_retrain_venv/bin/python3`,
  with `MUJOCO_GL=egl`, `MLSPACES_ASSETS_DIR=/root/prox_learning/assets`, and `PYTHONPATH` set to the
  repo's `submodules/molmospaces`. Check `pgrep -fc "python.*(eval_|run_pact_place)"`; plain
  `pgrep -fc eval_` self-matches the checking shell.
