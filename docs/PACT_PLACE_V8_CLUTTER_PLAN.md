# v8: clutter that degrades RGB and exposes the arm links, as a 24-episode pilot

*You are picking this up cold. Everything you need is here; the "What already exists" table tells you
what not to rebuild.*

## Context

The corridor pick-and-place task works. What does not work is the **clutter**, which exists to make
plain ACT's vision insufficient while giving Proximity+ACT (PACT) a measurable collision-rate
advantage. Two attempts have failed, each in a way worth knowing about before you start.

**v6c** passed Phase 0 at 23/24 with four 5 × 10 × 10 cm welded boxes at |y| = 0.34, x ∈ {0.70, 0.75}.
Its geometry sweep evaluated **exactly two candidate sets**, differing only in height (0.08 vs 0.10 m)
— verified in `diagnostics_output/pact_place_clutter_sweep_v6c/analysis.json`. It never optimised for
link proximity, layout diversity, or sensor response. Worse, the boxes' only recorded interaction was
**carried-cup vs box**: the hazard sat entirely distal to the sensor suite.

**v7** (the previous plan, already executed — see `pact_place_corridor_v7_design_review/`) tried to fix
this by maximising a `skin_engagement` metric and minimising wrist-camera visibility, subject to
staying ≥30 mm outside the expert's swept volume. The optimiser did exactly that and produced **13
postage stamps glued to the walls and ceiling** at z = 1.0–1.385, every one scoring
`wrist_fov_visibility: 0.0`. The dispersion check passed. `skin_engagement` rose 6.6× (0.0185 →
0.122). Both numbers were satisfied by a result visually worse than v6c.

**The two lessons that shape this plan:**

1. **`skin_engagement` was a broken instrument.** It counted any sensor-step where clutter fell within
   *max sensor range*, so a tile 20 cm above link5 scored the same as a genuine near-miss. It is
   deleted here, not tuned. Its replacement measures **near-miss clearance bands**.
2. **"Stay ≥C outside the swept volume" + "maximise engagement" drives obstacles to the enclosure
   periphery**, because the periphery is the only place satisfying both. Selection must optimise for
   **coverage** (farthest-point), never top-N by a single score.

### The ten decisions this plan implements

| # | Decision |
|---|---|
| 1 | RGB is degraded two ways: **occlude the hazard itself** *and* **occlude the workspace/target** |
| 2 | **Vision stays useful but imperfect.** Visibility spans a measured range; never driven to 0 |
| 3 | The endpoint counts **arm-link (link1–link6) collisions only** — the bodies proximity covers |
| 4 | **The expert stays frozen.** PACT's edge comes from sensing, not imitated avoidance |
| 5 | **6 stratified layout families × 4 episodes**, 2-left/2-right each; farthest-point selection |
| 6 | **24-episode pilot now**; full 100+ only after the user validates it |
| 7 | Clutter is **real objects** from the asset library (mugs, apples, bowls), varied in shape and size |
| 8 | Target must be **visible early, occluded at the critical moment** — a solvability floor |
| 9 | Objects are **movable free bodies**; knocking one over is a failure |
| 10 | Endpoint: **same-environment ACT vs PACT, 3 seeds**, primary metric = arm-link contact rate |

**Decision 9 carries real risk and was taken deliberately.** Free bodies can topple, roll, and cascade;
this is the first screen with movable clutter, so the gate is genuinely less predictable than v6c's.
The plan mitigates it (settle gate, stability assertion, full-qpos replay) rather than avoiding it.
If it destabilises the screen, report that — do not quietly weld the objects.

## What already exists — do not rebuild

| Thing | State |
|---|---|
| `pact_place_corridor_v1/v2/v3.xml` | frozen; v3 backs v6b/v6c. **Never edit** |
| `pact_place_corridor_v4.xml`, `PactPlaceCorridorV4Sampler` | v7's welded 16-box pool. Superseded; leave in place, do not extend |
| v5 (22/24), v6b (20/24), v6c (23/24) screens + clips | frozen baselines. **Never modify** |
| `assets/datagen/pact_place_corridor_v2/recovered_152/` | the only place dataset with real proximity |
| `scripts/run_pact_place_v6c_replay_videos.py` | the renderer to copy. Do not edit it, or v5's or v6b's |
| `scripts/pact_place_corridor_contract.py` | hash/provenance contract; register new versions here |

**Naming for this work** (the scene and screen version numbers differ by convention — v3 scene backed
the v6b/v6c screens):

- scene `pact_place_corridor_v5.xml`, environment version `pact_place_corridor_v5`
- sampler `PactPlaceCorridorV5Sampler(PactPlaceCorridorV3Sampler)`
- config `configs/pact_place_corridor_v8.json`, screen `diagnostics_output/pact_place_corridor_v8/`

## Geometry you will need

```
TUBE_X0      = 0.58    aperture plane            SHELF_TOP_Z = 0.72   shelf floor
back wall    = TUBE_X0 + depth + 0.02, depth sampled per episode 0.182–0.256  → x 0.782–0.856
side walls   = |y| = ap_w/2 + 0.02 (inner face ap_w/2 ≈ 0.425 at ap_w = 0.85); ap_w is sampled
enclosure top= SHELF_TOP_Z + ap_h + 0.02 ≈ 1.44
target rest  = x 0.76, y ∈ [−0.04, 0.04]  (`_obj_rest`, decorrelated from intrusion_side)
carried cup  : origin 54 mm below TCP; origin-to-bottom 73.7 mm; reaches |y| = 0.227, edge 0.262
```

**Critical mechanism.** The expert does **not** avoid clutter. `_seg_margin`
(`enclosure_reach.py:396-408`) is a *speed law* — it slows the TCP near geometry in
`obstacle_aabbs` and never re-plans, and clutter is not in that list. v6c stayed clean purely
because the boxes sat outside the swept envelope. **Do not add clutter to `obstacle_aabbs`** and do
not modify the expert. Consequence to state in your report: the demonstrations contain no avoidance
behaviour; the expert threads obstacles because geometry permits it.

## The six layout families

Each family is a *type of proximity event*, not a random seed. 4 episodes each, **2 left-intrusion
and 2 right-intrusion**, so family membership stays independent of `intrusion_side` and cannot leak
the route.

| | Family | Intent |
|---|---|---|
| F1 | `near_forearm_left` | close approach to link4/link5 on the left of the traversal |
| F2 | `near_forearm_right` | mirror of F1 |
| F3 | `front_stagger` | near-event early, close to the aperture, staggered in x |
| F4 | `rear_stagger` | near-event late, during outbound traversal |
| F5 | `overhead_elbow` | nearest geometry is a proximal link (link3/link4) passing under overhead clutter |
| F6 | `target_occluding` | clutter positioned to hide the target during the risk phase (decision 8) |

## Metrics — the instrument, defined once

Compute these identically for the **v6c baseline** and every new candidate, so improvement is
measured, not asserted. `skin_engagement` is **deleted**.

**Proximity exposure** (the primary design objective):

```
min_clearance_by_link[link1..link6]      minimum over the episode
frames_link_clearance_lt_5cm / _10cm / _15cm
n_distinct_links_exposed                 links with any frame under 10 cm
phase_of_min_clearance                   which task phase the closest approach falls in
cup_is_closest_body                      True if the cup beats every link — should be rare
```

Bands `NEAR=0.05 / MEDIUM=0.10 / FAR=0.15` are **diagnostic thresholds, not constants**. Set them in
the config and revisit after seeing the v6c distribution.

**Visibility** (decisions 1, 2, 8) — measured, and deliberately *not* minimised:

```
clutter_visible_frames, first_visible_frame
visibility_at_min_link_clearance         the key number: was the hazard visible when it mattered?
target_visible_frames, target_visibility_by_phase
```

Reuse the raycast in `_cam_visible_label` (`enclosure_reach.py:~250`). Use segmentation masks for
pixel fractions if available; if not, say so rather than substituting a proxy.

**Distances must come from geom-level queries**, not TCP-to-AABB. The v6/v6b debugging already proved
the TCP envelope misses the carried-cup geometry.

## B0 — baseline the existing runs

No new rollouts. Replay the 24 recorded **v6c** trajectories (`trajectory.json`, full model qpos) with
`mj_forward`, copying the model setup and scene-name guard from
`scripts/run_pact_place_v6c_replay_videos.py`. Also replay v7's 3 review episodes for reference.

Write `scripts/run_pact_place_v8_baseline.py` →
`diagnostics_output/pact_place_v8_baseline/analysis.json`, containing:

- every metric above, for v6c and v7
- **`swept_volume_by_link`** — union of each link's world AABB over all steps and episodes
- **`link_occupancy_voxels`** — 2 cm voxelisation per link, so you know where link4/5/6 *actually*
  travel rather than where they are assumed to

Expect this to confirm that v6c's `cup_is_closest_body` is true in most episodes. That is the defect.

## B1 — build the object palette

Clutter is **real objects**, drawn through the same path that spawns the target:
`get_valid_pickupable_obja_uids()` (`molmo_spaces.utils.synset_utils`) plus
`ObjectMeta.annotation(uid)` (`molmo_spaces.utils.object_metadata`). The library holds **1,629
pickupable uids across 655 categories**, including cup (31), mug (8), apple (29), bowl (98), plate
(31), potato (30), tomato (16), vase (20), candle (18).

Select a palette of **12–20 uids** spanning three size classes (max-dim roughly ≤ 0.10 / 0.10–0.18 /
0.18–0.28 m). Exclude anything larger than ~0.30 m, and exclude `egg` (already excluded for targets).
Freeze the palette by uid in the config — never re-sample it at episode start.

**Do not use near-identical copies of the target cup as decoys.** Decision 3 makes arm-link contact
the endpoint; wrong-object grasps would inject task failures that have nothing to do with proximity.
Visual variety is the goal, not target ambiguity.

### The naming trap — read this twice

`classify_contact` tests `"cavity_obj_" in blob` **first** and returns `grasp_target`, which is
**exempt from scoring**. Objects spawned into that namespace are silently unscored. A blanket
exemption of exactly this kind hid a mis-placed tray for four consecutive screens.

**Every clutter body must be named `pact_clutter_*`** and must contain none of `cavity_obj_`,
`pact_intrusion_`, `place_receptacle`. The sampler already raises on those substrings
(`enclosure_reach.py:~1712`) and the contract enforces the prefix
(`pact_place_corridor_contract.py:474`). Keep both.

## B2 — trajectory-aware sweep with coverage selection

Write `scripts/run_pact_place_clutter_sweep_v8.py`. No rollouts; replay against the B0 tracks.

**Generate a large pool — at least 400 candidate layouts**, not 2 and not 150. Vary position, uid,
size class, and support (shelf-standing, wall-adjacent, overhead).

**Hard rejections.** Reject a candidate if any holds:

```
intersects swept_volume_by_link + C          (C = 0.030 m to start)
would contact any robot link, the hand, or the carried cup
initial contact at reset, or overlaps another object
outside the enclosure, or outside the SHALLOWEST sampled back wall (0.782) —
    assert against each episode's own depth, never the median
blocks the target or the tray, or makes the expert trajectory infeasible
target visibility floor violated (see below)
```

Never let the sweep manufacture a collision to make proximity look informative.

**Target visibility floor (decision 8).** The target must be clearly visible for at least **N
consecutive frames during the approach phase** before any occlusion. Set N in the config. Occlusion is
permitted — encouraged, for F6 — only *after* that window. An episode where the target is never
visible is rejected as unsolvable, not scored as a hard case.

**Selection: farthest-point, not top-N.** Seed with the best admissible candidate, then repeatedly add
the candidate maximising its minimum distance to those already chosen, in the feature space:

```
[ clutter x, |y|, z, size class, support type,
  closest robot link, phase_of_min_clearance,
  min_clearance_by_link, frames_in_near_band,
  visibility_at_min_link_clearance ]
```

subject to the family quotas (4 per family, 2 left / 2 right). This is what stops 24 individually good
layouts from collapsing into 24 variations of one.

**What a good candidate looks like:** an arm link comes within **5–10 cm** of an object for a
meaningful interval, the cup and every link stay collision-free, and `visibility_at_min_link_clearance`
varies across families rather than sitting at 0.

Record every candidate, admitted and rejected, with the reason.

## B3 — scene and sampler, with movable objects

**Scene** `pact_place_corridor_v5.xml`, copied from v3. Clutter objects are **free bodies with mass**,
not mocap. Park unused slots far outside the workspace, the idiom already used for `PROTR` and
`pact_intrusion_*` (`enclosure_reach.py:206`, `1379-1383`).

**Sampler** `PactPlaceCorridorV5Sampler(PactPlaceCorridorV3Sampler)`. V3 hardcodes welded mocap boxes
with scalar half-extents and z pinned to `SHELF_TOP_Z + half_z`. Generalise so each slot carries its
own uid, pose and support type — and **leave V3 and V4 behaviour byte-identical** so v6b, v6c and v7
stay reproducible.

Three things that will bite you:

1. **Model qpos grows.** Free bodies add 7 DOF each, so the full model qpos is no longer 916. Record
   the **complete** qpos in `trajectory.json` and make the renderer read the width from the model.
   Anything hardcoding 916 will silently mis-index. Replay determinism is preserved as long as the
   object free joints are in the recorded qpos.
2. **Objects must be settled before step 0.** Settle them under gravity, then assert max body velocity
   is below a threshold at step 0, and record the settled pose. An object still drifting at step 0
   makes the episode irreproducible.
3. **Cascades.** Object-object contact is not robot contact. Log it separately; do not count it toward
   the arm-link metric.

## B4 — scoring check, before anything else runs

Verify **by construction**, not by reading code:

- place one object deliberately inside the swept volume, run a single episode, and confirm the row
  comes back **unclean** with the contact attributed to clutter
- do this for a **link** contact, not only a cup contact. `robot_environment_contact_pairs` admits a
  pair only when exactly one side is rooted `robot_0/`; link-vs-clutter is the case v6c never
  exercised
- confirm a **toppled** object registers as contact rather than being lost as a physics event
- confirm the classification is `other_environment`, which breaks clean success

Delete the deliberate object afterwards and record the check in the report. **Do not skip this.**

## B5 — family review: one success per family, every attempt on video

**This is the human checkpoint. Nothing after it runs unattended.**

Write `scripts/run_pact_place_v8_family_review.py`. For **each of the 6 families**, run episodes until
**1 clean success**, capped at **4 attempts per family** (24 attempts maximum overall). Keep every
episode, passed or failed.

**Render every attempt**, successes and failures alike, named so the family and outcome are readable
without opening the file:

```
F1_near_forearm_left__attempt01_FAIL_outbound_approach_ik_cascade.mp4
F1_near_forearm_left__attempt02_clean_success.mp4
F5_overhead_elbow__attempt01_clean_success.mp4
```

Overlay per-link minimum clutter distance and the current phase, so the corridor can be seen being
threaded rather than taken on trust. v7's clips failed precisely here: stopping at 3 successes
produced three success clips and no failures to compare them against.

Seeds come from a **separate `family_review_seeds` stream**, non-overlapping with the gate's. The gate
later runs on a fresh master seed reusing none of these episodes — otherwise the review becomes a
selection step and the gate is no longer honest.

Output `diagnostics_output/pact_place_corridor_v8_family_review/` with
`role: human_design_review_not_a_gate`, `authorizes_gate: false`, `authorizes_collection: false`.

**Then stop and report:** the clips, attempts needed per family, failure branches, and the metric table
versus v6c. If a family cannot produce a success in 4 attempts, report it — that is information about
the family, not a reason to run more.

Its clean rate is **not** an estimate: the stopping rule guarantees the last attempt in each family is
a success. Never quote it as one.

## B6 — Phase 0 gate (only after approval)

New master seed. `N_EXPERT_ROWS = 24` (6 families × 4), `MIN_CLEAN_SUCCESSES = 20`, side balance
12/12, jitter bounds unchanged. Keep `RELEASE_CLEARANCE_M = 0.005`, the disarmed empty-gripper check
on `placement_descent`, N = 3 persistence, and the initial-observation reject. Record `trajectory.json`.

Require **zero clutter contact of any kind** — link, hand, or cup. The endpoint counts link contacts
only, but an expert that knocks something with the carried cup has still disturbed the scene.

**Freeze the prediction in the config before the first episode** as `attempt8_prediction`. Prior
screens: 18, 18, 18, 15, 22, 20, 23. **Predict 19–23.** The band is wider than v6c's on purpose: this
is the first screen with movable objects, tighter placement, and an occlusion constraint.

- ≥ 20 → pass, proceed to the pilot collection
- 19 → honest and marginal; report, do not tune
- ≤ 18 → return to **B2 with a larger C**, not to parameter tuning

Note in the config that v6c scored 23 against a predicted 19–22 — my band was set too low then, so a
top-of-band result here is not evidence of anything.

## B7 — pilot collection, 24 episodes

**Must use the datagen path** — `scripts/run_pact_collision_collection.py`
(`ParallelRolloutRunner.run_single_rollout`, `save_mp4s=True`) — and must leave
`proximity_sensor_period_ms` at its default. The expert *screen* harness reduces
`expert_rollout_sensor_polling` to `["qpos","tcp_pose"]` and sets `proximity_sensor_period_ms = 0.0`,
collapsing proximity from `(T,4,8,8)` to `(T,1,8,8)`. That combination produced 152 episodes with no
actions, no wrist RGB and no proximity, and it went unnoticed for 174 rows.

Follow it with a gate that **asserts on the produced files**, never on the config:

```
trajectory.h5 opens; traj_0/actions non-empty
exactly 40 proximity sensors, each (T,4,8,8)
wrist mp4 decodes, frame count == T
24/24 manifest rows reconciled; 12 left / 12 right; 6 families × 4
0 clutter, hazard, and other-environment contacts
realized layouts match the frozen manifest
```

Two known traps during collection: `save_utils.py:374` does a bare `json.dumps` on `obs_scene` and the
place sampler's `scene_params["cam_visible"]` is a `numpy.bool_`, which NumPy 2 names `bool` — this
kills an episode *after* a complete rollout. And cgroup `pids.max = 3840` with ~319 tasks per worker
means 12 workers die with "Thread creation failed"; pin the thread-pool env vars.

Then **stop for user validation.** The full 100+ collection is authorized separately.

## B8 — endpoint (for reference; not authorized yet)

After the full collection: train ACT and PACT on the new distribution, **3 seeds each**, evaluate both
in that same distribution on held-out episodes.

- **primary:** arm-link (link1–link6) contact rate, ACT vs PACT
- **secondary:** task success — expect parity. The established result is that proximity cuts contact
  without costing success
- **report alongside:** contact rate split by `visibility_at_min_link_clearance`, which is what
  answers "did you just blind the baseline?" with data instead of assertion

`PACT_PERMUTED` is the valid ablation. `PACT_ZERO` is invalid — it is out of distribution.

## Verification

- B0 reports every metric for v6c and v7 through the same code path as v8's.
- v8 substantially improves `frames_link_clearance_lt_10cm` and `n_distinct_links_exposed` over v6c,
  and **`cup_is_closest_body` is false in the large majority of episodes**. This is the claim the
  whole change rests on; report it as numbers.
- `visibility_at_min_link_clearance` **spans a range** across families and is not concentrated at 0.
- Target visibility floor met in every episode; no episode is unsolvable.
- Farthest-point selection actually ran: report min pairwise layout distance and show it materially
  exceeds v6c's.
- Family quotas exact: 6 families × 4, each 2 left / 2 right.
- Scoring check confirmed a **link**-vs-clutter contact and a **toppled** object both score unclean.
- All objects settled and stable at step 0; full qpos recorded; replay reproduces step 0 to 1e-6 m.
- B5 rendered **every** attempt, failures included, and work stopped there for approval.
- B5 seeds do not overlap B6's, and no B5 episode appears in the gate.
- Gate ≥ 20/24, zero clutter/hazard/other-environment contact, 12/12 sides. **Re-derive the count from
  the raw `result.json` rows, not from `expert_screen.json`.**
- Clutter body names contain none of `cavity_obj_`, `pact_intrusion_`, `place_receptacle`.
- `pact_collision_corridor.xml` still `f8c04b07…`; place scenes v1/v2/v3/v4 byte-identical; shared
  `pact_contact_audit.py` unmodified; V3 and V4 sampler behaviour unchanged.
- v5 (22/24), v6b (20/24), v6c (23/24) and their clip directories untouched.

## Constraints

- **Do not modify the expert.** No changes to routing, `obstacle_aabbs`, `_seg_margin`, `ENV_LO`/
  `ENV_HI`, or the speed law. If clutter and a clean gate prove incompatible, stop and report — that
  is a finding, not something to tune away.
- Do not change `gripper_empty_threshold`, `tcp_pos_err_threshold`, `tcp_rot_err_threshold`,
  `max_sequential_ik_failures`, or `MIN_CLEAN_SUCCESSES`.
- Do not widen the corridor, aperture, panel clearance, tray, or target, and do not filter or re-rank
  grasp candidates.
- Do not minimise wrist visibility as an objective. Measure it; let it vary.
- Do not weld the objects if the movable physics proves awkward — report it and let the user decide.
- **Stop after B5, and again after B7.** Neither the gate nor the full 100+ collection runs without
  explicit approval.
- Work in `/root/prox_learning_pact_remediation`. Interpreter `/root/act_retrain_venv/bin/python3`
  (`/root/old/.venv` no longer exists), with `MUJOCO_GL=egl`,
  `MLSPACES_ASSETS_DIR=/root/prox_learning/assets`, and `PYTHONPATH` set to the repo's
  `submodules/molmospaces`. Check `pgrep -fc "python.*(eval_|run_pact_place)"`; plain `pgrep -fc eval_`
  self-matches the checking shell.
