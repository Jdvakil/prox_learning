# v8c C0: the overhead bar cannot be sited. Three measured walls, not one.

**Status: stopped at C0. C1–C4 were not built.** C0 confirmed the design's core claim and then
found three independent reasons the rest of the plan cannot be executed as written. All three are
structural properties of the corridor's kinematics, not tuning problems.

Role of every artifact below: `authorizes_gate: false`, `authorizes_collection: false`,
replay-only, `physics_stepped: false`.

## What C0 was asked to do, and what it did

C0 replayed the 24 frozen v6c trajectories (11,352 control steps) with `mj_forward` only and scored
**667 candidate overhead bars** — 567 carried by `pact_intrusion_{other side}` and 100 by a rotated
`protr_l` — with exact `mj_geomDistance` against the cup, fingers, hand, link7, link6 and link5
separately. Every candidate is recorded with its per-body clearances and, when rejected, the reason.

```
scripts/run_pact_place_v8c_c0_siting.py              -> c0_siting.json, c0_per_episode_rows.json
scripts/run_pact_place_v8c_c0_visibility_ceiling.py  -> c0_visibility_ceiling.json
scripts/run_pact_place_v8c_c0_duck_feasibility.py    -> c0_duck_feasibility.json
```
all under `diagnostics_output/pact_place_corridor_v8c_c0/`.

## Review clips

Three episodes rendered with the chosen bar composited into the replay, so the findings can be
checked by eye: `diagnostics_output/pact_place_corridor_v8c_c0_review/`
(`scripts/run_pact_place_v8c_c0_review_videos.py`, `role: human_design_review_not_a_gate`).

| clip | panel | steps | bar in wrist FOV | frames with links inside the bar | min cup clearance |
|---|---|---:|---:|---:|---:|
| `row00_panel_left_with_c0_bar.mp4` | left | 561 | **0** | 223 | 0.156 m |
| `row04_panel_right_with_c0_bar.mp4` | right | 418 | **0** | 260 | 0.159 m |
| `row17_panel_left_with_c0_bar.mp4` | left | 624 | **0** | 280 | 0.159 m |

Panes are wrist (the policy's only RGB camera), third-person, and a corridor view with the bar
tinted orange; the tint writes `geom_rgba` only. Every frame carries the exact per-body clearances,
the wrist-FOV flag, and the TCP drop the duck would need against the shelf top.

These are replays with the bar composited in: the recorded expert never saw the bar and never
avoids it, and no physics is stepped. The v6c renderer's faithfulness spot-checks run unchanged
(max object residual 7.0e-5 m, max TCP residual 1.2e-4 m, against a 1e-6 m step-0 indexing check).

A cross-check falls out of this: MuJoCo's own contact detection flags the bar in exactly the same
frames that `mj_geomDistance` reports negative clearance — 223/223, 260/260, 280/280. The two
instruments agree frame for frame.

## Result 1 — the cup-free band works. This part of the design is confirmed.

The plan's opening was real. Across **all 567** hazard-class candidates, in **24/24** episodes:

| measure | result |
|---|---|
| `cup_is_closest_body` | **0 of 24, for every one of the 567 candidates** |
| minimum cup clearance, over all candidates and episodes | **0.134 m** |
| `mean_distinct_v8b_links_lt_10cm` at the chosen bar | **2.0 of a maximum of 2** |

This is the first design since v6c that is link-primary by construction rather than by hope, and it
is not marginal — the cup stays 13 cm away at the worst candidate. The wall that stopped v6c, v7 and
v8b is genuinely cleared.

The chosen candidate under the plan's stated objective (maximise `frames_link_clearance_lt_10cm`
subject to zero predicted contact with cup, fingers or hand):

| | |
|---|---|
| body | `pact_intrusion_{other side}`, half `[0.055, 0.240, 0.090]` |
| centre | `[0.655, +0.100, 1.161]`, bottom **1.071**, top **1.251** |
| `frames_link_clearance_lt_10cm` / `lt_5cm` | 9,272 / 7,644 of 11,352 |
| episodes with `lt_10cm` / `lt_5cm` | 24/24 / 24/24 |
| cup / fingers / hand clearance | 0.156 / 0.069 / **0.010** |
| `visibility_at_min_link_clearance` | **0 of 24** |

## Result 2 — the retained visibility gate is unsatisfiable, and it is unsatisfiable *because* the band is cup-free

The plan kept `non-zero visibility_at_min >= 1/3` unchanged, on the grounds that "vision must stay
useful but imperfect". No candidate can reach it:

| measure | over all 667 candidates |
|---|---|
| `visibility_at_min_link_clearance` | **0 of 24, every candidate** |
| bar centre ever inside the wrist camera's FOV cone | **0 of 24, every candidate** |

Occlusion is not the cause; the camera never points there. Measured directly on a 2 cm lattice over
the enclosure interior, across all 11,352 frames, **ignoring occlusion** (so it is a strict upper
bound on what the camera could see):

```
wrist camera height              z <= 1.111 m
wrist camera pitch               mean -68.6 deg, shallowest -57.7 deg  (it looks down)
visibility ceiling, cone only          z = 1.030 m
visibility ceiling, occlusion-aware    z = 1.030 m
lattice points at or above the band bottom (1.05) ever in the cone     0
```

Set these two numbers side by side:

```
cup swept ceiling      z = 1.011 m     <- what makes the band cup-free
wrist visibility ceiling z = 1.030 m   <- what makes the band camera-blind
band bottom            z = 1.050 m
```

They differ by 19 mm, and they are not independent: **the cup and the wrist camera are bolted to the
same rigid body.** The band is cup-free precisely because the gripper never goes there, and the
camera goes wherever the gripper goes. Any obstacle placed where the cup cannot reach is, by the same
kinematics, an obstacle the wrist camera cannot see.

So `cup_is_closest_body <= 0.25` and `non-zero visibility_at_min >= 1/3` are not two gates that
happen to be hard together. In this corridor they are mutually exclusive. The plan's own constraint —
*"Do not set a gate threshold that C0 has not demonstrated reachable. That is what sank v8b"* —
forbids carrying this gate into C2, and it equally forbids quietly lowering it.

The wrist camera is the only RGB camera in the collection observation
(`scripts/run_pact_place_recovery_datagen.py:386` asserts camera 0 is `wrist_camera`, followed only
by proximity sensors), so there is no third-person view to fall back on. `_cam_visible_label`'s
synthetic exo camera, which the plan named for C0, is a sampler-internal labelling aid and is not in
the policy's observation.

## Result 3 — the expert cannot duck. The colliding body is 43 cm above the TCP.

Both maneuvers C1 would extend act on the **TCP**: the inbound branch shifts `z_travel` by `dz`
(`enclosure_reach.py:608`), and `_bow_segment` displaces TCP waypoints (`:2507`). That works for the
side panel because the panel sits at the wrist's own height. An overhead bar is hit by link5/link6,
which ride far above the TCP. Measured over the frozen trajectories, restricted to frames where
link5/link6 are inside the corridor:

| measure | value |
|---|---|
| median height of link5/link6 top above the TCP | **0.426 m** |
| TCP drop needed to clear a bar at z = 1.05 | median **0.277 m**, max **0.288 m** |
| episodes where that drop puts the TCP below the shelf top (0.72 m) | **24 of 24** |
| episodes with any frame above the band bottom | 24 of 24 |

To lower the elbow out of the band the expert would have to drive the gripper down through the shelf
the cup is standing on. Clearing the bar without moving the TCP means re-posing the elbow in the
nullspace — a general obstacle planner, which the plan's constraints explicitly forbid.

### Where the bar is struck: the phases with no freedom to move

Broken down by policy phase over the three review episodes (1,603 frames):

| phase | struck / frames | leg |
|---|---:|---|
| `pregrasp` | 251 / 359 | inbound |
| `grasp` | **45 / 45** | inbound |
| `gripper-close` | **27 / 27** | inbound |
| `lift` | **57 / 57** | inbound |
| `outbound_approach` | **217 / 217** | outbound |
| `outbound_pass` | 166 / 245 | outbound |
| `outbound_exit`, `preplace`, `placement_descent`, `retreat`, `gripper-open` | 0 | — |

Both legs are obstructed, so C1 would have had to extend the inbound branch *and* write the outbound
vertical bow. But the decisive rows are `grasp`, `gripper-close` and `lift`, struck in **100% of
their frames**. Those are not traversal segments: the TCP is pinned to the cup on the shelf, so
there is no waypoint to bow and no `z_travel` to shift. The arm's shape there is fixed by where the
cup is, and the elbow is in the band because reaching down and into the cup is what puts it there.

For those frames no bespoke maneuver exists at all — not one that is hard to tune, one that has no
free parameter. Only a nullspace elbow re-pose would clear them, which is the general planner the
plan forbids.

This is consistent with the third supporting measurement: **every one of the 392 admissible
candidates penetrates link5/link6** in the frozen replay, by 26 mm at the shallowest and 99 mm at
the median, in 24/24 episodes. There is no candidate anywhere in the band that merely grazes the arm.
The band is cup-free because link5/link6 fill it.

## Result 4 — contact attribution: only one body works, and the plan's assumption was wrong

The plan said the overhead bar could reuse "a `pact_intrusion_*` or `protr_*` body" because
"`classify_contact` already routes it there", and asked for this to be verified rather than assumed.
It does not hold. `pact_contact_audit.py:16` sets `HAZARD_BODY_PREFIX = "pact_intrusion_"`, so
`protr_s/m/l` fall through to `other_environment`. Both break `clean_success`
(`run_pact_place_expert_screen.py:349`), but only `pact_intrusion_*` is *attributed* to `hazard_bar`.
100 of the 667 candidates are rejected on this ground alone. Had the C1 scoring check been run
against `protr_l`, it would have failed in a way that looks like a scoring bug.

## Result 5 — a measurement defect worth carrying forward

`mj_geomDistance` in MuJoCo 3.5 has a second failure mode beyond the one v8b found. v8b's fallback
covers the case where the scalar is 0.0 but `fromto` holds distinct endpoints. There is also a case
where the scalar is 0.0 **and `fromto` is left untouched**, at every `distmax` tried
(10.0 / 1.0 / 0.5 / 0.2), for geoms that are demonstrably far apart — one instance was a cup collider
0.25 m clear of the bar reported as touching it. Left alone this fabricates contacts, which is the
most damaging error possible for this study.

The C0 instrument clears the `fromto` buffer before every call, and when both the scalar and the
segment are empty it uses the two geoms' world-AABB gap **only to disprove the contact** (an AABB gap
is a strict lower bound on true separation, so a positive gap proves the zero is spurious). Such
pairs are counted and skipped; no AABB value is ever returned as a clearance. **52 spurious zeros
were rejected** in this sweep. Before hardening, they had put two candidates' cup clearance at
exactly 0.0 when the true value was ~0.18 m.

`scripts/measure_pact_place_v8b_realized.py:41` (`_true_distance`) still carries the unhardened instrument, and it also
reuses one `fromto` buffer across calls without clearing it.

## What this means for v8c

The design's premise is confirmed and its execution path is closed. An obstacle in the cup-free band
is link-primary — and is, for the same reason, invisible to the only camera the policy has and
unreachable by the only maneuver the expert is allowed.

Three options, none of which C0 can choose:

1. **Drop or re-express the visibility gate, and abandon the overhead bar.** Worth noting that a
   hazard invisible to the camera is what the settled PACT picture already describes; the
   `>= 1/3` gate encodes the opposite intent, presumably so the comparison is not rigged in
   proximity's favour. Whether that intent still applies is a design call.
2. **Allow the expert a nullspace elbow maneuver.** This is the only way to clear a bar in the band,
   and it is a general planner by another name.
3. **Accept that the corridor cannot produce link-primary, camera-visible proximity** and stop
   iterating on clutter and obstacle geometry. This is the third distinct design (v8, v8b, v8c) to
   hit the same kinematic fact from a different direction.

No scene file was modified. `pact_place_corridor_v2.xml` and `pact_place_corridor_v3.xml` are
untouched, as are all v6c/v7/v8b artifacts, the sampler and the expert.
