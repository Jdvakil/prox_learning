# Fix the raw PACT admission blocker: hazards the skin can actually resolve

**Scope: replaces V0c siting in
[`docs/PACT_PLACE_V9_ENVIRONMENT_PLAN.md`](PACT_PLACE_V9_ENVIRONMENT_PLAN.md).**
V1 (expert wiring), V1b (3+3 human review) and V2 (Phase-0 gate) in that document are unchanged and
still follow. Collection onward is unchanged in
[`docs/PACT_PLACE_V9_TRAIN_EVAL_PLAN.md`](PACT_PLACE_V9_TRAIN_EVAL_PLAN.md).

## Context

V9.5 stopped correctly at its raw-first gate: `passing_variant_count: 1` of 8, inbound vessel
detected left 1/4 and right 0/4. The status doc concludes the next intervention is "sensing or
trajectory design." The measurement says something narrower and more actionable.

### The skin cannot resolve objects that small

Each sensor is 8x8 depth over a 45 deg cone (`camera_configs.py:428-435`), so the pixel pitch at
range R is `2*tan(22.5deg)/8 * R = 0.1036*R`. An object registers only when its width across the
view axis exceeds that pitch:

| object | width | 1 px at | 2 px at | 4 px at |
|---|---:|---:|---:|---:|
| intrusion panel | 0.480 | 4.64 m | 2.32 m | 1.16 m |
| soapbottle | 0.089 | 0.86 m | 0.43 m | 0.22 m |
| candle | 0.016 | 0.15 m | 0.08 m | 0.04 m |

The measured causal effects match that model. From variant 6, the only one that passed
(`pact_place_v95_v0c5_raw_prerequisite/validation.json`):

```
panel             11 sensors   up to 23,004 changed values   max delta 2.00 m
outbound bottle    2 sensors        448 + 60 values          max delta 0.77 m
inbound bottle     1 sensor              40 values           max delta 0.26 m
```

The tensor is `[474, 40, 4, 8, 8]` = 4.85M values. The inbound vessel changes **40 of them, 8 parts
per million**. That is below the sensor's resolving power, not a weak signal awaiting tuning.
Lowering and widening it, as V9.5 did, could not have worked.

### Three things this corrects

1. **Sensing happens at range, from links that never enter the enclosure.** The sensors that see
   the panel are `link2_sensor_3/4/5`, `link3_sensor_1/2/4`, `link5_back_sensor_0-4`. The v7 fact
   that links 1-4 have zero voxels inside the enclosure bounds **collision**, not sensing. Siting
   has been reasoning about the wrong constraint.

2. **The gripper, hand and link7 carry no sensors**, and the inbound hazard sits close to the
   gripper and far from link5/link6. `_protrusion_detected`'s docstring already says it: *"the rest
   are genuinely invisible to a skin without hand coverage."* The right-side asymmetry is a
   symptom; head-on placement near an unsensed body is the cause.

3. **The admission bar is "any nonzero pixel."**
   `run_pact_place_v9_v0c3_causal_proximity.py:414-417` sets
   `passed = panel.changed_values > 0 and inbound.changed_values > 0 and outbound.changed_values > 0`.
   The per-value threshold (`max(ABS_DELTA_FLOOR_M, noise_floor*10)`) is sound; the aggregate rule
   is not. A placement can "pass" with a signal no policy could use. Strengthening it is part of
   the fix.

### This narrows the claim, and the report must say so

The skin resolves a contiguous silhouette of roughly 0.25 m at working range and nothing smaller.
So a cluster of bottles is, to this sensor, **a panel with a mug texture**: it reads as household
clutter to the wrist camera while being sensed as a slab. That is a legitimate design, but it
changes what v9 can conclude.

- **Not:** "PACT avoids clutter."
- **But:** "PACT avoids large obstacles the camera cannot see, some of which are built from
  household items."

The intrusion panel already demonstrates that, at -17.9 pp contact|failure in the reach corridor.
What v9 adds is not a new mechanism but a harder test of the same one: the place task has never
reproduced the panel advantage (12 vs 13 hazard contacts at N=40), and v9 raises the count to three
sensed hazards per episode, two of them movable, scored under a stricter endpoint.

Carry this into V7's report. **Do not describe the clustered hazards as "clutter the skin senses"
without also stating the resolving-power floor that forced them to be that size.** Mugs and apples
at 0.075-0.09 m sit below that floor at any useful range; they remain RGB-only decor by physics,
not by choice.

## What has never been computed

Every siting sweep to date scored TCP clearance or link **collision** clearance. None scored
**angular subtense at the sensors**. That is the quantity that decides whether the skin sees
anything, and it is cheap to compute.

## Decisions

| | |
|---|---|
| Intervention | **Make the sensed hazard large** — >= 0.25 m frontal width per leg |
| Sensor suite | **Frozen at 40.** No new cameras, no encoder change, comparability preserved |
| Hazard form | **Clustered items on a shared base**, plus small items as RGB-only decor |
| Placement | **Lateral**, in the band where link5_back demonstrably sees the panel |
| Review | **Mandatory stop after W3.** Validation goes to the user before Phase 0 |
| Pace | Move fast on the environment; the training/eval budget is not the constraint |

## W1 — build the resolving-power instrument, and prove it predicts the known signals

New script `scripts/pact_skin_resolvability.py`. For a candidate hazard and a frozen trajectory,
for every frame and each of the 40 sensors:

- `R` = distance from the sensor origin to the hazard's nearest point
- in-cone test against the sensor's 22.5 deg half-angle, using `data.cam_xpos` / `cam_xmat` and the
  `-z` view convention already used by `_protrusion_detected` (`enclosure_reach.py:536-557`)
- `W_perp` = hazard extent projected perpendicular to that sensor's view axis
- **`subtense_px = W_perp / (0.1036 * R)`**

Report per (frame, sensor): `subtense_px`, and aggregate `n_sensor_frames_ge_1px / _ge_2px / _ge_4px`,
`max_subtense_px`, and which sensors ever clear 2 px.

**Validate before use. The model must reproduce the three signals already measured**, on the same
frozen V9.5 trajectories:

| hazard | measured | model must predict |
|---|---|---|
| panel | 11 sensors, 23,004 values | many sensors, high subtense |
| outbound bottle | 2 sensors, 448 + 60 values | 1-2 sensors marginally over 1 px |
| inbound bottle | 1 sensor, 40 values | at or below 1 px almost everywhere |

**If the ordering does not reproduce, stop and report.** A prefilter that cannot retrodict the known
answers must not be used to search. This is the check v8b's AABB instrument never had.

## W2 — site clustered hazards by subtense, not by clearance

Replay the existing frozen trajectories (no rollouts, `mj_forward` only), scoring candidates with
W1's instrument.

**Hazard construction.** Each leg gets one hazard **cluster**: 3-4 items on a shared low base,
spanning **>= 0.25 m** across the expected view axis, with inter-item gaps **< 0.04 m** so the
silhouette stays contiguous at the pixel pitch of the ranges involved. Draw from the tall vessels
already accepted in `diagnostics_output/pact_place_v9_v0b/palette_v9_1.json` (soapbottle 0.089 x
0.089 x 0.247 is the workhorse) plus mugs, cups and pots. The single-item route cannot work: the
palette filter caps `max_dimension` at 0.30 m (`run_pact_place_v8b_palette.py:101`), and a 0.30 m
tall bottle is still only ~0.09 m wide.

**Placement.** Site laterally, at `|y|` comparable to the panel's inner face (0.100) out to ~0.34,
in `z` roughly 0.87-1.15 — the band where `link5_back_sensor_*` already resolves the panel. Do not
site head-on at `y = 0` in front of the gripper; that is where the inbound vessel failed, and no
sensor covers it.

**Score and select** on `n_sensor_frames_ge_2px` and the count of distinct sensors clearing 2 px,
**balanced across left and right panel sides**. Record every candidate, admitted or rejected, with
its reason. Do not top-N a single scalar — that is what drove v7's obstacles to the periphery.

## W3 — raw confirmation, with an admission floor derived from measurement

Re-run the existing validator `scripts/run_pact_place_v9_v0c3_causal_proximity.py` unchanged in its
rendering path — present-versus-parked counterfactual at frozen qpos on the real `[40, 4, 8, 8]`
tensor. It is the ground truth and it is already correct; `baseline_repeat_max_abs_delta_m: 0.0`
shows the renderer is deterministic.

**Replace only the aggregate pass rule at `:414-417`.** Require, per role and per side:

- at least **3 distinct sensors** with changed values, and
- changed values at least matching the **outbound vessel's current left-side result (448 on
  `link3_sensor_2`)** — a floor that is demonstrably reachable because it has already been reached
  on one side, and
- left/right imbalance **<= 4x**, the existing criterion, and
- link5/link6 specifically among the responding sensors, so the response comes from bodies that
  enter the corridor

State the exact numbers in the config **before** running, taken from W2's predicted distribution.
Do not set a floor W2 has not shown reachable, and do not lower one after seeing the result.

## W4 — mandatory stop

Publish a short status doc beside `docs/PACT_PLACE_V95_LOW_WALL_STATUS.md` with the W1 validation
table, the W2 candidate distribution, and the W3 raw result per variant and side. Emit
`authorizes_gate: false`, `authorizes_collection: false`, `authorizes_v1b: false`.

**Then stop.** The user reviews the validation before anything proceeds to V1b or Phase 0.

On approval, hand back to `PACT_PLACE_V9_ENVIRONMENT_PLAN.md` at V1.

## Verification

- W1 reproduces the measured ordering panel >> outbound bottle > inbound bottle on the same frozen
  trajectories; if not, execution stopped.
- The W4 status doc and V7's report both state the ~0.25 m resolving-power floor and the narrowed
  claim it implies; neither calls the clustered hazards "clutter the skin senses" unqualified.
- No admission decision uses TCP clearance, collision clearance, or an AABB gap.
- Every W2 candidate recorded with subtense scores and, if rejected, the reason.
- Selected clusters span >= 0.25 m with gaps < 0.04 m; verified from posed geometry, not from the
  palette's nominal dimensions.
- W3 uses the unmodified rendering path; only the aggregate pass rule changed, and the new floor
  was written to config before the run.
- W3 passes on **both** sides for both roles, with link5/link6 among the responders.
- The 40-sensor contract is untouched: `_HYBRID_SKIN_SENSOR_NAMES` unchanged, encoder SHA-256 still
  `6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206`, tensor still `[T, 40, 4, 8, 8]`.
- Untouched: the panel's geometry and `obstacle_aabbs` entry, all v5/v6b/v6c/v7/v8b/v8c/v9.3/v9.4/
  v9.5 artifacts, and `inter_finger_dist`.

## Constraints

- **Stop after W3 for the user's review.** Do not start V1b or Phase 0.
- **Stop and report if W1 fails to retrodict the known signals.**
- Do not add, move or re-aim any proximity sensor. The suite is frozen at 40.
- Do not modify the encoder or the observation contract.
- Do not weaken the admission floor after seeing a result.
- Do not site a hazard head-on at `y = 0` in front of the gripper.
- Do not add ceiling fixtures; V9.4 measured every fixture bottom at z >= 1.230, which is 200 mm
  above the wrist-visibility ceiling and produced 0.0 wall-fixture bow in 6 of 6 attempts.
- Work in `/root/prox_learning_pact_remediation`; interpreter `/root/act_retrain_venv/bin/python3`;
  `MUJOCO_GL=egl`, `MLSPACES_ASSETS_DIR=/root/prox_learning/assets`, `PYTHONPATH` -> repo
  `submodules/molmospaces`, `OPENBLAS_NUM_THREADS=1` (the cgroup `pids.max` is 3840 and otherwise
  kills numpy imports outright).

## Schedule

```
W1  resolvability instrument + retrodiction check   ~2 h   CPU, replay only
W2  clustered-hazard siting sweep                   ~3 h   CPU, replay only
W3  raw counterfactual confirmation                 ~2 h   renders 40 cameras x 2 worlds
W4  status doc + MANDATORY STOP -> your review        --
                                              ~7 h to a reviewable answer
```

All three stages are replay-only on frozen trajectories. No physics is stepped, no episode is
generated, and nothing can authorize a gate.
