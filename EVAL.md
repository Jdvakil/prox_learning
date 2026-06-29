# EVAL — Deep two-level fridge pick-and-place (proximity-skin)

Audit of the **FridgeTwoLevelPnPV2** dataset: how much the 40-sensor proximity skin is
exercised, and how much that information is redundant with the cameras. Goal: a task where
proximity is *necessary* (vision insufficient), so PACT can beat vision-only ACT.

## Dataset

| | |
|---|---|
| Config | `FridgeTwoLevelPnPV2Config` (40-sensor hybrid skin, physical open-top fridge) |
| Valid demos | 74 (filter_for_successful_trajectories=True) |
| Mean episode length | 282 frames |
| Proximity stream | `obs/proximity/<sensor>` (T, 4 substeps, 8×8 depth), 40 sensors; non-zero (no zero-bug) |

## 1. Proximity activation (sensor sees a return < 0.5 m)

| Metric | Value |
|---|---|
| Activation rate (<0.5 m) | **29.7%** of all sensor-frames |
| Close rate (<0.2 m) | 13.8% |
| Mean sensors active / frame | **11.9 / 40** |
| Median depth of active sensors | 21.1 cm |
| Frames with ≥8 / ≥16 / ≥24 / ≥32 active | 98.4% / 12.3% / 0% / 0% |
| Sensors active >50% of the time | 10 / 40 (wrist-dominated: link6 cluster, link5_back, link3, link1_sensor_5) |
| Sensors essentially dark | 16 / 40 (forearm/base: link1_0/1/2/6, link2_1/2, link4_4, link5_front) |

Phase-resolved (mean #sensors active / 40, with median active-depth) — proximity is
**temporally concentrated in the grasp window**, matching the paper's thesis:

| Phase | mean #active/40 | median depth |
|---|---|---|
| gripper-close | **17.0** | 18.6 cm |
| grasp | 16.5 | 18.6 cm |
| lift | 15.6 | 18.6 cm |
| retreat | 13.8 | 23.1 cm |
| preplace / place | 11.6 / 11.1 | 20–24 cm |
| pregrasp | 11.1 | 21.2 cm |
| idle/approach | 8.9 | 28.2 cm |

**Read:** the fridge genuinely exercises proximity and concentrates it at grasp/close/lift,
but only the **wrist** is buried (peak ~17/40, never ≥24/40). Gate-A target (≥32/40) not met.

## 2. Vision redundancy (target-object visibility, occlusion-aware segmentation mask)

Fraction of frames the **target object** projects into each camera (`object_image_points`,
which uses a rendered segmentation mask, so walls hide pixels):

| Window | exo_camera_1 | wrist_camera | either | **neither (vision-blind)** |
|---|---|---|---|---|
| all frames | **100%** | 94% | 100% | **0%** |
| grasp window (pregrasp→lift) | **100%** | 78% | 100% | **0%** |
| pregrasp only | 100% | 48% | 100% | 0% |

**Read:** the exocentric camera (mounted high/behind at world (-1.03,-0.58,1.65)) sees the
target **100% of the time, every phase** — zero vision-blind frames in 20,838 frames. The
wrist cam self-occludes at pregrasp (48%) but the exo always covers it.

## Verdict

Proximity is **phase-structured and wrist-concentrated**, but **vision is fully sufficient**
for target localization (0 blind frames; exo at 100%). So on this scene proximity is
*redundant* for localizing the target; its only non-redundant role is near-contact wall
geometry (collision awareness), which affects a few failure cases, not the success rate.

**Predicted eval (no eval run yet): PACT ≈ ACT on success, and PACT-zeroed ≈ PACT** (the
policy can always fall back on the exo view) — the paper's "aggregate tie" pattern, by
construction.

**Lever:** the exo's continuous sightline is the entire redundancy. To make proximity
necessary, **blind the exo** during approach→grasp (see §3, in progress).

## 3. Blinded-exo prototype (front visor) — confirms the lever, but occluding ≠ free

Variant `fridge_two_level_v2_blind.xml`: v2 + a front **visor** (x=0.46, z 0.84–1.05) placed to
intercept the measured exo→target ray (crosses that plane at z≈0.87). Smoke:
`FridgeTwoLevelPnPV2BlindSmokeConfig` (filter off, 3 episodes).

| | exo | wrist | NEITHER (blind) | mean #active (grasp win) | success |
|---|---|---|---|---|---|
| baseline v2 (open-top) | 100% | 78% | **0%** | 14.2 / 40 | (74 valid demos) |
| blind (front visor) | **0%** | 0% | **100%** | 10.1 / 40 | **0 / 2** |

**Finding:** the exo is occludable — visibility went 100%→0%, confirming the redundancy lever
points the right way. **But the visor also blocks the arm**: both rollouts died at *pregrasp*
(length ~33, never reached the grasp phase), so the task is 0% and the 0%/100% numbers are
pre-grasp poses only, not a clean grasp-window measure. The exo's sightline into the
compartment overlaps the corridor the forearm sweeps to reach the target, so a *scene* occluder
that blinds the exo also blinds the robot.

**Conclusion / next step:** blind the exo via the **camera config** (remove or reposition
`exo_camera_1` in `FrankaSkinHybridCameraSystem`), not a scene wall — that removes the
redundancy with zero impact on the arm. The wrist already self-occludes at pregrasp (48%), so
dropping the exo should yield genuine vision-blind frames while the task stays physically
valid. Re-run this audit on an exo-dropped smoke to confirm blind-frame fraction ↑ AND
success preserved before collecting.

## 4. Exo reposition (low/side) — the fix that works

`FrankaSkinHybridCameraSystemExoBlind`: same physical v2 scene, but the exo is moved low + to
the side (`camera_offset=[-0.10,-0.95,0.20]`, `lookat=[0.47,0.05,0.10]`) so its line to the
deep target crosses the y=−0.29 side wall during grasp, while its line to the lifted object /
front pad stays clear. Smoke `FridgeTwoLevelPnPV2ExoBlindSmokeConfig` (filter off, n=3).

Per-phase **target** visibility (exo) and vision-blind fraction (neither camera):

| phase group | exo: v2 → blind | NEITHER(blind): v2 → blind | wrist (blind) | mean #active |
|---|---|---|---|---|
| grasp (2–4) | 100% → **21.6%** | 0% → **22.0%** | 77.6% | 14.6 |
| transit (5–6) | 100% → 23.9% | 0% → 0% | 100% | 14.9 |
| **place (7)** | 100% → **100%** | 0% → 0% | 100% | 12.2 |

**Result:** repositioning the exo blinds it during grasp (100%→21.6%, and 22% of grasp frames
are now fully vision-blind) **while keeping it at place (100%)** — exactly the goal. Critically,
the camera move doesn't touch the arm, so the **task is preserved: 3/3 reached place, episodes
succeed**, and proximity activation is unchanged (14.6 vs 13.9 at grasp). This is the opposite
of the visor (§3), which blinded the exo but broke the task.

**Per-phase refinement (the decisive result).** The grasp-window *aggregate* hides what
matters. Broken out (exo-blind run, n=4):

| phase | exo (blind) | wrist | NEITHER (vision-blind) |
|---|---|---|---|
| approach (0) | 100% | 100% | 0% |
| **pregrasp (2)** | **18%** | 49% | **50%** |
| grasp (3) | 17% | 100% | 0% |
| close (4) | 17% | 100% | 0% |
| place (7) | 100% | 100% | 0% |

At **pregrasp — the decision-critical alignment moment *before* contact — 50% of frames are now
fully vision-blind** (exo occluded by the side wall AND the wrist self-occluded). That is exactly
the proximity-necessity regime, and it emerged from the exo reposition alone. The wrist's 100%
at grasp/close is *post-decision* (the object is already in the gripper) and realistic, so it
does not undermine proximity's role. Pushing the exo to 0% would barely move the pregrasp-blind
fraction — it is capped near the wrist's ~49% pregrasp self-occlusion (NEITHER ≤ 1 − wrist).

**Conclusion:** the exo reposition alone is sufficient; attacking the wrist further has poor ROI
and risks physical realism. This physically valid, exo-reduced scene is **ready to scale** for
the PACT vs ACT collection. (n=4 here — confirm the ~50% pregrasp-blind holds at scale.)

## 5. Final collection (n=196) — structure holds at scale

`FridgeTwoLevelPnPV2ExoBlindConfig`, 200 houses, 12 workers → **196 valid demos, 92.0% success
rate** (196/213 attempts, 0 skipped), 195/196 are complete pick-place trajectories. Per-phase
target visibility + proximity activation over all 196 demos:

| phase | exo | wrist | NEITHER (vision-blind) | mean #active |
|---|---|---|---|---|
| approach (0) | 100% | 100% | 0% | 9.0 |
| **pregrasp (2)** | 14% | 47% | **53.1%** | 11.3 |
| grasp (3) | 20% | 100% | 0% | 16.7 |
| close (4) | 19% | 100% | 0% | 17.1 |
| transit (5–6) | 29% | 100% | 0% | 13.4 |
| place (7) | 100% | 100% | 0% | 11.1 |

The n=4 prototype held: **pregrasp is 53% fully vision-blind** at scale, the exo is preserved at
place (100%), and proximity is strongly active through the grasp window (~17/40). Dataset at
`assets/datagen/fridge_two_level_v2_exoblind/` is ready for the PACT vs vision-only ACT study.

## 6. ACT / PACT training + interim eval (PRELIMINARY — both ~0%)

Converted 196 demos → `act_style_data/fridge_exoblind_v1` (RGB exo+wrist + `proximity_positions
(T,40,3)`). Trained vanilla **ACT** and **PACT** (`--use_proximity --n_proximity_sensors 40`),
same recipe (2000 epochs, chunk 100, hidden 512). Best val loss: ACT 0.078, PACT 0.072 — both fit.

**Interim eval (n=6 each, exo-blind env): ACT 0/6, PACT 0/6.** Failure mode (from the saved
rollouts) is *not* a reach failure: the TCP reaches the object (**min ~7 cm**), but the gripper
**closes ~step 27 (too early, on air) and stays closed** — both policies, identically. With the
exo blinded at pregrasp, the policy times the grasp off proprioception, which mis-fires under
object-pose variation; proximity did not rescue it.

**Key caveat — this is NOT the paper's PACT.** The proximity feature here is a hand-crafted
3-D token per sensor (nearest-pixel x/y + normalized depth). The paper's PACT uses a *learned,
frozen proximity encoder* (~0.82 M-param transformer) that regresses the object's 3-D position
from the 8×8 depth — a far stronger signal for grasp localization/timing. So the current result
tests a weakened proximity path, not PACT proper.

**Decision:** the full n=50×3 eval was NOT run — at 0/6 vs 0/6 it would only confirm a null at
~4.5 h cost. Checkpoints: `submodules/act/ckpts/fridge_{act,pact}/…`.

## 7. Why proximity can't rescue this task — it senses walls, not the object

Attempted the paper's PACT front-end (a learned encoder regressing object 3-D position from each
8×8 depth patch; label = `extrinsic_cv @ obj_world`). Building the training set exposed the root
problem:

- Across **196 eps × 40 sensors × ~280 steps**, the pickup object is visible to *any* proximity
  sensor in only **99 sensor-frames**, by **3 sensors** (link1_sensor_4, link6_sensor_4/3).
- **Closest object sighting = 33.8 cm; 0 frames inside 20 cm; 0 inside grasp range.**

So the deep-fridge encapsulation we maximized is the sensors seeing **walls**, not the small
target. At grasp the object sits between the fingers, outside every sensor's FOV. Consequences:
the object-position encoder is **untrainable** here (99 far samples) and would be **useless** for
grasp timing anyway. At grasp the object is invisible to the exo (blinded by design) AND to
proximity (walls); only the wrist camera sees it, which BC from 196 demos can't exploit to time
the grasp — hence ACT 0 / PACT 0.

**Conclusion:** "maximize proximity activation" maximized *wall*-sensing (collision/encapsulation
context), which is a different quantity from *object* localization. For a PACT>ACT *success*
result, proximity must sense the **object** (e.g., fingertip/palm sensors at the grasp point, a
larger target, or object against a sensed surface) — or proximity's value should be measured as
**collision/safety**, not pick success. The exo-blind fridge as built does not give proximity an
object signal to win on.

## 8. Collision/safety reframe — also negative for current PACT

After the object-sensing result above, I reframed the evaluation around wall-contact safety. I
added contact-category logging to `obs_scene["collision_metrics"]` so rollouts now split robot
contacts into:

- `static_environment`: fridge / wall / scene contacts
- `pickup_object`: contacts with the manipulated object
- `place_receptacle`: contacts with the pad

This matters because the old scalar contact count mixed harmless or task-relevant object contact
with wall scraping. A smoke rollout confirmed the ambiguity: 26/26 old-style contacts were with
the pickup object and **0** were static-environment contacts.

Matched full-horizon safety eval on houses 0-5, exo-blind env, same checkpoints:

| Policy | Success | Static fridge contacts | Static contact steps | Pickup contacts | Total contacts |
|---|---:|---:|---:|---:|---:|
| ACT | 0/6 | **0** | **0** | 126 | 126 |
| PACT | 0/6 | **835** | **417** | 853 | 1688 |
| PACT-zero | 0/6 | **955** | **500** | 2149 | 3104 |

Per-house static contacts:

| house | ACT | PACT | PACT-zero |
|---|---:|---:|---:|
| house_0 | 0 | 139 | 457 |
| house_1 | 0 | 135 | 61 |
| house_2 | 0 | 188 | 202 |
| house_3 | 0 | 183 | 192 |
| house_4 | 0 | 190 | 43 |
| house_5 | 0 | 0 | 0 |

The proximity-clearance proxy agreed directionally: PACT and PACT-zero spend more frames below
5 cm and 10 cm nearest sensed depth than ACT.

**Conclusion:** the safety reframe does **not** rescue the current learned PACT policy. ACT fails
to grasp but does not scrape the static fridge in this matched n=6 eval. PACT fails to grasp and
scrapes the fridge heavily. Zeroing proximity does not fix it, so the unsafe behavior is not an
immediate "proximity helps avoid walls" effect; it is likely an off-distribution learned policy
behavior from this weakened proximity-token setup.

I also briefly launched a full n=50 ACT success eval, but stopped it before any H5s were saved
once the task/sensor mismatch and 0% short-run behavior were confirmed.

