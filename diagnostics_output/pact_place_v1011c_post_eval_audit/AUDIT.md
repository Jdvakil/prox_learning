# V10.11c low-success audit

## Finding

The low scores reflect real approach/pickup failures, not an incomplete evaluation being counted as complete or merely a strict collision-free score. V10.11c is substantially less reliably solved than V10.10 by both the collection expert and the learned policies. However, the comparison changes target placement, clutter, training-data quantity, optimizer-update count, training seeds and evaluation instances together. It does **not** isolate clutter height as the cause.

My assessment: this environment/data/training combination is currently too difficult for a useful high-success ACT–PACT comparison. It is not shown to be impossible. The first diagnostic priority is target visibility and approach behavior, followed by data coverage and a matched optimization budget—not another unrestricted large evaluation.

This audit is chunk 100 only. No retraining, new evaluation rollouts, environment changes, gripper-command analysis or statistical tests were performed. Original artifacts and the V10.11c corpus were read-only; authorization flags remain false.

## Recomputed historical comparison

Counts come from each completed rollout's result JSON, not report summary lines. V10.10 and V10.11c task endpoints were additionally checked against all retained H5 `success` samples and `task_info.success`. V10.10 below is the **repaired, complete** evaluation, not the earlier 3/80 thread-exhaustion attempt. V10.9 H5 trajectories are no longer retained, so its historical endpoints can only be checked against per-rollout result JSON.

| Evaluation | Training seeds | Instances per arm | ACT task success | PACT task success | ACT collision-free success | PACT collision-free success |
|---|---|---:|---:|---:|---:|---:|
| V10.9 | 3101 | 40 | 14/40 (35.0%) | 11/40 (27.5%) | 8/40 (20.0%) | 6/40 (15.0%) |
| V10.10, repaired | 3101 | 40 | 11/40 (27.5%) | 14/40 (35.0%) | 7/40 (17.5%) | 10/40 (25.0%) |
| V10.11c, seed 3103 | 3103 | 50 | 4/50 (8.0%) | 4/50 (8.0%) | 3/50 (6.0%) | 4/50 (8.0%) |
| V10.11c, seed 3104 | 3104 | 50 | 5/50 (10.0%) | 6/50 (12.0%) | 4/50 (8.0%) | 4/50 (8.0%) |
| V10.11c, seed 3105 | 3105 | 50 | 2/50 (4.0%) | 2/50 (4.0%) | 1/50 (2.0%) | 2/50 (4.0%) |
| V10.11c, all three | 3103/3104/3105 | 150 | 11/150 (7.3%) | 12/150 (8.0%) | 8/150 (5.3%) | 10/150 (6.7%) |

The current aggregate is also the arithmetic mean of the three seed-level rates because each contributes 50 instances. These are three independently trained policies per arm, not a three-model inference ensemble. Training three seeds was useful for showing that poor performance is not confined to one run; it does not increase demonstrations or training updates available to an individual model. Each seed block also has different held-out instances, so variation between blocks includes both training and evaluation randomness.

PACT's current advantage is only one task success and two collision-free successes across 150 pairs. This does not establish superiority. Historical one-seed results are not controlled estimates of the effect of changing the environment.

## Where the policies fail

The following counts are reconstructed from 380 complete H5 trajectories: 80 V10.10 and 300 V10.11c. Each contains the initial observation plus 900 control steps.

| Recorded milestone | V10.10 ACT, n=40 | V10.10 PACT, n=40 | V10.11c ACT, n=150 | V10.11c PACT, n=150 |
|---|---:|---:|---:|---:|
| Ever touches target with gripper geometry | 27 (67.5%) | 28 (70.0%) | 45 (30.0%) | 35 (23.3%) |
| Ever satisfies contact-only `held` heuristic | 12 (30.0%) | 20 (50.0%) | 16 (10.7%) | 18 (12.0%) |
| Ever supported by receptacle | 11 (27.5%) | 15 (37.5%) | 11 (7.3%) | 12 (8.0%) |
| Final task success | 11 (27.5%) | 14 (35.0%) | 11 (7.3%) | 12 (8.0%) |
| Earlier success lost by final frame | 0 | 1 | 0 | 0 |

Current mutually exclusive outcome groups are ACT: 105 never touch, 29 touch without ever satisfying held/support, 5 satisfy held but never receptacle support, 11 succeed. PACT: 115 never touch, 17 touch without held/support, 6 satisfy held but never receptacle support, 12 succeed.

Thus **70.0% of ACT and 76.7% of PACT rollouts fail without ever touching the target**. There are also 52 ACT and 53 PACT task failures with no recorded disallowed collision of any class. Tightening the collision-free criterion cannot explain the task-success collapse.

The 900-step endpoint is not discarding a large population of earlier successes: none of the current successes is subsequently lost. At step 635, ACT has the same 11 successes; PACT has 11, with one additional success later. No horizon change is warranted by this evidence.

`held` is explicitly a contact heuristic, not a verified stable grasp or lift. It means the target touches gripper geometry and no non-gripper geometry at that sample. This uses existing pickup-progress telemetry; it is not an audit or modification of gripper commands. See [GraspStateSensor](/root/prox_learning_pact_remediation/submodules/molmospaces/molmo_spaces/env/sensors.py:547).

## Target visibility is much worse in closed-loop evaluation

The retained `object_image_points/pickup_obj/wrist_camera/num_points` field is derived from the target segmentation mask. I checked that its count equals the number of finite stored points at every sampled frame. A positive count means some target silhouette was detected, not that the policy understood the target or that a large fraction was visible. Counts are capped at 10 and must not be interpreted as visible pixel area. See [ObjectImagePointsSensor](/root/prox_learning_pact_remediation/submodules/molmospaces/molmo_spaces/env/sensors.py:797).

| Wrist-camera target observation | V10.10 ACT | V10.10 PACT | V10.11c ACT | V10.11c PACT |
|---|---:|---:|---:|---:|
| Target ever segmented | 38/40 (95.0%) | 34/40 (85.0%) | 88/150 (58.7%) | 75/150 (50.0%) |
| Target never segmented | 2/40 | 6/40 | 62/150 | 75/150 |
| Median fraction of sampled frames with target segmented | 39.0% | 89.8% | 0.78% | 0.055% |

None of the 62 ACT or 75 PACT current rollouts with no segmented target ever touches it or succeeds. In contrast, every retained training demonstration in both versions acquires the target within the first 100 control steps. The median fraction of demonstration frames with a segmented target is 93.1% in V10.10 and 91.2% in V10.11c.

This is evidence of a substantial demonstration-to-closed-loop visibility/approach gap. It does **not** distinguish physical occlusion, target outside the camera field of view, and a bad approach trajectory that fails to acquire or retain a view. Visibility can be a consequence as well as a cause of poor actions. No complete RGB-video review or causal occlusion ablation was performed. Searches of the complete old repaired and current scientific rollout logs found none of the sensor's segmentation-error warnings or the previous `libgomp: Thread creation failed` error.

## The environment changed by more than height

V10.10 has four active clutter slots, 01/03/04/06. V10.11c has six, 01/03/04/06/08/09; slot 01 is replaced by a cylinder and 08/09 are additional near-target primitives. Current primitive heights are 32.585 cm for slot 01 and 23.94 cm for slots 08/09. V10.11b to V10.11c changes primitive height by 1.33×; **V10.10 to V10.11c is not a height-only ablation**.

The inherited V10.11 target draw also changes placement. The old corridor sampler uses a central lateral draw plus settling jitter. The new sampler draws across `±(ap_w/2 - 0.05)`, or ±37.5 cm for an 85 cm aperture, and changes the longitudinal position. See the [old target draw](/root/prox_learning_pact_remediation/submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py:1390), [new target draw](/root/prox_learning_pact_remediation/submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py:3645), and [V10.11c inheritance/height checks](/root/prox_learning_pact_remediation/submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py:3889).

Cached initial world-frame target positions reconstructed from H5 confirm this difference:

| Target position sample | Training lateral range | Evaluation lateral range | Training / evaluation mean X |
|---|---:|---:|---:|
| V10.10 | −9.87 to +8.92 cm | −7.78 to +5.18 cm | 78.10 / 78.04 cm |
| V10.11c | −35.32 to +34.68 cm | −37.23 to +36.61 cm | 70.06 / 71.10 cm |

The evaluated lateral span is 73.84 cm rather than 12.96 cm. These are observed sample ranges, not identically sized estimates of distribution support; the code independently confirms the wider draw. New training data also has the broad placement distribution, so this is not evidence that evaluation accidentally used a different sampler from collection.

Wider placement alone does not explain all failures. In a post-hoc current subset with `|y| < 10 cm`, both arms score 0/18. For `10 ≤ |y| < 20 cm`, task success is ACT 5/51 and PACT 6/51; at `|y| ≥ 20 cm`, both score 6/81. These are descriptive slices, not evidence that changing only the spawn range will fix the task.

## The expert and the training budget both changed materially

| Collection/training quantity | V10.10 | V10.11c |
|---|---:|---:|
| Collection attempts | 313 | 482 |
| Completed collection attempts | 303 | 482 |
| Expert task success / completed attempts | 240/303 (79.2%) | 210/482 (43.6%) |
| Accepted strict-clean / all attempts | 144/313 (46.0%) | 99/482 (20.5%) |
| Train / validation episodes | 120 / 24 | 75 / 24 |
| Train episodes per cell | 5 | 3–4 |
| Converted training timesteps | 59,064 | 28,566 |
| Epochs per model | 2,000 | 2,000 |
| Actual optimizer steps per model | 30,000 | 20,000 |

The expert results are collection-ledger diagnostics with different streams and collection stopping rules, not a paired expert benchmark. Still, the drop supports a genuinely less reliably solved task rather than a failure unique to ACT or PACT. Expert grasp-phase failures rose from 62 collection attempts to 248; attempts flagged `cup_not_lifted` rose from 30 to 235. Defect categories overlap and must not be summed as unique failed episodes.

The new models receive **37.5% fewer training episodes, 51.6% fewer recorded training timesteps, and one-third fewer optimizer updates**. The update count comes directly from all eight actual epoch logs, not a presumed configuration: at batch 8, 120 episodes yield 15 updates/epoch while 75 yield 10. Recorded timesteps describe demonstration coverage; training samples a chunk per episode, so they are not a count of optimizer steps.

All six current runs did finish 2,000 epochs. Current best validation loss is approximately 0.120–0.160 versus 0.104–0.106 previously, but these losses use different datasets and normalization and are not a controlled proof of undertraining. More epochs may help or may overfit; this audit cannot establish which. Matching 30,000 updates with the present 75 training episodes would require 3,000 epochs at the same batch size.

## Contact comparison is mixed, not a blanket PACT safety win

| Evaluation | Arm | Hazard-contact episodes | Hazard frames summed | Hazard frames per rollout | Clutter-contact episodes |
|---|---|---:|---:|---:|---:|
| V10.10 | ACT | 12/40 (30.0%) | 120,933 | 3,023.3 | 22/40 (55.0%) |
| V10.10 | PACT | 11/40 (27.5%) | 59,526 | 1,488.2 | 17/40 (42.5%) |
| V10.11c | ACT | 50/150 (33.3%) | 227,194 | 1,514.6 | 54/150 (36.0%) |
| V10.11c | PACT | 55/150 (36.7%) | 283,659 | 1,891.1 | 39/150 (26.0%) |

Frames here are audited physics samples, not video frames or contact-pair entries. Per-rollout means avoid comparing sums across 40 versus 150 episodes. Current PACT contacts clutter in fewer episodes, but contacts the hazard bar in more episodes and for more frames than ACT. Low target engagement also makes low contact counts ambiguous: not approaching the task can avoid clutter without demonstrating competent avoidance. Full current per-seed contact and slot 01/08/09 stability results remain in [EVAL.md](/root/prox_learning_pact_remediation/EVAL.md:1215).

Clutter here is effectively invisible to the proximity skin: inbound vessel `max_w_perp_m = 0.000` in 7/8 variants of [the retained resolvability artifact](/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v9_w1_resolvability_full/resolvability.json), and the 40 sensors cover link1–link6 only. A PACT–ACT difference is **not evidence that PACT senses this clutter**. This historical skin measurement and the current wrist-camera target-visibility diagnostic concern different sensors and different objects.

## Integrity checks, limitations and next diagnostic

The current completion ledger has exactly 300 rows, all `status=complete` and actual `returncode=0`. All 300 raw task endpoints are present. The original collection sampler is in `molmo_spaces.tasks.enclosure_reach`, on branch `experiment/pact-vs-act-remediation-v2`, commit `70dedc07f34ed7f8335aed7f694ddef7ef823d3d`; all nine frozen implementation hashes were rechecked successfully. The existing final verification also records 694 preserved source files and all 150 paired initial-observation checks. Chunk 100, temporal ensemble, frozen encoder, 127.5 threshold and the evaluation endpoint remain unchanged. No active evaluation or orphan `multiprocessing.spawn` worker was present at audit completion.

The experiment's earlier strict-RGB smoke failure and the intentional scheduler-resize interruption remain disclosed in [EVAL.md](/root/prox_learning_pact_remediation/EVAL.md:1248); this audit does not relabel those stages as successes. They did not remove scientific rows from the final denominator.

Recommended next work, **not executed**: a small separately versioned target-visibility/approach diagnostic that varies target placement and primitive height independently, retaining the original task as the reference. Any zero-shot scene intervention would be diagnostic, not a replacement benchmark score. Then choose a better-covered demonstration set and/or a validation-selected matched-update training comparison before committing to another full three-seed run. Do not tune checkpoints or parameters on these already inspected held-out outcomes and call the resulting score a fresh test.

### Reproducibility and audit correction

Authoritative machine-readable outcomes: [audit_v2.json](/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1011c_post_eval_audit/audit_v2.json). Positions and visibility: [geometry_visibility.json](/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1011c_post_eval_audit/geometry_visibility.json). The per-rollout records retain identity and raw-result hashes. Read-only derivation scripts are [outcome audit](/root/prox_learning_pact_remediation/scripts/audit_pact_place_v1011c_outcomes.py) and [geometry/visibility audit](/root/prox_learning_pact_remediation/scripts/audit_pact_place_v1011c_geometry.py).

The initial `audit.json` draft is **superseded**. It incorrectly treated the cached `obj_start` field as live object motion and mixed world-frame object coordinates with robot-frame TCP coordinates. Its derived lift, displacement and TCP-distance claims are withdrawn. The corrected `audit_v2.json` removes those fields; task-success, contact, support and pickup-contact counts were unaffected. Actual continuous target motion/lift was not reconstructed because the needed dynamic object pose was not retained. This report uses only the corrected quantities and explicitly checked sensor semantics.
