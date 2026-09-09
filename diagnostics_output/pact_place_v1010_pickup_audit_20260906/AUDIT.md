V10.10 does include missed pickups and empty-handed motions toward the placement area. It also includes unsuccessful target contacts, failed delivery after prolonged contact, and one lost placement. The retained telemetry does **not** establish how many episodes achieved a real lift or stable grasp, or how many failures were specifically slips from a previously stable grasp. Calling all contact-only `held` events successful pickups would overstate the evidence.

This audit starts from the corrected [original AUDIT.md](../pact_place_v1011c_post_eval_audit/AUDIT.md) and uses [V10.10 repaired full_run.json](../pact_place_v1010_eval_infra_repair_01/full_run.json): 80 distinct completed rows, 40 per arm, seed 3101, all return codes zero. It excludes the interrupted original run, smoke rows, and abandoned attempts. The comparison is the original wrist-only [V10.11c completed evaluation](../pact_place_v1011c_eval/full_run.json): 300 distinct completed rows, 150 per arm, seeds 3103/3104/3105. It is not the separate later dual-camera experiment.

I independently reconstructed all 380 result/H5/action records. Each H5 has observation steps 0–900, including the initial observation. All ledger identities, prior result hashes, completion statuses, final success values, and H5/task-info success arrays agree. All seven executed arm commands agree between `actions.npz`, its post-ensemble `model_output`, and the next-observation H5 action record. Ledger hashes match `audit_v2.json`. The superseded `audit.json` was not used for measurements.

| Recorded quantity | V10.10 ACT /40 | V10.10 PACT /40 | V10.11c ACT /150 | V10.11c PACT /150 |
|---|---:|---:|---:|---:|
| Ever gripper–target touch | 27 (67.5%) | 28 (70.0%) | 45 (30.0%) | 35 (23.3%) |
| Ever contact-only `held` | 12 (30.0%) | 20 (50.0%) | 16 (10.7%) | 18 (12.0%) |
| Actual lift | Undetermined | Undetermined | Undetermined | Undetermined |
| Stable grasp | Undetermined | Undetermined | Undetermined | Undetermined |
| Object carried toward receptacle | Undetermined | Undetermined | Undetermined | Undetermined |
| Ever recorded receptacle support | 11 (27.5%) | 15 (37.5%) | 11 (7.3%) | 12 (8.0%) |
| Final task success | 11 (27.5%) | 14 (35.0%) | 11 (7.3%) | 12 (8.0%) |
| Collision-free final success | 7 (17.5%) | 10 (25.0%) | 8 (5.3%) | 10 (6.7%) |
| Earlier success lost by step 900 | 0 | 1 | 0 | 0 |

“Undetermined” is not zero. Receptacle support establishes the recorded delivery milestone; it does not reconstruct whether the route involved a stable airborne grasp, pushing, dragging, or a release/fall. The earlier counts of 12/20 `held` episodes are not verified lift or stable-grasp counts.

The exact raw fields and their limits are:

| Quantity | Field / reconstruction | Meaning and limit |
|---|---|---|
| Target touch | Decode JSON bytes in `traj_0/obs/extra/grasp_state_pickup_obj`, then `['gripper']['touching']` | Contact involving target and gripper geometry at the 66 ms observation samples. |
| Contact-only held | Same JSON, `['gripper']['held']` | Target touches gripper geometry and no non-gripper geometry at that sample. No lift, duration, bilateral finger contact, slip, or force-stability requirement. |
| Receptacle support | Decode `traj_0/obs/extra/task_info`, then `['supported_by_receptacle']` | Task support criterion; code includes contact-force support, an object-on-receptacle heuristic, and relative-pose carry-forward. This is not merely robot contact with the tray. |
| Task success | `traj_0/success[-1]`, decoded `task_info[-1]['success']`, and result JSON `task_success` | All agree. Criterion requires support, no robot–object contact, and receptacle displacement/tilt within bounds. |
| Finer target-contact cross-check | Result JSON `contact_audit.frames_with_contact.grasp_target` | Robot–target contacts at the registered 2 ms physics sampling, with non-positive contact distance; broader robot geometry than the gripper sensor. Episode-positive counts also equal 27/28 and 45/35. Every no-touch episode has zero such contacts. |
| Hand movement | `obs/extra/tcp_pose` and `obs/extra/robot_base_pose` | TCP is in robot coordinates. Apply the recorded world-base rotation and translation before measuring world-frame paths. |
| Object pose | `obs/extra/obj_start`; `obs/extra/obj_end`; `traj_0/env_states` | In all 380 files, `obj_start` is constant, `obj_end` is all zeros, and the only env-state dataset is `articulations/panda`. No dynamic object pose is available. |
| Approach to receptacle | `task_info['position_error']` | Distance from receptacle AABB center to target AABB, not TCP distance, object height, or an object-center trajectory. Changes support geometric approach observations but cannot identify carried transport. |

The sensor explicitly documents the limitation in [GraspStateSensor](/root/prox_learning_pact_remediation/submodules/molmospaces/molmo_spaces/env/sensors.py:547). Initial/end-pose semantics are in [ObjectStartPoseSensor](/root/prox_learning_pact_remediation/submodules/molmospaces/molmo_spaces/env/sensors.py:646); frame semantics are in [TCPPoseSensor](/root/prox_learning_pact_remediation/submodules/molmospaces/molmo_spaces/env/sensors.py:101). Support/success and AABB-error definitions are in [PickAndPlaceTask.get_info](/root/prox_learning_pact_remediation/submodules/molmospaces/molmo_spaces/tasks/pick_and_place_task.py:76). Stored `policy_phase` is zero throughout all 380 trajectories, so it does not provide a learned pickup/transport/placement phase label.

The mutually exclusive outcome partition is:

| Outcome group | V10.10 ACT | V10.10 PACT | V10.11c ACT | V10.11c PACT |
|---|---:|---:|---:|---:|
| No target touch | 13 | 12 | 105 | 115 |
| Touch, never held or supported | 15 | 8 | 29 | 17 |
| Contact-only held, never supported | 1 | 5 | 5 | 6 |
| Supported, final failure | 0 | 1 | 0 | 0 |
| Final success | 11 | 14 | 11 | 12 |
| Total | 40 | 40 | 150 | 150 |

Thus V10.10 has 29 ACT and 26 PACT failures, including 13/12 definite missed target engagements and 16/14 failures after touch. V10.11c's failure distribution is much more dominated by never engaging the target. PACT's higher V10.10 contact-only `held` rate is not evidence of a correspondingly higher stable-grasp rate.

For a conservative empty-handed movement count, I reused the existing audit's descriptive criterion: the world-frame TCP comes within 10 cm in XY of the registered tray reference, despite **no target contact anywhere in the episode**. This yields **3/13 ACT and 10/12 PACT V10.10 no-touch failures**, versus **100/105 and 112/115 in V10.11c**. These are hand movements over the tray area, not verified entry into the tray or decoded policy intent. They are a lower-bound subset of placement-like failures; touched-then-lost cases are excluded. The finer contact cross-check rules out an explanation based simply on target contacts missed between the retained observation frames.

All 25 V10.10 no-touch failures have some disallowed environmental collision. Their causes cannot all be declared collision-independent. In V10.11c, 37 ACT and 40 PACT no-touch failures have no disallowed collision, demonstrating that physical obstruction cannot explain the entire newer failure population. Near-stationary final motion, using the existing <2 cm TCP bounding-box diagonal over the last 100 intervals, occurs in 6/13 and 11/12 older no-touch failures versus 78/105 and 81/115 newer ones.

I inspected six V10.10 failed trajectories through the raw event sequences, world-frame hand paths, task-distance signal, and original decoded gripper commands. [Representative plots](representatives.png) and [exact transition records](representatives.json) retain this inspection. None of the 380 scientific trajectory directories retains the referenced wrist RGB video; the H5 camera entry is a filename, not embedded RGB frames. This is telemetry inspection, not a visual confirmation of cup lift or slipping.

| Representative, raw directory | Verified sequence | Assessment |
|---|---|---|
| [ACT 024](../pact_place_v1010_eval_infra_repair_01/rollouts/act/024_4b25e4e9167cc579/trajectory.h5) | No target contact. Close command at observation 192; tray-area entry at 570; open command at 612. No support/success. | Strong empty-handed placement-like sequence after a missed pickup. |
| [PACT 002](../pact_place_v1010_eval_infra_repair_01/rollouts/pact/002_6a09b31ead300086/trajectory.h5) | Target never segmented; no target contact or close command. Tray-area entry at 233; target–tray AABB error remains about 45.4 cm. | Skips pickup and moves toward the tray with an empty hand. |
| [ACT 014](../pact_place_v1010_eval_infra_repair_01/rollouts/act/014_c89083f145f2ffd4/trajectory.h5) | First touch/close at 163; `held` only at isolated steps 174 and 177. Last touch at 205. Tray-area entry at 500; open command at 560. Final object–tray error 48.5 cm. | Failed contact acquisition followed by empty-handed tray motion; the two `held` samples cannot establish stable grasp. |
| [PACT 005](../pact_place_v1010_eval_infra_repair_01/rollouts/pact/005_efa01cb6941d3652/trajectory.h5) | `held` from 162–315 inclusive (154 samples); target contact persists through 900. Never reaches tray area or support. Final target–tray error 50.2 cm. | Prolonged exclusive gripper contact still does not establish useful transport; subsequent loss of `held` is not proof of a drop because touching persists. |
| [PACT 030](../pact_place_v1010_eval_infra_repair_01/rollouts/pact/030_eaf058117b2e80f0/trajectory.h5) | Repeated `held` intervals; TCP enters tray area at 367. Target–tray error decreases from 50.6 to 5.0 cm. Open commanded at 406, but final robot contact remains true and support is never true. | Geometric progress toward delivery, followed by failed support/release. Cannot classify it as empty-handed or a confirmed slip from a stable grasp. |
| [PACT 031](../pact_place_v1010_eval_infra_repair_01/rollouts/pact/031_2b94845a70f4dcc2/trajectory.h5) | `held` 161–343; final gripper touch 347. Support and success at 349–350, 353, 355–357; neither later. At 900 the tray is unchanged and robot contact false, but support false. | Verified loss of an earlier placement, not a never-delivered episode. The data do not show the exact physical loss mechanism. |

Gripper values above are read-only checks of the original decoder, which thresholds the averaged eighth output at 127.5 to 0/255. All six representative command arrays match raw H5 commands and that original threshold exactly. Steps refer to the observation reached by the command; `actions.npz[k-1]` leads into observation k. Closing or opening a command is not evidence that an object was grasped or released.

Action-label timing is a **verified shared supervision defect**, rather than a newly introduced V10.11c defect. I checked all 144 V10.10 and 99 V10.11c source/converted episode pairs, inside this workspace. Converted input `observations/qpos[t]` equals source observation t; converted arm `action[t]` equals `actions/joint_pos[t]`, the command already applied before observation t. Interior equality `joint_pos[t] == commanded_action[t]` holds exactly after float32 conversion. The forward imitation target should instead be `commanded_action[t+1]['arm']`. The simulated-data loader preserves the same-index pairing. Each corpus has two terminal status-only commands without an arm field, which must not be filled with invented zero targets.

The task stores the last action before polling the resulting observation in [task.step](/root/prox_learning_pact_remediation/submodules/molmospaces/molmo_spaces/tasks/task.py:398); the [sensor](/root/prox_learning_pact_remediation/submodules/molmospaces/molmo_spaces/env/sensors.py:181), [converter](/root/prox_learning_pact_remediation/scripts/convert_pact_place_v1011c_to_act.py:76), and [ACT loader](/root/prox_learning_pact_remediation/submodules/act/utils.py:104) confirm the mapping. The original V10.10 conversion inherited the same pattern; the raw/converted comparison verifies it independently. This is one 66 ms timing offset. Its contribution to failure counts is not measured, and V10.10's better results despite sharing it rule out treating it as a complete explanation of the version difference. These timing checks cover seven arm components, not a full gripper-label audit.

Camera visibility is another verified association with an unresolved causal direction:

| Wrist target segmentation | V10.10 ACT | V10.10 PACT | V10.11c ACT | V10.11c PACT |
|---|---:|---:|---:|---:|
| Ever positive | 38/40 | 34/40 | 88/150 | 75/150 |
| Never positive | 2/40 | 6/40 | 62/150 | 75/150 |
| Median fraction of frames positive | 39.01% | 89.79% | 0.777% | 0.0555% |

Every retained `object_image_points/pickup_obj/wrist_camera/num_points` count was checked against finite stored point pairs at every observation. The count is capped at 10: positive means segmentation presence, not visible pixel area or recognizable target detail. All never-segmented episodes fail. Bad approach may cause absent visibility; occlusion or limited field of view may cause bad approach. These traces cannot separate those directions.

Existing demonstration audits additionally show expert acquisition of the wrist view within 100 steps in both versions. The retained [table-camera audit](../pact_place_v1011c_post_eval_audit/learning_failure/table_camera_audit.json) reports initial V10.11c target visibility in 92/99 table videos versus 15/99 wrist videos. Those are collection observations, not evidence that table visibility was equally good in learned rollouts. The original models/evaluator consume wrist RGB only, with proximity additionally used by PACT; task contact flags and the unused table video are not policy inputs. I read this prior camera result without re-decoding its videos or running models.

Chunk-100 temporal averaging is a **credible contributor, not an established causal fix**. The [actual inference code](/root/prox_learning_pact_remediation/submodules/act/eval_pact_frontend_screen_row.py:121) predicts a new length-100 chunk every 66 ms. The current command blends each still-valid chunk's prediction for the current time, with normalized weight `exp(-0.01*age)`. It does not run 100 actions open-loop or average 100 previously executed commands. With full history, the oldest contributing observation is 6.534 s old, weighted mean observation age is 2.726 s, newest-prediction weight is 1.574%, and newest-ten weight is 15.05%. Those ages are not measured actuator latency. They make dilution of newly acquired visual evidence plausible.

I inspected the preserved offline inference results; **no inference was rerun**. On each version's 24 validation demonstrations, pre-touch mean absolute error per arm joint was:

| Existing offline diagnostic | ACT history 1 / 10 / 100 (rad) | PACT history 1 / 10 / 100 (rad) |
|---|---:|---:|
| V10.10, original lagged labels | 0.02116 / 0.02085 / 0.02684 | 0.01963 / 0.02011 / 0.02552 |
| V10.11c, original lagged labels | 0.02584 / 0.02676 / 0.03690 | 0.02637 / 0.02732 / 0.03846 |
| V10.11c, actual forward commands | 0.02591 / 0.02812 / 0.03857 | 0.02640 / 0.02857 / 0.03999 |

Sources: [V10.10 offline result](../pact_place_v1011c_post_eval_audit/offline_ensemble_validation_v1010.json), [V10.11c original-label result](../pact_place_v1011c_post_eval_audit/offline_ensemble_validation.json), and [V10.11c forward-label result](../pact_place_v1011c_post_eval_audit/learning_failure/forward_label_offline.json). The V10.10 measurement is against lagged labels and must not be represented as a forward-command test. V10.11c's improvement with shorter history survives forward-target scoring, but a current-position copying baseline still scores lower error (0.01154 rad) while commanding no useful approach. Teacher-forced error is not task success. These are different aggregation histories of models trained with chunk 100, not chunk-1/10 trained models. Original evaluation artifacts retain post-ensemble outputs only, so alternative commands cannot be recovered by “un-averaging” them.

The V10.11c comparison also changes target placement range, four versus six clutter objects and their geometry, training data (120 versus 75 train episodes), optimizer updates (30,000 versus 20,000), training seeds, and evaluation instances. The [corrected historical audit](../pact_place_v1011c_post_eval_audit/AUDIT.md) documents these confounds. The decline is real, but neither clutter height, label timing, visibility, nor temporal averaging has been isolated as its sole cause. Original V10.11c task successes by seed remain ACT 4/5/2 and PACT 4/6/2, each seed contributing 50 episodes per arm.

**Recommended next diagnostic, not executed:** a separately versioned small paired development panel using frozen chunk-100 ACT/PACT checkpoints, with the original history-100 decoder as reference and history 10 as the single intervention. Retain dynamic target/receptacle world poses and velocities, gripper-relative target pose, finger contacts/support forces, synchronized wrist/table RGB, and every constituent predicted chunk before averaging. Predeclare object-clearance, sustained-grasp/slip, transport, support, and endpoint criteria. This would both distinguish missed pickup from loss during transport and test averaging without changing training chunk size. Use fresh development instances rather than tuning on these inspected held-out results. Corrected-label and added-camera comparisons should be separate subsequent ablations if causal attribution is required. No rollout, training, conversion, or authorization for any proposed diagnostic is implied here.

The new machine-readable [reconstruction.json](reconstruction.json) preserves every episode, contact/support interval, selected event position, raw-result hash, unknown lift/transport status, and count. [reconstruct.py](reconstruct.py) and [representatives.py](representatives.py) contain only retained-file analysis and plotting. Both completed successfully. The reconstruction confirmed unchanged size/mtime for 2,012 input files/directories. Historical artifacts, source and converted datasets, configuration, and checkpoints were not written; no training, evaluation, collection, simulator replay, or model inference was launched. All work and all new outputs stayed in `/root/prox_learning_pact_remediation`; all authorization flags in the new artifacts are false and existing flags were left untouched.
