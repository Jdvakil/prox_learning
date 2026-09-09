# V10.11c learning failure: supervision timing and unused visual information

## Conclusion

Keep V10.11c's six-object clutter environment. The most promising next training candidate is **correctly aligned action targets plus wrist and existing table-camera RGB for both ACT and PACT**, still predicting chunks of 100 and using the same frozen proximity encoder. Test one development seed before repeating the full three-seed experiment. This is a recommendation, not an implemented fix or a measured success-rate improvement.

Two additional problems are now verified: a one-control-step action-label lag and the omission of a recorded camera that sees the target much more often. Limited continuous-state coverage, weak generalization, and long temporal averaging remain contributing hypotheses. No available evidence justifies promising >60% success or a PACT advantage after these changes.

No training, collection, dataset conversion, changed environment rollout, or gripper-command analysis was performed for this audit. All authorization flags remain false. Source datasets, checkpoints and historical artifacts were read-only. Only diagnostic scripts, new audit artifacts, and an appended current-run EVAL note were written.

## 1. Confirmed: training labels describe the preceding arm command

The rollout loop obtains an action from observation t, executes it, and then polls observation t+1. `LastCommandedJointPosSensor` and `LastActionSensor` therefore describe the command leading into their observation, not the command that should follow it. The saver does not shift these arrays. The converter nevertheless pairs observation t with `actions/joint_pos[t]`, and the ACT loader preserves that pairing because the dataset is marked simulated.

For a concrete index mapping:

| Observation used as input | Existing converted label | Forward command required as the imitation target |
|---|---|---|
| Observation t | Raw `actions/joint_pos[t]`: command already applied before observation t | Raw `actions/commanded_action[t+1]`: command chosen from observation t |

I checked every episode: **99/99 V10.11c and 144/144 V10.10**. On all interior samples, the stored joint-position label equals the same-index previously commanded arm action exactly after float32 conversion. Converted arm labels and states also match their same-index source arrays exactly. This is not a corrupt-file or wrong-checkpoint diagnosis: the files faithfully implement the wrong forward-imitation timing.

Two episodes in each corpus end with a status-only action, `{'success': [0.0]}`, without an arm field. These terminal status samples were explicitly excluded from forward-command comparisons, not replaced with zero actions. Any new converter must handle that boundary explicitly and check all required action fields.

On the 24 V10.11c validation demonstrations, before first target touch and within the first 150 observations, copying the current arm position has mean absolute joint error **0.00592 rad against the existing labels**, but **0.01154 rad against the actual next command**. The lag makes state copying appear roughly twice as accurate. The physical timing error is one 66 ms interval; it is not the ensemble's multi-second observation age.

This establishes a supervision mismatch. It does **not** establish how many failures it causes: V10.10 shared the mismatch and performed better, and longer predicted sequences can partly compensate. A correction should be published as a new derived dataset, preserving the source and current converted dataset. Recompute normalization statistics from its training split; do not silently reuse old target statistics.

Code evidence: [converter](/root/prox_learning_pact_remediation/scripts/convert_pact_place_v1011c_to_act.py:76), [joint-command sensor](/root/prox_learning_pact_remediation/submodules/molmospaces/molmo_spaces/env/sensors.py:181), [task step ordering](/root/prox_learning_pact_remediation/submodules/molmospaces/molmo_spaces/tasks/task.py:398), [ACT sample construction](/root/prox_learning_pact_remediation/submodules/act/utils.py:104). Raw reconstruction: [supervision_observability.json](supervision_observability.json).

## 2. Confirmed: the useful second camera already exists, but was not used

All 99 source episodes contain `exo_camera_1` RGB videos. I verified each against its recorded SHA-256 and decoded all frames, checking exact agreement with the source H5 frame count. Rendered target-segmentation counts were checked against the finite stored point coordinates for both cameras. These are actual image-space observations, not projections using the previously identified problematic calibration matrices.

| Target segmented at the initial observation | Training episodes | Validation episodes | All episodes |
|---|---:|---:|---:|
| Wrist camera | 10/75 | 5/24 | 15/99 |
| Existing table camera | 69/75 | 23/24 | 92/99 |

Across the first 100 expert observations, mean target-visible fraction is **66.5% for the wrist** and **91.5% for the table view**, over all 99 episodes. Seven episodes initially lack target segmentation in either view. Binary presence is not a guarantee of enough visible pixels or a reliably recognizable target; some examples are partly occluded or small. I inspected initial and intermediate wrist frames and corresponding initial table frames from deterministic negative, central and positive lateral-position representatives.

The current converter exports only wrist RGB, and all six run manifests use `camera_names=['wrist_camera']`. The inference path likewise feeds only wrist RGB. Thus the table video was recorded successfully, but is **not an input to either learned policy**. This followed the inherited training recipe; it was not lost during collection.

The expert's median first wrist sighting in the training set is step 33, after a median 0.568 rad seven-joint displacement from the initial pose. Expert access to task geometry allows target-directed movement before the wrist sees it. A learned wrist-only policy must infer an initial approach from much weaker information. The data do not prove that all indirect visual or proximity cues are absent, but they show why a privileged expert trajectory can be difficult to imitate with the selected observation space.

The existing rollout audit independently found target never touched in **105/150 ACT and 115/150 PACT** episodes; target never segmented in the wrist in **62/150 and 75/150**. Many robots move toward the tray empty-handed. In 77 never-touch rollouts without any disallowed collision, the arm closely tracks its commands. Therefore physical obstruction alone is not the explanation.

Adding table RGB would require a newly versioned conversion and training recipe plus a matching two-camera evaluator. It cannot be enabled reliably by handing an extra image to the old wrist-only checkpoints. Reuse the collection camera configuration, verify its image stream at evaluation, and give the same two RGB views to both arms. Keep geometry, clutter, success predicates and the proximity contract unchanged.

Artifact: [table_camera_audit.json](table_camera_audit.json). Observation design and the mismatch between validation loss and rollout performance are also supported by the authors' [robomimic study](https://robomimic.github.io/study/); its numerical gains on other tasks are not predictions for this experiment.

## 3. Shorter averaging remains plausible, but is not a demonstrated remedy

The earlier offline comparison scored predictions against the existing, lagged labels. I therefore reran all six frozen models on all 24 validation demonstrations, using the **actual next arm commands** as targets. No alternative policy was rolled out. All models still predict 100 actions at every observation.

Mean absolute arm-joint error, before first target touch and within the first 150 observations:

| Seed | Arm | Original history 100 | History 10 | Newest prediction only |
|---|---|---:|---:|---:|
| 3103 | ACT | 0.03893 | 0.02975 | 0.02809 |
| 3103 | PACT | 0.03980 | 0.02736 | 0.02441 |
| 3104 | ACT | 0.03857 | 0.02820 | 0.02593 |
| 3104 | PACT | 0.03970 | 0.02803 | 0.02627 |
| 3105 | ACT | 0.03821 | 0.02640 | 0.02370 |
| 3105 | PACT | 0.04047 | 0.03031 | 0.02852 |
| Three-seed pooled | ACT | 0.03857 | 0.02812 | 0.02591 |
| Three-seed pooled | PACT | 0.03999 | 0.02857 | 0.02640 |

History 10 reduces this error by about **27.1% ACT / 28.6% PACT** versus the original blend. This is 2,717 frames per model, or 8,151 model-frame comparisons per arm, not thousands of independent episodes.

I also tested using query index `age+1` instead of `age` to account for the trained timing offset. At a fixed history of 10, pooled errors become 0.02723 ACT / 0.02798 PACT: a modest additional change, not evidence of a complete fix. With only the newest prediction, shifting the query actually worsens error for ACT seed 3105 and PACT seed 3104. Full shifted history can retain only 99 predictions from a length-100 chunk; that difference is recorded explicitly.

Crucially, a trivial current-position predictor scores **0.01154 rad**, lower than any learned variant, yet would not provide the required approach motion. This is why neither a 28% error reduction nor a corrected output index can be translated into a promised success-rate gain. The test is teacher-forced on expert states; alternate actions would change subsequent observations in an actual rollout.

Temporal averaging is distinct from training chunk size, as described in the original [ACT paper](https://arxiv.org/html/2304.13705v1). We do not need chunk-25 training. The model is already queried every 66 ms; the original history gives contributing observations a weighted mean age of 2.726 seconds once full, not a 100-step open-loop execution interval.

Authoritative forward-target scores: [forward_label_offline.json](forward_label_offline.json). Earlier offline artifacts remain preserved as measurements against the old labels.

## 4. More training may help, but coverage is already a problem

There are only **75 training episodes**, three or four per family×side×pose cell. Balancing those categorical cells did not balance actual target locations: only **3/75 training episodes** have initial |Y| below 0.10 m, versus **18/150 paired evaluation instances**. Both arms fail all 18 of those instances. This is descriptive evidence of a coverage gap, not a causal isolation of lateral position from clutter layout.

All six runs completed 20,000 optimizer updates. Their mean validation L1 continues declining late in training, but much more slowly than training L1: comparing epochs 1300–1499 with 1800–1999, the relative decrease is **4.3% validation versus 12.4% training for ACT**, and **3.6% versus 12.2% for PACT**. Longer training is worth a controlled test after correcting the inputs/targets, but this is not simply evidence that every model needed more epochs. The widening fit/generalization gap and sparse coverage argue against blindly repeating the same data indefinitely.

The collection expert succeeded in 210/482 completed attempts (43.6%); strict-clean filtering then retained 99. That is not a physical upper bound on learned performance, but it warns that a successful-demonstration subset does not establish that the expert reliably solves the intended evaluation distribution. A matched development-instance expert audit would help separate planner limitations from imitation failures.

## Recommended sequence, not executed

1. **Create a corrected derivative of the existing V10.11c data.** Keep source episodes and their 75/24 split identities immutable. Align observations with forward commands, handle terminal status-only samples explicitly, include both recorded RGB views, and recompute train-only statistics. Preserve chunk 100, the frozen encoder SHA `6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206`, 40 sensors, and the 127.5 threshold. The initial pilot needs no new demonstrations.
2. **Run one development seed for both arms**, retaining checkpoints at matched update counts, for example 20k, 30k and 50k. Keep the original temporal ensemble initially. Compare against the existing wrist-only baseline on a fresh, cell-balanced development panel (24–48 instances), including an expert reference if practical. Measure target acquisition, final task success, collision-free success, hazard contacts and clutter contacts. Changing labels and camera together tests a combined remedy, not separate causal effects; a corrected-label wrist-only arm is the useful follow-up ablation if attribution is needed.
3. **Test shorter averaging with those chunk-100 models** on development instances if approach errors persist. Give ACT and PACT the same tuning budget and selection rule. Never select settings by making ACT worse or maximizing the observed gap on the final test set.
4. **Expand data only where the pilot shows remaining failures.** Prioritize underrepresented target positions and expert-corrected approach/recovery states, not more copies of already-easy successes or raw failed actions as imitation targets. Such recovery data addresses the policy-induced distribution shift highlighted by [DAgger](https://proceedings.mlr.press/v15/ross11a.html). A proposed next collection budget could target 10–12 training episodes per categorical cell, but that is an engineering budget, not a proven sample requirement; first ensure continuous-position coverage and a sufficiently reliable expert.
5. **Freeze the improved recipe and run seeds 3103, 3104 and 3105** for both arms on an untouched paired evaluation stream. Report each seed and their equally weighted mean. Do not call the exploratory pilot the three-seed result, discard an unfavorable seed, or reuse development scores as final evidence.

This keeps all six V10.11c clutter objects. If the matched expert itself fails too often, improve its approach/planning/recovery first; do not silently filter the evaluation to easy episodes or label a changed task as the original benchmark. Any changed sampling distribution needs its own version and comparison.

The central scientific limitation remains: clutter here is effectively invisible to the current proximity skin. The historical inbound-vessel `max_w_perp_m` is zero in 7/8 variants and the 40 sensors cover link1–link6 only. A PACT advantage would not, by itself, show that PACT senses this clutter. Extra RGB may improve ACT as well as PACT, which is necessary for a fair test rather than a problem to avoid.

## Execution and verification notes

- The supervision diagnostic initially exited 1 with `KeyError: 'arm'` on a terminal status-only action. Inspection identified the exact field omission; the corrected diagnostic explicitly excludes those terminal forward targets and completed with exit 0.
- The first forward-label inference invocation exited 1 with `NameError: name 'json' is not defined` while printing its first model summary. After adding the missing import, a fresh invocation completed all six strict checkpoint loads and all comparisons with exit 0. The failed invocation is not counted as a completed diagnostic. Torchvision emitted pretrained-argument deprecation warnings during both invocations.
- The camera audit completed with exit 0 and all 99 videos hash-checked and fully decoded. No skipped file is counted as verified.
- Workers were not spawned for these single-process diagnostics. Thread-pool caps were exported before numerical-library imports. GPU inference was read-only.
- Scripts: [supervision/camera audit](/root/prox_learning_pact_remediation/scripts/audit_pact_place_v1011c_learning_failure.py), [forward-target inference audit](/root/prox_learning_pact_remediation/scripts/audit_pact_place_v1011c_forward_labels.py).
