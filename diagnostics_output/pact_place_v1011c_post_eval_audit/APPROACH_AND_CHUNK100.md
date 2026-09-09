# Why the policies miss the cup, and the role of chunk 100

## Conclusion

The dominant observed failure is an inaccurate approach followed by tray-area motion without a pickup, often ending in a nearly stationary pose. It is not simply a stationary robot blocked by clutter. There is now specific offline evidence that the long temporal blend makes approach predictions less accurate. This makes aggregation a credible contributor, but does not prove it causes the low closed-loop success rate or that the model must be retrained with a shorter prediction chunk.

No model was trained, no new environment rollout was executed, and no scientific evaluation setting was changed. The new inference diagnostic uses frozen checkpoints and recorded validation observations only. All authorization flags remain false. No gripper commands were analyzed or changed.

## What the failed robots actually do

This follow-up reconstructs world-frame hand motion from the retained robot-frame TCP and world-frame robot-base poses, then checks commanded versus actual arm joints. It covers all 300 current and 80 repaired V10.10 evaluation trajectories, plus 75 current and 120 historical training demonstrations. Frame transformations, normalized quaternions, fixed base poses and action-record agreement were checked explicitly.

| Seed | Arm | Never touches target | Of those, passes over tray area | Of those, nearly stationary at end |
|---|---|---:|---:|---:|
| 3103 | ACT | 33/50 | 32/33 | 27/33 |
| 3103 | PACT | 36/50 | 36/36 | 27/36 |
| 3104 | ACT | 35/50 | 35/35 | 19/35 |
| 3104 | PACT | 38/50 | 36/38 | 27/38 |
| 3105 | ACT | 37/50 | 33/37 | 32/37 |
| 3105 | PACT | 41/50 | 40/41 | 27/41 |
| pooled | ACT | 105/150 | 100/105 | 78/105 |
| pooled | PACT | 115/150 | 112/115 | 81/115 |

“Tray area” means TCP XY is within 10 cm of the registered tray position at any retained sample; this does not mean the hand enters the receptacle. “Nearly stationary” means the diagonal of the TCP's position bounding box over the final 100 control intervals is below 2 cm. These are post-hoc descriptive thresholds, not new success criteria. None of the 195 training demonstrations passes over that tray area before first touching the target.

The never-touch policies are moving: median total TCP path length is 1.25 m for ACT and 1.27 m for PACT. But their median closest distance to the **initial target origin** is still 28.6 cm for both. Only 1/105 ACT and 0/115 PACT never-touch rollouts bring the TCP within 10 cm of that reference. In the current training demonstrations, 73/75 do so by first touch, with median closest distance 7.4 cm.

During the first 100 control steps, 68/105 ACT and 83/115 PACT never-touch failures reduce their distance to that initial target origin by less than 1 cm. Current expert training demonstrations have a median reduction of 22.5 cm over the same initial interval. Paths often head toward an incorrect lateral location or turn toward the tray without first acquiring the cup.

A concrete example is [ACT seed 3103, instance 000](/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1011c_eval/full/act_seed3103/000_a3177246e7677873/result.json). The target starts at world Y=+21.8 cm, while the hand reaches Y=−27.6 cm at step 100. It later ends near the tray without ever touching the cup. This rollout has no recorded disallowed collision.

There are 37 ACT and 40 PACT never-touch failures with **no disallowed collision at any time**; all 77 pass over the tray area. For these subsets, the median final-100-step arm tracking error (seven-joint L2 norm, then median over episodes) is only 0.00075 rad and 0.00110 rad respectively. The arm closely follows the requested stationary/near-stationary commands. Physical blockage cannot explain this entire subset.

Interpretation, not a decoded internal policy phase: this is consistent with executing an empty-handed placement-like routine after a bad approach. The learned policy does not consume the audit's task-success or target-contact flags; its inputs are wrist RGB and joint state, plus proximity embeddings for PACT. A failure to acquire the target visually can therefore compound into the wrong trajectory rather than trigger the expert's contact-aware recovery logic. We have not established how much is caused by visual ambiguity, limited demonstration coverage, or temporal blending.

## What chunk 100 means in this implementation

The [actual inference path](/root/prox_learning_pact_remediation/submodules/act/eval_pact_frontend_screen_row.py:121) generates a new 100-action prediction on **every control step**, through [InferencePolicy.get_action](/root/prox_learning_pact_remediation/submodules/molmospaces/molmo_spaces/policy/base_policy.py:111). It does not execute a 100-step chunk open-loop.

At time `t`, each surviving earlier prediction contributes its action for the *current* time, with normalized weight `exp(-0.01 * age)`. The original evaluator retains ages 0–99. These are different predictions of the current command, not a simple average of previously executed commands.

At the recorded 66 ms control interval, once all 100 predictions are available:

| Quantity | Value |
|---|---:|
| New observation/prediction interval | 0.066 s, about 15.15 Hz |
| Predicted chunk duration | 6.600 s |
| Oldest contributing observation | 6.534 s old |
| Weighted mean age of contributing observations | 2.726 s |
| Weight of the newest prediction alone | 1.57% |
| Combined weight of the newest 10 predictions | 15.05% |
| Combined weight from observations at least 3.3 s old | 37.75% |

The weighted age is **not a measured motor delay**. It does explain why newly acquired visual information may be diluted by predictions made before the target became visible. At the beginning of an episode the available history is shorter, so the full-history percentages do not apply immediately.

## Direct offline test with all six frozen models

I queried every current model on the first 150 recorded observations of all 24 validation demonstrations (one per cell). The same chunk-100 predictions were then combined with histories of 1, 10 or 100 predictions. History 1 uses only the newest prediction's first action. History 10 retains the same decay rule but drops older predictions. **These are not chunk-1 or chunk-10 trained models.**

Only the seven arm joint-position commands were scored against the existing demonstration labels. The primary diagnostic below is mean absolute error per joint, in radians, before the expert's first target touch and within those first 150 steps. Each model contributes 2,717 such frames. Pooling three models per arm gives 8,151 model-frame comparisons per arm, not 8,151 independent episodes.

| Seed | Arm | Newest prediction only | History 10 | Original history 100 |
|---|---|---:|---:|---:|
| 3103 | ACT | 0.02804 | 0.02850 | 0.03733 |
| 3103 | PACT | 0.02397 | 0.02536 | 0.03789 |
| 3104 | ACT | 0.02515 | 0.02651 | 0.03670 |
| 3104 | PACT | 0.02678 | 0.02714 | 0.03878 |
| 3105 | ACT | 0.02432 | 0.02527 | 0.03665 |
| 3105 | PACT | 0.02837 | 0.02947 | 0.03871 |
| pooled | ACT | 0.02584 | 0.02676 | 0.03690 |
| pooled | PACT | 0.02637 | 0.02732 | 0.03846 |

The original history-100 blend has **42.8% higher ACT and 45.8% higher PACT pre-touch error** than the newest prediction alone. History 10 reduces this error by 27.5% and 29.0% relative to the original blend. The direction is consistent across all six models.

The same effect is visible immediately after the target first appears in the expert wrist segmentation: over the following 10 recorded observations, before first touch, mean joint error is ACT 0.04722 with history 100 versus 0.02815 newest-only; PACT 0.04754 versus 0.02922. These are 240 frames per model, 720 model-frame comparisons per arm after pooling.

As a historical control, I repeated the identical offline procedure on both retained V10.10 seed-3101 models and their own 24 validation demonstrations. Pre-touch errors are ACT 0.02684 versus 0.02116 (history 100 versus newest-only) and PACT 0.02552 versus 0.01963. Thus the averaging penalty already exists historically; it is not a newly introduced decoder bug. Current errors are higher even with newest-only inference, consistent with additional environment/data/model difficulty. These cross-version comparisons still use different validation episodes and trained models.

Important limitation: the offline test forces all variants to receive the same expert states and images. Actual shorter-history rollouts would produce different states and future observations; reduced command error can coexist with worse closed-loop stability. No task-success improvement has been measured. The existing scientific rollouts retain only post-ensemble commands, not every constituent predicted chunk or full RGB sequence, so their actions cannot simply be “un-averaged” into an alternative rollout.

## Answer and next discriminating test

The evidence supports **long temporal aggregation as a plausible contributor**, alongside target visibility and weak approach generalization. It does not support declaring the trained 100-action horizon itself the sole cause. The same horizon previously achieved 27.5%/35.0% ACT/PACT task success in V10.10, and many current failures are long-lived empty-handed trajectories rather than a short transient delay.

The clean next experiment would retain the existing chunk-100 checkpoints and compare the original aggregation with a shorter history on the same fresh paired instances. Keep every other setting unchanged and score target acquisition/touch as well as final success and contacts, across all three seeds. This would test aggregation without paying for new training. It has **not** been launched; it would be a separately recorded diagnostic, not a replacement for the completed original evaluation. No chunk-25 work is needed for that test.

## Audit safeguards and artifacts

The distance analysis uses `world_TCP = world_base_rotation * robot_frame_TCP + world_base_translation`. It does not repeat the superseded audit's frame-mixing error. The target reference is its cached initial origin, not reconstructed live motion or a geometric surface-distance measurement; contacts or clutter could subsequently move the object. This qualification also applies to the 10 cm diagnostic.

I found a separate telemetry issue while checking possible camera projections: `CameraParameterSensor` unpacks the configured `(width, height)` tuple as `(height, width)`. For the observed 352×624 RGB image, the retained principal point is (176,312), reversed relative to the actual width/height. These matrices were **not used** for this approach analysis or for the previous segmentation-count analysis, and the ACT/PACT inference code does not consume them. No camera or calibration code was changed. The target segmentation diagnostic uses actual rendered object masks, not those matrices. This is not established as a cause of the policy failure.

Authoritative new data: [approach_chunk100.json](/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1011c_post_eval_audit/approach_chunk100.json), [current offline inference diagnostic](/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1011c_post_eval_audit/offline_ensemble_validation.json), and [historical offline control](/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1011c_post_eval_audit/offline_ensemble_validation_v1010.json). Scripts: [approach audit](/root/prox_learning_pact_remediation/scripts/audit_pact_place_v1011c_approach.py), [offline ensemble audit](/root/prox_learning_pact_remediation/scripts/audit_pact_place_v1011c_offline_ensemble.py). Both offline inference invocations completed with exit code 0; their output included torchvision pretrained-argument deprecation warnings, not failed model loads. All eight checkpoint loads were strict. No training/evaluation workers were spawned by these single-process diagnostics.
