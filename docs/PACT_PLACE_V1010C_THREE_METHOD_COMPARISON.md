# ACT vs PACT-Finetune vs PACT-Frozen

PACT-Finetune has the highest pooled task success and collision-free success rates, and the fewest collision frames. PACT-Frozen has the fewest hazard-bar contact frames. Frozen PACT also leads on both success rates for seed 3105.

## Evaluation scope and metric definitions

The comparison covers seeds **3103, 3104 and 3105**, with **50 matched scenes per seed per method**: 150 rollouts per method and 450 rollouts overall. Each rollout contains 900 actions and 29,701 audited physics contact samples, giving **1,485,050 samples per seed per method** and **4,455,150 samples pooled per method**. All frame counts include both successful and unsuccessful rollouts.

| Metric | Definition | Preferred direction |
|---|---|---|
| Task success rate | Rollouts with final task success, divided by all evaluated rollouts. | Higher |
| Collision-free success rate | Rollouts with final task success and zero forbidden contact throughout the rollout, divided by all evaluated rollouts. | Higher |
| Collision frames | Physics samples with any hazard-bar, clutter, other-environment or mounted-fixture contact. Simultaneous contacts count once. | Lower |
| Hazard frames | Physics samples with hazard-bar contact. These are a subset of collision frames. | Lower |

Intentional grasp-target and place-receptacle contacts are excluded from forbidden collisions. A frame here means an audited physics sample, nominally spaced 2 ms apart. Success rates use rollouts as their denominator; percentages beside frame counts use physics samples.

## Pooled results

Each row covers 150 rollouts and 4,455,150 physics samples.

| Method | Task success rate | Collision-free success rate | Collision frames | Hazard frames |
|---|---:|---:|---:|---:|
| ACT | 64/150 (42.7%) | 46/150 (30.7%) | 376,556 (8.452%) | 232,733 (5.224%) |
| PACT-Finetune | **83/150 (55.3%)** | **63/150 (42.0%)** | **85,933 (1.929%)** | 54,634 (1.226%) |
| PACT-Frozen | 74/150 (49.3%) | 51/150 (34.0%) | 135,897 (3.050%) | **44,673 (1.003%)** |

## Per-seed results

Each row covers 50 rollouts and 1,485,050 physics samples. Scenes are matched across methods within each seed.

| Seed | Method | Task success rate | Collision-free success rate | Collision frames | Hazard frames |
|---|---|---:|---:|---:|---:|
| 3103 | ACT | 19/50 (38.0%) | 16/50 (32.0%) | 141,834 (9.551%) | 66,395 (4.471%) |
| 3103 | PACT-Finetune | 27/50 (54.0%) | 24/50 (48.0%) | 29,000 (1.953%) | 20,148 (1.357%) |
| 3103 | PACT-Frozen | 20/50 (40.0%) | 16/50 (32.0%) | 59,086 (3.979%) | 18,183 (1.224%) |
| 3104 | ACT | 22/50 (44.0%) | 15/50 (30.0%) | 179,136 (12.063%) | 140,601 (9.468%) |
| 3104 | PACT-Finetune | 26/50 (52.0%) | 18/50 (36.0%) | 32,329 (2.177%) | 19,606 (1.320%) |
| 3104 | PACT-Frozen | 19/50 (38.0%) | 10/50 (20.0%) | 52,204 (3.515%) | 14,271 (0.961%) |
| 3105 | ACT | 23/50 (46.0%) | 15/50 (30.0%) | 55,586 (3.743%) | 25,737 (1.733%) |
| 3105 | PACT-Finetune | 30/50 (60.0%) | 21/50 (42.0%) | 24,604 (1.657%) | 14,880 (1.002%) |
| 3105 | PACT-Frozen | 35/50 (70.0%) | 25/50 (50.0%) | 24,607 (1.657%) | 12,219 (0.823%) |

## Interpretation

Compared with PACT-Frozen, PACT-Finetune adds **9 task successes (+6.0 percentage points)** and **12 collision-free successes (+8.0 points)**. Collision frames decrease by **36.8%**, while hazard frames increase by **22.3%**. The contact reduction comes mainly from less clutter contact.

Compared with ACT, PACT-Finetune adds **19 task successes (+12.7 percentage points)** and **17 collision-free successes (+11.3 points)**. It improves both success rates and reduces collision and hazard frames in every seed.

The improvement over PACT-Frozen varies by seed: fine-tuned PACT wins both success metrics in seeds 3103 and 3104, but loses five task successes and four collision-free successes in seed 3105. Frozen PACT has fewer hazard frames in all three seeds. Seed 3105 collision counts differ by only three frames despite the success-rate difference.

The earlier hazard/clutter-only contact total for ACT was **371,293** frames. This report's **376,556 collision frames** also include other-environment and mounted-fixture contacts, using the same forbidden classes as collision-free success. Both PACT variants have zero contacts in those additional classes, so their totals are unchanged. Hazard frames are already included in collision frames and must not be added to them.

## Method and evidence

ACT is the baseline without PACT proximity features. PACT-Frozen keeps the original proximity encoder frozen. PACT-Finetune uses the imported main-branch method: trainable encoder stem/transformer, 128-dimensional CLS readout and minimum pooling. Both PACT variants start from the same original pretrained encoder; the fine-tuned comparison therefore measures the full method change, including readout and pooling, rather than isolating encoder unfreezing alone.

The comparison uses seed-specific final checkpoints at 60,000 updates, the original training/validation split and history-100 controller. All 450 retained trajectories and all 150 three-way initial-state matches were independently verified. Collision-frame unions were additionally reconstructed from the retained contact telemetry.

Each seed uses its own original scene block, so differences between seeds reflect both training randomness and scene variation. These are previously evaluated regression scenes; this comparison does not establish performance on a fresh test set.

- [Audited success and contact metrics](../diagnostics_output/pact_place_v1010c_readout_s3_v1/comparison.json)
- [Frame-level collision and hazard metrics](../diagnostics_output/pact_place_v1010c_readout_s3_v1/root_review/frame_avoidance.json)
- [Frame-count reconstruction script](../diagnostics_output/pact_place_v1010c_readout_s3_v1/root_review/frame_avoidance.py)
- [Full experiment and verification report](../diagnostics_output/pact_place_v1010c_readout_s3_v1/FINAL_REVIEW.md)
