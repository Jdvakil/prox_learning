# ACT versus fine-tuned and frozen PACT: three seeds

Completed fresh fine-tuned PACT training for seeds 3104 and 3105 at 60,000 committed updates each and all 100 new evaluations. Together with completed seed3103, all methods have 150 matched evaluations. Independent raw reading verified all 450 trajectories and all 150 three-way initial-state groups.

| Seed | Method | Task success | Collision-free success | Pickup failures | Hazard/clutter frames |
|---|---|---:|---:|---:|---:|
| 3103 | ACT | 19/50 (38.0%) | 16/50 (32.0%) | 12/50 | 139,167 |
| 3103 | PACT-Finetune | 27/50 (54.0%) | 24/50 (48.0%) | 8/50 | 29,000 |
| 3103 | PACT-Frozen | 20/50 (40.0%) | 16/50 (32.0%) | 16/50 | 59,086 |
| 3104 | ACT | 22/50 (44.0%) | 15/50 (30.0%) | 12/50 | 179,136 |
| 3104 | PACT-Finetune | 26/50 (52.0%) | 18/50 (36.0%) | 7/50 | 32,329 |
| 3104 | PACT-Frozen | 19/50 (38.0%) | 10/50 (20.0%) | 18/50 | 52,204 |
| 3105 | ACT | 23/50 (46.0%) | 15/50 (30.0%) | 11/50 | 52,990 |
| 3105 | PACT-Finetune | 30/50 (60.0%) | 21/50 (42.0%) | 11/50 | 24,604 |
| 3105 | PACT-Frozen | 35/50 (70.0%) | 25/50 (50.0%) | 8/50 | 24,607 |
| pooled | ACT | 64/150 (42.7%) | 46/150 (30.7%) | 35/150 | 371,293 |
| pooled | PACT-Finetune | 83/150 (55.3%) | 63/150 (42.0%) | 26/150 | 85,933 |
| pooled | PACT-Frozen | 74/150 (49.3%) | 51/150 (34.0%) | 42/150 | 135,897 |

## Contact classes

| Method | Hazard bar | Clutter | Grasp target | Avoidance |
|---|---:|---:|---:|---:|
| ACT | 232,733 | 160,802 | 807,383 | 91.666% |
| PACT-Finetune | 54,634 | 32,207 | 1,014,190 | 98.071% |
| PACT-Frozen | 44,673 | 92,866 | 954,989 | 96.950% |

The comparison uses main’s full fine-tuning, 128-D CLS readout and minimum-pooling method, initialized from the same original pretrained encoder. It does not isolate unfreezing alone. Each seed uses its original scene block, so per-seed differences reflect both trained-policy and scene-block variation. These exposed regression scenes do not establish performance on a fresh test set. All original baselines and the seed3103 run were preserved.

[Full metrics, verification and paired changes](comparison.json) · [Per-scene CSV](paired_scene_comparison.csv) · [Execution plan](../../docs/PACT_PLACE_V1010C_THREE_SEED_PLAN.md)
