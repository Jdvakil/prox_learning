# Frame-level collision and hazard avoidance

Recomputed from all 450 retained telemetry files. Each method has 50 rollouts and 1,485,050 physics contact samples per seed; pooled totals are 150 rollouts and 4,455,150 samples per method.

Collision-free means no hazard-bar, clutter, other-environment or mounted-fixture contact in that sample. Intentional target/receptacle contact is excluded. Hazard-free only excludes hazard-bar contact. Hazard/clutter-free is the previous frame avoidance metric. These are sample-level rates, not collision-free task success or rendered video frames.

| Seed | Method | Collision-free frames | Hazard-free frames | Hazard/clutter-free frames (previous avoidance) |
|---|---|---:|---:|---:|
| 3103 | ACT | 1,343,216 (90.449%) | 1,418,655 (95.529%) | 1,345,883 (90.629%) |
| 3103 | PACT-Finetune | 1,456,050 (98.047%) | 1,464,902 (98.643%) | 1,456,050 (98.047%) |
| 3103 | PACT-Frozen | 1,425,964 (96.021%) | 1,466,867 (98.776%) | 1,425,964 (96.021%) |
| 3104 | ACT | 1,305,914 (87.937%) | 1,344,449 (90.532%) | 1,305,914 (87.937%) |
| 3104 | PACT-Finetune | 1,452,721 (97.823%) | 1,465,444 (98.680%) | 1,452,721 (97.823%) |
| 3104 | PACT-Frozen | 1,432,846 (96.485%) | 1,470,779 (99.039%) | 1,432,846 (96.485%) |
| 3105 | ACT | 1,429,464 (96.257%) | 1,459,313 (98.267%) | 1,432,060 (96.432%) |
| 3105 | PACT-Finetune | 1,460,446 (98.343%) | 1,470,170 (98.998%) | 1,460,446 (98.343%) |
| 3105 | PACT-Frozen | 1,460,443 (98.343%) | 1,472,831 (99.177%) | 1,460,443 (98.343%) |
| pooled | ACT | 4,078,594 (91.548%) | 4,222,417 (94.776%) | 4,083,857 (91.666%) |
| pooled | PACT-Finetune | 4,369,217 (98.071%) | 4,400,516 (98.774%) | 4,369,217 (98.071%) |
| pooled | PACT-Frozen | 4,319,253 (96.950%) | 4,410,477 (98.997%) | 4,319,253 (96.950%) |

Fine-tuning has the highest pooled collision-free and hazard/clutter-free frame rates. Frozen PACT has the highest hazard-only-free rate. The stricter collision-free metric additionally counts ACT contacts with other environment objects and mounted fixtures; overlapping contacts are counted only once.

Source comparison SHA-256: `c68a025fc73b83b4e39432e540282bb37a60b806716b5ee03abff6f68155c0b3`.
