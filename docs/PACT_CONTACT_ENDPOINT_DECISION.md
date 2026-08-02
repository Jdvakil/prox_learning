# PACT contact-endpoint decision

Decision: `CONTACT_REDUCTION_WITH_TASK_BENEFIT`.

The decision-bearing modality contrast is PACT minus PACT_PERMUTED on hazard-bar contact frames. Negative values favor PACT. PACT_ZERO is an out-of-distribution sensor-failure probe and is never used as modality evidence.

The predeclared wrist-camera partition was 285/285 vision-disadvantaged, so the subset analysis was dropped before rollout outcomes existed.

## Results by policy seed

### Seed 3101

| Arm | Collision-free task success | Hazard frames, mean (median) |
|---|---:|---:|
| ACT | 52/100 (52.0%; 95% CI [42.0%, 62.0%]) | 3885.8 (95% CI [2199.3, 5778.9]; median 0.0) |
| PACT | 60/100 (60.0%; 95% CI [50.0%, 69.0%]) | 380.5 (95% CI [16.7, 1032.1]; median 0.0) |
| PACT_ZERO (OOD sensor failure) | 3/100 (3.0%; 95% CI [0.0%, 7.0%]) | 3970.4 (95% CI [2343.8, 5756.9]; median 0.0) |
| PACT_PERMUTED | 57/100 (57.0%; 95% CI [47.0%, 67.0%]) | 2936.9 (95% CI [1474.7, 4641.7]; median 0.0) |

PACT − PACT_PERMUTED hazard frames: -2556.4 (instance-bootstrap 95% CI [-4121.9, -1184.1]).
PACT − ACT collision-free task success: +8.0 pp (instance-bootstrap 95% CI [-2.0, +18.0] pp).

### Seed 3102

| Arm | Collision-free task success | Hazard frames, mean (median) |
|---|---:|---:|
| ACT | 50/100 (50.0%; 95% CI [40.0%, 60.0%]) | 1763.4 (95% CI [783.5, 2949.1]; median 0.0) |
| PACT | 51/100 (51.0%; 95% CI [41.0%, 61.0%]) | 3062.8 (95% CI [1764.6, 4494.1]; median 0.0) |
| PACT_ZERO (OOD sensor failure) | 9/100 (9.0%; 95% CI [4.0%, 15.0%]) | 3989.2 (95% CI [2617.3, 5510.0]; median 0.0) |
| PACT_PERMUTED | 52/100 (52.0%; 95% CI [42.0%, 62.0%]) | 4729.9 (95% CI [3048.8, 6562.5]; median 0.0) |

PACT − PACT_PERMUTED hazard frames: -1667.1 (instance-bootstrap 95% CI [-2854.8, -702.7]).
PACT − ACT collision-free task success: +1.0 pp (instance-bootstrap 95% CI [-10.0, +12.0] pp).

### Seed 3103

| Arm | Collision-free task success | Hazard frames, mean (median) |
|---|---:|---:|
| ACT | 57/100 (57.0%; 95% CI [47.0%, 67.0%]) | 3421.1 (95% CI [1956.9, 5082.1]; median 0.0) |
| PACT | 60/100 (60.0%; 95% CI [51.0%, 69.0%]) | 1591.4 (95% CI [704.2, 2596.6]; median 0.0) |
| PACT_ZERO (OOD sensor failure) | 8/100 (8.0%; 95% CI [3.0%, 14.0%]) | 3778.3 (95% CI [2315.0, 5409.5]; median 0.0) |
| PACT_PERMUTED | 50/100 (50.0%; 95% CI [40.0%, 60.0%]) | 3307.2 (95% CI [1791.9, 4981.3]; median 0.0) |

PACT − PACT_PERMUTED hazard frames: -1715.8 (instance-bootstrap 95% CI [-3070.0, -552.1]).
PACT − ACT collision-free task success: +3.0 pp (instance-bootstrap 95% CI [-5.0, +11.0] pp).

## Pooled after the seed-specific results

Whole task instances are the bootstrap clusters; all arms and all three seed outcomes for a sampled instance move together.

| Arm | Collision-free task success | Hazard frames, mean (median) | Task success | Any hazard contact |
|---|---:|---:|---:|---:|
| ACT | 159/300 (53.0%; 95% CI [45.3%, 60.7%]) | 3023.4 (95% CI [1854.8, 4348.9]; median 0.0) | 56.3% | 22.3% |
| PACT | 171/300 (57.0%; 95% CI [48.7%, 65.0%]) | 1678.2 (95% CI [954.3, 2497.5]; median 0.0) | 61.0% | 14.0% |
| PACT_ZERO (OOD sensor failure) | 20/300 (6.7%; 95% CI [4.0%, 9.3%]) | 3912.6 (95% CI [2586.2, 5391.9]; median 0.0) | 8.0% | 35.7% |
| PACT_PERMUTED | 159/300 (53.0%; 95% CI [45.0%, 61.0%]) | 3658.0 (95% CI [2235.8, 5268.9]; median 0.0) | 57.0% | 23.3% |

PACT − PACT_PERMUTED hazard frames: -1979.8 (95% CI [-3152.9, -965.2]).

PACT − ACT collision-free task success: +4.0 pp (95% CI [-2.3, +10.3] pp).

Diagnostic PACT − PACT_PERMUTED hazard frames when both arms succeeded at manipulation: -0.8 (95% CI [-19.8, +22.0]).

## Pooled full contrast set

Negative contact/count/depth differences favor the first arm; positive success differences favor it. PACT_ZERO rows remain OOD diagnostics. Fisher p-values are required cluster-unaware descriptive tests; the adjacent whole-instance bootstrap intervals are the cluster-aware inference.

| Contrast | Endpoint | Difference | Whole-instance 95% CI | Fisher exact p |
|---|---|---:|---:|---:|
| PACT − PACT_PERMUTED | collision_free_task_success | +4.0 pp | [+0.0, +8.0] pp | 0.3667 |
| PACT − PACT_PERMUTED | hazard_bar_contact_frames | -1979.8 | [-3153, -965.2] | — |
| PACT − PACT_PERMUTED | hazard_bar_contact_entries | -3230.6 | [-5573, -1354] | — |
| PACT − PACT_PERMUTED | hazard_bar_any_contact | -9.3 pp | [-14.3, -5.0] pp | 0.004526 |
| PACT − PACT_PERMUTED | other_environment_contact_frames | +2.1 | [-1.19, +7.58] | — |
| PACT − PACT_PERMUTED | other_environment_contact_entries | +4.1 | [-1.38, +13.54] | — |
| PACT − PACT_PERMUTED | hazard_bar_maximum_penetration_depth_m | -0.000340 m | [-0.000546, -0.000159] m | — |
| PACT − PACT_PERMUTED | other_environment_maximum_penetration_depth_m | +0.000000 m | [-0.000002, +0.000003] m | — |
| PACT − PACT_PERMUTED | non_target_maximum_penetration_depth_m | -0.000340 m | [-0.000545, -0.000161] m | — |
| PACT − PACT_PERMUTED | manipulation_success | +4.0 pp | [-0.7, +8.7] pp | 0.3612 |
| PACT − PACT_PERMUTED | ordinary_task_success | +4.0 pp | [-0.7, +8.7] pp | 0.3612 |
| PACT_PERMUTED − ACT | collision_free_task_success | +0.0 pp | [-6.0, +6.3] pp | 1 |
| PACT_PERMUTED − ACT | hazard_bar_contact_frames | +634.6 | [-234.1, +1570] | — |
| PACT_PERMUTED − ACT | hazard_bar_contact_entries | +2145.4 | [+480.6, +4033] | — |
| PACT_PERMUTED − ACT | hazard_bar_any_contact | +1.0 pp | [-2.7, +4.7] pp | 0.8458 |
| PACT_PERMUTED − ACT | other_environment_contact_frames | -1.6 | [-6.02, +1.19] | — |
| PACT_PERMUTED − ACT | other_environment_contact_entries | -2.6 | [-9.1, +1.38] | — |
| PACT_PERMUTED − ACT | hazard_bar_maximum_penetration_depth_m | +0.000046 m | [-0.000129, +0.000215] m | — |
| PACT_PERMUTED − ACT | other_environment_maximum_penetration_depth_m | +0.000000 m | [-0.000002, +0.000002] m | — |
| PACT_PERMUTED − ACT | non_target_maximum_penetration_depth_m | +0.000046 m | [-0.000132, +0.000217] m | — |
| PACT_PERMUTED − ACT | manipulation_success | +0.7 pp | [-5.7, +7.0] pp | 0.9344 |
| PACT_PERMUTED − ACT | ordinary_task_success | +0.7 pp | [-5.7, +7.0] pp | 0.9344 |
| PACT − ACT | collision_free_task_success | +4.0 pp | [-2.3, +10.3] pp | 0.3667 |
| PACT − ACT | hazard_bar_contact_frames | -1345.2 | [-2521, -279] | — |
| PACT − ACT | hazard_bar_contact_entries | -1085.2 | [-2404, +181.9] | — |
| PACT − ACT | hazard_bar_any_contact | -8.3 pp | [-13.3, -3.7] pp | 0.01082 |
| PACT − ACT | other_environment_contact_frames | +0.5 | [+0, +1.56] | — |
| PACT − ACT | other_environment_contact_entries | +1.5 | [+0, +4.44] | — |
| PACT − ACT | hazard_bar_maximum_penetration_depth_m | -0.000294 m | [-0.000486, -0.000127] m | — |
| PACT − ACT | other_environment_maximum_penetration_depth_m | +0.000000 m | [+0.000000, +0.000001] m | — |
| PACT − ACT | non_target_maximum_penetration_depth_m | -0.000294 m | [-0.000485, -0.000127] m | — |
| PACT − ACT | manipulation_success | +4.7 pp | [-2.3, +11.7] pp | 0.2811 |
| PACT − ACT | ordinary_task_success | +4.7 pp | [-2.3, +11.7] pp | 0.2811 |
| PACT − PACT_ZERO (OOD) | collision_free_task_success | +50.3 pp | [+42.3, +58.3] pp | 2.154e-43 |
| PACT − PACT_ZERO (OOD) | hazard_bar_contact_frames | -2234.4 | [-3397, -1240] | — |
| PACT − PACT_ZERO (OOD) | hazard_bar_contact_entries | -2286.8 | [-3853, -963] | — |
| PACT − PACT_ZERO (OOD) | hazard_bar_any_contact | -21.7 pp | [-28.3, -15.3] pp | 9.182e-10 |
| PACT − PACT_ZERO (OOD) | other_environment_contact_frames | +0.0 | [-4.227, +6.25] | — |
| PACT − PACT_ZERO (OOD) | other_environment_contact_entries | +1.2 | [-5.563, +11.82] | — |
| PACT − PACT_ZERO (OOD) | hazard_bar_maximum_penetration_depth_m | -0.000277 m | [-0.000446, -0.000114] m | — |
| PACT − PACT_ZERO (OOD) | other_environment_maximum_penetration_depth_m | -0.000032 m | [-0.000061, -0.000008] m | — |
| PACT − PACT_ZERO (OOD) | non_target_maximum_penetration_depth_m | -0.000292 m | [-0.000467, -0.000126] m | — |
| PACT − PACT_ZERO (OOD) | manipulation_success | +53.0 pp | [+45.3, +60.7] pp | 8.995e-46 |
| PACT − PACT_ZERO (OOD) | ordinary_task_success | +53.0 pp | [+45.3, +60.3] pp | 8.995e-46 |
| PACT_ZERO − ACT (OOD) | collision_free_task_success | -46.3 pp | [-54.3, -38.3] pp | 3.835e-38 |
| PACT_ZERO − ACT (OOD) | hazard_bar_contact_frames | +889.2 | [-1.98, +1798] | — |
| PACT_ZERO − ACT (OOD) | hazard_bar_contact_entries | +1201.6 | [+62.03, +2382] | — |
| PACT_ZERO − ACT (OOD) | hazard_bar_any_contact | +13.3 pp | [+7.7, +19.3] pp | 0.0004328 |
| PACT_ZERO − ACT (OOD) | other_environment_contact_frames | +0.5 | [-4.653, +4.29] | — |
| PACT_ZERO − ACT (OOD) | other_environment_contact_entries | +0.3 | [-7.18, +5.567] | — |
| PACT_ZERO − ACT (OOD) | hazard_bar_maximum_penetration_depth_m | -0.000018 m | [-0.000201, +0.000146] m | — |
| PACT_ZERO − ACT (OOD) | other_environment_maximum_penetration_depth_m | +0.000032 m | [+0.000009, +0.000061] m | — |
| PACT_ZERO − ACT (OOD) | non_target_maximum_penetration_depth_m | -0.000001 m | [-0.000184, +0.000165] m | — |
| PACT_ZERO − ACT (OOD) | manipulation_success | -48.3 pp | [-56.3, -40.7] pp | 1.953e-39 |
| PACT_ZERO − ACT (OOD) | ordinary_task_success | -48.3 pp | [-56.0, -40.3] pp | 1.953e-39 |

Failure taxonomies and every seed-specific full contrast are retained in the frozen `analysis.json`; the tables above show seeds first and the complete pooled contrast family.

## Interpretation

Contact reduction and positive pact minus act task success in every seed.

CONTACT_REDUCTION_WITH_TASK_BENEFIT
