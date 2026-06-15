# Comparison to Existing Sensor Activation Audits

New audit: `FrankaSkinHybridPnP5MassConfig_20260612_023111`

## Headline
- Dataset coverage: 18 H5 files, 170 trajectories, 12,882 frames, 40 sensors.
- Weighted all-sensor activation <0.20m: 8.1%.
- Weighted link5/link6 activation <0.20m: 13.6%; rank 3/7 among audits, descending.
- Link5/link6 pregrasp+grasp_lift activation <0.20m: 13.5%; rank 3/7.
- Suspicious house/sensor rows: 36 of 720 (5.0%); suspicious-ratio rank 2/7 descending.
- Top link/sensor: link6 at 17.4%; link1_sensor_5 at 100.0%.
- Audit decision flag `keep_environment_by_decision_criteria`: True.

## All Audits
| audit | trajs | frames | all <0.20m | link5/6 <0.20m | target phase | other phase | suspicious | top link | keep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cleaned_low_surface_mug_scale_audit_14h5_20260605 | n/a | n/a | 8.3% | 15.2% | 20.1% | 17.4% | 0.0% | link6 21.6% | n/a |
| FrankaSkinHybridPnP5MassConfig_20260612_023111 | 170 | 12,882 | 8.1% | 13.6% | 13.5% | 15.1% | 5.0% | link6 17.4% | True |
| FrankaSkinPickAndPlaceDataGenConfig_combined_15traj_20260606 | 15 | 3,881 | 1.7% | 2.4% | 2.7% | 3.3% | 2.3% | link6 3.9% | False |
| FrankaSkinPickAndPlaceOneHouseMugConfig_latest_20260605_194602 | 15 | 3,930 | 3.9% | 3.8% | 7.3% | 3.6% | 3.0% | link3 4.4% | True |
| FrankaSkinPickAndPlacePilotConfig_latest_20260606_122642 | 15 | 3,923 | 4.3% | 8.1% | 9.4% | 12.1% | 3.7% | link6 10.2% | True |
| FrankaSkinPickAndPlacePilotMediumConfig_latest_20260605_232256 | 15 | 3,965 | 3.5% | 4.6% | 8.1% | 4.8% | 2.1% | link6 7.6% | True |
| FrankaSkinProxNecessityPilotConfig | 29 | 7,303 | 15.8% | 20.9% | 24.0% | 24.8% | 21.4% | link6 22.6% | True |

## Files
- `comparison_to_existing_audits.csv`: headline metrics for every audit.
- `comparison_deltas_vs_existing_audits.csv`: new-minus-old deltas for headline metrics.
- `comparison_per_link_existing_audits.csv`: per-link activation rows across audits.
