# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_26 (link6_sensor_0, signal 100.0%), house_5 (link5_sensor_3, signal 100.0%), house_28 (link5_sensor_3, signal 100.0%), house_3 (link6_sensor_3, signal 68.8%), house_11 (link5_sensor_3, signal 65.3%), house_9 (link5_sensor_3, signal 50.8%), house_0 (link6_sensor_1, signal 50.0%), house_1 (link6_sensor_5, signal 22.5%), house_24 (link6_sensor_3, signal 16.9%)

## Which links/sensors carry the strongest valid signal?
Top links: link6 (10.2%, 31312 valid sensor-frames), link5 (5.3%, 23483 valid sensor-frames), link2 (1.5%, 27398 valid sensor-frames), link3 (0.0%, 31295 valid sensor-frames)
Top sensors: link6_sensor_4 (14.6%), link5_sensor_3 (13.9%), link6_sensor_5 (11.8%), link6_sensor_2 (10.4%), link6_sensor_3 (10.4%), link6_sensor_6 (10.3%), link6_sensor_0 (9.2%), link6_sensor_1 (8.6%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 13.6% over 1148 valid frames; pregrasp: 3.8% over 8554 valid frames; grasp_lift: 13.4% over 12152 valid frames; transit: 10.1% over 3696 valid frames; place: 13.2% over 4886 valid frames.
Activation is not concentrated only in pregrasp/grasp_lift; inspect the phase table for spread.

## Which rows should be excluded?
16 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_2 link2_sensor_6 (activation_lt_0_20m_close_to_1;extremely_low_frame_to_frame_variation;activation_spread_across_many_phases), house_1 link5_sensor_3 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_1 link6_sensor_3 (activation_spread_across_many_phases), house_1 link6_sensor_4 (activation_spread_across_many_phases), house_1 link5_sensor_4 (activation_spread_across_many_phases), house_1 link5_sensor_1 (activation_spread_across_many_phases), house_5 link6_sensor_1 (activation_spread_across_many_phases), house_1 link5_sensor_5 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Old audit comparison unavailable: no_old_audit_dir_supplied.

## Decision
Keep the environment: link5/link6 retain meaningful valid activation in pregrasp or grasp_lift.
