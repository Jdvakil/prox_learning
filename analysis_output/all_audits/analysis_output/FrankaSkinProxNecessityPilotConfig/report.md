# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_14 (link6_sensor_2, signal 100.0%), house_16 (link5_sensor_4, signal 96.2%), house_31 (link6_sensor_2, signal 85.1%), house_13 (link6_sensor_5, signal 71.4%), house_20 (link6_sensor_0, signal 69.1%), house_11 (link6_sensor_3, signal 50.0%), house_33 (link6_sensor_5, signal 35.4%), house_23 (link5_sensor_3, signal 32.3%), house_12 (link5_sensor_3, signal 12.6%)

## Which links/sensors carry the strongest valid signal?
Top links: link6 (22.6%, 58185 valid sensor-frames), link5 (18.7%, 43610 valid sensor-frames), link2 (12.6%, 51037 valid sensor-frames), link3 (9.6%, 58312 valid sensor-frames)
Top sensors: link6_sensor_3 (27.1%), link5_sensor_3 (25.5%), link6_sensor_5 (23.8%), link5_sensor_4 (22.8%), link6_sensor_2 (22.5%), link6_sensor_4 (22.5%), link2_sensor_6 (22.4%), link6_sensor_1 (21.9%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 25.3% over 2331 valid frames; pregrasp: 14.3% over 9520 valid frames; grasp_lift: 27.3% over 27804 valid frames; transit: 23.2% over 7814 valid frames; place: 26.0% over 9662 valid frames.
Activation is not concentrated only in pregrasp/grasp_lift; inspect the phase table for spread.

## Which rows should be excluded?
93 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_17 link5_sensor_3 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_17 link6_sensor_3 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_17 link6_sensor_7 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_13 link2_sensor_0 (activation_lt_0_20m_close_to_1;extremely_low_frame_to_frame_variation;min_depth_consistently_near_lower_bound;activation_spread_across_many_phases), house_13 link2_sensor_1 (activation_lt_0_20m_close_to_1;extremely_low_frame_to_frame_variation;activation_spread_across_many_phases), house_13 link2_sensor_2 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_13 link2_sensor_3 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_13 link2_sensor_4 (activation_lt_0_20m_close_to_1;extremely_low_frame_to_frame_variation;activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Old audit comparison unavailable: no_old_audit_dir_supplied.

## Decision
Keep the environment: link5/link6 retain meaningful valid activation in pregrasp or grasp_lift.
