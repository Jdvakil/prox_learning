# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_1 (link6_sensor_0, signal 100.0%), house_15 (link5_sensor_3, signal 100.0%), house_2 (link6_sensor_0, signal 100.0%), house_3 (link6_sensor_1, signal 100.0%), house_4 (link6_sensor_2, signal 100.0%)

## Which links/sensors carry the strongest valid signal?
Top links: link6 (7.6%, 31600 valid sensor-frames), link3 (2.5%, 31639 valid sensor-frames), link2 (2.5%, 27685 valid sensor-frames), link5 (0.5%, 23676 valid sensor-frames)
Top sensors: link3_sensor_2 (14.0%), link2_sensor_6 (11.9%), link6_sensor_6 (11.0%), link6_sensor_1 (10.8%), link6_sensor_0 (9.7%), link6_sensor_5 (9.5%), link6_sensor_2 (9.0%), link3_sensor_6 (5.6%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 5.1% over 1148 valid frames; pregrasp: 2.8% over 8022 valid frames; grasp_lift: 11.5% over 12026 valid frames; transit: 7.8% over 4588 valid frames; place: 2.3% over 5547 valid frames.
By weighted valid-frame rate, pregrasp/grasp_lift exceed approach/transit/place.

## Which rows should be excluded?
9 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_5 link3_sensor_2 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_4 link2_sensor_6 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_4 link3_sensor_2 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_5 link2_sensor_6 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_5 link3_sensor_6 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_4 link6_sensor_0 (activation_spread_across_many_phases), house_4 link6_sensor_6 (activation_spread_across_many_phases), house_5 link2_sensor_0 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Old audit comparison unavailable: no_old_audit_dir_supplied.

## Decision
Keep the environment: link5/link6 retain meaningful valid activation in pregrasp or grasp_lift.
