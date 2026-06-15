# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_37 (link6_sensor_1, signal 100.0%), house_3 (link6_sensor_3, signal 100.0%), house_29 (link5_sensor_1, signal 100.0%), house_26 (link6_sensor_6, signal 98.4%), house_16 (link6_sensor_6, signal 76.1%), house_11 (link6_sensor_4, signal 73.9%), house_14 (link6_sensor_0, signal 50.0%)

## Which links/sensors carry the strongest valid signal?
Top links: link3 (4.4%, 31318 valid sensor-frames), link6 (4.2%, 31255 valid sensor-frames), link2 (3.3%, 27412 valid sensor-frames), link5 (3.3%, 23454 valid sensor-frames)
Top sensors: link3_sensor_2 (18.7%), link2_sensor_6 (13.9%), link3_sensor_6 (11.2%), link6_sensor_4 (5.6%), link5_sensor_3 (5.6%), link2_sensor_2 (5.4%), link6_sensor_3 (5.2%), link6_sensor_5 (5.0%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 4.6% over 1078 valid frames; pregrasp: 2.8% over 7485 valid frames; grasp_lift: 10.2% over 11432 valid frames; transit: 3.4% over 5013 valid frames; place: 3.5% over 5389 valid frames.
By weighted valid-frame rate, pregrasp/grasp_lift exceed approach/transit/place.

## Which rows should be excluded?
13 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_11 link3_sensor_6 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_14 link3_sensor_2 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_11 link2_sensor_6 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_6 link2_sensor_6 (activation_spread_across_many_phases), house_11 link3_sensor_2 (activation_spread_across_many_phases), house_4 link3_sensor_2 (activation_spread_across_many_phases), house_11 link2_sensor_2 (activation_spread_across_many_phases), house_1 link5_sensor_3 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Old audit comparison unavailable: no_old_audit_dir_supplied.

## Decision
Keep the environment: link5/link6 retain meaningful valid activation in pregrasp or grasp_lift.
