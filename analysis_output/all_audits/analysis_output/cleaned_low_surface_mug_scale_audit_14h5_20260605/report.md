# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_16 (link6_sensor_0, signal 100.0%), house_88 (link6_sensor_4, signal 100.0%), house_29 (link5_sensor_3, signal 100.0%), house_34 (link6_sensor_0, signal 100.0%), house_50 (link6_sensor_0, signal 100.0%), house_81 (link5_sensor_3, signal 100.0%), house_103 (link6_sensor_5, signal 94.4%), house_52 (link5_sensor_3, signal 89.7%), house_14 (link6_sensor_0, signal 45.5%)

## Which links/sensors carry the strongest valid signal?
Top links: link6 (21.6%, 29433 valid sensor-frames), link5 (6.7%, 22081 valid sensor-frames), link3 (3.7%, 29624 valid sensor-frames), link2 (0.0%, 25921 valid sensor-frames)
Top sensors: link6_sensor_6 (27.8%), link6_sensor_1 (25.7%), link6_sensor_2 (25.1%), link6_sensor_5 (25.1%), link6_sensor_0 (24.9%), link6_sensor_4 (23.2%), link5_sensor_3 (13.8%), link3_sensor_2 (13.7%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 15.5% over 1033 valid frames; pregrasp: 14.9% over 7566 valid frames; grasp_lift: 23.7% over 10853 valid frames; transit: 20.6% over 4777 valid frames; place: 14.7% over 4774 valid frames.
By weighted valid-frame rate, pregrasp/grasp_lift exceed approach/transit/place.

## Which rows should be excluded?
36 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_103 link5_sensor_4 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_103 link6_sensor_7 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_62 link3_sensor_2 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_14 link3_sensor_2 (activation_spread_across_many_phases), house_103 link5_sensor_5 (activation_spread_across_many_phases), house_63 link6_sensor_6 (activation_spread_across_many_phases), house_103 link6_sensor_3 (activation_spread_across_many_phases), house_88 link6_sensor_2 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Compared with the old audit, 29 rows had zero/negative old min depths versus 0 after filtering. Max absolute delta in <0.20m activation was 0.3%; max delta in <0.05m near-saturation was 0.3%.
Material-change flag: yes. Treat cleaned numbers as authoritative.

## Decision
Keep the environment: link5/link6 retain meaningful valid activation in pregrasp or grasp_lift.

## Strict near-zero check
Strict output directory: `analysis_output\cleaned_low_surface_mug_scale_audit_14h5_20260605_strict_0p05`.
Max absolute sensor delta in <0.20m activation: 0.73%.
Max absolute sensor delta in pregrasp <0.20m activation: 0.47%.
Max absolute sensor delta in grasp_lift <0.20m activation: 2.69%.
Near-zero artifacts drive the link5/link6 signal: no.
