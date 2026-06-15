# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_8 (link5_sensor_4, signal 77.1%), house_10 (link6_sensor_0, signal 50.0%), house_1 (link6_sensor_0, signal 27.2%)

## Which links/sensors carry the strongest valid signal?
Top links: link6 (3.9%, 30992 valid sensor-frames), link2 (2.2%, 27118 valid sensor-frames), link5 (0.4%, 23244 valid sensor-frames), link3 (0.2%, 30992 valid sensor-frames)
Top sensors: link6_sensor_6 (7.6%), link6_sensor_0 (7.0%), link2_sensor_3 (6.8%), link2_sensor_6 (6.6%), link6_sensor_1 (5.4%), link6_sensor_5 (3.9%), link6_sensor_2 (3.8%), link2_sensor_1 (1.7%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 4.0% over 1204 valid frames; pregrasp: 1.5% over 7826 valid frames; grasp_lift: 3.5% over 12894 valid frames; transit: 2.9% over 3780 valid frames; place: 3.5% over 4956 valid frames.
Activation is not concentrated only in pregrasp/grasp_lift; inspect the phase table for spread.

## Which rows should be excluded?
6 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_11 link6_sensor_6 (activation_spread_across_many_phases), house_11 link6_sensor_1 (activation_spread_across_many_phases), house_11 link6_sensor_0 (activation_spread_across_many_phases), house_10 link2_sensor_3 (activation_spread_across_many_phases), house_11 link6_sensor_2 (activation_spread_across_many_phases), house_11 link6_sensor_5 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Old audit comparison unavailable: no_old_audit_dir_supplied.

## Decision
Do not use the prior activation signal as evidence: link5/link6 valid activation did not survive filtering.
