# Proximity Activation Audit

## Which houses have useful valid proximity activation?
house_11 (link5_back_sensor_5, signal 100.0%), house_10 (link5_back_sensor_5, signal 89.2%), house_18 (link6_sensor_1, signal 85.1%), house_12 (link6_sensor_1, signal 82.4%), house_20 (link5_back_sensor_5, signal 80.3%), house_23 (link6_sensor_1, signal 78.2%), house_21 (link5_back_sensor_5, signal 77.8%), house_15 (link5_back_sensor_5, signal 77.3%), house_16 (link6_sensor_1, signal 70.0%), house_6 (link6_sensor_1, signal 69.9%), house_19 (link5_back_sensor_5, signal 68.4%), house_13 (link5_back_sensor_5, signal 67.3%)

## Which links/sensors carry the strongest valid signal?
Top links: link6 (17.4%, 77292 valid sensor-frames), link1 (14.3%, 90174 valid sensor-frames), link5 (11.3%, 128820 valid sensor-frames), link3 (0.6%, 64410 valid sensor-frames), link2 (0.3%, 90174 valid sensor-frames)
Top sensors: link1_sensor_5 (100.0%), link5_back_sensor_5 (51.8%), link5_back_sensor_4 (36.8%), link6_sensor_1 (36.4%), link6_sensor_2 (33.8%), link5_back_sensor_2 (22.2%), link6_sensor_0 (13.4%), link6_sensor_3 (10.6%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 15.1% over 4320 valid frames; pregrasp: 8.7% over 86816 valid frames; grasp_lift: 17.2% over 114976 valid frames; transit: n/a over 0 valid frames; place: n/a over 0 valid frames.
Activation is not concentrated only in pregrasp/grasp_lift; inspect the phase table for spread.

## Which rows should be excluded?
36 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_1 link1_sensor_5 (activation_lt_0_20m_close_to_1;extremely_low_frame_to_frame_variation), house_6 link1_sensor_5 (activation_lt_0_20m_close_to_1;extremely_low_frame_to_frame_variation), house_7 link1_sensor_5 (activation_lt_0_20m_close_to_1;extremely_low_frame_to_frame_variation), house_8 link1_sensor_5 (activation_lt_0_20m_close_to_1;extremely_low_frame_to_frame_variation), house_10 link1_sensor_5 (activation_lt_0_20m_close_to_1;extremely_low_frame_to_frame_variation), house_11 link1_sensor_5 (activation_lt_0_20m_close_to_1;extremely_low_frame_to_frame_variation), house_12 link1_sensor_5 (activation_lt_0_20m_close_to_1;extremely_low_frame_to_frame_variation), house_13 link1_sensor_5 (activation_lt_0_20m_close_to_1;extremely_low_frame_to_frame_variation)

## Decision
Keep the environment: link5/link6 retain meaningful valid activation in pregrasp or grasp_lift.
