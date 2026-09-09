"""Canonical 40-sensor stacking order for the hybrid Franka skin.

This is the order convert, train, and eval must use. It is NOT
``_HYBRID_SKIN_SENSOR_NAMES`` in the env — ``link5_back`` precedes ``link5_front``.

Historically this list lived in ``assets/safety/cvae_v3/meta.json``. The Safety-CVAE
weights were dropped: PACT-raw never used them (peak closeness bypasses the net),
and ``trunk``/``delta`` lost as policy features. The order stays here so the skin
tensor is stable without those checkpoints.
"""

# Closeness map used by PACT-raw (same numbers the old CVAE featurize used).
D_MAX = 0.5
DEAD_PIXEL_M = 0.005

HYBRID_SKIN_SENSOR_ORDER: tuple[str, ...] = (
    "link1_sensor_0",
    "link1_sensor_1",
    "link1_sensor_2",
    "link1_sensor_3",
    "link1_sensor_4",
    "link1_sensor_5",
    "link1_sensor_6",
    "link2_sensor_0",
    "link2_sensor_1",
    "link2_sensor_2",
    "link2_sensor_3",
    "link2_sensor_4",
    "link2_sensor_5",
    "link2_sensor_6",
    "link3_sensor_0",
    "link3_sensor_1",
    "link3_sensor_2",
    "link3_sensor_3",
    "link3_sensor_4",
    "link4_sensor_0",
    "link4_sensor_1",
    "link4_sensor_2",
    "link4_sensor_3",
    "link4_sensor_4",
    "link5_back_sensor_0",
    "link5_back_sensor_1",
    "link5_back_sensor_2",
    "link5_back_sensor_3",
    "link5_back_sensor_4",
    "link5_back_sensor_5",
    "link5_front_sensor_0",
    "link5_front_sensor_1",
    "link5_front_sensor_2",
    "link5_front_sensor_3",
    "link6_sensor_0",
    "link6_sensor_1",
    "link6_sensor_2",
    "link6_sensor_3",
    "link6_sensor_4",
    "link6_sensor_5",
)
