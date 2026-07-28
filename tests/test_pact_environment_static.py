from __future__ import annotations

import json
import inspect
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
sys.path.insert(0, str(MOLMO))

from molmo_spaces.configs.camera_configs import (  # noqa: E402
    FrankaSkinHybridWristOnlyCameraSystem,
)
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (  # noqa: E402
    FrankaSkinPACTCollisionCorridorConfig,
)
from molmo_spaces.tasks.enclosure_reach import (  # noqa: E402
    PactCollisionCorridorPolicy,
    PactCollisionCorridorSampler,
)
from molmo_spaces.env.sensors import get_core_sensors  # noqa: E402
from molmo_spaces.tasks.task import BaseMujocoTask  # noqa: E402


def test_scene_has_exactly_two_parkable_intrusions():
    scene = (
        MOLMO
        / "molmo_spaces"
        / "data_generation"
        / "custom_scenes"
        / "pact_collision_corridor.xml"
    )
    root = ET.parse(scene).getroot()
    bodies = {
        body.attrib["name"]: body
        for body in root.findall(".//body")
        if body.attrib.get("name", "").startswith("pact_intrusion_")
    }
    assert set(bodies) == {"pact_intrusion_left", "pact_intrusion_right"}
    for body in bodies.values():
        assert body.attrib["mocap"] == "true"
        geom = body.find("geom")
        assert geom is not None
        assert geom.attrib["size"] == "0.030 0.240 0.080"
    metadata = json.loads(
        scene.with_name("pact_collision_corridor_metadata.json").read_text()
    )
    assert metadata == {"objects": {}}


def test_runtime_config_is_wrist_plus_all_40_skin_sensors():
    camera = FrankaSkinHybridWristOnlyCameraSystem()
    names = [spec.name for spec in camera.cameras]
    assert names[0] == "wrist_camera"
    assert len(names) == 41
    assert len(set(names[1:])) == 40
    assert "exo_camera_1" not in names
    config = FrankaSkinPACTCollisionCorridorConfig()
    assert config.task_sampler_config.task_sampler_class is PactCollisionCorridorSampler
    assert config.policy_config.policy_cls is PactCollisionCorridorPolicy
    assert PactCollisionCorridorSampler.BASE_FWD == 0.14
    assert PactCollisionCorridorSampler.TARGET_UID == "Cup_10"
    assert PactCollisionCorridorSampler.SASH_APERTURE_HEIGHT == 0.70
    assert PactCollisionCorridorSampler.PANEL_Z == 0.88
    assert PactCollisionCorridorSampler.PANEL_INNER_FACE_Y == 0.08
    assert PactCollisionCorridorPolicy.SAFE_GAP == 0.10


def test_manifest_sensor_order_matches_camera_config():
    manifest = json.loads(
        (ROOT / "configs" / "pact_collision_candidate_manifest_v1.json").read_text()
    )
    runtime = [spec.name for spec in FrankaSkinHybridWristOnlyCameraSystem().cameras][
        1:
    ]
    assert manifest["sensor_names"] == runtime


def test_skin_camera_parameter_intrinsics_use_native_eight_by_eight_resolution():
    config = FrankaSkinPACTCollisionCorridorConfig()
    sensors = get_core_sensors(config)
    parameters = {
        sensor.camera_name: sensor
        for sensor in sensors
        if sensor.uuid.startswith("sensor_param_")
    }
    for name in [
        spec.name
        for spec in FrankaSkinHybridWristOnlyCameraSystem().cameras
        if getattr(spec, "is_proximity_sensor", False)
    ]:
        assert parameters[name].img_resolution == (8, 8)


def test_contact_audit_hook_runs_inside_the_control_physics_loop():
    source = inspect.getsource(BaseMujocoTask.step)
    physics = source.index("self._env.step(self._n_sim_steps_per_ctrl)")
    hook = source.index('getattr(self, "_contact_audit_hook", None)')
    proximity = source.index("sim_steps_in_policy +=")
    assert physics < hook < proximity
