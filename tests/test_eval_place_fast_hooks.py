"""Metrics-only evaluation must not compute discarded dataset annotations."""
import sys
from pathlib import Path
from types import ModuleType

ACT = Path(__file__).resolve().parents[1] / "submodules" / "act"
sys.path.insert(0, str(ACT))

from eval_place_fast_hooks import _install_metrics_only_sensor_filter, _without_export_sensors


def sensor(class_name, uuid, result=None):
    def get_observation(self):
        if self.uuid in ("object_image_points", "env_states"):
            raise AssertionError("discarded export sensor was polled")
        return result

    return type(class_name, (), {"uuid": uuid, "get_observation": get_observation})()


def test_filter_preserves_live_policy_inputs_and_stateful_diagnostics():
    observations = {"qpos": {"arm": [0.1] * 7}, "wrist_camera": b"rgb",
                    "link1_sensor_0": b"depth", "grasp_state_pickup_obj": True}
    kept = [sensor(name, uuid, observations[uuid]) for name, uuid in (
        ("RobotJointPositionSensor", "qpos"), ("CameraSensor", "wrist_camera"),
        ("ProximityDepthBufferSensor", "link1_sensor_0"),
        ("GraspStateSensor", "grasp_state_pickup_obj"),
    )]
    exports = [sensor("ObjectImagePointsSensor", "object_image_points"),
               sensor("EnvStateSensor", "env_states")]
    sensors = kept + exports
    filtered = _without_export_sensors(sensors)
    assert filtered == kept
    assert len(sensors) == 6  # Leave the original collection intact.
    assert {s.uuid: s.get_observation() for s in filtered} == observations


def test_install_filters_new_suites_once_without_mutating_factory_output(monkeypatch):
    root = ModuleType("molmo_spaces")
    env = ModuleType("molmo_spaces.env")
    sensors_module = ModuleType("molmo_spaces.env.sensors")
    root.env = env
    env.sensors = sensors_module
    for module in (root, env, sensors_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    calls = []
    original_sensors = [sensor("CameraSensor", "wrist_camera", b"rgb"),
                        sensor("ObjectImagePointsSensor", "object_image_points")]

    def factory(config):
        calls.append(config)
        return original_sensors

    sensors_module.get_core_sensors = factory
    _install_metrics_only_sensor_filter()
    installed = sensors_module.get_core_sensors
    _install_metrics_only_sensor_filter()
    assert sensors_module.get_core_sensors is installed
    for config in ("episode1", "episode2"):
        assert installed(config) == original_sensors[:1]
    assert calls == ["episode1", "episode2"]
    assert len(original_sensors) == 2


def test_contract_gate_retains_substep_pool_at_query_boundaries(monkeypatch):
    """A chunk of three actions consumes reset, then step-three depths [31, 32]."""
    from types import SimpleNamespace
    from eval_place_fast_hooks import _install_contract_sensor_gate
    root = ModuleType('molmo_spaces')
    env_module = ModuleType('molmo_spaces.env')
    sensors = ModuleType('molmo_spaces.env.abstract_sensors')
    tasks = ModuleType('molmo_spaces.tasks')
    task_module = ModuleType('molmo_spaces.tasks.task')
    root.env, root.tasks = env_module, tasks
    env_module.abstract_sensors, tasks.task = sensors, task_module
    for mod in (root, env_module, sensors, tasks, task_module):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    calls = {'depth': 0, 'rgb': 0, 'state': 0}
    def read_depth(self, env, task, **kwargs):
        calls['depth'] += 1
        return list(env.buffer)
    def read_rgb(self, env, task, **kwargs):
        calls['rgb'] += 1
        return env.step
    def read_state(self, env, task, **kwargs):
        calls['state'] += 1
        return env.step
    class Suite:
        def __init__(self):
            self.sensors = {
                'prox': type('ProximityDepthBufferSensor', (), {'get_observation': read_depth})(),
                'rgb': type('CameraSensor', (), {'get_observation': read_rgb})(),
                'state': type('GraspStateSensor', (), {'get_observation': read_state})(),
            }
        def get_observations(self, env, task, **kwargs):
            return {k: s.get_observation(env=env, task=task) for k, s in self.sensors.items()}
    class Task:
        def __init__(self):
            self._proximity_camera_names = ['skin']
            self.env = SimpleNamespace(step=0, buffer=[1, 2])
            self._registered_policy = SimpleNamespace(needs_fresh_policy_observation=lambda: self.env.step % 3 == 0)
            self.suite = Suite()
        def step(self):
            if self._proximity_camera_names:
                self.env.buffer = [10 * self.env.step + 1, 10 * self.env.step + 2]
            return self.suite.get_observations(self.env, self)
    sensors.SensorSuite, task_module.BaseMujocoTask = Suite, Task
    _install_contract_sensor_gate()
    task = Task()
    assert task.suite.get_observations(task.env, task)['prox'] == [1, 2]
    for i in (1, 2, 3):
        task.env.step = i  # Mirrors policy advancing before task.step.
        observation = task.step()
        assert observation['state'] == i
        assert task._proximity_camera_names == ['skin']
    assert observation['prox'] == [31, 32]  # No replacement with a single current depth.
    assert calls == {'depth': 2, 'rgb': 2, 'state': 4}
