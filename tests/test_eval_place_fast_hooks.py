"""Metrics-only evaluation must not compute discarded dataset annotations."""
import sys
from pathlib import Path
from types import ModuleType
import pytest

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


@pytest.mark.parametrize('readout', [False, True])
def test_contract_gate_retains_substep_pool_at_query_boundaries(monkeypatch, readout):
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
            self._registered_policy.needs_fresh_proximity_observation = lambda: readout or self.env.step % 3 == 0
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
        if readout:
            assert observation['prox'] == [10 * i + 1, 10 * i + 2]
        assert task._proximity_camera_names == ['skin']
    assert observation['prox'] == [31, 32]  # No replacement with a single current depth.
    assert calls == {'depth': 4 if readout else 2, 'rgb': 2, 'state': 4}


def test_single_sample_rgb_context_restores_model_for_native_proximity(monkeypatch):
    from types import SimpleNamespace
    from eval_place_fast_hooks import _install_deterministic_rgb_renderer
    root = ModuleType('molmo_spaces')
    env_package = ModuleType('molmo_spaces.env')
    env_module = ModuleType('molmo_spaces.env.env')
    renderer_package = ModuleType('molmo_spaces.renderer')
    renderer_module = ModuleType('molmo_spaces.renderer.opengl_rendering')
    root.env, root.renderer = env_package, renderer_package
    env_package.env = env_module
    renderer_package.opengl_rendering = renderer_module
    env_module.HAS_FILAMENT = False
    for module in (root, env_package, env_module, renderer_package, renderer_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    samples_seen = []
    class Renderer:
        def __init__(self, model=None, model_bindings=None, fail=False):
            model = model if model is not None else model_bindings.model
            samples_seen.append(model.vis.quality.offsamples)
            if fail:
                raise RuntimeError('context failure')
    renderer_module.MjOpenGLRenderer = Renderer
    model = SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=4)))
    _install_deterministic_rgb_renderer()
    installed = Renderer.__init__
    _install_deterministic_rgb_renderer()
    assert Renderer.__init__ is installed
    Renderer(model=model)
    assert model.vis.quality.offsamples == 4  # Native proximity still sees collection setting.
    Renderer(model_bindings=SimpleNamespace(model=model))
    with pytest.raises(RuntimeError, match='context failure'):
        Renderer(model=model, fail=True)
    assert model.vis.quality.offsamples == 4
    assert samples_seen == [0, 0, 0]
    env_module.HAS_FILAMENT = True
    with pytest.raises(ValueError, match='classic OpenGL'):
        _install_deterministic_rgb_renderer()


def test_v12_settle_park_drops_parked_bottle_before_overlap_check(monkeypatch):
    from types import SimpleNamespace
    from eval_place_fast_hooks import _install_v12_preview_settle_park

    calls = []
    monkeypatch.setitem(sys.modules, 'mujoco', SimpleNamespace(
        mj_forward=lambda model, data: calls.append(('forward', model, data))))
    env = SimpleNamespace(current_model='model', current_data='data')
    parked = 'pact_clutter_01/Soap_Bottle_30'
    kept = 'pact_clutter_06/Soap_Bottle_11'

    def original_settle(received):
        calls.append(('settle', list(sampler._pact_active_clutter_names)))
        assert received is env
        raise ValueError('settled clutter overlaps target')

    sampler = SimpleNamespace(
        _pact_active_clutter_names=[parked, kept],
        _settle_injected_object=original_settle,
    )
    overlay = SimpleNamespace(
        _park_household=lambda model, data: calls.append(('park', model, data)),
        _is_parked_household=lambda name: name.split('/')[-1] == 'Soap_Bottle_30',
    )
    _install_v12_preview_settle_park(sampler, overlay)
    with pytest.raises(ValueError, match='overlaps target'):
        sampler._settle_injected_object(env)
    assert calls == [('park', 'model', 'data'), ('forward', 'model', 'data'), ('settle', [kept])]
    assert sampler._pact_active_clutter_names == [parked, kept]
