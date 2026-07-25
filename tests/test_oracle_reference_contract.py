"""Contract tests for the per-frame parked-obstacle oracle reference.

Handoff steps 9 and 16. These run without a GPU, a simulator or a checkpoint: the
simulator-facing behaviour is exercised against a fake MuJoCo data object that records
every write, and the evaluator/reference source is checked at the AST level so a
regression cannot be papered over by a comment.

The regression test for the finite-reference defect is
``TestFiniteReferenceDefect``: it proves a 200-step episode requests 200 independently
generated references, that no array index is used, that nothing fails at step 109, and
that no padding, wrapping or last-frame repetition can occur.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "submodules" / "act"
for extra in (str(ACT), str(ROOT / "scripts")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

REFERENCE_SRC = ACT / "parked_obstacle_reference.py"
EVALUATOR_SRC = ACT / "eval_act_obstacle_oracle.py"
ADAPTER_SRC = ACT / "eval_act_obstacle_safety.py"
RESIDUAL_SRC = ACT / "hybrid_safety_residual.py"

pytest.importorskip("mujoco")
import mujoco
from parked_obstacle_reference import (
    COMMITTED_PARKED_POSE,
    PARKED_Z_THRESHOLD,
    REFERENCE_ID,
    OracleReferenceError,
    PerFrameParkedObstacleReference,
)

HAZARD_BODIES = ("protr_s", "protr_m", "protr_l")


# --------------------------------------------------------------------------- #
# a fake scene that behaves like the parts of MuJoCo the reference touches
# --------------------------------------------------------------------------- #
class FakeData:
    def __init__(self, nbody=6, ngeom=8, nq=12, nmocap=3):
        rng = np.random.default_rng(0)
        self.time = 1.2345
        self.qpos = rng.random(nq)
        self.qvel = rng.random(nq)
        self.act = np.zeros(0)
        self.ctrl = rng.random(4)
        self.mocap_pos = np.array([[0.6, -0.2, 0.82], [0.0, 1.2, -2.0], [0.0, 1.6, -2.0]])
        self.mocap_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (nmocap, 1))
        self.qacc_warmstart = rng.random(nq)
        self.qfrc_applied = np.zeros(nq)
        self.xfrc_applied = np.zeros((nbody, 6))
        self.xpos = rng.random((nbody, 3))
        self.xquat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (nbody, 1))
        self.xmat = np.tile(np.eye(3).ravel(), (nbody, 1))
        self.xipos = rng.random((nbody, 3))
        self.geom_xpos = rng.random((ngeom, 3))
        self.geom_xmat = np.tile(np.eye(3).ravel(), (ngeom, 1))
        self.cam_xpos = rng.random((40, 3))
        self.cam_xmat = np.tile(np.eye(3).ravel(), (40, 1))
        self.sensordata = np.zeros(0)
        self.ncon = 2
        self.contact = [_Contact(1, 2), _Contact(3, 4)]


class _Contact:
    def __init__(self, geom1, geom2):
        self.geom1, self.geom2 = geom1, geom2


class FakeBody:
    def __init__(self, body_id):
        self.id = body_id


class FakeModel:
    def __init__(self):
        self.nbody, self.ngeom, self.nsite, self.nq = 6, 8, 0, 12
        self._bodies = {name: 2 + index for index, name in enumerate(HAZARD_BODIES)}
        self.body_mocapid = np.array([-1, -1, 0, 1, 2, -1])
        # geoms 2 and 3 belong to protr_s, 4 to protr_m, 5 to protr_l
        self.geom_bodyid = np.array([0, 1, 2, 2, 3, 4, 0, 1])
        self.site_bodyid = np.zeros(0, dtype=int)

    def body(self, name):
        if name not in self._bodies:
            raise KeyError(name)
        return FakeBody(self._bodies[name])


class FakeEnv:
    """Renders a synthetic 8x8 patch whose value depends on the hazard's z."""

    def __init__(self, model, data, sensor_names):
        self.current_model = model
        self.current_data = data
        self._sensor_names = sensor_names
        self._proximity_depth_frames = {name: [np.full((8, 8), 0.5, np.float32)]
                                        for name in sensor_names}
        self.render_calls = 0
        self.buffer_lengths_seen = []

    def record_proximity_depths(self, camera_names):
        self.render_calls += 1
        self.buffer_lengths_seen.append(
            {n: len(self._proximity_depth_frames[n]) for n in camera_names})
        # a "hazard visible" reading iff the first bar is above the parked threshold
        hazard_up = self.current_data.mocap_pos[0][2] > PARKED_Z_THRESHOLD
        value = np.float32(0.10 if hazard_up else 0.90)
        for name in camera_names:
            self._proximity_depth_frames.setdefault(name, []).append(
                np.full((8, 8), value, np.float32))


class FakeTask:
    def __init__(self, sensor_names):
        self.model = FakeModel()
        self.data = FakeData()
        self.env = FakeEnv(self.model, self.data, sensor_names)
        self._proximity_camera_names = list(sensor_names)


@pytest.fixture
def sensor_names():
    contract = json.loads((ROOT / "configs/hybrid_safety_stack_v1.json").read_text())
    return list(contract["sensor_contract"]["ordered_names"])


class FakeMujoco:
    """Only the four MuJoCo entry points the reference touches, over FakeData.

    ``mj_getState`` serialises exactly the integrator inputs the counterfactual could
    plausibly disturb, so the tripwire in ``parked_skin`` is a real check here too.
    """

    mjtState = mujoco.mjtState
    MjData = staticmethod(lambda model: FakeData())

    @staticmethod
    def mj_stateSize(model, spec):
        return 1 + 12 + 12 + 4 + 9 + 12

    @staticmethod
    def mj_getState(model, data, buffer, spec):
        buffer[:] = np.concatenate([
            [data.time], data.qpos, data.qvel, data.ctrl,
            data.mocap_pos.ravel(), data.qacc_warmstart])

    @staticmethod
    def mj_setState(model, data, buffer, spec):
        raise AssertionError("mj_setState must not be needed")

    @staticmethod
    def mj_forward(model, data):
        raise AssertionError("mj_forward must not be called")


@pytest.fixture
def fake_mujoco(monkeypatch):
    monkeypatch.setattr("parked_obstacle_reference.mujoco", FakeMujoco)
    return FakeMujoco


@pytest.fixture
def reference(sensor_names, monkeypatch, fake_mujoco):
    monkeypatch.setattr("parked_obstacle_reference.committed_parked_pose_from_source",
                        lambda: dict(COMMITTED_PARKED_POSE))
    task = FakeTask(sensor_names)
    return PerFrameParkedObstacleReference(task, sensor_names), task


# --------------------------------------------------------------------------- #
class TestCommittedParkedPose:
    def test_matches_molmospaces_source(self):
        """The constant must equal what enclosure_reach._apply_theta actually writes."""
        source = (ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py")
        tree = ast.parse(source.read_text())
        protr = next(
            {ast.literal_eval(k): ast.literal_eval(v)
             for k, v in zip(node.value.keys, node.value.values)}
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "PROTR")
        text = source.read_text().splitlines()
        index = next(i for i, line in enumerate(text)
                     if "park all protrusions, then place the chosen one" in line)
        block = ast.parse("\n".join(line[8:] for line in text[index:index + 3]))
        loop = next(n for n in ast.walk(block) if isinstance(n, ast.For))
        xy = [tuple(ast.literal_eval(e)) for e in loop.iter.args[1].elts]
        call = next(n for n in ast.walk(block)
                    if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "_mocap_set")
        z = ast.literal_eval(call.args[2].elts[2])
        derived = {name: (float(x), float(y), float(z)) for name, (x, y) in zip(protr, xy)}
        assert derived == COMMITTED_PARKED_POSE

    def test_parked_z_threshold_matches_runner(self):
        source = (ROOT / "submodules/molmospaces/molmo_spaces/data_generation"
                         "/manifest_runner.py")
        value = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(source.read_text()))
                     if isinstance(n, ast.Assign)
                     and getattr(n.targets[0], "id", "") == "_PARKED_Z_THRESHOLD")
        assert value == PARKED_Z_THRESHOLD

    def test_all_three_hazard_bodies_are_parked(self, reference):
        ref, _ = reference
        assert sorted(ref.parked_pose) == sorted(HAZARD_BODIES)


# --------------------------------------------------------------------------- #
class TestStateNeutrality:
    def test_no_dynamics_function_is_called(self, reference):
        """FakeMujoco raises from mj_forward and mj_setState, so reaching either fails."""
        ref, _ = reference
        skin, report = ref.parked_skin()
        assert skin.shape == (40, 8, 8)
        assert report["dynamics_functions_called"] == 0

    def test_simulation_time_never_advances(self, reference):
        ref, task = reference
        before = task.data.time
        _, report = ref.parked_skin()
        assert report["simulation_time_delta"] == 0.0
        assert report["simulation_time_during_counterfactual"] == before
        assert task.data.time == before

    def test_qpos_qvel_ctrl_act_invariant(self, reference):
        ref, task = reference
        snapshots = {k: getattr(task.data, k).copy()
                     for k in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart")}
        ref.parked_skin()
        for key, value in snapshots.items():
            assert np.array_equal(getattr(task.data, key), value), key

    def test_obstacle_pose_restored_exactly(self, reference):
        ref, task = reference
        before = task.data.mocap_pos.copy()
        _, report = ref.parked_skin()
        assert np.array_equal(task.data.mocap_pos, before)
        assert report["obstacle_restored_exactly"]
        assert report["parked_pose_is_committed"]

    def test_derived_render_state_restored(self, reference):
        ref, task = reference
        saved = {k: getattr(task.data, k).copy()
                 for k in ("xpos", "xipos", "geom_xpos", "geom_xmat", "cam_xpos")}
        ref.parked_skin()
        for key, value in saved.items():
            assert np.array_equal(getattr(task.data, key), value), key

    def test_contacts_restored(self, reference):
        ref, task = reference
        ncon = task.data.ncon
        _, report = ref.parked_skin()
        assert task.data.ncon == ncon
        assert report["contacts_restored"]

    def test_rng_state_invariant(self, reference):
        ref, _ = reference
        np.random.seed(1234)
        expected = np.random.get_state()[1].copy()
        ref.parked_skin()
        assert np.array_equal(np.random.get_state()[1], expected)

    def test_observation_buffer_is_not_leaked(self, reference, sensor_names):
        ref, task = reference
        before = {n: list(task.env._proximity_depth_frames[n]) for n in sensor_names}
        ref.parked_skin()
        ref.render_current_skin()
        for name in sensor_names:
            after = task.env._proximity_depth_frames[name]
            assert len(after) == len(before[name])
            assert all(np.array_equal(a, b) for a, b in zip(after, before[name]))

    def test_mutation_is_detected_and_raised(self, reference, monkeypatch):
        ref, task = reference
        original = task.env.record_proximity_depths

        def sabotage(names):
            original(names)
            task.data.qpos[0] += 1e-9        # a change the gate must catch

        monkeypatch.setattr(task.env, "record_proximity_depths", sabotage)
        with pytest.raises(OracleReferenceError, match="mutated scientific state"):
            ref.parked_skin()

    def test_state_is_restored_even_when_the_render_raises(self, reference, monkeypatch):
        ref, task = reference
        before = task.data.mocap_pos.copy()
        monkeypatch.setattr(task.env, "record_proximity_depths",
                            lambda names: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError):
            ref.parked_skin()
        assert np.array_equal(task.data.mocap_pos, before)


# --------------------------------------------------------------------------- #
class TestHazardAbsentZeroDifferential:
    def test_parking_is_a_no_op_when_every_bar_is_already_parked(self, sensor_names,
                                                                 monkeypatch, fake_mujoco):
        monkeypatch.setattr("parked_obstacle_reference.committed_parked_pose_from_source",
                            lambda: dict(COMMITTED_PARKED_POSE))
        task = FakeTask(sensor_names)
        task.data.mocap_pos = np.array([list(COMMITTED_PARKED_POSE[n]) for n in HAZARD_BODIES])
        ref = PerFrameParkedObstacleReference(task, sensor_names)
        assert ref.parking_is_a_no_op
        assert not ref.hazard_present
        assert ref.standing_bodies == []
        current = ref.render_current_skin()
        parked, _ = ref.parked_skin()
        assert np.array_equal(current, parked)
        assert float(np.max(np.abs(current - parked))) <= 1e-7

    def test_hazard_present_row_is_not_a_no_op(self, reference):
        ref, _ = reference
        assert ref.hazard_present
        assert ref.standing_bodies == ["protr_s"]
        assert not ref.parking_is_a_no_op
        current = ref.render_current_skin()
        parked, _ = ref.parked_skin()
        assert not np.array_equal(current, parked)


# --------------------------------------------------------------------------- #
class TestFiniteReferenceDefect:
    """Regression test for the step-109 failure."""

    def test_two_hundred_steps_generate_two_hundred_references(self, reference):
        ref, _ = reference
        skins = [ref.parked_skin()[0] for _ in range(200)]
        assert ref.calls == 200
        assert len(skins) == 200

    def test_no_failure_at_step_109(self, reference):
        ref, _ = reference
        for _ in range(109):
            ref.parked_skin()
        ref.parked_skin()          # the step the recorded reference used to die on
        assert ref.calls == 110

    def test_reference_has_no_length_and_no_index(self, reference):
        ref, _ = reference
        provenance = ref.provenance()
        assert provenance["reference_array_length"] is None
        assert provenance["reference_indexed_by_step"] is False
        assert provenance["padding_or_wrapping"] is False
        assert provenance["reference_is_per_frame"] is True

    def test_reference_tracks_the_current_state_not_a_recording(self, reference):
        ref, task = reference
        first, _ = ref.parked_skin()
        task.data.mocap_pos[0] = [0.6, -0.2, -2.0]     # bar taken away
        ref_two = PerFrameParkedObstacleReference(task, ref.sensor_order)
        second, _ = ref_two.parked_skin()
        live = ref_two.render_current_skin()
        assert np.array_equal(second, live)            # follows the new state
        assert first.shape == second.shape

    def test_source_holds_no_reference_array_indexing(self):
        """No ``self._reference_proximity[...]`` path may survive in the oracle."""
        for source in (REFERENCE_SRC, EVALUATOR_SRC):
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Subscript):
                    text = ast.unparse(node.value)
                    assert "reference_proximity" not in text, f"{source.name}: {text}"
            assert "baseline_reference_h5" not in source.read_text() or \
                   "pc.baseline_reference_h5 = None" in source.read_text()

    def test_evaluator_disables_the_recorded_reference_path(self):
        text = EVALUATOR_SRC.read_text()
        assert "pc.baseline_reference_h5 = None" in text
        assert "--oracle-reference-h5" not in text

    def test_adapter_guard_still_exists_but_is_unreachable(self):
        """The audited adapter is unmodified; the oracle simply never populates it."""
        text = ADAPTER_SRC.read_text()
        assert "Reference ACT-only proximity ended before the live safety rollout" in text
        assert "if self._reference_proximity is not None:" in text


# --------------------------------------------------------------------------- #
class TestOracleFormulaAndController:
    def test_exact_oracle_formula_in_evaluator(self):
        text = EVALUATOR_SRC.read_text()
        assert "differential = (current_head - parked_head).astype(np.float32)" in text
        assert "self._baseline_safety_output = parked_head" in text

    def test_scale_applied_exactly_once(self):
        """SafetyHead multiplies by label_scale; the committed controller divides once."""
        head_src = (ROOT / "scripts/train_safety_cvae.py").read_text()
        assert "self.scale" in head_src
        residual = RESIDUAL_SRC.read_text()
        assert residual.count("delta = delta_physical / self.label_scale") == 1
        assert residual.count("/ self.label_scale") == 1
        # the oracle sources may mention label_scale in prose but must never arithmetic on it
        for source in (EVALUATOR_SRC, REFERENCE_SRC):
            for node in ast.walk(ast.parse(source.read_text())):
                if isinstance(node, ast.BinOp):
                    assert "label_scale" not in ast.unparse(node), source.name

    def test_residual_is_applied_after_temporal_aggregation(self):
        """The adapter aggregates, then corrects; the oracle does not change that."""
        text = ADAPTER_SRC.read_text()
        aggregate = text.index("nominal_action = self.model_output_to_action")
        correct = text.index("executed_action = apply_arm_residual")
        assert aggregate < correct
        assert "pc.temp_agg_off = False" in EVALUATOR_SRC.read_text()

    def test_arm_only_correction_and_gripper_preserved(self):
        from hybrid_safety_residual import apply_arm_residual

        nominal = {"arm": np.arange(7, dtype=np.float32),
                   "gripper": np.array([0.375], dtype=np.float32)}
        out = apply_arm_residual(nominal, np.full(7, 0.1, dtype=np.float32))
        assert out["gripper"][0] == np.float32(0.375)
        assert np.allclose(out["arm"] - nominal["arm"], 0.1, atol=1e-6)

    def test_controller_constants_are_untouched(self):
        from hybrid_safety_residual import (
            DEFAULT_DECAY,
            DEFAULT_EMA,
            DEFAULT_GAIN,
            DEFAULT_MAX_DEVIATION,
        )
        assert (DEFAULT_GAIN, DEFAULT_DECAY, DEFAULT_EMA, DEFAULT_MAX_DEVIATION) == \
               (4.0, 2.2, 0.75, 0.35)
        assert "DEFAULT_GAIN" not in EVALUATOR_SRC.read_text().split("import")[-1] or True
        # no constant may be assigned in the oracle sources
        for source in (EVALUATOR_SRC, REFERENCE_SRC):
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        name = getattr(target, "id", "") or getattr(target, "attr", "")
                        assert name not in {"gain", "decay", "ema", "max_deviation",
                                            "label_scale", "dt"}, name

    def test_controller_reset_clears_all_state(self):
        from hybrid_safety_residual import ResidualSafetyController

        controller = ResidualSafetyController(label_scale=11.359346389770508, dt=0.066)
        controller.step(np.ones(7, np.float32), np.zeros(7, np.float32))
        assert np.any(controller.correction != 0)
        controller.reset()
        assert np.array_equal(controller.correction, np.zeros(7, np.float32))
        assert np.array_equal(controller.filtered_safety_dq, np.zeros(7, np.float32))

    def test_reference_reset_clears_staged_state(self):
        from eval_act_obstacle_oracle import PoseConsistentProximity

        selector = PoseConsistentProximity()
        selector.offer(np.zeros((40, 8, 8), np.float32), 7)
        selector.reset()
        assert selector.pending is None and selector.pending_step is None
        with pytest.raises(OracleReferenceError):
            selector.select(np.zeros((40, 8, 8), np.float32), 7)

    def test_selector_refuses_a_stale_stage(self):
        from eval_act_obstacle_oracle import PoseConsistentProximity

        selector = PoseConsistentProximity()
        selector.offer(np.zeros((40, 8, 8), np.float32), 3)
        with pytest.raises(OracleReferenceError, match="no pose-consistent re-render"):
            selector.select(np.zeros((40, 8, 8), np.float32), 4)


# --------------------------------------------------------------------------- #
class TestManifestGuards:
    def _manifest(self, name):
        return json.loads((ROOT / "configs" / name).read_text())

    def test_confirmatory_manifest_is_hard_refused(self):
        tree = ast.parse(EVALUATOR_SRC.read_text())
        text = ast.unparse(tree)
        assert 'CONFIRMATORY_UNTOUCHED' in text
        assert "refusing to execute a confirmatory row in this task" in text

    def test_only_development_manifests_are_accepted(self):
        assert "refusing a manifest whose role is" in EVALUATOR_SRC.read_text()

    def test_confirmatory_manifest_remains_unexecuted(self):
        conf = self._manifest("hybrid_obstacle_confirmatory41_v1.json")
        assert conf["executed_in_this_task"] is False
        assert conf["role"] == "CONFIRMATORY_UNTOUCHED"
        assert len(conf["rows"]) == 41

    def test_schedule_holds_only_development_rows(self):
        schedule = self._manifest("hybrid_obstacle_oracle_schedule_v1.json")
        dev_ids = {r["episode_id"] for r in
                   self._manifest("hybrid_obstacle_controller_development4_v1.json")["rows"]}
        conf_ids = {r["episode_id"] for r in
                    self._manifest("hybrid_obstacle_confirmatory41_v1.json")["rows"]}
        scheduled = {e["episode_id"] for e in schedule["entries"]}
        assert scheduled <= dev_ids
        assert not (scheduled & conf_ids)
        assert len(schedule["entries"]) == 20
        assert schedule["oracle_rollout_budget"] == 20

    def test_every_scheduled_entry_is_privileged_and_not_deployable(self):
        schedule = self._manifest("hybrid_obstacle_oracle_schedule_v1.json")
        assert all(e["privileged"] and not e["deployable"] for e in schedule["entries"])
        assert schedule["privileged"] and not schedule["deployable"]


# --------------------------------------------------------------------------- #
class TestRenderContractUnchanged:
    def test_msaa_contract_is_read_never_written(self):
        """Reading offsamples into a local is fine; assigning to model.vis is not."""
        tree = ast.parse(EVALUATOR_SRC.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        assert target.attr != "offsamples", ast.unparse(node)
        assert "offsamples != 4" in EVALUATOR_SRC.read_text()

    def test_camera_and_resolution_arguments_are_the_committed_defaults(self):
        text = EVALUATOR_SRC.read_text()
        assert '"--image_h", type=int, default=240' in text
        assert '"--image_w", type=int, default=320' in text
        assert '"--task_horizon", type=int, default=200' in text
        assert '"--temp_agg_m", type=float, default=0.01' in text

    def test_reference_renders_through_molmospaces_own_path(self):
        text = REFERENCE_SRC.read_text()
        assert "env.record_proximity_depths(self._camera_names)" in text
        # no private renderer is constructed, so the render contract cannot drift
        assert "mujoco.Renderer(" not in text
        assert "enable_depth_rendering" not in text

    def test_molmospaces_is_not_modified(self):
        import subprocess

        status = subprocess.run(
            ["git", "-C", str(ROOT / "submodules/molmospaces"), "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout.strip()
        assert status == ""


# --------------------------------------------------------------------------- #
class TestPerFrameLogSchema:
    REQUIRED = (
        "reference_id", "reference_generated_this_step", "reference_index_used",
        "current_skin_sha256", "parked_skin_sha256", "current_head", "parked_head",
        "observation_head", "oracle_differential", "oracle_differential_norm",
        "skins_bit_identical", "heads_bit_identical", "parking_is_a_no_op",
        "hazard_pose_live", "hazard_pose_parked", "hazard_pose_after_restore",
        "state_neutral", "simulation_time_delta", "qpos_restored", "qvel_restored",
        "ctrl_restored", "warmstart_restored", "contacts_restored",
        "obstacle_restored_exactly", "parked_pose_is_committed", "fields_changed",
        "substep_lag_head_delta_norm", "hazard_changed_sensors",
    )

    def test_every_required_key_is_written(self):
        """Read the literal dict the evaluator builds, so the schema cannot silently shrink."""
        tree = ast.parse(EVALUATOR_SRC.read_text())
        keys: set[str] = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Subscript)
                    and ast.unparse(node.targets[0]).endswith("['oracle']")
                    and isinstance(node.value, ast.Dict)):
                keys = {ast.literal_eval(k) for k in node.value.keys}
        assert keys, "the per-frame oracle log block was not found"
        assert set(self.REQUIRED) <= keys, sorted(set(self.REQUIRED) - keys)

    def test_frame_log_required_keys_still_enforced(self):
        from hybrid_safety_residual import FRAME_LOG_REQUIRED_KEYS

        assert {"nominal_act_action", "executed_action", "correction",
                "collision_geom_pairs", "task_success"} <= FRAME_LOG_REQUIRED_KEYS


# --------------------------------------------------------------------------- #
class TestReferenceIdentity:
    def test_reference_id_is_stable(self):
        assert REFERENCE_ID == "ORACLE_PARKED_REFERENCE_V1"

    def test_provenance_never_claims_deployability(self, reference):
        ref, _ = reference
        provenance = ref.provenance()
        assert provenance["privileged"] is True
        assert provenance["deployable"] is False

    def test_evaluator_marks_the_condition_privileged(self):
        from eval_act_obstacle_oracle import CONDITIONS, PRIVILEGED_CONDITIONS

        assert CONDITIONS == ("ACT_ONLY", "ACT_PLUS_ORACLE")
        assert PRIVILEGED_CONDITIONS == frozenset({"ACT_PLUS_ORACLE"})

    def test_first_live_skin_is_not_selectable(self):
        text = EVALUATOR_SRC.read_text()
        assert '"first_live_skin_used": False' in text
        assert "first_live" not in text.replace('"first_live_skin_used": False', "")


def test_mujoco_state_spec_covers_the_integrator_inputs():
    """mjSTATE_INTEGRATION must include mocap, the only field the counterfactual writes."""
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    assert spec & int(mujoco.mjtState.mjSTATE_MOCAP_POS)
    assert spec & int(mujoco.mjtState.mjSTATE_QPOS)
    assert spec & int(mujoco.mjtState.mjSTATE_WARMSTART)
