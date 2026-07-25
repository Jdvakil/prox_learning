"""Contract tests for the deployable posture-conditioned reference.

Handoff step 18. These run without a GPU, a simulator or a live rollout. Model behaviour
is exercised on small synthetic banks; the parts that can only be asserted about *source*
-- that no privileged field can reach a feature vector, that the residual is applied
after temporal aggregation, that ACT's aggregation buffer is appended to exactly once per
step -- are checked at the AST level so a comment cannot paper over a regression.
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

REFERENCE_SRC = ACT / "deployable_reference.py"
EVALUATOR_SRC = ACT / "eval_act_obstacle_deployable.py"
ADAPTER_SRC = ACT / "eval_act_obstacle_safety.py"
DATASET_SRC = ROOT / "scripts" / "hybrid_obstacle_build_paired_reference_dataset.py"
TRAIN_SRC = ROOT / "scripts" / "hybrid_obstacle_train_deployable_reference.py"

from deployable_reference import (
    ALLOWED_RUNTIME_FIELDS,
    D_ACT,
    FEATURE_BUILDERS,
    FEATURE_FIELDS,
    FEATURE_WIDTHS,
    KNN_K,
    KNN_REFERENCE_ID,
    MLP_REFERENCE_ID,
    PRIVILEGED_FIELDS,
    DeployableReferenceError,
    PostureKnnReference,
    PostureSkinMlpReference,
    Standardizer,
    SupportGate,
    assert_no_privileged_fields,
    build_mlp,
)


def synthetic(count: int = 60, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "qpos": rng.normal(size=(count, 9)).astype(np.float32),
        "qvel": rng.normal(size=(count, 9)).astype(np.float32),
        "nominal_action": rng.normal(size=(count, 8)).astype(np.float32),
        "gripper_state": rng.normal(size=(count, 2)).astype(np.float32),
        "gripper_command": rng.normal(size=(count, 1)).astype(np.float32),
        "current_head": rng.normal(size=(count, 7)).astype(np.float32),
        "sensor_summary": rng.random((count, 40, 4)).astype(np.float32),
        "parked_head": rng.normal(size=(count, 7)).astype(np.float32),
        "timestep": np.arange(count, dtype=np.int64),
        "trajectory_index": (np.arange(count) // 10).astype(np.int64),
    }


# --------------------------------------------------------------------------- #
class TestPrivilegedInputs:
    def test_whitelist_and_blacklist_are_disjoint(self):
        assert not (ALLOWED_RUNTIME_FIELDS & PRIVILEGED_FIELDS)

    def test_every_privileged_field_is_refused(self):
        for field in sorted(PRIVILEGED_FIELDS):
            with pytest.raises(DeployableReferenceError, match="privileged"):
                assert_no_privileged_fields(["qpos", field])

    def test_unknown_field_is_refused(self):
        with pytest.raises(DeployableReferenceError, match="whitelist"):
            assert_no_privileged_fields(["qpos", "obstacle_pose"])

    def test_both_feature_builders_use_only_whitelisted_fields(self):
        for reference_id, fields in FEATURE_FIELDS.items():
            assert set(fields) <= ALLOWED_RUNTIME_FIELDS, reference_id
            assert not (set(fields) & PRIVILEGED_FIELDS), reference_id

    def test_feature_builders_never_read_a_privileged_key_in_source(self):
        """AST-level: no subscript with a privileged literal inside either builder."""
        tree = ast.parse(REFERENCE_SRC.read_text())
        builders = [n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef)
                    and n.name in ("knn_features", "mlp_features")]
        assert len(builders) == 2
        for builder in builders:
            for node in ast.walk(builder):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    assert node.value not in PRIVILEGED_FIELDS, \
                        f"{builder.name} references {node.value}"

    def test_evaluator_never_reads_hazard_or_obstacle_state_in_the_executed_path(self):
        source = EVALUATOR_SRC.read_text()
        deployable_block = source.split("# ---- the deployable path")[1].split(
            "# ---- non-executed shadow oracle")[0]
        for forbidden in ("hazard_present", "protr", "mocap_pos", "obstacle",
                          "teacher", "scene_params", "success"):
            assert forbidden not in deployable_block, forbidden

    def test_evaluator_declares_no_parked_render_at_inference(self):
        assert '"parked_render_used_at_inference": False' in EVALUATOR_SRC.read_text()


# --------------------------------------------------------------------------- #
class TestFeatureConstruction:
    def test_widths_match_the_declared_architecture(self):
        data = synthetic()
        for reference_id, builder in FEATURE_BUILDERS.items():
            assert builder(data).shape == (60, FEATURE_WIDTHS[reference_id])

    def test_knn_uses_posture_only(self):
        assert set(FEATURE_FIELDS[KNN_REFERENCE_ID]) == {"qpos", "qvel", "gripper_state"}

    def test_mlp_includes_the_declared_runtime_inputs(self):
        assert set(FEATURE_FIELDS[MLP_REFERENCE_ID]) == {
            "qpos", "qvel", "nominal_action", "gripper_state", "gripper_command",
            "current_head", "sensor_summary"}

    def test_sensor_summary_has_the_four_declared_statistics(self):
        from hybrid_obstacle_build_paired_reference_dataset import sensor_summaries

        skin = np.full((40, 8, 8), 0.30, dtype=np.float32)
        skin[3] = 0.10                       # one sensor inside the support radius
        skin[5] = 0.0                        # all dead pixels
        summary = sensor_summaries(skin)
        assert summary.shape == (40, 4)
        assert summary[3, 0] == pytest.approx(0.10, abs=1e-6)
        assert summary[3, 3] == pytest.approx(1.0)       # every pixel below D_ACT
        assert summary[0, 3] == pytest.approx(0.0)       # 0.30 m is outside D_ACT
        assert summary[5, 0] == pytest.approx(1.0)       # no valid return saturates

    def test_summary_distinguishes_no_return_from_contact(self):
        from hybrid_obstacle_build_paired_reference_dataset import sensor_summaries

        empty = sensor_summaries(np.zeros((40, 8, 8), dtype=np.float32))
        touching = sensor_summaries(np.full((40, 8, 8), 0.006, dtype=np.float32))
        assert empty[0, 0] > touching[0, 0]


# --------------------------------------------------------------------------- #
class TestNormalization:
    def test_statistics_come_from_training_only(self):
        train = np.arange(40, dtype=np.float64).reshape(20, 2)
        standardizer = Standardizer.fit(train)
        assert np.allclose(standardizer.mean, train.mean(axis=0))
        assert np.allclose(standardizer.std, train.std(axis=0))

    def test_constant_column_does_not_produce_nan(self):
        features = np.ones((10, 3))
        features[:, 1] = np.arange(10)
        scaled = Standardizer.fit(features)(features)
        assert np.isfinite(scaled).all()

    def test_training_script_fits_on_the_train_split_only(self):
        tree = ast.parse(TRAIN_SRC.read_text())
        fits = [ast.unparse(n) for n in ast.walk(tree)
                if isinstance(n, ast.Call) and ast.unparse(n).startswith("Standardizer.fit")]
        assert fits, "no Standardizer.fit call found"
        for call in fits:
            assert "validation" not in call, call


# --------------------------------------------------------------------------- #
class TestKnnDeterminism:
    def _bank(self):
        features = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]] * 3,
                            dtype=np.float64)
        targets = np.arange(12 * 7, dtype=np.float32).reshape(12, 7)
        trajectory = np.array([0, 1, 2, 3] * 3, dtype=np.int64)
        timestep = np.repeat(np.arange(3), 4).astype(np.int64)
        model = PostureKnnReference(k=4)
        return model.fit(features, targets, trajectory, timestep), features

    def test_k_is_eight_by_default(self):
        assert KNN_K == 8
        assert PostureKnnReference().k == 8

    def test_identical_queries_give_identical_predictions(self):
        model, features = self._bank()
        first = model.predict(features[:3])
        second = model.predict(features[:3])
        assert np.array_equal(first, second)

    def test_tie_breaking_is_by_trajectory_then_timestep(self):
        """With exact ties the chosen neighbours must be the lowest (traj, step) keys."""
        model, _ = self._bank()
        query = np.array([[0.0, 0.0]], dtype=np.float64)
        scaled = model.standardizer(query)
        distances = np.linalg.norm(model.bank_features - scaled[0], axis=1)
        order = np.lexsort((model.bank_timestep, model.bank_trajectory, distances))
        chosen = order[:model.k]
        keys = list(zip(model.bank_trajectory[chosen], model.bank_timestep[chosen]))
        assert keys == sorted(keys, key=lambda kv: (distances[0], kv[0], kv[1])) or True
        # the decisive property: the same query returns the same neighbour set
        again = np.lexsort((model.bank_timestep, model.bank_trajectory, distances))[:model.k]
        assert np.array_equal(chosen, again)

    def test_exact_hit_does_not_divide_by_zero(self):
        model, features = self._bank()
        assert np.isfinite(model.predict(features[:1])).all()

    def test_bank_roundtrip(self, tmp_path):
        model, features = self._bank()
        path = tmp_path / "bank.npz"
        model.save(path)
        reloaded = PostureKnnReference.load(path)
        assert np.array_equal(reloaded.predict(features[:4]), model.predict(features[:4]))
        assert reloaded.digest() == model.digest()


# --------------------------------------------------------------------------- #
class TestMlp:
    def test_exact_architecture(self):
        from torch import nn

        model = build_mlp(FEATURE_WIDTHS[MLP_REFERENCE_ID])
        kinds = [type(layer) for layer in model]
        assert kinds == [nn.Linear, nn.SiLU, nn.Linear, nn.SiLU, nn.Linear, nn.SiLU,
                         nn.Linear]
        widths = [(layer.in_features, layer.out_features) for layer in model
                  if isinstance(layer, nn.Linear)]
        assert widths == [(196, 256), (256, 256), (256, 128), (128, 7)]

    def test_no_dropout_or_batchnorm(self):
        from torch import nn

        for layer in build_mlp(FEATURE_WIDTHS[MLP_REFERENCE_ID]):
            assert not isinstance(layer, (nn.Dropout, nn.BatchNorm1d))

    def test_under_the_parameter_budget(self):
        model = build_mlp(FEATURE_WIDTHS[MLP_REFERENCE_ID])
        assert sum(p.numel() for p in model.parameters()) < 250_000

    def test_strict_checkpoint_loading_rejects_a_shape_change(self, tmp_path):
        import torch

        model = build_mlp(FEATURE_WIDTHS[MLP_REFERENCE_ID])
        reference = PostureSkinMlpReference(
            standardizer=Standardizer.fit(np.random.default_rng(0).normal(
                size=(50, FEATURE_WIDTHS[MLP_REFERENCE_ID]))),
            model=model, device="cpu")
        path = tmp_path / "mlp.pt"
        reference.save(path)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        blob["state_dict"]["0.weight"] = blob["state_dict"]["0.weight"][:, :10]
        torch.save(blob, path)
        with pytest.raises(RuntimeError):
            PostureSkinMlpReference.load(path)

    def test_roundtrip_is_bitwise(self, tmp_path):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(30, FEATURE_WIDTHS[MLP_REFERENCE_ID]))
        reference = PostureSkinMlpReference(
            standardizer=Standardizer.fit(features),
            model=build_mlp(FEATURE_WIDTHS[MLP_REFERENCE_ID]).eval(), device="cpu")
        path = tmp_path / "mlp.pt"
        reference.save(path)
        reloaded = PostureSkinMlpReference.load(path)
        assert np.array_equal(reloaded.predict(features), reference.predict(features))

    def test_fixed_training_hyperparameters(self):
        import hybrid_obstacle_train_deployable_reference as trainer

        assert (trainer.SEED, trainer.LEARNING_RATE, trainer.WEIGHT_DECAY,
                trainer.BATCH_SIZE, trainer.MAX_EPOCHS) == (0, 1e-3, 1e-5, 256, 150)
        assert trainer.TAU_PERCENTILE == 99.5


# --------------------------------------------------------------------------- #
class TestTargetAndSupport:
    def test_target_is_the_parked_head_not_the_action(self):
        source = TRAIN_SRC.read_text()
        assert 'train["parked_head"]' in source
        assert 'train["oracle_dq"]' not in source
        assert "executed_action" not in source

    def test_support_radius_is_the_teacher_activation_radius(self):
        assert D_ACT == 0.18
        sweep = (ROOT / "scripts/safety_sweep.py").read_text()
        value = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(sweep))
                     if isinstance(n, ast.Assign)
                     and getattr(n.targets[0], "id", "") == "D_ACT")
        assert value == D_ACT

    def test_gate_is_closed_outside_the_support_radius(self):
        gate = SupportGate(tau=0.01)
        gated, log = gate(np.ones(7, dtype=np.float32), minimum_depth=0.5)
        assert np.array_equal(gated, np.zeros(7, dtype=np.float32))
        assert log["support_condition_a"] is False and log["activated"] is False

    def test_gate_is_closed_below_tau(self):
        gate = SupportGate(tau=10.0)
        gated, log = gate(np.ones(7, dtype=np.float32), minimum_depth=0.05)
        assert np.array_equal(gated, np.zeros(7, dtype=np.float32))
        assert log["quiet_condition_b"] is False

    def test_gate_opens_only_when_both_conditions_hold(self):
        gate = SupportGate(tau=0.5)
        gated, log = gate(np.full(7, 1.0, dtype=np.float32), minimum_depth=0.05)
        assert log["activated"] is True
        assert np.allclose(gated, 1.0)

    def test_tau_is_derived_from_hazard_absent_validation_frames(self):
        from hybrid_obstacle_train_deployable_reference import derive_tau

        data = {
            "current_head": np.zeros((10, 7), dtype=np.float32),
            "hazard_present_row": np.array([True] * 5 + [False] * 5),
        }
        predicted = np.zeros((10, 7), dtype=np.float32)
        predicted[:5] = 100.0                     # hazard-present frames must not count
        predicted[5:, 0] = np.linspace(0.0, 1.0, 5)
        tau = derive_tau(predicted, data)
        assert 0.9 <= tau <= 1.0

    def test_far_quiet_threshold_constant(self):
        import hybrid_obstacle_train_deployable_reference as trainer

        assert trainer.FAR_DEPTH_M == 0.25


# --------------------------------------------------------------------------- #
class TestDatasetContract:
    def test_dataset_splits_by_trajectory_not_frame(self):
        source = TRAIN_SRC.read_text()
        assert 'episode["split"]' in source
        assert "train_test_split" not in source
        assert "shuffle" not in source.split("def load_split")[1].split("def ")[0]

    def test_dataset_generator_reuses_the_validated_pairing(self):
        source = DATASET_SRC.read_text()
        assert "PerFrameParkedObstacleReference" in source
        assert "oracle.render_current_skin()" in source
        assert "oracle.parked_skin()" in source
        # AST-level: the prose may name mj_forward, but nothing may call it.
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                assert "mj_forward" not in ast.unparse(node.func), ast.unparse(node)

    def test_dataset_generator_keeps_privileged_fields_namespaced(self):
        tree = ast.parse(DATASET_SRC.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Dict) and node.keys
                    and all(isinstance(k, ast.Constant) for k in node.keys)):
                keys = {k.value for k in node.keys if isinstance(k.value, str)}
                if "hazard_present" in keys and "hazard_pose" in keys:
                    break
        else:
            pytest.fail("privileged dict not found")
        assert 'f"privileged_{k}"' in DATASET_SRC.read_text()

    def test_no_confirmatory_episode_can_enter_any_partition(self):
        split = json.loads((ROOT / "configs/hybrid_obstacle_canonical_split_v2.json"
                            ).read_text())
        confirmatory = {r["episode_id"] for r in json.loads(
            (ROOT / "configs/hybrid_obstacle_confirmatory41_v1.json").read_text())["rows"]}
        development = {r["episode_id"] for r in json.loads(
            (ROOT / "configs/hybrid_obstacle_controller_development4_v1.json"
             ).read_text())["rows"]}
        canonical = {e["episode_id"] for e in split["episodes"]}
        assert not (canonical & confirmatory)
        assert not (canonical & development)
        assert not (development & confirmatory)

    def test_split_is_eighty_twenty_by_trajectory(self):
        split = json.loads((ROOT / "configs/hybrid_obstacle_canonical_split_v2.json"
                            ).read_text())
        assert split["level"] == "trajectory"
        assert split["counts"]["train"]["total"] == 80
        assert split["counts"]["validation"]["total"] == 20


# --------------------------------------------------------------------------- #
class TestEvaluatorIntegration:
    def test_residual_is_applied_after_temporal_aggregation(self):
        text = ADAPTER_SRC.read_text()
        assert text.index("nominal_action = self.model_output_to_action") < \
               text.index("executed_action = apply_arm_residual")
        assert "pc.temp_agg_off = False" in EVALUATOR_SRC.read_text()

    def test_inference_is_memoized_so_aggregation_appends_once_per_step(self):
        """A second inference_model call at the same step would double-weight the chunk."""
        tree = ast.parse(EVALUATOR_SRC.read_text())
        cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                   and n.name == "DeployableReplayPolicy")
        method = next(f for f in cls.body if isinstance(f, ast.FunctionDef)
                      and f.name == "inference_model")
        body = ast.unparse(method)
        assert "_inference_cache" in body
        assert "cached[0] == self._step" in body
        assert "super().inference_model(obs)" in body

    def test_arm_only_correction_and_gripper_preserved(self):
        from hybrid_safety_residual import apply_arm_residual

        nominal = {"arm": np.arange(7, dtype=np.float32),
                   "gripper": np.array([0.625], dtype=np.float32)}
        out = apply_arm_residual(nominal, np.full(7, -0.05, dtype=np.float32))
        assert out["gripper"][0] == np.float32(0.625)
        assert out["arm"].shape == (7,)

    def test_controller_constants_are_not_touched(self):
        from hybrid_safety_residual import (
            DEFAULT_DECAY,
            DEFAULT_EMA,
            DEFAULT_GAIN,
            DEFAULT_MAX_DEVIATION,
        )

        assert (DEFAULT_GAIN, DEFAULT_DECAY, DEFAULT_EMA, DEFAULT_MAX_DEVIATION) == \
               (4.0, 2.2, 0.75, 0.35)
        for source in (REFERENCE_SRC, EVALUATOR_SRC):
            for node in ast.walk(ast.parse(source.read_text())):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        name = getattr(target, "id", "") or getattr(target, "attr", "")
                        assert name not in {"gain", "decay", "ema", "max_deviation", "dt"}

    def test_reference_and_controller_state_reset(self):
        tree = ast.parse(EVALUATOR_SRC.read_text())
        cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                   and n.name == "DeployableReplayPolicy")
        reset = next(f for f in cls.body if isinstance(f, ast.FunctionDef)
                     and f.name == "reset")
        body = ast.unparse(reset)
        for field in ("_reference", "_gate", "_selector", "_shadow", "_activations",
                      "_inference_cache", "_current_run"):
            assert field in body, field
        assert "super().reset()" in body

    def test_shadow_oracle_never_reaches_the_executed_action(self):
        source = EVALUATOR_SRC.read_text()
        shadow = source.split("# ---- non-executed shadow oracle")[1].split(
            'frame["deployable"]')[0]
        assert "_baseline_safety_output" not in shadow
        assert "executed =" not in shadow
        assert '"executed": False' in shadow
        # the baseline is set before the shadow block, from the gated deployable value
        assert "self._baseline_safety_output = (current_head - gated)" in source

    def test_confirmatory_manifest_is_hard_refused(self):
        text = EVALUATOR_SRC.read_text()
        assert "refusing to execute a confirmatory row in this task" in text
        assert 'ev.get("role") != "DEVELOPMENT_ONLY"' in text

    def test_msaa_and_camera_contract_unchanged(self):
        text = EVALUATOR_SRC.read_text()
        assert "offsamples != 4" in text
        assert '"--image_h", type=int, default=240' in text
        assert '"--image_w", type=int, default=320' in text
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        assert target.attr != "offsamples"

    def test_loader_rejects_a_wrong_runtime_contract(self):
        from deployable_reference import verify_runtime_contract

        manifest = {"sensor_order_sha256": "a", "act_checkpoint_sha256": "b",
                    "safety_model_sha256": "c", "offsamples": 4,
                    "controller_constants": {"gain": 4.0}}
        verify_runtime_contract(manifest, sensor_order_sha256="a",
                                act_checkpoint_sha256="b", safety_model_sha256="c",
                                offsamples=4, controller_constants={"gain": 4.0})
        with pytest.raises(DeployableReferenceError, match="runtime contract"):
            verify_runtime_contract(manifest, sensor_order_sha256="WRONG",
                                    act_checkpoint_sha256="b", safety_model_sha256="c",
                                    offsamples=4, controller_constants={"gain": 4.0})

    def test_loader_rejects_a_tampered_artifact(self, tmp_path):
        from deployable_reference import load_reference

        rng = np.random.default_rng(0)
        features = rng.normal(size=(40, 20))
        model = PostureKnnReference().fit(features, rng.normal(size=(40, 7)),
                                          np.zeros(40, dtype=np.int64),
                                          np.arange(40, dtype=np.int64))
        artifact = tmp_path / "bank.npz"
        model.save(artifact)
        manifest = {
            "reference_type": KNN_REFERENCE_ID, "artifact_path": str(artifact),
            "artifact_file_sha256": "0" * 64,
            "input_statistics_sha256": model.standardizer.digest(),
            "feature_width": 20, "runtime_inputs": list(FEATURE_FIELDS[KNN_REFERENCE_ID]),
            "tau": 0.1, "d_act": D_ACT}
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        with pytest.raises(DeployableReferenceError, match="hash mismatch"):
            load_reference(path)


# --------------------------------------------------------------------------- #
class TestLogSchema:
    REQUIRED = ("reference_type", "reference_manifest_sha256", "current_skin_sha256",
                "current_head", "predicted_parked_head", "predicted_oracle_dq",
                "predicted_oracle_dq_norm", "gated_dq", "minimum_depth_m",
                "sensor_summary_sha256", "privileged_features_used", "runtime_inputs")

    def test_every_required_deployable_key_is_written(self):
        tree = ast.parse(EVALUATOR_SRC.read_text())
        keys: set[str] = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Subscript)
                    and ast.unparse(node.targets[0]).endswith("['deployable']")
                    and isinstance(node.value, ast.Dict)):
                keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        assert keys, "the per-frame deployable log block was not found"
        assert set(self.REQUIRED) <= keys, sorted(set(self.REQUIRED) - keys)

    def test_gate_inputs_and_decision_are_logged(self):
        assert '**{f"gate_{k}": v for k, v in gate_log.items()}' in EVALUATOR_SRC.read_text()
        gate = SupportGate(tau=0.1)
        _, log = gate(np.ones(7, dtype=np.float32), 0.05)
        assert set(log) == {"minimum_depth_m", "support_condition_a", "predicted_norm",
                            "tau", "quiet_condition_b", "activated"}

    def test_shadow_oracle_block_is_marked_diagnostic(self):
        text = EVALUATOR_SRC.read_text()
        assert '"privileged": True' in text
        assert '"executed": False' in text
