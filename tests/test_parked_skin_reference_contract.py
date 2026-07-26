"""Contract tests for CAUSAL_PARKED_SKIN_REFERENCE_V1 and the parked-skin data contract.

Handoff step 23. No GPU, no simulator, no rollout. The model is exercised on synthetic
fields; the data-contract properties are asserted against the committed artifacts.

The model in this task was implemented and verified but **never trained**: the paired
dataset stores no parked 40x8x8 field, so its training target does not exist. These tests
therefore cover architecture, the physical counterfactual, the closeness transform, the
causal-history contract, gating, loading and the data audit -- everything that does not
require the missing target.
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

MODEL_SRC = ACT / "parked_skin_reference.py"
ADAPTER_SRC = ACT / "eval_act_obstacle_safety.py"
AUDIT_SRC = ROOT / "scripts/hybrid_obstacle_parked_skin_data_audit.py"

from parked_skin_reference import (
    CAUSAL_FRAMES,
    CONTEXT_WIDTH,
    D_MAX,
    DEAD_PIXEL_BELOW_M,
    PARAMETER_BUDGET,
    PRIVILEGED_FIELDS,
    REFERENCE_ID,
    RUNTIME_FIELDS,
    ActivityGate,
    CausalHistory,
    ParkedSkinContractError,
    assert_no_privileged_fields,
    build_model,
    closeness_to_depth,
    depth_to_closeness,
    parameter_count,
    parked_closeness_from_removable,
    verify_runtime_contract,
    violates_physical_constraint,
)

torch = pytest.importorskip("torch")


def config(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text())


# --------------------------------------------------------------------------- #
class TestClosenessTransform:
    def test_canonical_formula(self):
        depth = np.array([0.5, 0.25, 0.125, 0.0], dtype=np.float32)
        assert np.allclose(depth_to_closeness(depth[:3]), [0.0, 0.5, 0.75], atol=1e-6)

    def test_dead_pixels_map_to_zero_closeness(self):
        """A sub-5 mm reading is 'no return', not a surface at contact."""
        assert depth_to_closeness(np.array([0.004], dtype=np.float32))[0] == 0.0
        assert depth_to_closeness(np.array([0.0], dtype=np.float32))[0] == 0.0
        assert depth_to_closeness(np.array([0.005], dtype=np.float32))[0] > 0.98

    def test_clipped_to_unit_interval(self):
        wide = np.array([-1.0, 0.0, 0.006, 5.0], dtype=np.float32)
        value = depth_to_closeness(wide)
        assert value.min() >= 0.0 and value.max() <= 1.0

    def test_zero_closeness_maps_to_far_depth(self):
        assert float(closeness_to_depth(np.array([0.0]))[0]) == pytest.approx(D_MAX)

    def test_roundtrip_on_valid_returns(self):
        depth = np.array([0.01, 0.1, 0.3, 0.49], dtype=np.float32)
        assert np.allclose(closeness_to_depth(depth_to_closeness(depth)), depth, atol=1e-6)

    def test_d_max_matches_the_frozen_head_metadata(self):
        meta = json.loads((ROOT / "assets/safety/cvae_v3/meta.json").read_text())
        assert float(meta["d_max_input"]) == D_MAX

    def test_dead_pixel_threshold_matches_the_repository_rule(self):
        residual = (ACT / "hybrid_safety_residual.py").read_text()
        assert "DEFAULT_DEAD_PIXEL_BELOW_M = 0.005" in residual
        assert DEAD_PIXEL_BELOW_M == 0.005


# --------------------------------------------------------------------------- #
class TestPhysicalCounterfactual:
    def test_parked_never_exceeds_current(self):
        rng = np.random.default_rng(0)
        current = rng.random((40, 8, 8)).astype(np.float32)
        fraction = rng.random((40, 8, 8)).astype(np.float32)
        parked = parked_closeness_from_removable(current, fraction)
        assert not violates_physical_constraint(current, parked).any()
        assert (parked >= 0).all() and (parked <= 1).all()

    def test_cannot_invent_a_return_where_there_is_none(self):
        current = np.zeros((40, 8, 8), dtype=np.float32)
        parked = parked_closeness_from_removable(current, np.ones((40, 8, 8), np.float32))
        assert np.array_equal(parked, current)

    def test_zero_fraction_preserves_unchanged_pixels_exactly(self):
        rng = np.random.default_rng(1)
        current = rng.random((40, 8, 8)).astype(np.float32)
        parked = parked_closeness_from_removable(current, np.zeros_like(current))
        assert np.array_equal(parked, current)

    def test_constraint_detector_flags_a_mispairing(self):
        current = np.full((2, 2), 0.2, dtype=np.float32)
        parked = np.full((2, 2), 0.5, dtype=np.float32)
        assert violates_physical_constraint(current, parked).all()

    def test_network_output_is_bounded_by_construction(self):
        model = build_model().eval()
        history = torch.rand(4, CAUSAL_FRAMES, 40, 8, 8)
        state = torch.randn(4, CONTEXT_WIDTH)
        with torch.no_grad():
            parked, _, _ = model(history, state)
        current = history[:, -1]
        assert bool((parked <= current + 1e-6).all())
        assert bool((parked >= 0).all()) and bool((parked <= 1).all())

    def test_bound_holds_for_extreme_logits(self):
        """Even a saturated network cannot break the bound."""
        model = build_model().eval()
        with torch.no_grad():
            model.change_head.bias.fill_(50.0)
            history = torch.rand(2, CAUSAL_FRAMES, 40, 8, 8)
            parked, _, _ = model(history, torch.randn(2, CONTEXT_WIDTH))
        assert bool((parked <= history[:, -1] + 1e-6).all())
        assert bool((parked >= 0).all())


# --------------------------------------------------------------------------- #
class TestArchitecture:
    def test_parameter_budget(self):
        count = parameter_count(build_model())
        assert count < PARAMETER_BUDGET
        assert count == 331_713

    def test_declared_shapes(self):
        from torch import nn

        model = build_model()
        assert isinstance(model.sensor_encoder, nn.Linear)
        assert (model.sensor_encoder.in_features,
                model.sensor_encoder.out_features) == (256, 128)
        assert tuple(model.sensor_embedding.shape) == (40, 128)
        assert len(model.encoder.layers) == 2
        layer = model.encoder.layers[0]
        assert layer.self_attn.num_heads == 4
        assert layer.linear1.out_features == 256
        assert layer.norm_first is True
        assert float(layer.dropout.p) == 0.0
        assert (model.change_head.in_features, model.change_head.out_features) == (128, 64)
        assert (model.activity_head.in_features, model.activity_head.out_features) == (128, 1)

    def test_context_width_matches_the_declared_inputs(self):
        assert CONTEXT_WIDTH == 9 + 9 + 8 + 2 + 1

    def test_two_outputs(self):
        model = build_model().eval()
        with torch.no_grad():
            parked, logits, activity = model(torch.rand(3, CAUSAL_FRAMES, 40, 8, 8),
                                             torch.randn(3, CONTEXT_WIDTH))
        assert parked.shape == (3, 40, 8, 8)
        assert logits.shape == (3, 40, 8, 8)
        assert activity.shape == (3,)

    def test_sensor_tokens_are_per_sensor(self):
        """Changing one sensor's history must not be equivalent to changing another's."""
        model = build_model().eval()
        base = torch.zeros(1, CAUSAL_FRAMES, 40, 8, 8)
        first, second = base.clone(), base.clone()
        first[0, :, 3] = 0.7
        second[0, :, 11] = 0.7
        with torch.no_grad():
            a, _, _ = model(first, torch.zeros(1, CONTEXT_WIDTH))
            b, _, _ = model(second, torch.zeros(1, CONTEXT_WIDTH))
        assert not torch.allclose(a, b)


# --------------------------------------------------------------------------- #
class TestCausalHistory:
    def test_pads_at_episode_start_and_never_returns_a_future_frame(self):
        history = CausalHistory()
        history.push(np.full((40, 8, 8), 0.4, np.float32), 0)
        stack = history.stack()
        assert stack.shape == (CAUSAL_FRAMES, 40, 8, 8)
        assert np.array_equal(stack[:3], np.zeros((3, 40, 8, 8), np.float32))
        assert np.allclose(stack[-1], 0.4)

    def test_latest_slot_is_the_decision_state(self):
        history = CausalHistory()
        for step in range(6):
            history.push(np.full((40, 8, 8), step / 10.0, np.float32), step)
        stack = history.stack()
        assert np.allclose(stack[-1], 0.5)
        assert history.latest_step == 5
        assert np.allclose(stack[0], 0.2)          # a shifted window, oldest first

    def test_refuses_a_non_monotonic_step(self):
        history = CausalHistory()
        history.push(np.zeros((40, 8, 8), np.float32), 5)
        with pytest.raises(ParkedSkinContractError, match="monotonically"):
            history.push(np.zeros((40, 8, 8), np.float32), 4)
        with pytest.raises(ParkedSkinContractError):
            history.push(np.zeros((40, 8, 8), np.float32), 5)

    def test_reset_clears_the_buffer(self):
        history = CausalHistory()
        history.push(np.ones((40, 8, 8), np.float32), 0)
        history.reset()
        with pytest.raises(ParkedSkinContractError):
            history.stack()
        history.push(np.zeros((40, 8, 8), np.float32), 0)      # step 0 valid again

    def test_window_never_exceeds_four(self):
        history = CausalHistory()
        for step in range(20):
            history.push(np.full((40, 8, 8), step, np.float32), step)
        assert len(history.frames) == CAUSAL_FRAMES
        assert history.stack().shape[0] == CAUSAL_FRAMES


# --------------------------------------------------------------------------- #
class TestPrivilegedInputs:
    def test_whitelist_and_blacklist_are_disjoint(self):
        assert not (set(RUNTIME_FIELDS) & PRIVILEGED_FIELDS)

    def test_every_privileged_field_is_refused(self):
        for field in sorted(PRIVILEGED_FIELDS):
            with pytest.raises(ParkedSkinContractError, match="privileged"):
                assert_no_privileged_fields(["qpos", field])

    def test_runtime_whitelist_accepted(self):
        assert_no_privileged_fields(RUNTIME_FIELDS)

    def test_unknown_field_refused(self):
        with pytest.raises(ParkedSkinContractError, match="whitelist"):
            assert_no_privileged_fields(["qpos", "rgb_image"])

    def test_no_rgb_input(self):
        assert not any("rgb" in f or "camera" in f for f in RUNTIME_FIELDS)

    def test_model_forward_takes_only_history_and_state(self):
        tree = ast.parse(MODEL_SRC.read_text())
        forward = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == "forward")
        names = [a.arg for a in forward.args.args]
        assert names == ["self", "causal_closeness", "state"]


# --------------------------------------------------------------------------- #
class TestActivityGate:
    def test_gates_on_activity_probability_not_predicted_norm(self):
        gate = ActivityGate(threshold=0.5, rho_max=10.0)
        big = np.full(7, 100.0, dtype=np.float32)
        gated, log = gate(big, activity_probability=0.1)
        assert np.array_equal(gated, np.zeros(7, dtype=np.float32))
        assert log["activated"] is False
        assert log["gated_on_predicted_norm_alone"] is False

    def test_small_vector_still_activates_when_confident(self):
        gate = ActivityGate(threshold=0.5, rho_max=10.0)
        gated, log = gate(np.full(7, 1e-3, dtype=np.float32), activity_probability=0.9)
        assert log["activated"] is True
        assert np.allclose(gated, 1e-3)

    def test_caps_norm_and_preserves_direction(self):
        gate = ActivityGate(threshold=0.5, rho_max=2.0)
        vector = np.array([6.0, 8.0, 0, 0, 0, 0, 0], dtype=np.float32)
        gated, log = gate(vector, activity_probability=1.0)
        assert log["capped"] is True
        assert float(np.linalg.norm(gated)) == pytest.approx(2.0, abs=1e-5)
        assert np.allclose(gated / np.linalg.norm(gated),
                           vector / np.linalg.norm(vector), atol=1e-6)

    def test_log_schema(self):
        gate = ActivityGate(threshold=0.4, rho_max=3.0)
        _, log = gate(np.ones(7, dtype=np.float32), 0.8)
        assert set(log) == {"activity_probability", "activity_threshold", "activated",
                            "predicted_norm", "rho_max", "capped", "executed_norm",
                            "gated_on_predicted_norm_alone"}

    def test_previous_tau_is_not_reused(self):
        previous = json.loads((ROOT / "diagnostics_output"
                               / "hybrid_obstacle_deployable_reference"
                               / "deployment_manifest.json").read_text())
        assert str(previous["tau"]) not in MODEL_SRC.read_text()


# --------------------------------------------------------------------------- #
class TestLoaderContract:
    def _manifest(self, tmp_path, **overrides):
        artifact = tmp_path / "model.pt"
        torch.save({"state_dict": build_model().state_dict()}, artifact)
        import hashlib as _h

        manifest = {
            "reference_type": REFERENCE_ID,
            "artifact_path": str(artifact),
            "artifact_file_sha256": _h.sha256(artifact.read_bytes()).hexdigest(),
            "runtime_inputs": list(RUNTIME_FIELDS),
            "causal_frames": CAUSAL_FRAMES,
            "activity_threshold": 0.5, "rho_max": 2.0,
        }
        manifest.update(overrides)
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_loads_a_well_formed_manifest(self, tmp_path):
        from parked_skin_reference import load_reference

        model, gate, manifest = load_reference(self._manifest(tmp_path))
        assert manifest["reference_type"] == REFERENCE_ID
        assert gate.threshold == 0.5 and gate.rho_max == 2.0
        assert parameter_count(model) == 331_713

    def test_rejects_a_tampered_artifact(self, tmp_path):
        from parked_skin_reference import load_reference

        path = self._manifest(tmp_path, artifact_file_sha256="0" * 64)
        with pytest.raises(ParkedSkinContractError, match="hash mismatch"):
            load_reference(path)

    def test_rejects_a_non_causal_history_declaration(self, tmp_path):
        from parked_skin_reference import load_reference

        path = self._manifest(tmp_path, causal_frames=1)
        with pytest.raises(ParkedSkinContractError, match="four-frame"):
            load_reference(path)

    def test_rejects_a_privileged_runtime_input(self, tmp_path):
        from parked_skin_reference import load_reference

        path = self._manifest(tmp_path, runtime_inputs=["qpos", "parked_skin"])
        with pytest.raises(ParkedSkinContractError, match="privileged"):
            load_reference(path)

    def test_rejects_a_wrong_runtime_contract(self):
        manifest = {"sensor_order_sha256": "a", "act_checkpoint_sha256": "b",
                    "safety_model_sha256": "c", "offsamples": 4,
                    "controller_constants": {"gain": 4.0}}
        verify_runtime_contract(manifest, sensor_order_sha256="a",
                                act_checkpoint_sha256="b", safety_model_sha256="c",
                                offsamples=4, controller_constants={"gain": 4.0})
        for field, bad in (("sensor_order_sha256", "X"), ("act_checkpoint_sha256", "X"),
                           ("safety_model_sha256", "X")):
            kwargs = {"sensor_order_sha256": "a", "act_checkpoint_sha256": "b",
                      "safety_model_sha256": "c", "offsamples": 4,
                      "controller_constants": {"gain": 4.0}}
            kwargs[field] = bad
            with pytest.raises(ParkedSkinContractError, match="runtime contract"):
                verify_runtime_contract(manifest, **kwargs)

    def test_rejects_a_changed_offsamples(self):
        manifest = {"sensor_order_sha256": "a", "act_checkpoint_sha256": "b",
                    "safety_model_sha256": "c", "offsamples": 4,
                    "controller_constants": {"gain": 4.0}}
        with pytest.raises(ParkedSkinContractError):
            verify_runtime_contract(manifest, sensor_order_sha256="a",
                                    act_checkpoint_sha256="b", safety_model_sha256="c",
                                    offsamples=1, controller_constants={"gain": 4.0})


# --------------------------------------------------------------------------- #
class TestPartitionAndLeakage:
    def test_partition_is_reused_exactly(self):
        partition = config("hybrid_obstacle_reference_partition_v2.json")
        prior = json.loads((ROOT / "diagnostics_output"
                            / "hybrid_obstacle_on_policy_reference"
                            / "final_decision.json").read_text())
        assert partition["partition_sha256"] == prior["reference_partition"]["sha256"]
        assert partition["composition"] == prior["reference_partition"]["composition"]

    def test_expected_composition(self):
        composition = config("hybrid_obstacle_reference_partition_v2.json")["composition"]
        assert (composition["reference_train"]["total"],
                composition["reference_calibration"]["total"],
                composition["reference_validation"]["total"],
                composition["offline_reference_test"]["total"]) == (64, 8, 8, 20)

    def test_no_development_or_confirmatory_leakage(self):
        partition = config("hybrid_obstacle_reference_partition_v2.json")
        development = {r["episode_id"] for r in
                       config("hybrid_obstacle_controller_development4_v1.json")["rows"]}
        confirmatory = {r["episode_id"] for r in
                        config("hybrid_obstacle_confirmatory41_v1.json")["rows"]}
        for name, rows in partition["partitions"].items():
            ids = {r["episode_id"] for r in rows}
            assert not (ids & development), name
            assert not (ids & confirmatory), name

    def test_confirmatory_remains_unexecuted(self):
        confirmatory = config("hybrid_obstacle_confirmatory41_v1.json")
        assert confirmatory["executed_in_this_task"] is False
        assert len(confirmatory["rows"]) == 41


# --------------------------------------------------------------------------- #
class TestDataAudit:
    @pytest.fixture(scope="class")
    def audit(self):
        path = (ROOT / "diagnostics_output/hybrid_obstacle_parked_skin_reference"
                / "paired_skin_data_audit.json")
        if not path.is_file():
            pytest.skip("data audit has not been run")
        return json.loads(path.read_text())

    def test_audit_reports_the_missing_target(self, audit):
        assert audit["valid"] is False
        assert audit["violations"]["missing_target_field_in_every_distribution"] is True
        assert audit["frames_meeting_both_contracts"] == 0

    def test_audit_does_not_silently_repair(self, audit):
        assert audit["silent_repair_performed"] is False

    def test_audit_reports_the_input_side_that_does_exist(self, audit):
        assert audit["families"]["expert"]["has_causal_current_skin"] is True
        assert audit["families"]["expert"]["causal_frame_shape"] == [4, 40, 8, 8]
        assert audit["checks_on_the_fields_that_do_exist"]["all_finite"] is True

    def test_audit_marks_the_constraint_unevaluable_rather_than_passing_it(self, audit):
        constraint = audit["physical_constraint_check"]
        assert constraint["evaluable"] is False
        assert constraint["violations_by_sensor"] is None

    def test_audit_source_never_fabricates_a_parked_field(self):
        source = AUDIT_SRC.read_text()
        assert "silent_repair_performed" in source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                text = ast.unparse(node.func)
                assert "randn" not in text and "default_rng" not in text


# --------------------------------------------------------------------------- #
class TestFrozenStack:
    def test_safety_head_is_not_modified(self):
        source = MODEL_SRC.read_text()
        assert "SafetyHead" not in source or "train" not in source.split("SafetyHead")[0][-40:]
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Attribute) and node.attr in {"requires_grad_",
                                                                 "load_state_dict"}:
                assert "safety" not in ast.unparse(node).lower()

    def test_residual_constants_untouched(self):
        from hybrid_safety_residual import (
            DEFAULT_DECAY,
            DEFAULT_EMA,
            DEFAULT_GAIN,
            DEFAULT_MAX_DEVIATION,
        )

        assert (DEFAULT_GAIN, DEFAULT_DECAY, DEFAULT_EMA, DEFAULT_MAX_DEVIATION) == \
               (4.0, 2.2, 0.75, 0.35)
        for node in ast.walk(ast.parse(MODEL_SRC.read_text())):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    name = getattr(target, "id", "") or getattr(target, "attr", "")
                    assert name not in {"gain", "decay", "ema", "max_deviation", "dt"}

    def test_residual_applied_after_temporal_aggregation(self):
        text = ADAPTER_SRC.read_text()
        assert text.index("nominal_action = self.model_output_to_action") < \
               text.index("executed_action = apply_arm_residual")

    def test_arm_only_and_gripper_preserved(self):
        from hybrid_safety_residual import apply_arm_residual

        nominal = {"arm": np.zeros(7, dtype=np.float32),
                   "gripper": np.array([0.125], dtype=np.float32)}
        out = apply_arm_residual(nominal, np.full(7, 0.3, dtype=np.float32))
        assert out["gripper"][0] == np.float32(0.125)
        assert out["arm"].shape == (7,)

    def test_broken_clearance_metric_is_not_referenced(self):
        for source in (MODEL_SRC, AUDIT_SRC):
            assert "minimum_clearance_m" not in source.read_text()
