"""Contract tests for the on-policy aggregation round.

Handoff step 17. No GPU, no simulator, no rollout. Behaviour is exercised on synthetic
arrays; the properties that can only be asserted about *source* -- one aggregation round,
no continuation from V1 weights, calibration fitted on calibration data only, no
global-min-depth dependency in the V2 gate -- are checked at the AST level.
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
EVALUATOR_SRC = ACT / "eval_act_obstacle_on_policy.py"
ADAPTER_SRC = ACT / "eval_act_obstacle_safety.py"
PARTITION_SRC = ROOT / "scripts/hybrid_obstacle_reference_partition.py"
SCHEDULE_SRC = ROOT / "scripts/hybrid_obstacle_on_policy_schedule.py"
TRAIN_SRC = ROOT / "scripts/hybrid_obstacle_train_on_policy_reference.py"
AUDIT_SRC = ROOT / "scripts/hybrid_obstacle_on_policy_dataset_audit.py"

from deployable_reference import (
    FEATURE_FIELDS,
    FEATURE_WIDTHS,
    MLP_REFERENCE_ID,
    PRIVILEGED_FIELDS,
    DeployableReferenceError,
    SupportEnvelopeGate,
    assert_no_privileged_fields,
    build_mlp,
)


def config(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text())


# --------------------------------------------------------------------------- #
class TestReferencePartition:
    @pytest.fixture(scope="class")
    def partition(self):
        return config("hybrid_obstacle_reference_partition_v2.json")

    def test_exact_compositions(self, partition):
        expected = {
            "reference_train": (48, 16), "reference_calibration": (6, 2),
            "reference_validation": (6, 2), "offline_reference_test": (15, 5)}
        for name, (present, absent) in expected.items():
            composition = partition["composition"][name]
            assert (composition["hazard_present"], composition["hazard_absent"]) == \
                   (present, absent), name

    def test_all_pairwise_disjoint(self, partition):
        assert partition["all_pairwise_disjoint"]
        assert partition["pairwise_overlaps"] == {}

    def test_no_development_or_confirmatory_leakage(self, partition):
        development = {r["episode_id"] for r in
                       config("hybrid_obstacle_controller_development4_v1.json")["rows"]}
        confirmatory = {r["episode_id"] for r in
                        config("hybrid_obstacle_confirmatory41_v1.json")["rows"]}
        for name, rows in partition["partitions"].items():
            identifiers = {r["episode_id"] for r in rows}
            assert not (identifiers & development), name
            assert not (identifiers & confirmatory), name

    def test_three_reference_partitions_cover_act_train_exactly(self, partition):
        split = config("hybrid_obstacle_canonical_split_v2.json")
        act_train = {e["episode_id"] for e in split["episodes"] if e["split"] == "train"}
        covered = set()
        for name in ("reference_train", "reference_calibration", "reference_validation"):
            covered |= {r["episode_id"] for r in partition["partitions"][name]}
        assert covered == act_train

    def test_offline_test_is_the_act_validation_split(self, partition):
        split = config("hybrid_obstacle_canonical_split_v2.json")
        act_validation = {e["episode_id"] for e in split["episodes"]
                          if e["split"] == "validation"}
        assert {r["episode_id"] for r in
                partition["partitions"]["offline_reference_test"]} == act_validation

    def test_assignment_is_deterministic_from_committed_ranks(self):
        """No randomness, no file-order dependence -- checked as calls, not substrings."""
        tree = ast.parse(PARTITION_SRC.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                text = ast.unparse(node.func)
                for forbidden in ("random", "shuffle", "default_rng", "listdir", "glob",
                                  "iterdir"):
                    assert forbidden not in text, ast.unparse(node)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert "random" not in ast.unparse(node)
        assert "predeclared_stratum_rank" in PARTITION_SRC.read_text()

    def test_act_split_is_untouched(self, partition):
        assert partition["act_split_unchanged"] is True
        assert partition["act_split_manifest_sha256"] == \
               config("hybrid_obstacle_canonical_split_v2.json")["split_manifest_sha256"]


# --------------------------------------------------------------------------- #
class TestSchedules:
    def test_labelling_schedule_is_200_balanced_rollouts(self):
        schedule = config("hybrid_obstacle_on_policy_labelling_schedule_v2.json")
        assert schedule["rollouts"] == 200
        assert schedule["rollout_budget"] == 200
        counts = schedule["order_balance"]["first_condition_counts"]
        assert counts == {"ACT_ONLY_ON_POLICY": 50, "ACT_PLUS_ORACLE_ON_POLICY": 50}

    def test_every_row_gets_both_conditions_exactly_once(self):
        schedule = config("hybrid_obstacle_on_policy_labelling_schedule_v2.json")
        seen: dict[tuple[str, str], int] = {}
        for entry in schedule["entries"]:
            key = (entry["episode_id"], entry["condition"])
            seen[key] = seen.get(key, 0) + 1
        assert set(seen.values()) == {1}
        assert len(seen) == 200

    def test_no_development_or_confirmatory_row_in_the_labelling_schedule(self):
        schedule = config("hybrid_obstacle_on_policy_labelling_schedule_v2.json")
        scheduled = {e["episode_id"] for e in schedule["entries"]}
        development = {r["episode_id"] for r in
                       config("hybrid_obstacle_controller_development4_v1.json")["rows"]}
        confirmatory = {r["episode_id"] for r in
                        config("hybrid_obstacle_confirmatory41_v1.json")["rows"]}
        assert not (scheduled & development)
        assert not (scheduled & confirmatory)
        assert schedule["confirmatory_rows_included"] is False

    def test_schedule_refuses_a_non_development_manifest_for_live(self):
        assert "refusing a manifest whose role is" in SCHEDULE_SRC.read_text()


# --------------------------------------------------------------------------- #
class TestSupportEnvelopeGate:
    def test_gate_has_no_minimum_depth_condition(self):
        """The V1 depth gate was open on 76% of frames; V2 must not depend on it.

        Behavioural check: the returned vector is invariant to the diagnostic summary,
        including one that says every sensor is saturated far away.
        """
        gate = SupportEnvelopeGate(tau=0.5, rho_max=5.0)
        vector = np.ones(7, dtype=np.float32)
        near = np.zeros((40, 4), dtype=np.float32)
        far = np.ones((40, 4), dtype=np.float32)
        without, _ = gate(vector)
        with_near, _ = gate(vector, near)
        with_far, _ = gate(vector, far)
        assert np.array_equal(without, with_near)
        assert np.array_equal(without, with_far)
        # and structurally: neither the activation nor the cap reads it
        tree = ast.parse(REFERENCE_SRC.read_text())
        gate_class = next(n for n in ast.walk(tree)
                          if isinstance(n, ast.ClassDef) and n.name == "SupportEnvelopeGate")
        call = next(f for f in gate_class.body
                    if isinstance(f, ast.FunctionDef) and f.name == "__call__")
        for node in ast.walk(call):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") in {"gated", "activated", "capped", "norm"}
                    for t in node.targets):
                assert "diagnostic_summary" not in ast.unparse(node.value)
        assert "d_act" not in ast.unparse(call)

    def test_silent_below_tau(self):
        gate = SupportEnvelopeGate(tau=1.0, rho_max=5.0)
        gated, log = gate(np.full(7, 0.1, dtype=np.float32))
        assert np.array_equal(gated, np.zeros(7, dtype=np.float32))
        assert log["activated"] is False

    def test_passes_through_between_tau_and_rho_max(self):
        gate = SupportEnvelopeGate(tau=0.5, rho_max=10.0)
        vector = np.array([1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        gated, log = gate(vector)
        assert np.allclose(gated, vector)
        assert log["activated"] and not log["capped"]

    def test_caps_norm_but_preserves_direction(self):
        gate = SupportEnvelopeGate(tau=0.5, rho_max=2.0)
        vector = np.array([6.0, 8.0, 0, 0, 0, 0, 0], dtype=np.float32)   # norm 10
        gated, log = gate(vector)
        assert log["capped"] is True
        assert float(np.linalg.norm(gated)) == pytest.approx(2.0, abs=1e-5)
        unit_in = vector / np.linalg.norm(vector)
        unit_out = gated / np.linalg.norm(gated)
        assert np.allclose(unit_in, unit_out, atol=1e-6)

    def test_cap_would_have_bounded_the_v1_failure(self):
        """V1 reached ~6.9x the true oracle norm; the envelope bounds that."""
        gate = SupportEnvelopeGate(tau=0.1, rho_max=1.0)
        gated, _ = gate(np.array([6.9, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        assert float(np.linalg.norm(gated)) <= 1.0 + 1e-6

    def test_gate_log_schema(self):
        gate = SupportEnvelopeGate(tau=0.5, rho_max=2.0)
        _, log = gate(np.ones(7, dtype=np.float32))
        assert set(log) == {"predicted_norm", "tau", "activated", "rho_max", "capped",
                            "executed_norm"}


# --------------------------------------------------------------------------- #
class TestCalibration:
    def _data(self, oracle_norms, predicted_norms, hazard_present=True):
        count = len(oracle_norms)
        oracle = np.zeros((count, 7), dtype=np.float32)
        oracle[:, 0] = oracle_norms
        predicted = np.zeros((count, 7), dtype=np.float32)
        predicted[:, 0] = predicted_norms
        return predicted, {
            "privileged_oracle_dq": oracle,
            "privileged_oracle_norm": np.asarray(oracle_norms, dtype=np.float64),
            "privileged_teacher_active": np.zeros(count, dtype=bool),
            "teacher_evaluable_row": np.zeros(count, dtype=bool),
            "hazard_present_row": np.full(count, hazard_present, dtype=bool),
        }

    def test_tau_retains_recall_and_suppresses_zero_frames(self):
        from hybrid_obstacle_train_on_policy_reference import calibrate

        oracle = [1.0] * 10 + [0.0] * 10
        predicted = [2.0] * 10 + [0.1] * 10
        fit = calibrate(*self._data(oracle, predicted))
        assert fit["feasible"]
        assert fit["recall"] >= 0.80
        assert fit["calibration_false_activation"] == 0.0
        assert 0.1 < fit["tau"] <= 2.0

    def test_infeasible_when_direction_is_wrong(self):
        from hybrid_obstacle_train_on_policy_reference import calibrate

        count = 20
        oracle = np.zeros((count, 7), dtype=np.float32)
        oracle[:, 0] = 1.0
        predicted = np.zeros((count, 7), dtype=np.float32)
        predicted[:, 0] = -2.0                      # exactly reversed
        data = {
            "privileged_oracle_dq": oracle,
            "privileged_oracle_norm": np.ones(count),
            "privileged_teacher_active": np.zeros(count, dtype=bool),
            "teacher_evaluable_row": np.zeros(count, dtype=bool),
            "hazard_present_row": np.ones(count, dtype=bool),
        }
        assert calibrate(predicted, data)["feasible"] is False

    def test_highest_tau_wins_among_equal_false_activation(self):
        from hybrid_obstacle_train_on_policy_reference import calibrate

        oracle = [1.0] * 10 + [0.0] * 5
        predicted = [3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        fit = calibrate(*self._data(oracle, predicted))
        # every tau <= 2.0 gives recall 1.0 and zero false activation; the rule takes the
        # largest such tau
        assert fit["tau"] == pytest.approx(2.0)

    def test_rho_max_is_the_privileged_oracle_percentile(self):
        from hybrid_obstacle_train_on_policy_reference import RHO_PERCENTILE, calibrate

        oracle = list(np.linspace(1.0, 5.0, 20))
        predicted = [10.0] * 20
        fit = calibrate(*self._data(oracle, predicted))
        assert fit["rho_max"] == pytest.approx(
            float(np.percentile(np.asarray(oracle), RHO_PERCENTILE)), rel=1e-6)

    def test_calibration_is_fitted_on_calibration_rows_only(self):
        tree = ast.parse(TRAIN_SRC.read_text())
        calls = [ast.unparse(n) for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and ast.unparse(n).startswith("calibrate(")]
        assert calls, "no calibrate() call found"
        for call in calls:
            assert "calibration" in call
            assert "validation" not in call and "test" not in call

    def test_tau_is_not_inherited_from_v1(self):
        previous = json.loads((ROOT / "diagnostics_output"
                               / "hybrid_obstacle_deployable_reference"
                               / "deployment_manifest.json").read_text())
        source = TRAIN_SRC.read_text()
        assert str(previous["tau"]) not in source
        assert "percentile of predicted" not in source


# --------------------------------------------------------------------------- #
class TestAggregation:
    def test_exactly_one_aggregation_round(self):
        source = TRAIN_SRC.read_text()
        tree = ast.parse(source)
        choices = [ast.unparse(n) for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and "add_argument" in ast.unparse(n)
                   and "--round" in ast.unparse(n)]
        assert choices and "'0', '1'" in choices[0].replace('"', "'")

    def test_round_one_uses_four_distributions_at_equal_weight(self):
        source = TRAIN_SRC.read_text()
        assert 'distributions["learner_on_policy"]' in source
        assert "weight = 1.0 / len(distributions)" in source

    def test_fresh_initialisation_never_continues_from_v1(self):
        """The only load is the strict-reload check of the model just written."""
        source = TRAIN_SRC.read_text()
        assert '"continued_from_previous_checkpoint": False' in source
        assert '"fresh_initialisation": True' in source
        loads = [ast.unparse(n) for n in ast.walk(ast.parse(source))
                 if isinstance(n, ast.Call)
                 and ast.unparse(n).startswith("PostureSkinMlpReference.load")]
        assert loads == ["PostureSkinMlpReference.load(artifact, device=device)"], loads
        # `artifact` is the path this run just saved, never a V1 checkpoint
        assert "reference.save(artifact)" in source
        assert "posture_skin_mlp_reference_v1" not in source
        # training always starts from build_mlp inside train_mlp, which is V1's function
        assert "train_mlp(features, targets" in source

    def test_architecture_and_training_configuration_are_the_v1_ones(self):
        import hybrid_obstacle_train_deployable_reference as v1
        import hybrid_obstacle_train_on_policy_reference as v2

        assert (v2.SEED, v2.LEARNING_RATE, v2.WEIGHT_DECAY, v2.BATCH_SIZE,
                v2.MAX_EPOCHS) == (v1.SEED, v1.LEARNING_RATE, v1.WEIGHT_DECAY,
                                   v1.BATCH_SIZE, v1.MAX_EPOCHS)
        assert v2.train_mlp is v1.train_mlp
        model = build_mlp(FEATURE_WIDTHS[MLP_REFERENCE_ID])
        assert sum(p.numel() for p in model.parameters()) == 150023

    def test_sampling_is_uniform_over_rows_then_frames(self):
        from hybrid_obstacle_train_on_policy_reference import sample_balanced

        def row(count, value):
            return {
                "qpos": np.full((count, 9), value, dtype=np.float32),
                "qvel": np.zeros((count, 9), dtype=np.float32),
                "nominal_action": np.zeros((count, 8), dtype=np.float32),
                "gripper_state": np.zeros((count, 2), dtype=np.float32),
                "gripper_command": np.zeros((count, 1), dtype=np.float32),
                "current_head": np.zeros((count, 7), dtype=np.float32),
                "sensor_summary": np.zeros((count, 40, 4), dtype=np.float32),
                "privileged_parked_head": np.full((count, 7), value, dtype=np.float32),
            }
        # one distribution has 100x the frames of the other; equal weighting must ignore that
        distributions = {"a": [row(1000, 1.0)], "b": [row(10, 2.0)]}
        features, targets, provenance = sample_balanced(distributions, 500, 0)
        assert len(features) == 1000
        assert provenance["a"]["frames_drawn"] == provenance["b"]["frames_drawn"] == 500
        assert abs((targets[:, 0] == 1.0).sum() - (targets[:, 0] == 2.0).sum()) == 0


# --------------------------------------------------------------------------- #
class TestPrivilegedInputs:
    def test_runtime_schema_is_unchanged_from_v1(self):
        previous = json.loads((ROOT / "diagnostics_output"
                               / "hybrid_obstacle_deployable_reference"
                               / "deployment_manifest.json").read_text())
        assert list(FEATURE_FIELDS[MLP_REFERENCE_ID]) == previous["runtime_inputs"]
        assert FEATURE_WIDTHS[MLP_REFERENCE_ID] == previous["feature_width"]

    def test_privileged_fields_are_refused(self):
        for field in sorted(PRIVILEGED_FIELDS):
            with pytest.raises(DeployableReferenceError):
                assert_no_privileged_fields(["qpos", field])

    def test_evaluator_builds_features_from_runtime_only(self):
        source = EVALUATOR_SRC.read_text()
        block = source.split("# ---- runtime-observable side")[1].split(
            "# ---- privileged label generation")[0]
        for forbidden in ("parked_skin", "privileged", "hazard_present", "protr",
                          "mocap_pos", "teacher"):
            assert forbidden not in block, forbidden

    def test_labels_live_in_a_separate_namespace(self):
        source = EVALUATOR_SRC.read_text()
        assert '"privileged_parked_head"' in source
        assert '"privileged_oracle_dq"' in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Dict) and node.keys
                    and all(isinstance(k, ast.Constant) for k in node.keys)):
                keys = {k.value for k in node.keys if isinstance(k.value, str)}
                if keys == set(FEATURE_FIELDS[MLP_REFERENCE_ID]):
                    break
        else:
            pytest.fail("the runtime feature dict was not found")

    def test_audit_checks_the_namespace(self):
        assert "no_privileged_field_outside_namespace" in AUDIT_SRC.read_text()


# --------------------------------------------------------------------------- #
class TestEvaluatorContract:
    def test_confirmatory_manifest_is_hard_refused(self):
        source = EVALUATOR_SRC.read_text()
        assert "refusing to execute a confirmatory row in this task" in source
        assert "ACCEPTED_ROLES = (\"REFERENCE_PARTITION\", \"DEVELOPMENT_ONLY\")" in source

    def test_oracle_pairing_is_the_validated_one(self):
        source = EVALUATOR_SRC.read_text()
        assert "PerFrameParkedObstacleReference" in source
        assert "render_current_skin()" in source
        assert "parked_skin()" in source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                assert "mj_forward" not in ast.unparse(node.func)

    def test_inference_is_memoized_once_per_step(self):
        tree = ast.parse(EVALUATOR_SRC.read_text())
        cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                   and n.name == "OnPolicyLabelledPolicy")
        method = next(f for f in cls.body if isinstance(f, ast.FunctionDef)
                      and f.name == "inference_model")
        body = ast.unparse(method)
        assert "_inference_cache" in body and "cached[0] == self._step" in body

    def test_residual_after_temporal_aggregation(self):
        text = ADAPTER_SRC.read_text()
        assert text.index("nominal_action = self.model_output_to_action") < \
               text.index("executed_action = apply_arm_residual")
        assert "pc.temp_agg_off = False" in EVALUATOR_SRC.read_text()

    def test_arm_only_and_gripper_preserved(self):
        from hybrid_safety_residual import apply_arm_residual

        nominal = {"arm": np.zeros(7, dtype=np.float32),
                   "gripper": np.array([0.875], dtype=np.float32)}
        out = apply_arm_residual(nominal, np.full(7, 0.2, dtype=np.float32))
        assert out["gripper"][0] == np.float32(0.875)
        assert out["arm"].shape == (7,)

    def test_controller_constants_untouched(self):
        from hybrid_safety_residual import (
            DEFAULT_DECAY,
            DEFAULT_EMA,
            DEFAULT_GAIN,
            DEFAULT_MAX_DEVIATION,
        )

        assert (DEFAULT_GAIN, DEFAULT_DECAY, DEFAULT_EMA, DEFAULT_MAX_DEVIATION) == \
               (4.0, 2.2, 0.75, 0.35)
        for source in (EVALUATOR_SRC, TRAIN_SRC, REFERENCE_SRC):
            for node in ast.walk(ast.parse(source.read_text())):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        name = getattr(target, "id", "") or getattr(target, "attr", "")
                        assert name not in {"gain", "decay", "ema", "max_deviation"}

    def test_causal_and_controller_state_reset(self):
        tree = ast.parse(EVALUATOR_SRC.read_text())
        cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                   and n.name == "OnPolicyLabelledPolicy")
        reset = next(f for f in cls.body if isinstance(f, ast.FunctionDef)
                     and f.name == "reset")
        body = ast.unparse(reset)
        for field in ("_oracle", "_selector", "_reference", "_gate", "_inference_cache",
                      "_records", "_current_run"):
            assert field in body, field
        assert "super().reset()" in body

    def test_msaa_and_camera_contract(self):
        source = EVALUATOR_SRC.read_text()
        assert "offsamples != 4" in source
        assert '"--image_h", type=int, default=240' in source
        assert '"--image_w", type=int, default=320' in source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        assert target.attr != "offsamples"

    def test_broken_clearance_metric_is_not_used_as_evidence(self):
        """minimum_clearance_m is pinned at <=0 by a per-geom mj_geomDistance defect."""
        for source in (EVALUATOR_SRC, TRAIN_SRC,
                       ROOT / "scripts/hybrid_obstacle_on_policy_analysis.py"):
            if not Path(source).is_file():
                continue
            tree = ast.parse(Path(source).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Subscript):
                    assert "minimum_clearance_m" not in ast.unparse(node), source


# --------------------------------------------------------------------------- #
class TestLogSchema:
    REQUIRED = ("condition", "current_skin_sha256", "parked_skin_sha256",
                "feature_vector_sha256", "current_head", "predicted_parked_head",
                "predicted_dq", "predicted_dq_norm", "executed_dq", "oracle_parked_head",
                "oracle_dq", "oracle_dq_norm", "oracle_active", "cosine_with_oracle",
                "norm_ratio", "state_neutral", "minimum_depth_m")

    def test_every_required_key_is_written(self):
        tree = ast.parse(EVALUATOR_SRC.read_text())
        keys: set[str] = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Subscript)
                    and ast.unparse(node.targets[0]).endswith("['on_policy']")
                    and isinstance(node.value, ast.Dict)):
                keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        assert keys, "the per-frame on-policy log block was not found"
        assert set(self.REQUIRED) <= keys, sorted(set(self.REQUIRED) - keys)

    def test_gate_decision_is_logged(self):
        assert '**{f"gate_{k}": v for k, v in gate_log.items()}' in EVALUATOR_SRC.read_text()

    def test_frames_npz_carries_both_namespaces(self):
        source = EVALUATOR_SRC.read_text()
        for field in ("privileged_parked_head", "privileged_oracle_dq",
                      "privileged_oracle_active", "privileged_state_neutral",
                      "predicted_dq", "executed_dq", "correction"):
            assert f'"{field}"' in source, field
