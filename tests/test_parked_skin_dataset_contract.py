"""Contract tests for the parked-skin supervision dataset.

Handoff step 20. No GPU and no rollout: retention, history reconstruction, the closeness
transform, atomicity and resume are exercised on synthetic trajectories, and the frozen
manifest is asserted against the committed configs.
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

RETENTION_SRC = ACT / "parked_skin_retention.py"
EVALUATOR_SRC = ACT / "eval_act_obstacle_on_policy.py"
EXPERT_SRC = ROOT / "scripts/hybrid_obstacle_parked_skin_expert_reconstruct.py"
CONTRACT_SRC = ROOT / "scripts/hybrid_obstacle_parked_skin_dataset_contract.py"

pytest.importorskip("h5py")

from parked_skin_retention import (
    CAUSAL_FRAMES,
    CHANGED_PIXEL_EPSILON,
    DATASET_VERSION,
    DEPLOYABLE_FIELDS,
    INTEGRITY_FIELDS,
    PRIVILEGED_FIELDS,
    TOLERANCES,
    CompletionLedger,
    ParkedSkinRetentionError,
    TrajectoryRetention,
    closeness_to_depth,
    depth_to_closeness,
    load_trajectory,
    reconstruct_all_histories,
    reconstruct_history,
    resume_identity,
)


def config(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text())


def synthetic(frames: int = 6, hazard: bool = True, seed: int = 0):
    """A retention object with physically consistent paired fields."""
    rng = np.random.default_rng(seed)
    identity = {"dataset_version": DATASET_VERSION, "distribution": "EXPERT_RECONSTRUCTED",
                "partition": "reference_train", "trajectory_id": f"t{seed}",
                "manifest_row_id": f"row{seed}", "episode_id": f"ep{seed}",
                "candidate_index": seed, "hazard_present": hazard,
                "policy_condition": "EXPERT_RECONSTRUCTED", "source_h5_sha256": "x"}
    retention = TrajectoryRetention(identity=identity, provenance={"offsamples": 4})
    for step in range(frames):
        current_depth = rng.uniform(0.01, 0.5, size=(40, 8, 8)).astype(np.float32)
        parked_depth = current_depth.copy()
        if hazard:
            # removing the hazard can only push a surface FURTHER away
            parked_depth[0, :2] = np.minimum(current_depth[0, :2] * 4.0, 0.5)
        retention.add(
            episode_step=step, control_timestamp=step * 0.066,
            current_depth=current_depth, parked_depth=parked_depth,
            qpos=np.zeros(9, np.float32), qvel=np.zeros(9, np.float32),
            nominal_action=np.zeros(8, np.float32), gripper_state=np.zeros(2, np.float32),
            gripper_command=np.zeros(1, np.float32),
            current_head=np.full(7, 1.0, np.float32),
            parked_head=np.full(7, 1.0 if not hazard else 0.5, np.float32),
            scientific_state_sha256=f"s{step}", state_neutral=True)
    return retention


# --------------------------------------------------------------------------- #
class TestManifestComposition:
    @pytest.fixture(scope="class")
    def manifest(self):
        return config("hybrid_obstacle_parked_skin_supervision_v1.json")

    def test_exact_rollout_counts(self, manifest):
        assert manifest["total_policy_rollouts"] == 264
        assert manifest["total_reconstructions"] == 100
        per = manifest["distributions"]
        assert per["EXPERT_RECONSTRUCTED"]["rows"] == 100
        assert per["EXPERT_RECONSTRUCTED"]["policy_rollouts"] == 0
        assert per["ACT_ONLY_ON_POLICY"]["policy_rollouts"] == 100
        assert per["ORACLE_ON_POLICY"]["policy_rollouts"] == 100
        assert per["LEARNER_INDUCED_ON_POLICY"]["policy_rollouts"] == 64

    def test_balanced_condition_order(self, manifest):
        assert manifest["condition_order_balance"] == {"ACT_ONLY_ON_POLICY": 50,
                                                       "ORACLE_ON_POLICY": 50}

    def test_learner_rows_ordered_after_all_act_oracle_identities(self, manifest):
        orders = {e["distribution"]: [] for e in manifest["entries"]}
        for entry in manifest["entries"]:
            orders[entry["distribution"]].append(entry["execution_order"])
        first_learner = min(orders["LEARNER_INDUCED_ON_POLICY"])
        assert first_learner > max(orders["ACT_ONLY_ON_POLICY"])
        assert first_learner > max(orders["ORACLE_ON_POLICY"])

    def test_learner_rows_are_the_64_training_rows(self, manifest):
        partition = config("hybrid_obstacle_reference_partition_v2.json")
        training = {r["episode_id"] for r in partition["partitions"]["reference_train"]}
        learner = {e["episode_id"] for e in manifest["entries"]
                   if e["distribution"] == "LEARNER_INDUCED_ON_POLICY"}
        assert learner == training and len(learner) == 64

    def test_development4_and_confirmatory41_excluded(self, manifest):
        scheduled = {e["episode_id"] for e in manifest["entries"]}
        development = {r["episode_id"] for r in
                       config("hybrid_obstacle_controller_development4_v1.json")["rows"]}
        confirmatory = {r["episode_id"] for r in
                        config("hybrid_obstacle_confirmatory41_v1.json")["rows"]}
        assert not (scheduled & development)
        assert not (scheduled & confirmatory)
        assert manifest["development4_excluded"] and manifest["confirmatory41_excluded"]

    def test_partition_reused_unchanged(self, manifest):
        partition = config("hybrid_obstacle_reference_partition_v2.json")
        assert manifest["partition_sha256"] == partition["partition_sha256"]
        assert manifest["partition_composition"] == partition["composition"]

    def test_concurrency_capped_at_two(self, manifest):
        assert manifest["max_concurrent_rollout_processes"] == 2

    def test_learner_checkpoint_is_round_zero(self, manifest):
        frozen = json.loads((ROOT / "diagnostics_output/hybrid_obstacle_on_policy_reference"
                             / "round0_deployment_manifest.json").read_text())
        assert manifest["distributions"]["LEARNER_INDUCED_ON_POLICY"][
            "checkpoint_sha256"] == frozen["artifact_file_sha256"]
        assert "ROUND0" in manifest["distributions"]["LEARNER_INDUCED_ON_POLICY"]["label"]


# --------------------------------------------------------------------------- #
class TestClosenessTransform:
    def test_canonical_formula_and_validity(self):
        depth = np.array([0.25, 0.5, 0.0, 0.004, 0.005], dtype=np.float32)
        closeness, valid = depth_to_closeness(depth)
        assert np.allclose(closeness[:2], [0.5, 0.0], atol=1e-6)
        assert closeness[2] == 0.0 and closeness[3] == 0.0
        assert closeness[4] > 0.98
        assert valid.tolist() == [True, True, False, False, True]

    def test_dead_pixels_are_zero_not_one(self):
        closeness, valid = depth_to_closeness(np.zeros(4, dtype=np.float32))
        assert (closeness == 0).all() and not valid.any()

    def test_validity_is_stored_not_inferred_from_closeness(self):
        """0.5 m is a genuine return with zero closeness; 0 m is no return at all."""
        closeness, valid = depth_to_closeness(np.array([0.5, 0.0], dtype=np.float32))
        assert closeness[0] == closeness[1] == 0.0
        assert valid[0] and not valid[1]

    def test_inverse(self):
        depth = np.array([0.02, 0.2, 0.45], dtype=np.float32)
        closeness, _ = depth_to_closeness(depth)
        assert np.allclose(closeness_to_depth(closeness), depth, atol=1e-6)


# --------------------------------------------------------------------------- #
class TestHistoryReconstruction:
    def test_committed_padding_rule(self):
        sequence = np.arange(8 * 40 * 8 * 8, dtype=np.float32).reshape(8, 40, 8, 8)
        expected = {0: [0, 0, 0, 0], 1: [0, 0, 0, 1], 2: [0, 0, 1, 2], 3: [0, 1, 2, 3],
                    4: [1, 2, 3, 4]}
        for step, indices in expected.items():
            _, sources = reconstruct_history(sequence, step)
            assert sources == indices, step

    def test_shape_and_last_frame(self):
        sequence = np.random.default_rng(0).random((9, 40, 8, 8)).astype(np.float32)
        windows, sources = reconstruct_all_histories(sequence)
        assert windows.shape == (9, CAUSAL_FRAMES, 40, 8, 8)
        assert np.array_equal(windows[:, -1], sequence)
        assert np.array_equal(sources[:, -1], np.arange(9))

    def test_never_reads_a_future_frame(self):
        sequence = np.random.default_rng(1).random((12, 40, 8, 8)).astype(np.float32)
        _, sources = reconstruct_all_histories(sequence)
        assert (sources <= np.arange(12)[:, None]).all()

    def test_refuses_an_out_of_range_step(self):
        sequence = np.zeros((3, 40, 8, 8), dtype=np.float32)
        for step in (-1, 3, 99):
            with pytest.raises(ParkedSkinRetentionError):
                reconstruct_history(sequence, step)

    def test_window_is_scoped_to_one_trajectory(self, tmp_path):
        """Reconstruction takes a single file's sequence, so it cannot straddle files."""
        first, second = synthetic(5, seed=1), synthetic(5, seed=2)
        first.publish(tmp_path / "a.h5")
        second.publish(tmp_path / "b.h5")
        loaded = load_trajectory(tmp_path / "a.h5")
        windows, _ = reconstruct_all_histories(loaded["current_closeness"])
        assert len(windows) == 5

    def test_no_duplicated_history_tensor_is_stored(self, tmp_path):
        import h5py

        synthetic(6).publish(tmp_path / "t.h5")
        with h5py.File(tmp_path / "t.h5") as handle:
            for group in handle:
                for name in handle[group]:
                    assert handle[f"{group}/{name}"].ndim != 5, f"{group}/{name}"

    def test_contract_source_declares_sequence_storage(self):
        text = CONTRACT_SRC.read_text()
        assert "(T,40,8,8)" in text and "never stored duplicated" in text


# --------------------------------------------------------------------------- #
class TestRetentionAndPairing:
    def test_physical_inequality_enforced_at_publish(self, tmp_path):
        retention = synthetic(3)
        # inject a parked field CLOSER than current -- a mispairing
        retention.frames[1]["parked_closeness"] = np.minimum(
            retention.frames[1]["current_closeness"] + 0.5, 1.0)
        with pytest.raises(ParkedSkinRetentionError, match="parked <= current"):
            retention.publish(tmp_path / "bad.h5")
        assert not (tmp_path / "bad.h5").exists()

    def test_hazard_absent_is_exactly_zero(self, tmp_path):
        record = synthetic(5, hazard=False).publish(tmp_path / "absent.h5")
        loaded = load_trajectory(tmp_path / "absent.h5", allow_privileged=True)
        assert np.array_equal(loaded["current_closeness"], loaded["parked_closeness"])
        assert float(np.abs(loaded["removable_closeness"]).max()) == 0.0
        assert not loaded["changed_pixel_mask"].any()
        assert float(np.abs(loaded["oracle_dq"]).max()) == 0.0
        assert record["changed_pixel_fraction"] == 0.0

    def test_changed_mask_matches_removable(self, tmp_path):
        synthetic(4).publish(tmp_path / "t.h5")
        loaded = load_trajectory(tmp_path / "t.h5", allow_privileged=True)
        assert np.array_equal(loaded["changed_pixel_mask"],
                              loaded["removable_closeness"] > CHANGED_PIXEL_EPSILON)

    def test_state_neutrality_failure_blocks_publication(self, tmp_path):
        retention = synthetic(3)
        retention.frames[2]["state_neutral"] = False
        with pytest.raises(ParkedSkinRetentionError, match="mutated live state"):
            retention.publish(tmp_path / "bad.h5")

    def test_non_contiguous_steps_rejected(self, tmp_path):
        retention = synthetic(4)
        retention.frames[2]["episode_step"] = 7
        with pytest.raises(ParkedSkinRetentionError, match="contiguous"):
            retention.publish(tmp_path / "bad.h5")

    def test_non_monotonic_timestamps_rejected(self, tmp_path):
        retention = synthetic(4)
        retention.frames[2]["control_timestamp"] = 0.0
        with pytest.raises(ParkedSkinRetentionError, match="monotonic"):
            retention.publish(tmp_path / "bad.h5")

    def test_empty_trajectory_rejected(self, tmp_path):
        retention = TrajectoryRetention(identity={}, provenance={})
        with pytest.raises(ParkedSkinRetentionError, match="no frames"):
            retention.publish(tmp_path / "bad.h5")

    def test_tolerances_are_predeclared(self):
        assert TOLERANCES == {"head_output_max_abs_delta": 1e-6,
                              "oracle_differential_max_abs_delta": 1e-6,
                              "closeness_inequality": 1e-7}


# --------------------------------------------------------------------------- #
class TestSchemaSeparation:
    def test_groups_are_separated(self, tmp_path):
        import h5py

        synthetic(3).publish(tmp_path / "t.h5")
        with h5py.File(tmp_path / "t.h5") as handle:
            assert set(handle.keys()) == {"deployable", "privileged", "integrity"}
            assert set(handle["deployable"].keys()) == set(DEPLOYABLE_FIELDS)
            assert set(handle["privileged"].keys()) == set(PRIVILEGED_FIELDS)
            assert set(handle["integrity"].keys()) == set(INTEGRITY_FIELDS)
            assert "PRIVILEGED" in handle["privileged"].attrs["warning"]

    def test_loader_withholds_privileged_by_default(self, tmp_path):
        synthetic(3).publish(tmp_path / "t.h5")
        default = load_trajectory(tmp_path / "t.h5")
        assert not (set(default) & set(PRIVILEGED_FIELDS))
        opened = load_trajectory(tmp_path / "t.h5", allow_privileged=True)
        assert set(PRIVILEGED_FIELDS) <= set(opened)

    def test_no_target_lives_in_the_deployable_group(self):
        assert not (set(DEPLOYABLE_FIELDS) & set(PRIVILEGED_FIELDS))
        for name in ("parked_closeness", "parked_head", "oracle_dq", "removable_closeness"):
            assert name not in DEPLOYABLE_FIELDS

    def test_every_required_per_frame_field_is_present(self, tmp_path):
        synthetic(3).publish(tmp_path / "t.h5")
        loaded = load_trajectory(tmp_path / "t.h5", allow_privileged=True)
        for name in ("current_closeness", "current_valid_mask", "qpos", "qvel",
                     "nominal_action", "gripper_state", "gripper_command",
                     "episode_step", "control_timestamp", "parked_closeness",
                     "parked_valid_mask", "removable_closeness", "changed_pixel_mask",
                     "current_head", "parked_head", "oracle_dq", "oracle_active",
                     "teacher_dq", "teacher_valid", "current_field_sha256",
                     "parked_field_sha256", "scientific_state_sha256", "state_neutral"):
            assert name in loaded, name

    def test_identity_and_provenance_attributes(self, tmp_path):
        import h5py

        synthetic(3).publish(tmp_path / "t.h5")
        with h5py.File(tmp_path / "t.h5") as handle:
            for key in ("dataset_version", "distribution", "partition", "trajectory_id",
                        "manifest_row_id", "episode_id", "candidate_index",
                        "policy_condition", "frames", "offsamples",
                        "causal_history_rule"):
                assert key in handle.attrs, key


# --------------------------------------------------------------------------- #
class TestAtomicityAndResume:
    def test_publish_is_atomic(self, tmp_path):
        path = tmp_path / "t.h5"
        synthetic(3).publish(path)
        assert path.is_file()
        assert not list(tmp_path.glob("*.partial.h5"))

    def test_failed_validation_leaves_no_output(self, tmp_path):
        retention = synthetic(3)
        retention.frames[1]["state_neutral"] = False
        with pytest.raises(ParkedSkinRetentionError):
            retention.publish(tmp_path / "t.h5")
        assert not (tmp_path / "t.h5").exists()

    def test_resume_identity_is_the_committed_tuple(self):
        base = {"distribution": "ACT_ONLY_ON_POLICY", "episode_id": "e1",
                "policy_condition": "ACT_ONLY_ON_POLICY", "manifest_row_id": "r1"}
        assert resume_identity(base) == resume_identity(dict(base))
        for field in ("distribution", "episode_id", "policy_condition",
                      "manifest_row_id"):
            changed = {**base, field: "other"}
            assert resume_identity(changed) != resume_identity(base), field

    def test_ledger_refuses_a_duplicate_identity(self, tmp_path):
        ledger = CompletionLedger(tmp_path / "ledger.jsonl")
        entry = {"distribution": "ACT_ONLY_ON_POLICY", "episode_id": "e1",
                 "policy_condition": "ACT_ONLY_ON_POLICY", "manifest_row_id": "r1",
                 "frames": 10}
        ledger.record(entry)
        assert ledger.contains(entry)
        with pytest.raises(ParkedSkinRetentionError, match="duplicate scientific identity"):
            ledger.record(entry)

    def test_ledger_survives_reopen(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        entry = {"distribution": "ORACLE_ON_POLICY", "episode_id": "e2",
                 "policy_condition": "ORACLE_ON_POLICY", "manifest_row_id": "r2"}
        CompletionLedger(path).record(entry)
        assert CompletionLedger(path).contains(entry)

    def test_runner_skips_finalized_outputs(self):
        runner = Path("/tmp/claude-0/-root-prox-learning-hybrid-safety"
                      "/ef021269-325a-47b3-98bd-db3bd3d4c292/scratchpad/run_parked_skin.sh")
        if not runner.is_file():
            pytest.skip("runner not present")
        assert "already finalized, skipping" in runner.read_text()


# --------------------------------------------------------------------------- #
class TestCollectionSources:
    def test_shadow_oracle_does_not_alter_the_executed_action(self):
        """Retention reads the fields; it never writes the baseline or the action."""
        source = EVALUATOR_SRC.read_text()
        block = source.split("if self.retention is not None:")[1].split(
            'frame["on_policy"]')[0]
        for forbidden in ("_baseline_safety_output", "executed", "self._step ="):
            assert forbidden not in block, forbidden

    def test_stored_parked_field_is_the_one_the_oracle_used(self):
        source = EVALUATOR_SRC.read_text()
        assert "parked_depth=parked_skin" in source
        # the parked field is rendered exactly once per frame
        assert source.count("self._oracle.parked_skin()") == 1

    def test_expert_reconstruction_never_runs_act(self):
        source = EXPERT_SRC.read_text()
        assert "inference_model" not in source
        assert "policy_best" not in source
        assert '"act_run": False' in source

    def test_expert_reconstruction_opens_the_source_read_only(self):
        source = EXPERT_SRC.read_text()
        assert 'h5py.File(source, "r")' in source

    def test_no_dynamics_advancing_call_in_retention_or_expert(self):
        for path in (RETENTION_SRC, EXPERT_SRC):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call):
                    assert "mj_forward" not in ast.unparse(node.func)

    def test_retention_never_subsamples(self):
        source = RETENTION_SRC.read_text()
        for forbidden in ("subsample", "if active", "skip_zero", "balance"):
            assert forbidden not in source, forbidden
        manifest = config("hybrid_obstacle_parked_skin_supervision_v1.json")
        assert "no balancing and no" in manifest["retention_rule"]
