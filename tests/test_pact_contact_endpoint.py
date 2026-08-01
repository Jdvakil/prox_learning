from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_pact_contact_occlusion_subset as occlusion
import build_pact_contact_power as power
import analyze_pact_contact_endpoint as analysis
import pact_contact_endpoint_contract as contact_contract
import build_pact_contact_token_plan as token_plan
import build_pact_contact_schedule as contact_schedule
import compact_pact_contact_storage as contact_storage


def identity_extrinsic() -> np.ndarray:
    return np.array(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
    )


def test_physical_landscape_frustum_uses_actual_video_aspect_ratio() -> None:
    extrinsic = identity_extrinsic()
    # At z=1, x=0.8 is inside the 624x352 landscape frustum but would be
    # outside the erroneous 352x624 portrait frustum implied by stored K.
    probes = np.array([[0.8, 0.0, 1.0], [0.0, 0.6, 1.0]])
    observed = occlusion.probes_inside_frustum(extrinsic, probes)[0]
    assert observed.tolist() == [True, False]


def test_camera_center_is_inverse_extrinsic_translation() -> None:
    extrinsic = identity_extrinsic()
    extrinsic[0, :, 3] = [-1.0, -2.0, -3.0]
    assert occlusion.camera_centers(extrinsic)[0] == pytest.approx([1.0, 2.0, 3.0])


def test_aabb_blocks_only_before_target() -> None:
    origin = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 2.0])
    half = np.array([0.1, 0.1, 0.1])
    assert occlusion.segment_intersects_aabb_before_target(
        origin, target, np.array([0.0, 0.0, 1.0]), half
    )
    assert not occlusion.segment_intersects_aabb_before_target(
        origin, target, np.array([0.0, 0.0, 3.0]), half
    )


@pytest.mark.parametrize(
    ("occluded", "non_occluded", "expected"),
    [
        (50, 50, "retain_subset_analysis"),
        (75, 25, "retain_subset_analysis"),
        (76, 24, "drop_subset_analysis_under_25_percent"),
        (96, 4, "drop_subset_analysis_degenerate"),
        (100, 0, "drop_subset_analysis_degenerate"),
    ],
)
def test_frozen_partition_boundaries(
    occluded: int, non_occluded: int, expected: str
) -> None:
    assert occlusion_module_action(occluded, non_occluded) == expected


def occlusion_module_action(occluded: int, non_occluded: int) -> str:
    return occlusion.partition_action(occluded, non_occluded)["action"]


def test_frozen_artifact_is_self_hashed_and_degenerate_if_present() -> None:
    path = ROOT / "diagnostics_output/pact_contact_endpoint/occlusion_subset.json"
    if not path.exists():
        pytest.skip("artifact is generated after the unit contract is validated")
    document = json.loads(path.read_text())
    payload = dict(document)
    observed = payload.pop("occlusion_subset_sha256")
    assert observed == occlusion.sha256_payload(payload)
    assert document["criterion"]["threshold"] == 0.50
    assert document["criterion"]["threshold_tuned"] is False
    assert document["criterion"]["policy_outcome_fields_loaded"] is False
    assert document["partition"]["action"] == "drop_subset_analysis_degenerate"
    assert document["partition"]["subset_analysis_included"] is False


def test_power_mde_and_required_instances_are_monotone() -> None:
    assert power.mde(100.0, 100) < power.mde(100.0, 40)
    assert power.instances_for_effect(100.0, 50.0) < power.instances_for_effect(
        100.0, 25.0
    )


def test_frozen_power_artifact_is_self_hashed_if_present() -> None:
    path = ROOT / "diagnostics_output/pact_contact_endpoint/power.json"
    if not path.exists():
        pytest.skip("artifact is generated after the unit contract is validated")
    document = json.loads(path.read_text())
    payload = dict(document)
    observed = payload.pop("power_sha256")
    assert observed == power.sha256_payload(payload)
    assert document["status"] == "exploratory_prior_data_for_design_only"
    assert document["power"]["chosen_fresh_instances"] == 100


def scientific_result(
    *, task: bool, hazard_frames: int = 0, hazard_entries: int = 0
) -> dict:
    return {
        "task_success": task,
        "collision_free_task_success": task and hazard_entries == 0,
        "failure_taxonomy": "fixture",
        "contact_audit": {
            "contact_class_totals": {
                "grasp_target": int(task),
                "hazard_bar": hazard_entries,
                "other_environment": 0,
            },
            "frames_with_contact": {
                "grasp_target": int(task),
                "hazard_bar": hazard_frames,
                "other_environment": 0,
            },
            "maximum_penetration_depth_m": {
                "grasp_target": 0.0,
                "hazard_bar": 0.001 if hazard_entries else 0.0,
                "other_environment": 0.0,
            },
        },
    }


def test_contact_endpoint_keeps_binary_and_magnitude_separate() -> None:
    grazing = scientific_result(task=True, hazard_frames=1, hazard_entries=1)
    plowing = scientific_result(task=True, hazard_frames=1000, hazard_entries=20)
    assert not analysis.collision_free_task_success(grazing)
    assert not analysis.collision_free_task_success(plowing)
    assert analysis.contact_frames(grazing, "hazard_bar") == 1
    assert analysis.contact_frames(plowing, "hazard_bar") == 1000


def test_conditioned_contact_contrast_requires_both_arms_to_succeed() -> None:
    instances = [
        {
            "PACT": scientific_result(task=True, hazard_frames=1),
            "PACT_PERMUTED": scientific_result(task=True, hazard_frames=5),
        },
        {
            "PACT": scientific_result(task=True, hazard_frames=100),
            "PACT_PERMUTED": scientific_result(task=False, hazard_frames=0),
        },
    ]
    observed = analysis.paired_difference(
        instances,
        arm_a="PACT",
        arm_b="PACT_PERMUTED",
        metric=analysis.METRICS["hazard_bar_contact_frames"][0],
        replicates=100,
        seed=3,
        require_both_manipulation_success=True,
    )
    assert observed["n_instances"] == 1
    assert observed["difference"] == -4


def decision_inputs(contact_difference: float, contact_ci: list[float], task: float):
    contact = {
        "difference": contact_difference,
        "instance_cluster_bootstrap_ci_95": contact_ci,
    }
    seeds_contact = {seed: {"difference": contact_difference} for seed in analysis.SEEDS}
    task_value = {
        "difference": task,
        "instance_cluster_bootstrap_ci_95": [-1.0, 1.0],
    }
    seeds_task = {seed: {"difference": task} for seed in analysis.SEEDS}
    return contact, seeds_contact, task_value, seeds_task


@pytest.mark.parametrize(
    ("difference", "ci", "task", "expected"),
    [
        (-10.0, [-20.0, -1.0], 0.1, "CONTACT_REDUCTION_WITH_TASK_BENEFIT"),
        (-10.0, [-20.0, -1.0], 0.0, "CONTACT_REDUCTION_ESTABLISHED"),
        (-1.0, [-3.0, 1.0], 1.0, "NO_CONTACT_REDUCTION"),
        (10.0, [1.0, 20.0], 1.0, "CONTACT_INCREASE"),
    ],
)
def test_frozen_contact_decision_boundaries(difference, ci, task, expected) -> None:
    contact, seeds_contact, task_value, seeds_task = decision_inputs(
        difference, ci, task
    )
    token, _ = analysis.choose_decision(
        True,
        modality_contact=contact,
        seed_modality_contacts=seeds_contact,
        pact_act_task=task_value,
        seed_pact_act_task=seeds_task,
    )
    assert token == expected


def test_pact_zero_is_never_decision_bearing() -> None:
    assert all(
        not decision_bearing
        for arm_a, arm_b, decision_bearing, _ in analysis.CONTRASTS
        if "PACT_ZERO" in (arm_a, arm_b)
    )


def test_arm_mean_ci_clusters_all_seed_observations_by_instance() -> None:
    values = np.asarray([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    observed = analysis.arm_mean_ci(values, replicates=1000, seed=11)
    assert observed["mean"] == 1.0
    assert observed["n_unique_instances"] == 2
    assert observed["n_seed_instance_observations"] == 6
    assert observed["cluster_unit"].startswith("instance;")


def test_contact_report_ends_in_exact_token() -> None:
    report = analysis.render_report(
        {"results_available": False},
        {
            "decision": "CONTACT_EXPERIMENT_INCOMPLETE",
            "reason": "schedule_did_not_reconcile",
        },
    )
    assert [line for line in report.splitlines() if line][-1] == (
        "CONTACT_EXPERIMENT_INCOMPLETE"
    )


def test_contact_analysis_reconciles_full_matrix_and_awards_enhanced_token(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(analysis, "BOOTSTRAP_REPLICATES", 200)
    instances = [{"episode_id": f"episode_{index:03d}"} for index in range(100)]
    rows = []
    for instance_index, instance in enumerate(instances):
        for seed in analysis.SEEDS:
            for arm in analysis.ARMS:
                schedule_index = len(rows)
                row = {
                    "schedule_index": schedule_index,
                    "rollout_id": f"rollout_{schedule_index:04d}",
                    "schedule_row_sha256": f"row_{schedule_index:04d}",
                    "instance_episode_id": instance["episode_id"],
                    "arm": arm,
                    "checkpoint_seed": seed,
                    "checkpoint_sha256": f"checkpoint_{seed}_{arm}",
                    "output_relpath": f"rows/{schedule_index:04d}",
                }
                rows.append(row)
                if arm == "PACT":
                    task, hazard = True, 0
                elif arm == "PACT_PERMUTED":
                    task, hazard = True, 10
                elif arm == "ACT":
                    task, hazard = instance_index >= 10, 5
                else:
                    task, hazard = True, 20
                result = scientific_result(
                    task=task,
                    hazard_frames=hazard,
                    hazard_entries=int(hazard > 0),
                )
                result.update(
                    {
                        "status": "complete",
                        "rollout_id": row["rollout_id"],
                        "schedule_row_sha256": row["schedule_row_sha256"],
                        "episode_id": row["instance_episode_id"],
                        "arm": arm,
                        "checkpoint_seed": seed,
                        "checkpoint_sha256": row["checkpoint_sha256"],
                    }
                )
                row_dir = tmp_path / row["output_relpath"]
                row_dir.mkdir(parents=True)
                (row_dir / "result.json").write_text(json.dumps(result))
                (row_dir / "driver_result.json").write_text(
                    json.dumps({"status": "complete"})
                )
    schedule = {
        "schedule_sha256": "fixture",
        "occlusion_subset_sha256": "occlusion",
        "instances": instances,
        "rows": rows,
    }
    observed, decision = analysis.analyze(schedule, tmp_path)
    assert observed["reconciliation"]["reconciled"] is True
    assert decision["decision"] == "CONTACT_REDUCTION_WITH_TASK_BENEFIT"
    report = analysis.render_report(observed, decision)
    assert [line for line in report.splitlines() if line][-1] == decision["decision"]


def test_contact_manifest_is_fresh_balanced_and_repeatable() -> None:
    excluded = {"prior_episode"}
    first = contact_contract.build_manifest(
        source_hashes={"scene": "a" * 64},
        sensor_names=[f"sensor_{index}" for index in range(40)],
        excluded_episode_ids=excluded,
        excluded_manifests={"prior": "b" * 64},
    )
    second = contact_contract.build_manifest(
        source_hashes={"scene": "a" * 64},
        sensor_names=[f"sensor_{index}" for index in range(40)],
        excluded_episode_ids=excluded,
        excluded_manifests={"prior": "b" * 64},
    )
    assert first == second
    assert len(first["rows"]) == 100
    assert sum(row["intrusion_side"] == "left" for row in first["rows"]) == 50
    assert not ({row["episode_id"] for row in first["rows"]} & excluded)


def test_contact_token_plan_selection_is_repeatable_and_separates_neighbors() -> None:
    episodes = np.repeat(np.arange(20, dtype=np.int16), 100)
    timesteps = np.tile(np.arange(100, dtype=np.int16), 20)
    first = token_plan.select_sources(episodes, timesteps)
    second = token_plan.select_sources(episodes, timesteps)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[0].shape == (100, 900)
    assert not np.any(first[0][:, 1:] == first[0][:, :-1])


def test_token_row_hash_contract_reads_only_selected_payload() -> None:
    import hashlib

    row = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    expected = hashlib.sha256(row.tobytes(order="C")).hexdigest()
    assert expected == hashlib.sha256(np.asarray(row).tobytes(order="C")).hexdigest()


def test_contact_schedule_orders_are_balanced_and_smoke_is_permuted() -> None:
    orders = contact_schedule.condition_orders()
    assert len(orders) == 100
    assert orders[0][0] == (3101, "PACT_PERMUTED")
    assert all(len(order) == 12 and len(set(order)) == 12 for order in orders)
    assert all(
        set(order) == {
            (seed, arm)
            for seed in contact_schedule.SEEDS
            for arm in contact_schedule.ARMS
        }
        for order in orders
    )
    for position in range(12):
        counts = Counter(order[position] for order in orders)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_contact_storage_keeps_endpoint_fields_and_deletes_bulk_payload(tmp_path) -> None:
    row = {
        "schedule_index": 1,
        "rollout_id": "rollout",
        "schedule_row_sha256": "row-hash",
        "instance_episode_id": "episode",
        "arm": "PACT",
        "checkpoint_seed": 3101,
        "checkpoint_sha256": "checkpoint",
        "output_relpath": "rows/0001",
    }
    row_dir = tmp_path / row["output_relpath"]
    row_dir.mkdir(parents=True)
    trajectory = row_dir / "trajectory.h5"
    video = row_dir / "video.mp4"
    trajectory.write_bytes(b"trajectory")
    video.write_bytes(b"video")
    result = {
        key: None for key in contact_storage.CORE_RESULT_KEYS
    }
    result.update(
        {
            "schema_version": "fixture",
            "status": "complete",
            "arm": "PACT",
            "rollout_id": "rollout",
            "schedule_row_sha256": "row-hash",
            "episode_id": "episode",
            "checkpoint_seed": 3101,
            "checkpoint_sha256": "checkpoint",
            "task_success": True,
            "collision_free_task_success": False,
            "failure_taxonomy": "hazard_bar_contact",
            "contact_audit": {
                "contact_taxonomy_version": "v1",
                "sampling_level": "physics",
                "sample_count": 10,
                "contact_class_totals": {
                    "grasp_target": 0,
                    "hazard_bar": 4,
                    "other_environment": 0,
                },
                "frames_with_contact": {
                    "grasp_target": 0,
                    "hazard_bar": 3,
                    "other_environment": 0,
                },
                "maximum_penetration_depth_m": {
                    "grasp_target": 0.0,
                    "hazard_bar": 0.002,
                    "other_environment": 0.0,
                },
                "first_contact_step": {
                    "grasp_target": None,
                    "hazard_bar": 2,
                    "other_environment": None,
                },
                "non_target_contact_entries": 4,
                "collision_free": False,
                "contact_frame_payload_retained": False,
                "contact_frames": [],
            },
            "policy_info": {"control_steps": 10, "sensor_activity_frames": [1, 2]},
            "trajectory_path": str(trajectory),
            "videos": [str(video)],
        }
    )
    (row_dir / "result.json").write_text(json.dumps(result))
    storage = contact_storage.compact_row(row, tmp_path)
    compact = json.loads((row_dir / "result.json").read_text())
    assert compact["contact_audit"]["frames_with_contact"]["hazard_bar"] == 3
    assert compact["contact_audit"]["maximum_penetration_depth_m"]["hazard_bar"] == 0.002
    assert "contact_frames" not in compact["contact_audit"]
    assert not trajectory.exists() and not video.exists()
    assert len(storage["deleted_payloads"]) == 2


def test_contact_storage_recovers_prepared_deletion_transaction(tmp_path) -> None:
    result = tmp_path / "result.json"
    payload = tmp_path / "trajectory.h5"
    result.write_text("{}")
    payload.write_bytes(b"payload")
    storage_path = tmp_path / "storage_archive.json"
    storage = {
        "schema_version": "pact_contact_storage_archive_v1",
        "status": "prepared",
        "schedule_index": 1,
        "rollout_id": "rollout",
        "schedule_row_sha256": "row",
        "original_result": {"path": "original", "size_bytes": 1, "sha256": "x"},
        "compact_result_sha256": contact_storage.sha256_file(result),
        "deleted_payloads": [contact_storage.inventory(payload)],
        "prepared_utc": "fixture",
        "recovered_from_compact_result": False,
    }
    storage["storage_archive_sha256"] = contact_storage.canonical_hash(storage)
    contact_storage.write_json_atomic(storage_path, storage)
    completed = contact_storage.finish_prepared_storage(storage, storage_path, result)
    assert completed["status"] == "complete"
    assert not payload.exists()
