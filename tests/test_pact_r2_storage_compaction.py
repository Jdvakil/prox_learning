from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_pact_confirmatory as analyzer
import compact_pact_r2_storage as compactor


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _row(index: int) -> dict:
    return {
        "schedule_index": index,
        "rollout_id": f"rollout-{index}",
        "schedule_row_sha256": f"schedule-row-{index}",
        "instance_episode_id": index // 6,
        "arm": "PACT",
        "checkpoint_sha256": "checkpoint-sha",
        "checkpoint_seed": 3101,
        "output_relpath": f"rows/{index:03d}",
    }


def _result(row: dict, trajectory_path: Path, videos: list[Path]) -> dict:
    return {
        "schema_version": "synthetic",
        "status": "complete",
        "arm": row["arm"],
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "episode_id": row["instance_episode_id"],
        "candidate_index": 0,
        "row_sha256": "candidate-row",
        "manifest_sha256": "manifest",
        "intrusion_side": "left",
        "sampling_retry_index": 0,
        "sampling_retry_history": [],
        "seed": 123,
        "checkpoint_seed": row["checkpoint_seed"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "stats_sha256": "stats",
        "surface_encoder_sha256": "encoder",
        "attempt_index": 0,
        "inflight_recovery_event_sha256": None,
        "abandoned_payload_archive": None,
        "initial_observation_accepted": True,
        "initial_observation_boundary_sha256": "boundary",
        "task_success": True,
        "collision_free_task_success": True,
        "failure_taxonomy": "none",
        "contact_audit": {
            "collision_free": True,
            "contact_class_totals": {
                "grasp_target": 3,
                "hazard_bar": 0,
                "other_environment": 0,
            },
            "contact_frames": [{"large": "x" * 100_000}],
            "sample_count": 100,
        },
        "policy_info": {
            "arm": row["arm"],
            "control_steps": 100,
            "gripper_close_commanded": True,
            "proximity_consumed_for_action": True,
            "proximity_zeroed_for_action": False,
            "sensor_activity_frames": [{"large": "y" * 100_000}],
        },
        "trajectory_path": str(trajectory_path),
        "videos": [str(path) for path in videos],
    }


def _make_completed_row(tmp_path: Path, index: int = 1):
    row = _row(index)
    row_dir = tmp_path / row["output_relpath"]
    row_dir.mkdir(parents=True)
    trajectory = row_dir / "trajectory.h5"
    trajectory_bytes = (b"trajectory-block-" * 16_384) + b"end"
    trajectory.write_bytes(trajectory_bytes)
    videos = [row_dir / "camera.mp4", row_dir / "depth.mp4"]
    for number, video in enumerate(videos):
        video.write_bytes((f"video-{number}".encode()) * 100)
    result = _result(row, trajectory, videos)
    result_bytes = (
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"
    )
    (row_dir / "result.json").write_bytes(result_bytes)
    (row_dir / "driver_result.json").write_text(
        json.dumps({"status": "complete"})
    )
    return row, row_dir, result_bytes, trajectory_bytes, videos


def test_compaction_archives_restore_originals_and_preserves_analyzer_view(
    tmp_path,
):
    row, row_dir, result_bytes, trajectory_bytes, videos = (
        _make_completed_row(tmp_path)
    )
    video_hashes = {path: _sha(path.read_bytes()) for path in videos}

    observed = compactor.compact_row(
        output_root=tmp_path,
        row=row,
        threads=1,
        level=1,
    )

    assert observed["status"] == "compacted"
    assert not (row_dir / "trajectory.h5").exists()
    assert compactor.decompressed_sha256(
        row_dir / "result.full.json.zst"
    ) == _sha(result_bytes)
    assert compactor.decompressed_sha256(
        row_dir / "trajectory.h5.zst"
    ) == _sha(trajectory_bytes)
    compact = json.loads((row_dir / "result.json").read_text())
    assert compact["contact_audit"]["contact_class_totals"] == {
        "grasp_target": 3,
        "hazard_bar": 0,
        "other_environment": 0,
    }
    assert "contact_frames" not in compact["contact_audit"]
    assert "sensor_activity_frames" not in compact["policy_info_summary"]
    assert compact["storage_compaction"]["outcome_based_selection"] is False
    for video in videos:
        assert _sha(video.read_bytes()) == video_hashes[video]
    results, reconciliation = analyzer._load_results(
        {"rows": [row]}, tmp_path
    )
    assert reconciliation["reconciled"] is True
    assert len(results) == 1
    verified = compactor.verify_compacted(row_dir, row)
    assert verified["original_payloads_recoverable"] is True


def test_interrupted_publication_is_recovered_idempotently(tmp_path):
    row, row_dir, _, _, _ = _make_completed_row(tmp_path)
    compactor.compact_row(
        output_root=tmp_path,
        row=row,
        threads=1,
        level=1,
    )
    manifest_path = row_dir / "storage_archive.json"
    manifest_path.unlink()

    recovered = compactor.compact_row(
        output_root=tmp_path,
        row=row,
        threads=1,
        level=1,
    )
    repeated = compactor.compact_row(
        output_root=tmp_path,
        row=row,
        threads=1,
        level=1,
    )

    assert recovered["status"] == "interrupted_compaction_recovered"
    assert repeated["status"] == "already_compacted_verified"
    assert manifest_path.exists()


@pytest.mark.parametrize("index", sorted(compactor.EXCLUDED_SCHEDULE_INDICES))
def test_declared_exclusions_remain_intact(tmp_path, index):
    row, row_dir, result_bytes, trajectory_bytes, _ = _make_completed_row(
        tmp_path, index=index
    )
    observed = compactor.compact_row(
        output_root=tmp_path,
        row=row,
        threads=1,
        level=1,
    )
    assert observed["status"] == "excluded_intact"
    assert (row_dir / "result.json").read_bytes() == result_bytes
    assert (row_dir / "trajectory.h5").read_bytes() == trajectory_bytes
    assert not (row_dir / "storage_archive.json").exists()


def test_frozen_real_amendment_binds_schedule_and_compactor():
    config, schedule = compactor.validate_inputs(
        config_path=ROOT / "configs/pact_r2_storage_amendment_v1.json",
        schedule_path=(
            ROOT / "diagnostics_output/pact_vs_act_r2/schedule.json"
        ),
        output_root=Path(
            "/root/pact_remediation_artifacts_v2/confirmatory_r2_35e1377c"
        ),
    )
    assert schedule["schedule_sha256"] == config["schedule_sha256"]
    assert config["excluded_intact_schedule_indices"] == [0, 959]
    assert config["provenance"]["endpoint_outcomes_inspected_before_amendment"] is False
    assert all(
        value is False
        for value in config["frozen_scientific_contract"].values()
    )
