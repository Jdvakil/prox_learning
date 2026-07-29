from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "submodules" / "act"
MOLMO = ROOT / "submodules" / "molmospaces"
for path in (ACT, MOLMO, ROOT / "scripts"):
    sys.path.insert(0, str(path))


def _load():
    spec = importlib.util.spec_from_file_location(
        "eval_pact_collision_row", ACT / "eval_pact_collision_row.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluation = _load()


def test_failure_taxonomy_exempts_target_contact():
    audit = {
        "contact_class_totals": {
            "grasp_target": 4,
            "hazard_bar": 0,
            "other_environment": 0,
        }
    }
    assert (
        evaluation.failure_taxonomy(
            task_success=True,
            contact_audit=audit,
            gripper_close_commanded=True,
        )
        == "collision_free_task_success"
    )


def test_failure_taxonomy_prioritizes_non_target_contact():
    audit = {
        "contact_class_totals": {
            "grasp_target": 2,
            "hazard_bar": 1,
            "other_environment": 9,
        }
    }
    assert (
        evaluation.failure_taxonomy(
            task_success=True,
            contact_audit=audit,
            gripper_close_commanded=True,
        )
        == "hazard_bar_contact"
    )


def test_evaluator_marks_boundary_after_reset_and_before_any_action():
    source = (ACT / "eval_pact_collision_row.py").read_text()
    model_at = source.index("policy.prepare_model()")
    reset_at = source.index("initial_reset_result = task.reset()")
    boundary_at = source.index(
        "_write_json_atomic(boundary_path, boundary)", reset_at
    )
    rollout_at = source.index(
        "ParallelRolloutRunner.run_single_rollout(", boundary_at
    )
    assert model_at < reset_at < boundary_at < rollout_at
    assert "initial_reset_result=initial_reset_result" in source[rollout_at:]
    assert (
        "existing initial observation requires a frozen R2 group-recovery event"
        in source
    )


def test_r2_recovery_archives_abandoned_payload_without_deleting_it(tmp_path):
    (tmp_path / "trajectory.h5").write_bytes(b"trajectory")
    (tmp_path / "episode_00000000_wrist_camera.mp4").write_bytes(b"video")
    event = {
        "recovery_event_sha256": "event",
        "rows": [{"rollout_id": "row", "attempt_index": 3}],
    }
    record = evaluation.archive_abandoned_payload(
        tmp_path, event, rollout_id="row"
    )
    assert record is not None
    archive = tmp_path / "abandoned_payload_attempt_003"
    assert not (tmp_path / "trajectory.h5").exists()
    assert (archive / "trajectory.h5").read_bytes() == b"trajectory"
    assert (
        archive / "episode_00000000_wrist_camera.mp4"
    ).read_bytes() == b"video"
    manifest = json.loads((archive / "archive_manifest.json").read_text())
    assert manifest["payload_deleted"] is False
