from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_pact_collision_collection",
        ROOT / "scripts" / "run_pact_collision_collection.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = _load()


def _row():
    return {"episode_id": "e" * 64, "row_sha256": "a" * 64}


def test_terminal_result_is_identity_checked(tmp_path):
    path = tmp_path / "result.json"
    payload = {**_row(), "status": "task_failure"}
    path.write_text(json.dumps(payload))
    assert runner._terminal_result(path, _row()) == payload

    payload["row_sha256"] = "b" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="row hash"):
        runner._terminal_result(path, _row())


def test_nonterminal_result_is_rejected(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({**_row(), "status": "running"}))
    with pytest.raises(RuntimeError, match="non-terminal"):
        runner._terminal_result(path, _row())


def test_status_counts_do_not_treat_unknown_as_terminal():
    results = [
        {"status": "success"},
        {"status": "task_failure"},
        {"status": "unreconciled_worker_failure"},
    ]
    counts = runner._result_counts(results)
    assert counts["success"] == 1
    assert counts["task_failure"] == 1
    assert sum(counts.values()) == 2


def test_construction_retry_precedes_terminal_rollout_boundary():
    source = (ROOT / "scripts" / "run_pact_collision_collection.py").read_text()
    reset_at = source.index("initial_reset_result = task.reset()")
    marker_at = source.index("_write_json_atomic(", reset_at)
    boundary_at = source.index("rollout_started = True", marker_at)
    rollout_at = source.index(
        "ParallelRolloutRunner.run_single_rollout(", boundary_at
    )
    assert reset_at < marker_at < boundary_at < rollout_at
    assert runner.BOUNDARY_FILENAME == "initial_observation_accepted.json"
    assert "pre_rollout_construction_failure" in source[reset_at:marker_at]
    assert "initial_reset_result=initial_reset_result" in source[rollout_at:]


def test_canonical_runner_can_consume_prevalidated_reset():
    source = (
        ROOT
        / "submodules"
        / "molmospaces"
        / "molmo_spaces"
        / "data_generation"
        / "pipeline.py"
    ).read_text()
    assert "initial_reset_result=None" in source
    assert "observation, _info = initial_reset_result" in source


def test_runtime_assets_and_isolated_output_are_separate_and_recorded():
    source = (ROOT / "scripts" / "run_pact_collision_collection.py").read_text()
    assert "runtime config did not retain the isolated collection output" in source
    assert "collection output must stay inside the isolated worktree" in source
    assert '"runtime_assets_dir"' in source
    assert '"isolated_output_dir"' in source


def test_resumed_collection_skips_terminal_rows_before_worker_dispatch():
    source = (ROOT / "scripts" / "run_pact_collision_collection.py").read_text()
    prefilter_at = source.index("pending_rows: list[dict[str, Any]]")
    executor_at = source.index("ProcessPoolExecutor(", prefilter_at)
    assert prefilter_at < executor_at
    assert "for row in pending_rows" in source[executor_at:]
    assert "max_tasks_per_child=MAX_TASKS_PER_CHILD" in source[executor_at:]
    assert runner.MAX_TASKS_PER_CHILD == 8
