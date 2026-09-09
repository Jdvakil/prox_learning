import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pact_wrist280_eval_capacity as capacity


def put(path, value='data'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def test_cache_excludes_live_files_models_and_external_symlinks(tmp_path, monkeypatch):
    work = tmp_path / 'work'
    monkeypatch.setattr(capacity, 'WORK', work)
    raw = put(work / 'raw/done/trajectory.h5')
    put(raw.parent / 'exit_receipt.json', '{}')
    put(work / 'raw/live/trajectory.h5')
    converted = put(work / 'converted/episode_0.h5')
    put(work / 'processes/conversion/exit_receipt.json', '{"returncode":0}')
    evaluated = put(work / 'evaluation/stage/done/trajectory.h5')
    put(evaluated.parent / 'exit_receipt.json', '{}')
    put(work / 'evaluation/stage/live/trajectory.h5')
    put(work / 'checkpoints/model.h5')
    external = put(tmp_path / 'external.h5')
    (work / 'converted/external.h5').symlink_to(external)
    assert set(capacity.cache_candidates()) == {raw, converted, evaluated}
    put(work / 'processes/conversion/exit_receipt.json', '{"returncode":1}')
    assert set(capacity.cache_candidates()) == {raw, evaluated}


def test_rollouts_start_at_twelve_and_backoff_waits_for_drain(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, 'WORK', tmp_path)
    monkeypatch.setattr(capacity, 'STATUS', tmp_path / 'status.json')
    monkeypatch.setattr(capacity.recovery, 'WORK', tmp_path)
    monkeypatch.setattr(capacity.recovery.original, 'WORK', tmp_path)
    release = Mock()
    monkeypatch.setattr(capacity, 'release_closed_cache', release)
    pool = capacity.CapacityPool('development_h10')
    assert pool.limit == 12
    release.assert_called_once()
    pool.active = {i: object() for i in range(12)}
    reason = 'three consecutive resource pressure samples'
    pool.reduce(reason)
    assert pool.limit == 10 and pool.pause_reason is None
    pool.reduce(reason)
    assert pool.pause_reason is None
    pool.active = {i: object() for i in range(10)}
    pool.reduce(reason)
    assert pool.pause_reason == reason
    assert capacity.CapacityPool('train_budget_60000', 1).limit == 1


def test_eta_uses_actual_fallback_and_keeps_deadline_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, 'WORK', tmp_path)
    monkeypatch.setattr(capacity, 'STATUS', tmp_path / 'status.json')
    monkeypatch.setattr(capacity.time, 'time', lambda: 0)
    monkeypatch.setattr(capacity, 'SAFE_STOP', 200000)
    put(tmp_path / 'training_benchmark.json', '{"chosen_updates_per_second":10}')
    put(tmp_path / 'checkpoints/act/progress.json', '{"updates":60000}')
    put(tmp_path / 'evaluation/dev_ledger.jsonl', '{"valid_completion":true,"elapsed_s":1200}\n')
    put(capacity.STATUS, '{"limit":12}')
    twelve = capacity.remaining_eta()
    assert twelve['remaining_estimate_hours'] == pytest.approx((30000 + 403 * 100) / 3600)
    put(capacity.STATUS, '{"limit":10}')
    ten = capacity.remaining_eta()
    assert ten['remaining_estimate_hours'] > twelve['remaining_estimate_hours']
    monkeypatch.setattr(capacity, 'SAFE_STOP', 1)
    with pytest.raises(capacity.recovery.original.Paused, match='no longer fits'):
        capacity.remaining_eta()
