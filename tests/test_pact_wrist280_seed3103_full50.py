import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pact_wrist280_seed3103_full50 as extension


def jobs(tmp_path):
    return [{'directory': str(tmp_path / f'{i}_{arm}'),
             'command': ['python', 'original_worker', '--manifest', 'original_manifest'],
             'schedule': {'episode_id': str(i), 'rollout_id': f'{i}_{arm}', 'arm': arm,
                          'checkpoint_seed': 3103, 'averaging_history': 100}}
            for i in range(50) for arm in ('ACT', 'PACT')]


def test_full_generator_retains_original_instances_and_only_changes_adapter_prefix(tmp_path, monkeypatch):
    original = jobs(tmp_path)
    monkeypatch.setattr(extension, 'BASE_JOBS', lambda role, seed, history: json.loads(json.dumps(original)))
    actual = extension.full_jobs()
    assert len(actual) == 100
    for before, after in zip(original, actual):
        assert before['directory'] == after['directory']
        assert before['schedule'] == after['schedule']
        assert after['command'][4:] == before['command'][2:]


def test_adoption_requires_exact_prior_jobs_and_observed_zero_exits(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(extension, 'WORK', tmp_path)
    all_jobs = jobs(tmp_path)
    prior = all_jobs[:34]
    extension.atomic(tmp_path / 'evaluation/final_3103_h100_schedule.json', prior)
    for job in prior:
        directory = Path(job['directory'])
        extension.atomic(directory / 'job.json', job)
        extension.atomic(directory / 'exit_receipt.json', {'returncode': 0, 'job_sha256': extension.digest(job)})
    assert extension.verify_prior_jobs(all_jobs) == prior
    bad = Path(prior[0]['directory']) / 'exit_receipt.json'
    extension.atomic(bad, {'returncode': 1, 'job_sha256': extension.digest(prior[0])})
    with pytest.raises(AssertionError):
        extension.verify_prior_jobs(all_jobs)


def test_extended_deadline_applies_to_supervisor_and_monitor_and_preserves_original_contract(monkeypatch):
    from types import SimpleNamespace
    import pact_wrist288_common as common
    original_stop = common.SAFE_STOP
    monitor = SimpleNamespace(SAFE_STOP=original_stop, DEADLINE=common.DEADLINE)
    run = SimpleNamespace(SAFE_STOP=original_stop, DEADLINE=common.DEADLINE)
    installed = (None, None, monitor, None, run)
    monkeypatch.setattr(extension, 'verify_extension', lambda: {})
    monkeypatch.setattr(extension.recovery, 'install', lambda: installed)
    assert extension.install() == installed
    assert monitor.DEADLINE == run.DEADLINE == common.DEADLINE + 3 * 3600
    assert monitor.SAFE_STOP == run.SAFE_STOP == monitor.DEADLINE - 3600
    assert common.SAFE_STOP == original_stop == extension.DEADLINE - 60 * 60


def test_eta_allows_final_batch_to_finish_after_launch_stop_but_refuses_late_starts(tmp_path, monkeypatch):
    import pytest
    from datetime import datetime, timezone
    monkeypatch.setattr(extension, 'WORK', tmp_path)
    monkeypatch.setattr(extension, 'lines', lambda path: [
        {'rollout_id': str(i), 'valid_completion': True, 'elapsed_s': 1200}
        for i in range(100)])
    monkeypatch.setattr(extension, 'append', lambda *args: None)
    pending = [{'directory': str(tmp_path / str(i))} for i in range(66)]
    at = datetime(2026, 9, 8, 0, 10, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(extension.time, 'time', lambda: at)
    result = extension.remaining_eta(pending)
    assert datetime.fromisoformat(result['estimated_last_launch_utc']).timestamp() < extension.LAUNCH_STOP
    assert at + result['remaining_estimate_hours'] * 3600 > extension.LAUNCH_STOP
    assert at + result['remaining_estimate_hours'] * 3600 < extension.EXTENDED_DEADLINE
    monkeypatch.setattr(extension.time, 'time', lambda: at + 20 * 60)
    with pytest.raises(extension.recovery.capacity.recovery.original.Paused):
        extension.remaining_eta(pending)
