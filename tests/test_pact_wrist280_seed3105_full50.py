import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pact_wrist280_seed3105_full50 as extension


def jobs(tmp_path):
    return [{'directory': str(tmp_path / f'{i}_{arm}'),
             'command': ['python', 'original_worker', '--manifest', 'original_manifest'],
             'schedule': {'episode_id': str(i), 'rollout_id': f'{i}_{arm}', 'arm': arm,
                          'checkpoint_seed': 3105, 'averaging_history': 100}}
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
    prior = all_jobs[::6][:16] + all_jobs[1::6][:16]
    extension.atomic(tmp_path / 'evaluation/final_3105_h100_schedule.json', prior)
    for job in prior:
        directory = Path(job['directory'])
        extension.atomic(directory / 'job.json', job)
        extension.atomic(directory / 'exit_receipt.json', {'returncode': 0, 'job_sha256': extension.digest(job)})
    assert extension.verify_prior_jobs(all_jobs) == prior
    bad = Path(prior[0]['directory']) / 'exit_receipt.json'
    extension.atomic(bad, {'returncode': 1, 'job_sha256': extension.digest(prior[0])})
    with pytest.raises(AssertionError):
        extension.verify_prior_jobs(all_jobs)
