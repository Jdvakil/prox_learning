import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pact_wrist280_final_cache_recovery as recovery


def put(path, content='data'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_hdf5_and_completed_training_are_covered_but_live_inputs_are_kept(tmp_path, monkeypatch):
    work = tmp_path / 'work'
    monkeypatch.setattr(recovery, 'WORK', work)
    converted = put(work / 'converted/episode_0.hdf5')
    put(work / 'processes/conversion/exit_receipt.json', '{"returncode":0}')
    put(work / 'checkpoints/done/progress.json', '{"updates":60000}')
    released = {put(work / 'checkpoints/done' / name) for name in
                ('policy_best.ckpt', 'policy_last.ckpt', 'policy_update_30000.ckpt', 'resume_bundle.ckpt')}
    put(work / 'checkpoints/done/policy_update_60000.ckpt')
    put(work / 'checkpoints/live/progress.json', '{"updates":30000}')
    put(work / 'checkpoints/live/resume_bundle.ckpt')
    outside = put(tmp_path / 'outside.hdf5')
    (work / 'converted/link.hdf5').symlink_to(outside)
    assert set(recovery.extra_cache_candidates()) == released | {converted}
    put(work / 'processes/conversion/exit_receipt.json', '{"returncode":1}')
    assert set(recovery.extra_cache_candidates()) == released


def test_original_closed_h5_files_are_still_included(monkeypatch):
    monkeypatch.setattr(recovery, 'prior_candidates', lambda: iter(['raw.h5', 'trajectory.h5']))
    monkeypatch.setattr(recovery, 'extra_cache_candidates', lambda: iter(['converted.hdf5', 'resume_bundle.ckpt']))
    assert list(recovery.candidates()) == ['raw.h5', 'trajectory.h5', 'converted.hdf5', 'resume_bundle.ckpt']
