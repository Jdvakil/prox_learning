import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import pact_v1010c_run as run


def test_retained_reconstruction_accepts_documented_render_difference():
    run.replay_check(run.read(run.C / 'evaluation_manifest.json')['scenes'][0])


@pytest.mark.parametrize('fault', ['prediction', 'proximity', 'source_hash'])
def test_reconstruction_rejects_unexplained_difference(monkeypatch, fault):
    original = run.read
    scene = original(run.C / 'evaluation_manifest.json')['scenes'][0]

    def corrupted(path):
        data = original(path)
        if Path(path).name == 'frozen_query_reconstruction.json':
            data = copy.deepcopy(data)
            if fault == 'prediction':
                data['comparisons']['new']['first_vector_exact'] = False
            elif fault == 'proximity':
                data['inputs']['new']['proximity_sha256'] = 'different'
            else:
                key = next(iter(data['source_hashes']))
                data['source_hashes'][key] = 'different'
        return data

    monkeypatch.setattr(run, 'read', corrupted)
    with pytest.raises(AssertionError):
        run.replay_check(scene)
