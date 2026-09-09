"""V10.11c owner-requested experiment; historical artifacts stay read-only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'scripts', ROOT / 'submodules/act', ROOT / 'submodules/molmospaces'):
    sys.path.insert(0, str(path))

THREAD_ENV = {key: '1' for key in (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_THREAD_LIMIT',
    'BLIS_NUM_THREADS', 'RAYON_NUM_THREADS')}
os.environ.update(THREAD_ENV)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from pact_place_v109_contract import (CANONICAL_SENSOR_NAMES, SENSOR_ORDER_SHA256,
    ENCODER_PATH, ENCODER_SHA256, empty_authorization, sha256_file)
import pact_place_v1011c_collection_contract as collection

WORK = ROOT / 'diagnostics_output/pact_place_v1011c_train_eval'
DATA = ROOT / 'assets/act_style_data/pact_place_v1011c_99'
TRAIN = WORK / 'checkpoints'
EVAL = ROOT / 'diagnostics_output/pact_place_v1011c_eval'
SEEDS = (3103, 3104, 3105)
SPLIT_SEED = 2026090401
EVAL_MASTER_SEED = 2026090402
SMOKE_MASTER_SEED = 2026090403
RETRY_MASTER_SEED = 2026090404


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f'refusing to overwrite {path}')
    else:
        with path.open('x') as stream:
            stream.write(text)


def environment():
    env = dict(os.environ)
    env.update(THREAD_ENV)
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', MUJOCO_GL='egl',
               PYOPENGL_PLATFORM='egl', MLSPACES_ASSETS_DIR=str(ROOT / 'assets'),
               PACT_CONTACT_AUDIT_SUMMARY_ONLY='1', PACT_V109_TRAJECTORY_H5_ONLY='1')
    env.pop('DISPLAY', None)
    env['PYTHONPATH'] = os.pathsep.join(str(ROOT / p) for p in (
        'scripts', 'submodules/act', 'submodules/molmospaces'))
    return env


def retry_seed(row, retry_index):
    raw = hashlib.sha256(f"{RETRY_MASTER_SEED}:{row['episode_id']}:{retry_index}".encode()).digest()
    value = int.from_bytes(raw[:8], 'big')
    return {'seed_u32': value % 2**32, 'seed_u64': value}


def load_eval_manifest(path):
    doc = read(path)
    assert doc['schema_version'] == 'pact_place_v1011c_paired_eval_v1'
    assert doc['manifest_sha256'] == digest({k:v for k,v in doc.items() if k != 'manifest_sha256'})
    for row in doc['rows'] + doc['smoke']['rows']:
        assert row['task_sampler_class'] == collection.SAMPLER_CLASS
        assert row['row_sha256'] == digest({k:v for k,v in row.items() if k != 'row_sha256'})
    return doc
