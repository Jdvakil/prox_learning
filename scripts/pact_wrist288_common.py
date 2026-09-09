"""New-run-only contracts for the V10.10 wrist288 three-seed experiment."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/prox_learning_pact_remediation')
WORK = ROOT / 'diagnostics_output/pact_place_v1010_wrist288_s3_v1'
CODE = ROOT / 'scripts'
NAMESPACE = 'pact_place_v1010_wrist288_s3_v1'
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    'OMP_THREAD_LIMIT', 'BLIS_NUM_THREADS', 'RAYON_NUM_THREADS')}
os.environ.update(THREAD_ENV)
for path in (ROOT, CODE, ROOT / 'submodules/act', ROOT / 'submodules/molmospaces'):
    sys.path.insert(0, str(path))
from pact_place_v1010_contract import cells, cell_key, SCENE_BY_POSE, SAMPLER_CLASS, ACTIVE_CLUTTER_SLOTS
from pact_place_v109_contract import CANONICAL_SENSOR_NAMES, SENSOR_ORDER_SHA256, ENCODER_PATH, ENCODER_SHA256, empty_authorization
SEEDS = (3103, 3104, 3105)
DEADLINE = datetime(2026, 9, 8, tzinfo=timezone.utc).timestamp()
SAFE_STOP = DEADLINE - 3600

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)

def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f'immutable artifact mismatch: {path}')
    else:
        atomic(path, value)

def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())

def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s] if Path(path).exists() else []

def environment():
    env = dict(os.environ, **THREAD_ENV)
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', MUJOCO_GL='egl',
        PYOPENGL_PLATFORM='egl', MLSPACES_ASSETS_DIR=str(ROOT/'assets'),
        PACT_CONTACT_AUDIT_SUMMARY_ONLY='1', PACT_V109_TRAJECTORY_H5_ONLY='1',
        PACT_WRIST288_OWNER=NAMESPACE,
        PYTHONPATH=os.pathsep.join(map(str,(CODE,ROOT/'submodules/act',ROOT/'submodules/molmospaces',ROOT))))
    env.pop('DISPLAY', None)
    return env

def pin_torch():
    import torch
    if torch.get_num_threads()!=1: torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)

def derive(role, cell, ordinal, retry=0, collision=0):
    identity = [NAMESPACE, role, cell, int(ordinal), int(retry), int(collision)]
    value = int(digest(identity)[:16], 16)
    return {'seed_u32':value % 2**32, 'seed_u64':value, 'derivation':identity}

def check_bindings():
    config = read(WORK/'config.json')
    assert config['config_sha256'] == digest({k:v for k,v in config.items() if k!='config_sha256'})
    for path, expected in config['file_hashes'].items():
        if sha(ROOT/path) != expected:
            raise RuntimeError(f'frozen input drift: {path}')
    assert sha(ENCODER_PATH) == ENCODER_SHA256
    assert sha(WORK/'seed_registry.json')==config['seed_registry_sha256']
    assert sha(WORK/'historical_seeds.json')==config['historical_seed_inventory_sha256']
    for entry in config['manifests'].values():
        assert sha(WORK/entry['path'])==entry['sha256']
    return config

# Spawned renderer workers re-import their main module without calling main().
# Configure their PyTorch pools at import time too; no CUDA context is created.
if os.environ.get('PACT_WRIST288_OWNER')==NAMESPACE:
    pin_torch()
