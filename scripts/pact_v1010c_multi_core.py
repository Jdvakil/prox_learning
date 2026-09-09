"""Adapters around byte-identical main-branch encoder and fine-tuning code."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import os
import pickle
import sys
from pathlib import Path

from pact_wrist288_common import (ROOT, WORK as W, CODE, THREAD_ENV, ENCODER_PATH,
    ENCODER_SHA256, CANONICAL_SENSOR_NAMES, atomic, append, digest, freeze, lines,
    now, pin_torch, read, sha, environment as old_environment)

TRAINING_SEED = int(os.environ['PACT_FINETUNE_SEED'])
assert TRAINING_SEED in (3104, 3105)
PREVIOUS = ROOT / 'diagnostics_output/pact_place_v1010c_readout_s3103'
MULTI = ROOT / 'diagnostics_output/pact_place_v1010c_readout_s3_v1'
C = MULTI / f'seed{TRAINING_SEED}'
UPSTREAM = PREVIOUS / 'upstream'
MAIN = UPSTREAM / 'prox_learning'
SCHEMA = f'pact_place_v1010c_readout_s{TRAINING_SEED}_v1'
os.environ.update(THREAD_ENV)
sys.path.insert(0, str(MAIN))
pin_torch()
import h5py
import numpy as np
import torch
from encoders.pact import build_pact_encoder, encode_for_act, causal_pooled_window
from encoders.peak_closeness import stack_obs_proximity
from encoders.surface_geometry import save_encoder_checkpoint
from utils import EpisodicDataset as FrozenDataset


def environment():
    env = old_environment()
    env.pop('PACT_WRIST288_OWNER', None)
    env['PACT_READOUT_OWNER'] = SCHEMA
    env['PYTHONPATH'] = os.pathsep.join(map(str, (CODE, MAIN, ROOT / 'submodules/act',
                                               ROOT / 'submodules/molmospaces', ROOT)))
    return env


def verify_upstream():
    doc = read(C / 'upstream_manifest.json')
    for row in doc['files']:
        assert row['byte_identical'] and sha(ROOT / row['copied_path']) == row['sha256'], row['path']
    return doc


def exact_symbol(file, name, namespace=None):
    """Compile an unchanged upstream definition, excluding unrelated entry points."""
    source = Path(file).read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name == name)
    text = ''.join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    target = globals() if namespace is None else namespace
    exec(compile(text, str(file), 'exec'), target)
    return target[name]


# These definitions are executed directly from the unmodified main-branch files.
verify_upstream()
MainEpisodicDataset = exact_symbol(UPSTREAM / 'act/utils.py', 'EpisodicDataset')
blur_images = exact_symbol(UPSTREAM / 'act/imitate_episodes.py', 'blur_images')
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
dropout_modalities = exact_symbol(UPSTREAM / 'act/imitate_episodes.py', 'dropout_modalities')
forward_pass = exact_symbol(UPSTREAM / 'act/imitate_episodes.py', 'forward_pass')
save_prox_encoder = exact_symbol(UPSTREAM / 'act/imitate_episodes.py', '_save_prox_encoder')


class LiveDataset(MainEpisodicDataset):
    """Keep our fixed loader construction; execute main's exact sample method."""
    def __init__(self, original):
        self.__dict__.update(original.__dict__)
        self.dataset_dir = str(C / 'data_views')
        self.load_proximity = True
        self.proximity_layout = 'raw_causal'
        self.proximity_feature_dim = 128


class PolicyCallAdapter:
    """The old policy lacks main's optional image-dropout keyword (disabled here)."""
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, *args, image_dropped=None, **kwargs):
        assert image_dropped is None, 'modality dropout is outside this comparison'
        return self.policy(*args, **kwargs)


def add_main_encoder_group(optimizer, encoder, lr=1e-5):
    """Execute main's original optimizer-registration block without rewriting it."""
    file = UPSTREAM / 'act/imitate_episodes.py'
    source = file.read_text()
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_bc')
    node = next(n for n in function.body if isinstance(n, ast.If)
                and isinstance(n.test, ast.Name) and n.test.id == 'finetune_prox')
    code = ''.join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    import textwrap
    exec(compile(textwrap.dedent(code), str(file), 'exec'),
         {'finetune_prox': True, 'prox_encoder': encoder, 'optimizer': optimizer,
          'config': {'prox_encoder_lr': lr, 'lr': 1e-5}})
    return optimizer


def policy_config():
    config = dict(read(W / 'checkpoints/pact_seed3103/run_manifest.json')['policy_config'])
    assert config['proximity_feature_dim'] == 32
    config['proximity_feature_dim'] = 128
    return config


def make_policy():
    import imitate_episodes as original
    return original.make_policy('ACT', policy_config(), argv_guard=original._detr_safe_argv(
        'ACT', 'pact_v1010c_readout', TRAINING_SEED, 2000, str(C / 'checkpoint'))).cuda()


def make_encoder(checkpoint=None, train=True):
    checkpoint = checkpoint or read(C / 'initialization.json')['path']
    encoder = build_pact_encoder('surface_embedding', checkpoint=checkpoint,
        device='cuda', layout='per_sensor', tokens_per_sensor=1, frozen=False, policy_tap='readout')
    # Shared sensor processing is unchanged; our policy retains its original slot order.
    encoder.sensor_order = list(CANONICAL_SENSOR_NAMES)
    assert encoder.act_feat_dim == 128 and encoder.n_act_sensors == 40
    encoder.train(train)
    return encoder


def make_loaders():
    from fixed_split_data import load_split_manifest
    from pact_fixed_split_data import load_pact_fixed_split_data
    split = load_split_manifest(W / 'split_manifest.json')
    train, val, stats, meta = load_pact_fixed_split_data(str(W / 'converted'), split,
        ['wrist_camera'], 8, 8, 100, seed=TRAINING_SEED, num_workers=4, n_proximity_sensors=40,
        proximity_feature_dim=32, expected_encoder_sha256=ENCODER_SHA256)
    shared = pickle.loads((W / 'dataset_stats.pkl').read_bytes())
    assert stats.keys() == shared.keys() and all(np.array_equal(stats[k], shared[k]) for k in stats)
    for loader in (train, val):
        object.__setattr__(loader, 'dataset', LiveDataset(loader.dataset))
    assert len(train) == 30 and len(val) == 5
    assert len(train.dataset) == 240 and len(val.dataset) == 40
    return train, val, stats, meta


def prepare_data():
    """New min-pooled data with exact links to original images/actions/state."""
    source = read(W / 'conversion_manifest.json')
    directory = C / 'data_views'
    directory.mkdir(exist_ok=True)
    entries = []
    for ep in source['episodes']:
        original = W / 'converted' / ep['act_file']
        assert sha(original) == ep['act_file_sha256']
        destination = directory / ep['act_file']
        with h5py.File(original) as src:
            raw = src['observations/proximity'][()]
            assert raw.ndim == 5 and raw.shape[1:] == (40, 4, 8, 8)
            # Execute main's exact per-frame pooling helper and preserve slot order.
            pooled = np.stack([stack_obs_proximity(dict(zip(CANONICAL_SENSOR_NAMES, frame)),
                list(CANONICAL_SENSOR_NAMES), pool='min') for frame in raw])
            assert np.array_equal(pooled, raw.min(axis=2)) and np.isfinite(pooled).all()
            if not destination.exists():
                temporary = destination.with_suffix('.partial.hdf5')
                assert not temporary.exists()
                with h5py.File(temporary, 'x') as out:
                    for key, value in src.attrs.items():
                        out.attrs[key] = value
                    out.attrs['v1010c_proximity_pool'] = 'min'
                    for key in src.keys():
                        if key != 'observations':
                            out[key] = h5py.ExternalLink(str(original), '/' + key)
                    observations = out.create_group('observations')
                    for key in src['observations'].keys():
                        if key != 'proximity':
                            observations[key] = h5py.ExternalLink(str(original), '/observations/' + key)
                    observations.create_dataset('proximity', data=pooled, chunks=(1, 40, 8, 8),
                                                compression='gzip', compression_opts=4, shuffle=True)
                os.replace(temporary, destination)
            with h5py.File(destination) as out:
                assert np.array_equal(out['observations/proximity'][()], pooled)
                for key in ('action', 'observations/qpos', 'observations/qvel',
                            'observations/images', 'observations/proximity_sensor_names'):
                    link = out.get(key, getlink=True)
                    assert isinstance(link, h5py.ExternalLink) and link.filename == str(original)
        entries.append({'file':ep['act_file'], 'source_sha256':ep['act_file_sha256'],
                        'view_sha256':sha(destination), 'shape':list(pooled.shape)})
    freeze(C / 'data_views_manifest.json', {'episodes':entries, 'count':len(entries),
        'source_manifest_sha256':sha(W / 'conversion_manifest.json'),
        'pool':'min', 'sensor_order':list(CANONICAL_SENSOR_NAMES),
        'actions_images_states':'External links to original byte-identical arrays'})
    return len(entries)


def safe_torch_save(value, path):
    path = Path(path)
    assert path.is_relative_to(C)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_pair(policy, encoder, directory, step):
    """Publish exact main-format encoder and policy together with update hashes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    policy_path = directory / 'policy_last.ckpt'
    encoder_path = directory / 'prox_encoder.pt'
    safe_torch_save(policy.state_dict(), policy_path)
    temporary = encoder_path.with_suffix('.partial.pt')
    save_prox_encoder(temporary, encoder)
    os.replace(temporary, encoder_path)
    record = {'encoder':encoder_path.name, 'policy_sha256':sha(policy_path),
              'encoder_sha256':sha(encoder_path), 'global_step':step}
    atomic(directory / 'checkpoint_pairs.json', {'policy_last.ckpt':record})
    return record


def load_pair(directory, expected_step=60000):
    directory = Path(directory)
    # Execute main's unchanged missing-pair/hash checks before any model load.
    from scripts.pact_checkpoint import paired_encoder_checkpoint
    path = paired_encoder_checkpoint(directory, {'finetune_prox_encoder':True}, 'policy_last.ckpt')
    record = read(directory / 'checkpoint_pairs.json')['policy_last.ckpt']
    assert record['global_step'] == expected_step
    policy = make_policy()
    policy.load_state_dict(torch.load(directory / 'policy_last.ckpt', map_location='cuda', weights_only=False), strict=True)
    encoder = make_encoder(path, train=False)
    policy.eval()
    return policy, encoder, record


if __name__ == '__main__':
    print(json.dumps({'data_views':prepare_data()}), flush=True)
