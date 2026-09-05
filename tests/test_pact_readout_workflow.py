"""Readout runs must preserve gradients, temporal inputs and checkpoint pairs."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'submodules/act')]

from pact import train_command
from pact_checkpoint import paired_encoder_checkpoint
from eval_pact import configure_proximity


def readout_config():
    return dict(prox_feature='surface_embedding', finetune_prox_encoder=True,
                prox_policy_tap='readout', prox_feat_dim=128, n_proximity_sensors=40,
                prox_tokens_per_sensor=1, prox_layout='per_sensor',
                proximity_layout='raw_causal', prox_pool='min')


def test_wrapper_uses_learned_readout_and_explicit_encoder(tmp_path):
    encoder = tmp_path / 'initial.pt'
    encoder.touch()
    args = SimpleNamespace(dataset='v12', run='v12_readout_test', arm='readout',
                           batch_size=8, seed=2, epochs=10, lr=1e-5,
                           encoder_checkpoint=str(encoder), encoder_lr=2e-6)
    command = train_command(args, {'profile': {'chunk_size': 50}})
    for key, value in {'--prox_feature': 'surface_embedding', '--prox_policy_tap': 'readout',
                       '--prox_tokens_per_sensor': '1', '--prox_encoder_ckpt': str(encoder),
                       '--prox_encoder_lr': '2e-06', '--wandb_run_name': args.run}.items():
        assert command[command.index(key) + 1] == value
    assert '--finetune_prox_encoder' in command
    args.arm = 'raw'
    with pytest.raises(ValueError, match='require --arm readout'):
        train_command(args, {'profile': {'chunk_size': 50}})
    args.arm = 'readout'
    encoder.unlink()
    with pytest.raises(ValueError, match='Missing pretrained'):
        train_command(args, {'profile': {'chunk_size': 50}})


def test_finetuned_weights_never_fall_back_to_initial_or_wrong_epoch(tmp_path):
    cfg = readout_config()
    (tmp_path / 'pretrained.pt').touch()
    cfg['prox_encoder_ckpt'] = str(tmp_path / 'pretrained.pt')
    with pytest.raises(ValueError, match='Missing matching'):
        paired_encoder_checkpoint(tmp_path, cfg)
    (tmp_path / 'prox_encoder_best.pt').touch()
    assert paired_encoder_checkpoint(tmp_path, cfg).name == 'prox_encoder_best.pt'
    with pytest.raises(ValueError, match='Missing matching'):
        paired_encoder_checkpoint(tmp_path, cfg, 'policy_epoch_100_seed_0.ckpt')
    (tmp_path / 'prox_encoder_epoch_100_seed_0.pt').touch()
    assert paired_encoder_checkpoint(tmp_path, cfg, 'policy_epoch_100_seed_0.ckpt').name == 'prox_encoder_epoch_100_seed_0.pt'


def test_adapter_requires_live_128d_readout_and_selected_pair(tmp_path):
    cfg = readout_config()
    path = tmp_path / 'prox_config.json'
    path.write_text(json.dumps(cfg))
    (tmp_path / 'prox_encoder_best.pt').touch()
    pc = SimpleNamespace()
    configure_proximity(pc, tmp_path, 'policy_best.ckpt', {'prox_pool': 'min'})
    assert pc.finetune_prox_encoder and pc.prox_policy_tap == 'readout'
    assert pc.prox_encoder_ckpt == str(tmp_path / 'prox_encoder_best.pt')
    cfg['prox_feat_dim'] = 32
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='prox_feat_dim'):
        configure_proximity(pc, tmp_path, 'policy_best.ckpt', {'prox_pool': 'min'})


def policy_methods():
    """Exercise real policy methods without importing simulator/GPU dependencies."""
    from encoders.pact import is_geometry_feature
    tree = ast.parse((ROOT / 'submodules/act/eval_act_obstacle.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ACTInferencePolicy')
    names = {'reset', 'record_proximity_observation', 'inference_model',
             'needs_fresh_camera_observation', 'needs_fresh_policy_observation',
             'needs_fresh_proximity_observation'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {'np': np, 'is_geometry_feature': is_geometry_feature,
          'stack_obs_proximity': lambda obs, order, pool: obs['skin']}
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<policy-methods>', 'exec'), ns)
    return type('Policy', (), {name: ns[name] for name in names})


def test_cached_actions_still_collect_consecutive_history_and_reset():
    from encoders.pact import causal_pooled_window
    policy = policy_methods()()
    policy.pc = SimpleNamespace(temp_agg_off=True, use_proximity=True, prox_feature='surface_embedding')
    policy._policy, policy._stats = object(), {}
    policy._prox_encoder = SimpleNamespace(sensor_order=['s'])
    policy._prox_pool = 'min'
    policy._pending_chunks = []
    policy.reset()
    chunk = np.zeros((50, 8), dtype=np.float32)
    policy._pending_chunks = [(0, chunk)]
    data = np.arange(51, dtype=np.float32)[:, None, None, None] * np.ones((51, 1, 8, 8), np.float32)
    for t in range(50):
        policy._step = t
        assert not policy.needs_fresh_camera_observation()
        assert policy.needs_fresh_proximity_observation()
        np.testing.assert_equal(policy.inference_model({'skin': data[t]}), chunk[t])
        # A repeated accessor in one step must not duplicate the frame.
        policy.record_proximity_observation({'skin': data[t]})
        hist = np.stack(policy._prox_hist)
        padded = np.concatenate([np.repeat(hist[:1], 8 - len(hist), axis=0), hist])
        np.testing.assert_equal(padded, causal_pooled_window(data, t))
    policy._step = 50
    assert policy.needs_fresh_camera_observation()
    policy.record_proximity_observation({'skin': data[50]})
    np.testing.assert_equal(np.stack(policy._prox_hist), data[43:51])
    policy.reset()
    assert policy._prox_hist == [] and policy._prox_hist_step is None


def test_policy_pair_writer_detects_replaced_encoder(tmp_path):
    import torch
    from encoders.surface_geometry import SurfaceEmbeddingEncoder, SurfaceGeometryEncoder
    # Load only save helpers; importing the trainer would import sim_env/W&B.
    tree = ast.parse((ROOT / 'submodules/act/imitate_episodes.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('_save_prox_encoder', '_save_policy_pair')]
    ns = dict(torch=torch, Path=Path, json=json)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<checkpoint-writers>', 'exec'), ns)
    enc = SurfaceGeometryEncoder(kind='embedding', inner=SurfaceEmbeddingEncoder(),
                                 frozen=False, policy_tap='readout')
    for name in ('policy_best.ckpt', 'policy_last.ckpt', 'policy_epoch_100_seed_0.ckpt'):
        ns['_save_policy_pair'](tmp_path / name, {'weight': torch.ones(1)}, enc)
        paired_encoder_checkpoint(tmp_path, readout_config(), name)
    (tmp_path / 'prox_encoder_best.pt').write_bytes(b'wrong encoder')
    with pytest.raises(ValueError, match='hash mismatch'):
        paired_encoder_checkpoint(tmp_path, readout_config())
