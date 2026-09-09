"""Integration checks for live main-branch readout in the retained PACT architecture."""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import pact_v1010c_core as c
import h5py
import numpy as np
import pytest
import torch


def test_upstream_byte_identity_and_runtime_origin():
    manifest = c.verify_upstream()
    assert len(manifest['files']) >= 15
    import encoders.surface_geometry as geometry
    import scripts.pact_checkpoint as pairing
    assert Path(geometry.__file__).is_relative_to(c.MAIN)
    assert Path(pairing.__file__).is_relative_to(c.MAIN)
    assert c.forward_pass.__code__.co_filename == str(c.UPSTREAM/'act/imitate_episodes.py')
    assert c.MainEpisodicDataset.__getitem__.__code__.co_filename == str(c.UPSTREAM/'act/utils.py')


def test_original_sample_tensors_and_causal_history(monkeypatch):
    import pickle
    stats = pickle.loads((c.W/'dataset_stats.pkl').read_bytes())
    old = c.FrozenDataset([0], str(c.W/'converted'), ['wrist_camera'], stats, 100,
        init_probe=False, use_proximity=True, n_proximity_sensors=40,
        expected_proximity_encoder_sha256=c.ENCODER_SHA256, proximity_feature_dim=32)
    live = c.LiveDataset(old)
    with h5py.File(c.W/'converted/episode_0.hdf5') as h:
        raw = h['observations/proximity'][()]
    original_choice=np.random.choice
    for t in (0,1,7,8,len(raw)-1):
        monkeypatch.setattr(np.random, 'choice', lambda *a, **kw:t)
        before, after = old[0], live[0]
        assert all(torch.equal(a,b) for a,b in zip(before[:4],after[:4]))
        live_frames = np.stack([c.stack_obs_proximity(dict(zip(c.CANONICAL_SENSOR_NAMES,frame)),
            list(c.CANONICAL_SENSOR_NAMES), pool='min') for frame in raw[:t+1]])
        reference = c.causal_pooled_window(live_frames,t)
        assert np.array_equal(after[4].numpy(), reference)
        assert after[4].shape == (8,40,8,8)
    # Reset means the new episode's first frame supplies all initial padding.
    assert np.array_equal(c.causal_pooled_window(raw[-1:].min(axis=2),0),
                          np.repeat(raw[-1:].min(axis=2),8,axis=0))
    monkeypatch.setattr(np.random,'choice',original_choice)
    np.random.seed(3103);before=old[0];old_rng=np.random.get_state()
    np.random.seed(3103);after=live[0];live_rng=np.random.get_state()
    assert all(torch.equal(a,b) for a,b in zip(before[:4],after[:4]))
    assert old_rng[0]==live_rng[0] and np.array_equal(old_rng[1],live_rng[1])
    assert old_rng[2:]==live_rng[2:]


def test_encoder_initialization_gradient_and_actual_policy_loss():
    encoder = c.make_encoder()
    payload = torch.load(c.ENCODER_PATH,map_location='cpu',weights_only=False)
    assert all(torch.equal(v.cpu(),payload['model_state_dict'][k])
               for k,v in encoder.inner.state_dict().items())
    train, val, stats, meta = c.make_loaders()
    from utils import set_seed
    set_seed(3103)
    policy = c.make_policy()
    optimizer = c.add_main_encoder_group(policy.configure_optimizers(),encoder)
    sample = tuple(t.unsqueeze(0) for t in train.dataset[0])
    old = encoder.inner.conv_stem[0].weight.detach().clone()
    before_aux = {k:v.detach().clone() for k,v in encoder.inner.state_dict().items()
                  if any(s in k for s in ('embedding_head','auxiliary_head','reconstruction_head'))}
    policy.train();encoder.train()
    losses = c.forward_pass(sample,c.PolicyCallAdapter(policy),encoder)
    assert set(losses)=={'l1','kl','loss'}
    losses['loss'].backward()
    assert torch.isfinite(encoder.inner.conv_stem[0].weight.grad).all()
    assert torch.count_nonzero(encoder.inner.conv_stem[0].weight.grad)>0
    optimizer.step();optimizer.zero_grad()
    assert not torch.equal(old,encoder.inner.conv_stem[0].weight)
    assert all(torch.equal(v,encoder.inner.state_dict()[k]) for k,v in before_aux.items())
    assert tuple(policy.model.input_proj_proximity.weight.shape)==(512,128)
    encoder.eval();policy.eval()
    with torch.inference_mode():
        tokens = c.encode_for_act(encoder,sample[4].cuda())
        assert tokens.shape==(1,40,128) and torch.isfinite(tokens).all()
        # Same function and same causal inputs at train/eval produce exact eval tokens.
        assert torch.equal(tokens,encoder.encode_pooled_history(sample[4].cuda()))
    del policy,optimizer,encoder
    torch.cuda.empty_cache()


def test_joint_encoder_optimizer_rng_resume_equivalence():
    import imitate_episodes as trainer
    from utils import set_seed
    set_seed(17)
    encoder = c.make_encoder()
    head = torch.nn.Linear(128,8).cuda()
    optimizer = torch.optim.AdamW(head.parameters(),lr=1e-5)
    c.add_main_encoder_group(optimizer,encoder)
    x = torch.linspace(.01,.25,2*8*3*8*8,device='cuda').reshape(2,8,3,8,8)
    def update():
        optimizer.zero_grad()
        loss = head(c.encode_for_act(encoder,x)).square().mean()
        loss.backward();optimizer.step()
    update()
    saved = copy.deepcopy({'encoder':encoder.state_dict(),'head':head.state_dict(),
                           'optimizer':optimizer.state_dict(),'rng':trainer._rng_state()})
    update()
    expected_encoder = copy.deepcopy(encoder.state_dict())
    expected_head = copy.deepcopy(head.state_dict())
    expected_rng = torch.get_rng_state().clone()
    encoder.load_state_dict(saved['encoder']);head.load_state_dict(saved['head'])
    optimizer.load_state_dict(saved['optimizer']);trainer._restore_rng_state(saved['rng'])
    update()
    assert all(torch.equal(v,encoder.state_dict()[k]) for k,v in expected_encoder.items())
    assert all(torch.equal(v,head.state_dict()[k]) for k,v in expected_head.items())
    assert torch.equal(expected_rng,torch.get_rng_state())
    del optimizer,encoder,head
    torch.cuda.empty_cache()


def test_upstream_checkpoint_pair_refuses_missing_or_mismatched(tmp_path):
    from scripts.pact_checkpoint import paired_encoder_checkpoint,file_digest
    config={'finetune_prox_encoder':True}
    policy=tmp_path/'policy_last.ckpt';encoder=tmp_path/'prox_encoder.pt'
    policy.write_bytes(b'policy')
    with pytest.raises(ValueError,match='Missing matching'):
        paired_encoder_checkpoint(tmp_path,config,'policy_last.ckpt')
    encoder.write_bytes(b'encoder')
    index={'policy_last.ckpt':{'encoder':encoder.name,'policy_sha256':file_digest(policy),
                               'encoder_sha256':file_digest(encoder)}}
    (tmp_path/'checkpoint_pairs.json').write_text(json.dumps(index))
    assert paired_encoder_checkpoint(tmp_path,config,'policy_last.ckpt')==encoder
    encoder.write_bytes(b'wrong update')
    with pytest.raises(ValueError,match='hash mismatch'):
        paired_encoder_checkpoint(tmp_path,config,'policy_last.ckpt')


def test_actual_live_history_adapter_matches_training():
    import collections
    import pact_v1010c_eval as evaluator
    evaluator.ACTIVE={'variant':'readout60000'}
    encoder=c.make_encoder(train=False)
    policy=evaluator.ReadoutInferencePolicy.__new__(evaluator.ReadoutInferencePolicy)
    policy._surface_encoder=encoder
    policy._readout_history=collections.deque(maxlen=8)
    policy._readout_frames=0
    with h5py.File(c.W/'converted/episode_0.hdf5') as h:
        raw=h['observations/proximity'][:10]
    for index,frame in enumerate(raw):
        policy._step=index
        observed=policy._surface_positions(frame)
        history=c.causal_pooled_window(raw[:index+1].min(axis=2),index)
        with torch.inference_mode():
            expected=c.encode_for_act(encoder,torch.from_numpy(history).cuda().unsqueeze(0))
        assert torch.equal(observed,expected)
    assert evaluator.ReadoutInferencePolicy.model_output_to_action is evaluator.v109.PactPlaceV109InferencePolicy.model_output_to_action
    del policy,encoder
    torch.cuda.empty_cache()
