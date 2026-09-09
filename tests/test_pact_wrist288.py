"""Scientific-contract checks before the new wrist288 run consumes compute."""
import copy
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pact_wrist288_common import *
import numpy as np
import h5py
from pact_wrist288_data import forward_controls,split_rows,verify_ledger
from pact_wrist288_analysis import rgb_difference,select_history,validate_completions
from pact_wrist288_prepare import allocate,build_row

def raw_fixture(path,terminal=True):
    def encoded(values):
        return np.asarray([np.frombuffer(json.dumps(v).encode().ljust(300,b'\0'),dtype=np.uint8) for v in values])
    actions=[{'arm':[i/10]*7,'gripper':[255 if i%2 else 0]} for i in range(4)]
    with h5py.File(path,'w') as h:
        g=h.create_group('traj_0')
        g.create_dataset('actions/commanded_action',data=encoded([{},actions[1],actions[2],actions[2] if terminal else actions[3]]))
        g.create_dataset('actions/joint_pos',data=encoded([actions[0],actions[1],actions[2],{} if terminal else actions[3]]))
        for name in ('qpos','qvel'):
            g.create_dataset('obs/agent/'+name,data=encoded([{'arm':[i/100]*7,'gripper':[0.003,0.003]} for i in range(4)]))
        g.create_dataset('terminated',data=[False,False,False,terminal]);g.create_dataset('truncated',data=[False]*4)
    return path

def test_forward_alignment_inspects_status_not_fixed_tail(tmp_path):
    for terminal,expected in [(True,2),(False,3)]:
        with h5py.File(raw_fixture(tmp_path/f'{terminal}.h5',terminal)) as h:
            action,qpos,qvel,audit=forward_controls(h['traj_0'])
            assert len(action)==expected and action[0,0]==np.float32(.1) and qpos[0,0]==0
            assert action[0,7]==255 and action[1,7]==0
            assert audit['command_indices']==list(range(1,expected+1))

def test_forward_missing_fields_fail(tmp_path):
    path=raw_fixture(tmp_path/'raw.h5')
    with h5py.File(path,'a') as h:del h['traj_0/actions/commanded_action']
    with h5py.File(path) as h:
        with pytest.raises(KeyError):forward_controls(h['traj_0'])

def test_nonterminal_hole_fails(tmp_path):
    path=raw_fixture(tmp_path/'raw.h5')
    with h5py.File(path,'a') as h:h['traj_0/terminated'][-1]=False
    with h5py.File(path) as h:
        with pytest.raises(AssertionError,match='nonterminal'):forward_controls(h['traj_0'])

@pytest.fixture(scope='module')
def registry():return allocate({'seed_u32_values':[derive('collection',cell_key(*cells()[0]),0)['seed_u32']]})

def test_seed_roles_retries_historical_collision_and_extras(registry):
    seeds=[s['seed_u32'] for values in registry['entries'].values() for s in values]
    assert len(seeds)==len(set(seeds)) and len(registry['collisions_rejected'])>=1
    assert len(set(registry['extras']))==6
    for seed in SEEDS:assert len([k for k in registry['entries'] if k.startswith(f'final_{seed}:')])==50
    assert sum(k.startswith('smoke:') for k in registry['entries'])==4
    assert sum(k.startswith('development:') for k in registry['entries'])==24

def ledger_rows(registry):
    rows=[]
    for cell in map(lambda c:cell_key(*c),cells()):
        for ordinal in range(12):
            row=build_row('collection',cell,ordinal,registry)
            rows.append({'row':row,'cell':cell,'attempt_id':row['attempt_id'],'row_sha256':row['row_sha256'],'accepted':True,'returncode':0})
    return rows

def test_quotas_deterministic_split_and_identity(registry):
    rows=ledger_rows(registry);assert len(verify_ledger(rows))==288
    validation=split_rows(rows,registry)
    assert len(validation)==48 and validation==split_rows(list(reversed(rows)),registry)
    for cell in {r['cell'] for r in rows}:
        assert sum(r['attempt_id'] in validation for r in rows if r['cell']==cell)==2
    assert rows[0]['row']['task_sampler_class']==SAMPLER_CLASS
    assert rows[0]['row']['pact_v1010_active_clutter_slots']==['01','03','04','06']
    with pytest.raises(AssertionError):verify_ledger(rows[:-1])
    with pytest.raises(AssertionError):verify_ledger(rows+[rows[0]])

def test_rgb_tolerance_boundaries():
    a=np.zeros((100,100,3),np.uint8);b=a.copy();b.flat[:30]=2
    assert rgb_difference(a,b)[0]['passed']
    b.flat[30]=1;assert not rgb_difference(a,b)[0]['passed']
    b=a.copy();b.flat[0]=3;assert not rgb_difference(a,b)[0]['passed']
    with pytest.raises(AssertionError):rgb_difference(a,b.astype(float))

def test_combined_history_selection_and_gate():
    rows=[]
    for history in (100,10):
        for arm in ('ACT','PACT'):
            count={(100,'ACT'):6,(100,'PACT'):15,(10,'ACT'):12,(10,'PACT'):13}[history,arm]
            rows.extend({'averaging_history':history,'arm':arm,'task_success':i<count,
                'collision_free_task_success':i<count,'hazard_frames':0} for i in range(24))
    selected=select_history(rows)
    assert selected['history']==10 and not selected['passed'] # advantage cannot drive selection
    for row in rows:
        row.update(task_success=False,collision_free_task_success=False)
    assert select_history(rows)['history']==100

def test_false_completion_fails_without_receipts():
    with pytest.raises(AssertionError):validate_completions([], [{'schedule':{'rollout_id':'one'}}])
    with pytest.raises(AssertionError):validate_completions([{'rollout_id':'one','returncode':1,'valid_completion':True}], [{'schedule':{'rollout_id':'one'}}])

def test_encoder_scene_bindings_and_sensor_order():
    assert sha(ENCODER_PATH)==ENCODER_SHA256
    assert len(CANONICAL_SENSOR_NAMES)==40 and digest(list(CANONICAL_SENSOR_NAMES))==SENSOR_ORDER_SHA256
    for entry in SCENE_BY_POSE.values():assert sha(ROOT/entry['relative'])==entry['sha256']
    from surface_proximity_encoder import load_frozen_surface_embedding_encoder
    model,payload=load_frozen_surface_embedding_encoder(ENCODER_PATH,map_location='cpu')
    assert payload['policy_feature_dim']==32

def test_sim_loader_no_second_shift_and_train_only_stats(tmp_path):
    from fixed_split_data import compute_train_only_norm_stats
    from utils import EpisodicDataset
    for i in range(3):
        with h5py.File(tmp_path/f'episode_{i}.hdf5','w') as h:
            h.attrs['sim']=True
            h.create_dataset('action',data=np.full((4,8),i*10,dtype=np.float32)+np.arange(4)[:,None])
            for name in ('qpos','qvel'):h.create_dataset('observations/'+name,data=np.full((4,9),i,dtype=np.float32))
            h.create_dataset('observations/images/wrist_camera',data=np.zeros((4,2,2,3),np.uint8))
    stats,meta=compute_train_only_norm_stats(tmp_path,[0,1])
    assert meta['contributing_trajectories']==2 and not meta['validation_included']
    assert np.allclose(stats['action_mean'],6.5)
    dataset=EpisodicDataset([0],str(tmp_path),['wrist_camera'],stats,100,init_probe=False)
    np.random.seed(7);expected=np.random.choice(4);np.random.seed(7)
    _,_,action,pad=dataset[0]
    assert np.allclose(action[0].numpy()*stats['action_std']+stats['action_mean'],expected)
    assert int((~pad).sum())==4-expected

def test_resume_optimizer_rng_equivalence():
    import torch
    import imitate_episodes as historical
    torch.manual_seed(17);np.random.seed(17)
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.3),torch.nn.Linear(4,2))
    opt=torch.optim.Adam(model.parameters(),lr=1e-5)
    def step():
        opt.zero_grad();loss=model(torch.randn(2,3)).square().mean();loss.backward();opt.step()
    for _ in range(3):step()
    saved=copy.deepcopy({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':historical._rng_state()})
    for _ in range(3):step()
    uninterrupted=copy.deepcopy(model.state_dict())
    model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);historical._restore_rng_state(saved['rng'])
    for _ in range(3):step()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in uninterrupted.items())
    assert {int(v['step']) for v in opt.state_dict()['state'].values()}=={6}

def test_bundle_step_accounting():
    import torch
    from pact_wrist288_train import validate_bundle
    b={'wrist288_config_sha256':'x','global_step':30,'epoch':0,'optimizer_state':{'state':{0:{'step':torch.tensor(30)}}},'model_state':{'x':torch.ones(2)}}
    assert validate_bundle(b,'x')==30
    b['global_step']=29
    with pytest.raises(AssertionError):validate_bundle(b,'x')

def test_matching_arm_training_flags(tmp_path,monkeypatch):
    import pact_wrist288_train as train
    from pact_place_v109_contract import command_diff
    monkeypatch.setattr(train,'WORK',tmp_path)
    atomic(tmp_path/'split_manifest.json',{'split_manifest_sha256':'s'})
    atomic(tmp_path/'conversion_manifest.json',{'converted_tree_file_sha256':'t','timesteps':{'converted_t_max':640}})
    a,p=[train.command(arm,3103,900) for arm in ('act','pact')]
    assert command_diff(a,p)['identical_except_allowance']
    assert a[a.index('--episode_horizon')+1]=='648'
    assert a[a.index('--num_epochs')+1]=='2000'

def test_averaging_matches_age_weight_for_both_histories(monkeypatch):
    # Execute the actual inherited inference with fake model output; this checks
    # adapter trimming together with the original age-weighted implementation.
    import inspect
    import eval_pact_collision_row as legacy
    source=inspect.getsource(legacy.PactCollisionInferencePolicy.inference_model)
    assert 'weights.append(np.exp(-0.01 * age))' in source
    worker_source=(CODE/'pact_wrist288_eval_worker.py').read_text()
    assert 'self._pending_chunks[-(HISTORY-1):]' in worker_source
    for history in (100,10):
        pending=[]
        for t in range(150):
            pending=pending[-(history-1):]
            pending.append((t,np.full((100,8),t,dtype=np.float32)))
            pending=[(start,value) for start,value in pending if t-start<len(value)]
            weights=np.exp(-.01*np.array([t-s for s,_ in pending]));weights/=weights.sum()
            actual=(np.stack([v[t-s] for s,v in pending])*weights[:,None]).sum(0)
            ages=np.arange(min(t+1,history));expected=np.average(t-ages,weights=np.exp(-.01*ages))
            assert np.allclose(actual,expected)
