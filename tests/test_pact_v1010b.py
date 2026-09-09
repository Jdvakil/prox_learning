"""Mechanism, control invariance, fail-closed completion and storage fixtures."""
from __future__ import annotations
import ast
import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import sys
import tempfile
import time
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pact_v1010b_contract import *
os.environ.update(environment())
pin_torch()
import h5py
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R
from pact_v1010b_dataset import AcquisitionWindowDataset,choose_start
from pact_v1010b_metrics import frame_world,longest,duration,recompute
from pact_v1010b_pairing import audit_initial_group
from pact_v1010b_storage import verify_h5_equal,repack
from pact_wrist288_analysis import rgb_difference
import utils


@pytest.fixture
def tmp():
    B.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=B,prefix='unit_') as p:yield Path(p)


def test_source_compiles_and_no_baseline_writer():
    for path in CODE.glob('pact_v1010b_*.py'):
        compile(path.read_text(),str(path),'exec')
    from pact_v1010b_eval_worker import verify_bindings,instrument_limit
    assert verify_bindings()[4]=='PactFrontendScreenInferencePolicy'
    assert instrument_limit(None)==300 and instrument_limit(120)==180 and instrument_limit(400)==300
    assert not read(A/'evidence.json')['new_rollouts']


def test_effective280_training_binding_and_recovery_state():
    from pact_v1010b_train import assert_nested_exact
    binding=training_config_binding()
    assert binding['config_sha256']!=read(W/'config.json')['config_sha256']
    for seed in SEEDS:
        for arm in ('act','pact'):
            assert read(W/f'checkpoints/{arm}_seed{seed}/progress.json')['config_sha256']==binding['config_sha256']
    value={'optimizer':{'step':torch.tensor(60000.),'momentum':torch.arange(4.)},
        'rng':('numpy',np.arange(4,dtype=np.uint32),[torch.arange(5,dtype=torch.uint8)])}
    assert_nested_exact(value,copy.deepcopy(value))
    changed=copy.deepcopy(value);changed['optimizer']['momentum'][2]+=1
    with pytest.raises(AssertionError):assert_nested_exact(value,changed)
    changed=copy.deepcopy(value);changed['rng'][1][1]+=1
    with pytest.raises(AssertionError):assert_nested_exact(value,changed)


def test_staged_storage_credit_preserves_both_reserves():
    from pact_v1010b_run import staged_storage_check
    assert staged_storage_check(49*GIB,40*GIB,1.25*GIB,5*GIB,12)['passed']
    assert not staged_storage_check(49*GIB,40*GIB,0,5*GIB,12)['passed']
    assert not staged_storage_check(12*GIB,1*GIB,1.25*GIB,1*GIB,12)['passed']
    with pytest.raises(AssertionError):staged_storage_check(49*GIB,40*GIB,2*GIB,5*GIB,12)


def test_frames_local_mount_and_next_index():
    rotation=R.from_euler('xyz',[90,0,90],degrees=True);q=rotation.as_quat()[[3,0,1,2]]
    base=np.r_[[2.,3.,4.],q];local=np.array([.4,.1,.2]);mount=np.array([0.,0.,.35])
    assert np.allclose(frame_world(local,base,mount),rotation.apply(local+mount)+base[:3])
    assert not np.allclose(frame_world(local,base,mount),rotation.apply(local)+base[:3]+mount)
    positions=np.arange(901);command=np.arange(900)
    assert np.array_equal(positions[1:]-command,np.ones(900))


def test_cross24_seed_is_independent_of_checkpoint():
    rows=select_cross24();assert len(rows)==24
    assert collections.Counter(r['scene_block'] for r in rows)=={3103:7,3104:11,3105:6}
    for row in rows:
        assert 'source_seed' not in row['physical_row'] and digest(row['physical_row'])==row['physical_row_digest']
    args=('development_v1',3103,rows[0]['physical_row']['cell'],0,0)
    seed=derive_scene(*args)
    assert seed==derive_scene(*args) and seed!=derive_scene('final_v1',*args[1:])


def fixture_dataset(tmp,proximity=False):
    T=120;stats={'qpos_mean':np.arange(9,dtype=np.float32),'qpos_std':np.arange(9,dtype=np.float32)+1,
        'action_mean':np.arange(8,dtype=np.float32),'action_std':np.arange(8,dtype=np.float32)+1}
    action=np.arange(T*8,dtype=np.float32).reshape(T,8);action[:,7]=np.where(np.arange(T)>=50,255.,0.)
    with h5py.File(tmp/'episode_0.hdf5','w') as h:
        h.attrs['sim']=True;h.attrs['pact_surface_encoder_sha256']=ENCODER_SHA256
        h.create_dataset('action',data=action);h.create_dataset('observations/qpos',data=np.arange(T*9,dtype=np.float32).reshape(T,9))
        h.create_dataset('observations/qvel',data=np.zeros((T,9),np.float32))
        h.create_dataset('observations/images/wrist_camera',data=np.arange(T*4*5*3,dtype=np.uint8).reshape(T,4,5,3))
        h.create_dataset('observations/proximity_embeddings',data=np.arange(T*40*32,dtype=np.float32).reshape(T,40,32))
    dataset=utils.EpisodicDataset([0],tmp,['wrist_camera'],stats,100,init_probe=False,use_proximity=proximity,
        n_proximity_sensors=40 if proximity else 0,proximity_feature_dim=32,expected_proximity_encoder_sha256=ENCODER_SHA256)
    labels=[{'act_episode_index':0,'episode_id':'fixture','T':T,'close':50}]
    return dataset,labels,action


@pytest.mark.parametrize('proximity',[False,True])
def test_p0_all_tensors_and_rng_identical(tmp,proximity):
    dataset,labels,actions=fixture_dataset(tmp,proximity)
    wrapped=AcquisitionWindowDataset.wrap(dataset,labels,3104,0.)
    for seed in range(20):
        np.random.seed(seed);before=np.random.get_state();expected=dataset[0];after=np.random.get_state()
        np.random.set_state(before);actual=wrapped[0];newstate=np.random.get_state()
        assert all(torch.equal(a,b) for a,b in zip(expected,actual))
        assert after[0]==newstate[0] and np.array_equal(after[1],newstate[1]) and after[2:]==newstate[2:]
        # Undo normalization for all eight forward-aligned labels and check tail masking.
        np.random.set_state(before);start=int(np.random.choice(120));n=min(100,120-start)
        recovered=actual[2]*dataset.norm_stats['action_std']+dataset.norm_stats['action_mean']
        assert np.allclose(recovered[:n].numpy(),actions[start:start+n],atol=1e-4)
        assert actual[3][:n].sum()==0 and actual[3][n:].all()


def test_candidate_only_sampling_distribution_and_arm_match():
    np.random.seed(44);uniform=[choose_start(500,150,'x',2000,i,0.,training_seed=3103) for i in range(20000)]
    state=np.random.get_state();np.random.seed(44)
    candidate=[choose_start(500,150,'x',2000,i,.25,training_seed=3103) for i in range(20000)]
    assert np.array_equal(state[1],np.random.get_state()[1]) and state[2:]==np.random.get_state()[2:]
    u=np.mean((np.array(uniform)>=120)&(np.array(uniform)<=180));c=np.mean((np.array(candidate)>=120)&(np.array(candidate)<=180))
    assert abs(u-61/500)<.02 and abs(c-(.25+.75*61/500))<.02
    np.random.seed(44);other=[choose_start(500,150,'x',2000,i,.25,training_seed=3103) for i in range(20000)]
    assert other==candidate


def test_original_aggregation_zero_chunks_expiry_decoder():
    source=(ROOT/'submodules/act/eval_pact_frontend_screen_row.py').read_text()
    body=source[source.index('        self._pending_chunks.append'):source.index('\n    def get_info',source.index('        self._pending_chunks.append'))]
    ns={'np':np};exec('def aggregate(self, chunk):\n'+body,ns)
    policy=SimpleNamespace(_step=0,_pending_chunks=[])
    for step in range(130):
        policy._step=step;policy._pending_chunks=policy._pending_chunks[-99:]
        chunk=np.full((100,8),step,np.float32) if step!=37 else np.zeros((100,8),np.float32)
        result=ns['aggregate'](policy,chunk);starts=np.arange(max(0,step-99),step+1)
        expected=np.average(np.where(starts==37,0,starts),weights=np.exp(-.01*(step-starts)))
        assert np.allclose(result,expected,atol=1e-5)
    from eval_pact_collision_row import PactCollisionInferencePolicy
    import hashlib
    p=SimpleNamespace(_model_output_trace_sha256=hashlib.sha256(),_model_output_trace_steps=0,_gripper_close_commanded=False)
    for score,expected in [(127.499,0),(127.5,255),(127.501,255)]:
        assert PactCollisionInferencePolicy.model_output_to_action(p,np.r_[np.zeros(7),score])['gripper'][0]==expected


def test_contact_union_continuity_and_terminal_convention():
    hazard=np.array([True,True,False]);clutter=np.array([False,True,True])
    assert (hazard|clutter).sum()==3 and hazard.sum()+clutter.sum()==4
    assert longest([1,1,0,1,1],.25)==.5
    left=np.array([1,0,1,0],bool);right=~left;assert longest(left&right,.5)==0
    fixed=np.array([[1,0],[1,0],[0,1],[0,1]],bool)
    assert longest(fixed.any(1),.5)==2. and max(longest(fixed[:,i],.5) for i in range(2))==1.
    assert duration([1,1,1],[0,.002,.004])==.004


def test_false_grasp_counterexamples_and_no_runtime_gate():
    evidence=read(A/'evidence.json');rows=evidence['rollouts']
    assert sum(r['success'] and r.get('observable_close_gap_gate',False) and r['arm']=='PACT' for r in rows)==66
    assert any(r['held'] and not r['lift'] and not r['success'] for r in rows)
    source=(CODE/'pact_v1010b_eval_worker.py').read_text()
    assert '127.5' not in inspect.getsource(__import__('pact_v1010b_eval_worker').Wrist1010bInferencePolicy.inference_model)


def write_initial(path,rgb=None,physical=1,root_attr=1):
    path.mkdir(parents=True,exist_ok=True)
    with h5py.File(path/'initial_observation.h5','w') as h:
        h.attrs['physical']=root_attr
        h.create_dataset('physics/qpos',data=np.array([physical],np.float64));h.create_dataset('physics/qvel',data=np.zeros(1))
        h.create_dataset('model/geom_pos',data=np.zeros((1,3)))
        h.create_dataset('configuration/row',data=np.bytes_('row'))
        h.create_dataset('observation/wrist_camera',data=np.zeros((20,50,3),np.uint8) if rgb is None else rgb)


def pairrow(path,model):
    return {'scene_id':'scene','physical_row_digest':'row','task_seed':{'seed_u32':1,'seed_u64':1},
        'directory':str(path),'provenance':{'training_seed':model,'checkpoint_sha256':str(model)}}


def test_pair_model_allowed_physics_and_rgb_rejected(tmp):
    a=tmp/'a';b=tmp/'b';write_initial(a);write_initial(b)
    assert audit_initial_group([pairrow(a,3103),pairrow(b,3105)],tmp/'good.json')['passed']
    write_initial(b,physical=2)
    with pytest.raises(AssertionError):audit_initial_group([pairrow(a,3103),pairrow(b,3105)],tmp/'bad.json')
    write_initial(b,root_attr=2)
    with pytest.raises(AssertionError):audit_initial_group([pairrow(a,3103),pairrow(b,3105)],tmp/'attrs.json')
    image=np.zeros((20,50,3),np.uint8);other=image.copy();other.flat[:3]=2
    assert rgb_difference(image,other)[0]['passed'];other.flat[3]=1
    assert not rgb_difference(image,other)[0]['passed'];other=image.copy();other.flat[0]=3
    assert not rgb_difference(image,other)[0]['passed']


def test_lossless_storage_all_attributes_and_failure(tmp):
    path=tmp/'trajectory.h5'
    with h5py.File(path,'w') as h:
        h.attrs['root']='unchanged';g=h.create_group('g');g.attrs['vector']=np.arange(3)
        g.create_dataset('floats',data=np.array([1.,np.nan,-0.,np.inf]));g.create_dataset('integers',data=np.arange(100).reshape(10,10))
        g.create_dataset('text',data=np.bytes_('raw string'));g.create_dataset('scalar',data=np.int64(7))
        g['floats'].attrs['units']='m';g.create_dataset('empty',data=np.zeros((0,3)))
    backup=tmp/'source_copy.h5';shutil.copyfile(path,backup);before=sha(backup)
    result=repack(path);assert result['datasets_exact']==5 and verify_h5_equal(backup,path)==5 and sha(backup)==before
    assert sha(path)==result['final_sha256']
    with h5py.File(path,'r+') as h:h['g/integers'][0,0]=99
    with pytest.raises(AssertionError):verify_h5_equal(backup,path)


def test_completion_requires_observed_exit(tmp):
    from pact_v1010b_run import validate_completion
    job={'job_sha256':'hash'}
    for receipt in [{'returncode':1,'observed_by_parent':True,'job_sha256':'hash'},
        {'returncode':0,'observed_by_parent':False,'job_sha256':'hash'},
        {'returncode':0,'observed_by_parent':True,'job_sha256':'bad'}]:
        with pytest.raises(AssertionError):validate_completion(job,tmp,receipt)


def test_continuation_interrupted_fixture_exact():
    def model():return torch.nn.Linear(2,1)
    def steps(m,opt,n):
        for _ in range(n):
            x=torch.randn(4,2);loss=(m(x)-torch.randn(4,1)).square().mean();assert torch.isfinite(loss)
            loss.backward();opt.step();opt.zero_grad()
    torch.manual_seed(42);m=model();o=torch.optim.AdamW(m.parameters(),lr=1e-5)
    steps(m,o,3);state={'model':copy.deepcopy(m.state_dict()),'optimizer':copy.deepcopy(o.state_dict()),'rng':torch.get_rng_state()}
    steps(m,o,4);expected=copy.deepcopy(m.state_dict());rng=torch.get_rng_state()
    n=model();p=torch.optim.AdamW(n.parameters(),lr=1e-5);n.load_state_dict(state['model']);p.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng']);steps(n,p,4)
    assert all(torch.equal(expected[k],n.state_dict()[k]) for k in expected) and torch.equal(rng,torch.get_rng_state())
    assert {int(s['step']) for s in p.state.values()}=={7}


def test_frozen_actual_fork_and_raw_case():
    from pact_v1010b_train import validate_fork
    source=W/'checkpoints/pact_seed3104'
    bundle=torch.load(source/'resume_bundle.ckpt',map_location='cpu',weights_only=False)
    model=torch.load(source/'policy_update_60000.ckpt',map_location='cpu',weights_only=False)
    assert validate_fork(bundle,model)['global_step']==60000
    del bundle,model
    evidence=read(A/'evidence.json')
    old=next(r for r in evidence['rollouts'] if r['arm']=='PACT' and r['episode_id'].startswith('98c64a'))
    row=recompute(ROOT/old['directory'])
    assert row['failure_stage']==old['failure_stage'] and row['close_command_index']==148
    assert abs(row['close_tracking_gap_z_m']-old['close_gap_z_m'])<1e-10
    assert abs(row['translation_only_rim']['minimum_early_achieved_depth_m']-old['minimum_rim_relative_tcp_early_grasp_m'])<1e-10


def test_mujoco_force_frame_and_sign():
    import mujoco
    from pact_v1010b_eval_worker import force_world_on_geom2
    xml='<mujoco><option timestep="0.002" gravity="0 0 -9.81"/><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .101"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>'
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    for _ in range(1000):mujoco.mj_step(model,data)
    forces=[]
    for i in range(data.ncon):
        f=np.zeros(6);mujoco.mj_contactForce(model,data,i,f)
        c=data.contact[i];world=force_world_on_geom2(c,f)
        forces.append(world if c.geom2==1 else -world)
    total=np.sum(forces,axis=0)
    assert abs(total[2]-9.81)<.02 and np.linalg.norm(total[:2])<.01


def test_pool_parent_exception_drains_and_observes(tmp,monkeypatch):
    from pact_v1010b_run import Pool
    pool=Pool.__new__(Pool);pool.active={1:'owned'};pool.deadline=time.time()+60;pool.paused=None
    observed=[]
    def fail(*args):raise RuntimeError('fixture parent failure')
    def poll():observed.append('waited');pool.active.clear()
    pool._execute=fail;pool.poll=poll
    with pytest.raises(RuntimeError,match='fixture parent failure'):pool.execute([])
    assert observed==['waited'] and not pool.active


def test_pair_seed_change_and_absent_member_block(tmp):
    a=tmp/'a';b=tmp/'b';write_initial(a);write_initial(b)
    rows=[pairrow(a,3103),pairrow(b,3104)];rows[1]['task_seed']['seed_u32']=2
    with pytest.raises(AssertionError):audit_initial_group(rows,tmp/'seed.json')
    with pytest.raises(AssertionError):audit_initial_group(rows[:1],tmp/'missing.json')


def test_storage_partial_publication_preserved(tmp):
    source=tmp/'trajectory.h5'
    with h5py.File(source,'w') as h:h.create_dataset('data',data=np.arange(10))
    partial=tmp/'trajectory.repacking.h5';partial.write_bytes(b'partial')
    before=sha(source)
    with pytest.raises(AssertionError,match='partial repack'):repack(source)
    assert sha(source)==before and partial.read_bytes()==b'partial'


def test_raster_onset_resolution_rejects_unexplained_differences():
    from pact_v1010b_run import replay_raster_checks
    comparison={'initial_pairing_passed':True,'outcome_exact':True,'failure_stage_exact':True,
        'actions':{k:{'exact':False,'first_different_step':0} for k in ('arm','model_output')}}
    proof={'inputs':{k:{'qpos_sha256':'q','proximity_sha256':'p','image_tensor_sha256':k} for k in ('original','new')},
        'comparisons':{k:{'first_vector_exact':True,'first_vector_max_abs':0} for k in ('original','new')},
        'repeated_inference_exact':True,'preprocessed_rgb_changed_channels':12,'predicted_delta':[.01,.02],'retained_delta':[.01,.02]}
    assert all(replay_raster_checks(comparison,proof).values())
    changed=copy.deepcopy(comparison);changed['outcome_exact']=False
    assert not all(replay_raster_checks(changed,proof).values())
    changed=copy.deepcopy(comparison);changed['actions']['arm']['first_different_step']=1
    assert not all(replay_raster_checks(changed,proof).values())
    for kind in ('qpos_sha256','proximity_sha256'):
        changed=copy.deepcopy(proof);changed['inputs']['new'][kind]='wrong'
        assert not all(replay_raster_checks(comparison,changed).values())
    changed=copy.deepcopy(proof);changed['retained_delta'][1]+=.001
    assert not all(replay_raster_checks(comparison,changed).values())
    changed=copy.deepcopy(proof);changed['comparisons']['original']['first_vector_exact']=False
    assert not all(replay_raster_checks(comparison,changed).values())


def test_single_held_observation_scope_rejects_broader_changes():
    from pact_v1010b_run import replay_scope_checks
    if not (B/'replay_scope_resolutions.json').exists():pytest.skip('Executed replay scope fixture not present')
    scope=read(B/'replay_scope_resolutions.json')['cases'][0]
    comparison=next(r for r in read(B/'replay_comparison.json')['comparisons'] if r['scene_id']==scope['scene_id'] and r['arm']==scope['arm'])
    diagnosis=read(scope['diagnosis_path']);contacts=read(scope['contact_reconstruction_path'])
    assert all(replay_scope_checks(scope,comparison,diagnosis,contacts).values())
    changed=copy.deepcopy(comparison);changed['scene_id']='another-scene'
    assert not all(replay_scope_checks(scope,changed,diagnosis,contacts).values())
    changed=copy.deepcopy(diagnosis);changed['new']['held_indices'].append(154)
    assert not all(replay_scope_checks(scope,comparison,changed,contacts).values())
    changed=copy.deepcopy(diagnosis);changed['new']['max_target_lift_m']=.02
    assert not all(replay_scope_checks(scope,comparison,changed,contacts).values())
    changed=copy.deepcopy(contacts)
    held=next(r for r in changed['rows'] if r['recorded']['held']);held['other_contact_geoms'].append({'is_gripper':False})
    assert not all(replay_scope_checks(scope,comparison,diagnosis,changed).values())
