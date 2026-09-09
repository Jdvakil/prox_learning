"""Immutable identities and schedules for the bounded V10.10b continuation."""
from __future__ import annotations
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from pact_wrist288_common import (ROOT, WORK as W, CODE, SEEDS, THREAD_ENV, ENCODER_PATH,
    ENCODER_SHA256, CANONICAL_SENSOR_NAMES, SENSOR_ORDER_SHA256, cells, cell_key,
    SCENE_BY_POSE, SAMPLER_CLASS, ACTIVE_CLUTTER_SLOTS, now, sha, digest, read,
    atomic, freeze, append, lines, pin_torch, environment as old_environment)
A = ROOT / 'diagnostics_output/pact_place_v1010b_grasp_audit'
B = ROOT / 'diagnostics_output/pact_place_v1010b_grasp_v1'
SCHEMA = 'pact_v1010b_grasp_v1'
NAMESPACE = 'pact_place_v1010b_grasp_v1'
PLAN = ROOT / 'docs/PACT_PLACE_V1010B_GRASP_FIX_PLAN.md'
PYTHON = '/root/act_retrain_venv/bin/python'
VARIANTS = ('frozen60000', 'uniform63000', 'acquisition63000')
GIB = 2**30
MIB = 2**20
EFFECTIVE_TRAINING_CONFIG = W / 'amendments/wrist280/effective_config.json'


def training_config_binding():
    config=read(EFFECTIVE_TRAINING_CONFIG)
    assert config['config_sha256']==digest({k:v for k,v in config.items() if k!='config_sha256'})
    assert config['split']['train']==240 and config['split']['validation']==40
    assert config['collection']['episodes']==280 and config['training']['batch_size']==8
    return {'path':str(EFFECTIVE_TRAINING_CONFIG),'sha256':sha(EFFECTIVE_TRAINING_CONFIG),
        'config_sha256':config['config_sha256'],'train_episodes':240,'validation_episodes':40}


def environment():
    env = old_environment()
    env.update(PACT_V1010B_OWNER=NAMESPACE, WANDB_MODE='disabled')
    return env


def inside(path):
    path = Path(path).resolve()
    assert path.is_relative_to(B.resolve()), f'new-run writes must remain under {B}: {path}'
    return path


def rank(role, identity):
    return hashlib.sha256((NAMESPACE+'/'+role+'/'+identity).encode()).hexdigest()


def derive_scene(role, block, cell, ordinal, retry):
    fields = [NAMESPACE, str(role), int(block), str(cell), int(ordinal), int(retry)]
    value = int(digest(fields)[:16], 16)
    return {'seed_u32': value % 2**32, 'seed_u64': value, 'derivation': fields}


def historical_rows():
    found = {}
    for seed in SEEDS:
        for row in read(W/f'manifests/final_{seed}.json')['rows']:
            found[row['episode_id']] = (seed, row)
    assert len(found) == 150
    return found


def retained_wrapper(row, source_seed, evidence):
    original = [r for r in evidence['rollouts'] if r['episode_id'] == row['episode_id']]
    assert len(original) == 2 and {r['arm'] for r in original} == {'ACT', 'PACT'}
    results = {r['arm']: read(ROOT/r['directory']/'result.json') for r in original}
    assert results['ACT']['seed'] == results['PACT']['seed']
    retry = results['PACT']['sampling_retry_index']
    assert all(results[a]['sampling_retry_index'] == retry for a in results)
    expected = {k: row['sampling_seeds'][retry][k] for k in ('seed_u32','seed_u64')}
    assert expected == results['PACT']['seed']
    return {'scene_id': row['episode_id'], 'scene_block': source_seed, 'physical_row': row,
        'physical_row_digest': digest(row), 'selected_retry': retry, 'selected_seed': expected,
        'historical_directories': {r['arm']: r['directory'] for r in original},
        'source_seed': source_seed, 'exposed_diagnostic_only': True}


def select_cross24():
    selection = read(A/'cross24_scenes.json'); evidence = read(A/'evidence.json')
    originals = historical_rows(); allrows = list(originals.values())
    wrappers = []
    for selected in selection['rows']:
        seed, row = originals[selected['episode_id']]
        assert selected == dict(row, source_seed=seed), 'selection wrapper must not enter physics'
        samecell = [r for _,r in allrows if r['cell'] == row['cell']]
        assert min(samecell, key=lambda r: rank('cross24',r['episode_id']))['episode_id'] == row['episode_id']
        wrappers.append(retained_wrapper(row, seed, evidence))
    assert len(wrappers) == 24 and len({r['physical_row']['cell'] for r in wrappers}) == 24
    return wrappers


def instrument_six(evidence):
    required = ['98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f',
        '621d8bd12acc6e7256e2a671b85f49c0168e469e53ac02193dc8e0ba09be2d11',
        'c0ff4fd9273e6de8a16efdb2c68d4827f22a2bf32772633c610c90a732ec735b']
    for seed in SEEDS:
        eligible = [r for r in evidence['rollouts'] if r['arm']=='PACT' and r['seed']==seed
            and r['episode_id'] not in required and (r['success'] if seed != 3105 else r.get('rim_blocked_signature',False))]
        required.append(min(eligible, key=lambda r:rank('instrument',r['episode_id']))['episode_id'])
    original = historical_rows()
    return [retained_wrapper(original[e][1], original[e][0], evidence) for e in required]


def seed_inventory():
    pattern = r'"(?:task_seed(?:_u32|_u64)?|seed_u32|seed_u64|episode_seed|seed)"\s*:\s*[0-9]+'
    import re
    proc = subprocess.run(['rg','--no-ignore','--no-heading','--with-filename','-o', pattern,
        '-g','*.json','-g','*.jsonl','diagnostics_output','assets','configs'], cwd=ROOT,
        text=True,capture_output=True)
    assert proc.returncode in (0,1), proc.stderr
    used=set(); sources=collections.Counter()
    for line in proc.stdout.splitlines():
        path,match=line.split(':',1)
        if str(B.relative_to(ROOT)) in path: continue
        used.add(int(re.search(r':\s*([0-9]+)$',match)[1]) % 2**32); sources[path]+=1
    assert len(used)>1000
    return {'seed_u32_values':sorted(used),'source_matches':dict(sources)}


def fresh_rows(role, selection, used):
    import pact_place_v1010_contract as old
    wrappers=[]
    for block, cell, ordinal in selection:
        values=[derive_scene(role,block,cell,ordinal,r) for r in range(13)]
        assert len({v['seed_u32'] for v in values})==13
        # Domain is fixed. A collision stops preparation; it never causes an outcome-driven redraw.
        assert not (set(v['seed_u32'] for v in values)&used), 'task seed collision requires a versioned derivation amendment'
        used.update(v['seed_u32'] for v in values)
        family,side,pose=cell.split('|'); row=old.build_row(family,side,pose,ordinal)
        identity=digest([NAMESPACE,role,block,cell,ordinal])
        row.update(attempt_id=identity,episode_id=identity,seed_stream=NAMESPACE+'/'+role,
            experiment_namespace=NAMESPACE,stream_role=role,task_seed_u32=values[0]['seed_u32'],
            task_seed_u64=values[0]['seed_u64'],sampling_seeds=values,role_index=ordinal,
            candidate_index=ordinal,schema_version=SCHEMA,role=role)
        row['row_sha256']=digest({k:v for k,v in row.items() if k!='row_sha256'})
        wrappers.append({'scene_id':identity,'scene_block':block,'physical_row':row,
            'physical_row_digest':digest(row),'selected_retry':None,'selected_seed':None,
            'historical_directories':{},'exposed_diagnostic_only':role!='final_v1'})
    return wrappers


def make_labels(evidence):
    import h5py
    import numpy as np
    from fixed_split_data import load_split_manifest
    split=load_split_manifest(W/'split_manifest.json'); train={e['source_episode_id'] for e in split['train']}
    validation={e['source_episode_id'] for e in split['val']}; assert len(train)==240 and len(validation)==40 and not train&validation
    rows=[]
    for demo in evidence['demos']:
        if demo['episode_id'] not in train: continue
        path=Path(demo['converted_path'])
        with h5py.File(path) as h:
            actions=h['action'][()]; assert h.attrs['sim']
            assert actions.shape[1]==8 and np.isfinite(actions).all()
            closes=np.flatnonzero((actions[:,7]>=127.5)&np.r_[True,actions[:-1,7]<127.5])
            assert len(closes)==1
            close=int(closes[0]); T=len(actions)
            assert close==demo['first_close_command_step'] and demo['alignment_exact']
        rows.append({'episode_id':demo['episode_id'],'act_episode_index':demo['act_episode_index'],
            'converted_path':str(path),'converted_sha256':sha(path),'T':T,'close':close,
            'window':[max(0,close-30),min(T-1,close+30)]})
    assert len(rows)==240
    return {'schema':SCHEMA,'rows':sorted(rows,key=lambda r:r['act_episode_index']),
        'validation_ids':sorted(validation),'split_sha256':sha(W/'split_manifest.json'),
        'rule':'first aligned action[:,7]>=127.5; inclusive close +/-30; no loader shift'}


def build_contract(start_utc):
    B.mkdir(parents=True,exist_ok=True)
    assert not (B/'contract.json').exists(), 'existing contract: use verification/resume'
    t0=datetime.fromisoformat(start_utc.replace('Z','+00:00'))
    assert t0.tzinfo is not None
    evidence=read(A/'evidence.json'); verification=read(A/'final_verification.json')
    assert verification['status']=='AUDIT_COMPLETE' and verification['rollouts']==300
    for path,expected in verification['deliverable_hashes'].items(): assert sha(path)==expected,path
    cross=select_cross24(); curated=instrument_six(evidence)
    mechanism=sorted(cross,key=lambda r:rank('mechanism12',r['scene_id']))[:12]
    inventory=seed_inventory(); freeze(B/'historical_seed_inventory.json',inventory)
    used=set(inventory['seed_u32_values']); families=list(dict.fromkeys(c[0] for c in cells()))
    sides=list(dict.fromkeys(c[1] for c in cells())); poses=list(dict.fromkeys(c[2] for c in cells()))
    devselect=[(block,cell_key(family,sides[(bi+fi+pi)%2],pose),0)
        for bi,block in enumerate(SEEDS) for fi,family in enumerate(families) for pi,pose in enumerate(poses)]
    development=fresh_rows('development_v1',devselect,used)
    extras=sorted([cell_key(*c) for c in cells()],key=lambda c:rank('final_extras_v1',c))[:6]
    finalselect=[(block,cell_key(*c),ordinal) for block in SEEDS for c in cells() for ordinal in (0,1)]
    finalselect += [(block,c,2) for bi,block in enumerate(SEEDS) for c in extras[2*bi:2*bi+2]]
    final=fresh_rows('final_v1',finalselect,used)
    assert len(development)==36 and len(final)==150
    assert len({r['scene_id'] for r in development+final})==186
    manifests={}
    for name,rows in [('A1',curated),('A2',cross),('B',mechanism),('C',development),('D',final)]:
        doc={'schema':SCHEMA,'stage':name,'sensor_names':list(CANONICAL_SENSOR_NAMES),'scenes':rows}
        doc['manifest_sha256']=digest(doc); freeze(B/f'scene_manifests/{name}.json',doc)
        manifests[name]={'sha256':sha(B/f'scene_manifests/{name}.json'),'count':len(rows)}
    freeze(B/'training_labels.json',make_labels(evidence))
    inputs={str(PLAN):sha(PLAN),str(A/'final_verification.json'):sha(A/'final_verification.json'),
        str(A/'evidence.json'):sha(A/'evidence.json'),str(A/'cross24_scenes.json'):sha(A/'cross24_scenes.json')}
    oldconfig=read(W/'config.json')
    inputs.update({str(ROOT/p):v for p,v in oldconfig['file_hashes'].items()})
    for p in [W/'config.json',EFFECTIVE_TRAINING_CONFIG,W/'split_manifest.json',W/'conversion_manifest.json',W/'dataset_stats.pkl',ENCODER_PATH]: inputs[str(p)]=sha(p)
    models={}
    for entry in evidence['checkpoints']:
        directory=Path(entry['directory']); model=directory/'policy_update_60000.ckpt'
        assert sha(model)==entry['checkpoint_sha256']
        models[directory.name]={'path':str(model),'sha256':entry['checkpoint_sha256'],
            'resume_path':str(directory/'resume_bundle.ckpt'),'resume_sha256':sha(directory/'resume_bundle.ckpt'),
            'stats_path':str(directory/'dataset_stats.pkl'),'stats_sha256':entry['stats_sha256'],
            'run_manifest_sha256':sha(directory/'run_manifest.json')}
    code={str(p):sha(p) for p in sorted(CODE.glob('pact_v1010b_*.py'))}
    code[str(ROOT/'tests/test_pact_v1010b.py')]=sha(ROOT/'tests/test_pact_v1010b.py')
    doc={'schema':SCHEMA,'namespace':NAMESPACE,'start_utc':t0.isoformat(),
        'safe_stop_utc':(t0+timedelta(hours=47)).isoformat(),'deadline_utc':(t0+timedelta(hours=48)).isoformat(),
        'output':str(B),'plan_sha256':sha(PLAN),'input_hashes':inputs,'code_hashes':code,'models':models,
        'scene_manifests':manifests,'training_labels_sha256':sha(B/'training_labels.json'),
        'historical_seed_inventory_sha256':sha(B/'historical_seed_inventory.json'),
        'sampling':{'p':0.25,'half_window':30,'uniform_p':0.0},
        'training':{'start':60000,'end':63000,'new_updates_per_branch':3000,'branches':12,'total_new_updates':36000,
            'batch_size':8,'epochs':100,'updates_per_epoch':30,'checkpoint_every':300,'serial':True,
            'source_config':training_config_binding()},
        'runtime':{'history':100,'num_queries':100,'horizon':900,'end_on_success':False,'threshold':127.5,
            'action_noise':False,'runtime_grasp_gate':False,'sensor_names':list(CANONICAL_SENSOR_NAMES)},
        'instrumentation':{'physics_dt':0.002,'cap_steps':300,'after_close':60,'rgb_cap':150,'max_bytes':100*MIB},
        'counts':{'A1':12,'A2_new':48,'A2_analyzed':72,'B_new':180,'B_analyzed':216,'C':216,'D':600,
            'main_path':1056,'reuse_replacements_max':24,'infra_retries_max':12,'per_job_infra_retries_max':1},
        'resources':{'workers':12,'backoff_workers':10,'vram_max':.85,'ram_max':.80,'pid_max':.75,
            'pressure_samples':3,'sample_seconds':60,'cache_release_ram':.73,'disk_reserve_bytes':10*GIB,
            'standard_rollout_bytes':34*MIB,'model_budget_bytes':5*GIB,'temporary_budget_bytes':3*GIB,
            'instrument_budget_bytes':1*GIB,'planning_rollout_seconds':1237*1.2},
        'final_extra_cells':extras,'final_balance':{str(s):dict(collections.Counter(r['physical_row']['cell'].split('|')[1] for r in final if r['scene_block']==s)) for s in SEEDS},
        'preserve_user_eval_sha256':sha(ROOT/'EVAL.md'),'historical_protected_hashes':evidence['source_hashes'],
        'historical_authorizations_unchanged':True}
    doc['contract_sha256']=digest(doc); freeze(B/'contract.json',doc)
    for stage in ('A1','A2','B','C','D'): freeze(B/f'schedules/{stage}.json',build_jobs(stage))
    return doc


def verify_contract(full=False):
    doc=read(B/'contract.json'); assert doc['contract_sha256']==digest({k:v for k,v in doc.items() if k!='contract_sha256'})
    for key in ('input_hashes','code_hashes'):
        for path,expected in doc[key].items(): assert sha(path)==expected, f'frozen {key} drift: {path}'
    for stage,value in doc['scene_manifests'].items(): assert sha(B/f'scene_manifests/{stage}.json')==value['sha256']
    for file,key in [('training_labels.json','training_labels_sha256'),('historical_seed_inventory.json','historical_seed_inventory_sha256')]: assert sha(B/file)==doc[key]
    for model in doc['models'].values():
        for k,h in [('path','sha256'),('resume_path','resume_sha256'),('stats_path','stats_sha256')]: assert sha(model[k])==model[h]
    assert sha(ROOT/'EVAL.md')==doc['preserve_user_eval_sha256'], 'pre-existing EVAL.md changed'
    if full:
        for path,expected in doc['historical_protected_hashes'].items(): assert sha(path)==expected,path
    return doc


def build_jobs(stage):
    contract=read(B/'contract.json'); manifest=B/f'scene_manifests/{stage}.json'; scenes=read(manifest)['scenes']; jobs=[]
    for scene in scenes:
        for seed in ([scene['scene_block']] if stage in ('A1','C','D') else SEEDS):
            for arm in (('PACT',) if stage=='A2' else ('ACT','PACT')):
                for variant in (('frozen60000',) if stage.startswith('A') else (('frozen60000','acquisition63000') if stage=='D' else VARIANTS)):
                    model=contract['models'][f'{arm.lower()}_seed{seed}']
                    historical=scene['historical_directories'].get(arm) if variant=='frozen60000' and seed==scene.get('source_seed') else None
                    reuse=bool(stage=='A2' and historical) or bool(stage=='B' and arm=='PACT' and variant=='frozen60000')
                    checkpoint=(Path(model['path']) if variant=='frozen60000' else B/f'checkpoints/{arm.lower()}_seed{seed}_{variant}/policy_update_63000.ckpt')
                    identity=f'{stage}_{arm}_{seed}_{variant}_{scene["scene_id"][:16]}'
                    job={'schema':SCHEMA,'stage':stage,'job_id':identity,'scene_id':scene['scene_id'],
                        'scene_block':scene['scene_block'],'training_seed':seed,'arm':arm,'variant':variant,
                        'instrumentation':'grasp' if stage=='A1' else 'none','record_chunks':stage in ('A1','A2'),
                        'manifest':str(manifest),'manifest_sha256':sha(manifest),'checkpoint_path':str(checkpoint),
                        'checkpoint_sha256':model['sha256'] if variant=='frozen60000' else None,
                        'physical_row_digest':scene['physical_row_digest'],'selected_retry':scene['selected_retry'],
                        'selected_seed':scene['selected_seed'],'reuse':reuse,'historical_directory':historical,
                        'output_dir':str(B/f'rollouts/{stage}/{identity}/attempt_00'),
                        'contract_sha256':contract['contract_sha256']}
                    job['job_sha256']=digest(job); jobs.append(job)
    expected={'A1':12,'A2':72,'B':216,'C':216,'D':600}[stage]
    assert len(jobs)==expected and len({j['job_id'] for j in jobs})==expected
    return jobs


def require_stage(stage):
    predecessor={'training':'A','B':'training','C':'B','D':'C'}.get(stage)
    if predecessor:
        path=B/f'gates/{predecessor}.json'
        assert path.exists() and read(path)['passed'], f'{stage} prerequisite {predecessor} is not passed'
