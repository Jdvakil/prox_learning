"""Run the authorized seed3104/3105 extension with the seed3103 scientific method."""
from __future__ import annotations
import argparse
import fcntl
import signal
import subprocess
import time
import traceback
from datetime import datetime, timedelta, timezone
from pact_v1010c_multi_core import *
from pact_v1010c_multi_train import resource_sample

# Reuse the previously exercised observed-exit/resource supervisor verbatim.
for _name in ('valid_records', 'run_jobs'):
    exact_symbol(CODE/'pact_v1010c_run.py', _name, globals())


def verify_contract(full=False):
    doc = read(C/'contract.json')
    assert doc['sha256'] == digest({k:v for k,v in doc.items() if k!='sha256'})
    assert doc['training_seed'] == TRAINING_SEED
    verify_upstream()
    for key in ('code_hashes', 'input_hashes'):
        for path, expected in doc[key].items():
            assert sha(ROOT/path) == expected, path
    if full:
        for row in read(W/'conversion_manifest.json')['episodes']:
            assert sha(W/'converted'/row['act_file']) == row['act_file_sha256']
        for row in read(C/'data_views_manifest.json')['episodes']:
            assert sha(C/'data_views'/row['file']) == row['view_sha256']
    return doc


def prepare():
    if (C/'contract.json').exists():
        return verify_contract(full=True)
    from pact_v1010b_metrics import recompute
    tests = read(MULTI/'preflight.json')
    assert all(tests[str(seed)]['passed'] for seed in (3104,3105))
    physical = {r['episode_id']:r for r in read(W/f'manifests/final_{TRAINING_SEED}.json')['rows']}
    jobs = read(W/f'evaluation/final_{TRAINING_SEED}_full50_h100_schedule.json')
    assert len(jobs) == 100 and len(physical) == 50
    scenes, baseline, protected = [], [], {}
    for job in jobs:
        directory = Path(job['directory'])
        result, receipt = read(directory/'result.json'), read(directory/'exit_receipt.json')
        assert receipt['returncode'] == 0 and result['checkpoint_seed'] == TRAINING_SEED
        raw = recompute(directory)
        row = physical[result['episode_id']]
        retry = result['sampling_retry_index']
        selected = {k:row['sampling_seeds'][retry][k] for k in ('seed_u32','seed_u64')}
        assert result['seed'] == selected
        raw['physical_row_digest'] = digest(row)
        raw['method'] = 'ACT' if job['schedule']['arm']=='ACT' else 'PACT-Frozen'
        baseline.append(raw)
        for name in ('result.json','exit_receipt.json','initial_observation.h5','actions.npz','telemetry.h5','trajectory.h5'):
            protected[str(directory/name)] = sha(directory/name)
        if job['schedule']['arm']=='PACT':
            scenes.append({'scene_id':result['episode_id'], 'physical_row':row,
                'selected_retry':retry, 'selected_seed':selected, 'frozen_directory':str(directory),
                'frozen_result_sha256':sha(directory/'result.json'),
                'frozen_exit_sha256':sha(directory/'exit_receipt.json')})
    assert len(scenes) == 50
    expected = {3104:{'ACT':(22,15),'PACT-Frozen':(19,10)},
                3105:{'ACT':(23,15),'PACT-Frozen':(35,25)}}[TRAINING_SEED]
    for method,(success,cfts) in expected.items():
        rows = [r for r in baseline if r['method']==method]
        assert len(rows)==50 and sum(r['task_success'] for r in rows)==success
        assert sum(r['collision_free_task_success'] for r in rows)==cfts
    spec=importlib.util.spec_from_file_location('multiseed_baseline_pairing',CODE/'pact_v1010b_pairing.py')
    pairing=importlib.util.module_from_spec(spec);spec.loader.exec_module(pairing)
    def inside(path):
        path=Path(path).resolve();assert path.is_relative_to(C);return path
    pairing.inside=inside
    for scene in scenes:
        group=[r for r in baseline if r['scene_id']==scene['scene_id']]
        assert len(group)==2 and {r['method'] for r in group}=={'ACT','PACT-Frozen'}
        destination=C/'baseline_pairings'/f'{scene["scene_id"]}.json'
        if destination.exists():assert read(destination)['passed']
        else:pairing.audit_initial_group(group,destination)
    manifest = {'schema':SCHEMA, 'training_seed':TRAINING_SEED, 'scenes':scenes,
        'exposed_regression_scenes':True, 'selection':'All 50 original final scenes for this seed'}
    manifest['sha256'] = digest(manifest)
    freeze(C/'evaluation_manifest.json', manifest)
    freeze(C/'baseline_metrics.json', {'rows':baseline, 'n':100})
    freeze(C/'baseline_hashes.json', protected)
    previous = read(PREVIOUS/'contract.json')
    protected = {**previous['input_hashes'], **{str(ROOT/p):s for p,s in previous['code_hashes'].items()}}
    for arm in ('act','pact'):
        for name in ('policy_update_60000.ckpt','dataset_stats.pkl','run_manifest.json'):
            path = W/f'checkpoints/{arm}_seed{TRAINING_SEED}'/name
            protected[str(path)] = sha(path)
    for name in ('evaluation_manifest.json','baseline_metrics.json','baseline_hashes.json',
                 'initialization.json','data_views_manifest.json','upstream_manifest.json'):
        protected[str(C/name)] = sha(C/name)
    protected[str(MULTI/'preflight.json')] = sha(MULTI/'preflight.json')
    protected[str(MULTI/'adapter_provenance.json')] = sha(MULTI/'adapter_provenance.json')
    paths = list(CODE.glob('pact_v1010c_multi_*.py'))
    paths += [ROOT/'tests/test_pact_v1010c_multi.py', ROOT/'docs/PACT_PLACE_V1010C_THREE_SEED_PLAN.md']
    start = datetime.now(timezone.utc)
    contract = {'schema':SCHEMA,'start_utc':start.isoformat(),
        'hard_deadline_utc':(start+timedelta(hours=48)).isoformat(),
        'training_seed':TRAINING_SEED,'updates':60000,'batch_size':8,'epochs':2000,
        'encoder_lr':1e-5,'policy_config':policy_config(),'eval_workers':6,'evaluation_count':50,
        'code_hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths},'input_hashes':protected,
        'intervention':'Identical seed3103 full main fine-tune method; only training seed and original scene block change'}
    contract['sha256'] = digest(contract)
    freeze(C/'contract.json',contract)
    verify_contract(full=True)
    return contract


def train_job(stop):
    return {'id':f'training_to_{stop}','kind':'training','stop':stop,
        'directory':str(C/'checkpoint'),
        'command':[sys.executable,str(CODE/'pact_v1010c_multi_train.py'),'--stop-updates',str(stop)]}


def eval_job(scene):
    assert read(C/'checkpoint/completed.json')['global_step']==60000
    checkpoint,encoder = C/'checkpoint/policy_last.ckpt', C/'checkpoint/prox_encoder.pt'
    name = f'readout60000_{scene["scene_id"][:20]}'
    job = {'schema':SCHEMA,'id':name,'scene_id':scene['scene_id'],'variant':'readout60000',
        'directory':str(C/'rollouts'/name),'checkpoint_path':str(checkpoint),
        'checkpoint_sha256':sha(checkpoint),'encoder_path':str(encoder),'encoder_sha256':sha(encoder)}
    job['sha256'] = digest(job)
    path = C/'jobs'/f'{name}.json'
    freeze(path,job)
    return {'id':name,'kind':'evaluation','directory':job['directory'],
        'command':[sys.executable,str(CODE/'pact_v1010c_multi_eval.py'),'--job',str(path)]}


def release_training_cache():
    assert read(C/'checkpoint/completed.json')['global_step']==60000
    before=resource_sample()
    for row in read(W/'conversion_manifest.json')['episodes']:
        path=W/'converted'/row['act_file']; info=path.stat()
        fd=os.open(path,os.O_RDONLY)
        try:
            os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        after=path.stat()
        assert (info.st_size,info.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    append(C/'cache_releases.jsonl',{'utc':now(),'files':280,'before':before,
        'after':resource_sample(),'read_only_advice':True,'training_complete_updates':60000})


def finish():
    from pact_v1010b_metrics import recompute, compare_rows
    verify_contract(full=True)
    for path,expected in read(C/'baseline_hashes.json').items():
        assert sha(path)==expected,path
    records=valid_records()
    scenes=read(C/'evaluation_manifest.json')['scenes']
    new=[]
    pair=read(C/'checkpoint/checkpoint_pairs.json')['policy_last.ckpt']
    assert pair['global_step']==60000
    for scene in scenes:
        path=C/'rollouts'/f'readout60000_{scene["scene_id"][:20]}'
        assert read(path/'initial_pairing.json')['passed']
        result=read(path/'result.json')
        assert result['checkpoint_seed']==result['training_seed']==TRAINING_SEED
        assert result['checkpoint_sha256']==pair['policy_sha256']
        assert result['surface_encoder_sha256']==pair['encoder_sha256']
        row=recompute(path);row['method']='PACT-Finetune';new.append(row)
    old=[r for r in read(C/'baseline_metrics.json')['rows'] if r['method']=='PACT-Frozen']
    assert len(new)==len(old)==50
    comparison=compare_rows(new,old)
    freeze(C/'comparison.json',{'utc':now(),'training_seed':TRAINING_SEED,'n':50,
        'comparison':comparison,'frozen_rows':old,'readout_rows':new,'contract_sha256':read(C/'contract.json')['sha256']})
    freeze(C/'final_verification.json',{'utc':now(),'raw_recomputation':True,
        'matched_pairs':50,'original_source_hashes_unchanged':True,'completed_training_updates':60000,
        'valid_observed_exit_records':len(records),'all_initial_pairings_passed':True})
    print(json.dumps({'seed':TRAINING_SEED,'comparison':comparison}),flush=True)


def supervise():
    lock=(C/'parent.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    contract=verify_contract()
    atomic(C/'run_status.json',{'utc':now(),'status':'RUNNING','active_parent':True,'parent_pid':os.getpid()})
    status,reason='INCOMPLETE',None
    try:
        run_jobs([train_job(300)],1,'training_pilot')
        pilot=read(C/'checkpoint/process_0_300.json')
        estimated=pilot['elapsed_s']/300*59700*1.25+9*1400+3600
        available=datetime.fromisoformat(contract['hard_deadline_utc']).timestamp()-time.time()
        freeze(C/'pilot_budget.json',{'utc':now(),'remaining_estimate_s':estimated,'available_s':available,
            'pilot_updates':300,'passed':estimated<available})
        assert estimated<available,'measured ETA exceeds budget'
        run_jobs([train_job(60000)],1,'training')
        release_training_cache()
        run_jobs([eval_job(s) for s in read(C/'evaluation_manifest.json')['scenes']],6,'evaluation')
        finish();status='COMPLETED_COMPARISON'
    except BaseException as exc:
        reason=repr(exc)
        atomic(C/f'error_{time.time_ns()}.json',{'utc':now(),'error':reason,'traceback':traceback.format_exc()})
        raise
    finally:
        closure={'utc':now(),'status':status,'reason':reason,'parent_pid':os.getpid(),
            'active_parent':False,'valid_records':len(lines(C/'valid_ledger.jsonl'))}
        atomic(C/'parent_closure.json',closure);atomic(C/'run_status.json',closure)
        atomic(C/'progress.json',dict(closure,stage='closed',active_jobs=[]))


def pipeline():
    lock=(MULTI/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status,reason='INCOMPLETE',None
    atomic(MULTI/'run_status.json',{'utc':now(),'status':'RUNNING','active_parent':True,'parent_pid':os.getpid()})
    try:
        for seed in (3104,3105):
            for stage in ('prepare','supervise'):
                receipt=MULTI/f'seed{seed}_{stage}_exit.json'
                if receipt.exists():
                    assert read(receipt)['returncode']==0
                    continue
                env=environment();env['PACT_FINETUNE_SEED']=str(seed)
                with (MULTI/f'seed{seed}_{stage}.log').open('a') as log:
                    p=subprocess.Popen([sys.executable,__file__,stage],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    atomic(MULTI/'progress.json',{'utc':now(),'seed':seed,'stage':stage,'child_pid':p.pid,
                        'proc_start_ticks':Path(f'/proc/{p.pid}/stat').read_text().split()[21]})
                    rc=p.wait()
                freeze(receipt,{'utc':now(),'seed':seed,'stage':stage,'pid':p.pid,'returncode':rc,'observed_by_parent':True})
                assert rc==0,f'seed{seed} {stage} exited {rc}'
        status='COMPLETED_TRAINING_AND_EVALUATIONS'
    except BaseException as exc:
        reason=repr(exc);raise
    finally:
        atomic(MULTI/'run_status.json',{'utc':now(),'status':status,'reason':reason,
            'active_parent':False,'parent_pid':os.getpid()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','supervise','pipeline','verify'))
    args=parser.parse_args()
    if args.stage=='prepare': print(json.dumps(prepare()),flush=True)
    elif args.stage=='supervise': supervise()
    elif args.stage=='pipeline': pipeline()
    else: finish()
