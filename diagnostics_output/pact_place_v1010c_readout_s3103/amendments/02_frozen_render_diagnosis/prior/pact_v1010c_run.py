"""Observed-exit supervision and matched comparison for the requested seed3103 run."""
from __future__ import annotations
import argparse
import fcntl
import shutil
import signal
import subprocess
import time
import traceback
from datetime import datetime,timedelta,timezone
from pact_v1010c_core import *
from pact_v1010c_train import resource_sample


def prepare():
    if (C/'contract.json').exists():
        return verify_contract()
    assert read(C/'data_views_manifest.json')['count']==280
    assert '5 passed' in (C/'preflight_tests.log').read_text()
    assert '1 passed' in (C/'preflight_live_history.log').read_text()
    from pact_v1010b_metrics import recompute
    schedule=read(W/'evaluation/final_3103_full50_h100_schedule.json')
    old_jobs=[j for j in schedule if j['schedule']['arm']=='PACT']
    physical={r['episode_id']:r for r in read(W/'manifests/final_3103.json')['rows']}
    assert len(old_jobs)==len(physical)==50
    scenes=[];baseline=[]
    for job in old_jobs:
        directory=Path(job['directory']);result=read(directory/'result.json')
        receipt=read(directory/'exit_receipt.json')
        assert receipt['returncode']==0
        raw=recompute(directory)
        row=physical[result['episode_id']]
        retry=result['sampling_retry_index']
        selected={k:row['sampling_seeds'][retry][k] for k in ('seed_u32','seed_u64')}
        assert result['seed']==selected and result['checkpoint_seed']==3103
        scenes.append({'scene_id':result['episode_id'],'physical_row':row,'selected_retry':retry,
            'selected_seed':selected,'frozen_directory':str(directory),
            'frozen_result_sha256':sha(directory/'result.json'),
            'frozen_exit_sha256':sha(directory/'exit_receipt.json')})
        baseline.append(raw)
    assert sum(r['task_success'] for r in baseline)==20
    manifest={'schema':SCHEMA,'scenes':scenes,'training_seed':3103,
              'exposed_regression_scenes':True,'selection':'All 50 original seed3103 final scenes'}
    manifest['sha256']=digest(manifest)
    freeze(C/'evaluation_manifest.json',manifest)
    freeze(C/'frozen_baseline_metrics.json',{'rows':baseline,'n':50})
    inputs={str(W/'split_manifest.json'):sha(W/'split_manifest.json'),
            str(W/'conversion_manifest.json'):sha(W/'conversion_manifest.json'),
            str(W/'dataset_stats.pkl'):sha(W/'dataset_stats.pkl'),
            str(W/'checkpoints/pact_seed3103/policy_update_60000.ckpt'):sha(W/'checkpoints/pact_seed3103/policy_update_60000.ckpt'),
            str(ROOT/'EVAL.md'):sha(ROOT/'EVAL.md')}
    for path, expected in read(W/'config.json')['file_hashes'].items():
        assert sha(ROOT/path)==expected,path
        inputs[str(ROOT/path)]=expected
    inputs[str(C/'data_views_manifest.json')]=sha(C/'data_views_manifest.json')
    inputs[str(C/'upstream_manifest.json')]=sha(C/'upstream_manifest.json')
    inputs[str(C/'initialization.json')]=sha(C/'initialization.json')
    inputs[str(C/'evaluation_manifest.json')]=sha(C/'evaluation_manifest.json')
    paths=[f'scripts/pact_v1010c_{name}.py' for name in ('core','train','eval','run')]
    paths+=['tests/test_pact_v1010c.py','docs/PACT_PLACE_V1010C_READOUT_3103_PLAN.md']
    start=datetime.now(timezone.utc)
    contract={'schema':SCHEMA,'start_utc':start.isoformat(),
              'hard_deadline_utc':(start+timedelta(hours=48)).isoformat(),
              'code_hashes':{p:sha(ROOT/p) for p in paths},'input_hashes':inputs,
              'policy_config':policy_config(),'training_seed':3103,'updates':60000,
              'batch_size':8,'epochs':2000,'encoder_lr':1e-5,'eval_workers':8,
              'evaluation_count':50,'frozen_replay_smoke_count':1,
              'comparison':'Full main readout method versus original frozen32 PACT; exposed matched50'}
    contract['sha256']=digest(contract)
    freeze(C/'contract.json',contract)
    verify_contract(full=True)
    return contract


def verify_contract(full=False):
    c=read(C/'contract.json')
    assert c['sha256']==digest({k:v for k,v in c.items() if k!='sha256'})
    verify_upstream()
    for path,expected in c['code_hashes'].items():
        assert sha(ROOT/path)==expected,path
    for path,expected in c['input_hashes'].items():
        assert sha(path)==expected,path
    if full:
        manifest=read(W/'conversion_manifest.json')
        for row in manifest['episodes']:
            assert sha(W/'converted'/row['act_file'])==row['act_file_sha256']
        for row in read(C/'data_views_manifest.json')['episodes']:
            assert sha(C/'data_views'/row['file'])==row['view_sha256']
    return c


def eval_job(scene,variant):
    if variant=='frozen60000':
        checkpoint=W/'checkpoints/pact_seed3103/policy_update_60000.ckpt'
        encoder=Path(ENCODER_PATH)
    else:
        checkpoint=C/'checkpoint/policy_last.ckpt';encoder=C/'checkpoint/prox_encoder.pt'
        assert read(C/'checkpoint/completed.json')['global_step']==60000
    job_id=f'{variant}_{scene["scene_id"][:20]}'
    job={'schema':SCHEMA,'id':job_id,'scene_id':scene['scene_id'],'variant':variant,
         'directory':str(C/'rollouts'/job_id),'checkpoint_path':str(checkpoint),
         'checkpoint_sha256':sha(checkpoint),'encoder_path':str(encoder),'encoder_sha256':sha(encoder)}
    job['sha256']=digest(job)
    path=C/'jobs'/f'{job_id}.json'
    freeze(path,job)
    return {'id':job_id,'kind':'evaluation','directory':job['directory'],
            'command':[sys.executable,str(CODE/'pact_v1010c_eval.py'),'--job',str(path)]}


def train_job(stop):
    return {'id':f'training_to_{stop}','kind':'training','stop':stop,
            'directory':str(C/'checkpoint'),
            'command':[sys.executable,str(CODE/'pact_v1010c_train.py'),'--stop-updates',str(stop)]}


def valid_records():
    records=lines(C/'valid_ledger.jsonl')
    assert len({r['id'] for r in records})==len(records)
    for record in records:
        receipt=read(record['receipt'])
        assert receipt['observed_by_parent'] and receipt['returncode']==0
        assert sha(record['receipt'])==record['receipt_sha256']
        assert sha(record['result_path'])==record['result_sha256']
    return records


def run_jobs(jobs,workers,stage):
    valid={r['id'] for r in valid_records()}
    pending=[j for j in jobs if j['id'] not in valid]
    active={};error=None;sampled=0.;pressure=0
    baseline=resource_sample();append(C/'resources.jsonl',dict(baseline,stage=stage))
    deadline=datetime.fromisoformat(read(C/'contract.json')['hard_deadline_utc']).timestamp()
    try:
        while pending or active:
            if time.time()>=deadline:
                # Popen.poll verifies these are still live children of this supervisor.
                for process,job,log,started,attempt in active.values():
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                end=time.monotonic()+30
                while time.monotonic()<end and any(p.poll() is None for p,_,_,_,_ in active.values()):
                    time.sleep(1)
                for process,job,log,started,attempt in active.values():
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGKILL)
                error=error or 'hard deadline reached; owned workers terminated'
            for pid,(process,job,log,started,attempt) in list(active.items()):
                rc=process.poll()
                if rc is None:continue
                log.close();del active[pid]
                receipt={'utc':now(),'id':job['id'],'pid':pid,'returncode':rc,
                         'observed_by_parent':True,'elapsed_s':time.monotonic()-started}
                freeze(attempt/'exit_receipt.json',receipt)
                try:
                    assert rc==0,f'worker exit {rc}: {attempt}/worker.log'
                    if job['kind']=='training':
                        result_path=C/'checkpoint'/('completed.json' if job['stop']==60000 else 'process_0_300.json')
                        result=read(result_path)
                        assert result['global_step']==job['stop'] and result['strict_pair_reload_exact']
                    else:
                        result_path=Path(job['directory'])/'result.json'
                        result=read(result_path)
                        assert result['status']=='complete'
                        assert read(Path(job['directory'])/'initial_pairing.json')['passed']
                        assert result['policy_info']['control_steps']==900
                    append(C/'valid_ledger.jsonl',{'id':job['id'],'kind':job['kind'],
                        'directory':job['directory'],'receipt':str(attempt/'exit_receipt.json'),
                        'receipt_sha256':sha(attempt/'exit_receipt.json'),
                        'result_path':str(result_path),'result_sha256':sha(result_path),'utc':now()})
                except Exception as exc:
                    error=repr(exc)
                    append(C/'invalid_ledger.jsonl',{'utc':now(),'id':job['id'],'error':error,
                        'returncode':rc,'receipt':str(attempt/'exit_receipt.json')})
            if time.monotonic()-sampled>=60:
                resource=resource_sample()
                append(C/'resources.jsonl',dict(resource,stage=stage,active=len(active)))
                over=resource['ram_fraction']>.80 or resource['pid_fraction']>.75 or resource['vram_fraction']>.85
                pressure=pressure+1 if over else 0
                if pressure>=3 or resource['oom_kill']>baseline['oom_kill'] or resource['disk_free_bytes']<10*2**30:
                    error=error or 'resource guard stopped launches'
                atomic(C/'progress.json',{'utc':now(),'stage':stage,'active_jobs':[j['id'] for _,j,_,_,_ in active.values()],
                    'pending':len(pending),'completed':len(valid_records()),'paused':error,'resources':resource})
                print(json.dumps({'utc':now(),'stage':stage,'active':len(active),'pending':len(pending),'paused':error}),flush=True)
                sampled=time.monotonic()
            if time.time()>deadline-3600:
                error=error or 'deadline reserve reached'
            while pending and len(active)<workers and error is None:
                if pending[0]['kind']=='evaluation' and time.time()+1400>deadline-3600:
                    error='evaluation completion would enter the deadline reserve'
                    break
                resource=resource_sample()
                if resource['pid_fraction']>=.75 or resource['ram_fraction']>=.80:
                    break
                job=pending[0]
                attempt=C/'attempts'/job['id']/'attempt_00'
                attempt.mkdir(parents=True,exist_ok=True)
                assert not (attempt/'launch.json').exists(),'preserve failed attempt before retry'
                log=(attempt/'worker.log').open('w')
                try:
                    process=subprocess.Popen(job['command'],cwd=ROOT,env=environment(),stdout=log,
                                             stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:
                    log.close();raise
                active[process.pid]=(process,job,log,time.monotonic(),attempt)
                pending.pop(0)
                freeze(attempt/'launch.json',{'utc':now(),'pid':process.pid,'parent_pid':os.getpid(),
                    'command':job['command'],'id':job['id'],
                    'proc_start_ticks':Path(f'/proc/{process.pid}/stat').read_text().split()[21]})
            if error and not active:
                raise RuntimeError(error)
            time.sleep(1)
    finally:
        # Drain every owned child and record actual exits on parent errors as well.
        for process,job,log,started,attempt in active.values():
            rc=process.wait();log.close()
            freeze(attempt/'exit_receipt.json',{'utc':now(),'id':job['id'],'pid':process.pid,
                'returncode':rc,'observed_by_parent':True,'elapsed_s':time.monotonic()-started,
                'drained_after_parent_exception':True})


def replay_check(scene):
    directory=C/'rollouts'/f'frozen60000_{scene["scene_id"][:20]}'
    old=Path(scene['frozen_directory'])
    comparison={}
    with np.load(directory/'actions.npz') as a,np.load(old/'actions.npz') as b:
        for key in ('model_output','arm','gripper'):
            comparison[key]={'exact':np.array_equal(a[key],b[key]),
                             'max_abs':float(np.max(np.abs(a[key]-b[key])))}
    result=read(directory/'result.json');prior=read(old/'result.json')
    comparison['same_task_success']=result['task_success']==prior['task_success']
    comparison['same_cfts']=result['collision_free_task_success']==prior['collision_free_task_success']
    freeze(C/'frozen_replay_check.json',comparison)
    assert all(comparison[k]['exact'] for k in ('model_output','arm','gripper')), 'frozen replay action mismatch requires diagnosis'
    assert comparison['same_task_success'] and comparison['same_cfts']


def finish():
    from pact_v1010b_metrics import recompute,summarize,compare_rows
    contract=verify_contract(full=True)
    records=valid_records()
    manifest=read(C/'evaluation_manifest.json')
    frozen=[];candidate=[]
    for scene in manifest['scenes']:
        frozen.append(recompute(scene['frozen_directory']))
        directory=C/'rollouts'/f'readout60000_{scene["scene_id"][:20]}'
        candidate.append(recompute(directory))
        assert read(directory/'initial_pairing.json')['passed']
    assert len(frozen)==len(candidate)==50
    comparison=compare_rows(candidate,frozen)
    doc={'schema':SCHEMA,'utc':now(),'training_seed':3103,'n':50,'comparison':comparison,
         'frozen_rows':frozen,'readout_rows':candidate,'contract_sha256':contract['sha256'],
         'interpretation':'Full main128-readout method versus frozen32 on exposed matched scenes; no isolation of unfreezing alone'}
    freeze(C/'comparison.json',doc)
    old,new=comparison['control'],comparison['candidate']
    text=['# Seed3103: main fine-tuned readout versus frozen PACT','',
          'Completed60000 new PACT updates and50 matched readout evaluations. The original frozen50 outcomes were verified and reused; one extra frozen replay checked the unchanged evaluator.','',
          '| Endpoint | Frozen32 | Fine-tuned128 |','|---|---:|---:|']
    for key,label in [('success','Task success /50'),('cfts','Collision-free task success /50'),
                      ('pickup','Exclusive pickup failures /50'),('union_frames','Hazard/clutter union frames'),
                      ('avoidance_percent','Frame avoidance %')]:
        text.append(f'| {label} | {old[key]} | {new[key]} |')
    text+=['','Paired wins/losses: '+json.dumps(comparison['paired']), '',
           'This uses the same50 exposed seed3103 scenes, so it is a regression comparison. It changes fine-tuning,32→128 readout and pooling together, as required to reproduce main. It does not establish an unfreezing-only effect or performance on other seeds.', '',
           'The exact upstream modules are retained with byte hashes. The original dataset split, non-proximity inputs, action semantics, task geometry and history100 controller were preserved. Same-update policy and encoder hashes were validated before inference. Full raw endpoints and comparisons are in comparison.json.','']
    (C/'EVAL.md').write_text('\n'.join(text))
    freeze(C/'final_verification.json',{'utc':now(),'raw_recomputation':True,'matched_pairs':50,
        'original_source_hashes_unchanged':True,'completed_training_updates':60000,
        'valid_observed_exit_records':len(records),'all_initial_pairings_passed':True})
    return comparison


def supervise():
    lock=(C/'parent.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    contract=verify_contract(full=True)
    atomic(C/'run_status.json',{'utc':now(),'status':'RUNNING','active_parent':True,'parent_pid':os.getpid()})
    status='INCOMPLETE';reason=None
    try:
        scenes=read(C/'evaluation_manifest.json')['scenes']
        run_jobs([eval_job(scenes[0],'frozen60000')],1,'frozen_replay')
        replay_check(scenes[0])
        run_jobs([train_job(300)],1,'training_pilot')
        pilot=read(C/'checkpoint/process_0_300.json')
        estimated=pilot['elapsed_s']/300*59700*1.25+6*1400+3600
        available=datetime.fromisoformat(contract['hard_deadline_utc']).timestamp()-time.time()
        freeze(C/'pilot_budget.json',{'utc':now(),'remaining_estimate_s':estimated,'available_s':available,
                                     'pilot_updates':300,'passed':estimated<available})
        assert estimated<available,'measured full-run ETA exceeds budget'
        run_jobs([train_job(60000)],1,'training')
        run_jobs([eval_job(scene,'readout60000') for scene in scenes],8,'evaluation')
        finish()
        status='COMPLETED_COMPARISON'
    except BaseException as exc:
        reason=repr(exc)
        atomic(C/f'error_{time.time_ns()}.json',{'utc':now(),'error':reason,'traceback':traceback.format_exc()})
        raise
    finally:
        closure={'utc':now(),'status':status,'reason':reason,'parent_pid':os.getpid(),
                 'active_parent':False,'valid_records':len(lines(C/'valid_ledger.jsonl'))}
        atomic(C/'parent_closure.json',closure)
        atomic(C/'run_status.json',closure)
        atomic(C/'progress.json',dict(closure,stage='closed',active_jobs=[]))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','supervise','verify'))
    args=parser.parse_args()
    if args.stage=='prepare':
        print(json.dumps(prepare()),flush=True)
    elif args.stage=='verify':
        print(json.dumps(finish()),flush=True)
    else:
        supervise()
