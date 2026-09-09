"""Durable, bounded experiment supervisor with receipts and staged scientific gates."""
from __future__ import annotations
import collections
import fcntl
import signal
import subprocess
import time
import traceback
from functools import lru_cache
from pact_wrist288_common import *
from pact_wrist288_monitor import resources,owned_processes,hourly_run_check

class Paused(RuntimeError): pass
POOLS=[]

def stage(name):
    check_bindings()
    if (WORK/'PAUSE.json').exists() or time.time()>=SAFE_STOP: raise Paused('Launch paused: disk/deadline/manual guard')
    atomic(WORK/'state.json',{'stage':name,'started_utc':now(),'supervisor_pid':os.getpid()})
    print(f'[{now()}] stage={name}',flush=True)

class Pool:
    def __init__(self,kind,limit=14):
        POOLS.append(self)
        self.kind=kind;self.limit=limit;self.active={};self.last_launch=0;self.next_sample=0
        self.pressure=0;self.pause_reason=None;self.peak={'vram_fraction':0,'ram_fraction':0,'pid_fraction':0}
    def reduce(self,reason):
        before=self.limit
        if self.limit>10: self.limit={14:12,12:10}.get(self.limit,10)
        else: self.pause_reason=reason
        append(WORK/'monitoring/concurrency_changes.jsonl',{'utc':now(),'stage':self.kind,'before':before,
            'after':self.limit,'reason':reason,'paused':bool(self.pause_reason)})
    def sample(self,force=False):
        if force or time.monotonic()>=self.next_sample:
            value=resources();append(WORK/'monitoring/supervisor_resources.jsonl',{'stage':self.kind,**value})
            for key in self.peak:self.peak[key]=max(self.peak[key],value[key])
            high=value['vram_fraction']>0.85 or value['ram_fraction']>0.80 or value['pid_fraction']>0.75
            self.pressure=self.pressure+1 if high else 0
            if self.pressure>=3:
                self.reduce('three consecutive resource pressure samples');self.pressure=0
            if value['disk_free_gib']<10:self.pause_reason='disk reserve below 10 GiB'
            self.next_sample=time.monotonic()+60
        if (WORK/'PAUSE.json').exists() or time.time()>=SAFE_STOP:self.pause_reason='disk/deadline/manual guard'
    def can_launch(self):
        self.sample()
        return not self.pause_reason and len(self.active)<self.limit and time.monotonic()-self.last_launch>=2
    def launch(self,job):
        directory=Path(job['directory']);directory.mkdir(parents=True,exist_ok=True)
        assert directory.is_relative_to(WORK)
        assert not (directory/'launch.json').exists(),'previous launch requires an observed exit receipt'
        freeze(directory/'job.json',job)
        stream=(directory/'worker.log').open('x')
        process=subprocess.Popen(job['command'],cwd=ROOT,env=environment(),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        launch={'utc':now(),'pid':process.pid,'job_sha256':digest(job),'command':job['command']}
        freeze(directory/'launch.json',launch)
        self.active[process.pid]={'process':process,'stream':stream,'job':job,'started':time.time(),'scan_offset':0}
        self.last_launch=time.monotonic()
    def poll(self):
        self.sample();done=[]
        for pid,meta in list(self.active.items()):
            directory=Path(meta['job']['directory'])
            # Read new errors while the child runs, rather than waiting for hourly checks.
            with (directory/'worker.log').open('r',errors='replace') as log:
                log.seek(meta['scan_offset']);chunk=log.read();meta['scan_offset']=log.tell()
            if any(s in chunk.lower() for s in ('out of memory','pthread_create failed','can\'t start new thread','cannot create thread')):
                self.reduce('OOM/thread creation error; launches paused for diagnosis')
                self.pause_reason='OOM/thread creation error'
            code=meta['process'].poll()
            if code is None:
                if self.pause_reason and self.kind.startswith('train') and not meta.get('stop_sent'):
                    meta['process'].send_signal(signal.SIGTERM);meta['stop_sent']=True
                continue
            meta['stream'].close()
            receipt={'utc':now(),'pid':pid,'returncode':code,'elapsed_s':time.time()-meta['started'],
                'job_sha256':digest(meta['job']),'exit_evidence':'subprocess.Popen.poll / waitpid'}
            freeze(directory/'exit_receipt.json',receipt)
            done.append((meta['job'],receipt));del self.active[pid]
            if code!=0:self.pause_reason=f'worker {pid} exited {code}; inspect {directory}/worker.log'
        return done
    def drain(self):
        while self.active:
            for job,receipt in self.poll():yield job,receipt
            if self.active:time.sleep(1)

def worker_command(script,job_path):return [sys.executable,str(CODE/script),'--job',str(job_path)]

def collection_record(job,receipt):
    directory=Path(job['directory']);row=job['row']
    result=read(directory/'worker_result.json') if (directory/'worker_result.json').exists() else {}
    accepted=receipt['returncode']==0 and result.get('accepted',False)
    if accepted:
        assert result['row']==row and result['schema_validation']['passed']
        assert sha(ROOT/result['trajectory_h5'])==result['trajectory_h5_sha256']
    return {**empty_authorization(),'utc':now(),'attempt_id':row['attempt_id'],'cell':row['cell'],'row_sha256':row['row_sha256'],
        'row':row,'directory':str(directory),'returncode':receipt['returncode'],'accepted':bool(accepted),
        'status':result.get('status','worker_failure'),'defects':result.get('defects',[]),'error':result.get('error'),
        'trajectory_h5':result.get('trajectory_h5'),'trajectory_h5_sha256':result.get('trajectory_h5_sha256'),
        'elapsed_s':receipt['elapsed_s'],'exit_receipt_sha256':sha(directory/'exit_receipt.json')}

def run_preflight():
    stage('preflight');rows=read(WORK/'manifests/preflight.json')['rows'];pool=Pool('preflight');pending=[];results=[]
    for row in rows:
        directory=WORK/'preflight'/row['attempt_id'][:16]
        job={'kind':'preflight','row':row,'directory':str(directory),
            'command':worker_command('pact_wrist288_collect.py',directory/'job.json')}
        if (directory/'exit_receipt.json').exists():
            receipt=read(directory/'exit_receipt.json');assert receipt['returncode']==0 and receipt['job_sha256']==digest(job)
            result=read(directory/'worker_result.json');assert result['passed'];results.append(result)
        else:pending.append(job)
    while pending or pool.active:
        if pending and pool.can_launch():pool.launch(pending.pop(0))
        for job,receipt in pool.poll():
            assert receipt['returncode']==0,job
            result=read(Path(job['directory'])/'worker_result.json');results.append(result)
            if not result['passed']:pool.pause_reason=f'preflight failed: {result}'
        if pool.pause_reason and not pool.active:raise Paused(pool.pause_reason)
        time.sleep(1)
    assert len(results)==24 and all(r['passed'] for r in results)
    freeze(WORK/'preflight_complete.json',{'cells':results,'passed':True,'config_sha256':read(WORK/'config.json')['config_sha256']})

def run_collection():
    from pact_wrist288_prepare import build_row
    from pact_wrist288_data import verify_ledger
    stage('collection');registry=read(WORK/'seed_registry.json');pool=Pool('collection')
    ledger=WORK/'collection/ledger.jsonl';records=lines(ledger)
    seen={r['attempt_id'] for r in records};assert len(seen)==len(records)
    # Adopt only durable exits observed by this supervisor, never apparent files.
    for path in sorted((WORK/'raw').glob('*/job.json')):
        job=read(path)
        if job['row']['attempt_id'] in seen:continue
        assert (path.parent/'exit_receipt.json').exists(),f'launched attempt lacks observed exit: {path.parent}'
        receipt=read(path.parent/'exit_receipt.json');assert receipt['job_sha256']==digest(job)
        record=collection_record(job,receipt);append(ledger,record);records.append(record);seen.add(record['attempt_id'])
    for record in records:
        if record['accepted']:
            assert sha(ROOT/record['trajectory_h5'])==record['trajectory_h5_sha256']
            assert sha(Path(record['directory'])/'exit_receipt.json')==record['exit_receipt_sha256']
    counts=collections.Counter(r['cell'] for r in records if r['accepted'])
    attempted=collections.Counter(r['cell'] for r in records)
    def finish(job,receipt):
        record=collection_record(job,receipt);append(ledger,record);records.append(record)
        if record['accepted']:counts[record['cell']]+=1
        print(f'accepted={sum(counts.values())}/288 attempts={len(records)} cell={record["cell"]} status={record["status"]}',flush=True)
    try:
        while sum(counts.values())<288 or pool.active:
            busy={m['job']['row']['cell'] for m in pool.active.values()}
            eligible=[cell_key(*c) for c in cells() if counts[cell_key(*c)]<12 and cell_key(*c) not in busy]
            if eligible and pool.can_launch():
                cell=min(eligible,key=lambda c:(counts[c],attempted[c],c));ordinal=attempted[cell]
                assert ordinal<128,'per-cell seed registry exhausted; pause without changing quotas'
                row=build_row('collection',cell,ordinal,registry);directory=WORK/'raw'/row['attempt_id'][:16]
                job={'kind':'collection','row':row,'directory':str(directory),
                    'command':worker_command('pact_wrist288_collect.py',directory/'job.json')}
                pool.launch(job);attempted[cell]+=1
            for job,receipt in pool.poll():finish(job,receipt)
            if pool.pause_reason and not pool.active:raise Paused(pool.pause_reason)
            time.sleep(1)
    finally:
        for job,receipt in pool.drain():finish(job,receipt)
    verify_ledger(records)
    freeze(WORK/'collection_complete.json',{'accepted':288,'ledger_sha256':sha(ledger),'by_cell':dict(counts)})

def single_process(script,name):
    stage(name);directory=WORK/'processes'/name
    job={'directory':str(directory),'command':[sys.executable,str(CODE/script)]}
    if (directory/'exit_receipt.json').exists():
        receipt=read(directory/'exit_receipt.json');assert receipt['returncode']==0 and receipt['job_sha256']==digest(job)
        return receipt
    pool=Pool(name,1);pool.launch(job)
    result=list(pool.drain())[0][1]
    if result['returncode']!=0:raise Paused(f'{name} exit {result["returncode"]}; see {directory}/worker.log')
    return result

def train_group(seed,arms,stop,label):
    from pact_wrist288_train import command
    stage('train_'+label);pool=Pool('train_'+label,len(arms));pending=[];receipts=[]
    for arm in arms:
        model=WORK/f'checkpoints/{arm}_seed{seed}'
        processes=sorted(model.glob(f'process_*_{stop}.json'))
        if processes:
            assert len(processes)==1
            r=read(processes[0]);assert r['completed_updates']==stop
            receipts.append(r);continue
        cmd=command(arm,seed,stop);directory=WORK/'processes'/f'{label}_{arm}_{seed}_{stop}'
        job={'directory':str(directory),'command':cmd,'arm':arm,'seed':seed,'stop':stop}
        pending.append(job)
    while pending or pool.active:
        if pending and pool.can_launch():pool.launch(pending.pop(0))
        for job,receipt in pool.poll():
            if receipt['returncode']!=0:continue
            model=WORK/f'checkpoints/{job["arm"]}_seed{seed}'
            matches=list(model.glob(f'process_*_{stop}.json'))
            if len(matches)!=1:pool.pause_reason='trainer did not reach required update boundary';continue
            result=read(matches[0]);assert result['completed_updates']==stop
            receipts.append(result)
        if pool.pause_reason and not pool.active:raise Paused(pool.pause_reason)
        # Benchmark GPU pressure at one-second resolution; resize uses minute samples.
        value=resources()
        for k in pool.peak:pool.peak[k]=max(pool.peak[k],value[k])
        time.sleep(1)
    return receipts,pool.peak

def remaining_eta():
    # Estimate the untouched work from measured collection, optimizer and rollout rates.
    bench=read(WORK/'training_benchmark.json') if (WORK/'training_benchmark.json').exists() else None
    if not bench:return
    done=0
    for path in (WORK/'checkpoints').glob('*/progress.json'):done+=read(path)['updates']
    rate=bench['chosen_updates_per_second']
    records=sum((lines(p) for p in (WORK/'evaluation').glob('*_ledger.jsonl')),[])
    valid=[r for r in records if r.get('valid_completion')]
    eval_seconds=(404-len(valid))*sum(r['elapsed_s'] for r in valid)/len(valid)/14 if valid else 404*8*60/14
    estimate=max(0,360000-done)/rate+max(0,eval_seconds)
    receipt={'utc':now(),'remaining_estimate_hours':estimate/3600,'available_hours':(SAFE_STOP-time.time())/3600,
        'optimizer_rate':rate,'rollout_rate_measured':bool(valid),'rollout_estimate_minutes_if_unmeasured':8}
    append(WORK/'monitoring/eta.jsonl',receipt)
    if estimate>SAFE_STOP-time.time():raise Paused(f'Measured remaining work no longer fits: {receipt}')

def train_pilot():
    serial=[]
    for arm in ('act','pact'):
        receipts,_=train_group(3103,[arm],900,'benchmark_serial');serial+=receipts
    concurrent,peak=train_group(3103,['act','pact'],1800,'benchmark_concurrent')
    serial_rate=1800/sum(r['elapsed_s'] for r in serial)
    concurrent_rate=1800/max(r['elapsed_s'] for r in concurrent)
    keep=concurrent_rate>=1.2*serial_rate and peak['vram_fraction']<0.85 and peak['ram_fraction']<0.8
    freeze(WORK/'training_benchmark.json',{'serial':serial,'concurrent':concurrent,'peak':peak,
        'serial_updates_per_second':serial_rate,'concurrent_updates_per_second':concurrent_rate,
        'keep_two_trainers':keep,'chosen_updates_per_second':concurrent_rate if keep else serial_rate})
    remaining_eta()
    finish_seed(3103)

def finish_seed(seed):
    parallel=read(WORK/'training_benchmark.json')['keep_two_trainers']
    groups=[['act','pact']] if parallel else [['act'],['pact']]
    for stop in (30000,60000):
        for arms in groups:train_group(seed,arms,stop,f'budget_{stop}')
    for arm in ('act','pact'):
        directory=WORK/f'checkpoints/{arm}_seed{seed}'
        logs=lines(directory/'epoch_log.jsonl')
        assert len(logs)==2000 and [r['global_step'] for r in logs]==list(range(30,60001,30))
        assert (directory/'policy_update_30000.ckpt').exists() and (directory/'policy_update_60000.ckpt').exists()
        manifest=read(directory/'run_manifest.json')
        assert len(manifest['train_act_indices'])==240 and len(manifest['val_act_indices'])==48
    a,p=[read(WORK/f'checkpoints/{arm}_seed{seed}/run_manifest.json') for arm in ('act','pact')]
    for key in ('train_act_indices','val_act_indices','dataset_stats_pkl_sha256','seed','num_epochs','episode_horizon','lr','batch_size','chunk_size','statistics'):
        assert a[key]==p[key],f'arm settings differ: {key}'

@lru_cache(maxsize=None)
def cached_sha(path):return sha(path)

def eval_jobs(role,seed,history):
    manifest=WORK/f'manifests/{role}.json';rows=read(manifest)['rows'];jobs=[]
    for row in rows:
        for arm in ('ACT','PACT'):
            model=WORK/f'checkpoints/{arm.lower()}_seed{seed}'
            schedule={'arm':arm,'checkpoint_seed':seed,'episode_id':row['episode_id'],'row_sha256':row['row_sha256'],
                'checkpoint_sha256':cached_sha(model/'policy_update_60000.ckpt'),'stats_sha256':cached_sha(model/'dataset_stats.pkl'),
                'rollout_id':f'{role}_h{history}_{arm}_{row["episode_id"][:16]}','averaging_history':history}
            schedule['schedule_row_sha256']=digest(schedule)
            directory=WORK/'evaluation'/f'{role}_h{history}'/schedule['rollout_id']
            cmd=[sys.executable,str(CODE/'pact_wrist288_eval_worker.py'),'--arm',arm,'--checkpoint-dir',str(model),
                '--manifest',str(manifest),'--episode-id',row['episode_id'],'--checkpoint-seed',str(seed),
                '--checkpoint-sha256',schedule['checkpoint_sha256'],'--stats-sha256',schedule['stats_sha256'],
                '--schedule-row-sha256',schedule['schedule_row_sha256'],'--rollout-id',schedule['rollout_id'],
                '--output-dir',str(directory),'--save-trajectory','--h5-only','--averaging-history',str(history)]
            if arm=='PACT':cmd+=['--surface-encoder',str(ENCODER_PATH),'--surface-encoder-sha256',ENCODER_SHA256]
            jobs.append({'directory':str(directory),'schedule':schedule,'command':cmd})
    return jobs

def run_eval(role,seed,history):
    from pact_wrist288_analysis import validate_completions,check_pair
    from pact_wrist288_metrics import metrics
    stage('evaluation_'+role+f'_h{history}');jobs=eval_jobs(role,seed,history)
    name=f'{role}_h{history}';freeze(WORK/f'evaluation/{name}_schedule.json',jobs)
    ledger=WORK/f'evaluation/{name}_ledger.jsonl';records=lines(ledger);pending=[];pool=Pool(name)
    seen={r['rollout_id'] for r in records};assert len(seen)==len(records)
    def finish(job,receipt):
        directory=Path(job['directory']);valid=False;error=None
        if receipt['returncode']==0:
            try:metrics(directory,job['schedule']);valid=True
            except Exception:error=traceback.format_exc();pool.pause_reason=f'raw completion validation failed: {directory}'
        record={**job['schedule'],'directory':str(directory),'returncode':receipt['returncode'],
            'elapsed_s':receipt['elapsed_s'],'valid_completion':valid,'validation_error':error,
            'exit_receipt_sha256':sha(directory/'exit_receipt.json'),
            'result_sha256':sha(directory/'result.json') if (directory/'result.json').exists() else None}
        append(ledger,record);records.append(record)
        print(f'{name}: {sum(r["valid_completion"] for r in records)}/{len(jobs)} validated',flush=True)
    for job in jobs:
        if job['schedule']['rollout_id'] in seen:continue
        directory=Path(job['directory'])
        if (directory/'exit_receipt.json').exists():
            receipt=read(directory/'exit_receipt.json');assert receipt['job_sha256']==digest(job);finish(job,receipt)
        else:pending.append(job)
    try:
        while pending or pool.active:
            if pending and pool.can_launch():pool.launch(pending.pop(0))
            for job,receipt in pool.poll():finish(job,receipt)
            if pool.pause_reason and not pool.active:raise Paused(pool.pause_reason)
            time.sleep(1)
    finally:
        for job,receipt in pool.drain():finish(job,receipt)
    outcomes=validate_completions(records,jobs)
    pairs=collections.defaultdict(dict)
    for row in outcomes:pairs[row['episode_id']][row['arm']]=row
    for identity,pair in pairs.items():check_pair(pair['ACT'],pair['PACT'],WORK/f'evaluation/pair_audits/{name}_{identity}.json')
    freeze(WORK/f'evaluation/{name}_complete.json',{'count':len(outcomes),'rows':outcomes,'ledger_sha256':sha(ledger)})
    return outcomes

def cleanup_orphans():
    cleaned=[]
    for process in owned_processes():
        if process['ppid']==1 and 'multiprocessing.spawn' in process['command']:
            try:os.kill(process['pid'],signal.SIGTERM);cleaned.append(process)
            except ProcessLookupError:pass
    return cleaned

def main():
    os.environ.update(environment());WORK.mkdir(parents=True,exist_ok=True)
    lock=(WORK/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (WORK/'SAFE_SHUTDOWN.json').exists(),'closed run requires an explicit reviewed continuation'
    from pact_wrist288_analysis import report,select_history
    verdict='INCOMPLETE';reason='';outcomes=[]
    try:
        run_preflight();run_collection();single_process('pact_wrist288_data.py','conversion');train_pilot()
        run_eval('smoke',3103,100)
        development=run_eval('development',3103,100)+run_eval('development',3103,10)
        choice=select_history(development);freeze(WORK/'development_gate.json',choice)
        if not choice['passed']:
            verdict='DEVELOPMENT GATE MISSED';reason=f'Prespecified development gate missed: {choice}. Seeds 3104/3105 and final evaluation were not launched.'
            outcomes=[r for r in development if r['averaging_history']==choice['history']]
        else:
            remaining_eta()
            for seed in (3104,3105):finish_seed(seed);remaining_eta()
            for seed in SEEDS:outcomes+=run_eval(f'final_{seed}',seed,choice['history'])
            assert len(outcomes)==300 and len({(r['seed'],r['episode_id'],r['arm']) for r in outcomes})==300
            success={arm:sum(r['task_success'] for r in outcomes if r['arm']==arm) for arm in ('ACT','PACT')}
            met=success['PACT']>=76 and success['PACT']-success['ACT']>=15
            verdict='TARGET MET' if met else 'TARGET MISSED'
            reason=f'Completed all 300 valid paired rollouts. Pooled task successes: {success}. Experimental target; no significance claim.'
    except BaseException as exc:
        reason=f'{type(exc).__name__}: {exc}'
        atomic(WORK/'stop_error.json',{'utc':now(),'error':reason,'traceback':traceback.format_exc()})
        print(traceback.format_exc(),flush=True)
    finally:
        for pool in POOLS:
            pool.pause_reason=pool.pause_reason or 'supervisor shutdown'
            for job,receipt in pool.drain():
                append(WORK/'monitoring/shutdown_drain.jsonl',{'job':job,'receipt':receipt})
        report(verdict,reason,outcomes)
        cleaned=cleanup_orphans();hourly_run_check(final=True)
        atomic(WORK/'SAFE_SHUTDOWN.json',{**empty_authorization(),'utc':now(),'verdict':verdict,'reason':reason,'cleaned_owned_orphans':cleaned})
        print(f'{verdict}: {reason}',flush=True)

if __name__=='__main__':main()
