"""Minute resource sampling and the explicitly requested hourly_run_check task."""
from __future__ import annotations
import argparse
import collections
import shutil
import subprocess
import time
from pact_wrist288_common import *

def resources():
    cg = Path('/sys/fs/cgroup')
    values = {k:(cg/k).read_text().strip() for k in ('memory.current','memory.max','pids.current','pids.max','cpu.max','cpu.stat')}
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    return {'utc':now(),'gpu_used_mib':int(gpu[0]),'gpu_total_mib':int(gpu[1]),'gpu_util_pct':int(gpu[2]),
        'vram_fraction':int(gpu[0])/int(gpu[1]),'ram_fraction':int(values['memory.current'])/int(values['memory.max']),
        'pid_fraction':int(values['pids.current'])/int(values['pids.max']),
        'disk_free_gib':shutil.disk_usage(WORK).free/2**30,'cgroup':values}

def owned_processes():
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            env=(entry/'environ').read_bytes().split(b'\0')
            if f'PACT_WRIST288_OWNER={NAMESPACE}'.encode() not in env: continue
            command=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            status=(entry/'status').read_text()
            ppid=int(next(s.split()[1] for s in status.splitlines() if s.startswith('PPid:')))
            found.append({'pid':int(entry.name),'ppid':ppid,'command':command})
        except (OSError,StopIteration,ValueError): pass
    return found

def hourly_run_check(final=False):
    records=lines(WORK/'collection/ledger.jsonl')
    verified=[];issues=[];exits=[];running=[]
    for path in WORK.rglob('launch.json'):
        if 'precollection_revision_00' in path.parts:continue
        launch=read(path);receipt=path.parent/'exit_receipt.json'
        if receipt.exists():
            value=read(receipt)
            if value['job_sha256']!=launch['job_sha256']:issues.append(f'exit binding mismatch: {path.parent}')
            exits.append({'directory':str(path.parent),'returncode':value['returncode']})
        else:
            alive=Path(f'/proc/{launch["pid"]}').exists()
            running.append({'directory':str(path.parent),'pid':launch['pid'],'alive':alive})
            if not alive:issues.append(f'worker disappeared without exit receipt: {path.parent}')
    for row in records:
        if not row.get('accepted'):continue
        try:
            import h5py
            directory=Path(row['directory'])
            receipt=read(directory/'exit_receipt.json')
            assert receipt['returncode']==row['returncode']==0
            assert sha(directory/'exit_receipt.json')==row['exit_receipt_sha256']
            assert sha(ROOT/row['trajectory_h5'])==row['trajectory_h5_sha256']
            result=read(directory/'worker_result.json')
            assert result['accepted'] and result['row_sha256']==row['row_sha256']
            with h5py.File(ROOT/row['trajectory_h5']) as h:
                assert h['traj_0/success'].shape==(result['episode_steps']+1,) and bool(h['traj_0/success'][-1])
            verified.append(row)
        except Exception as error:issues.append(f'collection validation: {row["attempt_id"]}: {error}')
    accepted=collections.Counter(r['cell'] for r in verified)
    updates={}
    for path in (WORK/'checkpoints').glob('*/epoch_log.jsonl'):
        rows=lines(path)
        updates[path.parent.name]=rows[-1]['global_step'] if rows else 0
        if rows:
            import math
            if not all(math.isfinite(v) for key in ('train','val') for v in rows[-1][key].values()):
                issues.append(f'nonfinite latest training losses: {path.parent}')
    completed={}
    for path in (WORK/'evaluation').glob('*_ledger.jsonl'):
        rows=lines(path)
        count=0
        for row in rows:
            if not row.get('valid_completion'):continue
            try:
                from pact_wrist288_metrics import metrics
                directory=Path(row['directory']);receipt=read(directory/'exit_receipt.json')
                assert receipt['returncode']==row['returncode']==0
                assert sha(directory/'exit_receipt.json')==row['exit_receipt_sha256']
                assert sha(directory/'result.json')==row['result_sha256']
                metrics(directory,row);count+=1
            except Exception as error:issues.append(f'evaluation validation: {row["rollout_id"]}: {error}')
        completed[path.stem]=count
    state=read(WORK/'state.json') if (WORK/'state.json').exists() else {'stage':'implementation'}
    out={**empty_authorization(),'task':'hourly_run_check','utc':now(),'final':final,
        'resources':resources(),'state':state,'accepted_by_cell':dict(accepted),'accepted_total':sum(accepted.values()),
        'collection_attempts':len(records),'training_updates':updates,'validated_rollouts':completed,
        'processes':owned_processes(),'hours_to_deadline':max(0,(DEADLINE-time.time())/3600),
        'observed_exit_codes':exits,'expected_running_workers':running,'verification_issues':issues}
    supervisor=state.get('supervisor_pid')
    out['supervisor_alive']=bool(supervisor and Path(f'/proc/{supervisor}').exists())
    error_logs=[]
    for entry in running:
        log=Path(entry['directory'])/'worker.log'
        if not log.exists():continue
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-5000));tail=stream.read().decode(errors='replace')
        if any(token in tail for token in ('Traceback (most recent call last)','CUDA out of memory','pthread_create failed','can\'t start new thread')):
            error_logs.append({'path':str(log),'tail':tail})
    out['new_worker_errors']=error_logs
    out['owned_orphans']=[p for p in out['processes'] if p['ppid']==1 and 'multiprocessing.spawn' in p['command']]
    previous=lines(WORK/'monitoring/hourly_run_check.jsonl')
    if previous:
        old=previous[-1]
        out['progress_advanced']=(old['accepted_total'],old['training_updates'],old['validated_rollouts']) != (out['accepted_total'],updates,completed)
        out['progress_note']='Inspect worker logs and live throughput when unchanged; counters alone do not establish a stall.'
    elapsed=max(1,time.time()-datetime.fromisoformat(state.get('started_utc',now())).timestamp())
    if state['stage']=='collection' and sum(accepted.values()):
        out['collection_eta_hours']=(288-sum(accepted.values()))*elapsed/sum(accepted.values())/3600
    eta=lines(WORK/'monitoring/eta.jsonl')
    if eta:out['remaining_eta']=eta[-1]
    if not out.get('progress_advanced',True) and running:
        out['live_log_progress']=[{'directory':r['directory'],'latest_log_mtime':(Path(r['directory'])/'worker.log').stat().st_mtime} for r in running]
    append(WORK/'monitoring/hourly_run_check.jsonl',out)
    print(json.dumps({k:out[k] for k in ('utc','accepted_total','training_updates','validated_rollouts','hours_to_deadline')}) ,flush=True)
    return out

def main():
    WORK.mkdir(parents=True,exist_ok=True)
    os.environ.update(environment())
    atomic(WORK/'monitoring/monitor.json',{'pid':os.getpid(),'started_utc':now(),'task':'hourly_run_check'})
    next_hour=0
    while not (WORK/'SAFE_SHUTDOWN.json').exists():
        if time.monotonic()>=next_hour:
            hourly_run_check(); next_hour=time.monotonic()+3600
        sample=resources(); append(WORK/'monitoring/resources.jsonl',sample)
        if time.time()>=SAFE_STOP or sample['disk_free_gib']<10:
            atomic(WORK/'PAUSE.json',{'utc':now(),'reason':'reporting_hour_or_disk_reserve','resources':sample})
        time.sleep(30)
    hourly_run_check(final=True)

if __name__=='__main__':
    main()
