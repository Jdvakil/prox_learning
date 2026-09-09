"""Resumable scheduler with actual exits, raw metrics, and frozen held-out jobs."""
from pact_place_v1011c_dualcam import *
from pact_place_v1011c_dualcam_metrics import metrics, check_pair, aggregate
import run_pact_place_v1011c_eval as old_dispatch
import argparse
import concurrent.futures
import shutil
import time
import traceback
import subprocess
from functools import lru_cache


def payload(row,arm,stage,step):
    key = f'{arm.lower()}_seed3103_step{step}'
    directory = TRAIN/key
    receipt = read(directory/'checkpoint_step.json')
    hashes = {'checkpoint':file_hash(directory/'policy_best.ckpt'),'stats':file_hash(directory/'dataset_stats.pkl')}
    assert hashes['checkpoint'] == receipt['checkpoint_sha256']
    assert hashes['stats'] == receipt['dataset_stats_sha256']
    schedule = {'arm':arm,'checkpoint_seed':3103,'training_global_step':step,
        'episode_id':row['episode_id'],'candidate_index':row['candidate_index'],'row_sha256':row['row_sha256'],
        'checkpoint_sha256':hashes['checkpoint'],'stats_sha256':hashes['stats'],
        'rollout_id':f'dualcam_{stage}_{key}_{row["candidate_index"]:03d}_{row["episode_id"][:16]}'}
    schedule['schedule_row_sha256'] = digest(schedule)
    output = EVAL/stage/key/f'{row["candidate_index"]:03d}_{row["episode_id"][:16]}'
    command = [sys.executable,str(ROOT/'scripts/eval_pact_place_v1011c_dualcam_row.py'),
        '--arm',arm,'--checkpoint-dir',str(directory),'--manifest',str(EVAL/'eval_manifest.json'),
        '--episode-id',row['episode_id'],'--checkpoint-seed','3103',
        '--checkpoint-sha256',hashes['checkpoint'],'--stats-sha256',hashes['stats'],
        '--schedule-row-sha256',schedule['schedule_row_sha256'],'--rollout-id',schedule['rollout_id'],
        '--output-dir',str(output),'--save-video','--h5-only']
    if arm == 'PACT':
        command += ['--surface-encoder',ENCODER_PATH,'--surface-encoder-sha256',ENCODER_SHA256]
    return {'schedule':schedule,'output':str(output),'command':command}


file_hash = lru_cache(maxsize=None)(sha256_file)


def resources(stage, pending):
    sample = {'unix_time':time.time(),'stage':stage,'pending_jobs':pending,
        'memory_current_bytes':int(Path('/sys/fs/cgroup/memory.current').read_text()),
        'pids_current':int(Path('/sys/fs/cgroup/pids.current').read_text()),
        'disk_free_bytes':shutil.disk_usage(ROOT).free}
    for key,command in (
        ('gpu', ['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits']),
        ('gpu_processes', ['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'])):
        result = subprocess.run(command,capture_output=True,text=True,timeout=20)
        sample[key] = result.stdout.strip()
        sample[key+'_returncode'] = result.returncode
    return sample


def run_stage(stage,workers,step=None):
    assert 1 <= workers <= 16
    manifest = load_eval_manifest(EVAL/'eval_manifest.json')
    if stage == 'smoke':
        rows,steps = manifest['smoke']['rows'],[20000]
    elif stage == 'development':
        rows,steps = [r for r in manifest['rows'] if r['role']=='development'],list(MILESTONES)
        assert read(EVAL/'smoke_run.json')['infrastructure_healthy']
    else:
        assert stage == 'final' and step in MILESTONES
        assert read(EVAL/'development_run.json')['infrastructure_healthy']
        assert read(EVAL/'checkpoint_selection.json')['selected_global_step'] == step
        rows,steps = [r for r in manifest['rows'] if r['role']=='final'],[step]
    jobs = [payload(row,arm,stage,s) for s in steps for row in rows for arm in ('ACT','PACT')]
    freeze(EVAL/f'{stage}_schedule.json',{**empty_authorization(),'manifest_sha256':manifest['manifest_sha256'],'jobs':jobs})
    ledger = EVAL/f'{stage}_ledger.jsonl'
    recorded = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()] if ledger.exists() else []
    known = {r['rollout_id']:r for r in recorded}
    assert len(known) == len(recorded)
    results,remaining = [],[]
    with ledger.open('a') as stream:
        for job in jobs:
            receipt = Path(job['output'])/'worker_completion.json'
            if receipt.exists():
                result = read(receipt)
                assert result['status']=='complete' and result['returncode']==0, result
                assert all(result[k]==v for k,v in job['schedule'].items())
                metrics(Path(job['output']),job['schedule'])
                if result['rollout_id'] in known:
                    assert result == known[result['rollout_id']]
                else:
                    stream.write(json.dumps(result,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
                results.append(result)
            else:
                assert job['schedule']['rollout_id'] not in known
                remaining.append(job)
        old_dispatch.metrics = metrics
        old_dispatch.environment = environment
        started = time.time()
        halted = False
        resource_samples = []
        next_sample = time.time()+60
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending,iterator = {},iter(remaining)
            def submit():
                job = next(iterator,None)
                if job is not None:
                    assert shutil.disk_usage(ROOT).free > 3*2**30, '3 GiB disk reserve reached'
                    pending[pool.submit(old_dispatch.run_one,job)] = job
            for _ in range(workers):
                submit()
            while pending:
                done,_ = concurrent.futures.wait(pending,timeout=60,return_when=concurrent.futures.FIRST_COMPLETED)
                if time.time() >= next_sample:
                    sample = resources(stage,len(pending))
                    resource_samples.append(sample)
                    freeze(EVAL/'resources'/f'{stage}_{int(sample["unix_time"])}.json',sample)
                    next_sample = time.time()+60
                for future in done:
                    job = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception:
                        result = {**job['schedule'],'directory':str(Path(job['output']).relative_to(ROOT)),
                            'status':'failed','returncode':None,'error':traceback.format_exc()}
                    results.append(result)
                    stream.write(json.dumps(result,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
                    halted |= result['status']!='complete' or result['returncode']!=0
                    completed = sum(r['status']=='complete' and r['returncode']==0 for r in results)
                    print(f'{stage}: {completed}/{len(jobs)} actual-exit/raw-verified; failures={len(results)-completed}; elapsed={(time.time()-started)/3600:.3f}h',flush=True)
                    if halted:
                        print(json.dumps(result),flush=True)
                    else:
                        submit()
    recorded = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    assert len(recorded) == len(results)
    complete = [r for r in recorded if r['status']=='complete' and r['returncode']==0]
    grouped, raw = {},[]
    for result in complete:
        row = metrics(ROOT/result['directory'],result)
        raw.append(row)
        grouped.setdefault((row['training_global_step'],row['episode_id']),{})[row['arm']] = row
    checks,errors = [],[]
    for key,pair in grouped.items():
        try:
            assert set(pair) == {'ACT','PACT'}
            checks.append({'global_step':key[0],**check_pair(pair['ACT'],pair['PACT'])})
        except Exception:
            errors.append({'pair':key,'error':traceback.format_exc()})
    healthy = len(complete)==len(jobs) and len(checks)*2==len(jobs) and not errors
    doc = {**empty_authorization(),'stage':stage,'workers':workers,'rollouts_expected':len(jobs),
        'rollouts_complete':len(complete),'infrastructure_healthy':healthy,'pair_checks':checks,'pairing_errors':errors,
        'ledger_sha256':sha256_file(ledger),'results':recorded,'raw_metrics':raw,
        'resource_samples':resource_samples,
        'aggregate':{str(s):{a:aggregate([r for r in raw if r['training_global_step']==s and r['arm']==a])
            for a in ('ACT','PACT') if any(r['training_global_step']==s and r['arm']==a for r in raw)} for s in steps}}
    freeze(EVAL/f'{stage}_run.json',doc)
    print(json.dumps({k:doc[k] for k in ('stage','rollouts_expected','rollouts_complete','infrastructure_healthy','aggregate')}),flush=True)
    assert healthy, doc.get('pairing_errors')
    return doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('smoke','development','final'),required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--step',type=int)
    parser.add_argument('--h5-only',action='store_true',required=True)
    args = parser.parse_args()
    run_stage(args.stage,args.workers,args.step)


if __name__ == '__main__':
    main()
