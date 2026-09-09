"""Bounded paired evaluation; success requires ledger/raw-artifact agreement."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import signal
import shutil
import subprocess
import sys
import time
import traceback
from pact_place_v1011c_experiment import *
from pact_place_v1011c_metrics import metrics
from pact_place_v1011c_pairing import check_pair


def payload(row,arm,role,checkpoint_hash=sha256_file):
    key = f'{arm.lower()}_seed{row["checkpoint_seed"]}'
    model = read(WORK / 'models' / (key+'.json'))
    assert model['verified'] and model['returncode'] == 0
    hashes = model['verification']['hashes']
    directory = TRAIN / key
    assert checkpoint_hash(directory / 'policy_best.ckpt') == hashes['policy_best.ckpt']
    schedule = {'arm':arm,'checkpoint_seed':row['checkpoint_seed'],
        'episode_id':row['episode_id'],'candidate_index':row['candidate_index'],
        'row_sha256':row['row_sha256'],'checkpoint_sha256':hashes['policy_best.ckpt'],
        'stats_sha256':hashes['dataset_stats.pkl'],
        'rollout_id':f'v1011c_{role}_{key}_{row["candidate_index"]:03d}_{row["episode_id"][:16]}'}
    schedule['schedule_row_sha256'] = digest(schedule)
    output = EVAL / role / key / f'{row["candidate_index"]:03d}_{row["episode_id"][:16]}'
    command = [sys.executable,str(ROOT / 'scripts/eval_pact_place_v1011c_row.py'),
        '--arm',arm,'--checkpoint-dir',str(directory),'--manifest',str(EVAL / 'eval_manifest.json'),
        '--episode-id',row['episode_id'],'--checkpoint-seed',str(row['checkpoint_seed']),
        '--checkpoint-sha256',schedule['checkpoint_sha256'],'--stats-sha256',schedule['stats_sha256'],
        '--schedule-row-sha256',schedule['schedule_row_sha256'],'--rollout-id',schedule['rollout_id'],
        '--output-dir',str(output),'--save-video','--h5-only']
    if arm == 'PACT':
        command += ['--surface-encoder',ENCODER_PATH,'--surface-encoder-sha256',ENCODER_SHA256]
    return {'schedule':schedule,'output':str(output),'command':command}


def run_one(job):
    directory = Path(job['output'])
    directory.mkdir(parents=True,exist_ok=True)
    result = {**job['schedule'],'directory':str(directory.relative_to(ROOT))}
    assert not (directory / 'result.json').exists(), 'Terminal artifacts must be reconciled with an actual exit-code receipt before reuse.'
    assert not (directory / 'initial_observation_accepted.json').exists(), 'Prior accepted observation has no terminal result; refusing automatic retry.'
    assert not (directory / 'rollout.log').exists(), 'Prior failed attempt requires explicit diagnosis.'
    started = time.time()
    with (directory / 'rollout.log').open('x') as stream:
        child = subprocess.Popen(job['command'],cwd=ROOT,env=environment(),stdout=stream,
            stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code = child.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
            code = -9
    result.update(returncode=code,elapsed_s=time.time()-started,status='failed')
    if code == 0:
        try:
            result['metrics'] = metrics(directory,job['schedule'])
            result['status'] = 'complete'
        except Exception:
            result['validation_error'] = traceback.format_exc()
    if result['status'] != 'complete':
        result['output_tail'] = (directory / 'rollout.log').read_text()[-8000:]
    freeze(directory/'worker_completion.json', result)
    return result


def reconcile_resume(jobs, ledger, adopted):
    """Keep exactly one successful, actually observed completion per frozen job."""
    by_id = {j['schedule']['rollout_id']: j for j in jobs}
    assert len(by_id) == len(jobs)
    recorded = [json.loads(line) for line in ledger.read_text().splitlines()]
    seen = {}
    for result in recorded + adopted:
        job = by_id[result['rollout_id']]
        assert all(result[k] == v for k,v in job['schedule'].items())
        assert ROOT/result['directory'] == Path(job['output'])
        assert result['status'] == 'complete' and result['returncode'] == 0, 'A failed attempt cannot be silently retried.'
        if result.get('adopted_after_scheduler_drain'):
            evidence = result['exit_code_evidence']
            assert evidence['source'] == '/proc/PID/stat field 52 while zombie'
            assert os.waitstatus_to_exitcode(evidence['linux_wait_status']) == result['returncode']
        assert not result.get('reused_validated_result'), 'Inferred exit code is not sufficient.'
        metrics(Path(job['output']), job['schedule'])
        identifier = result['rollout_id']
        if identifier in seen:
            assert result == seen[identifier], 'Conflicting duplicate completion receipt'
        else:
            seen[identifier] = result
    assert len({r['rollout_id'] for r in recorded}) == len(recorded)
    additions = [r for key,r in seen.items() if key not in {x['rollout_id'] for x in recorded}]
    return recorded + additions, additions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage',choices=('smoke','full'),required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--h5-only',action='store_true',required=True)
    args = p.parse_args()
    assert 1 <= args.workers <= 12
    assert not args.resume or args.stage == 'full'
    assert read(WORK / 'training_verification.json')['verified']
    manifest = load_eval_manifest(EVAL / 'eval_manifest.json')
    implementation_paths = ('scripts/eval_pact_place_v1011c_row.py',
        'scripts/pact_place_v1011c_metrics.py','scripts/pact_place_v1011c_experiment.py',
        'submodules/act/eval_pact_place_row.py','submodules/act/eval_pact_place_v109_row.py',
        'submodules/act/eval_pact_frontend_screen_row.py','submodules/act/eval_pact_collision_row.py',
        'submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py',
        'submodules/molmospaces/molmo_spaces/tasks/pact_place_contact_audit.py')
    freeze(EVAL / ('implementation_bindings_full.json' if args.stage=='full' else 'implementation_bindings.json'),{**empty_authorization(),
        'files':{path:sha256_file(ROOT/path) for path in implementation_paths}})
    role = args.stage
    # The reference pipeline's smoke is four disjoint instances, both arms.
    # Other frozen smoke rows remain reserved and are not counted as run.
    rows = ([r for r in manifest['smoke']['rows'] if r['checkpoint_seed']==SEEDS[0]]
            if role=='smoke' else manifest['rows'])
    if role == 'full':
        smoke = read(EVAL / ('smoke_reconciliation.json' if (EVAL/'smoke_reconciliation.json').exists() else 'smoke_run.json'))
        assert smoke['infrastructure_healthy'] and smoke['rollouts_complete']==8
        if 'pairing_implementation_sha256' in smoke:
            assert sha256_file(ROOT/'scripts/pact_place_v1011c_pairing.py')==smoke['pairing_implementation_sha256']
    report = EVAL / (role+'_run.json')
    if report.exists():
        prior = read(report)
        assert prior['infrastructure_healthy']
        for result in prior['results']:
            assert result['status']=='complete' and result['returncode']==0
            metrics(ROOT / result['directory'], result)
        print(f'{role}: all {len(prior["results"])} existing results revalidated',flush=True)
        return
    schedule_path = EVAL / (role+'_schedule.json')
    if args.resume:
        frozen_schedule = read(schedule_path)
        assert frozen_schedule['manifest_sha256'] == manifest['manifest_sha256']
        jobs = frozen_schedule['jobs']
        # Recreate the schedule with checkpoint hashing cached once per model.
        from functools import lru_cache
        cached_hash = lru_cache(maxsize=None)(sha256_file)
        assert jobs == [payload(row,arm,role,cached_hash) for row in rows for arm in ('ACT','PACT')]
    else:
        jobs = [payload(row,arm,role) for row in rows for arm in ('ACT','PACT')]
        freeze(schedule_path,{**empty_authorization(),'manifest_sha256':manifest['manifest_sha256'],'jobs':jobs})
    ledger = EVAL / (role+'_ledger.jsonl')
    started = time.time()
    results = []
    previous_elapsed_hours = 0
    if args.resume:
        assert ledger.exists()
        resize = EVAL/'scheduler_resize_01'
        assert read(resize/'drained.json')['all_actual_returncodes_zero']
        assert read(resize/'pause.json')['schedule_sha256'] == sha256_file(schedule_path)
        adopted = [read(path) for path in sorted(resize.glob('v1011c_full_*.json'))]
        assert len(adopted) == read(resize/'drained.json')['rollouts_drained']
        adopted += [read(Path(job['output'])/'worker_completion.json') for job in jobs
                    if (Path(job['output'])/'worker_completion.json').exists()]
        results, additions = reconcile_resume(jobs, ledger, adopted)
        original_stage = read(WORK/'stages/11_eval_full.json')
        assert original_stage['returncode'] == -9
        previous_elapsed_hours = original_stage['elapsed_s']/3600
        freeze(EVAL/'full_resume_01.json', {**empty_authorization(), 'workers': args.workers,
            'started_unix_s': started, 'completed_before_resume': len(results),
            'adopted_additions': len(additions), 'prior_ledger_sha256': sha256_file(ledger),
            'scheduler_sha256': sha256_file(Path(__file__)),
            'original_stage_returncode': original_stage['returncode']})
        with ledger.open('a') as stream:
            for row in additions:
                stream.write(json.dumps(row,sort_keys=True)+'\n')
            stream.flush();os.fsync(stream.fileno())
        print(f'{role}: resumed {len(results)}/{len(jobs)} actual-exit/raw-verified completions; workers={args.workers}',flush=True)
    else:
        assert not ledger.exists(), 'Existing ledger requires reconciliation before restart.'
    done = {r['rollout_id'] for r in results}
    remaining_jobs = [j for j in jobs if j['schedule']['rollout_id'] not in done]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool,ledger.open('a' if args.resume else 'x') as stream:
        pending = {}
        iterator = iter(remaining_jobs)
        def submit():
            job = next(iterator,None)
            if job is not None:
                assert shutil.disk_usage(ROOT).free / 2**30 > 3, 'Disk reserve exhausted'
                pending[pool.submit(run_one,job)] = job
        for _ in range(args.workers):
            submit()
        halted = False
        while pending:
            completed,_ = concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in completed:
                job = pending.pop(future)
                try:
                    result = future.result()
                except Exception:
                    result = {**job['schedule'],'directory':str(Path(job['output']).relative_to(ROOT)),
                        'status':'failed','returncode':-1,'error':traceback.format_exc()}
                results.append(result)
                stream.write(json.dumps(result,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
                halted |= result['status'] != 'complete' or result['returncode'] != 0
                complete = sum(r['status']=='complete' and r['returncode']==0 for r in results)
                print(f'{role}: {complete}/{len(jobs)} verified complete; {len(results)-complete} failures; elapsed {(time.time()-started)/3600:.2f} h',flush=True)
                if not halted:
                    submit()
    recorded = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(recorded) == len(results)
    complete = [r for r in recorded if r['status']=='complete' and r['returncode']==0]
    by_id = {}
    pair_checks = []
    for result in complete:
        m = metrics(ROOT / result['directory'],result)
        by_id.setdefault(result['episode_id'],{})[result['arm']] = m
    pairing_errors = []
    for identifier,pair in by_id.items():
        if set(pair)=={'ACT','PACT'}:
            try:
                pair_checks.append(check_pair(pair['ACT'],pair['PACT']))
            except Exception:
                pairing_errors.append({'episode_id':identifier,'error':traceback.format_exc()})
    healthy = len(complete)==len(jobs) and len(pair_checks)==len(rows) and not pairing_errors
    doc = {**empty_authorization(),'stage':role,'rollouts_expected':len(jobs),'rollouts_attempted':len(recorded),
        'rollouts_complete':len(complete),'infrastructure_healthy':healthy,'results':recorded,
        'pair_checks':pair_checks,'pairing_errors':pairing_errors,'ledger_sha256':sha256_file(ledger),
        'elapsed_hours':previous_elapsed_hours+(time.time()-started)/3600,
        'resumed_scheduler_elapsed_hours':(time.time()-started)/3600,
        'previous_scheduler_elapsed_hours':previous_elapsed_hours,
        'workers':args.workers,'worker_history':[4,args.workers] if args.resume else [args.workers],
        'scheduler_resize_receipt':'scheduler_resize_01/drained.json' if args.resume else None,'h5_only':True,
        'smoke_gates_on_performance':False,'unused_reserved_smoke_rows':8 if role=='smoke' else 0}
    freeze(report,doc)
    print(json.dumps({k:doc[k] for k in ('rollouts_expected','rollouts_attempted','rollouts_complete','infrastructure_healthy','elapsed_hours')}),flush=True)
    if not healthy:
        print(json.dumps({'pairing_errors':pairing_errors}),flush=True)
        for r in recorded:
            if r['status']!='complete':
                print(json.dumps(r,indent=2),flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
