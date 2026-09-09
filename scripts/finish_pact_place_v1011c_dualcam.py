"""Durable stage handoffs; deadline-aware one-seed development and held-out test."""
from pact_place_v1011c_dualcam import *
import subprocess
import time
import traceback
import signal
from datetime import datetime, timezone


def stage(name, arguments):
    receipt = WORK/'stages'/f'{name}.json'
    if receipt.exists():
        result = read(receipt)
        assert result['returncode'] == 0, result
        return result
    log = WORK/'logs'/f'{name}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    command = [sys.executable,*arguments]
    started = time.time()
    print(f'STAGE {name} START {datetime.now(timezone.utc).isoformat()}',flush=True)
    with log.open('x') as stream:
        child = subprocess.Popen(command,cwd=ROOT,env=environment(),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        freeze(WORK/'launches'/f'{name}.json',{'pid':child.pid,'command':command,'started_unix':started})
        code = child.wait()
    result = {**empty_authorization(),'stage':name,'command':command,'returncode':code,
        'elapsed_seconds':time.time()-started,'log':str(log.relative_to(ROOT)),
        'output_tail':log.read_text()[-8000:]}
    freeze(receipt,result)
    print(f'STAGE {name} EXIT {code} AFTER {result["elapsed_seconds"]/60:.2f} min',flush=True)
    if code:
        print(result['output_tail'],flush=True)
        raise RuntimeError(f'{name} actual exit {code}: see {log}')
    return result


def choose_workers(development):
    # Owner amendment at 2026-09-06 00:56 UTC: use 10 or 12 workers.
    # Do not use the original plan's optional 14-worker resource gate.
    workers = 12
    freeze(EVAL/'final_worker_selection.json',{**empty_authorization(),'workers':workers,
        'rule':'Owner requested 10 or 12 workers; use 12, never auto-escalate to 14.',
        'owner_amendment':'worker_cap_amendment_01.json'})
    return workers


def cleanup_orphans():
    removed = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry/'status').read_text()
            if not any(line.startswith('PPid:') and line.split()[1]=='1' for line in status.splitlines()):
                continue
            command = (entry/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if 'multiprocessing.spawn' not in command:
                continue
            cwd = Path(os.readlink(entry/'cwd')).resolve()
            if not cwd.is_relative_to(ROOT):
                continue
            pid = int(entry.name)
            os.kill(pid,signal.SIGTERM)
            removed.append({'pid':pid,'cwd':str(cwd),'command':command,'signal':'SIGTERM'})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    freeze(WORK/'orphan_cleanup.json',{'scoped_orphans_signaled':removed})


def main():
    deadline = datetime.fromisoformat(DEADLINE_UTC).timestamp()
    # The already-running training supervisor owns the children and records
    # their real exits. Never infer its exit from a missing process or log text.
    last_report = 0
    while not (WORK/'training_completion.json').exists():
        for receipt in (WORK/'models').glob('*.json') if (WORK/'models').exists() else []:
            model = read(receipt)
            assert model['verified'] and model['returncode']==0, model
        assert time.time() < deadline, 'Deadline reached while waiting for training; no evaluation declared complete.'
        if time.time()-last_report > 3600:
            progress = {}
            for arm in ('act','pact'):
                paths = sorted((TRAIN/f'{arm}_seed3103'/'progress').glob('step_*.json'))
                progress[arm] = read(paths[-1]) if paths else None
            print('HOURLY_TRAINING_STATUS',json.dumps(progress),flush=True)
            last_report = time.time()
        time.sleep(30)
    assert read(WORK/'training_completion.json')['complete']
    stage('03_unit_tests',['-m','unittest','discover','-s','tests','-p','test_pact_place_v1011c_dualcam.py','-v'])
    stage('04_model_verification',[str(ROOT/'scripts/verify_pact_place_v1011c_dualcam_models.py')])
    stage('05_smoke',[str(ROOT/'scripts/run_pact_place_v1011c_dualcam_eval.py'),'--stage','smoke','--workers','12','--h5-only'])
    stage('06_development',[str(ROOT/'scripts/run_pact_place_v1011c_dualcam_eval.py'),'--stage','development','--workers','12','--h5-only'])
    development = read(EVAL/'development_run.json')
    assert development['infrastructure_healthy'] and development['rollouts_complete']==48
    ranking = []
    for step in MILESTONES:
        arms = development['aggregate'][str(step)]
        ranking.append({'global_step':step,
            'two_arm_task_success_count':sum(arms[a]['counts']['task_success'] for a in ('ACT','PACT')),
            'two_arm_collision_free_success_count':sum(arms[a]['counts']['collision_free_task_success'] for a in ('ACT','PACT')),
            'two_arm_hazard_frames':sum(arms[a]['hazard_contact_frames_total'] for a in ('ACT','PACT'))})
    selected = max(ranking,key=lambda r:(r['two_arm_task_success_count'],r['two_arm_collision_free_success_count'],-r['two_arm_hazard_frames'],-r['global_step']))
    freeze(EVAL/'checkpoint_selection.json',{**empty_authorization(),'selected_global_step':selected['global_step'],
        'ranking_inputs':ranking,'rule':read(WORK/'pilot_plan.json')['checkpoint_selection'],
        'final_test_outcomes_used':False,'development_ledger_sha256':sha256_file(EVAL/'development_ledger.jsonl')})
    workers = choose_workers(development)
    durations = [r['elapsed_s'] for r in development['results']]
    projected_seconds = sum(durations)/len(durations)*100/workers*1.25
    remaining = deadline-time.time()
    freeze(EVAL/'final_eta.json',{'measured_development_mean_rollout_seconds':sum(durations)/len(durations),
        'final_workers':workers,'final_rollouts':100,'projected_seconds_with_25pct_margin':projected_seconds,
        'remaining_to_deadline_seconds':remaining,'fits_deadline_projection':projected_seconds<remaining})
    assert projected_seconds < remaining, 'Measured final evaluation ETA exceeds remaining 12-hour budget; final stage not started.'
    stage('07_final',[str(ROOT/'scripts/run_pact_place_v1011c_dualcam_eval.py'),'--stage','final','--workers',str(workers),
                      '--step',str(selected['global_step']),'--h5-only'])
    stage('08_final_raw_audit',[str(ROOT/'scripts/finalize_pact_place_v1011c_dualcam.py')])
    cleanup_orphans()
    freeze(WORK/'pilot_completion.json',{**empty_authorization(),'complete':True,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'before_deadline':time.time()<deadline,
        'final_rollouts':100,'final_pairs':50,'seed':3103,'selected_global_step':selected['global_step'],
        'analysis_sha256':sha256_file(EVAL/'analysis.json')})
    print('PILOT COMPLETE: 50 raw-verified paired final instances.',flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        freeze(WORK/'pilot_failure.json',{**empty_authorization(),'complete':False,'error':failure,
            'utc':datetime.now(timezone.utc).isoformat()})
        print(failure,flush=True)
        raise
