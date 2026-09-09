"""Read-only hourly raw-artifact checks; write only new monitoring reports."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pact_place_v1011c_experiment import *
from pact_place_v1011c_metrics import metrics
from pact_place_v1011c_pairing import check_pair

MONITOR = EVAL/'hourly_monitor'


def processes():
    workers, runners, orphans = [], [], []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            command = (proc/'cmdline').read_bytes().split(b'\0')[:-1]
            worker = any(c.endswith(b'/eval_pact_place_v1011c_row.py') for c in command)
            runner = any(c.endswith(b'/run_pact_place_v1011c_eval.py') for c in command)
            orphan = any(b'multiprocessing.spawn' in c for c in command)
            if not (worker or runner or orphan):
                continue
            if Path(os.readlink(proc/'cwd')).resolve() != ROOT:
                continue
            stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
            if stat[0] == 'Z':
                continue
            row = {'pid':int(proc.name),'ppid':int(stat[1]),'state':stat[0],
                   'threads':int(stat[17]),'rss_GiB':int(stat[21])*os.sysconf('SC_PAGE_SIZE')/2**30}
            if worker:
                env = dict(x.split(b'=',1) for x in (proc/'environ').read_bytes().split(b'\0') if b'=' in x)
                row['thread_caps'] = {key:env.get(key.encode(),b'').decode() for key in THREAD_ENV}
                workers.append(row)
            if runner:
                runners.append(row)
            if orphan and row['ppid'] == 1:
                orphans.append(row)
        except (OSError,UnicodeError,ValueError,IndexError):
            continue
    return workers,runners,orphans


def summarize(pairs):
    groups = {}
    keys = ('task_success','collision_free_task_success','hazard_contact_episode',
            'hazard_frames','hazard_entries','clutter_contact_episode','clutter_stability_events')
    for seed in (*SEEDS,'pooled_completed'):
        chosen = [pair for (s,_),pair in pairs.items() if s==seed or seed=='pooled_completed']
        group = {'paired_instances':len(chosen),'arms':{}}
        for arm in ('ACT','PACT'):
            rows = [p[arm] for p in chosen]
            group['arms'][arm] = {key:sum(r[key] for r in rows) for key in keys}
            group['arms'][arm]['per_object'] = {slot:{key:sum(r['per_object'][slot][key] for r in rows)
                for key in ('contact_episode','contact_frames','contact_entries','stable','stability_event')}
                for slot in ('01','08','09')}
        groups[str(seed)] = group
    return groups


def snapshot():
    started = time.time()
    doc = {**empty_authorization(),'checked_at_utc':datetime.now(timezone.utc).isoformat(),
           'checked_unix_s':started,'errors':[],'verification_passed':False}
    manifest = load_eval_manifest(EVAL/'eval_manifest.json')
    scheduled = read(EVAL/'full_schedule.json')
    assert scheduled['manifest_sha256']==manifest['manifest_sha256']
    jobs = {j['schedule']['rollout_id']:j for j in scheduled['jobs']}
    assert len(jobs)==300
    ledger_bytes = (EVAL/'full_ledger.jsonl').read_bytes()
    # The scheduler may be appending while this read-only snapshot is taken.
    partial = bool(ledger_bytes and not ledger_bytes.endswith(b'\n'))
    if partial:
        ledger_bytes = ledger_bytes[:ledger_bytes.rfind(b'\n')+1]
    ledger = [json.loads(line) for line in ledger_bytes.splitlines()]
    assert len({r['rollout_id'] for r in ledger})==len(ledger)
    doc.update(expected_rollouts=300,ledger_records=len(ledger),
        ledger_snapshot_sha256=hashlib.sha256(ledger_bytes).hexdigest(),partial_append_deferred=partial)
    verified, pair_rows, failed = [], {}, []
    for row in ledger:
        job = jobs[row['rollout_id']]
        assert all(row[k]==v for k,v in job['schedule'].items())
        assert ROOT/row['directory']==Path(job['output'])
        if row['status']!='complete' or row['returncode']!=0:
            failed.append(row)
            continue
        value = metrics(ROOT/row['directory'],job['schedule'])
        assert value==row['metrics'], 'Ledger metrics differ from raw reconstruction'
        verified.append(row)
        pair_rows.setdefault((value['seed'],value['episode_id']),{})[value['arm']] = value
    pairs = {key:pair for key,pair in pair_rows.items() if set(pair)=={'ACT','PACT'}}
    pairing_checks = []
    for key,pair in pairs.items():
        try:
            pairing_checks.append(check_pair(pair['ACT'],pair['PACT']))
        except Exception:
            doc['errors'].append({'pair':list(key),'error':traceback.format_exc()})
    workers,runners,orphans = processes()
    for worker in workers:
        if worker['thread_caps'] != THREAD_ENV:
            doc['errors'].append({'worker':worker,'error':'Worker thread caps differ from frozen environment.'})
    if failed:
        doc['errors'].append({'failed_ledger_records':failed})
    if len(verified)<300 and not runners:
        doc['errors'].append({'error':'Incomplete evaluation has no live scheduler.'})
    doc.update(raw_verified_rollouts=len(verified),failed_rollouts=len(failed),
        completed_pairs=len(pairs),verified_pairs=len(pairing_checks),
        completed_counts={str(seed):{arm:sum(r['checkpoint_seed']==seed and r['arm']==arm for r in verified)
            for arm in ('ACT','PACT')} for seed in SEEDS},
        groups=summarize(pairs),workers=workers,schedulers=runners,orphan_spawn_workers=orphans,
        disk_free_GiB=shutil.disk_usage(ROOT).free/2**30,
        memory_current_GiB=int(Path('/sys/fs/cgroup/memory.current').read_text())/2**30,
        memory_max_GiB=int(Path('/sys/fs/cgroup/memory.max').read_text())/2**30,
        pids_current=int(Path('/sys/fs/cgroup/pids.current').read_text()))
    if doc['disk_free_GiB']<3:
        doc['errors'].append({'error':'Disk free below 3 GiB reserve.'})
    gpu = subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu','--format=csv'],
        cwd=ROOT,env=environment(),capture_output=True,text=True,timeout=15)
    doc['gpu'] = {'returncode':gpu.returncode,'output':gpu.stdout,'error':gpu.stderr}
    durations = [r['elapsed_s'] for r in verified if not r.get('adopted_after_scheduler_drain')]
    if durations and workers:
        mean = statistics.mean(durations[-30:])
        doc['eta_remaining_hours_service_rate'] = (300-len(verified))*mean/len(workers)/3600
        doc['recent_mean_rollout_minutes'] = mean/60
        doc['eta_note'] = 'Service-time estimate; includes active rollouts as whole jobs and excludes final reporting overhead.'
    stage = WORK/'stages/12_analysis.json'
    doc['pipeline_finished'] = stage.exists() and read(stage)['returncode']==0
    if doc['pipeline_finished']:
        assert len(verified)==300 and len(pairing_checks)==150 and not failed
        assert read(EVAL/'analysis.json')['verified']
        if orphans:
            doc['errors'].append({'error':'Worktree-owned orphan spawn workers remain after finalization.','orphans':orphans})
    doc['verification_passed'] = not doc['errors']
    doc['check_elapsed_s'] = time.time()-started
    return doc


def save(doc):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    freeze(MONITOR/'checks'/f'{stamp}.json',doc)
    for filename,content in [('latest.json',json.dumps(doc,indent=2,sort_keys=True)+'\n'),
        ('progress.md', '\n'.join([
            '# V10.11c hourly verification\n',f"Checked: {doc['checked_at_utc']}\n",
            f"Raw-verified completions: {doc.get('raw_verified_rollouts','unavailable')}/300; "
            f"paired checks: {doc.get('verified_pairs','unavailable')}/150; "
            f"recorded failures: {doc.get('failed_rollouts','unavailable')}.\n",
            f"Verification passed: {doc['verification_passed']}; evaluation finished: {doc.get('pipeline_finished',False)}.\n",
            f"Active workers: {len(doc.get('workers',[]))}; estimated hours remaining: "
            f"{doc.get('eta_remaining_hours_service_rate','unavailable')}.\n",
            'Incomplete seed blocks are not a final three-seed average. Full per-seed counts and raw-rederived '
            'outcomes are in `latest.json`; historical snapshots are in `checks/`.\n',
            'These checks do not modify models, rollouts, evaluation rules or authorization flags. '
            'This local monitor writes files; it does not send chat notifications.\n',
            'Errors: '+json.dumps(doc['errors'])+'\n']))]:
        temporary=MONITOR/(filename+'.tmp')
        temporary.write_text(content)
        os.replace(temporary,MONITOR/filename)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    MONITOR.mkdir(exist_ok=True)
    if args.detach:
        command=[sys.executable,str(Path(__file__).resolve())]
        with (MONITOR/'monitor.log').open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=environment(),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        freeze(MONITOR/'launch.json',{**empty_authorization(),'pid':child.pid,'command':command,
            'started_unix_s':time.time(),'interval_seconds':3600})
        print(f'Hourly raw verifier PID {child.pid}; reports: {MONITOR}',flush=True)
        return
    lock=(MONITOR/'monitor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        started=time.time()
        try:
            doc=snapshot()
        except Exception:
            doc={**empty_authorization(),'checked_at_utc':datetime.now(timezone.utc).isoformat(),
                'verification_passed':False,'pipeline_finished':False,'errors':[traceback.format_exc()]}
        doc['next_check_unix_s']=None if doc.get('pipeline_finished') else started+3600
        save(doc)
        print(json.dumps({k:doc.get(k) for k in ('checked_at_utc','raw_verified_rollouts','verified_pairs',
            'failed_rollouts','verification_passed','eta_remaining_hours_service_rate','pipeline_finished','errors')}),flush=True)
        if args.once or doc.get('pipeline_finished'):
            return
        while time.time()<started+3600:
            time.sleep(min(60,started+3600-time.time()))


if __name__=='__main__':
    main()
