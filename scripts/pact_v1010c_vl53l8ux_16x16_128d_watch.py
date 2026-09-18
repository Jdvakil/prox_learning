#!/usr/bin/env python3
"""Hourly read-only health checks; writes reports, never alters scientific runs."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import time
from datetime import datetime,timezone
from pathlib import Path

import pact_v1010c_vl53l8ux_16x16_128d as study


def processes():
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            argv=[a.decode(errors='replace') for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
            state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        if argv and state not in ('Z','X'): found.append(dict(pid=int(proc.name),argv=argv))
    return found


def check(interval):
    issues=[]; current=time.time()
    records=study.completed_jobs()
    accepted={arm:sum(r['arm']==arm for r in records.values()) for arm in (*study.ARMS,'LIVE_PREFLIGHT')}
    proc=processes()
    supervisors=[p for p in proc if any(a.endswith('/pact_v1010c_vl53l8ux_16x16_128d.py') or a=='scripts/pact_v1010c_vl53l8ux_16x16_128d.py' for a in p['argv']) and any(a in ('all','run','preflight','prepare') for a in p['argv'])]
    workers=[p for p in proc if any(a.endswith('/pact_v1010c_vl53l8ux_16x16_128d_eval.py') for a in p['argv'])]
    jobs={p['argv'][p['argv'].index('--job')+1]:p for p in workers if '--job' in p['argv']}
    done=False
    if (study.OUT/'finished.json').exists():
        marker=study.read(study.OUT/'finished.json'); analysis=study.read(study.OUT/'analysis.json')
        study.require(study.sha(study.OUT/'analysis.json')==marker['analysis_sha256'] and
            analysis['complete'] and all(accepted[a]==study.N for a in study.ARMS),'completion audit failed')
        done=True
    if not done and not supervisors: issues.append('Supervisor is absent while the study is incomplete.')
    if len(supervisors)>1: issues.append('Multiple study supervisors found.')
    unfinished=[]
    for directory in sorted(study.OUT.glob('*/**/attempt_*')):
        if not directory.is_dir() or not (directory/'job.json').exists(): continue
        job=study.checked_json(directory/'job.json')
        if job['id'] in records: continue
        worker=jobs.get(str(directory/'job.json'))
        progress=study.read(directory/'progress.json') if (directory/'progress.json').exists() else None
        receipt=study.read(directory/'exit_receipt.json') if (directory/'exit_receipt.json').exists() else None
        elapsed=current-(directory/'progress.json' if progress else directory/'job.json').stat().st_mtime
        unfinished.append(dict(id=job['id'],pid=worker['pid'] if worker else None,
            step=progress['control_step'] if progress else None,progress_age_s=elapsed,exit_recorded=receipt is not None))
        if worker and elapsed>300: issues.append(job['id']+': progress/init is stale for over five minutes.')
        elif not worker and receipt is None and elapsed>120: issues.append(job['id']+': worker absent with no recorded exit.')
        if receipt and receipt['returncode']!=0 and (not receipt.get('retryable') or job['attempt']>0):
            issues.append(job['id']+': unresolved technical failure.')
    resource=study.resources()
    if not done and not study.capacity_ok(resource): issues.append('A configured resource launch limit is exceeded.')
    history=study.lines(study.OUT/'resources.jsonl')
    if history and resource['oom_kill']>history[0]['oom_kill']: issues.append('OOM-kill counter increased during the study.')
    return dict(utc=study.now(),status='complete' if done and not issues else 'issues' if issues else 'running',
        complete=done,accepted=accepted,expected_scientific=study.N,
        supervisor_pids=[p['pid'] for p in supervisors],worker_pids=[p['pid'] for p in workers],
        unfinished=unfinished,resources=resource,issues=issues,interval_seconds=interval,
        next_check_utc=None if done else datetime.fromtimestamp(current+interval,timezone.utc).isoformat(),
        automatic_restarts=False,chat_notifications=False)


def publish(report):
    target=study.OUT/'health'
    study.atomic(target/'latest.json',report); study.append(target/'checks.jsonl',report)
    if report['issues']: study.append(target/'issues.jsonl',report)
    doc=['# L8UX-inspired 16x16 profile health','',f"Updated: {report['utc']}",'',f"Status: **{report['status']}**",'']
    if 'accepted' in report:
        doc += [f"- {arm}: {report['accepted'][arm]}/{study.N}" for arm in study.ARMS]
        doc += [f"- Active workers: {len(report['worker_pids'])}",f"- Supervisor PIDs: {report['supervisor_pids']}",'']
    doc += ['## Issues','']+[('- '+x) for x in report['issues']]
    if not report['issues']: doc += ['No issues detected.']
    doc += ['',f"Next check: {report.get('next_check_utc') or 'none; complete'}",'',
        'The local monitor records health; it does not restart workers, change results, or send messages.','',
        '[Machine-readable status](latest.json) · [History](checks.jsonl) · [Results](../RESULTS.md)','']
    temporary=target/'STATUS.md.tmp'; temporary.write_text('\n'.join(doc)); temporary.replace(target/'STATUS.md')
    print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval',type=int,default=3600); parser.add_argument('--once',action='store_true')
    args=parser.parse_args(); study.require(args.interval>=60,'minimum interval is 60 seconds')
    directory=study.OUT/'health'; directory.mkdir(parents=True,exist_ok=True)
    with (directory/'watchdog.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        study.atomic(directory/'watchdog.json',dict(pid=os.getpid(),started_utc=study.now(),interval_seconds=args.interval))
        while True:
            started=time.monotonic()
            try: report=check(args.interval)
            except Exception as exc:
                report=dict(utc=study.now(),status='monitor_error',complete=False,issues=[f'{type(exc).__name__}: {exc}'])
            publish(report)
            if args.once or report['complete']: break
            time.sleep(max(1.,args.interval-(time.monotonic()-started)))


if __name__=='__main__': main()
