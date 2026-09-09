"""Hourly read-only pilot checks, with durable reports in the new experiment only."""
from pact_place_v1011c_dualcam import *
from pact_place_v1011c_dualcam_metrics import metrics, check_pair, aggregate
import argparse
import fcntl
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone

MONITOR = WORK / 'hourly_monitor'
EXPECTED = {'smoke': 8, 'development': 48, 'final': 100}


def processes():
    found = {'workers': [], 'controllers': [], 'orphan_spawn_workers': []}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')[:-1]
            worker = any(a.endswith(b'/eval_pact_place_v1011c_dualcam_row.py') for a in args)
            controller = any(a.endswith(b'/finish_pact_place_v1011c_dualcam.py') for a in args)
            spawn = any(b'multiprocessing.spawn' in a for a in args)
            if not (worker or controller or spawn):
                continue
            if not Path(os.readlink(proc / 'cwd')).resolve().is_relative_to(ROOT):
                continue
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z':
                continue
            row = {'pid': int(proc.name), 'ppid': int(stat[1]), 'state': stat[0],
                   'threads': int(stat[17]),
                   'rss_gib': int(stat[21]) * os.sysconf('SC_PAGE_SIZE') / 2**30}
            if worker:
                env = dict(x.split(b'=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if b'=' in x)
                row['thread_caps'] = {k: env.get(k.encode(), b'').decode() for k in THREAD_ENV}
                found['workers'].append(row)
            if controller:
                found['controllers'].append(row)
            if spawn and row['ppid'] == 1:
                found['orphan_spawn_workers'].append(row)
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return found


def snapshot():
    doc = {**empty_authorization(), 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
           'errors': [], 'stages': {}, 'pipeline_finished': False}
    manifest = load_eval_manifest(EVAL / 'eval_manifest.json')
    for stage, expected in EXPECTED.items():
        schedule_path = EVAL / f'{stage}_schedule.json'
        if not schedule_path.exists():
            doc['stages'][stage] = {'expected': expected, 'started': False, 'raw_verified': 0}
            continue
        schedule = read(schedule_path)
        assert schedule['manifest_sha256'] == manifest['manifest_sha256']
        jobs = {j['schedule']['rollout_id']: j for j in schedule['jobs']}
        assert len(jobs) == len(schedule['jobs']) == expected
        ledger_path = EVAL / f'{stage}_ledger.jsonl'
        content = ledger_path.read_bytes() if ledger_path.exists() else b''
        partial = bool(content and not content.endswith(b'\n'))
        if partial:
            content = content[:content.rfind(b'\n') + 1]
        ledger = [json.loads(line) for line in content.splitlines() if line.strip()]
        assert len({r['rollout_id'] for r in ledger}) == len(ledger)
        rows, pairs, failed = [], {}, []
        for receipt in ledger:
            job = jobs[receipt['rollout_id']]
            assert all(receipt[k] == v for k, v in job['schedule'].items())
            assert ROOT / receipt['directory'] == Path(job['output'])
            if receipt['status'] != 'complete' or receipt['returncode'] != 0:
                failed.append(receipt)
                continue
            assert receipt == read(Path(job['output']) / 'worker_completion.json')
            row = metrics(Path(job['output']), job['schedule'])
            assert row == receipt['metrics'], 'Raw metrics differ from ledger'
            rows.append(row)
            pair = pairs.setdefault((row['training_global_step'], row['episode_id']), {})
            assert row['arm'] not in pair
            pair[row['arm']] = row
        checks = [check_pair(p['ACT'], p['PACT']) for p in pairs.values() if set(p) == {'ACT', 'PACT'}]
        groups = {(r['training_global_step'], r['arm']) for r in rows}
        doc['stages'][stage] = {'expected': expected, 'started': True, 'ledger_rows': len(ledger),
            'raw_verified': len(rows), 'verified_pairs': len(checks), 'failed': len(failed),
            'partial_append_deferred': partial, 'ledger_snapshot_sha256': hashlib.sha256(content).hexdigest(),
            'groups': {f'{step}_{arm}': aggregate([r for r in rows if (r['training_global_step'], r['arm']) == (step, arm)])
                       for step, arm in sorted(groups)}}
        if failed:
            doc['errors'].append({'stage': stage, 'failed_receipts': failed})
    doc.update(processes())
    if any(w['thread_caps'] != THREAD_ENV for w in doc['workers']):
        doc['errors'].append('A live worker is missing thread-pool caps.')
    if len(doc['workers']) > 12:
        doc['errors'].append('Worker count exceeds the owner cap of 12.')
    doc['disk_free_gib'] = shutil.disk_usage(ROOT).free / 2**30
    doc['cgroup'] = {k: Path('/sys/fs/cgroup', k).read_text().strip() for k in
                     ('memory.current', 'memory.max', 'memory.events', 'pids.current', 'pids.max', 'pids.events')}
    if doc['disk_free_gib'] < 3:
        doc['errors'].append('Free disk is below the 3 GiB reserve.')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                          '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
    doc['gpu'] = {'returncode': gpu.returncode, 'output': gpu.stdout.strip(), 'error': gpu.stderr.strip()}
    failure = WORK / 'pilot_failure.json'
    if failure.exists():
        doc['errors'].append({'pipeline_failure': read(failure)})
    completion = WORK / 'pilot_completion.json'
    if completion.exists():
        assert read(completion)['complete']
        assert read(WORK / 'stages/08_final_raw_audit.json')['returncode'] == 0
        assert read(EVAL / 'analysis.json')['verified']
        assert doc['stages']['final']['raw_verified'] == 100
        assert doc['stages']['final']['verified_pairs'] == 50
        if doc['orphan_spawn_workers']:
            doc['errors'].append('Scoped orphan spawn workers remain after completion.')
        doc['pipeline_finished'] = True
    elif not doc['controllers']:
        doc['errors'].append('Unfinished pilot has no live completion controller.')
    doc['verification_passed'] = not doc['errors']
    return doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    MONITOR.mkdir(exist_ok=True)
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve())]
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        with (MONITOR / f'monitor_{stamp}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        freeze(MONITOR / f'launch_{stamp}.json', {**empty_authorization(), 'pid': child.pid,
            'command': command, 'started_unix': time.time(), 'interval_seconds': 3600,
            'scope': 'Read-only checks; writes monitoring reports only. No automatic retries or chat notifications.'})
        print(f'Hourly verifier PID {child.pid}; reports: {MONITOR}', flush=True)
        return
    with (MONITOR / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            started = time.time()
            try:
                doc = snapshot()
            except Exception:
                doc = {**empty_authorization(), 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                       'verification_passed': False, 'pipeline_finished': False, 'errors': [traceback.format_exc()]}
            doc['check_elapsed_seconds'] = time.time() - started
            doc['next_check_unix'] = None if doc['pipeline_finished'] else started + 3600
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            temporary = MONITOR / f'{stamp}.tmp'
            temporary.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
            os.replace(temporary, MONITOR / f'{stamp}.json')
            print(json.dumps(doc), flush=True)
            if args.once or doc['pipeline_finished']:
                return
            while time.time() < started + 3600:
                time.sleep(min(60, started + 3600 - time.time()))


if __name__ == '__main__':
    main()
