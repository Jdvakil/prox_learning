"""User-authorized collection capacity amendment; original frozen code stays intact.

The first supervisor has no live resize interface. At the approved boundary this
controller requests a drain, verifies real exits, archives that closure, and
continues the same ledger under a collection-only adaptive Pool subclass.
"""
from __future__ import annotations
import argparse
import fcntl
import shutil
import subprocess
import time
import traceback
from pact_wrist288_common import *
from pact_wrist288_monitor import resources, owned_processes
import pact_wrist288_run as original

AMENDMENT = WORK / 'monitoring/capacity_amendment.json'
HOLD_UNTIL = datetime(2026, 9, 6, 11, tzinfo=timezone.utc).timestamp()
LADDER = (10, 12, 14, 16, 18)


def verify_amendment():
    value = read(AMENDMENT)
    assert value['amendment_sha256'] == digest({k: v for k, v in value.items() if k != 'amendment_sha256'})
    assert value['config_sha256'] == read(WORK / 'config.json')['config_sha256']
    assert value['hold_until_epoch'] == HOLD_UNTIL
    assert value['scope'] == 'collection concurrency only'
    for path, expected in value['file_hashes'].items():
        assert sha(ROOT / path) == expected, f'capacity amendment code drift: {path}'
    return value


def status(phase, **details):
    value = {'utc': now(), 'controller_pid': os.getpid(), 'phase': phase, **details}
    atomic(WORK / 'monitoring/capacity_status.json', value)
    return value


def process_identity(pid):
    """Include kernel start time; a recycled PID is never the same process."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def snapshot(target, active_pids=None):
    value = resources()
    processes = owned_processes()
    if active_pids is None:
        active_pids = [p['pid'] for p in processes if 'pact_wrist288_collect.py --job ' in p['command']]
    children = {}
    for process in processes:
        children.setdefault(process['ppid'], []).append(process['pid'])

    def tree_pss(pid):
        total = 0
        try:
            for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
                if line.startswith('Pss:'):
                    total += int(line.split()[1]) * 1024
        except OSError:
            pass
        return total + sum(tree_pss(child) for child in children.get(pid, []))

    value.update(epoch=time.time(), target=target, active=len(active_pids),
                 worker_tree_pss_max_bytes=max([tree_pss(pid) for pid in active_pids] or [0]))
    return value


def capacity_decision(samples, current, at, last_change=0):
    """Use sustained full-pool measurements, with a 1% RAM safety margin."""
    out = {'current': current, 'proposed': current, 'reason': 'collecting full-pool measurements'}
    if at < HOLD_UNTIL:
        return {**out, 'reason': 'hold 14 through 11:00 UTC'}
    if current >= 18:
        return {**out, 'reason': 'maximum authorized collection concurrency'}
    if at - last_change < 15 * 60:
        return {**out, 'reason': '15-minute observation period after concurrency change'}
    full = [s for s in samples if at - 15 * 60 <= s['epoch'] <= at
            and s['target'] == current and s['active'] == current]
    if len(full) < 10 or max(s['epoch'] for s in full) - min(s['epoch'] for s in full) < 9 * 60:
        return out
    candidate = LADDER[LADDER.index(current) + 1]
    peak = {key: max(s[key] for s in full) for key in ('ram_fraction', 'vram_fraction', 'pid_fraction')}
    memory_max = min(int(s['cgroup']['memory.max']) for s in full)
    # The largest worker plus its renderer/children, padded for transient allocations.
    extra_worker = max(5 * 2**30, 1.10 * max(s['worker_tree_pss_max_bytes'] for s in full))
    predicted = {'ram_fraction': peak['ram_fraction'] + (candidate - current) * extra_worker / memory_max,
                 'vram_fraction': peak['vram_fraction'] * candidate / current,
                 'pid_fraction': peak['pid_fraction'] * candidate / current}
    fits = (predicted['ram_fraction'] <= .79 and predicted['vram_fraction'] <= .83
            and predicted['pid_fraction'] <= .73 and min(s['disk_free_gib'] for s in full) >= 12)
    return {**out, 'proposed': candidate if fits else current,
            'reason': 'measured headroom permits two more workers' if fits else 'retain current pool: projected resource margin insufficient',
            'full_pool_samples': len(full), 'observed_peak': peak, 'predicted': predicted,
            'extra_worker_bytes': extra_worker, 'candidate': candidate}


class CollectionPool(original.Pool):
    def __init__(self, kind, limit=14):
        super().__init__(kind, limit)
        self.capacity_samples = []
        self.last_capacity_change = time.time()
        if kind == 'collection':
            handoff = read(WORK / 'monitoring/capacity_handoff.json')
            assert handoff['amendment_sha256'] == verify_amendment()['amendment_sha256']
            self.limit = handoff['resume_target']
            assert self.limit in (14, 16)
            self.change_log(14, 'authorized continuation after verified drain')

    def change_log(self, before, reason):
        event = {'utc': now(), 'stage': self.kind, 'before': before, 'after': self.limit,
                 'reason': reason, 'paused': bool(self.pause_reason)}
        append(WORK / 'monitoring/concurrency_changes.jsonl', event)
        atomic(WORK / 'monitoring/collection_capacity.json', event)

    def reduce(self, reason):
        if self.kind != 'collection':
            return super().reduce(reason)
        before = self.limit
        if self.limit > 10:
            self.limit = LADDER[LADDER.index(self.limit) - 1]
        else:
            self.pause_reason = reason
        self.last_capacity_change = time.time()
        self.change_log(before, reason)

    def sample(self, force=False):
        if self.kind != 'collection':
            return super().sample(force)
        if force or time.monotonic() >= self.next_sample:
            value = snapshot(self.limit, list(self.active))
            append(WORK / 'monitoring/capacity_samples.jsonl', value)
            append(WORK / 'monitoring/supervisor_resources.jsonl', {'stage': self.kind, **value})
            self.capacity_samples = [s for s in self.capacity_samples if value['epoch'] - s['epoch'] <= 900] + [value]
            for key in self.peak:
                self.peak[key] = max(self.peak[key], value[key])
            high = value['vram_fraction'] > .85 or value['ram_fraction'] > .80 or value['pid_fraction'] > .75
            self.pressure = self.pressure + 1 if high else 0
            if self.pressure >= 3:
                self.reduce('three consecutive resource pressure samples')
                self.pressure = 0
            if value['disk_free_gib'] < 10:
                self.pause_reason = 'disk reserve below 10 GiB'
            if not high and not self.pause_reason and not (WORK / 'PAUSE.json').exists() and time.time() < SAFE_STOP:
                decision = capacity_decision(self.capacity_samples, self.limit, value['epoch'], self.last_capacity_change)
                append(WORK / 'monitoring/capacity_decisions.jsonl', {'utc': now(), **decision})
                if decision['proposed'] > self.limit:
                    before = self.limit
                    self.limit = decision['proposed']
                    self.last_capacity_change = time.time()
                    self.change_log(before, decision)
            self.next_sample = time.monotonic() + 60
        if (WORK / 'PAUSE.json').exists() or time.time() >= SAFE_STOP:
            self.pause_reason = 'disk/deadline/manual guard'


def amended_supervisor():
    verify_amendment()
    binding_check = original.check_bindings

    def checked_bindings():
        verify_amendment()
        return binding_check()

    original_stage = original.stage

    def resumed_stage(name):
        original_stage(name)
        if name == 'collection':
            state = read(WORK / 'state.json')
            state['resumed_utc'] = state['started_utc']
            state['started_utc'] = read(WORK / 'monitoring/capacity_handoff.json')['collection_started_utc']
            atomic(WORK / 'state.json', state)

    original.check_bindings = checked_bindings
    original.stage = resumed_stage
    original.Pool = CollectionPool
    original.main()


def verify_drained():
    closure = read(WORK / 'SAFE_SHUTDOWN.json')
    assert closure['verdict'] == 'INCOMPLETE'
    assert closure['reason'] == 'Paused: disk/deadline/manual guard', closure
    pause = read(WORK / 'PAUSE.json')
    assert pause['reason'] == 'user_authorized_collection_capacity_handoff'
    assert pause['amendment_sha256'] == verify_amendment()['amendment_sha256']
    assert time.time() < SAFE_STOP and resources()['disk_free_gib'] >= 12
    records = lines(WORK / 'collection/ledger.jsonl')
    assert len(records) == len({r['attempt_id'] for r in records})
    by_id = {r['attempt_id']: r for r in records}
    for path in (WORK / 'raw').glob('*/job.json'):
        job = read(path)
        receipt = read(path.parent / 'exit_receipt.json')
        assert receipt['returncode'] == 0 and receipt['job_sha256'] == digest(job)
        record = by_id[job['row']['attempt_id']]
        assert record['exit_receipt_sha256'] == sha(path.parent / 'exit_receipt.json')
        if record['accepted']:
            assert sha(ROOT / record['trajectory_h5']) == record['trajectory_h5_sha256']
        log = (path.parent / 'worker.log').read_text(errors='replace').lower()
        assert not any(token in log for token in ('out of memory', 'pthread_create failed', "can't start new thread", 'cannot create thread'))
    assert not any('pact_wrist288_collect.py --job ' in p['command'] for p in owned_processes())
    return {'verified_attempts': len(records), 'accepted': sum(r['accepted'] for r in records)}


def perform_handoff(decision, amendment):
    verify_amendment()
    check_bindings()
    assert read(WORK / 'state.json')['stage'] == 'collection'
    assert not (WORK / 'collection_complete.json').exists()
    supervisor = read(WORK / 'state.json')['supervisor_pid']
    collection_started = read(WORK / 'state.json')['started_utc']
    identity = process_identity(supervisor)
    assert identity is not None
    request = {'utc': now(), 'reason': 'user_authorized_collection_capacity_handoff',
               'amendment_sha256': amendment['amendment_sha256'], 'decision': decision}
    # Never replace a disk/deadline/error pause written by another monitor.
    with (WORK / 'PAUSE.json').open('x') as stream:
        json.dump(request, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    while process_identity(supervisor) == identity:
        status('draining existing attempts', supervisor_pid=supervisor, decision=decision)
        time.sleep(10)
    verified = verify_drained()
    monitor = read(WORK / 'monitoring/monitor.json')['pid']
    monitor_identity = process_identity(monitor)
    # The original monitor exits naturally after seeing the recorded closure.
    until = time.monotonic() + 180
    while monitor_identity is not None and process_identity(monitor) == monitor_identity:
        assert time.monotonic() < until, 'old monitor did not finish closure audit'
        status('waiting for monitor closure audit', **verified)
        time.sleep(10)
    verify_drained()
    idle = resources()
    predicted_idle_ram = (int(idle['cgroup']['memory.current']) + 16 * decision['extra_worker_bytes']) / int(idle['cgroup']['memory.max'])
    resume_target = 16 if predicted_idle_ram <= .79 and idle['vram_fraction'] < .83 and idle['pid_fraction'] < .73 else 14
    freeze(WORK / 'monitoring/capacity_handoff.json', {
        'utc': now(), 'amendment_sha256': amendment['amendment_sha256'], 'old_supervisor_pid': supervisor,
        'collection_started_utc': collection_started,
        'old_monitor_pid': monitor, 'decision': decision, 'resume_target': resume_target,
        'idle_resources': idle, 'predicted_idle_ram_fraction_at_16': predicted_idle_ram, **verified})
    archive = WORK / 'monitoring/capacity_handoff_archive'
    archive.mkdir(exist_ok=False)
    for name in ('SAFE_SHUTDOWN.json', 'stop_error.json', 'PAUSE.json', 'EVAL.md', 'state.json',
                 'supervisor_launch.json', 'supervisor.log', 'monitoring/monitor.json'):
        source = WORK / name
        if source.exists():
            target = archive / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source, target)
    env = environment()
    with (WORK / 'monitoring/capacity_monitor.log').open('x') as stream:
        monitor_process = subprocess.Popen([sys.executable, str(CODE / 'pact_wrist288_monitor.py')],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    with (WORK / 'supervisor.log').open('x') as stream:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--supervisor'],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    atomic(WORK / 'supervisor_launch.json', {'utc': now(), 'pid': process.pid,
        'config_sha256': amendment['config_sha256'], 'amendment_sha256': amendment['amendment_sha256']})
    atomic(WORK / 'monitoring/capacity_continuation.json', {'utc': now(), 'status': 'RUNNING', 'supervisor_pid': process.pid})
    (WORK / 'EVAL.md').write_text('# Wrist288 experiment\n\nRUNNING: collection resumed after the authorized capacity handoff. Scientific outcomes remain unmeasured. See RUN.md and monitoring/capacity_handoff.json.\n')
    while process.poll() is None:
        status('adaptive collection supervisor running', supervisor_pid=process.pid,
               monitor_pid=monitor_process.pid, resume_target=resume_target,
               current_stage=read(WORK / 'state.json') if (WORK / 'state.json').exists() else None)
        if monitor_process.poll() is not None and not (WORK / 'SAFE_SHUTDOWN.json').exists():
            raise RuntimeError('hourly monitor exited during active continuation')
        time.sleep(30)
    atomic(WORK / 'monitoring/capacity_supervisor_exit.json', {
        'utc': now(), 'pid': process.pid, 'returncode': process.returncode, 'exit_evidence': 'subprocess.Popen.poll / waitpid'})
    assert (WORK / 'SAFE_SHUTDOWN.json').exists(), 'continuation supervisor exited without closure'
    status('experiment closed', closure=read(WORK / 'SAFE_SHUTDOWN.json'))


def controller():
    os.environ.update(environment())
    lock = (WORK / 'monitoring/capacity_controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    amendment = verify_amendment()
    check_bindings()
    samples = []
    try:
        while time.time() < SAFE_STOP:
            if (WORK / 'SAFE_SHUTDOWN.json').exists() or (WORK / 'PAUSE.json').exists():
                status('stopped: existing pause or closure')
                return
            state = read(WORK / 'state.json')
            if state['stage'] != 'collection':
                status('collection finished; no resize needed', state=state)
                return
            assert process_identity(state['supervisor_pid']) is not None, 'collection supervisor disappeared'
            changes = [r for r in lines(WORK / 'monitoring/concurrency_changes.jsonl') if r['stage'] == 'collection']
            current = changes[-1]['after'] if changes else 14
            if current != 14:
                status('original resource fallback active; hourly monitor must diagnose', current=current)
                return
            sample = snapshot(current)
            append(WORK / 'monitoring/capacity_samples.jsonl', sample)
            samples = [s for s in samples if sample['epoch'] - s['epoch'] <= 900] + [sample]
            decision = capacity_decision(samples, current, sample['epoch'])
            append(WORK / 'monitoring/capacity_decisions.jsonl', {'utc': now(), **decision})
            status('holding 14 until 11:00 UTC' if sample['epoch'] < HOLD_UNTIL else 'checking headroom at 14',
                   decision=decision, resources=sample)
            if decision['proposed'] == 16:
                perform_handoff(decision, amendment)
                return
            time.sleep(60)
        status('reporting-hour guard reached')
    except BaseException as error:
        atomic(WORK / 'monitoring/capacity_error.json', {'utc': now(), 'error': str(error), 'traceback': traceback.format_exc()})
        status('capacity controller stopped with error', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervisor', action='store_true')
    args = parser.parse_args()
    amended_supervisor() if args.supervisor else controller()
