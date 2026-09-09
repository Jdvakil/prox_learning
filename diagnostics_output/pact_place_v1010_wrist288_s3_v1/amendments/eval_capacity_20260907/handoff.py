"""Drain the old supervisor, archive its closure, and start the bound continuation."""
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/prox_learning_pact_remediation')
sys.path.insert(0, str(ROOT / 'scripts'))
from pact_wrist288_common import *
from pact_wrist288_capacity import process_identity
import pact_wrist280_eval_capacity as capacity

directory = capacity.CAPACITY
old_supervisor, old_monitor = 1059731, 1061731
assert read(WORK / 'state.json')['supervisor_pid'] == old_supervisor
assert read(WORK / 'PAUSE.json') == read(directory / 'request.json')
capacity.verify_capacity()
print(f'{now()} waiting for healthy current rollouts and old monitor to drain', flush=True)
deadline = time.monotonic() + 2400
while not (WORK / 'SAFE_SHUTDOWN.json').exists() or process_identity(old_supervisor) or process_identity(old_monitor):
    assert time.monotonic() < deadline, 'drain exceeded 40 minutes; inspect without replacing running workers'
    time.sleep(2)

assert time.time() < SAFE_STOP
assert read(WORK / 'PAUSE.json') == read(directory / 'request.json')
closure = read(WORK / 'SAFE_SHUTDOWN.json')
assert closure['verdict'] == 'INCOMPLETE' and 'disk/deadline/manual guard' in closure['reason'], closure
audit = lines(WORK / 'monitoring/hourly_run_check.jsonl')[-1]
assert not audit['verification_issues'] and not audit['new_worker_errors'], 'inspect failed audit'
for path in WORK.rglob('launch.json'):
    if 'precollection_revision_00' in path.parts:
        continue
    assert (path.parent / 'exit_receipt.json').exists(), f'unobserved worker exit: {path}'
for path in (WORK / 'evaluation').glob('*_ledger.jsonl'):
    rows = lines(path)
    assert all(r['valid_completion'] and r['returncode'] == 0 for r in rows)
    assert len(rows) == len({r['rollout_id'] for r in rows})

lock = (WORK / 'supervisor.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
archive = directory / 'prior_closure'
archive.mkdir(exist_ok=False)
for name in ('SAFE_SHUTDOWN.json', 'PAUSE.json', 'stop_error.json', 'state.json', 'EVAL.md', 'supervisor.log'):
    path = WORK / name
    if path.exists():
        shutil.move(path, archive / name)
monitor_log = WORK / 'monitoring/resource_recovery_monitor.log'
if monitor_log.exists():
    shutil.copy2(monitor_log, archive / monitor_log.name)
preserved = {p.name: {'count': len(lines(p)), 'sha256': sha(p)}
             for p in (WORK / 'evaluation').glob('*_ledger.jsonl')}
freeze(directory / 'drain_receipt.json', {'utc': now(), 'prior_supervisor_pid': old_supervisor,
       'prior_monitor_pid': old_monitor, 'safe_shutdown_sha256': sha(archive / 'SAFE_SHUTDOWN.json'),
       'all_worker_exits_observed': True, 'preserved_evaluation_ledgers': preserved,
       'handoff_script_sha256': sha(__file__)})
(WORK / 'EVAL.md').write_text(
    '# V10.10 wrist280 three-seed experiment (amended cohort)\n\nVerdict: **RUNNING**.\n\n'
    'Resuming evaluation at the user-requested 12-worker target. Completed artifact file cache '
    'is released within this experiment to retain the original memory guards.\n\n'
    'Collection/conversion: 280 episodes. ACT and PACT seed 3103: 60,000 updates each. '
    'Completed rollouts and pairing audits are preserved. The development gate is pending.\n\n'
    'See `amendments/eval_capacity_20260907/contract.json` and `drain_receipt.json`. '
    'Prior report is archived in that amendment’s `prior_closure/EVAL.md`. '
    'All `authorizes_*` flags remain false.\n')
with (WORK / 'RUN.md').open('a') as stream:
    stream.write(f'\n{now()}: user-requested 12-worker evaluation continuation, with scoped closed-artifact '
                 'file-cache release and original resource thresholds. '
                 'See `amendments/eval_capacity_20260907/contract.json`.\n')
fcntl.flock(lock, fcntl.LOCK_UN)
lock.close()
command = [sys.executable, str(ROOT / 'scripts/pact_wrist280_eval_capacity.py'), '--mode']
with (WORK / 'supervisor.log').open('x') as stream:
    supervisor = subprocess.Popen(command + ['supervisor'], cwd=ROOT, env=environment(),
                                  stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
with (directory / 'monitor.log').open('x') as stream:
    monitor = subprocess.Popen(command + ['monitor'], cwd=ROOT, env=environment(),
                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {'utc': now(), 'supervisor_pid': supervisor.pid, 'monitor_pid': monitor.pid,
           'capacity_contract_sha256': read(directory / 'contract.json')['sha256']}
freeze(directory / 'continuation_launch.json', receipt)
time.sleep(5)
assert supervisor.poll() is None and monitor.poll() is None, 'new process exited; inspect logs'
print(json.dumps(receipt), flush=True)
