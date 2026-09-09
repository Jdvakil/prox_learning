"""Drain a frozen four-worker scheduler and retain actual Linux child exit codes."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pact_place_v1011c_experiment import *
from pact_place_v1011c_metrics import metrics


def process(pid):
    proc = Path('/proc') / str(pid)
    fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
            'start_ticks': int(fields[19]), 'wait_status': int(fields[49]),
            'command': (proc/'cmdline').read_bytes().split(b'\0')[:-1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner-pid', type=int, required=True)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    directory = EVAL/'scheduler_resize_01'
    directory.mkdir(exist_ok=True)
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve()), '--runner-pid',
                   str(args.runner_pid), '--workers', str(args.workers)]
        with (directory/'drain.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment(),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        freeze(directory/'launch.json', {**empty_authorization(), 'pid': child.pid,
               'command': command, 'started_unix_s': time.time()})
        print(f'Detached drain PID {child.pid}', flush=True)
        return
    assert 1 <= args.workers <= 12
    parent = process(args.runner_pid)
    assert any(b'run_pact_place_v1011c_eval.py' in c for c in parent['command'])
    assert Path(os.readlink(f'/proc/{args.runner_pid}/cwd')).resolve() == ROOT
    original_coordinator = process(parent['ppid'])
    assert any(b'finish_pact_place_v1011c_pipeline.py' in c for c in original_coordinator['command'])
    assert not (EVAL/'full_run.json').exists()
    jobs = read(EVAL/'full_schedule.json')['jobs']
    by_output = {j['output']: j for j in jobs}
    terminated = False
    os.kill(args.runner_pid, signal.SIGSTOP)
    try:
        while process(args.runner_pid)['state'] not in ('T', 't'):
            time.sleep(0.05)
        children = []
        for path in Path('/proc').glob('[0-9]*'):
            try:
                info = process(int(path.name))
            except (OSError, ValueError, IndexError):
                continue
            if info['ppid'] != args.runner_pid:
                continue
            command = [s.decode() for s in info.pop('command')]
            assert any('eval_pact_place_v1011c_row.py' in c for c in command)
            output = command[command.index('--output-dir')+1]
            job = by_output[output]
            assert command == job['command']
            info.update(job=job, command=command)
            children.append(info)
        assert children, 'No active children to drain'
        freeze(directory/'pause.json', {**empty_authorization(),
            'runner_pid': args.runner_pid, 'runner_start_ticks': parent['start_ticks'],
            'original_coordinator_pid': parent['ppid'], 'children': children,
            'paused_unix_s': time.time(), 'target_workers': args.workers,
            'schedule_sha256': sha256_file(EVAL/'full_schedule.json')})
        print(f'Scheduler paused; {len(children)} active rollouts continue unchanged.', flush=True)
        pending = {c['pid']: c for c in children}
        deadline = time.time()+3600
        while pending:
            assert time.time() < deadline, 'Drain timeout; old scheduler will be continued'
            for pid, child in list(pending.items()):
                info = process(pid)
                assert info['start_ticks'] == child['start_ticks']
                if info['state'] != 'Z':
                    continue
                code = os.waitstatus_to_exitcode(info['wait_status'])
                assert code == 0, f'Worker {pid} actually exited {code}; not adopting failure'
                job = child['job']
                row = {**job['schedule'], 'directory': str(Path(job['output']).relative_to(ROOT)),
                       'status': 'complete', 'returncode': code, 'metrics': metrics(Path(job['output']), job['schedule']),
                       'exit_code_evidence': {'source': '/proc/PID/stat field 52 while zombie',
                           'pid': pid, 'start_ticks': info['start_ticks'],
                           'linux_wait_status': info['wait_status'], 'observed_unix_s': time.time()},
                       'adopted_after_scheduler_drain': True}
                uptime = float(Path('/proc/uptime').read_text().split()[0])
                row['elapsed_s'] = uptime-child['start_ticks']/os.sysconf('SC_CLK_TCK')
                row['elapsed_s_note'] = 'Upper bound observed at drain polling (within 10 s, plus validation); not exact completion time.'
                freeze(directory/(job['schedule']['rollout_id']+'.json'), row)
                print(f'Raw verified {row["rollout_id"]}, actual exit {code}', flush=True)
                del pending[pid]
            if pending:
                time.sleep(10)
        freeze(directory/'drained.json', {**empty_authorization(),
            'rollouts_drained': len(children), 'all_actual_returncodes_zero': True,
            'completed_unix_s': time.time(), 'runner_pid': args.runner_pid,
            'reason': 'Owner requested higher resource concurrency; no in-flight worker interrupted.'})
        assert process(args.runner_pid)['start_ticks'] == parent['start_ticks']
        os.kill(args.runner_pid, signal.SIGKILL)
        terminated = True
        print('Drained scheduler terminated with SIGKILL; original coordinator records actual -9.', flush=True)
    finally:
        if not terminated:
            os.kill(args.runner_pid, signal.SIGCONT)
    # Allow original stage receipt and coordinator lock release before resuming.
    deadline = time.time()+120
    while True:
        assert time.time() < deadline, 'Original coordinator did not finish'
        try:
            info = process(parent['ppid'])
        except FileNotFoundError:
            break
        if info['state'] == 'Z':
            break
        time.sleep(1)
    assert read(WORK/'stages/11_eval_full.json')['returncode'] == -9
    command = [sys.executable, str(ROOT/'scripts/finish_pact_place_v1011c_pipeline.py'),
               '--training-pid', '3041262', '--adopt-training', '--resume-full',
               '--workers', str(args.workers), '--detach']
    result = subprocess.run(command, cwd=ROOT, env=environment())
    assert result.returncode == 0


if __name__ == '__main__':
    main()
