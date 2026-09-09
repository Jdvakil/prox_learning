"""Capture process/thread counts if shared-host pressure returns; never signal it."""
import datetime
import json
import pathlib
import time

C = pathlib.Path(__file__).resolve().parents[1]
cg = pathlib.Path('/sys/fs/cgroup')
last_capture = 0
while True:
    status = json.loads((C / 'run_status.json').read_text())
    if not status['active_parent']:
        print(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          'stopped_with_parent_status': status['status']}), flush=True)
        break
    used = int((cg / 'pids.current').read_text())
    limit = int((cg / 'pids.max').read_text())
    if used / limit > .3 and time.monotonic() - last_capture >= 30:
        processes = []
        for proc in pathlib.Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                fields = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines() if ':' in line)
                processes.append({'pid': int(proc.name), 'ppid': int(fields['PPid']),
                    'name': fields['Name'].strip(), 'threads': int(fields['Threads'])})
            except (OSError, KeyError):
                continue
        record = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'pids_current': used, 'pids_max': limit, 'parent_pid': status['parent_pid'],
            'processes': sorted(processes, key=lambda item: item['threads'], reverse=True)}
        with (C / 'root_review/thread_pressure.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps({'utc': record['utc'], 'pids_current': used, 'pids_max': limit}), flush=True)
        last_capture = time.monotonic()
    time.sleep(10)
