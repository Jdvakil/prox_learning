"""Pause the unchanged loader before allocating host processes or queue threads."""
import datetime
import json
import pathlib
import time

C = pathlib.Path(__file__).resolve().parents[2]
CG = pathlib.Path('/sys/fs/cgroup')


def pid_fraction():
    return int((CG / 'pids.current').read_text()) / int((CG / 'pids.max').read_text())


def await_capacity():
    if pid_fraction() < .5:
        return
    started = time.monotonic()
    record = {'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'initial_pid_fraction': pid_fraction()}
    while pid_fraction() >= .4:
        if time.monotonic() - started > 1800:
            raise RuntimeError('Host process capacity did not recover within 30 minutes')
        time.sleep(.25)
    record.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  waited_s=time.monotonic()-started, final_pid_fraction=pid_fraction())
    with (C / 'recoveries/04_capacity_wait/capacity_waits.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')


class CapacityLoader:
    def __init__(self, inner, gate=await_capacity):
        self.inner, self.gate = inner, gate

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __len__(self):
        return len(self.inner)

    def __iter__(self):
        self.gate()
        iterator = iter(self.inner)
        while True:
            self.gate()
            try:
                value = next(iterator)
            except StopIteration:
                return
            yield value
