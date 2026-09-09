"""User-directed 50-pair follow-up after the development gate was missed."""
from __future__ import annotations

import argparse
import collections
import fcntl
import time
import traceback

from pact_wrist288_common import *
import pact_wrist280_eval_capacity as capacity

AMEND = WORK / 'amendments/final50_20260907'
COUNTS = {3103: 17, 3104: 17, 3105: 16}


def select_subset(manifests):
    """Two pairs per cell globally, plus two balanced extras; no outcomes used."""
    by_seed = {}
    for seed in SEEDS:
        by_seed[seed] = collections.defaultdict(list)
        for row in manifests[seed]['rows']:
            by_seed[seed][row['cell']].append(row)
    cell_names = sorted(by_seed[3103], key=lambda c: digest([NAMESPACE, 'final50_cell_order', c]))
    assert len(cell_names) == 24 and all(set(by_seed[s]) == set(cell_names) for s in SEEDS)
    assigned = {seed: [] for seed in SEEDS}
    for round_index in range(2):
        for index, cell in enumerate(cell_names):
            assigned[SEEDS[(index + round_index) % 3]].append(cell)
    # Each seed has 16 cells. Add one previously unassigned cell to each of
    # 3103 and 3104, on opposite sides and different pendant poses.
    candidates = []
    for first in cell_names:
        if first in assigned[3103]:
            continue
        for second in cell_names:
            if second in assigned[3104] or first == second:
                continue
            a, b = by_seed[3103][first][0], by_seed[3104][second][0]
            if a['intrusion_side'] != b['intrusion_side'] and a['pose_id'] != b['pose_id']:
                candidates.append((first, second))
    assert candidates
    extras = min(candidates, key=lambda pair: digest([NAMESPACE, 'final50_extras', *pair]))
    for seed, cell in zip((3103, 3104), extras):
        assigned[seed].append(cell)
    selected = {}
    for seed in SEEDS:
        chosen = [min(by_seed[seed][cell], key=lambda r: digest([NAMESPACE, 'final50_instance', r['row_sha256']]))
                  for cell in assigned[seed]]
        selected[seed] = sorted(chosen, key=lambda r: digest([NAMESPACE, 'final50_order', r['episode_id']]))
        assert len(selected[seed]) == COUNTS[seed]
    all_rows = [row for seed in SEEDS for row in selected[seed]]
    assert len(all_rows) == len({r['episode_id'] for r in all_rows}) == 50
    counts = collections.Counter(r['cell'] for r in all_rows)
    assert len(counts) == 24 and sorted(counts.values()) == [2] * 22 + [3] * 2
    return selected


def verify_followup():
    contract = read(AMEND / 'contract.json')
    assert contract['sha256'] == digest({k: v for k, v in contract.items() if k != 'sha256'})
    assert sha(capacity.CAPACITY / 'contract.json') == contract['capacity_contract_file_sha256']
    assert sha(WORK / 'development_gate.json') == contract['development_gate_sha256']
    assert sha(AMEND / 'selection.json') == contract['selection_sha256']
    assert sha(AMEND / 'request.json') == contract['request_sha256']
    for path, expected in contract['file_hashes'].items():
        assert sha(ROOT / path) == expected, path
    selection = read(AMEND / 'selection.json')
    manifests = {s: read(WORK / f'manifests/final_{s}.json') for s in SEEDS}
    assert selection['rows_by_seed'] == {str(s): rows for s, rows in select_subset(manifests).items()}
    gate = read(WORK / 'development_gate.json')
    assert gate['history'] == 100 and gate['passed'] is False
    return contract


def remaining_eta():
    bench = read(WORK / 'training_benchmark.json')
    updates = sum(read(p)['updates'] for p in (WORK / 'checkpoints').glob('*/progress.json'))
    valid = [r for p in (WORK / 'evaluation').glob('*_ledger.jsonl') for r in lines(p)
             if r.get('valid_completion')]
    timed = [r for r in valid if r['averaging_history'] == 100]
    mean_seconds = sum(r['elapsed_s'] for r in timed) / len(timed)
    final_done = sum(r['rollout_id'].startswith('final_') for r in valid)
    workers = read(capacity.STATUS)['limit'] if capacity.STATUS.exists() else 12
    assert workers in (10, 12) and final_done <= 100
    estimate = (max(0, 360000 - updates) / bench['chosen_updates_per_second']
                + (100 - final_done) * mean_seconds / workers)
    result = {'utc': now(), 'remaining_estimate_hours': estimate / 3600,
              'available_hours': (SAFE_STOP - time.time()) / 3600,
              'optimizer_rate': bench['chosen_updates_per_second'], 'rollout_rate_measured': True,
              'evaluation_workers': workers, 'final_pair_budget': 50,
              'final_rollouts_remaining': 100 - final_done,
              'measured_rollout_mean_minutes': mean_seconds / 60}
    append(WORK / 'monitoring/eta.jsonl', result)
    if estimate > SAFE_STOP - time.time():
        raise capacity.recovery.original.Paused(f'Measured remaining work no longer fits: {result}')
    return result


def install():
    verify_followup()
    data, train, monitor, analysis, run = capacity.install()
    prior_jobs = run.eval_jobs
    selected = read(AMEND / 'selection.json')['rows_by_seed']

    def jobs(role, seed, history):
        all_jobs = prior_jobs(role, seed, history)
        if not role.startswith('final_'):
            return all_jobs
        assert role == f'final_{seed}' and history == 100
        lookup = {j['schedule']['episode_id']: [] for j in all_jobs}
        for job in all_jobs:
            lookup[job['schedule']['episode_id']].append(job)
        result = [job for row in selected[str(seed)] for job in lookup[row['episode_id']]]
        assert len(result) == 2 * COUNTS[seed]
        return result

    run.eval_jobs = jobs
    run.remaining_eta = remaining_eta
    prior_report = analysis.report

    def report(verdict, reason, rows=None):
        prior_report(verdict, reason, rows)
        with (WORK / 'EVAL.md').open('a') as stream:
            stream.write('\nUser-directed follow-up: the original development gate remains missed '
                         '(PACT 10/24, ACT 8/24). The user explicitly requested proceeding with '
                         '50 final pairs total across all three seeds, allocated 17/17/16. '
                         'History 100 remains selected. The subset was frozen without final outcomes; '
                         'all 24 cells have two pairs globally, with two balanced extra pairs. '
                         'This reduced follow-up does not assess the original 150-pair target at '
                         'its specified sample size. See `amendments/final50_20260907/contract.json`.\n')

    analysis.report = report
    return data, train, monitor, analysis, run


def supervise(installed):
    data, train, monitor, analysis, run = installed
    lock = (WORK / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (WORK / 'SAFE_SHUTDOWN.json').exists(), 'archive previous closure before continuation'
    verdict, reason, outcomes = 'INCOMPLETE', '', []
    try:
        run.stage('user_authorized_final50_continuation')
        # Adopt the completed pilot; the immutable development result remains
        # recorded as a miss rather than being reclassified as a pass.
        run.finish_seed(3103)
        remaining_eta()
        for seed in (3104, 3105):
            run.finish_seed(seed)
            remaining_eta()
        for seed in SEEDS:
            outcomes += run.run_eval(f'final_{seed}', seed, 100)
        assert len(outcomes) == 100
        assert len({(r['seed'], r['episode_id'], r['arm']) for r in outcomes}) == 100
        for seed in SEEDS:
            assert collections.Counter(r['arm'] for r in outcomes if r['seed'] == seed) == {
                'ACT': COUNTS[seed], 'PACT': COUNTS[seed]}
        successes = {arm: sum(r['task_success'] for r in outcomes if r['arm'] == arm) for arm in ('ACT', 'PACT')}
        verdict = 'REDUCED FINAL EVALUATION COMPLETE'
        reason = f'Completed all 50 valid pairs / 100 rollouts across three seeds. Pooled successes: {successes}. The original development gate was overridden by explicit user instruction; the original 150-pair target was not assessed.'
    except BaseException as error:
        reason = f'{type(error).__name__}: {error}'
        atomic(WORK / 'stop_error.json', {'utc': now(), 'error': reason, 'traceback': traceback.format_exc()})
        print(traceback.format_exc(), flush=True)
    finally:
        for pool in run.POOLS:
            pool.pause_reason = pool.pause_reason or 'supervisor shutdown'
            for job, receipt in pool.drain():
                append(WORK / 'monitoring/shutdown_drain.jsonl', {'job': job, 'receipt': receipt})
        analysis.report(verdict, reason, outcomes)
        cleaned = run.cleanup_orphans()
        run.hourly_run_check(final=True)
        atomic(WORK / 'SAFE_SHUTDOWN.json', {**empty_authorization(), 'utc': now(),
               'verdict': verdict, 'reason': reason, 'cleaned_owned_orphans': cleaned})
        print(f'{verdict}: {reason}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('supervisor', 'monitor'), required=True)
    args = parser.parse_args()
    os.environ.update(environment())
    installed = install()
    if args.mode == 'monitor':
        installed[2].main()
    else:
        supervise(installed)
