"""Operational continuation: settled resource backoff and durable stage adoption."""
from __future__ import annotations
import collections
import inspect
from pact_wrist288_common import *
import pact_wrist280_amendment as amendment
import pact_wrist288_run as original

RECOVERY = WORK / 'amendments/resource_recovery_20260907'
BasePool = original.Pool


class SettledPool(BasePool):
    def __init__(self, kind, limit=14):
        rollout = kind.startswith(('smoke_', 'development_', 'final_'))
        super().__init__(kind, 10 if rollout else limit)
        if rollout:
            append(WORK / 'monitoring/concurrency_changes.jsonl', {
                'utc': now(), 'stage': kind, 'before': limit, 'after': 10,
                'reason': 'resume at authorized fallback after measured 14-worker memory pressure',
                'paused': False, 'active_workers': 0})

    def reduce(self, reason):
        if reason == 'three consecutive resource pressure samples' and len(self.active) > self.limit:
            append(WORK / 'monitoring/resource_backoff_deferrals.jsonl', {
                'utc': now(), 'stage': self.kind, 'target': self.limit, 'active_workers': len(self.active),
                'reason': 'existing jobs have not drained to the previous reduced target; defer further pressure backoff'})
            return
        return super().reduce(reason)


def verify_recovery():
    effective = amendment.check_effective()
    contract = read(RECOVERY / 'contract.json')
    assert contract['sha256'] == digest({k: v for k, v in contract.items() if k != 'sha256'})
    assert contract['effective_config_sha256'] == effective['config_sha256']
    assert sha(WORK / 'training_benchmark.json') == contract['training_benchmark_sha256']
    for path, expected in contract['file_hashes'].items():
        assert sha(ROOT / path) == expected, path
    for path, expected in contract['checkpoint_hashes'].items():
        assert sha(WORK / path) == expected, path
    return contract


def install():
    verify_recovery()
    data, train, monitor, analysis, run = amendment.install()
    run.Pool = SettledPool
    prior_pilot = run.train_pilot

    def resume_pilot():
        if (WORK / 'training_benchmark.json').exists():
            # Its measured timings and peaks are immutable. Empty adopted pools
            # cannot recreate the original benchmark's measured resource peak.
            run.finish_seed(3103)
        else:
            prior_pilot()

    run.train_pilot = resume_pilot
    prior_eval = run.run_eval

    def resume_eval(role, seed, history):
        name = f'{role}_h{history}'
        complete_path = WORK / f'evaluation/{name}_complete.json'
        if not complete_path.exists():
            return prior_eval(role, seed, history)
        jobs = run.eval_jobs(role, seed, history)
        outcomes = analysis.validate_completions(lines(WORK / f'evaluation/{name}_ledger.jsonl'), jobs)
        complete = read(complete_path)
        assert outcomes == complete['rows'] and len(outcomes) == complete['count']
        assert sha(WORK / f'evaluation/{name}_ledger.jsonl') == complete['ledger_sha256']
        pairs = collections.defaultdict(dict)
        for row in outcomes:
            pairs[row['episode_id']][row['arm']] = row
        for identity, pair in pairs.items():
            prior_path = WORK / f'evaluation/pair_audits/{name}_{identity}.json'
            recheck = RECOVERY / f'pair_rechecks/{name}_{identity}.json'
            analysis.check_pair(pair['ACT'], pair['PACT'], recheck)
            assert read(recheck) == read(prior_path)
        append(RECOVERY / 'adopted_evaluations.jsonl', {'utc': now(), 'stage': name, 'count': len(outcomes),
               'completion_sha256': sha(complete_path), 'rerun_rollouts': 0})
        return outcomes

    run.run_eval = resume_eval
    amendment.replace_function(run, 'remaining_eta', [('/14', '/10', 2)])
    prior_report = analysis.report

    def report(verdict, reason, rows=None):
        prior_report(verdict, reason, rows)
        path = WORK / 'EVAL.md'
        path.write_text(path.read_text() + '\nOperational recovery: evaluation resumes at 10 workers after the 14-worker pool exceeded resource limits. Further resource backoff waits until existing jobs reach the reduced target. Trained weights, data/splits, inference parameters, scientific instances and completed outcomes remain unchanged. The measured training benchmark is adopted rather than recreated from empty pools. See `amendments/resource_recovery_20260907/contract.json`.\n')

    analysis.report = report
    return data, train, monitor, analysis, run


if __name__ == '__main__':
    os.environ.update(environment())
    install()[-1].main()
