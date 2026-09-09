"""Expand seed 3104 to its original full 50-pair manifest, preserving prior results."""
from __future__ import annotations
import argparse
import collections
import fcntl
import math
import time
import traceback

from pact_wrist288_common import *
import pact_wrist280_receipt_recovery as recovery

EXT = WORK / 'amendments/seed3104_full50_20260907'
ROLE = 'final_3104_full50'
# User explicitly extended the original deadline by three hours.
EXTENDED_DEADLINE = DEADLINE + 3 * 3600
LAUNCH_STOP = EXTENDED_DEADLINE - 3600
# Capture the unfiltered generator before installing the prior 50-total adapter.
BASE_JOBS = recovery.capacity.recovery.original.eval_jobs


def full_jobs():
    jobs = BASE_JOBS('final_3104', 3104, 100)
    for job in jobs:
        job['command'][:2] = [sys.executable,
                              str(recovery.capacity.recovery.amendment.AMEND_CODE), '--mode', 'eval']
    assert len(jobs) == 100
    assert len({j['schedule']['episode_id'] for j in jobs}) == 50
    assert collections.Counter(j['schedule']['arm'] for j in jobs) == {'ACT': 50, 'PACT': 50}
    assert all(j['schedule']['checkpoint_seed'] == 3104 and j['schedule']['averaging_history'] == 100 for j in jobs)
    return jobs


def verify_extension():
    contract = read(EXT / 'contract.json')
    assert contract['sha256'] == digest({k: v for k, v in contract.items() if k != 'sha256'})
    assert sha(recovery.RECOVERY / 'contract.json') == contract['receipt_recovery_contract_file_sha256']
    for relative, expected in contract['protected_artifacts'].items():
        assert sha(WORK / relative) == expected, relative
    for relative, expected in contract['file_hashes'].items():
        assert sha(ROOT / relative) == expected, relative
    assert contract['launch_stop_utc'] == datetime.fromtimestamp(LAUNCH_STOP, timezone.utc).isoformat()
    assert contract['hard_deadline_utc'] == datetime.fromtimestamp(EXTENDED_DEADLINE, timezone.utc).isoformat()
    return contract


def install():
    verify_extension()
    installed = recovery.install()
    monitor, run = installed[2], installed[-1]
    monitor.SAFE_STOP = run.SAFE_STOP = LAUNCH_STOP
    monitor.DEADLINE = run.DEADLINE = EXTENDED_DEADLINE

    def jobs(role, seed, history):
        assert (role, seed, history) == (ROLE, 3104, 100)
        return full_jobs()

    run.eval_jobs = jobs
    return installed


def verify_prior_jobs(jobs):
    lookup = {j['schedule']['rollout_id']: j for j in jobs}
    prior = read(WORK / 'evaluation/final_3104_h100_schedule.json')
    assert len(prior) == 34
    for job in prior:
        assert lookup[job['schedule']['rollout_id']] == job
        assert read(Path(job['directory']) / 'job.json') == job
        receipt = read(Path(job['directory']) / 'exit_receipt.json')
        assert receipt['returncode'] == 0 and receipt['job_sha256'] == digest(job)
    return prior


def remaining_eta(jobs):
    pending = sum(not (Path(j['directory']) / 'exit_receipt.json').exists() for j in jobs)
    # Recent completed seed-3105 extension, counted once per canonical rollout.
    records = [r for r in lines(WORK / 'evaluation/final_3105_full50_h100_ledger.jsonl')
               if r.get('valid_completion')]
    assert len(records) == len({r['rollout_id'] for r in records}) == 100
    mean = sum(r['elapsed_s'] for r in records) / len(records)
    # Include the partly-filled last batch in the estimate.
    estimate = math.ceil(pending / 12) * mean + 120
    result = {'utc': now(), 'additional_rollouts_pending': pending,
              'evaluation_workers': 12, 'mean_rollout_minutes': mean / 60,
              'remaining_estimate_hours': estimate / 3600,
              'available_hours': (LAUNCH_STOP - time.time()) / 3600}
    append(EXT / 'eta.jsonl', result)
    if estimate > LAUNCH_STOP - time.time():
        raise recovery.capacity.recovery.original.Paused(f'Extension no longer fits before reporting reserve: {result}')
    return result


def write_report(analysis, verdict, reason, rows):
    old_ids = {r['episode_id'] for r in read(WORK / 'evaluation/final_3104_h100_complete.json')['rows']}
    text = ['# Seed 3104 — full 50-pair evaluation', '', f'Verdict: **{verdict}**.', '', reason, '',
            f'Updated: {now()}. ACT and PACT use their existing 60,000-update checkpoints, history 100, and the original frozen final-3104 manifest.', '']
    if rows:
        assert len(rows) == 100 and len({(r['episode_id'], r['arm']) for r in rows}) == 100
        groups = {'prior_17': [r for r in rows if r['episode_id'] in old_ids],
                  'added_33': [r for r in rows if r['episode_id'] not in old_ids], 'full_50': rows}
        summaries = {group: {arm: analysis.aggregate([r for r in rr if r['arm'] == arm])
                             for arm in ('ACT', 'PACT')} for group, rr in groups.items()}
        text += ['| Instances | ACT success | PACT success | PACT−ACT | ACT collision-free success | PACT collision-free success |',
                 '|---|---:|---:|---:|---:|---:|']
        for group, label in [('prior_17', 'Previously completed 17'), ('added_33', 'Additional 33'), ('full_50', 'Full 50')]:
            a, p = [summaries[group][arm] for arm in ('ACT', 'PACT')]
            n = a['n'];assert n == p['n']
            text.append(f'| {label} | {a["task_success"]}/{n} ({100*a["task_success"]/n:.1f}%) | {p["task_success"]}/{n} ({100*p["task_success"]/n:.1f}%) | {100*(p["task_success"]-a["task_success"])/n:+.1f} pp | {a["collision_free_task_success"]}/{n} | {p["collision_free_task_success"]}/{n} |')
        full = summaries['full_50'];a, p = full['ACT'], full['PACT']
        benchmark = {'PACT_at_least_26_of_50': p['task_success'] >= 26,
                     'PACT_at_least_five_ahead': p['task_success'] - a['task_success'] >= 5}
        text += ['', f'Descriptive 50-pair goals: PACT ≥26 successes: **{benchmark["PACT_at_least_26_of_50"]}**; PACT at least five successes ahead of ACT: **{benchmark["PACT_at_least_five_ahead"]}**.', '',
                 '| Arm | Collision-free rollouts | Hazard-contact rollouts | Hazard frames | Mean hazard frames | Touch | Hold | Lift ≥1 cm | Receptacle support |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm, r in full.items():
            text.append(f'| {arm} | {r["collision_free"]}/50 | {r["hazard_contact_episode"]}/50 | {r["hazard_frames"]} | {r["mean_hazard_frames"]:.2f} | {r["target_touch"]} | {r["target_held"]} | {r["target_lifted_1cm"]} | {r["placement_support"]} |')
        text += ['', '| Arm | Slot | Contact rollouts | Contact frames | Stability-event rollouts | Stable rollouts |',
                 '|---|---|---:|---:|---:|---:|']
        for arm, r in full.items():
            for slot, counts in r['per_object'].items():
                text.append(f'| {arm} | {slot} | {counts["contact_episode"]}/50 | {counts["contact_frames"]} | {counts["stability_event"]}/50 | {counts["stable"]}/50 |')
        paired = analysis.paired_counts(rows)
        text += ['', f'Paired successes: ACT-only {paired["ACT_only"]}; PACT-only {paired["PACT_only"]}.',
                 '', 'Failure stages: ' + json.dumps({arm: r['failure_stages'] for arm, r in full.items()}, sort_keys=True) + '.']
        atomic(EXT / 'analysis.json', {**empty_authorization(), 'verdict': verdict, 'seed': 3104,
               'groups': summaries, 'paired': paired, 'benchmark': benchmark, 'rows': rows})
    text += ['', 'The user selected seed 3104 for expansion after reviewing its initial 17-pair result. This is a follow-up on that selected seed; the earlier three-seed report remains unchanged. The original development gate remains missed.', '',
             'The user extended the deadline by three hours: September 8 03:00 UTC, with new launches stopping at 02:00 UTC to reserve one hour for reporting. The original experiment contracts remain preserved.', '',
             'All 50 pairs use the original frozen instances and full initial-state/RGB pairing audits. Prior completed pairs are adopted without rerunning them. Report all failures; no outcome-based replacement or checkpoint selection is permitted.', '',
             'Slots 08/09 are absent. The 40 proximity sensors cover link1–link6; the prior audit found the inbound vessel outside sensor visibility in 7/8 variants. A PACT advantage does not establish clutter sensing. All `authorizes_*` flags remain false.', '',
             'Artifacts: `contract.json`, `analysis.json`, `adoption.json`; canonical full-stage ledger and pairing audits under `../../evaluation/`.']
    (EXT / 'EVAL.md').write_text('\n'.join(text) + '\n')


def supervise(installed):
    _, _, _, analysis, run = installed
    lock = (WORK / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (WORK / 'SAFE_SHUTDOWN.json').exists()
    verdict, reason, rows = 'INCOMPLETE', '', []
    try:
        run.stage('seed3104_full50_extension')
        jobs = full_jobs();prior = verify_prior_jobs(jobs)
        freeze(EXT / 'adoption.json', {'prior_pairs': 17, 'prior_jobs_sha256': digest(prior),
               'full_jobs_sha256': digest(jobs), 'additional_pairs': 33, 'new_training': False})
        remaining_eta(jobs)
        rows = run.run_eval(ROLE, 3104, 100)
        assert len(rows) == 100 and {r['seed'] for r in rows} == {3104}
        assert {r['episode_id'] for r in rows} == {r['episode_id'] for r in read(WORK / 'manifests/final_3104.json')['rows']}
        verdict = 'SEED 3104 FULL 50-PAIR EVALUATION COMPLETE'
        reason = 'Completed 50 valid paired instances / 100 rollouts: 17 existing pairs adopted and 33 additional pairs evaluated.'
    except BaseException as error:
        reason = f'{type(error).__name__}: {error}'
        atomic(EXT / 'stop_error.json', {'utc': now(), 'error': reason, 'traceback': traceback.format_exc()})
        print(traceback.format_exc(), flush=True)
    finally:
        for pool in run.POOLS:
            pool.pause_reason = pool.pause_reason or 'supervisor shutdown'
            for job, receipt in pool.drain():
                append(WORK / 'monitoring/shutdown_drain.jsonl', {'job': job, 'receipt': receipt})
        write_report(analysis, verdict, reason, rows)
        cleaned = run.cleanup_orphans()
        run.hourly_run_check(final=True)
        atomic(WORK / 'SAFE_SHUTDOWN.json', {**empty_authorization(), 'utc': now(), 'verdict': verdict,
               'reason': reason, 'report': str((EXT / 'EVAL.md').relative_to(WORK)), 'cleaned_owned_orphans': cleaned})
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
