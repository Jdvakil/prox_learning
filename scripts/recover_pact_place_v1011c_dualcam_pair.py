"""One bounded, recorded same-instance repeat; never relax frozen pairing gates."""
from pact_place_v1011c_dualcam import *
from pact_place_v1011c_dualcam_metrics import metrics, check_pair, aggregate
from finish_pact_place_v1011c_dualcam import stage, cleanup_orphans
import argparse
import subprocess
import time
import traceback
from datetime import datetime, timezone

RECOVERY = WORK / 'recovery_03'


def main():
    plan = read(WORK / 'pairing_recovery_plan_01.json')
    assert plan['max_additional_attempts_per_arm'] == 1
    assert not plan['relax_pairing_tolerance'] and not plan['new_seed_sampling']
    deadline = datetime.fromisoformat(DEADLINE_UTC).timestamp()
    prior_pid = read(WORK / 'pilot_launch_03.json')['pid']
    while not (WORK / 'stages/07_final.json').exists() or not (WORK / 'pilot_failure.json').exists():
        assert time.time() < deadline, 'Original stage did not reach a recorded failure before deadline.'
        if (WORK / 'pilot_completion.json').exists():
            raise RuntimeError('Original pipeline already completed; refusing unnecessary repeat.')
        time.sleep(30)
    # The original controller must have finished writing its failure evidence.
    while Path(f'/proc/{prior_pid}').exists():
        assert time.time() < deadline
        command = Path(f'/proc/{prior_pid}/cmdline').read_bytes()
        if b'finish_pact_place_v1011c_dualcam.py' not in command:
            break
        time.sleep(1)
    prior_stage = read(WORK / 'stages/07_final.json')
    assert prior_stage['returncode'] == 1, prior_stage
    prior_run = read(EVAL / 'final_run.json')
    assert prior_run['rollouts_complete'] == prior_run['rollouts_expected'] == 100
    assert not prior_run['infrastructure_healthy']
    assert {p['pair'][1] for p in prior_run['pairing_errors']} == {plan['episode_id']}, \
        'Unexpected additional pairing failure: do not expand the recorded repair.'
    assert len(prior_run['pair_checks']) == 49
    scheduled = read(EVAL / 'final_schedule.json')['jobs']
    jobs = {j['schedule']['rollout_id']: j for j in scheduled}
    original = [json.loads(x) for x in (EVAL / 'final_ledger.jsonl').read_text().splitlines()]
    assert len(original) == len(jobs) == 100
    original_rows = []
    for receipt in original:
        assert receipt['status'] == 'complete' and receipt['returncode'] == 0
        job = jobs[receipt['rollout_id']]
        assert receipt == read(Path(job['output']) / 'worker_completion.json')
        row = metrics(Path(job['output']), job['schedule'])
        assert row == receipt['metrics']
        original_rows.append(row)
    affected = [r for r in original if r['episode_id'] == plan['episode_id']]
    assert len(affected) == 2 and {r['arm'] for r in affected} == {'ACT', 'PACT'}
    RECOVERY.mkdir(exist_ok=False)
    freeze(RECOVERY / 'original_outcomes.json', {**empty_authorization(), 'raw_verified_rollouts': 100,
        'strictly_verified_pairs': 49,
        'aggregate': {a: aggregate([r for r in original_rows if r['arm'] == a]) for a in ('ACT', 'PACT')},
        'affected_pair_raw_metrics': [r['metrics'] for r in affected]})
    moves = []
    def archive(source, target):
        assert source.exists() and not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        files = sorted(source.rglob('*')) if source.is_dir() else [source]
        hashes = {str(p.relative_to(source) if source.is_dir() else Path(p.name)): sha256_file(p)
                  for p in files if p.is_file()}
        was_directory = source.is_dir()
        source.rename(target)
        for name, value in hashes.items():
            assert sha256_file(target / name if was_directory else target) == value
        moves.append({'original': str(source.relative_to(ROOT)), 'archive': str(target.relative_to(ROOT)),
                      'file_hashes': hashes})
    for relative in ('stages/07_final.json', 'logs/07_final.log', 'launches/07_final.json', 'pilot_failure.json'):
        archive(WORK / relative, RECOVERY / relative)
    for name in ('final_run.json', 'final_ledger.jsonl'):
        archive(EVAL / name, RECOVERY / name)
    for receipt in affected:
        archive(ROOT / receipt['directory'], RECOVERY / 'original_pair' / receipt['arm'])
    freeze(RECOVERY / 'archive_receipt.json', {**empty_authorization(), 'moves': moves,
        'original_stage_actual_returncode': 1, 'original_controller_exit_code': None,
        'original_controller_observed_exited': True, 'plan_sha256': sha256_file(WORK / 'pairing_recovery_plan_01.json'),
        'pairing_code_sha256': sha256_file(ROOT / 'scripts/pact_place_v1011c_dualcam_metrics.py'),
        'outcome_selection': False, 'additional_attempt_limit_per_arm': 1})
    retained = [r for r in original if r['episode_id'] != plan['episode_id']]
    assert len(retained) == 98
    with (EVAL / 'final_ledger.jsonl').open('x') as stream:
        for receipt in retained:
            stream.write(json.dumps(receipt, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    print('REPAIR: archived original failed gate and both original attempts; reconciling 98 completions, repeating exactly two jobs.', flush=True)
    selected = read(EVAL / 'checkpoint_selection.json')['selected_global_step']
    stage('07_final', [str(ROOT / 'scripts/run_pact_place_v1011c_dualcam_eval.py'),
                      '--stage', 'final', '--workers', '12', '--step', str(selected), '--h5-only'])
    final = [json.loads(x) for x in (EVAL / 'final_ledger.jsonl').read_text().splitlines()]
    repeated = [r for r in final if r['episode_id'] == plan['episode_id']]
    assert len(final) == 100 and len(repeated) == 2
    repeat_rows = {r['arm']: metrics(ROOT / r['directory'], r) for r in repeated}
    pairing = check_pair(repeat_rows['ACT'], repeat_rows['PACT'])
    archived_rows = {r['arm']: metrics(RECOVERY / 'original_pair' / r['arm'], r) for r in affected}
    for receipt in affected:
        old = dict(archived_rows[receipt['arm']])
        old['directory'] = receipt['metrics']['directory']
        assert old == receipt['metrics']
    freeze(RECOVERY / 'repair_completion.json', {**empty_authorization(), 'complete': True,
        'episode_id': plan['episode_id'], 'additional_rollouts': 2,
        'original_pair_raw_metrics': archived_rows, 'repeat_pair_raw_metrics': repeat_rows,
        'repeat_pair_check': pairing, 'pairing_tolerance_unchanged': True,
        'both_original_and_repeat_report_required': True,
        'completed_utc': datetime.now(timezone.utc).isoformat()})
    stage('08_final_raw_audit', [str(ROOT / 'scripts/finalize_pact_place_v1011c_dualcam.py')])
    cleanup_orphans()
    freeze(WORK / 'pilot_completion.json', {**empty_authorization(), 'complete': True,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'before_deadline': time.time() < deadline,
        'final_rollouts': 100, 'final_pairs': 50, 'seed': 3103, 'selected_global_step': selected,
        'additional_pairing_recovery_rollouts': 2,
        'analysis_sha256': sha256_file(EVAL / 'analysis.json')})
    print('PILOT COMPLETE with one disclosed, bounded same-instance pairing repeat.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve())]
        with (WORK / 'pairing_recovery_01.log').open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        freeze(WORK / 'pairing_recovery_launch_01.json', {**empty_authorization(), 'pid': child.pid,
            'command': command, 'started_utc': datetime.now(timezone.utc).isoformat()})
        print(f'Bounded pairing-recovery controller PID {child.pid}', flush=True)
    else:
        try:
            main()
        except Exception:
            failure = traceback.format_exc()
            freeze(WORK / 'pairing_recovery_failure.json', {**empty_authorization(), 'complete': False,
                'error': failure, 'utc': datetime.now(timezone.utc).isoformat()})
            print(failure, flush=True)
            raise
