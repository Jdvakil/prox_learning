"""Read-only scientific recount after a failed gate; never declares completion."""
from pact_place_v1011c_dualcam import *
from pact_place_v1011c_dualcam_metrics import metrics, check_pair, aggregate
from finalize_pact_place_v1011c_dualcam import authorization_check
from datetime import datetime, timezone
import h5py
import numpy as np


def main():
    assert read(WORK/'pairing_recovery_failure.json')['complete'] is False
    assert read(WORK/'stages/07_final.json')['returncode'] == 1
    assert not (WORK/'pilot_completion.json').exists()
    assert not (WORK/'stages/08_final_raw_audit.json').exists()
    jobs = read(EVAL/'final_schedule.json')['jobs']
    ledger = [json.loads(x) for x in (EVAL/'final_ledger.jsonl').read_text().splitlines() if x.strip()]
    expected = {j['schedule']['rollout_id']: j for j in jobs}
    assert len(ledger) == len(expected) == 100
    assert {r['rollout_id'] for r in ledger} == set(expected)
    rows, pairs = [], {}
    for index, receipt in enumerate(ledger):
        job = expected[receipt['rollout_id']]
        assert receipt['returncode'] == 0 and receipt['status'] == 'complete'
        assert receipt == read(Path(job['output'])/'worker_completion.json')
        row = metrics(Path(job['output']), job['schedule'])
        assert row['seed'] == 3103 and row['training_global_step'] == 30000
        assert row['arm'] not in pairs.setdefault(row['episode_id'], {})
        pairs[row['episode_id']][row['arm']] = row
        rows.append(row)
        if (index+1) % 20 == 0:
            print(f'Independent blocked-run raw recount: {index+1}/100', flush=True)
    passed, failed = [], []
    for episode_id, pair in pairs.items():
        assert set(pair) == {'ACT', 'PACT'}
        try:
            passed.append(check_pair(pair['ACT'], pair['PACT']))
        except AssertionError as error:
            failed.append({'episode_id': episode_id, 'error': str(error)})
    assert len(passed) == 49 and len(failed) == 1
    affected = failed[0]['episode_id']
    original_rows = [metrics(WORK/'recovery_03/original_pair'/arm) for arm in ('ACT', 'PACT')]
    assert all(r['episode_id'] == affected for r in original_rows)
    original_panel = [r for r in rows if r['episode_id'] != affected] + original_rows
    original_aggregate = {a: aggregate([r for r in original_panel if r['arm'] == a]) for a in ('ACT', 'PACT')}
    assert original_aggregate == read(WORK/'recovery_03/original_outcomes.json')['aggregate']
    # Continue through every initial dataset for diagnosis, WITHOUT changing
    # or bypassing the strict gate above. A failed gate remains failed.
    detailed = {}
    pair = pairs[affected]
    with h5py.File(ROOT/pair['ACT']['directory']/'initial_observation.h5', 'r') as a, h5py.File(ROOT/pair['PACT']['directory']/'initial_observation.h5', 'r') as b:
        names, other = [], []
        a.visititems(lambda n,o: names.append(n) if isinstance(o,h5py.Dataset) else None)
        b.visititems(lambda n,o: other.append(n) if isinstance(o,h5py.Dataset) else None)
        assert names == other
        detailed = {'exact_non_rgb_datasets': 0, 'rgb': {}}
        for name in names:
            av, bv = np.asarray(a[name][()]), np.asarray(b[name][()])
            assert av.dtype == bv.dtype and av.shape == bv.shape
            if name in ['observation/'+c for c in CAMERAS]:
                delta = np.abs(av.astype(np.int16)-bv.astype(np.int16))
                values, counts = np.unique(delta, return_counts=True)
                detailed['rgb'][name] = {'channel_values': int(av.size), 'absolute_difference_histogram': {str(int(v)): int(c) for v,c in zip(values,counts)}}
            else:
                assert av.tobytes() == bv.tobytes(), name
                detailed['exact_non_rgb_datasets'] += 1
    source = read(WORK/'source_manifest.json')
    checked, hash_errors = 0, []
    hashes = source['protected_source_files'] | source['implementation_hashes']
    for relpath, expected_hash in hashes.items():
        actual = sha256_file(ROOT/relpath)
        checked += 1
        if actual != expected_hash:
            hash_errors.append({'path': relpath, 'expected': expected_hash, 'actual': actual})
    for path in WORK.rglob('*.json'):
        authorization_check(read(path))
    document = {**empty_authorization(), 'complete': False, 'verified_paired_evaluation': False,
        'diagnostic_recount_completed': True, 'utc': datetime.now(timezone.utc).isoformat(),
        'actual_exit_zero_rollouts': 100, 'raw_recomputed_rollouts_including_original_pair': 102,
        'strict_pairs_passed': 49, 'strict_pairs_failed': failed,
        'repeat_initial_full_comparison': detailed,
        'original_first_attempt_aggregate': original_aggregate,
        'repeat_panel_aggregate': {a: aggregate([r for r in rows if r['arm'] == a]) for a in ('ACT', 'PACT')},
        'passing_pairs_descriptive_subset': {a: aggregate([r for r in rows if r['arm'] == a and r['episode_id'] != affected]) for a in ('ACT', 'PACT')},
        'subset_is_not_replacement_benchmark': True, 'raw_metrics': rows, 'original_pair_raw_metrics': original_rows,
        'source_hashes_checked': checked, 'source_hash_errors': hash_errors,
        'all_authorization_flags_false': True, 'normal_stage_08_was_not_run': True,
        'pairing_tolerance_unchanged': True, 'further_rollouts_launched': False}
    freeze(EVAL/'blocked_raw_audit.json', document)
    print(json.dumps({k:v for k,v in document.items() if k not in ('raw_metrics','original_pair_raw_metrics')}), flush=True)


if __name__ == '__main__':
    main()
