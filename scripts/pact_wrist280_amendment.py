"""Frozen 280-episode continuation of wrist288, authorized after collection.

Original code, configuration, seed streams, and raw ledgers stay immutable.
Only cohort size/balance and validation size change; 240 training episodes keep
the original 30-update epochs and exact checkpoint/optimizer schedule intact.
"""
from __future__ import annotations
import argparse
import collections
import copy
import inspect
import time
from pact_wrist288_common import *
import pact_wrist288_common as common

AMEND = WORK / 'amendments/wrist280'
SELECTED = AMEND / 'selected_ledger.jsonl'
AMEND_CODE = CODE / 'pact_wrist280_amendment.py'
BASE_CHECK = common.check_bindings


def validation_allocation(counts, registry):
    assert len(counts) == 24 and sum(counts.values()) == 280
    assert all(2 <= n <= 12 for n in counts.values())
    quotas = {cell: max(1, n * 40 // 280) for cell, n in counts.items()}
    # Largest remainders allocate exactly 40. Ties use the existing split seed;
    # no new random stream, model outcome, or trajectory content is inspected.
    order = sorted(counts, key=lambda cell: (
        -(counts[cell] * 40 % 280),
        digest([NAMESPACE, 'wrist280_validation_apportionment',
                registry['entries'][f'split:{cell}:0'][0]['derivation'], cell])))
    for cell in order[:40 - sum(quotas.values())]:
        quotas[cell] += 1
    assert sum(quotas.values()) == 40
    assert all(1 <= quotas[cell] < counts[cell] for cell in counts)
    return quotas


def validation_ids(rows, registry, quotas):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row['cell']].append(row)
    assert set(grouped) == set(quotas)
    result = set()
    for cell, members in grouped.items():
        seed = registry['entries'][f'split:{cell}:0'][0]['derivation']
        ranked = sorted(members, key=lambda r: digest([NAMESPACE, seed, r['attempt_id']]))
        result.update(r['attempt_id'] for r in ranked[:quotas[cell]])
    assert len(result) == 40
    return result


def verify_rows(records, expected_counts):
    identities = {r['attempt_id'] for r in records}
    assert len(identities) == len(records), 'duplicate collection attempt'
    accepted = [r for r in records if r['accepted']]
    assert len(accepted) == 280
    assert dict(collections.Counter(r['cell'] for r in accepted)) == expected_counts
    for row in records:
        assert row['returncode'] == 0 or not row['accepted']
        assert row['attempt_id'] == row['row']['attempt_id']
        assert row['row_sha256'] == row['row']['row_sha256']
    return accepted


def check_effective():
    base = BASE_CHECK()
    value = read(AMEND / 'effective_config.json')
    assert value['config_sha256'] == digest({k: v for k, v in value.items() if k != 'config_sha256'})
    amendment = value['dataset_amendment']
    assert amendment['base_config_sha256'] == base['config_sha256']
    assert sha(SELECTED) == amendment['selected_ledger_sha256']
    assert sha(AMEND / 'request.json') == amendment['request_sha256']
    assert sha(AMEND / 'cohort.json') == amendment['cohort_sha256']
    for path, expected in amendment['adapter_file_hashes'].items():
        assert sha(ROOT / path) == expected, f'amendment input drift: {path}'
    return value


def prepare():
    base = BASE_CHECK()
    request = read(AMEND / 'request.json')
    assert sha(SELECTED) == request['selected_ledger_sha256']
    records = lines(SELECTED)
    rows = verify_rows(records, request['by_cell'])
    live = {r['attempt_id']: r for r in lines(WORK / 'collection/ledger.jsonl')}
    for row in records:
        assert live[row['attempt_id']] == row
    for row in rows:
        directory = Path(row['directory'])
        receipt = read(directory / 'exit_receipt.json')
        result = read(directory / 'worker_result.json')
        assert receipt['returncode'] == 0
        assert sha(directory / 'exit_receipt.json') == row['exit_receipt_sha256']
        assert sha(ROOT / row['trajectory_h5']) == row['trajectory_h5_sha256']
        assert result['accepted'] and not result['defects'] and result['schema_validation']['passed']
    registry = read(WORK / 'seed_registry.json')
    quotas = validation_allocation(request['by_cell'], registry)
    validation = validation_ids(rows, registry, quotas)
    cohort = {'episodes': sorted(({'episode_id': r['attempt_id'], 'cell': r['cell'],
                'row_sha256': r['row_sha256'], 'source_h5_sha256': r['trajectory_h5_sha256'],
                'split': 'validation' if r['attempt_id'] in validation else 'train'} for r in rows),
                key=lambda r: r['episode_id']),
              'counts': {'total': 280, 'train': 240, 'validation': 40},
              'selected_by_cell': request['by_cell'], 'validation_by_cell': quotas,
              'train_by_cell': {cell: n - quotas[cell] for cell, n in request['by_cell'].items()},
              'selected_ledger_sha256': sha(SELECTED), 'seed_registry_sha256': sha(WORK / 'seed_registry.json')}
    freeze(AMEND / 'cohort.json', cohort)
    effective = copy.deepcopy(base)
    effective.pop('config_sha256')
    effective['collection'].update(episodes=280, per_cell=request['by_cell'],
        completion_rule='first 280 strictly accepted committed episodes, frozen at user-requested cutoff')
    effective['split'].update(validation=40, validation_per_cell=quotas,
        ranking='40 validation episodes apportioned across 24 cells by largest remainder of 40*n_cell/280 (minimum 1); split-seed hash breaks quota ties; original SHA256(namespace, split seed derivation, accepted episode identity) ranks within each cell')
    adapter_files = ['scripts/pact_wrist280_amendment.py', 'tests/test_pact_wrist280_amendment.py',
                     'docs/PACT_PLACE_V1010_WRIST280_AMENDMENT.md']
    effective['dataset_amendment'] = {
        'base_config_sha256': base['config_sha256'], 'request_sha256': sha(AMEND / 'request.json'),
        'selected_ledger_sha256': sha(SELECTED), 'cohort_sha256': sha(AMEND / 'cohort.json'),
        'adapter_file_hashes': {p: sha(ROOT / p) for p in adapter_files},
        'timing': 'after collection observations, before any conversion/training/development/final evaluation',
        'limitations': ['training-cell balance differs from original 12-per-cell plan',
                       'validation is 40 episodes, with one or two per cell, instead of 48',
                       'stopping collection responds to observed collection yield; final/development manifests remain frozen independently']}
    effective['file_hashes'].update(effective['dataset_amendment']['adapter_file_hashes'])
    effective['config_sha256'] = digest(effective)
    freeze(AMEND / 'effective_config.json', effective)
    check_effective()
    print(json.dumps({'config_sha256': effective['config_sha256'], 'counts': cohort['counts'],
                      'validation_by_cell': quotas}), flush=True)


def replace_function(module, function, edits):
    source = inspect.getsource(getattr(module, function))
    for old, new, count in edits:
        assert source.count(old) == count, (function, old, source.count(old), count)
        source = source.replace(old, new)
    exec(compile(source, str(AMEND_CODE), 'exec'), module.__dict__)


def install():
    effective = check_effective()
    common.check_bindings = check_effective
    import pact_wrist288_data as data
    import pact_wrist288_train as train
    import pact_wrist288_monitor as monitor
    import pact_wrist288_analysis as analysis
    import pact_wrist288_run as run
    for module in (data, train, monitor, analysis, run):
        module.check_bindings = check_effective
        module.AMEND_CODE = AMEND_CODE
    cohort = read(AMEND / 'cohort.json')

    def selected_verifier(records):
        assert records == lines(SELECTED), 'converter must use the frozen selected cohort ledger'
        return verify_rows(records, cohort['selected_by_cell'])

    def selected_split(rows, registry):
        chosen = validation_ids(rows, registry, cohort['validation_by_cell'])
        assert chosen == {r['episode_id'] for r in cohort['episodes'] if r['split'] == 'validation'}
        return chosen

    data.verify_ledger = selected_verifier
    data.split_rows = selected_split
    replace_function(data, 'main', [
        ("'collection/ledger.jsonl'", "'amendments/wrist280/selected_ledger.jsonl'", 2),
        ("'validation':48", "'validation':40", 2),
        ("'converted':288", "'converted':280", 1),
        ('/288 T=', '/280 T=', 1),
        ("'converter_module_sha256':sha(__file__)",
         "'converter_module_sha256':sha(__file__),'adapter_module_sha256':sha(AMEND_CODE)", 1)])

    original_command = train.command

    def training_command(arm, seed, stop):
        command = original_command(arm, seed, stop)
        command[:2] = [sys.executable, str(AMEND_CODE), '--mode', 'train']
        return command

    train.command = training_command
    original_jobs = run.eval_jobs

    def evaluation_jobs(role, seed, history):
        jobs = original_jobs(role, seed, history)
        for job in jobs:
            job['command'][:2] = [sys.executable, str(AMEND_CODE), '--mode', 'eval']
        return jobs

    run.eval_jobs = evaluation_jobs
    replace_function(run, 'finish_seed', [("len(manifest['val_act_indices'])==48", "len(manifest['val_act_indices'])==40", 1)])
    replace_function(run, 'single_process', [
        ('[sys.executable,str(CODE/script)]', "[sys.executable,str(AMEND_CODE),'--mode','convert']", 1)])

    def adopt_preflight():
        run.stage('preflight_adopted_for_wrist280')
        prior = read(WORK / 'preflight_complete.json')
        assert prior['passed'] and len(prior['cells']) == 24 and all(r['passed'] for r in prior['cells'])
        assert prior['config_sha256'] == effective['dataset_amendment']['base_config_sha256']
        freeze(AMEND / 'preflight_adoption.json', {
            'original_preflight_sha256': sha(WORK / 'preflight_complete.json'),
            'effective_config_sha256': effective['config_sha256'], 'environment_changed': False})

    def close_collection():
        run.stage('collection_closed_at_280')
        rows = selected_verifier(lines(SELECTED))
        actual = lines(WORK / 'collection/ledger.jsonl')
        live = {r['attempt_id']: r for r in actual}
        for row in rows:
            assert live[row['attempt_id']] == row
            assert sha(ROOT / row['trajectory_h5']) == row['trajectory_h5_sha256']
            assert sha(Path(row['directory']) / 'exit_receipt.json') == row['exit_receipt_sha256']
        selected = {r['attempt_id'] for r in rows}
        freeze(WORK / 'collection_complete.json', {'accepted': 280,
            'ledger_sha256': sha(WORK / 'collection/ledger.jsonl'), 'selected_ledger_sha256': sha(SELECTED),
            'by_cell': cohort['selected_by_cell'], 'effective_config_sha256': effective['config_sha256'],
            'excluded_post_cutoff_accepted_ids': [r['attempt_id'] for r in actual if r['accepted'] and r['attempt_id'] not in selected]})

    run.run_preflight = adopt_preflight
    run.run_collection = close_collection
    replace_function(analysis, 'report', [
        ('V10.10 wrist288 three-seed experiment', 'V10.10 wrist280 three-seed experiment (amended cohort)', 1),
        ('WORK/"collection/ledger.jsonl"', 'WORK/"amendments/wrist280/selected_ledger.jsonl"', 1),
        ('/288.', '/280.', 2)])
    original_report = analysis.report

    def report(verdict, reason, rows=None):
        original_report(verdict, reason, rows)
        path = WORK / 'EVAL.md'
        body = path.read_text() + '\nDataset amendment: 280 selected strict-clean episodes; 240 training / 40 validation. All 24 cells are represented; collected totals are 12 in 21 cells and 8/9/11 in the three F2-left cells. This replaces the original balanced 288-episode / 48-validation design at the user’s request, before any model training. The 60,000-update budget, development gate, final manifests and evaluation rules are unchanged.\n\n'
        body += '| Cell | Selected | Train | Validation |\n|---|---:|---:|---:|\n'
        for cell in sorted(cohort['selected_by_cell']):
            body += f"| {cell.replace('|', ' / ')} | {cohort['selected_by_cell'][cell]} | {cohort['train_by_cell'][cell]} | {cohort['validation_by_cell'][cell]} |\n"
        body += '\nBindings: `amendments/wrist280/effective_config.json`, `cohort.json`, `selected_ledger.jsonl`; original `config.json` and full collection ledger retained.\n'
        path.write_text(body)

    analysis.report = report
    replace_function(monitor, 'hourly_run_check', [
        ("'collection/ledger.jsonl'", "'amendments/wrist280/selected_ledger.jsonl'", 1),
        ('288-sum(accepted.values())', '280-sum(accepted.values())', 1)])
    run.hourly_run_check = monitor.hourly_run_check
    original_resources = monitor.resources

    def resources_with_retry():
        for attempt in range(3):
            try:
                return original_resources()
            except BlockingIOError:
                if attempt == 2:
                    raise
                time.sleep(1)

    monitor.resources = resources_with_retry
    run.resources = resources_with_retry
    return data, train, monitor, analysis, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['prepare', 'convert', 'train', 'eval', 'monitor', 'supervisor'])
    args, rest = parser.parse_known_args()
    sys.argv = [str(AMEND_CODE), *rest]
    os.environ.update(environment())
    if args.mode == 'prepare':
        prepare()
        return
    data, train, monitor, analysis, run = install()
    if args.mode == 'convert':
        data.main()
    elif args.mode == 'train':
        train.main()
    elif args.mode == 'eval':
        import pact_wrist288_eval_worker as worker
        worker.main()
    elif args.mode == 'monitor':
        monitor.main()
    else:
        run.main()


if __name__ == '__main__':
    main()
