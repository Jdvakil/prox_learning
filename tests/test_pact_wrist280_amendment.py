import copy
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pact_wrist280_amendment as amendment


def fixture_cohort():
    registry = amendment.read(amendment.WORK / 'seed_registry.json')
    records = amendment.lines(amendment.SELECTED)
    counts = amendment.read(amendment.AMEND / 'request.json')['by_cell']
    return registry, records, counts


def test_actual_cohort_apportionment_and_original_ranking():
    registry, records, counts = fixture_cohort()
    rows = amendment.verify_rows(records, counts)
    quotas = amendment.validation_allocation(counts, registry)
    validation = amendment.validation_ids(rows, registry, quotas)
    assert len(rows) == 280 and len(validation) == 40
    assert len(set(r['attempt_id'] for r in rows) - validation) == 240
    assert len(quotas) == 24 and sorted(quotas.values()) == [1] * 8 + [2] * 16
    assert all(quotas[cell] == 1 for cell in counts if counts[cell] < 12)
    for cell in counts:
        seed = registry['entries'][f'split:{cell}:0'][0]['derivation']
        ranked = sorted((r for r in rows if r['cell'] == cell),
                        key=lambda r: amendment.digest([amendment.NAMESPACE, seed, r['attempt_id']]))
        assert [r['attempt_id'] in validation for r in ranked] == [True] * quotas[cell] + [False] * (counts[cell] - quotas[cell])


def test_selection_is_order_independent_and_postcutoff_ids_stay_out():
    registry, records, counts = fixture_cohort()
    rows = amendment.verify_rows(records, counts)
    quotas = amendment.validation_allocation(counts, registry)
    expected = amendment.validation_ids(rows, registry, quotas)
    shuffled = rows.copy()
    random.Random(7).shuffle(shuffled)
    assert amendment.validation_ids(shuffled, registry, quotas) == expected
    all_rows = amendment.lines(amendment.WORK / 'collection/ledger.jsonl')
    selected = {r['attempt_id'] for r in rows}
    assert selected.issubset({r['attempt_id'] for r in all_rows if r['accepted']})
    assert expected.issubset(selected)


def test_selected_ledger_rejects_identity_count_and_receipt_tampering():
    _, records, counts = fixture_cohort()
    with pytest.raises(AssertionError):
        amendment.verify_rows(records + [records[0]], counts)
    bad = copy.deepcopy(records)
    row = next(r for r in bad if r['accepted'])
    row['returncode'] = 1
    with pytest.raises(AssertionError):
        amendment.verify_rows(bad, counts)
    row['returncode'] = 0
    row['row_sha256'] = 'tampered'
    with pytest.raises(AssertionError):
        amendment.verify_rows(bad, counts)
    with pytest.raises(AssertionError):
        amendment.verify_rows(records, {**counts, next(iter(counts)): 0})


def test_function_adaptation_fails_closed_on_unexpected_source():
    import types
    module = types.SimpleNamespace(fn=fixture_cohort, __dict__={})
    with pytest.raises(AssertionError):
        amendment.replace_function(module, 'fn', [('absent sentinel', 'x', 1)])


def test_all_runtime_adapters_compile_and_training_flags_stay_identical(tmp_path):
    import subprocess
    code = r'''
import sys, copy
from pathlib import Path
sys.path.insert(0, 'scripts')
import pact_wrist280_amendment as a
import pact_wrist288_train as trainer
records = a.lines(a.SELECTED)
counts = a.read(a.AMEND/'request.json')['by_cell']
registry = a.read(a.WORK/'seed_registry.json')
rows = a.verify_rows(records, counts)
quotas = a.validation_allocation(counts, registry)
validation = a.validation_ids(rows, registry, quotas)
cohort = {'selected_by_cell':counts,'validation_by_cell':quotas,
 'episodes':[{'episode_id':r['attempt_id'],'split':'validation' if r['attempt_id'] in validation else 'train'} for r in rows]}
base = a.read(a.WORK/'config.json')
effective = copy.deepcopy(base)
effective['dataset_amendment'] = {'base_config_sha256':base['config_sha256']}
a.check_effective = lambda: effective
real_read = a.read
a.read = lambda path: cohort if Path(path)==a.AMEND/'cohort.json' else real_read(path)
old_command = trainer.command
data, trainer, monitor, analysis, run = a.install()
assert len(data.verify_ledger(records)) == 280
assert data.split_rows(rows,registry) == validation
trainer.WORK = Path(sys.argv[1])
a.atomic(trainer.WORK/'split_manifest.json', {'split_manifest_sha256':'split'})
a.atomic(trainer.WORK/'conversion_manifest.json', {'converted_tree_file_sha256':'tree','timesteps':{'converted_t_max':640}})
for arm in ('act','pact'):
 for stop in (900,1800,30000,60000):
  expected = old_command(arm,3103,stop)
  expected[:2] = [sys.executable,str(a.AMEND_CODE),'--mode','train']
  assert trainer.command(arm,3103,stop) == expected
assert data.main.__code__.co_filename == str(a.AMEND_CODE)
assert run.finish_seed.__code__.co_filename == str(a.AMEND_CODE)
assert run.single_process.__code__.co_filename == str(a.AMEND_CODE)
assert run.hourly_run_check is monitor.hourly_run_check
assert run.resources is monitor.resources
print('Runtime adapters compiled; ACT/PACT training arguments preserve all original settings.')
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)],
                            cwd=amendment.ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
