"""Ledger-derived source freeze and split; no collection closeout required."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
from pact_place_v1011c_experiment import *


def source():
    from verify_pact_place_v1010_source import inspect
    ledger = ROOT / collection.COLLECTION_ROOT / 'ledger.jsonl'
    records = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    accepted = [row for row in records if row['accepted']]
    counts = collections.Counter(row['cell'] for row in accepted)
    assert len(accepted) == 99 and set(counts) == {collection.cell_key(*c) for c in collection.cells()}
    assert set(counts.values()) <= {4, 5}
    assert len({r['attempt_id'] for r in records}) == len(records)
    assert len({r['task_seed_u32'] for r in accepted}) == len(accepted)
    branch = subprocess.check_output(['git', '-C', str(ROOT / 'submodules/molmospaces'),
                                     'branch', '--show-current'], text=True).strip()
    assert branch == 'experiment/pact-vs-act-remediation-v2', branch
    assert sha256_file(Path(ENCODER_PATH)) == ENCODER_SHA256
    order = {collection.cell_key(*c): i for i,c in enumerate(collection.cells())}
    accepted.sort(key=lambda r:(order[r['cell']], r['attempt_index'], r['attempt_id']))
    rows, tasks, protected = [], [], {str(ledger.relative_to(ROOT)): sha256_file(ledger)}
    for index, row in enumerate(accepted):
        assert row['status'] == 'complete' and row['clean_success'] and row['task_success']
        assert not row['defects'] and row['error'] is None
        assert row['clutter_stability_event_count'] == 0
        assert all(row['contact_class_totals'][k] == 0 for k in ('clutter','mounted_fixture','hazard_bar','other_environment'))
        assert all(row['clutter_stability_events_by_slot'][s] == 0 for s in collection.ACTIVE_CLUTTER_SLOTS)
        assert row['coordinator_validation']['passed']
        expected = collection.build_row(row['family_id'],row['intrusion_side'],row['pose_id'],row['attempt_index'])
        assert expected['row_sha256'] == row['row_sha256']
        assert expected['attempt_id'] == row['attempt_id']
        assert expected['task_seed_u32'] == row['task_seed_u32']
        path = ROOT / collection.DATASET_ROOT / 'rows' / row['attempt_id'][:16] / 'trajectory.h5'
        assert path == ROOT / row['trajectory_h5'] and path.is_file()
        result = read(path.parent / 'result.json')
        assert result['status'] == 'complete'
        for file in sorted(path.parent.iterdir()):
            if file.is_file():
                protected[str(file.relative_to(ROOT))] = sha256_file(file)
        tasks.append((row['attempt_id'], str(path), row['trajectory_h5_sha256']))
        rows.append({**row,'act_episode_index':index,
                     'identity_sha256':expected['pact_v1011_identity_sha256']})
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        inspections = list(pool.map(inspect, tasks))
    by_id = {r['attempt_id']:r for r in inspections}
    for row in rows:
        got = by_id[row['attempt_id']]
        assert not got['problems'], got
        assert got['timesteps'] == row['episode_steps'] + 1
        row['timesteps'] = got['timesteps']
    lengths = [r['timesteps'] for r in rows]
    doc = {**empty_authorization(), 'schema_version':'pact_place_v1011c_source_manifest_v1',
           'is_phase0_pass':False, 'ledger_path':str(ledger.relative_to(ROOT)),
           'ledger_sha256':sha256_file(ledger), 'sensor_names':list(CANONICAL_SENSOR_NAMES),
           'sensor_order_sha256':SENSOR_ORDER_SHA256, 'verified':True,
           'counts':{'attempts':len(records),'accepted':len(rows),'t_min':min(lengths),
                     't_max':max(lengths),'t_sum':sum(lengths)},
           'balance':{'by_cell':dict(sorted(counts.items()))}, 'rows':rows,
           'sampler_class':collection.SAMPLER_CLASS, 'sampler_module':'molmo_spaces.tasks.enclosure_reach',
           'molmospaces_branch':branch, 'encoder_sha256':ENCODER_SHA256,
           'no_closeout_required':True, 'protected_source_files':protected,
           'implementation_hashes':{p:sha256_file(ROOT / p) for p in collection.IMPLEMENTATION_PATHS}}
    doc['payload_sha256'] = digest(doc)
    freeze(WORK / 'source_manifest.json',doc)
    print(json.dumps(doc['counts']), flush=True)


def split():
    src = read(WORK / 'source_manifest.json')
    conv = read(WORK / 'conversion_manifest_encoded.json')
    rows = src['rows']
    assert {r['attempt_id'] for r in rows} == {e['episode_id'] for e in conv['episodes']}
    cells = collections.defaultdict(list)
    for row in rows:
        cells[row['cell']].append(row)
    validation = {min(members,key=lambda r:hashlib.sha256(f"{SPLIT_SEED}:{r['attempt_id']}".encode()).hexdigest())['attempt_id']
                  for members in cells.values()}
    episodes, ranks = [], collections.Counter()
    for row in rows:
        label = 'validation' if row['attempt_id'] in validation else 'train'
        episodes.append({'act_episode_index':row['act_episode_index'],'episode_id':row['attempt_id'],
                         'candidate_index':row['attempt_index'],'hazard_present':True,'split':label,
                         'split_rank':ranks[label],'source_h5_sha256':row['trajectory_h5_sha256'],
                         **{k:row[k] for k in ('cell','family_id','intrusion_side','pose_id')}})
        ranks[label] += 1
    assert ranks['validation'] == len(cells) == 24
    assert ranks['train'] == len(rows) - len(cells)
    doc = {**empty_authorization(), 'schema':'hybrid_obstacle_canonical_split_v2',
           'experiment':'pact_place_v1011c_99', 'split_master_seed':SPLIT_SEED,
           'canonical_manifest_sha256':conv['source_manifest_payload_sha256'],
           'source_collection_tree_sha256':conv['converted_tree_file_sha256'],
           'split_rule':'One ledger episode per cell, lowest SHA256(split seed:attempt_id), goes to validation; all remaining episodes train.',
           'counts':{k:{'total':v,'hazard_present':v,'hazard_absent':0} for k,v in ranks.items()},
           'stratification':{k:dict(sorted(collections.Counter(e['cell'] for e in episodes if e['split']==k).items())) for k in ranks},
           'episodes':episodes}
    doc['split_manifest_sha256'] = digest(doc)
    freeze(WORK / 'split_manifest.json', doc)
    from fixed_split_data import load_split_manifest
    loaded = load_split_manifest(WORK / 'split_manifest.json')
    assert len(loaded['train']) == ranks['train'] and len(loaded['val']) == ranks['validation']
    print(json.dumps(doc['counts']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('source','split'),required=True)
    args = parser.parse_args()
    {'source':source,'split':split}[args.stage]()
