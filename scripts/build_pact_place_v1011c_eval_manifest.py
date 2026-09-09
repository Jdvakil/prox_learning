"""Freeze 50 paired instances per training seed, independent of outcomes."""
import collections
import hashlib
import re
import subprocess
from pact_place_v1011c_experiment import *


def historical_seeds():
    # Scan actual artifact fields, including historical retries, not summaries.
    query = r'"(?:task_seed_u32|seed_u32)"\s*:\s*[0-9]+'
    result = subprocess.run(['rg','--no-filename','--only-matching',query,
        'diagnostics_output','configs','--glob','*.json','--glob','*.jsonl'],
        cwd=ROOT,capture_output=True,text=True)
    assert result.returncode in (0,1), result.stderr
    seeds = {int(match.rsplit(':',1)[1]) for match in result.stdout.splitlines()}
    seeds.update(int(collection.cell_seed(*c,i)['seed_u32'])
                 for c in collection.cells() for i in range(collection.MAX_SCIENTIFIC_ATTEMPTS))
    seeds.update(SEEDS)
    return seeds


def main():
    historical = historical_seeds()
    used = set()
    all_rows, all_smoke = [], []
    cell_list = collection.cells()
    master_seeds = (SPLIT_SEED,EVAL_MASTER_SEED,SMOKE_MASTER_SEED,RETRY_MASTER_SEED)
    assert len(set(master_seeds)) == 4
    assert not set(master_seeds) & set(collection.HISTORICAL_MASTER_SEEDS)
    for seed_index, seed in enumerate(SEEDS):
        # 48 balanced draws plus two declared extra cells on opposite sides.
        extra = (cell_list[(seed_index*6)%24],cell_list[(seed_index*6+11)%24])
        plan = list(cell_list)*2 + list(extra)
        assert len(plan) == 50
        for role, master, cells in (
            ('eval',EVAL_MASTER_SEED,plan),
            ('smoke',SMOKE_MASTER_SEED,[cell_list[i] for i in (0,7,14,21)])):
            for index, cell in enumerate(cells):
                attempt = index
                while True:
                    row = collection.environment.build_row(*cell,attempt,
                        stream=f'pact_place_v1011c_{role}_seed{seed}', master_seed=master,role_index=index)
                    row.update(schema_version='pact_place_v1011c_paired_eval_v1',role=role,
                        candidate_index=index,role_index=index,checkpoint_seed=seed,
                        episode_id=hashlib.sha256(f'v1011c:{role}:{master}:{seed}:{index}:{attempt}'.encode()).hexdigest(),
                        max_sampling_retries=collection.MAX_SAMPLING_RETRIES,hazard_present=True)
                    candidate = {int(row['task_seed_u32'])}
                    retry = [retry_seed(row,i) for i in range(1,row['max_sampling_retries']+1)]
                    candidate.update(int(r['seed_u32']) for r in retry)
                    if len(candidate) == 1+len(retry) and not candidate & (historical|used):
                        break
                    attempt += 1000
                used.update(candidate)
                row.pop('row_sha256',None)
                row['row_sha256'] = digest(row)
                (all_rows if role=='eval' else all_smoke).append(row)
    assert len(all_rows) == 150 and len(all_smoke) == 12
    doc = {**empty_authorization(),'schema_version':'pact_place_v1011c_paired_eval_v1',
        'environment_version':collection.ENVIRONMENT_VERSION,'sampler_class':collection.SAMPLER_CLASS,
        'sampler_module':'molmo_spaces.tasks.enclosure_reach','molmospaces_branch':'experiment/pact-vs-act-remediation-v2',
        'master_seeds':{'split':SPLIT_SEED,'eval':EVAL_MASTER_SEED,'smoke':SMOKE_MASTER_SEED,'retry':RETRY_MASTER_SEED},
        'training_seeds':list(SEEDS),'num_queries':100,'training_chunk':100,'task_horizon':900,
        'end_on_success':False,'action_noise_enabled':False,'paired':True,'instances_per_seed':50,
        'total_candidates':150,'scientific_rollouts':300,'role':'eval',
        'sensor_names':list(CANONICAL_SENSOR_NAMES),'sensor_order_sha256':SENSOR_ORDER_SHA256,
        'encoder_sha256':ENCODER_SHA256,'rows':all_rows,'smoke':{'rows':all_smoke,'performance_gate':False},
        'seed_audit':{'historical_unique_seeds':len(historical),'initial_and_retry_seeds':len(used),
                      'disjoint':not bool(used & historical),'retry_seeds_mutually_disjoint':True},
        'balance':{str(seed):{k:dict(sorted(collections.Counter(r[k] for r in all_rows if r['checkpoint_seed']==seed).items()))
                            for k in ('cell','family_id','intrusion_side','pose_id')} for seed in SEEDS}}
    doc['manifest_sha256'] = digest(doc)
    freeze(EVAL / 'eval_manifest.json',doc)
    load_eval_manifest(EVAL / 'eval_manifest.json')
    print(json.dumps({'instances':150,'rollouts':300,'seed_audit':doc['seed_audit'],'balance':doc['balance']}),flush=True)


if __name__ == '__main__':
    main()
