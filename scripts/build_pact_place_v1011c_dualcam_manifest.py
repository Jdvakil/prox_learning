"""Preregister disjoint smoke, development, and final-test streams."""
from pact_place_v1011c_dualcam import *
from build_pact_place_v1011c_eval_manifest import historical_seeds
import collections


def main():
    historical = historical_seeds()
    # Explicitly include the completed current experiment even if rg's ignore
    # rules change. Reserve every old initial and retry seed.
    prior = original.load_eval_manifest(original.EVAL/'eval_manifest.json')
    for row in prior['rows']+prior['smoke']['rows']:
        historical.add(row['task_seed_u32'])
        historical.update(original.retry_seed(row,i)['seed_u32'] for i in range(1,row['max_sampling_retries']+1))
    cells = collection.cells()
    # One side per family/pose, alternating side, gives 12 development cells.
    families = sorted({c[0] for c in cells})
    poses = sorted({c[2] for c in cells})
    sides = sorted({c[1] for c in cells})
    dev = [c for c in cells if sides.index(c[1]) == (families.index(c[0])+poses.index(c[2])) % 2]
    assert len(dev) == 12
    plan = [('final',EVAL_MASTER_SEED,list(cells)*2+[cells[0],cells[11]]),
            ('development',DEV_MASTER_SEED,dev),
            ('smoke',SMOKE_MASTER_SEED,[cells[i] for i in (0,7,14,21)])]
    assert len({EVAL_MASTER_SEED,DEV_MASTER_SEED,SMOKE_MASTER_SEED,RETRY_MASTER_SEED}) == 4
    assert not {EVAL_MASTER_SEED,DEV_MASTER_SEED,SMOKE_MASTER_SEED,RETRY_MASTER_SEED} & set(collection.HISTORICAL_MASTER_SEEDS)
    used, rows, smoke = set(),[],[]
    for role,master,selected in plan:
        for index,cell in enumerate(selected):
            attempt = index
            while True:
                row = collection.environment.build_row(*cell,attempt,stream=f'v1011c_dualcam_{role}_seed3103',master_seed=master,role_index=index)
                row.update(schema_version='pact_place_v1011c_paired_eval_v1',role=role,candidate_index=index,
                    checkpoint_seed=3103,episode_id=hashlib.sha256(f'dualcam:{role}:{master}:3103:{index}:{attempt}'.encode()).hexdigest(),
                    max_sampling_retries=collection.MAX_SAMPLING_RETRIES,hazard_present=True)
                candidate = {int(row['task_seed_u32'])}
                candidate.update(retry_seed(row,i)['seed_u32'] for i in range(1,row['max_sampling_retries']+1))
                if len(candidate) == row['max_sampling_retries']+1 and not candidate & (historical|used):
                    break
                attempt += 1000
            used.update(candidate)
            row.pop('row_sha256',None)
            row['row_sha256'] = digest(row)
            (smoke if role == 'smoke' else rows).append(row)
    doc = {**empty_authorization(),'schema_version':'pact_place_v1011c_paired_eval_v1',
        'environment_version':collection.ENVIRONMENT_VERSION,'sampler_class':collection.SAMPLER_CLASS,
        'sampler_module':'molmo_spaces.tasks.enclosure_reach','molmospaces_branch':'experiment/pact-vs-act-remediation-v2',
        'training_seeds':[3103],'training_chunk':100,'num_queries':100,'task_horizon':900,
        'paired':True,'end_on_success':False,'action_noise_enabled':False,'camera_names':CAMERAS,
        'camera_system':'FrankaSkinHybridCameraSystem','action_alignment':ALIGNMENT,
        'master_seeds':{'final':EVAL_MASTER_SEED,'development':DEV_MASTER_SEED,'smoke':SMOKE_MASTER_SEED,'retry':RETRY_MASTER_SEED},
        'sensor_names':list(CANONICAL_SENSOR_NAMES),'sensor_order_sha256':SENSOR_ORDER_SHA256,'encoder_sha256':ENCODER_SHA256,
        'rows':rows,'smoke':{'rows':smoke,'performance_gate':False},
        'role_counts':{'final':50,'development':12,'smoke':4},
        'seed_audit':{'historical_unique':len(historical),'reserved_initial_and_retry':len(used),'disjoint':not bool(used&historical)},
        'pairing_rule':{'non_rgb':'exact bytes, dtype, shape, seed and row',
            'both_rgb_max_abs_delta_u8':1,'both_rgb_max_changed_fraction':0.001},
        'checkpoint_selection':read(WORK/'pilot_plan.json')['checkpoint_selection'],
        'balance':{role:{k:dict(collections.Counter(r[k] for r in rows+smoke if r['role']==role))
                         for k in ('cell','family_id','intrusion_side','pose_id')} for role,_,_ in plan}}
    doc['manifest_sha256'] = digest(doc)
    freeze(EVAL/'eval_manifest.json',doc)
    load_eval_manifest(EVAL/'eval_manifest.json')
    print(json.dumps({'roles':doc['role_counts'],'seed_audit':doc['seed_audit'],'balance':doc['balance']}),flush=True)


if __name__ == '__main__':
    main()
