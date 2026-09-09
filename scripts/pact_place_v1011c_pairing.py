"""Exact physical pairing with a disclosed, pre-full-eval RGB tolerance.

The first smoke failed strict image-byte equality: three pairs differed by one
uint8 intensity level in 15, 24 and 48 of 658944 channel values. Every non-RGB
observation was byte-identical. Raw observations and the failed receipt remain
unchanged. This rule is frozen before any scientific rollout is run.
"""
import json
import h5py
import numpy as np
from pact_place_v1011c_experiment import ROOT, EVAL, WORK, read, freeze, sha256_file, empty_authorization

RGB_MAX_ABS_DELTA = 1
RGB_MAX_CHANGED_FRACTION = 0.001


def check_pair(first,second):
    assert first['episode_id'] == second['episode_id']
    assert first['task_seed'] == second['task_seed']
    assert first['row_sha256'] == second['row_sha256']
    info={'episode_id':first['episode_id'],'exact_non_rgb_datasets':0,'rgb':{}}
    with h5py.File(ROOT/first['directory']/'initial_observation.h5','r') as a, \
         h5py.File(ROOT/second['directory']/'initial_observation.h5','r') as b:
        na,nb=[],[]
        a.visititems(lambda n,o:na.append(n) if isinstance(o,h5py.Dataset) else None)
        b.visititems(lambda n,o:nb.append(n) if isinstance(o,h5py.Dataset) else None)
        assert na == nb and na
        for name in na:
            assert a[name].dtype == b[name].dtype and a[name].shape == b[name].shape, name
            av,bv=np.asarray(a[name][()]),np.asarray(b[name][()])
            if name == 'observation/wrist_camera':
                assert av.dtype == np.uint8 and av.ndim==3 and av.shape[-1]==3
                delta=np.abs(av.astype(np.int16)-bv.astype(np.int16))
                changed=int(np.count_nonzero(delta));maximum=int(delta.max())
                info['rgb']={'shape':list(av.shape),'channel_values':int(av.size),
                    'changed_channel_values':changed,'changed_fraction':changed/av.size,
                    'max_abs_delta_u8':maximum,'exact':changed==0}
                assert maximum<=RGB_MAX_ABS_DELTA and changed/av.size<=RGB_MAX_CHANGED_FRACTION, info
            else:
                assert av.tobytes()==bv.tobytes(),f'initial non-RGB mismatch: {name}'
                info['exact_non_rgb_datasets']+=1
        assert info['rgb'] and info['exact_non_rgb_datasets']>0
        info['total_datasets']=len(na)
    return info


def reconcile_smoke():
    from pact_place_v1011c_metrics import metrics
    old=read(EVAL/'smoke_run.json')
    assert not old['infrastructure_healthy']
    assert old['rollouts_complete']==old['rollouts_expected']==8
    assert old['pairing_errors'] and all('initial observation mismatch: observation/wrist_camera' in r['error'] for r in old['pairing_errors'])
    assert not (EVAL/'full_ledger.jsonl').exists(), 'Pairing amendment must precede scientific evaluation.'
    ledger=[json.loads(x) for x in (EVAL/'smoke_ledger.jsonl').read_text().splitlines()]
    assert len(ledger)==8
    pairs={}
    for record in ledger:
        assert record['status']=='complete' and record['returncode']==0
        row=metrics(ROOT/record['directory'],record)
        pairs.setdefault(row['episode_id'],{})[row['arm']]=row
    checks=[check_pair(pair['ACT'],pair['PACT']) for pair in pairs.values()]
    assert len(checks)==4
    doc={**empty_authorization(),'infrastructure_healthy':True,'rollouts_complete':8,
        'original_smoke_status':'failed_strict_rgb_byte_equality','original_smoke_sha256':sha256_file(EVAL/'smoke_run.json'),
        'original_smoke_ledger_sha256':sha256_file(EVAL/'smoke_ledger.jsonl'),
        'models_or_rollouts_rerun':False,'raw_observations_modified':False,
        'scientific_rollouts_started':False,'pair_checks':checks,
        'pairing_rule':{'non_rgb':'exact bytes, shape, dtype, selected seed and row identity',
            'wrist_rgb_max_abs_delta_u8':RGB_MAX_ABS_DELTA,'wrist_rgb_max_changed_channel_fraction':RGB_MAX_CHANGED_FRACTION},
        'reason':'Sparse one-level RGB differences with exact physical observations. Cause is not conclusively established; tolerance and actual differences are reported explicitly.',
        'performance_used_for_gate':False,'pairing_implementation_sha256':sha256_file(__import__('pathlib').Path(__file__))}
    freeze(EVAL/'smoke_reconciliation.json',doc)
    print(json.dumps(doc,indent=2),flush=True)


if __name__=='__main__':
    reconcile_smoke()
