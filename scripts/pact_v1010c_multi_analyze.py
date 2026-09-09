"""Independent 450-trajectory comparison of ACT, fine-tuned and frozen PACT."""
import collections
import csv
import hashlib
import importlib.util
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
W=ROOT/'diagnostics_output/pact_place_v1010_wrist288_s3_v1'
PREVIOUS=ROOT/'diagnostics_output/pact_place_v1010c_readout_s3103'
MULTI=ROOT/'diagnostics_output/pact_place_v1010c_readout_s3_v1'
METHODS=('ACT','PACT-Finetune','PACT-Frozen')


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def main():
    from pact_v1010b_metrics import recompute
    from pact_wrist288_common import digest
    raw_reader=module(PREVIOUS/'root_review/final_review.py','independent_readout_reader')
    pairing=module(ROOT/'scripts/pact_v1010b_pairing.py','multiseed_pairing')
    def inside(path):
        path=Path(path).resolve();assert path.is_relative_to(MULTI);return path
    pairing.inside=inside
    previous=read(PREVIOUS/'comparison.json')
    saved=[]
    protected={}
    for seed in (3103,3104,3105):
        physical={r['episode_id']:r for r in read(W/f'manifests/final_{seed}.json')['rows']}
        if seed==3103:
            rows=[]
            jobs=read(W/f'evaluation/final_{seed}_full50_h100_schedule.json')
            for job in jobs:
                if job['schedule']['arm']=='ACT':
                    row=recompute(job['directory']);row['method']='ACT';rows.append(row)
            for method,key in (('PACT-Frozen','frozen_rows'),('PACT-Finetune','readout_rows')):
                rows += [dict(r,method=method) for r in previous[key]]
            contract=read(PREVIOUS/'contract.json')
            checkpoint_root=PREVIOUS
        else:
            c=MULTI/f'seed{seed}'
            assert read(c/'run_status.json')['status']=='COMPLETED_COMPARISON'
            rows=read(c/'baseline_metrics.json')['rows']
            rows += read(c/'comparison.json')['readout_rows']
            contract=read(c/'contract.json');checkpoint_root=c
            for path,expected in read(c/'baseline_hashes.json').items():
                assert sha(path)==expected,path
                protected[path]=expected
            records=[json.loads(x) for x in (c/'valid_ledger.jsonl').read_text().splitlines()]
            assert len(records)==52 and len({r['id'] for r in records})==52
            for record in records:
                receipt=read(record['receipt'])
                assert receipt['observed_by_parent'] and receipt['returncode']==0
                assert sha(record['receipt'])==record['receipt_sha256']
                assert sha(record['result_path'])==record['result_sha256']
        for key in ('code_hashes','input_hashes'):
            for path,expected in contract[key].items():
                assert sha(ROOT/path)==expected,path
        pair=read(checkpoint_root/'checkpoint/checkpoint_pairs.json')['policy_last.ckpt']
        completed=read(checkpoint_root/'checkpoint/completed.json')
        assert pair['global_step']==completed['global_step']==60000
        assert completed['strict_pair_reload_exact'] and completed['gradient_check']['stem_and_transformer_nonzero']
        assert sha(checkpoint_root/'checkpoint/policy_last.ckpt')==pair['policy_sha256']
        assert sha(checkpoint_root/'checkpoint/prox_encoder.pt')==pair['encoder_sha256']
        assert len(rows)==150
        for row in rows:
            assert row['training_seed']==seed
            row['physical_row_digest']=digest(physical[row['scene_id']])
            directory=Path(row['directory']);result=read(directory/'result.json')
            assert sha(directory/'trajectory.h5')==result['trajectory_retention']['full_h5_sha256']
            if row['method']=='PACT-Finetune':
                assert result['checkpoint_sha256']==pair['policy_sha256']
                assert result['surface_encoder_sha256']==pair['encoder_sha256']
                info=result['policy_info']
                assert info['control_steps']==info['proximity_projection_calls']==info['consecutive_proximity_history_frames']==900
                assert info['proximity_feature_dim']==128 and info['averaging_history']==100
            else:
                assert read(directory/'exit_receipt.json')['returncode']==0
            saved.append(row)
    assert len(saved)==450
    indexed={(r['training_seed'],r['scene_id'],r['method']):r for r in saved}
    assert len(indexed)==450
    scenes=sorted({(r['training_seed'],r['scene_id']) for r in saved})
    assert len(scenes)==150
    for seed,scene in scenes:
        group=[indexed[seed,scene,m] for m in METHODS]
        target=MULTI/'pairings'/f'{seed}_{scene}.json'
        if target.exists():assert read(target)['passed']
        else:pairing.audit_initial_group(group,target)
    independent=[]
    for i,row in enumerate(saved):
        checked=raw_reader.read_raw(row);checked['method']=row['method'];checked['directory']=row['directory']
        independent.append(checked)
        if (i+1)%25==0:print(json.dumps({'verified_raw_trajectories':i+1,'total':450}),flush=True)
    summaries={}
    for block in ('pooled',3103,3104,3105):
        summaries[str(block)]={method:raw_reader.summary([r for r in independent
            if r['method']==method and (block=='pooled' or r['training_seed']==block)]) for method in METHODS}
    paired={}
    for a,b in itertools.combinations(METHODS,2):
        paired[f'{a} versus {b}']={}
        for block in ('pooled',3103,3104,3105):
            selected=[s for s in scenes if block=='pooled' or s[0]==block]
            paired[f'{a} versus {b}'][str(block)]={key:{
                'first_only':sum(indexed[*s,a][key] and not indexed[*s,b][key] for s in selected),
                'second_only':sum(indexed[*s,b][key] and not indexed[*s,a][key] for s in selected)}
                for key in ('task_success','collision_free_task_success','pickup_failure')}
    doc={'utc':datetime.now(timezone.utc).isoformat(),'independent_raw_trajectories':450,
        'three_way_matched_scene_groups':150,'all_initial_pairings_passed':True,
        'all_protected_hashes_passed':True,'all_raw_action_horizons_and_endpoints_passed':True,
        'summaries':summaries,'paired':paired,'rows':saved,'independent_rows':independent,
        'analysis_sha256':sha(__file__),'independent_reader_sha256':sha(PREVIOUS/'root_review/final_review.py'),
        'scope':'Three training seeds, their original 50-scene blocks; previously exposed regression scenes; full main method'}
    write(MULTI/'comparison.json',doc)
    csvrows=[]
    for seed,scene in scenes:
        row={'training_seed':seed,'scene_id':scene}
        for method in METHODS:
            r=indexed[seed,scene,method]
            for key in ('task_success','collision_free_task_success','pickup_failure','failure_stage','union_frames'):
                row[f'{method}_{key}']=r[key]
        csvrows.append(row)
    with (MULTI/'paired_scene_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    text=['# ACT versus fine-tuned and frozen PACT: three seeds','',
        'Completed fresh fine-tuned PACT training for seeds 3104 and 3105 at 60,000 committed updates each and all 100 new evaluations. Together with completed seed3103, all methods have 150 matched evaluations. Independent raw reading verified all 450 trajectories and all 150 three-way initial-state groups.','',
        '| Seed | Method | Task success | Collision-free success | Pickup failures | Hazard/clutter frames |',
        '|---|---|---:|---:|---:|---:|']
    for block in ('3103','3104','3105','pooled'):
        for method in METHODS:
            s=summaries[block][method];n=s['n']
            text.append(f"| {block} | {method} | {s['success']}/{n} ({100*s['success']/n:.1f}%) | {s['cfts']}/{n} ({100*s['cfts']/n:.1f}%) | {s['pickup']}/{n} | {s['union_frames']:,} |")
    text += ['', '## Contact classes', '', '| Method | Hazard bar | Clutter | Grasp target | Avoidance |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        s=summaries['pooled'][method];f=s['contact_frames']
        text.append(f"| {method} | {f['hazard_bar']:,} | {f['clutter']:,} | {f['grasp_target']:,} | {s['avoidance_percent']:.3f}% |")
    text += ['', 'The comparison uses main’s full fine-tuning, 128-D CLS readout and minimum-pooling method, initialized from the same original pretrained encoder. It does not isolate unfreezing alone. Each seed uses its original scene block, so per-seed differences reflect both trained-policy and scene-block variation. These exposed regression scenes do not establish performance on a fresh test set. All original baselines and the seed3103 run were preserved.', '',
        '[Full metrics, verification and paired changes](comparison.json) · [Per-scene CSV](paired_scene_comparison.csv) · [Execution plan](../../docs/PACT_PLACE_V1010C_THREE_SEED_PLAN.md)', '']
    (MULTI/'FINAL_REVIEW.md').write_text('\n'.join(text))
    print(json.dumps({'status':'VERIFIED_COMPARISON','summaries':summaries,'paired':paired}),flush=True)


if __name__=='__main__':main()
