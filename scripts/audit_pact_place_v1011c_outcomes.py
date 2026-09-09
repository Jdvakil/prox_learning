"""Post-evaluation audit: read historical artifacts, write only a new audit."""
from collections import Counter, defaultdict
import json
import math
import statistics
from pathlib import Path
import h5py
import numpy as np
from pact_place_v1011c_experiment import ROOT, empty_authorization, freeze, read, sha256_file

OUT = ROOT/'diagnostics_output/pact_place_v1011c_post_eval_audit'
SOURCES = {
    'V10.9': 'pact_place_v109_eval',
    'V10.10': 'pact_place_v1010_eval_infra_repair_01',
    'V10.11c': 'pact_place_v1011c_eval',
}


def decode(row):
    return json.loads(row.tobytes().split(b'\0',1)[0])


def inspect_rollout(record):
    directory = Path(record['row_dir'] if 'row_dir' in record else ROOT/record['directory'])
    result = read(directory/'result.json')
    assert record['status']=='complete' and record['returncode']==0 and result['status']=='complete'
    for key in ('arm','episode_id','checkpoint_seed','checkpoint_sha256'):
        assert record[key]==result[key]
    info=result['policy_info'];audit=result['contact_audit']
    assert info['num_queries']==100 and info['control_steps']==900
    totals=audit['contact_class_totals'];frames=audit['frames_with_contact']
    assert all(c in totals and c in frames for c in ('hazard_bar','other_environment','clutter','mounted_fixture'))
    collision=any(totals[c]>0 for c in ('hazard_bar','other_environment','clutter','mounted_fixture'))
    assert result['collision_free_task_success']==(result['task_success'] and not collision)
    row={'arm':result['arm'],'seed':result['checkpoint_seed'],'episode_id':result['episode_id'],
         'directory':str(directory.relative_to(ROOT)),'task_success':result['task_success'],
         'collision_free_success':result['collision_free_task_success'],'collision_episode':collision,
         'hazard_episode':frames['hazard_bar']>0,'hazard_frames':frames['hazard_bar'],
         'clutter_episode':frames['clutter']>0,'clutter_frames':frames['clutter'],
         'result_sha256':sha256_file(directory/'result.json')}
    path=directory/'trajectory.h5'
    row['raw_trajectory_available']=path.exists()
    if not path.exists():
        return row
    with h5py.File(path,'r') as h:
        success=h['traj_0/success'][()]
        assert success.shape==(901,) and bool(success[-1])==row['task_success']
        task=[decode(x) for x in h['traj_0/obs/extra/task_info'][()]]
        grasp=[decode(x) for x in h['traj_0/obs/extra/grasp_state_pickup_obj'][()]]
        assert len(task)==len(grasp)==901
        assert all(set(('success','supported_by_receptacle','robot_contact','position_error'))<=set(x) for x in task)
        assert all('gripper' in x and set(('held','touching'))<=set(x['gripper']) for x in grasp)
        assert np.array_equal(success,np.array([x['success'] for x in task]))
        touched=np.array([x['gripper']['touching'] for x in grasp],dtype=bool)
        held=np.array([x['gripper']['held'] for x in grasp],dtype=bool)
        supported=np.array([x['supported_by_receptacle'] for x in task],dtype=bool)
        initial_positions=h['traj_0/obs/extra/obj_start'][()][:,:3]
        # ObjectStartPoseSensor is a cached initial pose, NOT object motion.
        # TCPPoseSensor is in robot coordinates, not this world coordinate frame.
        assert np.array_equal(initial_positions,np.broadcast_to(initial_positions[0],initial_positions.shape))
        scene=json.loads(h['traj_0/obs_scene'][()])
        params=scene['scene_params']
        row.update(policy_dt_ms=scene['policy_dt_ms'],
            environment_version=params['pact_place_environment_version'],
            cell=params['cell'],family=params['pact_clutter_layout']['family_id'] if 'family_id' in params['pact_clutter_layout'] else None,
            pose=params['pact_v106_pose_id'],side=params['pact_intrusion_side'],
            active_clutter=len(params['pact_clutter_active_body_names']),
            primitive_heights=params.get('pact_v1011c_primitive_heights_m'),
            task_success_criterion=params['task_success_criterion'],
            any_task_success=bool(success.any()),transient_success_lost=bool(success.any() and not success[-1]),
            success_at_635=bool(success[635]),
            any_target_touch=bool(touched.any()),any_target_held=bool(held.any()),
            any_receptacle_support=bool(supported.any()),
            initial_target_xyz_m=initial_positions[0].astype(float).tolist(),
            min_place_position_error_m=min(x['position_error'] for x in task),
            first_success_step=int(np.flatnonzero(success)[0]) if success.any() else None)
        if success[-1]: stage='success'
        elif supported.any(): stage='supported_but_not_success_at_end'
        elif held.any(): stage='held_but_never_supported'
        elif touched.any(): stage='touched_but_never_held_or_supported'
        else: stage='never_touched_held_or_supported'
        row['failure_stage']=stage
    return row


def aggregate(rows):
    keys=('task_success','collision_free_success','collision_episode','hazard_episode','hazard_frames',
          'clutter_episode','clutter_frames')
    out={key:sum(r[key] for r in rows) for key in keys};out['n']=len(rows)
    available=[r for r in rows if r['raw_trajectory_available']]
    out['raw_trajectories_available']=len(available)
    if available:
        for key in ('any_task_success','transient_success_lost','success_at_635','any_target_touch',
                    'any_target_held','any_receptacle_support'):
            out[key]=sum(r[key] for r in available)
        out['failure_stages']=dict(Counter(r['failure_stage'] for r in available))
        out['failed_without_hazard']=sum(not r['task_success'] and not r['hazard_episode'] for r in available)
        out['failed_without_any_disallowed_collision']=sum(not r['task_success'] and not r['collision_episode'] for r in available)
    return out


def dataset(version):
    base=ROOT/f'diagnostics_output/pact_place_{version}_train_eval'
    source=read(base/'source_manifest.json');converted=read(base/'conversion_manifest.json');split=read(base/'split_manifest.json')
    ledger_path=ROOT/source['ledger_path']
    ledger=[json.loads(x) for x in ledger_path.read_text().splitlines()]
    train_ids={r['episode_id'] for r in split['episodes'] if r['split']=='train'}
    validation_ids={r['episode_id'] for r in split['episodes'] if r['split']=='validation'}
    assert not train_ids & validation_ids
    data={'episodes':len(converted['episodes']),'train':len(train_ids),'validation':len(validation_ids),
          'train_timesteps':sum(r['timesteps'] for r in converted['episodes'] if r['episode_id'] in train_ids),
          'total_converted_timesteps':sum(r['timesteps'] for r in converted['episodes']),
          'train_episodes_per_cell':dict(Counter(r['cell'] for r in split['episodes'] if r['split']=='train')),
          'collection_attempts':len(ledger),'collection_complete':sum(r['status']=='complete' for r in ledger),
          'collection_task_success':sum(r['task_success'] for r in ledger),
          'collection_strict_clean':sum(r['accepted'] for r in ledger),
          'collection_defect_families':dict(Counter(k.split('=')[0] for r in ledger for k in r['defects'])),
          'max_demo_steps':max(r['timesteps'] for r in converted['episodes']),
          'median_demo_steps':statistics.median(r['timesteps'] for r in converted['episodes'])}
    if version=='v1010':
        models=read(base/'training_verification.json')['arms']
        directories=[Path(m['checkpoint_dir']) for m in models.values()]
    else:
        directories=sorted((base/'checkpoints').glob('*_seed*'))
    data['training_epoch_logs']={}
    for directory in directories:
        logs=[json.loads(line) for line in (directory/'epoch_log.jsonl').read_text().splitlines()]
        assert len(logs)==2000 and logs[-1]['global_step']==2000*math.ceil(len(train_ids)/8)
        data['training_epoch_logs'][directory.name]={'epochs':len(logs),'optimizer_steps':logs[-1]['global_step'],
            'best_epoch':logs[-1]['best_epoch'],'best_val_loss':logs[-1]['min_val_loss'],
            'final_train':logs[-1]['train'],'final_val':logs[-1]['val']}
    return data


def main():
    doc={**empty_authorization(),'scope':'Read-only post-hoc audit; no retraining, re-evaluation or scoring change.',
         'versions':{},'datasets':{},'limitations':[
             'Historical V10.9 trajectory H5 files are no longer retained; its endpoint counts can only be re-derived from per-rollout result JSON.',
             'Version comparisons confound environment, training corpus, optimizer-update count, model seeds and held-out instances.',
             'Recorded target held/touching states describe pickup progress, not gripper-command decoder analysis.',
             'Post-hoc timing/funnel thresholds are descriptive diagnostics, not replacements for the registered endpoint.',
             'The initial audit.json draft incorrectly treated obj_start as live object pose and subtracted robot-frame TCP from world-frame coordinates. All derived motion/lift/distance fields from that draft are withdrawn; they are absent here. Raw success/contact/support counts were unaffected.',
             'Continuous target motion and lift are not reconstructed: obj_start is a cached initial pose; object dynamics are not retained in env_states.']}
    for version,folder in SOURCES.items():
        base=ROOT/'diagnostics_output'/folder;run=read(base/'full_run.json')
        records=run['results'];assert len(records)==(300 if version=='V10.11c' else 80)
        assert len({r['rollout_id'] for r in records})==len(records)
        rows=[]
        for record in records: rows.append(inspect_rollout(record))
        arms={arm:aggregate([r for r in rows if r['arm']==arm]) for arm in ('ACT','PACT')}
        seeds={str(seed):{arm:aggregate([r for r in rows if r['arm']==arm and r['seed']==seed]) for arm in ('ACT','PACT')}
               for seed in sorted({r['seed'] for r in rows})}
        strata={key:{str(value):{arm:aggregate([r for r in rows if r['arm']==arm and r.get(key)==value]) for arm in ('ACT','PACT')}
                     for value in sorted({r[key] for r in rows if key in r})} for key in ('pose','side')}
        doc['versions'][version]={'arms':arms,'seeds':seeds,'strata':strata,'rows':rows,
                                 'full_run_sha256':sha256_file(base/'full_run.json')}
        print(version,json.dumps(arms),flush=True)
    for version in ('v1010','v1011c'):
        doc['datasets'][version]=dataset(version)
        print(version,'dataset',json.dumps({k:v for k,v in doc['datasets'][version].items() if k!='train_episodes_per_cell'}),flush=True)
    freeze(OUT/'audit_v2.json',doc)
    print(f'Created {OUT}/audit_v2.json',flush=True)


if __name__=='__main__':
    main()
