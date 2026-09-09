"""Read-only paper-readiness inventory and historical endpoint reconstruction."""
from collections import Counter
from pathlib import Path
import json
import sys

from pact_place_v1011c_experiment import ROOT, read, freeze, empty_authorization, sha256_file
import h5py
import numpy as np
import cv2
from analyze_pact_contact_endpoint import validate_result

OUT=ROOT/'diagnostics_output/pact_paper_readiness_audit'


def decode(row):
    return json.loads(row.tobytes().split(b'\0',1)[0])


def summarize(rows):
    return {'n':len(rows),**{k:sum(r[k] for r in rows) for k in (
        'task_success','collision_free_success','hazard_episode','hazard_frames',
        'other_environment_episode','other_environment_frames')}}


def inventory_models():
    doc={**empty_authorization(),'historical_checkpoints':[],'training_budgets':[],
         'historical_encoded_dataset':{}}
    provenance=read(ROOT/'diagnostics_output/pact_contact_endpoint/provenance.json')
    for row in provenance['checkpoint_records']:
        path=Path(row['path'])
        assert path.is_file() and sha256_file(path)==row['sha256']
        doc['historical_checkpoints'].append({**row,'hash_verified':True})
        run=read(path.parent/'run_manifest.json')
        epochs=[json.loads(x) for x in (path.parent/'epoch_log.jsonl').read_text().splitlines()]
        assert len(epochs)==run['num_epochs']==2000 and epochs[-1]['epoch']==1999
        doc['training_budgets'].append({'experiment':'original_pick_corridor','seed':run['seed'],
            'arm':'PACT' if run['proximity_consumed'] else 'ACT','global_step':epochs[-1]['global_step'],
            'train_episodes':run['train_episodes'],'val_episodes':run['val_episodes'],
            'chunk_size':run['chunk_size'],'epoch_log':str(path.parent/'epoch_log.jsonl')})
    for version in ('v1010','v1011c'):
        verification=read(ROOT/f'diagnostics_output/pact_place_{version}_train_eval/training_verification.json')
        entries=verification['arms'] if version=='v1010' else {k:v['verification'] for k,v in verification['models'].items()}
        for name, row in entries.items():
            path=Path(row['checkpoint_dir'])
            run=read(path/'run_manifest.json')
            epochs=[json.loads(x) for x in (path/'epoch_log.jsonl').read_text().splitlines()]
            assert len(epochs)==run['num_epochs']==2000 and epochs[-1]['epoch']==1999
            doc['training_budgets'].append({'experiment':version,'seed':run['seed'],
                'arm':'PACT' if run['proximity_consumed'] else 'ACT','global_step':epochs[-1]['global_step'],
                'train_episodes':run['train_episodes'],'val_episodes':run['val_episodes'],
                'chunk_size':run['chunk_size'],'epoch_log':str(path/'epoch_log.jsonl')})
    conversion=read(Path('/root/pact_frontend_screen_artifacts/manifests/encoded_conversion_v1.json'))
    base=Path(conversion['destination']);total=0
    for row in conversion['episodes']:
        path=base/row['act_file']
        assert path.is_file() and sha256_file(path)==row['act_h5_sha256']
        with h5py.File(path,'r') as h:
            assert len(h['action'])==len(h['observations/qpos'])==len(h['observations/images/wrist_camera'])
            assert h['observations/proximity_embeddings'].shape[1:]==(40,32)
        total+=path.stat().st_size
    doc['historical_encoded_dataset']={'directory':str(base),'episodes_verified':len(conversion['episodes']),
        'bytes':total,'note':'ACT could ignore proximity in these files, but any new manifest binding must be explicit; old missing ACT directory is not silently substituted.'}
    assert len(conversion['episodes'])==255
    freeze(OUT/'model_inventory.json',doc)
    print(json.dumps({'verified_checkpoints':len(doc['historical_checkpoints']),
        'training_budgets':[{**r,'train_episodes':len(r['train_episodes']),
                            'val_episodes':len(r['val_episodes'])} for r in doc['training_budgets']],
        'historical_encoded_dataset':doc['historical_encoded_dataset']},indent=2),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    doc={**empty_authorization(),'historical_contact':{},'v1010_dataset':{},'limitations':[
        'No training, collection, changed evaluation or external download performed.',
        'Historical contact endpoints reconstructed from per-rollout JSON and driver receipts; most full historical trajectories were deleted.',
        'Dataset visual examples are deterministic side/pose representatives, not a random success-rate estimate.'
    ]}
    schedule=read(ROOT/'diagnostics_output/pact_contact_endpoint/schedule.json')
    base=Path('/root/pact_contact_endpoint_artifacts/evaluation_v1')
    rows=[];raw=0
    for row in schedule['rows']:
        folder=base/row['output_relpath'];result=read(folder/'result.json');driver=read(folder/'driver_result.json')
        validate_result(result,row)
        assert driver['status']=='complete' and driver['returncode']==0
        assert driver['rollout_id']==row['rollout_id'] and driver['schedule_row_sha256']==row['schedule_row_sha256']
        raw+=bool(result.get('trajectory_path') and Path(result['trajectory_path']).exists())
        c=result['contact_audit'];totals=c['contact_class_totals'];frames=c['frames_with_contact']
        x={'arm':row['arm'],'seed':row['checkpoint_seed'],'episode_id':row['instance_episode_id'],
           'task_success':bool(result['task_success']),
           'collision_free_success':bool(result['task_success'] and totals['hazard_bar']==0 and totals['other_environment']==0),
           'hazard_episode':totals['hazard_bar']>0,'hazard_frames':frames['hazard_bar'],
           'other_environment_episode':totals['other_environment']>0,'other_environment_frames':frames['other_environment'],
           'result_sha256':sha256_file(folder/'result.json'),'directory':str(folder)}
        assert x['collision_free_success']==result['collision_free_task_success']
        rows.append(x)
    assert len(rows)==1200 and len({(r['arm'],r['seed'],r['episode_id']) for r in rows})==1200
    by_seed={str(s):{a:summarize([r for r in rows if r['seed']==s and r['arm']==a]) for a in schedule['arms']} for s in schedule['checkpoint_seeds']}
    pooled={a:summarize([r for r in rows if r['arm']==a]) for a in schedule['arms']}
    doc['historical_contact']={'by_seed':by_seed,'pooled':pooled,'rows':rows,'raw_trajectories_present':raw,
        'schedule_sha256':sha256_file(ROOT/'diagnostics_output/pact_contact_endpoint/schedule.json')}
    print('Historical contact:',json.dumps({'by_seed':by_seed,'pooled':pooled,'raw_trajectories_present':raw}),flush=True)
    work=ROOT/'diagnostics_output/pact_place_v1010_train_eval'
    source=read(work/'source_manifest.json');conversion=read(work/'conversion_manifest_encoded.json');split=read(work/'split_manifest.json')
    converted={r['act_episode_index']:r for r in conversion['episodes']}
    dataset=[]
    for row in source['rows']:
        path=ROOT/row['trajectory_h5'];entry=converted[row['act_episode_index']]
        act=ROOT/'assets/act_style_data/pact_place_v1010_144'/entry['act_file']
        assert sha256_file(path)==row['trajectory_h5_sha256']
        assert sha256_file(act)==entry['act_h5_sha256']
        videos=list(path.parent.glob('episode_*_wrist_camera.mp4'))
        assert len(videos)==1
        with h5py.File(path,'r') as h:
            assert bool(h['traj_0/success'][-1]) and len(h['traj_0/success'])==entry['raw_timesteps']
            touch=np.array([decode(r)['gripper']['touching'] for r in h['traj_0/obs/extra/grasp_state_pickup_obj'][()]])
            seen=h['traj_0/obs/extra/object_image_points/pickup_obj/wrist_camera/num_points'][()].reshape(-1)>0
            scene=json.loads(h['traj_0/obs_scene'][()])
            assert scene['scene_params']['pact_place_environment_version']=='pact_place_corridor_v10_10_four_object'
            assert scene['task_type']=='pick_and_place'
        with h5py.File(act,'r') as h:
            t=len(h['action'])
            assert h['observations/images/wrist_camera'].shape==(t,240,320,3)
            assert h['observations/proximity_embeddings'].shape==(t,40,32)
            assert np.isfinite(h['observations/proximity_embeddings'][()]).all()
        dataset.append({**{k:row[k] for k in ('attempt_id','act_episode_index','cell','pose_id','intrusion_side','family_id')},
            'source_path':str(path),'converted_path':str(act),'video_path':str(videos[0]),
            'converted_steps':t,'first_touch_step':int(np.flatnonzero(touch)[0]),
            'first_segmented_step':int(np.flatnonzero(seen)[0])})
        if len(dataset)%24==0: print('Verified source/converted episodes',len(dataset),flush=True)
    assert len(dataset)==144
    train=[r for r in split['episodes'] if r['split']=='train'];val=[r for r in split['episodes'] if r['split']=='validation']
    doc['v1010_dataset']={'episodes':len(dataset),'train':len(train),'validation':len(val),
        'source_and_converted_hashes_verified':len(dataset)*2,'cell_counts':dict(Counter(r['cell'] for r in dataset)),
        'train_cell_counts':dict(Counter(r['cell'] for r in train)),
        'all_wrist_videos_present':True,'rows':dataset}
    samples=[]
    for pose in ('neg5','center','pos5'):
        for side in ('left','right'):
            row=min((r for r in dataset if r['pose_id']==pose and r['intrusion_side']==side),key=lambda r:r['attempt_id'])
            with h5py.File(row['converted_path'],'r') as h:
                t=min(row['first_touch_step'],row['converted_steps']-1)
                image_path=OUT/f'v1010_{pose}_{side}_{row["attempt_id"][:16]}_first_touch.png'
                if not image_path.exists():
                    assert cv2.imwrite(str(image_path),cv2.cvtColor(h['observations/images/wrist_camera'][t],cv2.COLOR_RGB2BGR))
            samples.append({'attempt_id':row['attempt_id'],'cell':row['cell'],'step':t,'image_path':str(image_path)})
    doc['v1010_dataset']['visual_samples']=samples
    # A copy-current-arm baseline illustrates why low offline error is not proof of a good policy.
    doc['copy_current_arm_offline']={}
    for version in ('v1010','v1011c'):
        b=ROOT/f'diagnostics_output/pact_place_{version}_train_eval';s=read(b/'split_manifest.json');m=read(b/'source_manifest.json')
        by_index={r['act_episode_index']:r for r in m['rows']};errors=[]
        for r in s['episodes']:
            if r['split']!='validation': continue
            i=r['act_episode_index']
            with h5py.File(ROOT/by_index[i]['trajectory_h5'],'r') as h:
                held=np.array([decode(x)['gripper']['touching'] for x in h['traj_0/obs/extra/grasp_state_pickup_obj'][()]])
                stop=min(150,int(np.flatnonzero(held)[0]))
            count=144 if version=='v1010' else 99
            with h5py.File(ROOT/f'assets/act_style_data/pact_place_{version}_{count}/episode_{i}.hdf5','r') as h:
                errors.append(np.abs(h['observations/qpos'][:stop,:7]-h['action'][:stop,:7]))
        x=np.concatenate(errors)
        doc['copy_current_arm_offline'][version]={'frames':len(x),'mean_abs_joint_error_rad':float(x.mean()),
            'warning':'Diagnostic trivial state-copy baseline; not executed as a rollout and not a task-success estimate.'}
    print('Copy-current-arm offline:',json.dumps(doc['copy_current_arm_offline']),flush=True)
    freeze(OUT/'raw_audit.json',doc)
    print('Created paper-readiness raw_audit.json',flush=True)


if __name__=='__main__':
    inventory_models() if '--inventory-models' in sys.argv else main()
