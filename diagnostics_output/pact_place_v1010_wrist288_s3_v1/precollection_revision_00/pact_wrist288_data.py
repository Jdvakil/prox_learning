"""Forward-command conversion, accepted-ledger verification and shared statistics."""
from __future__ import annotations
import argparse
import collections
import pickle
import time
from pact_wrist288_common import *

def decode(blob):
    value=json.loads(bytes(blob).split(b'\0',1)[0] or b'{}')
    if not isinstance(value,dict): raise ValueError('JSON row must be a dictionary')
    return value

def vector(value,gripper_dim):
    import numpy as np
    assert len(value['arm'])==7 and len(value['gripper'])==gripper_dim
    out=np.asarray(value['arm']+value['gripper'],dtype=np.float32)
    assert np.isfinite(out).all()
    assert np.max(np.abs(out[:7]))<7, 'arm components must be joint radians'
    if gripper_dim==1:
        assert out[-1] in (0,255), 'preserve FR3 gripper actuator units'
    else:
        assert np.max(np.abs(out[-2:]))<10, 'finger state components use metres (or m/s for qvel)'
    return out

def forward_controls(group):
    import numpy as np
    commands=group['actions/commanded_action'];joints=group['actions/joint_pos']
    n=len(commands)
    assert len(joints)==n and n>=3
    assert decode(commands[0])=={},'initial command must be dummy'
    indices=[];actions=[];excluded=[];ended=False
    for i in range(1,n):
        command=decode(commands[i]);joint=decode(joints[i])
        if not joint or command.get('done',False):
            assert bool(group['terminated'][i] or group['truncated'][i]),'nonterminal command hole'
            ended=True;excluded.append({'action_row':i,'reason':'terminal_status_only'});continue
        assert not ended,'move command after terminal status'
        action=vector(command,1)
        vector(joint,1)
        assert not bool(group['terminated'][i-1] or group['truncated'][i-1]),'observation already terminal'
        indices.append(i-1);actions.append(action)
    assert indices and indices==list(range(len(indices))), 'valid transitions must form a prefix'
    qpos=np.stack([vector(decode(group['obs/agent/qpos'][i]),2) for i in indices])
    # Joint velocities can exceed joint-angle bounds; validate dimensions and finiteness separately.
    velocities=[decode(group['obs/agent/qvel'][i]) for i in indices]
    assert all(len(v['arm'])==7 and len(v['gripper'])==2 for v in velocities)
    qvel=np.asarray([v['arm']+v['gripper'] for v in velocities],dtype=np.float32)
    assert np.isfinite(qvel).all()
    return np.stack(actions),qpos,qvel,{'raw_timesteps':n,'observation_indices':indices,
        'command_indices':[i+1 for i in indices],'excluded':excluded,'initial_dummy_excluded':True,
        'action_units':['radian']*7+['FR3 actuator 0/255'],'gripper_threshold':127.5}

def verify_ledger(records):
    counts=collections.Counter(r['cell'] for r in records if r.get('accepted'))
    assert len({r['attempt_id'] for r in records})==len(records)
    expected={cell_key(*c):12 for c in cells()}
    assert dict(counts)==expected,(counts,expected)
    for row in records:
        assert row['returncode']==0 or not row['accepted']
        assert row['attempt_id']==row['row']['attempt_id']
        assert row['row_sha256']==row['row']['row_sha256']
    return [r for r in records if r['accepted']]

def split_rows(rows,registry):
    grouped=collections.defaultdict(list)
    for row in rows: grouped[row['cell']].append(row)
    assert len(grouped)==24 and all(len(v)==12 for v in grouped.values())
    validation=set()
    for cell,members in grouped.items():
        seed=registry['entries'][f'split:{cell}:0'][0]['derivation']
        ranked=sorted(members,key=lambda r:digest([NAMESPACE,seed,r['attempt_id']]))
        validation.update(r['attempt_id'] for r in ranked[:2])
    return validation

def convert_one(record,index,encoder):
    import h5py
    import numpy as np
    from convert_pact_place_v109_to_act import extract_proximity, _find_wrist_video
    from convert_obstacle_to_act import _video_frames
    from encode_pact_embedding_tokens import encode_episode
    source=ROOT/record['trajectory_h5'];directory=source.parent
    assert sha(source)==record['trajectory_h5_sha256']
    from run_pact_place_v108_collect import validate_trainable,row_defects
    result=read(directory/'result.json')
    assert not row_defects(result) and result['task_success']
    assert result['attempt_id']==record['attempt_id'] and result['row_sha256']==record['row_sha256']
    assert validate_trainable(directory)['passed']
    with h5py.File(source,'r') as h:
        group=h['traj_0'];action,qpos,qvel,audit=forward_controls(group);t=len(action)
        assert audit['raw_timesteps']==result['episode_steps']+1
        assert bool(group['success'][-1])
        proximity,extrinsic,intrinsic=extract_proximity(group,t)
    images=_video_frames(_find_wrist_video(directory),240,320)
    assert len(images)==audit['raw_timesteps']
    destination=WORK/f'converted/episode_{index}.hdf5'
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.partial.hdf5')
    assert not destination.exists() and not temporary.exists()
    with h5py.File(temporary,'x') as h:
        h.attrs.update(sim=True,pact_episode_id=record['attempt_id'],pact_row_sha256=record['row_sha256'],
            pact_sensor_order_sha256=SENSOR_ORDER_SHA256,supervision='observation[t] -> commanded_action[t+1]',
            experiment_namespace=NAMESPACE)
        for name,value in [('action',action),('observations/qpos',qpos),('observations/qvel',qvel),
            ('observations/images/wrist_camera',images[:t]),('observations/proximity',proximity),
            ('observations/proximity_extrinsic_cv',extrinsic),('observations/proximity_intrinsic_cv',intrinsic)]:
            h.create_dataset(name,data=value,chunks=(1,*value.shape[1:]),compression='gzip',compression_opts=1)
        h.create_dataset('observations/proximity_sensor_names',data=np.asarray(CANONICAL_SENSOR_NAMES,dtype=object),dtype=h5py.string_dtype())
        h.create_dataset('pact_provenance/row',data=json.dumps(record['row'],sort_keys=True).encode())
        h.create_dataset('pact_provenance/alignment',data=json.dumps(audit,sort_keys=True).encode())
    encoded=encode_episode(temporary,model=encoder,device=__import__('torch').device('cuda'),batch_size=1024,checkpoint_sha256=ENCODER_SHA256)
    with h5py.File(temporary,'r') as h:
        assert h['observations/proximity_embeddings'].shape==(t,40,32)
        assert np.isfinite(h['observations/proximity_embeddings'][()]).all()
        assert np.array_equal(h['action'][()],action)
        assert np.array_equal(h['observations/qpos'][()],qpos)
    os.replace(temporary,destination)
    return {'act_episode_index':index,'act_file':destination.name,'episode_id':record['attempt_id'],
        'candidate_index':record['row']['attempt_index'],'cell':record['cell'],'timesteps':t,
        'raw_timesteps':audit['raw_timesteps'],'act_file_sha256':sha(destination),'source_h5_sha256':sha(source),
        'wrist_video_sha256':sha(_find_wrist_video(directory)),'alignment_audit':audit,'embedding':encoded}

def main():
    os.environ.update(environment());pin_torch();config=check_bindings()
    import torch
    from surface_proximity_encoder import load_frozen_surface_embedding_encoder
    records=verify_ledger(lines(WORK/'collection/ledger.jsonl'))
    records.sort(key=lambda r:(r['cell'],r['row']['attempt_index'],r['attempt_id']))
    registry=read(WORK/'seed_registry.json');validation=split_rows(records,registry)
    encoder,_=load_frozen_surface_embedding_encoder(ENCODER_PATH,map_location='cuda');encoder=encoder.cuda().eval()
    converted=lines(WORK/'conversion_ledger.jsonl')
    for i,record in enumerate(records):
        if i<len(converted):
            ep=converted[i]
            assert ep['episode_id']==record['attempt_id'] and sha(WORK/'converted'/ep['act_file'])==ep['act_file_sha256']
            continue
        if (WORK/'PAUSE.json').exists(): raise RuntimeError('conversion paused before next episode')
        with torch.inference_mode(): ep=convert_one(record,i,encoder)
        append(WORK/'conversion_ledger.jsonl',ep);converted.append(ep)
        print(f'converted {len(converted)}/288 T={ep["timesteps"]}',flush=True)
    tree=hashlib.sha256()
    for ep in converted: tree.update(f'{ep["act_file"]}\x1f{ep["act_file_sha256"]}\n'.encode())
    conversion={**empty_authorization(),'episodes':converted,'converted_tree_file_sha256':tree.hexdigest(),
        'converter_module_sha256':sha(__file__),'canonical_manifest_sha256':sha(WORK/'collection/ledger.jsonl'),
        'timesteps':{'converted_t_max':max(e['timesteps'] for e in converted)},
        'embedding_token_encoding':{'encoder_sha256':ENCODER_SHA256}}
    freeze(WORK/'conversion_manifest.json',conversion)
    episodes=[];ranks=collections.Counter()
    for i,record in enumerate(records):
        label='validation' if record['attempt_id'] in validation else 'train'
        episodes.append({'act_episode_index':i,'episode_id':record['attempt_id'],'candidate_index':record['row']['attempt_index'],
            'hazard_present':True,'split':label,'split_rank':ranks[label],'source_h5_sha256':record['trajectory_h5_sha256'],'cell':record['cell']})
        ranks[label]+=1
    assert ranks=={'train':240,'validation':48}
    split={**empty_authorization(),'schema':'hybrid_obstacle_canonical_split_v2','episodes':episodes,
        'counts':{k:{'total':v} for k,v in ranks.items()},'source_collection_tree_sha256':tree.hexdigest(),
        'canonical_manifest_sha256':conversion['canonical_manifest_sha256'],'split_rule':config['split']['ranking']}
    split['split_manifest_sha256']=digest(split);freeze(WORK/'split_manifest.json',split)
    from fixed_split_data import load_split_manifest,verify_dataset,compute_train_only_norm_stats
    loaded=load_split_manifest(WORK/'split_manifest.json')
    verify_dataset(WORK/'converted',loaded,WORK/'conversion_manifest.json',tree.hexdigest())
    stats,meta=compute_train_only_norm_stats(WORK/'converted',[e['act_episode_index'] for e in loaded['train']])
    path=WORK/'dataset_stats.pkl'
    if path.exists():
        previous=pickle.loads(path.read_bytes())
        assert all(__import__('numpy').array_equal(previous[k],stats[k]) for k in stats)
    else: path.write_bytes(pickle.dumps(stats))
    freeze(WORK/'dataset_stats_manifest.json',meta)
    freeze(WORK/'data_complete.json',{'utc':now(),'config_sha256':config['config_sha256'],'converted':288,'train':240,'validation':48,
        'conversion_sha256':sha(WORK/'conversion_manifest.json'),'split_sha256':sha(WORK/'split_manifest.json'),'statistics_sha256':sha(path)})

if __name__=='__main__': main()
