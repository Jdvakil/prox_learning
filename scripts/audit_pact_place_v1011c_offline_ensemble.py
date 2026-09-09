"""Frozen chunk-100 models: read-only, teacher-forced arm-command diagnostic.

This does NOT train chunk 1/10 models or execute alternative closed-loop policies.
It compares aggregation histories on the same recorded validation observations.
"""
import json
import pickle
import sys
from pathlib import Path
import argparse
from contextlib import contextmanager

from pact_place_v1011c_experiment import ROOT, DATA, TRAIN, WORK, read, freeze, empty_authorization, sha256_file
import h5py
import numpy as np
import torch
from policy import ACTPolicy

OUT=ROOT/'diagnostics_output/pact_place_v1011c_post_eval_audit'
STEPS=150
BATCH=16


@contextmanager
def model_argv(directory, seed):
    saved=sys.argv
    sys.argv=[saved[0],'--ckpt_dir',str(directory),'--policy_class','ACT',
              '--task_name','obstacle_baseline','--seed',str(seed),'--num_epochs','1']
    try:
        yield
    finally:
        sys.argv=saved


def decode(row):
    return json.loads(row.tobytes().split(b'\0',1)[0])


def errors(predictions,truth,mask):
    if not mask.any():
        return {'frames':0}
    d=(predictions-truth)[mask,:7]
    return {'frames':int(mask.sum()),'arm_mean_abs_error_rad':float(np.abs(d).mean()),
            'arm_mean_l2_error_rad':float(np.linalg.norm(d,axis=1).mean())}


def aggregate(rows):
    out={}
    for period in ('first150','before_first_touch','first10_visible_before_touch'):
        out[period]={}
        for history in ('1','10','100'):
            parts=[r['errors'][period][history] for r in rows]
            n=sum(p['frames'] for p in parts)
            out[period][history]={'frames':n, **{key:sum(p.get(key,0)*p['frames'] for p in parts)/n if n else None
                for key in ('arm_mean_abs_error_rad','arm_mean_l2_error_rad')}}
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--historical-v1010',action='store_true')
    args=parser.parse_args()
    work=ROOT/'diagnostics_output/pact_place_v1010_train_eval' if args.historical_v1010 else WORK
    data_root=ROOT/'assets/act_style_data/pact_place_v1010_144' if args.historical_v1010 else DATA
    if args.historical_v1010:
        historical_models=read(work/'training_verification.json')['arms']
        directories={arm:Path(historical_models[arm.lower()]['checkpoint_dir']) for arm in ('ACT','PACT')}
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    split=read(work/'split_manifest.json');source=read(work/'source_manifest.json')
    selected=sorted((r for r in split['episodes'] if r['split']=='validation'),key=lambda r:r['act_episode_index'])
    assert len(selected)==24 and len({r['cell'] for r in selected})==24
    by_index={r['act_episode_index']:r for r in source['rows']}
    data=[]
    for row in selected:
        i=row['act_episode_index']
        with h5py.File(data_root/f'episode_{i}.hdf5','r') as h:
            assert bool(h.attrs['sim']) and len(h['action'])>=STEPS
            x={k:h[path][:STEPS] for k,path in (
                ('image','observations/images/wrist_camera'),('qpos','observations/qpos'),
                ('proximity','observations/proximity_embeddings'),('truth','action'))}
            assert x['image'].shape==(STEPS,240,320,3) and x['qpos'].shape==(STEPS,9)
        with h5py.File(ROOT/by_index[i]['trajectory_h5'],'r') as h:
            touch=np.array([decode(r)['gripper']['touching'] for r in h['traj_0/obs/extra/grasp_state_pickup_obj'][()]])
            seen=h['traj_0/obs/extra/object_image_points/pickup_obj/wrist_camera/num_points'][()].reshape(-1)>0
        first_touch=int(np.flatnonzero(touch)[0])
        first_seen=int(np.flatnonzero(seen)[0])
        x.update(episode_index=i,cell=row['cell'],first_touch_step=first_touch,first_segmented_step=first_seen)
        data.append(x)
    doc={**empty_authorization(),'scope':'Offline teacher-forced command-error diagnostic only; no alternative closed-loop outcomes.',
         'training_chunk':100,'prediction_chunk':100,'histories_compared':[1,10,100],
         'validation_episodes':24,'steps_per_episode':STEPS,'models':{},
         'limitations':[
             'Every policy receives expert states/images, not the counterfactual states a shorter history would generate.',
             'Only the seven arm joint commands are scored; no gripper-command analysis.',
             'This is a descriptive reuse of validation data, not a new held-out task-success evaluation.',
             'Comparison changes only the offline aggregation history, not learned prediction horizon.',
             'Pre-touch masks come from each expert trajectory; trajectories may have different first-touch times.'
         ]}
    for seed in ((3101,) if args.historical_v1010 else (3103,3104,3105)):
        for arm in ('ACT','PACT'):
            directory=directories[arm] if args.historical_v1010 else TRAIN/f'{arm.lower()}_seed{seed}'
            config=read(directory/'run_manifest.json')['policy_config']
            assert config['num_queries']==100
            with (directory/'dataset_stats.pkl').open('rb') as f:
                stats=pickle.load(f)
            with model_argv(directory,seed):
                model=ACTPolicy(config)
            model.load_state_dict(torch.load(directory/'policy_best.ckpt',map_location='cpu',weights_only=True),strict=True)
            model.cuda().eval()
            results=[]
            for item in data:
                chunks=[]
                with torch.inference_mode():
                    for start in range(0,STEPS,BATCH):
                        stop=min(start+BATCH,STEPS)
                        q=(item['qpos'][start:stop]-stats['qpos_mean'])/stats['qpos_std']
                        image=torch.from_numpy(item['image'][start:stop]).float().permute(0,3,1,2).unsqueeze(1).cuda()/255
                        prox=torch.from_numpy(item['proximity'][start:stop]).float().cuda() if arm=='PACT' else None
                        pred=model(torch.from_numpy(q).float().cuda(),image,proximity_positions=prox)
                        chunks.append(pred.cpu().numpy()*stats['action_std']+stats['action_mean'])
                chunks=np.concatenate(chunks)
                assert chunks.shape==(STEPS,100,8) and np.isfinite(chunks).all()
                outputs={}
                for history in (1,10,100):
                    values=[]
                    for t in range(STEPS):
                        ages=np.arange(min(t+1,history));weights=np.exp(-.01*ages);weights/=weights.sum()
                        values.append((chunks[t-ages,ages]*weights[:,None]).sum(0))
                    outputs[str(history)]=np.array(values)
                assert np.allclose(outputs['1'],chunks[:,0])
                for x in outputs.values():
                    assert np.allclose(x[0],chunks[0,0])
                t=np.arange(STEPS)
                masks={'first150':np.ones(STEPS,dtype=bool),'before_first_touch':t<item['first_touch_step'],
                       'first10_visible_before_touch':(t>=item['first_segmented_step'])&(t<item['first_segmented_step']+10)&(t<item['first_touch_step'])}
                result={k:item[k] for k in ('episode_index','cell','first_touch_step','first_segmented_step')}
                result['errors']={name:{hist:errors(pred,item['truth'],mask) for hist,pred in outputs.items()} for name,mask in masks.items()}
                results.append(result)
            name=f'{arm.lower()}_seed{seed}'
            doc['models'][name]={'arm':arm,'seed':seed,'checkpoint_sha256':sha256_file(directory/'policy_best.ckpt'),
                                 'episodes':results,'aggregate':aggregate(results)}
            print(name,json.dumps(doc['models'][name]['aggregate']),flush=True)
            del model
            torch.cuda.empty_cache()
    doc['pooled']={arm:aggregate([r for m in doc['models'].values() if m['arm']==arm for r in m['episodes']]) for arm in ('ACT','PACT')}
    filename='offline_ensemble_validation_v1010.json' if args.historical_v1010 else 'offline_ensemble_validation.json'
    freeze(OUT/filename,doc)
    print('Verified '+filename,flush=True)


if __name__=='__main__':
    main()
