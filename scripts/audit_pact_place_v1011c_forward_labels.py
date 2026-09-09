"""Frozen-model diagnostic scored against the NEXT commanded arm action."""
import pickle
import json
from pact_place_v1011c_experiment import ROOT, DATA, TRAIN, WORK, read, freeze, empty_authorization, sha256_file
from audit_pact_place_v1011c_offline_ensemble import model_argv, decode
import numpy as np
import h5py
import torch
from policy import ACTPolicy

OUT=ROOT/'diagnostics_output/pact_place_v1011c_post_eval_audit/learning_failure'
STEPS=150


def combine(rows):
    n=sum(r['frames'] for r in rows)
    return {'frames':n,'arm_mae_rad':sum(r['frames']*r['arm_mae_rad'] for r in rows)/n}


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    source={r['act_episode_index']:r for r in read(WORK/'source_manifest.json')['rows']}
    selection=[r for r in read(WORK/'split_manifest.json')['episodes'] if r['split']=='validation']
    assert len(selection)==24
    items=[]
    for r in selection:
        idx=r['act_episode_index']
        with h5py.File(DATA/f'episode_{idx}.hdf5','r') as h:
            item={k:h[p][:STEPS] for k,p in [('qpos','observations/qpos'),('images','observations/images/wrist_camera'),
                ('proximity','observations/proximity_embeddings')]}
        with h5py.File(ROOT/source[idx]['trajectory_h5'],'r') as h:
            actions=[decode(x) for x in h['traj_0/actions/commanded_action'][1:STEPS+1]]
            assert len(actions)==STEPS and all(len(x['arm'])==7 for x in actions)
            item['truth']=np.array([x['arm'] for x in actions])
            touch=np.array([decode(x)['gripper']['touching'] for x in h['traj_0/obs/extra/grasp_state_pickup_obj'][()]])
            item['mask']=np.arange(STEPS)<int(np.flatnonzero(touch)[0])
        item['index']=idx;items.append(item)
    doc={**empty_authorization(),'training_chunk':100,'models':{},
        'scope':'Teacher-forced validation diagnostic; no training or counterfactual environment rollouts.',
        'truth':'arm from raw actions/commanded_action[t+1] for observation[t]',
        'limitations':['Only the seven arm commands are scored.',
            'Lower offline error is not established improvement in task success.',
            'Query shift 1 is a diagnostic reinterpretation of existing outputs, not an installed evaluator change.',
            'The shifted full-history diagnostic has at most 99 contributors because outputs have length 100.']}
    variants=[(0,1),(0,10),(0,100),(1,1),(1,10),(1,99)]
    for seed in (3103,3104,3105):
        for arm in ('ACT','PACT'):
            directory=TRAIN/f'{arm.lower()}_seed{seed}'
            cfg=read(directory/'run_manifest.json')['policy_config']
            assert cfg['num_queries']==100
            with (directory/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
            with model_argv(directory,seed):model=ACTPolicy(cfg)
            model.load_state_dict(torch.load(directory/'policy_best.ckpt',map_location='cpu',weights_only=True),strict=True)
            model.cuda().eval();rows=[]
            for item in items:
                chunks=[]
                with torch.inference_mode():
                    for start in range(0,STEPS,16):
                        stop=min(start+16,STEPS)
                        q=(item['qpos'][start:stop]-stats['qpos_mean'])/stats['qpos_std']
                        images=torch.from_numpy(item['images'][start:stop]).float().permute(0,3,1,2).unsqueeze(1).cuda()/255
                        prox=torch.from_numpy(item['proximity'][start:stop]).float().cuda() if arm=='PACT' else None
                        pred=model(torch.from_numpy(q).float().cuda(),images,proximity_positions=prox)
                        chunks.append(pred.cpu().numpy()*stats['action_std']+stats['action_mean'])
                chunks=np.concatenate(chunks);assert chunks.shape==(STEPS,100,8)
                outputs={}
                for shift,history in variants:
                    values=[]
                    for t in range(STEPS):
                        ages=np.arange(min(t+1,history));w=np.exp(-.01*ages);w/=w.sum()
                        values.append((chunks[t-ages,ages+shift,:7]*w[:,None]).sum(0))
                    outputs[f'query_shift_{shift}_history_{history}']=np.array(values)
                outputs['copy_current_arm']=item['qpos'][:,:7]
                mask=item['mask'];n=int(mask.sum())
                rows.append({'episode_index':item['index'],'errors':{k:{'frames':n,
                    'arm_mae_rad':float(np.abs(v-item['truth'])[mask].mean())} for k,v in outputs.items()}})
            name=f'{arm.lower()}_seed{seed}'
            aggregate={k:combine([r['errors'][k] for r in rows]) for k in rows[0]['errors']}
            doc['models'][name]={'arm':arm,'seed':seed,'episodes':rows,'aggregate':aggregate,
                'checkpoint_sha256':sha256_file(directory/'policy_best.ckpt')}
            print(name,json.dumps(aggregate),flush=True)
            del model;torch.cuda.empty_cache()
    doc['pooled']={a:{k:combine([r['errors'][k] for m in doc['models'].values() if m['arm']==a for r in m['episodes']])
        for k in next(iter(doc['models'].values()))['aggregate']} for a in ('ACT','PACT')}
    freeze(OUT/'forward_label_offline.json',doc)
    print('Created forward_label_offline.json',flush=True)


if __name__=='__main__':
    main()
