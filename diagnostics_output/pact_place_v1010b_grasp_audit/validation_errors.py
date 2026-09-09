"""Frozen-checkpoint offline phase errors; never constructs an environment.

Six deterministic anchors on every one of 40 validation episodes, six models.
No sampling/selection of checkpoints; inference prior (zero latent), no training.
"""
from audit import *
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
sys.path[:0]=[str(ROOT/'submodules/act'),str(ROOT/'scripts'),str(ROOT)]
from policy import ACTPolicy

def main():
    start=time.time(); data=read(OUT/'data_audit.json')
    demos=[r for r in data['rows'] if r['split']=='validation']
    assert len(demos)==40
    stats=pickle.loads((W/'dataset_stats.pkl').read_bytes());sha(W/'dataset_stats.pkl')
    fk,_=kinematics(); samples=[]
    for d in demos:
        c=d['first_close_command_step'];T=d['T']
        anchors=dict(approach=max(0,c-30),preclose=max(0,c-10),close=c,
                     postclose=min(T-1,c+10),transport=(c+T)//2,ending=max(0,T-30))
        with h5py.File(d['converted_path'],'r') as h:
            for phase,t in anchors.items():
                samples.append(dict(episode_id=d['episode_id'],phase=phase,t=t,close=c,
                    image=h['observations/images/wrist_camera'][t],qpos=h['observations/qpos'][t],
                    proximity=h['observations/proximity_embeddings'][t],
                    target=h['action'][t:min(t+100,T)],T=T))
    output=[]
    for seed in (3103,3104,3105):
        for arm in ('ACT','PACT'):
            directory=W/f'checkpoints/{arm.lower()}_seed{seed}'
            manifest=read(directory/'run_manifest.json')
            sys.argv=[sys.argv[0],'--ckpt_dir',str(directory),'--policy_class','ACT','--task_name','obstacle_baseline','--seed',str(seed),'--num_epochs','1']
            torch.manual_seed(seed)
            model=ACTPolicy(manifest['policy_config'])
            path=directory/'policy_update_60000.ckpt';sha(path)
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
            model.cuda().eval()
            for batch_start in range(0,len(samples),8):
                batch=samples[batch_start:batch_start+8]
                image=torch.from_numpy(np.stack([s['image'] for s in batch]).transpose(0,3,1,2)[:,None]).float().cuda()/255
                q=torch.from_numpy((np.stack([s['qpos'] for s in batch])-stats['qpos_mean'])/stats['qpos_std']).float().cuda()
                prox=torch.from_numpy(np.stack([s['proximity'] for s in batch])).float().cuda() if arm=='PACT' else None
                with torch.inference_mode(): pred=model(q,image,proximity_positions=prox).cpu().numpy()
                pred=pred*stats['action_std']+stats['action_mean']
                for s,p in zip(batch,pred):
                    n=len(s['target']);target=s['target'];p=p[:n]
                    delta=p-target
                    pred_tcp=np.array([fk(x[:7]) for x in p]); target_tcp=np.array([fk(x[:7]) for x in target])
                    diff=pred_tcp-target_tcp
                    record=dict(arm=arm,seed=seed,episode_id=s['episode_id'],phase=s['phase'],observation_step=s['t'],valid_chunk_rows=n,
                        normalized_valid_l1=float(np.abs(delta/stats['action_std']).mean()),
                        arm_joint_mae_rad=float(np.abs(delta[:,:7]).mean()),
                        tcp_fk_mean_error_m=float(np.linalg.norm(diff,axis=1).mean()),
                        tcp_fk_first_error_m=float(np.linalg.norm(diff[0])),tcp_fk_first_dz_m=float(diff[0,2]),
                        gripper_wrong_rows=int(((p[:,7]>=127.5)!=(target[:,7]>=127.5)).sum()))
                    close_offset=s['close']-s['t']
                    if 0<=close_offset<n:
                        record.update(close_target_fk_error_m=float(np.linalg.norm(diff[close_offset])),
                            close_target_fk_xy_error_m=float(np.linalg.norm(diff[close_offset,:2])),
                            close_target_fk_dz_m=float(diff[close_offset,2]),
                            close_target_gripper_wrong=bool((p[close_offset,7]>=127.5)!=(target[close_offset,7]>=127.5)))
                    output.append(record)
            del model;torch.cuda.empty_cache()
            print(arm,seed,'COMPLETE',round(time.time()-start,1),'s',flush=True)
    summary={}
    for seed in (3103,3104,3105):
        for arm in ('ACT','PACT'):
            summary[f'{arm}_{seed}']={}
            for phase in ('approach','preclose','close','postclose','transport','ending'):
                rr=[r for r in output if r['seed']==seed and r['arm']==arm and r['phase']==phase]
                assert len(rr)==40
                summary[f'{arm}_{seed}'][phase]=dict(n=40,
                    median_first_fk_error_m=med([r['tcp_fk_first_error_m'] for r in rr]),
                    median_valid_chunk_l1=med([r['normalized_valid_l1'] for r in rr]),
                    median_close_xy_error_m=med([r.get('close_target_fk_xy_error_m') for r in rr]),
                    median_close_dz_m=med([r.get('close_target_fk_dz_m') for r in rr]),
                    close_gripper_errors=sum(r.get('close_target_gripper_wrong',False) for r in rr),
                    gripper_wrong_rows=sum(r['gripper_wrong_rows'] for r in rr),
                    valid_rows=sum(r['valid_chunk_rows'] for r in rr))
    write('validation_errors.json',dict(status='COMPLETE',rows=output,summary=summary,source_hashes=HASHES,
        samples_per_model=len(samples),checkpoint_count=6,elapsed_s=time.time()-start,
        limitation='Teacher-forced demonstration observations and individual chunks, not final-scene images or closed-loop aggregated actions; expert action error does not uniquely measure geometric feasibility.'))

if __name__=='__main__': main()
