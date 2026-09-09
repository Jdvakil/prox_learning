"""Read-only supervision timing, observability and training-progress diagnostics."""
from collections import Counter
from pathlib import Path
import json
import sys

from pact_place_v1011c_experiment import ROOT, read, freeze, empty_authorization, sha256_file
import h5py
import numpy as np
import cv2

OUT=ROOT/'diagnostics_output/pact_place_v1011c_post_eval_audit/learning_failure'


def decode(x):
    return json.loads(x.tobytes().split(b'\0',1)[0])


def extract(version,count):
    base=ROOT/f'diagnostics_output/pact_place_{version}_train_eval'
    source=read(base/'source_manifest.json');split=read(base/'split_manifest.json')
    roles={r['act_episode_index']:r['split'] for r in split['episodes']}
    result=[]
    for row in source['rows']:
        p=ROOT/row['trajectory_h5'];idx=row['act_episode_index']
        with h5py.File(p,'r') as h:
            g=h['traj_0']
            raw_j=[decode(x) for x in g['actions/joint_pos'][()]]
            raw_c=[decode(x) for x in g['actions/commanded_action'][()]]
            raw_q=[decode(x) for x in g['obs/agent/qpos'][()]]
            assert raw_j[-1]=={} and raw_c[0]=={}
            assert all(len(x['arm'])==7 for x in raw_j[:-1]+raw_c[1:-1]+raw_q)
            assert 'arm' in raw_c[-1] or set(raw_c[-1])=={'success'}
            j=np.array([x['arm'] for x in raw_j[:-1]],dtype=np.float32)
            valid_next=np.array(['arm' in x for x in raw_c[1:]])
            c=np.array([x['arm'] if 'arm' in x else [np.nan]*7 for x in raw_c[1:]],dtype=np.float32)
            q=np.array([x['arm'] for x in raw_q],dtype=np.float32)
            assert j.shape==c.shape==(len(q)-1,7)
            # H5 index k>0 contains the command that led into observation k.
            assert np.allclose(j[1:],c[:-1],atol=3e-7,rtol=0)
            seen=g['obs/extra/object_image_points/pickup_obj/wrist_camera/num_points'][()].reshape(-1)>0
            touch=np.array([decode(x)['gripper']['touching'] for x in g['obs/extra/grasp_state_pickup_obj'][()]])
            first_seen=int(np.flatnonzero(seen)[0]);first_touch=int(np.flatnonzero(touch)[0])
            target=g['obs/extra/obj_start'][0,:3].astype(float)
            phase=g['obs/extra/policy_phase'][()]
        converted=ROOT/f'assets/act_style_data/pact_place_{version}_{count}/episode_{idx}.hdf5'
        with h5py.File(converted,'r') as h:
            assert np.array_equal(h['action'][:,:7],j)
            assert np.array_equal(h['observations/qpos'][:,:7],q[:-1])
        end=min(first_touch,150)
        pre=slice(0,end)
        r={k:row[k] for k in ('attempt_id','cell','act_episode_index')}
        r.update(split=roles[idx],source_path=str(p),converted_path=str(converted),steps=int(valid_next.sum()),
            converted_steps=len(j),terminal_status_without_arm=int((~valid_next).sum()),
            target_xyz_m=target.tolist(),first_seen=first_seen,first_touch=first_touch,
            command_lag_max_error_rad=float(np.abs(j[1:]-c[:-1]).max()),
            old_label_vs_next_command_mae_rad=float(np.abs(j-c)[valid_next].mean()),
            old_label_current_arm_mae_rad=float(np.abs(j-q[:-1])[valid_next].mean()),
            next_command_current_arm_mae_rad=float(np.abs(c-q[:-1])[valid_next].mean()),
            pre_touch_frames=end,
            pre_touch_old_label_current_arm_mae_rad=float(np.abs(j[pre]-q[pre]).mean()),
            pre_touch_next_command_current_arm_mae_rad=float(np.abs(c[pre]-q[pre]).mean()),
            expert_joint_displacement_before_first_seen_l2_rad=float(np.linalg.norm(q[first_seen]-q[0])),
            expert_joint_displacement_first50_l2_rad=float(np.linalg.norm(q[min(50,len(q)-1)]-q[0])))
        result.append(r)
    assert len(result)==count
    summaries={}
    for role in ('train','validation'):
        rows=[r for r in result if r['split']==role]
        steps=sum(r['steps'] for r in rows);pretouch=sum(r['pre_touch_frames'] for r in rows)
        summaries[role]={'n':len(rows),'total_action_steps':steps,
            'command_lag_verified_episodes':len(rows),
            'max_lag_equality_error_rad':max(r['command_lag_max_error_rad'] for r in rows),
            'target_seen_initially':sum(r['first_seen']==0 for r in rows),
            'target_first_seen_step_quantiles':np.quantile([r['first_seen'] for r in rows],[0,.25,.5,.75,1]).tolist(),
            'target_first_touch_step_quantiles':np.quantile([r['first_touch'] for r in rows],[0,.25,.5,.75,1]).tolist(),
            'median_expert_motion_before_first_seen_l2_rad':float(np.median([r['expert_joint_displacement_before_first_seen_l2_rad'] for r in rows])),
            'cell_counts':dict(Counter(r['cell'] for r in rows))}
        for k in ('old_label_vs_next_command_mae_rad','old_label_current_arm_mae_rad','next_command_current_arm_mae_rad'):
            summaries[role][k]=sum(r[k]*r['steps'] for r in rows)/steps
        for k in ('pre_touch_old_label_current_arm_mae_rad','pre_touch_next_command_current_arm_mae_rad'):
            summaries[role][k]=sum(r[k]*r['pre_touch_frames'] for r in rows)/pretouch
    return {'rows':result,'summaries':summaries}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    doc={**empty_authorization(),'versions':{},'training_progress':{},'visual_samples':[],
         'limitations':['No new training, data conversion or environment rollout.',
             'Command timing is tested on arm joints only; gripper commands are not analyzed.',
             'The one-step lag is an established supervision mismatch, not a measured causal explanation of the entire success deficit.',
             'Target segmentation absence does not establish absence of every indirect visual cue.']}
    for version,count in [('v1011c',99),('v1010',144)]:
        doc['versions'][version]=extract(version,count)
        print(version,json.dumps(doc['versions'][version]['summaries']),flush=True)
    for directory in sorted((ROOT/'diagnostics_output/pact_place_v1011c_train_eval/checkpoints').glob('*_seed*')):
        rows=[json.loads(x) for x in (directory/'epoch_log.jsonl').read_text().splitlines()]
        assert len(rows)==2000 and rows[-1]['global_step']==20000
        doc['training_progress'][directory.name]={'global_step':rows[-1]['global_step'],'best_epoch':rows[-1]['best_epoch'],
            'window_means':{str(stop):{role:{metric:float(np.mean([r[role][metric] for r in rows[stop-200:stop]]))
                for metric in ('loss','l1','kl')} for role in ('train','val')} for stop in (500,1000,1500,2000)}}
    # Fixed representatives: earliest accepted id from each third of target Y.
    rows=[r for r in doc['versions']['v1011c']['rows'] if r['split']=='train']
    for label,lo,hi in [('negative',-1,-.15),('center',-.15,.15),('positive',.15,1)]:
        row=min((r for r in rows if lo<=r['target_xyz_m'][1]<hi),key=lambda r:r['attempt_id'])
        with h5py.File(row['converted_path'],'r') as h:
            for t in sorted({0,25,50,row['first_seen'],row['first_touch']}):
                path=OUT/f'{label}_{row["attempt_id"][:16]}_step{t:03}.png'
                if not path.exists():
                    assert cv2.imwrite(str(path),cv2.cvtColor(h['observations/images/wrist_camera'][t],cv2.COLOR_RGB2BGR))
                doc['visual_samples'].append({'path':str(path),'step':t,'target_xyz_m':row['target_xyz_m'],
                    'first_seen':row['first_seen'],'first_touch':row['first_touch']})
    freeze(OUT/'supervision_observability.json',doc)
    print('Created supervision_observability.json',flush=True)


def camera_audit():
    source=read(ROOT/'diagnostics_output/pact_place_v1011c_train_eval/source_manifest.json')
    prior=read(OUT/'supervision_observability.json')
    roles={r['act_episode_index']:r['split'] for r in prior['versions']['v1011c']['rows']}
    samples={Path(r['path']).name.split('_')[1] for r in prior['visual_samples']}
    rows=[]
    for row in source['rows']:
        p=ROOT/row['trajectory_h5']
        video=p.parent/'episode_00000000_exo_camera_1.mp4'
        expected=row['coordinator_validation']['table_camera']['detail']['rgb_video_sha256']
        assert sha256_file(video)==expected
        with h5py.File(p,'r') as h:
            g=h['traj_0'];count=len(g['success'])
            values={}
            for cam in ('wrist_camera','exo_camera_1'):
                b=g[f'obs/extra/object_image_points/pickup_obj/{cam}']
                n=b['num_points'][()].reshape(-1);points=b['points'][()]
                assert np.array_equal(np.isfinite(points).all(-1).sum(-1),n)
                assert len(n)==count
                seen=n>0;initial_run=next((i for i,x in enumerate(seen) if not x),len(seen))
                values[cam]={'seen_initially':bool(seen[0]),'first_seen':int(np.flatnonzero(seen)[0]) if seen.any() else None,
                    'visible_fraction_first100':float(seen[:100].mean()),'visible_fraction':float(seen.mean()),
                    'consecutive_visible_from_start':initial_run}
        cap=cv2.VideoCapture(str(video));decoded=0;first=None
        assert cap.isOpened()
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                if decoded==0:first=frame.copy()
                decoded+=1
        finally:cap.release()
        assert decoded==count
        image_path=None
        if row['attempt_id'][:16] in samples:
            image_path=OUT/f'exo_{row["attempt_id"][:16]}_step000.png'
            if not image_path.exists():assert cv2.imwrite(str(image_path),first)
        rows.append({'attempt_id':row['attempt_id'],'split':roles[row['act_episode_index']],
            'video_path':str(video),'video_sha256':expected,'frames_decoded':decoded,'cameras':values,
            'sample_image':str(image_path) if image_path else None})
        if len(rows)%24==0:print('Decoded and checked table-camera episodes',len(rows),flush=True)
    summaries={}
    for role in ('train','validation','all'):
        selected=[r for r in rows if r['split']==role or role=='all']
        summaries[role]={'n':len(selected),'cameras':{cam:{
            'target_seen_initially':sum(r['cameras'][cam]['seen_initially'] for r in selected),
            'mean_visible_fraction_first100':float(np.mean([r['cameras'][cam]['visible_fraction_first100'] for r in selected])),
            'median_visible_fraction':float(np.median([r['cameras'][cam]['visible_fraction'] for r in selected]))}
            for cam in ('wrist_camera','exo_camera_1')}}
    doc={**empty_authorization(),'rows':rows,'summaries':summaries,
        'scope':'Existing source camera videos and rendered segmentation only; no camera, dataset or evaluator changed.',
        'limitations':['Binary segmentation presence is not a measure of useful visible area.',
            'The existing checkpoints were trained on wrist_camera only and cannot gain a second camera merely by supplying an extra image.']}
    freeze(OUT/'table_camera_audit.json',doc)
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':
    camera_audit() if '--camera-audit' in sys.argv else main()
