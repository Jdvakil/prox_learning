"""Read-only target-position and recorded wrist-segmentation audit."""
from pathlib import Path
from collections import Counter
import json
import h5py
import numpy as np
from pact_place_v1011c_experiment import ROOT,read,freeze,empty_authorization

OUT=ROOT/'diagnostics_output/pact_place_v1011c_post_eval_audit'


def extract(path):
    with h5py.File(path,'r') as h:
        positions=h['traj_0/obs/extra/obj_start'][()]
        assert np.array_equal(positions,np.broadcast_to(positions[0],positions.shape))
        base='traj_0/obs/extra/object_image_points/pickup_obj/wrist_camera/'
        count=h[base+'num_points'][()].reshape(-1)
        points=h[base+'points'][()]
        assert len(count)==len(positions) and points.shape==(len(count),10,2)
        assert np.array_equal(np.isfinite(points).all(-1).sum(-1),count)
        return {'target_initial_xyz_m':positions[0,:3].astype(float).tolist(),
                'wrist_target_segmented_initially':bool(count[0]>0),
                'wrist_target_segmented_first100':bool(np.any(count[:101]>0)),
                'wrist_target_segmented_ever':bool(np.any(count>0)),
                'wrist_target_segmented_frames':int(np.count_nonzero(count)),
                'frames':len(count)}


def summary(rows):
    coords=np.array([r['target_initial_xyz_m'] for r in rows])
    return {'n':len(rows),'target_min_xyz_m':coords.min(0).tolist(),
            'target_max_xyz_m':coords.max(0).tolist(),'target_mean_xyz_m':coords.mean(0).tolist(),
            **{key:sum(r[key] for r in rows) for key in ('wrist_target_segmented_initially',
                'wrist_target_segmented_first100','wrist_target_segmented_ever')},
            'median_fraction_frames_with_segmented_target':float(np.median([r['wrist_target_segmented_frames']/r['frames'] for r in rows]))}


def main():
    audit=read(OUT/'audit_v2.json')
    doc={**empty_authorization(),'definitions':{
        'position':'Cached world-frame initial target pose, not a motion trajectory.',
        'segmented':'ObjectImagePointsSensor.num_points > 0; detected target silhouette. Not a measure of visible area, graspability or policy understanding.',
        'grasp_held':'Contact-only GraspStateSensor heuristic, not a guarantee of stable grasp.'},'versions':{}}
    for version,tag in (('V10.10','v1010'),('V10.11c','v1011c')):
        evaluation=[]
        for row in audit['versions'][version]['rows']:
            x=extract(ROOT/row['directory']/'trajectory.h5')
            x.update({k:row[k] for k in ('arm','seed','episode_id','task_success','any_target_touch','any_target_held')})
            evaluation.append(x)
        b=ROOT/f'diagnostics_output/pact_place_{tag}_train_eval'
        source=read(b/'source_manifest.json');split=read(b/'split_manifest.json')
        train_indices={r['act_episode_index'] for r in split['episodes'] if r['split']=='train'}
        training=[]
        for row in source['rows']:
            if row['act_episode_index'] in train_indices:
                x=extract(ROOT/row['trajectory_h5']);x.update(attempt_id=row['attempt_id'],cell=row['cell'])
                training.append(x)
        assert len(training)==len(train_indices)
        by_arm={a:summary([r for r in evaluation if r['arm']==a]) for a in ('ACT','PACT')}
        strata={}
        for label,low,high in (('abs_y_under_0.1m',0,.1),('abs_y_0.1_to_0.2m',.1,.2),('abs_y_at_least_0.2m',.2,10)):
            strata[label]={}
            for arm in ('ACT','PACT'):
                rows=[r for r in evaluation if r['arm']==arm and low<=abs(r['target_initial_xyz_m'][1])<high]
                strata[label][arm]={'n':len(rows),**{k:sum(r[k] for r in rows) for k in ('task_success','any_target_touch','any_target_held')}}
        doc['versions'][version]={'training':summary(training),'evaluation':by_arm,
            'lateral_strata':strata,'training_rows':training,'evaluation_rows':evaluation}
        print(version,json.dumps({'training':doc['versions'][version]['training'],'evaluation':by_arm}),flush=True)
    freeze(OUT/'geometry_visibility.json',doc)


if __name__=='__main__':
    main()
