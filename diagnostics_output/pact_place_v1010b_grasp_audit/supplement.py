"""All-case stratification and final independent structural/preservation checks."""
from audit import *

def main():
    evidence=read(OUT/'evidence.json'); rows=evidence['rollouts']; strata={}
    physical=[r for r in rows if r['arm']=='PACT']
    bounds={key:np.quantile([r[key] if key=='initial_target_yaw_deg' else r['initial_target_xyz'][0] for r in physical],[.25,.5,.75]) for key in ('initial_target_yaw_deg','initial_target_x_m')}
    for seed in (3103,3104,3105):
        for arm in ('ACT','PACT'):
            rr=[r for r in rows if r['seed']==seed and r['arm']==arm]
            groups={}
            for key in ('family','side','pose_id','initial_target_yaw_deg','initial_target_x_m'):
                partition=collections.defaultdict(list)
                for r in rr:
                    label=r.get(key) if key not in bounds else int(np.searchsorted(bounds[key],r[key] if key=='initial_target_yaw_deg' else r['initial_target_xyz'][0],side='right'))
                    partition[str(label)].append(r)
                groups[key]={label:dict(n=len(sub),success=sum(r['success'] for r in sub),
                     exclusive_pickup=sum(r['failure_stage']=='touched_without_hold' for r in sub),
                     median_minimum_depth_m=med([r.get('minimum_rim_relative_tcp_early_grasp_m') for r in sub]),
                     median_command_close_xy_m=med([r.get('command_at_close_xy_error_m') for r in sub])) for label,sub in partition.items()}
                assert sum(x['n'] for x in groups[key].values())==50
            strata[f'{arm}_{seed}']=groups
    fixed_wall=[]; runtime=[]
    for r in rows:
        d=ROOT/r['directory']; result=read(d/'result.json');p=result['policy_info']
        assert p['control_steps']==p['model_output_trace_steps']==900 and p['num_queries']==p['averaging_history']==100
        assert (r['arm']=='ACT' and p['proximity_projection_calls']==0) or (r['arm']=='PACT' and p['proximity_projection_calls']==900)
        with h5py.File(d/'telemetry.h5','r') as h:
            ids=json.loads(h['contacts'].attrs['pair_identities']);pc=h['contacts/pair_counts'][()]
            t=h['contacts/sim_time_s'][()]; pads=np.zeros((len(t),2,5),bool)
        for j,identity in enumerate(ids):
            text=' '.join(identity['names']);wall=re.search(r'Cup_10_cup_10_PrimitiveCollider_(\d)',text)
            if not wall or identity['class']!='grasp_target':continue
            ix=pc[pc[:,1]==j,0].astype(int)
            for k,side in enumerate(('left','right')):
                if f'gripper/{side}_pad' in text:pads[ix,k,int(wall[1])]=True
        durations={wall:float(maxrun(pads[:,0,wall]&pads[:,1,wall])*np.median(np.diff(t))) for wall in (0,1,3,4)}
        fixed_wall.append(dict(rollout_id=r['rollout_id'],by_wall_longest_s=durations,
             fixed_wall_1s=max(durations.values())>=1,any_same_wall_1s=r['same_wall_longest_s']>=1))
        # Applied move-group joint target recorder excludes the final terminal sentinel.
        with h5py.File(d/'trajectory.h5','r') as h:
            g=h['traj_0'];commands=g['actions/commanded_action'];joints=g['actions/joint_pos'];checked=0;different=[]
            for k in range(1,len(commands)):
                a=decode(commands[k]);b=decode(joints[k])
                if not b:continue
                checked+=1
                if not all(np.array_equal(np.asarray(a[v],np.float32),np.asarray(b[v],np.float32)) for v in ('arm','gripper')): different.append(k)
        runtime.append(dict(rollout_id=r['rollout_id'],nonterminal_applied_targets_checked=checked,mismatching_steps=different))
    assert all(not r['mismatching_steps'] for r in runtime)
    original=read(G/'final_verification.json')
    for name,expected in original['artifacts'].items(): assert sha(G/name)==expected,name
    assert sha(W/'amendments/seed3103_full50_20260908/three_seed_full150.json')==original['official_full150_sha256']
    configuration=read(W/'config.json')
    for name,expected in configuration['file_hashes'].items():assert sha(ROOT/name)==expected,name
    expected_success=[19,20,22,19,23,35]; expected_cfts=[16,16,15,10,15,25]
    for (seed,arm),ss,cc in zip(itertools.product((3103,3104,3105),('ACT','PACT')),expected_success,expected_cfts):
        rr=[r for r in rows if r['seed']==seed and r['arm']==arm]
        assert len(rr)==50 and sum(r['success'] for r in rr)==ss and sum(r['collision_free_success'] for r in rr)==cc
    assert len(evidence['pair_audits'])==150 and all(r['passed'] for r in evidence['pair_audits'])
    assert len(evidence['validation_phase_errors']['rows'])==1440
    for path in [OUT/'AUDIT.md',ROOT/'docs/PACT_PLACE_V1010B_GRASP_FIX_PLAN.md',OUT/'storage_probe_six.json']:
        sha(path)
    write('supplement.json',dict(strata=strata,quartile_boundaries=bounds,fixed_wall_continuity=fixed_wall,
        applied_targets=runtime,source_hashes=HASHES))
    write('final_verification.json',dict(status='AUDIT_COMPLETE',rollouts=300,pairs=150,demos=280,
        frozen_validation_predictions=1440,new_simulator_rollouts=0,new_training_updates=0,
        original_diagnosis_hashes_unchanged=True,original_baseline_source_hashes_unchanged=True,
        raw_outcomes_reconciled=True,applied_targets_all_match=True,
        fixed_vs_any_wall_1s_disagreements=sum(r['fixed_wall_1s']!=r['any_same_wall_1s'] for r in fixed_wall),
        deliverable_hashes={str(p):sha(p) for p in [OUT/'AUDIT.md',OUT/'evidence.json',ROOT/'docs/PACT_PLACE_V1010B_GRASP_FIX_PLAN.md']},
        verification_code_sha256=sha(Path(__file__))))
    print('Supplement and final verification COMPLETE')

if __name__=='__main__':main()
