"""Explicit V10.10b worker with unchanged learned control and diagnostic hooks."""
from __future__ import annotations
import argparse
import collections
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from pact_v1010b_contract import *
os.environ.update(environment())
pin_torch()
import h5py
import numpy as np
import mujoco
import torch
import cv2
import pact_wrist288_eval_worker as wrist
import eval_pact_collision_row as legacy
import eval_pact_place_row as place
import eval_pact_place_v109_row as v109
import eval_pact_frontend_screen_row as frontend
from pact_v1010b_storage import repack

ACTIVE = None
OUTPUT = None


def force_world_on_geom2(contact, force):
    """MuJoCo frame rows are normal/tangents; positive force acts on geom2."""
    return np.asarray(contact.frame).reshape(3,3).T @ np.asarray(force)[:3]


def instrument_limit(close):
    return min(300, close+60) if close is not None else 300


def contributor_record(step, pending):
    queries=np.array([q for q,value in pending if 0<=step-q<len(value)],np.int32)
    ages=step-queries; weights=np.exp(-.01*ages.astype(np.float64)); weights/=weights.sum()
    assert len(queries)<=100 and np.all((ages>=0)&(ages<100))
    return queries,ages,weights


class GraspRecorder:
    def __init__(self,policy):
        self.policy=policy;self.close=None;self.physics=[];self.control=[];self.contacts=[]
        self.frames=collections.deque(maxlen=150);self.last_time=None;self.metadata=None
        self.control_times=set();self.chunk_steps=[];self.chunks=[];self.contributors=[];self.chunk_checks=0
        self.input_hashes=[];self._written_info=None

    def initialize(self,env):
        model=env.current_model
        bodies=[i for i in range(model.nbody) if 'cavity_obj_0/Cup_10' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or '')]
        # Cup root body has its free joint; nested primitive body shares its pose.
        root=next(i for i in bodies if int(model.body_jntnum[i])>0)
        pads=[i for i in range(model.nbody) if any(f'gripper/{s}_pad' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or '') for s in ('left','right'))]
        assert len(pads)==2
        joints=[i for i in range(model.njnt) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) or '').startswith('robot_0/')]
        self.body_ids=[root,*pads];self.joints=joints
        self.qpos_ids=[int(model.jnt_qposadr[j]) for j in joints];self.qvel_ids=[int(model.jnt_dofadr[j]) for j in joints]
        geom_ids=[i for i in range(model.ngeom) if 'Cup_10_cup_10_PrimitiveCollider_' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '')]
        assert len(geom_ids)==5 and all(model.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX for i in geom_ids)
        self.geom_ids=geom_ids
        self.metadata={'body_ids':self.body_ids,'body_names':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) for i in self.body_ids],
            'joint_ids':joints,'joint_names':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in joints],
            'qpos_indices':self.qpos_ids,'qvel_indices':self.qvel_ids,'cup_geom_ids':geom_ids,
            'cup_geom_sizes':model.geom_size[geom_ids].tolist(),
            'geom_names':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(model.ngeom)],
            'geom_body_ids':model.geom_bodyid.tolist(),
            'body_all_names':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(model.nbody)],
            'force_convention':'mj_contactForce local force on geom2; frame rows [normal,tangent1,tangent2]; world = frame.T @ local; opposite acts on geom1',
            'time_convention':'contact callback after mj_step; contact geometry/solver force share the solver state; integration may have advanced qpos to timestamp',
            'physics_columns':['time_s','control_step','requested_control_index'],
            'contact_columns':['frame_index','contact_index','geom1','geom2','body1','body2','distance','pos_x','pos_y','pos_z',
                'normal_x','normal_y','normal_z','force_normal','force_tangent1','force_tangent2','torque_normal','torque_tangent1','torque_tangent2','force_world_geom2_x','force_world_geom2_y','force_world_geom2_z']}

    def measure(self,env,step,physics):
        if self.metadata is None:self.initialize(env)
        data=env.current_data;stamp=float(data.time)
        row={'time':stamp,'step':int(step),'command_index':int(step)-1,
            'body_pose':np.concatenate([np.r_[data.xpos[b],data.xquat[b]] for b in self.body_ids]),
            'joint_qpos':data.qpos[self.qpos_ids].copy(),'joint_qvel':data.qvel[self.qvel_ids].copy(),
            'ctrl':data.ctrl.copy(),'requested':np.r_[self.policy._v109_arm[-1],self.policy._v109_gripper[-1]] if self.policy._v109_arm else np.zeros(8,np.float32),
            'requested_valid':bool(self.policy._v109_arm),
            'cup_geom_pos':data.geom_xpos[self.geom_ids].copy(),'cup_geom_rotation':data.geom_xmat[self.geom_ids].copy()}
        if physics:
            if self.last_time==stamp:return
            self.last_time=stamp; index=len(self.physics);self.physics.append(row)
            for ci in range(data.ncon):
                contact=data.contact[ci];force=np.zeros(6,np.float64)
                mujoco.mj_contactForce(env.current_model,data,ci,force)
                g1,g2=int(contact.geom1),int(contact.geom2)
                self.contacts.append([index,ci,g1,g2,int(env.current_model.geom_bodyid[g1]),int(env.current_model.geom_bodyid[g2]),float(contact.dist),
                    *contact.pos,*np.asarray(contact.frame)[:3],*force,*force_world_on_geom2(contact,force)])
        elif stamp not in self.control_times:
            self.control_times.add(stamp);self.control.append(row)

    def observe_physics(self,env,step):
        if ACTIVE['instrumentation']=='grasp' and step<=instrument_limit(self.close): self.measure(env,step,True)

    def observe_control(self,observation,step):
        if ACTIVE['instrumentation']!='grasp':return
        self.measure(self.policy.task.env,step,False)
        if step>instrument_limit(self.close):return
        image=np.asarray(observation['wrist_camera'])
        if image.dtype!=np.uint8:
            image=(image*255. if float(image.max())<=1 else image).astype(np.uint8)
        if image.shape[:2]!=(240,320):image=cv2.resize(image,(320,240),interpolation=cv2.INTER_AREA)
        if self.close is None or step>=self.close-30:self.frames.append((step,float(self.policy.task.env.current_data.time),image.copy()))

    def closed(self,step,action):
        if self.close is None and float(action['gripper'][0])>=127.5:
            self.close=step
            self.frames=collections.deque([r for r in self.frames if r[0]>=step-30],maxlen=150)

    def write(self):
        if self._written_info is not None:return self._written_info
        if not ACTIVE['record_chunks'] and ACTIVE['instrumentation']!='grasp':return {}
        path=OUTPUT/'grasp_diagnostics.h5'
        with h5py.File(path,'x') as h:
            h.attrs['schema']=SCHEMA;h.attrs['stats_sha256']=ACTIVE['stats_sha256']
            h.attrs['close_control_index']=-1 if self.close is None else self.close
            h.attrs['instrumentation']=ACTIVE['instrumentation'];h.attrs['policy_horizon']=900
            def put(name,value):
                a=np.asarray(value);h.create_dataset(name,data=a,compression='gzip',compression_opts=4,shuffle=True)
            if ACTIVE['record_chunks']:
                assert len(self.chunks)==900 and self.chunk_checks==900
                put('chunks/query_control_index',self.chunk_steps);put('chunks/normalized_action',self.chunks)
                put('chunks/action_mean',self.policy._stats['action_mean']);put('chunks/action_std',self.policy._stats['action_std'])
                put('chunks/contributors',self.contributors)
                h['chunks'].attrs['contributor_columns']=json.dumps(['control_step','query_index','age','weight'])
                h['chunks'].attrs['hook_tensor_equality_checks']=self.chunk_checks
                put('chunks/post_aggregate',self.policy._v109_model_outputs)
                h['chunks'].attrs['input_hashes']=json.dumps(self.input_hashes)
            if ACTIVE['instrumentation']=='grasp':
                assert len(self.control)==901 and len(self.frames)<=150
                assert max(r['step'] for r in self.physics)<=300
                h.attrs['metadata']=json.dumps(self.metadata,sort_keys=True)
                for name,rows in [('physics',self.physics),('control',self.control)]:
                    for key in rows[0]:put(name+'/'+key,[r[key] for r in rows])
                put('contacts/solver_samples',np.asarray(self.contacts).reshape(-1,22))
                put('wrist/control_index',[r[0] for r in self.frames]);put('wrist/time_s',[r[1] for r in self.frames]);put('wrist/rgb',[r[2] for r in self.frames])
        clips=[]
        if self.frames:
            import imageio.v2 as imageio
            clip=OUTPUT/'new_replay_wrist.mp4'
            writer=imageio.get_writer(clip,fps=15,codec='libx264',quality=7,macro_block_size=1)
            try:
                for _,_,frame in self.frames:writer.append_data(frame)
            finally:writer.close()
            clips=[str(clip)]
        total=path.stat().st_size+sum(Path(p).stat().st_size for p in clips)
        assert ACTIVE['instrumentation']!='grasp' or total<=100*MIB, f'instrumentation exceeds100MiB: {total}'
        self._written_info={'path':str(path),'sha256':sha(path),'bytes':total,'new_replay_clips':clips,
            'chunk_queries':len(self.chunks),'hook_tensor_equality_checks':self.chunk_checks,
            'physics_samples':len(self.physics),'control_samples':len(self.control),'rgb_frames':len(self.frames)}
        return self._written_info


class DiagnosticContactAudit(wrist.Wrist288ContactAudit):
    def observe(self,env,step):
        super().observe(env,step)
        self.recorder.observe_physics(env,step)


class Wrist1010bInferencePolicy(wrist.Wrist288InferencePolicy):
    def reset(self):
        super().reset();self.recorder=GraspRecorder(self)
        self._contact_audit=DiagnosticContactAudit();self._contact_audit.recorder=self.recorder
        self.task._contact_audit_hook=self._contact_audit

    def prepare_model(self):
        super().prepare_model()
        assert self._surface_positions.__func__ is frontend.PactFrontendScreenInferencePolicy._surface_positions
        if self.pc.arm=='PACT':
            assert tuple(self._policy.model.input_proj_proximity.weight.shape)==(512,32)
            assert not any(p.requires_grad for p in self._surface_encoder.parameters())
        if ACTIVE['record_chunks'] and not hasattr(self,'_chunk_hook'):
            def capture(module,inputs,kwargs,output):
                predicted=output[0]
                assert predicted.shape==(1,100,8) and predicted.dtype==torch.float32
                original=predicted.detach().clone()
                self.recorder.chunks.append(predicted.detach().squeeze(0).cpu().numpy().copy())
                self.recorder.chunk_steps.append(int(self._step))
                h=hashlib.sha256()
                for value in [*inputs,*[kwargs[k] for k in sorted(kwargs)]]:
                    if isinstance(value,torch.Tensor):h.update(value.detach().cpu().contiguous().numpy().tobytes())
                self.recorder.input_hashes.append(h.hexdigest())
                assert torch.equal(original,predicted);self.recorder.chunk_checks+=1
                return None
            self._chunk_hook=self._policy.model.register_forward_hook(capture,with_kwargs=True)

    def inference_model(self,observation):
        self.recorder.observe_control(observation,self._step)
        output=super().inference_model(observation)
        if ACTIVE['record_chunks']:
            queries,ages,weights=contributor_record(self._step,self._pending_chunks)
            values=np.stack([v[self._step-q] for q,v in self._pending_chunks])
            reconstructed=(values*weights[:,None]).sum(axis=0).astype(np.float32)
            assert np.array_equal(output,reconstructed), 'hook changed aggregation'
            self.recorder.contributors.extend([self._step,int(q),int(a),float(w)] for q,a,w in zip(queries,ages,weights))
        return output

    def model_output_to_action(self,model_output):
        action=super().model_output_to_action(model_output)
        self.recorder.closed(int(self._step),action)
        return action

    def get_info(self):
        if ACTIVE['instrumentation']=='grasp':self.recorder.measure(self.task.env,self._step,False)
        info=super().get_info()
        info['v1010b_diagnostics']=self.recorder.write()
        info['v1010b_provenance']={k:ACTIVE[k] for k in ('training_seed','scene_id','scene_block','variant','instrumentation','checkpoint_sha256','stats_sha256')}
        info['v1010b_binding_path']=[c.__name__ for c in type(self).__mro__]
        return info


class Wrist1010bPolicyConfig(wrist.Wrist288PolicyConfig):
    policy_cls:type=Wrist1010bInferencePolicy


class Wrist1010bEvalConfig(wrist.Wrist288EvalConfig):
    pass


def verify_bindings():
    path=[c.__name__ for c in Wrist1010bInferencePolicy.__mro__]
    assert path[:6]==['Wrist1010bInferencePolicy','Wrist288InferencePolicy','PactPlaceV109InferencePolicy',
        'PactPlaceInferencePolicy','PactFrontendScreenInferencePolicy','PactCollisionInferencePolicy']
    assert Wrist1010bInferencePolicy._surface_positions is frontend.PactFrontendScreenInferencePolicy._surface_positions
    assert Wrist1010bInferencePolicy.model_output_to_action.__module__==__name__
    return path


def run(args):
    global ACTIVE,OUTPUT
    OUTPUT=inside(args.output_dir);OUTPUT.mkdir(parents=True,exist_ok=True)
    assert not (OUTPUT/'result.json').exists()
    manifest=read(args.manifest)
    assert manifest['manifest_sha256']==digest({k:v for k,v in manifest.items() if k!='manifest_sha256'})
    scene=next(r for r in manifest['scenes'] if r['scene_id']==args.scene_id)
    assert scene['scene_block']==args.scene_block
    row=scene['physical_row'];assert digest(row)==scene['physical_row_digest']
    checkpoint=Path(args.checkpoint_path).resolve();assert sha(checkpoint)==args.checkpoint_sha256
    stats=checkpoint.parent/'dataset_stats.pkl';assert sha(stats)==sha(W/'dataset_stats.pkl')
    config=read(B/'contract.json')
    assert config['code_hashes'][str(Path(__file__).resolve())]==sha(__file__)
    verify_bindings()
    ACTIVE={'training_seed':args.training_seed,'scene_id':args.scene_id,'scene_block':args.scene_block,
        'variant':args.variant,'instrumentation':args.instrumentation,'checkpoint_sha256':args.checkpoint_sha256,
        'stats_sha256':sha(stats),'record_chunks':manifest['stage'] in ('A1','A2')}
    scene_path=ROOT/row['pact_v1010_scene_relative'];assert sha(scene_path)==row['pact_v106_scene_sha256']
    for path in [inspect.getfile(wrist.PactPlaceCorridorV1010FourObjectSampler),inspect.getfile(frontend.PactFrontendScreenInferencePolicy)]:
        assert sha(path)==config['input_hashes'][str(Path(path).resolve())]
    assert sha(ENCODER_PATH)==ENCODER_SHA256
    wrist.OUTPUT=OUTPUT;wrist.ACTIVE_ROW=row;wrist.HISTORY=100
    v109._ACTIVE_SCENE=str(scene_path);v109._ACTIVE_SCENE_SHA256=row['pact_v106_scene_sha256'];v109._ACTIVE_OUTPUT_DIR=str(OUTPUT)
    exp=Wrist1010bEvalConfig(output_dir=OUTPUT,num_workers=1,
        policy_config=Wrist1010bPolicyConfig(arm=args.arm,checkpoint_dir=str(checkpoint.parent),checkpoint_path=str(checkpoint),
            stats_path=str(stats),checkpoint_seed=args.training_seed,surface_encoder_path=str(ENCODER_PATH),
            sensor_names=tuple(CANONICAL_SENSOR_NAMES),blur_sigma=0.,blind_rgb=False))
    assert exp.task_horizon==900 and not exp.end_on_success and not exp.robot_config.action_noise_config.enabled
    sampler=task=policy=None;failures=[]; selected=None; retry=None
    try:
        retries=[scene['selected_retry']] if scene['selected_retry'] is not None else list(range(13))
        for retry in retries:
            seed={k:row['sampling_seeds'][retry][k] for k in ('seed_u32','seed_u64')}
            sampler=wrist.PactPlaceCorridorV1010FourObjectSampler(exp)
            sampler.seed_task_sampling(seed['seed_u32']);sampler.set_pact_manifest_row(row)
            try:task=sampler.sample_task(house_index=int(row['scene_template_house_index']))
            except legacy.HouseInvalidForTask as exc:
                failures.append({'retry_index':retry,'seed':seed,'reason':str(exc)})
                sampler.close();sampler=None
                if scene['selected_retry'] is not None:raise RuntimeError('retained successful sampling seed failed replay') from exc
                continue
            if task is None:
                raise RuntimeError('sample_task returned None; do not redraw infrastructure failures')
            policy=Wrist1010bInferencePolicy(exp,task);task.register_policy(policy);policy.prepare_model()
            initial=task.reset();selected=seed;break
        assert selected is not None and task is not None and policy is not None
        assert abs(task.env.current_model.opt.timestep-.002)<1e-12
        assert task._n_sim_steps_per_ctrl==1,'2ms instrumentation callback contract changed'
        boundary={'schema':SCHEMA,'scene_id':args.scene_id,'physical_row_digest':scene['physical_row_digest'],
            'selected_retry':retry,'selected_seed':selected,'model_provenance':ACTIVE,'binding_path':verify_bindings()}
        freeze(OUTPUT/'initial_observation_accepted.json',boundary)
        success=bool(legacy.ParallelRolloutRunner.run_single_rollout(episode_seed=selected['seed_u64'],task=task,
            policy=policy,end_on_success=False,initial_reset_result=initial))
        info=policy.get_info();audit=info['pact_contact_audit']
        result={'schema_version':SCHEMA,'status':'complete','arm':args.arm,'episode_id':args.scene_id,
            'rollout_id':args.job_id,'schedule_row_sha256':args.job_sha256,'candidate_index':row['candidate_index'],
            'row_sha256':row['row_sha256'],'manifest_sha256':manifest['manifest_sha256'],'intrusion_side':row['intrusion_side'],
            'sampling_retry_index':retry,'sampling_retry_history':failures,'seed':selected,
            'checkpoint_seed':args.training_seed,'checkpoint_sha256':args.checkpoint_sha256,'stats_sha256':sha(stats),
            'surface_encoder_sha256':ENCODER_SHA256 if args.arm=='PACT' else None,'blur_sigma':0.,'blind_rgb':False,
            'initial_observation_accepted':True,'task_success':success,
            'collision_free_task_success':bool(success and audit['collision_free']),'contact_audit':audit,
            'failure_taxonomy':legacy.failure_taxonomy(task_success=success,contact_audit=audit,gripper_close_commanded=info['gripper_close_commanded']),
            'policy_info':info,'scene_id':args.scene_id,'scene_block':args.scene_block,'training_seed':args.training_seed,
            'variant':args.variant,'physical_row_digest':scene['physical_row_digest'],'instrumentation':args.instrumentation}
        full,videos=v109._ORIGINAL_PUBLISH_EPISODE(row_dir=OUTPUT,task=task,config=exp,save_videos=False,row=row,
            manifest_sha256=manifest['manifest_sha256'],result=result)
        assert not videos and Path(full)==OUTPUT/'trajectory.h5'
        storage=repack(full)
        result.update(trajectory_path=str(full),videos=[],trajectory_retention={'mode':'full_h5_lossless_gzip4',
            'full_h5_sha256':storage['final_sha256'],'all_proximity_frames_retained':True,
            'physics_contact_and_control_stability_retained':True,'storage_provenance_sha256':sha(OUTPUT/'storage_provenance.json')})
        freeze(OUTPUT/'provenance.json',{'schema':SCHEMA,'model':ACTIVE,'physical':{'row_digest':digest(row),'selected_seed':selected,'selected_retry':retry,
            'scene_sha256':sha(scene_path),'sampler_sha256':sha(inspect.getfile(wrist.PactPlaceCorridorV1010FourObjectSampler))},
            'contract_sha256':config['contract_sha256'],'worker_sha256':sha(__file__)})
        freeze(OUTPUT/'result.json',result)
        print(json.dumps({'job_id':args.job_id,'status':'complete','task_success':success,'result_sha256':sha(OUTPUT/'result.json')}),flush=True)
        return 0
    finally:
        legacy.cleanup_episode_resources(task=task,policy=policy,task_sampler=sampler,preloaded_policy=None,close_task_sampler=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=('ACT','PACT'),required=True)
    p.add_argument('--training-seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--scene-id',required=True);p.add_argument('--scene-block',type=int,required=True)
    p.add_argument('--checkpoint-path',required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--instrumentation',choices=('none','grasp'),required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--job-id',required=True);p.add_argument('--job-sha256',required=True)
    return run(p.parse_args())

if __name__=='__main__':raise SystemExit(main())
