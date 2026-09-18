#!/usr/bin/env python3
"""Frozen PACT worker; idealized native 16x16 sensing with 8x8 minimum adapter, original controller."""
from __future__ import annotations
import argparse
import hashlib
import os
import pickle
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
import pact_v1010c_vl53l8ux_16x16_128d as study
import pact_v1010c_core as core
os.environ.update(core.environment())
import pact_v1010c_eval as original
from pact_vl53l8ux_16x16_sensor import MultizoneProfile, compare_initial
import h5py
import numpy as np
import torch

JOB = SCENE = OUTPUT = None


class NonfinitePolicyOutput(RuntimeError):
    """Scientific failure; never repeat a completed policy failure."""


class TransferPolicy(original.ReadoutInferencePolicy):
    def reset(self):
        super().reset()
        self._initial_audited = False
        self._last_progress = 0.
        self._raw_digest = hashlib.sha256()
        self._native_audit = None
        profile = getattr(self, '_sensor_profile', None)
        if profile is not None: profile.reset()

    def prepare_model(self):
        super().prepare_model()
        self._policy.eval().requires_grad_(False)
        self._surface_encoder.eval().requires_grad_(False)
        self._weight_hashes = (state_hash(self._policy), state_hash(self._surface_encoder))

    def _surface_positions(self, raw):
        profile = getattr(self, '_sensor_profile', None)
        if profile is not None: profile.check_consumed(raw, self._step)
        self._raw_digest.update(np.ascontiguousarray(raw).tobytes())
        return super()._surface_positions(raw)

    def inference_model(self, observation):
        output = super().inference_model(observation)
        if not self._initial_audited:
            reference = dict(directory=SCENE['live_directory'], scene_id=SCENE['scene_id'],
                physical_row_digest=study.digest(SCENE['physical_row']), task_seed=SCENE['selected_seed'])
            if JOB['experiment_arm'] == 'LIVE_PREFLIGHT':
                original.pairing.audit_initial_group([reference, {**reference, 'directory':str(OUTPUT)}],
                    OUTPUT/'initial_pairing.json')
            else:
                src = Path(SCENE['live_directory'])/'initial_observation.h5'
                dst = OUTPUT/'initial_observation.h5'
                detail = compare_initial(src, dst, list(core.CANONICAL_SENSOR_NAMES))
                study.freeze(OUTPUT/'initial_pairing.json', dict(detail,
                    scene_id=SCENE['scene_id'], reference=str(src), comparison=str(dst),
                    reference_sha256=study.sha(src), comparison_sha256=study.sha(dst)))
                study.require(detail['passed'], 'physical/RGB/undeclared-observation mismatch')
            self._initial_audited = True
        if not np.isfinite(output).all():
            raise NonfinitePolicyOutput(f'nonfinite policy output at control step {self._step}')
        if time.monotonic()-self._last_progress > 30:
            study.atomic(OUTPUT/'progress.json', dict(utc=study.now(), control_step=self._step,
                readout_frames=self._readout_frames, projection_checks=self._proximity_hook_calls,
                initial_pairing_passed=self._initial_audited,
                multizone=getattr(self, '_sensor_profile', None) is not None))
            self._last_progress = time.monotonic()
        return output

    def get_info(self):
        info = super().get_info()
        study.require(self._readout_frames == self._proximity_hook_calls == 900, 'incomplete encoder delivery')
        study.require(self._weight_hashes == (state_hash(self._policy), state_hash(self._surface_encoder)),
            'inference mutated policy or encoder weights')
        profile = getattr(self, '_sensor_profile', None)
        if profile is not None and self._native_audit is None:
            self._native_audit = profile.save_audit(OUTPUT/'native_multizone.h5')
        native = self._native_audit
        info.update(experiment_arm=JOB['experiment_arm'], inference_only=True, unchanged_weights=True,
            sensor_profile_sha256=JOB['sensor_profile_sha256'], native_sensor_audit=native,
            input_tensor_sha256=self._raw_digest.hexdigest(), initial_pairing_passed=self._initial_audited,
            encoder_checkpoint_actually_loaded=str(study.SOURCE/'checkpoint/prox_encoder.pt'))
        return info


class TransferPolicyConfig(original.ReadoutPolicyConfig):
    policy_cls: type = TransferPolicy


def bind(job):
    global JOB, SCENE, OUTPUT
    JOB = job
    study.require(job['implementation_hashes'] == study.implementation_hashes(), 'worker code drift')
    study.verify_models()
    source = study.checked_json(study.OUT/'source_manifest.json')
    profile = study.checked_json(study.OUT/'sensor_profile.json')
    study.require(source['sha256'] == job['source_manifest_sha256'] and
        profile['sha256'] == job['sensor_profile_sha256'], 'job inputs changed')
    SCENE = source['scenes'][job['scene_index']]
    study.require(job == study.make_job(SCENE, job['experiment_arm'], job['attempt'], job['protocol_sha256']),
        'job does not match frozen scenario/profile')
    for name, expected in SCENE['artifacts'].items():
        study.require(study.sha(Path(SCENE['live_directory'])/name) == expected, 'live evidence drift')
    if job['experiment_arm'] in study.ARMS:
        protocol = study.checked_json(study.OUT/'protocol.json')
        study.require(protocol['sha256'] == job['protocol_sha256'] and
            protocol['implementation_hashes'] == job['implementation_hashes'] and
            protocol['sensor_profile_sha256'] == job['sensor_profile_sha256'], 'unfrozen scientific invocation')
    else: study.require(job['experiment_arm'] == 'LIVE_PREFLIGHT', 'unknown arm')
    OUTPUT = study.inside(job['directory'])
    study.require(not (OUTPUT/'result.json').exists(), 'attempt already has outcome')
    original.ACTIVE=job; original.OUTPUT=OUTPUT
    original.storage.inside=study.inside; original.pairing.inside=study.inside
    row=SCENE['physical_row']; scene_path=study.ROOT/row['pact_v1010_scene_relative']
    study.require(study.sha(scene_path)==row['pact_v106_scene_sha256'],'scene XML drift')
    original.wrist.OUTPUT=OUTPUT; original.wrist.ACTIVE_ROW=row; original.wrist.HISTORY=100
    original.v109._ACTIVE_SCENE=str(scene_path)
    original.v109._ACTIVE_SCENE_SHA256=row['pact_v106_scene_sha256']
    original.v109._ACTIVE_OUTPUT_DIR=str(OUTPUT)


def save_policy_failure(policy,error):
    audit=policy._contact_audit
    with h5py.File(OUTPUT/'partial_telemetry.h5','x') as h:
        for name,values in [('sim_time_s',audit.raw_times),('control_step',audit.raw_steps),
            ('class_entries',audit.raw_classes),('slot_entries',audit.raw_slots),
            ('pair_counts',np.asarray(audit.raw_pair_counts).reshape(-1,3))]:
            h.create_dataset(name,data=np.asarray(values),compression='gzip')
        h.attrs['class_names']=study.json.dumps(original.wrist.CLASSES)
        h.attrs['pair_identities']=study.json.dumps(audit.pair_identities)
        h.attrs['censored']=True
    np.savez_compressed(OUTPUT/'partial_actions.npz',model_outputs=np.asarray(policy._v109_model_outputs),
        arm=np.asarray(policy._v109_arm),gripper=np.asarray(policy._v109_gripper))
    study.freeze(OUTPUT/'result.json',dict(schema_version=study.SCHEMA,status='policy_failure',
        experiment_arm=JOB['experiment_arm'],scene_id=SCENE['scene_id'],schedule_row_sha256=JOB['sha256'],
        task_success=False,collision_free_task_success=False,contact_exposure_censored=True,
        control_steps=policy._step,failure=str(error),checkpoint_sha256=JOB['checkpoint_sha256'],
        encoder_sha256=JOB['encoder_sha256'],sensor_profile_sha256=JOB['sensor_profile_sha256']))

def run(job_path):
    job=study.checked_json(job_path); bind(job)
    row=SCENE['physical_row']; checkpoint=study.SOURCE/'checkpoint/policy_last.ckpt'
    exp=original.wrist.Wrist288EvalConfig(output_dir=OUTPUT,num_workers=1,
        policy_config=TransferPolicyConfig(arm='PACT',checkpoint_dir=str(checkpoint.parent),
            checkpoint_path=str(checkpoint),stats_path=str(checkpoint.parent/'dataset_stats.pkl'),
            checkpoint_seed=3103,surface_encoder_path=str(checkpoint.parent/'prox_encoder.pt'),
            sensor_names=tuple(core.CANONICAL_SENSOR_NAMES),blur_sigma=0.,blind_rgb=False))
    study.require(exp.task_horizon==900 and not exp.end_on_success and
        not exp.robot_config.action_noise_config.enabled,'execution configuration drift')
    sampler=task=policy=profile=None
    try:
        retry,selected=SCENE['selected_retry'],SCENE['selected_seed']
        study.require(selected=={k:row['sampling_seeds'][retry][k] for k in ('seed_u32','seed_u64')},
            'sampling retry mismatch')
        sampler=original.wrist.PactPlaceCorridorV1010FourObjectSampler(exp)
        sampler.seed_task_sampling(selected['seed_u32']); sampler.set_pact_manifest_row(row)
        task=sampler.sample_task(house_index=int(row['scene_template_house_index']))
        study.require(task is not None,'historical scenario failed to replay')
        policy=TransferPolicy(exp,task); task.register_policy(policy); policy.prepare_model()
        if job['experiment_arm'] in study.ARMS:
            profile=MultizoneProfile(task,core.CANONICAL_SENSOR_NAMES)
            policy._sensor_profile=profile
        initial=task.reset()
        study.require(abs(task.env.current_model.opt.timestep-.002)<1e-12 and task._n_sim_steps_per_ctrl==1,
            'physics frequency changed')
        try:
            success=bool(original.legacy.ParallelRolloutRunner.run_single_rollout(
                episode_seed=selected['seed_u64'],task=task,policy=policy,end_on_success=False,initial_reset_result=initial))
        except NonfinitePolicyOutput as exc:
            save_policy_failure(policy,exc); return
        info=policy.get_info(); audit=info['pact_contact_audit']
        result=dict(schema_version=study.SCHEMA,status='complete',arm='PACT',
            experiment_arm=job['experiment_arm'],episode_id=SCENE['scene_id'],rollout_id=job['id'],
            schedule_row_sha256=job['sha256'],candidate_index=row['candidate_index'],row_sha256=row['row_sha256'],
            manifest_sha256=job['source_manifest_sha256'],intrusion_side=row['intrusion_side'],
            sampling_retry_index=retry,sampling_retry_history=[],seed=selected,checkpoint_seed=3103,
            checkpoint_sha256=job['checkpoint_sha256'],stats_sha256=study.EXPECTED['dataset_stats.pkl'],
            surface_encoder_sha256=job['encoder_sha256'],blur_sigma=0.,blind_rgb=False,
            initial_observation_accepted=True,task_success=success,
            collision_free_task_success=bool(success and audit['collision_free']),contact_audit=audit,
            failure_taxonomy=original.legacy.failure_taxonomy(task_success=success,contact_audit=audit,
                gripper_close_commanded=info['gripper_close_commanded']),policy_info=info,
            scene_id=SCENE['scene_id'],scene_block=3103,training_seed=3103,variant='readout60000',
            physical_row_digest=study.digest(row),protocol_sha256=job['protocol_sha256'],
            sensor_profile_sha256=job['sensor_profile_sha256'])
        full,videos=original.v109._ORIGINAL_PUBLISH_EPISODE(row_dir=OUTPUT,task=task,config=exp,
            save_videos=False,row=row,manifest_sha256=job['source_manifest_sha256'],result=result)
        study.require(not videos and Path(full)==OUTPUT/'trajectory.h5','unexpected output')
        receipt=original.storage.repack(full)
        result.update(trajectory_path=str(full),videos=[],trajectory_retention=dict(mode='full_h5_lossless_gzip4',
            full_h5_sha256=receipt['final_sha256'],all_proximity_frames_retained=True,
            physics_contact_and_control_stability_retained=True))
        study.freeze(OUTPUT/'result.json',result)
        study.freeze(OUTPUT/'metrics.json',study.audit_metrics(OUTPUT))
        study.verify_models()
        print(study.json.dumps(dict(id=job['id'],status='complete',task_success=success,
            strict=result['collision_free_task_success'],projection_checks=policy._proximity_hook_calls)),flush=True)
    finally:
        if profile is not None: profile.close()
        original.legacy.cleanup_episode_resources(task=task,policy=policy,task_sampler=sampler,
            preloaded_policy=None,close_task_sampler=True)

def state_hash(module):
    h=hashlib.sha256()
    for name,tensor in module.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def observation_from_h5(group):
    result={}
    for key,value in group.items():
        if isinstance(value,h5py.Group): result[key]=observation_from_h5(value)
        else:
            data=value[()]
            if isinstance(data,bytes):
                try: data=study.json.loads(data)
                except (ValueError,UnicodeDecodeError): data=data.decode()
            result[key]=data
    return result


def static_sensor_check(source):
    """Ten infrastructure steps, excluded from scientific outcomes/selection."""
    global JOB, SCENE, OUTPUT
    SCENE=source['scenes'][0]; row=SCENE['physical_row']
    OUTPUT=study.OUT/'preflight/sensor_interface'; OUTPUT.mkdir(parents=True,exist_ok=False)
    JOB=dict(variant='readout60000',experiment_arm='SENSOR_INTERFACE_PREFLIGHT',
        checkpoint_sha256=study.EXPECTED['policy_last.ckpt'],encoder_sha256=study.EXPECTED['prox_encoder.pt'])
    original.ACTIVE=JOB; original.OUTPUT=OUTPUT
    original.storage.inside=study.inside; original.pairing.inside=study.inside
    original.wrist.OUTPUT=OUTPUT; original.wrist.ACTIVE_ROW=row; original.wrist.HISTORY=100
    original.v109._ACTIVE_SCENE=str(study.ROOT/row['pact_v1010_scene_relative'])
    original.v109._ACTIVE_SCENE_SHA256=row['pact_v106_scene_sha256']
    original.v109._ACTIVE_OUTPUT_DIR=str(OUTPUT)
    checkpoint=study.SOURCE/'checkpoint'
    exp=original.wrist.Wrist288EvalConfig(output_dir=OUTPUT,num_workers=1,
        policy_config=TransferPolicyConfig(arm='PACT',checkpoint_dir=str(checkpoint),
            checkpoint_path=str(checkpoint/'policy_last.ckpt'),stats_path=str(checkpoint/'dataset_stats.pkl'),
            checkpoint_seed=3103,surface_encoder_path=str(checkpoint/'prox_encoder.pt'),
            sensor_names=tuple(core.CANONICAL_SENSOR_NAMES),blur_sigma=0.,blind_rgb=False))
    sampler=task=policy=profile=None
    try:
        sampler=original.wrist.PactPlaceCorridorV1010FourObjectSampler(exp)
        sampler.seed_task_sampling(SCENE['selected_seed']['seed_u32']); sampler.set_pact_manifest_row(row)
        task=sampler.sample_task(house_index=int(row['scene_template_house_index']))
        study.require(task is not None,'historical infrastructure scenario failed to replay')
        policy=TransferPolicy(exp,task); task.register_policy(policy); policy.prepare_model()
        profile=MultizoneProfile(task,core.CANONICAL_SENSOR_NAMES); policy._sensor_profile=profile
        observation,_=task.reset()
        for _ in range(10):
            action=policy.get_action(observation)
            observation,*_=task.step(action)
        _, _, native_audit = profile.audit_arrays(10)
        study.require(policy._initial_audited and native_audit['native_acquisitions_per_sensor']==10,
            'static sensor integration audit failed')
        study.require(policy._weight_hashes==(state_hash(policy._policy),state_hash(policy._surface_encoder)),
            'preflight changed weights')
        study.freeze(OUTPUT/'report.json',dict(passed=True,control_steps=10,
            native_sensor_audit=native_audit,
            initial_pairing_sha256=study.sha(OUTPUT/'initial_pairing.json'),
            scene_id=SCENE['scene_id'],excluded_from_scientific_results=True,
            no_outcome_used_for_selection=True))
        return study.sha(OUTPUT/'report.json')
    finally:
        if profile is not None: profile.close()
        original.legacy.cleanup_episode_resources(task=task,policy=policy,task_sampler=sampler,
            preloaded_policy=None,close_task_sampler=True)


def offline_check():
    source, _, _ = study.verify_inputs()
    model, encoder, pair = core.load_pair(study.SOURCE/'checkpoint')
    model.eval().requires_grad_(False); encoder.eval().requires_grad_(False)
    before = (state_hash(model), state_hash(encoder))
    with h5py.File(study.DATA/'converted/episode_0.hdf5', 'r') as h:
        training_raw = h['observations/proximity'][:105]
    with h5py.File(Path(source['scenes'][0]['live_directory'])/'initial_observation.h5', 'r') as h:
        observation = observation_from_h5(h['observation'])
    observation['qpos'] = {k:np.asarray(v).tolist() for k,v in observation['qpos'].items()}
    stats = pickle.loads((study.SOURCE/'checkpoint/dataset_stats.pkl').read_bytes())
    original.ACTIVE = {'variant':'readout60000'}
    def construct(cls):
        p = cls.__new__(cls); p.task=SimpleNamespace(); p.reset()
        p.pc=SimpleNamespace(arm='PACT',blur_sigma=0.,blind_rgb=False,sensor_names=list(core.CANONICAL_SENSOR_NAMES))
        p._policy,p._surface_encoder,p._stats=model,encoder,stats
        p._proximity_feature_dim=128; p._input_proj_proximity_shape=[512,128]
        p._initial_written=p._initial_audited=True; p._last_progress=float('inf')
        p.get_phase=lambda:'offline_parity'
        return p
    captured=[]
    hook=model.model.input_proj_proximity.register_forward_pre_hook(
        lambda module,inputs:captured.append(inputs[0].detach().cpu().clone()))
    reference, identity=construct(original.ReadoutInferencePolicy),construct(TransferPolicy)
    checks=[]
    with torch.inference_mode():
        for step,raw in enumerate(training_raw):
            observation.update(dict(zip(core.CANONICAL_SENSOR_NAMES,raw)))
            reference._step=identity._step=step
            a=reference.inference_model(observation); b=identity.inference_model(observation)
            study.require(np.array_equal(a,b) and torch.equal(captured[-2],captured[-1]),
                f'live wrapper differs from source controller at {step}')
            checks.append(dict(step=step,action_exact=True,readout_exact=True))
        first=[]
        for p in (reference,identity):
            p.reset(); p._initial_written=p._initial_audited=True; p._last_progress=float('inf')
            observation.update(dict(zip(core.CANONICAL_SENSOR_NAMES,training_raw[0]))); p._step=0
            first.append(p.inference_model(observation))
        study.require(np.array_equal(*first), 'reset parity failed')
        # Exercise spatial grids with repeated polls without learning/adapting weights.
        from pact_vl53l8ux_16x16_sensor import encode_zones, policy_grid
        grid_raw=np.stack([np.stack([np.repeat(policy_grid(encode_zones(np.repeat(np.repeat(sensor[0],2,axis=0),2,axis=1))[0])[None],4,axis=0)
                                    for sensor in frame]) for frame in training_raw[:10]])
        p=construct(TransferPolicy)
        for step,raw in enumerate(grid_raw):
            p._step=step; observation.update(dict(zip(core.CANONICAL_SENSOR_NAMES,raw)))
            action=p.inference_model(observation)
            expected=core.encode_for_act(encoder,torch.from_numpy(core.causal_pooled_window(
                grid_raw[:step+1].min(axis=2),step)).cuda().unsqueeze(0))
            study.require(torch.equal(captured[-1],expected.cpu()) and np.isfinite(action).all(),
                '16x16-to-8x8 held-frame history failed frozen-encoder compatibility')
    hook.remove()
    study.require(before==(state_hash(model),state_hash(encoder)), 'inference changed weights')
    sensor_check_sha256=static_sensor_check(source)
    study.verify_models()
    study.freeze(study.OUT/'preflight/offline.json', dict(passed=True,
        implementation_hashes=study.implementation_hashes(), checkpoint_pair=pair,
        identity_action_history_checks=checks, reset_exact=True, held_grid_history_checks=10,
        unchanged_policy=True, unchanged_encoder=True, no_gradients=True, sensor_interface_sha256=sensor_check_sha256,
        note='Training frames and source initial RGB/qpos used only for controller/interface checks; no shifted rollout outcomes'))
    print('Offline controller identity, 105-step history/aggregation, held 8x8 input and fixed-weight checks passed.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--job',type=Path); group.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    if args.offline:
        offline_check(); return
    try: run(args.job)
    except BaseException as exc:
        retryable=isinstance(exc,(MemoryError,torch.cuda.OutOfMemoryError)) or (
            isinstance(exc,OSError) and exc.errno in (5,11,12))
        directory=Path(study.read(args.job)['directory'])
        study.freeze(directory/'failure.json',dict(utc=study.now(),type=type(exc).__name__,message=str(exc),
            retryable=retryable,traceback=traceback.format_exc()))
        raise


if __name__=='__main__': main()
