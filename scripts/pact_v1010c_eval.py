"""Fine-tuned or original PACT on the unchanged seed3103 physical evaluation."""
from __future__ import annotations
import argparse
import collections
import inspect
from pact_v1010c_core import *
os.environ.update(environment())
import pact_wrist288_eval_worker as wrist
import eval_pact_collision_row as legacy
import eval_pact_place_row as place
import eval_pact_place_v109_row as v109
import eval_pact_frontend_screen_row as frontend

ACTIVE = None
OUTPUT = None


def inside(path):
    path = Path(path).resolve()
    assert path.is_relative_to(C), path
    return path


def scoped_module(file, name):
    spec = importlib.util.spec_from_file_location(name, CODE/file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.inside = inside
    module.B = C
    return module


storage = scoped_module('pact_v1010b_storage.py','v1010c_storage')
pairing = scoped_module('pact_v1010b_pairing.py','v1010c_pairing')


class ReadoutInferencePolicy(wrist.Wrist288InferencePolicy):
    def reset(self):
        super().reset()
        self._readout_history = collections.deque(maxlen=8)
        self._readout_frames = 0

    def prepare_model(self):
        if ACTIVE['variant']=='frozen60000':
            return super().prepare_model()
        assert ACTIVE['variant']=='readout60000'
        self._policy, self._surface_encoder, pair = load_pair(C/'checkpoint')
        assert pair['policy_sha256']==ACTIVE['checkpoint_sha256']
        assert pair['encoder_sha256']==ACTIVE['encoder_sha256']
        self._stats = pickle.loads((C/'checkpoint/dataset_stats.pkl').read_bytes())
        self._proximity_feature_dim = 128
        self._input_proj_proximity_shape = list(self._policy.model.input_proj_proximity.weight.shape)
        assert self._input_proj_proximity_shape==[512,128]
        if not hasattr(self,'_readout_hook'):
            def consumed(module, inputs, output):
                assert inputs[0].shape[-2:]==(40,128) and torch.isfinite(inputs[0]).all()
                self._proximity_hook_calls += 1
            self._readout_hook=self._policy.model.input_proj_proximity.register_forward_hook(consumed)

    def _surface_positions(self, raw):
        if ACTIVE['variant']=='frozen60000':
            return super()._surface_positions(raw)
        pooled=stack_obs_proximity(dict(zip(CANONICAL_SENSOR_NAMES,raw)),
                                   list(CANONICAL_SENSOR_NAMES),pool='min')
        self._readout_history.append(pooled.copy())
        self._readout_frames += 1
        assert self._readout_frames==self._step+1
        history=causal_pooled_window(np.stack(self._readout_history),len(self._readout_history)-1)
        with torch.inference_mode():
            return encode_for_act(self._surface_encoder,
                                  torch.from_numpy(history).cuda().unsqueeze(0))

    def get_info(self):
        info=super().get_info()
        if ACTIVE['variant']=='readout60000':
            assert self._readout_frames==900
            info.update(frontend_variant='main_finetuned_readout128',
                        proximity_feature_dim=128,finetune_prox_encoder=True,
                        prox_policy_tap='readout',prox_pool='min',
                        encoder_sha256=ACTIVE['encoder_sha256'],
                        consecutive_proximity_history_frames=self._readout_frames)
        info['v1010c_variant']=ACTIVE['variant']
        info['v1010c_binding_path']=[x.__name__ for x in type(self).__mro__]
        return info


class ReadoutPolicyConfig(wrist.Wrist288PolicyConfig):
    policy_cls:type=ReadoutInferencePolicy


def run(job_path):
    global ACTIVE,OUTPUT
    job=read(job_path)
    assert job['sha256']==digest({k:v for k,v in job.items() if k!='sha256'})
    OUTPUT=inside(job['directory'])
    OUTPUT.mkdir(parents=True,exist_ok=True)
    assert not (OUTPUT/'result.json').exists()
    contract=read(C/'contract.json')
    for path, expected in contract['code_hashes'].items():
        assert sha(ROOT/path)==expected,path
    manifest=read(C/'evaluation_manifest.json')
    scene=next(s for s in manifest['scenes'] if s['scene_id']==job['scene_id'])
    row=scene['physical_row']
    checkpoint=Path(job['checkpoint_path'])
    assert sha(checkpoint)==job['checkpoint_sha256']
    stats=checkpoint.parent/'dataset_stats.pkl'
    assert sha(stats)==sha(W/'dataset_stats.pkl')
    assert sha(job['encoder_path'])==job['encoder_sha256']
    ACTIVE=job
    scene_path=ROOT/row['pact_v1010_scene_relative']
    assert sha(scene_path)==row['pact_v106_scene_sha256']
    wrist.OUTPUT=OUTPUT;wrist.ACTIVE_ROW=row;wrist.HISTORY=100
    v109._ACTIVE_SCENE=str(scene_path)
    v109._ACTIVE_SCENE_SHA256=row['pact_v106_scene_sha256']
    v109._ACTIVE_OUTPUT_DIR=str(OUTPUT)
    exp=wrist.Wrist288EvalConfig(output_dir=OUTPUT,num_workers=1,
        policy_config=ReadoutPolicyConfig(arm='PACT',checkpoint_dir=str(checkpoint.parent),
            checkpoint_path=str(checkpoint),stats_path=str(stats),checkpoint_seed=3103,
            surface_encoder_path=str(ENCODER_PATH),sensor_names=tuple(CANONICAL_SENSOR_NAMES),
            blur_sigma=0.,blind_rgb=False))
    assert exp.task_horizon==900 and not exp.end_on_success and not exp.robot_config.action_noise_config.enabled
    sampler=task=policy=None
    try:
        retry=scene['selected_retry'];selected=scene['selected_seed']
        assert selected=={k:row['sampling_seeds'][retry][k] for k in ('seed_u32','seed_u64')}
        sampler=wrist.PactPlaceCorridorV1010FourObjectSampler(exp)
        sampler.seed_task_sampling(selected['seed_u32']);sampler.set_pact_manifest_row(row)
        task=sampler.sample_task(house_index=int(row['scene_template_house_index']))
        assert task is not None, 'retained successful scene failed replay'
        policy=ReadoutInferencePolicy(exp,task);task.register_policy(policy);policy.prepare_model()
        initial=task.reset()
        assert abs(task.env.current_model.opt.timestep-.002)<1e-12
        assert task._n_sim_steps_per_ctrl==1
        success=bool(legacy.ParallelRolloutRunner.run_single_rollout(episode_seed=selected['seed_u64'],
            task=task,policy=policy,end_on_success=False,initial_reset_result=initial))
        info=policy.get_info();audit=info['pact_contact_audit']
        result={'schema_version':SCHEMA,'status':'complete','arm':'PACT','episode_id':scene['scene_id'],
            'rollout_id':job['id'],'schedule_row_sha256':job['sha256'],'candidate_index':row['candidate_index'],
            'row_sha256':row['row_sha256'],'manifest_sha256':manifest['sha256'],
            'intrusion_side':row['intrusion_side'],'sampling_retry_index':retry,'sampling_retry_history':[],
            'seed':selected,'checkpoint_seed':3103,'checkpoint_sha256':job['checkpoint_sha256'],
            'stats_sha256':sha(stats),'surface_encoder_sha256':job['encoder_sha256'],
            'blur_sigma':0.,'blind_rgb':False,'initial_observation_accepted':True,'task_success':success,
            'collision_free_task_success':bool(success and audit['collision_free']),'contact_audit':audit,
            'failure_taxonomy':legacy.failure_taxonomy(task_success=success,contact_audit=audit,
                gripper_close_commanded=info['gripper_close_commanded']),
            'policy_info':info,'scene_id':scene['scene_id'],'scene_block':3103,'training_seed':3103,
            'variant':job['variant'],'physical_row_digest':digest(row)}
        full,videos=v109._ORIGINAL_PUBLISH_EPISODE(row_dir=OUTPUT,task=task,config=exp,
            save_videos=False,row=row,manifest_sha256=manifest['sha256'],result=result)
        assert not videos and Path(full)==OUTPUT/'trajectory.h5'
        receipt=storage.repack(full)
        result.update(trajectory_path=str(full),videos=[],trajectory_retention={'mode':'full_h5_lossless_gzip4',
            'full_h5_sha256':receipt['final_sha256'],'all_proximity_frames_retained':True,
            'physics_contact_and_control_stability_retained':True})
        freeze(OUTPUT/'result.json',result)
        from pact_v1010b_metrics import recompute
        current=recompute(OUTPUT)
        old=recompute(scene['frozen_directory'])
        old['physical_row_digest']=digest(row)
        pairing.audit_initial_group([old,current],OUTPUT/'initial_pairing.json')
        freeze(OUTPUT/'metrics.json',current)
        print(json.dumps({'id':job['id'],'success':success,'result_sha256':sha(OUTPUT/'result.json')}),flush=True)
    finally:
        legacy.cleanup_episode_resources(task=task,policy=policy,task_sampler=sampler,
                                         preloaded_policy=None,close_task_sampler=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    run(parser.parse_args().job)
