"""Strict reload and actual input-consumption checks on both retained milestones."""
from pact_place_v1011c_dualcam import *
from verify_pact_place_v1011c_models import detr_argv
import pickle
import h5py
import numpy as np
import torch
from policy import ACTPolicy
from fixed_split_data import compute_train_only_norm_stats


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    split = read(WORK/'split_manifest.json')
    train_ids = [r['act_episode_index'] for r in split['episodes'] if r['split']=='train']
    val_ids = sorted(r['act_episode_index'] for r in split['episodes'] if r['split']=='validation')[:4]
    reference_stats, metadata = compute_train_only_norm_stats(str(DATA),train_ids)
    assert metadata['contributing_timesteps'] == 28564
    images,qpos,proximity = [],[],[]
    for i in val_ids:
        with h5py.File(DATA/f'episode_{i}.hdf5','r') as h:
            images.append(np.stack([h[f'observations/images/{c}'][40] for c in CAMERAS]))
            qpos.append(h['observations/qpos'][40])
            proximity.append(h['observations/proximity_embeddings'][40])
    image = torch.from_numpy(np.stack(images)).float().permute(0,1,4,2,3).cuda()/255
    q = torch.from_numpy((np.stack(qpos)-reference_stats['qpos_mean'])/reference_stats['qpos_std']).float().cuda()
    prox = torch.from_numpy(np.stack(proximity)).float().cuda()
    checks = {}
    for arm in ('act','pact'):
        base = TRAIN/f'{arm}_seed3103'
        epochs = [json.loads(x) for x in (base/'epoch_log.jsonl').read_text().splitlines() if x.strip()]
        assert [r['global_step'] for r in epochs] == list(range(10,30001,10))
        with (base/'dataset_stats.pkl').open('rb') as f:
            stats = pickle.load(f)
        for key,value in reference_stats.items():
            assert np.array_equal(value,stats[key]), (arm,key)
        config = read(base/'run_manifest.json')['policy_config']
        assert config['camera_names'] == CAMERAS and config['num_queries'] == 100
        assert config['n_proximity_sensors'] == (40 if arm=='pact' else 0)
        with detr_argv(str(base),3103):
            policy = ACTPolicy(config).cuda().eval()
        for step in MILESTONES:
            directory = TRAIN/f'{arm}_seed3103_step{step}'
            receipt = read(directory/'checkpoint_step.json')
            assert receipt['checkpoint_sha256'] == sha256_file(directory/'policy_best.ckpt')
            policy.load_state_dict(torch.load(directory/'policy_best.ckpt',map_location='cpu',weights_only=True),strict=True)
            pp = prox if arm=='pact' else None
            with torch.inference_mode():
                prediction = policy(q,image,proximity_positions=pp)
                repeat = policy(q,image,proximity_positions=pp)
                camera_deltas = {}
                for camera_index,camera in enumerate(CAMERAS):
                    masked = image.clone()
                    masked[:,camera_index] = 0
                    alternative = policy(q,masked,proximity_positions=pp)
                    camera_deltas[camera] = float((alternative-prediction).abs().max())
                    assert camera_deltas[camera] > 1e-5, ('unused camera',arm,step,camera)
                prox_delta = None
                if arm=='pact':
                    alternative = policy(q,image,proximity_positions=torch.zeros_like(prox))
                    prox_delta = float((alternative-prediction).abs().max())
                    assert prox_delta > 1e-5
                    assert list(policy.model.input_proj_proximity.weight.shape) == [512,32]
            assert prediction.shape == (4,100,8) and bool(torch.isfinite(prediction).all())
            assert torch.equal(prediction,repeat)
            checks[f'{arm}_step{step}'] = {'strict_reload':True,'output_shape':list(prediction.shape),
                'finite':True,'deterministic':True,'max_action_delta_camera_zeroed':camera_deltas,
                'max_action_delta_proximity_zeroed':prox_delta,'checkpoint_sha256':receipt['checkpoint_sha256'],
                'train_only_stats_recomputed_equal':True,'actual_training_updates':30000}
            print(arm,step,json.dumps(checks[f'{arm}_step{step}']),flush=True)
        del policy
        torch.cuda.empty_cache()
    freeze(WORK/'model_verification.json',{**empty_authorization(),'verified':True,
        'checks':checks,'fixed_validation_episode_indices':val_ids,
        'limitation':'Input sensitivity verifies wiring, not useful sensing or task performance.'})


if __name__ == '__main__':
    main()
