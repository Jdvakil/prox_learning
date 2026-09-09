import torch
import numpy as np
import os
import sys
import datetime
import math
import pickle
import json
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from sim_env import BOX_POSE
except Exception:
    # sim_env pulls in dm_control, which is only needed for the upstream ALOHA
    # eval path (eval_bc). Training and the in-env obstacle eval never use it.
    BOX_POSE = [None]

import IPython
e = IPython.embed

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    experiment = None
    is_sim = task_name[:4] == 'sim_'
    if args.get('experiment_manifest'):
        from scripts.pact_workflow import load_contract, check_dataset_files, resolve
        experiment = load_contract(args['experiment_manifest'])
        check_dataset_files(experiment)
        if is_eval:
            raise ValueError('Use scripts/pact.py eval for contract-bound robot evaluation')
        is_sim = False
        task_config = dict(experiment['profile'])
        task_config.update(dataset_dir=str(resolve(task_config['data_dir'])),
                           num_episodes=len(experiment['episodes']),
                           episode_len=max(e['length'] for e in experiment['episodes']))
    elif is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # Robot dims. sim_* = ALOHA bimanual (14/14). Every TASK_CONFIGS row is Franka
    # FR3: qpos 9 (arm7 + 2 fingers), action 8 (arm7 + 1 gripper). Do not fall
    # through to 14 — that builds Linear(14, 512) and dies on hdf5 actions (T, 8)
    # with RuntimeError: 400x8 vs 14x512.
    if is_sim:
        state_dim = 14
        action_dim = 14
    else:
        state_dim = int(task_config.get('state_dim', 9))
        action_dim = int(task_config.get('action_dim', 8))
    print(
        f"[dims] task={task_name} state_dim={state_dim} action_dim={action_dim} "
        f"cameras={camera_names}"
    )
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim,
                         'action_dim': action_dim,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'state_dim': state_dim,
                         'action_dim': action_dim}
    else:
        raise NotImplementedError

    # P+ACT (PACT): load the skin encoder and switch on proximity tokens.
    # Frozen by default (CVAE taps / 32-d embedding). --finetune_prox_encoder
    # unfreezes the geometry stem and feeds 128-d CLS readout tokens at train
    # and eval. With --use_proximity OFF the model is bit-identical to vanilla ACT.
    use_proximity = args.get('use_proximity', False)
    if args.get('finetune_prox_encoder') and not use_proximity:
        raise SystemExit('[P+ACT] --finetune_prox_encoder needs --use_proximity')
    if is_eval and use_proximity:
        raise SystemExit(
            "[P+ACT] imitate_episodes.py --eval never feeds proximity_positions. "
            "Use eval_act_obstacle.py --temp_agg_off (obstacle pick) or "
            "eval_act_place_corridor.py --temp_agg_off (place-corridor). "
            "Refusing a fake PACT eval."
        )
    prox_encoder = None
    prox_cfg_json = None
    prox_layout_data = "raw"
    if use_proximity:
        if policy_class != 'ACT':
            raise NotImplementedError('proximity fusion is only implemented for ACT')
        from encoders.pact import (
            build_pact_encoder,
            hdf5_proximity_layout,
            is_geometry_feature,
        )
        prox_feature = args.get('prox_feature') or 'raw'
        prox_ckpt = args.get('prox_encoder_ckpt') or ''
        finetune_prox = bool(args.get('finetune_prox_encoder'))
        policy_tap = args.get('prox_policy_tap') or None
        if finetune_prox:
            if not is_geometry_feature(prox_feature):
                raise SystemExit(
                    '[P+ACT] --finetune_prox_encoder needs --prox_feature '
                    'surface_embedding (or nearest_surface).'
                )
            if not prox_ckpt:
                raise SystemExit(
                    '[P+ACT] --finetune_prox_encoder needs --prox_encoder_ckpt '
                    '(pretrained pact_surface_*_v1 start).'
                )
            if not policy_tap:
                policy_tap = 'readout'
        if prox_feature in ('trunk', 'delta') and not prox_ckpt:
            raise SystemExit(
                '[P+ACT] trunk/delta need --prox_encoder_ckpt; CVAE weights were removed. '
                'Use --prox_feature raw or a surface-geometry checkpoint.'
            )
        if is_geometry_feature(prox_feature) and not prox_ckpt:
            print(
                '[P+ACT] geometry encoder has NO checkpoint — untrained weights. '
                'Pass --prox_encoder_ckpt with a pact_surface_*_v1 file.'
            )
        prox_layout = args.get('prox_layout') or 'per_sensor'
        prox_K = int(args.get('prox_tokens_per_sensor') or 8)
        if is_geometry_feature(prox_feature) and prox_K == 8:
            prox_K = 1
        prox_encoder = build_pact_encoder(
            prox_feature,
            checkpoint=prox_ckpt or None,
            device='cuda',
            layout=prox_layout,
            tokens_per_sensor=prox_K,
            frozen=not finetune_prox,
            policy_tap=policy_tap,
        )
        policy_config['n_proximity_sensors'] = prox_encoder.n_act_sensors
        policy_config['prox_tokens_per_sensor'] = prox_K
        policy_config['prox_feat_dim'] = prox_encoder.act_feat_dim
        prox_pool = args.get('prox_pool')
        meta_path = os.path.join(dataset_dir, 'convert_meta.json')
        if not prox_pool:
            prox_pool = 'mean'
            if os.path.isfile(meta_path):
                prox_pool = json.load(open(meta_path)).get('prox_pool', 'mean')
        prox_layout_data = hdf5_proximity_layout(
            dataset_dir, prox_feature, force_live=finetune_prox,
        )
        # Baked tokens skip the net. Finetune / live readout must keep the encoder.
        train_encoder = None if (
            (not finetune_prox) and prox_layout_data in ('embeddings', 'positions')
        ) else prox_encoder
        prox_cfg_json = {
            'use_proximity': True,
            'prox_encoder_ckpt': str(prox_ckpt) if prox_ckpt else '',
            'prox_feature': prox_feature,
            'prox_layout': prox_encoder.layout,
            'prox_pool': prox_pool,
            'prox_tokens_per_sensor': prox_K,
            'prox_feat_dim': prox_encoder.act_feat_dim,
            'n_proximity_sensors': prox_encoder.n_act_sensors,
            'sensor_order': prox_encoder.sensor_order,
            'proximity_layout': prox_layout_data,
            'finetune_prox_encoder': finetune_prox,
            'prox_policy_tap': getattr(prox_encoder, 'policy_tap', policy_tap or ''),
            'prox_encoder_finetuned': 'prox_encoder_best.pt' if finetune_prox else '',
            'prox_encoder_lr': args.get('prox_encoder_lr'),
        }
        if finetune_prox:
            from scripts.pact_checkpoint import file_digest
            prox_cfg_json['pretrained_encoder_sha256'] = file_digest(prox_ckpt)
            prox_cfg_json['history_sampling'] = 'consecutive_control_steps_v1'
        print(f"[P+ACT] proximity fusion ON: feature={prox_feature} layout={prox_encoder.layout} "
              f"n_sensors={prox_encoder.n_act_sensors} feat_dim={prox_encoder.act_feat_dim} "
              f"K={prox_K} pool={prox_pool} data={prox_layout_data} ckpt={prox_ckpt} "
              f"frozen={not finetune_prox} tap={prox_cfg_json['prox_policy_tap']}")
        prox_encoder = train_encoder

    # wandb: log by default (opt out with --no_wandb). Run name auto-built as
    # taskname_numepochs_chunk_lr_seed unless --wandb_run_name is given.
    use_wandb = not args.get('no_wandb', False)
    wandb_run_name = args.get('wandb_run_name') or (
        f"{task_name}_{num_epochs}_{args.get('chunk_size')}_{args['lr']}_{args['seed']}"
    )

    # Each training run gets its own dated folder: <ckpt_dir root>/<task>/<datetime>_<runname>/.
    # --ckpt_dir is treated as the root (default 'ckpts'). Eval keeps the exact dir passed.
    if not is_eval:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        ckpt_dir = args.get('run_dir') or os.path.join(ckpt_dir, task_name, f"{timestamp}_{wandb_run_name}")
        if args.get('run_dir') and os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir):
            raise ValueError(f'Refusing to overwrite nonempty run: {ckpt_dir}')
        print(f"[ckpt] saving this run to {ckpt_dir}")

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'use_wandb': use_wandb,
        'wandb_project': args.get('wandb_project', 'act-obstacle-baseline'),
        'wandb_run_name': wandb_run_name,
        'use_proximity': use_proximity,
        'prox_encoder': prox_encoder,
        'prox_feature': (args.get('prox_feature') or 'raw') if use_proximity else None,
        'finetune_prox_encoder': bool(args.get('finetune_prox_encoder')),
        'prox_encoder_lr': args.get('prox_encoder_lr'),
        'blur_sigma0': args.get('blur_sigma0') or 0.0,
        'blur_curriculum_steps': args.get('blur_curriculum_steps'),
        'blur_mode': args.get('blur_mode') or 'curriculum',
        'image_dropout_p': args.get('image_dropout_p') or 0.0,
        'prox_dropout_p': args.get('prox_dropout_p') or 0.0,
        'image_dropout_mode': args.get('image_dropout_mode') or 'all',
        'zero_latent_on_drop': not args.get('no_zero_latent_on_drop', False),
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,
        args['chunk_size'], load_proximity=use_proximity,
        proximity_layout=prox_layout_data,
        n_proximity_sensors=policy_config.get('n_proximity_sensors', 0) if use_proximity else 0,
        proximity_feature_dim=policy_config.get('prox_feat_dim', 3) if use_proximity else 3,
        split=experiment['split'] if experiment else None,
    )

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    if experiment:
        from scripts.pact_workflow import write_json
        write_json(Path(ckpt_dir) / 'experiment.json', experiment)
        write_json(Path(ckpt_dir) / 'training_config.json', {
            'arguments': args, 'policy_config': policy_config,
            'experiment_sha256': experiment['sha256'],
        })
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # P+ACT: persist the proximity fusion config so eval_act_obstacle.py can rebuild
    # the exact same extractor + token layout without re-specifying every flag.
    if prox_cfg_json is not None:
        with open(os.path.join(ckpt_dir, 'prox_config.json'), 'w') as f:
            json.dump(prox_cfg_json, f, indent=2)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    _save_policy_pair(ckpt_path, best_state_dict,
                      config['prox_encoder'] if config['finetune_prox_encoder'] else None)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def blur_images(image_data, sigma):
    """FACTR visual curriculum: Gaussian-blur a (B, num_cam, C, H, W) 0-1 image batch.

    Training batches only — validation and eval always see sharp frames. Blurring
    the 0-1 tensor is equivalent to blurring after the ImageNet normalization inside
    the policy (the blur commutes with a per-channel affine).
    """
    if sigma < 0.1:
        return image_data
    from torchvision.transforms.functional import gaussian_blur
    b, k, c, h, w = image_data.shape
    kernel = 2 * math.ceil(3 * sigma) + 1
    flat = image_data.reshape(b * k, c, h, w)
    return gaussian_blur(flat, kernel_size=kernel, sigma=sigma).reshape(b, k, c, h, w)


# The ImageNet mean maps to exactly zero after the Normalize inside ACTPolicy,
# so a mean-filled frame is the least-OOD constant input for the frozen-BN ResNet.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])


def dropout_modalities(image_data, image_dropout_p, prox_dropout_p, mode='all'):
    """Modality dropout: per-sample hard dropout of vision and/or proximity.

    Samples independent per-sample Bernoulli masks over the batch dim of a
    (B, num_cam, C, H, W) 0-1 image batch. Vision-dropped samples are filled
    with the ImageNet mean (-> exactly-zero input post-normalize); mode='all'
    fills every camera, mode='single' fills one randomly chosen camera per
    dropped sample. The prox mask is sampled disjointly from the image mask so
    no sample is ever blind on both modalities (the caller zeroes the prox
    feature rows). Training batches only — validation and eval stay clean.

    Returns (image_data, img_mask, prox_mask) with (B,) bool masks.
    """
    b, num_cam = image_data.shape[:2]
    device = image_data.device
    img_mask = torch.rand(b, device=device) < image_dropout_p
    prox_mask = (torch.rand(b, device=device) < prox_dropout_p) & ~img_mask
    if img_mask.any():
        fill = IMAGENET_MEAN.to(device=device, dtype=image_data.dtype).view(1, 3, 1, 1)
        if mode == 'all':
            image_data[img_mask] = fill.unsqueeze(0)  # broadcast over num_cam
        elif mode == 'single':
            dropped_idx = torch.nonzero(img_mask, as_tuple=False).squeeze(1)
            cam_idx = torch.randint(num_cam, (dropped_idx.numel(),), device=device)
            image_data[dropped_idx, cam_idx] = fill
        else:
            raise ValueError(f"unknown image_dropout_mode {mode!r}")
    return image_data, img_mask, prox_mask


def forward_pass(data, policy, prox_encoder=None, blur_sigma=0.0,
                 image_dropout_p=0.0, prox_dropout_p=0.0, dropout_mode='all',
                 zero_latent_on_drop=True):
    if len(data) == 5:
        # P+ACT: 5th element is raw (B, 40, 8, 8) proximity depths. The frozen
        # extractor turns it into the (B, 1, feat_dim) conditioning feature.
        image_data, qpos_data, action_data, is_pad, prox_data = data
        image_data, qpos_data, action_data, is_pad, prox_data = (
            image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(), prox_data.cuda())
        image_data = blur_images(image_data, blur_sigma)
        # Modality dropout AFTER the blur so the curriculum math stays untouched
        # (blur of a constant fill would be the constant anyway).
        img_mask = prox_mask = None
        if image_dropout_p > 0 or prox_dropout_p > 0:
            image_data, img_mask, prox_mask = dropout_modalities(
                image_data, image_dropout_p, prox_dropout_p, dropout_mode)
        proximity_positions = None
        if prox_data is not None:
            from encoders.pact import encode_for_act
            proximity_positions = encode_for_act(prox_encoder, prox_data)
        if proximity_positions is not None and prox_mask is not None:
            # Zero the dropped prox rows; the tensor stays not-None, keeping
            # DETRVAE's n_proximity_sensors > 0 contract intact.
            proximity_positions = proximity_positions.masked_fill(
                prox_mask.view(-1, 1, 1), 0.0)
        return policy(qpos_data, image_data, action_data, is_pad,
                      proximity_positions=proximity_positions,
                      image_dropped=img_mask if zero_latent_on_drop else None)
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    image_data = blur_images(image_data, blur_sigma)
    # Vanilla branch: image dropout only, so a vanilla+dropout control arm trains.
    img_mask = None
    if image_dropout_p > 0:
        image_data, img_mask, _ = dropout_modalities(
            image_data, image_dropout_p, 0.0, dropout_mode)
    if img_mask is not None and zero_latent_on_drop:
        # Only ACTPolicy accepts image_dropped; dropout with CNNMLP is unsupported.
        return policy(qpos_data, image_data, action_data, is_pad,
                      image_dropped=img_mask)
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def _save_prox_encoder(path, prox_encoder):
    """Write inner surface-encoder weights next to the ACT policy ckpt."""
    from encoders.surface_geometry import save_encoder_checkpoint
    inner = getattr(prox_encoder, "inner", prox_encoder)
    kind = getattr(prox_encoder, "kind", "embedding")
    extra = {
        "policy_tap": getattr(prox_encoder, "policy_tap", "readout"),
        "validity_threshold": getattr(prox_encoder, "validity_threshold", 0.5),
    }
    save_encoder_checkpoint(
        path, inner, kind, extra=extra, frozen=False,
    )


def _save_policy_pair(path, state, encoder=None):
    """Persist a policy and its same-update encoder, then record their hashes."""
    torch.save(state, path)
    if encoder is None:
        return
    from scripts.pact_checkpoint import encoder_filename, file_digest
    from scripts.pact_workflow import write_json
    path = Path(path)
    encoder_path = path.with_name(encoder_filename(path.name))
    _save_prox_encoder(encoder_path, encoder)
    index = path.with_name('checkpoint_pairs.json')
    pairs = json.loads(index.read_text()) if index.exists() else {}
    pairs[path.name] = {'encoder': encoder_path.name, 'policy_sha256': file_digest(path),
                        'encoder_sha256': file_digest(encoder_path)}
    write_json(index, pairs)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    use_wandb = config.get('use_wandb', False) and _WANDB_AVAILABLE
    if config.get('use_wandb', False) and not _WANDB_AVAILABLE:
        print('[wandb] requested but not installed — run `pip install wandb`. Skipping logging.')

    set_seed(seed)

    if use_wandb:
        run = wandb.init(
            project=config.get('wandb_project', 'act-obstacle-baseline'),
            name=config.get('wandb_run_name'),
            dir=ckpt_dir,
            config={
                'task_name': config['task_name'],
                'policy_class': policy_class,
                'num_epochs': num_epochs,
                'seed': seed,
                'state_dim': config['state_dim'],
                'action_dim': config['action_dim'],
                'camera_names': config['camera_names'],
                'blur_sigma0': config.get('blur_sigma0', 0.0),
                'blur_curriculum_steps': config.get('blur_curriculum_steps'),
                'image_dropout_p': config.get('image_dropout_p', 0.0),
                'prox_dropout_p': config.get('prox_dropout_p', 0.0),
                'image_dropout_mode': config.get('image_dropout_mode', 'all'),
                'zero_latent_on_drop': config.get('zero_latent_on_drop', True),
                **{k: v for k, v in policy_config.items() if isinstance(v, (int, float, str, list, tuple, bool))},
            },
        )
        print(f"[wandb] logging run '{config.get('wandb_run_name')}' "
              f"(project {config.get('wandb_project')}): {run.url}")

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)
    # P+ACT: skin encoder. Frozen by default; --finetune_prox_encoder adds it
    # to the optimizer and uses CLS readout tokens at train and eval.
    prox_encoder = config.get('prox_encoder')
    finetune_prox = bool(config.get('finetune_prox_encoder')) and prox_encoder is not None
    if prox_encoder is not None and not finetune_prox:
        prox_encoder.eval()
    if finetune_prox:
        enc_lr = float(config.get('prox_encoder_lr') or config.get('lr') or 1e-5)
        encoder_params = [p for p in prox_encoder.parameters() if p.requires_grad]
        if not encoder_params:
            raise SystemExit('[P+ACT] --finetune_prox_encoder but encoder has no grads')
        optimizer.add_param_group({'params': encoder_params, 'lr': enc_lr})
        n_enc = sum(p.numel() for p in encoder_params)
        print(f"[P+ACT] finetune skin encoder ON: {n_enc} params lr={enc_lr} "
              f"tap={getattr(prox_encoder, 'policy_tap', None)} "
              f"feat_dim={getattr(prox_encoder, 'act_feat_dim', None)}")

    # FACTR visual curriculum: blur_sigma_n = sigma0 * (1 - n/N) at global training
    # step n; images start strongly blurred and sharpen linearly, forcing the policy
    # to lean on non-visual tokens (qpos, proximity) early. sigma0=0 disables it.
    blur_sigma0 = float(config.get('blur_sigma0') or 0.0)
    blur_mode = config.get('blur_mode') or 'curriculum'
    steps_per_epoch = max(1, len(train_dataloader))
    blur_total_steps = int(config.get('blur_curriculum_steps') or
                           max(1, (num_epochs * steps_per_epoch) // 2))
    if blur_sigma0 > 0:
        if blur_mode == 'constant':
            print(f"[blur] CONSTANT blur ON: sigma={blur_sigma0} on every training frame "
                  f"(no anneal); validation/eval stay sharp.")
        else:
            print(f"[FACTR] blur curriculum ON: sigma0={blur_sigma0} "
                  f"annealed to 0 over N={blur_total_steps} steps "
                  f"({steps_per_epoch} steps/epoch)")
    global_step = 0
    blur_sigma = 0.0

    # Modality dropout: per-sample vision/prox dropout on TRAINING batches only
    # (applied after the blur). Constant p throughout — annealing to 0 would
    # restore the vision+qpos redundancy exactly when the best-val ckpt is picked.
    image_dropout_p = float(config.get('image_dropout_p') or 0.0)
    prox_dropout_p = float(config.get('prox_dropout_p') or 0.0)
    dropout_mode = config.get('image_dropout_mode') or 'all'
    zero_latent_on_drop = config.get('zero_latent_on_drop', True)
    if image_dropout_p > 0 or prox_dropout_p > 0:
        print(f"[dropout] modality dropout ON: p_img={image_dropout_p} "
              f"p_prox={prox_dropout_p} mode={dropout_mode} "
              f"zero_latent_on_drop={zero_latent_on_drop}")

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    best_encoder_state = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            if prox_encoder is not None:
                prox_encoder.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy, prox_encoder)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                if finetune_prox:
                    best_encoder_state = deepcopy(prox_encoder.inner.state_dict())
                    _save_prox_encoder(
                        os.path.join(ckpt_dir, 'prox_encoder_best.pt'), prox_encoder
                    )
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        if finetune_prox:
            prox_encoder.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            if blur_mode == 'constant':
                blur_sigma = blur_sigma0
            else:
                blur_sigma = blur_sigma0 * max(0.0, 1.0 - global_step / blur_total_steps)
            global_step += 1
            forward_dict = forward_pass(data, policy, prox_encoder, blur_sigma=blur_sigma,
                                        image_dropout_p=image_dropout_p,
                                        prox_dropout_p=prox_dropout_p,
                                        dropout_mode=dropout_mode,
                                        zero_latent_on_drop=zero_latent_on_drop)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if use_wandb:
            log_dict = {'epoch': epoch, 'min_val_loss': float(min_val_loss)}
            if blur_sigma0 > 0:
                log_dict['train/blur_sigma'] = float(blur_sigma)
            for k, v in epoch_summary.items():
                log_dict[f'train/{k}'] = float(v.item())
            for k, v in validation_history[-1].items():
                log_dict[f'val/{k}'] = float(v.item())
            wandb.log(log_dict, step=epoch)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            _save_policy_pair(ckpt_path, policy.state_dict(), prox_encoder if finetune_prox else None)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    _save_policy_pair(ckpt_path, policy.state_dict(), prox_encoder if finetune_prox else None)
    if finetune_prox:
        if best_encoder_state is not None:
            prox_encoder.inner.load_state_dict(best_encoder_state)
            _save_prox_encoder(
                os.path.join(ckpt_dir, 'prox_encoder_best.pt'), prox_encoder
            )

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    _save_policy_pair(ckpt_path, best_state_dict, prox_encoder if finetune_prox else None)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        # dropout diagnostics (l1_img_dropped/l1_clean) exist only in training
        # summaries (validation always runs clean) — plot what each side has.
        val_values = [summary[key].item() for summary in validation_history if key in summary]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        if val_values:
            plt.plot(np.linspace(0, num_epochs-1, len(val_values)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--experiment_manifest', type=str, default=None)
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, default='ckpts',
                        help='ckpt root; a run saves to <ckpt_dir>/<task>/<datetime>_<runname>/')
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    # P+ACT (PACT): proximity-CVAE fusion. OFF by default -> vanilla ACT.
    parser.add_argument('--use_proximity', action='store_true',
                        help='Fuse the frozen Safety-CVAE skin feature into ACT as extra '
                             'encoder tokens (PACT). Requires the obstacle_pact task data.')
    parser.add_argument('--prox_encoder_ckpt', type=str, default=None,
                        help='Encoder start. CVAE dir for trunk/delta; '
                             'pact_surface_*_v1.pt for nearest_surface / surface_embedding. '
                             'raw needs none. Finetune writes a new copy in the run dir.')
    parser.add_argument('--finetune_prox_encoder', action='store_true',
                        help='Unfreeze the geometry encoder and train it with ACT. '
                             'Policy tap is the 128-d CLS readout (not the frozen 32-d '
                             'embedding). Live raw_causal skin; do not bake tokens.')
    parser.add_argument('--prox_policy_tap', type=str, default=None,
                        choices=('embedding', 'readout', 'xyz'),
                        help="Geometry policy feature. Default: embedding when frozen, "
                             "readout when --finetune_prox_encoder.")
    parser.add_argument('--prox_encoder_lr', type=float, default=None,
                        help='LR for the unfrozen skin encoder. Default: same as --lr.')
    parser.add_argument('--prox_feature', type=str, default='raw',
                        choices=('trunk', 'delta', 'raw', 'peak_closeness',
                                 'nearest_surface', 'surface_embedding',
                                 'xyz', 'embedding'),
                        help="Skin feature: 'raw'/'peak_closeness' (40-d peak closeness), "
                             "'nearest_surface'/'xyz' (3-d local point), "
                             "'surface_embedding'/'embedding' (32-d frozen or 128-d CLS readout), "
                             "'trunk'/'delta' (deleted CVAE taps).")
    parser.add_argument('--prox_layout', type=str, default='per_sensor',
                        choices=('global', 'per_sensor'),
                        help="'per_sensor' (default): 40 named tokens, K=1. "
                             "'global': mash all sensors into one vector then K tokens.")
    parser.add_argument('--prox_pool', type=str, default=None, choices=('mean', 'min'),
                        help="Live substep pool. Default: convert_meta.json or mean.")
    parser.add_argument('--prox_tokens_per_sensor', type=int, default=8,
                        help='K encoder tokens to expand the single skin feature into.')

    # FACTR-style visual curriculum: Gaussian-blur ALL camera images at TRAINING
    # time with sigma_n = sigma0 * (1 - n/N) at global train step n. Strong blur
    # early forces the policy onto non-visual tokens (qpos, proximity); the blur
    # anneals away by step N. Validation and eval always see sharp frames.
    parser.add_argument('--blur_sigma0', type=float, default=0.0,
                        help='Initial Gaussian blur sigma (pixels) for the FACTR visual '
                             'curriculum. 0 (default) disables it. FACTR uses 8.')
    parser.add_argument('--blur_curriculum_steps', type=int, default=None,
                        help='N: number of training steps to linearly anneal the blur to '
                             'zero over. Default: half of the total training steps. '
                             '(Ignored when --blur_mode constant.)')
    parser.add_argument('--blur_mode', type=str, default='curriculum',
                        choices=('curriculum', 'constant'),
                        help="'curriculum' (default, FACTR): blur anneals sigma0 -> 0 over N "
                             "steps. 'constant': hold sigma0 on EVERY training frame for the "
                             "whole run (no anneal) -> a fixed-blur degraded-vision training "
                             "handicap. Validation/eval stay sharp in both modes.")

    # Modality dropout (training only, applied AFTER the blur): per-sample hard
    # dropout of the vision modality (fill with the ImageNet mean -> exactly-zero
    # post-normalize input) so on dropped samples the L1 loss can only be reduced
    # through the qpos + proximity tokens; plus low-p prox dropout, sampled
    # disjointly, to keep vision sufficient too. Constant p all training (no
    # anneal). Validation and eval always see clean inputs.
    parser.add_argument('--image_dropout_p', type=float, default=0.0,
                        help='Per-sample probability of dropping the vision modality '
                             'at training time (mean-filled frames). 0 (default) '
                             'disables it; the recommended arm uses 0.3.')
    parser.add_argument('--prox_dropout_p', type=float, default=0.0,
                        help='Per-sample probability of zeroing the proximity feature '
                             'at training time. Disjoint from image dropout so no '
                             'sample is ever blind on both. Recommended 0.1.')
    parser.add_argument('--image_dropout_mode', type=str, default='all',
                        choices=('all', 'single'),
                        help="'all' mean-fills every camera of a dropped sample; "
                             "'single' fills exactly one randomly chosen camera.")
    parser.add_argument('--no_zero_latent_on_drop', action='store_true',
                        help='Ablation: do NOT zero the CVAE style latent z on vision-'
                             'dropped samples (leaves the action-chunk z-leakage path '
                             'open and disables the split-L1 diagnostics).')

    # wandb — on by default; opt out with --no_wandb. Run name auto-built as
    # taskname_numepochs_chunk_lr_seed unless --wandb_run_name overrides it.
    parser.add_argument('--use_wandb', action='store_true',
                        help='(deprecated, on by default) Log to Weights & Biases.')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging.')
    parser.add_argument('--wandb_project', type=str, default='act-obstacle-baseline')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Override the auto run name taskname_numepochs_chunk_lr_seed.')

    main(vars(parser.parse_args()))
