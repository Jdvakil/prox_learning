import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, num_queries,
                 load_proximity=False, proximity_layout="raw",
                 n_proximity_sensors=0, proximity_feature_dim=3,
                 expected_proximity_encoder_sha256=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.num_queries = num_queries
        # P+ACT: when True, __getitem__ also returns a proximity tensor.
        # layout "raw"        -> (40, 8, 8) metres (peak-closeness encoder)
        # layout "raw_causal" -> (8, 40, 8, 8) last 8 pooled steps (geometry encoder)
        # layout "embeddings" -> (40, 32) frozen surface-embedding tokens
        # layout "positions"  -> (40, 3)  frozen nearest-surface XYZ
        self.load_proximity = load_proximity
        self.proximity_layout = proximity_layout
        self.n_proximity_sensors = int(n_proximity_sensors)
        self.proximity_feature_dim = int(proximity_feature_dim)
        self.expected_proximity_encoder_sha256 = expected_proximity_encoder_sha256
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs.get('sim', False)  # Default to False if not present
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            proximity = None
            if self.load_proximity:
                proximity = self._load_proximity(root, dataset_path, start_ts)
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        # Pad actions to num_queries for ACT model consistency. When the
        # remaining horizon is longer than num_queries we truncate (the
        # network only predicts num_queries steps ahead anyway).
        padded_action = np.zeros((self.num_queries, original_action_shape[1]), dtype=np.float32)
        n_real = min(action_len, self.num_queries)
        padded_action[:n_real] = action[:n_real]
        is_pad = np.zeros(self.num_queries)
        is_pad[n_real:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        if self.load_proximity:
            prox_data = torch.from_numpy(np.asarray(proximity, dtype=np.float32)).float()
            return image_data, qpos_data, action_data, is_pad, prox_data

        return image_data, qpos_data, action_data, is_pad

    def _load_proximity(self, root, dataset_path, start_ts):
        layout = self.proximity_layout
        if layout in ("embeddings", "positions"):
            feature_name = (
                "proximity_positions" if layout == "positions" else "proximity_embeddings"
            )
            feature_path = f"/observations/{feature_name}"
            if feature_path not in root:
                raise ValueError(
                    f"{dataset_path}: {feature_name} missing — run "
                    f"`python -m encoders.encode_tokens`."
                )
            proximity_positions = root[feature_path][start_ts]
            expected_shape = (self.n_proximity_sensors, self.proximity_feature_dim)
            if proximity_positions.shape != expected_shape:
                raise ValueError(
                    f"{dataset_path}: {feature_name} shape "
                    f"{proximity_positions.shape} != {expected_shape}"
                )
            observed_sha = root.attrs.get("pact_surface_encoder_sha256")
            if isinstance(observed_sha, bytes):
                observed_sha = observed_sha.decode()
            expected = self.expected_proximity_encoder_sha256
            if expected and observed_sha != expected:
                raise ValueError(
                    f"{dataset_path}: surface encoder sha256 {observed_sha} != {expected}"
                )
            return proximity_positions
        if "/observations/proximity" not in root:
            raise KeyError(
                f"{dataset_path} has no /observations/proximity — re-run "
                f"convert_obstacle_to_act.py with --with_proximity."
            )
        prox = root["/observations/proximity"]
        if layout == "raw_causal":
            start = max(0, int(start_ts) - 7)
            block = np.asarray(prox[start : int(start_ts) + 1], dtype=np.float32)
            if block.ndim == 5 and block.shape[2] == 4:
                block = block.mean(axis=2)
            if len(block) < 8:
                block = np.concatenate(
                    (np.repeat(block[:1], 8 - len(block), axis=0), block), axis=0
                )
            return block
        return prox[start_ts]


def get_norm_stats(dataset_dir, num_episodes, episode_ids=None):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in (range(num_episodes) if episode_ids is None else episode_ids):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    # Concatenate instead of stack to handle different episode lengths
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": all_qpos_data[0].numpy()}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, num_queries,
              load_proximity=False, proximity_layout="raw",
              n_proximity_sensors=0, proximity_feature_dim=3,
              expected_proximity_encoder_sha256=None, split=None):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    if split is None:
        shuffled_indices = np.random.permutation(num_episodes)
        train_indices = shuffled_indices[:int(0.8 * num_episodes)]
        val_indices = shuffled_indices[int(0.8 * num_episodes):]
    else:
        train_indices, val_indices = split['train'], split['val']
        if (not train_indices or not val_indices or
                len(set(train_indices)) != len(train_indices) or
                len(set(val_indices)) != len(val_indices) or
                set(train_indices) & set(val_indices) or
                set(train_indices) | set(val_indices) != set(range(num_episodes))):
            raise ValueError('explicit split must partition the dataset exactly once')
        if split.get('normalization') != 'train_only':
            raise ValueError('explicit split requires train_only normalization')

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes,
                               episode_ids=train_indices if split is not None else None)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, num_queries,
                                    load_proximity=load_proximity,
                                    proximity_layout=proximity_layout,
                                    n_proximity_sensors=n_proximity_sensors,
                                    proximity_feature_dim=proximity_feature_dim,
                                    expected_proximity_encoder_sha256=expected_proximity_encoder_sha256)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, num_queries,
                                  load_proximity=load_proximity,
                                  proximity_layout=proximity_layout,
                                  n_proximity_sensors=n_proximity_sensors,
                                  proximity_feature_dim=proximity_feature_dim,
                                  expected_proximity_encoder_sha256=expected_proximity_encoder_sha256)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
