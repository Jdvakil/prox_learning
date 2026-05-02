"""PLADataset — sliding-window DataLoader over HDF5 trajectory shards.

Yields per item::

    {
      'tof':     [N_sensors, 8, 8]            float32  (normalized)
      'rgb':     [K=2, 3, 224, 224]            float32  (in [0, 1])
      'qpos':    [7]                           float32  (normalized)
      'language': str                          (instruction; per-episode)
      'actions': [chunk_size, 7]                float32  (normalized)
    }

The chunk-size sliding window matches Zhao et al. 2023: at training step we
condition on a single observation timestep ``t`` and predict ``actions[t:t+k]``.

Sensor mask
    ``sensor_mask=[i, j, ...]`` zeros out those sensor indices in the output
    tof tensor. Used for the wrist-only ablation and post-hoc sensor
    importance — both are configured at the DataLoader, not the model, so we
    can swap masks without re-instantiating the model.

Train/val split
    Files are deterministically permuted with ``seed`` and the first
    ``val_frac`` are val. CRITICALLY this must use the *same* seed as
    ``pla.data.normalize`` so the val files are not in the stats.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PLADataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        stats_path: str | Path,
        chunk_size: int = 100,
        split: str = "train",
        val_frac: float = 0.1,
        seed: int = 0,
        sensor_mask: Iterable[int] | None = None,
        rgb_history: int = 2,
        language_default: str = "pick up the object",
    ) -> None:
        self.chunk_size = int(chunk_size)
        self.rgb_history = int(rgb_history)
        self.language_default = language_default

        with open(stats_path) as f:
            stats = json.load(f)
        self.stats = stats
        self.tof_mean = torch.tensor(stats["tof_mean"], dtype=torch.float32)
        self.tof_std = torch.tensor(stats["tof_std"], dtype=torch.float32)
        self.qpos_mean = torch.tensor(stats["qpos_mean"], dtype=torch.float32)
        self.qpos_std = torch.tensor(stats["qpos_std"], dtype=torch.float32)
        self.act_mean = torch.tensor(stats["act_mean"], dtype=torch.float32)
        self.act_std = torch.tensor(stats["act_std"], dtype=torch.float32)

        self.sensor_mask = None
        if sensor_mask is not None:
            mask = np.zeros(stats["n_sensors"], dtype=bool)
            for i in sensor_mask:
                mask[int(i)] = True
            self.sensor_mask = mask

        all_files = sorted(p for p in Path(data_dir).rglob("*.h5"))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(all_files))
        n_val = max(1, int(len(all_files) * val_frac))
        if split == "val":
            files = [all_files[i] for i in perm[:n_val]]
        elif split == "train":
            files = [all_files[i] for i in perm[n_val:]]
        else:
            raise ValueError(f"split must be train/val, got {split!r}")

        # Build sliding-window index.
        self.index: list[tuple[Path, str, int]] = []
        for p in files:
            with h5py.File(p, "r") as h:
                for ep in h.keys():
                    T = h[f"{ep}/observations/tof"].shape[0]
                    if T <= chunk_size + rgb_history:
                        continue
                    for t in range(rgb_history - 1, T - chunk_size):
                        self.index.append((p, ep, t))

        self.files = files

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        fpath, ep, t = self.index[idx]
        k = self.chunk_size
        h = self.rgb_history

        with h5py.File(fpath, "r") as f:
            obs = f[f"{ep}/observations"]
            tof_t = np.asarray(obs["tof"][t])                      # [N, 8, 8]
            rgb_window = np.asarray(obs["rgb"][t - h + 1 : t + 1]) # [h, 3, 224, 224]
            qpos_t = np.asarray(obs["qpos"][t])                    # [7]
            actions_chunk = np.asarray(f[f"{ep}/actions"][t : t + k])  # [k, 7]
            language = f[ep].attrs.get("language", self.language_default)
            if isinstance(language, bytes):
                language = language.decode()

        tof_t = torch.from_numpy(tof_t).float()
        if self.sensor_mask is not None:
            tof_t[self.sensor_mask] = 0.0
        tof_n = (tof_t - self.tof_mean) / (self.tof_std + 1e-6)

        rgb = torch.from_numpy(rgb_window).float() / 255.0          # [h, 3, 224, 224]

        qpos_n = (torch.from_numpy(qpos_t).float() - self.qpos_mean) / (self.qpos_std + 1e-6)
        acts_n = (torch.from_numpy(actions_chunk).float() - self.act_mean) / (self.act_std + 1e-6)

        return {
            "tof": tof_n,
            "rgb": rgb,
            "qpos": qpos_n,
            "language": str(language),
            "actions": acts_n,
        }


def collate_pla(batch: list[dict]) -> dict:
    """Default collate that stacks tensors and keeps language as a list."""
    out = {
        "tof": torch.stack([b["tof"] for b in batch]),
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "qpos": torch.stack([b["qpos"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
        "language": [b["language"] for b in batch],
    }
    return out
