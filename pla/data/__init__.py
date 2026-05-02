"""Data layer.

Top-level exports:
    PLADataset, collate_pla    — sliding-window training DataLoader
    compute_stats, normalize, unnormalize  — per-channel stats
    verify_dataset             — Day-2 sanity check
    validate, proximity_informative_fraction  — schema utilities
    collect_episode            — single-episode HDF5 writer
"""
from pla.data.collect import collect_episode
from pla.data.dataset import PLADataset, collate_pla
from pla.data.normalize import compute_stats, normalize, unnormalize
from pla.data.schema import proximity_informative_fraction, validate
from pla.data.verify import verify_dataset

__all__ = [
    "PLADataset",
    "collate_pla",
    "compute_stats",
    "collect_episode",
    "normalize",
    "proximity_informative_fraction",
    "unnormalize",
    "validate",
    "verify_dataset",
]
