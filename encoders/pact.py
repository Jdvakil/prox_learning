"""ACT glue for skin encoders.

Live readout tokens (or frozen 32-d / XYZ) go into DETRVAE + EpisodicDataset
from here. Finetune path never reads baked HDF5 embeddings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import load_encoder, resolve_encoder_name

GEOMETRY_KEYS = frozenset({"nearest_surface", "surface_embedding"})


def is_geometry_feature(name: str) -> bool:
    try:
        return resolve_encoder_name(name) in GEOMETRY_KEYS
    except ValueError:
        return False


def build_pact_encoder(
    feature: str,
    *,
    checkpoint: str | Path | None = None,
    device: str = "cuda",
    layout: str = "per_sensor",
    tokens_per_sensor: int = 8,
    frozen: bool = True,
    policy_tap: str | None = None,
):
    """One constructor for peak-closeness, CVAE taps, and surface geometry."""
    return load_encoder(
        feature,
        checkpoint=checkpoint or None,
        device=device,
        layout=layout,
        tokens_per_sensor=tokens_per_sensor,
        frozen=frozen,
        policy_tap=policy_tap,
    )


def hdf5_proximity_layout(
    dataset_dir: str | Path,
    feature: str,
    *,
    force_live: bool = False,
) -> str:
    """How the dataloader should read skin from episode_0.hdf5."""
    key = resolve_encoder_name(feature)
    if force_live and key in GEOMETRY_KEYS:
        return "raw_causal"
    first = Path(dataset_dir) / "episode_0.hdf5"
    if not first.is_file():
        return "raw_causal" if key in GEOMETRY_KEYS else "raw"
    import h5py

    with h5py.File(first, "r") as handle:
        obs = handle["observations"]
        if key == "surface_embedding" and "proximity_embeddings" in obs:
            return "embeddings"
        if key == "nearest_surface" and "proximity_positions" in obs:
            return "positions"
        if key in GEOMETRY_KEYS:
            return "raw_causal"
        return "raw"


def encode_for_act(encoder, prox_data: torch.Tensor) -> torch.Tensor:
    """Map a dataloader skin tensor to DETRVAE ``proximity_positions``.

    Layouts:
      (B, 40, 8, 8)     peak-closeness / tiled geometry snapshot
      (B, H, 40, 8, 8)  geometry causal history (H<=8 pooled steps)
      (B, 40, D)        already-encoded embeddings or XYZ
    """
    if encoder is None:
        return prox_data
    if prox_data.ndim == 3:
        return prox_data
    if hasattr(encoder, "encode_pooled_history") and prox_data.ndim == 5:
        return encoder.encode_pooled_history(prox_data)
    return encoder(prox_data)


def causal_pooled_window(proximity: np.ndarray, timestep: int) -> np.ndarray:
    """Last 8 pooled frames ending at ``timestep``, left-padded. ``(8, S, 8, 8)``."""
    values = np.asarray(proximity, dtype=np.float32)
    if values.ndim == 5 and values.shape[2] == 4:
        values = values.mean(axis=2)
    if values.ndim != 4 or values.shape[-2:] != (8, 8):
        raise ValueError(f"expected (T,S,8,8), got {values.shape}")
    start = max(0, int(timestep) - 7)
    block = values[start : int(timestep) + 1]
    if len(block) < 8:
        block = np.concatenate(
            (np.repeat(block[:1], 8 - len(block), axis=0), block), axis=0
        )
    return block


def resolve_act_encoder_load(
    ckpt_dir: str | Path,
    pcfg: dict[str, Any],
    cli_ckpt: str = "",
) -> dict[str, Any]:
    """Kwargs for ``build_pact_encoder`` at eval. Finetuned run-dir weights win."""
    finetune = bool(pcfg.get("finetune_prox_encoder"))
    tap = pcfg.get("prox_policy_tap") or None
    if not tap:
        tap = "readout" if finetune else None
    run_dir = Path(ckpt_dir)
    finetuned_name = pcfg.get("prox_encoder_finetuned") or "prox_encoder_best.pt"
    candidates = [
        run_dir / finetuned_name,
        run_dir / "prox_encoder_best.pt",
        run_dir / "prox_encoder.pt",
    ]
    checkpoint = None
    if finetune:
        for path in candidates:
            if path.is_file():
                checkpoint = str(path)
                break
    if checkpoint is None:
        checkpoint = cli_ckpt or pcfg.get("prox_encoder_ckpt") or None
        if checkpoint == "":
            checkpoint = None
    return {
        "checkpoint": checkpoint,
        "frozen": not finetune,
        "policy_tap": tap,
    }
