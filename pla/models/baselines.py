"""Baseline models for the ablation ladder.

This file is now thin: ``VLMOnlyACT`` is just ``PLA(vlm_only=True)``, so we
re-export it as a named alias for clarity in scripts and configs. We keep
``PropOnlyMLP`` here as the absolute floor — qpos -> action chunk through a
small MLP, with no vision and no proximity. Its number tells reviewers how
hard the task actually is.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pla.models.pla import PLA


def VLMOnlyACT(**kwargs) -> PLA:  # noqa: N802 (ScreamingSnake intentional alias)
    """Construct PLA with the proximity stream disabled.

    Identical to PLA in every other respect (same backbone, same ACT decoder,
    same chunk_size, same hyperparams). The *only* difference is
    ``vlm_only=True`` -- this is the headline comparison of the paper.
    """
    kwargs["vlm_only"] = True
    return PLA(**kwargs)


class PropOnlyMLP(nn.Module):
    """qpos -> action chunk through a small MLP. No vision, no proximity.

    This is the "is the task even reasonable?" floor baseline. If PropOnlyMLP
    solves it, the task is trivial and the proximity claim is unfalsifiable.
    """

    def __init__(
        self,
        proprio_dim: int = 7,
        action_dim: int = 7,
        chunk_size: int = 100,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, chunk_size * action_dim),
        )

    def forward(self, qpos: torch.Tensor) -> torch.Tensor:
        b = qpos.shape[0]
        return self.net(qpos).reshape(b, self.chunk_size, self.action_dim)
