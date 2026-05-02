"""PLA — full model: frozen Molmo2 + ProximityEncoder + proprio + ACT decoder.

Token layout fed to the ACT decoder (concat fusion, primary):

    [ proximity tokens        ]   N_sensors from ProximityEncoder
    [ visual-language tokens  ]   ~192 per RGB frame from Molmo2 (frozen)
    [ proprio token           ]   1 from a Linear(7 -> d_model)

Concatenated, layer-normed, then decoded to a chunk of 100 joint-delta actions.

The ``vlm_only`` flag at construction time *removes* the proximity stream from
the context entirely. That single flag is the difference between PLA and the
VLM-only ACT baseline — the headline comparison of the paper. We expose it as
a config switch so the *only* uncontrolled variable between the two runs is
the presence of proximity.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from pla.models.act import ACTDecoder
from pla.models.fusion import ModalityFusion
from pla.models.proximity_encoder import (
    Conv2DToFEncoder,
    HandcraftedToFEncoder,
    ProximityEncoder,
)


def build_proximity_encoder(
    encoder_type: str,
    n_sensors: int,
    d_model: int,
) -> nn.Module:
    if encoder_type == "shared_mlp":
        return ProximityEncoder(n_sensors=n_sensors, d_model=d_model)
    if encoder_type == "handcrafted":
        return HandcraftedToFEncoder(n_sensors=n_sensors, d_model=d_model)
    if encoder_type == "conv2d":
        return Conv2DToFEncoder(n_sensors=n_sensors, d_model=d_model)
    raise ValueError(f"unknown encoder_type {encoder_type!r}")


class PLA(nn.Module):
    """Peripersonal Language-Action policy.

    Args:
        n_sensors: number of ToF sensors on the body. Must match the dataset.
        d_model: hidden dim throughout. 512 matches ACT defaults.
        chunk_size: number of joint-delta actions predicted per forward pass.
        action_dim, proprio_dim: 7 for FR3 (joint deltas / qpos).
        encoder_type: ``shared_mlp`` (PLA), ``handcrafted`` or ``conv2d``
            (ablations).
        fusion_type: ``concat`` (PLA) or ``cross_attn`` (ablation).
        vlm_only: if True the proximity stream is never built / never fused.
            Use this for the baseline run.
        vl_backbone: a ``nn.Module`` mapping ``(rgb, language)`` -> tokens
            ``[B, N_vis, d_model]``. Pass ``DummyVLBackbone`` for unit tests.
        kl_weight: ACT beta. 10 by default.
        sensor_mask: optional list of sensor indices to zero out at the input
            of the proximity encoder. Used by the wrist-only ablation and
            sensor-importance computation.
    """

    def __init__(
        self,
        *,
        n_sensors: int = 32,
        d_model: int = 512,
        chunk_size: int = 100,
        action_dim: int = 7,
        proprio_dim: int = 7,
        encoder_type: str = "shared_mlp",
        fusion_type: str = "concat",
        vlm_only: bool = False,
        vl_backbone: nn.Module | None = None,
        kl_weight: float = 10.0,
        sensor_mask: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.vlm_only = vlm_only
        self.encoder_type = encoder_type
        self.fusion_type = fusion_type

        self.vl = vl_backbone
        self.fusion = ModalityFusion(
            d_model=d_model, proprio_dim=proprio_dim, fusion_type=fusion_type,
        )
        self.act_decoder = ACTDecoder(
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            kl_weight=kl_weight,
        )

        if vlm_only:
            self.proximity_encoder: nn.Module | None = None
        else:
            self.proximity_encoder = build_proximity_encoder(
                encoder_type, n_sensors=n_sensors, d_model=d_model
            )

        if sensor_mask is None:
            self.register_buffer(
                "sensor_mask", torch.zeros(n_sensors, dtype=torch.bool), persistent=False
            )
        else:
            mask = torch.zeros(n_sensors, dtype=torch.bool)
            for i in sensor_mask:
                mask[int(i)] = True
            self.register_buffer("sensor_mask", mask, persistent=False)

    def _maybe_mask(self, tof: torch.Tensor) -> torch.Tensor:
        if not torch.any(self.sensor_mask):
            return tof
        tof = tof.clone()
        tof[:, self.sensor_mask] = 0.0
        return tof

    def encode_context(
        self,
        rgb: torch.Tensor,
        language: list[str],
        tof: torch.Tensor | None,
        qpos: torch.Tensor,
    ):
        if self.vl is None:
            raise RuntimeError("vl_backbone not configured")
        vlm_tokens = self.vl(rgb, language)
        if self.vlm_only or tof is None or self.proximity_encoder is None:
            ctx, attn = self.fusion(None, vlm_tokens, qpos)
            return ctx, attn
        tof_in = self._maybe_mask(tof)
        tof_tokens = self.proximity_encoder(tof_in)
        # Handcrafted / conv2d encoders return [B, 1, d] not [B, N, d];
        # the fusion layer is shape-agnostic so this just works.
        ctx, attn = self.fusion(tof_tokens, vlm_tokens, qpos)
        return ctx, attn

    def forward(
        self,
        rgb: torch.Tensor,
        language: list[str],
        tof: torch.Tensor | None,
        qpos: torch.Tensor,
        actions: torch.Tensor | None = None,
    ):
        ctx, _ = self.encode_context(rgb, language, tof, qpos)
        if actions is not None:
            pred, mu, logvar = self.act_decoder(ctx, actions=actions, qpos=qpos)
            return pred, mu, logvar
        return self.act_decoder(ctx)

    @torch.no_grad()
    def get_action(
        self,
        obs: dict,
        stats: dict,
        device: str | torch.device = "cuda",
    ):
        """Inference-time helper: produce *unnormalized* actions from a *normalized* obs.

        ``obs`` must already have been normalized (see ``pla.data.normalize``).
        We unnormalize the predicted action chunk before returning it so the
        controller receives joint deltas in the original scale.
        """
        import numpy as np

        self.eval()
        tof = (
            torch.as_tensor(obs["tof"], dtype=torch.float32, device=device).unsqueeze(0)
            if obs.get("tof") is not None
            else None
        )
        rgb = torch.as_tensor(obs["rgb"], dtype=torch.float32, device=device).unsqueeze(0)
        qpos = torch.as_tensor(obs["qpos"], dtype=torch.float32, device=device).unsqueeze(0)
        language = [obs.get("language", "")]
        pred = self(rgb, language, tof, qpos, actions=None)
        act_mean = torch.as_tensor(stats["act_mean"], dtype=torch.float32, device=device)
        act_std = torch.as_tensor(stats["act_std"], dtype=torch.float32, device=device)
        actions = pred[0] * act_std + act_mean
        return actions.cpu().numpy().astype(np.float32)
