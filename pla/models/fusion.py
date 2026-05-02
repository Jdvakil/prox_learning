"""Modality fusion — concat (primary) and cross-attention (ablation).

Concat fusion (PROJECT.md §3.4):
    context = LayerNorm( concat([tof_tokens, vlm_tokens, prop_token], dim=1) )

Why concat first? It is the simplest fusion that lets every downstream
attention layer in the ACT decoder freely mix any subset of tokens; nothing
about the architecture privileges one stream. The LayerNorm puts proximity,
visual-language, and proprio tokens on the same scale — without it the
decoder tends to ignore the lower-magnitude stream.

Cross-attention ablation:
    tof_attended = MHA(query=tof, key=value=vlm)         # [B, N, d]
    tof_enhanced = LayerNorm(tof + tof_attended)
    context      = concat([tof_enhanced, vlm, prop])

This biases the model toward "given what I see, which proximity readings
matter?" The concat version makes no such commitment. Comparing the two tests
whether the bias helps generalization in low-data regimes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ModalityFusion(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        proprio_dim: int = 7,
        fusion_type: str = "concat",
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        if fusion_type not in ("concat", "cross_attn"):
            raise ValueError(f"unknown fusion_type {fusion_type!r}")
        self.fusion_type = fusion_type
        self.d_model = d_model
        self.prop_proj = nn.Linear(proprio_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        if fusion_type == "cross_attn":
            self.xattn = nn.MultiheadAttention(
                d_model, num_heads=n_heads, batch_first=True
            )
            self.xnorm = nn.LayerNorm(d_model)

    def forward(
        self,
        tof_tokens: torch.Tensor | None,
        vlm_tokens: torch.Tensor,
        qpos: torch.Tensor,
    ):
        """Return fused context [B, N_total, d_model].

        If ``tof_tokens`` is None, the proximity stream is omitted entirely
        (this is what the VLM-only baseline does). Cross-attention is skipped
        in that case.
        """
        prop = self.prop_proj(qpos).unsqueeze(1)  # [B, 1, d]

        if tof_tokens is None:
            return self.norm(torch.cat([vlm_tokens, prop], dim=1)), None

        if self.fusion_type == "concat":
            return self.norm(torch.cat([tof_tokens, vlm_tokens, prop], dim=1)), None

        # cross_attn: tof attends over vlm
        attended, attn_weights = self.xattn(
            tof_tokens, vlm_tokens, vlm_tokens, need_weights=True
        )
        tof_enhanced = self.xnorm(tof_tokens + attended)
        ctx = self.norm(torch.cat([tof_enhanced, vlm_tokens, prop], dim=1))
        return ctx, attn_weights
