"""ACT — Action Chunking with Transformers.

Faithful reimplementation of Zhao et al. 2023 (RSS 2023). The decoder predicts
``chunk_size`` future joint deltas in a single shot from a fused context of
modality tokens. The CVAE encoder is used at training time only — it digests
the *ground-truth* future action chunk plus current qpos into a latent ``z``,
which is reparameterized and concatenated to the decoder context. At
inference ``z = 0`` (the prior mean), so the encoder is *discarded*.

Loss = L1(pred, gt) + beta * KL(q(z|chunk) || p(z))
beta = 10, constant. Higher beta = the latent carries less stochastic
information; the policy converges to the mean modal trajectory.

Why this architecture (over diffusion / autoregressive heads)?
    * One forward pass produces ``chunk_size`` actions => simple, fast
      inference, no temporal-ensembling needed at the decoder boundary.
    * The VAE bottleneck regularizes against multimodality in human/expert
      demos — important here because we collect demos from a heuristic
      grasp policy with multiple acceptable trajectories.
    * It is the established baseline in robot-learning papers; this matches
      what reviewers expect for the VLM-only ACT comparison.

Shapes follow PROJECT.md §3 token table:
    context [B, N_ctx, d_model]   -- N_ctx ~= 192 (vlm) + 32 (tof) + 1 (prop)
    output  [B, chunk_size, action_dim]
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # [1, max_len, d]


class ACTDecoder(nn.Module):
    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 100,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 7,
        ffn_dim: int = 3200,
        z_dim: int = 32,
        kl_weight: float = 10.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.z_dim = z_dim
        self.kl_weight = kl_weight

        # CVAE encoder (training only).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True,
        )
        self.cvae_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)
        self.action_in = nn.Linear(action_dim, d_model)
        self.qpos_in = nn.Linear(action_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mu_head = nn.Linear(d_model, z_dim)
        self.logvar_head = nn.Linear(d_model, z_dim)

        # Latent -> context.
        self.z_proj = nn.Linear(z_dim, d_model)

        # Positional encoding for the encoder seq (cls + qpos + chunk).
        self.register_buffer(
            "pos_enc", _sinusoidal_pe(max_len=chunk_size + 8, d_model=d_model),
            persistent=False,
        )

        # Decoder.
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_decoder_layers)
        self.query_embed = nn.Embedding(chunk_size, d_model)
        self.action_head = nn.Linear(d_model, action_dim)

    def encode(self, actions: torch.Tensor, qpos: torch.Tensor):
        """Returns (mu, logvar), each [B, z_dim]. Training only."""
        b = actions.shape[0]
        a = self.action_in(actions)                    # [B, k, d]
        q = self.qpos_in(qpos).unsqueeze(1)            # [B, 1, d]
        cls = self.cls_token.expand(b, -1, -1)         # [B, 1, d]
        seq = torch.cat([cls, q, a], dim=1)            # [B, k+2, d]
        seq = seq + self.pos_enc[:, : seq.shape[1]]
        enc = self.cvae_encoder(seq)
        cls_out = enc[:, 0]                            # [B, d]
        return self.mu_head(cls_out), self.logvar_head(cls_out)

    def decode(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        z_token = self.z_proj(z).unsqueeze(1)          # [B, 1, d]
        memory = torch.cat([z_token, context], dim=1)  # [B, N+1, d]
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # [B, k, d]
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)               # [B, k, action_dim]

    def forward(
        self,
        context: torch.Tensor,
        actions: torch.Tensor | None = None,
        qpos: torch.Tensor | None = None,
    ):
        """If ``actions`` and ``qpos`` are provided, runs the CVAE encoder and
        returns ``(pred, mu, logvar)``. Otherwise sets ``z = 0`` and returns
        ``pred`` only.
        """
        b = context.shape[0]
        if actions is not None and qpos is not None:
            mu, logvar = self.encode(actions, qpos)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            pred = self.decode(z, context)
            return pred, mu, logvar
        z = torch.zeros(b, self.z_dim, device=context.device, dtype=context.dtype)
        return self.decode(z, context)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ):
        """L = L1 + kl_weight * KL.

        KL is the standard analytical KL between N(mu, diag(exp(logvar)))
        and N(0, I); summed across z dims and averaged across batch.
        """
        l1 = F.l1_loss(pred, target)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = l1 + self.kl_weight * kl
        return total, l1, kl
