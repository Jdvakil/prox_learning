"""Shared-MLP encoder for whole-body ToF proximity readings.

Input  : [B, N_sensors, 8, 8] float (millimetres, normalized).
Output : [B, N_sensors, d_model] tokens to be concatenated with vision/proprio
         tokens before the ACT decoder.

Why a shared MLP?
  * All sensors are the same hardware (VL53L5CX 8x8 SPAD) — same encoding
    function makes physical sense.
  * The model gets ``N_sensors`` x more gradient signal per training step.
  * Sensor identity is recoverable downstream via positional embeddings if
    needed (see PROJECT.md §3.4).

Why a final LayerNorm?
  * Without it the resulting tokens have a different scale than the
    visual-language tokens coming out of Molmo2 (which the fusion concats
    against). Mismatched scales let the decoder ignore one stream entirely.
  * Empirically, removing the LayerNorm leads to encoder gradient collapse
    (grad_norm < 1e-8) — Day-6 sanity check in TIMELINE.md flags this.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ProximityEncoder(nn.Module):
    """Shared MLP applied independently to each sensor's 8x8 depth grid."""

    def __init__(self, n_sensors: int = 32, d_model: int = 512, hidden: int = 128) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h, w = x.shape
        assert (h, w) == (8, 8), f"expected 8x8 grid, got {h}x{w}"
        if s != self.n_sensors:
            raise ValueError(f"expected {self.n_sensors} sensors, got {s}")
        x = x.reshape(b * s, h * w)
        tokens = self.mlp(x)
        tokens = self.norm(tokens)
        return tokens.reshape(b, s, self.d_model)

    def verify_gradients(self) -> dict[str, float]:
        """Return the L2 grad-norm of every parameter — used by Day-6 check."""
        out: dict[str, float] = {}
        for name, p in self.named_parameters():
            if p.grad is not None:
                out[name] = float(p.grad.norm().item())
            else:
                out[name] = float("nan")
        return out


class HandcraftedToFEncoder(nn.Module):
    """Ablation encoder: replaces the learned MLP with engineered features.

    For each sensor we take three statistics from the 8x8 depth patch:
      * min depth (closest pixel)
      * mean depth
      * contact flag = 1 iff min < contact_thresh_mm (in normalized units)

    Concatenated and projected to a *single* token of d_model — intentionally
    coarser than PLA so the ablation tests whether per-sensor token granularity
    matters.
    """

    def __init__(
        self,
        n_sensors: int = 32,
        d_model: int = 512,
        contact_thresh_mm: float = 100.0,
        max_depth_mm: float = 4000.0,
    ) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.d_model = d_model
        # Threshold in *normalized* units; we assume depth was scaled by max range.
        self.register_buffer(
            "contact_thresh",
            torch.tensor(contact_thresh_mm / max_depth_mm, dtype=torch.float32),
        )
        self.proj = nn.Linear(n_sensors * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, h, w = x.shape
        assert (h, w) == (8, 8) and n == self.n_sensors
        flat = x.reshape(b, n, -1)               # [B, N, 64]
        min_d = flat.min(dim=-1).values           # [B, N]
        mean_d = flat.mean(dim=-1)                # [B, N]
        contact = (min_d < self.contact_thresh).float()
        feats = torch.cat([min_d, mean_d, contact], dim=-1)  # [B, 3N]
        return self.norm(self.proj(feats)).unsqueeze(1)       # [B, 1, d_model]


class Conv2DToFEncoder(nn.Module):
    """Ablation encoder: tiny per-sensor 2D ConvNet over the 8x8 grid.

    Tests whether spatial structure inside the 8x8 SPAD grid carries useful
    information that a flat MLP misses (e.g. direction of approach).
    """

    def __init__(self, n_sensors: int = 32, d_model: int = 512, mid_ch: int = 32) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.d_model = d_model
        self.conv = nn.Sequential(
            nn.Conv2d(1, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_ch, mid_ch * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(mid_ch * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, h, w = x.shape
        x = x.reshape(b * n, 1, h, w)
        feat = self.conv(x).flatten(1)            # [B*N, mid_ch*2]
        tokens = self.norm(self.proj(feat))
        return tokens.reshape(b, n, self.d_model)
