"""Frozen Molmo2-4B vision-language backbone.

We use Molmo2 to produce ``[B, N_vis, d_vlm]`` visual tokens from RGB frames
plus a language instruction. All Molmo2 weights are frozen — only the linear
projection from ``d_vlm`` to ``d_model`` is trained. This is what makes PLA
fit in 9 GB of VRAM during fine-tuning.

Two backends are supplied:

* ``FrozenMolmo2`` — the real backbone. Loads ``allenai/Molmo-4B-D-0924`` via
  HuggingFace transformers, takes an RGB tensor in [0, 1], runs the vision
  tower, returns last-layer hidden states. The exact API drifts across Molmo2
  releases — see ``submodules/MolmoBot/MolmoBot/olmo/models/molmobot/`` for
  the current call signature, and update ``_extract_vision_tokens`` if it
  diverges.

* ``DummyVLBackbone`` — for sanity checks and unit tests. Returns random tokens
  with the documented shape so the rest of the pipeline can be exercised
  without a 9 GB model load. PLA forward-pass tests use this.

Why freeze the backbone?
    The ablation in PROJECT.md §6 shows fine-tuning Molmo2 with proximity
    data starves the vision tower (3.7B params) of gradient signal — it
    forgets common-sense visual recognition while gaining nothing on
    proximity. Frozen backbone + small projection = stable + fast.
"""
from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.nn as nn


class DummyVLBackbone(nn.Module):
    """Random-token backbone used for forward-pass and shape tests."""

    def __init__(self, d_model: int = 512, n_vis_tokens: int = 192) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_vis = n_vis_tokens
        # Tiny trainable projection so a non-zero gradient flows through this
        # path during forward-pass tests.
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        rgb: torch.Tensor,
        language: Iterable[str] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rgb.ndim == 5:
            b = rgb.shape[0]
        else:
            b = rgb.shape[0]
        out = torch.randn(
            b, self.n_vis, self.d_model, device=rgb.device, dtype=rgb.dtype
        )
        return self.proj(out)


class FrozenMolmo2(nn.Module):
    """HuggingFace Molmo2-4B vision tower frozen + linear projection trained.

    Loaded lazily on first forward to keep ``import pla`` cheap. If
    ``transformers`` is not installed the constructor raises.
    """

    def __init__(
        self,
        model_name: str = "allenai/Molmo-4B-D-0924",
        d_model: int = 512,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.d_model = d_model
        self._device = device
        self._dtype = dtype
        self.proj: nn.Linear | None = None  # built when we know d_vlm
        self._processor = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers is required for FrozenMolmo2. "
                "Install with `pip install transformers` or use DummyVLBackbone "
                "for sanity tests."
            ) from e

        local_only = bool(os.environ.get("PLA_HF_LOCAL_ONLY"))
        self._processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True, local_files_only=local_only,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self._dtype,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        for p in self._model.parameters():
            p.requires_grad = False
        d_vlm = int(self._model.config.hidden_size)
        self.proj = nn.Linear(d_vlm, self.d_model)
        if self._device is not None:
            self._model.to(self._device)
            self.proj.to(self._device)

    @torch.no_grad()
    def _extract_vision_tokens(
        self, rgb_frames: torch.Tensor, language: list[str]
    ) -> torch.Tensor:
        """Returns last-hidden-state ``[B, seq_len, d_vlm]`` for a single frame.

        ``rgb_frames`` is ``[B, 3, H, W]`` in [0, 1]. The exact processor call
        differs slightly between Molmo2 release tags — the implementation here
        follows the convention used by MolmoBot inference; if upstream changes
        you will see a runtime error from ``self._processor`` and should adjust
        the kwargs dict below.
        """
        assert self._processor is not None and self._model is not None
        proc_inputs = self._processor(
            images=rgb_frames,
            text=language,
            return_tensors="pt",
        )
        proc_inputs = {
            k: v.to(rgb_frames.device) if hasattr(v, "to") else v
            for k, v in proc_inputs.items()
        }
        out = self._model(**proc_inputs, output_hidden_states=True)
        return out.hidden_states[-1]

    def forward(
        self, rgb: torch.Tensor, language: list[str]
    ) -> torch.Tensor:
        """rgb: ``[B, K, 3, H, W]`` or ``[B, 3, H, W]``. Returns ``[B, N_vis, d_model]``.

        For ``K`` frames the per-frame token grids are mean-pooled along the
        frame axis (after extraction). This is the cheap option; the
        cross-attn fusion ablation (see ``fusion.py``) has more capacity to
        attend across frames if needed.
        """
        self._ensure_loaded()
        assert self.proj is not None
        if rgb.ndim == 5:
            b, k = rgb.shape[:2]
            per_frame = []
            for i in range(k):
                tokens = self._extract_vision_tokens(rgb[:, i], language)
                per_frame.append(tokens)
            stacked = torch.stack(per_frame, dim=1)  # [B, K, seq_len, d_vlm]
            tokens = stacked.mean(dim=1)             # [B, seq_len, d_vlm]
        else:
            tokens = self._extract_vision_tokens(rgb, language)

        return self.proj(tokens.float())
