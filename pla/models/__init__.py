"""Model layer.

Top-level exports:
    PLA                — full peripersonal language-action model
    VLMOnlyACT         — PLA(vlm_only=True), the headline baseline
    PropOnlyMLP        — qpos -> action chunk floor baseline
    ProximityEncoder   — shared MLP over [N, 8, 8] depth grids
    HandcraftedToFEncoder, Conv2DToFEncoder — encoder ablations
    ModalityFusion     — concat (primary) / cross-attn (ablation)
    ACTDecoder         — Action Chunking Transformer head
    FrozenMolmo2, DummyVLBackbone — vision-language backbones
"""
from pla.models.act import ACTDecoder
from pla.models.baselines import PropOnlyMLP, VLMOnlyACT
from pla.models.fusion import ModalityFusion
from pla.models.pla import PLA, build_proximity_encoder
from pla.models.proximity_encoder import (
    Conv2DToFEncoder,
    HandcraftedToFEncoder,
    ProximityEncoder,
)
from pla.models.vlm_backbone import DummyVLBackbone, FrozenMolmo2

__all__ = [
    "ACTDecoder",
    "Conv2DToFEncoder",
    "DummyVLBackbone",
    "FrozenMolmo2",
    "HandcraftedToFEncoder",
    "ModalityFusion",
    "PLA",
    "PropOnlyMLP",
    "ProximityEncoder",
    "VLMOnlyACT",
    "build_proximity_encoder",
]
