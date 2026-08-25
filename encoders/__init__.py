"""Proximity-skin encoders — one import, two functions.

Names are the *job*, not the coauthor:

- ``peak_closeness`` — 40-sensor snapshot → per-sensor peak closeness (PACT-raw)
- ``nearest_surface`` — causal 8×8 history → sensor-local XYZ (20 cm cap)
- ``surface_embedding`` — same net → frozen 32-d geometry embedding

Usage (repo root on ``PYTHONPATH``, which is true when you run from
``/home/jaydv/code/prox_learning``):

    from encoders import load_encoder

    prox = ...  # (B, 40, 8, 8) metres, same tensor PACT trains on

    raw = load_encoder("peak_closeness")
    feat = raw.policy_features(prox)          # (B, 40, 1) in [0, 1]

    geom = load_encoder("nearest_surface", checkpoint="surface_v1.pt")
    xyz = geom.policy_features(prox)          # (B, 40, 3) metres

    emb = load_encoder("surface_embedding", checkpoint="embed_v1.pt")
    z = emb.policy_features(prox)             # (B, 40, 32)

No checkpoint → randomly initialized geometry net (shapes still work).
Peak-closeness never needs weights.

ACT train/eval still import ``prox_cvae.ProxCVAEEncoder``; that file is a shim
to ``encoders.peak_closeness``. Math: README §10. Range trap: peak closeness
uses 50 cm; surface geometry uses 20 cm.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .peak_closeness import (
    DEFAULT_CKPT,
    D_MAX,
    DEAD_PIXEL_M,
    HYBRID_SKIN_SENSOR_ORDER,
    PeakClosenessEncoder,
    ProxCVAEEncoder,
    SafetyCVAE,
    feat_dim_for,
    featurize_np,
    featurize_torch,
    resolve_prox_layout,
    stack_obs_proximity,
)

if TYPE_CHECKING:
    from .surface_geometry import SurfaceGeometryEncoder

_ALIASES = {
    "raw": "peak_closeness",
    "peak": "peak_closeness",
    "pact_raw": "peak_closeness",
    "closeness": "peak_closeness",
    "trunk": "cvae_trunk",
    "cvae": "cvae_trunk",
    "delta": "cvae_delta",
    "retreat": "cvae_delta",
    "xyz": "nearest_surface",
    "surface_xyz": "nearest_surface",
    "surface_point": "nearest_surface",
    "surface": "nearest_surface",
    "embedding": "surface_embedding",
    "surface_embed": "surface_embedding",
    "geom_embed": "surface_embedding",
}


def list_encoders() -> dict[str, str]:
    """Canonical names → one-line job description."""
    return {
        "peak_closeness": (
            "Per-sensor peak closeness in [0, 1], 50 cm cap. No weights. "
            "Input (B, 40, 8, 8) m → (B, 40, 1)."
        ),
        "cvae_trunk": (
            "Frozen Safety-CVAE decoder trunk (256-d retreat embedding). "
            "Needs checkpoint dir with model.pt. Input (B, 40, 8, 8) m → (B, 1, 256)."
        ),
        "cvae_delta": (
            "Frozen Safety-CVAE 7-DoF joint retreat. Needs checkpoint dir. "
            "Input (B, 40, 8, 8) m → (B, 1, 7)."
        ),
        "nearest_surface": (
            "Shared conv-transformer. Nearest in-range XYZ, 20 cm cap. "
            "Input (B, 40, 8, 8) m → (B, 40, 3). Pass checkpoint for trained weights."
        ),
        "surface_embedding": (
            "Same net. Frozen 32-d geometry embedding. "
            "Input (B, 40, 8, 8) m → (B, 40, 32). Pass checkpoint for trained weights."
        ),
    }


def resolve_encoder_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in list_encoders():
        known = ", ".join(list_encoders())
        raise ValueError(f"unknown encoder {name!r}. Canonical names: {known}")
    return key


def load_encoder(
    name: str,
    *,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
    **kwargs: Any,
) -> PeakClosenessEncoder | SurfaceGeometryEncoder:
    """Build an encoder by function name. See ``list_encoders()``."""
    key = resolve_encoder_name(name)
    if key == "peak_closeness":
        layout = kwargs.pop("layout", "per_sensor")
        tokens_per_sensor = kwargs.pop("tokens_per_sensor", 8)
        if kwargs:
            raise TypeError(f"unexpected kwargs for peak_closeness: {sorted(kwargs)}")
        return PeakClosenessEncoder(
            ckpt_dir=checkpoint,
            feature="raw",
            device=device,
            layout=layout,
            tokens_per_sensor=tokens_per_sensor,
        )
    if key in ("cvae_trunk", "cvae_delta"):
        layout = kwargs.pop("layout", "global")
        tokens_per_sensor = kwargs.pop("tokens_per_sensor", 8)
        if kwargs:
            raise TypeError(f"unexpected kwargs for {key}: {sorted(kwargs)}")
        feature = "trunk" if key == "cvae_trunk" else "delta"
        return PeakClosenessEncoder(
            ckpt_dir=checkpoint,
            feature=feature,
            device=device,
            layout=layout,
            tokens_per_sensor=tokens_per_sensor,
        )
    if kwargs:
        raise TypeError(f"unexpected kwargs for {key}: {sorted(kwargs)}")
    from .surface_geometry import SurfaceGeometryEncoder as _SurfaceGeometryEncoder

    kind = "xyz" if key == "nearest_surface" else "embedding"
    return _SurfaceGeometryEncoder(kind=kind, checkpoint=checkpoint, device=device)


_SURFACE_EXPORTS = {
    "CAUSAL_FRAMES",
    "MAX_SURFACE_RANGE_M",
    "SURFACE_EMBEDDING_DIM",
    "SurfaceEmbeddingEncoder",
    "SurfaceGeometryEncoder",
    "SurfaceProximityEncoder",
    "causal_sensor_window",
    "depth_to_closeness",
    "load_frozen_proximity_encoder",
    "load_frozen_surface_embedding_encoder",
    "load_frozen_surface_encoder",
    "nearest_surface_target",
    "parameter_count",
    "to_causal_closeness",
}


def __getattr__(name: str):
    if name in _SURFACE_EXPORTS:
        from . import surface_geometry as _sg

        return getattr(_sg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "CAUSAL_FRAMES",
    "DEFAULT_CKPT",
    "D_MAX",
    "DEAD_PIXEL_M",
    "HYBRID_SKIN_SENSOR_ORDER",
    "MAX_SURFACE_RANGE_M",
    "PeakClosenessEncoder",
    "ProxCVAEEncoder",
    "SURFACE_EMBEDDING_DIM",
    "SafetyCVAE",
    "SurfaceEmbeddingEncoder",
    "SurfaceGeometryEncoder",
    "SurfaceProximityEncoder",
    "causal_sensor_window",
    "depth_to_closeness",
    "feat_dim_for",
    "featurize_np",
    "featurize_torch",
    "list_encoders",
    "load_encoder",
    "load_frozen_proximity_encoder",
    "load_frozen_surface_embedding_encoder",
    "load_frozen_surface_encoder",
    "nearest_surface_target",
    "parameter_count",
    "resolve_encoder_name",
    "resolve_prox_layout",
    "stack_obs_proximity",
    "to_causal_closeness",
]
