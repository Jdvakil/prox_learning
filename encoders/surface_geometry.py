"""Surface-geometry skin encoder (nearest XYZ / 32-d embedding).

Function: map a causal 8x8 depth history on one sensor to the nearest in-range
surface point in the sensor-local OpenCV frame, or to a 32-d geometry embedding.
Weights are shared across all 40 sensors.

The low-level nets (``SurfaceProximityEncoder``, ``SurfaceEmbeddingEncoder``)
take ``(B, 32, 8, 8)`` closeness. ``SurfaceGeometryEncoder`` is the easy wrapper:
throw a full-skin metres tensor at it (same layout as PACT-raw) and get
``(B, 40, 3)`` or ``(B, 40, 32)``.

Distances beyond 20 cm are invalid, not regression targets. That cap is **not**
the 50 cm closeness used by peak-closeness / PACT-raw (README §10).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ._paths import ensure_act_on_path

ensure_act_on_path()
from hybrid_skin_sensors import HYBRID_SKIN_SENSOR_ORDER  # noqa: E402

MAX_SURFACE_RANGE_M = 0.20
SENSOR_FOVY_DEG = 45.0
CAUSAL_CONTROL_STEPS = 8
SUBFRAMES_PER_CONTROL_STEP = 4
CAUSAL_FRAMES = CAUSAL_CONTROL_STEPS * SUBFRAMES_PER_CONTROL_STEP
SURFACE_EMBEDDING_DIM = 32


def native_camera_intrinsic(
    height: int = 8,
    width: int = 8,
    fovy_deg: float = SENSOR_FOVY_DEG,
) -> np.ndarray:
    """Intrinsic for the dedicated square native-resolution MuJoCo renderer."""
    fy = 0.5 * float(height) / math.tan(math.radians(fovy_deg) / 2.0)
    fx = fy
    return np.asarray(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def nearest_surface_target(
    depth: np.ndarray,
    *,
    max_range_m: float = MAX_SURFACE_RANGE_M,
    fovy_deg: float = SENSOR_FOVY_DEG,
) -> tuple[np.ndarray, bool]:
    """Nearest valid depth pixel as sensor-local XYZ.

    ``depth`` is the most recent native 8x8 frame. Pixel centers use
    ``(u + 0.5, v + 0.5)`` and OpenCV axes (+x right, +y down, +z forward).
    """
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"depth must be 2-D, got {values.shape}")
    valid = np.isfinite(values) & (values > 0.0) & (values <= max_range_m)
    if not np.any(valid):
        return np.zeros(3, dtype=np.float32), False
    masked = np.where(valid, values, np.inf)
    flat_index = int(np.argmin(masked))
    v, u = np.unravel_index(flat_index, values.shape)
    z = float(values[v, u])
    intrinsic = native_camera_intrinsic(values.shape[0], values.shape[1], fovy_deg)
    x = (float(u) + 0.5 - float(intrinsic[0, 2])) * z / float(intrinsic[0, 0])
    y = (float(v) + 0.5 - float(intrinsic[1, 2])) * z / float(intrinsic[1, 1])
    return np.asarray([x, y, z], dtype=np.float32), True


def depth_to_closeness(
    depth: np.ndarray,
    *,
    max_range_m: float = MAX_SURFACE_RANGE_M,
) -> np.ndarray:
    """Map valid in-range depth to [0,1] closeness; all other pixels to zero."""
    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0.0) & (values <= max_range_m)
    output = np.zeros_like(values, dtype=np.float32)
    output[valid] = 1.0 - values[valid] / float(max_range_m)
    return output


def depth_to_closeness_torch(
    depth: torch.Tensor,
    *,
    max_range_m: float = MAX_SURFACE_RANGE_M,
) -> torch.Tensor:
    """Torch cousin of ``depth_to_closeness``; any leading shape is fine."""
    values = depth.float()
    valid = torch.isfinite(values) & (values > 0.0) & (values <= max_range_m)
    return torch.where(
        valid,
        1.0 - values / float(max_range_m),
        torch.zeros_like(values),
    )


def causal_sensor_window(
    episode_proximity: np.ndarray,
    timestep: int,
    sensor_index: int,
    *,
    control_steps: int = CAUSAL_CONTROL_STEPS,
) -> np.ndarray:
    """Return a left-padded ``(32,8,8)`` causal window for one sensor."""
    values = np.asarray(episode_proximity)
    if values.ndim != 5 or values.shape[2:] != (4, 8, 8):
        raise ValueError(f"expected (T,S,4,8,8), got {values.shape}")
    if not 0 <= timestep < values.shape[0]:
        raise IndexError(timestep)
    if not 0 <= sensor_index < values.shape[1]:
        raise IndexError(sensor_index)
    start = max(0, timestep - control_steps + 1)
    window = values[start : timestep + 1, sensor_index]
    if len(window) < control_steps:
        pad = np.repeat(window[:1], control_steps - len(window), axis=0)
        window = np.concatenate((pad, window), axis=0)
    flattened = window.reshape(control_steps * 4, 8, 8)
    return depth_to_closeness(flattened)


def as_subframe_episode(proximity: np.ndarray) -> np.ndarray:
    """ACT-pooled ``(T, S, 8, 8)`` or native ``(T, S, 4, 8, 8)`` -> ``(T, S, 4, 8, 8)``.

    Convert writes pooled 8x8 tiles. The geometry net wants 4 native subframes
    per control step. Repeating the pooled frame four times is the adapter.
    """
    values = np.asarray(proximity, dtype=np.float32)
    if values.ndim == 5 and values.shape[2:] == (4, 8, 8):
        return values
    if values.ndim == 4 and values.shape[-2:] == (8, 8):
        return np.repeat(values[:, :, None], SUBFRAMES_PER_CONTROL_STEP, axis=2)
    raise ValueError(f"expected (T,S,8,8) or (T,S,4,8,8), got {values.shape}")


def to_causal_closeness(
    skin: torch.Tensor | np.ndarray,
    *,
    unit: str = "metres",
) -> tuple[torch.Tensor, bool, bool]:
    """Normalize common skin layouts to ``(B, S, 32, 8, 8)`` closeness.

    Accepted layouts (last two axes always 8x8):

    * ``(8, 8)`` one sensor, current frame
    * ``(32, 8, 8)`` one sensor, causal window
    * ``(S, 8, 8)`` full skin, current frame, no batch
    * ``(B, 32, 8, 8)`` one sensor, causal window
    * ``(B, S, 8, 8)`` full skin, current frame (PACT-raw layout)
    * ``(B, S, 4, 8, 8)`` full skin, one control step with 4 subframes
    * ``(B, S, 32, 8, 8)`` full skin, causal windows

    ``unit='metres'`` runs the 20 cm closeness map. ``unit='closeness'`` skips it
    (use this for windows already produced by ``causal_sensor_window``).

    Returns ``(windows, squeeze_batch, squeeze_sensor)``.
    """
    if unit not in ("metres", "closeness"):
        raise ValueError(f"unit must be 'metres' or 'closeness', got {unit!r}")
    x = torch.as_tensor(skin, dtype=torch.float32)
    squeeze_batch = False
    squeeze_sensor = False

    if x.ndim == 2:
        if x.shape != (8, 8):
            raise ValueError(f"expected (8, 8), got {tuple(x.shape)}")
        x = x.view(1, 1, 8, 8)
        squeeze_batch = True
        squeeze_sensor = True
    elif x.ndim == 3:
        if x.shape[-2:] != (8, 8):
            raise ValueError(f"expected (*, 8, 8), got {tuple(x.shape)}")
        if x.shape[0] == CAUSAL_FRAMES:
            x = x.view(1, 1, CAUSAL_FRAMES, 8, 8)
            squeeze_batch = True
            squeeze_sensor = True
        else:
            x = x.unsqueeze(0)
            squeeze_batch = True
    elif x.ndim == 4:
        if x.shape[-2:] != (8, 8):
            raise ValueError(f"expected (*, *, 8, 8), got {tuple(x.shape)}")
        if x.shape[1] == CAUSAL_FRAMES:
            x = x.unsqueeze(1)
            squeeze_sensor = True
        # else (B, S, 8, 8)
    elif x.ndim == 5:
        if x.shape[-2:] != (8, 8):
            raise ValueError(f"expected (..., 8, 8), got {tuple(x.shape)}")
        if x.shape[2] == SUBFRAMES_PER_CONTROL_STEP:
            batch, n_sensors, sub, height, width = x.shape
            x = (
                x.unsqueeze(2)
                .expand(batch, n_sensors, CAUSAL_CONTROL_STEPS, sub, height, width)
                .reshape(batch, n_sensors, CAUSAL_FRAMES, height, width)
            )
        elif x.shape[2] != CAUSAL_FRAMES:
            raise ValueError(
                f"5-D skin dim 2 must be {SUBFRAMES_PER_CONTROL_STEP} or "
                f"{CAUSAL_FRAMES}, got {tuple(x.shape)}"
            )
    else:
        raise ValueError(f"skin must be 2-D to 5-D, got {tuple(x.shape)}")

    if unit == "metres":
        x = depth_to_closeness_torch(x)

    if x.ndim == 4:
        # (B, S, 8, 8) current frame: tile across the 32-frame history.
        x = x.unsqueeze(2).expand(-1, -1, CAUSAL_FRAMES, -1, -1).clone()
    return x, squeeze_batch, squeeze_sensor


class SinusoidalSequenceEncoding(nn.Module):
    def __init__(self, length: int, width: int) -> None:
        super().__init__()
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32)
            * (-math.log(10000.0) / width)
        )
        encoding = torch.zeros(length, width, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence + self.encoding[:, : sequence.shape[1]]


def _transformer_encoder(layer: nn.Module, num_layers: int) -> nn.TransformerEncoder:
    """norm_first blocks nested-tensor fast path; disable it to skip the torch warning."""
    try:
        return nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=num_layers)


class SurfaceProximityEncoder(nn.Module):
    """~0.82M-parameter conv-stem transformer shared by all 40 sensors."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.frame_projection = nn.Linear(32 * 8 * 8, 128)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.position = SinusoidalSequenceEncoding(CAUSAL_FRAMES + 1, 128)
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = _transformer_encoder(layer, num_layers=4)
        self.head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 4),
        )
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.ndim != 4 or frames.shape[1:] != (CAUSAL_FRAMES, 8, 8):
            raise ValueError(
                f"expected (B,{CAUSAL_FRAMES},8,8), got {tuple(frames.shape)}"
            )
        batch = frames.shape[0]
        features = self.conv_stem(frames.reshape(batch * CAUSAL_FRAMES, 1, 8, 8))
        tokens = self.frame_projection(features.flatten(1)).reshape(
            batch, CAUSAL_FRAMES, 128
        )
        cls = self.cls_token.expand(batch, -1, -1)
        encoded = self.transformer(self.position(torch.cat((cls, tokens), dim=1)))
        output = self.head(encoded[:, 0])
        xyz_normalized = output[:, :3]
        validity_logit = output[:, 3]
        return xyz_normalized, validity_logit

    @torch.no_grad()
    def predict(
        self,
        frames: torch.Tensor,
        *,
        validity_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz_normalized, logits = self(frames)
        probabilities = torch.sigmoid(logits)
        valid = probabilities >= validity_threshold
        xyz_m = xyz_normalized * MAX_SURFACE_RANGE_M
        xyz_m = torch.where(valid[:, None], xyz_m, torch.zeros_like(xyz_m))
        return xyz_m, valid, probabilities

    @torch.no_grad()
    def policy_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Return the frozen v1 policy input (the predicted local XYZ point)."""
        xyz_m, _valid, _probabilities = self.predict(frames)
        return xyz_m


class SurfaceEmbeddingEncoder(nn.Module):
    """Frozen 32-D geometry embedding with surface/validity auxiliary heads.

    The policy consumes the embedding. A reconstruction head preserves the
    latest native 8x8 closeness map, while the auxiliary head keeps the v1
    nearest-surface and validity metrics directly comparable.
    """

    def __init__(self, embedding_dim: int = SURFACE_EMBEDDING_DIM) -> None:
        super().__init__()
        if embedding_dim != SURFACE_EMBEDDING_DIM:
            raise ValueError(
                f"frozen screen requires embedding_dim={SURFACE_EMBEDDING_DIM}"
            )
        self.embedding_dim = int(embedding_dim)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.frame_projection = nn.Linear(32 * 8 * 8, 128)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.position = SinusoidalSequenceEncoding(CAUSAL_FRAMES + 1, 128)
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = _transformer_encoder(layer, num_layers=4)
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )
        self.auxiliary_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if frames.ndim != 4 or frames.shape[1:] != (CAUSAL_FRAMES, 8, 8):
            raise ValueError(
                f"expected (B,{CAUSAL_FRAMES},8,8), got {tuple(frames.shape)}"
            )
        batch = frames.shape[0]
        features = self.conv_stem(
            frames.reshape(batch * CAUSAL_FRAMES, 1, 8, 8)
        )
        tokens = self.frame_projection(features.flatten(1)).reshape(
            batch, CAUSAL_FRAMES, 128
        )
        cls = self.cls_token.expand(batch, -1, -1)
        encoded = self.transformer(
            self.position(torch.cat((cls, tokens), dim=1))
        )
        embedding = self.embedding_head(encoded[:, 0])
        auxiliary = self.auxiliary_head(embedding)
        xyz_normalized = auxiliary[:, :3]
        validity_logit = auxiliary[:, 3]
        reconstructed_closeness = torch.sigmoid(
            self.reconstruction_head(embedding)
        ).reshape(batch, 8, 8)
        return (
            embedding,
            xyz_normalized,
            validity_logit,
            reconstructed_closeness,
        )

    @torch.no_grad()
    def predict(
        self,
        frames: torch.Tensor,
        *,
        validity_threshold: float = 0.5,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        embedding, xyz_normalized, logits, reconstruction = self(frames)
        probabilities = torch.sigmoid(logits)
        valid = probabilities >= validity_threshold
        xyz_m = xyz_normalized * MAX_SURFACE_RANGE_M
        xyz_m = torch.where(valid[:, None], xyz_m, torch.zeros_like(xyz_m))
        return embedding, xyz_m, valid, probabilities, reconstruction

    @torch.no_grad()
    def policy_features(self, frames: torch.Tensor) -> torch.Tensor:
        embedding, _xyz, _valid, _probabilities, _reconstruction = self.predict(
            frames
        )
        return embedding


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_frozen_surface_encoder(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SurfaceProximityEncoder, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    if payload.get("schema_version") != "pact_surface_encoder_v1":
        raise ValueError("not a pact_surface_encoder_v1 checkpoint")
    if payload.get("frozen") is not True:
        raise ValueError("front-end checkpoint is not marked frozen")
    model = SurfaceProximityEncoder()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def load_frozen_surface_embedding_encoder(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SurfaceEmbeddingEncoder, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    if payload.get("schema_version") != "pact_surface_embedding_encoder_v1":
        raise ValueError("not a pact_surface_embedding_encoder_v1 checkpoint")
    if payload.get("frozen") is not True:
        raise ValueError("front-end checkpoint is not marked frozen")
    if payload.get("policy_feature_dim") != SURFACE_EMBEDDING_DIM:
        raise ValueError("front-end policy feature dimension changed")
    model = SurfaceEmbeddingEncoder()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def load_frozen_proximity_encoder(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load either frozen front-end without changing the legacy v1 loader."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    schema = payload.get("schema_version")
    if schema == "pact_surface_encoder_v1":
        return load_frozen_surface_encoder(
            checkpoint_path, map_location=map_location
        )
    if schema == "pact_surface_embedding_encoder_v1":
        return load_frozen_surface_embedding_encoder(
            checkpoint_path, map_location=map_location
        )
    raise ValueError(f"unsupported frozen proximity encoder schema: {schema}")


class SurfaceGeometryEncoder(nn.Module):
    """Full-skin wrapper around the shared conv-transformer.

    ``kind='xyz'`` -> ``(B, S, 3)`` local metres (zeros when invalid).
    ``kind='embedding'`` -> ``(B, S, 32)`` geometry embedding.

    Without ``checkpoint`` the net is randomly initialized (shape tests / wiring).
    Pass a frozen ``pact_surface_*_v1`` checkpoint for real features.
    """

    def __init__(
        self,
        kind: str = "xyz",
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        *,
        inner: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if kind not in ("xyz", "embedding"):
            raise ValueError(f"kind must be 'xyz' or 'embedding', got {kind!r}")
        self.kind = kind
        self.name = "nearest_surface" if kind == "xyz" else "surface_embedding"
        self.sensor_order: list[str] = list(HYBRID_SKIN_SENSOR_ORDER)
        self.n_sensors = len(self.sensor_order)
        self.n_act_sensors = self.n_sensors
        self.layout = "per_sensor"
        self.tokens_per_sensor = 1
        self.device = device
        self.payload: dict[str, Any] | None = None

        if inner is not None:
            self.inner = inner
        elif checkpoint is not None:
            loader = (
                load_frozen_surface_encoder
                if kind == "xyz"
                else load_frozen_surface_embedding_encoder
            )
            self.inner, self.payload = loader(checkpoint, map_location=device)
        else:
            self.inner = (
                SurfaceProximityEncoder()
                if kind == "xyz"
                else SurfaceEmbeddingEncoder()
            )
            self.inner.eval()

        self.act_feat_dim = 3 if kind == "xyz" else SURFACE_EMBEDDING_DIM
        self.feat_dim = self.act_feat_dim
        self.inner.eval()
        for parameter in self.inner.parameters():
            parameter.requires_grad_(False)
        self.to(device)
        self.eval()

    def _windows_on_device(
        self,
        skin: torch.Tensor | np.ndarray,
        unit: str,
    ) -> tuple[torch.Tensor, bool, bool]:
        windows, squeeze_batch, squeeze_sensor = to_causal_closeness(skin, unit=unit)
        device = next(self.parameters()).device
        return windows.to(device), squeeze_batch, squeeze_sensor

    def _reshape_out(
        self,
        feat: torch.Tensor,
        batch: int,
        n_sensors: int,
        squeeze_batch: bool,
        squeeze_sensor: bool,
    ) -> torch.Tensor:
        feat = feat.reshape(batch, n_sensors, -1)
        if squeeze_sensor:
            feat = feat.squeeze(1)
        if squeeze_batch:
            feat = feat.squeeze(0)
        return feat

    @torch.no_grad()
    def policy_features(
        self,
        skin: torch.Tensor | np.ndarray,
        *,
        unit: str = "metres",
    ) -> torch.Tensor:
        """Encode a skin tensor. See ``to_causal_closeness`` for accepted shapes."""
        windows, squeeze_batch, squeeze_sensor = self._windows_on_device(skin, unit)
        batch, n_sensors = windows.shape[:2]
        flat = windows.reshape(batch * n_sensors, CAUSAL_FRAMES, 8, 8)
        feat = self.inner.policy_features(flat)
        return self._reshape_out(feat, batch, n_sensors, squeeze_batch, squeeze_sensor)

    @torch.no_grad()
    def forward(
        self,
        skin: torch.Tensor | np.ndarray,
        *,
        unit: str = "metres",
    ) -> torch.Tensor:
        return self.policy_features(skin, unit=unit)

    @torch.no_grad()
    def predict_skin(
        self,
        skin: torch.Tensor | np.ndarray,
        *,
        unit: str = "metres",
        validity_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Inner ``predict`` over a full skin; keys depend on ``kind``."""
        windows, squeeze_batch, squeeze_sensor = self._windows_on_device(skin, unit)
        batch, n_sensors = windows.shape[:2]
        flat = windows.reshape(batch * n_sensors, CAUSAL_FRAMES, 8, 8)
        raw = self.inner.predict(flat, validity_threshold=validity_threshold)

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.reshape(batch, n_sensors, *tensor.shape[1:])
            if squeeze_sensor:
                tensor = tensor.squeeze(1)
            if squeeze_batch:
                tensor = tensor.squeeze(0)
            return tensor

        if self.kind == "xyz":
            xyz_m, valid, probabilities = raw
            return {
                "xyz_m": pack(xyz_m),
                "valid": pack(valid),
                "probabilities": pack(probabilities),
            }
        embedding, xyz_m, valid, probabilities, reconstruction = raw
        return {
            "embedding": pack(embedding),
            "xyz_m": pack(xyz_m),
            "valid": pack(valid),
            "probabilities": pack(probabilities),
            "reconstruction": pack(reconstruction),
        }

    @torch.no_grad()
    def encode_pooled_history(
        self,
        history: torch.Tensor | np.ndarray,
        *,
        unit: str = "metres",
    ) -> torch.Tensor:
        """Encode last ``H<=8`` pooled control steps.

        ``history`` is ``(B, H, S, 8, 8)`` or ``(H, S, 8, 8)`` metres (ACT convert
        layout). Left-pads to 8 steps, repeats each step into 4 fake subframes,
        then runs the shared transformer. Returns ``(B, S, feat)`` / ``(S, feat)``.
        """
        x = torch.as_tensor(history, dtype=torch.float32)
        squeeze_batch = False
        if x.ndim == 3 and x.shape[-2:] == (8, 8):
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze_batch = True
        elif x.ndim == 4 and x.shape[-2:] == (8, 8):
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.ndim != 5 or x.shape[-2:] != (8, 8):
            raise ValueError(
                f"encode_pooled_history expected (B,H,S,8,8) or (H,S,8,8), got {tuple(x.shape)}"
            )
        batch, n_hist, n_sensors, height, width = x.shape
        if n_hist < CAUSAL_CONTROL_STEPS:
            pad = x[:, :1].expand(batch, CAUSAL_CONTROL_STEPS - n_hist, n_sensors, height, width)
            x = torch.cat((pad, x), dim=1)
        elif n_hist > CAUSAL_CONTROL_STEPS:
            x = x[:, -CAUSAL_CONTROL_STEPS:]
        # (B, 8, S, 8, 8) -> (B, S, 32, 8, 8)
        x = x.permute(0, 2, 1, 3, 4)
        x = (
            x.unsqueeze(3)
            .expand(-1, -1, -1, SUBFRAMES_PER_CONTROL_STEP, -1, -1)
            .reshape(batch, n_sensors, CAUSAL_FRAMES, height, width)
        )
        feat = self.policy_features(x, unit=unit)
        if squeeze_batch:
            return feat[0]
        return feat

    @torch.no_grad()
    def encode_episode(
        self,
        episode_proximity: np.ndarray,
        *,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """``(T, S, 8, 8)`` or ``(T, S, 4, 8, 8)`` metres -> ``(T, S, feat_dim)``."""
        packed = self.encode_episode_full(episode_proximity, batch_size=batch_size)
        key = "embedding" if self.kind == "embedding" else "xyz_m"
        return packed[key]

    @torch.no_grad()
    def encode_episode_full(
        self,
        episode_proximity: np.ndarray,
        *,
        batch_size: int = 512,
        validity_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Same as ``encode_episode`` plus XYZ / validity for the token writer."""
        values = as_subframe_episode(episode_proximity)
        n_steps, n_sensors = values.shape[:2]
        windows = np.empty(
            (n_steps, n_sensors, CAUSAL_FRAMES, 8, 8), dtype=np.float32
        )
        for timestep in range(n_steps):
            for sensor_index in range(n_sensors):
                windows[timestep, sensor_index] = causal_sensor_window(
                    values, timestep, sensor_index
                )
        device = next(self.parameters()).device
        flat = torch.from_numpy(windows).reshape(
            n_steps * n_sensors, CAUSAL_FRAMES, 8, 8
        )
        chunks: list[dict[str, torch.Tensor]] = []
        for start in range(0, flat.shape[0], int(batch_size)):
            piece = flat[start : start + batch_size].to(device)
            raw = self.inner.predict(piece, validity_threshold=validity_threshold)
            if self.kind == "xyz":
                xyz_m, valid, probabilities = raw
                chunks.append(
                    {
                        "xyz_m": xyz_m.cpu(),
                        "valid": valid.cpu(),
                        "probabilities": probabilities.cpu(),
                    }
                )
            else:
                embedding, xyz_m, valid, probabilities, reconstruction = raw
                chunks.append(
                    {
                        "embedding": embedding.cpu(),
                        "xyz_m": xyz_m.cpu(),
                        "valid": valid.cpu(),
                        "probabilities": probabilities.cpu(),
                        "reconstruction": reconstruction.cpu(),
                    }
                )

        def _cat(name: str) -> torch.Tensor:
            tensor = torch.cat([chunk[name] for chunk in chunks], dim=0)
            return tensor.reshape(n_steps, n_sensors, *tensor.shape[1:])

        return {name: _cat(name) for name in chunks[0]}
