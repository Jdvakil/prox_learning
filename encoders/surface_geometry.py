"""Surface-geometry skin encoder (nearest XYZ / 32-d embedding).

Function: map a causal 8x8 depth history on one sensor to the nearest in-range
surface point in the sensor-local OpenCV frame, or to a 32-d geometry embedding.
Weights are shared across all 40 sensors.

The low-level nets (``SurfaceProximityEncoder``, ``SurfaceEmbeddingEncoder``)
take ``(B, 32, 8, 8)`` closeness. ``SurfaceGeometryEncoder`` is the easy wrapper:
throw a full-skin metres tensor at it (same layout as PACT-raw) and get
``(B, 40, 3)`` or ``(B, 40, 32)``.

Distances below 5 mm (dead pixels) or beyond 20 cm are invalid, not regression
targets. That cap is **not** the 50 cm closeness used by peak-closeness /
PACT-raw (README §10).
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
MIN_SURFACE_RANGE_M = 0.005
SENSOR_FOVY_DEG = 45.0
CAUSAL_CONTROL_STEPS = 8
SUBFRAMES_PER_CONTROL_STEP = 4
CAUSAL_FRAMES = CAUSAL_CONTROL_STEPS * SUBFRAMES_PER_CONTROL_STEP
SURFACE_EMBEDDING_DIM = 32
SURFACE_READOUT_DIM = 128
SCHEMA_SURFACE_XYZ = "pact_surface_encoder_v1"
SCHEMA_SURFACE_EMBEDDING = "pact_surface_embedding_encoder_v1"
POLICY_TAPS = frozenset({"embedding", "readout", "xyz"})


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
    """Minimum valid axial-depth pixel as sensor-local XYZ.

    ``depth`` is the most recent native 8x8 frame. Pixel centers use
    ``(u + 0.5, v + 0.5)`` and OpenCV axes (+x right, +y down, +z forward).
    """
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"depth must be 2-D, got {values.shape}")
    valid = (
        np.isfinite(values)
        & (values >= MIN_SURFACE_RANGE_M)
        & (values <= max_range_m)
    )
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


def nearest_surface_target_batch(
    depth: np.ndarray,
    *,
    max_range_m: float = MAX_SURFACE_RANGE_M,
    fovy_deg: float = SENSOR_FOVY_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ``nearest_surface_target``. ``depth`` is ``(..., H, W)`` metres.

    Returns XYZ ``(..., 3)`` and a boolean valid mask ``(...)``.
    """
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError(f"depth must be at least 2-D, got {values.shape}")
    height, width = values.shape[-2:]
    valid_pixel = (
        np.isfinite(values)
        & (values >= MIN_SURFACE_RANGE_M)
        & (values <= max_range_m)
    )
    valid = valid_pixel.any(axis=(-2, -1))
    masked = np.where(valid_pixel, values, np.inf)
    flat = masked.reshape(*values.shape[:-2], height * width)
    flat_index = np.argmin(flat, axis=-1)
    row = flat_index // width
    col = flat_index % width
    z = np.take_along_axis(
        values.reshape(*values.shape[:-2], height * width),
        flat_index[..., None],
        axis=-1,
    )[..., 0]
    intrinsic = native_camera_intrinsic(height, width, fovy_deg)
    x = (col.astype(np.float32) + 0.5 - float(intrinsic[0, 2])) * z / float(
        intrinsic[0, 0]
    )
    y = (row.astype(np.float32) + 0.5 - float(intrinsic[1, 2])) * z / float(
        intrinsic[1, 1]
    )
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    xyz = np.where(valid[..., None], xyz, np.zeros_like(xyz))
    return xyz, valid


def depth_to_closeness(
    depth: np.ndarray,
    *,
    max_range_m: float = MAX_SURFACE_RANGE_M,
) -> np.ndarray:
    """Map valid in-range depth to [0,1] closeness; all other pixels to zero."""
    values = np.asarray(depth, dtype=np.float32)
    valid = (
        np.isfinite(values)
        & (values >= MIN_SURFACE_RANGE_M)
        & (values <= max_range_m)
    )
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
    valid = (
        torch.isfinite(values)
        & (values >= MIN_SURFACE_RANGE_M)
        & (values <= max_range_m)
    )
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
        self.n_readout = 1
        self.readout_dim = SURFACE_READOUT_DIM

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Frame tokens + CLS readout. ``(B, 1+32, 128)``. Gradients flow."""
        if frames.ndim != 4 or frames.shape[1:] != (CAUSAL_FRAMES, 8, 8):
            raise ValueError(
                f"expected (B,{CAUSAL_FRAMES},8,8), got {tuple(frames.shape)}"
            )
        batch = frames.shape[0]
        features = self.conv_stem(frames.reshape(batch * CAUSAL_FRAMES, 1, 8, 8))
        tokens = self.frame_projection(features.flatten(1)).reshape(
            batch, CAUSAL_FRAMES, SURFACE_READOUT_DIM
        )
        cls = self.cls_token.expand(batch, -1, -1)
        return self.transformer(self.position(torch.cat((cls, tokens), dim=1)))

    def readout_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """CLS hidden states the policy should consume. ``(B, 1, 128)``."""
        return self.encode_sequence(frames)[:, : self.n_readout]

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_sequence(frames)
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
    """Conv-transformer with a CLS readout plus 32-d embedding auxiliary heads.

    Pretrain uses the 32-d embedding (XYZ / validity / reconstruction). The
    ACT policy can consume either that frozen embedding or the 128-d CLS
    readout (same token, no extra projection) and fine-tune the stem.
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
        self.n_readout = 1
        self.readout_dim = SURFACE_READOUT_DIM

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Frame tokens + CLS readout. ``(B, 1+32, 128)``. Gradients flow."""
        if frames.ndim != 4 or frames.shape[1:] != (CAUSAL_FRAMES, 8, 8):
            raise ValueError(
                f"expected (B,{CAUSAL_FRAMES},8,8), got {tuple(frames.shape)}"
            )
        batch = frames.shape[0]
        features = self.conv_stem(
            frames.reshape(batch * CAUSAL_FRAMES, 1, 8, 8)
        )
        tokens = self.frame_projection(features.flatten(1)).reshape(
            batch, CAUSAL_FRAMES, SURFACE_READOUT_DIM
        )
        cls = self.cls_token.expand(batch, -1, -1)
        return self.transformer(self.position(torch.cat((cls, tokens), dim=1)))

    def readout_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """CLS hidden states the policy should consume. ``(B, 1, 128)``."""
        return self.encode_sequence(frames)[:, : self.n_readout]

    def forward(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encode_sequence(frames)
        batch = frames.shape[0]
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


def _validate_checkpoint_semantics(payload: dict[str, Any]) -> None:
    """Reject new checkpoints whose depth geometry differs from this runtime."""
    expected = {
        "max_surface_range_m": MAX_SURFACE_RANGE_M,
        "min_surface_range_m": MIN_SURFACE_RANGE_M,
        "sensor_fovy_deg": SENSOR_FOVY_DEG,
        "causal_frames": CAUSAL_FRAMES,
    }
    for key, value in expected.items():
        if key in payload and payload[key] != value:
            raise ValueError(
                f"surface encoder {key}={payload[key]!r}, runtime expects {value!r}"
            )
    if "sensor_order" in payload and list(payload["sensor_order"]) != list(
        HYBRID_SKIN_SENSOR_ORDER
    ):
        raise ValueError("surface encoder sensor_order differs from runtime")


def _apply_freeze(model: nn.Module, frozen: bool) -> None:
    if frozen:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return
    for parameter in model.parameters():
        parameter.requires_grad_(True)


def load_surface_encoder(
    checkpoint_path: str | Path,
    *,
    kind: str,
    map_location: str | torch.device = "cpu",
    frozen: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a ``pact_surface_*_v1`` file. ``frozen=False`` keeps grads for ACT."""
    if kind not in ("xyz", "embedding"):
        raise ValueError(f"kind must be 'xyz' or 'embedding', got {kind!r}")
    payload = torch.load(checkpoint_path, map_location=map_location)
    expected = SCHEMA_SURFACE_XYZ if kind == "xyz" else SCHEMA_SURFACE_EMBEDDING
    if payload.get("schema_version") != expected:
        raise ValueError(
            f"not a {expected} checkpoint (got {payload.get('schema_version')!r})"
        )
    if kind == "embedding" and payload.get("policy_feature_dim") not in (
        None,
        SURFACE_EMBEDDING_DIM,
        SURFACE_READOUT_DIM,
    ):
        raise ValueError("front-end policy feature dimension changed")
    _validate_checkpoint_semantics(payload)
    model: nn.Module = (
        SurfaceProximityEncoder() if kind == "xyz" else SurfaceEmbeddingEncoder()
    )
    model.load_state_dict(payload["model_state_dict"])
    _apply_freeze(model, frozen)
    return model, payload


def load_frozen_surface_encoder(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SurfaceProximityEncoder, dict[str, Any]]:
    model, payload = load_surface_encoder(
        checkpoint_path, kind="xyz", map_location=map_location, frozen=True
    )
    return model, payload  # type: ignore[return-value]


def load_frozen_surface_embedding_encoder(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SurfaceEmbeddingEncoder, dict[str, Any]]:
    model, payload = load_surface_encoder(
        checkpoint_path, kind="embedding", map_location=map_location, frozen=True
    )
    return model, payload  # type: ignore[return-value]


def load_frozen_proximity_encoder(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load either frozen front-end without changing the legacy v1 loader."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    schema = payload.get("schema_version")
    if schema == SCHEMA_SURFACE_XYZ:
        return load_frozen_surface_encoder(
            checkpoint_path, map_location=map_location
        )
    if schema == SCHEMA_SURFACE_EMBEDDING:
        return load_frozen_surface_embedding_encoder(
            checkpoint_path, map_location=map_location
        )
    raise ValueError(f"unsupported frozen proximity encoder schema: {schema}")


def pack_frozen_payload(
    model: nn.Module,
    kind: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the dict ``load_frozen_*`` expects. Marks the net frozen for ACT."""
    if kind == "xyz":
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_SURFACE_XYZ,
            "frozen": True,
            "model_state_dict": model.state_dict(),
            "max_surface_range_m": MAX_SURFACE_RANGE_M,
            "min_surface_range_m": MIN_SURFACE_RANGE_M,
            "sensor_fovy_deg": SENSOR_FOVY_DEG,
            "sensor_order": list(HYBRID_SKIN_SENSOR_ORDER),
            "causal_frames": CAUSAL_FRAMES,
            "readout_dim": SURFACE_READOUT_DIM,
            "preprocessing_schema": "pact_surface_closeness_v1",
        }
    elif kind == "embedding":
        payload = {
            "schema_version": SCHEMA_SURFACE_EMBEDDING,
            "frozen": True,
            "policy_feature_dim": SURFACE_EMBEDDING_DIM,
            "readout_dim": SURFACE_READOUT_DIM,
            "model_state_dict": model.state_dict(),
            "max_surface_range_m": MAX_SURFACE_RANGE_M,
            "min_surface_range_m": MIN_SURFACE_RANGE_M,
            "sensor_fovy_deg": SENSOR_FOVY_DEG,
            "sensor_order": list(HYBRID_SKIN_SENSOR_ORDER),
            "causal_frames": CAUSAL_FRAMES,
            "preprocessing_schema": "pact_surface_closeness_v1",
        }
    else:
        raise ValueError(f"kind must be 'xyz' or 'embedding', got {kind!r}")
    if extra:
        payload.update(extra)
    return payload


def save_frozen_checkpoint(
    path: str | Path,
    model: nn.Module,
    kind: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a frozen front-end ``.pt`` that ``load_encoder`` / probe can load."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack_frozen_payload(model, kind, extra), dest)
    return dest


def save_encoder_checkpoint(
    path: str | Path,
    model: nn.Module,
    kind: str,
    extra: dict[str, Any] | None = None,
    *,
    frozen: bool = True,
) -> Path:
    """Write a surface encoder ``.pt``. ``frozen=False`` marks an ACT-finetuned net."""
    extra = dict(extra or {})
    extra["frozen"] = bool(frozen)
    if not frozen and extra.get("policy_tap") == "readout":
        extra["policy_feature_dim"] = SURFACE_READOUT_DIM
    return save_frozen_checkpoint(path, model, kind, extra)


def _default_policy_tap(kind: str, policy_tap: str | None) -> str:
    if policy_tap is None:
        return "xyz" if kind == "xyz" else "embedding"
    if policy_tap not in POLICY_TAPS:
        raise ValueError(
            f"policy_tap must be one of {sorted(POLICY_TAPS)}, got {policy_tap!r}"
        )
    if policy_tap == "embedding" and kind != "embedding":
        raise ValueError("policy_tap='embedding' needs kind='embedding'")
    if policy_tap == "xyz" and kind != "xyz":
        raise ValueError("policy_tap='xyz' needs kind='xyz'")
    return policy_tap


class SurfaceGeometryEncoder(nn.Module):
    """Full-skin wrapper around the shared conv-transformer.

    ``kind='xyz'`` + tap ``xyz`` -> ``(B, S, 3)`` local metres (zeros when invalid).
    ``kind='embedding'`` + tap ``embedding`` -> ``(B, S, 32)`` frozen compressor.
    Either kind + tap ``readout`` -> ``(B, S, 128)`` CLS hidden (ACT finetune).

    Without ``checkpoint`` the net is randomly initialized (shape tests / wiring).
    Pass a ``pact_surface_*_v1`` checkpoint for real features. ``frozen=False``
    leaves grads on so ACT can finetune the stem; train and eval then both run
    the live readout, not baked HDF5 tokens.
    """

    def __init__(
        self,
        kind: str = "xyz",
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        *,
        inner: nn.Module | None = None,
        frozen: bool = True,
        policy_tap: str | None = None,
    ) -> None:
        super().__init__()
        if kind not in ("xyz", "embedding"):
            raise ValueError(f"kind must be 'xyz' or 'embedding', got {kind!r}")
        self.kind = kind
        self.name = "nearest_surface" if kind == "xyz" else "surface_embedding"
        self.frozen = bool(frozen)
        self.policy_tap = _default_policy_tap(kind, policy_tap)
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
            self.inner, self.payload = load_surface_encoder(
                checkpoint,
                kind=kind,
                map_location=device,
                frozen=self.frozen,
            )
        else:
            self.inner = (
                SurfaceProximityEncoder()
                if kind == "xyz"
                else SurfaceEmbeddingEncoder()
            )

        if self.policy_tap == "readout":
            self.act_feat_dim = SURFACE_READOUT_DIM
        elif self.kind == "xyz":
            self.act_feat_dim = 3
        else:
            self.act_feat_dim = SURFACE_EMBEDDING_DIM
        self.feat_dim = self.act_feat_dim
        self.validity_threshold = float(
            self.payload.get("validity_threshold", 0.5) if self.payload else 0.5
        )
        _apply_freeze(self.inner, self.frozen)
        self.to(device)
        if self.frozen:
            self.eval()
        else:
            self.train()

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

    def _encode_policy(
        self,
        skin: torch.Tensor | np.ndarray,
        unit: str,
    ) -> torch.Tensor:
        windows, squeeze_batch, squeeze_sensor = self._windows_on_device(skin, unit)
        batch, n_sensors = windows.shape[:2]
        flat = windows.reshape(batch * n_sensors, CAUSAL_FRAMES, 8, 8)
        if self.policy_tap == "readout":
            feat = self.inner.readout_tokens(flat).reshape(batch * n_sensors, -1)
        elif self.kind == "xyz":
            feat, _valid, _probabilities = self.inner.predict(
                flat, validity_threshold=self.validity_threshold
            )
        else:
            feat = self.inner.policy_features(flat)
        return self._reshape_out(
            feat, batch, n_sensors, squeeze_batch, squeeze_sensor
        )

    def policy_features(
        self,
        skin: torch.Tensor | np.ndarray,
        *,
        unit: str = "metres",
    ) -> torch.Tensor:
        """Encode a skin tensor. See ``to_causal_closeness`` for accepted shapes."""
        if self.frozen:
            with torch.no_grad():
                return self._encode_policy(skin, unit)
        return self._encode_policy(skin, unit)

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
        validity_threshold: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Inner ``predict`` over a full skin; keys depend on ``kind``."""
        if validity_threshold is None:
            validity_threshold = self.validity_threshold
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
            pad = x[:, :1].expand(
                batch, CAUSAL_CONTROL_STEPS - n_hist, n_sensors, height, width
            )
            x = torch.cat((pad, x), dim=1)
        elif n_hist > CAUSAL_CONTROL_STEPS:
            x = x[:, -CAUSAL_CONTROL_STEPS:]
        # (B, 8, S, 8, 8) -> (B, S, 32, 8, 8)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.unsqueeze(3).repeat(1, 1, 1, SUBFRAMES_PER_CONTROL_STEP, 1, 1)
        x = x.reshape(batch, n_sensors, CAUSAL_FRAMES, height, width)
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
        validity_threshold: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Same as ``encode_episode`` plus XYZ / validity for the token writer."""
        values = as_subframe_episode(episode_proximity)
        times = np.arange(values.shape[0], dtype=np.int64)
        return self.encode_episode_at_times(
            values,
            times,
            batch_size=batch_size,
            validity_threshold=validity_threshold,
        )

    @torch.no_grad()
    def encode_episode_at_times(
        self,
        episode_proximity: np.ndarray,
        times: np.ndarray,
        *,
        batch_size: int = 512,
        validity_threshold: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode selected control steps. Causal windows still use the full episode."""
        values = as_subframe_episode(episode_proximity)
        if validity_threshold is None:
            validity_threshold = self.validity_threshold
        times = np.asarray(times, dtype=np.int64).reshape(-1)
        n_steps = int(times.shape[0])
        n_sensors = int(values.shape[1])
        windows = np.empty(
            (n_steps, n_sensors, CAUSAL_FRAMES, 8, 8), dtype=np.float32
        )
        for step_index, timestep in enumerate(times):
            for sensor_index in range(n_sensors):
                windows[step_index, sensor_index] = causal_sensor_window(
                    values, int(timestep), sensor_index
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
