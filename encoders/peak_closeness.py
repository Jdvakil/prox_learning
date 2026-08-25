"""Peak-closeness skin encoder (headline PACT-raw) plus optional CVAE taps.

Function: map a 40-sensor depth snapshot to per-sensor closeness, or to a
frozen Safety-CVAE retreat feature.

Headline path is ``feature='raw'``: per-sensor peak closeness in ``[0, 1]``.
That math does **not** load Safety-CVAE weights. ``trunk`` / ``delta`` still
exist as negative-control taps and require a ``model.pt`` that is no longer
in this repo.

Sensor order lives in ``hybrid_skin_sensors.HYBRID_SKIN_SENSOR_ORDER``
(``link5_back`` before ``link5_front``). Convert and live eval must use that list.

PACT still imports this module as ``submodules/act/prox_cvae.py`` (shim).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ._paths import ensure_act_on_path

ensure_act_on_path()
from hybrid_skin_sensors import DEAD_PIXEL_M, D_MAX, HYBRID_SKIN_SENSOR_ORDER  # noqa: E402

DEFAULT_CKPT = ""  # empty: raw does not need a CVAE dir


class SafetyCVAE(nn.Module):
    """Vendored copy of scripts/train_safety_cvae.py:SafetyCVAE (state_dict-compatible).

    Only used if someone still has a ``model.pt`` and asks for trunk/delta.
    """

    def __init__(self, n_in: int, n_out: int = 7, z_dim: int = 8) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(
            nn.Linear(n_in + n_out, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 2 * z_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(n_in + z_dim, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, n_out),
        )

    def act(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(x.shape[0], self.z_dim, device=x.device, dtype=x.dtype)
        return self.dec(torch.cat([x, z], -1))

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(x.shape[0], self.z_dim, device=x.device, dtype=x.dtype)
        h = torch.cat([x, z], -1)
        h = self.dec[1](self.dec[0](h))
        h = self.dec[3](self.dec[2](h))
        return h


def featurize_np(prox: np.ndarray) -> np.ndarray:
    """(N, S, 8, 8) depths in meters -> (N, S*64) closeness in [0, 1]."""
    d = prox.astype(np.float32)
    c = np.clip(1.0 - d / D_MAX, 0.0, 1.0)
    c[d < DEAD_PIXEL_M] = 0.0
    return c.reshape(len(c), -1)


def featurize_torch(prox: torch.Tensor) -> torch.Tensor:
    """prox: (B, S, 8, 8) meters -> (B, S*64) closeness."""
    d = prox.float()
    c = torch.clamp(1.0 - d / D_MAX, 0.0, 1.0)
    c = torch.where(d < DEAD_PIXEL_M, torch.zeros_like(c), c)
    return c.reshape(c.shape[0], -1)


def feat_dim_for(ckpt_dir: str | Path | None = None, feature: str = "raw") -> int:
    if feature == "trunk":
        return 256
    if feature == "raw":
        return len(HYBRID_SKIN_SENSOR_ORDER)
    if feature == "delta":
        if ckpt_dir:
            meta = json.loads((Path(ckpt_dir) / "meta.json").read_text())
            return int(meta["n_out"])
        return 7
    raise ValueError(f"unknown prox feature {feature!r} (expected 'trunk', 'delta' or 'raw')")


def resolve_prox_layout(
    feature: str,
    layout: str,
    n_sensors: int,
    feat_dim: int,
    tokens_per_sensor: int,
) -> tuple[int, int, int, str]:
    if layout not in ("global", "per_sensor"):
        raise ValueError(f"unknown prox layout {layout!r} (expected 'global' or 'per_sensor')")
    if layout == "per_sensor":
        if feature != "raw":
            raise ValueError("prox_layout='per_sensor' requires prox_feature='raw'")
        k = 1 if int(tokens_per_sensor) == 8 else int(tokens_per_sensor)
        return n_sensors, 1, k, layout
    return 1, int(feat_dim), int(tokens_per_sensor), layout


def _pool_substeps(arr: np.ndarray, pool: str) -> np.ndarray:
    if pool == "min":
        return arr.min(axis=0)
    if pool == "mean":
        return arr.mean(axis=0)
    raise ValueError(f"unknown prox pool {pool!r} (expected 'mean' or 'min')")


class PeakClosenessEncoder(nn.Module):
    """Skin snapshot -> ACT proximity tokens.

    ``feature='raw'`` (headline): no ``model.pt``. Peak closeness per sensor.
    ``trunk`` / ``delta``: load a Safety-CVAE if ``ckpt_dir/model.pt`` exists.
    """

    def __init__(
        self,
        ckpt_dir: str | Path | None = None,
        feature: str = "raw",
        device: str = "cuda",
        layout: str = "global",
        tokens_per_sensor: int = 8,
    ) -> None:
        super().__init__()
        if feature not in ("trunk", "delta", "raw"):
            raise ValueError(f"unknown prox feature {feature!r} (expected 'trunk'/'delta'/'raw')")
        self.name = {
            "raw": "peak_closeness",
            "trunk": "cvae_trunk",
            "delta": "cvae_delta",
        }[feature]
        self.feature = feature
        self.sensor_order: list[str] = list(HYBRID_SKIN_SENSOR_ORDER)
        self.n_sensors = len(self.sensor_order)
        self.n_out = 7
        self.label_scale = 1.0
        self.model: SafetyCVAE | None = None

        ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        if feature in ("trunk", "delta"):
            if ckpt_dir is None or not (ckpt_dir / "model.pt").is_file():
                raise SystemExit(
                    "[prox] feature={feature!r} needs a Safety-CVAE model.pt. Those "
                    "weights were removed; train with --prox_feature raw.".format(feature=feature)
                )
            meta = json.loads((ckpt_dir / "meta.json").read_text())
            self.label_scale = float(meta["label_scale"])
            self.sensor_order = list(meta["sensors"])
            self.n_sensors = len(self.sensor_order)
            self.n_out = int(meta["n_out"])
            model = SafetyCVAE(int(meta["n_in"]), int(meta["n_out"]), int(meta["z_dim"]))
            state = torch.load(ckpt_dir / "model.pt", map_location=device)
            model.load_state_dict(state)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.model = model

        self.feat_dim = {"trunk": 256, "delta": self.n_out, "raw": self.n_sensors}[feature]
        n_act, act_dim, k, layout = resolve_prox_layout(
            feature, layout, self.n_sensors, self.feat_dim, tokens_per_sensor
        )
        self.layout = layout
        self.n_act_sensors = n_act
        self.act_feat_dim = act_dim
        self.tokens_per_sensor = k
        self.device = device
        self.to(device)
        src = f"CVAE {ckpt_dir}" if self.model is not None else "raw closeness (no CVAE)"
        print(
            f"[prox] {src} | feature={feature} layout={layout} "
            f"n_act_sensors={n_act} act_feat_dim={act_dim} K={k} | {self.n_sensors} sensors"
        )

    @torch.no_grad()
    def forward(self, prox: torch.Tensor) -> torch.Tensor:
        """(B, 40, 8, 8) raw depths (m) -> (B, n_act_sensors, act_feat_dim)."""
        if prox.dim() != 4 or prox.shape[1] != self.n_sensors:
            raise ValueError(
                f"prox must be (B, {self.n_sensors}, 8, 8); got {tuple(prox.shape)}"
            )
        prox = prox.to(self.device)
        x = featurize_torch(prox)
        if self.feature == "trunk":
            feat = self.model.trunk(x)
        elif self.feature == "raw":
            feat = x.reshape(x.shape[0], self.n_sensors, -1).amax(dim=2)
        else:
            feat = self.model.act(x) * self.label_scale
        if self.layout == "per_sensor":
            return feat.unsqueeze(-1)
        return feat.unsqueeze(1)

    @torch.no_grad()
    def policy_features(self, skin: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Same as ``forward``, also accepts unbatched ``(40, 8, 8)`` or numpy."""
        if not torch.is_tensor(skin):
            skin = torch.as_tensor(skin, dtype=torch.float32)
        squeeze_batch = False
        if skin.ndim == 3:
            skin = skin.unsqueeze(0)
            squeeze_batch = True
        out = self.forward(skin)
        if squeeze_batch:
            return out[0]
        return out


ProxCVAEEncoder = PeakClosenessEncoder


def stack_obs_proximity(
    obs: dict, sensor_order: list[str], pool: str = "mean"
) -> np.ndarray:
    """Build a (40, 8, 8) float32 depth array from a live env observation dict."""
    frames = []
    for name in sensor_order:
        if name not in obs:
            avail = [k for k in obs if "sensor" in str(k)]
            raise KeyError(
                f"proximity sensor {name!r} not in observation. "
                f"{len(avail)} sensor-like keys present, e.g. {avail[:6]}. "
                f"Is the hybrid-skin ProximityDepthBufferSensor enabled?"
            )
        arr = np.asarray(obs[name], dtype=np.float32)
        if arr.ndim == 3:
            arr = _pool_substeps(arr, pool)
        elif arr.ndim != 2:
            raise ValueError(f"sensor {name!r} has unexpected shape {arr.shape}")
        frames.append(arr)
    return np.stack(frames, axis=0)
