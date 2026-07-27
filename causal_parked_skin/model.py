"""CausalParkedSkinReferenceV1 and the three comparison baselines.

Every model here shares one output parameterization:

    changed_probability = sigmoid(mask_logits)
    delta_magnitude     = current_closeness * sigmoid(magnitude_logits)
    predicted_delta     = changed_probability * delta_magnitude
    predicted_parked    = current_closeness - predicted_delta

which gives ``0 <= predicted_parked <= current_closeness <= 1`` by construction. Both
factors lie in [0, 1] and the magnitude is scaled by the current field itself, so the
subtraction can neither overshoot zero nor exceed the current value. Nothing is clipped
afterwards: a clamp would hide a violation instead of making it impossible, and the
readiness gate that counts violations would then be measuring the clamp.

The factorization is deliberately two-headed. A single sigmoid conflates "this pixel
changes" with "it changes by this much", and the changed-pixel prevalence here is about
4e-5, so the magnitude head would spend almost all its gradient on pixels that never move.

Baselines differ only in what drives the logits, never in the constraint, the targets or
the evaluation path.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

SENSORS = 40
PIXEL_ROWS = 8
PIXEL_COLS = 8
PIXELS = PIXEL_ROWS * PIXEL_COLS
STATE_WIDTH = 29
CAUSAL_FRAMES = 4
PARAMETER_BUDGET = 3_000_000

MODEL_ID = "CausalParkedSkinReferenceV1"

BASELINE_ZERO = "ZERO_DIFFERENTIAL"
BASELINE_QPOS = "QPOS_ONLY"
BASELINE_CURRENT = "CURRENT_FRAME_ONLY"
BASELINE_FULL = "FULL_CAUSAL"


class FrozenSafetyHead(nn.Module):
    """The canonical SafetyHead, frozen, with a differentiable closeness entry point.

    The shipped ``SafetyHead.__call__`` takes numpy depths and ``SafetyCVAE.act`` is
    decorated ``@torch.no_grad()``, so neither can carry a gradient back into the parked
    field. The decoder is therefore called directly, at ``z = 0`` exactly as ``act`` does.

    The dataset stores closeness, and ``featurize`` maps depth to closeness with the same
    transform, so no conversion happens here -- converting closeness to depth and back
    would apply the transform twice. ``label_scale`` is applied once, at the end.
    """

    def __init__(self, decoder: nn.Module, z_dim: int, label_scale: float) -> None:
        super().__init__()
        self.decoder = decoder
        self.z_dim = int(z_dim)
        self.register_buffer("label_scale",
                             torch.tensor(float(label_scale), dtype=torch.float32))
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.decoder.eval()

    @classmethod
    def load(cls, checkpoint_dir: str | Path, device: str = "cpu") -> FrozenSafetyHead:
        import sys

        root = Path(__file__).resolve().parents[1]
        if str(root / "scripts") not in sys.path:
            sys.path.insert(0, str(root / "scripts"))
        from train_safety_cvae import SafetyCVAE

        checkpoint_dir = Path(checkpoint_dir)
        meta = json.loads((checkpoint_dir / "meta.json").read_text())
        model = SafetyCVAE(meta["n_in"], meta["n_out"], meta["z_dim"])
        model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device))
        head = cls(model.dec, meta["z_dim"], meta["label_scale"]).to(device)
        head.eval()
        return head

    def train(self, mode: bool = True) -> FrozenSafetyHead:
        # stays in eval no matter what the surrounding training loop does
        return super().train(False)

    def forward(self, closeness: torch.Tensor) -> torch.Tensor:
        """(B, 40, 8, 8) closeness -> (B, 7) differential in physical units."""
        flat = closeness.reshape(closeness.shape[0], -1)
        latent = torch.zeros(flat.shape[0], self.z_dim,
                             device=flat.device, dtype=flat.dtype)
        return self.decoder(torch.cat([flat, latent], dim=-1)) * self.label_scale

    def frozen(self) -> bool:
        return not any(p.requires_grad for p in self.decoder.parameters())


def constrained_parked(current: torch.Tensor, mask_logits: torch.Tensor,
                       magnitude_logits: torch.Tensor):
    """The shared output constraint. Returns (parked, delta, changed_probability)."""
    changed_probability = torch.sigmoid(mask_logits)
    delta_magnitude = current * torch.sigmoid(magnitude_logits)
    delta = changed_probability * delta_magnitude
    return current - delta, delta, changed_probability


class _SensorDecoder(nn.Module):
    """Shared per-sensor spatial decoder producing the two logit maps."""

    def __init__(self, feature_channels: int, token_width: int, hidden: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(feature_channels + token_width, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, 2, 3, padding=1)
        self.act = nn.SiLU()
        # start near "nothing changes": the prior is that a frame is quiet, and 76% of
        # frames are. Without this the model spends its first epochs unlearning a
        # uniform 0.25 change probability over every pixel.
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor([-4.0, 0.0]))

    def forward(self, features: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        spread = token[..., None, None].expand(-1, -1, features.shape[-2],
                                               features.shape[-1])
        hidden = self.act(self.project(torch.cat([features, spread], dim=1)))
        return self.out(hidden)


class CausalParkedSkinReferenceV1(nn.Module):
    """Per-sensor spatiotemporal encoder, cross-sensor attention, per-sensor decoder.

    ``history_frames`` selects the baseline: 4 is FULL_CAUSAL, 1 is CURRENT_FRAME_ONLY.
    ``use_proximity=False`` is QPOS_ONLY -- the logits then come from robot state alone,
    though the output constraint still references the current field, because that
    constraint is what makes every model's output physically admissible. The ablation
    isolates what drives the logits, which is the thing under test.
    """

    def __init__(self, *, hidden: int = 192, blocks: int = 2, heads: int = 4,
                 history_frames: int = CAUSAL_FRAMES, use_proximity: bool = True,
                 link_ids: torch.Tensor | None = None, link_count: int = 7,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if history_frames < 1 or history_frames > CAUSAL_FRAMES:
            raise ValueError("history_frames must be within the stored causal window")
        self.hidden = int(hidden)
        self.blocks = int(blocks)
        self.history_frames = int(history_frames)
        self.use_proximity = bool(use_proximity)

        if link_ids is None:
            link_ids = torch.zeros(SENSORS, dtype=torch.long)
        self.register_buffer("link_ids", link_ids.to(torch.long))

        # closeness and validity for each retained frame -> explicit channels
        channels = 2 * self.history_frames
        if self.use_proximity:
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, 32, 3, padding=1), nn.SiLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.SiLU())
            # mean and max pooling: mean carries how much is near, max carries the peak,
            # and a purely mean-pooled token cannot tell one hot pixel from a warm patch
            self.to_token = nn.Linear(128, self.hidden)
            feature_channels = 64
        else:
            self.encoder = None
            self.to_token = None
            # A learned per-pixel prior, NOT the current field. Feeding the current field
            # to the decoder here would give the state-only baseline exactly the
            # proximity information the ablation is supposed to withhold, and it would
            # then be a slightly worse CURRENT_FRAME_ONLY rather than a control. The
            # output constraint still references the current field -- every model shares
            # that -- but nothing that drives the logits does.
            feature_channels = 8
            self.pixel_prior = nn.Parameter(
                torch.zeros(1, feature_channels, PIXEL_ROWS, PIXEL_COLS))
            nn.init.normal_(self.pixel_prior, std=0.02)

        self.sensor_embedding = nn.Parameter(torch.zeros(SENSORS, self.hidden))
        nn.init.normal_(self.sensor_embedding, std=0.02)
        self.link_embedding = nn.Parameter(torch.zeros(link_count, self.hidden))
        nn.init.normal_(self.link_embedding, std=0.02)

        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_WIDTH, self.hidden), nn.SiLU(),
            nn.Linear(self.hidden, self.hidden))

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden, nhead=heads, dim_feedforward=2 * self.hidden,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.cross_sensor = nn.TransformerEncoder(layer, num_layers=self.blocks)

        self.decoder = _SensorDecoder(feature_channels, self.hidden, hidden=64)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, history: torch.Tensor, history_valid: torch.Tensor,
                state: torch.Tensor):
        """history (B, F, 40, 8, 8) closeness; history_valid same shape; state (B, 29).

        The last history frame is the current one, by the dataset's causal rule.
        """
        batch = history.shape[0]
        current = history[:, -1]
        keep = self.history_frames

        if self.use_proximity:
            closeness = history[:, -keep:]
            validity = history_valid[:, -keep:].to(history.dtype)
            # (B*40, 2F, 8, 8): every sensor is encoded by the same weights
            stacked = torch.cat([closeness, validity], dim=1)
            stacked = stacked.permute(0, 2, 1, 3, 4).reshape(
                batch * SENSORS, 2 * keep, PIXEL_ROWS, PIXEL_COLS)
            features = self.encoder(stacked)
            pooled = torch.cat([features.mean(dim=(2, 3)),
                                features.amax(dim=(2, 3))], dim=-1)
            tokens = self.to_token(pooled).reshape(batch, SENSORS, self.hidden)
            decoder_features = features
        else:
            tokens = torch.zeros(batch, SENSORS, self.hidden,
                                 device=history.device, dtype=history.dtype)
            decoder_features = self.pixel_prior.expand(
                batch * SENSORS, -1, -1, -1).to(history.dtype)

        tokens = (tokens + self.sensor_embedding
                  + self.link_embedding[self.link_ids]
                  + self.state_encoder(state).unsqueeze(1))
        tokens = self.cross_sensor(tokens)

        logits = self.decoder(decoder_features,
                              tokens.reshape(batch * SENSORS, self.hidden))
        logits = logits.reshape(batch, SENSORS, 2, PIXEL_ROWS, PIXEL_COLS)
        parked, delta, changed_probability = constrained_parked(
            current, logits[:, :, 0], logits[:, :, 1])
        return {"parked": parked, "delta": delta,
                "changed_probability": changed_probability,
                "mask_logits": logits[:, :, 0], "current": current}


def build_model(variant: str, *, hidden: int = 192, blocks: int = 2,
                link_ids: torch.Tensor | None = None, link_count: int = 7,
                dropout: float = 0.0) -> CausalParkedSkinReferenceV1:
    """Construct one of the learned variants. ZERO_DIFFERENTIAL has no parameters."""
    if variant == BASELINE_FULL:
        return CausalParkedSkinReferenceV1(
            hidden=hidden, blocks=blocks, history_frames=CAUSAL_FRAMES,
            use_proximity=True, link_ids=link_ids, link_count=link_count, dropout=dropout)
    if variant == BASELINE_CURRENT:
        return CausalParkedSkinReferenceV1(
            hidden=hidden, blocks=blocks, history_frames=1, use_proximity=True,
            link_ids=link_ids, link_count=link_count, dropout=dropout)
    if variant == BASELINE_QPOS:
        return CausalParkedSkinReferenceV1(
            hidden=hidden, blocks=blocks, history_frames=1, use_proximity=False,
            link_ids=link_ids, link_count=link_count, dropout=dropout)
    raise ValueError(f"{variant!r} is not a learned variant")


def zero_differential(history: torch.Tensor) -> dict:
    """ZERO_DIFFERENTIAL: predict the current field unchanged, i.e. no obstacle to remove."""
    current = history[:, -1]
    zeros = torch.zeros_like(current)
    return {"parked": current, "delta": zeros, "changed_probability": zeros,
            "mask_logits": torch.full_like(current, -30.0), "current": current}
