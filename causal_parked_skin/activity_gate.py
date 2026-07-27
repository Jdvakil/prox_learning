"""PROX_EVIDENCE_ACTIVITY_GATE_V1 — an activity gate that can only see proximity.

The frozen model has no isolated activity path: proprioceptive state is summed into every
sensor token before the cross-sensor transformer, and the activity value is a max-reduction
of the same mask-logit channel that builds the parked field. Whatever the activity decision
depends on, it is structurally free to depend on posture.

This gate removes that freedom by construction rather than by regularisation. Its forward
signature accepts only the current closeness field, its validity mask, and learned sensor
identity. There is no argument through which qpos, qvel, actions, episode counters, phase,
hazard labels or manifest identity could arrive, so the prohibition is enforced by the type
of the function rather than by a runtime check that someone could forget to call.

The architecture is fixed by the handoff and must not change after training starts.
"""
from __future__ import annotations

import torch
from torch import nn

GATE_ID = "PROX_EVIDENCE_ACTIVITY_GATE_V1"

SENSORS = 40
PIXEL_ROWS = 8
PIXEL_COLS = 8
PIXELS = PIXEL_ROWS * PIXEL_COLS
PER_SENSOR_INPUT = 2 * PIXELS      # closeness and validity mask
EMBED_DIM = 64
BLOCKS = 2
HEADS = 4
FEED_FORWARD = 128
POOLED_DIM = 2 * EMBED_DIM
PARAMETER_BUDGET = 250_000

# the only fields the gate may ever receive
PERMITTED_INPUTS = ("current_closeness", "current_valid_mask", "sensor_identity")
PROHIBITED_INPUTS = (
    "qpos", "qvel", "nominal_action", "gripper_state", "gripper_command",
    "predicted_parked_field", "predicted_differential", "episode_step",
    "control_timestamp", "trajectory_length", "task_phase", "hazard_present",
    "manifest_row_id", "episode_id", "oracle_dq", "oracle_active",
    "changed_pixel_mask", "parked_closeness",
)


class ProxEvidenceActivityGate(nn.Module):
    """Per-sensor MLP, cross-sensor transformer, mean+max pooling, activity head."""

    def __init__(self) -> None:
        super().__init__()
        self.per_sensor = nn.Sequential(
            nn.Linear(PER_SENSOR_INPUT, EMBED_DIM), nn.SiLU(),
            nn.Linear(EMBED_DIM, EMBED_DIM), nn.SiLU())
        self.sensor_embedding = nn.Parameter(torch.zeros(SENSORS, EMBED_DIM))
        nn.init.normal_(self.sensor_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=HEADS, dim_feedforward=FEED_FORWARD,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.cross_sensor = nn.TransformerEncoder(layer, num_layers=BLOCKS)
        self.activity_head = nn.Sequential(
            nn.Linear(POOLED_DIM, EMBED_DIM), nn.SiLU(),
            nn.Linear(EMBED_DIM, 1))
        # start biased towards "quiet": most frames are, and an unbiased start spends the
        # first epochs unlearning a uniform 0.5 activity
        with torch.no_grad():
            self.activity_head[-1].bias.fill_(-2.0)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, closeness: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """closeness (B, 40, 8, 8), valid_mask (B, 40, 8, 8) -> (B,) activity logit.

        Deliberately takes no other argument.
        """
        if closeness.shape[1:] != (SENSORS, PIXEL_ROWS, PIXEL_COLS):
            raise ValueError(f"closeness has shape {tuple(closeness.shape)}")
        if valid_mask.shape != closeness.shape:
            raise ValueError("valid mask must match the closeness field")
        batch = closeness.shape[0]
        features = torch.cat([closeness.reshape(batch, SENSORS, PIXELS),
                              valid_mask.to(closeness.dtype).reshape(
                                  batch, SENSORS, PIXELS)], dim=-1)
        tokens = self.per_sensor(features) + self.sensor_embedding
        tokens = self.cross_sensor(tokens)
        pooled = torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1)
        return self.activity_head(pooled).squeeze(-1)

    @torch.no_grad()
    def probability(self, closeness: torch.Tensor,
                    valid_mask: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(closeness, valid_mask))


def build_gate() -> ProxEvidenceActivityGate:
    gate = ProxEvidenceActivityGate()
    count = gate.parameter_count()
    if count > PARAMETER_BUDGET:
        raise ValueError(f"gate has {count} parameters, budget is {PARAMETER_BUDGET}")
    return gate


def gate_feature_hash(closeness, valid_mask) -> str:
    """Hash of exactly what the gate consumed, kept separate from the field model's."""
    import hashlib

    import numpy as np

    digest = hashlib.sha256()
    for array in (np.ascontiguousarray(np.asarray(closeness, dtype=np.float32)),
                  np.ascontiguousarray(np.asarray(valid_mask, dtype=np.bool_))):
        digest.update(str(array.dtype.str).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def assert_gate_inputs(payload) -> None:
    """Runtime guard for the deployment path (handoff step 16)."""
    if not isinstance(payload, dict):
        raise TypeError("gate input payload must be a mapping")
    offending = sorted(set(payload) & set(PROHIBITED_INPUTS))
    if offending:
        raise ValueError(f"prohibited fields reached the activity gate: {offending}")
    extra = sorted(set(payload) - set(PERMITTED_INPUTS))
    if extra:
        raise ValueError(f"unexpected fields reached the activity gate: {extra}")
    missing = sorted({"current_closeness", "current_valid_mask"} - set(payload))
    if missing:
        raise ValueError(f"activity gate is missing required inputs: {missing}")
