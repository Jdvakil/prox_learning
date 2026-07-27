"""Loss components for the parked-skin reference.

The supervision is extremely imbalanced in two independent ways: about 24% of frames carry
any oracle differential at all, and within those frames roughly 4e-5 of pixels actually
change. An unweighted pixel loss is therefore minimised almost perfectly by predicting
"nothing ever changes", which is exactly the ZERO_DIFFERENTIAL baseline the model has to
beat. Each component below exists to stop one version of that collapse.

Weights are selected on validation only. No component is tuned against offline test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    """Relative weights of the six components."""

    changed_mask: float = 1.0
    active_delta: float = 10.0
    all_valid_field: float = 1.0
    head_consistency: float = 1.0
    quiet: float = 1.0
    temporal_smoothness: float = 0.0

    focal_gamma: float = 2.0
    huber_beta: float = 0.01
    # Cap on the BCE positive weight. Uncapped it is negatives/positives ~= 1200 at this
    # prevalence, which balances the mask objective but makes the mask head fire on
    # essentially every pixel; the delta is then subtracted field-wide and the frozen
    # head turns a ~4e-3 per-pixel error into a ~0.7 differential error. The cap keeps
    # the positives visible without letting them dictate the operating point.
    mask_pos_weight_cap: float = 32.0

    def as_dict(self) -> dict:
        return {"changed_mask": self.changed_mask, "active_delta": self.active_delta,
                "all_valid_field": self.all_valid_field,
                "head_consistency": self.head_consistency, "quiet": self.quiet,
                "temporal_smoothness": self.temporal_smoothness,
                "focal_gamma": self.focal_gamma, "huber_beta": self.huber_beta,
                "mask_pos_weight_cap": self.mask_pos_weight_cap}


def focal_bce(logits: torch.Tensor, target: torch.Tensor, *, gamma: float,
              valid: torch.Tensor | None = None,
              pos_weight_cap: float = 32.0) -> torch.Tensor:
    """Focal BCE with per-batch positive weighting.

    ``pos_weight`` is computed from the batch rather than fixed, because the stratified
    sampler changes the positive rate between configurations; a constant weight tuned for
    one active fraction silently becomes wrong for another.
    """
    target = target.to(logits.dtype)
    if valid is None:
        valid = torch.ones_like(target)
    else:
        valid = valid.to(logits.dtype)
    positives = (target * valid).sum()
    negatives = valid.sum() - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0),
                             max=float(pos_weight_cap))

    raw = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none",
        pos_weight=pos_weight.to(logits.dtype))
    probability = torch.sigmoid(logits)
    p_t = probability * target + (1.0 - probability) * (1.0 - target)
    loss = raw * (1.0 - p_t).clamp(min=1e-6) ** gamma
    return (loss * valid).sum() / valid.sum().clamp(min=1.0)


def masked_huber(prediction: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor, *, beta: float) -> torch.Tensor:
    mask = mask.to(prediction.dtype)
    total = mask.sum()
    if float(total) == 0.0:
        return prediction.sum() * 0.0
    raw = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    return (raw * mask).sum() / total


def masked_l1(prediction: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(prediction.dtype)
    total = mask.sum()
    if float(total) == 0.0:
        return prediction.sum() * 0.0
    return ((prediction - target).abs() * mask).sum() / total


@dataclass
class LossTerms:
    total: torch.Tensor
    parts: dict = field(default_factory=dict)


def compute_losses(output: dict, batch: dict, head, weights: LossWeights) -> LossTerms:
    """All six components plus the frozen-head consistency path.

    ``head`` is the frozen SafetyHead. Gradients flow through it into the predicted parked
    field; its own parameters are frozen and must stay that way.
    """
    parked_pred = output["parked"]
    current = output["current"]
    parked_true = batch["parked"]
    changed = batch["changed"]
    valid = batch["parked_valid"] & batch["current_valid"]
    oracle_dq = batch["oracle_dq"]
    active = batch["oracle_active"]

    # 1. changed-pixel mask, focal + class balanced, scored only on valid pixels
    mask_loss = focal_bce(output["mask_logits"], changed,
                          gamma=weights.focal_gamma, valid=valid,
                          pos_weight_cap=weights.mask_pos_weight_cap)

    # 2. the differential itself, on pixels that genuinely move
    delta_true = current - parked_true
    active_delta_loss = masked_huber(output["delta"], delta_true,
                                     changed & valid, beta=weights.huber_beta)

    # 3. the whole field, so the model cannot buy mask accuracy with field drift
    field_loss = masked_l1(parked_pred, parked_true, valid)

    # 4. frozen-head consistency: the quantity the controller actually consumes
    current_dq = head(current)
    predicted_parked_dq = head(parked_pred)
    predicted_oracle_dq = current_dq - predicted_parked_dq
    head_loss = F.smooth_l1_loss(predicted_oracle_dq, oracle_dq, beta=weights.huber_beta)

    # 5. quiet when the oracle is quiet -- the false-positive rate gate lives or dies here
    quiet_mask = (~active).to(parked_pred.dtype).unsqueeze(-1)
    quiet_total = quiet_mask.sum()
    quiet_loss = ((predicted_oracle_dq ** 2) * quiet_mask).sum() / \
        torch.clamp(quiet_total * predicted_oracle_dq.shape[-1], min=1.0)

    total = (weights.changed_mask * mask_loss
             + weights.active_delta * active_delta_loss
             + weights.all_valid_field * field_loss
             + weights.head_consistency * head_loss
             + weights.quiet * quiet_loss)

    parts = {"changed_mask": float(mask_loss.detach()),
             "active_delta": float(active_delta_loss.detach()),
             "all_valid_field": float(field_loss.detach()),
             "head_consistency": float(head_loss.detach()),
             "quiet": float(quiet_loss.detach())}

    if weights.temporal_smoothness > 0.0 and "previous_parked" in batch:
        # causal only: penalises change against the *previous* prediction, never a future
        # one, and is disabled on frames where the oracle itself jumps, so genuine
        # obstacle onset is not smoothed away
        onset = (batch["oracle_dq"] - batch["previous_oracle_dq"]).abs().amax(-1)
        gate = (onset < 1e-3).to(parked_pred.dtype).reshape(-1, 1, 1, 1)
        smooth = ((parked_pred - batch["previous_parked"]).abs() * gate).mean()
        total = total + weights.temporal_smoothness * smooth
        parts["temporal_smoothness"] = float(smooth.detach())

    parts["total"] = float(total.detach())
    return LossTerms(total=total, parts=parts)
