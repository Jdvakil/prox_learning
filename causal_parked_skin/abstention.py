"""The combined activity-plus-uncertainty deployment contract.

The controller executes the frozen seed-0 correction only when two independent gates agree:
the seed-0 activity probability clears the already-frozen activity threshold, *and* the
bootstrap ensemble agrees with seed 0 about which pixels change. Uncertainty can only ever
turn "execute" into "abstain"; it can never change the direction, the magnitude, or which
model supplies the correction.

``anchor_mask_agreement`` is anchored on seed 0 deliberately. Mean pairwise agreement among
the five members would measure how much the members agree with *each other*, which is a
different question -- an ensemble can be internally consistent while collectively disagreeing
with the model actually being deployed. What matters is whether the deployed prediction is
supported.
"""
from __future__ import annotations

import numpy as np

# fixed by the handoff; never tuned
PIXEL_MASK_THRESHOLD = 0.5
EMPTY_MASK_AGREEMENT = 1.0


def changed_mask(probability: np.ndarray) -> np.ndarray:
    """(N, 40, 8, 8) change probability -> flat boolean mask at the fixed threshold."""
    array = np.asarray(probability)
    return (array.reshape(array.shape[0], -1) >= PIXEL_MASK_THRESHOLD)


def jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Jaccard. Two empty masks agree completely rather than dividing by zero."""
    intersection = (a & b).sum(axis=-1).astype(np.float64)
    union = (a | b).sum(axis=-1).astype(np.float64)
    return np.where(union > 0, intersection / np.maximum(union, 1.0),
                    EMPTY_MASK_AGREEMENT)


def anchor_mask_agreement(anchor: np.ndarray, members: list[np.ndarray]) -> np.ndarray:
    """Mean Jaccard between the frozen seed-0 mask and each bootstrap member's mask."""
    if not members:
        raise ValueError("anchor agreement needs at least one member")
    return np.stack([jaccard(anchor, member) for member in members]).mean(axis=0)


def mean_pairwise_agreement(members: list[np.ndarray]) -> np.ndarray:
    """Diagnostic only: how much the members agree among themselves."""
    if len(members) < 2:
        return np.ones(len(members[0]))
    scores = [jaccard(members[i], members[j])
              for i in range(len(members)) for j in range(i + 1, len(members))]
    return np.stack(scores).mean(axis=0)


def combined_decision(seed0_activity: np.ndarray, activity_threshold: float,
                      agreement: np.ndarray, agreement_threshold: float) -> dict:
    """The deployment rule. Returns the component gates and the final decision."""
    activity_pass = seed0_activity >= activity_threshold
    agreement_pass = agreement >= agreement_threshold
    return {
        "activity_pass": activity_pass,
        "agreement_pass": agreement_pass,
        "execute": activity_pass & agreement_pass,
        "abstained": activity_pass & ~agreement_pass,
    }


def apply_abstention(seed0_differential: np.ndarray, execute: np.ndarray) -> np.ndarray:
    """Zero the correction where the contract abstains. Nothing else is touched.

    The executed vector is either exactly the seed-0 differential or exactly zeros; no
    scaling, blending or member substitution occurs anywhere on this path.
    """
    out = np.zeros_like(seed0_differential)
    out[execute] = seed0_differential[execute]
    return out
