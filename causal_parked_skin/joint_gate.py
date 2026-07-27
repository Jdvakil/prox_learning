"""FULL_DATA_SEED_AGREEMENT_V1: the two-gate execution contract.

Seed 0 is the only deployment predictor. Seeds 1 and 2 exist to produce binary changed-pixel
masks and nothing else; no field, head output, differential or action is ever read from them,
so there is no route by which one could supply direction, magnitude, or a correction of its
own.

Two implementation details are pinned here deliberately rather than reconstructed, because
two earlier modules in this project disagree about them:

* **The mask comparison is strict.** The identifiability audit that produced the validated
  17/17 result built its masks with ``probability > 0.5``. The later bootstrap module used
  ``>=``. This module reuses the audit's strict form, since the audit is the measurement the
  owner decision rests on.
* **The controlling agreement is anchor-based.** The audit's ``changed_pixel_mask_agreement``
  averaged all three pairs, including (seed1, seed2). The handoff specifies the anchor form,
  ``mean(J(0,1), J(0,2))``, which asks a different and more relevant question: whether the
  *deployed* prediction is supported, rather than whether the two diagnostics agree with each
  other. The three-pair value is computed and logged as a secondary metric but can never
  alter execution.
"""
from __future__ import annotations

import numpy as np

MODE = "FULL_DATA_SEED_AGREEMENT_V1"
PIXEL_MASK_THRESHOLD = 0.5
EMPTY_MASK_AGREEMENT = 1.0
DEPLOYMENT_SEED = 0
UNCERTAINTY_SEEDS = (1, 2)

# a bootstrap-ensemble manifest must never be accepted in this mode
REJECTED_ENSEMBLE_IDS = ("PARKED_SKIN_TRAJECTORY_BOOTSTRAP_ENSEMBLE_V1",)
BOOTSTRAP_DISPOSITION = (
    "invalid_for_deployment_uncertainty_due_to_cluster_omission_variance")


class JointGateManifestError(RuntimeError):
    """The manifest is not a valid FULL_DATA_SEED_AGREEMENT_V1 contract."""


def changed_mask(probability: np.ndarray) -> np.ndarray:
    """(N, 40, 8, 8) change probability -> flat boolean mask.

    Strictly greater than 0.5, matching the identifiability audit exactly.
    """
    array = np.asarray(probability)
    return array.reshape(array.shape[0], -1) > PIXEL_MASK_THRESHOLD


def jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Jaccard; two empty masks agree completely."""
    intersection = (a & b).sum(axis=-1).astype(np.float64)
    union = (a | b).sum(axis=-1).astype(np.float64)
    return np.where(union > 0, intersection / np.maximum(union, 1.0),
                    EMPTY_MASK_AGREEMENT)


def anchor_mask_agreement(mask0: np.ndarray, mask1: np.ndarray,
                          mask2: np.ndarray) -> np.ndarray:
    """mean(J(seed0, seed1), J(seed0, seed2)) -- the controlling metric."""
    return 0.5 * (jaccard(mask0, mask1) + jaccard(mask0, mask2))


def three_pair_agreement(mask0: np.ndarray, mask1: np.ndarray,
                         mask2: np.ndarray) -> np.ndarray:
    """The identifiability audit's form, logged for comparison only."""
    return (jaccard(mask0, mask1) + jaccard(mask0, mask2)
            + jaccard(mask1, mask2)) / 3.0


def joint_decision(activity: np.ndarray, activity_threshold: float,
                   agreement: np.ndarray, agreement_threshold: float) -> dict:
    """The two-gate rule. Both must pass for the seed-0 correction to execute."""
    activity_pass = np.asarray(activity) >= activity_threshold
    agreement_pass = np.asarray(agreement) >= agreement_threshold
    return {
        "activity_pass": activity_pass,
        "agreement_pass": agreement_pass,
        "execute": activity_pass & agreement_pass,
        "abstained_by_uncertainty": activity_pass & ~agreement_pass,
    }


def apply_gate(seed0_differential: np.ndarray, execute: np.ndarray) -> np.ndarray:
    """Either exactly the seed-0 differential, or exactly zeros. Nothing between."""
    out = np.zeros_like(seed0_differential)
    out[execute] = seed0_differential[execute]
    return out


def assert_not_bootstrap(manifest: dict) -> None:
    """Refuse a bootstrap-ensemble manifest in FULL_DATA_SEED_AGREEMENT_V1 mode."""
    identifier = manifest.get("ensemble_id") or manifest.get("mode")
    if identifier in REJECTED_ENSEMBLE_IDS:
        raise JointGateManifestError(
            f"{identifier} is {BOOTSTRAP_DISPOSITION}; it may not supply uncertainty "
            f"in mode {MODE}")
    for record in manifest.get("member_records", []):
        if "bootstrap_seed" in record:
            raise JointGateManifestError(
                "a bootstrap member reached a FULL_DATA_SEED_AGREEMENT_V1 manifest")


def validate_seed_roster(seeds) -> None:
    """Exactly seeds 0, 1, 2 in that order; seed 0 first because it is the anchor."""
    roster = list(seeds)
    if roster != [DEPLOYMENT_SEED, *UNCERTAINTY_SEEDS]:
        raise JointGateManifestError(f"seed roster {roster} is not [0, 1, 2]")
