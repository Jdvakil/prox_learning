"""Trajectory-aware activation-threshold calibration for the frozen parked reference.

The previous threshold was the frame-level 99th percentile of oracle-zero scores over eight
calibration trajectories. Treating 3,821 frames as 3,821 independent observations overstates
the precision of that quantile badly: proximity fields at 15 Hz are strongly autocorrelated,
and whole trajectories share a scene, a hazard pose and a policy. The effective sample size
is closer to the number of trajectories than to the number of frames, which is why a
threshold that produced ~1% exceedance by construction on calibration produced 2.15% on a
different partition.

Everything here therefore works at trajectory level, and the uncertainty comes from a
cluster bootstrap that resamples whole episodes.

Two definitions are fixed here, in source, before any candidate is scored:

* **Activity probability.** The frozen model has no frame-level activity head -- it emits a
  per-pixel changed-probability map. The frame-level activity probability is the maximum of
  that map: the model's confidence that *at least one* pixel is removable. This is a
  parameter-free reduction of an output the frozen checkpoint already produces, so it
  involves no training and no new weights. Gating on it rather than on the predicted
  differential norm is required: the norm conflates "something is there" with "the
  correction is large".
* **Cluster.** One episode. The same episode identity appears in three source
  distributions, so those three trajectories are not independent of each other; resampling
  files rather than episodes would reintroduce the very independence assumption this
  recalibration exists to remove.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

# fixed before any candidate is scored
BOOTSTRAP_SEED = 20260727
BOOTSTRAP_REPLICATES = 10_000
CONFIDENCE = 0.95

# feasibility contract, quoted from the handoff and not modified afterwards
MIN_MEDIAN_ACTIVE_RECALL = 0.80
MIN_MEDIAN_ACTIVE_COSINE = 0.75
MIN_MEDIAN_POSITIVE_COSINE_FRACTION = 0.85
MAX_BOOTSTRAP_UPPER_FPR = 0.02
MAX_HAZARD_ABSENT_TRAJECTORY_FPR = 0.05
MAX_CONSECUTIVE_FALSE_POSITIVE_RUN = 2
ALLOW_PERSISTENT_ACTIVATION_AFTER_ORACLE = False

# blocking checks for the consumed diagnostic set (handoff step 9)
DIAGNOSTIC_MAX_MEAN_ZERO_FPR = 0.03
DIAGNOSTIC_MAX_HAZARD_ABSENT_ACTIVE_FRACTION = 0.10
DIAGNOSTIC_MIN_MEDIAN_RECALL = 0.70
DIAGNOSTIC_MIN_MEDIAN_COSINE = 0.70
DIAGNOSTIC_MAX_PERSISTENT_RUN = 5

COSINE_EPSILON = 1e-9


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def activity_probability(changed_probability: np.ndarray) -> np.ndarray:
    """(N, 40, 8, 8) per-pixel changed probability -> (N,) frame activity probability."""
    array = np.asarray(changed_probability)
    return array.reshape(array.shape[0], -1).max(axis=1)


def max_true_run(flags: np.ndarray) -> int:
    """Longest run of consecutive True values."""
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return 0
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int((edges[1::2] - edges[0::2]).max())


def persists_after_oracle(fired: np.ndarray, active: np.ndarray) -> bool:
    """True when the gate is still firing after the oracle has finished for good."""
    active = np.asarray(active, dtype=bool)
    fired = np.asarray(fired, dtype=bool)
    if not active.any():
        return bool(fired.any())
    return bool(fired[np.flatnonzero(active)[-1] + 1:].any())


@dataclass
class TrajectoryScores:
    """Per-frame quantities for one trajectory, in temporal order."""

    trajectory_id: str
    episode_id: str
    distribution: str
    partition: str
    hazard_present: bool
    activity: np.ndarray
    oracle_active: np.ndarray
    cosine: np.ndarray
    predicted_norm: np.ndarray
    oracle_norm: np.ndarray
    changed_true: np.ndarray          # per-frame count of truly changed pixels
    changed_predicted: np.ndarray     # per-frame count of predicted changed pixels
    changed_hit: np.ndarray           # per-frame count of correct changed pixels

    @property
    def frames(self) -> int:
        return len(self.activity)

    def metrics_at(self, threshold: float) -> dict:
        """Every trajectory-level metric the contract needs, at one threshold."""
        fired = self.activity >= threshold
        active = self.oracle_active.astype(bool)
        zero = ~active
        retained = fired & active
        false_positive = fired & zero

        recall = float(retained.sum() / active.sum()) if active.any() else None
        fpr = float(false_positive.sum() / zero.sum()) if zero.any() else None
        cosines = self.cosine[retained]
        predicted_changed = self.changed_predicted[retained].sum()
        return {
            "trajectory_id": self.trajectory_id,
            "episode_id": self.episode_id,
            "distribution": self.distribution,
            "partition": self.partition,
            "hazard_present": bool(self.hazard_present),
            "frames": self.frames,
            "oracle_active_frames": int(active.sum()),
            "oracle_zero_frames": int(zero.sum()),
            "active_recall": recall,
            "oracle_zero_false_positive_rate": fpr,
            "median_active_cosine": float(np.median(cosines)) if cosines.size else None,
            "positive_cosine_fraction": (float((cosines > 0).mean())
                                         if cosines.size else None),
            "changed_pixel_precision": (
                float(self.changed_hit[retained].sum() / predicted_changed)
                if predicted_changed > 0 else None),
            "max_consecutive_false_positive_run": max_true_run(false_positive),
            "persists_after_oracle": persists_after_oracle(fired, active),
            "false_positive_frames": int(false_positive.sum()),
            "retained_active_frames": int(retained.sum()),
        }


def trajectory_fpr_matrix(trajectories: list[TrajectoryScores],
                          thresholds: np.ndarray) -> np.ndarray:
    """(n_trajectories, n_thresholds) oracle-zero false-positive rate.

    Computed by rank rather than by looping over thresholds: for a sorted array of
    oracle-zero activity values, the count above a threshold is a searchsorted lookup.
    """
    out = np.zeros((len(trajectories), thresholds.size), dtype=np.float64)
    for row, trajectory in enumerate(trajectories):
        zero_values = np.sort(trajectory.activity[~trajectory.oracle_active.astype(bool)])
        if zero_values.size == 0:
            out[row] = np.nan
            continue
        above = zero_values.size - np.searchsorted(zero_values, thresholds, side="left")
        out[row] = above / zero_values.size
    return out


def trajectory_recall_matrix(trajectories: list[TrajectoryScores],
                             thresholds: np.ndarray) -> np.ndarray:
    out = np.zeros((len(trajectories), thresholds.size), dtype=np.float64)
    for row, trajectory in enumerate(trajectories):
        active_values = np.sort(
            trajectory.activity[trajectory.oracle_active.astype(bool)])
        if active_values.size == 0:
            out[row] = np.nan
            continue
        above = active_values.size - np.searchsorted(active_values, thresholds,
                                                     side="left")
        out[row] = above / active_values.size
    return out


def cluster_bootstrap_upper_bound(cluster_values: np.ndarray, *,
                                  replicates: int = BOOTSTRAP_REPLICATES,
                                  seed: int = BOOTSTRAP_SEED,
                                  confidence: float = CONFIDENCE,
                                  chunk: int = 256) -> np.ndarray:
    """One-sided upper confidence bound on the mean, resampling whole clusters.

    ``cluster_values`` is (n_clusters, n_thresholds): each row is one episode's mean
    trajectory-level FPR. Resampling rows keeps every trajectory of an episode together,
    so the bound reflects between-episode variation rather than between-frame variation.

    The same bootstrap index matrix is used for every threshold, which makes the resulting
    bounds directly comparable and the whole computation reproducible from the seed.
    """
    clusters, n_thresholds = cluster_values.shape
    rng = np.random.default_rng(seed)
    index = rng.integers(0, clusters, size=(replicates, clusters))
    bounds = np.empty(n_thresholds, dtype=np.float64)
    for start in range(0, n_thresholds, chunk):
        stop = min(start + chunk, n_thresholds)
        block = cluster_values[:, start:stop]                  # (clusters, c)
        resampled = block[index]                               # (replicates, clusters, c)
        means = np.nanmean(resampled, axis=1)                  # (replicates, c)
        bounds[start:stop] = np.quantile(means, confidence, axis=0)
    return bounds


def cluster_means(values: np.ndarray, cluster_of_row: np.ndarray,
                  n_clusters: int) -> np.ndarray:
    """Average per-trajectory values within each episode, giving (n_clusters, n_thr)."""
    out = np.full((n_clusters, values.shape[1]), np.nan, dtype=np.float64)
    for cluster in range(n_clusters):
        rows = np.flatnonzero(cluster_of_row == cluster)
        if rows.size:
            out[cluster] = np.nanmean(values[rows], axis=0)
    return out


def select_threshold(feasible: np.ndarray, thresholds: np.ndarray,
                     upper_bound: np.ndarray, median_recall: np.ndarray) -> int | None:
    """Lexicographic choice: lowest bound, then highest recall, then highest threshold."""
    candidates = np.flatnonzero(feasible)
    if candidates.size == 0:
        return None
    order = sorted(candidates.tolist(),
                   key=lambda i: (upper_bound[i], -median_recall[i], -thresholds[i]))
    return int(order[0])
