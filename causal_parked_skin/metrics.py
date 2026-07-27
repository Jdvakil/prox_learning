"""Frozen metric definitions for the parked-skin reference.

Every formula here is fixed before offline-test results are opened. The epsilon values,
the oracle-active definition and the direction-cosine inclusion rule are constants in this
module for exactly that reason: if they were arguments, a disappointing test number could
be improved by changing one.
"""
from __future__ import annotations

import numpy as np

# frozen constants -- see module docstring
NORM_EPSILON = 1e-9
# a frame counts as oracle-active iff the stored boolean says so; the norm threshold that
# produced that boolean lives in the dataset, not here, so it cannot drift
DIRECTION_COSINE_MIN_NORM = 1e-4   # below this the true vector has no direction to match
CONSTRAINT_TOLERANCE = 1e-7


def safe_norm(vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vectors, axis=-1)


def direction_cosine(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Cosine between predicted and true 7-D differentials."""
    dot = np.sum(predicted * true, axis=-1)
    denominator = safe_norm(predicted) * safe_norm(true) + NORM_EPSILON
    return dot / denominator


def direction_cosine_included(true: np.ndarray) -> np.ndarray:
    """Frames whose true differential is long enough to have a meaningful direction."""
    return safe_norm(true) >= DIRECTION_COSINE_MIN_NORM


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, computed exactly by rank.

    Written out rather than imported so the definition is pinned with the report; the
    step-wise sum matches sklearn's ``average_precision_score``.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    true_positives = np.cumsum(hits)
    precision = true_positives / np.arange(1, hits.size + 1)
    return float((precision * hits).sum() / positives)


def binary_scores(predicted: np.ndarray, true: np.ndarray) -> dict:
    predicted = np.asarray(predicted).ravel().astype(bool)
    true = np.asarray(true).ravel().astype(bool)
    tp = float(np.sum(predicted & true))
    fp = float(np.sum(predicted & ~true))
    fn = float(np.sum(~predicted & true))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if tp + fp and tp + fn and (precision + recall) > 0 else float("nan"))
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn}


def constraint_violations(parked: np.ndarray, current: np.ndarray) -> dict:
    """0 <= parked <= current <= 1, counted rather than clamped."""
    above = int(np.sum(parked > current + CONSTRAINT_TOLERANCE))
    below = int(np.sum(parked < -CONSTRAINT_TOLERANCE))
    over_one = int(np.sum(parked > 1.0 + CONSTRAINT_TOLERANCE))
    return {"parked_above_current": above, "parked_below_zero": below,
            "parked_above_one": over_one, "total": above + below + over_one,
            "tolerance": CONSTRAINT_TOLERANCE}


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values ** 2)))


def stratified(values: np.ndarray, groups: np.ndarray, names) -> dict:
    """Mean of ``values`` within each group, plus the group's size."""
    out = {}
    for index, name in enumerate(names):
        hits = groups == index
        count = int(hits.sum())
        out[name] = {"count": count,
                     "value": float(np.mean(values[hits])) if count else None}
    return out


def summarize_seeds(values) -> dict:
    """Mean, standard deviation and coefficient of variation across seeds."""
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                       dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "std": None, "coefficient_of_variation": None, "n": 0}
    mean = float(array.mean())
    # population std over the three seeds actually run, not an estimate of a wider pool
    std = float(array.std(ddof=0))
    return {"mean": mean, "std": std,
            "coefficient_of_variation": abs(std / mean) if mean else None,
            "n": int(array.size), "values": [float(v) for v in array]}
