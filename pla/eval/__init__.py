"""Evaluation layer.

Top-level exports:
    bootstrap_ci, paired_bootstrap_p
    evaluate_checkpoint, print_results_table
    compute_sensor_importance
    REGISTRY, TaskSpec, get
    FailureType, EpisodeOutcome, categorize
"""
from pla.eval.bootstrap import bootstrap_ci, paired_bootstrap_p
from pla.eval.failure_analysis import (
    EpisodeOutcome,
    FailureType,
    categorize,
)
from pla.eval.run_eval import evaluate_checkpoint, print_results_table
from pla.eval.sensor_importance import compute_sensor_importance
from pla.eval.tasks import REGISTRY, TaskSpec, get

__all__ = [
    "bootstrap_ci",
    "paired_bootstrap_p",
    "evaluate_checkpoint",
    "print_results_table",
    "compute_sensor_importance",
    "REGISTRY",
    "TaskSpec",
    "get",
    "FailureType",
    "EpisodeOutcome",
    "categorize",
]
