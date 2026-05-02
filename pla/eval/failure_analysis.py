"""Failure case analysis.

Categorize per-episode failures and (optionally) dump qualitative videos
with proximity sensor readings overlaid (PROJECT.md §4.4).

Categories:
    approach_collision  — arm hits obstacle (proximity should help)
    grasp_miss          — gripper misses target (dense EE sensors may help)
    place_failure       — object dropped (proximity unlikely to help)
    language_failure    — wrong object selected (VLM error)
    success             — task completed

The categorizer reads MolmoSpaces' per-episode info dict. The keys it looks
for are listed below; if your env emits different keys, override
``RULES`` or pass a custom ``rule_table`` to ``categorize``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping


class FailureType(str, Enum):
    APPROACH_COLLISION = "approach_collision"
    GRASP_MISS = "grasp_miss"
    PLACE_FAILURE = "place_failure"
    LANGUAGE_FAILURE = "language_failure"
    SUCCESS = "success"


@dataclass
class EpisodeOutcome:
    seed: int
    task: str
    success: bool
    failure_type: FailureType
    proximity_min_mm: float
    notes: str = ""


# Ordered rule list: first matching rule wins. Each rule is
# (info_key, expected_value or callable, FailureType).
RULES: list[tuple[str, object, FailureType]] = [
    ("success", True, FailureType.SUCCESS),
    ("collided_with_obstacle", True, FailureType.APPROACH_COLLISION),
    ("grasp_failed", True, FailureType.GRASP_MISS),
    ("object_dropped", True, FailureType.PLACE_FAILURE),
    ("wrong_object_picked", True, FailureType.LANGUAGE_FAILURE),
]


def categorize(
    info: Mapping,
    rule_table: list[tuple[str, object, FailureType]] | None = None,
) -> FailureType:
    rules = rule_table or RULES
    for key, expected, ftype in rules:
        v = info.get(key)
        if callable(expected):
            if expected(v):
                return ftype
        elif v == expected:
            return ftype
    # Fallback: if it isn't a labeled failure, lean toward grasp miss for PnP.
    return FailureType.GRASP_MISS


def summarize(outcomes: Iterable[EpisodeOutcome]) -> dict[str, int]:
    out: dict[str, int] = {ft.value: 0 for ft in FailureType}
    for o in outcomes:
        out[o.failure_type.value] += 1
    return out


def write_outcomes(outcomes: Iterable[EpisodeOutcome], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(o) for o in outcomes], indent=2, default=str))
