#!/usr/bin/env python3
"""V10.5 runtime: one registered speed cap, gated on the marker AND the schedule.

The single permitted route change is the V10.4 initial free-space cap of
0.12 m/s. Everything else about the V9.3 expert — waypoints, IK, bows, later
segment speeds — is inherited untouched, and the pendant never enters a planner
obstacle list or a surface-distance speed law.

The cap is gated twice: on the exact V10.5 environment marker, and on the hash
of the baseline speed schedule. If the inherited V9.3 plan ever changes shape,
the schedule hash stops matching and the amendment refuses rather than silently
capping a different segment.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np

INITIAL_FREE_SPACE_SPEED_CAP_M_S = 0.12
TASK_HORIZON_V105 = 1050
PENDANT_GEOM_PREFIX = "pact_clutter_mount_v105_"


class SpeedAmendmentError(RuntimeError):
    """The registered speed cap could not be bound to exactly one segment."""


def locate_initial_free_space_segment(primitives: Sequence[Any]) -> dict[str, Any]:
    """Bind the capped segment by primitive order, not by a phase string.

    The target is the first segment of the first TCP move sequence that
    precedes the first gripper-close action. Anything else is refused rather
    than guessed.
    """
    from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (  # noqa: E501
        GripperAction,
        TCPMoveSequence,
    )

    close_index = None
    for index, primitive in enumerate(primitives):
        if isinstance(primitive, GripperAction) and not bool(primitive.target_open):
            close_index = index
            break
    if close_index is None:
        raise SpeedAmendmentError("no gripper-close action in the primitive list")
    sequence_index = None
    for index in range(close_index):
        if isinstance(primitives[index], TCPMoveSequence):
            sequence_index = index
            break
    if sequence_index is None:
        raise SpeedAmendmentError("no TCP move sequence before the gripper close")
    segments = getattr(primitives[sequence_index], "_move_segments", None) or []
    if not segments:
        raise SpeedAmendmentError("the first TCP move sequence has no segments")
    return {
        "primitive_index": int(sequence_index),
        "gripper_close_primitive_index": int(close_index),
        "segment_index": 0,
        "segment": segments[0],
        "n_segments": int(len(segments)),
    }


def speed_schedule(primitives: Sequence[Any]) -> list[dict[str, Any]]:
    """Every commanded segment speed, in plan order."""
    from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (  # noqa: E501
        TCPMoveSequence,
    )

    out: list[dict[str, Any]] = []
    for primitive_index, primitive in enumerate(primitives):
        if not isinstance(primitive, TCPMoveSequence):
            continue
        for segment_index, segment in enumerate(
            getattr(primitive, "_move_segments", None) or []
        ):
            out.append(
                {
                    "primitive_index": int(primitive_index),
                    "segment_index": int(segment_index),
                    "name": str(segment.name),
                    "speed_m_s": round(float(segment.speed), 9),
                }
            )
    return out


def schedule_sha256(primitives: Sequence[Any]) -> str:
    return hashlib.sha256(
        json.dumps(speed_schedule(primitives), sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()


def plan_signature(primitives: Sequence[Any]) -> dict[str, Any]:
    """Poses and speeds, so a caller can prove exactly one value changed."""
    from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (  # noqa: E501
        TCPMoveSequence,
    )

    entries: list[dict[str, Any]] = []
    for primitive_index, primitive in enumerate(primitives):
        if not isinstance(primitive, TCPMoveSequence):
            entries.append(
                {"primitive_index": int(primitive_index),
                 "type": type(primitive).__name__}
            )
            continue
        for segment_index, segment in enumerate(
            getattr(primitive, "_move_segments", None) or []
        ):
            entries.append(
                {
                    "primitive_index": int(primitive_index),
                    "segment_index": int(segment_index),
                    "type": "TCPMoveSegment",
                    "name": str(segment.name),
                    "speed_m_s": round(float(segment.speed), 9),
                    "start_position_m": [
                        round(float(v), 9) for v in segment.start_pose[:3, 3]
                    ],
                    "end_position_m": [
                        round(float(v), 9) for v in segment.end_pose[:3, 3]
                    ],
                    "start_rotation": [
                        round(float(v), 9)
                        for v in np.asarray(segment.start_pose[:3, :3]).reshape(9)
                    ],
                    "end_rotation": [
                        round(float(v), 9)
                        for v in np.asarray(segment.end_pose[:3, :3]).reshape(9)
                    ],
                }
            )
    return {"entries": entries, "n": len(entries)}


def verify_plan_matches_baseline(baseline, amended) -> dict[str, Any]:
    """Exactly one speed may differ; every pose must be identical."""
    base_entries = baseline["entries"]
    amended_entries = amended["entries"]
    if len(base_entries) != len(amended_entries):
        return {"identical_shape": False, "n_speed_changes": None,
                "poses_identical": False, "passed": False,
                "failures": ["primitive/segment count changed"]}
    failures: list[str] = []
    changes: list[dict[str, Any]] = []
    for before, after in zip(base_entries, amended_entries):
        for key in ("primitive_index", "segment_index", "type", "name",
                    "start_position_m", "end_position_m",
                    "start_rotation", "end_rotation"):
            if before.get(key) != after.get(key):
                failures.append(f"{key} changed at {before.get('primitive_index')}")
        if before.get("speed_m_s") != after.get("speed_m_s"):
            changes.append(
                {
                    "primitive_index": before.get("primitive_index"),
                    "segment_index": before.get("segment_index"),
                    "segment_name": before.get("name"),
                    "before_m_s": before.get("speed_m_s"),
                    "after_m_s": after.get("speed_m_s"),
                }
            )
    return {
        "identical_shape": True,
        "poses_identical": not failures,
        "n_speed_changes": len(changes),
        "speed_changes": changes,
        "failures": failures,
        "passed": not failures and len(changes) <= 1,
    }


def apply_initial_free_space_speed_cap(primitives: Sequence[Any]) -> dict[str, Any]:
    """Cap exactly one segment. Refuses if more than one value would change."""
    baseline_schedule = schedule_sha256(primitives)
    located = locate_initial_free_space_segment(primitives)
    segment = located["segment"]
    before = round(float(segment.speed), 9)
    after = round(min(before, INITIAL_FREE_SPACE_SPEED_CAP_M_S), 9)
    changed = after != before
    if changed:
        segment.speed = float(after)
    schedule = speed_schedule(primitives)
    n_changed = sum(
        1
        for item in schedule
        if item["primitive_index"] == located["primitive_index"]
        and item["segment_index"] == located["segment_index"]
        and item["speed_m_s"] != before
    )
    if changed and n_changed != 1:
        raise SpeedAmendmentError(
            f"the cap must alter exactly one segment, altered {n_changed}"
        )
    return {
        "applied": bool(changed),
        "cap_m_s": INITIAL_FREE_SPACE_SPEED_CAP_M_S,
        "primitive_index": located["primitive_index"],
        "segment_index": located["segment_index"],
        "segment_name": str(segment.name),
        "gripper_close_primitive_index": located["gripper_close_primitive_index"],
        "speed_before_m_s": before,
        "speed_after_m_s": after,
        "baseline_schedule_sha256": baseline_schedule,
        "amended_schedule_sha256": schedule_sha256(primitives),
        "n_segments_altered": 1 if changed else 0,
    }


__all__ = [
    "INITIAL_FREE_SPACE_SPEED_CAP_M_S",
    "PENDANT_GEOM_PREFIX",
    "SpeedAmendmentError",
    "TASK_HORIZON_V105",
    "apply_initial_free_space_speed_cap",
    "locate_initial_free_space_segment",
    "plan_signature",
    "schedule_sha256",
    "speed_schedule",
    "verify_plan_matches_baseline",
]
