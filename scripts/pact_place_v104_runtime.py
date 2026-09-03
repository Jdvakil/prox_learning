#!/usr/bin/env python3
"""V10.4 runtime helpers: the single speed amendment and per-frame telemetry.

Everything here is gated on the exact V10.4 environment marker. No V6c or
historical environment can reach any of it, and nothing here poses, resizes, or
re-bounds a geom at runtime.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

ENVIRONMENT_VERSION_V104 = "pact_place_corridor_v10_4_first_shot_static_pendant"
SAMPLER_CLASS_V104 = "PactPlaceCorridorV104Sampler"

# The one registered production speed change.
INITIAL_FREE_SPACE_SPEED_CAP_M_S = 0.12
TASK_HORIZON_V104 = 1050
HORIZON_UTILISATION_LIMIT = 0.80
CONTROL_PERIOD_S = 0.066

PENDANT_GEOM_PREFIX = "pact_clutter_mount_v104"
MOUNTED_FIXTURE_CLASS = "mounted_fixture"


class SpeedAmendmentError(RuntimeError):
    """The registered initial segment could not be bound unambiguously."""


def locate_initial_free_space_segment(primitives: Sequence[Any]) -> dict[str, Any]:
    """Bind the capped segment by primitive order, not by a phase string.

    The target is the first segment of the first TCP move sequence that precedes
    the first gripper-close action. Anything else is refused rather than guessed.
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


def segment_pose_signature(segment: Any) -> dict[str, Any]:
    return {
        "name": str(segment.name),
        "start_position_m": [float(v) for v in segment.start_pose[:3, 3]],
        "end_position_m": [float(v) for v in segment.end_pose[:3, 3]],
        "start_rotation": [float(v) for v in np.asarray(segment.start_pose[:3, :3]).reshape(9)],
        "end_rotation": [float(v) for v in np.asarray(segment.end_pose[:3, :3]).reshape(9)],
    }


def plan_signature(primitives: Sequence[Any]) -> list[dict[str, Any]]:
    """Order, names, poses, and speeds of every primitive and segment."""
    out: list[dict[str, Any]] = []
    for index, primitive in enumerate(primitives):
        segments = getattr(primitive, "_move_segments", None)
        record: dict[str, Any] = {
            "primitive_index": index,
            "primitive": type(primitive).__name__,
            "is_holding_object": bool(getattr(primitive, "is_holding_object", False)),
        }
        if segments:
            record["segments"] = [
                {
                    "segment_index": j,
                    "speed_m_s": float(seg.speed),
                    **segment_pose_signature(seg),
                }
                for j, seg in enumerate(segments)
            ]
        else:
            record["target_open"] = getattr(primitive, "target_open", None)
        out.append(record)
    return out


def apply_initial_free_space_speed_cap(
    primitives: Sequence[Any],
    *,
    cap_m_s: float = INITIAL_FREE_SPACE_SPEED_CAP_M_S,
) -> dict[str, Any]:
    """Cap exactly one segment. Returns the audit record.

    Refuses to act if the located segment is already at or below the cap in a
    way that would make the amendment a no-op silently, and verifies afterwards
    that exactly one segment speed changed.
    """
    before = plan_signature(primitives)
    located = locate_initial_free_space_segment(primitives)
    segment = located["segment"]
    original = float(segment.speed)
    signature = segment_pose_signature(segment)
    segment.speed = float(cap_m_s)
    after = plan_signature(primitives)

    changed: list[dict[str, Any]] = []
    for prim_before, prim_after in zip(before, after):
        for seg_before, seg_after in zip(
            prim_before.get("segments") or [], prim_after.get("segments") or []
        ):
            if abs(seg_before["speed_m_s"] - seg_after["speed_m_s"]) > 1e-12:
                changed.append(
                    {
                        "primitive_index": prim_before["primitive_index"],
                        "segment_index": seg_before["segment_index"],
                        "segment_name": seg_before["name"],
                        "from_m_s": seg_before["speed_m_s"],
                        "to_m_s": seg_after["speed_m_s"],
                    }
                )
    if len(changed) != 1:
        raise SpeedAmendmentError(
            f"the amendment changed {len(changed)} segments, expected exactly 1"
        )
    return {
        "applied": True,
        "cap_m_s": float(cap_m_s),
        "original_speed_m_s": original,
        "primitive_index": located["primitive_index"],
        "segment_index": located["segment_index"],
        "gripper_close_primitive_index": located["gripper_close_primitive_index"],
        "segment_signature": signature,
        "n_segments_changed": len(changed),
        "changed": changed,
    }


def verify_plan_matches_baseline(
    baseline: Sequence[dict[str, Any]],
    amended: Sequence[dict[str, Any]],
    *,
    position_atol_m: float = 1e-9,
    rotation_atol: float = 1e-9,
) -> dict[str, Any]:
    """Every pose and every other speed must match the V6c plan exactly."""
    failures: list[dict[str, Any]] = []
    if len(baseline) != len(amended):
        failures.append({"code": "primitive_count", "baseline": len(baseline), "amended": len(amended)})
    speed_changes: list[dict[str, Any]] = []
    for prim_b, prim_a in zip(baseline, amended):
        if prim_b["primitive"] != prim_a["primitive"]:
            failures.append({"code": "primitive_type", "index": prim_b["primitive_index"]})
            continue
        segs_b = prim_b.get("segments") or []
        segs_a = prim_a.get("segments") or []
        if len(segs_b) != len(segs_a):
            failures.append({"code": "segment_count", "index": prim_b["primitive_index"]})
            continue
        for seg_b, seg_a in zip(segs_b, segs_a):
            if seg_b["name"] != seg_a["name"]:
                failures.append({"code": "segment_name", "index": prim_b["primitive_index"]})
            for key, atol in (
                ("start_position_m", position_atol_m),
                ("end_position_m", position_atol_m),
                ("start_rotation", rotation_atol),
                ("end_rotation", rotation_atol),
            ):
                if not np.allclose(
                    np.asarray(seg_b[key], dtype=float),
                    np.asarray(seg_a[key], dtype=float),
                    atol=atol,
                    rtol=0.0,
                ):
                    failures.append(
                        {
                            "code": f"segment_{key}",
                            "primitive_index": prim_b["primitive_index"],
                            "segment_index": seg_b["segment_index"],
                        }
                    )
            if abs(seg_b["speed_m_s"] - seg_a["speed_m_s"]) > 1e-12:
                speed_changes.append(
                    {
                        "primitive_index": prim_b["primitive_index"],
                        "segment_index": seg_b["segment_index"],
                        "segment_name": seg_b["name"],
                        "from_m_s": seg_b["speed_m_s"],
                        "to_m_s": seg_a["speed_m_s"],
                    }
                )
    return {
        "poses_identical": not failures,
        "failures": failures,
        "speed_changes": speed_changes,
        "n_speed_changes": len(speed_changes),
        "exactly_one_speed_change": len(speed_changes) == 1,
    }


def pendant_contact_state(model, data, pendant_geom_ids: Sequence[int]) -> dict[str, Any]:
    """Live per-component contact from ``data.contact``."""
    ids = {int(v) for v in pendant_geom_ids}
    pairs: list[dict[str, Any]] = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if float(contact.dist) > 0.0:
            continue
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 not in ids and geom2 not in ids:
            continue
        pairs.append(
            {
                "geom1": model.geom(geom1).name or f"geom_{geom1}",
                "geom2": model.geom(geom2).name or f"geom_{geom2}",
                "distance_m": float(contact.dist),
            }
        )
    return {
        "n_pairs": len(pairs),
        "contact": bool(pairs),
        "pairs": pairs[:6],
    }
