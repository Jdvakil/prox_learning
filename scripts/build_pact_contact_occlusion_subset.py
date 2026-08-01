#!/usr/bin/env python3
"""Freeze the preregistered wrist-occlusion viability partition.

The partition is computed only from the committed expert trajectory's wrist
camera pose, the committed panel geometry, and static scene AABBs.  Scientific
policy outcomes are deliberately not loaded.  The pre-grasp window ends at the
first commanded gripper close, matching the environment-gate sighting window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.convert_obstacle_to_act import _decode_action


SCHEMA_VERSION = "pact_contact_occlusion_subset_v1"
ELIGIBLE_STATUSES = {"success", "task_failure"}
SOURCE_ROLES = ("full_train", "full_validation")
ACTUAL_IMAGE_WIDTH = 624
ACTUAL_IMAGE_HEIGHT = 352
WRIST_FOVY_DEGREES = 56.74
DISADVANTAGED_FRACTION_THRESHOLD = 0.50
MIN_PARTITION_FRACTION = 0.25
DEGENERATE_PARTITION_FRACTION = 0.95
PROBE_FACE_X_FRACTION = -1.0
PROBE_OFFSETS = (
    (0.0, 0.0),
    (+0.6, 0.0),
    (-0.6, 0.0),
    (0.0, +0.6),
    (0.0, -0.6),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_probe_points(center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Return the same five camera-facing panel probes as the sighting test."""
    points = []
    for y_fraction, z_fraction in PROBE_OFFSETS:
        points.append(
            center
            + np.array(
                [
                    PROBE_FACE_X_FRACTION * half[0],
                    y_fraction * half[1],
                    z_fraction * half[2],
                ],
                dtype=np.float64,
            )
        )
    return np.asarray(points, dtype=np.float64)


def camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    rotations = extrinsics[:, :, :3]
    translations = extrinsics[:, :, 3]
    return -np.einsum("tji,tj->ti", rotations, translations)


def probes_inside_frustum(extrinsics: np.ndarray, probes: np.ndarray) -> np.ndarray:
    """Return a (steps, probes) mask using the physical landscape frustum."""
    rotations = extrinsics[:, :, :3]
    translations = extrinsics[:, :, 3]
    camera = np.einsum("tij,pj->tpi", rotations, probes) + translations[:, None, :]
    z = camera[:, :, 2]
    tan_y = math.tan(math.radians(WRIST_FOVY_DEGREES) / 2.0)
    tan_x = tan_y * (ACTUAL_IMAGE_WIDTH / ACTUAL_IMAGE_HEIGHT)
    return (
        (z > 0.0)
        & (np.abs(camera[:, :, 0]) <= z * tan_x)
        & (np.abs(camera[:, :, 1]) <= z * tan_y)
    )


def segment_intersects_aabb_before_target(
    origin: np.ndarray,
    target: np.ndarray,
    center: np.ndarray,
    half: np.ndarray,
    *,
    epsilon: float = 1e-9,
) -> bool:
    """Whether an AABB blocks the open line segment from origin to target."""
    direction = target - origin
    lower = center - half
    upper = center + half
    enter, leave = 0.0, 1.0
    for axis in range(3):
        if abs(float(direction[axis])) <= epsilon:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return False
            continue
        first = (lower[axis] - origin[axis]) / direction[axis]
        second = (upper[axis] - origin[axis]) / direction[axis]
        if first > second:
            first, second = second, first
        enter = max(enter, float(first))
        leave = min(leave, float(second))
        if enter > leave:
            return False
    return leave > epsilon and enter < 1.0 - epsilon


def static_occlusion_mask(
    centers: np.ndarray,
    probes: np.ndarray,
    inside_frustum: np.ndarray,
    obstacle_aabbs: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    blocked = np.zeros_like(inside_frustum, dtype=bool)
    for step, probe_index in np.argwhere(inside_frustum):
        blocked[step, probe_index] = any(
            segment_intersects_aabb_before_target(
                centers[step], probes[probe_index], box_center, box_half
            )
            for box_center, box_half in obstacle_aabbs
        )
    return blocked


def first_close_step(actions: h5py.Dataset) -> int:
    for step, encoded in enumerate(actions):
        action, valid = _decode_action(encoded)
        if valid and float(action[7]) >= 127.5:
            return int(step)
    return int(len(actions))


def partition_action(occluded: int, non_occluded: int) -> dict[str, Any]:
    total = occluded + non_occluded
    if total <= 0:
        raise ValueError("occlusion partition has no eligible episodes")
    fractions = {
        "vision_disadvantaged": occluded / total,
        "non_vision_disadvantaged": non_occluded / total,
    }
    largest = max(fractions.values())
    if largest > DEGENERATE_PARTITION_FRACTION:
        action = "drop_subset_analysis_degenerate"
        reason = "more_than_95_percent_on_one_side"
    elif min(fractions.values()) < MIN_PARTITION_FRACTION:
        action = "drop_subset_analysis_under_25_percent"
        reason = "at_least_one_side_below_25_percent"
    else:
        action = "retain_subset_analysis"
        reason = "both_sides_at_least_25_percent"
    return {
        "action": action,
        "reason": reason,
        "subset_analysis_included": action == "retain_subset_analysis",
        "fractions": fractions,
    }


def _static_boxes(scene_params: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    panel_center = np.asarray(scene_params["protr_center"], dtype=np.float64)
    panel_half = np.asarray(scene_params["protr_half"], dtype=np.float64)
    boxes = []
    skipped_panel = False
    for center_value, half_value in scene_params["obstacle_aabbs"]:
        center = np.asarray(center_value, dtype=np.float64)
        half = np.asarray(half_value, dtype=np.float64)
        if np.allclose(center, panel_center) and np.allclose(half, panel_half):
            skipped_panel = True
            continue
        boxes.append((center, half))
    if not skipped_panel:
        raise ValueError("active panel was not present in obstacle_aabbs")
    return boxes


def analyze_episode(result_path: Path, expected_row: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    for key in ("episode_id", "row_sha256", "role", "role_index"):
        if result.get(key) != expected_row.get(key):
            raise ValueError(f"source identity mismatch for {result_path}: {key}")
    if result.get("intrusion_side") is not None and result.get(
        "intrusion_side"
    ) != expected_row.get("intrusion_side"):
        raise ValueError(f"source identity mismatch for {result_path}: intrusion_side")
    status = str(result.get("status"))
    if status not in ELIGIBLE_STATUSES:
        return {
            "episode_id": expected_row["episode_id"],
            "role": expected_row["role"],
            "role_index": expected_row["role_index"],
            "intrusion_side": expected_row["intrusion_side"],
            "row_sha256": expected_row["row_sha256"],
            "eligible": False,
            "exclusion_reason": status,
        }

    trajectory_path = result_path.parent / "trajectory.h5"
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    scene_params = result["scene_params"]
    center = np.asarray(scene_params["protr_center"], dtype=np.float64)
    half = np.asarray(scene_params["protr_half"], dtype=np.float64)
    probes = panel_probe_points(center, half)
    with h5py.File(trajectory_path, "r") as handle:
        trajectories = [key for key in handle if key.startswith("traj_")]
        if trajectories != ["traj_0"]:
            raise ValueError(f"expected only traj_0 in {trajectory_path}: {trajectories}")
        trajectory = handle["traj_0"]
        stop = first_close_step(trajectory["actions/joint_pos"])
        extrinsics = np.asarray(
            trajectory["obs/sensor_param/wrist_camera/extrinsic_cv"][:stop],
            dtype=np.float64,
        )
        if len(extrinsics) != stop or stop <= 0:
            raise ValueError(f"invalid pre-close window in {trajectory_path}: {stop}")
        intrinsic = np.asarray(
            trajectory["obs/sensor_param/wrist_camera/intrinsic_cv"][0],
            dtype=np.float64,
        )

    inside = probes_inside_frustum(extrinsics, probes)
    blocked = static_occlusion_mask(
        camera_centers(extrinsics), probes, inside, _static_boxes(scene_params)
    )
    visible = np.any(inside & ~blocked, axis=1)
    disadvantaged = ~visible
    fraction = float(np.mean(disadvantaged))
    return {
        "episode_id": expected_row["episode_id"],
        "role": expected_row["role"],
        "role_index": expected_row["role_index"],
        "intrusion_side": expected_row["intrusion_side"],
        "row_sha256": expected_row["row_sha256"],
        "eligible": True,
        "pregrasp_control_steps": int(stop),
        "panel_visible_steps": int(np.count_nonzero(visible)),
        "panel_outside_frustum_steps": int(np.count_nonzero(~np.any(inside, axis=1))),
        "panel_geometry_occluded_steps": int(
            np.count_nonzero(np.any(inside, axis=1) & ~visible)
        ),
        "vision_disadvantaged_steps": int(np.count_nonzero(disadvantaged)),
        "vision_disadvantaged_fraction": fraction,
        "vision_disadvantaged_subset": bool(
            fraction >= DISADVANTAGED_FRACTION_THRESHOLD
        ),
        "recorded_intrinsic_audit": {
            "principal_point_xy": [float(intrinsic[0, 2]), float(intrinsic[1, 2])],
            "implied_width_height": [
                float(2.0 * intrinsic[0, 2]),
                float(2.0 * intrinsic[1, 2]),
            ],
        },
    }


def build_document(manifest_path: Path, collection_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    payload = dict(manifest)
    observed_manifest_hash = payload.pop("manifest_sha256")
    if sha256_payload(payload) != observed_manifest_hash:
        raise ValueError("source manifest self-hash mismatch")
    source_rows = [row for row in manifest["rows"] if row["role"] in SOURCE_ROLES]
    analyzed = []
    for row in source_rows:
        result_path = collection_root / "rows" / row["episode_id"] / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        analyzed.append(analyze_episode(result_path, row))

    eligible = [row for row in analyzed if row["eligible"]]
    excluded = [row for row in analyzed if not row["eligible"]]
    occluded = sum(row["vision_disadvantaged_subset"] for row in eligible)
    non_occluded = len(eligible) - occluded
    action = partition_action(occluded, non_occluded)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "criterion": {
            "name": "wrist_vision_disadvantaged_at_least_half_pregrasp",
            "threshold": DISADVANTAGED_FRACTION_THRESHOLD,
            "threshold_tuned": False,
            "pregrasp_window": "trajectory start through the step before first commanded gripper close",
            "visibility": "at least one of five fixed camera-facing panel probes is inside the physical wrist-camera frustum and not blocked by a committed static-scene AABB",
            "probe_face_x_fraction": PROBE_FACE_X_FRACTION,
            "probe_yz_half_extent_fractions": [list(value) for value in PROBE_OFFSETS],
            "camera": {
                "actual_image_width": ACTUAL_IMAGE_WIDTH,
                "actual_image_height": ACTUAL_IMAGE_HEIGHT,
                "fovy_degrees": WRIST_FOVY_DEGREES,
                "extrinsic_source": "recorded obs/sensor_param/wrist_camera/extrinsic_cv",
                "intrinsic_source": "physical fovy and actual 624x352 renderer dimensions",
                "recorded_intrinsic_used": False,
                "recorded_intrinsic_issue": "CameraParameterSensor reverses the configured width and height; the stored principal point implies 352x624 while recorded video is 624x352.",
            },
            "policy_outcome_fields_loaded": False,
        },
        "viability_rules": {
            "retain_subset_if_each_side_at_least_fraction": MIN_PARTITION_FRACTION,
            "degenerate_if_more_than_fraction_on_one_side": DEGENERATE_PARTITION_FRACTION,
        },
        "source": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": observed_manifest_hash,
            "manifest_file_sha256": sha256_file(manifest_path),
            "collection_root": str(collection_root),
            "roles": list(SOURCE_ROLES),
            "requested_episode_count": len(source_rows),
            "eligible_episode_count": len(eligible),
            "excluded_episode_count": len(excluded),
            "excluded_status_counts": dict(
                sorted(Counter(row["exclusion_reason"] for row in excluded).items())
            ),
        },
        "partition": {
            "vision_disadvantaged_count": occluded,
            "non_vision_disadvantaged_count": non_occluded,
            **action,
        },
        "rows": eligible,
        "excluded_rows": excluded,
    }
    document["occlusion_subset_sha256"] = sha256_payload(document)
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="configs/pact_collision_candidate_manifest_v2.json"
    )
    parser.add_argument(
        "--collection-root",
        default="assets/datagen/pact_collision_corridor_v2/full_cba7ff88",
    )
    parser.add_argument(
        "--output",
        default="diagnostics_output/pact_contact_endpoint/occlusion_subset.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    document = build_document(Path(args.manifest), Path(args.collection_root))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {output}: {document['partition']['vision_disadvantaged_count']}/"
        f"{document['source']['eligible_episode_count']} vision-disadvantaged; "
        f"action={document['partition']['action']}"
    )
    print(f"sha256={document['occlusion_subset_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
