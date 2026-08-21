#!/usr/bin/env python3
"""B1/B2: freeze a real-object palette and select 24 diverse v8 layouts.

The sweep performs no rollouts.  It evaluates at least 400 layouts against the
geom-level AABB tracks produced by ``run_pact_place_v8_baseline.py`` and uses
quota-constrained farthest-point selection, never a top-N ranking.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for search_path in (ROOT / "scripts", MOLMO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from molmo_spaces.utils.object_metadata import ObjectMeta  # noqa: E402
from molmo_spaces.utils.synset_utils import get_valid_pickupable_obja_uids  # noqa: E402
from pact_place_corridor_contract import build_v8_contract, sha256_payload  # noqa: E402

BASELINE = ROOT / "diagnostics_output/pact_place_v8_baseline/analysis.json"
OUTPUT = ROOT / "diagnostics_output/pact_place_clutter_sweep_v8/analysis.json"
SELECTED_OUTPUT = (
    ROOT / "diagnostics_output/pact_place_clutter_sweep_v8/selected_layouts.json"
)
CONFIG_OUTPUT = ROOT / "configs/pact_place_corridor_v8.json"
LINKS = ("link1", "link2", "link3", "link4", "link5", "link6")
TRACK_BODIES = LINKS + ("hand_assembly", "cup")
FAMILIES = (
    "F1_near_forearm_left",
    "F2_near_forearm_right",
    "F3_front_stagger",
    "F4_rear_stagger",
    "F5_overhead_elbow",
    "F6_target_occluding",
)
SUPPORTS = ("shelf_standing", "wall_adjacent", "overhead")
CLEARANCE_C_M = 0.030
NEAR_M = 0.05
MEDIUM_M = 0.10
FAR_M = 0.15
TARGET_VISIBLE_CONSECUTIVE_N = 5
SHELF_TOP_Z = 0.72
SHALLOWEST_BACK_WALL_X_M = 0.782
ENCLOSURE_Y_M = 0.425
ENCLOSURE_TOP_Z_M = 1.42
WRIST_FOV_DEG = 56.74

# Deliberately excludes cups and mugs: visual variety without target ambiguity.
PALETTE_UIDS = (
    "Apple_19",
    "Potato_22",
    "Candle_1",
    "Apple_27",
    "Apple_22",
    "Apple_8",
    "Potato_11",
    "Tomato_12",
    "Candle_3",
    "Bowl_15",
    "Candle_2",
    "Bowl_27",
    "Bowl_6",
    "Bowl_17",
    "Plate_5",
    "Vase_Open_1",
    "Vase_Decorative_1",
    "Plate_15",
)


def size_class(max_dim: float) -> str:
    if max_dim <= 0.10:
        return "small"
    if max_dim <= 0.18:
        return "medium"
    return "large"


def build_palette() -> list[dict[str, Any]]:
    pickupable = set(get_valid_pickupable_obja_uids())
    palette = []
    for slot, uid in enumerate(PALETTE_UIDS):
        if uid not in pickupable:
            raise RuntimeError(f"palette uid is not pickupable: {uid}")
        annotation = ObjectMeta.annotation(uid) or {}
        category = str(annotation.get("category") or "object")
        if "egg" in category.lower() or "egg" in uid.lower():
            raise RuntimeError(f"egg is forbidden from the palette: {uid}")
        if "cup" in category.lower() or "mug" in category.lower():
            raise RuntimeError(f"target-like decoy is forbidden: {uid}")
        bbox = annotation.get("boundingBox") or {}
        dims = [float(bbox.get(axis, 0.0)) for axis in "xyz"]
        maximum = max(dims)
        if min(dims) <= 0.0 or maximum > 0.30:
            raise RuntimeError(f"palette dimensions outside support: {uid} {dims}")
        palette.append(
            {
                "slot": f"{slot:02d}",
                "uid": uid,
                "category": category,
                "dimensions_m": dims,
                "max_dimension_m": maximum,
                "size_class": size_class(maximum),
                "body_prefix": f"pact_clutter_{slot:02d}/",
            }
        )
    if len(palette) < 12 or len(palette) > 20:
        raise RuntimeError("palette must contain 12-20 frozen uids")
    if set(item["size_class"] for item in palette) != {"small", "medium", "large"}:
        raise RuntimeError("palette does not span all three size classes")
    return palette


def _aabb_distance_frames(aabbs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Minimum candidate-to-geom-AABB distance for each replay frame."""
    geom_lo = aabbs[..., :3]
    geom_hi = aabbs[..., 3:]
    gap = np.maximum(0.0, np.maximum(geom_lo - hi, lo - geom_hi))
    return np.linalg.norm(gap, axis=-1).min(axis=-1)


def _ray_box_intersects_segment(
    origin: np.ndarray, target: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    direction = target - origin
    inv = np.divide(
        1.0,
        direction,
        out=np.full_like(direction, np.inf),
        where=np.abs(direction) > 1e-9,
    )
    t1 = (lo - origin) * inv
    t2 = (hi - origin) * inv
    enter = np.max(np.minimum(t1, t2), axis=-1)
    leave = np.min(np.maximum(t1, t2), axis=-1)
    return (leave >= np.maximum(enter, 0.0)) & (enter < 0.98) & (leave > 0.01)


def _longest_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values.tolist():
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _candidate_visible(
    center: np.ndarray, wrist_pos: np.ndarray, wrist_fwd: np.ndarray
) -> np.ndarray:
    delta = center - wrist_pos
    distance = np.linalg.norm(delta, axis=-1)
    direction = np.divide(
        delta,
        distance[:, None],
        out=np.zeros_like(delta),
        where=distance[:, None] > 1e-9,
    )
    return np.sum(direction * wrist_fwd, axis=-1) >= math.cos(
        math.radians(WRIST_FOV_DEG / 2.0)
    )


def _box_visible(
    center: np.ndarray,
    half: np.ndarray,
    wrist_pos: np.ndarray,
    wrist_fwd: np.ndarray,
) -> np.ndarray:
    samples = [center]
    for axis in range(3):
        low = center.copy()
        high = center.copy()
        low[axis] -= half[axis]
        high[axis] += half[axis]
        samples.extend((low, high))
    return np.logical_or.reduce(
        [_candidate_visible(point, wrist_pos, wrist_fwd) for point in samples]
    )


def load_tracks(baseline: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for row in baseline["rows"]:
        if row["source"] != "v6c":
            continue
        path = ROOT / row["track_path"]
        payload = np.load(path)
        grouped[str(row["intrusion_side"])].append(
            {
                "role_index": int(row["role_index"]),
                "phases": payload["phases"].astype(str),
                "target_visible": payload["target_visible"].astype(bool),
                "wrist_pos": payload["wrist_pos"].astype(float),
                "wrist_fwd": payload["wrist_fwd"].astype(float),
                "target_center": payload["target_center"].astype(float),
                **{
                    f"geom_aabb_{body}": payload[f"geom_aabb_{body}"].astype(float)
                    for body in TRACK_BODIES
                },
            }
        )
    if any(len(rows) != 12 for rows in grouped.values()):
        raise RuntimeError(
            f"expected 12 v6c tracks per intrusion side, got "
            f"{ {key: len(value) for key, value in grouped.items()} }"
        )
    return grouped


def _placement(
    family: str,
    palette_item: dict[str, Any],
    x: float,
    y: float,
    *,
    role: str,
) -> dict[str, Any]:
    dims = np.asarray(palette_item["dimensions_m"], dtype=float)
    half = dims / 2.0
    support = "shelf_standing"
    if abs(y) + half[1] > 0.34:
        support = "wall_adjacent"
    if family == "F5_overhead_elbow" and role == "proximity_event":
        support = "overhead"
    center = np.array([x, y, SHELF_TOP_Z + half[2]], dtype=float)
    return {
        "role": role,
        "palette_slot": palette_item["slot"],
        "uid": palette_item["uid"],
        "category": palette_item["category"],
        "size_class": palette_item["size_class"],
        "support": support,
        "center_m": center.tolist(),
        "half_m": half.tolist(),
        # THOR assets use Y-up roots; rotate them into the corridor's Z-up
        # world, matching the target object's established settling path.
        "quat_wxyz": [2**-0.5, 2**-0.5, 0.0, 0.0],
    }


def _feasible_x_interval(item: dict[str, Any]) -> tuple[float, float] | None:
    half_x = float(item["dimensions_m"][0]) / 2.0
    lo = 0.58 + half_x + 0.003
    hi = SHALLOWEST_BACK_WALL_X_M - half_x - 0.003
    return None if lo > hi else (lo, hi)


def _x_at_fraction(item: dict[str, Any], fraction: float) -> float | None:
    interval = _feasible_x_interval(item)
    if interval is None:
        return None
    return float(interval[0] + float(fraction) * (interval[1] - interval[0]))


def generate_candidates(palette: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family_grid = {
        "F1_near_forearm_left": ([0.05, 0.30, 0.55, 0.80], [0.18, 0.24, 0.30, 0.35]),
        "F2_near_forearm_right": ([0.05, 0.30, 0.55, 0.80], [-0.18, -0.24, -0.30, -0.35]),
        "F3_front_stagger": ([0.00, 0.18, 0.36], [-0.32, -0.24, 0.24, 0.32]),
        "F4_rear_stagger": ([0.64, 0.82, 1.00], [-0.32, -0.24, 0.24, 0.32]),
        "F5_overhead_elbow": (
            [0.00, 0.10, 0.20, 0.30, 0.40],
            [-0.36, -0.32, -0.28, -0.24, -0.20, -0.16, 0.32, 0.36],
        ),
        "F6_target_occluding": (
            [0.00, 0.18, 0.36, 0.54, 0.60, 0.64, 0.72],
            [-0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16],
        ),
    }
    visual_by_uid = {item["uid"]: item for item in palette}
    candidates = []
    candidate_id = 0
    for family in FAMILIES:
        x_fractions, ys = family_grid[family]
        for route_side in ("left", "right"):
            for item_index, item in enumerate(palette):
                # At the aperture plane a low-profile real object makes link4,
                # rather than the distal hand/cup, the nearest body.  Taller
                # assets were measured to move the event back to link5.
                if family == "F5_overhead_elbow" and not (
                    0.060 <= float(item["dimensions_m"][2]) <= 0.105
                ):
                    continue
                if family == "F6_target_occluding" and (
                    float(item["dimensions_m"][1]) > 0.060
                ):
                    continue
                for x_fraction in x_fractions:
                    x = _x_at_fraction(item, x_fraction)
                    if x is None:
                        continue
                    for y_value in ys:
                        half_y = float(item["dimensions_m"][1]) / 2.0
                        y = float(y_value)
                        focal = _placement(
                            family, item, x, y, role="proximity_event"
                        )
                        # A second, small free object supplies workspace clutter on the
                        # opposite side.  It is deliberately far from the focal event so
                        # family identity is defined by the proximity object, not by a
                        # repeated decorative placement.
                        visual_item = visual_by_uid[
                            "Candle_3" if item["uid"] == "Candle_1" else "Candle_1"
                        ]
                        visual_half_y = float(visual_item["dimensions_m"][1]) / 2.0
                        sign = -1.0 if y >= 0.0 else 1.0
                        visual_y = sign * (0.365 + visual_half_y)
                        visual_fractions = (
                            (0.14, 0.86)
                            if family == "F6_target_occluding"
                            else (0.86 if x_fraction < 0.5 else 0.14,)
                        )
                        for visual_fraction in visual_fractions:
                            visual_x = _x_at_fraction(visual_item, visual_fraction)
                            if visual_x is None:
                                continue
                            visual = _placement(
                                family,
                                visual_item,
                                visual_x,
                                visual_y,
                                role="workspace_occluder",
                            )
                            candidates.append(
                                {
                                    "candidate_id": candidate_id,
                                    "family": family,
                                    "intrusion_side": route_side,
                                    # Top-level focal fields are the coverage-selection
                                    # coordinates named in the plan.
                                    "palette_slot": focal["palette_slot"],
                                    "uid": focal["uid"],
                                    "category": focal["category"],
                                    "size_class": focal["size_class"],
                                    "support": focal["support"],
                                    "center_m": focal["center_m"],
                                    "half_m": focal["half_m"],
                                    "quat_wxyz": focal["quat_wxyz"],
                                    "objects": [focal, visual],
                                }
                            )
                            candidate_id += 1
    if len(candidates) < 400:
        raise RuntimeError(f"candidate pool too small: {len(candidates)}")
    return candidates


def reject_geometry(candidate: dict[str, Any]) -> str | None:
    boxes = []
    target_lo = np.array([0.72, -0.09, 0.72])
    target_hi = np.array([0.80, 0.09, 0.82])
    for obj in candidate["objects"]:
        center = np.asarray(obj["center_m"])
        half = np.asarray(obj["half_m"])
        lo, hi = center - half, center + half
        if lo[0] < 0.58:
            return "outside_aperture_plane"
        if hi[0] > SHALLOWEST_BACK_WALL_X_M:
            return "outside_shallowest_episode_back_wall"
        if lo[1] < -ENCLOSURE_Y_M or hi[1] > ENCLOSURE_Y_M:
            return "outside_enclosure_y"
        if lo[2] < SHELF_TOP_Z - 1e-9 or hi[2] > ENCLOSURE_TOP_Z_M:
            return "outside_enclosure_z"
        gap = np.maximum(0.0, np.maximum(lo - target_hi, target_lo - hi))
        if float(np.linalg.norm(gap)) < 0.025:
            return "blocks_or_overlaps_target_rest_envelope"
        boxes.append((lo, hi))
    for index, (lo, hi) in enumerate(boxes):
        for other_lo, other_hi in boxes[:index]:
            gap = np.maximum(0.0, np.maximum(lo - other_hi, other_lo - hi))
            if float(np.linalg.norm(gap)) < 0.010:
                return "overlaps_another_object"
    return None


def score_candidate(candidate: dict[str, Any], tracks: list[dict[str, Any]]) -> dict[str, Any]:
    centers = [np.asarray(obj["center_m"], dtype=float) for obj in candidate["objects"]]
    boxes = [
        (
            center - np.asarray(obj["half_m"], dtype=float),
            center + np.asarray(obj["half_m"], dtype=float),
        )
        for center, obj in zip(centers, candidate["objects"])
    ]
    distances = {body: [] for body in TRACK_BODIES}
    phases: list[str] = []
    visible: list[bool] = []
    target_visible_after: list[bool] = []
    target_floor_runs: list[int] = []
    row_target_floor: list[dict[str, Any]] = []
    target_occluded_frames = 0
    for track in tracks:
        row_distances = {}
        for body in TRACK_BODIES:
            values = np.minimum.reduce(
                [
                    _aabb_distance_frames(track[f"geom_aabb_{body}"], lo, hi)
                    for lo, hi in boxes
                ]
            )
            distances[body].append(values)
            row_distances[body] = values
        row_visible = np.logical_or.reduce(
            [
                _box_visible(
                    center,
                    np.asarray(obj["half_m"], dtype=float),
                    track["wrist_pos"],
                    track["wrist_fwd"],
                )
                for center, obj in zip(centers, candidate["objects"])
            ]
        )
        occluded = np.logical_or.reduce(
            [
                _ray_box_intersects_segment(
                    track["wrist_pos"], track["target_center"], lo, hi
                )
                for lo, hi in boxes
            ]
        )
        row_target_visible = track["target_visible"] & ~occluded
        approach = track["phases"] == "pregrasp"
        candidate_occlusion = track["target_visible"] & occluded
        occluded_indices = np.flatnonzero(approach & candidate_occlusion)
        before_occlusion = approach.copy()
        if len(occluded_indices):
            before_occlusion[np.arange(len(before_occlusion)) >= occluded_indices[0]] = False
        run = _longest_true_run(row_target_visible & before_occlusion)
        target_occluded_frames += int(np.sum(candidate_occlusion))
        target_floor_runs.append(run)
        row_target_floor.append(
            {"role_index": track["role_index"], "longest_approach_run": run}
        )
        phases.extend(track["phases"].tolist())
        visible.extend(row_visible.tolist())
        target_visible_after.extend(row_target_visible.tolist())
    distances = {
        body: np.concatenate(chunks, axis=0) for body, chunks in distances.items()
    }
    link_matrix = np.stack([distances[link] for link in LINKS], axis=1)
    frame_link_min = link_matrix.min(axis=1)
    frame_index = int(np.argmin(frame_link_min))
    min_by_link = {link: float(distances[link].min()) for link in LINKS}
    closest_link = min(LINKS, key=min_by_link.get)
    minimum = float(frame_link_min[frame_index])
    cup_min = float(distances["cup"].min())
    hand_min = float(distances["hand_assembly"].min())
    visible_array = np.asarray(visible, dtype=bool)
    target_array = np.asarray(target_visible_after, dtype=bool)
    return {
        "min_clearance_by_link_m": min_by_link,
        "min_link_clearance_m": minimum,
        "min_cup_clearance_m": cup_min,
        "min_hand_clearance_m": hand_min,
        "closest_robot_link": closest_link,
        "phase_of_min_clearance": phases[frame_index],
        "frames_link_clearance_lt_5cm": int(np.sum(frame_link_min < NEAR_M)),
        "frames_link_clearance_lt_10cm": int(np.sum(frame_link_min < MEDIUM_M)),
        "frames_link_clearance_lt_15cm": int(np.sum(frame_link_min < FAR_M)),
        "n_distinct_links_exposed": int(
            sum(value < MEDIUM_M for value in min_by_link.values())
        ),
        "cup_is_closest_body": bool(cup_min < minimum),
        "clutter_visible_frame_fraction": float(visible_array.mean()),
        "visibility_at_min_link_clearance": bool(visible_array[frame_index]),
        "target_visible_frame_fraction": float(target_array.mean()),
        "target_visibility_floor_min_consecutive_frames": min(target_floor_runs),
        "target_visibility_floor_by_row": row_target_floor,
        "target_occluded_frames": int(target_occluded_frames),
    }


def hard_reject(candidate: dict[str, Any], score: dict[str, Any]) -> str | None:
    if score["min_link_clearance_m"] < CLEARANCE_C_M:
        return "intersects_swept_volume_plus_C"
    if score["min_hand_clearance_m"] <= 0.0:
        return "would_contact_hand_or_fingers"
    if score["min_cup_clearance_m"] <= 0.0:
        return "would_contact_carried_cup"
    if score["target_visibility_floor_min_consecutive_frames"] < TARGET_VISIBLE_CONSECUTIVE_N:
        return "target_visibility_floor_violated"
    if score["frames_link_clearance_lt_15cm"] == 0:
        return "no_link_proximity_exposure"
    if candidate["family"] == "F5_overhead_elbow" and score["closest_robot_link"] not in {
        "link3",
        "link4",
    }:
        return "overhead_family_not_proximal_link"
    if candidate["family"] == "F6_target_occluding" and score["target_occluded_frames"] == 0:
        return "target_occluding_family_does_not_occlude"
    return None


def quality(score: dict[str, Any], family: str) -> float:
    value = -abs(score["min_link_clearance_m"] - 0.075)
    value += min(score["frames_link_clearance_lt_10cm"], 100) * 0.0005
    value += score["n_distinct_links_exposed"] * 0.01
    value += 0.01 if not score["cup_is_closest_body"] else -0.2
    if family == "F6_target_occluding":
        value += min(score["target_occluded_frames"], 100) * 0.0002
    return float(value)


def _raw_features(candidate: dict[str, Any]) -> list[Any]:
    score = candidate["score"]
    visual_center = candidate["objects"][1]["center_m"]
    return [
        candidate["center_m"][0],
        abs(candidate["center_m"][1]),
        candidate["center_m"][2],
        candidate["size_class"],
        candidate["support"],
        score["closest_robot_link"],
        score["phase_of_min_clearance"],
        *[score["min_clearance_by_link_m"][link] for link in LINKS],
        score["frames_link_clearance_lt_5cm"],
        int(score["visibility_at_min_link_clearance"]),
        visual_center[0],
        abs(visual_center[1]),
        visual_center[2],
    ]


def encode_features(candidates: list[dict[str, Any]]) -> np.ndarray:
    raw = [_raw_features(candidate) for candidate in candidates]
    numeric_columns = list(range(3)) + list(range(7, 18))
    categorical_columns = (3, 4, 5, 6)
    numeric = np.asarray(
        [[float(row[index]) for index in numeric_columns] for row in raw], dtype=float
    )
    lo = numeric.min(axis=0)
    span = np.maximum(numeric.max(axis=0) - lo, 1e-9)
    blocks = [(numeric - lo) / span]
    for column in categorical_columns:
        values = sorted({str(row[column]) for row in raw})
        blocks.append(
            np.asarray(
                [[float(str(row[column]) == value) for value in values] for row in raw]
            )
        )
    return np.concatenate(blocks, axis=1)


def farthest_point_select(admitted: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray]:
    features = encode_features(admitted)
    quotas = {(family, side): 2 for family in FAMILIES for side in ("left", "right")}
    used = Counter()
    chosen_indices: list[int] = []
    remaining = set(range(len(admitted)))
    seed = max(remaining, key=lambda index: (admitted[index]["quality"], -index))
    chosen_indices.append(seed)
    remaining.remove(seed)
    used[(admitted[seed]["family"], admitted[seed]["intrusion_side"])] += 1
    while len(chosen_indices) < 24:
        eligible = [
            index
            for index in remaining
            if used[(admitted[index]["family"], admitted[index]["intrusion_side"])]
            < quotas[(admitted[index]["family"], admitted[index]["intrusion_side"])]
        ]
        if not eligible:
            raise RuntimeError(f"farthest-point selection exhausted at {len(chosen_indices)}")
        selected_features = features[chosen_indices]
        best = max(
            eligible,
            key=lambda index: (
                float(np.linalg.norm(selected_features - features[index], axis=1).min()),
                admitted[index]["quality"],
                -index,
            ),
        )
        chosen_indices.append(best)
        remaining.remove(best)
        used[(admitted[best]["family"], admitted[best]["intrusion_side"])] += 1
    if used != Counter(quotas):
        raise RuntimeError(f"family/side quotas not exact: {used}")
    return [admitted[index] for index in chosen_indices], features[chosen_indices]


def min_pairwise(features: np.ndarray) -> float:
    if len(features) < 2:
        return 0.0
    best = float("inf")
    for index in range(len(features)):
        if index:
            best = min(
                best,
                float(np.linalg.norm(features[index] - features[:index], axis=1).min()),
            )
    return best


def main() -> int:
    baseline = json.loads(BASELINE.read_text())
    if int((baseline.get("aggregates") or {}).get("v6c", {}).get("n_episodes", 0)) != 24:
        raise SystemExit("B0 must contain all 24 v6c episodes before B2 runs")
    palette = build_palette()
    tracks = load_tracks(baseline)
    records = []
    for candidate in generate_candidates(palette):
        reason = reject_geometry(candidate)
        if reason is None:
            side_tracks = tracks[candidate["intrusion_side"]]
            if candidate["family"] in {
                "F5_overhead_elbow",
                "F6_target_occluding",
            }:
                reference_evaluations = []
                for track in side_tracks:
                    track_score = score_candidate(candidate, [track])
                    track_reason = hard_reject(candidate, track_score)
                    reference_evaluations.append((track, track_score, track_reason))
                admissible_references = [
                    item for item in reference_evaluations if item[2] is None
                ]
                if admissible_references:
                    reference_track, score, reason = max(
                        admissible_references,
                        key=lambda item: quality(item[1], candidate["family"]),
                    )
                else:
                    reference_track, score, reason = reference_evaluations[0]
                candidate["reference_track_rejections"] = {
                    str(track["role_index"]): track_reason
                    for track, _track_score, track_reason in reference_evaluations
                }
            else:
                reference_track = side_tracks[
                    int(candidate["candidate_id"]) % len(side_tracks)
                ]
                score = score_candidate(candidate, [reference_track])
                reason = hard_reject(candidate, score)
            candidate["reference_track_role_index"] = int(
                reference_track["role_index"]
            )
            candidate["score"] = score
            if reason is None:
                candidate["quality"] = quality(score, candidate["family"])
        candidate["admitted"] = reason is None
        candidate["reject_reason"] = reason
        records.append(candidate)
    admitted = [candidate for candidate in records if candidate["admitted"]]
    availability = Counter(
        (candidate["family"], candidate["intrusion_side"]) for candidate in admitted
    )
    missing = {key: count for key, count in availability.items() if count < 2}
    expected = {(family, side) for family in FAMILIES for side in ("left", "right")}
    missing.update({key: 0 for key in expected - set(availability)})
    if missing:
        raise RuntimeError(f"insufficient admitted candidates for quotas: {missing}")
    link_primary = [
        candidate
        for candidate in admitted
        if not candidate["score"]["cup_is_closest_body"]
    ]
    link_primary_availability = Counter(
        (candidate["family"], candidate["intrusion_side"])
        for candidate in link_primary
    )
    link_primary_missing = {
        key: link_primary_availability.get(key, 0)
        for key in expected
        if link_primary_availability.get(key, 0) < 2
    }
    if link_primary_missing:
        raise RuntimeError(
            "insufficient arm-link-primary candidates for quotas: "
            f"{link_primary_missing}"
        )
    chosen, chosen_features = farthest_point_select(link_primary)
    for layout_index, candidate in enumerate(chosen):
        candidate["layout_id"] = f"v8_layout_{layout_index:02d}"
    quotas = Counter((c["family"], c["intrusion_side"]) for c in chosen)
    visibility_values = [
        int(c["score"]["visibility_at_min_link_clearance"]) for c in chosen
    ]
    reject_counts = Counter(c["reject_reason"] or "admitted" for c in records)
    selected_document = {
        "schema_version": "pact_place_v8_selected_layouts_v1",
        "palette": palette,
        "layouts": chosen,
    }
    selected_document["selected_layouts_sha256"] = sha256_payload(selected_document)
    analysis = {
        "schema_version": "pact_place_clutter_sweep_v8_v1",
        "role": "b1_b2_replay_sweep_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "no_rollouts": True,
        "baseline_analysis_sha256": baseline["analysis_sha256"],
        "distance_instrument": "per_geom_world_AABB_from_B0_tracks_not_TCP",
        "clearance_C_m": CLEARANCE_C_M,
        "target_visible_consecutive_frames_N": TARGET_VISIBLE_CONSECUTIVE_N,
        "shallowest_back_wall_x_m": SHALLOWEST_BACK_WALL_X_M,
        "palette": palette,
        "palette_size_class_counts": dict(Counter(p["size_class"] for p in palette)),
        "palette_category_counts": dict(Counter(p["category"] for p in palette)),
        "n_candidates": len(records),
        "n_admitted": len(admitted),
        "n_arm_link_primary_admitted": len(link_primary),
        "n_rejected": len(records) - len(admitted),
        "reject_counts": dict(reject_counts),
        "selection_rule": "quota_constrained_farthest_point_seeded_by_best_admissible",
        "selection_eligibility": "cup_is_closest_body_false",
        "feature_space": [
            "clutter_x",
            "abs_y",
            "z",
            "size_class",
            "support_type",
            "closest_robot_link",
            "phase_of_min_clearance",
            "min_clearance_by_link",
            "frames_in_near_band",
            "visibility_at_min_link_clearance",
            "workspace_occluder_x",
            "workspace_occluder_abs_y",
            "workspace_occluder_z",
        ],
        "chosen_n": len(chosen),
        "family_side_quotas": {
            f"{family}/{side}": quotas[(family, side)]
            for family in FAMILIES
            for side in ("left", "right")
        },
        "min_pairwise_selected_layout_distance": min_pairwise(chosen_features),
        "visibility_at_min_link_clearance_values": visibility_values,
        "visibility_spans_range": len(set(visibility_values)) > 1,
        "cup_is_closest_body_count": sum(
            c["score"]["cup_is_closest_body"] for c in chosen
        ),
        "selected_layouts": chosen,
        "candidates": records,
    }
    analysis["analysis_sha256"] = sha256_payload(analysis)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    SELECTED_OUTPUT.write_text(
        json.dumps(selected_document, indent=2, sort_keys=True) + "\n"
    )
    contract = build_v8_contract(SELECTED_OUTPUT)
    CONFIG_OUTPUT.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "n_candidates": len(records),
                "n_admitted": len(admitted),
                "chosen_n": len(chosen),
                "family_side_quotas": analysis["family_side_quotas"],
                "min_pairwise_selected_layout_distance": analysis[
                    "min_pairwise_selected_layout_distance"
                ],
                "visibility_spans_range": analysis["visibility_spans_range"],
                "cup_is_closest_body_count": analysis["cup_is_closest_body_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
