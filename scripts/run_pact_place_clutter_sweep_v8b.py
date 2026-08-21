#!/usr/bin/env python3
"""B2b Pass 1: propose dense V8B layouts and freeze an interim review config."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for path in (ROOT / "scripts", MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_pact_place_clutter_sweep_v8 as v8  # noqa: E402
from pact_place_corridor_contract import sha256_file, sha256_payload  # noqa: E402

PALETTE_PATH = ROOT / "diagnostics_output/pact_place_clutter_sweep_v8b/palette_stability.json"
BASELINE_PATH = ROOT / "diagnostics_output/pact_place_v8_baseline/analysis.json"
OUTPUT_DIR = ROOT / "diagnostics_output/pact_place_clutter_sweep_v8b"
ANALYSIS_PATH = OUTPUT_DIR / "analysis_pass1.json"
SELECTED_PATH = OUTPUT_DIR / "selected_layouts_pass1.json"
CONFIG_PATH = ROOT / "configs/pact_place_corridor_v8b_pass1.json"
FAMILIES = v8.FAMILIES
SHELF_Z = 0.72
SIDE_WALL_Y = 0.425
CEILING_Z = 1.42
# Retry seeds expose the sampler's true 0.18 m depth lower bound. The B0 sample
# happened to bottom out at 0.782, but runtime containment must use 0.760.
SHALLOW_BACK_X = 0.758
QUAT = [2**-0.5, 2**-0.5, 0.0, 0.0]
MASTER_REVIEW = 2026083001
MASTER_GATE = 2026083002


def _object(item: dict[str, Any], center: list[float], support: str, role: str) -> dict[str, Any]:
    half = (np.asarray(item["dimensions_m"], dtype=float) / 2.0).tolist()
    return {
        "role": role,
        "palette_slot": item["slot"],
        "uid": item["uid"],
        "category": item["category"],
        "size_class": item["size_class"],
        "support": support,
        "center_m": list(map(float, center)),
        "half_m": half,
        "quat_wxyz": QUAT,
    }


def _pack_props(items: list[dict[str, Any]], reverse: bool, variant: int) -> list[dict[str, Any]] | None:
    ordered = list(items)
    random.Random(7919 + variant).shuffle(ordered)
    if reverse:
        ordered.reverse()
    gap = 0.010 + 0.002 * (variant % 3)
    result: list[dict[str, Any]] = []
    for group_index, group in enumerate((ordered[:3], ordered[3:])):
        sign = -1.0 if group_index == 0 else 1.0
        cursor = SIDE_WALL_Y - 0.002
        group_objects = []
        for index, item in enumerate(group):
            half = np.asarray(item["dimensions_m"], dtype=float) / 2.0
            y_abs = cursor - half[1]
            cursor -= 2.0 * half[1] + gap
            if index == 2:
                y_abs = 0.25 + 0.015 * (variant % 4)
            y = sign * y_abs
            x_lo = 0.58 + half[0] + 0.004
            x_hi = SHALLOW_BACK_X - half[0] - 0.004
            if x_lo > x_hi:
                return None
            # The innermost object is the only floor prop deliberately put in
            # the arm's proximity band. Outer props provide density safely at
            # the two side walls.
            rear = index != 2 and (index + variant + group_index) % 2 == 1
            fraction = (0.72 if rear else 0.10) + 0.04 * ((variant // 3) % 3)
            fraction = float(np.clip(fraction, 0.05, 0.88))
            x = x_lo + fraction * (x_hi - x_lo)
            center = [float(x), float(y), float(SHELF_Z + half[2])]
            group_objects.append(_object(item, center, "shelf_standing", "scene_filler"))
        result.extend(group_objects)
    # Role assignment follows geometry, not shuffled list position.
    by_inwardness = sorted(result, key=lambda item: abs(float(item["center_m"][1])))
    for index, item in enumerate(by_inwardness):
        item["role"] = "proximity_event" if index < 2 else (
            "workspace_occluder" if index < 4 else "scene_filler"
        )
    return result


def _mount_objects(
    mounts: dict[str, dict[str, Any]], family: str, side: str, variant: int
) -> list[dict[str, Any]]:
    sign = 1.0 if side == "left" else -1.0
    if family == "F5_overhead_elbow":
        slots = ("04", "00")
    else:
        pairs = (("00", "01"), ("00", "04"), ("01", "04"))
        slots = pairs[variant % len(pairs)]
    result = []
    for index, slot in enumerate(slots):
        item = mounts[slot]
        half = np.asarray(item["dimensions_m"], dtype=float) / 2.0
        mount_sign = sign if index == 0 else -sign
        if family == "F5_overhead_elbow" and index == 0:
            mount_sign = 1.0
        elif family == "F5_overhead_elbow":
            mount_sign = -1.0
        y = mount_sign * (SIDE_WALL_Y - half[1] - 0.002)
        x_lo = 0.58 + half[0] + 0.003
        x_hi = SHALLOW_BACK_X - half[0] - 0.003
        x = x_lo + (0.25 + 0.5 * ((variant + index) % 2)) * (x_hi - x_lo)
        z = max(0.95 + half[2], 1.04 + 0.12 * index + 0.02 * (variant % 3))
        if family == "F5_overhead_elbow" and index == 0:
            x = x_hi if side == "left" else x_lo
            z = 1.10 + 0.05 * (variant % 4)
        role = "proximity_event" if family == "F5_overhead_elbow" and index == 0 else "scene_filler"
        result.append(_object(item, [x, y, z], "overhead", role))
    if family == "F5_overhead_elbow":
        # Keep role totals exact: the first prop remains the second proximity
        # event and the second prop becomes filler.
        pass
    return result


def support_reject(candidate: dict[str, Any]) -> str | None:
    objects = candidate["objects"]
    if not 8 <= len(objects) <= 12:
        return "object_count"
    roles = Counter(item["role"] for item in objects)
    if not (2 <= roles["proximity_event"] <= 3 and 2 <= roles["workspace_occluder"] <= 3 and 4 <= roles["scene_filler"] <= 6):
        return "role_composition"
    if max(Counter(item["category"] for item in objects).values()) > 2:
        return "layout_category_cap"
    overhead = 0
    bounds = []
    target_lo = np.asarray([0.70, -0.10, 0.72])
    target_hi = np.asarray([0.79, 0.10, 0.84])
    for item in objects:
        center = np.asarray(item["center_m"], dtype=float)
        half = np.asarray(item["half_m"], dtype=float)
        lo, hi = center - half, center + half
        if lo[0] < 0.58 or hi[0] > SHALLOW_BACK_X or lo[1] < -SIDE_WALL_Y or hi[1] > SIDE_WALL_Y or lo[2] < SHELF_Z or hi[2] > CEILING_Z:
            return "outside_shallowest_episode_shell"
        if item["support"] == "shelf_standing":
            if abs(float(lo[2]) - SHELF_Z) > 0.002:
                return "shelf_support_mismatch"
        elif item["support"] == "wall_adjacent":
            if abs(float(lo[2]) - SHELF_Z) > 0.002 or min(abs(float(lo[1]) + SIDE_WALL_Y), abs(float(hi[1]) - SIDE_WALL_Y)) > 0.030:
                return "wall_support_mismatch"
        elif item["support"] == "overhead":
            overhead += 1
            if float(lo[2]) < 0.95 or min(abs(float(lo[1]) + SIDE_WALL_Y), abs(float(hi[1]) - SIDE_WALL_Y), abs(float(hi[2]) - CEILING_Z)) > 0.020:
                return "mount_support_mismatch"
        else:
            return "unknown_support"
        if item["support"] != "overhead":
            gap = np.maximum(0.0, np.maximum(lo - target_hi, target_lo - hi))
            if float(np.linalg.norm(gap)) < 0.012:
                return "blocks_or_overlaps_target_rest_envelope"
        bounds.append((lo, hi))
    if overhead < 2:
        return "overhead_count"
    for index, (lo, hi) in enumerate(bounds):
        for other_lo, other_hi in bounds[:index]:
            gap = np.maximum(0.0, np.maximum(lo - other_hi, other_lo - hi))
            if float(np.linalg.norm(gap)) < 0.006:
                return "objects_overlap"
    return None


def pass1_hard_reject(candidate: dict[str, Any], score: dict[str, Any]) -> str | None:
    reason = v8.hard_reject(candidate, score)
    # V8's F5 proxy required link3/4. The realized robot grouping identifies
    # the ceiling-adjacent elbow/forearm segment as link5 as well; V8B keeps
    # the geometric requirement (the proximity object itself is overhead).
    if reason == "overhead_family_not_proximal_link" and score["closest_robot_link"] == "link5":
        return None
    return reason


def generate_candidates(palette: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mounts = {item["slot"]: item for item in palette if item["slot_class"] == "mount"}
    props = {item["slot"]: item for item in palette if item["slot_class"] == "prop"}
    # Six stable compact props fit as three-object clusters at each side wall.
    compact_default = [props[slot] for slot in ("05", "12", "13", "14", "15", "16")]
    compact_f6 = [props[slot] for slot in ("05", "06", "11", "12", "13", "16")]
    candidates = []
    for family in FAMILIES:
        for side in ("left", "right"):
            for variant in range(48):
                for reverse in (False, True):
                    compact = compact_f6 if family == "F6_target_occluding" else compact_default
                    prop_objects = _pack_props(compact, reverse, variant)
                    if prop_objects is None:
                        continue
                    mount_objects = _mount_objects(mounts, family, side, variant)
                    if family == "F6_target_occluding":
                        occluder = next(
                            item for item in prop_objects if item["palette_slot"] == "06"
                        )
                        f6_positions = (
                            ((0.605, 0.04), (0.605, 0.08), (0.643, -0.12))
                            if side == "left"
                            else ((0.625, 0.0), (0.625, 0.0), (0.625, 0.0))
                        )
                        occluder["center_m"][:2] = f6_positions[variant % 3]
                        for item in prop_objects:
                            item["role"] = "scene_filler"
                            if item is not occluder and abs(float(item["center_m"][1])) < 0.30:
                                item["center_m"][1] = 0.31 if float(item["center_m"][1]) >= 0 else -0.31
                        for item in mount_objects:
                            item["role"] = "proximity_event"
                        occluder["role"] = "workspace_occluder"
                        next(
                            item
                            for item in prop_objects
                            if item is not occluder
                        )["role"] = "workspace_occluder"
                    if family == "F5_overhead_elbow":
                        next(
                            item
                            for item in prop_objects
                            if item["role"] == "proximity_event"
                        )["role"] = "scene_filler"
                    objects = [*mount_objects, *prop_objects]
                    focal = next(item for item in objects if item["role"] == "proximity_event")
                    candidate = {
                        "candidate_id": len(candidates),
                        "family": family,
                        "intrusion_side": side,
                        "palette_slot": focal["palette_slot"],
                        "uid": focal["uid"],
                        "category": focal["category"],
                        "size_class": focal["size_class"],
                        "support": focal["support"],
                        "center_m": focal["center_m"],
                        "half_m": focal["half_m"],
                        "quat_wxyz": focal["quat_wxyz"],
                        "objects": objects,
                    }
                    candidates.append(candidate)
    if len(candidates) < 400:
        raise RuntimeError(f"candidate pool too small: {len(candidates)}")
    return candidates


def _rows(layouts: list[dict[str, Any]], palette: list[dict[str, Any]], stream: str, master: int) -> list[dict[str, Any]]:
    ordered = []
    for family in FAMILIES:
        left = sorted((x for x in layouts if x["family"] == family and x["intrusion_side"] == "left"), key=lambda x: x["layout_id"])
        right = sorted((x for x in layouts if x["family"] == family and x["intrusion_side"] == "right"), key=lambda x: x["layout_id"])
        if len(left) != 2 or len(right) != 2:
            raise RuntimeError(f"quota failure for {family}")
        ordered.extend((left[0], right[0], left[1], right[1]))
    result = []
    attempts = Counter()
    for index, layout in enumerate(ordered):
        family = layout["family"]
        attempts[family] += 1
        digest = hashlib.sha256(f"pact-place-v8b:{stream}:{master}:{index}".encode()).digest()
        seed32, seed64 = int.from_bytes(digest[:4], "big"), int.from_bytes(digest[:8], "big")
        rng = random.Random(seed64)
        row = {
            "role_index": index,
            "family": family,
            "family_attempt": attempts[family],
            "layout_id": layout["layout_id"],
            "episode_id": hashlib.sha256(f"pact-place-v8b:{stream}:expert:{master}:{index}:{family}".encode()).hexdigest(),
            "intrusion_side": layout["intrusion_side"],
            "panel_x_jitter_m": round(rng.uniform(-0.015, 0.015), 9),
            "panel_face_jitter_m": round(rng.uniform(-0.005, 0.005), 9),
            "scene_template_house_index": 1,
            "task_seed_u32": seed32,
            "task_seed_u64": seed64,
            "max_sampling_retries": 4,
            "pact_clutter_palette": palette,
            "pact_clutter_layout": layout,
        }
        row["row_sha256"] = sha256_payload(row)
        result.append(row)
    return result


def build_config(layouts: list[dict[str, Any]], palette: list[dict[str, Any]], selected_hash: str) -> dict[str, Any]:
    review = _rows(layouts, palette, "family-review", MASTER_REVIEW)
    gate = _rows(layouts, palette, "phase0-gate", MASTER_GATE)
    source_paths = (
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml",
        "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
        "submodules/molmospaces/molmo_spaces/tasks/pact_place_contact_audit.py",
        "scripts/run_pact_place_clutter_sweep_v8b.py",
        "scripts/run_pact_place_v8b_palette.py",
        "scripts/run_pact_place_expert_screen.py",
    )
    document = {
        "schema_version": "pact_place_corridor_v8b_pass1",
        "status": "pass1_selected_pass2_not_yet_measured",
        "role": "human_design_review_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "scene": {
            "xml": "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml",
            "sampler_class": "PactPlaceCorridorV5Sampler",
            "environment_version": "pact_place_corridor_v5",
            "clutter_mounts_are_mocap": True,
            "clutter_props_are_free_bodies": True,
            "clutter_added_to_obstacle_aabbs": False,
        },
        "palette": palette,
        "selected_layouts_path": str(SELECTED_PATH.relative_to(ROOT)),
        "selected_layouts_sha256": selected_hash,
        "selected_layouts": layouts,
        "family_review": {"stream": "pact-place-v8b-family-review", "one_clean_success_per_family": True, "max_attempts_per_family": 4},
        "family_review_rows": review,
        "phase0_gate": {"stream": "pact-place-v8b-phase0-gate", "requires_explicit_user_approval": True, "n": 24, "prediction": [17, 22]},
        "expert_screen_rows": gate,
        "source_sha256": {path: sha256_file(ROOT / path) for path in source_paths},
    }
    document["config_sha256"] = sha256_payload(document)
    return document


def main() -> int:
    palette_doc = json.loads(PALETTE_PATH.read_text())
    palette = palette_doc["palette"]
    baseline = json.loads(BASELINE_PATH.read_text())
    tracks = v8.load_tracks(baseline)
    records = []
    for candidate in generate_candidates(palette):
        reason = support_reject(candidate)
        if reason is None:
            evaluations = []
            for reference in tracks[candidate["intrusion_side"]]:
                value = v8.score_candidate(candidate, [reference])
                evaluations.append((reference, value, pass1_hard_reject(candidate, value)))
            admissible_references = [item for item in evaluations if item[2] is None]
            reference, score, reason = max(
                admissible_references or evaluations,
                key=lambda item: v8.quality(item[1], candidate["family"]),
            )
            candidate["score"] = score
            candidate["reference_track_role_index"] = reference["role_index"]
            if reason is None:
                candidate["quality"] = v8.quality(score, candidate["family"])
        candidate["admitted"] = reason is None
        candidate["reject_reason"] = reason
        records.append(candidate)
    admitted = [item for item in records if item["admitted"] and not item["score"]["cup_is_closest_body"]]
    availability = Counter((item["family"], item["intrusion_side"]) for item in admitted)
    missing = {(family, side): availability[(family, side)] for family in FAMILIES for side in ("left", "right") if availability[(family, side)] < 2}
    if missing:
        raise SystemExit(f"insufficient Pass 1 candidates: {missing}")
    chosen, features = v8.farthest_point_select(admitted)
    for index, layout in enumerate(chosen):
        layout["layout_id"] = f"v8b_pass1_layout_{index:02d}"
    selected = {"schema_version": "pact_place_v8b_selected_layouts_pass1_v1", "palette": palette, "layouts": chosen}
    selected["selected_layouts_sha256"] = sha256_payload(selected)
    analysis = {
        "schema_version": "pact_place_clutter_sweep_v8b_pass1_v1",
        "role": "b2b_pass1_approximation_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "distance_instrument": "B0_per_geom_world_AABB_first_approximation_only",
        "pass1_is_not_realized_admission": True,
        "n_candidates": len(records),
        "n_admitted_link_primary": len(admitted),
        "reject_counts": dict(Counter(item["reject_reason"] or "admitted" for item in records)),
        "chosen_n": 24,
        "selection_rule": "quota_constrained_farthest_point",
        "min_pairwise_selected_layout_distance": v8.min_pairwise(features),
        "family_side_quotas": {f"{family}/{side}": sum(x["family"] == family and x["intrusion_side"] == side for x in chosen) for family in FAMILIES for side in ("left", "right")},
        "selected_layouts": chosen,
        "candidates": records,
    }
    analysis["analysis_sha256"] = sha256_payload(analysis)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    SELECTED_PATH.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    config = build_config(chosen, palette, selected["selected_layouts_sha256"])
    CONFIG_PATH.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_candidates": len(records), "n_admitted_link_primary": len(admitted), "chosen_n": 24, "config": str(CONFIG_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
