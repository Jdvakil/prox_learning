#!/usr/bin/env python3
"""Deterministic V10.3 static-pendant joint-route contract.

V10.3 starts unauthorized and stays unauthorized. Every artifact built from this
module carries authorizes_gate / authorizes_collection / authorizes_training /
authorizes_evaluation / phase0_passed false. Only an explicit owner
``approve_phase0`` bound to the preflight, review, causal, scene, route, and
implementation hashes may unlock the 24-row gate, and even a passing gate
authorizes nothing downstream.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

from pact_place_corridor_contract import sha256_file
from pact_place_v101_empirical_qualification_contract import (
    PROTECTED_ARTIFACTS as V101_PROTECTED_ARTIFACTS,
)
from pact_place_v103_geometry import (
    ALL_GEOMS_V103,
    ENVIRONMENT_VERSION_V103,
    HEIGHT_LATTICE_M,
    PENDANT_BODY_V103,
    SAMPLER_CLASS_V103,
    SCENE_XML_RELATIVE_V103,
    enumerate_v103_assemblies,
)
from pact_place_v103_joint_route import (
    CORNER_MIN_CLEARANCE_M,
    EDGE_MIN_CLEARANCE_M,
    LANE_MAGNITUDES_M,
    MIN_DETOUR_M,
    NODE_MIN_CLEARANCE_M,
    N_HALTON_SEEDS,
    PASS_Z_OFFSETS_M,
    STAGING_BUFFERS_M,
    enumerate_templates,
)
from pact_place_v102_route import speed_schedule, speed_schedule_sha256
from pact_place_v9_contract import sha256_payload
from pact_place_v95_contract import build_v95_layout, load_v95_palette
from run_pact_place_v96_cluster_causal_proximity import ADMISSION_FLOOR, CORRIDOR_LINKS
from run_pact_place_v9_v0c3_causal_proximity import (
    ABS_DELTA_FLOOR_M,
    MAX_PAIRED_CHANGED_VALUE_RATIO,
)

ROOT = Path(__file__).resolve().parents[1]

CONTRACT_VERSION = "pact_place_v103_static_pendant_joint_route_v1"
ENVIRONMENT_VERSION = ENVIRONMENT_VERSION_V103
SAMPLER_CLASS = SAMPLER_CLASS_V103

SMOKE_STREAM = "pact-place-v10.3-static-pendant-joint-route-smoke"
SMOKE_MASTER_SEED = 2026103000
REVIEW_STREAM = "pact-place-v10.3-static-pendant-joint-route-human-review"
REVIEW_MASTER_SEED = 2026103002
GATE_STREAM = "pact-place-v10.3-static-pendant-joint-route-phase0"
GATE_MASTER_SEED = 2026103001

N_SMOKE_REPEATS = 1
N_REVIEW_REPEATS = 2
N_GATE_REPEATS = 4
N_SMOKE_ROWS = 6
N_REVIEW_ROWS = 12
N_GATE_ROWS = 24

PHYSICS_CLEAN_FAMILIES = (
    "F0_target_side_stagger",
    "F1_inner_panel_stagger",
    "F2_outer_panel_stagger",
)

MIN_SMOKE_CLEAN_SUCCESSES = 6
MIN_REVIEW_CLEAN_SUCCESSES = 10
MIN_CLEAN_PER_REVIEW_CELL = 1
MIN_GATE_CLEAN_SUCCESSES = 20
MIN_CLEAN_PER_GATE_CELL = 3
MIN_ROW_CLEARANCE_M = 0.015
MAX_SIDE_IMBALANCE = MAX_PAIRED_CHANGED_VALUE_RATIO

POLICY_TIMESTEP_MS = 66.0
REVIEW_FPS = 1000.0 / POLICY_TIMESTEP_MS
REVIEW_FRAME_STRIDE = 1

SEARCH_ROOT_RELATIVE = "diagnostics_output/pact_place_v103_ik_search"
PREFLIGHT_ROOT_RELATIVE = "diagnostics_output/pact_place_v103_preflight"
SMOKE_ROOT_RELATIVE = "diagnostics_output/pact_place_v103_screen"
REVIEW_ROOT_RELATIVE = "diagnostics_output/pact_place_v103_review"
CAUSAL_ROOT_RELATIVE = "diagnostics_output/pact_place_v103_causal"
PHASE0_ROOT_RELATIVE = "diagnostics_output/pact_place_v103_phase0"

V102_PREFLIGHT_RELATIVE = (
    "diagnostics_output/pact_place_v102_preflight/preflight.json"
)
V102_PREFLIGHT_SHA256 = (
    "6c5079916775e8a2093defb1547a3fa85ef9b32dcc4fddcf785ffa6c3276976d"
)
V102_CONTACT_PARITY_RELATIVE = (
    "diagnostics_output/pact_place_v102_preflight/contact_parity_root_cause.json"
)
V102_CONTACT_PARITY_SHA256 = (
    "e4e544a999534b322d177d8b296aa4c7580d9b7627bf89bf0d42630fdd0774df"
)
V102_ITEM5_COMPLETE_CASES = 5
V102_ITEM6_CASES_MEETING_FLOOR = 0

PROTECTED_ARTIFACTS = tuple(V101_PROTECTED_ARTIFACTS) + (
    (V102_PREFLIGHT_RELATIVE, V102_PREFLIGHT_SHA256, "payload"),
    (V102_CONTACT_PARITY_RELATIVE, V102_CONTACT_PARITY_SHA256, "payload"),
)

IMPLEMENTATION_PATHS = (
    "scripts/pact_place_v103_contract.py",
    "scripts/pact_place_v103_geometry.py",
    "scripts/pact_place_v103_joint_route.py",
    "scripts/search_pact_place_v103_joint_route.py",
    "scripts/run_pact_place_v103_preflight.py",
    "scripts/run_pact_place_v103_screen.py",
    "scripts/run_pact_place_v103_review.py",
    "scripts/run_pact_place_v103_causal.py",
    "scripts/run_pact_place_v103_phase0.py",
    "scripts/pact_place_v102_geometry.py",
    "scripts/pact_place_v102_route.py",
    "scripts/pact_geom_distance.py",
    "scripts/run_pact_place_expert_screen.py",
    "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
)


def empty_authorization() -> dict[str, bool]:
    return {
        "authorizes_gate": False,
        "authorizes_collection": False,
        "authorizes_training": False,
        "authorizes_evaluation": False,
        "authorizes_eval": False,
        "phase0_passed": False,
        "eligible_for_separate_collection_authorization": False,
        "environment_qualified": False,
    }


# ---------------------------------------------------------------------------
# V10.2 errata, read from the immutable payload rather than restated
# ---------------------------------------------------------------------------
def v102_item5_case_table() -> list[dict[str, Any]]:
    document = json.loads((ROOT / V102_PREFLIGHT_RELATIVE).read_text())
    return list(document["items"]["5_complete_sequential_ik"]["cases"])


def v102_item5_complete_count() -> int:
    return sum(
        1 for case in v102_item5_case_table() if case["complete_sequential_ik"]
    )


def v102_item6_cases_meeting_floor() -> int:
    document = json.loads((ROOT / V102_PREFLIGHT_RELATIVE).read_text())
    cases = document["items"]["6_per_component_pendant_clearance"]["cases"]
    return sum(1 for case in cases if not case["components_below_floor"])


def _payload_sha256(path: Path) -> str:
    document = json.loads(path.read_text())
    stored = document.get("artifact_sha256") or document.get("analysis_sha256")
    if not stored:
        raise ValueError(f"{path} is missing artifact_sha256")
    return str(stored)


def verify_protected_artifacts() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected, kind in PROTECTED_ARTIFACTS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"protected artifact missing: {relative}")
        digest = _payload_sha256(path) if kind == "payload" else sha256_file(path)
        if digest != expected:
            raise ValueError(
                f"protected artifact hash mismatch: {relative} {digest} != {expected}"
            )
        observed[relative] = digest
    return observed


def implementation_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in IMPLEMENTATION_PATHS:
        target = ROOT / path
        out[path] = sha256_file(target) if target.is_file() else "absent"
    return out


def implementation_sha256() -> str:
    return sha256_payload(implementation_hashes())


def search_lattice() -> dict[str, Any]:
    return {
        "height_lattice_m": list(HEIGHT_LATTICE_M),
        "lane_magnitudes_m": list(LANE_MAGNITUDES_M),
        "staging_buffers_m": list(STAGING_BUFFERS_M),
        "pass_z_offsets_m": list(PASS_Z_OFFSETS_M),
        "left_pass_rotations": [item[0] for item in enumerate_templates("left")[:0]] or
        sorted({item["pass_rotation_key"] for item in enumerate_templates("left")}),
        "right_pass_rotations": sorted(
            {item["pass_rotation_key"] for item in enumerate_templates("right")}
        ),
        "n_templates_per_side": len(enumerate_templates("left")),
        "n_halton_seeds": int(N_HALTON_SEEDS),
        "node_min_clearance_m": float(NODE_MIN_CLEARANCE_M),
        "edge_min_clearance_m": float(EDGE_MIN_CLEARANCE_M),
        "corner_min_clearance_m": float(CORNER_MIN_CLEARANCE_M),
        "min_detour_m": float(MIN_DETOUR_M),
    }


# ---------------------------------------------------------------------------
# Manifest rows
# ---------------------------------------------------------------------------
def _seed(stream: str, index: int, master_seed: int) -> tuple[int, int]:
    digest = hashlib.sha256(
        f"{stream}:{int(master_seed)}:{int(index)}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big"), int.from_bytes(digest[:8], "big")


def _layouts(palette_document: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        family_id: {
            side: build_v95_layout(
                palette_document, family_id=family_id, intrusion_side=side
            )
            for side in ("left", "right")
        }
        for family_id in PHYSICS_CLEAN_FAMILIES
    }


def _rows(
    *,
    stream: str,
    master_seed: int,
    n_repeats: int,
    palette_document: dict[str, Any],
    assembly: dict[str, Any],
    route_mapping: dict[str, Any],
    scene_sha256: str,
    implementation_digest: str,
) -> list[dict[str, Any]]:
    rng = random.Random(int(master_seed))
    layouts = _layouts(palette_document)
    pairs = [
        (family_id, repeat)
        for family_id in PHYSICS_CLEAN_FAMILIES
        for repeat in range(n_repeats)
    ]
    rng.shuffle(pairs)
    palette = list(palette_document["palette"])
    assembly_digest = sha256_payload(assembly)
    mapping_digest = sha256_payload(route_mapping)
    rows: list[dict[str, Any]] = []
    for pair_index, (family_id, repeat) in enumerate(pairs):
        x_jitter = {
            "01": round(rng.uniform(-0.020, 0.020), 9),
            "06": round(rng.uniform(-0.005, 0.005), 9),
        }
        y_jitter = {
            "01": round(rng.uniform(-0.005, 0.005), 9),
            "06": round(rng.uniform(-0.010, 0.010), 9),
        }
        panel_x_jitter = round(rng.uniform(-0.015, 0.015), 9)
        panel_face_jitter = round(rng.uniform(-0.005, 0.005), 9)
        for side in ("left", "right"):
            index = len(rows)
            layout = layouts[family_id][side]
            seed_u32, seed_u64 = _seed(stream, index, master_seed)
            cell = f"{family_id}:{side}"
            row = {
                "role_index": index,
                "episode_id": hashlib.sha256(
                    f"{stream}:expert:{master_seed}:{pair_index}:{side}".encode()
                ).hexdigest(),
                "intrusion_side": side,
                "panel_x_jitter_m": panel_x_jitter,
                "panel_face_jitter_m": panel_face_jitter,
                "clutter_x_jitter_m": dict(x_jitter),
                "clutter_y_jitter_m": dict(y_jitter),
                "target_x_jitter_m": 0.0,
                "target_y_jitter_m": 0.0,
                "paired_side_cell": pair_index,
                "family_repeat": repeat,
                "scene_template_house_index": 1,
                "task_seed_u32": seed_u32,
                "task_seed_u64": seed_u64,
                "max_sampling_retries": 12,
                "sampler_class": SAMPLER_CLASS,
                "pact_clutter_palette": palette,
                "pact_clutter_layout": layout,
                "layout_id": layout["layout_id"],
                "layout_family_id": family_id,
                "family": family_id,
                "seed_stream": stream,
                "pact_v103_assembly": assembly,
                "pact_v103_assembly_sha256": assembly_digest,
                "pact_v103_scene_sha256": scene_sha256,
                "pact_v103_route_template": dict(route_mapping["cells"][cell]),
                "pact_v103_route_mapping_sha256": mapping_digest,
                "pact_v103_contract_version": CONTRACT_VERSION,
                "implementation_sha256": implementation_digest,
            }
            row["row_sha256"] = sha256_payload(row)
            rows.append(row)
    return rows


def build_contract(
    *, assembly: dict[str, Any], route_mapping: dict[str, Any], scene_sha256: str
) -> dict[str, Any]:
    """Build the frozen V10.3 contract for one selected geometry and mapping."""
    palette_document = load_v95_palette()
    digest = implementation_sha256()
    kwargs = dict(
        palette_document=palette_document,
        assembly=assembly,
        route_mapping=route_mapping,
        scene_sha256=scene_sha256,
        implementation_digest=digest,
    )
    smoke_rows = _rows(
        stream=SMOKE_STREAM,
        master_seed=SMOKE_MASTER_SEED,
        n_repeats=N_SMOKE_REPEATS,
        **kwargs,
    )
    review_rows = _rows(
        stream=REVIEW_STREAM,
        master_seed=REVIEW_MASTER_SEED,
        n_repeats=N_REVIEW_REPEATS,
        **kwargs,
    )
    gate_rows = _rows(
        stream=GATE_STREAM,
        master_seed=GATE_MASTER_SEED,
        n_repeats=N_GATE_REPEATS,
        **kwargs,
    )
    for label, rows, expected in (
        ("smoke", smoke_rows, N_SMOKE_ROWS),
        ("review", review_rows, N_REVIEW_ROWS),
        ("gate", gate_rows, N_GATE_ROWS),
    ):
        if len(rows) != expected:
            raise RuntimeError(f"expected {expected} {label} rows, got {len(rows)}")
    seeds = {
        "smoke": {int(row["task_seed_u32"]) for row in smoke_rows},
        "review": {int(row["task_seed_u32"]) for row in review_rows},
        "gate": {int(row["task_seed_u32"]) for row in gate_rows},
    }
    for left in seeds:
        for right in seeds:
            if left < right and seeds[left] & seeds[right]:
                raise RuntimeError(f"{left} and {right} task-seed streams intersect")
    payload = {
        "schema_version": CONTRACT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "scene_xml": SCENE_XML_RELATIVE_V103,
        "scene_sha256": scene_sha256,
        "pendant_body": PENDANT_BODY_V103,
        "pendant_geoms": list(ALL_GEOMS_V103),
        "pendant_static": True,
        "pendant_runtime_repositioned": False,
        "assembly": assembly,
        "assembly_sha256": sha256_payload(assembly),
        "route_mapping": route_mapping,
        "route_mapping_sha256": sha256_payload(route_mapping),
        "search_lattice": search_lattice(),
        "speed_schedule": speed_schedule(),
        "speed_schedule_sha256": speed_schedule_sha256(),
        "min_row_clearance_m": float(MIN_ROW_CLEARANCE_M),
        "review_fps": REVIEW_FPS,
        "review_frame_stride": REVIEW_FRAME_STRIDE,
        "policy_timestep_ms": POLICY_TIMESTEP_MS,
        "families": list(PHYSICS_CLEAN_FAMILIES),
        "smoke_stream": SMOKE_STREAM,
        "review_stream": REVIEW_STREAM,
        "gate_stream": GATE_STREAM,
        "smoke_master_seed": SMOKE_MASTER_SEED,
        "review_master_seed": REVIEW_MASTER_SEED,
        "gate_master_seed": GATE_MASTER_SEED,
        "n_smoke_rows": N_SMOKE_ROWS,
        "n_review_rows": N_REVIEW_ROWS,
        "n_gate_rows": N_GATE_ROWS,
        "admission_floor": ADMISSION_FLOOR,
        "abs_delta_floor_m": ABS_DELTA_FLOOR_M,
        "max_paired_changed_value_ratio": MAX_SIDE_IMBALANCE,
        "corridor_links": list(CORRIDOR_LINKS),
        "implementation_sha256": digest,
        "implementation_files": implementation_hashes(),
        "protected_artifacts": {
            relative: expected for relative, expected, _kind in PROTECTED_ARTIFACTS
        },
        "v102_item5_complete_cases": V102_ITEM5_COMPLETE_CASES,
        "v102_item6_cases_meeting_floor": V102_ITEM6_CASES_MEETING_FLOOR,
        "smoke_rows": smoke_rows,
        "review_rows": review_rows,
        "gate_rows": gate_rows,
        **empty_authorization(),
    }
    payload["contract_sha256"] = sha256_payload(payload)
    return payload


def cell_key(family_id: str, intrusion_side: str) -> tuple[str, str]:
    return str(family_id), str(intrusion_side)


def all_cell_keys() -> tuple[str, ...]:
    return tuple(
        f"{family}:{side}"
        for family in PHYSICS_CLEAN_FAMILIES
        for side in ("left", "right")
    )


def distribution_counts(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = cell_key(str(row["layout_family_id"]), str(row["intrusion_side"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Row admission
# ---------------------------------------------------------------------------
REQUIRED_ROUTE_KEYS = (
    "direction",
    "route_key",
    "qpos_sha256",
    "start_qpos_matches",
    "geometry_sha256",
    "cell_key",
    "target_held",
    "n_waypoints",
    "executed_joint_sequence",
    "used_tcp_ik_at_runtime",
    "min_clearance_m",
    "segments",
)


def _route_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    block = result.get("pact_v103_joint_route") or {}
    return [dict(block.get("inbound") or {}), dict(block.get("outbound") or {})]


def route_telemetry_complete(result: dict[str, Any]) -> bool:
    records = _route_records(result)
    if len(records) != 2 or any(not record for record in records):
        return False
    for record in records:
        if any(key not in record for key in REQUIRED_ROUTE_KEYS):
            return False
        if record.get("executed_joint_sequence") is not True:
            return False
        if record.get("used_tcp_ik_at_runtime") is not False:
            return False
        if record.get("start_qpos_matches") is not True:
            return False
    return True


def route_defects(result: dict[str, Any]) -> list[str]:
    if not route_telemetry_complete(result):
        return ["missing_route_telemetry"]
    defects: list[str] = []
    for prefix, record in zip(("inbound", "outbound"), _route_records(result)):
        if record.get("fallback_taken"):
            defects.append(f"{prefix}_fallback_to_tcp_ik")
        if record.get("route_hash_matches") is False:
            defects.append(f"{prefix}_route_hash_mismatch")
        minimum = record.get("min_clearance_m")
        if minimum is None:
            defects.append(f"{prefix}_missing_clearance")
        elif float(minimum) < MIN_ROW_CLEARANCE_M - 1e-12:
            defects.append(f"{prefix}_clearance_below_floor")
    return sorted(set(defects))


def frame_defects(result: dict[str, Any]) -> list[str]:
    frames = result.get("pendant_frame_telemetry") or {}
    if not frames:
        return ["missing_frame_telemetry"]
    defects: list[str] = []
    if int(frames.get("n_frames") or 0) <= 0:
        return ["missing_frame_telemetry"]
    if int(frames.get("n_frames_measured") or 0) != int(frames.get("n_frames") or 0):
        defects.append("incomplete_frame_clearance")
    if int(frames.get("live_pendant_contact_frames") or 0) > 0:
        defects.append("live_pendant_contact")
    minimum = frames.get("min_clearance_m")
    if minimum is None:
        defects.append("missing_frame_clearance")
    elif float(minimum) < MIN_ROW_CLEARANCE_M - 1e-12:
        defects.append("frame_clearance_below_floor")
    for name, value in (frames.get("per_component_min_clearance_m") or {}).items():
        if value is None:
            defects.append(f"missing_component_clearance:{name}")
        elif float(value) < MIN_ROW_CLEARANCE_M - 1e-12:
            defects.append(f"component_clearance_below_floor:{name}")
    return sorted(set(defects))


def contact_defects(result: dict[str, Any]) -> list[str]:
    audit = result.get("contact_audit") or {}
    totals = audit.get("contact_class_totals") or {}
    if not totals:
        return ["missing_contact_audit"]
    defects = [
        f"{key}_contact"
        for key in ("mounted_fixture", "hazard_bar", "other_environment", "clutter")
        if int(totals.get(key, 0)) > 0
    ]
    if result.get("clutter_stability_events"):
        defects.append("clutter_stability_event")
    return sorted(set(defects))


def row_defects(result: dict[str, Any]) -> list[str]:
    status = result.get("status")
    if status == "sampling_failure":
        return ["sampling_failure"]
    if status == "infrastructure_failure":
        return ["infrastructure_failure"]
    if status != "complete":
        return ["nonterminal"]
    defects = list(route_defects(result))
    defects.extend(contact_defects(result))
    defects.extend(frame_defects(result))
    if not result.get("task_success"):
        defects.append("task_failure")
    if not result.get("clean_success"):
        defects.append("not_strict_clean_success")
    return sorted(set(defects))


def is_v103_clean_success(result: dict[str, Any]) -> bool:
    return not row_defects(result)


def _stage_eligibility(
    rows: Sequence[dict[str, Any]],
    results: Sequence[dict[str, Any]],
    *,
    expected_rows: int,
    min_clean: int,
    min_clean_per_cell: int,
    passed_key: str,
) -> dict[str, Any]:
    by_episode = {str(item["episode_id"]): item for item in results}
    failures: list[dict[str, Any]] = []
    reconciled = True
    if len(rows) != expected_rows or len(results) != expected_rows:
        reconciled = False
        failures.append(
            {
                "code": "row_count",
                "detail": f"rows={len(rows)} results={len(results)} expected={expected_rows}",
            }
        )
    infrastructure = 0
    clean_by_cell = {
        (family, side): 0
        for family in PHYSICS_CLEAN_FAMILIES
        for side in ("left", "right")
    }
    pendant_contact_rows = 0
    n_clean = 0
    for row in rows:
        result = by_episode.get(str(row["episode_id"]))
        if result is None:
            reconciled = False
            failures.append(
                {"code": "missing_result", "role_index": row["role_index"]}
            )
            continue
        if result.get("row_sha256") != row.get("row_sha256"):
            reconciled = False
            failures.append({"code": "row_sha_mismatch", "role_index": row["role_index"]})
        if result.get("status") == "infrastructure_failure":
            infrastructure += 1
        totals = (result.get("contact_audit") or {}).get("contact_class_totals") or {}
        frames = result.get("pendant_frame_telemetry") or {}
        if int(totals.get("mounted_fixture", 0)) or int(
            frames.get("live_pendant_contact_frames") or 0
        ):
            pendant_contact_rows += 1
            failures.append(
                {"code": "pendant_contact", "role_index": row["role_index"]}
            )
        for defect in row_defects(result):
            failures.append(
                {
                    "code": defect,
                    "role_index": row["role_index"],
                    "family": row.get("layout_family_id"),
                    "intrusion_side": row.get("intrusion_side"),
                }
            )
        if is_v103_clean_success(result):
            n_clean += 1
            clean_by_cell[
                cell_key(str(row["layout_family_id"]), str(row["intrusion_side"]))
            ] += 1
    cell_failures = [
        {
            "code": "cell_clean_shortfall",
            "family": family,
            "intrusion_side": side,
            "clean": count,
            "required": min_clean_per_cell,
        }
        for (family, side), count in sorted(clean_by_cell.items())
        if count < min_clean_per_cell
    ]
    failures.extend(cell_failures)
    passed = bool(
        reconciled
        and infrastructure == 0
        and pendant_contact_rows == 0
        and n_clean >= min_clean
        and not cell_failures
    )
    return {
        passed_key: passed,
        "reconciled": reconciled,
        "n_rows": len(results),
        "infrastructure_failures": infrastructure,
        "pendant_contact_rows": pendant_contact_rows,
        "clean_successes": n_clean,
        "min_clean_successes": min_clean,
        "min_clean_per_cell": min_clean_per_cell,
        "clean_by_cell": {
            f"{family}:{side}": count
            for (family, side), count in sorted(clean_by_cell.items())
        },
        "failures": failures,
        **empty_authorization(),
    }


def smoke_eligibility(rows, results) -> dict[str, Any]:
    return _stage_eligibility(
        rows,
        results,
        expected_rows=N_SMOKE_ROWS,
        min_clean=MIN_SMOKE_CLEAN_SUCCESSES,
        min_clean_per_cell=1,
        passed_key="smoke_passed",
    )


def review_eligibility(rows, results) -> dict[str, Any]:
    return _stage_eligibility(
        rows,
        results,
        expected_rows=N_REVIEW_ROWS,
        min_clean=MIN_REVIEW_CLEAN_SUCCESSES,
        min_clean_per_cell=MIN_CLEAN_PER_REVIEW_CELL,
        passed_key="eligible_for_human_review",
    )


def gate_eligibility(rows, results) -> dict[str, Any]:
    return _stage_eligibility(
        rows,
        results,
        expected_rows=N_GATE_ROWS,
        min_clean=MIN_GATE_CLEAN_SUCCESSES,
        min_clean_per_cell=MIN_CLEAN_PER_GATE_CELL,
        passed_key="phase0_passed",
    )


def lowest_clean_row_per_cell(rows, results) -> dict[tuple[str, str], dict[str, Any]]:
    by_episode = {str(item["episode_id"]): item for item in results}
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: int(item["role_index"])):
        result = by_episode.get(str(row["episode_id"]))
        if result is None or not is_v103_clean_success(result):
            continue
        key = cell_key(str(row["layout_family_id"]), str(row["intrusion_side"]))
        selected.setdefault(key, {"row": row, "result": result})
    return selected


def assert_phase0_approval(
    approval: dict[str, Any] | None,
    *,
    contract_sha256: str,
    preflight_sha256: str,
    review_manifest_sha256: str,
    causal_artifact_sha256: str,
    scene_sha256: str,
    route_mapping_sha256: str,
    implementation_sha256_value: str,
) -> None:
    if not approval:
        raise PermissionError("Phase 0 requires an explicit owner human_approval.json")
    if approval.get("decision") != "approve_phase0":
        raise PermissionError(f"Phase 0 refused: decision={approval.get('decision')!r}")
    expected = {
        "contract_sha256": contract_sha256,
        "preflight_sha256": preflight_sha256,
        "review_manifest_sha256": review_manifest_sha256,
        "causal_artifact_sha256": causal_artifact_sha256,
        "scene_sha256": scene_sha256,
        "route_mapping_sha256": route_mapping_sha256,
        "implementation_sha256": implementation_sha256_value,
    }
    for key, digest in expected.items():
        if approval.get(key) != digest:
            raise PermissionError(
                f"Phase 0 approval is not bound to {key}: "
                f"{approval.get(key)!r} != {digest!r}"
            )
