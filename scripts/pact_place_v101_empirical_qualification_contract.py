#!/usr/bin/env python3
"""Deterministic V10.1 empirical pendant qualification contract.

Does not authorize collection, training, evaluation, Phase 0 without an
explicit owner decision, alternative pendants, alternative lanes, F3, or
three-lobe search. Route-v2 is retained as a historical result of a flawed
scalar environment predicate, not as physical infeasibility.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

from pact_place_corridor_contract import PLACE_V5_SCENE_SHA256, sha256_file
from pact_place_v10_compound_pendant_contract import (
    EMPIRICAL_LIVE_CONTACT_V1,
    ENDPOINT_ONLY_PRIMITIVE,
    ENVIRONMENT_VERSION,
    MIN_DETOUR_M,
    PLACE_V10_SCENE_SHA256,
    PROBE_NEGATIVE_LOBE,
    PROBE_POSITIVE_LOBE,
    ROUTE_RELATIVE,
    ROUTE_V1_PAYLOAD_SHA256,
    ROUTE_V2_GEOMETRY_RELATIVE,
    ROUTE_V2_RELATIVE,
    SAMPLER_CLASS,
    SCENE_XML_RELATIVE,
    V2_ATOMIC_SCORES_RELATIVE,
    V2_ATOMIC_SCORES_SHA256,
    V2_CATALOG_RELATIVE,
    V2_CATALOG_SHA256,
    V2_PREFILTER_INDICES_RELATIVE,
    V2_PREFILTER_INDICES_SHA256,
    V2_SITING_PAYLOAD_SHA256,
    V2_SITING_RELATIVE,
    V5_SCENE_XML_RELATIVE,
    V99_RECONSTRUCTION_RELATIVE,
    V99_RECONSTRUCTION_SHA256,
    V99_SITING_SHA256,
    V99_SNAPSHOT_RELATIVE,
    V99_SNAPSHOT_SHA256,
)
from pact_place_v10_geometry import planning_probe_assembly
from pact_place_v9_contract import sha256_payload
from pact_place_v95_contract import V95_LAYOUT_FAMILIES, build_v95_layout, load_v95_palette
from run_pact_place_v96_cluster_causal_proximity import ADMISSION_FLOOR, CORRIDOR_LINKS
from run_pact_place_v9_v0c3_causal_proximity import (
    ABS_DELTA_FLOOR_M,
    MAX_PAIRED_CHANGED_VALUE_RATIO,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_VERSION = "pact_place_v101_empirical_qualification_v1"
REVIEW_STREAM = "pact-place-v10.1-pendant-human-review"
GATE_STREAM = "pact-place-v10.1-pendant-phase0"
REVIEW_MASTER_SEED = 2026091002
GATE_MASTER_SEED = 2026091001
N_REVIEW_ROWS = 12
N_GATE_ROWS = 24
N_REVIEW_REPEATS = 2
N_GATE_REPEATS = 4
PHYSICS_CLEAN_FAMILIES = (
    "F0_target_side_stagger",
    "F1_inner_panel_stagger",
    "F2_outer_panel_stagger",
)
LEFT_LANE_Y_M = -0.30
RIGHT_LANE_Y_M = 0.30
SLAB_PADDING_M = 0.08
MIN_REVIEW_CLEAN_SUCCESSES = 10
MIN_CLEAN_PER_CELL = 1
MAX_SIDE_IMBALANCE = MAX_PAIRED_CHANGED_VALUE_RATIO
ROUTE_V1_MASK_RELATIVE = (
    "diagnostics_output/pact_place_v10_route/route_morphology_mask.npz"
)
ROUTE_V1_MASK_SHA256 = (
    "6c2609d11dccb69537970fbd7decc2e1b4efc8c2b83275fe3ef65275728a8274"
)
ROUTE_V2_PAYLOAD_SHA256 = (
    "e311ba01c77c14b3a930be8dd9d4d40e9de483710521f8662d2e3a55357f71e1"
)
ROUTE_V2_GEOMETRY_SHA256 = (
    "48e643e6d6b768b3a2dba491c3199c859c8ba1287a75379e504a09b5fdefc74a"
)
ROUTE_V2_GEOMETRY_MASK_RELATIVE = (
    "diagnostics_output/pact_place_v10_route_v2/geometry_mask.npz"
)
ROUTE_V2_GEOMETRY_MASK_SHA256 = (
    "19978e65a7a239a543058919d127a4183f55c85143f79b89ed2e973290b9b509"
)
ROUTE_V2_MASK_RELATIVE = (
    "diagnostics_output/pact_place_v10_route_v2/route_morphology_mask.npz"
)
ROUTE_V2_MASK_SHA256 = (
    "a20d2e6c0ced5e0807103b3f0f7050a798a5c89c447945fe9898460f2e286bb1"
)
ENVIRONMENT_DUMP_RELATIVE = (
    "diagnostics_output/pact_place_v10_siting_v2/environment_geoms.pkl.gz"
)
ENVIRONMENT_DUMP_SHA256 = (
    "c6388adff0976fa25626c24617766d68647bc4cfceda6b691f7ecc24fca9d448"
)
V1_CATALOG_RELATIVE = "diagnostics_output/pact_place_v10_siting/exact_survivors.npz"
V1_CATALOG_SHA256 = (
    "63369af3552bbb806a61fea97d281011374ee25bb375004876704b920b6f3443"
)
V99_SITING_RELATIVE = "diagnostics_output/pact_place_v99_siting/siting.json"
IMPLEMENTATION_PATHS = (
    "scripts/pact_place_v101_empirical_qualification_contract.py",
    "scripts/run_pact_place_v101_empirical_review.py",
    "scripts/run_pact_place_v101_empirical_causal.py",
    "scripts/run_pact_place_v101_empirical_phase0.py",
    "scripts/pact_place_v10_compound_pendant_contract.py",
    "scripts/pact_place_v10_geometry.py",
    "scripts/pact_place_v10_route.py",
    "scripts/pact_place_v10_scene.py",
    "scripts/run_pact_place_expert_screen.py",
    "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    SCENE_XML_RELATIVE,
)
PROTECTED_ARTIFACTS = (
    (V2_SITING_RELATIVE, V2_SITING_PAYLOAD_SHA256, "payload"),
    (V2_CATALOG_RELATIVE, V2_CATALOG_SHA256, "bytes"),
    (V2_ATOMIC_SCORES_RELATIVE, V2_ATOMIC_SCORES_SHA256, "bytes"),
    (V2_PREFILTER_INDICES_RELATIVE, V2_PREFILTER_INDICES_SHA256, "bytes"),
    (ENVIRONMENT_DUMP_RELATIVE, ENVIRONMENT_DUMP_SHA256, "bytes"),
    (V1_CATALOG_RELATIVE, V1_CATALOG_SHA256, "bytes"),
    (ROUTE_RELATIVE, ROUTE_V1_PAYLOAD_SHA256, "payload"),
    (ROUTE_V1_MASK_RELATIVE, ROUTE_V1_MASK_SHA256, "bytes"),
    (ROUTE_V2_RELATIVE, ROUTE_V2_PAYLOAD_SHA256, "payload"),
    (ROUTE_V2_GEOMETRY_RELATIVE, ROUTE_V2_GEOMETRY_SHA256, "payload"),
    (ROUTE_V2_GEOMETRY_MASK_RELATIVE, ROUTE_V2_GEOMETRY_MASK_SHA256, "bytes"),
    (ROUTE_V2_MASK_RELATIVE, ROUTE_V2_MASK_SHA256, "bytes"),
    (V99_RECONSTRUCTION_RELATIVE, V99_RECONSTRUCTION_SHA256, "payload"),
    (V99_SNAPSHOT_RELATIVE, V99_SNAPSHOT_SHA256, "payload"),
    (V99_SITING_RELATIVE, V99_SITING_SHA256, "payload"),
    (V5_SCENE_XML_RELATIVE, PLACE_V5_SCENE_SHA256, "bytes"),
    (SCENE_XML_RELATIVE, PLACE_V10_SCENE_SHA256, "bytes"),
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


def frozen_probe_lobes() -> dict[str, Any]:
    return {
        "negative_lobe": {
            "center_m": list(PROBE_NEGATIVE_LOBE["center_m"]),
            "half_m": list(PROBE_NEGATIVE_LOBE["half_m"]),
        },
        "positive_lobe": {
            "center_m": list(PROBE_POSITIVE_LOBE["center_m"]),
            "half_m": list(PROBE_POSITIVE_LOBE["half_m"]),
        },
    }


def frozen_assembly() -> dict[str, Any]:
    assembly = planning_probe_assembly()
    expected = frozen_probe_lobes()
    lobes = [item for item in assembly["components"] if item.get("role") == "lobe"]
    negative = next(item for item in lobes if item.get("side") == "negative")
    positive = next(item for item in lobes if item.get("side") == "positive")
    if list(negative["center_m"]) != expected["negative_lobe"]["center_m"]:
        raise ValueError("probe_v2 negative lobe center drifted")
    if list(negative["half_m"]) != expected["negative_lobe"]["half_m"]:
        raise ValueError("probe_v2 negative lobe half drifted")
    if list(positive["center_m"]) != expected["positive_lobe"]["center_m"]:
        raise ValueError("probe_v2 positive lobe center drifted")
    if list(positive["half_m"]) != expected["positive_lobe"]["half_m"]:
        raise ValueError("probe_v2 positive lobe half drifted")
    if assembly.get("probe_label") != "probe_v2":
        raise ValueError("planning probe is not probe_v2")
    return assembly


def frozen_route_for_side(intrusion_side: str) -> dict[str, Any]:
    side = str(intrusion_side)
    if side == "left":
        lane = LEFT_LANE_Y_M
    elif side == "right":
        lane = RIGHT_LANE_Y_M
    else:
        raise ValueError(f"intrusion_side must be left or right, got {intrusion_side!r}")
    return {
        "rewrite_primitive": ENDPOINT_ONLY_PRIMITIVE,
        "qualification_mode": EMPIRICAL_LIVE_CONTACT_V1,
        "inbound_lane_y_m": float(lane),
        "outbound_lane_y_m": float(lane),
        "inbound_padding_m": float(SLAB_PADDING_M),
        "outbound_padding_m": float(SLAB_PADDING_M),
        "slab_padding_m": float(SLAB_PADDING_M),
        "left_lane_y_m": float(LEFT_LANE_Y_M),
        "right_lane_y_m": float(RIGHT_LANE_Y_M),
        "min_detour_m": float(MIN_DETOUR_M),
        "max_segment_translation_m": 0.005,
        "max_segment_rotation_deg": 2.0,
    }


def implementation_hashes() -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def implementation_sha256() -> str:
    return sha256_payload(implementation_hashes())


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
    if sha256_file(ROOT / SCENE_XML_RELATIVE) != PLACE_V10_SCENE_SHA256:
        raise ValueError("V10 scene hash mismatch")
    return observed


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
            route = frozen_route_for_side(side)
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
                "pact_v10_pendant_assembly": assembly,
                "pact_v10_route": route,
                "pact_v101_contract_version": CONTRACT_VERSION,
                "implementation_sha256": implementation_digest,
            }
            row["row_sha256"] = sha256_payload(row)
            rows.append(row)
    return rows


def build_contract() -> dict[str, Any]:
    palette_document = load_v95_palette()
    assembly = frozen_assembly()
    digest = implementation_sha256()
    review_rows = _rows(
        stream=REVIEW_STREAM,
        master_seed=REVIEW_MASTER_SEED,
        n_repeats=N_REVIEW_REPEATS,
        palette_document=palette_document,
        assembly=assembly,
        implementation_digest=digest,
    )
    gate_rows = _rows(
        stream=GATE_STREAM,
        master_seed=GATE_MASTER_SEED,
        n_repeats=N_GATE_REPEATS,
        palette_document=palette_document,
        assembly=assembly,
        implementation_digest=digest,
    )
    if len(review_rows) != N_REVIEW_ROWS:
        raise RuntimeError(f"expected {N_REVIEW_ROWS} review rows, got {len(review_rows)}")
    if len(gate_rows) != N_GATE_ROWS:
        raise RuntimeError(f"expected {N_GATE_ROWS} gate rows, got {len(gate_rows)}")
    review_seeds = {int(row["task_seed_u32"]) for row in review_rows}
    gate_seeds = {int(row["task_seed_u32"]) for row in gate_rows}
    if review_seeds & gate_seeds:
        raise RuntimeError("review and gate task-seed streams intersect")
    payload = {
        "schema_version": CONTRACT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "scene_xml": SCENE_XML_RELATIVE,
        "scene_sha256": PLACE_V10_SCENE_SHA256,
        "review_stream": REVIEW_STREAM,
        "gate_stream": GATE_STREAM,
        "review_master_seed": REVIEW_MASTER_SEED,
        "gate_master_seed": GATE_MASTER_SEED,
        "n_review_rows": N_REVIEW_ROWS,
        "n_gate_rows": N_GATE_ROWS,
        "families": list(PHYSICS_CLEAN_FAMILIES),
        "probe_lobes": frozen_probe_lobes(),
        "assembly_id": assembly.get("assembly_id"),
        "assembly": assembly,
        "route_constants": {
            "rewrite_primitive": ENDPOINT_ONLY_PRIMITIVE,
            "qualification_mode": EMPIRICAL_LIVE_CONTACT_V1,
            "slab_padding_m": SLAB_PADDING_M,
            "left_lane_y_m": LEFT_LANE_Y_M,
            "right_lane_y_m": RIGHT_LANE_Y_M,
            "min_detour_m": MIN_DETOUR_M,
        },
        "admission_floor": ADMISSION_FLOOR,
        "abs_delta_floor_m": ABS_DELTA_FLOOR_M,
        "max_paired_changed_value_ratio": MAX_SIDE_IMBALANCE,
        "corridor_links": list(CORRIDOR_LINKS),
        "implementation_sha256": digest,
        "implementation_files": implementation_hashes(),
        "protected_artifacts": {relative: expected for relative, expected, _kind in PROTECTED_ARTIFACTS},
        "review_rows": review_rows,
        "gate_rows": gate_rows,
        **empty_authorization(),
    }
    payload["contract_sha256"] = sha256_payload(payload)
    return payload


def cell_key(family_id: str, intrusion_side: str) -> tuple[str, str]:
    return str(family_id), str(intrusion_side)


def _route_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    block = result.get("pendant_v10") or {}
    return [dict(block.get("inbound") or {}), dict(block.get("outbound") or {})]


def route_telemetry_complete(result: dict[str, Any]) -> bool:
    required = (
        "rewrite_primitive",
        "qualification_mode",
        "lane_y_m",
        "padding_m",
        "min_abs_detour_m",
        "fallback_taken",
        "clipped",
        "wrong_way",
        "frozen_endpoint_preserved",
        "offline_strict_environment_preclearance_used",
        "strict_environment_preclearance_intentionally_not_used",
    )
    records = _route_records(result)
    if len(records) != 2 or any(not record for record in records):
        return False
    for record in records:
        if any(key not in record for key in required):
            return False
        if record.get("rewrite_primitive") != ENDPOINT_ONLY_PRIMITIVE:
            return False
        if record.get("qualification_mode") != EMPIRICAL_LIVE_CONTACT_V1:
            return False
        if record.get("offline_strict_environment_preclearance_used") is not False:
            return False
        if record.get("strict_environment_preclearance_intentionally_not_used") is not True:
            return False
    return True


def route_defects(result: dict[str, Any]) -> list[str]:
    defects: list[str] = []
    if not route_telemetry_complete(result):
        defects.append("missing_telemetry")
        return defects
    for prefix, record in zip(("inbound", "outbound"), _route_records(result)):
        if record.get("fallback_taken"):
            defects.append(f"{prefix}_fallback")
        if record.get("clipped"):
            defects.append(f"{prefix}_clipped")
        if record.get("wrong_way"):
            defects.append(f"{prefix}_wrong_way")
        if record.get("frozen_endpoint_preserved") is not True:
            defects.append(f"{prefix}_endpoint_mutation")
        if record.get("detour_meets_minimum") is False:
            defects.append(f"{prefix}_insufficient_detour")
    return defects


def is_v101_clean_success(result: dict[str, Any]) -> bool:
    if result.get("status") != "complete":
        return False
    if not result.get("task_success"):
        return False
    if not result.get("clean_success"):
        return False
    if route_defects(result):
        return False
    return True


def distribution_counts(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = cell_key(str(row["layout_family_id"]), str(row["intrusion_side"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def review_eligibility(rows: Sequence[dict[str, Any]], results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_episode = {str(item["episode_id"]): item for item in results}
    failures: list[dict[str, Any]] = []
    reconciled = True
    if len(rows) != N_REVIEW_ROWS or len(results) != N_REVIEW_ROWS:
        reconciled = False
        failures.append(
            {
                "code": "row_count",
                "detail": f"rows={len(rows)} results={len(results)} expected={N_REVIEW_ROWS}",
            }
        )
    infrastructure = 0
    clean_by_cell: dict[tuple[str, str], int] = {
        (family, side): 0
        for family in PHYSICS_CLEAN_FAMILIES
        for side in ("left", "right")
    }
    n_clean = 0
    for row in rows:
        result = by_episode.get(str(row["episode_id"]))
        if result is None:
            reconciled = False
            failures.append(
                {"code": "missing_result", "episode_id": row["episode_id"], "role_index": row["role_index"]}
            )
            continue
        if result.get("row_sha256") != row.get("row_sha256"):
            reconciled = False
            failures.append(
                {"code": "row_sha_mismatch", "role_index": row["role_index"]}
            )
        if result.get("status") == "infrastructure_failure":
            infrastructure += 1
            failures.append(
                {"code": "infrastructure_failure", "role_index": row["role_index"]}
            )
        defects = route_defects(result) if result.get("status") == "complete" else ["missing_telemetry"]
        for defect in defects:
            failures.append(
                {"code": defect, "role_index": row["role_index"], "episode_id": row["episode_id"]}
            )
        if is_v101_clean_success(result):
            n_clean += 1
            key = cell_key(str(row["layout_family_id"]), str(row["intrusion_side"]))
            clean_by_cell[key] = clean_by_cell.get(key, 0) + 1
    cell_failures = [
        {"code": "cell_clean_shortfall", "family": family, "intrusion_side": side, "clean": count}
        for (family, side), count in sorted(clean_by_cell.items())
        if count < MIN_CLEAN_PER_CELL
    ]
    failures.extend(cell_failures)
    eligible = bool(
        reconciled
        and infrastructure == 0
        and n_clean >= MIN_REVIEW_CLEAN_SUCCESSES
        and not cell_failures
        and not any(
            item["code"]
            in {
                "missing_telemetry",
                "inbound_fallback",
                "outbound_fallback",
                "inbound_clipped",
                "outbound_clipped",
                "inbound_wrong_way",
                "outbound_wrong_way",
                "inbound_endpoint_mutation",
                "outbound_endpoint_mutation",
            }
            for item in failures
        )
    )
    return {
        "eligible_for_human_review": eligible,
        "reconciled": reconciled,
        "n_rows": len(results),
        "infrastructure_failures": infrastructure,
        "clean_successes": n_clean,
        "min_clean_successes": MIN_REVIEW_CLEAN_SUCCESSES,
        "clean_by_cell": {
            f"{family}:{side}": count for (family, side), count in sorted(clean_by_cell.items())
        },
        "authorizes_gate": False,
        "authorizes_collection": False,
        "failures": failures,
    }


def lowest_clean_row_per_cell(
    rows: Sequence[dict[str, Any]], results: Sequence[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    by_episode = {str(item["episode_id"]): item for item in results}
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: int(item["role_index"])):
        result = by_episode.get(str(row["episode_id"]))
        if result is None or not is_v101_clean_success(result):
            continue
        key = cell_key(str(row["layout_family_id"]), str(row["intrusion_side"]))
        if key not in selected:
            selected[key] = {"row": row, "result": result}
    return selected


def assert_phase0_approval(
    approval: dict[str, Any] | None,
    *,
    review_manifest_sha256: str,
    causal_artifact_sha256: str,
    contract_sha256: str,
) -> None:
    if not approval:
        raise PermissionError("Phase 0 requires an explicit owner human_approval.json")
    if approval.get("decision") != "approve_phase0":
        raise PermissionError(
            f"Phase 0 refused: decision={approval.get('decision')!r}"
        )
    expected = {
        "review_manifest_sha256": review_manifest_sha256,
        "causal_artifact_sha256": causal_artifact_sha256,
        "contract_sha256": contract_sha256,
    }
    for key, digest in expected.items():
        if approval.get(key) != digest:
            raise PermissionError(
                f"Phase 0 approval is not bound to {key}: "
                f"{approval.get(key)!r} != {digest!r}"
            )


def paired_side_clutter_identical(rows: Sequence[dict[str, Any]]) -> bool:
    by_pair: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(int(row["paired_side_cell"]), []).append(row)
    for pair_rows in by_pair.values():
        if len(pair_rows) != 2:
            return False
        left, right = pair_rows
        if {left["intrusion_side"], right["intrusion_side"]} != {"left", "right"}:
            return False
        if left["clutter_x_jitter_m"] != right["clutter_x_jitter_m"]:
            return False
        if left["clutter_y_jitter_m"] != right["clutter_y_jitter_m"]:
            return False
        if left["panel_x_jitter_m"] != right["panel_x_jitter_m"]:
            return False
        if left["panel_face_jitter_m"] != right["panel_face_jitter_m"]:
            return False
        left_xy = {
            str(item["palette_slot"]): list(item["center_m"][:2])
            for item in left["pact_clutter_layout"]["objects"]
            if str(item.get("role")) in {"inbound_vessel", "outbound_vessel"}
        }
        right_xy = {
            str(item["palette_slot"]): list(item["center_m"][:2])
            for item in right["pact_clutter_layout"]["objects"]
            if str(item.get("role")) in {"inbound_vessel", "outbound_vessel"}
        }
        if left_xy != right_xy:
            return False
        if left["pact_clutter_layout"]["inbound_vessel_center_xy_m"] != right[
            "pact_clutter_layout"
        ]["inbound_vessel_center_xy_m"]:
            return False
    return True


def family_ids_allowed() -> tuple[str, ...]:
    return PHYSICS_CLEAN_FAMILIES


def v95_family_coordinate_source() -> dict[str, Any]:
    return {family: V95_LAYOUT_FAMILIES[family] for family in PHYSICS_CLEAN_FAMILIES}


def admit_fixed_route_on_stock(
    positions,
    rotations,
    *,
    panel_side: str,
    direction: str,
    assembly: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from pact_place_v10_route import plan_lane_endpoint_only

    assembly = assembly or frozen_assembly()
    route = frozen_route_for_side(panel_side)
    freeze_start = str(direction) == "outbound"
    freeze_final = str(direction) == "inbound"
    planned = plan_lane_endpoint_only(
        positions,
        rotations,
        assembly=assembly,
        panel_side=panel_side,
        lane_y_m=float(route["inbound_lane_y_m"]),
        padding_m=float(SLAB_PADDING_M),
        freeze_start=freeze_start,
        freeze_final=freeze_final,
    )
    admitted = bool(
        (not planned["clipped"])
        and (not planned["wrong_way"])
        and planned["detour"]["meets_minimum"]
        and planned["frozen_endpoints"]["preserved"]
        and planned["continuous_after_densify"]
    )
    return {
        "admitted": admitted,
        "panel_side": panel_side,
        "direction": direction,
        "lane_y_m": float(route["inbound_lane_y_m"]),
        "padding_m": float(SLAB_PADDING_M),
        "rewrite_primitive": ENDPOINT_ONLY_PRIMITIVE,
        "clipped": bool(planned["clipped"]),
        "wrong_way": bool(planned["wrong_way"]),
        "min_abs_detour_m": float(planned["detour"]["min_abs_detour_m"]),
        "frozen_endpoint_preserved": bool(planned["frozen_endpoints"]["preserved"]),
        "continuous_after_densify": bool(planned["continuous_after_densify"]),
    }


def admit_six_cell_fixed_route(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    from pact_place_v10_route import stock_tcp_from_cell

    assembly = frozen_assembly()
    reports = []
    for cell in cells:
        side = str(cell["intrusion_side"])
        family = str(cell.get("family") or cell.get("layout_family_id") or "")
        for direction in ("inbound", "outbound"):
            positions, rotations = stock_tcp_from_cell(cell, direction)
            item = admit_fixed_route_on_stock(
                positions,
                rotations,
                panel_side=side,
                direction=direction,
                assembly=assembly,
            )
            item["family"] = family
            item["role_index"] = int(cell["role_index"])
            reports.append(item)
    admitted = sum(1 for item in reports if item["admitted"])
    if admitted != 12:
        raise RuntimeError(
            f"fixed endpoint-only route admitted {admitted}/12 cell×direction evaluations"
        )
    return reports

