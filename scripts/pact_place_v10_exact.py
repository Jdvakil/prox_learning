#!/usr/bin/env python3
"""Exact retained-qpos scoring for V10 compound assemblies.

AABB overlap is only a broad-phase screen. Necessity bits come from lobe GJK
contact. Stem or crossbar contact cannot satisfy the twelve necessity bits.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _search_path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_search_path) not in sys.path:
        sys.path.insert(0, str(_search_path))

from pact_geom_distance import CONTACT_DISTANCE_M, GeomShape
from pact_place_corridor_contract import PLACE_V5_SCENE_SHA256, sha256_file
from pact_place_v10_compound_pendant_contract import (
    MIN_NOMINAL_CLEARANCE_M,
    N_CLEAN_CELLS,
    N_NECESSITY_BITS,
    V5_SCENE_XML_RELATIVE,
    V99_RECONSTRUCTION_RELATIVE,
    V99_SNAPSHOT_RELATIVE,
    component_aabb,
)
from pact_place_v10_geometry import (
    NECESSITY_ALL_BITS,
    active_components,
    covers_all_necessity,
    necessity_bit,
    planning_probe_assembly,
    stem_for_lobe,
)
from pact_place_v99_exact import (
    load_clean_snapshots,
    scan_min_distance,
    verify_reconstruction_bundle,
)


def verify_v99_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    reconstruction_root = (ROOT / V99_RECONSTRUCTION_RELATIVE).parent
    snapshot_root = (ROOT / V99_SNAPSHOT_RELATIVE).parent
    reconstruction = verify_reconstruction_bundle(reconstruction_root)
    v5 = ROOT / V5_SCENE_XML_RELATIVE
    if sha256_file(v5) != PLACE_V5_SCENE_SHA256:
        raise ValueError("V5 scene XML does not match the frozen contract hash")
    snapshot_document, cells = load_clean_snapshots(
        snapshot_root, reconstruction=reconstruction
    )
    if len(cells) != N_CLEAN_CELLS:
        raise RuntimeError(f"expected {N_CLEAN_CELLS} snapshots, found {len(cells)}")
    return reconstruction, snapshot_document, cells


def _box_shape(center_m: Sequence[float], half_m: Sequence[float]) -> tuple[GeomShape, np.ndarray, np.ndarray]:
    lo, hi = component_aabb(center_m, half_m)
    box_lo = np.asarray(lo, dtype=np.float64)
    box_hi = np.asarray(hi, dtype=np.float64)
    center = 0.5 * (box_lo + box_hi)
    half = 0.5 * (box_hi - box_lo)
    return GeomShape.posed_axis_aligned_box(center, half), box_lo, box_hi


def _robot_kwargs(cell: dict[str, Any]) -> dict[str, Any]:
    return dict(
        gtype=cell["robot_gtype"],
        size=cell["robot_size"],
        pos=cell["robot_pos"],
        mat=cell["robot_mat"],
        verts=cell["robot_verts"],
        lo=cell["robot_lo"],
        hi=cell["robot_hi"],
    )


def _with_target(cell: dict[str, Any]) -> dict[str, Any]:
    kwargs = _robot_kwargs(cell)
    kwargs.update(
        extra_gtype=cell["target_gtype"],
        extra_size=cell["target_size"],
        extra_pos=cell["target_pos"],
        extra_mat=cell["target_mat"],
        extra_verts=cell["target_verts"],
        extra_lo=cell["target_lo"],
        extra_hi=cell["target_hi"],
    )
    return kwargs


def score_component_initial_target(
    component: dict[str, Any],
    cell: dict[str, Any],
) -> dict[str, Any]:
    """Initial-state target contact only. Does not rescore the robot lattice."""
    from pact_place_v99_exact import scan_min_distance

    box_shape, box_lo, box_hi = _box_shape(component["center_m"], component["half_m"])
    n_frames = int(np.asarray(cell["target_lo"]).shape[0])
    empty_gtype = np.zeros((0,), dtype=np.int32)
    empty_size = np.zeros((0, 3), dtype=np.float64)
    empty_pos = np.zeros((n_frames, 0, 3), dtype=np.float64)
    empty_mat = np.zeros((n_frames, 0, 9), dtype=np.float64)
    empty_lo = np.zeros((n_frames, 0, 3), dtype=np.float64)
    empty_hi = np.zeros((n_frames, 0, 3), dtype=np.float64)
    witness = scan_min_distance(
        gtype=empty_gtype,
        size=empty_size,
        pos=empty_pos,
        mat=empty_mat,
        verts=[],
        lo=empty_lo,
        hi=empty_hi,
        mask=cell["initial_mask"],
        box_shape=box_shape,
        box_lo=box_lo,
        box_hi=box_hi,
        extra_gtype=cell["target_gtype"],
        extra_size=cell["target_size"],
        extra_pos=cell["target_pos"],
        extra_mat=cell["target_mat"],
        extra_verts=cell["target_verts"],
        extra_lo=cell["target_lo"],
        extra_hi=cell["target_hi"],
        stop_at_contact=True,
    )
    contact = bool(
        witness.get("distance_m") is not None
        and float(witness["distance_m"]) <= CONTACT_DISTANCE_M
    )
    return {
        "initial_target_clear": not contact,
        "initial_clear": not contact,
        "witness": witness,
        "phase": "initial_target",
        "role_index": cell.get("role_index"),
    }


def score_unique_keys_initial_target(
    unique_keys: np.ndarray,
    cells: Sequence[dict[str, Any]],
    *,
    role: str,
) -> dict[str, Any]:
    packed = np.round(np.asarray(unique_keys, dtype=np.float64), 9)
    flags = np.ones(len(packed), dtype=bool)
    witnesses: list[dict[str, Any]] = []
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    for index, row in enumerate(packed):
        component = {
            "role": role,
            "center_m": row[:3].tolist(),
            "half_m": row[3:].tolist(),
        }
        per_cell = []
        clear_all = True
        for cell in cells:
            report = score_component_initial_target(component, cell)
            per_cell.append(report)
            if not report["initial_target_clear"]:
                clear_all = False
                witnesses.append(
                    {
                        "component_index": int(index),
                        "role": role,
                        "role_index": cell.get("role_index"),
                        "distance_m": (report["witness"] or {}).get("distance_m"),
                        "phase": "initial_target",
                    }
                )
        flags[index] = bool(clear_all)
        cache[tuple(float(value) for value in row)] = {
            "initial_target_clear_all": bool(clear_all),
            "per_cell": per_cell,
        }
        if (index + 1) % 250 == 0 or index + 1 == len(packed):
            print(
                f"[v10-initial-target-{role}] {index + 1}/{len(packed)}",
                flush=True,
            )
    return {
        "keys": packed,
        "initial_target_clear_all": flags,
        "cache": cache,
        "contact_witnesses": witnesses,
        "role": role,
    }


def score_component_on_cell(
    component: dict[str, Any],
    cell: dict[str, Any],
    *,
    min_clearance_m: float = MIN_NOMINAL_CLEARANCE_M,
) -> dict[str, Any]:
    box_shape, box_lo, box_hi = _box_shape(component["center_m"], component["half_m"])
    robot_only = dict(_robot_kwargs(cell), box_shape=box_shape, box_lo=box_lo, box_hi=box_hi)
    with_target = dict(_with_target(cell), box_shape=box_shape, box_lo=box_lo, box_hi=box_hi)
    inbound = scan_min_distance(
        **robot_only, mask=cell["inbound_mask"], stop_at_contact=True
    )
    outbound = scan_min_distance(
        **with_target, mask=cell["outbound_mask"], stop_at_contact=True
    )
    initial = scan_min_distance(
        **with_target, mask=cell["initial_mask"], stop_at_contact=True
    )
    grasp = scan_min_distance(
        **with_target,
        mask=cell["grasp_mask"],
        gap_limit_m=float(min_clearance_m),
        stop_at_contact=False,
    )
    inbound_contact = bool(
        inbound["distance_m"] is not None
        and float(inbound["distance_m"]) <= CONTACT_DISTANCE_M
    )
    outbound_contact = bool(
        outbound["distance_m"] is not None
        and float(outbound["distance_m"]) <= CONTACT_DISTANCE_M
    )
    initial_contact = bool(
        initial["distance_m"] is not None
        and float(initial["distance_m"]) <= CONTACT_DISTANCE_M
    )
    if grasp["distance_m"] is None:
        grasp_clear = True
        grasp_margin = None
    else:
        grasp_clear = bool(float(grasp["distance_m"]) + 1e-12 >= float(min_clearance_m))
        grasp_margin = float(grasp["distance_m"]) - float(min_clearance_m)
    return {
        "role": component["role"],
        "name": component.get("name"),
        "slot": component.get("slot"),
        "inbound_contact": inbound_contact,
        "outbound_contact": outbound_contact,
        "initial_clear": not initial_contact,
        "grasp_clear": grasp_clear,
        "inbound_witness": inbound,
        "outbound_witness": outbound,
        "initial_witness": initial,
        "grasp_witness": grasp,
        "grasp_clearance_margin_m": grasp_margin,
    }


def lobe_necessity_bits(
    lobe: dict[str, Any],
    cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Score inbound/outbound lobe contact on every cell. No early termination."""
    bits = 0
    per_cell = []
    for index, cell in enumerate(cells):
        report = score_component_on_cell(lobe, cell)
        if report["inbound_contact"]:
            bits |= necessity_bit(index, False)
        if report["outbound_contact"]:
            bits |= necessity_bit(index, True)
        per_cell.append(report)
    return {
        "key": list(lobe["key"]),
        "side": lobe["side"],
        "center_m": list(lobe["center_m"]),
        "half_m": list(lobe["half_m"]),
        "bits": int(bits),
        "covers_all": bits == NECESSITY_ALL_BITS,
        "per_cell": per_cell,
        "grasp_clear_all": all(item["grasp_clear"] for item in per_cell),
        "initial_clear_all": all(item["initial_clear"] for item in per_cell),
        "min_grasp_clearance_margin_m": _min_margin(per_cell),
    }


def _min_margin(rows: Sequence[dict[str, Any]]) -> float | None:
    margins = [
        float(item["grasp_clearance_margin_m"])
        for item in rows
        if item.get("grasp_clearance_margin_m") is not None
    ]
    if not margins:
        return None
    return float(min(margins))


def compact_clearance_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "grasp_clear_all": all(bool(item["grasp_clear"]) for item in rows),
        "initial_clear_all": all(bool(item["initial_clear"]) for item in rows),
        "min_grasp_clearance_margin_m": _min_margin(rows),
    }


def component_clearance_summary(
    component: dict[str, Any],
    cells: Sequence[dict[str, Any]],
    *,
    min_clearance_m: float = MIN_NOMINAL_CLEARANCE_M,
) -> dict[str, Any]:
    per_cell = [
        score_component_on_cell(component, cell, min_clearance_m=min_clearance_m)
        for cell in cells
    ]
    return compact_clearance_summary(per_cell)


def compact_lobe_score(scored: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": list(scored["key"]),
        "side": scored["side"],
        "center_m": list(scored["center_m"]),
        "half_m": list(scored["half_m"]),
        "bits": int(scored["bits"]),
        "grasp_clear_all": bool(scored["grasp_clear_all"]),
        "initial_clear_all": bool(scored["initial_clear_all"]),
        "min_grasp_clearance_margin_m": scored.get("min_grasp_clearance_margin_m"),
    }


def _score_lobe_chunk(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker: AABB then exact lobe/stem scores. Skip GJK when AABB bits are 0."""
    cells = payload["cells"]
    worker_id = int(payload.get("worker_id", 0))
    score_stems = bool(payload.get("score_stems", True))
    min_clearance_m = float(payload.get("min_clearance_m", MIN_NOMINAL_CLEARANCE_M))
    lobes = list(payload["lobes"])
    rows: list[dict[str, Any]] = []
    for index, lobe in enumerate(lobes):
        aabb_bits = aabb_lobe_bits(lobe, cells)
        exact = None
        stem = None
        if aabb_bits:
            scored = lobe_necessity_bits(lobe, cells)
            exact = compact_lobe_score(scored)
            if (
                score_stems
                and exact["grasp_clear_all"]
                and exact["initial_clear_all"]
            ):
                try:
                    stem_geom = stem_for_lobe(lobe)
                except ValueError:
                    stem = {
                        "grasp_clear_all": False,
                        "initial_clear_all": False,
                        "min_grasp_clearance_margin_m": None,
                    }
                else:
                    stem = component_clearance_summary(
                        stem_geom, cells, min_clearance_m=min_clearance_m
                    )
        rows.append(
            {
                "key": list(lobe["key"]),
                "aabb_bits": int(aabb_bits),
                "exact": exact,
                "stem": stem,
            }
        )
        if index == 0 or index + 1 == len(lobes) or (index + 1) % 250 == 0:
            print(
                f"[v10-exact-worker {worker_id}] {index + 1}/{len(lobes)}",
                flush=True,
            )
    return {"rows": rows}


def score_lattice_parallel(
    lobes: Sequence[dict[str, Any]],
    cells: Sequence[dict[str, Any]],
    *,
    workers: int = 8,
    min_clearance_m: float = MIN_NOMINAL_CLEARANCE_M,
    score_stems: bool = True,
) -> tuple[dict[tuple[float, ...], int], dict[tuple[float, ...], dict[str, Any]], dict[tuple[float, ...], dict[str, Any]]]:
    """Exact-score every AABB-eligible lobe (and its stem) across workers."""
    if not lobes:
        return {}, {}, {}
    n_workers = max(1, min(int(workers), len(lobes)))
    chunks = [list(lobes[index::n_workers]) for index in range(n_workers)]
    payloads = [
        {
            "lobes": chunk,
            "cells": cells,
            "min_clearance_m": min_clearance_m,
            "score_stems": score_stems,
            "worker_id": index,
        }
        for index, chunk in enumerate(chunks)
        if chunk
    ]
    if n_workers == 1:
        results = [_score_lobe_chunk(payloads[0])]
    else:
        context = multiprocessing.get_context("spawn")
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, mp_context=context
        ) as pool:
            futures = [pool.submit(_score_lobe_chunk, payload) for payload in payloads]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    aabb_bits: dict[tuple[float, ...], int] = {}
    exact_cache: dict[tuple[float, ...], dict[str, Any]] = {}
    stem_cache: dict[tuple[float, ...], dict[str, Any]] = {}
    for result in results:
        for row in result["rows"]:
            key = tuple(row["key"])
            aabb_bits[key] = int(row["aabb_bits"])
            if row["exact"] is not None:
                exact_cache[key] = row["exact"]
            if row["stem"] is not None:
                stem_cache[key] = row["stem"]
    return aabb_bits, exact_cache, stem_cache


def crossbar_cache_key(component: dict[str, Any]) -> tuple[float, ...]:
    return tuple(round(float(value), 9) for value in list(component["center_m"]) + list(component["half_m"]))


def evaluate_assembly_from_component_caches(
    assembly: dict[str, Any],
    *,
    lobe_scores: dict[tuple[float, ...], dict[str, Any]],
    stem_scores: dict[tuple[float, ...], dict[str, Any]],
    crossbar_score: dict[str, Any],
) -> dict[str, Any]:
    """Combine cached lobe/stem scores with a scored crossbar. No GJK here."""
    components = active_components(assembly)
    lobes = [item for item in components if item["role"] == "lobe"]
    stems = [item for item in components if item["role"] == "stem"]
    lobe_bits = 0
    grasp_ok = bool(crossbar_score["grasp_clear_all"])
    initial_ok = bool(crossbar_score["initial_clear_all"])
    margins: list[float] = []
    bar_margin = crossbar_score.get("min_grasp_clearance_margin_m")
    if bar_margin is not None:
        margins.append(float(bar_margin))
    for lobe in lobes:
        scored = lobe_scores[tuple(lobe["key"])]
        lobe_bits |= int(scored["bits"])
        grasp_ok = grasp_ok and bool(scored["grasp_clear_all"])
        initial_ok = initial_ok and bool(scored["initial_clear_all"])
        margin = scored.get("min_grasp_clearance_margin_m")
        if margin is not None:
            margins.append(float(margin))
    for stem in stems:
        lobe = next(item for item in lobes if item.get("slot") == stem.get("slot"))
        scored = stem_scores[tuple(lobe["key"])]
        grasp_ok = grasp_ok and bool(scored["grasp_clear_all"])
        initial_ok = initial_ok and bool(scored["initial_clear_all"])
        margin = scored.get("min_grasp_clearance_margin_m")
        if margin is not None:
            margins.append(float(margin))
    necessity_ok = covers_all_necessity([lobe_bits])
    return {
        "assembly_id": assembly["assembly_id"],
        "topology": assembly["topology"],
        "lobe_necessity_bits": int(lobe_bits),
        "lobe_necessity_ok": bool(necessity_ok),
        "n_lobes": len(lobes),
        "grasp_window_clear": bool(grasp_ok),
        "initial_state_clear": bool(initial_ok),
        "min_grasp_clearance_margin_m": None if not margins else float(min(margins)),
        "accepted": bool(necessity_ok and grasp_ok and initial_ok),
        "volume_m3": assembly["volume_m3"],
        "stem_or_crossbar_counted_as_necessity": False,
    }


def compact_survivor_record(assembly: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    lobes = [item for item in active_components(assembly) if item["role"] == "lobe"]
    return {
        "assembly_id": assembly["assembly_id"],
        "topology": assembly["topology"],
        "volume_m3": assembly["volume_m3"],
        "lobe_necessity_bits": int(report["lobe_necessity_bits"]),
        "min_grasp_clearance_margin_m": report.get("min_grasp_clearance_margin_m"),
        "lobe_keys": [list(item["key"]) for item in lobes],
    }


def write_survivor_catalog(path: Path, survivors: Sequence[dict[str, Any]]) -> str:
    ordered = sorted(survivors, key=lambda item: str(item.get("assembly_id", "")))
    n_rows = len(ordered)
    keys = np.full((n_rows, 3, 6), np.nan, dtype=np.float64)
    n_lobes = np.zeros(n_rows, dtype=np.int32)
    volume = np.zeros(n_rows, dtype=np.float64)
    bits = np.zeros(n_rows, dtype=np.int32)
    margin = np.full(n_rows, np.nan, dtype=np.float64)
    topology = np.asarray([str(item.get("topology", "")) for item in ordered])
    assembly_ids = np.asarray([str(item.get("assembly_id", "")) for item in ordered])
    for index, item in enumerate(ordered):
        lobe_keys = list(item.get("lobe_keys") or [])
        n_lobes[index] = len(lobe_keys)
        for slot, key in enumerate(lobe_keys[:3]):
            keys[index, slot, :] = np.asarray(key, dtype=np.float64)
        volume[index] = float(item.get("volume_m3") or 0.0)
        bits[index] = int(item.get("lobe_necessity_bits") or 0)
        if item.get("min_grasp_clearance_margin_m") is not None:
            margin[index] = float(item["min_grasp_clearance_margin_m"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        assembly_ids=assembly_ids,
        topology=topology,
        n_lobes=n_lobes,
        lobe_keys=keys,
        volume_m3=volume,
        lobe_necessity_bits=bits,
        min_grasp_clearance_margin_m=margin,
    )
    return sha256_file(path)


def aabb_lobe_bits(lobe: dict[str, Any], cells: Sequence[dict[str, Any]]) -> int:
    from pact_place_v99_geometry import geoms_intersect_box

    box_lo, box_hi = component_aabb(lobe["center_m"], lobe["half_m"])
    bits = 0
    for index, cell in enumerate(cells):
        inbound = bool(
            np.any(
                geoms_intersect_box(cell["robot_lo"], cell["robot_hi"], box_lo, box_hi)[
                    cell["inbound_mask"]
                ]
            )
        )
        outbound = bool(
            np.any(
                geoms_intersect_box(cell["robot_lo"], cell["robot_hi"], box_lo, box_hi)[
                    cell["outbound_mask"]
                ]
            )
        )
        target_lo = cell.get("target_lo")
        target_hi = cell.get("target_hi")
        if target_lo is not None and target_hi is not None and np.asarray(target_lo).size:
            outbound = outbound or bool(
                np.any(
                    geoms_intersect_box(target_lo, target_hi, box_lo, box_hi)[
                        cell["outbound_mask"]
                    ]
                )
            )
        if inbound:
            bits |= necessity_bit(index, False)
        if outbound:
            bits |= necessity_bit(index, True)
    return int(bits)


def evaluate_assembly_exact(
    assembly: dict[str, Any],
    cells: Sequence[dict[str, Any]],
    *,
    min_clearance_m: float = MIN_NOMINAL_CLEARANCE_M,
) -> dict[str, Any]:
    """All four predicates on all six cells, every active component. No early exit."""
    components = active_components(assembly)
    lobes = [item for item in components if item["role"] == "lobe"]
    per_component = []
    lobe_bits = 0
    for component in components:
        cell_reports = [
            score_component_on_cell(component, cell, min_clearance_m=min_clearance_m)
            for cell in cells
        ]
        per_component.append(
            {
                "name": component["name"],
                "role": component["role"],
                "slot": component.get("slot"),
                "per_cell": cell_reports,
                "grasp_clear_all": all(item["grasp_clear"] for item in cell_reports),
                "initial_clear_all": all(item["initial_clear"] for item in cell_reports),
            }
        )
        if component["role"] == "lobe":
            bits = 0
            for index, report in enumerate(cell_reports):
                if report["inbound_contact"]:
                    bits |= necessity_bit(index, False)
                if report["outbound_contact"]:
                    bits |= necessity_bit(index, True)
            lobe_bits |= bits
            per_component[-1]["bits"] = bits
    grasp_ok = all(item["grasp_clear_all"] for item in per_component)
    initial_ok = all(item["initial_clear_all"] for item in per_component)
    necessity_ok = covers_all_necessity([lobe_bits])
    margins = [
        report["grasp_clearance_margin_m"]
        for item in per_component
        for report in item["per_cell"]
        if report["grasp_clearance_margin_m"] is not None
    ]
    return {
        "assembly_id": assembly["assembly_id"],
        "topology": assembly["topology"],
        "lobe_necessity_bits": int(lobe_bits),
        "lobe_necessity_ok": bool(necessity_ok),
        "n_lobes": len(lobes),
        "grasp_window_clear": bool(grasp_ok),
        "initial_state_clear": bool(initial_ok),
        "min_grasp_clearance_margin_m": None if not margins else float(min(margins)),
        "accepted": bool(necessity_ok and grasp_ok and initial_ok),
        "per_component": per_component,
        "volume_m3": assembly["volume_m3"],
        "stem_or_crossbar_counted_as_necessity": False,
    }


def evaluate_planning_probe(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    assembly = planning_probe_assembly()
    report = evaluate_assembly_exact(assembly, cells)
    report["assembly"] = assembly
    report["reproduced_probe"] = bool(
        report["lobe_necessity_ok"] and report["grasp_window_clear"]
    )
    report["n_necessity_bits"] = N_NECESSITY_BITS
    return report


def write_component_witnesses(
    path: Path,
    reports: Sequence[dict[str, Any]],
    *,
    role_indices: Sequence[int],
) -> str:
    """Deterministic NPZ of component-level contact and clearance witnesses."""
    ordered = sorted(reports, key=lambda item: str(item.get("assembly_id", "")))
    assembly_ids = np.asarray([str(item["assembly_id"]) for item in ordered])
    topologies = np.asarray([str(item.get("topology", "")) for item in ordered])
    lobe_bits = np.asarray(
        [int(item.get("lobe_necessity_bits", 0)) for item in ordered], dtype=np.int32
    )
    accepted = np.asarray(
        [bool(item.get("accepted", False)) for item in ordered], dtype=np.bool_
    )
    n_components = max((len(item.get("per_component") or []) for item in ordered), default=0)
    n_cells = len(role_indices)
    inbound = np.zeros((len(ordered), n_components, n_cells), dtype=np.bool_)
    outbound = np.zeros((len(ordered), n_components, n_cells), dtype=np.bool_)
    grasp_margin = np.full((len(ordered), n_components, n_cells), np.nan, dtype=np.float64)
    roles = np.full((len(ordered), n_components), "", dtype=object)
    names = np.full((len(ordered), n_components), "", dtype=object)
    for row_index, report in enumerate(ordered):
        for component_index, component in enumerate(report.get("per_component") or []):
            roles[row_index, component_index] = str(component.get("role", ""))
            names[row_index, component_index] = str(component.get("name", ""))
            for cell_index, cell in enumerate(component.get("per_cell") or []):
                inbound[row_index, component_index, cell_index] = bool(
                    cell.get("inbound_contact")
                )
                outbound[row_index, component_index, cell_index] = bool(
                    cell.get("outbound_contact")
                )
                margin = cell.get("grasp_clearance_margin_m")
                if margin is not None:
                    grasp_margin[row_index, component_index, cell_index] = float(margin)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        assembly_ids=assembly_ids,
        topologies=topologies,
        lobe_necessity_bits=lobe_bits,
        accepted=accepted,
        inbound_contact=inbound,
        outbound_contact=outbound,
        grasp_clearance_margin_m=grasp_margin,
        roles=roles,
        names=names,
        role_indices=np.asarray(role_indices, dtype=np.int32),
        stem_or_crossbar_counted_as_necessity=np.asarray(False),
    )
    return sha256_file(path)


def live_component_role_parity_cases() -> dict[str, Any]:
    """GJK / true_distance parity for lobe, stem, and crossbar boxes."""
    import mujoco
    from pact_geom_distance import CONTACT_DISTANCE_M, GeomShape, gjk_distance, true_distance
    from pact_place_v99_exact import LIVE_PARITY_ATOL_M, NEAR_THRESHOLD_M

    sizes = {
        "lobe": (0.02, 0.04, 0.04),
        "stem": (0.003, 0.003, 0.25),
        "crossbar": (0.003, 0.28, 0.005),
    }
    robot_half = 0.04
    report: dict[str, Any] = {"parity_ok": True, "roles": {}}
    for role, half in sizes.items():
        separated_x = robot_half + 0.080 + half[0]
        near_x = robot_half + NEAR_THRESHOLD_M + half[0]
        xml = f"""
        <mujoco>
          <worldbody>
            <body name="robot" pos="0 0 0">
              <geom name="robot_g" type="box" size="{robot_half} {robot_half} {robot_half}"
                    contype="1" conaffinity="1"/>
            </body>
            <body name="fixture" pos="{separated_x} 0 0">
              <geom name="fixture_g" type="box" size="{half[0]} {half[1]} {half[2]}"
                    contype="1" conaffinity="1"/>
            </body>
            <body name="nested" pos="0 0 0">
              <geom name="nested_g" type="box" size="0.01 0.01 0.01"
                    contype="1" conaffinity="1"/>
            </body>
            <body name="near" pos="{near_x} 0 0">
              <geom name="near_g" type="box" size="{half[0]} {half[1]} {half[2]}"
                    contype="1" conaffinity="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        def _pair(left_name: str, right_name: str) -> dict[str, float]:
            left = int(model.geom(left_name).id)
            right = int(model.geom(right_name).id)
            live = float(true_distance(model, data, [left], [right]))
            snapshot = float(
                gjk_distance(
                    GeomShape.from_mujoco(model, data, left),
                    GeomShape.from_mujoco(model, data, right),
                )
            )
            return {
                "true_distance_m": live,
                "gjk_m": snapshot,
                "abs_delta_m": abs(live - snapshot),
            }

        nested = _pair("robot_g", "nested_g")
        separated = _pair("robot_g", "fixture_g")
        near = _pair("robot_g", "near_g")
        role_ok = bool(
            nested["abs_delta_m"] <= LIVE_PARITY_ATOL_M
            and separated["abs_delta_m"] <= 1e-5
            and near["abs_delta_m"] <= 1e-5
            and nested["true_distance_m"] <= CONTACT_DISTANCE_M
        )
        report["roles"][role] = {
            "nested": nested,
            "separated": separated,
            "near": near,
            "parity_ok": role_ok,
        }
        report["parity_ok"] = bool(report["parity_ok"] and role_ok)
    return report

