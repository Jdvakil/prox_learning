#!/usr/bin/env python3
"""Exact retained-qpos geometry filter for the V9.9 ceiling pendant.

AABB overlap is only a broad-phase screen. This module restores reconstructed
collision geoms with mj_forward only (no env.step, no episodes) and scores
candidates with the hardened GJK instrument from pact_geom_distance.

A candidate survives only if every clean cell has:
  * actual inbound stock-route contact (GJK distance 0),
  * actual outbound stock-route contact, including the carried target,
  * at least 25 mm exact robot-and-target clearance on the pregrasp / grasp /
    close / lift window,
  * no exact initial-state contact.

Close-out scoring evaluates all four predicates on all six cells. It does not
stop at the first failing cell or predicate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_geom_distance import (  # noqa: E402
    CONTACT_DISTANCE_M,
    GeomShape,
    gjk_distance,
    true_distance,
)
from pact_place_corridor_contract import (  # noqa: E402
    PLACE_V5_SCENE_SHA256,
    sha256_file,
    sha256_payload,
)
from pact_place_v99_geometry import fixture_key  # noqa: E402
from pact_place_v99_pendant_contract import (  # noqa: E402
    MIN_NOMINAL_CLEARANCE_M,
    N_CLEAN_CELLS,
    PENDANT_BODY,
    PENDANT_GEOM,
    PHYSICS_CLEAN_FAMILIES,
    SCENE_XML_RELATIVE,
    pendant_aabb,
    pendant_volume_m3,
)
from reconstruct_pact_place_v99_baseline import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_RECONSTRUCTION,
    IMPLEMENTATION_PATHS,
    SOURCE_SUMMARY,
    reconstruct_one_row,
    write_immutable,
)

DEFAULT_SNAPSHOT_ROOT = ROOT / "diagnostics_output/pact_place_v99_siting/snapshots"
SCOPED_CONCLUSION = (
    "no survivor in the registered fixed rectangular-box lattice"
)
SNAPSHOT_SCHEMA = "pact_place_v9_9_exact_snapshots_v2"
MISSING_INDEX = -1
LIVE_PARITY_ATOL_M = 1e-6
NEAR_THRESHOLD_M = MIN_NOMINAL_CLEARANCE_M


def unpack_mesh_verts(
    start: np.ndarray, count: np.ndarray, verts: np.ndarray
) -> list[np.ndarray | None]:
    packed = (
        np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        if np.asarray(verts).size
        else np.zeros((0, 3), dtype=np.float64)
    )
    out: list[np.ndarray | None] = []
    for mesh_start, mesh_count in zip(np.asarray(start), np.asarray(count)):
        if int(mesh_start) < 0 or int(mesh_count) <= 0:
            out.append(None)
        else:
            lo = int(mesh_start)
            hi = lo + int(mesh_count)
            out.append(np.asarray(packed[lo:hi], dtype=np.float64))
    return out


def aabb_gaps_to_box(
    lo: np.ndarray, hi: np.ndarray, box_lo: np.ndarray, box_hi: np.ndarray
) -> np.ndarray:
    lo_v = np.asarray(lo, dtype=np.float64)
    hi_v = np.asarray(hi, dtype=np.float64)
    delta = np.maximum(0.0, np.maximum(box_lo - hi_v, lo_v - box_hi))
    gap = np.linalg.norm(delta, axis=-1)
    overlap = np.all(lo_v <= box_hi, axis=-1) & np.all(box_lo <= hi_v, axis=-1)
    return np.where(overlap, 0.0, gap)


def _shape_at(
    *,
    gtype: np.ndarray,
    size: np.ndarray,
    pos: np.ndarray,
    mat: np.ndarray,
    verts: Sequence[np.ndarray | None],
    frame: int,
    geom: int,
) -> GeomShape:
    return GeomShape(
        int(gtype[geom]),
        np.asarray(pos[frame, geom], dtype=np.float64),
        np.asarray(mat[frame, geom], dtype=np.float64),
        np.asarray(size[geom], dtype=np.float64),
        verts[geom],
    )


def _empty_witness(*, aabb_disproof: bool) -> dict[str, Any]:
    return {
        "frame": None,
        "geom_index": None,
        "geom_group": None,
        "distance_m": None,
        "aabb_disproof": bool(aabb_disproof),
    }


def scan_min_distance(
    *,
    gtype: np.ndarray,
    size: np.ndarray,
    pos: np.ndarray,
    mat: np.ndarray,
    verts: Sequence[np.ndarray | None],
    lo: np.ndarray,
    hi: np.ndarray,
    mask: np.ndarray,
    box_shape: GeomShape,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    extra_gtype: np.ndarray | None = None,
    extra_size: np.ndarray | None = None,
    extra_pos: np.ndarray | None = None,
    extra_mat: np.ndarray | None = None,
    extra_verts: Sequence[np.ndarray | None] | None = None,
    extra_lo: np.ndarray | None = None,
    extra_hi: np.ndarray | None = None,
    gap_limit_m: float | None = None,
    stop_at_contact: bool = False,
) -> dict[str, Any]:
    """Minimum GJK distance over AABB-eligible pairs. Never returns an AABB gap."""
    groups = [
        ("robot", gtype, size, pos, mat, verts, lo, hi),
    ]
    if extra_gtype is not None and extra_lo is not None and extra_hi is not None:
        groups.append(
            (
                "target",
                extra_gtype,
                extra_size,
                extra_pos,
                extra_mat,
                extra_verts or [],
                extra_lo,
                extra_hi,
            )
        )
    frame_idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if frame_idx.size == 0:
        return _empty_witness(aabb_disproof=True)
    best = float("inf")
    witness: dict[str, Any] | None = None
    measured = False
    limit = float("inf") if gap_limit_m is None else float(gap_limit_m)
    for group, gtype_g, size_g, pos_g, mat_g, verts_g, lo_g, hi_g in groups:
        if lo_g is None or np.asarray(lo_g).size == 0:
            continue
        gaps = aabb_gaps_to_box(lo_g[frame_idx], hi_g[frame_idx], box_lo, box_hi)
        if gap_limit_m is None:
            local_f, geom_i = np.nonzero(gaps <= CONTACT_DISTANCE_M)
        else:
            local_f, geom_i = np.nonzero(gaps < min(limit, best))
        for local, geom in zip(local_f.tolist(), geom_i.tolist()):
            gap = float(gaps[int(local), int(geom)])
            if gap >= best:
                continue
            if gap_limit_m is None and gap > CONTACT_DISTANCE_M:
                continue
            frame = int(frame_idx[int(local)])
            if int(geom) >= int(gtype_g.shape[0]):
                continue
            shape = _shape_at(
                gtype=gtype_g,
                size=size_g,
                pos=pos_g,
                mat=mat_g,
                verts=verts_g,
                frame=frame,
                geom=int(geom),
            )
            if not shape.supported:
                continue
            distance = float(gjk_distance(shape, box_shape))
            measured = True
            if distance < best:
                best = distance
                witness = {
                    "frame": frame,
                    "geom_index": int(geom),
                    "geom_group": group,
                    "distance_m": distance,
                    "aabb_disproof": False,
                }
            if best <= CONTACT_DISTANCE_M:
                return witness
    if not measured:
        return _empty_witness(aabb_disproof=True)
    return witness or _empty_witness(aabb_disproof=False)


def evaluate_cell_predicates(
    cell: dict[str, Any],
    *,
    box_shape: GeomShape,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    min_clearance_m: float,
) -> dict[str, Any]:
    """Score all four predicates on one cell. No early termination."""
    robot_only = dict(
        gtype=cell["robot_gtype"],
        size=cell["robot_size"],
        pos=cell["robot_pos"],
        mat=cell["robot_mat"],
        verts=cell["robot_verts"],
        lo=cell["robot_lo"],
        hi=cell["robot_hi"],
        box_shape=box_shape,
        box_lo=box_lo,
        box_hi=box_hi,
    )
    with_target = dict(
        **robot_only,
        extra_gtype=cell["target_gtype"],
        extra_size=cell["target_size"],
        extra_pos=cell["target_pos"],
        extra_mat=cell["target_mat"],
        extra_verts=cell["target_verts"],
        extra_lo=cell["target_lo"],
        extra_hi=cell["target_hi"],
    )
    inbound = scan_min_distance(
        **robot_only,
        mask=cell["inbound_mask"],
        stop_at_contact=True,
    )
    outbound = scan_min_distance(
        **with_target, mask=cell["outbound_mask"], stop_at_contact=True
    )
    initial = scan_min_distance(
        **robot_only,
        mask=cell["initial_mask"],
        stop_at_contact=True,
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
        grasp_clear = bool(
            float(grasp["distance_m"]) + 1e-12 >= float(min_clearance_m)
        )
        grasp_margin = float(grasp["distance_m"]) - float(min_clearance_m)
    return {
        "role_index": cell.get("role_index"),
        "inbound_stock_contact": inbound_contact,
        "outbound_stock_contact": outbound_contact,
        "initial_state_clear": not initial_contact,
        "grasp_window_clear": grasp_clear,
        "inbound_witness": inbound,
        "outbound_witness": outbound,
        "initial_witness": initial,
        "grasp_witness": grasp,
        "grasp_clearance_margin_m": grasp_margin,
        "accepted": bool(
            inbound_contact
            and outbound_contact
            and (not initial_contact)
            and grasp_clear
        ),
    }


def evaluate_fixture_exact(
    fixture: dict[str, Any],
    cells: Sequence[dict[str, Any]],
    *,
    min_clearance_m: float = MIN_NOMINAL_CLEARANCE_M,
) -> dict[str, Any]:
    box_lo, box_hi = (np.asarray(item, dtype=np.float64) for item in pendant_aabb(fixture))
    center = 0.5 * (box_lo + box_hi)
    half = 0.5 * (box_hi - box_lo)
    box_shape = GeomShape.posed_axis_aligned_box(center, half)
    per_cell = [
        evaluate_cell_predicates(
            cell,
            box_shape=box_shape,
            box_lo=box_lo,
            box_hi=box_hi,
            min_clearance_m=min_clearance_m,
        )
        for cell in cells
    ]
    inbound_ok = all(item["inbound_stock_contact"] for item in per_cell)
    outbound_ok = all(item["outbound_stock_contact"] for item in per_cell)
    initial_ok = all(item["initial_state_clear"] for item in per_cell)
    grasp_ok = all(item["grasp_window_clear"] for item in per_cell)
    grasp_distances = [
        item["grasp_witness"]["distance_m"]
        for item in per_cell
        if item["grasp_witness"]["distance_m"] is not None
    ]
    return {
        "inbound_stock_contact": inbound_ok,
        "outbound_stock_contact": outbound_ok,
        "initial_state_clear": initial_ok,
        "grasp_window_clear": grasp_ok,
        "min_grasp_window_exact_clearance_m": (
            None if not grasp_distances else float(min(grasp_distances))
        ),
        "accepted": bool(inbound_ok and outbound_ok and initial_ok and grasp_ok),
        "volume_m3": pendant_volume_m3(fixture),
        "fixture": fixture,
        "key": fixture_key(fixture),
        "per_cell": per_cell,
    }


def _witness_to_arrays(report: dict[str, Any], n_cells: int) -> dict[str, np.ndarray]:
    inbound_frame = np.full(n_cells, MISSING_INDEX, dtype=np.int32)
    inbound_geom = np.full(n_cells, MISSING_INDEX, dtype=np.int32)
    inbound_group = np.full(n_cells, MISSING_INDEX, dtype=np.int32)
    inbound_distance = np.full(n_cells, np.inf, dtype=np.float64)
    outbound_frame = inbound_frame.copy()
    outbound_geom = inbound_geom.copy()
    outbound_group = inbound_group.copy()
    outbound_distance = inbound_distance.copy()
    initial_frame = inbound_frame.copy()
    initial_geom = inbound_geom.copy()
    initial_group = inbound_group.copy()
    initial_distance = inbound_distance.copy()
    grasp_frame = inbound_frame.copy()
    grasp_geom = inbound_geom.copy()
    grasp_group = inbound_group.copy()
    grasp_distance = inbound_distance.copy()
    grasp_margin = np.full(n_cells, np.nan, dtype=np.float64)
    inbound_ok = np.zeros(n_cells, dtype=np.bool_)
    outbound_ok = np.zeros(n_cells, dtype=np.bool_)
    initial_ok = np.zeros(n_cells, dtype=np.bool_)
    grasp_ok = np.zeros(n_cells, dtype=np.bool_)
    group_code = {"robot": 0, "target": 1}

    def _fill(prefix: str, witness: dict[str, Any], index: int) -> None:
        if witness.get("frame") is None:
            return
        mapping = {
            "inbound": (inbound_frame, inbound_geom, inbound_group, inbound_distance),
            "outbound": (outbound_frame, outbound_geom, outbound_group, outbound_distance),
            "initial": (initial_frame, initial_geom, initial_group, initial_distance),
            "grasp": (grasp_frame, grasp_geom, grasp_group, grasp_distance),
        }
        frames, geoms, groups, distances = mapping[prefix]
        frames[index] = int(witness["frame"])
        geoms[index] = int(witness["geom_index"])
        groups[index] = int(group_code.get(str(witness.get("geom_group")), MISSING_INDEX))
        distances[index] = float(witness["distance_m"])

    for index, cell in enumerate(report["per_cell"]):
        _fill("inbound", cell["inbound_witness"], index)
        _fill("outbound", cell["outbound_witness"], index)
        _fill("initial", cell["initial_witness"], index)
        _fill("grasp", cell["grasp_witness"], index)
        inbound_ok[index] = cell["inbound_stock_contact"]
        outbound_ok[index] = cell["outbound_stock_contact"]
        initial_ok[index] = cell["initial_state_clear"]
        grasp_ok[index] = cell["grasp_window_clear"]
        if cell["grasp_clearance_margin_m"] is not None:
            grasp_margin[index] = float(cell["grasp_clearance_margin_m"])
    return {
        "inbound_frame": inbound_frame,
        "inbound_geom": inbound_geom,
        "inbound_group": inbound_group,
        "inbound_distance_m": inbound_distance,
        "outbound_frame": outbound_frame,
        "outbound_geom": outbound_geom,
        "outbound_group": outbound_group,
        "outbound_distance_m": outbound_distance,
        "initial_frame": initial_frame,
        "initial_geom": initial_geom,
        "initial_group": initial_group,
        "initial_distance_m": initial_distance,
        "grasp_frame": grasp_frame,
        "grasp_geom": grasp_geom,
        "grasp_group": grasp_group,
        "grasp_distance_m": grasp_distance,
        "grasp_clearance_margin_m": grasp_margin,
        "inbound_ok": inbound_ok,
        "outbound_ok": outbound_ok,
        "initial_ok": initial_ok,
        "grasp_ok": grasp_ok,
    }


def _eval_chunk(payload: dict[str, Any]) -> dict[str, Any]:
    cells = payload["cells"]
    min_clearance_m = float(payload["min_clearance_m"])
    worker_id = int(payload.get("worker_id", 0))
    fixtures = list(payload["fixtures"])
    n_fixtures = len(fixtures)
    n_cells = len(cells)
    survivors = []
    n_inbound_fail = n_outbound_fail = n_grasp_fail = n_initial_fail = 0
    packed: dict[str, np.ndarray] = {
        "center_m": np.zeros((n_fixtures, 3), dtype=np.float64),
        "half_m": np.zeros((n_fixtures, 3), dtype=np.float64),
        "accepted": np.zeros(n_fixtures, dtype=np.bool_),
    }
    template = _witness_to_arrays(
        {
            "per_cell": [
                {
                    "inbound_witness": _empty_witness(aabb_disproof=True),
                    "outbound_witness": _empty_witness(aabb_disproof=True),
                    "initial_witness": _empty_witness(aabb_disproof=True),
                    "grasp_witness": _empty_witness(aabb_disproof=True),
                    "inbound_stock_contact": False,
                    "outbound_stock_contact": False,
                    "initial_state_clear": True,
                    "grasp_window_clear": True,
                    "grasp_clearance_margin_m": None,
                }
                for _ in range(n_cells)
            ]
        },
        n_cells,
    )
    for key, array in template.items():
        packed[key] = np.zeros((n_fixtures,) + array.shape, dtype=array.dtype)

    for index, fixture in enumerate(fixtures):
        report = evaluate_fixture_exact(
            fixture, cells, min_clearance_m=min_clearance_m
        )
        packed["center_m"][index] = np.asarray(fixture["center_m"], dtype=np.float64)
        packed["half_m"][index] = np.asarray(fixture["half_m"], dtype=np.float64)
        packed["accepted"][index] = report["accepted"]
        arrays = _witness_to_arrays(report, n_cells)
        for key, array in arrays.items():
            packed[key][index] = array
        if not report["inbound_stock_contact"]:
            n_inbound_fail += 1
        if not report["outbound_stock_contact"]:
            n_outbound_fail += 1
        if not report["grasp_window_clear"]:
            n_grasp_fail += 1
        if not report["initial_state_clear"]:
            n_initial_fail += 1
        if report["accepted"]:
            survivors.append(
                {
                    "fixture": report["fixture"],
                    "key": list(report["key"]),
                    "volume_m3": report["volume_m3"],
                    "min_grasp_window_exact_clearance_m": report[
                        "min_grasp_window_exact_clearance_m"
                    ],
                    "per_cell": report["per_cell"],
                }
            )
        if index == 0 or index + 1 == n_fixtures or (index + 1) % 250 == 0:
            print(
                f"[exact-worker {worker_id}] {index + 1}/{n_fixtures} "
                f"survivors={len(survivors)} inbound_fail={n_inbound_fail} "
                f"outbound_fail={n_outbound_fail} grasp_fail={n_grasp_fail} "
                f"initial_fail={n_initial_fail}",
                flush=True,
            )
    return {
        "survivors": survivors,
        "n_inbound_fail": n_inbound_fail,
        "n_outbound_fail": n_outbound_fail,
        "n_grasp_fail": n_grasp_fail,
        "n_initial_fail": n_initial_fail,
        "n_scored": n_fixtures,
        "packed": packed,
    }


def filter_exact_survivors(
    fixtures: Sequence[dict[str, Any]],
    cells: Sequence[dict[str, Any]],
    *,
    min_clearance_m: float = MIN_NOMINAL_CLEARANCE_M,
    workers: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, np.ndarray]]:
    empty_counts = {
        "inbound_no_contact": 0,
        "outbound_no_contact": 0,
        "grasp_below_nominal": 0,
        "initial_contact": 0,
        "n_scored": 0,
    }
    empty_packed: dict[str, np.ndarray] = {}
    if not fixtures:
        return [], empty_counts, empty_packed
    n_workers = max(1, min(int(workers), len(fixtures)))
    payloads = []
    if n_workers == 1:
        payloads = [
            {
                "fixtures": list(fixtures),
                "cells": cells,
                "min_clearance_m": min_clearance_m,
                "worker_id": 0,
            }
        ]
        results = [_eval_chunk(payloads[0])]
    else:
        chunks = [list(fixtures[i::n_workers]) for i in range(n_workers)]
        context = multiprocessing.get_context("spawn")
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, mp_context=context
        ) as pool:
            futures = [
                pool.submit(
                    _eval_chunk,
                    {
                        "fixtures": chunk,
                        "cells": cells,
                        "min_clearance_m": min_clearance_m,
                        "worker_id": index,
                    },
                )
                for index, chunk in enumerate(chunks)
                if chunk
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    survivors: list[dict[str, Any]] = []
    packed_parts: list[dict[str, np.ndarray]] = []
    for result in results:
        survivors.extend(result["survivors"])
        empty_counts["inbound_no_contact"] += result["n_inbound_fail"]
        empty_counts["outbound_no_contact"] += result["n_outbound_fail"]
        empty_counts["grasp_below_nominal"] += result["n_grasp_fail"]
        empty_counts["initial_contact"] += result["n_initial_fail"]
        empty_counts["n_scored"] += result["n_scored"]
        packed_parts.append(result["packed"])
    survivors.sort(key=lambda item: tuple(item["key"]))
    packed = {
        key: np.concatenate([part[key] for part in packed_parts], axis=0)
        for key in packed_parts[0]
    }
    return survivors, empty_counts, packed


def verify_sha256(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    digest = sha256_file(path)
    if digest != str(expected):
        raise ValueError(f"{label} SHA-256 mismatch for {path}: {digest} != {expected}")
    return digest


def verify_reconstruction_bundle(
    reconstruction_root: Path,
    *,
    source_summary: Path = SOURCE_SUMMARY,
) -> dict[str, Any]:
    """Verify reconstruction.json, NPZs, source rows, and the V5 scene before load."""
    document_path = reconstruction_root / "reconstruction.json"
    document = json.loads(document_path.read_text())
    stored = document.get("artifact_sha256")
    payload = dict(document)
    payload.pop("artifact_sha256", None)
    if sha256_payload(payload) != stored:
        raise ValueError("reconstruction.json self-hash mismatch")
    npz_files = document.get("npz_files") or []
    if len(npz_files) != 8:
        raise ValueError(f"reconstruction.json is missing NPZ hashes: {len(npz_files)}")
    for item in npz_files:
        verify_sha256(reconstruction_root / item["path"], item["sha256"], "reconstruction NPZ")
    verify_sha256(
        source_summary,
        document["baseline_summary_sha256"],
        "V9.5 baseline summary",
    )
    scene = ROOT / SCENE_XML_RELATIVE
    scene_digest = verify_sha256(scene, document["scene_xml_sha256"], "V5 scene XML")
    if scene_digest != PLACE_V5_SCENE_SHA256:
        raise ValueError("V5 scene XML does not match the frozen contract hash")
    if not document.get("scene_xml_sha256_matches_contract"):
        raise ValueError("reconstruction.json does not attest the frozen V5 scene hash")
    for row in document["rows"]:
        role = int(row["role_index"])
        episode = str(row["episode_id"])
        row_dir = source_summary.parent / "expert_screen_rows" / (
            f"{role:03d}_{episode[:16]}"
        )
        verify_sha256(row_dir / "result.json", row["result_sha256"], f"role {role} result.json")
        verify_sha256(
            row_dir / "trajectory.json",
            row["trajectory_sha256"],
            f"role {role} trajectory.json",
        )
        npz_relative = row.get("npz_relative_path") or f"rows/{role:03d}_{episode[:16]}.npz"
        verify_sha256(
            reconstruction_root / npz_relative,
            row["npz_sha256"],
            f"role {role} reconstruction NPZ",
        )
    return document


def implementation_hashes() -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def aabb_cells_from_snapshots(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for cell in cells:
        out.append(
            {
                "role_index": cell.get("role_index"),
                "family": cell.get("family"),
                "intrusion_side": cell.get("intrusion_side"),
                "robot_lo": np.asarray(cell["robot_lo"], dtype=np.float64),
                "robot_hi": np.asarray(cell["robot_hi"], dtype=np.float64),
                "target_lo": np.asarray(cell["target_lo"], dtype=np.float64),
                "target_hi": np.asarray(cell["target_hi"], dtype=np.float64),
                "inbound_mask": np.asarray(cell["inbound_mask"], dtype=bool),
                "outbound_mask": np.asarray(cell["outbound_mask"], dtype=bool),
                "grasp_mask": np.asarray(cell["grasp_mask"], dtype=bool),
                "initial_mask": np.asarray(cell["initial_mask"], dtype=bool),
            }
        )
    return out


def load_cell_snapshot(path: Path) -> dict[str, Any]:
    packed = np.load(path, allow_pickle=False)
    for key in ("robot_pos", "robot_mat", "robot_lo", "robot_hi", "target_pos", "target_mat", "target_lo", "target_hi"):
        if packed[key].dtype != np.float64:
            raise ValueError(f"{path} {key} must be float64, got {packed[key].dtype}")
    robot_verts = unpack_mesh_verts(
        packed["robot_mesh_start"], packed["robot_mesh_count"], packed["robot_mesh_verts"]
    )
    target_verts = unpack_mesh_verts(
        packed["target_mesh_start"], packed["target_mesh_count"], packed["target_mesh_verts"]
    )
    return {
        "role_index": int(packed["role_index"]) if "role_index" in packed.files else None,
        "robot_geom_ids": (
            np.asarray(packed["robot_geom_ids"], dtype=np.int32)
            if "robot_geom_ids" in packed.files
            else None
        ),
        "target_geom_ids": (
            np.asarray(packed["target_geom_ids"], dtype=np.int32)
            if "target_geom_ids" in packed.files
            else None
        ),
        "robot_gtype": np.asarray(packed["robot_gtype"]),
        "robot_size": np.asarray(packed["robot_size"], dtype=np.float64),
        "robot_pos": np.asarray(packed["robot_pos"], dtype=np.float64),
        "robot_mat": np.asarray(packed["robot_mat"], dtype=np.float64),
        "robot_verts": robot_verts,
        "robot_lo": np.asarray(packed["robot_lo"], dtype=np.float64),
        "robot_hi": np.asarray(packed["robot_hi"], dtype=np.float64),
        "target_gtype": np.asarray(packed["target_gtype"]),
        "target_size": np.asarray(packed["target_size"], dtype=np.float64),
        "target_pos": np.asarray(packed["target_pos"], dtype=np.float64),
        "target_mat": np.asarray(packed["target_mat"], dtype=np.float64),
        "target_verts": target_verts,
        "target_lo": np.asarray(packed["target_lo"], dtype=np.float64),
        "target_hi": np.asarray(packed["target_hi"], dtype=np.float64),
        "inbound_mask": np.asarray(packed["inbound_mask"], dtype=bool),
        "outbound_mask": np.asarray(packed["outbound_mask"], dtype=bool),
        "grasp_mask": np.asarray(packed["grasp_mask"], dtype=bool),
        "initial_mask": np.asarray(packed["initial_mask"], dtype=bool),
        "tcp_m": np.asarray(packed["tcp_m"], dtype=np.float64) if "tcp_m" in packed.files else None,
        "tcp_mat": np.asarray(packed["tcp_mat"], dtype=np.float64) if "tcp_mat" in packed.files else None,
    }


def snapshot_jobs_from_reconstruction(
    reconstruction: dict[str, Any],
    source_summary: Path = SOURCE_SUMMARY,
) -> list[dict[str, Any]]:
    source = json.loads(source_summary.read_text())
    rows = list(source["manifest_rows"])
    by_episode = {item["episode_id"]: item for item in source["results"]}
    jobs = []
    for row in reconstruction["rows"]:
        if not row.get("clean_success"):
            continue
        family = str(row.get("family") or "")
        if family not in PHYSICS_CLEAN_FAMILIES:
            continue
        episode = str(row["episode_id"])
        directory = source_summary.parent / "expert_screen_rows" / (
            f"{int(row['role_index']):03d}_{episode[:16]}"
        )
        result = json.loads((directory / "result.json").read_text())
        manifest = next(item for item in rows if item["episode_id"] == episode)
        jobs.append(
            {
                "row_dir": str(directory),
                "manifest_row": manifest,
                "expected": by_episode[episode],
                "selected_seed": result.get("selected_seed")
                or {
                    "seed_u32": manifest["task_seed_u32"],
                    "seed_u64": manifest["task_seed_u64"],
                },
                "family": family,
                "intrusion_side": row.get("intrusion_side"),
                "result_sha256": row["result_sha256"],
                "trajectory_sha256": row["trajectory_sha256"],
                "npz_sha256": row["npz_sha256"],
                "npz_relative_path": row.get("npz_relative_path"),
            }
        )
    if len(jobs) != N_CLEAN_CELLS:
        raise RuntimeError(f"expected {N_CLEAN_CELLS} clean snapshot jobs, found {len(jobs)}")
    return jobs


def dump_collision_snapshots(
    reconstruction_root: Path,
    snapshot_root: Path,
    *,
    workers: int = 4,
    source_summary: Path = SOURCE_SUMMARY,
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    reconstruction = verify_reconstruction_bundle(
        reconstruction_root, source_summary=source_summary
    )
    snapshot_root.mkdir(parents=True, exist_ok=True)
    jobs = snapshot_jobs_from_reconstruction(reconstruction, source_summary)
    files = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(int(workers), 4),
        mp_context=context,
        max_tasks_per_child=1,
    ) as pool:
        futures = [pool.submit(reconstruct_one_row, job) for job in jobs]
        by_role = {
            int(Path(job["row_dir"]).name.split("_")[0]): job for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            name = item["npz_name"].replace(".npz", "_snapshot.npz")
            path = snapshot_root / name
            np.savez_compressed(path, **item["snapshot"])
            role = int(item["summary"]["role_index"])
            matching = by_role[role]
            files.append(
                {
                    "path": str(path.relative_to(snapshot_root.parent)),
                    "sha256": sha256_file(path),
                    "role_index": role,
                    "episode_id": item["summary"]["episode_id"],
                    "family": matching.get("family") or item["summary"].get("family"),
                    "intrusion_side": matching.get("intrusion_side")
                    or item["summary"].get("intrusion_side"),
                    "max_tcp_residual_m": item["summary"]["max_tcp_residual_m"],
                    "result_sha256": matching["result_sha256"],
                    "trajectory_sha256": matching["trajectory_sha256"],
                    "reconstruction_npz_sha256": matching["npz_sha256"],
                    "reconstruction_npz_path": matching.get("npz_relative_path"),
                    "dtype": "float64",
                }
            )
    files.sort(key=lambda item: int(item["role_index"]))
    document = {
        "schema_version": SNAPSHOT_SCHEMA,
        "n_cells": len(files),
        "physics_stepped": False,
        "episodes_run": False,
        "dtype": "float64",
        "reconstruction_sha256": reconstruction["artifact_sha256"],
        "baseline_summary_sha256": reconstruction["baseline_summary_sha256"],
        "scene_xml_sha256": reconstruction["scene_xml_sha256"],
        "implementation_sha256": sha256_payload(implementation_hashes()),
        "implementation_files": implementation_hashes(),
        "reconstruction_npz_files": reconstruction.get("npz_files"),
        "files": files,
    }
    digest = write_immutable(snapshot_root / "snapshots.json", document)
    document["artifact_sha256"] = digest
    return document


def load_clean_snapshots(
    snapshot_root: Path,
    *,
    reconstruction: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta_path = snapshot_root / "snapshots.json"
    document = json.loads(meta_path.read_text())
    stored = document.get("artifact_sha256")
    payload = dict(document)
    payload.pop("artifact_sha256", None)
    if sha256_payload(payload) != stored:
        raise ValueError("snapshots.json self-hash mismatch")
    if reconstruction is not None:
        if document.get("reconstruction_sha256") != reconstruction.get("artifact_sha256"):
            raise ValueError("snapshot metadata is not bound to this reconstruction.json")
        if document.get("baseline_summary_sha256") != reconstruction.get(
            "baseline_summary_sha256"
        ):
            raise ValueError("snapshot metadata is not bound to the V9.5 source summary")
        if document.get("scene_xml_sha256") != reconstruction.get("scene_xml_sha256"):
            raise ValueError("snapshot metadata is not bound to the frozen V5 scene")
        if document.get("implementation_sha256") != sha256_payload(implementation_hashes()):
            raise ValueError("snapshot metadata is not bound to the current implementation")
        by_role = {int(row["role_index"]): row for row in reconstruction["rows"]}
    else:
        by_role = {}
    cells = []
    for item in document["files"]:
        path = snapshot_root.parent / item["path"]
        if not path.is_file():
            path = snapshot_root / Path(item["path"]).name
        verify_sha256(path, item["sha256"], f"snapshot role {item['role_index']}")
        if by_role:
            row = by_role[int(item["role_index"])]
            if item.get("result_sha256") != row.get("result_sha256"):
                raise ValueError(
                    f"snapshot role {item['role_index']} is not bound to source result.json"
                )
            if item.get("trajectory_sha256") != row.get("trajectory_sha256"):
                raise ValueError(
                    f"snapshot role {item['role_index']} is not bound to source trajectory.json"
                )
            if item.get("reconstruction_npz_sha256") != row.get("npz_sha256"):
                raise ValueError(
                    f"snapshot role {item['role_index']} is not bound to reconstruction NPZ"
                )
        cell = load_cell_snapshot(path)
        cell["role_index"] = item["role_index"]
        cell["episode_id"] = item["episode_id"]
        cell["family"] = item.get("family")
        cell["intrusion_side"] = item.get("intrusion_side")
        cells.append(cell)
    cells.sort(key=lambda item: int(item["role_index"]))
    if len(cells) != N_CLEAN_CELLS:
        raise RuntimeError(f"expected {N_CLEAN_CELLS} snapshots, found {len(cells)}")
    return document, cells


def live_mj_forward_parity_cases() -> dict[str, Any]:
    """Contact-parity and near-threshold checks on a live mj_forward world."""
    import mujoco

    xml = f"""
    <mujoco>
      <worldbody>
        <body name="left" pos="0 0 0">
          <geom name="left_g" type="box" size="0.5 0.5 0.5" contype="1" conaffinity="1"/>
        </body>
        <body name="sphere" pos="1.2 1.2 0.5">
          <geom name="sphere_g" type="sphere" size="0.25" contype="1" conaffinity="1"/>
        </body>
        <body name="nested" pos="0 0 0">
          <geom name="nested_g" type="box" size="0.1 0.1 0.1" contype="1" conaffinity="1"/>
        </body>
        <body name="near" pos="{1.0 + NEAR_THRESHOLD_M} 0 0">
          <geom name="near_g" type="box" size="0.5 0.5 0.5" contype="1" conaffinity="1"/>
        </body>
        <body name="inside" pos="{1.0 + NEAR_THRESHOLD_M - 0.001} 0 0">
          <geom name="inside_g" type="box" size="0.5 0.5 0.5" contype="1" conaffinity="1"/>
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

    nested = _pair("left_g", "nested_g")
    sphere = _pair("left_g", "sphere_g")
    near = _pair("left_g", "near_g")
    inside = _pair("left_g", "inside_g")
    return {
        "nested_contact": nested,
        "aabb_overlap_separated_sphere": sphere,
        "near_threshold_25mm": near,
        "below_threshold": inside,
        "nested_is_contact": nested["true_distance_m"] <= CONTACT_DISTANCE_M,
        "sphere_clearance_gt_25mm": sphere["true_distance_m"] > NEAR_THRESHOLD_M,
        "near_gap_m": near["true_distance_m"],
        "parity_ok": bool(
            nested["abs_delta_m"] <= LIVE_PARITY_ATOL_M
            and sphere["abs_delta_m"] <= 1e-5
            and near["abs_delta_m"] <= 1e-5
            and inside["abs_delta_m"] <= 1e-5
        ),
    }


def live_scene_snapshot_parity(
    reconstruction: dict[str, Any],
    cells: Sequence[dict[str, Any]],
    fixture: dict[str, Any],
    source_summary: Path = SOURCE_SUMMARY,
) -> dict[str, Any]:
    """Compare snapshot GJK to live true_distance on one retained-qpos frame."""
    import mujoco

    from reconstruct_pact_place_v99_baseline import (
        cleanup_task,
        collision_enabled_robot_geom_ids,
        pickup_collision_geom_ids,
        prepare_task,
    )
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    cell = cells[0]
    jobs = snapshot_jobs_from_reconstruction(reconstruction, source_summary)
    job = next(
        item
        for item in jobs
        if Path(item["row_dir"]).name.startswith(f"{int(cell['role_index']):03d}_")
    )
    trajectory = json.loads((Path(job["row_dir"]) / "trajectory.json").read_text())
    inbound_frames = np.flatnonzero(cell["inbound_mask"])
    if inbound_frames.size == 0:
        raise RuntimeError("no inbound frames for live parity")
    frame = int(inbound_frames[min(5, inbound_frames.size - 1)])
    task = sampler = scratch = None
    try:
        task, sampler, scratch = prepare_task(
            job["manifest_row"], seed_u32=(job.get("selected_seed") or {}).get("seed_u32")
        )
        env = task.env
        model, data = env.current_model, env.current_data
        apply_recorded_qpos(env, trajectory["steps"][frame]["qpos"])
        center = np.asarray(fixture["center_m"], dtype=np.float64)
        half = np.asarray(fixture["half_m"], dtype=np.float64)
        pendant_gid = int(model.geom(PENDANT_GEOM).id)
        model.geom_size[pendant_gid] = half
        mocap_id = int(model.body_mocapid[int(model.body(PENDANT_BODY).id)])
        data.mocap_pos[mocap_id] = center
        mujoco.mj_forward(model, data)
        robot_ids = collision_enabled_robot_geom_ids(model)
        target_ids = pickup_collision_geom_ids(task)
        live = float(true_distance(model, data, robot_ids + target_ids, [pendant_gid]))
        box = GeomShape.posed_axis_aligned_box(center, half)
        snapshot_min = float("inf")
        for geom in range(cell["robot_gtype"].shape[0]):
            shape = _shape_at(
                gtype=cell["robot_gtype"],
                size=cell["robot_size"],
                pos=cell["robot_pos"],
                mat=cell["robot_mat"],
                verts=cell["robot_verts"],
                frame=frame,
                geom=geom,
            )
            snapshot_min = min(snapshot_min, float(gjk_distance(shape, box)))
        for geom in range(cell["target_gtype"].shape[0]):
            if cell["target_lo"].shape[1] <= geom:
                continue
            shape = _shape_at(
                gtype=cell["target_gtype"],
                size=cell["target_size"],
                pos=cell["target_pos"],
                mat=cell["target_mat"],
                verts=cell["target_verts"],
                frame=frame,
                geom=geom,
            )
            snapshot_min = min(snapshot_min, float(gjk_distance(shape, box)))
        return {
            "role_index": int(cell["role_index"]),
            "frame": frame,
            "live_true_distance_m": live,
            "snapshot_gjk_m": snapshot_min,
            "abs_delta_m": abs(live - snapshot_min),
            "parity_ok": bool(abs(live - snapshot_min) <= 1e-4),
            "physics_stepped": False,
        }
    finally:
        if task is not None:
            cleanup_task(task, sampler, scratch)


def write_witness_npz(
    path: Path,
    packed: dict[str, np.ndarray],
    *,
    role_indices: Sequence[int],
) -> str:
    payload = dict(packed)
    payload["role_indices"] = np.asarray(role_indices, dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction-root", type=Path, default=DEFAULT_RECONSTRUCTION)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dump-snapshots", action="store_true")
    args = parser.parse_args()
    if args.dump_snapshots:
        document = dump_collision_snapshots(
            args.reconstruction_root,
            args.snapshot_root,
            workers=args.workers,
        )
        print(
            json.dumps(
                {
                    "snapshots": document["files"],
                    "artifact_sha256": document["artifact_sha256"],
                    "dtype": "float64",
                },
                indent=2,
            )
        )
        return 0
    raise SystemExit("pass --dump-snapshots or import this module from search")


if __name__ == "__main__":
    raise SystemExit(main())
