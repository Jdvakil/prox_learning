#!/usr/bin/env python3
"""V10 siting v2 close-out: exact environment predicate on the v1 prefilter.

Does not overwrite the superseded v1 robot/target catalog. Does not run
routing, sequential IK, signal screens, episodes, or env.step scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import PLACE_V5_SCENE_SHA256, sha256_file  # noqa: E402
from pact_place_v10_catalog import (  # noqa: E402
    assembly_from_two_lobe_keys,
    crossbar_keys_from_lobe_keys,
    load_prefilter_bits,
    load_prefilter_lobe_keys,
    load_prefilter_margins,
    load_prefilter_volumes,
    stem_keys_from_lobe_keys,
    unique_rounded_keys,
    unique_union_count,
    verify_prefilter_catalog,
    write_survivor_catalog_v2,
)
from pact_place_v10_compound_pendant_contract import (  # noqa: E402
    AUDIT_PANEL_CLEAR_COUNT,
    AUDIT_PANEL_CLEAR_UNION_COUNT,
    AUDIT_ROBOT_TARGET_PREFILTER_COUNT,
    CATALOG_SCHEMA_V2,
    PLACE_V10_SCENE_SHA256,
    SCENE_XML_RELATIVE,
    SITING_SCHEMA_V2,
    V1_PREFILTER_CATALOG_RELATIVE,
    V1_PREFILTER_CATALOG_SHA256,
    V1_SITING_PAYLOAD_SHA256,
    V1_SITING_RELATIVE,
    V2_SITING_RELATIVE,
    V5_SCENE_XML_RELATIVE,
    V99_RECONSTRUCTION_SHA256,
    V99_SCOPED_CONCLUSION,
    V99_SITING_SHA256,
    V99_SNAPSHOT_SHA256,
    empty_authorization,
    v10_implementation_hashes,
)
from pact_place_v10_environment import (  # noqa: E402
    assembly_panel_clear_on_side,
    assembly_row_environment_flags,
    combine_assembly_environment_cache,
    dump_cell_environments,
    merge_component_caches,
    panel_aabb_provenance_ok,
    panel_boxes_from_dump,
    panels_for_side,
    panels_from_v95_result_json,
    scan_panel_clear_mask,
    score_assembly_environment,
    score_unique_keys_environment,
)
from pact_place_v10_exact import (  # noqa: E402
    evaluate_assembly_exact,
    score_unique_keys_initial_target,
    verify_v99_inputs,
)
from pact_place_v10_geometry import (  # noqa: E402
    next_search_family,
    planning_probe_assembly,
    planning_probe_v1_invalid_assembly,
)
from pact_place_v10_runtime import establish_v10_runtime_env, write_immutable  # noqa: E402
from pact_place_v99_exact import snapshot_jobs_from_reconstruction  # noqa: E402

DEFAULT_OUTPUT = ROOT / Path(V2_SITING_RELATIVE).parent
NEAR_THRESHOLD_M = 0.002


class ProvenanceError(RuntimeError):
    """Stop: computed counts or hashes disagree with the bound artifacts."""


def _probe_lobe_keys(assembly: dict[str, Any]) -> np.ndarray:
    lobes = [
        item
        for item in assembly["components"]
        if item["role"] == "lobe" and item.get("active", True)
    ]
    return np.asarray([item["key"] for item in lobes], dtype=np.float64)


def _find_row(keys: np.ndarray, probe_keys: np.ndarray) -> int | None:
    packed = np.round(np.asarray(keys, dtype=np.float64), 9)
    target = np.round(np.asarray(probe_keys, dtype=np.float64), 9)
    match = np.all(packed == target, axis=(1, 2))
    match |= np.all(packed[:, ::-1, :] == target, axis=(1, 2))
    hits = np.flatnonzero(match)
    if hits.size == 0:
        return None
    return int(hits[0])


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def _stop(output_root: Path, document: dict[str, Any], reason: str) -> int:
    document = dict(document)
    document.update(empty_authorization())
    document["stop_reason"] = reason
    document["routing_run"] = False
    document["physics_stepped"] = False
    document["episodes_run"] = False
    digest = write_immutable(output_root / "siting.json", document)
    print(json.dumps({"path": str(output_root / "siting.json"), "stop_reason": reason, "artifact_sha256": digest}, indent=2))
    return 1


def _parity_report(
    *,
    label: str,
    assembly: dict[str, Any],
    cells_env: list[dict[str, Any]],
    cache: dict[tuple[float, ...], dict[str, Any]],
    snapshot_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    direct = score_assembly_environment(assembly, cells_env)
    cached = combine_assembly_environment_cache(assembly, cache)
    robot_target = evaluate_assembly_exact(assembly, snapshot_cells)
    return {
        "label": label,
        "assembly_id": assembly.get("assembly_id"),
        "direct_panel_clear": bool(direct["panel_clear"]),
        "direct_environment_clear": bool(direct["environment_clear"]),
        "cached_panel_clear": bool(cached["panel_clear"]),
        "cached_environment_clear": bool(cached["environment_clear"]),
        "cache_direct_agree": bool(
            direct["panel_clear"] == cached["panel_clear"]
            and direct["environment_clear"] == cached["environment_clear"]
        ),
        "robot_target_necessity_ok": bool(robot_target["lobe_necessity_ok"]),
        "robot_target_grasp_clear": bool(robot_target["grasp_window_clear"]),
        "robot_target_initial_clear": bool(robot_target["initial_state_clear"]),
        "evaluated_all_cells": bool(direct["evaluated_all_cells"]),
        "n_cells": int(direct["n_cells"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    establish_v10_runtime_env()
    output_root = args.output_root.resolve()
    v1_root = (ROOT / V1_SITING_RELATIVE).resolve().parent
    if output_root == v1_root:
        raise ProvenanceError("refusing to write siting v2 into the superseded v1 directory")
    output_root.mkdir(parents=True, exist_ok=True)

    reconstruction, snapshot, cells = verify_v99_inputs()
    if reconstruction.get("artifact_sha256") != V99_RECONSTRUCTION_SHA256:
        raise ProvenanceError("V9.9 reconstruction hash changed")
    if snapshot.get("artifact_sha256") != V99_SNAPSHOT_SHA256:
        raise ProvenanceError("V9.9 snapshot hash changed")
    v99_siting = json.loads((ROOT / "diagnostics_output/pact_place_v99_siting/siting.json").read_text())
    if v99_siting.get("artifact_sha256") != V99_SITING_SHA256:
        raise ProvenanceError("V9.9 siting hash changed")

    v5_sha = sha256_file(ROOT / V5_SCENE_XML_RELATIVE)
    v10_sha = sha256_file(ROOT / SCENE_XML_RELATIVE)
    if v5_sha != PLACE_V5_SCENE_SHA256:
        raise ProvenanceError(f"V5 scene SHA mismatch: {v5_sha}")
    if v10_sha != PLACE_V10_SCENE_SHA256:
        raise ProvenanceError(f"V10 scene SHA mismatch: {v10_sha}")

    prefilter_path = ROOT / V1_PREFILTER_CATALOG_RELATIVE
    catalog_meta = verify_prefilter_catalog(
        prefilter_path, expected_sha256=V1_PREFILTER_CATALOG_SHA256
    )
    v1_siting = json.loads((ROOT / V1_SITING_RELATIVE).read_text())
    if v1_siting.get("artifact_sha256") != V1_SITING_PAYLOAD_SHA256:
        raise ProvenanceError("superseded v1 siting payload hash mismatch")

    print("[v10-siting-v2] loading prefilter lobe keys", flush=True)
    keys = load_prefilter_lobe_keys(prefilter_path)
    volumes = load_prefilter_volumes(prefilter_path)
    bits = load_prefilter_bits(prefilter_path)
    margins = load_prefilter_margins(prefilter_path)
    robot_target_prefilter_count = int(keys.shape[0])
    if robot_target_prefilter_count != AUDIT_ROBOT_TARGET_PREFILTER_COUNT:
        raise ProvenanceError(
            "prefilter row count "
            f"{robot_target_prefilter_count} != audit {AUDIT_ROBOT_TARGET_PREFILTER_COUNT}"
        )

    document: dict[str, Any] = {
        "schema_version": SITING_SCHEMA_V2,
        **empty_authorization(),
        "superseded_v1": {
            "path": V1_SITING_RELATIVE,
            "siting_sha256": V1_SITING_PAYLOAD_SHA256,
            "catalog_path": V1_PREFILTER_CATALOG_RELATIVE,
            "catalog_sha256": V1_PREFILTER_CATALOG_SHA256,
            "label": "robot_target_prefilter",
            "note": (
                "v1 counted robot/target necessity, grasp, initial robot, stems, "
                "crossbars, and hardcoded enclosure boxes. It did not score posed "
                "intrusion panels or clutter."
            ),
        },
        "reconstruction_sha256": reconstruction.get("artifact_sha256"),
        "snapshot_sha256": snapshot.get("artifact_sha256"),
        "v99_siting_sha256": v99_siting.get("artifact_sha256"),
        "v99_closed_untouched": True,
        "v99_scoped_conclusion": V99_SCOPED_CONCLUSION,
        "v5_scene_xml_sha256": v5_sha,
        "v10_scene_xml_sha256": v10_sha,
        "implementation_sha256": v10_implementation_hashes(),
        "physics_stepped": False,
        "episodes_run": False,
        "routing_run": False,
        "dtype": "float64",
        "catalog_schema": CATALOG_SCHEMA_V2,
        "robot_target_prefilter_count": robot_target_prefilter_count,
        "selected": [],
        "survivors": [],
        "v10_closed": False,
    }

    jobs = snapshot_jobs_from_reconstruction(reconstruction)
    manifest_panels = panels_from_v95_result_json(jobs)
    geoms_path = output_root / "environment_geoms.pkl.gz"
    summary_path = output_root / "environment_cells.json"
    if geoms_path.is_file() and summary_path.is_file():
        print("[v10-siting-v2] reusing dumped six-cell environment geoms", flush=True)
        from pact_place_v10_environment import load_environment_geoms

        dumped = load_environment_geoms(geoms_path)
        live_summary = json.loads(summary_path.read_text())
        live_by_role = {
            int(item["role_index"]): item for item in live_summary.get("cells", [])
        }
        for cell in dumped:
            summary = live_by_role.get(int(cell["role_index"]), {})
            cell["live_probes"] = {
                "probe_v1_invalid_panel_overlap": {
                    "live_environment_clear": summary.get("probe_v1_live_environment_clear"),
                    "parity_ok": summary.get("probe_v1_parity_ok"),
                },
                "probe_v2": {
                    "live_environment_clear": summary.get("probe_v2_live_environment_clear"),
                    "parity_ok": summary.get("probe_v2_parity_ok"),
                },
            }
    else:
        print("[v10-siting-v2] dumping six-cell environment geoms", flush=True)
        dumped = dump_cell_environments(output_root=output_root)
    dumped_panels = panel_boxes_from_dump(dumped)
    provenance = panel_aabb_provenance_ok(dumped_panels, manifest_panels)
    document["panel_aabb_provenance"] = provenance
    if not provenance.get("ok"):
        document["note"] = "Live dumped panels disagree with frozen V9.5 result.json AABBs."
        return _stop(output_root, document, "panel_aabb_provenance_mismatch")

    print("[v10-siting-v2] scanning panel AABB predicate", flush=True)
    panel_clear = scan_panel_clear_mask(keys, dumped_panels)
    panel_clear_count = int(np.count_nonzero(panel_clear))
    panel_clear_union_count = unique_union_count(keys[panel_clear]) if panel_clear_count else 0
    document["panel_clear_count"] = panel_clear_count
    document["panel_clear_union_aabb_count"] = panel_clear_union_count
    if panel_clear_count != AUDIT_PANEL_CLEAR_COUNT:
        document["note"] = (
            f"Computed panel-clear count {panel_clear_count} != audit {AUDIT_PANEL_CLEAR_COUNT}."
        )
        return _stop(output_root, document, "panel_clear_count_mismatch")
    if panel_clear_union_count != AUDIT_PANEL_CLEAR_UNION_COUNT:
        document["note"] = (
            "Computed panel-clear union AABB count "
            f"{panel_clear_union_count} != audit {AUDIT_PANEL_CLEAR_UNION_COUNT}."
        )
        return _stop(output_root, document, "panel_clear_union_count_mismatch")

    print("[v10-siting-v2] unique component tables", flush=True)
    unique_lobes, lobe_inv = unique_rounded_keys(keys)
    lobe_inv = lobe_inv.reshape(keys.shape[0], 2)
    stem_keys = stem_keys_from_lobe_keys(keys)
    unique_stems, stem_inv = unique_rounded_keys(stem_keys)
    stem_inv = stem_inv.reshape(keys.shape[0], 2)
    bar_keys = crossbar_keys_from_lobe_keys(keys)
    unique_bars, bar_inv = unique_rounded_keys(bar_keys)

    lobe_env = score_unique_keys_environment(unique_lobes, dumped, role="lobe")
    stem_env = score_unique_keys_environment(unique_stems, dumped, role="stem")
    bar_env = score_unique_keys_environment(unique_bars, dumped, role="crossbar")
    env_cache = merge_component_caches(
        lobe_env["cache"], stem_env["cache"], bar_env["cache"]
    )
    row_flags = assembly_row_environment_flags(
        lobe_inv=lobe_inv,
        stem_inv=stem_inv,
        bar_inv=bar_inv,
        lobe_env=lobe_env["environment_clear_all"],
        stem_env=stem_env["environment_clear_all"],
        bar_env=bar_env["environment_clear_all"],
        lobe_panel=lobe_env["panel_clear_all"],
        stem_panel=stem_env["panel_clear_all"],
        bar_panel=bar_env["panel_clear_all"],
    )
    cached_panel = row_flags["panel_clear"]
    if not np.array_equal(cached_panel, panel_clear):
        document["note"] = (
            "Cached component-wise panel flags disagree with the vectorized AABB scan."
        )
        document["cached_panel_clear_count"] = int(np.count_nonzero(cached_panel))
        return _stop(output_root, document, "cached_panel_flag_mismatch")

    print("[v10-siting-v2] initial-target unique scores", flush=True)
    lobe_target = score_unique_keys_initial_target(unique_lobes, cells, role="lobe")
    stem_target = score_unique_keys_initial_target(unique_stems, cells, role="stem")
    bar_target = score_unique_keys_initial_target(unique_bars, cells, role="crossbar")
    from pact_place_v10_environment import combine_row_flags

    target_clear = (
        np.all(combine_row_flags(lobe_inv, lobe_target["initial_target_clear_all"]), axis=1)
        & np.all(combine_row_flags(stem_inv, stem_target["initial_target_clear_all"]), axis=1)
        & combine_row_flags(bar_inv, bar_target["initial_target_clear_all"])
    )

    env_clear = row_flags["environment_clear"]
    if np.any(env_clear & np.logical_not(panel_clear)):
        document["note"] = "Environment-clear rows that are not panel-clear."
        return _stop(output_root, document, "environment_not_subset_of_panel_clear")

    full_mask = panel_clear & env_clear & target_clear
    full_count = int(np.count_nonzero(full_mask))
    full_union_count = unique_union_count(keys[full_mask]) if full_count else 0
    document["full_environment_exact_survivor_count"] = full_count
    document["corrected_unique_union_count"] = full_union_count
    document["rejection_counts"] = {
        "robot_target_prefilter": robot_target_prefilter_count,
        "rejected_by_panel": robot_target_prefilter_count - panel_clear_count,
        "panel_clear": panel_clear_count,
        "rejected_by_environment_after_panel": panel_clear_count
        - int(np.count_nonzero(panel_clear & env_clear)),
        "environment_clear_including_panels": int(np.count_nonzero(env_clear)),
        "rejected_by_initial_target_after_environment": int(
            np.count_nonzero(panel_clear & env_clear & np.logical_not(target_clear))
        ),
        "full_environment_exact_survivors": full_count,
    }
    document["unique_component_counts"] = {
        "lobes": int(unique_lobes.shape[0]),
        "stems": int(unique_stems.shape[0]),
        "crossbars": int(unique_bars.shape[0]),
    }

    atomic_path = output_root / "atomic_component_env_scores.npz"
    np.savez(
        atomic_path,
        lobe_keys=unique_lobes,
        stem_keys=unique_stems,
        crossbar_keys=unique_bars,
        lobe_panel_clear_all=lobe_env["panel_clear_all"],
        stem_panel_clear_all=stem_env["panel_clear_all"],
        crossbar_panel_clear_all=bar_env["panel_clear_all"],
        lobe_environment_clear_all=lobe_env["environment_clear_all"],
        stem_environment_clear_all=stem_env["environment_clear_all"],
        crossbar_environment_clear_all=bar_env["environment_clear_all"],
        lobe_initial_target_clear_all=lobe_target["initial_target_clear_all"],
        stem_initial_target_clear_all=stem_target["initial_target_clear_all"],
        crossbar_initial_target_clear_all=bar_target["initial_target_clear_all"],
        lobe_min_env_distance_m=lobe_env["min_distance_m"],
        stem_min_env_distance_m=stem_env["min_distance_m"],
        crossbar_min_env_distance_m=bar_env["min_distance_m"],
    )
    document["atomic_component_scores"] = {
        "path": "atomic_component_env_scores.npz",
        "sha256": sha256_file(atomic_path),
    }
    witness_digest = _write_json(
        output_root / "environment_contact_witnesses.json",
        {
            "lobe": lobe_env["contact_witnesses"][:4096],
            "stem": stem_env["contact_witnesses"][:4096],
            "crossbar": bar_env["contact_witnesses"][:4096],
            "initial_target": (
                lobe_target["contact_witnesses"]
                + stem_target["contact_witnesses"]
                + bar_target["contact_witnesses"]
            )[:4096],
            "truncated": True,
        },
    )
    document["environment_contact_witnesses"] = {
        "path": "environment_contact_witnesses.json",
        "sha256": witness_digest,
    }

    probe_v1 = planning_probe_v1_invalid_assembly()
    probe_v2 = planning_probe_assembly()
    left_panels = panels_for_side(dumped_panels, "left")
    right_panels = panels_for_side(dumped_panels, "right")
    probe_v1_left = assembly_panel_clear_on_side(probe_v1, left_panels)
    probe_v1_right = assembly_panel_clear_on_side(probe_v1, right_panels)
    if probe_v1_left or probe_v1_right:
        document["note"] = (
            "probe_v1_invalid_panel_overlap was not rejected by both side-specific "
            f"panel layouts (left_clear={probe_v1_left}, right_clear={probe_v1_right})."
        )
        return _stop(output_root, document, "probe_v1_not_rejected_by_both_sides")

    probe_reports = {
        "probe_v1_invalid_panel_overlap": _parity_report(
            label="probe_v1_invalid_panel_overlap",
            assembly=probe_v1,
            cells_env=dumped,
            cache=env_cache,
            snapshot_cells=cells,
        ),
        "probe_v2": _parity_report(
            label="probe_v2",
            assembly=probe_v2,
            cells_env=dumped,
            cache=env_cache,
            snapshot_cells=cells,
        ),
    }
    live = {
        int(cell["role_index"]): cell.get("live_probes") or {}
        for cell in dumped
    }
    probe_v2_live_clear = all(
        (live[int(cell["role_index"])].get("probe_v2") or {}).get("live_environment_clear")
        for cell in dumped
    )
    probe_v2_parity = all(
        (live[int(cell["role_index"])].get("probe_v2") or {}).get("parity_ok")
        for cell in dumped
    )
    if not probe_reports["probe_v2"]["direct_environment_clear"]:
        document["planning_probes"] = probe_reports
        document["note"] = "Corrected planning-probe-v2 failed the complete environment predicate."
        return _stop(output_root, document, "probe_v2_failed_environment_predicate")
    if not probe_reports["probe_v2"]["robot_target_necessity_ok"]:
        document["planning_probes"] = probe_reports
        document["note"] = "Planning-probe-v2 failed robot/target exact predicates."
        return _stop(output_root, document, "probe_v2_failed_robot_target_predicate")
    if not probe_v2_live_clear or not probe_v2_parity:
        document["planning_probes"] = probe_reports
        document["note"] = "Planning-probe-v2 live-scene checks failed or disagreed with dumped GJK."
        return _stop(output_root, document, "probe_v2_live_scene_failed")
    if not probe_reports["probe_v2"]["cache_direct_agree"]:
        document["planning_probes"] = probe_reports
        return _stop(output_root, document, "probe_v2_cache_direct_mismatch")
    if probe_reports["probe_v1_invalid_panel_overlap"]["direct_environment_clear"]:
        document["planning_probes"] = probe_reports
        return _stop(output_root, document, "probe_v1_incorrectly_environment_clear")

    v2_row = _find_row(keys, _probe_lobe_keys(probe_v2))
    if v2_row is None:
        document["planning_probes"] = probe_reports
        document["note"] = "Planning-probe-v2 is absent from the SHA-bound robot/target prefilter."
        return _stop(output_root, document, "probe_v2_missing_from_prefilter")
    if not bool(full_mask[v2_row]):
        document["planning_probes"] = probe_reports
        document["probe_v2_prefilter_index"] = v2_row
        document["note"] = "Planning-probe-v2 is not a full-environment exact survivor."
        return _stop(output_root, document, "probe_v2_not_full_exact_survivor")

    near_cases = []
    for scored, inv, name in (
        (lobe_env, lobe_inv, "lobe"),
        (stem_env, stem_inv, "stem"),
        (bar_env, bar_inv, "crossbar"),
    ):
        distances = scored["min_distance_m"]
        near = np.where(
            np.isfinite(distances)
            & (distances > 0.0)
            & (distances < NEAR_THRESHOLD_M)
        )[0]
        for unique_index in near.tolist()[:8]:
            key = scored["keys"][int(unique_index)]
            component = {
                "role": name,
                "name": f"{name}_near_{unique_index}",
                "center_m": key[:3].tolist(),
                "half_m": key[3:].tolist(),
                "active": True,
            }
            if name == "lobe":
                rows = np.flatnonzero(np.any(inv == int(unique_index), axis=1))
            elif name == "stem":
                rows = np.flatnonzero(np.any(inv == int(unique_index), axis=1))
            else:
                rows = np.flatnonzero(inv == int(unique_index))
            if rows.size == 0:
                continue
            assembly = assembly_from_two_lobe_keys(keys[int(rows[0])])
            near_cases.append(
                _parity_report(
                    label=f"near_{name}_{unique_index}",
                    assembly=assembly,
                    cells_env=dumped,
                    cache=env_cache,
                    snapshot_cells=cells,
                )
            )

    rng = np.random.RandomState(int(V1_PREFILTER_CATALOG_SHA256[:8], 16) % (2**31))
    sample_indices = []
    cx = np.round(keys[:, 0, 0], 9)
    for value in sorted(set(cx[panel_clear].tolist())):
        pool = np.flatnonzero(panel_clear & (cx == value))
        sample_indices.append(int(pool[int(rng.randint(0, len(pool)))]))
    from pact_place_v10_catalog import union_aabb_from_two_lobe_keys

    union_lo, union_hi = union_aabb_from_two_lobe_keys(keys[panel_clear])
    union_packed = np.round(np.concatenate([union_lo, union_hi], axis=1), 9)
    _, union_inv = np.unique(union_packed, axis=0, return_inverse=True)
    panel_idx = np.flatnonzero(panel_clear)
    chosen_unions = set()
    for _ in range(16):
        pick = int(rng.randint(0, len(panel_idx)))
        group = int(union_inv[pick])
        if group in chosen_unions:
            continue
        chosen_unions.add(group)
        sample_indices.append(int(panel_idx[pick]))
    sample_indices = sorted(set(sample_indices))
    sample_reports = []
    for index in sample_indices:
        assembly = assembly_from_two_lobe_keys(keys[int(index)])
        sample_reports.append(
            _parity_report(
                label=f"sample_{index}",
                assembly=assembly,
                cells_env=dumped,
                cache=env_cache,
                snapshot_cells=cells,
            )
        )
    parity_ok = all(item["cache_direct_agree"] for item in [probe_reports["probe_v1_invalid_panel_overlap"], probe_reports["probe_v2"], *near_cases, *sample_reports])
    document["cache_direct_parity"] = {
        "ok": bool(parity_ok),
        "probe_v1": probe_reports["probe_v1_invalid_panel_overlap"],
        "probe_v2": probe_reports["probe_v2"],
        "threshold_near": near_cases,
        "deterministic_sample": sample_reports,
        "n_sample_rows": len(sample_indices),
        "sample_center_x_values": sorted(set(float(cx[i]) for i in sample_indices)),
    }
    if not parity_ok:
        document["note"] = "Cached environment combination disagreed with direct live-scene evaluation."
        return _stop(output_root, document, "cache_direct_parity_failed")

    family = next_search_family(
        two_lobe_exact_survivors=[True] if full_count else [],
        two_lobe_failed_later=False,
    )
    document["three_lobe"] = {
        "searched": family == "three_lobe",
        "exact_survivors": 0,
        "counts": {},
    }
    if family == "three_lobe":
        document["note"] = (
            "Corrected two-lobe full-environment exact set is empty. "
            "Three-lobe escalation is registered but was not executed in this close-out "
            "because a new triple crossbar lattice would require a separate authorized run."
        )
        return _stop(output_root, document, "two_lobe_empty_three_lobe_not_run")

    catalog_path = output_root / "exact_survivors_v2.npz"
    catalog_sha = write_survivor_catalog_v2(
        catalog_path,
        lobe_keys=keys[full_mask],
        volume_m3=volumes[full_mask],
        lobe_necessity_bits=bits[full_mask],
        min_grasp_clearance_margin_m=margins[full_mask],
        topology="two_lobe",
    )
    index_path = output_root / "survivor_prefilter_indices.npy"
    np.save(index_path, np.flatnonzero(full_mask).astype(np.int64))
    document["survivor_catalog"] = {
        "path": "exact_survivors_v2.npz",
        "sha256": catalog_sha,
        "n": full_count,
        "schema": CATALOG_SCHEMA_V2,
        "prefilter_indices_path": "survivor_prefilter_indices.npy",
        "prefilter_indices_sha256": sha256_file(index_path),
    }
    document["planning_probes"] = probe_reports
    document["probe_v2_prefilter_index"] = v2_row
    document["planning_probe"] = {
        "probe_label": "probe_v2",
        "assembly_id": probe_v2["assembly_id"],
        "trust_anchor": True,
        **{key: probe_reports["probe_v2"][key] for key in probe_reports["probe_v2"] if key != "label"},
    }
    document["environment_dump"] = {
        "summary_path": "environment_cells.json",
        "geoms_path": "environment_geoms.pkl.gz",
        "geoms_sha256": sha256_file(output_root / "environment_geoms.pkl.gz"),
        "panel_npz_path": "environment_cells.npz",
        "panel_npz_sha256": sha256_file(output_root / "environment_cells.npz"),
    }
    document["prefilter_catalog"] = catalog_meta
    document["stop_reason"] = "exact_survivors_route_not_run"
    document["note"] = (
        f"{full_count} full-environment exact two-lobe survivor(s) after the "
        f"{robot_target_prefilter_count}-row robot/target prefilter and "
        f"{panel_clear_count} panel-clear rows. Routing, sensing, paired screens, "
        "and collection were not run."
    )
    document.update(empty_authorization())
    digest = write_immutable(output_root / "siting.json", document)
    print(
        json.dumps(
            {
                "path": str(output_root / "siting.json"),
                "robot_target_prefilter_count": robot_target_prefilter_count,
                "panel_clear_count": panel_clear_count,
                "full_environment_exact_survivor_count": full_count,
                "corrected_unique_union_count": full_union_count,
                "catalog_sha256": catalog_sha,
                "artifact_sha256": digest,
                "routing_run": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
