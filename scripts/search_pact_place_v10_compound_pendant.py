#!/usr/bin/env python3
"""V10 compound-pendant exact search. Does not modify V9.9 artifacts.

Default stage is the planning-probe regression, then two-lobe exact set-cover.
Three-lobe search runs only when two-lobe exact is empty. Routing is a later
stage and is not started from this script unless --stage route is passed after
exact survivors exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_v10_compound_pendant_contract import (  # noqa: E402
    CONTRACT_VERSION,
    empty_authorization,
)
from pact_place_v10_exact import (  # noqa: E402
    compact_survivor_record,
    component_clearance_summary,
    crossbar_cache_key,
    evaluate_assembly_exact,
    evaluate_assembly_from_component_caches,
    evaluate_planning_probe,
    score_lattice_parallel,
    verify_v99_inputs,
    write_component_witnesses,
    write_survivor_catalog,
)
from pact_place_v10_geometry import (  # noqa: E402
    NECESSITY_ALL_BITS,
    build_assembly,
    enumerate_lobes,
    lattice_raw_count,
    next_search_family,
    planning_probe_assembly,
    stream_covering_three_lobe_sets,
    stream_covering_two_lobe_pairs,
)
from pact_place_v10_runtime import (  # noqa: E402
    establish_v10_runtime_env,
    write_immutable,
)

DEFAULT_OUTPUT = ROOT / "diagnostics_output/pact_place_v10_siting"


def _base_document(reconstruction: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "pact_place_v10_siting_v1",
        "contract_version": CONTRACT_VERSION,
        **empty_authorization(),
        "reconstruction_sha256": reconstruction.get("artifact_sha256"),
        "snapshot_sha256": snapshot.get("artifact_sha256"),
        "scene_xml_sha256": reconstruction.get("scene_xml_sha256"),
        "v99_closed_untouched": True,
        "v99_scoped_conclusion": (
            "no survivor in the registered fixed rectangular-box lattice"
        ),
        "physics_stepped": False,
        "episodes_run": False,
        "routing_run": False,
        "dtype": "float64",
        "evaluated_all_predicates_all_cells": True,
        "lattice_raw_count": lattice_raw_count(),
        "selected": [],
        "survivors": [],
        "stop_reason": None,
        "v10_closed": False,
    }


def write_closeout(
    output_root: Path,
    document: dict[str, Any],
    *,
    stop_reason: str,
) -> str:
    document = dict(document)
    document["stop_reason"] = stop_reason
    document.update(empty_authorization())
    document["routing_run"] = False
    if stop_reason == "no_exact_compound_survivor":
        document["v10_closed"] = True
        document["geometry_search_conclusive"] = True
    return write_immutable(output_root / "siting.json", document)


def _eligible_lobes(
    lobes: list[dict[str, Any]],
    exact_cache: dict[tuple[float, ...], dict[str, Any]],
    stem_cache: dict[tuple[float, ...], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[float, ...], int]]:
    eligible: list[dict[str, Any]] = []
    bits: dict[tuple[float, ...], int] = {}
    for lobe in lobes:
        key = tuple(lobe["key"])
        scored = exact_cache.get(key)
        stem = stem_cache.get(key)
        if scored is None or stem is None:
            continue
        if not scored["grasp_clear_all"] or not scored["initial_clear_all"]:
            continue
        if not stem["grasp_clear_all"] or not stem["initial_clear_all"]:
            continue
        if not int(scored["bits"]):
            continue
        eligible.append(lobe)
        bits[key] = int(scored["bits"])
    return eligible, bits


def _search_family(
    *,
    family: str,
    lobes: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    exact_cache: dict[tuple[float, ...], dict[str, Any]],
    stem_cache: dict[tuple[float, ...], dict[str, Any]],
    max_survivors: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    counts = {
        "pairs_streamed": 0,
        "aabb_cover": 0,
        "exact_lobe_cover": 0,
        "assembly_accepted": 0,
        "assembly_build_rejected": 0,
        "crossbar_rejected": 0,
        "eligible_lobes": 0,
    }
    survivors: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    eligible, exact_bits = _eligible_lobes(lobes, exact_cache, stem_cache)
    counts["eligible_lobes"] = len(eligible)
    print(
        f"[v10-{family}] start set-cover eligible_lobes={len(eligible)}",
        flush=True,
    )
    if family == "two_lobe":
        iterator = stream_covering_two_lobe_pairs(eligible, exact_bits)
    else:
        iterator = stream_covering_three_lobe_sets(eligible, exact_bits)
    crossbar_cache: dict[tuple[float, ...], dict[str, Any]] = {}
    for group in iterator:
        lobe_list = list(group)
        counts["pairs_streamed"] += 1
        counts["aabb_cover"] += 1
        counts["exact_lobe_cover"] += 1
        try:
            assembly = build_assembly(lobe_list)
        except ValueError:
            counts["assembly_build_rejected"] += 1
            if counts["pairs_streamed"] == 1 or counts["pairs_streamed"] % 50000 == 0:
                print(
                    f"[v10-{family}] streamed={counts['pairs_streamed']} "
                    f"accepted={counts['assembly_accepted']} "
                    f"build_rejected={counts['assembly_build_rejected']}",
                    flush=True,
                )
            continue
        bar = next(item for item in assembly["components"] if item["role"] == "crossbar")
        bar_key = crossbar_cache_key(bar)
        if bar_key not in crossbar_cache:
            crossbar_cache[bar_key] = component_clearance_summary(bar, cells)
        report = evaluate_assembly_from_component_caches(
            assembly,
            lobe_scores=exact_cache,
            stem_scores=stem_cache,
            crossbar_score=crossbar_cache[bar_key],
        )
        if not report["accepted"]:
            counts["crossbar_rejected"] += 1
            if counts["pairs_streamed"] == 1 or counts["pairs_streamed"] % 50000 == 0:
                print(
                    f"[v10-{family}] streamed={counts['pairs_streamed']} "
                    f"accepted={counts['assembly_accepted']} "
                    f"crossbar_rejected={counts['crossbar_rejected']}",
                    flush=True,
                )
            continue
        counts["assembly_accepted"] += 1
        survivors.append(compact_survivor_record(assembly, report))
        if len(reports) < 8:
            reports.append(evaluate_assembly_exact(assembly, cells))
        if max_survivors is not None and len(survivors) >= max_survivors:
            break
        if counts["assembly_accepted"] == 1 or counts["assembly_accepted"] % 1000 == 0:
            print(
                f"[v10-{family}] streamed={counts['pairs_streamed']} "
                f"survivors={len(survivors)} "
                f"crossbar_cache={len(crossbar_cache)}",
                flush=True,
            )
    survivors.sort(key=lambda item: item["assembly_id"])
    reports.sort(key=lambda item: str(item.get("assembly_id", "")))
    counts["unique_crossbars"] = len(crossbar_cache)
    return survivors, counts, reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        choices=("probe", "exact", "route"),
        default="probe",
    )
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    establish_v10_runtime_env()
    output_root = args.output_root.resolve()
    v1_siting = ROOT / "diagnostics_output/pact_place_v10_siting/siting.json"
    if output_root == v1_siting.parent.resolve() and v1_siting.is_file():
        raise RuntimeError(
            "refusing to overwrite the superseded V10 siting v1 artifact; "
            "use scripts/search_pact_place_v10_siting_v2.py"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    reconstruction, snapshot, cells = verify_v99_inputs()
    document = _base_document(reconstruction, snapshot)
    document["lattice_enumerated_lobes"] = None

    if args.stage == "route":
        document["note"] = (
            "Routing was requested but this runner will not start it until an "
            "exact compound survivor exists and a later command explicitly "
            "authorizes the route stage."
        )
        digest = write_closeout(output_root, document, stop_reason="route_not_started")
        print(json.dumps({"path": str(output_root / "siting.json"), "artifact_sha256": digest}, indent=2))
        return 1

    probe_assembly = planning_probe_assembly()
    probe = evaluate_planning_probe(cells)
    document["planning_probe"] = {
        "assembly_id": probe_assembly["assembly_id"],
        "reproduced_probe": probe["reproduced_probe"],
        "lobe_necessity_ok": probe["lobe_necessity_ok"],
        "lobe_necessity_bits": probe["lobe_necessity_bits"],
        "grasp_window_clear": probe["grasp_window_clear"],
        "initial_state_clear": probe["initial_state_clear"],
        "min_grasp_clearance_margin_m": probe["min_grasp_clearance_margin_m"],
    }
    role_indices = [int(cell["role_index"]) for cell in cells]
    witness_digest = write_component_witnesses(
        output_root / "exact_witnesses.npz",
        [probe],
        role_indices=role_indices,
    )
    document["witness_npz"] = {
        "path": "exact_witnesses.npz",
        "sha256": witness_digest,
    }
    if not probe["reproduced_probe"]:
        document["note"] = (
            "Planning-probe retained-qpos result was not reproduced. "
            "The V10 search implementation is not trustworthy."
        )
        digest = write_closeout(
            output_root, document, stop_reason="planning_probe_not_reproduced"
        )
        print(json.dumps({"path": str(output_root / "siting.json"), "artifact_sha256": digest}, indent=2))
        return 1
    if args.stage == "probe":
        document["stop_reason"] = "planning_probe_reproduced"
        document["note"] = (
            "Planning probe reproduced lobe contact on all twelve cell/"
            "traversal requirements with no component below the 25 mm "
            "grasp-window threshold. Exact set-cover was not run."
        )
        document.update(empty_authorization())
        digest = write_immutable(output_root / "siting.json", document)
        print(
            json.dumps(
                {
                    "path": str(output_root / "siting.json"),
                    "reproduced_probe": True,
                    "artifact_sha256": digest,
                },
                indent=2,
            )
        )
        return 0

    lobes = list(enumerate_lobes())
    document["lattice_enumerated_lobes"] = len(lobes)
    print(f"[v10-exact] enumerated {len(lobes)} lobes workers={args.workers}", flush=True)
    aabb_bits, exact_cache, stem_cache = score_lattice_parallel(
        lobes,
        cells,
        workers=max(1, int(args.workers)),
    )
    n_full = sum(1 for bits in aabb_bits.values() if bits == NECESSITY_ALL_BITS)
    n_any = sum(1 for bits in aabb_bits.values() if bits)
    n_exact = len(exact_cache)
    n_grasp = sum(
        1
        for scored in exact_cache.values()
        if scored["grasp_clear_all"] and scored["initial_clear_all"]
    )
    print(
        f"[v10-aabb] done any_bits={n_any} all_twelve={n_full} of {len(aabb_bits)} "
        f"exact_scored={n_exact} grasp_and_initial_clear={n_grasp} "
        f"stems_scored={len(stem_cache)}",
        flush=True,
    )
    document["aabb_any_bits"] = n_any
    document["aabb_all_twelve"] = n_full
    document["exact_lobes_scored"] = n_exact
    document["exact_lobes_grasp_and_initial_clear"] = n_grasp
    two_survivors, two_counts, two_reports = _search_family(
        family="two_lobe",
        lobes=lobes,
        cells=cells,
        exact_cache=exact_cache,
        stem_cache=stem_cache,
    )
    document["two_lobe"] = {
        "exact_survivors": len(two_survivors),
        "counts": two_counts,
    }
    family = next_search_family(
        two_lobe_exact_survivors=two_survivors,
        two_lobe_failed_later=False,
    )
    three_survivors: list[dict[str, Any]] = []
    three_counts: dict[str, int] = {}
    three_reports: list[dict[str, Any]] = []
    if family == "three_lobe":
        three_survivors, three_counts, three_reports = _search_family(
            family="three_lobe",
            lobes=lobes,
            cells=cells,
            exact_cache=exact_cache,
            stem_cache=stem_cache,
        )
    document["three_lobe"] = {
        "searched": family == "three_lobe",
        "exact_survivors": len(three_survivors),
        "counts": three_counts,
    }
    survivors = two_survivors or three_survivors
    reports = [probe] + two_reports + three_reports
    witness_digest = write_component_witnesses(
        output_root / "exact_witnesses.npz",
        reports,
        role_indices=role_indices,
    )
    document["witness_npz"] = {
        "path": "exact_witnesses.npz",
        "sha256": witness_digest,
    }
    catalog_digest = write_survivor_catalog(
        output_root / "exact_survivors.npz", survivors
    )
    document["survivor_catalog"] = {
        "path": "exact_survivors.npz",
        "sha256": catalog_digest,
        "n": len(survivors),
    }
    document["survivors"] = survivors if len(survivors) <= 64 else []
    document["exact_survivors"] = len(survivors)
    if not survivors:
        document["note"] = (
            "No fully connected exact two-lobe or three-lobe survivor in the "
            "registered lattice. Routing was not run."
        )
        digest = write_closeout(
            output_root, document, stop_reason="no_exact_compound_survivor"
        )
        print(json.dumps({"path": str(output_root / "siting.json"), "artifact_sha256": digest}, indent=2))
        return 1
    document["stop_reason"] = "exact_survivors_route_not_run"
    document["note"] = (
        f"{len(survivors)} exact compound survivor(s). V10 stays stopped "
        "before routing until that stage is explicitly run."
    )
    digest = write_immutable(output_root / "siting.json", document)
    print(
        json.dumps(
            {
                "path": str(output_root / "siting.json"),
                "exact_survivors": len(survivors),
                "topology": survivors[0]["topology"],
                "catalog_sha256": catalog_digest,
                "artifact_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
