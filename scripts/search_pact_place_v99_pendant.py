#!/usr/bin/env python3
"""V9.9 lattice close-out: AABB broad-phase, then exact retained-qpos scoring.

V9.9 stays stopped. This script does not run sequential-IK routing, paired
screens, or collection. Dual-transit AABB hits are the broad-phase pool for
an exact true_distance / GJK pass on retained qpos.
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

from pact_place_v99_exact import (  # noqa: E402
    DEFAULT_SNAPSHOT_ROOT,
    SCOPED_CONCLUSION,
    aabb_cells_from_snapshots,
    dump_collision_snapshots,
    filter_exact_survivors,
    live_mj_forward_parity_cases,
    live_scene_snapshot_parity,
    load_clean_snapshots,
    verify_reconstruction_bundle,
    write_witness_npz,
)
from pact_place_v99_geometry import (  # noqa: E402
    enumerate_lattice,
    filter_lattice_dual_transit,
    filter_lattice_for_cells,
    lattice_raw_count,
)
from pact_place_v99_pendant_contract import (  # noqa: E402
    CONTRACT_VERSION,
    MIN_NOMINAL_CLEARANCE_M,
    empty_authorization,
)
from reconstruct_pact_place_v99_baseline import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_RECONSTRUCTION,
    write_immutable,
)

DEFAULT_OUTPUT = ROOT / "diagnostics_output/pact_place_v99_siting"

AABB_BROADPHASE_NOTE = (
    "AABB overlap is only a broad-phase screen. It disproves contact when "
    "boxes are separated and can conservatively certify a candidate when the "
    "grasp-window AABB gap is at least 25 mm, but it does not prove that the "
    "mesh occupies the pendant and it is not an exact clearance measurement. "
    "See scripts/pact_geom_distance.py. Dual-transit AABB hits are the pool "
    "for the exact retained-qpos pass."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction-root", type=Path, default=DEFAULT_RECONSTRUCTION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--stage",
        choices=("aabb-broadphase", "exact", "closeout"),
        default="closeout",
    )
    parser.add_argument(
        "--aabb-only",
        action="store_true",
        help="Deprecated alias for --stage aabb-broadphase",
    )
    parser.add_argument(
        "--force-dump-snapshots",
        action="store_true",
        help="Rebuild float64 snapshots even if snapshots.json exists",
    )
    args = parser.parse_args()
    stage = "aabb-broadphase" if args.aabb_only else args.stage
    if stage == "closeout":
        stage = "exact"
    reconstruction = verify_reconstruction_bundle(args.reconstruction_root)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_meta = args.snapshot_root / "snapshots.json"
    if stage != "aabb-broadphase" and (
        args.force_dump_snapshots or not snapshot_meta.is_file()
    ):
        dump_collision_snapshots(
            args.reconstruction_root,
            args.snapshot_root,
            workers=min(args.workers, 4),
        )
    snapshot_document = None
    snapshot_cells: list[dict[str, Any]] = []
    if stage != "aabb-broadphase":
        snapshot_document, snapshot_cells = load_clean_snapshots(
            args.snapshot_root, reconstruction=reconstruction
        )
        aabb_cells = aabb_cells_from_snapshots(snapshot_cells)
    else:
        raise SystemExit("AABB-only close-out is not used; run --stage closeout")
    fixtures = enumerate_lattice()
    aabb_certified = filter_lattice_for_cells(
        fixtures, aabb_cells, min_grasp_clearance_m=MIN_NOMINAL_CLEARANCE_M
    )
    dual_hit = filter_lattice_dual_transit(fixtures, aabb_cells)
    document = {
        "schema_version": "pact_place_v9_9_siting_v3",
        "contract_version": CONTRACT_VERSION,
        **empty_authorization(),
        "reconstruction_sha256": reconstruction.get("artifact_sha256"),
        "reconstruction_npz_files": reconstruction.get("npz_files"),
        "baseline_summary_sha256": reconstruction.get("baseline_summary_sha256"),
        "scene_xml_sha256": reconstruction.get("scene_xml_sha256"),
        "snapshot_sha256": None if snapshot_document is None else snapshot_document.get(
            "artifact_sha256"
        ),
        "implementation_sha256": None
        if snapshot_document is None
        else snapshot_document.get("implementation_sha256"),
        "lattice_raw_count": lattice_raw_count(),
        "lattice_enumerated": len(fixtures),
        "aabb_certified_survivors": len(aabb_certified),
        "aabb_survivors": len(aabb_certified),
        "aabb_dual_hit_count": len(dual_hit),
        "aabb_filter_note": AABB_BROADPHASE_NOTE,
        "aabb_only": False,
        "exact_survivors": None,
        "geometry_search_conclusive": False,
        "v99_closed": False,
        "scoped_conclusion": None,
        "selected": [],
        "stop_reason": None,
        "survivors": [],
        "physics_stepped": False,
        "episodes_run": False,
        "routing_run": False,
        "dtype": "float64",
        "evaluated_all_predicates_all_cells": True,
    }
    live_mini = live_mj_forward_parity_cases()
    document["live_mj_forward_parity"] = {
        "nested_is_contact": live_mini["nested_is_contact"],
        "sphere_clearance_gt_25mm": live_mini["sphere_clearance_gt_25mm"],
        "near_gap_m": live_mini["near_gap_m"],
        "parity_ok": live_mini["parity_ok"],
        "cases": {
            key: live_mini[key]
            for key in (
                "nested_contact",
                "aabb_overlap_separated_sphere",
                "near_threshold_25mm",
                "below_threshold",
            )
        },
    }
    if not live_mini["parity_ok"]:
        raise SystemExit("live mj_forward contact-parity failed")
    exact, exact_counts, packed = filter_exact_survivors(
        [item["fixture"] for item in dual_hit],
        snapshot_cells,
        min_clearance_m=MIN_NOMINAL_CLEARANCE_M,
        workers=args.workers,
    )
    witness_rel = "exact_witnesses.npz"
    witness_digest = write_witness_npz(
        output_root / witness_rel,
        packed,
        role_indices=[int(cell["role_index"]) for cell in snapshot_cells],
    )
    if dual_hit:
        scene_parity = live_scene_snapshot_parity(
            reconstruction,
            snapshot_cells,
            dual_hit[0]["fixture"],
        )
    else:
        scene_parity = {"parity_ok": True, "skipped": "no_dual_hit"}
    document["live_scene_snapshot_parity"] = scene_parity
    if not scene_parity.get("parity_ok", False):
        raise SystemExit("live scene snapshot GJK/true_distance parity failed")
    document["exact_survivors"] = len(exact)
    document["exact_reject_counts"] = exact_counts
    document["witness_npz"] = {"path": witness_rel, "sha256": witness_digest}
    document["survivors"] = [
        {
            "fixture": item["fixture"],
            "key": item["key"],
            "volume_m3": item["volume_m3"],
            "min_grasp_window_exact_clearance_m": item[
                "min_grasp_window_exact_clearance_m"
            ],
        }
        for item in exact
    ]
    document["authorizes_paired_screen"] = False
    if not exact:
        document["stop_reason"] = "no_exact_survivor"
        document["geometry_search_conclusive"] = True
        document["v99_closed"] = True
        document["scoped_conclusion"] = SCOPED_CONCLUSION
        document["note"] = (
            f"The exact retained-qpos close-out scored all {len(dual_hit)} "
            "dual-transit AABB hits with float64 GJK/true_distance geometry, "
            "all four predicates on all six cells, and found zero candidates. "
            f"Rejects (overlapping): inbound_no_contact="
            f"{exact_counts['inbound_no_contact']}, outbound_no_contact="
            f"{exact_counts['outbound_no_contact']}, grasp_below_nominal="
            f"{exact_counts['grasp_below_nominal']}, initial_contact="
            f"{exact_counts['initial_contact']}. "
            f"Scoped conclusion: {SCOPED_CONCLUSION}. "
            "Routing, paired screens, and collection were not run. The lattice "
            "and thresholds were not relaxed."
        )
    else:
        document["stop_reason"] = "exact_survivors_v99_stays_stopped"
        document["geometry_search_conclusive"] = False
        document["v99_closed"] = False
        document["note"] = (
            f"{len(exact)} exact retained-qpos survivors. V9.9 stays stopped: "
            "routing, paired screens, and collection were not run."
        )
    digest = write_immutable(output_root / "siting.json", document)
    print(
        json.dumps(
            {
                "path": str(output_root / "siting.json"),
                "lattice_enumerated": document["lattice_enumerated"],
                "aabb_certified_survivors": document["aabb_certified_survivors"],
                "aabb_dual_hit_count": document["aabb_dual_hit_count"],
                "exact_survivors": document["exact_survivors"],
                "stop_reason": document["stop_reason"],
                "geometry_search_conclusive": document["geometry_search_conclusive"],
                "v99_closed": document["v99_closed"],
                "scoped_conclusion": document["scoped_conclusion"],
                "artifact_sha256": digest,
                "authorizes_paired_screen": False,
            },
            indent=2,
        )
    )
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
