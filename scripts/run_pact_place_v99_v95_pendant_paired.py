#!/usr/bin/env python3
"""Paired eight-row V9.9 screen against the frozen V9.5 baseline.

Runs one named candidate (signal or clearance) on the same eight rows.
Does not authorize collection. Stop if preservation or detour fails.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_corridor_contract import sha256_payload  # noqa: E402
from pact_place_v99_pendant_contract import (  # noqa: E402
    CONTRACT_VERSION,
    MIN_DETOUR_M,
    SAMPLER_CLASS,
    empty_authorization,
    grasp_posture_preserved,
)
from reconstruct_pact_place_v99_baseline import write_immutable  # noqa: E402
from run_pact_place_expert_screen import run_row  # noqa: E402

SOURCE_SUMMARY = ROOT / "diagnostics_output/pact_place_v95_raw_smoke/summary.json"
SCENE_XML = ROOT / (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v5.xml"
)
DEFAULT_SITING = ROOT / "diagnostics_output/pact_place_v99_siting/siting.json"
DEFAULT_RECONSTRUCTION = (
    ROOT / "diagnostics_output/pact_place_v99_baseline_reconstruction/reconstruction.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output/pact_place_v99_paired"


def _with_pendant(row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    patched = dict(row)
    patched["sampler_class"] = SAMPLER_CLASS
    patched["pact_v99_pendant_fixture"] = dict(candidate["fixture"])
    patched["pact_v99_route"] = dict(candidate.get("routes", {}).get(str(row["role_index"])) or candidate.get("route") or {})
    patched["pact_v99_contract_version"] = CONTRACT_VERSION
    patched["pact_v99_pendant_parked"] = False
    patched["pact_v98_pendant_lateral_bow"] = False
    return patched


def _detour(result: dict[str, Any], side: str) -> dict[str, Any]:
    telemetry = (result.get("pendant_v99") or {}).get(side) or {}
    return {
        f"{side}_lane_y_m": telemetry.get("lane_y_m"),
        f"{side}_min_abs_detour_m": telemetry.get("min_abs_detour_m"),
        f"{side}_detour_ok": bool(telemetry.get("detour_meets_minimum")),
        f"{side}_fallback_taken": bool(telemetry.get("fallback_taken")),
        f"{side}_clipped": bool(telemetry.get("clipped")),
        f"{side}_wrong_way": bool(telemetry.get("wrong_way")),
        f"{side}_ik_ok": bool(telemetry.get("ik_ok")),
        f"{side}_nominal_clearance_m": telemetry.get("nominal_clearance_m"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-role", choices=("signal", "clearance"), required=True)
    parser.add_argument("--siting", type=Path, default=DEFAULT_SITING)
    parser.add_argument("--reconstruction", type=Path, default=DEFAULT_RECONSTRUCTION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    siting = json.loads(args.siting.read_text())
    selected = [
        item for item in list(siting.get("selected") or []) if item.get("rank_role") == args.candidate_role
    ]
    if not selected:
        raise SystemExit(f"siting.json has no {args.candidate_role} candidate")
    candidate = selected[0]
    reconstruction = json.loads(args.reconstruction.read_text())
    canonical = {
        int(item["role_index"]): item for item in reconstruction.get("canonical_clean_cells") or []
    }
    source = json.loads(SOURCE_SUMMARY.read_text())
    rows = [_with_pendant(row, candidate) for row in source["manifest_rows"]]
    expected = {item["episode_id"]: item for item in source["results"]}
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    config_sha256 = sha256_payload(
        {
            "schema_version": "pact_place_v9_9_paired_v1",
            "candidate_role": args.candidate_role,
            "fixture": candidate["fixture"],
        }
    )
    results: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        max_tasks_per_child=1,
    ) as executor:
        futures = {
            executor.submit(
                run_row,
                row,
                config_sha256=config_sha256,
                output_root=str(output_root),
                scene_xml=str(SCENE_XML),
            ): row
            for row in rows
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: int(item["role_index"]))
    comparisons = []
    for result in results:
        expected_row = expected[result["episode_id"]]
        audit = result.get("contact_audit") or {}
        totals = audit.get("contact_class_totals") or {}
        posture = None
        canon = canonical.get(int(result["role_index"]))
        live_joints = (
            ((result.get("grasp_diagnostics") or {}).get("terminal_arm_joints_rad"))
        )
        if canon and live_joints:
            posture = grasp_posture_preserved(canon["grasp_arm_joints_rad"], live_joints)
        comparisons.append(
            {
                "role_index": result.get("role_index"),
                "episode_id": result.get("episode_id"),
                "family": expected_row.get("family_id") or expected_row.get("layout_family_id"),
                "intrusion_side": result.get("intrusion_side"),
                "baseline_clean_success": bool(expected_row.get("clean_success")),
                "pendant_clean_success": bool(result.get("clean_success")),
                "pendant_task_success": bool(result.get("task_success")),
                "status": result.get("status"),
                "mounted_fixture_contacts": int(totals.get("mounted_fixture") or 0),
                "clutter_contacts": int(totals.get("clutter") or 0),
                "other_strict_contacts": int(totals.get("other_strict") or totals.get("environment") or 0),
                "clutter_stability_ok": bool(result.get("clutter_stability_ok", True)),
                "grasp_posture": posture,
                **_detour(result, "inbound"),
                **_detour(result, "outbound"),
            }
        )
    preserved = all(
        (not item["baseline_clean_success"])
        or (
            item["pendant_clean_success"]
            and item["status"] == "complete"
            and item["mounted_fixture_contacts"] == 0
            and item["clutter_contacts"] == 0
            and not item["inbound_fallback_taken"]
            and not item["outbound_fallback_taken"]
            and not item["inbound_clipped"]
            and not item["outbound_clipped"]
            and item["inbound_detour_ok"]
            and item["outbound_detour_ok"]
            and (item["grasp_posture"] or {}).get("preserved", True)
        )
        for item in comparisons
    )
    document = {
        "schema_version": "pact_place_v9_9_paired_v1",
        **empty_authorization(),
        "candidate_role": args.candidate_role,
        "fixture": candidate["fixture"],
        "baseline_clean_rows_preserved": preserved,
        "authorizes_24_row_gate": False,
        "n": len(comparisons),
        "comparisons": comparisons,
        "min_detour_m": MIN_DETOUR_M,
    }
    digest = write_immutable(output_root / "paired.json", document)
    print(
        json.dumps(
            {
                "preserved": preserved,
                "path": str(output_root / "paired.json"),
                "artifact_sha256": digest,
            },
            indent=2,
        )
    )
    return 0 if preserved else 2


if __name__ == "__main__":
    raise SystemExit(main())
