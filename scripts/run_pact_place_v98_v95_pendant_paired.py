#!/usr/bin/env python3
"""Run the stored V9.5 8-row smoke with a ceiling pendant added.

Same seed, layout, and jitters as the fixture-free 6/8 baseline. Passing
``scene_xml`` is mandatory. The lateral bow is on by default; pass
``--no-lateral-bow`` only for the already-recorded bow-off control.
``inbound_ceiling_fixture_bow_took_effect`` is scored on ``status==complete``
rows only, so a sampling failure does not void a real bow.
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
from pact_place_v98_pendant_contract import (  # noqa: E402
    CONTRACT_VERSION,
    PENDANT_BOTTOM_Z_BOUNDS_M,
    PENDANT_CENTER_X_M,
    PENDANT_CENTER_Y_M,
    PENDANT_HALF_X_M,
    SAMPLER_CLASS,
    build_pendant_fixture,
)
from run_pact_place_expert_screen import run_row  # noqa: E402

SOURCE_SUMMARY = ROOT / "diagnostics_output/pact_place_v95_raw_smoke/summary.json"
SITING = ROOT / "diagnostics_output/pact_place_v98_pendant_siting/siting.json"
SCENE_XML = ROOT / (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v5.xml"
)
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output/pact_place_v98_v95_pendant_paired"
BASELINE_CLEAN = 6
CLIP_EPS_M = 1e-6


def _pendant_fixture(
    *, half_y: float | None, bottom_z: float, center_y: float | None
) -> dict[str, Any]:
    if half_y is not None:
        return build_pendant_fixture(
            bottom_z_m=float(bottom_z),
            half_y_m=float(half_y),
            center_x_m=PENDANT_CENTER_X_M,
            center_y_m=float(
                PENDANT_CENTER_Y_M if center_y is None else center_y
            ),
            half_x_m=PENDANT_HALF_X_M,
        )
    selected = json.loads(SITING.read_text()).get("selected") or {}
    fixture = dict(selected.get("fixture") or {})
    if not fixture:
        raise SystemExit("S1 siting.json has no selected pendant fixture")
    return fixture


def _with_pendant(
    row: dict[str, Any],
    fixture: dict[str, Any],
    *,
    lateral_bow: bool,
) -> dict[str, Any]:
    patched = dict(row)
    patched["sampler_class"] = SAMPLER_CLASS
    patched["pact_mounted_ceiling_fixture"] = dict(fixture)
    patched["pact_v98_contract_version"] = CONTRACT_VERSION
    patched["pact_v98_pendant_lateral_bow"] = bool(lateral_bow)
    patched["source_episode_id"] = row["episode_id"]
    patched["source_sampler_class"] = row.get("sampler_class")
    patched.pop("row_sha256", None)
    patched["row_sha256"] = sha256_payload(patched)
    return patched


def _pendant_bow(result: dict[str, Any]) -> dict[str, Any]:
    block = result.get("pendant_bow") or {}
    diagnostics = result.get("bow_diagnostics") or {}
    inbound = dict(
        block.get("inbound") or diagnostics.get("inbound_ceiling_fixture") or {}
    )
    outbound = dict(
        block.get("outbound") or diagnostics.get("outbound_ceiling_fixture") or {}
    )
    planned = inbound.get("planned_bow_m")
    accepted = inbound.get("accepted_bow_m")
    clipped = False
    if planned is not None and accepted is not None:
        clipped = float(accepted) + CLIP_EPS_M < float(planned)
    waypoint_y = inbound.get("waypoint_y_m")
    waypoint_side = inbound.get("waypoint_side")
    if waypoint_side is None and waypoint_y is not None:
        waypoint_side = -1.0 if float(waypoint_y) < 0.0 else 1.0
    return {
        "inbound_planned_bow_m": planned,
        "inbound_accepted_bow_m": accepted,
        "inbound_bow_fallback_taken": bool(inbound.get("bow_fallback_taken")),
        "inbound_waypoint_clipped": clipped,
        "inbound_waypoint_y_m": waypoint_y,
        "inbound_waypoint_side": waypoint_side,
        "outbound_planned_bow_m": outbound.get("planned_bow_m"),
        "outbound_accepted_bow_m": outbound.get("accepted_bow_m"),
        "outbound_bow_fallback_taken": bool(outbound.get("bow_fallback_taken")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--source-summary", type=Path, default=SOURCE_SUMMARY)
    parser.add_argument(
        "--half-y",
        type=float,
        help="Override pendant half-width (m). Rebuilds the fixture at x=0.72.",
    )
    parser.add_argument(
        "--center-y",
        type=float,
        help="Override pendant center y (m). Default is the v2 offset 0.100.",
    )
    parser.add_argument(
        "--bottom-z",
        type=float,
        default=PENDANT_BOTTOM_Z_BOUNDS_M[0],
        help="Pendant bottom z when --half-y is set (default 1.10).",
    )
    parser.add_argument(
        "--no-lateral-bow",
        action="store_true",
        help="Leave pact_v98_pendant_lateral_bow unset/false (bow-off control).",
    )
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    source = json.loads(args.source_summary.resolve().read_text())
    lateral_bow = not args.no_lateral_bow
    fixture = _pendant_fixture(
        half_y=args.half_y, bottom_z=args.bottom_z, center_y=args.center_y
    )
    baseline = {item["episode_id"]: item for item in source["results"]}
    rows = [
        _with_pendant(row, fixture, lateral_bow=lateral_bow)
        for row in source["manifest_rows"]
    ]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_sha256 = sha256_payload(
        {
            "schema_version": "pact_place_v9_8_v95_pendant_paired_v3",
            "fixture": fixture,
            "lateral_bow": lateral_bow,
            "half_y": args.half_y,
            "center_y": args.center_y,
            "bottom_z": args.bottom_z,
            "scene_xml": str(SCENE_XML.relative_to(ROOT)),
        }
    )
    context = multiprocessing.get_context("spawn")
    results: list[dict[str, Any]] = []
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
            row = futures[future]
            result = future.result()
            print(
                f"row={row['role_index']} side={row['intrusion_side']} "
                f"clean={result.get('clean_success')} status={result.get('status')}",
                flush=True,
            )
            results.append(result)
    results.sort(key=lambda item: int(item["role_index"]))
    rows.sort(key=lambda item: int(item["role_index"]))
    comparisons = []
    for row, result in zip(rows, results):
        expected = baseline[row["source_episode_id"]]
        bow = _pendant_bow(result)
        side = str(row["intrusion_side"])
        expected_bow_side = -1.0 if side == "left" else 1.0
        actual_side = bow.get("inbound_waypoint_side")
        wrong_way = (
            actual_side is not None
            and (float(actual_side) < 0.0) != (expected_bow_side < 0.0)
        )
        comparisons.append(
            {
                "role_index": row["role_index"],
                "family": row.get("layout_family_id") or row.get("family"),
                "intrusion_side": side,
                "task_seed_u64": row["task_seed_u64"],
                "baseline_clean_success": bool(expected["clean_success"]),
                "pendant_clean_success": bool(result.get("clean_success")),
                "pendant_task_success": bool(result.get("task_success")),
                "status": result.get("status"),
                "terminal_policy_phase": result.get("terminal_policy_phase"),
                "clutter_contacts": int(
                    (result.get("contact_audit") or {})
                    .get("contact_class_totals", {})
                    .get("clutter", 0)
                ),
                "mounted_fixture_contacts": int(
                    (result.get("contact_audit") or {})
                    .get("contact_class_totals", {})
                    .get("mounted_fixture", 0)
                ),
                "expected_inbound_bow_side": expected_bow_side,
                "inbound_bow_wrong_way": wrong_way,
                **bow,
            }
        )
    clean = sum(item["pendant_clean_success"] for item in comparisons)
    complete = [item for item in comparisons if item.get("status") == "complete"]
    inbound_planned = [
        float(item.get("inbound_planned_bow_m") or 0.0) for item in complete
    ]
    inbound_accepted = [
        float(item.get("inbound_accepted_bow_m") or 0.0) for item in complete
    ]
    any_clipped = any(
        item.get("inbound_waypoint_clipped") for item in complete
    )
    bow_took_effect = bool(inbound_planned) and all(
        value > CLIP_EPS_M for value in inbound_planned
    )
    baseline_clean_rows_preserved = all(
        (not item["baseline_clean_success"])
        or (
            bool(item["pendant_clean_success"])
            and int(item["mounted_fixture_contacts"]) == 0
            and item.get("status") == "complete"
        )
        for item in comparisons
    )
    any_wrong_way = any(item.get("inbound_bow_wrong_way") for item in complete)
    document = {
        "schema_version": "pact_place_v9_8_v95_pendant_paired_v3",
        "role": "pendant_cost_instrument_not_a_gate",
        "authorizes_collection": False,
        "authorizes_gate": False,
        "baseline_clean": BASELINE_CLEAN,
        "pendant_clean": clean,
        "pendant_matches_baseline": clean == BASELINE_CLEAN,
        "baseline_clean_rows_preserved": baseline_clean_rows_preserved,
        "lateral_bow": lateral_bow,
        "inbound_ceiling_fixture_bow_took_effect": bow_took_effect if lateral_bow else None,
        "inbound_waypoint_clipped_any": any_clipped,
        "inbound_bow_wrong_way_any": any_wrong_way,
        "inbound_ceiling_fixture_planned_bow_m": inbound_planned,
        "inbound_ceiling_fixture_accepted_bow_m": inbound_accepted,
        "n": len(comparisons),
        "fixture": fixture,
        "comparisons": comparisons,
    }
    path = output_root / "paired.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "pendant_clean": clean,
                "baseline_clean": BASELINE_CLEAN,
                "baseline_clean_rows_preserved": baseline_clean_rows_preserved,
                "lateral_bow": lateral_bow,
                "bow_took_effect": bow_took_effect if lateral_bow else None,
                "inbound_waypoint_clipped_any": any_clipped,
                "inbound_bow_wrong_way_any": any_wrong_way,
                "path": str(path),
            }
        )
    )
    if lateral_bow and not bow_took_effect:
        print(
            "VOID: lateral bow requested but inbound planned_bow_m is 0.0",
            flush=True,
        )
        return 2
    if not lateral_bow and any(value > CLIP_EPS_M for value in inbound_planned):
        print(
            "VOID: bow-off requested but inbound planned_bow_m is not 0.0",
            flush=True,
        )
        return 2
    if any_wrong_way:
        print("VOID: a complete row bowed the wrong way", flush=True)
        return 2
    if any_clipped:
        print("VOID: inbound waypoint clipped; row is uninformative", flush=True)
        return 2
    return 0 if all(item["status"] == "complete" for item in comparisons) else 2


if __name__ == "__main__":
    raise SystemExit(main())
