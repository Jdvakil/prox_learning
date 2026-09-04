#!/usr/bin/env python3
"""Run one non-gating V9.3 expert smoke for every layout/panel-side variant."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v9_contract import (
    LAYOUT_FAMILIES,
    PALETTE_PATH,
    SAMPLER_CLASS,
    build_layout,
    load_palette,
    sha256_payload,
)
from run_pact_place_expert_screen import run_row

DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v93_panel_smoke"
SCENE_XML = (
    ROOT
    / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"
)
IMPLEMENTATION_FILES = (
    ROOT / "scripts/pact_place_v9_contract.py",
    ROOT / "scripts/run_pact_place_expert_screen.py",
    ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
)


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_FILES:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _row(
    *,
    index: int,
    family_id: str,
    side: str,
    palette_document: dict[str, Any],
    implementation_sha256: str,
    seed: int,
) -> dict[str, Any]:
    family_index = list(LAYOUT_FAMILIES).index(family_id)
    vessel_jitter = (
        ({"01": -0.015, "06": -0.004}, {"01": -0.004, "06": 0.009}),
        ({"01": -0.005, "06": 0.003}, {"01": 0.003, "06": -0.006}),
        ({"01": 0.006, "06": -0.002}, {"01": -0.002, "06": 0.0045}),
        ({"01": 0.015, "06": 0.004}, {"01": 0.004, "06": -0.009}),
    )[family_index]
    layout = build_layout(
        palette_document,
        family_id=family_id,
        intrusion_side=side,
    )
    episode_id = hashlib.sha256(
        f"pact-v9.3-panel-smoke:{implementation_sha256}:{family_id}:{side}:{seed}".encode()
    ).hexdigest()
    row = {
        "role_index": index,
        "episode_id": episode_id,
        "family": family_id,
        "layout_family_id": family_id,
        "layout_id": layout["layout_id"],
        "family_attempt": 0,
        "intrusion_side": side,
        "scene_template_house_index": 1,
        "task_seed_u32": int(seed),
        "task_seed_u64": int(seed),
        "max_sampling_retries": 12,
        "clutter_x_jitter_m": dict(vessel_jitter[0]),
        "clutter_y_jitter_m": dict(vessel_jitter[1]),
        "panel_face_jitter_m": 0.0,
        "panel_x_jitter_m": 0.0,
        "target_x_jitter_m": 0.0,
        "target_y_jitter_m": 0.0,
        "sampler_class": SAMPLER_CLASS,
        "pact_clutter_palette": list(palette_document["palette"]),
        "pact_clutter_layout": layout,
    }
    row["row_sha256"] = sha256_payload(row)
    return row


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    result = run_row(
        job["row"],
        config_sha256=job["config_sha256"],
        output_root=job["output_root"],
        scene_xml=str(SCENE_XML),
    )
    audit = result.get("contact_audit") or {}
    totals = audit.get("contact_class_totals") or {}
    return {
        "role_index": int(job["role_index"]),
        "family_id": str(job["family_id"]),
        "intrusion_side": str(job["side"]),
        "episode_id": result.get("episode_id"),
        "status": result.get("status"),
        "task_success": bool(result.get("task_success")),
        "clean_success": bool(result.get("clean_success")),
        "collision_free": audit.get("collision_free"),
        "hazard_bar_contacts": int(totals.get("hazard_bar", 0)),
        "clutter_contacts": int(totals.get("clutter", 0)),
        "clutter_stability_events": list(result.get("clutter_stability_events") or []),
        "bow_diagnostics": result.get("bow_diagnostics") or {},
        "clutter_x_jitter_m": dict(job["row"]["clutter_x_jitter_m"]),
        "clutter_y_jitter_m": dict(job["row"]["clutter_y_jitter_m"]),
        "paired_side_cell": str(job["family_id"]),
        "detected_hazards_are_geometry_proxy": True,
        "panel_active": (result.get("scene_params") or {}).get(
            "pact_v9_legacy_panel_active"
        ),
        "protrusion_present": (result.get("scene_params") or {}).get(
            "protrusion_present"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=804339)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--family", choices=list(LAYOUT_FAMILIES), action="append")
    parser.add_argument("--side", choices=("left", "right"), action="append")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("workers must be in [1, 4]")

    families = args.family or list(LAYOUT_FAMILIES)
    sides = args.side or ["left", "right"]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    palette_document = load_palette(PALETTE_PATH)
    implementation_sha256 = _implementation_sha256()
    config_sha256 = sha256_payload(
        {
            "schema_version": "pact_place_v9_3_panel_smoke_config_v1",
            "implementation_sha256": implementation_sha256,
            "seed": int(args.seed),
        }
    )

    jobs = []
    for family_index, family_id in enumerate(LAYOUT_FAMILIES):
        if family_id not in families:
            continue
        for side_index, side in enumerate(("left", "right")):
            if side not in sides:
                continue
            role_index = 200 + family_index * 2 + side_index
            row = _row(
                index=role_index,
                family_id=family_id,
                side=side,
                palette_document=palette_document,
                implementation_sha256=implementation_sha256,
                seed=int(args.seed),
            )
            jobs.append(
                {
                    "role_index": role_index,
                    "family_id": family_id,
                    "side": side,
                    "row": row,
                    "config_sha256": config_sha256,
                    "output_root": str(output_root),
                }
            )

    for job in jobs:
        print(f"Queued {job['family_id']} / panel {job['side']}", flush=True)
    results = []
    if args.workers == 1:
        for job in jobs:
            summary = _run_job(job)
            print(json.dumps(summary, sort_keys=True), flush=True)
            results.append(summary)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(_run_job, job): job for job in jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                summary = future.result()
                print(json.dumps(summary, sort_keys=True), flush=True)
                results.append(summary)
    results.sort(key=lambda item: int(item["role_index"]))

    pairs = {}
    for item in results:
        pairs.setdefault(item["family_id"], {})[item["intrusion_side"]] = item
    paired_side_ok = all(
        set(pair) == {"left", "right"}
        and pair["left"]["clutter_x_jitter_m"] == pair["right"]["clutter_x_jitter_m"]
        and pair["left"]["clutter_y_jitter_m"] == pair["right"]["clutter_y_jitter_m"]
        for pair in pairs.values()
    ) and len(pairs) == 4
    both_vessel_maneuvers_ok = all(
        float(item["bow_diagnostics"].get("inbound_vessel", {}).get("accepted_bow_m", 0.0)) > 0.0
        and float(item["bow_diagnostics"].get("outbound_vessel", {}).get("accepted_bow_m", 0.0)) > 0.0
        for item in results
    )
    clean_reachability_ok = len(results) == 8 and all(
        item["clean_success"]
        and item["clutter_contacts"] == 0
        and not item["clutter_stability_events"]
        and item["panel_active"] is True
        for item in results
    )
    inbound_bows = {
        round(float(item["bow_diagnostics"]["inbound_vessel"]["accepted_bow_m"]), 9)
        for item in results
        if "inbound_vessel" in item["bow_diagnostics"]
    }
    outbound_bows = {
        round(float(item["bow_diagnostics"]["outbound_vessel"]["accepted_bow_m"]), 9)
        for item in results
        if "outbound_vessel" in item["bow_diagnostics"]
    }
    clearance_response_varies = len(inbound_bows) > 1 and len(outbound_bows) > 1

    document = {
        "schema_version": "pact_place_v9_3_panel_smoke_summary_v1",
        "role": "development_smoke_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "development_admission": {
            "passed": bool(
                paired_side_ok
                and both_vessel_maneuvers_ok
                and clean_reachability_ok
                and clearance_response_varies
            ),
            "paired_side_geometry_identical": paired_side_ok,
            "both_vessel_maneuvers_present": both_vessel_maneuvers_ok,
            "clean_reachability": clean_reachability_ok,
            "clearance_response_varies": clearance_response_varies,
            "inbound_distinct_bow_count": len(inbound_bows),
            "outbound_distinct_bow_count": len(outbound_bows),
            "raw_proximity_required_next": True,
        },
        "implementation_sha256": implementation_sha256,
        "config_sha256": config_sha256,
        "seed": int(args.seed),
        "results": results,
        "counts": {
            "total": len(results),
            "task_success": sum(item["task_success"] for item in results),
            "clean_success": sum(item["clean_success"] for item in results),
            "panel_active": sum(item["panel_active"] is True for item in results),
        },
    }
    document["summary_sha256"] = sha256_payload(document)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
