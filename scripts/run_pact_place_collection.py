#!/usr/bin/env python3
"""Collect clean PACT place-corridor demonstrations after Phase 0 pass."""

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
MOLMO = ROOT / "submodules" / "molmospaces"
for search_path in (ROOT / "scripts", MOLMO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_collection_contract import (  # noqa: E402
    N_CANDIDATES,
    TARGET_CLEAN,
    YIELD_BURN_IN,
    load_collection_contract,
)
from pact_place_corridor_contract import sha256_payload  # noqa: E402
from run_pact_place_expert_screen import (  # noqa: E402
    _protected_eval_processes,
    _result_path,
    _validate_existing,
    place_receptacle_outside_placement,
    run_row,
    verify_protected_artifacts,
    write_json_atomic,
)

BATCH_SIZE = 24


def discard_record(result: dict[str, Any]) -> dict[str, Any]:
    tracking = result.get("terminal_tracking") or {}
    grasp = result.get("grasp_diagnostics") or {}
    scene = result.get("scene_params") or {}
    return {
        "role_index": result.get("role_index"),
        "episode_id": result.get("episode_id"),
        "status": result.get("status"),
        "clean_success": result.get("clean_success"),
        "task_success": result.get("task_success"),
        "grasp_phase_success": result.get("grasp_phase_success"),
        "place_phase_success": result.get("place_phase_success"),
        "failure_class": {
            "terminal_policy_phase": result.get("terminal_policy_phase"),
            "check_failure_branch": tracking.get("check_failure_branch"),
            "sequential_ik_failures": tracking.get("sequential_ik_failures"),
            "empty_gripper_streak": tracking.get("empty_gripper_streak"),
        },
        "grasp_pose": {
            "stock_grasp_world_position_m": grasp.get(
                "stock_grasp_world_position_m"
            ),
            "adjusted_grasp_world_position_m": grasp.get(
                "adjusted_grasp_world_position_m"
            ),
            "place_position_m": grasp.get("place_position_m"),
        },
        "panel_geometry": {
            "intrusion_side": result.get("intrusion_side"),
            "panel_x_jitter_m": result.get("panel_x_jitter_m"),
            "panel_face_jitter_m": result.get("panel_face_jitter_m"),
            "protr_center": scene.get("protr_center"),
            "protr_half": scene.get("protr_half"),
            "pact_panel_inner_face_y_m": scene.get("pact_panel_inner_face_y_m"),
            "cam_visible": scene.get("cam_visible"),
        },
        "clutter_slot_offsets": scene.get("pact_clutter"),
        "clutter_x_jitter_m": result.get("clutter_x_jitter_m"),
        "clutter_y_jitter_m": result.get("clutter_y_jitter_m"),
        "place_receptacle_outside_placement_entries": (
            place_receptacle_outside_placement(result.get("contact_audit") or {})
        ),
    }


def summarize_collection(
    contract: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    stopped_for_yield: bool,
) -> dict[str, Any]:
    complete = [item for item in results if item.get("status") == "complete"]
    kept = [item for item in results if item.get("clean_success") is True]
    discarded = [item for item in results if item.get("clean_success") is not True]
    n = len(results)
    clean = len(kept)
    screen_rate = float(contract["collection"]["screen_clean_rate"])
    running_rate = clean / n if n else None
    summary: dict[str, Any] = {
        "schema_version": "pact_place_collection_v5",
        "config_sha256": contract["config_sha256"],
        "screen_config_sha256": contract["screen_config_sha256"],
        "n_attempted": n,
        "n_candidates_frozen": N_CANDIDATES,
        "n_complete": len(complete),
        "clean_kept": clean,
        "discarded": len(discarded),
        "target_clean": TARGET_CLEAN,
        "running_clean_rate": running_rate,
        "screen_clean_rate": screen_rate,
        "yield_floor": contract["collection"]["yield_floor"],
        "stopped_for_yield": stopped_for_yield,
        "encoder_training_eval_authorized": False,
        "kept_role_indices": [item["role_index"] for item in kept],
        "discarded_role_indices": [item["role_index"] for item in discarded],
        "discard_log": [discard_record(item) for item in discarded],
    }
    if n:
        kept_sides = [item.get("intrusion_side") for item in kept]
        discarded_sides = [item.get("intrusion_side") for item in discarded]
        summary["limitation_kept_vs_discarded_side_counts"] = {
            "kept_left": kept_sides.count("left"),
            "kept_right": kept_sides.count("right"),
            "discarded_left": discarded_sides.count("left"),
            "discarded_right": discarded_sides.count("right"),
        }
    summary["collection_sha256"] = sha256_payload(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be in [1, 8]")
    protected = _protected_eval_processes()
    if protected:
        raise SystemExit(f"protected confirmatory evaluation is active: {protected}")
    contract = load_collection_contract(args.config)
    verify_protected_artifacts(contract)
    rows = contract["collection_rows"]
    scene_xml = str(ROOT / contract["scene"]["xml"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in rows:
        existing = _validate_existing(
            _result_path(args.output_root, row), row, contract["config_sha256"]
        )
        if existing is None:
            pending.append(row)
        else:
            results.append(existing)
    stopped_for_yield = False
    context = multiprocessing.get_context("spawn")
    floor = float(contract["collection"]["yield_floor"])
    while pending and not stopped_for_yield:
        batch = pending[:BATCH_SIZE]
        pending = pending[BATCH_SIZE:]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            max_tasks_per_child=1,
        ) as executor:
            futures = {
                executor.submit(
                    run_row,
                    row,
                    config_sha256=contract["config_sha256"],
                    output_root=str(args.output_root),
                    scene_xml=scene_xml,
                ): row
                for row in batch
            }
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                result = future.result()
                results.append(result)
                print(
                    f"row={row['role_index']:03d} side={row['intrusion_side']} "
                    f"status={result['status']} clean={result.get('clean_success')} "
                    f"grasp={result.get('grasp_phase_success')} "
                    f"place={result.get('place_phase_success')}",
                    flush=True,
                )
        n = len(results)
        clean = sum(item.get("clean_success") is True for item in results)
        rate = clean / n if n else 0.0
        print(
            f"progress attempted={n} kept={clean} rate={rate:.3f} floor={floor:.3f}",
            flush=True,
        )
        if n >= YIELD_BURN_IN and rate < floor:
            stopped_for_yield = True
            pending = []
    results.sort(key=lambda item: item["role_index"])
    verify_protected_artifacts(contract)
    summary = summarize_collection(
        contract, results, stopped_for_yield=stopped_for_yield
    )
    write_json_atomic(args.output_root / "collection.json", summary)
    discards = {
        "schema_version": "pact_place_collection_discards_v5",
        "config_sha256": contract["config_sha256"],
        "n": len(summary["discard_log"]),
        "rows": summary["discard_log"],
    }
    discards["discards_sha256"] = sha256_payload(discards)
    write_json_atomic(args.output_root / "discards.json", discards)
    print(json.dumps({
        "attempted": summary["n_attempted"],
        "kept": summary["clean_kept"],
        "discarded": summary["discarded"],
        "running_clean_rate": summary["running_clean_rate"],
        "stopped_for_yield": stopped_for_yield,
        "target_clean": TARGET_CLEAN,
    }, indent=2, sort_keys=True))
    if stopped_for_yield:
        return 2
    return 0 if summary["n_attempted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
