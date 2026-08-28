#!/usr/bin/env python3
"""Generate and run the V10.1 12-row empirical review pack.

Does not authorize the Phase-0 gate, collection, training, or evaluation.
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

from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v10_runtime import establish_v10_runtime_env, write_immutable  # noqa: E402
from pact_place_v101_empirical_qualification_contract import (  # noqa: E402
    CONTRACT_VERSION,
    N_REVIEW_ROWS,
    PLACE_V10_SCENE_SHA256,
    REVIEW_MASTER_SEED,
    REVIEW_STREAM,
    SAMPLER_CLASS,
    SCENE_XML_RELATIVE,
    V99_SNAPSHOT_RELATIVE,
    admit_six_cell_fixed_route,
    build_contract,
    empty_authorization,
    implementation_sha256,
    review_eligibility,
    sha256_payload,
    verify_protected_artifacts,
    is_v101_clean_success,
)
from pact_place_v99_exact import load_clean_snapshots  # noqa: E402
from run_pact_place_expert_screen import _result_path, run_row  # noqa: E402
from run_pact_place_v9_v1b_review import _render_review_video, clip_stem  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v101_empirical_review"
SCENE_XML = ROOT / SCENE_XML_RELATIVE
SNAPSHOT_ROOT = ROOT / Path(V99_SNAPSHOT_RELATIVE).parent


def _preflight() -> dict[str, Any]:
    protected = verify_protected_artifacts()
    _meta, cells = load_clean_snapshots(SNAPSHOT_ROOT)
    geometry = admit_six_cell_fixed_route(cells)
    return {
        "schema_version": "pact_place_v101_empirical_preflight_v1",
        "scene_sha256": PLACE_V10_SCENE_SHA256,
        "protected_artifacts": protected,
        "geometry_admitted": 12,
        "geometry_required": 12,
        "geometry": geometry,
        "snapshot_root": str(Path(V99_SNAPSHOT_RELATIVE).parent),
        "n_snapshot_cells": len(cells),
    }


def _render_job(row: dict[str, Any], result: dict[str, Any], output_root: Path) -> dict[str, Any]:
    stem = clip_stem(int(row["role_index"]) + 1, str(row["intrusion_side"]), result)
    family = str(row["layout_family_id"])
    video_path = output_root / "videos" / f"{int(row['role_index']):02d}_{family}_{stem}.mp4"
    result_path = _result_path(output_root, row)
    trajectory_path = result_path.parent / "trajectory.json"
    return _render_review_video(
        {
            "attempt": int(row["role_index"]) + 1,
            "row": row,
            "result_path": str(result_path),
            "trajectory_path": str(trajectory_path),
            "video_path": str(video_path),
            "clip_label": (
                f"v10.1 {family} {row['intrusion_side']} r{row['family_repeat']} "
                f"{result.get('status')}"
            ),
            "scene_xml": str(SCENE_XML),
            "sampler_class": SAMPLER_CLASS,
            "frame_stride": 2,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise SystemExit("workers must be in [1, 12]")
    establish_v10_runtime_env()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = _preflight()
    write_immutable(output_root / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps({"geometry_admitted": 12, "preflight": str(output_root / "preflight.json")}))
        return 0
    contract = build_contract()
    rows = list(contract["review_rows"])
    if len(rows) != N_REVIEW_ROWS:
        raise RuntimeError("review contract must contain exactly 12 rows")
    config = {
        "schema_version": "pact_place_v101_empirical_review_config_v1",
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "implementation_sha256": implementation_sha256(),
        "review_stream": REVIEW_STREAM,
        "review_master_seed": REVIEW_MASTER_SEED,
        "scene_xml": SCENE_XML_RELATIVE,
        "scene_sha256": PLACE_V10_SCENE_SHA256,
        "sampler_class": SAMPLER_CLASS,
        "n_rows": N_REVIEW_ROWS,
        **empty_authorization(),
        "expert_screen_rows": rows,
    }
    config["config_sha256"] = sha256_payload(config)
    (output_root / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
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
                config_sha256=config["config_sha256"],
                output_root=str(output_root),
                scene_xml=str(SCENE_XML),
            ): row
            for row in rows
        }
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"row={row['role_index']:02d} {row['layout_family_id']} "
                f"{row['intrusion_side']} status={result['status']} "
                f"clean={result.get('clean_success')}",
                flush=True,
            )
    results.sort(key=lambda item: int(item["role_index"]))
    videos = []
    for row, result in zip(rows, results):
        videos.append(_render_job(row, result, output_root))
    eligibility = review_eligibility(rows, results)
    manifest = {
        "schema_version": "pact_place_v101_empirical_review_manifest_v1",
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "config_sha256": config["config_sha256"],
        "implementation_sha256": implementation_sha256(),
        "review_stream": REVIEW_STREAM,
        "n_rows": N_REVIEW_ROWS,
        "rows": rows,
        "results": [
            {
                "role_index": item["role_index"],
                "episode_id": item["episode_id"],
                "status": item["status"],
                "task_success": item.get("task_success"),
                "clean_success": item.get("clean_success"),
                "v101_clean_success": is_v101_clean_success(item),
                "result_sha256": item.get("result_sha256"),
                "failure_cause": item.get("failure_cause"),
                "pendant_v10": item.get("pendant_v10"),
                "contact_audit": item.get("contact_audit"),
            }
            for item in results
        ],
        "videos": videos,
        "eligibility": eligibility,
        **empty_authorization(),
        "authorizes_gate": False,
        "authorizes_collection": False,
    }
    write_immutable(output_root / "review_manifest.json", manifest)
    summary = {
        "schema_version": "pact_place_v101_empirical_review_summary_v1",
        "n_rows": N_REVIEW_ROWS,
        "statuses": {item["status"]: 1 for item in results},
        "eligibility": eligibility,
        "authorizes_gate": False,
        "authorizes_collection": False,
        "videos": [item.get("video_path") for item in videos],
    }
    status_counts: dict[str, int] = {}
    for item in results:
        status_counts[str(item["status"])] = status_counts.get(str(item["status"]), 0) + 1
    summary["statuses"] = status_counts
    write_immutable(output_root / "summary.json", summary)
    print(json.dumps({"eligibility": eligibility, "output": str(output_root)}, indent=2))
    return 0 if eligibility["eligible_for_human_review"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
