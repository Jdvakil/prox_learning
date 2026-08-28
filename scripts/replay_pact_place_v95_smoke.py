#!/usr/bin/env python3
"""Replay the stored V9.5 8-row smoke and require a row-for-row 6/8 match.

Passing ``scene_xml`` is mandatory. Omitting it falls back to
``pact_place_corridor_v1.xml`` and every row fails with
``KeyError: pact_clutter_l0``.
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
from run_pact_place_expert_screen import run_row  # noqa: E402

SOURCE_SUMMARY = ROOT / "diagnostics_output/pact_place_v95_raw_smoke/summary.json"
SCENE_XML = ROOT / (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v5.xml"
)
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output/pact_place_v95_smoke_repro_guard"
REQUIRED_CLEAN = 6
REQUIRED_ROWS = 8


def _compare(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": expected["episode_id"],
        "family_id": expected["family_id"],
        "intrusion_side": expected["intrusion_side"],
        "expected_clean_success": bool(expected["clean_success"]),
        "replay_clean_success": bool(actual.get("clean_success")),
        "expected_task_success": bool(expected.get("task_success")),
        "replay_task_success": bool(actual.get("task_success")),
        "replay_status": actual.get("status"),
        "row_match": bool(expected["clean_success"]) == bool(actual.get("clean_success")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--source-summary", type=Path, default=SOURCE_SUMMARY)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    source = json.loads(args.source_summary.resolve().read_text())
    rows = list(source["manifest_rows"])
    expected = {item["episode_id"]: item for item in source["results"]}
    if len(rows) != REQUIRED_ROWS or len(expected) != REQUIRED_ROWS:
        raise SystemExit("stored V9.5 smoke is not the frozen 8-row artifact")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_sha256 = sha256_payload(
        {
            "schema_version": "pact_place_v9_5_smoke_repro_guard_v1",
            "source_summary": str(args.source_summary),
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
    comparisons = [
        _compare(expected[item["episode_id"]], item) for item in results
    ]
    clean = sum(item["replay_clean_success"] for item in comparisons)
    matched = sum(item["row_match"] for item in comparisons)
    passed = (
        clean == REQUIRED_CLEAN
        and matched == REQUIRED_ROWS
        and all(item["replay_status"] == "complete" for item in comparisons)
    )
    document = {
        "schema_version": "pact_place_v9_5_smoke_repro_guard_v1",
        "role": "regression_guard_not_a_gate",
        "authorizes_collection": False,
        "passed": passed,
        "required_clean": REQUIRED_CLEAN,
        "replay_clean": clean,
        "row_matches": matched,
        "n": REQUIRED_ROWS,
        "comparisons": comparisons,
    }
    path = output_root / "guard.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "clean": clean, "row_matches": matched, "path": str(path)}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
