#!/usr/bin/env python3
"""Render the mandatory V9.8 human-review sample when the expert gate permits it."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_corridor_contract import sha256_file, sha256_payload  # noqa: E402
from run_pact_place_v9_v1b_review import _render_review_video  # noqa: E402

DEFAULT_SOURCE_ROOT = ROOT / "diagnostics_output/pact_place_v98_expert_gate"
DEFAULT_CONFIGURATION = ROOT / "configs/pact_place_corridor_v98.json"
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output/pact_place_v98_pendant_review"


def _result_rows(source_root: Path, configuration: dict[str, Any]) -> list[dict[str, Any]]:
    configured = {
        int(row["role_index"]): row
        for row in list(configuration.get("expert_screen_rows") or [])
    }
    rows = []
    for result_path in sorted((source_root / "expert_screen_rows").glob("*/result.json")):
        result = json.loads(result_path.read_text())
        row = configured.get(int(result.get("role_index", -1)))
        if row is None:
            continue
        rows.append(
            {
                "row": row,
                "result": result,
                "result_path": str(result_path),
                "trajectory_path": str(result_path.parent / "trajectory.json"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--configuration", type=Path, default=DEFAULT_CONFIGURATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    configuration = json.loads(args.configuration.resolve().read_text())
    rows = _result_rows(source_root, configuration)
    successes = [item for item in rows if item["result"].get("clean_success") is True]
    failures = [item for item in rows if item["result"].get("clean_success") is not True]
    if len(successes) < 3 or len(failures) < 3:
        document = {
            "schema_version": "pact_place_v9_8_pendant_review_v1",
            "role": "mandatory_human_review_blocked",
            "authorizes_gate": False,
            "authorizes_collection": False,
            "source_root": str(source_root.relative_to(ROOT)),
            "success_count": len(successes),
            "failure_count": len(failures),
            "reason": "S3 expert gate did not provide three successes and three failures",
        }
        document["document_sha256"] = sha256_payload(document)
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "review.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(document, sort_keys=True))
        return 2

    selected = successes[:3] + failures[:3]
    video_root = args.output_root.resolve() / "videos"
    jobs = []
    for index, item in enumerate(selected):
        jobs.append(
            {
                "attempt": index,
                "row": item["row"],
                "result_path": item["result_path"],
                "trajectory_path": item["trajectory_path"],
                "video_path": str(video_root / f"{item['row']['role_index']:02d}_{'success' if index < 3 else 'failure'}.mp4"),
                "frame_stride": 3,
            }
        )
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        videos = list(executor.map(_render_review_video, jobs))
    document = {
        "schema_version": "pact_place_v9_8_pendant_review_v1",
        "role": "mandatory_human_review_pending",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "source_root": str(source_root.relative_to(ROOT)),
        "videos": videos,
        "human_verdict_required": True,
    }
    document["document_sha256"] = sha256_payload(document)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "review.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    print(args.output_root / "review.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
