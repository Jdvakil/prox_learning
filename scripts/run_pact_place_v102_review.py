#!/usr/bin/env python3
"""V10.2 Step-3 twelve-row human-review pack.

Runs only after the six-row screen passes. Two paired repeats per F0/F1/F2
family and side, every row rendered at true real time including failures.
Emits no approval file and infers no approval.
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
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v10_compound_pendant_contract import (  # noqa: E402
    PLACE_V10_SCENE_SHA256,
    SCENE_XML_RELATIVE,
)
from pact_place_v10_runtime import establish_v10_runtime_env, write_immutable  # noqa: E402
from pact_place_v102_raised_pendant_contract import (  # noqa: E402
    CONTRACT_VERSION,
    ENVIRONMENT_VERSION,
    N_REVIEW_ROWS,
    SAMPLER_CLASS,
    REVIEW_MASTER_SEED,
    REVIEW_STREAM,
    build_contract,
    empty_authorization,
    implementation_sha256,
    is_v102_clean_success,
    row_defects,
    review_eligibility,
    sha256_payload,
)
from run_pact_place_expert_screen import _result_path, run_row  # noqa: E402
from run_pact_place_v102_review_video import (  # noqa: E402
    clip_stem,
    render_v102_review_video,
)

DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v102_review"
DEFAULT_PREFLIGHT = (
    ROOT / "diagnostics_output" / "pact_place_v102_preflight" / "preflight.json"
)
DEFAULT_SCREEN_MANIFEST = (
    ROOT / "diagnostics_output" / "pact_place_v102_screen" / "screen_manifest.json"
)
SCENE_XML = ROOT / SCENE_XML_RELATIVE


def load_passed_artifact(path: Path, key: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required V10.2 artifact: {path}")
    document = json.loads(path.read_text())
    stored = document.get("artifact_sha256")
    payload = dict(document)
    payload.pop("artifact_sha256", None)
    from pact_place_v10_runtime import canonicalize

    if stored != sha256_payload(canonicalize(payload)):
        raise SystemExit(f"{path} self-hash mismatch")
    if not document.get(key):
        raise SystemExit(f"{path} reports {key}=false; V10.2 stops here")
    return document


def load_screen_manifest(path: Path) -> dict[str, Any]:
    document = load_passed_artifact(path, "eligibility")
    eligibility = document.get("eligibility") or {}
    if not eligibility.get("screen_passed"):
        raise SystemExit(
            f"{path} reports screen_passed=false; the V10.2 review is not admitted"
        )
    return document


def render_job(row: dict[str, Any], result: dict[str, Any], output_root: Path) -> dict[str, Any]:
    family = str(row["layout_family_id"])
    stem = clip_stem(int(row["role_index"]), str(row["intrusion_side"]), result)
    result_path = _result_path(output_root, row)
    return render_v102_review_video(
        {
            "role_index": int(row["role_index"]),
            "row": row,
            "result_path": str(result_path),
            "trajectory_path": str(result_path.parent / "trajectory.json"),
            "video_path": str(output_root / "videos" / f"{int(row['role_index']):02d}_{family}_{stem}.mp4"),
            "clip_label": (
                f"v10.2 review {family} {row['intrusion_side']} {result.get('status')}"
            ),
            "scene_xml": str(SCENE_XML),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--screen-manifest", type=Path, default=DEFAULT_SCREEN_MANIFEST)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise SystemExit("workers must be in [1, 12]")
    establish_v10_runtime_env()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    preflight = load_passed_artifact(args.preflight.resolve(), "preflight_passed")
    screen = load_screen_manifest(args.screen_manifest.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    contract = build_contract()
    if preflight.get("implementation_sha256") != contract["implementation_sha256"]:
        raise SystemExit(
            "preflight implementation hash does not match the current V10.2 "
            "implementation; the pack is invalidated and needs a new version"
        )
    rows = list(contract["review_rows"])
    if len(rows) != N_REVIEW_ROWS:
        raise RuntimeError("review contract must contain exactly 12 rows")
    config = {
        "schema_version": "pact_place_v102_review_config_v1",
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "implementation_sha256": implementation_sha256(),
        "preflight_sha256": preflight.get("artifact_sha256"),
        "screen_manifest_sha256": screen.get("artifact_sha256"),
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
        max_workers=args.workers, mp_context=context, max_tasks_per_child=1
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
                f"clean={is_v102_clean_success(result)}",
                flush=True,
            )
    results.sort(key=lambda item: int(item["role_index"]))
    videos = [render_job(row, result, output_root) for row, result in zip(rows, results)]
    eligibility = review_eligibility(rows, results)
    manifest = {
        "schema_version": "pact_place_v102_review_manifest_v1",
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "config_sha256": config["config_sha256"],
        "preflight_sha256": preflight.get("artifact_sha256"),
        "screen_manifest_sha256": screen.get("artifact_sha256"),
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
                "v102_clean_success": is_v102_clean_success(item),
                "v102_defects": row_defects(item),
                "result_sha256": item.get("result_sha256"),
                "failure_cause": item.get("failure_cause"),
                "pendant_v10": item.get("pendant_v10"),
                "pendant_frame_telemetry": item.get("pendant_frame_telemetry"),
                "contact_audit": item.get("contact_audit"),
            }
            for item in results
        ],
        "videos": videos,
        "eligibility": eligibility,
        **empty_authorization(),
    }
    digest = write_immutable(output_root / "review_manifest.json", manifest)
    status_counts: dict[str, int] = {}
    for item in results:
        status_counts[str(item["status"])] = status_counts.get(str(item["status"]), 0) + 1
    summary = {
        "schema_version": "pact_place_v102_review_summary_v1",
        "n_rows": N_REVIEW_ROWS,
        "statuses": status_counts,
        "eligibility": eligibility,
        "review_manifest_sha256": digest,
        "videos": [item.get("video_path") for item in videos],
        **empty_authorization(),
    }
    write_immutable(output_root / "summary.json", summary)
    print(json.dumps({"eligibility": eligibility, "output": str(output_root)}, indent=2))
    return 0 if eligibility["eligible_for_human_review"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
