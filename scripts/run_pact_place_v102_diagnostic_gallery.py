#!/usr/bin/env python3
"""V10.2 diagnostic gallery — owner-requested, explicitly outside the gate.

The registered V10.2 review pack was never run: Step 0 failed items 5, 6 and 7
and `docs/PACT_PLACE_V102_RAISED_PENDANT_REMEDIATION_PLAN.md` stops there. The
owner asked to see the twelve clips anyway, so this script runs the twelve
registered review rows and renders them at real time into a *separate*
diagnostic path.

What this is NOT:

- not `diagnostics_output/pact_place_v102_review/`, and not a review manifest;
- not eligible for human review, and never citable as one;
- not an input to causal proximity or Phase 0;
- not authorization for anything. Every authorization stays false, and the
  Step-0 stop artifact is untouched.

It is a look at what the raised pendant actually does to the arm. Expect the
overlay to show large negative clearances: preflight item 6 measured 56-77 mm
of lobe interpenetration on all twelve cell/direction cases, and item 7 showed
MuJoCo cannot generate pendant contacts for the runtime-posed assembly, so the
arm passes through the geometry without physical resistance.
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
    REVIEW_MASTER_SEED,
    REVIEW_STREAM,
    SAMPLER_CLASS,
    build_contract,
    empty_authorization,
    implementation_sha256,
    is_v102_clean_success,
    row_defects,
    sha256_payload,
)
import numpy as np  # noqa: E402

import run_pact_place_v102_review_video as review_video  # noqa: E402
from run_pact_place_expert_screen import _result_path, run_row  # noqa: E402
from run_pact_place_v102_review_video import (  # noqa: E402
    clip_stem,
    render_v102_review_video,
)

# The registered renderer's pendant pane sits at y = -1.15, outside the hood,
# so hood_side_r occludes it: the pane shows a wall. This gallery overrides the
# camera in-process only. No file in the V10.2 implementation hash is touched,
# and the override is recorded in the manifest rather than applied silently.
PENDANT_CAM_OVERRIDE = {
    "reason": (
        "registered pendant side camera at y=-1.15 m is outside the hood and "
        "occluded by hood_side_r; this pose is inside the aperture and frames "
        "both lobes, both stems, the crossbar and the arm"
    ),
    "pos_m": [0.20, -0.07, 1.42],
    "target_m": [0.70, 0.01, 1.20],
    "fov_deg": 44.0,
    "registered_pos_m": [0.70, -1.15, 1.30],
    "registered_target_m": [0.70, 0.05, 1.24],
    "registered_fov_deg": 45.0,
}


def _apply_camera_override() -> None:
    review_video.PENDANT_SIDE_CAM_POS = np.asarray(
        PENDANT_CAM_OVERRIDE["pos_m"], dtype=float
    )
    review_video.PENDANT_SIDE_CAM_TARGET = np.asarray(
        PENDANT_CAM_OVERRIDE["target_m"], dtype=float
    )
    review_video.PENDANT_SIDE_CAM_FOV = float(PENDANT_CAM_OVERRIDE["fov_deg"])

DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v102_diagnostic_gallery"
DEFAULT_PREFLIGHT = (
    ROOT / "diagnostics_output" / "pact_place_v102_preflight" / "preflight.json"
)
SCENE_XML = ROOT / SCENE_XML_RELATIVE
NOT_A_GATE = "owner_requested_diagnostic_gallery_not_the_registered_review"


def _render(row: dict[str, Any], result: dict[str, Any], output_root: Path) -> dict[str, Any]:
    family = str(row["layout_family_id"])
    stem = clip_stem(int(row["role_index"]), str(row["intrusion_side"]), result)
    result_path = _result_path(output_root, row)
    return render_v102_review_video(
        {
            "role_index": int(row["role_index"]),
            "row": row,
            "result_path": str(result_path),
            "trajectory_path": str(result_path.parent / "trajectory.json"),
            "video_path": str(
                output_root / "videos" / f"{int(row['role_index']):02d}_{family}_{stem}.mp4"
            ),
            "clip_label": (
                f"v10.2 DIAGNOSTIC (not the gate) {family} {row['intrusion_side']} "
                f"{result.get('status')}"
            ),
            "scene_xml": str(SCENE_XML),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise SystemExit("workers must be in [1, 12]")
    establish_v10_runtime_env()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")

    preflight = json.loads(args.preflight.resolve().read_text())
    if preflight.get("preflight_passed"):
        raise SystemExit(
            "preflight passed: use the registered review runner, not this diagnostic"
        )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    contract = build_contract()
    rows = list(contract["review_rows"])
    if len(rows) != N_REVIEW_ROWS:
        raise RuntimeError("expected exactly 12 registered review rows")

    config = {
        "schema_version": "pact_place_v102_diagnostic_gallery_config_v1",
        "role": NOT_A_GATE,
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "implementation_sha256": implementation_sha256(),
        "preflight_sha256": preflight.get("artifact_sha256"),
        "preflight_passed": bool(preflight.get("preflight_passed")),
        "preflight_failures": preflight.get("failures"),
        "review_stream": REVIEW_STREAM,
        "review_master_seed": REVIEW_MASTER_SEED,
        "scene_xml": SCENE_XML_RELATIVE,
        "scene_sha256": PLACE_V10_SCENE_SHA256,
        "sampler_class": SAMPLER_CLASS,
        "n_rows": N_REVIEW_ROWS,
        "is_registered_review": False,
        "eligible_for_human_review": False,
        **empty_authorization(),
        "expert_screen_rows": rows,
    }
    config["config_sha256"] = sha256_payload(config)
    (output_root / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    results: list[dict[str, Any]] = []
    if args.render_only:
        for row in rows:
            results.append(json.loads(_result_path(output_root, row).read_text()))
    else:
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
                frames = result.get("pendant_frame_telemetry") or {}
                print(
                    f"row={row['role_index']:02d} {row['layout_family_id']} "
                    f"{row['intrusion_side']} status={result['status']} "
                    f"clean={is_v102_clean_success(result)} "
                    f"min_clearance={frames.get('min_clearance_m')} "
                    f"contact_frames={frames.get('live_pendant_contact_frames')}",
                    flush=True,
                )
    results.sort(key=lambda item: int(item["role_index"]))

    _apply_camera_override()
    videos = []
    for row, result in zip(rows, results):
        videos.append(_render(row, result, output_root))
        print(f"rendered {videos[-1]['video_path']}", flush=True)

    manifest = {
        "schema_version": "pact_place_v102_diagnostic_gallery_manifest_v1",
        "role": NOT_A_GATE,
        "is_registered_review": False,
        "eligible_for_human_review": False,
        "preflight_passed": False,
        "preflight_sha256": preflight.get("artifact_sha256"),
        "preflight_failures": preflight.get("failures"),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "config_sha256": config["config_sha256"],
        "implementation_sha256": implementation_sha256(),
        "n_rows": N_REVIEW_ROWS,
        "results": [
            {
                "role_index": item["role_index"],
                "episode_id": item["episode_id"],
                "family": item.get("scene_params", {}).get("pact_clutter_layout", {}).get(
                    "layout_family_id"
                ),
                "intrusion_side": item.get("intrusion_side"),
                "status": item["status"],
                "task_success": item.get("task_success"),
                "clean_success": item.get("clean_success"),
                "v102_clean_success": is_v102_clean_success(item),
                "v102_defects": row_defects(item),
                "failure_cause": item.get("failure_cause"),
                "pendant_frame_telemetry": item.get("pendant_frame_telemetry"),
                "pendant_v10": item.get("pendant_v10"),
                "contact_audit": item.get("contact_audit"),
                "result_sha256": item.get("result_sha256"),
            }
            for item in results
        ],
        "videos": videos,
        "pendant_side_camera_override": PENDANT_CAM_OVERRIDE,
        **empty_authorization(),
    }
    digest = write_immutable(output_root / "gallery_manifest.json", manifest)
    print(
        json.dumps(
            {
                "role": NOT_A_GATE,
                "n_videos": len(videos),
                "gallery_manifest_sha256": digest,
                "output": str(output_root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
