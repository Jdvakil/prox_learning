#!/usr/bin/env python3
"""Render a compact, explicitly non-gating gallery from V9.1 smoke trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v9_contract import sha256_payload
from run_pact_place_v9_v1b_review import _make_review_row, _render_review_video

SOURCE_ROOT = (
    ROOT / "diagnostics_output" / "pact_place_v9_smoke_redesign" / "expert_screen_rows"
)
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v9_visual_gallery"

CLIPS = (
    {
        "role_index": 117,
        "kind": "success",
        "label": "SUCCESS  F0 NEAR-TARGET +Y",
        "filename": "01_SUCCESS_F0_near_target_plus_y.mp4",
        "interpretation": "Clean success; near-target lower blocker, +y avoidance route.",
        "code_status": "current_post_fix",
    },
    {
        "role_index": 118,
        "kind": "success",
        "label": "SUCCESS  F1 MID-UPPER -Y",
        "filename": "02_SUCCESS_F1_mid_upper_minus_y.mp4",
        "interpretation": "Clean success; mid-route upper blocker, -y avoidance route.",
        "code_status": "current_post_fix",
    },
    {
        "role_index": 121,
        "kind": "success",
        "label": "SUCCESS  F3 APERTURE-UPPER -Y",
        "filename": "03_SUCCESS_F3_aperture_upper_minus_y.mp4",
        "interpretation": "Clean success; aperture upper blocker, -y avoidance route.",
        "code_status": "current_post_fix",
    },
    {
        "role_index": 103,
        "kind": "failure",
        "label": "FAIL  PRE-FIX BLOCKER STRIKE",
        "filename": "04_FAIL_pre_fix_blocker_strike.mp4",
        "interpretation": (
            "Pre-fix failure: inbound motion contacted and displaced the route-blocking soap bottle."
        ),
        "code_status": "pre_inbound_bow_fix",
    },
    {
        "role_index": 101,
        "kind": "failure",
        "label": "FAIL  DISCARDED-PALETTE DRIFT",
        "filename": "05_FAIL_discarded_palette_drift.mp4",
        "interpretation": (
            "Development failure: placement succeeded, but two objects in a discarded palette drifted."
        ),
        "code_status": "discarded_intermediate_palette",
    },
    {
        "role_index": 120,
        "kind": "failure",
        "label": "FAIL  CURRENT GRASP MISS",
        "filename": "06_FAIL_current_grasp_miss.mp4",
        "interpretation": (
            "Current-layout failure: collision-free approach, then empty-gripper detection during lift."
        ),
        "code_status": "current_post_fix",
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_dir(role_index: int) -> Path:
    matches = sorted(SOURCE_ROOT.glob(f"{role_index}_*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one source directory for role {role_index}, found {matches}")
    return matches[0]


def _row_from_result(result: dict[str, Any]) -> dict[str, Any]:
    scene_params = result["scene_params"]
    row = _make_review_row(
        int(result["role_index"]),
        str(result["intrusion_side"]),
        list(scene_params["pact_clutter_palette"]),
        dict(scene_params["pact_clutter_layout"]),
    )
    row.update(
        {
            "episode_id": result["episode_id"],
            "task_seed_u32": int(result["selected_seed"]["seed_u32"]),
            "task_seed_u64": int(result["selected_seed"]["seed_u64"]),
        }
    )
    row["row_sha256"] = sha256_payload({key: value for key, value in row.items() if key != "row_sha256"})
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=3,
        help="Render every Nth trajectory frame; final frame is always retained.",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rendered = []
    for attempt, spec in enumerate(CLIPS, start=1):
        source_dir = _source_dir(int(spec["role_index"]))
        result_path = source_dir / "result.json"
        trajectory_path = source_dir / "trajectory.json"
        result = json.loads(result_path.read_text())
        expected_success = spec["kind"] == "success"
        if bool(result.get("clean_success")) != expected_success:
            raise RuntimeError(
                f"Role {spec['role_index']} outcome changed: clean_success={result.get('clean_success')}"
            )

        print(f"[{attempt}/{len(CLIPS)}] Rendering {spec['filename']}", flush=True)
        render_info = _render_review_video(
            {
                "attempt": attempt,
                "row": _row_from_result(result),
                "result_path": str(result_path),
                "trajectory_path": str(trajectory_path),
                "video_path": str(output_root / str(spec["filename"])),
                "frame_stride": args.frame_stride,
                "clip_label": spec["label"],
            }
        )
        rendered.append(
            {
                **spec,
                "episode_id": result["episode_id"],
                "layout_family_id": result["scene_params"]["pact_clutter_layout"][
                    "layout_family_id"
                ],
                "intrusion_side": result["intrusion_side"],
                "task_success": bool(result.get("task_success")),
                "clean_success": bool(result.get("clean_success")),
                "terminal_policy_phase": result.get("terminal_policy_phase"),
                "source_result_path": str(result_path.relative_to(ROOT)),
                "source_trajectory_path": str(trajectory_path.relative_to(ROOT)),
                **render_info,
            }
        )

    manifest = {
        "schema_version": "pact_place_v9_visual_gallery_v1",
        "created_at": _utc_now(),
        "purpose": "Human-readable visualization of three successes and three real development failures.",
        "role": "visual_gallery_not_evaluation_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "failure_selection_note": (
            "The failure set is intentionally heterogeneous and explicitly versioned; it is not an estimate "
            "of the current policy failure distribution."
        ),
        "render": {
            "fps": 10,
            "frame_stride": max(1, int(args.frame_stride)),
            "panes": ["wrist_rgb", "third_person", "corridor_review"],
        },
        "counts": {"success": 3, "failure": 3, "total": 6},
        "clips": rendered,
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
