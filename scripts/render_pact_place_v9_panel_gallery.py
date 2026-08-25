#!/usr/bin/env python3
"""Render representative V9.2 panel-active smoke trajectories."""

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

SMOKE_ROOT = ROOT / "diagnostics_output" / "pact_place_v9_panel_smoke"
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v9_panel_gallery"
SELECTION = (
    {
        "role_index": 200,
        "label": "SUCCESS  PANEL-LEFT  F0 TARGET-SIDE",
        "filename": "01_SUCCESS_panel_left_F0_target_side.mp4",
    },
    {
        "role_index": 205,
        "label": "SUCCESS  PANEL-RIGHT  F2 OUTER",
        "filename": "02_SUCCESS_panel_right_F2_outer.mp4",
    },
    {
        "role_index": 206,
        "label": "SUCCESS  PANEL-LEFT  F3 APERTURE-SIDE",
        "filename": "03_SUCCESS_panel_left_F3_aperture_side.mp4",
    },
)


def _row_from_result(result: dict[str, Any]) -> dict[str, Any]:
    scene = result["scene_params"]
    row = _make_review_row(
        int(result["role_index"]),
        str(result["intrusion_side"]),
        list(scene["pact_clutter_palette"]),
        dict(scene["pact_clutter_layout"]),
    )
    row.update(
        {
            "episode_id": result["episode_id"],
            "task_seed_u32": int(result["selected_seed"]["seed_u32"]),
            "task_seed_u64": int(result["selected_seed"]["seed_u64"]),
            "panel_x_jitter_m": float(result.get("panel_x_jitter_m") or 0.0),
            "panel_face_jitter_m": float(result.get("panel_face_jitter_m") or 0.0),
        }
    )
    row["row_sha256"] = sha256_payload(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frame-stride", type=int, default=4)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    smoke = json.loads((SMOKE_ROOT / "summary.json").read_text())
    result_by_role = {int(item["role_index"]): item for item in smoke["results"]}
    clips = []
    for attempt, spec in enumerate(SELECTION, start=1):
        summary = result_by_role[int(spec["role_index"])]
        if summary.get("clean_success") is not True or summary.get("panel_active") is not True:
            raise RuntimeError(f"selected role is not a clean panel-active smoke: {summary}")
        row_dir = (
            SMOKE_ROOT
            / "expert_screen_rows"
            / f"{int(spec['role_index']):02d}_{str(summary['episode_id'])[:16]}"
        )
        result_path = row_dir / "result.json"
        trajectory_path = row_dir / "trajectory.json"
        result = json.loads(result_path.read_text())
        print(f"[{attempt}/{len(SELECTION)}] Rendering {spec['filename']}", flush=True)
        rendered = _render_review_video(
            {
                "attempt": attempt,
                "row": _row_from_result(result),
                "result_path": str(result_path),
                "trajectory_path": str(trajectory_path),
                "video_path": str(output_root / spec["filename"]),
                "frame_stride": max(1, int(args.frame_stride)),
                "clip_label": spec["label"],
            }
        )
        clips.append(
            {
                **spec,
                "episode_id": result["episode_id"],
                "layout_family_id": result["scene_params"]["pact_clutter_layout"][
                    "layout_family_id"
                ],
                "intrusion_side": result["intrusion_side"],
                "panel_active": result["scene_params"]["pact_v9_legacy_panel_active"],
                "task_success": bool(result["task_success"]),
                "clean_success": bool(result["clean_success"]),
                "source_result_path": str(result_path.relative_to(ROOT)),
                "source_trajectory_path": str(trajectory_path.relative_to(ROOT)),
                **rendered,
            }
        )

    manifest = {
        "schema_version": "pact_place_v9_2_panel_gallery_v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "role": "visual_gallery_not_evaluation_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "source_smoke_summary": str((SMOKE_ROOT / "summary.json").relative_to(ROOT)),
        "selection_note": (
            "Three representative clean panel-active smokes are shown. The matched-seed smoke "
            "had 8/8 clean outcomes, so no current failure was invented for this gallery."
        ),
        "render": {
            "fps": 10,
            "frame_stride": max(1, int(args.frame_stride)),
            "panes": ["wrist_rgb", "third_person", "corridor_review"],
            "review_colors": {
                "active_panel": "original_material",
                "route_blocker": "orange",
            },
        },
        "counts": {"success": 3, "failure": 0, "total": 3},
        "clips": clips,
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
