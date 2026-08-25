#!/usr/bin/env python3
"""Generate four non-gating V9.4 successes with wall/ceiling-mounted clutter."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v9_contract import (  # noqa: E402
    LAYOUT_FAMILIES,
    PALETTE_PATH,
    load_palette,
    sha256_payload,
)
from run_pact_place_expert_screen import _result_path, run_row  # noqa: E402
from run_pact_place_v9_panel_smoke import _row as _v93_row  # noqa: E402
from run_pact_place_v9_v1b_review import _render_review_video  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v94_mounted_preview"
SCENE_XML = (
    ROOT
    / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"
)
SAMPLER_CLASS = "PactPlaceCorridorV94MountedPreviewSampler"
RAW_VALIDATION = (
    ROOT
    / "diagnostics_output/pact_place_v93_v0c4_causal_proximity/validation.json"
)
IMPLEMENTATION_FILES = (
    ROOT / "scripts/run_pact_place_v94_mounted_preview.py",
    ROOT / "scripts/run_pact_place_expert_screen.py",
    ROOT / "scripts/run_pact_place_v9_v1b_review.py",
    ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    ROOT
    / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v3.xml",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_FILES:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_fixtures(*, wall_support: str, ceiling_side: int, seed: int) -> list[dict[str, Any]]:
    """Draw attached fixtures from bounded, visibly useful preview ranges."""
    if wall_support not in {"wall_left", "wall_right"}:
        raise ValueError(wall_support)
    if ceiling_side not in {-1, 1}:
        raise ValueError(ceiling_side)
    rng = np.random.default_rng(seed)

    wall_half = np.asarray(
        [rng.uniform(0.042, 0.052), rng.uniform(0.105, 0.125), rng.uniform(0.045, 0.060)]
    )
    wall_sign = 1.0 if wall_support == "wall_left" else -1.0
    wall_center = np.asarray(
        [rng.uniform(0.86, 0.96), wall_sign * (0.45 - wall_half[1]), rng.uniform(1.28, 1.36)]
    )

    ceiling_half = np.asarray(
        [rng.uniform(0.050, 0.062), rng.uniform(0.060, 0.078), rng.uniform(0.085, 0.110)]
    )
    ceiling_center = np.asarray(
        [
            rng.uniform(0.64, 0.71),
            ceiling_side * rng.uniform(0.055, 0.105),
            1.515 - ceiling_half[2],
        ]
    )
    return [
        {
            "fixture_id": "wall_block",
            "support": wall_support,
            "center_m": wall_center.round(9).tolist(),
            "half_m": wall_half.round(9).tolist(),
            "appearance": "neutral_matte_structure",
        },
        {
            "fixture_id": "ceiling_drop",
            "support": "ceiling",
            "center_m": ceiling_center.round(9).tolist(),
            "half_m": ceiling_half.round(9).tolist(),
            "appearance": "neutral_matte_structure",
        },
    ]


def build_row(
    *,
    cell_index: int,
    candidate: int,
    panel_side: str,
    wall_support: str,
    ceiling_side: int,
    palette_document: dict[str, Any],
    implementation_sha256: str,
    seed: int,
) -> dict[str, Any]:
    role_index = 400 + cell_index * 20 + candidate
    family_ids = tuple(LAYOUT_FAMILIES)
    family_id = family_ids[(cell_index + candidate) % len(family_ids)]
    candidate_seed = int(seed + cell_index * 1009 + candidate * 7919)
    row = _v93_row(
        index=role_index,
        family_id=family_id,
        side=panel_side,
        palette_document=palette_document,
        implementation_sha256=implementation_sha256,
        seed=candidate_seed,
    )
    row.update(
        {
            "sampler_class": SAMPLER_CLASS,
            "preview_cell_index": cell_index,
            "preview_candidate": candidate,
            "mounted_wall_support": wall_support,
            "mounted_ceiling_side": "+y" if ceiling_side > 0 else "-y",
            "pact_mounted_fixtures": build_fixtures(
                wall_support=wall_support,
                ceiling_side=ceiling_side,
                seed=candidate_seed ^ 0x94CE1,
            ),
        }
    )
    row["episode_id"] = hashlib.sha256(
        (
            f"pact-v9.4-mounted:{implementation_sha256}:{cell_index}:"
            f"{candidate}:{candidate_seed}"
        ).encode()
    ).hexdigest()
    row.pop("row_sha256", None)
    row["row_sha256"] = sha256_payload(row)
    return row


def _summary(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    totals = ((result.get("contact_audit") or {}).get("contact_class_totals") or {})
    bows = result.get("bow_diagnostics") or {}
    mounted_bow_m = max(
        (
            float(record.get("accepted_bow_m", 0.0))
            for prefix, record in bows.items()
            if "wall_fixture" in prefix or "ceiling_fixture" in prefix
        ),
        default=0.0,
    )
    return {
        "cell_index": row["preview_cell_index"],
        "candidate": row["preview_candidate"],
        "episode_id": row["episode_id"],
        "panel_side": row["intrusion_side"],
        "wall_support": row["mounted_wall_support"],
        "ceiling_side": row["mounted_ceiling_side"],
        "fixtures": row["pact_mounted_fixtures"],
        "status": result.get("status"),
        "task_success": bool(result.get("task_success")),
        "clean_success": bool(result.get("clean_success")),
        "clutter_contacts": int(totals.get("clutter", 0)),
        "hazard_bar_contacts": int(totals.get("hazard_bar", 0)),
        "other_environment_contacts": int(totals.get("other_environment", 0)),
        "failure_cause": result.get("failure_cause"),
        "bow_diagnostics": bows,
        "mounted_fixture_bow_m": mounted_bow_m,
        "episode_steps": result.get("episode_steps"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=940024)
    parser.add_argument("--max-candidates-per-cell", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    if args.max_candidates_per_cell < 1:
        raise ValueError("max-candidates-per-cell must be positive")

    output_root = args.output_root.resolve()
    video_root = output_root / "videos"
    output_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)
    palette_document = load_palette(PALETTE_PATH)
    implementation_sha256 = _implementation_sha256()
    config_sha256 = sha256_payload(
        {
            "schema_version": "pact_place_v9_4_mounted_preview_config_v1",
            "implementation_sha256": implementation_sha256,
            "seed": int(args.seed),
            "role": "development_preview_not_a_gate",
        }
    )
    # The four cells balance panel and wall support; ceiling side is crossed so
    # fixture placement cannot encode the panel label.
    cells = (
        ("left", "wall_left", 1),
        ("left", "wall_right", -1),
        ("right", "wall_left", -1),
        ("right", "wall_right", 1),
    )
    attempts: list[dict[str, Any]] = []
    accepted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for cell_index, (panel_side, wall_support, ceiling_side) in enumerate(cells):
        for candidate in range(args.max_candidates_per_cell):
            row = build_row(
                cell_index=cell_index,
                candidate=candidate,
                panel_side=panel_side,
                wall_support=wall_support,
                ceiling_side=ceiling_side,
                palette_document=palette_document,
                implementation_sha256=implementation_sha256,
                seed=args.seed,
            )
            result = run_row(
                row,
                config_sha256=config_sha256,
                output_root=str(output_root),
                scene_xml=str(SCENE_XML),
            )
            item = _summary(row, result)
            attempts.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)
            if item["clean_success"] and item["clutter_contacts"] == 0:
                accepted.append((row, result))
                break
        else:
            break

    videos: list[dict[str, Any]] = []
    if len(accepted) == 4 and not args.skip_render:
        for display_index, (row, result) in enumerate(accepted, 1):
            result_path = _result_path(output_root, row)
            video_path = video_root / (
                f"success_{display_index}_panel-{row['intrusion_side']}_"
                f"{row['mounted_wall_support']}_{row['mounted_ceiling_side']}.mp4"
            )
            videos.append(
                _render_review_video(
                    {
                        "attempt": display_index,
                        "row": row,
                        "result_path": str(result_path),
                        "trajectory_path": str(result["trajectory_path"]),
                        "video_path": str(video_path),
                        "frame_stride": int(args.frame_stride),
                        "clip_label": "V9.4 MOUNTED CLUTTER — NON-GATING PREVIEW",
                    }
                )
            )

    raw_validation = None
    if RAW_VALIDATION.is_file():
        raw_validation = json.loads(RAW_VALIDATION.read_text()).get("passed")
    manifest = {
        "schema_version": "pact_place_v9_4_mounted_preview_manifest_v1",
        "created_at": _utc_now(),
        "role": "development_preview_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "raw_proximity_validation_passed": raw_validation,
        "environment_version": "pact_place_corridor_v9_4_mounted_preview",
        "sampler_class": SAMPLER_CLASS,
        "implementation_sha256": implementation_sha256,
        "config_sha256": config_sha256,
        "requested_successes": 4,
        "clean_successes": len(accepted),
        "mounted_fixture_response_successes": sum(
            max(
                (
                    float(record.get("accepted_bow_m", 0.0))
                    for prefix, record in (result.get("bow_diagnostics") or {}).items()
                    if "wall_fixture" in prefix or "ceiling_fixture" in prefix
                ),
                default=0.0,
            )
            > 0.0
            for _, result in accepted
        ),
        "balanced_panel_wall_matrix": len(accepted) == 4,
        "attempts": attempts,
        "accepted_episode_ids": [row["episode_id"] for row, _ in accepted],
        "videos": videos,
    }
    manifest_path = output_root / "preview_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest={manifest_path}", flush=True)
    return 0 if len(accepted) == 4 and (args.skip_render or len(videos) == 4) else 1


if __name__ == "__main__":
    raise SystemExit(main())
