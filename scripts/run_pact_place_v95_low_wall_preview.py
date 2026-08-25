#!/usr/bin/env python3
"""Generate the gated four-cell V9.5 low-wall human-review preview."""

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

from pact_place_v9_contract import LAYOUT_FAMILIES, sha256_payload
from pact_place_v95_contract import build_v95_layout, load_v95_palette
from run_pact_place_expert_screen import _result_path, run_row
from run_pact_place_v9_panel_smoke import _row as _v93_row
from run_pact_place_v9_v1b_review import _render_review_video

DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output/pact_place_v95_low_wall_preview"
RAW_PREREQUISITE = ROOT / "diagnostics_output/pact_place_v95_v0c5_raw_prerequisite/validation.json"
SCENE_XML = ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"
SAMPLER_CLASS = "PactPlaceCorridorV95LowWallSampler"
MIN_FIXTURE_BOW_M = 0.040
MIN_REALIZED_CLEARANCE_M = 0.020


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_wall_fixture(*, support: str, seed: int) -> dict[str, Any]:
    """Draw one attached fixture wholly inside the measured low-wall band."""
    if support not in {"wall_left", "wall_right"}:
        raise ValueError(support)
    rng = np.random.default_rng(seed)
    half_x = float(rng.uniform(0.040, 0.050))
    half_y = float(rng.uniform(0.145, 0.175))
    x = float(rng.uniform(0.64 + half_x, 0.81 - half_x))
    bottom = float(rng.uniform(0.87, 0.95))
    top = float(rng.uniform(max(1.06, bottom + 0.14), 1.15))
    side = 1.0 if support == "wall_left" else -1.0
    half_z = (top - bottom) / 2.0
    return {
        "fixture_id": "low_wall_block",
        "support": support,
        "center_m": [x, side * (0.45 - half_y), (bottom + top) / 2.0],
        "half_m": [half_x, half_y, half_z],
        "bottom_z_m": bottom,
        "top_z_m": top,
        "appearance": "neutral_matte_structure",
        "siting_contract": "link5_link6_low_wall_band",
    }


def build_row(
    *,
    cell_index: int,
    candidate: int,
    panel_side: str,
    wall_support: str,
    palette_document: dict[str, Any],
    implementation_sha256: str,
    seed: int,
) -> dict[str, Any]:
    family_ids = tuple(LAYOUT_FAMILIES)
    family_id = family_ids[(cell_index + candidate) % len(family_ids)]
    candidate_seed = int(seed + (cell_index // 2) * 1009 + candidate * 7919)
    row = _v93_row(
        index=500 + cell_index * 20 + candidate,
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
            "pact_mounted_wall_fixture": build_wall_fixture(
                support=wall_support,
                # Geometry depends on wall support/candidate, never panel side.
                seed=int(seed + (0 if wall_support == "wall_left" else 1) * 1009 + candidate * 7919),
            ),
        }
    )
    row["pact_clutter_palette"] = list(palette_document["palette"])
    row["pact_clutter_layout"] = build_v95_layout(
        palette_document, family_id=family_id, intrusion_side=panel_side
    )
    row["layout_id"] = row["pact_clutter_layout"]["layout_id"]
    row["episode_id"] = hashlib.sha256(
        f"pact-v9.5-low-wall:{implementation_sha256}:{cell_index}:{candidate}:{candidate_seed}".encode()
    ).hexdigest()
    row.pop("row_sha256", None)
    row["row_sha256"] = sha256_payload(row)
    return row


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        ROOT / "scripts/run_pact_place_expert_screen.py",
        ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
        ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v3.xml",
    ):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw-prerequisite", type=Path, default=RAW_PREREQUISITE)
    parser.add_argument("--seed", type=int, default=950024)
    parser.add_argument("--max-candidates-per-cell", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=3)
    args = parser.parse_args()

    raw_path = args.raw_prerequisite.resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"V9.5 is blocked until raw prerequisite exists: {raw_path}")
    raw = json.loads(raw_path.read_text())
    if raw.get("passed") is not True:
        raise RuntimeError("V9.5 is blocked because paired-side raw vessel admission failed")

    output_root = args.output_root.resolve()
    video_root = output_root / "videos"
    output_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)
    palette = load_v95_palette()
    implementation_sha256 = _implementation_sha256()
    config_sha256 = sha256_payload(
        {
            "schema_version": "pact_place_v9_5_low_wall_preview_config_v1",
            "implementation_sha256": implementation_sha256,
            "raw_validation_sha256": raw.get("validation_sha256"),
            "seed": args.seed,
        }
    )
    cells = (
        ("left", "wall_left"),
        ("left", "wall_right"),
        ("right", "wall_left"),
        ("right", "wall_right"),
    )
    attempts: list[dict[str, Any]] = []
    accepted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for cell_index, (panel_side, wall_support) in enumerate(cells):
        for candidate in range(args.max_candidates_per_cell):
            row = build_row(
                cell_index=cell_index,
                candidate=candidate,
                panel_side=panel_side,
                wall_support=wall_support,
                palette_document=palette,
                implementation_sha256=implementation_sha256,
                seed=args.seed,
            )
            result = run_row(
                row,
                config_sha256=config_sha256,
                output_root=str(output_root),
                scene_xml=str(SCENE_XML),
            )
            totals = ((result.get("contact_audit") or {}).get("contact_class_totals") or {})
            wall_bow = max(
                (
                    float(record.get("accepted_bow_m", 0.0))
                    for prefix, record in (result.get("bow_diagnostics") or {}).items()
                    if "wall_fixture" in prefix
                ),
                default=0.0,
            )
            item = {
                "cell_index": cell_index,
                "candidate": candidate,
                "panel_side": panel_side,
                "wall_support": wall_support,
                "episode_id": row["episode_id"],
                "status": result.get("status"),
                "clean_success": bool(result.get("clean_success")),
                "wall_fixture_bow_m": wall_bow,
                "clutter_contacts": int(totals.get("clutter", 0)),
                "hazard_bar_contacts": int(totals.get("hazard_bar", 0)),
                "other_environment_contacts": int(totals.get("other_environment", 0)),
                "fixture": row["pact_mounted_wall_fixture"],
                "row": row,
                "result_path": str(_result_path(output_root, row)),
                "trajectory_path": result.get("trajectory_path"),
            }
            attempts.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)
            if item["clean_success"] and wall_bow >= MIN_FIXTURE_BOW_M:
                accepted.append((row, result))
                break
        else:
            break

    # Exact link-clearance and fixture raw-counterfactual admission are written
    # by the dedicated validator. Never render merely from TCP-bow success.
    admission_path = output_root / "fixture_admission.json"
    admission = json.loads(admission_path.read_text()) if admission_path.is_file() else None
    videos: list[dict[str, Any]] = []
    if len(accepted) == 4 and admission and admission.get("passed") is True:
        admitted_ids = set(admission.get("admitted_episode_ids") or [])
        if admitted_ids != {row["episode_id"] for row, _ in accepted}:
            raise RuntimeError("fixture admission episode set does not match accepted rows")
        for index, (row, result) in enumerate(accepted, 1):
            videos.append(
                _render_review_video(
                    {
                        "attempt": index,
                        "row": row,
                        "result_path": str(_result_path(output_root, row)),
                        "trajectory_path": str(result["trajectory_path"]),
                        "video_path": str(
                            video_root
                            / f"success_{index}_panel-{row['intrusion_side']}_{row['mounted_wall_support']}.mp4"
                        ),
                        "frame_stride": args.frame_stride,
                        "clip_label": "V9.5 LOW WALL — RAW + LINK ADMITTED",
                    }
                )
            )

    manifest = {
        "schema_version": "pact_place_v9_5_low_wall_preview_manifest_v1",
        "created_at": utc_now(),
        "role": "human_review_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "raw_prerequisite_path": str(raw_path.relative_to(ROOT)),
        "raw_prerequisite_passed": True,
        "requested_successes": 4,
        "tcp_candidate_successes": len(accepted),
        "fixture_admission_passed": bool(admission and admission.get("passed") is True),
        "videos": videos,
        "attempts": attempts,
    }
    (output_root / "preview_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return 0 if len(videos) == 4 else 2


if __name__ == "__main__":
    raise SystemExit(main())
