#!/usr/bin/env python3
"""Build and optionally run the V9.8 24-row pendant expert preview."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_corridor_contract import (  # noqa: E402
    _protected_artifact_hashes,
    sha256_file,
    sha256_payload,
)
from pact_place_v95_contract import (  # noqa: E402
    build_v95_layout,
    load_v95_palette,
)
from pact_place_v98_pendant_contract import (  # noqa: E402
    ADMISSION_FLOOR,
    CONTRACT_VERSION,
    PHYSICS_CLEAN_FAMILIES,
    SAMPLER_CLASS,
    build_pendant_fixture,
)
from run_pact_place_expert_screen import _result_path, run_row  # noqa: E402
from run_pact_place_v9_panel_smoke import _row as _v93_row  # noqa: E402
from pact_place_v9_contract import load_palette  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v98_pendant_preview"
SCENE_XML = ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"
DEFAULT_SEED = 980024


def build_v98_palette_and_layout(family_id: str, panel_side: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the exact settled V9.5 palette/layout for renderer invariance."""
    palette = load_v95_palette()
    layout = build_v95_layout(
        palette, family_id=family_id, intrusion_side=panel_side
    )
    return palette, layout


def _candidate_fixture(candidate: int) -> dict[str, Any]:
    candidates = (
        (1.15, 0.15),
        (1.14, 0.15),
        (1.16, 0.15),
        (1.18, 0.14),
    )
    try:
        bottom, half_y = candidates[int(candidate) % len(candidates)]
    except (TypeError, ValueError):
        raise ValueError(f"invalid V9.8 candidate: {candidate!r}") from None
    return build_pendant_fixture(bottom_z_m=bottom, half_y_m=half_y)


def build_row(
    *,
    cell_index: int,
    candidate: int,
    panel_side: str,
    palette_document: dict[str, Any] | None = None,
    implementation_sha256: str = "",
    seed: int = DEFAULT_SEED,
    pendant_fixture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one row; the fixture is a function of candidate, never panel side."""
    families = tuple(PHYSICS_CLEAN_FAMILIES)
    family_id = families[int(cell_index) % len(families)]
    candidate_seed = int(seed + (cell_index // 2) * 1009 + int(candidate) * 7919)
    base = _v93_row(
        index=800 + int(cell_index) * 20 + int(candidate),
        family_id=family_id,
        side=panel_side,
        palette_document=load_palette(),
        implementation_sha256=implementation_sha256 or "v98-preview",
        seed=candidate_seed,
    )
    palette, layout = build_v98_palette_and_layout(family_id, panel_side)
    fixture = dict(pendant_fixture or _candidate_fixture(candidate))
    row = {
        **base,
        "sampler_class": SAMPLER_CLASS,
        "pact_clutter_palette": list(palette["palette"]),
        "pact_clutter_layout": layout,
        "layout_id": layout["layout_id"],
        "preview_cell_index": int(cell_index),
        "preview_candidate": int(candidate),
        "pact_mounted_ceiling_fixture": fixture,
        "pact_v98_contract_version": CONTRACT_VERSION,
    }
    row["episode_id"] = hashlib.sha256(
        f"pact-v9.8-pendant:{implementation_sha256}:{cell_index}:{candidate}:{candidate_seed}".encode()
    ).hexdigest()
    row.pop("row_sha256", None)
    row["row_sha256"] = sha256_payload(row)
    return row


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        ROOT / "scripts/run_pact_place_expert_screen.py",
        ROOT / "scripts/pact_place_v98_pendant_contract.py",
        ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
        ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v3.xml",
    ):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_manifest(
    seed: int = DEFAULT_SEED,
    candidate: int = 0,
    pendant_fixture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    implementation_sha256 = _implementation_sha256()
    rows = []
    for cell_index, family_id in enumerate(PHYSICS_CLEAN_FAMILIES):
        for panel_side in ("left", "right"):
            row = build_row(
                cell_index=cell_index * 2 + (0 if panel_side == "left" else 1),
                candidate=candidate,
                panel_side=panel_side,
                implementation_sha256=implementation_sha256,
                seed=seed,
                pendant_fixture=pendant_fixture,
            )
            rows.append(row)
            # Four rows per family/side variant keeps the 24-row screen balanced
            # while holding the pendant geometry independent of panel side.
            for repeat in range(1, 4):
                rows.append(
                    build_row(
                        cell_index=cell_index * 2 + (0 if panel_side == "left" else 1),
                        candidate=(candidate + repeat) % 4,
                        panel_side=panel_side,
                        implementation_sha256=implementation_sha256,
                        seed=seed,
                        pendant_fixture=pendant_fixture,
                    )
                )
    for index, row in enumerate(rows):
        row["role_index"] = index
        row.pop("row_sha256", None)
        row["row_sha256"] = sha256_payload(row)
    return {
        "schema_version": "pact_place_v9_8_pendant_preview_v1",
        "role": "preregistered_expert_screen_manifest",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "contract_version": CONTRACT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "implementation_sha256": implementation_sha256,
        "scene": {"xml": str(SCENE_XML.relative_to(ROOT))},
        "expert_screen_rows": rows,
    }


def build_config(manifest: dict[str, Any], manifest_sha256: str) -> dict[str, Any]:
    config = {
        "schema_version": "pact_place_corridor_v9_8",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "role": "preregistered_v98_environment_gate",
        "min_clean_successes": 20,
        "admission_floor": ADMISSION_FLOOR,
        "scene": {
            "xml": str(SCENE_XML.relative_to(ROOT)),
            "aperture_plane_x_m": 0.58,
            "place_tray_x_bounds_m": [0.25, 0.45],
            "clutter_body_prefix": "pact_clutter_",
            "environment_version": "pact_place_corridor_v9_8_pendant",
            "sampler_class": SAMPLER_CLASS,
            "ceiling_fixture_body": "pact_clutter_mount_ceiling",
            "ceiling_fixture_geom": "pact_clutter_mount_ceiling_g",
            "lateral_lane_cost_m": 0.0,
        },
        "frozen_artifacts": {
            "encoder_sha256": "6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206",
            "production_tensor": [40, 4, 8, 8],
            "sensor_count": 40,
        },
        "manifest_sha256": manifest_sha256,
        "implementation_sha256": manifest["implementation_sha256"],
        "protected_artifact_sha256_before": _protected_artifact_hashes(),
        "expert_screen_rows": manifest["expert_screen_rows"],
    }
    config["config_sha256"] = sha256_payload(config)
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--siting", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    pendant_fixture = None
    if args.siting:
        siting = json.loads(args.siting.resolve().read_text())
        selected = siting.get("selected") or {}
        pendant_fixture = dict(selected.get("fixture") or {})
        if not pendant_fixture:
            raise SystemExit("--siting does not contain a selected pendant fixture")
    manifest = build_manifest(args.seed, args.candidate, pendant_fixture)
    if len(manifest["expert_screen_rows"]) != 24:
        raise RuntimeError("V9.8 manifest must contain exactly 24 rows")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    if args.config_output:
        config = build_config(manifest, sha256_file(manifest_path))
        config_path = args.config_output.resolve()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        print(config_path)
    if not args.run:
        return 0
    config = build_config(manifest, sha256_file(manifest_path))
    config_path = output_root / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    for row in manifest["expert_screen_rows"]:
        result = run_row(
            row,
            config_sha256=config["config_sha256"],
            output_root=str(output_root),
            scene_xml=str(SCENE_XML),
        )
        print(json.dumps({"role_index": row["role_index"], "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
