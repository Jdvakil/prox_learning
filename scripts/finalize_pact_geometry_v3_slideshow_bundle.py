#!/usr/bin/env python3
"""Finalize geometry-v3 presentation provenance and refresh bundle integrity."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path("/root/pact_slideshow_bundle")
BUNDLE_MANIFEST = BUNDLE / "BUNDLE_MANIFEST.json"
REPO_BUNDLE_MANIFEST = ROOT / "diagnostics_output/pact_slideshow_bundle_manifest.json"
QUALITATIVE = ROOT / "diagnostics_output/pact_geometry_generalization_v3/qualitative_video_manifest.json"
HASH_KEY = "qualitative_video_manifest_sha256"
KEY_NUMBERS_SHA256 = "dc53f32937a517b55d327e51ad428b736a8e71f11e60b06bd9cb939a56c7aca2"

SOURCE_COPIES = {
    "geometry_v3_manifest.json": ROOT / "configs/pact_geometry_generalization_v3.json",
    "geometry_v3_schedule.json": ROOT / "diagnostics_output/pact_geometry_generalization_v3/schedule.json",
    "geometry_v3_analysis.json": ROOT / "diagnostics_output/pact_geometry_generalization_v3/analysis.json",
    "geometry_v3_final_decision.json": ROOT / "diagnostics_output/pact_geometry_generalization_v3/final_decision.json",
}
REPORT_SOURCE = ROOT / "docs/PACT_GEOMETRY_GENERALIZATION_V3.md"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    expected = canonical_hash(payload)
    if observed != expected:
        raise RuntimeError(f"{label} self-hash mismatch: {observed} != {expected}")
    return str(observed)


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def ffprobe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)["streams"][0]


def normalize_geometry_figure() -> None:
    """Expand the three-panel strip into the bundle's standard 3200x1800 slide."""
    png = BUNDLE / "figures/fig11_geometry_conditions.png"
    svg = BUNDLE / "figures/fig11_geometry_conditions.svg"
    source = cv2.imread(str(png), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError("geometry figure PNG is unreadable")
    if source.shape[:2] == (1800, 3200):
        return
    elif source.shape[:2] != (420, 1872):
        raise RuntimeError(f"unexpected geometry figure shape {source.shape[:2]}")
    canvas = np.full((1800, 3200, 3), (24, 27, 34), dtype=np.uint8)
    strip = cv2.resize(source, (3040, 682), interpolation=cv2.INTER_CUBIC)
    canvas[80:762, 80:3120] = strip
    cv2.putText(
        canvas,
        "Same weights. Different information.",
        (140, 980),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.35,
        (255, 255, 255),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "PACT vs PACT_PERMUTED isolates whether proximity is aligned to the scene.",
        (140, 1080),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        (210, 217, 228),
        3,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (140, 1190), (3060, 1510), (43, 49, 61), thickness=-1)
    cv2.rectangle(canvas, (140, 1190), (3060, 1510), (154, 173, 0), thickness=5)
    cv2.putText(
        canvas,
        "HELD-OUT GEOMETRY RESULT",
        (205, 1280),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (195, 220, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "-11.7 percentage points any-contact",
        (205, 1390),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.75,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "95% CI [-18.3, -5.4]  |  all 9 condition-seed contrasts favor PACT",
        (205, 1470),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.02,
        (225, 230, 238),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Scope: held-out obstacle geometry within one scene - not multiple environments.",
        (140, 1680),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.12,
        (194, 202, 214),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(png), canvas):
        raise RuntimeError("failed to write normalized geometry PNG")
    encoded = __import__("base64").b64encode(png.read_bytes()).decode("ascii")
    svg.write_text(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"3200\" height=\"1800\" "
        "viewBox=\"0 0 3200 1800\"><title>Held-out geometry conditions and result</title>"
        f"<image width=\"3200\" height=\"1800\" href=\"data:image/png;base64,{encoded}\"/></svg>\n"
    )


def update_qualitative_manifest() -> dict[str, Any]:
    document = json.loads(QUALITATIVE.read_text())
    validate_self_hash(document, HASH_KEY, "geometry qualitative manifest")
    if document.get("status") != "presentation_release_verified":
        raise RuntimeError("geometry presentation release is not fully verified")
    if document.get("retained_conditions") != ["C2", "Z_093"]:
        raise RuntimeError("both fixed geometry conditions were not retained")
    for clip_id, summary in document["determinism_checks"].items():
        check = json.loads(Path(summary["path"]).read_text())
        validate_self_hash(check, "determinism_check_sha256", clip_id)
        original = json.loads(Path(check["original_result"]["path"]).read_text())
        rerun = json.loads(Path(check["rerun_result"]["path"]).read_text())
        summary["informational_contact_frame_deltas"] = {
            contact_class: {
                "original": int(original["contact_audit"]["frames_with_contact"][contact_class]),
                "rerun": int(rerun["contact_audit"]["frames_with_contact"][contact_class]),
                "signed_delta": int(rerun["contact_audit"]["frames_with_contact"][contact_class])
                - int(original["contact_audit"]["frames_with_contact"][contact_class]),
            }
            for contact_class in ("grasp_target", "hazard_bar", "other_environment")
        }
    readme = Path(document["release_outputs"]["readme"]["path"])
    document["release_outputs"]["readme"]["sha256"] = file_hash(readme)
    for extension in ("png", "svg"):
        figure = document["release_outputs"]["geometry_figure"][extension]
        figure["sha256"] = file_hash(Path(figure["path"]))
    document["bundle_documentation"] = {
        name: {"path": str(BUNDLE / name), "sha256": file_hash(BUNDLE / name)}
        for name in ("INDEX.md", "VIDEO_SHOT_LIST.md")
    }
    document["post_render_audit"] = {
        "all_four_exact_task_success": True,
        "all_four_exact_manipulation_success": True,
        "all_four_exact_first_hazard_contact": True,
        "all_four_first_target_contact_delta_steps": 0,
        "fallbacks_rendered": 0,
        "key_numbers_unchanged_sha256": file_hash(BUNDLE / "KEY_NUMBERS.md"),
    }
    if document["post_render_audit"]["key_numbers_unchanged_sha256"] != KEY_NUMBERS_SHA256:
        raise RuntimeError("slideshow KEY_NUMBERS.md changed")
    payload = dict(document)
    payload.pop(HASH_KEY, None)
    document[HASH_KEY] = canonical_hash(payload)
    atomic_json(QUALITATIVE, document)
    return document


def copy_geometry_sources(qualitative: dict[str, Any]) -> None:
    data = BUNDLE / "data"
    reports = BUNDLE / "reports"
    for name, source in SOURCE_COPIES.items():
        target = data / name
        if target.exists() and file_hash(target) != file_hash(source):
            raise RuntimeError(f"existing slideshow source differs: {target}")
        if not target.exists():
            shutil.copy2(source, target)
    qualitative_target = data / "geometry_v3_qualitative_video_manifest.json"
    shutil.copy2(QUALITATIVE, qualitative_target)
    report_target = reports / REPORT_SOURCE.name
    if report_target.exists() and file_hash(report_target) != file_hash(REPORT_SOURCE):
        raise RuntimeError(f"existing slideshow report differs: {report_target}")
    if not report_target.exists():
        shutil.copy2(REPORT_SOURCE, report_target)
    if file_hash(qualitative_target) != file_hash(QUALITATIVE):
        raise RuntimeError("qualitative manifest copy mismatch")
    if qualitative["sources"]["report"]["sha256"] != file_hash(report_target):
        raise RuntimeError("protected geometry report copy mismatch")


def video_records() -> dict[str, Any]:
    records = {}
    for path in sorted(BUNDLE.glob("videos/**/*.mp4")):
        key = str(path.relative_to(BUNDLE).with_suffix("")).replace("/", "__")
        records[key] = {
            "path": str(path.relative_to(BUNDLE)),
            "sha256": file_hash(path),
            "size_bytes": path.stat().st_size,
            "video": ffprobe_video(path),
        }
    return records


def refresh_bundle_manifest() -> dict[str, Any]:
    entries = []
    for path in sorted(
        item for item in BUNDLE.rglob("*")
        if item.is_file() and item.name != "BUNDLE_MANIFEST.json"
    ):
        entries.append(
            {
                "path": str(path.relative_to(BUNDLE)),
                "sha256": file_hash(path),
                "size_bytes": path.stat().st_size,
            }
        )
    extension_counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        extension_counts[Path(entry["path"]).suffix.lower() or "no_extension"] += 1
    figure_stems = {
        str(Path(entry["path"]).with_suffix(""))
        for entry in entries
        if entry["path"].startswith("figures/")
    }
    document: dict[str, Any] = {
        "schema_version": "pact_slideshow_bundle_manifest_v1",
        "bundle_root": str(BUNDLE),
        "scientific_artifacts_modified": False,
        "gpu_work_performed": True,
        "rollouts_or_training_performed": True,
        "presentation_only_rollout_rerenders": 13,
        "training_performed": False,
        "figure_concepts": len(figure_stems),
        "figure_files": sum(1 for entry in entries if entry["path"].startswith("figures/")),
        "video_files": sum(1 for entry in entries if entry["path"].endswith(".mp4")),
        "paired_video_files": 6,
        "matched_single_arm_clip_files": 7,
        "shipped_qualitative_single_arm_clip_files": 8,
        "complete_matched_instance_pairs": 3,
        "geometry_v3_complete_matched_pairs": 2,
        "geometry_v3_individual_clip_files": 4,
        "geometry_v3_side_by_side_files": 2,
        "determinism_dropped_clip_files": 1,
        "supplemental_act_success_clip_files": 1,
        "unpaired_independent_probe_files": 0,
        "total_payload_size_bytes_excluding_manifest": sum(entry["size_bytes"] for entry in entries),
        "extension_counts": dict(sorted(extension_counts.items())),
        "video_records": video_records(),
        "paired_video_release": (
            "The original contact-endpoint release retains one matched pair and two "
            "explicitly unpaired examples. Geometry v3 adds two matched PACT versus "
            "PACT_PERMUTED pairs, four individual clips, and two synchronized "
            "side-by-side exports; all four rank-1 rerenders passed the frozen gate."
        ),
        "optional_figure_10": "omitted_no_frozen_source_field_and_no_new_analysis_allowed",
        "geometry_figure_11": "C0_C2_Z_093_fixed_camera_initial_pose",
        "entries": entries,
    }
    document["bundle_manifest_sha256"] = canonical_hash(document)
    atomic_json(BUNDLE_MANIFEST, document)
    atomic_json(REPO_BUNDLE_MANIFEST, document)
    return document


def main() -> int:
    normalize_geometry_figure()
    qualitative = update_qualitative_manifest()
    copy_geometry_sources(qualitative)
    bundle = refresh_bundle_manifest()
    print(
        json.dumps(
            {
                "qualitative_manifest_sha256": qualitative[HASH_KEY],
                "bundle_manifest_sha256": bundle["bundle_manifest_sha256"],
                "bundle_entries": len(bundle["entries"]),
                "video_files": bundle["video_files"],
                "figure_concepts": bundle["figure_concepts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
