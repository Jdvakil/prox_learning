#!/usr/bin/env python3
"""Freeze the blur-media selection and render the calibration figures."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "submodules/act"
if str(ACT) not in sys.path:
    sys.path.insert(0, str(ACT))

import pact_blur  # noqa: E402


CALIBRATION = ROOT / "diagnostics_output/pact_blur_sweep/calibration.json"
SCHEDULE = ROOT / "diagnostics_output/pact_blur_sweep/schedule.json"
RESULT_ROOT = Path("/root/pact_blur_sweep_artifacts/evaluation_v1")
MANIFEST = ROOT / "diagnostics_output/pact_blur_sweep/media_manifest.json"
BUNDLE = Path("/root/pact_slideshow_bundle")
SIGMAS = (0.0, 0.5, 1.0, 2.0)
HASH_KEY = "media_manifest_sha256"
EXPECTED_QUALIFYING_COUNT = 62
EXPECTED_SELECTION = {
    "episode_id": "65f2ab175bedf3c89542d314259ec47f04290b628bd34ef0d2202632f1f67b1b",
    "checkpoint_seed": 3101,
    "schedule_indices": [2, 16, 22, 28],
}
PROTECTED = {
    ROOT / "docs/PACT_BLUR_SWEEP.md": "a215db8269aada479385c72b343ea78c032d35e1ae9162d837812304aeb224bf",
    ROOT / "diagnostics_output/pact_blur_sweep/analysis.json": "12d47f635d82c55ad0a59418a973f4701a72d2cfd760610bf37ad4cdecdedd3e",
    ROOT / "diagnostics_output/pact_blur_sweep/final_decision.json": "9901aa7ba804c2f5207d887f4dee49954c94f89884895a90e05d42a0396949c5",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    expected = canonical_hash(payload)
    if observed != expected:
        raise RuntimeError(f"{label} self-hash mismatch: {observed} != {expected}")
    return str(observed)


def svg_from_png(png: Path, svg: Path, width: int, height: int, title: str) -> None:
    encoded = base64.b64encode(png.read_bytes()).decode("ascii")
    svg.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        f"<title>{title}</title><image width=\"{width}\" height=\"{height}\" "
        f'href="data:image/png;base64,{encoded}"/></svg>\n'
    )


def load_calibrated_frames() -> tuple[dict[str, Any], dict[float, np.ndarray]]:
    calibration = json.loads(CALIBRATION.read_text())
    calibration_hash = validate_self_hash(
        calibration, "calibration_sha256", "blur calibration"
    )
    source = calibration["source"]
    source_path = Path(source["path"])
    if file_hash(source_path) != source["sha256"]:
        raise RuntimeError("calibration source HDF5 changed")
    with h5py.File(source_path, "r") as handle:
        frame = np.asarray(
            handle[source["dataset_key"]][int(source["frame_index"])],
            dtype=np.uint8,
        )
    if frame.shape != (240, 320, 3) or array_hash(frame) != source["frame_sha256"]:
        raise RuntimeError("calibration wrist frame changed")
    tensor = torch.from_numpy(
        np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1))[None, None]
    )
    measurements = {
        float(row["sigma"]): row for row in calibration["measurements"]
    }
    frames: dict[float, np.ndarray] = {}
    for sigma in SIGMAS:
        output = pact_blur.blur_images(tensor, sigma)
        image = np.clip(
            np.rint(output[0, 0].permute(1, 2, 0).cpu().numpy() * 255.0),
            0,
            255,
        ).astype(np.uint8)
        if array_hash(image) != measurements[sigma]["output_rgb_sha256"]:
            raise RuntimeError(f"sigma {sigma} image differs from frozen calibration")
        frames[sigma] = image
    if not torch.equal(pact_blur.blur_images(tensor, 0.0), tensor):
        raise RuntimeError("sigma-zero tensor is not an exact identity")
    if array_hash(frames[0.0]) != "45082f1aaec76023434378b4d6b784d5ab1c253c14f3cb1dc64705d23470f9e3":
        raise RuntimeError("sigma-zero pixel hash is not the pinned source hash")
    calibration["_validated_self_hash"] = calibration_hash
    return calibration, frames


def label(sigma: float, measurement: dict[str, Any]) -> str:
    return (
        f"sigma={sigma:g}  |  kernel {measurement['kernel_size']}x"
        f"{measurement['kernel_size']}  |  retained detail "
        f"{100.0 * float(measurement['retained_fraction_of_sharp']):.1f}%"
    )


def render_figures(
    calibration: dict[str, Any], frames: dict[float, np.ndarray]
) -> dict[str, dict[str, Any]]:
    figure_dir = BUNDLE / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    measurements = {
        float(row["sigma"]): row for row in calibration["measurements"]
    }
    outputs = {
        "sigma_0p5": (
            figure_dir / "fig_blur_sigma_0p5.png",
            figure_dir / "fig_blur_sigma_0p5.svg",
        ),
        "sigma_panel": (
            figure_dir / "fig_blur_sigma_panel.png",
            figure_dir / "fig_blur_sigma_panel.svg",
        ),
    }
    single = cv2.cvtColor(
        cv2.resize(frames[0.5], (1280, 960), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_RGB2BGR,
    )
    shade = single.copy()
    cv2.rectangle(shade, (0, 0), (1280, 112), (0, 0, 0), thickness=-1)
    single = cv2.addWeighted(shade, 0.76, single, 0.24, 0.0)
    cv2.putText(
        single,
        "POLICY WRIST-CAMERA INPUT",
        (30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        single,
        label(0.5, measurements[0.5]),
        (30, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (230, 235, 242),
        2,
        cv2.LINE_AA,
    )
    single_png, single_svg = outputs["sigma_0p5"]
    if not cv2.imwrite(str(single_png), single):
        raise RuntimeError("failed to write sigma-0.5 screenshot")
    svg_from_png(single_png, single_svg, 1280, 960, "Wrist input at sigma 0.5")

    panel_width, image_height, header = 640, 480, 120
    panel = np.full((header + image_height, panel_width * 4, 3), (24, 27, 34), np.uint8)
    for index, sigma in enumerate(SIGMAS):
        x = index * panel_width
        image = cv2.cvtColor(
            cv2.resize(frames[sigma], (panel_width, image_height), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_RGB2BGR,
        )
        panel[header:, x : x + panel_width] = image
        cv2.putText(
            panel,
            f"sigma={sigma:g}  |  kernel {measurements[sigma]['kernel_size']}x{measurements[sigma]['kernel_size']}",
            (x + 18, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"retained sharp detail: {100.0 * float(measurements[sigma]['retained_fraction_of_sharp']):.1f}%",
            (x + 18, 87),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.67,
            (215, 222, 232),
            2,
            cv2.LINE_AA,
        )
        if index:
            cv2.line(panel, (x, 0), (x, header + image_height), (92, 99, 112), 2)
    panel_png, panel_svg = outputs["sigma_panel"]
    if not cv2.imwrite(str(panel_png), panel):
        raise RuntimeError("failed to write blur panel")
    svg_from_png(panel_png, panel_svg, panel_width * 4, header + image_height, "Blur sigma calibration panel")

    return {
        name: {
            extension: {
                "path": str(path),
                "sha256": file_hash(path),
                "size_bytes": path.stat().st_size,
            }
            for extension, path in zip(("png", "svg"), pair)
        }
        for name, pair in outputs.items()
    }


def load_result(row: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    path = RESULT_ROOT / row["output_relpath"] / "result.json"
    result = json.loads(path.read_text())
    expected = {
        "status": "complete",
        "arm": row["arm"],
        "episode_id": row["instance_episode_id"],
        "checkpoint_seed": row["checkpoint_seed"],
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "blur_sigma": row["blur_sigma"],
    }
    if {key: result.get(key) for key in expected} != expected:
        raise RuntimeError(f"row {row['schedule_index']} result identity mismatch")
    return result, path


def freeze_selection(schedule: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[tuple[str, int], dict[float, tuple[dict[str, Any], dict[str, Any], Path]]] = {}
    for row in schedule["rows"]:
        if row["arm"] != "PACT":
            continue
        result, path = load_result(row)
        key = (row["instance_episode_id"], int(row["checkpoint_seed"]))
        grouped.setdefault(key, {})[float(row["blur_sigma"])] = (row, result, path)
    candidates = []
    for (episode_id, seed), cells in grouped.items():
        if set(cells) != set(SIGMAS):
            continue
        sharp_hazard = int(cells[0.0][1]["contact_audit"]["frames_with_contact"]["hazard_bar"])
        sigma2_hazard = int(cells[2.0][1]["contact_audit"]["frames_with_contact"]["hazard_bar"])
        if sharp_hazard == 0 and sigma2_hazard == 0:
            candidates.append((int(cells[0.0][0]["schedule_index"]), episode_id, seed, cells))
    candidates.sort(key=lambda value: (value[0], value[1], value[2]))
    if len(candidates) != EXPECTED_QUALIFYING_COUNT:
        raise RuntimeError(
            f"qualifying count {len(candidates)} != {EXPECTED_QUALIFYING_COUNT}"
        )
    _, episode_id, seed, selected = candidates[0]
    indices = [int(selected[sigma][0]["schedule_index"]) for sigma in SIGMAS]
    if {
        "episode_id": episode_id,
        "checkpoint_seed": seed,
        "schedule_indices": indices,
    } != EXPECTED_SELECTION:
        raise RuntimeError("mechanical blur-media selection changed")
    records = []
    for sigma in SIGMAS:
        row, result, path = selected[sigma]
        audit = result["contact_audit"]
        records.append(
            {
                "sigma": sigma,
                "schedule_index": int(row["schedule_index"]),
                "rollout_id": row["rollout_id"],
                "schedule_row_sha256": row["schedule_row_sha256"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "dataset_stats_path": row["dataset_stats_path"],
                "dataset_stats_sha256": row["dataset_stats_sha256"],
                "surface_encoder_path": row["surface_encoder_path"],
                "surface_encoder_sha256": row["surface_encoder_sha256"],
                "original_result_path": str(path),
                "original_result_sha256": file_hash(path),
                "outcome": {
                    "task_success": bool(result["task_success"]),
                    "manipulation_success": bool(result["task_success"]),
                    "hazard_contact_frames": int(audit["frames_with_contact"]["hazard_bar"]),
                    "first_hazard_bar_contact_step": audit["first_contact_step"]["hazard_bar"],
                    "first_grasp_target_contact_step": audit["first_contact_step"]["grasp_target"],
                    "contact_pair_sample_count": int(audit["sample_count"]),
                    "contact_class_totals": audit["contact_class_totals"],
                },
            }
        )
    return records, len(candidates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace frozen media manifest: {args.output}")
    for path, expected in PROTECTED.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"protected blur artifact changed: {path}")
    calibration, frames = load_calibrated_frames()
    figures = render_figures(calibration, frames)
    schedule = json.loads(SCHEDULE.read_text())
    schedule_hash = validate_self_hash(schedule, "schedule_sha256", "blur schedule")
    rows, qualifying_count = freeze_selection(schedule)
    measurement = {
        float(row["sigma"]): row for row in calibration["measurements"]
    }
    document: dict[str, Any] = {
        "schema_version": "pact_blur_media_manifest_v1",
        "status": "selection_and_gate_frozen_pre_render",
        "created_at_utc": utc_now(),
        "selection_rule": (
            "among episode-seed groups where PACT has zero hazard frames at sigma 0 "
            "and sigma 2, choose the group whose sigma-0 row has the smallest frozen "
            "schedule index; break any remaining tie by episode ID then checkpoint seed"
        ),
        "qualifying_instance_seed_count": qualifying_count,
        "selection": {
            "arm": "PACT",
            "episode_id": EXPECTED_SELECTION["episode_id"],
            "checkpoint_seed": EXPECTED_SELECTION["checkpoint_seed"],
            "intrusion_side": next(
                row["intrusion_side"]
                for row in schedule["rows"]
                if int(row["schedule_index"]) == EXPECTED_SELECTION["schedule_indices"][0]
            ),
            "rows": rows,
        },
        "determinism_gate": {
            "declared_before_render": True,
            "task_success": {"comparison": "exact"},
            "manipulation_success": {
                "comparison": "exact",
                "represented_by": "task_success",
            },
            "first_hazard_bar_contact_step": {"comparison": "exact"},
            "first_grasp_target_contact_step": {
                "comparison": "absolute_step_delta_lte",
                "tolerance_steps": 2,
            },
            "contact_pair_sample_counts": {
                "comparison": "informational_only",
                "record_delta": True,
            },
            "on_breach": "drop_clip_without_retry",
        },
        "visual_contract": {
            "left_pane": "wrist camera after the exact policy-input blur primitive",
            "right_pane": "unblurred render-only third-person camera",
            "playback_speed_factor": 3.0,
            "retained_detail_percent_by_sigma": {
                str(sigma): 100.0 * float(measurement[sigma]["retained_fraction_of_sharp"])
                for sigma in SIGMAS
            },
            "overlay_fields": [
                "arm and checkpoint seed",
                "blur sigma and retained-detail percent",
                "episode ID first 12 characters",
                "task success yes/no",
                "hazard-contact frames running cumulative",
            ],
        },
        "caption": (
            "Re-rendered from the analyzed rollout. Task success, manipulation success, "
            "and first hazard-contact step reproduce exactly; first target contact is "
            "within the declared two-step tolerance; contact-pair counts are informational."
        ),
        "sources": {
            "calibration": {
                "path": str(CALIBRATION),
                "sha256": file_hash(CALIBRATION),
                "calibration_sha256": calibration["_validated_self_hash"],
            },
            "schedule": {
                "path": str(SCHEDULE),
                "sha256": file_hash(SCHEDULE),
                "schedule_sha256": schedule_hash,
            },
            "protected_scientific_artifacts": {
                str(path): expected for path, expected in PROTECTED.items()
            },
        },
        "figures": figures,
        "planned_video_outputs": [
            str(BUNDLE / "videos/blur_sweep" / f"sigma_{str(sigma).replace('.', 'p')}_pact.mp4")
            for sigma in SIGMAS
        ],
        "scientific_record_changed": False,
    }
    document[HASH_KEY] = canonical_hash(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "media_manifest_sha256": document[HASH_KEY],
                "episode_id": EXPECTED_SELECTION["episode_id"],
                "checkpoint_seed": EXPECTED_SELECTION["checkpoint_seed"],
                "qualifying_instance_seed_count": qualifying_count,
                "figures": figures,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
