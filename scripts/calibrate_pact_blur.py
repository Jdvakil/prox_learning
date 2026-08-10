#!/usr/bin/env python3
"""Calibrate the provisional blur grid on one preserved real wrist frame."""

from __future__ import annotations

import argparse
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
ACT_DIR = ROOT / "submodules/act"
if str(ACT_DIR) not in sys.path:
    sys.path.insert(0, str(ACT_DIR))

import pact_blur  # noqa: E402


DEFAULT_SOURCE = (
    ROOT
    / "assets/act_style_data/pact_collision_corridor_v2_full_cba7ff88/episode_0.hdf5"
)
DEFAULT_OUTPUT = ROOT / "diagnostics_output/pact_blur_sweep/calibration.json"
CANDIDATE_SIGMAS = (0.0, 0.25, 0.5, 1.0, 2.0)


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


def laplacian_variance(image_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calibrate(source: Path) -> dict[str, Any]:
    with h5py.File(source, "r") as handle:
        images = handle["observations/images/wrist_camera"]
        frame_index = int(len(images) // 2)
        frame = np.asarray(images[frame_index], dtype=np.uint8)
        proximity = np.asarray(
            handle["observations/proximity"][frame_index], dtype=np.float32
        )
    if frame.shape != (240, 320, 3):
        raise RuntimeError(f"calibration frame shape changed: {frame.shape}")
    tensor = torch.from_numpy(
        np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1))[None, None]
    )
    sharp_variance = laplacian_variance(frame)
    measurements = []
    zero_identity = False
    for sigma in CANDIDATE_SIGMAS:
        output = pact_blur.blur_images(tensor, sigma)
        if sigma == 0.0:
            zero_identity = output is tensor and torch.equal(output, tensor)
        output_rgb = np.clip(
            np.rint(
                output[0, 0].permute(1, 2, 0).detach().cpu().numpy()
                * 255.0
            ),
            0,
            255,
        ).astype(np.uint8)
        variance = laplacian_variance(output_rgb)
        measurements.append(
            {
                "sigma": sigma,
                "kernel_size": (
                    1
                    if sigma < 0.1
                    else 2 * int(np.ceil(3.0 * sigma)) + 1
                ),
                "variance_of_laplacian": variance,
                "retained_fraction_of_sharp": variance / sharp_variance,
                "output_rgb_sha256": array_hash(output_rgb),
                "tensor_changed_from_sharp": not torch.equal(output, tensor),
            }
        )
    variances = [row["variance_of_laplacian"] for row in measurements]
    if not all(left >= right for left, right in zip(variances, variances[1:])):
        raise RuntimeError("variance of Laplacian is not monotone nonincreasing")
    if not zero_identity:
        raise RuntimeError("sigma=0 is not an exact tensor identity")
    proximity_after = np.array(proximity, copy=True)
    if not np.array_equal(proximity, proximity_after):
        raise RuntimeError("calibration unexpectedly changed proximity")
    document: dict[str, Any] = {
        "schema_version": "pact_blur_calibration_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "status": "provisional_grid_measured_not_frozen",
        "source": {
            "path": str(source.resolve()),
            "sha256": file_hash(source),
            "dataset_key": "observations/images/wrist_camera",
            "selection_rule": "floor(number_of_frames / 2)",
            "frame_index": frame_index,
            "frame_shape": list(frame.shape),
            "frame_dtype": str(frame.dtype),
            "frame_sha256": array_hash(frame),
            "proximity_key": "observations/proximity",
            "proximity_frame_sha256": array_hash(proximity),
        },
        "primitive": {
            "path": str((ACT_DIR / "pact_blur.py").resolve()),
            "sha256": file_hash(ACT_DIR / "pact_blur.py"),
            "upstream_commit": "ec447930e1d025fed549ef2f58354aa87001c28c",
            "sigma_zero_same_object_and_bit_identical": zero_identity,
            "proximity_unchanged": True,
        },
        "candidate_sigmas": list(CANDIDATE_SIGMAS),
        "metric": "variance of cv2 Laplacian on rounded uint8 RGB converted to grayscale",
        "measurements": measurements,
        "monotone_nonincreasing": True,
        "freeze_rule": {
            "control_sigma": 0.0,
            "positive_points": 3,
            "transition_retained_fraction_open_interval": [0.05, 0.8],
            "minimum_transition_points": 2,
            "note": (
                "choose the final grid only from image calibration, before any "
                "blurred policy outcome exists"
            ),
        },
    }
    document["calibration_sha256"] = canonical_hash(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace calibration: {args.output}")
    document = calibrate(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
