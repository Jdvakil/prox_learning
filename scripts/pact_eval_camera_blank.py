"""CLI + eval hooks for blanking policy cameras at inference.

Default off (no camera blanked) so existing evals do not change. A blanked
camera is still rendered by the env (videos, first frames and contact audit are
unchanged); only the policy input is replaced by an all-black RGB frame
(uint8 0 -> 0.0 after /255, before ACT's ImageNet normalisation).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def add_cli_flags(parser) -> None:
    parser.add_argument(
        "--blank_cameras",
        nargs="*",
        default=[],
        help="Policy cameras fed as all-black frames at inference ('all' = every "
        "policy camera). Proximity and qpos unchanged. Protocol field: new "
        "--output_dir.",
    )


def resolve(args, cameras: Sequence[str]) -> list[str]:
    """Validate --blank_cameras against the policy cameras; store the result on args."""
    names = list(getattr(args, "blank_cameras", None) or [])
    if "all" in names:
        names = list(cameras)
    unknown = [n for n in names if n not in cameras]
    if unknown:
        raise SystemExit(
            f"--blank_cameras {unknown} not in policy cameras {list(cameras)}"
        )
    resolved = [c for c in cameras if c in set(names)]
    args.blank_cameras = resolved
    return resolved


def protocol_fields(args) -> dict:
    """Only the non-default value is a key, so existing dirs keep their identity."""
    names = list(getattr(args, "blank_cameras", None) or [])
    return {"blank_cameras": names} if names else {}


def blank_mismatch(old: Mapping[str, Any], new: Mapping[str, Any]) -> str | None:
    """Checked both ways: a missing key (either side) means no camera blanked."""
    old_val = list(old.get("blank_cameras") or [])
    new_val = list(new.get("blank_cameras") or [])
    if old_val != new_val:
        return f"protocol blank_cameras={old_val!r} != {new_val!r}"
    return None


def apply(img: np.ndarray, camera: str, blanked: Sequence[str]) -> np.ndarray:
    """All-black frame of the same shape/dtype when ``camera`` is blanked."""
    if camera in blanked:
        return np.zeros_like(img)
    return img
