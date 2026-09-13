"""Slideshow of t=0 RGB from every HF row. One still per episode. H.264.

Does not decode full trajectories. Sidecar mp4 frame 0 only (not converted hdf5).
Do not use scripts/dataset_viz.py for this.

  conda activate mlspaces
  cd /home/jaydv/code/prox_learning
  python scripts/dataset_first_frames.py \\
    --data /mnt/laptop/data/pact_pick_n_place_v2/data/v1011d \\
    --cameras exo_camera_1 --fps 2
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_OUT_BASE = Path("/home/jaydv/code/prox_learning/experiments_output/default/dataset_viz")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.convert_pact_place_to_act import _row_dirs, find_rgb_mp4  # noqa: E402

KNOWN_CAMS = ("exo_camera_1", "wrist_camera", "table_camera")


def out_dir_for(ds_path: Path) -> Path:
    """Same mapping as dataset_viz.out_dir_for (dump outside repo keeps abs path)."""
    p = ds_path.expanduser().resolve()
    if p.is_file():
        p = p.parent / p.stem
    for base in (_ROOT / "data", _ROOT):
        try:
            rel = p.relative_to(base)
        except ValueError:
            continue
        if rel.parts:
            return _OUT_BASE / rel
    return _OUT_BASE / Path(*p.parts[1:])


def parse_cameras(raw: str) -> list[str]:
    cams = [c.strip() for c in raw.split(",") if c.strip()]
    if not cams:
        raise SystemExit("--cameras empty")
    bad = [c for c in cams if c not in KNOWN_CAMS]
    if bad:
        raise SystemExit(f"unknown camera {bad}; want {KNOWN_CAMS}")
    return cams


def first_frame_bgr(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit(f"no frame 0 in {path}")
    return frame


def pad_height(img: np.ndarray, height: int) -> np.ndarray:
    h = img.shape[0]
    if h == height:
        return img
    if h > height:
        raise SystemExit(f"pad_height got {h} > {height}")
    top = (height - h) // 2
    bot = height - h - top
    return cv2.copyMakeBorder(img, top, bot, 0, 0, cv2.BORDER_CONSTANT)


def even_hw(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    pad_b = h % 2
    pad_r = w % 2
    if not pad_b and not pad_r:
        return img
    return cv2.copyMakeBorder(img, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT)


def overlay(bgr: np.ndarray, text: str) -> np.ndarray:
    out = bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, out.shape[0] / 520.0)
    thick = max(1, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 6
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(
        out, text, (pad, pad + th), font, scale, (255, 255, 255), thick, cv2.LINE_AA
    )
    return out


def still_for_row(row: Path, cams: list[str], index: int) -> np.ndarray:
    tiles = []
    missing = []
    for cam in cams:
        path = find_rgb_mp4(row, cam)
        if path is None:
            missing.append(cam)
            continue
        tiles.append(first_frame_bgr(path))
    if missing:
        raise SystemExit(f"{row} missing RGB {missing}")
    h = max(t.shape[0] for t in tiles)
    stacked = np.hstack([pad_height(t, h) for t in tiles])
    return overlay(even_hw(stacked), f"{index:03d}  {row.name}")


def encode_h264(frames: list[np.ndarray], out: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg missing — need libx264 for IDE playback")
    h, w = frames[0].shape[:2]
    for i, fr in enumerate(frames):
        if fr.shape[:2] != (h, w) or fr.dtype != np.uint8 or fr.ndim != 3:
            raise SystemExit(f"frame {i} shape {fr.shape} dtype {fr.dtype} != {(h, w, 3)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-framerate", str(fps),
        "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "20",
        "-movflags", "+faststart",
        str(out),
    ]
    blob = b"".join(np.ascontiguousarray(fr).tobytes() for fr in frames)
    r = subprocess.run(cmd, input=blob, capture_output=True)
    if r.returncode != 0 or not out.is_file():
        err = (r.stderr or b"").decode("utf-8", "ignore")[-400:]
        raise SystemExit(f"ffmpeg h264 failed: {err}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, required=True, help="HF dump with rows/*/trajectory.h5")
    ap.add_argument(
        "--cameras",
        default="exo_camera_1",
        help="comma list: exo_camera_1 (default), wrist_camera, table_camera",
    )
    ap.add_argument("--fps", type=float, default=2.0, help="stills per second (default 2 = 0.5 s hold)")
    args = ap.parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0")

    src = args.data.expanduser().resolve()
    cams = parse_cameras(args.cameras)
    rows = _row_dirs(src)
    frames = [still_for_row(row, cams, i) for i, row in enumerate(rows)]
    out = out_dir_for(src) / "first_frames.mp4"
    encode_h264(frames, out, args.fps)
    print(f"wrote {out}  n={len(frames)}  fps={args.fps}  cams={cams}")


if __name__ == "__main__":
    main()
