"""Concatenate a folder of trajectory HDF5s into one Foxglove MCAP + one tiled video.

Auto-detects three layouts this repo actually uses:

  ACT          episode_*.hdf5  — RGB inside the file, optional (T, 40, 8, 8) proximity
  HF rows      rows/*/trajectory.h5 + episode_00000000_*.mp4  (v5 wrist-only clones)
               or accepted/<sha>/trajectory.h5 + episode_<sha>_{wrist,table}_camera.mp4
  datagen      house_*/trajectories*.h5 + episode_*_*_batch_*.mp4

Every episode is laid end-to-end on one timeline (0.5 s gap). The MCAP is the
interactive visualizer (Foxglove). The MP4s + index.html are the browsing path:
wrist / table RGB, 8x8 skin mosaic, live prox-3D (FK + back-projected returns),
and joint position / velocity plots.

Video is written as one short clip per episode, filed by trajectory type:

  episodes/<behavior_class>/<idx>_<label>.mp4

index.html lists the clips by type and plays one at a time. --group-by picks a
different attribute for the folder name. --one-video restores the old single
concatenated dataset.mp4.

Output is not selectable. Everything lands under the fixed root
/home/jaydv/code/prox_learning/experiments_output/default/dataset_viz, in a folder
that mirrors the dataset path with the <repo>/data prefix removed:

  data/molmo-pi0-eval-videos/data/fumehood/pick
      -> <root>/molmo-pi0-eval-videos/data/fumehood/pick/

  conda activate mlspaces
  cd /home/jaydv/code/prox_learning
  python scripts/dataset_viz.py --reencode experiments_output/default/dataset_viz
  python scripts/dataset_viz.py --data /home/jaydv/code/prox_learning/data --list
  python scripts/dataset_viz.py --data /home/jaydv/code/prox_learning/data \
      --each --cam3d --no-mcap --stride 2
  python scripts/dataset_viz.py --dashboard
  python scripts/dataset_viz.py --data data/pact_place_corridor_v5 --max-episodes 2
  python scripts/dataset_viz.py --data data/molmo-pi0-eval-videos/data/fumehood/pick --list
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np
from matplotlib import colormaps

import foxglove
from foxglove import channels as C
from foxglove.schemas import (
    CompressedImage,
    FrameTransform,
    FrameTransforms,
    Log,
    LogLevel,
    Pose,
    PoseInFrame,
    Quaternion,
    Timestamp,
    Vector3,
)

_ROOT = Path(__file__).resolve().parents[1]
_OUT_BASE = Path("/home/jaydv/code/prox_learning/experiments_output/default/dataset_viz")
_SCRIPTS = Path(__file__).resolve().parent
_ACT = _ROOT / "submodules" / "act"
for _p in (_SCRIPTS, _ACT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hybrid_skin_sensors import HYBRID_SKIN_SENSOR_ORDER  # noqa: E402

import foxglove_viz as fv  # noqa: E402

_TURBO = colormaps["turbo"]
EP_GAP_S = 0.5
DEFAULT_DT = 0.066  # datagen policy_dt; ACT hdf5 does not store dt
DEAD_PIXEL_M = 0.005
JPEG_QUALITY = 80
TILE_W, TILE_H = 320, 160  # RGB strip; leftover left column is the heatmap
HEAT_H = 320
VIEW3D_W, VIEW3D_H = 400, TILE_H + HEAT_H  # 3D tab sits beside wrist+heatmap
PLOT_H = 110
HUD_H = 40
RGB_W = TILE_W * 2
CANVAS_W = RGB_W + VIEW3D_W
CANVAS_H = TILE_H + HEAT_H + PLOT_H * 2 + HUD_H
# H.264 yuv420p needs even W/H (VS Code / Cursor media preview).
CANVAS_W += CANVAS_W % 2
CANVAS_H += CANVAS_H % 2

CAM_ALIAS = {
    "wrist_camera": "wrist",
    "wrist": "wrist",
    "wrist_rgb": "wrist",
    "exo_camera_1": "table",
    "exo_camera": "table",
    "exo": "table",
    "table_camera": "table",
    "table": "table",
    "top": "table",
}

RGB_STEMS = ("wrist_camera", "exo_camera_1", "table_camera")
OPTIONAL_STEMS = ("sensors_rgb256", "wrist_camera_depth", "exo_camera_1_depth")

_PLOT_BGR = [
    (255, 180, 80),
    (160, 255, 80),
    (80, 200, 255),
    (120, 80, 255),
    (255, 80, 160),
    (80, 255, 220),
    (80, 140, 255),
    (200, 200, 200),
]

_EE_BODIES = (
    "robot_0/fr3_link7",
    "robot_0/panda_link7",
    "robot_0/hand",
    "robot_0/fr3_hand",
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _mp4_codec(path: Path) -> str:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return probe.stdout.strip().lower() if probe.returncode == 0 else ""


def encode_h264_ide(mp4_path: Path) -> None:
    """MPEG-4 Part 2 (OpenCV mp4v) does not play in VS Code / Cursor. H.264 does."""
    if not mp4_path.is_file():
        return
    if _mp4_codec(mp4_path) == "h264":
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"ffmpeg missing — {mp4_path} stays MPEG-4 (IDE will not play it)")
        return
    tmp = mp4_path.with_name(mp4_path.stem + ".h264tmp.mp4")
    cmd = [
        ffmpeg, "-y", "-i", str(mp4_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "20",
        "-movflags", "+faststart",
        str(tmp),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.is_file():
        print(f"ffmpeg h264 failed on {mp4_path}: {(r.stderr or '')[-400:]}")
        if tmp.exists():
            tmp.unlink()
        return
    tmp.replace(mp4_path)
    print(f"  h264    {mp4_path}")


def _json_row(blob) -> dict:
    raw = bytes(np.asarray(blob).tobytes()).split(b"\x00", 1)[0]
    if not raw:
        return {}
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _jsonable(v):
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            return str(v)
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore")
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _attrs(obj) -> dict:
    try:
        return {str(k): _jsonable(v) for k, v in dict(obj.attrs).items()}
    except Exception:
        return {}


def _decode_obs_scene(grp) -> dict:
    if "obs_scene" not in grp:
        return {}
    raw = np.asarray(grp["obs_scene"])
    raw = raw[()] if raw.shape == () else raw[0]
    if isinstance(raw, dict):
        return raw
    s = (raw.tobytes() if isinstance(raw, np.ndarray) else raw)
    if isinstance(s, bytes):
        s = s.decode("utf-8", "ignore").rstrip("\x00")
    try:
        return json.loads(s)
    except Exception:
        return {}


def _as_rgb(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def _letterbox(rgb: np.ndarray, w: int, h: int, fill: int = 16) -> np.ndarray:
    canvas = np.full((h, w, 3), fill, np.uint8)
    if rgb is None or rgb.size == 0:
        return canvas
    ih, iw = rgb.shape[:2]
    scale = min(w / max(iw, 1), h / max(ih, 1))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _slate(w: int, h: int, text: str) -> np.ndarray:
    img = np.full((h, w, 3), 24, np.uint8)
    cv2.putText(img, text, (16, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (160, 160, 160), 1, cv2.LINE_AA)
    return img


def _cam_topic(name: str) -> str:
    return "/camera/" + CAM_ALIAS.get(name, re.sub(r"[^a-zA-Z0-9_]+", "_", name))


def _link_prefix(name: str) -> str:
    if "_sensor_" in name:
        return name.rsplit("_sensor_", 1)[0]
    return name


def _ep_num(name: str) -> int:
    m = re.search(r"(\d+)", Path(name).stem)
    return int(m.group(1)) if m else 0


def _pool_prox(arr: np.ndarray, pool: str) -> np.ndarray:
    a = np.asarray(arr, np.float32)
    if a.ndim == 4:  # (T, sub, 8, 8)
        return a.min(axis=1) if pool == "min" else a.mean(axis=1)
    if a.ndim == 3:
        return a
    raise ValueError(f"proximity shape {a.shape}")


# --------------------------------------------------------------------------- #
# Heatmap mosaic (numpy, no matplotlib per frame)
# --------------------------------------------------------------------------- #
def render_heatmap(d8s: np.ndarray, names: list[str], near: float, far: float,
                   out_w: int, out_h: int) -> np.ndarray:
    """d8s: (N, 8, 8) metres -> BGR uint8 mosaic, one row per link.

    Tiles stretch to fill ``out_w`` x ``out_h`` (no leftover blank band).
    """
    canvas = np.full((out_h, out_w, 3), 18, np.uint8)
    if d8s is None or len(d8s) == 0:
        cv2.putText(canvas, "no proximity in this file", (16, out_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1, cv2.LINE_AA)
        return canvas
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for i, n in enumerate(names):
        groups.setdefault(_link_prefix(n), []).append(i)
    n_rows = max(len(groups), 1)
    pad = 3
    caption_h = 12
    link_w = 70
    grid_w = max(out_w - link_w, 32)
    for r, (link, idxs) in enumerate(groups.items()):
        y0 = r * out_h // n_rows
        y1 = (r + 1) * out_h // n_rows
        row_h = y1 - y0
        n_c = max(len(idxs), 1)
        cell_w = max(8, (grid_w - pad * (n_c + 1)) // n_c)
        cell_h = max(8, row_h - caption_h - pad)
        cv2.putText(canvas, link, (out_w - link_w + 2, y0 + min(14, row_h - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 200, 255), 1, cv2.LINE_AA)
        for c, i in enumerate(idxs):
            patch = np.asarray(d8s[i], np.float32)
            valid = np.isfinite(patch) & (patch >= DEAD_PIXEL_M)
            norm = np.clip((patch - near) / max(far - near, 1e-6), 0.0, 1.0)
            col = (_TURBO(1.0 - norm)[..., :3] * 255).astype(np.uint8)
            col[~valid] = 40
            tile = cv2.resize(col, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
            tile = tile[:, :, ::-1]  # RGB -> BGR
            x = pad + c * (cell_w + pad)
            y = y0 + caption_h
            y2, x2 = min(y + cell_h, out_h), min(x + cell_w, grid_w)
            canvas[y:y2, x:x2] = tile[: y2 - y, : x2 - x]
            cv2.putText(canvas, names[i].replace("sensor_", "s"),
                        (x, y0 + caption_h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (180, 180, 180), 1, cv2.LINE_AA)
    return canvas


def _look_at(eye: np.ndarray, target: np.ndarray):
    f = target - eye
    n = np.linalg.norm(f) + 1e-9
    f = f / n
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, np.array([1.0, 0.0, 0.0]))
    r = r / (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f)
    R = np.stack([r, -u, f], axis=0)
    tvec = -R @ eye
    return R, tvec


def _project(pts: np.ndarray, R: np.ndarray, tvec: np.ndarray, w: int, h: int, fov=42.0):
    if pts.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, bool)
    f = (0.5 * h) / np.tan(np.deg2rad(fov / 2))
    pc = (R @ pts.T).T + tvec
    z = pc[:, 2]
    vis = z > 0.08
    zc = np.clip(z, 0.08, None)
    u = f * pc[:, 0] / zc + (w - 1) / 2.0
    v = f * pc[:, 1] / zc + (h - 1) / 2.0
    return u, v, z, vis


def proximity_world_points(ep: Episode, t: int, model, data, cam_id: dict,
                           near: float, d_max: float) -> tuple[np.ndarray, np.ndarray]:
    if ep.proximity is None or not ep.sensor_names:
        return np.zeros((0, 3)), np.zeros((0,))
    all_pts, all_d = [], []
    for i, name in enumerate(ep.sensor_names):
        d8 = ep.proximity[t, i]
        if name in ep.cam2w:
            c2w = ep.cam2w[name][min(t, len(ep.cam2w[name]) - 1)]
        elif model is not None and name in cam_id:
            c2w = np.eye(4)
            cid = cam_id[name]
            c2w[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
            c2w[:3, 3] = data.cam_xpos[cid]
        else:
            continue
        pts, d = fv.backproject(d8, c2w, near, d_max)
        if len(pts):
            all_pts.append(pts)
            all_d.append(d)
    if not all_pts:
        return np.zeros((0, 3)), np.zeros((0,))
    return np.concatenate(all_pts), np.concatenate(all_d)


def _scene_cam(ep: Episode) -> str | None:
    """RGB camera to draw the returns on: the table / exo view, else the wrist."""
    for want in ("table", "wrist"):
        for name in ep.images:
            if CAM_ALIAS.get(name) == want and name in ep.cam_params:
                return name
    return None


def camera_matrices(ep: Episode, cam: str, t: int, w: int, h: int):
    """(R, tvec, K) that carry a world point to a pixel of this camera's frame.

    sensor_param stores the calibration of a square render (cx = cy = 240 for a
    480x480 sensor), while the sidecar mp4 is a wider frame (624x352) with the
    same vertical field of view. Scale the focal length by the height ratio and
    move the principal point to the middle of the decoded frame. The correction
    is a no-op when the frame already matches the stored sensor size.
    """
    prm = ep.cam_params.get(cam)
    if not prm:
        return None
    ext = np.asarray(prm["extrinsic_cv"])
    kin = np.asarray(prm["intrinsic_cv"])
    e = ext[min(t, len(ext) - 1)]
    k = np.array(kin[min(t, len(kin) - 1)], np.float64)
    cy = float(k[1, 2])
    if cy <= 1e-6:
        return None
    scale = h / (2.0 * cy)
    k[0, 0] *= scale
    k[1, 1] *= scale
    k[0, 2] = (w - 1) / 2.0
    k[1, 2] = (h - 1) / 2.0
    return np.asarray(e[:, :3], np.float64), np.asarray(e[:, 3], np.float64), k


def render_cam3d(ep: Episode, t: int, pts: np.ndarray, depths: np.ndarray,
                 near: float, far: float, w: int, h: int,
                 cam: str | None = None) -> np.ndarray | None:
    """BGR panel: the skin returns drawn on the scene camera's own frame.

    The synthetic panel of render_view3d shows the returns in empty space. This
    one puts them on the pixels of whatever they bounced off, so a point can be
    read against the object it belongs to. Returns None when the episode has no
    usable camera calibration, so the caller keeps the old panel.
    """
    name = cam or _scene_cam(ep)
    if name is None or name not in ep.images:
        return None
    frame = _index(ep.images[name], t, ep.T)
    if frame is None:
        return None
    img = np.ascontiguousarray(_as_rgb(frame)[:, :, ::-1])   # RGB -> BGR
    fh, fw = img.shape[:2]
    mats = camera_matrices(ep, name, t, fw, fh)
    if mats is None:
        return None
    R, tvec, K = mats

    n = 0
    if len(pts):
        p = np.asarray(pts, np.float64)
        d = np.asarray(depths, np.float64)
        if len(p) > 2400:
            step = int(np.ceil(len(p) / 2400))
            p, d = p[::step], d[::step]
        pc = (R @ p.T).T + tvec
        z = pc[:, 2]
        front = z > 0.02
        uvw = (K @ pc.T).T
        zc = np.where(np.abs(uvw[:, 2]) < 1e-9, 1e-9, uvw[:, 2])
        u, v = uvw[:, 0] / zc, uvw[:, 1] / zc
        norm = np.clip((d - near) / max(far - near, 1e-6), 0, 1)
        rgb = (_TURBO(1.0 - norm)[:, :3] * 255).astype(np.uint8)
        for i in np.argsort(-z):          # far first, so near points land on top
            if not front[i]:
                continue
            x, y = int(round(u[i])), int(round(v[i]))
            if 0 <= x < fw and 0 <= y < fh:
                col = (int(rgb[i, 2]), int(rgb[i, 1]), int(rgb[i, 0]))
                cv2.circle(img, (x, y), 2, col, -1, cv2.LINE_AA)
                n += 1

    panel = _letterbox(img, w, h)
    # the camera frame is mostly bright, so outline the captions
    for text, org, size in ((f"prox on {CAM_ALIAS.get(name, name)}", (8, 18), 0.5),
                            (f"{n} pts  red=near", (8, h - 10), 0.4)):
        cv2.putText(panel, text, org, cv2.FONT_HERSHEY_SIMPLEX, size,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, text, org, cv2.FONT_HERSHEY_SIMPLEX, size,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def render_view3d(model, data, pts: np.ndarray, depths: np.ndarray,
                   near: float, far: float, w: int, h: int) -> np.ndarray:
    """BGR panel: robot skeleton + live skin returns in world (turbo: red=near)."""
    img = np.full((h, w, 3), 16, np.uint8)
    cv2.putText(img, "prox 3D", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 230, 230), 1, cv2.LINE_AA)
    bones = []
    center = np.array([0.35, 0.0, 0.45])
    if model is not None and data is not None:
        xs = []
        for i in range(model.nbody):
            name = model.body(i).name
            if not name or name == "world":
                continue
            pid = int(model.body_parentid[i])
            pb = data.xpos[i]
            xs.append(pb)
            if pid >= 0:
                bones.append((data.xpos[pid].copy(), pb.copy()))
        if xs:
            center = np.mean(np.stack(xs), axis=0)
    if len(pts):
        center = 0.6 * center + 0.4 * pts.mean(axis=0)
    eye = center + np.array([1.55, -1.75, 1.15])
    R, tvec = _look_at(eye, center)

    # ground grid
    grid = []
    for a in np.linspace(-0.6, 1.4, 9):
        grid.append((np.array([a, -0.8, 0.0]), np.array([a, 0.8, 0.0])))
        grid.append((np.array([-0.4, a - 0.4, 0.0]), np.array([1.4, a - 0.4, 0.0])))
    for a, b in grid:
        segs = np.stack([a, b])
        u, v, z, vis = _project(segs, R, tvec, w, h)
        if vis.all():
            cv2.line(img, (int(u[0]), int(v[0])), (int(u[1]), int(v[1])),
                     (40, 40, 44), 1, cv2.LINE_AA)

    for a, b in bones:
        segs = np.stack([a, b])
        u, v, z, vis = _project(segs, R, tvec, w, h)
        if vis.all():
            cv2.line(img, (int(np.clip(u[0], 0, w - 1)), int(np.clip(v[0], 0, h - 1))),
                     (int(np.clip(u[1], 0, w - 1)), int(np.clip(v[1], 0, h - 1))),
                     (180, 175, 170), 2, cv2.LINE_AA)

    n = 0
    if len(pts):
        if len(pts) > 2400:
            step = int(np.ceil(len(pts) / 2400))
            pts, depths = pts[::step], depths[::step]
        u, v, z, vis = _project(pts, R, tvec, w, h)
        order = np.argsort(-z)
        norm = np.clip((depths - near) / max(far - near, 1e-6), 0, 1)
        rgb = (_TURBO(1.0 - norm)[:, :3] * 255).astype(np.uint8)
        for i in order:
            if not vis[i]:
                continue
            x, y = int(u[i]), int(v[i])
            if 0 <= x < w and 0 <= y < h:
                col = (int(rgb[i, 2]), int(rgb[i, 1]), int(rgb[i, 0]))  # RGB->BGR
                cv2.circle(img, (x, y), 2, col, -1, cv2.LINE_AA)
                n += 1
    cv2.putText(img, f"{n} pts  red=near", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (160, 160, 160), 1, cv2.LINE_AA)
    if model is None:
        cv2.putText(img, "no FK", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (80, 80, 200), 1, cv2.LINE_AA)
    return img


def _sparkline(dst: np.ndarray, ys: np.ndarray, t: int, title: str) -> None:
    h, w = dst.shape[:2]
    dst[:] = 18
    cv2.putText(dst, title, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (200, 200, 200), 1, cv2.LINE_AA)
    if ys is None or ys.size == 0:
        return
    ys = np.asarray(ys, np.float32)
    if ys.ndim == 1:
        ys = ys[:, None]
    T, D = ys.shape
    lo = np.nanmin(ys, axis=0)
    hi = np.nanmax(ys, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    xs = np.linspace(0, w - 1, T).astype(np.int32)
    for d in range(min(D, len(_PLOT_BGR))):
        yn = (ys[:, d] - lo[d]) / span[d]
        ypix = (h - 6 - yn * (h - 22)).astype(np.int32)
        pts = np.stack([xs, np.clip(ypix, 0, h - 1)], axis=1)
        cv2.polylines(dst, [pts.reshape(-1, 1, 2)], False, _PLOT_BGR[d], 1, cv2.LINE_AA)
    xc = int(round(t / max(T - 1, 1) * (w - 1)))
    cv2.line(dst, (xc, 16), (xc, h - 1), (255, 255, 255), 1)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeRef:
    kind: str
    path: Path
    key: str
    vid_id: int
    label: str
    h5_dir: Path


@dataclass
class DatasetRoot:
    kind: str
    path: Path
    n_eps: int
    label: str


_SKIP_ALWAYS = {".git", "dataset_viz"}
_EVAL_PARTS = {
    "results", "eval", "eval_output", "eval_outputs", "eval_outputs_nonvideo",
    "eval_output_full", "final_outputs", "early_smoke_evals", "videos",
}
# Nested copies of act_style_52 / raw_h5 inside pact_20260622/data/.
_DUP_NEEDLES = (
    "pact_20260622/data/openfrontcluttered_52_act",
    "pact_20260622/data/raw_openfrontcluttered",
)


def _skip_path(p: Path, include_eval: bool) -> bool:
    parts = p.parts
    if any(x in _SKIP_ALWAYS for x in parts):
        return True
    if "debug" in parts:
        return True
    if not include_eval and any(x in _EVAL_PARTS for x in parts):
        return True
    return False


def _count_trajs(h5_path: Path) -> int:
    try:
        with h5py.File(h5_path, "r") as f:
            return sum(1 for k in f if str(k).startswith("traj_"))
    except Exception:
        return 0


def scan_dataset_roots(root: Path, include_eval: bool = False) -> list[DatasetRoot]:
    """Independent dataset folders under root (does not smash mixed trees)."""
    root = root.expanduser().resolve()
    if root.is_file():
        return []

    found: list[DatasetRoot] = []
    seen: set[Path] = set()

    act_dirs: set[Path] = set()
    for p in root.rglob("episode_*.hdf5"):
        if _skip_path(p, include_eval):
            continue
        act_dirs.add(p.parent)
    for d in sorted(act_dirs):
        if d in seen:
            continue
        n = len(list(d.glob("episode_*.hdf5")))
        found.append(DatasetRoot("act", d, n, d.name))
        seen.add(d)

    hf_roots: set[Path] = set()
    for p in root.rglob("trajectory.h5"):
        if _skip_path(p, include_eval):
            continue
        if p.parent.parent.name == "rows":
            hf_roots.add(p.parent.parent.parent)
        else:
            hf_roots.add(p.parent.parent)
    for d in sorted(hf_roots):
        if d in seen or not d.is_dir():
            continue
        n = len(list(d.glob("rows/*/trajectory.h5"))) or len(
            [x for x in d.rglob("trajectory.h5") if not _skip_path(x, include_eval)]
        )
        found.append(DatasetRoot("hf", d, n, d.name))
        seen.add(d)

    run_dirs: set[Path] = set()
    for p in root.rglob("trajectories*.h5"):
        if _skip_path(p, include_eval):
            continue
        run_dirs.add(p.parent.parent if p.parent.name.startswith("house_") else p.parent)
    for d in sorted(run_dirs):
        if d in seen:
            continue
        h5s = sorted(d.glob("house_*/trajectories*.h5")) or sorted(d.glob("trajectories*.h5"))
        n = sum(_count_trajs(h) for h in h5s)
        found.append(DatasetRoot("datagen", d, n, d.name))
        seen.add(d)

    found.sort(key=lambda x: (x.kind, str(x.path)))
    return found


def is_dup_dataset(ds: DatasetRoot) -> bool:
    s = str(ds.path)
    return any(n in s for n in _DUP_NEEDLES)


def unique_catalog(roots: list[DatasetRoot], keep_dups: bool) -> tuple[list[DatasetRoot], list[DatasetRoot]]:
    if keep_dups:
        return list(roots), []
    kept, skipped = [], []
    for ds in roots:
        (skipped if is_dup_dataset(ds) else kept).append(ds)
    return kept, skipped


def print_catalog(roots: list[DatasetRoot], scan_root: Path) -> None:
    print(f"datasets under {scan_root}  n={len(roots)}")
    for i, ds in enumerate(roots):
        try:
            rel = ds.path.relative_to(scan_root)
        except ValueError:
            rel = ds.path
        mark = "  DUP" if is_dup_dataset(ds) else ""
        print(f"  [{i:02d}] {ds.kind:8s}  eps={ds.n_eps:4d}  {rel}{mark}")


def episode_gaps(ep: Episode, *, has_fk: bool) -> list[str]:
    gaps = []
    if not any(k in ep.images for k in ("wrist_camera", "wrist", "wrist_rgb")):
        gaps.append("no wrist RGB")
    if not any(k in ep.images for k in ("exo_camera_1", "table", "table_camera", "top")):
        gaps.append("no table RGB")
    if ep.proximity is None:
        gaps.append("no proximity")
    if not has_fk:
        gaps.append("no FK (3D skeleton off)")
    elif ep.proximity is not None and not ep.cam2w:
        gaps.append("prox 3D from FK cameras (no saved cam2w)")
    return gaps


_AUDIT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>dataset viz</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root { color-scheme: dark; --bg:#111; --panel:#181818; --line:#2a2a2a; --txt:#ddd; --mut:#888; --acc:#8ab4ff; --ok:#7dce9a; --bad:#f6a; }
* { box-sizing: border-box; }
html, body { margin:0; height:100%; background:var(--bg); color:var(--txt); font:13px/1.4 ui-sans-serif, system-ui, sans-serif; }
a { color:var(--acc); }
header { display:flex; gap:12px; align-items:center; flex-wrap:wrap; padding:10px 14px; border-bottom:1px solid var(--line); background:#161616; }
h1 { font-size:15px; margin:0; font-weight:600; }
.kpis { display:flex; gap:14px; color:var(--mut); font-variant-numeric: tabular-nums; }
.kpis b { color:var(--txt); font-weight:600; }
input, select { background:#222; color:var(--txt); border:1px solid #333; padding:4px 8px; }
label.chk { color:var(--mut); }
#app { display:grid; grid-template-columns: 320px minmax(0,1fr); height: calc(100% - 48px); }
#list { overflow:auto; border-right:1px solid var(--line); }
.row { padding:8px 12px; border-bottom:1px solid var(--line); cursor:pointer; }
.row:hover { background:#1e1e1e; }
.row.on { background:#1c2a38; }
.row .slug { font:12px ui-monospace,monospace; word-break:break-all; }
.row .meta { color:var(--mut); font-size:11px; margin-top:2px; }
.chip { display:inline-block; padding:0 6px; margin:1px 2px 0 0; border:1px solid #333; font-size:11px; color:#bbb; }
.chip.gap { color:var(--bad); border-color:#633; }
.chip.ok { color:var(--ok); border-color:#364; }
.meta { color:var(--mut); font-size:11px; }
#main { overflow:auto; padding:12px; display:flex; flex-direction:column; gap:10px; }
#stats { display:flex; flex-wrap:wrap; gap:8px; }
.card { background:var(--panel); border:1px solid var(--line); padding:8px 10px; min-width:110px; }
.card .l { color:var(--mut); font-size:11px; }
.card .v { font-size:16px; font-variant-numeric: tabular-nums; }
.split { display:grid; grid-template-columns: minmax(0,1.6fr) 280px; gap:10px; min-height: 320px; }
video { width:100%; background:#000; max-height: 480px; }
#eps { overflow:auto; max-height: 480px; background:var(--panel); border:1px solid var(--line); }
#eps .grp { position:sticky; top:0; background:#223040; color:#9cf; font-size:12px; font-weight:600; padding:4px 8px; }
#eps button { display:block; width:100%; text-align:left; background:none; border:0; color:#ccc; padding:4px 8px; cursor:pointer; font:inherit; }
#eps button:hover, #eps button.active { background:#2a3a4a; color:#fff; }
.plot { height:200px; }
#msg { color:var(--mut); padding:24px; }
#src { color:var(--mut); font:11px ui-monospace,monospace; word-break:break-all; }
</style>
</head>
<body>
<header>
  <h1>dataset viz</h1>
  <div class="kpis" id="kpis">loading…</div>
  <input id="q" placeholder="filter slug / group / path" size="28"/>
  <select id="kind"><option value="all">all kinds</option></select>
  <label class="chk"><input type="checkbox" id="gaps"/> gaps only</label>
</header>
<div id="app">
  <div id="list"></div>
  <div id="main"><div id="msg">pick a dataset</div></div>
</div>
<script type="application/json" id="bootstrap">%%BOOTSTRAP%%</script>
<script>
let cat = {rows:[]};
let sel = null, tl = null, cur = null, cache = {};
const $ = id => document.getElementById(id);
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => (
    {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]
  ));
}

function fmt(n, d) { return (n==null || n!==n) ? "—" : Number(n).toFixed(d); }
function vidUrl(r, ep) {
  if (ep && ep.video) return r.slug + "/" + ep.video;
  if (r.video) return r.slug + "/" + r.video;
  return r.slug + "/dataset.mp4";
}

function renderHeader() {
  const rows = cat.rows || [];
  const nEps = rows.reduce((s,r)=>s+(r.n_eps_exported||0),0);
  const nVid = rows.reduce((s,r)=>s+(r.n_videos||0),0);
  const nGap = rows.filter(r => (r.gaps||[]).length).length;
  $("kpis").innerHTML =
    `<span><b>${rows.length}</b> datasets</span>` +
    `<span><b>${nEps}</b> eps</span>` +
    `<span><b>${nVid}</b> clips</span>` +
    `<span><b>${nGap}</b> with gaps</span>` +
    (cat.by_kind ? `<span>${Object.entries(cat.by_kind).map(([k,n])=>k+"="+n).join(" · ")}</span>` : "");
  const kinds = [...new Set(rows.map(r => r.kind).filter(Boolean))].sort();
  const k = $("kind");
  const cur = k.value;
  k.innerHTML = '<option value="all">all kinds</option>' +
    kinds.map(x => `<option value="${x}">${x}</option>`).join("");
  if ([...k.options].some(o => o.value===cur)) k.value = cur;
}

function filtered() {
  const q = $("q").value.toLowerCase();
  const kind = $("kind").value;
  const gapOnly = $("gaps").checked;
  return (cat.rows||[]).filter(r => {
    if (kind !== "all" && r.kind !== kind) return false;
    if (gapOnly && !(r.gaps||[]).length) return false;
    if (!q) return true;
    const hay = [r.slug, r.src, r.kind, ...(r.groups||[]), ...(r.cams||[])].join(" ").toLowerCase();
    return hay.includes(q);
  });
}

function renderList() {
  const rows = filtered();
  $("list").innerHTML = rows.map(r => {
    const on = sel && r.slug === sel.slug ? " on" : "";
    const gaps = (r.gaps||[]).map(g => `<span class="chip gap">${g}</span>`).join("");
    const gr = (r.groups||[]).slice(0,4).map(g => `<span class="chip">${g}</span>`).join("");
    return `<div class="row${on}" data-slug="${esc(r.slug)}">
      <div class="slug">${esc(r.slug)}</div>
      <div class="meta">${r.kind||"?"} · ${r.n_eps_exported||0}/${r.n_eps_total||"?"} eps · ${r.n_videos||0} clips
        ${r.has_prox ? "· prox" : ""} ${r.has_wrist?"· wrist":""} ${r.has_table?"· table":""}</div>
      <div>${gr}${gaps || (r.has_prox && r.has_wrist ? '<span class="chip ok">ok</span>' : "")}</div>
    </div>`;
  }).join("") || `<div class="meta" style="padding:12px">no rows</div>`;
  $("list").querySelectorAll(".row").forEach(el => {
    el.onclick = () => select(cat.rows.find(x => x.slug === el.dataset.slug));
  });
}

function cards(r, tlo) {
  const items = [
    ["kind", r.kind],
    ["exported", `${r.n_eps_exported||0} / ${r.n_eps_total||"?"}`],
    ["skipped", r.n_eps_skipped||0],
    ["clips", r.n_videos||0],
    ["duration", fmt(r.duration_s,1)+" s"],
    ["stride", r.stride],
    ["prox", r.has_prox ? (r.prox_shape||[]).join("×") : "none"],
    ["wrist", r.has_wrist ? "yes" : "no"],
    ["table", r.has_table ? "yes" : "no"],
    ["FK", r.has_fk ? "yes" : "no"],
  ];
  if (tlo) {
    items.push(["frames", tlo.n_frames], ["dt", fmt(tlo.dt,3)+" s"]);
  }
  if (r.skin_min_min != null) items.push(["skin min", fmt(r.skin_min_min,3)+" m"]);
  return items.map(([l,v]) => `<div class="card"><div class="l">${l}</div><div class="v">${v}</div></div>`).join("");
}

function renderMain() {
  const r = sel;
  if (!r) { $("main").innerHTML = '<div id="msg">pick a dataset</div>'; return; }
  const gaps = (r.gaps||[]).map(g => `<span class="chip gap">${g}</span>`).join("") || '<span class="chip ok">no gaps</span>';
  $("main").innerHTML = `
    <div id="stats">${cards(r, tl)}</div>
    <div>${gaps} ${(r.groups||[]).map(g=>`<span class="chip">${esc(g)}</span>`).join("")}
      ${(r.cams||[]).map(c=>`<span class="chip">${esc(c)}</span>`).join("")}
      <a href="${esc(r.slug)}/index.html">per-dataset html</a></div>
    <div class="split">
      <div>
        <video id="v" controls preload="none"></video>
        <div class="meta" id="hud"></div>
      </div>
      <div id="eps"></div>
    </div>
    <div id="pq" class="plot"></div>
    <div id="pv" class="plot"></div>
    <div id="ps" class="plot"></div>
    <div id="src">${esc(r.src||"")}</div>`;
  fillEpisodes();
  drawPlots();
}

function fillEpisodes() {
  const box = $("eps");
  if (!box) return;
  box.innerHTML = "";
  const eps = (tl && tl.episodes) || [];
  if (!eps.length) {
    box.innerHTML = '<div class="meta" style="padding:8px">no episode list — open after viz finishes</div>';
    const v = $("v");
    if (sel.video) { v.src = vidUrl(sel); }
    return;
  }
  const groups = new Map();
  eps.forEach(e => {
    const g = e.group || "all";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(e);
  });
  groups.forEach((list, g) => {
    const h = document.createElement("div");
    h.className = "grp";
    h.textContent = g + "  (" + list.length + ")";
    box.appendChild(h);
    list.forEach(e => {
      const b = document.createElement("button");
      b.textContent = e.label + "  T=" + e.T;
      b.onclick = () => playEp(e, true);
      box.appendChild(b);
      e._btn = b;
    });
  });
  playEp(eps[0], false);
}

function playEp(e, play) {
  cur = e;
  const v = $("v");
  if (!v) return;
  const url = vidUrl(sel, e);
  if (v.getAttribute("src") !== url) v.src = url;
  if (tl && tl.per_episode) v.currentTime = 0;
  else if (e) v.currentTime = e.start_s || 0;
  if (play) v.play();
  ((tl && tl.episodes) || []).forEach(x => x._btn && x._btn.classList.toggle("active", x===e));
  zoom(e);
  v.ontimeupdate = () => {
    const per = tl && tl.per_episode;
    const t = per ? ((cur ? cur.start_s : 0) + v.currentTime) : v.currentTime;
    cursor(t);
    const span = per && cur ? (cur.dur_s || 0) : (tl ? tl.duration_s : 0);
    $("hud").textContent = (cur ? cur.label : "") + "   t=" +
      (per ? v.currentTime : t).toFixed(2) + "s / " + Number(span||0).toFixed(1) + "s";
  };
}

function plotLayout(title) {
  return {margin:{t:28,r:12,b:32,l:48}, paper_bgcolor:"#111", plot_bgcolor:"#161616",
    font:{color:"#ccc", size:11}, legend:{orientation:"h", y:1.12},
    title, xaxis:{title:"t (s)", gridcolor:"#333"}, yaxis:{gridcolor:"#333"}};
}
function traces(group, names) {
  if (!tl || !tl[group]) return [];
  return names.filter(n => tl[group][n]).map(n => ({
    x: tl.t, y: tl[group][n], name: n, type: "scattergl", mode: "lines", line:{width:1}
  }));
}
function drawPlots() {
  if (typeof Plotly === "undefined") {
    const el = $("pq");
    if (el) el.innerHTML = '<div class="meta">Plotly CDN blocked — plots skip</div>';
    return;
  }
  if (!tl) {
    ["pq","pv","ps"].forEach(id => { const el=$(id); if (el) el.innerHTML = ""; });
    return;
  }
  Plotly.react("pq", traces("qpos", ["q1","q2","q3","q4","q5","q6","q7"]), plotLayout("qpos (rad)"));
  Plotly.react("pv", traces("qvel", ["v1","v2","v3","v4","v5","v6","v7"]), plotLayout("qvel (rad/s)"));
  const skin = tl.skin_min ? [{x: tl.t, y: tl.skin_min, name:"skin_min", type:"scattergl", mode:"lines", line:{width:1.4}}] : [];
  Plotly.react("ps", skin, plotLayout("skin min (m)"));
}
function cursor(t) {
  if (typeof Plotly === "undefined") return;
  const shape = [{type:"line", x0:t, x1:t, y0:0, y1:1, yref:"paper", line:{color:"#fff", width:1}}];
  ["pq","pv","ps"].forEach(id => { if ($(id)) Plotly.relayout(id, {shapes: shape}); });
}
function zoom(e) {
  if (typeof Plotly === "undefined" || !tl) return;
  const per = tl.per_episode;
  const r = (per && e) ? {"xaxis.range": [e.start_s, e.start_s + (e.dur_s || 1)]} : {"xaxis.autorange": true};
  ["pq","pv","ps"].forEach(id => { if ($(id)) Plotly.relayout(id, r); });
}

function loadTimelineJs(slug) {
  return new Promise((resolve, reject) => {
    const prev = document.getElementById("tl-script");
    if (prev) prev.remove();
    const s = document.createElement("script");
    s.id = "tl-script";
    s.src = slug + "/timeline.js";
    s.onload = () => resolve(window.DATASET_TIMELINE || null);
    s.onerror = () => reject(new Error("timeline.js"));
    document.head.appendChild(s);
  });
}

function select(r) {
  if (!r) return;
  sel = r;
  try { history.replaceState(null, "", "#" + encodeURIComponent(r.slug)); } catch (e) {}
  renderList();
  tl = cache[r.slug] || null;
  renderMain();
  if (cache[r.slug]) return;
  loadTimelineJs(r.slug).then(data => {
    if (!sel || sel.slug !== r.slug) return;
    tl = data;
    if (data) cache[r.slug] = data;
    renderMain();
  }).catch(() => {
    const hud = $("hud");
    if (hud && !tl) hud.textContent = "no timeline.js — python scripts/dataset_viz.py --dashboard";
  });
}

function pickInitial() {
  const hash = decodeURIComponent((location.hash || "").replace(/^#/, ""));
  const rows = cat.rows || [];
  const fromHash = hash && rows.find(x => x.slug === hash);
  const shown = filtered();
  select(fromHash || shown[0] || rows[0] || null);
}

$("q").oninput = $("kind").onchange = $("gaps").onchange = renderList;
document.addEventListener("keydown", e => {
  if (e.target && e.target.tagName === "INPUT") return;
  const rows = filtered();
  if (!rows.length) return;
  const i = Math.max(0, rows.findIndex(x => sel && x.slug === sel.slug));
  if (e.key === "j" || e.key === "ArrowDown") {
    e.preventDefault();
    select(rows[Math.min(rows.length - 1, i + 1)]);
  } else if (e.key === "k" || e.key === "ArrowUp") {
    e.preventDefault();
    select(rows[Math.max(0, i - 1)]);
  } else if ((e.key === "n" || e.key === "N") && tl && tl.episodes) {
    const ei = Math.max(0, tl.episodes.indexOf(cur));
    playEp(tl.episodes[Math.min(tl.episodes.length - 1, ei + 1)], true);
  } else if ((e.key === "p" || e.key === "P") && tl && tl.episodes) {
    const ei = Math.max(0, tl.episodes.indexOf(cur));
    playEp(tl.episodes[Math.max(0, ei - 1)], true);
  }
});
try { cat = JSON.parse($("bootstrap").textContent); } catch (e) { cat = {rows:[]}; }
renderHeader();
pickInitial();
</script>
</body>
</html>
"""


def _collect_audit_rows(out_base: Path) -> list[dict]:
    """Walk dataset folders only. Skip episodes/ (all the mp4s)."""
    skip_slugs = {"openfront_52"}
    skip_dirs = {"episodes", "__pycache__"}
    rows = []
    root = out_base.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        if "audit.json" not in filenames:
            continue
        folder = Path(dirpath)
        if folder == root:
            continue
        slug = str(folder.relative_to(root))
        if slug in skip_slugs:
            continue
        try:
            rec = json.loads((folder / "audit.json").read_text())
        except Exception:
            continue
        rec["slug"] = slug
        rows.append(rec)
    rows.sort(key=lambda r: r.get("slug") or "")
    return rows


def write_timeline_js(out_dir: Path, timeline: dict | None = None) -> None:
    """timeline.js is a script tag payload so VS Code / file preview can plot without fetch."""
    if timeline is None:
        src = out_dir / "timeline.json"
        if not src.is_file():
            return
        raw = src.read_text().strip()
        dst = out_dir / "timeline.js"
        if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
            return
    else:
        raw = json.dumps(timeline)
        (out_dir / "timeline.json").write_text(raw)
    (out_dir / "timeline.js").write_text("window.DATASET_TIMELINE = " + raw + ";\n")


def write_audit_index(
    out_base: Path,
    n_skipped_dups: int = 0,
    quiet: bool = False,
) -> None:
    """Write the dashboard catalog. Cheap. No encode."""
    out_base.mkdir(parents=True, exist_ok=True)
    rows = _collect_audit_rows(out_base)
    for rec in rows:
        slug = rec.get("slug")
        if slug:
            write_timeline_js(out_base / slug)
    kinds = Counter(r.get("kind") or "?" for r in rows)
    catalog = {
        "n": len(rows),
        "n_skipped_dups": n_skipped_dups,
        "n_eps": sum(int(r.get("n_eps_exported") or 0) for r in rows),
        "n_videos": sum(int(r.get("n_videos") or 0) for r in rows),
        "n_gaps": sum(1 for r in rows if r.get("gaps")),
        "by_kind": dict(kinds),
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": rows,
    }
    (out_base / "audit.json").write_text(json.dumps(catalog) + "\n")
    boot = json.dumps(catalog).replace("</", "<\\/")
    (out_base / "index.html").write_text(_AUDIT_HTML.replace("%%BOOTSTRAP%%", boot))
    if not quiet:
        print(f"audit index  {out_base / 'index.html'}  n={len(rows)}")


def discover(root: Path) -> tuple[str, list[EpisodeRef]]:
    root = root.expanduser().resolve()
    if root.is_file():
        return _discover_file(root)

    act = sorted(root.glob("episode_*.hdf5")) + sorted(root.glob("episode_*.h5"))
    rows = sorted(
        p for p in root.glob("**/trajectory.h5")
        if p.is_file() and not _skip_path(p, include_eval=True)
    )
    batches = sorted(
        p for p in root.glob("**/trajectories*.h5")
        if p.is_file() and not _skip_path(p, include_eval=True)
    )

    if act:
        refs = [
            EpisodeRef("act", p, "", 0, p.stem, p.parent)
            for p in act
        ]
        refs.sort(key=lambda r: (_ep_num(r.path.name), r.path.name))
        return "act", refs

    if rows:
        refs = []
        for p in rows:
            refs.append(EpisodeRef(
                "datagen", p, "traj_0", 0,  # vid_id=0 is traj index; mp4 id may be a sha
                f"{p.parent.name}/traj_0", p.parent,
            ))
        refs.sort(key=lambda r: r.path.parent.name)
        return "hf", refs

    if batches:
        refs = []
        for p in batches:
            with h5py.File(p, "r") as f:
                idxs = sorted(int(k.split("_")[1]) for k in f if k.startswith("traj_"))
            for i in idxs:
                refs.append(EpisodeRef(
                    "datagen", p, f"traj_{i}", i,
                    f"{p.parent.name}/traj_{i}", p.parent,
                ))
        return "datagen", refs

    raise SystemExit(
        f"no trajectory hdf5 under {root}\n"
        "  expected episode_*.hdf5  OR  **/trajectory.h5  OR  **/trajectories*.h5"
    )


def _discover_file(path: Path) -> tuple[str, list[EpisodeRef]]:
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if "observations" in f and "qpos" in f["observations"]:
            return "act", [EpisodeRef("act", path, "", 0, path.stem, path.parent)]
        trajs = sorted(int(k.split("_")[1]) for k in keys if k.startswith("traj_"))
        if trajs:
            refs = [
                EpisodeRef("datagen", path, f"traj_{i}", i,
                           f"{path.parent.name}/traj_{i}", path.parent)
                for i in trajs
            ]
            return "datagen", refs
    raise SystemExit(f"{path} is not an ACT episode hdf5 or a molmospaces traj_* file")


def glob_mp4(h5_dir: Path, vid_id: int, stem: str) -> Path | None:
    """Sidecar mp4 for one camera stem next to the h5.

    Name patterns this repo actually writes:

      episode_{idx:08d}_{stem}.mp4                  HF v5 clones
      episode_{idx:08d}_{stem}_batch_*_of_*.mp4     datagen houses
      episode_{sha256}_{stem}.mp4                   v10 hallway dumps
                                                    (folder name == sha)
    """
    padded = f"{vid_id:08d}"
    exact = sorted(h5_dir.glob(f"episode_{padded}_{stem}.mp4"))
    if exact:
        return exact[0]
    batch = sorted(h5_dir.glob(f"episode_{padded}_{stem}_batch_*.mp4"))
    if batch:
        return batch[0]
    hashed = h5_dir / f"episode_{h5_dir.name}_{stem}.mp4"
    if hashed.is_file():
        return hashed
    loose = sorted(
        p for p in h5_dir.glob(f"episode_*_{stem}.mp4")
        if "_batch_" not in p.name
    )
    if len(loose) == 1:
        return loose[0]
    return None


def decode_mp4(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return np.zeros((0, 1, 1, 3), np.uint8)
    return np.stack(frames)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
@dataclass
class Episode:
    label: str
    T: int
    dt: float
    images: dict
    qpos: np.ndarray
    qvel: np.ndarray
    action: np.ndarray | None
    proximity: np.ndarray | None
    sensor_names: list
    cam2w: dict
    tcp: np.ndarray | None
    base: np.ndarray | None
    phase: np.ndarray | None
    success: bool | None
    scene: dict
    targets: dict
    attrs: dict
    extras: dict = field(default_factory=dict)
    embeddings: np.ndarray | None = None
    cam_params: dict = field(default_factory=dict)  # rgb cam -> extrinsic_cv / intrinsic_cv


def load_act(ref: EpisodeRef, prox_pool: str, dt_override: float | None) -> Episode:
    with h5py.File(ref.path, "r") as f:
        obs = f["observations"]
        qpos = np.asarray(obs["qpos"], np.float32)
        T = int(qpos.shape[0])
        qvel = np.asarray(obs["qvel"], np.float32) if "qvel" in obs else np.zeros_like(qpos)
        action = np.asarray(f["action"], np.float32) if "action" in f else None
        images = {}
        if "images" in obs:
            for name in obs["images"]:
                images[name] = _as_rgb(obs["images"][name][:T])
        proximity = None
        names: list[str] = []
        if "proximity" in obs:
            proximity = np.asarray(obs["proximity"][:T], np.float32)
            n = int(proximity.shape[1]) if proximity.ndim >= 2 else 0
            names = list(HYBRID_SKIN_SENSOR_ORDER[:n]) if n <= len(HYBRID_SKIN_SENSOR_ORDER) \
                else [f"sensor_{i}" for i in range(n)]
        emb = None
        if "proximity_embeddings" in obs:
            emb = np.asarray(obs["proximity_embeddings"][:T], np.float32)
        extras = {}
        for key in ("rewards", "success", "fail"):
            if key in f and np.asarray(f[key]).shape[:1] == (T,):
                extras[key] = np.asarray(f[key][:T], np.float32)
        for key in ("proximity_valid",):
            if key in obs:
                arr = np.asarray(obs[key][:T])
                extras["prox_valid_frac"] = arr.reshape(T, -1).astype(np.float32).mean(axis=1)
        attrs = _attrs(f)
        attrs["file"] = str(ref.path)
    return Episode(
        label=ref.label, T=T, dt=dt_override or DEFAULT_DT, images=images,
        qpos=qpos, qvel=qvel, action=action, proximity=proximity,
        sensor_names=names, cam2w={}, tcp=None, base=None, phase=None,
        success=bool(attrs["clean_success"]) if "clean_success" in attrs else None,
        scene={}, targets={}, attrs=attrs, extras=extras, embeddings=emb,
    )


def load_datagen(ref: EpisodeRef, prox_pool: str, dt_override: float | None,
                 include_sensor_rgb: bool, include_depth: bool) -> Episode:
    with h5py.File(ref.path, "r") as f:
        if ref.key not in f:
            raise SystemExit(f"{ref.path} missing {ref.key}")
        t = f[ref.key]
        tcp = np.asarray(t["obs/extra/tcp_pose"], np.float64) if "obs/extra/tcp_pose" in t else None
        T = int(tcp.shape[0]) if tcp is not None else int(t["obs/agent/qpos"].shape[0])
        base = np.asarray(t["obs/extra/robot_base_pose"], np.float64) if "obs/extra/robot_base_pose" in t else None
        phase = np.asarray(t["obs/extra/policy_phase"][:T], int) if "obs/extra/policy_phase" in t else None
        success = bool(np.asarray(t["success"])[-1]) if "success" in t else None
        qpos_blob = t["obs/agent/qpos"][:T]
        qvel_blob = t["obs/agent/qvel"][:T] if "obs/agent/qvel" in t else None
        act_blob = t["actions/joint_pos"][:T] if "actions/joint_pos" in t else None
        q_rows, dq_rows, grip = [], [], []
        for i in range(T):
            qp = _json_row(qpos_blob[i])
            arm = list(qp.get("arm") or [0.0] * 7)
            g = list(qp.get("gripper") or [0.0])
            q_rows.append(arm[:7] + (g[:2] if len(g) >= 2 else g + [0.0]))
            qv = _json_row(qvel_blob[i]) if qvel_blob is not None else {}
            arm_v = list(qv.get("arm") or [0.0] * 7)
            gv = list(qv.get("gripper") or [0.0, 0.0])
            dq_rows.append(arm_v[:7] + (gv[:2] if len(gv) >= 2 else gv + [0.0]))
            grip.append(float((g or [0.0])[0]))
        qpos = np.asarray(q_rows, np.float32)
        qvel = np.asarray(dq_rows, np.float32)
        action = None
        if act_blob is not None:
            a_rows = []
            for i in range(T):
                d = _json_row(act_blob[i])
                arm = list(d.get("arm") or [0.0] * 7)
                g = list(d.get("gripper") or [0.0])
                a_rows.append(arm[:7] + [float(g[0]) if g else 0.0])
            action = np.asarray(a_rows, np.float32)
        prox = None
        names: list[str] = []
        cam2w = {}
        cam_params = {}
        for cam in RGB_STEMS:
            grp = f"obs/sensor_param/{cam}"
            if f"{grp}/extrinsic_cv" in t and f"{grp}/intrinsic_cv" in t:
                cam_params[cam] = {
                    "extrinsic_cv": np.asarray(t[f"{grp}/extrinsic_cv"][:T], np.float64),
                    "intrinsic_cv": np.asarray(t[f"{grp}/intrinsic_cv"][:T], np.float64),
                }
        if "obs/proximity" in t:
            order = [n for n in HYBRID_SKIN_SENSOR_ORDER if n in t["obs/proximity"]]
            extra = [n for n in t["obs/proximity"] if n not in order]
            names = order + extra
            chans = [_pool_prox(t[f"obs/proximity/{n}"][:T], prox_pool) for n in names]
            prox = np.stack(chans, axis=1) if chans else None
            if "obs/sensor_param" in t:
                for n in names:
                    p = f"obs/sensor_param/{n}/cam2world_gl"
                    if p in t:
                        cam2w[n] = t[p][:T].astype(np.float64)
        scene = _decode_obs_scene(t)
        targets = {}
        for k in ("obj_start", "obj_end"):
            if f"obs/extra/{k}" in t:
                targets[k] = np.asarray(t[f"obs/extra/{k}"][0], np.float64)
        extras = {}
        for key in ("rewards", "success", "fail", "terminated", "truncated"):
            if key in t and np.asarray(t[key]).shape[:1] == (T,):
                extras[key] = np.asarray(t[key][:T], np.float32)
        if phase is not None:
            extras["phase"] = phase.astype(np.float32)
        attrs = {"file": str(ref.path), "traj": ref.key}
        attrs.update({k: scene[k] for k in ("behavior_class", "task_description") if k in scene})
        dt = dt_override or float(scene.get("policy_dt_ms", DEFAULT_DT * 1000.0)) / 1000.0

    images: dict[str, np.ndarray] = {}
    stems = list(RGB_STEMS)
    if include_sensor_rgb or include_depth:
        for s in OPTIONAL_STEMS:
            if include_sensor_rgb and s == "sensors_rgb256":
                stems.append(s)
            if include_depth and s.endswith("_depth"):
                stems.append(s)
    for stem in stems:
        p = glob_mp4(ref.h5_dir, ref.vid_id, stem)
        if p is None:
            continue
        frames = decode_mp4(p)
        if stem in CAM_ALIAS:
            images[stem] = frames
        elif stem == "sensors_rgb256":
            images["sensors_rgb"] = frames
        else:
            images[stem] = frames

    return Episode(
        label=ref.label, T=T, dt=dt, images=images, qpos=qpos, qvel=qvel,
        action=action, proximity=prox, sensor_names=names, cam2w=cam2w,
        tcp=tcp, base=base, phase=phase, success=success, scene=scene,
        targets=targets, attrs=attrs, extras=extras, cam_params=cam_params,
    )


def load_episode(ref: EpisodeRef, **kw) -> Episode:
    if ref.kind == "act":
        return load_act(ref, prox_pool=kw["prox_pool"], dt_override=kw["dt"])
    return load_datagen(
        ref, prox_pool=kw["prox_pool"], dt_override=kw["dt"],
        include_sensor_rgb=kw["include_sensor_rgb"],
        include_depth=kw["include_depth"],
    )


def peek_summary(ref: EpisodeRef) -> str:
    with h5py.File(ref.path, "r") as f:
        if ref.kind == "act":
            obs = f["observations"]
            T = int(obs["qpos"].shape[0])
            cams = list(obs["images"].keys()) if "images" in obs else []
            prox = tuple(obs["proximity"].shape) if "proximity" in obs else None
            return f"{ref.label:28s} T={T:4d}  cams={cams}  prox={prox}  qpos={obs['qpos'].shape}"
        t = f[ref.key]
        T = int(t["obs/extra/tcp_pose"].shape[0]) if "obs/extra/tcp_pose" in t \
            else int(t["obs/agent/qpos"].shape[0])
        nprox = len(t["obs/proximity"]) if "obs/proximity" in t else 0
        mp4s = [glob_mp4(ref.h5_dir, ref.vid_id, stem) for stem in RGB_STEMS]
        n_rgb = sum(p is not None for p in mp4s)
        return (f"{ref.label:28s} T={T:4d}  prox_sensors={nprox:2d}  "
                f"rgb_mp4s={n_rgb}  file={ref.path.name}")


# --------------------------------------------------------------------------- #
# Joint / skin JSON
# --------------------------------------------------------------------------- #
def joint_msg(ep: Episode, t: int) -> dict:
    q, dq = ep.qpos[t], ep.qvel[t]
    msg = {f"q{i+1}": float(q[i]) for i in range(min(7, q.shape[0]))}
    msg.update({f"v{i+1}": float(dq[i]) for i in range(min(7, dq.shape[0]))})
    if q.shape[0] > 7:
        msg["grip"] = float(q[7])
    if q.shape[0] > 8:
        msg["grip2"] = float(q[8])
    if ep.action is not None and t < len(ep.action):
        a = ep.action[t]
        msg.update({f"a{i+1}": float(a[i]) for i in range(min(7, a.shape[0]))})
        if a.shape[0] > 7:
            msg["a_grip"] = float(a[7])
    if ep.proximity is not None:
        d = ep.proximity[t]
        per = d.reshape(d.shape[0], -1)
        finite = np.where(np.isfinite(per), per, np.inf)
        mins = np.min(finite, axis=1)
        msg["skin_min"] = float(np.min(mins)) if mins.size else float("nan")
        by_link: dict[str, list[float]] = {}
        for i, n in enumerate(ep.sensor_names):
            by_link.setdefault(_link_prefix(n), []).append(float(mins[i]))
        for link, vals in by_link.items():
            msg[f"skin_{link}"] = float(min(vals))
    for k, arr in ep.extras.items():
        if t < len(arr) and np.ndim(arr[t]) == 0:
            msg[str(k)] = float(arr[t])
    msg["t"] = int(t)
    msg["T"] = int(ep.T)
    return msg


def _index(arr: np.ndarray | None, t: int, T: int):
    if arr is None or len(arr) == 0:
        return None
    if len(arr) == T:
        return arr[t]
    return arr[int(round(t * (len(arr) - 1) / max(T - 1, 1)))]


# --------------------------------------------------------------------------- #
# Foxglove layout
# --------------------------------------------------------------------------- #
def build_layout(has_table: bool, has_wrist: bool, has_heatmap: bool,
                 has_3d: bool) -> dict:
    plots_q = {
        "title": "Joint positions (rad)",
        "paths": [{"value": f"/joints.q{i}", "enabled": True, "timestampMethod": "receiveTime"}
                  for i in range(1, 8)],
        "showXAxisLabels": True, "showYAxisLabels": True, "showLegend": True,
        "isSynced": True, "xAxisVal": "timestamp",
    }
    plots_v = {
        "title": "Joint velocities (rad/s)",
        "paths": [{"value": f"/joints.v{i}", "enabled": True, "timestampMethod": "receiveTime"}
                  for i in range(1, 8)],
        "showXAxisLabels": True, "showYAxisLabels": True, "showLegend": True,
        "isSynced": True, "xAxisVal": "timestamp",
    }
    plots_a = {
        "title": "Action (rad) + gripper",
        "paths": (
            [{"value": f"/joints.a{i}", "enabled": True, "timestampMethod": "receiveTime"}
             for i in range(1, 8)]
            + [{"value": "/joints.a_grip", "enabled": True, "timestampMethod": "receiveTime"},
               {"value": "/joints.grip", "enabled": True, "timestampMethod": "receiveTime"}]
        ),
        "showXAxisLabels": True, "showYAxisLabels": True, "showLegend": True,
        "isSynced": True, "xAxisVal": "timestamp",
    }
    plots_skin = {
        "title": "Skin min distance (m)",
        "paths": [
            {"value": "/joints.skin_min", "enabled": True, "timestampMethod": "receiveTime"},
            {"value": "/joints.skin_link1", "enabled": True, "timestampMethod": "receiveTime"},
            {"value": "/joints.skin_link2", "enabled": True, "timestampMethod": "receiveTime"},
            {"value": "/joints.skin_link3", "enabled": True, "timestampMethod": "receiveTime"},
            {"value": "/joints.skin_link4", "enabled": True, "timestampMethod": "receiveTime"},
            {"value": "/joints.skin_link5_back", "enabled": True, "timestampMethod": "receiveTime"},
            {"value": "/joints.skin_link5_front", "enabled": True, "timestampMethod": "receiveTime"},
            {"value": "/joints.skin_link6", "enabled": True, "timestampMethod": "receiveTime"},
        ],
        "showXAxisLabels": True, "showYAxisLabels": True, "showLegend": True,
        "isSynced": True, "xAxisVal": "timestamp",
    }
    cfg: dict = {
        "Plot!q": plots_q,
        "Plot!v": plots_v,
        "Plot!a": plots_a,
        "Plot!skin": plots_skin,
        "Log!task": {"topicToRender": "/task", "searchTerms": [], "minLogLevel": 1},
    }
    if has_3d:
        cfg["3D!main"] = {
            "followMode": "follow-pose",
            "followTf": "robot_0/base",
            "scene": {"transforms": {"axisScale": 0.0, "showLabel": False}},
            "cameraState": {
                "distance": 2.6, "perspective": True, "phi": 62, "thetaOffset": 40,
                "targetOffset": [0.45, 0, 0.85], "fovy": 45, "near": 0.01, "far": 100,
            },
            "topics": {
                "/robot": {"visible": True},
                "/proximity": {"visible": True, "pointSize": 5, "colorMode": "rgba-fields"},
                "/scene_gt": {"visible": True},
                "/targets": {"visible": True},
                "/tcp": {"visible": True, "axisScale": 0.12},
            },
            "layers": {},
        }
    if has_wrist:
        cfg["Image!wrist"] = {
            "imageMode": {"imageTopic": "/camera/wrist"},
            "cameraTopic": "/camera/wrist",
        }
    if has_table:
        cfg["Image!table"] = {
            "imageMode": {"imageTopic": "/camera/table"},
            "cameraTopic": "/camera/table",
        }
    if has_heatmap:
        cfg["Image!heat"] = {
            "imageMode": {"imageTopic": "/sensors/heatmap"},
            "cameraTopic": "/sensors/heatmap",
        }

    cam_col: dict | str
    cam_ids = [k for k in ("Image!wrist", "Image!table", "Image!heat") if k in cfg]
    if not cam_ids:
        cam_col = "Log!task"
    elif len(cam_ids) == 1:
        cam_col = cam_ids[0]
    else:
        cam_col = {"direction": "column", "first": cam_ids[0], "second": cam_ids[1],
                   "splitPercentage": 50}
        for extra in cam_ids[2:]:
            cam_col = {"direction": "column", "first": cam_col, "second": extra,
                       "splitPercentage": 55}

    plots = {
        "direction": "column", "splitPercentage": 25,
        "first": "Plot!q",
        "second": {
            "direction": "column", "splitPercentage": 33,
            "first": "Plot!v",
            "second": {
                "direction": "column", "splitPercentage": 50,
                "first": "Plot!a",
                "second": "Plot!skin",
            },
        },
    }
    left: dict | str
    if has_3d:
        left = {
            "direction": "row", "splitPercentage": 58,
            "first": "3D!main",
            "second": cam_col,
        }
    else:
        left = cam_col
    layout = {
        "direction": "row", "splitPercentage": 70,
        "first": {
            "direction": "column", "splitPercentage": 78,
            "first": left,
            "second": "Log!task",
        },
        "second": plots,
    }
    return {
        "configById": cfg,
        "globalVariables": {},
        "userNodes": {},
        "playbackConfig": {"speed": 1.0},
        "layout": layout,
    }


# --------------------------------------------------------------------------- #
# HTML player
# --------------------------------------------------------------------------- #
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>dataset viz</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #111; color: #ddd; font: 14px/1.4 ui-sans-serif, system-ui, sans-serif; }
  header { padding: 12px 16px; background: #1a1a1a; border-bottom: 1px solid #333; }
  h1 { font-size: 16px; margin: 0 0 4px; }
  .sub { color: #888; font-size: 12px; }
  main { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(280px, 1fr); gap: 8px; padding: 8px; }
  video { width: 100%; background: #000; border-radius: 4px; }
  #ep { max-height: 460px; overflow: auto; background: #1a1a1a; border-radius: 4px; }
  #ep .grp { position: sticky; top: 0; background: #223040; color: #9cf; font-size: 12px;
             font-weight: 600; padding: 4px 8px; }
  #ep button { display: block; width: 100%; text-align: left; background: none; border: 0;
               color: #ccc; padding: 4px 8px; cursor: pointer; font: inherit; }
  #ep button:hover, #ep button.active { background: #2a3a4a; color: #fff; }
  #plots { grid-column: 1 / -1; }
  .plot { height: 220px; }
  a { color: #8ab4ff; }
</style>
</head>
<body>
<header>
  <h1 id="title">dataset viz</h1>
  <div class="sub">
    One short clip per episode, grouped by trajectory type. Plots follow the playhead.
    Foxglove: open <code>dataset.mcap</code>, import <code>foxglove_layout.json</code>
    (<a href="https://app.foxglove.dev">app.foxglove.dev</a> or desktop).
  </div>
</header>
<main>
  <div>
    <video id="v" controls></video>
    <div class="sub" id="hud"></div>
  </div>
  <div id="ep"></div>
  <div id="plots">
    <div id="pq" class="plot"></div>
    <div id="pv" class="plot"></div>
    <div id="ps" class="plot"></div>
  </div>
</main>
<script src="timeline.js"></script>
<script>
function init(tl) {
  document.getElementById("title").textContent = tl.title + "  (" + tl.n_episodes + " eps, " +
    (tl.duration_s).toFixed(1) + "s)";
  const v = document.getElementById("v");
  const box = document.getElementById("ep");
  const per = !!tl.per_episode;
  let cur = null;
  function select(e, play) {
    cur = e;
    if (per) {
      if (e.video && v.getAttribute("src") !== e.video) v.src = e.video;
      v.currentTime = 0;
    } else {
      v.currentTime = e.start_s;
    }
    if (play) v.play();
    tl.episodes.forEach(x => x._btn.classList.toggle("active", x === e));
    zoom(e);
  }
  const groups = new Map();
  tl.episodes.forEach(e => {
    const g = e.group || "all";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(e);
  });
  groups.forEach((eps, g) => {
    const h = document.createElement("div");
    h.className = "grp";
    h.textContent = g + "  (" + eps.length + ")";
    box.appendChild(h);
    eps.forEach(e => {
      const b = document.createElement("button");
      b.textContent = e.label + "  T=" + e.T;
      b.onclick = () => select(e, true);
      box.appendChild(b);
      e._btn = b;
    });
  });
  if (!per) v.src = tl.video;
  const layout = {margin:{t:24,r:12,b:32,l:48}, paper_bgcolor:"#111", plot_bgcolor:"#161616",
    font:{color:"#ccc", size:11}, legend:{orientation:"h", y:1.12},
    xaxis:{title:"t (s)", gridcolor:"#333"}, yaxis:{gridcolor:"#333"}};
  function traces(group, names) {
    return names.filter(n => tl[group] && tl[group][n]).map(n => ({
      x: tl.t, y: tl[group][n], name: n, type: "scattergl", mode: "lines", line:{width:1}
    }));
  }
  Plotly.newPlot("pq", traces("qpos", ["q1","q2","q3","q4","q5","q6","q7"]),
                 Object.assign({}, layout, {title:"qpos (rad)"}));
  Plotly.newPlot("pv", traces("qvel", ["v1","v2","v3","v4","v5","v6","v7"]),
                 Object.assign({}, layout, {title:"qvel (rad/s)"}));
  const skin = [];
  if (tl.skin_min) skin.push({x: tl.t, y: tl.skin_min, name: "skin_min", type:"scattergl", mode:"lines", line:{width:1.4}});
  Plotly.newPlot("ps", skin, Object.assign({}, layout, {title:"skin min (m)"}));
  function cursor(t) {
    const shape = [{type:"line", x0:t, x1:t, y0:0, y1:1, yref:"paper",
                    line:{color:"#fff", width:1}}];
    ["pq","pv","ps"].forEach(id => Plotly.relayout(id, {shapes: shape}));
  }
  function zoom(e) {
    const r = (per && e) ? {"xaxis.range": [e.start_s, e.start_s + (e.dur_s || 1)]}
                         : {"xaxis.autorange": true};
    ["pq","pv","ps"].forEach(id => Plotly.relayout(id, r));
  }
  v.ontimeupdate = () => {
    const t = per ? ((cur ? cur.start_s : 0) + v.currentTime) : v.currentTime;
    cursor(t);
    if (!per) {
      cur = tl.episodes[0];
      for (const e of tl.episodes) if (t >= e.start_s) cur = e;
      tl.episodes.forEach(e => e._btn.classList.toggle("active", e === cur));
    }
    const span = per && cur ? (cur.dur_s || 0) : tl.duration_s;
    document.getElementById("hud").textContent =
      (cur ? cur.label : "") + "   t=" + (per ? v.currentTime : t).toFixed(2) +
      "s / " + span.toFixed(1) + "s";
  };
  if (tl.episodes.length) select(tl.episodes[0], false);
}
if (window.DATASET_TIMELINE) init(window.DATASET_TIMELINE);
</script>
</body>
</html>
"""


def write_html(out_dir: Path, timeline: dict) -> None:
    (out_dir / "index.html").write_text(_HTML)
    write_timeline_js(out_dir, timeline)


def downsample_series(t: list, series: dict, max_pts: int = 8000) -> tuple[list, dict]:
    n = len(t)
    if n <= max_pts:
        return t, series
    stride = int(np.ceil(n / max_pts))
    t2 = t[::stride]
    out = {k: (v[::stride] if isinstance(v, list) else
               {kk: vv[::stride] for kk, vv in v.items()})
           for k, v in series.items()}
    return t2, out


# --------------------------------------------------------------------------- #
# MCAP + MP4 export
# --------------------------------------------------------------------------- #
class Channels:
    def __init__(self, ctx, image_topics: list[str]):
        self.tf = C.FrameTransformsChannel("/tf", context=ctx)
        self.mesh = C.SceneUpdateChannel("/robot", context=ctx)
        self.gt = C.SceneUpdateChannel("/scene_gt", context=ctx)
        self.tgt = C.SceneUpdateChannel("/targets", context=ctx)
        self.pc = C.PointCloudChannel("/proximity", context=ctx)
        self.tcp = C.PoseInFrameChannel("/tcp", context=ctx)
        self.log = C.LogChannel("/task", context=ctx)
        self.img = {tp: C.CompressedImageChannel(tp, context=ctx) for tp in image_topics}
        self.joints = foxglove.Channel("/joints", message_encoding="json", context=ctx)


def _jpeg(rgb: np.ndarray, quality: int) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def _init_robot(sensor_names: list[str], mount_z: float):
    hybrid = (not sensor_names) or any(
        "link5_front" in s or s.startswith("link1_") for s in sensor_names
    )
    robot_xml = fv.ROBOT_DIR / ("model_hybrid.xml" if hybrid else "model.xml")
    if not robot_xml.is_file():
        return None, None, None, []
    model = fv.build_robot_model(robot_xml, mount_z)
    data = __import__("mujoco").MjData(model)
    mesh_update = fv.robot_mesh_scene_update(
        fv.extract_body_meshes(model), Timestamp(sec=0, nsec=0)
    )
    pub_bodies = sorted({
        model.body(i).name for i in range(model.nbody)
        if model.body(i).name and model.body(i).name != "world"
    })
    return model, data, mesh_update, pub_bodies


def _ee_body(model) -> str | None:
    names = {model.body(i).name for i in range(model.nbody)}
    for n in _EE_BODIES:
        if n in names:
            return n
    return None


def export_episode(ep: Episode, ch: Channels | None, *, model, data, mesh_update,
                   pub_bodies, offset_ns: int, near: float, far: float,
                   d_max: float, stride: int, jpeg_q: int,
                   writer, cam3d: bool = False) -> tuple[int, list]:
    """Write one episode. Returns (duration_ns, per-frame records for timeline)."""
    mujoco = __import__("mujoco") if model is not None else None
    T, dt = ep.T, ep.dt
    arm_qadr, finger_qadr, base_mid, cam_id, ee_name = [], [], None, {}, None
    if model is not None:
        ns = fv.NS
        joint_names = [model.joint(i).name for i in range(model.njnt)]
        arm_qadr = [model.joint(f"{ns}fr3_joint{i}").qposadr[0] for i in range(1, 8)
                    if f"{ns}fr3_joint{i}" in joint_names]
        finger_qadr = [model.joint(f"{ns}gripper/{n}").qposadr[0]
                       for n in ("left_driver_joint", "right_driver_joint")
                       if f"{ns}gripper/{n}" in joint_names]
        try:
            base_mid = int(model.body_mocapid[model.body(f"{ns}base").id])
        except Exception:
            base_mid = None
        for s in ep.sensor_names:
            try:
                cam_id[s] = model.camera(f"{ns}{s}").id
            except Exception:
                pass
        ee_name = _ee_body(model)

    def ts_at(t):
        ns = offset_ns + int(round(t * dt * 1e9))
        return Timestamp(sec=ns // 1_000_000_000, nsec=ns % 1_000_000_000), ns

    ts0, ns0 = ts_at(0)
    bits = [f"=== {ep.label}"]
    for k, v in ep.attrs.items():
        if k in ("file",):
            continue
        bits.append(f"{k}={v}")
    desc = ep.scene.get("task_description") or ""
    if desc:
        bits.append(desc)
    if ch is not None:
        ch.log.log(Log(timestamp=ts0, level=LogLevel.Info,
                       message="  ".join(str(b) for b in bits), name="task"),
                   log_time=ns0)
        gt = fv.scene_gt(ep.scene, ts0) if ep.scene else None
        if gt:
            ch.gt.log(gt, log_time=ns0)
        tgt = fv.target_markers(ep.targets, ts0) if ep.targets else None
        if tgt:
            ch.tgt.log(tgt, log_time=ns0)

    inv_phase = {v: k for k, v in (ep.scene.get("policy_phases") or {}).items()}
    last_phase = None
    records = []
    bp = ep.base[0] if ep.base is not None else None

    for t in range(0, T, stride):
        ts, ns = ts_at(t)
        q = ep.qpos[t]

        if model is not None:
            if base_mid is not None and bp is not None:
                data.mocap_pos[base_mid] = bp[:3]
                data.mocap_quat[base_mid] = bp[3:7]
            for adr, val in zip(arm_qadr, q[:7]):
                data.qpos[adr] = float(val)
            grip = float(q[7]) if q.shape[0] > 7 else 0.0
            for adr in finger_qadr:
                data.qpos[adr] = grip
            mujoco.mj_forward(model, data)
            if ch is not None:
                tfs = []
                for bname in pub_bodies:
                    bid = model.body(bname).id
                    p, qt = data.xpos[bid], data.xquat[bid]
                    tfs.append(FrameTransform(
                        timestamp=ts, parent_frame_id="world", child_frame_id=bname,
                        translation=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                        rotation=Quaternion(x=float(qt[1]), y=float(qt[2]),
                                           z=float(qt[3]), w=float(qt[0]))))
                ch.tf.log(FrameTransforms(transforms=tfs), log_time=ns)
                if t == 0:
                    ch.mesh.log(mesh_update, log_time=ns)

            pts, dd = proximity_world_points(ep, t, model, data, cam_id, near, d_max)
            if ch is not None:
                ch.pc.log(fv.pack_cloud(pts, dd, ts, near, far), log_time=ns)

            tcp_pose = None
            if ep.tcp is not None:
                tcp_pose = ep.tcp[t]
            elif ee_name is not None:
                bid = model.body(ee_name).id
                p, qt = data.xpos[bid], data.xquat[bid]
                tcp_pose = np.array([p[0], p[1], p[2], qt[0], qt[1], qt[2], qt[3]], np.float64)
            if ch is not None and tcp_pose is not None:
                ch.tcp.log(PoseInFrame(timestamp=ts, frame_id="world", pose=Pose(
                    position=Vector3(x=float(tcp_pose[0]), y=float(tcp_pose[1]),
                                     z=float(tcp_pose[2])),
                    orientation=Quaternion(x=float(tcp_pose[4]), y=float(tcp_pose[5]),
                                           z=float(tcp_pose[6]), w=float(tcp_pose[3])))),
                           log_time=ns)
            view3d = render_cam3d(ep, t, pts, dd, near, far,
                                  VIEW3D_W, VIEW3D_H) if cam3d else None
            if view3d is None:
                view3d = render_view3d(model, data, pts, dd, near, far,
                                       VIEW3D_W, VIEW3D_H)
        else:
            pts, dd = proximity_world_points(ep, t, None, None, {}, near, d_max)
            view3d = render_cam3d(ep, t, pts, dd, near, far,
                                  VIEW3D_W, VIEW3D_H) if cam3d else None
            if view3d is None:
                view3d = render_view3d(None, None, pts, dd, near, far,
                                       VIEW3D_W, VIEW3D_H)

        if ch is not None:
            for name, arr in ep.images.items():
                fr = _index(arr, t, T)
                if fr is None:
                    continue
                payload = _jpeg(_as_rgb(fr), jpeg_q)
                if not payload:
                    continue
                topic = _cam_topic(name) if name in CAM_ALIAS or name in ("wrist_camera", "exo_camera_1") \
                    else f"/camera/{name}"
                if name == "sensors_rgb":
                    topic = "/sensors/rgb256"
                if topic not in ch.img:
                    continue
                ch.img[topic].log(
                    CompressedImage(timestamp=ts, frame_id=topic.strip("/"),
                                    data=payload, format="jpeg"),
                    log_time=ns,
                )

            if ep.proximity is not None:
                heat_rgb = render_heatmap(ep.proximity[t], ep.sensor_names, near, far,
                                          RGB_W, HEAT_H)
                heat_rgb = heat_rgb[:, :, ::-1]  # BGR -> RGB for _jpeg
                payload = _jpeg(heat_rgb, jpeg_q)
                if payload and "/sensors/heatmap" in ch.img:
                    ch.img["/sensors/heatmap"].log(
                        CompressedImage(timestamp=ts, frame_id="sensors/heatmap",
                                        data=payload, format="jpeg"),
                        log_time=ns,
                    )

            if ep.embeddings is not None and "/sensors/embeddings" in ch.img:
                emb = ep.embeddings[t]
                lo, hi = float(np.min(emb)), float(np.max(emb))
                norm = (emb - lo) / max(hi - lo, 1e-6)
                img = (_TURBO(norm)[..., :3] * 255).astype(np.uint8)
                img = cv2.resize(img, (max(img.shape[1] * 8, 32), max(img.shape[0] * 8, 40)),
                                 interpolation=cv2.INTER_NEAREST)
                payload = _jpeg(img, jpeg_q)
                if payload:
                    ch.img["/sensors/embeddings"].log(
                        CompressedImage(timestamp=ts, frame_id="sensors/embeddings",
                                        data=payload, format="jpeg"),
                        log_time=ns,
                    )

        jmsg = joint_msg(ep, t)
        if ch is not None:
            ch.joints.log(jmsg, log_time=ns)
            if ep.phase is not None:
                ph = int(ep.phase[t])
                if ph != last_phase:
                    ch.log.log(Log(timestamp=ts, level=LogLevel.Info,
                                   message=f"phase -> {inv_phase.get(ph, ph)}", name="phase"),
                               log_time=ns)
                    last_phase = ph

        if writer is not None:
            frame = compose_frame(ep, t, None, view3d, near, far)
            writer.write(frame)

        rec = {
            "t": (offset_ns / 1e9) + t * dt,
            "q": [float(q[i]) for i in range(min(7, q.shape[0]))],
            "v": [float(ep.qvel[t, i]) for i in range(min(7, ep.qvel.shape[0]))],
            "skin_min": jmsg.get("skin_min"),
        }
        records.append(rec)

    ts_end, ns_end = ts_at(T - 1)
    ok = ep.success
    level = LogLevel.Info if ok else (LogLevel.Warning if ok is False else LogLevel.Info)
    ending = "SUCCESS" if ok else ("FAIL" if ok is False else "END")
    if ch is not None:
        ch.log.log(Log(timestamp=ts_end, level=level,
                       message=f"=== {ep.label} {ending}", name="task"),
                   log_time=ns_end)
    return int(round(T * dt * 1e9)), records


def _present_rgb(ep: Episode, t: int) -> list[tuple[str, np.ndarray]]:
    """RGB tiles that actually have pixels. Missing cams are omitted (no slate)."""
    out: list[tuple[str, np.ndarray]] = []
    wrist = None
    for k in ("wrist_camera", "wrist", "wrist_rgb"):
        if k in ep.images:
            wrist = _index(ep.images[k], t, ep.T)
            if wrist is not None:
                break
    if wrist is not None:
        out.append(("wrist", _as_rgb(wrist)))
    table = None
    for k in ("exo_camera_1", "table", "table_camera", "top"):
        if k in ep.images:
            table = _index(ep.images[k], t, ep.T)
            if table is not None:
                break
    if table is not None:
        out.append(("table", _as_rgb(table)))
    return out


def compose_frame(ep: Episode, t: int, heat_rgb: np.ndarray | None,
                   view3d_bgr: np.ndarray | None, near: float = 0.02,
                   far: float = 0.60) -> np.ndarray:
    """BGR mosaic: present RGB only | prox-3D, heatmap, qpos, qvel, HUD.

    Missing wrist/table/heatmap are not drawn as slates — leftover space goes
    to the panels that have data.
    """
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 12, np.uint8)
    rgbs = _present_rgb(ep, t)
    has_heat = ep.proximity is not None or heat_rgb is not None
    n = len(rgbs)

    if view3d_bgr is None:
        view3d_bgr = _slate(VIEW3D_W, VIEW3D_H, "no 3D / no FK")
    else:
        view3d_bgr = cv2.resize(view3d_bgr, (VIEW3D_W, VIEW3D_H),
                                interpolation=cv2.INTER_AREA)

    if n == 0 and not has_heat:
        canvas[0:VIEW3D_H, 0:CANVAS_W] = cv2.resize(
            view3d_bgr, (CANVAS_W, VIEW3D_H), interpolation=cv2.INTER_AREA)
    else:
        canvas[0:VIEW3D_H, RGB_W:RGB_W + VIEW3D_W] = view3d_bgr
        rgb_h = VIEW3D_H if not has_heat else TILE_H
        if n == 1:
            img = _letterbox(rgbs[0][1], RGB_W, rgb_h)
            canvas[0:rgb_h, 0:RGB_W] = img[:, :, ::-1]
            cv2.putText(canvas, rgbs[0][0], (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
        elif n >= 2:
            tw = RGB_W // 2
            for i, (lab, fr) in enumerate(rgbs[:2]):
                img = _letterbox(fr, tw, rgb_h)
                x = i * tw
                canvas[0:rgb_h, x:x + tw] = img[:, :, ::-1]
                cv2.putText(canvas, lab, (x + 8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if has_heat:
            hy = 0 if n == 0 else TILE_H
            hh = VIEW3D_H if n == 0 else HEAT_H
            if heat_rgb is not None:
                heat_bgr = heat_rgb[:, :, ::-1] if heat_rgb.shape[2] == 3 else heat_rgb
                heat_bgr = cv2.resize(heat_bgr, (RGB_W, hh), interpolation=cv2.INTER_NEAREST)
            else:
                heat_bgr = render_heatmap(ep.proximity[t], ep.sensor_names, near, far,
                                          RGB_W, hh)
            canvas[hy:hy + hh, 0:RGB_W] = heat_bgr

    y = VIEW3D_H
    _sparkline(canvas[y:y + PLOT_H], ep.qpos[:ep.T, : min(7, ep.qpos.shape[1])], t,
               "qpos q1..q7 (rad)")
    y += PLOT_H
    _sparkline(canvas[y:y + PLOT_H], ep.qvel[:ep.T, : min(7, ep.qvel.shape[1])], t,
               "qvel v1..v7 (rad/s)")
    y += PLOT_H
    hud = canvas[y:y + HUD_H]
    hud[:] = 28
    skin = ""
    if ep.proximity is not None:
        d = ep.proximity[t]
        skin = f"  skin_min={float(np.min(d)):.3f}m"
    attr = ""
    for k in ("behavior_class", "intrusion_side", "has_bar", "clean_success"):
        if k in ep.attrs:
            attr += f"  {k}={ep.attrs[k]}"
    text = f"{ep.label}  t={t}/{ep.T - 1}{skin}{attr}"
    cv2.putText(hud, text[:140], (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def out_dir_for(ds_path: Path) -> Path:
    """Map a dataset path to its output folder under the fixed _OUT_BASE root.

    The output mirrors the dataset path, with the <repo>/data prefix removed:

        <repo>/data/molmo-pi0-eval-videos/data/fumehood/pick
            -> <_OUT_BASE>/molmo-pi0-eval-videos/data/fumehood/pick

    A dataset elsewhere in the checkout keeps its repo-relative path
    (act_style_data/...). A dataset outside the checkout keeps its absolute path
    without the leading "/". A single .h5 file maps to <parent>/<stem>.
    """
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


def _out_paths(src: Path) -> tuple[Path, Path]:
    out_dir = out_dir_for(src)
    return out_dir, out_dir / "dataset.mcap"


_GROUP_KEYS = ("behavior_class", "intrusion_side", "has_bar", "clean_success")


def _safe_name(s) -> str:
    """Filesystem-safe fragment for a clip file name or a group folder."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "unnamed"


def episode_group(ep: Episode, key: str | None = None) -> str:
    """Trajectory type of one episode — the folder its clip goes in.

    Without --group-by, the first attribute of _GROUP_KEYS that the episode
    carries wins: datagen writes behavior_class, obstacle ACT writes the rest.
    An episode with none of them falls back to its success flag, then to "all".
    """
    if key:
        val = ep.attrs.get(key)
    else:
        val = next((ep.attrs[k] for k in _GROUP_KEYS
                    if ep.attrs.get(k) not in (None, "")), None)
        if val is None and ep.success is not None:
            val = "success" if ep.success else "fail"
    return _safe_name(val) if val not in (None, "") else "all"


def _open_writer(path: Path, dt: float, stride: int = 1):
    """Open an mp4 writer that plays back at wall-clock speed.

    --stride N keeps every Nth step, so the frame rate has to drop by N too.
    Without that division a strided clip runs N times too fast.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1.0, min(60.0, (1.0 / max(dt, 1e-3)) / max(int(stride), 1)))
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                        (CANVAS_W, CANVAS_H))
    if not w.isOpened():
        raise SystemExit(f"cv2 could not open {path} for write")
    return w


def _has_output(dest: Path) -> bool:
    """True when dest holds a finished run: an audit plus at least one video."""
    if not (dest / "audit.json").is_file():
        return False
    if (dest / "dataset.mp4").is_file():
        return True
    if any((dest / "episodes").rglob("*.mp4")):
        return True
    audit = _read_audit(dest)
    return bool(audit and int(audit.get("n_videos") or 0) == 0
                and (dest / "timeline.json").is_file())


def wanted_n(n_eps: int, args) -> int:
    n = max(0, int(n_eps) - int(getattr(args, "start_episode", 0) or 0))
    cap = getattr(args, "max_episodes", None)
    if cap is not None:
        n = min(n, int(cap))
    return n


def _read_audit(dest: Path) -> dict | None:
    p = dest / "audit.json"
    if not p.is_file():
        return None
    try:
        rec = json.loads(p.read_text())
    except Exception:
        return None
    return rec if isinstance(rec, dict) else None


def dest_exported(dest: Path) -> tuple[int, dict | None]:
    if not _has_output(dest):
        return 0, None
    audit = _read_audit(dest)
    if not audit:
        return 0, None
    return int(audit.get("n_eps_exported") or 0), audit


def viz_action(dest: Path, n_eps: int, args) -> str:
    """skip = already on dashboard; grow = more episodes than last audit; run = encode."""
    if getattr(args, "force", False):
        return "run"
    exported, audit = dest_exported(dest)
    want = wanted_n(n_eps, args)
    if exported <= 0 or audit is None:
        return "run"
    if exported >= want:
        return "skip"
    if getattr(args, "one_video", False):
        return "run"
    return "grow"


def _usable_done(out_dir: Path, old_tl: dict) -> dict[str, dict]:
    """Episode metas we can keep (label + clip still on disk)."""
    out: dict[str, dict] = {}
    for e in old_tl.get("episodes") or []:
        lab = e.get("label")
        if not lab:
            continue
        vid = e.get("video")
        if vid and not (out_dir / vid).is_file():
            continue
        out[str(lab)] = e
    return out


def _cat_series(old, new):
    if isinstance(old, dict) or isinstance(new, dict):
        old = old or {}
        new = new or {}
        keys = list(dict.fromkeys([*old, *new]))
        return {k: list(old.get(k) or []) + list(new.get(k) or []) for k in keys}
    return list(old or []) + list(new or [])


def run_one(src: Path, out_dir: Path, mcap_path: Path, args) -> None:
    kind, refs = discover(src)
    refs = refs[args.start_episode:]
    if args.max_episodes is not None:
        refs = refs[: args.max_episodes]
    if not refs:
        raise SystemExit(f"no episodes after filters under {src}")

    print(f"format={kind}  n={len(refs)}  root={src}")
    preview = refs if len(refs) <= 8 else list(refs[:4]) + list(refs[-2:])
    for r in preview:
        try:
            print(" ", peek_summary(r))
        except Exception as e:
            print(f"  {r.label} peek fail: {e}")
    if len(refs) > 8:
        print(f"  ... {len(refs) - 6} more")
    if args.list and not args.each:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(prox_pool=args.prox_pool, dt=args.dt,
              include_sensor_rgb=args.include_sensor_rgb,
              include_depth=args.include_depth)

    old_tl = None
    done_by_label: dict[str, dict] = {}
    if not args.force and not args.one_video:
        tl_path = out_dir / "timeline.json"
        if tl_path.is_file():
            try:
                old_tl = json.loads(tl_path.read_text())
            except Exception:
                old_tl = None
        if isinstance(old_tl, dict):
            done_by_label = _usable_done(out_dir, old_tl)

    first = None
    first_ref = None
    for ref in refs:
        if ref.label in done_by_label:
            continue
        try:
            first = load_episode(ref, **kw)
            first_ref = ref
            break
        except Exception as e:
            print(f"SKIP {ref.label}: {e}")
    if first is None:
        if done_by_label:
            print(f"no new episodes to encode under {src}  (kept {len(done_by_label)})")
            return
        raise SystemExit(f"every episode failed to load under {src}")
    image_topics = ["/camera/wrist", "/camera/table", "/sensors/heatmap"]
    if args.include_sensor_rgb:
        image_topics.append("/sensors/rgb256")
    if args.include_depth:
        image_topics += ["/camera/wrist_camera_depth", "/camera/exo_camera_1_depth"]
    if first.embeddings is not None:
        image_topics.append("/sensors/embeddings")
    cam_names = set(first.images)
    for ref in refs[1:min(8, len(refs))]:
        if ref.kind == "act":
            with h5py.File(ref.path, "r") as f:
                if "observations/images" in f:
                    cam_names.update(f["observations/images"].keys())
        else:
            for stem in RGB_STEMS:
                if glob_mp4(ref.h5_dir, ref.vid_id, stem):
                    cam_names.add(stem)
    for name in cam_names:
        topic = _cam_topic(name) if name in CAM_ALIAS else f"/camera/{name}"
        if name == "sensors_rgb":
            topic = "/sensors/rgb256"
        if topic not in image_topics:
            image_topics.append(topic)

    model = data = mesh_update = pub_bodies = None
    try:
        model, data, mesh_update, pub_bodies = _init_robot(first.sensor_names, args.mount_z)
        if model is None:
            print("robot xml missing — MCAP will skip /tf /robot /proximity cloud")
    except Exception as e:
        print(f"robot FK disabled ({e})")
        model = data = mesh_update = None
        pub_bodies = []

    one_video = bool(args.one_video)
    writer = None
    video_path = out_dir / "dataset.mp4"
    if not args.no_video and one_video:
        writer = _open_writer(video_path, first.dt, args.stride)

    ctx = ch = mcap_writer = None
    append_mode = bool(done_by_label) and not args.force
    if not args.no_mcap:
        if append_mode:
            print("append: skip mcap rewrite (--force for a full Foxglove file)")
        else:
            ctx = foxglove.Context()
            mcap_writer = foxglove.open_mcap(str(mcap_path), allow_overwrite=True, context=ctx)
            ch = Channels(ctx, image_topics)

    has_table = any(
        k in cam_names for k in ("exo_camera_1", "table", "table_camera", "top")
    )
    has_wrist = "wrist_camera" in cam_names or "wrist" in cam_names
    layout = build_layout(
        has_table=has_table, has_wrist=has_wrist,
        has_heatmap=first.proximity is not None, has_3d=model is not None,
    )
    (out_dir / "foxglove_layout.json").write_text(json.dumps(layout, indent=2) + "\n")

    offset_ns = 0
    all_t, all_q, all_v, all_skin = [], [], [], []
    episodes_meta = list(done_by_label.values())
    if episodes_meta:
        last = episodes_meta[-1]
        offset_ns = int(
            (float(last.get("start_s") or 0) + float(last.get("dur_s") or 0) + EP_GAP_S) * 1e9
        )
        print(f"keep {len(episodes_meta)} already-exported episode(s)")
    n_keep = len(episodes_meta)
    n_ok = n_keep
    n_total = len(refs)
    n_skip = 0
    for i, ref in enumerate(refs):
        if ref.label in done_by_label:
            continue
        try:
            ep = first if ref is first_ref else load_episode(ref, **kw)
        except Exception as e:
            print(f"  SKIP {ref.label}: {e}")
            n_skip += 1
            continue
        start_s = offset_ns / 1e9
        group = episode_group(ep, args.group_by)
        print(f"[{i+1}/{len(refs)}] {ep.label} [{group}] T={ep.T} dt={ep.dt:.3f}s "
              f"cams={list(ep.images)} prox={None if ep.proximity is None else ep.proximity.shape}")
        ep_video = None
        ep_writer = writer
        if not args.no_video and not one_video:
            ep_video = out_dir / "episodes" / group / f"{i:04d}_{_safe_name(ep.label)}.mp4"
            ep_writer = _open_writer(ep_video, ep.dt, args.stride)
        try:
            dur_ns, recs = export_episode(
                ep, ch, model=model, data=data, mesh_update=mesh_update,
                pub_bodies=pub_bodies or [], offset_ns=offset_ns, near=args.near,
                far=args.far, d_max=args.d_max, stride=args.stride,
                jpeg_q=args.jpeg_quality, writer=ep_writer, cam3d=args.cam3d,
            )
        except Exception as e:
            if ep_video is not None:
                ep_writer.release()
                ep_video.unlink(missing_ok=True)
            print(f"  SKIP export {ep.label}: {e}")
            n_skip += 1
            continue
        if ep_video is not None:
            ep_writer.release()
            encode_h264_ide(ep_video)
        for rec in recs:
            all_t.append(rec["t"])
            all_q.append(rec["q"])
            all_v.append(rec["v"])
            all_skin.append(rec["skin_min"])
        episodes_meta.append({
            "label": ep.label,
            "T": ep.T,
            "start_s": start_s,
            "dur_s": dur_ns / 1e9,
            "group": group,
            "video": str(ep_video.relative_to(out_dir)) if ep_video is not None else None,
            "attrs": {k: ep.attrs[k] for k in ep.attrs
                      if k not in ("file",) and not str(k).startswith("pact_")},
        })
        offset_ns += dur_ns + int(EP_GAP_S * 1e9)
        n_ok += 1

    if writer is not None:
        writer.release()
        encode_h264_ide(video_path)
    if mcap_writer is not None:
        mcap_writer.close()

    qpos = {f"q{j+1}": [row[j] if j < len(row) else None for row in all_q]
            for j in range(7)}
    qvel = {f"v{j+1}": [row[j] if j < len(row) else None for row in all_v]
            for j in range(7)}
    t_ds, packed = downsample_series(all_t, {"qpos": qpos, "qvel": qvel, "skin_min": all_skin})
    n_frames = len(all_t)
    if append_mode and old_tl:
        t_ds = _cat_series(old_tl.get("t"), t_ds)
        packed = {
            "qpos": _cat_series(old_tl.get("qpos"), packed["qpos"]),
            "qvel": _cat_series(old_tl.get("qvel"), packed["qvel"]),
            "skin_min": _cat_series(old_tl.get("skin_min"), packed["skin_min"]),
        }
        n_frames = int(old_tl.get("n_frames") or 0) + len(all_t)
    ep_videos = [e["video"] for e in episodes_meta if e.get("video")]
    ep_groups = sorted({e["group"] for e in episodes_meta if e.get("group")})
    timeline = {
        "title": src.name,
        "per_episode": not one_video,
        "video": "dataset.mp4" if one_video and not args.no_video else None,
        "n_episodes": n_ok,
        "n_frames": n_frames,
        "duration_s": (offset_ns / 1e9) if n_ok else 0.0,
        "dt": first.dt,
        "t": t_ds,
        "qpos": packed["qpos"],
        "qvel": packed["qvel"],
        "skin_min": packed["skin_min"],
        "episodes": episodes_meta,
        "mcap": str(mcap_path.name) if not args.no_mcap else None,
    }
    write_html(out_dir, timeline)

    gaps = episode_gaps(first, has_fk=model is not None)
    try:
        slug = str(out_dir.relative_to(_OUT_BASE))
    except ValueError:
        slug = out_dir.name
    audit = {
        "slug": slug,
        "src": str(src),
        "kind": kind,
        "n_eps_total": n_total,
        "n_eps_exported": n_ok,
        "n_eps_skipped": n_skip,
        "cams": sorted(cam_names),
        "has_prox": first.proximity is not None,
        "prox_shape": None if first.proximity is None else list(first.proximity.shape),
        "has_table": has_table,
        "has_wrist": has_wrist,
        "has_fk": model is not None,
        "has_cam2w": bool(first.cam2w),
        "gaps": gaps,
        "video": ("dataset.mp4" if one_video and not args.no_video
                  else (ep_videos[0] if ep_videos else None)),
        "n_videos": (1 if one_video else len(ep_videos)),
        "groups": ep_groups,
        "duration_s": (offset_ns / 1e9) if n_ok else 0.0,
        "stride": args.stride,
        "cam3d": bool(args.cam3d),
        "has_cam_params": sorted(first.cam_params),
        "no_mcap": bool(args.no_mcap),
    }
    skin_vals = [float(x) for x in (packed.get("skin_min") or [])
                 if x is not None and np.isfinite(x)]
    if skin_vals:
        audit["skin_min_min"] = min(skin_vals)
        audit["skin_min_mean"] = float(sum(skin_vals) / len(skin_vals))
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    print(f"\n{n_ok} episode(s) ({n_keep} keep, {n_ok - n_keep} new), "
          f"{n_skip} skipped, {offset_ns/1e9:.1f}s timeline")
    print(f"  html    {out_dir / 'index.html'}")
    if not args.no_video and one_video:
        print(f"  video   {video_path}")
    elif not args.no_video:
        print(f"  videos  {len(ep_videos)} clip(s) in {out_dir / 'episodes'}"
              f"  groups={ep_groups}")
    if not args.no_mcap:
        print(f"  mcap    {mcap_path}")
        print(f"  layout  {out_dir / 'foxglove_layout.json'}")
        print("Open the mcap in Foxglove (app.foxglove.dev or desktop) and import the layout.")
        print("Or open index.html in a browser for the tiled dataset video + plots.")


def _run_each(catalog: list[DatasetRoot], args, n_skipped_dups: int = 0) -> None:
    n_skip = n_run = n_grow = 0
    mismatches: list[str] = []
    for ds in catalog:
        dest = out_dir_for(ds.path)
        try:
            slug = str(dest.relative_to(_OUT_BASE))
        except ValueError:
            slug = dest.name
        action = viz_action(dest, ds.n_eps, args)
        if action == "skip":
            n_skip += 1
            audit = _read_audit(dest)
            if audit is not None:
                same_cam = bool(audit.get("cam3d")) == bool(args.cam3d)
                same_stride = int(audit.get("stride") or 1) == int(args.stride)
                if not (same_cam and same_stride):
                    mismatches.append(slug)
            continue
        if action == "grow":
            n_grow += 1
            print(f"\n======== {slug} grow ({ds.kind}, {ds.n_eps} eps) ========")
        else:
            n_run += 1
            print(f"\n======== {slug} ({ds.kind}, {ds.n_eps} eps) ========")
        run_one(ds.path, dest, dest / "dataset.mcap", args)
        write_audit_index(_OUT_BASE, n_skipped_dups=n_skipped_dups, quiet=True)
    print(f"\nincremental  skip={n_skip}  new={n_run}  grow={n_grow}")
    if mismatches:
        print(f"  {len(mismatches)} skipped use other --cam3d/--stride; --force to redo")
    write_audit_index(_OUT_BASE, n_skipped_dups=n_skipped_dups)


def _maybe_run_one(src: Path, n_eps: int, args) -> None:
    dest, mcap_path = _out_paths(src)
    if getattr(args, "list", False):
        run_one(src, dest, mcap_path, args)
        return
    action = viz_action(dest, n_eps, args)
    try:
        slug = str(dest.relative_to(_OUT_BASE))
    except ValueError:
        slug = dest.name
    if action == "skip":
        print(f"{slug} already on dashboard ({n_eps} eps). --force to redo")
        write_audit_index(_OUT_BASE)
        return
    if action == "grow":
        print(f"{slug} grow — encode new episodes only")
    run_one(src, dest, mcap_path, args)
    write_audit_index(_OUT_BASE)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", default=None, type=Path,
                    help="folder of h5 / hdf5, a parent of many datasets, or one file")
    ap.add_argument("--reencode", type=Path, default=None,
                    help="H.264-encode every dataset.mp4 under this dir (Cursor / VS Code)")
    ap.add_argument("--list", action="store_true",
                    help="print catalog (mixed tree) or episode list (one dataset)")
    ap.add_argument("--each", action="store_true",
                    help="one visualizer per dataset under --data "
                         "(skip finished; encode new datasets / new episodes)")
    ap.add_argument("--include-eval", action="store_true",
                    help="also catalog results/ eval rollouts")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--start-episode", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--dt", type=float, default=None,
                    help="seconds per step (default: obs_scene.policy_dt_ms or 0.066)")
    ap.add_argument("--prox-pool", choices=("mean", "min"), default="mean")
    ap.add_argument("--mount-z", type=float, default=0.35)
    ap.add_argument("--near", type=float, default=0.02)
    ap.add_argument("--far", type=float, default=0.60)
    ap.add_argument("--d-max", type=float, default=1.5)
    ap.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    ap.add_argument("--no-mcap", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--cam3d", action="store_true",
                    help="right panel = the table camera frame with the skin "
                         "returns drawn on it, instead of the synthetic 3D view")
    ap.add_argument("--one-video", action="store_true",
                    help="old behaviour: one concatenated dataset.mp4 instead of "
                         "one clip per episode under episodes/<type>/")
    ap.add_argument("--group-by", default=None, metavar="ATTR",
                    help="episode attribute naming the episodes/<type>/ folder "
                         "(default: first present of "
                         "behavior_class, intrusion_side, has_bar, clean_success)")
    ap.add_argument("--keep-dups", action="store_true",
                    help="include nested copies (pact_20260622 copies of act_style_52)")
    ap.add_argument("--force", action="store_true",
                    help="redo datasets already on the dashboard (without this, "
                         "only new folders and new episodes encode)")
    ap.add_argument("--include-sensor-rgb", action="store_true",
                    help="also ingest sensors_rgb256 sidecar (huge)")
    ap.add_argument("--include-depth", action="store_true")
    ap.add_argument("--dashboard", action="store_true",
                    help="rebuild root index.html + audit.json only (no encode)")
    args = ap.parse_args()

    if args.dashboard:
        write_audit_index(_OUT_BASE)
        return

    if args.reencode is not None:
        root = args.reencode.expanduser().resolve()
        files = sorted(root.rglob("*.mp4")) if root.is_dir() else ([root] if root.is_file() else [])
        if not files:
            raise SystemExit(f"no .mp4 under {root}")
        for p in files:
            print(f"encode {p}")
            encode_h264_ide(p)
        return

    if args.data is None:
        raise SystemExit("pass --data PATH  or  --dashboard  or  --reencode DIR")

    src = args.data.expanduser().resolve()
    catalog = [] if src.is_file() else scan_dataset_roots(src, include_eval=args.include_eval)

    if len(catalog) > 1:
        print_catalog(catalog, src)
        if args.list and not args.each:
            print("\npass one child path, or --each (one viz per row). "
                  "--include-eval adds results/ rollouts. DUP rows skipped by --each "
                  "unless --keep-dups.")
            return
        if not args.each:
            raise SystemExit(
                "mixed tree — pick one child from the list, or pass --each"
            )
        catalog, skipped_dups = unique_catalog(catalog, args.keep_dups)
        if skipped_dups:
            print(f"skip {len(skipped_dups)} DUP cop(y/ies); --keep-dups to include")
        _run_each(catalog, args, n_skipped_dups=len(skipped_dups))
        return

    if len(catalog) == 1 and src != catalog[0].path and src.is_dir():
        src = catalog[0].path

    if args.each and catalog:
        _run_each(catalog, args)
        return

    n_eps = catalog[0].n_eps if catalog else None
    if n_eps is None:
        _, refs = discover(src)
        n_eps = len(refs)
    _maybe_run_one(src, n_eps, args)


if __name__ == "__main__":
    main()
