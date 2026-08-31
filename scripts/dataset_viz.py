"""Concatenate a folder of trajectory HDF5s into one Foxglove MCAP + one tiled video.

Auto-detects three layouts this repo actually uses:

  ACT          episode_*.hdf5  — RGB inside the file, optional (T, 40, 8, 8) proximity
  HF rows      rows/*/trajectory.h5 + episode_00000000_*.mp4  (wrist-only clones)
  datagen      house_*/trajectories*.h5 + episode_*_*_batch_*.mp4

Every episode is laid end-to-end on one timeline (0.5 s gap). The MCAP is the
interactive visualizer (Foxglove). The MP4 + index.html is the "whole dataset
in one video" path with wrist / table RGB, a live 8x8 skin mosaic, and joint
position / velocity plots.

  conda activate mlspaces
  cd /home/jaydv/code/prox_learning
  python scripts/dataset_viz.py --data /home/jaydv/code/prox_learning/data --list
  python scripts/dataset_viz.py --data /home/jaydv/code/prox_learning/data --each --max-episodes 2
  python scripts/dataset_viz.py --data data/pact_place_corridor_v5 --max-episodes 2
  python scripts/dataset_viz.py --data data/molmo-pi0-eval-videos/data/fumehood/pick --list
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
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
TILE_W, TILE_H = 320, 240
HEAT_H = 240
PLOT_H = 110
HUD_H = 40
CANVAS_W = TILE_W * 2
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

RGB_STEMS = ("wrist_camera", "exo_camera_1")
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
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


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
    """d8s: (N, 8, 8) metres -> BGR uint8 mosaic, one row per link."""
    canvas = np.full((out_h, out_w, 3), 18, np.uint8)
    if d8s is None or len(d8s) == 0:
        cv2.putText(canvas, "no proximity in this file", (16, out_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1, cv2.LINE_AA)
        return canvas
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for i, n in enumerate(names):
        groups.setdefault(_link_prefix(n), []).append(i)
    n_rows = max(len(groups), 1)
    n_cols = max((len(v) for v in groups.values()), default=1)
    pad = 2
    label_h = 12
    cell = min((out_w - pad * (n_cols + 1)) // n_cols,
                (out_h - pad * (n_rows + 1) - label_h * n_rows) // n_rows)
    cell = max(cell, 8)
    for r, (link, idxs) in enumerate(groups.items()):
        for c, i in enumerate(idxs):
            patch = np.asarray(d8s[i], np.float32)
            valid = np.isfinite(patch) & (patch >= DEAD_PIXEL_M)
            norm = np.clip((patch - near) / max(far - near, 1e-6), 0.0, 1.0)
            col = (_TURBO(1.0 - norm)[..., :3] * 255).astype(np.uint8)
            col[~valid] = 40
            tile = cv2.resize(col, (cell, cell), interpolation=cv2.INTER_NEAREST)
            tile = tile[:, :, ::-1]  # RGB -> BGR
            x = pad + c * (cell + pad)
            y = pad + r * (cell + pad + label_h) + label_h
            y2, x2 = min(y + cell, out_h), min(x + cell, out_w)
            canvas[y:y2, x:x2] = tile[: y2 - y, : x2 - x]
            cv2.putText(canvas, names[i].replace("sensor_", "s"),
                        (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, link, (out_w - 90, pad + r * (cell + pad + label_h) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 200, 255), 1, cv2.LINE_AA)
    return canvas


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


def print_catalog(roots: list[DatasetRoot], scan_root: Path) -> None:
    print(f"datasets under {scan_root}  n={len(roots)}")
    for i, ds in enumerate(roots):
        try:
            rel = ds.path.relative_to(scan_root)
        except ValueError:
            rel = ds.path
        print(f"  [{i:02d}] {ds.kind:8s}  eps={ds.n_eps:4d}  {rel}")


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
                "datagen", p, "traj_0", 0,
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
    exact = sorted(h5_dir.glob(f"episode_{vid_id:08d}_{stem}.mp4"))
    batch = sorted(h5_dir.glob(f"episode_{vid_id:08d}_{stem}_batch_*.mp4"))
    cands = exact + batch
    return cands[0] if cands else None


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
        q_rows, dq_rows, grip = [], [], []
        for i in range(T):
            qp = _json_row(t["obs/agent/qpos"][i])
            arm = list(qp.get("arm") or [0.0] * 7)
            g = list(qp.get("gripper") or [0.0])
            q_rows.append(arm[:7] + (g[:2] if len(g) >= 2 else g + [0.0]))
            qv = _json_row(t["obs/agent/qvel"][i]) if "obs/agent/qvel" in t else {}
            arm_v = list(qv.get("arm") or [0.0] * 7)
            gv = list(qv.get("gripper") or [0.0, 0.0])
            dq_rows.append(arm_v[:7] + (gv[:2] if len(gv) >= 2 else gv + [0.0]))
            grip.append(float((g or [0.0])[0]))
        qpos = np.asarray(q_rows, np.float32)
        qvel = np.asarray(dq_rows, np.float32)
        action = None
        if "actions/joint_pos" in t:
            a_rows = []
            for i in range(T):
                d = _json_row(t["actions/joint_pos"][i])
                arm = list(d.get("arm") or [0.0] * 7)
                g = list(d.get("gripper") or [0.0])
                a_rows.append(arm[:7] + [float(g[0]) if g else 0.0])
            action = np.asarray(a_rows, np.float32)
        prox = None
        names: list[str] = []
        cam2w = {}
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
        targets=targets, attrs=attrs, extras=extras,
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
        mp4s = [p.name for p in ref.h5_dir.glob(f"episode_{ref.vid_id:08d}_*.mp4")]
        return (f"{ref.label:28s} T={T:4d}  prox_sensors={nprox:2d}  "
                f"mp4s={len(mp4s)}  file={ref.path.name}")


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
  #ep { max-height: 280px; overflow: auto; background: #1a1a1a; border-radius: 4px; }
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
    Tiled video = whole dataset. Plots follow playhead.
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
<script>
fetch("timeline.json").then(r => r.json()).then(init);
function init(tl) {
  document.getElementById("title").textContent = tl.title + "  (" + tl.n_episodes + " eps, " +
    (tl.duration_s).toFixed(1) + "s)";
  const v = document.getElementById("v");
  v.src = tl.video;
  const box = document.getElementById("ep");
  tl.episodes.forEach((e, i) => {
    const b = document.createElement("button");
    b.textContent = e.label + "  T=" + e.T + (e.attrs && e.attrs.behavior_class ? "  " + e.attrs.behavior_class : "");
    b.onclick = () => { v.currentTime = e.start_s; v.play(); };
    box.appendChild(b);
    e._btn = b;
  });
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
  v.ontimeupdate = () => {
    const t = v.currentTime;
    cursor(t);
    let cur = tl.episodes[0];
    for (const e of tl.episodes) if (t >= e.start_s) cur = e;
    tl.episodes.forEach(e => e._btn.classList.toggle("active", e === cur));
    document.getElementById("hud").textContent =
      (cur ? cur.label : "") + "   t=" + t.toFixed(2) + "s / " + tl.duration_s.toFixed(1) + "s";
  };
}
</script>
</body>
</html>
"""


def write_html(out_dir: Path, timeline: dict) -> None:
    (out_dir / "index.html").write_text(_HTML)
    (out_dir / "timeline.json").write_text(json.dumps(timeline))


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


def export_episode(ep: Episode, ch: Channels, *, model, data, mesh_update,
                   pub_bodies, offset_ns: int, near: float, far: float,
                   d_max: float, stride: int, jpeg_q: int,
                   writer) -> tuple[int, list]:
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

            all_pts, all_d = [], []
            if ep.proximity is not None:
                for i, name in enumerate(ep.sensor_names):
                    d8 = ep.proximity[t, i]
                    if name in ep.cam2w:
                        c2w = ep.cam2w[name][min(t, len(ep.cam2w[name]) - 1)]
                    elif name in cam_id:
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
                pts = np.concatenate(all_pts) if all_pts else np.zeros((0, 3))
                dd = np.concatenate(all_d) if all_d else np.zeros((0,))
                ch.pc.log(fv.pack_cloud(pts, dd, ts, near, far), log_time=ns)

            tcp_pose = None
            if ep.tcp is not None:
                tcp_pose = ep.tcp[t]
            elif ee_name is not None:
                bid = model.body(ee_name).id
                p, qt = data.xpos[bid], data.xquat[bid]
                tcp_pose = np.array([p[0], p[1], p[2], qt[0], qt[1], qt[2], qt[3]], np.float64)
            if tcp_pose is not None:
                ch.tcp.log(PoseInFrame(timestamp=ts, frame_id="world", pose=Pose(
                    position=Vector3(x=float(tcp_pose[0]), y=float(tcp_pose[1]),
                                     z=float(tcp_pose[2])),
                    orientation=Quaternion(x=float(tcp_pose[4]), y=float(tcp_pose[5]),
                                           z=float(tcp_pose[6]), w=float(tcp_pose[3])))),
                           log_time=ns)

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

        heat = None
        if ep.proximity is not None:
            heat_rgb = render_heatmap(ep.proximity[t], ep.sensor_names, near, far,
                                      CANVAS_W, HEAT_H)
            heat_rgb = heat_rgb[:, :, ::-1]  # BGR -> RGB for _jpeg
            payload = _jpeg(heat_rgb, jpeg_q)
            if payload and "/sensors/heatmap" in ch.img:
                ch.img["/sensors/heatmap"].log(
                    CompressedImage(timestamp=ts, frame_id="sensors/heatmap",
                                    data=payload, format="jpeg"),
                    log_time=ns,
                )
            heat = heat_rgb

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
        ch.joints.log(jmsg, log_time=ns)

        if ep.phase is not None:
            ph = int(ep.phase[t])
            if ph != last_phase:
                ch.log.log(Log(timestamp=ts, level=LogLevel.Info,
                               message=f"phase -> {inv_phase.get(ph, ph)}", name="phase"),
                           log_time=ns)
                last_phase = ph

        if writer is not None:
            frame = compose_frame(ep, t, heat)
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
    ch.log.log(Log(timestamp=ts_end, level=level,
                   message=f"=== {ep.label} {ending}", name="task"),
               log_time=ns_end)
    return int(round(T * dt * 1e9)), records


def compose_frame(ep: Episode, t: int, heat_rgb: np.ndarray | None) -> np.ndarray:
    """BGR tiled frame: wrist | table / heatmap / qpos / qvel / HUD."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 12, np.uint8)
    wrist = _index(ep.images.get("wrist_camera"), t, ep.T)
    table = None
    for k in ("exo_camera_1", "table", "table_camera", "top"):
        if k in ep.images:
            table = _index(ep.images[k], t, ep.T)
            break
    wimg = _letterbox(_as_rgb(wrist), TILE_W, TILE_H) if wrist is not None \
        else _slate(TILE_W, TILE_H, "no wrist RGB")
    timg = _letterbox(_as_rgb(table), TILE_W, TILE_H) if table is not None \
        else _slate(TILE_W, TILE_H, "no table RGB")
    canvas[0:TILE_H, 0:TILE_W] = wimg[:, :, ::-1]
    canvas[0:TILE_H, TILE_W:CANVAS_W] = timg[:, :, ::-1]
    y = TILE_H
    if heat_rgb is None and ep.proximity is not None:
        heat_bgr = render_heatmap(ep.proximity[t], ep.sensor_names, 0.02, 0.60,
                                 CANVAS_W, HEAT_H)
    elif heat_rgb is not None:
        heat_bgr = heat_rgb[:, :, ::-1] if heat_rgb.shape[2] == 3 else heat_rgb
        heat_bgr = cv2.resize(heat_bgr, (CANVAS_W, HEAT_H), interpolation=cv2.INTER_NEAREST)
    else:
        heat_bgr = _slate(CANVAS_W, HEAT_H, "no proximity")
    canvas[y:y + HEAT_H] = heat_bgr
    y += HEAT_H
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
    cv2.putText(hud, text[:110], (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "wrist", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "table", (TILE_W + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _out_paths(src: Path, args) -> tuple[Path, Path]:
    if args.out is None:
        out_dir = (src.parent if src.is_file() else src) / "dataset_viz"
        mcap_path = out_dir / "dataset.mcap"
    elif str(args.out).endswith(".mcap"):
        mcap_path = args.out.expanduser().resolve()
        out_dir = mcap_path.parent
    else:
        out_dir = args.out.expanduser().resolve()
        mcap_path = out_dir / "dataset.mcap"
    return out_dir, mcap_path


def _slug(scan_root: Path, ds_path: Path) -> str:
    try:
        rel = ds_path.relative_to(scan_root)
        s = str(rel) if str(rel) != "." else ds_path.name
    except ValueError:
        s = ds_path.name
    return s.replace("/", "_").replace(" ", "_")


def run_one(src: Path, out_dir: Path, mcap_path: Path, args) -> None:
    kind, refs = discover(src)
    refs = refs[args.start_episode:]
    if args.max_episodes is not None:
        refs = refs[: args.max_episodes]
    if not refs:
        raise SystemExit(f"no episodes after filters under {src}")

    print(f"format={kind}  n={len(refs)}  root={src}")
    for r in refs:
        print(" ", peek_summary(r))
    if args.list and not args.each:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(prox_pool=args.prox_pool, dt=args.dt,
              include_sensor_rgb=args.include_sensor_rgb,
              include_depth=args.include_depth)

    first = load_episode(refs[0], **kw)
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

    writer = None
    video_path = out_dir / "dataset.mp4"
    if not args.no_video:
        fps = max(1, min(60, int(round(1.0 / max(first.dt, 1e-3)))))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (CANVAS_W, CANVAS_H))
        if not writer.isOpened():
            raise SystemExit(f"cv2 could not open {video_path} for write")

    ctx = ch = mcap_writer = None
    if not args.no_mcap:
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
    episodes_meta = []
    n_ok = 0
    for i, ref in enumerate(refs):
        ep = first if i == 0 else load_episode(ref, **kw)
        start_s = offset_ns / 1e9
        print(f"[{i+1}/{len(refs)}] {ep.label} T={ep.T} dt={ep.dt:.3f}s "
              f"cams={list(ep.images)} prox={None if ep.proximity is None else ep.proximity.shape}")
        if args.no_mcap:
            recs = []
            for t in range(0, ep.T, args.stride):
                if writer is not None:
                    writer.write(compose_frame(ep, t, None))
                q = ep.qpos[t]
                recs.append({
                    "t": start_s + t * ep.dt,
                    "q": [float(q[j]) for j in range(min(7, q.shape[0]))],
                    "v": [float(ep.qvel[t, j]) for j in range(min(7, ep.qvel.shape[0]))],
                    "skin_min": (float(np.min(ep.proximity[t]))
                                 if ep.proximity is not None else None),
                })
            dur_ns = int(round(ep.T * ep.dt * 1e9))
        else:
            dur_ns, recs = export_episode(
                ep, ch, model=model, data=data, mesh_update=mesh_update,
                pub_bodies=pub_bodies, offset_ns=offset_ns, near=args.near,
                far=args.far, d_max=args.d_max, stride=args.stride,
                jpeg_q=args.jpeg_quality, writer=writer,
            )
        for rec in recs:
            all_t.append(rec["t"])
            all_q.append(rec["q"])
            all_v.append(rec["v"])
            all_skin.append(rec["skin_min"])
        episodes_meta.append({
            "label": ep.label,
            "T": ep.T,
            "start_s": start_s,
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
    timeline = {
        "title": src.name,
        "video": "dataset.mp4",
        "n_episodes": n_ok,
        "n_frames": len(all_t),
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

    print(f"\n{n_ok} episode(s), {offset_ns/1e9:.1f}s timeline")
    print(f"  html    {out_dir / 'index.html'}")
    if not args.no_video:
        print(f"  video   {video_path}")
    if not args.no_mcap:
        print(f"  mcap    {mcap_path}")
        print(f"  layout  {out_dir / 'foxglove_layout.json'}")
        print("Open the mcap in Foxglove (app.foxglove.dev or desktop) and import the layout.")
        print("Or open index.html in a browser for the tiled dataset video + plots.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", required=True, type=Path,
                    help="folder of h5 / hdf5, a parent of many datasets, or one file")
    ap.add_argument("--out", default=None, type=Path,
                    help="output dir (default: <data>/dataset_viz) or a .mcap path")
    ap.add_argument("--list", action="store_true",
                    help="print catalog (mixed tree) or episode list (one dataset)")
    ap.add_argument("--each", action="store_true",
                    help="one visualizer per dataset under --data")
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
    ap.add_argument("--include-sensor-rgb", action="store_true",
                    help="also ingest sensors_rgb256 sidecar (huge)")
    ap.add_argument("--include-depth", action="store_true")
    args = ap.parse_args()

    src = args.data.expanduser().resolve()
    catalog = [] if src.is_file() else scan_dataset_roots(src, include_eval=args.include_eval)

    if len(catalog) > 1:
        print_catalog(catalog, src)
        if args.list and not args.each:
            print("\npass one child path, or --each (one viz per row). "
                  "--include-eval adds results/ rollouts.")
            return
        if not args.each:
            raise SystemExit(
                "mixed tree — pick one child from the list, or pass --each"
            )
        out_base = (args.out.expanduser().resolve() if args.out
                    else _ROOT / "experiments_output/default/dataset_viz")
        for ds in catalog:
            slug = _slug(src, ds.path)
            print(f"\n======== {slug} ({ds.kind}, {ds.n_eps} eps) ========")
            run_one(ds.path, out_base / slug, out_base / slug / "dataset.mcap", args)
        return

    if len(catalog) == 1 and src != catalog[0].path and src.is_dir():
        src = catalog[0].path

    if args.each and catalog:
        out_base = (args.out.expanduser().resolve() if args.out
                    else _ROOT / "experiments_output/default/dataset_viz")
        slug = _slug(src.parent, catalog[0].path)
        run_one(catalog[0].path, out_base / slug, out_base / slug / "dataset.mcap", args)
        return

    out_dir, mcap_path = _out_paths(src, args)
    run_one(src, out_dir, mcap_path, args)


if __name__ == "__main__":
    main()
