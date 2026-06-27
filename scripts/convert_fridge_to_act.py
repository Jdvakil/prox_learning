"""Convert the two-level fridge datagen run into ACT/PACT episode HDF5s.

The output is compatible with the local ACT trainer. It contains the standard
RGB + qpos/action fields used by vanilla ACT plus:

    /observations/proximity_positions  (T, N, 3) float32

Each proximity token is a compact local feature from one 8x8 depth sensor:
normalized pixel x/y of the nearest valid return and normalized depth. This is
the feature consumed by the existing PACT transformer hook in submodules/act.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

CAM_NAMES = ("exo_camera_1", "wrist_camera")
ARM_DIM = 7
GRIP_DIM_OBS = 2
GRIP_DIM_ACT = 1
QPOS_DIM = ARM_DIM + GRIP_DIM_OBS
ACTION_DIM = ARM_DIM + GRIP_DIM_ACT


def _decode_jsonrow(blob: np.ndarray) -> dict:
    raw = bytes(blob).split(b"\x00", 1)[0]
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _decode_qpos_qvel(blob: np.ndarray) -> np.ndarray:
    out = np.zeros(QPOS_DIM, dtype=np.float32)
    try:
        d = _decode_jsonrow(blob)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return out
    arm = d.get("arm") or []
    grip = d.get("gripper") or []
    out[: min(len(arm), ARM_DIM)] = arm[:ARM_DIM]
    out[ARM_DIM : ARM_DIM + min(len(grip), GRIP_DIM_OBS)] = grip[:GRIP_DIM_OBS]
    return out


def _decode_action(blob: np.ndarray) -> tuple[np.ndarray, bool]:
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    try:
        d = _decode_jsonrow(blob)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return out, False
    arm = d.get("arm") or []
    if not arm:
        return out, False
    out[: min(len(arm), ARM_DIM)] = arm[:ARM_DIM]
    grip = d.get("gripper") or []
    if grip:
        out[ARM_DIM] = float(grip[0])
    return out, True


def _episode_failed(grp: h5py.Group) -> bool:
    if "fail" not in grp:
        return False
    fail = np.asarray(grp["fail"])
    return bool(fail[-1]) if fail.size else False


def _video_frames(path: Path, image_h: int | None, image_w: int | None) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (
                image_h is not None
                and image_w is not None
                and frame.shape[:2] != (image_h, image_w)
            ):
                frame = cv2.resize(frame, (image_w, image_h), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"{path} decoded to 0 frames")
    return np.stack(frames, axis=0)


def _find_h5_files(src: Path) -> list[Path]:
    if src.is_file() and src.suffix == ".h5":
        return [src]
    files = sorted(src.glob("house_*/trajectories*.h5"))
    if not files:
        files = sorted(src.glob("trajectories*.h5"))
    if not files:
        raise SystemExit(f"no trajectories*.h5 found under {src}")
    return files


def _sensor_sort_key(name: str) -> tuple[int, int, str]:
    link = 999
    idx = 999
    parts = name.split("_sensor_")
    if len(parts) == 2:
        try:
            link = int(parts[0].replace("link", "").replace("_front", "").replace("_back", ""))
            idx = int(parts[1])
        except ValueError:
            pass
    return link, idx, name


def _proximity_features(grp: h5py.Group, T: int, depth_max_m: float) -> tuple[np.ndarray, list[str]]:
    prox = grp["obs/proximity"]
    sensor_names = sorted(prox.keys(), key=_sensor_sort_key)
    feats = np.zeros((T, len(sensor_names), 3), dtype=np.float32)
    for si, name in enumerate(sensor_names):
        depth = np.asarray(prox[name][:T], dtype=np.float32).mean(axis=1)
        for t in range(T):
            img = depth[t]
            valid = np.isfinite(img) & (img > 0.0) & (img < depth_max_m)
            if not valid.any():
                feats[t, si] = (0.0, 0.0, 1.0)
                continue
            masked = np.where(valid, img, np.inf)
            y, x = np.unravel_index(int(np.argmin(masked)), masked.shape)
            h, w = img.shape
            x_norm = (float(x) / max(w - 1, 1)) * 2.0 - 1.0
            y_norm = (float(y) / max(h - 1, 1)) * 2.0 - 1.0
            d_norm = float(np.clip(masked[y, x] / depth_max_m, 0.0, 1.0))
            feats[t, si] = (x_norm, y_norm, d_norm)
    return feats, sensor_names


def convert(
    srcs: list[Path],
    dst_dir: Path,
    image_h: int | None,
    image_w: int | None,
    only_success: bool,
    max_episodes: int | None,
    depth_max_m: float,
    append: bool,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    h5_files = []
    for src in srcs:
        h5_files.extend(_find_h5_files(src))
    h5_files = sorted(h5_files)
    print(f"[convert-fridge] {len(h5_files)} h5 file(s) from {len(srcs)} source(s)")

    if append:
        existing = sorted(dst_dir.glob("episode_*.hdf5"))
        global_idx = 1 + max((int(p.stem.split("_", 1)[1]) for p in existing), default=-1)
    else:
        for old in dst_dir.glob("episode_*.hdf5"):
            old.unlink()
        global_idx = 0
    start_idx = global_idx
    n_skipped_fail = 0
    max_t = 0
    sensor_names_ref: list[str] | None = None
    for h5_path in h5_files:
        if max_episodes is not None and global_idx >= max_episodes:
            break
        h5_parent = h5_path.parent
        with h5py.File(h5_path, "r") as f:
            traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_", 1)[1]))
            for traj_key in tqdm(traj_keys, desc=h5_parent.name):
                if max_episodes is not None and global_idx >= max_episodes:
                    break
                ep_idx = int(traj_key.split("_", 1)[1])
                grp = f[traj_key]
                if only_success and _episode_failed(grp):
                    n_skipped_fail += 1
                    continue

                T_full = grp["actions/joint_pos"].shape[0]
                actions = np.zeros((T_full, ACTION_DIM), dtype=np.float32)
                valid = np.zeros(T_full, dtype=bool)
                for t in range(T_full):
                    actions[t], valid[t] = _decode_action(grp["actions/joint_pos"][t])
                T = int(valid.sum())
                if T == 0:
                    print(f"[skip] {h5_parent.name}/{traj_key}: no valid action rows")
                    continue
                actions = actions[:T]

                qpos = np.stack([_decode_qpos_qvel(grp["obs/agent/qpos"][t]) for t in range(T)])
                qvel = np.stack([_decode_qpos_qvel(grp["obs/agent/qvel"][t]) for t in range(T)])
                proximity, sensor_names = _proximity_features(grp, T, depth_max_m)
                if sensor_names_ref is None:
                    sensor_names_ref = sensor_names
                elif sensor_names != sensor_names_ref:
                    raise RuntimeError("proximity sensor order changed across episodes")

                images = {}
                ok_videos = True
                for cam in CAM_NAMES:
                    mp4 = h5_parent / f"episode_{ep_idx:08d}_{cam}_batch_1_of_1.mp4"
                    if not mp4.exists():
                        print(f"[skip] {h5_parent.name}/{traj_key}: missing {mp4.name}")
                        ok_videos = False
                        break
                    frames = _video_frames(mp4, image_h, image_w)
                    if frames.shape[0] < T:
                        print(f"[skip] {h5_parent.name}/{traj_key}/{cam}: {frames.shape[0]} < {T}")
                        ok_videos = False
                        break
                    images[cam] = frames[:T]
                if not ok_videos:
                    continue

                out_path = dst_dir / f"episode_{global_idx}.hdf5"
                with h5py.File(out_path, "w") as dst:
                    dst.attrs["sim"] = True
                    dst.attrs["depth_max_m"] = float(depth_max_m)
                    dst.create_dataset("action", data=actions, dtype="float32")
                    obs = dst.create_group("observations")
                    obs.create_dataset("qpos", data=qpos.astype(np.float32))
                    obs.create_dataset("qvel", data=qvel.astype(np.float32))
                    obs.create_dataset("proximity_positions", data=proximity, dtype="float32")
                    obs.create_dataset(
                        "proximity_sensor_names",
                        data=np.asarray(sensor_names, dtype=h5py.string_dtype("utf-8")),
                    )
                    imgs = obs.create_group("images")
                    for cam, arr in images.items():
                        imgs.create_dataset(
                            cam,
                            data=arr,
                            dtype="uint8",
                            chunks=(1, arr.shape[1], arr.shape[2], 3),
                            compression="gzip",
                            compression_opts=4,
                        )
                max_t = max(max_t, T)
                global_idx += 1

    print(
        f"\n[convert-fridge] DONE wrote {global_idx - start_idx} new episodes "
        f"({global_idx} total indexed) to {dst_dir}\n"
        f"[convert-fridge] skipped {n_skipped_fail} failed episodes "
        f"({'only_success=ON' if only_success else 'only_success=OFF'})\n"
        f"[convert-fridge] sensors={len(sensor_names_ref or [])}, max_T={max_t}\n"
        f"[convert-fridge] set constants.py fridge_*: num_episodes={global_idx}, "
        f"episode_len={max_t + 2}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, nargs="+", required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--depth_max_m", type=float, default=0.55)
    p.add_argument("--keep_failures", action="store_true")
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--append", action="store_true")
    args = p.parse_args()
    convert(
        srcs=args.src,
        dst_dir=args.dst,
        image_h=args.image_h,
        image_w=args.image_w,
        only_success=not args.keep_failures,
        max_episodes=args.max_episodes,
        depth_max_m=args.depth_max_m,
        append=args.append,
    )


if __name__ == "__main__":
    main()
