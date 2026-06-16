"""Convert the hybrid-skin OBSTACLE datagen run into ACT-style per-episode HDF5s.

This is the *vanilla ACT baseline* converter: RGB (exo + wrist) + proprioception
(qpos) only. No proximity / skin channels are exported — that is deliberately the
no-proximity baseline the safety experiments are compared against.

Source layout (one datagen run dir):
  <run>/house_<k>/trajectories_batch_1_of_1.h5     groups traj_0, traj_1, ...
      obs/agent/qpos      (T, 2000) uint8  JSON rows {"arm":[7], "gripper":[2]}
      obs/agent/qvel      (T, 2000) uint8  JSON rows {"arm":[7], "gripper":[2]}
      actions/joint_pos   (T, 2000) uint8  JSON rows {"arm":[7], "gripper":[1]}
      fail                (T,)      bool   (fail[-1] == episode failed)
  <run>/house_<k>/episode_<i:08d>_exo_camera_1_batch_1_of_1.mp4   RGB, T frames
  <run>/house_<k>/episode_<i:08d>_wrist_camera_batch_1_of_1.mp4   RGB, T frames
  (the `episode_<i>` index matches the `traj_<i>` group in that house's h5.)

Target layout (ACT trainer — see submodules/act/utils.py:EpisodicDataset):
  <dst>/episode_<g>.hdf5            g = contiguous 0..N-1 across all houses
    attrs['sim'] = True
    /action                              (T, 8)        arm(7) + gripper_cmd(1)
    /observations/qpos                   (T, 9)        arm(7) + gripper(2)
    /observations/qvel                   (T, 9)        arm(7) + gripper(2)
    /observations/images/exo_camera_1    (T, H, W, 3)  uint8
    /observations/images/wrist_camera    (T, H, W, 3)  uint8

The gripper command in `actions/joint_pos` is the FR3 hand actuator value (0=one
extreme, 255=the other); it is exported verbatim and the ACT eval policy snaps the
network's prediction back to {0, 255} at rollout time.

The trailing `actions/joint_pos` row in each trajectory is an empty `{}` (no command
issued on the final frame); it is dropped, so each episode has T = T_h5 - 1 steps.

Run (from repo root, mlspaces env):
    python -m scripts.convert_obstacle_to_act \
        --src assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855 \
        --dst act_style_data/obstacle_v1 \
        --image_h 240 --image_w 320

The converter prints the final episode count and the max episode length — paste both
into the `obstacle_baseline` entry of submodules/act/constants.py (num_episodes /
episode_len).
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
GRIP_DIM_OBS = 2  # qpos/qvel: two finger joints
GRIP_DIM_ACT = 1  # joint_pos action: one gripper command
QPOS_DIM = ARM_DIM + GRIP_DIM_OBS  # 9
ACTION_DIM = ARM_DIM + GRIP_DIM_ACT  # 8


def _decode_jsonrow(blob: np.ndarray) -> dict:
    raw = bytes(blob).split(b"\x00", 1)[0]
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _decode_qpos_qvel(blob: np.ndarray) -> np.ndarray:
    """Decode an arm+gripper JSON row into a 9-d vector."""
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
    """Decode a joint_pos action row into an 8-d vector.

    Returns (vec, is_valid). The trailing row in each trajectory is `{}` and must be
    filtered by the caller.
    """
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


def _video_frames(path: Path, image_h: int | None, image_w: int | None) -> np.ndarray:
    """Decode every frame of an MP4 into an (N, H, W, 3) uint8 RGB array."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()  # BGR
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


def _episode_failed(grp: h5py.Group) -> bool:
    """Episode failed iff the last `fail` flag is set (matches analyze_obstacle_dataset)."""
    if "fail" not in grp:
        return False
    fail = np.asarray(grp["fail"])
    return bool(fail[-1]) if fail.size else False


def _find_h5_files(src: Path) -> list[Path]:
    """Accept a single .h5, a house dir, or a whole run dir (globs house_*/...h5)."""
    if src.is_file() and src.suffix == ".h5":
        return [src]
    files = sorted(src.glob("house_*/trajectories*.h5"))
    if not files:  # maybe `src` is itself a house dir
        files = sorted(src.glob("trajectories*.h5"))
    if not files:
        raise SystemExit(f"no trajectories*.h5 found under {src}")
    return files


def convert(
    src: Path,
    dst_dir: Path,
    image_h: int | None,
    image_w: int | None,
    only_success: bool,
    max_episodes: int | None,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    h5_files = _find_h5_files(src)
    print(f"[convert] {len(h5_files)} h5 file(s) under {src}")

    global_idx = 0
    n_skipped_fail = 0
    max_T = 0
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
                    a, ok = _decode_action(grp["actions/joint_pos"][t])
                    actions[t] = a
                    valid[t] = ok
                T = int(valid.sum())
                if T == 0:
                    print(f"[skip] {h5_parent.name}/{traj_key}: no valid action rows")
                    continue
                actions = actions[:T]

                qpos = np.stack([_decode_qpos_qvel(grp["obs/agent/qpos"][t]) for t in range(T)])
                qvel = np.stack([_decode_qpos_qvel(grp["obs/agent/qvel"][t]) for t in range(T)])

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
                        print(
                            f"[skip] {h5_parent.name}/{traj_key}/{cam}: "
                            f"{frames.shape[0]} frames < required {T}"
                        )
                        ok_videos = False
                        break
                    images[cam] = frames[:T]
                if not ok_videos:
                    continue

                out_path = dst_dir / f"episode_{global_idx}.hdf5"
                with h5py.File(out_path, "w") as dst:
                    dst.attrs["sim"] = True
                    dst.create_dataset("action", data=actions, dtype="float32")
                    obs = dst.create_group("observations")
                    obs.create_dataset("qpos", data=qpos.astype(np.float32))
                    obs.create_dataset("qvel", data=qvel.astype(np.float32))
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
                max_T = max(max_T, T)
                global_idx += 1

    print(
        f"\n[convert] DONE — wrote {global_idx} episodes to {dst_dir}\n"
        f"[convert] skipped {n_skipped_fail} failed episodes "
        f"({'only_success=ON' if only_success else 'only_success=OFF'})\n"
        f"[convert] >>> set submodules/act/constants.py obstacle_baseline: "
        f"num_episodes={global_idx}, episode_len={max_T + 2}  (max T = {max_T})"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        type=Path,
        required=True,
        help="datagen run dir (globs house_*/trajectories*.h5), a house dir, or a single .h5",
    )
    p.add_argument("--dst", type=Path, required=True, help="output dir for episode_*.hdf5")
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument(
        "--keep_failures",
        action="store_true",
        help="keep episodes whose fail[-1] is set (default: drop them, train on successes only)",
    )
    p.add_argument("--max_episodes", type=int, default=None, help="cap total episodes (smoke test)")
    args = p.parse_args()
    convert(
        args.src,
        args.dst,
        args.image_h,
        args.image_w,
        only_success=not args.keep_failures,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
