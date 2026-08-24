"""Convert the hybrid-skin OBSTACLE datagen run into ACT-style per-episode HDF5s.

Default mode is the *vanilla ACT baseline*: RGB (exo + wrist) + proprioception (qpos)
only. Pass --with_proximity to ALSO export the 40-sensor skin depths as
/observations/proximity (T, 40, 8, 8) in meters, stacked in the Safety-CVAE's
meta.json sensor order (the single source of truth — note link5_back precedes
link5_front there). The proximity group is what P+ACT (PACT) feeds to the frozen
Safety-CVAE feature extractor; vanilla ACT simply ignores it, so ONE --with_proximity
dataset serves both arms of the comparison with byte-identical RGB/qpos/action.

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
import shutil
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
DEFAULT_PROX_META = "assets/safety/cvae_v3/meta.json"
APPROACH_PHASES = {2, 3}  # pregrasp, grasp — same as analyze_obstacle_dataset.py
GRIP_CLOSE = 4


def _load_sensor_order(meta_path: Path) -> list[str]:
    """The authoritative 40-sensor stacking order = cvae_v3 meta.json['sensors']."""
    meta = json.loads(Path(meta_path).read_text())
    order = list(meta["sensors"])
    print(f"[convert] proximity sensor order from {meta_path}: {len(order)} sensors")
    return order


def _episode_proximity(
    grp: h5py.Group, T: int, sensor_order: list[str], pool: str = "mean"
) -> np.ndarray:
    """Read obs/proximity/<sensor> for each sensor in order -> (T, 40, 8, 8) float32 (m).

    Source per-sensor shape is (T_full, n_substeps, 8, 8); substeps are pooled
    ('mean' is the old convention; 'min' keeps a one-substep graze) and the first
    T rows are kept to align with action/qpos.
    """
    prox_grp = grp.get("obs/proximity")
    if prox_grp is None:
        raise SystemExit(
            "--with_proximity: source has no obs/proximity group. This run was not "
            "collected with the hybrid skin (model_hybrid.xml / 40 sensors)."
        )
    chans = []
    for name in sensor_order:
        if name not in prox_grp:
            raise SystemExit(
                f"--with_proximity: sensor {name!r} missing from obs/proximity (run has "
                f"{len(prox_grp)} sensors, not the 40-sensor hybrid skin the CVAE expects)."
            )
        d = np.asarray(prox_grp[name][:T], dtype=np.float32)
        if d.ndim == 4:        # (T, n_substeps, 8, 8)
            if pool == "min":
                d = d.min(axis=1)
            elif pool == "mean":
                d = d.mean(axis=1)
            else:
                raise SystemExit(f"unknown --prox_pool {pool!r}")
        elif d.ndim != 3:      # expect (T, 8, 8)
            raise SystemExit(f"sensor {name!r} has unexpected shape {d.shape}")
        chans.append(d)        # (T, 8, 8)
    return np.stack(chans, axis=1)  # (T, 40, 8, 8)


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


def _episode_tags(grp: h5py.Group) -> dict:
    """Behavior class + inbound arm-vs-env contacts. Same rules as analyze_obstacle_dataset."""
    T = int(grp["rewards"].shape[0]) if "rewards" in grp else int(grp["actions/joint_pos"].shape[0])
    empty = {
        "behavior_class": "?",
        "has_bar": False,
        "approach_contacts": 0.0,
        "any_approach_collision": False,
        "ok_scene": False,
    }
    if "obs_scene" not in grp:
        return empty
    scene = json.loads(np.asarray(grp["obs_scene"]).item())
    sp = scene.get("scene_params", {})
    cm = scene.get("collision_metrics", {})
    psc = np.asarray(cm.get("per_step_contacts", [0] * T), dtype=float)
    if psc.shape[0] != T:
        psc = np.resize(psc, T)
    if "obs/extra/policy_phase" in grp:
        phase = grp["obs/extra/policy_phase"][:].astype(int)
    else:
        phase = np.zeros(T, dtype=int)
    first_close = np.where(phase == GRIP_CLOSE)[0]
    grip_start = int(first_close[0]) if first_close.size else T
    inbound = np.arange(T) < grip_start
    approach_steps = np.isin(phase, list(APPROACH_PHASES)) & inbound
    approach_contacts = float(psc[approach_steps].sum())
    return {
        "behavior_class": str(scene.get("behavior_class", "?")),
        "has_bar": bool(sp.get("protrusion_present", False)),
        "approach_contacts": approach_contacts,
        "any_approach_collision": approach_contacts > 0,
        "ok_scene": True,
    }


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
    with_proximity: bool = False,
    prox_meta: Path | None = None,
    prox_pool: str = "mean",
    skip_approach_collision: bool = False,
    keep_deflect_collisions: bool = False,
    only_deflect: bool = False,
    upsample_deflect: int = 1,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    h5_files = _find_h5_files(src)
    print(f"[convert] {len(h5_files)} h5 file(s) under {src}")

    sensor_order = _load_sensor_order(prox_meta or DEFAULT_PROX_META) if with_proximity else None

    global_idx = 0
    n_skipped_fail = 0
    n_skipped_collision = 0
    n_skipped_not_deflect = 0
    n_deflect_written = 0
    n_free_written = 0
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

                tags = _episode_tags(grp)
                is_deflect = tags["behavior_class"] == "deflect"
                if only_deflect and not is_deflect:
                    n_skipped_not_deflect += 1
                    continue
                if skip_approach_collision and tags["any_approach_collision"]:
                    if not (keep_deflect_collisions and is_deflect):
                        n_skipped_collision += 1
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

                proximity = None
                if with_proximity:
                    proximity = _episode_proximity(grp, T, sensor_order, pool=prox_pool)

                n_copies = max(1, int(upsample_deflect)) if is_deflect else 1
                first_path = None
                for copy_i in range(n_copies):
                    out_path = dst_dir / f"episode_{global_idx}.hdf5"
                    if copy_i == 0:
                        with h5py.File(out_path, "w") as dst:
                            dst.attrs["sim"] = True
                            dst.attrs["behavior_class"] = tags["behavior_class"]
                            dst.attrs["has_bar"] = tags["has_bar"]
                            dst.attrs["approach_contacts"] = tags["approach_contacts"]
                            dst.create_dataset("action", data=actions, dtype="float32")
                            obs = dst.create_group("observations")
                            obs.create_dataset("qpos", data=qpos.astype(np.float32))
                            obs.create_dataset("qvel", data=qvel.astype(np.float32))
                            if proximity is not None:
                                obs.create_dataset(
                                    "proximity",
                                    data=proximity.astype(np.float32),
                                    dtype="float32",
                                    chunks=(1, proximity.shape[1], proximity.shape[2], proximity.shape[3]),
                                    compression="gzip",
                                    compression_opts=4,
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
                        first_path = out_path
                    else:
                        shutil.copy2(first_path, out_path)
                    if is_deflect:
                        n_deflect_written += 1
                    else:
                        n_free_written += 1
                    max_T = max(max_T, T)
                    global_idx += 1

    meta = {
        "src": str(src),
        "with_proximity": with_proximity,
        "prox_pool": prox_pool,
        "skip_approach_collision": skip_approach_collision,
        "keep_deflect_collisions": keep_deflect_collisions,
        "only_deflect": only_deflect,
        "upsample_deflect": int(upsample_deflect),
        "num_episodes": global_idx,
        "episode_len": max_T + 2,
        "max_T": max_T,
        "n_skipped_fail": n_skipped_fail,
        "n_skipped_collision": n_skipped_collision,
        "n_skipped_not_deflect": n_skipped_not_deflect,
        "n_deflect_written": n_deflect_written,
        "n_free_written": n_free_written,
    }
    (dst_dir / "convert_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"\n[convert] DONE — wrote {global_idx} episodes to {dst_dir}\n"
        f"[convert] skipped fail={n_skipped_fail} collision={n_skipped_collision} "
        f"not_deflect={n_skipped_not_deflect}\n"
        f"[convert] written deflect={n_deflect_written} free={n_free_written} "
        f"prox_pool={prox_pool}\n"
        f"[convert] >>> set submodules/act/constants.py obstacle_pact_avoid_v1: "
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
    p.add_argument(
        "--with_proximity",
        action="store_true",
        help="also export /observations/proximity (T,40,8,8) for P+ACT (PACT)",
    )
    p.add_argument(
        "--prox_meta",
        type=Path,
        default=Path(DEFAULT_PROX_META),
        help="Safety-CVAE meta.json giving the authoritative 40-sensor stacking order",
    )
    p.add_argument(
        "--prox_pool",
        choices=("mean", "min"),
        default="mean",
        help="how to pool the 4 skin substeps: 'mean' (old) or 'min' (keeps a graze)",
    )
    p.add_argument(
        "--skip_approach_collision",
        action="store_true",
        help="drop episodes with arm-vs-env contacts during pregrasp/grasp (do not teach scrapes)",
    )
    p.add_argument(
        "--keep_deflect_collisions",
        action="store_true",
        help="with --skip_approach_collision, still keep deflect episodes even if they grazed",
    )
    p.add_argument(
        "--only_deflect",
        action="store_true",
        help="keep only behavior_class=deflect episodes",
    )
    p.add_argument(
        "--upsample_deflect",
        type=int,
        default=1,
        help="write each deflect episode this many times (1 = no upsample)",
    )
    args = p.parse_args()
    convert(
        args.src,
        args.dst,
        args.image_h,
        args.image_w,
        only_success=not args.keep_failures,
        max_episodes=args.max_episodes,
        with_proximity=args.with_proximity,
        prox_meta=args.prox_meta,
        prox_pool=args.prox_pool,
        skip_approach_collision=args.skip_approach_collision,
        keep_deflect_collisions=args.keep_deflect_collisions,
        only_deflect=args.only_deflect,
        upsample_deflect=args.upsample_deflect,
    )


if __name__ == "__main__":
    main()
