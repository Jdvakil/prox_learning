"""Convert coauthor pact_place_corridor rows into ACT per-episode HDF5s.

Source (HuggingFace `Lundii/pact_place_corridor_v5`, cloned to
`data/pact_place_corridor_v5`): one folder per recovered episode

    rows/<idx>_<sha>/
        trajectory.h5          group traj_0, same molmospaces layout as obstacle datagen
        episode_00000000_wrist_camera.mp4
        result.json            clean_success / scene_params / contact_audit

There is **no exo camera**. The collect used `FrankaSkinHybridWristOnlyCameraSystem`
on molmospaces `experiment/pact-vs-act-remediation-v2`. Scene XML version in
`result.json` is `pact_place_corridor_v2`; the HF name `v5` is the recovery schema.

Vanilla ACT ignores `/observations/proximity`, so one `--with_proximity` convert
serves both arms.

    python -m scripts.convert_pact_place_to_act \\
        --src data/pact_place_corridor_v5 \\
        --dst act_style_data/pact_place_corridor_v5 \\
        --with_proximity --prox_pool min --image_h 240 --image_w 320
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from scripts.convert_obstacle_to_act import (
    _decode_action,
    _decode_qpos_qvel,
    _episode_failed,
    _episode_proximity,
    _load_sensor_order,
    _video_frames,
)

CAM = "wrist_camera"


def _decode_obs_scene(grp) -> dict:
    if "obs_scene" not in grp:
        return {}
    raw = np.asarray(grp["obs_scene"]).item()
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    return {}


def _row_dirs(src: Path) -> list[Path]:
    rows = src / "rows" if (src / "rows").is_dir() else src
    dirs = [p for p in rows.iterdir() if p.is_dir() and (p / "trajectory.h5").is_file()]
    dirs.sort(key=lambda p: p.name)
    if not dirs:
        raise SystemExit(f"no rows/*/trajectory.h5 under {src}")
    return dirs


def _result(row: Path) -> dict:
    path = row / "result.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def convert(
    src: Path,
    dst_dir: Path,
    image_h: int,
    image_w: int,
    with_proximity: bool,
    prox_pool: str,
    max_episodes: int | None,
    require_clean: bool,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    rows = _row_dirs(src)
    print(f"[convert-place] {len(rows)} row dirs under {src}")
    sensor_order = _load_sensor_order(None) if with_proximity else None

    global_idx = 0
    n_skip_dirty = 0
    n_skip_video = 0
    n_skip_action = 0
    max_T = 0
    sides: dict[str, int] = {}
    for row in tqdm(rows, desc="rows"):
        if max_episodes is not None and global_idx >= max_episodes:
            break
        res = _result(row)
        if require_clean and not res.get("clean_success", True):
            n_skip_dirty += 1
            continue
        mp4 = row / "episode_00000000_wrist_camera.mp4"
        if not mp4.is_file():
            print(f"[skip] {row.name}: missing {mp4.name}")
            n_skip_video += 1
            continue
        with h5py.File(row / "trajectory.h5", "r") as handle:
            if "traj_0" not in handle:
                print(f"[skip] {row.name}: no traj_0")
                n_skip_action += 1
                continue
            grp = handle["traj_0"]
            if require_clean and _episode_failed(grp):
                n_skip_dirty += 1
                continue
            T_full = int(grp["actions/joint_pos"].shape[0])
            actions = np.zeros((T_full, 8), dtype=np.float32)
            valid = np.zeros(T_full, dtype=bool)
            for t in range(T_full):
                vec, ok = _decode_action(grp["actions/joint_pos"][t])
                actions[t] = vec
                valid[t] = ok
            T = int(valid.sum())
            if T == 0:
                print(f"[skip] {row.name}: no valid action rows")
                n_skip_action += 1
                continue
            actions = actions[:T]
            qpos = np.stack([_decode_qpos_qvel(grp["obs/agent/qpos"][t]) for t in range(T)])
            qvel = np.stack([_decode_qpos_qvel(grp["obs/agent/qvel"][t]) for t in range(T)])
            frames = _video_frames(mp4, image_h, image_w)
            if frames.shape[0] < T:
                print(f"[skip] {row.name}: {frames.shape[0]} wrist frames < {T}")
                n_skip_video += 1
                continue
            images = frames[:T]
            proximity = None
            if with_proximity:
                proximity = _episode_proximity(grp, T, sensor_order, pool=prox_pool)

            scene = _decode_obs_scene(grp)
            sp = res.get("scene_params") or scene.get("scene_params") or {}
            side = str(sp.get("pact_intrusion_side") or "unknown")
            sides[side] = sides.get(side, 0) + 1

            out_path = dst_dir / f"episode_{global_idx}.hdf5"
            with h5py.File(out_path, "w") as dst:
                dst.attrs["sim"] = True
                dst.attrs["source_row"] = row.name
                dst.attrs["episode_id"] = str(res.get("episode_id") or "")
                dst.attrs["behavior_class"] = str(scene.get("behavior_class") or "?")
                dst.attrs["has_bar"] = bool(sp.get("protrusion_present", True))
                dst.attrs["intrusion_side"] = side
                dst.attrs["inbound_deflected"] = bool(scene.get("inbound_deflected", False))
                dst.attrs["outbound_deflected"] = bool(scene.get("outbound_deflected", False))
                dst.attrs["clean_success"] = bool(res.get("clean_success", False))
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
                imgs.create_dataset(
                    CAM,
                    data=images,
                    dtype="uint8",
                    chunks=(1, images.shape[1], images.shape[2], 3),
                    compression="gzip",
                    compression_opts=4,
                )
        max_T = max(max_T, T)
        global_idx += 1

    meta = {
        "src": str(src),
        "with_proximity": with_proximity,
        "prox_pool": prox_pool,
        "camera_names": [CAM],
        "num_episodes": global_idx,
        "episode_len": max_T + 2,
        "max_T": max_T,
        "n_skip_dirty": n_skip_dirty,
        "n_skip_video": n_skip_video,
        "n_skip_action": n_skip_action,
        "intrusion_sides": sides,
        "image_h": image_h,
        "image_w": image_w,
    }
    (dst_dir / "convert_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"\n[convert-place] DONE — wrote {global_idx} episodes to {dst_dir}\n"
        f"[convert-place] skipped dirty={n_skip_dirty} video={n_skip_video} "
        f"action={n_skip_action} sides={sides}\n"
        f"[convert-place] >>> set TASK_CONFIGS['pact_place_corridor_v5']: "
        f"num_episodes={global_idx}, episode_len={max_T + 2}  (max T = {max_T})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--image_h", type=int, default=240)
    parser.add_argument("--image_w", type=int, default=320)
    parser.add_argument("--with_proximity", action="store_true")
    parser.add_argument("--prox_pool", choices=("mean", "min"), default="min")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--keep_dirty",
        action="store_true",
        help="keep rows whose result.json clean_success is false",
    )
    args = parser.parse_args()
    convert(
        src=args.src,
        dst_dir=args.dst,
        image_h=args.image_h,
        image_w=args.image_w,
        with_proximity=args.with_proximity,
        prox_pool=args.prox_pool,
        max_episodes=args.max_episodes,
        require_clean=not args.keep_dirty,
    )


if __name__ == "__main__":
    main()
