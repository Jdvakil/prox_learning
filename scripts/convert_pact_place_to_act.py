"""Convert pact_place / pick-and-place rows into ACT per-episode HDF5s.

Source: one folder per episode

    rows/<idx>_<sha>/
        trajectory.h5
        episode_00000000_wrist_camera.mp4          # required
        episode_00000000_exo_camera_1.mp4          # optional; v1011d / v12 table cam
        episode_{sha}_table_camera.mp4             # optional; hashed v10 names
        result.json

Wrist-only dumps (Lundii v5) still convert. Dumps that also ship a table/exo RGB
mp4 write that camera under `/observations/images/<stem>` so TASK_CONFIGS can
list `exo_camera_1` (or `table_camera`). Depth and heatmap mp4s are ignored —
ACT never reads them.

Vanilla ACT ignores `/observations/proximity`, so one `--with_proximity` convert
serves both arms.

    python -m scripts.convert_pact_place_to_act \\
        --src data/pact_pick_n_place_v2/data/v1011d \\
        --dst act_style_data/pact_pick_n_place_v2/data/v1011d \\
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

# ACT image keys. Wrist is required. Table/exo is written when the sibling mp4 exists.
RGB_STEMS = ("wrist_camera", "exo_camera_1", "table_camera")
# Match obstacle TASK_CONFIGS: exo/table first, then wrist.
CAMERA_ORDER = ("exo_camera_1", "table_camera", "wrist_camera")
CLEAN_KEYS = ("clean_success", "v108_clean_success", "task_success", "accepted")


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


def _row_is_clean(res: dict) -> bool:
    """Keep if the first present success flag is true. Missing every flag → keep."""
    for key in CLEAN_KEYS:
        if key in res:
            return bool(res[key])
    return True


def find_rgb_mp4(row: Path, stem: str) -> Path | None:
    """Sidecar RGB mp4 for one camera stem. Same name patterns as dataset_viz.glob_mp4.

    Skips `*_depth.mp4` because those files do not end in `_{stem}.mp4`.
    """
    padded = "00000000"
    exact = row / f"episode_{padded}_{stem}.mp4"
    if exact.is_file():
        return exact
    batch = sorted(row.glob(f"episode_{padded}_{stem}_batch_*.mp4"))
    if batch:
        return batch[0]
    hashed = row / f"episode_{row.name}_{stem}.mp4"
    if hashed.is_file():
        return hashed
    loose = sorted(
        p
        for p in row.glob(f"episode_*_{stem}.mp4")
        if "_batch_" not in p.name and not p.name.endswith("_depth.mp4")
    )
    if len(loose) == 1:
        return loose[0]
    return None


def find_rgb_mp4s(row: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for stem in RGB_STEMS:
        path = find_rgb_mp4(row, stem)
        if path is not None:
            found[stem] = path
    return found


def ordered_camera_names(found: dict[str, Path]) -> list[str]:
    return [stem for stem in CAMERA_ORDER if stem in found]


def convert(
    src: Path,
    dst_dir: Path,
    image_h: int,
    image_w: int,
    with_proximity: bool,
    prox_pool: str,
    max_episodes: int | None,
    require_clean: bool,
    task_name: str | None = None,
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
    camera_names: list[str] | None = None
    for row in tqdm(rows, desc="rows"):
        if max_episodes is not None and global_idx >= max_episodes:
            break
        res = _result(row)
        if require_clean and not _row_is_clean(res):
            n_skip_dirty += 1
            continue
        found = find_rgb_mp4s(row)
        if "wrist_camera" not in found:
            print(f"[skip] {row.name}: missing wrist RGB mp4")
            n_skip_video += 1
            continue
        cams = ordered_camera_names(found)
        if camera_names is None:
            camera_names = cams
            print(f"[convert-place] cameras this convert: {camera_names}")
        elif cams != camera_names:
            print(
                f"[skip] {row.name}: cameras {cams} != {camera_names} "
                "(every episode must write the same image keys)"
            )
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
            images: dict[str, np.ndarray] = {}
            bad_video = False
            for cam in camera_names:
                frames = _video_frames(found[cam], image_h, image_w)
                if frames.shape[0] < T:
                    print(
                        f"[skip] {row.name}: {frames.shape[0]} {cam} frames < {T}"
                    )
                    n_skip_video += 1
                    bad_video = True
                    break
                images[cam] = frames[:T]
            if bad_video:
                continue
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
                dst.attrs["clean_success"] = _row_is_clean(res)
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
        max_T = max(max_T, T)
        global_idx += 1

    if camera_names is None:
        camera_names = []
    task_hint = task_name or dst_dir.name
    meta = {
        "src": str(src),
        "with_proximity": with_proximity,
        "prox_pool": prox_pool,
        "camera_names": camera_names,
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
        f"[convert-place] cameras={camera_names}\n"
        f"[convert-place] >>> set TASK_CONFIGS[{task_hint!r}]: "
        f"num_episodes={global_idx}, episode_len={max_T + 2}, "
        f"camera_names={camera_names}  (max T = {max_T})"
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
        "--task_name",
        type=str,
        default=None,
        help="TASK_CONFIGS key to print at the end (default: dst folder name)",
    )
    parser.add_argument(
        "--keep_dirty",
        action="store_true",
        help="keep rows whose success flag is false",
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
        task_name=args.task_name,
    )


if __name__ == "__main__":
    main()
