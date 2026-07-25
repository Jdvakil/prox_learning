#!/usr/bin/env python3
"""Manifest-driven ACT conversion of the canonical 100-row subset.

Handoff step 14. The committed converter ``scripts/convert_obstacle_to_act.py``
holds the conversion semantics, and this wrapper imports and reuses its decode
and video functions verbatim -- ``_decode_action``, ``_decode_qpos_qvel``,
``_video_frames`` -- together with its dimension constants and output schema.
Nothing in the committed converter is modified.

Two committed assumptions do not hold for the v2 manifest-runner layout, which
is why the committed ``convert()`` entry point cannot be called directly:

1. ``_find_h5_files`` globs ``house_*/trajectories*.h5``; the manifest runner
   writes ``rows/<episode_id>/trajectory.h5``.
2. It expects ``episode_<i:08d>_<cam>_batch_1_of_1.mp4``; the runner writes
   ``episode_00000000_<cam>.mp4`` (one episode per row directory).

More importantly, the committed entry point assigns the ACT episode index from
filesystem iteration order. This wrapper instead takes the episode set *and*
the output index from the canonical manifest's ``act_episode_index``, so the
conversion never depends on directory order.

Output per episode matches the committed target layout exactly::

    episode_<g>.hdf5
      attrs['sim'] = True
      /action                            (T, 8)
      /observations/qpos                 (T, 9)
      /observations/qvel                 (T, 9)
      /observations/images/exo_camera_1  (T, H, W, 3) uint8
      /observations/images/wrist_camera  (T, H, W, 3) uint8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def _repo_root() -> Path:
    """Locate the repo whether this file sits in scripts/ or is run from elsewhere."""
    marker = Path("scripts") / "convert_obstacle_to_act.py"
    for cand in (Path(__file__).resolve().parents[1], Path.cwd(), Path("/root/prox_learning_hybrid_safety")):
        if (cand / marker).is_file():
            return cand
    raise SystemExit("cannot locate the prox_learning_hybrid_safety repo root")


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

from scripts.convert_obstacle_to_act import (
    ACTION_DIM,
    CAM_NAMES,
    QPOS_DIM,
    _decode_action,
    _decode_qpos_qvel,
    _video_frames,
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def semantic_hash(path: Path) -> str:
    """Hash only the scientific content of an ACT episode file.

    Excludes every HDF5 container detail (allocation order, btree layout,
    superblock timestamps) so that two conversions can be compared on content
    alone when the container bytes differ for nonsemantic reasons.
    """
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        h.update(f"sim={bool(f.attrs['sim'])}".encode())
        names: list[str] = []
        f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        for name in sorted(names):
            ds = f[name]
            arr = np.ascontiguousarray(ds[()])
            h.update(name.encode())
            h.update(b"\x1f")
            h.update(str(arr.dtype.str).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def convert_one(src_h5: Path, row_dir: Path, dst: Path, image_h: int, image_w: int) -> dict:
    with h5py.File(src_h5, "r") as f:
        traj_keys = [k for k in f if k.startswith("traj_")]
        if len(traj_keys) != 1:
            raise RuntimeError(f"{src_h5}: expected 1 traj group, found {len(traj_keys)}")
        g = f[traj_keys[0]]

        T_full = g["actions/joint_pos"].shape[0]
        actions = np.zeros((T_full, ACTION_DIM), dtype=np.float32)
        valid = np.zeros(T_full, dtype=bool)
        for t in range(T_full):
            a, ok = _decode_action(g["actions/joint_pos"][t])
            actions[t] = a
            valid[t] = ok
        T = int(valid.sum())
        if T == 0:
            raise RuntimeError(f"{src_h5}: no valid action rows")
        actions = actions[:T]

        qpos = np.stack([_decode_qpos_qvel(g["obs/agent/qpos"][t]) for t in range(T)])
        qvel = np.stack([_decode_qpos_qvel(g["obs/agent/qvel"][t]) for t in range(T)])

    images = {}
    for cam in CAM_NAMES:
        hits = [p for p in sorted(row_dir.glob(f"episode_*_{cam}.mp4")) if "_depth" not in p.name]
        if len(hits) != 1:
            raise RuntimeError(f"{row_dir}: expected exactly 1 {cam} mp4, found {len(hits)}")
        frames = _video_frames(hits[0], image_h, image_w)
        if frames.shape[0] < T:
            raise RuntimeError(f"{hits[0]}: {frames.shape[0]} frames < required {T}")
        images[cam] = frames[:T]

    with h5py.File(dst, "w") as out:
        out.attrs["sim"] = True
        out.create_dataset("action", data=actions, dtype="float32")
        obs = out.create_group("observations")
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

    return {
        "T": T,
        "T_h5": int(T_full),
        "action_shape": [T, ACTION_DIM],
        "qpos_shape": [T, QPOS_DIM],
        "qvel_shape": [T, QPOS_DIM],
        "image_shape": [T, image_h, image_w, 3],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--canonical", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--manifest-out", required=True, type=Path)
    ap.add_argument("--image_h", type=int, default=240)
    ap.add_argument("--image_w", type=int, default=320)
    args = ap.parse_args()

    canonical = json.loads(args.canonical.read_text())
    selected = sorted(canonical["selected"], key=lambda e: e["act_episode_index"])
    if len(selected) != 100:
        raise SystemExit(f"canonical manifest holds {len(selected)} rows, expected 100")

    if args.dst.exists() and any(args.dst.iterdir()):
        raise SystemExit(f"refusing to convert into a non-empty directory: {args.dst}")
    args.dst.mkdir(parents=True, exist_ok=True)

    episodes = []
    for e in selected:
        row_dir = args.run / "rows" / e["episode_id"]
        src_h5 = row_dir / "trajectory.h5"
        actual_src_hash = sha256_file(src_h5)
        if e["source_h5_sha256"] and actual_src_hash != e["source_h5_sha256"]:
            raise SystemExit(
                f"source H5 hash changed for {e['episode_id']}: "
                f"{actual_src_hash} != manifest {e['source_h5_sha256']}"
            )
        dst = args.dst / f"episode_{e['act_episode_index']}.hdf5"
        info = convert_one(src_h5, row_dir, dst, args.image_h, args.image_w)
        episodes.append(
            {
                "act_episode_index": e["act_episode_index"],
                "act_file": dst.name,
                "episode_id": e["episode_id"],
                "candidate_index": e["candidate_index"],
                "row_sha256": e["row_sha256"],
                "hazard_present": e["hazard_present"],
                "split": e["split"],
                "split_rank": e["split_rank"],
                "predeclared_stratum_rank": e["predeclared_stratum_rank"],
                "source_relpath": e["source_relpath"],
                "source_h5_sha256": actual_src_hash,
                "act_file_sha256": sha256_file(dst),
                "act_semantic_sha256": semantic_hash(dst),
                **info,
            }
        )
        print(f"  [{e['act_episode_index']:3d}] {e['episode_id'][:12]} T={info['T']:4d} -> {dst.name}")

    tree_file = hashlib.sha256()
    tree_sem = hashlib.sha256()
    for ep in episodes:
        tree_file.update(f"{ep['act_file']}\x1f{ep['act_file_sha256']}\n".encode())
        tree_sem.update(f"{ep['act_file']}\x1f{ep['act_semantic_sha256']}\n".encode())

    doc = {
        "schema": "hybrid_obstacle_act_conversion_v2",
        "canonical_manifest_sha256": canonical["manifest_sha256"],
        "source_collection_tree_sha256": canonical["source_collection_tree_sha256"],
        "converter_module": "scripts/convert_obstacle_to_act.py",
        "converter_module_sha256": sha256_file(ROOT / "scripts" / "convert_obstacle_to_act.py"),
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "image_h": args.image_h,
        "image_w": args.image_w,
        "episode_count": len(episodes),
        "hazard_present": sum(1 for e in episodes if e["hazard_present"]),
        "hazard_absent": sum(1 for e in episodes if not e["hazard_present"]),
        "train_count": sum(1 for e in episodes if e["split"] == "train"),
        "validation_count": sum(1 for e in episodes if e["split"] == "validation"),
        "min_T": min(e["T"] for e in episodes),
        "max_T": max(e["T"] for e in episodes),
        "converted_tree_file_sha256": tree_file.hexdigest(),
        "converted_tree_semantic_sha256": tree_sem.hexdigest(),
        "episodes": episodes,
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    print(f"\nepisodes            {doc['episode_count']}")
    print(f"hazard present/absent {doc['hazard_present']}/{doc['hazard_absent']}")
    print(f"train/validation      {doc['train_count']}/{doc['validation_count']}")
    print(f"T range               {doc['min_T']}..{doc['max_T']}")
    print(f"tree file sha256      {doc['converted_tree_file_sha256']}")
    print(f"tree semantic sha256  {doc['converted_tree_semantic_sha256']}")
    print(f"wrote {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
