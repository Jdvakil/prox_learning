#!/usr/bin/env python3
"""Provenance capture for the hybrid-obstacle clean ACT retraining run.

Read-only auditing helpers: this script records repository/environment state and
hashes existing files. It never trains, mutates a checkpoint, or edits a dataset.

Subcommands
-----------
starting-state         Repository, submodule, environment and hardware capture.
source-manifest        Hash every file of a datagen run and emit a content-tree hash.
converted-manifest     Hash every converted ACT episode and record its shapes.
conversion-provenance  One-to-one source-trajectory -> converted-episode mapping,
                       confirmed by re-decoding the source rows.
training-manifest      Training config, artifact hashes and per-epoch metrics.
tree-hash              Content-tree SHA-256 of an arbitrary directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=300, check=False
        )
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"<error: {exc}>"


def git_state(repo: Path) -> dict[str, Any]:
    return {
        "path": str(repo),
        "commit": run(["git", "rev-parse", "HEAD"], repo),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo),
        "status_porcelain": run(["git", "status", "--porcelain"], repo),
        "clean": run(["git", "status", "--porcelain"], repo) == "",
    }


def file_tree(root: Path) -> tuple[list[dict[str, Any]], str, int]:
    """Return (sorted per-file records, content-tree SHA-256, total bytes).

    The content-tree hash covers the sorted ``relpath\\0sha256\\n`` lines, so it is
    stable against traversal order and independent of mtimes/ownership.
    """
    records: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        records.append(
            {"relpath": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    payload = "".join(f"{r['relpath']}\0{r['sha256']}\n" for r in records)
    tree = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return records, tree, sum(r["bytes"] for r in records)


def cmd_starting_state(args: argparse.Namespace) -> dict[str, Any]:
    wt = Path(args.worktree).resolve()
    state: dict[str, Any] = {
        "schema_version": "hybrid_clean_retrain_starting_state_v1",
        "utc": run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        "repositories": {
            "root": git_state(wt),
            "act": git_state(wt / "submodules/act"),
            "molmospaces": git_state(wt / "submodules/molmospaces"),
        },
        "gitlinks": {},
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
    }

    for line in run(["git", "submodule", "status"], wt).splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            state["gitlinks"][parts[1]] = parts[0].lstrip("+-U")

    # Hardware / driver identity.
    state["gpu"] = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,uuid",
            "--format=csv,noheader",
        ]
    )
    state["memory"] = run(["free", "-b"]).splitlines()[:2]
    usage = shutil.disk_usage("/")
    state["disk_root"] = {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }

    # Package versions actually importable in this interpreter.
    packages: dict[str, str] = {}
    for name in (
        "numpy",
        "torch",
        "torchvision",
        "mujoco",
        "h5py",
        "cv2",
        "einops",
        "matplotlib",
        "scipy",
        "trimesh",
        "imageio",
        "wandb",
        "jax",
    ):
        try:
            mod = __import__(name)
            packages[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:
            packages[name] = f"<unavailable: {type(exc).__name__}>"
    state["packages"] = packages

    try:
        import torch

        state["torch_cuda"] = {
            "version": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        state["torch_cuda"] = f"<unavailable: {type(exc).__name__}>"

    state["env"] = {
        k: os.environ.get(k)
        for k in (
            "MLSPACES_ASSETS_DIR",
            "PYTHONPATH",
            "MUJOCO_GL",
            "PYOPENGL_PLATFORM",
            "PYTHONHASHSEED",
            "OMP_NUM_THREADS",
            "TZ",
            "WANDB_MODE",
        )
    }

    # Canonical frozen hashes declared by the approved manifest.
    canonical = {
        "hybrid_contract": (
            "configs/hybrid_safety_stack_v1.json",
            "aef29d762a909d0ce8610b79b4cb9a89a85c08fbe4d014910f27af44fc90df2b",
        ),
        "safety_cvae_model": (
            "assets/safety/cvae_v3/model.pt",
            "1fb2fc2b6023e64d2b9cbcf67fd5a24402968ec6f902c1e8a8595690396e7405",
        ),
        "safety_cvae_meta": (
            "assets/safety/cvae_v3/meta.json",
            "7c873756fb16e4f3fc565f96c41a581816981ab93886f1db5d27e62110a5fc81",
        ),
        "live_adapter": (
            "submodules/act/eval_act_obstacle_safety.py",
            "21e8ccbe489cd278e9e946fde4d72a6772de5394aff47d484c882b8699c292ee",
        ),
        "residual_controller": (
            "submodules/act/hybrid_safety_residual.py",
            "655a2e926351eef59c44896eb2cd6b142bfbf6fd5444e26c133c200443eaeeca",
        ),
        "paired_launcher": (
            "submodules/act/run_paired_hybrid_safety_eval.py",
            "4623ce5fffc7f9a136ce9b96f7e989942d9190db5074084a5172692a424e3fc2",
        ),
        "converter": (
            "scripts/convert_obstacle_to_act.py",
            "74b60458754b782393d65d508174b4168a94dbe6c539ff5d2005076994856695",
        ),
        "act_utils": (
            "submodules/act/utils.py",
            "494ea056bd716fbb939d8d52a4c34edfd1b36c9116869acf1093dbc3060bbf84",
        ),
        "molmospaces_config": (
            ("submodules/molmospaces/molmo_spaces/data_generation/config/"
             "object_manipulation_datagen_configs.py"),
            "cd8891e005ee22e7fd631b5c1077b9e38f1cef72678d4b9dece2a8eeb3b8ff3f",
        ),
        "model_hybrid_xml": (
            "assets/robots/franka_skin/model_hybrid.xml",
            "50924661e0411f92ab529c790512b17b674e789434c592c3dbc6d2359164d4c6",
        ),
    }
    checks = {}
    for name, (rel, expected) in canonical.items():
        path = wt / rel
        actual = sha256_file(path) if path.is_file() else None
        checks[name] = {
            "path": rel,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
        }
    # The sensor-order hash is a content hash over the canonical JSON name list.
    contract = json.loads((wt / "configs/hybrid_safety_stack_v1.json").read_text())
    names = contract["sensor_contract"]["ordered_names"]
    payload = json.dumps(list(names), separators=(",", ":"), ensure_ascii=True).encode("ascii")
    order_hash = hashlib.sha256(payload).hexdigest()
    checks["sensor_order"] = {
        "path": "configs/hybrid_safety_stack_v1.json:sensor_contract.ordered_names",
        "expected_sha256": "c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858",
        "actual_sha256": order_hash,
        "match": order_hash
        == "c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858",
        "sensor_count": len(names),
    }
    state["canonical_hash_checks"] = checks
    state["all_canonical_hashes_match"] = all(c["match"] for c in checks.values())
    return state


def cmd_source_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.run_dir).resolve()
    records, tree, total = file_tree(root)
    by_ext: dict[str, dict[str, int]] = {}
    for rec in records:
        ext = Path(rec["relpath"]).suffix or "<none>"
        slot = by_ext.setdefault(ext, {"count": 0, "bytes": 0})
        slot["count"] += 1
        slot["bytes"] += rec["bytes"]
    return {
        "schema_version": "hybrid_clean_retrain_source_manifest_v1",
        "run_dir": str(root),
        "collection_id": root.name,
        "file_count": len(records),
        "total_bytes": total,
        "content_tree_sha256": tree,
        "by_extension": by_ext,
        "files": records,
    }


def cmd_conversion_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Rebuild the converter's selection and *prove* each converted episode's source.

    The committed converter walks ``sorted(house_*/trajectories*.h5)`` and then
    numeric ``traj_<i>`` order, dropping ``fail[-1]`` episodes and stopping at
    ``--max_episodes``. Replaying that order gives a candidate mapping; each pair is
    then confirmed by re-decoding the source rows and requiring the converted
    ``qpos``/``action`` arrays to match byte for byte. A mapping that cannot be
    confirmed this way is reported as unverified rather than assumed.
    """
    import h5py

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from convert_obstacle_to_act import (
        _decode_action,
        _decode_qpos_qvel,
        _episode_failed,
        _find_h5_files,
    )

    src = Path(args.run_dir).resolve()
    dst = Path(args.dataset_dir).resolve()
    limit = args.max_episodes

    episodes: list[dict[str, Any]] = []
    skipped_failed: list[dict[str, Any]] = []
    global_idx = 0
    for h5_path in _find_h5_files(src):
        if global_idx >= limit:
            break
        house = h5_path.parent.name
        h5_hash = sha256_file(h5_path)
        with h5py.File(h5_path, "r") as f:
            for traj_key in sorted(f.keys(), key=lambda k: int(k.split("_", 1)[1])):
                if global_idx >= limit:
                    break
                grp = f[traj_key]
                ep_idx = int(traj_key.split("_", 1)[1])
                if _episode_failed(grp):
                    skipped_failed.append({"house": house, "traj": traj_key})
                    continue

                scene = json.loads(np.asarray(grp["obs_scene"]).item())
                params = scene.get("scene_params", {})
                collision = scene.get("collision_metrics", {})

                t_full = grp["actions/joint_pos"].shape[0]
                decoded_actions = []
                for t in range(t_full):
                    vec, ok = _decode_action(grp["actions/joint_pos"][t])
                    if ok:
                        decoded_actions.append(vec)
                t_valid = len(decoded_actions)
                src_actions = np.stack(decoded_actions) if t_valid else np.zeros((0, 8))
                src_qpos = np.stack(
                    [_decode_qpos_qvel(grp["obs/agent/qpos"][t]) for t in range(t_valid)]
                )

                videos = {}
                for cam in ("exo_camera_1", "wrist_camera"):
                    mp4 = h5_path.parent / f"episode_{ep_idx:08d}_{cam}_batch_1_of_1.mp4"
                    videos[cam] = {
                        "path": str(mp4),
                        "exists": mp4.is_file(),
                        "sha256": sha256_file(mp4) if mp4.is_file() else None,
                    }

                converted = dst / f"episode_{global_idx}.hdf5"
                record: dict[str, Any] = {
                    "converted_episode": global_idx,
                    "converted_path": str(converted),
                    "converted_sha256": sha256_file(converted) if converted.is_file() else None,
                    "source_house": house,
                    "source_h5": str(h5_path),
                    "source_h5_sha256": h5_hash,
                    "source_traj_key": traj_key,
                    "source_trajectory_id": f"{house}/{traj_key}",
                    "source_videos": videos,
                    "hazard_present": bool(params.get("protrusion_present", False)),
                    "target_uid": params.get("target_uid"),
                    "behavior_class": scene.get("behavior_class"),
                    "collided": bool(np.asarray(collision.get("collided", False)).item())
                    if "collided" in collision
                    else None,
                    "n_collision_steps": int(np.asarray(collision.get("n_collision_steps", 0)).item())
                    if "n_collision_steps" in collision
                    else None,
                    "total_contacts": int(np.asarray(collision.get("total_contacts", 0)).item())
                    if "total_contacts" in collision
                    else None,
                    "source_frames_full": int(t_full),
                    "source_frames_valid": int(t_valid),
                    "conversion_length_rule": "T = T_h5 - 1 (trailing empty {} action row dropped)",
                    "length_rule_holds": t_valid == t_full - 1,
                }

                if converted.is_file():
                    with h5py.File(converted, "r") as cf:
                        conv_action = cf["/action"][()]
                        conv_qpos = cf["/observations/qpos"][()]
                        cams = sorted(cf["/observations/images"].keys())
                        record["converted_frames"] = int(conv_action.shape[0])
                        record["converted_cameras"] = cams
                        record["converted_camera_shapes"] = {
                            c: list(cf[f"/observations/images/{c}"].shape) for c in cams
                        }
                        record["contains_proximity"] = any(
                            "prox" in k.lower() or "skin" in k.lower()
                            for k in _walk_keys(cf)
                        )
                    n = min(len(conv_action), len(src_actions))
                    record["action_bytes_identical"] = bool(
                        conv_action.shape == src_actions.shape
                        and np.array_equal(conv_action, src_actions.astype(np.float32))
                    )
                    record["qpos_bytes_identical"] = bool(
                        conv_qpos.shape == src_qpos.shape
                        and np.array_equal(conv_qpos, src_qpos.astype(np.float32))
                    )
                    record["frames_agree"] = record["converted_frames"] == t_valid
                    record["provenance_verified"] = bool(
                        record["action_bytes_identical"]
                        and record["qpos_bytes_identical"]
                        and record["frames_agree"]
                        and n > 0
                    )
                else:
                    record["provenance_verified"] = False

                episodes.append(record)
                global_idx += 1

    hazard = sum(1 for e in episodes if e["hazard_present"])
    uids = sorted({e["target_uid"] for e in episodes})
    houses: dict[str, int] = {}
    for e in episodes:
        houses[e["source_house"]] = houses.get(e["source_house"], 0) + 1
    return {
        "schema_version": "hybrid_clean_retrain_conversion_provenance_v1",
        "source_run_dir": str(src),
        "converted_dataset_dir": str(dst),
        "converted_episode_count": len(episodes),
        "skipped_failed_source_trajectories": len(skipped_failed),
        "hazard_present_count": hazard,
        "hazard_present_fraction": hazard / len(episodes) if episodes else 0.0,
        "distinct_target_uids": uids,
        "single_canonical_target": uids == ["4afa0cdde045417ab31f98ae7745b039"],
        "house_distribution": houses,
        "all_provenance_verified": all(e["provenance_verified"] for e in episodes),
        "all_length_rule_holds": all(e["length_rule_holds"] for e in episodes),
        "any_converted_contains_proximity": any(
            e.get("contains_proximity", False) for e in episodes
        ),
        "duplicate_converted_hashes": len(
            {e["converted_sha256"] for e in episodes}
        )
        != len(episodes),
        "duplicate_source_trajectory_ids": len(
            {e["source_trajectory_id"] for e in episodes}
        )
        != len(episodes),
        "episodes": episodes,
    }


def cmd_converted_manifest(args: argparse.Namespace) -> dict[str, Any]:
    import h5py

    root = Path(args.dataset_dir).resolve()
    episodes = []
    for path in sorted(
        root.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[1])
    ):
        with h5py.File(path, "r") as f:
            cams = sorted(f["/observations/images"].keys())
            episodes.append(
                {
                    "episode": int(path.stem.split("_")[1]),
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "sim_attr": bool(f.attrs.get("sim", False)),
                    "frames": int(f["/action"].shape[0]),
                    "action_shape": list(f["/action"].shape),
                    "qpos_shape": list(f["/observations/qpos"].shape),
                    "qvel_shape": list(f["/observations/qvel"].shape),
                    "cameras": cams,
                    "camera_shapes": {
                        c: list(f[f"/observations/images/{c}"].shape) for c in cams
                    },
                    "has_proximity": any(
                        "prox" in k.lower() or "skin" in k.lower() or "sensor" in k.lower()
                        for k in _walk_keys(f)
                    ),
                }
            )
    records, tree, total = file_tree(root)
    return {
        "schema_version": "hybrid_clean_retrain_converted_manifest_v1",
        "dataset_dir": str(root),
        "episode_count": len(episodes),
        "total_bytes": total,
        "content_tree_sha256": tree,
        "episodes": episodes,
        "files": records,
    }


def _walk_keys(node, prefix: str = "") -> list[str]:
    import h5py

    keys: list[str] = []
    for name, item in node.items():
        full = f"{prefix}/{name}"
        keys.append(full)
        if isinstance(item, h5py.Group):
            keys.extend(_walk_keys(item, full))
    return keys


def cmd_training_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble the training manifest: config, hashes and per-epoch metrics.

    Per-epoch metrics are parsed out of the trainer's own stdout, which is the only
    place the committed trainer emits them.
    """
    import re

    run_dir = Path(args.run_dir).resolve()
    log_text = Path(args.training_log).read_text(errors="replace") if args.training_log else ""

    epochs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending: str | None = None
    for line in log_text.splitlines():
        line = line.strip()
        m = re.match(r"^Epoch (\d+)$", line)
        if m:
            if current is not None:
                epochs.append(current)
            current = {"epoch": int(m.group(1))}
            pending = None
            continue
        if current is None:
            continue
        m = re.match(r"^Val loss:\s+([0-9.eE+-]+)$", line)
        if m:
            current["val_loss"] = float(m.group(1))
            pending = "val"
            continue
        m = re.match(r"^Train loss:\s+([0-9.eE+-]+)$", line)
        if m:
            current["train_loss"] = float(m.group(1))
            pending = "train"
            continue
        if pending and re.match(r"^(l1|kl|loss):", line):
            parts = {
                k: float(v)
                for k, v in re.findall(r"([a-z0-9_]+): ([0-9.eE+-]+)", line)
            }
            for key, value in parts.items():
                current[f"{pending}_{key}"] = value
            pending = None
    if current is not None:
        epochs.append(current)

    best_line = re.search(
        r"Best ckpt, val loss ([0-9.eE+-]+) @ epoch(\d+)", log_text
    )
    finished = re.search(
        r"Training finished:\s*\nSeed (\d+), val loss ([0-9.eE+-]+) at epoch (\d+)", log_text
    )

    artifacts = {}
    for name in sorted(p.name for p in run_dir.glob("*") if p.is_file()):
        path = run_dir / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    return {
        "schema_version": "hybrid_clean_retrain_training_manifest_v1",
        "run_dir": str(run_dir),
        "command": Path(args.command_file).read_text().strip() if args.command_file else None,
        "declared_config": {
            "task_name": "obstacle_baseline",
            "policy_class": "ACT",
            "seed": 0,
            "num_epochs": 2000,
            "batch_size": 8,
            "lr": 1e-5,
            "chunk_size": 100,
            "hidden_dim": 512,
            "dim_feedforward": 3200,
            "kl_weight": 10,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": ["exo_camera_1", "wrist_camera"],
            "image_size": [240, 320],
            "qpos_dim": 9,
            "action_dim": 8,
            "proximity_inputs": 0,
        },
        "epoch_count_logged": len(epochs),
        "epochs": epochs,
        "best_epoch": int(best_line.group(2)) if best_line else None,
        "best_val_loss": float(best_line.group(1)) if best_line else None,
        "selection_rule": "minimum mean validation total loss over the 20 held-out "
        "trajectories, evaluated before each epoch's optimizer steps "
        "(imitate_episodes.train_bc: epoch_val_loss < min_val_loss)",
        "training_finished_line": finished.group(0) if finished else None,
        "artifacts": artifacts,
    }


def cmd_tree_hash(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.directory).resolve()
    records, tree, total = file_tree(root)
    return {
        "schema_version": "hybrid_clean_retrain_tree_hash_v1",
        "directory": str(root),
        "file_count": len(records),
        "total_bytes": total,
        "content_tree_sha256": tree,
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("starting-state")
    p.add_argument("--worktree", default=str(REPO_ROOT))
    p.set_defaults(func=cmd_starting_state)

    p = sub.add_parser("source-manifest")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_source_manifest)

    p = sub.add_parser("converted-manifest")
    p.add_argument("dataset_dir")
    p.set_defaults(func=cmd_converted_manifest)

    p = sub.add_parser("conversion-provenance")
    p.add_argument("run_dir")
    p.add_argument("dataset_dir")
    p.add_argument("--max_episodes", type=int, default=100)
    p.set_defaults(func=cmd_conversion_provenance)

    p = sub.add_parser("training-manifest")
    p.add_argument("run_dir")
    p.add_argument("--training_log")
    p.add_argument("--command_file")
    p.set_defaults(func=cmd_training_manifest)

    p = sub.add_parser("tree-hash")
    p.add_argument("directory")
    p.set_defaults(func=cmd_tree_hash)

    args = parser.parse_args()
    json.dump(args.func(args), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
