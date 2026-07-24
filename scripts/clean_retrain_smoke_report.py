#!/usr/bin/env python3
"""Summarise the bounded four-pair ACT-only vs ACT+Safety-CVAE smoke.

Reads the paired launcher's output tree and derives the per-pair table the clean
retraining report requires: hazard presence, scene/target equivalence, first
nominal ACT command equality, task success (with and without collisions),
collision frames and geom pairs, penetration, clearance, safety activation,
raw/filtered correction magnitudes, arm deviation, return-to-nominal error and
gripper-command equality. Read-only; it never launches a rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    return json.loads(str(value).rstrip("\x00"))


def load_arm(run_dir: Path) -> dict[str, Any]:
    """Load one rollout: hybrid frame log, scene params, output paths and hashes."""
    h5_files = sorted(run_dir.rglob("trajectories_batch_*.h5"))
    if len(h5_files) != 1:
        raise RuntimeError(f"expected one trajectory H5 under {run_dir}, found {h5_files}")
    h5_path = h5_files[0]
    with h5py.File(h5_path, "r") as h5:
        names = sorted(n for n in h5 if n.startswith("traj"))
        traj = h5[names[0]]
        scene = decode(np.asarray(traj["obs_scene"]))
        log = scene.get("hybrid_safety_stack")
        if log is None:
            raise RuntimeError(f"{h5_path} has no hybrid_safety_stack log")
        collision_metrics = scene.get("collision_metrics", {})
        per_step = np.asarray(collision_metrics.get("per_step_contacts", []))
    videos = sorted(str(p) for p in run_dir.rglob("*.mp4"))
    return {
        "run_dir": str(run_dir),
        "h5_path": str(h5_path),
        "h5_sha256": sha256_file(h5_path),
        "videos": [
            {"path": v, "sha256": sha256_file(Path(v))} for v in videos
        ],
        "scene_params": scene.get("scene_params", {}),
        "object_name": scene.get("object_name"),
        "log": log,
        "frames": log["frames"],
        "metrics": log["episode_metrics"],
        "mode": log["mode"],
        "controller": log["controller"],
        "sensor_order_hash": log.get("sensor_order_hash"),
        "artifact_hashes": log.get("artifact_hashes", {}),
        "per_step_contacts": per_step.tolist() if per_step.size else [],
    }


def norms(frames: list[dict[str, Any]], key: str) -> np.ndarray:
    if not frames or key not in frames[0]:
        return np.zeros(0)
    return np.linalg.norm(
        np.asarray([f[key] for f in frames], dtype=np.float64), axis=1
    )


def summarise_pair(seed: int, act: dict[str, Any], saf: dict[str, Any]) -> dict[str, Any]:
    dt = float(saf["controller"]["dt"])
    n = min(len(act["frames"]), len(saf["frames"]))

    first_act = np.asarray(act["frames"][0]["nominal_act_action"], dtype=np.float64)
    first_saf = np.asarray(saf["frames"][0]["nominal_act_action"], dtype=np.float64)

    # Gripper is the last element of the executed 8-vector and is owned by ACT.
    grip_act = np.asarray([f["executed_action"][-1] for f in act["frames"][:n]])
    grip_saf = np.asarray([f["executed_action"][-1] for f in saf["frames"][:n]])
    grip_nom = np.asarray([f["nominal_act_action"][-1] for f in saf["frames"][:n]])

    raw = norms(saf["frames"], "raw_safety_dq")
    filt = norms(saf["frames"], "filtered_safety_dq")
    corr = norms(saf["frames"], "correction")

    def collision_frames(arm: dict[str, Any]) -> tuple[int, list[str]]:
        count = 0
        pairs: set[str] = set()
        for frame in arm["frames"]:
            geoms = frame.get("collision_geom_pairs") or []
            if geoms:
                count += 1
                for pair in geoms:
                    if isinstance(pair, dict):
                        pairs.add(f"{pair.get('geom_a')}|{pair.get('geom_b')}")
                    else:
                        pairs.add(str(pair))
        return count, sorted(pairs)

    act_cf, act_pairs = collision_frames(act)
    saf_cf, saf_pairs = collision_frames(saf)

    def arm_summary(arm: dict[str, Any], cf: int, pairs: list[str]) -> dict[str, Any]:
        m = arm["metrics"]
        return {
            "task_success": bool(m["task_success"]),
            "collision_frames": cf,
            "collision_geom_pairs": pairs,
            "collision_count_metric": m.get("collision_count"),
            "collision_free_task_success": bool(m["task_success"]) and cf == 0,
            "maximum_penetration_m": m.get("maximum_penetration_m"),
            "minimum_clearance_m": m.get("minimum_clearance_m"),
            "h5": arm["h5_path"],
            "h5_sha256": arm["h5_sha256"],
            "videos": arm["videos"],
        }

    return {
        "seed": seed,
        "hazard_present": bool(act["scene_params"].get("protrusion_present", False)),
        "equivalence": {
            "scene_params_identical": act["scene_params"] == saf["scene_params"],
            "target_object_identical": act["object_name"] == saf["object_name"],
            "target_uid": act["scene_params"].get("target_uid"),
            "first_nominal_act_command_max_abs_diff": float(
                np.abs(first_act - first_saf).max()
            ),
            "first_nominal_act_command_equal": bool(
                np.allclose(first_act, first_saf, rtol=0.0, atol=1e-6)
            ),
            "act_checkpoint_sha256_identical": act["artifact_hashes"].get(
                "act_checkpoint"
            ) == saf["artifact_hashes"].get("act_checkpoint"),
            "dataset_stats_sha256_identical": act["artifact_hashes"].get(
                "dataset_stats"
            ) == saf["artifact_hashes"].get("dataset_stats"),
            "sensor_order_hash_identical": act["sensor_order_hash"]
            == saf["sensor_order_hash"],
            "dt_identical": act["controller"]["dt"] == saf["controller"]["dt"],
        },
        "act_only": arm_summary(act, act_cf, act_pairs),
        "normal_safety": arm_summary(saf, saf_cf, saf_pairs),
        "safety_signal": {
            "first_activation_step": int(np.argmax(filt > 1e-6))
            if (filt > 1e-6).any()
            else None,
            "time_to_first_activation_s": saf["metrics"].get(
                "time_to_first_safety_activation_s"
            ),
            "max_raw_safety_dq_norm": float(raw.max()) if raw.size else 0.0,
            "integrated_raw_safety_dq_norm": float(raw.sum() * dt) if raw.size else 0.0,
            "max_filtered_safety_dq_norm": float(filt.max()) if filt.size else 0.0,
            "integrated_filtered_safety_dq_norm": float(filt.sum() * dt)
            if filt.size
            else 0.0,
            "max_arm_deviation_rad": float(
                np.abs(
                    np.asarray(
                        [f["correction"] for f in saf["frames"]], dtype=np.float64
                    )
                ).max()
            )
            if saf["frames"]
            else 0.0,
            "max_correction_norm": float(corr.max()) if corr.size else 0.0,
            "integrated_correction_norm": float(corr.sum() * dt) if corr.size else 0.0,
            "return_to_nominal_error": saf["metrics"].get("return_to_nominal_error"),
            "within_max_deviation_0_35": bool(
                np.abs(
                    np.asarray(
                        [f["correction"] for f in saf["frames"]], dtype=np.float64
                    )
                ).max()
                <= 0.35 + 1e-9
            )
            if saf["frames"]
            else True,
        },
        "gripper": {
            "gripper_command_equal_across_arms": bool(np.array_equal(grip_act, grip_saf)),
            "safety_gripper_equals_its_nominal": bool(np.array_equal(grip_saf, grip_nom)),
        },
        "compared_frames": n,
        "act_only_frames": len(act["frames"]),
        "normal_frames": len(saf["frames"]),
        "task_success_loss_from_safety": int(act["metrics"]["task_success"])
        - int(saf["metrics"]["task_success"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired_dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()

    root = Path(args.paired_dir).resolve()
    pairs = []
    for seed in args.seeds:
        act = load_arm(root / f"seed_{seed}" / "act_only")
        saf = load_arm(root / f"seed_{seed}" / "normal")
        pairs.append(summarise_pair(seed, act, saf))

    hazard = sum(p["hazard_present"] for p in pairs)
    report = {
        "schema_version": "hybrid_clean_retrain_paired_smoke_v1",
        "paired_dir": str(root),
        "seeds": args.seeds,
        "pair_count": len(pairs),
        "total_rollouts": 2 * len(pairs),
        "within_bounds": len(pairs) <= 4 and 2 * len(pairs) <= 8,
        "hazard_present_pairs": hazard,
        "hazard_absent_pairs": len(pairs) - hazard,
        "all_pairs_scene_equivalent": all(
            p["equivalence"]["scene_params_identical"]
            and p["equivalence"]["target_object_identical"]
            for p in pairs
        ),
        "all_pairs_first_command_equal": all(
            p["equivalence"]["first_nominal_act_command_equal"] for p in pairs
        ),
        "all_pairs_gripper_equal": all(
            p["gripper"]["gripper_command_equal_across_arms"] for p in pairs
        ),
        "all_pairs_within_max_deviation": all(
            p["safety_signal"]["within_max_deviation_0_35"] for p in pairs
        ),
        "act_only_successes": sum(p["act_only"]["task_success"] for p in pairs),
        "normal_safety_successes": sum(p["normal_safety"]["task_success"] for p in pairs),
        "act_only_collision_free_successes": sum(
            p["act_only"]["collision_free_task_success"] for p in pairs
        ),
        "normal_collision_free_successes": sum(
            p["normal_safety"]["collision_free_task_success"] for p in pairs
        ),
        "total_task_success_loss_from_safety": sum(
            p["task_success_loss_from_safety"] for p in pairs
        ),
        "pairs": pairs,
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
