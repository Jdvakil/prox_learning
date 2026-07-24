#!/usr/bin/env python3
"""Smoke-test lossless JSON persistence of the hybrid per-frame policy log."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import h5py
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MOLMO_ROOT = REPO_ROOT / "submodules/molmospaces"
if str(MOLMO_ROOT) not in sys.path:
    sys.path.insert(0, str(MOLMO_ROOT))

from molmo_spaces.utils.save_utils import save_trajectories  # noqa: E402


def representative_stack_log() -> dict:
    contract = json.loads(
        (REPO_ROOT / "configs/hybrid_safety_stack_v1.json").read_text()
    )
    frame = {
        "step": 0,
        "episode_seed": 17,
        "safety_mode": "normal",
        "nominal_act_action": [0.0] * 8,
        "raw_safety_dq": [0.0] * 7,
        "baseline_safety_output": [0.0] * 7,
        "subtracted_dq": [0.0] * 7,
        "filtered_safety_dq": [0.0] * 7,
        "correction": [0.0] * 7,
        "executed_action": [0.0] * 8,
        "active_sensors": [],
        "active_links": [],
        "minimum_environment_distance_m": 1.0,
        "collision_geom_pairs": [],
        "task_phase": "act_live",
        "task_success": False,
    }
    return {
        "schema_version": "hybrid_safety_frame_log_v1",
        "sensor_order": contract["sensor_contract"]["ordered_names"],
        "sensor_order_hash": contract["sensor_contract"]["sensor_order_hash"],
        "frames": [frame],
    }


def main() -> None:
    stack = representative_stack_log()
    episode = {
        "qpos": torch.zeros((1, 9)),
        "qvel": torch.zeros((1, 9)),
        "actions/joint_pos": torch.zeros((1, 8)),
        "terminateds": torch.tensor([True]),
        "truncateds": torch.tensor([False]),
        "successes": torch.tensor([False]),
        "rewards": torch.tensor([0.0]),
        "obs_scene": json.dumps({"hybrid_safety_stack": stack}),
    }
    with tempfile.TemporaryDirectory(prefix="hybrid-safety-storage-") as directory:
        output = save_trajectories(
            [episode],
            directory,
            fps=15.0,
            save_mp4s=False,
        )
        with h5py.File(output, "r") as h5:
            raw = h5["traj_0/obs_scene"][()]
            if isinstance(raw, bytes):
                raw = raw.decode()
            loaded = json.loads(raw)["hybrid_safety_stack"]
        if loaded != stack:
            raise SystemExit("Hybrid safety log changed during H5 persistence")
        print(
            json.dumps(
                {
                    "result": "pass",
                    "sensor_count": len(loaded["sensor_order"]),
                    "frame_count": len(loaded["frames"]),
                    "sensor_order_hash": loaded["sensor_order_hash"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
