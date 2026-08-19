#!/usr/bin/env python3
"""Step-3 gate: assert the recovery produced trainable files, not a valid config.

The v5 collection passed every config-level check it had and still produced no
proximity, no RGB and no actions.  This gate therefore opens each produced
``trajectory.h5`` and each wrist MP4 and asserts on their contents.  Nothing
downstream -- conversion, encoder transfer, training -- is authorized unless
every requested episode passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import sha256_payload  # noqa: E402
from pact_place_recovery_contract import load_recovery_contract  # noqa: E402
from run_pact_place_recovery_datagen import row_dir, write_json_atomic  # noqa: E402

from convert_obstacle_to_act import (  # noqa: E402
    _decode_action,
    _decode_qpos_qvel,
    _video_frames,
)

DEFAULT_REPORT = (
    ROOT / "diagnostics_output" / "pact_place_v5_recovery" / "keys_verified.json"
)
QPOS_DIM = 9
ACTION_DIM = 8


def _trajectory_group(handle: h5py.File) -> h5py.Group:
    keys = [key for key in handle if key.startswith("traj_")]
    if len(keys) != 1:
        raise RuntimeError(f"expected one trajectory group, found {keys}")
    return handle[keys[0]]


def _valid_action_prefix(group: h5py.Group) -> int:
    source = group["actions/joint_pos"]
    valid = np.zeros(len(source), dtype=bool)
    for step in range(len(source)):
        _, valid[step] = _decode_action(source[step])
    count = int(valid.sum())
    if count == 0:
        raise RuntimeError("no valid joint actions")
    if not np.all(valid[:count]) or np.any(valid[count:]):
        raise RuntimeError("valid joint actions are not one contiguous prefix")
    return count


def verify_episode(
    destination: Path, sensor_names: list[str], substeps: int, patch: list[int]
) -> dict[str, Any]:
    """Open the produced files and report every check, never raising past the row."""
    record: dict[str, Any] = {"row_dir": destination.name, "checks": {}}
    checks = record["checks"]
    try:
        source = destination / "trajectory.h5"
        checks["trajectory_h5_exists"] = source.is_file()
        if not source.is_file():
            raise RuntimeError("trajectory.h5 is missing")

        with h5py.File(source, "r") as handle:
            group = _trajectory_group(handle)
            checks["trajectory_h5_opens"] = True

            timesteps = _valid_action_prefix(group)
            record["timesteps"] = timesteps
            checks["actions_joint_pos_nonempty"] = timesteps > 0

            commanded = group["actions/commanded_action"]
            record["commanded_action_rows"] = int(len(commanded))
            checks["actions_commanded_nonempty"] = bool(
                len(commanded) >= timesteps
                and any(
                    _decode_action(commanded[step])[1] for step in range(timesteps)
                )
            )

            qpos = np.stack(
                [_decode_qpos_qvel(group["obs/agent/qpos"][step]) for step in range(timesteps)]
            )
            qvel = np.stack(
                [_decode_qpos_qvel(group["obs/agent/qvel"][step]) for step in range(timesteps)]
            )
            record["qpos_shape"] = list(qpos.shape)
            record["qvel_shape"] = list(qvel.shape)
            checks["qpos_qvel_present_and_aligned"] = bool(
                qpos.shape == qvel.shape == (timesteps, QPOS_DIM)
            )

            proximity = group["obs/proximity"]
            observed = sorted(proximity.keys())
            record["n_proximity_sensors"] = len(observed)
            checks["proximity_sensor_set_exact"] = observed == sorted(sensor_names)
            expected_shape = (timesteps, substeps, patch[0], patch[1])
            shapes = {
                name: tuple(np.asarray(proximity[name].shape).tolist())
                for name in sensor_names
                if name in proximity
            }
            bad_shape = {
                name: list(shape)
                for name, shape in shapes.items()
                if shape[1:] != expected_shape[1:] or shape[0] < timesteps
            }
            bad_dtype = {
                name: str(proximity[name].dtype)
                for name in shapes
                if proximity[name].dtype != np.float32
            }
            record["proximity_expected_shape"] = list(expected_shape)
            record["proximity_bad_shape"] = bad_shape
            record["proximity_bad_dtype"] = bad_dtype
            checks["proximity_shapes_and_dtypes"] = not (bad_shape or bad_dtype)

            # An all-zero skin would pass every shape check and teach nothing.
            first = sensor_names[0]
            sample = np.asarray(proximity[first][:timesteps], dtype=np.float32)
            record["proximity_first_sensor_finite_fraction"] = float(
                np.isfinite(sample).mean()
            )
            checks["proximity_not_degenerate"] = bool(
                np.isfinite(sample).all() and float(np.ptp(sample)) > 0.0
            )

            for name in ("sensor_param", "sensor_data"):
                checks[f"obs_{name}_present"] = name in group["obs"]
            extrinsic = group[f"obs/sensor_param/{first}/extrinsic_cv"]
            intrinsic = group[f"obs/sensor_param/{first}/intrinsic_cv"]
            checks["sensor_param_shapes"] = bool(
                extrinsic.shape[0] >= timesteps
                and tuple(extrinsic.shape[1:]) == (3, 4)
                and intrinsic.shape[0] >= timesteps
                and tuple(intrinsic.shape[1:]) == (3, 3)
            )

        videos = [
            path
            for path in sorted(destination.glob("episode_*_wrist_camera.mp4"))
            if "_depth" not in path.name
        ]
        record["wrist_video"] = videos[0].name if len(videos) == 1 else None
        checks["wrist_video_unique"] = len(videos) == 1
        if len(videos) == 1:
            frames = _video_frames(videos[0], None, None)
            record["wrist_video_frames"] = int(frames.shape[0])
            record["wrist_video_frame_shape"] = list(frames.shape[1:])
            checks["wrist_video_decodes"] = frames.shape[0] > 0
            checks["wrist_video_covers_timesteps"] = bool(frames.shape[0] >= timesteps)
    except Exception as error:  # noqa: BLE001 - one row must not hide the rest
        record["error"] = f"{type(error).__name__}: {error}"
    record["passed"] = bool(record.get("checks")) and all(checks.values()) and (
        "error" not in record
    )
    record["failed_checks"] = sorted(
        name for name, value in checks.items() if not value
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    contract = load_recovery_contract(args.config)
    recovery = contract["recovery"]
    output_root = args.output_root or (ROOT / recovery["output_root"])
    sensor_names = list(recovery["proximity_sensor_names"])
    substeps = int(recovery["proximity_substeps"])
    patch = list(recovery["proximity_patch"])

    records: list[dict[str, Any]] = []
    for row in contract["recovery_rows"]:
        destination = row_dir(output_root, row)
        record = verify_episode(destination, sensor_names, substeps, patch)
        record["role_index"] = row["role_index"]
        record["episode_id"] = row["episode_id"]
        records.append(record)
        print(
            f"row={row['role_index']:03d} passed={record['passed']} "
            f"T={record.get('timesteps')} "
            f"frames={record.get('wrist_video_frames')} "
            f"{','.join(record['failed_checks']) or ''}"
            f"{' ' + record['error'] if 'error' in record else ''}",
            flush=True,
        )

    failed = [item for item in records if not item["passed"]]
    tally: dict[str, int] = {}
    for item in records:
        for name, value in item["checks"].items():
            tally[name] = tally.get(name, 0) + int(bool(value))

    report: dict[str, Any] = {
        "schema_version": "pact_place_v5_recovery_keys_v1",
        "config_sha256": contract["config_sha256"],
        "output_root": str(output_root),
        "n_requested": len(contract["recovery_rows"]),
        "n_verified": len(records),
        "n_passed": len(records) - len(failed),
        "n_failed": len(failed),
        "check_pass_counts": dict(sorted(tally.items())),
        "timesteps_total": sum(item.get("timesteps", 0) for item in records),
        "failed_rows": [
            {
                "role_index": item["role_index"],
                "episode_id": item["episode_id"],
                "failed_checks": item["failed_checks"],
                "error": item.get("error"),
            }
            for item in failed
        ],
        "all_passed": not failed,
        "conversion_authorized": not failed
        and len(records) == len(contract["recovery_rows"]),
        "training_authorized": False,
        "next_action": (
            "convert_to_act_format"
            if not failed
            else "stop_do_not_convert_a_partial_set"
        ),
        "rows": records,
    }
    report["keys_verified_sha256"] = sha256_payload(report)
    write_json_atomic(args.report, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "n_requested",
                    "n_passed",
                    "n_failed",
                    "all_passed",
                    "conversion_authorized",
                    "next_action",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(str(args.report))
    if failed:
        print("PACT PLACE RECOVERY KEY VERIFICATION FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
