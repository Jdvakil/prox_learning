#!/usr/bin/env python3
"""Frozen recovery contract for the 152 kept PACT place-corridor v5 rows.

The v5 collection ran the feasibility-screen harness, whose ``run_row``
truncates the sensor suite to ``qpos``/``tcp_pose``.  The 152 kept rows are a
valid screen record but carry no proximity, no RGB and no action arrays, so
none of them is a trainable demonstration.

This contract freezes the re-record: the same 152 rows, the same seeds, the
same expert, scene and clean-success filter, run through the datagen pipeline
with the full sensor suite so ``trajectory.h5`` and the wrist MP4 are written.
It selects nothing.  Each row carries the outcome the original screen recorded
so the re-record can be checked against it row by row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pact_place_corridor_contract import ROOT, sha256_file, sha256_payload
from pact_place_collection_contract import load_collection_contract

COLLECTION_CONFIG = ROOT / "configs" / "pact_place_corridor_v5_collection.json"
COLLECTION_OUTPUT = ROOT / "diagnostics_output" / "pact_place_corridor_v5_collection"
COLLISION_MANIFEST = ROOT / "configs" / "pact_collision_candidate_manifest_v2.json"
RECOVERY_CONFIG = ROOT / "configs" / "pact_place_v5_recovery.json"
RECOVERY_OUTPUT_ROOT = "assets/datagen/pact_place_corridor_v2/recovered_152"

N_RECOVERY_ROWS = 152
N_PROXIMITY_SENSORS = 40
PROXIMITY_SUBSTEPS = 4
PROXIMITY_PATCH = [8, 8]

# The fields every re-recorded row must reproduce from its screen row.
REPRODUCED_KEYS = (
    "task_success",
    "clean_success",
    "terminal_policy_phase",
    "terminal_action_index",
)

# Datasets the Step-3 gate asserts on the produced file, not on the config.
REQUIRED_H5_KEYS = (
    "traj_0/actions/joint_pos",
    "traj_0/actions/commanded_action",
    "traj_0/obs/agent/qpos",
    "traj_0/obs/agent/qvel",
)


def _screen_row_dir(role_index: int, episode_id: str) -> Path:
    return (
        COLLECTION_OUTPUT
        / "expert_screen_rows"
        / f"{role_index:02d}_{episode_id[:16]}"
    )


def _source_hashes() -> dict[str, str]:
    paths = [
        "scripts/pact_place_recovery_contract.py",
        "scripts/run_pact_place_recovery_datagen.py",
        "scripts/verify_pact_place_recovery_keys.py",
        "scripts/run_pact_place_expert_screen.py",
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v2.xml",
        "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
        "submodules/molmospaces/molmo_spaces/tasks/pact_place_contact_audit.py",
        "submodules/molmospaces/molmo_spaces/configs/camera_configs.py",
    ]
    return {
        relative: sha256_file(ROOT / relative)
        for relative in paths
        if (ROOT / relative).is_file()
    }


def build_recovery_contract() -> dict[str, Any]:
    collection = load_collection_contract(COLLECTION_CONFIG)
    summary = json.loads((COLLECTION_OUTPUT / "collection.json").read_text())

    payload = dict(summary)
    observed = payload.pop("collection_sha256")
    if observed != sha256_payload(payload):
        raise ValueError("v5 collection summary self-hash mismatch")

    kept = list(summary["kept_role_indices"])
    if len(kept) != N_RECOVERY_ROWS:
        raise ValueError(f"expected {N_RECOVERY_ROWS} kept rows, found {len(kept)}")
    if len(set(kept)) != len(kept):
        raise ValueError("kept role indices are not unique")

    by_index = {row["role_index"]: row for row in collection["collection_rows"]}
    sensor_names = list(json.loads(COLLISION_MANIFEST.read_text())["sensor_names"])
    if len(sensor_names) != N_PROXIMITY_SENSORS:
        raise ValueError(f"expected {N_PROXIMITY_SENSORS} proximity sensors")

    rows: list[dict[str, Any]] = []
    for role_index in sorted(kept):
        source = by_index.get(role_index)
        if source is None:
            raise ValueError(f"kept role index {role_index} is not in the frozen config")
        screen = json.loads(
            (_screen_row_dir(role_index, source["episode_id"]) / "result.json").read_text()
        )
        if screen["episode_id"] != source["episode_id"]:
            raise ValueError(f"screen row {role_index} episode ID does not match")
        if screen["row_sha256"] != source["row_sha256"]:
            raise ValueError(f"screen row {role_index} row hash does not match")
        if screen.get("clean_success") is not True:
            raise ValueError(f"kept row {role_index} is not a clean success")
        row = dict(source)
        row["screen_result_sha256"] = screen["result_sha256"]
        row["screen_selected_seed"] = dict(screen["selected_seed"])
        row["screen_retry_index"] = len(screen["retry_history"])
        row["screen_episode_steps"] = int(screen["episode_steps"])
        row["expected"] = {key: screen[key] for key in REPRODUCED_KEYS}
        rows.append(row)

    sides = [row["intrusion_side"] for row in rows]
    recorded_sides = summary["limitation_kept_vs_discarded_side_counts"]
    if (sides.count("left"), sides.count("right")) != (
        recorded_sides["kept_left"],
        recorded_sides["kept_right"],
    ):
        raise ValueError("recovered side balance does not match the kept record")

    document: dict[str, Any] = {
        "schema_version": "pact_place_v5_recovery_v1",
        "status": "recovery_preregistered",
        "created_utc": "2026-08-19T00:00:00Z",
        "purpose": (
            "re-record the 152 kept v5 rows through the datagen pipeline with the "
            "full sensor suite; this re-records the kept set, it does not re-select it"
        ),
        "source_collection_config_sha256": collection["config_sha256"],
        "source_collection_sha256": summary["collection_sha256"],
        "screen_config_sha256": collection["screen_config_sha256"],
        "scene": dict(collection["scene"]),
        "success_criterion": dict(collection["success_criterion"]),
        "expert": dict(collection["expert"]),
        "recovery": {
            "n_rows": len(rows),
            "output_root": RECOVERY_OUTPUT_ROOT,
            "sensor_suite": "full",
            "sensor_suite_truncation_forbidden": True,
            "reuse_of_screen_run_row_forbidden": True,
            "save_videos_required": True,
            "proximity_sensor_names": sensor_names,
            "n_proximity_sensors": N_PROXIMITY_SENSORS,
            "proximity_substeps": PROXIMITY_SUBSTEPS,
            "proximity_patch": PROXIMITY_PATCH,
            "required_h5_keys": list(REQUIRED_H5_KEYS),
            "reproduced_keys": list(REPRODUCED_KEYS),
            "divergence_is_a_stop_condition": True,
            "selection_unchanged": True,
            "training_not_authorized": True,
            "conversion_not_authorized_before_key_verification": True,
        },
        "recovery_rows": rows,
        "source_sha256": _source_hashes(),
        "protected_artifact_sha256_before": dict(
            collection["protected_artifact_sha256_before"]
        ),
    }
    document["config_sha256"] = sha256_payload(document)
    validate_recovery_contract(document)
    return document


def validate_recovery_contract(document: dict[str, Any]) -> None:
    payload = dict(document)
    observed = payload.pop("config_sha256")
    if observed != sha256_payload(payload):
        raise ValueError("recovery config self-hash mismatch")
    rows = document["recovery_rows"]
    if len(rows) != N_RECOVERY_ROWS:
        raise ValueError(f"expected {N_RECOVERY_ROWS} recovery rows")
    if len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("recovery episode IDs are not unique")
    for row in rows:
        payload = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "row_sha256",
                "screen_result_sha256",
                "screen_selected_seed",
                "screen_retry_index",
                "screen_episode_steps",
                "expected",
            }
        }
        if row["row_sha256"] != sha256_payload(payload):
            raise ValueError(f"row self-hash mismatch at {row['role_index']}")
        missing = [key for key in REPRODUCED_KEYS if key not in row["expected"]]
        if missing:
            raise ValueError(f"row {row['role_index']} expected block missing {missing}")
        if row["expected"]["clean_success"] is not True:
            raise ValueError(f"row {row['role_index']} is not a kept clean success")
    recovery = document["recovery"]
    if recovery["sensor_suite"] != "full" or not recovery["save_videos_required"]:
        raise ValueError("recovery must run the full sensor suite and write videos")
    if len(recovery["proximity_sensor_names"]) != N_PROXIMITY_SENSORS:
        raise ValueError("recovery must declare all 40 proximity sensors")
    if not Path(recovery["output_root"]).parts[:2] == ("assets", "datagen"):
        raise ValueError("recovery output must live under assets/datagen")


def load_recovery_contract(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    validate_recovery_contract(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RECOVERY_CONFIG)
    args = parser.parse_args()
    if args.output.resolve() == RECOVERY_CONFIG.resolve() and args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen recovery contract {args.output}")
    document = build_recovery_contract()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(document["config_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
