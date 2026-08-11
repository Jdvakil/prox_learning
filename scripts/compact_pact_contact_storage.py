#!/usr/bin/env python3
"""Aggressively compact contact-endpoint rows under a frozen storage rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXCLUDED_SCHEDULE_INDICES = {0, 1199}
CONTACT_CLASSES = ("grasp_target", "hazard_bar", "other_environment")
CORE_RESULT_KEYS = (
    "schema_version",
    "status",
    "arm",
    "rollout_id",
    "schedule_row_sha256",
    "episode_id",
    "candidate_index",
    "row_sha256",
    "manifest_sha256",
    "intrusion_side",
    "sampling_retry_index",
    "sampling_retry_history",
    "seed",
    "checkpoint_seed",
    "checkpoint_sha256",
    "stats_sha256",
    "surface_encoder_sha256",
    "blur_sigma",
    "attempt_index",
    "inflight_recovery_event_sha256",
    "abandoned_payload_archive",
    "initial_observation_accepted",
    "initial_observation_boundary_sha256",
    "task_success",
    "collision_free_task_success",
    "failure_taxonomy",
)
POLICY_SUMMARY_KEYS = (
    "arm",
    "control_steps",
    "gripper_close_commanded",
    "proximity_consumed_for_action",
    "proximity_zeroed_for_action",
    "proximity_feature_dim",
    "frontend_variant",
    "ablation",
    "live_proximity_aligned_with_action",
    "token_plan_sha256",
    "token_plan_frames_consumed",
    "blur_sigma",
    "blur_diagnostic",
    "first_raw_proximity_sha256",
    "model_output_trace_sha256",
    "model_output_trace_steps",
)


class ContactStorageError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != canonical_hash(payload):
        raise ContactStorageError(f"{label} self-hash mismatch")
    return str(observed)


def compact_contact_audit(audit: dict[str, Any]) -> dict[str, Any]:
    required = (
        "contact_taxonomy_version",
        "sampling_level",
        "sample_count",
        "contact_class_totals",
        "frames_with_contact",
        "maximum_penetration_depth_m",
        "first_contact_step",
        "non_target_contact_entries",
        "collision_free",
        "contact_frame_payload_retained",
    )
    missing = [key for key in required if key not in audit]
    if missing:
        raise ContactStorageError(f"contact audit lacks endpoint fields: {missing}")
    for key in (
        "contact_class_totals",
        "frames_with_contact",
        "maximum_penetration_depth_m",
        "first_contact_step",
    ):
        if set(audit[key]) != set(CONTACT_CLASSES):
            raise ContactStorageError(f"contact class set changed in {key}")
    return {key: audit[key] for key in required}


def inventory(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_identity(result: dict[str, Any], row: dict[str, Any]) -> None:
    expected = {
        "status": "complete",
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "episode_id": row["instance_episode_id"],
        "arm": row["arm"],
        "checkpoint_seed": row["checkpoint_seed"],
        "checkpoint_sha256": row["checkpoint_sha256"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ContactStorageError(
                f"row {row['schedule_index']} identity mismatch: {key}"
            )


def finish_prepared_storage(
    storage: dict[str, Any], storage_path: Path, result_path: Path
) -> dict[str, Any]:
    if storage.get("status") not in ("prepared", "complete"):
        raise ContactStorageError(f"invalid storage transaction state: {storage_path}")
    payload = dict(storage)
    observed = payload.pop("storage_archive_sha256", None)
    if observed != canonical_hash(payload):
        raise ContactStorageError(f"storage transaction self-hash mismatch: {storage_path}")
    if sha256_file(result_path) != storage["compact_result_sha256"]:
        raise ContactStorageError(f"compact result changed: {result_path}")
    if storage["status"] == "complete":
        for item in storage["deleted_payloads"]:
            if Path(item["path"]).exists():
                raise ContactStorageError(f"deleted payload returned: {item['path']}")
        return storage
    for item in storage["deleted_payloads"]:
        path = Path(item["path"])
        if path.exists():
            if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item[
                "sha256"
            ]:
                raise ContactStorageError(f"prepared payload changed: {path}")
            path.unlink()
    storage.pop("storage_archive_sha256", None)
    storage["status"] = "complete"
    storage["completed_utc"] = utc_now()
    storage["storage_archive_sha256"] = canonical_hash(storage)
    write_json_atomic(storage_path, storage)
    return storage


def compact_row(row: dict[str, Any], output_root: Path) -> dict[str, Any]:
    row_dir = output_root / row["output_relpath"]
    result_path = row_dir / "result.json"
    storage_path = row_dir / "storage_archive.json"
    if storage_path.is_file():
        storage = json.loads(storage_path.read_text())
        return finish_prepared_storage(storage, storage_path, result_path)
    original = json.loads(result_path.read_text())
    validate_identity(original, row)
    if original.get("storage_compaction") is not None:
        reference = original["storage_compaction"]
        storage = {
            "schema_version": "pact_contact_storage_archive_v1",
            "status": "prepared",
            "schedule_index": row["schedule_index"],
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "original_result": reference["original_result"],
            "compact_result_sha256": sha256_file(result_path),
            "deleted_payloads": reference["deleted_payloads"],
            "prepared_utc": utc_now(),
            "recovered_from_compact_result": True,
        }
        storage["storage_archive_sha256"] = canonical_hash(storage)
        write_json_atomic(storage_path, storage)
        return finish_prepared_storage(storage, storage_path, result_path)
    missing = [key for key in CORE_RESULT_KEYS if key not in original]
    if missing:
        raise ContactStorageError(f"scientific result lacks core keys: {missing}")
    if original["contact_audit"].get("contact_frame_payload_retained") is not False:
        raise ContactStorageError("summary-only contact instrumentation was not active")
    if original["contact_audit"].get("contact_frames") not in ([], None):
        raise ContactStorageError("unexpected full contact-frame payload")
    deleted_paths = []
    trajectory_raw = original.get("trajectory_path")
    if trajectory_raw:
        trajectory = Path(trajectory_raw)
        if trajectory.is_file():
            deleted_paths.append(trajectory)
    for raw_path in original.get("videos", []):
        path = Path(raw_path)
        if path.is_file():
            deleted_paths.append(path)
    deleted_payloads = [inventory(path) for path in deleted_paths]
    original_result = inventory(result_path)
    compact = {key: original[key] for key in CORE_RESULT_KEYS}
    compact["contact_audit"] = compact_contact_audit(original["contact_audit"])
    policy_info = original.get("policy_info", {})
    policy_summary = {
        key: policy_info[key] for key in POLICY_SUMMARY_KEYS if key in policy_info
    }
    compact["policy_info_summary"] = policy_summary
    # New studies validate intervention metadata through the original key.
    # Preserve the compact summary under both names; no endpoint payload is added.
    compact["policy_info"] = policy_summary
    compact["trajectory_path"] = None
    compact["videos"] = []
    compact["storage_compaction"] = {
        "schema_version": "pact_contact_storage_reference_v1",
        "content_transform": "endpoint_complete_summary_plus_sha256_inventory",
        "original_result": original_result,
        "deleted_payloads": deleted_payloads,
        "raw_payload_byte_content_retained": False,
        "outcome_based_selection": False,
        "endpoint_values_emitted_during_compaction": False,
        "recovery": "payload hashes and sizes remain; deleted bytes are not recoverable",
    }
    write_json_atomic(result_path, compact)
    storage: dict[str, Any] = {
        "schema_version": "pact_contact_storage_archive_v1",
        "status": "prepared",
        "schedule_index": row["schedule_index"],
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "original_result": original_result,
        "compact_result_sha256": sha256_file(result_path),
        "deleted_payloads": deleted_payloads,
        "prepared_utc": utc_now(),
        "recovered_from_compact_result": False,
    }
    storage["storage_archive_sha256"] = canonical_hash(storage)
    write_json_atomic(storage_path, storage)
    return finish_prepared_storage(storage, storage_path, result_path)


def validate_inputs(
    schedule_path: Path, amendment_path: Path, output_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    schedule = json.loads(schedule_path.read_text())
    schedule_sha = validate_self_hash(schedule, "schedule_sha256", "schedule")
    amendment = json.loads(amendment_path.read_text())
    validate_self_hash(amendment, "storage_amendment_sha256", "storage amendment")
    if amendment.get("schema_version") != "pact_contact_storage_amendment_v1":
        raise ContactStorageError("storage amendment schema changed")
    if amendment.get("schedule_sha256") != schedule_sha:
        raise ContactStorageError("storage amendment binds another schedule")
    if amendment.get("compactor_sha256") != sha256_file(Path(__file__).resolve()):
        raise ContactStorageError("compactor differs from frozen amendment")
    if amendment.get("excluded_intact_schedule_indices") != sorted(
        EXCLUDED_SCHEDULE_INDICES
    ):
        raise ContactStorageError("intact-row set changed")
    if Path(amendment["output_root"]).resolve() != output_root.resolve():
        raise ContactStorageError("storage amendment output root changed")
    return schedule, amendment


def run(schedule: dict[str, Any], output_root: Path, *, poll_seconds: float) -> int:
    rows = [
        row
        for row in schedule["rows"]
        if int(row["schedule_index"]) not in EXCLUDED_SCHEDULE_INDICES
    ]
    heartbeat_path = output_root / "storage_compactor_heartbeat.json"
    while True:
        compacted = 0
        eligible = 0
        last_error = None
        for row in rows:
            row_dir = output_root / row["output_relpath"]
            result_path = row_dir / "result.json"
            driver_path = row_dir / "driver_result.json"
            storage_path = row_dir / "storage_archive.json"
            if storage_path.is_file():
                eligible += 1
                try:
                    storage = compact_row(row, output_root)
                    if storage.get("status") != "complete":
                        raise ContactStorageError("storage recovery remained incomplete")
                    compacted += 1
                except Exception as error:
                    last_error = f"{type(error).__name__}: {error}"
                    break
                continue
            if not result_path.is_file() or not driver_path.is_file():
                continue
            driver = json.loads(driver_path.read_text())
            if driver.get("status") != "complete":
                continue
            eligible += 1
            try:
                compact_row(row, output_root)
                compacted += 1
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                break
        stat = os.statvfs(output_root)
        heartbeat: dict[str, Any] = {
            "schema_version": "pact_contact_storage_heartbeat_v1",
            "pid": os.getpid(),
            "schedule_sha256": schedule["schedule_sha256"],
            "compacted_count": compacted,
            "eligible_completed_count": eligible,
            "excluded_intact_schedule_indices": sorted(EXCLUDED_SCHEDULE_INDICES),
            "free_bytes": stat.f_bavail * stat.f_frsize,
            "last_error": last_error,
            "updated_utc": utc_now(),
        }
        heartbeat["storage_compactor_heartbeat_sha256"] = canonical_hash(heartbeat)
        write_json_atomic(heartbeat_path, heartbeat)
        if last_error:
            raise ContactStorageError(last_error)
        execution_summary = output_root / "full_execution_summary.json"
        if execution_summary.is_file() and compacted == len(rows):
            summary: dict[str, Any] = {
                "schema_version": "pact_contact_storage_compaction_summary_v1",
                "schedule_sha256": schedule["schedule_sha256"],
                "expected_compacted_count": len(rows),
                "compacted_count": compacted,
                "excluded_intact_schedule_indices": sorted(
                    EXCLUDED_SCHEDULE_INDICES
                ),
                "all_expected_rows_compacted": True,
                "completed_utc": utc_now(),
            }
            summary["storage_compaction_summary_sha256"] = canonical_hash(summary)
            write_json_atomic(output_root / "storage_compaction_summary.json", summary)
            return 0
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--storage-amendment", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schedule, _amendment = validate_inputs(
        args.schedule.resolve(), args.storage_amendment.resolve(), args.output_root.resolve()
    )
    return run(schedule, args.output_root.resolve(), poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
