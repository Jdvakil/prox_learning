#!/usr/bin/env python3
"""Losslessly archive R2 payloads while retaining analyzer-compatible results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ZSTD = Path("/usr/bin/zstd")
EXCLUDED_SCHEDULE_INDICES = {0, 959}
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
)


class StorageCompactionError(RuntimeError):
    """The content-independent R2 storage transform could not be verified."""


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
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
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


def load_self_hashed(
    path: Path, *, hash_key: str, schema_version: str
) -> dict[str, Any]:
    document = json.loads(path.read_text())
    payload = dict(document)
    observed = payload.pop(hash_key, None)
    if canonical_hash(payload) != observed:
        raise StorageCompactionError(f"{path}: {hash_key} mismatch")
    if document.get("schema_version") != schema_version:
        raise StorageCompactionError(f"{path}: schema mismatch")
    return document


def decompressed_sha256(path: Path) -> str:
    process = subprocess.Popen(
        [str(ZSTD), "-q", "-d", "-c", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    digest = hashlib.sha256()
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    process.stdout.close()
    assert process.stderr is not None
    stderr = process.stderr.read()
    returncode = process.wait()
    if returncode != 0:
        raise StorageCompactionError(
            f"zstd verification failed for {path}: "
            f"{stderr.decode(errors='replace')}"
        )
    return digest.hexdigest()


def archive_file(
    source: Path,
    archive: Path,
    *,
    expected_source_sha256: str | None = None,
    threads: int = 2,
    level: int = 1,
) -> dict[str, Any]:
    if not source.exists():
        raise StorageCompactionError(f"archive source is absent: {source}")
    source_sha = expected_source_sha256 or sha256_file(source)
    source_size = source.stat().st_size
    if archive.exists():
        if decompressed_sha256(archive) != source_sha:
            raise StorageCompactionError(
                f"existing archive does not restore {source}"
            )
    else:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{archive.name}.", dir=archive.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                completed = subprocess.run(
                    [
                        str(ZSTD),
                        f"-T{threads}",
                        f"-{level}",
                        "-q",
                        "-c",
                        str(source),
                    ],
                    stdout=output,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                output.flush()
                os.fsync(output.fileno())
            if completed.returncode != 0:
                raise StorageCompactionError(
                    f"zstd failed for {source}: "
                    f"{completed.stderr.decode(errors='replace')}"
                )
            temporary_path = Path(temporary)
            if decompressed_sha256(temporary_path) != source_sha:
                raise StorageCompactionError(
                    f"new archive does not restore {source}"
                )
            os.replace(temporary_path, archive)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {
        "original_path": str(source),
        "original_size_bytes": source_size,
        "original_sha256": source_sha,
        "archive_path": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "codec": "zstd",
        "zstd_level": level,
        "zstd_threads": threads,
        "decompression_verified": True,
    }


def compact_contact_audit(audit: dict[str, Any]) -> dict[str, Any]:
    if "contact_class_totals" not in audit:
        raise StorageCompactionError("contact audit lacks class totals")
    compact = {
        key: value
        for key, value in audit.items()
        if value is None or isinstance(value, (bool, int, float, str))
    }
    compact["contact_class_totals"] = audit["contact_class_totals"]
    return compact


def build_compact_result(
    original: dict[str, Any],
    *,
    result_archive: dict[str, Any],
    trajectory_archive: dict[str, Any],
) -> dict[str, Any]:
    missing = [key for key in CORE_RESULT_KEYS if key not in original]
    if missing:
        raise StorageCompactionError(
            f"scientific result lacks frozen core keys: {missing}"
        )
    compact = {key: original[key] for key in CORE_RESULT_KEYS}
    compact["contact_audit"] = compact_contact_audit(
        original["contact_audit"]
    )
    policy_info = original.get("policy_info", {})
    compact["policy_info_summary"] = {
        key: policy_info[key]
        for key in POLICY_SUMMARY_KEYS
        if key in policy_info
    }
    compact["trajectory_path"] = trajectory_archive["archive_path"]
    compact["videos"] = original.get("videos", [])
    compact["storage_compaction"] = {
        "schema_version": "pact_r2_storage_compaction_reference_v1",
        "content_transform": "lossless_archive_plus_analyzer_view",
        "full_result_archive": result_archive,
        "trajectory_archive": trajectory_archive,
        "full_result_retained_byte_exact": True,
        "trajectory_retained_byte_exact": True,
        "outcome_based_selection": False,
        "endpoint_values_emitted_during_compaction": False,
    }
    return compact


def validate_row_identity(
    result: dict[str, Any], row: dict[str, Any]
) -> None:
    expected = {
        "status": "complete",
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "episode_id": row["instance_episode_id"],
        "arm": row["arm"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "checkpoint_seed": row["checkpoint_seed"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise StorageCompactionError(
                f"row {row['schedule_index']}: {key} identity mismatch"
            )


def video_records(row_dir: Path, paths: list[str]) -> list[dict[str, Any]]:
    records = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = row_dir / path
        if not path.exists():
            raise StorageCompactionError(f"video is absent: {path}")
        records.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def storage_manifest(
    *,
    row: dict[str, Any],
    compact_result: Path,
    result_archive: dict[str, Any],
    trajectory_archive: dict[str, Any],
    videos: list[dict[str, Any]],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "pact_r2_storage_archive_v1",
        "schedule_index": row["schedule_index"],
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "result_archive": result_archive,
        "trajectory_archive": trajectory_archive,
        "compact_result": {
            "path": str(compact_result),
            "size_bytes": compact_result.stat().st_size,
            "sha256": sha256_file(compact_result),
            "frozen_analyzer_compatible": True,
        },
        "videos": videos,
        "original_payloads_deleted_only_after_verified_archives": True,
        "original_payloads_recoverable": True,
        "outcome_based_selection": False,
        "endpoint_values_emitted_during_compaction": False,
        "compacted_utc": utc_now(),
    }
    document["storage_archive_sha256"] = canonical_hash(document)
    return document


def verify_compacted(row_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    manifest_path = row_dir / "storage_archive.json"
    manifest = load_self_hashed(
        manifest_path,
        hash_key="storage_archive_sha256",
        schema_version="pact_r2_storage_archive_v1",
    )
    if (
        manifest["rollout_id"] != row["rollout_id"]
        or manifest["schedule_row_sha256"] != row["schedule_row_sha256"]
    ):
        raise StorageCompactionError("storage manifest row identity mismatch")
    result_path = row_dir / "result.json"
    compact = json.loads(result_path.read_text())
    validate_row_identity(compact, row)
    if "storage_compaction" not in compact:
        raise StorageCompactionError("result is not the compact analyzer view")
    if sha256_file(result_path) != manifest["compact_result"]["sha256"]:
        raise StorageCompactionError("compact result hash mismatch")
    for key in ("result_archive", "trajectory_archive"):
        record = manifest[key]
        archive = Path(record["archive_path"])
        if sha256_file(archive) != record["archive_sha256"]:
            raise StorageCompactionError(f"{key} archive hash mismatch")
        if decompressed_sha256(archive) != record["original_sha256"]:
            raise StorageCompactionError(f"{key} decompression hash mismatch")
    for record in manifest["videos"]:
        video = Path(record["path"])
        if (
            not video.exists()
            or video.stat().st_size != record["size_bytes"]
            or sha256_file(video) != record["sha256"]
        ):
            raise StorageCompactionError(f"video verification failed: {video}")
    if (row_dir / "trajectory.h5").exists():
        raise StorageCompactionError("unarchived trajectory remains after compaction")
    return manifest


def recover_interrupted_compaction(
    *,
    row_dir: Path,
    row: dict[str, Any],
    compact: dict[str, Any],
) -> dict[str, Any]:
    """Finish publication after an interruption following compact-result replace."""
    validate_row_identity(compact, row)
    reference = compact.get("storage_compaction")
    if not isinstance(reference, dict):
        raise StorageCompactionError("compact result lacks archive references")
    result_record = reference["full_result_archive"]
    trajectory_record = reference["trajectory_archive"]
    for name, record in (
        ("result", result_record),
        ("trajectory", trajectory_record),
    ):
        archive = Path(record["archive_path"])
        if (
            not archive.exists()
            or archive.stat().st_size != record["archive_size_bytes"]
            or sha256_file(archive) != record["archive_sha256"]
            or decompressed_sha256(archive) != record["original_sha256"]
        ):
            raise StorageCompactionError(
                f"interrupted {name} archive cannot be verified"
            )
    trajectory_path = row_dir / "trajectory.h5"
    if trajectory_path.exists():
        if (
            trajectory_path.stat().st_size
            != trajectory_record["original_size_bytes"]
            or sha256_file(trajectory_path)
            != trajectory_record["original_sha256"]
        ):
            raise StorageCompactionError(
                "remaining trajectory differs from verified archive source"
            )
        trajectory_path.unlink()
    videos = video_records(row_dir, compact.get("videos", []))
    manifest = storage_manifest(
        row=row,
        compact_result=row_dir / "result.json",
        result_archive=result_record,
        trajectory_archive=trajectory_record,
        videos=videos,
    )
    write_json_atomic(row_dir / "storage_archive.json", manifest)
    return verify_compacted(row_dir, row)


def compact_row(
    *,
    output_root: Path,
    row: dict[str, Any],
    threads: int,
    level: int,
) -> dict[str, Any]:
    index = int(row["schedule_index"])
    if index in EXCLUDED_SCHEDULE_INDICES:
        return {"schedule_index": index, "status": "excluded_intact"}
    row_dir = output_root / row["output_relpath"]
    manifest_path = row_dir / "storage_archive.json"
    if manifest_path.exists():
        manifest = verify_compacted(row_dir, row)
        return {
            "schedule_index": index,
            "status": "already_compacted_verified",
            "storage_archive_sha256": manifest["storage_archive_sha256"],
        }
    driver = json.loads((row_dir / "driver_result.json").read_text())
    if driver.get("status") != "complete":
        raise StorageCompactionError(f"row {index}: driver is not complete")
    result_path = row_dir / "result.json"
    original = json.loads(result_path.read_text())
    if "storage_compaction" in original:
        manifest = recover_interrupted_compaction(
            row_dir=row_dir,
            row=row,
            compact=original,
        )
        return {
            "schedule_index": index,
            "status": "interrupted_compaction_recovered",
            "storage_archive_sha256": manifest["storage_archive_sha256"],
        }
    validate_row_identity(original, row)
    trajectory_path = row_dir / "trajectory.h5"
    result_record = archive_file(
        result_path,
        row_dir / "result.full.json.zst",
        threads=threads,
        level=level,
    )
    trajectory_record = archive_file(
        trajectory_path,
        row_dir / "trajectory.h5.zst",
        threads=threads,
        level=level,
    )
    videos = video_records(row_dir, original.get("videos", []))
    compact = build_compact_result(
        original,
        result_archive=result_record,
        trajectory_archive=trajectory_record,
    )
    write_json_atomic(result_path, compact)
    if sha256_file(result_path) == result_record["original_sha256"]:
        raise StorageCompactionError("compact result unexpectedly equals original")
    trajectory_path.unlink()
    manifest = storage_manifest(
        row=row,
        compact_result=result_path,
        result_archive=result_record,
        trajectory_archive=trajectory_record,
        videos=videos,
    )
    write_json_atomic(manifest_path, manifest)
    verify_compacted(row_dir, row)
    return {
        "schedule_index": index,
        "status": "compacted",
        "storage_archive_sha256": manifest["storage_archive_sha256"],
        "bytes_before": (
            result_record["original_size_bytes"]
            + trajectory_record["original_size_bytes"]
        ),
        "bytes_after": (
            result_record["archive_size_bytes"]
            + trajectory_record["archive_size_bytes"]
            + manifest["compact_result"]["size_bytes"]
        ),
    }


def eligible_completed_rows(
    schedule: dict[str, Any],
    completion_ledger: dict[str, Any],
) -> list[dict[str, Any]]:
    if completion_ledger["schedule_sha256"] != schedule["schedule_sha256"]:
        raise StorageCompactionError("completion ledger schedule mismatch")
    completed = {
        item["rollout_id"] for item in completion_ledger["completions"]
    }
    return [
        row
        for row in schedule["rows"]
        if row["rollout_id"] in completed
        and int(row["schedule_index"]) not in EXCLUDED_SCHEDULE_INDICES
    ]


def write_heartbeat(
    path: Path,
    *,
    schedule: dict[str, Any],
    compacted: int,
    eligible: int,
    last_error: str | None,
) -> None:
    stat = os.statvfs(path.parent)
    document: dict[str, Any] = {
        "schema_version": "pact_r2_storage_compactor_heartbeat_v1",
        "pid": os.getpid(),
        "schedule_sha256": schedule["schedule_sha256"],
        "compacted_count": compacted,
        "eligible_completed_count": eligible,
        "excluded_intact_schedule_indices": sorted(
            EXCLUDED_SCHEDULE_INDICES
        ),
        "free_bytes": stat.f_bavail * stat.f_frsize,
        "last_error": last_error,
        "updated_utc": utc_now(),
    }
    document["storage_compactor_heartbeat_sha256"] = canonical_hash(document)
    write_json_atomic(path, document)


def validate_inputs(
    *,
    config_path: Path,
    schedule_path: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_self_hashed(
        config_path,
        hash_key="storage_amendment_sha256",
        schema_version="pact_r2_storage_amendment_v1",
    )
    schedule = json.loads(schedule_path.read_text())
    schedule_payload = dict(schedule)
    observed = schedule_payload.pop("schedule_sha256")
    if canonical_hash(schedule_payload) != observed:
        raise StorageCompactionError("schedule self-hash mismatch")
    if observed != config["schedule_sha256"]:
        raise StorageCompactionError("storage amendment schedule mismatch")
    if Path(config["output_root"]).resolve() != output_root.resolve():
        raise StorageCompactionError("storage amendment output root mismatch")
    if sha256_file(Path(__file__).resolve()) != config["compactor_sha256"]:
        raise StorageCompactionError("compactor differs from frozen amendment")
    if set(config["excluded_intact_schedule_indices"]) != (
        EXCLUDED_SCHEDULE_INDICES
    ):
        raise StorageCompactionError("intact-row exclusions changed")
    return config, schedule


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    _config, schedule = validate_inputs(
        config_path=args.storage_amendment,
        schedule_path=args.schedule,
        output_root=output_root,
    )
    pid_path = output_root / "storage_compactor_pid.json"
    heartbeat_path = output_root / "storage_compactor_heartbeat.json"
    summary_path = output_root / "storage_compaction_summary.json"
    write_json_atomic(
        pid_path,
        {
            "schema_version": "pact_r2_storage_compactor_pid_v1",
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "process_group_id": os.getpgrp(),
            "session_id": os.getsid(0),
            "schedule_sha256": schedule["schedule_sha256"],
            "started_utc": utc_now(),
        },
    )
    totals = {
        "rows_newly_compacted": 0,
        "bytes_before": 0,
        "bytes_after": 0,
    }
    processed_rollout_ids: set[str] = set()
    while True:
        ledger_path = output_root / "completion_ledger.json"
        ledger = json.loads(ledger_path.read_text())
        eligible = eligible_completed_rows(schedule, ledger)
        for row in eligible:
            if row["rollout_id"] in processed_rollout_ids:
                continue
            result = compact_row(
                output_root=output_root,
                row=row,
                threads=args.threads,
                level=args.level,
            )
            processed_rollout_ids.add(row["rollout_id"])
            if result["status"] == "compacted":
                totals["rows_newly_compacted"] += 1
                totals["bytes_before"] += result["bytes_before"]
                totals["bytes_after"] += result["bytes_after"]
        compacted = len(
            list((output_root / "rows").glob("*/storage_archive.json"))
        )
        write_heartbeat(
            heartbeat_path,
            schedule=schedule,
            compacted=compacted,
            eligible=len(eligible),
            last_error=None,
        )
        execution_path = output_root / "execution_summary.json"
        reconciled = False
        if execution_path.exists():
            execution = json.loads(execution_path.read_text())
            reconciled = execution.get("scientific_schedule_reconciled") is True
        expected_compacted = len(schedule["rows"]) - len(
            EXCLUDED_SCHEDULE_INDICES
        )
        if args.once or (reconciled and compacted == expected_compacted):
            summary: dict[str, Any] = {
                "schema_version": "pact_r2_storage_compaction_summary_v1",
                "schedule_sha256": schedule["schedule_sha256"],
                "reconciled_execution_observed": reconciled,
                "compacted_count": compacted,
                "expected_compacted_count": expected_compacted,
                "excluded_intact_schedule_indices": sorted(
                    EXCLUDED_SCHEDULE_INDICES
                ),
                **totals,
                "outcome_based_selection": False,
                "endpoint_values_emitted_during_compaction": False,
                "finished_utc": utc_now(),
            }
            summary["storage_compaction_summary_sha256"] = canonical_hash(
                summary
            )
            write_json_atomic(summary_path, summary)
            return 0
        time.sleep(args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--storage-amendment", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.level != 1 or args.poll_seconds <= 0:
        raise SystemExit("R2 storage compactor arguments violate amendment")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
