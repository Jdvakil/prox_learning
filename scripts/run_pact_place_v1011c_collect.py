#!/usr/bin/env python3
"""Collect exactly 100 balanced strict-clean V10.11c demonstrations.

The runtime is the already validated V10.11c sampler plus the established
hybrid camera publisher.  Accepted rows contain wrist RGB, table-camera RGB,
per-frame table-camera calibration, and all 40 raw proximity cameras.  Contact
audit storage is summary-only so the scientific predicate is unchanged while
avoiding the large duplicate diagnostic payload in every training row.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pact_place_v1011c_collection_contract as contract  # noqa: E402
from pact_place_v105_contract import (  # noqa: E402
    canonical_payload_sha256,
    empty_authorization,
    sha256_file,
    sha256_payload,
    write_immutable_create_only,
)

THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
PER_IN_FLIGHT_RESERVE_GIB = 0.20
SMOKE_COLLECTION_ROOT = ROOT / "diagnostics_output/pact_place_v1011c_collection_100_smoke"
SMOKE_DATASET_ROOT = ROOT / "assets/datagen/pact_place_corridor_v10_11c_100_smoke"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _root_relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed process can leave one torn final line.  Earlier fsynced
            # records remain authoritative and the partial line is ignored.
            continue
    return rows


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def tally(
    records: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    quota = contract.quotas()
    accepted = {key: 0 for key in quota}
    attempted = {key: 0 for key in quota}
    next_index = {key: 0 for key in quota}
    seen_attempts: set[str] = set()
    for record in records:
        key = str(record.get("cell", ""))
        identifier = str(record.get("attempt_id", ""))
        if key not in quota:
            raise RuntimeError(f"ledger contains unknown cell {key!r}")
        if not identifier or identifier in seen_attempts:
            raise RuntimeError(f"ledger contains duplicate/missing attempt {identifier!r}")
        seen_attempts.add(identifier)
        index = int(record["attempt_index"])
        attempted[key] += 1
        next_index[key] = max(next_index[key], index + 1)
        if record.get("accepted"):
            accepted[key] += 1
    over = {key: count for key, count in accepted.items() if count > quota[key]}
    if over:
        raise RuntimeError(f"ledger exceeds frozen quota: {over}")
    return accepted, attempted, next_index


def remaining(accepted: dict[str, int]) -> dict[str, int]:
    quota = contract.quotas()
    return {key: quota[key] - accepted.get(key, 0) for key in quota}


def quota_met(accepted: dict[str, int]) -> bool:
    return all(value == 0 for value in remaining(accepted).values())


def _upstream_preflight() -> dict[str, Any]:
    problems: list[str] = []
    try:
        upstream = contract.upstream_bindings()
    except Exception as error:  # noqa: BLE001
        upstream = {}
        problems.append(f"upstream:{type(error).__name__}:{error}")
    try:
        implementation = contract.implementation_bindings()
    except Exception as error:  # noqa: BLE001
        implementation = {}
        problems.append(f"implementation:{type(error).__name__}:{error}")
    streams = contract.streams_are_disjoint()
    if not streams["disjoint"]:
        problems.append(f"seed streams overlap: {streams['overlap']}")
    totals = contract.quota_totals()
    if totals["total"] != contract.TARGET_SUCCESSES:
        problems.append("quota total differs from target")
    scenes: dict[str, Any] = {}
    for pose, entry in contract.SCENE_BY_POSE.items():
        path = ROOT / entry["relative"]
        observed = sha256_file(path) if path.is_file() else None
        scenes[pose] = {"path": entry["relative"], "expected": entry["sha256"],
                        "observed": observed, "passed": observed == entry["sha256"]}
        if observed != entry["sha256"]:
            problems.append(f"scene drift: {pose}")
    free_gib = shutil.disk_usage(ROOT).free / 2**30
    if free_gib < contract.MIN_FREE_GIB + PER_IN_FLIGHT_RESERVE_GIB:
        problems.append(f"only {free_gib:.2f} GiB free")
    return {
        "passed": not problems,
        "problems": problems,
        "upstream": upstream,
        "implementation": implementation,
        "streams": streams,
        "quota_totals": totals,
        "scenes": scenes,
        "disk_free_gib": free_gib,
        "minimum_terminal_free_gib": contract.MIN_FREE_GIB,
    }


def _smoke_row(ordinal: int) -> dict[str, Any]:
    family, side, pose = contract.cells()[int(ordinal) % len(contract.cells())]
    attempt_index = int(ordinal) // len(contract.cells())
    row = contract.environment.build_row(
        family,
        side,
        pose,
        attempt_index,
        stream=contract.SMOKE_STREAM,
        master_seed=contract.SMOKE_MASTER_SEED,
        role_index=0,
    )
    identifier = sha256_payload(
        [contract.SMOKE_STREAM, contract.SMOKE_MASTER_SEED,
         family, side, pose, attempt_index]
    )
    row.update({
        "attempt_id": identifier,
        "episode_id": identifier,
        "cell": contract.cell_key(family, side, pose),
        "contract_version": contract.CONTRACT_VERSION,
        "seed_stream": contract.SMOKE_STREAM,
        "pact_v1011c_table_camera_required": True,
        "smoke_only": True,
    })
    row.pop("row_sha256", None)
    row["row_sha256"] = sha256_payload(row)
    return row


def worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Use the proven hybrid-camera publisher with the V10.11c sampler."""
    for name, value in THREAD_ENV.items():
        os.environ[name] = value
    os.environ["PACT_CONTACT_AUDIT_SUMMARY_ONLY"] = "1"
    import run_pact_place_v1010_tablecam_validation as tablecam

    # run_attempt reads both names from its module at execution time.  Binding
    # both is essential: the second controls retry seeds as well as the first
    # scientific seed in the manifest row.
    tablecam.SAMPLER_CLASS = contract.SAMPLER_CLASS
    if payload["row"].get("smoke_only"):
        def smoke_cell_seed(
            family: str, side: str, pose: str, attempt_index: int
        ) -> dict[str, int]:
            digest = hashlib.sha256(
                f"{contract.SMOKE_STREAM}:{contract.SMOKE_MASTER_SEED}:"
                f"{family}:{side}:{pose}:{int(attempt_index)}".encode()
            ).digest()
            value = int.from_bytes(digest[:8], "big")
            return {"seed_u32": value % (2**32), "seed_u64": value}

        tablecam.cell_seed = smoke_cell_seed
    else:
        tablecam.cell_seed = contract.cell_seed
    return tablecam.run_attempt(payload)


def _coordinator_validation(row_dir: Path) -> dict[str, Any]:
    import run_pact_place_v1010_tablecam_validation as tablecam
    import run_pact_place_v108_collect as v108

    base = v108.validate_trainable(row_dir)
    table = tablecam.validate_table_camera(row_dir)
    return {"base": base, "table_camera": table,
            "passed": bool(base["passed"] and table["passed"])}


def compact_record(
    result: dict[str, Any], row: dict[str, Any], validation: dict[str, Any] | None,
) -> dict[str, Any]:
    audit = result.get("contact_audit") or {}
    telemetry = result.get("pact_v106_frame_telemetry") or {}
    stability = result.get("clutter_stability_events") or []
    by_slot = {slot: 0 for slot in contract.ACTIVE_CLUTTER_SLOTS}
    for event in stability:
        body = str(event.get("body", ""))
        for slot in by_slot:
            if f"pact_clutter_{slot}" in body:
                by_slot[slot] += 1
    return {
        "attempt_id": row["attempt_id"],
        "cell": row["cell"],
        "family_id": row["family_id"],
        "intrusion_side": row["intrusion_side"],
        "pose_id": row["pose_id"],
        "attempt_index": int(row["attempt_index"]),
        "task_seed_u32": int(row["task_seed_u32"]),
        "task_seed_u64": int(row["task_seed_u64"]),
        "row_sha256": row["row_sha256"],
        "environment_version": contract.ENVIRONMENT_VERSION,
        "status": result.get("status"),
        "accepted": bool(result.get("accepted") and validation and validation["passed"]),
        "clean_success": bool(result.get("v108_clean_success")),
        "task_success": bool(result.get("task_success")),
        "defects": result.get("v108_defects") or [],
        "episode_steps": result.get("episode_steps"),
        "contact_class_totals": audit.get("contact_class_totals") or {},
        "clutter_stability_event_count": len(stability),
        "clutter_stability_events_by_slot": by_slot,
        "min_pendant_clearance_m": telemetry.get("min_clearance_m"),
        "pendant_contact_frames": telemetry.get(
            "pendant_robot_or_target_contact_frames"
        ),
        "trajectory_h5": result.get("trajectory_h5"),
        "trajectory_h5_sha256": result.get("trajectory_h5_sha256"),
        "table_camera_rgb_required": True,
        "coordinator_validation": validation,
        "elapsed_s": result.get("elapsed_s"),
        "error": result.get("error"),
        "result_sha256": result.get("result_sha256"),
    }


def _status_payload(
    *, accepted: dict[str, int], attempted: dict[str, int], started: float,
    in_flight: set[str], stop_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "pact_place_v1011c_collection_100_status_v1",
        "updated_utc": utc_now(),
        "target_successes": contract.TARGET_SUCCESSES,
        "accepted_total": sum(accepted.values()),
        "attempts_total": sum(attempted.values()),
        "accepted_by_cell": dict(sorted(accepted.items())),
        "remaining_by_cell": dict(sorted(remaining(accepted).items())),
        "in_flight_cells": sorted(in_flight),
        "elapsed_hours": (time.time() - started) / 3600.0,
        "disk_free_gib": shutil.disk_usage(ROOT).free / 2**30,
        "stop_reason": stop_reason,
    }


def _verify_existing(dataset_root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    problems: list[str] = []
    accepted = [record for record in records if record.get("accepted")]
    for index, record in enumerate(accepted):
        row_dir = dataset_root / "rows" / str(record["attempt_id"])[:16]
        validation = _coordinator_validation(row_dir)
        if not validation["passed"]:
            problems.append(f"row {index} {record['attempt_id']}: {validation}")
        h5 = row_dir / "trajectory.h5"
        if not h5.is_file() or sha256_file(h5) != record.get("trajectory_h5_sha256"):
            problems.append(f"row {index} {record['attempt_id']}: HDF5 hash mismatch")
    counts, attempted, _ = tally(records)
    return {
        "passed": not problems and quota_met(counts),
        "problems": problems,
        "accepted_total": len(accepted),
        "attempts_total": sum(attempted.values()),
        "quota_met": quota_met(counts),
        "accepted_by_cell": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=contract.DEFAULT_WORKERS)
    parser.add_argument("--max-attempts", type=int,
                        default=contract.MAX_SCIENTIFIC_ATTEMPTS)
    parser.add_argument("--collection-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-attempts", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    for name, value in THREAD_ENV.items():
        os.environ[name] = value
    os.environ["PACT_CONTACT_AUDIT_SUMMARY_ONLY"] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    preflight = _upstream_preflight()
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0 if preflight["passed"] else 1
    if not preflight["passed"]:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 1

    if args.smoke_only:
        collection_root = (args.collection_root or SMOKE_COLLECTION_ROOT).resolve()
        dataset_root = (args.dataset_root or SMOKE_DATASET_ROOT).resolve()
    else:
        collection_root = (args.collection_root or (ROOT / contract.COLLECTION_ROOT)).resolve()
        dataset_root = (args.dataset_root or (ROOT / contract.DATASET_ROOT)).resolve()
    # run_attempt records paths relative to this repository; external or /tmp
    # destinations would violate that published schema.
    _root_relative(collection_root)
    _root_relative(dataset_root)
    rows_root = dataset_root / "rows"
    rows_root.mkdir(parents=True, exist_ok=True)
    collection_root.mkdir(parents=True, exist_ok=True)

    frozen = contract.build_contract()
    frozen_path = collection_root / "contract.json"
    if frozen_path.exists():
        observed = json.loads(frozen_path.read_text())
        if observed != frozen:
            raise RuntimeError("frozen collection contract differs from current inputs/code")
    else:
        write_immutable_create_only(frozen_path, frozen)

    ledger_path = collection_root / "ledger.jsonl"
    records = _read_jsonl(ledger_path)
    if args.verify_only:
        report = _verify_existing(dataset_root, records)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1

    if args.smoke_only:
        if records:
            print(json.dumps({"smoke_already_ran": True, "records": records}, indent=2))
            return 0 if any(record.get("accepted") for record in records) else 1
        n_smoke = max(1, int(args.smoke_attempts))
        payloads: list[dict[str, Any]] = []
        for ordinal in range(n_smoke):
            row = _smoke_row(ordinal)
            row_dir = rows_root / row["attempt_id"][:16]
            if row_dir.exists():
                raise RuntimeError(f"refusing existing smoke row {row_dir}")
            payloads.append({
                "row": row,
                "row_dir": str(row_dir),
                "scene_xml": str(
                    ROOT / contract.SCENE_BY_POSE[row["pose_id"]]["relative"]
                ),
            })
        smoke_records: list[dict[str, Any]] = []
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, n_smoke),
            mp_context=context,
            max_tasks_per_child=1,
        ) as pool:
            future_payload = {pool.submit(worker, payload): payload for payload in payloads}
            for future in concurrent.futures.as_completed(future_payload):
                payload = future_payload[future]
                row = payload["row"]
                try:
                    result = future.result()
                except BaseException as error:  # noqa: BLE001
                    result = {
                        "status": "infrastructure_failure",
                        "error": f"worker died: {type(error).__name__}: {error}",
                        "v108_clean_success": False,
                        "v108_defects": ["worker_died"],
                        "accepted": False,
                    }
                row_dir = Path(payload["row_dir"])
                validation = (
                    _coordinator_validation(row_dir) if result.get("accepted") else None
                )
                record = compact_record(result, row, validation)
                _append_jsonl(ledger_path, record)
                smoke_records.append(record)
                if not record["accepted"]:
                    import run_pact_place_v108_collect as v108

                    v108.prune_failed(row_dir)
        accepted_smoke = [record for record in smoke_records if record["accepted"]]
        row_sizes = {}
        for record in accepted_smoke:
            row_dir = rows_root / record["attempt_id"][:16]
            row_sizes[record["attempt_id"]] = sum(
                path.stat().st_size for path in row_dir.rglob("*") if path.is_file()
            )
        report = {
            "accepted": bool(accepted_smoke),
            "accepted_count": len(accepted_smoke),
            "attempt_count": len(smoke_records),
            "records": smoke_records,
            "accepted_row_bytes": row_sizes,
            "maximum_accepted_row_mib": (
                max(row_sizes.values()) / 2**20 if row_sizes else None
            ),
        }
        _atomic_status(collection_root / "status.json", report)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0 if accepted_smoke else 1

    accepted, attempted, next_index = tally(records)
    started = time.time()
    started_utc = utc_now()
    deadline = started + contract.MAX_WALL_CLOCK_HOURS * 3600.0
    budget = min(int(args.max_attempts), contract.MAX_SCIENTIFIC_ATTEMPTS)
    in_flight: set[str] = set()
    futures: dict[Any, dict[str, Any]] = {}
    infrastructure: list[dict[str, Any]] = []
    stop_reason: str | None = None
    context = multiprocessing.get_context("spawn")

    print(
        f"V10.11c collection: {sum(accepted.values())}/{contract.TARGET_SUCCESSES} "
        f"accepted, {sum(attempted.values())} attempts, {args.workers} workers",
        flush=True,
    )
    _atomic_status(collection_root / "status.json", _status_payload(
        accepted=accepted, attempted=attempted, started=started, in_flight=in_flight
    ))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        max_tasks_per_child=1,
    ) as pool:
        while True:
            if quota_met(accepted) and not futures:
                stop_reason = "target_successes_reached"
                break
            if sum(attempted.values()) >= budget and not futures:
                stop_reason = "scientific_attempt_budget_exhausted"
                break
            if time.time() >= deadline and not futures:
                stop_reason = "wall_clock_budget_exhausted"
                break
            if infrastructure and not futures:
                stop_reason = "infrastructure_or_schema_defect"
                break

            while (
                not infrastructure
                and len(futures) < args.workers
                and sum(attempted.values()) + len(futures) < budget
                and time.time() < deadline
            ):
                free = shutil.disk_usage(ROOT).free / 2**30
                required = contract.MIN_FREE_GIB + PER_IN_FLIGHT_RESERVE_GIB * (
                    len(futures) + 1
                )
                if free < required:
                    if not futures:
                        stop_reason = "insufficient_disk"
                    break
                open_cells = [
                    key for key, need in remaining(accepted).items()
                    if need > 0 and key not in in_flight
                ]
                if not open_cells:
                    break
                # Complete the common four-per-cell base before the four bonus
                # rows; attempt count breaks ties without changing quotas.
                key = min(open_cells, key=lambda item: (
                    accepted[item], attempted[item], item
                ))
                family, side, pose = key.split("|")
                index = next_index[key]
                row = contract.build_row(family, side, pose, index)
                row_dir = rows_root / row["attempt_id"][:16]
                if row_dir.exists():
                    raise RuntimeError(f"refusing existing row directory {row_dir}")
                payload = {
                    "row": row,
                    "row_dir": str(row_dir),
                    "scene_xml": str(ROOT / contract.SCENE_BY_POSE[pose]["relative"]),
                }
                next_index[key] += 1
                in_flight.add(key)
                futures[pool.submit(worker, payload)] = payload

            if not futures:
                stop_reason = stop_reason or "no_schedulable_cells"
                break

            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                payload = futures.pop(future)
                row = payload["row"]
                key = row["cell"]
                in_flight.discard(key)
                try:
                    result = future.result()
                except BaseException as error:  # noqa: BLE001
                    result = {
                        "status": "infrastructure_failure",
                        "error": f"worker died: {type(error).__name__}: {error}"[:400],
                        "traceback": traceback.format_exc()[-1800:],
                        "v108_clean_success": False,
                        "v108_defects": ["worker_died"],
                        "accepted": False,
                    }

                validation = None
                if result.get("accepted"):
                    validation = _coordinator_validation(Path(payload["row_dir"]))
                is_infrastructure = bool(
                    result.get("status") == "infrastructure_failure"
                    or (result.get("v108_clean_success") and not result.get("accepted"))
                    or (result.get("accepted") and not validation["passed"])
                )
                record = compact_record(result, row, validation)
                if is_infrastructure:
                    record["accepted"] = False
                    record["infrastructure_defect"] = True
                    infrastructure.append(record)
                    _append_jsonl(collection_root / "infrastructure.jsonl", record)
                    print(json.dumps({"HALT": record}, default=str), flush=True)
                    continue

                # Only scientific outcomes advance the registered stream.
                _append_jsonl(ledger_path, record)
                records.append(record)
                attempted[key] += 1
                if record["accepted"]:
                    accepted[key] += 1
                else:
                    import run_pact_place_v108_collect as v108

                    v108.prune_failed(Path(payload["row_dir"]))
                print(json.dumps({
                    "accepted": sum(accepted.values()),
                    "target": contract.TARGET_SUCCESSES,
                    "attempts": sum(attempted.values()),
                    "cell": key,
                    "outcome": "ACCEPT" if record["accepted"] else "reject",
                    "defects": record["defects"][:3],
                    "disk_free_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2),
                }), flush=True)

            _atomic_status(collection_root / "status.json", _status_payload(
                accepted=accepted,
                attempted=attempted,
                started=started,
                in_flight=in_flight,
                stop_reason="infrastructure_or_schema_defect" if infrastructure else None,
            ))

    records = _read_jsonl(ledger_path)
    accepted, attempted, _ = tally(records)
    closeout = {
        **empty_authorization(),
        "schema_version": "pact_place_v1011c_collection_100_closeout_v1",
        "contract_version": contract.CONTRACT_VERSION,
        "contract_payload_sha256": frozen["payload_sha256"],
        "environment_version": contract.ENVIRONMENT_VERSION,
        "sampler_class": contract.SAMPLER_CLASS,
        "owner_requested_target": 100,
        "previous_144_target_superseded_before_collection": True,
        "stop_reason": stop_reason,
        "target_successes": contract.TARGET_SUCCESSES,
        "accepted_total": sum(accepted.values()),
        "attempts_total": sum(attempted.values()),
        "accepted_by_cell": dict(sorted(accepted.items())),
        "attempted_by_cell": dict(sorted(attempted.items())),
        "remaining_by_cell": dict(sorted(remaining(accepted).items())),
        "quota_totals": contract.quota_totals(),
        "quotas_met": quota_met(accepted),
        "one_in_flight_per_cell": True,
        "infrastructure_defects": infrastructure,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "elapsed_hours": (time.time() - started) / 3600.0,
        "ledger": _root_relative(ledger_path),
        "ledger_sha256": sha256_file(ledger_path) if ledger_path.is_file() else None,
        "dataset_root": _root_relative(dataset_root),
        "disk_free_gib": shutil.disk_usage(ROOT).free / 2**30,
        "observations": frozen["observations"],
        "does_not_claim_phase0_pass": True,
        "authorizes_conversion": False,
        "authorizes_training": False,
        "authorizes_evaluation": False,
    }
    closeout["payload_sha256"] = canonical_payload_sha256(closeout)
    closeout_path = collection_root / "closeout.json"
    if closeout_path.exists():
        closeout_path = collection_root / f"closeout_{int(time.time())}.json"
    write_immutable_create_only(closeout_path, closeout)
    _atomic_status(collection_root / "status.json", _status_payload(
        accepted=accepted, attempted=attempted, started=started,
        in_flight=set(), stop_reason=stop_reason
    ))
    print(json.dumps({
        "accepted": closeout["accepted_total"],
        "target": closeout["target_successes"],
        "attempts": closeout["attempts_total"],
        "quotas_met": closeout["quotas_met"],
        "stop_reason": stop_reason,
        "elapsed_hours": closeout["elapsed_hours"],
        "closeout": _root_relative(closeout_path),
    }, indent=2), flush=True)
    if infrastructure:
        return 2
    return 0 if quota_met(accepted) else 1


if __name__ == "__main__":
    raise SystemExit(main())
