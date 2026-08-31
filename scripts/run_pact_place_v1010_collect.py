#!/usr/bin/env python3
"""V10.10 four-object collection: exactly six strict-clean successes per cell.

Reuses the V10.8 attempt runner and trainable validator unchanged, with the
sampler class rebound per worker. That runner already refuses a draw whose
initial state carries a disallowed contact, which is the rejection the plan
requires before a scientific rollout.

Scheduling differs from V10.8 deliberately. V10.8 dispatched a batch and could
wrap within it, putting two attempts for one cell in flight at once and
over-collecting. Here a cell with an attempt in flight is never scheduled again
until that attempt lands, so parallel completion cannot exceed quota.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "submodules" / "molmospaces"))

from pact_place_v1010_contract import (  # noqa: E402
    ACTIVE_CLUTTER_SLOTS, COLLECTION_ROOT, CONTRACT_VERSION_V1010,
    DATASET_ROOT, ENVIRONMENT_VERSION, MAX_IN_FLIGHT_PER_CELL,
    MAX_SCIENTIFIC_ATTEMPTS, MAX_WALL_CLOCK_HOURS, OBJECT_LABELS,
    QUOTA_PER_CELL, SAMPLER_CLASS, SCENE_BY_POSE, TARGET_SUCCESSES,
    build_row, canonical_payload_sha256, cell_key, cells, empty_authorization,
    quotas, sha256_file, write_immutable_create_only,
)

MIN_FREE_GIB = 6.0


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def per_object_telemetry(result: dict[str, Any]) -> dict[str, Any]:
    """Contact and stability broken out for each active clutter slot."""
    audit = result.get("contact_audit") or {}
    frames = audit.get("contact_frames") or []
    contacts = {slot: 0 for slot in ACTIVE_CLUTTER_SLOTS}
    frame_hits = {slot: 0 for slot in ACTIVE_CLUTTER_SLOTS}
    for frame in frames:
        seen: set[str] = set()
        for pair in frame.get("pairs") or []:
            blob = " ".join(str(pair.get(k, "")) for k in
                            ("geom1", "geom2", "body1", "body2", "root1", "root2"))
            for slot in ACTIVE_CLUTTER_SLOTS:
                if f"pact_clutter_{slot}" in blob:
                    contacts[slot] += 1
                    seen.add(slot)
        for slot in seen:
            frame_hits[slot] += 1
    events = result.get("clutter_stability_events") or []
    stability = {slot: 0 for slot in ACTIVE_CLUTTER_SLOTS}
    for event in events:
        body = str(event.get("body", ""))
        for slot in ACTIVE_CLUTTER_SLOTS:
            if f"pact_clutter_{slot}" in body:
                stability[slot] += 1
    return {
        "labels": {slot: OBJECT_LABELS[slot] for slot in ACTIVE_CLUTTER_SLOTS},
        "contact_pair_entries": contacts,
        "frames_with_contact": frame_hits,
        "stability_events": stability,
        "any_contact": sorted(s for s, v in contacts.items() if v),
        "any_stability": sorted(s for s, v in stability.items() if v),
        "frames_inspected": len(frames),
    }


def worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebind the sampler for this process, then run the V10.8 attempt path."""
    os.environ.setdefault("PACT_CONTACT_AUDIT_SUMMARY_ONLY", "0")
    import run_pact_place_v108_collect as v108

    v108.SAMPLER_CLASS = SAMPLER_CLASS
    result = v108.run_attempt(payload)
    if result.get("published"):
        validation = v108.validate_trainable(Path(payload["row_dir"]))
        result["v1010_schema_validation"] = validation
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    collection = ROOT / COLLECTION_ROOT
    collection.mkdir(parents=True, exist_ok=True)
    rows_root = ROOT / DATASET_ROOT / "rows"
    rows_root.mkdir(parents=True, exist_ok=True)
    ledger_path = collection / "ledger.jsonl"

    import run_pact_place_v108_collect as v108

    quota = quotas()
    accepted: dict[str, int] = {c: 0 for c in quota}
    attempted: dict[str, int] = {c: 0 for c in quota}
    in_flight: set[str] = set()
    records: list[dict[str, Any]] = []
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                records.append(record)
                attempted[record["cell"]] = max(
                    attempted[record["cell"]], int(record["attempt_index"]) + 1)
                accepted[record["cell"]] += int(bool(record["accepted"]))

    context = multiprocessing.get_context("spawn")
    started = time.monotonic()
    started_utc = utc_now()
    stop_reason = "quota_met"
    ledger = ledger_path.open("a")

    def schedulable() -> list[str]:
        return [c for c in quota
                if accepted[c] < quota[c] and c not in in_flight]

    print(f"V10.10 collection: {TARGET_SUCCESSES} successes, "
          f"{QUOTA_PER_CELL}/cell, {args.workers} workers", flush=True)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context,
            max_tasks_per_child=1) as pool:
        futures: dict[Any, dict[str, Any]] = {}
        while True:
            total_attempts = sum(attempted.values())
            elapsed_h = (time.monotonic() - started) / 3600
            if sum(accepted.values()) >= TARGET_SUCCESSES and not futures:
                break
            if total_attempts >= MAX_SCIENTIFIC_ATTEMPTS and not futures:
                stop_reason = "attempt_budget_exhausted"
                break
            if elapsed_h >= MAX_WALL_CLOCK_HOURS and not futures:
                stop_reason = "wall_clock_exhausted"
                break
            free_gib = shutil.disk_usage("/root").free / 2**30
            if free_gib < MIN_FREE_GIB and not futures:
                stop_reason = "insufficient_disk"
                break

            while (len(futures) < args.workers
                   and total_attempts + len(futures) < MAX_SCIENTIFIC_ATTEMPTS
                   and elapsed_h < MAX_WALL_CLOCK_HOURS):
                open_cells = schedulable()
                if not open_cells:
                    break
                # Deterministic: the cell furthest from quota, then cell order.
                cell = min(open_cells,
                           key=lambda c: (accepted[c], attempted[c], c))
                family, side, pose = cell.split("|")
                index = attempted[cell]
                row = build_row(family, side, pose, index)
                row_dir = rows_root / row["attempt_id"][:16]
                payload = {"row": row, "row_dir": str(row_dir),
                           "scene_xml": str(ROOT / SCENE_BY_POSE[pose]["relative"])}
                in_flight.add(cell)
                attempted[cell] = index + 1
                futures[pool.submit(worker, payload)] = {
                    "cell": cell, "row": row, "row_dir": str(row_dir)}
            if not futures:
                stop_reason = "no_schedulable_cells"
                break

            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                meta = futures.pop(future)
                cell, row = meta["cell"], meta["row"]
                in_flight.discard(cell)
                try:
                    result = future.result()
                except BaseException as error:  # noqa: BLE001
                    result = {"status": "worker_died",
                              "error": f"{type(error).__name__}: {error}"[:300]}
                defects = v108.row_defects(result) if result.get("status") else ["worker_died"]
                clean = not defects and bool(result.get("task_success"))
                validation = result.get("v1010_schema_validation") or {}
                ok = bool(clean and validation.get("passed"))
                if accepted[cell] >= quota[cell]:
                    ok = False
                    defects = list(defects) + ["cell_already_at_quota"]
                record = {
                    "attempt_id": row["attempt_id"], "cell": cell,
                    "family_id": row["family_id"],
                    "intrusion_side": row["intrusion_side"],
                    "pose_id": row["pose_id"],
                    "attempt_index": int(row["attempt_index"]),
                    "task_seed_u32": int(row["task_seed_u32"]),
                    "row_sha256": row["row_sha256"],
                    "identity_sha256": row["pact_v1010_identity_sha256"],
                    "environment_version": ENVIRONMENT_VERSION,
                    "status": result.get("status"),
                    "error": result.get("error"),
                    "task_success": bool(result.get("task_success")),
                    "clean_success": bool(clean),
                    "accepted": ok,
                    "defects": defects,
                    "episode_steps": result.get("episode_steps"),
                    "elapsed_s": result.get("elapsed_s"),
                    "contact_class_totals": (result.get("contact_audit") or {}).get(
                        "contact_class_totals"),
                    "clutter_stability_events": len(
                        result.get("clutter_stability_events") or []),
                    "per_object": per_object_telemetry(result),
                    "min_pendant_clearance_m": result.get("min_pendant_clearance_m"),
                    "trajectory_h5": result.get("trajectory_h5"),
                    "trajectory_h5_sha256": result.get("trajectory_h5_sha256"),
                    "schema_validation_passed": bool(validation.get("passed")),
                }
                records.append(record)
                ledger.write(json.dumps(record, sort_keys=True) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
                if ok:
                    accepted[cell] += 1
                else:
                    v108.prune_failed(Path(meta["row_dir"]))
                total = sum(accepted.values())
                print(f"  accepted {total}/{TARGET_SUCCESSES} | attempts "
                      f"{sum(attempted.values())} | {cell} "
                      f"{'ACCEPT' if ok else 'reject:' + ','.join(defects[:2])}",
                      flush=True)
    ledger.close()

    short = {c: quota[c] - accepted[c] for c in quota if accepted[c] < quota[c]}
    over = {c: accepted[c] - quota[c] for c in quota if accepted[c] > quota[c]}
    document = {
        **empty_authorization(),
        "schema_version": "pact_place_v1010_collection_closeout_v1",
        "contract_version": CONTRACT_VERSION_V1010,
        "role": "V10.10 four-object collection",
        "is_phase0_pass": False,
        "human_review_skipped_by_owner": True,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "active_clutter_slots": list(ACTIVE_CLUTTER_SLOTS),
        "target_successes": TARGET_SUCCESSES,
        "accepted_total": sum(accepted.values()),
        "attempts_total": sum(attempted.values()),
        "accepted_by_cell": dict(sorted(accepted.items())),
        "attempted_by_cell": dict(sorted(attempted.items())),
        "cells_short": short, "cells_over_quota": over,
        "quotas_met": not short and not over,
        "max_in_flight_per_cell": MAX_IN_FLIGHT_PER_CELL,
        "stop_reason": stop_reason,
        "started_utc": started_utc, "finished_utc": utc_now(),
        "elapsed_hours": round((time.monotonic() - started) / 3600, 3),
        "ledger_path": str(ledger_path.relative_to(ROOT)),
        "ledger_sha256": sha256_file(ledger_path) if ledger_path.exists() else None,
        "ledger_records": len(records),
        "disk_free_gib": round(shutil.disk_usage("/root").free / 2**30, 2),
    }
    document["payload_sha256"] = canonical_payload_sha256(document)
    write_immutable_create_only(collection / "closeout.json", document)
    print(json.dumps({k: document[k] for k in (
        "accepted_total", "attempts_total", "quotas_met", "stop_reason",
        "elapsed_hours", "cells_short", "disk_free_gib")}, indent=2))
    return 0 if document["quotas_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
