#!/usr/bin/env python3
"""V10.9 step 1: independently freeze the 141-row V10.8 source population.

Every number is re-derived from ``ledger.jsonl`` and the retained row files. The
previous V10.8 narrative and ``closeout.json`` are treated as untrusted: they are
loaded only to be *compared against*, never to supply a value.

Writes a create-only source manifest in canonical order (registered V10.8 cell
order, then attempt_index, then attempt_id).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v109_contract import (  # noqa: E402
    ACCEPTED_MIN_CLEARANCE_M,
    CANONICAL_SENSOR_NAMES,
    COLLECTION_ROOT,
    CONTRACT_VERSION_V109,
    LEDGER_SHA256,
    N_ACCEPTED,
    N_ATTEMPTS,
    N_REJECTED,
    N_SENSORS,
    PENDANT_CONTACT_ATTEMPT_ID,
    SENSOR_ORDER_SHA256,
    SOURCE_DATASET_ROOT,
    T_MAX,
    T_MIN,
    T_SUM,
    WORK_ROOT,
    canonical_payload_sha256,
    canonical_row_order,
    cell_key,
    cell_seed,
    cells,
    empty_authorization,
    quotas,
    sha256_file,
    write_immutable_create_only,
)

SENSOR_SET = set(CANONICAL_SENSOR_NAMES)


def _trajectory_group(handle: h5py.File) -> h5py.Group:
    keys = [k for k in handle if k.startswith("traj_")]
    if len(keys) != 1:
        raise RuntimeError(f"expected one trajectory group, found {keys}")
    return handle[keys[0]]


def inspect_episode(task: tuple[str, str, str]) -> dict[str, Any]:
    """Read one accepted episode end to end. Runs in a worker process."""
    attempt_id, h5_path, expected_sha = task
    problems: list[str] = []
    path = Path(h5_path)
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        problems.append(f"h5 sha256 {actual_sha} != ledger {expected_sha}")
    with h5py.File(path, "r") as handle:
        group = _trajectory_group(handle)
        proximity = group["obs/proximity"]
        observed = set(proximity.keys())
        if observed != SENSOR_SET:
            problems.append(
                f"sensor set mismatch: missing={sorted(SENSOR_SET - observed)} "
                f"unexpected={sorted(observed - SENSOR_SET)}"
            )
        if len(observed) != N_SENSORS:
            problems.append(f"{len(observed)} proximity sensors, expected {N_SENSORS}")
        timesteps: set[int] = set()
        nonfinite = 0
        constant: list[str] = []
        dtypes: set[str] = set()
        for name in sorted(observed & SENSOR_SET):
            dataset = proximity[name]
            dtypes.add(str(dataset.dtype))
            if dataset.shape[1:] != (4, 8, 8):
                problems.append(f"{name} shape {dataset.shape} tail != (4,8,8)")
            timesteps.add(int(dataset.shape[0]))
            array = np.asarray(dataset[()], dtype=np.float32)
            if not np.isfinite(array).all():
                nonfinite += 1
            if float(array.min()) == float(array.max()):
                constant.append(name)
        if dtypes != {"float32"}:
            problems.append(f"proximity dtypes {sorted(dtypes)} != float32")
        if len(timesteps) != 1:
            problems.append(f"sensors disagree on T: {sorted(timesteps)}")
        if nonfinite:
            problems.append(f"{nonfinite} sensors carry non-finite values")
        if constant:
            problems.append(f"constant sensors: {constant}")
        t = int(next(iter(timesteps))) if len(timesteps) == 1 else -1
        # action / qpos presence (decoded later by the converter)
        for key in ("actions/joint_pos", "obs/agent/qpos"):
            if key not in group:
                problems.append(f"missing {key}")
        wrist_params = "obs/sensor_param/wrist_camera/intrinsic_cv" in group
        if not wrist_params:
            problems.append("missing wrist_camera intrinsics")
        for name in sorted(SENSOR_SET):
            if f"obs/sensor_param/{name}/extrinsic_cv" not in group:
                problems.append(f"missing extrinsics for {name}")
                break
    videos = sorted(path.parent.glob("episode_*_wrist_camera.mp4"))
    videos = [v for v in videos if "_depth" not in v.name]
    if len(videos) != 1:
        problems.append(f"expected one wrist RGB MP4, found {len(videos)}")
    return {
        "attempt_id": attempt_id,
        "h5_path": h5_path,
        "h5_sha256": actual_sha,
        "timesteps": t,
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path,
                        default=ROOT / WORK_ROOT / "source_manifest.json")
    args = parser.parse_args()

    collection = ROOT / COLLECTION_ROOT
    ledger_path = collection / "ledger.jsonl"
    problems: list[str] = []
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append((name, bool(ok), detail))
        if not ok:
            problems.append(f"{name}: {detail}")
        return bool(ok)

    ledger_sha = sha256_file(ledger_path)
    check("ledger_sha256", ledger_sha == LEDGER_SHA256, f"{ledger_sha} vs {LEDGER_SHA256}")

    rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    check("ledger_records", len(rows) == N_ATTEMPTS, f"{len(rows)} vs {N_ATTEMPTS}")
    check("unique_attempt_ids", len({r["attempt_id"] for r in rows}) == len(rows),
          f"{len({r['attempt_id'] for r in rows})} unique of {len(rows)}")
    check("unique_row_sha256", len({r["row_sha256"] for r in rows}) == len(rows), "")
    check("unique_result_sha256", len({r["result_sha256"] for r in rows}) == len(rows), "")
    check("unique_task_seeds", len({r["task_seed_u32"] for r in rows}) == len(rows),
          f"{len({r['task_seed_u32'] for r in rows})} unique of {len(rows)}")

    accepted = [r for r in rows if r["accepted"]]
    rejected = [r for r in rows if not r["accepted"]]
    check("accepted_count", len(accepted) == N_ACCEPTED, f"{len(accepted)} vs {N_ACCEPTED}")
    check("rejected_count", len(rejected) == N_REJECTED, f"{len(rejected)} vs {N_REJECTED}")

    # strict-clean: accepted rows must have clean_success and no defects
    unclean = [r["attempt_id"] for r in accepted
               if not r["clean_success"] or r["defects"] or r["status"] != "complete"]
    check("accepted_all_strict_clean", not unclean, f"{len(unclean)} unclean: {unclean[:3]}")

    # seed streams reproduce from the frozen contract
    seed_mismatch = [
        r["attempt_id"] for r in rows
        if cell_seed(*r["cell"].split("|"), int(r["attempt_index"]))["seed_u32"]
        != int(r["task_seed_u32"])
    ]
    check("task_seeds_reproduce_from_cell_streams", not seed_mismatch,
          f"{len(seed_mismatch)} mismatched")

    # pendant involvement
    acc_pendant = [r["attempt_id"] for r in accepted if r["pendant_contact_frames"]]
    check("accepted_pendant_contact_rows_zero", not acc_pendant, f"{acc_pendant}")
    any_pendant = [r for r in rows if r["pendant_contact_frames"]
                   or r["contact_class_totals"].get("mounted_fixture", 0)]
    check("exactly_one_pendant_contact_attempt", len(any_pendant) == 1,
          f"{len(any_pendant)} rows")
    check("pendant_contact_attempt_id_matches",
          bool(any_pendant) and any_pendant[0]["attempt_id"] == PENDANT_CONTACT_ATTEMPT_ID,
          any_pendant[0]["attempt_id"] if any_pendant else "none")
    check("pendant_contact_attempt_rejected",
          bool(any_pendant) and not any_pendant[0]["accepted"], "")

    acc_clearances = [float(r["min_pendant_clearance_m"]) for r in accepted]
    check("accepted_min_clearance",
          bool(acc_clearances) and min(acc_clearances) == ACCEPTED_MIN_CLEARANCE_M,
          f"{min(acc_clearances) if acc_clearances else None!r}")

    # HDF5 files
    dataset_root = ROOT / SOURCE_DATASET_ROOT
    found = sorted(dataset_root.glob("rows/*/trajectory.h5"))
    check("h5_files_on_disk", len(found) == N_ACCEPTED, f"{len(found)} vs {N_ACCEPTED}")
    declared = {r["attempt_id"]: r for r in accepted}
    tasks = []
    for row in accepted:
        rel = row["trajectory_h5"]
        if not rel:
            problems.append(f"accepted row {row['attempt_id'][:16]} has no trajectory_h5")
            continue
        path = ROOT / rel if not os.path.isabs(rel) else Path(rel)
        if not path.is_file():
            problems.append(f"missing HDF5 for {row['attempt_id'][:16]}: {path}")
            continue
        tasks.append((row["attempt_id"], str(path), row["trajectory_h5_sha256"]))
    check("every_accepted_row_has_a_readable_h5", len(tasks) == N_ACCEPTED,
          f"{len(tasks)} resolvable of {N_ACCEPTED}")
    declared_paths = {t[1] for t in tasks}
    check("no_extra_h5_on_disk", declared_paths == {str(p) for p in found},
          f"{len(declared_paths ^ {str(p) for p in found})} unmatched")

    print(f"inspecting {len(tasks)} episodes with {args.workers} workers ...", flush=True)
    inspections: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(inspect_episode, t): t[0] for t in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - a worker failure is a finding
                aid = futures[future]
                result = {"attempt_id": aid, "h5_path": "", "h5_sha256": "",
                          "timesteps": -1, "problems": [f"inspection raised: {exc!r}"]}
            inspections[result["attempt_id"]] = result
            if done % 25 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}", flush=True)

    bad = {a: r["problems"] for a, r in inspections.items() if r["problems"]}
    check("every_episode_passes_schema_and_hash", not bad,
          f"{len(bad)} episodes with problems: {list(bad)[:2]}")

    lengths = [r["timesteps"] for r in inspections.values() if r["timesteps"] > 0]
    check("t_min", bool(lengths) and min(lengths) == T_MIN, f"{min(lengths) if lengths else None}")
    check("t_max", bool(lengths) and max(lengths) == T_MAX, f"{max(lengths) if lengths else None}")
    check("t_sum", sum(lengths) == T_SUM, f"{sum(lengths)} vs {T_SUM}")
    check("t_equals_episode_steps_plus_one",
          all(inspections[r["attempt_id"]]["timesteps"] == int(r["episode_steps"]) + 1
              for r in accepted if r["attempt_id"] in inspections), "")

    check("h5_sha256_unique",
          len({r["h5_sha256"] for r in inspections.values()}) == len(inspections), "")

    # distribution
    by_cell = collections.Counter(r["cell"] for r in accepted)
    by_family = collections.Counter(r["family_id"] for r in accepted)
    by_side = collections.Counter(r["intrusion_side"] for r in accepted)
    by_pose = collections.Counter(r["pose_id"] for r in accepted)
    quota = quotas()
    at_quota = sorted(c for c in quota if by_cell.get(c, 0) == quota[c])
    over_quota = {c: by_cell[c] - quota[c] for c in quota if by_cell.get(c, 0) > quota[c]}
    short = {c: quota[c] - by_cell.get(c, 0) for c in quota if by_cell.get(c, 0) < quota[c]}
    check("cells_exactly_at_quota_is_17", len(at_quota) == 17, f"{len(at_quota)}")
    check("cells_over_quota_is_2", len(over_quota) == 2, f"{len(over_quota)}: {over_quota}")
    check("cells_at_or_over_quota_is_19", len(at_quota) + len(over_quota) == 19,
          f"{len(at_quota) + len(over_quota)}")
    check("cells_short_is_5", len(short) == 5, f"{len(short)}: {short}")
    check("family_counts", dict(by_family) == {
        "F0_target_side_stagger": 39, "F1_inner_panel_stagger": 38,
        "F2_outer_panel_stagger": 38, "F3_aperture_side_stagger": 26},
        f"{dict(by_family)}")
    check("side_counts", dict(by_side) == {"left": 75, "right": 66}, f"{dict(by_side)}")
    check("sole_row_cell_has_one",
          by_cell.get("F3_aperture_side_stagger|right|neg5", 0) == 1, "")
    check("two_row_cell_has_two",
          by_cell.get("F3_aperture_side_stagger|right|pos5", 0) == 2, "")

    ordered = canonical_row_order(accepted)
    check("canonical_order_is_total", len(ordered) == N_ACCEPTED, "")

    manifest_rows = []
    for index, row in enumerate(ordered):
        inspection = inspections.get(row["attempt_id"], {})
        manifest_rows.append({
            "act_episode_index": index,
            "attempt_id": row["attempt_id"],
            "attempt_index": int(row["attempt_index"]),
            "cell": row["cell"],
            "family_id": row["family_id"],
            "intrusion_side": row["intrusion_side"],
            "pose_id": row["pose_id"],
            "task_seed_u32": int(row["task_seed_u32"]),
            "episode_steps": int(row["episode_steps"]),
            "timesteps": int(inspection.get("timesteps", -1)),
            "min_pendant_clearance_m": float(row["min_pendant_clearance_m"]),
            "pendant_contact_frames": int(row["pendant_contact_frames"]),
            "row_sha256": row["row_sha256"],
            "result_sha256": row["result_sha256"],
            "trajectory_h5": row["trajectory_h5"],
            "trajectory_h5_sha256": row["trajectory_h5_sha256"],
        })

    document: dict[str, Any] = {
        **empty_authorization(),
        "schema_version": "pact_place_v109_source_manifest_v1",
        "contract_version": CONTRACT_VERSION_V109,
        "role": "frozen 141-row V10.8 source population for V10.9 conversion",
        "is_phase0_pass": False,
        "v107_phase0_result": "failed_8_of_24_permanently_closed",
        "v108_stop_reason": "owner_instructed_early_stop",
        "authoritative_population": "the 141 accepted ledger rows",
        "not_authoritative": [
            "diagnostics_output/pact_place_v108_collection/collection.json (stale smoke record)",
            "a glob of assets/datagen/pact_place_corridor_v10_8/rows",
        ],
        "ledger_path": str(ledger_path.relative_to(ROOT)),
        "ledger_sha256": ledger_sha,
        "canonical_order_rule":
            "registered V10.8 cell order, then attempt_index, then attempt_id",
        "sensor_order_sha256": SENSOR_ORDER_SHA256,
        "sensor_names": list(CANONICAL_SENSOR_NAMES),
        "counts": {
            "attempts": len(rows), "accepted": len(accepted), "rejected": len(rejected),
            "t_min": min(lengths) if lengths else None,
            "t_max": max(lengths) if lengths else None,
            "t_sum": sum(lengths),
            "sensor_windows": sum(lengths) * N_SENSORS,
        },
        "underrepresentation": {
            "note": "USED AS COLLECTED. No row dropped, no F3 resampling, "
                    "no duplication, no family weights.",
            "by_family": dict(sorted(by_family.items())),
            "by_side": dict(sorted(by_side.items())),
            "by_pose": dict(sorted(by_pose.items())),
            "by_cell": dict(sorted(by_cell.items())),
            "quota_by_cell": quota,
            "cells_exactly_at_quota": len(at_quota),
            "cells_over_quota": over_quota,
            "cells_short": short,
            "sole_row_cell": "F3_aperture_side_stagger|right|neg5",
            "two_row_cell": "F3_aperture_side_stagger|right|pos5",
        },
        "pendant": {
            "accepted_rows_with_pendant_contact": len(acc_pendant),
            "attempts_with_any_pendant_involvement": len(any_pendant),
            "pendant_contact_attempt_id": PENDANT_CONTACT_ATTEMPT_ID,
            "accepted_min_clearance_m": min(acc_clearances) if acc_clearances else None,
            "accepted_max_clearance_m": max(acc_clearances) if acc_clearances else None,
        },
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        "checks_total": len(checks),
        "checks_failed": sum(1 for _, p, _ in checks if not p),
        "problems": problems,
        "verified": not problems,
        "rows": manifest_rows,
    }
    document["payload_sha256"] = canonical_payload_sha256(document)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = write_immutable_create_only(args.out, document)
    print(json.dumps({
        "verified": document["verified"],
        "checks_total": document["checks_total"],
        "checks_failed": document["checks_failed"],
        "problems": problems[:10],
        "counts": document["counts"],
        "payload_sha256": document["payload_sha256"],
        "raw_file_sha256": written.get("raw_file_sha256"),
        "path": str(args.out.relative_to(ROOT)),
    }, indent=2))
    return 0 if document["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
