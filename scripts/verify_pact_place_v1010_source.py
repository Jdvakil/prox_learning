#!/usr/bin/env python3
"""V10.10: freeze the 144-row source population from the ledger and the files.

Produces a source manifest in exactly the shape the V10.9 converter consumes, so
the proven conversion path is reused unchanged.
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

from pact_place_v109_contract import CANONICAL_SENSOR_NAMES, SENSOR_ORDER_SHA256  # noqa: E402
from pact_place_v1010_contract import (  # noqa: E402
    ACTIVE_CLUTTER_SLOTS, COLLECTION_ROOT, CONTRACT_VERSION_V1010,
    QUOTA_PER_CELL, TARGET_SUCCESSES, WORK_ROOT, canonical_payload_sha256,
    cell_key, cells, empty_authorization, quotas, sha256_file,
    write_immutable_create_only,
)

SENSOR_SET = set(CANONICAL_SENSOR_NAMES)


def inspect(task: tuple[str, str, str]) -> dict[str, Any]:
    attempt_id, path_str, expected = task
    problems: list[str] = []
    path = Path(path_str)
    actual = sha256_file(path)
    if actual != expected:
        problems.append(f"h5 sha256 {actual} != ledger {expected}")
    with h5py.File(path, "r") as handle:
        keys = [k for k in handle if k.startswith("traj_")]
        if len(keys) != 1:
            return {"attempt_id": attempt_id, "timesteps": -1,
                    "h5_sha256": actual, "problems": [f"traj groups {keys}"]}
        group = handle[keys[0]]
        proximity = group["obs/proximity"]
        observed = set(proximity.keys())
        if observed != SENSOR_SET:
            problems.append(f"sensor mismatch: {sorted(observed ^ SENSOR_SET)[:3]}")
        lengths, dtypes, nonfinite, constant = set(), set(), 0, 0
        for name in sorted(observed & SENSOR_SET):
            dataset = proximity[name]
            dtypes.add(str(dataset.dtype))
            lengths.add(int(dataset.shape[0]))
            if dataset.shape[1:] != (4, 8, 8):
                problems.append(f"{name} tail {dataset.shape[1:]}")
            array = np.asarray(dataset[()], dtype=np.float32)
            nonfinite += int(not np.isfinite(array).all())
            constant += int(float(array.min()) == float(array.max()))
        if dtypes != {"float32"}:
            problems.append(f"dtypes {sorted(dtypes)}")
        if len(lengths) != 1:
            problems.append(f"sensors disagree on T: {sorted(lengths)}")
        if nonfinite:
            problems.append(f"{nonfinite} non-finite sensors")
        if constant:
            problems.append(f"{constant} constant sensors")
        timesteps = int(next(iter(lengths))) if len(lengths) == 1 else -1
        for key in ("actions/joint_pos", "obs/agent/qpos"):
            if key not in group:
                problems.append(f"missing {key}")
    videos = [v for v in sorted(path.parent.glob("episode_*_wrist_camera.mp4"))
              if "_depth" not in v.name]
    if len(videos) != 1:
        problems.append(f"{len(videos)} wrist MP4s")
    return {"attempt_id": attempt_id, "timesteps": timesteps,
            "h5_sha256": actual, "problems": problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path,
                        default=ROOT / WORK_ROOT / "source_manifest.json")
    args = parser.parse_args()

    ledger_path = ROOT / COLLECTION_ROOT / "ledger.jsonl"
    rows = [json.loads(x) for x in ledger_path.read_text().splitlines() if x.strip()]
    accepted = [r for r in rows if r["accepted"]]
    checks: list[tuple[str, bool, str]] = []
    problems: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        if not ok:
            problems.append(f"{name}: {detail}")

    quota = quotas()
    by_cell = collections.Counter(r["cell"] for r in accepted)
    check("accepted_total", len(accepted) == TARGET_SUCCESSES,
          f"{len(accepted)} vs {TARGET_SUCCESSES}")
    check("every_cell_exactly_quota",
          all(by_cell.get(c, 0) == quota[c] for c in quota),
          json.dumps({c: by_cell.get(c, 0) for c in quota if by_cell.get(c, 0) != quota[c]}))
    check("unique_attempt_ids", len({r["attempt_id"] for r in accepted}) == len(accepted), "")
    check("unique_task_seeds", len({r["task_seed_u32"] for r in accepted}) == len(accepted), "")
    check("all_strict_clean",
          all(r["clean_success"] and not r["defects"] for r in accepted), "")
    check("all_task_successful", all(r["task_success"] for r in accepted), "")
    check("zero_stability_events",
          all(int(r["clutter_stability_events"]) == 0 for r in accepted), "")
    check("single_identity_hash",
          len({r["identity_sha256"] for r in accepted}) == 1, "")
    disallowed = ("clutter", "mounted_fixture", "hazard_bar", "other_environment")
    check("zero_disallowed_contact",
          all(int((r["contact_class_totals"] or {}).get(k, 0)) == 0
              for r in accepted for k in disallowed), "")

    tasks = []
    for row in accepted:
        rel = row["trajectory_h5"]
        path = ROOT / rel if rel and not os.path.isabs(rel) else Path(rel or "")
        if not rel or not path.is_file():
            problems.append(f"missing HDF5 for {row['attempt_id'][:16]}")
            continue
        tasks.append((row["attempt_id"], str(path), row["trajectory_h5_sha256"]))
    check("all_h5_present", len(tasks) == len(accepted), f"{len(tasks)}/{len(accepted)}")

    print(f"inspecting {len(tasks)} episodes", flush=True)
    inspections: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(inspect, t): t[0] for t in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                out = future.result()
            except Exception as exc:  # noqa: BLE001
                out = {"attempt_id": futures[future], "timesteps": -1,
                       "h5_sha256": "", "problems": [f"raised {exc!r}"]}
            inspections[out["attempt_id"]] = out
            if done % 25 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}", flush=True)
    bad = {a: r["problems"] for a, r in inspections.items() if r["problems"]}
    check("every_episode_passes", not bad, f"{len(bad)} bad: {list(bad)[:2]}")
    lengths = [r["timesteps"] for r in inspections.values() if r["timesteps"] > 0]
    check("all_h5_readable", len(lengths) == len(tasks), "")

    order = {cell_key(*c): i for i, c in enumerate(cells())}
    ordered = sorted(accepted, key=lambda r: (order[r["cell"]],
                                              int(r["attempt_index"]), r["attempt_id"]))
    manifest_rows = []
    for index, row in enumerate(ordered):
        got = inspections.get(row["attempt_id"], {})
        manifest_rows.append({
            "act_episode_index": index,
            "attempt_id": row["attempt_id"], "attempt_index": int(row["attempt_index"]),
            "cell": row["cell"], "family_id": row["family_id"],
            "intrusion_side": row["intrusion_side"], "pose_id": row["pose_id"],
            "task_seed_u32": int(row["task_seed_u32"]),
            "episode_steps": int(row["episode_steps"]),
            "timesteps": int(got.get("timesteps", -1)),
            "row_sha256": row["row_sha256"],
            "identity_sha256": row["identity_sha256"],
            "trajectory_h5": row["trajectory_h5"],
            "trajectory_h5_sha256": row["trajectory_h5_sha256"],
            "per_object": row.get("per_object"),
            "min_pendant_clearance_m": row.get("min_pendant_clearance_m"),
        })
    document: dict[str, Any] = {
        **empty_authorization(),
        "schema_version": "pact_place_v1010_source_manifest_v1",
        "contract_version": CONTRACT_VERSION_V1010,
        "role": "frozen 144-row four-object source population",
        "is_phase0_pass": False,
        "ledger_path": str(ledger_path.relative_to(ROOT)),
        "ledger_sha256": sha256_file(ledger_path),
        "canonical_order_rule": "registered cell order, then attempt_index, then attempt_id",
        "sensor_order_sha256": SENSOR_ORDER_SHA256,
        "sensor_names": list(CANONICAL_SENSOR_NAMES),
        "active_clutter_slots": list(ACTIVE_CLUTTER_SLOTS),
        "counts": {"attempts": len(rows), "accepted": len(accepted),
                   "rejected": len(rows) - len(accepted),
                   "t_min": min(lengths) if lengths else None,
                   "t_max": max(lengths) if lengths else None,
                   "t_sum": sum(lengths)},
        "balance": {"by_cell": dict(sorted(by_cell.items())),
                    "quota_per_cell": QUOTA_PER_CELL,
                    "by_family": dict(sorted(collections.Counter(
                        r["family_id"] for r in accepted).items())),
                    "by_side": dict(sorted(collections.Counter(
                        r["intrusion_side"] for r in accepted).items())),
                    "by_pose": dict(sorted(collections.Counter(
                        r["pose_id"] for r in accepted).items()))},
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        "checks_failed": sum(1 for _, p, _ in checks if not p),
        "problems": problems,
        "verified": not problems,
        "rows": manifest_rows,
    }
    document["payload_sha256"] = canonical_payload_sha256(document)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_create_only(args.out, document)
    print(json.dumps({"verified": document["verified"],
                      "checks_failed": document["checks_failed"],
                      "problems": problems[:6], "counts": document["counts"],
                      "balance": document["balance"]["by_family"]}, indent=2))
    return 0 if document["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
