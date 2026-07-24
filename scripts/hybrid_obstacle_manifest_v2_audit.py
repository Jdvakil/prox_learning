#!/usr/bin/env python3
"""Invariance audit for the manifest-driven hybrid-obstacle smoke runs.

Compares Run A (1 worker), Run B (4 workers) and Run C (4 workers, interrupted
and resumed) BY EPISODE ID -- never by file order.

Tolerances are encoded here BEFORE Runs A/B/C execute and are not changed
afterwards. A tolerance match is insufficient whenever any discrete event
differs: a different grasp, retry count, rejection sequence, planner phase path,
success/failure outcome, hazard geometry, object identity or collision outcome
fails invariance outright, regardless of how close the arrays are.

Usage:
    python scripts/hybrid_obstacle_manifest_v2_audit.py \
        --run A=/path/run_a B=/path/run_b C=/path/run_c \
        --out diagnostics_output/hybrid_obstacle_seeding/invariance_report.json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules" / "molmospaces"))

# --------------------------------------------------------------------------- #
# FROZEN TOLERANCES. Encoded before execution; never relaxed afterwards.
# Exact (bit-identical) equality is preferred and is reported separately.
# --------------------------------------------------------------------------- #
TOLERANCES = {
    "qpos": 1e-6,
    "actions": 1e-6,
    "object_pose": 1e-7,
    "robot_pose": 1e-7,
    "proximity_depth": 1e-5,
}

#: Fields that must match EXACTLY. Any difference fails invariance; there is no
#: tolerance path for a discrete event.
EXACT_MATCH_FIELDS = [
    "episode_id",
    "manifest_row_sha256",
    "manifest_version",
    "candidate_index",
    "hazard_present",
    "stratum_rank",
    "split",
    "scene_template_id",
    "seed_map",
    "selected_object",
    "selected_grasp",
    "obstacle_theta",
    "observed_hazard_present",
    "robot_initial_qpos",
    "object_initial_pose",
    "retry_count",
    "retry_reasons",
    "planner_phase_path",
    "behavior_class",
    "status",
    "episode_spec_sha256",
    "sensor_order_sha256",
]

#: Discrete events that always fail invariance when they differ.
DISCRETE_EVENT_FIELDS = [
    "selected_grasp",
    "retry_count",
    "retry_reasons",
    "planner_phase_path",
    "status",
    "observed_hazard_present",
    "obstacle_theta",
    "selected_object",
]

#: The 40-sensor proximity depth stack lives directly under this group. Other
#: datasets mention link names too (per-sensor object image points), so matching
#: on the group is what actually identifies the sensor order.
PROXIMITY_GROUP = "obs/proximity/"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_run(run_dir: Path) -> dict[str, Any]:
    """Collect every finalised row of a run, keyed by episode ID."""
    rows_dir = run_dir / "rows"
    if not rows_dir.exists():
        raise SystemExit(f"{run_dir} has no rows/ directory")

    episodes: dict[str, dict[str, Any]] = {}
    for row_dir in sorted(rows_dir.iterdir()):
        outcome_path = row_dir / "outcome.json"
        if not outcome_path.exists():
            continue
        outcome = json.loads(outcome_path.read_text())
        observations = outcome.get("observations", {}) or {}
        record: dict[str, Any] = {
            "episode_id": outcome["episode_id"],
            "manifest_row_sha256": outcome["row_sha256"],
            "manifest_version": outcome["manifest_version"],
            "candidate_index": outcome["candidate_index"],
            "hazard_present": outcome["hazard_present"],
            "stratum_rank": outcome["stratum_rank"],
            "split": outcome["split"],
            "status": outcome["status"],
            "retry_count": outcome.get("retry_count", 0),
            "retry_reasons": [
                entry.get("reason") for entry in outcome.get("retry_history", []) or []
            ],
            "worker_id_descriptive": outcome.get("worker_id_descriptive"),
            "seed_map": (outcome.get("seed_contract", {}) or {}).get("seed_map"),
            "scene_template_id": observations.get("scene_template_id"),
            "selected_object": observations.get("selected_object"),
            "selected_grasp": observations.get("selected_grasp"),
            "obstacle_theta": observations.get("obstacle_theta"),
            "observed_hazard_present": observations.get("observed_hazard_present"),
            "robot_initial_qpos": observations.get("robot_initial_qpos"),
            "object_initial_pose": observations.get("object_initial_pose"),
            "planner_phase_path": observations.get("planner_phase_path"),
            "behavior_class": observations.get("behavior_class"),
            "episode_spec_sha256": observations.get("episode_spec_sha256"),
            "sensor_order_sha256": None,
            "arrays": {},
            "h5_schema": {},
            "trajectory_sha256": None,
        }

        h5_path = row_dir / "trajectory.h5"
        if h5_path.exists():
            record.update(load_h5(h5_path))
        episodes[record["episode_id"]] = record

    summary_path = run_dir / "collection_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return {"episodes": episodes, "summary": summary, "dir": str(run_dir)}


def load_h5(path: Path) -> dict[str, Any]:
    """Read the scientific arrays and the schema shape from a published row."""
    arrays: dict[str, np.ndarray] = {}
    schema: dict[str, list[int]] = {}
    extra: dict[str, Any] = {}

    with h5py.File(path, "r") as handle:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                schema[name] = list(obj.shape)
                if obj.dtype.kind in "fiu" and obj.size and obj.size < 50_000_000:
                    arrays[name] = np.asarray(obj[()])

        handle.visititems(visit)

        manifest_group = handle.get("manifest")
        if manifest_group is not None and "sensor_order_sha256" in manifest_group:
            extra["sensor_order_sha256"] = _decode(manifest_group["sensor_order_sha256"][()])
        for attr in ("episode_id", "manifest_row_sha256"):
            if attr in handle.attrs:
                extra.setdefault(f"h5_attr_{attr}", str(handle.attrs[attr]))

    proximity_names = sorted(name for name in schema if PROXIMITY_GROUP in name)
    extra["ordered_sensor_names"] = [n.rsplit("/", 1)[-1] for n in proximity_names]
    extra["sensor_order_names_sha256"] = hashlib.sha256(
        "\x1f".join(extra["ordered_sensor_names"]).encode()
    ).hexdigest()

    # Full qpos+action trajectory hash: two distinct episode IDs sharing this
    # value would be the exact replica signature that invalidated the previous
    # collection.
    trajectory_digest = hashlib.sha256()
    for name in sorted(arrays):
        base = name.rsplit("/", 1)[-1]
        if base in ("qpos", "qvel", "actions"):
            trajectory_digest.update(name.encode())
            trajectory_digest.update(np.ascontiguousarray(arrays[name]).tobytes())
    extra["trajectory_sha256"] = trajectory_digest.hexdigest()

    extra["arrays"] = arrays
    extra["h5_schema"] = schema
    return extra


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def _tolerance_for(dataset_name: str) -> float:
    base = dataset_name.rsplit("/", 1)[-1].lower()
    if PROXIMITY_GROUP in dataset_name or "depth" in base:
        return TOLERANCES["proximity_depth"]
    if "action" in base:
        return TOLERANCES["actions"]
    if base in ("qpos", "qvel"):
        return TOLERANCES["qpos"]
    if "pose" in base or base.startswith("obj_"):
        return TOLERANCES["object_pose"]
    # Anything unclassified gets the tightest tolerance rather than the loosest.
    return TOLERANCES["object_pose"]


def compare_arrays(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, Any]:
    names = sorted(set(left) | set(right))
    bit_identical = True
    within_tolerance = True
    failures: list[dict[str, Any]] = []
    max_deltas: dict[str, float] = {}

    for name in names:
        if name not in left or name not in right:
            bit_identical = within_tolerance = False
            failures.append({"dataset": name, "reason": "present in only one run"})
            continue
        a, b = left[name], right[name]
        if a.shape != b.shape:
            bit_identical = within_tolerance = False
            failures.append(
                {"dataset": name, "reason": f"shape {a.shape} != {b.shape}"}
            )
            continue
        # NaN is used as padding in several per-sensor point arrays. Plain
        # array_equal reports NaN != NaN, and a naive max|a-b| over them yields
        # NaN, which would then slip through a `delta > tolerance` test because
        # every NaN comparison is False. Both are handled explicitly: the NaN
        # PATTERN must match, and the magnitude check runs over finite entries.
        if a.dtype.kind == "f":
            nan_a, nan_b = np.isnan(a), np.isnan(b)
            if not np.array_equal(nan_a, nan_b):
                bit_identical = within_tolerance = False
                failures.append({"dataset": name, "reason": "NaN pattern differs"})
                continue
            if np.array_equal(a, b, equal_nan=True):
                continue
            finite = ~nan_a
        else:
            if np.array_equal(a, b):
                continue
            finite = np.ones(a.shape, dtype=bool)

        bit_identical = False
        if a.dtype.kind not in "fiu":
            within_tolerance = False
            failures.append({"dataset": name, "reason": "non-numeric mismatch"})
            continue

        if not finite.any():
            continue
        delta = float(
            np.max(np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64)))
        )
        max_deltas[name] = delta
        tolerance = _tolerance_for(name)
        if not np.isfinite(delta) or delta > tolerance:
            within_tolerance = False
            failures.append(
                {
                    "dataset": name,
                    "reason": "exceeds frozen tolerance"
                    if np.isfinite(delta)
                    else "non-finite delta",
                    "max_abs_delta": delta,
                    "tolerance": tolerance,
                }
            )
    return {
        "bit_identical": bit_identical,
        "within_tolerance": within_tolerance,
        "max_abs_deltas": max_deltas,
        "failures": failures,
    }


def compare_episode(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    field_mismatches = []
    discrete_mismatches = []
    for field in EXACT_MATCH_FIELDS:
        if left.get(field) != right.get(field):
            entry = {"field": field, "left": left.get(field), "right": right.get(field)}
            field_mismatches.append(entry)
            if field in DISCRETE_EVENT_FIELDS:
                discrete_mismatches.append(entry)

    schema_match = left.get("h5_schema") == right.get("h5_schema")
    sensor_match = left.get("ordered_sensor_names") == right.get("ordered_sensor_names")
    array_result = compare_arrays(left.get("arrays", {}), right.get("arrays", {}))

    invariant = (
        not field_mismatches
        and not discrete_mismatches
        and schema_match
        and sensor_match
        and array_result["within_tolerance"]
    )
    return {
        "episode_id": left["episode_id"],
        "candidate_index": left["candidate_index"],
        "field_mismatches": field_mismatches,
        "discrete_event_mismatches": discrete_mismatches,
        "h5_schema_match": schema_match,
        "sensor_order_match": sensor_match,
        "arrays": array_result,
        "bit_identical": array_result["bit_identical"],
        "invariant": invariant,
        "worker_ids": {
            "left": left.get("worker_id_descriptive"),
            "right": right.get("worker_id_descriptive"),
        },
    }


def compare_runs(name: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_ids = set(left["episodes"])
    right_ids = set(right["episodes"])
    per_episode = [
        compare_episode(left["episodes"][eid], right["episodes"][eid])
        for eid in sorted(left_ids & right_ids)
    ]
    return {
        "pair": name,
        "episode_id_sets_match": left_ids == right_ids,
        "only_in_left": sorted(left_ids - right_ids),
        "only_in_right": sorted(right_ids - left_ids),
        "episodes_compared": len(per_episode),
        "all_invariant": bool(per_episode)
        and all(e["invariant"] for e in per_episode)
        and left_ids == right_ids,
        "all_bit_identical": bool(per_episode) and all(e["bit_identical"] for e in per_episode),
        "episodes": per_episode,
    }


def replica_audit(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """No two distinct episode IDs may share a spec hash or a trajectory hash.

    This is the direct test for the failure that invalidated the previous
    collection: 50 replica classes of three members each.
    """
    findings = {}
    for label, run in runs.items():
        by_spec: dict[str, list[str]] = {}
        by_traj: dict[str, list[str]] = {}
        for episode_id, record in run["episodes"].items():
            spec = record.get("episode_spec_sha256")
            traj = record.get("trajectory_sha256")
            if spec:
                by_spec.setdefault(spec, []).append(episode_id)
            if traj:
                by_traj.setdefault(traj, []).append(episode_id)
        spec_collisions = {k: sorted(v) for k, v in by_spec.items() if len(v) > 1}
        traj_collisions = {k: sorted(v) for k, v in by_traj.items() if len(v) > 1}
        findings[label] = {
            "distinct_episode_ids": len(run["episodes"]),
            "distinct_spec_hashes": len(by_spec),
            "distinct_trajectory_hashes": len(by_traj),
            "spec_hash_collisions": spec_collisions,
            "trajectory_hash_collisions": traj_collisions,
            "largest_replica_class": max(
                [len(v) for v in list(spec_collisions.values()) + list(traj_collisions.values())],
                default=1,
            ),
            "clean": not spec_collisions and not traj_collisions,
        }
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--expected-rows", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs: dict[str, dict[str, Any]] = {}
    for spec in args.run:
        label, _, directory = spec.partition("=")
        runs[label] = load_run(Path(directory))

    labels = sorted(runs)
    pairs = [
        (f"{a}_vs_{b}", a, b) for i, a in enumerate(labels) for b in labels[i + 1 :]
    ]
    comparisons = [compare_runs(name, runs[a], runs[b]) for name, a, b in pairs]

    per_run = {}
    for label, run in runs.items():
        episodes = run["episodes"]
        reconciliation = (run["summary"].get("row_reconciliation") or {})
        per_run[label] = {
            "dir": run["dir"],
            "episodes_finalized": len(episodes),
            "succeeded": sorted(
                e for e, r in episodes.items() if r["status"] == "success"
            ),
            "failed": sorted(e for e, r in episodes.items() if r["status"] != "success"),
            "expected_rows": args.expected_rows,
            "reconciles_exactly_once": len(episodes) == args.expected_rows,
            "summary_complete": run["summary"].get("complete"),
            "row_reconciliation_ok": reconciliation.get("ok"),
            "worker_assignment": {
                episode_id: record.get("worker_id_descriptive")
                for episode_id, record in sorted(episodes.items())
            },
        }

    worker_assignments_differ = (
        len({json.dumps(v["worker_assignment"], sort_keys=True) for v in per_run.values()}) > 1
    )

    report = {
        "schema": "hybrid_obstacle_manifest_v2_invariance_report",
        "frozen_tolerances": TOLERANCES,
        "exact_match_fields": EXACT_MATCH_FIELDS,
        "discrete_event_fields": DISCRETE_EVENT_FIELDS,
        "per_run": per_run,
        "comparisons": comparisons,
        "replica_audit": replica_audit(runs),
        "worker_scheduling_changed_between_runs": worker_assignments_differ,
    }

    all_invariant = all(c["all_invariant"] for c in comparisons)
    all_reconciled = all(v["reconciles_exactly_once"] for v in per_run.values())
    replicas_clean = all(v["clean"] for v in report["replica_audit"].values())
    same_id_sets = all(c["episode_id_sets_match"] for c in comparisons)

    report["verdict"] = {
        "all_pairs_invariant": all_invariant,
        "all_runs_reconcile_exactly_once": all_reconciled,
        "no_replica_classes": replicas_clean,
        "same_completed_episode_id_set": same_id_sets,
        "all_bit_identical": all(c["all_bit_identical"] for c in comparisons),
        "pass": bool(all_invariant and all_reconciled and replicas_clean and same_id_sets),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"wrote {args.out}")
    for label, values in sorted(per_run.items()):
        print(
            f"  run {label}: {values['episodes_finalized']} rows finalised, "
            f"{len(values['succeeded'])} succeeded, reconciled="
            f"{values['reconciles_exactly_once']}"
        )
    for comparison in comparisons:
        print(
            f"  {comparison['pair']}: invariant={comparison['all_invariant']} "
            f"bit_identical={comparison['all_bit_identical']} "
            f"episodes={comparison['episodes_compared']}"
        )
    print(f"  VERDICT pass={report['verdict']['pass']}")
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
