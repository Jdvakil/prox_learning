#!/usr/bin/env python3
"""Independent re-verification of the hybrid-obstacle source collection.

This is a *second, from-scratch implementation* of the checks that decide the
canonical-dataset question. It deliberately shares no code with
``hybrid_obstacle_dataset_audit.py`` so that agreement between the two is
evidence, not a shared bug.

It answers four questions and nothing else:

1. Does the immutable source collection still hash to the expected content-tree
   SHA-256 recovered from the committed clean-retrain manifest?
2. How many *distinct* trajectories does the collection actually contain, under
   two independent notions of identity?
3. What are the hazard / success cross-tabulations and their exact
   (Clopper-Pearson) 95% intervals?
4. Does the recorded hazard label agree with the scene geometry that was
   actually compiled into the model on every trajectory?

The source collection is opened read-only and never written.

Usage::

    python scripts/hybrid_obstacle_reverify.py \
        --run_dir /root/act_retrain_assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260724_183407 \
        --expected_tree_sha256 09c98aee08d015b3a561b08674415df9a4ed398186940207f41ef384251cdf24 \
        --output diagnostics_output/hybrid_obstacle_dataset/independent_reverification.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCHEMA_VERSION = "hybrid_obstacle_independent_reverification_v1"

# The 40 proximity streams the hybrid skin is expected to publish, in the
# order the collection writes them.
EXPECTED_SENSOR_COUNTS = {
    "link1": 7,
    "link2": 7,
    "link3": 5,
    "link4": 5,
    "link5_back": 6,
    "link5_front": 4,
    "link6": 6,
}
EXPECTED_SENSOR_TOTAL = 40

# Fields the converter and the audits require to be present on every trajectory.
REQUIRED_LEAVES = (
    "obs/agent/qpos",
    "actions/joint_pos",
    "obs/extra/tcp_pose",
    "obs/extra/obj_start",
    "obs/extra/task_info",
    "env_states/articulations/panda",
    "success",
    "fail",
)

# Companion media written alongside every episode.
EXPECTED_MEDIA_SUFFIXES = (
    "exo_camera_1",
    "exo_camera_1_depth",
    "sensors_depth8_heatmap",
    "sensors_rgb256",
    "wrist_camera",
    "wrist_camera_depth",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_tree(root: Path) -> tuple[str, int, int, list[dict[str, Any]]]:
    """Content-tree hash over sorted ``relpath\\0sha256\\n`` records.

    Same construction as the committed provenance tool, reimplemented here so
    the agreement is an independent check rather than a shared call.
    """
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "relpath": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    records.sort(key=lambda r: r["relpath"])
    payload = "".join(f"{r['relpath']}\0{r['sha256']}\n" for r in records)
    tree = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return tree, len(records), sum(r["bytes"] for r in records), records


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> dict[str, float]:
    """Exact binomial 95% interval. Uses the Beta quantile identity."""
    from scipy.stats import beta

    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return {"k": k, "n": n, "point": k / n if n else float("nan"), "lower": lower, "upper": upper}


def leaf_paths(group: h5py.Group, prefix: str = "") -> list[str]:
    out: list[str] = []
    for key in sorted(group.keys()):
        obj = group[key]
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(obj, h5py.Dataset):
            out.append(path)
        else:
            out.extend(leaf_paths(obj, path))
    return out


def scan_trajectory(group: h5py.Group) -> dict[str, Any]:
    """Read one trajectory group and derive both identity hashes plus labels."""
    leaves = leaf_paths(group)

    # Identity A: every leaf dataset, name and bytes. Sensitive to derived
    # float-noise fields, so it is the most permissive notion of "distinct".
    full = hashlib.sha256()
    for leaf in leaves:
        full.update(leaf.encode("utf-8"))
        full.update(np.ascontiguousarray(group[leaf][()]).tobytes())

    # Identity B: the recorded state and the commanded action only. This is the
    # notion that matters for whether two trajectories are the same episode.
    dynamics = hashlib.sha256()
    for leaf in ("obs/agent/qpos", "actions/joint_pos"):
        dynamics.update(np.ascontiguousarray(group[leaf][()]).tobytes())

    scene = json.loads(bytes(group["obs_scene"][()]).decode("utf-8"))
    params = scene["scene_params"]

    fail = np.asarray(group["fail"][()])
    success = np.asarray(group["success"][()])
    # The committed converter's criterion: episode failed iff fail[-1] is set.
    failed = bool(fail[-1]) if fail.size else False

    proximity = [leaf.split("/")[-1] for leaf in leaves if leaf.startswith("obs/proximity/")]
    prox_shapes = {group[f"obs/proximity/{name}"].shape[1:] for name in proximity}

    hazard = bool(params["protrusion_present"])
    n_boxes = len(params["obstacle_aabbs"])
    has_bar_geometry = params.get("protr_center") is not None and params.get("protr_half") is not None
    # A hazard scene must carry the extra protrusion box; a clear scene must not.
    geometry_agrees = (
        hazard
        and n_boxes == 7
        and has_bar_geometry
        and bool(params.get("protr_name"))
        and params.get("protr_wall") in ("left", "right")
    ) or (not hazard and n_boxes == 6 and not has_bar_geometry)

    return {
        "T": int(fail.shape[0]),
        "content_sha256": full.hexdigest(),
        "dynamics_sha256": dynamics.hexdigest(),
        "scene_params_sha256": hashlib.sha256(
            json.dumps(params, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "failed": failed,
        "successful": not failed,
        "success_last": bool(success[-1]) if success.size else False,
        "hazard_present": hazard,
        "n_obstacle_aabbs": n_boxes,
        "hazard_label_matches_geometry": bool(geometry_agrees),
        "proximity_stream_count": len(proximity),
        "proximity_stream_names": proximity,
        "proximity_trailing_shapes": sorted(str(s) for s in prox_shapes),
        "missing_required_leaves": [leaf for leaf in REQUIRED_LEAVES if leaf not in leaves],
        "target_uid": params.get("target_uid"),
        "leaf_count": len(leaves),
    }


def strata(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Collapse rows into equivalence classes under ``key`` and describe them."""
    classes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        classes[row[key]].append(row)
    successful_classes = {
        k: v for k, v in classes.items() if any(r["successful"] for r in v)
    }
    inconsistent = [k for k, v in classes.items() if len({r["hazard_present"] for r in v}) > 1]
    hazard_classes = sum(1 for v in successful_classes.values() if v[0]["hazard_present"])
    return {
        "identity_field": key,
        "distinct_classes_all_written": len(classes),
        "multiplicity_histogram": {
            str(m): c for m, c in sorted(Counter(len(v) for v in classes.values()).items())
        },
        "distinct_classes_successful": len(successful_classes),
        "distinct_successful_hazard_present": hazard_classes,
        "distinct_successful_hazard_absent": len(successful_classes) - hazard_classes,
        "classes_with_inconsistent_hazard_label": len(inconsistent),
        "replica_groups": sorted(
            sorted(r["trajectory_id"] for r in v) for v in classes.values() if len(v) > 1
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--expected_tree_sha256", required=True)
    parser.add_argument("--target_hazard_present", type=int, default=75)
    parser.add_argument("--target_hazard_absent", type=int, default=25)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"run_dir not found: {run_dir}", file=sys.stderr)
        return 2

    tree, file_count, total_bytes, _per_file = content_tree(run_dir)
    tree_matches = tree == args.expected_tree_sha256

    rows: list[dict[str, Any]] = []
    h5_open_failures: list[str] = []
    h5_hashes: dict[str, str] = {}
    media_missing: list[str] = []

    for house_dir in sorted(run_dir.glob("house_*"), key=lambda p: int(p.name.split("_")[1])):
        h5_path = house_dir / "trajectories_batch_1_of_1.h5"
        h5_hashes[house_dir.name] = sha256_file(h5_path)
        try:
            handle = h5py.File(h5_path, "r")
        except Exception as exc:  # pragma: no cover - a real corruption would land here
            h5_open_failures.append(f"{house_dir.name}: {exc!r}")
            continue
        with handle as f:
            for traj_key in sorted(f.keys(), key=lambda s: int(s.split("_")[1])):
                row = scan_trajectory(f[traj_key])
                row["house"] = house_dir.name
                row["house_index"] = int(house_dir.name.split("_")[1])
                row["traj_key"] = traj_key
                row["trajectory_id"] = f"{house_dir.name}/{traj_key}"
                row["source_h5_sha256"] = h5_hashes[house_dir.name]
                rows.append(row)

                index = int(traj_key.split("_")[1])
                for suffix in EXPECTED_MEDIA_SUFFIXES:
                    media = house_dir / f"episode_{index:08d}_{suffix}_batch_1_of_1.mp4"
                    if not media.is_file() or media.stat().st_size == 0:
                        media_missing.append(str(media.relative_to(run_dir)))

    successful = [r for r in rows if r["successful"]]

    def count(pred) -> int:
        return sum(1 for r in rows if pred(r))

    written_hazard = count(lambda r: r["hazard_present"])
    succ_hazard = sum(1 for r in successful if r["hazard_present"])
    succ_clear = len(successful) - succ_hazard

    identity_dynamics = strata(rows, "dynamics_sha256")
    identity_content = strata(rows, "content_sha256")
    identity_scene = strata(rows, "scene_params_sha256")

    # Feasibility is decided on the *most permissive* identity that still refuses
    # to count a bit-identical episode twice, so the conclusion cannot be an
    # artifact of choosing a strict notion of "distinct".
    best_present = max(
        identity_dynamics["distinct_successful_hazard_present"],
        identity_content["distinct_successful_hazard_present"],
    )
    best_absent = max(
        identity_dynamics["distinct_successful_hazard_absent"],
        identity_content["distinct_successful_hazard_absent"],
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "verifier": {
            "independent_of": "scripts/hybrid_obstacle_dataset_audit.py",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "hash_algorithm": "SHA-256",
            "tree_hash_construction": "sha256 over sorted 'relpath\\0sha256\\n' records",
        },
        "source_collection": {
            "expected_content_tree_sha256": args.expected_tree_sha256,
            "recomputed_content_tree_sha256": tree,
            "matches": tree_matches,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "h5_sha256_by_house": h5_hashes,
        },
        "h5_integrity": {
            "h5_files": len(h5_hashes),
            "h5_open_failures": h5_open_failures,
            "trajectories_read": len(rows),
            "trajectories_with_missing_required_leaves": [
                r["trajectory_id"] for r in rows if r["missing_required_leaves"]
            ],
            "unique_trajectory_ids": len({r["trajectory_id"] for r in rows}) == len(rows),
            "companion_media_missing": media_missing,
        },
        "proximity": {
            "expected_total": EXPECTED_SENSOR_TOTAL,
            "expected_per_link": EXPECTED_SENSOR_COUNTS,
            "distinct_stream_counts": sorted({r["proximity_stream_count"] for r in rows}),
            "distinct_trailing_shapes": sorted(
                {shape for r in rows for shape in r["proximity_trailing_shapes"]}
            ),
            "stream_order_consistent": len({tuple(r["proximity_stream_names"]) for r in rows}) == 1,
            "stream_order_sha256": hashlib.sha256(
                "\0".join(rows[0]["proximity_stream_names"]).encode("utf-8")
            ).hexdigest()
            if rows
            else None,
        },
        "hazard_label_vs_geometry": {
            "checked": len(rows),
            "disagreements": [
                r["trajectory_id"] for r in rows if not r["hazard_label_matches_geometry"]
            ],
            "aabb_count_by_hazard": {
                f"hazard_{h}_boxes_{n}": c
                for (h, n), c in sorted(
                    Counter((r["hazard_present"], r["n_obstacle_aabbs"]) for r in rows).items(),
                    key=lambda kv: (kv[0][0], kv[0][1]),
                )
            },
        },
        "counts": {
            "written": len(rows),
            "successful": len(successful),
            "failed": len(rows) - len(successful),
            "written_hazard_present": written_hazard,
            "written_hazard_absent": len(rows) - written_hazard,
            "successful_hazard_present": succ_hazard,
            "successful_hazard_absent": succ_clear,
            "failed_hazard_present": written_hazard - succ_hazard,
            "failed_hazard_absent": (len(rows) - written_hazard) - succ_clear,
            "by_house": dict(sorted(Counter(r["house"] for r in rows).items())),
            "by_house_x_hazard": {
                f"{h}|hazard_{z}": c
                for (h, z), c in sorted(
                    Counter((r["house"], r["hazard_present"]) for r in rows).items()
                )
            },
            "by_house_x_success": {
                f"{h}|success_{s}": c
                for (h, s), c in sorted(
                    Counter((r["house"], r["successful"]) for r in rows).items()
                )
            },
            "single_target_uid": sorted({r["target_uid"] for r in rows}),
        },
        "identity_analysis": {
            "dynamics_qpos_and_joint_pos": identity_dynamics,
            "all_leaf_datasets": identity_content,
            "scene_params": identity_scene,
        },
        "exact_binomial_intervals": {
            "note": (
                "Clopper-Pearson 95%. The as_written rates are inflated by exact replicas "
                "and are NOT valid sampling intervals; only the distinct rates carry "
                "independent information."
            ),
            "as_written": {
                "written_hazard_rate": clopper_pearson(written_hazard, len(rows)),
                "successful_hazard_rate": clopper_pearson(succ_hazard, len(successful)),
                "success_given_hazard_present": clopper_pearson(succ_hazard, written_hazard),
                "success_given_hazard_absent": clopper_pearson(
                    succ_clear, len(rows) - written_hazard
                ),
            },
            "distinct": _distinct_intervals(rows, "dynamics_sha256"),
        },
        "canonical_target_feasibility": {
            "target_hazard_present": args.target_hazard_present,
            "target_hazard_absent": args.target_hazard_absent,
            "available_hazard_present_most_permissive": best_present,
            "available_hazard_absent_most_permissive": best_absent,
            "shortfall_hazard_present": max(0, args.target_hazard_present - best_present),
            "shortfall_hazard_absent": max(0, args.target_hazard_absent - best_absent),
            "feasible": best_present >= args.target_hazard_present
            and best_absent >= args.target_hazard_absent,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    summary = {
        "tree_hash_matches": tree_matches,
        "written": len(rows),
        "distinct_dynamics_classes": identity_dynamics["distinct_classes_all_written"],
        "distinct_successful": identity_dynamics["distinct_classes_successful"],
        "distinct_successful_hazard_present": identity_dynamics[
            "distinct_successful_hazard_present"
        ],
        "distinct_successful_hazard_absent": identity_dynamics[
            "distinct_successful_hazard_absent"
        ],
        "hazard_geometry_disagreements": len(report["hazard_label_vs_geometry"]["disagreements"]),
        "feasible": report["canonical_target_feasibility"]["feasible"],
        "output": str(args.output),
    }
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")

    if not tree_matches:
        return 2
    return 0 if report["canonical_target_feasibility"]["feasible"] else 3


def _distinct_intervals(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Intervals computed over one representative per identity class."""
    seen: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: r["trajectory_id"]):
        seen.setdefault(row[key], row)
    reps = list(seen.values())
    successful = [r for r in reps if r["successful"]]
    hazard = sum(1 for r in reps if r["hazard_present"])
    succ_hazard = sum(1 for r in successful if r["hazard_present"])
    succ_clear = len(successful) - succ_hazard
    return {
        "identity_field": key,
        "n_distinct": len(reps),
        "written_hazard_rate": clopper_pearson(hazard, len(reps)),
        "successful_hazard_rate": clopper_pearson(succ_hazard, len(successful)),
        "success_given_hazard_present": clopper_pearson(succ_hazard, hazard),
        "success_given_hazard_absent": clopper_pearson(succ_clear, len(reps) - hazard),
    }


if __name__ == "__main__":
    raise SystemExit(main())
