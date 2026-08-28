#!/usr/bin/env python3
"""V10.5 Step 2: score the full lattice against every reconstructed cell.

No early termination: every (x, r, d) is scored against every applicable
retained clean trajectory, and every rejection reason is recorded. Tight AABB
bounds are broad phase only; hardened exact GJK is the decision instrument.

No ``env.step`` and no new episode occur here.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v105_contract import (  # noqa: E402
    CONTRACT_VERSION_V105,
    ENVIRONMENT_VERSION_V105,
    INTRUSION_SIDES,
    RECONSTRUCTION_ROOT,
    SITING_ROOT,
    V95_LAYOUT_FAMILY_IDS,
    build_specification_contract,
    empty_authorization,
    recompute_payload_sha256,
    sha256_file,
    sha256_payload,
    write_immutable_create_only,
)
from pact_place_v105_geometry import (  # noqa: E402
    CLEARANCE_FLOOR_M,
    POSE_IDS,
    POSE_OFFSETS_M,
    RISK_BAND_M,
    assembly_sha256,
    build_assembly,
    lattice_candidates,
    scene_xml_sha256,
)

BASE_SCENE = (
    ROOT
    / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes"
    / "pact_place_corridor_v5.xml"
)


def _pin_threads() -> None:
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"


def score_one_row(job: dict[str, Any]) -> dict[str, Any]:
    """Snapshot one retained trajectory and score every lattice candidate."""
    _pin_threads()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    from pact_place_v105_siting_core import (
        environment_candidate_geoms,
        environment_clearance,
        score_candidate_against_snapshot,
        snapshot_row,
    )

    row_dir = ROOT / job["row_dir"]
    started = time.time()
    try:
        snap = snapshot_row(
            row_dir,
            job["family_id"],
            job["intrusion_side"],
            int(job["seed_u32"]),
            base_scene=BASE_SCENE,
        )
    except Exception as error:  # noqa: BLE001 - recorded, never silently dropped
        return {
            "row_dir": job["row_dir"],
            "family_id": job["family_id"],
            "intrusion_side": job["intrusion_side"],
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }
    assemblies = {
        (x, r, pose): build_assembly(x, r, POSE_OFFSETS_M[pose], pose_id=pose)
        for x, r in lattice_candidates()
        for pose in POSE_IDS
    }
    env_ids = environment_candidate_geoms(snap, list(assemblies.values()))
    scores: dict[str, Any] = {}
    for (x, r, pose), assembly in assemblies.items():
        result = score_candidate_against_snapshot(
            assembly, snap, job["intrusion_side"]
        )
        env = environment_clearance(assembly, snap, env_ids=env_ids)
        scores[f"{x:.3f}|{r:.3f}|{pose}"] = {
            "min_clearance_m": result["min_clearance_m"],
            "min_witness": result["min_witness"],
            "min_lobe_stem_m": result["min_lobe_stem_m"],
            "risk_witness": result["risk_witness"],
            "risk_by_direction_m": result["risk_by_direction_m"],
            "window_min_m": result["window_min_m"],
            "initial_min_m": result["initial_min_m"],
            "robot_or_target_contact": bool(result["contact"]),
            "env_min_m": env["min_m"],
            "env_witness": env["witness"],
            "env_intersects": bool(env["intersects"]),
        }
    return {
        "row_dir": job["row_dir"],
        "family_id": job["family_id"],
        "intrusion_side": job["intrusion_side"],
        "seed_u32": int(job["seed_u32"]),
        "n_frames": int(snap["n_frames"]),
        "n_env_geoms_screened": len(env_ids),
        "ok": True,
        "elapsed_s": time.time() - started,
        "scores": scores,
    }


def evaluate_bundle(
    x: float, r: float, per_row: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply the nine Step-2 predicates to one three-pose bundle."""
    reasons: list[str] = []
    poses = list(POSE_IDS)
    stats: dict[str, Any] = {}
    risk_groups: dict[str, list[float]] = {}
    direction_witness: dict[str, set] = {side: set() for side in INTRUSION_SIDES}
    all_min: list[float] = []
    side_medians: dict[str, list[float]] = {s: [] for s in INTRUSION_SIDES}
    for pose in poses:
        key_suffix = f"{x:.3f}|{r:.3f}|{pose}"
        entries = [
            (row, row["scores"][key_suffix]) for row in per_row if row.get("ok")
        ]
        pose_stats: dict[str, Any] = {
            "n_rows": len(entries),
            "min_clearance_m": None,
            "min_lobe_stem_m": None,
            "env_min_m": None,
            "contact_rows": [],
            "below_floor_rows": [],
            "env_intersect_rows": [],
            "window_violations": [],
            "initial_violations": [],
            "witness_role_violations": [],
        }
        pose_min = float("inf")
        pose_risk = float("inf")
        env_min = float("inf")
        for row, score in entries:
            side = row["intrusion_side"]
            group = f"{pose}|{side}"
            if score["robot_or_target_contact"]:
                pose_stats["contact_rows"].append(row["row_dir"])
            if score["env_intersects"]:
                pose_stats["env_intersect_rows"].append(row["row_dir"])
            value = score["min_clearance_m"]
            if value is not None:
                pose_min = min(pose_min, float(value))
                all_min.append(float(value))
                if float(value) < CLEARANCE_FLOOR_M:
                    pose_stats["below_floor_rows"].append(
                        {"row": row["row_dir"], "min_m": float(value)}
                    )
            risk = score["min_lobe_stem_m"]
            if risk is not None:
                pose_risk = min(pose_risk, float(risk))
                risk_groups.setdefault(group, []).append(float(risk))
                side_medians[side].append(float(risk))
            # A direction is a risk-band witness when that direction's own
            # minimum lands in the band. Gating this on the row's overall
            # minimum would hide an in-band inbound witness whenever the
            # loaded-outbound leg happened to pass closer -- which it usually
            # does, so the gate silently suppressed every inbound witness.
            for direction, dvalue in (score["risk_by_direction_m"] or {}).items():
                if dvalue is None:
                    continue
                if RISK_BAND_M[0] <= float(dvalue) <= RISK_BAND_M[1]:
                    direction_witness[side].add(direction)
            if score["env_min_m"] is not None:
                env_min = min(env_min, float(score["env_min_m"]))
            for name, value in (score["window_min_m"] or {}).items():
                if value is not None and float(value) < CLEARANCE_FLOOR_M:
                    pose_stats["window_violations"].append(
                        {"row": row["row_dir"], "window": name,
                         "min_m": float(value)}
                    )
            for name, value in (score["initial_min_m"] or {}).items():
                if value is not None and float(value) < CLEARANCE_FLOOR_M:
                    pose_stats["initial_violations"].append(
                        {"row": row["row_dir"], "probe": name,
                         "min_m": float(value)}
                    )
            witness = score["risk_witness"] or {}
            if witness and witness.get("role") not in ("lobe", "stem"):
                pose_stats["witness_role_violations"].append(row["row_dir"])
            want = "negative" if side == "left" else "positive"
            if witness and witness.get("side") not in ("", want):
                pose_stats["witness_role_violations"].append(row["row_dir"])
        pose_stats["min_clearance_m"] = (
            None if pose_min == float("inf") else pose_min
        )
        pose_stats["min_lobe_stem_m"] = (
            None if pose_risk == float("inf") else pose_risk
        )
        pose_stats["env_min_m"] = None if env_min == float("inf") else env_min
        stats[pose] = pose_stats

    # 1 - environment intersection
    if any(stats[p]["env_intersect_rows"] for p in poses):
        reasons.append("pendant intersects a collision-enabled environment geom")
    if any(
        stats[p]["env_min_m"] is not None and stats[p]["env_min_m"] <= 0.0
        for p in poses
    ):
        reasons.append("pendant-to-environment clearance is not positive")
    # 2 - initial clearance
    if any(stats[p]["initial_violations"] for p in poses):
        reasons.append("initial robot/target clearance below 15 mm")
    # 3 - contact on a historically clean trajectory
    if any(stats[p]["contact_rows"] for p in poses):
        reasons.append("robot or carried-target pendant contact on a clean row")
    # 4 - minimum clearance
    if any(stats[p]["below_floor_rows"] for p in poses):
        reasons.append("minimum clearance below 15 mm on a clean row")
    # 5 - phase windows
    if any(stats[p]["window_violations"] for p in poses):
        reasons.append("a grasp/lift/release window falls below 15 mm")
    # 6 - witness binds a lobe/stem on the route's own side
    if any(stats[p]["witness_role_violations"] for p in poses):
        reasons.append("closest-risk witness does not bind the route-side lobe/stem")
    # 7 - risk band per pose x side
    missing_groups = []
    for pose in poses:
        for side in INTRUSION_SIDES:
            values = risk_groups.get(f"{pose}|{side}", [])
            if not any(RISK_BAND_M[0] <= v <= RISK_BAND_M[1] for v in values):
                missing_groups.append(f"{pose}|{side}")
    if missing_groups:
        reasons.append(f"no 15-35 mm witness for groups {missing_groups}")
    # 8 - inbound and loaded-outbound witness per side
    missing_directions = []
    for side in INTRUSION_SIDES:
        have = direction_witness[side]
        if "inbound" not in have:
            missing_directions.append(f"{side}:inbound")
        if "loaded_outbound" not in have:
            missing_directions.append(f"{side}:loaded_outbound")
    if missing_directions:
        reasons.append(f"missing direction witnesses {missing_directions}")

    n_band = sum(
        1
        for values in risk_groups.values()
        for v in values
        if RISK_BAND_M[0] <= v <= RISK_BAND_M[1]
    )
    medians = {
        side: (float(np.median(values)) if values else None)
        for side, values in side_medians.items()
    }
    all_risk = [v for values in risk_groups.values() for v in values]
    median_all = float(np.median(all_risk)) if all_risk else None
    imbalance = (
        abs(medians["left"] - medians["right"])
        if medians["left"] is not None and medians["right"] is not None
        else None
    )
    return {
        "x_m": float(x),
        "r_m": float(r),
        "poses": stats,
        "n_risk_band_witnesses": n_band,
        "risk_group_counts": {k: len(v) for k, v in sorted(risk_groups.items())},
        "median_lobe_stem_m": median_all,
        "median_by_side_m": medians,
        "median_deviation_from_25mm_m": (
            None if median_all is None else abs(median_all - 0.025)
        ),
        "left_right_imbalance_m": imbalance,
        "direction_witnesses": {k: sorted(v) for k, v in direction_witness.items()},
        "rejection_reasons": reasons,
        "survives": not reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / SITING_ROOT)
    parser.add_argument(
        "--reconstruction", type=Path,
        default=ROOT / RECONSTRUCTION_ROOT / "reconstruction.json",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--from-scores", type=Path, default=None,
        help="re-derive bundles from a previous per_row_scores.npz, no rescoring",
    )
    parser.add_argument(
        "--rows-per-cell", type=int, default=0,
        help="0 uses every reconstructed strict-clean row in every cell",
    )
    args = parser.parse_args()
    _pin_threads()
    started = time.time()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")
    reconstruction_path = args.reconstruction.resolve()
    reconstruction = json.loads(reconstruction_path.read_text())
    if not reconstruction.get("reconstruction_passed"):
        raise SystemExit("Step 1 reconstruction did not pass; no siting")

    corpus = np.load(
        reconstruction_path.parent / "corpus_index.npz", allow_pickle=True
    )
    jobs: list[dict[str, Any]] = []
    for index in range(len(corpus["row_dir"])):
        if not bool(corpus["available"][index]):
            continue
        if not bool(corpus["derived_strict_clean"][index]):
            continue
        jobs.append(
            {
                "row_dir": str(corpus["row_dir"][index]),
                "family_id": str(corpus["family_id"][index]),
                "intrusion_side": str(corpus["intrusion_side"][index]),
                "seed_u32": int(corpus["seed_u32"][index]),
            }
        )
    if args.rows_per_cell > 0:
        limited: list[dict[str, Any]] = []
        seen: dict[str, int] = {}
        for job in jobs:
            key = f"{job['family_id']}|{job['intrusion_side']}"
            if seen.get(key, 0) >= args.rows_per_cell:
                continue
            seen[key] = seen.get(key, 0) + 1
            limited.append(job)
        jobs = limited

    per_row: list[dict[str, Any]] = []
    if args.from_scores is not None:
        payload = np.load(args.from_scores.resolve(), allow_pickle=True)
        per_row = [json.loads(str(item)) for item in payload["rows"]]
        jobs = [
            {"row_dir": item["row_dir"], "family_id": item["family_id"],
             "intrusion_side": item["intrusion_side"],
             "seed_u32": item.get("seed_u32", 0)}
            for item in per_row
        ]
        print(f"re-deriving bundles from {len(per_row)} stored row scores",
              flush=True)
        per_row.sort(key=lambda item: item["row_dir"])
        failed = [item["row_dir"] for item in per_row if not item.get("ok")]
        return _finish(args, output_root, reconstruction_path, jobs, per_row,
                       failed, started)

    print(f"scoring {len(lattice_candidates())} candidates x 3 poses "
          f"against {len(jobs)} retained clean trajectories", flush=True)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, min(args.workers, max(1, len(jobs)))),
        mp_context=context,
        max_tasks_per_child=1,
    ) as executor:
        futures = [executor.submit(score_one_row, job) for job in jobs]
        for done, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            per_row.append(record)
            print(
                json.dumps(
                    {
                        "done": done,
                        "of": len(jobs),
                        "row": record["row_dir"].split("/")[-1],
                        "ok": record.get("ok"),
                        "elapsed_s": round(record.get("elapsed_s", 0.0), 1),
                    }
                ),
                flush=True,
            )
    per_row.sort(key=lambda item: item["row_dir"])
    failed = [item["row_dir"] for item in per_row if not item.get("ok")]
    return _finish(args, output_root, reconstruction_path, jobs, per_row,
                   failed, started)


def _finish(args, output_root, reconstruction_path, jobs, per_row, failed,
            started):
    bundles = [
        evaluate_bundle(x, r, per_row) for x, r in lattice_candidates()
    ]
    survivors = [item for item in bundles if item["survives"]]
    ranked = sorted(
        survivors,
        key=lambda item: (
            -int(item["n_risk_band_witnesses"]),
            float(item["median_deviation_from_25mm_m"] or 1e9),
            float(item["left_right_imbalance_m"] or 1e9),
            -float(item["r_m"]),
            -float(item["x_m"]),
        ),
    )
    selected = ranked[0] if ranked else None

    scenes: dict[str, Any] = {}
    if selected is not None:
        for pose in POSE_IDS:
            assembly = build_assembly(
                selected["x_m"], selected["r_m"], POSE_OFFSETS_M[pose],
                pose_id=pose,
            )
            scenes[pose] = {
                "assembly": assembly,
                "assembly_sha256": assembly_sha256(assembly),
                "scene_xml_sha256": scene_xml_sha256(assembly),
            }

    document = {
        "schema_version": "pact_place_v105_siting_v1",
        "contract_version": CONTRACT_VERSION_V105,
        "environment_version": ENVIRONMENT_VERSION_V105,
        "specification_contract_payload_sha256": sha256_payload(
            build_specification_contract()
        ),
        "reconstruction_payload_sha256": recompute_payload_sha256(reconstruction_path),
        "reconstruction_raw_file_sha256": sha256_file(reconstruction_path),
        "lattice": {
            "n_candidates": len(lattice_candidates()),
            "n_poses": len(POSE_IDS),
            "n_scenes_scored": len(lattice_candidates()) * len(POSE_IDS),
            "extended_after_results": False,
        },
        "n_source_rows": len(jobs),
        "n_source_rows_ok": sum(1 for item in per_row if item.get("ok")),
        "failed_rows": failed,
        "search_exhaustive": not failed,
        "early_termination": False,
        "clearance_floor_m": CLEARANCE_FLOOR_M,
        "risk_band_m": list(RISK_BAND_M),
        "bundles": bundles,
        "n_survivors": len(survivors),
        "ranking_truncated": False,
        "ranked_survivors": [
            {
                "x_m": item["x_m"], "r_m": item["r_m"],
                "n_risk_band_witnesses": item["n_risk_band_witnesses"],
                "median_lobe_stem_m": item["median_lobe_stem_m"],
                "median_deviation_from_25mm_m": item[
                    "median_deviation_from_25mm_m"
                ],
                "left_right_imbalance_m": item["left_right_imbalance_m"],
            }
            for item in ranked
        ],
        "selected": None if selected is None else {
            "x_m": selected["x_m"], "r_m": selected["r_m"],
            "n_risk_band_witnesses": selected["n_risk_band_witnesses"],
            "median_lobe_stem_m": selected["median_lobe_stem_m"],
            "median_by_side_m": selected["median_by_side_m"],
            "direction_witnesses": selected["direction_witnesses"],
            "poses": selected["poses"],
        },
        "selected_scenes": scenes,
        "creates_episode": False,
        "calls_env_step": False,
        "elapsed_s": time.time() - started,
        **empty_authorization(),
        "siting_passed": bool(selected is not None and not failed),
        "stop_reason": (
            None if selected is not None
            else "no_complete_three_pose_bundle_survived_the_lattice"
        ),
    }
    hashes = write_immutable_create_only(output_root / "siting.json", document)
    np.savez_compressed(
        output_root / "per_row_scores.npz",
        rows=np.array([json.dumps(item) for item in per_row], dtype=object),
        allow_pickle=True,
    )
    print(json.dumps({
        "siting_passed": document["siting_passed"],
        "n_survivors": len(survivors),
        "selected": None if selected is None
        else {"x_m": selected["x_m"], "r_m": selected["r_m"]},
        "n_source_rows": len(jobs),
        "failed_rows": failed,
        **hashes,
    }, indent=2))
    return 0 if document["siting_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
