#!/usr/bin/env python3
"""V10.6 Step 3: rescore the asymmetric lattice against all 98 clean trajectories.

Reuses the V10.5 snapshot core unchanged -- it is pendant-agnostic, so the
expensive half (task sampling and replay) is identical and only the candidate
geometry differs. Aggregation uses the audit's flat-table implementation, and
the preregistered admission rule from the contract is applied without
modification.

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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v106_contract import (  # noqa: E402
    CONTRACT_VERSION_V106,
    ENVIRONMENT_VERSION_V106,
    INTRUSION_SIDES,
    N_EVALUATIONS_PER_BUNDLE,
    SITING_ROOT,
    V105_AUDIT_ARTIFACT,
    V105_RECON_ARTIFACT,
    admit_candidate,
    build_specification_contract,
    empty_authorization,
    rank_key,
    recompute_payload_sha256,
    sha256_file,
    sha256_payload,
    write_immutable_create_only,
)
from pact_place_v106_geometry import (  # noqa: E402
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
    ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes"
    / "pact_place_corridor_v5.xml"
)


def _pin_threads() -> None:
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"


def score_one_row(job: dict[str, Any]) -> dict[str, Any]:
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

    started = time.time()
    try:
        snap = snapshot_row(
            ROOT / job["row_dir"], job["family_id"], job["intrusion_side"],
            int(job["seed_u32"]), base_scene=BASE_SCENE,
        )
    except Exception as error:  # noqa: BLE001
        return {**job, "ok": False, "error": f"{type(error).__name__}: {error}"}
    assemblies = {
        (x, rn, rp, pose): build_assembly(x, rn, rp, POSE_OFFSETS_M[pose],
                                          pose_id=pose)
        for x, rn, rp in lattice_candidates()
        for pose in POSE_IDS
    }
    env_ids = environment_candidate_geoms(snap, list(assemblies.values()))
    scores: dict[str, Any] = {}
    for (x, rn, rp, pose), assembly in assemblies.items():
        result = score_candidate_against_snapshot(
            assembly, snap, job["intrusion_side"]
        )
        env = environment_clearance(assembly, snap, env_ids=env_ids)
        scores[f"{x:.3f}|{rn:.3f}|{rp:.3f}|{pose}"] = {
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
        "row_dir": job["row_dir"], "family_id": job["family_id"],
        "intrusion_side": job["intrusion_side"], "seed_u32": int(job["seed_u32"]),
        "n_frames": int(snap["n_frames"]), "ok": True,
        "elapsed_s": time.time() - started, "scores": scores,
    }


def aggregate(per_row: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Flat-table aggregation, the same implementation the V10.5 audit used."""
    flat: list[dict[str, Any]] = []
    for row in per_row:
        if not row.get("ok"):
            continue
        for key, score in row["scores"].items():
            x_t, rn_t, rp_t, pose = key.split("|")
            flat.append({
                "x_m": float(x_t), "r_neg_m": float(rn_t), "r_pos_m": float(rp_t),
                "pose": pose, "side": row["intrusion_side"],
                "row_dir": row["row_dir"], **score,
            })
    by_bundle: dict[tuple[float, float, float], list[dict]] = defaultdict(list)
    for item in flat:
        by_bundle[(item["x_m"], item["r_neg_m"], item["r_pos_m"])].append(item)

    out: dict[str, dict[str, Any]] = {}
    for (x_m, rn, rp), items in sorted(by_bundle.items()):
        clear = [float(i["min_clearance_m"]) for i in items
                 if i["min_clearance_m"] is not None]
        risks = [float(i["min_lobe_stem_m"]) for i in items
                 if i["min_lobe_stem_m"] is not None]
        below = [i for i in items if i["min_clearance_m"] is not None
                 and float(i["min_clearance_m"]) < CLEARANCE_FLOOR_M]
        contacts = [i for i in items if i["robot_or_target_contact"]]
        groups, band = {}, {}
        for pose in POSE_IDS:
            for side in INTRUSION_SIDES:
                sel = [i for i in items if i["pose"] == pose and i["side"] == side]
                ok = [i for i in sel if i["min_clearance_m"] is not None
                      and float(i["min_clearance_m"]) >= CLEARANCE_FLOOR_M]
                groups[f"{pose}|{side}"] = {"n": len(sel), "n_ge_floor": len(ok)}
                band[f"{pose}|{side}"] = sum(
                    1 for i in sel if i["min_lobe_stem_m"] is not None
                    and RISK_BAND_M[0] <= float(i["min_lobe_stem_m"]) <= RISK_BAND_M[1]
                )
        directions: dict[str, set] = {s: set() for s in INTRUSION_SIDES}
        for i in items:
            for direction, value in (i["risk_by_direction_m"] or {}).items():
                if value is None:
                    continue
                if RISK_BAND_M[0] <= float(value) <= RISK_BAND_M[1]:
                    directions[i["side"]].add(direction)
        window_below = [
            {"row": i["row_dir"], "pose": i["pose"], "window": n, "min_m": float(v)}
            for i in items for n, v in (i["window_min_m"] or {}).items()
            if v is not None and float(v) < CLEARANCE_FLOOR_M
        ]
        initial_below = [
            {"row": i["row_dir"], "pose": i["pose"], "probe": n, "min_m": float(v)}
            for i in items for n, v in (i["initial_min_m"] or {}).items()
            if v is not None and float(v) < CLEARANCE_FLOOR_M
        ]
        n_eval = len(items)
        out[f"{x_m:.3f}|{rn:.3f}|{rp:.3f}"] = {
            "x_m": x_m, "r_neg_m": rn, "r_pos_m": rp,
            "n_evaluations": n_eval,
            "absolute_min_clearance_m": min(clear) if clear else None,
            "n_below_floor": len(below),
            "below_floor": below[:8],
            "n_contacts": len(contacts),
            "fraction_ge_floor": (n_eval - len(below)) / n_eval if n_eval else None,
            "median_lobe_stem_m": float(np.median(risks)) if risks else None,
            "band_evaluations_by_group": band,
            "evaluations_ge_floor_by_group": groups,
            "direction_band_witnesses": {k: sorted(v) for k, v in directions.items()},
            "n_window_below_floor": len(window_below),
            "window_below_floor": window_below[:8],
            "n_initial_below_floor": len(initial_below),
            "n_env_intersections": sum(1 for i in items if i["env_intersects"]),
            "min_env_clearance_m": min(
                (float(i["env_min_m"]) for i in items if i["env_min_m"] is not None),
                default=None,
            ),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / SITING_ROOT)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    _pin_threads()
    started = time.time()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")
    audit_path = ROOT / V105_AUDIT_ARTIFACT
    audit = json.loads(audit_path.read_text())
    if not audit.get("audit_passed"):
        raise SystemExit("the V10.5 audit did not pass; V10.6 does not run")

    recon_path = ROOT / V105_RECON_ARTIFACT
    corpus = np.load(recon_path.parent / "corpus_index.npz", allow_pickle=True)
    jobs = [
        {
            "row_dir": str(corpus["row_dir"][i]),
            "family_id": str(corpus["family_id"][i]),
            "intrusion_side": str(corpus["intrusion_side"][i]),
            "seed_u32": int(corpus["seed_u32"][i]),
        }
        for i in range(len(corpus["row_dir"]))
        if bool(corpus["available"][i]) and bool(corpus["derived_strict_clean"][i])
    ]
    print(f"scoring {len(lattice_candidates())} asymmetric candidates x "
          f"{len(POSE_IDS)} poses against {len(jobs)} clean trajectories",
          flush=True)

    context = multiprocessing.get_context("spawn")
    per_row: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, min(args.workers, max(1, len(jobs)))),
        mp_context=context, max_tasks_per_child=1,
    ) as executor:
        futures = [executor.submit(score_one_row, job) for job in jobs]
        for done, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            per_row.append(record)
            print(json.dumps({
                "done": done, "of": len(jobs),
                "row": record["row_dir"].split("/")[-1],
                "ok": record.get("ok"),
                "elapsed_s": round(record.get("elapsed_s", 0.0), 1),
            }), flush=True)
    per_row.sort(key=lambda item: item["row_dir"])
    failed = [item["row_dir"] for item in per_row if not item.get("ok")]

    bundles = aggregate(per_row)
    decisions = {key: admit_candidate(stats) for key, stats in bundles.items()}
    admitted = [key for key, value in decisions.items() if value["admitted"]]
    ranked = sorted(
        admitted, key=lambda key: rank_key(bundles[key], decisions[key])
    )
    selected_key = ranked[0] if ranked else None
    selected = bundles[selected_key] if selected_key else None

    scenes: dict[str, Any] = {}
    if selected is not None:
        for pose in POSE_IDS:
            assembly = build_assembly(
                selected["x_m"], selected["r_neg_m"], selected["r_pos_m"],
                POSE_OFFSETS_M[pose], pose_id=pose,
            )
            scenes[pose] = {
                "assembly": assembly,
                "assembly_sha256": assembly_sha256(assembly),
                "scene_xml_sha256": scene_xml_sha256(assembly),
            }

    document = {
        "schema_version": "pact_place_v106_siting_v1",
        "contract_version": CONTRACT_VERSION_V106,
        "environment_version": ENVIRONMENT_VERSION_V106,
        "specification_contract_payload_sha256": sha256_payload(
            build_specification_contract()
        ),
        "v105_audit_payload_sha256": recompute_payload_sha256(audit_path),
        "v105_audit_raw_file_sha256": sha256_file(audit_path),
        "lattice": {
            "n_candidates": len(lattice_candidates()),
            "n_poses": len(POSE_IDS),
            "n_scenes_scored": len(lattice_candidates()) * len(POSE_IDS),
            "asymmetric": True, "per_family_placement": False,
            "extended_after_results": False,
        },
        "n_source_rows": len(jobs),
        "n_source_rows_ok": sum(1 for item in per_row if item.get("ok")),
        "failed_rows": failed,
        "expected_evaluations_per_bundle": N_EVALUATIONS_PER_BUNDLE,
        "search_exhaustive": not failed,
        "early_termination": False,
        "bundles": bundles,
        "admission_decisions": decisions,
        "n_admitted": len(admitted),
        "n_universal": sum(
            1 for v in decisions.values() if v["universal_clearance"]
        ),
        "ranked_admitted": ranked,
        "ranking_truncated": False,
        "selected_key": selected_key,
        "selected": selected,
        "selected_admission": decisions.get(selected_key) if selected_key else None,
        "selected_scenes": scenes,
        "creates_episode": False, "calls_env_step": False,
        "elapsed_s": time.time() - started,
        **empty_authorization(),
        "siting_passed": bool(selected is not None and not failed),
        "stop_reason": (
            None if selected is not None
            else "no_asymmetric_candidate_met_universal_or_fallback_admission"
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
        "n_admitted": len(admitted), "n_universal": document["n_universal"],
        "selected_key": selected_key,
        "n_source_rows": len(jobs), "failed_rows": failed, **hashes,
    }, indent=2))
    return 0 if document["siting_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
