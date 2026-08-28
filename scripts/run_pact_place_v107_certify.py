#!/usr/bin/env python3
"""V10.7 Step 3: recompile the selected scenes and certify every witness.

Four instruments must agree at every one of the six group minima and at every
threshold-near witness: analytic GJK, hardened signed ``mj_geomDistance``, live
``data.contact``, and the place contact audit. Any disagreement fails closed --
the run does not "mostly" certify.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v107_contract import (  # noqa: E402
    ALL_GEOMS_V106,
    CERT_ROOT,
    CLEARANCE_FLOOR_M,
    CONTRACT_VERSION_V107,
    ENVIRONMENT_VERSION,
    INTRUSION_SIDES,
    N_GROUPS,
    PENDANT_BODY,
    POSE_IDS,
    POSE_OFFSETS_M,
    SCENES_DIR_RELATIVE,
    SELECTION_ROOT,
    SPEC_ROOT,
    THRESHOLD_NEAR_M,
    V106_SITING_NPZ,
    assert_no_drift,
    build_assembly,
    empty_authorization,
    group_key,
    recompute_payload_sha256,
    scene_xml_text,
    sha256_file,
    write_immutable_create_only,
)

SCENES_DIR = ROOT / SCENES_DIR_RELATIVE
DISTANCE_TOLERANCE_M = 1e-6


def scene_name(pose_id: str) -> str:
    return f"pact_place_corridor_v10_7_{pose_id}.xml"


NO_PENDANT_NAME = "pact_place_corridor_v10_7_no_pendant.xml"


def publish_scenes(selected: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {"poses": {}, "created": [], "already_present": []}
    for pose in POSE_IDS:
        assembly = build_assembly(
            selected["x_m"], selected["r_neg_m"], selected["r_pos_m"],
            POSE_OFFSETS_M[pose], pose_id=pose,
        )
        path = SCENES_DIR / scene_name(pose)
        text = scene_xml_text(assembly)
        if path.exists():
            if path.read_text() != text:
                raise RuntimeError(f"refusing to overwrite {path}")
            out["already_present"].append(path.name)
        else:
            path.write_text(text)
            out["created"].append(path.name)
        out["poses"][pose] = {
            "relative": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "assembly": assembly,
        }
    control = SCENES_DIR / NO_PENDANT_NAME
    text = scene_xml_text(None)
    if control.exists():
        if control.read_text() != text:
            raise RuntimeError(f"refusing to overwrite {control}")
        out["already_present"].append(control.name)
    else:
        control.write_text(text)
        out["created"].append(control.name)
    out["no_pendant"] = {
        "relative": str(control.relative_to(ROOT)),
        "sha256": sha256_file(control),
    }
    return out


def compile_checks(pose: str, path: Path, assembly: dict[str, Any]):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    body_id = int(model.body(PENDANT_BODY).id)
    geoms = []
    for item in assembly["components"]:
        gid = int(model.geom(item["geom"]).id)
        half = np.asarray(item["half_m"], dtype=float)
        centre = np.asarray(item["center_m"], dtype=float)
        aabb = np.asarray(model.geom_aabb[gid], dtype=float)
        geoms.append({
            "geom": item["geom"],
            "pos_matches": bool(np.allclose(model.geom_pos[gid], centre, atol=1e-9)),
            "size_matches": bool(np.allclose(model.geom_size[gid], half, atol=1e-9)),
            "aabb_encloses": bool(np.all(aabb[3:] >= half - 1e-12)),
            "rbound_encloses": bool(
                float(model.geom_rbound[gid]) >= float(np.linalg.norm(half)) - 1e-9
            ),
            "collision_enabled": bool(
                int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid])
            ),
        })
    return {
        "pose_id": pose,
        "scene_sha256": sha256_file(path),
        "compiled_static": bool(
            int(model.body_dofnum[body_id]) == 0
            and int(model.body_jntnum[body_id]) == 0
            and int(model.body_mocapid[body_id]) < 0
        ),
        "n_bvh": int(model.nbvh),
        "bvh_finite": bool(
            np.all(np.isfinite(np.asarray(model.bvh_aabb, dtype=float)))
        ),
        "geoms": geoms,
        "bounds_ok": all(
            g["pos_matches"] and g["size_matches"] and g["aabb_encloses"]
            and g["rbound_encloses"] and g["collision_enabled"] for g in geoms
        ),
    }


def collect_witnesses(npz_path: Path, bundle_key: str) -> list[dict[str, Any]]:
    """Every group minimum plus every threshold-near evaluation."""
    per_row = [
        json.loads(str(item))
        for item in np.load(npz_path, allow_pickle=True)["rows"]
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_row:
        if not row.get("ok"):
            continue
        for pose in POSE_IDS:
            score = row["scores"].get(f"{bundle_key}|{pose}")
            if not score or score.get("min_clearance_m") is None:
                continue
            limiting = score.get("min_witness") or {}
            grouped.setdefault(group_key(pose, row["intrusion_side"]), []).append({
                "group": group_key(pose, row["intrusion_side"]),
                "pose_id": pose, "intrusion_side": row["intrusion_side"],
                "row_dir": row["row_dir"], "family_id": row["family_id"],
                "seed_u32": int(row["seed_u32"]),
                "min_clearance_m": float(score["min_clearance_m"]),
                "frame": int(limiting.get("frame") or 0),
                "component": limiting.get("box"),
                "probe_body": limiting.get("probe_body"),
                "phase": limiting.get("phase"),
            })
    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        items.sort(key=lambda i: i["min_clearance_m"])
        items[0]["role"] = "group_minimum"
        out.append(items[0])
        for item in items[1:]:
            if item["min_clearance_m"] <= THRESHOLD_NEAR_M:
                item["role"] = "threshold_near"
                out.append(item)
    return out


def certify(witnesses, scenes, assembly_by_pose) -> list[dict[str, Any]]:
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from pact_geom_distance import gjk_distance, true_distance
    from pact_place_v105_clearance import (
        _shape, assembly_boxes, frame_clearances, geom_shape_cache,
        pendant_contact_state, pendant_geom_ids, robot_collision_geom_ids,
        target_collision_geom_ids,
    )
    from pact_place_v107_contract import v95_row_payload
    from run_pact_place_expert_screen import _make_config
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    out: list[dict[str, Any]] = []
    for witness in witnesses:
        pose = witness["pose_id"]
        scene = ROOT / scenes["poses"][pose]["relative"]
        assembly = assembly_by_pose[pose]
        scratch = Path(tempfile.mkdtemp(prefix="v107_cert_"))
        task = sampler = None
        try:
            row = {
                "role_index": 0, "episode_id": "v107cert",
                "intrusion_side": witness["intrusion_side"],
                "task_seed_u32": int(witness["seed_u32"]),
                "task_seed_u64": int(witness["seed_u32"]),
                "sampler_class": "PactPlaceCorridorV93Sampler",
                **v95_row_payload(witness["family_id"], witness["intrusion_side"]),
            }
            config = _make_config(scratch / "d.json", scene_xml=scene,
                                  sampler_class="PactPlaceCorridorV93Sampler")
            sampler = config.task_sampler_config.task_sampler_class(config)
            sampler.seed_task_sampling(int(witness["seed_u32"]))
            sampler.set_pact_manifest_row(row)
            task = sampler.sample_task(house_index=1)
            env = task.env
            model, data = env.current_model, env.current_data
            probe = robot_collision_geom_ids(model) + target_collision_geom_ids(task)
            cache = geom_shape_cache(model, probe)
            boxes = assembly_boxes(assembly)
            pendant_ids = pendant_geom_ids(model, list(ALL_GEOMS_V106))
            before = np.asarray(data.qpos, dtype=float).copy()
            steps = json.loads(
                (ROOT / witness["row_dir"] / "trajectory.json").read_text()
            )["steps"]
            apply_recorded_qpos(env, steps[witness["frame"]]["qpos"])
            mujoco.mj_forward(model, data)

            report = frame_clearances(model, data, boxes, probe, cache)
            signed = float(true_distance(model, data, probe, pendant_ids))
            unsigned = float("inf")
            for gid in pendant_ids:
                shape_t = _shape(model, data, gid, geom_shape_cache(model, [gid]))
                for pid in probe:
                    shape = _shape(model, data, int(pid), cache)
                    if not shape.supported:
                        continue
                    unsigned = min(unsigned, float(gjk_distance(shape, shape_t)))
            live = pendant_contact_state(model, data, pendant_ids)
            exact = float(report["min_m"])
            instruments = {
                "exact_frame_clearance_m": exact,
                "signed_true_distance_m": signed,
                "analytic_gjk_m": None if not np.isfinite(unsigned) else unsigned,
                "live_contact": bool(live["contact"]),
                "live_robot_or_target_contact": bool(live["robot_or_target_contact"]),
                "audit_contact_classes": live["contact_classes"],
            }
            agreements = {
                "exact_matches_signed": abs(exact - signed) <= DISTANCE_TOLERANCE_M,
                "signed_matches_gjk": (
                    np.isfinite(unsigned)
                    and abs(signed - float(unsigned)) <= DISTANCE_TOLERANCE_M
                ),
                "positive_distance_implies_no_live_contact": (
                    (signed > 0.0) == (not live["contact"])
                ),
                "audit_agrees_with_live": (
                    bool(live["robot_or_target_contact"])
                    == ("mounted_fixture" in live["contact_classes"])
                ),
                "reproduces_recorded_within_1mm": (
                    abs(exact - witness["min_clearance_m"]) <= 0.001
                ),
                "above_floor": exact >= CLEARANCE_FLOOR_M,
            }
            data.qpos[:] = before
            mujoco.mj_forward(model, data)
            agreements["state_restored"] = bool(
                np.array_equal(np.asarray(data.qpos, dtype=float), before)
            )
            out.append({
                **witness,
                "scene_sha256": scenes["poses"][pose]["sha256"],
                "instruments": instruments,
                "limiting": report["limiting"],
                "agreements": agreements,
                "certified": all(agreements.values()),
            })
        except Exception as error:  # noqa: BLE001
            out.append({**witness, "certified": False,
                        "error": f"{type(error).__name__}: {error}"})
        finally:
            cleanup_episode_resources(
                task=task, policy=None, task_sampler=sampler,
                preloaded_policy=None, close_task_sampler=sampler is not None,
            )
            shutil.rmtree(scratch, ignore_errors=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / CERT_ROOT)
    parser.add_argument("--selection", type=Path,
                        default=ROOT / SELECTION_ROOT / "selection.json")
    parser.add_argument("--specification", type=Path,
                        default=ROOT / SPEC_ROOT / "specification.json")
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    started = time.time()

    spec = json.loads(args.specification.resolve().read_text())
    drift = assert_no_drift(spec)
    selection_path = args.selection.resolve()
    selection = json.loads(selection_path.read_text())
    if not selection.get("selection_passed"):
        raise SystemExit("V10.7 selection did not pass")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)

    selected = selection["selected"]
    scenes = publish_scenes(selected)
    assembly_by_pose = {p: scenes["poses"][p]["assembly"] for p in POSE_IDS}
    compiled = [
        compile_checks(p, ROOT / scenes["poses"][p]["relative"],
                       assembly_by_pose[p])
        for p in POSE_IDS
    ]
    witnesses = collect_witnesses(ROOT / V106_SITING_NPZ,
                                  selection["selected_key"])
    n_min = sum(1 for w in witnesses if w["role"] == "group_minimum")
    n_near = sum(1 for w in witnesses if w["role"] == "threshold_near")
    print(f"certifying {len(witnesses)} witnesses ({n_min} minima, {n_near} near)",
          flush=True)
    certified = certify(witnesses, scenes, assembly_by_pose)

    groups_covered = sorted({c["group"] for c in certified})
    all_certified = bool(certified) and all(c["certified"] for c in certified)
    compiled_ok = all(c["compiled_static"] and c["bounds_ok"] for c in compiled)
    disagreements = [
        {"group": c["group"], "role": c.get("role"),
         "failed": [k for k, v in (c.get("agreements") or {}).items() if not v],
         "error": c.get("error")}
        for c in certified if not c["certified"]
    ]

    npz_out = output_root / "certification_scores.npz"
    np.savez_compressed(
        npz_out,
        group=np.array([c["group"] for c in certified], dtype=object),
        role=np.array([c.get("role", "") for c in certified], dtype=object),
        row_dir=np.array([c["row_dir"] for c in certified], dtype=object),
        frame=np.array([c["frame"] for c in certified], dtype=np.int64),
        recorded_min_clearance_m=np.array(
            [c["min_clearance_m"] for c in certified], dtype=np.float64),
        exact_frame_clearance_m=np.array(
            [(c.get("instruments") or {}).get("exact_frame_clearance_m", np.nan)
             for c in certified], dtype=np.float64),
        signed_true_distance_m=np.array(
            [(c.get("instruments") or {}).get("signed_true_distance_m", np.nan)
             for c in certified], dtype=np.float64),
        analytic_gjk_m=np.array(
            [((c.get("instruments") or {}).get("analytic_gjk_m") or np.nan)
             for c in certified], dtype=np.float64),
        certified=np.array([c["certified"] for c in certified], dtype=bool),
        allow_pickle=True,
    )
    npz_sha = sha256_file(npz_out)

    document = {
        "schema_version": "pact_place_v107_certification_v1",
        "contract_version": CONTRACT_VERSION_V107,
        "environment_version": ENVIRONMENT_VERSION,
        "specification_payload_sha256": recompute_payload_sha256(
            args.specification.resolve()),
        "drift_check": drift,
        "selection_payload_sha256": recompute_payload_sha256(selection_path),
        "selection_raw_file_sha256": sha256_file(selection_path),
        "selected_key": selection["selected_key"],
        "selected": {k: selected[k] for k in ("x_m", "r_neg_m", "r_pos_m")},
        "published_scenes": {
            p: {"relative": scenes["poses"][p]["relative"],
                "sha256": scenes["poses"][p]["sha256"]} for p in POSE_IDS
        },
        "no_pendant_scene": scenes["no_pendant"],
        "scenes_created": scenes["created"],
        "scenes_already_present": scenes["already_present"],
        "compiled_checks": compiled,
        "compiled_static_and_bounded": compiled_ok,
        "certification_scores_npz": "certification_scores.npz",
        "certification_scores_raw_file_sha256": npz_sha,
        "threshold_near_m": THRESHOLD_NEAR_M,
        "distance_tolerance_m": DISTANCE_TOLERANCE_M,
        "n_witnesses": len(certified),
        "n_group_minima": n_min,
        "n_threshold_near": n_near,
        "groups_covered": groups_covered,
        "all_six_groups_covered": len(groups_covered) == N_GROUPS,
        "witnesses": certified,
        "disagreements": disagreements,
        "all_witnesses_certified": all_certified,
        "creates_episode": False, "calls_env_step": False,
        "elapsed_s": time.time() - started,
        **empty_authorization(),
        "certification_passed": bool(
            all_certified and compiled_ok and len(groups_covered) == N_GROUPS
        ),
    }
    hashes = write_immutable_create_only(
        output_root / "certification.json", document)
    print(json.dumps({
        "certification_passed": document["certification_passed"],
        "n_witnesses": len(certified), "n_minima": n_min, "n_near": n_near,
        "groups_covered": len(groups_covered),
        "compiled_static_and_bounded": compiled_ok,
        "disagreements": disagreements,
        "scenes_created": scenes["created"],
        "certification_scores_npz_sha256": npz_sha,
        **hashes,
    }, indent=2))
    return 0 if document["certification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
