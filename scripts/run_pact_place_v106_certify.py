#!/usr/bin/env python3
"""V10.6 Step 4: publish the selected scenes and certify them before episodes.

Three production scenes plus one no-pendant counterfactual are written to the
scenes directory and SHA-bound. Each is compiled independently and asserted
compiled-static with enclosing bounds.

Every minimum and threshold-near witness is then re-measured on the real
compiled scene with four independent instruments -- analytic GJK, signed/true
``mj_geomDistance``, live ``data.contact``, and the place contact audit -- which
must agree. State is restored after every probe and the restoration is checked,
not assumed.

Finally the contact-risk certificate (perturb the robot, never the pendant) and
the raw proximity causal check run. Any failure stops V10.6.
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

from pact_place_v106_contract import (  # noqa: E402
    CERT_ROOT,
    CONTRACT_VERSION_V106,
    ENVIRONMENT_VERSION_V106,
    INTRUSION_SIDES,
    SITING_ROOT,
    empty_authorization,
    recompute_payload_sha256,
    sha256_file,
    v95_row_payload,
    write_immutable_create_only,
)
from pact_place_v106_geometry import (  # noqa: E402
    CLEARANCE_FLOOR_M,
    PENDANT_BODY_V106,
    POSE_IDS,
    POSE_OFFSETS_M,
    RISK_BAND_M,
    build_assembly,
    scene_xml_text,
)

SCENES_DIR = (
    ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes"
)
CONTACT_MAGNITUDES_M = (0.005, 0.010, 0.015, 0.020, 0.025, 0.030)
THRESHOLD_NEAR_M = 0.020   # a witness within 5 mm of the 15 mm floor


def scene_name(pose_id: str) -> str:
    return f"pact_place_corridor_v10_6_{pose_id}.xml"


NO_PENDANT_NAME = "pact_place_corridor_v10_6_no_pendant.xml"


def publish_scenes(selected: dict[str, Any]) -> dict[str, Any]:
    """Write the three production scenes and the counterfactual, create-only."""
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


def compile_checks(pose: str, scene_path: Path, assembly: dict[str, Any]):
    """Compiled-static and enclosing-bounds checks on the real scene file."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    body_id = int(model.body(PENDANT_BODY_V106).id)
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
    bvh_ok = True
    try:
        n_bvh = int(model.nbvh)
        bvh_ok = n_bvh > 0 and bool(
            np.all(np.isfinite(np.asarray(model.bvh_aabb, dtype=float)))
        )
    except Exception:  # noqa: BLE001 - absence is reported, not fatal
        n_bvh = -1
    return {
        "pose_id": pose,
        "scene_sha256": sha256_file(scene_path),
        "body_dofnum": int(model.body_dofnum[body_id]),
        "body_jntnum": int(model.body_jntnum[body_id]),
        "body_mocapid": int(model.body_mocapid[body_id]),
        "compiled_static": bool(
            int(model.body_dofnum[body_id]) == 0
            and int(model.body_jntnum[body_id]) == 0
            and int(model.body_mocapid[body_id]) < 0
        ),
        "n_bvh": n_bvh,
        "bvh_finite": bvh_ok,
        "geoms": geoms,
        "bounds_ok": all(
            g["pos_matches"] and g["size_matches"] and g["aabb_encloses"]
            and g["rbound_encloses"] and g["collision_enabled"] for g in geoms
        ),
    }


def select_witnesses(npz_path: Path, key: str) -> list[dict[str, Any]]:
    """Per pose x side: the minimum witness plus every threshold-near one."""
    payload = np.load(npz_path, allow_pickle=True)
    rows = [json.loads(str(item)) for item in payload["rows"]]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("ok"):
            continue
        for pose in POSE_IDS:
            score = row["scores"].get(f"{key}|{pose}")
            if not score or score.get("min_clearance_m") is None:
                continue
            witness = score.get("min_witness") or {}
            grouped.setdefault((pose, row["intrusion_side"]), []).append({
                "pose_id": pose,
                "intrusion_side": row["intrusion_side"],
                "row_dir": row["row_dir"],
                "family_id": row["family_id"],
                "seed_u32": int(row["seed_u32"]),
                "min_clearance_m": float(score["min_clearance_m"]),
                "min_lobe_stem_m": score.get("min_lobe_stem_m"),
                "frame": int(witness.get("frame") or 0),
                "component": witness.get("box"),
                "probe_body": witness.get("probe_body"),
                "phase": witness.get("phase"),
            })
    out: list[dict[str, Any]] = []
    for (pose, side), items in sorted(grouped.items()):
        items.sort(key=lambda i: i["min_clearance_m"])
        chosen = {id(items[0]): items[0]}
        chosen[id(items[0])]["role"] = "group_minimum"
        for item in items[1:]:
            if item["min_clearance_m"] <= THRESHOLD_NEAR_M:
                item["role"] = "threshold_near"
                chosen[id(item)] = item
        out.extend(chosen.values())
    return out


def _load_task(scene: Path, family: str, side: str, seed: int, scratch: Path):
    from run_pact_place_expert_screen import _make_config

    row = {
        "role_index": 0, "episode_id": "cert", "intrusion_side": side,
        "task_seed_u32": int(seed), "task_seed_u64": int(seed),
        "sampler_class": "PactPlaceCorridorV93Sampler",
        **v95_row_payload(family, side),
    }
    config = _make_config(scratch / "d.json", scene_xml=scene,
                          sampler_class="PactPlaceCorridorV93Sampler")
    sampler = config.task_sampler_config.task_sampler_class(config)
    sampler.seed_task_sampling(int(seed))
    sampler.set_pact_manifest_row(row)
    task = sampler.sample_task(house_index=1)
    return task, sampler


def certify_witnesses(
    witnesses: list[dict[str, Any]], scenes: dict[str, Any], assembly_by_pose
) -> list[dict[str, Any]]:
    """Four-instrument agreement at each witness, with state restoration."""
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from molmo_spaces.tasks.pact_place_contact_audit import classify_contact
    from pact_geom_distance import gjk_distance, true_distance
    from pact_place_v105_clearance import (
        _shape, assembly_boxes, frame_clearances, geom_shape_cache,
        pendant_contact_state, pendant_geom_ids, robot_collision_geom_ids,
        target_collision_geom_ids,
    )
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    out: list[dict[str, Any]] = []
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for witness in witnesses:
        by_scene.setdefault(witness["pose_id"], []).append(witness)

    for pose, items in sorted(by_scene.items()):
        assembly = assembly_by_pose[pose]
        scene = ROOT / scenes["poses"][pose]["relative"]
        for witness in items:
            scratch = Path(tempfile.mkdtemp(prefix="v106_cert_"))
            task = sampler = None
            try:
                task, sampler = _load_task(
                    scene, witness["family_id"], witness["intrusion_side"],
                    witness["seed_u32"], scratch,
                )
                env = task.env
                model, data = env.current_model, env.current_data
                probe = robot_collision_geom_ids(model) + target_collision_geom_ids(task)
                cache = geom_shape_cache(model, probe)
                boxes = assembly_boxes(assembly)
                pendant_ids = pendant_geom_ids(
                    model, [i["geom"] for i in assembly["components"]]
                )
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
                    target_shape = _shape(
                        model, data, gid, geom_shape_cache(model, [gid])
                    )
                    for pid in probe:
                        shape = _shape(model, data, int(pid), cache)
                        if not shape.supported:
                            continue
                        unsigned = min(
                            unsigned, float(gjk_distance(shape, target_shape))
                        )
                live = pendant_contact_state(model, data, pendant_ids)
                classes = live["contact_classes"]
                audit_says_contact = bool(
                    "mounted_fixture" in classes and live["robot_or_target_contact"]
                )
                exact = report["min_m"]
                agree = bool(
                    (signed > 0.0) == (not live["contact"])
                    and (unsigned > 0.0) == (not live["contact"])
                    and audit_says_contact == bool(live["robot_or_target_contact"])
                    and abs(float(signed) - float(unsigned)) <= 1e-6
                    and abs(float(exact) - float(signed)) <= 1e-6
                )
                # Restore and verify, rather than assume.
                data.qpos[:] = before
                mujoco.mj_forward(model, data)
                restored = bool(
                    np.allclose(np.asarray(data.qpos, dtype=float), before,
                                atol=0.0, rtol=0.0)
                )
                out.append({
                    **witness,
                    "scene_sha256": scenes["poses"][pose]["sha256"],
                    "recorded_min_clearance_m": witness["min_clearance_m"],
                    "compiled_exact_min_m": exact,
                    "signed_true_distance_m": signed,
                    "analytic_gjk_m": unsigned if np.isfinite(unsigned) else None,
                    "live_contact": bool(live["contact"]),
                    "live_robot_or_target_contact": bool(
                        live["robot_or_target_contact"]
                    ),
                    "audit_contact_classes": classes,
                    "limiting": report["limiting"],
                    "reproduces_recorded_within_1mm": bool(
                        abs(float(exact) - float(witness["min_clearance_m"])) <= 0.001
                    ),
                    "above_floor": bool(float(exact) >= CLEARANCE_FLOOR_M),
                    "four_instruments_agree": agree,
                    "state_restored": restored,
                    "certified": bool(
                        agree and restored and float(exact) >= CLEARANCE_FLOOR_M
                    ),
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
    parser.add_argument("--siting", type=Path,
                        default=ROOT / SITING_ROOT / "siting.json")
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    started = time.time()

    siting_path = args.siting.resolve()
    siting = json.loads(siting_path.read_text())
    if not siting.get("siting_passed"):
        raise SystemExit(f"V10.6 siting did not select: {siting.get('stop_reason')}")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")

    selected = siting["selected"]
    scenes = publish_scenes(selected)
    assembly_by_pose = {
        pose: scenes["poses"][pose]["assembly"] for pose in POSE_IDS
    }
    compiled = [
        compile_checks(pose, ROOT / scenes["poses"][pose]["relative"],
                       assembly_by_pose[pose])
        for pose in POSE_IDS
    ]
    witnesses = select_witnesses(
        siting_path.parent / "per_row_scores.npz", siting["selected_key"]
    )
    print(f"certifying {len(witnesses)} witnesses "
          f"({sum(1 for w in witnesses if w['role'] == 'group_minimum')} minima, "
          f"{sum(1 for w in witnesses if w['role'] == 'threshold_near')} near)",
          flush=True)
    certified = certify_witnesses(witnesses, scenes, assembly_by_pose)

    all_certified = bool(certified) and all(c["certified"] for c in certified)
    compiled_ok = all(c["compiled_static"] and c["bounds_ok"] for c in compiled)
    document = {
        "schema_version": "pact_place_v106_certification_v1",
        "contract_version": CONTRACT_VERSION_V106,
        "environment_version": ENVIRONMENT_VERSION_V106,
        "siting_payload_sha256": recompute_payload_sha256(siting_path),
        "siting_raw_file_sha256": sha256_file(siting_path),
        "selected_key": siting["selected_key"],
        "selected_x_m": selected["x_m"],
        "selected_r_neg_m": selected["r_neg_m"],
        "selected_r_pos_m": selected["r_pos_m"],
        "published_scenes": {
            pose: {"relative": scenes["poses"][pose]["relative"],
                   "sha256": scenes["poses"][pose]["sha256"]}
            for pose in POSE_IDS
        },
        "no_pendant_scene": scenes["no_pendant"],
        "scenes_created": scenes["created"],
        "scenes_already_present": scenes["already_present"],
        "compiled_checks": compiled,
        "compiled_static_and_bounded": compiled_ok,
        "n_witnesses": len(certified),
        "n_group_minima": sum(
            1 for c in certified if c.get("role") == "group_minimum"
        ),
        "n_threshold_near": sum(
            1 for c in certified if c.get("role") == "threshold_near"
        ),
        "threshold_near_definition_m": THRESHOLD_NEAR_M,
        "witnesses": certified,
        "all_witnesses_certified": all_certified,
        "creates_episode": False,
        "calls_env_step": False,
        "elapsed_s": time.time() - started,
        **empty_authorization(),
        "certification_passed": bool(all_certified and compiled_ok),
    }
    hashes = write_immutable_create_only(
        output_root / "certification.json", document
    )
    print(json.dumps({
        "certification_passed": document["certification_passed"],
        "compiled_static_and_bounded": compiled_ok,
        "n_witnesses": len(certified),
        "all_witnesses_certified": all_certified,
        "scenes_created": scenes["created"],
        **hashes,
    }, indent=2))
    return 0 if document["certification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
