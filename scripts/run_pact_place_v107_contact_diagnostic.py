#!/usr/bin/env python3
"""V10.7 contact perturbation, retained as a DIAGNOSTIC and never as a gate.

The V10.6 version of this probe gated qualification and failed for reasons that
were about the probe, not the environment. It is kept here because a repaired
version is still informative, with four changes:

* the carried target moves **rigidly with the gripper**, so displacing the arm
  does not tear the grasp apart;
* only actual gripper-pad-to-carried-target grasp contacts are allowlisted, by
  geom name, not every contact that happens to involve the target;
* baseline penetrations are tracked, and a probe that makes an existing
  penetration **worse** is reported rather than silently tolerated;
* GJK, ``mj_geomDistance`` and live contact must agree before any contact is
  called.

``diagnostic_only: true`` and ``gates_qualification: false`` are written into
the artifact. Nothing downstream reads its outcome.
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
    CONTRACT_VERSION_V107,
    DIAGNOSTIC_ROOT,
    SPEC_ROOT,
    assert_no_drift,
    empty_authorization,
    recompute_payload_sha256,
    sha256_file,
    v95_row_payload,
    write_immutable_create_only,
)

MAGNITUDES_M = (0.005, 0.010, 0.015, 0.020, 0.025, 0.030)
PROBE_STEP_M = 0.005
GRIPPER_PAD_TOKENS = ("gripper/left_pad", "gripper/right_pad",
                      "gripper/left_finger", "gripper/right_finger")
CARRIED_TARGET_TOKENS = ("cavity_obj_", "grasp_target")


def is_grasp_pad_contact(pair: dict[str, Any]) -> bool:
    """Narrow allowlist: a gripper PAD against the carried target, only.

    A robot link against the carried target is a genuine collision and is not
    allowlisted; only the pads that are actually holding it are.
    """
    names = f"{pair.get('geom1','')} {pair.get('geom2','')}"
    bodies = f"{pair.get('body1','')} {pair.get('body2','')}"
    has_pad = any(token in names or token in bodies
                  for token in GRIPPER_PAD_TOKENS)
    has_target = any(token in bodies or token in names
                     for token in CARRIED_TARGET_TOKENS)
    return bool(has_pad and has_target)


def _carried_target_qpos_slice(model, task):
    """Free-joint qpos address of the carried target, if it has one."""
    for joint_id in range(int(model.njnt)):
        body_id = int(model.jnt_bodyid[joint_id])
        name = str(model.body(body_id).name or "")
        if not any(token in name for token in CARRIED_TARGET_TOKENS):
            continue
        if int(model.jnt_type[joint_id]) != 0:      # 0 == mjJNT_FREE
            continue
        address = int(model.jnt_qposadr[joint_id])
        return address, body_id
    return None, None


def diagnose(witness: dict[str, Any], scene: Path) -> dict[str, Any]:
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from molmo_spaces.tasks.pact_place_contact_audit import classify_contact
    from pact_geom_distance import gjk_distance, true_distance
    from pact_place_v105_clearance import (
        _shape, geom_shape_cache, pendant_geom_ids, robot_collision_geom_ids,
        target_collision_geom_ids,
    )
    from run_pact_place_expert_screen import _make_config
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    scratch = Path(tempfile.mkdtemp(prefix="v107_diag_"))
    task = sampler = None
    try:
        row = {
            "role_index": 0, "episode_id": "v107diag",
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
        steps = json.loads(
            (ROOT / witness["row_dir"] / "trajectory.json").read_text()
        )["steps"]
        apply_recorded_qpos(env, steps[witness["frame"]]["qpos"])
        mujoco.mj_forward(model, data)

        robot_view = env.current_robot.robot_view
        kinematics = env.current_robot.kinematics
        gripper_mg = robot_view.get_gripper_movegroup_ids()[0]
        probe = robot_collision_geom_ids(model) + target_collision_geom_ids(task)
        cache = geom_shape_cache(model, probe)
        pendant_ids = pendant_geom_ids(model, list(ALL_GEOMS_V106))
        component_geom = (
            f"pact_clutter_mount_v106_{witness.get('component') or 'lobe_0'}_g"
        )
        target_gid = int(model.geom(component_geom).id)

        saved = {k: np.asarray(v, dtype=float).copy()
                 for k, v in robot_view.get_qpos_dict().items()}
        saved_qpos = np.asarray(data.qpos, dtype=float).copy()
        address, target_body = _carried_target_qpos_slice(model, task)
        tcp0 = np.asarray(
            robot_view.get_gripper(gripper_mg).leaf_frame_to_world[:3, 3],
            dtype=float,
        ).copy()
        target_pos0 = (
            np.asarray(data.qpos[address:address + 3], dtype=float).copy()
            if address is not None else None
        )

        def move_target_rigidly(tcp_now: np.ndarray) -> None:
            """Translate the carried target by the same delta as the gripper."""
            if address is None or target_pos0 is None:
                return
            data.qpos[address:address + 3] = target_pos0 + (tcp_now - tcp0)

        def contact_pairs() -> dict[tuple[str, str], float]:
            out: dict[tuple[str, str], float] = {}
            for index in range(int(data.ncon)):
                contact = data.contact[index]
                if float(contact.dist) > 0.0:
                    continue
                g1, g2 = int(contact.geom1), int(contact.geom2)
                b1 = str(model.body(int(model.geom_bodyid[g1])).name or "")
                b2 = str(model.body(int(model.geom_bodyid[g2])).name or "")
                if not (b1.startswith("robot_0/") or b2.startswith("robot_0/")):
                    continue
                key = tuple(sorted((str(model.geom(g1).name or g1),
                                    str(model.geom(g2).name or g2))))
                out[key] = min(out.get(key, 0.0), float(contact.dist))
            return out

        baseline_pairs = contact_pairs()
        baseline_distance = float(true_distance(model, data, probe, [target_gid]))

        # Measured descent direction, chosen empirically with the target moved
        # rigidly so the probe reflects a real arm excursion.
        best_dir, best_reduction, probes = None, 0.0, []
        for axis in range(3):
            for sign in (+1.0, -1.0):
                candidate = np.zeros(3, dtype=float)
                candidate[axis] = sign
                robot_view.set_qpos_dict(saved)
                if address is not None:
                    data.qpos[address:address + 3] = target_pos0
                mujoco.mj_forward(model, data)
                pose = np.asarray(
                    robot_view.get_gripper(gripper_mg).leaf_frame_to_world,
                    dtype=float).copy()
                pose[:3, 3] = pose[:3, 3] + candidate * PROBE_STEP_M
                solution = kinematics.ik(
                    gripper_mg, pose, robot_view.move_group_ids(), saved,
                    base_pose=robot_view.base.pose)
                if solution is None:
                    probes.append({"axis": axis, "sign": sign, "ik_solved": False})
                    continue
                robot_view.set_qpos_dict(solution)
                mujoco.mj_forward(model, data)
                tcp_now = np.asarray(
                    robot_view.get_gripper(gripper_mg).leaf_frame_to_world[:3, 3],
                    dtype=float)
                move_target_rigidly(tcp_now)
                mujoco.mj_forward(model, data)
                value = float(true_distance(model, data, probe, [target_gid]))
                probes.append({"axis": axis, "sign": sign, "ik_solved": True,
                               "distance_m": value,
                               "reduction_m": baseline_distance - value})
                if baseline_distance - value > best_reduction:
                    best_reduction = baseline_distance - value
                    best_dir = candidate.copy()

        attempts: list[dict[str, Any]] = []
        reached = None
        worsened: list[dict[str, Any]] = []
        try:
            if best_dir is not None:
                for magnitude in MAGNITUDES_M:
                    robot_view.set_qpos_dict(saved)
                    if address is not None:
                        data.qpos[address:address + 3] = target_pos0
                    mujoco.mj_forward(model, data)
                    pose = np.asarray(
                        robot_view.get_gripper(gripper_mg).leaf_frame_to_world,
                        dtype=float).copy()
                    pose[:3, 3] = pose[:3, 3] + best_dir * float(magnitude)
                    solution = kinematics.ik(
                        gripper_mg, pose, robot_view.move_group_ids(), saved,
                        base_pose=robot_view.base.pose)
                    record: dict[str, Any] = {
                        "magnitude_m": float(magnitude),
                        "ik_solved": solution is not None,
                    }
                    if solution is None:
                        attempts.append(record)
                        continue
                    robot_view.set_qpos_dict(solution)
                    mujoco.mj_forward(model, data)
                    tcp_now = np.asarray(
                        robot_view.get_gripper(gripper_mg).leaf_frame_to_world[:3, 3],
                        dtype=float)
                    move_target_rigidly(tcp_now)
                    mujoco.mj_forward(model, data)

                    signed = float(true_distance(model, data, probe, [target_gid]))
                    shape_t = _shape(model, data, target_gid,
                                     geom_shape_cache(model, [target_gid]))
                    unsigned = float("inf")
                    for pid in probe:
                        shape = _shape(model, data, int(pid), cache)
                        if not shape.supported:
                            continue
                        unsigned = min(unsigned,
                                       float(gjk_distance(shape, shape_t)))
                        if unsigned == 0.0:
                            break
                    now = contact_pairs()
                    pendant_pairs, other_new, grasp_allowed, classes = [], [], [], set()
                    deeper: list[dict[str, Any]] = []
                    for key, distance in now.items():
                        g1n, g2n = key
                        pair = {"geom1": g1n, "geom2": g2n,
                                "body1": "", "body2": "", "distance_m": distance}
                        for index in range(int(data.ncon)):
                            c = data.contact[index]
                            a, b = int(c.geom1), int(c.geom2)
                            if tuple(sorted((str(model.geom(a).name or a),
                                             str(model.geom(b).name or b)))) == key:
                                pair["body1"] = str(
                                    model.body(int(model.geom_bodyid[a])).name or "")
                                pair["body2"] = str(
                                    model.body(int(model.geom_bodyid[b])).name or "")
                                break
                        if any(token in f"{g1n} {g2n}" for token in ALL_GEOMS_V106):
                            pendant_pairs.append(pair)
                            classes.add(classify_contact(pair))
                            continue
                        if key in baseline_pairs:
                            if distance < baseline_pairs[key] - 1e-9:
                                deeper.append({
                                    "pair": list(key),
                                    "baseline_m": baseline_pairs[key],
                                    "now_m": distance,
                                })
                            continue
                        if is_grasp_pad_contact(pair):
                            grasp_allowed.append(pair)
                            continue
                        other_new.append(pair)
                    if deeper:
                        worsened.append({"magnitude_m": float(magnitude),
                                         "pairs": deeper[:6]})
                    agree = bool(
                        pendant_pairs
                        and signed < 0.0
                        and unsigned == 0.0
                    )
                    record.update({
                        "signed_distance_m": signed,
                        "gjk_unsigned_m": (
                            None if not np.isfinite(unsigned) else unsigned),
                        "n_pendant_pairs": len(pendant_pairs),
                        "n_new_other_contacts": len(other_new),
                        "n_allowlisted_grasp_contacts": len(grasp_allowed),
                        "n_worsened_baseline_penetrations": len(deeper),
                        "instruments_agree_on_contact": agree,
                        "classes": sorted(classes),
                    })
                    attempts.append(record)
                    if pendant_pairs and agree and not other_new:
                        reached = {
                            "magnitude_m": float(magnitude),
                            "signed_distance_m": signed,
                            "classified_mounted_fixture": (
                                classes == {"mounted_fixture"}),
                        }
                        break
                    if other_new:
                        record["stopped_on_other_new_contact"] = True
                        break
        finally:
            robot_view.set_qpos_dict(saved)
            data.qpos[:] = saved_qpos
            mujoco.mj_forward(model, data)
        restored = bool(
            np.array_equal(np.asarray(data.qpos, dtype=float), saved_qpos)
        )
        return {
            **witness,
            "baseline_clearance_m": baseline_distance,
            "n_baseline_robot_contact_pairs": len(baseline_pairs),
            "carried_target_moved_rigidly": address is not None,
            "carried_target_body": (
                str(model.body(target_body).name) if target_body is not None else None
            ),
            "direction_probes": probes,
            "descent_direction": (
                None if best_dir is None else [float(v) for v in best_dir]
            ),
            "probe_reduction_m": best_reduction,
            "attempts": attempts,
            "reached_contact": reached,
            "worsened_baseline_penetrations": worsened,
            "state_restored": restored,
        }
    except Exception as error:  # noqa: BLE001
        return {**witness, "error": f"{type(error).__name__}: {error}"}
    finally:
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / DIAGNOSTIC_ROOT)
    parser.add_argument("--certification", type=Path,
                        default=ROOT / CERT_ROOT / "certification.json")
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
    cert_path = args.certification.resolve()
    cert = json.loads(cert_path.read_text())
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)

    minima = [w for w in cert["witnesses"] if w.get("role") == "group_minimum"]
    records = []
    for witness in sorted(minima, key=lambda w: w["group"]):
        scene = ROOT / cert["published_scenes"][witness["pose_id"]]["relative"]
        record = diagnose(witness, scene)
        records.append(record)
        print(json.dumps({
            "group": witness["group"],
            "reached": bool(record.get("reached_contact")),
            "worsened": len(record.get("worsened_baseline_penetrations") or []),
            "rigid_target": record.get("carried_target_moved_rigidly"),
        }), flush=True)

    document = {
        "schema_version": "pact_place_v107_contact_diagnostic_v1",
        "contract_version": CONTRACT_VERSION_V107,
        "diagnostic_only": True,
        "gates_qualification": False,
        "outcome_does_not_gate_this_successor": True,
        "specification_payload_sha256": recompute_payload_sha256(
            args.specification.resolve()),
        "drift_check": drift,
        "certification_payload_sha256": recompute_payload_sha256(cert_path),
        "selected_key": cert["selected_key"],
        "magnitudes_m": list(MAGNITUDES_M),
        "repairs": {
            "carried_target_moves_rigidly_with_gripper": True,
            "allowlist": "gripper-pad to carried-target grasp contacts only",
            "allowlist_excludes_robot_link_to_target": True,
            "tracks_worsening_baseline_penetrations": True,
            "requires_gjk_and_distance_and_live_agreement": True,
        },
        "groups": records,
        "n_groups": len(records),
        "n_reached_contact": sum(
            1 for r in records if r.get("reached_contact")),
        "n_with_worsened_baseline": sum(
            1 for r in records if r.get("worsened_baseline_penetrations")),
        "all_state_restored": all(r.get("state_restored") for r in records),
        "creates_episode": False, "calls_env_step": False,
        "elapsed_s": time.time() - started,
        **empty_authorization(),
    }
    hashes = write_immutable_create_only(
        output_root / "contact_diagnostic.json", document)
    print(json.dumps({
        "diagnostic_only": True, "gates_qualification": False,
        "n_groups": len(records),
        "n_reached_contact": document["n_reached_contact"],
        "n_with_worsened_baseline": document["n_with_worsened_baseline"],
        "all_state_restored": document["all_state_restored"],
        **hashes,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
