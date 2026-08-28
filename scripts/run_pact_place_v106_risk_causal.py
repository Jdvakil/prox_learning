#!/usr/bin/env python3
"""V10.6 Step 4b: contact-risk certificate and raw proximity causality.

The contact-risk certificate perturbs the ROBOT toward the pendant along the
measured separation direction until it genuinely touches. The pendant is never
moved, resized, or otherwise edited -- that is what makes this a statement
about the environment rather than about a diagnostic prop.

The causal check compares the selected compiled scene against the compiled
no-pendant scene at byte-identical robot, target, panel and clutter state, and
renders the real [40, 4, 8, 8] proximity tensor. A geometry proxy cannot pass.

No ``env.step`` and no new episode occur here.
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
    CAUSAL_ROOT,
    CERT_ROOT,
    CONTRACT_VERSION_V106,
    ENVIRONMENT_VERSION_V106,
    INTRUSION_SIDES,
    empty_authorization,
    recompute_payload_sha256,
    sha256_file,
    v95_row_payload,
    write_immutable_create_only,
)
from pact_place_v106_geometry import POSE_IDS  # noqa: E402
from run_pact_place_v9_v0c3_causal_proximity import (  # noqa: E402
    ABS_DELTA_FLOOR_M,
    _causal_metrics,
    _render_observation,
)

CONTACT_MAGNITUDES_M = (0.005, 0.010, 0.015, 0.020, 0.025, 0.030)
CAUSAL_MIN_CHANGED_VALUES = 448
CAUSAL_MIN_CHANGED_SENSORS = 3
CAUSAL_LINK_TOKENS = ("link5", "link6")
CAUSAL_MIN_ONSET_FRAMES = 5
CAUSAL_MIN_ONSET_SECONDS = 0.10
CAUSAL_MAX_SIDE_RATIO = 4.0
CAUSAL_WINDOW_FRAMES = 60
POLICY_PERIOD_S = 0.066


def _load(scene: Path, family: str, side: str, seed: int, scratch: Path,
          *, proximity: bool = False):
    from run_pact_place_expert_screen import _make_config

    row = {
        "role_index": 0, "episode_id": "v106", "intrusion_side": side,
        "task_seed_u32": int(seed), "task_seed_u64": int(seed),
        "sampler_class": "PactPlaceCorridorV93Sampler",
        **v95_row_payload(family, side),
    }
    config = _make_config(scratch / "d.json", scene_xml=scene,
                          sampler_class="PactPlaceCorridorV93Sampler")
    if proximity:
        config.proximity_sensor_period_ms = 16.6667
    sampler = config.task_sampler_config.task_sampler_class(config)
    sampler.seed_task_sampling(int(seed))
    sampler.set_pact_manifest_row(row)
    task = sampler.sample_task(house_index=1)
    return task, sampler, row


CARRIED_TARGET_TOKENS = ("cavity_obj_", "grasp_target")


def _is_grasp_pair(pair: dict[str, Any]) -> bool:
    """A robot geom against the carried target is the grasp, not a collision.

    These witnesses are all loaded-outbound frames, so the cup is held. Solving
    IK moves the arm while the recorded cup pose stays put, which perturbs the
    pad-to-cup contact set. Counting that as a new collision aborts the probe
    before it can ever reach the pendant.
    """
    blob = f"{pair.get('body1','')} {pair.get('body2','')}"
    return any(token in blob for token in CARRIED_TARGET_TOKENS)


def contact_risk(witness: dict[str, Any], scene: Path) -> dict[str, Any]:
    """Move the robot, never the pendant, until it truly touches."""
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from molmo_spaces.tasks.pact_place_contact_audit import classify_contact
    from pact_geom_distance import gjk_distance, true_distance
    from pact_place_v105_clearance import (
        _shape, geom_shape_cache, pendant_geom_ids, robot_collision_geom_ids,
        target_collision_geom_ids,
    )
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    scratch = Path(tempfile.mkdtemp(prefix="v106_risk_"))
    task = sampler = None
    try:
        task, sampler, _ = _load(
            scene, witness["family_id"], witness["intrusion_side"],
            witness["seed_u32"], scratch,
        )
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
        component_geom = f"pact_clutter_mount_v106_{witness['component']}_g"
        target_gid = int(model.geom(component_geom).id)
        pendant_ids = pendant_geom_ids(
            model, [f"pact_clutter_mount_v106_{n}_g"
                    for n in ("lobe_0", "lobe_1", "stem_0", "stem_1", "crossbar")]
        )
        saved = {k: np.asarray(v, dtype=float).copy()
                 for k, v in robot_view.get_qpos_dict().items()}
        saved_qpos = np.asarray(data.qpos, dtype=float).copy()
        baseline = float(true_distance(model, data, probe, [target_gid]))

        # Contacts already present in the retained state -- gripper on the cup,
        # robot self-contacts, robot on the bench. These are NOT caused by the
        # perturbation and must not be counted as a new collision.
        def robot_contact_pairs() -> set[tuple[str, str]]:
            pairs: set[tuple[str, str]] = set()
            for index in range(int(data.ncon)):
                contact = data.contact[index]
                if float(contact.dist) > 0.0:
                    continue
                g1, g2 = int(contact.geom1), int(contact.geom2)
                b1 = str(model.body(int(model.geom_bodyid[g1])).name or "")
                b2 = str(model.body(int(model.geom_bodyid[g2])).name or "")
                if not (b1.startswith("robot_0/") or b2.startswith("robot_0/")):
                    continue
                n1 = str(model.geom(g1).name or g1)
                n2 = str(model.geom(g2).name or g2)
                pairs.add(tuple(sorted((n1, n2))))
            return pairs



        baseline_pairs = robot_contact_pairs()

        # Measured descent direction. The limiting pair is a robot link against
        # the pendant, not the TCP against it, so a TCP-to-centre vector can
        # move the limiting link the wrong way -- it did, on the left side.
        # Probe the six axis directions and keep whichever actually reduces the
        # measured distance most.
        saved_for_probe = {k: np.asarray(v, dtype=float).copy()
                           for k, v in robot_view.get_qpos_dict().items()}
        probe_step = 0.005
        best_direction, best_delta, probes = None, 0.0, []
        for axis in range(3):
            for sign in (+1.0, -1.0):
                candidate = np.zeros(3, dtype=float)
                candidate[axis] = sign
                robot_view.set_qpos_dict(saved_for_probe)
                mujoco.mj_forward(model, data)
                pose = np.asarray(
                    robot_view.get_gripper(gripper_mg).leaf_frame_to_world,
                    dtype=float,
                ).copy()
                pose[:3, 3] = pose[:3, 3] + candidate * probe_step
                solution = kinematics.ik(
                    gripper_mg, pose, robot_view.move_group_ids(),
                    saved_for_probe, base_pose=robot_view.base.pose,
                )
                if solution is None:
                    probes.append({"axis": axis, "sign": sign, "ik_solved": False})
                    continue
                robot_view.set_qpos_dict(solution)
                mujoco.mj_forward(model, data)
                value = float(true_distance(model, data, probe, [target_gid]))
                delta = baseline - value
                probes.append({"axis": axis, "sign": sign, "ik_solved": True,
                               "distance_m": value, "reduction_m": delta})
                if delta > best_delta:
                    best_delta, best_direction = delta, candidate.copy()
        robot_view.set_qpos_dict(saved_for_probe)
        mujoco.mj_forward(model, data)
        if best_direction is None:
            return {**witness, "certified": False,
                    "reason": "no probed direction reduced the measured distance",
                    "direction_probes": probes,
                    "baseline_clearance_m": baseline}
        direction = best_direction

        attempts: list[dict[str, Any]] = []
        certified = None
        try:
            for magnitude in CONTACT_MAGNITUDES_M:
                robot_view.set_qpos_dict(saved)
                mujoco.mj_forward(model, data)
                pose = np.asarray(
                    robot_view.get_gripper(gripper_mg).leaf_frame_to_world,
                    dtype=float,
                ).copy()
                pose[:3, 3] = pose[:3, 3] + direction * float(magnitude)
                solution = kinematics.ik(
                    gripper_mg, pose, robot_view.move_group_ids(), saved,
                    base_pose=robot_view.base.pose,
                )
                record: dict[str, Any] = {
                    "magnitude_m": float(magnitude),
                    "ik_solved": solution is not None,
                }
                if solution is None:
                    attempts.append(record)
                    continue
                robot_view.set_qpos_dict(solution)
                mujoco.mj_forward(model, data)
                signed = float(true_distance(model, data, probe, [target_gid]))
                shape_t = _shape(model, data, target_gid,
                                 geom_shape_cache(model, [target_gid]))
                unsigned = float("inf")
                for pid in probe:
                    shape = _shape(model, data, int(pid), cache)
                    if not shape.supported:
                        continue
                    unsigned = min(unsigned, float(gjk_distance(shape, shape_t)))
                    if unsigned == 0.0:
                        break
                pendant_pairs, other_pairs, classes = [], [], set()
                for index in range(int(data.ncon)):
                    contact = data.contact[index]
                    if float(contact.dist) > 0.0:
                        continue
                    g1, g2 = int(contact.geom1), int(contact.geom2)
                    pair = {
                        "geom1": str(model.geom(g1).name or g1),
                        "geom2": str(model.geom(g2).name or g2),
                        "body1": str(model.body(int(model.geom_bodyid[g1])).name or ""),
                        "body2": str(model.body(int(model.geom_bodyid[g2])).name or ""),
                        "distance_m": float(contact.dist),
                    }
                    if not (pair["body1"].startswith("robot_0/")
                            or pair["body2"].startswith("robot_0/")):
                        continue
                    if g1 in pendant_ids or g2 in pendant_ids:
                        pendant_pairs.append(pair)
                        classes.add(classify_contact(pair))
                        continue
                    if tuple(sorted((pair["geom1"], pair["geom2"]))) in (
                        baseline_pairs
                    ):
                        continue                    # inherited from the retained state
                    if _is_grasp_pair(pair):
                        continue                    # the grasp itself, not a collision
                    other_pairs.append(pair)        # a genuinely new collision
                record.update({
                    "signed_distance_m": signed,
                    "gjk_unsigned_m": None if not np.isfinite(unsigned) else unsigned,
                    "n_live_pendant_pairs": len(pendant_pairs),
                    "n_other_new_contacts": len(other_pairs),
                    "classes": sorted(classes),
                })
                attempts.append(record)
                if pendant_pairs and not other_pairs:
                    certified = {
                        "magnitude_m": float(magnitude),
                        "signed_reports_penetration": signed < 0.0,
                        "gjk_reports_intersection": unsigned == 0.0,
                        "live_reports_contact": True,
                        "classified_mounted_fixture": classes == {"mounted_fixture"},
                        "first_new_collision_is_the_pendant": True,
                        "pairs": pendant_pairs[:4],
                    }
                    break
                if other_pairs:
                    record["first_new_collision_is_the_pendant"] = False
                    break
        finally:
            robot_view.set_qpos_dict(saved)
            data.qpos[:] = saved_qpos
            mujoco.mj_forward(model, data)
        restored = bool(np.allclose(np.asarray(data.qpos, dtype=float),
                                    saved_qpos, atol=0.0, rtol=0.0))
        agree = bool(
            certified and certified["signed_reports_penetration"]
            and certified["gjk_reports_intersection"]
            and certified["live_reports_contact"]
            and certified["classified_mounted_fixture"]
        )
        return {
            **witness,
            "baseline_clearance_m": baseline,
            "n_baseline_robot_contact_pairs": len(baseline_pairs),
            "direction_probes": probes,
            "descent_direction": [float(v) for v in direction],
            "probe_reduction_m": best_delta,
            "attempts": attempts,
            "certificate": certified,
            "state_restored": restored,
            "certified": bool(certified is not None and agree and restored),
        }
    except Exception as error:  # noqa: BLE001
        return {**witness, "certified": False,
                "error": f"{type(error).__name__}: {error}"}
    finally:
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


def causal_side(
    witness: dict[str, Any], scene: Path, control_scene: Path
) -> dict[str, Any]:
    """Selected scene vs no-pendant control at byte-identical state."""
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    steps = json.loads(
        (ROOT / witness["row_dir"] / "trajectory.json").read_text()
    )["steps"]
    closest = int(witness["frame"])
    lo = max(0, closest - CAUSAL_WINDOW_FRAMES)
    indices = list(range(lo, closest + 1))
    phases = [str(steps[i].get("policy_phase") or "") for i in indices]

    stacks: dict[str, np.ndarray] = {}
    sensor_names: list[str] = []
    for label, path in (("present", scene), ("no_pendant", control_scene)):
        scratch = Path(tempfile.mkdtemp(prefix=f"v106_causal_{label}_"))
        task = sampler = None
        try:
            task, sampler, _ = _load(
                path, witness["family_id"], witness["intrusion_side"],
                witness["seed_u32"], scratch, proximity=True,
            )
            task.reset()
            model, data = task.env.mj_model, task.env.current_data
            names = list(task._proximity_camera_names)
            if len(names) != 40 or len(set(names)) != 40:
                raise RuntimeError("V10.6 requires the frozen 40-sensor suite")
            sensor_names = names
            frames, repeat = [], []
            for index in indices:
                apply_recorded_qpos(task.env, steps[index]["qpos"])
                mujoco.mj_forward(model, data)
                frames.append(_render_observation(task, names))
                if label == "present":
                    mujoco.mj_forward(model, data)
                    repeat.append(_render_observation(task, names))
            stacks[label] = np.stack(frames).astype(np.float32)
            if label == "present":
                stacks["present_repeat"] = np.stack(repeat).astype(np.float32)
        finally:
            cleanup_episode_resources(
                task=task, policy=None, task_sampler=sampler,
                preloaded_policy=None, close_task_sampler=sampler is not None,
            )
            shutil.rmtree(scratch, ignore_errors=True)

    shape = tuple(stacks["present"].shape[1:])
    if shape[0] != 40 or shape[-2:] != (8, 8):
        raise RuntimeError(f"unexpected tensor contract {shape}")
    repeat_delta = float(
        np.max(np.abs(stacks["present"] - stacks["present_repeat"]))
    )
    threshold = max(float(ABS_DELTA_FLOOR_M), 10.0 * repeat_delta)
    metrics = _causal_metrics(
        stacks["present"], stacks["no_pendant"], sensor_names,
        np.asarray(indices, dtype=np.int32), phases, threshold,
    )
    responding = [
        item for item in (metrics.get("per_sensor") or [])
        if int(item.get("changed_values") or 0) > 0
    ]
    links = {str(item["link"]) for item in responding}
    delta = np.abs(
        stacks["present"].astype(np.float64) - stacks["no_pendant"].astype(np.float64)
    )
    per_frame = (delta > threshold).reshape(len(indices), -1).sum(axis=1)
    first = next(
        (i for i, count in enumerate(per_frame.tolist()) if count > 0), None
    )
    onset_frames = None if first is None else int(len(indices) - 1 - first)
    return {
        "pose_id": witness["pose_id"],
        "intrusion_side": witness["intrusion_side"],
        "row_dir": witness["row_dir"],
        "closest_frame": closest,
        "n_frames": len(indices),
        "tensor_shape": list(shape),
        "deterministic_repeat_max_abs_delta": repeat_delta,
        "deterministic": repeat_delta == 0.0,
        "threshold_m": threshold,
        "changed_values": int(metrics.get("changed_values") or 0),
        "changed_sensors": len(responding),
        "responding_links": sorted(links),
        "has_link5_or_link6": bool(
            any(token in link for link in links for token in CAUSAL_LINK_TOKENS)
        ),
        "onset_frames_before_closest": onset_frames,
        "onset_seconds_before_closest": (
            None if onset_frames is None else onset_frames * POLICY_PERIOD_S
        ),
        "per_frame_changed_counts": per_frame.astype(int).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / CAUSAL_ROOT)
    parser.add_argument("--certification", type=Path,
                        default=ROOT / CERT_ROOT / "certification.json")
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    started = time.time()

    cert_path = args.certification.resolve()
    cert = json.loads(cert_path.read_text())
    if not cert.get("certification_passed"):
        raise SystemExit("V10.6 certification did not pass; Step 4b does not run")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")

    minima = [w for w in cert["witnesses"] if w.get("role") == "group_minimum"]
    scenes = cert["published_scenes"]
    control = ROOT / cert["no_pendant_scene"]["relative"]

    print(f"contact-risk certificate on {len(minima)} group minima", flush=True)
    risk = []
    for witness in minima:
        record = contact_risk(witness, ROOT / scenes[witness["pose_id"]]["relative"])
        risk.append(record)
        print(json.dumps({
            "pose": witness["pose_id"], "side": witness["intrusion_side"],
            "certified": record["certified"],
            "magnitude_m": (record.get("certificate") or {}).get("magnitude_m"),
        }), flush=True)

    print("raw proximity causal, both sides", flush=True)
    causal = []
    for side in INTRUSION_SIDES:
        witness = min(
            (w for w in minima if w["intrusion_side"] == side),
            key=lambda w: w["compiled_exact_min_m"],
        )
        record = causal_side(
            witness, ROOT / scenes[witness["pose_id"]]["relative"], control
        )
        causal.append(record)
        print(json.dumps({
            "side": side, "changed_values": record["changed_values"],
            "changed_sensors": record["changed_sensors"],
            "deterministic": record["deterministic"],
            "onset_frames": record["onset_frames_before_closest"],
        }), flush=True)

    by_side = {r["intrusion_side"]: r for r in causal}
    values = [r["changed_values"] for r in causal]
    ratio = (max(values) / min(values)) if min(values) > 0 else float("inf")
    causal_checks = {
        "deterministic_control_repeat": all(r["deterministic"] for r in causal),
        "min_changed_values": all(
            r["changed_values"] >= CAUSAL_MIN_CHANGED_VALUES for r in causal
        ),
        "min_changed_sensors": all(
            r["changed_sensors"] >= CAUSAL_MIN_CHANGED_SENSORS for r in causal
        ),
        "link5_or_link6_responds": all(r["has_link5_or_link6"] for r in causal),
        "onset_frames_ok": all(
            (r["onset_frames_before_closest"] or 0) >= CAUSAL_MIN_ONSET_FRAMES
            for r in causal
        ),
        "onset_seconds_ok": all(
            (r["onset_seconds_before_closest"] or 0.0) >= CAUSAL_MIN_ONSET_SECONDS
            for r in causal
        ),
        "side_ratio_ok": ratio <= CAUSAL_MAX_SIDE_RATIO,
    }
    all_risk = bool(risk) and all(r["certified"] for r in risk)
    document = {
        "schema_version": "pact_place_v106_risk_causal_v1",
        "contract_version": CONTRACT_VERSION_V106,
        "environment_version": ENVIRONMENT_VERSION_V106,
        "certification_payload_sha256": recompute_payload_sha256(cert_path),
        "certification_raw_file_sha256": sha256_file(cert_path),
        "selected_key": cert["selected_key"],
        "contact_risk_magnitudes_m": list(CONTACT_MAGNITUDES_M),
        "contact_risk": risk,
        "all_groups_contact_certified": all_risk,
        "risk_certified_by_moving_the_robot_not_the_pendant": True,
        "causal": causal,
        "causal_side_ratio": ratio,
        "causal_thresholds": {
            "min_changed_values": CAUSAL_MIN_CHANGED_VALUES,
            "min_changed_sensors": CAUSAL_MIN_CHANGED_SENSORS,
            "min_onset_frames": CAUSAL_MIN_ONSET_FRAMES,
            "min_onset_seconds": CAUSAL_MIN_ONSET_SECONDS,
            "max_side_ratio": CAUSAL_MAX_SIDE_RATIO,
        },
        "causal_checks": causal_checks,
        "causal_passed": all(causal_checks.values()),
        "creates_episode": False, "calls_env_step": False,
        "elapsed_s": time.time() - started,
        **empty_authorization(),
        "step4b_passed": bool(all_risk and all(causal_checks.values())),
    }
    hashes = write_immutable_create_only(output_root / "risk_causal.json", document)
    print(json.dumps({
        "step4b_passed": document["step4b_passed"],
        "all_groups_contact_certified": all_risk,
        "causal_passed": document["causal_passed"],
        "causal_checks": causal_checks,
        **hashes,
    }, indent=2))
    return 0 if document["step4b_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
