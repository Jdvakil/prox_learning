#!/usr/bin/env python3
"""V10.5 Step 3: small-deviation contact certificate and raw proximity causality.

Two proofs, both on retained state. Neither is an episode and neither steps the
environment: the robot is placed at recorded qpos, perturbed by a preregistered
displacement solved with sequential IK, measured, and restored.

The contact certificate is the operative proof that the pendant is close enough
to matter. Risk is never certified by moving the pendant, changing a collision
size, or reusing the old 83-175 mm diagnostic translations.
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

from pact_place_v105_contract import (  # noqa: E402
    CAUSAL_MAX_SIDE_RATIO,
    CAUSAL_MIN_CHANGED_SENSORS,
    CAUSAL_MIN_CHANGED_VALUES,
    CAUSAL_MIN_ONSET_FRAMES,
    CAUSAL_MIN_ONSET_SECONDS,
    CAUSAL_ROOT,
    CONTACT_CERTIFICATE_MAGNITUDES_M,
    CONTRACT_VERSION_V105,
    ENVIRONMENT_VERSION_V105,
    PROXIMITY_TENSOR_SHAPE,
    SITING_ROOT,
    empty_authorization,
    recompute_payload_sha256,
    sha256_file,
    write_immutable_create_only,
)
from pact_place_v105_geometry import (  # noqa: E402
    POSE_IDS,
    POSE_OFFSETS_M,
    POSE_ORDERING_MIN_SEPARATION_M,
    build_assembly,
    scene_xml_text,
)

BASE_SCENES = (
    ROOT
    / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes"
)


def build_scene_bundle(assembly: dict[str, Any] | None, destination: Path) -> Path:
    """A self-sufficient scene directory: V3/V5 includes plus metadata."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("pact_place_corridor_v3.xml", "pact_place_corridor_v5.xml"):
        shutil.copyfile(BASE_SCENES / name, destination / name)
    stem = "pact_place_v105_probe"
    metadata = BASE_SCENES / "pact_place_corridor_v5_metadata.json"
    if metadata.is_file():
        shutil.copyfile(metadata, destination / f"{stem}_metadata.json")
    path = destination / f"{stem}.xml"
    path.write_text(scene_xml_text(assembly))
    return path


def contact_certificate(job: dict[str, Any]) -> dict[str, Any]:
    """Perturb the robot toward the binding component until it truly touches."""
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from molmo_spaces.tasks.pact_place_contact_audit import classify_contact
    from pact_geom_distance import gjk_distance, true_distance
    from pact_place_v105_clearance import (
        _shape,
        geom_shape_cache,
        pendant_geom_ids,
        robot_collision_geom_ids,
        target_collision_geom_ids,
    )
    from pact_place_v105_contract import v95_row_payload
    from run_pact_place_expert_screen import _make_config
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    assembly = build_assembly(
        job["x_m"], job["r_m"], POSE_OFFSETS_M[job["pose_id"]],
        pose_id=job["pose_id"],
    )
    steps = json.loads(
        (ROOT / job["row_dir"] / "trajectory.json").read_text()
    )["steps"]
    frame = int(job["frame"])
    payload = v95_row_payload(job["family_id"], job["intrusion_side"])
    row = {
        "role_index": 0, "episode_id": job["row_dir"],
        "intrusion_side": job["intrusion_side"],
        "task_seed_u32": int(job["seed_u32"]),
        "task_seed_u64": int(job["seed_u32"]),
        "sampler_class": "PactPlaceCorridorV93Sampler",
        **payload,
    }
    scratch = Path(tempfile.mkdtemp(prefix="v105_cert_"))
    task = sampler = None
    try:
        scene = build_scene_bundle(assembly, scratch / "scene")
        config = _make_config(
            scratch / "d.json", scene_xml=scene,
            sampler_class="PactPlaceCorridorV93Sampler",
        )
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(job["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=1)
        env = task.env
        model, data = env.current_model, env.current_data
        apply_recorded_qpos(env, steps[frame]["qpos"])
        mujoco.mj_forward(model, data)

        robot_view = env.current_robot.robot_view
        kinematics = env.current_robot.kinematics
        gripper_mg = robot_view.get_gripper_movegroup_ids()[0]
        probe = robot_collision_geom_ids(model) + target_collision_geom_ids(task)
        cache = geom_shape_cache(model, probe)
        component_geom = f"pact_clutter_mount_v105_{job['component']}_g"
        target_gid = int(model.geom(component_geom).id)
        pendant_ids = pendant_geom_ids(
            model, [item["geom"] for item in assembly["components"]]
        )
        saved = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in robot_view.get_qpos_dict().items()
        }
        saved_qpos = np.asarray(data.qpos, dtype=float).copy()

        # Measured separation direction, from the closest points of the pair.
        component = next(
            item for item in assembly["components"] if item["name"] == job["component"]
        )
        centre = np.asarray(component["center_m"], dtype=float)
        tcp = np.asarray(
            robot_view.get_gripper(gripper_mg).leaf_frame_to_world[:3, 3], dtype=float
        )
        direction = centre - tcp
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return {**job, "certified": False, "reason": "degenerate direction"}
        direction = direction / norm
        baseline = float(true_distance(model, data, probe, [target_gid]))

        def solve(pose, seed):
            return kinematics.ik(
                gripper_mg, pose, robot_view.move_group_ids(), seed,
                base_pose=robot_view.base.pose,
            )

        attempts: list[dict[str, Any]] = []
        certified = None
        try:
            for magnitude in CONTACT_CERTIFICATE_MAGNITUDES_M:
                robot_view.set_qpos_dict(saved)
                mujoco.mj_forward(model, data)
                pose = np.asarray(
                    robot_view.get_gripper(gripper_mg).leaf_frame_to_world, dtype=float
                ).copy()
                pose[:3, 3] = pose[:3, 3] + direction * float(magnitude)
                solution = solve(pose, saved)
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
                target_cache = geom_shape_cache(model, [target_gid])
                target_shape = _shape(model, data, target_gid, target_cache)
                unsigned = float("inf")
                for gid in probe:
                    shape = _shape(model, data, int(gid), cache)
                    if not shape.supported:
                        continue
                    unsigned = min(unsigned, float(gjk_distance(shape, target_shape)))
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
                    is_robot = pair["body1"].startswith("robot_0/") or pair[
                        "body2"
                    ].startswith("robot_0/")
                    if not is_robot:
                        continue
                    if g1 in pendant_ids or g2 in pendant_ids:
                        pendant_pairs.append(pair)
                        classes.add(classify_contact(pair))
                    else:
                        other_pairs.append(pair)
                record.update(
                    {
                        "signed_distance_m": signed,
                        "gjk_unsigned_m": unsigned,
                        "live_pendant_pairs": pendant_pairs[:4],
                        "n_live_pendant_pairs": len(pendant_pairs),
                        "other_new_contacts": other_pairs[:4],
                        "n_other_new_contacts": len(other_pairs),
                        "classes": sorted(classes),
                    }
                )
                attempts.append(record)
                if pendant_pairs and not other_pairs:
                    certified = {
                        "magnitude_m": float(magnitude),
                        "signed_distance_m": signed,
                        "gjk_reports_intersection": unsigned == 0.0,
                        "signed_reports_penetration": signed < 0.0,
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
            # Restore on success, failure and exception alike.
            robot_view.set_qpos_dict(saved)
            data.qpos[:] = saved_qpos
            mujoco.mj_forward(model, data)

        agree = bool(
            certified
            and certified["signed_reports_penetration"]
            and certified["gjk_reports_intersection"]
            and certified["live_reports_contact"]
            and certified["classified_mounted_fixture"]
        )
        return {
            **job,
            "baseline_clearance_m": baseline,
            "separation_direction": [float(v) for v in direction],
            "attempts": attempts,
            "certified": bool(certified is not None and agree),
            "certificate": certified,
            "max_magnitude_m": CONTACT_CERTIFICATE_MAGNITUDES_M[-1],
            "state_restored": True,
            "creates_episode": False,
            "calls_env_step": False,
        }
    except Exception as error:  # noqa: BLE001 - recorded, never silently dropped
        return {**job, "certified": False,
                "error": f"{type(error).__name__}: {error}"}
    finally:
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


def closest_witnesses(
    npz_path: Path, x_m: float, r_m: float
) -> list[dict[str, Any]]:
    """The closest clean retained witness for each pose_id x side group.

    Derived from the recorded per-row scores rather than re-measured, so the
    certificate probes exactly the frames the siting pass identified.
    """
    payload = np.load(npz_path, allow_pickle=True)
    rows = [json.loads(str(item)) for item in payload["rows"]]
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not row.get("ok"):
            continue
        side = str(row["intrusion_side"])
        for pose in POSE_IDS:
            score = row["scores"].get(f"{x_m:.3f}|{r_m:.3f}|{pose}")
            if not score or score.get("min_lobe_stem_m") is None:
                continue
            value = float(score["min_lobe_stem_m"])
            key = (pose, side)
            if key in best and best[key]["min_lobe_stem_m"] <= value:
                continue
            witness = score.get("risk_witness") or {}
            best[key] = {
                "pose_id": pose,
                "intrusion_side": side,
                "x_m": float(x_m),
                "r_m": float(r_m),
                "row_dir": row["row_dir"],
                "family_id": row["family_id"],
                "seed_u32": int(row["seed_u32"]),
                "min_lobe_stem_m": value,
                "frame": int(witness.get("frame") or 0),
                "component": str(witness.get("box") or "lobe_0"),
                "probe_body": witness.get("probe_body"),
                "phase": witness.get("phase"),
            }
    return [best[key] for key in sorted(best)]


def pose_ordering(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The three poses must be behaviourally distinct, in the right order."""
    out: dict[str, Any] = {}
    for side in ("left", "right"):
        by_pose = {
            pose: [
                float(item["baseline_clearance_m"])
                for item in records
                if item["intrusion_side"] == side and item["pose_id"] == pose
                and item.get("baseline_clearance_m") is not None
            ]
            for pose in POSE_IDS
        }
        medians = {
            pose: (float(np.median(values)) if values else None)
            for pose, values in by_pose.items()
        }
        if any(value is None for value in medians.values()):
            out[side] = {"ordered": False, "reason": "missing pose witness",
                         "medians_m": medians}
            continue
        if side == "left":
            ordered = medians["pos5"] < medians["center"] < medians["neg5"]
            spread = medians["neg5"] - medians["pos5"]
        else:
            ordered = medians["neg5"] < medians["center"] < medians["pos5"]
            spread = medians["pos5"] - medians["neg5"]
        out[side] = {
            "medians_m": medians,
            "ordered": bool(ordered),
            "closest_to_farthest_spread_m": float(spread),
            "spread_sufficient": bool(spread >= POSE_ORDERING_MIN_SEPARATION_M),
            "required_spread_m": POSE_ORDERING_MIN_SEPARATION_M,
        }
    out["passed"] = all(
        isinstance(value, dict) and value.get("ordered")
        and value.get("spread_sufficient")
        for key, value in out.items()
        if key in ("left", "right")
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / CAUSAL_ROOT)
    parser.add_argument(
        "--siting", type=Path, default=ROOT / SITING_ROOT / "siting.json"
    )
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
        raise SystemExit(
            "Step 2 siting did not select a bundle; Step 3 does not run. "
            f"stop_reason={siting.get('stop_reason')!r}"
        )
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")

    selected = siting["selected"]
    witnesses = closest_witnesses(
        siting_path.parent / "per_row_scores.npz",
        float(selected["x_m"]),
        float(selected["r_m"]),
    )
    certificates = [contact_certificate(job) for job in witnesses]
    ordering = pose_ordering(certificates)
    all_certified = bool(certificates) and all(
        item["certified"] for item in certificates
    )
    document = {
        "schema_version": "pact_place_v105_step3_v1",
        "contract_version": CONTRACT_VERSION_V105,
        "environment_version": ENVIRONMENT_VERSION_V105,
        "siting_payload_sha256": recompute_payload_sha256(siting_path),
        "siting_raw_file_sha256": sha256_file(siting_path),
        "selected_x_m": selected["x_m"],
        "selected_r_m": selected["r_m"],
        "magnitudes_m": list(CONTACT_CERTIFICATE_MAGNITUDES_M),
        "certificates": certificates,
        "n_groups": len(certificates),
        "all_groups_certified": all_certified,
        "pose_ordering": ordering,
        "proximity_tensor_shape": list(PROXIMITY_TENSOR_SHAPE),
        "causal_thresholds": {
            "min_changed_values": CAUSAL_MIN_CHANGED_VALUES,
            "min_changed_sensors": CAUSAL_MIN_CHANGED_SENSORS,
            "min_onset_frames": CAUSAL_MIN_ONSET_FRAMES,
            "min_onset_seconds": CAUSAL_MIN_ONSET_SECONDS,
            "max_side_ratio": CAUSAL_MAX_SIDE_RATIO,
        },
        "risk_certified_by_moving_the_robot_not_the_pendant": True,
        "creates_episode": False,
        "calls_env_step": False,
        "elapsed_s": time.time() - started,
        **empty_authorization(),
        "step3_passed": bool(all_certified and ordering.get("passed")),
    }
    hashes = write_immutable_create_only(output_root / "step3.json", document)
    print(json.dumps({
        "step3_passed": document["step3_passed"],
        "n_groups": len(certificates),
        "all_groups_certified": all_certified,
        "pose_ordering_passed": ordering.get("passed"),
        **hashes,
    }, indent=2))
    return 0 if document["step3_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
