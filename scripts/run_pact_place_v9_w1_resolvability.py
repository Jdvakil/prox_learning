#!/usr/bin/env python3
"""W1: replay the frozen V9.5 trajectories and retrodict the measured raw signals.

For every family/side variant of the V9.5 raw smoke this rebuilds the exact
scene, replays the retained qpos through ``mj_forward`` only (no physics step,
no render, no planner) and scores the three known hazards -- the active
intrusion panel, the inbound vessel and the outbound vessel -- with the
resolving-power instrument in ``pact_skin_resolvability``.

The output is a retrodiction: the model's predicted per-sensor response is
compared against the measured ``changed_values`` in the V9.5 raw prerequisite
``validation.json``.  If the ordering panel >> outbound vessel > inbound vessel
does not reproduce, the instrument must not be used to search, and this script
reports ``retrodiction_passed: false``.

Nothing here authorizes a gate, collection, or V1b.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for _path in (ROOT / "scripts", MOLMO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pact_skin_resolvability as psr  # noqa: E402
from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v9_contract import sha256_payload  # noqa: E402
from run_pact_place_v9_v0c3_causal_proximity import (  # noqa: E402
    INBOUND_DECISION_PHASES,
    OUTBOUND_DECISION_PHASES,
    SCENE_XML,
    _decision_indices,
    _find_episode_dir,
)

DEFAULT_SMOKE_ROOT = ROOT / "diagnostics_output" / "pact_place_v95_raw_smoke"
DEFAULT_MEASURED = (
    ROOT / "diagnostics_output" / "pact_place_v95_v0c5_raw_prerequisite" / "validation.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v9_w1_resolvability"
#: Roles are scored on the same phase window the validator measured them on.
ROLE_WINDOW = {
    "panel": OUTBOUND_DECISION_PHASES,
    "inbound_vessel": INBOUND_DECISION_PHASES,
    "outbound_vessel": OUTBOUND_DECISION_PHASES,
}
MESH_POINT_LIMIT = 240
PRIMITIVE_SPACING_M = 0.015


# --------------------------------------------------------------------------
# posed geometry extraction
# --------------------------------------------------------------------------
def _rel(path: Path) -> str:
    """Repo-relative path where possible, absolute otherwise (probe output roots)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def _subtree_body_ids(model, body_id: int) -> list[int]:
    found = [int(body_id)]
    index = 0
    while index < len(found):
        parent = found[index]
        index += 1
        found.extend(
            int(child)
            for child in range(model.nbody)
            if int(model.body_parentid[child]) == parent and int(child) != parent
        )
    return found


def _local_geom_points(model, geom_id: int) -> np.ndarray:
    """Surface points of one geom in its own frame."""
    import mujoco

    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        verts = np.asarray(
            model.mesh_vert[start : start + count], dtype=np.float64
        ).reshape(-1, 3)
        return psr.subsample_points(verts, MESH_POINT_LIMIT)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        half = size
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        half = np.array([size[0]] * 3)
    elif geom_type in (
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    ):
        cap = size[0] if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0
        half = np.array([size[0], size[0], size[1] + cap])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        half = size
    else:  # planes, height fields and anything exotic: fall back to rbound
        radius = float(model.geom_rbound[geom_id]) or 0.01
        half = np.array([radius] * 3)
    return psr.box_surface_points(np.zeros(3), half, spacing_m=PRIMITIVE_SPACING_M)


class HazardSource:
    """Cached local geometry for one hazard body, posed on demand."""

    def __init__(self, model, name: str, body_name: str) -> None:
        self.name = str(name)
        self.body_name = str(body_name)
        body_id = int(model.body(body_name).id)
        self.geom_ids: list[int] = []
        self.local_points: list[np.ndarray] = []
        for child in _subtree_body_ids(model, body_id):
            start = int(model.body_geomadr[child])
            for geom_id in range(start, start + int(model.body_geomnum[child])):
                if int(model.geom_group[geom_id]) not in psr.RENDERED_GEOM_GROUPS:
                    continue
                self.geom_ids.append(int(geom_id))
                self.local_points.append(_local_geom_points(model, geom_id))
        if not self.geom_ids:
            raise ValueError(f"hazard {name!r} exposes no renderable geom")
        self.geom_id_set = frozenset(self.geom_ids)

    def pose(self, data) -> psr.HazardGeometry:
        clouds: list[np.ndarray] = []
        boxes: list[tuple[np.ndarray, np.ndarray]] = []
        for geom_id, local in zip(self.geom_ids, self.local_points):
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            origin = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            world = local @ rotation.T + origin
            clouds.append(world)
            boxes.append((world.min(axis=0), world.max(axis=0)))
        points = np.concatenate(clouds, axis=0)
        return psr.HazardGeometry(self.name, points, boxes, self.geom_ids)


def _ray_hit_geoms(model, data, cam_pos: np.ndarray, cam_xmat: np.ndarray,
                   max_range_m: float) -> np.ndarray:
    """First-hit geom id for every (sensor, pixel), or -1 when nothing is struck."""
    import mujoco

    n_sensors = cam_pos.shape[0]
    out = np.full((n_sensors, psr.PIXEL_GRID * psr.PIXEL_GRID), -1, dtype=np.int32)
    geomid = np.zeros(1, dtype=np.int32)
    for sensor in range(n_sensors):
        origin = np.ascontiguousarray(cam_pos[sensor])
        dirs = psr.pixel_directions(cam_xmat[sensor])
        for pixel in range(dirs.shape[0]):
            distance = mujoco.mj_ray(
                model, data, origin, np.ascontiguousarray(dirs[pixel]),
                psr.MJ_GEOMGROUP, 1, -1, geomid,
            )
            if 0.0 <= distance <= max_range_m:
                out[sensor, pixel] = int(geomid[0])
    return out


# --------------------------------------------------------------------------
# per-variant replay
# --------------------------------------------------------------------------
def _run_variant(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_expert_screen import _make_config

    row = job["row"]
    result = json.loads(Path(job["result_path"]).read_text())
    steps = list(json.loads(Path(job["trajectory_path"]).read_text())["steps"])
    # The decision window is what the V9.5 validator measured, so it is what the
    # retrodiction must use.  The full window replays every retained step and
    # exists so W2 can score a cluster wherever the arm actually passes it,
    # rather than only where the *old* layout happened to name a phase.
    indices = (
        list(range(len(steps)))
        if str(job.get("window")) == "full"
        else _decision_indices(steps)
    )
    phases = [str(steps[index].get("policy_phase")) for index in indices]

    scratch = Path(tempfile.mkdtemp(prefix="pact_w1_resolvability_"))
    task = sampler = None
    try:
        config = _make_config(
            scratch / "dummy.json",
            scene_xml=SCENE_XML,
            sampler_class="PactPlaceCorridorV93Sampler",
        )
        config.proximity_sensor_period_ms = 16.6667
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("W1 replay sample_task returned None")
        task.reset()

        model = task.env.mj_model
        data = task.env.current_data
        sensor_names = list(task._proximity_camera_names)
        if len(sensor_names) != 40 or len(set(sensor_names)) != 40:
            raise RuntimeError(f"expected 40 unique proximity cameras: {sensor_names}")
        cam_ids = [int(model.camera(f"robot_0/{name}").id) for name in sensor_names]
        far_plane_m = float(model.stat.extent) * float(model.vis.map.zfar)
        max_range_m = min(float(job["max_range_m"]), far_plane_m)

        active_panel = f"pact_intrusion_{job['intrusion_side']}"
        panel_mocap_id = int(np.asarray(model.body(active_panel).mocapid).reshape(-1)[0])
        panel_position = np.asarray(data.mocap_pos[panel_mocap_id], dtype=float).copy()
        vessel_bodies = {
            str(obj["role"]): f"pact_clutter_{obj['palette_slot']}/{obj['uid']}"
            for obj in row["pact_clutter_layout"]["objects"]
            if str(obj.get("role")) in {"inbound_vessel", "outbound_vessel"}
        }
        sources = {
            "panel": HazardSource(model, "panel", active_panel),
            "inbound_vessel": HazardSource(
                model, "inbound_vessel", vessel_bodies["inbound_vessel"]
            ),
            "outbound_vessel": HazardSource(
                model, "outbound_vessel", vessel_bodies["outbound_vessel"]
            ),
        }

        n_frames = len(indices)
        n_sensors = len(sensor_names)
        cam_pos = np.zeros((n_frames, n_sensors, 3), dtype=np.float64)
        cam_xmat = np.zeros((n_frames, n_sensors, 3, 3), dtype=np.float64)
        arrays = {
            role: {
                key: np.zeros((n_frames, n_sensors), dtype=np.float64)
                for key in ("subtense_px", "range_m", "w_perp_m")
            }
            for role in sources
        }
        for role in sources:
            arrays[role]["aabb_pixel_hits"] = np.zeros((n_frames, n_sensors), np.int32)
            arrays[role]["ray_pixel_hits"] = np.zeros((n_frames, n_sensors), np.int32)
        hazard_extent_m = {role: np.zeros((n_frames, 3)) for role in sources}

        for local_index, trajectory_index in enumerate(indices):
            qpos = np.asarray(steps[trajectory_index]["qpos"], dtype=float)
            if qpos.shape != (int(model.nq),):
                raise RuntimeError(
                    f"qpos shape mismatch at {trajectory_index}: {qpos.shape} vs {model.nq}"
                )
            data.qpos[:] = qpos
            data.mocap_pos[panel_mocap_id] = panel_position
            mujoco.mj_forward(model, data)

            frame_pos = np.asarray(data.cam_xpos[cam_ids], dtype=np.float64)
            frame_xmat = np.asarray(data.cam_xmat[cam_ids], dtype=np.float64).reshape(-1, 3, 3)
            cam_pos[local_index] = frame_pos
            cam_xmat[local_index] = frame_xmat

            hit_geoms = _ray_hit_geoms(model, data, frame_pos, frame_xmat, max_range_m)
            for role, source in sources.items():
                hazard = source.pose(data)
                hazard_extent_m[role][local_index] = hazard.frontal_extent_m()
                metrics = psr.subtense_for_frame(frame_pos, frame_xmat, hazard, max_range_m)
                arrays[role]["subtense_px"][local_index] = metrics["subtense_px"]
                arrays[role]["range_m"][local_index] = metrics["range_m"]
                arrays[role]["w_perp_m"][local_index] = metrics["w_perp_m"]
                arrays[role]["aabb_pixel_hits"][local_index] = psr.aabb_pixel_hits_for_frame(
                    frame_pos, frame_xmat, hazard, max_range_m
                )
                owned = np.isin(hit_geoms, np.asarray(sorted(source.geom_id_set), np.int32))
                arrays[role]["ray_pixel_hits"][local_index] = owned.sum(axis=1)

        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "cam_pos": cam_pos.astype(np.float32),
            "cam_xmat": cam_xmat.astype(np.float32),
            "trajectory_indices": np.asarray(indices, dtype=np.int32),
            "policy_phases": np.asarray(phases, dtype="U40"),
            "sensor_names": np.asarray(sensor_names, dtype="U40"),
        }
        for role, block in arrays.items():
            for key, values in block.items():
                payload[f"{role}__{key}"] = (
                    values.astype(np.float32) if values.dtype == np.float64 else values
                )
            payload[f"{role}__extent_m"] = hazard_extent_m[role].astype(np.float32)
        np.savez_compressed(output_path, **payload)

        roles: dict[str, Any] = {}
        for role in sources:
            mask = np.asarray([phase in ROLE_WINDOW[role] for phase in phases], dtype=bool)
            if not mask.any():
                raise ValueError(f"no frames in the {role} decision window")
            # The role summaries always describe the validator's own window, so a
            # full-window replay stays comparable with the decision-window one.
            summary = psr.aggregate(arrays[role]["subtense_px"], sensor_names, mask)
            summary["window_phases"] = sorted(ROLE_WINDOW[role])
            summary["n_window_frames"] = int(mask.sum())
            summary["min_range_m"] = float(arrays[role]["range_m"][mask].min())
            summary["max_w_perp_m"] = float(arrays[role]["w_perp_m"][mask].max())
            summary["posed_extent_m"] = [
                float(value) for value in hazard_extent_m[role].max(axis=0)
            ]
            summary["aabb_prediction"] = psr.pixel_hit_summary(
                arrays[role]["aabb_pixel_hits"], sensor_names, mask
            )
            summary["ray_prediction"] = psr.pixel_hit_summary(
                arrays[role]["ray_pixel_hits"], sensor_names, mask
            )
            roles[role] = summary

        return {
            "family_id": job["family_id"],
            "intrusion_side": job["intrusion_side"],
            "episode_id": row["episode_id"],
            "row_sha256": row["row_sha256"],
            "cache_path": _rel(output_path),
            "cache_sha256": sha256_file(output_path),
            "n_window_frames": len(indices),
            "sensor_count": len(sensor_names),
            "far_plane_m": far_plane_m,
            "max_range_m": max_range_m,
            "active_panel": active_panel,
            "vessel_bodies": dict(vessel_bodies),
            "roles": roles,
        }
    finally:
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


# --------------------------------------------------------------------------
# retrodiction
# --------------------------------------------------------------------------
def _measured_index(measured: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(item["family_id"]), str(item["intrusion_side"])): item
        for item in measured["variants"]
    }


def _retrodict(results: list[dict[str, Any]], measured: dict[str, Any]) -> dict[str, Any]:
    index = _measured_index(measured)
    rows = []
    ordering_ok = True
    sensor_recall = []
    for result in results:
        key = (str(result["family_id"]), str(result["intrusion_side"]))
        variant = index.get(key)
        if variant is None:
            raise RuntimeError(f"no measured variant for {key}")
        entry: dict[str, Any] = {
            "family_id": key[0],
            "intrusion_side": key[1],
            "roles": {},
        }
        measured_values = {}
        predicted_values = {}
        for role in ("panel", "inbound_vessel", "outbound_vessel"):
            effect = variant[f"{role}_causal_effect"]
            measured_changed = int(effect["changed_values"])
            measured_sensors = {
                str(item["sensor_name"]): int(item["changed_values"])
                for item in effect["per_sensor"]
                if int(item["changed_values"]) > 0
            }
            ray = result["roles"][role]["ray_prediction"]
            predicted_changed = int(ray["predicted_changed_values"])
            predicted_sensors = dict(ray["predicted_changed_values_by_sensor"])
            measured_values[role] = measured_changed
            predicted_values[role] = predicted_changed
            hit = set(measured_sensors) & set(predicted_sensors)
            if measured_sensors:
                sensor_recall.append(len(hit) / len(measured_sensors))
            entry["roles"][role] = {
                "measured_changed_values": measured_changed,
                "predicted_changed_values": predicted_changed,
                "measured_changed_sensors": int(effect["changed_sensors"]),
                "predicted_changed_sensors": int(ray["n_responding_sensors"]),
                "measured_sensors": measured_sensors,
                "predicted_sensors": predicted_sensors,
                "sensors_recovered": sorted(hit),
                "sensors_missed": sorted(set(measured_sensors) - set(predicted_sensors)),
                "sensors_extra": sorted(set(predicted_sensors) - set(measured_sensors)),
                "max_subtense_px": float(result["roles"][role]["max_subtense_px"]),
                "n_sensors_ge_1px": int(result["roles"][role]["n_sensors_ge_1px"]),
                "n_sensors_ge_2px": int(result["roles"][role]["n_sensors_ge_2px"]),
                "min_range_m": float(result["roles"][role]["min_range_m"]),
                "max_w_perp_m": float(result["roles"][role]["max_w_perp_m"]),
            }
        # The claim under test: the panel dominates both vessels, and where a
        # vessel was measured at all the model must not rank it above the panel.
        panel_dominates = (
            predicted_values["panel"] > predicted_values["outbound_vessel"]
            and predicted_values["panel"] > predicted_values["inbound_vessel"]
        )
        measured_order = sorted(measured_values, key=lambda role: -measured_values[role])
        predicted_order = sorted(predicted_values, key=lambda role: -predicted_values[role])
        # Roles measured at exactly zero carry no ordering information.
        informative = [role for role in measured_order if measured_values[role] > 0]
        order_ok = [role for role in predicted_order if role in informative] == informative
        entry["panel_dominates"] = bool(panel_dominates)
        entry["measured_order"] = measured_order
        entry["predicted_order"] = predicted_order
        entry["informative_roles"] = informative
        entry["ordering_reproduced"] = bool(panel_dominates and order_ok)
        ordering_ok = ordering_ok and entry["ordering_reproduced"]
        rows.append(entry)

    flat = [
        (role_entry["measured_changed_values"], role_entry["predicted_changed_values"])
        for row in rows
        for role_entry in row["roles"].values()
    ]
    measured_arr = np.asarray([item[0] for item in flat], dtype=float)
    predicted_arr = np.asarray([item[1] for item in flat], dtype=float)
    nonzero = measured_arr > 0
    ratio = predicted_arr[nonzero] / measured_arr[nonzero]
    false_negatives = int(((measured_arr > 0) & (predicted_arr == 0)).sum())
    return {
        "per_variant": rows,
        "ordering_reproduced_in_all_variants": bool(ordering_ok),
        "n_role_measurements": int(measured_arr.size),
        "n_nonzero_measurements": int(nonzero.sum()),
        "predicted_over_measured_ratio": {
            "min": float(ratio.min()) if ratio.size else None,
            "median": float(np.median(ratio)) if ratio.size else None,
            "max": float(ratio.max()) if ratio.size else None,
        },
        "pearson_r_on_changed_values": float(
            np.corrcoef(measured_arr, predicted_arr)[0, 1]
        ) if measured_arr.size > 1 else None,
        "spearman_r_on_changed_values": float(
            np.corrcoef(
                np.argsort(np.argsort(measured_arr)).astype(float),
                np.argsort(np.argsort(predicted_arr)).astype(float),
            )[0, 1]
        ) if measured_arr.size > 1 else None,
        "measured_nonzero_predicted_zero": false_negatives,
        "mean_measured_sensor_recall": float(np.mean(sensor_recall)) if sensor_recall else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--measured", type=Path, default=DEFAULT_MEASURED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-range-m", type=float, default=4.0)
    parser.add_argument(
        "--window", choices=("decision", "full"), default="decision",
        help="decision: the V9.5 validator's phase window (required for retrodiction); "
             "full: every retained trajectory step, for W2 siting",
    )
    parser.add_argument("--family", action="append")
    parser.add_argument("--side", choices=("left", "right"), action="append")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("workers must be in [1, 4]")

    smoke_root = args.smoke_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    smoke_summary_path = smoke_root / "summary.json"
    smoke_summary = json.loads(smoke_summary_path.read_text())
    retained_rows = {
        (str(row["layout_family_id"]), str(row["intrusion_side"])): row
        for row in list(smoke_summary.get("manifest_rows") or [])
    }

    jobs = []
    for item in smoke_summary["results"]:
        family_id, side = str(item["family_id"]), str(item["intrusion_side"])
        if args.family and family_id not in set(args.family):
            continue
        if args.side and side not in set(args.side):
            continue
        row = retained_rows[(family_id, side)]
        if row["episode_id"] != item["episode_id"]:
            raise RuntimeError("retained smoke row reconstruction mismatch")
        episode_dir = _find_episode_dir(smoke_root, str(item["episode_id"]))
        jobs.append(
            {
                "family_id": family_id,
                "intrusion_side": side,
                "row": row,
                "result_path": str(episode_dir / "result.json"),
                "trajectory_path": str(episode_dir / "trajectory.json"),
                "output_path": str(output_root / "cache" / f"{family_id}_{side}.npz"),
                "max_range_m": float(args.max_range_m),
                "window": str(args.window),
            }
        )

    results: list[dict[str, Any]] = []
    if args.workers == 1:
        for job in jobs:
            print(f"Replaying {job['family_id']} / {job['intrusion_side']}", flush=True)
            results.append(_run_variant(job))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_variant, job): job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(
                    json.dumps(
                        {
                            "family_id": result["family_id"],
                            "side": result["intrusion_side"],
                            **{
                                f"{role}_predicted_changed": result["roles"][role][
                                    "ray_prediction"
                                ]["predicted_changed_values"]
                                for role in ("panel", "inbound_vessel", "outbound_vessel")
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                results.append(result)
    results.sort(key=lambda item: (item["family_id"], item["intrusion_side"]))

    measured = json.loads(args.measured.resolve().read_text())
    retrodiction = _retrodict(results, measured)
    # A full-window replay renders extra frames outside the validator's window,
    # so it may legitimately predict more than was measured; only the decision
    # window is a retrodiction of the V9.5 measurement.
    passed = bool(
        retrodiction["ordering_reproduced_in_all_variants"]
        and retrodiction["measured_nonzero_predicted_zero"] == 0
        and len(results) == 8
    )
    document = {
        "schema_version": "pact_place_v9_w1_resolvability_v1",
        "role": "non_authorizing_resolving_power_instrument",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "authorizes_v1b": False,
        "retrodiction_passed": passed,
        "instrument_path": "scripts/pact_skin_resolvability.py",
        "instrument_sha256": sha256_file(ROOT / "scripts/pact_skin_resolvability.py"),
        "driver_path": _rel(Path(__file__)),
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "measured_path": _rel(args.measured),
        "measured_sha256": sha256_file(args.measured.resolve()),
        "smoke_summary_path": _rel(smoke_summary_path),
        "smoke_summary_sha256": sha256_file(smoke_summary_path),
        "sensor_fov_deg": psr.SENSOR_FOV_DEG,
        "pixel_grid": psr.PIXEL_GRID,
        "pixel_pitch_coefficient_per_m": psr.PIXEL_PITCH_COEFF,
        "production_substeps": psr.PRODUCTION_SUBSTEPS,
        "resolving_floor_examples_m": {
            "one_px_at_0p50m": psr.resolving_floor_m(0.50, 1.0),
            "one_px_at_0p75m": psr.resolving_floor_m(0.75, 1.0),
            "two_px_at_0p50m": psr.resolving_floor_m(0.50, 2.0),
            "two_px_at_0p75m": psr.resolving_floor_m(0.75, 2.0),
        },
        "replay_window": str(args.window),
        "variant_count": len(results),
        "retrodiction": retrodiction,
        "variants": results,
    }
    document["document_sha256"] = sha256_payload(document)
    path = output_root / "resolvability.json"
    path.write_text(json.dumps(psr.jsonable(document), indent=2, sort_keys=True) + "\n")
    print(path, flush=True)
    print(
        json.dumps(
            {
                "retrodiction_passed": passed,
                "ordering_reproduced": retrodiction["ordering_reproduced_in_all_variants"],
                "measured_nonzero_predicted_zero": retrodiction[
                    "measured_nonzero_predicted_zero"
                ],
                "pearson_r": retrodiction["pearson_r_on_changed_values"],
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
