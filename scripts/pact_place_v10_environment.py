#!/usr/bin/env python3
"""Exact environment predicates for V10 compound assemblies.

Panels and pendant components are axis-aligned boxes, so AABB overlap is the
exact panel-intersection predicate. Mesh clutter uses hardened GJK whenever
AABBs do not prove separation. Reconstruction uses mj_forward only after the
frozen V9.5 sample restore; scoring does not call env.step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pact_geom_distance import CONTACT_DISTANCE_M, GeomShape, convex_hull_vertices, geom_world_aabb, gjk_distance, mesh_vertices
from pact_place_corridor_contract import sha256_file
from pact_place_v10_compound_pendant_contract import (
    ALL_GEOMS,
    HOOD_TOP_BOTTOM_Z_M,
    HOOD_TOP_GEOM_NAME,
    N_CLEAN_CELLS,
    PENDANT_BODY,
    component_aabb,
)
from pact_place_v10_exact import _box_shape, verify_v99_inputs
from pact_place_v10_geometry import active_components
from pact_place_v10_runtime import establish_v10_runtime_env, write_immutable
from pact_place_v99_exact import snapshot_jobs_from_reconstruction
from reconstruct_pact_place_v99_baseline import (
    cleanup_task,
    collision_enabled_robot_geom_ids,
    pack_geom_catalog,
    pickup_collision_geom_ids,
)

ROOT = Path(__file__).resolve().parents[1]
PANEL_PREFIX = "pact_intrusion_"
CLUTTER_PREFIX = "pact_clutter_"
ROBOT_PREFIX = "robot_0/"


def _catalog_verts(catalog: dict[str, np.ndarray], index: int) -> np.ndarray | None:
    start = int(catalog["mesh_start"][index])
    count = int(catalog["mesh_count"][index])
    if start < 0 or count <= 0:
        return None
    return np.asarray(catalog["mesh_verts"][start : start + count], dtype=np.float64)


def classify_environment_geom(name: str, body: str) -> str:
    if name == HOOD_TOP_GEOM_NAME:
        return "hood_top"
    if name.startswith(PANEL_PREFIX) or body.startswith(PANEL_PREFIX):
        return "panel"
    if body.startswith(CLUTTER_PREFIX) or name.startswith(CLUTTER_PREFIX):
        return "clutter"
    if "Cup" in body or "pickup" in body.lower():
        return "target"
    return "static"


def _mat_is_axis_aligned(mat: np.ndarray, *, atol: float = 1e-8) -> bool:
    matrix = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    return bool(np.allclose(np.abs(matrix), np.eye(3), atol=atol))


def aabb_overlap(
    lo_a: Sequence[float],
    hi_a: Sequence[float],
    lo_b: Sequence[float],
    hi_b: Sequence[float],
) -> bool:
    lo_a_v = np.asarray(lo_a, dtype=np.float64)
    hi_a_v = np.asarray(hi_a, dtype=np.float64)
    lo_b_v = np.asarray(lo_b, dtype=np.float64)
    hi_b_v = np.asarray(hi_b, dtype=np.float64)
    return bool(np.all(lo_a_v <= hi_b_v + 1e-12) and np.all(lo_b_v <= hi_a_v + 1e-12))


def hood_top_attachment_face_only(
    component: dict[str, Any], geom: dict[str, Any]
) -> bool:
    if component.get("role") != "crossbar":
        return False
    if str(geom.get("name")) != HOOD_TOP_GEOM_NAME and str(geom.get("role")) != "hood_top":
        return False
    _lo, hi = component_aabb(component["center_m"], component["half_m"])
    if float(hi[2]) > HOOD_TOP_BOTTOM_Z_M + 1e-8:
        return False
    return abs(float(hi[2]) - HOOD_TOP_BOTTOM_Z_M) <= 1e-8


def prepare_v10_parked_task(row: dict[str, Any], *, seed_u32: int | None = None):
    """Restore a frozen cell in the V10 scene with the compound pendant parked."""
    from pact_place_v10_compound_pendant_contract import SCENE_XML_RELATIVE

    patched = dict(row)
    patched["pact_v10_pendant_parked"] = True
    patched["sampler_class"] = "PactPlaceCorridorV10CompoundPendantSampler"
    return _prepare_with_scene(
        patched,
        seed_u32=seed_u32,
        scene_xml=ROOT / SCENE_XML_RELATIVE,
        sampler_class="PactPlaceCorridorV10CompoundPendantSampler",
    )


def _prepare_with_scene(row, *, seed_u32, scene_xml, sampler_class):
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime
    from molmo_spaces.env.abstract_sensors import SensorSuite
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV5Sampler
    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
    from run_pact_place_expert_screen import _make_config
    import tempfile

    assert_supported_runtime(strict=True)
    original_settle = PactPlaceCorridorV5Sampler._settle_injected_object

    def _settle_for_reconstruction(self, env):
        try:
            return original_settle(self, env)
        except ValueError as error:
            if "overlap" not in str(error) and "escapes the episode shell" not in str(error):
                raise
            return None

    PactPlaceCorridorV5Sampler._settle_injected_object = _settle_for_reconstruction
    try:
        scratch = Path(tempfile.mkdtemp(prefix="pact_place_v10_env_"))
        config = _make_config(
            scratch / "dummy.json",
            scene_xml=scene_xml,
            sampler_class=sampler_class,
        )
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(seed_u32 if seed_u32 is not None else row["task_seed_u32"]))
        sampler.set_pact_manifest_row(row)
        try:
            task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
            if task is None:
                raise RuntimeError("sample_task returned None")
            task._sensor_suite = SensorSuite(
                [task._sensor_suite.sensors[uuid] for uuid in ("qpos", "tcp_pose")]
            )
            task.reset()
            return task, sampler, scratch
        except (ValueError, HouseInvalidForTask, RuntimeError) as error:
            cleanup_task(None, sampler, scratch)
            raise RuntimeError(f"V10 environment restore failed: {error}") from error
    finally:
        PactPlaceCorridorV5Sampler._settle_injected_object = original_settle


def dump_environment_geoms_from_task(task) -> list[dict[str, Any]]:
    import mujoco

    env = task.env
    model, data = env.current_model, env.current_data
    mujoco.mj_forward(model, data)
    robot_ids = set(collision_enabled_robot_geom_ids(model))
    target_ids = set(pickup_collision_geom_ids(task))
    pendant_ids = set()
    for name in ALL_GEOMS:
        try:
            pendant_ids.add(int(model.geom(name).id))
        except KeyError:
            continue
    records: list[dict[str, Any]] = []
    env_ids: list[int] = []
    for geom_id in range(int(model.ngeom)):
        if geom_id in robot_ids or geom_id in target_ids or geom_id in pendant_ids:
            continue
        body = str(model.body(int(model.geom_bodyid[geom_id])).name or "")
        if body.startswith(ROBOT_PREFIX):
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue
        if body.startswith(PENDANT_BODY):
            continue
        env_ids.append(int(geom_id))
    catalog = pack_geom_catalog(model, env_ids)
    box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
    for index, geom_id in enumerate(env_ids):
        name = str(model.geom(int(geom_id)).name or "")
        body = str(model.body(int(model.geom_bodyid[int(geom_id)])).name or "")
        verts = _catalog_verts(catalog, index)
        if verts is not None:
            verts = convex_hull_vertices(verts)
        shape = GeomShape(
            int(catalog["gtype"][index]),
            data.geom_xpos[int(geom_id)],
            data.geom_xmat[int(geom_id)],
            catalog["size"][index],
            verts,
        )
        try:
            if shape.supported:
                lo, hi = shape.world_aabb()
            else:
                lo, hi = geom_world_aabb(model, data, int(geom_id))
        except ValueError:
            lo, hi = geom_world_aabb(model, data, int(geom_id))
        records.append(
            {
                "geom_id": int(geom_id),
                "name": name,
                "body": body,
                "role": classify_environment_geom(name, body),
                "gtype": int(catalog["gtype"][index]),
                "size": np.asarray(catalog["size"][index], dtype=np.float64),
                "pos": np.asarray(data.geom_xpos[int(geom_id)], dtype=np.float64),
                "mat": np.asarray(data.geom_xmat[int(geom_id)], dtype=np.float64).reshape(3, 3),
                "verts": verts,
                "lo": np.asarray(lo, dtype=np.float64),
                "hi": np.asarray(hi, dtype=np.float64),
                "axis_aligned_box": bool(
                    int(catalog["gtype"][index]) == box_type
                    and _mat_is_axis_aligned(data.geom_xmat[int(geom_id)])
                ),
            }
        )
    records.sort(key=lambda item: (str(item["role"]), str(item["name"]), int(item["geom_id"])))
    return records


def live_assembly_environment_report(model, data, assembly: dict[str, Any], geoms: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """GJK/true_distance of posed pendant geoms vs dumped environment geoms."""
    import mujoco
    from pact_geom_distance import true_distance
    from pact_place_v10_scene import pose_assembly_on_data

    pose_assembly_on_data(model, data, assembly, parked=False)
    mujoco.mj_forward(model, data)
    dumped = score_assembly_environment(
        assembly, [{"role_index": -1, "intrusion_side": None, "geoms": list(geoms)}]
    )
    live_contacts = []
    for item in active_components(assembly):
        gid = int(model.geom(item["geom"]).id)
        for geom in geoms:
            if hood_top_attachment_face_only(item, geom):
                continue
            env_id = int(geom["geom_id"])
            distance = float(true_distance(model, data, [gid], [env_id]))
            if distance <= CONTACT_DISTANCE_M:
                live_contacts.append(
                    {
                        "component": item.get("name"),
                        "env_name": geom["name"],
                        "role": geom["role"],
                        "distance_m": distance,
                    }
                )
    return {
        "dumped_panel_clear": bool(dumped["panel_clear"]),
        "dumped_environment_clear": bool(dumped["environment_clear"]),
        "live_contact_count": len(live_contacts),
        "live_environment_clear": len(live_contacts) == 0,
        "live_contacts": live_contacts[:32],
        "parity_ok": bool(dumped["environment_clear"]) == (len(live_contacts) == 0),
    }


def dump_cell_environments(
    *,
    output_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct the six frozen cells and dump collision-enabled environment geoms."""
    establish_v10_runtime_env()
    from run_pact_place_v7_replay_videos import apply_recorded_qpos as _apply

    reconstruction, _snapshot, cells = verify_v99_inputs()
    jobs = snapshot_jobs_from_reconstruction(reconstruction)
    dumped: list[dict[str, Any]] = []
    for job in jobs:
        result = json.loads((Path(job["row_dir"]) / "result.json").read_text())
        role_index = int(result["role_index"])
        cell_meta = next(item for item in cells if int(item["role_index"]) == role_index)
        trajectory = json.loads((Path(job["row_dir"]) / "trajectory.json").read_text())
        first_qpos = trajectory["steps"][0]["qpos"]
        task = sampler = scratch = None
        try:
            task, sampler, scratch = prepare_v10_parked_task(
                job["manifest_row"],
                seed_u32=(job.get("selected_seed") or {}).get("seed_u32"),
            )
            _apply(task.env, first_qpos)
            geoms = dump_environment_geoms_from_task(task)
            live_probes = {}
            from pact_place_v10_geometry import (
                planning_probe_assembly,
                planning_probe_v1_invalid_assembly,
            )
            from pact_place_v10_scene import pose_assembly_on_data
            import mujoco

            model, data = task.env.current_model, task.env.current_data
            for label, assembly in (
                ("probe_v1_invalid_panel_overlap", planning_probe_v1_invalid_assembly()),
                ("probe_v2", planning_probe_assembly()),
            ):
                pose_assembly_on_data(model, data, assembly, parked=False)
                mujoco.mj_forward(model, data)
                live_probes[label] = live_assembly_environment_report(
                    model, data, assembly, geoms
                )
            pose_assembly_on_data(model, data, None, parked=True)
            mujoco.mj_forward(model, data)
        finally:
            cleanup_task(task, sampler, scratch)
        dumped.append(
            {
                "role_index": role_index,
                "episode_id": result.get("episode_id"),
                "family": cell_meta.get("family"),
                "intrusion_side": cell_meta.get("intrusion_side"),
                "geoms": geoms,
                "n_geoms": len(geoms),
                "n_panels": sum(1 for item in geoms if item["role"] == "panel"),
                "n_clutter": sum(1 for item in geoms if item["role"] == "clutter"),
                "live_probes": live_probes,
            }
        )
    dumped.sort(key=lambda item: int(item["role_index"]))
    if len(dumped) != N_CLEAN_CELLS:
        raise RuntimeError(f"expected {N_CLEAN_CELLS} environment dumps, found {len(dumped)}")
    if output_root is not None:
        _write_environment_dump(output_root, dumped)
    return dumped


def _write_environment_dump(output_root: Path, dumped: list[dict[str, Any]]) -> str:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_cells": len(dumped),
        "cells": [
            {
                "role_index": item["role_index"],
                "episode_id": item["episode_id"],
                "family": item["family"],
                "intrusion_side": item["intrusion_side"],
                "n_geoms": item["n_geoms"],
                "n_panels": item["n_panels"],
                "n_clutter": item["n_clutter"],
                "panel_names": [
                    geom["name"] for geom in item["geoms"] if geom["role"] == "panel"
                ],
                "probe_v1_live_environment_clear": (
                    item.get("live_probes") or {}
                ).get("probe_v1_invalid_panel_overlap", {}).get("live_environment_clear"),
                "probe_v2_live_environment_clear": (
                    item.get("live_probes") or {}
                ).get("probe_v2", {}).get("live_environment_clear"),
                "probe_v1_parity_ok": (
                    item.get("live_probes") or {}
                ).get("probe_v1_invalid_panel_overlap", {}).get("parity_ok"),
                "probe_v2_parity_ok": (
                    item.get("live_probes") or {}
                ).get("probe_v2", {}).get("parity_ok"),
            }
            for item in dumped
        ],
        "physics_stepped": False,
        "episodes_run": False,
    }
    digest = write_immutable(output_root / "environment_cells.json", summary)
    packed_path = output_root / "environment_cells.npz"
    # Store compact AABB/panel arrays for the catalog scan.
    panel_lo = []
    panel_hi = []
    panel_cell = []
    panel_side = []
    panel_name = []
    for cell in dumped:
        for geom in cell["geoms"]:
            if geom["role"] != "panel":
                continue
            panel_lo.append(geom["lo"])
            panel_hi.append(geom["hi"])
            panel_cell.append(int(cell["role_index"]))
            panel_side.append(str(cell["intrusion_side"]))
            panel_name.append(str(geom["name"]))
    np.savez_compressed(
        packed_path,
        panel_lo=np.asarray(panel_lo, dtype=np.float64),
        panel_hi=np.asarray(panel_hi, dtype=np.float64),
        panel_cell=np.asarray(panel_cell, dtype=np.int32),
        panel_side=np.asarray(panel_side),
        panel_name=np.asarray(panel_name),
    )
    save_environment_geoms(output_root / "environment_geoms.pkl.gz", dumped)
    return digest


def _strip_live_probes(dumped: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for cell in dumped:
        item = dict(cell)
        item.pop("live_probes", None)
        out.append(item)
    return out


def save_environment_geoms(path: Path, dumped: Sequence[dict[str, Any]]) -> str:
    import gzip
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _strip_live_probes(dumped)
    with gzip.open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return sha256_file(path)


def load_environment_geoms(path: Path) -> list[dict[str, Any]]:
    import gzip
    import pickle

    with gzip.open(path, "rb") as handle:
        dumped = pickle.load(handle)
    if len(dumped) != N_CLEAN_CELLS:
        raise RuntimeError(f"expected {N_CLEAN_CELLS} pickled cells, found {len(dumped)}")
    return dumped


def panels_from_v95_result_json(jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Panel AABBs recorded in the frozen V9.5 result manifests."""
    boxes = []
    for job in jobs:
        result = json.loads((Path(job["row_dir"]) / "result.json").read_text())
        panel = None
        for candidate in (
            result.get("pact_v9_legacy_panel"),
            (result.get("scene_params") or {}).get("pact_v9_legacy_panel"),
            (result.get("task_info") or {}).get("pact_v9_legacy_panel"),
        ):
            if candidate:
                panel = candidate
                break
        if not panel:
            raise RuntimeError(f"missing pact_v9_legacy_panel in {job['row_dir']}")
        center = np.asarray(panel["center"], dtype=np.float64)
        half = np.asarray(panel["half"], dtype=np.float64)
        boxes.append(
            {
                "role_index": int(result["role_index"]),
                "intrusion_side": panel.get("side") or result.get("intrusion_side"),
                "name": str(panel.get("name") or ""),
                "lo": center - half,
                "hi": center + half,
                "axis_aligned_box": True,
            }
        )
    boxes.sort(key=lambda item: int(item["role_index"]))
    return boxes


def panel_aabb_provenance_ok(
    dumped_panels: Sequence[dict[str, Any]],
    manifest_panels: Sequence[dict[str, Any]],
    *,
    atol: float = 1e-6,
) -> dict[str, Any]:
    if len(dumped_panels) != len(manifest_panels):
        return {
            "ok": False,
            "reason": (
                f"panel count dumped={len(dumped_panels)} manifest={len(manifest_panels)}"
            ),
        }
    for dumped, manifest in zip(dumped_panels, manifest_panels):
        if int(dumped["role_index"]) != int(manifest["role_index"]):
            return {"ok": False, "reason": "role_index mismatch"}
        if not np.allclose(dumped["lo"], manifest["lo"], atol=atol):
            return {
                "ok": False,
                "reason": f"panel lo mismatch role {dumped['role_index']}",
                "dumped_lo": np.asarray(dumped["lo"]).tolist(),
                "manifest_lo": np.asarray(manifest["lo"]).tolist(),
            }
        if not np.allclose(dumped["hi"], manifest["hi"], atol=atol):
            return {
                "ok": False,
                "reason": f"panel hi mismatch role {dumped['role_index']}",
                "dumped_hi": np.asarray(dumped["hi"]).tolist(),
                "manifest_hi": np.asarray(manifest["hi"]).tolist(),
            }
    return {"ok": True, "n_panels": len(dumped_panels), "atol_m": atol}


def component_panel_aabb_contact(
    center_m: Sequence[float],
    half_m: Sequence[float],
    panel_lo: np.ndarray,
    panel_hi: np.ndarray,
) -> bool:
    lo, hi = component_aabb(center_m, half_m)
    return aabb_overlap(lo, hi, panel_lo, panel_hi)


def score_component_against_environment(
    component: dict[str, Any],
    geoms: Sequence[dict[str, Any]],
    *,
    keep_aabb_disproof: bool = False,
) -> dict[str, Any]:
    """Exact env contact for one component vs one cell's dumped geoms. No early exit."""
    box_shape, box_lo, box_hi = _box_shape(component["center_m"], component["half_m"])
    witnesses: list[dict[str, Any]] = []
    contact = False
    min_distance = None
    for geom in geoms:
        allowed = hood_top_attachment_face_only(component, geom)
        gap_delta = np.maximum(0.0, np.maximum(box_lo - geom["hi"], geom["lo"] - box_hi))
        aabb_gap = float(np.linalg.norm(gap_delta))
        overlapping = aabb_overlap(box_lo, box_hi, geom["lo"], geom["hi"])
        if not overlapping and aabb_gap > CONTACT_DISTANCE_M:
            if keep_aabb_disproof:
                witnesses.append(
                    {
                        "name": geom["name"],
                        "body": geom["body"],
                        "role": geom["role"],
                        "aabb_disproof": True,
                        "allowed_hood_top": allowed,
                        "contact": False,
                        "distance_m": None,
                        "phase": "environment",
                    }
                )
            continue
        if allowed:
            witnesses.append(
                {
                    "name": geom["name"],
                    "body": geom["body"],
                    "role": geom["role"],
                    "aabb_disproof": False,
                    "allowed_hood_top": True,
                    "contact": False,
                    "distance_m": 0.0,
                    "phase": "environment",
                }
            )
            continue
        if (
            geom.get("axis_aligned_box")
            and int(geom["gtype"]) == box_shape.gtype
        ):
            distance = 0.0 if overlapping else aabb_gap
        else:
            other = GeomShape(
                geom["gtype"], geom["pos"], geom["mat"], geom["size"], geom.get("verts")
            )
            if not other.supported:
                distance = 0.0 if overlapping else aabb_gap
            else:
                distance = float(gjk_distance(box_shape, other))
        hit = bool(distance <= CONTACT_DISTANCE_M)
        contact = contact or hit
        if min_distance is None or distance < min_distance:
            min_distance = float(distance)
        if hit or keep_aabb_disproof:
            witnesses.append(
                {
                    "name": geom["name"],
                    "body": geom["body"],
                    "role": geom["role"],
                    "aabb_disproof": False,
                    "allowed_hood_top": False,
                    "contact": hit,
                    "distance_m": float(distance),
                    "phase": "environment",
                }
            )
    panel_contact = any(
        item["contact"] and item["role"] == "panel" for item in witnesses
    )
    clutter_contact = any(
        item["contact"] and item["role"] == "clutter" for item in witnesses
    )
    static_contact = any(
        item["contact"] and item["role"] in {"static", "hood_top"} for item in witnesses
    )
    return {
        "role": component.get("role"),
        "name": component.get("name"),
        "slot": component.get("slot"),
        "environment_clear": not contact,
        "panel_clear": not panel_contact,
        "clutter_clear": not clutter_contact,
        "static_clear": not static_contact,
        "min_distance_m": min_distance,
        "witnesses": witnesses,
    }


def score_assembly_environment(
    assembly: dict[str, Any],
    cells_env: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """All active components × all six cells. No early termination."""
    components = active_components(assembly)
    per_component = []
    panel_clear = True
    env_clear = True
    for component in components:
        per_cell = []
        for cell in cells_env:
            report = score_component_against_environment(component, cell["geoms"])
            report["role_index"] = cell["role_index"]
            report["intrusion_side"] = cell.get("intrusion_side")
            per_cell.append(report)
            panel_clear = panel_clear and bool(report["panel_clear"])
            env_clear = env_clear and bool(report["environment_clear"])
        per_component.append(
            {
                "name": component.get("name"),
                "role": component.get("role"),
                "slot": component.get("slot"),
                "per_cell": per_cell,
                "panel_clear_all": all(item["panel_clear"] for item in per_cell),
                "environment_clear_all": all(item["environment_clear"] for item in per_cell),
            }
        )
    return {
        "assembly_id": assembly.get("assembly_id"),
        "panel_clear": bool(panel_clear),
        "environment_clear": bool(env_clear),
        "per_component": per_component,
        "evaluated_all_cells": True,
        "n_cells": len(cells_env),
        "n_components": len(components),
    }


def panel_boxes_from_dump(cells_env: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    boxes = []
    for cell in cells_env:
        side = str(cell.get("intrusion_side") or "")
        for geom in cell["geoms"]:
            if geom["role"] != "panel":
                continue
            if side and side not in str(geom.get("name") or ""):
                # Opposite-side panel is parked out of the corridor.
                continue
            boxes.append(
                {
                    "role_index": cell["role_index"],
                    "intrusion_side": cell.get("intrusion_side"),
                    "name": geom["name"],
                    "lo": np.asarray(geom["lo"], dtype=np.float64),
                    "hi": np.asarray(geom["hi"], dtype=np.float64),
                    "axis_aligned_box": bool(geom.get("axis_aligned_box", True)),
                }
            )
    return boxes


def _aabb_overlap_matrix(
    lo: np.ndarray, hi: np.ndarray, panel_lo: np.ndarray, panel_hi: np.ndarray
) -> np.ndarray:
    lo_v = np.asarray(lo, dtype=np.float64)[:, None, :]
    hi_v = np.asarray(hi, dtype=np.float64)[:, None, :]
    plo = np.asarray(panel_lo, dtype=np.float64)[None, :, :]
    phi = np.asarray(panel_hi, dtype=np.float64)[None, :, :]
    return np.all(lo_v <= phi + 1e-12, axis=2) & np.all(plo <= hi_v + 1e-12, axis=2)


def unique_rows(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    packed = np.asarray(array, dtype=np.float64)
    if packed.ndim == 1:
        packed = packed.reshape(-1, 1)
    view = np.ascontiguousarray(packed)
    unique, inverse = np.unique(view, axis=0, return_inverse=True)
    return unique, inverse.astype(np.int32)


def component_aabbs_from_two_lobe_keys(keys: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from pact_place_v10_compound_pendant_contract import (
        CEILING_TOP_Z_M,
        CROSSBAR_HEIGHT_M,
        STEM_HALF_M,
        STEM_TOP_Z_M,
    )

    packed = np.asarray(keys, dtype=np.float64)
    k0, k1 = packed[:, 0, :], packed[:, 1, :]
    lo0, hi0 = k0[:, :3] - k0[:, 3:], k0[:, :3] + k0[:, 3:]
    lo1, hi1 = k1[:, :3] - k1[:, 3:], k1[:, :3] + k1[:, 3:]
    n = packed.shape[0]
    stem_y0 = k0[:, 1] + np.where(k0[:, 1] < 0.0, -1.0, 1.0) * k0[:, 4]
    stem_y1 = k1[:, 1] + np.where(k1[:, 1] < 0.0, -1.0, 1.0) * k1[:, 4]
    lobe_top0 = k0[:, 2] + k0[:, 5]
    lobe_top1 = k1[:, 2] + k1[:, 5]
    stem_hz0 = (STEM_TOP_Z_M - lobe_top0) / 2.0
    stem_hz1 = (STEM_TOP_Z_M - lobe_top1) / 2.0
    stem_cz0 = lobe_top0 + stem_hz0
    stem_cz1 = lobe_top1 + stem_hz1
    stem_lo0 = np.stack(
        [k0[:, 0] - STEM_HALF_M, stem_y0 - STEM_HALF_M, stem_cz0 - stem_hz0], axis=1
    )
    stem_hi0 = np.stack(
        [k0[:, 0] + STEM_HALF_M, stem_y0 + STEM_HALF_M, stem_cz0 + stem_hz0], axis=1
    )
    stem_lo1 = np.stack(
        [k1[:, 0] - STEM_HALF_M, stem_y1 - STEM_HALF_M, stem_cz1 - stem_hz1], axis=1
    )
    stem_hi1 = np.stack(
        [k1[:, 0] + STEM_HALF_M, stem_y1 + STEM_HALF_M, stem_cz1 + stem_hz1], axis=1
    )
    y_lo = np.minimum(stem_y0, stem_y1) - STEM_HALF_M
    y_hi = np.maximum(stem_y0, stem_y1) + STEM_HALF_M
    bar_lo = np.stack(
        [k0[:, 0] - STEM_HALF_M, y_lo, np.full(n, CEILING_TOP_Z_M - CROSSBAR_HEIGHT_M)],
        axis=1,
    )
    bar_hi = np.stack(
        [k0[:, 0] + STEM_HALF_M, y_hi, np.full(n, CEILING_TOP_Z_M)],
        axis=1,
    )
    return {
        "lobe_0": (lo0, hi0),
        "lobe_1": (lo1, hi1),
        "stem_0": (stem_lo0, stem_hi0),
        "stem_1": (stem_lo1, stem_hi1),
        "crossbar": (bar_lo, bar_hi),
    }


def scan_panel_clear_mask(
    keys: np.ndarray, panels: Sequence[dict[str, Any]]
) -> np.ndarray:
    """Exact panel AABB intersection for every two-lobe row. True = panel-clear."""
    if not panels:
        raise ValueError("no posed intrusion panels in the reconstructed cells")
    panel_lo = np.stack([item["lo"] for item in panels])
    panel_hi = np.stack([item["hi"] for item in panels])
    aabbs = component_aabbs_from_two_lobe_keys(keys)
    hit = np.zeros(len(keys), dtype=bool)
    for _name, (lo, hi) in aabbs.items():
        overlap = _aabb_overlap_matrix(lo, hi, panel_lo, panel_hi)
        hit |= np.any(overlap, axis=1)
    return np.logical_not(hit)


def atomic_component_env_key(center_m: Sequence[float], half_m: Sequence[float]) -> tuple[float, ...]:
    from pact_place_v10_compound_pendant_contract import round_m

    return tuple(round_m(float(value)) for value in list(center_m) + list(half_m))


def score_unique_components_environment(
    components: Sequence[dict[str, Any]],
    cells_env: Sequence[dict[str, Any]],
) -> dict[tuple[float, ...], dict[str, Any]]:
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    for component in components:
        key = atomic_component_env_key(component["center_m"], component["half_m"])
        if key in cache:
            continue
        per_cell = []
        for cell in cells_env:
            report = score_component_against_environment(component, cell["geoms"])
            report["role_index"] = cell["role_index"]
            per_cell.append(report)
        cache[key] = {
            "key": list(key),
            "role": component.get("role"),
            "panel_clear_all": all(item["panel_clear"] for item in per_cell),
            "environment_clear_all": all(item["environment_clear"] for item in per_cell),
            "per_cell": per_cell,
        }
    return cache


def combine_assembly_environment_cache(
    assembly: dict[str, Any],
    cache: dict[tuple[float, ...], dict[str, Any]],
) -> dict[str, Any]:
    components = active_components(assembly)
    panel_clear = True
    env_clear = True
    missing = []
    for component in components:
        key = atomic_component_env_key(component["center_m"], component["half_m"])
        scored = cache.get(key)
        if scored is None:
            missing.append(key)
            continue
        panel_clear = panel_clear and bool(scored["panel_clear_all"])
        env_clear = env_clear and bool(scored["environment_clear_all"])
    if missing:
        raise KeyError(f"missing environment scores for {len(missing)} components")
    return {
        "assembly_id": assembly.get("assembly_id"),
        "panel_clear": bool(panel_clear),
        "environment_clear": bool(env_clear),
        "from_cache": True,
    }


def score_unique_keys_environment(
    unique_keys: np.ndarray,
    cells_env: Sequence[dict[str, Any]],
    *,
    role: str,
) -> dict[str, Any]:
    """Score unique center+half keys against dumped environment geoms."""
    packed = np.round(np.asarray(unique_keys, dtype=np.float64), 9)
    n_keys = int(packed.shape[0])
    panel_clear = np.ones(n_keys, dtype=bool)
    env_clear = np.ones(n_keys, dtype=bool)
    min_distance = np.full(n_keys, np.nan, dtype=np.float64)
    contact_witnesses: list[dict[str, Any]] = []
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    for index, row in enumerate(packed):
        component = {
            "role": role,
            "name": f"{role}_{index}",
            "center_m": row[:3].tolist(),
            "half_m": row[3:].tolist(),
        }
        per_cell = []
        for cell in cells_env:
            report = score_component_against_environment(
                component, cell["geoms"], keep_aabb_disproof=False
            )
            report["role_index"] = cell["role_index"]
            report["intrusion_side"] = cell.get("intrusion_side")
            per_cell.append(report)
            if not report["panel_clear"]:
                panel_clear[index] = False
            if not report["environment_clear"]:
                env_clear[index] = False
            distance = report.get("min_distance_m")
            if distance is not None and (
                np.isnan(min_distance[index]) or float(distance) < min_distance[index]
            ):
                min_distance[index] = float(distance)
            for witness in report["witnesses"]:
                if not witness.get("contact"):
                    continue
                contact_witnesses.append(
                    {
                        "component_index": int(index),
                        "role": role,
                        "center_m": component["center_m"],
                        "half_m": component["half_m"],
                        "role_index": cell["role_index"],
                        "intrusion_side": cell.get("intrusion_side"),
                        "geom": witness["name"],
                        "geom_role": witness["role"],
                        "distance_m": witness["distance_m"],
                        "phase": "environment",
                    }
                )
        key = tuple(float(value) for value in row)
        cache[key] = {
            "key": list(key),
            "role": role,
            "panel_clear_all": bool(panel_clear[index]),
            "environment_clear_all": bool(env_clear[index]),
            "min_distance_m": None
            if np.isnan(min_distance[index])
            else float(min_distance[index]),
            "per_cell": per_cell,
        }
        if (index + 1) % 250 == 0 or index + 1 == n_keys:
            print(
                f"[v10-env-{role}] {index + 1}/{n_keys}",
                flush=True,
            )
    return {
        "keys": packed,
        "panel_clear_all": panel_clear,
        "environment_clear_all": env_clear,
        "min_distance_m": min_distance,
        "cache": cache,
        "contact_witnesses": contact_witnesses,
        "role": role,
    }


def combine_row_flags(
    inverse: np.ndarray, unique_flags: np.ndarray
) -> np.ndarray:
    return np.asarray(unique_flags, dtype=bool)[np.asarray(inverse, dtype=np.int32)]


def assembly_row_environment_flags(
    *,
    lobe_inv: np.ndarray,
    stem_inv: np.ndarray,
    bar_inv: np.ndarray,
    lobe_env: np.ndarray,
    stem_env: np.ndarray,
    bar_env: np.ndarray,
    lobe_panel: np.ndarray,
    stem_panel: np.ndarray,
    bar_panel: np.ndarray,
) -> dict[str, np.ndarray]:
    lobe_env_rows = combine_row_flags(lobe_inv, lobe_env)
    stem_env_rows = combine_row_flags(stem_inv, stem_env)
    bar_env_rows = combine_row_flags(bar_inv, bar_env)
    lobe_panel_rows = combine_row_flags(lobe_inv, lobe_panel)
    stem_panel_rows = combine_row_flags(stem_inv, stem_panel)
    bar_panel_rows = combine_row_flags(bar_inv, bar_panel)
    if lobe_env_rows.ndim == 1:
        lobe_env_ok = lobe_env_rows
        lobe_panel_ok = lobe_panel_rows
    else:
        lobe_env_ok = np.all(lobe_env_rows, axis=1)
        lobe_panel_ok = np.all(lobe_panel_rows, axis=1)
    if stem_env_rows.ndim == 1:
        stem_env_ok = stem_env_rows
        stem_panel_ok = stem_panel_rows
    else:
        stem_env_ok = np.all(stem_env_rows, axis=1)
        stem_panel_ok = np.all(stem_panel_rows, axis=1)
    return {
        "panel_clear": lobe_panel_ok & stem_panel_ok & bar_panel_rows,
        "environment_clear": lobe_env_ok & stem_env_ok & bar_env_rows,
    }


def panels_for_side(
    panels: Sequence[dict[str, Any]], side: str
) -> list[dict[str, Any]]:
    return [item for item in panels if str(item.get("intrusion_side")) == str(side)]


def assembly_panel_clear_on_side(assembly: dict[str, Any], panels: Sequence[dict[str, Any]]) -> bool:
    from pact_place_v10_compound_pendant_contract import component_aabb as _aabb

    for component in active_components(assembly):
        lo, hi = _aabb(component["center_m"], component["half_m"])
        for panel in panels:
            if aabb_overlap(lo, hi, panel["lo"], panel["hi"]):
                return False
    return True


def merge_component_caches(
    *caches: dict[tuple[float, ...], dict[str, Any]],
) -> dict[tuple[float, ...], dict[str, Any]]:
    merged: dict[tuple[float, ...], dict[str, Any]] = {}
    for cache in caches:
        merged.update(cache)
    return merged


