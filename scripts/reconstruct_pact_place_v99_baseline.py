#!/usr/bin/env python3
"""Independently reconstruct the frozen V9.5 eight-row baseline for V9.9.

Restores each row from its own manifest and retained qpos, calls mj_forward
only, and records TCP residual, canonical grasp posture, and collision AABBs.
Does not insert a pendant, step physics, or authorize later stages by itself.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import (  # noqa: E402
    PLACE_V5_SCENE_SHA256,
    sha256_file,
    sha256_payload,
)
from pact_place_v99_geometry import window_masks  # noqa: E402
from pact_place_v99_pendant_contract import (  # noqa: E402
    CONTRACT_VERSION,
    MAX_TCP_RESIDUAL_M,
    N_BASELINE_ROWS,
    SOURCE_SUMMARY_RELATIVE,
    empty_authorization,
)
from pact_geom_distance import (  # noqa: E402
    PRIMITIVE_GEOM_TYPES,
    GeomShape,
    convex_hull_vertices,
    mesh_vertices,
)

SOURCE_SUMMARY = ROOT / SOURCE_SUMMARY_RELATIVE
SCENE_XML = ROOT / (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v5.xml"
)
DEFAULT_OUTPUT = ROOT / "diagnostics_output/pact_place_v99_baseline_reconstruction"
IMPLEMENTATION_PATHS = (
    "scripts/reconstruct_pact_place_v99_baseline.py",
    "scripts/pact_place_v99_pendant_contract.py",
    "scripts/pact_place_v99_geometry.py",
    "scripts/pact_place_v99_route.py",
    "scripts/pact_place_v99_exact.py",
    "scripts/pact_geom_distance.py",
    "scripts/search_pact_place_v99_pendant.py",
)


def _json_vec(values: Any) -> list[float]:
    return [float(item) for item in np.asarray(values, dtype=np.float64).reshape(-1)]


def collision_enabled_robot_geom_ids(model) -> list[int]:
    ids: list[int] = []
    for geom_id in range(int(model.ngeom)):
        body = model.body(int(model.geom_bodyid[geom_id])).name or ""
        if not str(body).startswith("robot_0/"):
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue
        ids.append(int(geom_id))
    if not ids:
        raise RuntimeError("no collision-enabled robot geoms")
    return ids


def pickup_collision_geom_ids(task) -> list[int]:
    env = task.env
    model = env.current_model
    manager = env.object_managers[env.current_batch_index]
    pickup = manager.get_object_by_name(task.config.task_config.pickup_obj_name)
    if pickup is None:
        raise RuntimeError("pickup object missing during reconstruction")
    body_name = str(getattr(pickup, "name", "") or getattr(pickup, "body_name", ""))
    ids: list[int] = []
    for geom_id in range(int(model.ngeom)):
        body = model.body(int(model.geom_bodyid[geom_id])).name or ""
        if body_name and body_name not in str(body):
            continue
        if "Cup" not in str(body) and body_name not in str(body):
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue
        ids.append(int(geom_id))
    return ids


def _catalog_verts(catalog: dict[str, np.ndarray], index: int) -> np.ndarray | None:
    start = int(catalog["mesh_start"][index])
    count = int(catalog["mesh_count"][index])
    if start < 0 or count <= 0:
        return None
    return np.asarray(catalog["mesh_verts"][start : start + count], dtype=np.float64)


def pack_geom_catalog(model, geom_ids: Sequence[int]) -> dict[str, np.ndarray]:
    types = []
    sizes = []
    starts = []
    counts = []
    verts: list[list[float]] = []
    hull_by_mesh: dict[int, np.ndarray] = {}
    for geom_id in geom_ids:
        gtype = int(model.geom_type[int(geom_id)])
        size = np.zeros(3, dtype=np.float64)
        raw = np.asarray(model.geom_size[int(geom_id)], dtype=np.float64).reshape(-1)
        size[: min(3, raw.size)] = raw[:3]
        types.append(gtype)
        sizes.append(size)
        mesh = None if gtype in PRIMITIVE_GEOM_TYPES else mesh_vertices(model, int(geom_id))
        if mesh is None or len(mesh) == 0:
            starts.append(-1)
            counts.append(0)
        else:
            mesh_id = int(model.geom_dataid[int(geom_id)])
            if mesh_id not in hull_by_mesh:
                hull_by_mesh[mesh_id] = convex_hull_vertices(mesh)
            hull = hull_by_mesh[mesh_id]
            starts.append(len(verts))
            counts.append(int(len(hull)))
            verts.extend(hull.tolist())
    return {
        "gtype": np.asarray(types, dtype=np.int32),
        "size": np.asarray(sizes, dtype=np.float64),
        "mesh_start": np.asarray(starts, dtype=np.int32),
        "mesh_count": np.asarray(counts, dtype=np.int32),
        "mesh_verts": (
            np.zeros((0, 3), dtype=np.float64)
            if not verts
            else np.asarray(verts, dtype=np.float64)
        ),
    }


def arm_joint_qpos(model, data) -> tuple[np.ndarray, list[str]]:
    names = []
    values = []
    for joint_id in range(int(model.njnt)):
        name = str(model.joint(joint_id).name or "")
        if "fr3_joint" not in name and "panda_joint" not in name:
            continue
        address = int(model.jnt_qposadr[joint_id])
        names.append(name)
        values.append(float(data.qpos[address]))
        if len(values) == 7:
            break
    if len(values) != 7:
        values = [float(item) for item in np.asarray(data.qpos[:7], dtype=np.float64)]
        names = [f"qpos[{index}]" for index in range(7)]
    return np.asarray(values, dtype=np.float64), names


def tcp_rotation(env) -> np.ndarray:
    robot_view = env.current_robot.robot_view
    gripper_id = robot_view.get_gripper_movegroup_ids()[0]
    tcp = robot_view.get_gripper(gripper_id).leaf_frame_to_world
    return np.asarray(tcp[:3, :3], dtype=np.float64)


def prepare_task(row: dict[str, Any], *, seed_u32: int | None = None, retries: int = 1):
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime
    from molmo_spaces.env.abstract_sensors import SensorSuite
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV5Sampler
    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
    from run_pact_place_expert_screen import _make_config

    assert_supported_runtime(strict=True)
    original_settle = PactPlaceCorridorV5Sampler._settle_injected_object

    def _settle_for_reconstruction(self, env):
        try:
            return original_settle(self, env)
        except ValueError as error:
            # Reconstruction restores recorded qpos. The sampling overlap
            # gate is not a reconstruction failure.
            if "overlap" not in str(error) and "escapes the episode shell" not in str(
                error
            ):
                raise
            return None

    PactPlaceCorridorV5Sampler._settle_injected_object = _settle_for_reconstruction
    last_error: Exception | None = None
    try:
        for _attempt in range(max(1, int(retries))):
            scratch = Path(tempfile.mkdtemp(prefix="pact_place_v99_reconstruct_"))
            config = _make_config(
                scratch / "dummy.json",
                scene_xml=SCENE_XML,
                sampler_class=row.get("sampler_class") or "PactPlaceCorridorV93Sampler",
            )
            sampler = config.task_sampler_config.task_sampler_class(config)
            sampler.seed_task_sampling(int(seed_u32 if seed_u32 is not None else row["task_seed_u32"]))
            sampler.set_pact_manifest_row(row)
            try:
                task = sampler.sample_task(
                    house_index=int(row["scene_template_house_index"])
                )
                if task is None:
                    raise RuntimeError("sample_task returned None")
                task._sensor_suite = SensorSuite(
                    [task._sensor_suite.sensors[uuid] for uuid in ("qpos", "tcp_pose")]
                )
                task.reset()
                return task, sampler, scratch
            except (ValueError, HouseInvalidForTask, RuntimeError) as error:
                last_error = error
                cleanup_task(None, sampler, scratch)
                continue
        raise RuntimeError(
            f"role {row.get('role_index')} sample_task failed after retries: {last_error}"
        )
    finally:
        PactPlaceCorridorV5Sampler._settle_injected_object = original_settle


def cleanup_task(task, sampler, scratch: Path) -> None:
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources

    try:
        cleanup_episode_resources(
            task=task,
            policy=None,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=True,
        )
    except Exception:
        pass
    if scratch is not None:
        shutil.rmtree(scratch, ignore_errors=True)


def reconstruct_one_row(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    from run_pact_place_v7_replay_videos import apply_recorded_qpos, tcp_position_m

    row_dir = Path(payload["row_dir"])
    result = json.loads((row_dir / "result.json").read_text())
    trajectory = json.loads((row_dir / "trajectory.json").read_text())
    steps = list(trajectory["steps"])
    windows = window_masks([step.get("policy_phase") for step in steps])
    task = sampler = scratch = None
    try:
        selected = payload.get("selected_seed") or {}
        task, sampler, scratch = prepare_task(
            payload["manifest_row"],
            seed_u32=selected.get("seed_u32"),
        )
        env = task.env
        model, data = env.current_model, env.current_data
        robot_ids = collision_enabled_robot_geom_ids(model)
        target_ids = pickup_collision_geom_ids(task)
        robot_catalog = pack_geom_catalog(model, robot_ids)
        target_catalog = pack_geom_catalog(model, target_ids)
        robot_verts = [
            _catalog_verts(robot_catalog, index) for index in range(len(robot_ids))
        ]
        target_verts = [
            _catalog_verts(target_catalog, index) for index in range(len(target_ids))
        ]
        recorded_tcp = []
        live_tcp = []
        arm_q = []
        robot_lo = np.zeros((len(steps), len(robot_ids), 3), dtype=np.float64)
        robot_hi = np.zeros_like(robot_lo)
        target_lo = np.zeros((len(steps), max(len(target_ids), 1), 3), dtype=np.float64)
        target_hi = np.zeros_like(target_lo)
        robot_pos = np.zeros((len(steps), len(robot_ids), 3), dtype=np.float64)
        robot_mat = np.zeros((len(steps), len(robot_ids), 9), dtype=np.float64)
        target_pos = np.zeros((len(steps), max(len(target_ids), 1), 3), dtype=np.float64)
        target_mat = np.zeros((len(steps), max(len(target_ids), 1), 9), dtype=np.float64)
        tcp_mat = np.zeros((len(steps), 9), dtype=np.float64)
        joint_names = None
        for index, step in enumerate(steps):
            apply_recorded_qpos(env, step["qpos"])
            tcp = tcp_position_m(env)
            live_tcp.append(tcp)
            recorded_tcp.append(np.asarray(step["tcp_position_m"], dtype=np.float64))
            joints, joint_names = arm_joint_qpos(model, data)
            arm_q.append(joints)
            robot_pos[index] = np.asarray(data.geom_xpos[robot_ids], dtype=np.float64)
            robot_mat[index] = np.asarray(data.geom_xmat[robot_ids], dtype=np.float64)
            if target_ids:
                target_pos[index, : len(target_ids)] = np.asarray(
                    data.geom_xpos[target_ids], dtype=np.float64
                )
                target_mat[index, : len(target_ids)] = np.asarray(
                    data.geom_xmat[target_ids], dtype=np.float64
                )
            tcp_mat[index] = tcp_rotation(env).reshape(9).astype(np.float64)
            for geom_index, geom_id in enumerate(robot_ids):
                shape = GeomShape(
                    int(robot_catalog["gtype"][geom_index]),
                    data.geom_xpos[int(geom_id)],
                    data.geom_xmat[int(geom_id)],
                    robot_catalog["size"][geom_index],
                    robot_verts[geom_index],
                )
                lo, hi = shape.world_aabb()
                robot_lo[index, geom_index] = lo
                robot_hi[index, geom_index] = hi
            for geom_index, geom_id in enumerate(target_ids):
                shape = GeomShape(
                    int(target_catalog["gtype"][geom_index]),
                    data.geom_xpos[int(geom_id)],
                    data.geom_xmat[int(geom_id)],
                    target_catalog["size"][geom_index],
                    target_verts[geom_index],
                )
                lo, hi = shape.world_aabb()
                target_lo[index, geom_index] = lo
                target_hi[index, geom_index] = hi
        recorded_tcp_a = np.asarray(recorded_tcp, dtype=np.float64)
        live_tcp_a = np.asarray(live_tcp, dtype=np.float64)
        residual = np.linalg.norm(live_tcp_a - recorded_tcp_a, axis=1)
        max_residual = float(np.max(residual)) if len(residual) else float("inf")
        grasp_i = int(windows["final_grasp_index"])
        pre_i = int(windows["pregrasp_index"])
        if pre_i > 0:
            pre_i = pre_i - 1 + 1
        pregrasp_last = int(windows["grasp_index"]) - 1
        npz_name = f"{int(result['role_index']):03d}_{str(result['episode_id'])[:16]}.npz"
        row_out = {
            "role_index": result.get("role_index"),
            "episode_id": result.get("episode_id"),
            "family": (
                result.get("layout_family_id")
                or result.get("family_id")
                or payload.get("manifest_row", {}).get("layout_family_id")
                or payload.get("manifest_row", {}).get("family")
                or payload.get("expected", {}).get("family_id")
                or payload.get("expected", {}).get("layout_family_id")
            ),
            "intrusion_side": result.get("intrusion_side"),
            "clean_success": bool(result.get("clean_success")),
            "task_success": bool(result.get("task_success")),
            "n_steps": len(steps),
            "max_tcp_residual_m": max_residual,
            "reconstruction_valid": bool(max_residual <= MAX_TCP_RESIDUAL_M),
            "robot_geom_count": len(robot_ids),
            "target_geom_count": len(target_ids),
            "arm_joint_names": joint_names,
            "canonical_pregrasp_tcp_m": _json_vec(live_tcp_a[pregrasp_last]),
            "canonical_grasp_tcp_m": _json_vec(live_tcp_a[grasp_i]),
            "canonical_grasp_arm_joints_rad": _json_vec(arm_q[grasp_i]),
            "result_sha256": sha256_file(row_dir / "result.json"),
            "trajectory_sha256": sha256_file(row_dir / "trajectory.json"),
            "npz_relative_path": f"rows/{npz_name}",
            "windows": {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in windows.items()
                if key.endswith("_mask") or key.endswith("_index")
            },
        }
        snapshot = {
            "role_index": np.int32(result["role_index"]),
            "tcp_m": live_tcp_a.astype(np.float64),
            "tcp_mat": tcp_mat,
            "robot_geom_ids": np.asarray(robot_ids, dtype=np.int32),
            "target_geom_ids": np.asarray(target_ids, dtype=np.int32),
            "robot_gtype": robot_catalog["gtype"],
            "robot_size": robot_catalog["size"],
            "robot_pos": robot_pos,
            "robot_mat": robot_mat,
            "robot_mesh_start": robot_catalog["mesh_start"],
            "robot_mesh_count": robot_catalog["mesh_count"],
            "robot_mesh_verts": robot_catalog["mesh_verts"],
            "target_gtype": target_catalog["gtype"],
            "target_size": target_catalog["size"],
            "target_pos": target_pos,
            "target_mat": target_mat,
            "target_mesh_start": target_catalog["mesh_start"],
            "target_mesh_count": target_catalog["mesh_count"],
            "target_mesh_verts": target_catalog["mesh_verts"],
            "robot_lo": robot_lo,
            "robot_hi": robot_hi,
            "target_lo": target_lo,
            "target_hi": target_hi,
            "inbound_mask": windows["inbound_mask"],
            "outbound_mask": windows["outbound_mask"],
            "grasp_mask": windows["grasp_mask"],
            "initial_mask": windows["initial_mask"],
        }
        return {
            "summary": row_out,
            "npz_name": npz_name,
            "npz": {
                "tcp_m": live_tcp_a,
                "recorded_tcp_m": recorded_tcp_a,
                "tcp_residual_m": residual,
                "arm_joints_rad": np.asarray(arm_q, dtype=np.float64),
                "robot_lo": robot_lo,
                "robot_hi": robot_hi,
                "target_lo": target_lo,
                "target_hi": target_hi,
                "inbound_mask": windows["inbound_mask"],
                "outbound_mask": windows["outbound_mask"],
                "grasp_mask": windows["grasp_mask"],
                "initial_mask": windows["initial_mask"],
            },
            "snapshot": snapshot,
        }
    finally:
        if task is not None:
            cleanup_task(task, sampler, scratch)


def write_immutable(path: Path, document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("artifact_sha256", None)
    digest = sha256_payload(payload)
    payload["artifact_sha256"] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return digest


def npz_relative_path_for_row(row: dict[str, Any]) -> str:
    if row.get("npz_relative_path"):
        return str(row["npz_relative_path"])
    episode = str(row["episode_id"])[:16]
    return f"rows/{int(row['role_index']):03d}_{episode}.npz"


def attach_npz_hashes(output_root: Path, document: dict[str, Any]) -> dict[str, Any]:
    """Record each reconstruction NPZ path and SHA-256 in the evidence document."""
    files = []
    for row in document["rows"]:
        relative = npz_relative_path_for_row(row)
        path = output_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"reconstruction NPZ missing: {path}")
        digest = sha256_file(path)
        row["npz_relative_path"] = relative
        row["npz_sha256"] = digest
        files.append(
            {
                "path": relative,
                "sha256": digest,
                "role_index": row["role_index"],
                "episode_id": row.get("episode_id"),
            }
        )
    document["npz_files"] = files
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--source-summary", type=Path, default=SOURCE_SUMMARY)
    parser.add_argument(
        "--hash-existing",
        action="store_true",
        help="Hash already-written reconstruction NPZs into reconstruction.json",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if args.hash_existing:
        path = output_root / "reconstruction.json"
        document = json.loads(path.read_text())
        document = attach_npz_hashes(output_root, document)
        digest = write_immutable(path, document)
        print(
            json.dumps(
                {
                    "path": str(path),
                    "npz_files": len(document["npz_files"]),
                    "artifact_sha256": digest,
                },
                indent=2,
            )
        )
        return 0
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    source = json.loads(args.source_summary.read_text())
    rows = list(source["manifest_rows"])
    results = list(source["results"])
    if len(rows) != N_BASELINE_ROWS or len(results) != N_BASELINE_ROWS:
        raise SystemExit("frozen V9.5 baseline is not eight rows")
    by_episode = {item["episode_id"]: item for item in results}
    row_dirs = sorted(
        (args.source_summary.parent / "expert_screen_rows").glob("*")
    )
    jobs = []
    for directory in row_dirs:
        if not (directory / "trajectory.json").is_file():
            continue
        result = json.loads((directory / "result.json").read_text())
        manifest = next(
            row for row in rows if row["episode_id"] == result["episode_id"]
        )
        jobs.append(
            {
                "row_dir": str(directory),
                "manifest_row": manifest,
                "expected": by_episode[result["episode_id"]],
                "selected_seed": result.get("selected_seed")
                or {
                    "seed_u32": manifest["task_seed_u32"],
                    "seed_u64": manifest["task_seed_u64"],
                },
            }
        )
    if len(jobs) != N_BASELINE_ROWS:
        raise SystemExit(f"expected eight row directories, found {len(jobs)}")
    output_root.mkdir(parents=True, exist_ok=True)
    npz_dir = output_root / "rows"
    npz_dir.mkdir(exist_ok=True)
    reconstructed: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(args.workers, 4),
        mp_context=context,
        max_tasks_per_child=1,
    ) as pool:
        futures = [pool.submit(reconstruct_one_row, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            np.savez_compressed(npz_dir / item["npz_name"], **item["npz"])
            reconstructed.append(item["summary"])
    reconstructed.sort(key=lambda item: int(item["role_index"]))
    valid = all(item["reconstruction_valid"] for item in reconstructed)
    clean = [item for item in reconstructed if item["clean_success"]]
    document = {
        "schema_version": "pact_place_v9_9_baseline_reconstruction_v1",
        "contract_version": CONTRACT_VERSION,
        **empty_authorization(),
        "authorizes_paired_screen": False,
        "n_rows": len(reconstructed),
        "n_clean_cells": len(clean),
        "all_tcp_residuals_le_1mm": valid,
        "max_tcp_residual_m": max(item["max_tcp_residual_m"] for item in reconstructed),
        "scene_xml_sha256": sha256_file(SCENE_XML),
        "scene_xml_sha256_matches_contract": sha256_file(SCENE_XML) == PLACE_V5_SCENE_SHA256,
        "baseline_summary_sha256": sha256_file(args.source_summary),
        "implementation_sha256": sha256_payload(
            {path: sha256_file(ROOT / path) for path in IMPLEMENTATION_PATHS}
        ),
        "rows": reconstructed,
        "canonical_clean_cells": [
            {
                "role_index": item["role_index"],
                "family": item["family"],
                "intrusion_side": item["intrusion_side"],
                "pregrasp_tcp_m": item["canonical_pregrasp_tcp_m"],
                "grasp_tcp_m": item["canonical_grasp_tcp_m"],
                "grasp_arm_joints_rad": item["canonical_grasp_arm_joints_rad"],
            }
            for item in clean
        ],
    }
    document = attach_npz_hashes(output_root, document)
    digest = write_immutable(output_root / "reconstruction.json", document)
    print(
        json.dumps(
            {
                "path": str(output_root / "reconstruction.json"),
                "valid": valid,
                "max_tcp_residual_m": document["max_tcp_residual_m"],
                "artifact_sha256": digest,
            },
            indent=2,
        )
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
