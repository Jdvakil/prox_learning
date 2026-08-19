#!/usr/bin/env python3
"""Re-record the 152 kept PACT place-corridor v5 rows as trainable demonstrations.

This is the datagen producer for the place corridor.  It is modelled on
``scripts/run_pact_collision_collection.py`` -- the proven producer for the
working ``pact_collision_corridor_v2`` reference -- and deliberately does NOT
import ``run_row`` from ``run_pact_place_expert_screen``: that harness truncates
the sensor suite to ``qpos``/``tcp_pose``, which is what left the v5 collection
without proximity, RGB or actions.

Seed selection, the initial-contact rejection filter, the expert, the scene and
the clean-success criterion are byte-identical to the screen.  Only the
observation suite differs, so every row is expected to reproduce its recorded
outcome.  Divergence is reported, never substituted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for search_path in (ROOT / "scripts", MOLMO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import (  # noqa: E402
    retry_seed,
    sha256_file,
    sha256_payload,
)
from pact_place_recovery_contract import (  # noqa: E402
    REPRODUCED_KEYS,
    load_recovery_contract,
)

# Reused read-only helpers.  None of these run a rollout; the screen's ``run_row``
# is deliberately not among them.
from run_pact_place_expert_screen import (  # noqa: E402
    _jsonable,
    _protected_eval_processes,
    assert_endpoint_scalars_emitted,
    disallowed_initial_contacts,
    initial_robot_environment_contacts,
    place_receptacle_outside_placement,
    verify_protected_artifacts,
    write_json_atomic,
)

TERMINAL_STATUSES = {"complete", "sampling_failure", "infrastructure_failure"}

# Each worker otherwise sizes libgomp/BLAS/torch pools to the core count (128 here),
# costing ~319 tasks per worker.  Twelve of those exhaust the container's
# cgroup pids.max before any rollout starts, and libgomp reports it only as
# "Thread creation failed: Resource temporarily unavailable".  The rollout is
# one episode on one process, so single-threaded pools cost nothing.
THREAD_POOL_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
# Measured headroom per worker under THREAD_POOL_ENV, with margin.
PID_BUDGET_PER_WORKER = 80


def _pids_headroom() -> int | None:
    """Tasks still available in this container's cgroup, or None if unlimited."""
    try:
        limit = Path("/sys/fs/cgroup/pids.max").read_text().strip()
        current = int(Path("/sys/fs/cgroup/pids.current").read_text().strip())
    except (OSError, ValueError):
        return None
    if limit == "max":
        return None
    return int(limit) - current
BOUNDARY_FILENAME = "initial_observation_accepted.json"
BATCH_SIZE = 24


def _git_head(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# Keys the recovery contract adds on top of the frozen collection row.  The
# sampler must see exactly the row the screen gave it, so they are stripped.
_RECOVERY_ONLY_KEYS = frozenset(
    {
        "screen_result_sha256",
        "screen_selected_seed",
        "screen_retry_index",
        "screen_episode_steps",
        "expected",
    }
)


def sampler_row(row: dict[str, Any]) -> dict[str, Any]:
    """The frozen collection row, without the recovery bookkeeping fields."""
    stripped = {
        key: value
        for key, value in row.items()
        if key not in _RECOVERY_ONLY_KEYS
    }
    payload = {key: value for key, value in stripped.items() if key != "row_sha256"}
    if sha256_payload(payload) != stripped["row_sha256"]:
        raise RuntimeError(f"row {row['role_index']} does not rehash to its frozen form")
    return stripped


def row_dir(output_root: Path, row: dict[str, Any]) -> Path:
    return output_root / "rows" / f"{row['role_index']:03d}_{row['episode_id'][:16]}"


def _validate_existing(
    path: Path, row: dict[str, Any], config_sha256: str
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if (
        result.get("status") not in TERMINAL_STATUSES
        or result.get("episode_id") != row["episode_id"]
        or result.get("row_sha256") != row["row_sha256"]
        or result.get("config_sha256") != config_sha256
    ):
        raise RuntimeError(f"invalid terminal recovery result: {path}")
    payload = dict(result)
    observed = payload.pop("result_sha256")
    if observed != sha256_payload(payload):
        raise RuntimeError(f"result self-hash mismatch: {path}")
    return result


def make_recovery_config(output_dir: Path, *, scene_xml: Path):
    """The screen's ``_make_config`` with the observation reductions removed.

    Two lines differ from ``run_pact_place_expert_screen._make_config``:

    * ``proximity_sensor_period_ms`` keeps the datagen default (16.6667 ms).  The
      screen set it to 0.0, which collapses the proximity buffer to a single
      sub-step; the trainable schema needs the four sub-steps the reference
      collection carries.
    * ``output_dir`` is the recovery root rather than the screen's parent.

    Everything that touches physics -- task, policy, sampler, scene, horizon,
    action noise -- is identical.
    """
    from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
        FrankaSkinPACTCollisionCorridorConfig,
    )
    from molmo_spaces.tasks.enclosure_reach import (
        PactPlaceCorridorPolicyConfig,
        PactPlaceCorridorSampler,
        PactPlaceCorridorTask,
        PactPlaceCorridorV2Sampler,
        PactPlaceCorridorV3Sampler,
    )

    config = FrankaSkinPACTCollisionCorridorConfig(
        output_dir=output_dir,
        num_workers=1,
    )
    config.task_type = "pick_and_place"
    config.task_horizon = 900
    config.end_on_success = False
    config.task_config = PickAndPlaceTaskConfig(task_cls=PactPlaceCorridorTask)
    config.policy_config = PactPlaceCorridorPolicyConfig()
    scene = Path(scene_xml)
    if scene.name == "pact_place_corridor_v3.xml":
        sampler_cls = PactPlaceCorridorV3Sampler
    elif scene.name == "pact_place_corridor_v2.xml":
        sampler_cls = PactPlaceCorridorV2Sampler
    else:
        sampler_cls = PactPlaceCorridorSampler
    config.task_sampler_config.task_sampler_class = sampler_cls
    config.task_sampler_config.scene_xml_paths = [str(scene)] * 2
    if hasattr(config.robot_config, "action_noise_config"):
        config.robot_config.action_noise_config.enabled = False
    return config


def _attach_h5_provenance(
    path: Path,
    *,
    row: dict[str, Any],
    config_sha256: str,
    result: dict[str, Any],
) -> None:
    import h5py

    with h5py.File(path, "a") as handle:
        handle.attrs["pact_schema_version"] = result["schema_version"]
        handle.attrs["pact_episode_id"] = row["episode_id"]
        handle.attrs["pact_row_sha256"] = row["row_sha256"]
        handle.attrs["pact_recovery_config_sha256"] = config_sha256
        handle.attrs["pact_screen_result_sha256"] = row["screen_result_sha256"]
        group = handle.require_group("pact_manifest")
        values = {
            "row": row,
            "result_without_paths": {
                key: value
                for key, value in result.items()
                if key not in {"trajectory_path", "videos", "trajectory_json_path"}
            },
        }
        for name, value in values.items():
            if name in group:
                del group[name]
            group.create_dataset(
                name,
                data=json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
            )


def _publish_episode(
    *,
    destination: Path,
    task,
    config,
    row: dict[str, Any],
    config_sha256: str,
    result: dict[str, Any],
) -> tuple[str, list[str]]:
    from molmo_spaces.utils.save_utils import (
        prepare_episode_for_saving,
        save_trajectories,
    )

    # save_utils.prepare_episode_for_saving calls a bare json.dumps on obs_scene.
    # The place sampler puts a numpy.bool_ in scene_params (cam_visible), which
    # NumPy 2 reports as type "bool" and json refuses; the collision corridor has
    # no such flag, which is why its producer never hit this.  Coerce the block to
    # plain JSON types here rather than patching the shared submodule.
    history = task.get_history()
    if "obs_scene" in history:
        history["obs_scene"] = _jsonable(history["obs_scene"])

    staging = Path(tempfile.mkdtemp(prefix=".staging.", dir=destination))
    try:
        prepared = prepare_episode_for_saving(
            history,
            task.sensor_suite,
            fps=config.fps,
            save_dir=staging,
            episode_idx=0,
            save_file_suffix="",
        )
        if prepared is None:
            raise RuntimeError("episode produced no saveable observations")
        save_trajectories(
            [prepared],
            save_dir=str(staging),
            fps=config.fps,
            save_file_suffix="",
            save_mp4s=True,
        )
        staged_h5 = staging / "trajectories.h5"
        if not staged_h5.exists():
            raise RuntimeError("save_trajectories did not write trajectories.h5")
        _attach_h5_provenance(
            staged_h5, row=row, config_sha256=config_sha256, result=result
        )
        final_h5 = destination / "trajectory.h5"
        if final_h5.exists():
            raise RuntimeError(f"refusing to overwrite abandoned payload {final_h5}")
        os.replace(staged_h5, final_h5)
        videos: list[str] = []
        for artifact in sorted(staging.iterdir()):
            target = destination / artifact.name
            if target.exists():
                raise RuntimeError(f"refusing to overwrite artifact {target}")
            os.replace(artifact, target)
            videos.append(target.name)
        return str(final_h5), videos
    finally:
        # A failure between prepare and publish leaves staged MP4s behind; drop the
        # whole staging tree so a retried row never inherits a partial episode.
        shutil.rmtree(staging, ignore_errors=True)


def _reproduction_report(
    row: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    expected = row["expected"]
    mismatched = {
        key: {"expected": expected[key], "observed": result.get(key)}
        for key in REPRODUCED_KEYS
        if result.get(key) != expected[key]
    }
    return {
        "reproduced": not mismatched,
        "mismatched": mismatched,
        "screen_selected_seed": row["screen_selected_seed"],
        "screen_retry_index": row["screen_retry_index"],
        "screen_episode_steps": row["screen_episode_steps"],
        "seed_reproduced": (
            result.get("selected_seed") == row["screen_selected_seed"]
        ),
        "retry_index_reproduced": (
            result.get("retry_index") == row["screen_retry_index"]
        ),
        "episode_steps_reproduced": (
            result.get("episode_steps") == row["screen_episode_steps"]
        ),
    }


def run_recovery_row(
    row: dict[str, Any],
    *,
    config_sha256: str,
    output_root: str,
    scene_xml: str,
    sensor_names: list[str],
) -> dict[str, Any]:
    """Worker entry point.  Heavy simulator imports happen only inside workers."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("PACT_CONTACT_AUDIT_SUMMARY_ONLY", "1")
    for name, value in THREAD_POOL_ENV.items():
        os.environ.setdefault(name, value)
    os.environ.pop("DISPLAY", None)

    destination = row_dir(Path(output_root), row)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "result.json"
    boundary_path = destination / BOUNDARY_FILENAME
    existing = _validate_existing(result_path, row, config_sha256)
    if existing is not None:
        existing["resume_action"] = "skipped_terminal_row"
        return existing
    if (destination / "trajectory.h5").exists():
        raise RuntimeError(
            f"{destination} has a trajectory but no terminal result; "
            "refusing automatic re-run"
        )

    from molmo_spaces.data_generation.pipeline import (
        ParallelRolloutRunner,
        cleanup_episode_resources,
        setup_policy,
    )
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime
    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask

    assert_supported_runtime(strict=True)
    retry_history: list[dict[str, Any]] = []
    task = policy = sampler = None
    rollout_started = False
    try:
        config = make_recovery_config(Path(output_root), scene_xml=Path(scene_xml))

        # Assert the produced observation geometry before spending a rollout.
        configured = [spec.name for spec in config.camera_config.cameras]
        if configured[0] != "wrist_camera":
            raise RuntimeError("runtime camera 0 is not the wrist camera")
        if configured[1:] != sensor_names:
            raise RuntimeError("runtime proximity-sensor order differs from contract")
        if not config.proximity_sensor_period_ms > 0:
            raise RuntimeError(
                "proximity sub-step recording is disabled; the recovery requires it"
            )

        frozen_row = sampler_row(row)
        selected_seed: dict[str, int] | None = None
        selected_retry_index: int | None = None
        initial_reset_result = None
        for retry_index in range(int(row["max_sampling_retries"]) + 1):
            if retry_index == 0:
                seed_u32 = int(row["task_seed_u32"])
                seed_u64 = int(row["task_seed_u64"])
            else:
                seed_u32, seed_u64 = retry_seed(frozen_row, retry_index)
            seed = {"seed_u32": seed_u32, "seed_u64": seed_u64}
            sampler = config.task_sampler_config.task_sampler_class(config)
            sampler.seed_task_sampling(seed_u32)
            sampler.set_pact_manifest_row(frozen_row)
            try:
                task = sampler.sample_task(
                    house_index=int(row["scene_template_house_index"])
                )
                if task is None:
                    raise HouseInvalidForTask("sample_task returned None")
                # The screen replaced task._sensor_suite here with a qpos/tcp_pose
                # pair.  That truncation is the defect this run recovers from, so
                # the suite is left exactly as sampling built it.
                policy = setup_policy(config, task, None, None)
                initial_reset_result = task.reset()
                rejected = disallowed_initial_contacts(
                    initial_robot_environment_contacts(task.env)
                )
                if rejected:
                    first = rejected[0]
                    raise HouseInvalidForTask(
                        "initial_robot_environment_contact "
                        f"n={len(rejected)} "
                        f"{first.get('body1')} vs {first.get('body2')}"
                    )
            except Exception as error:  # noqa: BLE001 - allowed only before boundary
                retry_history.append(
                    {
                        "retry_index": retry_index,
                        "seed": seed,
                        "reason": f"pre_boundary:{type(error).__name__}:{error}",
                    }
                )
                cleanup_episode_resources(
                    task=task,
                    policy=policy,
                    task_sampler=sampler,
                    preloaded_policy=None,
                    close_task_sampler=True,
                )
                task = policy = sampler = None
                continue
            selected_seed = seed
            selected_retry_index = retry_index
            break

        if selected_seed is None or initial_reset_result is None:
            result = {
                "schema_version": "pact_place_recovery_result_v1",
                "status": "sampling_failure",
                "episode_id": row["episode_id"],
                "row_sha256": row["row_sha256"],
                "config_sha256": config_sha256,
                "role_index": row["role_index"],
                "intrusion_side": row["intrusion_side"],
                "task_success": False,
                "clean_success": False,
                "retry_history": retry_history,
            }
            result["reproduction"] = _reproduction_report(row, result)
        else:
            write_json_atomic(
                boundary_path,
                {
                    "schema_version": "pact_place_recovery_boundary_v1",
                    "initial_observation_accepted": True,
                    "episode_id": row["episode_id"],
                    "row_sha256": row["row_sha256"],
                    "config_sha256": config_sha256,
                    "retry_index": selected_retry_index,
                    "seed": selected_seed,
                    "initial_robot_environment_contacts": _jsonable(
                        initial_robot_environment_contacts(task.env)
                    ),
                },
            )
            rollout_started = True
            task_success = bool(
                ParallelRolloutRunner.run_single_rollout(
                    episode_seed=int(selected_seed["seed_u64"]),
                    task=task,
                    policy=policy,
                    end_on_success=False,
                    initial_reset_result=initial_reset_result,
                )
            )
            policy_info = _jsonable(policy.get_info())
            audit = policy_info["pact_contact_audit"]
            totals = audit["contact_class_totals"]
            clean_success = bool(
                task_success
                and int(totals["hazard_bar"]) == 0
                and int(totals["other_environment"]) == 0
                and int(totals.get("clutter", 0)) == 0
                and place_receptacle_outside_placement(audit) == 0
            )
            endpoint_scalars = policy_info.get("endpoint_scalars") or {}
            trajectory = policy_info.pop("trajectory", [])
            trajectory_json = destination / "trajectory.json"
            write_json_atomic(
                trajectory_json,
                {
                    "schema_version": "pact_place_recovery_trajectory_v1",
                    "episode_id": row["episode_id"],
                    "row_sha256": row["row_sha256"],
                    "config_sha256": config_sha256,
                    "n": len(trajectory),
                    "steps": trajectory,
                },
            )
            result = {
                "schema_version": "pact_place_recovery_result_v1",
                "status": "complete",
                "episode_id": row["episode_id"],
                "row_sha256": row["row_sha256"],
                "config_sha256": config_sha256,
                "role_index": row["role_index"],
                "intrusion_side": row["intrusion_side"],
                "panel_x_jitter_m": row["panel_x_jitter_m"],
                "panel_face_jitter_m": row["panel_face_jitter_m"],
                "selected_seed": selected_seed,
                "retry_index": selected_retry_index,
                "retry_history": retry_history,
                "task_success": task_success,
                "clean_success": clean_success,
                "grasp_phase_success": bool(policy_info["grasp_phase_success"]),
                "place_phase_success": bool(policy_info["place_phase_success"]),
                "terminal_tracking": policy_info["terminal_tracking"],
                "grasp_diagnostics": policy_info["grasp_diagnostics"],
                "contact_audit": audit,
                "place_metrics": policy_info["place_metrics"],
                "scene_params": _jsonable(getattr(task, "scene_params", {}) or {}),
                "episode_steps": int(task.episode_step_count),
                "terminal_policy_phase": str(policy.get_phase()),
                "terminal_action_index": int(policy.action_idx),
                "endpoint_scalars": endpoint_scalars,
                "trajectory_n": len(trajectory),
                "trajectory_json_path": str(trajectory_json),
                "trajectory_json_sha256": sha256_file(trajectory_json),
                "root_source_commit": _git_head(ROOT),
                "molmospaces_source_commit": _git_head(MOLMO),
                "runtime_assets_dir": str(
                    Path(os.environ["MLSPACES_ASSETS_DIR"]).resolve()
                ),
                "proximity_sensor_period_ms": float(
                    config.proximity_sensor_period_ms
                ),
                "n_cameras_configured": len(configured),
            }
            assert_endpoint_scalars_emitted(result)
            result["reproduction"] = _reproduction_report(row, result)
            trajectory_path, videos = _publish_episode(
                destination=destination,
                task=task,
                config=config,
                row=row,
                config_sha256=config_sha256,
                result=result,
            )
            result["trajectory_path"] = trajectory_path
            result["videos"] = videos
    except Exception as error:  # noqa: BLE001 - terminal recovery ledger
        result = {
            "schema_version": "pact_place_recovery_result_v1",
            "status": "infrastructure_failure",
            "episode_id": row["episode_id"],
            "row_sha256": row["row_sha256"],
            "config_sha256": config_sha256,
            "role_index": row["role_index"],
            "intrusion_side": row["intrusion_side"],
            "task_success": False,
            "clean_success": False,
            "retry_history": retry_history,
            "rollout_started": rollout_started,
            "after_scientific_boundary": boundary_path.exists(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        result["reproduction"] = _reproduction_report(row, result)
    finally:
        if task is not None or policy is not None or sampler is not None:
            cleanup_episode_resources(
                task=task,
                policy=policy,
                task_sampler=sampler,
                preloaded_policy=None,
                close_task_sampler=sampler is not None,
            )
    result["result_sha256"] = sha256_payload(result)
    write_json_atomic(result_path, result)
    return result


def summarize(
    contract: dict[str, Any],
    results: list[dict[str, Any]],
    output_root: Path,
    requested: list[dict[str, Any]],
) -> dict[str, Any]:
    complete = [item for item in results if item.get("status") == "complete"]
    diverged = [
        {
            "role_index": item["role_index"],
            "episode_id": item["episode_id"],
            "status": item["status"],
            "mismatched": item["reproduction"]["mismatched"],
        }
        for item in results
        if not item["reproduction"]["reproduced"]
    ]
    summary: dict[str, Any] = {
        "schema_version": "pact_place_v5_recovery_summary_v1",
        "config_sha256": contract["config_sha256"],
        "source_collection_sha256": contract["source_collection_sha256"],
        "output_root": str(output_root),
        "n_requested": len(requested),
        "n_frozen_rows": len(contract["recovery_rows"]),
        "n_results": len(results),
        "n_complete": len(complete),
        "n_clean_success": sum(item.get("clean_success") is True for item in complete),
        "n_reproduced": sum(item["reproduction"]["reproduced"] for item in results),
        "n_seed_reproduced": sum(
            item["reproduction"]["seed_reproduced"] for item in results
        ),
        "n_episode_steps_reproduced": sum(
            item["reproduction"]["episode_steps_reproduced"] for item in results
        ),
        "divergences": diverged,
        "all_reproduced": not diverged,
        "training_authorized": False,
        "conversion_authorized": False,
        "next_action": (
            "run_verify_pact_place_recovery_keys"
            if not diverged and len(complete) == len(requested)
            else "stop_and_report_divergence"
        ),
    }
    summary["recovery_sha256"] = sha256_payload(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="run only the first N rows")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise SystemExit("workers must be in [1, 12]")
    protected = _protected_eval_processes()
    if protected:
        raise SystemExit(f"protected confirmatory evaluation is active: {protected}")

    # Set before the pool is built so spawned workers inherit capped pools from
    # their first import rather than after libgomp has already sized itself.
    for name, value in THREAD_POOL_ENV.items():
        os.environ.setdefault(name, value)
    headroom = _pids_headroom()
    if headroom is not None:
        affordable = headroom // PID_BUDGET_PER_WORKER
        if affordable < args.workers:
            raise SystemExit(
                f"cgroup pids headroom is {headroom}; it affords {affordable} "
                f"workers at {PID_BUDGET_PER_WORKER} tasks each, not {args.workers}"
            )
        print(f"pids_headroom {headroom} (affords {affordable} workers)")

    contract = load_recovery_contract(args.config)
    verify_protected_artifacts(contract)
    output_root = args.output_root or (ROOT / contract["recovery"]["output_root"])
    resolved = output_root.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise SystemExit("recovery output must stay inside the repository")
    if "pact_place_corridor_v5_collection" in str(resolved):
        raise SystemExit("refusing to write into the frozen v5 screen record")
    output_root.mkdir(parents=True, exist_ok=True)

    rows = contract["recovery_rows"]
    if args.limit is not None:
        if not 1 <= args.limit <= len(rows):
            raise SystemExit("--limit is outside the frozen recovery set")
        rows = rows[: args.limit]
    scene_xml = str(ROOT / contract["scene"]["xml"])
    sensor_names = list(contract["recovery"]["proximity_sensor_names"])

    print(f"config_sha256 {contract['config_sha256']}")
    print(f"rows          {len(rows)}")
    print(f"workers       {args.workers}")
    print(f"output_root   {output_root}")
    print(f"scene_xml     {scene_xml}", flush=True)

    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in rows:
        existing = _validate_existing(
            row_dir(output_root, row) / "result.json", row, contract["config_sha256"]
        )
        if existing is None:
            pending.append(row)
        else:
            existing["resume_action"] = "skipped_terminal_row"
            results.append(existing)
    print(f"terminal_resume {len(results)}")
    print(f"pending         {len(pending)}", flush=True)

    context = multiprocessing.get_context("spawn")
    while pending:
        batch, pending = pending[:BATCH_SIZE], pending[BATCH_SIZE:]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            max_tasks_per_child=1,
        ) as executor:
            futures = {
                executor.submit(
                    run_recovery_row,
                    row,
                    config_sha256=contract["config_sha256"],
                    output_root=str(output_root),
                    scene_xml=scene_xml,
                    sensor_names=sensor_names,
                ): row
                for row in batch
            }
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                except BaseException as error:  # noqa: BLE001 - worker death ledger
                    result = {
                        "status": "unreconciled_worker_failure",
                        "role_index": row["role_index"],
                        "episode_id": row["episode_id"],
                        "row_sha256": row["row_sha256"],
                        "task_success": False,
                        "clean_success": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    result["reproduction"] = _reproduction_report(row, result)
                results.append(result)
                report = result["reproduction"]
                print(
                    f"row={row['role_index']:03d} side={row['intrusion_side']} "
                    f"status={result['status']} clean={result.get('clean_success')} "
                    f"reproduced={report['reproduced']} "
                    f"steps={result.get('episode_steps')}"
                    f"/{report['screen_episode_steps']}",
                    flush=True,
                )
        done = len(results)
        print(
            f"progress complete={done}/{len(rows)} "
            f"reproduced={sum(r['reproduction']['reproduced'] for r in results)}",
            flush=True,
        )

    results.sort(key=lambda item: item["role_index"])
    verify_protected_artifacts(contract)
    summary = summarize(contract, results, output_root, rows)
    write_json_atomic(output_root / "recovery.json", summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "n_requested",
                    "n_results",
                    "n_complete",
                    "n_clean_success",
                    "n_reproduced",
                    "all_reproduced",
                    "next_action",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    if summary["divergences"]:
        print("PACT PLACE RECOVERY DIVERGED", file=sys.stderr)
        return 2
    return 0 if summary["n_complete"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
