#!/usr/bin/env python3
"""Collect committed PACT collision-corridor rows with 8--12 workers.

Each candidate has one immutable terminal result under
``<output>/rows/<episode_id>/result.json``. A terminal row is never re-run.
Task failures retain their trajectory. Deterministic construction failures
before the initial observation may advance to a predeclared retry seed and
retain their retry history. Once a rollout can begin, its row is never retried.
The launcher exits nonzero if any requested row is unreconciled.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MOLMO = ROOT / "submodules" / "molmospaces"
sys.path.insert(0, str(SCRIPTS))

from pact_collision_contract import (
    canonical_json,
    load_manifest,
    retry_seed,
    rows_for_role,
)

DEFAULT_MANIFEST = ROOT / "configs" / "pact_collision_candidate_manifest_v2.json"
TERMINAL_STATUSES = {
    "success",
    "task_failure",
    "sampling_failure",
    "infrastructure_failure",
}
BOUNDARY_FILENAME = "initial_observation_accepted.json"
MAX_TASKS_PER_CHILD = 8


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _terminal_result(path: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result.get("episode_id") != row["episode_id"]:
        raise RuntimeError(f"{path}: episode ID does not match requested row")
    if result.get("row_sha256") != row["row_sha256"]:
        raise RuntimeError(f"{path}: row hash does not match requested row")
    if result.get("status") not in TERMINAL_STATUSES:
        raise RuntimeError(f"{path}: non-terminal or unknown status")
    return result


def _attach_h5_provenance(
    path: Path,
    *,
    row: dict[str, Any],
    manifest_sha256: str,
    result: dict[str, Any],
) -> None:
    import h5py

    with h5py.File(path, "a") as handle:
        handle.attrs["pact_schema_version"] = row["schema_version"]
        handle.attrs["pact_environment_version"] = row["environment_version"]
        handle.attrs["pact_episode_id"] = row["episode_id"]
        handle.attrs["pact_row_sha256"] = row["row_sha256"]
        handle.attrs["pact_manifest_sha256"] = manifest_sha256
        group = handle.require_group("pact_manifest")
        values = {
            "row": row,
            "result_without_paths": {
                key: value
                for key, value in result.items()
                if key not in {"trajectory_path", "videos"}
            },
        }
        for name, value in values.items():
            if name in group:
                del group[name]
            group.create_dataset(name, data=canonical_json(value).encode("utf-8"))


def _publish_episode(
    *,
    row_dir: Path,
    task,
    config,
    save_videos: bool,
    row: dict[str, Any],
    manifest_sha256: str,
    result: dict[str, Any],
) -> tuple[str, list[str]]:
    from molmo_spaces.utils.save_utils import prepare_episode_for_saving, save_trajectories

    staging = Path(tempfile.mkdtemp(prefix=".staging.", dir=row_dir))
    try:
        prepared = prepare_episode_for_saving(
            task.get_history(),
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
            save_mp4s=bool(save_videos),
        )
        staged_h5 = staging / "trajectories.h5"
        if not staged_h5.exists():
            raise RuntimeError("save_trajectories did not write trajectories.h5")
        _attach_h5_provenance(
            staged_h5,
            row=row,
            manifest_sha256=manifest_sha256,
            result=result,
        )
        final_h5 = row_dir / "trajectory.h5"
        if final_h5.exists():
            raise RuntimeError(f"refusing to overwrite abandoned payload {final_h5}")
        os.replace(staged_h5, final_h5)
        videos: list[str] = []
        for artifact in sorted(staging.iterdir()):
            destination = row_dir / artifact.name
            if destination.exists():
                raise RuntimeError(f"refusing to overwrite artifact {destination}")
            os.replace(artifact, destination)
            videos.append(str(destination))
        return str(final_h5), videos
    finally:
        with __import__("contextlib").suppress(OSError):
            staging.rmdir()


def _run_row(
    row: dict[str, Any],
    *,
    manifest_sha256: str,
    sensor_names: list[str],
    output_dir: str,
    save_videos: bool,
) -> dict[str, Any]:
    """Worker entry point. Heavy simulator imports happen only inside workers."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    if str(MOLMO) not in sys.path:
        sys.path.insert(0, str(MOLMO))

    row_dir = Path(output_dir) / "rows" / row["episode_id"]
    row_dir.mkdir(parents=True, exist_ok=True)
    result_path = row_dir / "result.json"
    boundary_path = row_dir / BOUNDARY_FILENAME
    existing = _terminal_result(result_path, row)
    if existing is not None:
        existing["resume_action"] = "skipped_terminal_row"
        return existing
    if (row_dir / "trajectory.h5").exists():
        raise RuntimeError(
            f"{row_dir} has a trajectory but no terminal result; refusing automatic re-run"
        )
    if boundary_path.exists():
        boundary = json.loads(boundary_path.read_text())
        if boundary.get("episode_id") != row["episode_id"]:
            raise RuntimeError(f"{boundary_path}: episode ID does not match requested row")
        if boundary.get("row_sha256") != row["row_sha256"]:
            raise RuntimeError(f"{boundary_path}: row hash does not match requested row")
        result = {
            "schema_version": "pact_collision_collection_result_v2",
            "status": "infrastructure_failure",
            "episode_id": row["episode_id"],
            "row_sha256": row["row_sha256"],
            "manifest_sha256": manifest_sha256,
            "candidate_index": int(row["candidate_index"]),
            "role": row["role"],
            "role_index": int(row["role_index"]),
            "intrusion_side": row["intrusion_side"],
            "rollout_started": True,
            "accepted_observation_marker": str(boundary_path),
            "accepted_observation_marker_sha256": __import__("hashlib").sha256(
                boundary_path.read_bytes()
            ).hexdigest(),
            "error": (
                "accepted initial observation from an earlier worker process has "
                "no terminal result; conservatively terminalized without rerun"
            ),
            "task_success": False,
            "collision_free_task_success": False,
        }
        _write_json_atomic(result_path, result)
        return result

    from molmo_spaces.data_generation.config_registry import get_config_class
    from molmo_spaces.data_generation.main import auto_import_configs
    from molmo_spaces.data_generation.pipeline import (
        ParallelRolloutRunner,
        cleanup_episode_resources,
        setup_policy,
    )
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime
    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask

    assert_supported_runtime(strict=True)
    auto_import_configs()
    config_class = get_config_class("FrankaSkinPACTCollisionCorridorConfig")
    config = config_class(output_dir=Path(output_dir), num_workers=1)
    resolved_output = Path(output_dir).resolve()
    if Path(config.output_dir).resolve() != resolved_output:
        raise RuntimeError(
            "runtime config did not retain the isolated collection output directory"
        )
    if not resolved_output.is_relative_to(ROOT.resolve()):
        raise RuntimeError("collection output must stay inside the isolated worktree")

    configured_names = [spec.name for spec in config.camera_config.cameras][1:]
    if configured_names != sensor_names:
        raise RuntimeError("runtime proximity-sensor order differs from manifest")

    retry_history: list[dict[str, Any]] = []
    task = None
    policy = None
    sampler = None
    rollout_started = False
    try:
        max_retries = int(row["max_sampling_retries"])
        for retry_index in range(max_retries + 1):
            seed = (
                {
                    "seed_u32": int(row["task_seed_u32"]),
                    "seed_u64": int(row["task_seed_u64"]),
                }
                if retry_index == 0
                else retry_seed(row, retry_index)
            )
            sampler = config.task_sampler_config.task_sampler_class(config)
            sampler.seed_task_sampling(seed["seed_u32"])
            sampler.set_pact_manifest_row(row)
            try:
                task = sampler.sample_task(
                    house_index=int(row["scene_template_house_index"])
                )
            except HouseInvalidForTask as exc:
                retry_history.append(
                    {
                        "retry_index": retry_index,
                        "seed": seed,
                        "reason": f"HouseInvalidForTask: {exc.reason}",
                    }
                )
                sampler.close()
                sampler = None
                continue
            except Exception as exc:  # noqa: BLE001 - terminal sampling ledger
                retry_history.append(
                    {
                        "retry_index": retry_index,
                        "seed": seed,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                sampler.close()
                sampler = None
                continue
            if task is None:
                retry_history.append(
                    {"retry_index": retry_index, "seed": seed, "reason": "sample_task_none"}
                )
                sampler.close()
                sampler = None
                continue

            policy = setup_policy(config, task, None, None)
            try:
                initial_reset_result = task.reset()
            except Exception as exc:  # noqa: BLE001 - pre-boundary construction ledger
                retry_history.append(
                    {
                        "retry_index": retry_index,
                        "seed": seed,
                        "reason": (
                            "pre_rollout_construction_failure: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )
                cleanup_episode_resources(
                    task=task,
                    policy=policy,
                    task_sampler=sampler,
                    preloaded_policy=None,
                    close_task_sampler=True,
                )
                task = None
                policy = None
                sampler = None
                continue

            # The row becomes outcome-bearing here. No exception or result after
            # this point is eligible for a retry.
            _write_json_atomic(
                boundary_path,
                {
                    "schema_version": "pact_collection_scientific_boundary_v1",
                    "episode_id": row["episode_id"],
                    "row_sha256": row["row_sha256"],
                    "manifest_sha256": manifest_sha256,
                    "role": row["role"],
                    "role_index": int(row["role_index"]),
                    "retry_index": retry_index,
                    "seed": seed,
                    "boundary": "initial_observation_accepted",
                },
            )
            rollout_started = True
            task_success = bool(
                ParallelRolloutRunner.run_single_rollout(
                    episode_seed=seed["seed_u64"],
                    task=task,
                    policy=policy,
                    end_on_success=config.end_on_success,
                    initial_reset_result=initial_reset_result,
                )
            )
            policy_info = _jsonable(policy.get_info())
            audit = policy_info.get("pact_contact_audit", {})
            collision_free = bool(audit.get("collision_free", False))
            result: dict[str, Any] = {
                "schema_version": "pact_collision_collection_result_v2",
                "status": "success" if task_success else "task_failure",
                "episode_id": row["episode_id"],
                "row_sha256": row["row_sha256"],
                "manifest_sha256": manifest_sha256,
                "candidate_index": int(row["candidate_index"]),
                "role": row["role"],
                "role_index": int(row["role_index"]),
                "intrusion_side": row["intrusion_side"],
                "retry_index": retry_index,
                "retry_history": retry_history,
                "seed": seed,
                "task_success": task_success,
                "collision_free": collision_free,
                "collision_free_task_success": bool(task_success and collision_free),
                "contact_audit": audit,
                "behavior_class": getattr(policy, "behavior_class", None),
                "scene_params": _jsonable(getattr(task, "scene_params", {}) or {}),
                "root_source_commit": _git_head(ROOT),
                "molmospaces_source_commit": _git_head(MOLMO),
                "runtime_assets_dir": str(
                    Path(os.environ["MLSPACES_ASSETS_DIR"]).resolve()
                ),
                "isolated_output_dir": str(resolved_output),
            }
            trajectory_path, videos = _publish_episode(
                row_dir=row_dir,
                task=task,
                config=config,
                save_videos=save_videos,
                row=row,
                manifest_sha256=manifest_sha256,
                result=result,
            )
            result["trajectory_path"] = trajectory_path
            result["videos"] = videos
            _write_json_atomic(result_path, result)
            return result

        result = {
            "schema_version": "pact_collision_collection_result_v2",
            "status": "sampling_failure",
            "episode_id": row["episode_id"],
            "row_sha256": row["row_sha256"],
            "manifest_sha256": manifest_sha256,
            "candidate_index": int(row["candidate_index"]),
            "role": row["role"],
            "role_index": int(row["role_index"]),
            "intrusion_side": row["intrusion_side"],
            "retry_history": retry_history,
            "task_success": False,
            "collision_free_task_success": False,
        }
        _write_json_atomic(result_path, result)
        return result
    except Exception as exc:  # noqa: BLE001 - preserve terminal row provenance
        result = {
            "schema_version": "pact_collision_collection_result_v2",
            "status": "infrastructure_failure",
            "episode_id": row["episode_id"],
            "row_sha256": row["row_sha256"],
            "manifest_sha256": manifest_sha256,
            "candidate_index": int(row["candidate_index"]),
            "role": row["role"],
            "role_index": int(row["role_index"]),
            "intrusion_side": row["intrusion_side"],
            "retry_history": retry_history,
            "rollout_started": rollout_started,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "task_success": False,
            "collision_free_task_success": False,
        }
        _write_json_atomic(result_path, result)
        return result
    finally:
        if task is not None or policy is not None or sampler is not None:
            cleanup_episode_resources(
                task=task,
                policy=policy,
                task_sampler=sampler,
                preloaded_policy=None,
                close_task_sampler=sampler is not None,
            )


def _result_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        status: sum(result.get("status") == status for result in results)
        for status in sorted(TERMINAL_STATUSES)
    }


def _protected_eval_processes() -> list[int]:
    """Find the protected shared-GPU evaluation without matching this process."""
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"eval_act_obstacle_on_policy.py" in command:
            matches.append(int(entry.name))
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=(
        "development",
        "pilot_train",
        "full_train",
        "full_validation",
    ))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--no-save-videos",
        action="store_true",
        help="omit MP4s (development diagnostics only; converted training rows require them)",
    )
    args = parser.parse_args()
    if not 8 <= args.workers <= 12:
        raise SystemExit("--workers must be in [8, 12] for this experiment")
    active = _protected_eval_processes()
    if active:
        raise SystemExit(
            "protected confirmatory evaluation is still active; refusing datagen "
            f"(PIDs {active})"
        )

    manifest = load_manifest(args.manifest)
    rows = rows_for_role(manifest, args.role)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"manifest_sha256 {manifest['manifest_sha256']}")
    print(f"role            {args.role}")
    print(f"rows            {len(rows)}")
    print(f"workers         {args.workers}")
    print(f"output_dir      {args.output_dir}")

    context = multiprocessing.get_context("spawn")
    results: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    for row in rows:
        result_path = args.output_dir / "rows" / row["episode_id"] / "result.json"
        existing = _terminal_result(result_path, row)
        if existing is None:
            pending_rows.append(row)
        else:
            existing["resume_action"] = "skipped_terminal_row"
            results.append(existing)
    print(f"terminal_resume {len(results)}")
    print(f"pending         {len(pending_rows)}")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        max_tasks_per_child=MAX_TASKS_PER_CHILD,
    ) as executor:
        futures = {
            executor.submit(
                _run_row,
                row,
                manifest_sha256=manifest["manifest_sha256"],
                sensor_names=manifest["sensor_names"],
                output_dir=str(args.output_dir),
                save_videos=not args.no_save_videos,
            ): row
            for row in pending_rows
        }
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except BaseException as exc:  # noqa: BLE001 - worker death reconciliation
                result = {
                    "status": "unreconciled_worker_failure",
                    "episode_id": row["episode_id"],
                    "row_sha256": row["row_sha256"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            print(
                f"{row['role_index']:03d} {row['episode_id'][:12]} "
                f"{result['status']}"
            )

    by_id = {result["episode_id"]: result for result in results}
    missing = [row["episode_id"] for row in rows if row["episode_id"] not in by_id]
    bad_identity = [
        row["episode_id"]
        for row in rows
        if by_id.get(row["episode_id"], {}).get("row_sha256") != row["row_sha256"]
    ]
    nonterminal = [
        row["episode_id"]
        for row in rows
        if by_id.get(row["episode_id"], {}).get("status") not in TERMINAL_STATUSES
    ]
    summary = {
        "schema_version": "pact_collision_collection_summary_v2",
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "role": args.role,
        "workers": args.workers,
        "requested_episode_ids": [row["episode_id"] for row in rows],
        "status_counts": _result_counts(results),
        "missing": missing,
        "bad_identity": bad_identity,
        "nonterminal": nonterminal,
        "complete": not (missing or bad_identity or nonterminal),
    }
    _write_json_atomic(args.output_dir / f"{args.role}_summary.json", summary)
    print(json.dumps(summary["status_counts"], sort_keys=True))
    if not summary["complete"]:
        print("PACT COLLECTION INCOMPLETE", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
