#!/usr/bin/env python3
"""Run the preregistered privileged-expert solvability screen."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for path in (ROOT / "scripts", MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_geometry_generalization_contract import (  # noqa: E402
    CONDITIONS,
    canonical_json,
    load_manifest,
    retry_seed,
    sha256_payload,
)


TERMINAL_STATUSES = {"complete", "sampling_failure", "infrastructure_failure"}


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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


def sampler_class(name: str) -> type:
    from molmo_spaces.tasks.enclosure_reach import (
        PactCollisionCorridorControlSampler,
        PactCollisionCorridorDeeperHigherSampler,
        PactCollisionCorridorShallowerWiderSampler,
        PactCollisionCorridorTighterSampler,
    )

    allowed = {
        cls.__name__: cls
        for cls in (
            PactCollisionCorridorControlSampler,
            PactCollisionCorridorDeeperHigherSampler,
            PactCollisionCorridorTighterSampler,
            PactCollisionCorridorShallowerWiderSampler,
        )
    }
    if name not in allowed:
        raise ValueError(f"unregistered sampler class {name!r}")
    return allowed[name]


def _result_path(output_root: Path, row: dict[str, Any]) -> Path:
    return (
        output_root
        / "expert_screen_rows"
        / row["condition_id"]
        / f"{row['role_index']:02d}_{row['episode_id'][:16]}"
        / "result.json"
    )


def _validate_existing(path: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if (
        result.get("status") not in TERMINAL_STATUSES
        or result.get("episode_id") != row["episode_id"]
        or result.get("row_sha256") != row["row_sha256"]
    ):
        raise RuntimeError(f"invalid terminal expert result: {path}")
    return result


def run_row(row: dict[str, Any], *, manifest_sha256: str, output_root: str) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    destination = _result_path(Path(output_root), row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = _validate_existing(destination, row)
    if existing is not None:
        return existing

    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
        FrankaSkinPACTCollisionCorridorConfig,
    )
    from molmo_spaces.data_generation.pipeline import (
        ParallelRolloutRunner,
        cleanup_episode_resources,
        setup_policy,
    )
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime
    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask

    assert_supported_runtime(strict=True)
    config = FrankaSkinPACTCollisionCorridorConfig(
        output_dir=destination.parent,
        num_workers=1,
    )
    config.task_sampler_config.task_sampler_class = sampler_class(
        row["task_sampler_class"]
    )
    if hasattr(config.robot_config, "action_noise_config"):
        config.robot_config.action_noise_config.enabled = False
    task = policy = sampler = None
    retry_history: list[dict[str, Any]] = []
    boundary_path = destination.parent / "initial_observation_accepted.json"
    try:
        selected_seed = None
        initial_reset_result = None
        for retry_index in range(int(row["max_sampling_retries"]) + 1):
            seed = (
                {"seed_u32": row["task_seed_u32"], "seed_u64": row["task_seed_u64"]}
                if retry_index == 0
                else retry_seed(row, retry_index)
            )
            sampler = config.task_sampler_config.task_sampler_class(config)
            sampler.seed_task_sampling(int(seed["seed_u32"]))
            sampler.set_pact_manifest_row(row)
            try:
                task = sampler.sample_task(
                    house_index=int(row["scene_template_house_index"])
                )
            except HouseInvalidForTask as error:
                retry_history.append(
                    {"retry_index": retry_index, "seed": seed, "reason": str(error)}
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
            except Exception as error:  # noqa: BLE001 - preregistered pre-boundary retry
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
            break
        if task is None or policy is None or selected_seed is None or initial_reset_result is None:
            result = {
                "schema_version": "pact_geometry_expert_screen_row_v1",
                "status": "sampling_failure",
                "episode_id": row["episode_id"],
                "row_sha256": row["row_sha256"],
                "manifest_sha256": manifest_sha256,
                "condition_id": row["condition_id"],
                "task_success": False,
                "clean_success": False,
                "retry_history": retry_history,
            }
            write_json_atomic(destination, result)
            return result

        write_json_atomic(
            boundary_path,
            {
                "schema_version": "pact_geometry_expert_boundary_v1",
                "initial_observation_accepted": True,
                "episode_id": row["episode_id"],
                "row_sha256": row["row_sha256"],
                "manifest_sha256": manifest_sha256,
                "seed": selected_seed,
            },
        )
        task_success = bool(
            ParallelRolloutRunner.run_single_rollout(
                episode_seed=int(selected_seed["seed_u64"]),
                task=task,
                policy=policy,
                end_on_success=config.end_on_success,
                initial_reset_result=initial_reset_result,
            )
        )
        audit = _jsonable(policy.get_info().get("pact_contact_audit", {}))
        totals = audit.get("contact_class_totals", {})
        clean_success = bool(
            task_success
            and int(totals.get("hazard_bar", 0)) == 0
            and int(totals.get("other_environment", 0)) == 0
        )
        result = {
            "schema_version": "pact_geometry_expert_screen_row_v1",
            "status": "complete",
            "episode_id": row["episode_id"],
            "row_sha256": row["row_sha256"],
            "manifest_sha256": manifest_sha256,
            "condition_id": row["condition_id"],
            "condition_label": row["condition_label"],
            "task_sampler_class": row["task_sampler_class"],
            "instance_index": row["instance_index"],
            "instance_cluster_id": row["instance_cluster_id"],
            "intrusion_side": row["intrusion_side"],
            "realized_geometry": row["realized_geometry"],
            "seed": selected_seed,
            "retry_history": retry_history,
            "task_success": task_success,
            "clean_success": clean_success,
            "contact_audit": audit,
            "scene_params": _jsonable(getattr(task, "scene_params", {}) or {}),
        }
        result["result_sha256"] = sha256_payload(result)
        write_json_atomic(destination, result)
        return result
    except Exception as error:  # noqa: BLE001 - terminal expert-screen ledger
        result = {
            "schema_version": "pact_geometry_expert_screen_row_v1",
            "status": "infrastructure_failure",
            "episode_id": row["episode_id"],
            "row_sha256": row["row_sha256"],
            "manifest_sha256": manifest_sha256,
            "condition_id": row["condition_id"],
            "task_success": False,
            "clean_success": False,
            "retry_history": retry_history,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "after_scientific_boundary": boundary_path.exists(),
        }
        write_json_atomic(destination, result)
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


def active_protected_processes() -> list[int]:
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


def build_summary(manifest: dict[str, Any], results: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    conditions = {}
    for condition_id in CONDITIONS:
        cell = [result for result in results if result.get("condition_id") == condition_id]
        clean = sum(result.get("clean_success") is True for result in cell)
        task = sum(result.get("task_success") is True for result in cell)
        terminal = sum(result.get("status") in TERMINAL_STATUSES for result in cell)
        infrastructure = sum(result.get("status") == "infrastructure_failure" for result in cell)
        passed = len(cell) == 12 and terminal == 12 and infrastructure == 0 and clean >= 10
        conditions[condition_id] = {
            "n": len(cell),
            "terminal": terminal,
            "task_successes": task,
            "clean_successes": clean,
            "clean_success_rate": clean / len(cell) if cell else None,
            "infrastructure_failures": infrastructure,
            "gate_threshold": 10,
            "passed": passed,
            "action": "freeze" if passed else "drop_without_retuning",
        }
    surviving = [condition for condition, item in conditions.items() if item["passed"]]
    shifted = [condition for condition in surviving if condition != "C0"]
    reconciled = len(results) == 48 and all(
        result.get("status") in TERMINAL_STATUSES for result in results
    )
    continue_to_policy = reconciled and "C0" in surviving and len(shifted) >= 2
    summary: dict[str, Any] = {
        "schema_version": "pact_geometry_expert_screen_v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "workers": workers,
        "gate_frozen_before_execution": True,
        "clean_success_definition": "task_success and zero hazard_bar contact entries and zero other_environment contact entries",
        "conditions": conditions,
        "surviving_condition_ids": surviving,
        "surviving_shifted_condition_ids": shifted,
        "reconciled": reconciled,
        "continue_to_policy_evaluation": continue_to_policy,
        "stop_reason": (
            None
            if continue_to_policy
            else "C0_failed_fewer_than_two_shifted_conditions_passed_or_screen_unreconciled"
        ),
        "row_result_paths": sorted(
            str(_result_path(Path("diagnostics_output/pact_geometry_generalization"), row))
            for row in manifest["expert_screen_rows"]
        ),
    }
    summary["expert_screen_sha256"] = sha256_payload(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise SystemExit("workers must be in [1, 12]")
    protected = active_protected_processes()
    if protected:
        raise SystemExit(f"protected confirmatory evaluation is active: {protected}")
    manifest = load_manifest(args.manifest)
    rows = manifest["expert_screen_rows"]
    existing = []
    pending = []
    for row in rows:
        result = _validate_existing(_result_path(args.output_root, row), row)
        (existing if result is not None else pending).append(result if result is not None else row)
    results = list(existing)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        max_tasks_per_child=4,
    ) as executor:
        futures = {
            executor.submit(
                run_row,
                row,
                manifest_sha256=manifest["manifest_sha256"],
                output_root=str(args.output_root),
            ): row
            for row in pending
        }
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"{row['condition_id']} {row['role_index']:02d} "
                f"{result['status']} clean={result.get('clean_success')}"
            )
    summary = build_summary(manifest, results, args.workers)
    write_json_atomic(args.output_root / "expert_screen.json", summary)
    print(canonical_json(summary["conditions"]))
    return 0 if summary["reconciled"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
