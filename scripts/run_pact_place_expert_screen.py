#!/usr/bin/env python3
"""Run and reconcile the preregistered PACT place-corridor expert gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
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
    FAIL_TOKEN,
    MIN_CLEAN_SUCCESSES,
    N_EXPERT_ROWS,
    PASS_TOKEN,
    load_contract,
    retry_seed,
    sha256_file,
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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _result_path(output_root: Path, row: dict[str, Any]) -> Path:
    return (
        output_root
        / "expert_screen_rows"
        / f"{row['role_index']:02d}_{row['episode_id'][:16]}"
        / "result.json"
    )


def _validate_existing(
    path: Path,
    row: dict[str, Any],
    config_sha256: str | None = None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if (
        result.get("status") not in TERMINAL_STATUSES
        or result.get("episode_id") != row["episode_id"]
        or result.get("row_sha256") != row["row_sha256"]
        or (
            config_sha256 is not None
            and result.get("config_sha256") != config_sha256
        )
    ):
        raise RuntimeError(f"invalid terminal expert result: {path}")
    if "result_sha256" in result:
        payload = dict(result)
        observed = payload.pop("result_sha256")
        if observed != sha256_payload(payload):
            raise RuntimeError(f"result self-hash mismatch: {path}")
    return result


def _make_config(destination: Path):
    from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
        FrankaSkinPACTCollisionCorridorConfig,
    )
    from molmo_spaces.tasks.enclosure_reach import (
        PactPlaceCorridorPolicyConfig,
        PactPlaceCorridorSampler,
        PactPlaceCorridorTask,
    )

    config = FrankaSkinPACTCollisionCorridorConfig(
        output_dir=destination.parent,
        num_workers=1,
    )
    config.task_type = "pick_and_place"
    config.task_horizon = 900
    config.end_on_success = False
    config.task_config = PickAndPlaceTaskConfig(task_cls=PactPlaceCorridorTask)
    config.proximity_sensor_period_ms = 0.0
    config.policy_config = PactPlaceCorridorPolicyConfig()
    config.task_sampler_config.task_sampler_class = PactPlaceCorridorSampler
    scene = (
        MOLMO
        / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v1.xml"
    )
    config.task_sampler_config.scene_xml_paths = [str(scene)] * 2
    if hasattr(config.robot_config, "action_noise_config"):
        config.robot_config.action_noise_config.enabled = False
    return config


def run_row(
    row: dict[str, Any],
    *,
    config_sha256: str,
    output_root: str,
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("PACT_CONTACT_AUDIT_SUMMARY_ONLY", "1")
    os.environ.pop("DISPLAY", None)
    destination = _result_path(Path(output_root), row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = _validate_existing(destination, row, config_sha256)
    if existing is not None:
        return existing

    from molmo_spaces.data_generation.pipeline import (
        ParallelRolloutRunner,
        cleanup_episode_resources,
        setup_policy,
    )
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime
    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask

    assert_supported_runtime(strict=True)
    retry_history: list[dict[str, Any]] = []
    boundary_path = destination.parent / "initial_observation_accepted.json"
    task = policy = sampler = None
    try:
        config = _make_config(destination)
        selected_seed: dict[str, int] | None = None
        initial_reset_result = None
        for retry_index in range(int(row["max_sampling_retries"]) + 1):
            if retry_index == 0:
                seed_u32 = int(row["task_seed_u32"])
                seed_u64 = int(row["task_seed_u64"])
            else:
                seed_u32, seed_u64 = retry_seed(row, retry_index)
            seed = {"seed_u32": seed_u32, "seed_u64": seed_u64}
            sampler = config.task_sampler_config.task_sampler_class(config)
            sampler.seed_task_sampling(seed_u32)
            sampler.set_pact_manifest_row(row)
            try:
                task = sampler.sample_task(
                    house_index=int(row["scene_template_house_index"])
                )
                if task is None:
                    raise HouseInvalidForTask("sample_task returned None")
                # The expert is privileged and ignores its observation argument. Sampling
                # still builds the normal camera/sensor geometry, then Phase 0 replaces the
                # polling suite with only the planner-required qpos/tcp sensors so no
                # unused RGB/proximity arrays are rendered or retained during Phase 0.
                from molmo_spaces.env.abstract_sensors import SensorSuite

                task._sensor_suite = SensorSuite(
                    [
                        task._sensor_suite.sensors[uuid]
                        for uuid in ("qpos", "tcp_pose")
                    ]
                )
                policy = setup_policy(config, task, None, None)
                initial_reset_result = task.reset()
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
            break

        if selected_seed is None or initial_reset_result is None:
            result = {
                "schema_version": "pact_place_expert_screen_row_v1",
                "status": "sampling_failure",
                "episode_id": row["episode_id"],
                "row_sha256": row["row_sha256"],
                "config_sha256": config_sha256,
                "role_index": row["role_index"],
                "intrusion_side": row["intrusion_side"],
                "task_success": False,
                "clean_success": False,
                "grasp_phase_success": False,
                "place_phase_success": False,
                "retry_history": retry_history,
            }
        else:
            write_json_atomic(
                boundary_path,
                {
                    "schema_version": "pact_place_scientific_boundary_v1",
                    "initial_observation_accepted": True,
                    "episode_id": row["episode_id"],
                    "row_sha256": row["row_sha256"],
                    "config_sha256": config_sha256,
                    "seed": selected_seed,
                },
            )
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
            )
            result = {
                "schema_version": "pact_place_expert_screen_row_v1",
                "status": "complete",
                "episode_id": row["episode_id"],
                "row_sha256": row["row_sha256"],
                "config_sha256": config_sha256,
                "role_index": row["role_index"],
                "intrusion_side": row["intrusion_side"],
                "panel_x_jitter_m": row["panel_x_jitter_m"],
                "panel_face_jitter_m": row["panel_face_jitter_m"],
                "selected_seed": selected_seed,
                "retry_history": retry_history,
                "task_success": task_success,
                "clean_success": clean_success,
                "grasp_phase_success": bool(policy_info["grasp_phase_success"]),
                "cup_lifted_one_cm": bool(policy_info["cup_lifted_one_cm"]),
                "pickup_start_z_m": policy_info["pickup_start_z_m"],
                "pickup_max_z_m": policy_info["pickup_max_z_m"],
                "pickup_final_position_m": policy_info["pickup_final_position_m"],
                "gripper_width_min_m": policy_info["gripper_width_min_m"],
                "gripper_width_max_m": policy_info["gripper_width_max_m"],
                "grasp_diagnostics": policy_info["grasp_diagnostics"],
                "terminal_tracking": policy_info["terminal_tracking"],
                "terminal_robot_environment_contacts": policy_info[
                    "terminal_robot_environment_contacts"
                ],
                "place_phase_success": bool(policy_info["place_phase_success"]),
                "inbound_deflected": bool(policy_info["inbound_deflected"]),
                "outbound_deflected": bool(policy_info["outbound_deflected"]),
                "accepted_bow_m": policy_info.get("accepted_bow_m"),
                "planned_bow_m": policy_info.get("planned_bow_m"),
                "bow_fallback_taken": bool(policy_info.get("bow_fallback_taken")),
                "bow_diagnostics": policy_info.get("bow_diagnostics"),
                "contact_audit": audit,
                "place_metrics": policy_info["place_metrics"],
                "scene_params": _jsonable(getattr(task, "scene_params", {}) or {}),
                "episode_steps": int(task.episode_step_count),
                "terminal_policy_phase": str(policy.get_phase()),
                "terminal_action_index": int(policy.action_idx),
            }
    except Exception as error:  # noqa: BLE001 - terminal Phase-0 ledger
        result = {
            "schema_version": "pact_place_expert_screen_row_v1",
            "status": "infrastructure_failure",
            "episode_id": row["episode_id"],
            "row_sha256": row["row_sha256"],
            "config_sha256": config_sha256,
            "role_index": row["role_index"],
            "intrusion_side": row["intrusion_side"],
            "task_success": False,
            "clean_success": False,
            "grasp_phase_success": False,
            "place_phase_success": False,
            "retry_history": retry_history,
            "after_scientific_boundary": boundary_path.exists(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
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
    write_json_atomic(destination, result)
    return result


def summarize(
    contract: dict[str, Any],
    results: list[dict[str, Any]],
    workers: int,
    output_root: Path,
    role: str = "gate",
) -> dict[str, Any]:
    complete = [item for item in results if item["status"] == "complete"]
    clean = sum(item.get("clean_success") is True for item in results)
    task = sum(item.get("task_success") is True for item in results)
    grasp = sum(item.get("grasp_phase_success") is True for item in results)
    place = sum(item.get("place_phase_success") is True for item in results)
    inbound_hazard = sum(
        int(item.get("contact_audit", {}).get("inbound_hazard_contact_frames", 0)) > 0
        for item in complete
    )
    outbound_hazard = sum(
        int(item.get("contact_audit", {}).get("outbound_hazard_contact_frames", 0)) > 0
        for item in complete
    )
    other_contact = sum(
        int(
            item.get("contact_audit", {})
            .get("contact_class_totals", {})
            .get("other_environment", 0)
        )
        > 0
        for item in complete
    )
    bow_fallback = sum(item.get("bow_fallback_taken") is True for item in complete)
    try:
        row_root = Path(output_root).resolve().relative_to(ROOT)
    except ValueError:
        row_root = Path(output_root)
    reconciled = len(results) == N_EXPERT_ROWS and all(
        item["status"] in TERMINAL_STATUSES for item in results
    )
    no_infrastructure = all(
        item["status"] != "infrastructure_failure" for item in results
    )
    passed = reconciled and no_infrastructure and clean >= MIN_CLEAN_SUCCESSES
    is_diagnostic = role == "diagnostic"
    summary: dict[str, Any] = {
        "schema_version": "pact_place_expert_screen_v1",
        "config_sha256": contract["config_sha256"],
        "workers": workers,
        "gate_frozen_before_execution": False if is_diagnostic else True,
        "reconciled": reconciled,
        "n": len(results),
        "complete_rows": len(complete),
        "sampling_failures": sum(
            item["status"] == "sampling_failure" for item in results
        ),
        "infrastructure_failures": sum(
            item["status"] == "infrastructure_failure" for item in results
        ),
        "gate": {
            "minimum_clean_successes": MIN_CLEAN_SUCCESSES,
            "clean_successes": clean,
            "clean_success_rate": clean / N_EXPERT_ROWS,
            **({} if is_diagnostic else {"passed": passed}),
        },
        "task_success": {"count": task, "rate": task / N_EXPERT_ROWS},
        "grasp_phase_success": {"count": grasp, "rate": grasp / N_EXPERT_ROWS},
        "place_phase_success": {"count": place, "rate": place / N_EXPERT_ROWS},
        "place_success_given_grasp": {
            "numerator": sum(
                item.get("grasp_phase_success") is True
                and item.get("place_phase_success") is True
                for item in results
            ),
            "denominator": grasp,
        },
        "hazard_contact_episodes": {
            "inbound": inbound_hazard,
            "outbound": outbound_hazard,
        },
        "other_environment_contact_episodes": other_contact,
        "place_receptacle_contact_exempt": True,
        "bow_fallback_episodes": bow_fallback,
        "row_result_paths": [
            str(_result_path(row_root, row))
            for row in contract["expert_screen_rows"]
        ],
    }
    if is_diagnostic:
        summary["role"] = "diagnostic_not_a_gate"
        summary["authorizes_collection"] = False
        summary["diagnostic_clean_successes"] = clean
        summary["next_action"] = "none_diagnostic_only"
    else:
        summary["decision"] = PASS_TOKEN if passed else FAIL_TOKEN
        summary["next_action"] = (
            "proceed_to_collection_design"
            if passed
            else "stop_without_collection_or_training"
        )
    summary["expert_screen_sha256"] = sha256_payload(summary)
    return summary


def _protected_eval_processes() -> list[int]:
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


def verify_protected_artifacts(contract: dict[str, Any]) -> None:
    changed = {
        relative: {"expected": digest, "actual": sha256_file(ROOT / relative)}
        for relative, digest in contract["protected_artifact_sha256_before"].items()
        if not (ROOT / relative).is_file() or sha256_file(ROOT / relative) != digest
    }
    if changed:
        raise RuntimeError(f"protected frozen artifacts changed: {changed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--role",
        choices=("gate", "diagnostic"),
        default="gate",
        help="gate emits pass/fail tokens; diagnostic cannot.",
    )
    parser.add_argument(
        "--development-row",
        type=int,
        help="Run one non-gate diagnostic row under output-root; do not summarize.",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be in [1, 8]")
    protected = _protected_eval_processes()
    if protected:
        raise SystemExit(f"protected confirmatory evaluation is active: {protected}")
    contract = load_contract(args.config)
    verify_protected_artifacts(contract)
    rows = contract["expert_screen_rows"]
    if args.development_row is not None:
        if not 0 <= args.development_row < len(rows):
            raise SystemExit("development row index is outside the manifest")
        result = run_row(
            rows[args.development_row],
            config_sha256=contract["config_sha256"],
            output_root=str(args.output_root),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "complete" else 1

    results: list[dict[str, Any]] = []
    pending = []
    for row in rows:
        existing = _validate_existing(
            _result_path(args.output_root, row), row, contract["config_sha256"]
        )
        if existing is None:
            pending.append(row)
        else:
            results.append(existing)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        max_tasks_per_child=1,
    ) as executor:
        futures = {
            executor.submit(
                run_row,
                row,
                config_sha256=contract["config_sha256"],
                output_root=str(args.output_root),
            ): row
            for row in pending
        }
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"row={row['role_index']:02d} side={row['intrusion_side']} "
                f"status={result['status']} clean={result.get('clean_success')} "
                f"grasp={result.get('grasp_phase_success')} "
                f"place={result.get('place_phase_success')}",
                flush=True,
            )
    results.sort(key=lambda item: item["role_index"])
    summary = summarize(
        contract, results, args.workers, args.output_root, role=args.role
    )
    if args.role == "diagnostic":
        dumped = json.dumps(summary)
        if PASS_TOKEN in dumped or FAIL_TOKEN in dumped or "proceed_" in dumped:
            raise RuntimeError("diagnostic summary emitted a gate token or proceed action")
        role_record = {
            "role": "diagnostic_not_a_gate",
            "authorizes_collection": False,
            "next_action": "none_diagnostic_only",
            "gate_frozen_before_execution": False,
            "diagnostic_clean_successes": summary["diagnostic_clean_successes"],
            "config_sha256": contract["config_sha256"],
            "expert_screen_sha256": summary["expert_screen_sha256"],
        }
        write_json_atomic(args.output_root / "role.json", role_record)
    verify_protected_artifacts(contract)
    write_json_atomic(args.output_root / "expert_screen.json", summary)
    if args.role == "gate" and summary.get("decision") == FAIL_TOKEN:
        stop = {
            "schema_version": "pact_place_corridor_stop_v1",
            "config_sha256": contract["config_sha256"],
            "expert_screen_sha256": summary["expert_screen_sha256"],
            "reason": (
                "phase0_expert_gate_failed_or_did_not_reconcile; no collection, "
                "encoder work, training, or policy evaluation authorized"
            ),
            "decision": FAIL_TOKEN,
        }
        stop["stop_record_sha256"] = sha256_payload(stop)
        write_json_atomic(args.output_root / "stop_record.json", stop)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["reconciled"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
