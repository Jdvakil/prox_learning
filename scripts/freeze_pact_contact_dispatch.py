#!/usr/bin/env python3
"""Freeze schedule, models, analysis, token data, and contact runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("ACT", "PACT", "PACT_ZERO", "PACT_PERMUTED")
SEEDS = (3101, 3102, 3103)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != canonical_hash(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return str(observed)


def verified_models(schedule: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, dict[str, str]] = {
        "checkpoints": {},
        "dataset_stats": {},
        "surface_encoders": {},
    }
    for row in schedule["rows"]:
        groups["checkpoints"][row["checkpoint_path"]] = row["checkpoint_sha256"]
        groups["dataset_stats"][row["dataset_stats_path"]] = row[
            "dataset_stats_sha256"
        ]
        if row["surface_encoder_path"] is not None:
            groups["surface_encoders"][row["surface_encoder_path"]] = row[
                "surface_encoder_sha256"
            ]
    output: dict[str, list[dict[str, str]]] = {}
    for label, records in groups.items():
        output[label] = []
        for raw_path, expected in sorted(records.items()):
            path = Path(raw_path)
            observed = file_hash(path)
            if observed != expected:
                raise ValueError(f"{label} hash mismatch for {path}")
            output[label].append({"path": str(path), "sha256": observed})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--policy-registry", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--token-plan", required=True, type=Path)
    parser.add_argument("--occlusion", required=True, type=Path)
    parser.add_argument("--power", required=True, type=Path)
    parser.add_argument("--analysis-script", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    manifest = json.loads(args.manifest.read_text())
    registry = json.loads(args.policy_registry.read_text())
    preregistration = json.loads(args.preregistration.read_text())
    token_plan = json.loads(args.token_plan.read_text())
    occlusion = json.loads(args.occlusion.read_text())
    power = json.loads(args.power.read_text())
    schedule_sha = validate_self_hash(schedule, "schedule_sha256", "schedule")
    manifest_sha = validate_self_hash(manifest, "manifest_sha256", "manifest")
    registry_sha = validate_self_hash(
        registry, "policy_registry_sha256", "policy registry"
    )
    prereg_sha = validate_self_hash(
        preregistration, "preregistration_sha256", "preregistration"
    )
    token_sha = validate_self_hash(token_plan, "token_plan_sha256", "token plan")
    occlusion_sha = validate_self_hash(
        occlusion, "occlusion_subset_sha256", "occlusion partition"
    )
    power_sha = validate_self_hash(power, "power_sha256", "power")
    expected_counts = Counter(
        {(seed, arm): 100 for seed in SEEDS for arm in ARMS}
    )
    if (
        schedule.get("schema_version") != "pact_contact_endpoint_schedule_v1"
        or schedule.get("instance_count") != 100
        or schedule.get("rollouts") != 1200
        or schedule.get("workers") != 8
        or len(schedule.get("rows", [])) != 1200
        or Counter((row["checkpoint_seed"], row["arm"]) for row in schedule["rows"])
        != expected_counts
        or [row["schedule_index"] for row in schedule["rows"]] != list(range(1200))
        or len({row["rollout_id"] for row in schedule["rows"]}) != 1200
    ):
        raise ValueError("contact schedule design changed")
    if (
        schedule["manifest_sha256"] != manifest_sha
        or schedule["policy_registry_sha256"] != registry_sha
        or schedule["preregistration_sha256"] != prereg_sha
        or schedule["token_plan_sha256"] != token_sha
        or schedule["occlusion_subset_sha256"] != occlusion_sha
        or schedule["power_sha256"] != power_sha
    ):
        raise ValueError("schedule frozen-input binding changed")
    analyzer_sha = file_hash(args.analysis_script)
    if analyzer_sha != preregistration["analysis"]["frozen_analysis_script_sha256"]:
        raise ValueError("analysis script differs from preregistration")
    token_record = token_plan["files"]["tokens"]
    if file_hash(Path(token_record["path"])) != token_record["sha256"]:
        raise ValueError("global token tensor differs from token plan")
    smoke = schedule["rows"][0]
    if (
        smoke["arm"] != "PACT_PERMUTED"
        or smoke["checkpoint_seed"] != 3101
        or smoke["max_control_steps"] != 900
        or smoke["token_plan_sha256"] != token_sha
    ):
        raise ValueError("smoke does not exercise contact token plan")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise ValueError("contact output root is not empty")
    runtime_paths = {
        "contact_contract": ROOT / "scripts/pact_contact_endpoint_contract.py",
        "contact_supervisor": ROOT / "scripts/run_pact_contact_supervisor.py",
        "contact_launcher": ROOT / "scripts/launch_pact_contact_detached.py",
        "contact_evaluator": ROOT / "submodules/act/eval_pact_contact_endpoint_row.py",
        "contact_detachment_proof": ROOT / "scripts/prove_pact_contact_detachment.py",
        "contact_compactor": ROOT / "scripts/compact_pact_contact_storage.py",
        "contact_analyzer": args.analysis_script,
        "contact_finalizer": ROOT / "scripts/finalize_pact_contact_endpoint.py",
        "contact_committer": ROOT / "scripts/commit_pact_contact_results.py",
        "contact_full_stack_launcher": ROOT / "scripts/launch_pact_contact_full_stack.py",
        "contact_preparation_controller": ROOT / "scripts/prepare_and_launch_pact_contact.py",
        "frontend_supervisor_dependency": ROOT / "scripts/run_pact_frontend_screen_supervisor.py",
        "frontend_launcher_dependency": ROOT / "scripts/launch_pact_frontend_screen_detached.py",
        "frontend_detachment_dependency": ROOT / "scripts/prove_pact_frontend_screen_detachment.py",
        "generic_schedule_dependency": ROOT / "scripts/run_pact_confirmatory_schedule.py",
        "throughput": ROOT / "scripts/measure_pact_contact_throughput.py",
        "live_evaluator_dependency": ROOT / "submodules/act/eval_pact_frontend_screen_row.py",
        "permuted_evaluator_dependency": ROOT / "submodules/act/eval_pact_valid_ablation_row.py",
        "legacy_evaluator_dependency": ROOT / "submodules/act/eval_pact_collision_row.py",
        "contact_taxonomy": ROOT / "submodules/molmospaces/molmo_spaces/tasks/pact_contact_audit.py",
        "camera_config": ROOT / "submodules/molmospaces/molmo_spaces/configs/camera_configs.py",
        "robot_config": ROOT / "submodules/molmospaces/molmo_spaces/configs/robot_configs.py",
        "task_config": ROOT / "submodules/molmospaces/molmo_spaces/configs/task_configs.py",
        "datagen_config": ROOT / "submodules/molmospaces/molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py",
        "corridor_sampler": ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
        "corridor_scene_xml": ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml",
        "corridor_scene_metadata": ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_collision_corridor_metadata.json",
        "hybrid_robot_xml": ROOT / "assets/robots/franka_skin/model_hybrid.xml",
    }
    missing = [label for label, path in runtime_paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"contact runtime files are missing: {missing}")
    document: dict[str, Any] = {
        "schema_version": "pact_contact_endpoint_dispatch_v1",
        "scientific_schedule": {
            "path": str(args.schedule.resolve()),
            "file_sha256": file_hash(args.schedule),
            "schedule_sha256": schedule_sha,
            "rows": 1200,
            "workers": 8,
            "rows_changed": 0,
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": file_hash(args.manifest),
        },
        "execution": {
            "output_root": str(output_root),
            "fresh_subprocess_per_rollout": True,
            "fixed_worker_count": 8,
            "no_outcome_based_row_replacement": True,
            "outcomes_seen_before_freeze": False,
            "setsid": True,
            "nohup": True,
            "detached_stdin": True,
        },
        "launch_smoke": {
            "required_before_full_dispatch": True,
            "schedule_index": smoke["schedule_index"],
            "rollout_id": smoke["rollout_id"],
            "instance_episode_id": smoke["instance_episode_id"],
            "schedule_row_sha256": smoke["schedule_row_sha256"],
            "output_relpath": smoke["output_relpath"],
            "arm": smoke["arm"],
            "checkpoint_seed": smoke["checkpoint_seed"],
            "max_control_steps": smoke["max_control_steps"],
            "token_plan_sha256": smoke["token_plan_sha256"],
            "required_artifact": "launch_smoke.json",
            "required_result_status": "complete",
            "explicit_horizon_and_endpoint_instrumentation_check": True,
            "full_dispatch_must_reconcile_without_rerun": True,
        },
        "detachment_proof": {
            "required_before_full_dispatch": True,
            "required_artifact": "detachment_proof.json",
            "kill_launching_shell_during_smoke": True,
            "endpoint_outcome_fields_inspected": False,
        },
        "throughput": {
            "required_measurement_elapsed_minutes": 20,
            "required_artifact": "throughput_first_20_minutes.json",
            "endpoint_fields_read": False,
        },
        "storage": {
            "summary_only_contact_instrumentation": True,
            "raw_intact_schedule_indices": [0, 1199],
            "eligible_rows_compacted": list(range(1, 1199)),
            "payload_deletion_recoverable": False,
            "outcome_based_selection": False,
        },
        "boundary_amendment": {
            "row_terminal_boundary": "valid scientific result.json",
            "all_inflight_rows_rerun": True,
            "individual_post_observation_retry": False,
            "pre_observation_retry": True,
            "cohort_exit_window_seconds": 5,
        },
        "analysis": {
            "path": str(args.analysis_script.resolve()),
            "sha256": analyzer_sha,
            "bootstrap_replicates": 20000,
            "cluster": "whole instance; all arms/seeds move together",
            "seed_specific_results_first": True,
            "decision_rule": preregistration["decision_rule"],
        },
        "frozen_inputs": {
            "policy_registry": {
                "path": str(args.policy_registry.resolve()),
                "sha256": file_hash(args.policy_registry),
            },
            "preregistration": {
                "path": str(args.preregistration.resolve()),
                "sha256": file_hash(args.preregistration),
            },
            "token_plan": {
                "path": str(args.token_plan.resolve()),
                "sha256": file_hash(args.token_plan),
                "token_plan_sha256": token_sha,
                "global_tensor": token_record,
            },
            "occlusion": {
                "path": str(args.occlusion.resolve()),
                "sha256": file_hash(args.occlusion),
            },
            "power": {
                "path": str(args.power.resolve()),
                "sha256": file_hash(args.power),
            },
            "runtime": {
                label: {"path": str(path.resolve()), "sha256": file_hash(path)}
                for label, path in runtime_paths.items()
            },
            **verified_models(schedule),
        },
    }
    document["dispatch_contract_sha256"] = canonical_hash(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["dispatch_contract_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
