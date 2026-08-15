#!/usr/bin/env python3
"""Freeze execution and runtime provenance for the 450-row blind-RGB schedule."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pact_blind_rgb_contract import sha256_file, sha256_payload


ROOT = Path(__file__).resolve().parents[1]
WORKERS = 12
WATCHDOG_SECONDS = 600.0
ARMS = ("ACT", "PACT", "PACT_PERMUTED")


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != sha256_payload(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return str(observed)


def verify_frozen_models(schedule: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, dict[str, str]] = {"checkpoints": {}, "dataset_stats": {}, "surface_encoders": {}}
    for row in schedule["rows"]:
        groups["checkpoints"][row["checkpoint_path"]] = row["checkpoint_sha256"]
        groups["dataset_stats"][row["dataset_stats_path"]] = row["dataset_stats_sha256"]
        if row["surface_encoder_path"] is not None:
            groups["surface_encoders"][row["surface_encoder_path"]] = row["surface_encoder_sha256"]
    output = {}
    for label, records in groups.items():
        output[label] = []
        for raw, expected in sorted(records.items()):
            observed = sha256_file(raw)
            if observed != expected:
                raise ValueError(f"{label} changed: {raw}")
            output[label].append({"path": raw, "sha256": observed})
    return output


def worker_sizing() -> dict[str, Any]:
    observed_workers = 8
    observed_peak_mib = 12320.0
    ceiling_mib = 19 * 1024
    per_worker = observed_peak_mib / observed_workers
    arithmetic_limit = math.floor(ceiling_mib / per_worker)
    selected = min(12, arithmetic_limit)
    if selected != WORKERS:
        raise ValueError("worker memory arithmetic no longer permits twelve workers")
    return {
        "source": "outcome-blind v2 eight-worker telemetry reused by the blur sweep",
        "observed_workers": observed_workers,
        "observed_peak_gpu_memory_mib": observed_peak_mib,
        "observed_gpu_utilization_percent": 43.0,
        "observed_cpu_load": 8.5,
        "logical_cpu_cores": 128,
        "per_worker_peak_mib": per_worker,
        "ceiling_mib": ceiling_mib,
        "arithmetic_worker_limit": arithmetic_limit,
        "worker_cap": 12,
        "selected_workers": selected,
        "projected_peak_mib": selected * per_worker,
        "headroom_mib": ceiling_mib - selected * per_worker,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--policy-registry", required=True, type=Path)
    parser.add_argument("--token-plan", required=True, type=Path)
    parser.add_argument("--analysis-script", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace blind-RGB dispatch: {args.output}")
    schedule = json.loads(args.schedule.read_text())
    manifest = json.loads(args.manifest.read_text())
    registry = json.loads(args.policy_registry.read_text())
    token_plan = json.loads(args.token_plan.read_text())
    schedule_sha = validate_self_hash(schedule, "schedule_sha256", "schedule")
    manifest_sha = validate_self_hash(manifest, "manifest_sha256", "manifest")
    registry_sha = validate_self_hash(registry, "policy_registry_sha256", "policy registry")
    token_sha = validate_self_hash(token_plan, "token_plan_sha256", "token plan")
    if (
        schedule.get("schema_version") != "pact_blind_rgb_schedule_v1"
        or schedule.get("workers") != WORKERS
        or schedule.get("rollouts") != 450
        or len(schedule.get("rows", [])) != 450
        or schedule.get("manifest_sha256") != manifest_sha
        or schedule.get("policy_registry_sha256") != registry_sha
        or schedule.get("token_plan_sha256") != token_sha
        or schedule.get("analysis_script_sha256") != sha256_file(args.analysis_script)
    ):
        raise ValueError("blind-RGB schedule bindings changed")
    for relative, expected in manifest["protected_scientific_artifacts"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"protected scientific artifact changed: {relative}")
    token_file = Path(token_plan["files"]["tokens"]["path"])
    if sha256_file(token_file) != token_plan["files"]["tokens"]["sha256"]:
        raise ValueError("permuted token tensor changed")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("blind-RGB evaluation output root is not empty")
    smoke = schedule["rows"][0]
    if (
        smoke["vision_condition"] != "blind"
        or smoke["arm"] != "PACT_PERMUTED"
        or smoke["checkpoint_seed"] != 3101
    ):
        raise ValueError("smoke must exercise blind PACT_PERMUTED seed 3101")
    runtime_paths = {
        "main_contract": ROOT / "scripts/pact_blind_rgb_contract.py",
        "config_builder": ROOT / "scripts/build_pact_blind_rgb_config.py",
        "schedule_builder": ROOT / "scripts/build_pact_blind_rgb_schedule.py",
        "dispatch_builder": Path(__file__),
        "supervisor": ROOT / "scripts/run_pact_blind_rgb_supervisor.py",
        "detached_launcher": ROOT / "scripts/launch_pact_blind_rgb_detached.py",
        "full_launcher": ROOT / "scripts/launch_pact_blind_rgb_full.py",
        "detachment_proof": ROOT / "scripts/prove_pact_blind_rgb_detachment.py",
        "intervention_preflight": ROOT / "scripts/run_pact_blind_rgb_preflight.py",
        "storage_compactor": ROOT / "scripts/compact_pact_geometry_storage.py",
        "storage_dependency": ROOT / "scripts/compact_pact_contact_storage.py",
        "throughput": ROOT / "scripts/measure_pact_geometry_throughput.py",
        "evaluator": ROOT / "submodules/act/eval_pact_blind_rgb_row.py",
        "contact_evaluator": ROOT / "submodules/act/eval_pact_contact_endpoint_row.py",
        "legacy_evaluator": ROOT / "submodules/act/eval_pact_collision_row.py",
        "live_evaluator": ROOT / "submodules/act/eval_pact_frontend_screen_row.py",
        "permuted_evaluator": ROOT / "submodules/act/eval_pact_valid_ablation_row.py",
        "generic_schedule_dependency": ROOT / "scripts/run_pact_confirmatory_schedule.py",
        "frontend_supervisor_dependency": ROOT / "scripts/run_pact_frontend_screen_supervisor.py",
        "geometry_supervisor_dependency": ROOT / "scripts/run_pact_geometry_supervisor.py",
        "blur_supervisor_dependency": ROOT / "scripts/run_pact_blur_supervisor.py",
        "analysis": args.analysis_script,
        "rgb_primitive": ROOT / "submodules/act/pact_blur.py",
        "samplers": ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
        "scene_xml": ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml",
        "contact_taxonomy": ROOT / "submodules/molmospaces/molmo_spaces/tasks/pact_contact_audit.py",
    }
    missing = [label for label, path in runtime_paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"runtime files missing: {missing}")
    document: dict[str, Any] = {
        "schema_version": "pact_blind_rgb_dispatch_v1",
        "scientific_schedule": {
            "path": str(args.schedule.resolve()),
            "file_sha256": sha256_file(args.schedule),
            "schedule_sha256": schedule_sha,
            "rows": 450,
            "workers": WORKERS,
            "rows_changed": 0,
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
        },
        "execution": {
            "output_root": str(output_root),
            "fresh_subprocess_per_rollout": True,
            "fixed_worker_count": WORKERS,
            "cpu_affinity": list(range(32)),
            "no_outcome_based_row_replacement": True,
            "policy_outcomes_seen_before_freeze": False,
            "setsid": True,
            "nohup": True,
            "detached_stdin": True,
        },
        "worker_sizing": worker_sizing(),
        "watchdog": {
            "no_completion_seconds": WATCHDOG_SECONDS,
            "restart_only_if_no_active_initial_observation_accepted": True,
            "restart_all_active_rows_never_subset": True,
            "event_log_directory": "watchdog_events",
        },
        "launch_smoke": {
            "required_before_full_dispatch": True,
            "schedule_index": 0,
            "rollout_id": smoke["rollout_id"],
            "instance_episode_id": smoke["instance_episode_id"],
            "schedule_row_sha256": smoke["schedule_row_sha256"],
            "output_relpath": smoke["output_relpath"],
            "arm": smoke["arm"],
            "checkpoint_seed": smoke["checkpoint_seed"],
            "vision_condition": smoke["vision_condition"],
            "required_artifact": "launch_smoke.json",
            "required_result_status": "complete",
            "full_dispatch_must_reconcile_without_rerun": True,
        },
        "detachment_proof": {
            "required_before_full_dispatch": True,
            "required_artifact": "detachment_proof.json",
            "kill_launching_shell_during_smoke": True,
            "endpoint_outcome_fields_inspected": False,
        },
        "intervention_preflight": {
            "required_before_full_dispatch": True,
            "required_artifact": str((ROOT / "diagnostics_output/pact_blind_rgb/preflight.json").resolve()),
            "arms": list(ARMS),
            "vision_conditions": ["sighted", "blind"],
            "same_instance_and_seed": True,
            "require_exact_imagenet_mean": True,
            "require_action_trace_change": True,
            "require_proximity_byte_identity": True,
            "require_sighted_no_flag_bit_identity": True,
        },
        "throughput": {
            "outcome_blind_window_minutes": 20,
            "artifact": "throughput_first_20_minutes.json",
            "endpoint_fields_read": False,
        },
        "boundary_amendment": {
            "row_terminal_boundary": "valid scientific result.json",
            "all_inflight_rows_rerun": True,
            "individual_post_observation_retry": False,
            "pre_observation_retry": True,
            "cohort_exit_window_seconds": 5,
        },
        "storage": {
            "compact_every_completed_row": True,
            "retain_raw_rollout_payloads": False,
            "summary_only_contact_instrumentation": True,
            "outcome_based_selection": False,
        },
        "analysis": {
            "path": str(args.analysis_script.resolve()),
            "sha256": sha256_file(args.analysis_script),
            "bootstrap_replicates": schedule["bootstrap_replicates"],
            "cluster": schedule["analysis_contract"]["cluster"],
            "seeds_unpooled_first": True,
            "decision_rule": schedule["decision_rule"],
        },
        "protected_scientific_artifacts": manifest["protected_scientific_artifacts"],
        "frozen_inputs": {
            "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256_file(args.manifest)},
            "policy_registry": {"path": str(args.policy_registry.resolve()), "sha256": sha256_file(args.policy_registry)},
            "token_plan": {"path": str(args.token_plan.resolve()), "sha256": sha256_file(args.token_plan), "token_plan_sha256": token_sha},
            "models": verify_frozen_models(schedule),
            "runtime": {
                label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for label, path in sorted(runtime_paths.items())
            },
        },
    }
    document["dispatch_contract_sha256"] = sha256_payload(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["dispatch_contract_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
