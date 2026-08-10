#!/usr/bin/env python3
"""Freeze models, runtime, analysis, and RGB-blur durable dispatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pact_blur_sweep_contract import sha256_file, sha256_payload


ROOT = Path(__file__).resolve().parents[1]
WORKERS = 12
WATCHDOG_SECONDS = 600
FROZEN_CONTACT_HASHES = {
    ROOT / "docs/PACT_CONTACT_ENDPOINT_DECISION.md": "6ea97ea789e1c5d57572c7e231b30c867881f0e21b81ad11f950cef9bf860f1b",
    ROOT / "diagnostics_output/pact_contact_endpoint/analysis.json": "4bbf25c97472bbfd8f13c3352b6b1c8afae9284cd00351e0502149362ccde1f7",
    ROOT / "diagnostics_output/pact_contact_endpoint/final_decision.json": "52ce515bc09771d9ecdc8a39e31111f66e7e632f5949e82ed2497908ac2458bf",
}
FROZEN_V3_HASHES = {
    ROOT / "docs/PACT_GEOMETRY_GENERALIZATION_V3.md": "143497c7d9c100fb05400e3f716690199699d86ad4eccd745da9f74622467105",
    ROOT / "diagnostics_output/pact_geometry_generalization_v3/analysis.json": "29957bdbad56018390576b67f52237a03ed2f96c468ad4e393f3047806ca7b1a",
    ROOT / "diagnostics_output/pact_geometry_generalization_v3/final_decision.json": "ce8bb7770cf23bdd44b90006113807458a7822be4cf4a6f68889fbeae9337681",
}


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != sha256_payload(payload):
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
        groups["dataset_stats"][row["dataset_stats_path"]] = row["dataset_stats_sha256"]
        if row["surface_encoder_path"] is not None:
            groups["surface_encoders"][row["surface_encoder_path"]] = row[
                "surface_encoder_sha256"
            ]
    output = {}
    for label, items in groups.items():
        output[label] = []
        for raw_path, expected in sorted(items.items()):
            observed = sha256_file(raw_path)
            if observed != expected:
                raise ValueError(f"{label} changed: {raw_path}")
            output[label].append({"path": raw_path, "sha256": observed})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--policy-registry", required=True, type=Path)
    parser.add_argument("--token-plan", required=True, type=Path)
    parser.add_argument("--analysis-script", required=True, type=Path)
    parser.add_argument("--worker-sizing", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    manifest = json.loads(args.manifest.read_text())
    calibration = json.loads(args.calibration.read_text())
    registry = json.loads(args.policy_registry.read_text())
    token_plan = json.loads(args.token_plan.read_text())
    worker_sizing = json.loads(args.worker_sizing.read_text())
    schedule_sha = validate_self_hash(schedule, "schedule_sha256", "schedule")
    manifest_sha = validate_self_hash(manifest, "manifest_sha256", "manifest")
    calibration_sha = validate_self_hash(
        calibration, "calibration_sha256", "blur calibration"
    )
    registry_sha = validate_self_hash(registry, "policy_registry_sha256", "policy registry")
    token_sha = validate_self_hash(token_plan, "token_plan_sha256", "token plan")
    if (
        schedule.get("schema_version") != "pact_blur_sweep_schedule_v1"
        or schedule.get("workers") != WORKERS
        or schedule.get("rollouts") != 900
        or schedule.get("manifest_sha256") != manifest_sha
        or schedule.get("calibration_sha256") != calibration_sha
        or schedule.get("policy_registry_sha256") != registry_sha
        or schedule.get("token_plan_sha256") != token_sha
        or schedule.get("analysis_script_sha256") != sha256_file(args.analysis_script)
    ):
        raise ValueError("RGB blur schedule bindings changed")
    sizing_payload = dict(worker_sizing)
    sizing_sha = sizing_payload.pop("worker_sizing_sha256", None)
    if (
        sizing_sha != sha256_payload(sizing_payload)
        or worker_sizing.get("selected_workers") != WORKERS
        or worker_sizing.get("projected_peak_mib") > worker_sizing.get("ceiling_mib")
    ):
        raise ValueError("RGB-blur worker sizing is invalid")
    for path, expected in {**FROZEN_CONTACT_HASHES, **FROZEN_V3_HASHES}.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen contact endpoint changed: {path}")
    token_file = Path(token_plan["files"]["tokens"]["path"])
    if sha256_file(token_file) != token_plan["files"]["tokens"]["sha256"]:
        raise ValueError("permuted token tensor changed")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("RGB blur evaluation output root is not empty")
    smoke = schedule["rows"][0]
    if (
        smoke["blur_sigma"] != 0.0
        or smoke["arm"] != "PACT_PERMUTED"
        or smoke["checkpoint_seed"] != 3101
    ):
        raise ValueError("smoke must exercise sharp PACT_PERMUTED seed 3101")
    runtime_paths = {
        "main_contract": ROOT / "scripts/pact_blur_sweep_contract.py",
        "manifest_builder": ROOT / "scripts/build_pact_blur_manifest.py",
        "schedule_builder": ROOT / "scripts/build_pact_blur_schedule.py",
        "dispatch_builder": Path(__file__),
        "supervisor": ROOT / "scripts/run_pact_blur_supervisor.py",
        "detached_launcher": ROOT / "scripts/launch_pact_blur_detached.py",
        "full_launcher": ROOT / "scripts/launch_pact_blur_full.py",
        "detachment_proof": ROOT / "scripts/prove_pact_blur_detachment.py",
        "intervention_preflight": ROOT / "scripts/run_pact_blur_preflight.py",
        "proof_dependency": ROOT / "scripts/prove_pact_frontend_screen_detachment.py",
        "storage_compactor": ROOT / "scripts/compact_pact_geometry_storage.py",
        "storage_dependency": ROOT / "scripts/compact_pact_contact_storage.py",
        "throughput": ROOT / "scripts/measure_pact_geometry_throughput.py",
        "evaluator": ROOT / "submodules/act/eval_pact_blur_sweep_row.py",
        "contact_evaluator": ROOT / "submodules/act/eval_pact_contact_endpoint_row.py",
        "legacy_evaluator": ROOT / "submodules/act/eval_pact_collision_row.py",
        "live_evaluator": ROOT / "submodules/act/eval_pact_frontend_screen_row.py",
        "permuted_evaluator": ROOT / "submodules/act/eval_pact_valid_ablation_row.py",
        "supervisor_dependency": ROOT / "scripts/run_pact_frontend_screen_supervisor.py",
        "geometry_supervisor_dependency": ROOT / "scripts/run_pact_geometry_supervisor.py",
        "generic_schedule_dependency": ROOT / "scripts/run_pact_confirmatory_schedule.py",
        "analysis": args.analysis_script,
        "blur_primitive": ROOT / "submodules/act/pact_blur.py",
        "calibration": ROOT / "scripts/calibrate_pact_blur.py",
        "samplers": ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
        "scene_xml": ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml",
        "contact_taxonomy": ROOT / "submodules/molmospaces/molmo_spaces/tasks/pact_contact_audit.py",
    }
    missing = [label for label, path in runtime_paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"runtime files missing: {missing}")
    document: dict[str, Any] = {
        "schema_version": "pact_blur_sweep_dispatch",
        "scientific_schedule": {
            "path": str(args.schedule.resolve()),
            "file_sha256": sha256_file(args.schedule),
            "schedule_sha256": schedule_sha,
            "rows": 900,
            "workers": WORKERS,
            "rows_changed": 0,
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
        },
        "execution": {
            "output_root": str(output_root),
            "fresh_subprocess_per_rollout": True,
            "fixed_worker_count": WORKERS,
            "no_outcome_based_row_replacement": True,
            "policy_outcomes_seen_before_freeze": False,
            "setsid": True,
            "nohup": True,
            "detached_stdin": True,
        },
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
            "required_artifact": str(
                (ROOT / "diagnostics_output/pact_blur_sweep/preflight.json").resolve()
            ),
            "arms": ["ACT", "PACT", "PACT_PERMUTED"],
            "sigmas": [0.0, 2.0],
            "same_instance_and_seed": True,
            "require_visual_change": True,
            "require_action_trace_change": True,
            "require_proximity_identity": True,
            "require_sharp_v3_endpoint_identity": True,
            "endpoint_values_used_for_parameter_selection": False,
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
            "cluster": "instance; all sigmas, arms, and seeds move together",
            "seeds_unpooled_first": True,
            "decision_rule": schedule["decision_rule"],
        },
        "frozen_contact_endpoint_hashes": {
            str(path): digest for path, digest in FROZEN_CONTACT_HASHES.items()
        },
        "frozen_geometry_v3_hashes": {
            str(path): digest for path, digest in FROZEN_V3_HASHES.items()
        },
        "frozen_inputs": {
            "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256_file(args.manifest)},
            "calibration": {
                "path": str(args.calibration.resolve()),
                "sha256": sha256_file(args.calibration),
                "calibration_sha256": calibration_sha,
            },
            "policy_registry": {
                "path": str(args.policy_registry.resolve()),
                "sha256": sha256_file(args.policy_registry),
            },
            "token_plan": {
                "path": str(args.token_plan.resolve()),
                "sha256": sha256_file(args.token_plan),
                "token_plan_sha256": token_sha,
                "global_tensor": token_plan["files"]["tokens"],
            },
            "worker_sizing": {
                "path": str(args.worker_sizing.resolve()),
                "sha256": sha256_file(args.worker_sizing),
                "worker_sizing_sha256": sizing_sha,
            },
            "runtime": {
                label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for label, path in runtime_paths.items()
            },
            **verified_models(schedule),
        },
    }
    document["dispatch_contract_sha256"] = sha256_payload(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["dispatch_contract_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
