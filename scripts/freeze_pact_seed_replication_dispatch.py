#!/usr/bin/env python3
"""Freeze seed-replication schedule, models, analysis, and runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


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
        groups["dataset_stats"][row["dataset_stats_path"]] = row["dataset_stats_sha256"]
        if row["surface_encoder_path"] is not None:
            groups["surface_encoders"][row["surface_encoder_path"]] = row["surface_encoder_sha256"]
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
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--token-plan", required=True, type=Path)
    parser.add_argument("--analysis-script", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    manifest = json.loads(args.manifest.read_text())
    training = json.loads(args.training_summary.read_text())
    preregistration = json.loads(args.preregistration.read_text())
    token_plan = json.loads(args.token_plan.read_text())
    validate_self_hash(schedule, "schedule_sha256", "schedule")
    validate_self_hash(manifest, "manifest_sha256", "manifest")
    prereg_sha = validate_self_hash(preregistration, "preregistration_sha256", "preregistration")
    token_sha = validate_self_hash(token_plan, "token_plan_sha256", "token plan")
    if (
        schedule.get("schema_version") != "pact_seed_replication_schedule_v1"
        or schedule.get("instances") != 40
        or schedule.get("rollouts") != 120
        or schedule.get("workers") != 8
        or len(schedule.get("rows", [])) != 120
        or Counter(row["arm"] for row in schedule["rows"])
        != {"ACT": 40, "PACT": 40, "PACT_PERMUTED": 40}
        or [row["schedule_index"] for row in schedule["rows"]] != list(range(120))
        or len({row["rollout_id"] for row in schedule["rows"]}) != 120
    ):
        raise ValueError("seed-replication schedule design changed")
    if schedule["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("schedule manifest changed")
    if schedule["training_summary_sha256"] != canonical_hash(training):
        raise ValueError("training summary changed")
    if schedule["preregistration_sha256"] != prereg_sha:
        raise ValueError("preregistration changed")
    if schedule["token_plan_sha256"] != token_sha:
        raise ValueError("permutation plan changed")
    if (
        file_hash(args.analysis_script)
        != preregistration["analysis"]["frozen_analysis_script_sha256"]
    ):
        raise ValueError("analysis script differs from preregistration")
    smoke = schedule["rows"][0]
    if (
        smoke["arm"] != "PACT_PERMUTED"
        or smoke["max_control_steps"] != 900
        or smoke["token_plan_sha256"] != token_sha
    ):
        raise ValueError("smoke does not exercise the 900-step permutation plan")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(item.is_file() for item in output_root.rglob("*")):
        raise ValueError("seed-replication output root is not empty")
    runtime_paths = {
        "seed_supervisor": ROOT / "scripts/run_pact_seed_replication_supervisor.py",
        "seed_launcher": ROOT / "scripts/launch_pact_seed_replication_detached.py",
        "seed_evaluator": ROOT / "submodules/act/eval_pact_seed_replication_row.py",
        "seed_detachment_proof": ROOT / "scripts/prove_pact_seed_replication_detachment.py",
        "seed_compactor_wrapper": ROOT / "scripts/compact_pact_seed_replication_storage.py",
        "frontend_supervisor_dependency": ROOT / "scripts/run_pact_frontend_screen_supervisor.py",
        "frontend_launcher_dependency": ROOT / "scripts/launch_pact_frontend_screen_detached.py",
        "frontend_detachment_dependency": ROOT / "scripts/prove_pact_frontend_screen_detachment.py",
        "throughput": ROOT / "scripts/measure_pact_frontend_screen_throughput.py",
        "generic_compactor": ROOT / "scripts/compact_pact_r2_storage.py",
        "live_evaluator_dependency": ROOT / "submodules/act/eval_pact_frontend_screen_row.py",
        "permuted_evaluator_dependency": ROOT / "submodules/act/eval_pact_valid_ablation_row.py",
    }
    document: dict[str, Any] = {
        "schema_version": "pact_seed_replication_dispatch_v1",
        "scientific_schedule": {
            "path": str(args.schedule.resolve()),
            "file_sha256": file_hash(args.schedule),
            "schedule_sha256": schedule["schedule_sha256"],
            "rows": 120,
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
            "max_control_steps": smoke["max_control_steps"],
            "token_plan_sha256": smoke["token_plan_sha256"],
            "required_artifact": "launch_smoke.json",
            "required_result_status": "complete",
            "full_dispatch_must_reconcile_without_rerun": True,
        },
        "detachment_proof": {
            "required_before_full_dispatch": True,
            "required_artifact": "detachment_proof.json",
            "kill_launching_shell_during_smoke": True,
            "endpoint_fields_inspected": False,
        },
        "throughput": {
            "required_measurement_elapsed_minutes": 20,
            "required_artifact": "throughput_first_20_minutes.json",
            "endpoint_fields_read": False,
        },
        "storage": {
            "lossless_compaction_required": True,
            "raw_smoke_schedule_index_preserved": 0,
            "raw_final_schedule_index_preserved": 119,
            "eligible_rows_losslessly_compacted": list(range(1, 119)),
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
            "sha256": file_hash(args.analysis_script),
            "primary_contrast": schedule["primary_contrast"],
            "decision_rule": schedule["decision_rule"],
            "unpooled_seeds_reported_first": True,
            "pooled_bootstrap_cluster": "instance_identity",
        },
        "frozen_inputs": {
            "training_summary": {
                "path": str(args.training_summary.resolve()),
                "sha256": file_hash(args.training_summary),
            },
            "preregistration": {
                "path": str(args.preregistration.resolve()),
                "sha256": file_hash(args.preregistration),
            },
            "token_plan": {
                "path": str(args.token_plan.resolve()),
                "sha256": file_hash(args.token_plan),
                "token_plan_sha256": token_sha,
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
