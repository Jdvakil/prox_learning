#!/usr/bin/env python3
"""Freeze schedule, model, sampler, analysis, and runtime inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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


def validate_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != canonical_hash(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--analysis-script", required=True, type=Path)
    parser.add_argument("--token-plan", required=True, type=Path)
    parser.add_argument("--supervisor", required=True, type=Path)
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--detachment-proof", required=True, type=Path)
    parser.add_argument("--throughput-script", required=True, type=Path)
    parser.add_argument("--compactor", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    manifest = json.loads(args.manifest.read_text())
    prereg = json.loads(args.preregistration.read_text())
    token_plan = json.loads(args.token_plan.read_text())
    validate_hash(schedule, "schedule_sha256", "schedule")
    validate_hash(manifest, "manifest_sha256", "manifest")
    validate_hash(prereg, "preregistration_sha256", "preregistration")
    validate_hash(token_plan, "token_plan_sha256", "token plan")
    if (
        schedule["schema_version"] != "pact_valid_ablation_schedule_v1"
        or schedule["rollouts"] != 40
        or schedule["workers"] != 8
        or len(schedule["rows"]) != 40
        or {row["arm"] for row in schedule["rows"]}
        != {"PACT_PERMUTED"}
        or [row["schedule_index"] for row in schedule["rows"]]
        != list(range(40))
    ):
        raise ValueError("valid-ablation schedule changed")
    if schedule["candidate_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("schedule manifest changed")
    if schedule["token_plan_sha256"] != token_plan["token_plan_sha256"]:
        raise ValueError("schedule token plan changed")
    if file_hash(args.analysis_script) != prereg["analysis_script_sha256"]:
        raise ValueError("analysis script changed after preregistration")
    if args.output_root.exists() and any(args.output_root.rglob("*")):
        raise ValueError("valid-ablation output root is not empty")
    first = schedule["rows"][0]
    model_records = []
    for label, path_key, hash_key in (
        ("checkpoint", "checkpoint_path", "checkpoint_sha256"),
        ("dataset_stats", "dataset_stats_path", "dataset_stats_sha256"),
        ("surface_encoder", "surface_encoder_path", "surface_encoder_sha256"),
    ):
        path = Path(first[path_key])
        observed = file_hash(path)
        if observed != first[hash_key]:
            raise ValueError(f"{label} changed")
        model_records.append(
            {"label": label, "path": str(path), "sha256": observed}
        )
    runtime = {
        label: {"path": str(path.resolve()), "sha256": file_hash(path)}
        for label, path in (
            ("supervisor", args.supervisor),
            ("launcher", args.launcher),
            ("evaluator", args.evaluator),
            ("detachment_proof", args.detachment_proof),
            ("throughput", args.throughput_script),
            ("compactor", args.compactor),
        )
    }
    document: dict[str, Any] = {
        "schema_version": "pact_valid_ablation_dispatch_v1",
        "screen_not_confirmatory": True,
        "scientific_schedule": {
            "path": str(args.schedule.resolve()),
            "file_sha256": file_hash(args.schedule),
            "schedule_sha256": schedule["schedule_sha256"],
            "rows": 40,
            "workers": 8,
            "rows_changed": 0,
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": file_hash(args.manifest),
        },
        "execution": {
            "output_root": str(args.output_root.resolve()),
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
            "schedule_index": first["schedule_index"],
            "rollout_id": first["rollout_id"],
            "instance_episode_id": first["instance_episode_id"],
            "schedule_row_sha256": first["schedule_row_sha256"],
            "output_relpath": first["output_relpath"],
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
            "raw_smoke_schedule_index_preserved": 0,
            "raw_final_schedule_index_preserved": 39,
            "eligible_rows_losslessly_compacted": list(range(1, 39)),
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
            "primary_contrast": "PACT_minus_PACT_PERMUTED",
            "decision_rule": schedule["decision_rule"],
        },
        "frozen_inputs": {
            "models": model_records,
            "token_plan": {
                "path": str(args.token_plan.resolve()),
                "sha256": file_hash(args.token_plan),
                "token_plan_sha256": token_plan["token_plan_sha256"],
            },
            "runtime": runtime,
        },
    }
    document["dispatch_contract_sha256"] = canonical_hash(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    print(document["dispatch_contract_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
