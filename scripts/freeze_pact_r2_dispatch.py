#!/usr/bin/env python3
"""Freeze the complete scientific and durable-runtime contract for PACT R2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_ANALYZER_SHA256 = (
    "fd3c7f2e91a1737e248fc3ebe803018dcb4f9455d2b4e413d56946a4aebe25be"
)
EXPECTED_AUTHORIZATION_SHA256 = (
    "0f6c44eadbdfbb799041f3aa9a0809db80a7c615a3b04bb300be92432bbe1300"
)


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


def validate_self_hash(
    document: dict[str, Any], key: str, *, label: str
) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != canonical_hash(payload):
        raise ValueError(f"{label} {key} mismatch")
    return observed


def assert_clean_output_root(output_root: Path) -> None:
    if not output_root.exists():
        return
    observed = sorted(
        str(path)
        for path in output_root.rglob("*")
        if path.is_file()
    )
    if observed:
        raise ValueError(
            "R2 output root is not empty before dispatch freeze: "
            + ", ".join(observed[:5])
        )


def validate_schedule(
    schedule: dict[str, Any],
    *,
    manifest: dict[str, Any],
    training: dict[str, Any],
    gate: dict[str, Any],
    preregistration: dict[str, Any],
) -> None:
    validate_self_hash(schedule, "schedule_sha256", label="schedule")
    if (
        schedule.get("schema_version")
        != "pact_confirmatory_r2_schedule_v1"
        or schedule.get("instances") != 160
        or schedule.get("rollouts") != 960
        or schedule.get("workers") != 8
        or schedule.get("repeats_per_instance_per_arm") != 2
        or len(schedule.get("rows", [])) != 960
        or schedule.get("indiscriminate_all_inflight_recovery") is not True
    ):
        raise ValueError("schedule differs from the preregistered R2 design")
    if schedule["candidate_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("R2 manifest differs from schedule")
    if schedule["training_summary_sha256"] != canonical_hash(training):
        raise ValueError("training summary differs from schedule")
    if schedule["environment_gate_sha256"] != canonical_hash(gate):
        raise ValueError("environment gate differs from schedule")
    if (
        schedule["r2_preregistration_sha256"]
        != preregistration["preregistration_sha256"]
    ):
        raise ValueError("R2 preregistration differs from schedule")
    if (
        schedule.get("r1_schedule_quarantined") is not True
        or schedule.get("r1_endpoint_loaded") is not False
    ):
        raise ValueError("R1 quarantine is not explicit in schedule")
    rows = schedule["rows"]
    if [row["schedule_index"] for row in rows] != list(range(960)):
        raise ValueError("R2 schedule indices are not contiguous")
    if len({row["rollout_id"] for row in rows}) != 960:
        raise ValueError("R2 rollout IDs are not unique")
    if len({row["output_relpath"] for row in rows}) != 960:
        raise ValueError("R2 output paths are not unique")
    expected = {
        (arm, seed): 160
        for arm in ("ACT", "PACT", "PACT_ZERO")
        for seed in (3101, 3102)
    }
    if Counter((row["arm"], row["checkpoint_seed"]) for row in rows) != expected:
        raise ValueError("R2 arm/seed cells are not balanced")


def verified_model_files(
    schedule: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
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
    verified: dict[str, list[dict[str, str]]] = {}
    for label, records in groups.items():
        verified[label] = []
        for raw_path, expected in sorted(records.items()):
            path = Path(raw_path)
            observed = file_hash(path)
            if observed != expected:
                raise ValueError(f"{label} hash mismatch for {path}")
            verified[label].append({"path": str(path), "sha256": observed})
    return verified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--environment-gate", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--analysis-script", required=True, type=Path)
    parser.add_argument("--supervisor", required=True, type=Path)
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--detachment-proof-script", required=True, type=Path)
    parser.add_argument("--throughput-script", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    schedule = json.loads(args.schedule.read_text())
    manifest = json.loads(args.manifest.read_text())
    training = json.loads(args.training_summary.read_text())
    gate = json.loads(args.environment_gate.read_text())
    preregistration = json.loads(args.preregistration.read_text())
    validate_self_hash(
        preregistration, "preregistration_sha256", label="preregistration"
    )
    validate_schedule(
        schedule,
        manifest=manifest,
        training=training,
        gate=gate,
        preregistration=preregistration,
    )
    if gate.get("decision") != "PACT_ENVIRONMENT_ADEQUATE":
        raise ValueError("R2 dispatch requires the carried adequate gate")
    if manifest["r1_quarantine"]["overlap_episode_ids"]:
        raise ValueError("R2 manifest overlaps quarantined R1 episode IDs")
    if manifest["r1_quarantine"].get("r1_endpoint_loaded") is not False:
        raise ValueError("R2 manifest does not preserve R1 endpoint quarantine")
    if file_hash(args.authorization) != EXPECTED_AUTHORIZATION_SHA256:
        raise ValueError("R2 authorization SHA-256 mismatch")
    if file_hash(args.analysis_script) != EXPECTED_ANALYZER_SHA256:
        raise ValueError("frozen analysis SHA-256 mismatch")
    output_root = args.output_root.resolve()
    assert_clean_output_root(output_root)
    verified = verified_model_files(schedule)
    smoke = schedule["rows"][0]
    runtime_paths = {
        "supervisor": args.supervisor,
        "launcher": args.launcher,
        "evaluator": args.evaluator,
        "detachment_proof_script": args.detachment_proof_script,
        "throughput_script": args.throughput_script,
    }
    document: dict[str, Any] = {
        "schema_version": "pact_r2_dispatch_v1",
        "scientific_schedule": {
            "path": str(args.schedule.resolve()),
            "file_sha256": file_hash(args.schedule),
            "schedule_sha256": schedule["schedule_sha256"],
            "rows": 960,
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
            "confirmatory_outcomes_seen_before_freeze": False,
            "r1_endpoint_loaded": False,
            "detached_stdin": True,
            "setsid": True,
            "nohup": True,
        },
        "boundary_amendment": {
            "row_terminal_boundary": "valid scientific result.json",
            "all_inflight_rows_rerun": True,
            "individual_post_observation_retry": False,
            "pre_observation_retry": True,
            "cohort_exit_window_seconds": 5,
            "recovery_event_frozen_before_rerun": True,
        },
        "launch_smoke": {
            "required_before_full_dispatch": True,
            "schedule_index": smoke["schedule_index"],
            "rollout_id": smoke["rollout_id"],
            "instance_episode_id": smoke["instance_episode_id"],
            "schedule_row_sha256": smoke["schedule_row_sha256"],
            "output_relpath": smoke["output_relpath"],
            "required_artifact": "launch_smoke.json",
            "required_result_status": "complete",
            "full_dispatch_must_reconcile_without_rerun": True,
        },
        "detachment_proof": {
            "required_before_full_dispatch": True,
            "required_artifact": "detachment_proof.json",
            "kill_launching_shell_during_smoke": True,
            "heartbeat_must_advance_after_shell_death": True,
            "supervisor_and_evaluator_must_survive_or_complete": True,
            "smoke_result_count": 1,
            "endpoint_fields_inspected": False,
        },
        "throughput": {
            "required_measurement_elapsed_minutes": 20,
            "required_artifact": "throughput_first_20_minutes.json",
            "source": "completion ledger timestamps and row identities only",
            "endpoint_fields_read": False,
            "schedule_or_workers_changed": False,
        },
        "frozen_inputs": {
            "training_summary_path": str(args.training_summary.resolve()),
            "training_summary_sha256": file_hash(args.training_summary),
            "environment_gate_path": str(args.environment_gate.resolve()),
            "environment_gate_sha256": file_hash(args.environment_gate),
            "r2_preregistration_path": str(args.preregistration.resolve()),
            "r2_preregistration_sha256": file_hash(args.preregistration),
            "authorization_path": str(args.authorization.resolve()),
            "authorization_sha256": file_hash(args.authorization),
            "analysis_script_path": str(args.analysis_script.resolve()),
            "analysis_script_sha256": file_hash(args.analysis_script),
            "runtime": {
                name: {"path": str(path.resolve()), "sha256": file_hash(path)}
                for name, path in runtime_paths.items()
            },
            **verified,
        },
        "analysis": {
            "primary_endpoint": schedule["primary_endpoint"],
            "detectable_effect_statement": schedule[
                "detectable_effect_statement"
            ],
            "detectable_absolute_increase": schedule[
                "detectable_absolute_increase"
            ],
            "bootstrap_seed": schedule["bootstrap_seed"],
            "bootstrap_replicates": schedule["bootstrap_replicates"],
            "fisher_exact_two_sided": ["PACT_vs_ACT", "PACT_vs_PACT_ZERO"],
            "wilson_interval": 0.95,
            "instance_clustered": True,
        },
    }
    document["dispatch_contract_sha256"] = canonical_hash(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["dispatch_contract_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
