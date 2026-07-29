#!/usr/bin/env python3
"""Freeze the launch-smoke and input contract for confirmatory PACT evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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


def validate_self_hash(document: dict[str, Any], key: str) -> str:
    payload = dict(document)
    observed = payload.pop(key)
    if canonical_hash(payload) != observed:
        raise ValueError(f"{key} mismatch")
    return observed


def assert_no_outcomes(output_root: Path) -> None:
    if not output_root.exists():
        return
    forbidden_names = {
        "result.json",
        "initial_observation_accepted.json",
        "driver_result.json",
        "launch_smoke.json",
        "execution_summary.json",
    }
    observed = sorted(
        str(path)
        for path in output_root.rglob("*")
        if path.is_file() and path.name in forbidden_names
    )
    if observed:
        raise ValueError(
            "confirmatory output root already contains outcome/dispatch artifacts: "
            + ", ".join(observed[:5])
        )


def validate_schedule(
    schedule: dict[str, Any],
    training: dict[str, Any],
    manifest: dict[str, Any],
    machine: dict[str, Any],
) -> None:
    validate_self_hash(schedule, "schedule_sha256")
    if (
        schedule.get("schema_version") != "pact_confirmatory_schedule_v2"
        or schedule.get("instances") != 160
        or schedule.get("rollouts") != 960
        or schedule.get("workers") != 8
        or len(schedule.get("rows", [])) != 960
    ):
        raise ValueError("schedule does not match the frozen confirmatory design")
    design = machine["confirmatory_design"]
    expected_design = {
        "candidate_instances": schedule["instances"],
        "planned_total_rollouts": schedule["rollouts"],
        "worker_count": schedule["workers"],
        "checkpoint_seeds_per_arm": len(schedule["checkpoint_seeds"]),
        "repeats_per_instance_per_arm": schedule["repeats_per_instance_per_arm"],
    }
    for key, observed in expected_design.items():
        if design.get(key) != observed:
            raise ValueError(f"machine preregistration {key} mismatch")
    if not design.get("launch_smoke_required_before_dispatch"):
        raise ValueError("machine preregistration does not require launch smoke")
    if schedule["candidate_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("candidate manifest differs from schedule")
    if schedule["training_summary_sha256"] != canonical_hash(training):
        raise ValueError("training summary differs from schedule")
    rows = schedule["rows"]
    if [row["schedule_index"] for row in rows] != list(range(960)):
        raise ValueError("schedule indices are not contiguous")
    if len({row["rollout_id"] for row in rows}) != 960:
        raise ValueError("rollout IDs are not unique")
    if len({row["output_relpath"] for row in rows}) != 960:
        raise ValueError("output paths are not unique")
    expected_counts = {
        (arm, seed): 160
        for arm in ("ACT", "PACT", "PACT_ZERO")
        for seed in (3101, 3102)
    }
    if Counter((row["arm"], row["checkpoint_seed"]) for row in rows) != expected_counts:
        raise ValueError("arm/seed condition counts are not balanced")
    by_condition = {
        (record["arm"], int(record["seed"])): record
        for record in training["records"]
    }
    for row in rows:
        trained_arm = "ACT" if row["arm"] == "ACT" else "PACT"
        record = by_condition[(trained_arm, int(row["checkpoint_seed"]))]
        if (
            row["checkpoint_path"] != record["checkpoint"]
            or row["checkpoint_sha256"] != record["checkpoint_sha256"]
            or row["dataset_stats_path"] != record["dataset_stats"]
            or row["dataset_stats_sha256"] != record["dataset_stats_sha256"]
        ):
            raise ValueError("schedule row differs from frozen training record")


def verified_files(schedule: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
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
    result: dict[str, list[dict[str, str]]] = {}
    for label, paths in groups.items():
        result[label] = []
        for raw_path, expected in sorted(paths.items()):
            path = Path(raw_path)
            observed = file_hash(path)
            if observed != expected:
                raise ValueError(f"{label} hash mismatch for {path}")
            result[label].append({"path": str(path), "sha256": observed})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--environment-gate", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--machine-preregistration", required=True, type=Path)
    parser.add_argument("--analysis-script", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    schedule = json.loads(args.schedule.read_text())
    training = json.loads(args.training_summary.read_text())
    manifest = json.loads(args.manifest.read_text())
    gate = json.loads(args.environment_gate.read_text())
    machine = json.loads(args.machine_preregistration.read_text())
    validate_schedule(schedule, training, manifest, machine)
    if gate.get("decision") != "PACT_ENVIRONMENT_ADEQUATE":
        raise ValueError("confirmatory dispatch requires an adequate environment gate")
    if schedule["environment_gate_sha256"] != canonical_hash(gate):
        raise ValueError("environment gate differs from schedule")
    normalized_preregistration = " ".join(
        args.preregistration.read_text().lower().split()
    )
    required_phrases = (
        "960-rollout confirmatory schedule",
        "smoke row is part of the fixed schedule",
        "reconciled, not rerun",
    )
    for phrase in required_phrases:
        if phrase not in normalized_preregistration:
            raise ValueError(f"preregistration omits required phrase: {phrase}")
    output_root = args.output_root.resolve()
    assert_no_outcomes(output_root)
    verified = verified_files(schedule)
    smoke = schedule["rows"][0]
    document: dict[str, Any] = {
        "schema_version": "pact_confirmatory_dispatch_v2",
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
            "no_outcome_based_reruns": True,
            "pre_observation_infrastructure_retries_only": True,
            "confirmatory_outcomes_seen_before_freeze": False,
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
        "frozen_inputs": {
            "training_summary_path": str(args.training_summary.resolve()),
            "training_summary_sha256": file_hash(args.training_summary),
            "environment_gate_path": str(args.environment_gate.resolve()),
            "environment_gate_sha256": file_hash(args.environment_gate),
            "preregistration_path": str(args.preregistration.resolve()),
            "preregistration_sha256": file_hash(args.preregistration),
            "machine_preregistration_path": str(
                args.machine_preregistration.resolve()
            ),
            "machine_preregistration_sha256": file_hash(
                args.machine_preregistration
            ),
            "analysis_script_path": str(args.analysis_script.resolve()),
            "analysis_script_sha256": file_hash(args.analysis_script),
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
