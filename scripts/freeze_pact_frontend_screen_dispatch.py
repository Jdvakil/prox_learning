#!/usr/bin/env python3
"""Freeze all model, schedule, analysis, and durable screen runtime inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_self_hash(
    document: dict[str, Any], key: str, label: str
) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != canonical_hash(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return observed


def assert_empty(path: Path) -> None:
    if not path.exists():
        return
    files = [item for item in path.rglob("*") if item.is_file()]
    if files:
        raise ValueError(
            "screen output root is not empty before freeze: "
            + ", ".join(str(item) for item in files[:5])
        )


def verified_models(
    schedule: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, dict[str, str]] = {
        "checkpoints": {},
        "dataset_stats": {},
        "surface_encoders": {},
    }
    for row in schedule["rows"]:
        groups["checkpoints"][row["checkpoint_path"]] = row[
            "checkpoint_sha256"
        ]
        groups["dataset_stats"][row["dataset_stats_path"]] = row[
            "dataset_stats_sha256"
        ]
        if row["surface_encoder_path"] is not None:
            groups["surface_encoders"][
                row["surface_encoder_path"]
            ] = row["surface_encoder_sha256"]
    output: dict[str, list[dict[str, str]]] = {}
    for label, records in groups.items():
        output[label] = []
        for raw_path, expected in sorted(records.items()):
            path = Path(raw_path)
            observed = file_hash(path)
            if observed != expected:
                raise ValueError(
                    f"{label} hash mismatch for {path}"
                )
            output[label].append(
                {"path": str(path), "sha256": observed}
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--training-summary", required=True, type=Path
    )
    parser.add_argument("--encoder-report", required=True, type=Path)
    parser.add_argument(
        "--preregistration", required=True, type=Path
    )
    parser.add_argument(
        "--dataset-hash-amendment", required=True, type=Path
    )
    parser.add_argument("--analysis-script", required=True, type=Path)
    parser.add_argument("--supervisor", required=True, type=Path)
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument(
        "--detachment-proof-script", required=True, type=Path
    )
    parser.add_argument(
        "--throughput-script", required=True, type=Path
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    manifest = json.loads(args.manifest.read_text())
    training = json.loads(args.training_summary.read_text())
    encoder = json.loads(args.encoder_report.read_text())
    prereg = json.loads(args.preregistration.read_text())
    amendment = json.loads(
        args.dataset_hash_amendment.read_text()
    )
    validate_self_hash(schedule, "schedule_sha256", "schedule")
    prereg_hash = validate_self_hash(
        prereg, "preregistration_sha256", "preregistration"
    )
    amendment_hash = validate_self_hash(
        amendment, "amendment_sha256", "dataset amendment"
    )
    if (
        schedule.get("schema_version")
        != "pact_frontend_screen_schedule_v1"
        or schedule.get("instances") != 40
        or schedule.get("rollouts") != 120
        or schedule.get("workers") != 8
        or len(schedule.get("rows", [])) != 120
        or schedule.get("screen_not_confirmatory") is not True
    ):
        raise ValueError("schedule differs from screen design")
    if schedule["candidate_manifest_sha256"] != manifest[
        "manifest_sha256"
    ]:
        raise ValueError("manifest differs from schedule")
    if schedule["training_summary_sha256"] != canonical_hash(
        training
    ):
        raise ValueError("training summary differs from schedule")
    if schedule["encoder_report_sha256"] != canonical_hash(encoder):
        raise ValueError("encoder report differs from schedule")
    if schedule["preregistration_sha256"] != prereg_hash:
        raise ValueError("preregistration differs from schedule")
    if (
        schedule["dataset_hash_amendment_sha256"]
        != amendment_hash
    ):
        raise ValueError("dataset amendment differs from schedule")
    if file_hash(args.analysis_script) != prereg["analysis"][
        "frozen_analysis_script_sha256"
    ]:
        raise ValueError("analysis script differs from preregistration")
    if Counter(row["arm"] for row in schedule["rows"]) != {
        "ACT": 40,
        "PACT": 40,
        "PACT_ZERO": 40,
    }:
        raise ValueError("screen arms are not balanced")
    if [row["schedule_index"] for row in schedule["rows"]] != list(
        range(120)
    ):
        raise ValueError("screen schedule indices are not contiguous")
    if len({row["rollout_id"] for row in schedule["rows"]}) != 120:
        raise ValueError("screen rollout IDs are not unique")
    output_root = args.output_root.resolve()
    assert_empty(output_root)
    models = verified_models(schedule)
    runtime_paths = {
        "supervisor": args.supervisor,
        "launcher": args.launcher,
        "evaluator": args.evaluator,
        "detachment_proof_script": args.detachment_proof_script,
        "throughput_script": args.throughput_script,
    }
    smoke = schedule["rows"][0]
    document: dict[str, Any] = {
        "schema_version": "pact_frontend_screen_dispatch_v1",
        "screen_not_confirmatory": True,
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
            "screen_outcomes_seen_before_freeze": False,
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
            "instance_episode_id": smoke[
                "instance_episode_id"
            ],
            "schedule_row_sha256": smoke[
                "schedule_row_sha256"
            ],
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
            "endpoint_fields_inspected": False,
        },
        "throughput": {
            "required_measurement_elapsed_minutes": 20,
            "required_artifact": (
                "throughput_first_20_minutes.json"
            ),
            "source": (
                "completion ledger timestamps and row identities only"
            ),
            "endpoint_fields_read": False,
            "schedule_or_workers_changed": False,
        },
        "frozen_inputs": {
            "training_summary_path": str(
                args.training_summary.resolve()
            ),
            "training_summary_sha256": file_hash(
                args.training_summary
            ),
            "encoder_report_path": str(
                args.encoder_report.resolve()
            ),
            "encoder_report_sha256": file_hash(
                args.encoder_report
            ),
            "preregistration_path": str(
                args.preregistration.resolve()
            ),
            "preregistration_sha256": file_hash(
                args.preregistration
            ),
            "dataset_hash_amendment_path": str(
                args.dataset_hash_amendment.resolve()
            ),
            "dataset_hash_amendment_sha256": file_hash(
                args.dataset_hash_amendment
            ),
            "analysis_script_path": str(
                args.analysis_script.resolve()
            ),
            "analysis_script_sha256": file_hash(
                args.analysis_script
            ),
            "runtime": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": file_hash(path),
                }
                for name, path in runtime_paths.items()
            },
            **models,
        },
        "analysis": {
            "primary_contrast": "PACT_minus_PACT_ZERO",
            "secondary_contrast": "PACT_minus_ACT",
            "bootstrap_seed": schedule["bootstrap_seed"],
            "bootstrap_replicates": schedule[
                "bootstrap_replicates"
            ],
            "mcnemar_exact_two_sided": True,
            "fisher_exact_two_sided_secondary": True,
            "wilson_interval": 0.95,
            "paired_by_instance": True,
            "confirmatory_tokens_prohibited": prereg[
                "decision_rule"
            ]["confirmatory_tokens_prohibited"],
        },
        "storage": {
            "raw_smoke_schedule_index_preserved": 0,
            "raw_final_schedule_index_preserved": 119,
            "weights_and_rollout_payloads_not_committed": True,
        },
    }
    document["dispatch_contract_sha256"] = canonical_hash(
        document
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    print(document["dispatch_contract_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
