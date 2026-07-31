#!/usr/bin/env python3
"""Freeze 40 PACT_PERMUTED rows paired to the completed PACT screen arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


WORKERS = 8
INSTANCES = 40
ARM = "PACT_PERMUTED"
CHECKPOINT_SEED = 3101
BOOTSTRAP_SEED = 2026073106
BOOTSTRAP_REPLICATES = 20000


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
    document: dict[str, Any], key: str, label: str
) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != canonical_hash(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--screen-schedule", required=True, type=Path)
    parser.add_argument("--screen-output-root", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--token-plan", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--horizon-amendment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    screen = json.loads(args.screen_schedule.read_text())
    training = json.loads(args.training_summary.read_text())
    token_plan = json.loads(args.token_plan.read_text())
    prereg = json.loads(args.preregistration.read_text())
    horizon_amendment = json.loads(args.horizon_amendment.read_text())
    validate_self_hash(manifest, "manifest_sha256", "manifest")
    validate_self_hash(screen, "schedule_sha256", "screen schedule")
    token_plan_sha = validate_self_hash(
        token_plan, "token_plan_sha256", "token plan"
    )
    prereg_sha = validate_self_hash(
        prereg, "preregistration_sha256", "preregistration"
    )
    horizon_amendment_sha = validate_self_hash(
        horizon_amendment,
        "horizon_amendment_sha256",
        "horizon amendment",
    )
    if (
        token_plan["rows"] != INSTANCES
        or token_plan["max_control_steps"] != 900
        or token_plan["ablation"] != ARM
    ):
        raise ValueError("token plan design changed")
    if horizon_amendment["corrected_max_control_steps_per_row"] != 900:
        raise ValueError("horizon amendment changed")
    if prereg["design"] != {
        "arm": ARM,
        "instances": INSTANCES,
        "repeats_per_instance": 1,
        "rollouts": INSTANCES,
        "same_checkpoint_as_pact": True,
        "same_instances_as_screen": True,
        "screen_not_confirmatory": True,
        "workers": WORKERS,
    }:
        raise ValueError("preregistered valid-ablation design changed")
    checkpoint = Path(training["checkpoint"])
    stats = Path(training["dataset_stats"])
    encoder = Path(training["encoder"])
    for path, expected in (
        (checkpoint, training["checkpoint_sha256"]),
        (stats, training["dataset_stats_sha256"]),
        (encoder, training["encoder_sha256"]),
    ):
        if file_hash(path) != expected:
            raise ValueError(f"frozen model artifact changed: {path}")

    pact_screen_rows = {
        row["instance_episode_id"]: row
        for row in screen["rows"]
        if row["arm"] == "PACT"
    }
    instances = list(manifest["rows"])
    if len(instances) != INSTANCES or len(pact_screen_rows) != INSTANCES:
        raise ValueError("paired screen instance count changed")
    rows: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for index, instance in enumerate(instances):
        episode_id = instance["episode_id"]
        source = pact_screen_rows[episode_id]
        reference_dir = args.screen_output_root / source["output_relpath"]
        result_path = reference_dir / "result.json"
        driver_path = reference_dir / "driver_result.json"
        if not result_path.exists() or not driver_path.exists():
            raise ValueError("completed PACT reference row is absent")
        references.append(
            {
                "instance_episode_id": episode_id,
                "screen_schedule_index": source["schedule_index"],
                "screen_rollout_id": source["rollout_id"],
                "checkpoint_sha256": source["checkpoint_sha256"],
                "result_path": str(result_path.resolve()),
                "result_sha256": file_hash(result_path),
                "driver_path": str(driver_path.resolve()),
                "driver_sha256": file_hash(driver_path),
            }
        )
        identity = {
            "schedule_schema": "pact_valid_ablation_schedule_v1",
            "instance_episode_id": episode_id,
            "arm": ARM,
            "checkpoint_seed": CHECKPOINT_SEED,
            "token_plan_sha256": token_plan_sha,
            "token_plan_row": index,
        }
        rollout_id = canonical_hash(identity)
        row: dict[str, Any] = {
            "schedule_index": index,
            "instance_role_index": int(instance["role_index"]),
            "instance_episode_id": episode_id,
            "instance_row_sha256": instance["row_sha256"],
            "intrusion_side": instance["intrusion_side"],
            "arm": ARM,
            "checkpoint_seed": CHECKPOINT_SEED,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": training["checkpoint_sha256"],
            "dataset_stats_path": str(stats),
            "dataset_stats_sha256": training["dataset_stats_sha256"],
            "surface_encoder_path": str(encoder),
            "surface_encoder_sha256": training["encoder_sha256"],
            "proximity_feature_dim": 32,
            "token_plan_manifest_path": str(args.token_plan.resolve()),
            "token_plan_sha256": token_plan_sha,
            "token_plan_row": index,
            "rollout_id": rollout_id,
            "output_relpath": (
                f"rows/{index:03d}_{rollout_id[:16]}_pact_permuted"
            ),
        }
        row["schedule_row_sha256"] = canonical_hash(row)
        rows.append(row)
    document: dict[str, Any] = {
        "schema_version": "pact_valid_ablation_schedule_v1",
        "screen_not_confirmatory": True,
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "source_screen_schedule_sha256": screen["schedule_sha256"],
        "source_screen_schedule_file_sha256": file_hash(
            args.screen_schedule
        ),
        "training_summary_sha256": canonical_hash(training),
        "token_plan_sha256": token_plan_sha,
        "token_plan_file_sha256": file_hash(args.token_plan),
        "preregistration_sha256": prereg_sha,
        "horizon_amendment_sha256": horizon_amendment_sha,
        "instances": INSTANCES,
        "arms": [ARM],
        "checkpoint_seeds": [CHECKPOINT_SEED],
        "repeats_per_instance_per_arm": 1,
        "rollouts": INSTANCES,
        "workers": WORKERS,
        "fresh_subprocess_per_rollout": True,
        "no_outcome_based_row_replacement": True,
        "indiscriminate_all_inflight_recovery": True,
        "primary_endpoint": (
            "task_success and hazard_bar contacts == 0 and "
            "other_environment contacts == 0"
        ),
        "primary_contrast": "PACT_minus_PACT_PERMUTED",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "decision_rule": prereg["decision_rule"],
        "paired_pact_reference": references,
        "rows": rows,
    }
    document["schedule_sha256"] = canonical_hash(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    print(document["schedule_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
