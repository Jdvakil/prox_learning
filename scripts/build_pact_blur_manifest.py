#!/usr/bin/env python3
"""Freeze 25 paired C0 instances and the calibrated RGB blur grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pact_blur_sweep_contract import (
    BLUR_SIGMAS,
    ENVIRONMENT_VERSION,
    INSTANCE_COUNT,
    SCHEMA_VERSION,
    sha256_file,
    sha256_payload,
    validate_manifest,
)
from pact_geometry_generalization_v3_contract import (
    load_manifest as load_v3_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = ROOT / "diagnostics_output/pact_blur_sweep/calibration.json"
DEFAULT_V3_MANIFEST = ROOT / "configs/pact_geometry_generalization_v3.json"
DEFAULT_POLICY_REGISTRY = ROOT / "diagnostics_output/pact_contact_endpoint/policy_training.json"
DEFAULT_TOKEN_PLAN = ROOT / "diagnostics_output/pact_contact_endpoint/token_plan.json"
DEFAULT_OUTPUT = ROOT / "configs/pact_blur_sweep_v1.json"

PROTECTED_PATHS = (
    ROOT / "docs/PACT_CONTACT_ENDPOINT_DECISION.md",
    ROOT / "diagnostics_output/pact_contact_endpoint/analysis.json",
    ROOT / "diagnostics_output/pact_contact_endpoint/final_decision.json",
    ROOT / "docs/PACT_GEOMETRY_GENERALIZATION_V3.md",
    ROOT / "diagnostics_output/pact_geometry_generalization_v3/analysis.json",
    ROOT / "diagnostics_output/pact_geometry_generalization_v3/final_decision.json",
)
MODEL_CONSTRUCTION_PATHS = (
    ROOT / "submodules/act/detr/models/detr_vae.py",
    ROOT / "submodules/act/detr/main.py",
    ROOT / "submodules/act/policy.py",
)


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != sha256_payload(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return str(observed)


def select_source_rows(v3_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    c0 = [row for row in v3_manifest["rows"] if row["condition_id"] == "C0"]
    if len(c0) != 40:
        raise ValueError("source v3 manifest does not contain 40 C0 rows")
    ranked = sorted(
        c0,
        key=lambda row: sha256_payload(
            {
                "study": SCHEMA_VERSION,
                "episode_id": row["episode_id"],
            }
        ),
    )
    by_side = {
        side: [row for row in ranked if row["intrusion_side"] == side]
        for side in ("left", "right")
    }
    selected = by_side["left"][:12] + by_side["right"][:13]
    return sorted(
        selected,
        key=lambda row: sha256_payload(
            {"selected_blur_instance": row["episode_id"]}
        ),
    )


def build(
    *,
    calibration: dict[str, Any],
    calibration_path: Path,
    v3_manifest: dict[str, Any],
    v3_manifest_path: Path,
    registry: dict[str, Any],
    registry_path: Path,
    token_plan: dict[str, Any],
    token_plan_path: Path,
) -> dict[str, Any]:
    calibration_sha = validate_self_hash(
        calibration, "calibration_sha256", "blur calibration"
    )
    registry_sha = validate_self_hash(
        registry, "policy_registry_sha256", "policy registry"
    )
    token_sha = validate_self_hash(token_plan, "token_plan_sha256", "token plan")
    if calibration.get("candidate_sigmas") != [0.0, 0.25, 0.5, 1.0, 2.0]:
        raise ValueError("calibration candidate grid changed")
    retained = {
        float(row["sigma"]): float(row["retained_fraction_of_sharp"])
        for row in calibration["measurements"]
    }
    transition = [sigma for sigma in BLUR_SIGMAS if 0.05 < retained[sigma] < 0.8]
    if transition != [0.5, 1.0]:
        raise ValueError(f"calibrated transition points changed: {transition}")
    if calibration["primitive"]["sigma_zero_same_object_and_bit_identical"] is not True:
        raise ValueError("sigma zero did not calibrate as an exact identity")

    selected = select_source_rows(v3_manifest)
    rows = []
    for index, source in enumerate(selected):
        row = {
            **{
                key: source[key]
                for key in (
                    "master_seed",
                    "candidate_index",
                    "episode_id",
                    "scene_template_id",
                    "scene_template_house_index",
                    "intrusion_side",
                    "panel_x_jitter_m",
                    "panel_face_jitter_m",
                    "realized_geometry",
                    "task_seed_u32",
                    "task_seed_u64",
                    "max_sampling_retries",
                    "task_sampler_class",
                )
            },
            "schema_version": SCHEMA_VERSION,
            "environment_version": ENVIRONMENT_VERSION,
            "role": "blur_policy_eval",
            "role_index": index,
            "blur_role_index": index,
            "instance_index": index,
            "instance_cluster_id": f"blur_policy_eval:{index:02d}",
            "source_condition_id": "C0",
            "source_v3_instance_index": source["instance_index"],
            "source_v3_instance_cluster_id": source["instance_cluster_id"],
            "source_v3_row_sha256": source["row_sha256"],
        }
        row["row_sha256"] = sha256_payload(row)
        rows.append(row)

    checkpoints = []
    for seed in (3101, 3102, 3103):
        for arm in ("ACT", "PACT"):
            record = registry["seeds"][str(seed)][arm]
            if sha256_file(record["checkpoint_path"]) != record["checkpoint_sha256"]:
                raise ValueError(f"checkpoint changed: {seed} {arm}")
            if sha256_file(record["dataset_stats_path"]) != record["dataset_stats_sha256"]:
                raise ValueError(f"dataset stats changed: {seed} {arm}")
            checkpoints.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "path": record["checkpoint_path"],
                    "sha256": record["checkpoint_sha256"],
                }
            )
    if sha256_file(token_plan["files"]["tokens"]["path"]) != token_plan["files"]["tokens"]["sha256"]:
        raise ValueError("permuted token tensor changed")
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "instance_count": INSTANCE_COUNT,
        "blur_sigmas": BLUR_SIGMAS,
        "calibration_binding": {
            "path": str(calibration_path.resolve()),
            "calibration_sha256": calibration_sha,
            "selected_sigmas": BLUR_SIGMAS,
            "transition_sigmas": transition,
            "retained_fraction_by_sigma": {
                str(sigma): retained[sigma] for sigma in BLUR_SIGMAS
            },
            "selection_used_policy_outcomes": False,
            "rationale": (
                "sigma 0.5 and 1.0 land in the predeclared 5%-80% transition; "
                "sigma 2.0 measures the near-floor regime; sigma 0 is exact identity"
            ),
        },
        "source_v3": {
            "manifest_path": str(v3_manifest_path.resolve()),
            "manifest_sha256": v3_manifest["manifest_sha256"],
            "manifest_file_sha256": sha256_file(v3_manifest_path),
            "condition_id": "C0",
            "selection_rule": (
                "hash-rank all 40 C0 rows by study and episode; take first 12 "
                "left and first 13 right; re-rank selected rows by episode hash"
            ),
            "source_policy_outcomes_read": False,
        },
        "planned_design": {
            "arms": ["ACT", "PACT", "PACT_PERMUTED"],
            "checkpoint_seeds": [3101, 3102, 3103],
            "instances_shared_across_every_sigma_arm_seed": True,
            "instances": INSTANCE_COUNT,
            "sigmas": len(BLUR_SIGMAS),
            "rollouts": 900,
            "workers": 12,
            "no_retraining": True,
        },
        "sensor_names": v3_manifest["sensor_names"],
        "sensor_order_sha256": v3_manifest["sensor_order_sha256"],
        "frozen_artifacts": {
            "policy_registry": {
                "path": str(registry_path.resolve()),
                "file_sha256": sha256_file(registry_path),
                "policy_registry_sha256": registry_sha,
            },
            "token_plan": {
                "path": str(token_plan_path.resolve()),
                "file_sha256": sha256_file(token_plan_path),
                "token_plan_sha256": token_sha,
                "global_tensor": token_plan["files"]["tokens"],
            },
            "checkpoints": checkpoints,
            "surface_encoder": {
                "path": v3_manifest["frozen_policy_artifacts"]["encoder_path"],
                "sha256": v3_manifest["frozen_policy_artifacts"]["encoder_sha256"],
            },
        },
        "no_upstream_merge": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in MODEL_CONSTRUCTION_PATHS
        },
        "protected_scientific_artifacts": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in PROTECTED_PATHS
        },
        "rows": rows,
    }
    document["manifest_sha256"] = sha256_payload(document)
    validate_manifest(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--v3-manifest", type=Path, default=DEFAULT_V3_MANIFEST)
    parser.add_argument("--policy-registry", type=Path, default=DEFAULT_POLICY_REGISTRY)
    parser.add_argument("--token-plan", type=Path, default=DEFAULT_TOKEN_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace blur manifest: {args.output}")
    calibration = json.loads(args.calibration.read_text())
    v3_manifest = load_v3_manifest(args.v3_manifest)
    registry = json.loads(args.policy_registry.read_text())
    token_plan = json.loads(args.token_plan.read_text())
    document = build(
        calibration=calibration,
        calibration_path=args.calibration,
        v3_manifest=v3_manifest,
        v3_manifest_path=args.v3_manifest,
        registry=registry,
        registry_path=args.policy_registry,
        token_plan=token_plan,
        token_plan_path=args.token_plan,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
