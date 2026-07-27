#!/usr/bin/env python3
"""Freeze the threshold manifest and the live-development log schema.

Handoff steps 11-13. The manifest pins every artifact a live evaluator would have to agree
with before it may run, and the loader rejects any mismatch rather than warning about it.

This task stopped at the step-9 reused-diagnostic checks, so no live rollout was authorized.
The manifest is written anyway -- it is the record of exactly what was fitted, and the next
task should not have to re-derive it -- but it carries ``authorized_for_live: false`` and the
loader refuses to hand it to an evaluator while that flag is false. A manifest that looked
ready would be worse than no manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin import threshold as thr

# Per-frame fields a live development rollout must record (handoff step 13).
LIVE_LOG_SCHEMA = (
    "episode_id", "repeat_index", "timestep", "qpos", "qvel",
    "nominal_act_action", "aggregated_act_action", "current_field_sha256",
    "predicted_parked_field_sha256", "activity_probability", "selected_threshold",
    "activation_decision", "ungated_differential", "magnitude_capped_differential",
    "privileged_oracle_differential", "deployable_oracle_cosine",
    "deployable_oracle_norm_ratio", "false_active_sequence_length",
    "filtered_correction", "accumulated_correction", "clipping_saturation",
    "executed_action", "gripper_command", "contact_classes", "penetration",
    "task_success", "termination_reason", "artifact_hashes",
)

# Deliberately excluded: mj_geomDistance on robot_0/fr3_link7_collision returns exactly
# 0.0, which pins minimum_clearance_m at <= 0 regardless of the true geometry. It has been
# excluded from every evidence chain in this project and stays excluded.
EXCLUDED_LOG_FIELDS = ("minimum_clearance_m",)


class ThresholdManifestError(RuntimeError):
    """The manifest disagrees with the artifacts on disk, or is not live-authorized."""


def sha256_file(path) -> str | None:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def load_threshold_manifest(path, *, require_live_authorization: bool = True) -> dict:
    """Strict loader. Every pinned hash is recomputed; any mismatch raises."""
    manifest = json.loads(Path(path).read_text())

    recorded = manifest.pop("manifest_sha256", None)
    if thr.canonical_hash(manifest) != recorded:
        manifest["manifest_sha256"] = recorded
        raise ThresholdManifestError("manifest self-hash mismatch")
    manifest["manifest_sha256"] = recorded

    checkpoint = Path(manifest["model"]["checkpoint_path"])
    if sha256_file(checkpoint) != manifest["model"]["checkpoint_sha256"]:
        raise ThresholdManifestError(f"checkpoint hash mismatch: {checkpoint}")

    safety_dir = ROOT / manifest["safety_head"]["directory"]
    for name, expected in manifest["safety_head"]["hashes"].items():
        if sha256_file(safety_dir / name) != expected:
            raise ThresholdManifestError(f"SafetyHead hash mismatch: {name}")

    if manifest["model"]["variant"] != "CURRENT_FRAME_ONLY":
        raise ThresholdManifestError("only CURRENT_FRAME_ONLY may be deployed")
    if manifest["model"]["seed"] != 0:
        raise ThresholdManifestError("only seed 0 may be deployed")
    if manifest["model"]["history_frames"] != 1:
        raise ThresholdManifestError("temporal history is retired; history_frames must be 1")

    if require_live_authorization and not manifest.get("authorized_for_live", False):
        raise ThresholdManifestError(
            "manifest is not authorized for live execution: "
            f"{manifest.get('authorization_blocker', 'unspecified')}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provenance", required=True, type=Path)
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--calibration-manifest", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import numpy
    import torch

    provenance = json.loads(args.provenance.read_text())
    calibration = json.loads(args.calibration.read_text())
    audit = json.loads(args.audit.read_text())
    calibration16 = json.loads(args.calibration_manifest.read_text())
    stack = json.loads(args.stack.read_text())

    frozen = provenance["frozen_model"]
    authorized = bool(audit["blocking_checks"]["passed"]
                      and audit["inference_stability"]["stable"]
                      and calibration.get("feasible"))
    blocker = None
    if not authorized:
        blocker = ("reused-diagnostic blocking checks failed: "
                   + "; ".join(audit["blocking_checks"]["failures"])
                   if audit["blocking_checks"]["failures"] else "calibration infeasible")

    manifest = {
        "schema": "hybrid_obstacle_reference_threshold_manifest_v1",
        "authorized_for_live": authorized,
        "authorization_blocker": blocker,
        "model": {
            "reference_id": "CAUSAL_PARKED_SKIN_REFERENCE_V1",
            "variant": "CURRENT_FRAME_ONLY",
            "seed": 0,
            "history_frames": 1,
            "checkpoint_path": frozen["checkpoint_path"],
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "model_config_hash": frozen["model_config_hash"],
            "parameter_count": frozen["parameter_count"],
            "feature_contract_sha256": next(
                c["observed"] for c in provenance["checks"]
                if c["check"] == "input_contract_sha256"),
        },
        "activation": {
            "selected_threshold": calibration["selected"]["threshold"],
            "gate_variable": "frame activity probability",
            "activity_definition": calibration["activity_definition"],
            "gate_rule": calibration["gate_rule"],
            "gates_on_differential_norm_alone": False,
            "retired_threshold": calibration["retired_threshold"],
        },
        "magnitude_support": {
            "construction": ("changed_probability = sigmoid(mask_logits); "
                             "delta_magnitude = current_closeness * sigmoid(magnitude_logits); "
                             "predicted_delta = changed_probability * delta_magnitude; "
                             "predicted_parked = current_closeness - predicted_delta"),
            "guarantee": "0 <= predicted_parked <= current_closeness <= 1 by construction",
            "post_hoc_clipping": False,
            "recalculated_in_this_task": False,
            "source": "frozen at model qualification; carried forward unchanged",
        },
        "calibration": {
            "set": "threshold_calibration16",
            "manifest_sha256": calibration16["manifest_sha256"],
            "episodes": calibration16["episodes"],
            "episode_count": calibration16["episode_count"],
            "trajectory_count": calibration16["trajectory_count"],
            "total_frames": calibration16["total_frames"],
            "cluster_unit": calibration16["cluster_unit"],
            "bootstrap_seed": thr.BOOTSTRAP_SEED,
            "bootstrap_replicates": thr.BOOTSTRAP_REPLICATES,
            "bootstrap_upper_fpr_bound": calibration["selected"]["bootstrap_upper_fpr"],
            "median_active_recall": calibration["selected"]["median_active_recall"],
            "feasibility_contract": calibration["feasibility_contract"],
        },
        "safety_head": {
            "directory": "assets/safety/cvae_v3",
            "hashes": {p.name: sha256_file(p) for p in sorted(args.safety_dir.glob("*"))
                       if p.is_file() and p.suffix in (".pt", ".json")},
            "label_scale": 11.359346389770508,
            "frozen": True,
        },
        "act": {
            "best_epoch": 1738,
            "policy_best_sha256": next(
                c["observed"] for c in provenance["checks"]
                if c["check"] == "act_policy_best_sha256_expected"),
            "dataset_stats_sha256": next(
                c["observed"] for c in provenance["checks"]
                if c["check"] == "act_dataset_stats_sha256_expected"),
            "temporal_aggregation": "enabled; residual applies after aggregation",
        },
        "controller": {
            **{k: stack["residual_controller"][k] for k in
               ("gain", "decay_per_second", "ema", "max_deviation_rad_per_joint",
                "arm_only", "gripper_owner")},
            "dt_seconds": 0.066,
            "changed_in_this_task": False,
        },
        "sensor_order_sha256": stack["sensor_contract"]["sensor_order_hash"],
        "sensor_count": 40,
        "offsamples": 4,
        "live_schedule": {
            "manifest": "configs/hybrid_obstacle_controller_development4_v1.json",
            "rows": [106, 107, 108, 118],
            "repeats_per_row": 5,
            "total_rollouts": 20,
            "executed": 0,
            "confirmatory41_permitted": False,
        },
        "log_schema": list(LIVE_LOG_SCHEMA),
        "excluded_log_fields": list(EXCLUDED_LOG_FIELDS),
        "excluded_field_reason": (
            "mj_geomDistance returns exactly 0.0 for robot_0/fr3_link7_collision, pinning "
            "minimum_clearance_m at <= 0 regardless of true geometry"),
        "commits": {
            "root": git("rev-parse", "HEAD"),
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "act": git("rev-parse", "HEAD", repo=ROOT / "submodules" / "act"),
            "molmospaces": git("rev-parse", "HEAD",
                               repo=ROOT / "submodules" / "molmospaces"),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
        },
    }
    manifest["manifest_sha256"] = thr.canonical_hash(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"threshold          : {manifest['activation']['selected_threshold']:.8f}")
    print(f"authorized for live: {authorized}")
    if blocker:
        print(f"  blocker          : {blocker}")
    print(f"manifest sha256    : {manifest['manifest_sha256']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
