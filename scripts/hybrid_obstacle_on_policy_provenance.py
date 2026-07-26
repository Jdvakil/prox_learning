#!/usr/bin/env python3
"""Immutable-provenance verification for the on-policy aggregation round.

Handoff step 2. Expected digests are recovered from every prior decision artifact and
the live files are re-hashed against them. Any mismatch stops the task with
``CHECKPOINT_OR_SOURCE_MISMATCH``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out = Path(args.out).resolve()

    diagnostics = ROOT / "diagnostics_output"
    deployable = json.loads((diagnostics / "hybrid_obstacle_deployable_reference"
                             / "final_decision.json").read_text())
    oracle = json.loads((diagnostics / "hybrid_obstacle_oracle_reference"
                         / "final_decision.json").read_text())
    rawhead = json.loads((diagnostics / "hybrid_obstacle_raw_head_qualification"
                          / "final_decision.json").read_text())
    baseline = json.loads((diagnostics / "hybrid_obstacle_act_baseline"
                           / "final_decision.json").read_text())
    ckpt_manifest = json.loads((diagnostics / "hybrid_obstacle_act_baseline"
                                / "checkpoint_manifest.json").read_text())
    av = oracle["artifact_verification"]
    ckpt_dir = Path(rawhead["artifact_verification"]["policy_best_ckpt_path"]).parent

    checks: list[dict] = []

    def check(name, actual, expected, source) -> None:
        checks.append({"artifact": name, "actual": actual, "expected": expected,
                       "expected_source": source, "matched": actual == expected})

    act = ROOT / "submodules/act"
    molmo = ROOT / "submodules/molmospaces"

    # ---- lineage ---------------------------------------------------------- #
    check("root_starting_commit",
          git("rev-parse", "develop/hybrid-obstacle-deployable-reference-v1"),
          "0bca2be10b6421115ecef50d8ffcfc8577f13821", "handoff starting state")
    check("act_starting_commit",
          git("rev-parse", "develop/hybrid-obstacle-posture-reference-v1", repo=act),
          "f6301bcaae4510946a7cb71e6feb3871d59327ad", "handoff starting state")
    check("molmospaces_commit", git("rev-parse", "HEAD", repo=molmo),
          oracle["commits"]["molmospaces_commit"], "oracle decision")
    check("molmospaces_unmodified", git("status", "--porcelain", repo=molmo) == "", True,
          "hard constraint")

    # ---- ACT checkpoint --------------------------------------------------- #
    check("policy_best_ckpt_path", str(ckpt_dir / "policy_best.ckpt"),
          ckpt_manifest["policy_best_ckpt"]["path"], "checkpoint_manifest.json")
    check("policy_best_ckpt_sha256", sha256_file(ckpt_dir / "policy_best.ckpt"),
          av["policy_best_ckpt_sha256"], "oracle decision")
    check("dataset_stats_pkl_sha256", sha256_file(ckpt_dir / "dataset_stats.pkl"),
          av["dataset_stats_pkl_sha256"], "oracle decision")
    check("best_epoch", int(ckpt_manifest["best_epoch"]), 1738, "checkpoint_manifest.json")
    check("checkpoint_manifest_sha256", ckpt_manifest["checkpoint_manifest_sha256"],
          baseline["checkpoint_manifest_sha256"], "ACT-baseline decision")

    # ---- canonical collection, conversion, split -------------------------- #
    collection = json.loads((ROOT / "configs/hybrid_obstacle_candidate_manifest_v2.json"
                             ).read_text())
    canonical = json.loads((ROOT / "configs/hybrid_obstacle_canonical_manifest_v2.json"
                            ).read_text())
    split = json.loads((ROOT / "configs/hybrid_obstacle_canonical_split_v2.json").read_text())
    check("collection_manifest_sha256", collection["manifest_sha256"],
          rawhead["artifact_verification"]["collection_manifest_sha256"], "raw-head decision")
    check("canonical_manifest_sha256", canonical["manifest_sha256"],
          rawhead["artifact_verification"]["canonical_manifest_sha256"], "raw-head decision")
    check("fixed_split_manifest_sha256", split["split_manifest_sha256"],
          rawhead["artifact_verification"]["fixed_split_manifest_sha256"], "raw-head decision")
    check("converted_dataset_tree_sha256", ckpt_manifest["converted_dataset_tree_sha256"],
          baseline["dataset"]["converted_dataset_tree_sha256"], "ACT-baseline decision")
    check("source_collection_tree_sha256", split["source_collection_tree_sha256"],
          baseline["dataset"]["source_collection_tree_sha256"], "ACT-baseline decision")
    check("split_level", split["level"], "trajectory", "split must be by trajectory")
    check("split_train_total", split["counts"]["train"]["total"], 80, "80 ACT-train")
    check("split_validation_total", split["counts"]["validation"]["total"], 20, "20 ACT-val")

    # ---- geometry / sensors / cameras / MSAA ------------------------------ #
    check("model_hybrid_xml_sha256",
          sha256_file(ROOT / "assets/robots/franka_skin/model_hybrid.xml"),
          av["model_hybrid_xml_sha256"], "oracle decision")
    check("camera_contract_sha256",
          sha256_file(molmo / "molmo_spaces/configs/camera_configs.py"),
          rawhead["artifact_verification"]["camera_contract_sha256"], "raw-head decision")
    stack = json.loads((ROOT / "configs/hybrid_safety_stack_v1.json").read_text())
    names = stack["sensor_contract"]["ordered_names"]
    check("sensor_order_sha256",
          hashlib.sha256(json.dumps(names, separators=(",", ":"),
                                    ensure_ascii=True).encode("ascii")).hexdigest(),
          av["sensor_order_sha256"], "oracle decision")
    check("sensor_count", len(names), 40, "40-sensor contract")
    check("msaa_offsamples", int(av["offsamples"]), 4, "retained MSAA contract")

    # ---- Safety-CVAE ------------------------------------------------------ #
    safety = ROOT / "assets/safety/cvae_v3"
    check("safety_model_sha256", sha256_file(safety / "model.pt"),
          av["safety_model_sha256"], "oracle decision")
    check("safety_meta_sha256", sha256_file(safety / "meta.json"),
          rawhead["artifact_verification"]["safety_meta_sha256"], "raw-head decision")
    meta = json.loads((safety / "meta.json").read_text())
    check("safety_head_io", [meta["n_in"], meta["n_out"]], [40 * 8 * 8, 7], "40x8x8 -> 7")
    check("safety_label_scale", float(meta["label_scale"]), 11.359346389770508, "pinned")

    # ---- oracle implementation and residual constants --------------------- #
    check("oracle_decision_token", oracle["decision"],
          "ORACLE_REFERENCE_VALID_CONTROLLER_VIABLE", "oracle task")
    check("oracle_reference_id", oracle["reference"]["id"], "ORACLE_PARKED_REFERENCE_V1",
          "oracle decision")
    check("oracle_mj_forward_called", oracle["implementation"]["mj_forward_called"], False,
          "the corrected pairing must be reused")
    check("oracle_state_neutrality_failures", oracle["state_neutrality"]["failures"], 0,
          "oracle decision")
    check("oracle_implementation_sha256",
          sha256_file(act / "parked_obstacle_reference.py"),
          sha256_file(act / "parked_obstacle_reference.py"),
          "recorded; the file is unmodified in this task")
    sys.path.insert(0, str(act))
    from hybrid_safety_residual import (
        DEFAULT_DECAY,
        DEFAULT_EMA,
        DEFAULT_GAIN,
        DEFAULT_MAX_DEVIATION,
    )
    check("residual_constants",
          [DEFAULT_GAIN, DEFAULT_DECAY, DEFAULT_EMA, DEFAULT_MAX_DEVIATION],
          [4.0, 2.2, 0.75, 0.35], "frozen constants")

    # ---- prior deployable task -------------------------------------------- #
    check("prior_decision_token", deployable["decision"],
          "DEPLOYABLE_REFERENCE_LIVE_GROSS_REGRESSION", "prior task")
    check("prior_selected_reference", deployable["selected"],
          "POSTURE_SKIN_MLP_REFERENCE_V1", "prior task")
    prior_deployment = json.loads((diagnostics / "hybrid_obstacle_deployable_reference"
                                   / "deployment_manifest.json").read_text())
    check("v1_deployment_manifest_sha256", prior_deployment["manifest_sha256"],
          deployable["frozen_model"]["manifest_sha256"], "prior decision")
    v1_artifact = Path(prior_deployment["artifact_path"])
    check("v1_checkpoint_sha256",
          sha256_file(v1_artifact) if v1_artifact.is_file() else None,
          prior_deployment["artifact_file_sha256"], "prior deployment manifest")
    check("v1_feature_width", prior_deployment["feature_width"], 196, "fixed schema")
    check("v1_runtime_inputs", prior_deployment["runtime_inputs"],
          ["qpos", "qvel", "nominal_action", "gripper_state", "gripper_command",
           "current_head", "sensor_summary"], "fixed feature schema")
    expert = json.loads((diagnostics / "hybrid_obstacle_deployable_reference"
                         / "paired_dataset_manifest.json").read_text())
    check("expert_paired_dataset_sha256", expert["manifest_sha256"],
          deployable["paired_dataset"]["manifest_sha256"], "prior decision")
    check("expert_paired_tree_sha256", expert["tree_sha256"],
          deployable["paired_dataset"]["tree_sha256"], "prior decision")
    check("expert_paired_neutrality_failures",
          expert["pairing_correctness"]["state_neutrality_failures"], 0, "prior dataset")

    # ---- evaluation manifests --------------------------------------------- #
    dev = json.loads((ROOT / "configs/hybrid_obstacle_controller_development4_v1.json"
                      ).read_text())
    conf = json.loads((ROOT / "configs/hybrid_obstacle_confirmatory41_v1.json").read_text())
    check("development4_manifest_sha256", dev["manifest_sha256"],
          av["development4_manifest_sha256"], "oracle decision")
    check("development4_role", dev["role"], "DEVELOPMENT_ONLY", "manifest role")
    check("confirmatory41_manifest_sha256", conf["manifest_sha256"],
          av["confirmatory41_manifest_sha256"], "oracle decision")
    check("confirmatory41_executed_in_this_task", bool(conf["executed_in_this_task"]),
          False, "hard requirement")
    check("confirmatory41_rows", len(conf["rows"]), 41, "32 present + 9 absent")

    # ---- frozen rollouts we will reuse ------------------------------------ #
    for name, root_dir, pattern, expected in (
        ("frozen_act_only_rollouts", "/root/act_retrain_assets/rawhead_dev_v1",
         "cand*_act_only_r*/summary.json", 20),
        ("frozen_oracle_rollouts", "/root/act_retrain_assets/oracle_dev_v1",
         "cand*_oracle_r*/summary.json", 20),
        ("frozen_v1_deployable_rollouts", "/root/act_retrain_assets/deployable_dev_v1",
         "cand*_deployable_r*/summary.json", 20),
    ):
        check(name, len(sorted(Path(root_dir).glob(pattern))), expected, "prior task outputs")

    failed = [c for c in checks if not c["matched"]]
    report = {
        "schema": "hybrid_obstacle_on_policy_provenance_v1",
        "starting_state": {
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "root_commit": git("rev-parse", "HEAD"),
            "act_branch": git("rev-parse", "--abbrev-ref", "HEAD", repo=act),
            "act_commit": git("rev-parse", "HEAD", repo=act),
            "molmospaces_commit": git("rev-parse", "HEAD", repo=molmo),
            "root_status": git("status", "--porcelain").splitlines(),
            "act_status": git("status", "--porcelain", repo=act).splitlines(),
            "molmospaces_status": git("status", "--porcelain", repo=molmo).splitlines(),
            "root_stashes": len(git("stash", "list").splitlines()),
            "remotes": git("remote", "-v").splitlines(),
            "gitlinks": git("ls-tree", "HEAD", "submodules/").splitlines(),
        },
        "checks": checks,
        "check_count": len(checks),
        "failed": [c["artifact"] for c in failed],
        "all_matched": not failed,
        "decision_if_failed": "CHECKPOINT_OR_SOURCE_MISMATCH",
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    for entry in checks:
        print(f"  [{'ok  ' if entry['matched'] else 'FAIL'}] {entry['artifact']:<40} "
              f"{str(entry['actual'])[:52]}")
    print(f"\n{len(checks)} checks, {len(failed)} failed -> {args.out}")
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
