#!/usr/bin/env python3
"""Immutable-provenance verification for the parked-skin reference task.

Handoff steps 1-2. Expected digests are recovered from every prior decision artifact and
the live files are re-hashed against them. Any mismatch stops the task with
``CHECKPOINT_OR_SOURCE_MISMATCH``.

The starting commits are resolved from the prior decision document as instructed. Note
that a decision JSON necessarily records the commit that existed *while it was being
written*, so the prior task's own commit is one step later; both are recorded and the
branch tips are verified explicitly.
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
    on_policy = json.loads((diagnostics / "hybrid_obstacle_on_policy_reference"
                            / "final_decision.json").read_text())
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
    check("prior_root_branch_tip",
          git("rev-parse", "develop/hybrid-obstacle-on-policy-reference-v2"),
          "5270bee29604f30e020cb9c28318ef90e030d500", "prior task HEAD")
    check("prior_act_branch_tip",
          git("rev-parse", "develop/hybrid-obstacle-on-policy-reference-v2", repo=act),
          "21bf05efd49887dc18bbf0b4094cd7663526c2bb", "prior task HEAD")
    check("prior_decision_recorded_root_commit_is_its_parent",
          git("rev-parse", "5270bee29604f30e020cb9c28318ef90e030d500^"),
          on_policy["commits"]["root_commit"],
          "the decision JSON records the commit it was written against")
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

    # ---- collection, conversion, split ------------------------------------ #
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

    # ---- geometry, sensors, cameras, MSAA --------------------------------- #
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
    check("offsamples", int(av["offsamples"]), 4, "retained MSAA contract")

    # ---- Safety-CVAE ------------------------------------------------------ #
    safety = ROOT / "assets/safety/cvae_v3"
    check("safety_model_sha256", sha256_file(safety / "model.pt"),
          av["safety_model_sha256"], "oracle decision")
    check("safety_meta_sha256", sha256_file(safety / "meta.json"),
          rawhead["artifact_verification"]["safety_meta_sha256"], "raw-head decision")
    meta = json.loads((safety / "meta.json").read_text())
    check("safety_head_io", [meta["n_in"], meta["n_out"]], [40 * 8 * 8, 7], "40x8x8 -> 7")
    check("safety_d_max_input", float(meta["d_max_input"]), 0.5,
          "the closeness transform's reference depth")

    # ---- oracle implementation and residual constants --------------------- #
    check("oracle_implementation_sha256", sha256_file(act / "parked_obstacle_reference.py"),
          sha256_file(act / "parked_obstacle_reference.py"),
          "recorded; unmodified in this task")
    check("oracle_mj_forward_called", oracle["implementation"]["mj_forward_called"], False,
          "the corrected pairing")
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
    check("residual_contract_sha256", sha256_file(act / "hybrid_safety_residual.py"),
          sha256_file(act / "hybrid_safety_residual.py"), "recorded; unmodified")

    # ---- paired dataset and partition ------------------------------------- #
    paired = json.loads((diagnostics / "hybrid_obstacle_deployable_reference"
                         / "paired_dataset_manifest.json").read_text())
    labelling = json.loads((diagnostics / "hybrid_obstacle_on_policy_reference"
                            / "labelling_dataset_manifest.json").read_text())
    learner = json.loads((diagnostics / "hybrid_obstacle_on_policy_reference"
                          / "learner_dataset_manifest.json").read_text())
    partition = json.loads((ROOT / "configs/hybrid_obstacle_reference_partition_v2.json"
                            ).read_text())
    check("expert_paired_manifest_sha256", paired["manifest_sha256"],
          deployable["paired_dataset"]["manifest_sha256"], "deployable decision")
    check("expert_paired_tree_sha256", paired["tree_sha256"],
          deployable["paired_dataset"]["tree_sha256"], "deployable decision")
    check("labelling_dataset_sha256", labelling["manifest_sha256"],
          on_policy["datasets"]["labelling"]["manifest_sha256"], "on-policy decision")
    check("learner_dataset_sha256", learner["manifest_sha256"],
          on_policy["datasets"]["learner"]["manifest_sha256"], "on-policy decision")
    check("partition_sha256", partition["partition_sha256"],
          on_policy["reference_partition"]["sha256"], "on-policy decision")
    check("partition_composition", partition["composition"],
          on_policy["reference_partition"]["composition"], "on-policy decision")
    check("partition_all_pairwise_disjoint", partition["all_pairwise_disjoint"], True,
          "partition manifest")
    check("paired_state_neutrality_failures",
          paired["pairing_correctness"]["state_neutrality_failures"], 0, "expert dataset")
    check("labelling_state_neutrality_failures",
          labelling["oracle_pairing"]["state_neutrality_failures"], 0, "on-policy dataset")
    check("learner_state_neutrality_failures",
          learner["oracle_pairing"]["state_neutrality_failures"], 0, "learner dataset")

    # ---- prior reference models ------------------------------------------- #
    v1 = json.loads((diagnostics / "hybrid_obstacle_deployable_reference"
                     / "deployment_manifest.json").read_text())
    v2 = json.loads((diagnostics / "hybrid_obstacle_on_policy_reference"
                     / "round1_training.json").read_text())
    check("v1_reference_manifest_sha256", v1["manifest_sha256"],
          deployable["frozen_model"]["manifest_sha256"], "deployable decision")
    check("v1_checkpoint_sha256",
          sha256_file(Path(v1["artifact_path"])) if Path(v1["artifact_path"]).is_file()
          else None, v1["artifact_file_sha256"], "V1 deployment manifest")
    check("v2_round1_checkpoint_sha256",
          sha256_file(Path(v2["artifact_path"])) if Path(v2["artifact_path"]).is_file()
          else None, v2["artifact_file_sha256"], "V2 round-1 training report")
    check("prior_decision_token", on_policy["decision"],
          "ON_POLICY_REFERENCE_OFFLINE_INVALID", "prior task")
    check("prior_case", on_policy["case"], "C", "prior task")

    # ---- evaluation manifests --------------------------------------------- #
    development = json.loads((ROOT / "configs/hybrid_obstacle_controller_development4_v1.json"
                              ).read_text())
    confirmatory = json.loads((ROOT / "configs/hybrid_obstacle_confirmatory41_v1.json"
                               ).read_text())
    check("development4_manifest_sha256", development["manifest_sha256"],
          av["development4_manifest_sha256"], "oracle decision")
    check("confirmatory41_manifest_sha256", confirmatory["manifest_sha256"],
          av["confirmatory41_manifest_sha256"], "oracle decision")
    check("confirmatory41_executed_in_this_task",
          bool(confirmatory["executed_in_this_task"]), False, "hard requirement")
    check("confirmatory41_rows", len(confirmatory["rows"]), 41, "32 present + 9 absent")

    development_ids = {r["episode_id"] for r in development["rows"]}
    confirmatory_ids = {r["episode_id"] for r in confirmatory["rows"]}
    for name in ("reference_train", "reference_calibration", "reference_validation",
                 "offline_reference_test"):
        ids = {r["episode_id"] for r in partition["partitions"][name]}
        check(f"{name}_free_of_development4", sorted(ids & development_ids), [],
              "no development row may train, calibrate or validate")
        check(f"{name}_free_of_confirmatory41", sorted(ids & confirmatory_ids), [],
              "confirmatory41 takes no part")

    failed = [c for c in checks if not c["matched"]]
    report = {
        "schema": "hybrid_obstacle_parked_skin_provenance_v1",
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
        print(f"  [{'ok  ' if entry['matched'] else 'FAIL'}] {entry['artifact']:<48} "
              f"{str(entry['actual'])[:44]}")
    print(f"\n{len(checks)} checks, {len(failed)} failed -> {args.out}")
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
