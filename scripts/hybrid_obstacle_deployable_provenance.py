#!/usr/bin/env python3
"""Immutable-provenance verification for the deployable posture-conditioned reference.

Handoff step 2. Expected digests are recovered from the *prior* decision artifacts --
the oracle, raw-head, ACT-baseline and dataset decisions -- and the live files are
re-hashed against them. Any mismatch is reported and the caller stops with
``CHECKPOINT_OR_SOURCE_MISMATCH``.

Also proves the four data partitions are pairwise disjoint, which is what lets the
reference model use the canonical 80/20 split while confirmatory41 stays independent.
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

    def check(name: str, actual, expected, source: str) -> None:
        checks.append({"artifact": name, "actual": actual, "expected": expected,
                       "expected_source": source, "matched": actual == expected})

    act = ROOT / "submodules/act"
    molmo = ROOT / "submodules/molmospaces"

    # ---- source lineage -------------------------------------------------- #
    check("root_starting_commit",
          git("rev-parse", "eval/hybrid-obstacle-oracle-reference-v1"),
          "4193b776640886c1bddc6be5adc7bdaf35855643", "handoff starting state")
    check("act_starting_commit",
          git("rev-parse", "eval/hybrid-obstacle-per-frame-oracle-v1", repo=act),
          "709a22de62ac0e8c4640b75eb348416d6e29013d", "handoff starting state")
    check("molmospaces_commit", git("rev-parse", "HEAD", repo=molmo),
          oracle["commits"]["molmospaces_commit"], "oracle decision")
    check("molmospaces_unmodified", git("status", "--porcelain", repo=molmo) == "", True,
          "hard constraint")

    # ---- ACT checkpoint and statistics ----------------------------------- #
    check("policy_best_ckpt_path", str(ckpt_dir / "policy_best.ckpt"),
          ckpt_manifest["policy_best_ckpt"]["path"], "checkpoint_manifest.json")
    check("policy_best_ckpt_sha256", sha256_file(ckpt_dir / "policy_best.ckpt"),
          av["policy_best_ckpt_sha256"], "oracle decision artifact_verification")
    check("dataset_stats_pkl_sha256", sha256_file(ckpt_dir / "dataset_stats.pkl"),
          av["dataset_stats_pkl_sha256"], "oracle decision artifact_verification")
    check("best_epoch", int(ckpt_manifest["best_epoch"]), int(av["best_epoch"]),
          "checkpoint_manifest.json")
    check("checkpoint_manifest_sha256", ckpt_manifest["checkpoint_manifest_sha256"],
          baseline["checkpoint_manifest_sha256"], "ACT-baseline decision")

    # ---- canonical dataset, split, source collection ---------------------- #
    canonical = json.loads((ROOT / "configs/hybrid_obstacle_canonical_manifest_v2.json"
                            ).read_text())
    split = json.loads((ROOT / "configs/hybrid_obstacle_canonical_split_v2.json").read_text())
    collection = json.loads((ROOT / "configs/hybrid_obstacle_candidate_manifest_v2.json"
                             ).read_text())
    check("collection_manifest_sha256", collection["manifest_sha256"],
          av["collection_manifest_sha256"] if "collection_manifest_sha256" in av
          else rawhead["artifact_verification"]["collection_manifest_sha256"],
          "prior decision artifact_verification")
    check("canonical_manifest_sha256", canonical["manifest_sha256"],
          rawhead["artifact_verification"]["canonical_manifest_sha256"],
          "raw-head decision artifact_verification")
    check("fixed_split_manifest_sha256", split["split_manifest_sha256"],
          rawhead["artifact_verification"]["fixed_split_manifest_sha256"],
          "raw-head decision artifact_verification")
    check("split_level", split["level"], "trajectory", "split must be by trajectory")
    check("split_leakage_free", split["leakage_free"], True, "split manifest")
    check("split_counts", split["counts"]["train"]["total"], 80, "80 training trajectories")
    check("split_validation_count", split["counts"]["validation"]["total"], 20,
          "20 validation trajectories")
    check("source_collection_tree_sha256", split["source_collection_tree_sha256"],
          baseline["dataset"]["source_collection_tree_sha256"], "ACT-baseline decision")

    # ---- geometry / sensors / cameras ------------------------------------- #
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
    check("msaa_offsamples_contract", int(av["offsamples"]), 4, "retained MSAA contract")

    # ---- Safety-CVAE ------------------------------------------------------ #
    safety = ROOT / "assets/safety/cvae_v3"
    check("safety_model_sha256", sha256_file(safety / "model.pt"),
          av["safety_model_sha256"], "oracle decision")
    meta = json.loads((safety / "meta.json").read_text())
    check("safety_meta_sha256", sha256_file(safety / "meta.json"),
          rawhead["artifact_verification"]["safety_meta_sha256"], "raw-head decision")
    check("safety_head_io", [meta["n_in"], meta["n_out"]], [40 * 8 * 8, 7], "40x8x8 -> 7")
    check("safety_label_scale", float(meta["label_scale"]), 11.359346389770508,
          "pinned label_scale")
    check("safety_sensor_order_matches_contract", meta["sensors"], names, "contract")

    # ---- oracle implementation and controller constants ------------------- #
    check("oracle_decision_token", oracle["decision"],
          "ORACLE_REFERENCE_VALID_CONTROLLER_VIABLE", "prior task")
    check("oracle_case", oracle["case"], "A", "prior task")
    check("oracle_reference_id", oracle["reference"]["id"], "ORACLE_PARKED_REFERENCE_V1",
          "oracle decision")
    check("oracle_reference_not_deployable", oracle["reference"]["deployable"], False,
          "oracle decision")
    check("oracle_committed_parked_pose", oracle["reference"]["committed_parked_pose"],
          {"protr_s": [0.0, 0.8, -2.0], "protr_m": [0.0, 1.2, -2.0],
           "protr_l": [0.0, 1.6, -2.0]}, "oracle decision")
    check("oracle_state_neutrality_failures", oracle["state_neutrality"]["failures"], 0,
          "oracle decision")
    check("oracle_mj_forward_called", oracle["implementation"]["mj_forward_called"], False,
          "the corrected pairing must be reused")
    check("oracle_final_decision_sha256",
          canonical_hash({k: v for k, v in oracle.items()
                          if k != "final_decision_sha256"}),
          oracle["final_decision_sha256"], "self-consistency")
    sys.path.insert(0, str(act))
    from hybrid_safety_residual import (
        DEFAULT_DECAY,
        DEFAULT_EMA,
        DEFAULT_GAIN,
        DEFAULT_MAX_DEVIATION,
    )
    check("controller_constants",
          [DEFAULT_GAIN, DEFAULT_DECAY, DEFAULT_EMA, DEFAULT_MAX_DEVIATION],
          [4.0, 2.2, 0.75, 0.35], "frozen constants, untouched")
    check("oracle_controller_constants_untuned",
          oracle["constraints_honoured"]["controller_constants_or_scale_tuned"], False,
          "oracle decision")

    # ---- evaluation manifests --------------------------------------------- #
    dev = json.loads((ROOT / "configs/hybrid_obstacle_controller_development4_v1.json"
                      ).read_text())
    conf = json.loads((ROOT / "configs/hybrid_obstacle_confirmatory41_v1.json").read_text())
    check("development4_manifest_sha256", dev["manifest_sha256"],
          av["development4_manifest_sha256"], "oracle decision")
    check("development4_role", dev["role"], "DEVELOPMENT_ONLY", "manifest role")
    check("confirmatory41_manifest_sha256", conf["manifest_sha256"],
          av["confirmatory41_manifest_sha256"], "oracle decision")
    check("confirmatory41_executed_in_this_task", bool(conf["executed_in_this_task"]), False,
          "hard requirement")
    check("confirmatory41_rows", len(conf["rows"]), 41, "32 present + 9 absent")

    # ---- partition disjointness ------------------------------------------- #
    train = {e["episode_id"] for e in split["episodes"] if e["split"] == "train"}
    validation = {e["episode_id"] for e in split["episodes"] if e["split"] == "validation"}
    development = {r["episode_id"] for r in dev["rows"]}
    confirmatory = {r["episode_id"] for r in conf["rows"]}
    for left_name, left, right_name, right in (
        ("train", train, "validation", validation),
        ("train", train, "development4", development),
        ("validation", validation, "development4", development),
        ("train", train, "confirmatory41", confirmatory),
        ("validation", validation, "confirmatory41", confirmatory),
        ("development4", development, "confirmatory41", confirmatory),
    ):
        check(f"disjoint_{left_name}_{right_name}", sorted(left & right), [],
              "partitions must not overlap")

    # ---- the frozen oracle rollouts we will reuse -------------------------- #
    for name, root_dir, pattern, expected in (
        ("frozen_act_only_rollouts", "/root/act_retrain_assets/rawhead_dev_v1",
         "cand*_act_only_r*/summary.json", 20),
        ("frozen_oracle_rollouts", "/root/act_retrain_assets/oracle_dev_v1",
         "cand*_oracle_r*/summary.json", 20),
    ):
        found = sorted(Path(root_dir).glob(pattern))
        check(name, len(found), expected, "prior task outputs")

    failed = [c for c in checks if not c["matched"]]
    report = {
        "schema": "hybrid_obstacle_deployable_provenance_v1",
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
        "partitions": {
            "reference_training_trajectories": len(train),
            "reference_validation_trajectories": len(validation),
            "live_development_rows": len(development),
            "confirmatory_rows": len(confirmatory),
            "all_pairwise_disjoint": all(c["matched"] for c in checks
                                         if c["artifact"].startswith("disjoint_")),
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
        print(f"  [{'ok  ' if entry['matched'] else 'FAIL'}] {entry['artifact']:<44} "
              f"{str(entry['actual'])[:56]}")
    print(f"\n{len(checks)} checks, {len(failed)} failed -> {args.out}")
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
