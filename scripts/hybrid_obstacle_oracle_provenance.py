#!/usr/bin/env python3
"""Immutable-provenance verification for the per-frame parked-obstacle oracle task.

Handoff step 2. Recovers every expected digest from the *prior* decision artifacts
rather than from freshly computed values, then re-hashes the live files and requires
an exact match. Any mismatch is reported and the caller stops with
``CHECKPOINT_OR_SOURCE_MISMATCH``.

Nothing here writes to a verified artifact.
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
    ap.add_argument("--prior-decision", type=Path,
                    default=ROOT / "diagnostics_output/hybrid_obstacle_raw_head_qualification"
                                   "/final_decision.json")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    prior = json.loads(args.prior_decision.read_text())
    av = prior["artifact_verification"]
    ckpt_dir = Path(av["policy_best_ckpt_path"]).parent

    checks: list[dict] = []

    def check(name: str, actual, expected, source: str) -> None:
        checks.append({
            "artifact": name,
            "actual": actual,
            "expected": expected,
            "expected_source": source,
            "matched": actual == expected,
        })

    # ---- source lineage ------------------------------------------------- #
    act = ROOT / "submodules/act"
    molmo = ROOT / "submodules/molmospaces"
    check("root_starting_commit", git("rev-parse", "HEAD~0" if False else "eval/hybrid-obstacle-raw-head-development-v1"),
          "5a16963268f7581adcce2b7ec484bb6ee9adf610", "task-5 root commit (handoff)")
    check("act_starting_commit",
          git("rev-parse", "eval/hybrid-obstacle-raw-head-stochastic-v1", repo=act),
          "68713a1af620344818bb7873de4d30e22bbf6992", "task-5 ACT commit (handoff)")
    check("molmospaces_commit", git("rev-parse", "HEAD", repo=molmo),
          av["__molmo__"] if "__molmo__" in av else prior["commits"]["molmospaces_commit"],
          "prior decision commits.molmospaces_commit")
    check("molmospaces_unmodified", git("status", "--porcelain", repo=molmo) == "", True,
          "hard constraint: do not modify MolmoSpaces")

    # ---- checkpoints and statistics ------------------------------------- #
    check("policy_best_ckpt_sha256", sha256_file(ckpt_dir / "policy_best.ckpt"),
          av["policy_best_ckpt_sha256"], "prior decision artifact_verification")
    check("dataset_stats_pkl_sha256", sha256_file(ckpt_dir / "dataset_stats.pkl"),
          av["dataset_stats_pkl_sha256"], "prior decision artifact_verification")

    # The handoff calls this "checkpoint_manifest.json"; the file the ACT-baseline task
    # actually committed lives under diagnostics_output (the run dir holds run_manifest.json,
    # which is written before training and so carries no checkpoint digest).
    ckpt_manifest_path = (ROOT / "diagnostics_output/hybrid_obstacle_act_baseline"
                               / "checkpoint_manifest.json")
    manifest = json.loads(ckpt_manifest_path.read_text())
    baseline = json.loads((ROOT / "diagnostics_output/hybrid_obstacle_act_baseline"
                                / "final_decision.json").read_text())
    # The manifest embeds its own canonical digest, so the file hash necessarily differs;
    # verify the embedded value against the ACT-baseline decision and re-derive it.
    check("checkpoint_manifest_sha256", manifest["checkpoint_manifest_sha256"],
          baseline["checkpoint_manifest_sha256"], "ACT-baseline decision")
    check("checkpoint_manifest_self_hash_recomputes",
          canonical_hash({k: v for k, v in manifest.items()
                          if k != "checkpoint_manifest_sha256"}),
          manifest["checkpoint_manifest_sha256"], "canonical hash of the manifest payload")
    check("checkpoint_manifest_records_best_sha256", manifest["policy_best_ckpt"]["sha256"],
          av["policy_best_ckpt_sha256"], "checkpoint_manifest.json policy_best_ckpt")
    check("checkpoint_manifest_records_stats_sha256", manifest["dataset_stats_pkl_sha256"],
          av["dataset_stats_pkl_sha256"], "checkpoint_manifest.json dataset_stats_pkl_sha256")
    check("best_epoch", int(manifest["best_epoch"]), int(av["best_epoch"]),
          "checkpoint_manifest.json best_epoch")
    check("checkpoint_manifest_molmospaces_commit", manifest["molmospaces_commit"],
          prior["commits"]["molmospaces_commit"], "checkpoint_manifest.json")
    check("run_manifest_sha256",
          json.loads((ckpt_dir / "run_manifest.json").read_text())["run_manifest_sha256"],
          baseline["run_manifest_sha256"], "ACT-baseline decision")

    # ---- training configuration ----------------------------------------- #
    check("training_config_sha256",
          sha256_file(ROOT / "configs/hybrid_obstacle_act_baseline_v2.yaml"),
          av["training_config_sha256"], "prior decision artifact_verification")

    # ---- canonical dataset, split, source collection --------------------- #
    for name, path, key in (
        ("collection_manifest_sha256", "configs/hybrid_obstacle_candidate_manifest_v2.json",
         "manifest_sha256"),
        ("canonical_manifest_sha256", "configs/hybrid_obstacle_canonical_manifest_v2.json",
         "manifest_sha256"),
        ("fixed_split_manifest_sha256", "configs/hybrid_obstacle_canonical_split_v2.json",
         "split_manifest_sha256"),
    ):
        doc = json.loads((ROOT / path).read_text())
        embedded = doc.get(key) or doc.get("manifest_sha256") or doc.get("split_manifest_sha256")
        check(name, embedded, av[name], "prior decision artifact_verification")

    # ---- Safety-CVAE ------------------------------------------------------ #
    safety = ROOT / "assets/safety/cvae_v3"
    check("safety_model_sha256", sha256_file(safety / "model.pt"), av["safety_model_sha256"],
          "prior decision artifact_verification")
    check("safety_meta_sha256", sha256_file(safety / "meta.json"), av["safety_meta_sha256"],
          "prior decision artifact_verification")
    meta = json.loads((safety / "meta.json").read_text())
    check("safety_head_io", [meta["n_in"], meta["n_out"]], [40 * 8 * 8, 7], "40x8x8 -> 7 contract")
    check("safety_label_scale", float(meta["label_scale"]), 11.359346389770508,
          "pinned label_scale")

    # ---- geometry / sensors / cameras ------------------------------------ #
    check("model_hybrid_xml_sha256",
          sha256_file(ROOT / "assets/robots/franka_skin/model_hybrid.xml"),
          av["model_hybrid_xml_sha256"], "prior decision artifact_verification")
    check("camera_contract_sha256",
          sha256_file(molmo / "molmo_spaces/configs/camera_configs.py"),
          av["camera_contract_sha256"], "prior decision artifact_verification")
    stack = json.loads((ROOT / "configs/hybrid_safety_stack_v1.json").read_text())
    names = stack["sensor_contract"]["ordered_names"]
    order_hash = hashlib.sha256(
        json.dumps(names, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
    check("sensor_order_sha256", order_hash, av["sensor_order_sha256"],
          "prior decision artifact_verification")
    check("sensor_count", len(names), 40, "40-sensor contract")
    check("sensor_order_matches_safety_meta", meta["sensors"], names,
          "Safety-CVAE metadata sensor order")

    # ---- evaluation manifests -------------------------------------------- #
    dev = json.loads((ROOT / "configs/hybrid_obstacle_controller_development4_v1.json").read_text())
    conf = json.loads((ROOT / "configs/hybrid_obstacle_confirmatory41_v1.json").read_text())
    check("development4_manifest_sha256", dev["manifest_sha256"],
          "5aaf6ddb4aba56bc17434fb860f809c137ba8e5fd41b309cd6382c66c8a1bd0b",
          "prior decision artifacts.development4_manifest")
    check("development4_rows", len(dev["rows"]), 4, "four development rows")
    check("development4_role", dev["role"], "DEVELOPMENT_ONLY", "manifest role")
    check("confirmatory41_manifest_sha256", conf["manifest_sha256"],
          "7b4500e9b4b2868e2612d7e444c34762d72c5e6e7b4b7c38bcf31f027b51b69e",
          "prior decision artifacts.confirmatory41_manifest")
    check("confirmatory41_rows", len(conf["rows"]), 41, "32 present + 9 absent")
    check("confirmatory41_executed_in_this_task", bool(conf["executed_in_this_task"]), False,
          "hard requirement: must remain false")
    check("confirmatory41_role", conf["role"], "CONFIRMATORY_UNTOUCHED", "manifest role")
    dev_ids = {r["episode_id"] for r in dev["rows"]}
    conf_ids = {r["episode_id"] for r in conf["rows"]}
    check("development_and_confirmatory_disjoint", sorted(dev_ids & conf_ids), [],
          "no development row may appear in the confirmatory set")

    # ---- source collection trajectories ---------------------------------- #
    run = ROOT / dev["source_run_dir"] if not Path(dev["source_run_dir"]).is_absolute() \
        else Path(dev["source_run_dir"])
    for row in dev["rows"]:
        h5 = run / "rows" / row["episode_id"] / "trajectory.h5"
        check(f"source_h5_cand{row['candidate_index']}", sha256_file(h5),
              row["source_h5_sha256"], "development4 manifest row")

    # ---- prior raw-head artifacts ---------------------------------------- #
    check("prior_final_decision_sha256",
          canonical_hash({k: v for k, v in prior.items() if k != "final_decision_sha256"}),
          prior["final_decision_sha256"], "self-consistency of the prior decision JSON")
    check("prior_decision_token", prior["decision"], "RAW_HEAD_CONTROLLER_GROSS_REGRESSION",
          "task-5 outcome")

    # ---- MSAA / render contract ------------------------------------------ #
    check("offsamples_expected", int(av["offsamples"]), 4, "retained MSAA contract")

    failed = [c for c in checks if not c["matched"]]
    report = {
        "schema": "hybrid_obstacle_oracle_provenance_v1",
        "prior_decision": str(args.prior_decision.relative_to(ROOT)),
        "starting_commits": {
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "root_commit": git("rev-parse", "HEAD"),
            "act_branch": git("rev-parse", "--abbrev-ref", "HEAD", repo=act),
            "act_commit": git("rev-parse", "HEAD", repo=act),
            "molmospaces_commit": git("rev-parse", "HEAD", repo=molmo),
            "root_status_porcelain": git("status", "--porcelain").splitlines(),
            "act_status_porcelain": git("status", "--porcelain", repo=act).splitlines(),
            "molmospaces_status_porcelain": git("status", "--porcelain", repo=molmo).splitlines(),
            "root_stashes": len(git("stash", "list").splitlines()),
            "act_stashes": len(git("stash", "list", repo=act).splitlines()),
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

    for c in checks:
        mark = "ok  " if c["matched"] else "FAIL"
        shown = str(c["actual"])
        print(f"  [{mark}] {c['artifact']:<48} {shown[:64]}")
    print(f"\n{len(checks)} checks, {len(failed)} failed -> {args.out}")
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
