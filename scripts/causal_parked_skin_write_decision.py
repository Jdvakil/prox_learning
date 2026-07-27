#!/usr/bin/env python3
"""Assemble the final decision for the parked-skin reference learnability task.

Handoff steps 14, 17, 18 and 19. Every value is copied from an artifact produced earlier;
the decision token comes from ``causal_parked_skin.gates``, which is a pure function of
those artifacts, so the token cannot be argued into a different value here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin.gates import ALLOWED_DECISIONS, resolve


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-contract", required=True, type=Path)
    ap.add_argument("--partition-report", required=True, type=Path)
    ap.add_argument("--selection", required=True, type=Path)
    ap.add_argument("--final-training", required=True, type=Path)
    ap.add_argument("--dataset-decision", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    contract = json.loads(args.input_contract.read_text())
    partition = json.loads(args.partition_report.read_text())
    selection = json.loads(args.selection.read_text())
    final = json.loads(args.final_training.read_text())
    dataset = json.loads(args.dataset_decision.read_text())

    checkpoints_exist = all(
        Path(run["training"]["best_checkpoint"]).is_file()
        for run in final["runs"].values())

    resolved = resolve(final, contract, partition, dataset,
                       training_produced_checkpoints=checkpoints_exist)
    decision = resolved["decision"]
    if decision not in ALLOWED_DECISIONS:
        raise SystemExit(f"decision {decision!r} is not allowed")

    act = ROOT / "submodules" / "act"
    molmo = ROOT / "submodules" / "molmospaces"
    safety_hashes = {p.name: file_hash(p) for p in
                     sorted(args.safety_dir.glob("*")) if p.is_file()
                     and p.suffix in (".pt", ".json")}

    # Handoff step 17: every checkpoint pinned to the exact sources that produced it.
    pinned = []
    for key, run in sorted(final["runs"].items()):
        best = Path(run["training"]["best_checkpoint"])
        pinned.append({
            "run": key,
            "variant": run["variant"],
            "seed": run["seed"],
            "local_path": str(best),
            "exists": best.is_file(),
            "sha256": run["training"]["best_checkpoint_sha256"],
            "last_checkpoint_path": run["training"]["last_checkpoint"],
            "last_checkpoint_sha256": run["training"]["last_checkpoint_sha256"],
            "model_config_hash": run["training"]["config_hash"],
            "parameter_count": run["training"]["parameter_count"],
            "root_commit": git("rev-parse", "HEAD"),
            "act_commit": git("rev-parse", "HEAD", repo=act),
            "molmospaces_commit": git("rev-parse", "HEAD", repo=molmo),
            "safety_cvae_hashes": safety_hashes,
            "dataset_tree_sha256": dataset["dataset"]["tree_sha256"],
            "partition_sha256": dataset["dataset"]["partition_sha256"],
            "input_contract_sha256": contract["input_contract_sha256"],
            "best_epoch": run["training"]["best_epoch"],
            "validation_metric": run["training"]["best_validation_head_mae"],
            "calibrated_activation_threshold":
                run["calibration_threshold"]["threshold"],
        })

    payload = {
        "schema": "causal_parked_skin_reference_v1_final_decision_v1",
        "date": "2026-07-27",
        "task": ("Train and evaluate CAUSAL_PARKED_SKIN_REFERENCE_V1 using the frozen "
                 "parked-skin dataset"),
        "decision": decision,
        "frozen_primary_model": resolved["frozen_primary"],
        "gate_branch": resolved["branch"],
        "offline_learnability_task": True,
        "live_rollouts_run": False,
        "development4_executed": False,
        "confirmatory41_executed": False,
        "act_trained_or_modified": False,
        "safety_cvae_trained_or_modified": False,

        "commits": {
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "root_starting_commit": "9c4703a",
            "root_commit": git("rev-parse", "HEAD"),
            "act_commit": git("rev-parse", "HEAD", repo=act),
            "act_expected": "91fc42a",
            "act_modified": git("status", "--porcelain", repo=act) != "",
            "molmospaces_commit": git("rev-parse", "HEAD", repo=molmo),
            "molmospaces_expected": "678f2eb",
            "molmospaces_modified": git("status", "--porcelain", repo=molmo) != "",
        },

        "dataset": {
            "version": dataset["dataset"]["version"],
            "root": dataset["dataset"]["root"],
            "tree_sha256": dataset["dataset"]["tree_sha256"],
            "manifest_sha256": dataset["dataset"]["manifest_sha256"],
            "partition_sha256": dataset["dataset"]["partition_sha256"],
            "read_only": dataset["dataset"]["read_only"],
            "modified_by_this_task": False,
            "counts": dataset["dataset"]["counts"],
        },
        "partition_independence": {
            "total_crossings": partition["total_crossings"],
            "identity_keys_checked": partition["identity_keys_checked"],
            "composition": partition["composition"],
            "valid": partition["valid"],
        },
        "input_contract": {
            "sha256": contract["input_contract_sha256"],
            "model_input_fields": contract["contract"]["model_input_fields"],
            "state_vector_width": contract["contract"]["state_vector_width"],
            "causal_history_frames": contract["contract"]["causal_history_frames"],
            "prohibited_inputs_used": contract["prohibited_inputs_used"],
            "inputs_not_live_available": contract["inputs_not_live_available"],
            "live_evaluator_sha256": contract["live_evaluator"]["sha256"],
        },

        "selection": {
            "candidate_budget": selection["candidate_budget"],
            "candidates_run": selection["candidates_run"],
            "selection_partition": selection["selection_partition"],
            "offline_test_loaded_during_selection": selection["offline_test_loaded"],
            "pre_candidate_diagnostics": selection["pre_candidate_diagnostics"],
            "ranking": selection["ranking"],
            "selected": selection["selected"],
            "selected_config": selection["selected_config"],
            "selected_config_hash": selection["selected_config_hash"],
            "varied_axes": selection["varied_axes"],
            "fixed_axes": selection["fixed_axes"],
            "validation_head_mae_by_candidate": {
                k: v["validation_head_mae"] for k, v in selection["results"].items()},
        },

        "calibration": {
            key: run["calibration_threshold"] for key, run in
            sorted(final["runs"].items())},

        "results": {
            "seeds": final["seeds"],
            "runs": {key: {
                "variant": run["variant"], "seed": run["seed"],
                "best_epoch": run["training"]["best_epoch"],
                "epochs_run": run["training"]["epochs_run"],
                "early_stopped": run["training"]["early_stopped"],
                "validation_head_mae": run["training"]["best_validation_head_mae"],
                "metrics": run["metrics"],
            } for key, run in sorted(final["runs"].items())},
            "baselines": final["baselines"],
            "privileged_upper_bound": final["privileged_upper_bound"],
        },

        "sampling": next(iter(final["runs"].values()))["training"]["sampler"],
        "safety_head": final["safety_head"],
        "checkpoint_reload_determinism": final["checkpoint_reload_determinism"],
        "checkpoints": pinned,
        "checkpoints_committed": False,
        "checkpoint_policy": ("checkpoints remain external to git; no approved artifact "
                              "policy exists for binary model weights in this repository"),

        "gates": resolved["gates"],
        "full_causal_gates": resolved.get("full_causal_gates"),
        "temporal_history_value": resolved["gates"]["temporal_history_value"],

        "constraints_honoured": {
            "frozen_dataset_modified": False,
            "partitions_changed": False,
            "offline_test_used_for_selection": False,
            "oracle_zero_frames_removed": False,
            "act_trained_or_modified": False,
            "safety_cvae_trained_or_modified": False,
            "safety_head_weights_frozen": final["safety_head"]["frozen"],
            "development4_run": False,
            "confirmatory41_run": False,
            "simulator_or_policy_rollouts_run": False,
            "privileged_inputs_used": bool(contract["prohibited_inputs_used"]),
            "rgb_added": False,
            "pushed": False,
        },

        "artifacts": {
            "final_decision_md":
                "docs/CAUSAL_PARKED_SKIN_REFERENCE_V1_FINAL_DECISION.md",
            "final_decision_json":
                "diagnostics_output/causal_parked_skin_reference_v1/final_decision.json",
            "input_contract_audit":
                "diagnostics_output/causal_parked_skin_reference_v1/"
                "input_contract_audit.json",
            "partition_independence":
                "diagnostics_output/causal_parked_skin_reference_v1/"
                "partition_independence.json",
            "selection":
                "diagnostics_output/causal_parked_skin_reference_v1/selection.json",
            "final_training":
                "diagnostics_output/causal_parked_skin_reference_v1/final_training.json",
            "package": "causal_parked_skin/",
            "tests": "tests/test_causal_parked_skin_reference.py",
        },
        "report_hashes": {
            "input_contract_sha256": contract["report_sha256"],
            "partition_independence_sha256": partition["report_sha256"],
            "selection_sha256": selection["report_sha256"],
            "final_training_sha256": final["report_sha256"],
            "dataset_decision_sha256": dataset["final_decision_sha256"],
        },
    }
    payload["final_decision_sha256"] = canonical_hash(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    print(f"decision       : {decision}")
    print(f"frozen primary : {resolved['frozen_primary']}  ({resolved['branch']})")
    print("technical gates:")
    for gate in resolved["gates"]["technical"]:
        print(f"  [{'PASS' if gate['passed'] else 'FAIL'}] {gate['gate']}")
    print("generalization gates:")
    for gate in resolved["gates"]["generalization"]:
        print(f"  [{'PASS' if gate['passed'] else 'FAIL'}] {gate['gate']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
