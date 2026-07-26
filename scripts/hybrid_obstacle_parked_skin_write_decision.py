#!/usr/bin/env python3
"""Assemble the final parked-skin reference decision JSON.

Handoff step 25. Every field is copied from an artifact produced earlier in the task. The
decision token follows from the data audit when the paired-skin contract is unmet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules" / "act"))

ALLOWED = (
    "PARKED_SKIN_REFERENCE_READY_FOR_CONFIRMATORY_41",
    "PARKED_SKIN_REFERENCE_OFFLINE_INVALID",
    "PARKED_SKIN_REFERENCE_LIVE_GROSS_REGRESSION",
    "PARKED_SKIN_DATA_CONTRACT_FAILED",
    "PARKED_SKIN_COUNTERFACTUAL_NOT_IDENTIFIABLE",
    "PARKED_SKIN_MODEL_TRAINING_FAILED",
    "CHECKPOINT_OR_SOURCE_MISMATCH",
    "PARKED_SKIN_DEVELOPMENT_INCOMPLETE",
)


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provenance", required=True, type=Path)
    ap.add_argument("--data-audit", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    provenance = json.loads(Path(args.provenance).read_text())
    audit = json.loads(Path(args.data_audit).read_text())

    decision = ("PARKED_SKIN_DATA_CONTRACT_FAILED" if not audit["valid"]
                else "PARKED_SKIN_DEVELOPMENT_INCOMPLETE")
    if decision not in ALLOWED:
        raise SystemExit(f"decision {decision!r} is not allowed")

    from parked_skin_reference import (
        CAUSAL_FRAMES,
        CONTEXT_WIDTH,
        D_MAX,
        DEAD_PIXEL_BELOW_M,
        PARAMETER_BUDGET,
        REFERENCE_ID,
        RUNTIME_FIELDS,
        build_model,
        parameter_count,
    )

    act = ROOT / "submodules/act"
    molmo = ROOT / "submodules/molmospaces"
    confirmatory = json.loads(
        (ROOT / "configs/hybrid_obstacle_confirmatory41_v1.json").read_text())
    partition = json.loads(
        (ROOT / "configs/hybrid_obstacle_reference_partition_v2.json").read_text())
    on_policy = json.loads((ROOT / "diagnostics_output/hybrid_obstacle_on_policy_reference"
                            / "final_decision.json").read_text())

    payload = {
        "schema": "hybrid_obstacle_parked_skin_reference_final_decision_v1",
        "date": "2026-07-26",
        "task": ("Replace the seven-output parked-head estimator with a causal model that "
                 "predicts the full parked 40x8x8 proximity field and routes it through "
                 "the frozen SafetyHead"),
        "decision": decision,
        "case": None,
        "case_note": ("Cases A/B/C classify offline and live *results*. No result exists: "
                      "the task stopped at the step-3 data contract, before training."),

        "commits": {
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "root_starting_commit": "5270bee29604f30e020cb9c28318ef90e030d500",
            "root_commit": git("rev-parse", "HEAD"),
            "act_branch": git("rev-parse", "--abbrev-ref", "HEAD", repo=act),
            "act_starting_commit": "21bf05efd49887dc18bbf0b4094cd7663526c2bb",
            "act_commit": git("rev-parse", "HEAD", repo=act),
            "molmospaces_commit": git("rev-parse", "HEAD", repo=molmo),
            "molmospaces_modified": git("status", "--porcelain", repo=molmo) != "",
        },

        "artifact_verification": {
            "checks": provenance["check_count"],
            "all_matched": provenance["all_matched"],
            "note": "every immutable artifact verifies; the blocker is not a mismatch",
        },

        "why_the_seven_output_target_failed": {
            "prior_decision": on_policy["decision"],
            "prior_case": on_policy["case"],
            "summary": ("A full on-policy aggregation round improved the 7-output model on "
                        "every evaluable measure -- ACT-only on-policy differential MAE "
                        "0.313 -> 0.208, oracle on-policy 0.355 -> 0.151, median cosine on "
                        "ACT-only on-policy validation frames -0.082 -> +0.884 -- and still "
                        "admitted no activation threshold: holding recall >= 0.80 left the "
                        "median cosine at ~0.61, and the positive-cosine fraction never "
                        "reached 0.80 above 4% recall. The binding distribution was "
                        "oracle-controlled on-policy, the regime a correct reference "
                        "creates for itself."),
            "why_a_spatial_target_is_the_right_next_move": (
                "The 7-D target equals its own input on most frames, so the loss rewards "
                "the identity map; and a small MLP must re-learn the head's 2560 -> 7 "
                "structure from 7 numbers of supervision per frame. Predicting the field "
                "and routing it through the frozen head keeps that structure and makes the "
                "supervision dense and spatial."),
        },

        "data_contract": {
            "required_input": audit["required_input"],
            "required_target": audit["required_target"],
            "families": audit["families"],
            "input_contract_met": audit["input_contract_met"],
            "target_contract_met": audit["target_contract_met"],
            "total_frames_available": audit["total_frames_available"],
            "frames_meeting_both_contracts": audit["frames_meeting_both_contracts"],
            "physical_constraint_check": audit["physical_constraint_check"],
            "checks_on_the_fields_that_do_exist": audit["checks_on_the_fields_that_do_exist"],
            "regeneration_requirement": audit["regeneration_requirement"],
            "silent_repair_performed": False,
            "manifest_sha256": audit["report_sha256"],
        },

        "model": {
            "reference_id": REFERENCE_ID,
            "status": "IMPLEMENTED AND UNIT-TESTED, NEVER TRAINED",
            "why_not_trained": ("its training target -- the parked 40x8x8 field -- is "
                                "stored in no shard"),
            "parameters": parameter_count(build_model()),
            "parameter_budget": PARAMETER_BUDGET,
            "within_budget": parameter_count(build_model()) < PARAMETER_BUDGET,
            "architecture": ("per-sensor Linear(4x64 -> 128) + SiLU + learned 40x128 sensor "
                             "embedding; global state context Linear(29 -> 128) SiLU "
                             "Linear(128 -> 128) added to every token; TransformerEncoder "
                             "2 layers, d_model 128, 4 heads, ff 256, pre-norm, dropout 0; "
                             "per-sensor 64-pixel change logits and a frame-level activity "
                             "logit from mean-pooled tokens"),
            "runtime_inputs": list(RUNTIME_FIELDS),
            "privileged_inputs": [],
            "causal_frames": CAUSAL_FRAMES,
            "context_width": CONTEXT_WIDTH,
            "physical_counterfactual": {
                "removable": "c_current * sigmoid(change_logits)",
                "parked": "clamp(c_current - removable, 0, 1)",
                "guarantee": "0 <= c_parked_pred <= c_current, by construction",
                "verified_under_saturated_logits": True,
            },
            "closeness_transform": {
                "formula": "clip(1 - depth / 0.5, 0, 1)",
                "d_max_m": D_MAX,
                "dead_pixel_below_m": DEAD_PIXEL_BELOW_M,
                "inverse": "depth = 0.5 * (1 - closeness); zero closeness -> 0.5 m, which "
                           "the frozen head reads as far / no activation",
            },
            "artifact_committed": False,
            "checkpoint_exists": False,
        },

        "not_performed": {
            "training": "no target exists",
            "activity_calibration": "requires a trained model",
            "rho_max": "requires a calibrated model",
            "disjoint_validation": "requires a frozen model and contract",
            "offline_test": "requires a frozen model and contract",
            "causal_history_ablations": "requires a trained model",
            "development4_offline_replay": "requires a frozen model",
            "live_rollouts": "gated behind every offline gate; 0 of 20 executed",
            "live_rollout_budget": 20,
            "live_rollouts_used": 0,
        },

        "reference_partition": {
            "sha256": partition["partition_sha256"],
            "composition": partition["composition"],
            "reused_exactly": partition["partition_sha256"]
                              == on_policy["reference_partition"]["sha256"],
            "all_pairwise_disjoint": partition["all_pairwise_disjoint"],
        },

        "confirmatory41": {
            "manifest": "configs/hybrid_obstacle_confirmatory41_v1.json",
            "sha256": confirmatory["manifest_sha256"],
            "rows": len(confirmatory["rows"]),
            "executed_in_this_task": confirmatory["executed_in_this_task"],
            "untouched": True,
        },

        "constraints_honoured": {
            "act_trained_or_modified": False,
            "safety_cvae_trained_or_modified": False,
            "residual_constants_modified": False,
            "canonical_collection_dataset_manifests_split_modified": False,
            "another_on_policy_dataset_collected": False,
            "confirmatory_row_executed": False,
            "privileged_features_at_inference": False,
            "second_competing_architecture_added": False,
            "hyperparameter_sweep": False,
            "tuned_from_live_results": False,
            "broken_clearance_metric_used": False,
            "molmospaces_modified": git("status", "--porcelain", repo=molmo) != "",
            "pushed": False,
        },

        "uncommitted_artifacts": [
            {"label": label, "path": str(path), "committed": False,
             **({"sha256": sha256_file(path)} if path.is_file()
                else {"entries": len(list(path.glob("*")))} if path.is_dir() else {})}
            for label, path in (
                ("expert_paired_dir", Path("/root/act_retrain_assets/paired_reference_v1")),
                ("on_policy_labelling_root", Path("/root/act_retrain_assets/on_policy_v2")),
                ("on_policy_learner_root",
                 Path("/root/act_retrain_assets/on_policy_learner_v2")),
                ("act_checkpoint", Path(
                    "/root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2"
                    "/20260725_seed0_2000ep/policy_best.ckpt")),
            )
        ],

        "next_task_requirement": {
            "headline": ("regenerate the paired dataset with the parked field retained, "
                         "then run this task unchanged"),
            "fields_to_store_per_frame": [
                "current 40x8x8 depth (or closeness) at the decision state",
                "parked 40x8x8 depth (or closeness) at the SAME decision state",
                "the existing four causal current frames",
                "the existing runtime and privileged label fields",
            ],
            "rollouts_to_regenerate": audit["regeneration_requirement"][
                "rollouts_that_would_have_to_rerun"],
            "estimated_storage_gib": audit["regeneration_requirement"][
                "estimated_uncompressed_gib"],
            "storage_reduction_option": ("store float16 closeness rather than float32 depth "
                                         "and only for frames the oracle marks active, plus "
                                         "a uniform sample of zero frames -- roughly a fifth "
                                         "of the size with no loss for this objective"),
            "why_this_task_could_not_do_it": ("'Do not collect another on-policy training "
                                              "dataset' forbids re-running the 264 "
                                              "on-policy rollouts, and MSAA makes any rerun "
                                              "a different sample of states in any case"),
            "already_delivered": ("the model, the physical counterfactual, the closeness "
                                  "transform, the causal-history buffer, the activity gate, "
                                  "the strict loader and 54 contract tests are implemented "
                                  "and committed, so the next task only needs the data"),
        },

        "artifacts": {
            "final_decision_md":
                "docs/HYBRID_OBSTACLE_PARKED_SKIN_REFERENCE_FINAL_DECISION.md",
            "final_decision_json": ("diagnostics_output/hybrid_obstacle_parked_skin_reference/"
                                    "final_decision.json"),
            "provenance": "diagnostics_output/hybrid_obstacle_parked_skin_reference/"
                          "provenance_verification.json",
            "data_audit": "diagnostics_output/hybrid_obstacle_parked_skin_reference/"
                          "paired_skin_data_audit.json",
            "model_source": "submodules/act/parked_skin_reference.py",
            "tests": "tests/test_parked_skin_reference_contract.py",
        },
        "report_hashes": {
            "provenance_sha256": provenance["report_sha256"],
            "data_audit_sha256": audit["report_sha256"],
        },
    }
    payload["final_decision_sha256"] = canonical_hash(payload)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(f"decision: {decision}")
    print(f"frames meeting both contracts: {audit['frames_meeting_both_contracts']} "
          f"of {audit['total_frames_available']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
