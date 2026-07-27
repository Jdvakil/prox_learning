#!/usr/bin/env python3
"""Assemble the final decision for the activity-identifiability audit.

Handoff steps 11-12. The token follows from the recorded reports by the predeclared rules;
every trigger is evaluated explicitly and recorded with its evidence, including the ones
that did not fire, so the reasoning is auditable rather than asserted.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin import threshold as thr

ALLOWED = (
    "EPISTEMIC_UNCERTAINTY_SIGNAL_PRESENT",
    "CURRENT_OBSERVATION_NOT_IDENTIFIABLE",
    "UNCERTAINTY_SIGNAL_INCONCLUSIVE",
    "CHECKPOINT_OR_SOURCE_MISMATCH",
    "ACTIVITY_IDENTIFIABILITY_AUDIT_INCOMPLETE",
)

# A metric passes the letter of the separability gate if it rejects the failures while
# retaining active recall. That says nothing about how many *quiet* frames it also
# rejects, so a metric whose median on ordinary zero frames sits above its median on the
# historical failures is flagged: it would abstain almost everywhere.
def useful_metrics(ensemble: dict) -> tuple[list, list]:
    useful, degenerate = [], []
    for name in ensemble["metrics_satisfying_gate"]:
        block = ensemble["metrics"][name]
        groups = block["by_group"]
        historical = groups["A_HISTORICAL_FALSE_POSITIVE"]["median"]
        quiet = [groups[g]["median"] for g in
                 ("B_ONSET_ZERO", "C_LATE_ZERO", "F_HAZARD_ABSENT_ZERO")
                 if groups[g]["median"] is not None]
        higher_is_uncertain = block["higher_value_means_more_uncertain"]
        # separation must run the right way for the failures relative to quiet frames
        separates = all((q < historical) if higher_is_uncertain else (q > historical)
                        for q in quiet)
        (useful if separates else degenerate).append(name)
    return useful, degenerate


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("provenance", "groups", "ensemble", "collision", "onset-tail", "out"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    args = ap.parse_args()
    for field in ("provenance", "groups", "ensemble", "collision", "onset_tail", "out"):
        setattr(args, field, Path(getattr(args, field)).resolve())

    import numpy
    import torch

    provenance = json.loads(args.provenance.read_text())
    groups = json.loads(args.groups.read_text())
    ensemble = json.loads(args.ensemble.read_text())
    collision = json.loads(args.collision.read_text())
    onset_tail = json.loads(args.onset_tail.read_text())

    useful, degenerate = useful_metrics(ensemble)
    collisions = collision["collisions"]
    neighbours = collision["neighbours"]

    # ---- evaluate every predeclared trigger, firing or not -------------------------
    input_spaces = ("A_CURRENT_PROX_RAW", "B_CURRENT_PROX_EMBEDDING",
                    "C_FULL_DEPLOYABLE_INPUT")
    historical_opposite = {
        space: neighbours[space]["by_group"]["HISTORICAL_FALSE_POSITIVE"]["k=8"][
            "mean_opposite_label_fraction"] for space in neighbours}
    historical_entropy = {
        space: neighbours[space]["by_group"]["HISTORICAL_FALSE_POSITIVE"]["k=8"][
            "median_label_entropy"] for space in neighbours}
    ratio = onset_tail["onset_vs_late"]["separation_ratio"]

    best = max(useful, key=lambda n: ensemble["metrics"][n]["auroc_historical_vs_active"],
               default=None)
    best_block = ensemble["metrics"][best] if best else None

    not_identifiable_triggers = {
        "exact_or_near_identical_inputs_with_opposite_labels": {
            "fired": bool(collisions["exact_full_input_count"]
                          or collisions["near_identity_count"]),
            "evidence": {
                "exact_full_input": collisions["exact_full_input_count"],
                "exact_current_prox": collisions["exact_current_prox_count"],
                "near_identity": collisions["near_identity_count"],
                "tolerances": collisions["tolerances"]},
        },
        "historical_failures_show_low_ensemble_disagreement": {
            "fired": best is None,
            "evidence": {
                "metrics_separating_failures_from_quiet_frames": useful,
                "best_metric": best,
                "best_auroc": best_block["auroc_historical_vs_active"] if best else None,
                "historical_median": best_block["by_group"][
                    "A_HISTORICAL_FALSE_POSITIVE"]["median"] if best else None,
                "late_active_median": best_block["by_group"]["E_LATE_ACTIVE"]["median"]
                if best else None},
        },
        "high_neighbour_label_entropy_around_the_failures": {
            "fired": any((historical_opposite[s] or 0) > 0.25 for s in input_spaces),
            "evidence": {"opposite_label_fraction_k8": historical_opposite,
                         "median_label_entropy_k8": historical_entropy,
                         "input_spaces_considered": list(input_spaces),
                         "note": ("D_FROZEN_MODEL_EMBEDDING is the model's own "
                                  "representation, not the observation contract, so it "
                                  "is reported but not used as an identifiability "
                                  "trigger")},
        },
        "no_uncertainty_metric_rejects_failures_while_retaining_recall": {
            "fired": not useful,
            "evidence": {"gate_operating_point": best_block["gate_operating_point"]
                         if best else None},
        },
        "onset_active_and_zero_states_overlap_materially": {
            "fired": bool(ratio["onset"] < ratio["late"] * 0.9),
            "evidence": {
                "onset_separation_ratio": ratio["onset"],
                "late_separation_ratio": ratio["late"],
                "rule": ("onset overlap counts only if onset separates materially worse "
                         "than late, since late is demonstrably identifiable at AUROC "
                         "~0.998; raw between-class distance alone confuses embedding "
                         "compression with overlap")},
        },
    }
    fired = [k for k, v in not_identifiable_triggers.items() if v["fired"]]

    signal_present_criteria = {
        "a_metric_rejects_at_least_16_of_17": bool(
            best and best_block["gate_operating_point"]["historical_rejected"] >= 16),
        "active_recall_at_least_80pct": bool(
            best and best_block["gate_operating_point"]["active_recall"] >= 0.80),
        "hard_true_active_recall_at_least_80pct": bool(
            best and best_block["gate_operating_point"]["hard_true_active_recall"] >= 0.80),
        "no_substantial_opposite_label_collision_around_failures": bool(
            all((historical_opposite[s] or 0) <= 0.25 for s in input_spaces)),
        "no_exact_full_input_opposite_label_collision": not collisions[
            "any_exact_full_input_collision"],
    }

    if not provenance["all_matched"]:
        decision = "CHECKPOINT_OR_SOURCE_MISMATCH"
    elif fired:
        decision = "CURRENT_OBSERVATION_NOT_IDENTIFIABLE"
    elif all(signal_present_criteria.values()):
        decision = "EPISTEMIC_UNCERTAINTY_SIGNAL_PRESENT"
    else:
        decision = "UNCERTAINTY_SIGNAL_INCONCLUSIVE"
    if decision not in ALLOWED:
        raise SystemExit(f"decision {decision!r} is not allowed")

    payload = {
        "schema": "hybrid_obstacle_activity_identifiability_final_decision_v1",
        "date": "2026-07-27",
        "task": ("Determine whether the remaining activity failure is detectable through "
                 "predictive uncertainty or is an intrinsic identifiability failure of "
                 "the current deployable observation contract"),
        "decision": decision,
        "previous_decision": "PROX_ACTIVITY_GATE_CALIBRATION_INFEASIBLE",
        "previous_case": "B",

        "read_only_task": True,
        "training_performed": False,
        "controller_trained": False,
        "thresholds_changed": False,
        "live_rollouts_run": False,
        "confirmatory41_executed": False,
        "development4_executed": False,
        "seeds_1_2_selected_for_deployment": False,
        "models_ensembled_for_execution": False,
        "checkpoints_altered": False,
        "dataset_modified": False,

        "handoff_commit_discrepancy": {
            "handoff_stated_root_commit": "db326d7",
            "resolves_to_an_object": False,
            "actual_head": git("rev-parse", "HEAD"),
            "actual_previous_commit": "db326d1",
            "assessment": ("one-character difference from the real commit db326d1, which "
                           "is the commit that wrote the previous decision; treated as a "
                           "transcription slip rather than an artifact mismatch, since "
                           "every hash-verifiable artifact was checked independently and "
                           "all matched"),
        },

        "commits": {
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "root_commit": git("rev-parse", "HEAD"),
            "act_commit": git("rev-parse", "HEAD", repo=ROOT / "submodules" / "act"),
            "act_resolved_from_prior_artifacts":
                "91fc42a1bfda2acd4b973bb53549bbf42d1fe9a6",
            "act_modified": git("status", "--porcelain",
                                repo=ROOT / "submodules" / "act") != "",
            "molmospaces_commit": git("rev-parse", "HEAD",
                                      repo=ROOT / "submodules" / "molmospaces"),
            "molmospaces_expected": "678f2eb",
            "molmospaces_modified": git("status", "--porcelain",
                                        repo=ROOT / "submodules" / "molmospaces") != "",
        },

        "provenance": {"checks": provenance["check_count"],
                       "all_matched": provenance["all_matched"],
                       "frozen_model": provenance["frozen_model"]},
        "seed_disposition": ensemble["seed_disposition"],

        "diagnostic_groups": {k: v["count"] for k, v in groups["groups"].items()},
        "group_definitions": {
            "onset": groups["onset_definition"],
            "hard_true_active": groups["hard_true_active_definition"]},

        "ensemble_uncertainty": {
            "frames_scored": ensemble["frames_scored"],
            "model_weights_unchanged": ensemble["model_weights_unchanged"],
            "separability_gates": ensemble["separability_gates"],
            "metrics_satisfying_gate_letter": ensemble["metrics_satisfying_gate"],
            "metrics_separating_failures_from_quiet_frames": useful,
            "metrics_passing_letter_but_degenerate": degenerate,
            "degenerate_explanation": (
                "these reject the failures and retain active recall, but their median on "
                "ordinary quiet frames is on the wrong side of their median on the "
                "failures, so at the operating point they would also abstain on most "
                "genuinely quiet frames"),
            "selected_metric": best,
            "selected_metric_detail": best_block,
            "per_metric": {k: {kk: v[kk] for kk in
                               ("auroc_historical_vs_active", "partial_auroc_fpr_5pct",
                                "average_precision", "gate_satisfied",
                                "gate_operating_point", "historical_in_top")}
                           for k, v in ensemble["metrics"].items()},
        },

        "collisions": collisions,
        "neighbour_ambiguity": {
            "pool": collision["neighbour_pool"],
            "exclusions": collision["exclusions"],
            "feature_spaces": collision["feature_spaces"],
            "opposite_label_fraction_k8_by_group": {
                space: {group: block["k=8"]["mean_opposite_label_fraction"]
                        for group, block in neighbours[space]["by_group"].items()}
                for space in neighbours},
            "historical_frames": {space: neighbours[space]["historical_frames"]
                                  for space in neighbours},
            "asymmetry_finding": (
                "onset-ACTIVE frames sit among opposite-label neighbours most of the "
                "time, while onset-ZERO frames and the 17 failures do not. The ambiguity "
                "runs from active towards zero, which produces false negatives, not the "
                "false positives under investigation"),
        },
        "local_ambiguity": collision["local_ambiguity"],
        "onset_vs_late": onset_tail["onset_vs_late"],
        "score_tail": onset_tail["score_tail"],

        "decision_rules": {
            "not_identifiable_triggers": not_identifiable_triggers,
            "triggers_fired": fired,
            "signal_present_criteria": signal_present_criteria,
            "all_signal_present_criteria_met": all(signal_present_criteria.values()),
        },

        "interpretation": {
            "EPISTEMIC_UNCERTAINTY_SIGNAL_PRESENT": (
                "next task may train one predeclared trajectory-bootstrap ensemble, use "
                "uncertainty only as an abstention signal, retain the seed-0 mean "
                "parked-field model unless a new model contract is explicitly approved, "
                "and qualify on development4 before confirmatory41"),
        }.get(decision),

        "constraints_honoured": {
            "model_trained_or_finetuned": False,
            "seed_1_or_2_selected_for_deployment": False,
            "models_ensembled_for_execution": False,
            "checkpoint_altered": False,
            "act_safety_cvae_or_controller_modified": False,
            "activation_threshold_changed": False,
            "paired_dataset_altered": False,
            "new_simulator_trajectories_generated": False,
            "live_policy_evaluation_run": False,
            "development4_or_confirmatory41_used_for_fitting": False,
            "confirmatory41_executed": False,
            "uncertainty_claimed_merely_because_seeds_differ": False,
            "pushed": False,
        },

        "artifacts": {
            "final_decision_md":
                "docs/HYBRID_OBSTACLE_ACTIVITY_IDENTIFIABILITY_FINAL_DECISION.md",
            "final_decision_json":
                "diagnostics_output/hybrid_obstacle_activity_identifiability/"
                "final_decision.json",
            "provenance": str(args.provenance.relative_to(ROOT)),
            "diagnostic_groups": str(args.groups.relative_to(ROOT)),
            "ensemble_audit": str(args.ensemble.relative_to(ROOT)),
            "collision_audit": str(args.collision.relative_to(ROOT)),
            "onset_tail_audit": str(args.onset_tail.relative_to(ROOT)),
            "tests": "tests/test_activity_identifiability.py",
        },
        "report_hashes": {
            "provenance_sha256": provenance["report_sha256"],
            "groups_sha256": groups["groups_sha256"],
            "ensemble_sha256": ensemble["report_sha256"],
            "collision_sha256": collision["report_sha256"],
            "onset_tail_sha256": onset_tail["report_sha256"],
        },
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": numpy.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    payload["final_decision_sha256"] = thr.canonical_hash(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    print(f"decision : {decision}")
    print("not-identifiable triggers:")
    for name, block in not_identifiable_triggers.items():
        print(f"  [{'FIRED' if block['fired'] else ' no  '}] {name}")
    print("signal-present criteria:")
    for name, met in signal_present_criteria.items():
        print(f"  [{'PASS' if met else 'FAIL'}] {name}")
    print(f"selected metric: {best}")
    print(f"degenerate passes flagged: {degenerate}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
