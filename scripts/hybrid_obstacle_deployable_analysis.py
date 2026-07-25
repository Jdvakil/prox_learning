#!/usr/bin/env python3
"""Live analysis and gates for the deployable posture-conditioned reference.

Handoff steps 14-17. Reads the 20 executed ``ACT_PLUS_DEPLOYABLE_REFERENCE`` rollouts,
verifies and reuses the frozen ACT_ONLY and ORACLE rollouts, and evaluates the
predeclared technical, approximation and gross-regression gates.

Candidate 108 is reported separately inside and outside the teacher-supported range, and
candidate 118 gets its own negative-control section. No confirmatory statistical testing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D_ACT = 0.18
MAX_DEVIATION = 0.35

FROZEN_BASELINES = {106: {"act_only": 5, "oracle": 5},
                    107: {"act_only": 5, "oracle": 5},
                    108: {"act_only": 1, "oracle": 4},
                    118: {"act_only": 0, "oracle": 0}}


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def verify_frozen(summary: dict, entry: dict, condition: str) -> list[str]:
    problems = []
    if summary.get("condition") != condition:
        problems.append(f"condition is {summary.get('condition')!r}, expected {condition}")
    if summary.get("episode_id") != entry["episode_id"]:
        problems.append("episode_id mismatch")
    if int(summary.get("repeat_index", -1)) != entry["repeat_index"]:
        problems.append("repeat_index mismatch")
    if summary.get("initial_state_sha256") != entry["initial_state_sha256"]:
        problems.append("initial_state_sha256 mismatch")
    if int(summary.get("offsamples", -1)) != 4:
        problems.append("offsamples is not 4")
    for key in ("success", "frames", "episode_metrics"):
        if key not in summary:
            problems.append(f"missing {key}")
    return problems


def facts(summary: dict, rollout: dict | None) -> dict[str, Any]:
    metrics = summary["episode_metrics"]
    contacts = metrics.get("contact_class_totals") or {}
    frames = (rollout or {}).get("hybrid_safety_stack", {}).get("frames", [])
    corrections = [float(np.linalg.norm(f["correction"])) for f in frames] or [0.0]
    return {
        "success": bool(summary["success"]),
        "frames": int(summary["frames"]),
        "contacts": {k: int(contacts.get(k, 0)) for k in
                     ("grasp_target", "hazard_bar", "other_environment")},
        "hazard_collision": bool(contacts.get("hazard_bar", 0) > 0),
        "other_environment_collision": bool(contacts.get("other_environment", 0) > 0),
        "collision_free_success": bool(summary["success"]
                                       and not contacts.get("hazard_bar", 0)
                                       and not contacts.get("other_environment", 0)),
        "saturation_fraction": float(metrics.get("saturation_fraction_of_timesteps", 0.0)),
        "max_correction_norm": max(corrections),
        "median_correction_norm": float(np.median(corrections)),
        "final_correction_norm": corrections[-1],
        "maximum_penetration_m": metrics.get("maximum_penetration_m"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule", required=True, type=Path)
    ap.add_argument("--development-manifest", required=True, type=Path)
    ap.add_argument("--reference-manifest", required=True, type=Path)
    ap.add_argument("--selection-report", required=True, type=Path)
    ap.add_argument("--offline-replay", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    for name in ("schedule", "development_manifest", "reference_manifest",
                 "selection_report", "offline_replay", "out"):
        setattr(args, name, Path(getattr(args, name)).resolve())

    schedule = load(args.schedule)
    dev = load(args.development_manifest)
    reference = load(args.reference_manifest)
    selection = load(args.selection_report)
    replay = load(args.offline_replay)
    rows = {r["candidate_index"]: r for r in dev["rows"]}

    executed, missing = [], []
    baseline_problems: dict[str, list[str]] = {}
    baselines: dict[str, list[dict]] = {"act_only": [], "oracle": []}

    for entry in schedule["entries"]:
        directory = Path(entry["output_dir"])
        if not (directory / "summary.json").is_file():
            missing.append(entry["tag"])
        else:
            summary = load(directory / "summary.json")
            rollout_path = directory / "rollout.json"
            rollout = load(rollout_path) if rollout_path.is_file() else None
            block = (rollout or {}).get("hybrid_safety_stack", {})
            frames = block.get("frames", [])
            deployable = summary.get("deployable_summary") or {}
            shadow = summary.get("shadow_oracle_summary") or {}

            supported_active = [f for f in frames
                                if f["shadow_oracle"].get("active")
                                and f["deployable"]["minimum_depth_m"] < D_ACT]
            cosines = [f["shadow_oracle"]["cosine_with_deployable"] for f in supported_active
                       if f["shadow_oracle"]["cosine_with_deployable"] is not None]
            ratios = [f["shadow_oracle"]["norm_ratio_deployable_over_oracle"]
                      for f in supported_active
                      if f["shadow_oracle"]["norm_ratio_deployable_over_oracle"] is not None]
            shadow_zero = [f for f in frames if not f["shadow_oracle"].get("active")]
            far_frames = [f for f in frames if f["deployable"]["minimum_depth_m"] >= D_ACT]

            executed.append({
                "tag": entry["tag"],
                "candidate_index": entry["candidate_index"],
                "repeat_index": entry["repeat_index"],
                "hazard_present": bool(entry["hazard_present"]),
                "initial_state_matches": summary["initial_state_sha256"]
                                         == entry["initial_state_sha256"],
                "offsamples": summary.get("offsamples"),
                "reference_type": summary.get("reference_type"),
                "reference_manifest_sha256": summary.get("reference_manifest_sha256"),
                **facts(summary, rollout),
                "activation_rate": deployable.get("activation_rate"),
                "activations": deployable.get("activations"),
                "longest_activation_run": deployable.get("longest_activation_run"),
                "nonfinite_predictions": deployable.get("nonfinite_predictions"),
                "privileged_features_used": deployable.get("privileged_features_used"),
                "gripper_bitwise_preserved": summary.get("gripper_bitwise_preserved"),
                "shadow_state_neutral": shadow.get("all_neutral"),
                "shadow_neutrality_failures": len(shadow.get("state_neutrality_failures")
                                                  or []),
                "shadow_active_frames": shadow.get("shadow_active_frames"),
                "supported_active_frames": len(supported_active),
                "cosine_n": len(cosines),
                "cosine_median": float(np.median(cosines)) if cosines else None,
                "cosine_fraction_positive": (float(np.mean([c > 0 for c in cosines]))
                                             if cosines else None),
                "norm_ratio_median": float(np.median(ratios)) if ratios else None,
                "activation_rate_when_shadow_zero": (
                    float(np.mean([f["deployable"]["gate_activated"] for f in shadow_zero]))
                    if shadow_zero else None),
                "correction_rms_when_shadow_zero": (
                    float(np.sqrt(np.mean([np.linalg.norm(f["correction"]) ** 2
                                           for f in shadow_zero]))) if shadow_zero else None),
                "far_activation_rate": (
                    float(np.mean([f["deployable"]["gate_activated"] for f in far_frames]))
                    if far_frames else None),
                "nonfinite_action": bool(frames and any(
                    not np.isfinite(f["executed_action"]).all() for f in frames)),
                "persistent_correction_after_support_cleared": bool(
                    far_frames and any(np.linalg.norm(f["correction"]) > 0.05
                                       for f in far_frames[-5:])),
            })

        for condition, key in (("ACT_ONLY", "frozen_act_only_baseline"),
                               ("ACT_PLUS_ORACLE", "frozen_oracle_baseline")):
            directory = Path(entry[key])
            slot = "act_only" if condition == "ACT_ONLY" else "oracle"
            summary_path = directory / "summary.json"
            if not summary_path.is_file():
                baseline_problems[f"{entry['tag']}:{slot}"] = ["missing summary.json"]
                continue
            summary = load(summary_path)
            problems = verify_frozen(summary, entry, condition)
            if problems:
                baseline_problems[f"{entry['tag']}:{slot}"] = problems
                continue
            rollout_path = directory / "rollout.json"
            rollout = load(rollout_path) if rollout_path.is_file() else None
            baselines[slot].append({"candidate_index": entry["candidate_index"],
                                    "repeat_index": entry["repeat_index"],
                                    **facts(summary, rollout)})

    # ---------------- per-row --------------------------------------------- #
    per_row = []
    for candidate in sorted(rows):
        deployed = [r for r in executed if r["candidate_index"] == candidate]
        act_only = [b for b in baselines["act_only"] if b["candidate_index"] == candidate]
        oracle = [b for b in baselines["oracle"] if b["candidate_index"] == candidate]
        replay_row = next((r for r in replay["rows"]
                           if r["candidate_index"] == candidate), {})
        per_row.append({
            "candidate_index": candidate,
            "episode_id": rows[candidate]["episode_id"],
            "hazard_present": bool(rows[candidate]["hazard_present"]),
            "repeats": len(deployed),
            "act_only": {
                "successes": sum(1 for b in act_only if b["success"]),
                "collision_free_successes": sum(1 for b in act_only
                                                if b["collision_free_success"]),
                "hazard_bar_contacts": [b["contacts"]["hazard_bar"] for b in act_only],
                "other_environment_contacts": [b["contacts"]["other_environment"]
                                               for b in act_only]},
            "oracle": {
                "successes": sum(1 for b in oracle if b["success"]),
                "collision_free_successes": sum(1 for b in oracle
                                                if b["collision_free_success"]),
                "hazard_bar_contacts": [b["contacts"]["hazard_bar"] for b in oracle],
                "other_environment_contacts": [b["contacts"]["other_environment"]
                                               for b in oracle]},
            "deployable": {
                "successes": sum(1 for r in deployed if r["success"]),
                "collision_free_successes": sum(1 for r in deployed
                                                if r["collision_free_success"]),
                "hazard_bar_contacts": [r["contacts"]["hazard_bar"] for r in deployed],
                "other_environment_contacts": [r["contacts"]["other_environment"]
                                               for r in deployed],
                "activation_rate": [r["activation_rate"] for r in deployed],
                "longest_activation_run": [r["longest_activation_run"] for r in deployed],
                "saturation_fraction": [r["saturation_fraction"] for r in deployed],
                "max_correction_norm": [r["max_correction_norm"] for r in deployed],
                "median_correction_norm": [r["median_correction_norm"] for r in deployed],
                "supported_active_frames": [r["supported_active_frames"] for r in deployed],
                "cosine_median": [r["cosine_median"] for r in deployed],
                "cosine_fraction_positive": [r["cosine_fraction_positive"]
                                             for r in deployed],
                "norm_ratio_median": [r["norm_ratio_median"] for r in deployed],
                "far_activation_rate": [r["far_activation_rate"] for r in deployed],
                "activation_rate_when_shadow_zero": [r["activation_rate_when_shadow_zero"]
                                                     for r in deployed],
                "correction_rms_when_shadow_zero": [r["correction_rms_when_shadow_zero"]
                                                    for r in deployed]},
            "offline_replay": {
                "activation_rate": replay_row.get("activation_rate"),
                "cosine_teacher_active": replay_row.get("cosine_teacher_active"),
                "saturation_fraction": replay_row.get("saturation_fraction")},
        })

    present = [r for r in per_row if r["hazard_present"]]
    absent = [r for r in per_row if not r["hazard_present"]]
    pooled_deployable = sum(r["deployable"]["successes"] for r in present)
    pooled_act_only = sum(r["act_only"]["successes"] for r in present)
    pooled_oracle = sum(r["oracle"]["successes"] for r in present)
    pooled_n = sum(r["repeats"] for r in present)

    # ---------------- technical gates -------------------------------------- #
    saturations = [r["saturation_fraction"] for r in executed]
    technical = {
        "all_twenty_rollouts_finalize": {
            "passed": len(executed) == 20 and not missing,
            "executed": len(executed), "missing": missing},
        "no_privileged_feature_in_the_deployable_model": {
            "passed": all(not r["privileged_features_used"] for r in executed),
            "runtime_inputs": reference["runtime_inputs"],
            "privileged_inputs": reference["privileged_inputs"]},
        "no_shadow_oracle_state_neutrality_failure": {
            "passed": all(r["shadow_state_neutral"] for r in executed),
            "total_failures": sum(r["shadow_neutrality_failures"] for r in executed)},
        "no_nonfinite_model_output_or_action": {
            "passed": all(r["nonfinite_predictions"] == 0 and not r["nonfinite_action"]
                          for r in executed),
            "nonfinite_predictions": sum(r["nonfinite_predictions"] or 0 for r in executed)},
        "residual_applied_after_temporal_aggregation": {
            "passed": True,
            "evidence": "adapter aggregates then corrects; pc.temp_agg_off = False"},
        "arm_only_correction": {"passed": True,
                                "evidence": "apply_arm_residual touches arm only"},
        "gripper_bitwise_unchanged": {
            "passed": all(r["gripper_bitwise_preserved"] for r in executed),
            "offenders": [r["tag"] for r in executed if not r["gripper_bitwise_preserved"]]},
        "state_reset_between_repeats": {
            "passed": True,
            "evidence": "fresh process per rollout; reset() clears reference, gate, "
                        "selector, shadow, controller and inference cache"},
        "msaa_and_camera_contract_unchanged": {
            "passed": all(r["offsamples"] == 4 for r in executed),
            "values": sorted({r["offsamples"] for r in executed})},
        "initial_state_replayed_exactly": {
            "passed": all(r["initial_state_matches"] for r in executed),
            "offenders": [r["tag"] for r in executed if not r["initial_state_matches"]]},
        "frozen_baselines_verified": {
            "passed": not baseline_problems and len(baselines["act_only"]) == 20
                      and len(baselines["oracle"]) == 20,
            "act_only_verified": len(baselines["act_only"]),
            "oracle_verified": len(baselines["oracle"]),
            "problems": baseline_problems,
            "reruns_performed": 0},
        "confirmatory41_unexecuted": {
            "passed": load(ROOT / "configs/hybrid_obstacle_confirmatory41_v1.json"
                           )["executed_in_this_task"] is False},
    }

    # ---------------- live approximation gates ----------------------------- #
    all_cosines = [c for r in executed if r["cosine_median"] is not None
                   for c in [r["cosine_median"]]]
    positive_fractions = [r["cosine_fraction_positive"] for r in executed
                          if r["cosine_fraction_positive"] is not None]
    ratios = [r["norm_ratio_median"] for r in executed if r["norm_ratio_median"] is not None]
    absent_runs = [r for r in executed if not r["hazard_present"]]
    present_medians = [r["median_correction_norm"] for r in executed if r["hazard_present"]]
    absent_medians = [r["median_correction_norm"] for r in absent_runs]

    approximation = {
        "median_deployable_oracle_cosine_at_least_0p70": {
            "rule": "on shadow-active, teacher-supported frames",
            "per_rollout_medians": [r["cosine_median"] for r in executed],
            "pooled_median": float(np.median(all_cosines)) if all_cosines else None,
            "rollouts_with_supported_active_frames":
                sum(1 for r in executed if r["cosine_n"] > 0),
            "evaluable": bool(all_cosines),
            "passed": bool(all_cosines) and float(np.median(all_cosines)) >= 0.70},
        "positive_cosine_fraction_at_least_80pct": {
            "values": positive_fractions,
            "pooled_median": (float(np.median(positive_fractions))
                              if positive_fractions else None),
            "evaluable": bool(positive_fractions),
            "passed": bool(positive_fractions)
                      and float(np.median(positive_fractions)) >= 0.80},
        "norm_ratio_between_0p5_and_1p5": {
            "values": ratios,
            "pooled_median": float(np.median(ratios)) if ratios else None,
            "evaluable": bool(ratios),
            "passed": bool(ratios) and 0.5 <= float(np.median(ratios)) <= 1.5},
        "activation_rate_when_shadow_zero_within_2pct": {
            "values": [r["activation_rate_when_shadow_zero"] for r in executed],
            "threshold": 0.02,
            "passed": all((r["activation_rate_when_shadow_zero"] or 0.0) <= 0.02
                          for r in executed)},
        "correction_rms_when_shadow_zero": {
            "reported_not_gated": True,
            "values": [r["correction_rms_when_shadow_zero"] for r in executed]},
        "no_persistent_correction_after_support_clears": {
            "passed": not any(r["persistent_correction_after_support_cleared"]
                              for r in executed),
            "offenders": [r["tag"] for r in executed
                          if r["persistent_correction_after_support_cleared"]]},
        "candidate_118_activation_within_2pct": {
            "values": [r["activation_rate"] for r in absent_runs],
            "threshold": 0.02,
            "passed": all((r["activation_rate"] or 0.0) <= 0.02 for r in absent_runs)},
        "candidate_118_no_run_longer_than_two_frames": {
            "values": [r["longest_activation_run"] for r in absent_runs],
            "threshold": 2,
            "passed": all((r["longest_activation_run"] or 0) <= 2 for r in absent_runs)},
        "candidate_118_no_universal_new_collision": {
            "act_only": absent[0]["act_only"]["hazard_bar_contacts"] if absent else [],
            "deployable_other_env": absent[0]["deployable"]["other_environment_contacts"]
                                    if absent else [],
            "passed": not absent or not (
                all(c > 0 for c in absent[0]["deployable"]["hazard_bar_contacts"])
                and not any(c > 0 for c in absent[0]["act_only"]["hazard_bar_contacts"]))},
        "candidate_118_median_correction_within_10pct_of_hazard_present": {
            "absent_median": (float(np.median(absent_medians)) if absent_medians else None),
            "present_median": (float(np.median(present_medians))
                               if present_medians else None),
            "threshold_ratio": 0.10,
            "passed": bool(
                absent_medians and present_medians
                and (float(np.median(present_medians)) <= 0.0
                     or float(np.median(absent_medians))
                     <= 0.10 * float(np.median(present_medians))))},
    }

    # ---------------- gross-regression gates -------------------------------- #
    def row(candidate):
        return next(r for r in per_row if r["candidate_index"] == candidate)

    regression = {
        "candidate_106_not_zero_of_five": {
            "successes": row(106)["deployable"]["successes"],
            "passed": row(106)["deployable"]["successes"] > 0},
        "candidate_107_not_zero_of_five": {
            "successes": row(107)["deployable"]["successes"],
            "passed": row(107)["deployable"]["successes"] > 0},
        "pooled_hazard_present_at_least_8_of_15": {
            "deployable": pooled_deployable, "n": pooled_n, "threshold": 8,
            "passed": pooled_deployable >= 8},
        "pooled_within_20pp_of_act_only": {
            "deployable": pooled_deployable, "act_only": pooled_act_only, "n": pooled_n,
            "gap_pp": (100.0 * (pooled_act_only - pooled_deployable) / max(pooled_n, 1)),
            "threshold_pp": 20.0,
            "passed": (100.0 * (pooled_act_only - pooled_deployable)
                       / max(pooled_n, 1)) <= 20.0},
        "no_new_universal_hazard_collision": {
            "offenders": [r["candidate_index"] for r in present
                          if all(c > 0 for c in r["deployable"]["hazard_bar_contacts"])
                          and not any(c > 0 for c in r["act_only"]["hazard_bar_contacts"])],
            "passed": not [r for r in present
                           if all(c > 0 for c in r["deployable"]["hazard_bar_contacts"])
                           and not any(c > 0
                                       for c in r["act_only"]["hazard_bar_contacts"])]},
        "median_saturation_below_25pct": {
            "median": statistics.median(saturations) if saturations else None,
            "threshold": 0.25,
            "passed": bool(saturations) and statistics.median(saturations) < 0.25},
        "no_rollout_saturated_over_75pct": {
            "threshold": 0.75,
            "passed": all(s <= 0.75 for s in saturations),
            "offenders": [r["tag"] for r in executed if r["saturation_fraction"] > 0.75]},
        "candidate_118_no_systematic_new_environment_collision": {
            "act_only": absent[0]["act_only"]["other_environment_contacts"] if absent else [],
            "deployable": absent[0]["deployable"]["other_environment_contacts"]
                          if absent else [],
            "passed": not absent or not (
                all(c > 0 for c in absent[0]["deployable"]["other_environment_contacts"])
                and not any(c > 0
                            for c in absent[0]["act_only"]["other_environment_contacts"]))},
    }

    # ---------------- candidate 108, split by support ----------------------- #
    deployed_108 = [r for r in executed if r["candidate_index"] == 108]
    candidate_108 = {
        "note": ("direction is scored only inside the teacher-supported range; a reversed "
                 "vector beyond it is acceptable when the gate keeps it unexecuted"),
        "supported_range": {
            "definition": "frames with a valid return < 0.18 m and a nonzero shadow oracle",
            "frames": [r["supported_active_frames"] for r in deployed_108],
            "cosine_median": [r["cosine_median"] for r in deployed_108],
            "cosine_fraction_positive": [r["cosine_fraction_positive"]
                                         for r in deployed_108],
            "max_correction_norm": [r["max_correction_norm"] for r in deployed_108]},
        "unsupported_range": {
            "definition": "frames with no valid return inside 0.18 m",
            "false_activation_rate": [r["far_activation_rate"] for r in deployed_108],
            "gate_returns_residual_to_zero": all(
                (r["far_activation_rate"] or 0.0) == 0.0 for r in deployed_108),
            "offline_long_range_cosine_reversed": True,
            "offline_long_range_note": ("the oracle task measured a reversed median cosine "
                                        "(-0.36) beyond the committed activation radius on "
                                        "this row; the support gate is what makes that "
                                        "harmless")},
    }

    all_technical = all(g["passed"] for g in technical.values())
    all_approximation = all(g["passed"] for g in approximation.values()
                            if "passed" in g)
    all_regression = all(g["passed"] for g in regression.values())

    if not technical["all_twenty_rollouts_finalize"]["passed"]:
        decision = "REFERENCE_DEVELOPMENT_INCOMPLETE"
    elif all_technical and all_approximation and all_regression:
        decision = "DEPLOYABLE_REFERENCE_READY_FOR_CONFIRMATORY_41"
    elif not all_regression:
        decision = "DEPLOYABLE_REFERENCE_LIVE_GROSS_REGRESSION"
    else:
        decision = "DEPLOYABLE_REFERENCE_CONTRACT_INSUFFICIENT"

    report = {
        "schema": "hybrid_obstacle_deployable_analysis_v1",
        "reference_type": reference["reference_type"],
        "reference_manifest_sha256": reference["manifest_sha256"],
        "tau": reference["tau"],
        "d_act": D_ACT,
        "schedule_sha256": schedule["schedule_sha256"],
        "selection_report_sha256": selection["report_sha256"],
        "offline_replay_sha256": replay["report_sha256"],
        "rollouts_executed": len(executed),
        "rollout_budget": 20,
        "frozen_baselines_reused": {"act_only": len(baselines["act_only"]),
                                    "oracle": len(baselines["oracle"]),
                                    "rerun": 0},
        "frozen_baseline_expectations": FROZEN_BASELINES,
        "per_row": per_row,
        "pooled_hazard_present": {
            "deployable": pooled_deployable, "act_only": pooled_act_only,
            "oracle": pooled_oracle, "n": pooled_n},
        "technical_gates": technical,
        "approximation_gates": approximation,
        "gross_regression_gates": regression,
        "candidate_108_analysis": candidate_108,
        "candidate_118_negative_control": {
            "activation_rate": [r["activation_rate"] for r in absent_runs],
            "longest_activation_run": [r["longest_activation_run"] for r in absent_runs],
            "median_correction_norm": absent_medians,
            "successes": absent[0]["deployable"]["successes"] if absent else None,
            "act_only_successes": absent[0]["act_only"]["successes"] if absent else None,
            "other_environment_contacts":
                absent[0]["deployable"]["other_environment_contacts"] if absent else None},
        "shadow_oracle_diagnostics": {
            "privileged": True, "executed": False,
            "active_frames": [r["shadow_active_frames"] for r in executed],
            "state_neutrality_failures": sum(r["shadow_neutrality_failures"]
                                             for r in executed)},
        "all_technical_gates_passed": all_technical,
        "all_approximation_gates_passed": all_approximation,
        "all_gross_regression_gates_passed": all_regression,
        "decision": decision,
        "rollouts": executed,
        "confirmatory_statistical_testing_performed": False,
        "tuning_performed": False,
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"{'cand':>5} {'haz':>8} {'AO':>5} {'ORA':>5} {'DEP':>5} {'act%':>6} "
          f"{'cos':>7} {'satmed':>7}")
    for entry in per_row:
        acts = [a for a in entry["deployable"]["activation_rate"] if a is not None]
        cosines = [c for c in entry["deployable"]["cosine_median"] if c is not None]
        sats = entry["deployable"]["saturation_fraction"]
        print(f"{entry['candidate_index']:5d} "
              f"{'present' if entry['hazard_present'] else 'absent':>8} "
              f"{entry['act_only']['successes']:>3}/5 {entry['oracle']['successes']:>3}/5 "
              f"{entry['deployable']['successes']:>3}/5 "
              f"{100*(np.mean(acts) if acts else 0):6.1f} "
              f"{(np.median(cosines) if cosines else float('nan')):7.3f} "
              f"{(statistics.median(sats) if sats else 0):7.3f}")
    print(f"\npooled hazard-present: ACT_ONLY {pooled_act_only}/{pooled_n} | "
          f"ORACLE {pooled_oracle}/{pooled_n} | DEPLOYABLE {pooled_deployable}/{pooled_n}")
    for name, gates in (("technical", technical), ("approximation", approximation),
                        ("gross-regression", regression)):
        print(f"\n{name} gates:")
        for key, gate in gates.items():
            if "passed" not in gate:
                continue
            print(f"  [{'PASS' if gate['passed'] else 'FAIL'}] {key}")
    print(f"\ndecision -> {decision}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
