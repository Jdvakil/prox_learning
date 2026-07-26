#!/usr/bin/env python3
"""Live analysis and gates for the V2 on-policy reference.

Handoff steps 14-16. Reads the 20 live V2 rollouts, verifies and reuses the frozen
ACT-only, oracle and V1 rollouts, and evaluates the predeclared approximation and
gross-regression gates. Classifies the outcome as Case A, B or C.

``minimum_clearance_m`` is never read: a per-geom ``mj_geomDistance`` defect pins it at
<= 0 in every condition. Safety evidence is contact classification only.
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
ORACLE_ZERO_TOLERANCE = 1e-7

FROZEN = {106: {"act_only": 5, "oracle": 5, "v1": 4},
          107: {"act_only": 5, "oracle": 5, "v1": 5},
          108: {"act_only": 1, "oracle": 4, "v1": 3},
          118: {"act_only": 0, "oracle": 0, "v1": 0}}


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def verify_frozen(summary: dict, entry: dict, expected_conditions: tuple[str, ...]
                  ) -> list[str]:
    problems = []
    if summary.get("condition") not in expected_conditions:
        problems.append(f"condition {summary.get('condition')!r} not in {expected_conditions}")
    if summary.get("episode_id") != entry["episode_id"]:
        problems.append("episode_id mismatch")
    if int(summary.get("repeat_index", -1)) != entry["repeat_index"]:
        problems.append("repeat_index mismatch")
    if summary.get("initial_state_sha256") != entry["initial_state_sha256"]:
        problems.append("initial_state_sha256 mismatch")
    if int(summary.get("offsamples", -1)) != 4:
        problems.append("offsamples is not 4")
    return problems


def contact_facts(summary: dict) -> dict[str, Any]:
    metrics = summary["episode_metrics"]
    contacts = metrics.get("contact_class_totals") or {}
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
        "maximum_penetration_m": metrics.get("maximum_penetration_m"),
        "saturation_fraction": float(metrics.get("saturation_fraction_of_timesteps", 0.0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule", required=True, type=Path)
    ap.add_argument("--development-manifest", required=True, type=Path)
    ap.add_argument("--reference-manifest", required=True, type=Path)
    ap.add_argument("--training-report", required=True, type=Path)
    ap.add_argument("--shift-audit", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    for name in ("schedule", "development_manifest", "reference_manifest",
                 "training_report", "shift_audit", "out"):
        setattr(args, name, Path(getattr(args, name)).resolve())

    schedule = load(args.schedule)
    development = load(args.development_manifest)
    reference = load(args.reference_manifest)
    training = load(args.training_report)
    shift = load(args.shift_audit)
    rows = {r["candidate_index"]: r for r in development["rows"]}

    executed, missing = [], []
    baseline_problems: dict[str, list[str]] = {}
    baselines: dict[str, list[dict]] = {"act_only": [], "oracle": [], "v1": []}

    for entry in schedule["entries"]:
        directory = Path(entry["output_dir"])
        if not (directory / "summary.json").is_file():
            missing.append(entry["tag"])
        else:
            summary = load(directory / "summary.json")
            rollout_path = directory / "rollout.json"
            rollout = load(rollout_path) if rollout_path.is_file() else None
            frames = (rollout or {}).get("hybrid_safety_stack", {}).get("frames", [])
            on_policy = summary["on_policy_summary"]

            oracle_active = [f for f in frames if f["on_policy"]["oracle_active"]]
            activated_active = [f for f in oracle_active if f["on_policy"].get("gate_activated")]
            cosines = [f["on_policy"]["cosine_with_oracle"] for f in activated_active
                       if f["on_policy"]["cosine_with_oracle"] is not None]
            ratios = [float(np.linalg.norm(f["on_policy"]["executed_dq"])
                            / f["on_policy"]["oracle_dq_norm"])
                      for f in activated_active if f["on_policy"]["oracle_dq_norm"] > 1e-12]
            zero_frames = [f for f in frames if not f["on_policy"]["oracle_active"]]
            zero_activated = [f for f in zero_frames if f["on_policy"].get("gate_activated")]

            longest_false_run, current = 0, 0
            for frame in frames:
                if (not frame["on_policy"]["oracle_active"]
                        and frame["on_policy"].get("gate_activated")):
                    current += 1
                    longest_false_run = max(longest_false_run, current)
                else:
                    current = 0
            corrections = [float(np.linalg.norm(f["correction"])) for f in frames] or [0.0]
            tail = frames[-5:]
            executed.append({
                "tag": entry["tag"],
                "candidate_index": entry["candidate_index"],
                "repeat_index": entry["repeat_index"],
                "hazard_present": bool(entry["hazard_present"]),
                "initial_state_matches": summary["initial_state_sha256"]
                                         == entry["initial_state_sha256"],
                "offsamples": summary.get("offsamples"),
                "reference_manifest_sha256": on_policy.get("reference_manifest_sha256"),
                "tau": on_policy.get("tau"), "rho_max": on_policy.get("rho_max"),
                **contact_facts(summary),
                "max_correction_norm": max(corrections),
                "median_correction_norm": float(np.median(corrections)),
                "activations": on_policy["activations"],
                "activation_rate": on_policy["activation_rate"],
                "longest_activation_run": on_policy["longest_activation_run"],
                "nonfinite_predictions": on_policy["nonfinite_predictions"],
                "state_neutral": on_policy["all_neutral"],
                "neutrality_failures": len(on_policy["state_neutrality_failures"]),
                "gripper_bitwise_preserved": summary["gripper_bitwise_preserved"],
                "oracle_active_frames": len(oracle_active),
                "cosine_n": len(cosines),
                "cosine_median": float(np.median(cosines)) if cosines else None,
                "cosine_fraction_positive": (float(np.mean([c > 0 for c in cosines]))
                                             if cosines else None),
                "norm_ratio_median": float(np.median(ratios)) if ratios else None,
                "norm_ratio_p95": float(np.percentile(ratios, 95)) if ratios else None,
                "capped_frames": sum(1 for f in frames if f["on_policy"].get("gate_capped")),
                "zero_frame_activation_rate": (len(zero_activated) / len(zero_frames)
                                               if zero_frames else None),
                "longest_false_activation_run": longest_false_run,
                "correction_decays_after_support_clears": bool(
                    not tail or float(np.linalg.norm(tail[-1]["correction"])) <= 0.05),
                "nonfinite_action": bool(frames and any(
                    not np.isfinite(f["executed_action"]).all() for f in frames)),
            })

        for slot, key, conditions in (
            ("act_only", "frozen_act_only_baseline", ("ACT_ONLY",)),
            ("oracle", "frozen_oracle_baseline", ("ACT_PLUS_ORACLE",)),
            ("v1", "frozen_v1_baseline", ("ACT_PLUS_DEPLOYABLE_REFERENCE",)),
        ):
            directory = Path(entry[key])
            summary_path = directory / "summary.json"
            if not summary_path.is_file():
                baseline_problems[f"{entry['tag']}:{slot}"] = ["missing summary.json"]
                continue
            summary = load(summary_path)
            problems = verify_frozen(summary, entry, conditions)
            if problems:
                baseline_problems[f"{entry['tag']}:{slot}"] = problems
                continue
            baselines[slot].append({"candidate_index": entry["candidate_index"],
                                    "repeat_index": entry["repeat_index"],
                                    **contact_facts(summary)})

    per_row = []
    for candidate in sorted(rows):
        v2 = [r for r in executed if r["candidate_index"] == candidate]
        entry = {"candidate_index": candidate,
                 "episode_id": rows[candidate]["episode_id"],
                 "hazard_present": bool(rows[candidate]["hazard_present"]),
                 "repeats": len(v2)}
        for slot in ("act_only", "oracle", "v1"):
            runs = [b for b in baselines[slot] if b["candidate_index"] == candidate]
            entry[slot] = {
                "successes": sum(1 for b in runs if b["success"]),
                "collision_free_successes": sum(1 for b in runs
                                                if b["collision_free_success"]),
                "hazard_bar_contacts": [b["contacts"]["hazard_bar"] for b in runs],
                "other_environment_contacts": [b["contacts"]["other_environment"]
                                               for b in runs]}
        entry["v2"] = {
            "successes": sum(1 for r in v2 if r["success"]),
            "collision_free_successes": sum(1 for r in v2 if r["collision_free_success"]),
            "hazard_bar_contacts": [r["contacts"]["hazard_bar"] for r in v2],
            "other_environment_contacts": [r["contacts"]["other_environment"] for r in v2],
            "activation_rate": [r["activation_rate"] for r in v2],
            "longest_activation_run": [r["longest_activation_run"] for r in v2],
            "saturation_fraction": [r["saturation_fraction"] for r in v2],
            "max_correction_norm": [r["max_correction_norm"] for r in v2],
            "median_correction_norm": [r["median_correction_norm"] for r in v2],
            "cosine_median": [r["cosine_median"] for r in v2],
            "cosine_fraction_positive": [r["cosine_fraction_positive"] for r in v2],
            "norm_ratio_median": [r["norm_ratio_median"] for r in v2],
            "norm_ratio_p95": [r["norm_ratio_p95"] for r in v2],
            "zero_frame_activation_rate": [r["zero_frame_activation_rate"] for r in v2],
            "capped_frames": [r["capped_frames"] for r in v2],
            "oracle_active_frames": [r["oracle_active_frames"] for r in v2]}
        per_row.append(entry)

    present = [r for r in per_row if r["hazard_present"]]
    absent = [r for r in per_row if not r["hazard_present"]]
    pooled = {slot: sum(r[slot]["successes"] for r in present)
              for slot in ("act_only", "oracle", "v1", "v2")}
    pooled["n"] = sum(r["repeats"] for r in present)

    cosines = [r["cosine_median"] for r in executed if r["cosine_median"] is not None]
    positives = [r["cosine_fraction_positive"] for r in executed
                 if r["cosine_fraction_positive"] is not None]
    ratios = [r["norm_ratio_median"] for r in executed if r["norm_ratio_median"] is not None]
    p95_ratios = [r["norm_ratio_p95"] for r in executed if r["norm_ratio_p95"] is not None]
    zero_rates = [r["zero_frame_activation_rate"] for r in executed
                  if r["zero_frame_activation_rate"] is not None]
    absent_runs = [r for r in executed if not r["hazard_present"]]
    present_active_corrections = [r["max_correction_norm"] for r in executed
                                  if r["hazard_present"] and r["activations"]]
    saturations = [r["saturation_fraction"] for r in executed]

    technical = {
        "all_twenty_rollouts_finalize": {"passed": len(executed) == 20 and not missing,
                                         "executed": len(executed), "missing": missing},
        "no_privileged_feature_at_inference": {
            "passed": reference["privileged_inputs"] == [],
            "runtime_inputs": reference["runtime_inputs"]},
        "no_state_neutrality_failure": {
            "passed": all(r["state_neutral"] for r in executed),
            "total": sum(r["neutrality_failures"] for r in executed)},
        "no_nonfinite_output_or_action": {
            "passed": all(r["nonfinite_predictions"] == 0 and not r["nonfinite_action"]
                          for r in executed)},
        "gripper_bitwise_unchanged": {
            "passed": all(r["gripper_bitwise_preserved"] for r in executed)},
        "initial_state_replayed_exactly": {
            "passed": all(r["initial_state_matches"] for r in executed)},
        "msaa_unchanged": {"passed": all(r["offsamples"] == 4 for r in executed)},
        "frozen_baselines_verified": {
            "passed": not baseline_problems and all(len(v) == 20 for v in baselines.values()),
            "counts": {k: len(v) for k, v in baselines.items()},
            "problems": baseline_problems, "reruns": 0},
        "confirmatory41_unexecuted": {
            "passed": load(ROOT / "configs/hybrid_obstacle_confirmatory41_v1.json"
                           )["executed_in_this_task"] is False},
        "frozen_tau_and_rho_max_used": {
            "passed": all(r["tau"] == reference["tau"] and r["rho_max"] == reference["rho_max"]
                          for r in executed),
            "tau": reference["tau"], "rho_max": reference["rho_max"]},
    }

    approximation = {
        "median_cosine_at_least_0p70": {
            "pooled_median": float(np.median(cosines)) if cosines else None,
            "per_rollout": [r["cosine_median"] for r in executed],
            "passed": bool(cosines) and float(np.median(cosines)) >= 0.70},
        "positive_cosine_fraction_at_least_80pct": {
            "pooled_median": float(np.median(positives)) if positives else None,
            "passed": bool(positives) and float(np.median(positives)) >= 0.80},
        "median_norm_ratio_between_0p5_and_1p5": {
            "pooled_median": float(np.median(ratios)) if ratios else None,
            "passed": bool(ratios) and 0.5 <= float(np.median(ratios)) <= 1.5},
        "p95_norm_ratio_at_most_2p0": {
            "pooled": float(np.median(p95_ratios)) if p95_ratios else None,
            "values": p95_ratios,
            "passed": bool(p95_ratios) and float(np.median(p95_ratios)) <= 2.0},
        "zero_frame_activation_within_2pct": {
            "values": zero_rates,
            "passed": all(v <= 0.02 for v in zero_rates)},
        "no_rollout_over_5pct_zero_frame_activation": {
            "passed": all(v <= 0.05 for v in zero_rates),
            "offenders": [r["tag"] for r in executed
                          if (r["zero_frame_activation_rate"] or 0.0) > 0.05]},
        "no_more_than_two_consecutive_false_active_frames": {
            "values": [r["longest_false_activation_run"] for r in executed],
            "passed": all(r["longest_false_activation_run"] <= 2 for r in executed)},
        "correction_decays_after_support_clears": {
            "passed": all(r["correction_decays_after_support_clears"] for r in executed),
            "offenders": [r["tag"] for r in executed
                          if not r["correction_decays_after_support_clears"]]},
        "candidate_118_activation_within_2pct": {
            "values": [r["activation_rate"] for r in absent_runs],
            "passed": all(r["activation_rate"] <= 0.02 for r in absent_runs)},
        "candidate_118_no_episode_longer_than_two_frames": {
            "values": [r["longest_activation_run"] for r in absent_runs],
            "passed": all(r["longest_activation_run"] <= 2 for r in absent_runs)},
        "candidate_118_median_correction_within_10pct": {
            "absent": float(np.median([r["median_correction_norm"] for r in absent_runs]))
            if absent_runs else None,
            "present_active": float(np.median(present_active_corrections))
            if present_active_corrections else None,
            "passed": bool(
                absent_runs and (not present_active_corrections
                                 or float(np.median([r["median_correction_norm"]
                                                     for r in absent_runs]))
                                 <= 0.10 * float(np.median(present_active_corrections))))},
        "candidate_118_no_systematic_new_other_environment_contact": {
            "act_only": absent[0]["act_only"]["other_environment_contacts"] if absent else [],
            "v2": absent[0]["v2"]["other_environment_contacts"] if absent else [],
            "passed": not absent or not (
                all(c > 0 for c in absent[0]["v2"]["other_environment_contacts"])
                and not any(c > 0
                            for c in absent[0]["act_only"]["other_environment_contacts"]))},
    }

    def row(candidate):
        return next(r for r in per_row if r["candidate_index"] == candidate)

    regression = {
        "candidate_106_at_least_4_of_5": {
            "successes": row(106)["v2"]["successes"],
            "passed": row(106)["v2"]["successes"] >= 4},
        "candidate_107_at_least_4_of_5": {
            "successes": row(107)["v2"]["successes"],
            "passed": row(107)["v2"]["successes"] >= 4},
        "pooled_hazard_present_at_least_10_of_15": {
            "v2": pooled["v2"], "n": pooled["n"],
            "passed": pooled["v2"] >= 10},
        "no_new_universal_hazard_bar_contact": {
            "offenders": [r["candidate_index"] for r in present
                          if all(c > 0 for c in r["v2"]["hazard_bar_contacts"])
                          and not any(c > 0 for c in r["act_only"]["hazard_bar_contacts"])],
            "passed": not [r for r in present
                           if all(c > 0 for c in r["v2"]["hazard_bar_contacts"])
                           and not any(c > 0
                                       for c in r["act_only"]["hazard_bar_contacts"])]},
        "candidate_118_no_universal_other_environment_contact":
            approximation["candidate_118_no_systematic_new_other_environment_contact"],
        "median_saturation_below_25pct": {
            "median": statistics.median(saturations) if saturations else None,
            "passed": bool(saturations) and statistics.median(saturations) < 0.25},
        "no_rollout_saturated_over_75pct": {
            "passed": all(s <= 0.75 for s in saturations)},
        "all_gripper_commands_bitwise_nominal": {
            "passed": all(r["gripper_bitwise_preserved"] for r in executed)},
    }

    all_technical = all(g["passed"] for g in technical.values())
    all_approximation = all(g["passed"] for g in approximation.values())
    all_regression = all(g["passed"] for g in regression.values())
    offline_passed = bool(training.get("all_offline_gates_passed"))

    if not technical["all_twenty_rollouts_finalize"]["passed"]:
        case, decision = None, "ON_POLICY_REFERENCE_DEVELOPMENT_INCOMPLETE"
    elif not offline_passed:
        case, decision = "C", "ON_POLICY_REFERENCE_OFFLINE_INVALID"
    elif all_technical and all_approximation and all_regression:
        case, decision = "A", "ON_POLICY_REFERENCE_READY_FOR_CONFIRMATORY_41"
    else:
        case, decision = "B", "REFERENCE_FEATURE_CONTRACT_INSUFFICIENT"

    report = {
        "schema": "hybrid_obstacle_on_policy_analysis_v2",
        "reference_label": reference["label"],
        "reference_manifest_sha256": reference["manifest_sha256"],
        "tau": reference["tau"], "rho_max": reference["rho_max"],
        "schedule_sha256": schedule["schedule_sha256"],
        "training_report_sha256": training["report_sha256"],
        "shift_audit_sha256": shift["report_sha256"],
        "offline_gates_passed": offline_passed,
        "rollouts_executed": len(executed),
        "rollout_budget": 20,
        "frozen_baselines_reused": {k: len(v) for k, v in baselines.items()},
        "frozen_baseline_expectations": FROZEN,
        "per_row": per_row,
        "pooled_hazard_present": pooled,
        "technical_gates": technical,
        "approximation_gates": approximation,
        "gross_regression_gates": regression,
        "all_technical_gates_passed": all_technical,
        "all_approximation_gates_passed": all_approximation,
        "all_gross_regression_gates_passed": all_regression,
        "candidate_118_negative_control": {
            "activation_rate": [r["activation_rate"] for r in absent_runs],
            "longest_activation_run": [r["longest_activation_run"] for r in absent_runs],
            "median_correction_norm": [r["median_correction_norm"] for r in absent_runs],
            "other_environment_contacts":
                absent[0]["v2"]["other_environment_contacts"] if absent else None,
            "act_only_other_environment_contacts":
                absent[0]["act_only"]["other_environment_contacts"] if absent else None,
            "v1_other_environment_contacts":
                absent[0]["v1"]["other_environment_contacts"] if absent else None},
        "case": case,
        "decision": decision,
        "evidence_basis": ("contact classification only; minimum_clearance_m is excluded "
                           "because a per-geom mj_geomDistance defect pins it at <= 0"),
        "rollouts": executed,
        "confirmatory_statistical_testing_performed": False,
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"{'cand':>5} {'haz':>8} {'AO':>4} {'ORA':>4} {'V1':>4} {'V2':>4} "
          f"{'act%':>6} {'cos':>7} {'ratio':>7}")
    for entry in per_row:
        acts = [a for a in entry["v2"]["activation_rate"] if a is not None]
        cos = [c for c in entry["v2"]["cosine_median"] if c is not None]
        rat = [c for c in entry["v2"]["norm_ratio_median"] if c is not None]
        print(f"{entry['candidate_index']:5d} "
              f"{'present' if entry['hazard_present'] else 'absent':>8} "
              f"{entry['act_only']['successes']:>2}/5 {entry['oracle']['successes']:>2}/5 "
              f"{entry['v1']['successes']:>2}/5 {entry['v2']['successes']:>2}/5 "
              f"{100*(np.mean(acts) if acts else 0):6.1f} "
              f"{(np.median(cos) if cos else float('nan')):7.3f} "
              f"{(np.median(rat) if rat else float('nan')):7.3f}")
    print(f"\npooled hazard-present: AO {pooled['act_only']}/{pooled['n']} | "
          f"ORACLE {pooled['oracle']}/{pooled['n']} | V1 {pooled['v1']}/{pooled['n']} | "
          f"V2 {pooled['v2']}/{pooled['n']}")
    for name, gates in (("technical", technical), ("approximation", approximation),
                        ("gross-regression", regression)):
        print(f"\n{name} gates:")
        for key, gate in gates.items():
            print(f"  [{'PASS' if gate['passed'] else 'FAIL'}] {key}")
    print(f"\nCase {case} -> {decision}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
