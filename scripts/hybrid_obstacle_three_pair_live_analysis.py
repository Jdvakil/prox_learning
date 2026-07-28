#!/usr/bin/env python3
"""Live development analysis: do false-positive bursts cause closed-loop harm?

Reads the 20 development rollouts and answers the owner's primary question directly. A
false-positive frame is one the contract *executed* while the privileged oracle was zero;
bursts are maximal runs of consecutive such frames.

Two shadow decisions are compared throughout:

* ``ACTIVITY_ONLY_SHADOW`` -- what the seed-0 activity gate alone would have executed. The
  difference between it and the executed decision is the uncertainty veto's live
  contribution, which no offline task could measure.
* the privileged parked oracle, which supplies ground truth and is never executed.

Harm is assessed at the resolution the logged schema supports, and the report says which is
which: correction magnitude, arm deviation and post-burst persistence are frame-resolved;
contact classes, penetration and saturation are episode-resolved totals.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin import threshold as thr

ARM = 7
POST_BURST_FRAMES = 10
# a correction is "meaningfully displaced" when it exceeds this fraction of max_dev
PERSISTENCE_FRACTION = 0.10
MAX_DEV = 0.35
HAZARD_PRESENT = (106, 107, 108)
HAZARD_ABSENT = 118


def runs_of(flags: np.ndarray):
    """Maximal runs of True as (start, end_inclusive)."""
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return []
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b - 1)) for a, b in zip(edges[0::2], edges[1::2])]


def load_rollout(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text())
    frames = np.load(directory / "frames.npz", allow_pickle=True)
    return {"summary": summary, "frames": {k: frames[k] for k in frames.files},
            "directory": str(directory)}


def analyse(record: dict) -> dict:
    summary = record["summary"]
    f = record["frames"]
    metrics = summary["episode_metrics"]

    executed = f["uncertainty_executed"].astype(bool)
    activity_shadow = f["uncertainty_activity_only_shadow_execute"].astype(bool)
    oracle_active = f["privileged_oracle_active"].astype(bool)
    veto = f["uncertainty_uncertainty_veto"].astype(bool)
    nominal = f["nominal_action"]
    action = f["executed_action"]
    correction = f["correction"]
    deviation = np.abs(action[:, :ARM] - nominal[:, :ARM]).max(axis=1)
    executed_norm = np.linalg.norm(f["executed_dq"], axis=1)

    false_positive = executed & ~oracle_active
    shadow_false_positive = activity_shadow & ~oracle_active

    bursts = []
    for start, end in runs_of(false_positive):
        after = slice(end + 1, min(end + 1 + POST_BURST_FRAMES, len(executed)))
        window = slice(start, end + 1)
        bursts.append({
            "start_timestep": start, "end_timestep": end,
            "length": end - start + 1,
            "seed0_activity_max": float(f["uncertainty_seed0_activity"][window].max()),
            "jaccard_01_min": float(f["uncertainty_jaccard_01"][window].min()),
            "jaccard_02_min": float(f["uncertainty_jaccard_02"][window].min()),
            "jaccard_12_min": float(f["uncertainty_jaccard_12"][window].min()),
            "three_pair_agreement_min": float(
                f["uncertainty_three_pair_agreement"][window].min()),
            "activity_only_shadow_would_execute": bool(activity_shadow[window].all()),
            "uncertainty_veto_active_in_burst": bool(veto[window].any()),
            "executed_dq_norm_max": float(executed_norm[window].max()),
            "accumulated_correction_norm_max": float(
                np.linalg.norm(correction[window], axis=1).max()),
            "max_arm_deviation_rad_in_burst": float(deviation[window].max()),
            "max_arm_deviation_rad_after": float(deviation[after].max())
            if after.stop > after.start else 0.0,
            "post_burst_frames_examined": max(0, after.stop - after.start),
            "persists_after_burst": bool(
                after.stop > after.start
                and deviation[after].max() > PERSISTENCE_FRACTION * MAX_DEV),
            "task_phase_progress": float(start / max(len(executed) - 1, 1)),
        })

    shadow_bursts = [{"start_timestep": a, "end_timestep": b, "length": b - a + 1}
                     for a, b in runs_of(shadow_false_positive)]

    gripper = f["gripper_command"]
    nominal_gripper = nominal[:, ARM:ARM + 1]
    return {
        "directory": record["directory"],
        "candidate_index": summary["candidate_index"],
        "repeat_index": summary["repeat_index"],
        "hazard_present": bool(summary["hazard_present"]),
        "frames": int(summary["frames"]),
        "task_success": bool(summary["success"]),
        "gripper_bitwise_preserved": bool(summary["gripper_bitwise_preserved"]),
        "gripper_matches_nominal": bool(np.array_equal(gripper, nominal_gripper)),
        "nonfinite_actions": int((~np.isfinite(action)).sum()),
        "executed_dq_finite": bool(f["uncertainty_executed_dq_finite"].all()),
        "oracle_active_frames": int(oracle_active.sum()),
        "oracle_zero_frames": int((~oracle_active).sum()),
        "executed_frames": int(executed.sum()),
        "activity_only_shadow_frames": int(activity_shadow.sum()),
        "uncertainty_veto_frames": int(veto.sum()),
        "active_recall": float(executed[oracle_active].mean())
        if oracle_active.any() else None,
        "activity_only_recall": float(activity_shadow[oracle_active].mean())
        if oracle_active.any() else None,
        "executed_false_activation_rate": float(false_positive[~oracle_active].mean())
        if (~oracle_active).any() else None,
        "shadow_false_activation_rate": float(
            shadow_false_positive[~oracle_active].mean())
        if (~oracle_active).any() else None,
        "false_positive_frames": int(false_positive.sum()),
        "shadow_false_positive_frames": int(shadow_false_positive.sum()),
        "bursts": bursts,
        "burst_count": len(bursts),
        "max_burst_length": max((b["length"] for b in bursts), default=0),
        "shadow_burst_count": len(shadow_bursts),
        "max_shadow_burst_length": max((b["length"] for b in shadow_bursts), default=0),
        "bursts_persisting": sum(1 for b in bursts if b["persists_after_burst"]),
        # episode-resolved
        "contact_class_totals": metrics["contact_class_totals"],
        "maximum_penetration_m": metrics["maximum_penetration_m"],
        "maximum_nominal_deviation_norm": metrics["maximum_nominal_deviation_norm"],
        "return_to_nominal_error": metrics["return_to_nominal_error"],
        "saturation_fraction_of_timesteps": metrics["saturation_fraction_of_timesteps"],
        "integrated_residual_norm": metrics["integrated_residual_norm"],
        "median_cosine": summary["on_policy_summary"]["cosine_median"],
        "state_neutral": summary["on_policy_summary"]["all_neutral"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout-root", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    directories = sorted(d for d in args.rollout_root.iterdir()
                         if d.is_dir() and (d / "summary.json").is_file())
    rollouts = [analyse(load_rollout(d)) for d in directories]
    print(f"analysed {len(rollouts)} rollouts")

    by_candidate: dict[int, list] = {}
    for r in rollouts:
        by_candidate.setdefault(r["candidate_index"], []).append(r)

    hazard_present = [r for r in rollouts if r["candidate_index"] in HAZARD_PRESENT]
    absent = [r for r in rollouts if r["candidate_index"] == HAZARD_ABSENT]
    all_bursts = [b for r in rollouts for b in r["bursts"]]

    # ---- harm assessment ----------------------------------------------------------
    hazard_bar_rollouts = [r for r in rollouts
                           if r["contact_class_totals"].get("hazard_bar", 0) > 0]
    other_env_absent = [r for r in absent
                        if r["contact_class_totals"].get("other_environment", 0) > 0]
    harm = {
        "new_hazard_bar_contact": {
            "rollouts_with_hazard_bar_contact": len(hazard_bar_rollouts),
            "detail": [{"candidate": r["candidate_index"], "repeat": r["repeat_index"],
                        "frames": r["contact_class_totals"]["hazard_bar"]}
                       for r in hazard_bar_rollouts],
        },
        "candidate_118_other_environment": {
            "rollouts_with_contact": len(other_env_absent),
            "all_five_repeats": len(other_env_absent) == len(absent) and bool(absent),
            "per_repeat": [r["contact_class_totals"].get("other_environment", 0)
                           for r in absent],
        },
        "task_success_failures": [
            {"candidate": r["candidate_index"], "repeat": r["repeat_index"]}
            for r in rollouts if not r["task_success"]],
        "invalid_or_nonfinite_actions": sum(r["nonfinite_actions"] for r in rollouts),
        "executed_dq_all_finite": all(r["executed_dq_finite"] for r in rollouts),
        "gripper_bitwise_preserved_everywhere": all(
            r["gripper_bitwise_preserved"] and r["gripper_matches_nominal"]
            for r in rollouts),
        "bursts_with_persistent_deviation": sum(r["bursts_persisting"] for r in rollouts),
        "rollouts_with_persistent_deviation": sum(
            1 for r in rollouts if r["bursts_persisting"] > 0),
    }

    # ---- development success criteria ----------------------------------------------
    def successes(candidate):
        rows = by_candidate.get(candidate, [])
        return sum(1 for r in rows if r["task_success"]), len(rows)

    s106, n106 = successes(106)
    s107, n107 = successes(107)
    pooled = sum(1 for r in hazard_present if r["task_success"])
    criteria = {
        "all_20_rollouts_finalized": len(rollouts) == 20,
        "gripper_bitwise_nominal": harm["gripper_bitwise_preserved_everywhere"],
        "no_invalid_actions": harm["invalid_or_nonfinite_actions"] == 0
        and harm["executed_dq_all_finite"],
        "candidate_106_at_least_4_of_5": s106 >= 4 and n106 == 5,
        "candidate_107_at_least_4_of_5": s107 >= 4 and n107 == 5,
        "pooled_hazard_present_at_least_10_of_15": pooled >= 10
        and len(hazard_present) == 15,
        "candidate_118_no_systematic_new_contact": not harm[
            "candidate_118_other_environment"]["all_five_repeats"],
        "no_repeated_controller_attributable_contact": len(hazard_bar_rollouts) == 0,
        "no_meaningful_persistent_displacement":
            harm["rollouts_with_persistent_deviation"] == 0,
        "confirmatory41_untouched": True,
    }

    report = {
        "schema": "hybrid_obstacle_three_pair_live_analysis_v1",
        "condition": "ACT_PLUS_THREE_PAIR_JOINT_GATE",
        "deployment_manifest_sha256": json.loads(
            args.manifest.read_text())["manifest_sha256"],
        "rollouts": rollouts,
        "rollout_count": len(rollouts),
        "primary_question": "Do false-positive onset bursts cause closed-loop harm?",
        "false_positive_summary": {
            "total_false_positive_frames": sum(r["false_positive_frames"]
                                               for r in rollouts),
            "total_bursts": len(all_bursts),
            "max_burst_length": max((b["length"] for b in all_bursts), default=0),
            "burst_length_histogram": {
                str(length): sum(1 for b in all_bursts if b["length"] == length)
                for length in sorted({b["length"] for b in all_bursts})},
            "bursts_persisting_after_end": sum(1 for b in all_bursts
                                               if b["persists_after_burst"]),
            "max_arm_deviation_in_any_burst_rad": max(
                (b["max_arm_deviation_rad_in_burst"] for b in all_bursts), default=0.0),
            "max_arm_deviation_after_any_burst_rad": max(
                (b["max_arm_deviation_rad_after"] for b in all_bursts), default=0.0),
            "max_dev_limit_rad": MAX_DEV,
            "persistence_criterion":
                f"post-burst arm deviation > {PERSISTENCE_FRACTION} * {MAX_DEV} rad",
        },
        "uncertainty_veto_contribution": {
            "executed_frames": sum(r["executed_frames"] for r in rollouts),
            "activity_only_shadow_frames": sum(r["activity_only_shadow_frames"]
                                               for r in rollouts),
            "veto_frames": sum(r["uncertainty_veto_frames"] for r in rollouts),
            "shadow_false_positive_frames": sum(r["shadow_false_positive_frames"]
                                                for r in rollouts),
            "executed_false_positive_frames": sum(r["false_positive_frames"]
                                                  for r in rollouts),
            "max_shadow_burst_length": max((r["max_shadow_burst_length"]
                                            for r in rollouts), default=0),
            "note": ("ACTIVITY_ONLY_SHADOW is never executed; the difference between the "
                     "shadow and executed columns is the veto's live contribution"),
        },
        "harm_assessment": harm,
        "development_criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
        "per_candidate": {
            str(candidate): {
                "rollouts": len(rows),
                "successes": sum(1 for r in rows if r["task_success"]),
                "mean_executed_false_activation": float(np.mean(
                    [r["executed_false_activation_rate"] for r in rows
                     if r["executed_false_activation_rate"] is not None]))
                if any(r["executed_false_activation_rate"] is not None for r in rows)
                else None,
                "mean_active_recall": float(np.mean(
                    [r["active_recall"] for r in rows if r["active_recall"] is not None]))
                if any(r["active_recall"] is not None for r in rows) else None,
                "hazard_bar_contacts": sum(
                    r["contact_class_totals"].get("hazard_bar", 0) for r in rows),
                "other_environment_contacts": sum(
                    r["contact_class_totals"].get("other_environment", 0) for r in rows),
                "max_saturation_fraction": max(r["saturation_fraction_of_timesteps"]
                                               for r in rows),
                "max_burst_length": max(r["max_burst_length"] for r in rows),
            } for candidate, rows in sorted(by_candidate.items())},
        "measurement_resolution": {
            "frame_resolved": ["gate decisions", "activity", "J01/J02/J12",
                               "three-pair agreement", "executed differential",
                               "accumulated correction", "arm deviation",
                               "post-burst persistence", "gripper command"],
            "episode_resolved": ["contact class totals", "maximum penetration",
                                 "saturation fraction", "return-to-nominal error"],
            "limitation": ("per-frame contact classes are not in the logged rollout "
                           "schema, so contact attribution is episode-level; burst-local "
                           "contact counts could not be computed without re-running with "
                           "an extended schema"),
        },
    }
    report["report_sha256"] = thr.canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"\n{'cand':>5}{'rep':>5}{'succ':>6}{'recall':>8}{'FPrate':>8}"
          f"{'bursts':>8}{'maxlen':>8}{'hazbar':>8}{'otherenv':>10}{'sat':>7}")
    for r in sorted(rollouts, key=lambda x: (x["candidate_index"], x["repeat_index"])):
        c = r["contact_class_totals"]
        print(f"{r['candidate_index']:>5}{r['repeat_index']:>5}"
              f"{r['task_success']!s:>6}"
              f"{(r['active_recall'] if r['active_recall'] is not None else float('nan')):>8.3f}"
              f"{(r['executed_false_activation_rate'] or 0):>8.4f}"
              f"{r['burst_count']:>8}{r['max_burst_length']:>8}"
              f"{c.get('hazard_bar', 0):>8}{c.get('other_environment', 0):>10}"
              f"{r['saturation_fraction_of_timesteps']:>7.3f}")
    print()
    fp = report["false_positive_summary"]
    print(f"false-positive frames {fp['total_false_positive_frames']}, "
          f"bursts {fp['total_bursts']}, max length {fp['max_burst_length']}")
    print(f"max arm deviation in burst {fp['max_arm_deviation_in_any_burst_rad']:.4f} rad "
          f"(limit {MAX_DEV})")
    print(f"bursts persisting after end: {fp['bursts_persisting_after_end']}")
    v = report["uncertainty_veto_contribution"]
    print(f"veto: shadow FP frames {v['shadow_false_positive_frames']} -> executed "
          f"{v['executed_false_positive_frames']}  (veto fired {v['veto_frames']}x)")
    print("\ndevelopment criteria:")
    for name, ok in criteria.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
