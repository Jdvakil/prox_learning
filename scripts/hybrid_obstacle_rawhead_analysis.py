#!/usr/bin/env python3
"""Analyse the 40-rollout raw-head development schedule.

Handoff steps 11-14:

* step 11 -- quantify rendering-induced policy variability across the five ACT_ONLY
  repeats per row. This is MSAA render variation, not simulator-state
  nondeterminism: qpos and all 40 proximity streams start identical.
* step 12 -- per-row repeated outcomes, with the manifest row as the experimental
  block and the repeat as nested stochastic replication.
* step 13 -- technical and gross-regression gates.
* step 14 -- the predeclared repeat-count rule for the future confirmatory 41.

Descriptive only. No confirmatory significance test is run on four development rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

MAX_DEV = 0.35
SAFETY_RELEVANT = ("hazard_bar", "other_environment")


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def load(root: Path, tag: str) -> dict | None:
    p = root / tag / "rollout.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())["hybrid_safety_stack"]


def safety_relevant_contacts(block: dict) -> int:
    t = block["episode_metrics"]["contact_class_totals"]
    return sum(t[k] for k in SAFETY_RELEVANT)


def collision_free_success(block: dict) -> bool:
    return bool(block["episode_metrics"]["task_success"] and safety_relevant_contacts(block) == 0)


def render_variability(blocks: list[dict]) -> dict:
    """MSAA-induced observation and action variation across ACT_ONLY repeats."""
    exo, wrist, qpos, prox = [], [], [], []
    step0, first10, phases, contacts, succ, cfs = [], [], [], [], [], []
    for b in blocks:
        f0 = b["frames"][0]
        exo.append(f0.get("exo_camera_1_sha256"))
        wrist.append(f0.get("wrist_camera_sha256"))
        fp = b.get("first_observation_fingerprint") or {}
        qpos.append(fp.get("qpos_sha256"))
        prox.append(fp.get("proximity_sha256"))
        step0.append(np.asarray(f0["nominal_act_action"], dtype=np.float64))
        first10.append(np.asarray([fr["nominal_act_action"] for fr in b["frames"][:10]],
                                  dtype=np.float64))
        phases.append(tuple(fr["task_phase"] for fr in b["frames"]))
        contacts.append(tuple(
            (fr["step"], p["geom1"], p["geom2"]) for fr in b["frames"]
            for p in fr["collision_geom_pairs"]))
        succ.append(bool(b["episode_metrics"]["task_success"]))
        cfs.append(collision_free_success(b))

    s0 = np.stack(step0)
    n = min(len(x) for x in first10)
    f10 = np.stack([x[:n] for x in first10])
    # first materially diverging step: any pair differs by >1e-6 in the nominal action
    diverge = None
    L = min(len(b["frames"]) for b in blocks)
    for t in range(L):
        vals = np.stack([np.asarray(b["frames"][t]["nominal_act_action"], dtype=np.float64)
                         for b in blocks])
        if float(np.max(vals.max(axis=0) - vals.min(axis=0))) > 1e-6:
            diverge = t
            break
    pair_max = 0.0
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            m = min(len(blocks[i]["frames"]), len(blocks[j]["frames"]))
            a = np.asarray([blocks[i]["frames"][t]["nominal_act_action"] for t in range(m)])
            bb = np.asarray([blocks[j]["frames"][t]["nominal_act_action"] for t in range(m)])
            pair_max = max(pair_max, float(np.max(np.abs(a - bb))))
    return {
        "repeats": len(blocks),
        "observation_variation": {
            "unique_step0_exo_hashes": len(set(exo)),
            "unique_step0_wrist_hashes": len(set(wrist)),
            "unique_step0_qpos_hashes": len(set(qpos)),
            "unique_step0_proximity_hashes": len(set(prox)),
            "qpos_identical_across_repeats": len(set(qpos)) == 1,
            "proximity_identical_across_repeats": len(set(prox)) == 1,
            "attribution": ("RGB hashes vary because offsamples=4 makes the multisample resolve "
                            "non-reproducible. qpos and all 40 proximity streams start identical, "
                            "so this is render variation, NOT simulator-state nondeterminism."),
        },
        "policy_variation": {
            "step0_nominal_action_range_per_dim": (s0.max(axis=0) - s0.min(axis=0)).tolist(),
            "step0_nominal_action_max_range": float(np.max(s0.max(axis=0) - s0.min(axis=0))),
            "first10_nominal_action_variance_per_dim":
                f10.var(axis=0).mean(axis=0).tolist(),
            "first10_nominal_action_max_variance": float(f10.var(axis=0).max()),
            "max_pairwise_nominal_action_delta": pair_max,
            "first_materially_diverging_step": diverge,
            "unique_phase_paths": len(set(phases)),
            "unique_contact_sequences": len(set(contacts)),
            "task_success_values": succ,
            "task_success_count": sum(succ),
            "collision_free_success_values": cfs,
            "collision_free_success_count": sum(cfs),
            "success_varies": len(set(succ)) > 1,
            "collision_free_success_varies": len(set(cfs)) > 1,
        },
    }


def condition_stats(blocks: list[dict]) -> dict:
    succ = [bool(b["episode_metrics"]["task_success"]) for b in blocks]
    cfs = [collision_free_success(b) for b in blocks]
    haz = [b["episode_metrics"]["contact_class_totals"]["hazard_bar"] for b in blocks]
    oth = [b["episode_metrics"]["contact_class_totals"]["other_environment"] for b in blocks]
    clear = [b["episode_metrics"]["minimum_clearance_m"] for b in blocks
             if b["episode_metrics"]["minimum_clearance_m"] is not None]
    sat = [b["episode_metrics"]["saturation_fraction_of_timesteps"] for b in blocks]
    dur = [len(b["frames"]) for b in blocks]
    corr = [b["episode_metrics"]["maximum_nominal_deviation_norm"] for b in blocks]
    return {
        "rollouts": len(blocks),
        "task_success": f"{sum(succ)}/{len(succ)}",
        "task_success_count": sum(succ),
        "task_success_values": succ,
        "collision_free_success": f"{sum(cfs)}/{len(cfs)}",
        "collision_free_success_count": sum(cfs),
        "collision_free_success_values": cfs,
        "hazard_bar_contacts": haz,
        "hazard_bar_contact_total": sum(haz),
        "other_environment_contacts": oth,
        "other_environment_contact_total": sum(oth),
        "minimum_clearance_m": {"values": clear,
                                "median": statistics.median(clear) if clear else None,
                                "min": min(clear) if clear else None},
        "saturation_fraction": {"values": sat, "median": statistics.median(sat),
                                "max": max(sat), "min": min(sat)},
        "task_duration_steps": {"values": dur, "median": statistics.median(dur)},
        "max_correction_norm": {"values": corr, "median": statistics.median(corr),
                                "max": max(corr)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--schedule", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    sched = json.loads(args.schedule.read_text())
    entries = sorted(sched["entries"], key=lambda e: e["execution_order"])

    loaded, missing = {}, []
    for e in entries:
        b = load(args.root, e["tag"])
        if b is None:
            missing.append(e["tag"])
        else:
            loaded[e["tag"]] = b

    # ---- technical gates ---------------------------------------------------
    shadow = [b["shadow_zero_equivalence"] for b in loaded.values()
              if b.get("shadow_zero_equivalence")]
    nonfinite, gripper_bad, agg_bad, reset_bad, state_bad = [], [], [], [], []
    for tag, b in loaded.items():
        for fr in b["frames"]:
            ex = np.asarray(fr["executed_action"], dtype=np.float64)
            nom = np.asarray(fr["nominal_act_action"], dtype=np.float64)
            co = np.asarray(fr["correction"], dtype=np.float64)
            if not (np.isfinite(ex).all() and np.isfinite(co).all()):
                nonfinite.append(tag); break
            if ex[7] != nom[7]:
                gripper_bad.append(tag); break
            if float(np.max(np.abs(ex[:7] - nom[:7] - co))) > 1e-6:
                agg_bad.append(tag); break
        # Controller reset: frame 0 must be consistent with ZERO initial state, i.e.
        # filtered_0 == (1-ema)*delta_0 and correction_0 == clip(gain*filtered_0*dt).
        # Requiring correction_0 == 0 would be wrong: under a nonzero drive the
        # controller legitimately takes its first step at frame 0.
        if b["frames"]:
            f0 = b["frames"][0]
            delta0 = np.asarray(f0["subtracted_dq"], dtype=np.float64)
            filt0 = np.asarray(f0["filtered_safety_dq"], dtype=np.float64)
            corr0 = np.asarray(f0["correction"], dtype=np.float64)
            dt = b["controller"]["dt"]
            exp_filt = (1.0 - b["controller"]["ema"]) * delta0
            exp_corr = np.clip(b["controller"]["gain"] * exp_filt * dt, -MAX_DEV, MAX_DEV)
            if (float(np.max(np.abs(filt0 - exp_filt))) > 1e-5
                    or float(np.max(np.abs(corr0 - exp_corr))) > 1e-5):
                reset_bad.append(tag)
        mp = b["manifest_provenance"]
        if mp["replayed_initial_state_sha256"] != mp["expected_initial_state_sha256"] \
                or mp.get("offsamples") != 4:
            state_bad.append(tag)

    technical = {
        "all_40_primary_rollouts_finalized": len(loaded) == 40,
        "rollouts_loaded": len(loaded),
        "missing": missing,
        "shadow_zero_checked_rollouts": len(shadow),
        "shadow_zero_all_passed": all(s["passed"] for s in shadow) if shadow else False,
        "shadow_zero_max_arm_delta": max((s["max_arm_delta"] for s in shadow), default=None),
        "shadow_zero_gripper_all_equal": all(s["gripper_bitwise_equal_every_frame"]
                                             for s in shadow) if shadow else False,
        "no_nonfinite_action": not nonfinite,
        "gripper_unchanged_every_frame": not gripper_bad,
        "residual_arm_only_and_after_aggregation": not agg_bad,
        "controller_reset_between_repeats": not reset_bad,
        "no_artifact_or_initial_state_mismatch": not state_bad,
        "offenders": {"nonfinite": nonfinite, "gripper": gripper_bad, "aggregation": agg_bad,
                      "reset": reset_bad, "state": state_bad},
        "no_log_corruption": not missing,
    }
    technical["all_passed"] = all(v for k, v in technical.items()
                                  if isinstance(v, bool))

    # ---- per row ----------------------------------------------------------
    cands = sorted({e["candidate_index"] for e in entries})
    rows_out, unstable = [], []
    for c in cands:
        haz = next(e["hazard_present"] for e in entries if e["candidate_index"] == c)
        only = [loaded[e["tag"]] for e in entries
                if e["candidate_index"] == c and e["condition"] == "ACT_ONLY"
                and e["tag"] in loaded]
        raw = [loaded[e["tag"]] for e in entries
               if e["candidate_index"] == c and e["condition"] == "ACT_PLUS_RAW_HEAD"
               and e["tag"] in loaded]
        rv = render_variability(only) if len(only) >= 2 else None
        so = condition_stats(only) if only else None
        sr = condition_stats(raw) if raw else None
        is_unstable = bool(so and (so["task_success_count"] not in (0, len(only))
                                   or so["collision_free_success_count"] not in (0, len(only))))
        if is_unstable:
            unstable.append(c)
        rows_out.append({
            "candidate_index": c, "hazard_present": haz,
            "act_only": so, "act_plus_raw_head": sr,
            "render_induced_variability": rv,
            "act_only_outcome_unstable": is_unstable,
            "deltas": ({
                "task_success_delta": sr["task_success_count"] - so["task_success_count"],
                "collision_free_success_delta": (sr["collision_free_success_count"]
                                                 - so["collision_free_success_count"]),
                "hazard_bar_contact_delta": (sr["hazard_bar_contact_total"]
                                             - so["hazard_bar_contact_total"]),
                "other_environment_contact_delta": (sr["other_environment_contact_total"]
                                                    - so["other_environment_contact_total"]),
                "median_duration_delta": (sr["task_duration_steps"]["median"]
                                          - so["task_duration_steps"]["median"]),
            } if so and sr else None),
        })

    present_rows = [r for r in rows_out if r["hazard_present"]]
    absent_rows = [r for r in rows_out if not r["hazard_present"]]
    all_raw_sat = [s for r in rows_out if r["act_plus_raw_head"]
                   for s in r["act_plus_raw_head"]["saturation_fraction"]["values"]]
    pooled_present_only = sum(r["act_only"]["task_success_count"] for r in present_rows)
    pooled_present_raw = sum(r["act_plus_raw_head"]["task_success_count"] for r in present_rows)
    n_present = sum(r["act_only"]["rollouts"] for r in present_rows)

    catastrophic = [r["candidate_index"] for r in rows_out
                    if r["act_only"] and r["act_plus_raw_head"]
                    and r["act_only"]["task_success_count"] == r["act_only"]["rollouts"]
                    and r["act_plus_raw_head"]["task_success_count"] == 0]
    absent_new_collision = [
        r["candidate_index"] for r in absent_rows
        if r["act_only"] and r["act_plus_raw_head"]
        and r["act_only"]["hazard_bar_contact_total"] + r["act_only"]["other_environment_contact_total"] == 0
        and all(h + o > 0 for h, o in zip(r["act_plus_raw_head"]["hazard_bar_contacts"],
                                          r["act_plus_raw_head"]["other_environment_contacts"],
                                          strict=False))]
    median_sat = statistics.median(all_raw_sat) if all_raw_sat else None
    over75 = [s for s in all_raw_sat if s > 0.75]
    pooled_gap = ((pooled_present_only - pooled_present_raw) / n_present * 100
                  if n_present else None)

    gates = {
        "no_row_flips_5of5_to_0of5": {"passed": not catastrophic, "offenders": catastrophic},
        "hazard_absent_no_universal_new_collision": {"passed": not absent_new_collision,
                                                     "offenders": absent_new_collision},
        "median_raw_head_saturation_below_25pct": {
            "passed": median_sat is not None and median_sat < 0.25,
            "median": median_sat, "threshold": 0.25,
            "values": sorted(all_raw_sat)},
        "no_rollout_saturated_over_75pct": {"passed": not over75, "offenders": over75,
                                            "threshold": 0.75},
        "pooled_hazard_present_success_gap_within_20pp": {
            "passed": pooled_gap is not None and pooled_gap <= 20.0,
            "act_only": f"{pooled_present_only}/{n_present}",
            "raw_head": f"{pooled_present_raw}/{n_present}",
            "gap_percentage_points": pooled_gap, "threshold": 20.0},
    }
    gates["all_passed"] = all(g["passed"] for g in gates.values() if isinstance(g, dict))

    # ---- step 14: repeat count --------------------------------------------
    n_unstable = len(unstable)
    if n_unstable == 0:
        future_repeats, verdict = 3, "0 unstable rows -> 3 repeats per condition"
    elif n_unstable <= 2:
        future_repeats, verdict = 5, f"{n_unstable} unstable row(s) -> 5 repeats per condition"
    else:
        future_repeats, verdict = None, ("STOCHASTICITY_TOO_HIGH_FOR_CONFIRMATORY_PROTOCOL")

    report = {
        "schema": "hybrid_obstacle_rawhead_development_analysis_v1",
        "statistical_disclaimer": ("Four development rows. Descriptive only: no confirmatory "
            "significance test is run here, and none of this is evidence of a safety improvement. "
            "The manifest row is the experimental block and the repeat is nested stochastic "
            "replication."),
        "schedule_sha256": sched["schedule_sha256"],
        "technical_gates": technical,
        "gross_regression_gates": gates,
        "rows": rows_out,
        "aggregate": {
            "hazard_present_rows": len(present_rows),
            "hazard_absent_rows": len(absent_rows),
            "pooled_hazard_present_act_only_success": f"{pooled_present_only}/{n_present}",
            "pooled_hazard_present_raw_head_success": f"{pooled_present_raw}/{n_present}",
            "raw_head_saturation_median": median_sat,
            "raw_head_saturation_all": sorted(all_raw_sat),
        },
        "outcome_stability": {
            "unstable_rows": unstable, "unstable_count": n_unstable,
            "definition": ("ACT_ONLY task-success or collision-free-success count over 5 repeats "
                           "that is neither 0/5 nor 5/5"),
        },
        "future_confirmatory_repeat_count": {
            "selected": future_repeats, "rule_outcome": verdict,
            "future_rollout_count": (41 * 2 * future_repeats) if future_repeats else None,
            "confirmatory_manifest": "configs/hybrid_obstacle_confirmatory41_v1.json",
            "launched_in_this_task": False,
        },
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"rollouts loaded          : {len(loaded)}/40  missing={len(missing)}")
    print(f"technical gates          : {'PASS' if technical['all_passed'] else 'FAIL'}")
    print(f"  shadow-zero all passed : {technical['shadow_zero_all_passed']} "
          f"(max arm delta {technical['shadow_zero_max_arm_delta']})")
    print()
    print(f"{'cand':>5} {'hazard':>8} {'ACT_ONLY succ':>14} {'RAW succ':>9} "
          f"{'AO cfs':>7} {'RAW cfs':>8} {'RAW sat med':>12} {'unstable':>9}")
    for r in rows_out:
        so, sr = r["act_only"], r["act_plus_raw_head"]
        if not (so and sr):
            continue
        print(f"{r['candidate_index']:5d} {'present' if r['hazard_present'] else 'absent':>8} "
              f"{so['task_success']:>14} {sr['task_success']:>9} "
              f"{so['collision_free_success']:>7} {sr['collision_free_success']:>8} "
              f"{sr['saturation_fraction']['median']:12.3f} "
              f"{r['act_only_outcome_unstable']!s:>9}")
    print()
    for k, g in gates.items():
        if isinstance(g, dict):
            print(f"  {'PASS' if g['passed'] else 'FAIL'}  {k}")
    print()
    print(f"unstable rows            : {n_unstable} {unstable}")
    print(f"future repeat count      : {future_repeats} ({verdict})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
