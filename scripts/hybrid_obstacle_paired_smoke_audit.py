#!/usr/bin/env python3
"""Audit the bounded paired ACT / ACT+Safety smoke.

Covers handoff steps 10 and 13:

* **zero equivalence** -- ``ACT_PLUS_ZERO`` must reproduce ``ACT_ONLY`` exactly on
  the initial state, every nominal and executed action, the phase sequence, the
  success outcome and the collision sequence. Any discrete mismatch is a hard
  failure.
* **paired pilot metrics** -- per-pair task success, collision-free success,
  clearances, correction magnitudes, gripper equality and sensor attribution,
  reported separately for hazard-present and hazard-absent rows.

Four pairs cannot support a statistical superiority claim and none is made here.
The exact-McNemar and paired-bootstrap machinery is deliberately left for the
later 45-row evaluation.

The two link5-front sensors with known pre-existing self returns are reported in
their own column so their activation is never silently counted as
obstacle-responsive evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

#: Sensors with pre-existing self returns (known 38/40 finding). Geometry unchanged.
KNOWN_SELF_RETURN_SENSORS = ("link5_front_sensor_1", "link5_front_sensor_2")

#: The hazard bar is compiled as body ``protr_s`` / ``protr_m`` / ``protr_l``
#: (``enclosure_reach.py:716``). Contacts against it are the safety-relevant event.
HAZARD_BODY_PREFIX = "protr"
#: The grasp target is injected as ``cavity_obj_0/<uid>``. The robot is *supposed*
#: to touch it, so its contacts are the grasp, not a safety failure.
TARGET_ROOT_PREFIX = "cavity_obj_"


def classify_contact(pair: dict) -> str:
    """Partition a logged robot/environment contact.

    The audited adapter's ``_collision_geom_pairs`` excludes robot-robot pairs and
    the floor, but it does not exclude the grasp target. Counting the grasp as a
    collision would make collision-free success identically zero for every
    successful pick, so contacts are classified here rather than in the adapter,
    which stays byte-identical to commit 3d25c69.
    """
    blob = " ".join(str(pair.get(k, "")) for k in ("geom1", "geom2", "body1", "body2",
                                                   "root1", "root2"))
    if TARGET_ROOT_PREFIX in blob:
        return "grasp_target"
    if HAZARD_BODY_PREFIX in blob:
        return "hazard_bar"
    return "other_environment"


def contact_breakdown(frames: list[dict]) -> dict:
    counts = {"grasp_target": 0, "hazard_bar": 0, "other_environment": 0}
    frames_with = {k: set() for k in counts}
    worst = {k: 0.0 for k in counts}
    examples: dict[str, list[str]] = {k: [] for k in counts}
    for f in frames:
        for pair in f["collision_geom_pairs"]:
            kind = classify_contact(pair)
            counts[kind] += 1
            frames_with[kind].add(f["step"])
            worst[kind] = max(worst[kind], max(0.0, -float(pair["distance_m"])))
            label = f"{pair['geom1']} <-> {pair['geom2']}"
            if label not in examples[kind] and len(examples[kind]) < 4:
                examples[kind].append(label)
    return {
        "pair_entries": counts,
        "frames_with_contact": {k: len(v) for k, v in frames_with.items()},
        "maximum_penetration_m": worst,
        "example_pairs": examples,
        "first_hazard_bar_contact_step": min(frames_with["hazard_bar"], default=None),
        "first_other_environment_contact_step": min(frames_with["other_environment"], default=None),
        "safety_relevant_contact_entries": counts["hazard_bar"] + counts["other_environment"],
    }


def load_run(path: Path) -> dict:
    doc = json.loads((path / "rollout.json").read_text())
    return doc["hybrid_safety_stack"]


def vec(frame: dict, key: str) -> np.ndarray:
    return np.asarray(frame[key], dtype=np.float64)


def collision_signature(frames: list[dict]) -> list[tuple]:
    sig = []
    for f in frames:
        for pair in f["collision_geom_pairs"]:
            sig.append((f["step"], pair["geom1"], pair["geom2"], round(pair["distance_m"], 9)))
    return sig


# --------------------------------------------------------------------------- #
# step 10 -- zero equivalence
# --------------------------------------------------------------------------- #
def zero_equivalence(act_only: dict, act_zero: dict) -> dict:
    a, z = act_only["frames"], act_zero["frames"]
    result: dict[str, Any] = {
        "act_only_frames": len(a),
        "act_plus_zero_frames": len(z),
        "frame_counts_equal": len(a) == len(z),
        "initial_state_equal": (
            act_only["manifest_provenance"]["replayed_initial_state_sha256"]
            == act_zero["manifest_provenance"]["replayed_initial_state_sha256"]
        ),
        "first_observation_fingerprint_equal": (
            act_only.get("first_observation_fingerprint")
            == act_zero.get("first_observation_fingerprint")
        ),
        "success_equal": act_only["episode_metrics"]["task_success"]
        == act_zero["episode_metrics"]["task_success"],
        "phase_sequence_equal": [f["task_phase"] for f in a] == [f["task_phase"] for f in z],
        "collision_sequence_equal": collision_signature(a) == collision_signature(z),
    }
    n = min(len(a), len(z))
    nominal_max = executed_max = correction_max = 0.0
    nominal_mismatch_steps: list[int] = []
    executed_mismatch_steps: list[int] = []
    gripper_mismatch_steps: list[int] = []
    for i in range(n):
        dn = float(np.max(np.abs(vec(a[i], "nominal_act_action") - vec(z[i], "nominal_act_action"))))
        de = float(np.max(np.abs(vec(a[i], "executed_action") - vec(z[i], "executed_action"))))
        dc = float(np.max(np.abs(vec(a[i], "correction") - vec(z[i], "correction"))))
        nominal_max = max(nominal_max, dn)
        executed_max = max(executed_max, de)
        correction_max = max(correction_max, dc)
        if dn != 0.0:
            nominal_mismatch_steps.append(i)
        if de != 0.0:
            executed_mismatch_steps.append(i)
        if vec(a[i], "executed_action")[-1] != vec(z[i], "executed_action")[-1]:
            gripper_mismatch_steps.append(i)
    result.update({
        "steps_compared": n,
        "max_abs_nominal_action_diff": nominal_max,
        "max_abs_executed_action_diff": executed_max,
        "max_abs_correction_diff": correction_max,
        "nominal_action_bitwise_equal": not nominal_mismatch_steps,
        "executed_action_bitwise_equal": not executed_mismatch_steps,
        "gripper_bitwise_equal": not gripper_mismatch_steps,
        "nominal_mismatch_steps": nominal_mismatch_steps[:10],
        "executed_mismatch_steps": executed_mismatch_steps[:10],
        "zero_correction_is_identically_zero": all(
            float(np.max(np.abs(vec(f, "correction")))) == 0.0 for f in z
        ),
    })
    result["ok"] = bool(
        result["frame_counts_equal"]
        and result["initial_state_equal"]
        and result["success_equal"]
        and result["phase_sequence_equal"]
        and result["collision_sequence_equal"]
        and result["nominal_action_bitwise_equal"]
        and result["executed_action_bitwise_equal"]
        and result["gripper_bitwise_equal"]
        and result["zero_correction_is_identically_zero"]
    )
    return result


# --------------------------------------------------------------------------- #
# step 13 -- paired pilot metrics
# --------------------------------------------------------------------------- #
def sensor_attribution(block: dict) -> dict:
    counts: dict[str, int] = {}
    links: dict[str, int] = {}
    known_only_frames = 0
    any_active_frames = 0
    for f in block["frames"]:
        active = f["active_sensors"]
        if active:
            any_active_frames += 1
            names = {row["sensor"] for row in active}
            if names.issubset(set(KNOWN_SELF_RETURN_SENSORS)):
                known_only_frames += 1
        for row in active:
            counts[row["sensor"]] = counts.get(row["sensor"], 0) + 1
            links[row["link"]] = links.get(row["link"], 0) + 1
    known = {s: counts.get(s, 0) for s in KNOWN_SELF_RETURN_SENSORS}
    other = {s: c for s, c in counts.items() if s not in KNOWN_SELF_RETURN_SENSORS}
    return {
        "frames_with_any_active_sensor": any_active_frames,
        "frames_where_only_known_self_return_sensors_active": known_only_frames,
        "known_self_return_activation_counts": known,
        "other_sensor_activation_counts": dict(sorted(other.items(), key=lambda kv: -kv[1])),
        "distinct_other_sensors_active": len(other),
        "link_activation_counts": dict(sorted(links.items(), key=lambda kv: -kv[1])),
        "known_sensors_dominate_raw_activation": bool(
            any_active_frames > 0 and known_only_frames == any_active_frames
        ),
    }


def pair_metrics(only: dict, safe: dict) -> dict:
    mo, ms = only["episode_metrics"], safe["episode_metrics"]
    fo, fs = only["frames"], safe["frames"]

    cb_o, cb_s = contact_breakdown(fo), contact_breakdown(fs)

    def collision_free_success(block: dict, breakdown: dict) -> bool:
        """Success with no *safety-relevant* contact.

        The grasp target is excluded: touching the cup is the task. Reported
        alongside the adapter's raw all-non-floor count so both readings are visible.
        """
        return bool(block["episode_metrics"]["task_success"]
                    and breakdown["safety_relevant_contact_entries"] == 0)

    def collision_free_success_raw(block: dict) -> bool:
        return bool(block["episode_metrics"]["task_success"]
                    and block["episode_metrics"]["collision_count"] == 0)

    # gripper equality: the safety condition must copy ACT's gripper bit-for-bit
    grip_equal = all(
        vec(fs[i], "executed_action")[-1] == vec(fs[i], "nominal_act_action")[-1]
        for i in range(len(fs))
    )
    arm_only_residual = all(
        float(np.max(np.abs(
            vec(f, "executed_action")[:7] - vec(f, "nominal_act_action")[:7] - vec(f, "correction")
        ))) < 1e-6
        for f in fs
    )
    finite = all(
        np.isfinite(vec(f, "executed_action")).all() and np.isfinite(vec(f, "correction")).all()
        for f in fs
    )
    clip_respected = all(
        float(np.max(np.abs(vec(f, "correction")))) <= 0.35 + 1e-6 for f in fs
    )
    raw_norms = [float(np.linalg.norm(vec(f, "raw_safety_dq"))) for f in fs]
    sub_norms = [float(np.linalg.norm(vec(f, "subtracted_dq"))) for f in fs]
    corr_norms = [float(np.linalg.norm(vec(f, "correction"))) for f in fs]

    return {
        "episode_id": safe["manifest_provenance"]["episode_id"],
        "candidate_index": safe["manifest_provenance"]["candidate_index"],
        "hazard_present": safe["manifest_provenance"]["hazard_present"],
        "initial_state_sha256": safe["manifest_provenance"]["replayed_initial_state_sha256"],
        "paired_initial_state_equal": (
            only["manifest_provenance"]["replayed_initial_state_sha256"]
            == safe["manifest_provenance"]["replayed_initial_state_sha256"]
        ),
        "paired_first_observation_equal": (
            only.get("first_observation_fingerprint") == safe.get("first_observation_fingerprint")
        ),
        "paired_first_nominal_action_equal": (
            only.get("first_nominal_act_action") == safe.get("first_nominal_act_action")
        ),
        "reference_mode": safe["controller"]["baseline"],
        "reference_skin_sha256": safe.get("reference_skin_sha256"),
        "initial_reference_minimum_depth_m": safe.get("reference_skin_minimum_depth_m"),
        "act_only": {
            "task_success": mo["task_success"],
            "collision_free_success": collision_free_success(only, cb_o),
            "collision_free_success_including_grasp": collision_free_success_raw(only),
            "collision_frames": mo["collision_count"],
            "contact_breakdown": cb_o,
            "maximum_penetration_m": mo["maximum_penetration_m"],
            "minimum_environment_clearance_m": mo["minimum_clearance_m"],
            "frames": len(fo),
            "duration_s": len(fo) * only["controller"]["dt"],
        },
        "act_plus_safety_live": {
            "task_success": ms["task_success"],
            "collision_free_success": collision_free_success(safe, cb_s),
            "collision_free_success_including_grasp": collision_free_success_raw(safe),
            "collision_frames": ms["collision_count"],
            "contact_breakdown": cb_s,
            "maximum_penetration_m": ms["maximum_penetration_m"],
            "minimum_environment_clearance_m": ms["minimum_clearance_m"],
            "frames": len(fs),
            "duration_s": len(fs) * safe["controller"]["dt"],
            "time_to_first_safety_activation_s": ms["time_to_first_safety_activation_s"],
            "maximum_correction_norm": ms["maximum_nominal_deviation_norm"],
            "integrated_residual_norm": ms["integrated_residual_norm"],
            "return_to_nominal_error": ms["return_to_nominal_error"],
            "raw_safety_output_norm_max": max(raw_norms, default=0.0),
            "raw_safety_output_norm_rms": float(np.sqrt(np.mean(np.square(raw_norms)))) if raw_norms else 0.0,
            "subtracted_output_norm_max": max(sub_norms, default=0.0),
            "subtracted_output_norm_rms": float(np.sqrt(np.mean(np.square(sub_norms)))) if sub_norms else 0.0,
            "correction_norm_rms": float(np.sqrt(np.mean(np.square(corr_norms)))) if corr_norms else 0.0,
        },
        "deltas": {
            "task_duration_delta_s": (len(fs) - len(fo)) * safe["controller"]["dt"],
            "minimum_clearance_delta_m": (
                (ms["minimum_clearance_m"] - mo["minimum_clearance_m"])
                if (ms["minimum_clearance_m"] is not None and mo["minimum_clearance_m"] is not None)
                else None
            ),
            "collision_frame_delta": ms["collision_count"] - mo["collision_count"],
            "safety_relevant_contact_delta": (
                cb_s["safety_relevant_contact_entries"] - cb_o["safety_relevant_contact_entries"]
            ),
            "hazard_bar_contact_delta": (
                cb_s["pair_entries"]["hazard_bar"] - cb_o["pair_entries"]["hazard_bar"]
            ),
        },
        "integration_checks": {
            "gripper_bitwise_unchanged_from_nominal": grip_equal,
            "residual_is_arm_only": arm_only_residual,
            "all_actions_finite": finite,
            "correction_within_max_dev": clip_respected,
        },
        "sensor_attribution": sensor_attribution(safe),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path, help="paired smoke output root")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="cand:only_dir:safety_dir triples")
    ap.add_argument("--zero-pair", nargs=2, metavar=("ONLY_DIR", "ZERO_DIR"), required=True)
    args = ap.parse_args()

    only_dir, zero_dir = (args.root / d for d in args.zero_pair)
    zero = zero_equivalence(load_run(only_dir), load_run(zero_dir))

    pairs = []
    for spec in args.pairs:
        _cand, od, sd = spec.split(":")
        pairs.append(pair_metrics(load_run(args.root / od), load_run(args.root / sd)))
    pairs.sort(key=lambda p: p["candidate_index"])

    present = [p for p in pairs if p["hazard_present"]]
    absent = [p for p in pairs if not p["hazard_present"]]

    def tally(rows, arm, field):
        return sum(1 for r in rows if r[arm][field])

    report = {
        "schema": "hybrid_obstacle_paired_smoke_audit",
        "statistical_disclaimer": (
            "Four pairs. No statistical superiority claim is made or supported. Exact McNemar "
            "testing for binary paired outcomes and paired trajectory bootstrap intervals for "
            "continuous deltas are prepared for the later 45-row evaluation and are deliberately "
            "not applied here."
        ),
        "contact_classification_note": (
            "The audited adapter logs every robot contact against a non-floor body, which "
            "includes the grasp target cavity_obj_0/<uid>. Touching the target IS the task, so "
            "contacts are partitioned here into grasp_target, hazard_bar (body protr_*) and "
            "other_environment. collision_free_success counts only the latter two; the raw "
            "all-non-floor reading is reported as collision_free_success_including_grasp. The "
            "adapter itself is unmodified."
        ),
        "clearance_metric_caveat": (
            "minimum_environment_clearance_m comes from the adapter's mj_geomDistance sweep over "
            "all non-floor environment geoms, which also includes the grasp target, so it goes "
            "negative during a successful grasp. Per-geom distances are not logged for "
            "non-contacting bodies, so a hazard-bar-only minimum distance cannot be recovered "
            "from these logs; hazard-bar proximity is reported via contact entries and the "
            "40-sensor activation depths instead."
        ),
        "known_self_return_sensors": list(KNOWN_SELF_RETURN_SENSORS),
        "known_self_return_note": (
            "Kept in the Safety-CVAE input to match its training contract. Their raw activation "
            "is reported separately and is not counted as obstacle-responsive evidence."
        ),
        "zero_equivalence": zero,
        "pairs": pairs,
        "totals": {
            "pairs": len(pairs),
            "hazard_present_pairs": len(present),
            "hazard_absent_pairs": len(absent),
            "act_only_task_success": tally(pairs, "act_only", "task_success"),
            "safety_task_success": tally(pairs, "act_plus_safety_live", "task_success"),
            "act_only_collision_free_success": tally(pairs, "act_only", "collision_free_success"),
            "safety_collision_free_success": tally(pairs, "act_plus_safety_live", "collision_free_success"),
        },
        "hazard_present": {
            "pairs": len(present),
            "act_only_task_success": tally(present, "act_only", "task_success"),
            "safety_task_success": tally(present, "act_plus_safety_live", "task_success"),
            "act_only_collision_frames": sum(p["act_only"]["collision_frames"] for p in present),
            "safety_collision_frames": sum(p["act_plus_safety_live"]["collision_frames"] for p in present),
            "act_only_hazard_bar_contacts": sum(
                p["act_only"]["contact_breakdown"]["pair_entries"]["hazard_bar"] for p in present),
            "safety_hazard_bar_contacts": sum(
                p["act_plus_safety_live"]["contact_breakdown"]["pair_entries"]["hazard_bar"] for p in present),
            "act_only_other_env_contacts": sum(
                p["act_only"]["contact_breakdown"]["pair_entries"]["other_environment"] for p in present),
            "safety_other_env_contacts": sum(
                p["act_plus_safety_live"]["contact_breakdown"]["pair_entries"]["other_environment"] for p in present),
        },
        "hazard_absent": {
            "pairs": len(absent),
            "act_only_task_success": tally(absent, "act_only", "task_success"),
            "safety_task_success": tally(absent, "act_plus_safety_live", "task_success"),
            "raw_safety_output_norm_max": [p["act_plus_safety_live"]["raw_safety_output_norm_max"] for p in absent],
            "subtracted_output_norm_max": [p["act_plus_safety_live"]["subtracted_output_norm_max"] for p in absent],
            "correction_norm_rms": [p["act_plus_safety_live"]["correction_norm_rms"] for p in absent],
            "correction_norm_max": [p["act_plus_safety_live"]["maximum_correction_norm"] for p in absent],
            "outcome_regressed": [
                bool(p["act_only"]["task_success"] and not p["act_plus_safety_live"]["task_success"])
                for p in absent
            ],
            "collisions_regressed": [
                bool(p["act_plus_safety_live"]["collision_frames"] > p["act_only"]["collision_frames"])
                for p in absent
            ],
        },
        "integration_all_pairs_ok": all(
            all(p["integration_checks"].values()) for p in pairs
        ),
        "all_pairs_replayed_identically": all(
            p["paired_initial_state_equal"] for p in pairs
        ),
        "duplicate_episode_ids": sorted(
            {p["episode_id"] for p in pairs if
             [q["episode_id"] for q in pairs].count(p["episode_id"]) > 1}
        ),
    }
    report["ok"] = bool(
        zero["ok"]
        and report["integration_all_pairs_ok"]
        and report["all_pairs_replayed_identically"]
        and not report["duplicate_episode_ids"]
        and report["totals"]["pairs"] == 4
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"zero equivalence          : {'PASS' if zero['ok'] else 'FAIL'}")
    for k in ("frame_counts_equal", "initial_state_equal", "nominal_action_bitwise_equal",
              "executed_action_bitwise_equal", "gripper_bitwise_equal",
              "phase_sequence_equal", "collision_sequence_equal", "success_equal",
              "zero_correction_is_identically_zero"):
        print(f"  {k:38} {zero[k]}")
    print(f"  max |executed action diff|             {zero['max_abs_executed_action_diff']:.3e}")
    print()
    print("pairs:")
    for p in pairs:
        o, s = p["act_only"], p["act_plus_safety_live"]
        print(f"  cand {p['candidate_index']:3d} {'present' if p['hazard_present'] else 'absent ':7} "
              f"ACT_ONLY succ={o['task_success']!s:5} coll={o['collision_frames']:3d} | "
              f"SAFETY succ={s['task_success']!s:5} coll={s['collision_frames']:3d} "
              f"maxcorr={s['maximum_correction_norm']:.4f}")
    print()
    print(f"integration checks all ok : {report['integration_all_pairs_ok']}")
    print(f"all pairs replay-identical: {report['all_pairs_replayed_identically']}")
    print(f"AUDIT {'OK' if report['ok'] else 'FAILED'}")
    print(f"wrote {args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
