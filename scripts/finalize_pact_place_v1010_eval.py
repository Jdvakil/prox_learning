#!/usr/bin/env python3
"""V10.10: aggregate the paired evaluation and report it honestly.

The primary endpoint is collision-free task success, matching the V5 reporting.
Paired differences are reported with discordant-pair counts and the exact
McNemar test, because 40 paired instances is a small sample and the sign of a
difference is not evidence on its own.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v1010_contract import (  # noqa: E402
    ACTIVE_CLUTTER_SLOTS,
    CONTRACT_VERSION_V1010,
    EVAL_ROOT,
    EVAL_INSTANCES,
    OBJECT_LABELS,
    TRAIN_COUNT,
    TRAIN_PER_CELL,
    TRAINING_ROOT,
    VALIDATION_COUNT,
    VALIDATION_PER_CELL,
    WORK_ROOT,
    canonical_payload_sha256,
    empty_authorization,
    write_immutable_create_only,
)
from pact_place_v1010_eval_contract import load_manifest  # noqa: E402

ARMS = ("ACT", "PACT")
CONTACT_CLASSES = ("clutter", "mounted_fixture", "hazard_bar", "other_environment",
                   "place_receptacle", "grasp_target")
# The historical V5 chunk-100 result, quoted for context only.
V5_CHUNK100_TASK_SUCCESS = {"ACT": 13, "PACT": 19, "n": 40}
PRIMARY_ENDPOINT = "collision_free_task_success"


def validate_completed_run(
        run: dict[str, Any], manifest: dict[str, Any], verification: dict[str, Any],
        *, eval_root: str) -> list[str]:
    """Fail closed before interpreting a partial or mismatched rollout set."""
    problems: list[str] = []
    expected_rollouts = 2 * len(manifest["rows"])
    if len(manifest["rows"]) != EVAL_INSTANCES:
        problems.append(f"manifest has {len(manifest['rows'])} instances, expected "
                        f"{EVAL_INSTANCES}")
    if run.get("schema_version") != "pact_place_v1010_full_run_v1":
        problems.append(f"wrong full-run schema {run.get('schema_version')!r}")
    if run.get("role") != manifest.get("role"):
        problems.append("full-run role does not match the manifest")
    if run.get("manifest_sha256") != manifest.get("manifest_sha256"):
        problems.append("full-run manifest hash does not match the loaded manifest")
    if run.get("eval_root") != eval_root:
        problems.append("full-run output root does not match --eval-root")
    if int(run.get("instances", -1)) != len(manifest["rows"]):
        problems.append("full-run instance count is not the manifest row count")
    if int(run.get("rollouts_attempted", -1)) != expected_rollouts:
        problems.append(f"full run did not attempt exactly {expected_rollouts} rollouts")
    if int(run.get("rollouts_complete", -1)) != expected_rollouts:
        problems.append(f"full run did not complete exactly {expected_rollouts} rollouts")
    if run.get("failures"):
        problems.append(f"full run records {len(run['failures'])} infrastructure failures")

    results = run.get("results") or []
    if len(results) != expected_rollouts:
        problems.append(f"full run contains {len(results)} result records, expected "
                        f"{expected_rollouts}")
    expected = {(arm, int(row["candidate_index"]))
                for arm in ARMS for row in manifest["rows"]}
    observed = [(str(result.get("arm")), int(result.get("candidate_index", -1)))
                for result in results]
    if len(set(observed)) != len(observed):
        problems.append("full run contains duplicate arm/candidate result records")
    if set(observed) != expected:
        problems.append("full-run arm/candidate identities do not match the manifest")
    for result in results:
        if result.get("status") != "complete" or int(result.get("returncode", -1)) != 0:
            problems.append(
                f"{result.get('arm')} candidate {result.get('candidate_index')}: "
                f"terminal status {result.get('status')!r}, return code "
                f"{result.get('returncode')!r}")

    for arm in ARMS:
        recorded = ((verification.get("arms") or {}).get(arm.lower()) or {}).get(
            "hashes", {}).get("policy_best.ckpt")
        executed = ((run.get("arms") or {}).get(arm) or {}).get("checkpoint_sha256")
        if not recorded or executed != recorded:
            problems.append(f"{arm} executed checkpoint does not match verification")
    return problems


def totals(result: dict[str, Any]) -> dict[str, int]:
    audit = result.get("contact_audit") or {}
    raw = audit.get("contact_class_totals") or {}
    return {name: int(raw.get(name, 0)) for name in CONTACT_CLASSES}


def stability_events(result: dict[str, Any]) -> int:
    """Recorded by the V10.10 policy: the expert's tracker never runs here."""
    info = result.get("policy_info") or {}
    if "clutter_stability_event_count" in info:
        return int(info["clutter_stability_event_count"] or 0)
    events = info.get("clutter_stability_events")
    if isinstance(events, list):
        return len(events)
    raise KeyError(
        "result carries no clutter-stability telemetry; the V10.10 evaluator "
        "must record it because the expert-policy tracker does not run for a "
        "learned policy")


def endpoints(result: dict[str, Any]) -> dict[str, Any]:
    """Every endpoint is recomputed from the audit, not read from a summary flag."""
    counts = totals(result)
    task_success = bool(result.get("task_success"))
    stability = stability_events(result)
    pendant_frames = counts["mounted_fixture"]
    collision_free = bool(result.get("collision_free_task_success"))
    # PactPlaceContactAudit.summary() defines
    #   non_target = hazard_bar + other_environment + clutter + mounted_fixture
    #   collision_free = (non_target == 0)
    # grasp_target and place_receptacle are excluded. This is the definition the
    # V5 chunk-100 evaluation reported; it coincided with the narrower
    # hazard_bar/other_environment test only because the V2 corridor had no
    # clutter and no mounted fixture. On real V9.5 clutter it does not.
    non_target = (counts["hazard_bar"] + counts["other_environment"]
                  + counts["clutter"] + counts["mounted_fixture"])
    recomputed_collision_free = bool(task_success and non_target == 0)
    strict_clean = bool(
        task_success
        and counts["clutter"] == 0
        and counts["mounted_fixture"] == 0
        and counts["hazard_bar"] == 0
        and counts["other_environment"] == 0
        and stability == 0)
    info = result.get("policy_info") or {}
    frames = (result.get("contact_audit") or {}).get("contact_frames") or []
    per_object = {slot: 0 for slot in ACTIVE_CLUTTER_SLOTS}
    for frame in frames:
        for pair in frame.get("pairs") or []:
            blob = " ".join(str(pair.get(k, "")) for k in
                            ("geom1", "geom2", "body1", "body2", "root1", "root2"))
            for slot in ACTIVE_CLUTTER_SLOTS:
                if f"pact_clutter_{slot}" in blob:
                    per_object[slot] += 1
    stability_by_object = {slot: 0 for slot in ACTIVE_CLUTTER_SLOTS}
    for event in info.get("clutter_stability_events") or []:
        for slot in ACTIVE_CLUTTER_SLOTS:
            if f"pact_clutter_{slot}" in str(event.get("body", "")):
                stability_by_object[slot] += 1
    funnel = info.get("v109r_funnel") or {}
    return {
        "per_object_contact_entries": per_object,
        "per_object_labels": {s: OBJECT_LABELS[s] for s in ACTIVE_CLUTTER_SLOTS},
        "per_object_stability": stability_by_object,
        "touched": bool(funnel.get("touched")),
        "held": bool(funnel.get("held")),
        "first_touch_step": funnel.get("first_touch_step"),
        "gripper_close_step": funnel.get("close_step"),
        "touch_to_close_controls": funnel.get("touch_to_close_controls"),
        "task_success": task_success,
        "collision_free_task_success": collision_free,
        "collision_free_task_success_recomputed": recomputed_collision_free,
        "collision_free_agrees": collision_free == recomputed_collision_free,
        "strict_clean_task_success": strict_clean,
        "pendant_free_task_success": bool(task_success and pendant_frames == 0),
        "pendant_contact_episode": pendant_frames > 0,
        "pendant_contact_frames": pendant_frames,
        "non_target_contact_entries": non_target,
        "clutter_contact_episode": counts["clutter"] > 0,
        "clutter_contact_frames": counts["clutter"],
        "clutter_stability_events": stability,
        "other_environment_contact_episode": counts["other_environment"] > 0,
        "other_environment_frames": counts["other_environment"],
        "place_receptacle_contact_episode": counts["place_receptacle"] > 0,
        "place_receptacle_frames": counts["place_receptacle"],
        "hazard_bar_frames": counts["hazard_bar"],
        # hazard_bar is the sampler-injected intrusion panel (pact_intrusion_*),
        # the corridor hazard itself. It is the dominant non-target class here
        # and the main driver of collision-free success, so it is reported and
        # paired rather than left inside the taxonomy.
        "hazard_bar_contact_episode": counts["hazard_bar"] > 0,
        "hazard_free_task_success": bool(task_success and counts["hazard_bar"] == 0),
        "gripper_close_commanded": bool(info.get("gripper_close_commanded")),
        "control_steps": int(info.get("control_steps", 0) or 0),
        "failure_taxonomy": result.get("failure_taxonomy"),
        "status": result.get("status"),
    }


def wilson(successes: int, total: int, z: float = 1.96) -> dict[str, float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"point": p, "low": max(0.0, centre - margin), "high": min(1.0, centre + margin)}


def exact_mcnemar(b: int, c: int) -> dict[str, Any]:
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return {"defined": False, "reason": "no discordant pairs",
                "discordant_act_only": c, "discordant_pact_only": b}
    observed = min(b, c)
    tail = sum(math.comb(n, k) for k in range(observed + 1)) / (2 ** n)
    p_value = min(1.0, 2 * tail)
    return {"defined": True, "discordant_pact_only": b, "discordant_act_only": c,
            "discordant_total": n, "p_value": p_value}


def paired(act: list[bool], pact: list[bool]) -> dict[str, Any]:
    if len(act) != len(pact):
        raise ValueError("paired arms differ in length")
    n = len(act)
    b = sum(1 for a, p in zip(act, pact) if p and not a)   # PACT only
    c = sum(1 for a, p in zip(act, pact) if a and not p)   # ACT only
    both = sum(1 for a, p in zip(act, pact) if a and p)
    neither = n - b - c - both
    difference = (b - c) / n if n else 0.0
    variance = (b + c - (b - c) ** 2 / n) / (n * n) if n else 0.0
    se = math.sqrt(max(variance, 0.0))
    return {
        "n_pairs": n,
        "act_successes": sum(act), "pact_successes": sum(pact),
        "both": both, "neither": neither,
        "discordant_pact_only": b, "discordant_act_only": c,
        "paired_difference_pp": round(100 * difference, 2),
        "paired_95_interval_pp": [round(100 * (difference - 1.96 * se), 2),
                                  round(100 * (difference + 1.96 * se), 2)],
        "interval_crosses_zero":
            (difference - 1.96 * se) <= 0 <= (difference + 1.96 * se),
        "exact_mcnemar": exact_mcnemar(b, c),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=str, default=EVAL_ROOT)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / EVAL_ROOT / "eval_manifest.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    eval_root = args.eval_root
    if args.out is None:
        args.out = ROOT / eval_root / "analysis.json"

    manifest = load_manifest(args.manifest)
    run = json.loads((ROOT / eval_root / "full_run.json").read_text())
    verification = json.loads(
        (ROOT / WORK_ROOT / "training_verification.json").read_text())
    rows = {int(r["candidate_index"]): r for r in manifest["rows"]}

    per_arm: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    problems = validate_completed_run(
        run, manifest, verification, eval_root=eval_root)
    for arm in ARMS:
        directory = ROOT / eval_root / "rollouts" / arm.lower()
        for index, row in rows.items():
            path = (directory / f"{index:03d}_{row['episode_id'][:16]}" / "result.json")
            if not path.is_file():
                problems.append(f"{arm} candidate {index}: missing result")
                continue
            result = json.loads(path.read_text())
            if result["episode_id"] != row["episode_id"]:
                problems.append(f"{arm} candidate {index}: episode id mismatch")
            if result["row_sha256"] != row["row_sha256"]:
                problems.append(f"{arm} candidate {index}: row hash mismatch")
            payload = endpoints(result)
            if not payload["collision_free_agrees"]:
                problems.append(
                    f"{arm} candidate {index}: recorded collision-free flag disagrees "
                    "with the recomputation")
            payload.update({
                "candidate_index": index, "cell": row["cell"],
                "family_id": row["family_id"], "intrusion_side": row["intrusion_side"],
                "pose_id": row["pose_id"], "episode_id": row["episode_id"],
                "task_seed_u32": int(row["task_seed_u32"]),
            })
            per_arm[arm][index] = payload

    paired_indices = sorted(set(per_arm["ACT"]) & set(per_arm["PACT"]))
    if len(paired_indices) != len(rows):
        problems.append(f"{len(paired_indices)} paired instances of {len(rows)}")

    binary = ("touched", "held",
              "task_success", "collision_free_task_success", "strict_clean_task_success",
              "pendant_free_task_success", "pendant_contact_episode",
              "clutter_contact_episode", "other_environment_contact_episode",
              "place_receptacle_contact_episode", "gripper_close_commanded",
              "hazard_bar_contact_episode", "hazard_free_task_success")

    def arm_summary(arm: str) -> dict[str, Any]:
        entries = [per_arm[arm][i] for i in paired_indices]
        n = len(entries)
        summary: dict[str, Any] = {"n": n}
        for key in binary:
            successes = sum(1 for e in entries if e[key])
            summary[key] = {"count": successes, "rate": successes / n if n else None,
                            "wilson_95": wilson(successes, n)}
        summary["per_object"] = {
            slot: {
                "label": OBJECT_LABELS[slot],
                "contact_episodes": sum(
                    1 for e in entries if e["per_object_contact_entries"][slot]),
                "contact_entries": sum(
                    e["per_object_contact_entries"][slot] for e in entries),
                "stability_events": sum(
                    e["per_object_stability"][slot] for e in entries),
            } for slot in ACTIVE_CLUTTER_SLOTS}
        summary["funnel"] = {
            "touched": sum(1 for e in entries if e["touched"]),
            "held": sum(1 for e in entries if e["held"]),
            "task_success": sum(1 for e in entries if e["task_success"]),
            "touch_to_close_controls": sorted(
                e["touch_to_close_controls"] for e in entries
                if e["touch_to_close_controls"] is not None),
        }
        summary["hazard_bar_frames_total"] = sum(
            e["hazard_bar_frames"] for e in entries)
        summary["pendant_contact_frames_total"] = sum(
            e["pendant_contact_frames"] for e in entries)
        summary["clutter_contact_frames_total"] = sum(
            e["clutter_contact_frames"] for e in entries)
        summary["clutter_stability_events_total"] = sum(
            e["clutter_stability_events"] for e in entries)
        summary["other_environment_frames_total"] = sum(
            e["other_environment_frames"] for e in entries)
        summary["place_receptacle_frames_total"] = sum(
            e["place_receptacle_frames"] for e in entries)
        steps = sorted(e["control_steps"] for e in entries)
        summary["control_steps"] = {
            "min": steps[0] if steps else None, "max": steps[-1] if steps else None,
            "mean": sum(steps) / n if n else None,
            "median": steps[n // 2] if steps else None,
            "at_horizon": sum(1 for s in steps if s >= manifest["task_horizon"]),
        }
        summary["failure_taxonomy"] = dict(sorted(collections.Counter(
            str(e["failure_taxonomy"]) for e in entries
            if not e["task_success"]).items()))
        for axis in ("family_id", "intrusion_side", "pose_id", "cell"):
            stratum: dict[str, dict[str, Any]] = {}
            for entry in entries:
                bucket = stratum.setdefault(
                    entry[axis], {"n": 0, "task_success": 0,
                                  "collision_free_task_success": 0,
                                  "strict_clean_task_success": 0,
                                  "pendant_contact_episode": 0})
                bucket["n"] += 1
                for key in ("task_success", "collision_free_task_success",
                            "strict_clean_task_success", "pendant_contact_episode"):
                    bucket[key] += int(entry[key])
            summary[f"by_{axis}"] = dict(sorted(stratum.items()))
        return summary

    arms = {arm: arm_summary(arm) for arm in ARMS}
    paired_results = {
        key: paired([per_arm["ACT"][i][key] for i in paired_indices],
                    [per_arm["PACT"][i][key] for i in paired_indices])
        for key in ("task_success", "collision_free_task_success",
                    "strict_clean_task_success", "pendant_free_task_success",
                    "hazard_bar_contact_episode", "clutter_contact_episode",
                    "touched", "held")
    }

    outcome_table = [{
        "candidate_index": i,
        "cell": per_arm["ACT"][i]["cell"],
        "task_seed_u32": per_arm["ACT"][i]["task_seed_u32"],
        **{f"act_{k}": per_arm["ACT"][i][k] for k in
           ("task_success", "collision_free_task_success", "strict_clean_task_success",
            "pendant_contact_episode", "clutter_contact_frames", "control_steps",
            "gripper_close_commanded")},
        **{f"pact_{k}": per_arm["PACT"][i][k] for k in
           ("task_success", "collision_free_task_success", "strict_clean_task_success",
            "pendant_contact_episode", "clutter_contact_frames", "control_steps",
            "gripper_close_commanded")},
    } for i in paired_indices]

    primary = paired_results[PRIMARY_ENDPOINT]
    document: dict[str, Any] = {
        **empty_authorization(),
        "schema_version": "pact_place_v1010_eval_analysis_v1",
        "contract_version": CONTRACT_VERSION_V1010,
        "role": "paired ACT vs PACT held-out evaluation in the V10.10 "
                "four-object environment over the certified V10.7 pendant scenes",
        "is_phase0_pass": False,
        "exploratory": True,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "active_clutter_slots": list(ACTIVE_CLUTTER_SLOTS),
        "object_labels": dict(OBJECT_LABELS),
        "manifest_sha256": manifest["manifest_sha256"],
        "eval_root": eval_root,
        "run_payload_sha256": run["payload_sha256"],
        "training_verification_payload_sha256": verification["payload_sha256"],
        "checkpoints": {arm: run["arms"][arm]["checkpoint_sha256"] for arm in ARMS},
        "instances": len(paired_indices),
        "rollouts": 2 * len(paired_indices),
        "arms": arms,
        "paired": paired_results,
        "outcome_table": outcome_table,
        "historical_context": {
            "v5_chunk100_task_success": V5_CHUNK100_TASK_SUCCESS,
            "comparable": False,
            "note": "Quoted for context only. The environment (the V10.10 four-object "
                    "layout over the certified V10.7 static-pendant scenes vs the V2 "
                    "corridor) and the training corpus (144 balanced V10.10 "
                    "demonstrations vs 152 V5 demonstrations) both changed, so this "
                    "is not a like-for-like comparison.",
        },
        "run_pact_permuted": False,
        "interpretation_limits": [
            "single seed (3101), one training run per arm, no replication",
            "40 paired instances; the paired interval is wide by construction",
            f"the primary endpoint interval "
            f"{'crosses' if primary['interval_crosses_zero'] else 'does not cross'} zero",
            "validation loss alone is not evidence of policy superiority",
            f"the training corpus is balanced across all 24 cells: "
            f"{TRAIN_PER_CELL} train and {VALIDATION_PER_CELL} validation per cell "
            f"({TRAIN_COUNT}/{VALIDATION_COUNT} total)",
            "V10.7's Phase-0 gate failed at 8/24 and is permanently closed; nothing "
            "here reopens it",
        ],
        "problems": problems,
        "verified": not problems,
    }
    document["payload_sha256"] = canonical_payload_sha256(document)
    written = write_immutable_create_only(args.out, document)
    print(json.dumps({
        "verified": document["verified"], "problems": problems[:8],
        "instances": document["instances"],
        "act": {k: arms["ACT"][k]["count"] for k in
                ("task_success", "collision_free_task_success",
                 "strict_clean_task_success", "pendant_contact_episode")},
        "pact": {k: arms["PACT"][k]["count"] for k in
                 ("task_success", "collision_free_task_success",
                  "strict_clean_task_success", "pendant_contact_episode")},
        "paired_primary": primary,
        "payload_sha256": document["payload_sha256"],
        "raw_file_sha256": written.get("raw_file_sha256"),
    }, indent=2))
    return 0 if document["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
