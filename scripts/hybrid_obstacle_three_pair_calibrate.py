#!/usr/bin/env python3
"""Joint calibration with a selectable controlling agreement metric.

This is the previous joint-gate calibration with exactly one behavioural change: which
agreement definition drives execution. ``--agreement-metric three_pair`` uses the validated
mean of all three pairwise Jaccards; ``anchor`` reproduces the previous two-anchor run. Every
threshold grid, feasibility floor, selection ordering, gate and partition is untouched, so a
difference in outcome is attributable to the metric alone.

Original docstring follows.

Joint calibration of the seed-0 activity threshold and the full-seed agreement threshold.

Handoff steps 4-13. Both thresholds are fitted together as one two-dimensional contract: the
previous task showed they cannot be chosen independently, because a threshold calibrated for
a single-gate controller leaves no recall headroom for a veto to spend.

The full Cartesian product is evaluated, but not naively. Several conditions factorise --
agreement acceptance depends only on the agreement threshold, activity-alone recall only on
the activity threshold -- so those screens run first and the expensive per-trajectory
sequence metrics are computed only for pairs that could still be feasible. The rule is
unchanged; only the order of evaluation is.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from causal_parked_skin import threshold as thr
from causal_parked_skin.data import SOURCE_MODES, load_partition
from causal_parked_skin.engine import load_checkpoint, make_batch, set_sensor_names
from causal_parked_skin.joint_gate import (
    MODE,
    PIXEL_MASK_THRESHOLD,
    anchor_mask_agreement,
    changed_mask,
    jaccard,
    joint_decision,
    three_pair_agreement,
    validate_seed_roster,
)
from causal_parked_skin.model import BASELINE_CURRENT, FrozenSafetyHead

CONSUMED_PARTITIONS = ("offline_reference_test", "reference_calibration",
                       "reference_validation")

# coverage (the owner decision: the recall floor does not move)
MIN_MEDIAN_ACTIVE_RECALL = 0.80
MIN_MEDIAN_HARD_RETENTION = 0.80
MIN_RETAINED_COSINE = 0.75
MIN_RETAINED_POSITIVE_COSINE = 0.85
# false activation
MAX_BOOTSTRAP_UPPER_FALSE_ACTIVATION = 0.02
MAX_HAZARD_ABSENT_EXECUTED = 0.05
MAX_FALSE_ACTIVE_RUN = 2
# uncertainty anti-degeneracy
MIN_ZERO_ACCEPTANCE = 0.80
MIN_ACTIVE_ACCEPTANCE = 0.80
MIN_INACTIVE_ACCEPTANCE = 0.80
MAX_TRAJECTORY_ABSTENTION = 0.50
MAX_HAZARD_PRESENT_ACTIVE_ABSTENTION = 0.40
# activity anti-degeneracy
MIN_ACTIVITY_ALONE_ACTIVE_RETENTION = 0.85

# nested offline gates (step 11)
NESTED_MIN_ACTIVE_RECALL = 0.75
NESTED_MAX_MEAN_EXECUTED = 0.02
NESTED_MAX_HAZARD_ABSENT_EXECUTED = 0.05
# reused diagnostic gates (step 13)
DIAG_MAX_MEAN_EXECUTED = 0.03
DIAG_MAX_HAZARD_ABSENT_EXECUTED = 0.10
DIAG_MIN_ACTIVE_RECALL = 0.70
DIAG_MIN_HARD_RETENTION = 0.80
DIAG_MAX_RUN = 5
DIAG_MIN_ZERO_ACCEPTANCE = 0.75
DIAG_MAX_ABSTENTION = 0.60

OLD_ACTIVITY_THRESHOLD = 0.99960857629776
INFERENCE_REPEATS = 24
INFERENCE_TOLERANCE = 1e-7


class Trajectory:
    def __init__(self, meta, activity, agreement, three_pair, cosine, oracle_active,
                 hard_active, masks, pair_jaccard) -> None:
        self.__dict__.update(meta)
        self.activity = activity
        # `agreement` is whichever metric controls execution for this run; the other is
        # retained alongside so every decision can be audited against both definitions.
        self.agreement = agreement
        self.three_pair = three_pair
        self.cosine = cosine
        self.oracle_active = oracle_active.astype(bool)
        self.hard_active = hard_active.astype(bool)
        self.masks = masks
        self.pair_jaccard = pair_jaccard
        self.frames = len(activity)

    def metrics_at(self, activity_threshold: float, agreement_threshold: float) -> dict:
        gates = joint_decision(self.activity, activity_threshold,
                               self.agreement, agreement_threshold)
        execute = gates["execute"]
        active, zero = self.oracle_active, ~self.oracle_active
        agreement_pass, activity_pass = gates["agreement_pass"], gates["activity_pass"]
        retained = execute & active
        false_active = execute & zero
        cosines = self.cosine[retained]
        return {
            "trajectory_id": self.trajectory_id, "episode_id": self.episode_id,
            "distribution": self.distribution,
            "hazard_present": bool(self.hazard_present), "frames": self.frames,
            "oracle_active_frames": int(active.sum()),
            "oracle_zero_frames": int(zero.sum()),
            "active_recall": float(retained.sum() / active.sum()) if active.any() else None,
            "activity_alone_active_retention": float(activity_pass[active].mean())
            if active.any() else None,
            "hard_true_active_retention": float(execute[self.hard_active].mean())
            if self.hard_active.any() else None,
            "executed_false_activation_rate": float(false_active.sum() / zero.sum())
            if zero.any() else None,
            "executed_active_fraction": float(execute.mean()),
            "max_false_active_run": thr.max_true_run(false_active),
            "persistent_after_support": bool(
                thr.persists_after_oracle(false_active, active)) if active.any()
            else bool(false_active.any()),
            "median_retained_cosine": float(np.median(cosines)) if cosines.size else None,
            "retained_positive_cosine_fraction": float((cosines > 0).mean())
            if cosines.size else None,
            "agreement_acceptance_zero": float(agreement_pass[zero].mean())
            if zero.any() else None,
            "agreement_acceptance_active": float(agreement_pass[active].mean())
            if active.any() else None,
            "agreement_acceptance_activity_inactive": float(
                agreement_pass[~activity_pass].mean()) if (~activity_pass).any() else None,
            "uncertainty_abstention_fraction": float((~agreement_pass).mean()),
            "active_frame_abstention_fraction": float(
                (~agreement_pass)[active].mean()) if active.any() else None,
        }


def aggregate(trajectories, activity_threshold, agreement_threshold) -> dict:
    rows = [t.metrics_at(activity_threshold, agreement_threshold) for t in trajectories]

    def g(key, predicate=lambda r: True):
        return [r[key] for r in rows if predicate(r) and r[key] is not None]

    hazard_absent_exec = [r["executed_active_fraction"] for r in rows
                          if not r["hazard_present"]]
    return {
        "activity_threshold": float(activity_threshold),
        "agreement_threshold": float(agreement_threshold),
        "trajectories": rows,
        "median_active_recall": float(np.median(g("active_recall")))
        if g("active_recall") else None,
        "median_activity_alone_retention": float(np.median(
            g("activity_alone_active_retention")))
        if g("activity_alone_active_retention") else None,
        "median_hard_retention": float(np.median(g("hard_true_active_retention")))
        if g("hard_true_active_retention") else None,
        "mean_executed_false_activation": float(np.mean(
            g("executed_false_activation_rate")))
        if g("executed_false_activation_rate") else None,
        "max_hazard_absent_executed_fraction": max(hazard_absent_exec, default=None),
        "max_false_active_run": max(r["max_false_active_run"] for r in rows),
        "any_persistent_after_support": any(r["persistent_after_support"] for r in rows),
        "median_retained_cosine": float(np.median(g("median_retained_cosine")))
        if g("median_retained_cosine") else None,
        "median_retained_positive_cosine": float(np.median(
            g("retained_positive_cosine_fraction")))
        if g("retained_positive_cosine_fraction") else None,
        "mean_agreement_acceptance_zero": float(np.mean(g("agreement_acceptance_zero")))
        if g("agreement_acceptance_zero") else None,
        "mean_agreement_acceptance_active": float(np.mean(
            g("agreement_acceptance_active"))) if g("agreement_acceptance_active") else None,
        "mean_agreement_acceptance_inactive": float(np.mean(
            g("agreement_acceptance_activity_inactive")))
        if g("agreement_acceptance_activity_inactive") else None,
        "max_trajectory_abstention": max(r["uncertainty_abstention_fraction"]
                                         for r in rows),
        "max_hazard_present_active_abstention": max(
            [r["active_frame_abstention_fraction"] for r in rows
             if r["hazard_present"] and r["active_frame_abstention_fraction"] is not None],
            default=0.0),
        "by_distribution": {n: sum(1 for r in rows if r["distribution"] == n)
                            for n in SOURCE_MODES},
    }


def check_contract(block, upper_bound) -> dict:
    return {
        "median_active_recall": (block["median_active_recall"] or 0)
        >= MIN_MEDIAN_ACTIVE_RECALL,
        "median_hard_retention": (block["median_hard_retention"] is None
                                  or block["median_hard_retention"]
                                  >= MIN_MEDIAN_HARD_RETENTION),
        "retained_cosine": (block["median_retained_cosine"] or 0) >= MIN_RETAINED_COSINE,
        "retained_positive_cosine": (block["median_retained_positive_cosine"] or 0)
        >= MIN_RETAINED_POSITIVE_COSINE,
        "bootstrap_upper_false_activation": upper_bound
        <= MAX_BOOTSTRAP_UPPER_FALSE_ACTIVATION,
        "hazard_absent_executed": (block["max_hazard_absent_executed_fraction"] is None
                                   or block["max_hazard_absent_executed_fraction"]
                                   <= MAX_HAZARD_ABSENT_EXECUTED),
        "false_active_run": block["max_false_active_run"] <= MAX_FALSE_ACTIVE_RUN,
        "no_persistent_correction": not block["any_persistent_after_support"],
        "zero_acceptance": (block["mean_agreement_acceptance_zero"] or 0)
        >= MIN_ZERO_ACCEPTANCE,
        "active_acceptance": (block["mean_agreement_acceptance_active"] or 0)
        >= MIN_ACTIVE_ACCEPTANCE,
        "inactive_acceptance": (block["mean_agreement_acceptance_inactive"] is None
                                or block["mean_agreement_acceptance_inactive"]
                                >= MIN_INACTIVE_ACCEPTANCE),
        "trajectory_abstention_cap": block["max_trajectory_abstention"]
        <= MAX_TRAJECTORY_ABSTENTION,
        "hazard_present_active_abstention":
            block["max_hazard_present_active_abstention"]
            <= MAX_HAZARD_PRESENT_ACTIVE_ABSTENTION,
        "activity_alone_retention": (block["median_activity_alone_retention"] or 0)
        >= MIN_ACTIVITY_ALONE_ACTIVE_RETENTION,
    }


def score(models, head, partition, episodes, device, hard_threshold, *, batch=256,
          metric="three_pair"):
    import torch

    seed0, seed1, seed2 = models
    trajectory_index = np.asarray(partition["trajectory"])
    oracle_active = np.asarray(partition["oracle_active"]).astype(bool)
    hazard = np.asarray(partition["hazard_present"]).astype(bool)
    modes = np.asarray(partition["source_mode"])
    oracle_dq = np.asarray(partition["oracle_dq"], dtype=np.float64)
    keep = [i for i, e in enumerate(partition.episode_ids) if e in episodes]

    out = []
    for index in keep:
        rows = np.flatnonzero(trajectory_index == index)
        n = len(rows)
        activity = np.zeros(n)
        anchor = np.zeros(n)
        three = np.zeros(n)
        cosine = np.zeros(n)
        counts = np.zeros((3, n), dtype=np.int64)
        pairs = np.zeros((3, n))
        for start in range(0, n, batch):
            chunk = rows[start:start + batch]
            slot = slice(start, start + len(chunk))
            b = make_batch(partition, chunk, device)
            with torch.no_grad():
                a0 = seed0(b["history"], b["history_valid"], b["state"])
                p0 = a0["changed_probability"].cpu().numpy()
                dq = (head(a0["current"]) - head(a0["parked"])).double().cpu().numpy()
                # seeds 1 and 2 contribute masks only; their fields are never read
                p1 = seed1(b["history"], b["history_valid"],
                           b["state"])["changed_probability"].cpu().numpy()
                p2 = seed2(b["history"], b["history_valid"],
                           b["state"])["changed_probability"].cpu().numpy()
            m0, m1, m2 = changed_mask(p0), changed_mask(p1), changed_mask(p2)
            activity[slot] = p0.reshape(len(chunk), -1).max(axis=1)
            anchor[slot] = anchor_mask_agreement(m0, m1, m2)
            three[slot] = three_pair_agreement(m0, m1, m2)
            counts[0, slot] = m0.sum(axis=1)
            counts[1, slot] = m1.sum(axis=1)
            counts[2, slot] = m2.sum(axis=1)
            pairs[0, slot] = jaccard(m0, m1)
            pairs[1, slot] = jaccard(m0, m2)
            pairs[2, slot] = jaccard(m1, m2)
            true_dq = oracle_dq[chunk]
            denominator = (np.linalg.norm(dq, axis=-1)
                           * np.linalg.norm(true_dq, axis=-1) + thr.COSINE_EPSILON)
            cosine[slot] = (dq * true_dq).sum(-1) / denominator
        out.append(Trajectory(
            {"trajectory_id": partition.trajectory_ids[index],
             "episode_id": partition.episode_ids[index],
             "distribution": SOURCE_MODES[int(modes[rows[0]])],
             "hazard_present": bool(hazard[rows[0]]), "partition": partition.name},
            activity, three if metric == "three_pair" else anchor,
            anchor if metric == "three_pair" else three,
            cosine, oracle_active[rows],
            oracle_active[rows] & (activity < hard_threshold), counts, pairs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--checkpoint-root", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--groups", required=True, type=Path)
    ap.add_argument("--onset-audit", required=True, type=Path)
    ap.add_argument("--agreement-metric", choices=("three_pair", "anchor"),
                    default="three_pair",
                    help="which agreement definition controls execution")
    ap.add_argument("--deployment-manifest-out", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch

    stack = json.loads(args.stack.read_text())
    set_sensor_names(stack["sensor_contract"]["ordered_names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = FrozenSafetyHead.load(args.safety_dir, device=device)
    spec = json.loads(args.partition.read_text())
    hard_threshold = json.loads(args.groups.read_text())["historical_activity_range"][1]

    validate_seed_roster([0, 1, 2])
    models, checkpoints = [], []
    for seed in (0, 1, 2):
        path = args.checkpoint_root / f"CURRENT_FRAME_ONLY__seed{seed}" / "best.pt"
        model, payload = load_checkpoint(path, device)
        if payload["config"]["seed"] != seed or \
                payload["config"]["variant"] != BASELINE_CURRENT:
            raise SystemExit(f"seed {seed} checkpoint mismatch")
        models.append(model)
        checkpoints.append(path)
    print(f"loaded full-data seeds 0/1/2 ({MODE})")

    train_partition = load_partition(args.cache, "reference_train")
    started = time.time()

    calibration_episodes = set(spec["splits"]["threshold_calibration"]["episodes"])
    metric = args.agreement_metric
    print(f"controlling agreement metric: {metric}")
    calibration = score(models, head, train_partition, calibration_episodes, device,
                        hard_threshold, metric=metric)
    episodes = sorted({t.episode_id for t in calibration})
    cluster_of = {e: i for i, e in enumerate(episodes)}
    cluster_of_row = np.array([cluster_of[t.episode_id] for t in calibration])

    activity_values = np.unique(np.concatenate(
        [t.activity for t in calibration] + [np.array([0.0, 1.0])]))
    agreement_values = np.unique(np.concatenate(
        [t.agreement for t in calibration] + [np.array([0.0, 1.0])]))
    print(f"grid: {activity_values.size} activity x {agreement_values.size} agreement "
          f"= {activity_values.size * agreement_values.size} pairs")

    # --- factorised screens: these depend on only one axis each ---------------------
    activity_alone = np.array([
        float(np.median([float((t.activity[t.oracle_active] >= a).mean())
                         for t in calibration if t.oracle_active.any()]))
        for a in activity_values])
    activity_ok = activity_alone >= MIN_ACTIVITY_ALONE_ACTIVE_RETENTION

    zero_accept = np.array([
        float(np.mean([float((t.agreement[~t.oracle_active] >= u).mean())
                       for t in calibration if (~t.oracle_active).any()]))
        for u in agreement_values])
    active_accept = np.array([
        float(np.mean([float((t.agreement[t.oracle_active] >= u).mean())
                       for t in calibration if t.oracle_active.any()]))
        for u in agreement_values])
    abstention_cap = np.array([
        max(float((t.agreement < u).mean()) for t in calibration)
        for u in agreement_values])
    agreement_ok = ((zero_accept >= MIN_ZERO_ACCEPTANCE)
                    & (active_accept >= MIN_ACTIVE_ACCEPTANCE)
                    & (abstention_cap <= MAX_TRAJECTORY_ABSTENTION))
    print(f"  activity axis passing its own screen : {int(activity_ok.sum())}"
          f"/{activity_values.size}")
    print(f"  agreement axis passing its own screen: {int(agreement_ok.sum())}"
          f"/{agreement_values.size}")

    # For a fixed agreement threshold the executed set shrinks monotonically in A, so each
    # trajectory's false-activation rate across the whole activity axis is a searchsorted
    # lookup and the cluster bootstrap runs once per agreement value instead of once per
    # pair. Same rule, same grid; only the arithmetic order changes.
    candidate_a = activity_values[np.flatnonzero(activity_ok)]
    feasible = []
    evaluated = 0
    for ui in np.flatnonzero(agreement_ok):
        u = float(agreement_values[ui])
        rates = np.zeros((len(calibration), candidate_a.size))
        for row, t in enumerate(calibration):
            zero = ~t.oracle_active
            denominator = int(zero.sum())
            if denominator == 0:
                continue
            values = np.sort(t.activity[zero & (t.agreement >= u)])
            above = values.size - np.searchsorted(values, candidate_a, side="left")
            rates[row] = above / denominator
        upper = thr.cluster_bootstrap_upper_bound(
            thr.cluster_means(rates, cluster_of_row, len(episodes)))
        for position, a in enumerate(candidate_a):
            if upper[position] > MAX_BOOTSTRAP_UPPER_FALSE_ACTIVATION:
                continue
            block = aggregate(calibration, float(a), u)
            evaluated += 1
            checks = check_contract(block, float(upper[position]))
            if all(checks.values()):
                feasible.append({"activity_threshold": float(a),
                                 "agreement_threshold": u,
                                 "upper": float(upper[position]),
                                 "block": block, "checks": checks})
    print(f"  pairs fully evaluated: {evaluated}; feasible: {len(feasible)}")

    if not feasible:
        # record the frontier along each axis so the failure is diagnosable
        frontier = []
        for a in np.quantile(activity_values, np.linspace(0, 1, 11)):
            for u in np.quantile(agreement_values, np.linspace(0, 1, 11)):
                block = aggregate(calibration, float(a), float(u))
                rates = np.array([[r["executed_false_activation_rate"] or 0.0]
                                  for r in block["trajectories"]])
                upper = float(thr.cluster_bootstrap_upper_bound(
                    thr.cluster_means(rates, cluster_of_row, len(episodes)),
                    replicates=2000)[0])
                checks = check_contract(block, upper)
                frontier.append({
                    "activity_threshold": float(a), "agreement_threshold": float(u),
                    "median_active_recall": block["median_active_recall"],
                    "activity_alone_retention": block["median_activity_alone_retention"],
                    "mean_executed_false_activation":
                        block["mean_executed_false_activation"],
                    "zero_acceptance": block["mean_agreement_acceptance_zero"],
                    "bootstrap_upper": upper,
                    "failed": sorted(k for k, v in checks.items() if not v)})
        report = {"schema": "hybrid_obstacle_joint_gate_calibration_v1", "mode": MODE,
                  "feasible": False, "grid": {"activity": int(activity_values.size),
                                              "agreement": int(agreement_values.size)},
                  "axis_screens": {
                      "activity_axis_passing": int(activity_ok.sum()),
                      "agreement_axis_passing": int(agreement_ok.sum())},
                  "frontier": frontier,
                  "decision_if_infeasible":
                      "FULL_SEED_JOINT_GATE_CALIBRATION_INFEASIBLE"}
        report["report_sha256"] = thr.canonical_hash(report)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True,
                                       default=str) + "\n")
        print("NO FEASIBLE (activity, agreement) PAIR")
        return 7

    chosen = min(feasible, key=lambda f: (
        f["upper"], -(f["block"]["median_active_recall"] or 0),
        -(f["block"]["median_hard_retention"] or 0),
        -(f["block"]["mean_agreement_acceptance_zero"] or 0),
        -f["activity_threshold"], -f["agreement_threshold"]))
    A = chosen["activity_threshold"]
    U = chosen["agreement_threshold"]
    print(f"selected: activity {A:.8f}  agreement {U:.6f}  "
          f"(bootstrap upper {chosen['upper']:.5f}, "
          f"recall {chosen['block']['median_active_recall']:.4f})")

    # ---- freeze the deployment manifest before opening anything else ---------------
    import hashlib

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    agreement_source = (ROOT / "causal_parked_skin" / "joint_gate.py").read_bytes()
    deployment = {
        "schema": "hybrid_obstacle_full_seed_joint_gate_manifest_v1",
        "mode": MODE,
        "condition": "ACT_PLUS_FULL_SEED_JOINT_GATE",
        "deployment_seed": 0,
        "uncertainty_seeds": [1, 2],
        "seed_checkpoints": [str(p) for p in checkpoints],
        "seed_checkpoint_sha256": [sha(p) for p in checkpoints],
        "full_data_training_partition": "reference_train",
        "partition_manifest_sha256": spec["manifest_sha256"],
        "activity_threshold": A,
        "agreement_threshold": U,
        "retired_activity_threshold": OLD_ACTIVITY_THRESHOLD,
        "agreement_implementation_sha256": hashlib.sha256(agreement_source).hexdigest(),
        "controlling_metric": ("full_pairwise_agreement" if metric == "three_pair"
                               else "anchor_agreement"),
        "diagnostic_metric": ("anchor_agreement" if metric == "three_pair"
                              else "full_pairwise_agreement"),
        "agreement_definition": (
            "mean(J(s0,s1), J(s0,s2), J(s1,s2)); empty masks agree"
            if metric == "three_pair"
            else "mean(J(s0,s1), J(s0,s2)); empty masks agree"),
        "pairwise_definitions": {
            "J01": "Jaccard(mask_seed0, mask_seed1)",
            "J02": "Jaccard(mask_seed0, mask_seed2)",
            "J12": "Jaccard(mask_seed1, mask_seed2)"},
        "supersedes_two_anchor_manifest":
            "configs/hybrid_obstacle_full_seed_joint_gate_v1.json",
        "pixel_mask_threshold": PIXEL_MASK_THRESHOLD,
        "mask_comparison": "strictly greater than, matching the identifiability audit",
        "magnitude_support_bound": (
            "changed_probability * current_closeness * sigmoid(magnitude_logits); "
            "0 <= parked <= current <= 1 by construction, unchanged"),
        "calibration_episodes": sorted(calibration_episodes),
        "bootstrap_seed": thr.BOOTSTRAP_SEED,
        "bootstrap_replicates": thr.BOOTSTRAP_REPLICATES,
        "calibration_metrics": {k: v for k, v in chosen["block"].items()
                                if k != "trajectories"},
        "safety_head_sha256": sha(args.safety_dir / "model.pt"),
        "sensor_order_sha256": stack["sensor_contract"]["sensor_order_hash"],
        "controller": {k: stack["residual_controller"][k] for k in
                       ("gain", "decay_per_second", "ema",
                        "max_deviation_rad_per_joint", "arm_only", "gripper_owner")},
        "bootstrap_ensemble_disposition":
            "invalid_for_deployment_uncertainty_due_to_cluster_omission_variance",
        "averaging_permitted": False,
        "uncertainty_seed_may_replace_seed0": False,
    }
    deployment["manifest_sha256"] = thr.canonical_hash(deployment)
    args.deployment_manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.deployment_manifest_out.write_text(
        json.dumps(deployment, indent=2, sort_keys=True) + "\n")
    print(f"deployment manifest frozen: {deployment['manifest_sha256']}")

    # ---- nested offline evaluation --------------------------------------------------
    nested = score(models, head, train_partition,
                   set(spec["splits"]["nested_offline_evaluation"]["episodes"]),
                   device, hard_threshold, metric=metric)
    nested_block = aggregate(nested, A, U)
    nested_fail = []
    if (nested_block["median_active_recall"] or 0) < NESTED_MIN_ACTIVE_RECALL:
        nested_fail.append(f"active recall {nested_block['median_active_recall']}")
    if nested_block["median_hard_retention"] is not None and \
            nested_block["median_hard_retention"] < MIN_MEDIAN_HARD_RETENTION:
        nested_fail.append(f"hard retention {nested_block['median_hard_retention']}")
    if (nested_block["median_retained_cosine"] or 0) < MIN_RETAINED_COSINE:
        nested_fail.append("cosine")
    if (nested_block["median_retained_positive_cosine"] or 0) < MIN_RETAINED_POSITIVE_COSINE:
        nested_fail.append("positive cosine")
    if (nested_block["mean_executed_false_activation"] or 0) > NESTED_MAX_MEAN_EXECUTED:
        nested_fail.append("mean executed false activation")
    if (nested_block["max_hazard_absent_executed_fraction"] or 0) > \
            NESTED_MAX_HAZARD_ABSENT_EXECUTED:
        nested_fail.append("hazard-absent executed fraction")
    if nested_block["max_false_active_run"] > MAX_FALSE_ACTIVE_RUN:
        nested_fail.append("false-active run")
    if nested_block["any_persistent_after_support"]:
        nested_fail.append("persistent correction")
    for key, floor in (("mean_agreement_acceptance_zero", MIN_ZERO_ACCEPTANCE),
                       ("mean_agreement_acceptance_active", MIN_ACTIVE_ACCEPTANCE),
                       ("mean_agreement_acceptance_inactive", MIN_INACTIVE_ACCEPTANCE)):
        if nested_block[key] is not None and nested_block[key] < floor:
            nested_fail.append(f"{key} {nested_block[key]:.4f}")
    if nested_block["max_trajectory_abstention"] > MAX_TRAJECTORY_ABSTENTION:
        nested_fail.append("trajectory abstention")
    if nested_block["max_hazard_present_active_abstention"] > \
            MAX_HAZARD_PRESENT_ACTIVE_ABSTENTION:
        nested_fail.append("hazard-present active abstention")

    # ---- historical regression + consumed diagnostics --------------------------------
    historical = json.loads(args.onset_audit.read_text())["known_false_positive_frames"]
    historical_keys = {(r["trajectory_id"], r["step"]) for r in historical}
    diagnostic = []
    for name in CONSUMED_PARTITIONS:
        partition = load_partition(args.cache, name)
        diagnostic.extend(score(models, head, partition, set(partition.episode_ids),
                                device, hard_threshold, metric=metric))
    diagnostic_block = aggregate(diagnostic, A, U)

    historical_rows = []
    for t in diagnostic:
        for step in range(t.frames):
            if (t.trajectory_id, step) not in historical_keys:
                continue
            gates = joint_decision(t.activity[step:step + 1], A,
                                   t.agreement[step:step + 1], U)
            historical_rows.append({
                "trajectory_id": t.trajectory_id, "step": step,
                "hazard_present": t.hazard_present,
                "seed0_activity": float(t.activity[step]),
                "activity_threshold": A,
                "activity_pass": bool(gates["activity_pass"][0]),
                "seed0_mask_pixels": int(t.masks[0, step]),
                "seed1_mask_pixels": int(t.masks[1, step]),
                "seed2_mask_pixels": int(t.masks[2, step]),
                "jaccard_seed0_seed1": float(t.pair_jaccard[0, step]),
                "jaccard_seed0_seed2": float(t.pair_jaccard[1, step]),
                "jaccard_seed1_seed2": float(t.pair_jaccard[2, step]),
                "controlling_agreement": float(t.agreement[step]),
                "alternate_agreement": float(t.three_pair[step]),
                "agreement_threshold": U,
                "decision_changed_by_j12": bool(
                    (t.agreement[step] >= U) != (t.three_pair[step] >= U)),
                "agreement_pass": bool(gates["agreement_pass"][0]),
                "executed": bool(gates["execute"][0]),
                "rejected_by_disagreement": bool(gates["abstained_by_uncertainty"][0]),
            })
    executed_historical = [r for r in historical_rows if r["executed"]]
    by_disagreement = [r for r in historical_rows if r["rejected_by_disagreement"]]

    diagnostic_fail = []
    if (diagnostic_block["mean_executed_false_activation"] or 0) > DIAG_MAX_MEAN_EXECUTED:
        diagnostic_fail.append("mean executed false activation")
    if (diagnostic_block["max_hazard_absent_executed_fraction"] or 0) > \
            DIAG_MAX_HAZARD_ABSENT_EXECUTED:
        diagnostic_fail.append("hazard-absent executed fraction")
    if (diagnostic_block["median_active_recall"] or 0) < DIAG_MIN_ACTIVE_RECALL:
        diagnostic_fail.append("active recall")
    if diagnostic_block["median_hard_retention"] is not None and \
            diagnostic_block["median_hard_retention"] < DIAG_MIN_HARD_RETENTION:
        diagnostic_fail.append("hard retention")
    if diagnostic_block["max_false_active_run"] > DIAG_MAX_RUN:
        diagnostic_fail.append("false-active run")
    if (diagnostic_block["mean_agreement_acceptance_zero"] or 0) < DIAG_MIN_ZERO_ACCEPTANCE:
        diagnostic_fail.append("zero acceptance")
    if diagnostic_block["max_trajectory_abstention"] > DIAG_MAX_ABSTENTION:
        diagnostic_fail.append("trajectory abstention")

    # ---- frozen inference repeatability ---------------------------------------------
    fixed = load_partition(args.cache, "offline_reference_test")
    batch = make_batch(fixed, np.arange(256), device)
    runs = []
    with torch.no_grad():
        for _ in range(INFERENCE_REPEATS):
            probs = [m(batch["history"], batch["history_valid"],
                       batch["state"])["changed_probability"].cpu().numpy()
                     for m in models]
            masks = [changed_mask(p) for p in probs]
            runs.append({
                "activity": probs[0].reshape(256, -1).max(axis=1),
                "j01": jaccard(masks[0], masks[1]),
                "j02": jaccard(masks[0], masks[2]),
                "j12": jaccard(masks[1], masks[2]),
                "agreement_vector": (three_pair_agreement(*masks)
                                     if metric == "three_pair"
                                     else anchor_mask_agreement(*masks)),
                "execute": joint_decision(
                    probs[0].reshape(256, -1).max(axis=1), A,
                    three_pair_agreement(*masks) if metric == "three_pair"
                    else anchor_mask_agreement(*masks), U)["execute"],
            })
    stability = {
        "repeats": INFERENCE_REPEATS,
        "activity_max_abs_delta": float(max(
            np.abs(r["activity"] - runs[0]["activity"]).max() for r in runs)),
        "agreement_max_abs_delta": float(max(
            np.abs(r["agreement_vector"] - runs[0]["agreement_vector"]).max()
            for r in runs)),
        "jaccard_identical": bool(all(
            np.array_equal(r["j01"], runs[0]["j01"])
            and np.array_equal(r["j02"], runs[0]["j02"])
            and np.array_equal(r["j12"], runs[0]["j12"]) for r in runs)),
        "decisions_identical": bool(all(np.array_equal(r["execute"], runs[0]["execute"])
                                        for r in runs)),
        "tolerance": INFERENCE_TOLERANCE, "training_kernels_invoked": False,
    }
    stability["stable"] = bool(
        stability["activity_max_abs_delta"] <= INFERENCE_TOLERANCE
        and stability["agreement_max_abs_delta"] <= INFERENCE_TOLERANCE
        and stability["jaccard_identical"] and stability["decisions_identical"])

    old_block = aggregate(calibration, OLD_ACTIVITY_THRESHOLD, 0.0)
    report = {
        "schema": "hybrid_obstacle_three_pair_calibration_v1", "mode": MODE,
        "controlling_metric": ("full_pairwise_agreement" if metric == "three_pair"
                               else "anchor_agreement"),
        "feasible": True,
        "grid": {"activity": int(activity_values.size),
                 "agreement": int(agreement_values.size),
                 "cartesian_pairs": int(activity_values.size * agreement_values.size),
                 "pairs_fully_evaluated": evaluated,
                 "axis_screens_applied": ["activity_alone_retention",
                                          "agreement acceptance floors"]},
        "feasible_count": len(feasible),
        "selected": {"activity_threshold": A, "agreement_threshold": U,
                     "bootstrap_upper_false_activation": chosen["upper"],
                     "checks": chosen["checks"]},
        "selection_rule": ("lexicographic: lowest bootstrap upper false-activation, "
                           "highest median active recall, highest hard retention, "
                           "highest ordinary-zero acceptance, highest activity "
                           "threshold, highest agreement threshold"),
        "contract": {
            "min_median_active_recall": MIN_MEDIAN_ACTIVE_RECALL,
            "recall_floor_lowered": False,
            "min_activity_alone_retention": MIN_ACTIVITY_ALONE_ACTIVE_RETENTION,
            "min_zero_acceptance": MIN_ZERO_ACCEPTANCE,
            "min_active_acceptance": MIN_ACTIVE_ACCEPTANCE,
            "min_inactive_acceptance": MIN_INACTIVE_ACCEPTANCE,
            "max_trajectory_abstention": MAX_TRAJECTORY_ABSTENTION,
            "max_hazard_present_active_abstention":
                MAX_HAZARD_PRESENT_ACTIVE_ABSTENTION,
            "max_bootstrap_upper_false_activation":
                MAX_BOOTSTRAP_UPPER_FALSE_ACTIVATION,
        },
        "calibration": {k: v for k, v in chosen["block"].items() if k != "trajectories"},
        "calibration_trajectories": chosen["block"]["trajectories"],
        "old_threshold_comparison": {
            "old_activity_threshold": OLD_ACTIVITY_THRESHOLD,
            "new_activity_threshold": A,
            "old_median_active_recall_no_uncertainty":
                old_block["median_active_recall"],
            "new_activity_alone_retention":
                chosen["block"]["median_activity_alone_retention"],
            "new_final_active_recall": chosen["block"]["median_active_recall"],
            "old_mean_executed_false_activation":
                old_block["mean_executed_false_activation"],
            "new_mean_executed_false_activation":
                chosen["block"]["mean_executed_false_activation"],
            "quiet_frame_agreement_acceptance":
                chosen["block"]["mean_agreement_acceptance_zero"],
        },
        "deployment_manifest_sha256": deployment["manifest_sha256"],
        "nested_offline": {k: v for k, v in nested_block.items() if k != "trajectories"},
        "nested_trajectories": nested_block["trajectories"],
        "nested_failures": nested_fail, "nested_passed": not nested_fail,
        "historical_regression": {
            "frames": historical_rows, "count": len(historical_rows),
            "executed": len(executed_historical),
            "rejected": len(historical_rows) - len(executed_historical),
            "rejected_by_disagreement": len(by_disagreement),
            "passes": bool(len(historical_rows) - len(executed_historical) >= 16
                           and len(by_disagreement) >= 16
                           and not executed_historical),
            "note": "regression evidence only; never used for fitting",
        },
        "reused_diagnostic": {
            "label": "reused_nonconfirmatory_diagnostic",
            "metrics": {k: v for k, v in diagnostic_block.items()
                        if k != "trajectories"},
            "failures": diagnostic_fail, "passed": not diagnostic_fail,
        },
        "inference_stability": stability,
        "wall_seconds": time.time() - started,
    }
    # A failure that is *only* clustering or persistence gets its own token: every
    # coverage and rate gate passed, and the handoff forbids adding temporal logic here.
    temporal_only = {"false-active run", "persistent correction"}
    nested_non_temporal = [f for f in nested_fail
                           if not any(t in f for t in temporal_only)]
    diagnostic_non_temporal = [f for f in diagnostic_fail
                               if not any(t in f for t in temporal_only)]
    report["temporal_classification"] = {
        "nested_failures": nested_fail,
        "nested_non_temporal_failures": nested_non_temporal,
        "diagnostic_failures": diagnostic_fail,
        "diagnostic_non_temporal_failures": diagnostic_non_temporal,
        "only_temporal_failures": bool(
            (nested_fail or diagnostic_fail)
            and not nested_non_temporal and not diagnostic_non_temporal),
        "temporal_gate_names": sorted(temporal_only),
        "temporal_logic_added": False,
    }
    report["report_sha256"] = thr.canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"nested: recall={nested_block['median_active_recall']} "
          f"exec_false={nested_block['mean_executed_false_activation']} "
          f"passed={not nested_fail} {nested_fail}")
    print(f"historical: {report['historical_regression']['rejected']}/17 rejected, "
          f"{len(by_disagreement)} by disagreement, "
          f"{len(executed_historical)} executed")
    print(f"diagnostic passed: {not diagnostic_fail} {diagnostic_fail}")
    print(f"inference stable: {stability['stable']}")
    print(f"wrote {args.out}")
    return 0 if (not nested_fail and not diagnostic_fail
                 and report["historical_regression"]["passes"]
                 and stability["stable"]) else 9


if __name__ == "__main__":
    raise SystemExit(main())
