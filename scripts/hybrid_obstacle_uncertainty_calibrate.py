#!/usr/bin/env python3
"""Calibrate the agreement threshold, then qualify the combined contract offline.

Handoff steps 6-7 and 9-14, executed in that order in one process so the ordering is
structural: calibration sees only the eight uncertainty-calibration clusters, the combined
manifest is written and hashed, and only then are the nested evaluation clusters, the 17
historical frames and the consumed diagnostic set opened.

The anti-degeneracy floors are the point of this calibration. A metric that marks everything
uncertain would trivially reject the historical failures, so the contract also has to accept
80% of ordinary quiet frames, 80% of active frames, and 80% of frames the activity gate has
already declined -- and no trajectory may abstain on more than half its frames.
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
from causal_parked_skin.abstention import (
    PIXEL_MASK_THRESHOLD,
    anchor_mask_agreement,
    changed_mask,
    combined_decision,
    mean_pairwise_agreement,
)
from causal_parked_skin.data import SOURCE_MODES, load_partition
from causal_parked_skin.engine import load_checkpoint, make_batch, set_sensor_names
from causal_parked_skin.model import BASELINE_CURRENT, FrozenSafetyHead

CONSUMED_PARTITIONS = ("offline_reference_test", "reference_calibration",
                       "reference_validation")

# calibration contract (handoff step 9)
MIN_MEDIAN_ACTIVE_RECALL = 0.80
MIN_MEDIAN_HARD_RETENTION = 0.80
MAX_BOOTSTRAP_UPPER_FINAL_ACTIVATION = 0.02
MAX_HAZARD_ABSENT_FINAL_RATE = 0.05
MAX_FINAL_FALSE_ACTIVE_RUN = 2
MIN_RETAINED_COSINE = 0.75
MIN_RETAINED_POSITIVE_COSINE = 0.85
# anti-degeneracy floors
MIN_ZERO_ACCEPTANCE = 0.80
MIN_ACTIVE_ACCEPTANCE = 0.80
MIN_INACTIVE_ACCEPTANCE = 0.80
MAX_TRAJECTORY_ABSTENTION = 0.50

# nested offline gates (handoff step 11)
NESTED_MIN_ACTIVE_RECALL = 0.75
NESTED_MIN_HARD_RETENTION = 0.80
NESTED_MAX_MEAN_FINAL = 0.02
NESTED_MAX_HAZARD_ABSENT_FINAL = 0.05

# reused diagnostic gates (handoff step 13)
DIAG_MAX_MEAN_FINAL = 0.03
DIAG_MAX_HAZARD_ABSENT_FINAL = 0.10
DIAG_MIN_ACTIVE_RECALL = 0.70
DIAG_MIN_HARD_RETENTION = 0.80
DIAG_MAX_RUN = 5
DIAG_MIN_ZERO_ACCEPTANCE = 0.75

INFERENCE_REPEATS = 24
INFERENCE_TOLERANCE = 1e-7


class Scored:
    """Per-frame quantities for one trajectory under the combined contract."""

    def __init__(self, meta, activity, agreement, cosine, oracle_active, hard_active,
                 pairwise, seed0_mask_count, member_mask_counts, member_jaccard) -> None:
        self.__dict__.update(meta)
        self.activity = activity
        self.agreement = agreement
        self.cosine = cosine
        self.oracle_active = oracle_active.astype(bool)
        self.hard_active = hard_active.astype(bool)
        self.pairwise = pairwise
        self.seed0_mask_count = seed0_mask_count
        self.member_mask_counts = member_mask_counts
        self.member_jaccard = member_jaccard
        self.frames = len(activity)

    def metrics_at(self, activity_threshold: float, agreement_threshold: float) -> dict:
        gates = combined_decision(self.activity, activity_threshold,
                                  self.agreement, agreement_threshold)
        execute = gates["execute"]
        active = self.oracle_active
        zero = ~active
        agreement_pass = gates["agreement_pass"]
        activity_pass = gates["activity_pass"]
        retained = execute & active
        false_active = execute & zero
        cosines = self.cosine[retained]
        return {
            "trajectory_id": self.trajectory_id,
            "episode_id": self.episode_id,
            "distribution": self.distribution,
            "hazard_present": bool(self.hazard_present),
            "frames": self.frames,
            "oracle_active_frames": int(active.sum()),
            "oracle_zero_frames": int(zero.sum()),
            "active_recall": float(retained.sum() / active.sum()) if active.any() else None,
            "hard_true_active_retention": float(
                execute[self.hard_active].mean()) if self.hard_active.any() else None,
            "final_activation_rate_on_zero": (float(false_active.sum() / zero.sum())
                                              if zero.any() else None),
            "final_active_fraction": float(execute.mean()),
            "max_final_false_active_run": thr.max_true_run(false_active),
            "median_retained_cosine": float(np.median(cosines)) if cosines.size else None,
            "retained_positive_cosine_fraction": (float((cosines > 0).mean())
                                                  if cosines.size else None),
            # anti-degeneracy: what the agreement gate accepts, independent of activity
            "agreement_acceptance_zero": float(agreement_pass[zero].mean())
            if zero.any() else None,
            "agreement_acceptance_active": float(agreement_pass[active].mean())
            if active.any() else None,
            "agreement_acceptance_activity_inactive": float(
                agreement_pass[~activity_pass].mean()) if (~activity_pass).any() else None,
            "uncertainty_abstention_fraction": float((~agreement_pass).mean()),
            "abstained_from_active_candidates": int(gates["abstained"].sum()),
            "activity_gate_passes": int(activity_pass.sum()),
        }


def aggregate(trajectories, activity_threshold, agreement_threshold) -> dict:
    rows = [t.metrics_at(activity_threshold, agreement_threshold) for t in trajectories]

    def gather(key, predicate=lambda r: True):
        return [r[key] for r in rows if predicate(r) and r[key] is not None]

    recalls = gather("active_recall")
    finals = gather("final_activation_rate_on_zero")
    hard = gather("hard_true_active_retention")
    cosines = gather("median_retained_cosine")
    positives = gather("retained_positive_cosine_fraction")
    hazard_absent = gather("final_activation_rate_on_zero", lambda r: not r["hazard_present"])
    return {
        "agreement_threshold": float(agreement_threshold),
        "trajectories": rows,
        "median_active_recall": float(np.median(recalls)) if recalls else None,
        "median_hard_true_active_retention": float(np.median(hard)) if hard else None,
        "mean_final_activation_on_zero": float(np.mean(finals)) if finals else None,
        "max_hazard_absent_final_rate": float(np.max(hazard_absent))
        if hazard_absent else None,
        "max_final_false_active_run": max(r["max_final_false_active_run"] for r in rows),
        "median_retained_cosine": float(np.median(cosines)) if cosines else None,
        "median_retained_positive_cosine_fraction": float(np.median(positives))
        if positives else None,
        "min_agreement_acceptance_zero": min(
            gather("agreement_acceptance_zero"), default=None),
        "mean_agreement_acceptance_zero": float(np.mean(
            gather("agreement_acceptance_zero"))) if gather(
                "agreement_acceptance_zero") else None,
        "mean_agreement_acceptance_active": float(np.mean(
            gather("agreement_acceptance_active"))) if gather(
                "agreement_acceptance_active") else None,
        "mean_agreement_acceptance_activity_inactive": float(np.mean(
            gather("agreement_acceptance_activity_inactive"))) if gather(
                "agreement_acceptance_activity_inactive") else None,
        "max_trajectory_abstention_fraction": max(
            r["uncertainty_abstention_fraction"] for r in rows),
        "max_hazard_absent_final_active_fraction": max(
            [r["final_active_fraction"] for r in rows if not r["hazard_present"]],
            default=None),
        "by_distribution": {
            name: {"trajectories": sum(1 for r in rows if r["distribution"] == name)}
            for name in SOURCE_MODES},
    }


def score_split(seed0, members, head, partition, episodes, device, hard_threshold,
                *, batch=256):
    """Run seed 0 and all five members over one split."""
    import torch

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
        agreement = np.zeros(n)
        pairwise = np.zeros(n)
        cosine = np.zeros(n)
        seed0_counts = np.zeros(n, dtype=np.int64)
        member_counts = np.zeros((len(members), n), dtype=np.int64)
        member_jaccard = np.zeros((len(members), n))
        seed0_dq = np.zeros((n, 7))

        for start in range(0, n, batch):
            chunk = rows[start:start + batch]
            slot = slice(start, start + len(chunk))
            b = make_batch(partition, chunk, device)
            with torch.no_grad():
                anchor_out = seed0(b["history"], b["history_valid"], b["state"])
                anchor_probability = anchor_out["changed_probability"].cpu().numpy()
                dq = (head(anchor_out["current"]) - head(anchor_out["parked"])
                      ).double().cpu().numpy()
                member_probability = [
                    m(b["history"], b["history_valid"], b["state"]
                      )["changed_probability"].cpu().numpy() for m in members]

            anchor_mask = changed_mask(anchor_probability)
            member_masks = [changed_mask(p) for p in member_probability]
            activity[slot] = anchor_probability.reshape(
                len(chunk), -1).max(axis=1)
            agreement[slot] = anchor_mask_agreement(anchor_mask, member_masks)
            pairwise[slot] = mean_pairwise_agreement(member_masks)
            seed0_counts[slot] = anchor_mask.sum(axis=1)
            from causal_parked_skin.abstention import jaccard

            for m_index, mask in enumerate(member_masks):
                member_counts[m_index, slot] = mask.sum(axis=1)
                member_jaccard[m_index, slot] = jaccard(anchor_mask, mask)
            seed0_dq[slot] = dq
            true_dq = oracle_dq[chunk]
            denominator = (np.linalg.norm(dq, axis=-1)
                           * np.linalg.norm(true_dq, axis=-1) + thr.COSINE_EPSILON)
            cosine[slot] = (dq * true_dq).sum(-1) / denominator

        hard = oracle_active[rows] & (activity < hard_threshold)
        out.append(Scored(
            {"trajectory_id": partition.trajectory_ids[index],
             "episode_id": partition.episode_ids[index],
             "distribution": SOURCE_MODES[int(modes[rows[0]])],
             "hazard_present": bool(hazard[rows[0]]),
             "partition": partition.name, "rows": rows},
            activity, agreement, cosine, oracle_active[rows], hard, pairwise,
            seed0_counts, member_counts, member_jaccard))
    return out


def feasible_at(block, trajectories, activity_threshold, agreement_threshold,
                cluster_of_row, clusters) -> tuple[bool, dict, float]:
    finals = np.array([[r["final_activation_rate_on_zero"] or 0.0]
                       for r in block["trajectories"]])
    upper = float(thr.cluster_bootstrap_upper_bound(
        thr.cluster_means(finals, cluster_of_row, clusters))[0])
    checks = {
        "median_active_recall": (block["median_active_recall"] or 0) >= MIN_MEDIAN_ACTIVE_RECALL,
        "median_hard_retention": (block["median_hard_true_active_retention"] is None
                                  or block["median_hard_true_active_retention"]
                                  >= MIN_MEDIAN_HARD_RETENTION),
        "bootstrap_upper_final_activation": upper <= MAX_BOOTSTRAP_UPPER_FINAL_ACTIVATION,
        "hazard_absent_final_rate": (block["max_hazard_absent_final_rate"] is None
                                     or block["max_hazard_absent_final_rate"]
                                     <= MAX_HAZARD_ABSENT_FINAL_RATE),
        "final_false_active_run": block["max_final_false_active_run"]
        <= MAX_FINAL_FALSE_ACTIVE_RUN,
        "retained_cosine": (block["median_retained_cosine"] or 0) >= MIN_RETAINED_COSINE,
        "retained_positive_cosine": (block["median_retained_positive_cosine_fraction"] or 0)
        >= MIN_RETAINED_POSITIVE_COSINE,
        "zero_acceptance_floor": (block["mean_agreement_acceptance_zero"] or 0)
        >= MIN_ZERO_ACCEPTANCE,
        "active_acceptance_floor": (block["mean_agreement_acceptance_active"] or 0)
        >= MIN_ACTIVE_ACCEPTANCE,
        "inactive_acceptance_floor": (
            block["mean_agreement_acceptance_activity_inactive"] is None
            or block["mean_agreement_acceptance_activity_inactive"]
            >= MIN_INACTIVE_ACCEPTANCE),
        "trajectory_abstention_cap": block["max_trajectory_abstention_fraction"]
        <= MAX_TRAJECTORY_ABSTENTION,
    }
    return all(checks.values()), checks, upper


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--ensemble-manifest", required=True, type=Path)
    ap.add_argument("--seed0-checkpoint", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--activity-threshold", type=float, required=True)
    ap.add_argument("--groups", required=True, type=Path)
    ap.add_argument("--onset-audit", required=True, type=Path)
    ap.add_argument("--deployment-manifest-out", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch

    stack = json.loads(args.stack.read_text())
    set_sensor_names(stack["sensor_contract"]["ordered_names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = FrozenSafetyHead.load(args.safety_dir, device=device)
    ensemble = json.loads(args.ensemble_manifest.read_text())
    spec = json.loads(args.partition.read_text())
    groups = json.loads(args.groups.read_text())
    hard_threshold = groups["historical_activity_range"][1]

    seed0, seed0_payload = load_checkpoint(args.seed0_checkpoint, device)
    if seed0_payload["config"]["seed"] != 0:
        raise SystemExit("deployment predictor must be seed 0")
    members = []
    for record in ensemble["member_records"]:
        model, payload = load_checkpoint(Path(record["checkpoint"]), device)
        if payload["config"]["seed"] != record["bootstrap_seed"] or \
                payload["config"]["variant"] != BASELINE_CURRENT:
            raise SystemExit(f"member {record['index']} strict load mismatch")
        members.append(model)
    if len(members) != 5:
        raise SystemExit(f"expected 5 members, loaded {len(members)}")
    print(f"loaded seed-0 anchor and {len(members)} bootstrap members")

    train_partition = load_partition(args.cache, "reference_train")
    started = time.time()

    # ---- 1. calibration on the eight calibration clusters only --------------------
    calibration_episodes = set(spec["splits"]["threshold_calibration"]["episodes"])
    calibration = score_split(seed0, members, head, train_partition,
                              calibration_episodes, device, hard_threshold)
    episodes = sorted({t.episode_id for t in calibration})
    cluster_of = {e: i for i, e in enumerate(episodes)}
    cluster_of_row = np.array([cluster_of[t.episode_id] for t in calibration])

    candidates = np.unique(np.concatenate([t.agreement for t in calibration]))
    print(f"candidate agreement thresholds: {candidates.size}")
    feasible = []
    for candidate in candidates:
        block = aggregate(calibration, args.activity_threshold, candidate)
        ok, checks, upper = feasible_at(block, calibration, args.activity_threshold,
                                        candidate, cluster_of_row, len(episodes))
        if ok:
            feasible.append({"threshold": float(candidate), "upper": upper,
                             "block": block, "checks": checks})
    print(f"feasible thresholds: {len(feasible)}")

    if not feasible:
        # record the frontier so the failure is diagnosable
        sweep = []
        for candidate in np.quantile(candidates, np.linspace(0, 1, 21)):
            block = aggregate(calibration, args.activity_threshold, candidate)
            _, checks, upper = feasible_at(block, calibration, args.activity_threshold,
                                           candidate, cluster_of_row, len(episodes))
            sweep.append({"threshold": float(candidate), "bootstrap_upper": upper,
                          "checks": checks,
                          "median_active_recall": block["median_active_recall"],
                          "mean_final_activation": block["mean_final_activation_on_zero"],
                          "zero_acceptance": block["mean_agreement_acceptance_zero"],
                          "max_abstention": block["max_trajectory_abstention_fraction"]})
        report = {"schema": "hybrid_obstacle_uncertainty_calibration_v1",
                  "feasible": False, "candidate_count": int(candidates.size),
                  "sweep": sweep,
                  "decision_if_infeasible":
                      "UNCERTAINTY_ABSTENTION_CALIBRATION_INFEASIBLE"}
        report["report_sha256"] = thr.canonical_hash(report)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True,
                                       default=str) + "\n")
        print("NO FEASIBLE AGREEMENT THRESHOLD")
        for row in sweep:
            print(f"  tau={row['threshold']:.4f} recall={row['median_active_recall']} "
                  f"final={row['mean_final_activation']} "
                  f"zero_accept={row['zero_acceptance']} "
                  f"failed={[k for k, v in row['checks'].items() if not v]}")
        return 7

    chosen = min(feasible, key=lambda f: (
        f["upper"], -(f["block"]["median_active_recall"] or 0),
        -(f["block"]["mean_agreement_acceptance_zero"] or 0), -f["threshold"]))
    agreement_threshold = chosen["threshold"]
    print(f"selected agreement threshold {agreement_threshold:.6f} "
          f"(bootstrap upper final activation {chosen['upper']:.5f})")

    # ---- 2. freeze the combined deployment manifest BEFORE opening anything else ---
    deployment = {
        "schema": "hybrid_obstacle_uncertainty_deployment_manifest_v1",
        "condition": "ACT_PLUS_UNCERTAINTY_ABSTENTION",
        "seed0_checkpoint": str(args.seed0_checkpoint),
        "seed0_checkpoint_sha256": ensemble["seed0_deployment_sha256"],
        "ensemble_manifest_sha256": ensemble["manifest_sha256"],
        "member_checkpoint_sha256": [r["checkpoint_sha256"]
                                     for r in ensemble["member_records"]],
        "member_checkpoints": [r["checkpoint"] for r in ensemble["member_records"]],
        "activity_threshold": args.activity_threshold,
        "activity_threshold_source": "frozen by the previous threshold manifest; not refit",
        "anchor_agreement_definition": (
            "mean Jaccard between the frozen seed-0 changed-pixel mask and each of the "
            "five bootstrap-member masks; two empty masks agree completely"),
        "agreement_threshold": agreement_threshold,
        "pixel_mask_threshold": PIXEL_MASK_THRESHOLD,
        "magnitude_support_bound": (
            "changed_probability * current_closeness * sigmoid(magnitude_logits); "
            "0 <= predicted_parked <= current_closeness <= 1 by construction, unchanged"),
        "bootstrap_seed": thr.BOOTSTRAP_SEED,
        "bootstrap_replicates": thr.BOOTSTRAP_REPLICATES,
        "calibration_episodes": sorted(calibration_episodes),
        "calibration_manifest_sha256": spec["manifest_sha256"],
        "safety_head_sha256": ensemble["safety_head_sha256"],
        "sensor_order_sha256": ensemble["sensor_order_sha256"],
        "controller": {k: stack["residual_controller"][k] for k in
                       ("gain", "decay_per_second", "ema",
                        "max_deviation_rad_per_joint", "arm_only", "gripper_owner")},
        "uncertainty_may_only": "turn execute into abstain",
        "averaging_permitted": False,
        "member_may_replace_seed0": False,
    }
    deployment["manifest_sha256"] = thr.canonical_hash(deployment)
    args.deployment_manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.deployment_manifest_out.write_text(
        json.dumps(deployment, indent=2, sort_keys=True) + "\n")
    print(f"deployment manifest frozen: {deployment['manifest_sha256']}")

    # ---- 3. nested offline evaluation ---------------------------------------------
    nested_episodes = set(spec["splits"]["nested_offline_evaluation"]["episodes"])
    nested = score_split(seed0, members, head, train_partition, nested_episodes,
                         device, hard_threshold)
    nested_block = aggregate(nested, args.activity_threshold, agreement_threshold)
    nested_failures = []
    if (nested_block["median_active_recall"] or 0) < NESTED_MIN_ACTIVE_RECALL:
        nested_failures.append(f"active recall {nested_block['median_active_recall']}")
    if nested_block["median_hard_true_active_retention"] is not None and \
            nested_block["median_hard_true_active_retention"] < NESTED_MIN_HARD_RETENTION:
        nested_failures.append(
            f"hard retention {nested_block['median_hard_true_active_retention']}")
    if (nested_block["mean_final_activation_on_zero"] or 0) > NESTED_MAX_MEAN_FINAL:
        nested_failures.append(
            f"mean final activation {nested_block['mean_final_activation_on_zero']}")
    if (nested_block["max_hazard_absent_final_active_fraction"] or 0) > \
            NESTED_MAX_HAZARD_ABSENT_FINAL:
        nested_failures.append("hazard-absent final active fraction "
                               f"{nested_block['max_hazard_absent_final_active_fraction']}")
    if nested_block["max_final_false_active_run"] > MAX_FINAL_FALSE_ACTIVE_RUN:
        nested_failures.append(f"run {nested_block['max_final_false_active_run']}")
    if (nested_block["median_retained_cosine"] or 0) < MIN_RETAINED_COSINE:
        nested_failures.append(f"cosine {nested_block['median_retained_cosine']}")
    if (nested_block["median_retained_positive_cosine_fraction"] or 0) < \
            MIN_RETAINED_POSITIVE_COSINE:
        nested_failures.append("positive cosine")
    if (nested_block["mean_agreement_acceptance_zero"] or 0) < MIN_ZERO_ACCEPTANCE:
        nested_failures.append(
            f"zero acceptance {nested_block['mean_agreement_acceptance_zero']}")
    if (nested_block["mean_agreement_acceptance_active"] or 0) < MIN_ACTIVE_ACCEPTANCE:
        nested_failures.append(
            f"active acceptance {nested_block['mean_agreement_acceptance_active']}")
    if nested_block["max_trajectory_abstention_fraction"] > MAX_TRAJECTORY_ABSTENTION:
        nested_failures.append("trajectory abstention cap")

    # ---- 4. historical 17-frame regression + consumed diagnostics ------------------
    historical = json.loads(args.onset_audit.read_text())["known_false_positive_frames"]
    historical_keys = {(r["trajectory_id"], r["step"]) for r in historical}
    diagnostic = []
    for name in CONSUMED_PARTITIONS:
        partition = load_partition(args.cache, name)
        diagnostic.extend(score_split(seed0, members, head, partition,
                                      set(partition.episode_ids), device,
                                      hard_threshold))
    diagnostic_block = aggregate(diagnostic, args.activity_threshold, agreement_threshold)

    historical_rows = []
    for trajectory in diagnostic:
        for step in range(trajectory.frames):
            if (trajectory.trajectory_id, step) not in historical_keys:
                continue
            gates = combined_decision(
                trajectory.activity[step:step + 1], args.activity_threshold,
                trajectory.agreement[step:step + 1], agreement_threshold)
            historical_rows.append({
                "trajectory_id": trajectory.trajectory_id, "step": step,
                "hazard_present": trajectory.hazard_present,
                "seed0_activity": float(trajectory.activity[step]),
                "activity_gate_pass": bool(gates["activity_pass"][0]),
                "anchor_mask_agreement": float(trajectory.agreement[step]),
                "agreement_gate_pass": bool(gates["agreement_pass"][0]),
                "seed0_changed_pixels": int(trajectory.seed0_mask_count[step]),
                "member_changed_pixels": [int(c) for c in
                                          trajectory.member_mask_counts[:, step]],
                "member_to_anchor_jaccard": [float(j) for j in
                                             trajectory.member_jaccard[:, step]],
                "executed": bool(gates["execute"][0]),
                "rejected_by_uncertainty": bool(gates["abstained"][0]),
            })
    rejected = [r for r in historical_rows if not r["executed"]]
    by_uncertainty = [r for r in historical_rows if r["rejected_by_uncertainty"]]

    diagnostic_failures = []
    if (diagnostic_block["mean_final_activation_on_zero"] or 0) > DIAG_MAX_MEAN_FINAL:
        diagnostic_failures.append("mean final activation")
    if (diagnostic_block["max_hazard_absent_final_active_fraction"] or 0) > \
            DIAG_MAX_HAZARD_ABSENT_FINAL:
        diagnostic_failures.append("hazard-absent final active fraction")
    if (diagnostic_block["median_active_recall"] or 0) < DIAG_MIN_ACTIVE_RECALL:
        diagnostic_failures.append("active recall")
    if diagnostic_block["median_hard_true_active_retention"] is not None and \
            diagnostic_block["median_hard_true_active_retention"] < DIAG_MIN_HARD_RETENTION:
        diagnostic_failures.append("hard retention")
    if diagnostic_block["max_final_false_active_run"] > DIAG_MAX_RUN:
        diagnostic_failures.append("false-active run")
    if (diagnostic_block["mean_agreement_acceptance_zero"] or 0) < DIAG_MIN_ZERO_ACCEPTANCE:
        diagnostic_failures.append("zero acceptance")

    # ---- 5. frozen inference determinism -------------------------------------------
    fixed = load_partition(args.cache, "offline_reference_test")
    index = np.arange(256)
    batch = make_batch(fixed, index, device)
    runs = []
    with torch.no_grad():
        for _ in range(INFERENCE_REPEATS):
            anchor = seed0(batch["history"], batch["history_valid"], batch["state"])
            anchor_probability = anchor["changed_probability"].cpu().numpy()
            masks = [changed_mask(m(batch["history"], batch["history_valid"],
                                    batch["state"])["changed_probability"].cpu().numpy())
                     for m in members]
            agreement = anchor_mask_agreement(changed_mask(anchor_probability), masks)
            activity = anchor_probability.reshape(len(index), -1).max(axis=1)
            gates = combined_decision(activity, args.activity_threshold,
                                      agreement, agreement_threshold)
            runs.append({"activity": activity, "agreement": agreement,
                         "parked": anchor["parked"].double().cpu().numpy(),
                         "execute": gates["execute"]})
    stability = {
        "repeats": INFERENCE_REPEATS,
        "activity_max_abs_delta": float(max(
            np.abs(r["activity"] - runs[0]["activity"]).max() for r in runs)),
        "agreement_max_abs_delta": float(max(
            np.abs(r["agreement"] - runs[0]["agreement"]).max() for r in runs)),
        "parked_field_max_abs_delta": float(max(
            np.abs(r["parked"] - runs[0]["parked"]).max() for r in runs)),
        "decisions_identical": bool(all(np.array_equal(r["execute"], runs[0]["execute"])
                                        for r in runs)),
        "tolerance": INFERENCE_TOLERANCE,
        "training_kernels_invoked": False,
    }
    stability["stable"] = bool(
        stability["activity_max_abs_delta"] <= INFERENCE_TOLERANCE
        and stability["agreement_max_abs_delta"] <= INFERENCE_TOLERANCE
        and stability["parked_field_max_abs_delta"] <= INFERENCE_TOLERANCE
        and stability["decisions_identical"])

    report = {
        "schema": "hybrid_obstacle_uncertainty_calibration_v1",
        "feasible": True,
        "activity_threshold": args.activity_threshold,
        "agreement_threshold": agreement_threshold,
        "pixel_mask_threshold": PIXEL_MASK_THRESHOLD,
        "candidate_count": int(candidates.size),
        "feasible_count": len(feasible),
        "selection_rule": ("lexicographic: lowest bootstrap upper final-activation bound, "
                           "highest active recall, highest ordinary-zero acceptance, "
                           "highest agreement threshold"),
        "calibration": {k: v for k, v in chosen["block"].items() if k != "trajectories"},
        "calibration_trajectories": chosen["block"]["trajectories"],
        "calibration_checks": chosen["checks"],
        "calibration_bootstrap_upper": chosen["upper"],
        "contract": {
            "min_median_active_recall": MIN_MEDIAN_ACTIVE_RECALL,
            "min_median_hard_retention": MIN_MEDIAN_HARD_RETENTION,
            "max_bootstrap_upper_final_activation": MAX_BOOTSTRAP_UPPER_FINAL_ACTIVATION,
            "max_hazard_absent_final_rate": MAX_HAZARD_ABSENT_FINAL_RATE,
            "max_final_false_active_run": MAX_FINAL_FALSE_ACTIVE_RUN,
            "min_retained_cosine": MIN_RETAINED_COSINE,
            "min_retained_positive_cosine": MIN_RETAINED_POSITIVE_COSINE,
            "min_zero_acceptance": MIN_ZERO_ACCEPTANCE,
            "min_active_acceptance": MIN_ACTIVE_ACCEPTANCE,
            "min_inactive_acceptance": MIN_INACTIVE_ACCEPTANCE,
            "max_trajectory_abstention": MAX_TRAJECTORY_ABSTENTION,
        },
        "deployment_manifest_sha256": deployment["manifest_sha256"],
        "nested_offline": {k: v for k, v in nested_block.items() if k != "trajectories"},
        "nested_offline_trajectories": nested_block["trajectories"],
        "nested_failures": nested_failures,
        "nested_passed": not nested_failures,
        "historical_regression": {
            "frames": historical_rows,
            "count": len(historical_rows),
            "rejected": len(rejected),
            "rejected_by_uncertainty": len(by_uncertainty),
            "executed": len(historical_rows) - len(rejected),
            "passes": len(rejected) >= 16 and len(historical_rows) - len(rejected) == 0,
            "note": "regression test, not an independent statistical claim",
            "fitted_on_these_frames": False,
        },
        "reused_diagnostic": {
            "label": "reused_nonconfirmatory_diagnostic",
            "partitions": list(CONSUMED_PARTITIONS),
            "metrics": {k: v for k, v in diagnostic_block.items()
                        if k != "trajectories"},
            "failures": diagnostic_failures,
            "passed": not diagnostic_failures,
        },
        "inference_stability": stability,
        "wall_seconds": time.time() - started,
    }
    report["report_sha256"] = thr.canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"\nnested offline: recall={nested_block['median_active_recall']} "
          f"final={nested_block['mean_final_activation_on_zero']} "
          f"zero_accept={nested_block['mean_agreement_acceptance_zero']} "
          f"passed={not nested_failures} {nested_failures}")
    print(f"historical regression: {len(rejected)}/17 rejected "
          f"({len(by_uncertainty)} by uncertainty), executed "
          f"{len(historical_rows) - len(rejected)}")
    print(f"reused diagnostic passed: {not diagnostic_failures} {diagnostic_failures}")
    print(f"inference stable: {stability['stable']}")
    print(f"wrote {args.out}")
    return 0 if (not nested_failures and not diagnostic_failures
                 and report["historical_regression"]["passes"]
                 and stability["stable"]) else 9


if __name__ == "__main__":
    raise SystemExit(main())
