#!/usr/bin/env python3
"""Calibrate, evaluate and audit PROX_EVIDENCE_ACTIVITY_GATE_V1.

Handoff steps 11-14, executed in that order inside one process so the ordering is
structural rather than procedural:

1. calibrate the threshold on the eight threshold-calibration episodes only;
2. freeze it;
3. open the eight nested-offline-evaluation episodes exactly once;
4. run the offline controls at each method's own operating point;
5. only then re-score the previously consumed diagnostic trajectories.

The nested evaluation split is not loaded until the threshold is frozen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from causal_parked_skin import engine
from causal_parked_skin import threshold as thr
from causal_parked_skin.activity_gate import build_gate
from causal_parked_skin.data import SOURCE_MODES, load_partition
from causal_parked_skin.engine import load_checkpoint, make_batch
from causal_parked_skin.model import FrozenSafetyHead

PREVIOUS_THRESHOLD = 0.99960857629776
CONSUMED_PARTITIONS = ("offline_reference_test", "reference_calibration",
                       "reference_validation")
ONSET_MIN_FRAMES = 10
ONSET_FRACTION = 0.10

# nested offline gates (handoff step 13)
NESTED_MIN_RECALL = 0.75
NESTED_MAX_MEAN_FPR = 0.02
NESTED_MAX_HAZARD_ABSENT_ACTIVE = 0.05
NESTED_MAX_RUN = 2
NESTED_MIN_COSINE = 0.75
NESTED_MIN_POSITIVE_COSINE = 0.85

# reused-diagnostic blocking checks (handoff step 14)
DIAG_MAX_FAILING_EPISODE_RUN = 2
DIAG_MAX_HAZARD_ABSENT_ACTIVE = 0.10
DIAG_MAX_MEAN_FPR = 0.03
DIAG_MIN_MEDIAN_RECALL = 0.70


def onset_cutoff(length: int) -> int:
    return max(ONSET_MIN_FRAMES, int(np.ceil(ONSET_FRACTION * length)))


class GateTrajectory:
    """One trajectory scored by the gate, with the frozen model's cosine alongside."""

    def __init__(self, meta, probability, oracle_active, cosine, label) -> None:
        self.__dict__.update(meta)
        self.probability = probability
        self.oracle_active = oracle_active.astype(bool)
        self.cosine = cosine
        self.label = label.astype(bool)
        self.frames = len(probability)
        self.onset = np.arange(self.frames) < onset_cutoff(self.frames)

    def metrics_at(self, threshold: float) -> dict:
        fired = self.probability >= threshold
        active = self.oracle_active
        zero = ~active
        retained = fired & active
        false_positive = fired & zero
        onset_zero = zero & self.onset
        cosines = self.cosine[retained]
        return {
            "trajectory_id": self.trajectory_id,
            "episode_id": self.episode_id,
            "distribution": self.distribution,
            "hazard_present": bool(self.hazard_present),
            "frames": self.frames,
            "oracle_active_frames": int(active.sum()),
            "oracle_zero_frames": int(zero.sum()),
            "onset_zero_frames": int(onset_zero.sum()),
            "active_recall": float(retained.sum() / active.sum()) if active.any() else None,
            "oracle_zero_false_positive_rate": (float(false_positive.sum() / zero.sum())
                                                if zero.any() else None),
            "onset_zero_false_positive_rate": (
                float((fired & onset_zero).sum() / onset_zero.sum())
                if onset_zero.any() else None),
            "precision": (float(retained.sum() / fired.sum()) if fired.any() else None),
            "active_fraction": float(fired.mean()),
            "median_active_cosine": float(np.median(cosines)) if cosines.size else None,
            "positive_cosine_fraction": (float((cosines > 0).mean())
                                         if cosines.size else None),
            "max_consecutive_false_positive_run": thr.max_true_run(false_positive),
            "max_consecutive_onset_false_positive_run": thr.max_true_run(
                fired & onset_zero),
            "any_false_positive": bool(false_positive.any()),
            "any_onset_false_positive": bool((fired & onset_zero).any()),
            "false_positive_frames": int(false_positive.sum()),
        }


def aggregate(trajectories, threshold: float) -> dict:
    rows = [t.metrics_at(threshold) for t in trajectories]

    def collect(key, predicate=lambda r: True):
        return [r[key] for r in rows if predicate(r) and r[key] is not None]

    recalls = collect("active_recall")
    fprs = collect("oracle_zero_false_positive_rate")
    onset_fprs = collect("onset_zero_false_positive_rate")
    cosines = collect("median_active_cosine")
    positives = collect("positive_cosine_fraction")
    precisions = collect("precision")
    tp = sum(r["oracle_active_frames"] * (r["active_recall"] or 0.0) for r in rows)
    fp = sum(r["false_positive_frames"] for r in rows)
    fn = sum(r["oracle_active_frames"] for r in rows) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and precision + recall > 0 else None)
    return {
        "threshold": float(threshold),
        "trajectories": rows,
        "median_active_recall": float(np.median(recalls)) if recalls else None,
        "mean_trajectory_zero_fpr": float(np.mean(fprs)) if fprs else None,
        "mean_trajectory_onset_zero_fpr": float(np.mean(onset_fprs)) if onset_fprs else None,
        "max_trajectory_zero_fpr": float(np.max(fprs)) if fprs else None,
        "median_active_cosine": float(np.median(cosines)) if cosines else None,
        "median_positive_cosine_fraction": float(np.median(positives)) if positives else None,
        "median_precision": float(np.median(precisions)) if precisions else None,
        "frame_precision": precision, "frame_recall": recall, "frame_f1": f1,
        "max_consecutive_false_positive_run": max(
            r["max_consecutive_false_positive_run"] for r in rows),
        "max_consecutive_onset_false_positive_run": max(
            r["max_consecutive_onset_false_positive_run"] for r in rows),
        "trajectories_with_any_false_positive": sum(1 for r in rows
                                                    if r["any_false_positive"]),
        "onset_zero_trajectories_with_false_positive": sum(
            1 for r in rows if r["any_onset_false_positive"]),
        "max_hazard_absent_active_fraction": max(
            [r["active_fraction"] for r in rows if not r["hazard_present"]], default=None),
        "hazard_present": {
            "mean_zero_fpr": float(np.mean(collect(
                "oracle_zero_false_positive_rate", lambda r: r["hazard_present"]))) or 0.0
            if collect("oracle_zero_false_positive_rate", lambda r: r["hazard_present"])
            else None,
            "median_recall": float(np.median(collect(
                "active_recall", lambda r: r["hazard_present"])))
            if collect("active_recall", lambda r: r["hazard_present"]) else None,
        },
        "hazard_absent": {
            "mean_zero_fpr": float(np.mean(collect(
                "oracle_zero_false_positive_rate", lambda r: not r["hazard_present"])))
            if collect("oracle_zero_false_positive_rate", lambda r: not r["hazard_present"])
            else None,
            "max_active_fraction": max(
                [r["active_fraction"] for r in rows if not r["hazard_present"]],
                default=None),
        },
        "by_distribution": {
            name: {
                "trajectories": sum(1 for r in rows if r["distribution"] == name),
                "mean_zero_fpr": float(np.mean(collect(
                    "oracle_zero_false_positive_rate",
                    lambda r, n=name: r["distribution"] == n)))
                if collect("oracle_zero_false_positive_rate",
                           lambda r, n=name: r["distribution"] == n) else None,
                "median_recall": float(np.median(collect(
                    "active_recall", lambda r, n=name: r["distribution"] == n)))
                if collect("active_recall", lambda r, n=name: r["distribution"] == n)
                else None,
            } for name in SOURCE_MODES},
    }


def score(gate, model, head, partition, episodes, device, *, gate_mode="normal",
          rng=None) -> list:
    """Score one split. ``gate_mode`` selects the inference ablations E and F."""
    import torch

    trajectory_index = np.asarray(partition["trajectory"])
    oracle_active = np.asarray(partition["oracle_active"]).astype(bool)
    hazard = np.asarray(partition["hazard_present"]).astype(bool)
    modes = np.asarray(partition["source_mode"])
    changed = np.asarray(partition["changed"])
    keep = [i for i, e in enumerate(partition.episode_ids) if e in episodes]

    out = []
    for index in keep:
        rows = np.flatnonzero(trajectory_index == index)
        probability = np.zeros(len(rows))
        cosine = np.zeros(len(rows))
        for start in range(0, len(rows), 256):
            chunk = rows[start:start + 256]
            batch = make_batch(partition, chunk, device)
            closeness = batch["history"][:, -1].clone()
            valid = batch["history_valid"][:, -1].clone()
            if gate_mode == "fields_shuffled":
                order = torch.from_numpy(rng.permutation(len(chunk))).to(device)
                closeness = closeness[order]
                valid = valid[order]
            with torch.no_grad():
                probability[start:start + len(chunk)] = torch.sigmoid(
                    gate(closeness, valid)).cpu().numpy()
                field = model(batch["history"], batch["history_valid"], batch["state"])
                dq = head(field["current"]) - head(field["parked"])
            true_dq = np.asarray(partition["oracle_dq"], dtype=np.float64)[chunk]
            predicted = dq.double().cpu().numpy()
            denominator = (np.linalg.norm(predicted, axis=-1)
                           * np.linalg.norm(true_dq, axis=-1) + thr.COSINE_EPSILON)
            cosine[start:start + len(chunk)] = (predicted * true_dq).sum(-1) / denominator
        out.append(GateTrajectory(
            {"trajectory_id": partition.trajectory_ids[index],
             "episode_id": partition.episode_ids[index],
             "distribution": SOURCE_MODES[int(modes[rows[0]])],
             "hazard_present": bool(hazard[rows[0]])},
            probability, oracle_active[rows], cosine,
            changed[rows].reshape(len(rows), -1).any(axis=1)))
    return out


def calibrate(trajectories, episodes) -> dict:
    """Trajectory-wise calibration with a cluster bootstrap over episodes."""
    candidates = np.unique(np.concatenate([t.probability for t in trajectories]))
    cluster_of = {e: i for i, e in enumerate(sorted(episodes))}
    cluster_of_row = np.array([cluster_of[t.episode_id] for t in trajectories])

    fpr = np.zeros((len(trajectories), candidates.size))
    recall = np.zeros_like(fpr)
    for row, trajectory in enumerate(trajectories):
        zero = np.sort(trajectory.probability[~trajectory.oracle_active])
        active = np.sort(trajectory.probability[trajectory.oracle_active])
        fpr[row] = ((zero.size - np.searchsorted(zero, candidates, side="left")) / zero.size
                    if zero.size else np.nan)
        recall[row] = ((active.size - np.searchsorted(active, candidates, side="left"))
                       / active.size if active.size else np.nan)
    with np.errstate(invalid="ignore"):
        median_recall = np.nanmedian(recall, axis=0)
    screen = median_recall >= 0.80
    upper = np.full(candidates.size, np.inf)
    index = np.flatnonzero(screen)
    if index.size:
        upper[index] = thr.cluster_bootstrap_upper_bound(
            thr.cluster_means(fpr, cluster_of_row, len(cluster_of))[:, index])

    feasible = np.zeros(candidates.size, dtype=bool)
    detail = {}
    for i in index:
        if upper[i] > 0.02:
            continue
        block = aggregate(trajectories, candidates[i])
        detail[int(i)] = block
        hazard_absent_fpr = [r["oracle_zero_false_positive_rate"] for r in
                             block["trajectories"] if not r["hazard_present"]
                             and r["oracle_zero_false_positive_rate"] is not None]
        feasible[i] = bool(
            block["median_active_recall"] is not None
            and block["median_active_recall"] >= 0.80
            and (not hazard_absent_fpr or max(hazard_absent_fpr) <= 0.05)
            and block["max_consecutive_onset_false_positive_run"] <= 2
            and block["median_active_cosine"] is not None
            and block["median_active_cosine"] >= 0.75
            and block["median_positive_cosine_fraction"] is not None
            and block["median_positive_cosine_fraction"] >= 0.85)

    chosen = None
    if feasible.any():
        order = sorted(np.flatnonzero(feasible).tolist(), key=lambda i: (
            upper[i],
            detail[int(i)]["max_consecutive_onset_false_positive_run"],
            -median_recall[i], -candidates[i]))
        chosen = int(order[0])
    return {
        "candidate_count": int(candidates.size),
        "screened": int(screen.sum()),
        "feasible_count": int(feasible.sum()),
        "selected_index": chosen,
        "selected_threshold": float(candidates[chosen]) if chosen is not None else None,
        "bootstrap_upper_fpr": float(upper[chosen]) if chosen is not None else None,
        "detail": detail.get(chosen) if chosen is not None else None,
        "bootstrap": {"replicates": thr.BOOTSTRAP_REPLICATES, "seed": thr.BOOTSTRAP_SEED,
                      "resampled_unit": "episode", "one_sided": True,
                      "confidence": thr.CONFIDENCE},
        "selection_rule": ("lexicographic: lowest bootstrap FPR upper bound, lowest max "
                           "onset false-positive run, highest active recall, highest "
                           "threshold"),
    }


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUROC, written out so the definition travels with the report."""
    label = np.asarray(label, dtype=bool)
    positives = int(label.sum())
    negatives = int(label.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    return float((ranks[label].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def discrimination(gate, model, spec, partition, device) -> dict:
    """Gate vs old-head AUROC per split: does the gate underfit, or is proximity thin?"""
    import torch

    trajectory_index = np.asarray(partition["trajectory"])
    changed = np.asarray(partition["changed"])
    out = {}
    for split in ("gate_training", "threshold_calibration",
                  "nested_offline_evaluation"):
        episodes = set(spec["splits"][split]["episodes"])
        keep = [i for i, e in enumerate(partition.episode_ids) if e in episodes]
        rows = np.flatnonzero(np.isin(trajectory_index, keep))
        gate_score = np.zeros(len(rows))
        head_score = np.zeros(len(rows))
        for start in range(0, len(rows), 256):
            chunk = rows[start:start + 256]
            batch = make_batch(partition, chunk, device)
            with torch.no_grad():
                gate_score[start:start + len(chunk)] = torch.sigmoid(
                    gate(batch["history"][:, -1],
                         batch["history_valid"][:, -1])).cpu().numpy()
                field = model(batch["history"], batch["history_valid"], batch["state"])
                probability = field["changed_probability"]
                head_score[start:start + len(chunk)] = probability.reshape(
                    probability.shape[0], -1).amax(dim=1).cpu().numpy()
        label = changed[rows].reshape(len(rows), -1).any(axis=1)
        out[split] = {
            "frames": len(rows),
            "active_prevalence": float(label.mean()),
            "gate_auroc": auroc(gate_score, label),
            "old_head_auroc": auroc(head_score, label),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--gate-checkpoint", required=True, type=Path)
    ap.add_argument("--field-checkpoint", required=True, type=Path)
    ap.add_argument("--qpos-checkpoint", type=Path, default=None)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--onset-audit", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch

    stack = json.loads(args.stack.read_text())
    engine.set_sensor_names(stack["sensor_contract"]["ordered_names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = FrozenSafetyHead.load(args.safety_dir, device=device)
    model, field_payload = load_checkpoint(args.field_checkpoint, device)

    gate = build_gate().to(device)
    payload = torch.load(args.gate_checkpoint, map_location=device, weights_only=False)
    gate.load_state_dict(payload["gate"])
    gate.eval()

    spec = json.loads(args.partition.read_text())
    train = load_partition(args.cache, "reference_train")

    # ---- 1. calibrate on the calibration split only -------------------------------
    calibration_episodes = set(spec["splits"]["threshold_calibration"]["episodes"])
    calibration = score(gate, model, head, train, calibration_episodes, device)
    fitted = calibrate(calibration, calibration_episodes)
    if fitted["selected_threshold"] is None:
        # The task terminates here (handoff step 12). Everything below is characterisation
        # of *why*, not selection: no threshold, model or split choice can follow, so these
        # numbers cannot influence any decision. The nested split is touched only to
        # separate "the gate underfits" from "proximity is insufficient", and that use is
        # disclosed rather than hidden.
        sweep = []
        for tau in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999):
            block = aggregate(calibration, tau)
            hazard_absent = [r["oracle_zero_false_positive_rate"]
                             for r in block["trajectories"] if not r["hazard_present"]
                             and r["oracle_zero_false_positive_rate"] is not None]
            sweep.append({
                "threshold": tau,
                "median_active_recall": block["median_active_recall"],
                "mean_trajectory_zero_fpr": block["mean_trajectory_zero_fpr"],
                "max_hazard_absent_trajectory_fpr": max(hazard_absent, default=None),
                "max_consecutive_onset_false_positive_run":
                    block["max_consecutive_onset_false_positive_run"],
                "median_active_cosine": block["median_active_cosine"],
                "recall_meets_080": (block["median_active_recall"] or 0) >= 0.80,
                "fpr_meets_002": (block["mean_trajectory_zero_fpr"] or 1) <= 0.02,
            })
        separability = discrimination(gate, model, spec, train, device)
        report = {
            "schema": "hybrid_obstacle_prox_activity_evaluate_v1",
            "gate_checkpoint": str(args.gate_checkpoint),
            "gate_checkpoint_epoch": payload["epoch"],
            "gate_parameter_count": payload["parameter_count"],
            "partition_manifest_sha256": spec["manifest_sha256"],
            "feasible": False,
            "calibration": {k: v for k, v in fitted.items() if k != "detail"},
            "threshold_sweep": sweep,
            "no_threshold_satisfies": (
                "recall >= 0.80 and mean trajectory zero FPR <= 0.02 are not jointly "
                "attainable at any operating point"),
            "separability": separability,
            "post_termination_diagnostics_note": (
                "the sweep and AUROC below were computed after calibration had already "
                "failed and the task had terminated; no threshold, model or split "
                "selection follows them"),
            "decision_if_infeasible": "PROX_ACTIVITY_GATE_CALIBRATION_INFEASIBLE",
        }
        report["report_sha256"] = thr.canonical_hash(report)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True,
                                       default=str) + "\n")
        print("NO FEASIBLE THRESHOLD")
        print("  threshold sweep (calibration split):")
        for row in sweep:
            print(f"    tau={row['threshold']:<6} recall={row['median_active_recall']:.3f} "
                  f"meanFPR={row['mean_trajectory_zero_fpr']:.4f} "
                  f"maxHazAbsFPR={row['max_hazard_absent_trajectory_fpr']:.4f} "
                  f"onsetRun={row['max_consecutive_onset_false_positive_run']}")
        for split, block in separability.items():
            print(f"  AUROC {split:<28} gate={block['gate_auroc']:.4f} "
                  f"old_head={block['old_head_auroc']:.4f}")
        print(f"wrote {args.out}")
        return 7
    tau = fitted["selected_threshold"]
    print(f"calibrated threshold {tau:.8f} "
          f"(bootstrap upper FPR {fitted['bootstrap_upper_fpr']:.5f}, "
          f"{fitted['feasible_count']} feasible of {fitted['candidate_count']})")

    # ---- 2. nested offline evaluation, opened only now ----------------------------
    nested_episodes = set(spec["splits"]["nested_offline_evaluation"]["episodes"])
    nested = score(gate, model, head, train, nested_episodes, device)
    nested_block = aggregate(nested, tau)
    nested_failures = []
    if nested_block["median_active_recall"] < NESTED_MIN_RECALL:
        nested_failures.append(f"median recall {nested_block['median_active_recall']:.4f}"
                               f" < {NESTED_MIN_RECALL}")
    if nested_block["mean_trajectory_zero_fpr"] > NESTED_MAX_MEAN_FPR:
        nested_failures.append(f"mean zero FPR "
                               f"{nested_block['mean_trajectory_zero_fpr']:.4f} > "
                               f"{NESTED_MAX_MEAN_FPR}")
    if (nested_block["hazard_absent"]["max_active_fraction"] or 0) > \
            NESTED_MAX_HAZARD_ABSENT_ACTIVE:
        nested_failures.append("hazard-absent active fraction "
                               f"{nested_block['hazard_absent']['max_active_fraction']:.4f}"
                               f" > {NESTED_MAX_HAZARD_ABSENT_ACTIVE}")
    if nested_block["max_consecutive_false_positive_run"] > NESTED_MAX_RUN:
        nested_failures.append(
            f"FP run {nested_block['max_consecutive_false_positive_run']} > {NESTED_MAX_RUN}")
    if (nested_block["median_active_cosine"] or 0) < NESTED_MIN_COSINE:
        nested_failures.append(f"median cosine {nested_block['median_active_cosine']}")
    if (nested_block["median_positive_cosine_fraction"] or 0) < NESTED_MIN_POSITIVE_COSINE:
        nested_failures.append("positive-cosine fraction "
                               f"{nested_block['median_positive_cosine_fraction']}")

    # ---- 3. offline controls at each method's operating point ---------------------
    rng = np.random.default_rng(thr.BOOTSTRAP_SEED)
    controls: dict[str, dict] = {}

    # A: the old shared activity head, at its previously frozen threshold
    old_head = []
    trajectory_index = np.asarray(train["trajectory"])
    for trajectory in nested:
        index = next(i for i, t in enumerate(train.trajectory_ids)
                     if t == trajectory.trajectory_id)
        rows = np.flatnonzero(trajectory_index == index)
        activity = np.zeros(len(rows))
        for start in range(0, len(rows), 256):
            chunk = rows[start:start + 256]
            batch = make_batch(train, chunk, device)
            with torch.no_grad():
                out = model(batch["history"], batch["history_valid"], batch["state"])
                probability = out["changed_probability"]
                activity[start:start + len(chunk)] = probability.reshape(
                    probability.shape[0], -1).amax(dim=1).cpu().numpy()
        old_head.append(GateTrajectory(
            {"trajectory_id": trajectory.trajectory_id,
             "episode_id": trajectory.episode_id,
             "distribution": trajectory.distribution,
             "hazard_present": trajectory.hazard_present},
            activity, trajectory.oracle_active, trajectory.cosine, trajectory.label))
    controls["A_old_shared_activity_head"] = aggregate(old_head, PREVIOUS_THRESHOLD)
    controls["B_prox_evidence_gate"] = nested_block

    # C: constant base rate
    base_rate = float(np.mean([t.oracle_active.mean() for t in nested]))
    constant = [GateTrajectory(
        {"trajectory_id": t.trajectory_id, "episode_id": t.episode_id,
         "distribution": t.distribution, "hazard_present": t.hazard_present},
        np.full(t.frames, base_rate), t.oracle_active, t.cosine, t.label) for t in nested]
    controls["C_constant_base_rate"] = aggregate(constant, tau)
    controls["C_constant_base_rate"]["constant_probability"] = base_rate

    # D: QPOS_ONLY activity control, read-only from a prior artifact
    if args.qpos_checkpoint and args.qpos_checkpoint.is_file():
        qpos_model, _ = load_checkpoint(args.qpos_checkpoint, device)
        qpos = []
        for trajectory in nested:
            index = next(i for i, t in enumerate(train.trajectory_ids)
                         if t == trajectory.trajectory_id)
            rows = np.flatnonzero(trajectory_index == index)
            activity = np.zeros(len(rows))
            for start in range(0, len(rows), 256):
                chunk = rows[start:start + 256]
                batch = make_batch(train, chunk, device)
                with torch.no_grad():
                    out = qpos_model(batch["history"], batch["history_valid"],
                                     batch["state"])
                    probability = out["changed_probability"]
                    activity[start:start + len(chunk)] = probability.reshape(
                        probability.shape[0], -1).amax(dim=1).cpu().numpy()
            qpos.append(GateTrajectory(
                {"trajectory_id": trajectory.trajectory_id,
                 "episode_id": trajectory.episode_id,
                 "distribution": trajectory.distribution,
                 "hazard_present": trajectory.hazard_present},
                activity, trajectory.oracle_active, trajectory.cosine, trajectory.label))
        controls["D_qpos_only_activity"] = aggregate(qpos, PREVIOUS_THRESHOLD)
        controls["D_qpos_only_activity"]["source"] = str(args.qpos_checkpoint)
    else:
        controls["D_qpos_only_activity"] = {"available": False}

    # E: sensor identities shuffled (inference ablation of the same trained gate)
    saved = gate.sensor_embedding.detach().clone()
    with torch.no_grad():
        gate.sensor_embedding.copy_(saved[torch.from_numpy(
            rng.permutation(saved.shape[0])).to(device)])
    controls["E_sensor_identity_shuffled"] = aggregate(
        score(gate, model, head, train, nested_episodes, device), tau)
    with torch.no_grad():
        gate.sensor_embedding.copy_(saved)
    assert torch.equal(gate.sensor_embedding, saved)

    # F: current fields shuffled across frames
    controls["F_current_field_shuffled"] = aggregate(
        score(gate, model, head, train, nested_episodes, device,
              gate_mode="fields_shuffled", rng=np.random.default_rng(thr.BOOTSTRAP_SEED)),
        tau)

    # ---- 4. reused diagnostic audit ----------------------------------------------
    onset_audit = json.loads(args.onset_audit.read_text())
    historical = onset_audit["known_false_positive_frames"]
    diagnostic_trajectories = []
    historical_rows = []
    for name in CONSUMED_PARTITIONS:
        partition = load_partition(args.cache, name)
        episodes = set(partition.episode_ids)
        scored = score(gate, model, head, partition, episodes, device)
        for trajectory in scored:
            trajectory.partition = name
        diagnostic_trajectories.extend(scored)
    by_id = {t.trajectory_id: t for t in diagnostic_trajectories}
    for record in historical:
        trajectory = by_id.get(record["trajectory_id"])
        probability = (float(trajectory.probability[record["step"]])
                       if trajectory is not None else None)
        historical_rows.append({
            **record, "new_gate_probability": probability,
            "old_head_activity": record["activity"],
            "still_active": bool(probability is not None and probability >= tau)})
    diagnostic_block = aggregate(diagnostic_trajectories, tau)

    failing_episode = "499eee89fb91"
    failing = [t for t in diagnostic_trajectories
               if t.episode_id.startswith(failing_episode)]
    failing_rows = [t.metrics_at(tau) for t in failing]
    failing_max_run = max((r["max_consecutive_false_positive_run"] for r in failing_rows),
                          default=0)

    diagnostic_failures = []
    if failing_max_run > DIAG_MAX_FAILING_EPISODE_RUN:
        diagnostic_failures.append(
            f"previously failing episode still has a run of {failing_max_run}")
    if (diagnostic_block["max_hazard_absent_active_fraction"] or 0) > \
            DIAG_MAX_HAZARD_ABSENT_ACTIVE:
        diagnostic_failures.append(
            "hazard-absent active fraction "
            f"{diagnostic_block['max_hazard_absent_active_fraction']:.4f} > "
            f"{DIAG_MAX_HAZARD_ABSENT_ACTIVE}")
    if diagnostic_block["mean_trajectory_zero_fpr"] > DIAG_MAX_MEAN_FPR:
        diagnostic_failures.append(
            f"mean zero FPR {diagnostic_block['mean_trajectory_zero_fpr']:.4f} > "
            f"{DIAG_MAX_MEAN_FPR}")
    if diagnostic_block["median_active_recall"] < DIAG_MIN_MEDIAN_RECALL:
        diagnostic_failures.append(
            f"median recall {diagnostic_block['median_active_recall']:.4f} < "
            f"{DIAG_MIN_MEDIAN_RECALL}")

    still_active = [r for r in historical_rows if r["still_active"]]
    report = {
        "schema": "hybrid_obstacle_prox_activity_evaluate_v1",
        "gate_checkpoint": str(args.gate_checkpoint),
        "gate_checkpoint_epoch": payload["epoch"],
        "gate_parameter_count": payload["parameter_count"],
        "field_checkpoint_config_hash": field_payload["config_hash"],
        "partition_manifest_sha256": spec["manifest_sha256"],
        "feasible": True,
        "calibration": {k: v for k, v in fitted.items() if k != "detail"},
        "calibration_metrics": fitted["detail"],
        "selected_threshold": tau,
        "nested_offline": nested_block,
        "nested_gates": {
            "thresholds": {"min_recall": NESTED_MIN_RECALL,
                           "max_mean_fpr": NESTED_MAX_MEAN_FPR,
                           "max_hazard_absent_active": NESTED_MAX_HAZARD_ABSENT_ACTIVE,
                           "max_run": NESTED_MAX_RUN, "min_cosine": NESTED_MIN_COSINE,
                           "min_positive_cosine": NESTED_MIN_POSITIVE_COSINE},
            "failures": nested_failures, "passed": not nested_failures},
        "controls": controls,
        "historical_onset_regression": {
            "frames": historical_rows,
            "count": len(historical_rows),
            "still_active": len(still_active),
            "all_inactive": not still_active,
            "note": "regression test, not an independent generalization claim",
        },
        "reused_diagnostic": {
            "label": "reused_nonconfirmatory_diagnostic",
            "partitions": list(CONSUMED_PARTITIONS),
            "trajectories": len(diagnostic_trajectories),
            "metrics": {k: v for k, v in diagnostic_block.items() if k != "trajectories"},
            "previously_failing_episode": {
                "episode_prefix": failing_episode,
                "trajectories": failing_rows,
                "max_consecutive_false_positive_run": failing_max_run,
            },
            "blocking_checks": {"failures": diagnostic_failures,
                                "passed": not diagnostic_failures},
        },
        "decision_if_offline_invalid": "PROX_ACTIVITY_GATE_OFFLINE_INVALID",
    }
    report["report_sha256"] = thr.canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"\nnested offline: recall={nested_block['median_active_recall']} "
          f"meanFPR={nested_block['mean_trajectory_zero_fpr']:.5f} "
          f"run={nested_block['max_consecutive_false_positive_run']} "
          f"cos={nested_block['median_active_cosine']}")
    print(f"  nested gates passed: {report['nested_gates']['passed']} {nested_failures}")
    print("controls (onset FPR / overall FPR / recall):")
    for name, block in controls.items():
        if not block.get("trajectories") and "median_active_recall" not in block:
            print(f"  {name:<32} unavailable")
            continue
        print(f"  {name:<32} onset={block['mean_trajectory_onset_zero_fpr']} "
              f"overall={block['mean_trajectory_zero_fpr']} "
              f"recall={block['median_active_recall']}")
    print(f"historical onset frames still active: {len(still_active)}/17")
    print(f"previously failing episode max run  : {failing_max_run}")
    print(f"diagnostic blocking passed          : "
          f"{report['reused_diagnostic']['blocking_checks']['passed']} "
          f"{diagnostic_failures}")
    print(f"wrote {args.out}")
    return 0 if (report["nested_gates"]["passed"]
                 and report["reused_diagnostic"]["blocking_checks"]["passed"]
                 and report["historical_onset_regression"]["all_inactive"]) else 9


if __name__ == "__main__":
    raise SystemExit(main())
