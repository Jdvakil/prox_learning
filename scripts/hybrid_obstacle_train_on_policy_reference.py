#!/usr/bin/env python3
"""Aggregated training, tau/rho_max calibration and disjoint validation for V2.

Handoff steps 6-11.

* **round 0** -- retrain the *unchanged* fixed architecture from a fresh initialisation on
  an equal-weighted mixture of expert, ACT-only-on-policy and oracle-on-policy frames.
* **round 1** -- after the single permitted learner-induced collection, retrain again from
  a fresh initialisation on the four distributions at 25% each.
* **calibration** -- fit ``tau`` and ``rho_max`` on the 8 reference-calibration
  trajectories only, by the predeclared lexicographic rule.
* **validation** -- evaluate the frozen model, tau and rho_max, without refitting, on the
  8 reference-validation trajectories and the 20 offline-test trajectories.

Nothing about the architecture, optimiser, learning rate, weight decay, batch size, epoch
budget, loss or seed changes from V1. Neither round continues from V1 weights.

Distribution weighting is by *distribution*, then uniformly by row, then uniformly by
frame -- so a long expert trajectory cannot dominate simply by having more frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT / "scripts"), str(ROOT / "submodules" / "act")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from deployable_reference import (
    FEATURE_BUILDERS,
    FEATURE_FIELDS,
    FEATURE_WIDTHS,
    MLP_REFERENCE_ID,
    PostureSkinMlpReference,
    Standardizer,
    SupportEnvelopeGate,
)
from hybrid_obstacle_train_deployable_reference import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    SEED,
    WEIGHT_DECAY,
    train_mlp,
)

ORACLE_ZERO_TOLERANCE = 1e-7
#: frames drawn per distribution per epoch-equivalent; keeps the mixture exactly balanced
SAMPLES_PER_DISTRIBUTION = 6000
RECALL_FLOOR = 0.80
COSINE_FLOOR = 0.70
POSITIVE_FLOOR = 0.80
CALIBRATION_FALSE_ACTIVATION_CEILING = 0.01
RHO_PERCENTILE = 99.0


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
def load_expert(paired_dir: Path, episode_ids: list[str]) -> list[dict[str, np.ndarray]]:
    """Expert frames from the V1 paired dataset, renamed into the V2 label namespace."""
    rows = []
    for episode_id in episode_ids:
        path = paired_dir / f"{episode_id}.npz"
        if not path.is_file():
            raise SystemExit(f"expert paired file missing: {path}")
        blob = np.load(path, allow_pickle=False)
        oracle = np.asarray(blob["oracle_dq"], dtype=np.float32)
        rows.append({
            **{key: np.asarray(blob[key]) for key in FEATURE_FIELDS[MLP_REFERENCE_ID]},
            "privileged_parked_head": np.asarray(blob["parked_head"], dtype=np.float32),
            "privileged_oracle_dq": oracle,
            "privileged_oracle_norm": np.linalg.norm(oracle, axis=1),
            "privileged_teacher_dq": np.asarray(blob["teacher_dq"], dtype=np.float32),
            "privileged_teacher_active": np.asarray(blob["teacher_active"], dtype=bool),
            "minimum_depth": np.asarray(blob["minimum_depth"], dtype=np.float64),
            "episode_id": episode_id,
            "hazard_present": bool(np.any(blob["privileged_hazard_present"])),
        })
    return rows


def load_on_policy(schedule: dict, distribution: str,
                   episode_ids: set[str]) -> list[dict[str, np.ndarray]]:
    rows = []
    for entry in schedule["entries"]:
        if entry["episode_id"] not in episode_ids:
            continue
        path = Path(entry["output_dir"]) / "frames.npz"
        if not path.is_file():
            raise SystemExit(f"on-policy frames missing: {path}")
        blob = np.load(path, allow_pickle=False)
        summary = json.loads((Path(entry["output_dir"]) / "summary.json").read_text())
        if summary["distribution"] != distribution:
            continue
        count = len(blob["timestep"])
        rows.append({
            **{key: np.asarray(blob[key]) for key in FEATURE_FIELDS[MLP_REFERENCE_ID]},
            "privileged_parked_head": np.asarray(blob["privileged_parked_head"],
                                                 dtype=np.float32),
            "privileged_oracle_dq": np.asarray(blob["privileged_oracle_dq"],
                                               dtype=np.float32),
            "privileged_oracle_norm": np.asarray(blob["privileged_oracle_norm"],
                                                 dtype=np.float64),
            # the analytic teacher is not recomputed on-policy; absence is explicit
            "privileged_teacher_dq": np.zeros((count, 7), dtype=np.float32),
            "privileged_teacher_active": np.zeros(count, dtype=bool),
            "minimum_depth": np.asarray(blob["minimum_depth"], dtype=np.float64),
            "episode_id": entry["episode_id"],
            "hazard_present": bool(entry["hazard_present"]),
        })
    return rows


def stack(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = [k for k in rows[0] if isinstance(rows[0][k], np.ndarray)]
    out = {k: np.concatenate([r[k] for r in rows], axis=0) for k in keys}
    out["row_index"] = np.concatenate(
        [np.full(len(r["timestep"] if "timestep" in r else r["qpos"]), i, dtype=np.int64)
         for i, r in enumerate(rows)])
    out["hazard_present_row"] = np.concatenate(
        [np.full(len(r["qpos"]), r["hazard_present"], dtype=bool) for r in rows])
    return out


def sample_balanced(distributions: dict[str, list[dict[str, np.ndarray]]],
                    per_distribution: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Equal weight per distribution, uniform over rows, then uniform over frames."""
    rng = np.random.default_rng(seed)
    features, targets, provenance = [], [], {}
    for name in sorted(distributions):
        rows = distributions[name]
        if not rows:
            raise SystemExit(f"distribution {name} has no rows")
        picked_rows = rng.integers(0, len(rows), size=per_distribution)
        chosen_features, chosen_targets = [], []
        for row_index in picked_rows:
            row = rows[row_index]
            frame = int(rng.integers(0, len(row["qpos"])))
            chosen_features.append(
                FEATURE_BUILDERS[MLP_REFERENCE_ID](
                    {k: row[k][frame:frame + 1] for k in FEATURE_FIELDS[MLP_REFERENCE_ID]})[0])
            chosen_targets.append(row["privileged_parked_head"][frame])
        features.append(np.asarray(chosen_features, dtype=np.float32))
        targets.append(np.asarray(chosen_targets, dtype=np.float32))
        provenance[name] = {"rows": len(rows), "frames_drawn": per_distribution,
                            "total_frames_available": int(sum(len(r["qpos"]) for r in rows))}
    return (np.concatenate(features, axis=0), np.concatenate(targets, axis=0), provenance)


# --------------------------------------------------------------------------- #
def predict(model, data: dict[str, np.ndarray]) -> np.ndarray:
    features = FEATURE_BUILDERS[MLP_REFERENCE_ID](
        {k: data[k] for k in FEATURE_FIELDS[MLP_REFERENCE_ID]})
    return np.asarray(model.predict(features), dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (na > 1e-12) & (nb > 1e-12)
    out = np.full(len(a), np.nan)
    out[ok] = (a[ok] * b[ok]).sum(axis=1) / (na[ok] * nb[ok])
    return out


def calibrate(predicted_dq: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Lexicographic tau selection on the calibration trajectories only."""
    oracle = data["privileged_oracle_dq"]
    oracle_norm = np.asarray(data["privileged_oracle_norm"], dtype=np.float64)
    teacher_active = np.asarray(data["privileged_teacher_active"], dtype=bool)
    teacher_evaluable = bool(teacher_active.any())
    # oracle-active: above the numerical floor, and teacher-active where the teacher can
    # be evaluated at all. On-policy frames carry no teacher, so the second clause only
    # narrows the expert-derived frames.
    active = oracle_norm > ORACLE_ZERO_TOLERANCE
    if teacher_evaluable:
        active = active & (teacher_active | ~data["teacher_evaluable_row"])
    zero = oracle_norm <= ORACLE_ZERO_TOLERANCE
    norms = np.linalg.norm(predicted_dq, axis=1)
    cosines = cosine(predicted_dq, oracle)

    candidates = np.unique(norms)
    accepted, rejected = [], []
    for tau in candidates:
        retained = active & (norms >= tau)
        recall = retained.sum() / max(active.sum(), 1)
        if recall < RECALL_FLOOR:
            rejected.append({"tau": float(tau), "why": "recall", "recall": float(recall)})
            continue
        values = cosines[retained & np.isfinite(cosines)]
        if not values.size:
            rejected.append({"tau": float(tau), "why": "no finite cosine"})
            continue
        median = float(np.median(values))
        positive = float(np.mean(values > 0))
        if median < COSINE_FLOOR or positive < POSITIVE_FLOOR:
            rejected.append({"tau": float(tau), "why": "direction",
                             "median": median, "positive": positive})
            continue
        false_activation = float(np.mean(norms[zero] >= tau)) if zero.any() else 0.0
        accepted.append({"tau": float(tau), "recall": float(recall), "median_cosine": median,
                         "positive_fraction": positive,
                         "false_activation": false_activation})
    rho_max_value = (float(np.percentile(oracle_norm[active], RHO_PERCENTILE))
                     if active.any() else 0.0)
    if not accepted:
        return {"feasible": False, "tau": None,
                # rho_max depends only on the privileged oracle, never on the model, so it
                # is well defined even when no tau satisfies the direction gates
                "rho_max": rho_max_value,
                "rho_rule": f"{RHO_PERCENTILE}th percentile of the privileged oracle "
                            "differential norm on active calibration frames",
                "rejected_examples": rejected[:5],
                "best_direction_seen": max(
                    ({"median_cosine": r.get("median"), "positive": r.get("positive")}
                     for r in rejected if r.get("median") is not None),
                    key=lambda r: r["median_cosine"], default=None),
                "active_frames": int(active.sum()), "zero_frames": int(zero.sum())}
    # 4. minimise oracle-zero false activation, 5. highest tau among equals
    best = min(accepted, key=lambda c: (c["false_activation"], -c["tau"]))
    return {
        "feasible": True,
        "tau": best["tau"],
        "rho_max": rho_max_value,
        "rho_rule": f"{RHO_PERCENTILE}th percentile of the privileged oracle differential "
                    "norm on active calibration frames",
        "recall": best["recall"],
        "median_cosine": best["median_cosine"],
        "positive_fraction": best["positive_fraction"],
        "calibration_false_activation": best["false_activation"],
        "calibration_false_activation_within_1pct":
            best["false_activation"] <= CALIBRATION_FALSE_ACTIVATION_CEILING,
        "active_frames": int(active.sum()),
        "zero_frames": int(zero.sum()),
        "candidates_considered": len(candidates),
        "candidates_accepted": len(accepted),
        "teacher_evaluable": teacher_evaluable,
        "lexicographic_rule": ["recall >= 0.80", "median cosine >= 0.70",
                               "positive fraction >= 0.80",
                               "minimise oracle-zero false activation",
                               "highest tau among equals"],
    }


def evaluate(name: str, predicted_dq: np.ndarray, data: dict[str, np.ndarray],
             gate: SupportEnvelopeGate) -> dict[str, Any]:
    oracle = data["privileged_oracle_dq"]
    oracle_norm = np.asarray(data["privileged_oracle_norm"], dtype=np.float64)
    active = oracle_norm > ORACLE_ZERO_TOLERANCE
    zero = ~active
    hazard_absent = ~np.asarray(data["hazard_present_row"], dtype=bool)
    norms = np.linalg.norm(predicted_dq, axis=1)
    executed = np.stack([gate(predicted_dq[i])[0] for i in range(len(predicted_dq))])
    activated = np.linalg.norm(executed, axis=1) > 0
    cosines = cosine(executed, oracle)
    retained = active & activated
    finite = np.isfinite(cosines)

    ratios = np.divide(np.linalg.norm(executed, axis=1), oracle_norm,
                       out=np.full(len(norms), np.nan), where=oracle_norm > 1e-12)
    ratio_values = ratios[retained & np.isfinite(ratios)]

    runs, longest_after_zero, current = [], 0, 0
    for index in range(len(activated)):
        if activated[index] and zero[index]:
            current += 1
            longest_after_zero = max(longest_after_zero, current)
        else:
            if current:
                runs.append(current)
            current = 0
    if current:
        runs.append(current)

    per_row_activation = {}
    for row_index in np.unique(data["row_index"]):
        mask = data["row_index"] == row_index
        per_row_activation[int(row_index)] = {
            "activation_rate": float(np.mean(activated[mask])),
            "hazard_present": bool(data["hazard_present_row"][mask][0])}
    hazard_absent_rows_over_5pct = [k for k, v in per_row_activation.items()
                                    if not v["hazard_present"]
                                    and v["activation_rate"] > 0.05]

    teacher_active = np.asarray(data["privileged_teacher_active"], dtype=bool)
    teacher_cosines = cosine(executed, data["privileged_teacher_dq"])[
        teacher_active & activated]
    teacher_cosines = teacher_cosines[np.isfinite(teacher_cosines)]

    grasped = data["gripper_state"][:, 0] < np.median(data["gripper_state"][:, 0])

    return {
        "set": name,
        "frames": len(norms),
        "oracle_active_frames": int(active.sum()),
        "oracle_zero_frames": int(zero.sum()),
        "oracle_active_recall": float(retained.sum() / max(active.sum(), 1)),
        "median_oracle_cosine": (float(np.median(cosines[retained & finite]))
                                 if (retained & finite).any() else None),
        "positive_cosine_fraction": (float(np.mean(cosines[retained & finite] > 0))
                                     if (retained & finite).any() else None),
        "differential_mae_active": (float(np.mean(np.abs(executed[active] - oracle[active])))
                                    if active.any() else None),
        "differential_mae_all": float(np.mean(np.abs(executed - oracle))),
        "median_norm_ratio": float(np.median(ratio_values)) if ratio_values.size else None,
        "p95_norm_ratio": (float(np.percentile(ratio_values, 95))
                           if ratio_values.size else None),
        "oracle_zero_false_activation_rate": (float(np.mean(activated[zero]))
                                              if zero.any() else None),
        "false_activation_episode_count": len(runs),
        "max_consecutive_false_activation": int(longest_after_zero),
        "median_teacher_cosine": (float(np.median(teacher_cosines))
                                  if teacher_cosines.size else None),
        "positive_teacher_cosine_fraction": (float(np.mean(teacher_cosines > 0))
                                             if teacher_cosines.size else None),
        "teacher_frames": int(teacher_cosines.size),
        "oracle_zero_output_rms": (float(np.sqrt(np.mean(norms[zero] ** 2)))
                                   if zero.any() else None),
        "hazard_absent_activation_rate": (float(np.mean(activated[hazard_absent]))
                                          if hazard_absent.any() else None),
        "hazard_absent_rows_over_5pct": hazard_absent_rows_over_5pct,
        "grasp_contact_false_activation": (float(np.mean(activated[grasped & zero]))
                                           if (grasped & zero).any() else None),
        "uncapped_predicted_norm_p99": float(np.percentile(norms, 99)),
        "capped_fraction_of_activated": (
            float(np.mean(norms[activated] > gate.rho_max)) if activated.any() else None),
        "per_row_activation": per_row_activation,
    }


def gates_for(report: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "oracle_active_recall_at_least_75pct": (report["oracle_active_recall"], 0.75,
                                                report["oracle_active_recall"] >= 0.75),
        "median_oracle_cosine_at_least_0p70": (
            report["median_oracle_cosine"], 0.70,
            report["median_oracle_cosine"] is not None
            and report["median_oracle_cosine"] >= 0.70),
        "positive_cosine_fraction_at_least_80pct": (
            report["positive_cosine_fraction"], 0.80,
            report["positive_cosine_fraction"] is not None
            and report["positive_cosine_fraction"] >= 0.80),
        "oracle_zero_false_activation_within_2pct": (
            report["oracle_zero_false_activation_rate"], 0.02,
            (report["oracle_zero_false_activation_rate"] or 0.0) <= 0.02),
        "median_norm_ratio_between_0p5_and_1p5": (
            report["median_norm_ratio"], [0.5, 1.5],
            report["median_norm_ratio"] is not None
            and 0.5 <= report["median_norm_ratio"] <= 1.5),
        "no_hazard_absent_row_over_5pct_activation": (
            report["hazard_absent_rows_over_5pct"], 0,
            not report["hazard_absent_rows_over_5pct"]),
        "no_more_than_two_consecutive_false_active_frames": (
            report["max_consecutive_false_activation"], 2,
            report["max_consecutive_false_activation"] <= 2),
    }
    return {name: {"value": value, "threshold": threshold, "passed": bool(passed)}
            for name, (value, threshold, passed) in checks.items()}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--labelling-schedule", required=True, type=Path)
    ap.add_argument("--learner-schedule", type=Path, default=None)
    ap.add_argument("--expert-paired-dir", required=True, type=Path)
    ap.add_argument("--round", required=True, choices=("0", "1"))
    ap.add_argument("--artifact-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--deployment-manifest", type=Path, default=None)
    args = ap.parse_args()
    for name in ("partition", "labelling_schedule", "expert_paired_dir", "artifact_dir",
                 "out"):
        setattr(args, name, Path(getattr(args, name)).resolve())

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    partition = json.loads(args.partition.read_text())
    labelling = json.loads(args.labelling_schedule.read_text())
    train_ids = [r["episode_id"] for r in partition["partitions"]["reference_train"]]

    distributions = {
        "expert": load_expert(args.expert_paired_dir, train_ids),
        "act_only_on_policy": load_on_policy(labelling, "act_only_on_policy",
                                             set(train_ids)),
        "oracle_on_policy": load_on_policy(labelling, "oracle_on_policy", set(train_ids)),
    }
    if args.round == "1":
        if not args.learner_schedule:
            raise SystemExit("round 1 requires --learner-schedule")
        learner = json.loads(Path(args.learner_schedule).resolve().read_text())
        distributions["learner_on_policy"] = load_on_policy(
            learner, "learner_on_policy", set(train_ids))

    weight = 1.0 / len(distributions)
    features, targets, provenance = sample_balanced(distributions, SAMPLES_PER_DISTRIBUTION,
                                                    SEED)
    standardizer = Standardizer.fit(features)

    # validation inside training uses the reference-validation trajectories only
    validation_ids = [r["episode_id"] for r in partition["partitions"]["reference_validation"]]
    validation_rows = load_on_policy(labelling, "act_only_on_policy", set(validation_ids)) \
        + load_on_policy(labelling, "oracle_on_policy", set(validation_ids))
    validation = stack(validation_rows)
    validation_features = FEATURE_BUILDERS[MLP_REFERENCE_ID](
        {k: validation[k] for k in FEATURE_FIELDS[MLP_REFERENCE_ID]})

    model, training_report = train_mlp(features, targets, validation_features,
                                       validation["privileged_parked_head"],
                                       standardizer, device)
    reference = PostureSkinMlpReference(standardizer=standardizer, model=model,
                                        device=device)
    label = f"POSTURE_SKIN_MLP_REFERENCE_V2_ROUND{args.round}"
    reference.metadata = {"label": label, "seed": SEED, "lr": LEARNING_RATE,
                          "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                          "max_epochs": MAX_EPOCHS, "loss": "SmoothL1",
                          "optimizer": "AdamW", "fresh_initialisation": True,
                          "continued_from_v1": False,
                          "distributions": sorted(distributions)}
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.artifact_dir / f"{label.lower()}.pt"
    reference.save(artifact)
    reloaded = PostureSkinMlpReference.load(artifact, device=device)
    strict_ok = bool(np.array_equal(reloaded.predict(validation_features),
                                    reference.predict(validation_features)))

    report: dict[str, Any] = {
        "schema": "hybrid_obstacle_on_policy_reference_training_v2",
        "label": label,
        "round": int(args.round),
        "architecture_unchanged": True,
        "training_configuration": {"seed": SEED, "lr": LEARNING_RATE,
                                   "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                                   "max_epochs": MAX_EPOCHS, "loss": "SmoothL1",
                                   "optimizer": "AdamW"},
        "fresh_initialisation": True,
        "continued_from_previous_checkpoint": False,
        "feature_width": FEATURE_WIDTHS[MLP_REFERENCE_ID],
        "parameters": training_report["parameters"],
        "best_epoch": training_report["best_epoch"],
        "best_validation_loss": training_report["best_validation_loss"],
        "strict_reload_bitwise_identical": strict_ok,
        "distribution_weighting": {name: weight for name in distributions},
        "distribution_provenance": provenance,
        "samples_per_distribution": SAMPLES_PER_DISTRIBUTION,
        "sampling_rule": "uniform over rows, then uniform over frames, within each "
                         "distribution",
        "artifact_path": str(artifact),
        "artifact_file_sha256": sha256_file(artifact),
        "model_digest": reference.digest(),
        "input_statistics_sha256": standardizer.digest(),
        "partition_sha256": partition["partition_sha256"],
        "labelling_schedule_sha256": labelling["schedule_sha256"],
    }

    # ---- calibrate on the calibration trajectories only -------------------- #
    # Round 0 gets a *provisional* gate by the identical lexicographic rule, so the
    # learner-induced states it collects are the ones a gated deployment would actually
    # visit rather than the ones an ungated one would. It is refit from scratch for round
    # 1. No validation or offline-test trajectory participates in either fit.
    if True:
        calibration_ids = [r["episode_id"]
                           for r in partition["partitions"]["reference_calibration"]]
        calibration_rows = (
            load_expert(args.expert_paired_dir, calibration_ids)
            + load_on_policy(labelling, "act_only_on_policy", set(calibration_ids))
            + load_on_policy(labelling, "oracle_on_policy", set(calibration_ids)))
        calibration = stack(calibration_rows)
        calibration["teacher_evaluable_row"] = np.concatenate(
            [np.full(len(r["qpos"]), bool(r["privileged_teacher_active"].any()), dtype=bool)
             for r in calibration_rows])
        calibration_prediction = predict(reference, calibration)
        calibration_dq = (calibration["current_head"] - calibration_prediction).astype(
            np.float32)
        fit = calibrate(calibration_dq, calibration)
        report["calibration"] = fit
        report["calibration_trajectories"] = calibration_ids
        report["calibration_is_provisional"] = args.round == "0"
        if not fit["feasible"] and args.round == "0":
            # The single permitted aggregation round still has to be collected. With no
            # feasible quiet threshold, the round-0 model drives ungated but inside the
            # oracle-derived output bound, so the learner-induced states are the ones an
            # unfiltered learner reaches and no trajectory is driven by an unbounded
            # magnitude. Nothing is tuned and no validation or test data is touched; this
            # gate is for DATA COLLECTION ONLY and is never evaluated.
            report["tau"] = 0.0
            report["rho_max"] = fit["rho_max"]
            report["round0_collection_gate"] = {
                "tau": 0.0, "rho_max": fit["rho_max"],
                "why": ("no tau satisfies the lexicographic direction rule on the "
                        "calibration trajectories for the round-0 model, so the learner "
                        "round is collected ungated inside the oracle-derived output "
                        "bound"),
                "data_collection_only": True, "evaluated": False}
        elif not fit["feasible"]:
            report["decision"] = "ON_POLICY_REFERENCE_OFFLINE_INVALID"
            report["reason"] = "no tau satisfies the lexicographic rule on calibration"
        else:
            report["tau"] = fit["tau"]
            report["rho_max"] = fit["rho_max"]
        if fit["feasible"] and args.round == "1":
            gate = SupportEnvelopeGate(tau=fit["tau"], rho_max=fit["rho_max"])
            evaluations, gate_reports = {}, {}
            for set_name, ids in (("reference_validation", validation_ids),
                                  ("offline_reference_test",
                                   [r["episode_id"] for r in
                                    partition["partitions"]["offline_reference_test"]])):
                rows = (load_expert(args.expert_paired_dir, ids)
                        + load_on_policy(labelling, "act_only_on_policy", set(ids))
                        + load_on_policy(labelling, "oracle_on_policy", set(ids)))
                pooled = stack(rows)
                prediction = predict(reference, pooled)
                pooled_dq = (pooled["current_head"] - prediction).astype(np.float32)
                evaluations[set_name] = evaluate(set_name, pooled_dq, pooled, gate)
                gate_reports[set_name] = gates_for(evaluations[set_name])
                by_distribution = {}
                for distribution, loader in (
                    ("expert", lambda i=ids: load_expert(args.expert_paired_dir, i)),
                    ("act_only_on_policy",
                     lambda i=ids: load_on_policy(labelling, "act_only_on_policy", set(i))),
                    ("oracle_on_policy",
                     lambda i=ids: load_on_policy(labelling, "oracle_on_policy", set(i))),
                ):
                    subset = stack(loader())
                    subset_dq = (subset["current_head"]
                                 - predict(reference, subset)).astype(np.float32)
                    by_distribution[distribution] = evaluate(distribution, subset_dq,
                                                             subset, gate)
                evaluations[set_name]["by_distribution"] = by_distribution
            report["evaluations"] = evaluations
            report["gates"] = gate_reports
            report["all_offline_gates_passed"] = all(
                g["passed"] for gates in gate_reports.values() for g in gates.values())
            report["decision"] = ("OFFLINE_QUALIFIED" if report["all_offline_gates_passed"]
                                  else "ON_POLICY_REFERENCE_OFFLINE_INVALID")

    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    if args.deployment_manifest and report.get("tau") is not None:
        stack_contract = json.loads(
            (ROOT / "configs/hybrid_safety_stack_v1.json").read_text())
        names = stack_contract["sensor_contract"]["ordered_names"]
        deployment = {
            "schema": "hybrid_obstacle_on_policy_reference_manifest_v2",
            "reference_type": MLP_REFERENCE_ID,
            "label": label,
            "configuration": {"architecture": "Linear196-256 SiLU 256-256 SiLU 256-128 "
                                              "SiLU 128-7",
                              "parameters": training_report["parameters"],
                              "feature_fields": list(FEATURE_FIELDS[MLP_REFERENCE_ID]),
                              "feature_width": FEATURE_WIDTHS[MLP_REFERENCE_ID]},
            "artifact_path": str(artifact),
            "artifact_file_sha256": report["artifact_file_sha256"],
            "model_digest": report["model_digest"],
            "input_statistics_sha256": report["input_statistics_sha256"],
            "feature_width": FEATURE_WIDTHS[MLP_REFERENCE_ID],
            "runtime_inputs": list(FEATURE_FIELDS[MLP_REFERENCE_ID]),
            "privileged_inputs": [],
            "tau": report["tau"],
            "rho_max": report["rho_max"],
            "gate_type": "SupportEnvelopeGate",
            "gate_rule": ("silent below tau; above it the direction is preserved and the "
                          "norm capped at rho_max. No global-minimum-depth condition."),
            "calibration_trajectories": report["calibration_trajectories"],
            "d_act": 0.18,
            "sensor_order_sha256": hashlib.sha256(
                json.dumps(names, separators=(",", ":"),
                           ensure_ascii=True).encode("ascii")).hexdigest(),
            "act_checkpoint_sha256":
                "dd7cd108a64ce10e5aab21b525dc06190f54d4e5fe446f65715b6852c49e7d36",
            "safety_model_sha256":
                "1fb2fc2b6023e64d2b9cbcf67fd5a24402968ec6f902c1e8a8595690396e7405",
            "offsamples": 4,
            "controller_constants": {"gain": 4.0, "decay": 2.2, "ema": 0.75,
                                     "max_dev": 0.35, "dt": 0.066,
                                     "label_scale": 11.359346389770508},
            "training_report_sha256": report["report_sha256"],
            "partition_sha256": partition["partition_sha256"],
            "frozen_before_live_execution": True,
        }
        deployment["manifest_sha256"] = canonical_hash(deployment)
        Path(args.deployment_manifest).resolve().write_text(
            json.dumps(deployment, indent=2, sort_keys=True, default=str) + "\n")

    print(f"{label}: best epoch {report['best_epoch']} "
          f"val {report['best_validation_loss']:.6f} reload_ok={strict_ok}")
    for name, info in provenance.items():
        print(f"  {name:<22} {info['rows']:>3} rows, "
              f"{info['total_frames_available']:>6} frames available")
    fit = report["calibration"]
    print(f"  calibration feasible={fit['feasible']} tau={fit.get('tau')} "
          f"rho_max={fit.get('rho_max')} "
          f"provisional={report.get('calibration_is_provisional')}")
    if args.round == "1":
        for set_name, gates in (report.get("gates") or {}).items():
            print(f"  {set_name}:")
            for key, gate in gates.items():
                print(f"     [{'PASS' if gate['passed'] else 'FAIL'}] {key} = "
                      f"{gate['value']}")
        print(f"  -> {report.get('decision')}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
