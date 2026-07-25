#!/usr/bin/env python3
"""Train, evaluate and select the deployable posture-conditioned reference.

Handoff steps 4-8. Two predeclared candidates are fitted on the 80 canonical training
trajectories, evaluated on the 20 held-out validation trajectories, scored against the
predeclared lexicographic rule, and the winner is frozen into a deployment manifest.

No architecture search, no hyperparameter sweep, no retraining after live results. All
input normalization statistics come from the training trajectories only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT / "scripts"), str(ROOT / "submodules" / "act")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from deployable_reference import (
    D_ACT,
    FEATURE_BUILDERS,
    FEATURE_FIELDS,
    FEATURE_WIDTHS,
    KNN_K,
    KNN_REFERENCE_ID,
    MLP_REFERENCE_ID,
    PostureKnnReference,
    PostureSkinMlpReference,
    Standardizer,
    build_mlp,
    canonical_hash,
)

SEED = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256
MAX_EPOCHS = 150
TAU_PERCENTILE = 99.5
FAR_DEPTH_M = 0.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
def load_split(dataset_dir: Path, split_manifest: dict) -> dict[str, dict[str, Any]]:
    """Load the paired dataset, keyed by split. Splitting is by trajectory, never frame."""
    by_split: dict[str, dict[str, list]] = {"train": {}, "validation": {}}
    trajectories: dict[str, list[str]] = {"train": [], "validation": []}
    for episode in sorted(split_manifest["episodes"], key=lambda e: e["split_rank"]):
        path = dataset_dir / f"{episode['episode_id']}.npz"
        if not path.is_file():
            raise SystemExit(f"paired example file missing: {path}")
        blob = np.load(path, allow_pickle=False)
        split = episode["split"]
        trajectories[split].append(episode["episode_id"])
        store = by_split[split]
        for key in blob.files:
            store.setdefault(key, []).append(blob[key])
        store.setdefault("trajectory_index", []).append(
            np.full(len(blob["timestep"]), len(trajectories[split]) - 1, dtype=np.int64))
        store.setdefault("hazard_present_row", []).append(
            np.full(len(blob["timestep"]), bool(episode["hazard_present"]), dtype=bool))
    out = {}
    for split, store in by_split.items():
        out[split] = {key: np.concatenate(values, axis=0) for key, values in store.items()}
        out[split]["trajectory_ids"] = trajectories[split]
    return out


def train_mlp(features: np.ndarray, targets: np.ndarray,
              validation_features: np.ndarray, validation_targets: np.ndarray,
              standardizer: Standardizer, device: str) -> tuple[Any, dict[str, Any]]:
    """The fixed training recipe. Seeded, no sweep, minimum-validation checkpoint kept."""
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_mlp(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss()

    x_train = torch.from_numpy(standardizer(features)).float().to(device)
    y_train = torch.from_numpy(np.asarray(targets, dtype=np.float32)).to(device)
    x_val = torch.from_numpy(standardizer(validation_features)).float().to(device)
    y_val = torch.from_numpy(np.asarray(validation_targets, dtype=np.float32)).to(device)

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    best_state, best_loss, best_epoch = None, float("inf"), -1
    history = []
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(x_train), generator=generator).to(device)
        total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(x_val), y_val))
        history.append({"epoch": epoch, "train": total / len(order),
                        "validation": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_validation_loss": best_loss,
                   "epochs": MAX_EPOCHS, "history": history,
                   "parameters": sum(p.numel() for p in model.parameters())}


# --------------------------------------------------------------------------- #
def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 1e-12) & (nb > 1e-12)
    out = np.full(len(a), np.nan)
    out[ok] = (a[ok] * b[ok]).sum(axis=1) / (na[ok] * nb[ok])
    return out


def summarise(values) -> dict[str, Any]:
    clean = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    if not clean.size:
        return {"n": 0, "median": None, "mean": None, "fraction_positive": None}
    return {"n": int(clean.size), "median": float(np.median(clean)),
            "mean": float(np.mean(clean)),
            "fraction_positive": float(np.mean(clean > 0))}


def evaluate(name: str, predicted_parked: np.ndarray, data: dict[str, Any],
             tau: float | None) -> dict[str, Any]:
    """Full offline metric set for one candidate on the validation split."""
    current = data["current_head"]
    oracle = data["oracle_dq"]
    parked = data["parked_head"]
    predicted_dq = (current - predicted_parked).astype(np.float32)
    norms = np.linalg.norm(predicted_dq, axis=1)

    teacher_active = data["teacher_active"].astype(bool)
    hazard_absent = ~data["hazard_present_row"].astype(bool)
    oracle_active = np.linalg.norm(oracle, axis=1) > 1e-9
    supported = data["minimum_depth"] < D_ACT
    far = data["minimum_depth"] > FAR_DEPTH_M
    grasped = data["gripper_state"][:, 0] < np.median(data["gripper_state"][:, 0])

    def mae(mask):
        return float(np.mean(np.abs(predicted_dq[mask] - oracle[mask]))) if mask.any() \
            else None

    activation = (norms > tau) & supported if tau is not None else np.zeros(
        len(norms), dtype=bool)

    report = {
        "candidate": name,
        "frames": len(current),
        "active_range": {
            "teacher_active_frames": int(teacher_active.sum()),
            "oracle_nonzero_frames": int(oracle_active.sum()),
            "supported_frames": int(supported.sum()),
            "oracle_cosine_on_teacher_active":
                summarise(cosine(predicted_dq[teacher_active], oracle[teacher_active])),
            "oracle_cosine_on_oracle_nonzero":
                summarise(cosine(predicted_dq[oracle_active], oracle[oracle_active])),
            "analytic_teacher_cosine_on_teacher_active":
                summarise(cosine(predicted_dq[teacher_active],
                                 data["teacher_dq"][teacher_active])),
            "differential_mae_teacher_active": mae(teacher_active),
            "differential_mae_oracle_nonzero": mae(oracle_active),
            "differential_mae_all": mae(np.ones(len(current), dtype=bool)),
            "differential_norm_error_teacher_active": (
                float(np.mean(np.abs(norms[teacher_active]
                                     - np.linalg.norm(oracle[teacher_active], axis=1))))
                if teacher_active.any() else None),
            "jointwise_mae_teacher_active": (
                np.mean(np.abs(predicted_dq[teacher_active] - oracle[teacher_active]),
                        axis=0).tolist() if teacher_active.any() else None),
        },
        "quietness": {
            "tau": tau,
            "hazard_absent_frames": int(hazard_absent.sum()),
            "hazard_absent_false_activation_rate":
                float(np.mean(activation[hazard_absent])) if hazard_absent.any() else None,
            "far_frames_over_0p25m": int(far.sum()),
            "far_false_activation_rate":
                float(np.mean(activation[far])) if far.any() else None,
            "quiet_output_rms_hazard_absent":
                float(np.sqrt(np.mean(norms[hazard_absent] ** 2)))
                if hazard_absent.any() else None,
            "quiet_output_rms_oracle_zero":
                float(np.sqrt(np.mean(norms[~oracle_active] ** 2)))
                if (~oracle_active).any() else None,
            "correction_norm_p95": float(np.percentile(norms, 95)),
            "correction_norm_p99": float(np.percentile(norms, 99)),
            "grasp_contact_false_activation_rate":
                float(np.mean(activation[grasped & ~oracle_active]))
                if (grasped & ~oracle_active).any() else None,
        },
        "reference_quality": {
            "parked_head_mae": float(np.mean(np.abs(predicted_parked - parked))),
            "parked_head_jointwise_mae":
                np.mean(np.abs(predicted_parked - parked), axis=0).tolist(),
            "cancellation_error_when_oracle_zero": (
                float(np.mean(np.abs(predicted_dq[~oracle_active])))
                if (~oracle_active).any() else None),
            "parked_head_mae_grasped": float(np.mean(np.abs(
                predicted_parked[grasped] - parked[grasped]))) if grasped.any() else None,
            "parked_head_mae_ungrasped": float(np.mean(np.abs(
                predicted_parked[~grasped] - parked[~grasped]))) if (~grasped).any() else None,
            "parked_head_mae_hazard_present": float(np.mean(np.abs(
                predicted_parked[~hazard_absent] - parked[~hazard_absent]))),
            "parked_head_mae_hazard_absent": float(np.mean(np.abs(
                predicted_parked[hazard_absent] - parked[hazard_absent])))
                if hazard_absent.any() else None,
        },
    }
    return report


def derive_tau(predicted_parked: np.ndarray, data: dict[str, Any]) -> float:
    hazard_absent = ~data["hazard_present_row"].astype(bool)
    norms = np.linalg.norm(data["current_head"] - predicted_parked, axis=1)
    if not hazard_absent.any():
        raise SystemExit("no hazard-absent validation frames; tau cannot be derived")
    return float(np.percentile(norms[hazard_absent], TAU_PERCENTILE))


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--dataset-manifest", required=True, type=Path)
    ap.add_argument("--split-manifest", required=True, type=Path)
    ap.add_argument("--artifact-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--deployment-manifest", required=True, type=Path)
    args = ap.parse_args()
    for name in ("dataset_dir", "dataset_manifest", "split_manifest", "artifact_dir",
                 "out", "deployment_manifest"):
        setattr(args, name, Path(getattr(args, name)).resolve())

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_manifest = json.loads(args.split_manifest.read_text())
    dataset_manifest = json.loads(args.dataset_manifest.read_text())
    data = load_split(args.dataset_dir, split_manifest)
    train, validation = data["train"], data["validation"]
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"train frames {len(train['timestep'])} over {len(train['trajectory_ids'])} "
          f"trajectories | validation frames {len(validation['timestep'])} over "
          f"{len(validation['trajectory_ids'])}")

    candidates: dict[str, dict[str, Any]] = {}

    # ---------------- candidate A: posture KNN ---------------------------- #
    started = time.time()
    knn_train = FEATURE_BUILDERS[KNN_REFERENCE_ID](train)
    knn_validation = FEATURE_BUILDERS[KNN_REFERENCE_ID](validation)
    knn = PostureKnnReference().fit(knn_train, train["parked_head"],
                                    train["trajectory_index"], train["timestep"])
    knn_prediction = knn.predict(knn_validation)
    knn_path = args.artifact_dir / "posture_knn_reference_v1.npz"
    knn.save(knn_path)
    candidates[KNN_REFERENCE_ID] = {
        "reference_id": KNN_REFERENCE_ID, "prediction": knn_prediction,
        "artifact": knn_path, "model_digest": knn.digest(),
        "input_statistics_sha256": knn.standardizer.digest(),
        "configuration": {"k": KNN_K, "weighting": "inverse distance",
                          "tie_break": "(distance, trajectory_index, timestep) lexsort",
                          "bank_frames": len(knn_train),
                          "feature_fields": list(FEATURE_FIELDS[KNN_REFERENCE_ID]),
                          "feature_width": knn_train.shape[1]},
        "training": {"seconds": round(time.time() - started, 1)},
    }
    print(f"[A] KNN bank {len(knn_train)} frames, {round(time.time()-started,1)}s")

    # ---------------- candidate B: posture + skin MLP ---------------------- #
    started = time.time()
    mlp_train = FEATURE_BUILDERS[MLP_REFERENCE_ID](train)
    mlp_validation = FEATURE_BUILDERS[MLP_REFERENCE_ID](validation)
    standardizer = Standardizer.fit(mlp_train)
    model, training_report = train_mlp(mlp_train, train["parked_head"],
                                       mlp_validation, validation["parked_head"],
                                       standardizer, device)
    mlp = PostureSkinMlpReference(standardizer=standardizer, model=model, device=device)
    mlp_prediction = mlp.predict(mlp_validation)
    mlp_path = args.artifact_dir / "posture_skin_mlp_reference_v1.pt"
    mlp.metadata = {"seed": SEED, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                    "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
                    "loss": "SmoothL1", "optimizer": "AdamW",
                    "feature_fields": list(FEATURE_FIELDS[MLP_REFERENCE_ID])}
    mlp.save(mlp_path)
    reloaded = PostureSkinMlpReference.load(mlp_path, device=device)
    strict_ok = bool(np.allclose(reloaded.predict(mlp_validation), mlp_prediction,
                                 atol=0, rtol=0))
    candidates[MLP_REFERENCE_ID] = {
        "reference_id": MLP_REFERENCE_ID, "prediction": mlp_prediction,
        "artifact": mlp_path, "model_digest": mlp.digest(),
        "input_statistics_sha256": standardizer.digest(),
        "configuration": {"architecture": "Linear196-256 SiLU 256-256 SiLU 256-128 "
                                          "SiLU 128-7",
                          "parameters": training_report["parameters"],
                          "under_250k_parameters": training_report["parameters"] < 250_000,
                          "dropout": False, "batch_norm": False,
                          "feature_fields": list(FEATURE_FIELDS[MLP_REFERENCE_ID]),
                          "feature_width": mlp_train.shape[1]},
        "training": {k: v for k, v in training_report.items() if k != "history"},
        "training_history": training_report["history"],
        "strict_reload_bitwise_identical": strict_ok,
        "seconds": round(time.time() - started, 1),
    }
    print(f"[B] MLP best epoch {training_report['best_epoch']} "
          f"val {training_report['best_validation_loss']:.6f} "
          f"params {training_report['parameters']} reload_ok={strict_ok} "
          f"{round(time.time()-started,1)}s")

    # ---------------- baselines for gate 4 -------------------------------- #
    zeros = np.zeros_like(validation["parked_head"])
    first_live = np.zeros_like(validation["parked_head"])
    for index in np.unique(validation["trajectory_index"]):
        mask = validation["trajectory_index"] == index
        first_live[mask] = validation["current_head"][mask][0]
    baselines = {
        "raw_safety_head": evaluate("raw_safety_head", zeros, validation, None),
        "first_live_skin": evaluate("first_live_skin", first_live, validation, None),
    }

    # ---------------- tau, then metrics ------------------------------------ #
    for entry in candidates.values():
        entry["tau"] = derive_tau(entry["prediction"], validation)
        entry["metrics"] = evaluate(entry["reference_id"], entry["prediction"],
                                    validation, entry["tau"])

    # ---------------- predeclared selection rule --------------------------- #
    def gates_for(entry) -> dict[str, Any]:
        metrics = entry["metrics"]
        active = metrics["active_range"]["oracle_cosine_on_teacher_active"]
        analytic = metrics["active_range"]["analytic_teacher_cosine_on_teacher_active"]
        quiet = metrics["quietness"]
        mae = metrics["active_range"]["differential_mae_teacher_active"]
        comparisons = {
            "raw_safety_head":
                baselines["raw_safety_head"]["active_range"]["differential_mae_teacher_active"],
            "first_live_skin":
                baselines["first_live_skin"]["active_range"]["differential_mae_teacher_active"],
        }
        if entry["reference_id"] == MLP_REFERENCE_ID:
            comparisons["posture_knn"] = candidates[KNN_REFERENCE_ID]["metrics"][
                "active_range"]["differential_mae_teacher_active"]
        gate1 = {
            "rule": "median oracle cosine >= 0.70 and positive fraction >= 80% "
                    "on teacher-active validation frames",
            "median": active["median"], "fraction_positive": active["fraction_positive"],
            "n": active["n"],
            "evaluable": active["n"] > 0,
            "passed": bool(active["n"] > 0 and active["median"] is not None
                           and active["median"] >= 0.70
                           and active["fraction_positive"] >= 0.80),
        }
        gate2 = {
            "rule": "median analytic-teacher cosine >= 0.60 and positive fraction >= 75%",
            "median": analytic["median"],
            "fraction_positive": analytic["fraction_positive"], "n": analytic["n"],
            "evaluable": analytic["n"] > 0,
            "passed": bool(analytic["n"] > 0 and analytic["median"] is not None
                           and analytic["median"] >= 0.60
                           and analytic["fraction_positive"] >= 0.75),
        }
        gate3 = {
            "rule": "hazard-absent false activation <= 1% and far-frame <= 2%",
            "hazard_absent": quiet["hazard_absent_false_activation_rate"],
            "far": quiet["far_false_activation_rate"],
            "passed": bool((quiet["hazard_absent_false_activation_rate"] or 0.0) <= 0.01
                           and (quiet["far_false_activation_rate"] or 0.0) <= 0.02),
        }
        gate4 = {
            "rule": "active-frame differential MAE below every listed baseline",
            "candidate_mae": mae, "baselines": comparisons,
            "passed": bool(mae is not None and all(
                other is not None and mae < other for other in comparisons.values())),
        }
        return {"gate1_oracle_direction": gate1, "gate2_analytic_direction": gate2,
                "gate3_quietness": gate3, "gate4_beats_baselines": gate4,
                "all_passed": all(g["passed"] for g in (gate1, gate2, gate3, gate4))}

    for entry in candidates.values():
        entry["gates"] = gates_for(entry)

    passing = [e for e in candidates.values() if e["gates"]["all_passed"]]
    if len(passing) > 1:
        passing.sort(key=lambda e: (
            e["metrics"]["active_range"]["differential_mae_teacher_active"],
            e["metrics"]["quietness"]["hazard_absent_false_activation_rate"] or 0.0,
            0 if e["reference_id"] == KNN_REFERENCE_ID else 1))
    selected = passing[0] if passing else None
    decision = ("SELECTED" if selected else "DEPLOYABLE_REFERENCE_OFFLINE_INVALID")

    report = {
        "schema": "hybrid_obstacle_deployable_reference_selection_v1",
        "seed": SEED,
        "device": device,
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "split_manifest_sha256": split_manifest["split_manifest_sha256"],
        "train_trajectories": train["trajectory_ids"],
        "validation_trajectories": validation["trajectory_ids"],
        "train_frames": len(train["timestep"]),
        "validation_frames": len(validation["timestep"]),
        "normalization_from_training_only": True,
        "split_level": "trajectory",
        "tau_percentile": TAU_PERCENTILE,
        "d_act": D_ACT,
        "far_depth_m": FAR_DEPTH_M,
        "baselines": baselines,
        "candidates": {k: {kk: vv for kk, vv in v.items()
                           if kk not in ("prediction", "artifact", "training_history")}
                       for k, v in candidates.items()},
        "mlp_training_history": candidates[MLP_REFERENCE_ID]["training_history"],
        "selected": selected["reference_id"] if selected else None,
        "selection_outcome": decision,
        "selection_rule": ("lexicographic gates 1-4; ties on active-frame differential "
                           "MAE, then hazard-absent false activation, then in favour of "
                           "KNN as the simpler model"),
        "hyperparameter_sweep_performed": False,
        "architecture_search_performed": False,
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    if selected:
        stack = json.loads((ROOT / "configs/hybrid_safety_stack_v1.json").read_text())
        names = stack["sensor_contract"]["ordered_names"]
        deployment = {
            "schema": "hybrid_obstacle_deployable_reference_manifest_v1",
            "reference_type": selected["reference_id"],
            "configuration": selected["configuration"],
            "artifact_path": str(selected["artifact"]),
            "artifact_file_sha256": sha256_file(selected["artifact"]),
            "model_digest": selected["model_digest"],
            "input_statistics_sha256": selected["input_statistics_sha256"],
            "feature_width": FEATURE_WIDTHS[selected["reference_id"]],
            "runtime_inputs": list(FEATURE_FIELDS[selected["reference_id"]]),
            "privileged_inputs": [],
            "tau": selected["tau"],
            "tau_rule": (f"{TAU_PERCENTILE}th percentile of predicted oracle differential "
                         "norm on hazard-absent validation frames, frozen before live "
                         "execution"),
            "d_act": D_ACT,
            "training_data_sha256": dataset_manifest["train_tree_sha256"],
            "validation_data_sha256": dataset_manifest["validation_tree_sha256"],
            "paired_dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
            "sensor_order_sha256": hashlib.sha256(
                json.dumps(names, separators=(",", ":"),
                           ensure_ascii=True).encode("ascii")).hexdigest(),
            "act_checkpoint_sha256":
                "dd7cd108a64ce10e5aab21b525dc06190f54d4e5fe446f65715b6852c49e7d36",
            "dataset_stats_sha256":
                "c8119b904bfc80d66e3d33825722fcf9bb8bf3433c956dc09c27e6517d7c4ae2",
            "safety_model_sha256":
                "1fb2fc2b6023e64d2b9cbcf67fd5a24402968ec6f902c1e8a8595690396e7405",
            "offsamples": 4,
            "controller_constants": {"gain": 4.0, "decay": 2.2, "ema": 0.75,
                                     "max_dev": 0.35, "dt": 0.066,
                                     "label_scale": 11.359346389770508},
            "source_commits": {
                "root": "4193b776640886c1bddc6be5adc7bdaf35855643",
                "act": "709a22de62ac0e8c4640b75eb348416d6e29013d",
                "molmospaces": "678f2eb4a0ac0d9e3d14e555aaac0e099089b9a5"},
            "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "device": device},
            "selection_report_sha256": report["report_sha256"],
            "frozen_before_live_execution": True,
        }
        deployment["manifest_sha256"] = canonical_hash(deployment)
        args.deployment_manifest.write_text(
            json.dumps(deployment, indent=2, sort_keys=True, default=str) + "\n")

    for entry in candidates.values():
        gates = entry["gates"]
        print(f"\n{entry['reference_id']}  tau={entry['tau']:.5f}")
        for key, gate in gates.items():
            if key == "all_passed":
                continue
            print(f"   [{'PASS' if gate['passed'] else 'FAIL'}] {key}: "
                  + ", ".join(f"{k}={v}" for k, v in gate.items()
                              if k not in ("rule", "passed", "baselines")))
    print(f"\nselected: {report['selected']} -> {decision}")
    print(f"wrote {args.out}")
    return 0 if selected else 4


if __name__ == "__main__":
    raise SystemExit(main())
