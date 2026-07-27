#!/usr/bin/env python3
"""Final three-seed training, calibration and single offline-test evaluation.

Handoff steps 10-13. The configuration is read from the frozen selection report and is not
modified here. Order of operations matters and is enforced by the script rather than by
discipline:

1. train seeds 0, 1, 2 for the primary model and for each learned baseline;
2. calibrate the activation threshold on the calibration partition only;
3. evaluate validation and calibration;
4. only then load the offline-test partition and score it once.

The offline-test arrays are not opened until every checkpoint is final and every threshold
is fixed, so no hyperparameter, epoch or threshold can have been chosen with knowledge of a
test number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin.data import load_partition
from causal_parked_skin.engine import (
    TrainConfig,
    evaluate,
    load_checkpoint,
    make_batch,
    set_sensor_names,
    strip_arrays,
    train,
)
from causal_parked_skin.losses import LossWeights
from causal_parked_skin.model import (
    BASELINE_CURRENT,
    BASELINE_FULL,
    BASELINE_QPOS,
    BASELINE_ZERO,
    FrozenSafetyHead,
)

SEEDS = (0, 1, 2)
LEARNED_VARIANTS = (BASELINE_FULL, BASELINE_CURRENT, BASELINE_QPOS)

# Frozen before offline test is opened, per handoff step 10.
FALSE_POSITIVE_TARGET = 0.01


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def calibrate_threshold(result: dict) -> dict:
    """Threshold on ||predicted_oracle_dq|| giving at most 1% firing on oracle-zero frames.

    The quantile is taken over the oracle-zero frames only. Taking it over all frames would
    let a model with many true activations buy itself a lower threshold.
    """
    norms = np.asarray(result["predicted_norm"], dtype=np.float64)
    active = np.asarray(result["oracle_active"], dtype=bool)
    zero_norms = norms[~active]
    if zero_norms.size == 0:
        raise SystemExit("calibration partition has no oracle-zero frames")
    threshold = float(np.quantile(zero_norms, 1.0 - FALSE_POSITIVE_TARGET))
    achieved = float((zero_norms > threshold).mean())
    return {"threshold": threshold,
            "false_positive_target": FALSE_POSITIVE_TARGET,
            "achieved_false_positive_rate_on_calibration": achieved,
            "oracle_zero_frames": int(zero_norms.size),
            "rule": ("quantile(1 - 0.01) of ||predicted_oracle_dq|| over calibration "
                     "oracle-zero frames; frames fire when the norm is strictly above it"),
            "selected_on": "reference_calibration",
            "offline_test_used": False}


def privileged_upper_bound(partition, head, device) -> dict:
    """Feed the true parked field through the frozen head: the best any model could do."""
    import torch

    total = len(partition)
    predicted = np.zeros((total, 7), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, total, 256):
            index = np.arange(start, min(start + 256, total))
            batch = make_batch(partition, index, device)
            current_dq = head(batch["history"][:, -1])
            predicted[index] = (current_dq - head(batch["parked"])).double().cpu().numpy()
    oracle = np.asarray(partition["oracle_dq"], dtype=np.float64)
    return {"differential_mae": float(np.abs(predicted - oracle).mean()),
             "note": ("true parked field through the frozen head; nonzero only by float32 "
                      "rounding, since this is how the stored target was produced")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--selection", required=True, type=Path)
    ap.add_argument("--checkpoint-root", required=True, type=Path)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batches-per-epoch", type=int, default=300)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch

    stack = json.loads(args.stack.read_text())
    set_sensor_names(stack["sensor_contract"]["ordered_names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = FrozenSafetyHead.load(args.safety_dir, device=device)
    if not head.frozen():
        raise SystemExit("SafetyHead is not frozen")
    head_fingerprint = canonical_hash(
        [float(p.double().sum()) for p in head.decoder.parameters()])

    selection = json.loads(args.selection.read_text())
    chosen = selection["selected_config"]
    weights = LossWeights(**chosen["weights"])
    print(f"frozen configuration: {selection['selected']} "
          f"hidden={chosen['hidden']} blocks={chosen['blocks']}")

    train_partition = load_partition(args.cache, "reference_train")
    validation = load_partition(args.cache, "reference_validation")
    calibration = load_partition(args.cache, "reference_calibration")

    started = time.time()
    runs: dict[str, dict] = {}
    for variant in LEARNED_VARIANTS:
        for seed in SEEDS:
            key = f"{variant}__seed{seed}"
            config = TrainConfig(
                variant=variant, hidden=chosen["hidden"], blocks=chosen["blocks"],
                dropout=chosen["dropout"], batch_size=chosen["batch_size"],
                active_fraction=chosen["active_fraction"],
                learning_rate=chosen["learning_rate"],
                weight_decay=chosen["weight_decay"],
                max_epochs=args.max_epochs, patience=args.patience,
                batches_per_epoch=args.batches_per_epoch, seed=seed, weights=weights)
            print(f"\n=== {key} ===")
            record = train(config, train_partition, validation, head, device,
                           args.checkpoint_root / key, log_every=25)
            runs[key] = {"variant": variant, "seed": seed, "training": record}
            print(f"  -> best val head MAE {record['best_validation_head_mae']:.6f} "
                  f"@ epoch {record['best_epoch']} ({record['epochs_run']} run)")

    # ---- calibration, still without touching offline test ------------------------
    print("\n=== calibration ===")
    for key, run in runs.items():
        model, payload = load_checkpoint(Path(run["training"]["best_checkpoint"]), device)
        scored = evaluate(model, calibration, head, device, run["variant"],
                          collect_fields=False)
        run["calibration_threshold"] = calibrate_threshold(scored)
        run["checkpoint_epoch"] = payload["epoch"]
        print(f"  {key:<32} threshold={run['calibration_threshold']['threshold']:.6f} "
              f"fp={run['calibration_threshold']['achieved_false_positive_rate_on_calibration']:.4f}")
        del model

    zero_calibration = evaluate(None, calibration, head, device, BASELINE_ZERO,
                                collect_fields=False)
    zero_threshold = calibrate_threshold(zero_calibration)

    # ---- validation and calibration metrics --------------------------------------
    print("\n=== scoring validation and calibration ===")
    for key, run in runs.items():
        model, _ = load_checkpoint(Path(run["training"]["best_checkpoint"]), device)
        threshold = run["calibration_threshold"]["threshold"]
        run["metrics"] = {
            "reference_validation": strip_arrays(evaluate(
                model, validation, head, device, run["variant"], threshold=threshold)),
            "reference_calibration": strip_arrays(evaluate(
                model, calibration, head, device, run["variant"], threshold=threshold)),
        }
        del model

    baselines: dict[str, dict] = {BASELINE_ZERO: {"metrics": {
        "reference_validation": strip_arrays(evaluate(
            None, validation, head, device, BASELINE_ZERO,
            threshold=zero_threshold["threshold"])),
        "reference_calibration": strip_arrays(evaluate(
            None, calibration, head, device, BASELINE_ZERO,
            threshold=zero_threshold["threshold"])),
    }, "calibration_threshold": zero_threshold}}

    # ================== offline test opens here, exactly once =====================
    print("\n=== opening offline test (first and only read) ===")
    offline = load_partition(args.cache, "offline_reference_test")
    for key, run in runs.items():
        model, _ = load_checkpoint(Path(run["training"]["best_checkpoint"]), device)
        threshold = run["calibration_threshold"]["threshold"]
        run["metrics"]["offline_reference_test"] = strip_arrays(evaluate(
            model, offline, head, device, run["variant"], threshold=threshold))
        mae = run["metrics"]["offline_reference_test"]["head"]["differential_mae"]
        print(f"  {key:<32} offline head MAE {mae:.6f}")
        del model

    baselines[BASELINE_ZERO]["metrics"]["offline_reference_test"] = strip_arrays(
        evaluate(None, offline, head, device, BASELINE_ZERO,
                 threshold=zero_threshold["threshold"]))
    upper_bound = {name: privileged_upper_bound(part, head, device) for name, part in
                   (("reference_validation", validation),
                    ("reference_calibration", calibration),
                    ("offline_reference_test", offline))}

    # ---- checkpoint reload determinism on a fixed batch ---------------------------
    fixed_index = np.arange(128)
    determinism = {}
    for key, run in runs.items():
        model_a, _ = load_checkpoint(Path(run["training"]["best_checkpoint"]), device)
        model_b, _ = load_checkpoint(Path(run["training"]["best_checkpoint"]), device)
        batch = make_batch(offline, fixed_index, device)
        with torch.no_grad():
            a = model_a(batch["history"], batch["history_valid"], batch["state"])["parked"]
            b = model_b(batch["history"], batch["history_valid"], batch["state"])["parked"]
        determinism[key] = {"max_abs_delta": float((a - b).abs().max()),
                            "bitwise_identical": bool(torch.equal(a, b))}
        del model_a, model_b

    head_fingerprint_after = canonical_hash(
        [float(p.double().sum()) for p in head.decoder.parameters()])

    report = {
        "schema": "causal_parked_skin_final_training_v1",
        "selected_candidate": selection["selected"],
        "selected_config": chosen,
        "selected_config_hash": selection["selected_config_hash"],
        "seeds": list(SEEDS),
        "runs": runs,
        "baselines": baselines,
        "privileged_upper_bound": upper_bound,
        "checkpoint_reload_determinism": determinism,
        "safety_head": {
            "directory": str(args.safety_dir),
            "fingerprint_before": head_fingerprint,
            "fingerprint_after": head_fingerprint_after,
            "unchanged": head_fingerprint == head_fingerprint_after,
            "frozen": head.frozen(),
        },
        "offline_test_opened_after_all_training_and_calibration": True,
        "wall_seconds": time.time() - started,
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
