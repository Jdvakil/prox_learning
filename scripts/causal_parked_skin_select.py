#!/usr/bin/env python3
"""Bounded validation-only configuration selection for the parked-skin reference.

Handoff step 9. Exactly six seed-0 candidates, varying only loss-component weights, the
number of cross-sensor blocks, and hidden width. Input fields, causal history length,
partitions, target definition, output constraints and the SafetyHead are identical across
all six.

Selection reads the validation partition only. The offline-test partition is not loaded by
this script at all -- not to compute a "just curious" number, not for logging. The lock is
structural rather than a matter of discipline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin.data import load_partition
from causal_parked_skin.engine import (
    TrainConfig,
    evaluate,
    set_sensor_names,
    strip_arrays,
    train,
)
from causal_parked_skin.losses import LossWeights
from causal_parked_skin.model import (
    BASELINE_FULL,
    BASELINE_ZERO,
    FrozenSafetyHead,
)

# The six candidates. Fixed here, in source, before any of them runs.
CANDIDATES = {
    "c1_balanced": {
        "hidden": 192, "blocks": 2,
        "weights": {"changed_mask": 1.0, "active_delta": 10.0, "all_valid_field": 10.0,
                    "head_consistency": 20.0, "quiet": 20.0}},
    "c2_quiet_heavy": {
        "hidden": 192, "blocks": 2,
        "weights": {"changed_mask": 1.0, "active_delta": 10.0, "all_valid_field": 10.0,
                    "head_consistency": 20.0, "quiet": 50.0}},
    "c3_active_heavy": {
        "hidden": 192, "blocks": 2,
        "weights": {"changed_mask": 2.0, "active_delta": 30.0, "all_valid_field": 5.0,
                    "head_consistency": 20.0, "quiet": 20.0}},
    "c4_one_block": {
        "hidden": 192, "blocks": 1,
        "weights": {"changed_mask": 1.0, "active_delta": 10.0, "all_valid_field": 10.0,
                    "head_consistency": 20.0, "quiet": 20.0}},
    "c5_wide": {
        "hidden": 256, "blocks": 2,
        "weights": {"changed_mask": 1.0, "active_delta": 10.0, "all_valid_field": 10.0,
                    "head_consistency": 20.0, "quiet": 20.0}},
    "c6_narrow": {
        "hidden": 128, "blocks": 2,
        "weights": {"changed_mask": 1.0, "active_delta": 10.0, "all_valid_field": 10.0,
                    "head_consistency": 20.0, "quiet": 20.0}},
}

# Diagnostic runs that preceded the candidate budget, disclosed rather than hidden. They
# established that the BCE positive weight was mis-scaled (uncapped at ~1200, which forced
# the mask head to fire field-wide); that is a defect fix, not a selection decision.
PRE_CANDIDATE_DIAGNOSTICS = [
    {"tag": "smoke", "epochs": 8, "finding":
     "uncapped pos_weight -> val head MAE 0.2739, far worse than the zero baseline"},
    {"tag": "D1", "epochs": 12, "finding":
     "pos_weight capped at 32 -> val head MAE 0.0364, first result beating zero"},
    {"tag": "D2", "epochs": 12, "finding":
     "head/quiet-dominant weights -> val head MAE 0.0407, no better than zero"},
    {"tag": "D_long", "epochs": 23, "finding":
     "D1 weights trained to early stop -> val head MAE 0.0261; confirmed the schedule "
     "rather than the weights was the binding limit in the 12-epoch probes"},
]


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
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

    train_partition = load_partition(args.cache, "reference_train")
    validation = load_partition(args.cache, "reference_validation")

    zero = strip_arrays(evaluate(None, validation, head, device, BASELINE_ZERO,
                                 collect_fields=True))
    zero_mae = zero["head"]["differential_mae"]
    print(f"ZERO_DIFFERENTIAL validation head MAE {zero_mae:.6f}")

    results = {}
    started = time.time()
    for name, spec in CANDIDATES.items():
        config = TrainConfig(
            variant=BASELINE_FULL, hidden=spec["hidden"], blocks=spec["blocks"],
            max_epochs=args.max_epochs, patience=args.patience,
            batches_per_epoch=args.batches_per_epoch, seed=0,
            weights=LossWeights(**spec["weights"]))
        print(f"\n=== {name} hidden={spec['hidden']} blocks={spec['blocks']} ===")
        record = train(config, train_partition, validation, head, device,
                       args.checkpoint_root / name, log_every=20)
        results[name] = {
            "spec": spec,
            "config": record["config"],
            "config_hash": record["config_hash"],
            "parameter_count": record["parameter_count"],
            "best_epoch": record["best_epoch"],
            "epochs_run": record["epochs_run"],
            "validation_head_mae": record["best_validation_head_mae"],
            "improvement_over_zero": 1.0 - record["best_validation_head_mae"] / zero_mae,
            "best_checkpoint": record["best_checkpoint"],
            "best_checkpoint_sha256": record["best_checkpoint_sha256"],
            "wall_seconds": record["wall_seconds"],
            "history": record["history"],
            "sampler": record["sampler"],
        }
        print(f"  -> val head MAE {record['best_validation_head_mae']:.6f} "
              f"({results[name]['improvement_over_zero'] * 100:.1f}% better than zero) "
              f"params={record['parameter_count']:,}")

    ranked = sorted(results, key=lambda k: results[k]["validation_head_mae"])
    selected = ranked[0]

    report = {
        "schema": "causal_parked_skin_selection_v1",
        "selection_partition": "reference_validation",
        "offline_test_loaded": False,
        "candidate_budget": 6,
        "candidates_run": len(results),
        "pre_candidate_diagnostics": PRE_CANDIDATE_DIAGNOSTICS,
        "zero_baseline_validation": zero,
        "results": results,
        "ranking": ranked,
        "selected": selected,
        "selected_config": results[selected]["config"],
        "selected_config_hash": results[selected]["config_hash"],
        "varied_axes": ["loss_component_weights", "cross_sensor_blocks", "hidden_width"],
        "fixed_axes": ["input_fields", "causal_history_length", "partitions",
                       "target_definition", "output_constraints", "safety_head",
                       "checkpoint_selection_rule", "max_epochs", "patience"],
        "wall_seconds": time.time() - started,
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print("\nranking (validation head MAE):")
    for name in ranked:
        print(f"  {name:<18} {results[name]['validation_head_mae']:.6f}  "
              f"params={results[name]['parameter_count']:>9,}  "
              f"ep={results[name]['best_epoch']}")
    print(f"selected: {selected}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
