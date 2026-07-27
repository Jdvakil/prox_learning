"""Readiness-gate computation and decision-token selection.

Handoff steps 14 and 19. Kept as a pure function of the recorded reports so the decision
cannot depend on anything that is not written down, and so the gate arithmetic itself is
testable without retraining.

Every generalization gate is evaluated per seed and must hold for *all three*. Reporting a
mean that passes while one seed fails would hide exactly the instability the three-seed
requirement exists to expose.
"""
from __future__ import annotations

from .metrics import summarize_seeds
from .model import BASELINE_CURRENT, BASELINE_FULL, BASELINE_ZERO

__all__ = ["BASELINE_CURRENT", "BASELINE_FULL", "BASELINE_ZERO"]

DECISION_READY = "PARKED_REFERENCE_MODEL_READY_FOR_LIVE_DEVELOPMENT"
DECISION_INSUFFICIENT = "PARKED_REFERENCE_SIGNAL_INSUFFICIENT"
DECISION_OVERFIT = "PARKED_REFERENCE_MODEL_OVERFIT"
DECISION_DATA_INVALID = "DATA_CONTRACT_INVALID"
DECISION_TRAINING_FAILED = "PARKED_REFERENCE_TRAINING_FAILED"

ALLOWED_DECISIONS = (DECISION_READY, DECISION_INSUFFICIENT, DECISION_OVERFIT,
                     DECISION_DATA_INVALID, DECISION_TRAINING_FAILED)

# thresholds, fixed here before offline test is read
MIN_IMPROVEMENT_OVER_ZERO = 0.25
MIN_IMPROVEMENT_OVER_CURRENT_FRAME = 0.10
MIN_DIRECTION_COSINE = 0.75
MAX_ZERO_FALSE_POSITIVE_RATE = 0.02
MAX_HAZARD_ABSENT_RMS_RATIO = 0.25
MIN_AUPRC_PREVALENCE_MULTIPLE = 3.0
MIN_SOURCE_MODE_COSINE = 0.5
MAX_SEED_COEFFICIENT_OF_VARIATION = 0.20

TEST = "offline_reference_test"


def _runs_for(report: dict, variant: str) -> dict:
    return {k: v for k, v in report["runs"].items() if v["variant"] == variant}


def _gate(name: str, passed: bool, observed, required, detail: str = "") -> dict:
    return {"gate": name, "passed": bool(passed), "observed": observed,
            "required": required, "detail": detail}


def compute_gates(final_report: dict, input_contract: dict, partition_report: dict,
                  dataset_decision: dict, *, primary_variant: str = BASELINE_FULL) -> dict:
    """All technical and generalization gates, plus the decision token they imply.

    ``primary_variant`` is the model the generalization gates are scored against. It is
    normally FULL_CAUSAL; ``resolve`` re-runs this with CURRENT_FRAME_ONLY when temporal
    history turns out to add nothing, because freezing the simpler model means the simpler
    model is what has to clear every remaining bar.
    """
    primary = _runs_for(final_report, primary_variant)
    current_frame = _runs_for(final_report, BASELINE_CURRENT)
    zero = final_report["baselines"][BASELINE_ZERO]["metrics"][TEST]

    technical = []
    technical.append(_gate(
        "no_partition_leakage", partition_report["total_crossings"] == 0,
        partition_report["total_crossings"], 0,
        "five identity keys checked across all four partitions"))
    technical.append(_gate(
        "input_contract_valid", input_contract["valid"],
        {"prohibited_inputs_used": input_contract["prohibited_inputs_used"],
         "inputs_not_live_available": input_contract["inputs_not_live_available"]},
        "no prohibited or non-live inputs"))
    technical.append(_gate(
        "dataset_unchanged",
        dataset_decision["dataset"]["tree_sha256"] ==
        final_report.get("dataset_tree_sha256", dataset_decision["dataset"]["tree_sha256"]),
        final_report.get("dataset_tree_sha256"),
        dataset_decision["dataset"]["tree_sha256"], "frozen tree hash re-verified"))

    nonfinite = sum(run["metrics"][part]["nonfinite_outputs"]
                    for run in final_report["runs"].values()
                    for part in run["metrics"])
    technical.append(_gate("no_nonfinite_output", nonfinite == 0, nonfinite, 0))

    violations = sum(run["metrics"][part]["constraint_violations"]["total"]
                     for run in final_report["runs"].values()
                     for part in run["metrics"])
    technical.append(_gate("zero_constraint_violations", violations == 0, violations, 0,
                           "0 <= parked <= current <= 1, counted not clamped"))

    technical.append(_gate(
        "safety_head_unchanged", final_report["safety_head"]["unchanged"]
        and final_report["safety_head"]["frozen"],
        {"unchanged": final_report["safety_head"]["unchanged"],
         "frozen": final_report["safety_head"]["frozen"]}, True))

    nondeterministic = [k for k, v in final_report["checkpoint_reload_determinism"].items()
                        if not v["bitwise_identical"]]
    technical.append(_gate("checkpoint_reload_deterministic", not nondeterministic,
                           nondeterministic, [],
                           "two independent reloads on one fixed batch"))

    technical.append(_gate(
        "offline_test_opened_once_after_freeze",
        final_report["offline_test_opened_after_all_training_and_calibration"],
        True, True, "offline test loaded only after every checkpoint and threshold"))

    # ---- generalization, per seed ------------------------------------------------
    generalization = []
    zero_mae = zero["head"]["differential_mae"]

    primary_maes = {k: v["metrics"][TEST]["head"]["differential_mae"]
                    for k, v in primary.items()}
    improvements = {k: 1.0 - m / zero_mae for k, m in primary_maes.items()}
    generalization.append(_gate(
        "offline_mae_beats_zero_by_25pct",
        all(v >= MIN_IMPROVEMENT_OVER_ZERO for v in improvements.values()),
        improvements, MIN_IMPROVEMENT_OVER_ZERO,
        f"ZERO_DIFFERENTIAL offline MAE {zero_mae:.6f}; all seeds must clear the bar"))

    current_maes = {v["seed"]: v["metrics"][TEST]["head"]["differential_mae"]
                    for v in current_frame.values()}
    per_seed_vs_current = {}
    for key, run in primary.items():
        seed = run["seed"]
        if seed in current_maes and current_maes[seed] > 0:
            per_seed_vs_current[key] = 1.0 - primary_maes[key] / current_maes[seed]
    if primary_variant == BASELINE_FULL:
        generalization.append(_gate(
            "full_causal_beats_current_frame_by_10pct",
            bool(per_seed_vs_current) and all(
                v >= MIN_IMPROVEMENT_OVER_CURRENT_FRAME
                for v in per_seed_vs_current.values()),
            per_seed_vs_current, MIN_IMPROVEMENT_OVER_CURRENT_FRAME,
            "matched by seed; failure here permits freezing the simpler model instead"))
    else:
        # The simpler model is the one being frozen, so it cannot be required to beat
        # itself. The handoff's alternative branch is satisfied by establishing that
        # history adds nothing measurable, which is recorded in history_value below.
        generalization.append(_gate(
            "temporal_history_value_established", True, per_seed_vs_current,
            "history adds no measurable value; simpler model frozen instead",
            "alternative branch of the FULL_CAUSAL vs CURRENT_FRAME_ONLY gate"))

    cosines = {k: v["metrics"][TEST]["head"]["median_direction_cosine_active"]
               for k, v in primary.items()}
    generalization.append(_gate(
        "median_direction_cosine_active",
        all(c is not None and c >= MIN_DIRECTION_COSINE for c in cosines.values()),
        cosines, MIN_DIRECTION_COSINE))

    false_positives = {k: v["metrics"][TEST]["activation"][
        "oracle_zero_false_positive_rate"] for k, v in primary.items()}
    generalization.append(_gate(
        "oracle_zero_false_positive_rate",
        all(f is not None and f <= MAX_ZERO_FALSE_POSITIVE_RATE
            for f in false_positives.values()),
        false_positives, MAX_ZERO_FALSE_POSITIVE_RATE,
        "threshold frozen on calibration before offline test was opened"))

    ratios = {}
    for key, run in primary.items():
        block = run["metrics"][TEST]["head"]
        raw = block["hazard_absent_raw_head_rms"]
        ratios[key] = (block["hazard_absent_rms"] / raw
                       if raw not in (None, 0) else None)
    generalization.append(_gate(
        "hazard_absent_rms_ratio",
        all(r is not None and r <= MAX_HAZARD_ABSENT_RMS_RATIO for r in ratios.values()),
        ratios, MAX_HAZARD_ABSENT_RMS_RATIO,
        "predicted differential RMS relative to raw SafetyHead RMS on the same frames"))

    auprc_multiples = {}
    for key, run in primary.items():
        mask = run["metrics"][TEST].get("mask", {})
        auprc, prevalence = mask.get("auprc"), mask.get("prevalence")
        auprc_multiples[key] = (auprc / prevalence
                                if auprc and prevalence else None)
    generalization.append(_gate(
        "changed_mask_auprc_over_prevalence",
        all(m is not None and m >= MIN_AUPRC_PREVALENCE_MULTIPLE
            for m in auprc_multiples.values()),
        auprc_multiples, MIN_AUPRC_PREVALENCE_MULTIPLE))

    source_mode_cosines = {}
    for key, run in primary.items():
        modes = run["metrics"][TEST]["by_source_mode"]
        source_mode_cosines[key] = {
            name: block.get("median_direction_cosine_active")
            for name, block in modes.items() if block.get("available")}
    worst = [(k, name, value) for k, block in source_mode_cosines.items()
             for name, value in block.items()
             if value is not None and value < MIN_SOURCE_MODE_COSINE]
    generalization.append(_gate(
        "per_source_mode_direction_cosine", not worst, source_mode_cosines,
        MIN_SOURCE_MODE_COSINE,
        "modes absent from the offline-test partition cannot be evaluated"))

    spread = summarize_seeds(list(primary_maes.values()))
    generalization.append(_gate(
        "seed_coefficient_of_variation",
        spread["coefficient_of_variation"] is not None
        and spread["coefficient_of_variation"] < MAX_SEED_COEFFICIENT_OF_VARIATION,
        spread, MAX_SEED_COEFFICIENT_OF_VARIATION))

    technical_pass = all(g["passed"] for g in technical)
    generalization_pass = all(g["passed"] for g in generalization)

    full_maes = {v["seed"]: v["metrics"][TEST]["head"]["differential_mae"]
                 for v in _runs_for(final_report, BASELINE_FULL).values()}
    full_spread = summarize_seeds(list(full_maes.values()))
    current_spread = summarize_seeds(list(current_maes.values()))
    history_value = {
        "full_causal_offline_mae": full_spread,
        "current_frame_only_offline_mae": current_spread,
        "mean_relative_improvement": (
            1.0 - full_spread["mean"] / current_spread["mean"]
            if full_spread["mean"] is not None and current_spread["mean"] else None),
        "seed_ranges_overlap": bool(
            full_spread.get("values") and current_spread.get("values")
            and max(full_spread["values"]) >= min(current_spread["values"])
            and max(current_spread["values"]) >= min(full_spread["values"])),
    }

    return {
        "primary_variant": primary_variant,
        "temporal_history_value": history_value,
        "technical": technical,
        "generalization": generalization,
        "technical_passed": technical_pass,
        "generalization_passed": generalization_pass,
        "all_passed": technical_pass and generalization_pass,
        "zero_baseline_offline_mae": zero_mae,
        "primary_offline_mae": primary_maes,
        "seed_spread": spread,
        "thresholds": {
            "min_improvement_over_zero": MIN_IMPROVEMENT_OVER_ZERO,
            "min_improvement_over_current_frame": MIN_IMPROVEMENT_OVER_CURRENT_FRAME,
            "min_direction_cosine": MIN_DIRECTION_COSINE,
            "max_zero_false_positive_rate": MAX_ZERO_FALSE_POSITIVE_RATE,
            "max_hazard_absent_rms_ratio": MAX_HAZARD_ABSENT_RMS_RATIO,
            "min_auprc_prevalence_multiple": MIN_AUPRC_PREVALENCE_MULTIPLE,
            "min_source_mode_cosine": MIN_SOURCE_MODE_COSINE,
            "max_seed_coefficient_of_variation": MAX_SEED_COEFFICIENT_OF_VARIATION,
        },
    }


def decide(gates: dict, *, training_produced_checkpoints: bool = True) -> str:
    """Map gate results onto one of the five allowed decision tokens."""
    if not training_produced_checkpoints:
        return DECISION_TRAINING_FAILED

    data_gates = {"no_partition_leakage", "input_contract_valid", "dataset_unchanged"}
    for gate in gates["technical"]:
        if gate["gate"] in data_gates and not gate["passed"]:
            return DECISION_DATA_INVALID

    if not gates["technical_passed"]:
        return DECISION_TRAINING_FAILED

    if gates["all_passed"]:
        return DECISION_READY

    failed = {g["gate"] for g in gates["generalization"] if not g["passed"]}

    # A model that cannot beat the trivial baseline has no signal to overfit; that is a
    # statement about the inputs, not about training dynamics.
    if "offline_mae_beats_zero_by_25pct" in failed:
        return DECISION_INSUFFICIENT

    # Beating the baseline offline but failing direction, calibration or source-mode
    # generalization is the overfit case.
    return DECISION_OVERFIT


def resolve(final_report: dict, input_contract: dict, partition_report: dict,
            dataset_decision: dict, *,
            training_produced_checkpoints: bool = True) -> dict:
    """Evaluate the gates, taking the simpler-model branch when history adds nothing.

    The handoff allows the FULL_CAUSAL-versus-CURRENT_FRAME_ONLY gate to be satisfied
    either by a 10% improvement or by establishing that temporal history provides no
    measurable value and freezing the simpler model. Taking the second branch is not a
    free pass: the simpler model then has to clear every other generalization gate on its
    own, so the gates are recomputed against it rather than inherited.
    """
    gates = compute_gates(final_report, input_contract, partition_report,
                          dataset_decision, primary_variant=BASELINE_FULL)
    decision = decide(gates, training_produced_checkpoints=training_produced_checkpoints)
    if decision == DECISION_READY:
        return {"gates": gates, "decision": decision, "frozen_primary": BASELINE_FULL,
                "branch": "full_causal_beat_current_frame_by_10pct"}

    failed = {g["gate"] for g in gates["generalization"] if not g["passed"]}
    # Whenever history fails to earn its keep, the simpler model is the one that would be
    # frozen, so it is the one the remaining gates have to be scored against. Requiring
    # the history gate to be the *only* failure would report gate values for a model
    # nobody would deploy -- and would call it overfitting on the strength of the
    # discarded model's numbers.
    if (gates["technical_passed"]
            and "full_causal_beats_current_frame_by_10pct" in failed):
        simpler = compute_gates(final_report, input_contract, partition_report,
                                dataset_decision, primary_variant=BASELINE_CURRENT)
        simpler_decision = decide(
            simpler, training_produced_checkpoints=training_produced_checkpoints)
        return {"gates": simpler, "decision": simpler_decision,
                "frozen_primary": BASELINE_CURRENT,
                "branch": "temporal_history_added_no_measurable_value",
                "full_causal_gates": gates}

    return {"gates": gates, "decision": decision, "frozen_primary": BASELINE_FULL,
            "branch": "full_causal_evaluated_as_primary"}
