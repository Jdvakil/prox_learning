#!/usr/bin/env python3
"""V10.9R inference-only action-chunk synchronization repair: frozen contract.

**Inference only.** No checkpoint, dataset, manifest or existing evaluation
artifact is retrained, modified or reinterpreted. V10.9's results stand exactly
as published; this is a decoder repair evaluated on fresh, disjoint seeds.

The defect, measured on the retained V10.9 rollouts rather than assumed:

* PACT closes the gripper in 38/40 rollouts, so it is not failing to *predict*
  closure.
* On F2 it touches the cup in 9/10 but holds it in only 2/10.
* Its failed F2 grasps close a median **35 control steps** after first contact,
  after a median **150 mm** of TCP travel. ACT, on the same family, closes in a
  median 9 steps / 20 mm and holds 5/10.

The cause is in the decoder, not the policy. Inference averages all eight
outputs across every overlapping chunk and only then thresholds the gripper, so
a chunk saying "close now" is diluted by chunks saying "close in thirty steps",
and pregrasp arm plans are averaged with postgrasp ones.

Chunk 1 previously did not close at all and chunk 25 closed only about half the
time, so the chunk-100 horizon carries information that must be preserved. The
repair therefore keeps the horizon and changes only how chunks are combined.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

for _p in (ROOT / "scripts",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CONTRACT_VERSION_V109R = "pact_place_v109r_chunk_sync_repair_v1"
PLAN_RELATIVE = "docs/PACT_PLACE_V109R_DECODER_REPAIR_PLAN.md"

IS_PHASE0_PASS = False
IS_INFERENCE_ONLY = True
AUTHORIZES_TRAINING = False
AUTHORIZES_DOWNSTREAM_CLAIMS = False
V107_PHASE0_RESULT = "failed_8_of_24_permanently_closed"

# --- inherited, deliberately NOT retuned ------------------------------------
# Both values are carried over unchanged from the V10.9 decoder. Retuning either
# against rollout outcomes would make the repair unfalsifiable.
GRIPPER_THRESHOLD = 127.5
AGE_DECAY = 0.01           # chunk weight = exp(-AGE_DECAY * age)
CHUNK_SIZE = 100           # chunk-100 horizon preserved

# --- the one new decision rule, frozen before any rollout -------------------
# A transition fires only when chunks carrying at least this fraction of the
# live weight agree that an event is due, and the weighted-median event time
# has arrived. 0.5 is a weighted majority: the natural, non-tunable choice.
CONSENSUS_WEIGHT_FRACTION = 0.5

# --- arm slew limit, derived from training data, never from outcomes --------
# The largest single-step L2 arm delta anywhere in the 113 training
# demonstrations (57,292 transitions). Chosen as the maximum rather than a
# percentile so the limit can never clamp a motion the expert itself performed;
# it only catches a discontinuity larger than anything in the training data.
ARM_SLEW_LIMIT_L2 = 0.806605
ARM_SLEW_REFERENCE = {
    "training_episodes": 113,
    "transitions": 57292,
    "p50": 0.012167, "p90": 0.029261, "p99": 0.056361,
    "p99_9": 0.092246, "p99_99": 0.263673, "max": 0.806605,
    "source": "assets/act_style_data/pact_place_v108_141, split_manifest train rows",
}

# --- gripper state machine ---------------------------------------------------
STATE_OPEN = "open"
STATE_CLOSED = "closed"
STATE_RELEASED = "released"
STATES = (STATE_OPEN, STATE_CLOSED, STATE_RELEASED)
# open -> closed -> released. Closure never latches: a release transition must
# be decoded separately, from its own downward crossing. RELEASED is terminal,
# matching the place task's single grasp and single placement.
TRANSITIONS = {STATE_OPEN: STATE_CLOSED, STATE_CLOSED: STATE_RELEASED,
               STATE_RELEASED: None}

# --- measured V10.9 baseline the repair must beat ---------------------------
V109_BASELINE = {
    "pact_gripper_close_commanded": "38/40",
    "pact_f2_touched": "9/10", "pact_f2_held": "2/10", "pact_f2_success": "1/10",
    "act_f2_touched": "9/10", "act_f2_held": "5/10", "act_f2_success": "4/10",
    "pact_f2_failed_touch_to_close_median_controls": 35,
    "pact_f2_failed_tcp_travel_median_mm": 150,
    "act_f2_failed_touch_to_close_median_controls": 9,
    "act_f2_failed_tcp_travel_median_mm": 20,
    "act_all_funnel": {"touched": 29, "held": 17, "success": 13, "n": 40},
    "pact_all_funnel": {"touched": 27, "held": 13, "success": 11, "n": 40},
    "source": "diagnostics_output/pact_place_v109_eval_traj (retained trajectories)",
}

# --- outputs ------------------------------------------------------------------
WORK_ROOT = "diagnostics_output/pact_place_v109r_decoder"
DIAGNOSTIC_ROOT = "diagnostics_output/pact_place_v109r_diagnostic"
EVAL_ROOT = "diagnostics_output/pact_place_v109r_eval"
EVAL_MASTER_SEED = 2026083001   # disjoint from 2026082901 / 2026082902

# --- stop conditions, declared before execution -------------------------------
STOP_CONDITIONS = (
    "premature closure: a close event decoded before the gripper is at the cup",
    "failed release: a closure with no decoded release transition",
    "material arm discontinuity: a commanded step exceeding the training maximum",
    "artifact drift: any existing V10.9 artifact hash changing",
    "infrastructure error",
)


def build_contract() -> dict[str, Any]:
    from pact_place_v105_contract import canonical_payload_sha256, empty_authorization

    document = {
        **empty_authorization(),
        "schema_version": CONTRACT_VERSION_V109R,
        "plan": PLAN_RELATIVE,
        "is_phase0_pass": IS_PHASE0_PASS,
        "is_inference_only": IS_INFERENCE_ONLY,
        "authorizes_training": AUTHORIZES_TRAINING,
        "authorizes_downstream_claims": AUTHORIZES_DOWNSTREAM_CLAIMS,
        "v107_phase0_result": V107_PHASE0_RESULT,
        "inherited_unchanged": {
            "gripper_threshold": GRIPPER_THRESHOLD,
            "age_decay": AGE_DECAY,
            "chunk_size": CHUNK_SIZE,
        },
        "new_rules": {
            "consensus_weight_fraction": CONSENSUS_WEIGHT_FRACTION,
            "arm_slew_limit_l2": ARM_SLEW_LIMIT_L2,
            "arm_slew_reference": ARM_SLEW_REFERENCE,
        },
        "state_machine": {"states": list(STATES), "transitions": TRANSITIONS,
                          "released_is_terminal": True,
                          "closure_latches": False},
        "v109_baseline": V109_BASELINE,
        "stop_conditions": list(STOP_CONDITIONS),
        "applies_identically_to": ["ACT", "PACT"],
    }
    document["config_sha256"] = canonical_payload_sha256(document)
    return document


if __name__ == "__main__":
    import json
    print(json.dumps(build_contract(), indent=2, sort_keys=True))
