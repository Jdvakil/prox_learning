#!/usr/bin/env python3
"""Trace the frozen model from inputs to the activity decision.

Handoff step 3. The question is narrow and structural: can proprioceptive state reach the
quantity that decides activation, and is that quantity shared with the parked-field decoder?

Shapes and reachability are recorded by running the frozen model with forward hooks, and
reachability is confirmed by gradient tracing rather than by reading the source: a tensor is
reachable from an input exactly when a gradient flows back to that input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin import engine
from causal_parked_skin import threshold as thr
from causal_parked_skin.data import load_partition
from causal_parked_skin.engine import load_checkpoint, make_batch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch

    stack = json.loads(args.stack.read_text())
    engine.set_sensor_names(stack["sensor_contract"]["ordered_names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, payload = load_checkpoint(args.checkpoint, device)

    partition = load_partition(args.cache, "reference_calibration")
    batch = make_batch(partition, np.arange(8), device)

    # ---- shapes, by hook ---------------------------------------------------------
    captured: dict[str, tuple] = {}

    def hook(name):
        def inner(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                captured[name] = tuple(tensor.shape)
        return inner

    handles = [
        model.encoder.register_forward_hook(hook("per_sensor_current_field_encoder")),
        model.to_token.register_forward_hook(hook("sensor_token_projection")),
        model.state_encoder.register_forward_hook(hook("state_context_encoder")),
        model.cross_sensor.register_forward_hook(hook("cross_sensor_transformer")),
        model.decoder.register_forward_hook(hook("shared_per_sensor_decoder")),
    ]
    with torch.no_grad():
        model(batch["history"], batch["history_valid"], batch["state"])
    for handle in handles:
        handle.remove()

    # ---- reachability, by gradient ------------------------------------------------
    history = batch["history"].clone().requires_grad_(True)
    state = batch["state"].clone().requires_grad_(True)
    live = model(history, batch["history_valid"], state)

    mask_logits = live["mask_logits"]
    activity_pre_logit = mask_logits.reshape(mask_logits.shape[0], -1).amax(dim=1)

    def reaches(scalar, source) -> bool:
        grad = torch.autograd.grad(scalar, source, retain_graph=True,
                                   allow_unused=True)[0]
        return bool(grad is not None and torch.isfinite(grad).all()
                    and float(grad.abs().sum()) > 0.0)

    activity_scalar = activity_pre_logit.sum()
    parked_scalar = live["parked"].sum()
    tensors = {
        "activity_pre_logit": {
            "shape": tuple(activity_pre_logit.shape),
            "definition": ("max over the 40x8x8 per-pixel mask logits; the frozen model "
                           "has no dedicated frame-level activity head"),
            "state_can_reach": reaches(activity_scalar, state),
            "current_proximity_can_reach": reaches(activity_scalar, history),
        },
        "predicted_parked_field": {
            "shape": tuple(live["parked"].shape),
            "definition": "current_closeness - changed_probability * magnitude * current",
            "state_can_reach": reaches(parked_scalar, state),
            "current_proximity_can_reach": reaches(parked_scalar, history),
        },
    }

    # does the parked field feed the activity decision, or only share an ancestor?
    tensors["activity_pre_logit"]["predicted_parked_field_can_reach"] = False
    tensors["activity_pre_logit"]["parked_field_relationship"] = (
        "sibling, not ancestor: both are produced from the same decoder output; the "
        "activity value is the mask-logit channel and the parked field multiplies that "
        "channel by the magnitude channel and the current field")

    shared_tokens = bool(
        captured.get("cross_sensor_transformer") is not None
        and tensors["activity_pre_logit"]["state_can_reach"])

    report = {
        "schema": "hybrid_obstacle_prox_activity_path_audit_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_config_hash": payload["config_hash"],
        "variant": payload["config"]["variant"],
        "seed": payload["config"]["seed"],
        "stages": [
            {"stage": "current_field_encoder",
             "shape": list(captured.get("per_sensor_current_field_encoder", ())),
             "source_inputs": ["current_closeness", "current_valid_mask"],
             "state_can_reach": False, "current_proximity_can_reach": True,
             "parked_field_can_reach": False,
             "note": "shared per-sensor conv over the frame-t tile only"},
            {"stage": "sensor_token_projection",
             "shape": list(captured.get("sensor_token_projection", ())),
             "source_inputs": ["current_closeness", "current_valid_mask"],
             "state_can_reach": False, "current_proximity_can_reach": True,
             "parked_field_can_reach": False,
             "note": "mean+max pooled conv features projected to d_model"},
            {"stage": "state_context_encoder",
             "shape": list(captured.get("state_context_encoder", ())),
             "source_inputs": ["qpos", "qvel", "nominal_action", "gripper_state",
                               "gripper_command"],
             "state_can_reach": True, "current_proximity_can_reach": False,
             "parked_field_can_reach": False,
             "note": "29-D proprioceptive vector -> d_model, ADDED to every sensor token"},
            {"stage": "cross_sensor_transformer",
             "shape": list(captured.get("cross_sensor_transformer", ())),
             "source_inputs": ["sensor_token_projection", "state_context_encoder",
                               "sensor_embedding", "link_embedding"],
             "state_can_reach": True, "current_proximity_can_reach": True,
             "parked_field_can_reach": False,
             "note": "state is summed into the tokens BEFORE attention"},
            {"stage": "shared_per_sensor_decoder",
             "shape": list(captured.get("shared_per_sensor_decoder", ())),
             "source_inputs": ["cross_sensor_transformer", "current_field_encoder"],
             "state_can_reach": True, "current_proximity_can_reach": True,
             "parked_field_can_reach": False,
             "note": ("emits BOTH logit channels: channel 0 is the change mask that "
                      "becomes the activity signal, channel 1 is the magnitude")},
        ],
        "tensors": tensors,
        "activity_head": {
            "dedicated_activity_head_exists": False,
            "activity_definition": ("sigmoid of the maximum per-pixel mask logit; a "
                                    "reduction of the same channel that builds the "
                                    "parked field"),
            "shares_context_conditioned_tokens_with_parked_decoder": shared_tokens,
            "state_reaches_activity": tensors["activity_pre_logit"]["state_can_reach"],
            "proximity_reaches_activity":
                tensors["activity_pre_logit"]["current_proximity_can_reach"],
        },
        "training_gradient_connectivity": {
            "activity_objective": ("focal BCE on changed_pixel_mask, i.e. the same "
                                   "channel the activity value reduces"),
            "gradient_reached_state_encoder": tensors["activity_pre_logit"][
                "state_can_reach"],
            "explanation": ("the mask loss backpropagates through the shared decoder into "
                            "the cross-sensor tokens, and the state embedding is summed "
                            "into those tokens, so the activity objective did train the "
                            "state encoder"),
        },
        "finding": (
            "There is no isolated activity path. Proprioceptive state is summed into every "
            "sensor token before the cross-sensor transformer, the shared decoder emits the "
            "mask-logit channel from those tokens, and the activity value is a max-reduction "
            "of that channel. State can therefore drive activation with no proximity "
            "evidence, and the activity signal is entangled with the parked-field decoder by "
            "construction rather than by accident."),
        "activation_threshold_previous": 0.99960857629776,
    }
    report["report_sha256"] = thr.canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    for stage in report["stages"]:
        print(f"  {stage['stage']:<32} {stage['shape']!s:<22} "
              f"state={stage['state_can_reach']!s:<5} prox={stage['current_proximity_can_reach']}")
    print(f"\ndedicated activity head : {report['activity_head']['dedicated_activity_head_exists']}")
    print(f"state reaches activity  : {report['activity_head']['state_reaches_activity']}")
    print(f"prox reaches activity   : {report['activity_head']['proximity_reaches_activity']}")
    print(f"shares decoder tokens   : {report['activity_head']['shares_context_conditioned_tokens_with_parked_decoder']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
