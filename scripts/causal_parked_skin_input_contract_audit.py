#!/usr/bin/env python3
"""Inventory every field in the frozen parked-skin dataset and fix the input contract.

Handoff step 2. Nothing here is designed around what the model would like to have; the
inventory is read off the files, and live availability is read off the evaluator source
that would eventually consume the model. A field is a permitted input only when it is both
stored and demonstrably constructed at decision time by the live evaluator.

Writes diagnostics_output/causal_parked_skin_reference_v1/input_contract_audit.json.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules" / "act"))

# The live evaluator builds its reference-input dictionary here. Live availability is
# proven by reading this file, not asserted.
LIVE_EVALUATOR = ROOT / "submodules" / "act" / "eval_act_obstacle_on_policy.py"

ROLE_INPUT = "model_input"
ROLE_TARGET = "target"
ROLE_METADATA = "metadata"
ROLE_PROHIBITED = "prohibited"

# name -> (units, causal availability, role, rationale)
CLASSIFICATION = {
    # ---- deployable group -------------------------------------------------------
    "current_closeness": (
        "dimensionless closeness in [0,1]; clip(1 - depth_m/0.5, 0, 1)",
        "available at t; the four-frame window [t-3..t] is reconstructed causally",
        ROLE_INPUT,
        "the primary observation; the field the obstacle is visible in"),
    "current_valid_mask": (
        "boolean per pixel; False where the depth reading is below 5 mm",
        "available at t together with the field it annotates",
        ROLE_INPUT,
        ("without it a dead pixel and a true contact are the same number, and the model "
         "would learn to read sensor dropout as an obstacle")),
    "qpos": (
        "radians (7 arm joints) and metres (2 finger joints)",
        "available at t before the control decision",
        ROLE_INPUT,
        "posture determines which sensors can see the parked hazard at all"),
    "qvel": (
        "rad/s (7 arm joints) and m/s (2 finger joints)",
        "available at t before the control decision",
        ROLE_INPUT,
        "constructed live by the evaluator from obs['qvel']"),
    "nominal_action": (
        "radians (7 arm targets) plus 1 gripper command",
        "the ACT action for step t, produced before the reference runs",
        ROLE_INPUT,
        "the evaluator computes it first precisely because the reference consumes it"),
    "gripper_state": (
        "metres, 2 finger joints",
        "available at t",
        ROLE_INPUT,
        "duplicates qpos[7:9] but is kept for parity with the live input dictionary"),
    "gripper_command": (
        "commanded gripper scalar",
        "available at t",
        ROLE_INPUT,
        "duplicates nominal_action[7:8]; kept for parity with the live dictionary"),
    "episode_step": (
        "integer control step index",
        "trivially available live as a counter",
        ROLE_METADATA,
        ("EXCLUDED from model input by choice: on-policy rows all run a fixed 200-step "
         "horizon, so a step index lets the model learn a time-indexed prior over when "
         "the hazard is near instead of reading it from the proximity field")),
    "control_timestamp": (
        "seconds since episode start",
        "trivially available live as step * policy_dt",
        ROLE_METADATA,
        "excluded for the same reason as episode_step; it is an affine function of it"),

    # ---- privileged group -------------------------------------------------------
    "parked_closeness": (
        "dimensionless closeness in [0,1]",
        "counterfactual; never observable live",
        ROLE_TARGET,
        "the primary regression target"),
    "parked_valid_mask": (
        "boolean per pixel",
        "counterfactual; never observable live",
        ROLE_TARGET,
        "target-side validity, used for loss masking and mask-agreement diagnostics"),
    "removable_closeness": (
        "dimensionless; current - parked",
        "counterfactual; never observable live",
        ROLE_TARGET,
        "the differential the model must reproduce in pixel space"),
    "changed_pixel_mask": (
        "boolean per pixel; |current - parked| > 1e-5",
        "counterfactual; never observable live",
        ROLE_TARGET,
        "supervises the changed-probability head"),
    "current_head": (
        "SafetyHead 7-D output, physical units after label_scale",
        "recomputable live from current_closeness through the frozen head",
        ROLE_TARGET,
        ("not privileged in principle -- it is a deterministic function of a deployable "
         "field -- but it is read as a stored target here, never as a model input")),
    "parked_head": (
        "SafetyHead 7-D output",
        "counterfactual; never observable live",
        ROLE_TARGET,
        "SafetyHead-space target"),
    "oracle_dq": (
        "SafetyHead 7-D differential, current_head - parked_head",
        "counterfactual; never observable live",
        ROLE_TARGET,
        "the quantity the whole reference exists to estimate"),
    "oracle_active": (
        "boolean per frame",
        "counterfactual; never observable live",
        ROLE_TARGET,
        "stratification and metric conditioning only"),
    "teacher_dq": (
        "7-D differential from the previous learner, when populated",
        "not part of this task's contract",
        ROLE_PROHIBITED,
        "a previous model's output; using it would make this a distillation task"),
    "teacher_valid": (
        "boolean per frame",
        "not part of this task's contract",
        ROLE_PROHIBITED,
        "companion flag to teacher_dq"),

    # ---- integrity group --------------------------------------------------------
    "current_field_sha256": ("hex digest", "derived from a deployable field",
                             ROLE_METADATA, "provenance and leakage checks"),
    "parked_field_sha256": ("hex digest", "derived from a privileged field",
                            ROLE_METADATA, "provenance only; never an input"),
    "scientific_state_sha256": ("hex digest", "derived from deployable state",
                                ROLE_METADATA, "partition-independence checking"),
    "state_neutral": ("boolean per frame", "oracle bookkeeping",
                      ROLE_METADATA, "confirms parking perturbed only the hazard"),
}

PROHIBITED_AS_INPUT = (
    "parked_closeness", "parked_valid_mask", "removable_closeness", "changed_pixel_mask",
    "parked_head", "oracle_dq", "oracle_active", "teacher_dq", "teacher_valid",
    "current_head", "parked_field_sha256",
)


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def live_input_fields() -> tuple[list[str], bool, str]:
    """Read the reference-input dictionary keys straight out of the live evaluator.

    Targets the assignment to the variable named ``runtime`` specifically. Matching on
    "largest dict containing qpos" instead finds the per-frame diagnostic record, which
    legitimately logs privileged fields and would wrongly certify them as live-available.

    Also reports whether the full current field is rendered on the runtime-observable
    side: the 40x8x8 field never appears in ``runtime`` (only its 7-D head output and a
    per-sensor summary do), so its live availability has to be proven from the render
    call that produces it.
    """
    source = LIVE_EVALUATOR.read_text()
    tree = ast.parse(source)
    keys: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if "runtime" not in names:
            continue
        keys = [k.value for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        break
    renders_full_field = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "render_current_skin" for n in ast.walk(tree))
    return sorted(keys), renders_full_field, hashlib.sha256(source.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--sample-files", type=int, default=24)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import h5py
    from parked_skin_retention import (
        DEPLOYABLE_FIELDS,
        INTEGRITY_FIELDS,
        PRIVILEGED_FIELDS,
    )

    manifest = json.loads(args.manifest.read_text())
    stack = json.loads(args.stack.read_text())
    sensor_names = list(stack["sensor_contract"]["ordered_names"])
    link_of_sensor = [n.rsplit("_sensor_", 1)[0] for n in sensor_names]
    links = sorted(set(link_of_sensor))

    entries = manifest["entries"]
    live_keys, renders_full_field, evaluator_sha = live_input_fields()
    if not renders_full_field:
        raise SystemExit("the evaluator never renders the current field; the 40x8x8 "
                         "input cannot be certified live-available")
    # The field and its mask are produced by render_current_skin() on the
    # runtime-observable side, upstream of the `runtime` dictionary.
    live_available = set(live_keys) | {"current_closeness", "current_valid_mask",
                                       "episode_step", "control_timestamp"}

    # ---- inventory, cross-checked over a sample of files ------------------------
    seen: dict[str, dict] = {}
    inconsistent: list[str] = []
    step = max(1, len(entries) // args.sample_files)
    sampled = entries[::step][:args.sample_files]
    for entry in sampled:
        with h5py.File(entry["output"], "r") as handle:
            frames = int(handle.attrs["frames"])
            for group in handle:
                for name in handle[group]:
                    dataset = handle[group][name]
                    shape = list(dataset.shape)
                    record = {"group": group, "dtype": str(dataset.dtype),
                              "per_frame_shape": shape[1:]}
                    if shape[0] != frames:
                        inconsistent.append(f"{name}: leading axis {shape[0]} != {frames}")
                    if name in seen and seen[name] != record:
                        inconsistent.append(
                            f"{name}: {seen[name]} vs {record} in {entry['episode_id']}")
                    seen[name] = record

    expected = (set(DEPLOYABLE_FIELDS) | set(PRIVILEGED_FIELDS) | set(INTEGRITY_FIELDS))
    missing = sorted(expected - set(seen))
    unexpected = sorted(set(seen) - expected)

    fields = []
    for name in sorted(seen):
        units, causal, role, rationale = CLASSIFICATION.get(
            name, ("UNCLASSIFIED", "UNKNOWN", ROLE_PROHIBITED, "not in the contract table"))
        live = name in live_available
        fields.append({
            "name": name,
            "group": seen[name]["group"],
            "per_frame_shape": seen[name]["per_frame_shape"],
            "dtype": seen[name]["dtype"],
            "units": units,
            "causal_availability": causal,
            "available_in_live_evaluator": bool(live),
            "role": role,
            "rationale": rationale,
        })

    model_inputs = [f["name"] for f in fields if f["role"] == ROLE_INPUT]
    # every permitted input must be live-available; that is the whole point of the audit
    non_live_inputs = [f["name"] for f in fields
                       if f["role"] == ROLE_INPUT and not f["available_in_live_evaluator"]]
    leaked = sorted(set(model_inputs) & set(PROHIBITED_AS_INPUT))

    # the proprioceptive state vector the model actually receives
    state_layout = [("qpos", 9), ("qvel", 9), ("nominal_action", 8),
                    ("gripper_state", 2), ("gripper_command", 1)]
    state_width = sum(w for _, w in state_layout)

    contract = {
        "model_input_fields": model_inputs,
        "causal_history_frames": 4,
        "causal_history_rule": manifest.get(
            "causal_history_rule",
            "history(t) = [t-3,t-2,t-1,t], left-padded by repeating the earliest frame"),
        "state_vector_layout": [{"field": n, "width": w} for n, w in state_layout],
        "state_vector_width": state_width,
        "sensor_count": len(sensor_names),
        "pixels_per_sensor": 64,
        "sensor_order_sha256": stack["sensor_contract"]["sensor_order_hash"],
        "links": links,
        "link_of_sensor": link_of_sensor,
        "prohibited_as_input": sorted(PROHIBITED_AS_INPUT),
    }

    report = {
        "schema": "causal_parked_skin_input_contract_audit_v1",
        "dataset_version": manifest["dataset_version"],
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "files_sampled": len(sampled),
        "fields": fields,
        "field_count": len(fields),
        "missing_expected_fields": missing,
        "unexpected_fields": unexpected,
        "inconsistent_field_shapes": sorted(set(inconsistent)),
        "live_evaluator": {
            "path": str(LIVE_EVALUATOR.relative_to(ROOT)),
            "sha256": evaluator_sha,
            "runtime_dictionary_keys": live_keys,
            "renders_full_current_field": renders_full_field,
            "additionally_live_available": sorted(live_available - set(live_keys)),
            "note": ("keys parsed from the assignment to `runtime` in the evaluator AST. "
                     "Matching the largest dict containing qpos instead finds the "
                     "per-frame diagnostic record, which logs privileged fields and "
                     "would wrongly certify them live-available."),
        },
        "contract": contract,
        "input_contract_sha256": canonical_hash(contract),
        "prohibited_inputs_used": leaked,
        "inputs_not_live_available": non_live_inputs,
        "valid": bool(not missing and not unexpected and not inconsistent
                      and not leaked and not non_live_inputs),
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"fields inventoried : {len(fields)} over {len(sampled)} files")
    for f in fields:
        print(f"  {f['role']:<12} {f['group']:<11} {f['name']:<24} "
              f"{f['per_frame_shape']!s:<12} {f['dtype']:<8} "
              f"live={f['available_in_live_evaluator']}")
    print(f"live runtime keys  : {live_keys}")
    print(f"model inputs       : {model_inputs}")
    print(f"state width        : {state_width}")
    print(f"missing/unexpected : {missing} / {unexpected}")
    print(f"prohibited used    : {leaked}")
    print(f"not live-available : {non_live_inputs}")
    print(f"input contract sha : {report['input_contract_sha256']}")
    print(f"valid              : {report['valid']}")
    print(f"wrote {args.out}")
    return 0 if report["valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
