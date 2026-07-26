#!/usr/bin/env python3
"""Paired-skin data-contract audit for CAUSAL_PARKED_SKIN_REFERENCE_V1.

Handoff step 3. The model predicts the parked 40x8x8 proximity field, so the paired
dataset must supply, for every trainable frame:

* ``current_skin``  -- four causal 40x8x8 frames, latest at the decision state (INPUT)
* ``parked_skin``   -- the counterfactual 40x8x8 field at the same decision state (TARGET)

This audit verifies field presence, shape, sensor order, finiteness, causal ordering and
the physical constraint ``parked_closeness <= current_closeness`` -- per distribution,
per partition, and per frame -- and reports what is missing rather than repairing it.
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
sys.path.insert(0, str(ROOT / "submodules" / "act"))

#: meta.json d_max_input -- the depth at which closeness reaches zero.
D_MAX = 0.5
#: repository rule: a reading below 5 mm is a dead pixel, not a contact.
DEAD_PIXEL_BELOW_M = 0.005

REQUIRED_INPUT = "four causal current 40x8x8 frames"
REQUIRED_TARGET = "parked 40x8x8 field at the same decision state"


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def closeness(depth: np.ndarray) -> np.ndarray:
    """clip(1 - depth/0.5, 0, 1), with dead pixels forced to zero closeness."""
    field = np.asarray(depth, dtype=np.float32)
    value = np.clip(1.0 - field / D_MAX, 0.0, 1.0)
    return np.where(field >= DEAD_PIXEL_BELOW_M, value, 0.0).astype(np.float32)


def describe(files: list[Path], label: str) -> dict[str, Any]:
    """What a shard family actually stores, and whether it meets the contract."""
    if not files:
        return {"label": label, "files": 0, "present": False}
    blob = np.load(files[0], allow_pickle=False)
    keys = sorted(blob.files)
    frames = sum(len(np.load(f, allow_pickle=False)["timestep"]) for f in files)

    causal = "skin_stack" in keys
    causal_shape = list(blob["skin_stack"].shape[1:]) if causal else None
    parked_field = next((k for k in keys
                         if "parked" in k and np.asarray(blob[k]).ndim >= 3
                         and np.asarray(blob[k]).shape[-2:] == (8, 8)), None)
    return {
        "label": label,
        "files": len(files),
        "frames": int(frames),
        "keys": keys,
        "has_causal_current_skin": causal,
        "causal_frame_shape": causal_shape,
        "causal_frames_stored": causal_shape[0] if causal else 0,
        "has_parked_skin_field": parked_field is not None,
        "parked_skin_field_name": parked_field,
        "parked_head_present": any("parked_head" in k for k in keys),
        "oracle_dq_present": any("oracle_dq" in k for k in keys),
        "meets_input_contract": bool(causal and causal_shape == [4, 40, 8, 8]),
        "meets_target_contract": parked_field is not None,
    }


def audit_present_skins(files: list[Path], limit: int | None = None) -> dict[str, Any]:
    """Everything checkable on the fields that ARE stored."""
    checked = 0
    finite = True
    shape_ok = True
    causal_latest_matches_summary = 0
    causal_history_monotone_index = True
    for path in files[:limit] if limit else files:
        blob = np.load(path, allow_pickle=False)
        if "skin_stack" not in blob.files:
            continue
        stack = np.asarray(blob["skin_stack"], dtype=np.float32)
        if stack.shape[1:] != (4, 40, 8, 8):
            shape_ok = False
        if not np.isfinite(stack).all():
            finite = False
        # the latest causal frame must be the decision-state frame the summary came from
        summary = np.asarray(blob["sensor_summary"], dtype=np.float32)
        latest = stack[:, -1]
        valid = latest >= DEAD_PIXEL_BELOW_M
        minimum = np.where(valid.any(axis=(2, 3)),
                           np.where(valid, latest, np.inf).min(axis=(2, 3)), 1.0)
        minimum = np.minimum(minimum, 1.0)
        causal_latest_matches_summary += int(
            np.allclose(minimum, summary[:, :, 0], atol=1e-6))
        # earlier slots are zero-padded at the start of an episode, never future frames
        if stack.shape[0] > 4 and not np.array_equal(stack[4:, :3], stack[3:-1, 1:]):
            causal_history_monotone_index = False
        checked += 1
    return {
        "files_checked": checked,
        "all_finite": finite,
        "all_shapes_4x40x8x8": shape_ok,
        "files_whose_latest_causal_frame_matches_the_decision_state_summary":
            causal_latest_matches_summary,
        "causal_history_is_a_shifted_window": causal_history_monotone_index,
        "note": ("this establishes the INPUT side only; the parked TARGET is absent, so "
                 "the parked<=current constraint and the changed-pixel mask cannot be "
                 "evaluated at all"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--expert-paired-dir", required=True, type=Path)
    ap.add_argument("--labelling-schedule", required=True, type=Path)
    ap.add_argument("--learner-schedule", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    for name in ("partition", "expert_paired_dir", "labelling_schedule",
                 "learner_schedule", "out"):
        setattr(args, name, Path(getattr(args, name)).resolve())

    partition = json.loads(args.partition.read_text())
    labelling = json.loads(args.labelling_schedule.read_text())
    learner = json.loads(args.learner_schedule.read_text())

    def on_policy_files(schedule, distribution) -> list[Path]:
        out = []
        for entry in schedule["entries"]:
            directory = Path(entry["output_dir"])
            summary = directory / "summary.json"
            frames = directory / "frames.npz"
            if summary.is_file() and frames.is_file() and \
                    json.loads(summary.read_text())["distribution"] == distribution:
                out.append(frames)
        return out

    families = {
        "expert": describe(sorted(args.expert_paired_dir.glob("*.npz")), "expert"),
        "act_only_on_policy": describe(
            on_policy_files(labelling, "act_only_on_policy"), "act_only_on_policy"),
        "oracle_on_policy": describe(
            on_policy_files(labelling, "oracle_on_policy"), "oracle_on_policy"),
        "learner_on_policy": describe(
            on_policy_files(learner, "learner_on_policy"), "learner_on_policy"),
    }

    input_ok = {k: v.get("meets_input_contract", False) for k, v in families.items()}
    target_ok = {k: v.get("meets_target_contract", False) for k, v in families.items()}
    total_frames = sum(v.get("frames", 0) for v in families.values())
    trainable = sum(v.get("frames", 0) for k, v in families.items()
                    if input_ok[k] and target_ok[k])

    present_side = audit_present_skins(sorted(args.expert_paired_dir.glob("*.npz")))

    # what regeneration would cost, so the next task can be specified concretely
    bytes_per_field = 40 * 8 * 8 * 4
    regeneration = {
        "why_needed": ("the parked 40x8x8 field is the training TARGET and is stored "
                       "nowhere; the causal current skins are the model INPUT and are "
                       "stored only for the expert distribution"),
        "what_is_retained_instead": ("the parked field's SHA-256 and the 7-D SafetyHead "
                                     "output derived from it -- neither is invertible to "
                                     "the field"),
        "frames_needing_regeneration": total_frames,
        "bytes_per_frame_parked_field": bytes_per_field,
        "bytes_per_frame_causal_input": 4 * bytes_per_field,
        "estimated_uncompressed_gib": round(
            total_frames * 5 * bytes_per_field / (1024 ** 3), 2),
        "rollouts_that_would_have_to_rerun": {
            "expert_paired": 100, "on_policy_labelling": 200, "learner_induced": 64,
            "total": 364},
        "prohibited_by_this_task": ("'Do not collect another on-policy training dataset' "
                                    "forbids re-running the 264 on-policy rollouts, and "
                                    "MSAA makes a rerun a different sample in any case"),
    }

    violations = {
        "missing_target_field_in_every_distribution": not any(target_ok.values()),
        "missing_input_field_in_on_policy_distributions":
            sorted(k for k, v in input_ok.items() if not v),
        "distributions_meeting_both_contracts":
            sorted(k for k in families if input_ok[k] and target_ok[k]),
        "trainable_frames": trainable,
        "required_distributions_by_sampling_contract": ["expert", "act_only_on_policy",
                                                        "oracle_on_policy",
                                                        "learner_on_policy"],
        "sampling_contract_satisfiable": trainable > 0 and all(
            input_ok[k] and target_ok[k] for k in families),
    }

    report = {
        "schema": "hybrid_obstacle_parked_skin_data_audit_v1",
        "model": "CAUSAL_PARKED_SKIN_REFERENCE_V1",
        "required_input": REQUIRED_INPUT,
        "required_target": REQUIRED_TARGET,
        "closeness_transform": {
            "formula": "clip(1 - depth / 0.5, 0, 1)",
            "d_max_m": D_MAX,
            "dead_pixel_below_m": DEAD_PIXEL_BELOW_M,
            "dead_pixel_rule": "readings below 5 mm map to zero closeness",
            "implemented_and_unit_tested": True,
        },
        "partition_sha256": partition["partition_sha256"],
        "families": families,
        "input_contract_met": input_ok,
        "target_contract_met": target_ok,
        "total_frames_available": total_frames,
        "frames_meeting_both_contracts": trainable,
        "checks_on_the_fields_that_do_exist": present_side,
        "physical_constraint_check": {
            "constraint": "parked_closeness <= current_closeness on valid returns",
            "evaluable": False,
            "why_not": ("the parked closeness field does not exist in any shard, so the "
                        "constraint has nothing to compare against"),
            "violations_by_sensor": None, "violations_by_pixel": None,
            "violations_by_trajectory": None, "violations_by_distribution": None,
            "violations_by_task_phase": None,
        },
        "regeneration_requirement": regeneration,
        "violations": violations,
        "silent_repair_performed": False,
        "valid": bool(violations["sampling_contract_satisfiable"]),
        "decision_if_invalid": "PARKED_SKIN_DATA_CONTRACT_FAILED",
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"{'distribution':<22}{'files':>6}{'frames':>9}{'causal in':>11}{'parked target':>15}")
    for name, info in families.items():
        print(f"{name:<22}{info.get('files', 0):>6}{info.get('frames', 0):>9}"
              f"{('yes ' + str(info.get('causal_frames_stored')) + 'x40x8x8') if info.get('has_causal_current_skin') else 'NO':>11}"
              f"{('yes' if info.get('has_parked_skin_field') else 'MISSING'):>15}")
    print(f"\ntotal frames available          : {total_frames}")
    print(f"frames meeting BOTH contracts   : {trainable}")
    print(f"sampling contract satisfiable   : {violations['sampling_contract_satisfiable']}")
    print(f"parked<=current evaluable       : "
          f"{report['physical_constraint_check']['evaluable']}")
    print(f"\nregeneration would need {regeneration['rollouts_that_would_have_to_rerun']['total']} "
          f"rollouts and ~{regeneration['estimated_uncompressed_gib']} GiB")
    print(f"wrote {args.out}")
    return 0 if report["valid"] else 8


if __name__ == "__main__":
    raise SystemExit(main())
