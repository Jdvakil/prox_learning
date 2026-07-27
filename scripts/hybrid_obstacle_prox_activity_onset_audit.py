#!/usr/bin/env python3
"""Causal onset attribution on the frozen checkpoint. No training, no weight changes.

Handoff steps 4-5. The question is which input drives the onset false activations. Eight
interventions are applied to five matched frame groups; nothing is written back to the
dataset and no parameter is touched.

``CURRENT_FIELD_IDENTITY_CONTROL`` is the harness's own control: it passes the current field
through unchanged, so it must reproduce ``FULL_INPUT`` exactly. If it does not, the
intervention machinery is perturbing something and every other number here is suspect.
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
from causal_parked_skin.model import FrozenSafetyHead

PREVIOUS_THRESHOLD = 0.99960857629776
CONSUMED_PARTITIONS = ("offline_reference_test", "reference_calibration",
                       "reference_validation")
ONSET_MIN_FRAMES = 10
ONSET_FRACTION = 0.10

STATE_PRIOR_DOMINANT = "STATE_PRIOR_DOMINANT"
PROXIMITY_AMBIGUITY_DOMINANT = "PROXIMITY_AMBIGUITY_DOMINANT"
SHARED_REPRESENTATION_CONFOUND = "SHARED_REPRESENTATION_CONFOUND"

# predeclared decision rule (handoff step 5)
STATE_PRIOR_MIN_DROP = 0.5      # probability drop on a majority of the 17 frames


def onset_cutoff(length: int) -> int:
    return max(ONSET_MIN_FRAMES, int(np.ceil(ONSET_FRACTION * length)))


def collect_frames(cache: Path, model, head, device):
    """Score the consumed partitions and index every frame with what it needs."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from hybrid_obstacle_reference_threshold_calibrate import (
        build_trajectories,
        score_partition,
    )

    rows = []
    for name in CONSUMED_PARTITIONS:
        partition = load_partition(cache, name)
        scored = score_partition(model, partition, head, device)
        trajectories = build_trajectories(partition, scored)
        offset = 0
        for trajectory in trajectories:
            cutoff = onset_cutoff(trajectory.frames)
            for step in range(trajectory.frames):
                rows.append({
                    "partition": name,
                    "global_index": offset + step,
                    "trajectory_id": trajectory.trajectory_id,
                    "episode_id": trajectory.episode_id,
                    "distribution": trajectory.distribution,
                    "hazard_present": trajectory.hazard_present,
                    "step": step,
                    "trajectory_length": trajectory.frames,
                    "onset": step < cutoff,
                    "oracle_active": bool(trajectory.oracle_active[step]),
                    "activity": float(trajectory.activity[step]),
                    "fired": bool(trajectory.activity[step] >= PREVIOUS_THRESHOLD),
                })
            offset += trajectory.frames
    return rows


def make_groups(rows) -> dict:
    false_positives = [r for r in rows if r["fired"] and not r["oracle_active"]]
    return {
        "A_known_false_positives": false_positives,
        "B_onset_zero_hazard_absent": [
            r for r in rows if r["onset"] and not r["hazard_present"]
            and not r["oracle_active"]],
        "C_onset_active_hazard_present": [
            r for r in rows if r["onset"] and r["hazard_present"] and r["oracle_active"]],
        "D_later_active": [r for r in rows if not r["onset"] and r["oracle_active"]],
        "E_later_zero": [r for r in rows if not r["onset"] and not r["oracle_active"]],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--max-per-group", type=int, default=400)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch

    stack = json.loads(args.stack.read_text())
    engine.set_sensor_names(stack["sensor_contract"]["ordered_names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = FrozenSafetyHead.load(args.safety_dir, device=device)
    model, payload = load_checkpoint(args.checkpoint, device)
    before = [p.detach().clone() for p in model.parameters()]

    rows = collect_frames(args.cache, model, head, device)
    groups = make_groups(rows)
    print(f"group sizes: { {k: len(v) for k, v in groups.items()} }")

    # a fixed oracle-zero reference field: the elementwise mean current field over
    # oracle-zero frames of the training partition, deterministic and never fitted
    train = load_partition(args.cache, "reference_train")
    zero_mask = ~np.asarray(train["oracle_active"]).astype(bool)
    reference_field = np.asarray(train["current"])[zero_mask].mean(axis=0)
    reference_valid = np.ones_like(reference_field, dtype=bool)

    partitions = {name: load_partition(args.cache, name) for name in CONSUMED_PARTITIONS}
    hazard_by_partition = {name: np.asarray(p["hazard_present"]).astype(bool)
                           for name, p in partitions.items()}

    captured = {}

    def hook(name):
        def inner(_m, _i, output):
            captured[name] = output.detach()
        return inner

    model.state_encoder.register_forward_hook(hook("state"))
    model.to_token.register_forward_hook(hook("prox"))

    rng = np.random.default_rng(20260727)

    def evaluate(frames, intervention):
        """Run one intervention over a list of frame records."""
        if not frames:
            return None
        by_partition: dict[str, list[int]] = {}
        for position, record in enumerate(frames):
            by_partition.setdefault(record["partition"], []).append(position)

        activity = np.zeros(len(frames))
        pre_logit = np.zeros(len(frames))
        delta_norm = np.zeros(len(frames))
        dq_norm = np.zeros(len(frames))
        mask_fraction = np.zeros(len(frames))
        state_norm = np.zeros(len(frames))
        prox_norm = np.zeros(len(frames))

        for name, positions in by_partition.items():
            partition = partitions[name]
            index = np.array([frames[p]["global_index"] for p in positions])
            for start in range(0, len(index), 128):
                chunk = index[start:start + 128]
                slots = positions[start:start + 128]
                batch = make_batch(partition, chunk, device)
                history = batch["history"].clone()
                history_valid = batch["history_valid"].clone()
                state = batch["state"].clone()

                if intervention == "PROX_ONLY":
                    state = torch.zeros_like(state)
                elif intervention == "STATE_ONLY":
                    history[:, -1] = torch.from_numpy(reference_field).to(device)
                    history_valid[:, -1] = torch.from_numpy(reference_valid).to(device)
                elif intervention == "STATE_SHUFFLED_WITHIN_ONSET":
                    state = state[torch.from_numpy(
                        rng.permutation(state.shape[0])).to(device)]
                elif intervention == "STATE_SWAPPED_ACROSS_HAZARD_STRATA":
                    hazard = hazard_by_partition[name]
                    opposite = np.flatnonzero(hazard != hazard[chunk[0]])
                    if opposite.size:
                        donor = rng.choice(opposite, size=len(chunk))
                        state = make_batch(partition, np.sort(donor), device)["state"]
                        state = state[:len(chunk)]
                elif intervention == "PROX_SWAPPED_ACROSS_HAZARD_STRATA":
                    hazard = hazard_by_partition[name]
                    opposite = np.flatnonzero(hazard != hazard[chunk[0]])
                    if opposite.size:
                        donor = rng.choice(opposite, size=len(chunk))
                        other = make_batch(partition, np.sort(donor), device)
                        history[:, -1] = other["history"][:len(chunk), -1]
                        history_valid[:, -1] = other["history_valid"][:len(chunk), -1]
                elif intervention == "STATE_MEAN":
                    state = state.mean(dim=0, keepdim=True).expand_as(state).contiguous()
                elif intervention == "CURRENT_FIELD_IDENTITY_CONTROL":
                    history = history.clone()          # explicit no-op
                elif intervention != "FULL_INPUT":
                    raise SystemExit(f"unknown intervention {intervention}")

                with torch.no_grad():
                    out = model(history, history_valid, state)
                    probability = out["changed_probability"]
                    flat = probability.reshape(probability.shape[0], -1)
                    dq = head(out["current"]) - head(out["parked"])

                for offset, slot in enumerate(slots):
                    activity[slot] = float(flat[offset].max())
                    pre_logit[slot] = float(
                        out["mask_logits"][offset].reshape(-1).max())
                    delta_norm[slot] = float(out["delta"][offset].norm())
                    dq_norm[slot] = float(dq[offset].norm())
                    mask_fraction[slot] = float((flat[offset] > 0.5).float().mean())
                    state_norm[slot] = float(captured["state"][offset].norm())
                    prox_norm[slot] = float(captured["prox"].reshape(
                        len(chunk), 40, -1)[offset].norm())

        return {
            "frames": len(frames),
            "activity_probability": {"mean": float(activity.mean()),
                                     "median": float(np.median(activity)),
                                     "max": float(activity.max())},
            "predicted_removable_field_norm": float(delta_norm.mean()),
            "predicted_oracle_differential_norm": float(dq_norm.mean()),
            "change_mask_fraction": float(mask_fraction.mean()),
            "activity_pre_logit": float(pre_logit.mean()),
            "state_context_embedding_norm": float(state_norm.mean()),
            "proximity_token_embedding_norm": float(prox_norm.mean()),
            "_activity_values": activity,
        }

    interventions = ("FULL_INPUT", "PROX_ONLY", "STATE_ONLY",
                     "STATE_SHUFFLED_WITHIN_ONSET",
                     "STATE_SWAPPED_ACROSS_HAZARD_STRATA",
                     "PROX_SWAPPED_ACROSS_HAZARD_STRATA", "STATE_MEAN",
                     "CURRENT_FIELD_IDENTITY_CONTROL")

    results: dict[str, dict] = {}
    activity_by_group: dict[str, dict] = {}
    for group, frames in groups.items():
        subset = frames if len(frames) <= args.max_per_group else [
            frames[i] for i in np.linspace(0, len(frames) - 1, args.max_per_group,
                                           dtype=int)]
        results[group] = {}
        activity_by_group[group] = {}
        for intervention in interventions:
            block = evaluate(subset, intervention)
            if block is None:
                results[group][intervention] = None
                continue
            activity_by_group[group][intervention] = block.pop("_activity_values")
            results[group][intervention] = block
        print(f"  {group:<32} n={len(subset):>4} "
              f"full={results[group]['FULL_INPUT']['activity_probability']['mean']:.4f} "
              f"prox_only={results[group]['PROX_ONLY']['activity_probability']['mean']:.4f} "
              f"state_only={results[group]['STATE_ONLY']['activity_probability']['mean']:.4f}")

    # ---- harness control ---------------------------------------------------------
    identity_delta = max(
        float(np.abs(activity_by_group[g]["FULL_INPUT"]
                     - activity_by_group[g]["CURRENT_FIELD_IDENTITY_CONTROL"]).max())
        for g in activity_by_group)
    after = [p.detach().clone() for p in model.parameters()]
    weights_unchanged = all(torch.equal(a, b) for a, b in zip(before, after))

    # ---- predeclared classification ----------------------------------------------
    known = activity_by_group["A_known_false_positives"]
    full = known["FULL_INPUT"]
    drops = {name: full - known[name] for name in
             ("PROX_ONLY", "STATE_SHUFFLED_WITHIN_ONSET",
              "STATE_SWAPPED_ACROSS_HAZARD_STRATA", "STATE_MEAN")}
    majority_drop = {name: float((value >= STATE_PRIOR_MIN_DROP).mean())
                     for name, value in drops.items()}
    state_intervention_reduces = any(v > 0.5 for v in majority_drop.values())
    proximity_alone_reproduces = bool(
        np.median(known["PROX_ONLY"]) >= np.median(full) - STATE_PRIOR_MIN_DROP)

    if state_intervention_reduces and not proximity_alone_reproduces:
        classification = STATE_PRIOR_DOMINANT
    elif not state_intervention_reduces and proximity_alone_reproduces:
        classification = PROXIMITY_AMBIGUITY_DOMINANT
    else:
        classification = SHARED_REPRESENTATION_CONFOUND

    report = {
        "schema": "hybrid_obstacle_prox_activity_onset_audit_v1",
        "checkpoint_config_hash": payload["config_hash"],
        "previous_threshold": PREVIOUS_THRESHOLD,
        "onset_definition": (
            f"episode_step < max({ONSET_MIN_FRAMES}, "
            f"ceil({ONSET_FRACTION} * trajectory_length))"),
        "group_sizes": {k: len(v) for k, v in groups.items()},
        "known_false_positive_frames": [
            {k: r[k] for k in ("partition", "trajectory_id", "episode_id",
                               "distribution", "hazard_present", "step",
                               "trajectory_length", "activity")}
            for r in groups["A_known_false_positives"]],
        "known_false_positive_count": len(groups["A_known_false_positives"]),
        "interventions": list(interventions),
        "results": results,
        "harness_control": {
            "identity_control_max_activity_delta": identity_delta,
            "identity_control_exact": bool(identity_delta == 0.0),
            "model_weights_unchanged": weights_unchanged,
            "dataset_modified": False,
        },
        "classification_rule": {
            "state_prior_dominant": (
                f"a state intervention drops activity by >= {STATE_PRIOR_MIN_DROP} on a "
                "majority of the known false positives, and proximity-preserving "
                "interventions retain substantially lower activity"),
            "proximity_ambiguity_dominant": (
                "state interventions do not materially reduce the false positive and "
                "current proximity alone reproduces the activation"),
            "shared_representation_confound": "both contribute, or they cannot be isolated",
        },
        "classification_evidence": {
            "fraction_of_known_fps_dropping_by_0.5": majority_drop,
            "state_intervention_reduces_majority": state_intervention_reduces,
            "proximity_alone_reproduces_activation": proximity_alone_reproduces,
            "median_activity_full_input": float(np.median(full)),
            "median_activity_prox_only": float(np.median(known["PROX_ONLY"])),
            "median_activity_state_only": float(np.median(known["STATE_ONLY"])),
        },
        "classification": classification,
        "reference_field": {
            "definition": ("elementwise mean current closeness over all oracle-zero "
                           "frames of reference_train; deterministic, never fitted"),
            "mean_closeness": float(reference_field.mean()),
            "max_closeness": float(reference_field.max()),
        },
    }
    report["report_sha256"] = thr.canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"\nknown false positives      : {report['known_false_positive_count']}")
    print(f"identity control exact     : {report['harness_control']['identity_control_exact']}")
    print(f"model weights unchanged    : {weights_unchanged}")
    print(f"fraction dropping >= 0.5   : {majority_drop}")
    print(f"CLASSIFICATION             : {classification}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
