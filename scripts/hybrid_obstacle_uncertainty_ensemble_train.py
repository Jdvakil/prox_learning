#!/usr/bin/env python3
"""Train PARKED_SKIN_TRAJECTORY_BOOTSTRAP_ENSEMBLE_V1: five uncertainty estimators.

Handoff steps 3-5 and 8. These five members exist only to disagree. They never supply a
prediction the controller executes -- the frozen seed-0 model remains the sole source of the
parked field, the head output, the differential, its direction and its magnitude. A member
that happens to score better than seed 0 changes nothing.

The bootstrap resamples **trajectory clusters**, not frames and not distribution-specific
files. One manifest episode is one cluster, and all of its expert, ACT-only, oracle and
learner-induced rows move together. Resampling frames would let the same scene appear in a
member's training set under four labels' worth of correlated copies and make the members
agree for a reason that has nothing to do with epistemic uncertainty.

Resampling is stratified by hazard so every member sees 30 hazard-present and 10
hazard-absent clusters, matching the parent split. Without stratification a draw could omit
hazard-absent scenes entirely, and a member that has never seen a clear scene would disagree
about clear scenes for a trivial reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin import threshold as thr
from causal_parked_skin.data import Partition, load_partition
from causal_parked_skin.engine import TrainConfig, evaluate, set_sensor_names, train
from causal_parked_skin.losses import LossWeights
from causal_parked_skin.model import BASELINE_CURRENT, FrozenSafetyHead

ENSEMBLE_ID = "PARKED_SKIN_TRAJECTORY_BOOTSTRAP_ENSEMBLE_V1"
BOOTSTRAP_SEEDS = (20260731, 20260801, 20260802, 20260803, 20260804)
MEMBERS = len(BOOTSTRAP_SEEDS)
HAZARD_PRESENT_CLUSTERS = 30
HAZARD_ABSENT_CLUSTERS = 10


def canonical_hash(payload) -> str:
    return thr.canonical_hash(payload)


def git_head() -> str:
    import subprocess

    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def sha256_file(path) -> str | None:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def bootstrap_clusters(present: list[str], absent: list[str], seed: int):
    """Hazard-stratified cluster resample with replacement, multiplicity preserved."""
    rng = np.random.default_rng(seed)
    drawn_present = [present[i] for i in rng.integers(0, len(present),
                                                      HAZARD_PRESENT_CLUSTERS)]
    drawn_absent = [absent[i] for i in rng.integers(0, len(absent),
                                                    HAZARD_ABSENT_CLUSTERS)]
    return drawn_present + drawn_absent


def materialise(partition: Partition, clusters: list[str]) -> Partition:
    """Build a training view whose rows repeat a cluster as often as it was drawn.

    Multiplicity is preserved by index repetition rather than by reweighting: a cluster
    drawn twice genuinely appears twice, which is what the bootstrap means.
    """
    episodes = np.array(partition.episode_ids)
    trajectory = np.asarray(partition["trajectory"])
    rows: list[np.ndarray] = []
    trajectory_ids: list[str] = []
    episode_ids: list[str] = []
    for episode in clusters:
        for index in np.flatnonzero(episodes == episode):
            member_rows = np.flatnonzero(trajectory == index)
            rows.append(member_rows)
            trajectory_ids.append(partition.trajectory_ids[index])
            episode_ids.append(episode)
    if not rows:
        raise SystemExit("bootstrap sample selected no trajectories")

    order = np.concatenate(rows)
    arrays = {}
    for name, array in partition.arrays.items():
        if name == "history":
            continue
        arrays[name] = np.asarray(array)[order]
    # rebuild trajectory ids and causal history for the resampled view
    new_trajectory = np.concatenate(
        [np.full(len(r), i, dtype=np.int32) for i, r in enumerate(rows)])
    arrays["trajectory"] = new_trajectory
    history = []
    offset = 0
    from causal_parked_skin.data import causal_history_indices

    for member_rows in rows:
        history.append(causal_history_indices(len(member_rows), offset))
        offset += len(member_rows)
    arrays["history"] = np.concatenate(history, axis=0)
    arrays["step"] = np.concatenate([np.arange(len(r), dtype=np.int32) for r in rows])
    return Partition(name=f"bootstrap[{len(clusters)} clusters]", arrays=arrays,
                     trajectory_ids=trajectory_ids, episode_ids=episode_ids)


def subset(partition: Partition, episodes: set[str]) -> Partition:
    return materialise(partition, sorted(episodes))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--seed0-checkpoint", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--checkpoint-root", required=True, type=Path)
    ap.add_argument("--manifest-out", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch

    stack = json.loads(args.stack.read_text())
    set_sensor_names(stack["sensor_contract"]["ordered_names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = FrozenSafetyHead.load(args.safety_dir, device=device)

    spec = json.loads(args.partition.read_text())
    train_split = spec["splits"]["gate_training"]
    validation_split = spec["splits"]["checkpoint_validation"]

    full = load_partition(args.cache, "reference_train")
    hazard = np.asarray(full["hazard_present"]).astype(bool)
    trajectory = np.asarray(full["trajectory"])
    episode_hazard = {}
    for index, episode in enumerate(full.episode_ids):
        rows = np.flatnonzero(trajectory == index)
        episode_hazard[episode] = bool(hazard[rows[0]])

    present = sorted(e for e in train_split["episodes"] if episode_hazard[e])
    absent = sorted(e for e in train_split["episodes"] if not episode_hazard[e])
    if (len(present), len(absent)) != (HAZARD_PRESENT_CLUSTERS, HAZARD_ABSENT_CLUSTERS):
        raise SystemExit(f"training split is {len(present)}/{len(absent)}, expected "
                         f"{HAZARD_PRESENT_CLUSTERS}/{HAZARD_ABSENT_CLUSTERS}")

    validation = subset(full, set(validation_split["episodes"]))
    print(f"checkpoint validation: {len(validation)} frames / "
          f"{len(validation.trajectory_ids)} trajectories")

    # the seed-0 deployment model is loaded only to pin its identity here
    seed0_hash = sha256_file(args.seed0_checkpoint)
    seed0_payload = torch.load(args.seed0_checkpoint, map_location="cpu",
                               weights_only=False)
    config_template = seed0_payload["config"]
    if config_template["variant"] != BASELINE_CURRENT or config_template["seed"] != 0:
        raise SystemExit("seed-0 checkpoint is not CURRENT_FRAME_ONLY seed 0")

    members = []
    started = time.time()
    for index, seed in enumerate(BOOTSTRAP_SEEDS):
        clusters = bootstrap_clusters(present, absent, seed)
        multiplicity = Counter(clusters)
        sample = materialise(full, clusters)
        print(f"\n=== member {index} seed {seed}: {len(set(clusters))} unique clusters, "
              f"{len(sample)} frames ===")

        # identical architecture, preprocessing, objective, optimizer and schedule as the
        # frozen seed-0 model; only the training sample and the init differ
        config = TrainConfig(
            variant=BASELINE_CURRENT, hidden=config_template["hidden"],
            blocks=config_template["blocks"], dropout=config_template["dropout"],
            batch_size=config_template["batch_size"],
            active_fraction=config_template["active_fraction"],
            learning_rate=config_template["learning_rate"],
            weight_decay=config_template["weight_decay"],
            max_epochs=config_template["max_epochs"],
            patience=config_template["patience"],
            batches_per_epoch=config_template["batches_per_epoch"],
            seed=seed, weights=LossWeights(**config_template["weights"]))

        directory = args.checkpoint_root / f"member{index}_seed{seed}"
        record = train(config, sample, validation, head, device, directory, log_every=40)
        checkpoint = Path(record["best_checkpoint"])
        if not checkpoint.is_file():
            raise SystemExit(f"member {index} produced no checkpoint")

        # strict reload before the member is accepted
        from causal_parked_skin.engine import load_checkpoint

        reloaded, payload = load_checkpoint(checkpoint, device)
        if payload["config"]["seed"] != seed or \
                payload["config"]["variant"] != BASELINE_CURRENT:
            raise SystemExit(f"member {index} strict reload mismatch")
        scored = evaluate(reloaded, validation, head, device, BASELINE_CURRENT,
                          collect_fields=True)
        del reloaded

        members.append({
            "index": index, "bootstrap_seed": seed,
            "unique_clusters": len(set(clusters)),
            "out_of_bag_clusters": sorted(set(present + absent) - set(clusters)),
            "out_of_bag_count": len(set(present + absent) - set(clusters)),
            "cluster_multiplicity": dict(sorted(multiplicity.items())),
            "hazard_present_draws": HAZARD_PRESENT_CLUSTERS,
            "hazard_absent_draws": HAZARD_ABSENT_CLUSTERS,
            "training_frames": len(sample),
            "training_trajectories": len(sample.trajectory_ids),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "last_checkpoint": record["last_checkpoint"],
            "last_checkpoint_sha256": record["last_checkpoint_sha256"],
            "config_hash": record["config_hash"],
            "parameter_count": record["parameter_count"],
            "best_epoch": record["best_epoch"],
            "epochs_run": record["epochs_run"],
            "validation": {
                "parked_field_mae": scored["pixel"]["all_valid_parked_mae"],
                "changed_pixel_parked_mae": scored["pixel"]["changed_pixel_parked_mae"],
                "parked_head_mae": None,
                "oracle_differential_mae": scored["head"]["differential_mae"],
                "median_direction_cosine_active":
                    scored["head"]["median_direction_cosine_active"],
                "changed_mask_precision": scored.get("mask", {}).get("precision"),
                "changed_mask_recall": scored.get("mask", {}).get("recall"),
                "changed_mask_f1": scored.get("mask", {}).get("f1"),
                "oracle_zero_rms": scored["head"]["oracle_zero_rms"],
                "hazard_absent_rms": scored["head"]["hazard_absent_rms"],
                "hazard_absent_raw_head_rms": scored["head"]["hazard_absent_raw_head_rms"],
                "constraint_violations": scored["constraint_violations"]["total"],
                "nonfinite_outputs": scored["nonfinite_outputs"],
            },
        })
        block = members[-1]["validation"]
        print(f"  -> dq MAE {block['oracle_differential_mae']:.6f} "
              f"cos {block['median_direction_cosine_active']:.4f} "
              f"violations {block['constraint_violations']} "
              f"oob {members[-1]['out_of_bag_count']}")

    if len(members) != MEMBERS:
        raise SystemExit("not all members trained")

    # trivial baseline for the "no material collapse" requirement
    zero = evaluate(None, validation, head, device, "ZERO_DIFFERENTIAL",
                    collect_fields=False)
    zero_mae = zero["head"]["differential_mae"]
    failures = []
    for member in members:
        block = member["validation"]
        if block["nonfinite_outputs"]:
            failures.append(f"member {member['index']} has nonfinite outputs")
        if block["constraint_violations"]:
            failures.append(f"member {member['index']} violates the physical constraint")
        if block["oracle_differential_mae"] >= zero_mae:
            failures.append(f"member {member['index']} collapsed to the trivial baseline")

    manifest = {
        "ensemble_id": ENSEMBLE_ID,
        "members": MEMBERS,
        "bootstrap_seeds": list(BOOTSTRAP_SEEDS),
        "bootstrap_unit": "manifest trajectory/episode cluster",
        "bootstrap_rule": (
            f"{HAZARD_PRESENT_CLUSTERS} hazard-present and {HAZARD_ABSENT_CLUSTERS} "
            "hazard-absent clusters drawn with replacement from the parent training "
            "split; every policy distribution of a sampled cluster is included and "
            "multiplicity is preserved by index repetition"),
        "role": "uncertainty estimation only; never supplies a deployed prediction",
        "architecture": "CURRENT_FRAME_ONLY, identical to the frozen seed-0 model",
        "architecture_config_hash": canonical_hash(config_template),
        "member_records": [
            {k: v for k, v in m.items() if k != "validation"} for m in members],
        "seed0_deployment_checkpoint": str(args.seed0_checkpoint),
        "seed0_deployment_sha256": seed0_hash,
        "checkpoint_validation_episodes": validation_split["episodes"],
        "training_split_episodes": train_split["episodes"],
        "partition_manifest_sha256": spec["manifest_sha256"],
        "sensor_order_sha256": stack["sensor_contract"]["sensor_order_hash"],
        "safety_head_sha256": sha256_file(args.safety_dir / "model.pt"),
        "commits": {"root": git_head()},
        "runtime": {"torch": torch.__version__, "numpy": np.__version__},
        "averaging_permitted": False,
        "member_may_replace_seed0": False,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    report = {
        "schema": "hybrid_obstacle_uncertainty_ensemble_train_v1",
        "ensemble_id": ENSEMBLE_ID,
        "members": members,
        "member_count": len(members),
        "all_members_trained_and_reloaded": not failures,
        "acceptance_failures": failures,
        "zero_differential_validation_mae": zero_mae,
        "acceptance_rule": ("finite outputs, physical constraint validity, strict reload, "
                            "and no collapse to the trivial parked-field baseline; no "
                            "member is dropped for ordinary performance"),
        "members_dropped": 0,
        "members_replaced": 0,
        "diversity": {
            "unique_clusters_per_member": [m["unique_clusters"] for m in members],
            "out_of_bag_clusters_per_member": [m["out_of_bag_count"] for m in members],
            "mean_unique_clusters": float(np.mean([m["unique_clusters"]
                                                   for m in members])),
        },
        "manifest_sha256": manifest["manifest_sha256"],
        "wall_seconds": time.time() - started,
        "decision_if_failed": "UNCERTAINTY_ENSEMBLE_TRAINING_FAILED",
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"\nZERO_DIFFERENTIAL validation MAE {zero_mae:.6f}")
    print(f"members trained: {len(members)}; acceptance failures: {failures or 'none'}")
    print(f"unique clusters per member: {report['diversity']['unique_clusters_per_member']}")
    print(f"manifest sha256: {manifest['manifest_sha256']}")
    print(f"wrote {args.out}")
    return 0 if not failures else 5


if __name__ == "__main__":
    raise SystemExit(main())
