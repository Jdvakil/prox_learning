#!/usr/bin/env python3
"""Train PROX_EVIDENCE_ACTIVITY_GATE_V1 exactly once, seed 0.

Handoff steps 7-10. Every hyperparameter is fixed by the handoff and asserted here; the run
happens once and is not restarted with another seed after the results are seen.

Two choices in the objective are doing the real work:

* onset-zero frames carry weight 4.0 rather than 1.0, because they are the failure mode and
  a frame-uniform loss lets them be averaged away against ~13,000 ordinary quiet frames;
* the onset penalty is a *per-trajectory maximum*, not a mean. The failure this gate exists
  to remove was seven consecutive activations in one trajectory. A mean over onset frames
  would score that identically to seven isolated single-frame errors spread across seven
  trajectories, which is a far less dangerous thing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin import threshold as thr
from causal_parked_skin.activity_gate import GATE_ID, PARAMETER_BUDGET, build_gate
from causal_parked_skin.data import SOURCE_MODES, load_partition

# fixed by the handoff; asserted rather than configurable
SEED = 0
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256
MAX_EPOCHS = 80
GRAD_CLIP = 1.0
DROPOUT = 0.0
BATCHES_PER_EPOCH = 100

WEIGHT_ACTIVE = 1.0
WEIGHT_ZERO = 1.0
WEIGHT_ONSET_ZERO = 4.0
ONSET_PENALTY_WEIGHT = 1.0

ONSET_MIN_FRAMES = 10
ONSET_FRACTION = 0.10


def onset_cutoff(length: int) -> int:
    return max(ONSET_MIN_FRAMES, int(np.ceil(ONSET_FRACTION * length)))


class GateFrames:
    """Frame index for one split, with everything the sampler needs."""

    def __init__(self, partition, episodes: set[str]) -> None:
        trajectory = np.asarray(partition["trajectory"])
        keep_traj = [i for i, episode in enumerate(partition.episode_ids)
                     if episode in episodes]
        keep = np.isin(trajectory, keep_traj)
        self.rows = np.flatnonzero(keep)
        self.trajectory = trajectory[self.rows]
        self.step = np.asarray(partition["step"])[self.rows]
        self.distribution = np.asarray(partition["source_mode"])[self.rows]
        self.hazard = np.asarray(partition["hazard_present"])[self.rows].astype(bool)
        # label: any pixel changes between the current and parked fields
        changed = np.asarray(partition["changed"])
        self.label = changed[self.rows].reshape(len(self.rows), -1).any(axis=1)
        lengths = {t: int((trajectory == t).sum()) for t in keep_traj}
        self.length = np.array([lengths[t] for t in self.trajectory])
        self.onset = self.step < np.array([onset_cutoff(n) for n in self.length])
        self.partition = partition
        self.trajectory_ids = {t: partition.trajectory_ids[t] for t in keep_traj}
        self.episode_ids = {t: partition.episode_ids[t] for t in keep_traj}

        # sampling pools: distribution -> trajectory -> class -> row positions
        self.pools: dict[int, dict[int, dict[str, np.ndarray]]] = {}
        for mode in range(len(SOURCE_MODES)):
            in_mode = self.distribution == mode
            trajectories = np.unique(self.trajectory[in_mode])
            if trajectories.size == 0:
                continue
            self.pools[mode] = {}
            for t in trajectories:
                here = in_mode & (self.trajectory == t)
                self.pools[mode][int(t)] = {
                    "active": np.flatnonzero(here & self.label),
                    "onset_zero": np.flatnonzero(here & ~self.label & self.onset),
                    "later_zero": np.flatnonzero(here & ~self.label & ~self.onset),
                }

    def __len__(self) -> int:
        return len(self.rows)

    def tensors(self, positions, device):
        import torch

        global_index = self.rows[positions]
        closeness = np.asarray(self.partition["current"])[global_index]
        valid = np.asarray(self.partition["current_valid"])[global_index]
        return (torch.from_numpy(np.ascontiguousarray(closeness)).to(device),
                torch.from_numpy(np.ascontiguousarray(valid)).to(device),
                torch.from_numpy(self.label[positions].astype(np.float32)).to(device))


def sample_batch(frames: GateFrames, rng) -> np.ndarray:
    """Trajectory-balanced sampling in the predeclared order."""
    picks = []
    modes = sorted(frames.pools)
    while len(picks) < BATCH_SIZE:
        mode = modes[rng.integers(len(modes))]                       # 1. distribution
        trajectories = sorted(frames.pools[mode])
        t = trajectories[rng.integers(len(trajectories))]            # 2. trajectory
        pool = frames.pools[mode][t]
        if rng.random() < 0.5:                                       # 3. activity class
            group = pool["active"]
        else:
            key = "onset_zero" if rng.random() < 0.5 else "later_zero"   # 4. onset split
            group = pool[key]
            if group.size == 0:
                group = pool["onset_zero"] if key == "later_zero" else pool["later_zero"]
        if group.size == 0:
            continue
        picks.append(int(group[rng.integers(group.size)]))           # 5. real frame
    return np.array(picks)


def weighted_bce(logits, target, onset, active):
    import torch
    import torch.nn.functional as F

    weight = torch.where(active.bool(), torch.full_like(target, WEIGHT_ACTIVE),
                         torch.where(onset.bool(),
                                     torch.full_like(target, WEIGHT_ONSET_ZERO),
                                     torch.full_like(target, WEIGHT_ZERO)))
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (raw * weight).sum() / weight.sum().clamp(min=1.0)


def onset_max_penalty(logits, onset, active, trajectory):
    """Mean over represented trajectories of the max activity on their onset-zero frames."""
    import torch

    mask = onset.bool() & ~active.bool()
    if not bool(mask.any()):
        return logits.sum() * 0.0
    probability = torch.sigmoid(logits)[mask]
    ids = trajectory[mask]
    unique = torch.unique(ids)
    maxima = torch.stack([probability[ids == t].max() for t in unique])
    return maxima.mean()


def trajectory_balanced_bce(gate, frames: GateFrames, device) -> float:
    """Per-trajectory BCE averaged across trajectories, not pooled across frames."""
    import torch
    import torch.nn.functional as F

    gate.eval()
    losses = []
    with torch.no_grad():
        for mode in sorted(frames.pools):
            for t in sorted(frames.pools[mode]):
                positions = np.concatenate([frames.pools[mode][t][k] for k in
                                            ("active", "onset_zero", "later_zero")])
                if positions.size == 0:
                    continue
                total, count = 0.0, 0
                for start in range(0, positions.size, 512):
                    chunk = positions[start:start + 512]
                    closeness, valid, label = frames.tensors(chunk, device)
                    logits = gate(closeness, valid)
                    total += float(F.binary_cross_entropy_with_logits(
                        logits, label, reduction="sum"))
                    count += chunk.size
                losses.append(total / max(count, 1))
    gate.train()
    return float(np.mean(losses)) if losses else float("inf")


def atomic_save(payload, path: Path) -> str:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with open(temporary, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--checkpoint-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch

    partition_spec = json.loads(args.partition.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    train_partition = load_partition(args.cache, "reference_train")
    training = GateFrames(train_partition,
                          set(partition_spec["splits"]["gate_training"]["episodes"]))
    validation = GateFrames(
        train_partition,
        set(partition_spec["splits"]["checkpoint_validation"]["episodes"]))
    print(f"gate training frames  : {len(training)}  "
          f"(active {int(training.label.sum())}, "
          f"onset-zero {int((~training.label & training.onset).sum())})")
    print(f"checkpoint validation : {len(validation)} frames")

    gate = build_gate().to(device)
    parameters = gate.parameter_count()
    if parameters > PARAMETER_BUDGET:
        raise SystemExit(f"{parameters} parameters exceeds {PARAMETER_BUDGET}")
    optimizer = torch.optim.AdamW(gate.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)

    history = []
    best = {"epoch": -1, "metric": float("inf"), "sha256": None}
    started = time.time()
    for epoch in range(MAX_EPOCHS):
        gate.train()
        epoch_bce = epoch_onset = 0.0
        for _ in range(BATCHES_PER_EPOCH):
            positions = sample_batch(training, rng)
            closeness, valid, label = training.tensors(positions, device)
            onset = torch.from_numpy(
                training.onset[positions].astype(np.float32)).to(device)
            trajectory = torch.from_numpy(
                training.trajectory[positions].astype(np.int64)).to(device)
            logits = gate(closeness, valid)
            bce = weighted_bce(logits, label, onset, label)
            penalty = onset_max_penalty(logits, onset, label, trajectory)
            loss = bce + ONSET_PENALTY_WEIGHT * penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_bce += float(bce.detach())
            epoch_onset += float(penalty.detach())

        metric = trajectory_balanced_bce(gate, validation, device)
        history.append({"epoch": epoch, "train_bce": epoch_bce / BATCHES_PER_EPOCH,
                        "train_onset_penalty": epoch_onset / BATCHES_PER_EPOCH,
                        "validation_trajectory_balanced_bce": metric})
        improved = metric < best["metric"] - 1e-9
        if improved:
            best = {"epoch": epoch, "metric": metric}
            best["sha256"] = atomic_save({
                "gate": gate.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "seed": SEED, "parameter_count": parameters,
                "gate_id": GATE_ID, "validation_metric": metric,
                "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
            }, args.checkpoint_dir / "gate_best.pt")
        if epoch % 10 == 0 or improved:
            print(f"  epoch {epoch:>3} bce={history[-1]['train_bce']:.5f} "
                  f"onset={history[-1]['train_onset_penalty']:.5f} "
                  f"val={metric:.5f} {'*' if improved else ''}", flush=True)

    last_hash = atomic_save({
        "gate": gate.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": MAX_EPOCHS - 1, "seed": SEED, "parameter_count": parameters,
        "gate_id": GATE_ID, "validation_metric": history[-1][
            "validation_trajectory_balanced_bce"],
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
    }, args.checkpoint_dir / "gate_last.pt")

    report = {
        "schema": "hybrid_obstacle_prox_activity_train_v1",
        "gate_id": GATE_ID,
        "parameter_count": parameters,
        "parameter_budget": PARAMETER_BUDGET,
        "partition_manifest_sha256": partition_spec["manifest_sha256"],
        "label_definition": "activity_target = any changed_pixel_mask value is true",
        "onset_definition": (f"episode_step < max({ONSET_MIN_FRAMES}, "
                             f"ceil({ONSET_FRACTION} * trajectory_length))"),
        "sampling": {
            "order": ["distribution uniform", "trajectory uniform within distribution",
                      "activity class 50/50", "zero split 50/50 onset/non-onset",
                      "frame uniform within group"],
            "global_frame_count_weighting": False,
            "replacement": "within the selected trajectory group only",
        },
        "loss": {
            "weights": {"oracle_active": WEIGHT_ACTIVE, "ordinary_zero": WEIGHT_ZERO,
                        "onset_zero": WEIGHT_ONSET_ZERO},
            "dynamic_positive_weight": False,
            "focal_loss": False,
            "onset_penalty": ("mean over represented trajectories of the max predicted "
                              "activity on their onset-zero frames"),
            "onset_penalty_weight": ONSET_PENALTY_WEIGHT,
        },
        "optimization": {
            "seed": SEED, "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS, "batches_per_epoch": BATCHES_PER_EPOCH,
            "gradient_clipping": GRAD_CLIP, "dropout": DROPOUT,
            "checkpoint_rule": ("minimum trajectory-balanced BCE on the "
                                "checkpoint-validation episodes"),
            "training_runs": 1,
            "restarted_with_another_seed": False,
        },
        "frames": {
            "gate_training": len(training),
            "gate_training_active": int(training.label.sum()),
            "gate_training_onset_zero": int((~training.label & training.onset).sum()),
            "checkpoint_validation": len(validation),
        },
        "best_epoch": best["epoch"],
        "best_validation_metric": best["metric"],
        "best_checkpoint": str(args.checkpoint_dir / "gate_best.pt"),
        "best_checkpoint_sha256": best["sha256"],
        "last_checkpoint": str(args.checkpoint_dir / "gate_last.pt"),
        "last_checkpoint_sha256": last_hash,
        "history": history,
        "wall_seconds": time.time() - started,
    }
    report["report_sha256"] = thr.canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nbest epoch {best['epoch']} val {best['metric']:.6f}")
    print(f"parameters {parameters:,}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
