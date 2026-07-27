"""Loading, caching and sampling for the frozen parked-skin supervision dataset.

The frozen dataset is read-only and is never written to. Reading 364 gzip-compressed HDF5
files takes minutes, and the bounded selection plus three-seed schedule reads them ten
times over, so the arrays are consolidated once into a cache outside the dataset tree and
memory-mapped from there afterwards.

Two contract rules are enforced here rather than left to the training loop:

* the four-frame history is reconstructed from the contiguous current-field sequence by
  ``history(t) = [t-3, t-2, t-1, t]``, left-padded by repeating the earliest frame in the
  *same* trajectory, so no window can reach a future frame or cross a boundary;
* the deployable loader returns privileged arrays only to callers that ask for targets.
  Model input is assembled from the deployable group alone.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CAUSAL_FRAMES = 4
SENSORS = 40
PIXEL_ROWS = 8
PIXEL_COLS = 8
STATE_WIDTH = 29

PARTITIONS = ("reference_train", "reference_validation", "reference_calibration",
              "offline_reference_test")

SOURCE_MODES = ("EXPERT_RECONSTRUCTED", "ACT_ONLY_ON_POLICY", "ORACLE_ON_POLICY",
                "LEARNER_INDUCED_ON_POLICY")
MODE_INDEX = {name: i for i, name in enumerate(SOURCE_MODES)}

# assembled in this order into the 29-wide deployable state vector
STATE_LAYOUT = (("qpos", 9), ("qvel", 9), ("nominal_action", 8),
                ("gripper_state", 2), ("gripper_command", 1))

# arrays cached per partition: name -> (dtype, per-frame shape)
CACHE_SPEC = {
    "current": ("float32", (SENSORS, PIXEL_ROWS, PIXEL_COLS)),
    "current_valid": ("bool", (SENSORS, PIXEL_ROWS, PIXEL_COLS)),
    "parked": ("float32", (SENSORS, PIXEL_ROWS, PIXEL_COLS)),
    "parked_valid": ("bool", (SENSORS, PIXEL_ROWS, PIXEL_COLS)),
    "changed": ("bool", (SENSORS, PIXEL_ROWS, PIXEL_COLS)),
    "state": ("float32", (STATE_WIDTH,)),
    "current_head": ("float32", (7,)),
    "parked_head": ("float32", (7,)),
    "oracle_dq": ("float32", (7,)),
    "oracle_active": ("bool", ()),
    "hazard_present": ("bool", ()),
    "source_mode": ("int8", ()),
    "trajectory": ("int32", ()),
    "step": ("int32", ()),
    "history": ("int32", (CAUSAL_FRAMES,)),
}

DEPLOYABLE_INPUTS = ("current", "current_valid", "state", "history")
PRIVILEGED_TARGETS = ("parked", "parked_valid", "changed", "parked_head", "oracle_dq")


class ParkedSkinDataError(RuntimeError):
    """The dataset on disk disagrees with the contract this module assumes."""


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def causal_history_indices(length: int, offset: int = 0) -> np.ndarray:
    """(T, 4) absolute indices, left-padded inside the trajectory.

    ``max(0, t - k)`` clamps into the trajectory rather than wrapping to its end, which is
    what makes the padding causal instead of a silent look-ahead to the last frame.
    """
    steps = np.arange(length, dtype=np.int64)
    lags = np.arange(CAUSAL_FRAMES - 1, -1, -1, dtype=np.int64)
    idx = np.maximum(0, steps[:, None] - lags[None, :])
    return (idx + offset).astype(np.int32)


def sensor_link_ids(sensor_names) -> tuple[np.ndarray, list[str]]:
    """Map each sensor onto its parent link, keeping the frozen sensor order."""
    link_names = [n.rsplit("_sensor_", 1)[0] for n in sensor_names]
    links = sorted(set(link_names))
    lookup = {name: i for i, name in enumerate(links)}
    return np.array([lookup[n] for n in link_names], dtype=np.int64), links


@dataclass
class Partition:
    """One partition's arrays. Privileged arrays are present but never fed as input."""

    name: str
    arrays: dict[str, np.ndarray]
    trajectory_ids: list[str]
    episode_ids: list[str]

    def __len__(self) -> int:
        return len(self.arrays["current"])

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    @property
    def frames(self) -> int:
        return len(self)

    def deployable(self) -> dict[str, np.ndarray]:
        return {k: self.arrays[k] for k in DEPLOYABLE_INPUTS}

    def gather_history(self, index: np.ndarray) -> np.ndarray:
        """(B, 4, 40, 8, 8) closeness windows for the given frame indices."""
        return self.arrays["current"][self.arrays["history"][index]]

    def gather_history_valid(self, index: np.ndarray) -> np.ndarray:
        return self.arrays["current_valid"][self.arrays["history"][index]]


def build_cache(manifest_path: Path, cache_dir: Path, *, verbose: bool = True) -> dict:
    """Consolidate the frozen files into per-partition arrays. Never writes to the dataset."""
    import h5py

    manifest = json.loads(Path(manifest_path).read_text())
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    by_partition: dict[str, list] = {p: [] for p in PARTITIONS}
    for entry in manifest["entries"]:
        by_partition.setdefault(entry["partition"], []).append(entry)

    summary = {}
    for partition, entries in by_partition.items():
        if partition not in PARTITIONS:
            raise ParkedSkinDataError(f"unexpected partition {partition!r}")
        entries = sorted(entries, key=lambda e: (e["distribution"], e["episode_id"]))
        blocks: dict[str, list] = {k: [] for k in CACHE_SPEC}
        trajectory_ids, episode_ids = [], []
        offset = 0
        for entry in entries:
            with h5py.File(entry["output"], "r") as handle:
                frames = int(handle.attrs["frames"])
                current = handle["deployable/current_closeness"][:]
                parked = handle["privileged/parked_closeness"][:]
                if current.shape != (frames, SENSORS, PIXEL_ROWS, PIXEL_COLS):
                    raise ParkedSkinDataError(f"{entry['episode_id']}: {current.shape}")
                # the physical contract, re-checked at load rather than trusted
                if (parked > current + 1e-7).any():
                    raise ParkedSkinDataError(
                        f"{entry['episode_id']}: parked exceeds current")
                blocks["current"].append(current)
                blocks["parked"].append(parked)
                blocks["current_valid"].append(handle["deployable/current_valid_mask"][:])
                blocks["parked_valid"].append(handle["privileged/parked_valid_mask"][:])
                blocks["changed"].append(handle["privileged/changed_pixel_mask"][:])
                blocks["current_head"].append(handle["privileged/current_head"][:])
                blocks["parked_head"].append(handle["privileged/parked_head"][:])
                blocks["oracle_dq"].append(handle["privileged/oracle_dq"][:])
                blocks["oracle_active"].append(handle["privileged/oracle_active"][:])
                state = np.concatenate(
                    [np.asarray(handle[f"deployable/{name}"][:], dtype=np.float32
                                ).reshape(frames, -1) for name, _ in STATE_LAYOUT], axis=1)
                if state.shape[1] != STATE_WIDTH:
                    raise ParkedSkinDataError(f"state width {state.shape[1]}")
                blocks["state"].append(state)
                blocks["hazard_present"].append(
                    np.full(frames, bool(handle.attrs["hazard_present"]), dtype=bool))
                blocks["source_mode"].append(np.full(
                    frames, MODE_INDEX[str(handle.attrs["distribution"])], dtype=np.int8))
                blocks["trajectory"].append(
                    np.full(frames, len(trajectory_ids), dtype=np.int32))
                blocks["step"].append(np.arange(frames, dtype=np.int32))
                blocks["history"].append(causal_history_indices(frames, offset))
                trajectory_ids.append(str(handle.attrs["trajectory_id"]))
                episode_ids.append(str(handle.attrs["episode_id"]))
                offset += frames

        arrays = {k: np.concatenate(v, axis=0) for k, v in blocks.items()}
        for name, array in arrays.items():
            np.save(cache_dir / f"{partition}__{name}.npy", array)
        meta = {"frames": int(offset), "trajectories": len(trajectory_ids),
                "trajectory_ids": trajectory_ids, "episode_ids": episode_ids}
        (cache_dir / f"{partition}__meta.json").write_text(json.dumps(meta))
        summary[partition] = {"frames": meta["frames"],
                              "trajectories": meta["trajectories"]}
        if verbose:
            print(f"  cached {partition:<24} {meta['frames']:>6} frames "
                  f"/ {meta['trajectories']:>3} trajectories")

    index = {"dataset_manifest_sha256": manifest["manifest_sha256"],
             "dataset_version": manifest["dataset_version"],
             "partitions": summary, "cache_spec": {k: list(v) for k, v in
                                                   CACHE_SPEC.items()}}
    (cache_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    return index


def load_partition(cache_dir: Path, partition: str, *, mmap: bool = True) -> Partition:
    cache_dir = Path(cache_dir)
    mode = "r" if mmap else None
    arrays = {name: np.load(cache_dir / f"{partition}__{name}.npy", mmap_mode=mode)
              for name in CACHE_SPEC}
    meta = json.loads((cache_dir / f"{partition}__meta.json").read_text())
    return Partition(name=partition, arrays=arrays,
                     trajectory_ids=meta["trajectory_ids"],
                     episode_ids=meta["episode_ids"])


# --------------------------------------------------------------------------- sampling
STRATA = ("active_hazard_present", "zero_hazard_present", "zero_hazard_absent",
          "active_hazard_absent")


def frame_strata(partition: Partition) -> np.ndarray:
    """Stratum index per frame. Hazard-absent frames are an exact zero control."""
    active = np.asarray(partition["oracle_active"])
    hazard = np.asarray(partition["hazard_present"])
    strata = np.where(active & hazard, 0, np.where(~active & hazard, 1,
                                                   np.where(~active & ~hazard, 2, 3)))
    return strata.astype(np.int64)


class StratifiedBatchSampler:
    """Deterministic stratified sampler over oracle-active / zero / hazard strata.

    Zero frames are never removed from the dataset -- every frame stays addressable, and
    evaluation always runs the natural distribution. This only changes which frames a
    *training* batch sees, so gradient updates are not swamped by the 76% majority class.

    Source mode is used as a secondary stratum so a batch cannot end up drawn from one
    distribution: expert, ACT-only, oracle and learner-induced frames differ in posture
    statistics, and a mode-homogeneous batch would make the batch-norm-free encoder chase
    whichever mode dominates.
    """

    def __init__(self, partition: Partition, batch_size: int, *,
                 active_fraction: float = 0.5, seed: int = 0,
                 batches_per_epoch: int | None = None) -> None:
        if not 0.0 < active_fraction < 1.0:
            raise ValueError("active_fraction must lie strictly inside (0, 1)")
        self.partition = partition
        self.batch_size = int(batch_size)
        self.active_fraction = float(active_fraction)
        self.seed = int(seed)
        strata = frame_strata(partition)
        modes = np.asarray(partition["source_mode"])
        self.pools: dict[tuple[int, int], np.ndarray] = {}
        for stratum in range(len(STRATA)):
            for mode in range(len(SOURCE_MODES)):
                hits = np.flatnonzero((strata == stratum) & (modes == mode))
                if hits.size:
                    self.pools[(stratum, mode)] = hits
        self.active_pools = [k for k in self.pools if k[0] in (0, 3)]
        self.zero_pools = [k for k in self.pools if k[0] in (1, 2)]
        if not self.active_pools or not self.zero_pools:
            raise ParkedSkinDataError("a stratum is empty; cannot build balanced batches")
        self.natural_active_prevalence = float(
            np.asarray(partition["oracle_active"]).mean())
        self.batches_per_epoch = int(
            batches_per_epoch if batches_per_epoch is not None
            else max(1, len(partition) // self.batch_size))

    def epoch(self, epoch_index: int) -> list[np.ndarray]:
        """Deterministic given (seed, epoch_index): reproducible without storing indices."""
        rng = np.random.default_rng((self.seed, epoch_index))
        n_active = max(1, round(self.batch_size * self.active_fraction))
        n_zero = self.batch_size - n_active
        batches = []
        for _ in range(self.batches_per_epoch):
            picks = []
            for count, pools in ((n_active, self.active_pools), (n_zero, self.zero_pools)):
                order = rng.permutation(len(pools))
                for slot in range(count):
                    pool = self.pools[pools[order[slot % len(pools)]]]
                    picks.append(pool[rng.integers(pool.size)])
            batches.append(np.sort(np.asarray(picks, dtype=np.int64)))
        return batches

    def prevalence_report(self) -> dict:
        return {
            "natural_active_prevalence": self.natural_active_prevalence,
            "sampled_active_prevalence": self.active_fraction,
            "strata": {name: int((frame_strata(self.partition) == i).sum())
                       for i, name in enumerate(STRATA)},
            "secondary_stratum": "source_mode",
            "zero_frames_removed": False,
            "zero_frames_subsampled_from_dataset": False,
            "evaluation_uses_natural_distribution": True,
            "metrics_are": "population-weighted over the natural partition distribution",
        }
