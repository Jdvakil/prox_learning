#!/usr/bin/env python3
"""V10.10 training verification: epochs, checkpoints, strict reload, offline smoke.

Runs after both arms finish. Nothing here can change a trained model; it only
reads the artifacts back and states what is true of them.

The PACT proximity-consumption proof is causal rather than structural: the same
fixed batch is run twice, once with the real frozen embeddings and once with
them zeroed. A policy that ignores its proximity tokens returns identical
actions. ACT, which is built with ``n_proximity_sensors=0``, is checked for the
complementary property -- it must be bit-identical across repeated runs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "submodules" / "act"))

from pact_place_v109_contract import PROXIMITY_FEATURE_DIM, TRAIN_PARAMS  # noqa: E402
from pact_place_v1010_contract import (  # noqa: E402
    CONTRACT_VERSION_V1010,
    CONVERTED_DATASET_ROOT,
    ENCODER_SHA256,
    TRAINING_ROOT,
    WORK_ROOT,
    canonical_payload_sha256,
    empty_authorization,
    sha256_file,
    write_immutable_create_only,
)

ARMS = ("act", "pact")
SMOKE_BATCH = 8
SMOKE_TIMESTEP = 40
EXPECTED_EPOCHS = int(TRAIN_PARAMS["num_epochs"])
CHUNK = int(TRAIN_PARAMS["chunk_size"])
ACTION_DIM = int(TRAIN_PARAMS["action_dim"])


@contextmanager
def detr_argv(checkpoint_dir: str, seed: int):
    original = sys.argv
    sys.argv = [original[0], "--ckpt_dir", checkpoint_dir, "--policy_class", "ACT",
                "--task_name", "obstacle_baseline", "--seed", str(seed),
                "--num_epochs", "1"]
    try:
        yield
    finally:
        sys.argv = original


def smoke_batch(split: dict[str, Any], stats: dict[str, Any], want_proximity: bool):
    """A fixed batch from the first validation episodes, built identically for both arms."""
    import torch  # noqa: PLC0415

    dataset_dir = ROOT / CONVERTED_DATASET_ROOT
    indices = sorted(
        e["act_episode_index"] for e in split["episodes"] if e["split"] == "validation"
    )[:SMOKE_BATCH]
    images, qpos_rows, proximity_rows = [], [], []
    for index in indices:
        with h5py.File(dataset_dir / f"episode_{index}.hdf5", "r") as handle:
            timestep = min(SMOKE_TIMESTEP, int(handle["action"].shape[0]) - 1)
            images.append(np.asarray(
                handle["observations/images/wrist_camera"][timestep], dtype=np.uint8))
            qpos_rows.append(np.asarray(
                handle["observations/qpos"][timestep], dtype=np.float32))
            if want_proximity:
                proximity_rows.append(np.asarray(
                    handle["observations/proximity_embeddings"][timestep],
                    dtype=np.float32))
    image = torch.from_numpy(np.stack(images)).float().unsqueeze(1) / 255.0
    image = torch.einsum("b k h w c -> b k c h w", image)
    qpos = torch.from_numpy(np.stack(qpos_rows)).float()
    qpos = (qpos - stats["qpos_mean"]) / stats["qpos_std"]
    proximity = (torch.from_numpy(np.stack(proximity_rows)).float()
                 if want_proximity else None)
    return indices, image, qpos, proximity


def verify_arm(arm: str, split: dict[str, Any]) -> dict[str, Any]:
    import torch  # noqa: PLC0415
    from policy import ACTPolicy  # noqa: PLC0415

    problems: list[str] = []
    directory = Path(TRAINING_ROOT) / f"{arm}_seed3101"
    manifest = json.loads((directory / "run_manifest.json").read_text())

    epochs = [json.loads(line) for line in
              (directory / "epoch_log.jsonl").read_text().splitlines() if line.strip()]
    if len(epochs) != EXPECTED_EPOCHS:
        problems.append(f"{arm}: {len(epochs)} epoch records, expected {EXPECTED_EPOCHS}")
    if epochs and int(epochs[-1]["epoch"]) != EXPECTED_EPOCHS - 1:
        problems.append(f"{arm}: last epoch is {epochs[-1]['epoch']}")
    if [int(e["epoch"]) for e in epochs] != list(range(len(epochs))):
        problems.append(f"{arm}: epoch indices are not a dense 0..N-1 range")
    best = min(epochs, key=lambda e: float(e["val"]["loss"])) if epochs else {}
    recorded_best = int(epochs[-1]["best_epoch"]) if epochs else -1
    if best and int(best["epoch"]) != recorded_best:
        problems.append(
            f"{arm}: recomputed best epoch {best['epoch']} != recorded {recorded_best}")

    checkpoints = {
        name: sha256_file(directory / name)
        for name in ("policy_best.ckpt", "policy_last.ckpt", "dataset_stats.pkl",
                     "run_manifest.json")
        if (directory / name).is_file()
    }
    for required in ("policy_best.ckpt", "policy_last.ckpt", "dataset_stats.pkl"):
        if required not in checkpoints:
            problems.append(f"{arm}: missing {required}")

    with (directory / "dataset_stats.pkl").open("rb") as stream:
        stats = pickle.load(stream)

    use_proximity = arm == "pact"
    config = {
        "lr": 1e-5, "num_queries": CHUNK, "kl_weight": int(TRAIN_PARAMS["kl_weight"]),
        "hidden_dim": int(TRAIN_PARAMS["hidden_dim"]),
        "dim_feedforward": int(TRAIN_PARAMS["dim_feedforward"]),
        "lr_backbone": 1e-5, "backbone": "resnet18",
        "enc_layers": int(TRAIN_PARAMS["enc_layers"]),
        "dec_layers": int(TRAIN_PARAMS["dec_layers"]), "nheads": 8,
        "camera_names": ["wrist_camera"],
        "state_dim": int(TRAIN_PARAMS["state_dim"]), "action_dim": ACTION_DIM,
        "n_proximity_sensors": 40 if use_proximity else 0,
        "prox_tokens_per_sensor": 1,
        "proximity_feature_dim": PROXIMITY_FEATURE_DIM if use_proximity else 3,
    }
    with detr_argv(str(directory), int(TRAIN_PARAMS["seed"])):
        policy = ACTPolicy(config)
    state = torch.load(directory / "policy_best.ckpt", map_location="cpu")
    strict_reload_ok = True
    try:
        policy.load_state_dict(state, strict=True)
    except Exception as exc:  # noqa: BLE001 - a reload failure is the finding
        strict_reload_ok = False
        problems.append(f"{arm}: strict reload failed: {exc!r}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device).eval()

    projection = getattr(policy.model, "input_proj_proximity", None)
    projection_shape = list(projection.weight.shape) if projection is not None else None
    if use_proximity and projection_shape != [512, PROXIMITY_FEATURE_DIM]:
        problems.append(f"pact: input_proj_proximity shape {projection_shape}")

    indices, image, qpos, proximity = smoke_batch(split, stats, use_proximity)
    image, qpos = image.to(device), qpos.to(device)
    if proximity is not None:
        proximity = proximity.to(device)
    with torch.inference_mode():
        actions = policy(qpos, image, proximity_positions=proximity)
        repeat = policy(qpos, image, proximity_positions=proximity)
        if use_proximity:
            zeroed = policy(qpos, image, proximity_positions=torch.zeros_like(proximity))
        else:
            zeroed = None

    shape_ok = tuple(actions.shape) == (len(indices), CHUNK, ACTION_DIM)
    if not shape_ok:
        problems.append(f"{arm}: smoke output shape {tuple(actions.shape)}")
    finite = bool(torch.isfinite(actions).all())
    if not finite:
        problems.append(f"{arm}: smoke output is not finite")
    action_std = float(actions.float().std())
    per_sample_std = float(actions.float().std(dim=(1, 2)).min())
    if action_std <= 1e-6:
        problems.append(f"{arm}: smoke actions are constant (std {action_std})")
    deterministic = bool(torch.equal(actions, repeat))
    if not deterministic:
        problems.append(f"{arm}: inference is not deterministic across repeated calls")

    proximity_proof: dict[str, Any] = {"applicable": use_proximity}
    if use_proximity:
        delta = (actions - zeroed).abs()
        max_delta = float(delta.max())
        mean_delta = float(delta.mean())
        consumed = max_delta > 1e-5
        proximity_proof.update({
            "method": "same batch with real vs zeroed frozen embeddings",
            "max_abs_action_delta": max_delta,
            "mean_abs_action_delta": mean_delta,
            "proximity_consumed": consumed,
        })
        if not consumed:
            problems.append(
                "pact: zeroing the proximity embeddings did not change the actions; "
                "proximity is not being consumed")
        if manifest.get("proximity_consumed") is not True:
            problems.append("pact: run manifest does not record proximity_consumed")
        if manifest.get("surface_encoder_sha256") != ENCODER_SHA256:
            problems.append("pact: run manifest encoder hash differs from the contract")
    else:
        if manifest.get("proximity_consumed") is not False:
            problems.append("act: run manifest claims proximity was consumed")
        if int(manifest.get("n_proximity_sensors", -1)) != 0:
            problems.append("act: run manifest reports proximity sensors")

    return {
        "arm": arm,
        "checkpoint_dir": str(directory),
        "epochs_recorded": len(epochs),
        "epochs_expected": EXPECTED_EPOCHS,
        "completed_all_epochs": len(epochs) == EXPECTED_EPOCHS,
        "best_epoch": int(best["epoch"]) if best else None,
        "best_val_loss": float(best["val"]["loss"]) if best else None,
        "best_val_l1": float(best["val"]["l1"]) if best else None,
        "best_val_kl": float(best["val"]["kl"]) if best else None,
        "final_epoch_val_loss": float(epochs[-1]["val"]["loss"]) if epochs else None,
        "final_epoch_train_loss": float(epochs[-1]["train"]["loss"]) if epochs else None,
        "recorded_best_epoch": recorded_best,
        "hashes": checkpoints,
        "run_manifest_sha256": manifest.get("run_manifest_sha256"),
        "split_manifest_sha256": manifest.get("split_manifest_sha256"),
        "dataset_tree_sha256": (manifest.get("dataset_report") or {}).get("tree_sha256"),
        "strict_reload_ok": strict_reload_ok,
        "input_proj_proximity_shape": projection_shape,
        "offline_smoke": {
            "episodes": indices,
            "timestep": SMOKE_TIMESTEP,
            "batch": len(indices),
            "output_shape": list(actions.shape),
            "output_shape_expected": [len(indices), CHUNK, ACTION_DIM],
            "shape_ok": shape_ok,
            "finite": finite,
            "action_std": action_std,
            "min_per_sample_std": per_sample_std,
            "nonconstant": action_std > 1e-6,
            "deterministic_repeat": deterministic,
        },
        "proximity_proof": proximity_proof,
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / WORK_ROOT / "training_verification.json")
    args = parser.parse_args()

    split = json.loads((ROOT / WORK_ROOT / "split_manifest.json").read_text())
    preflight = json.loads((ROOT / WORK_ROOT / "training_preflight.json").read_text())
    timing_path = Path(TRAINING_ROOT) / "training_timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.is_file() else {}

    arms = {arm: verify_arm(arm, split) for arm in ARMS}
    problems = [p for arm in arms.values() for p in arm["problems"]]

    shared = {
        "split_manifest_sha256": {a: arms[a]["split_manifest_sha256"] for a in ARMS},
        "dataset_tree_sha256": {a: arms[a]["dataset_tree_sha256"] for a in ARMS},
    }
    for key, values in shared.items():
        if len(set(values.values())) != 1:
            problems.append(f"arms disagree on {key}: {values}")
    if arms["act"]["hashes"].get("policy_best.ckpt") == \
            arms["pact"]["hashes"].get("policy_best.ckpt"):
        problems.append("ACT and PACT produced an identical best checkpoint")

    document: dict[str, Any] = {
        **empty_authorization(),
        "schema_version": "pact_place_v1010_training_verification_v1",
        "contract_version": CONTRACT_VERSION_V1010,
        "role": "post-training verification of the V10.9 ACT/PACT pair",
        "is_phase0_pass": False,
        "training_root": TRAINING_ROOT,
        "preflight_payload_sha256": preflight["payload_sha256"],
        "command_diff": preflight["command_diff"],
        "timing": timing,
        "shared_bindings": shared,
        "arms": arms,
        "problems": problems,
        "verified": not problems,
        "cleared_for_live_evaluation": not problems,
    }
    document["payload_sha256"] = canonical_payload_sha256(document)
    written = write_immutable_create_only(args.out, document)
    print(json.dumps({
        "verified": document["verified"],
        "problems": problems,
        "act": {k: arms["act"][k] for k in
                ("completed_all_epochs", "best_epoch", "best_val_loss",
                 "strict_reload_ok")},
        "pact": {k: arms["pact"][k] for k in
                 ("completed_all_epochs", "best_epoch", "best_val_loss",
                  "strict_reload_ok")},
        "pact_proximity_proof": arms["pact"]["proximity_proof"],
        "payload_sha256": document["payload_sha256"],
        "raw_file_sha256": written.get("raw_file_sha256"),
    }, indent=2))
    return 0 if document["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
