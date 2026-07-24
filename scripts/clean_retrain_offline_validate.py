#!/usr/bin/env python3
"""Offline acceptance gate for the clean-retrained vanilla ACT obstacle bundle.

Runs every check the approved manifest requires *before* any simulator rollout:
strict checkpoint loading into the pinned architecture, deterministic inference,
checkpoint/statistics provenance pairing, and the offline Safety-CVAE integration
identities. Nothing here trains, mutates a checkpoint, or steps an environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ACT_ROOT = REPO_ROOT / "submodules/act"
for path in (ACT_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "submodules/molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid_safety_residual import (
    DEFAULT_DECAY,
    DEFAULT_EMA,
    DEFAULT_GAIN,
    DEFAULT_MAX_DEVIATION,
    ResidualSafetyController,
    apply_arm_residual,
    load_contract,
    sensor_order_hash,
)
from policy import ACTPolicy
from train_safety_cvae import SafetyHead

CANONICAL = {
    "state_dim": 9,
    "action_dim": 8,
    "chunk_size": 100,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "kl_weight": 10,
    "backbone": "resnet18",
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
    "camera_names": ["exo_camera_1", "wrist_camera"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_canonical_policy() -> ACTPolicy:
    """Construct the pinned two-camera qpos9/action8 ACT model.

    ``build_ACT_model_and_optimizer`` re-parses ``sys.argv`` through the DETR parser
    before applying its override dict, so argv is temporarily replaced with the same
    required flags the canonical training command supplies. Every architectural value
    below is then set explicitly by the override, exactly as in training.
    """
    config = {
        "lr": 1e-5,
        "lr_backbone": 1e-5,
        "num_queries": CANONICAL["chunk_size"],
        "kl_weight": CANONICAL["kl_weight"],
        "hidden_dim": CANONICAL["hidden_dim"],
        "dim_feedforward": CANONICAL["dim_feedforward"],
        "backbone": CANONICAL["backbone"],
        "enc_layers": CANONICAL["enc_layers"],
        "dec_layers": CANONICAL["dec_layers"],
        "nheads": CANONICAL["nheads"],
        "camera_names": CANONICAL["camera_names"],
        "state_dim": CANONICAL["state_dim"],
        "action_dim": CANONICAL["action_dim"],
    }
    saved_argv = sys.argv
    sys.argv = [
        "imitate_episodes.py",
        "--ckpt_dir", "offline_validation",
        "--policy_class", "ACT",
        "--task_name", "obstacle_baseline",
        "--seed", "0",
        "--num_epochs", "2000",
    ]
    try:
        return ACTPolicy(config)
    finally:
        sys.argv = saved_argv


def fixed_batch(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """One deterministic synthetic RGB+qpos batch (fixed bytes, no RNG draw)."""
    rng = np.random.RandomState(20260724)
    image = rng.rand(1, len(CANONICAL["camera_names"]), 3, 240, 320).astype(np.float32)
    qpos = rng.rand(1, CANONICAL["state_dim"]).astype(np.float32)
    return (
        torch.from_numpy(image).to(device),
        torch.from_numpy(qpos).to(device),
    )


def strict_load(run_dir: Path, ckpt_name: str, device: str) -> dict[str, Any]:
    ckpt_path = run_dir / ckpt_name
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy = build_canonical_policy().to(device)
    result = policy.load_state_dict(state, strict=True)
    missing = list(getattr(result, "missing_keys", []) or [])
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    policy.eval()
    return {
        "policy": policy,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "strict_load_ok": not missing and not unexpected,
        "checkpoint_sha256": sha256_file(ckpt_path),
        "param_count": int(sum(p.numel() for p in policy.parameters())),
    }


def check_architecture(policy: ACTPolicy) -> dict[str, Any]:
    model = policy.model
    # Upstream ACT deliberately shares a single backbone across all cameras
    # (detr_vae.py: ``self.backbones[0](image[:, cam_id])  # HARDCODED``), so the
    # camera count is carried by ``camera_names``, not by the backbone list length.
    return {
        "num_queries_chunk": int(model.num_queries),
        "hidden_dim": int(model.transformer.d_model),
        "dim_feedforward": int(
            model.transformer.encoder.layers[0].linear1.out_features
        ),
        "action_dim": int(model.action_head.out_features),
        "qpos_input_dim": int(model.input_proj_robot_state.in_features),
        "camera_names": list(model.camera_names),
        "camera_count": len(model.camera_names),
        "shared_backbone_count": len(model.backbones),
        "kl_weight": int(policy.kl_weight),
    }


def cmd_validate(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device
    run_dir = Path(args.run_dir).resolve()
    report: dict[str, Any] = {
        "schema_version": "hybrid_clean_retrain_offline_validation_v1",
        "run_dir": str(run_dir),
        "device": device,
        "checks": {},
    }
    checks = report["checks"]

    # --- 1. strict load into the pinned architecture --------------------------
    loaded = strict_load(run_dir, "policy_best.ckpt", device)
    policy = loaded.pop("policy")
    checks["strict_load_policy_best"] = loaded
    arch = check_architecture(policy)
    checks["architecture"] = arch
    checks["architecture_matches_canonical"] = {
        "qpos_9": arch["qpos_input_dim"] == 9,
        "action_8": arch["action_dim"] == 8,
        "chunk_100": arch["num_queries_chunk"] == 100,
        "hidden_512": arch["hidden_dim"] == 512,
        "feedforward_3200": arch["dim_feedforward"] == 3200,
        "two_cameras": arch["camera_count"] == 2,
        "expected_camera_names": arch["camera_names"] == CANONICAL["camera_names"],
        "kl_weight_10": arch["kl_weight"] == 10,
    }

    # policy_last must also strict-load into the same architecture.
    last = strict_load(run_dir, "policy_last.ckpt", device)
    last.pop("policy")
    checks["strict_load_policy_last"] = last

    # --- 2. deterministic held-out batch --------------------------------------
    image, qpos = fixed_batch(device)
    with torch.inference_mode():
        out_a = policy(qpos, image)
    finite = bool(torch.isfinite(out_a).all().item())

    # --- 3. reload and require bit-identical output ----------------------------
    reloaded = strict_load(run_dir, "policy_best.ckpt", device)
    policy_b = reloaded.pop("policy")
    with torch.inference_mode():
        out_b = policy_b(qpos, image)
    identical = bool(torch.equal(out_a.cpu(), out_b.cpu()))
    checks["deterministic_inference"] = {
        "output_shape": list(out_a.shape),
        "outputs_finite": finite,
        "reload_bit_identical": identical,
        "max_abs_difference": float((out_a.cpu() - out_b.cpu()).abs().max().item()),
        "output_sha256": hashlib.sha256(
            out_a.cpu().numpy().tobytes()
        ).hexdigest(),
    }

    # --- 4. checkpoint / statistics provenance pairing -------------------------
    stats_path = run_dir / "dataset_stats.pkl"
    with stats_path.open("rb") as handle:
        stats = pickle.load(handle)
    same_dir = (
        (run_dir / "policy_best.ckpt").parent.resolve() == stats_path.parent.resolve()
    )
    checks["stats_pairing"] = {
        "dataset_stats_sha256": sha256_file(stats_path),
        "checkpoint_and_stats_same_run_dir": same_dir,
        "qpos_mean_shape": list(np.shape(stats["qpos_mean"])),
        "qpos_std_shape": list(np.shape(stats["qpos_std"])),
        "action_mean_shape": list(np.shape(stats["action_mean"])),
        "action_std_shape": list(np.shape(stats["action_std"])),
        "shapes_ok": (
            np.shape(stats["qpos_mean"]) == (9,)
            and np.shape(stats["qpos_std"]) == (9,)
            and np.shape(stats["action_mean"]) == (8,)
            and np.shape(stats["action_std"]) == (8,)
        ),
        "all_finite": all(
            bool(np.isfinite(np.asarray(stats[k])).all())
            for k in ("qpos_mean", "qpos_std", "action_mean", "action_std")
        ),
        "std_strictly_positive": bool(
            (np.asarray(stats["qpos_std"]) > 0).all()
            and (np.asarray(stats["action_std"]) > 0).all()
        ),
    }

    # --- 5. statistics must match the converted dataset exactly ----------------
    if args.dataset_dir:
        from utils import get_norm_stats

        recomputed = get_norm_stats(str(Path(args.dataset_dir).resolve()), args.num_episodes)
        diffs = {}
        for key in ("qpos_mean", "qpos_std", "action_mean", "action_std", "example_qpos"):
            a = np.asarray(stats[key], dtype=np.float64)
            b = np.asarray(recomputed[key], dtype=np.float64)
            diffs[key] = {
                "max_abs_difference": float(np.abs(a - b).max()),
                "exact": bool(np.array_equal(a, b)),
            }
        checks["stats_match_converted_dataset"] = {
            "per_key": diffs,
            "all_exact": all(v["exact"] for v in diffs.values()),
            "max_abs_difference": max(v["max_abs_difference"] for v in diffs.values()),
        }

    # --- 6. wrong hashes must be rejected --------------------------------------
    from hybrid_safety_residual import HybridSafetyContractError, require_sha256

    rejection = {}
    for label, path in (
        ("checkpoint", run_dir / "policy_best.ckpt"),
        ("dataset_stats", stats_path),
    ):
        wrong = "0" * 64
        try:
            require_sha256(path, wrong, label)
            rejection[label] = {"rejected": False}
        except HybridSafetyContractError as exc:
            rejection[label] = {"rejected": True, "error": str(exc)[:200]}
        # and the correct hash must be accepted
        try:
            require_sha256(path, sha256_file(path), label)
            rejection[label]["correct_hash_accepted"] = True
        except HybridSafetyContractError:
            rejection[label]["correct_hash_accepted"] = False
    checks["wrong_hash_rejection"] = rejection

    # --- 7. offline Safety-CVAE integration identities -------------------------
    checks["safety_integration"] = safety_offline_checks(args, policy, device)

    report["passed"] = evaluate(report)
    return report


def safety_offline_checks(
    args: argparse.Namespace, policy: ACTPolicy, device: str
) -> dict[str, Any]:
    """ACT-only/zero equivalence, arm-only residual, gripper copy, sensor order."""
    contract = load_contract(REPO_ROOT / "configs/hybrid_safety_stack_v1.json")
    names = list(contract["sensor_contract"]["ordered_names"])
    out: dict[str, Any] = {
        "sensor_count": len(names),
        "sensor_count_is_40": len(names) == 40,
        "sensor_order_hash": sensor_order_hash(names),
        "sensor_order_hash_matches_contract": sensor_order_hash(names)
        == contract["sensor_contract"]["sensor_order_hash"],
        "sensor_order_hash_is_canonical": sensor_order_hash(names)
        == "c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858",
    }

    head = SafetyHead.load(str(REPO_ROOT / "assets/safety/cvae_v3"), device=device)
    dt = 66.0 / 1000.0
    out["dt_s"] = dt
    out["controller_constants"] = {
        "gain": DEFAULT_GAIN,
        "decay": DEFAULT_DECAY,
        "ema": DEFAULT_EMA,
        "max_deviation": DEFAULT_MAX_DEVIATION,
        "unchanged": (
            DEFAULT_GAIN == 4.0
            and DEFAULT_DECAY == 2.2
            and DEFAULT_EMA == 0.75
            and DEFAULT_MAX_DEVIATION == 0.35
        ),
    }

    # Real proximity frames (40, 4, 8, 8) drawn from a source episode when given,
    # otherwise a deterministic synthetic stand-in.
    if args.proximity_h5:
        from hybrid_safety_residual import load_proximity_sequence_h5

        seq = load_proximity_sequence_h5(args.proximity_h5, names)
        frame_a = seq[0]
        frame_b = seq[min(len(seq) - 1, args.proximity_step)]
        out["proximity_source"] = args.proximity_h5
    else:
        rng = np.random.RandomState(7)
        frame_a = rng.rand(len(names), 8, 8).astype(np.float32) * 0.5
        frame_b = rng.rand(len(names), 8, 8).astype(np.float32) * 0.5
        out["proximity_source"] = "synthetic"

    nominal = {
        "arm": np.arange(7, dtype=np.float32) * 0.11,
        "gripper": np.array([137.0], dtype=np.float32),
    }

    def run_mode(mode: str, current, reference) -> dict[str, Any]:
        controller = ResidualSafetyController(
            label_scale=head.scale,
            dt=dt,
            gain=DEFAULT_GAIN,
            decay=DEFAULT_DECAY,
            ema=DEFAULT_EMA,
            max_deviation=DEFAULT_MAX_DEVIATION,
        )
        if mode == "act_only":
            raw = np.zeros(7, dtype=np.float32)
            base = np.zeros(7, dtype=np.float32)
        else:
            raw = head(current)
            base = head(reference)
        step = controller.step(raw, base, safety_enabled=mode not in ("act_only", "zero"))
        executed = apply_arm_residual(nominal, step.correction)
        return {
            "correction": step.correction.tolist(),
            "subtracted_dq": step.subtracted_dq.tolist(),
            "executed_arm": executed["arm"].tolist(),
            "executed_gripper": np.asarray(executed["gripper"]).tolist(),
        }

    act_only = run_mode("act_only", frame_b, frame_a)
    zero = run_mode("zero", frame_b, frame_a)
    normal = run_mode("normal", frame_b, frame_a)

    out["act_only_equals_zero"] = {
        "executed_arm_identical": np.array_equal(
            act_only["executed_arm"], zero["executed_arm"]
        ),
        "correction_both_zero": bool(
            np.array_equal(act_only["correction"], [0.0] * 7)
            and np.array_equal(zero["correction"], [0.0] * 7)
        ),
        "gripper_identical": act_only["executed_gripper"] == zero["executed_gripper"],
    }
    out["act_only_executed_equals_nominal"] = bool(
        np.array_equal(np.asarray(act_only["executed_arm"], dtype=np.float32), nominal["arm"])
    )
    out["normal_changes_only_seven_arm_joints"] = {
        "arm_changed": not np.array_equal(
            np.asarray(normal["executed_arm"], dtype=np.float32), nominal["arm"]
        ),
        "arm_length": len(normal["executed_arm"]),
        "correction_length": len(normal["correction"]),
        "gripper_copied_exactly": normal["executed_gripper"]
        == np.asarray(nominal["gripper"]).tolist(),
    }

    # The nominal ACT tensor itself must be untouched by the safety layer.
    image, qpos = fixed_batch(device)
    with torch.inference_mode():
        nominal_tensor = policy(qpos, image)
    _ = run_mode("normal", frame_b, frame_a)
    with torch.inference_mode():
        nominal_tensor_after = policy(qpos, image)
    out["normal_leaves_nominal_act_tensor_unchanged"] = bool(
        torch.equal(nominal_tensor.cpu(), nominal_tensor_after.cpu())
    )

    # Baseline-subtraction identity: identical current and reference skin must
    # produce exactly zero subtracted dq and exactly zero residual correction.
    identical = run_mode("normal", frame_a, frame_a)
    out["identical_skin_zero_residual"] = {
        "subtracted_dq_exactly_zero": bool(
            np.array_equal(np.asarray(identical["subtracted_dq"]), np.zeros(7))
        ),
        "correction_exactly_zero": bool(
            np.array_equal(np.asarray(identical["correction"]), np.zeros(7))
        ),
        "executed_equals_nominal": bool(
            np.array_equal(
                np.asarray(identical["executed_arm"], dtype=np.float32), nominal["arm"]
            )
        ),
    }

    # The two known self-returning link5-front sensors must not create a residual
    # once the canonical reference subtraction has been applied.
    idx = [names.index("link5_front_sensor_1"), names.index("link5_front_sensor_2")]
    out["known_self_return_sensors"] = {
        "indices": idx,
        "names": ["link5_front_sensor_1", "link5_front_sensor_2"],
        "values_are_nonzero_in_frame": bool(
            np.any(frame_a[idx[0]] > 0) or np.any(frame_a[idx[1]] > 0)
        ),
        "cancel_after_baseline_subtraction": out["identical_skin_zero_residual"][
            "correction_exactly_zero"
        ],
    }
    return out


def evaluate(report: dict[str, Any]) -> bool:
    c = report["checks"]
    conditions = [
        c["strict_load_policy_best"]["strict_load_ok"],
        c["strict_load_policy_last"]["strict_load_ok"],
        all(c["architecture_matches_canonical"].values()),
        c["deterministic_inference"]["outputs_finite"],
        c["deterministic_inference"]["reload_bit_identical"],
        c["stats_pairing"]["shapes_ok"],
        c["stats_pairing"]["all_finite"],
        c["stats_pairing"]["std_strictly_positive"],
        c["stats_pairing"]["checkpoint_and_stats_same_run_dir"],
        all(v["rejected"] for v in c["wrong_hash_rejection"].values()),
        all(v["correct_hash_accepted"] for v in c["wrong_hash_rejection"].values()),
        c["safety_integration"]["sensor_count_is_40"],
        c["safety_integration"]["sensor_order_hash_is_canonical"],
        c["safety_integration"]["controller_constants"]["unchanged"],
        c["safety_integration"]["act_only_equals_zero"]["executed_arm_identical"],
        c["safety_integration"]["act_only_executed_equals_nominal"],
        c["safety_integration"]["normal_changes_only_seven_arm_joints"][
            "gripper_copied_exactly"
        ],
        c["safety_integration"]["normal_leaves_nominal_act_tensor_unchanged"],
        c["safety_integration"]["identical_skin_zero_residual"][
            "subtracted_dq_exactly_zero"
        ],
        c["safety_integration"]["identical_skin_zero_residual"]["correction_exactly_zero"],
    ]
    if "stats_match_converted_dataset" in c:
        conditions.append(c["stats_match_converted_dataset"]["all_exact"])
    return all(bool(x) for x in conditions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset_dir")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--proximity_h5")
    parser.add_argument("--proximity_step", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(func=cmd_validate)
    args = parser.parse_args()
    json.dump(args.func(args), sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
