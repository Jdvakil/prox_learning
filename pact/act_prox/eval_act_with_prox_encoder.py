"""Rollout-time eval for ACT + frozen prox-encoder (P+ACT).

Mirrors `submodules/act/eval_act_mug_random.py` (so the env config and
ParallelRolloutRunner plumbing are unchanged) but:
  * Loads `FrozenProxFeatureExtractor` from the encoder checkpoint.
  * Maintains a per-sensor ring buffer of the last W=8 control-step substep
    readings (each step's `(4, 8, 8)` block) → assembles `(1, 29, W*4, 8, 8)`
    each control step, z-scores with the encoder's `prox_mean`/`prox_std`, and
    runs the extractor → `(1, 29, 3)` metres.
  * Passes the resulting positions into the policy via `proximity_positions=`.

The new eval does NOT touch the existing `submodules/act/eval_act_with_prox.py`
(which depended on the now-deleted residual-head module).

Run from the repo root with the conda env that has molmospaces installed:

    cd /home/jaydv/code/prox_learning/submodules/act
    PYTHONPATH="$PWD:/home/jaydv/code/prox_learning:$PYTHONPATH" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python /home/jaydv/code/prox_learning/pact/act_prox/eval_act_with_prox_encoder.py \
        --ckpt_dir <runs/act_prox_mug_v1> \
        --prox_encoder_ckpt /home/jaydv/code/prox_learning/pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt \
        --prox_mapping_json /home/jaydv/code/prox_learning/act_style_data/mug_house1_random_everything/prox_mapping.json \
        --output_dir /home/jaydv/code/prox_learning/eval_output/act_prox_mug_v1
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

import argparse
import json
import pickle
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None  # type: ignore

_WANDB_RUN = None
_ROLLOUT_INDEX = 0

# Make ACT modules importable.
_ACT_DIR = Path(__file__).resolve().parents[2] / "submodules" / "act"
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))
# Make pact modules importable.
_PACT_PARENT = Path(__file__).resolve().parents[2]
if str(_PACT_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACT_PARENT))


@contextmanager
def _detr_argv(ckpt_dir: str, seed: int, n_proximity_sensors: int):
    """Hide this script's CLI flags from detr/main.py's nested argparse."""
    orig = sys.argv
    sys.argv = [
        orig[0] if orig else "eval_act_with_prox_encoder.py",
        "--ckpt_dir", ckpt_dir,
        "--policy_class", "ACT",
        "--task_name", "pla_house1_mug_random",
        "--seed", str(seed),
        "--num_epochs", "1",
        "--n_proximity_sensors", str(n_proximity_sensors),
    ]
    try:
        yield
    finally:
        sys.argv = orig


from policy import ACTPolicy                                              # noqa: E402
from utils import set_seed                                                # noqa: E402

from molmo_spaces.configs.policy_configs import BasePolicyConfig          # noqa: E402
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (  # noqa: E402
    FrankaSkinPickAndPlacePilotMediumConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner    # noqa: E402
from molmo_spaces.policy.base_policy import InferencePolicy                # noqa: E402

from pact.act_prox.prox_features import FrozenProxFeatureExtractor         # noqa: E402


# ----------------------------------------------------------------------
# Inference policy.
# ----------------------------------------------------------------------
class ACTWithProxEncoderInferencePolicy(InferencePolicy):
    def __init__(self, exp_config, task=None) -> None:
        super().__init__(exp_config)
        self.task = task
        pc: "ACTProxEncoderPolicyConfig" = exp_config.policy_config
        self.pc = pc
        self.ckpt_path = str(Path(pc.ckpt_dir) / pc.ckpt_name)
        self.stats_path = str(Path(pc.ckpt_dir) / "dataset_stats.pkl")

        with open(pc.prox_mapping_json, "r") as f:
            mp = json.load(f)
        self.sensor_names: list[str] = list(mp["sensor_names"])
        self.n_sensors: int = int(mp["n_sensors"])

        self._step: int = 0
        self._pending_chunks: list[Tuple[int, np.ndarray]] = []
        self._policy: Optional[ACTPolicy] = None
        self._stats = None
        self._extractor: Optional[FrozenProxFeatureExtractor] = None

        # Ring buffer (per sensor) of last W control steps, each (4, 8, 8).
        # Lazy-built on first inference call so we know the encoder's window.
        self._buffer: Optional[np.ndarray] = None   # (W, N_sensors, 4, 8, 8)
        self._buffer_filled: int = 0
        self._window: Optional[int] = None
        self._prox_mean: Optional[np.ndarray] = None
        self._prox_std: Optional[np.ndarray] = None

    def reset(self) -> None:
        global _ROLLOUT_INDEX
        if _WANDB_RUN is not None and self._step > 0:
            _WANDB_RUN.log(
                {"rollout/episode_idx": _ROLLOUT_INDEX,
                 "rollout/episode_length": int(self._step)},
                step=_ROLLOUT_INDEX,
            )
            _ROLLOUT_INDEX += 1
        self._step = 0
        self._pending_chunks.clear()
        self._buffer = None
        self._buffer_filled = 0

    def prepare_model(self, model_name: str | None = None) -> None:
        pc = self.pc
        # Load encoder extractor.
        self._extractor = FrozenProxFeatureExtractor(pc.prox_encoder_ckpt, device=torch.device("cuda"))
        self._window = int(self._extractor.window)
        self._prox_mean = self._extractor.prox_mean.cpu().numpy().astype(np.float32)   # (4, 8, 8)
        self._prox_std  = self._extractor.prox_std.cpu().numpy().astype(np.float32)    # (4, 8, 8)
        print(f"[act-prox-eval] loaded prox encoder (window={self._window}, n_sensors={self.n_sensors})")

        # Build the ACT policy with prox tokens enabled.
        policy_config = {
            "lr": pc.lr,
            "num_queries": pc.chunk_size,
            "kl_weight": pc.kl_weight,
            "hidden_dim": pc.hidden_dim,
            "dim_feedforward": pc.dim_feedforward,
            "lr_backbone": pc.lr_backbone,
            "backbone": pc.backbone,
            "enc_layers": pc.enc_layers,
            "dec_layers": pc.dec_layers,
            "nheads": pc.nheads,
            "camera_names": list(pc.camera_names),
            "state_dim": pc.state_dim,
            "action_dim": pc.action_dim,
            "n_proximity_sensors": self.n_sensors,
        }
        with _detr_argv(self.pc.ckpt_dir, self.pc.seed, self.n_sensors):
            policy = ACTPolicy(policy_config)
        sd = torch.load(self.ckpt_path, map_location="cuda")
        policy.load_state_dict(sd)
        policy.cuda().eval()
        self._policy = policy

        with open(self.stats_path, "rb") as f:
            self._stats = pickle.load(f)
        print(f"[act-prox-eval] loaded ACT {self.ckpt_path}")

    def obs_to_model_input(self, obs):
        if isinstance(obs, list | tuple):
            obs = obs[0]
        return obs

    def _push_proximity(self, obs) -> None:
        """Push the current control step's (N_sensors, 4, 8, 8) block onto the ring buffer."""
        block = np.zeros((self.n_sensors, 4, 8, 8), dtype=np.float32)
        for i, sname in enumerate(self.sensor_names):
            arr = np.asarray(obs[sname], dtype=np.float32)
            if arr.shape == (4, 8, 8):
                block[i] = arr
            elif arr.shape == (8, 8):
                # Simulator returned a per-step mean-pool — replicate across substeps.
                block[i] = np.broadcast_to(arr[None], (4, 8, 8)).copy()
            else:
                raise ValueError(f"unexpected proximity shape {arr.shape} for {sname}")
        W = self._window
        if self._buffer is None:
            # Cold-start: fill all W slots with the first block.
            self._buffer = np.broadcast_to(block[None], (W, self.n_sensors, 4, 8, 8)).copy()
            self._buffer_filled = 1
        else:
            # Roll left by one control step and place the new block at the end.
            self._buffer[:-1] = self._buffer[1:]
            self._buffer[-1] = block
            self._buffer_filled = min(W, self._buffer_filled + 1)

    def _build_prox_window_tensor(self) -> torch.Tensor:
        """Assemble the (1, N_sensors, W*4, 8, 8) z-scored input for the encoder."""
        # buf shape: (W, N_sensors, 4, 8, 8) -> normalise across substep stats:
        buf = self._buffer.astype(np.float32, copy=False)
        normed = (buf - self._prox_mean[None, None]) / self._prox_std[None, None]
        # Transpose to (N_sensors, W, 4, 8, 8) and fold (W*4) for the encoder.
        normed = np.transpose(normed, (1, 0, 2, 3, 4))                     # (N, W, 4, 8, 8)
        normed = normed.reshape(self.n_sensors, self._window * 4, 8, 8)    # (N, W*4, 8, 8)
        return torch.from_numpy(normed).unsqueeze(0).cuda()                # (1, N, W*4, 8, 8)

    def inference_model(self, obs):
        if self._policy is None:
            self.prepare_model()

        pc = self.pc
        stats = self._stats

        # qpos: arm + gripper.
        arm = np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32)
        grip = np.asarray((obs["qpos"].get("gripper") or [0.0, 0.0])[:2], dtype=np.float32)
        qpos = np.concatenate([arm, grip], axis=0).astype(np.float32)
        qpos = (qpos - stats["qpos_mean"]) / stats["qpos_std"]
        qpos_t = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

        # Image stack.
        cams = []
        for cam in pc.camera_names:
            img = obs[cam]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            if img.shape[:2] != (pc.image_h, pc.image_w):
                img = cv2.resize(img, (pc.image_w, pc.image_h), interpolation=cv2.INTER_AREA)
            cams.append(img.astype(np.float32) / 255.0)
        image = np.stack(cams, axis=0)
        image = np.transpose(image, (0, 3, 1, 2))
        image_t = torch.from_numpy(image).float().cuda().unsqueeze(0)

        # Proximity ring buffer → encoder → predicted positions.
        self._push_proximity(obs)
        prox_window = self._build_prox_window_tensor()                     # (1, N, W*4, 8, 8)
        prox_pos = self._extractor(prox_window)                            # (1, N, 3)

        # Diagnostics: log per-step prox stats periodically.
        if _WANDB_RUN is not None and self._step % 25 == 0:
            _WANDB_RUN.log(
                {
                    "rollout/prox_pos_x_mean": float(prox_pos[..., 0].mean().item()),
                    "rollout/prox_pos_y_mean": float(prox_pos[..., 1].mean().item()),
                    "rollout/prox_pos_z_mean": float(prox_pos[..., 2].mean().item()),
                    "rollout/prox_buffer_filled": int(self._buffer_filled),
                    "rollout/step_in_episode": int(self._step),
                },
                step=_ROLLOUT_INDEX,
            )

        with torch.no_grad():
            a_hat = self._policy(qpos_t, image_t, proximity_positions=prox_pos)  # (1, chunk, A) normalised

        new_chunk = a_hat.squeeze(0).cpu().numpy()
        new_chunk = new_chunk * stats["action_std"] + stats["action_mean"]

        if pc.temp_agg_off:
            self._pending_chunks = [(self._step, new_chunk)]
            return new_chunk[0]
        H = new_chunk.shape[0]
        self._pending_chunks.append((self._step, new_chunk))
        self._pending_chunks = [
            (s, c) for (s, c) in self._pending_chunks if self._step - s < H
        ]
        preds, weights = [], []
        for (start, chunk) in self._pending_chunks:
            k = self._step - start
            if 0 <= k < H:
                preds.append(chunk[k])
                weights.append(np.exp(-pc.temp_agg_m * k))
        preds_a = np.stack(preds, axis=0)
        w = np.asarray(weights, dtype=np.float64)
        w /= w.sum()
        return (preds_a * w[:, None]).sum(axis=0).astype(np.float32)

    def model_output_to_action(self, model_output):
        arm = np.asarray(model_output[:7], dtype=np.float32)
        gripper_raw = float(model_output[7]) if len(model_output) >= 8 else 0.0
        gripper = 0.0 if gripper_raw < 127.5 else 255.0
        return {"arm": arm, "gripper": np.asarray([gripper], dtype=np.float32)}

    def get_action(self, obs):
        action = super().get_action(obs)
        self._step += 1
        return action


# ----------------------------------------------------------------------
# Policy + env configs.
# ----------------------------------------------------------------------
class ACTProxEncoderPolicyConfig(BasePolicyConfig):
    policy_cls: type = ACTWithProxEncoderInferencePolicy
    policy_type: str = "learned"

    ckpt_dir: str = ""
    ckpt_name: str = "policy_best.ckpt"
    prox_encoder_ckpt: str = ""
    prox_mapping_json: str = ""

    image_h: int = 240
    image_w: int = 320
    camera_names: tuple[str, ...] = ("exo_camera_1", "wrist_camera")
    chunk_size: int = 20
    temp_agg_m: float = 0.01
    temp_agg_off: bool = False
    kl_weight: int = 10
    hidden_dim: int = 256
    dim_feedforward: int = 2048
    enc_layers: int = 4
    dec_layers: int = 7
    nheads: int = 8
    state_dim: int = 9
    action_dim: int = 8
    backbone: str = "resnet18"
    lr: float = 1e-5
    lr_backbone: float = 1e-5
    seed: int = 0


class ACTProxEncoderMugRandomEvalConfig(FrankaSkinPickAndPlacePilotMediumConfig):
    """ACT+prox-encoder eval against the same env used to collect
    mug_house_1_random_everything. Inherits all task sampler knobs verbatim
    and only attaches a learned policy."""

    policy_config: ACTProxEncoderPolicyConfig = ACTProxEncoderPolicyConfig()
    use_wandb: bool = False
    filter_for_successful_trajectories: bool = False
    save_videos: bool = True
    use_passive_viewer: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True,
                   help="Directory containing policy_best.ckpt + dataset_stats.pkl")
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument("--prox_encoder_ckpt", required=True)
    p.add_argument("--prox_mapping_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--task_horizon", type=int, default=300)
    p.add_argument("--chunk_size", type=int, default=20)
    p.add_argument("--kl_weight", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dim_feedforward", type=int, default=2048)
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--temp_agg_off", action="store_true")
    p.add_argument("--temp_agg_m", type=float, default=0.01)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="pact-eval")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default="act_prox_mug_v1")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[act-prox-eval] ignoring extra args: {unknown}")
    return args


def main() -> None:
    args = parse_args()

    eval_cfg = ACTProxEncoderMugRandomEvalConfig()
    eval_cfg.task_horizon = args.task_horizon
    eval_cfg.output_dir = Path(args.output_dir).resolve()
    eval_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(eval_cfg.seed) if eval_cfg.seed is not None else 2026)

    pc = eval_cfg.policy_config
    pc.ckpt_dir = str(Path(args.ckpt_dir).resolve())
    pc.ckpt_name = args.ckpt_name
    pc.prox_encoder_ckpt = str(Path(args.prox_encoder_ckpt).resolve())
    pc.prox_mapping_json = str(Path(args.prox_mapping_json).resolve())
    pc.chunk_size = args.chunk_size
    pc.kl_weight = args.kl_weight
    pc.hidden_dim = args.hidden_dim
    pc.dim_feedforward = args.dim_feedforward
    pc.image_h = args.image_h
    pc.image_w = args.image_w
    pc.temp_agg_off = args.temp_agg_off
    pc.temp_agg_m = args.temp_agg_m

    eval_cfg.save_config()
    print(f"[act-prox-eval] writing rollouts to {eval_cfg.output_dir}")
    print(f"[act-prox-eval] env config seed={eval_cfg.seed}  "
          f"samples_per_house={eval_cfg.task_sampler_config.samples_per_house}  "
          f"house_inds={list(eval_cfg.task_sampler_config.house_inds)}")

    global _WANDB_RUN, _ROLLOUT_INDEX
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("--use_wandb but wandb is not installed.")
        run_name = args.wandb_run_name or f"eval_act_prox_mug_{int(time.time())}"
        _WANDB_RUN = wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            config={
                "ckpt_dir": pc.ckpt_dir,
                "prox_encoder_ckpt": pc.prox_encoder_ckpt,
                "prox_mapping_json": pc.prox_mapping_json,
                "task_horizon": args.task_horizon,
                "chunk_size": pc.chunk_size,
                "kl_weight": pc.kl_weight,
                "hidden_dim": pc.hidden_dim,
                "dim_feedforward": pc.dim_feedforward,
                "env_config_class": "FrankaSkinPickAndPlacePilotMediumConfig",
                "env_seed": eval_cfg.seed,
            },
            tags=["pact", "act_prox", "house1_mug_random", "eval"],
        )
        _ROLLOUT_INDEX = 0
        print(f"[act-prox-eval] wandb run: {_WANDB_RUN.url}  (group={args.wandb_group})")

    try:
        runner = ParallelRolloutRunner(eval_cfg)
        success, total = runner.run()
        print(f"[act-prox-eval] success {success}/{total}")
        if _WANDB_RUN is not None:
            rate = (success / total) if total > 0 else 0.0
            _WANDB_RUN.log({"eval/success": int(success), "eval/total": int(total),
                            "eval/success_rate": float(rate)})
            _WANDB_RUN.summary["success"] = int(success)
            _WANDB_RUN.summary["total"] = int(total)
            _WANDB_RUN.summary["success_rate"] = float(rate)
    finally:
        if _WANDB_RUN is not None:
            _WANDB_RUN.finish()
            _WANDB_RUN = None


if __name__ == "__main__":
    main()
