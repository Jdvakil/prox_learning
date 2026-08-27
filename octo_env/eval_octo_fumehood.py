"""Closed-loop eval of an Octo policy (served by octo_policy_server.py) in the
cluttered fume-hood environment.

Runs in the molmospaces (torch) env — the only extra dependency is stdlib
urllib talking to the Octo server. The env config reuses the fumehood_env
suite, so hood sizes, clutter and deep reach all apply.

    # terminal 1 (octo venv):
    python octo_env/octo_policy_server.py --checkpoint ~/octo_runs/vp_octo --port 8555

    # terminal 2 (molmospaces venv, PYTHONPATH=prox_learning):
    python octo_env/eval_octo_fumehood.py --server http://127.0.0.1:8555 \
        --houses 1,313 --samples 2 --output_dir /tmp/octo_check [--no_proximity]
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

import argparse
import io
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.policy.base_policy import InferencePolicy

from fumehood_env.cluttered_fumehood_configs import FrankaSkinClutteredFumehoodConfig

N_SENSORS = 40           # must match the RLDS builder
DEAD_PIXEL_M = 0.005
FAR_M = 4.0


def _post(url: str, payload: dict | None = None):
    body = b""
    if payload is not None:
        buf = io.BytesIO()
        np.savez_compressed(buf, **payload)
        body = buf.getvalue()
    req = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    return data


class OctoRemotePolicy(InferencePolicy):
    """Queries the Octo server every control step and executes the first action
    of the returned chunk — maximum reactivity, per the responsiveness concern
    in plan.md; switch to open-loop chunk execution later if latency demands."""

    def __init__(self, exp_config, task=None) -> None:
        super().__init__(exp_config)
        self.task = task
        self.pc = exp_config.policy_config

    def reset(self) -> None:
        try:
            _post(self.pc.server_url + "/reset")
        except Exception as e:
            print(f"[octo-client] reset failed: {e}")

    def prepare_model(self, model_name: str | None = None) -> None:
        with urllib.request.urlopen(self.pc.server_url + "/info", timeout=10) as r:
            print("[octo-client] server:", json.loads(r.read()))

    def obs_to_model_input(self, obs):
        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        return obs

    @staticmethod
    def _resize(img, hw):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        return cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _sensor_names(obs) -> list[str]:
        """Live observations expose each proximity sensor as a TOP-LEVEL key named
        after its camera (ProximityDepthBufferSensor is constructed with
        uuid=camera_name); only the recorded h5 groups them under obs/proximity.
        Reading obs["proximity"] therefore found nothing and the state silently
        stayed 9-dim, which the model rejected as (1,2,9) against (1,2,49).
        Sorted by name to match the dataset builder's sensor order exactly."""
        return sorted(k for k in obs
                      if k.startswith("link") and "sensor_" in k
                      and not k.endswith(("_viz_rgb", "_viz_depth_turbo")))

    def _proximity_vector(self, obs):
        names = self._sensor_names(obs)
        if len(names) != N_SENSORS:
            raise RuntimeError(
                f"expected {N_SENSORS} proximity sensors in the observation, found "
                f"{len(names)}: {names[:5]}{'...' if len(names) > 5 else ''}")
        vals = []
        for n in names:
            d = np.asarray(obs[n]).reshape(-1)
            d = d[d > DEAD_PIXEL_M]
            vals.append(float(d.min()) if d.size else FAR_M)
        return np.asarray(vals, dtype=np.float32)

    def inference_model(self, obs):
        arm = np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32)
        grip = np.asarray((obs["qpos"].get("gripper") or [0.0, 0.0])[:2], dtype=np.float32)
        state = np.concatenate([arm, grip])
        if self.pc.include_proximity:
            state = np.concatenate([state, self._proximity_vector(obs)])

        data = _post(self.pc.server_url + "/act", {
            "image_primary": self._resize(obs["exo_camera_1"], (256, 256)),
            "image_wrist": self._resize(obs["wrist_camera"], (128, 128)),
            "state": state.astype(np.float32),
        })
        chunk = np.load(io.BytesIO(data))["action_chunk"]
        return chunk[0]

    def model_output_to_action(self, model_output):
        arm = np.asarray(model_output[:7], dtype=np.float32)
        gripper_raw = float(model_output[7]) if len(model_output) >= 8 else 0.0
        gripper = 0.0 if gripper_raw < 127.5 else 255.0
        return {"arm": arm, "gripper": np.asarray([gripper], dtype=np.float32)}


class OctoRemotePolicyConfig(BasePolicyConfig):
    policy_cls: type = OctoRemotePolicy
    policy_type: str = "learned"
    server_url: str = "http://127.0.0.1:8555"
    include_proximity: bool = True


class OctoFumehoodEvalConfig(FrankaSkinClutteredFumehoodConfig):
    policy_config: OctoRemotePolicyConfig = OctoRemotePolicyConfig()
    use_wandb: bool = False
    filter_for_successful_trajectories: bool = False
    save_videos: bool = True
    use_passive_viewer: bool = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8555")
    ap.add_argument("--houses", default="1,313")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--task_horizon", type=int, default=300)
    ap.add_argument("--no_proximity", action="store_true",
                    help="send qpos-only state (for the vision_only checkpoint)")
    ap.add_argument("--no_videos", action="store_true")
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"[octo-eval] ignoring extra args: {unknown}")

    cfg = OctoFumehoodEvalConfig()
    cfg.task_horizon = args.task_horizon
    cfg.output_dir = Path(args.output_dir).resolve()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if args.no_videos:
        cfg.save_videos = False
    cfg.task_sampler_config.house_inds = [int(h) for h in args.houses.split(",")]
    cfg.task_sampler_config.samples_per_house = args.samples
    cfg.policy_config.server_url = args.server.rstrip("/")
    cfg.policy_config.include_proximity = not args.no_proximity

    cfg.save_config()
    print(f"[octo-eval] server={cfg.policy_config.server_url} "
          f"proximity={cfg.policy_config.include_proximity} "
          f"houses={cfg.task_sampler_config.house_inds}")
    success, total = ParallelRolloutRunner(cfg).run()
    print(f"[octo-eval] success {success}/{total}")


if __name__ == "__main__":
    main()
