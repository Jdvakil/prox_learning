"""Frozen ACT / PACT eval for Hub v6 two-object only.

Protocol is the v1011d frozen evaluator (open-loop chunk, gated sensors,
``FrozenACTPolicy`` local to this file) with the v6 world swapped in.

Do not import ``eval_act_obstacle.ACTInferencePolicy``. Do not edit
``eval_act_pact_pick_n_place.py``, ``eval_act_place_corridor.py``,
``eval_pact.py``, or ``eval_place_fast_hooks.py``.

World: ``PactPlaceCorridorV1010TwoObjectSampler`` + Hub contract 24-cell
rows + ``pact_place_corridor_v10_7_{neg5,center,pos5}.xml``.

Headline defaults: ``--skin_substeps snapshot``, ``--exec_horizon`` = chunk
(50), open-loop, gated sensors, terminal ``judge_success``. ``--history
consecutive`` uses the 8-step causal window (prefetch the last 7 idle
chunk steps plus the query). Every-step skin is worse (17/48 vs 20/48).
``--skin rays`` is the v6 default.

    conda activate mlspaces-warp111
    cd /home/ekshan/prox_learning_v6_train/submodules/act
    export OMP_NUM_THREADS=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export MLSPACES_ASSETS_DIR=/home/ekshan/prox_learning/assets

    python eval_act_v6.py \\
      --ckpt_dir ckpts/pact_pick_n_place_v6/<run> \\
      --molmo /home/ekshan/prox_learning_v1011d_n100/submodules/molmospaces \\
      --num_rollouts 2 --spread_cells --output_dir /tmp/v6_smoke
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_LIVE_RENDER = any(a in ("--live", "--render", "--viewer") for a in sys.argv[1:])
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PACT_CONTACT_AUDIT_SUMMARY_ONLY", "1")
if not _LIVE_RENDER:
    os.environ.pop("DISPLAY", None)

_ACT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ACT_DIR.parents[1]
_ENV_VERSION = "pact_place_corridor_v10_10_two_object"
_LOG = "[eval_act_v6]"
_DEFAULT_CAMERAS = ("exo_camera_1", "wrist_camera")
_TRAIN_PERIOD_MS = 16.6667
_EXO_CAM = "exo_camera_1"
_N_CELLS = 24
# Geometry readout: last 8 control steps ending at the chunk query.
# Do not render skin on the other 42 idle chunk steps.
PROX_CAUSAL_STEPS = 8


def _argv_value(*names: str, default: str | None = None) -> str | None:
    args = sys.argv[1:]
    for i, a in enumerate(args):
        for name in names:
            if a == name and i + 1 < len(args):
                return args[i + 1]
            if a.startswith(name + "="):
                return a.split("=", 1)[1]
    return default


_DEFAULT_MOLMO = os.environ.get(
    "MOLMOSPACES_PACT_V1010",
    "/home/ekshan/prox_learning_v1011d_n100/submodules/molmospaces",
)
_MOLMO_ROOT = Path(
    _argv_value("--molmo", "--molmospaces_root", default=_DEFAULT_MOLMO)
).resolve()
if not (_MOLMO_ROOT / "molmo_spaces").is_dir():
    raise SystemExit(
        f"{_LOG} molmospaces missing at {_MOLMO_ROOT}. "
        "Point --molmo at the v1011d collect tree."
    )

os.environ.setdefault("MLSPACES_ASSETS_DIR", str(_REPO_ROOT / "assets"))
os.environ["MOLMOSPACES_PACT_PLACE"] = str(_MOLMO_ROOT)
os.environ["MOLMOSPACES_PACT_V1010"] = str(_MOLMO_ROOT)
sys.path.insert(0, str(_MOLMO_ROOT))
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
import time
from contextlib import contextmanager

import cv2
import numpy as np
import torch

from encoders.pact import (
    build_pact_encoder,
    encode_for_act,
    is_geometry_feature,
    resolve_act_encoder_load,
)
from encoders.peak_closeness import stack_obs_proximity
from eval_place_v1010_scene import (
    V1010_SCENE_BY_POSE,
    assert_v1010_scene_hashes,
    resolve_v1010_scenes_dir,
    spread_episode_count,
    v1010_cell,
)
from molmo_spaces.configs.camera_configs import FrankaSkinHybridCameraSystem
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinPACTCollisionCorridorConfig,
)
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.tasks.pact_place_contact_audit import PactPlaceContactAudit
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
from pact_place_v12_contract import EVAL_MASTER_SEED, build_row
from policy import ACTPolicy
from utils import set_seed

try:
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV1010TwoObjectSampler
except ImportError as exc:
    raise SystemExit(
        f"{_LOG} PactPlaceCorridorV1010TwoObjectSampler missing on {_MOLMO_ROOT}: {exc}"
    ) from exc

_EPISODE_METRICS: list[dict] = []
_METRICS_JSONL: Path | None = None

_RENDER_CLASS_NAMES = frozenset(
    {
        "CameraParameterSensor",
        "CameraSensor",
        "DepthSensor",
        "ProximityDepthBufferSensor",
        "ProximityVizDepthSensor",
        "ProximityVizRGBSensor",
    }
)


def _policy_wants_fresh(task) -> bool:
    policy = getattr(task, "_registered_policy", None)
    if policy is not None and hasattr(policy, "needs_fresh_policy_observation"):
        return bool(policy.needs_fresh_policy_observation())
    return True


def _is_render_sensor(sensor, render_types: tuple) -> bool:
    if isinstance(sensor, render_types):
        return True
    return type(sensor).__name__ in _RENDER_CLASS_NAMES


def _is_prox_depth_sensor(sensor, prox_types: tuple) -> bool:
    if prox_types and isinstance(sensor, prox_types):
        return True
    return type(sensor).__name__ == "ProximityDepthBufferSensor"


def _heartbeat(suite, extra: str = "") -> None:
    n_fresh = int(getattr(suite, "_fast_eval_n_fresh", 0) or 0)
    n_skip = int(getattr(suite, "_fast_eval_n_skip", 0) or 0)
    total = n_fresh + n_skip
    if total in (1, 10, 50) or (total > 0 and total % 100 == 0):
        skin_s = float(getattr(suite, "_fast_eval_skin_s", 0.0) or 0.0)
        print(
            f"[act-eval-place] sensors fresh={n_fresh} skip={n_skip} "
            f"skin_s={skin_s:.2f}{extra}",
            flush=True,
        )


def _patch_all_named_get_observation(class_name: str) -> int:
    patched = 0
    seen: set[int] = set()
    for mod_name, mod in list(sys.modules.items()):
        if not isinstance(mod_name, str) or not mod_name.startswith("molmo_spaces"):
            continue
        cls = getattr(mod, class_name, None)
        if not isinstance(cls, type):
            continue
        key = id(cls)
        if key in seen or not hasattr(cls, "get_observation"):
            continue
        seen.add(key)
        orig = cls.get_observation

        def get_observation(self, env, task, *args, _orig=orig, **kwargs):
            if (not _policy_wants_fresh(task)) and getattr(
                self, "_fast_eval_last_frame", None
            ) is not None:
                return self._fast_eval_last_frame
            out = _orig(self, env, task, *args, **kwargs)
            self._fast_eval_last_frame = out
            return out

        cls.get_observation = get_observation
        patched += 1
    return patched


def _install_chunk_gated_sensors() -> None:
    from molmo_spaces.env.abstract_sensors import SensorSuite
    from molmo_spaces.env.sensors_cameras import (
        CameraParameterSensor,
        CameraSensor,
        DepthSensor,
        ProximityDepthBufferSensor,
        ProximityVizDepthSensor,
        ProximityVizRGBSensor,
    )

    render_types = (
        CameraParameterSensor,
        CameraSensor,
        DepthSensor,
        ProximityDepthBufferSensor,
        ProximityVizDepthSensor,
        ProximityVizRGBSensor,
    )
    prox_types = (ProximityDepthBufferSensor,)
    orig = SensorSuite.get_observations
    n_prox_patch = _patch_all_named_get_observation("ProximityDepthBufferSensor")

    def get_observations(self, env, task, **kwargs):
        fresh = _policy_wants_fresh(task)
        last = getattr(self, "_fast_eval_last_obs", None)
        if (not fresh) and last is not None:
            obs = dict(last)
            for uuid, sensor in self.sensors.items():
                if _is_render_sensor(sensor, render_types):
                    continue
                obs[uuid] = sensor.get_observation(env=env, task=task, **kwargs)
            self._fast_eval_n_skip = getattr(self, "_fast_eval_n_skip", 0) + 1
            _heartbeat(self)
            return obs
        prox_names = [
            sensor.camera_name
            for sensor in self.sensors.values()
            if _is_prox_depth_sensor(sensor, prox_types)
        ]
        t0 = time.perf_counter()
        if prox_names and hasattr(env, "reset_proximity_depth_buffer"):
            ready = bool(getattr(task, "_fast_eval_substep_ready", False))
            frames = getattr(env, "_proximity_depth_frames", {}) or {}
            have = ready and all(len(frames.get(n) or []) > 0 for n in prox_names)
            if not have:
                env.reset_proximity_depth_buffer(prox_names)
                if hasattr(env, "record_proximity_depths"):
                    env.record_proximity_depths(prox_names)
                snapshot = True
            else:
                snapshot = False
            task._fast_eval_substep_ready = False
            if snapshot:
                task._eval_snapshot_renders = int(
                    getattr(task, "_eval_snapshot_renders", 0) or 0
                ) + 1
            else:
                task._eval_substep_queries = int(
                    getattr(task, "_eval_substep_queries", 0) or 0
                ) + 1
        obs = orig(self, env, task, **kwargs)
        dt = time.perf_counter() - t0
        self._fast_eval_last_obs = obs
        self._fast_eval_n_fresh = getattr(self, "_fast_eval_n_fresh", 0) + 1
        self._fast_eval_skin_s = getattr(self, "_fast_eval_skin_s", 0.0) + dt
        n_fresh = self._fast_eval_n_fresh
        if prox_names and (n_fresh <= 3 or n_fresh % 10 == 0):
            print(
                f"[act-eval-place] skin query #{n_fresh} n_cam={len(prox_names)} "
                f"{dt:.3f}s",
                flush=True,
            )
        _heartbeat(self)
        return obs

    SensorSuite.get_observations = get_observations
    print(
        "[act-eval-place] chunk-gated sensors ON "
        f"(RGB/skin only on chunk query; prox_get_obs patches={n_prox_patch})",
        flush=True,
    )


def _pixel_dirs_cam(h: int = 8, w: int = 8, fovy_deg: float = 45.0):
    yfov = float(np.tan(np.radians(fovy_deg) / 2.0))
    ys, xs = np.meshgrid(
        1.0 - 2.0 * (np.arange(h) + 0.5) / h,
        2.0 * (np.arange(w) + 0.5) / w - 1.0,
        indexing="ij",
    )
    dirs = np.stack([xs * yfov, ys * yfov, -np.ones_like(xs)], axis=-1)
    dirs = dirs.reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs.astype(np.float64)


def _install_raycast_proximity() -> None:
    import mujoco
    from molmo_spaces.env.env import CPUMujocoEnv

    orig = CPUMujocoEnv.record_proximity_depths
    dirs_by_fovy: dict[float, object] = {}
    geomgroup = np.ones((6, 1), dtype=np.uint8)
    geomgroup[2, 0] = 0
    cutoff = 10.0
    _multi_wants_normal = False
    if hasattr(mujoco, "mj_multiRay"):
        import inspect

        try:
            _multi_wants_normal = "normal" in inspect.signature(
                mujoco.mj_multiRay
            ).parameters
        except (TypeError, ValueError):
            _multi_wants_normal = True

    def record_proximity_depths(self, camera_names):
        model = self.mj_model
        data = self.current_data
        for name in camera_names:
            full = (
                self._proximity_cam_full_name(name)
                if hasattr(self, "_proximity_cam_full_name")
                else None
            )
            if not full:
                orig(self, [name])
                continue
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, full)
            if cid < 0:
                orig(self, [name])
                continue
            fovy = float(model.cam_fovy[cid])
            dirs_cam = dirs_by_fovy.get(fovy)
            if dirs_cam is None:
                dirs_cam = _pixel_dirs_cam(fovy_deg=fovy)
                dirs_by_fovy[fovy] = dirs_cam
            nray = int(dirs_cam.shape[0])
            pos = np.asarray(data.cam_xpos[cid], dtype=np.float64)
            rot = np.asarray(data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
            dirs_w = dirs_cam @ rot.T
            look = -rot[:, 2]
            pnt = np.ascontiguousarray(
                (pos + 1e-4 * look).reshape(3, 1), dtype=np.float64
            )
            geomid = np.zeros((nray, 1), dtype=np.int32)
            dist = np.empty((nray, 1), dtype=np.float64)
            vec = np.ascontiguousarray(dirs_w.reshape(-1, 1), dtype=np.float64)
            if hasattr(mujoco, "mj_multiRay"):
                if _multi_wants_normal:
                    mujoco.mj_multiRay(
                        model, data, pnt, vec, geomgroup, 1, -1,
                        geomid, dist, None, nray, cutoff,
                    )
                else:
                    mujoco.mj_multiRay(
                        model, data, pnt, vec, geomgroup, 1, -1,
                        geomid, dist, nray, cutoff,
                    )
            else:
                one_gid = np.zeros((1, 1), dtype=np.int32)
                for i in range(nray):
                    dist[i, 0] = mujoco.mj_ray(
                        model, data, pnt,
                        np.ascontiguousarray(dirs_w[i].reshape(3, 1), dtype=np.float64),
                        geomgroup, 1, -1, one_gid,
                    )
            depth = dist.reshape(8, 8).astype(np.float32)
            depth[depth < 0] = np.float32(cutoff)
            self._proximity_depth_frames.setdefault(name, []).append(depth)

    CPUMujocoEnv.record_proximity_depths = record_proximity_depths
    print(f"{_LOG} proximity mj_multiRay ON", flush=True)


def _configure_eval_cameras(eval_cfg, *, need_skin: bool, period_ms: float) -> None:
    eval_cfg.proximity_sensor_period_ms = float(period_ms)
    cams = []
    for cam in list(eval_cfg.camera_config.cameras):
        is_prox = bool(getattr(cam, "is_proximity_sensor", False))
        if (not need_skin) and is_prox:
            continue
        if not is_prox:
            update = {"record_depth": False}
            if hasattr(cam, "model_copy"):
                cam = cam.model_copy(update=update)
            elif hasattr(cam, "copy"):
                cam = cam.copy(update=update)
            else:
                cam.record_depth = False
        cams.append(cam)
    eval_cfg.camera_config.cameras = cams
    n_prox = sum(1 for c in cams if getattr(c, "is_proximity_sensor", False))
    print(
        f"[act-eval-place] cameras={len(cams)} proximity={n_prox} "
        f"period_ms={eval_cfg.proximity_sensor_period_ms} "
        f"({'0=snapshot' if float(period_ms) <= 0 else 'train-substeps'})"
    )


def _record_place_metric(
    task, success: bool, audit: dict, *, extra: dict | None = None
) -> None:
    frames = audit.get("frames_with_contact") or {}
    totals = audit.get("contact_class_totals") or {}
    bar_frames = int(frames.get("hazard_bar") or 0)
    other_frames = int(frames.get("other_environment") or 0)
    clutter_frames = int(frames.get("clutter") or 0)
    suite = getattr(task, "sensor_suite", None) or getattr(task, "_sensor_suite", None)
    n_fresh = int(getattr(suite, "_fast_eval_n_fresh", 0) or 0)
    n_skip = int(getattr(suite, "_fast_eval_n_skip", 0) or 0)
    policy = getattr(task, "_registered_policy", None)
    rec = {
        "episode_idx": int((extra or {}).get("episode_idx", len(_EPISODE_METRICS))),
        "success": int(success),
        "hit_bar": int(bar_frames > 0),
        "bar_contact_frames": bar_frames,
        "other_environment_frames": other_frames,
        "clutter_frames": clutter_frames,
        "collision_free": int(bool(audit.get("collision_free"))),
        "collision_free_task_success": int(bool(success) and bool(audit.get("collision_free"))),
        "gripper_close_commanded": int(bool(getattr(policy, "gripper_close_commanded", False))),
        "intrusion_side": str(
            (getattr(task, "scene_params", {}) or {}).get("pact_intrusion_side") or ""
        ),
        "contact_class_totals": totals,
        "first_contact_step": audit.get("first_contact_step") or {},
        "sensor_fresh_renders": n_fresh,
        "sensor_skipped_renders": n_skip,
        "snapshot_renders": int(getattr(task, "_eval_snapshot_renders", 0) or 0),
        "substep_queries": int(getattr(task, "_eval_substep_queries", 0) or 0),
        "arm": "PACT" if getattr(getattr(policy, "pc", None), "use_proximity", False) else "ACT",
    }
    if extra:
        rec.update(extra)
    _EPISODE_METRICS.append(rec)
    print(
        f"[act-eval-place] ep{rec['episode_idx']:03d} success={success} "
        f"hit_bar={rec['hit_bar']} collision_free={rec['collision_free']} "
        f"strict={rec['collision_free_task_success']} "
        f"fresh={n_fresh} skip={n_skip} cell={rec.get('cell')}",
        flush=True,
    )
    if _METRICS_JSONL is not None:
        with _METRICS_JSONL.open("a") as handle:
            handle.write(json.dumps(rec) + "\n")


def _summarize_place_metrics() -> dict | None:
    if not _EPISODE_METRICS:
        return None
    n = len(_EPISODE_METRICS)
    successes = sum(int(m["success"]) for m in _EPISODE_METRICS)
    collision_free = sum(int(m["collision_free"]) for m in _EPISODE_METRICS)
    strict = sum(int(m.get("collision_free_task_success", 0)) for m in _EPISODE_METRICS)
    bar_hits = sum(int(m["hit_bar"]) for m in _EPISODE_METRICS)
    grip = sum(int(m.get("gripper_close_commanded", 0)) for m in _EPISODE_METRICS)
    return {
        "episodes": n,
        "success": successes,
        "success_rate": successes / n,
        "bar_hits": bar_hits,
        "bar_hit_rate": bar_hits / n,
        "collision_free": collision_free,
        "collision_free_rate": collision_free / n,
        "collision_free_task_success": strict,
        "collision_free_task_success_rate": strict / n,
        "gripper_close_commanded": grip,
        "gripper_close_rate": grip / n,
        "episodes_detail": list(_EPISODE_METRICS),
    }


@contextmanager
def _detr_argv(ckpt_dir: str, seed: int):
    orig = sys.argv
    sys.argv = [
        orig[0] if orig else "eval_act_v6.py",
        "--ckpt_dir", ckpt_dir,
        "--policy_class", "ACT",
        "--task_name", "obstacle_baseline",
        "--seed", str(seed),
        "--num_epochs", "1",
    ]
    try:
        yield
    finally:
        sys.argv = orig


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_bool(value) -> bool:
    arr = np.asarray(value)
    if arr.size == 0:
        return False
    return bool(arr.reshape(-1)[0])


def _left_pad_hist(frames: list[np.ndarray], length: int = PROX_CAUSAL_STEPS) -> np.ndarray:
    if not frames:
        raise ValueError("proximity history is empty at encode time")
    block = np.stack(frames, axis=0)
    if len(block) < length:
        block = np.concatenate(
            (np.repeat(block[:1], length - len(block), axis=0), block), axis=0
        )
    return block[-length:]


def _chunk_from_weights(weights: dict) -> int:
    key = "model.query_embed.weight"
    if key not in weights:
        raise SystemExit(f"{_LOG} checkpoint missing {key}")
    return int(weights[key].shape[0])


def _load_state_dict(ckpt_path: Path) -> dict:
    try:
        weights = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        weights = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        weights = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(weights, dict):
        raise SystemExit(f"{_LOG} checkpoint is not a state dict: {ckpt_path}")
    return weights


def _apply_training_arch(pc, ckpt_dir: Path) -> None:
    training = ckpt_dir / "training_config.json"
    if not training.is_file():
        return
    saved = json.loads(training.read_text())
    policy_config = saved.get("policy_config") or {}
    for name in (
        "hidden_dim", "dim_feedforward", "enc_layers", "dec_layers",
        "nheads", "backbone", "state_dim", "action_dim", "kl_weight",
    ):
        if name in policy_config:
            setattr(pc, name, policy_config[name])


class FrozenACTPolicy(InferencePolicy):
    """Open-loop chunk ACT. Query-step skin history by default. Local to this file."""

    def __init__(self, exp_config, task=None) -> None:
        super().__init__(exp_config)
        self.task = task
        pc: ACTPolicyConfig = exp_config.policy_config
        self.pc = pc
        self.ckpt_path = str(Path(pc.ckpt_dir) / pc.ckpt_name)
        self.stats_path = str(Path(pc.ckpt_dir) / "dataset_stats.pkl")
        self.history_mode = str(getattr(pc, "history_mode", "query"))
        self.exec_horizon = int(getattr(pc, "exec_horizon", pc.chunk_size) or pc.chunk_size)
        self._step = 0
        self._pending_chunks: list[tuple[int, np.ndarray]] = []
        self._policy = None
        self._stats = None
        self._prox_encoder = None
        self._prox_hist: list[np.ndarray] = []
        self._prox_pcfg: dict | None = None
        self._encoder_path: str | None = None
        self.gripper_close_commanded = False

    def reset(self) -> None:
        self._step = 0
        self._pending_chunks.clear()
        self._prox_hist = []
        self.gripper_close_commanded = False

    def prepare_model(self, model_name: str | None = None) -> None:
        pc = self.pc
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
        }
        if pc.use_proximity:
            ckpt = pc.prox_encoder_ckpt or None
            k = int(pc.prox_tokens_per_sensor)
            if is_geometry_feature(pc.prox_feature) and k == 8:
                k = 1
            pcfg_path = Path(pc.ckpt_dir) / "prox_config.json"
            if pcfg_path.is_file():
                self._prox_pcfg = json.loads(pcfg_path.read_text())
                load_kwargs = resolve_act_encoder_load(
                    pc.ckpt_dir, self._prox_pcfg, ckpt or ""
                )
            else:
                load_kwargs = {
                    "checkpoint": ckpt,
                    "frozen": not bool(getattr(pc, "finetune_prox_encoder", False)),
                    "policy_tap": getattr(pc, "prox_policy_tap", None) or None,
                }
            self._encoder_path = load_kwargs.get("checkpoint")
            self._prox_encoder = build_pact_encoder(
                pc.prox_feature,
                device="cuda",
                layout=getattr(pc, "prox_layout", "per_sensor"),
                tokens_per_sensor=k,
                **load_kwargs,
            )
            if self._prox_encoder is not None:
                self._prox_encoder.eval()
                policy_config["n_proximity_sensors"] = self._prox_encoder.n_act_sensors
                policy_config["prox_tokens_per_sensor"] = k
                policy_config["prox_feat_dim"] = self._prox_encoder.act_feat_dim
            self._prox_pool = getattr(pc, "prox_pool", "min")
        with _detr_argv(self.pc.ckpt_dir, self.pc.seed):
            policy = ACTPolicy(policy_config)
        sd = torch.load(self.ckpt_path, map_location="cuda")
        policy.load_state_dict(sd)
        policy.cuda()
        policy.eval()
        self._policy = policy
        with open(self.stats_path, "rb") as handle:
            self._stats = pickle.load(handle)
        print(
            f"{_LOG} loaded {self.ckpt_path} history={self.history_mode} "
            f"exec_horizon={self.exec_horizon} n_prox_tokens="
            f"{policy_config.get('n_proximity_sensors', 0) * policy_config.get('prox_tokens_per_sensor', 1)}"
        )

    def obs_to_model_input(self, obs):
        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        return obs

    def needs_fresh_camera_observation(self) -> bool:
        if not self._pending_chunks:
            return True
        start, chunk = self._pending_chunks[0]
        age = self._step - start
        return not (0 <= age < len(chunk))

    def _wants_prox_prefetch(self, age: int, chunk_len: int) -> bool:
        """True on the last 7 idle chunk steps plus the next query (8 frames)."""
        if self.history_mode != "consecutive":
            return False
        if self._prox_encoder is None or not is_geometry_feature(self.pc.prox_feature):
            return False
        n = PROX_CAUSAL_STEPS
        if n <= 1:
            return False
        return age >= chunk_len - (n - 1)

    def needs_fresh_proximity_observation(self) -> bool:
        if not self.pc.use_proximity:
            return False
        if self.history_mode == "consecutive" and is_geometry_feature(self.pc.prox_feature):
            if not self._pending_chunks:
                return True
            start, chunk = self._pending_chunks[0]
            age = self._step - start
            return self._wants_prox_prefetch(age, len(chunk))
        return self.needs_fresh_camera_observation()

    def needs_fresh_policy_observation(self) -> bool:
        return self.needs_fresh_camera_observation() or self.needs_fresh_proximity_observation()

    def record_proximity_observation(self, obs) -> None:
        if self._prox_encoder is None:
            return
        if not is_geometry_feature(self.pc.prox_feature) and self.history_mode != "consecutive":
            return
        if self.history_mode == "query" and not self.needs_fresh_camera_observation():
            return
        if self.history_mode == "consecutive" and self._pending_chunks:
            start, chunk = self._pending_chunks[0]
            age = self._step - start
            if not self._wants_prox_prefetch(age, len(chunk)):
                return
        frame = stack_obs_proximity(
            obs, self._prox_encoder.sensor_order, pool=self._prox_pool
        )
        self._prox_hist.append(np.array(frame, copy=True))
        self._prox_hist = self._prox_hist[-PROX_CAUSAL_STEPS:]

    def inference_model(self, obs):
        if self._policy is None:
            self.prepare_model()
        pc = self.pc
        stats = self._stats
        if self._pending_chunks:
            start, chunk = self._pending_chunks[0]
            k = self._step - start
            if 0 <= k < len(chunk):
                if self._wants_prox_prefetch(k, len(chunk)):
                    self.record_proximity_observation(obs)
                return chunk[k]

        self.record_proximity_observation(obs)

        arm = np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32)
        grip = np.asarray((obs["qpos"].get("gripper") or [0.0, 0.0])[:2], dtype=np.float32)
        qpos = np.concatenate([arm, grip], axis=0).astype(np.float32)
        qpos = (qpos - stats["qpos_mean"]) / stats["qpos_std"]
        qpos_t = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

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

        proximity_positions = None
        if self._prox_encoder is not None:
            if is_geometry_feature(pc.prox_feature):
                hist = _left_pad_hist(self._prox_hist)
                proximity_positions = encode_for_act(
                    self._prox_encoder,
                    torch.from_numpy(hist).float().cuda().unsqueeze(0),
                )
            else:
                prox_np = stack_obs_proximity(
                    obs, self._prox_encoder.sensor_order, pool=self._prox_pool
                )
                prox_t = torch.from_numpy(prox_np).float().cuda().unsqueeze(0)
                proximity_positions = encode_for_act(self._prox_encoder, prox_t)
            if pc.prox_ablate == "zero":
                proximity_positions = torch.zeros_like(proximity_positions)
            if self._step == 0:
                print(
                    f"{_LOG} proximity ON history={self.history_mode} "
                    f"prefetch={PROX_CAUSAL_STEPS if self.history_mode == 'consecutive' else 0} "
                    f"feature={pc.prox_feature} ablate={pc.prox_ablate} "
                    f"exec_horizon={self.exec_horizon}"
                )

        with torch.no_grad():
            a_hat = self._policy(qpos_t, image_t, proximity_positions=proximity_positions)
        new_chunk = a_hat.squeeze(0).cpu().numpy()
        new_chunk = new_chunk * stats["action_std"] + stats["action_mean"]
        k = max(1, min(int(self.exec_horizon), int(len(new_chunk))))
        self._pending_chunks = [(self._step, new_chunk[:k])]
        return new_chunk[0]

    def model_output_to_action(self, model_output):
        arm = np.asarray(model_output[:7], dtype=np.float32)
        gripper_raw = float(model_output[7]) if len(model_output) >= 8 else 0.0
        gripper = 0.0 if gripper_raw < 127.5 else 255.0
        if gripper == 255.0:
            self.gripper_close_commanded = True
        return {"arm": arm, "gripper": np.asarray([gripper], dtype=np.float32)}

    def get_action(self, obs):
        action = super().get_action(obs)
        self._step += 1
        return action


class ACTPolicyConfig(BasePolicyConfig):
    policy_cls: type = FrozenACTPolicy
    policy_type: str = "learned"
    ckpt_dir: str = ""
    ckpt_name: str = "policy_best.ckpt"
    image_h: int = 240
    image_w: int = 320
    camera_names: tuple[str, ...] = ("exo_camera_1", "wrist_camera")
    chunk_size: int = 50
    exec_horizon: int = 50
    history_mode: str = "query"
    kl_weight: int = 10
    hidden_dim: int = 512
    dim_feedforward: int = 3200
    enc_layers: int = 4
    dec_layers: int = 7
    nheads: int = 8
    state_dim: int = 9
    action_dim: int = 8
    backbone: str = "resnet18"
    lr: float = 1e-5
    lr_backbone: float = 1e-5
    seed: int = 0
    use_proximity: bool = False
    prox_encoder_ckpt: str = ""
    prox_feature: str = "raw"
    prox_layout: str = "per_sensor"
    prox_pool: str = "min"
    prox_tokens_per_sensor: int = 8
    finetune_prox_encoder: bool = False
    prox_policy_tap: str = ""
    prox_ablate: str = "none"


class V6EvalConfig(FrankaSkinPACTCollisionCorridorConfig):
    policy_config: ACTPolicyConfig = ACTPolicyConfig()
    task_type: str = "pick_and_place"
    task_config: PickAndPlaceTaskConfig = PickAndPlaceTaskConfig(task_cls=PickAndPlaceTask)
    task_horizon: int | None = 1050
    viz_sensor_rgb: bool = False
    filter_for_successful_trajectories: bool = False
    use_wandb: bool = False
    num_workers: int = 1
    save_videos: bool = False
    use_passive_viewer: bool = False


def _disable_action_noise(eval_cfg) -> None:
    if hasattr(eval_cfg.robot_config, "action_noise_config"):
        noise = eval_cfg.robot_config.action_noise_config
        if noise is not None and hasattr(noise, "enabled"):
            noise.enabled = False


def _set_cfg(eval_cfg, name: str, value) -> None:
    fields = getattr(type(eval_cfg), "model_fields", None)
    if isinstance(fields, dict) and name not in fields:
        return
    setattr(eval_cfg, name, value)


def _build_eval_rows(num_rollouts: int, spread_cells: bool, house_ind: int) -> list[dict]:
    if spread_cells:
        per_cell, total = spread_episode_count(num_rollouts)
        rows = []
        for cell_i in range(_N_CELLS):
            family, side, pose = v1010_cell(cell_i)
            for k in range(per_cell):
                row = build_row(family, side, pose, attempt_index=k)
                row["role_index"] = len(rows)
                row["eval_master_seed"] = EVAL_MASTER_SEED
                rows.append(row)
        return rows[:total]
    family, side, pose = v1010_cell(house_ind)
    rows = []
    for k in range(num_rollouts):
        row = build_row(family, side, pose, attempt_index=k)
        row["role_index"] = k
        rows.append(row)
    return rows


def _bind_scene(eval_cfg, row: dict, scenes_dir: Path) -> Path:
    pose = str(row.get("pose_id") or "center")
    xml = scenes_dir / V1010_SCENE_BY_POSE[pose]["filename"]
    row["environment_version"] = _ENV_VERSION
    row["pact_v106_scene_sha256"] = hashlib.sha256(xml.read_bytes()).hexdigest()
    eval_cfg.task_sampler_config.scene_xml_paths = [str(xml)] * 2
    return xml


def _sample_row(eval_cfg, row: dict, scenes_dir: Path):
    xml = _bind_scene(eval_cfg, row, scenes_dir)
    last_error = None
    max_retries = int(row.get("max_sampling_retries", 8))
    for retry_index in range(max_retries + 1):
        seed = {
            "seed_u32": int(row["task_seed_u32"]) + retry_index,
            "seed_u64": int(row["task_seed_u64"]) + retry_index,
        }
        sampler = PactPlaceCorridorV1010TwoObjectSampler(eval_cfg)
        sampler.seed_task_sampling(seed["seed_u32"])
        sampler.set_pact_manifest_row(row)
        try:
            task = sampler.sample_task(
                house_index=int(row.get("scene_template_house_index") or 1)
            )
        except HouseInvalidForTask as exc:
            last_error = f"HouseInvalidForTask: {exc.reason}"
            sampler.close()
            continue
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            sampler.close()
            continue
        if task is None:
            last_error = "sample_task returned None"
            sampler.close()
            continue
        return task, sampler, seed, xml
    raise RuntimeError(
        f"row {row.get('role_index')} sampling failed after "
        f"{max_retries + 1} tries: {last_error}"
    )


def _default_history(prox_feature: str | None) -> str:
    if prox_feature and is_geometry_feature(prox_feature):
        return "consecutive"
    return "query"


def _fill_policy_config(eval_cfg, args, cameras: tuple[str, ...], chunk: int) -> None:
    pc = ACTPolicyConfig()
    pc.ckpt_dir = str(Path(args.ckpt_dir).resolve())
    pc.ckpt_name = args.ckpt_name
    pc.chunk_size = chunk
    pc.exec_horizon = int(getattr(args, "exec_horizon", None) or chunk)
    pc.camera_names = cameras
    pc.seed = args.seed_base
    _apply_training_arch(pc, Path(pc.ckpt_dir))
    prox_cfg_path = Path(pc.ckpt_dir) / "prox_config.json"
    if prox_cfg_path.exists():
        pcfg = json.loads(prox_cfg_path.read_text())
        pc.use_proximity = True
        pc.prox_feature = pcfg.get("prox_feature", "raw")
        pc.prox_layout = pcfg.get("prox_layout", "per_sensor")
        pc.prox_pool = pcfg.get("prox_pool", "min")
        pc.prox_tokens_per_sensor = int(pcfg.get("prox_tokens_per_sensor", 8))
        pc.prox_encoder_ckpt = pcfg.get("prox_encoder_ckpt", "")
        pc.finetune_prox_encoder = bool(pcfg.get("finetune_prox_encoder", False))
        pc.prox_policy_tap = pcfg.get("prox_policy_tap") or ""
    pc.prox_ablate = str(getattr(args, "prox_ablate", "none") or "none")
    if args.history in (None, "auto"):
        pc.history_mode = _default_history(pc.prox_feature if pc.use_proximity else None)
    else:
        pc.history_mode = args.history
    print(
        f"{_LOG} PACT={'ON' if pc.use_proximity else 'OFF'} "
        f"feature={pc.prox_feature if pc.use_proximity else '-'} "
        f"K={pc.prox_tokens_per_sensor} history={pc.history_mode} "
        f"ablate={pc.prox_ablate} exec_horizon={pc.exec_horizon}"
    )
    eval_cfg.policy_config = pc


def _prox_camera_names(task) -> list[str]:
    names = list(getattr(task, "_eval_prox_cam_names", None) or [])
    if names:
        return names
    return list(getattr(task, "_proximity_camera_names", None) or [])


def _disarm_prox_cameras(task) -> None:
    names = list(getattr(task, "_proximity_camera_names", None) or [])
    if names:
        task._eval_prox_cam_names = list(names)
    task._proximity_camera_names = []
    task._fast_eval_substep_ready = False


def _reset_prox_buffer(task) -> None:
    env = getattr(task, "_env", None)
    names = _prox_camera_names(task)
    if env is None or not names:
        return
    if hasattr(env, "reset_proximity_depth_buffer"):
        env.reset_proximity_depth_buffer(names)


def _grab_exo_rgb(task) -> np.ndarray | None:
    env = getattr(task, "_env", None)
    if env is None or not hasattr(env, "render_rgb_frame"):
        return None
    try:
        frame = env.render_rgb_frame(_EXO_CAM)
    except Exception:
        return None
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class _ExoMp4:
    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = float(fps)
        self._writer: cv2.VideoWriter | None = None
        self._wh: tuple[int, int] | None = None
        self.n_frames = 0

    def write(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        if self._writer is None:
            h, w = int(frame.shape[0]), int(frame.shape[1])
            w -= w % 2
            h -= h % 2
            frame = frame[:h, :w]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
            )
            if not writer.isOpened():
                raise RuntimeError(f"{_LOG} VideoWriter failed for {self.path}")
            self._writer = writer
            self._wh = (w, h)
        w, h = self._wh
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.n_frames += 1

    def close(self) -> Path | None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.n_frames <= 0:
            return None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            tmp = self.path.with_name(self.path.stem + ".h264tmp.mp4")
            result = subprocess.run(
                [
                    ffmpeg, "-y", "-loglevel", "error", "-i", str(self.path),
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "fast", "-crf", "20", "-movflags", "+faststart",
                    str(tmp),
                ],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and tmp.is_file():
                tmp.replace(self.path)
            elif tmp.exists():
                tmp.unlink()
        return self.path


def _rollout(
    task, policy, horizon: int, *, skin_substeps: str, exo_video: Path | None = None,
    video_fps: float = 1000.0 / 66.0,
) -> tuple[bool, dict]:
    policy.reset()
    task._eval_snapshot_renders = 0
    task._eval_substep_queries = 0
    writer = _ExoMp4(exo_video, video_fps) if exo_video is not None else None
    extra: dict = {}
    try:
        if skin_substeps == "train":
            _reset_prox_buffer(task)
        observation, _ = task.reset()
        if writer is not None:
            writer.write(_grab_exo_rgb(task))
        for _step in range(int(horizon)):
            if hasattr(task, "is_done") and task.is_done():
                break
            action = policy.get_action(observation)
            if skin_substeps == "train" and policy.needs_fresh_policy_observation():
                names = _prox_camera_names(task)
                task._proximity_camera_names = list(names)
                task._fast_eval_substep_ready = True
            else:
                task._proximity_camera_names = []
                task._fast_eval_substep_ready = False
            observation, _, terminal, truncated, _ = task.step(action)
            task._proximity_camera_names = []
            if writer is not None:
                writer.write(_grab_exo_rgb(task))
            if getattr(task, "observation_cache", None):
                task.observation_cache[-1] = [{}]
            if _one_bool(terminal) or _one_bool(truncated):
                break
        return _one_bool(task.judge_success()), extra
    finally:
        if writer is not None:
            path = writer.close()
            extra["video_frames"] = int(writer.n_frames)
            if path is not None:
                extra["video_path"] = str(path)


def _load_resume(jsonl: Path) -> set[int]:
    done: set[int] = set()
    if not jsonl.is_file():
        return done
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        _EPISODE_METRICS.append(rec)
        if "role_index" in rec:
            done.add(int(rec["role_index"]))
        elif "episode_idx" in rec:
            done.add(int(rec["episode_idx"]))
    return done


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_dir", "--checkpoint-dir", dest="ckpt_dir", required=True)
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument("--molmo", "--molmospaces_root", dest="molmo", default=None)
    p.add_argument("--num_rollouts", type=int, default=2)
    p.add_argument("--house_ind", type=int, default=1)
    p.add_argument("--horizon", "--task_horizon", dest="horizon", type=int, default=1050)
    p.add_argument("--chunk_size", type=int, default=None)
    p.add_argument("--exec_horizon", type=int, default=None)
    p.add_argument("--skin_substeps", choices=("snapshot", "train"), default="snapshot")
    p.add_argument("--seed_base", type=int, default=EVAL_MASTER_SEED)
    p.add_argument("--skin", choices=("egl", "rays"), default="rays")
    p.add_argument(
        "--history",
        choices=("auto", "query", "consecutive"),
        default="auto",
        help="auto: consecutive for readout/geometry, query for raw/ACT.",
    )
    p.add_argument(
        "--prox_ablate",
        choices=("none", "zero"),
        default="none",
        help="Readout ablation. 'zero' keeps skin token count but feeds zeros.",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--spread_cells", action="store_true")
    p.add_argument("--row_start", type=int, default=0)
    p.add_argument("--row_end", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.molmo is not None and Path(args.molmo).resolve() != _MOLMO_ROOT:
        raise SystemExit(
            f"{_LOG} --molmo must be on the command line before import-time path setup"
        )

    ckpt_dir = Path(args.ckpt_dir).resolve()
    ckpt_path = ckpt_dir / args.ckpt_name
    stats_path = ckpt_dir / "dataset_stats.pkl"
    if not ckpt_path.is_file():
        raise SystemExit(f"{_LOG} missing {ckpt_path}")
    if not stats_path.is_file():
        raise SystemExit(f"{_LOG} missing {stats_path}")

    weights = _load_state_dict(ckpt_path)
    chunk = _chunk_from_weights(weights)
    if args.chunk_size is not None and int(args.chunk_size) != chunk:
        raise SystemExit(
            f"{_LOG} --chunk_size {args.chunk_size} != ckpt query_embed {chunk}"
        )
    exec_horizon = int(args.exec_horizon) if args.exec_horizon is not None else chunk
    args.exec_horizon = exec_horizon
    del weights

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = V6EvalConfig()
    _set_cfg(eval_cfg, "task_horizon", args.horizon)
    _set_cfg(eval_cfg, "end_on_success", False)
    _set_cfg(eval_cfg, "terminate_upon_success", False)
    _set_cfg(eval_cfg, "output_dir", output_dir)
    _set_cfg(eval_cfg, "num_workers", 1)
    _set_cfg(eval_cfg, "save_videos", False)
    _set_cfg(eval_cfg, "use_passive_viewer", False)
    _set_cfg(eval_cfg, "viz_sensor_rgb", False)
    eval_cfg.camera_config = FrankaSkinHybridCameraSystem()
    eval_cfg.task_sampler_config.task_sampler_class = PactPlaceCorridorV1010TwoObjectSampler
    eval_cfg.task_sampler_config.house_inds = [1]
    _disable_action_noise(eval_cfg)

    scenes_dir = resolve_v1010_scenes_dir(_MOLMO_ROOT)
    assert_v1010_scene_hashes(scenes_dir)
    cameras = _DEFAULT_CAMERAS
    _fill_policy_config(eval_cfg, args, cameras, chunk)
    pc = eval_cfg.policy_config
    if pc.history_mode == "consecutive" and args.skin == "egl":
        print(
            f"{_LOG} WARNING: --history consecutive with --skin egl renders "
            "40 EGL cameras every control step. Use --skin rays.",
            flush=True,
        )
    period_ms = _TRAIN_PERIOD_MS if args.skin_substeps == "train" else 0.0
    _configure_eval_cameras(
        eval_cfg, need_skin=bool(pc.use_proximity), period_ms=period_ms
    )
    _install_chunk_gated_sensors()
    if args.skin == "rays" and pc.use_proximity:
        _install_raycast_proximity()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(args.seed_base)

    global _METRICS_JSONL
    _EPISODE_METRICS.clear()
    _METRICS_JSONL = output_dir / "episodes.jsonl"
    summary_path = output_dir / "eval_summary.json"
    done = _load_resume(_METRICS_JSONL)

    policy = FrozenACTPolicy(eval_cfg)
    policy.prepare_model()
    horizon = int(eval_cfg.task_horizon)
    rows = _build_eval_rows(args.num_rollouts, args.spread_cells, args.house_ind)
    row_end = len(rows) if args.row_end is None else min(int(args.row_end), len(rows))
    row_start = max(0, int(args.row_start))
    rows = rows[row_start:row_end]
    print(
        f"{_LOG} molmospaces={_MOLMO_ROOT} env={_ENV_VERSION} "
        f"sampler=PactPlaceCorridorV1010TwoObjectSampler n={len(rows)} "
        f"rows[{row_start}:{row_end}] history={pc.history_mode} "
        f"skin={args.skin} skin_substeps={args.skin_substeps} "
        f"exec_horizon={exec_horizon} horizon={horizon}",
        flush=True,
    )

    dt_ms = float(getattr(eval_cfg, "policy_dt_ms", 66.0) or 66.0)
    video_fps = 1000.0 / dt_ms if dt_ms > 0 else 15.0
    video_dir = output_dir / "videos"

    def summary_payload() -> dict:
        collision = _summarize_place_metrics()
        return {
            "ckpt_dir": str(ckpt_dir),
            "ckpt_name": args.ckpt_name,
            "ckpt_sha256": _file_sha256(ckpt_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _file_sha256(Path(__file__).resolve()),
            "protocol_source": "eval_act_v1011d.py FrozenACTPolicy + Hub v6 two-object",
            "molmospaces_root": str(_MOLMO_ROOT),
            "task": "pact_pick_n_place_v6",
            "environment_version": _ENV_VERSION,
            "task_sampler_class": "PactPlaceCorridorV1010TwoObjectSampler",
            "camera_names": list(cameras),
            "num_rollouts": len(rows),
            "completed": len(_EPISODE_METRICS),
            "task_horizon": horizon,
            "chunk_size": chunk,
            "exec_horizon": exec_horizon,
            "spread_cells": bool(args.spread_cells),
            "protocol": {
                "history": pc.history_mode,
                "skin": args.skin,
                "skin_substeps": args.skin_substeps,
                "exec_horizon": exec_horizon,
                "chunk": chunk,
                "success": "terminal",
                "temporal_aggregation": False,
                "two_object": True,
            },
            "use_proximity": bool(pc.use_proximity),
            "prox_feature": pc.prox_feature if pc.use_proximity else None,
            "prox_ablate": pc.prox_ablate,
            "collision": collision,
            "success": None if collision is None else collision["success"],
            "total": None if collision is None else collision["episodes"],
            "success_rate": None if collision is None else collision["success_rate"],
        }

    try:
        for row in rows:
            role = int(row["role_index"])
            if role in done:
                print(f"{_LOG} skip existing role={role}", flush=True)
                continue
            sampler = None
            t0 = time.monotonic()
            try:
                task, sampler, seed, xml = _sample_row(eval_cfg, row, scenes_dir)
                _disarm_prox_cameras(task)
                if args.skin_substeps == "train":
                    _reset_prox_buffer(task)
                audit = PactPlaceContactAudit()
                task._contact_audit_hook = audit
                task.register_policy(policy)
                exo_path = None
                if args.save_video:
                    video_dir.mkdir(parents=True, exist_ok=True)
                    exo_path = video_dir / f"ep{role:03d}_{_EXO_CAM}.mp4"
                terminal, video_extra = _rollout(
                    task, policy, horizon,
                    skin_substeps=args.skin_substeps,
                    exo_video=exo_path,
                    video_fps=video_fps,
                )
                _record_place_metric(
                    task, bool(terminal), audit.summary(),
                    extra={
                        "episode_idx": role,
                        "role_index": role,
                        "cell": row.get("cell"),
                        "episode_id": row.get("episode_id"),
                        "scene_xml": str(xml),
                        "environment_version": _ENV_VERSION,
                        "seed_u64": int(seed["seed_u64"]),
                        **video_extra,
                    },
                )
                done.add(role)
                print(
                    f"{_LOG} row={role:03d} {row.get('cell')} "
                    f"ok={int(terminal)} {time.monotonic() - t0:.1f}s",
                    flush=True,
                )
            finally:
                if sampler is not None:
                    sampler.close()
            summary_path.write_text(json.dumps(summary_payload(), indent=2, default=str) + "\n")
    finally:
        collision = _summarize_place_metrics()
        if collision is not None:
            print(
                f"{_LOG} place {collision['success']}/{collision['episodes']} "
                f"({collision['success_rate']*100:.1f}%)  "
                f"strict {collision['collision_free_task_success']}/{collision['episodes']} "
                f"({collision['collision_free_task_success_rate']*100:.1f}%)  "
                f"collfree {collision['collision_free']}/{collision['episodes']} "
                f"({collision['collision_free_rate']*100:.1f}%)",
                flush=True,
            )
        summary_path.write_text(json.dumps(summary_payload(), indent=2, default=str) + "\n")
        print(f"{_LOG} wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
