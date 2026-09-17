"""Frozen ACT / PACT eval. Experiment eval for data/ dumps that are wired.

Protocol source: ``old_eval_act_place_corridor.py`` (commit 1bfe693). That
snapshot is the reference. Do not run it: it imports live
``ACTInferencePolicy`` and defaults to mj_multiRay. Do not import
``eval_act_obstacle.ACTInferencePolicy``. Do not edit the live corridor
evaluator, ``eval_pact.py``, or ``eval_place_fast_hooks.py``.
``scripts/pact.py eval`` / ``verify`` is parked. Convert/train still use
``pact.py``.

Wired ``--task`` today: hallway (``data/pact_place_corridor_v5``) and
v1011d (``data/pact_pick_n_place_v2/data/v1011d``). Other ``data/`` dumps
are not a ``--task`` yet (v12 overlay, v12.1, v107, mixed, table_smoke,
molmo-pi0 videos).

Headline protocol (defaults):
  open-loop chunk, no temporal aggregation
  skin history = last 8 *query* frames (JSON history_mode query_steps_train_mismatch)
  gated EGL skin (paper); ~15 min/ep is expected
  terminal judge_success at horizon; also log ever-success
  end_on_success off, metrics only

    conda activate mlspaces
    cd /home/jaydv/code/prox_learning
    export OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export MLSPACES_ASSETS_DIR="$PWD/assets"

    python eval_act.py \\
      --ckpt_dir submodules/act/ckpts/pact_place_corridor_v5/20260828_003136_pact_place_corridor_readout_s0 \\
      --task hallway --cameras wrist_camera --num_rollouts 2 \\
      --output_dir eval_output/simple_hallway_smoke
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

_REPO_ROOT = Path(__file__).resolve().parent
_ACT_DIR = _REPO_ROOT / "submodules" / "act"
_HALLWAY_MOLMO = Path("/home/jaydv/code/molmospaces-pact-place")
_V1011D_MOLMO = _REPO_ROOT / "submodules" / "molmospaces"


def _argv_value(*names: str, default: str | None = None) -> str | None:
    args = sys.argv[1:]
    for i, a in enumerate(args):
        for name in names:
            if a == name and i + 1 < len(args):
                return args[i + 1]
            if a.startswith(name + "="):
                return a.split("=", 1)[1]
    return default


_WIRED_TASKS = ("hallway", "v1011d")
_TASK_ALIASES = {
    "pact_place_corridor_v5": "hallway",
    "pact_pick_n_place_v2": "v1011d",
}
_UNWIRED_TASKS = {
    "v12": (
        "data/pact_pick_n_place_v2/data/v12 needs kitchen overlay, settle-park, "
        "and a pre-policy contact fix. Not wired in this file. "
        "scripts/pact.py eval is parked; do not use it."
    ),
    "v12.1": "data/pact_pick_n_place_v2/data/v12.1 is a 5-ep table-cam preview, not a suite.",
    "v107": "data/pact_place_corridor v107 dumps are not a --task yet.",
    "v107_spaced": (
        "Use repo-root eval_act_v107spaced.py. Not a --task in this file. "
        "data/pact_place_corridor/data/v107_spaced."
    ),
    "v1010": "data/pact_place_corridor/data/v1010 is not a --task yet.",
    "mixed": "data/mixed_v1011_clutter_geometry is viz / clutter-geometry, not this eval.",
    "table_smoke": "data/table_smoke is a 10-ep schema check. Do not eval as a suite.",
    "pi0": "data/molmo-pi0-eval-videos is videos, not MuJoCo policy eval.",
}


def _canonical_task(raw: str | None) -> str:
    name = (raw or "hallway").strip()
    if name in _TASK_ALIASES:
        return _TASK_ALIASES[name]
    if name in _UNWIRED_TASKS:
        raise SystemExit(f"[eval_act] --task {name} not wired. {_UNWIRED_TASKS[name]}")
    if name not in _WIRED_TASKS:
        raise SystemExit(
            "[eval_act] --task must be hallway or v1011d "
            "(aliases: pact_place_corridor_v5, pact_pick_n_place_v2).\n"
            "  wired here: data/pact_place_corridor_v5, data/pact_pick_n_place_v2/data/v1011d\n"
            "  v107_spaced: python eval_act_v107spaced.py (not this file)\n"
            "  not wired: v12, v12.1, v107, v1010, mixed, table_smoke, molmo-pi0-eval-videos\n"
            "  scripts/pact.py eval is parked. Convert/train still use pact.py."
        )
    return name


_TASK = _canonical_task(_argv_value("--task", default="hallway"))

_DEFAULT_MOLMO = _HALLWAY_MOLMO if _TASK == "hallway" else _V1011D_MOLMO
_MOLMO_ROOT = Path(
    _argv_value("--molmo", "--molmospaces_root", default=str(_DEFAULT_MOLMO))
).resolve()
if not (_MOLMO_ROOT / "molmo_spaces").is_dir():
    raise SystemExit(
        f"[eval_act] molmospaces missing at {_MOLMO_ROOT}.\n"
        "  hallway: worktree 977acd6719a8c05b688d3e70da356d61dd32d259 "
        f"at {_HALLWAY_MOLMO}\n"
        "  v1011d: submodules/molmospaces (has V1011D) or a 70dedc0 worktree"
    )

os.environ.setdefault("MLSPACES_ASSETS_DIR", str(_REPO_ROOT / "assets"))
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
import subprocess
import time
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version

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
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.tasks.pact_place_contact_audit import PactPlaceContactAudit
from policy import ACTPolicy
from utils import set_seed
import pact_eval_sensor_keep
import pact_eval_wandb

_EPISODE_METRICS: list[dict] = []
_METRICS_JSONL: Path | None = None

# Copied from old_eval_act_place_corridor.py (1bfe693) lines 110-264.
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
    """Patch ``class_name`` only on ``molmo_spaces.*`` modules.

    Walking every ``sys.modules`` entry ``getattr``s HuggingFace lazy
    image processors and prints a wall of ``ProximityDepthBufferSensor``
    alias warnings. Those classes are not ours.
    """
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
    """Skip RGB / 8x8 skin EGL on steps the open-loop chunk will ignore.

    Copied from old_eval_act_place_corridor.py (1bfe693).
    """
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
            env.reset_proximity_depth_buffer(prox_names)
            if hasattr(env, "record_proximity_depths"):
                env.record_proximity_depths(prox_names)
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
    """Replace 40 EGL ``update_scene`` calls with ``mj_multiRay``.

    Copied from old_eval_act_place_corridor.py (1bfe693). Iteration only.
    """
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
                        model,
                        data,
                        pnt,
                        vec,
                        geomgroup,
                        1,
                        -1,
                        geomid,
                        dist,
                        None,
                        nray,
                        cutoff,
                    )
                else:
                    mujoco.mj_multiRay(
                        model,
                        data,
                        pnt,
                        vec,
                        geomgroup,
                        1,
                        -1,
                        geomid,
                        dist,
                        nray,
                        cutoff,
                    )
            else:
                one_gid = np.zeros((1, 1), dtype=np.int32)
                for i in range(nray):
                    dist[i, 0] = mujoco.mj_ray(
                        model,
                        data,
                        pnt,
                        np.ascontiguousarray(dirs_w[i].reshape(3, 1), dtype=np.float64),
                        geomgroup,
                        1,
                        -1,
                        one_gid,
                    )
            depth = dist.reshape(8, 8).astype(np.float32)
            depth[depth < 0] = np.float32(cutoff)
            self._proximity_depth_frames.setdefault(name, []).append(depth)

    CPUMujocoEnv.record_proximity_depths = record_proximity_depths
    print(
        "[act-eval-place] proximity mj_multiRay ON "
        "(iteration only; not the paper-table EGL rasterizer)",
        flush=True,
    )


def _configure_eval_cameras(
    eval_cfg, *, need_skin: bool, record_depth: bool = False
) -> None:
    """Policy-rate skin. RGB depth only when --save_first_frame (uuid {cam}_depth)."""
    eval_cfg.proximity_sensor_period_ms = 0.0
    cams = []
    for cam in list(eval_cfg.camera_config.cameras):
        is_prox = bool(getattr(cam, "is_proximity_sensor", False))
        if (not need_skin) and is_prox:
            continue
        if not is_prox:
            update = {"record_depth": bool(record_depth)}
            if hasattr(cam, "model_copy"):
                cam = cam.model_copy(update=update)
            elif hasattr(cam, "copy"):
                cam = cam.copy(update=update)
            else:
                cam.record_depth = bool(record_depth)
        cams.append(cam)
    eval_cfg.camera_config.cameras = cams
    n_prox = sum(1 for c in cams if getattr(c, "is_proximity_sensor", False))
    print(
        f"[act-eval-place] cameras={len(cams)} proximity={n_prox} "
        f"period_ms={eval_cfg.proximity_sensor_period_ms} (0=policy-rate) "
        f"record_depth={int(bool(record_depth))}"
    )


def _record_place_metric(
    task,
    success: bool,
    audit: dict,
    *,
    episode_idx: int | None = None,
    seed: int | None = None,
    ever_success: bool | None = None,
    extra: dict | None = None,
) -> None:
    """Bar / free fields copied from old_eval_act_place_corridor.py:509."""
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
        "episode_idx": int(episode_idx) if episode_idx is not None else len(_EPISODE_METRICS),
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
    }
    if seed is not None:
        rec["seed"] = int(seed)
    if ever_success is not None:
        rec["ever_success"] = int(bool(ever_success))
    if extra:
        rec.update(extra)
    _EPISODE_METRICS.append(rec)
    extra = f" renders={n_fresh} skip={n_skip}" if (n_fresh or n_skip) else ""
    ever_s = rec.get("ever_success")
    print(
        f"[act-eval-place] ep{rec['episode_idx']:03d} success={success} "
        f"ever={ever_s} "
        f"hit_bar={rec['hit_bar']} bar_frames={bar_frames} "
        f"other={other_frames} clutter={clutter_frames} "
        f"collision_free={rec['collision_free']} "
        f"grip_close={rec['gripper_close_commanded']} "
        f"side={rec['intrusion_side'] or '-'}{extra}",
        flush=True,
    )
    if _METRICS_JSONL is not None:
        with _METRICS_JSONL.open("a") as handle:
            handle.write(json.dumps(rec) + "\n")


def _summarize_place_metrics() -> dict | None:
    """Copied from old_eval_act_place_corridor.py:553 plus ever-success."""
    if not _EPISODE_METRICS:
        return None
    n = len(_EPISODE_METRICS)
    bar_hits = sum(int(m["hit_bar"]) for m in _EPISODE_METRICS)
    successes = sum(int(m["success"]) for m in _EPISODE_METRICS)
    collision_free = sum(int(m["collision_free"]) for m in _EPISODE_METRICS)
    strict = sum(int(m.get("collision_free_task_success", 0)) for m in _EPISODE_METRICS)
    if strict == 0:
        strict = sum(
            1 for m in _EPISODE_METRICS if int(m["success"]) and int(m["collision_free"])
        )
    grip = sum(int(m.get("gripper_close_commanded", 0)) for m in _EPISODE_METRICS)
    ever = sum(int(m.get("ever_success", m["success"])) for m in _EPISODE_METRICS)
    return {
        "episodes": n,
        "success": successes,
        "success_rate": successes / n,
        "ever_success": ever,
        "ever_success_rate": ever / n,
        "bar_hits": bar_hits,
        "bar_hit_rate": bar_hits / n,
        "collision_free": collision_free,
        "collision_free_rate": collision_free / n,
        "collision_rate": 1.0 - (collision_free / n),
        "collision_free_task_success": strict,
        "collision_free_task_success_rate": strict / n,
        "gripper_close_commanded": grip,
        "gripper_close_rate": grip / n,
        "episodes_detail": list(_EPISODE_METRICS),
    }


@contextmanager
def _detr_argv(ckpt_dir: str, seed: int):
    """Shield DETR's main.py:get_args_parser from this script's CLI flags."""
    orig = sys.argv
    sys.argv = [
        orig[0] if orig else "eval_act.py",
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
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _git_rev(path: Path) -> str | None:
    path = Path(path).resolve()
    if (path / ".git").exists():
        try:
            return subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
        except subprocess.CalledProcessError:
            return None
    marker = path / "runtime.json"
    if marker.is_file():
        return json.loads(marker.read_text()).get("revision")
    return None


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _one_bool(value) -> bool:
    arr = np.asarray(value)
    if arr.size == 0:
        return False
    return bool(arr.reshape(-1)[0])


def _left_pad_hist(frames: list[np.ndarray], length: int = 8) -> np.ndarray:
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
        raise SystemExit(f"[eval_act] checkpoint missing {key}")
    return int(weights[key].shape[0])


def _n_backbones(weights: dict) -> int:
    ids: set[int] = set()
    for name in weights:
        parts = str(name).split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "backbones" and parts[2].isdigit():
            ids.add(int(parts[2]))
    return len(ids)


def _load_state_dict(ckpt_path: Path) -> dict:
    try:
        weights = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        weights = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        weights = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(weights, dict):
        raise SystemExit(f"[eval_act] checkpoint is not a state dict: {ckpt_path}")
    return weights


def _resolve_cameras(args: argparse.Namespace, ckpt_dir: Path) -> tuple[str, ...]:
    training = ckpt_dir / "training_config.json"
    from_json: list[str] | None = None
    if training.is_file():
        saved = json.loads(training.read_text())
        pc = saved.get("policy_config") or {}
        names = pc.get("camera_names")
        if names:
            from_json = [str(n) for n in names]
    if args.cameras:
        cameras = tuple(args.cameras)
        if from_json is not None and list(cameras) != from_json:
            raise SystemExit(
                f"[eval_act] --cameras {list(cameras)} != training_config.json {from_json}"
            )
        return cameras
    if from_json is not None:
        return tuple(from_json)
    raise SystemExit(
        "[eval_act] pass --cameras (hallway: wrist_camera; "
        "v1011d: exo_camera_1 wrist_camera). Not stored in prox_config.json."
    )


def _apply_training_arch(pc, ckpt_dir: Path) -> None:
    training = ckpt_dir / "training_config.json"
    if not training.is_file():
        return
    saved = json.loads(training.read_text())
    policy_config = saved.get("policy_config") or {}
    for name in (
        "hidden_dim",
        "dim_feedforward",
        "enc_layers",
        "dec_layers",
        "nheads",
        "backbone",
        "state_dim",
        "action_dim",
        "kl_weight",
    ):
        if name in policy_config:
            setattr(pc, name, policy_config[name])


class FrozenACTPolicy(pact_eval_sensor_keep.SensorKeepMixin, InferencePolicy):
    """Open-loop chunk ACT. Query-step skin history by default. Local to this file."""

    def __init__(self, exp_config, task=None) -> None:
        super().__init__(exp_config)
        self.task = task
        pc: ACTPolicyConfig = exp_config.policy_config
        self.pc = pc
        self.ckpt_path = str(Path(pc.ckpt_dir) / pc.ckpt_name)
        self.stats_path = str(Path(pc.ckpt_dir) / "dataset_stats.pkl")
        self.history_mode = str(getattr(pc, "history_mode", "query"))
        self._step = 0
        self._pending_chunks: list[tuple[int, np.ndarray]] = []
        self._policy = None
        self._stats = None
        self._prox_encoder = None
        self._prox_hist: list[np.ndarray] = []
        self._prox_pcfg: dict | None = None
        self._encoder_path: str | None = None
        self.gripper_close_commanded = False
        self._sensor_keep_frac = 1.0
        self._sensor_mask_seed = 0
        self._sensor_mask_fixed = False
        self._sensor_keep_mask = None
        self._sensor_keep_info = None

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
                    pc.ckpt_dir, self._prox_pcfg, ckpt or "", policy_name=pc.ckpt_name
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
        print(f"[eval_act] loaded {self.ckpt_path}")

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

    def needs_fresh_proximity_observation(self) -> bool:
        if not self.pc.use_proximity:
            return False
        if self.history_mode == "consecutive" and is_geometry_feature(self.pc.prox_feature):
            return True
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
        frame = stack_obs_proximity(
            obs, self._prox_encoder.sensor_order, pool=self._prox_pool
        )
        frame = self.mask_proximity(frame)
        self._prox_hist.append(np.array(frame, copy=True))
        self._prox_hist = self._prox_hist[-8:]

    def inference_model(self, obs):
        if self._policy is None:
            self.prepare_model()
        pc = self.pc
        stats = self._stats
        if self._pending_chunks:
            start, chunk = self._pending_chunks[0]
            k = self._step - start
            if 0 <= k < len(chunk):
                if self.history_mode == "consecutive":
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
                prox_np = self.mask_proximity(prox_np)
                prox_t = torch.from_numpy(prox_np).float().cuda().unsqueeze(0)
                proximity_positions = encode_for_act(self._prox_encoder, prox_t)
            if self._step == 0:
                print(
                    f"[eval_act] proximity ON history={self.history_mode} "
                    f"feature={pc.prox_feature}"
                )

        with torch.no_grad():
            a_hat = self._policy(qpos_t, image_t, proximity_positions=proximity_positions)
        new_chunk = a_hat.squeeze(0).cpu().numpy()
        new_chunk = new_chunk * stats["action_std"] + stats["action_mean"]
        self._pending_chunks = [(self._step, new_chunk)]
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
    camera_names: tuple[str, ...] = ("wrist_camera",)
    chunk_size: int = 50
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


def _disable_action_noise(eval_cfg) -> None:
    if hasattr(eval_cfg.robot_config, "action_noise_config"):
        noise = eval_cfg.robot_config.action_noise_config
        if noise is not None and hasattr(noise, "enabled"):
            noise.enabled = False


def _build_hallway_cfg(args, output_dir: Path):
    from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
    try:
        from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
            FrankaSkinPACTCollisionCorridorConfig,
        )
        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV2Sampler
    except ImportError as exc:
        raise SystemExit(
            "[eval_act] hallway config missing on this molmospaces pin "
            f"({_MOLMO_ROOT}): {exc}. Use 977acd6 at {_HALLWAY_MOLMO}."
        ) from exc
    from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask

    class ACTPlaceCorridorEvalConfig(FrankaSkinPACTCollisionCorridorConfig):
        policy_config: ACTPolicyConfig = ACTPolicyConfig()
        task_type: str = "pick_and_place"
        task_config: PickAndPlaceTaskConfig = PickAndPlaceTaskConfig(task_cls=PickAndPlaceTask)
        task_horizon: int | None = 800
        viz_sensor_rgb: bool = False
        filter_for_successful_trajectories: bool = False
        use_wandb: bool = False
        num_workers: int = 1
        save_videos: bool = False
        use_passive_viewer: bool = False
        output_dir: Path = ASSETS_DIR / "datagen" / "act_place_corridor_eval"

        @property
        def tag(self) -> str:
            return "act_place_corridor_eval"

    eval_cfg = ACTPlaceCorridorEvalConfig()
    eval_cfg.task_horizon = args.horizon or 800
    eval_cfg.end_on_success = False
    eval_cfg.output_dir = output_dir
    eval_cfg.num_workers = 1
    eval_cfg.save_videos = False
    eval_cfg.use_passive_viewer = False
    _disable_action_noise(eval_cfg)
    import molmo_spaces as _ms

    scenes = Path(_ms.__file__).resolve().parent / "data_generation" / "custom_scenes"
    xml = scenes / "pact_place_corridor_v2.xml"
    if not xml.is_file():
        raise SystemExit(
            f"[eval_act] missing {xml}. Hallway needs molmospaces 977acd6 "
            f"(default {_HALLWAY_MOLMO}), not submodule main."
        )
    eval_cfg.task_sampler_config.task_sampler_class = PactPlaceCorridorV2Sampler
    eval_cfg.task_sampler_config.scene_xml_paths = [str(xml)] * 2
    eval_cfg.task_sampler_config.house_inds = [args.house_ind]
    eval_cfg.task_sampler_config.samples_per_house = 1
    return eval_cfg, xml, PactPlaceCorridorV2Sampler


def _build_v1011d_cfg(args, output_dir: Path):
    from molmo_spaces.configs.camera_configs import FrankaSkinHybridCameraSystem

    ood = False
    try:
        from molmo_spaces.data_generation.config.pact_place_datagen_configs import (
            FrankaSkinPactPlaceV1011DRandomizedClutterConfig,
        )

        eval_cfg = FrankaSkinPactPlaceV1011DRandomizedClutterConfig()
        sampler_cls = eval_cfg.task_sampler_config.task_sampler_class
    except ImportError:
        from molmo_spaces.data_generation.config.pact_place_datagen_configs import (
            FrankaSkinPactPlaceV1010FourObjectConfig,
        )
        from molmo_spaces.tasks.pact_place import PactPlaceCorridorV1010FourObjectSampler

        eval_cfg = FrankaSkinPactPlaceV1010FourObjectConfig()
        sampler_cls = PactPlaceCorridorV1010FourObjectSampler
        eval_cfg.task_sampler_config.task_sampler_class = sampler_cls
        ood = True
        print(
            "[eval_act] WARNING: no V1011D config on this molmospaces pin. "
            "Falling back to V1010 four-object sampler (OOD vs v1011d train). "
            "Point --molmo at submodules/molmospaces or 70dedc0.",
            flush=True,
        )
    eval_cfg.policy_config = ACTPolicyConfig()
    eval_cfg.task_horizon = args.horizon or 1050
    eval_cfg.end_on_success = False
    if hasattr(eval_cfg, "terminate_upon_success"):
        eval_cfg.terminate_upon_success = False
    eval_cfg.output_dir = output_dir
    eval_cfg.num_workers = 1
    eval_cfg.save_videos = False
    eval_cfg.use_passive_viewer = False
    eval_cfg.viz_sensor_rgb = False
    eval_cfg.camera_config = FrankaSkinHybridCameraSystem()
    _disable_action_noise(eval_cfg)
    eval_cfg.task_sampler_config.house_inds = [args.house_ind]
    eval_cfg.task_sampler_config.samples_per_house = 1
    paths = list(eval_cfg.task_sampler_config.scene_xml_paths or [])
    xml = Path(paths[args.house_ind % len(paths)]) if paths else Path("unknown.xml")
    eval_cfg._eval_act_ood = ood
    return eval_cfg, xml, sampler_cls


def _fill_policy_config(eval_cfg, args, cameras: tuple[str, ...], chunk: int) -> None:
    pc = ACTPolicyConfig()
    pc.ckpt_dir = str(Path(args.ckpt_dir).resolve())
    pc.ckpt_name = args.ckpt_name
    pc.chunk_size = chunk
    pc.camera_names = cameras
    pc.history_mode = args.history
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
        pc.finetune_prox_encoder = bool(pcfg.get("finetune_prox_encoder", False))
        pc.prox_policy_tap = pcfg.get("prox_policy_tap") or ""
        print(
            f"[eval_act] PACT ckpt -> proximity ON "
            f"(feature={pc.prox_feature}, layout={pc.prox_layout}, "
            f"K={pc.prox_tokens_per_sensor}, pool={pc.prox_pool})"
        )
    eval_cfg.policy_config = pc


def _rollout(
    task,
    policy,
    horizon: int,
    on_step=None,
    *,
    episode_idx: int = 0,
    seed: int = 0,
    save_first_frame: bool = False,
    output_dir: Path | None = None,
    cameras: tuple[str, ...] | None = None,
) -> tuple[bool, bool, int]:
    """Full-horizon loop. Headline success is terminal judge_success."""
    policy.reset()
    if hasattr(policy, "begin_sensor_keep"):
        policy.begin_sensor_keep(episode_idx)
    observation, _ = task.reset()
    if save_first_frame:
        if output_dir is None:
            raise SystemExit("[eval_act] --save_first_frame needs --output_dir")
        pact_eval_sensor_keep.dump_first_frame(
            obs=observation,
            policy=policy,
            cameras=cameras or tuple(policy.pc.camera_names),
            output_dir=Path(output_dir),
            episode_idx=episode_idx,
            seed=seed,
            prox_pool=str(getattr(policy, "_prox_pool", "min")),
        )
    ever = False
    last_step = 0
    for _step in range(int(horizon)):
        if hasattr(task, "is_done") and task.is_done():
            break
        action = policy.get_action(observation)
        observation, _, terminal, truncated, _ = task.step(action)
        last_step = int(_step)
        if on_step is not None:
            on_step(_step, task)
        if _one_bool(task.judge_success()):
            ever = True
        if getattr(task, "observation_cache", None):
            task.observation_cache[-1] = [{}]
        if _one_bool(terminal) or _one_bool(truncated):
            break
    terminal = _one_bool(task.judge_success())
    return terminal, ever or terminal, last_step


def _load_resume(jsonl: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if not jsonl.is_file():
        return done
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        _EPISODE_METRICS.append(rec)
        if "episode_idx" in rec and "seed" in rec:
            done.add((int(rec["episode_idx"]), int(rec["seed"])))
    return done


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _history_json_label(mode: str) -> str:
    if mode == "query":
        return "query_steps_train_mismatch"
    return "consecutive_control_steps"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_dir", "--checkpoint-dir", dest="ckpt_dir", required=True)
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument(
        "--task",
        required=True,
        choices=(*_WIRED_TASKS, *tuple(_TASK_ALIASES), *tuple(_UNWIRED_TASKS)),
    )
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument("--molmo", "--molmospaces_root", dest="molmo", default=None)
    p.add_argument("--num_rollouts", type=int, default=2)
    p.add_argument("--house_ind", type=int, default=1)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--chunk_size", type=int, default=None)
    p.add_argument("--seed_base", type=int, default=2026)
    p.add_argument("--skin", choices=("egl", "rays"), default="egl")
    p.add_argument("--history", choices=("query", "consecutive"), default="query")
    p.add_argument(
        "--output_dir",
        default="/home/jaydv/code/prox_learning/eval_output/eval_act",
    )
    pact_eval_sensor_keep.add_cli_flags(p)
    pact_eval_wandb.add_cli_flags(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.task = _canonical_task(args.task)
    if args.task != _TASK:
        raise SystemExit("[eval_act] --task must match the value used at import")
    if args.molmo is not None and Path(args.molmo).resolve() != _MOLMO_ROOT:
        raise SystemExit("[eval_act] --molmo must be on the command line before import-time path setup; got a mismatch")
    if args.history == "consecutive" and args.skin == "egl":
        print(
            "[eval_act] WARNING: --history consecutive with --skin egl renders "
            "40 EGL cameras every control step (~15 min/ep). Non-headline. Use --skin rays.",
            flush=True,
        )
    if args.history == "query" and args.skin == "egl":
        print(
            "[eval_act] skin=egl history=query is the paper path. "
            "~15 min/ep gated EGL is expected.",
            flush=True,
        )
    if args.history == "consecutive":
        print(
            "[eval_act] --history consecutive is non-headline "
            "(matches train raw_causal; not the n=50 paper table).",
            flush=True,
        )

    ckpt_dir = Path(args.ckpt_dir).resolve()
    ckpt_path = ckpt_dir / args.ckpt_name
    stats_path = ckpt_dir / "dataset_stats.pkl"
    if not ckpt_path.is_file():
        raise SystemExit(f"[eval_act] missing {ckpt_path}")
    if not stats_path.is_file():
        raise SystemExit(f"[eval_act] missing {stats_path}")

    cameras = _resolve_cameras(args, ckpt_dir)
    weights = _load_state_dict(ckpt_path)
    chunk = _chunk_from_weights(weights)
    if args.chunk_size is not None and int(args.chunk_size) != chunk:
        raise SystemExit(
            f"[eval_act] --chunk_size {args.chunk_size} != ckpt query_embed {chunk}"
        )
    n_bb = _n_backbones(weights)
    if n_bb > 1 and n_bb != len(cameras):
        raise SystemExit(
            f"[eval_act] checkpoint has {n_bb} backbones, --cameras has {len(cameras)}"
        )
    if n_bb == 1 and len(cameras) != 1:
        print(
            f"[eval_act] DETRVAE shares 1 backbone across {len(cameras)} cameras "
            f"{list(cameras)}",
            flush=True,
        )
    del weights

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "hallway":
        eval_cfg, xml, sampler_cls = _build_hallway_cfg(args, output_dir)
    else:
        eval_cfg, xml, sampler_cls = _build_v1011d_cfg(args, output_dir)
    _fill_policy_config(eval_cfg, args, cameras, chunk)
    _configure_eval_cameras(
        eval_cfg,
        need_skin=bool(eval_cfg.policy_config.use_proximity),
        record_depth=bool(args.save_first_frame),
    )
    _install_chunk_gated_sensors()
    if args.skin == "rays":
        if eval_cfg.policy_config.use_proximity:
            _install_raycast_proximity()
    elif eval_cfg.policy_config.use_proximity:
        print(
            "[eval_act] --skin egl: 40-cam rasterizer on query steps. "
            "Headline n=50 was this path (~17 fresh / ~785 skip per 800).",
            flush=True,
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(args.seed_base)

    global _METRICS_JSONL
    _EPISODE_METRICS.clear()
    _METRICS_JSONL = output_dir / "episodes.jsonl"
    summary_path = output_dir / "eval_summary.json"
    keep_protocol = pact_eval_sensor_keep.protocol_fields(args)
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text())
        old = saved.get("protocol") or {}
        mismatch = pact_eval_sensor_keep.keep_fields_mismatch(old, keep_protocol)
        if mismatch:
            raise SystemExit(
                f"[eval_act] refuse resume into {summary_path.parent}: {mismatch}. "
                "Use a new --output_dir."
            )
    done = _load_resume(_METRICS_JSONL)

    policy = FrozenACTPolicy(eval_cfg)
    policy.prepare_model()
    policy.configure_sensor_keep(
        keep_frac=float(args.sensor_keep_frac),
        mask_seed=pact_eval_sensor_keep.resolved_mask_seed(args),
        mask_fixed=bool(args.sensor_mask_fixed),
    )
    horizon = int(eval_cfg.task_horizon)
    sampler = sampler_cls(eval_cfg)
    history_label = _history_json_label(args.history)
    encoder_path = getattr(policy, "_encoder_path", None)
    wb = pact_eval_wandb.start(
        args,
        output_dir=output_dir,
        horizon=horizon,
        extra_config={
            "task": args.task,
            "ckpt_dir": str(ckpt_dir),
            "ckpt_name": args.ckpt_name,
            "camera_names": list(cameras),
            "skin": args.skin,
            "history": args.history,
            "house_ind": args.house_ind,
            "chunk_size": chunk,
            **keep_protocol,
            "sensor_mask_seed": pact_eval_sensor_keep.resolved_mask_seed(args),
            "save_first_frame": bool(args.save_first_frame),
        },
    )
    wb.log_running(_EPISODE_METRICS, _summarize_place_metrics())

    def summary_payload(n_done: int) -> dict:
        collision = _summarize_place_metrics()
        return {
            "ckpt_dir": str(ckpt_dir),
            "ckpt_name": args.ckpt_name,
            "ckpt_sha256": _file_sha256(ckpt_path),
            "encoder_path": encoder_path,
            "encoder_sha256": _file_sha256(Path(encoder_path)) if encoder_path else None,
            "script": str(Path(__file__).resolve()),
            "script_sha256": _file_sha256(Path(__file__).resolve()),
            "protocol_source": "old_eval_act_place_corridor.py@1bfe69398722d916cca5b1e5a6ae308e90f8b476",
            "molmospaces_root": str(_MOLMO_ROOT),
            "molmospaces_commit": _git_rev(_MOLMO_ROOT),
            "task": args.task,
            "task_sampler_class": sampler_cls.__name__,
            "v1011d_ood_four_object": bool(getattr(eval_cfg, "_eval_act_ood", False)),
            "scene_xml": str(xml),
            "camera_names": list(cameras),
            "house_ind": args.house_ind,
            "num_rollouts": args.num_rollouts,
            "completed": n_done,
            "task_horizon": horizon,
            "chunk_size": chunk,
            "protocol": {
                "history": args.history,
                "history_mode": history_label,
                "skin": args.skin,
                "chunk": chunk,
                "success": "terminal",
                "end_on_success": False,
                "temporal_aggregation": False,
                **keep_protocol,
            },
            "save_first_frame": bool(args.save_first_frame),
            "sensor_keep": pact_eval_sensor_keep.summary_keep_block(_EPISODE_METRICS),
            "package_versions": {
                "torch": torch.__version__,
                "mujoco": _pkg_version("mujoco"),
            },
            "collision": collision,
            "success": None if collision is None else collision["success"],
            "total": None if collision is None else collision["episodes"],
            "success_rate": None if collision is None else collision["success_rate"],
            "ever_success_rate": None if collision is None else collision["ever_success_rate"],
        }

    print(
        f"[eval_act] molmospaces={_MOLMO_ROOT} commit={_git_rev(_MOLMO_ROOT)} "
        f"sampler={sampler_cls.__name__} xml={Path(xml).name} cameras={cameras} "
        f"horizon={horizon} n={args.num_rollouts} skin={args.skin} "
        f"history={history_label} sensor_keep_frac={args.sensor_keep_frac} "
        f"sensor_mask_fixed={int(bool(args.sensor_mask_fixed))}",
        flush=True,
    )

    try:
        for i in range(args.num_rollouts):
            seed = int(args.seed_base) + i
            if (i, seed) in done:
                print(f"[eval_act] skip existing ep={i} seed={seed}", flush=True)
                _write_summary(summary_path, summary_payload(len(_EPISODE_METRICS)))
                continue
            set_seed(seed)
            sampler.seed_task_sampling(seed)
            task = sampler.sample_task(house_index=args.house_ind)
            if task is None:
                raise RuntimeError(f"sample_task returned None for ep={i} seed={seed}")
            audit = PactPlaceContactAudit()
            task._contact_audit_hook = audit
            task.register_policy(policy)
            t0 = time.monotonic()
            terminal, ever, last_step = _rollout(
                task,
                policy,
                horizon,
                on_step=lambda step, rolled, i=i: wb.maybe_log_step(i, step, rolled),
                episode_idx=i,
                seed=seed,
                save_first_frame=bool(args.save_first_frame),
                output_dir=output_dir,
                cameras=cameras,
            )
            print(
                f"[eval_act] ep={i} seed={seed} wall={time.monotonic() - t0:.1f}s "
                f"terminal={int(terminal)} ever={int(ever)}",
                flush=True,
            )
            _record_place_metric(
                task,
                bool(terminal),
                audit.summary(),
                episode_idx=i,
                seed=seed,
                ever_success=ever,
                extra=policy.sensor_keep_log(),
            )
            done.add((i, seed))
            _write_summary(summary_path, summary_payload(len(_EPISODE_METRICS)))
            wb.log_episode(
                episode=i,
                ep_step=last_step,
                rec=_EPISODE_METRICS[-1],
                summary=_summarize_place_metrics(),
                records=_EPISODE_METRICS,
            )
    finally:
        sampler.close()
        wb.finish()

    collision = _summarize_place_metrics()
    if collision is not None:
        print(
            f"[eval_act] terminal {collision['success']}/{collision['episodes']} "
            f"({collision['success_rate']*100:.1f}%)  "
            f"ever {collision['ever_success']}/{collision['episodes']} "
            f"({collision['ever_success_rate']*100:.1f}%)  "
            f"bar_hit {collision['bar_hits']}/{collision['episodes']} "
            f"({collision['bar_hit_rate']*100:.1f}%)  "
            f"collision_free {collision['collision_free']}/{collision['episodes']} "
            f"({collision['collision_free_rate']*100:.1f}%)",
            flush=True,
        )
    _write_summary(summary_path, summary_payload(len(_EPISODE_METRICS)))
    print(f"[eval_act] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
