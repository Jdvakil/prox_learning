"""Frozen ACT / PACT eval for v107_spaced only.

Do not import or edit ``eval_act.py`` (hallway n=50 lives there). Do not use
``eval_act_v1011d.py`` for this dump (randomized V10.11d clutter, different
env). Protocol source for the gate / open-loop chunk is
``old_eval_act_place_corridor.py`` (commit 1bfe693). Do not run that snapshot.
Do not import ``eval_act_obstacle.ACTInferencePolicy``. Do not edit the live
corridor evaluator, ``eval_pact.py``, or ``eval_place_fast_hooks.py``.

World: ``FrankaSkinPactPlaceV107SpacedBenchConfig`` +
``PactPlaceCorridorV107SpacedBenchSampler`` + the three
``pact_place_corridor_v10_7_{neg5,center,pos5}.xml`` files. Env version
``pact_place_corridor_v10_7_spaced_bench``. Molmo is this checkout's
``submodules/molmospaces`` (needs the V107 spaced sampler, commit
``4c6a215`` / ``main``).

Train hdf5 cameras are ``table_camera`` + ``wrist_camera`` (210 eps under
``act_style_data/pact_place_corridor/data/v107_spaced/accepted``). The public
datagen config is wrist-only; this script enables the hybrid table RGB cam
and names it ``table_camera`` so policy obs keys match convert.

Headline defaults: ``--skin_substeps snapshot``, ``--exec_horizon`` = chunk
(50), open-loop (no temporal aggregation), gated EGL, terminal
``judge_success``. ``--skin_substeps train`` matches convert min-pool of 4
substeps at 16.67 ms. No ``--clutter_xy_scale`` (that is v1011d only).

    conda activate mlspaces
    cd /home/jaydv/code/prox_learning
    export OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export MLSPACES_ASSETS_DIR="$PWD/assets"

    python eval_act_v107spaced.py \\
      --ckpt_dir submodules/act/ckpts/pact_place_corridor_v107_spaced/pact_place_corridor_v107_spaced_PACT_READOUT_s0_bs8_cs50_lr1e-5_e2000 \\
      --num_rollouts 2 --skin_substeps snapshot \\
      --output_dir eval_output/simple_v107_spaced_smoke

    # Wrist-only ablation (OOD vs 210-ep table+wrist train). New dir.
    python eval_act_v107spaced.py \\
      --ckpt_dir submodules/act/ckpts/pact_place_corridor_v107_spaced/pact_place_corridor_v107_spaced_PACT_READOUT_s0_bs8_cs50_lr1e-5_e2000 \\
      --num_rollouts 2 --skin_substeps snapshot --cameras wrist_camera \\
      --output_dir eval_output/simple_v107_spaced_wrist
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
_V107_MOLMO = _REPO_ROOT / "submodules" / "molmospaces"
_ENV_VERSION = "pact_place_corridor_v10_7_spaced_bench"
_EXPECTED_XML = frozenset(
    {
        "pact_place_corridor_v10_7_neg5.xml",
        "pact_place_corridor_v10_7_center.xml",
        "pact_place_corridor_v10_7_pos5.xml",
    }
)
_N_CELLS = 24
_DEFAULT_CAMERAS = ("table_camera", "wrist_camera")
_LOG = "[eval_act_v107spaced]"
_SKIN_SUBSTEPS = "snapshot"
_TRAIN_PERIOD_MS = 16.6667
_TABLE_CAM = "table_camera"
_CONSTRUCTION_RETRY_STRIDE = 1_000_003
_CONSTRUCTION_RETRY_MAX = 64
# Log tags only. Retry is any ValueError from sample_task.
_CONSTRUCTION_FAIL_MARKERS = (
    "settled clutter overlaps target",
    "settled clutter objects overlap",
    "clutter drifted during settle",
    "clutter did not settle",
    "target-relative annulus is empty",
    "could not place target-relative clutter",
)


def _argv_value(*names: str, default: str | None = None) -> str | None:
    args = sys.argv[1:]
    for i, a in enumerate(args):
        for name in names:
            if a == name and i + 1 < len(args):
                return args[i + 1]
            if a.startswith(name + "="):
                return a.split("=", 1)[1]
    return default


_MOLMO_ROOT = Path(
    _argv_value("--molmo", "--molmospaces_root", default=str(_V107_MOLMO))
).resolve()
if not (_MOLMO_ROOT / "molmo_spaces").is_dir():
    raise SystemExit(
        f"{_LOG} molmospaces missing at {_MOLMO_ROOT}. "
        "Point --molmo at this checkout's submodules/molmospaces."
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
import shutil
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
        snapshot = True
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
                n_snap = int(getattr(task, "_eval_snapshot_renders", 0) or 0) + 1
                task._eval_snapshot_renders = n_snap
                self._eval_snapshot_renders = int(
                    getattr(self, "_eval_snapshot_renders", 0) or 0
                ) + 1
            else:
                n_sub = int(getattr(task, "_eval_substep_queries", 0) or 0) + 1
                task._eval_substep_queries = n_sub
                self._eval_substep_queries = int(
                    getattr(self, "_eval_substep_queries", 0) or 0
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
    eval_cfg, *, need_skin: bool, period_ms: float, record_depth: bool = False
) -> None:
    """RGB depth only when --save_first_frame (uuid {cam}_depth). period_ms=0 is snapshot."""
    eval_cfg.proximity_sensor_period_ms = float(period_ms)
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
        f"period_ms={eval_cfg.proximity_sensor_period_ms} "
        f"({'0=snapshot' if float(period_ms) <= 0 else 'train-substeps'}) "
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
        "snapshot_renders": int(getattr(task, "_eval_snapshot_renders", 0) or 0),
        "substep_queries": int(getattr(task, "_eval_substep_queries", 0) or 0),
    }
    if seed is not None:
        rec["seed"] = int(seed)
    if ever_success is not None:
        rec["ever_success"] = int(bool(ever_success))
    if extra:
        rec.update(extra)
    _EPISODE_METRICS.append(rec)
    extra = (
        f" renders={n_fresh} skip={n_skip}"
        f" snapshot={rec.get('snapshot_renders', 0)}"
        f" substep={rec.get('substep_queries', 0)}"
        if (n_fresh or n_skip or rec.get("snapshot_renders") or rec.get("substep_queries"))
        else ""
    )
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
        orig[0] if orig else "eval_act_v107spaced.py",
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
        raise SystemExit(f"{_LOG} checkpoint missing {key}")
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
        raise SystemExit(f"{_LOG} checkpoint is not a state dict: {ckpt_path}")
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
            print(
                f"{_LOG} WARNING: --cameras {list(cameras)} != "
                f"training_config.json {from_json}. OOD ablation.",
                flush=True,
            )
        return cameras
    if from_json is not None:
        return tuple(from_json)
    return _DEFAULT_CAMERAS


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
        print(f"{_LOG} loaded {self.ckpt_path}")

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
                    f"{_LOG} proximity ON history={self.history_mode} "
                    f"feature={pc.prox_feature} exec_horizon={self.exec_horizon}"
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
    camera_names: tuple[str, ...] = ("table_camera", "wrist_camera")
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


def _disable_action_noise(eval_cfg) -> None:
    if hasattr(eval_cfg.robot_config, "action_noise_config"):
        noise = eval_cfg.robot_config.action_noise_config
        if noise is not None and hasattr(noise, "enabled"):
            noise.enabled = False


def _set_cfg(eval_cfg, name: str, value) -> None:
    """Pydantic configs reject unknown fields. Skip names this class does not have."""
    fields = getattr(type(eval_cfg), "model_fields", None)
    if isinstance(fields, dict) and name not in fields:
        return
    setattr(eval_cfg, name, value)


def _enable_table_camera(eval_cfg) -> None:
    """Public V107 config is wrist-only. Train hdf5 is table_camera + wrist.

    Hybrid system has that table view as ``exo_camera_1``. Rename so policy
    obs keys match convert_meta / TASK_CONFIGS.
    """
    from molmo_spaces.configs.camera_configs import FrankaSkinHybridCameraSystem

    eval_cfg.camera_config = FrankaSkinHybridCameraSystem()
    cams = []
    renamed = 0
    for cam in list(eval_cfg.camera_config.cameras):
        if getattr(cam, "name", None) == "exo_camera_1":
            if hasattr(cam, "model_copy"):
                cam = cam.model_copy(update={"name": _TABLE_CAM})
            elif hasattr(cam, "copy"):
                cam = cam.copy(update={"name": _TABLE_CAM})
            else:
                cam.name = _TABLE_CAM
            renamed += 1
        cams.append(cam)
    eval_cfg.camera_config.cameras = cams
    if renamed != 1:
        raise SystemExit(
            f"{_LOG} expected to rename 1 exo_camera_1 -> {_TABLE_CAM}, got {renamed}"
        )


def _build_v107_cfg(args, output_dir: Path):
    try:
        from molmo_spaces.data_generation.config.pact_place_datagen_configs import (
            FrankaSkinPactPlaceV107SpacedBenchConfig,
        )
        from molmo_spaces.tasks.pact_place import PactPlaceCorridorV107SpacedBenchSampler
    except ImportError as exc:
        raise SystemExit(
            f"{_LOG} V107 spaced config/sampler missing on {_MOLMO_ROOT}: {exc}. "
            "Need submodules/molmospaces at 4c6a215 / main (not hallway 977acd6, "
            "not v1010 4bba4cb, not v1011d pin 70dedc0)."
        ) from exc

    eval_cfg = FrankaSkinPactPlaceV107SpacedBenchConfig()
    sampler_cls = eval_cfg.task_sampler_config.task_sampler_class
    if sampler_cls is not PactPlaceCorridorV107SpacedBenchSampler:
        raise SystemExit(
            f"{_LOG} sampler is {getattr(sampler_cls, '__name__', sampler_cls)}, "
            "expected PactPlaceCorridorV107SpacedBenchSampler. "
            "No four-object / v1011d fallback."
        )
    version = str(getattr(sampler_cls, "PACT_PLACE_ENVIRONMENT_VERSION", "") or "")
    if version != _ENV_VERSION:
        raise SystemExit(
            f"{_LOG} sampler PACT_PLACE_ENVIRONMENT_VERSION={version!r} "
            f"!= {_ENV_VERSION!r}"
        )
    eval_cfg.policy_config = ACTPolicyConfig()
    _set_cfg(eval_cfg, "task_horizon", args.horizon or 1050)
    _set_cfg(eval_cfg, "end_on_success", False)
    _set_cfg(eval_cfg, "terminate_upon_success", False)
    _set_cfg(eval_cfg, "output_dir", output_dir)
    _set_cfg(eval_cfg, "num_workers", 1)
    _set_cfg(eval_cfg, "save_videos", False)
    _set_cfg(eval_cfg, "use_passive_viewer", False)
    _set_cfg(eval_cfg, "viz_sensor_rgb", False)
    _enable_table_camera(eval_cfg)
    _disable_action_noise(eval_cfg)
    paths = [Path(p) for p in list(eval_cfg.task_sampler_config.scene_xml_paths or [])]
    names = {p.name for p in paths}
    if names != set(_EXPECTED_XML):
        raise SystemExit(
            f"{_LOG} scene XML set {sorted(names)} != {sorted(_EXPECTED_XML)}"
        )
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"{_LOG} missing scene XML: {missing}")
    if len(paths) != _N_CELLS:
        raise SystemExit(
            f"{_LOG} expected {_N_CELLS} scene paths (24-cell grid), got {len(paths)}"
        )
    eval_cfg.task_sampler_config.house_inds = list(range(_N_CELLS))
    eval_cfg.task_sampler_config.samples_per_house = 1
    xml = paths[0] if paths else Path("unknown.xml")
    return eval_cfg, xml, sampler_cls, paths


def _fill_policy_config(eval_cfg, args, cameras: tuple[str, ...], chunk: int) -> None:
    pc = ACTPolicyConfig()
    pc.ckpt_dir = str(Path(args.ckpt_dir).resolve())
    pc.ckpt_name = args.ckpt_name
    pc.chunk_size = chunk
    pc.exec_horizon = int(getattr(args, "exec_horizon", None) or chunk)
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
            f"{_LOG} PACT ckpt -> proximity ON "
            f"(feature={pc.prox_feature}, layout={pc.prox_layout}, "
            f"K={pc.prox_tokens_per_sensor}, pool={pc.prox_pool})"
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


def _assert_v107_task(task, house: int) -> str:
    from molmo_spaces.data_generation.pact_place.contracts import v107_spaced_cell

    params = dict(getattr(task, "scene_params", None) or {})
    version = params.get("pact_place_environment_version")
    if version != _ENV_VERSION:
        raise RuntimeError(
            f"{_LOG} scene_params pact_place_environment_version={version!r} "
            f"!= {_ENV_VERSION!r}"
        )
    if params.get("pact_v1011d_all_clutter_randomized") is True:
        raise RuntimeError(
            f"{_LOG} scene looks like v1011d randomized clutter; wrong sampler"
        )
    layout = dict(params.get("pact_clutter_layout") or {})
    bench = params.get("pact_v107_spaced_bench")
    if bench is not True and layout.get("spaced_bench") is not True:
        raise RuntimeError(
            f"{_LOG} scene_params missing v107 spaced-bench marker "
            f"(pact_v107_spaced_bench={bench!r}, layout.spaced_bench="
            f"{layout.get('spaced_bench')!r})"
        )
    family, side, pose = v107_spaced_cell(house)
    return f"{family}|{side}|{pose}"


def _house_index(i: int, args: argparse.Namespace) -> int:
    if args.house_ind is None:
        return int(i) % _N_CELLS
    return int(args.house_ind)


def _house_schedule(args: argparse.Namespace) -> str:
    if args.house_ind is None:
        return "cycle_24"
    return f"pin_{int(args.house_ind)}"


def _protocol_identity(args: argparse.Namespace, exec_horizon: int) -> dict:
    cameras = list(getattr(args, "cameras", None) or _DEFAULT_CAMERAS)
    return {
        "exec_horizon": int(exec_horizon),
        "skin_substeps": str(args.skin_substeps),
        "house_schedule": _house_schedule(args),
        "cameras": cameras,
        **pact_eval_sensor_keep.protocol_fields(args),
    }


def _refuse_resume_mismatch(summary_path: Path, protocol: dict) -> None:
    if not summary_path.is_file():
        return
    saved = json.loads(summary_path.read_text())
    old = saved.get("protocol") or {}
    for key, value in protocol.items():
        old_val = old.get(key)
        if old_val is None and key == "cameras":
            old_val = list(_DEFAULT_CAMERAS)
        if old_val is None and key == "sensor_keep_frac":
            old_val = 1.0
        if old_val is None and key == "sensor_mask_fixed":
            old_val = False
        if key == "cameras":
            mismatch = list(old_val) != list(value)
        elif key == "sensor_keep_frac":
            try:
                mismatch = abs(float(old_val) - float(value)) > 1e-12
            except (TypeError, ValueError):
                mismatch = True
        elif key == "sensor_mask_fixed":
            mismatch = bool(old_val) != bool(value)
        else:
            mismatch = old_val != value
        if mismatch:
            raise SystemExit(
                f"{_LOG} refuse resume into {summary_path.parent}: "
                f"protocol {key}={old_val!r} != {value!r}. "
                "Use a new --output_dir."
            )


def _construction_tag(exc: BaseException) -> str:
    msg = str(exc)
    if any(m in msg for m in _CONSTRUCTION_FAIL_MARKERS):
        return "known"
    return "valueerror"


def _sample_v107_task(sampler, house: int, seed: int):
    """Datagen retries settle failures. Eval must too or n=50 dies on hairline contact.

    Attempt 0 keeps ``seed``. Later attempts offset the RNG so resume is
    deterministic. Do not drop the episode. Any ``ValueError`` from
    ``sample_task`` is construction (settle, annulus, slot place, hash bind).
    """
    last: BaseException | None = None
    for attempt in range(_CONSTRUCTION_RETRY_MAX):
        draw_seed = int(seed) + attempt * _CONSTRUCTION_RETRY_STRIDE
        set_seed(draw_seed)
        sampler.seed_task_sampling(draw_seed)
        try:
            task = sampler.sample_task(house_index=house)
        except ValueError as exc:
            last = exc
            print(
                f"{_LOG} construction retry house={house} seed={seed} "
                f"attempt={attempt} draw_seed={draw_seed} "
                f"tag={_construction_tag(exc)}: {exc}",
                flush=True,
            )
            continue
        if task is None:
            last = RuntimeError("sample_task returned None")
            print(
                f"{_LOG} construction retry house={house} seed={seed} "
                f"attempt={attempt}: sample_task returned None",
                flush=True,
            )
            continue
        return task, attempt
    raise RuntimeError(
        f"{_LOG} construction failed house={house} seed={seed} "
        f"after {_CONSTRUCTION_RETRY_MAX} attempts: {last}"
    ) from last


def _mp4_codec(path: Path) -> str:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return probe.stdout.strip().lower() if probe.returncode == 0 else ""


def encode_h264_ide(mp4_path: Path) -> Path | None:
    """MPEG-4 Part 2 (OpenCV mp4v) does not play in VS Code / Cursor. H.264 does."""
    if not mp4_path.is_file():
        return None
    if _mp4_codec(mp4_path) == "h264":
        return mp4_path
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"{_LOG} ffmpeg missing — {mp4_path} stays MPEG-4 (IDE will not play it)", flush=True)
        return mp4_path
    tmp = mp4_path.with_name(mp4_path.stem + ".h264tmp.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(mp4_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not tmp.is_file():
        print(
            f"{_LOG} ffmpeg h264 failed on {mp4_path}: {(result.stderr or '')[-400:]}",
            flush=True,
        )
        if tmp.exists():
            tmp.unlink()
        return mp4_path
    tmp.replace(mp4_path)
    print(f"{_LOG} h264 {mp4_path} codec={_mp4_codec(mp4_path) or 'unknown'}", flush=True)
    return mp4_path


def _grab_exo_rgb(task) -> np.ndarray | None:
    """Current-world exo RGB. Independent of the chunk gate (skip steps reuse stale RGB)."""
    env = getattr(task, "_env", None)
    if env is None or not hasattr(env, "render_rgb_frame"):
        return None
    try:
        frame = env.render_rgb_frame(_TABLE_CAM)
    except Exception as exc:
        print(f"{_LOG} {_TABLE_CAM} render failed: {exc}", flush=True)
        return None
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        print(f"{_LOG} {_TABLE_CAM} unexpected shape {arr.shape}", flush=True)
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class _ExoMp4:
    """Stream one RGB camera to MP4. Do not keep the episode in RAM."""

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
            if (w % 2) or (h % 2):
                w -= w % 2
                h -= h % 2
                frame = frame[:h, :w]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
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
            print(f"{_LOG} no {_TABLE_CAM} frames; skipped {self.path}", flush=True)
            return None
        encoded = encode_h264_ide(self.path)
        print(
            f"{_LOG} wrote {self.path} frames={self.n_frames} fps={self.fps:.3f} "
            f"codec={_mp4_codec(self.path) or 'unknown'}",
            flush=True,
        )
        return encoded or self.path


def _rollout(
    task,
    policy,
    horizon: int,
    *,
    skin_substeps: str,
    exo_video: Path | None = None,
    video_fps: float = 1000.0 / 66.0,
    on_step=None,
    episode_idx: int = 0,
    seed: int = 0,
    save_first_frame: bool = False,
    output_dir: Path | None = None,
    cameras: tuple[str, ...] | None = None,
) -> tuple[bool, bool, dict, int]:
    """Full-horizon loop. Headline success is terminal judge_success."""
    policy.reset()
    if hasattr(policy, "begin_sensor_keep"):
        policy.begin_sensor_keep(episode_idx)
    task._eval_snapshot_renders = 0
    task._eval_substep_queries = 0
    writer = _ExoMp4(exo_video, video_fps) if exo_video is not None else None
    extra: dict = {}
    last_step = 0
    try:
        if skin_substeps == "train":
            _reset_prox_buffer(task)
        observation, _ = task.reset()
        if save_first_frame:
            if output_dir is None:
                raise SystemExit(f"{_LOG} --save_first_frame needs --output_dir")
            pact_eval_sensor_keep.dump_first_frame(
                obs=observation,
                policy=policy,
                cameras=cameras or tuple(policy.pc.camera_names),
                output_dir=Path(output_dir),
                episode_idx=episode_idx,
                seed=seed,
                prox_pool=str(getattr(policy, "_prox_pool", "min")),
            )
        if writer is not None:
            writer.write(_grab_exo_rgb(task))
        ever = False
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
            last_step = int(_step)
            if on_step is not None:
                on_step(_step, task)
            task._proximity_camera_names = []
            if writer is not None:
                writer.write(_grab_exo_rgb(task))
            if _one_bool(task.judge_success()):
                ever = True
            if getattr(task, "observation_cache", None):
                task.observation_cache[-1] = [{}]
            if _one_bool(terminal) or _one_bool(truncated):
                break
        terminal = _one_bool(task.judge_success())
        return terminal, ever or terminal, extra, last_step
    finally:
        if writer is not None:
            path = writer.close()
            extra["video_camera"] = _TABLE_CAM
            extra["video_frames"] = int(writer.n_frames)
            if path is not None:
                extra["video_path"] = str(path)


def _load_resume(jsonl: Path, seed_base: int) -> set[tuple[int, int]]:
    """Load finished episodes. Refuse records that are not from this run.

    Every record must have ``seed == seed_base + episode_idx`` and a unique
    ``episode_idx``. Otherwise a run with another ``--seed_base`` wrote into
    this directory, and its records would be averaged into this summary
    (2026-09-17 v1011d dirs: 52 records for n=50). Read-only; never edits
    the file. Startup check only; no effect on rollouts.
    """
    done: set[tuple[int, int]] = set()
    if not jsonl.is_file():
        return done
    records: list[dict] = []
    for lineno, line in enumerate(jsonl.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        rec = json.loads(line)
        idx, seed = rec.get("episode_idx"), rec.get("seed")
        if idx is None or seed is None:
            problem = "has no episode_idx/seed"
        elif int(seed) != int(seed_base) + int(idx):
            problem = (
                f"episode_idx={idx} seed={seed} is not from --seed_base {seed_base} "
                f"(expected seed {int(seed_base) + int(idx)})"
            )
        elif (int(idx), int(seed)) in done:
            problem = f"duplicates episode_idx={idx} seed={seed}"
        else:
            problem = None
        if problem:
            raise SystemExit(
                f"{_LOG} refuse resume into {jsonl.parent}: {jsonl.name} line {lineno} "
                f"{problem}. Use a new --output_dir."
            )
        done.add((int(idx), int(seed)))
        records.append(rec)
    _EPISODE_METRICS.extend(records)
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
        "--cameras",
        nargs="+",
        default=None,
        help="Policy RGB. Default table_camera wrist_camera (train-matched). "
        "wrist_camera alone is a hallway-style ablation; train was both. "
        "Env still has table_camera for --save_video. New --output_dir.",
    )
    p.add_argument("--molmo", "--molmospaces_root", dest="molmo", default=None)
    p.add_argument("--num_rollouts", type=int, default=2)
    p.add_argument(
        "--house_ind",
        type=int,
        default=None,
        help="Pin one 24-cell index. Default cycles house_index = i %% 24.",
    )
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--chunk_size", type=int, default=None)
    p.add_argument(
        "--exec_horizon",
        type=int,
        default=None,
        help="Re-query every K steps; execute chunk[:K]. Default = ckpt chunk size.",
    )
    p.add_argument(
        "--skin_substeps",
        choices=("snapshot", "train"),
        default="snapshot",
        help="snapshot: period 0, one EGL at query (hallway-like). "
        "train: 16.67 ms, keep 4 substep frames at query.",
    )
    p.add_argument("--seed_base", type=int, default=2026)
    p.add_argument("--skin", choices=("egl", "rays"), default="egl")
    p.add_argument(
        "--history",
        choices=("query", "consecutive"),
        default="query",
        help="Kept for a future readout arm. Raw ckpt is a single frame.",
    )
    p.add_argument(
        "--output_dir",
        default="/home/jaydv/code/prox_learning/eval_output/eval_act_v107spaced",
    )
    p.add_argument(
        "--save_video",
        "--save_videos",
        action="store_true",
        help="Write table_camera RGB MP4 every control step under "
        "output_dir/videos/. Remux to H.264 yuv420p +faststart so VS Code / "
        "Cursor can play it. Skin gate stays on. Does not set molmospaces "
        "save_videos (that keeps the obs cache). Not a protocol field.",
    )
    pact_eval_sensor_keep.add_cli_flags(p)
    pact_eval_wandb.add_cli_flags(p)
    return p.parse_args()


def main() -> None:
    global _SKIN_SUBSTEPS
    args = parse_args()
    _SKIN_SUBSTEPS = args.skin_substeps
    if args.molmo is not None and Path(args.molmo).resolve() != _MOLMO_ROOT:
        raise SystemExit(
            f"{_LOG} --molmo must be on the command line before import-time path setup"
        )
    if args.history == "consecutive" and args.skin == "egl":
        print(
            f"{_LOG} WARNING: --history consecutive with --skin egl renders "
            "40 EGL cameras every control step. Non-headline. Use --skin rays.",
            flush=True,
        )
    if args.skin_substeps == "snapshot" and args.skin == "egl":
        print(
            f"{_LOG} skin_substeps=snapshot (hallway-like). "
            "Train-matched squeeze is --skin_substeps train.",
            flush=True,
        )
    if args.skin_substeps == "train":
        print(
            f"{_LOG} skin_substeps=train: period={_TRAIN_PERIOD_MS} ms, "
            "keep 4 substep frames at query. Reset obs is still a snapshot.",
            flush=True,
        )

    ckpt_dir = Path(args.ckpt_dir).resolve()
    ckpt_path = ckpt_dir / args.ckpt_name
    stats_path = ckpt_dir / "dataset_stats.pkl"
    if not ckpt_path.is_file():
        raise SystemExit(f"{_LOG} missing {ckpt_path}")
    if not stats_path.is_file():
        raise SystemExit(f"{_LOG} missing {stats_path}")

    cameras = _resolve_cameras(args, ckpt_dir)
    args.cameras = list(cameras)
    if "wrist_camera" not in cameras:
        raise SystemExit(
            f"{_LOG} cameras {list(cameras)} must include wrist_camera"
        )
    if list(cameras) != list(_DEFAULT_CAMERAS):
        print(
            f"{_LOG} WARNING: policy cameras {list(cameras)} != train "
            f"{list(_DEFAULT_CAMERAS)} (210-ep hdf5 is table+wrist). "
            "Hallway-style ablation. New --output_dir. Do not mix rates.",
            flush=True,
        )
    weights = _load_state_dict(ckpt_path)
    chunk = _chunk_from_weights(weights)
    if args.chunk_size is not None and int(args.chunk_size) != chunk:
        raise SystemExit(
            f"{_LOG} --chunk_size {args.chunk_size} != ckpt query_embed {chunk}"
        )
    exec_horizon = int(args.exec_horizon) if args.exec_horizon is not None else chunk
    if exec_horizon <= 0:
        raise SystemExit(f"{_LOG} --exec_horizon must be positive")
    args.exec_horizon = exec_horizon
    n_bb = _n_backbones(weights)
    if n_bb > 1 and n_bb != len(cameras):
        raise SystemExit(
            f"{_LOG} checkpoint has {n_bb} backbones, --cameras has {len(cameras)}"
        )
    if n_bb == 1 and len(cameras) != 1:
        print(
            f"{_LOG} DETRVAE shares 1 backbone across {len(cameras)} cameras "
            f"{list(cameras)}",
            flush=True,
        )
    del weights

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg, xml, sampler_cls, xml_paths = _build_v107_cfg(args, output_dir)
    _fill_policy_config(eval_cfg, args, cameras, chunk)
    period_ms = _TRAIN_PERIOD_MS if args.skin_substeps == "train" else 0.0
    _configure_eval_cameras(
        eval_cfg,
        need_skin=bool(eval_cfg.policy_config.use_proximity),
        period_ms=period_ms,
        record_depth=bool(args.save_first_frame),
    )
    _install_chunk_gated_sensors()
    if args.skin == "rays":
        if eval_cfg.policy_config.use_proximity:
            _install_raycast_proximity()
    elif eval_cfg.policy_config.use_proximity:
        print(
            f"{_LOG} --skin egl: 40-cam rasterizer on query steps.",
            flush=True,
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(args.seed_base)

    global _METRICS_JSONL
    _EPISODE_METRICS.clear()
    _METRICS_JSONL = output_dir / "episodes.jsonl"
    summary_path = output_dir / "eval_summary.json"
    protocol = _protocol_identity(args, exec_horizon)
    _refuse_resume_mismatch(summary_path, protocol)
    done = _load_resume(_METRICS_JSONL, args.seed_base)

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
    dt_ms = float(getattr(eval_cfg, "policy_dt_ms", 66.0) or 66.0)
    video_fps = 1000.0 / dt_ms if dt_ms > 0 else 15.0
    video_dir = output_dir / "videos"
    wb = pact_eval_wandb.start(
        args,
        output_dir=output_dir,
        horizon=horizon,
        extra_config={
            "task": "v107_spaced",
            "ckpt_dir": str(ckpt_dir),
            "ckpt_name": args.ckpt_name,
            "camera_names": list(cameras),
            "skin": args.skin,
            "history": args.history,
            "skin_substeps": args.skin_substeps,
            "exec_horizon": exec_horizon,
            "chunk_size": chunk,
            "house_schedule": protocol["house_schedule"],
            "sensor_keep_frac": protocol["sensor_keep_frac"],
            "sensor_mask_fixed": protocol["sensor_mask_fixed"],
            "sensor_mask_seed": pact_eval_sensor_keep.resolved_mask_seed(args),
            "save_first_frame": bool(args.save_first_frame),
        },
    )
    wb.log_running(_EPISODE_METRICS, _summarize_place_metrics())
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"{_LOG} --save_video: {_TABLE_CAM} every control step -> {video_dir} "
            f"fps={video_fps:.3f} H.264 yuv420p +faststart. "
            "Skin gate unchanged. molmospaces save_videos=False.",
            flush=True,
        )

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
            "task": "v107_spaced",
            "config_class": "FrankaSkinPactPlaceV107SpacedBenchConfig",
            "task_sampler_class": sampler_cls.__name__,
            "environment_version": _ENV_VERSION,
            "scene_xmls": [p.name for p in xml_paths],
            "scene_xml": str(xml),
            "camera_names": list(cameras),
            "house_ind": args.house_ind,
            "house_schedule": protocol["house_schedule"],
            "num_rollouts": args.num_rollouts,
            "completed": n_done,
            "task_horizon": horizon,
            "chunk_size": chunk,
            "exec_horizon": exec_horizon,
            "protocol": {
                **protocol,
                "history": args.history,
                "history_mode": history_label,
                "skin": args.skin,
                "chunk": chunk,
                "success": "terminal",
                "end_on_success": False,
                "temporal_aggregation": False,
            },
            "save_video": bool(args.save_video),
            "save_first_frame": bool(args.save_first_frame),
            "video_camera": _TABLE_CAM if args.save_video else None,
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
        f"{_LOG} molmospaces={_MOLMO_ROOT} commit={_git_rev(_MOLMO_ROOT)} "
        f"config=FrankaSkinPactPlaceV107SpacedBenchConfig "
        f"sampler={sampler_cls.__name__} version={_ENV_VERSION} "
        f"xml={sorted(_EXPECTED_XML)} cameras={cameras} "
        f"horizon={horizon} n={args.num_rollouts} skin={args.skin} "
        f"skin_substeps={args.skin_substeps} exec_horizon={exec_horizon} "
        f"house_schedule={protocol['house_schedule']} history={history_label} "
        f"save_video={int(bool(args.save_video))}",
        flush=True,
    )

    try:
        for i in range(args.num_rollouts):
            seed = int(args.seed_base) + i
            house = _house_index(i, args)
            if (i, seed) in done:
                print(f"{_LOG} skip existing ep={i} seed={seed}", flush=True)
                _write_summary(summary_path, summary_payload(len(_EPISODE_METRICS)))
                continue
            set_seed(seed)
            task, n_retry = _sample_v107_task(sampler, house, seed)
            cell = _assert_v107_task(task, house)
            _disarm_prox_cameras(task)
            if args.skin_substeps == "train":
                _reset_prox_buffer(task)
            audit = PactPlaceContactAudit()
            task._contact_audit_hook = audit
            task.register_policy(policy)
            t0 = time.monotonic()
            exo_path = None
            if args.save_video:
                exo_path = (
                    video_dir
                    / f"ep{i:03d}_house{house:02d}_seed{seed}_{_TABLE_CAM}.mp4"
                )
            terminal, ever, video_extra, last_step = _rollout(
                task,
                policy,
                horizon,
                skin_substeps=args.skin_substeps,
                exo_video=exo_path,
                video_fps=video_fps,
                on_step=lambda step, rolled, i=i: wb.maybe_log_step(i, step, rolled),
                episode_idx=i,
                seed=seed,
                save_first_frame=bool(args.save_first_frame),
                output_dir=output_dir,
                cameras=cameras,
            )
            print(
                f"{_LOG} ep={i} seed={seed} house={house} cell={cell} "
                f"wall={time.monotonic() - t0:.1f}s "
                f"terminal={int(terminal)} ever={int(ever)} "
                f"snapshot={getattr(task, '_eval_snapshot_renders', 0)} "
                f"substep={getattr(task, '_eval_substep_queries', 0)} "
                f"settle_retry={n_retry}",
                flush=True,
            )
            _record_place_metric(
                task,
                bool(terminal),
                audit.summary(),
                episode_idx=i,
                seed=seed,
                ever_success=ever,
                extra={
                    "house_index": house,
                    "cell": cell,
                    "construction_retries": int(n_retry),
                    **video_extra,
                    **policy.sensor_keep_log(),
                },
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
            f"{_LOG} terminal {collision['success']}/{collision['episodes']} "
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
    print(f"{_LOG} wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
