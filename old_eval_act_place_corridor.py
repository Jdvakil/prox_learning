"""In-env ACT / PACT eval on the coauthor pick-and-place corridor.

Training data: HF `Lundii/pact_place_corridor_v5` (wrist RGB + 40-sensor skin).
The recovered rows record molmospaces `1cbb180` but the scene XML version is
`pact_place_corridor_v2`, which first appears at `977acd6` together with
`FrankaSkinHybridWristOnlyCameraSystem`. Those pieces live on
`origin/experiment/pact-vs-act-remediation-v2`, not on this checkout's
molmospaces `main`. Point PYTHONPATH / --molmospaces_root at a worktree of
`977acd6`.

    git -C submodules/molmospaces worktree add \\
        /home/jaydv/code/molmospaces-pact-place \\
        977acd6719a8c05b688d3e70da356d61dd32d259

    cd submodules/act
    PYTHONPATH="/home/jaydv/code/molmospaces-pact-place:$PWD:$PYTHONPATH" \\
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
    python eval_act_place_corridor.py \\
        --ckpt_dir ckpts/pact_place_corridor_v5/<run> \\
        --output_dir /home/jaydv/code/prox_learning/eval_output/place_corridor_<arm> \\
        --num_rollouts 20 --chunk_size 50 --temp_agg_off --task_horizon 800

Default is metrics-only: no MP4/HDF5, policy loaded once, vanilla skips the
40-sensor 60 Hz depth stack. With ``--temp_agg_off``, RGB/skin EGL also skip
on idle chunk steps. Start with n=20, then n=50. Kill-safe progress is
`episodes.jsonl` next to `eval_summary.json`.

Pass ``--manifest configs/pact_place_eval_chunk100_manifest.json`` to run
Amine's frozen 40-row place protocol on a *local* ckpt (chunk 50, prox_config).
Do not use ``amine/act/eval_pact_place_chunk100_row.py`` for these ckpts —
that worker wants Amine's chunk-100 / 32-d hashed PACT.

Never `imitate_episodes.py --eval` on a PACT ckpt.
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ACT_DIR = Path(__file__).resolve().parent
_DEFAULT_WORKTREE = Path("/home/jaydv/code/molmospaces-pact-place")


def _molmospaces_root_from_argv() -> Path:
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--molmospaces_root" and i + 1 < len(args):
            return Path(args[i + 1]).resolve()
        if a.startswith("--molmospaces_root="):
            return Path(a.split("=", 1)[1]).resolve()
    env = os.environ.get("MOLMOSPACES_PACT_PLACE")
    return Path(env).resolve() if env else _DEFAULT_WORKTREE


_MOLMO_ROOT = _molmospaces_root_from_argv()
if not (_MOLMO_ROOT / "molmo_spaces").is_dir():
    raise SystemExit(
        f"[act-eval-place] molmospaces worktree missing at {_MOLMO_ROOT}.\n"
        "  git -C /home/jaydv/code/prox_learning/submodules/molmospaces worktree add \\\n"
        f"      {_MOLMO_ROOT} 977acd6719a8c05b688d3e70da356d61dd32d259"
    )
sys.path.insert(0, str(_MOLMO_ROOT))
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import argparse
import json
import time
from pathlib import Path as _Path  # noqa: E402

from eval_act_obstacle import (  # noqa: E402
    ACTInferencePolicy,
    ACTPolicyConfig,
    _EPISODE_METRICS,
)
from utils import set_seed  # noqa: E402

from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig  # noqa: E402
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (  # noqa: E402
    FrankaSkinPACTCollisionCorridorConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner  # noqa: E402
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR  # noqa: E402
from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV2Sampler  # noqa: E402
from molmo_spaces.tasks.pact_place_contact_audit import PactPlaceContactAudit  # noqa: E402
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask  # noqa: E402

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None

_METRICS_JSONL: _Path | None = None
_PENDING_ROW_META: dict | None = None

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

    ``SensorSuite.get_observations`` is the 0.75 s/step PACT cost. Wrist + 40
    proximity cameras still render on every query (chunk boundary). Physics and
    the 2 ms contact audit are unchanged.

    Also patches ``ProximityDepthBufferSensor.get_observation`` by class name so
    a second ``molmo_spaces`` on ``sys.path`` cannot bypass ``isinstance``.
    Fresh queries batch all 40 skins into one ``record_proximity_depths`` call.
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
    import numpy as np

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

    Same 45° 8×8 grid, geom group 2 hidden. Not bit-identical to the EGL
    rasterizer (pixel is a fat cone; a ray is the pixel center). Use for
    iteration. Paper table stays on EGL.
    """
    import numpy as np
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
            full = self._proximity_cam_full_name(name) if hasattr(
                self, "_proximity_cam_full_name"
            ) else None
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


def _install_metrics_only_hooks() -> None:
    """Do not keep 800-step RGB/depth histories or write MP4/HDF5.

    The datagen pipeline otherwise retains every episode until the house
    finishes, which OOM-kills a 50-ep eval on a 62 GB box.
    """
    import molmo_spaces.data_generation.pipeline as pipeline
    from molmo_spaces.tasks.task import BaseMujocoTask

    def _skip_save(worker_logger, house_raw_histories, *args, **kwargs):
        n = len(house_raw_histories or [])
        worker_logger.info(
            f"[act-eval-place] skip trajectory/video save ({n} episodes, metrics-only)"
        )
        if house_raw_histories is not None:
            for ep in house_raw_histories:
                if isinstance(ep, dict):
                    ep.pop("sensor_suite", None)
                    ep.pop("history", None)
            house_raw_histories.clear()
        import gc

        gc.collect()

    def _tiny_history(self):
        # Pipeline keeps house_raw_histories until the house ends and stores
        # task.sensor_suite on every episode. Drop the suite here so 50 eval
        # episodes cannot pin 50 copies of 41 cameras.
        if getattr(self, "_sensor_suite", None) is not None:
            self._sensor_suite = None
        return {
            "observations": [],
            "rewards": [],
            "terminals": [],
            "truncateds": [],
            "successes": [],
            "actions": [],
            "obs_scene": {"collision_metrics": self._summarize_collisions()},
        }

    _orig_cache = BaseMujocoTask.get_and_cache_all_step_information

    def _cache_drop_frames(self):
        out = _orig_cache(self)
        if self.observation_cache:
            n_batch = len(self.observation_cache[-1])
            self.observation_cache[-1] = [{} for _ in range(n_batch)]
        return out

    pipeline.save_house_trajectories = _skip_save
    BaseMujocoTask.get_history = _tiny_history
    BaseMujocoTask.get_and_cache_all_step_information = _cache_drop_frames


def _configure_eval_cameras(eval_cfg, *, need_skin: bool) -> None:
    """Vanilla: wrist RGB only. PACT: skin at policy rate, not 60 Hz substeps."""
    eval_cfg.proximity_sensor_period_ms = 0.0
    cams = []
    for cam in list(eval_cfg.camera_config.cameras):
        if (not need_skin) and getattr(cam, "is_proximity_sensor", False):
            continue
        if getattr(cam, "name", None) == "wrist_camera":
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
        f"period_ms={eval_cfg.proximity_sensor_period_ms} (0=policy-rate)"
    )


class ACTPlaceCorridorEvalConfig(FrankaSkinPACTCollisionCorridorConfig):
    """Wrist-only hybrid-skin pick-and-place corridor; ACT policy swapped in at runtime."""

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
    output_dir: _Path = ASSETS_DIR / "datagen" / "act_place_corridor_eval"

    @property
    def tag(self) -> str:
        return "act_place_corridor_eval"


class _PlaceEvalRunner(ParallelRolloutRunner):
    """Attach the place contact audit so ACT rollouts still score hazard-bar hits."""

    @staticmethod
    def run_single_rollout(episode_seed, task, policy, **kwargs):
        audit = PactPlaceContactAudit()
        task._contact_audit_hook = audit
        success = ParallelRolloutRunner.run_single_rollout(
            episode_seed=episode_seed, task=task, policy=policy, **kwargs
        )
        try:
            _record_place_metric(task, bool(success), audit.summary())
        except Exception as e:
            print(f"[act-eval-place] contact metric capture failed: {e}")
        return success


def _record_place_metric(task, success: bool, audit: dict) -> None:
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
        "episode_idx": len(_EPISODE_METRICS),
        "success": int(success),
        "hit_bar": int(bar_frames > 0),
        "bar_contact_frames": bar_frames,
        "other_environment_frames": other_frames,
        "clutter_frames": clutter_frames,
        "collision_free": int(bool(audit.get("collision_free"))),
        "collision_free_task_success": int(bool(success) and bool(audit.get("collision_free"))),
        "gripper_close_commanded": int(bool(getattr(policy, "gripper_close_commanded", False))),
        "intrusion_side": str((getattr(task, "scene_params", {}) or {}).get("pact_intrusion_side") or ""),
        "contact_class_totals": totals,
        "first_contact_step": audit.get("first_contact_step") or {},
        "sensor_fresh_renders": n_fresh,
        "sensor_skipped_renders": n_skip,
    }
    if _PENDING_ROW_META:
        rec.update(_PENDING_ROW_META)
    _EPISODE_METRICS.append(rec)
    extra = f" renders={n_fresh} skip={n_skip}" if (n_fresh or n_skip) else ""
    print(
        f"[act-eval-place] ep{rec['episode_idx']:03d} success={success} "
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
    return {
        "episodes": n,
        "success": successes,
        "success_rate": successes / n,
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


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _selected_manifest_rows(manifest: dict, args: argparse.Namespace) -> list[dict]:
    rows = list(manifest["rows"])
    by_id = {str(row["episode_id"]): row for row in rows}
    if args.episode_id:
        if args.episode_id not in by_id:
            raise SystemExit(f"[act-eval-place] episode-id not in manifest: {args.episode_id}")
        return [by_id[args.episode_id]]
    if args.role_index is not None:
        wanted = [int(args.role_index)]
    elif args.role_indices:
        wanted = [int(part.strip()) for part in args.role_indices.split(",") if part.strip()]
    else:
        return rows
    by_role = {int(row["role_index"]): row for row in rows}
    missing = [idx for idx in wanted if idx not in by_role]
    if missing:
        raise SystemExit(f"[act-eval-place] role-index not in manifest: {missing}")
    return [by_role[idx] for idx in wanted]


def _row_output_dir(output_dir: Path, row: dict) -> Path:
    epid = str(row["episode_id"])[:16]
    return output_dir / f"{int(row['role_index']):03d}_{epid}"


def _sample_frozen_row(eval_cfg, row: dict):
    from pact_place_eval_chunk100_contract import retry_seed
    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask

    last_error: str | None = None
    max_retries = int(row.get("max_sampling_retries", 4))
    for retry_index in range(max_retries + 1):
        if retry_index == 0:
            seed = {
                "seed_u32": int(row["task_seed_u32"]),
                "seed_u64": int(row["task_seed_u64"]),
            }
        else:
            seed = retry_seed(row, retry_index)
        sampler = PactPlaceCorridorV2Sampler(eval_cfg)
        sampler.seed_task_sampling(seed["seed_u32"])
        sampler.set_pact_manifest_row(row)
        try:
            task = sampler.sample_task(
                house_index=int(row["scene_template_house_index"])
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
        return task, sampler, seed
    raise RuntimeError(
        f"frozen row {row.get('role_index')} sampling failed after "
        f"{max_retries + 1} tries: {last_error}"
    )


def _run_frozen_manifest_rows(eval_cfg, policy, args, pc) -> tuple[int, int]:
    from pact_place_eval_chunk100_contract import load_manifest

    global _PENDING_ROW_META

    manifest = load_manifest(args.manifest)
    rows = _selected_manifest_rows(manifest, args)
    arm_name = args.arm_name or ("PACT" if pc.use_proximity else "ACT")
    print(
        f"[act-eval-place] frozen 40-row protocol "
        f"manifest={Path(args.manifest).name} arm={arm_name} "
        f"n={len(rows)} horizon={eval_cfg.task_horizon} "
        f"chunk={pc.chunk_size} temp_agg_off={pc.temp_agg_off} "
        f"fast_prox_rays={bool(pc.use_proximity) and not bool(getattr(args, 'egl_prox', False))}",
        flush=True,
    )
    successes = 0
    for row in rows:
        row_dir = _row_output_dir(eval_cfg.output_dir, row)
        result_path = row_dir / "result.json"
        if args.skip_existing and result_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get("status") == "complete":
                rec = result.get("episode_metric")
                if isinstance(rec, dict):
                    rec = dict(rec)
                    rec["episode_idx"] = len(_EPISODE_METRICS)
                    _EPISODE_METRICS.append(rec)
                    if _METRICS_JSONL is not None:
                        with _METRICS_JSONL.open("a") as handle:
                            handle.write(json.dumps(rec) + "\n")
                if result.get("task_success"):
                    successes += 1
                print(
                    f"[act-eval-place] skip existing "
                    f"row={row['role_index']:03d} {result_path}"
                )
                continue
        sampler = None
        _PENDING_ROW_META = {
            "episode_id": row["episode_id"],
            "role_index": int(row["role_index"]),
            "row_sha256": row.get("row_sha256"),
            "arm": arm_name,
        }
        n_before = len(_EPISODE_METRICS)
        t_row = time.monotonic()
        print(
            f"[act-eval-place] start row={int(row['role_index']):03d} {arm_name}",
            flush=True,
        )
        try:
            task, sampler, seed = _sample_frozen_row(eval_cfg, row)
            task.register_policy(policy)
            policy.reset()
            ok = bool(
                _PlaceEvalRunner.run_single_rollout(
                    episode_seed=int(seed["seed_u64"]),
                    task=task,
                    policy=policy,
                    end_on_success=False,
                )
            )
        finally:
            _PENDING_ROW_META = None
            if sampler is not None:
                sampler.close()
        if len(_EPISODE_METRICS) == n_before:
            raise RuntimeError(
                f"frozen row {row['role_index']} produced no contact metric"
            )
        rec = _EPISODE_METRICS[-1]
        result = {
            "schema_version": "pact_place_eval_chunk100_jay_row_v1",
            "status": "complete",
            "arm": arm_name,
            "episode_id": row["episode_id"],
            "role_index": int(row["role_index"]),
            "row_sha256": row.get("row_sha256"),
            "task_success": ok,
            "collision_free_task_success": bool(
                rec.get("collision_free_task_success")
                or (rec.get("success") and rec.get("collision_free"))
            ),
            "gripper_close_commanded": bool(rec.get("gripper_close_commanded")),
            "seed": seed,
            "episode_metric": rec,
            "policy_info": {
                "num_queries": int(pc.chunk_size),
                "temp_agg_off": bool(pc.temp_agg_off),
                "use_proximity": bool(pc.use_proximity),
                "prox_feature": getattr(pc, "prox_feature", None),
                "fast_prox_rays": bool(pc.use_proximity)
                and not bool(getattr(args, "egl_prox", False)),
                "egl_prox": bool(getattr(args, "egl_prox", False)),
                "contact_audit_class": "PactPlaceContactAudit",
                "ckpt_dir": pc.ckpt_dir,
            },
        }
        _write_json(result_path, result)
        if ok:
            successes += 1
        print(
            f"[act-eval-place] wrote {result_path}  "
            f"{time.monotonic() - t_row:.1f}s",
            flush=True,
        )
    return successes, len(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", "--checkpoint-dir", dest="ckpt_dir", required=True)
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument(
        "--output_dir",
        default="/home/jaydv/code/prox_learning/eval_output/pact_place_corridor_v5",
    )
    p.add_argument("--num_rollouts", type=int, default=50)
    p.add_argument("--house_ind", type=int, default=1)
    p.add_argument("--task_horizon", type=int, default=800)
    p.add_argument("--chunk_size", type=int, default=50)
    p.add_argument("--kl_weight", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--temp_agg_off", action="store_true")
    p.add_argument("--temp_agg_m", type=float, default=0.01)
    p.add_argument("--eval_blur_sigma", type=float, default=0.0)
    p.add_argument("--use_proximity", action="store_true")
    p.add_argument("--prox_encoder_ckpt", type=str, default="")
    p.add_argument("--prox_feature", type=str, default="raw")
    p.add_argument("--prox_tokens_per_sensor", type=int, default=8)
    p.add_argument("--prox_layout", type=str, default="per_sensor")
    p.add_argument(
        "--fast_prox_rays",
        action="store_true",
        default=True,
        help="Default. Skin via mj_multiRay (not 40 EGL update_scene).",
    )
    p.add_argument(
        "--egl_prox",
        action="store_true",
        help=(
            "40-cam EGL rasterizer. Measured ~18 min/ep even with chunk-gate "
            "(19×40 update_scene). Default is mj_multiRay."
        ),
    )
    p.add_argument(
        "--save_trajectories",
        action="store_true",
        help="Keep datagen MP4/HDF5 (slow, OOM-prone). Default is metrics-only.",
    )
    p.add_argument("--live", "--render", "--viewer", dest="live", action="store_true")
    p.add_argument(
        "--molmospaces_root",
        type=str,
        default=str(_DEFAULT_WORKTREE),
        help="Worktree of molmospaces 977acd6 (pact_place_corridor_v2 env).",
    )
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="act-place-corridor-eval")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument(
        "--manifest",
        default="",
        help="Amine chunk100 40-row JSON. Pins intrusion side / jitters / task seeds.",
    )
    p.add_argument("--episode-id", dest="episode_id", default="")
    p.add_argument("--role-index", dest="role_index", type=int, default=None)
    p.add_argument(
        "--role-indices",
        dest="role_indices",
        default="",
        help="Comma-separated role_index list. Empty = every manifest row.",
    )
    p.add_argument("--arm-name", dest="arm_name", default="")
    p.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Rerun frozen rows even when result.json already exists.",
    )
    p.set_defaults(skip_existing=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _EPISODE_METRICS.clear()
    if not args.temp_agg_off:
        print(
            "[act-eval-place] WARNING: --temp_agg_off not set. Open-loop chunking "
            "is the valid PACT path; temp-agg-on almost ignores live skin."
        )

    eval_cfg = ACTPlaceCorridorEvalConfig()
    eval_cfg.task_horizon = args.task_horizon
    eval_cfg.end_on_success = False
    eval_cfg.output_dir = Path(args.output_dir).resolve()
    eval_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(eval_cfg.robot_config, "action_noise_config"):
        noise = eval_cfg.robot_config.action_noise_config
        if noise is not None and hasattr(noise, "enabled"):
            noise.enabled = False

    import molmo_spaces as _ms

    scenes = Path(_ms.__file__).resolve().parent / "data_generation" / "custom_scenes"
    xml = scenes / "pact_place_corridor_v2.xml"
    if not xml.is_file():
        raise SystemExit(f"[act-eval-place] missing scene XML {xml} (wrong molmospaces root?)")
    eval_cfg.task_sampler_config.task_sampler_class = PactPlaceCorridorV2Sampler
    eval_cfg.task_sampler_config.scene_xml_paths = [str(xml)] * 2
    eval_cfg.task_sampler_config.samples_per_house = args.num_rollouts
    eval_cfg.task_sampler_config.house_inds = [args.house_ind]

    live = args.live
    if live and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("[act-eval-place] --live requested but no display; headless")
        live = False
    eval_cfg.use_passive_viewer = live
    if live:
        eval_cfg.num_workers = 1

    set_seed(int(eval_cfg.seed) if eval_cfg.seed is not None else 2026)

    pc = eval_cfg.policy_config
    pc.ckpt_dir = str(Path(args.ckpt_dir).resolve())
    pc.ckpt_name = args.ckpt_name
    pc.chunk_size = args.chunk_size
    pc.kl_weight = args.kl_weight
    pc.hidden_dim = args.hidden_dim
    pc.dim_feedforward = args.dim_feedforward
    pc.image_h = args.image_h
    pc.image_w = args.image_w
    pc.temp_agg_off = args.temp_agg_off
    pc.temp_agg_m = args.temp_agg_m
    pc.eval_blur_sigma = float(args.eval_blur_sigma)
    pc.camera_names = ("wrist_camera",)

    prox_cfg_path = Path(pc.ckpt_dir) / "prox_config.json"
    if prox_cfg_path.exists():
        pcfg = json.loads(prox_cfg_path.read_text())
        pc.use_proximity = True
        pc.prox_feature = pcfg.get("prox_feature", "raw")
        pc.prox_layout = pcfg.get("prox_layout", "per_sensor")
        pc.prox_pool = pcfg.get("prox_pool", "min")
        pc.prox_tokens_per_sensor = int(pcfg.get("prox_tokens_per_sensor", 8))
        pc.prox_encoder_ckpt = args.prox_encoder_ckpt or pcfg.get("prox_encoder_ckpt", "")
        print(
            f"[act-eval-place] PACT ckpt -> proximity ON "
            f"(feature={pc.prox_feature}, layout={pc.prox_layout}, "
            f"K={pc.prox_tokens_per_sensor}, pool={pc.prox_pool})"
        )
    elif args.use_proximity:
        pc.use_proximity = True
        pc.prox_feature = args.prox_feature
        pc.prox_layout = args.prox_layout
        pc.prox_tokens_per_sensor = args.prox_tokens_per_sensor
        pc.prox_encoder_ckpt = args.prox_encoder_ckpt
        print("[act-eval-place] proximity FORCED ON via CLI")

    global _METRICS_JSONL
    _METRICS_JSONL = eval_cfg.output_dir / "episodes.jsonl"
    _configure_eval_cameras(eval_cfg, need_skin=bool(pc.use_proximity))
    if args.temp_agg_off:
        _install_chunk_gated_sensors()
    use_rays = bool(pc.use_proximity) and not bool(args.egl_prox)
    if use_rays:
        _install_raycast_proximity()
    elif pc.use_proximity:
        print(
            "[act-eval-place] WARNING: --egl_prox ON. "
            "PACT smoke was 2121s / 2 eps (~18 min/ep) with renders=19 skip=883. "
            "40× update_scene is the tax. Default is mj_multiRay.",
            flush=True,
        )
    if not args.save_trajectories:
        _install_metrics_only_hooks()
        print("[act-eval-place] metrics-only eval (no MP4/HDF5, policy loaded once)")

    eval_cfg.save_config()
    protocol = "pact_place_eval_chunk100" if args.manifest else "random_house"
    print(
        f"[act-eval-place] molmospaces={_MOLMO_ROOT} "
        f"sampler={eval_cfg.task_sampler_config.task_sampler_class.__name__} "
        f"xml={xml.name} cameras={pc.camera_names} "
        f"horizon={eval_cfg.task_horizon} n={args.num_rollouts} "
        f"protocol={protocol}"
    )
    print(f"[act-eval-place] writing {eval_cfg.output_dir}")

    os.environ.pop("WANDB_RUN_NAME", None)
    os.environ.pop("WANDB_PROJECT_NAME", None)

    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"eval_place_corridor_{int(time.time())}",
            config={"ckpt_dir": pc.ckpt_dir, "num_rollouts": args.num_rollouts},
        )

    policy = ACTInferencePolicy(eval_cfg)
    policy.prepare_model()
    if args.manifest:
        success, total = _run_frozen_manifest_rows(eval_cfg, policy, args, pc)
    else:
        runner = _PlaceEvalRunner(eval_cfg)
        success, total = runner.run(preloaded_policy=policy)
    rate = (success / total) if total else 0.0
    print(f"[act-eval-place] success {success}/{total}  ({rate*100:.1f}%)")
    collision = _summarize_place_metrics()
    if collision is not None:
        print(
            f"[act-eval-place] bar_hit {collision['bar_hits']}/{collision['episodes']} "
            f"({collision['bar_hit_rate']*100:.1f}%)  "
            f"collision_free {collision['collision_free']}/{collision['episodes']} "
            f"({collision['collision_free_rate']*100:.1f}%)  "
            f"strict {collision.get('collision_free_task_success', 0)}/"
            f"{collision['episodes']}  "
            f"grip_close {collision.get('gripper_close_commanded', 0)}/"
            f"{collision['episodes']}"
        )

    summary = {
        "ckpt_dir": pc.ckpt_dir,
        "ckpt_name": pc.ckpt_name,
        "task": "pact_place_corridor_v5",
        "molmospaces_root": str(_MOLMO_ROOT),
        "task_sampler_class": eval_cfg.task_sampler_config.task_sampler_class.__name__,
        "scene_xml": str(xml),
        "camera_names": list(pc.camera_names),
        "num_rollouts": int(total) if args.manifest else args.num_rollouts,
        "task_horizon": args.task_horizon,
        "chunk_size": pc.chunk_size,
        "temp_agg_off": pc.temp_agg_off,
        "use_proximity": pc.use_proximity,
        "protocol": protocol,
        "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
        "arm_name": args.arm_name or None,
        "success": int(success),
        "total": int(total),
        "success_rate": float(rate),
        "collision": collision,
    }
    out = eval_cfg.output_dir / "eval_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[act-eval-place] wrote {out}")


if __name__ == "__main__":
    main()