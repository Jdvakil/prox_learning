"""Frozen ACT / PACT eval for the V10.x place worlds without their own script.

``--env`` picks the world: v1010, v1011c, v6, v107_spaced, v12 (and v1011d, only for
``--target_support train`` reruns). Protocol is ``eval_act_v1011d.py``: this file imports
that module and runs its ``main()``. Only the world is swapped (datagen config, sampler,
environment-version check, cell count, default policy cameras, --save_video camera). Rollout
loop, FrozenACTPolicy, chunk gate, lazy skin cameras, sensor keep, resume guard and W&B are
that module's functions, not copies (2026-09-22 check: same ckpt / seeds through both
scripts gave identical outcome fields; contact-frame counts drift as much as the original
script rerun against itself). ``--clutter_xy_scale`` works for v1011d only.

Hallway (v5, v5_ext) stays on ``eval_act.py``. Published T-1011d and every uniform-cup v1011d
run stay on ``eval_act_v1011d.py``. T-107 JSONs came from ``eval_act_v107spaced.py``.

``table_camera`` checkpoints (v1010, batman v107_spaced): the batman table_camera is the
hybrid ``exo_camera_1`` pose rendered at fovy 58, while molmospaces renders exo_camera_1 at
45 (its look-at camera path drops ``fov``; the hub exo datasets were recorded at 45). The exo
slot is replaced by a 58 deg quaternion-mode ``table_camera``; ``--table_camera_fov 45`` is
the old rename (T-107).

v12: the standing kitchen is installed after ``sample_task``, as the datagen expert does in
``reset`` (after planning, before the first observation). A layout that puts an extra in the
motion lane is a construction failure and is redrawn like any other (``seed + k * 1000003``).

Cup side (v1011c, v1011d): the V10.11 sampler draws the cup y uniform over the aperture for
both panel sides, but every demo has the cup on the side away from the panel. With the
default ``--target_support uniform`` (= eval_act_v1011d.py) about half the episodes start
with the cup under the panel, a state no demo covers (T-1011d: 21/50, 0 placed in every arm).
``--target_support train`` redraws those scenes; in-support draws are untouched, so they
match uniform at the same seed. Every world logs ``target_start_xy`` per episode; v1011c /
v1011d also log ``target_in_train_support`` and the summary splits success by it.

    conda activate mlspaces
    cd /home/jaydv/code/prox_learning
    export OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export MLSPACES_ASSETS_DIR="$PWD/assets"

    python eval_act_place.py --env v1010 \\
      --ckpt_dir submodules/act/ckpts/pact_place_corridor_v1010/pact_place_corridor_v1010_PACT_READOUT_s0_bs8_cs50_lr1e-5_e2000 \\
      --num_rollouts 2 --skin_substeps snapshot --history consecutive \\
      --output_dir eval_output/_smoke_v1010
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

_V10_7_XMLS = frozenset(
    {
        "pact_place_corridor_v10_7_neg5.xml",
        "pact_place_corridor_v10_7_center.xml",
        "pact_place_corridor_v10_7_pos5.xml",
    }
)
_EXO = "exo_camera_1"
_TABLE = "table_camera"


@dataclass(frozen=True)
class World:
    config: str
    sampler: str
    env_version: str
    cameras: tuple[str, ...]
    # TASK_CONFIGS keys whose checkpoints belong in this world.
    tasks: tuple[str, ...]
    n_cells: int = 24
    xmls: frozenset[str] = _V10_7_XMLS
    cell_fn: str = "v1010_cell"
    # The sampler draws the cup y once, uniform over the aperture, whatever the panel side
    # (V10.11 _prepare_pact_clutter_layout). --target_support train applies to these worlds.
    target_rule: bool = False
    # Keep eval_act_v1011d's --clutter_xy_scale (v1011d only).
    clutter_scale: bool = False


WORLDS = {
    "v1010": World(
        config="FrankaSkinPactPlaceV1010FourObjectConfig",
        sampler="PactPlaceCorridorV1010FourObjectSampler",
        env_version="pact_place_corridor_v10_10_four_object",
        cameras=(_TABLE, "wrist_camera"),
        tasks=("pact_place_corridor_v1010",),
    ),
    "v1011c": World(
        config="FrankaSkinPactPlaceV1011CMixedClutterConfig",
        sampler="PactPlaceCorridorV1011C33PctTallerPrimitiveSampler",
        env_version="pact_place_corridor_v10_11c_33pct_taller_primitives",
        cameras=(_EXO, "wrist_camera"),
        tasks=("pact_place_corridor_v10_11c_100",),
        target_rule=True,
    ),
    "v6": World(
        config="FrankaSkinPactPlaceV1010TwoObjectConfig",
        sampler="PactPlaceCorridorV1010TwoObjectSampler",
        env_version="pact_place_corridor_v10_10_two_object",
        cameras=(_EXO, "wrist_camera"),
        tasks=("pact_pick_n_place_v2_v6",),
    ),
    "v107_spaced": World(
        config="FrankaSkinPactPlaceV107SpacedBenchConfig",
        sampler="PactPlaceCorridorV107SpacedBenchSampler",
        env_version="pact_place_corridor_v10_7_spaced_bench",
        cameras=(_EXO, "wrist_camera"),
        tasks=("pact_pick_n_place_v2_v107_spaced", "pact_place_corridor_v107_spaced"),
    ),
    # Only for --target_support train reruns of T-1011d. The published T-1011d set and every
    # uniform-cup v1011d run stay on eval_act_v1011d.py.
    "v1011d": World(
        config="FrankaSkinPactPlaceV1011DRandomizedClutterConfig",
        sampler="PactPlaceCorridorV1011DRandomizedLayoutSampler",
        env_version="pact_place_corridor_v10_11d_randomized_clutter",
        cameras=(_EXO, "wrist_camera"),
        tasks=("pact_pick_n_place_v2_v1011d",),
        target_rule=True,
        clutter_scale=True,
    ),
    "v12": World(
        config="FrankaSkinPactPlaceV1011PreviewOneBottleConfig",
        sampler="PactPlaceCorridorV1011PreviewOneBottleSampler",
        env_version="pact_place_corridor_v10_11_preview_onebottle",
        cameras=(_EXO, "wrist_camera"),
        tasks=("pact_pick_n_place_v2_v12",),
        n_cells=8,
        xmls=frozenset({"pact_place_corridor_v10_11_center_preview.xml"}),
        cell_fn="v1011_preview_cell",
    ),
}


def _pop_flag(argv: list[str], name: str) -> str | None:
    """Remove ``name VALUE`` / ``name=VALUE`` from argv. The v1011d parser never sees it."""
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            value = argv[i + 1]
            del argv[i : i + 2]
            return value
        if arg.startswith(name + "="):
            del argv[i]
            return arg.split("=", 1)[1]
    return None


def _argv_list(argv: list[str], name: str) -> list[str] | None:
    if name not in argv:
        return None
    out = []
    for arg in argv[argv.index(name) + 1 :]:
        if arg.startswith("--"):
            break
        out.append(arg)
    return out or None


ENV = _pop_flag(sys.argv, "--env")
if ENV not in WORLDS:
    raise SystemExit(
        f"[eval_act_place] --env must be one of {sorted(WORLDS)} (got {ENV!r}). "
        "v1011d: eval_act_v1011d.py. hallway v5 / v5_ext: eval_act.py --task hallway."
    )
WORLD = WORLDS[ENV]
_POLICY_CAMERAS = tuple(_argv_list(sys.argv, "--cameras") or WORLD.cameras)
# Checkpoints converted with table_camera (batman publish renders) get the exo slot replaced.
_RENAME_EXO = _TABLE in _POLICY_CAMERAS
# The batman table_camera is the hybrid exo_camera_1 pose rendered at fovy 58. The hybrid
# exo_camera_1 itself renders at 45 (CameraManager's lookat branch drops `fov`), and the hub
# exo datasets were recorded through that same path, so only table_camera gets 58.
# 45 = the old rename used by eval_act_v107spaced.py for T-107.
_TABLE_FOV = float(_pop_flag(sys.argv, "--table_camera_fov") or 58.0)
if _TABLE_FOV not in (45.0, 58.0):
    raise SystemExit(f"[eval_act_place] --table_camera_fov must be 58 (train-matched) or 45 (T-107 rename)")
# uniform (default) = the sampler's own cup draw, same as eval_act_v1011d.py. train = redraw
# (construction retry) any cup outside the demos' support; see _target_in_train_support.
_TARGET_SUPPORT = _pop_flag(sys.argv, "--target_support") or "uniform"
if _TARGET_SUPPORT not in ("uniform", "train"):
    raise SystemExit(f"[eval_act_place] --target_support must be uniform or train (got {_TARGET_SUPPORT!r})")
if _TARGET_SUPPORT == "train" and not WORLD.target_rule:
    raise SystemExit(
        f"[eval_act_place] --target_support train needs a world with the V10.11 cup draw "
        f"({sorted(k for k, w in WORLDS.items() if w.target_rule)}); {ENV} cups already sit in the demo range"
    )

import eval_act_v1011d as base  # noqa: E402  (argv must be cleaned first)

_LOG = f"[eval_act_place {ENV}]"
_SCRIPT = Path(__file__).resolve()
_SCRIPT_SHA256 = hashlib.sha256(_SCRIPT.read_bytes()).hexdigest()


def _cell(house: int) -> tuple[str, str, str]:
    from molmo_spaces.data_generation.pact_place import contracts

    return getattr(contracts, WORLD.cell_fn)(house)


def _rename_exo_to_table(eval_cfg) -> None:
    """Same rename as eval_act_v107spaced._enable_table_camera."""
    cams = []
    renamed = 0
    for cam in list(eval_cfg.camera_config.cameras):
        if getattr(cam, "name", None) == _EXO:
            if hasattr(cam, "model_copy"):
                cam = cam.model_copy(update={"name": _TABLE})
            elif hasattr(cam, "copy"):
                cam = cam.copy(update={"name": _TABLE})
            else:
                cam.name = _TABLE
            renamed += 1
        cams.append(cam)
    eval_cfg.camera_config.cameras = cams
    if renamed != 1:
        raise SystemExit(f"{_LOG} expected to rename 1 {_EXO} -> {_TABLE}, got {renamed}")


# exo_camera_1 pose relative to robot_0/fr3_link0: offset [-1.05, -0.55, 1.30] looking at
# [0.55, 0.0, 0.45], up +z, as a wxyz quaternion (MuJoCo camera looks along its -z).
_TABLE_OFFSET = [-1.05, -0.55, 1.30]
_TABLE_LOOKAT = [0.55, 0.0, 0.45]
_TABLE_QUAT_WXYZ = [0.69280985, 0.42726385, -0.30493084, -0.49444645]


def _check_table_quaternion() -> None:
    import numpy as np

    w, x, y, z = _TABLE_QUAT_WXYZ
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    want = np.subtract(_TABLE_LOOKAT, _TABLE_OFFSET)
    want = want / np.linalg.norm(want)
    angle = np.degrees(np.arccos(np.clip(float(-rot[:, 2] @ want), -1.0, 1.0)))
    if angle > 0.05 or rot[2, 1] <= 0.0:
        raise SystemExit(f"{_LOG} table_camera quaternion is off the exo lookat by {angle:.3f} deg")


def _install_table_camera(eval_cfg) -> None:
    """Put the batman table_camera in the exo slot.

    fov 58: exo_camera_1 pose in quaternion mode so the fov reaches the renderer. Checked on 19
    batman frame-0 renders (v1010, v107_spaced 210-ep, v5 ext): RGB MAE ~2.9 at 58 vs ~29 at 45,
    fovy sweep minimum at 58.0 in every episode. fov 45: the old rename (T-107).
    """
    if _TABLE_FOV == 45.0:
        _rename_exo_to_table(eval_cfg)
        return
    from molmo_spaces.configs.camera_configs import RobotMountedCameraConfig

    _check_table_quaternion()
    table = RobotMountedCameraConfig(
        name=_TABLE,
        reference_body_names=["robot_0/fr3_link0"],
        camera_offset=list(_TABLE_OFFSET),
        camera_quaternion=list(_TABLE_QUAT_WXYZ),
        fov=_TABLE_FOV,
        record_depth=True,
        visibility_constraints={"__task_objects__": 0.001},
    )
    cams = []
    swapped = 0
    for cam in list(eval_cfg.camera_config.cameras):
        if getattr(cam, "name", None) == _EXO:
            cam = table
            swapped += 1
        cams.append(cam)
    eval_cfg.camera_config.cameras = cams
    if swapped != 1:
        raise SystemExit(f"{_LOG} expected to replace 1 {_EXO} with {_TABLE}, got {swapped}")


def _build_world_cfg(args, output_dir: Path):
    from molmo_spaces.configs.camera_configs import FrankaSkinHybridCameraSystem
    import molmo_spaces.data_generation.config.pact_place_datagen_configs as configs
    import molmo_spaces.tasks.pact_place as pact_place

    cfg_cls = getattr(configs, WORLD.config, None)
    expected = getattr(pact_place, WORLD.sampler, None)
    if cfg_cls is None or expected is None:
        raise SystemExit(
            f"{_LOG} {WORLD.config} / {WORLD.sampler} missing on {base._MOLMO_ROOT}. "
            "Need submodules/molmospaces at d57f350 / main or newer."
        )
    eval_cfg = cfg_cls()
    sampler_cls = eval_cfg.task_sampler_config.task_sampler_class
    if sampler_cls is not expected:
        raise SystemExit(
            f"{_LOG} sampler is {getattr(sampler_cls, '__name__', sampler_cls)}, "
            f"expected {WORLD.sampler}"
        )
    version = str(getattr(sampler_cls, "PACT_PLACE_ENVIRONMENT_VERSION", "") or "")
    if version != WORLD.env_version:
        raise SystemExit(
            f"{_LOG} sampler PACT_PLACE_ENVIRONMENT_VERSION={version!r} != {WORLD.env_version!r}"
        )
    eval_cfg.policy_config = base.ACTPolicyConfig()
    base._set_cfg(eval_cfg, "task_horizon", args.horizon or 1050)
    base._set_cfg(eval_cfg, "end_on_success", False)
    base._set_cfg(eval_cfg, "terminate_upon_success", False)
    base._set_cfg(eval_cfg, "output_dir", output_dir)
    base._set_cfg(eval_cfg, "num_workers", 1)
    base._set_cfg(eval_cfg, "save_videos", False)
    base._set_cfg(eval_cfg, "use_passive_viewer", False)
    base._set_cfg(eval_cfg, "viz_sensor_rgb", False)
    eval_cfg.camera_config = FrankaSkinHybridCameraSystem()
    if _RENAME_EXO:
        _install_table_camera(eval_cfg)
    base._disable_action_noise(eval_cfg)
    paths = [Path(p) for p in list(eval_cfg.task_sampler_config.scene_xml_paths or [])]
    names = {p.name for p in paths}
    if names != set(WORLD.xmls):
        raise SystemExit(f"{_LOG} scene XML set {sorted(names)} != {sorted(WORLD.xmls)}")
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"{_LOG} missing scene XML: {missing}")
    if len(paths) != WORLD.n_cells:
        raise SystemExit(f"{_LOG} expected {WORLD.n_cells} scene paths, got {len(paths)}")
    eval_cfg.task_sampler_config.house_inds = list(range(WORLD.n_cells))
    eval_cfg.task_sampler_config.samples_per_house = 1
    return eval_cfg, paths[0], sampler_cls, paths


def _no_clutter_xy_scale(scale: float) -> None:
    if abs(float(scale) - 1.0) > 1e-12:
        raise SystemExit(f"{_LOG} --clutter_xy_scale is v1011d only (got {scale})")
    print(f"{_LOG} clutter layout = {WORLD.sampler} default", flush=True)


def _assert_world_task(task, house: int) -> str:
    params = dict(getattr(task, "scene_params", None) or {})
    version = params.get("pact_place_environment_version")
    if version != WORLD.env_version:
        raise RuntimeError(
            f"{_LOG} scene_params pact_place_environment_version={version!r} "
            f"!= {WORLD.env_version!r}"
        )
    randomized = params.get("pact_v1011d_all_clutter_randomized") is True
    if ENV == "v1011d" and not randomized:
        raise RuntimeError(f"{_LOG} scene_params pact_v1011d_all_clutter_randomized is not True")
    if ENV != "v1011d" and randomized:
        raise RuntimeError(f"{_LOG} scene looks like v1011d randomized clutter; wrong sampler")
    if ENV == "v12":
        placed = params.get("pact_v1011_preview_placed_bodies")
        if not placed:
            raise RuntimeError(f"{_LOG} v12 standing kitchen not installed")
    return "|".join(_cell(house))


def _install_v12_kitchen(task) -> None:
    """PactPlaceCorridorV1011PreviewPolicy.reset, minus the expert.

    Raises ValueError when an extra lands in the motion lane (the expert raises
    HouseInvalidForTask there), so the construction retry redraws the scene.
    """
    from molmo_spaces.tasks import pact_place as pp

    params = getattr(task, "scene_params", None)
    if params is None:
        raise RuntimeError(f"{_LOG} v12 task has no scene_params")
    extra = list(params.get(pp.V1011_PREVIEW_EXTRA_BODIES_KEY, []))
    if not extra:
        raise RuntimeError(f"{_LOG} v12 sampler attached no standing-kitchen bodies")
    model = task.env.current_model
    data = task.env.current_data
    placed = pp._v1011_preview_install_preview_layout(model, data, extra, params)
    params[pp.V1011_PREVIEW_PLACED_BODIES_KEY] = list(placed)
    behind = tuple(
        str(item["uid"]) for item in pp.V1011_PREVIEW_STANDING_KITCHEN if item.get("behind_grasp")
    )
    intruders = [
        name
        for name in pp._v1011_preview_extras_overlap_motion_lane(model, data, placed)
        if not any(uid in name for uid in behind)
    ]
    if intruders:
        raise ValueError("extras_in_motion_lane " + ",".join(intruders[:3]))


# Cup support of the V10.11 demos (v1011c 99 rows, v1011d 200 rows): the cup is always on the
# side away from the panel (left panel hangs at +y, so left rows have y < 0; right rows y > 0),
# and |y| >= 0.05 m in 298/299. The sampler draws y uniform on [-0.375, 0.375] for either side;
# the expert failed the panel side, so those attempts never became demos. Replaying the
# 2026-09 v1011d n=50 scenes: 18 reproducible panel-side episodes, 0/18 success for ACT,
# PACT_RAW and PACT_READOUT; 11-12/26 on the demo side.
_TRAIN_TARGET_MIN_ABS_Y = 0.05


def _target_in_train_support(side: str, y: float) -> bool:
    away_from_panel = y < 0.0 if side == "left" else y > 0.0
    return bool(away_from_panel and abs(float(y)) >= _TRAIN_TARGET_MIN_ABS_Y)


def _install_target_support() -> None:
    """--target_support train: reject the cup draw outside the demo support.

    The ValueError reaches base._sample_v1011d_task, which redraws with the usual
    ``seed + k * 1000003`` stride. A draw that is in support is not touched, so those
    episodes are identical to --target_support uniform at the same seed.
    """
    import molmo_spaces.tasks.pact_place as pact_place

    cls = getattr(pact_place, WORLD.sampler)
    orig = cls._prepare_pact_clutter_layout

    def _prepare(self, th, _orig=orig):
        _orig(self, th)
        side = str(th.get("pact_intrusion_side") or "")
        y = float(self._v1011_target_rest[1])
        if not _target_in_train_support(side, y):
            raise ValueError(f"target outside train support: side={side} y={y:+.3f}")

    cls._prepare_pact_clutter_layout = _prepare


def _tag_target(task) -> None:
    """Remember where the cup starts, for episodes.jsonl. Reads state only."""
    model, data = task.env.current_model, task.env.current_data
    xy = None
    for body in range(int(model.nbody)):
        name = model.body(body).name or ""
        if name.startswith("cavity_obj_") and name.count("/") == 1:
            xy = [round(float(data.xpos[body][0]), 4), round(float(data.xpos[body][1]), 4)]
            break
    rec: dict = {"target_start_xy": xy}
    if WORLD.target_rule and xy is not None:
        side = str((getattr(task, "scene_params", None) or {}).get("pact_intrusion_side") or "")
        rec["target_in_train_support"] = int(_target_in_train_support(side, xy[1]))
    task._eval_target_start = rec


_BASE_RECORD = base._record_place_metric


def _record_place_metric(task, success, audit, *, extra=None, **kwargs):
    merged = dict(getattr(task, "_eval_target_start", None) or {})
    merged.update(extra or {})
    return _BASE_RECORD(task, success, audit, extra=merged, **kwargs)


_BASE_SAMPLE = base._sample_v1011d_task


def _sample_world_task(sampler, house: int, seed: int):
    """base._sample_v1011d_task, plus the v12 kitchen install inside the same retry."""
    if ENV != "v12":
        task, attempt = _BASE_SAMPLE(sampler, house, seed)
        _tag_target(task)
        return task, attempt
    last: BaseException | None = None
    for attempt in range(base._CONSTRUCTION_RETRY_MAX):
        draw_seed = int(seed) + attempt * base._CONSTRUCTION_RETRY_STRIDE
        base.set_seed(draw_seed)
        sampler.seed_task_sampling(draw_seed)
        try:
            task = sampler.sample_task(house_index=house)
            if task is None:
                raise ValueError("sample_task returned None")
            _install_v12_kitchen(task)
        except ValueError as exc:
            last = exc
            print(
                f"{_LOG} construction retry house={house} seed={seed} attempt={attempt} "
                f"draw_seed={draw_seed}: {exc}",
                flush=True,
            )
            continue
        _tag_target(task)
        return task, attempt
    raise RuntimeError(
        f"{_LOG} construction failed house={house} seed={seed} "
        f"after {base._CONSTRUCTION_RETRY_MAX} attempts: {last}"
    ) from last


def _house_schedule(args) -> str:
    if args.house_ind is None:
        return f"cycle_{WORLD.n_cells}"
    return f"pin_{int(args.house_ind)}"


_BASE_PROTOCOL = base._protocol_identity


def _protocol_identity(args, exec_horizon: int) -> dict:
    identity = {
        **_BASE_PROTOCOL(args, exec_horizon),
        "env": ENV,
        "environment_version": WORLD.env_version,
        "table_camera_fov": _TABLE_FOV if _RENAME_EXO else None,
    }
    # Only the non-default value is a key, so uniform dirs keep their old identity.
    if _TARGET_SUPPORT != "uniform":
        identity["target_support"] = _TARGET_SUPPORT
    return identity


_BASE_REFUSE = base._refuse_resume_mismatch


def _refuse_resume_mismatch(summary_path: Path, protocol: dict) -> None:
    """The base guard only sees keys of the new run; also refuse train -> uniform."""
    if summary_path.is_file():
        import json

        saved = json.loads(summary_path.read_text()).get("protocol") or {}
        old = saved.get("target_support", "uniform")
        if old != _TARGET_SUPPORT:
            raise SystemExit(
                f"{_LOG} refuse resume into {summary_path.parent}: protocol "
                f"target_support={old!r} != {_TARGET_SUPPORT!r}. Use a new --output_dir."
            )
    _BASE_REFUSE(summary_path, protocol)


_BASE_WRITE_SUMMARY = base._write_summary


def _target_support_block(payload: dict) -> dict | None:
    recs = list(((payload.get("collision") or {}).get("episodes_detail")) or [])
    tagged = [r for r in recs if "target_in_train_support" in r]
    if not tagged:
        return None
    block = {"rule": f"cup away from panel side and |y| >= {_TRAIN_TARGET_MIN_ABS_Y} m"}
    for key, flag in (("in_support", 1), ("out_of_support", 0)):
        group = [r for r in tagged if int(r["target_in_train_support"]) == flag]
        block[key] = {
            "episodes": len(group),
            "success": sum(int(r.get("success", 0)) for r in group),
            "ever_success": sum(int(r.get("ever_success", r.get("success", 0))) for r in group),
        }
    return block


def _write_summary(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload.update(
        {
            "task": ENV,
            "env": ENV,
            "config_class": WORLD.config,
            # "script" stays eval_act_v1011d.py: that file is the protocol code.
            "wrapper_script": str(_SCRIPT),
            "wrapper_script_sha256": _SCRIPT_SHA256,
            "target_support": _TARGET_SUPPORT,
            "target_train_support": _target_support_block(payload),
        }
    )
    _BASE_WRITE_SUMMARY(path, payload)


_BASE_WANDB_START = base.pact_eval_wandb.start


def _wandb_start(args, *, extra_config: dict | None = None, **kwargs):
    extra = dict(extra_config or {})
    extra.update({"task": ENV, "environment_version": WORLD.env_version, "config_class": WORLD.config})
    return _BASE_WANDB_START(args, extra_config=extra, **kwargs)


def _check_ckpt_matches_world() -> None:
    """Warn when the checkpoint's TASK folder or cameras do not match this world."""
    ckpt = base._argv_value("--ckpt_dir", "--checkpoint-dir")
    if not ckpt:
        return
    task = Path(ckpt).resolve().parent.name
    if task not in WORLD.tasks:
        print(
            f"{_LOG} WARNING: checkpoint task {task!r} is not trained on this world "
            f"{list(WORLD.tasks)}. Cross-env eval. New --output_dir.",
            flush=True,
        )
        return
    try:
        from constants import TASK_CONFIGS
    except ImportError:
        return
    trained = list((TASK_CONFIGS.get(task) or {}).get("camera_names") or [])
    if trained and trained != list(_POLICY_CAMERAS):
        print(
            f"{_LOG} WARNING: policy cameras {list(_POLICY_CAMERAS)} != TASK_CONFIGS"
            f"[{task!r}] {trained}. Ablation. New --output_dir.",
            flush=True,
        )


def _install_world() -> None:
    base._LOG = _LOG
    base._ENV_VERSION = WORLD.env_version
    base._EXPECTED_XML = WORLD.xmls
    base._N_CELLS = WORLD.n_cells
    base._DEFAULT_CAMERAS = WORLD.cameras
    base._EXO_CAM = _TABLE if _RENAME_EXO else _EXO
    base._build_v1011d_cfg = _build_world_cfg
    if not WORLD.clutter_scale:
        base._install_clutter_xy_scale = _no_clutter_xy_scale
    base._assert_v1011d_task = _assert_world_task
    base._sample_v1011d_task = _sample_world_task
    base._house_schedule = _house_schedule
    base._protocol_identity = _protocol_identity
    base._refuse_resume_mismatch = _refuse_resume_mismatch
    base._record_place_metric = _record_place_metric
    base._write_summary = _write_summary
    base.pact_eval_wandb.start = _wandb_start
    if _TARGET_SUPPORT == "train":
        _install_target_support()
    if ENV == "v12":
        from molmo_spaces.tasks import pact_place as pp

        # Score standing-kitchen contacts as clutter (the v12 expert does this in __init__).
        pp._v1011_preview_install_preview_contact_classes()


def main() -> None:
    _install_world()
    _check_ckpt_matches_world()
    print(
        f"{_LOG} world config={WORLD.config} sampler={WORLD.sampler} "
        f"version={WORLD.env_version} cells={WORLD.n_cells} "
        f"policy_cameras={list(_POLICY_CAMERAS)} "
        + (f"{_TABLE}=exo pose fovy {_TABLE_FOV:g} " if _RENAME_EXO else f"{_EXO} fovy 45 ")
        + (f"target_support={_TARGET_SUPPORT} " if WORLD.target_rule else "")
        + "(protocol: eval_act_v1011d.main; its banner below still names V1011D)",
        flush=True,
    )
    base.main()


if __name__ == "__main__":
    main()
