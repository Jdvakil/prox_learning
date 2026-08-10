"""Data-generation configs for the cluttered, size-varying fume-hood tasks.

House indices are spaced 24 apart because that is what keeps every episode on
the same red-cup task; each of those indices is mapped to a different hood size
variant, so one collection run sweeps all 27 geometries without changing the
task.

This module lives in prox_learning (not molmospaces) on purpose: it only needs
molmo_spaces importable. Run with both repos on PYTHONPATH, e.g.

    PYTHONPATH=/path/to/prox_learning:$PYTHONPATH \
    python fumehood_env/collect_dense.py --config FrankaSkinClutteredFumehoodCheckConfig ...
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from molmo_spaces.configs import BasePolicyConfig
from molmo_spaces.configs.policy_configs import PickAndPlacePlannerPolicyConfig
from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
from molmo_spaces.configs.task_sampler_configs import PickTaskSamplerConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinHybridInvisObstacleConfig,
)
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

from fumehood_env.cluttered_fumehood import (
    ClutteredFumehoodPickAndPlaceSampler,
    ClutteredFumehoodPickSampler,
)

_SCENES = Path(__file__).resolve().parent / "custom_scenes"
_VARIANTS = sorted(_SCENES.glob("fumehood_v*.xml"))
if not _VARIANTS:
    raise ImportError(
        f"no fumehood_v*.xml under {_SCENES} — run fumehood_env/gen_fumehood_variants.py"
    )
_HOUSES = [1 + 24 * k for k in range(len(_VARIANTS))]

# scene_xml_paths is indexed by house id, so the list has to span the largest one.
_PATHS = [str(_VARIANTS[0])] * (max(_HOUSES) + 1)
for _k, _h in enumerate(_HOUSES):
    _PATHS[_h] = str(_VARIANTS[_k])


@register_config("FrankaSkinClutteredFumehoodConfig")
class FrankaSkinClutteredFumehoodConfig(FrankaSkinHybridInvisObstacleConfig):
    """Cluttered pick across all hood sizes. Single worker: the FrankaSkin
    configs seed one task sampler per worker from a fixed config seed, so
    parallel workers replay identical episodes."""

    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=ClutteredFumehoodPickSampler,
        scene_xml_paths=_PATHS,
        house_inds=_HOUSES,
        samples_per_house=5,
        added_pickup_objects=None,
        num_added_pickups=0,
        check_robot_placement_visibility=False,
        max_total_attempts_multiplier=10,
        max_allowed_sequential_task_sampler_failures=300,
        max_allowed_sequential_rollout_failures=300,
        max_allowed_sequential_irrecoverable_failures=10000,
        robot_object_z_offset_random_min=-np.random.uniform(0.0, 1.0),
        robot_object_z_offset_random_max=np.random.uniform(0.0, 1.0),
        robot_placement_rotation_range_rad=0.52,
        randomize_textures=True,
        randomize_lighting=False,
    )
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "datagen" / "cluttered_fumehood_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_cluttered_fumehood_v1"


@register_config("FrankaSkinClutteredFumehoodCheckConfig")
class FrankaSkinClutteredFumehoodCheckConfig(FrankaSkinClutteredFumehoodConfig):
    """Preflight: three hood sizes x 2 episodes."""

    task_sampler_config: PickTaskSamplerConfig = (
        FrankaSkinClutteredFumehoodConfig.model_fields["task_sampler_config"].default.model_copy(
            update={"house_inds": [_HOUSES[0], _HOUSES[len(_HOUSES) // 2], _HOUSES[-1]],
                    "samples_per_house": 2}))
    output_dir: Path = ASSETS_DIR / "datagen" / "cluttered_fumehood_check"

    @property
    def tag(self) -> str:
        return "franka_skin_cluttered_fumehood_check"


@register_config("FrankaSkinClutteredFumehoodPnPConfig")
class FrankaSkinClutteredFumehoodPnPConfig(FrankaSkinClutteredFumehoodConfig):
    """Fume-hood pick-AND-place: reach into the hood, grasp, and set the object
    down on the tray. Three changes over the pick config — the place-aware task
    config, the planner that knows how to place, and the sampler that returns a
    PickAndPlaceTask."""

    task_config: PickAndPlaceTaskConfig = PickAndPlaceTaskConfig(
        place_receptacle_name="place_tray")
    policy_config: BasePolicyConfig = PickAndPlacePlannerPolicyConfig()
    task_sampler_config: PickTaskSamplerConfig = (
        FrankaSkinClutteredFumehoodConfig.model_fields["task_sampler_config"].default.model_copy(
            update={"task_sampler_class": ClutteredFumehoodPickAndPlaceSampler}))
    output_dir: Path = ASSETS_DIR / "datagen" / "cluttered_fumehood_pnp_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_cluttered_fumehood_pnp_v1"


@register_config("FrankaSkinClutteredFumehoodPnPCheckConfig")
class FrankaSkinClutteredFumehoodPnPCheckConfig(FrankaSkinClutteredFumehoodPnPConfig):
    """Preflight: two hood sizes x 2 episodes."""

    task_sampler_config: PickTaskSamplerConfig = (
        FrankaSkinClutteredFumehoodPnPConfig.model_fields["task_sampler_config"].default.model_copy(
            update={"house_inds": [_HOUSES[0], _HOUSES[len(_HOUSES) // 2]],
                    "samples_per_house": 2}))
    output_dir: Path = ASSETS_DIR / "datagen" / "cluttered_fumehood_pnp_check"

    @property
    def tag(self) -> str:
        return "franka_skin_cluttered_fumehood_pnp_check"
