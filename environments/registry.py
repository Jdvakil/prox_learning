#!/usr/bin/env python3
"""Map each published Hugging Face split to the environment that collected it.

The dataset is Ekshan267/pact_pick_n_place_v2. Its folder names (``v12``,
``v1011d``) are hub labels, not environment versions, and the two do not line
up: hub ``v12`` was collected by the V10.11 one-bottle preview environment, not
by ``pact_place_corridor_v12``. Everything below is read off the collect code
that produced each dump, so the values here are what a rollout must reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HF_DATASET = "Ekshan267/pact_pick_n_place_v2"

# molmospaces commit whose working tree byte-matches what ran on the collect
# host for both dumps. molmospaces main is a later refactor: it drops
# data_generation/runtime_compat.py and carries a trimmed corridor policy, so
# it cannot replay either dump.
MOLMOSPACES_COMMIT = "70dedc07f34ed7f8335aed7f694ddef7ef823d3d"
MOLMOSPACES_BRANCH = "experiment/pact-vs-act-remediation-v2"


@dataclass(frozen=True)
class EnvSpec:
    """One published split and the environment that produced it."""

    hub_split: str
    hub_path: str
    n_episodes: int
    environment_version: str
    sampler_class: str
    policy_class: str
    schema_version: str
    scene_relative: tuple[str, ...]
    collect_entrypoint: str
    contract_module: str
    notes: str = ""

    @property
    def scene_paths(self) -> tuple[Path, ...]:
        return tuple(ROOT / rel for rel in self.scene_relative)


HF_V12 = EnvSpec(
    hub_split="v12",
    hub_path="data/v12",
    n_episodes=165,
    environment_version="pact_place_corridor_v10_11_preview_onebottle",
    sampler_class="PactPlaceCorridorV1010FourObjectSampler",
    policy_class="PactPlaceCorridorPolicy",
    schema_version="pact_pick_n_place_v2_v12",
    scene_relative=("custom_scenes/pact_place_corridor_v10_11_center_preview.xml",),
    collect_entrypoint="scripts/run_pact_place_v1011_preview_collect.py",
    contract_module="pact_place_v1010_contract",
    notes=(
        "Four-object household plus standing kitchen extras attached by the "
        "preview renderer. Not pact_place_corridor_v12, which has no published "
        "dataset. The scene is vendored under custom_scenes/ rather than read "
        "from the submodule; the file bytes, and so its sha256, are unchanged."
    ),
)

HF_V1011D = EnvSpec(
    hub_split="v1011d",
    hub_path="data/v1011d",
    n_episodes=200,
    environment_version="pact_place_corridor_v10_11d_randomized_clutter",
    sampler_class="PactPlaceCorridorV1011DRandomizedLayoutSampler",
    policy_class="PactPlaceCorridorPolicy",
    schema_version="pact_place_v1011d_accepted_v1",
    scene_relative=(
        "custom_scenes/pact_place_corridor_v10_7_neg5.xml",
        "custom_scenes/pact_place_corridor_v10_7_center.xml",
        "custom_scenes/pact_place_corridor_v10_7_pos5.xml",
    ),
    collect_entrypoint="scripts/run_pact_place_v1011d_n200_collect.py",
    contract_module="pact_place_v1011d_contract",
    notes=(
        "200 accepted of 777 attempts. The published manifest shortens the "
        "environment to pact_place_corridor_v10_11d and its folder README "
        "credits PactPlaceCorridorV1010FourObjectSampler; both were written by "
        "hand at publish time and disagree with the collect code. Evaluating "
        "this checkpoint against the four-object sampler is a domain mismatch."
    ),
)

HF_SPLITS: dict[str, EnvSpec] = {spec.hub_split: spec for spec in (HF_V12, HF_V1011D)}


def get(hub_split: str) -> EnvSpec:
    try:
        return HF_SPLITS[hub_split]
    except KeyError:
        known = ", ".join(sorted(HF_SPLITS))
        raise KeyError(f"unknown hub split {hub_split!r}; known splits: {known}") from None
