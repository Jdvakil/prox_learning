#!/usr/bin/env python3
"""Environment for Hugging Face ``data/v12`` (165 accepted episodes).

Despite the hub label, this is ``pact_place_corridor_v10_11_preview_onebottle``
and has nothing to do with ``pact_place_corridor_v12``.
"""

from __future__ import annotations

from ..spec import EnvSpec

SPEC = EnvSpec(
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
    molmospaces_commit="70dedc07f34ed7f8335aed7f694ddef7ef823d3d",
    required_artifacts=("diagnostics_output/pact_place_v9_v0b/palette_v9_1.json",),
    notes=(
        "Four-object household plus standing kitchen extras attached by the "
        "preview renderer. Not pact_place_corridor_v12, which has no published "
        "dataset. Collected against molmospaces ed045d7; the pinned commit is a "
        "descendant whose planner and four-object sampler are byte-identical. "
        "The scene is vendored under custom_scenes/ rather than read from the "
        "submodule, where it was never committed; the bytes, and so the "
        "recorded sha256, are unchanged."
    ),
)

__all__ = ["SPEC"]
