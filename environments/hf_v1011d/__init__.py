#!/usr/bin/env python3
"""Environment for Hugging Face ``data/v1011d`` (200 accepted episodes).

This is ``pact_place_corridor_v10_11d_randomized_clutter`` driven by
``PactPlaceCorridorV1011DRandomizedLayoutSampler``, not the four-object sampler
named in the published folder README.
"""

from __future__ import annotations

from ..spec import EnvSpec

SPEC = EnvSpec(
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
    molmospaces_commit="70dedc07f34ed7f8335aed7f694ddef7ef823d3d",
    required_artifacts=(
        "diagnostics_output/pact_place_v9_v0b/palette_v9_1.json",
        "diagnostics_output/pact_place_v1011d_contract/contract.json",
        "diagnostics_output/pact_place_v1011d_preflight/preflight.json",
        "diagnostics_output/pact_place_v1011d_review/review_manifest.json",
    ),
    notes=(
        "200 accepted of 777 attempts. The published manifest shortens the "
        "environment to pact_place_corridor_v10_11d and its folder README "
        "credits PactPlaceCorridorV1010FourObjectSampler; both were written by "
        "hand at publish time and disagree with the collect code. Evaluating "
        "this checkpoint against the four-object sampler is a domain mismatch."
    ),
)

__all__ = ["SPEC"]
