"""Eval task definitions.

Four tasks, all built on MolmoSpaces FrankaPickandPlace (procthor-objaverse,
randomized cameras):

  pnp           — open workspace, no nearby obstacles (small expected delta)
  near_contact  — fixed obstacle 5-8 cm from expert path (PRIMARY)
  pnp_color     — language-specified object among colored distractors
  pnp_next_to   — spatial relation: place next to a reference

The primary scientific claim (PROJECT.md §8) is tested on ``near_contact``.

Each ``TaskSpec`` carries everything ``run_eval.py`` needs:
    env_module/env_class — Python entry point for the simulator
    env_kwargs            — kwargs passed at construction time
    language              — the instruction string to feed Molmo2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    obstacle: bool
    language_relation: bool
    language: str
    env_module: str = "molmo_spaces.benchmarks.franka_pickandplace"
    env_class: str = "FrankaPickandPlaceEnv"
    env_kwargs: dict[str, Any] = field(default_factory=dict)


REGISTRY: dict[str, TaskSpec] = {
    "pnp": TaskSpec(
        name="pnp",
        description="Open workspace pick-and-place. Baseline competence test.",
        obstacle=False,
        language_relation=False,
        language="pick up the object and place it on the target",
        env_kwargs={"variant": "open_workspace"},
    ),
    "near_contact": TaskSpec(
        name="near_contact",
        description=(
            "Pick-and-place with a fixed obstacle 5-8 cm from the expert "
            "arm path. Expected to show the largest proximity advantage."
        ),
        obstacle=True,
        language_relation=False,
        language="pick up the object and place it on the target",
        env_kwargs={"variant": "near_contact", "obstacle_distance_m": 0.06},
    ),
    "pnp_color": TaskSpec(
        name="pnp_color",
        description="Language-specified object among colored distractors.",
        obstacle=False,
        language_relation=True,
        language="pick up the {color} object and place it on the target",
        env_kwargs={"variant": "color_distractors"},
    ),
    "pnp_next_to": TaskSpec(
        name="pnp_next_to",
        description=(
            "Place object next to a reference object. Most challenging "
            "language task."
        ),
        obstacle=False,
        language_relation=True,
        language="place the object next to the {reference}",
        env_kwargs={"variant": "place_next_to"},
    ),
}


def get(name: str) -> TaskSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown task {name!r}; known: {list(REGISTRY)}")
    return REGISTRY[name]
