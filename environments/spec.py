#!/usr/bin/env python3
"""The shape of an environment declaration.

One :class:`EnvSpec` describes a published dataset split and everything a clean
checkout needs to reproduce it. Each ``environments/<name>/__init__.py`` builds
exactly one and assigns it to ``SPEC``; nothing else has to be edited to add an
environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HF_DATASET = "Ekshan267/pact_pick_n_place_v2"


@dataclass(frozen=True)
class EnvSpec:
    """One published split and the environment that produced it.

    ``environment_version``, ``sampler_class`` and ``schema_version`` are read
    off the collect code that produced the dump, not off the hub metadata. Where
    the two disagree the code wins, and the disagreement belongs in ``notes``.
    """

    # Hub identity.
    hub_split: str
    hub_path: str
    n_episodes: int

    # Recorded at collect time. Changing any of these invalidates the dump.
    environment_version: str
    sampler_class: str
    policy_class: str
    schema_version: str

    # What the checkout must provide.
    scene_relative: tuple[str, ...]
    collect_entrypoint: str
    contract_module: str
    molmospaces_commit: str

    # Gate artifacts the collect binds before it will roll out. Missing ones
    # surface as a bare FileNotFoundError deep inside preflight, so they are
    # listed here and checked up front.
    required_artifacts: tuple[str, ...] = field(default=())

    notes: str = ""

    @property
    def scene_paths(self) -> tuple[Path, ...]:
        return tuple(ROOT / rel for rel in self.scene_relative)

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        return tuple(ROOT / rel for rel in self.required_artifacts)
