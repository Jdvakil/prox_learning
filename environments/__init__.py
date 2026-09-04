#!/usr/bin/env python3
"""Environments behind the published Hugging Face splits.

Each subpackage is named for the hub folder it reproduces, so
``environments/hf_v12`` is the environment for ``data/v12`` on
Ekshan267/pact_pick_n_place_v2, and declares a single ``SPEC``. Subpackages are
discovered at call time, so adding an environment means adding a directory —
see ``environments/README.md``.
"""

from __future__ import annotations

import importlib
import pkgutil
import runpy
import sys

from .spec import HF_DATASET, ROOT, EnvSpec

__all__ = ["EnvSpec", "HF_DATASET", "ROOT", "all_specs", "get", "run_entrypoint"]


def all_specs() -> dict[str, EnvSpec]:
    """Every environment in this package, keyed by hub split."""
    specs: dict[str, EnvSpec] = {}
    for module in pkgutil.iter_modules(__path__):
        if not module.ispkg:
            continue
        spec = getattr(importlib.import_module(f"{__name__}.{module.name}"), "SPEC", None)
        if isinstance(spec, EnvSpec):
            specs[spec.hub_split] = spec
    return dict(sorted(specs.items()))


def get(hub_split: str) -> EnvSpec:
    specs = all_specs()
    try:
        return specs[hub_split]
    except KeyError:
        raise KeyError(
            f"unknown hub split {hub_split!r}; known splits: {', '.join(specs)}"
        ) from None


def run_entrypoint(spec: EnvSpec, argv: list[str] | None = None) -> None:
    """Run the collect script for ``spec`` with the repo's import layout.

    The vendored scripts import their siblings by bare module name and expect
    the submodule on the path, the same way every other script in this repo is
    invoked.
    """
    for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    entrypoint = ROOT / spec.collect_entrypoint
    if not entrypoint.is_file():
        raise FileNotFoundError(f"missing collect entrypoint: {entrypoint}")

    sys.argv = [str(entrypoint), *(argv if argv is not None else sys.argv[1:])]
    runpy.run_path(str(entrypoint), run_name="__main__")
