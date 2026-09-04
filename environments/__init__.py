#!/usr/bin/env python3
"""Environments behind the published Hugging Face splits.

Each subpackage is named for the hub folder it reproduces, so
``environments/hf_v12`` is the environment for ``data/v12`` on
Ekshan267/pact_pick_n_place_v2. The environment version strings themselves stay
as they were recorded at collect time; see :mod:`environments.registry`.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from .registry import HF_SPLITS, MOLMOSPACES_BRANCH, MOLMOSPACES_COMMIT, ROOT, EnvSpec, get

__all__ = [
    "EnvSpec",
    "HF_SPLITS",
    "MOLMOSPACES_BRANCH",
    "MOLMOSPACES_COMMIT",
    "ROOT",
    "get",
    "run_entrypoint",
]


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
