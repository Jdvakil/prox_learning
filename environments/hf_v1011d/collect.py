#!/usr/bin/env python3
"""Collect episodes for this environment's hub split.

    python environments/hf_v1011d/collect.py --smoke-only --smoke-attempts 2

Arguments are forwarded verbatim to the collect script named by the package
SPEC. This file is identical in every environment package; copy it as is.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from environments import run_entrypoint  # noqa: E402

SPEC = importlib.import_module(f"environments.{HERE.name}").SPEC

if __name__ == "__main__":
    run_entrypoint(SPEC)
