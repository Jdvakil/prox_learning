#!/usr/bin/env python3
"""Collect episodes for Hugging Face ``data/v1011d``.

    python environments/hf_v1011d/collect.py --smoke-only --smoke-attempts 2

Forwards to scripts/run_pact_place_v1011d_n200_collect.py, which is the script
that produced the published 200.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments import run_entrypoint  # noqa: E402
from environments.registry import HF_V1011D as SPEC  # noqa: E402

if __name__ == "__main__":
    run_entrypoint(SPEC)
