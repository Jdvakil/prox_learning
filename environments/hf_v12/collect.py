#!/usr/bin/env python3
"""Collect episodes for Hugging Face ``data/v12``.

    python environments/hf_v12/collect.py --target 2 --max-attempts 4

Forwards to scripts/run_pact_place_v1011_preview_collect.py, which is the
script that produced the published 165.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments import run_entrypoint  # noqa: E402
from environments.registry import HF_V12 as SPEC  # noqa: E402

if __name__ == "__main__":
    run_entrypoint(SPEC)
