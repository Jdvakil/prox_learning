#!/usr/bin/env python3
"""Prove the valid-ablation smoke survives its launching shell."""

from __future__ import annotations

from pathlib import Path

import prove_pact_frontend_screen_detachment as proof


proof.LAUNCHER = (
    Path(__file__).resolve().parent
    / "launch_pact_valid_ablation_detached.py"
)


if __name__ == "__main__":
    raise SystemExit(proof.main())
