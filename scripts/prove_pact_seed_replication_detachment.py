#!/usr/bin/env python3
"""Prove the seed-replication smoke survives its launching shell."""

from __future__ import annotations

from pathlib import Path

import prove_pact_frontend_screen_detachment as implementation

implementation.LAUNCHER = (
    Path(__file__).resolve().parent / "launch_pact_seed_replication_detached.py"
)


if __name__ == "__main__":
    raise SystemExit(implementation.main())
