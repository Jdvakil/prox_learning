#!/usr/bin/env python3
"""Launch the seed-replication supervisor with setsid/nohup."""

from __future__ import annotations

from pathlib import Path

import launch_pact_frontend_screen_detached as implementation

implementation.SUPERVISOR = (
    Path(__file__).resolve().parent / "run_pact_seed_replication_supervisor.py"
)


if __name__ == "__main__":
    raise SystemExit(implementation.main())
