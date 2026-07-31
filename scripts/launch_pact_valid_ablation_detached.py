#!/usr/bin/env python3
"""Launch the valid-ablation supervisor under the proven detached wrapper."""

from __future__ import annotations

from pathlib import Path

import launch_pact_frontend_screen_detached as launcher


launcher.SUPERVISOR = (
    Path(__file__).resolve().parent
    / "run_pact_valid_ablation_supervisor.py"
)


if __name__ == "__main__":
    raise SystemExit(launcher.main())
