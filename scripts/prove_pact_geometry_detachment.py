#!/usr/bin/env python3
"""Kill the smoke launching shell and prove the geometry pool survives."""

from __future__ import annotations

from pathlib import Path

import prove_pact_frontend_screen_detachment as proof


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    # The proof protocol is intentionally shared: launch through a shell that
    # remains alive, observe one evaluator, SIGKILL that shell, require an
    # advancing heartbeat, and wait for the immutable smoke row to complete.
    proof.LAUNCHER = ROOT / "scripts/launch_pact_geometry_detached.py"
    return proof.main()


if __name__ == "__main__":
    raise SystemExit(main())
