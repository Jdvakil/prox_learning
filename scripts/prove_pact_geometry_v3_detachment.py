#!/usr/bin/env python3
"""Kill the attempt-3 smoke launching shell and prove pool survival."""

from pathlib import Path

import prove_pact_frontend_screen_detachment as proof


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    proof.LAUNCHER = ROOT / "scripts/launch_pact_geometry_v3_detached.py"
    return proof.main()


if __name__ == "__main__":
    raise SystemExit(main())
