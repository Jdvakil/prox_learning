#!/usr/bin/env python3
"""Launch the contact-endpoint supervisor through the proven detached launcher."""

from __future__ import annotations

import launch_pact_frontend_screen_detached as implementation


implementation.SUPERVISOR = (
    implementation.ROOT / "scripts/run_pact_contact_supervisor.py"
)


if __name__ == "__main__":
    raise SystemExit(implementation.main())
