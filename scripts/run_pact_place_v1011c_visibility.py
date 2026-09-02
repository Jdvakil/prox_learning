#!/usr/bin/env python3
"""Run paired V10.11b-to-V10.11c raw-skin visibility validation."""

from __future__ import annotations

import os

os.environ["PACT_PLACE_VISIBILITY_PARENT_CONTRACT"] = (
    "pact_place_v1011b_contract"
)
os.environ["PACT_PLACE_VISIBILITY_TALL_CONTRACT"] = "pact_place_v1011c_contract"

from run_pact_place_v1011b_visibility import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
