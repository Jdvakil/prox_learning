#!/usr/bin/env python3
"""Run the frozen V10.11d 96-layout preflight."""

from __future__ import annotations

import os

os.environ["PACT_PLACE_V1011_CONTRACT_MODULE"] = "pact_place_v1011d_contract"

from run_pact_place_v1011_preflight import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
