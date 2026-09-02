#!/usr/bin/env python3
"""Generate the frozen V10.11b six-video owner-review packet."""

from __future__ import annotations

import os

os.environ["PACT_PLACE_V1011_CONTRACT_MODULE"] = "pact_place_v1011b_contract"
os.environ["PACT_PLACE_V1011_REVIEW_ENTRYPOINT"] = (
    "scripts/run_pact_place_v1011b_review.py"
)

from run_pact_place_v1011_review import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
