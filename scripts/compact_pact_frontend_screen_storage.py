#!/usr/bin/env python3
"""Run the frozen lossless compactor for the 120-row front-end screen."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import compact_pact_r2_storage as implementation


EXCLUDED_SCHEDULE_INDICES = {0, 119}


def _validate_wrapper_binding() -> None:
    try:
        index = sys.argv.index("--storage-amendment")
        amendment_path = Path(sys.argv[index + 1])
    except (ValueError, IndexError) as error:
        raise SystemExit("--storage-amendment is required") from error
    amendment = json.loads(amendment_path.read_text())
    observed = amendment.get("screen_compactor_wrapper_sha256")
    calculated = implementation.sha256_file(Path(__file__).resolve())
    if observed != calculated:
        raise SystemExit("screen compactor wrapper differs from amendment")


def main() -> int:
    _validate_wrapper_binding()
    implementation.EXCLUDED_SCHEDULE_INDICES = set(
        EXCLUDED_SCHEDULE_INDICES
    )
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
