#!/usr/bin/env python3
"""Losslessly compact seed-replication rows except smoke and final row."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import compact_pact_r2_storage as implementation

EXCLUDED_SCHEDULE_INDICES = {0, 119}


def validate_wrapper() -> None:
    try:
        index = sys.argv.index("--storage-amendment")
        amendment_path = Path(sys.argv[index + 1])
    except (ValueError, IndexError) as error:
        raise SystemExit("--storage-amendment is required") from error
    amendment = json.loads(amendment_path.read_text())
    expected = amendment.get("seed_replication_compactor_wrapper_sha256")
    if expected != implementation.sha256_file(Path(__file__).resolve()):
        raise SystemExit("seed-replication compactor differs from amendment")


def main() -> int:
    validate_wrapper()
    implementation.EXCLUDED_SCHEDULE_INDICES = set(EXCLUDED_SCHEDULE_INDICES)
    output_index = sys.argv.index("--output-root")
    output_root = Path(sys.argv[output_index + 1]).resolve()
    compatibility = output_root / "execution_summary.json"
    if not compatibility.exists() and not compatibility.is_symlink():
        compatibility.symlink_to("full_execution_summary.json")
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
