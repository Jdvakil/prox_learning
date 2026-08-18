#!/usr/bin/env python3
"""Build the deterministic Phase-0 PACT place-corridor preregistration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pact_place_corridor_contract import ROOT, build_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs" / "pact_place_corridor_v1.json",
    )
    args = parser.parse_args()
    document = build_contract()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(document["config_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
